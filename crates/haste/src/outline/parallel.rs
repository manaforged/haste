use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};

use anyhow::{Result, anyhow};

use super::build::{Loader, Others};
use super::tree::{Crate, Files, View};
use super::{Outline, dependents_of};

/// What every worker reads and never changes: the file cache, each crate's dependencies, and the crates whose exports each crate may read.
struct Shared<'a> {
    cache: &'a Files,
    deps: &'a [Vec<(String, usize)>],
    reads: Vec<Vec<usize>>,
}

/// The state of every crate while their levels load on several threads.
struct Board {
    /// Each crate, out while a worker loads one of its levels.
    crates: Vec<Option<Crate>>,
    /// What each crate showed before its first level and after each level it finished.
    views: Vec<Vec<Arc<View>>>,
    /// Whether each crate has no declared module left.
    done: Vec<bool>,
    running: usize,
    /// The first failure by level and crate.
    failed: Option<((usize, usize), anyhow::Error)>,
}

/// One level of one crate to load: the crate, taken out of the board, and by crate the views of the crates it reads.
type Task = (usize, Crate, Vec<Option<Arc<View>>>);

/// The lock over the board, and the signal that a task finished.
type Lock = (Mutex<Board>, Condvar);

/// Loads every module of every crate of `outline` on `workers` threads to the outline that loading level by level on one thread gives.
pub(super) fn load_all(outline: &mut Outline, workers: usize) -> Result<()> {
    let board = Board::new(std::mem::take(&mut outline.crates));
    let shared = Shared { cache: &outline.cache, deps: &outline.deps, reads: reads(&outline.deps) };
    let lock: Lock = (Mutex::new(board), Condvar::new());
    std::thread::scope(|scope| {
        for _ in 1..workers {
            scope.spawn(|| work(&lock, &shared));
        }
        work(&lock, &shared);
    });
    let board = lock.0.into_inner().map_err(|_| anyhow!("crate loading poisoned"))?;
    outline.crates = board.crates.into_iter().map(Option::unwrap_or_default).collect();
    board.failed.map_or(Ok(()), |(_, error)| Err(error))
}

/// The crates that each crate's loader may read the exports of: every crate its dependencies reach.
fn reads(deps: &[Vec<(String, usize)>]) -> Vec<Vec<usize>> {
    let edges: Vec<Vec<usize>> = deps.iter().map(|each| each.iter().map(|(_, dep)| *dep).collect()).collect();
    let reach = |krate: usize| -> Vec<usize> { dependents_of(&edges, &mut vec![false; deps.len()], [krate]).into_iter().filter(|&other| other != krate).collect() };
    (0..deps.len()).map(reach).collect()
}

impl Board {
    fn new(crates: Vec<Crate>) -> Board {
        let views = crates.iter().map(|tree| vec![Arc::new(tree.view())]).collect();
        let done = crates.iter().map(|tree| !tree.partial()).collect();
        Board { crates: crates.into_iter().map(Some).collect(), views, done, running: 0, failed: None }
    }

    /// The view of crate `other` that crate `krate` reads at its next level, as the serial order shows it: after the same level for a lower crate, after the level before for a higher one; none until `other` gets there.
    fn shown(&self, krate: usize, other: usize) -> Option<&Arc<View>> {
        let views = &self.views[other];
        let at = self.views[krate].len() - 1 + usize::from(other < krate);
        views.get(at).or_else(|| views.last().filter(|_| self.done[other]))
    }

    fn ready(&self, krate: usize, reads: &[usize]) -> bool {
        !self.done[krate] && self.crates[krate].is_some() && reads.iter().all(|&other| self.shown(krate, other).is_some())
    }

    /// Takes out the ready crate at the lowest level, the lowest crate first, with the views it reads.
    fn start(&mut self, shared: &Shared) -> Option<Task> {
        let ready = (0..self.crates.len()).filter(|&krate| self.ready(krate, &shared.reads[krate]));
        let krate = ready.min_by_key(|&krate| (self.views[krate].len(), krate))?;
        let mut seen = vec![None; self.crates.len()];
        for &other in &shared.reads[krate] {
            seen[other] = self.shown(krate, other).cloned();
        }
        let holder = self.crates[krate].take()?;
        self.running += 1;
        Some((krate, holder, seen))
    }

    /// Puts crate `krate` back after one level, and shows what it exports now, or keeps its failure when it is the first by level and crate.
    fn finish(&mut self, krate: usize, holder: Crate, result: Result<()>) {
        self.running -= 1;
        match result {
            Ok(()) => {
                self.views[krate].push(Arc::new(holder.view()));
                self.done[krate] = !holder.partial();
            }
            Err(error) => {
                let at = (self.views[krate].len() - 1, krate);
                if self.failed.as_ref().is_none_or(|(first, _)| at < *first) {
                    self.failed = Some((at, error));
                }
            }
        }
        self.crates[krate] = Some(holder);
    }

    fn over(&self) -> bool {
        self.failed.is_some() || self.done.iter().all(|done| *done)
    }
}

/// Loads levels of crates as they become ready until every crate is read or one failed.
fn work(lock: &Lock, shared: &Shared) {
    while let Some((krate, mut holder, seen)) = next(lock, shared) {
        let result = level(shared, krate, &mut holder, &seen);
        if let Ok(mut board) = lock.0.lock() {
            board.finish(krate, holder, result);
        }
        lock.1.notify_all();
    }
}

/// The next ready task, waiting while running ones may make one ready; none once loading is over.
fn next(lock: &Lock, shared: &Shared) -> Option<Task> {
    let mut board = lock.0.lock().ok()?;
    loop {
        if board.over() {
            return None;
        }
        if let Some(task) = board.start(shared) {
            return Some(task);
        }
        if board.running == 0 {
            board.failed = Some(((usize::MAX, usize::MAX), anyhow!("crate loading stalled with modules left to read")));
            lock.1.notify_all();
            return None;
        }
        board = lock.1.wait(board).ok()?;
    }
}

/// Reads the modules of crate `krate` that are declared now, in order, seeing the other crates through `seen`.
fn level(shared: &Shared, krate: usize, holder: &mut Crate, seen: &[Option<Arc<View>>]) -> Result<()> {
    let modules: Vec<usize> = holder.declared().collect();
    let paths: Vec<PathBuf> = modules.iter().flat_map(|&module| holder.unread_files(module)).cloned().collect();
    shared.cache.read_all(&paths);
    let deps = &shared.deps[krate];
    for module in modules {
        Loader::new(krate, shared.cache, Others::Views(seen), deps, holder).load_module(module)?;
    }
    Ok(())
}

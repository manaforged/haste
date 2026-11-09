//! The `case_` functions of a workspace built into one image and run, from cold or again after an edit in a process that kept the build.

use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime};

use anyhow::Result;
use haste_front::graph::{self, Targets};
use haste_front::image::Image;
use haste_front::names::{ItemRef, Names};
use haste_front::outline::{self, edit::Edit};
use haste_front::sema::reach::{Instance, Reached, reach, reach_again, reach_anew};
use haste_front::sema::{self, Edited, Store};
use rustc_hash::FxHashMap as HashMap;

use super::{Failures, answer, case_items, eager_front, error_kind, is_member, print_failures};

const INPUTS: [i64; 5] = [-5, 0, 1, 3, 20];

/// A built workspace: its outline, names and inference store, the reached functions, and their image.
pub struct Build {
    outline: outline::Outline,
    names: Names,
    store: Option<Store>,
    labels: Vec<String>,
    roots: Vec<usize>,
    reached: Reached,
    image: Image,
    broken: HashMap<usize, String>,
    stamps: Vec<(PathBuf, Option<(SystemTime, u64)>)>,
}

fn stamp(path: &std::path::Path) -> Option<(SystemTime, u64)> {
    let meta = std::fs::metadata(path).ok()?;
    Some((meta.modified().ok()?, meta.len()))
}

fn stamps(workspace: &graph::Workspace, outline: &outline::Outline) -> Vec<(PathBuf, Option<(SystemTime, u64)>)> {
    member_files(workspace, outline).into_iter().map(|path| (path.clone(), stamp(&path))).collect()
}

fn case_roots(workspace: &graph::Workspace, outline: &outline::Outline) -> (Vec<ItemRef>, Vec<Instance>) {
    let cases = case_items(workspace, outline);
    let roots = cases.iter().map(|item| (*item, Vec::new())).collect();
    (cases, roots)
}

fn recompile(image: &mut Image, broken: &mut HashMap<usize, String>, (cx, reached): (&mut sema::Context<'_>, &Reached), fresh: &[usize]) {
    fresh.iter().for_each(|at| _ = broken.remove(at));
    broken.extend(image.compile(cx, reached, fresh));
}

impl Build {
    /// Reads, infers and compiles every case of `workspace` from nothing.
    pub fn cold(workspace: &graph::Workspace, targets: &Targets) -> Result<Build> {
        let (mut outline, mut names) = eager_front(workspace, targets)?;
        let (cases, roots) = case_roots(workspace, &outline);
        let (mut store, mut last) = (Store::default(), false);
        loop {
            let mut cx = sema::Context::new(&outline, &names, store);
            let reached = reach(&mut cx, &roots);
            if last || cx.settled() {
                let mut image = Image::new(&reached)?;
                let (mut broken, all) = (HashMap::default(), (0..reached.list.len()).collect::<Vec<_>>());
                recompile(&mut image, &mut broken, (&mut cx, &reached), &all);
                let labels = cases.iter().map(|item| cx.name(*item).to_owned()).collect();
                let store = Some(cx.release().0);
                let (roots, stamps) = (roots.iter().map(|root| reached.index[root]).collect(), stamps(workspace, &outline));
                return Ok(Build { outline, names, store, labels, roots, reached, image, broken, stamps });
            }
            let released = cx.release();
            let more;
            (store, more) = answer(&mut names, &mut outline, released)?;
            last = !more;
        }
    }

    /// Brings the build up to date with the member files: infers and compiles again the functions of files whose bodies changed, reads again a crate whose items changed, and builds from nothing when that fails.
    pub fn refresh(&mut self, workspace: &graph::Workspace, targets: &Targets) -> Result<()> {
        let (mut stale, mut crates) = (Vec::new(), Vec::new());
        for at in 0..self.stamps.len() {
            let now = stamp(&self.stamps[at].0);
            if now == self.stamps[at].1 {
                continue;
            }
            self.stamps[at].1 = now;
            match self.outline.reread(&self.stamps[at].0) {
                Ok(Edit::Same) => {}
                Ok(Edit::Bodies(slots)) => stale.extend(slots),
                Ok(Edit::Items(edited)) => crates.extend(edited),
                Err(_) => return self.rebuild(workspace, targets),
            }
        }
        let done = match (stale.is_empty(), crates.is_empty()) {
            (true, true) => return Ok(()),
            (_, false) => self.reload(workspace, crates.into_iter().chain(stale.iter().map(|&(krate, _)| krate)).collect()),
            (false, true) => self.again(&stale),
        };
        if !done {
            return self.rebuild(workspace, targets);
        }
        Ok(())
    }

    fn again(&mut self, stale: &[(u32, u32)]) -> bool {
        let Some(mut store) = self.store.take() else { return false };
        let touched = store.touched(self.reached.list.iter().map(|(instance, _)| instance), Edited::Files(stale));
        store.forget(&self.outline, Edited::Files(stale));
        let mut cx = sema::Context::new(&self.outline, &self.names, store);
        let fresh = reach_again(&mut cx, &mut self.reached, &touched);
        if !cx.settled() {
            return false;
        }
        recompile(&mut self.image, &mut self.broken, (&mut cx, &self.reached), &fresh);
        self.store = Some(cx.release().0);
        true
    }

    fn reload(&mut self, workspace: &graph::Workspace, mut crates: Vec<u32>) -> bool {
        crates.sort_unstable();
        crates.dedup();
        let Some(mut store) = self.store.take() else { return false };
        let touched = store.touched(self.reached.list.iter().map(|(instance, _)| instance), Edited::Crates(&crates));
        for &krate in &crates {
            if self.outline.reload(krate as usize).is_err() {
                return false;
            }
            self.names.reset(krate as usize);
        }
        if self.names.settle(&mut self.outline).is_err() {
            return false;
        }
        store.forget(&self.outline, Edited::Crates(&crates));
        let (cases, roots) = case_roots(workspace, &self.outline);
        let mut cx = sema::Context::new(&self.outline, &self.names, store);
        let fresh = reach_anew(&mut cx, &mut self.reached, &touched, &roots);
        if !cx.settled() {
            return false;
        }
        recompile(&mut self.image, &mut self.broken, (&mut cx, &self.reached), &fresh);
        self.labels = cases.iter().map(|item| cx.name(*item).to_owned()).collect();
        self.roots = roots.iter().map(|root| self.reached.index[root]).collect();
        self.store = Some(cx.release().0);
        self.stamps = stamps(workspace, &self.outline);
        true
    }

    fn rebuild(&mut self, workspace: &graph::Workspace, targets: &Targets) -> Result<()> {
        *self = Build::cold(workspace, targets)?;
        Ok(())
    }

    /// Runs every case whose functions all compiled and prints what failed and how long building took.
    pub fn report(&self, started: Instant, built: Duration) -> Result<()> {
        let mut failures = Failures::new();
        let mut runnable = Vec::new();
        for (&root, name) in self.roots.iter().zip(&self.labels) {
            let closure = self.reached.closure(root);
            let failure = closure.iter().rev().find_map(|at| self.reached.list[*at].1.as_ref().err()).or_else(|| closure.iter().rev().find_map(|at| self.broken.get(at)));
            match failure {
                Some(error) => failures.entry(error_kind(error)).or_default().push(format!("{name}: {error}")),
                None => runnable.push((root, closure, name.as_str())),
            }
        }
        let statuses = self.image.run_cases(&runnable, &INPUTS)?;
        for ((_, _, name), status) in runnable.iter().zip(statuses).filter(|(_, status)| *status != 0) {
            failures.entry(format!("crashed with status {status}")).or_default().push((*name).to_owned());
        }
        print_failures(&failures);
        let ms = |at: Duration| at.as_secs_f64() * 1e3;
        eprintln!("cases {} found, {} ran, {} functions reached, built in {:.1} ms, total {:.1} ms", self.labels.len(), runnable.len(), self.reached.list.len(), ms(built), ms(started.elapsed()));
        Ok(())
    }
}

fn member_files(workspace: &graph::Workspace, outline: &outline::Outline) -> Vec<PathBuf> {
    let members = outline.roots.iter().zip(&outline.crates).filter(|(root, _)| is_member(workspace, root));
    let mut paths: Vec<PathBuf> = members.flat_map(|(_, tree)| tree.files.iter().filter(|file| !file.expanded).map(|file| file.source.path.clone())).collect();
    paths.sort();
    paths.dedup();
    paths
}

//! The `case_` functions of a workspace built into one image and run, from cold or again after an edit in a process that kept the build.

use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Result, bail};
use crate::graph::{self, Targets};
use crate::image::Image;
use crate::names::{ItemRef, Names};
use crate::outline::{self, edit::Edit};
use crate::sema::reach::{Instance, Reached, main_roots, reach_again, reach_anew};
use crate::sema::{self, Edited, Cache};
use rustc_hash::FxHashMap as HashMap;

use crate::cli::{Failures, case_items, error_kind, is_member, print_failures};

const INPUTS: [i64; 5] = [-5, 0, 1, 3, 20];

/// A built workspace: its outline, names and inference store, the reached functions, and their image.
pub struct Build {
    outline: outline::Outline,
    names: Names,
    store: Option<Cache>,
    labels: Vec<String>,
    roots: Vec<usize>,
    entry: Option<[usize; 2]>,
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

/// The `main` of the first member binary.
fn main_item(workspace: &graph::Workspace, outline: &outline::Outline) -> Option<ItemRef> {
    let krate = outline.roots.iter().position(|root| root.role == outline::Role::Bin && is_member(workspace, root))?;
    outline.main(krate)
}

fn with_entry(roots: &[Instance], entry: &Result<Option<[Instance; 2]>>) -> Vec<Instance> {
    let mut all = roots.to_vec();
    if let Ok(Some(pair)) = entry {
        all.extend(pair.iter().cloned());
    }
    all
}

fn entry_at(reached: &Reached, entry: Result<Option<[Instance; 2]>>) -> Result<Option<[usize; 2]>> {
    Ok(entry?.map(|[start, main]| [reached.index[&start], reached.index[&main]]))
}

fn recompile(image: &mut Image, broken: &mut HashMap<usize, String>, (cx, reached): (&mut sema::Context<'_>, &Reached), fresh: &[usize]) {
    fresh.iter().for_each(|at| _ = broken.remove(at));
    broken.extend(image.compile(cx, reached, fresh));
}

impl Build {
    /// Reads, infers and compiles every case of `workspace` from nothing.
    pub fn cold(workspace: &graph::Workspace, targets: &Targets) -> Result<Build> {
        let (mut outline, mut names) = crate::macros::load_eager(workspace, targets, &[])?;
        let (cases, roots) = case_roots(workspace, &outline);
        let main = main_item(workspace, &outline);
        let entry_of = |cx: &mut sema::Context<'_>| main.map(|item| main_roots(cx, item)).transpose();
        let built = crate::macros::reach_resolved(&mut outline, &mut names, |cx| with_entry(&roots, &entry_of(cx)), |mut cx, reached| {
            let entry = entry_at(&reached, entry_of(&mut cx))?;
            let (image, broken) = Image::build(&mut cx, &reached)?;
            let labels = cases.iter().map(|item| cx.name(*item).to_owned()).collect();
            let roots = roots.iter().map(|root| reached.index[root]).collect();
            Ok((Some(cx.release().0), labels, roots, entry, reached, image, broken))
        })?;
        let (store, labels, roots, entry, reached, image, broken) = built;
        let stamps = stamps(workspace, &outline);
        Ok(Build { outline, names, store, labels, roots, entry, reached, image, broken, stamps })
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
        if !cx.resolved() {
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
        if crate::macros::resolve(&mut self.outline, &mut self.names).is_err() {
            return false;
        }
        store.forget(&self.outline, Edited::Crates(&crates));
        let (cases, roots) = case_roots(workspace, &self.outline);
        let main = main_item(workspace, &self.outline);
        let mut cx = sema::Context::new(&self.outline, &self.names, store);
        let entry = main.map(|item| main_roots(&mut cx, item)).transpose();
        let fresh = reach_anew(&mut cx, &mut self.reached, &touched, &with_entry(&roots, &entry));
        if !cx.resolved() {
            return false;
        }
        let Ok(entry) = entry_at(&self.reached, entry) else { return false };
        self.entry = entry;
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
            let closure = self.reached.closure(&[root]);
            let failure = closure.iter().rev().find_map(|at| self.reached.list[*at].1.as_ref().err()).or_else(|| closure.iter().rev().find_map(|at| self.broken.get(at)));
            match failure {
                Some(error) => failures.entry(error_kind(error)).or_default().push(format!("{name}: {error}")),
                None => runnable.push((root, name.as_str())),
            }
        }
        let statuses = self.image.run_cases(&runnable, &INPUTS)?;
        for ((_, name), status) in runnable.iter().zip(statuses).filter(|(_, status)| *status != 0) {
            failures.entry(format!("crashed with status {status}")).or_default().push((*name).to_owned());
        }
        print_failures(&failures);
        let ms = |at: Duration| at.as_secs_f64() * 1e3;
        eprintln!("cases {} found, {} ran, {} functions reached, built in {:.1} ms, total {:.1} ms", self.labels.len(), runnable.len(), self.reached.list.len(), ms(built), ms(started.elapsed()));
        Ok(())
    }
}

impl Build {
    /// Runs the binary's `main` with `args` as argv, or fails with every reason a function it reaches did not compile.
    pub fn run(&self, args: &[String]) -> Result<i32> {
        let Some([start, main]) = self.entry else { bail!("no member binary has a `main`") };
        let closure = self.reached.closure(&[start, main]);
        let mut failures = Failures::new();
        for at in &closure {
            if let Some(error) = self.reached.list[*at].1.as_ref().err().or_else(|| self.broken.get(at)) {
                failures.entry(error_kind(error)).or_default().push(error.clone());
            }
        }
        if !failures.is_empty() {
            print_failures(&failures);
            let total = self.reached.list.iter().filter(|(_, typed)| typed.is_err()).count() + self.broken.len();
            bail!("{} of {} functions that `main` reaches did not compile ({total} of {} reached in all)", failures.values().map(Vec::len).sum::<usize>(), closure.len(), self.reached.list.len());
        }
        self.image.run_main((start, main), args)
    }
}

/// Runs the binary's `main` with `args` as argv, compiling each function on its first call.
pub fn run(workspace: &graph::Workspace, targets: &Targets, args: &[String]) -> Result<i32> {
    let (mut outline, mut names) = crate::macros::load_eager(workspace, targets, &[])?;
    let Some(main) = main_item(workspace, &outline) else { bail!("no member binary has a `main`") };
    let status = crate::image::run_on_demand(&mut outline, &mut names, main, args);
    std::mem::forget((outline, names));
    status
}

fn member_files(workspace: &graph::Workspace, outline: &outline::Outline) -> Vec<PathBuf> {
    let members = outline.roots.iter().zip(&outline.crates).filter(|(root, _)| is_member(workspace, root));
    let mut paths: Vec<PathBuf> = members.flat_map(|(_, tree)| tree.files.iter().filter(|file| !file.expanded).map(|file| file.source.path.clone())).collect();
    paths.sort();
    paths.dedup();
    paths
}

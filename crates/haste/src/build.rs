//! A workspace loaded once and kept: its `case_` functions built into one image, and its binary compiled on first call, each brought up to date after an edit.

use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Result, bail};
use crate::graph::{self, Targets};
use crate::image::{Image, Program};
use crate::macros::Compilation;
use crate::names::ItemRef;
use crate::outline::{self, edit::Edit};
use crate::sema::reach::{Instance, Reached, reach_again, reach_anew};
use crate::sema::{self, Edited, Cache};
use rustc_hash::FxHashMap as HashMap;

use crate::cli::{Failures, case_items, error_kind, is_member, print_failures};

const INPUTS: [i64; 5] = [-5, 0, 1, 3, 20];

/// A loaded workspace kept between requests: its compilation, the `case_` functions once a request needs them, and the binary once a request runs it.
pub struct Build {
    compilation: Compilation,
    cases: Option<Cases>,
    program: Option<Program>,
    stamps: Vec<(PathBuf, Option<(SystemTime, u64)>)>,
}

/// The `case_` functions of the members inferred and compiled into one image: the inference store, their names and indexes, what they reach, and why each function that did not compile failed.
struct Cases {
    store: Option<Cache>,
    labels: Vec<String>,
    roots: Vec<usize>,
    reached: Reached,
    image: Image,
    broken: HashMap<usize, String>,
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

fn recompile(image: &mut Image, broken: &mut HashMap<usize, String>, (cx, reached): (&mut sema::Context<'_>, &Reached), fresh: &[usize]) {
    fresh.iter().for_each(|at| _ = broken.remove(at));
    broken.extend(image.compile(cx, reached, fresh));
}

impl Cases {
    /// Infers and compiles every case of `workspace` from nothing.
    fn new(workspace: &graph::Workspace, compilation: &mut Compilation) -> Result<Cases> {
        let (cases, roots) = case_roots(workspace, &compilation.outline);
        crate::macros::reach_resolved(compilation, |_| roots.clone(), |mut cx, reached| {
            let (image, broken) = Image::build(&mut cx, &reached)?;
            let labels = cases.iter().map(|item| cx.name(*item).to_owned()).collect();
            let roots = roots.iter().map(|root| reached.index[root]).collect();
            Ok(Cases { store: Some(cx.release().0), labels, roots, reached, image, broken })
        })
    }

    fn again(&mut self, compilation: &Compilation, stale: &[(u32, u32)]) -> bool {
        let Some(mut store) = self.store.take() else { return false };
        let touched = store.touched(self.reached.list.iter().map(|(instance, _)| instance), Edited::Files(stale));
        store.forget(&compilation.outline, Edited::Files(stale));
        let mut cx = sema::Context::new(&compilation.outline, &compilation.names, store);
        let fresh = reach_again(&mut cx, &mut self.reached, &touched);
        if !cx.resolved() {
            return false;
        }
        recompile(&mut self.image, &mut self.broken, (&mut cx, &self.reached), &fresh);
        self.store = Some(cx.release().0);
        true
    }

    fn reload(&mut self, workspace: &graph::Workspace, compilation: &Compilation, crates: &[u32]) -> bool {
        let Some(mut store) = self.store.take() else { return false };
        let touched = store.touched(self.reached.list.iter().map(|(instance, _)| instance), Edited::Crates(crates));
        store.forget(&compilation.outline, Edited::Crates(crates));
        let (cases, roots) = case_roots(workspace, &compilation.outline);
        let mut cx = sema::Context::new(&compilation.outline, &compilation.names, store);
        let fresh = reach_anew(&mut cx, &mut self.reached, &touched, &roots);
        if !cx.resolved() {
            return false;
        }
        recompile(&mut self.image, &mut self.broken, (&mut cx, &self.reached), &fresh);
        self.labels = cases.iter().map(|item| cx.name(*item).to_owned()).collect();
        self.roots = roots.iter().map(|root| self.reached.index[root]).collect();
        self.store = Some(cx.release().0);
        true
    }

    /// Runs every case whose functions all compiled and prints what failed and how long building took.
    fn report(&self, started: Instant, built: Duration) -> Result<()> {
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
    /// Loads every crate of `workspace`, resolves its names and expands its macros, inferring and compiling nothing yet.
    pub fn load(workspace: &graph::Workspace, targets: &Targets) -> Result<Build> {
        let compilation = crate::macros::load_eager(workspace, targets, &[])?;
        let stamps = stamps(workspace, &compilation.outline);
        Ok(Build { compilation, cases: None, program: None, stamps })
    }

    /// Infers and compiles the `case_` functions unless that is done.
    pub fn compile_cases(&mut self, workspace: &graph::Workspace) -> Result<()> {
        if self.cases.is_none() {
            self.cases = Some(Cases::new(workspace, &mut self.compilation)?);
        }
        Ok(())
    }

    /// Brings the build up to date with the member files: infers and compiles again the functions of files whose bodies changed, reads again a crate whose items changed, and loads from nothing when that fails.
    pub fn refresh(&mut self, workspace: &graph::Workspace, targets: &Targets) -> Result<()> {
        let (mut stale, mut crates) = (Vec::new(), Vec::new());
        for at in 0..self.stamps.len() {
            let now = stamp(&self.stamps[at].0);
            if now == self.stamps[at].1 {
                continue;
            }
            self.stamps[at].1 = now;
            match self.compilation.outline.reread(&self.stamps[at].0) {
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
        if let Some(program) = &mut self.program {
            program.forget(&self.compilation.outline, Edited::Files(stale));
        }
        self.cases.as_mut().is_none_or(|cases| cases.again(&self.compilation, stale))
    }

    fn reload(&mut self, workspace: &graph::Workspace, edited: Vec<u32>) -> bool {
        let mut crates: Vec<u32> = Vec::new();
        for krate in edited {
            if crates.contains(&krate) {
                continue;
            }
            let Ok(reloaded) = self.compilation.outline.reload(krate as usize) else { return false };
            crates.extend(reloaded.into_iter().map(|at| at as u32));
        }
        crates.sort_unstable();
        crates.dedup();
        crates.iter().for_each(|&krate| self.compilation.names.reset(krate as usize));
        if crate::macros::resolve(&mut self.compilation).is_err() {
            return false;
        }
        if let Some(program) = self.program.take() {
            let store = program.into_store(&self.compilation.outline, Edited::Crates(&crates));
            let main = main_item(workspace, &self.compilation.outline);
            self.program = main.and_then(|main| Program::new(&mut self.compilation, store, main).ok());
        }
        self.stamps = stamps(workspace, &self.compilation.outline);
        self.cases.as_mut().is_none_or(|cases| cases.reload(workspace, &self.compilation, &crates))
    }

    fn rebuild(&mut self, workspace: &graph::Workspace, targets: &Targets) -> Result<()> {
        let had_cases = self.cases.is_some();
        *self = Build::load(workspace, targets)?;
        if had_cases {
            self.compile_cases(workspace)?;
        }
        Ok(())
    }

    /// Runs every case whose functions all compiled and prints what failed and how long building took.
    pub fn report(&self, started: Instant, built: Duration) -> Result<()> {
        let Some(cases) = &self.cases else { bail!("the cases are not compiled") };
        cases.report(started, built)
    }

    /// Runs the binary's `main` with `args` as argv in a forked child, compiling each function on its first call and keeping that code for the next run.
    pub fn run(&mut self, workspace: &graph::Workspace, args: &[String]) -> Result<i32> {
        let program = match self.program.take() {
            Some(program) => program,
            None => {
                let Some(main) = main_item(workspace, &self.compilation.outline) else { bail!("no member binary has a `main`") };
                Program::new(&mut self.compilation, Cache::default(), main)?
            }
        };
        let program = self.program.insert(program);
        program.run(&mut self.compilation, args)
    }
}

/// Runs the binary's `main` with `args` as argv, compiling each function on its first call.
pub fn run(workspace: &graph::Workspace, targets: &Targets, args: &[String]) -> Result<i32> {
    let mut compilation = crate::macros::load_eager(workspace, targets, &[])?;
    let Some(main) = main_item(workspace, &compilation.outline) else { bail!("no member binary has a `main`") };
    let status = crate::image::run_on_demand(&mut compilation, (sema::Cache::default(), main), args);
    std::mem::forget(compilation);
    status
}

fn member_files(workspace: &graph::Workspace, outline: &outline::Outline) -> Vec<PathBuf> {
    let members = outline.roots.iter().zip(&outline.crates).filter(|(root, _)| is_member(workspace, root));
    let mut paths: Vec<PathBuf> = members.flat_map(|(_, tree)| tree.files.iter().filter(|file| !file.expanded).map(|file| file.source.path.clone())).collect();
    paths.sort();
    paths.dedup();
    paths
}

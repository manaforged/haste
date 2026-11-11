//! A workspace loaded once and kept: its `case_` functions and its binary's reach each built into one image, and its binary compiled on first call, each brought up to date after an edit.

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

use crate::cli::{Failures, case_items, error_kind, print_failures};

const INPUTS: [i64; 5] = [-5, 0, 1, 3, 20];

/// A loaded workspace kept between requests: its compilation, the `case_` functions and the binary's reach once a request needs them, and the binary once a request runs it.
pub struct Build {
    compilation: Compilation,
    cases: Option<Batch>,
    binary: Option<Batch>,
    program: Option<Program>,
    stamps: Vec<Watch>,
    outline_span: sema::census::Span,
}

/// A file that a kept build watches: its path, its last time and size, and for a file that a macro includes, the file that holds the call.
struct Watch {
    path: PathBuf,
    stamp: Stamp,
    by: Option<PathBuf>,
}

/// The roots a batch compiles from, picked from the loaded workspace.
type Pick = fn(&graph::Workspace, &outline::Outline) -> Vec<ItemRef>;

/// Root functions inferred and compiled ahead of time into one image: how to pick them, the inference store, their names and indexes, what they reach, and why each function that did not compile failed.
struct Batch {
    pick: Pick,
    store: Option<Cache>,
    labels: Vec<String>,
    roots: Vec<usize>,
    reached: Reached,
    image: Image,
    broken: HashMap<usize, String>,
    reach_span: sema::census::Span,
    codegen_span: sema::census::Span,
}

/// The last-modified time and size of the file at `path`, which change when it is written.
pub(crate) type Stamp = Option<(SystemTime, u64)>;

/// The stamp of the file at `path`, or none when it cannot be read.
pub(crate) fn stamp(path: &std::path::Path) -> Stamp {
    let meta = std::fs::metadata(path).ok()?;
    Some((meta.modified().ok()?, meta.len()))
}

/// The source files of every crate whose package is local, which an edit can change, the files `include!` read into them, and the files their `include_str!` and `include_bytes!` calls read.
fn stamps(workspace: &graph::Workspace, outline: &outline::Outline) -> Vec<Watch> {
    let local: Vec<usize> = (0..outline.roots.len()).filter(|&krate| outline.package(workspace, krate).local()).collect();
    let mut sources: Vec<PathBuf> = local.iter().flat_map(|&krate| outline.crates[krate].files.iter().map(|file| file.source.path.clone())).collect();
    sources.sort();
    sources.dedup();
    let included = local.iter().flat_map(|&krate| outline.included(krate)).map(|(path, by)| (path, Some(by)));
    sources.into_iter().map(|path| (path, None)).chain(included).map(|(path, by)| Watch { stamp: stamp(&path), path, by }).collect()
}

fn batch_roots(workspace: &graph::Workspace, outline: &outline::Outline, pick: Pick) -> (Vec<ItemRef>, Vec<Instance>) {
    let cases = pick(workspace, outline);
    let roots = cases.iter().map(|item| (*item, sema::substs::Substs::EMPTY)).collect();
    (cases, roots)
}

/// The `main` of the first member binary.
fn main_item(workspace: &graph::Workspace, outline: &outline::Outline) -> Option<ItemRef> {
    let krate = outline.members(workspace).find(|&krate| outline.roots[krate].role == outline::Role::Bin)?;
    outline.main(krate)
}

fn main_items(workspace: &graph::Workspace, outline: &outline::Outline) -> Vec<ItemRef> {
    main_item(workspace, outline).into_iter().collect()
}

fn recompile(image: &mut Image, broken: &mut HashMap<usize, String>, (cx, reached): (&mut sema::Context<'_>, &Reached), fresh: &[usize]) {
    fresh.iter().for_each(|at| drop(broken.remove(at)));
    broken.extend(image.compile(cx, reached, fresh));
}

impl Batch {
    /// Infers and compiles every root `pick` finds in `workspace` from nothing.
    fn new(workspace: &graph::Workspace, compilation: &mut Compilation, pick: Pick) -> Result<Batch> {
        let reach_clock = sema::census::Clock::now();
        let (cases, roots) = batch_roots(workspace, &compilation.outline, pick);
        crate::macros::reach_resolved(compilation, |_| roots.clone(), |mut cx, reached| {
            let reach_span = reach_clock.elapsed();
            let codegen_clock = sema::census::Clock::now();
            let (image, broken) = Image::build(&mut cx, &reached)?;
            let codegen_span = codegen_clock.elapsed();
            let labels = cases.iter().map(|item| cx.name(*item).to_owned()).collect();
            let roots = roots.iter().map(|root| reached.index[root]).collect();
            Ok(Batch { pick, store: Some(cx.release().0), labels, roots, reached, image, broken, reach_span, codegen_span })
        })
    }

    fn again(&mut self, compilation: &Compilation, stale: &[(u32, u32)]) -> bool {
        let Some(mut store) = self.store.take() else { return false };
        let touched = store.invalidate(&compilation.outline, self.reached.list.iter().map(|(instance, _)| instance), Edited::Files(stale));
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
        let touched = store.invalidate(&compilation.outline, self.reached.list.iter().map(|(instance, _)| instance), Edited::Crates(crates));
        let (cases, roots) = batch_roots(workspace, &compilation.outline, self.pick);
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

    /// Sorts the roots into the failures of those that reach a function that did not infer or compile, and the rest by index and name.
    fn sort(&self) -> (Failures, Vec<(usize, &str)>) {
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
        (failures, runnable)
    }

    /// Prints what failed to build and how long building took, and returns 1 when anything failed.
    fn check(&self, built: Duration) -> i32 {
        let (failures, _) = self.sort();
        print_failures(&failures);
        eprintln!("{} functions reached, {} failed, built in {:.1} ms", self.reached.list.len(), failures.values().map(Vec::len).sum::<usize>(), built.as_secs_f64() * 1e3);
        i32::from(!failures.is_empty())
    }

    /// Runs every case whose functions all compiled and prints what failed and how long building took.
    fn report(&self, started: Instant, built: Duration) -> Result<()> {
        let (mut failures, runnable) = self.sort();
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
        let outline_clock = sema::census::Clock::now();
        let compilation = crate::macros::load_eager(workspace, targets, &[])?;
        let stamps = stamps(workspace, &compilation.outline);
        let outline_span = outline_clock.elapsed();
        Ok(Build { compilation, cases: None, binary: None, program: None, stamps, outline_span })
    }

    /// Infers and compiles the `case_` functions unless that is done.
    pub fn compile_cases(&mut self, workspace: &graph::Workspace) -> Result<()> {
        if self.cases.is_none() {
            self.cases = Some(Batch::new(workspace, &mut self.compilation, case_items)?);
        }
        Ok(())
    }

    /// Infers and compiles every function the binary's `main` reaches unless that is done.
    pub fn compile_binary(&mut self, workspace: &graph::Workspace) -> Result<()> {
        if main_item(workspace, &self.compilation.outline).is_none() {
            bail!("no member binary has a `main`");
        }
        if self.binary.is_none() {
            self.binary = Some(Batch::new(workspace, &mut self.compilation, main_items)?);
        }
        Ok(())
    }

    /// Brings the build up to date with the local files: infers and compiles again the functions of files whose bodies changed, reads again a crate whose items changed, and loads from nothing when that fails.
    pub fn refresh(&mut self, workspace: &graph::Workspace, targets: &Targets) -> Result<()> {
        let (mut stale, mut crates) = (Vec::new(), Vec::new());
        for watch in &mut self.stamps {
            let now = stamp(&watch.path);
            if now == watch.stamp {
                continue;
            }
            watch.stamp = now;
            let edit = match &watch.by {
                Some(by) => self.compilation.outline.reexpand(by),
                None => self.compilation.outline.reread(&watch.path),
            };
            match edit {
                Ok(Edit::Same) => {}
                Ok(Edit::Bodies(slots)) if slots.iter().any(|&(krate, _)| expands(workspace, &self.compilation.outline, krate)) => crates.extend(slots.iter().map(|&(krate, _)| krate)),
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
        let compilation = &self.compilation;
        [&mut self.cases, &mut self.binary].into_iter().flatten().all(|batch| batch.again(compilation, stale))
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
        let seen: HashMap<PathBuf, Stamp> = self.stamps.iter().map(|watch| (watch.path.clone(), watch.stamp)).collect();
        self.stamps = stamps(workspace, &self.compilation.outline);
        for watch in &mut self.stamps {
            if let Some(&stamp) = seen.get(&watch.path) {
                watch.stamp = stamp;
            }
        }
        let compilation = &self.compilation;
        [&mut self.cases, &mut self.binary].into_iter().flatten().all(|batch| batch.reload(workspace, compilation, &crates))
    }

    fn rebuild(&mut self, workspace: &graph::Workspace, targets: &Targets) -> Result<()> {
        let had = (self.cases.is_some(), self.binary.is_some());
        *self = Build::load(workspace, targets)?;
        if had.0 {
            self.compile_cases(workspace)?;
        }
        if had.1 {
            self.compile_binary(workspace)?;
        }
        Ok(())
    }

    /// Runs every case whose functions all compiled and prints what failed and how long building took.
    pub fn report(&self, started: Instant, built: Duration) -> Result<()> {
        let Some(cases) = &self.cases else { bail!("the cases are not compiled") };
        cases.report(started, built)
    }

    /// Prints what failed to build in the binary's reach and how long building took, and returns 1 when anything failed.
    pub fn check(&self, built: Duration) -> Result<i32> {
        let Some(binary) = &self.binary else { bail!("the binary is not compiled") };
        Ok(binary.check(built))
    }

    /// Writes the equivalence-class, hazard, origin, and code-size census of the compiled binary to `path`.
    pub fn write_stats(&mut self, workspace: &graph::Workspace, graph: sema::census::Span, path: &std::path::Path) -> Result<()> {
        let Some(binary) = &mut self.binary else { bail!("the binary is not compiled") };
        let Some(store) = binary.store.take() else { bail!("the binary has no inference store") };
        let timings = sema::census::Timings { graph, outline: self.outline_span, reach: binary.reach_span, codegen: binary.codegen_span };
        let input = sema::census::CensusInput { workspace, reached: &binary.reached, code_sizes: binary.image.code_sizes(), compile_times: binary.image.compile_times(), timings };
        let mut cx = sema::Context::new(&self.compilation.outline, &self.compilation.names, store);
        let result = sema::census::write_stats(&mut cx, input, path);
        binary.store = Some(cx.release().0);
        result
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

/// Whether crate `krate` is a procedural macro crate, whose bodies decide the code of the crates that call its macros.
fn expands(workspace: &graph::Workspace, outline: &outline::Outline, krate: u32) -> bool {
    outline.package(workspace, krate as usize).manifest.proc_macro
}

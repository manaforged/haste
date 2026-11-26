use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use rustc_hash::FxHashMap as HashMap;

use crate::cli::{Failures, test_items};
use crate::graph::{self, Targets};
use crate::image::Program;
use crate::macros::Compilation;
use crate::names::ItemRef;
use crate::outline::{self, tree::Profile};
use crate::sema::{self, Cache, Edited};

mod batch;
mod report;
pub(crate) mod substrate;
mod test_args;
mod watch;

use batch::{Batch, BatchKind};
use substrate::{cache_with_substrate, save_jit_substrate};
use test_args::map_tests;
use watch::{Stamp, Watch, body_edits, edits, paths, pre_stamps, stamps};

/// A loaded workspace kept between requests: its compilation, the `case_` functions and the binary's reach once a request needs them, and the binary once a request runs it.
pub struct Build {
    compilation: Compilation,
    cases: Option<Batch>,
    binary: Option<Batch>,
    program: Option<Program>,
    stamps: Vec<Watch>,
    outline_span: sema::census::Span,
}

/// The `main` of the selected member binary or example (or the first member binary when neither `--bin` nor `--example` is set).
pub(crate) fn main_item(
    workspace: &graph::Workspace,
    outline: &outline::Outline,
) -> Option<ItemRef> {
    let sel = crate::selection();
    let (role, wanted) = match &sel.example {
        Some(name) => (outline::Role::Example, Some(name.as_str())),
        None => (outline::Role::Bin, sel.bin.as_deref()),
    };
    let krate = outline.members(workspace).find(|&krate| {
        outline.roots[krate].role == role
            && wanted.is_none_or(|name| outline.roots[krate].name == name)
    })?;
    outline.main(krate)
}

fn missing_target_error() -> anyhow::Error {
    let sel = crate::selection();
    if let Some(name) = &sel.example {
        anyhow::anyhow!("no example target named '{name}'")
    } else if let Some(name) = &sel.bin {
        anyhow::anyhow!("no bin target named '{name}'")
    } else {
        anyhow::anyhow!("no member binary has a `main`")
    }
}

fn require_main(workspace: &graph::Workspace, outline: &outline::Outline) -> Result<ItemRef> {
    main_item(workspace, outline).ok_or_else(missing_target_error)
}

impl Build {
    /// Loads every crate of `workspace`, resolves its names and expands its macros, inferring and compiling nothing yet.
    pub fn load(workspace: &graph::Workspace, targets: &Targets) -> Result<Build> {
        let outline_clock = sema::census::Clock::now();
        let pre = pre_stamps(workspace);
        let compilation = crate::macros::load_lazy(workspace, targets)?;
        let mut stamps = stamps(workspace, &compilation.outline);
        for watch in &mut stamps {
            if let Some(&old) = pre.get(&watch.path) {
                watch.stamp = old;
            }
        }
        let outline_span = outline_clock.elapsed();
        Ok(Build {
            compilation,
            cases: None,
            binary: None,
            program: None,
            stamps,
            outline_span,
        })
    }

    fn load_members(&mut self, workspace: &graph::Workspace) -> Result<()> {
        let members: Vec<usize> = self.compilation.outline.members(workspace).collect();
        let mut loaded_more = false;
        for member in members {
            if self.compilation.outline.crates[member].partial() {
                self.compilation.outline.load_crate(member)?;
                loaded_more = true;
            }
        }
        if loaded_more {
            crate::macros::resolve(&mut self.compilation)?;
            let errors = self.compilation.outline.local_errors(workspace);
            if !errors.is_empty() {
                anyhow::bail!("{}", errors.join("\n"));
            }
        }
        Ok(())
    }

    /// Infers and compiles the `#[test]` or `case_` functions unless that is done.
    pub fn compile_cases(&mut self, workspace: &graph::Workspace) -> Result<()> {
        if self.cases.is_none() {
            self.load_members(workspace)?;
            let discovered = test_items(workspace, &self.compilation.outline);
            #[cfg(feature = "diff-harness")]
            let kind = if discovered.is_empty() {
                BatchKind::Cases
            } else {
                BatchKind::Tests
            };
            #[cfg(not(feature = "diff-harness"))]
            let kind = BatchKind::Tests;
            let mut batch = Batch::new(workspace, &mut self.compilation, kind, false)?;
            if kind == BatchKind::Tests {
                let starts: Vec<Option<usize>> = batch.tests.iter().map(|(_, s)| *s).collect();
                batch.tests = map_tests(&batch.roots, &batch.reached, &starts, discovered);
            }
            self.cases = Some(batch);
        }
        Ok(())
    }

    /// Infers and compiles every function the binary's `main` reaches unless that is done.
    pub fn compile_binary(&mut self, workspace: &graph::Workspace) -> Result<()> {
        self.compile_binary_with(workspace, false)
    }

    /// Infers and check-compiles `main` for one-shot `haste build`, dropping `Store.finals` before codegen.
    pub fn compile_binary_oneshot(&mut self, workspace: &graph::Workspace) -> Result<()> {
        self.compile_binary_with(workspace, true)
    }

    fn compile_binary_with(
        &mut self,
        workspace: &graph::Workspace,
        drop_finals: bool,
    ) -> Result<()> {
        if self.binary.is_none() {
            let sel = crate::selection();
            let kind = if main_item(workspace, &self.compilation.outline).is_some() {
                self.load_members(workspace)?;
                BatchKind::MainCheck
            } else if sel.bin.is_some() || sel.example.is_some() {
                return Err(missing_target_error());
            } else {
                self.load_members(workspace)?;
                BatchKind::LibCheck
            };
            self.binary = Some(Batch::new(
                workspace,
                &mut self.compilation,
                kind,
                drop_finals,
            )?);
        }
        Ok(())
    }

    /// Number of reached functions in the compiled binary batch.
    pub(crate) fn reached_count(&self) -> usize {
        self.binary.as_ref().map_or(0, |bin| bin.reached.list.len())
    }

    /// Local source and included file paths watched by this build.
    pub(crate) fn watch_paths(&self) -> Vec<PathBuf> {
        paths(&self.stamps)
    }

    /// Brings the build up to date with the local files: infers and compiles again the functions of files whose bodies changed, reads again a crate whose items changed, and loads from nothing when that fails.
    pub fn refresh(&mut self, workspace: &graph::Workspace, targets: &Targets) -> Result<()> {
        let Ok((stale, crates)) = edits(workspace, &mut self.stamps, &mut self.compilation.outline)
        else {
            return self.rebuild(workspace, targets);
        };
        let done = match (stale.is_empty(), crates.is_empty()) {
            (true, true) => return Ok(()),
            (_, false) => self.reload(
                workspace,
                crates
                    .into_iter()
                    .chain(stale.iter().map(|&(krate, _)| krate))
                    .collect(),
            ),
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
        [&mut self.cases, &mut self.binary]
            .into_iter()
            .flatten()
            .all(|batch| batch.again(compilation, stale))
    }

    fn reload(&mut self, workspace: &graph::Workspace, edited: Vec<u32>) -> bool {
        let mut crates: Vec<u32> = Vec::new();
        for krate in edited {
            if crates.contains(&krate) {
                continue;
            }
            let Ok(reloaded) = self.compilation.outline.reload(krate as usize) else {
                return false;
            };
            crates.extend(reloaded.into_iter().map(|at| at as u32));
        }
        crates.sort_unstable();
        crates.dedup();
        crates
            .iter()
            .for_each(|&krate| self.compilation.names.reset(krate as usize));
        if crate::macros::resolve(&mut self.compilation).is_err() {
            return false;
        }
        if let Some(program) = self.program.take() {
            let store = program.into_store(&self.compilation.outline, Edited::Crates(&crates));
            let main = main_item(workspace, &self.compilation.outline);
            self.program =
                main.and_then(|main| Program::new(&mut self.compilation, store, main).ok());
        }
        let seen: HashMap<PathBuf, Stamp> = self
            .stamps
            .iter()
            .map(|watch| (watch.path.clone(), watch.stamp))
            .collect();
        self.stamps = stamps(workspace, &self.compilation.outline);
        for watch in &mut self.stamps {
            if let Some(&stamp) = seen.get(&watch.path) {
                watch.stamp = stamp;
            }
        }
        let compilation = &self.compilation;
        [&mut self.cases, &mut self.binary]
            .into_iter()
            .flatten()
            .all(|batch| batch.reload(workspace, compilation, &crates))
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

    /// Compiles `main` and `std::rt::lang_start` into an AOT `Image` and writes a standalone executable to `path`.
    pub(crate) fn emit_executable(
        &mut self,
        workspace: &graph::Workspace,
        path: &std::path::Path,
        drop_finals: bool,
    ) -> Result<i32> {
        let started = Instant::now();
        if self
            .binary
            .as_ref()
            .is_none_or(|bin| bin.kind != BatchKind::MainAot)
        {
            self.binary = Some(Batch::new(
                workspace,
                &mut self.compilation,
                BatchKind::MainAot,
                drop_finals,
            )?);
        }
        let bin = self.binary.as_mut().context("the binary is not compiled")?;
        let failed = bin.print_diagnostics(&self.compilation.outline);
        if failed > 0 {
            bail!("{failed} functions failed to compile");
        }
        bin.write_aot(&self.compilation, path)?;
        eprintln!(
            "built {} ({} functions reached, {:.1} ms)",
            path.display(),
            bin.reached.list.len(),
            started.elapsed().as_secs_f64() * 1e3
        );
        Ok(0)
    }

    /// Runs every case or `#[test]` whose functions all compiled and returns 0 when all pass, else 101.
    pub fn report(&self, started: Instant, built: Duration, filters: &[String]) -> Result<i32> {
        let Some(cases) = &self.cases else {
            bail!("the cases are not compiled")
        };
        cases.report(&self.compilation.outline, started, built, filters)
    }

    /// Prints what failed to build in the binary's reach and how long building took, and returns 1 when anything failed.
    pub fn check(&self, built: Duration) -> Result<i32> {
        let Some(binary) = &self.binary else {
            bail!("the binary is not compiled")
        };
        Ok(binary.check(&self.compilation.outline, built))
    }

    /// Why each function the binary's `main` reaches did not type-check or compile, at the file and line of its item, keyed by the kind of error.
    pub fn failures(&mut self) -> Result<Failures> {
        let Some(binary) = &mut self.binary else {
            bail!("the binary is not compiled")
        };
        let Some(store) = binary.store.take() else {
            bail!("the binary has no inference store")
        };
        let cx = sema::Context::new(&self.compilation.outline, &self.compilation.names, store);
        let failures = binary.failures(&cx);
        binary.store = Some(cx.release().0);
        Ok(failures)
    }

    /// Writes the equivalence-class, hazard, origin, and code-size census of the compiled binary to `path`.
    pub fn write_stats(
        &mut self,
        workspace: &graph::Workspace,
        graph: sema::census::Span,
        path: &std::path::Path,
    ) -> Result<()> {
        let Some(binary) = &mut self.binary else {
            bail!("the binary is not compiled")
        };
        let Some(store) = binary.store.take() else {
            bail!("the binary has no inference store")
        };
        let timings = sema::census::Timings {
            graph,
            outline: self.outline_span,
            reach: binary.reach_span,
            codegen: binary.codegen_span,
        };
        let input = sema::census::CensusInput {
            workspace,
            reached: &binary.reached,
            code_sizes: binary.image.code_sizes(),
            compile_times: binary.image.compile_times(),
            timings,
        };
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
                let main = require_main(workspace, &self.compilation.outline)?;
                let (_, cache) = cache_with_substrate(workspace, true);
                Program::new(&mut self.compilation, cache, main)?
            }
        };
        let program = self.program.insert(program);
        program.run(&mut self.compilation, args)
    }
}

/// Compiles the program whose `main` is `main` ahead of time and writes it as a standalone executable to `path`.
pub(crate) fn emit_main(
    workspace: &graph::Workspace,
    compilation: &mut Compilation,
    main: ItemRef,
    path: &std::path::Path,
) -> Result<()> {
    Batch::rooted(
        workspace,
        compilation,
        (BatchKind::MainAot, Some(main)),
        true,
    )?
    .write_aot(compilation, path)
}

/// Runs the binary's `main` with `args` as argv, compiling each function on its first call, and, when prepared code is switched on, under the release profile calling prepped code where it can, racing a prep of what changed since the library was built.
pub fn run(workspace: &graph::Workspace, targets: &Targets, args: &[String]) -> Result<i32> {
    let mut ahead = if crate::profile() == Profile::Release && crate::prep::prepared() {
        crate::prep::ahead(workspace, targets)
    } else {
        None
    };
    let status = run_loaded(workspace, targets, args, ahead.as_mut());
    if !matches!(status, Ok(0))
        && let Some(said) = ahead.as_mut().and_then(crate::prep::Ahead::rejected)
    {
        eprintln!("haste: rustc rejects this program too:\n{said}");
    }
    status
}

fn run_loaded(
    workspace: &graph::Workspace,
    targets: &Targets,
    args: &[String],
    ahead: Option<&mut crate::prep::Ahead>,
) -> Result<i32> {
    let use_prep = crate::profile() == Profile::Release && crate::prep::prepared();
    let mut compilation = if use_prep {
        crate::macros::load_eager(workspace, targets, &[])?
    } else {
        crate::macros::load_lazy(workspace, targets)?
    };
    let main = require_main(workspace, &compilation.outline)?;
    if use_prep {
        bind_prepped(workspace, &mut compilation, args, ahead);
    }
    let (slot, cache) = cache_with_substrate(workspace, true);
    let status = match crate::image::watch::enabled() {
        true => run_watched(workspace, &mut compilation, cache, main, args),
        false => {
            let mut program = Program::new(&mut compilation, cache, main)?;
            let status = program.run(&mut compilation, args);
            if let Some(slot) = slot {
                let (store, image) = program.store_and_image();
                save_jit_substrate(&compilation.outline, store, image, slot);
            }
            std::mem::forget(program);
            status
        }
    };
    std::mem::forget(compilation);
    status
}

fn run_watched(
    workspace: &graph::Workspace,
    compilation: &mut Compilation,
    cache: Cache,
    main: ItemRef,
    args: &[String],
) -> Result<i32> {
    let mut watched = stamps(workspace, &compilation.outline);
    let files = paths(&watched);
    let mut reread = |compilation: &mut Compilation| {
        body_edits(workspace, &mut watched, &mut compilation.outline)
    };
    let mut program = Program::new(compilation, cache, main)?;
    let status = program.run_watched(compilation, args, (files, &mut reread));
    std::mem::forget(program);
    status
}

/// Builds the prepped library of the build of `workspace`, starting rustc before the sources load when it can, and returns what it holds.
pub fn prep(workspace: &graph::Workspace, targets: &Targets) -> Result<crate::prep::Counts> {
    let started = crate::prep::begin(workspace, targets)?;
    let compilation = match crate::macros::load_eager(workspace, targets, &[]) {
        Ok(compilation) => compilation,
        Err(error) => {
            started.cancel();
            return Err(error);
        }
    };
    let counts = started.end(&compilation.outline, workspace);
    std::mem::forget(compilation);
    counts
}

fn bind_prepped(
    workspace: &graph::Workspace,
    compilation: &mut Compilation,
    args: &[String],
    mut ahead: Option<&mut crate::prep::Ahead>,
) {
    if let Some(ahead) = ahead.as_deref_mut() {
        ahead.wait();
    }
    let (manifest, current) = match crate::prep::load(&compilation.outline, workspace, ahead) {
        Ok(loaded) => loaded,
        Err(error) => (Err(error), true),
    };
    if !current && let Err(error) = crate::prep::refresh(&compilation.outline, workspace) {
        eprintln!("haste: not preparing in the background: {error:#}");
    }
    if let Err(error) = manifest.and_then(|manifest| {
        manifest.map_or(Ok(()), |manifest| crate::prep::install(manifest, args))
    }) {
        eprintln!("haste: running without prepped code: {error:#}");
    }
}

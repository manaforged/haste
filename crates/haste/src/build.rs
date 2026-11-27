use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};

use crate::cli::{Failures, test_items};
use crate::graph::{self, Targets};
use crate::image::Program;
use crate::macros::Compilation;
use crate::names::ItemRef;
use crate::outline::{self, tree::Profile};
use crate::sema::{self, Cache, Edited};

mod batch;
mod json_diag;
mod refresh;
mod report;
pub(crate) mod substrate;
mod test_args;
mod watch;

use batch::{Batch, BatchKind};
use refresh::{refresh_edits, update_stamps};
use substrate::{cache_with_substrate, save_jit_substrate};
use test_args::map_tests;
use watch::{Watch, paths, pre_stamps, run_watched, stamps_with_pre, sync_included_stamps};

/// A loaded workspace kept between requests: its compilation, the `case_` functions and the binary's reach once a request needs them, and the binary once a request runs it.
pub struct Build {
    compilation: Compilation,
    cases: Option<Batch>,
    binary: Option<Batch>,
    program: Option<Program>,
    run_checked: bool,
    stamps: Vec<Watch>,
    outline_span: sema::census::Span,
}

fn default_run_match(
    workspace: &graph::Workspace,
    outline: &outline::Outline,
    krate: usize,
) -> bool {
    let pkg = &workspace.packages[workspace.graph.units[outline.roots[krate].unit].package];
    pkg.manifest.default_run.as_deref() == Some(outline.roots[krate].name.as_str())
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
    if wanted.is_none()
        && role == outline::Role::Bin
        && let Some(krate) = outline.members(workspace).find(|&k| {
            outline.roots[k].role == role
                && outline.main(k).is_some()
                && default_run_match(workspace, outline, k)
        })
    {
        return outline.main(krate);
    }
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
    let sel = crate::selection();
    if sel.bin.is_none() && sel.example.is_none() {
        let bins: Vec<usize> = outline
            .members(workspace)
            .filter(|&k| outline.roots[k].role == outline::Role::Bin && outline.main(k).is_some())
            .collect();
        if bins.len() > 1 {
            let default_matches: Vec<usize> = bins
                .iter()
                .copied()
                .filter(|&k| default_run_match(workspace, outline, k))
                .collect();
            if default_matches.len() == 1 {
                return outline
                    .main(default_matches[0])
                    .ok_or_else(missing_target_error);
            }
            let names: Vec<&str> = bins
                .iter()
                .map(|&k| outline.roots[k].name.as_str())
                .collect();
            bail!(
                "multiple binary targets found ({}); pass `-p <PKG>`, `--bin <NAME>`, or set `default-run` in Cargo.toml",
                names.join(", ")
            );
        }
    }
    main_item(workspace, outline).ok_or_else(missing_target_error)
}

pub(super) fn load_members(
    workspace: &graph::Workspace,
    compilation: &mut Compilation,
) -> Result<()> {
    let members: Vec<usize> = compilation.outline.members(workspace).collect();
    let mut loaded_more = false;
    for member in members {
        if compilation.outline.crates[member].partial() {
            compilation.outline.load_crate(member)?;
            loaded_more = true;
        }
    }
    if loaded_more {
        crate::macros::resolve(compilation)?;
        let errors = compilation.outline.local_errors(workspace);
        if !errors.is_empty() {
            anyhow::bail!("{}", errors.join("\n"));
        }
    }
    Ok(())
}

fn check_and_print_member_items(
    workspace: &graph::Workspace,
    compilation: &mut Compilation,
    cache: Cache,
    target: Option<usize>,
) -> Result<std::result::Result<Cache, i32>> {
    let (cache, diags) = batch::check_member_items(workspace, compilation, cache, target)?;
    if diags.is_empty() {
        return Ok(Ok(cache));
    }
    report::print_formatted_diagnostics(Some(&compilation.outline), &diags);
    Ok(Err(101))
}

thread_local! {
    static JSON_FINISHED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn mark_json_finished() {
    JSON_FINISHED.with(|slot| slot.set(true));
}

/// Emits JSON `compiler-message` and `build-finished` for an early error when JSON output is active; returns `true` when handled.
pub(crate) fn emit_early_json_error(err: &anyhow::Error) -> bool {
    let fmt = crate::selection().message_format;
    if !matches!(
        fmt,
        crate::MessageFormat::Json | crate::MessageFormat::JsonRenderDiagnostics
    ) {
        return false;
    }
    if JSON_FINISHED.with(|slot| slot.replace(false)) {
        if fmt == crate::MessageFormat::JsonRenderDiagnostics {
            eprintln!("error: {err:#}");
        }
        return true;
    }
    let text = format!("{err:#}");
    let diags = report::split_error_diagnostics(&text);
    report::print_formatted_diagnostics(None, &diags);
    crate::diag::print_build_finished(false);
    true
}

impl Build {
    /// Loads every crate of `workspace`, resolves its names and expands its macros, inferring and compiling nothing yet.
    pub fn load(workspace: &graph::Workspace, targets: &Targets) -> Result<Build> {
        let outline_clock = sema::census::Clock::now();
        let pre = pre_stamps(workspace);
        let compilation = crate::macros::load_lazy(workspace, targets)?;
        let stamps = stamps_with_pre(workspace, &compilation.outline, &pre);
        let outline_span = outline_clock.elapsed();
        Ok(Build {
            compilation,
            cases: None,
            binary: None,
            program: None,
            run_checked: false,
            stamps,
            outline_span,
        })
    }

    /// Loads every crate of `workspace` for a one-shot CLI invocation without pre-collecting file watch timestamps.
    pub(crate) fn load_oneshot(workspace: &graph::Workspace, targets: &Targets) -> Result<Build> {
        let outline_clock = sema::census::Clock::now();
        let compilation = crate::macros::load_lazy(workspace, targets)?;
        let outline_span = outline_clock.elapsed();
        Ok(Build {
            compilation,
            cases: None,
            binary: None,
            program: None,
            run_checked: false,
            stamps: Vec::new(),
            outline_span,
        })
    }

    fn load_members(&mut self, workspace: &graph::Workspace) -> Result<()> {
        load_members(workspace, &mut self.compilation)
    }

    /// Infers and compiles the `#[test]` or `case_` functions unless that is done.
    pub fn compile_cases(&mut self, workspace: &graph::Workspace) -> Result<()> {
        self.compile_cases_filtered(workspace, &[])
    }

    /// Infers and compiles matching `#[test]` or `case_` functions unless that is done.
    pub fn compile_cases_filtered(
        &mut self,
        workspace: &graph::Workspace,
        filters: &[String],
    ) -> Result<()> {
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
            let mut batch =
                Batch::new_filtered(workspace, &mut self.compilation, kind, false, filters)?;
            if kind == BatchKind::Tests {
                let starts: Vec<Option<usize>> = batch.tests.iter().map(|(_, s)| *s).collect();
                batch.tests = map_tests(&batch.roots, &batch.reached, &starts, discovered);
                self.materialize_test_bins(workspace, &batch.tests)?;
            }
            self.cases = Some(batch);
        }
        Ok(())
    }

    fn materialize_test_bins(
        &mut self,
        workspace: &graph::Workspace,
        tests: &[(crate::cli::TestItem, Option<usize>)],
    ) -> Result<()> {
        let mut to_emit: Vec<(ItemRef, std::path::PathBuf)> = Vec::new();
        let mut seen = rustc_hash::FxHashSet::default();
        for (meta, _) in tests {
            let krate = meta.item.krate as usize;
            if self.compilation.outline.roots[krate].role != outline::Role::Test {
                continue;
            }
            let test_unit = self.compilation.outline.roots[krate].unit;
            let test_pkg = workspace.graph.units[test_unit].package;
            for (k, v) in &self.compilation.outline.crates[krate].env.vars {
                let Some(bin_name) = k.strip_prefix("CARGO_BIN_EXE_") else {
                    continue;
                };
                let exe = std::path::PathBuf::from(v);
                if !seen.insert(exe.clone()) {
                    continue;
                }
                let bin_krate = self.compilation.outline.roots.iter().position(|r| {
                    r.role == outline::Role::Bin
                        && r.name == bin_name
                        && workspace.graph.units[r.unit].package == test_pkg
                });
                if let Some(bk) = bin_krate
                    && let Some(main_fn) = self.compilation.outline.main(bk)
                {
                    to_emit.push((main_fn, exe));
                }
            }
        }
        for (main_fn, exe) in to_emit {
            if let Some(parent) = exe.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let mut bin_batch = Batch::rooted(
                workspace,
                &mut self.compilation,
                (BatchKind::MainAot, Some(main_fn)),
                (true, &[]),
            )?;
            bin_batch.write_aot(&self.compilation, &exe)?;
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
            let lib_only = sel.lib
                && !sel.bins
                && !sel.all_targets
                && sel.bin.is_none()
                && sel.example.is_none();
            let kind = if !lib_only && main_item(workspace, &self.compilation.outline).is_some() {
                self.load_members(workspace)?;
                BatchKind::MainCheck
            } else if sel.bin.is_some() || sel.example.is_some() {
                return Err(missing_target_error());
            } else if lib_only
                && !self
                    .compilation
                    .outline
                    .members(workspace)
                    .any(|k| self.compilation.outline.roots[k].role == outline::Role::Lib)
            {
                bail!("no library target found");
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
        self.binary
            .as_ref()
            .map_or(0, |bin| bin.reached.live_count())
    }

    /// Local source and included file paths watched by this build.
    pub(crate) fn watch_paths(&self, workspace: &graph::Workspace) -> Vec<PathBuf> {
        if !self.stamps.is_empty() {
            paths(&self.stamps)
        } else {
            watch::watch_paths(workspace, &self.compilation.outline)
        }
    }

    pub(crate) fn env_reads(&self) -> Vec<(String, Option<String>)> {
        let mut out = Vec::new();
        for tree in &self.compilation.outline.crates {
            let reads = crate::sync::lock(&tree.env.env_reads);
            for (k, v) in reads.iter() {
                out.push((k.clone(), v.clone()));
            }
        }
        out.sort();
        out.dedup();
        out
    }

    /// Brings the build up to date with the local files: infers and compiles again the functions of files whose bodies changed, reads again a crate whose items changed, and loads from nothing when that fails.
    pub fn refresh(&mut self, workspace: &graph::Workspace, targets: &Targets) -> Result<()> {
        let Ok((stale, crates, paths)) =
            refresh_edits(workspace, &mut self.stamps, &mut self.compilation.outline)
        else {
            return self.rebuild(workspace, targets);
        };
        let done = match (stale.is_empty(), crates.is_empty()) {
            (true, true) => return Ok(()),
            (_, false) => {
                let mut edited: Vec<u32> = crates
                    .into_iter()
                    .chain(stale.iter().map(|&(krate, _)| krate))
                    .collect();
                edited.sort_unstable();
                edited.dedup();
                self.reload(workspace, edited, &stale, &paths)?
            }
            (false, true) => self.again(workspace, &stale),
        };
        if !done {
            return self.rebuild(workspace, targets);
        }
        Ok(())
    }

    fn again(&mut self, workspace: &graph::Workspace, stale: &[(u32, u32)]) -> bool {
        self.run_checked = false;
        if let Some(program) = &mut self.program {
            program.forget(&self.compilation.outline, Edited::Files(stale));
        }
        let compilation = &self.compilation;
        let ok = [&mut self.cases, &mut self.binary]
            .into_iter()
            .flatten()
            .all(|batch| batch.again(workspace, compilation, stale));
        if ok && !self.stamps.is_empty() {
            sync_included_stamps(workspace, &mut self.stamps, &self.compilation.outline);
        }
        ok
    }

    fn reload(
        &mut self,
        workspace: &graph::Workspace,
        edited: Vec<u32>,
        stale_bodies: &[(u32, u32)],
        changed_paths: &[PathBuf],
    ) -> Result<bool> {
        self.run_checked = false;
        let pre = refresh::snapshot_edited(
            &self.compilation.outline,
            &mut self.compilation.names,
            &edited,
        );
        let Ok(reloaded) = self
            .compilation
            .outline
            .reload_edited(&edited, changed_paths)
        else {
            return Ok(false);
        };
        let Ok(Some((crates, dependents, scoped))) = refresh::resolve_with_dependents(
            &mut self.compilation,
            pre,
            reloaded,
            stale_bodies,
            changed_paths,
        ) else {
            return Ok(false);
        };
        let errors = self.compilation.outline.local_errors(workspace);
        if !errors.is_empty() {
            update_stamps(
                workspace,
                &mut self.stamps,
                &self.compilation.outline,
                &crates,
            );
            self.cases = None;
            self.binary = None;
            self.program = None;
            bail!("{}", errors.join("\n"));
        }
        if let Some(program) = self.program.take() {
            let store = program.into_store(&self.compilation.outline, Edited::Crates(&dependents));
            let main = main_item(workspace, &self.compilation.outline);
            self.program =
                main.and_then(|main| Program::new(&mut self.compilation, store, main).ok());
        }
        update_stamps(
            workspace,
            &mut self.stamps,
            &self.compilation.outline,
            &crates,
        );
        let batch_crates = if scoped.is_some() {
            &crates
        } else {
            &dependents
        };
        let compilation = &self.compilation;
        Ok([&mut self.cases, &mut self.binary]
            .into_iter()
            .flatten()
            .all(|batch| batch.reload(workspace, compilation, batch_crates, scoped.as_ref())))
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
            crate::diag::print_build_finished(false);
            mark_json_finished();
            bail!("{failed} functions failed to compile");
        }
        bin.write_aot(&self.compilation, path)?;
        let sel = crate::selection();
        if !sel.quiet && sel.message_format != crate::MessageFormat::Json {
            eprintln!(
                "built {} ({} functions reached, {:.1} ms)",
                path.display(),
                bin.reached.list.len(),
                started.elapsed().as_secs_f64() * 1e3
            );
        }
        crate::diag::print_build_finished(true);
        Ok(0)
    }

    /// Runs every case or `#[test]` whose functions all compiled and returns 0 when all pass, else 101.
    pub fn report(&self, started: Instant, built: Duration, filters: &[String]) -> Result<i32> {
        let Some(cases) = &self.cases else {
            bail!("the cases are not compiled")
        };
        if crate::with_cfg(|c| c.no_run) {
            let status = cases.check(&self.compilation.outline, built);
            crate::diag::print_build_finished(status == 0);
            return Ok(if status == 0 { 0 } else { 101 });
        }
        cases.report(&self.compilation.outline, started, built, filters)
    }

    /// Prints what failed to build in the binary's reach and how long building took, and returns 1 when anything failed.
    pub fn check(&self, built: Duration) -> Result<i32> {
        let Some(binary) = &self.binary else {
            bail!("the binary is not compiled")
        };
        let status = binary.check(&self.compilation.outline, built);
        crate::diag::print_build_finished(status == 0);
        Ok(status)
    }

    fn batch_diagnostics(&self, batch: Option<&Batch>) -> Option<String> {
        batch.and_then(|batch| batch.formatted_diagnostics(&self.compilation.outline))
    }

    /// Formatted diagnostics from the compiled binary batch, if any failed.
    pub(crate) fn binary_diagnostics(&self) -> Option<String> {
        self.batch_diagnostics(self.binary.as_ref())
    }

    /// Formatted diagnostics from the compiled test or case batch, if any failed.
    pub(crate) fn cases_diagnostics(&self) -> Option<String> {
        self.batch_diagnostics(self.cases.as_ref())
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
        let main = require_main(workspace, &self.compilation.outline)?;
        let target = Some(main.krate as usize);
        if self.program.is_some() {
            if !self.run_checked {
                let (_, cache) = cache_with_substrate(workspace, false);
                if let Err(code) =
                    check_and_print_member_items(workspace, &mut self.compilation, cache, target)?
                {
                    return Ok(code);
                }
                self.run_checked = true;
            }
            let Some(mut program) = self.program.take() else {
                bail!("program missing");
            };
            let status = program.run(&mut self.compilation, args);
            self.program = Some(program);
            return status;
        }
        let (_, cache) = cache_with_substrate(workspace, true);
        let cache =
            match check_and_print_member_items(workspace, &mut self.compilation, cache, target)? {
                Ok(cache) => cache,
                Err(code) => return Ok(code),
            };
        self.run_checked = true;
        let program = self
            .program
            .insert(Program::new(&mut self.compilation, cache, main)?);
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
        (true, &[]),
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
    let cache = match check_and_print_member_items(
        workspace,
        &mut compilation,
        cache,
        Some(main.krate as usize),
    )? {
        Ok(cache) => cache,
        Err(code) => {
            std::mem::forget(compilation);
            return Ok(code);
        }
    };
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

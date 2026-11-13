//! A workspace loaded once and kept: its `case_` functions and its binary's reach each built into one image, and its binary compiled on first call, each brought up to date after an edit.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use crate::graph::{self, Targets};
use crate::image::{Image, Program};
use crate::macros::Compilation;
use crate::names::ItemRef;
use crate::outline::{self, edit::Edit, tree::Profile};
use crate::sema::reach::{Instance, Reached, reach_again, reach_anew};
use crate::sema::{self, Cache, Edited};
use anyhow::{Context, Result, bail};
use rustc_hash::FxHashMap as HashMap;

use crate::cli::{Failures, case_items, error_kind, print_failures};

include!("../../../tests/case_inputs.rs");

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

#[derive(Clone, Copy, PartialEq, Eq)]
enum BatchKind {
    Cases,
    MainCheck,
    MainAot,
}

impl BatchKind {
    fn mode(self) -> crate::image::ImageMode {
        match self {
            BatchKind::Cases => crate::image::ImageMode::Jit,
            BatchKind::MainCheck => crate::image::ImageMode::Check,
            BatchKind::MainAot => crate::image::ImageMode::Aot,
        }
    }

    fn pick(self, workspace: &graph::Workspace, outline: &outline::Outline) -> Vec<ItemRef> {
        match self {
            BatchKind::Cases => case_items(workspace, outline),
            BatchKind::MainCheck | BatchKind::MainAot => main_items(workspace, outline),
        }
    }
}

/// Root functions inferred and compiled ahead of time into one image: how to pick them, the inference store, their names and indexes, what they reach, and why each function that did not compile failed.
struct Batch {
    kind: BatchKind,
    store: Option<Cache>,
    labels: Vec<String>,
    roots: Vec<usize>,
    reached: Reached,
    image: Image,
    broken: HashMap<usize, String>,
    reach_span: sema::census::Span,
    codegen_span: sema::census::Span,
}

pub(crate) use crate::graph::store::{FileStamp as Stamp, file_stamp as stamp};

/// The source files of every crate whose package is local, which an edit can change, the files `include!` read into them, and the files their `include_str!` and `include_bytes!` calls read.
fn stamps(workspace: &graph::Workspace, outline: &outline::Outline) -> Vec<Watch> {
    let local: Vec<usize> = (0..outline.roots.len()).filter(|&krate| outline.package(workspace, krate).local()).collect();
    let mut sources: Vec<PathBuf> = local.iter().flat_map(|&krate| outline.crates[krate].files.iter().map(|file| file.source.path.clone())).collect();
    sources.sort();
    sources.dedup();
    let included = local.iter().flat_map(|&krate| outline.included(krate)).map(|(path, by)| (path, Some(by)));
    sources.into_iter().map(|path| (path, None)).chain(included).map(|(path, by)| Watch { stamp: stamp(&path), path, by }).collect()
}

fn batch_roots(workspace: &graph::Workspace, outline: &outline::Outline, kind: BatchKind) -> (Vec<ItemRef>, Vec<Instance>) {
    let items = kind.pick(workspace, outline);
    let roots = items.iter().map(|item| (*item, sema::substs::Substs::EMPTY)).collect();
    (items, roots)
}

/// The `main` of the first member binary.
pub(crate) fn main_item(workspace: &graph::Workspace, outline: &outline::Outline) -> Option<ItemRef> {
    let krate = outline.members(workspace).find(|&krate| outline.roots[krate].role == outline::Role::Bin)?;
    outline.main(krate)
}

fn main_items(workspace: &graph::Workspace, outline: &outline::Outline) -> Vec<ItemRef> {
    main_item(workspace, outline).into_iter().collect()
}

fn recompile(image: &mut Image, broken: &mut HashMap<usize, String>, (cx, reached): (&mut sema::Context<'_>, &Reached), fresh: &[usize]) {
    let mut expanded = fresh.to_vec();
    if crate::profile() == outline::tree::Profile::Release && !fresh.is_empty() && fresh.len() < reached.list.len() {
        let mut dirty = vec![false; reached.list.len()];
        for &at in fresh {
            dirty[at] = true;
        }
        for _ in 0..3 {
            let added: Vec<usize> = (0..reached.list.len())
                .filter(|&u| !dirty[u] && reached.direct_calls(u).is_some_and(|callees| callees.iter().any(|&c| dirty[c])))
                .collect();
            if added.is_empty() {
                break;
            }
            for u in added {
                dirty[u] = true;
                expanded.push(u);
            }
        }
    }
    expanded.iter().for_each(|at| drop(broken.remove(at)));
    broken.extend(image.compile(cx, reached, &expanded));
}

type SubstrateSlot = (Option<(u64, u64)>, Option<std::sync::Arc<crate::substrate::Substrate>>);

fn open_batch_substrate(workspace: &graph::Workspace) -> SubstrateSlot {
    let key = crate::substrate::enabled().then(|| {
        let target = crate::host_target();
        (crate::substrate::toolchain_key(workspace, &target), crate::substrate::target_hash(&target))
    });
    let existing = key.and_then(|(toolchain_key, target_key)| match crate::substrate::Substrate::open_toolchain(toolchain_key) {
        Ok(sub) if sub.toolchain_hash() == toolchain_key && sub.target_hash() == target_key => Some(std::sync::Arc::new(sub)),
        _ => None,
    });
    (key, existing)
}

fn finish_batch(
    (kind, main): (BatchKind, Option<ItemRef>),
    (cases, reached): (&[ItemRef], Reached),
    drop_finals: bool,
    substrate: (Option<(u64, u64)>, &Option<std::sync::Arc<crate::substrate::Substrate>>),
    (mut cx, reach_clock): (sema::Context<'_>, sema::census::Clock),
) -> Result<Batch> {
    let mode = kind.mode();
    let reach_span = reach_clock.elapsed();
    let codegen_clock = sema::census::Clock::now();
    let pre_builder = export_pre_summaries(&mut cx, drop_finals, substrate.0);
    let valid: Vec<ItemRef> = match kind {
        BatchKind::Cases => cases.iter().copied().filter(|&item| sema::infer::is_case_sig(&mut cx, item)).collect(),
        BatchKind::MainCheck => cases.to_vec(),
        BatchKind::MainAot => Vec::new(),
    };
    let roots: Vec<usize> = match main {
        Some(main) => sema::reach::main_roots(&mut cx, main)?.into_iter().map(|inst| reached.index[&inst]).collect(),
        None => valid.iter().map(|item| reached.index[&(*item, sema::substs::Substs::EMPTY)]).collect(),
    };
    if drop_finals {
        cx.compact_for_oneshot_codegen();
    }
    let (image, broken) = if mode == crate::image::ImageMode::Aot && roots.len() == 2 {
        Image::build_aot(&mut cx, &reached, [roots[0], roots[1]])?
    } else {
        Image::build(&mut cx, &reached, mode)?
    };
    let codegen_span = codegen_clock.elapsed();
    save_batch_substrate(&cx, &image, drop_finals, substrate, pre_builder);
    let labels = valid.iter().map(|item| cx.name(*item).to_owned()).collect();
    Ok(Batch { kind, store: Some(cx.release().0), labels, roots, reached, image, broken, reach_span, codegen_span })
}

fn export_pre_summaries(cx: &mut sema::Context<'_>, drop_finals: bool, substrate_key: Option<(u64, u64)>) -> Option<crate::substrate::SubstrateBuilder> {
    if !drop_finals || !cx.store().has_new_sysroot_summaries() {
        return None;
    }
    let (key, target_key) = substrate_key?;
    let mut builder = crate::substrate::Substrate::builder().toolchain_hash(key).target_hash(target_key);
    if let Err(error) = cx.store().export_sysroot_summaries(cx.outline, &mut builder) {
        eprintln!("haste: cannot export sysroot summaries: {error:#}");
        return None;
    }
    Some(builder)
}

fn save_batch_substrate(
    cx: &sema::Context<'_>,
    image: &Image,
    drop_finals: bool,
    (substrate_key, existing_sub): (Option<(u64, u64)>, &Option<std::sync::Arc<crate::substrate::Substrate>>),
    pre_builder: Option<crate::substrate::SubstrateBuilder>,
) {
    let Some((key, target_key)) = substrate_key else { return };
    if !image.has_new_frozen_stencils() && pre_builder.is_none() {
        return;
    }
    let mut builder = pre_builder.unwrap_or_else(|| crate::substrate::Substrate::builder().toolchain_hash(key).target_hash(target_key));
    let save_res = (|| -> Result<()> {
        if let Some(ex) = existing_sub {
            builder.merge_from(ex)?;
        }
        if !drop_finals {
            cx.store().export_sysroot_summaries(cx.outline, &mut builder)?;
        }
        image.export_sysroot_stencils(&mut builder)?;
        let sub = builder.build()?;
        if sub.summary_count() > 0 || sub.stencil_count() > 0 {
            sub.save_toolchain()?;
        }
        Ok(())
    })();
    if let Err(error) = save_res {
        eprintln!("haste: cannot save toolchain substrate: {error:#}");
    }
}

impl Batch {
    /// Infers and compiles every root `kind` finds in `workspace` from nothing.
    fn new(workspace: &graph::Workspace, compilation: &mut Compilation, kind: BatchKind, drop_finals: bool) -> Result<Batch> {
        let mode = kind.mode();
        let reach_clock = sema::census::Clock::now();
        let main = (kind == BatchKind::MainAot).then(|| main_item(workspace, &compilation.outline).context("no member binary has a `main`")).transpose()?;
        let (cases, root_insts) = if main.is_some() { (Vec::new(), Vec::new()) } else { batch_roots(workspace, &compilation.outline, kind) };
        let (substrate_key, existing_sub) = open_batch_substrate(workspace);
        let mut cache = Cache::default();
        cache.set_warm_sysroot_bodies(mode != crate::image::ImageMode::Check);
        if let Some(sub) = &existing_sub {
            cache.set_substrate(Some(std::sync::Arc::clone(sub)));
        }
        crate::macros::reach_resolved_with_cache(
            compilation,
            cache,
            |cx| match kind {
                BatchKind::MainAot => sema::reach::main_roots(cx, main.expect("main checked")).map_or_else(|_| vec![(main.expect("main checked"), sema::substs::Substs::EMPTY)], Into::into),
                BatchKind::Cases => root_insts.iter().copied().filter(|&(item, _)| sema::infer::is_case_sig(cx, item)).collect(),
                BatchKind::MainCheck => root_insts.clone(),
            },
            |cx, reached| finish_batch((kind, main), (&cases, reached), drop_finals, (substrate_key, &existing_sub), (cx, reach_clock)),
        )
    }

    fn again(&mut self, compilation: &Compilation, stale: &[(u32, u32)]) -> bool {
        let Some(mut store) = self.store.take() else {
            return false;
        };
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
        let Some(mut store) = self.store.take() else {
            return false;
        };
        let touched = store.invalidate(&compilation.outline, self.reached.list.iter().map(|(instance, _)| instance), Edited::Crates(crates));
        let (mut cases, mut roots) = batch_roots(workspace, &compilation.outline, self.kind);
        let mut cx = sema::Context::new(&compilation.outline, &compilation.names, store);
        if self.kind == BatchKind::Cases {
            cases.retain(|&item| sema::infer::is_case_sig(&mut cx, item));
            roots = cases.iter().map(|&item| (item, sema::substs::Substs::EMPTY)).collect();
        }
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

    /// Why each function the roots reach did not type-check or compile, at the file and line of its item and with its type arguments, keyed by the kind of error.
    fn failures(&self, cx: &sema::Context<'_>) -> Failures {
        let mut failures = Failures::new();
        for at in self.reached.closure(&self.roots) {
            if let Some(error) = self.reached.list[at].1.as_ref().err().or_else(|| self.broken.get(&at)) {
                let (item, substs) = self.reached.list[at].0;
                let (src, entry) = cx.item(item);
                let substs: Vec<String> = cx.types.substs(substs).iter().map(|&ty| cx.type_text(ty, &|ty| ty)).collect();
                failures.entry(error_kind(error)).or_default().push(format!("{}:{} {error} [{}]", src.path.display(), src.line(entry.start), substs.join(", ")));
            }
        }
        failures
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
        let statuses = self.image.run_cases(&runnable, &CASE_INPUTS)?;
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
        let compilation = crate::macros::load_lazy(workspace, targets)?;
        let stamps = stamps(workspace, &compilation.outline);
        let outline_span = outline_clock.elapsed();
        Ok(Build { compilation, cases: None, binary: None, program: None, stamps, outline_span })
    }

    /// Infers and compiles the `case_` functions unless that is done.
    pub fn compile_cases(&mut self, workspace: &graph::Workspace) -> Result<()> {
        if self.cases.is_none() {
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
            self.cases = Some(Batch::new(workspace, &mut self.compilation, BatchKind::Cases, false)?);
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

    fn compile_binary_with(&mut self, workspace: &graph::Workspace, drop_finals: bool) -> Result<()> {
        if main_item(workspace, &self.compilation.outline).is_none() {
            bail!("no member binary has a `main`");
        }
        if self.binary.is_none() {
            self.binary = Some(Batch::new(workspace, &mut self.compilation, BatchKind::MainCheck, drop_finals)?);
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
            let Ok(reloaded) = self.compilation.outline.reload(krate as usize) else {
                return false;
            };
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

    /// Compiles `main` and `std::rt::lang_start` into an AOT `Image` and writes a standalone executable to `path`.
    pub(crate) fn emit_executable(&mut self, workspace: &graph::Workspace, path: &std::path::Path) -> Result<i32> {
        let started = Instant::now();
        if self.binary.is_none() {
            self.binary = Some(Batch::new(workspace, &mut self.compilation, BatchKind::MainAot, false)?);
        }
        let bin = self.binary.as_ref().context("the binary is not compiled")?;
        if !bin.broken.is_empty() {
            for (idx, err) in &bin.broken {
                eprintln!("broken [{idx}]: {err}");
            }
            bail!("{} functions failed to compile", bin.broken.len());
        }
        let target = crate::host_target();
        bin.image.to_aot(&target, (bin.roots[0], bin.roots[1]))?.write(path)?;
        eprintln!("built {} ({} functions reached, {:.1} ms)", path.display(), bin.reached.list.len(), started.elapsed().as_secs_f64() * 1e3);
        Ok(0)
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

    /// Why each function the binary's `main` reaches did not type-check or compile, at the file and line of its item, keyed by the kind of error.
    pub fn failures(&mut self) -> Result<Failures> {
        let Some(binary) = &mut self.binary else { bail!("the binary is not compiled") };
        let Some(store) = binary.store.take() else { bail!("the binary has no inference store") };
        let cx = sema::Context::new(&self.compilation.outline, &self.compilation.names, store);
        let failures = binary.failures(&cx);
        binary.store = Some(cx.release().0);
        Ok(failures)
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

/// Runs the binary's `main` with `args` as argv, compiling each function on its first call, and, when prepared code is switched on, under the release profile calling prepped code where it can, racing a prep of what changed since the library was built.
pub fn run(workspace: &graph::Workspace, targets: &Targets, args: &[String]) -> Result<i32> {
    let mut ahead = if crate::profile() == Profile::Release && crate::prep::prepared() { crate::prep::ahead(workspace, targets) } else { None };
    let status = run_loaded(workspace, targets, args, ahead.as_mut());
    if !matches!(status, Ok(0))
        && let Some(said) = ahead.as_mut().and_then(crate::prep::Ahead::rejected)
    {
        eprintln!("haste: rustc rejects this program too:\n{said}");
    }
    status
}

fn run_loaded(workspace: &graph::Workspace, targets: &Targets, args: &[String], ahead: Option<&mut crate::prep::Ahead>) -> Result<i32> {
    let mut compilation = crate::macros::load_eager(workspace, targets, &[])?;
    let Some(main) = main_item(workspace, &compilation.outline) else { bail!("no member binary has a `main`") };
    if crate::profile() == Profile::Release && crate::prep::prepared() {
        bind_prepped(workspace, &mut compilation, args, ahead);
    }
    let status = crate::image::run_on_demand(&mut compilation, (sema::Cache::default(), main), args);
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

/// Binds this process to what of the prepped library of the build still matches the sources, once the prep `ahead` races lands or its wait runs out, and starts a prep in the background when the library is missing or older than the sources; a build that cannot bind it compiles everything.
fn bind_prepped(workspace: &graph::Workspace, compilation: &mut Compilation, args: &[String], mut ahead: Option<&mut crate::prep::Ahead>) {
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
    if let Err(error) = manifest.and_then(|manifest| manifest.map_or(Ok(()), |manifest| crate::prep::install(manifest, args))) {
        eprintln!("haste: running without prepped code: {error:#}");
    }
}

/// Whether crate `krate` is a procedural macro crate, whose bodies decide the code of the crates that call its macros.
fn expands(workspace: &graph::Workspace, outline: &outline::Outline, krate: u32) -> bool {
    outline.package(workspace, krate as usize).manifest.proc_macro
}

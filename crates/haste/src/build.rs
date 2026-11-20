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

/// The path of each file of `stamps`.
fn paths(stamps: &[Watch]) -> Vec<PathBuf> {
    stamps.iter().map(|watch| watch.path.clone()).collect()
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

fn recompile(image: &mut Image, broken: &mut HashMap<usize, String>, (cx, reached): (&mut sema::Context<'_>, &Reached), fresh: &[usize], roots: &[usize]) {
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
    if let &[start, main] = roots {
        let mut tried: rustc_hash::FxHashSet<usize> = expanded.into_iter().collect();
        loop {
            let missing: Vec<usize> = image.missing_aot_targets([start, main], reached).into_iter().filter(|idx| !broken.contains_key(idx) && tried.insert(*idx)).collect();
            if missing.is_empty() {
                break;
            }
            broken.extend(image.compile(cx, reached, &missing));
        }
    }
}

struct SubstrateSlot {
    toolchain_key: u64,
    deps_key: Option<u64>,
    workspace_key: Option<u64>,
    target_key: u64,
    toolchain: Option<std::sync::Arc<crate::substrate::Substrate>>,
    deps: Option<std::sync::Arc<crate::substrate::Substrate>>,
    workspace: Option<std::sync::Arc<crate::substrate::Substrate>>,
}

struct PreBuilders {
    toolchain: crate::substrate::SubstrateBuilder,
    deps: Option<crate::substrate::SubstrateBuilder>,
    workspace: Option<crate::substrate::SubstrateBuilder>,
}

impl PreBuilders {
    fn shrink_blobs(&mut self) {
        self.toolchain.shrink_blob();
        if let Some(deps) = &mut self.deps {
            deps.shrink_blob();
        }
        if let Some(ws) = &mut self.workspace {
            ws.shrink_blob();
        }
    }
}

fn open_batch_substrate(workspace: &graph::Workspace) -> Option<SubstrateSlot> {
    if !crate::substrate::enabled() {
        return None;
    }
    let target = crate::host_target();
    let toolchain_key = crate::substrate::toolchain_key(workspace, &target);
    let target_key = crate::substrate::target_hash(&target);
    let deps_key = crate::substrate::deps_key(workspace, &target, toolchain_key);
    let workspace_key = crate::substrate::workspace_key(workspace, &target, toolchain_key);
    let toolchain = match crate::substrate::Substrate::open_toolchain(toolchain_key) {
        Ok(sub) if sub.toolchain_hash() == toolchain_key && sub.target_hash() == target_key => Some(std::sync::Arc::new(sub)),
        _ => None,
    };
    let deps = deps_key.and_then(|dk| match crate::substrate::Substrate::open_deps(dk, target_key) {
        Ok(sub) => Some(std::sync::Arc::new(match &toolchain {
            Some(tc) => sub.with_parent(std::sync::Arc::clone(tc)),
            None => sub,
        })),
        _ => None,
    });
    let base = deps.as_ref().or(toolchain.as_ref());
    let workspace_sub = workspace_key.and_then(|wk| match crate::substrate::Substrate::open_workspace(wk, target_key) {
        Ok(sub) => Some(std::sync::Arc::new(match base {
            Some(parent) => sub.with_parent(std::sync::Arc::clone(parent)),
            None => sub,
        })),
        _ => None,
    });
    Some(SubstrateSlot { toolchain_key, deps_key, workspace_key, target_key, toolchain, deps, workspace: workspace_sub })
}

fn finish_batch(
    (kind, main): (BatchKind, Option<ItemRef>),
    (cases, mut reached): (&[ItemRef], Reached),
    drop_finals: bool,
    substrate: Option<SubstrateSlot>,
    (mut cx, reach_clock): (sema::Context<'_>, sema::census::Clock),
) -> Result<Batch> {
    let mode = kind.mode();
    let reach_span = reach_clock.elapsed();
    let codegen_clock = sema::census::Clock::now();
    let pre_builder = export_pre_summaries(&mut cx, drop_finals, substrate.as_ref());
    let valid: Vec<ItemRef> = match kind {
        BatchKind::Cases => cases.iter().copied().filter(|&item| sema::infer::is_case_sig(&mut cx, item)).collect(),
        BatchKind::MainCheck => cases.to_vec(),
        BatchKind::MainAot => Vec::new(),
    };
    let root_instances: Vec<sema::reach::Instance> = match main {
        Some(main) => sema::reach::main_roots(&mut cx, main)?.to_vec(),
        None => valid.iter().map(|item| (*item, sema::substs::Substs::EMPTY)).collect(),
    };
    reached.canonicalize(&cx, &root_instances);
    let roots: Vec<usize> = root_instances.iter().map(|inst| reached.index[inst]).collect();
    if drop_finals {
        cx.compact_for_oneshot_codegen();
    }
    let (mut image, broken) = if mode == crate::image::ImageMode::Aot && roots.len() == 2 {
        Image::build_aot(&mut cx, &reached, [roots[0], roots[1]])?
    } else {
        Image::build(&mut cx, &reached, mode)?
    };
    let codegen_span = codegen_clock.elapsed();
    let labels = valid.iter().map(|item| cx.name(*item).to_owned()).collect();
    let outline = cx.outline;
    let mut store = cx.release().0;
    save_batch_substrate((outline, &mut store), &mut image, (drop_finals, kind), substrate, pre_builder);
    Ok(Batch { kind, store: Some(store), labels, roots, reached, image, broken, reach_span, codegen_span })
}

fn make_builders(slot: &SubstrateSlot) -> PreBuilders {
    let toolchain = crate::substrate::Substrate::builder().toolchain_hash(slot.toolchain_key).target_hash(slot.target_key);
    let deps = slot.deps_key.map(|dk| crate::substrate::Substrate::builder().toolchain_hash(dk).target_hash(slot.target_key));
    let workspace = slot.workspace_key.map(|wk| crate::substrate::Substrate::builder().toolchain_hash(wk).target_hash(slot.target_key));
    PreBuilders { toolchain, deps, workspace }
}

fn export_pre_summaries(cx: &mut sema::Context<'_>, drop_finals: bool, slot: Option<&SubstrateSlot>) -> Option<PreBuilders> {
    if !drop_finals || !cx.store().has_new_sysroot_summaries() {
        return None;
    }
    let mut builders = make_builders(slot?);
    if let Err(error) = cx.store().export_sysroot_summaries(cx.outline, Some(&mut builders.toolchain), builders.deps.as_mut()) {
        eprintln!("haste: cannot export sysroot summaries: {error:#}");
        return None;
    }
    builders.shrink_blobs();
    Some(builders)
}

fn save_substrate_layer(
    builder: crate::substrate::SubstrateBuilder,
    (workspace, kind): (u64, BatchKind),
    existing: Option<std::sync::Arc<crate::substrate::Substrate>>,
    save: fn(crate::substrate::SubstrateBuilder) -> Result<Option<PathBuf>>,
) -> Result<()> {
    if builder.is_empty() {
        return Ok(());
    }
    let mut builder = builder.consumer(workspace, kind as u32);
    if let Some(ex) = existing {
        builder.merge_from(&ex)?;
        drop(ex);
    }
    save(builder)?;
    Ok(())
}

fn finish_oneshot_compact(drop_finals: bool, store: &mut Cache, image: &mut Image) {
    if drop_finals {
        store.set_substrate(None);
        image.compact_for_oneshot_aot();
    }
}

fn export_and_save_layers(
    (outline, store): (&outline::Outline, &mut Cache),
    image: &mut Image,
    (drop_finals, kind): (bool, BatchKind),
    mut slot: SubstrateSlot,
    mut builders: PreBuilders,
) -> Result<()> {
    if !drop_finals {
        store.export_sysroot_summaries(outline, Some(&mut builders.toolchain), builders.deps.as_mut())?;
    }
    if image.has_new_frozen_stencils() {
        image.export_sysroot_stencils(Some(&mut builders.toolchain), builders.deps.as_mut())?;
    }
    image.export_inline_meta(Some(&mut builders.toolchain), builders.deps.as_mut())?;
    finish_oneshot_compact(drop_finals, store, image);
    let workspace = slot.workspace_key.or(slot.deps_key).unwrap_or(slot.toolchain_key);
    if let Some(ws_builder) = builders.workspace {
        save_substrate_layer(ws_builder, (workspace, kind), slot.workspace.take(), crate::substrate::SubstrateBuilder::save_workspace)?;
    }
    if let Some(deps_builder) = builders.deps {
        save_substrate_layer(deps_builder, (workspace, kind), slot.deps.take(), crate::substrate::SubstrateBuilder::save_deps)?;
    }
    save_substrate_layer(builders.toolchain, (workspace, kind), slot.toolchain.take(), crate::substrate::SubstrateBuilder::save_toolchain)
}

fn save_batch_substrate(
    (outline, store): (&outline::Outline, &mut Cache),
    image: &mut Image,
    (drop_finals, kind): (bool, BatchKind),
    slot: Option<SubstrateSlot>,
    pre_builder: Option<PreBuilders>,
) {
    let has_new = image.has_new_frozen_stencils() || image.has_new_inline_meta() || pre_builder.is_some() || (!drop_finals && store.has_new_sysroot_summaries());
    let Some(slot) = slot.filter(|_| has_new) else {
        finish_oneshot_compact(drop_finals, store, image);
        return;
    };
    let builders = pre_builder.unwrap_or_else(|| make_builders(&slot));
    if let Err(error) = export_and_save_layers((outline, store), image, (drop_finals, kind), slot, builders) {
        eprintln!("haste: cannot save substrate: {error:#}");
    }
}

impl Batch {
    /// Infers and compiles every root `kind` finds in `workspace` from nothing.
    fn new(workspace: &graph::Workspace, compilation: &mut Compilation, kind: BatchKind, drop_finals: bool) -> Result<Batch> {
        let main = (kind == BatchKind::MainAot).then(|| main_item(workspace, &compilation.outline).context("no member binary has a `main`")).transpose()?;
        Self::rooted(workspace, compilation, (kind, main), drop_finals)
    }

    /// Infers and compiles from nothing the roots `kind` finds in `workspace`, or the program whose `main` is `main`.
    fn rooted(workspace: &graph::Workspace, compilation: &mut Compilation, (kind, main): (BatchKind, Option<ItemRef>), drop_finals: bool) -> Result<Batch> {
        let mode = kind.mode();
        let reach_clock = sema::census::Clock::now();
        let (cases, root_insts) = if main.is_some() { (Vec::new(), Vec::new()) } else { batch_roots(workspace, &compilation.outline, kind) };
        let mut substrate = open_batch_substrate(workspace);
        let mut cache = Cache::default();
        cache.set_warm_sysroot_bodies(mode != crate::image::ImageMode::Check);
        if let Some(slot) = &substrate
            && let Some(sub) = slot.workspace.as_ref().or(slot.deps.as_ref()).or(slot.toolchain.as_ref())
        {
            cache.set_substrate(Some(std::sync::Arc::clone(sub)));
        }
        let batch = crate::macros::reach_resolved_with_cache(
            compilation,
            cache,
            |cx| match (kind, main) {
                (BatchKind::MainAot, Some(main_ref)) => sema::reach::main_roots(cx, main_ref).map_or_else(|_| vec![(main_ref, sema::substs::Substs::EMPTY)], Into::into),
                (BatchKind::Cases, _) => root_insts.iter().copied().filter(|&(item, _)| sema::infer::is_case_sig(cx, item)).collect(),
                _ => root_insts.clone(),
            },
            |cx, reached| finish_batch((kind, main), (&cases, reached), drop_finals, substrate.take(), (cx, reach_clock)),
        )?;
        crate::substrate::save_fixed_files(workspace, crate::host_target().triple, &compilation.outline);
        Ok(batch)
    }

    /// Writes this ahead-of-time batch as a standalone executable to `path`, unless a function it reaches failed to compile.
    fn write_aot(&mut self, compilation: &Compilation, path: &std::path::Path) -> Result<()> {
        if !self.broken.is_empty() {
            bail!("{} functions failed to compile", self.broken.len());
        }
        let store = self.store.take().context("the binary's inference store is in use")?;
        let mut cx = sema::Context::new(&compilation.outline, &compilation.names, store);
        let exe = self.image.to_aot(&crate::host_target(), (self.roots[0], self.roots[1]), &mut cx, &self.reached);
        self.store = Some(cx.release().0);
        exe?.write(path)
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
        recompile(&mut self.image, &mut self.broken, (&mut cx, &self.reached), &fresh, &self.roots);
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
        match self.kind {
            BatchKind::Cases => {
                cases.retain(|&item| sema::infer::is_case_sig(&mut cx, item));
                roots = cases.iter().map(|&item| (item, sema::substs::Substs::EMPTY)).collect();
            }
            BatchKind::MainAot => {
                let Some(&main) = cases.first() else { return false };
                let Ok(pair) = sema::reach::main_roots(&mut cx, main) else { return false };
                cases.clear();
                roots = pair.to_vec();
            }
            BatchKind::MainCheck => {}
        }
        let fresh = reach_anew(&mut cx, &mut self.reached, &touched, &roots);
        if !cx.resolved() {
            return false;
        }
        self.labels = cases.iter().map(|item| cx.name(*item).to_owned()).collect();
        self.roots = roots.iter().map(|root| self.reached.index[root]).collect();
        recompile(&mut self.image, &mut self.broken, (&mut cx, &self.reached), &fresh, &self.roots);
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
        let Ok((stale, crates)) = edits(workspace, &mut self.stamps, &mut self.compilation.outline) else {
            return self.rebuild(workspace, targets);
        };
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
    pub(crate) fn emit_executable(&mut self, workspace: &graph::Workspace, path: &std::path::Path, drop_finals: bool) -> Result<i32> {
        let started = Instant::now();
        if self.binary.as_ref().is_none_or(|bin| bin.kind != BatchKind::MainAot) {
            self.binary = Some(Batch::new(workspace, &mut self.compilation, BatchKind::MainAot, drop_finals)?);
        }
        if !self.binary.as_ref().context("the binary is not compiled")?.broken.is_empty() {
            crate::cli::failure_lines(&self.failures()?).iter().for_each(|line| eprintln!("{line}"));
        }
        let bin = self.binary.as_mut().context("the binary is not compiled")?;
        bin.write_aot(&self.compilation, path)?;
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

/// Compiles the program whose `main` is `main` ahead of time and writes it as a standalone executable to `path`.
pub(crate) fn emit_main(workspace: &graph::Workspace, compilation: &mut Compilation, main: ItemRef, path: &std::path::Path) -> Result<()> {
    Batch::rooted(workspace, compilation, (BatchKind::MainAot, Some(main)), true)?.write_aot(compilation, path)
}

/// What changed in the files of `stamps` since their last stamps: the `(crate, file)` slots whose function bodies changed, which `outline` now holds, and the crates to read again; fails when a file does not read.
fn edits(workspace: &graph::Workspace, stamps: &mut [Watch], outline: &mut outline::Outline) -> Result<(Vec<(u32, u32)>, Vec<u32>)> {
    let (mut stale, mut crates) = (Vec::new(), Vec::new());
    for watch in stamps {
        let now = stamp(&watch.path);
        if now == watch.stamp {
            continue;
        }
        watch.stamp = now;
        let edit = match &watch.by {
            Some(by) => outline.reexpand(by)?,
            None => outline.reread(&watch.path)?,
        };
        match edit {
            Edit::Same => {}
            Edit::Bodies(slots) if slots.iter().any(|&(krate, _)| expands(workspace, outline, krate)) => crates.extend(slots.iter().map(|&(krate, _)| krate)),
            Edit::Bodies(slots) => stale.extend(slots),
            Edit::Items(edited) => crates.extend(edited),
        }
    }
    Ok((stale, crates))
}

/// The `(crate, file)` slots whose function bodies changed in the files of `stamps`, which `outline` now holds, or why the edit needs a restart.
fn body_edits(workspace: &graph::Workspace, stamps: &mut [Watch], outline: &mut outline::Outline) -> Result<Vec<(u32, u32)>, String> {
    match edits(workspace, stamps, outline) {
        Ok((stale, crates)) if crates.is_empty() => Ok(stale),
        Ok((_, mut crates)) => {
            crates.sort_unstable();
            crates.dedup();
            let names: Vec<&str> = crates.iter().map(|&krate| outline.roots[krate as usize].name.as_str()).collect();
            Err(format!("more than function bodies changed in {}", names.join(", ")))
        }
        Err(error) => Err(format!("{error:#}")),
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
    let status = match crate::image::watch::enabled() {
        true => run_watched(workspace, &mut compilation, main, args),
        false => crate::image::run_on_demand(&mut compilation, (sema::Cache::default(), main), args),
    };
    std::mem::forget(compilation);
    status
}

/// Runs the binary's `main` with `args` as argv like `run_on_demand`, while it watches the local source files and swaps in the function bodies an edit changes.
fn run_watched(workspace: &graph::Workspace, compilation: &mut Compilation, main: ItemRef, args: &[String]) -> Result<i32> {
    let mut watched = stamps(workspace, &compilation.outline);
    let files = paths(&watched);
    let mut reread = |compilation: &mut Compilation| body_edits(workspace, &mut watched, &mut compilation.outline);
    let mut program = Program::new(compilation, Cache::default(), main)?;
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

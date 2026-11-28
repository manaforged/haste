use anyhow::{Context, Result, bail};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

#[cfg(feature = "diff-harness")]
use crate::cli::case_items;
use crate::cli::{TestItem, test_items};
use crate::graph;
use crate::image::{Image, ImageMode};
use crate::macros::Compilation;
use crate::names::ItemRef;
use crate::outline;
use crate::sema::infer::is_concrete_fn;
use crate::sema::reach::{Instance, Reached, check_poly_items, reach_again, reach_anew_in};
use crate::sema::substs::Substs;
use crate::sema::{self, Cache, Edited};

use super::substrate::{
    SubstrateSlot, cache_with_substrate, export_pre_summaries, load_local_check_crates,
    prepare_local_check, save_batch_substrate,
};
use super::test_args::{
    check_fn_items, check_target, map_tests, select_test_roots, test_label, validate_test_sigs,
};
use super::{main_item, require_main};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum BatchKind {
    #[cfg(feature = "diff-harness")]
    Cases,
    Tests,
    LibCheck,
    MainCheck,
    MainAot,
}

impl BatchKind {
    fn forces_unwind(self) -> bool {
        #[cfg(feature = "diff-harness")]
        if self == BatchKind::Cases {
            return true;
        }
        self == BatchKind::Tests
    }

    fn mode(self) -> ImageMode {
        match self {
            BatchKind::MainAot => ImageMode::Aot,
            BatchKind::LibCheck | BatchKind::MainCheck => ImageMode::Check,
            _ => ImageMode::Jit,
        }
    }

    fn pick(self, workspace: &graph::Workspace, outline: &outline::Outline) -> Vec<ItemRef> {
        match self {
            #[cfg(feature = "diff-harness")]
            BatchKind::Cases => case_items(workspace, outline),
            BatchKind::Tests => test_items(workspace, outline)
                .into_iter()
                .map(|t| t.item)
                .collect(),
            BatchKind::LibCheck => check_fn_items(workspace, outline, None),
            BatchKind::MainCheck => {
                check_fn_items(workspace, outline, check_target(workspace, outline, true))
            }
            BatchKind::MainAot => main_items(workspace, outline),
        }
    }
}

pub(super) struct Batch {
    pub(super) kind: BatchKind,
    pub(super) store: Option<Cache>,
    pub(super) labels: Vec<String>,
    pub(super) roots: Vec<usize>,
    pub(super) tests: Vec<(TestItem, Option<usize>)>,
    pub(super) reached: Reached,
    pub(super) image: Image,
    pub(super) broken: HashMap<usize, String>,
    pub(super) poly_failures: Vec<(ItemRef, String)>,
    pub(super) member_errors: Vec<String>,
    pub(super) reach_span: sema::census::Span,
    pub(super) codegen_span: sema::census::Span,
}

/// Type-checks and borrow-checks every concrete and polymorphic function in the target member crate(s) without emitting code.
pub(super) fn check_member_items(
    workspace: &graph::Workspace,
    compilation: &mut Compilation,
    cache: Cache,
    target: Option<usize>,
) -> Result<(Cache, Vec<String>)> {
    check_member_items_for(workspace, compilation, cache, target, None)
}

fn item_roots(items: &[ItemRef]) -> Vec<Instance> {
    items.iter().map(|&item| (item, Substs::EMPTY)).collect()
}

fn concrete_roots(cx: &mut sema::Context<'_>, items: &[ItemRef]) -> Vec<Instance> {
    items
        .iter()
        .copied()
        .filter(|&item| is_concrete_fn(cx, item))
        .map(|item| (item, Substs::EMPTY))
        .collect()
}

fn save_fixed(workspace: &graph::Workspace, outline: &outline::Outline) {
    crate::substrate::save_fixed_files(workspace, crate::host_target().triple, outline);
}

fn test_indices(starts: &[Option<Instance>], reached: &Reached) -> Vec<Option<usize>> {
    starts
        .iter()
        .map(|opt| opt.and_then(|inst| reached.index.get(&inst).copied()))
        .collect()
}

fn check_member_items_for(
    workspace: &graph::Workspace,
    compilation: &mut Compilation,
    mut cache: Cache,
    target: Option<usize>,
    only_items: Option<Vec<ItemRef>>,
) -> Result<(Cache, Vec<String>)> {
    let prev_warm = cache.warm_sysroot_bodies();
    cache.set_warm_sysroot_bodies(false);
    super::load_members(workspace, compilation)?;
    let items =
        only_items.unwrap_or_else(|| check_fn_items(workspace, &compilation.outline, target));
    let (store, mut errs) = crate::macros::reach_resolved_with_cache(
        compilation,
        cache,
        |cx| concrete_roots(cx, &items),
        |cx, reached| {
            let errs: Vec<_> = reached
                .list
                .iter()
                .filter_map(|((item, _), res)| res.as_ref().err().map(|err| (*item, err.clone())))
                .collect();
            Ok((cx.release().0, errs))
        },
    )?;
    let (mut store, poly) = check_poly_resolved(compilation, store, &items)?;
    store.set_warm_sysroot_bodies(prev_warm);
    errs.extend(poly);
    let (mut out, mut seen) = (Vec::new(), HashSet::default());
    for (item, err) in errs {
        let msg = Batch::format_entry(Some(&compilation.outline), item, None, &err);
        if seen.insert(msg.clone()) {
            out.push(msg);
        }
    }
    Ok((store, out))
}

fn attach_warm_cache(workspace: &graph::Workspace, cache: &mut Cache) {
    if !crate::substrate::enabled() {
        return;
    }
    let target = crate::host_target();
    let tc_key = crate::substrate::toolchain_key(workspace, &target);
    let key = sema::summary::WarmKey {
        toolchain_key: tc_key,
        deps_key: crate::substrate::deps_key(workspace, &target, tc_key),
        target_key: crate::substrate::target_hash(&target),
        workspace_key: crate::substrate::workspace_key(workspace, &target, tc_key),
        release: crate::profile() == outline::tree::Profile::Release,
    };
    let backing = crate::substrate::Backing::open(&key.path(), sema::summary::WARM_MIN_LEN).ok();
    cache.warm_state().set(key, backing);
}

fn batch_labels(cx: &mut sema::Context<'_>, kind: BatchKind, items: &[ItemRef]) -> Vec<String> {
    if matches!(
        kind,
        BatchKind::LibCheck | BatchKind::MainCheck | BatchKind::MainAot
    ) {
        return Vec::new();
    }
    items
        .iter()
        .map(|&item| match kind {
            BatchKind::Tests => test_label(cx, item),
            _ => cx.name(item).to_owned(),
        })
        .collect()
}

fn main_items(workspace: &graph::Workspace, outline: &outline::Outline) -> Vec<ItemRef> {
    let mut items: Vec<ItemRef> = main_item(workspace, outline).into_iter().collect();
    for krate in outline.members(workspace) {
        let tree = &outline.crates[krate];
        for (file_idx, file) in tree.files.iter().enumerate() {
            for (item_idx, entry) in file.items.items.iter().enumerate() {
                if entry.kind != outline::item::ItemKind::Fn || entry.attrs.is_empty() {
                    continue;
                }
                let exported = tree
                    .env
                    .find_attr(&entry.attrs, outline::tree::Truth::Yes, &|attr| {
                        matches!(
                            *attr,
                            outline::item::Attr::Other(at)
                                if matches!(file.source.text(at), "no_mangle" | "export_name")
                        )
                        .then_some(())
                    })
                    .is_some();
                if exported {
                    let item = ItemRef {
                        krate: krate as u32,
                        file: file_idx as u32,
                        item: item_idx as u32,
                    };
                    if !items.contains(&item) {
                        items.push(item);
                    }
                }
            }
        }
    }
    items
}

fn check_poly_resolved(
    compilation: &mut Compilation,
    store: Cache,
    items: &[ItemRef],
) -> Result<(Cache, Vec<(ItemRef, String)>)> {
    let mut pending = items.to_vec();
    crate::macros::resolve_sema(compilation, (store, crate::macros::load), |mut cx, last| {
        let failures = check_poly_items(&mut cx, &pending);
        let settled = cx.resolved();
        let released = cx.release();
        if settled || last || failures.is_empty() {
            Ok(Ok((released.0, failures)))
        } else {
            pending = failures.into_iter().map(|(item, _)| item).collect();
            Ok(Err(released))
        }
    })
}

fn recompile(
    image: &mut Image,
    broken: &mut HashMap<usize, String>,
    (cx, reached): (&mut sema::Context<'_>, &Reached),
    fresh: &[usize],
    roots: &[usize],
    is_check: bool,
) {
    let mut expanded = fresh.to_vec();
    if crate::profile() == outline::tree::Profile::Release
        && !fresh.is_empty()
        && fresh.len() < reached.list.len()
    {
        let mut dirty = vec![false; reached.list.len()];
        for &at in fresh {
            dirty[at] = true;
        }
        for _ in 0..3 {
            let added: Vec<usize> = (0..reached.list.len())
                .filter(|&u| {
                    !dirty[u]
                        && reached
                            .direct_calls(u)
                            .is_some_and(|callees| callees.iter().any(|&c| dirty[c]))
                })
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
    if is_check {
        expanded.retain(|&idx| !cx.outline.is_fixed(reached.list[idx].0.0.krate));
    }
    expanded.iter().for_each(|at| drop(broken.remove(at)));
    broken.extend(image.compile(cx, reached, &expanded));
    if let &[start, main] = roots {
        let mut tried: HashSet<usize> = expanded.into_iter().collect();
        loop {
            let missing: Vec<usize> = image
                .missing_aot_targets([start, main], reached)
                .into_iter()
                .filter(|idx| !broken.contains_key(idx) && tried.insert(*idx))
                .collect();
            if missing.is_empty() {
                break;
            }
            broken.extend(image.compile(cx, reached, &missing));
        }
    }
}

fn empty_batch(
    kind: BatchKind,
    store: Cache,
    member_errors: Vec<String>,
    reach_span: sema::census::Span,
) -> Result<Batch> {
    let reached = Reached::default();
    let image = Image::create(&reached, crate::image::ImageMode::Check, (0, 0, 1))?;
    Ok(Batch {
        kind,
        store: Some(store),
        labels: Vec::new(),
        roots: Vec::new(),
        tests: Vec::new(),
        reached,
        image,
        broken: HashMap::default(),
        poly_failures: Vec::new(),
        member_errors,
        reach_span,
        codegen_span: sema::census::Span::default(),
    })
}

fn pick_roots(
    cx: &mut sema::Context<'_>,
    (kind, main): (BatchKind, Option<ItemRef>),
    cases: &[ItemRef],
    root_insts: &[Instance],
    test_filters: &[String],
) -> Vec<Instance> {
    let mut out = match (kind, main) {
        (BatchKind::MainAot, Some(main_ref)) => {
            let mut roots: Vec<_> = sema::reach::main_roots(cx, main_ref)
                .map_or_else(|_| vec![(main_ref, Substs::EMPTY)], Into::into);
            if cases.len() > 1 {
                roots.extend(item_roots(&cases[1..]));
            }
            roots
        }
        #[cfg(feature = "diff-harness")]
        (BatchKind::Cases, _) => root_insts
            .iter()
            .copied()
            .filter(|&(item, _)| sema::infer::is_case_sig(cx, item))
            .collect(),
        (BatchKind::Tests, _) => select_test_roots(cx, cases, test_filters).2,
        (BatchKind::LibCheck | BatchKind::MainCheck, _) => root_insts
            .iter()
            .copied()
            .filter(|&(item, _)| is_concrete_fn(cx, item))
            .collect(),
        _ => root_insts.to_vec(),
    };
    if kind.forces_unwind()
        && crate::profile() == outline::tree::Profile::Release
        && let Some(bounds) = cx.lang_item(sema::reach::PANIC_BOUNDS_CHECK)
    {
        out.push((bounds, Substs::EMPTY));
    }
    out
}

fn finish_batch(
    (kind, main): (BatchKind, Option<ItemRef>),
    (cases, mut reached): (&[ItemRef], Reached),
    (drop_finals, test_filters): (bool, &[String]),
    substrate: Option<SubstrateSlot>,
    (mut cx, reach_clock): (sema::Context<'_>, sema::census::Clock),
) -> Result<Batch> {
    let mode = kind.mode();
    let reach_span = reach_clock.elapsed();
    if kind.forces_unwind() {
        reached.set_force_unwind(true);
    }
    if kind == BatchKind::Tests {
        let member_errors = validate_test_sigs(&mut cx, cases);
        if !member_errors.is_empty() {
            return empty_batch(kind, cx.release().0, member_errors, reach_span);
        }
    }
    let codegen_clock = sema::census::Clock::now();
    let pre_builder = export_pre_summaries(&mut cx, drop_finals, substrate.as_ref());
    let mut test_starts: Vec<Option<Instance>> = Vec::new();
    let valid: Vec<ItemRef> = match kind {
        #[cfg(feature = "diff-harness")]
        BatchKind::Cases => cases
            .iter()
            .copied()
            .filter(|&item| sema::infer::is_case_sig(&mut cx, item))
            .collect(),
        BatchKind::Tests => {
            let (valid, starts, _) = select_test_roots(&mut cx, cases, test_filters);
            test_starts = starts;
            valid
        }
        BatchKind::LibCheck | BatchKind::MainCheck => cases
            .iter()
            .copied()
            .filter(|&i| is_concrete_fn(&mut cx, i))
            .collect(),
        BatchKind::MainAot => Vec::new(),
    };
    let mut root_instances: Vec<Instance> = match main {
        Some(main) => sema::reach::main_roots(&mut cx, main)?.to_vec(),
        None => item_roots(&valid),
    };
    let mut extra_starts: Vec<Instance> = test_starts.iter().flatten().copied().collect();
    if kind == BatchKind::MainAot && cases.len() > 1 {
        extra_starts.extend(item_roots(&cases[1..]));
    }
    let primary_len = root_instances.len();
    root_instances.extend(extra_starts);
    reached.canonicalize(&cx, &root_instances);
    let roots: Vec<usize> = root_instances[..primary_len]
        .iter()
        .map(|inst| reached.index[inst])
        .collect();
    let start_indices = test_indices(&test_starts, &reached);
    if drop_finals {
        cx.compact_for_oneshot_codegen();
    }
    let (mut image, broken) = if mode == ImageMode::Aot && roots.len() == 2 {
        let mut aot_roots = vec![roots[0], roots[1]];
        aot_roots.extend(
            root_instances[primary_len..]
                .iter()
                .filter_map(|inst| reached.index.get(inst).copied()),
        );
        Image::build_aot(&mut cx, &reached, &aot_roots)?
    } else {
        Image::build(&mut cx, &reached, mode)?
    };
    let codegen_span = codegen_clock.elapsed();
    let labels = batch_labels(&mut cx, kind, &valid);
    let outline = cx.outline;
    let mut store = cx.release().0;
    save_batch_substrate(
        (outline, &mut store),
        &mut image,
        (drop_finals, kind),
        substrate,
        pre_builder,
    );
    let item = ItemRef {
        krate: 0,
        file: 0,
        item: 0,
    };
    let tests = start_indices
        .into_iter()
        .map(|s| {
            (
                TestItem {
                    item,
                    ignored: false,
                    should_panic: false,
                    expected: None,
                },
                s,
            )
        })
        .collect();
    Ok(Batch {
        kind,
        store: Some(store),
        labels,
        roots,
        tests,
        reached,
        image,
        broken,
        poly_failures: Vec::new(),
        member_errors: Vec::new(),
        reach_span,
        codegen_span,
    })
}

fn run_member_check(
    workspace: &graph::Workspace,
    compilation: &mut Compilation,
    cache: Cache,
    (kind, main): (BatchKind, Option<ItemRef>),
    local_check: Option<&super::substrate::LocalCheckState>,
) -> Result<(Cache, Vec<String>)> {
    if !matches!(kind, BatchKind::Tests | BatchKind::MainAot) {
        return Ok((cache, Vec::new()));
    }
    let target = main
        .filter(|_| kind == BatchKind::MainAot)
        .map(|m| m.krate as usize);
    if kind == BatchKind::MainAot
        && let Some(state) = local_check
    {
        if state.all_cached() {
            return Ok((cache, Vec::new()));
        }
        let only = state.dirty_cases(&compilation.outline).0;
        state.set_fixed(&mut compilation.outline, true);
        let res = check_member_items_for(workspace, compilation, cache, target, Some(only));
        state.set_fixed(&mut compilation.outline, false);
        return res;
    }
    check_member_items(workspace, compilation, cache, target)
}

impl Batch {
    pub(super) fn new(
        workspace: &graph::Workspace,
        compilation: &mut Compilation,
        kind: BatchKind,
        drop_finals: bool,
    ) -> Result<Batch> {
        Self::new_filtered(workspace, compilation, kind, drop_finals, &[])
    }

    pub(super) fn new_filtered(
        workspace: &graph::Workspace,
        compilation: &mut Compilation,
        kind: BatchKind,
        drop_finals: bool,
        test_filters: &[String],
    ) -> Result<Batch> {
        let main = (kind == BatchKind::MainAot)
            .then(|| require_main(workspace, &compilation.outline))
            .transpose()?;
        Self::rooted(
            workspace,
            compilation,
            (kind, main),
            (drop_finals, test_filters),
        )
    }

    pub(super) fn rooted(
        workspace: &graph::Workspace,
        compilation: &mut Compilation,
        (kind, main): (BatchKind, Option<ItemRef>),
        (drop_finals, test_filters): (bool, &[String]),
    ) -> Result<Batch> {
        let mode = kind.mode();
        let reach_clock = sema::census::Clock::now();
        let is_check_kind = matches!(kind, BatchKind::LibCheck | BatchKind::MainCheck);
        let use_local_check = drop_finals && (is_check_kind || kind == BatchKind::MainAot);
        if use_local_check {
            load_local_check_crates(workspace, compilation, kind != BatchKind::LibCheck)?;
        }
        let (mut substrate, mut cache) = cache_with_substrate(workspace, mode != ImageMode::Check);
        if substrate.is_some() && mode == ImageMode::Aot {
            attach_warm_cache(workspace, &mut cache);
        }
        let local_check = use_local_check
            .then(|| {
                let with_bin = kind != BatchKind::LibCheck;
                let sub = substrate.as_ref();
                prepare_local_check(workspace, &compilation.outline, with_bin, sub)
            })
            .flatten();
        let check_state = is_check_kind.then_some(local_check.as_ref()).flatten();
        if check_state.is_some_and(|state| state.all_cached()) {
            save_fixed(workspace, &compilation.outline);
            return empty_batch(kind, cache, Vec::new(), reach_clock.elapsed());
        }
        let (cases, root_insts) = if main.is_some() {
            (main_items(workspace, &compilation.outline), Vec::new())
        } else if let Some(ref state) = local_check {
            state.dirty_cases(&compilation.outline)
        } else {
            let items = kind.pick(workspace, &compilation.outline);
            let roots = item_roots(&items);
            (items, roots)
        };
        let (cache, member_errors) = run_member_check(
            workspace,
            compilation,
            cache,
            (kind, main),
            local_check.as_ref(),
        )?;
        if !member_errors.is_empty() {
            return empty_batch(kind, cache, member_errors, reach_clock.elapsed());
        }
        if let Some(state) = check_state {
            state.set_fixed(&mut compilation.outline, true);
        }
        let run_result = (|| -> Result<Batch> {
            let mut batch = crate::macros::reach_resolved_with_cache(
                compilation,
                cache,
                |cx| pick_roots(cx, (kind, main), &cases, &root_insts, test_filters),
                |cx, reached| {
                    finish_batch(
                        (kind, main),
                        (&cases, reached),
                        (drop_finals, test_filters),
                        substrate.take(),
                        (cx, reach_clock),
                    )
                },
            )?;
            if is_check_kind && let Some(store) = batch.store.take() {
                let (store, poly_failures) = check_poly_resolved(compilation, store, &cases)?;
                batch.store = Some(store);
                batch.poly_failures = poly_failures;
            }
            Ok(batch)
        })();
        if let Some(state) = check_state {
            state.set_fixed(&mut compilation.outline, false);
        }
        let batch = run_result?;
        if let Some(ref state) = local_check {
            state.record_batch(&compilation.outline, &batch);
        }
        save_fixed(workspace, &compilation.outline);
        Ok(batch)
    }

    pub(super) fn write_aot(
        &mut self,
        compilation: &Compilation,
        path: &std::path::Path,
    ) -> Result<()> {
        if !self.member_errors.is_empty() {
            bail!("{}", self.member_errors.join("\n"));
        }
        if !self.broken.is_empty() {
            bail!("{} functions failed to compile", self.broken.len());
        }
        let mut seen_crates = vec![false; compilation.outline.crates.len()];
        let mut queue = Vec::new();
        for ((item, _), _) in &self.reached.list {
            let k = item.krate as usize;
            if !seen_crates[k] {
                seen_crates[k] = true;
                queue.push(k);
            }
        }
        let mut has_static_archive = None;
        while let Some(k) = queue.pop() {
            if has_static_archive.is_none()
                && let Some(script) = &compilation.outline.crates[k].env.script
                && let Some((_, lib)) = script
                    .link_libs()
                    .find(|(kind, _)| *kind == crate::graph::script::LinkKind::Static)
            {
                let name = compilation.outline.roots[k].name.to_owned();
                has_static_archive = Some((lib.to_owned(), name));
            }
            for &(_, dep) in &compilation.outline.deps[k] {
                if !seen_crates[dep] {
                    seen_crates[dep] = true;
                    queue.push(dep);
                }
            }
        }
        let native_bundle = if let Some((lib, name)) = has_static_archive {
            if cfg!(unix) && crate::compile_target() == haste_codegen::Target::host() {
                crate::host::native::bundle(&compilation.outline)?
            } else {
                bail!(
                    "cross-target standalone AOT (-o) does not yet link static C archive `{lib}` from crate `{name}`; build on the target host or use `cargo build`"
                );
            }
        } else {
            None
        };
        let store = self.store.take().context("inference store in use")?;
        let mut cx = sema::Context::new(&compilation.outline, &compilation.names, store);
        let exe = self.image.to_aot(
            &crate::host_target(),
            (self.roots[0], self.roots[1]),
            &mut cx,
            &self.reached,
        );
        self.store = Some(cx.release().0);
        let mut exe = exe?;
        if let Some(bundle) = native_bundle {
            exe.extra_libs.push(bundle.to_string_lossy().into_owned());
        }
        exe.write(path)
            .with_context(|| format!("writing {}", path.display()))
    }

    fn needs_member_recheck(&self) -> bool {
        !self.member_errors.is_empty() || matches!(self.kind, BatchKind::Tests | BatchKind::MainAot)
    }

    fn finish_incremental(&mut self, mut cx: sema::Context<'_>, fresh: &[usize]) -> bool {
        recompile(
            &mut self.image,
            &mut self.broken,
            (&mut cx, &self.reached),
            fresh,
            &self.roots,
            self.kind.mode() == ImageMode::Check,
        );
        self.store = Some(cx.release().0);
        true
    }

    pub(super) fn again(&mut self, compilation: &Compilation, stale: &[(u32, u32)]) -> bool {
        if self.needs_member_recheck() {
            return false;
        }
        let Some(mut store) = self.store.take() else {
            return false;
        };
        let touched = store.invalidate(
            &compilation.outline,
            self.reached.list.iter().map(|(instance, _)| instance),
            Edited::Files(stale),
        );
        let mut cx = sema::Context::new(&compilation.outline, &compilation.names, store);
        let fresh = reach_again(&mut cx, &mut self.reached, &touched);
        cx.resolved() && self.finish_incremental(cx, &fresh)
    }

    pub(super) fn reload(
        &mut self,
        workspace: &graph::Workspace,
        compilation: &Compilation,
        crates: &[u32],
    ) -> bool {
        if self.needs_member_recheck() {
            return false;
        }
        let Some(mut store) = self.store.take() else {
            return false;
        };
        let touched = store.invalidate(
            &compilation.outline,
            self.reached.list.iter().map(|(instance, _)| instance),
            Edited::Crates(crates),
        );
        let mut cases = self.kind.pick(workspace, &compilation.outline);
        let roots;
        let mut cx = sema::Context::new(&compilation.outline, &compilation.names, store);
        let mut test_starts = Vec::new();
        match self.kind {
            #[cfg(feature = "diff-harness")]
            BatchKind::Cases => {
                cases.retain(|&item| sema::infer::is_case_sig(&mut cx, item));
                roots = item_roots(&cases);
            }
            BatchKind::Tests => {
                let (valid, starts, test_roots) = select_test_roots(&mut cx, &cases, &[]);
                cases = valid;
                test_starts = starts;
                roots = test_roots;
            }
            BatchKind::LibCheck | BatchKind::MainCheck => {
                self.poly_failures
                    .retain(|(i, _)| !crates.contains(&i.krate));
                let edited: Vec<ItemRef> = cases
                    .iter()
                    .copied()
                    .filter(|item| crates.contains(&item.krate))
                    .collect();
                self.poly_failures
                    .extend(check_poly_items(&mut cx, &edited));
                roots = concrete_roots(&mut cx, &cases);
            }
            BatchKind::MainAot => {
                let Some(&main) = cases.first() else {
                    return false;
                };
                let Ok(pair) = sema::reach::main_roots(&mut cx, main) else {
                    return false;
                };
                cases.clear();
                roots = pair.to_vec();
            }
        }
        let fresh = reach_anew_in(&mut cx, &mut self.reached, &touched, &roots, Some(crates));
        if !cx.resolved() {
            return false;
        }
        self.labels = batch_labels(&mut cx, self.kind, &cases);
        if self.kind == BatchKind::Tests {
            self.roots = cases
                .iter()
                .map(|&item| self.reached.index[&(item, Substs::EMPTY)])
                .collect();
            let starts = test_indices(&test_starts, &self.reached);
            self.tests = map_tests(
                &self.roots,
                &self.reached,
                &starts,
                test_items(workspace, &compilation.outline),
            );
        } else {
            self.roots = roots.iter().map(|root| self.reached.index[root]).collect();
        }
        touched.iter().for_each(|idx| drop(self.broken.remove(idx)));
        self.finish_incremental(cx, &fresh)
    }
}

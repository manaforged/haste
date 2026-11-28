use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use rustc_hash::FxHashSet as HashSet;

use crate::graph;
use crate::image::Image;
use crate::names::ItemRef;
use crate::outline;
use crate::sema::{self, Cache};
use crate::substrate::{
    Substrate, SubstrateBuilder, deps_key, enabled, local, target_hash, toolchain_key,
    workspace_key,
};

use super::BatchKind;
use super::batch::Batch;
use super::test_args::{check_fn_items_in, check_target, checked_member_crates};

pub(crate) struct SubstrateSlot {
    toolchain_key: u64,
    deps_key: Option<u64>,
    workspace_key: Option<u64>,
    target_key: u64,
    pub(super) toolchain: Option<Arc<Substrate>>,
    pub(super) deps: Option<Arc<Substrate>>,
    pub(super) workspace: Option<Arc<Substrate>>,
}

impl SubstrateSlot {
    pub(crate) fn top(&self) -> Option<Arc<Substrate>> {
        self.workspace
            .as_ref()
            .or(self.deps.as_ref())
            .or(self.toolchain.as_ref())
            .cloned()
    }
}

pub(super) struct PreBuilders {
    toolchain: SubstrateBuilder,
    deps: Option<SubstrateBuilder>,
    workspace: Option<SubstrateBuilder>,
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

fn cache_from_slot(
    slot: Option<SubstrateSlot>,
    warm_sysroot_bodies: bool,
) -> (Option<SubstrateSlot>, Cache) {
    let mut cache = Cache::default();
    cache.set_warm_sysroot_bodies(warm_sysroot_bodies);
    if let Some(sub) = slot.as_ref().and_then(SubstrateSlot::top) {
        cache.set_substrate(Some(sub));
    }
    (slot, cache)
}

pub(crate) fn cache_with_substrate(
    workspace: &graph::Workspace,
    warm_sysroot_bodies: bool,
) -> (Option<SubstrateSlot>, Cache) {
    cache_from_slot(open_batch_substrate(workspace), warm_sysroot_bodies)
}

pub(crate) fn cache_with_toolchain_substrate(
    workspace: &graph::Workspace,
) -> (Option<SubstrateSlot>, Cache) {
    cache_from_slot(open_toolchain_substrate(workspace), true)
}

fn open_toolchain_layer(
    workspace: &graph::Workspace,
) -> Option<(u64, u64, haste_codegen::Target, Option<Arc<Substrate>>)> {
    if !enabled() {
        return None;
    }
    let target = crate::host_target();
    let toolchain_key = toolchain_key(workspace, &target);
    let target_key = target_hash(&target);
    let toolchain = match Substrate::open_toolchain(toolchain_key) {
        Ok(sub) if sub.toolchain_hash() == toolchain_key && sub.target_hash() == target_key => {
            Some(Arc::new(sub))
        }
        _ => None,
    };
    Some((toolchain_key, target_key, target, toolchain))
}

pub(crate) fn open_toolchain_substrate(workspace: &graph::Workspace) -> Option<SubstrateSlot> {
    let (toolchain_key, target_key, _, toolchain) = open_toolchain_layer(workspace)?;
    Some(SubstrateSlot {
        toolchain_key,
        deps_key: None,
        workspace_key: None,
        target_key,
        toolchain,
        deps: None,
        workspace: None,
    })
}

pub(crate) fn open_batch_substrate(workspace: &graph::Workspace) -> Option<SubstrateSlot> {
    let (toolchain_key, target_key, target, toolchain) = open_toolchain_layer(workspace)?;
    let deps_key = deps_key(workspace, &target, toolchain_key);
    let workspace_key = workspace_key(workspace, &target, toolchain_key);
    let deps = deps_key.and_then(|dk| match Substrate::open_deps(dk, target_key) {
        Ok(sub) => Some(Arc::new(match &toolchain {
            Some(tc) => sub.with_parent(Arc::clone(tc)),
            None => sub,
        })),
        _ => None,
    });
    let base = deps.as_ref().or(toolchain.as_ref());
    let workspace_sub =
        workspace_key.and_then(|wk| match Substrate::open_workspace(wk, target_key) {
            Ok(sub) => Some(Arc::new(match base {
                Some(parent) => sub.with_parent(Arc::clone(parent)),
                None => sub,
            })),
            _ => None,
        });
    Some(SubstrateSlot {
        toolchain_key,
        deps_key,
        workspace_key,
        target_key,
        toolchain,
        deps,
        workspace: workspace_sub,
    })
}

fn make_builders(slot: &SubstrateSlot) -> PreBuilders {
    let toolchain = Substrate::builder()
        .toolchain_hash(slot.toolchain_key)
        .target_hash(slot.target_key);
    let deps = slot.deps_key.map(|dk| {
        Substrate::builder()
            .toolchain_hash(dk)
            .target_hash(slot.target_key)
    });
    let workspace = slot.workspace_key.map(|wk| {
        Substrate::builder()
            .toolchain_hash(wk)
            .target_hash(slot.target_key)
    });
    PreBuilders {
        toolchain,
        deps,
        workspace,
    }
}

pub(super) fn export_pre_summaries(
    cx: &mut sema::Context<'_>,
    drop_finals: bool,
    slot: Option<&SubstrateSlot>,
) -> Option<PreBuilders> {
    if !drop_finals || !cx.store().has_new_sysroot_summaries() {
        return None;
    }
    let mut builders = make_builders(slot?);
    if let Err(error) = cx.store().export_sysroot_summaries(
        cx.outline,
        Some(&mut builders.toolchain),
        builders.deps.as_mut(),
    ) {
        eprintln!("haste: cannot export sysroot summaries: {error:#}");
        return None;
    }
    builders.shrink_blobs();
    Some(builders)
}

fn save_substrate_layer(
    builder: SubstrateBuilder,
    (workspace, kind): (u64, BatchKind),
    existing: Option<Arc<Substrate>>,
    save: fn(SubstrateBuilder) -> Result<Option<PathBuf>>,
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
        store.export_sysroot_summaries(
            outline,
            Some(&mut builders.toolchain),
            builders.deps.as_mut(),
        )?;
    }
    if image.has_new_frozen_stencils() {
        image.export_sysroot_stencils(Some(&mut builders.toolchain), builders.deps.as_mut())?;
    }
    image.export_inline_meta(Some(&mut builders.toolchain), builders.deps.as_mut())?;
    finish_oneshot_compact(drop_finals, store, image);
    let workspace = slot
        .workspace_key
        .or(slot.deps_key)
        .unwrap_or(slot.toolchain_key);
    if let Some(ws_builder) = builders.workspace {
        save_substrate_layer(
            ws_builder,
            (workspace, kind),
            slot.workspace.take(),
            SubstrateBuilder::save_workspace,
        )?;
    }
    if let Some(deps_builder) = builders.deps {
        save_substrate_layer(
            deps_builder,
            (workspace, kind),
            slot.deps.take(),
            SubstrateBuilder::save_deps,
        )?;
    }
    save_substrate_layer(
        builders.toolchain,
        (workspace, kind),
        slot.toolchain.take(),
        SubstrateBuilder::save_toolchain,
    )
}

pub(crate) fn save_jit_substrate(
    outline: &outline::Outline,
    store: &Cache,
    image: &Image,
    mut slot: SubstrateSlot,
) {
    let has_summaries = store.has_new_sysroot_summaries();
    let has_stencils = image.has_new_frozen_stencils();
    if !has_summaries && !has_stencils {
        return;
    }
    let mut builders = make_builders(&slot);
    if has_summaries
        && let Err(error) = store.export_sysroot_summaries(
            outline,
            Some(&mut builders.toolchain),
            builders.deps.as_mut(),
        )
    {
        eprintln!("haste: cannot export jit summaries: {error:#}");
        return;
    }
    if has_stencils
        && let Err(error) =
            image.export_sysroot_stencils(Some(&mut builders.toolchain), builders.deps.as_mut())
    {
        eprintln!("haste: cannot export jit stencils: {error:#}");
        return;
    }
    let workspace = slot
        .workspace_key
        .or(slot.deps_key)
        .unwrap_or(slot.toolchain_key);
    let save = || -> Result<()> {
        if let Some(deps_builder) = builders.deps {
            save_substrate_layer(
                deps_builder,
                (workspace, BatchKind::MainAot),
                slot.deps.take(),
                SubstrateBuilder::save_deps,
            )?;
        }
        save_substrate_layer(
            builders.toolchain,
            (workspace, BatchKind::MainAot),
            slot.toolchain.take(),
            SubstrateBuilder::save_toolchain,
        )
    };
    if let Err(error) = save() {
        eprintln!("haste: cannot save jit substrate: {error:#}");
    }
}

pub(super) fn save_batch_substrate(
    (outline, store): (&outline::Outline, &mut Cache),
    image: &mut Image,
    (drop_finals, kind): (bool, BatchKind),
    slot: Option<SubstrateSlot>,
    pre_builder: Option<PreBuilders>,
) {
    let has_new = image.has_new_frozen_stencils()
        || image.has_new_inline_meta()
        || pre_builder.is_some()
        || (!drop_finals && store.has_new_sysroot_summaries());
    let Some(slot) = slot.filter(|_| has_new) else {
        finish_oneshot_compact(drop_finals, store, image);
        return;
    };
    let builders = pre_builder.unwrap_or_else(|| make_builders(&slot));
    if let Err(error) =
        export_and_save_layers((outline, store), image, (drop_finals, kind), slot, builders)
    {
        eprintln!("haste: cannot save substrate: {error:#}");
    }
}

pub(super) struct LocalCheckState {
    hits: Vec<usize>,
    dirty: Vec<local::DirtyCrate>,
}

fn check_crates(
    workspace: &graph::Workspace,
    outline: &outline::Outline,
    is_main: bool,
) -> Vec<usize> {
    let target = check_target(workspace, outline, is_main);
    let mut seen = vec![false; outline.crates.len()];
    let mut stack = checked_member_crates(workspace, outline, target);
    let mut out = Vec::new();
    while let Some(k) = stack.pop() {
        if std::mem::replace(&mut seen[k], true) || outline.is_fixed(k as u32) {
            continue;
        }
        out.push(k);
        for &(_, dep) in &outline.deps[k] {
            if !outline.is_fixed(dep as u32) && outline.roots[dep].role == outline::Role::Lib {
                stack.push(dep);
            }
        }
    }
    out
}

pub(super) fn load_local_check_crates(
    workspace: &graph::Workspace,
    compilation: &mut crate::macros::Compilation,
    is_main: bool,
) -> Result<()> {
    super::load_members(workspace, compilation)?;
    if !local::active() {
        return Ok(());
    }
    let crates = check_crates(workspace, &compilation.outline, is_main);
    let mut loaded_more = false;
    for k in crates {
        if compilation.outline.crates[k].partial() {
            compilation.outline.load_crate(k)?;
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

impl LocalCheckState {
    pub(super) fn all_cached(&self) -> bool {
        self.dirty.is_empty()
    }

    pub(super) fn dirty_cases(
        &self,
        outline: &outline::Outline,
    ) -> (Vec<ItemRef>, Vec<sema::reach::Instance>) {
        let dirty: Vec<(usize, Option<&[usize]>)> = self
            .dirty
            .iter()
            .map(|entry| (entry.krate, entry.dirty_files.as_deref()))
            .collect();
        let items = check_fn_items_in(outline, &dirty);
        let roots = items
            .iter()
            .map(|&item| (item, sema::substs::Substs::EMPTY))
            .collect();
        (items, roots)
    }

    pub(super) fn set_fixed(&self, outline: &mut outline::Outline, fixed: bool) {
        for &k in &self.hits {
            outline.crates[k].env.fixed = fixed;
        }
    }

    pub(super) fn record_batch(&self, outline: &outline::Outline, batch: &Batch) {
        if !batch.reached.unlaid.is_empty() || !batch.member_errors.is_empty() {
            return;
        }
        let mut failed_crates = HashSet::default();
        for ((item, _), res) in &batch.reached.list {
            if res.is_err() {
                failed_crates.insert(item.krate);
            }
        }
        for &idx in batch.broken.keys() {
            failed_crates.insert(batch.reached.list[idx].0.0.krate);
        }
        for &(item, _) in &batch.poly_failures {
            failed_crates.insert(item.krate);
        }
        local::record(outline, &self.dirty, &failed_crates);
    }
}

pub(super) fn prepare_local_check(
    workspace: &graph::Workspace,
    outline: &outline::Outline,
    is_main: bool,
    slot: Option<&SubstrateSlot>,
) -> Option<LocalCheckState> {
    if !local::active() {
        return None;
    }
    let crates = check_crates(workspace, outline, is_main);
    if crates.is_empty() {
        return None;
    }
    let (tc_key, tgt_key) = match slot {
        Some(s) => (s.toolchain_key, s.target_key),
        None => {
            let target = crate::host_target();
            (toolchain_key(workspace, &target), target_hash(&target))
        }
    };
    let part = local::partition(workspace, outline, &crates, (tc_key, tgt_key));
    Some(LocalCheckState {
        hits: part.hits,
        dirty: part.dirty,
    })
}

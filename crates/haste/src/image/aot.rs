use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock};

use haste_codegen::{Code, CodeUnwindDrop, Site, StencilReloc};
use rustc_hash::{FxHashMap as HashMap, FxHasher};

use crate::host::HostSymbol;
use crate::lower::{
    FootprintId, RelocPlan, RodataId, RodataReloc, RodataTarget, Slots, SymbolSource, VtableEntry,
    VtableId, Word,
};
use crate::names::ItemRef;
use crate::sema::Context;
use crate::sema::reach::Reached;
use crate::sync::lock;

pub(crate) mod cache;
mod emit;
mod icf;
mod queue;
mod reach;
mod rodata;
mod vdce;

pub(super) use queue::{AotQueue, AotWork, bucket_by_item, collect_warm_callees, lpt_chunks};
pub(super) use reach::AotReach;
use reach::{FnEdge, for_each_fn_edge};

use super::Image;

#[derive(Clone)]
pub(super) struct AotCode {
    bytes: Vec<u8>,
    sites: Vec<Site>,
    relocs: Vec<StencilReloc>,
    loops: Vec<std::ops::Range<u32>>,
    lines: Vec<(u32, u64)>,
    unwind_drops: Vec<CodeUnwindDrop>,
    saved_regs: u32,
}

impl AotCode {
    pub(super) fn from_code(code: &Code) -> Arc<Self> {
        Arc::new(Self {
            bytes: code.bytes().to_vec(),
            sites: code.sites().to_vec(),
            relocs: code.relocs().to_vec(),
            loops: code.loops().to_vec(),
            lines: code.lines().to_vec(),
            unwind_drops: code.unwind_drops().to_vec(),
            saved_regs: code.saved_regs(),
        })
    }

    pub(super) fn saved_regs(&self) -> u32 {
        self.saved_regs
    }

    pub(super) fn sites(&self) -> &[Site] {
        &self.sites
    }

    pub(super) fn loops(&self) -> &[std::ops::Range<u32>] {
        &self.loops
    }

    pub(super) fn lines(&self) -> &[(u32, u64)] {
        &self.lines
    }

    pub(super) fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub(super) fn relocs(&self) -> &[StencilReloc] {
        &self.relocs
    }

    pub(super) fn unwind_drops(&self) -> &[CodeUnwindDrop] {
        &self.unwind_drops
    }

    #[cfg(test)]
    pub(super) fn for_test(bytes: Vec<u8>, relocs: Vec<StencilReloc>) -> Arc<Self> {
        Arc::new(Self {
            bytes,
            sites: Vec::new(),
            relocs,
            loops: Vec::new(),
            lines: Vec::new(),
            unwind_drops: Vec::new(),
            saved_regs: 0,
        })
    }
}

#[derive(Clone)]
pub(super) enum AotFn {
    Direct {
        code: Arc<AotCode>,
        relocs: Vec<SymbolSource>,
    },
    Dict {
        body: Arc<AotCode>,
        relocs: Vec<SymbolSource>,
    },
    Alias(usize),
}

#[derive(Clone)]
pub(super) enum CachedAotTemplate {
    Direct {
        code: Arc<AotCode>,
        reloc_plan: Arc<RelocPlan>,
    },
    Dict {
        body: Arc<AotCode>,
        reloc_plan: Arc<RelocPlan>,
    },
}

impl CachedAotTemplate {
    pub(super) fn reloc_plan(&self) -> &Arc<RelocPlan> {
        match self {
            Self::Direct { reloc_plan, .. } | Self::Dict { reloc_plan, .. } => reloc_plan,
        }
    }
}

pub(super) enum AotStatic {
    Local(u32),
    Foreign {
        weak: bool,
        host: Option<HostSymbol>,
    },
}

fn hash_key<T: Hash + ?Sized>(val: &T) -> u64 {
    let mut h = FxHasher::default();
    val.hash(&mut h);
    h.finish()
}

pub(crate) struct AotState {
    pub(super) rodata: Vec<(Vec<u8>, Vec<RodataReloc>)>,
    rodata_buckets: HashMap<u64, Vec<RodataId>>,
    pub(super) vtables: Vec<Vec<VtableEntry>>,
    vtable_buckets: HashMap<u64, Vec<VtableId>>,
    pub(super) statics: HashMap<usize, AotStatic>,
    pub(super) promoted: Vec<((usize, u32), u32)>,
    pub(super) promoted_seen: HashMap<(usize, u32), usize>,
    pub(super) foreign_fns: HashMap<usize, HostSymbol>,
    pub(super) functions: Vec<Option<AotFn>>,
    pub(super) stencils: HashMap<(ItemRef, FootprintId), CachedAotTemplate>,
    pub(super) frontier: Option<Arc<AotQueue>>,
    frontier_vtables: Vec<bool>,
    frontier_rodata: Vec<bool>,
    pub(super) cache: Option<Arc<cache::AotCache>>,
    pub(super) need_materialize: Vec<usize>,
}

impl AotState {
    pub(super) fn new(len: usize) -> Mutex<Self> {
        Mutex::new(Self {
            rodata: Vec::new(),
            rodata_buckets: HashMap::default(),
            vtables: Vec::new(),
            vtable_buckets: HashMap::default(),
            statics: HashMap::default(),
            promoted: Vec::new(),
            promoted_seen: HashMap::default(),
            foreign_fns: HashMap::default(),
            functions: vec![None; len],
            stencils: HashMap::default(),
            frontier: None,
            frontier_vtables: Vec::new(),
            frontier_rodata: Vec::new(),
            cache: None,
            need_materialize: Vec::new(),
        })
    }

    pub(super) fn ensure_len(&mut self, len: usize) {
        if self.functions.len() < len {
            self.functions.resize(len, None);
        }
    }

    pub(super) fn alloc_rodata(&mut self, bytes: &[u8], relocs: &[RodataReloc]) -> RodataId {
        let id = RodataId(self.rodata.len() as u32);
        self.rodata.push((bytes.to_vec(), relocs.to_vec()));
        id
    }

    pub(super) fn intern_rodata(&mut self, bytes: &[u8], relocs: &[RodataReloc]) -> RodataId {
        let hash = hash_key(&(bytes, relocs));
        if let Some(bucket) = self.rodata_buckets.get(&hash) {
            for &id in bucket {
                if let Some((b, r)) = self.rodata.get(id.0 as usize)
                    && b.as_slice() == bytes
                    && r.as_slice() == relocs
                {
                    return id;
                }
            }
        }
        let id = self.alloc_rodata(bytes, relocs);
        self.rodata_buckets.entry(hash).or_default().push(id);
        id
    }

    pub(super) fn intern_vtable(&mut self, entries: &[VtableEntry]) -> VtableId {
        let hash = hash_key(entries);
        if let Some(bucket) = self.vtable_buckets.get(&hash) {
            for &id in bucket {
                if self
                    .vtables
                    .get(id.0 as usize)
                    .is_some_and(|e| e.as_slice() == entries)
                {
                    return id;
                }
            }
        }
        let id = VtableId(self.vtables.len() as u32);
        self.vtables.push(entries.to_vec());
        self.vtable_buckets.entry(hash).or_default().push(id);
        id
    }

    fn collect_vtable_targets(
        &mut self,
        mut vstack: Vec<usize>,
        queue: &AotQueue,
        out: &mut Vec<usize>,
    ) {
        while let Some(id) = vstack.pop() {
            if id >= self.frontier_vtables.len() {
                self.frontier_vtables.resize(id + 1, false);
            }
            if self.frontier_vtables[id] {
                continue;
            }
            self.frontier_vtables[id] = true;
            for entry in self.vtables.get(id).into_iter().flatten() {
                match *entry {
                    VtableEntry::Word(Word::Code(fn_idx)) if queue.claim(fn_idx) => {
                        out.push(fn_idx);
                    }
                    VtableEntry::Super(super_id) => vstack.push(super_id.0 as usize),
                    VtableEntry::Word(_) => {}
                }
            }
        }
    }

    fn collect_rodata_targets(
        &mut self,
        mut rstack: Vec<usize>,
        queue: &AotQueue,
        out: &mut Vec<usize>,
    ) {
        while let Some(id) = rstack.pop() {
            if id >= self.frontier_rodata.len() {
                self.frontier_rodata.resize(id + 1, false);
            }
            if std::mem::replace(&mut self.frontier_rodata[id], true) {
                continue;
            }
            for reloc in self.rodata.get(id).into_iter().flat_map(|(_, r)| r) {
                match reloc.target {
                    RodataTarget::Rodata(child) => rstack.push(child.0 as usize),
                    RodataTarget::Fn(fn_idx) if queue.claim(fn_idx) => out.push(fn_idx),
                    RodataTarget::Fn(_) => {}
                }
            }
        }
    }

    fn collect_new_targets(&mut self, entry: &AotFn, queue: &AotQueue) -> Vec<usize> {
        let (mut out, mut vstack, mut rstack) = (Vec::new(), Vec::new(), Vec::new());
        for_each_fn_edge(entry, true, |edge| match edge {
            FnEdge::Sym(
                SymbolSource::Slot(idx) | SymbolSource::StaticCell { index: idx, .. },
                _,
            ) => {
                if queue.claim(*idx) {
                    out.push(*idx);
                }
            }
            FnEdge::Sym(SymbolSource::Vtable(id), _) => vstack.push(id.0 as usize),
            FnEdge::Sym(SymbolSource::Rodata(id), _) => rstack.push(id.0 as usize),
            FnEdge::Sym(_, _) => {}
            FnEdge::Alias(leader) => {
                if queue.claim(leader) {
                    out.push(leader);
                }
            }
        });
        self.collect_vtable_targets(vstack, queue, &mut out);
        self.collect_rodata_targets(rstack, queue, &mut out);
        out
    }
}

impl Image {
    /// Drops cached inline bodies, stencil passes, and file indices before writing a one-shot AOT executable.
    pub(crate) fn compact_for_oneshot_aot(&mut self) {
        self.bodies = crate::lower::Bodies::default();
        self.pass = Default::default();
        self.file_index = OnceLock::new();
    }

    /// Returns live AOT-reachable non-foreign function indices from `roots` not yet compiled into `AotState`.
    pub(crate) fn missing_aot_targets(
        &self,
        roots: [usize; 2],
        reached: &crate::sema::reach::Reached,
    ) -> Vec<usize> {
        let Some(aot) = self.aot() else {
            return Vec::new();
        };
        let mut state = lock(aot);
        state.ensure_len(reached.list.len());
        let mask = vdce::VtableMask::all_live();
        let live = AotReach::collect(&state, &roots, &mask, true);
        live.order
            .into_iter()
            .filter(|&idx| {
                state.functions.get(idx).and_then(Option::as_ref).is_none()
                    && !state.foreign_fns.contains_key(&idx)
                    && !state.statics.contains_key(&idx)
                    && reached.list.get(idx).is_some_and(|(_, r)| r.is_ok())
            })
            .collect()
    }

    pub(super) fn aot_cache(&self) -> Option<Arc<cache::AotCache>> {
        self.aot().and_then(|aot| lock(aot).cache.clone())
    }

    pub(super) fn record_aot(&self, index: usize, entry: AotFn, followers: &[usize]) {
        if let Some(aot) = self.aot() {
            let mut state = lock(aot);
            state.ensure_len(index + 1);
            let frontier = state.frontier.clone();
            let discovered = frontier
                .as_ref()
                .map(|q| state.collect_new_targets(&entry, q));
            state.functions[index] = Some(entry);
            for &follower in followers {
                state.ensure_len(follower + 1);
                state.functions[follower] = Some(AotFn::Alias(index));
            }
            drop(state);
            if let (Some(q), Some(items)) = (frontier, discovered)
                && !items.is_empty()
            {
                q.push_loose(items);
            }
        }
    }

    pub(super) fn record_aot_batch(&self, hits: Vec<(usize, AotFn)>) {
        if hits.is_empty() {
            return;
        }
        let Some(aot) = self.aot() else {
            return;
        };
        let Some(max_idx) = hits.iter().map(|(idx, _)| *idx).max() else {
            return;
        };
        let mut state = lock(aot);
        state.ensure_len(max_idx + 1);
        let frontier = state.frontier.clone();
        let mut discovered = Vec::new();
        for (index, entry) in hits {
            if let Some(q) = frontier.as_ref() {
                discovered.extend(state.collect_new_targets(&entry, q));
            }
            state.functions[index] = Some(entry);
        }
        drop(state);
        if let Some(q) = frontier
            && !discovered.is_empty()
        {
            q.push_loose(discovered);
        }
    }

    pub(super) fn drain_aot_cache_hits(
        &self,
        cx: &mut Context<'_>,
        reached: &Reached,
        work: &[usize],
        pass: &mut super::Pass,
    ) -> Vec<usize> {
        let Some(cache) = self.aot_cache() else {
            let mut misses = Vec::with_capacity(work.len());
            let mut memo_misses = Vec::new();
            for &index in work {
                if reached.is_reach_memo(index) {
                    memo_misses.push(index);
                } else {
                    misses.push(index);
                }
            }
            if !memo_misses.is_empty()
                && let Some(aot) = self.aot()
            {
                lock(aot).need_materialize.extend(memo_misses);
            }
            return misses;
        };
        if !cache.has_loaded_bundle() {
            cache.mark_miss();
            let mut misses = Vec::with_capacity(work.len());
            let mut memo_misses = Vec::new();
            for &index in work {
                if reached.is_reach_memo(index) {
                    memo_misses.push(index);
                } else {
                    misses.push(index);
                }
            }
            if !memo_misses.is_empty()
                && let Some(aot) = self.aot()
            {
                lock(aot).need_materialize.extend(memo_misses);
            }
            return misses;
        }
        let mut misses = Vec::new();
        let mut memo_misses = Vec::new();
        let mut ty_memo = HashMap::default();
        let mut hit_inlined = Vec::new();
        let mut hits = Vec::with_capacity(work.len());
        let mut hit_count = 0usize;
        for &index in work {
            if let Some(key) = cache.instance_key(cx, reached, index, &mut ty_memo)
                && let Some((entry, len)) =
                    cache.lookup_instance_batched(index, &key, self, cx, reached, &mut hit_inlined)
            {
                hit_count += 1;
                let relocs = match &entry {
                    AotFn::Direct { relocs, .. } | AotFn::Dict { relocs, .. } => relocs.as_slice(),
                    AotFn::Alias(_) => &[],
                };
                if relocs.iter().any(|sym| {
                    matches!(
                        sym,
                        SymbolSource::StaticCell { .. }
                            | SymbolSource::PromotedCell { .. }
                            | SymbolSource::ThreadSlot(_)
                    )
                }) {
                    self.register_aot_relocs(relocs);
                }
                pass.record_size(index, len, 0);
                hits.push((index, entry));
            } else {
                cache.note_miss();
                if reached.is_reach_memo(index) {
                    memo_misses.push(index);
                } else {
                    misses.push(index);
                }
            }
        }
        cache.note_hits(hit_count);
        cache.record_hit_inlined_batch(hit_inlined);
        self.record_aot_batch(hits);
        if !memo_misses.is_empty() {
            cache.mark_miss();
            if let Some(aot) = self.aot() {
                lock(aot).need_materialize.extend(memo_misses);
            }
        }
        if !misses.is_empty() {
            cache.mark_miss();
            if self.allow_inline()
                && let Some(aot) = self.aot()
                && let Some(queue) = lock(aot).frontier.clone()
            {
                queue.push_warm(collect_warm_callees(cx, reached, &queue, &misses));
            }
        }
        misses
    }

    pub(super) fn record_aot_inlined_evals(
        &self,
        cx: &Context<'_>,
        reached: &Reached,
        plan: &RelocPlan,
        evaluated: &[(usize, Vec<SymbolSource>)],
    ) {
        let Some(cache) = self.aot_cache() else {
            return;
        };
        for (index, relocs) in evaluated {
            cache.record_compiled_inlined(cx, reached, plan, *index, relocs);
        }
    }

    pub(super) fn record_aot_inlined_groups(
        &self,
        cx: &Context<'_>,
        reached: &Reached,
        plan: &RelocPlan,
        evaluated: &[(usize, &Vec<usize>, Vec<SymbolSource>)],
    ) {
        let Some(cache) = self.aot_cache() else {
            return;
        };
        for (leader, _, relocs) in evaluated {
            cache.record_compiled_inlined(cx, reached, plan, *leader, relocs);
        }
    }

    pub(super) fn record_aot_inlined_indices(
        &self,
        cx: &Context<'_>,
        reached: &Reached,
        plan: &RelocPlan,
        indices: &[usize],
    ) {
        let Some(cache) = self.aot_cache() else {
            return;
        };
        let Some(aot) = self.aot() else {
            return;
        };
        let pairs: Vec<(usize, Vec<SymbolSource>)> = {
            let state = lock(aot);
            indices
                .iter()
                .filter_map(|&idx| {
                    let mut cur = idx;
                    for _ in 0..state.functions.len() {
                        match state.functions.get(cur).and_then(Option::as_ref)? {
                            AotFn::Alias(next) if *next < state.functions.len() => cur = *next,
                            AotFn::Direct { relocs, .. } | AotFn::Dict { relocs, .. } => {
                                return Some((idx, relocs.clone()));
                            }
                            AotFn::Alias(_) => return None,
                        }
                    }
                    None
                })
                .collect()
        };
        for (idx, relocs) in pairs {
            cache.record_compiled_inlined(cx, reached, plan, idx, &relocs);
        }
    }
}

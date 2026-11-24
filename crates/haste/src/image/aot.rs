use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock};

use haste_codegen::{Code, CodeUnwindDrop, Site, StencilReloc};
use rustc_hash::{FxHashMap as HashMap, FxHasher};

use crate::host::HostSymbol;
use crate::lower::{
    FootprintId, RelocPlan, RodataId, RodataReloc, RodataTarget, SymbolSource, VtableEntry,
    VtableId, Word,
};
use crate::names::ItemRef;
use crate::sync::lock;

mod emit;
mod icf;
mod queue;
mod reach;
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
        })
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
        for_each_fn_edge(entry, |edge| match edge {
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
        let live = AotReach::collect(&state, roots, &mask);
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
}

use std::collections::VecDeque;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use anyhow::{Context, Result, bail};
use haste_codegen::{Code, RelocKind, StencilReloc, Target};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet, FxHasher};

use crate::aot::{Executable, Symbol};
use crate::host::HostSymbol;
use crate::lower::{FootprintId, RelocPlan, RodataId, RodataReloc, RodataTarget, SymbolSource, VtableEntry, VtableId, Word};
use crate::names::ItemRef;
use crate::sync::lock;

mod vdce;

use super::Image;

const FN_ALIGN: u32 = 64;
const FN_ALIGN_ENV: &str = "HASTE_FN_ALIGN";

fn fn_align_override() -> Option<u32> {
    static ALIGN: OnceLock<Option<u32>> = OnceLock::new();
    *ALIGN.get_or_init(|| std::env::var(FN_ALIGN_ENV).ok().and_then(|v| v.parse().ok()).filter(|a: &u32| a.is_power_of_two() && *a >= 16))
}

fn place(exe: &mut Executable, code: &AotCode) -> u32 {
    let align = fn_align_override().unwrap_or(if code.loops().is_empty() { 16 } else { FN_ALIGN });
    exe.function(code.bytes(), align, code.loops())
}

#[derive(Clone)]
pub(super) struct AotCode {
    bytes: Vec<u8>,
    relocs: Vec<StencilReloc>,
    loops: Vec<std::ops::Range<u32>>,
}

impl AotCode {
    pub(super) fn from_code(code: &Code) -> Arc<Self> {
        Arc::new(Self { bytes: code.bytes().to_vec(), relocs: code.relocs().to_vec(), loops: code.loops().to_vec() })
    }

    pub(super) fn loops(&self) -> &[std::ops::Range<u32>] { &self.loops }
    pub(super) fn bytes(&self) -> &[u8] { &self.bytes }
    pub(super) fn relocs(&self) -> &[StencilReloc] { &self.relocs }
}

#[derive(Clone)]
pub(super) enum AotFn {
    Direct { code: Arc<AotCode>, relocs: Vec<SymbolSource> },
    Dict { body: Arc<AotCode>, relocs: Vec<SymbolSource> },
    Alias(usize),
}

#[derive(Clone)]
pub(super) enum CachedAotTemplate {
    Direct { code: Arc<AotCode>, reloc_plan: Arc<RelocPlan> },
    Dict { body: Arc<AotCode>, reloc_plan: Arc<RelocPlan> },
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
    Foreign { weak: bool, host: Option<HostSymbol> },
}

struct QueueState {
    buckets: VecDeque<Vec<usize>>,
    loose: Vec<usize>,
    active: usize,
}

pub(super) fn bucket_by_item(work: &[usize], item_of: impl Fn(usize) -> ItemRef) -> Vec<Vec<usize>> {
    let mut pos_map: HashMap<ItemRef, usize> = HashMap::default();
    let mut buckets: Vec<Vec<usize>> = Vec::new();
    for &index in work {
        let item = item_of(index);
        if let Some(&pos) = pos_map.get(&item) {
            buckets[pos].push(index);
        } else {
            pos_map.insert(item, buckets.len());
            buckets.push(vec![index]);
        }
    }
    buckets
}

pub(super) fn lpt_chunks(buckets: Vec<Vec<usize>>, max_bucket: usize) -> Vec<Vec<usize>> {
    let mut chunks: Vec<Vec<usize>> = buckets
        .into_iter()
        .flat_map(|b| if b.len() <= max_bucket { vec![b] } else { b.chunks(max_bucket).map(|c| c.to_vec()).collect() })
        .collect();
    chunks.sort_by_key(|b| std::cmp::Reverse(b.len()));
    chunks
}

pub(super) struct AotQueue {
    claimed: Vec<AtomicBool>,
    items: Box<[ItemRef]>,
    max_bucket: usize,
    state: Mutex<QueueState>,
    wake: Condvar,
}

pub(super) struct TaskGuard<'a>(&'a AotQueue);

impl Drop for TaskGuard<'_> {
    fn drop(&mut self) {
        self.0.finish_task();
    }
}

impl AotQueue {
    pub(super) fn new(claimed: Vec<AtomicBool>, items: Box<[ItemRef]>, max_bucket: usize, initial_buckets: Vec<Vec<usize>>) -> Arc<Self> {
        Arc::new(Self {
            claimed,
            items,
            max_bucket,
            state: Mutex::new(QueueState { buckets: initial_buckets.into(), loose: Vec::new(), active: 0 }),
            wake: Condvar::new(),
        })
    }

    pub(super) fn claim(&self, idx: usize) -> bool {
        idx < self.claimed.len() && !self.claimed[idx].swap(true, Ordering::AcqRel)
    }

    pub(super) fn push_loose(&self, items: Vec<usize>) {
        if items.is_empty() {
            return;
        }
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.loose.extend(items);
        self.wake.notify_all();
    }

    pub(super) fn reseed_buckets(&self, buckets: Vec<Vec<usize>>) {
        if buckets.is_empty() {
            return;
        }
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.buckets.extend(buckets);
        self.wake.notify_all();
    }

    pub(super) fn pop(&self) -> Option<(Vec<usize>, TaskGuard<'_>)> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(bucket) = state.buckets.pop_front() {
                state.active += 1;
                return Some((bucket, TaskGuard(self)));
            }
            if !state.loose.is_empty() {
                let loose = std::mem::take(&mut state.loose);
                let buckets = lpt_chunks(bucket_by_item(&loose, |idx| self.items[idx]), self.max_bucket);
                state.buckets.extend(buckets);
                if state.buckets.len() > 1 {
                    self.wake.notify_all();
                }
                continue;
            }
            if state.active == 0 {
                self.wake.notify_all();
                return None;
            }
            state = self.wake.wait(state).unwrap_or_else(|e| e.into_inner());
        }
    }

    pub(super) fn take_all_pending(&self) -> Vec<usize> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let mut out = Vec::new();
        while let Some(bucket) = state.buckets.pop_front() {
            out.extend(bucket);
        }
        out.append(&mut state.loose);
        out
    }

    fn finish_task(&self) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.active = state.active.saturating_sub(1);
        if state.active == 0 && state.buckets.is_empty() && state.loose.is_empty() {
            self.wake.notify_all();
        }
    }
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
}

impl AotState {
    pub(super) fn new(len: usize) -> Mutex<Self> {
        Mutex::new(Self {
            rodata: Vec::new(), rodata_buckets: HashMap::default(),
            vtables: Vec::new(), vtable_buckets: HashMap::default(),
            statics: HashMap::default(), promoted: Vec::new(), promoted_seen: HashMap::default(),
            foreign_fns: HashMap::default(), functions: vec![None; len],
            stencils: HashMap::default(), frontier: None, frontier_vtables: Vec::new(),
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
                if self.vtables.get(id.0 as usize).is_some_and(|e| e.as_slice() == entries) {
                    return id;
                }
            }
        }
        let id = VtableId(self.vtables.len() as u32);
        self.vtables.push(entries.to_vec());
        self.vtable_buckets.entry(hash).or_default().push(id);
        id
    }

    fn collect_vtable_targets(&mut self, mut vstack: Vec<usize>, queue: &AotQueue, out: &mut Vec<usize>) {
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
                    VtableEntry::Word(Word::Code(fn_idx)) if queue.claim(fn_idx) => out.push(fn_idx),
                    VtableEntry::Super(super_id) => vstack.push(super_id.0 as usize),
                    VtableEntry::Word(_) => {}
                }
            }
        }
    }

    fn collect_new_targets(&mut self, entry: &AotFn, queue: &AotQueue) -> Vec<usize> {
        let mut out = Vec::new();
        let mut vstack = Vec::new();
        for_each_fn_edge(entry, |edge| match edge {
            FnEdge::Sym(SymbolSource::Slot(idx) | SymbolSource::StaticCell { index: idx, .. }, _) => {
                if queue.claim(*idx) {
                    out.push(*idx);
                }
            }
            FnEdge::Sym(SymbolSource::Vtable(id), _) => vstack.push(id.0 as usize),
            FnEdge::Sym(_, _) => {}
            FnEdge::Alias(leader) => {
                if queue.claim(leader) {
                    out.push(leader);
                }
            }
        });
        self.collect_vtable_targets(vstack, queue, &mut out);
        out
    }
}

enum FnEdge<'a> {
    Sym(&'a SymbolSource, bool),
    Alias(usize),
}

fn for_each_fn_edge(entry: &AotFn, mut on_edge: impl FnMut(FnEdge<'_>)) {
    match entry {
        AotFn::Direct { code, relocs } => {
            for reloc in code.relocs() {
                if reloc.as_runtime().is_some() {
                    continue;
                }
                if let Some(sym) = relocs.get(reloc.symbol as usize) {
                    on_edge(FnEdge::Sym(
                        sym,
                        matches!(reloc.kind, RelocKind::Arm64Call26 | RelocKind::PcRel32),
                    ));
                }
            }
        }
        AotFn::Dict { relocs, .. } => {
            for sym in relocs {
                on_edge(FnEdge::Sym(sym, false));
            }
        }
        AotFn::Alias(leader) => on_edge(FnEdge::Alias(*leader)),
    }
}

/// Keys in the order the reach walk first meets them, which follows the call graph and not the scheduling-dependent reached indexes.
struct FirstSeen<K> {
    seen: HashSet<K>,
    list: Vec<K>,
}

impl<K> Default for FirstSeen<K> {
    fn default() -> Self {
        Self { seen: HashSet::default(), list: Vec::new() }
    }
}

impl<K: Copy + Eq + Hash> FirstSeen<K> {
    fn insert(&mut self, key: K) {
        if self.seen.insert(key) {
            self.list.push(key);
        }
    }
}

pub(super) struct AotReach {
    fns: Vec<bool>,
    order: Vec<usize>,
    rodata: Vec<bool>,
    rodata_order: Vec<usize>,
    vtables: Vec<bool>,
    vtable_order: Vec<usize>,
    slots: FirstSeen<usize>,
    statics: FirstSeen<usize>,
    promoted: FirstSeen<(usize, u32)>,
    thread_slots: FirstSeen<usize>,
    work_fns: Vec<usize>,
    work_rodata: Vec<usize>,
    work_vtables: Vec<usize>,
}

fn mark_id(seen: &mut Vec<bool>, order: &mut Vec<usize>, work: &mut Vec<usize>, id: usize) {
    if id >= seen.len() {
        seen.resize(id + 1, false);
    }
    if !seen[id] {
        seen[id] = true;
        order.push(id);
        work.push(id);
    }
}

impl AotReach {
    pub(super) fn new(len: usize, roots: [usize; 2]) -> Self {
        let mut this = Self {
            fns: vec![false; len], order: Vec::new(),
            rodata: Vec::new(), rodata_order: Vec::new(),
            vtables: Vec::new(), vtable_order: Vec::new(),
            slots: FirstSeen::default(), statics: FirstSeen::default(),
            promoted: FirstSeen::default(), thread_slots: FirstSeen::default(),
            work_fns: Vec::new(), work_rodata: Vec::new(), work_vtables: Vec::new(),
        };
        for root in roots {
            this.mark_fn(root);
        }
        this
    }

    fn collect(state: &AotState, roots: [usize; 2], vmask: &vdce::VtableMask) -> Self {
        let mut this = Self::new(state.functions.len(), roots);
        this.rodata.resize(state.rodata.len(), false);
        this.vtables.resize(state.vtables.len(), false);
        this.drain(state, None, vmask);
        this
    }

    fn mark_fn(&mut self, idx: usize) {
        if idx < self.fns.len() && !self.fns[idx] {
            self.fns[idx] = true;
            self.order.push(idx);
            self.work_fns.push(idx);
        }
    }

    fn mark_sym(&mut self, sym: &SymbolSource) {
        match sym {
            SymbolSource::Slot(idx) => {
                self.slots.insert(*idx);
                self.mark_fn(*idx);
            }
            SymbolSource::Vtable(id) => mark_id(&mut self.vtables, &mut self.vtable_order, &mut self.work_vtables, id.0 as usize),
            SymbolSource::Rodata(id) => mark_id(&mut self.rodata, &mut self.rodata_order, &mut self.work_rodata, id.0 as usize),
            SymbolSource::StaticCell { index, .. } => {
                self.statics.insert(*index);
                self.mark_fn(*index);
            }
            SymbolSource::PromotedCell { owner, seq, .. } => {
                self.promoted.insert((*owner, *seq));
            }
            SymbolSource::ThreadSlot(index) => {
                self.thread_slots.insert(*index);
            }
            SymbolSource::Host(_) | SymbolSource::Word(_) => {}
        }
    }

    fn drain_fns(&mut self, state: &AotState, pending: &mut Option<(&[bool], &mut Vec<usize>)>) {
        while let Some(idx) = self.work_fns.pop() {
            if let Some((compiled, next_wave)) = pending.as_mut()
                && !compiled.get(idx).copied().unwrap_or(true)
            {
                next_wave.push(idx);
                continue;
            }
            let Some(entry) = state.functions.get(idx).and_then(Option::as_ref) else { continue };
            for_each_fn_edge(entry, |edge| match edge {
                FnEdge::Sym(sym, direct) => {
                    if direct && let SymbolSource::Slot(callee) = *sym {
                        self.mark_fn(callee);
                    } else {
                        self.mark_sym(sym);
                    }
                }
                FnEdge::Alias(leader) => self.mark_fn(leader),
            });
        }
    }

    fn drain_rodata_vtables(&mut self, state: &AotState, vmask: &vdce::VtableMask) {
        while let Some(id) = self.work_rodata.pop() {
            for reloc in state.rodata.get(id).into_iter().flat_map(|(_, r)| r) {
                let RodataTarget::Rodata(target) = reloc.target;
                mark_id(&mut self.rodata, &mut self.rodata_order, &mut self.work_rodata, target.0 as usize);
            }
        }
        while let Some(id) = self.work_vtables.pop() {
            for (slot, entry) in state.vtables.get(id).into_iter().flatten().enumerate() {
                if !vmask.slot_live(id, slot) {
                    continue;
                }
                match *entry {
                    VtableEntry::Word(Word::Int(_)) => {}
                    VtableEntry::Word(Word::Code(fn_idx)) => self.mark_fn(fn_idx),
                    VtableEntry::Super(super_id) => mark_id(&mut self.vtables, &mut self.vtable_order, &mut self.work_vtables, super_id.0 as usize),
                }
            }
        }
    }

    fn drain(&mut self, state: &AotState, mut pending: Option<(&[bool], &mut Vec<usize>)>, vmask: &vdce::VtableMask) {
        while !self.work_fns.is_empty() || !self.work_rodata.is_empty() || !self.work_vtables.is_empty() {
            self.drain_fns(state, &mut pending);
            self.drain_rodata_vtables(state, vmask);
        }
        if let Some((_, next_wave)) = pending {
            self.work_fns.extend_from_slice(next_wave);
        }
    }
}

struct SymTable<'a> {
    slot_offs: &'a HashMap<usize, u32>,
    rodata_offs: &'a [Option<u32>],
    vtable_offs: &'a [Option<u32>],
    static_syms: &'a HashMap<usize, Symbol>,
    promoted_syms: &'a HashMap<(usize, u32), Symbol>,
    thread_slots: &'a HashMap<usize, u64>,
}

impl SymTable<'_> {
    fn translate(&self, sym: &SymbolSource) -> Result<Symbol> {
        Ok(match sym {
            SymbolSource::Slot(index) => Symbol::Rodata(*self.slot_offs.get(index).context("missing slot offset")?),
            SymbolSource::Vtable(id) => Symbol::Rodata(self.vtable_offs.get(id.0 as usize).copied().flatten().context("missing vtable offset")?),
            SymbolSource::Rodata(id) => Symbol::Rodata(self.rodata_offs.get(id.0 as usize).copied().flatten().context("missing rodata offset")?),
            SymbolSource::StaticCell { index, .. } => self.static_syms.get(index).cloned().context("missing static cell")?,
            SymbolSource::PromotedCell { owner, seq, .. } => self.promoted_syms.get(&(*owner, *seq)).cloned().context("missing promoted cell")?,
            SymbolSource::Host(host) => Symbol::Host(host.clone()),
            SymbolSource::Word(word) => Symbol::Word(*word),
            SymbolSource::ThreadSlot(index) => Symbol::Word(self.thread_slots.get(index).copied().unwrap_or(*index as u64)),
        })
    }
}

fn emit_dict_bodies(state: &AotState, order: &[usize], exe: &mut Executable) -> Result<HashMap<usize, u32>> {
    let mut dict_bodies: HashMap<usize, u32> = HashMap::default();
    for &idx in order {
        let Some(AotFn::Dict { body, .. }) = &state.functions[idx] else { continue };
        let key = Arc::as_ptr(body) as usize;
        if dict_bodies.contains_key(&key) {
            continue;
        }
        let off = place(exe, body);
        for reloc in body.relocs() {
            let rt = reloc.as_runtime().context("dict body contains non-runtime relocation")?;
            exe.patch(off + reloc.offset, reloc.kind, Symbol::Runtime(rt));
        }
        dict_bodies.insert(key, off);
    }
    Ok(dict_bodies)
}

fn slot_target(fn_offs: &[Option<u32>], foreign_fns: &HashMap<usize, HostSymbol>, idx: usize) -> Symbol {
    if let Some(off) = fn_offs.get(idx).copied().flatten() {
        Symbol::Text(off)
    } else if let Some(host) = foreign_fns.get(&idx) {
        Symbol::Host(host.clone())
    } else {
        Symbol::Host(HostSymbol::Abort)
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum RelocTargetKey {
    SelfCall,
    Fn(usize),
    Sym(Symbol),
}

type IcfKey<'a> = (&'a [u8], Vec<(u32, RelocKind, RelocTargetKey)>);

fn alias_roots(functions: &[Option<AotFn>]) -> Vec<usize> {
    (0..functions.len())
        .map(|mut cur| {
            for _ in 0..functions.len() {
                match functions.get(cur).and_then(Option::as_ref) {
                    Some(AotFn::Alias(next)) if *next < functions.len() => cur = *next,
                    _ => break,
                }
            }
            cur
        })
        .collect()
}

fn foreign_sym(foreign_fns: &HashMap<usize, HostSymbol>, idx: usize) -> Symbol {
    foreign_fns.get(&idx).map_or(Symbol::Host(HostSymbol::Abort), |host| Symbol::Host(host.clone()))
}

fn icf_call_target(self_idx: usize, callee: usize, state: &AotState, roots: &[usize], canon: &[usize]) -> RelocTargetKey {
    let root = roots.get(callee).copied().unwrap_or(callee);
    if state.functions.get(root).and_then(Option::as_ref).is_some() {
        if root == self_idx { RelocTargetKey::SelfCall } else { RelocTargetKey::Fn(canon[root]) }
    } else {
        RelocTargetKey::Sym(foreign_sym(&state.foreign_fns, callee))
    }
}

fn icf_key<'a>(self_idx: usize, code: &'a AotCode, relocs: &[SymbolSource], state: &AotState, syms: &SymTable<'_>, roots: &[usize], canon: &[usize]) -> Result<IcfKey<'a>> {
    let mut sig = Vec::with_capacity(code.relocs().len());
    for reloc in code.relocs() {
        let target = if let Some(rt) = reloc.as_runtime() {
            RelocTargetKey::Sym(Symbol::Runtime(rt))
        } else if matches!(reloc.kind, RelocKind::Arm64Call26 | RelocKind::PcRel32)
            && let SymbolSource::Slot(callee) = relocs[reloc.symbol as usize]
        {
            icf_call_target(self_idx, callee, state, roots, canon)
        } else {
            RelocTargetKey::Sym(syms.translate(&relocs[reloc.symbol as usize])?)
        };
        sig.push((reloc.offset, reloc.kind, target));
    }
    Ok((code.bytes(), sig))
}

fn fold_direct_functions(state: &AotState, live_fns: &[bool], syms: &SymTable<'_>, roots: &[usize]) -> Result<Vec<usize>> {
    let mut canon = roots.to_vec();
    for _ in 0..4 {
        let mut seen: HashMap<IcfKey<'_>, usize> = HashMap::default();
        let mut changed = false;
        let prev = canon.clone();
        for idx in 0..state.functions.len() {
            if !live_fns[idx] || roots[idx] != idx {
                continue;
            }
            let Some(AotFn::Direct { code, relocs }) = &state.functions[idx] else { continue };
            let key = icf_key(idx, code, relocs, state, syms, roots, &prev)?;
            let leader = *seen.entry(key).or_insert(idx);
            if canon[idx] != leader {
                canon[idx] = leader;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    Ok(canon)
}

fn place_dict_fn(target: &Target, syms: &SymTable<'_>, dict_bodies: &HashMap<usize, u32>, (body, relocs): (&Arc<AotCode>, &[SymbolSource]), exe: &mut Executable) -> Result<u32> {
    let dict_off = exe.rodata(&vec![0u8; relocs.len() * 8], 8);
    for (i, reloc) in relocs.iter().enumerate() {
        exe.pointer(true, dict_off + (i * 8) as u32, syms.translate(reloc)?);
    }
    let stub = target.dict_entry_stub(0, 1usize << 40, 0, None)?;
    let [(dict_rel, dict_kind), (body_rel, body_kind)] = target.dict_stub_relocs()?;
    let off = exe.text(&stub);
    let body_off = dict_bodies[&(Arc::as_ptr(body) as usize)];
    exe.patch(off + dict_rel, dict_kind, Symbol::Rodata(dict_off));
    exe.patch(off + body_rel, body_kind, Symbol::Text(body_off));
    Ok(off)
}

fn patch_pending_relocs(state: &AotState, syms: &SymTable<'_>, fn_offs: &[Option<u32>], pending: Vec<(u32, &Arc<AotCode>, &[SymbolSource])>, exe: &mut Executable) -> Result<()> {
    for (off, code, relocs) in pending {
        for reloc in code.relocs() {
            let target_sym = if let Some(rt) = reloc.as_runtime() {
                Symbol::Runtime(rt)
            } else if matches!(reloc.kind, RelocKind::Arm64Call26 | RelocKind::PcRel32)
                && let SymbolSource::Slot(callee_idx) = relocs[reloc.symbol as usize]
            {
                slot_target(fn_offs, &state.foreign_fns, callee_idx)
            } else {
                syms.translate(&relocs[reloc.symbol as usize])?
            };
            exe.patch(off + reloc.offset, reloc.kind, target_sym);
        }
    }
    Ok(())
}

fn emit_functions(state: &AotState, live: &AotReach, target: &Target, syms: &SymTable<'_>, dict_bodies: &HashMap<usize, u32>, exe: &mut Executable) -> Result<Vec<Option<u32>>> {
    let roots = alias_roots(&state.functions);
    let canon = fold_direct_functions(state, &live.fns, syms, &roots)?;
    let mut fn_offs = vec![None; state.functions.len()];
    let mut pending = Vec::new();
    for &idx in &live.order {
        let leader = canon[roots[idx]];
        if fn_offs[leader].is_none() {
            fn_offs[leader] = match &state.functions[leader] {
                Some(AotFn::Direct { code, relocs }) => {
                    let off = place(exe, code);
                    pending.push((off, code, relocs.as_slice()));
                    Some(off)
                }
                Some(AotFn::Dict { body, relocs }) => Some(place_dict_fn(target, syms, dict_bodies, (body, relocs), exe)?),
                Some(AotFn::Alias(_)) | None => None,
            };
        }
        fn_offs[idx] = fn_offs[leader];
    }
    patch_pending_relocs(state, syms, &fn_offs, pending, exe)?;
    Ok(fn_offs)
}

fn emit_rodata_and_vtables(state: &AotState, live: &AotReach, exe: &mut Executable) -> Result<(Vec<Option<u32>>, Vec<Option<u32>>)> {
    let mut rodata_offs = vec![None; state.rodata.len()];
    for &id in &live.rodata_order {
        rodata_offs[id] = Some(exe.rodata(&state.rodata[id].0, 16));
    }
    for &id in &live.rodata_order {
        let Some(base) = rodata_offs[id] else { continue };
        for reloc in &state.rodata[id].1 {
            let RodataTarget::Rodata(target) = reloc.target;
            let target_off = rodata_offs.get(target.0 as usize).copied().flatten().context("missing rodata target")?;
            exe.pointer(true, base + reloc.offset, Symbol::Rodata(target_off));
        }
    }
    let mut vtable_offs = vec![None; state.vtables.len()];
    for &id in &live.vtable_order {
        vtable_offs[id] = Some(exe.rodata(&vec![0u8; state.vtables[id].len() * 8], 8));
    }
    Ok((rodata_offs, vtable_offs))
}

fn emit_static_syms(state: &AotState, live: &AotReach, exe: &mut Executable) -> (HashMap<usize, Symbol>, HashMap<(usize, u32), Symbol>) {
    let mut static_syms = HashMap::default();
    for &index in &live.statics.list {
        let Some(entry) = state.statics.get(&index) else { continue };
        let sym = match entry {
            AotStatic::Local(size) => Symbol::Bss(exe.bss(*size + 1, 16)),
            AotStatic::Foreign { weak: false, host: Some(host) } => Symbol::Host(host.clone()),
            AotStatic::Foreign { weak: false, host: None } => Symbol::Data(exe.data(&[0; 8], 8)),
            AotStatic::Foreign { weak: true, host } => {
                let off = exe.data(&[0; 8], 8);
                if let Some(host) = host {
                    exe.pointer(false, off, Symbol::Host(host.clone()));
                }
                Symbol::Data(off)
            }
        };
        static_syms.insert(index, sym);
    }
    let mut promoted_syms = HashMap::default();
    for &key in &live.promoted.list {
        let Some(&at) = state.promoted_seen.get(&key) else { continue };
        promoted_syms.insert(key, Symbol::Bss(exe.bss(state.promoted[at].1, 16)));
    }
    (static_syms, promoted_syms)
}

fn patch_vtables(state: &AotState, live: &AotReach, vmask: &vdce::VtableMask, (vtable_offs, fn_offs): (&[Option<u32>], &[Option<u32>]), exe: &mut Executable) -> Result<()> {
    for &id in &live.vtable_order {
        let Some(base) = vtable_offs[id] else { continue };
        for (i, entry) in state.vtables[id].iter().enumerate() {
            if !vmask.slot_live(id, i) {
                exe.pointer(true, base + (i * 8) as u32, Symbol::Word(0));
                continue;
            }
            let target_sym = match *entry {
                VtableEntry::Word(Word::Int(val)) => Symbol::Word(val),
                VtableEntry::Word(Word::Code(fn_idx)) => slot_target(fn_offs, &state.foreign_fns, fn_idx),
                VtableEntry::Super(super_id) => Symbol::Rodata(vtable_offs.get(super_id.0 as usize).copied().flatten().context("missing super vtable")?),
            };
            exe.pointer(true, base + (i * 8) as u32, target_sym);
        }
    }
    Ok(())
}

impl Image {
    pub(super) fn record_aot(&self, index: usize, entry: AotFn, followers: &[usize]) {
        if let Some(aot) = self.aot() {
            let mut state = lock(aot);
            state.ensure_len(index + 1);
            let frontier = state.frontier.clone();
            let discovered = frontier.as_ref().map(|q| state.collect_new_targets(&entry, q));
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

    /// Freezes a compiled AOT image into a standalone executable starting at `(lang_start, user_main)`.
    pub(crate) fn to_aot(&self, target: &Target, (lang_start, user_main): (usize, usize), cx: &mut crate::sema::Context<'_>, reached: &crate::sema::reach::Reached) -> Result<Executable> {
        let state = lock(self.aot().context("image has no AOT recording")?);
        let (live, vmask) = vdce::VtableMask::converge(cx, reached, &state.vtables, |vmask| {
            let live = AotReach::collect(&state, [lang_start, user_main], vmask);
            let counts = (live.order.len(), live.vtable_order.len());
            let fns = live.fns.clone();
            (live, fns, counts)
        });
        let mut exe = Executable::new(target);
        let mut slot_offs = HashMap::default();
        for &idx in &live.slots.list {
            slot_offs.insert(idx, exe.rodata(&[0; 8], 8));
        }
        let thread_slots: HashMap<usize, u64> = live.thread_slots.list.iter().enumerate().map(|(rank, &index)| (index, rank as u64)).collect();
        let (rodata_offs, vtable_offs) = emit_rodata_and_vtables(&state, &live, &mut exe)?;
        let (static_syms, promoted_syms) = emit_static_syms(&state, &live, &mut exe);
        let syms = SymTable {
            slot_offs: &slot_offs, rodata_offs: &rodata_offs, vtable_offs: &vtable_offs,
            static_syms: &static_syms, promoted_syms: &promoted_syms, thread_slots: &thread_slots,
        };
        let dict_bodies = emit_dict_bodies(&state, &live.order, &mut exe)?;
        let fn_offs = emit_functions(&state, &live, target, &syms, &dict_bodies, &mut exe)?;
        patch_vtables(&state, &live, &vmask, (&vtable_offs, &fn_offs), &mut exe)?;
        for &idx in &live.slots.list {
            exe.pointer(true, slot_offs[&idx], slot_target(&fn_offs, &state.foreign_fns, idx));
        }
        let Some(start_off) = fn_offs.get(lang_start).copied().flatten() else {
            bail!("`std::rt::lang_start` failed to compile");
        };
        let Some(main_off) = fn_offs.get(user_main).copied().flatten() else {
            bail!("`main` failed to compile");
        };
        exe.entry(start_off, main_off);
        let mut named: HashMap<u32, usize> = HashMap::default();
        for &index in &live.order {
            if let Some(off) = fn_offs[index] {
                named.entry(off).or_insert(index);
            }
        }
        exe.symbols(named.into_iter().map(|(off, index)| (off, reached.list.get(index).map_or_else(|| format!("fn{index}"), |((item, _), _)| cx.qualified(*item)))).collect());
        Ok(exe)
    }
}

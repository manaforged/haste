mod escape;
mod frame;
mod loop_mem;

use rustc_hash::FxHashMap as HashMap;

use crate::cfg::Cfg;
use crate::ir::{AbiParam, AtomicOrd, Binary, Block, BlockData, Convert, Function, Inst, Op, StackSlot, Type, Value};
use crate::reloc::RuntimeSymbol;
pub(crate) use loop_mem::{LoopMem, PromoteBase, PromoteCand};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum MemRoot {
    Slot(StackSlot),
    NoAliasParam(u16),
    ReadonlyParam(u16),
    Global(u32),
    UniqueHeap(Option<Value>),
    ThreadCell(u32),
    CallerLoc,
    Unknown,
}

impl MemRoot {
    /// Whether the root names one known object: a stack slot, a noalias or readonly parameter, or a unique heap allocation.
    pub(crate) fn is_identified(&self) -> bool {
        matches!(self, Self::Slot(_) | Self::NoAliasParam(_) | Self::ReadonlyParam(_) | Self::UniqueHeap(_))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum MemBase {
    Slot(StackSlot),
    Global(u32),
    Addr(Value),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct MemLoc {
    pub root: MemRoot,
    pub base: MemBase,
    pub offset: i64,
    pub size: u32,
}

impl MemLoc {
    pub(crate) fn new(root: MemRoot, base: MemBase, offset: i64, size: u32) -> Self {
        Self { root, base, offset, size }
    }

    pub(crate) fn slot(slot: StackSlot, offset: i64, size: u32) -> Self {
        Self { root: MemRoot::Slot(slot), base: MemBase::Slot(slot), offset, size }
    }

    pub(crate) fn overlaps(self, other: Self) -> bool {
        if self.size == 0 || other.size == 0 {
            return false;
        }
        let (a0, b0) = (i128::from(self.offset), i128::from(other.offset));
        let (a1, b1) = (a0 + i128::from(self.size), b0 + i128::from(other.size));
        a0 < b1 && b0 < a1
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum TrackRoot {
    Unset,
    Set(MemRoot),
    Conflict,
}

impl TrackRoot {
    fn join(self, other: Self) -> Self {
        match (self, other) {
            (Self::Unset, x) | (x, Self::Unset) => x,
            (Self::Set(a), Self::Set(b)) if a == b => Self::Set(a),
            (Self::Set(MemRoot::UniqueHeap(_)), Self::Set(MemRoot::UniqueHeap(_))) => Self::Set(MemRoot::UniqueHeap(None)),
            _ => Self::Conflict,
        }
    }

    fn across_edge(self, entry: &[Value]) -> Self {
        match self {
            Self::Set(MemRoot::UniqueHeap(Some(owner))) if !entry.contains(&owner) => Self::Set(MemRoot::UniqueHeap(None)),
            other => other,
        }
    }
}

#[derive(Default)]
pub(crate) struct Provenance {
    pub slot_offsets: Vec<Option<(StackSlot, i32)>>,
    pub unescaped_slots: Vec<bool>,
    pub addr_unescaped_slots: Vec<bool>,
    pub call_readonly_slots: Vec<bool>,
    pub return_only_slots: Vec<bool>,
    pub noalias_escaped: Vec<bool>,
    pub param_roots: Vec<MemRoot>,
    pub symbol_defs: Vec<Option<u32>>,
    frame_runtime: Vec<Option<RuntimeSymbol>>,
    pub ro_bytes: Vec<u32>,
    /// Values that are parameters of an entry block with no predecessors: addresses the caller made before this frame existed.
    pub incoming: Vec<bool>,
    pub affine: Vec<(Value, i64)>,
    pub spans: Vec<Option<(MemBase, i64, i64)>>,
    frontiers: HashMap<u32, escape::Frontier>,
    inst_at: Vec<(u32, u32)>,
    frame: frame::FrameFacts,
}

fn slot_addresses(func: &Function) -> Vec<Option<(StackSlot, i32)>> {
    func.slot_offsets(|_| None)
        .0
        .into_iter()
        .map(|entry| entry.and_then(|(slot, off)| i32::try_from(off).ok().map(|o| (slot, o))))
        .collect()
}

fn step_track(op: &Op, ty: Type, tracked: &[TrackRoot]) -> TrackRoot {
    if ty != Type::I64 {
        return TrackRoot::Conflict;
    }
    match *op {
        Op::StackAddr(slot, _) => TrackRoot::Set(MemRoot::Slot(slot)),
        Op::Symbol(sym) | Op::SlotAddr(sym) => TrackRoot::Set(MemRoot::Global(sym)),
        Op::Convert(Convert::Bitcast, x) => tracked[x.index()],
        Op::Binary(Binary::Iadd, a, b) => match (tracked[a.index()], tracked[b.index()]) {
            (TrackRoot::Unset, _) | (_, TrackRoot::Unset) => TrackRoot::Unset,
            (TrackRoot::Set(r), TrackRoot::Conflict) | (TrackRoot::Conflict, TrackRoot::Set(r)) => TrackRoot::Set(r),
            _ => TrackRoot::Conflict,
        },
        Op::Binary(Binary::Isub, a, b) => match (tracked[a.index()], tracked[b.index()]) {
            (TrackRoot::Unset, _) | (_, TrackRoot::Unset) => TrackRoot::Unset,
            (TrackRoot::Set(r), TrackRoot::Conflict) => TrackRoot::Set(r),
            _ => TrackRoot::Conflict,
        },
        Op::Select(_, a, b) => tracked[a.index()].join(tracked[b.index()]),
        _ => TrackRoot::Conflict,
    }
}

fn join_track_args(tracked: &mut [TrackRoot], (blocks, entry): (&[BlockData], &[Value]), target: Block, args: &[Value]) -> bool {
    let mut changed = false;
    for (&param, &arg) in blocks[target.index()].params.iter().zip(args) {
        if arg != param {
            let next = tracked[param.index()].join(tracked[arg.index()].across_edge(entry));
            if next != tracked[param.index()] {
                tracked[param.index()] = next;
                changed = true;
            }
        }
    }
    changed
}

fn seed_entry_roots(func: &Function, cfg: &Cfg, tracked: &mut [TrackRoot]) {
    let Some(entry) = cfg.entry else { return };
    for (i, (&param, abi)) in func.blocks[entry.index()].params.iter().zip(&func.signature.params).enumerate() {
        let root = u16::try_from(i).ok().and_then(|idx| {
            if abi.noalias {
                Some(MemRoot::NoAliasParam(idx))
            } else if abi.readonly {
                Some(MemRoot::ReadonlyParam(idx))
            } else {
                None
            }
        });
        tracked[param.index()] = root.map_or(TrackRoot::Conflict, TrackRoot::Set);
    }
}

fn is_derived_i64(op: &Op, ty: Type) -> bool {
    ty == Type::I64 && matches!(op, Op::Convert(Convert::Bitcast, _) | Op::Binary(Binary::Iadd | Binary::Isub, ..) | Op::Select(..))
}

fn track_def(func: &Function, inst: Inst, tracked: &mut [TrackRoot], affine: &[(Value, i64)]) -> bool {
    let data = &func.insts[inst.index()];
    let Some(res) = data.result else { return false };
    let ty = func.ty(res);
    let next = if ty == Type::I64 && func.is_unique_ptr(res) {
        let owner = match data.op {
            Op::Load(addr, _) => affine.get(addr.index()).map(|&(root, _)| root),
            _ => None,
        };
        TrackRoot::Set(MemRoot::UniqueHeap(owner))
    } else {
        step_track(&data.op, ty, tracked)
    };
    let changed = next != tracked[res.index()];
    tracked[res.index()] = next;
    changed
}

fn track_edges(func: &Function, entry: &[Value], inst: Inst, tracked: &mut [TrackRoot]) -> bool {
    let blocks = (func.blocks.as_slice(), entry);
    match &func.insts[inst.index()].op {
        Op::Jump(t, args) => join_track_args(tracked, blocks, *t, args),
        Op::Brif(_, t, ta, f, fa) => join_track_args(tracked, blocks, *t, ta) | join_track_args(tracked, blocks, *f, fa),
        _ => false,
    }
}

/// Gives each unique pointer that a call returns and the function stores into an entry parameter's object that parameter as its owner, since the object predates the allocation.
fn seed_fresh_owners(func: &Function, entry: &[Value], affine: &[(Value, i64)], tracked: &mut [TrackRoot]) {
    let mut called = vec![false; func.types.len()];
    for data in &func.insts {
        if let Op::Call { results, .. } = &data.op {
            results.iter().for_each(|result| called[result.index()] = true);
        }
    }
    for &inst in func.blocks.iter().flat_map(|block| &block.insts) {
        if let Op::Store(value, addr, _) = func.insts[inst.index()].op
            && called[value.index()]
            && func.is_unique_ptr(value)
            && let Some(&(owner, _)) = affine.get(addr.index())
            && entry.contains(&owner)
        {
            tracked[value.index()] = TrackRoot::Set(MemRoot::UniqueHeap(Some(owner)));
        }
    }
}

fn seed_thread_cells(func: &Function, tracked: &mut [TrackRoot]) -> bool {
    let zeros = func.thread_cell_zeros();
    if zeros.is_empty() {
        return false;
    }
    let mut consts = vec![None; func.types.len()];
    for inst in &func.insts {
        if let (Op::Const(imm), Some(res)) = (&inst.op, inst.result) {
            consts[res.index()] = Some(*imm);
        }
    }
    let mut found_caller = false;
    for data in &func.insts {
        if let Some(res) = func.caller_loc_result(&data.op, |v| zeros[v.index()]) {
            tracked[res.index()] = TrackRoot::Set(MemRoot::CallerLoc);
            found_caller = true;
        } else if let Some((res, slot)) = func.thread_cell_slot(&data.op, |v| consts[v.index()]) {
            tracked[res.index()] = TrackRoot::Set(MemRoot::ThreadCell(slot));
        }
    }
    found_caller
}

fn caller_loc_escaped(func: &Function, cfg: &Cfg, tracked: &[TrackRoot]) -> bool {
    let is_cl = |v: Value| tracked[v.index()] == TrackRoot::Set(MemRoot::CallerLoc);
    for &b in &cfg.rpo {
        for &inst in &func.blocks[b.index()].insts {
            let op = &func.insts[inst.index()].op;
            match *op {
                Op::Load(addr, 0) if is_cl(addr) => {}
                Op::Store(val, addr, 0) if is_cl(addr) && !is_cl(val) => {}
                Op::Convert(Convert::Bitcast, _) | Op::Jump(..) | Op::Brif(..) => {}
                _ => {
                    let mut bad = false;
                    op.for_operands(|v| bad |= is_cl(v));
                    if bad {
                        return true;
                    }
                }
            }
        }
    }
    false
}

fn compute_tracked_roots(func: &Function, cfg: &Cfg, affine: &[(Value, i64)]) -> Vec<TrackRoot> {
    let mut tracked = vec![TrackRoot::Unset; func.types.len()];
    seed_entry_roots(func, cfg, &mut tracked);
    let entry = cfg.entry.map_or(&[][..], |block| func.blocks[block.index()].params.as_slice());
    seed_fresh_owners(func, entry, affine, &mut tracked);
    let has_caller_loc = seed_thread_cells(func, &mut tracked);
    let mut steps = Vec::new();
    let mut changed = false;
    for &b in &cfg.rpo {
        let insts = &func.blocks[b.index()].insts;
        for &inst in insts {
            changed |= track_def(func, inst, &mut tracked, affine);
            let data = &func.insts[inst.index()];
            if data.result.is_some_and(|res| is_derived_i64(&data.op, func.ty(res))) {
                steps.push((inst, false));
            }
        }
        if let Some(&last) = insts.last()
            && matches!(func.insts[last.index()].op, Op::Jump(..) | Op::Brif(..))
        {
            changed |= track_edges(func, entry, last, &mut tracked);
            steps.push((last, true));
        }
    }
    let mut rounds = 1;
    while changed && rounds < 64 {
        changed = false;
        for &(inst, edges) in &steps {
            changed |= if edges { track_edges(func, entry, inst, &mut tracked) } else { track_def(func, inst, &mut tracked, affine) };
        }
        rounds += 1;
    }
    if changed {
        tracked.fill(TrackRoot::Conflict);
    } else {
        tracked.iter_mut().filter(|item| **item == TrackRoot::Unset).for_each(|item| *item = TrackRoot::Conflict);
        if has_caller_loc && caller_loc_escaped(func, cfg, &tracked) {
            tracked.iter_mut().filter(|item| **item == TrackRoot::Set(MemRoot::CallerLoc)).for_each(|item| *item = TrackRoot::Conflict);
        }
    }
    tracked
}

fn step_affine(op: &Op, out: &[(Value, i64)], consts: &[Option<i64>], types: &[Type]) -> Option<(Value, i64)> {
    match *op {
        Op::Binary(Binary::Iadd, a, b) => {
            let (base, k) = if let Some(k) = consts[b.index()] {
                (a, k)
            } else {
                (b, consts[a.index()]?)
            };
            let (root, off) = out[base.index()];
            off.checked_add(k).map(|total| (root, total))
        }
        Op::Binary(Binary::Isub, a, b) => {
            let k = consts[b.index()]?.checked_neg()?;
            let (root, off) = out[a.index()];
            off.checked_add(k).map(|total| (root, total))
        }
        Op::Convert(Convert::Bitcast, x) => (types[x.index()] == Type::I64).then_some(out[x.index()]),
        Op::Select(_, a, b) => (out[a.index()] == out[b.index()]).then_some(out[a.index()]),
        _ => None,
    }
}

pub(crate) fn compute_affine(func: &Function) -> Vec<(Value, i64)> {
    let count = func.types.len();
    let mut consts = vec![None; count];
    let mut out: Vec<(Value, i64)> = (0..count as u32).map(|i| (Value(i), 0)).collect();
    let mut steps = Vec::new();
    for inst in func.blocks.iter().flat_map(|b| &b.insts) {
        let data = &func.insts[inst.index()];
        let Some(res) = data.result else { continue };
        if let Op::Const(imm) = data.op {
            consts[res.index()] = Some(imm);
        } else if is_derived_i64(&data.op, func.ty(res)) {
            steps.push((data, res));
        }
    }
    for _ in 0..8 {
        let mut changed = false;
        for &(data, res) in &steps {
            if let Some(cand) = step_affine(&data.op, &out, &consts, &func.types)
                && cand != out[res.index()]
            {
                out[res.index()] = cand;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    out
}

impl Provenance {
    /// The coroutine frame runtime that `callee` names, if any: `Malloc` writes no memory the function can read, and `Free` writes only the block it frees.
    pub(crate) fn frame_runtime(&self, callee: Value) -> Option<RuntimeSymbol> {
        self.frame_runtime.get(callee.index()).copied().flatten()
    }

    fn build(func: &Function, cfg: &Cfg, (is_spill, positions): (Option<&[bool]>, bool), affine: Option<Vec<(Value, i64)>>) -> Self {
        let affine = affine.unwrap_or_else(|| compute_affine(func));
        let tracked = compute_tracked_roots(func, cfg, &affine);
        let mut prov = Self::from_roots(func, cfg, &tracked, affine);
        let esc = escape::classify_uses(func, cfg, &prov.slot_offsets, &tracked, is_spill);
        if positions {
            prov.frontiers = escape::frontiers(func, &esc);
        }
        prov.unescaped_slots = esc.unescaped;
        prov.addr_unescaped_slots = esc.addr_unescaped;
        prov.call_readonly_slots = esc.call_readonly;
        prov.return_only_slots = esc.return_only;
        prov.noalias_escaped = esc.noalias;
        if !prov.frontiers.is_empty() {
            prov.inst_at = vec![(u32::MAX, 0); func.insts.len()];
            for &b in &cfg.rpo {
                for (pos, &inst) in func.blocks[b.index()].insts.iter().enumerate() {
                    prov.inst_at[inst.index()] = (b.index() as u32, pos as u32);
                }
            }
        }
        prov.spans = loop_mem::compute_spans(func, cfg, &prov.slot_offsets, &prov.symbol_defs);
        prov
    }

    fn from_roots(func: &Function, cfg: &Cfg, tracked: &[TrackRoot], affine: Vec<(Value, i64)>) -> Self {
        let slot_offsets = slot_addresses(func);
        let mut param_roots = vec![MemRoot::Unknown; func.types.len()];
        for (i, t) in tracked.iter().enumerate() {
            if let TrackRoot::Set(root) = *t {
                param_roots[i] = root;
            }
        }
        let mut symbol_defs = vec![None; func.types.len()];
        let mut frame_runtime = vec![None; func.types.len()];
        for inst in func.blocks.iter().flat_map(|b| &b.insts) {
            let data = &func.insts[inst.index()];
            match (&data.op, data.result) {
                (Op::Symbol(sym) | Op::SlotAddr(sym), Some(res)) => symbol_defs[res.index()] = Some(*sym),
                (&Op::Runtime(rt @ (RuntimeSymbol::Malloc | RuntimeSymbol::Free)), Some(res)) => frame_runtime[res.index()] = Some(rt),
                _ => {}
            }
        }
        let mut ro_bytes = vec![0_u32; func.types.len()];
        let mut incoming = vec![false; func.types.len()];
        if let Some(entry) = cfg.entry {
            let fresh = cfg.pred_count(entry) == 0;
            for (&param, abi) in func.blocks[entry.index()].params.iter().zip(&func.signature.params) {
                ro_bytes[param.index()] = abi.deref_bytes;
                incoming[param.index()] = fresh;
            }
        }
        Self { slot_offsets, param_roots, symbol_defs, frame_runtime, ro_bytes, incoming, affine, frame: frame::FrameFacts::new(func, cfg), ..Self::default() }
    }

    /// The address facts alone, for `loc` and `ro_bytes`: no escape classification and no loop spans.
    pub(crate) fn addresses(func: &Function, cfg: &Cfg) -> Self {
        let affine = compute_affine(func);
        let tracked = compute_tracked_roots(func, cfg, &affine);
        Self::from_roots(func, cfg, &tracked, affine)
    }

    pub(crate) fn new(func: &Function, cfg: &Cfg) -> Self {
        Self::build(func, cfg, (None, false), None)
    }

    /// Like `new`, taking `affine` when the caller already holds `compute_affine` of this exact function, and also finding where each slot escapes so a call before that point may skip it.
    pub(crate) fn with_affine(func: &Function, cfg: &Cfg, affine: Option<Vec<(Value, i64)>>) -> Self {
        Self::build(func, cfg, (None, true), affine)
    }

    pub(crate) fn compute(func: &Function) -> Self {
        let cfg = func.cfg();
        Self::build(func, &cfg, (None, false), None)
    }

    pub(crate) fn compute_with_spills(func: &Function, is_spill: Option<&[bool]>) -> Self {
        let cfg = func.cfg();
        Self::build(func, &cfg, (is_spill, false), None)
    }

    pub(crate) fn loc(&self, addr: Value, offset: i64, size: u32) -> MemLoc {
        if let Some((slot, base)) = self.slot_offsets.get(addr.index()).copied().flatten()
            && let Some(total) = i64::from(base).checked_add(offset)
        {
            return MemLoc::slot(slot, total, size);
        }
        let (root_val, base_off) = self.affine.get(addr.index()).copied().unwrap_or((addr, 0));
        let (eff_val, eff_off) = match base_off.checked_add(offset) {
            Some(total) => (root_val, total),
            None => (addr, offset),
        };
        if let Some((slot, base)) = self.slot_offsets.get(eff_val.index()).copied().flatten()
            && let Some(total) = i64::from(base).checked_add(eff_off)
        {
            return MemLoc::slot(slot, total, size);
        }
        let base = match self.symbol_defs.get(eff_val.index()).copied().flatten() {
            Some(sym) => MemBase::Global(sym),
            None => MemBase::Addr(eff_val),
        };
        let root = self.param_roots.get(eff_val.index()).copied().unwrap_or(MemRoot::Unknown);
        MemLoc::new(root, base, eff_off, size)
    }

    pub(crate) fn loc_span(&self, loc: MemLoc) -> (MemBase, i128, i128) {
        let size = i128::from(loc.size);
        let off = i128::from(loc.offset);
        if let MemBase::Addr(v) = loc.base
            && let Some((base, blo, bhi)) = self.spans.get(v.index()).copied().flatten()
        {
            return (base, i128::from(blo) + off, i128::from(bhi) + off + size);
        }
        (loc.base, off, off + size)
    }

    /// Whether `slot`'s address may have left the function by the time `at` runs; `None` means anywhere.
    pub(crate) fn slot_escaped_at(&self, slot: StackSlot, at: Option<Inst>) -> bool {
        let (Some(inst), Some(frontier)) = (at, self.frontiers.get(&slot.0)) else { return true };
        match self.inst_at.get(inst.index()) {
            Some(&(block, pos)) if block != u32::MAX => frontier.reached((block, pos)),
            _ => true,
        }
    }

    /// Whether `base` is an address the caller passed in, which cannot point into this frame's slots.
    pub(crate) fn is_incoming(&self, base: MemBase) -> bool {
        matches!(base, MemBase::Addr(v) if self.incoming.get(v.index()).copied().unwrap_or(false))
    }

    pub(crate) fn slot_unescaped(&self, slot: StackSlot) -> bool {
        self.unescaped_slots.get(slot.index()).copied().unwrap_or(false)
    }

    /// Whether `slot`'s address never escapes to normal memory ops or calls, ignoring unwind-drop table reads.
    pub(crate) fn slot_addr_unescaped(&self, slot: StackSlot) -> bool {
        self.addr_unescaped_slots.get(slot.index()).copied().unwrap_or(false)
    }

    pub(crate) fn escaped_slots(&self) -> Vec<bool> {
        self.unescaped_slots.iter().map(|&u| !u).collect()
    }

    pub(crate) fn addr_escaped_slots(&self) -> Vec<bool> {
        self.addr_unescaped_slots.iter().map(|&u| !u).collect()
    }

    /// Whether every call that takes `slot`'s address takes it as the memory for its result, so only those calls write it.
    pub(crate) fn slot_return_only(&self, slot: StackSlot) -> bool {
        self.return_only_slots.get(slot.index()).copied().unwrap_or(false)
    }

    pub(crate) fn slot_call_readonly(&self, slot: StackSlot) -> bool {
        self.call_readonly_slots.get(slot.index()).copied().unwrap_or(false)
    }

    pub(crate) fn noalias_unescaped(&self, param: u16) -> bool {
        !self.noalias_escaped.get(usize::from(param)).copied().unwrap_or(true)
    }
}

pub(crate) fn may_alias(a: MemLoc, b: MemLoc, prov: &Provenance) -> bool {
    if a.size == 0 || b.size == 0 {
        return false;
    }
    if a.base == b.base {
        return a.overlaps(b);
    }
    if let Some(alias) = prov.frame_alias(a, b) {
        return alias;
    }
    if a.size != u32::MAX && b.size != u32::MAX {
        let (ba, alo, ahi) = prov.loc_span(a);
        let (bb, blo, bhi) = prov.loc_span(b);
        if ba == bb {
            return alo < bhi && blo < ahi;
        }
    }
    match (a.root, b.root) {
        (MemRoot::CallerLoc, MemRoot::CallerLoc) => a.overlaps(b),
        (MemRoot::CallerLoc, _) | (_, MemRoot::CallerLoc) => false,
        (MemRoot::ThreadCell(ta), MemRoot::ThreadCell(tb)) => ta == tb,
        (MemRoot::ThreadCell(_), MemRoot::Slot(_) | MemRoot::Global(_) | MemRoot::UniqueHeap(_))
        | (MemRoot::Slot(_) | MemRoot::Global(_) | MemRoot::UniqueHeap(_), MemRoot::ThreadCell(_)) => false,
        (MemRoot::Slot(sa), MemRoot::Slot(sb)) => sa == sb,
        (MemRoot::NoAliasParam(pa), MemRoot::NoAliasParam(pb)) => pa == pb,
        (MemRoot::Global(ga), MemRoot::Global(gb)) => ga == gb,
        (MemRoot::NoAliasParam(_), MemRoot::Slot(_) | MemRoot::Global(_) | MemRoot::ReadonlyParam(_))
        | (MemRoot::Slot(_) | MemRoot::Global(_) | MemRoot::ReadonlyParam(_), MemRoot::NoAliasParam(_))
        | (MemRoot::UniqueHeap(_), MemRoot::Slot(_) | MemRoot::NoAliasParam(_) | MemRoot::Global(_) | MemRoot::ReadonlyParam(_))
        | (MemRoot::Slot(_) | MemRoot::NoAliasParam(_) | MemRoot::Global(_) | MemRoot::ReadonlyParam(_), MemRoot::UniqueHeap(_)) => false,
        (MemRoot::UniqueHeap(Some(owner)), _) if b.base == MemBase::Addr(owner) => false,
        (_, MemRoot::UniqueHeap(Some(owner))) if a.base == MemBase::Addr(owner) => false,
        (MemRoot::NoAliasParam(p), MemRoot::Unknown) | (MemRoot::Unknown, MemRoot::NoAliasParam(p)) => !prov.noalias_unescaped(p),
        (MemRoot::Slot(_), _) if prov.is_incoming(b.base) => false,
        (_, MemRoot::Slot(_)) if prov.is_incoming(a.base) => false,
        (MemRoot::Slot(s), _) | (_, MemRoot::Slot(s)) => !prov.slot_addr_unescaped(s) && !prov.slot_return_only(s),
        _ => true,
    }
}

fn call_writes_noalias(params: &[AbiParam], args: &[Value], prov: &Provenance, p: u16) -> bool {
    args.iter().enumerate().any(|(i, arg)| {
        prov.param_roots.get(arg.index()).copied() == Some(MemRoot::NoAliasParam(p))
            && !params.get(i).is_some_and(|pa| pa.readonly)
    })
}

/// Whether `loc` lies in a read-only global section (`func.rodata_bytes`).
pub(crate) fn is_rodata(func: &Function, loc: MemLoc) -> bool {
    matches!(loc.root, MemRoot::Global(sym) if func.rodata_bytes.iter().any(|&(s, _)| s == sym))
}

/// Whether `[off, off + bytes)` lies within `[0, size)` without underflow or overflow.
pub(crate) fn in_range(off: i64, bytes: u32, size: u32) -> bool {
    off >= 0 && (off as u64).checked_add(u64::from(bytes)).is_some_and(|end| end <= u64::from(size))
}

/// Whether `op` is a `StackStore` whose byte range in `slot` overlaps `[off, off + bytes)`.
pub(crate) fn stack_store_overlaps(func: &Function, op: &Op, slot: StackSlot, (off, bytes): (i32, u32)) -> Option<bool> {
    let Op::StackStore(val, wslot, woff) = *op else { return None };
    Some(wslot == slot && MemLoc::slot(slot, i64::from(off), bytes).overlaps(MemLoc::slot(wslot, i64::from(woff), func.ty(val).bytes())))
}

/// Whether `op` is an atomic memory op or fence.
pub(crate) fn is_atomic_barrier(op: &Op) -> bool {
    matches!(op, Op::AtomicLoad(..) | Op::AtomicStore(..) | Op::AtomicRmw(..) | Op::AtomicCas(..) | Op::Fence(..))
}

/// Whether `op` prevents hoisting, sinking, or rematerializing memory accesses across it.
pub(crate) fn is_hard_barrier(op: &Op) -> bool {
    is_atomic_barrier(op)
        || matches!(op, Op::Call { .. } | Op::TryCall { .. } | Op::Trap | Op::Trapz(_) | Op::Assume(_)
            | Op::Binary(Binary::Sdiv | Binary::Udiv | Binary::Srem | Binary::Urem, ..))
}

fn stack_store_clobbers(func: &Function, op: &Op, (val, slot, off): (Value, StackSlot, i32), loc: MemLoc, prov: &Provenance) -> bool {
    if let MemBase::Slot(lslot) = loc.base && loc.root == MemRoot::Slot(lslot)
        && let Ok(loff) = i32::try_from(loc.offset) && let Some(hit) = stack_store_overlaps(func, op, lslot, (loff, loc.size))
    {
        return hit;
    }
    may_alias(MemLoc::slot(slot, i64::from(off), func.ty(val).bytes()), loc, prov)
}

pub(crate) fn clobbers(func: &Function, op: &Op, loc: MemLoc, prov: &Provenance) -> bool {
    clobbers_op(func, (op, None), loc, prov)
}

/// Whether a write of unknown length at `addr` may overwrite `loc`.
fn block_clobbers(addr: Value, loc: MemLoc, prov: &Provenance) -> bool {
    let w = prov.loc(addr, 0, u32::MAX);
    w.base == loc.base || (w.root != MemRoot::Unknown && w.root == loc.root) || may_alias(w, loc, prov)
}

/// Whether `op`, run as instruction `at` when known, may overwrite `loc`; `at` rules out slots whose address leaves only later.
pub(crate) fn clobbers_op(func: &Function, (op, at): (&Op, Option<Inst>), loc: MemLoc, prov: &Provenance) -> bool {
    match *op {
        Op::StackStore(val, slot, off) => stack_store_clobbers(func, op, (val, slot, off), loc, prov),
        Op::Store(val, addr, off) => may_alias(prov.loc(addr, i64::from(off), func.ty(val).bytes()), loc, prov),
        Op::Move(dst, _, bytes) => may_alias(prov.loc(dst, 0, bytes), loc, prov),
        Op::Copy(dst, _, _) => block_clobbers(dst, loc, prov),
        Op::Call { callee, ref args, .. } if let Some(rt) = prov.frame_runtime(callee) => {
            loc.root != MemRoot::CallerLoc && rt == RuntimeSymbol::Free && args.first().is_none_or(|&freed| block_clobbers(freed, loc, prov))
        }
        Op::Call { sig, ref args, .. } | Op::TryCall { sig, ref args, .. } => match loc.root {
            MemRoot::Slot(s) => {
                if matches!(*op, Op::TryCall { .. }) {
                    !prov.slot_unescaped(s) && !prov.slot_call_readonly(s)
                } else {
                    !prov.slot_addr_unescaped(s)
                        && prov.slot_escaped_at(s, at)
                        && !prov.slot_call_readonly(s)
                        && (!prov.slot_return_only(s) || args.iter().any(|&arg| prov.loc(arg, 0, 1).root == MemRoot::Slot(s)))
                }
            }
            MemRoot::NoAliasParam(p) => {
                !prov.noalias_unescaped(p) || call_writes_noalias(&func.sigs[sig.index()].params, args, prov, p)
            }
            _ => matches!(*op, Op::TryCall { .. }) || !prov.private_frame(loc),
        },
        _ => is_atomic_barrier(op)
            && loc.root != MemRoot::CallerLoc
            && !(prov.private_frame(loc) && !prov.touches_frame(op))
            && !matches!(loc.root, MemRoot::Slot(s) if prov.slot_addr_unescaped(s) || !prov.slot_escaped_at(s, at)),
    }
}

/// Whether `op` may overwrite `loc` for load forwarding; `Relaxed` and `Release` atomics cannot acquire writes from other threads.
pub(crate) fn clobbers_load(func: &Function, op: (&Op, Option<Inst>), loc: MemLoc, prov: &Provenance) -> bool {
    match *op.0 {
        Op::AtomicLoad(_, AtomicOrd::Relaxed) => false,
        Op::AtomicStore(val, addr, AtomicOrd::Relaxed | AtomicOrd::Release)
        | Op::AtomicRmw(_, AtomicOrd::Relaxed | AtomicOrd::Release, addr, val)
        | Op::AtomicCas(AtomicOrd::Relaxed | AtomicOrd::Release, addr, _, val) => {
            !matches!(loc.root, MemRoot::Slot(s) if prov.slot_addr_unescaped(s) || !prov.slot_escaped_at(s, op.1))
                && may_alias(prov.loc(addr, 0, func.ty(val).bytes()), loc, prov)
        }
        _ => clobbers_op(func, op, loc, prov),
    }
}

#[derive(Clone)]
pub(crate) struct KnownLoads {
    by_base: HashMap<MemBase, (MemRoot, Vec<(i64, Type, Value)>)>,
}

impl KnownLoads {
    pub(crate) fn new() -> Self {
        Self { by_base: HashMap::default() }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.by_base.is_empty()
    }

    pub(crate) fn get(&self, base: MemBase, offset: i64, ty: Type) -> Option<Value> {
        self.by_base.get(&base)?.1.iter().find(|&&(off, t, _)| off == offset && t == ty).map(|&(_, _, val)| val)
    }

    pub(crate) fn entries_for(&self, base: MemBase) -> &[(i64, Type, Value)] {
        self.by_base.get(&base).map_or(&[], |entry| entry.1.as_slice())
    }

    pub(crate) fn remember(&mut self, loc: MemLoc, ty: Type, value: Value) {
        let (_, list) = self.by_base.entry(loc.base).or_insert_with(|| (loc.root, Vec::new()));
        if let Some(entry) = list.iter_mut().find(|(off, t, _)| *off == loc.offset && *t == ty) {
            entry.2 = value;
        } else {
            list.push((loc.offset, ty, value));
        }
    }

    pub(crate) fn write(&mut self, loc: MemLoc, ty: Type, value: Value, prov: &Provenance) {
        self.by_base.retain(|&base, (root, entries)| {
            entries.retain(|&(off, t, _)| !may_alias(loc, MemLoc::new(*root, base, off, t.bytes()), prov));
            !entries.is_empty()
        });
        self.remember(loc, ty, value);
    }

    pub(crate) fn invalidate(&mut self, func: &Function, op: (&Op, Option<Inst>), prov: &Provenance) {
        self.by_base.retain(|&base, (root, entries)| {
            entries.retain(|&(off, t, _)| !clobbers_load(func, op, MemLoc::new(*root, base, off, t.bytes()), prov));
            !entries.is_empty()
        });
    }
}

mod loop_mem;

use rustc_hash::FxHashMap as HashMap;

use crate::cfg::Cfg;
use crate::ir::{Binary, Block, BlockData, Convert, Function, Op, StackSlot, Type, Value};
pub(crate) use loop_mem::{LoopMem, PromoteBase, PromoteCand};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum MemRoot {
    Slot(StackSlot),
    NoAliasParam(u16),
    ReadonlyParam(u16),
    Global(u32),
    Unknown,
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
enum TrackRoot {
    Unset,
    Set(MemRoot),
    Conflict,
}

impl TrackRoot {
    fn join(self, other: Self) -> Self {
        match (self, other) {
            (Self::Unset, x) | (x, Self::Unset) => x,
            (Self::Set(a), Self::Set(b)) if a == b => Self::Set(a),
            _ => Self::Conflict,
        }
    }
}

pub(crate) struct Provenance {
    pub slot_offsets: Vec<Option<(StackSlot, i32)>>,
    pub unescaped_slots: Vec<bool>,
    pub call_readonly_slots: Vec<bool>,
    pub noalias_escaped: Vec<bool>,
    pub param_roots: Vec<MemRoot>,
    pub symbol_defs: Vec<Option<u32>>,
    pub ro_bytes: Vec<u32>,
    pub affine: Vec<(Value, i64)>,
}

struct UseCounts {
    total: Vec<u32>,
    safe: Vec<u32>,
    ro_calls: Vec<u32>,
}

struct SlotEscapes {
    unescaped: Vec<bool>,
    call_readonly: Vec<bool>,
    noalias: Vec<bool>,
}

fn slot_addresses(func: &Function) -> Vec<Option<(StackSlot, i32)>> {
    func.slot_offsets(|_| None)
        .0
        .into_iter()
        .map(|entry| entry.and_then(|(slot, off)| i32::try_from(off).ok().map(|o| (slot, o))))
        .collect()
}

fn finish_slot_escape(
    func: &Function,
    slots: &[Option<(StackSlot, i32)>],
    tracked: &[TrackRoot],
    counts: &UseCounts,
    is_spill: Option<&[bool]>,
) -> SlotEscapes {
    let mut unescaped = vec![true; func.slots.len()];
    let mut call_readonly = vec![true; func.slots.len()];
    let mut noalias = vec![false; func.signature.params.len()];
    for (v, slot) in slots.iter().enumerate() {
        let resolved = slot.map(|(s, _)| s).or(match tracked[v] {
            TrackRoot::Set(MemRoot::Slot(s)) => Some(s),
            _ => None,
        });
        if let Some(s) = resolved {
            let spilled = is_spill.is_some_and(|sp| sp.get(v).copied().unwrap_or(false));
            if spilled || counts.safe[v] != counts.total[v] {
                unescaped[s.index()] = false;
            }
            if spilled || counts.safe[v] + counts.ro_calls[v] != counts.total[v] {
                call_readonly[s.index()] = false;
            }
        }
        if let TrackRoot::Set(MemRoot::NoAliasParam(p)) = tracked[v]
            && counts.safe[v] != counts.total[v]
        {
            noalias[usize::from(p)] = true;
        }
    }
    SlotEscapes { unescaped, call_readonly, noalias }
}

fn count_safe_args(func: &Function, target: Block, args: &[Value], tracked: &[TrackRoot], safe: &mut [u32]) {
    for (&param, &arg) in func.blocks[target.index()].params.iter().zip(args) {
        if let TrackRoot::Set(root) = tracked[param.index()]
            && tracked[arg.index()] == TrackRoot::Set(root)
        {
            safe[arg.index()] += 1;
        }
    }
}

fn classify_uses(
    func: &Function,
    cfg: &Cfg,
    slots: &[Option<(StackSlot, i32)>],
    tracked: &[TrackRoot],
    is_spill: Option<&[bool]>,
) -> SlotEscapes {
    let n = func.types.len();
    let mut is_const = vec![false; n];
    for &b in &cfg.rpo {
        for &inst in &func.blocks[b.index()].insts {
            let data = &func.insts[inst.index()];
            if let (&Op::Const(_), Some(res)) = (&data.op, data.result) {
                is_const[res.index()] = true;
            }
        }
    }
    let mut counts = UseCounts { total: vec![0_u32; n], safe: vec![0_u32; n], ro_calls: vec![0_u32; n] };
    let mark_safe = |v: Value, safe: &mut [u32]| {
        if slots[v.index()].is_some() || matches!(tracked[v.index()], TrackRoot::Set(_)) {
            safe[v.index()] += 1;
        }
    };
    let mark_param_safe = |v: Value, safe: &mut [u32]| {
        if slots[v.index()].is_none()
            && matches!(tracked[v.index()], TrackRoot::Set(MemRoot::NoAliasParam(_) | MemRoot::ReadonlyParam(_)))
        {
            safe[v.index()] += 1;
        }
    };
    for &b in &cfg.rpo {
        for &inst in &func.blocks[b.index()].insts {
            let data = &func.insts[inst.index()];
            data.op.for_operands(|v| counts.total[v.index()] += 1);
            match &data.op {
                Op::Load(addr, _) => mark_safe(*addr, &mut counts.safe),
                Op::Trapz(addr) => mark_param_safe(*addr, &mut counts.safe),
                Op::Store(val, addr, _) if val != addr => mark_safe(*addr, &mut counts.safe),
                Op::Move(dst, src, _) | Op::Copy(dst, src, _) => {
                    mark_param_safe(*dst, &mut counts.safe);
                    mark_param_safe(*src, &mut counts.safe);
                }
                Op::Binary(Binary::Iadd | Binary::Isub, a, b) => {
                    if data.result.is_some_and(|r| slots[r.index()].is_some()) {
                        let target = if slots[a.index()].is_some() { a } else { b };
                        counts.safe[target.index()] += 1;
                    } else if data.result.is_some_and(|r| matches!(tracked[r.index()], TrackRoot::Set(_))) {
                        let target = if matches!(tracked[a.index()], TrackRoot::Set(_)) { a } else { b };
                        counts.safe[target.index()] += 1;
                    }
                }
                Op::Convert(Convert::Bitcast, x) if data.result.is_some_and(|r| matches!(tracked[r.index()], TrackRoot::Set(_))) => {
                    counts.safe[x.index()] += 1;
                }
                Op::Select(_, a, b) => {
                    if let Some(r) = data.result
                        && let TrackRoot::Set(root) = tracked[r.index()]
                    {
                        for v in [a, b] {
                            if tracked[v.index()] == TrackRoot::Set(root) {
                                counts.safe[v.index()] += 1;
                            }
                        }
                    }
                }
                Op::Icmp(_, a, b) => {
                    for (v, other) in [(a, b), (b, a)] {
                        let is_noalias = matches!(tracked[v.index()], TrackRoot::Set(MemRoot::NoAliasParam(_)));
                        let same_or_const = (slots[v.index()].is_some() || matches!(tracked[v.index()], TrackRoot::Set(_)))
                            && (is_const[other.index()]
                                || matches!((tracked[v.index()], tracked[other.index()]), (TrackRoot::Set(ra), TrackRoot::Set(rb)) if ra == rb));
                        if is_noalias || same_or_const {
                            counts.safe[v.index()] += 1;
                        }
                    }
                }
                Op::Jump(target, args) => count_safe_args(func, *target, args, tracked, &mut counts.safe),
                Op::Brif(_, then_b, then_args, else_b, else_args) => {
                    count_safe_args(func, *then_b, then_args, tracked, &mut counts.safe);
                    count_safe_args(func, *else_b, else_args, tracked, &mut counts.safe);
                }
                Op::Call { sig, args, .. } | Op::TryCall { sig, args, .. } => {
                    let params = &func.sigs[sig.index()].params;
                    for (i, arg) in args.iter().enumerate() {
                        if params.get(i).is_some_and(|p| p.readonly) {
                            counts.ro_calls[arg.index()] += 1;
                        }
                    }
                }
                _ => {}
            }
        }
    }
    finish_slot_escape(func, slots, tracked, &counts, is_spill)
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

fn join_track_args(tracked: &mut [TrackRoot], blocks: &[BlockData], target: Block, args: &[Value]) -> bool {
    let mut changed = false;
    for (&param, &arg) in blocks[target.index()].params.iter().zip(args) {
        if arg != param {
            let next = tracked[param.index()].join(tracked[arg.index()]);
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

fn compute_tracked_roots(func: &Function, cfg: &Cfg) -> Vec<TrackRoot> {
    let mut tracked = vec![TrackRoot::Unset; func.types.len()];
    seed_entry_roots(func, cfg, &mut tracked);
    let mut converged = false;
    for _ in 0..64 {
        let mut changed = false;
        for &b in &cfg.rpo {
            let block = &func.blocks[b.index()];
            for &inst in &block.insts {
                let data = &func.insts[inst.index()];
                if let Some(res) = data.result {
                    let next = step_track(&data.op, func.ty(res), &tracked);
                    if next != tracked[res.index()] {
                        tracked[res.index()] = next;
                        changed = true;
                    }
                }
            }
            match block.insts.last().map(|i| &func.insts[i.index()].op) {
                Some(Op::Jump(t, args)) => changed |= join_track_args(&mut tracked, &func.blocks, *t, args),
                Some(Op::Brif(_, t, ta, f, fa)) => {
                    changed |= join_track_args(&mut tracked, &func.blocks, *t, ta);
                    changed |= join_track_args(&mut tracked, &func.blocks, *f, fa);
                }
                _ => {}
            }
        }
        if !changed {
            converged = true;
            break;
        }
    }
    if !converged {
        tracked.fill(TrackRoot::Conflict);
    } else {
        tracked.iter_mut().filter(|item| **item == TrackRoot::Unset).for_each(|item| *item = TrackRoot::Conflict);
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
    for inst in func.blocks.iter().flat_map(|b| &b.insts) {
        let data = &func.insts[inst.index()];
        if let (&Op::Const(imm), Some(res)) = (&data.op, data.result) {
            consts[res.index()] = Some(imm);
        }
    }
    for _ in 0..8 {
        let mut changed = false;
        for inst in func.blocks.iter().flat_map(|b| &b.insts) {
            let data = &func.insts[inst.index()];
            let Some(res) = data.result else { continue };
            if func.ty(res) != Type::I64 {
                continue;
            }
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
    fn build(func: &Function, cfg: &Cfg, is_spill: Option<&[bool]>) -> Self {
        let slot_offsets = slot_addresses(func);
        let tracked = compute_tracked_roots(func, cfg);
        let esc = classify_uses(func, cfg, &slot_offsets, &tracked, is_spill);
        let mut param_roots = vec![MemRoot::Unknown; func.types.len()];
        for (i, t) in tracked.iter().enumerate() {
            if let TrackRoot::Set(root) = *t {
                param_roots[i] = root;
            }
        }
        let mut symbol_defs = vec![None; func.types.len()];
        for inst in func.blocks.iter().flat_map(|b| &b.insts) {
            let data = &func.insts[inst.index()];
            if let (Op::Symbol(sym) | Op::SlotAddr(sym), Some(res)) = (&data.op, data.result) {
                symbol_defs[res.index()] = Some(*sym);
            }
        }
        let mut ro_bytes = vec![0_u32; func.types.len()];
        if let Some(entry) = cfg.entry {
            for (&param, abi) in func.blocks[entry.index()].params.iter().zip(&func.signature.params) {
                ro_bytes[param.index()] = abi.deref_bytes;
            }
        }
        let affine = compute_affine(func);
        Self {
            slot_offsets,
            unescaped_slots: esc.unescaped,
            call_readonly_slots: esc.call_readonly,
            noalias_escaped: esc.noalias,
            param_roots,
            symbol_defs,
            ro_bytes,
            affine,
        }
    }

    pub(crate) fn new(func: &Function, cfg: &Cfg) -> Self {
        Self::build(func, cfg, None)
    }

    pub(crate) fn compute(func: &Function) -> Self {
        let cfg = func.cfg();
        Self::build(func, &cfg, None)
    }

    pub(crate) fn compute_with_spills(func: &Function, is_spill: Option<&[bool]>) -> Self {
        let cfg = func.cfg();
        Self::build(func, &cfg, is_spill)
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

    pub(crate) fn slot_unescaped(&self, slot: StackSlot) -> bool {
        self.unescaped_slots.get(slot.index()).copied().unwrap_or(false)
    }

    pub(crate) fn escaped_slots(&self) -> Vec<bool> {
        self.unescaped_slots.iter().map(|&u| !u).collect()
    }

    pub(crate) fn slot_call_readonly(&self, slot: StackSlot) -> bool {
        self.call_readonly_slots.get(slot.index()).copied().unwrap_or(false)
    }

    pub(crate) fn noalias_unescaped(&self, param: u16) -> bool {
        !self.noalias_escaped.get(usize::from(param)).copied().unwrap_or(true)
    }
}

pub(crate) fn may_alias(a: MemLoc, b: MemLoc, prov: &Provenance) -> bool {
    if a.base == b.base {
        return a.overlaps(b);
    }
    match (a.root, b.root) {
        (MemRoot::Slot(sa), MemRoot::Slot(sb)) => sa == sb,
        (MemRoot::NoAliasParam(pa), MemRoot::NoAliasParam(pb)) => pa == pb,
        (MemRoot::Global(ga), MemRoot::Global(gb)) => ga == gb,
        (MemRoot::NoAliasParam(_), MemRoot::Slot(_) | MemRoot::Global(_) | MemRoot::ReadonlyParam(_))
        | (MemRoot::Slot(_) | MemRoot::Global(_) | MemRoot::ReadonlyParam(_), MemRoot::NoAliasParam(_)) => false,
        (MemRoot::NoAliasParam(p), MemRoot::Unknown) | (MemRoot::Unknown, MemRoot::NoAliasParam(p)) => !prov.noalias_unescaped(p),
        (MemRoot::Slot(s), _) | (_, MemRoot::Slot(s)) => !prov.slot_unescaped(s),
        _ => true,
    }
}

pub(crate) fn clobbers(func: &Function, op: &Op, loc: MemLoc, prov: &Provenance) -> bool {
    match *op {
        Op::StackStore(val, slot, off) => may_alias(MemLoc::slot(slot, i64::from(off), func.ty(val).bytes()), loc, prov),
        Op::Store(val, addr, off) => may_alias(prov.loc(addr, i64::from(off), func.ty(val).bytes()), loc, prov),
        Op::Move(dst, _, bytes) => may_alias(prov.loc(dst, 0, bytes), loc, prov),
        Op::Copy(dst, _, _) => {
            let w = prov.loc(dst, 0, u32::MAX);
            w.base == loc.base || (w.root != MemRoot::Unknown && w.root == loc.root) || may_alias(w, loc, prov)
        }
        Op::Call { .. } | Op::TryCall { .. } => match loc.root {
            MemRoot::Slot(s) => !prov.slot_unescaped(s) && !prov.slot_call_readonly(s),
            MemRoot::NoAliasParam(p) => !prov.noalias_unescaped(p),
            _ => true,
        },
        Op::AtomicLoad(_) | Op::AtomicStore(..) | Op::AtomicRmw(..) | Op::AtomicCas(..) | Op::Fence => true,
        _ => false,
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

    pub(crate) fn invalidate(&mut self, func: &Function, op: &Op, prov: &Provenance) {
        self.by_base.retain(|&base, (root, entries)| {
            entries.retain(|&(off, t, _)| !clobbers(func, op, MemLoc::new(*root, base, off, t.bytes()), prov));
            !entries.is_empty()
        });
    }
}

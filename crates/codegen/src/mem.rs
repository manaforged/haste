use rustc_hash::FxHashMap as HashMap;

use crate::cfg::{Cfg, Loop};
use crate::ir::{Binary, Block, BlockData, Function, Inst, Op, StackSlot, Type, Value};

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
    (total, safe, ro_calls): (&[u32], &[u32], &[u32]),
) -> (Vec<bool>, Vec<bool>, Vec<bool>) {
    let mut unescaped_slots = vec![true; func.slots.len()];
    let mut call_readonly_slots = vec![true; func.slots.len()];
    let mut noalias_escaped = vec![false; func.signature.params.len()];
    for (v, slot) in slots.iter().enumerate() {
        let resolved = slot.map(|(s, _)| s).or(match tracked[v] {
            TrackRoot::Set(MemRoot::Slot(s)) => Some(s),
            _ => None,
        });
        if let Some(s) = resolved {
            if safe[v] != total[v] {
                unescaped_slots[s.index()] = false;
            }
            if safe[v] + ro_calls[v] != total[v] {
                call_readonly_slots[s.index()] = false;
            }
        }
        if let TrackRoot::Set(MemRoot::NoAliasParam(p)) = tracked[v]
            && safe[v] != total[v]
        {
            noalias_escaped[usize::from(p)] = true;
        }
    }
    (unescaped_slots, call_readonly_slots, noalias_escaped)
}

fn classify_uses(
    func: &Function,
    slots: &[Option<(StackSlot, i32)>],
    tracked: &[TrackRoot],
) -> (Vec<bool>, Vec<bool>, Vec<bool>) {
    let n = func.types.len();
    let (mut total, mut safe, mut ro_calls) = (vec![0_u32; n], vec![0_u32; n], vec![0_u32; n]);
    for inst in func.blocks.iter().flat_map(|b| &b.insts) {
        let data = &func.insts[inst.index()];
        data.op.for_operands(|v| total[v.index()] += 1);
        match &data.op {
            Op::Load(addr, _) => safe[addr.index()] += 1,
            Op::Store(val, addr, _) if val != addr => safe[addr.index()] += 1,
            Op::Move(to, from, _) => {
                safe[to.index()] += 1;
                safe[from.index()] += 1;
            }
            Op::Binary(Binary::Iadd | Binary::Isub, a, b) => {
                let res_tracked = data.result.is_some_and(|r| slots[r.index()].is_some() || matches!(tracked[r.index()], TrackRoot::Set(_)));
                if res_tracked {
                    if slots[a.index()].is_some() || matches!(tracked[a.index()], TrackRoot::Set(_)) {
                        safe[a.index()] += 1;
                    } else {
                        safe[b.index()] += 1;
                    }
                }
            }
            Op::Call { sig, args, .. } | Op::TryCall { sig, args, .. } => {
                let params = &func.sigs[sig.index()].params;
                for (i, arg) in args.iter().enumerate() {
                    if params.get(i).is_some_and(|p| p.readonly) {
                        ro_calls[arg.index()] += 1;
                    }
                }
            }
            _ => {}
        }
    }
    finish_slot_escape(func, slots, tracked, (&total, &safe, &ro_calls))
}

fn step_track(op: &Op, ty: Type, tracked: &[TrackRoot]) -> TrackRoot {
    if ty != Type::I64 {
        return TrackRoot::Conflict;
    }
    match *op {
        Op::StackAddr(slot, _) => TrackRoot::Set(MemRoot::Slot(slot)),
        Op::Symbol(sym) | Op::SlotAddr(sym) => TrackRoot::Set(MemRoot::Global(sym)),
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

fn step_affine(op: &Op, out: &[(Value, i64)], consts: &[Option<i64>]) -> Option<(Value, i64)> {
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
            if let Some(cand) = step_affine(&data.op, &out, &consts)
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
    pub(crate) fn new(func: &Function, cfg: &Cfg) -> Self {
        let slot_offsets = slot_addresses(func);
        let tracked = compute_tracked_roots(func, cfg);
        let (unescaped_slots, call_readonly_slots, noalias_escaped) = classify_uses(func, &slot_offsets, &tracked);
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
        Self { slot_offsets, unescaped_slots, call_readonly_slots, noalias_escaped, param_roots, symbol_defs, ro_bytes, affine }
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

pub(crate) struct LoopMem {
    pub writes: bool,
    pub unknown_mem_writes: bool,
    pub calls: bool,
    pub param_writes: Vec<bool>,
    pub slot_clobbered: Vec<bool>,
    pub slot_write_ranges: Vec<Vec<(i64, i64)>>,
}

impl LoopMem {
    pub(crate) fn analyze(func: &Function, each: &Loop, prov: &Provenance) -> Self {
        let mut fx = Self {
            writes: false,
            unknown_mem_writes: false,
            calls: false,
            param_writes: vec![false; func.signature.params.len()],
            slot_clobbered: vec![false; func.slots.len()],
            slot_write_ranges: vec![Vec::new(); func.slots.len()],
        };
        for inst in (0..each.body.len()).filter(|b| each.body[*b]).flat_map(|b| &func.blocks[b].insts) {
            let op = &func.insts[inst.index()].op;
            let is_write = matches!(
                op,
                Op::Store(..)
                    | Op::StackStore(..)
                    | Op::Copy(..)
                    | Op::Move(..)
                    | Op::AtomicLoad(_)
                    | Op::AtomicStore(..)
                    | Op::AtomicRmw(..)
                    | Op::AtomicCas(..)
                    | Op::Fence
                    | Op::Call { .. }
                    | Op::TryCall { .. }
            );
            if is_write {
                fx.writes = true;
            }
            if matches!(op, Op::Call { .. } | Op::TryCall { .. }) {
                fx.calls = true;
            }
            match *op {
                Op::StackStore(val, slot, off) => fx.record_slot(slot, i64::from(off), func.ty(val).bytes()),
                Op::Store(val, addr, off) => fx.record_loc(prov.loc(addr, i64::from(off), func.ty(val).bytes())),
                Op::Move(addr, _, bytes) => fx.record_clobber_loc(prov.loc(addr, 0, bytes)),
                Op::Copy(addr, _, _) => fx.record_clobber_loc(prov.loc(addr, 0, u32::MAX)),
                Op::AtomicLoad(_) | Op::AtomicStore(..) | Op::AtomicRmw(..) | Op::AtomicCas(..) | Op::Fence | Op::Call { .. } | Op::TryCall { .. } => {
                    fx.unknown_mem_writes = true;
                }
                _ => {}
            }
        }
        fx
    }

    fn record_slot(&mut self, slot: StackSlot, lo: i64, bytes: u32) {
        if lo < 0 {
            self.slot_clobbered[slot.index()] = true;
        } else {
            self.slot_write_ranges[slot.index()].push((lo, lo + i64::from(bytes)));
        }
    }

    fn record_loc(&mut self, loc: MemLoc) {
        if let MemBase::Slot(s) = loc.base {
            self.record_slot(s, loc.offset, loc.size);
            return;
        }
        match loc.root {
            MemRoot::Slot(s) => self.slot_clobbered[s.index()] = true,
            MemRoot::NoAliasParam(p) => self.param_writes[usize::from(p)] = true,
            _ => self.unknown_mem_writes = true,
        }
    }

    fn record_clobber_loc(&mut self, loc: MemLoc) {
        match loc.root {
            MemRoot::Slot(s) => self.slot_clobbered[s.index()] = true,
            MemRoot::NoAliasParam(p) => self.param_writes[usize::from(p)] = true,
            _ => self.unknown_mem_writes = true,
        }
    }

    fn slot_load_ok(&self, func: &Function, prov: &Provenance, slot: StackSlot, total: i64, width: u32) -> bool {
        let end_ok = total >= 0 && (total as u64).checked_add(u64::from(width)).is_some_and(|end| end <= u64::from(func.slots[slot.index()].size));
        let mut_ok = prov.slot_call_readonly(slot) || (!self.calls && !self.unknown_mem_writes);
        let written = self.slot_clobbered[slot.index()]
            || self.slot_write_ranges[slot.index()].iter().any(|&(wlo, whi)| total < whi && wlo < total + i64::from(width));
        end_ok && mut_ok && !written
    }

    pub(crate) fn can_hoist_load(&self, func: &Function, prov: &Provenance, head_prefix: bool, inst: Inst) -> bool {
        let data = &func.insts[inst.index()];
        let Some(res) = data.result else { return false };
        let width = func.ty(res).bytes();
        match data.op {
            Op::StackLoad(slot, offset) => self.slot_load_ok(func, prov, slot, i64::from(offset), width),
            Op::Load(addr, offset) => {
                let loc = prov.loc(addr, i64::from(offset), width);
                if let MemBase::Slot(slot) = loc.base {
                    return self.slot_load_ok(func, prov, slot, loc.offset, width);
                }
                let deref_ok = !self.unknown_mem_writes
                    && offset >= 0
                    && (offset as u64) + u64::from(width) <= u64::from(prov.ro_bytes.get(addr.index()).copied().unwrap_or(0));
                let root_ok = match loc.root {
                    MemRoot::Slot(slot) => {
                        prov.slot_call_readonly(slot)
                            && !self.slot_clobbered[slot.index()]
                            && self.slot_write_ranges[slot.index()].is_empty()
                    }
                    MemRoot::NoAliasParam(p) => !self.unknown_mem_writes && !self.param_writes.get(usize::from(p)).copied().unwrap_or(true),
                    MemRoot::ReadonlyParam(_) => !self.unknown_mem_writes,
                    _ => false,
                };
                !self.calls && head_prefix && (!self.writes || root_ok || deref_ok)
            }
            _ => false,
        }
    }
}

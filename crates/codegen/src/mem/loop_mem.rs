use crate::cfg::{Cfg, Loop};
use crate::ir::{Binary, Block, Convert, Function, Inst, Op, SigRef, StackSlot, Type, Value};

use super::{MemBase, MemLoc, MemRoot, Provenance, in_range, is_atomic_barrier, may_alias};

pub(super) const SPAN_LIMIT: i64 = 1_i64 << 60;

fn mask_hi(hi: i64) -> Option<i64> {
    let u = u64::try_from(hi).ok()?;
    let m = if u == 0 { 0 } else { u64::MAX >> u.leading_zeros() };
    i64::try_from(m).ok().filter(|&v| v <= SPAN_LIMIT)
}

fn shift_amt(ty: Type, ry: Option<(i64, i64)>) -> Option<u32> {
    let (_, sh) = ry.filter(|&(lo, hi)| lo == hi)?;
    Some((sh as u32) & (ty.bits() - 1))
}

fn eval_band(is_64: bool, umax: i64, rx: Option<(i64, i64)>, ry: Option<(i64, i64)>) -> Option<(i64, i64)> {
    if !is_64 {
        return Some((0, rx.map_or(umax, |r| r.1).min(ry.map_or(umax, |r| r.1))));
    }
    match (rx.filter(|r| r.0 >= 0), ry.filter(|r| r.0 >= 0)) {
        (Some(a), Some(b)) => Some((0, a.1.min(b.1))),
        (Some(a), None) => Some((0, a.1)),
        (None, Some(b)) => Some((0, b.1)),
        (None, None) => None,
    }
}

fn eval_bor(is_64: bool, umax: i64, rx: Option<(i64, i64)>, ry: Option<(i64, i64)>) -> Option<(i64, i64)> {
    let (Some((la, ha)), Some((lb, hb))) = (rx, ry) else {
        return None;
    };
    if la < 0 || lb < 0 {
        return None;
    }
    let hi = mask_hi(ha)? | mask_hi(hb)?;
    (is_64 || hi <= umax).then_some((la.max(lb), hi))
}

fn eval_ushr(ty: Type, rx: Option<(i64, i64)>, ry: Option<(i64, i64)>) -> Option<(i64, i64)> {
    let k = shift_amt(ty, ry)?;
    if k == 0 {
        return rx;
    }
    if ty != Type::I64 {
        let (lo, hi) = rx.unwrap_or((0, ty.unsigned_max() as i64));
        return Some(((lo as u64 >> k) as i64, (hi as u64 >> k) as i64));
    }
    match rx.filter(|r| r.0 >= 0) {
        Some((lo, hi)) => Some(((lo as u64 >> k) as i64, (hi as u64 >> k) as i64)),
        None => Some((0, (u64::MAX >> k) as i64)),
    }
}

fn eval_ishl(ty: Type, rx: Option<(i64, i64)>, ry: Option<(i64, i64)>) -> Option<(i64, i64)> {
    let k = shift_amt(ty, ry)?;
    let (lo, hi) = rx?;
    if k == 0 {
        return Some((lo, hi));
    }
    let fits = if ty == Type::I64 {
        lo >= (i64::MIN >> k) && hi <= (i64::MAX >> k)
    } else {
        hi <= (ty.unsigned_max() >> k) as i64
    };
    fits.then_some((lo << k, hi << k))
}

fn eval_bit_range(ty: Type, bin: Binary, rx: Option<(i64, i64)>, ry: Option<(i64, i64)>) -> Option<(i64, i64)> {
    let (umax, is_64) = (ty.unsigned_max() as i64, ty == Type::I64);
    match bin {
        Binary::Band => eval_band(is_64, umax, rx, ry),
        Binary::Bor => eval_bor(is_64, umax, rx, ry),
        Binary::Ushr => eval_ushr(ty, rx, ry),
        Binary::Ishl => eval_ishl(ty, rx, ry),
        _ => None,
    }
}

fn eval_udiv_urem(is_rem: bool, ty: Type, rx: Option<(i64, i64)>, ry: Option<(i64, i64)>) -> Option<(i64, i64)> {
    let (_, d) = ry.filter(|&(lo, hi)| lo == hi && lo > 0)?;
    if is_rem {
        let hi = rx.filter(|r| r.0 >= 0).map_or(d - 1, |r| r.1.min(d - 1));
        return Some((0, hi));
    }
    let (lo, hi) = if ty == Type::I64 { rx.filter(|r| r.0 >= 0)? } else { rx.unwrap_or((0, ty.unsigned_max() as i64)) };
    Some((lo / d, hi / d))
}

fn eval_imul(is_64: bool, umax: i64, a: (i64, i64), b: (i64, i64)) -> Option<(i64, i64)> {
    let (la, ha) = a;
    let (lb, hb) = b;
    let ps = [la.checked_mul(lb)?, la.checked_mul(hb)?, ha.checked_mul(lb)?, ha.checked_mul(hb)?];
    let lo = *ps.iter().min()?;
    let hi = *ps.iter().max()?;
    (is_64 || hi <= umax).then_some((lo, hi))
}

fn eval_arith_range(ty: Type, bin: Binary, rx: Option<(i64, i64)>, ry: Option<(i64, i64)>) -> Option<(i64, i64)> {
    let (umax, is_64) = (ty.unsigned_max() as i64, ty == Type::I64);
    match bin {
        Binary::Urem => eval_udiv_urem(true, ty, rx, ry),
        Binary::Udiv => eval_udiv_urem(false, ty, rx, ry),
        Binary::Iadd | Binary::Isub | Binary::Imul => {
            let (Some(a), Some(b)) = (rx, ry) else { return None };
            match bin {
                Binary::Iadd => a.0.checked_add(b.0).zip(a.1.checked_add(b.1)).filter(|&(_, hi)| is_64 || hi <= umax),
                Binary::Isub if is_64 || a.0 >= b.1 => a.0.checked_sub(b.1).zip(a.1.checked_sub(b.0)),
                Binary::Imul => eval_imul(is_64, umax, a, b),
                _ => None,
            }
        }
        _ => None,
    }
}

fn eval_convert_range(func: &Function, kind: Convert, x: Value, ty: Type, ranges: &[Option<(i64, i64)>]) -> Option<(i64, i64)> {
    let umax = ty.unsigned_max() as i64;
    match kind {
        Convert::Uextend if func.ty(x).is_small_int() => Some(ranges[x.index()].unwrap_or((0, func.ty(x).unsigned_max() as i64))),
        Convert::Sextend => {
            let (lo, hi) = ranges[x.index()]?;
            let lo_ok = ty == Type::I64 || lo >= 0;
            let hi_ok = func.ty(x) == Type::I64 || hi <= (func.ty(x).unsigned_max() >> 1) as i64;
            (lo_ok && hi_ok).then_some((lo, hi))
        }
        Convert::Ireduce => match ranges[x.index()] {
            Some((lo, hi)) if lo >= 0 && hi <= umax => Some((lo, hi)),
            _ => Some((0, umax)),
        },
        Convert::Bitcast if func.ty(x) == ty => ranges[x.index()],
        _ => None,
    }
}

fn eval_range(func: &Function, op: &Op, ty: Type, ranges: &[Option<(i64, i64)>]) -> Option<(i64, i64)> {
    if !ty.is_small_int() {
        return None;
    }
    let r = match *op {
        Op::Const(imm) => {
            let v = if ty == Type::I64 { imm } else { ((imm as u64) & ty.unsigned_max()) as i64 };
            (v, v)
        }
        Op::Icmp(..) | Op::Fcmp(..) | Op::Fits(..) => (0, 1),
        Op::Convert(kind, x) => eval_convert_range(func, kind, x, ty, ranges)?,
        Op::Select(_, a, b) => match (ranges[a.index()], ranges[b.index()]) {
            (Some((la, ha)), Some((lb, hb))) => (la.min(lb), ha.max(hb)),
            _ => return None,
        },
        Op::Binary(bin @ (Binary::Band | Binary::Bor | Binary::Ushr | Binary::Ishl), x, y) => {
            eval_bit_range(ty, bin, ranges[x.index()], ranges[y.index()])?
        }
        Op::Binary(bin, x, y) => eval_arith_range(ty, bin, ranges[x.index()], ranges[y.index()])?,
        _ => return None,
    };
    (r.0 >= -SPAN_LIMIT && r.1 <= SPAN_LIMIT && r.0 <= r.1).then_some(r)
}

fn ptr_base(
    v: Value,
    spans: &[Option<(MemBase, i64, i64)>],
    slots: &[Option<(StackSlot, i32)>],
    symbols: &[Option<u32>],
) -> (MemBase, i64, i64) {
    if let Some(s) = spans[v.index()] {
        return s;
    }
    if let Some((slot, off)) = slots[v.index()] {
        return (MemBase::Slot(slot), i64::from(off), i64::from(off));
    }
    if let Some(sym) = symbols[v.index()] {
        return (MemBase::Global(sym), 0, 0);
    }
    (MemBase::Addr(v), 0, 0)
}

fn eval_ptr_offset(
    is_sub: bool,
    a: Value,
    b: Value,
    ranges: &[Option<(i64, i64)>],
    ptrs: (&[Option<(MemBase, i64, i64)>], &[Option<(StackSlot, i32)>], &[Option<u32>]),
) -> Option<(MemBase, i64, i64)> {
    let (ptr, off, sign) = match (is_sub, ranges[a.index()], ranges[b.index()]) {
        (true, None, Some(off)) => (a, off, true),
        (false, None, Some(off)) => (a, off, false),
        (false, Some(off), None) => (b, off, false),
        _ => return None,
    };
    let (base, plo, phi) = ptr_base(ptr, ptrs.0, ptrs.1, ptrs.2);
    if sign {
        plo.checked_sub(off.1).zip(phi.checked_sub(off.0)).map(|(lo, hi)| (base, lo, hi))
    } else {
        plo.checked_add(off.0).zip(phi.checked_add(off.1)).map(|(lo, hi)| (base, lo, hi))
    }
}

fn eval_span(
    func: &Function,
    op: &Op,
    ty: Type,
    ranges: &[Option<(i64, i64)>],
    ptrs: (&[Option<(MemBase, i64, i64)>], &[Option<(StackSlot, i32)>], &[Option<u32>]),
) -> Option<(MemBase, i64, i64)> {
    if ty != Type::I64 {
        return None;
    }
    let span = match *op {
        Op::StackAddr(slot, off) => (MemBase::Slot(slot), i64::from(off), i64::from(off)),
        Op::Symbol(sym) | Op::SlotAddr(sym) => (MemBase::Global(sym), 0, 0),
        Op::Convert(Convert::Bitcast, x) if func.ty(x) == Type::I64 => ptrs.0[x.index()]?,
        Op::Select(_, a, b) => {
            let ((ba, la, ha), (bb, lb, hb)) = (ptr_base(a, ptrs.0, ptrs.1, ptrs.2), ptr_base(b, ptrs.0, ptrs.1, ptrs.2));
            if ba != bb {
                return None;
            }
            (ba, la.min(lb), ha.max(hb))
        }
        Op::Binary(bin @ (Binary::Iadd | Binary::Isub), a, b) => eval_ptr_offset(bin == Binary::Isub, a, b, ranges, ptrs)?,
        _ => return None,
    };
    (span.1 >= -SPAN_LIMIT && span.2 <= SPAN_LIMIT && span.1 <= span.2).then_some(span)
}

pub(super) fn compute_spans(
    func: &Function,
    cfg: &Cfg,
    slots: &[Option<(StackSlot, i32)>],
    symbols: &[Option<u32>],
) -> Vec<Option<(MemBase, i64, i64)>> {
    let n = func.types.len();
    let mut ranges = vec![None; n];
    let mut spans = vec![None; n];
    for &b in &cfg.rpo {
        for &inst in &func.blocks[b.index()].insts {
            let data = &func.insts[inst.index()];
            let Some(res) = data.result else { continue };
            let ty = func.ty(res);
            ranges[res.index()] = eval_range(func, &data.op, ty, &ranges);
            spans[res.index()] = eval_span(func, &data.op, ty, &ranges, (&spans, slots, symbols));
        }
    }
    spans
}

struct WriteRanges {
    len: usize,
    clobbered: Vec<usize>,
    ranges: Vec<(usize, i64, i64)>,
}

impl WriteRanges {
    fn new(len: usize) -> Self {
        Self { len, clobbered: Vec::new(), ranges: Vec::new() }
    }

    fn clobber(&mut self, idx: usize) {
        if idx < self.len {
            self.clobbered.push(idx);
        }
    }

    fn sort(&mut self) {
        self.clobbered.sort_unstable();
        self.ranges.sort_unstable();
    }

    fn is_clobbered(&self, idx: usize) -> bool {
        idx >= self.len || self.clobbered.binary_search(&idx).is_ok()
    }

    fn of(&self, idx: usize) -> &[(usize, i64, i64)] {
        let start = self.ranges.partition_point(|range| range.0 < idx);
        let count = self.ranges[start..].partition_point(|range| range.0 == idx);
        &self.ranges[start..start + count]
    }

    fn record(&mut self, idx: usize, lo: i64, bytes: u32) {
        let Some(hi) = (lo >= 0).then(|| lo.checked_add(i64::from(bytes))).flatten() else {
            self.clobber(idx);
            return;
        };
        self.record_span(idx, lo, hi);
    }

    fn record_span(&mut self, idx: usize, lo: i64, hi: i64) {
        if lo < 0 || hi < lo {
            self.clobber(idx);
            return;
        }
        if idx < self.len {
            self.ranges.push((idx, lo, hi));
        }
    }

    fn overlaps_span(&self, idx: usize, lo: i64, hi: i64) -> bool {
        if lo < 0 || hi < lo {
            return true;
        }
        self.is_clobbered(idx) || self.of(idx).iter().any(|&(_, wlo, whi)| lo < whi && wlo < hi)
    }

    fn any(&self, idx: usize) -> bool {
        self.is_clobbered(idx) || !self.of(idx).is_empty()
    }
}

#[derive(Default)]
struct ParamMask(u64, bool);

impl ParamMask {
    fn set(&mut self, p: u16) {
        if p < 64 {
            self.0 |= 1_u64 << p;
        } else {
            self.1 = true;
        }
    }

    fn get(&self, p: u16) -> bool {
        if p < 64 { (self.0 & (1_u64 << p)) != 0 } else { self.1 }
    }
}

pub(crate) struct LoopMem {
    pub writes: bool,
    pub unknown_mem_writes: bool,
    pub unique_heap_writes: bool,
    pub calls: bool,
    try_calls: bool,
    pub(crate) atomics: bool,
    pub(crate) block_copies: bool,
    slot_writes: WriteRanges,
    param_writes: WriteRanges,
    calls_write_noalias: ParamMask,
    calls_read_noalias: ParamMask,
    head_noalias_before_unknown: u64,
    copy_locs: Vec<MemLoc>,
}

fn entry_param(func: &Function, param: u16) -> Option<Value> {
    let entry = func.entry?;
    func.blocks[entry.index()].params.get(usize::from(param)).copied()
}

fn param_span_deref_ok(func: &Function, param: u16, span: (i128, i128)) -> bool {
    let deref_bytes = func.signature.params.get(usize::from(param)).map_or(0, |abi| abi.deref_bytes);
    span.0 >= 0 && span.1 <= i128::from(deref_bytes)
}

impl LoopMem {
    pub(crate) fn analyze(func: &Function, each: &Loop, prov: &Provenance) -> Self {
        let mut fx = Self {
            writes: false,
            unknown_mem_writes: false,
            unique_heap_writes: false,
            calls: false,
            try_calls: false,
            atomics: false,
            block_copies: false,
            slot_writes: WriteRanges::new(func.slots.len()),
            param_writes: WriteRanges::new(func.signature.params.len()),
            calls_write_noalias: ParamMask::default(),
            calls_read_noalias: ParamMask::default(),
            head_noalias_before_unknown: 0,
            copy_locs: Vec::new(),
        };
        for inst in (0..each.body.len()).filter(|b| each.body[*b]).flat_map(|b| &func.blocks[b].insts) {
            fx.analyze_op(func, prov, &func.insts[inst.index()].op);
        }
        for &inst in &func.blocks[each.head.index()].insts {
            let data = &func.insts[inst.index()];
            if fx.record_head_noalias(func, prov, &data.op, data.result) {
                break;
            }
        }
        fx.slot_writes.sort();
        fx.param_writes.sort();
        fx
    }

    fn head_touches_noalias(&self, p: u16) -> bool {
        p < 64 && (self.head_noalias_before_unknown & (1_u64 << p)) != 0
    }

    fn mark_head_noalias(&mut self, root: MemRoot) {
        if let MemRoot::NoAliasParam(p) = root
            && p < 64
        {
            self.head_noalias_before_unknown |= 1_u64 << p;
        }
    }

    fn record_head_noalias(&mut self, func: &Function, prov: &Provenance, op: &Op, res: Option<Value>) -> bool {
        if is_atomic_barrier(op) || matches!(op, Op::Call { .. } | Op::TryCall { .. }) {
            return true;
        }
        match *op {
            Op::Load(addr, off) => {
                let bytes = res.map_or(1, |r| func.ty(r).bytes());
                self.mark_head_noalias(prov.loc(addr, i64::from(off), bytes).root);
            }
            Op::Store(val, addr, off) => {
                let root = prov.loc(addr, i64::from(off), func.ty(val).bytes()).root;
                self.mark_head_noalias(root);
                return !matches!(root, MemRoot::Slot(_) | MemRoot::NoAliasParam(_) | MemRoot::UniqueHeap);
            }
            Op::Move(dst, _, bytes) => return !matches!(prov.loc(dst, 0, bytes).root, MemRoot::Slot(_) | MemRoot::NoAliasParam(_) | MemRoot::UniqueHeap),
            Op::Copy(dst, _, _) => return !matches!(prov.loc(dst, 0, u32::MAX).root, MemRoot::Slot(_) | MemRoot::NoAliasParam(_) | MemRoot::UniqueHeap),
            _ => {}
        }
        false
    }

    fn record_call_noalias(&mut self, func: &Function, prov: &Provenance, sig: SigRef, args: &[Value]) {
        let params = &func.sigs[sig.index()].params;
        for (i, &arg) in args.iter().enumerate() {
            if let Some(MemRoot::NoAliasParam(p)) = prov.param_roots.get(arg.index()).copied() {
                self.calls_read_noalias.set(p);
                if !params.get(i).is_some_and(|pa| pa.readonly) {
                    self.calls_write_noalias.set(p);
                    self.param_writes.clobber(usize::from(p));
                }
            }
        }
    }

    fn analyze_op(&mut self, func: &Function, prov: &Provenance, op: &Op) {
        let is_atomic = is_atomic_barrier(op);
        let is_try_call = matches!(op, Op::TryCall { .. });
        let is_call = is_try_call || matches!(op, Op::Call { .. });
        self.calls |= is_call;
        self.try_calls |= is_try_call;
        self.atomics |= is_atomic;
        self.unknown_mem_writes |= is_atomic || is_call;
        self.writes |= is_atomic || is_call || matches!(op, Op::Store(..) | Op::StackStore(..) | Op::Copy(..) | Op::Move(..));
        match *op {
            Op::StackStore(val, slot, off) => self.slot_writes.record(slot.index(), i64::from(off), func.ty(val).bytes()),
            Op::Store(val, addr, off) => self.record_loc(func, prov, prov.loc(addr, i64::from(off), func.ty(val).bytes())),
            Op::Move(dst, src, bytes) => {
                self.block_copies = true;
                let dloc = prov.loc(dst, 0, bytes);
                self.record_loc(func, prov, dloc);
                self.copy_locs.push(dloc);
                self.copy_locs.push(prov.loc(src, 0, bytes));
            }
            Op::Copy(dst, src, _) => {
                self.block_copies = true;
                let dloc = prov.loc(dst, 0, u32::MAX);
                self.record_clobber_loc(dloc);
                self.copy_locs.push(dloc);
                self.copy_locs.push(prov.loc(src, 0, u32::MAX));
            }
            Op::Call { sig, ref args, .. } | Op::TryCall { sig, ref args, .. } => {
                self.record_call_noalias(func, prov, sig, args);
            }
            _ => {}
        }
    }

    fn record_loc(&mut self, func: &Function, prov: &Provenance, loc: MemLoc) {
        let (base, lo, hi) = prov.loc_span(loc);
        let span = i64::try_from(lo).ok().zip(i64::try_from(hi).ok());
        match (base, loc.root, span) {
            (MemBase::Slot(s), _, Some((lo, hi))) => self.slot_writes.record_span(s.index(), lo, hi),
            (MemBase::Slot(s), _, None) => self.slot_writes.clobber(s.index()),
            (_, MemRoot::NoAliasParam(p), Some((lo, hi))) if entry_param(func, p).is_some_and(|ep| base == MemBase::Addr(ep)) => {
                self.param_writes.record_span(usize::from(p), lo, hi);
            }
            _ => self.record_clobber_loc(loc),
        }
    }

    fn record_clobber_loc(&mut self, loc: MemLoc) {
        match loc.root {
            MemRoot::Slot(s) => self.slot_writes.clobber(s.index()),
            MemRoot::NoAliasParam(p) => self.param_writes.clobber(usize::from(p)),
            MemRoot::UniqueHeap => self.unique_heap_writes = true,
            _ => self.unknown_mem_writes = true,
        }
    }

    fn slot_load_ok(&self, func: &Function, prov: &Provenance, slot: StackSlot, span: (i128, i128)) -> bool {
        let (lo, hi) = span;
        let end_ok = lo >= 0 && hi <= i128::from(func.slots[slot.index()].size);
        let mut_ok = prov.slot_unescaped(slot)
            || (prov.slot_call_readonly(slot) && !self.unknown_mem_writes)
            || (!self.calls && !self.unknown_mem_writes);
        let no_write = i64::try_from(lo)
            .ok()
            .zip(i64::try_from(hi).ok())
            .is_some_and(|(lo, hi)| !self.slot_writes.overlaps_span(slot.index(), lo, hi));
        end_ok && mut_ok && no_write
    }

    pub(crate) fn can_hoist_load(&self, func: &Function, prov: &Provenance, head_prefix: bool, inst: Inst) -> bool {
        let data = &func.insts[inst.index()];
        let Some(res) = data.result else { return false };
        let width = func.ty(res).bytes();
        match data.op {
            Op::StackLoad(slot, offset) => {
                let lo = i128::from(offset);
                self.slot_load_ok(func, prov, slot, (lo, lo + i128::from(width)))
            }
            Op::Load(addr, offset) => self.can_hoist_mem_load(func, prov, head_prefix, (addr, offset, width)),
            _ => false,
        }
    }

    fn can_hoist_mem_load(&self, func: &Function, prov: &Provenance, head_prefix: bool, load: (Value, i32, u32)) -> bool {
        let (addr, offset, width) = load;
        let loc = prov.loc(addr, i64::from(offset), width);
        let (base, lo, hi) = prov.loc_span(loc);
        if let MemBase::Slot(slot) = base {
            return self.slot_load_ok(func, prov, slot, (lo, hi));
        }
        let deref_ok = in_range(i64::from(offset), width, prov.ro_bytes.get(addr.index()).copied().unwrap_or(0));
        if self.atomics {
            return false;
        }
        let (root_ok, trap_ok, readonly_ok, call_safe) = self.root_load_ok(func, prov, loc, (base, lo, hi));
        if !(head_prefix || deref_ok || trap_ok) {
            return false;
        }
        readonly_ok || call_safe || (!self.calls && (!self.writes || root_ok))
    }

    fn slot_root_ok(&self, func: &Function, prov: &Provenance, slot: StackSlot, loc: MemLoc) -> (bool, bool, bool, bool) {
        let trap_ok = in_range(loc.offset, loc.size, func.slots[slot.index()].size);
        let slot_safe = prov.slot_unescaped(slot) || (prov.slot_call_readonly(slot) && !self.unknown_mem_writes);
        let no_write = !self.slot_writes.any(slot.index());
        let root_ok = (slot_safe || (!self.calls && !self.unknown_mem_writes)) && no_write;
        (root_ok, trap_ok, false, slot_safe && no_write)
    }

    fn noalias_load_ok(&self, func: &Function, prov: &Provenance, p: u16, span: (MemBase, i128, i128)) -> (bool, bool, bool, bool) {
        let (base, lo, hi) = span;
        let is_entry = entry_param(func, p).is_some_and(|ep| base == MemBase::Addr(ep));
        let trap_ok = is_entry && param_span_deref_ok(func, p, (lo, hi));
        let span64 = i64::try_from(lo).ok().zip(i64::try_from(hi).ok());
        let no_param_write = match span64.filter(|_| is_entry) {
            Some((slo, shi)) => !self.param_writes.overlaps_span(usize::from(p), slo, shi),
            None => !self.param_writes.any(usize::from(p)),
        };
        let unescaped_safe = prov.noalias_unescaped(p) && (self.head_touches_noalias(p) || !self.unknown_mem_writes);
        let readonly_ok = no_param_write && func.signature.params.get(usize::from(p)).is_some_and(|abi| abi.readonly);
        let call_safe = no_param_write && unescaped_safe && !self.calls_write_noalias.get(p);
        let root_ok = no_param_write && (unescaped_safe || !self.unknown_mem_writes);
        (root_ok, trap_ok, readonly_ok, call_safe)
    }

    fn root_load_ok(&self, func: &Function, prov: &Provenance, loc: MemLoc, span: (MemBase, i128, i128)) -> (bool, bool, bool, bool) {
        let (base, lo, hi) = span;
        match loc.root {
            MemRoot::Slot(slot) => self.slot_root_ok(func, prov, slot, loc),
            MemRoot::NoAliasParam(p) => self.noalias_load_ok(func, prov, p, span),
            MemRoot::ReadonlyParam(p) => {
                let is_entry = entry_param(func, p).is_some_and(|ep| base == MemBase::Addr(ep));
                (true, is_entry && param_span_deref_ok(func, p, (lo, hi)), true, false)
            }
            _ => (false, false, false, false),
        }
    }

    pub(crate) fn promote_candidates(
        &self,
        func: &Function,
        each: &Loop,
        prov: &Provenance,
        home: &[Option<Block>],
    ) -> Vec<PromoteCand> {
        if self.atomics || self.try_calls {
            return Vec::new();
        }
        let cx = PromoteCx { func, each, prov, home };
        let accesses = collect_loop_accesses(&cx);
        let mut cands = Vec::new();
        for &acc in &accesses {
            let key = (acc.loc.base, acc.loc.offset, acc.ty);
            if cands.iter().any(|c: &PromoteCand| (c.loc.base, c.loc.offset, c.ty) == key) {
                continue;
            }
            let Some(cand) = check_promote_key(self, &cx, &accesses, acc) else { continue };
            cands.push(cand);
        }
        cands
    }
}

struct PromoteCx<'a> {
    func: &'a Function,
    each: &'a Loop,
    prov: &'a Provenance,
    home: &'a [Option<Block>],
}

#[derive(Clone, Copy)]
pub(crate) enum PromoteBase {
    Slot(StackSlot),
    Addr(Value),
}

#[derive(Clone, Copy)]
pub(crate) struct PromoteCand {
    pub loc: MemLoc,
    pub base: PromoteBase,
    pub ty: Type,
    pub offset: i32,
}

#[derive(Clone, Copy)]
struct LoopAccess {
    loc: MemLoc,
    ty: Type,
    is_write: bool,
    head_prefix: bool,
}

fn loop_inst_access(func: &Function, op: &Op, res: Option<Value>, prov: &Provenance, head_prefix: bool) -> Option<LoopAccess> {
    match (op, res) {
        (&Op::StackLoad(slot, off), Some(r)) => {
            let ty = func.ty(r);
            Some(LoopAccess { loc: MemLoc::slot(slot, i64::from(off), ty.bytes()), ty, is_write: false, head_prefix })
        }
        (&Op::Load(addr, off), Some(r)) => {
            let ty = func.ty(r);
            Some(LoopAccess { loc: prov.loc(addr, i64::from(off), ty.bytes()), ty, is_write: false, head_prefix })
        }
        (&Op::StackStore(val, slot, off), _) => {
            let ty = func.ty(val);
            Some(LoopAccess { loc: MemLoc::slot(slot, i64::from(off), ty.bytes()), ty, is_write: true, head_prefix })
        }
        (&Op::Store(val, addr, off), _) => {
            let ty = func.ty(val);
            Some(LoopAccess { loc: prov.loc(addr, i64::from(off), ty.bytes()), ty, is_write: true, head_prefix })
        }
        _ => None,
    }
}

fn collect_loop_accesses(cx: &PromoteCx<'_>) -> Vec<LoopAccess> {
    let mut accesses = Vec::new();
    for b in (0..cx.each.body.len()).filter(|b| cx.each.body[*b]) {
        let mut head_prefix = b == cx.each.head.index();
        for &inst in &cx.func.blocks[b].insts {
            let data = &cx.func.insts[inst.index()];
            if let Some(acc) = loop_inst_access(cx.func, &data.op, data.result, cx.prov, head_prefix) {
                accesses.push(acc);
            }
            if matches!(
                data.op,
                Op::Call { .. }
                    | Op::TryCall { .. }
                    | Op::Trap
                    | Op::Trapz(_)
                    | Op::Brif(..)
                    | Op::BrTable(..)
                    | Op::Jump(..)
                    | Op::Return(_)
                    | Op::Binary(Binary::Sdiv | Binary::Udiv | Binary::Srem | Binary::Urem, ..)
            ) {
                head_prefix = false;
            }
        }
    }
    accesses
}

fn promote_safe(mem: &LoopMem, loc: MemLoc, prov: &Provenance) -> bool {
    if mem.copy_locs.iter().any(|&cloc| may_alias(loc, cloc, prov)) {
        return false;
    }
    match loc.root {
        MemRoot::Slot(s) => !mem.calls || prov.slot_unescaped(s),
        MemRoot::NoAliasParam(p) => {
            (!mem.calls || (prov.noalias_unescaped(p) && !mem.calls_read_noalias.get(p)))
                && (!mem.unknown_mem_writes || mem.head_touches_noalias(p))
        }
        _ => !mem.calls,
    }
}

fn check_promote_key(mem: &LoopMem, cx: &PromoteCx<'_>, accesses: &[LoopAccess], sample: LoopAccess) -> Option<PromoteCand> {
    let (loc, ty) = (sample.loc, sample.ty);
    if !(ty.is_small_int() || ty.is_float()) || !promote_safe(mem, loc, cx.prov) {
        return None;
    }
    let offset = i32::try_from(loc.offset).ok()?;
    let (mut has_read, mut has_write, mut head_accessed) = (false, false, false);
    for &acc in accesses {
        if (acc.loc.base, acc.loc.offset, acc.ty) == (loc.base, loc.offset, ty) {
            has_read |= !acc.is_write;
            has_write |= acc.is_write;
            head_accessed |= acc.head_prefix;
        } else if may_alias(loc, acc.loc, cx.prov) {
            return None;
        }
    }
    if !has_read || !has_write {
        return None;
    }
    let base = promote_base_eligible(cx, loc, ty, head_accessed)?;
    Some(PromoteCand { loc, base, ty, offset })
}

fn promote_base_eligible(cx: &PromoteCx<'_>, loc: MemLoc, ty: Type, head_accessed: bool) -> Option<PromoteBase> {
    let end = loc.offset.checked_add(i64::from(ty.bytes()))?;
    if loc.offset < 0 {
        return None;
    }
    match loc.base {
        MemBase::Slot(slot) => {
            (cx.prov.slot_unescaped(slot) && end as u64 <= u64::from(cx.func.slots[slot.index()].size)).then_some(PromoteBase::Slot(slot))
        }
        MemBase::Addr(addr) => {
            let MemRoot::NoAliasParam(p) = loc.root else { return None };
            let abi = cx.func.signature.params.get(usize::from(p))?;
            if abi.readonly || !cx.prov.noalias_unescaped(p) {
                return None;
            }
            let invariant = cx.home.get(addr.index()).copied().flatten().is_none_or(|at| !cx.each.body.get(at.index()).copied().unwrap_or(false));
            if !invariant {
                return None;
            }
            let deref_ok = end as u64 <= u64::from(cx.prov.ro_bytes.get(addr.index()).copied().unwrap_or(0));
            (head_accessed || deref_ok).then_some(PromoteBase::Addr(addr))
        }
        MemBase::Global(_) => None,
    }
}

mod promote;
mod spans;

pub(crate) use promote::{PromoteBase, PromoteCand};
pub(super) use spans::compute_spans;

use crate::cfg::Loop;
use crate::ir::{Function, Inst, Op, SigRef, StackSlot, Value};

use super::{MemBase, MemLoc, MemRoot, Provenance, in_range, is_atomic_barrier, is_rodata};

pub(super) const SPAN_LIMIT: i64 = 1_i64 << 60;

struct WriteRanges {
    len: usize,
    clobbered: Vec<usize>,
    ranges: Vec<(usize, i64, i64)>,
}

impl WriteRanges {
    fn new(len: usize) -> Self {
        Self {
            len,
            clobbered: Vec::new(),
            ranges: Vec::new(),
        }
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
        let Some(hi) = (lo >= 0)
            .then(|| lo.checked_add(i64::from(bytes)))
            .flatten()
        else {
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
        self.is_clobbered(idx)
            || self
                .of(idx)
                .iter()
                .any(|&(_, wlo, whi)| lo < whi && wlo < hi)
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
        if p < 64 {
            (self.0 & (1_u64 << p)) != 0
        } else {
            self.1
        }
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

fn is_unknown_write(root: MemRoot) -> bool {
    !matches!(
        root,
        MemRoot::Slot(_) | MemRoot::NoAliasParam(_) | MemRoot::UniqueHeap(_)
    )
}

fn entry_param(func: &Function, param: u16) -> Option<Value> {
    let entry = func.entry?;
    func.blocks[entry.index()]
        .params
        .get(usize::from(param))
        .copied()
}

fn param_span_deref_ok(func: &Function, param: u16, span: (i128, i128)) -> bool {
    let deref_bytes = func
        .signature
        .params
        .get(usize::from(param))
        .map_or(0, |abi| abi.deref_bytes);
    span.0 >= 0 && span.1 <= i128::from(deref_bytes)
}

fn is_pure_loop_call(func: &Function, zeros: &[bool], op: &Op) -> bool {
    match op {
        Op::Call { callee, .. } => {
            func.is_fresh_alloc(*callee)
                || (!zeros.is_empty()
                    && func
                        .pure_thread_cell_result(op, |v| zeros.get(v.index()) == Some(&true))
                        .is_some())
        }
        _ => false,
    }
}

impl LoopMem {
    pub(crate) fn analyze(func: &Function, each: &Loop, prov: &Provenance) -> Self {
        let zeros = func.thread_cell_zeros();
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
        for inst in (0..each.body.len())
            .filter(|b| each.body[*b])
            .flat_map(|b| &func.blocks[b].insts)
        {
            fx.analyze_op(func, prov, &zeros, &func.insts[inst.index()].op);
        }
        for &inst in &func.blocks[each.head.index()].insts {
            let data = &func.insts[inst.index()];
            if fx.record_head_noalias(func, prov, &zeros, &data.op, data.result) {
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

    fn record_head_noalias(
        &mut self,
        func: &Function,
        prov: &Provenance,
        zeros: &[bool],
        op: &Op,
        res: Option<Value>,
    ) -> bool {
        if is_atomic_barrier(op)
            || (matches!(op, Op::Call { .. } | Op::TryCall { .. })
                && !is_pure_loop_call(func, zeros, op))
        {
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
                return is_unknown_write(root);
            }
            Op::Move(dst, _, bytes) => return is_unknown_write(prov.loc(dst, 0, bytes).root),
            Op::Copy(dst, _, _) => return is_unknown_write(prov.loc(dst, 0, u32::MAX).root),
            _ => {}
        }
        false
    }

    fn record_call_noalias(
        &mut self,
        func: &Function,
        prov: &Provenance,
        sig: SigRef,
        args: &[Value],
    ) {
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

    fn analyze_op(&mut self, func: &Function, prov: &Provenance, zeros: &[bool], op: &Op) {
        let is_atomic = is_atomic_barrier(op);
        let is_try_call = matches!(op, Op::TryCall { .. });
        let is_call = is_try_call || matches!(op, Op::Call { .. });
        let pure_call = is_call && !is_try_call && is_pure_loop_call(func, zeros, op);
        let mem_call = is_call && !pure_call;
        self.calls |= is_call;
        self.try_calls |= is_try_call;
        self.atomics |= is_atomic;
        self.unknown_mem_writes |= is_atomic || mem_call;
        self.writes |= is_atomic
            || mem_call
            || matches!(
                op,
                Op::Store(..) | Op::StackStore(..) | Op::Copy(..) | Op::Move(..)
            );
        match *op {
            Op::StackStore(val, slot, off) => {
                self.slot_writes
                    .record(slot.index(), i64::from(off), func.ty(val).bytes())
            }
            Op::Store(val, addr, off) => self.record_loc(
                func,
                prov,
                prov.loc(addr, i64::from(off), func.ty(val).bytes()),
            ),
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
            Op::Call { sig, ref args, .. } | Op::TryCall { sig, ref args, .. } if mem_call => {
                self.record_call_noalias(func, prov, sig, args);
            }
            _ => {}
        }
    }

    fn record_loc(&mut self, func: &Function, prov: &Provenance, loc: MemLoc) {
        let (base, lo, hi) = prov.loc_span(loc);
        let span = i64::try_from(lo).ok().zip(i64::try_from(hi).ok());
        match (base, loc.root, span) {
            (MemBase::Slot(s), _, Some((lo, hi))) => {
                self.slot_writes.record_span(s.index(), lo, hi)
            }
            (MemBase::Slot(s), _, None) => self.slot_writes.clobber(s.index()),
            (_, MemRoot::NoAliasParam(p), Some((lo, hi)))
                if entry_param(func, p).is_some_and(|ep| base == MemBase::Addr(ep)) =>
            {
                self.param_writes.record_span(usize::from(p), lo, hi);
            }
            _ => self.record_clobber_loc(loc),
        }
    }

    fn record_clobber_loc(&mut self, loc: MemLoc) {
        match loc.root {
            MemRoot::Slot(s) => self.slot_writes.clobber(s.index()),
            MemRoot::NoAliasParam(p) => self.param_writes.clobber(usize::from(p)),
            MemRoot::UniqueHeap(_) => self.unique_heap_writes = true,
            _ => self.unknown_mem_writes = true,
        }
    }

    fn slot_load_ok(
        &self,
        func: &Function,
        prov: &Provenance,
        slot: StackSlot,
        span: (i128, i128),
        unknown: bool,
    ) -> bool {
        let (lo, hi) = span;
        let end_ok = lo >= 0 && hi <= i128::from(func.slots[slot.index()].size);
        let mut_ok = prov.slot_unescaped(slot)
            || (prov.slot_call_readonly(slot) && !unknown)
            || (!self.calls && !unknown);
        let no_write = i64::try_from(lo)
            .ok()
            .zip(i64::try_from(hi).ok())
            .is_some_and(|(lo, hi)| !self.slot_writes.overlaps_span(slot.index(), lo, hi));
        end_ok && mut_ok && no_write
    }

    pub(crate) fn can_hoist_load(
        &self,
        func: &Function,
        prov: &Provenance,
        head_prefix: bool,
        inst: Inst,
    ) -> bool {
        self.load_hoists(func, prov, (head_prefix, self.unknown_mem_writes), inst)
    }

    /// Whether the load at `inst` hoists even when the loop's unique heap writes count as unknown writes.
    pub(crate) fn hoists_past_heap_writes(
        &self,
        func: &Function,
        prov: &Provenance,
        head_prefix: bool,
        inst: Inst,
    ) -> bool {
        self.load_hoists(
            func,
            prov,
            (
                head_prefix,
                self.unknown_mem_writes || self.unique_heap_writes,
            ),
            inst,
        )
    }

    fn load_hoists(
        &self,
        func: &Function,
        prov: &Provenance,
        mode: (bool, bool),
        inst: Inst,
    ) -> bool {
        let data = &func.insts[inst.index()];
        let Some(res) = data.result else { return false };
        let width = func.ty(res).bytes();
        match data.op {
            Op::StackLoad(slot, offset) => {
                let lo = i128::from(offset);
                self.slot_load_ok(func, prov, slot, (lo, lo + i128::from(width)), mode.1)
            }
            Op::Load(addr, offset) => {
                self.can_hoist_mem_load(func, prov, mode, (addr, offset, width))
            }
            _ => false,
        }
    }

    fn can_hoist_mem_load(
        &self,
        func: &Function,
        prov: &Provenance,
        mode: (bool, bool),
        load: (Value, i32, u32),
    ) -> bool {
        let ((head_prefix, unknown), (addr, offset, width)) = (mode, load);
        let loc = prov.loc(addr, i64::from(offset), width);
        let (base, lo, hi) = prov.loc_span(loc);
        if let MemBase::Slot(slot) = base {
            return self.slot_load_ok(func, prov, slot, (lo, hi), unknown);
        }
        let deref_ok = in_range(
            i64::from(offset),
            width,
            prov.ro_bytes.get(addr.index()).copied().unwrap_or(0),
        );
        let ro_global = is_rodata(func, loc);
        let (root_ok, trap_ok, readonly_ok, call_safe) =
            self.root_load_ok(func, prov, loc, (base, lo, hi), unknown);
        if self.atomics && !readonly_ok && !ro_global {
            return false;
        }
        if !(head_prefix || deref_ok || trap_ok || ro_global) {
            return false;
        }
        ro_global || readonly_ok || call_safe || (!self.calls && (!self.writes || root_ok))
    }

    fn slot_root_ok(
        &self,
        func: &Function,
        prov: &Provenance,
        slot: StackSlot,
        loc: MemLoc,
        unknown: bool,
    ) -> (bool, bool, bool, bool) {
        let trap_ok = in_range(loc.offset, loc.size, func.slots[slot.index()].size);
        let slot_safe = prov.slot_unescaped(slot) || (prov.slot_call_readonly(slot) && !unknown);
        let no_write = !self.slot_writes.any(slot.index());
        let root_ok = (slot_safe || (!self.calls && !unknown)) && no_write;
        (root_ok, trap_ok, false, slot_safe && no_write)
    }

    fn noalias_load_ok(
        &self,
        func: &Function,
        prov: &Provenance,
        p: u16,
        span: (MemBase, i128, i128),
        unknown: bool,
    ) -> (bool, bool, bool, bool) {
        let (base, lo, hi) = span;
        let is_entry = entry_param(func, p).is_some_and(|ep| base == MemBase::Addr(ep));
        let trap_ok = is_entry && param_span_deref_ok(func, p, (lo, hi));
        let span64 = i64::try_from(lo).ok().zip(i64::try_from(hi).ok());
        let no_param_write = match span64.filter(|_| is_entry) {
            Some((slo, shi)) => !self.param_writes.overlaps_span(usize::from(p), slo, shi),
            None => !self.param_writes.any(usize::from(p)),
        };
        let unescaped_safe =
            prov.noalias_unescaped(p) && (self.head_touches_noalias(p) || !unknown);
        let readonly_ok = no_param_write
            && func
                .signature
                .params
                .get(usize::from(p))
                .is_some_and(|abi| abi.readonly);
        let call_safe = no_param_write && unescaped_safe && !self.calls_write_noalias.get(p);
        let root_ok = no_param_write && (unescaped_safe || !unknown);
        (root_ok, trap_ok, readonly_ok, call_safe)
    }

    fn root_load_ok(
        &self,
        func: &Function,
        prov: &Provenance,
        loc: MemLoc,
        span: (MemBase, i128, i128),
        unknown: bool,
    ) -> (bool, bool, bool, bool) {
        let (base, lo, hi) = span;
        match loc.root {
            MemRoot::Slot(slot) => self.slot_root_ok(func, prov, slot, loc, unknown),
            MemRoot::NoAliasParam(p) => self.noalias_load_ok(func, prov, p, span, unknown),
            MemRoot::ReadonlyParam(p) => {
                let is_entry = entry_param(func, p).is_some_and(|ep| base == MemBase::Addr(ep));
                (
                    true,
                    is_entry && param_span_deref_ok(func, p, (lo, hi)),
                    true,
                    false,
                )
            }
            _ => (false, false, false, false),
        }
    }
}

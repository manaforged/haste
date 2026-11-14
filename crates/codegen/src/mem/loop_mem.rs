use crate::cfg::Loop;
use crate::ir::{Binary, Block, Function, Inst, Op, StackSlot, Type, Value};

use super::{MemBase, MemLoc, MemRoot, Provenance, may_alias};

struct WriteRanges {
    clobbered: Vec<bool>,
    ranges: Vec<Vec<(i64, i64)>>,
}

impl WriteRanges {
    fn new(len: usize) -> Self {
        Self { clobbered: vec![false; len], ranges: vec![Vec::new(); len] }
    }

    fn clobber(&mut self, idx: usize) {
        if let Some(slot) = self.clobbered.get_mut(idx) {
            *slot = true;
        }
    }

    fn record(&mut self, idx: usize, lo: i64, bytes: u32) {
        let Some(hi) = (lo >= 0).then(|| lo.checked_add(i64::from(bytes))).flatten() else {
            self.clobber(idx);
            return;
        };
        if let Some(list) = self.ranges.get_mut(idx) {
            list.push((lo, hi));
        }
    }

    fn overlaps(&self, idx: usize, lo: i64, bytes: u32) -> bool {
        let Some(hi) = (lo >= 0).then(|| lo.checked_add(i64::from(bytes))).flatten() else {
            return true;
        };
        self.clobbered.get(idx).copied().unwrap_or(true)
            || self.ranges.get(idx).is_none_or(|list| list.iter().any(|&(wlo, whi)| lo < whi && wlo < hi))
    }

    fn any(&self, idx: usize) -> bool {
        self.clobbered.get(idx).copied().unwrap_or(true)
            || self.ranges.get(idx).is_none_or(|list| !list.is_empty())
    }
}

pub(crate) struct LoopMem {
    pub writes: bool,
    pub unknown_mem_writes: bool,
    pub calls: bool,
    try_calls: bool,
    pub(crate) atomics: bool,
    pub(crate) block_copies: bool,
    slot_writes: WriteRanges,
    param_writes: WriteRanges,
    copy_locs: Vec<MemLoc>,
}

fn entry_param(func: &Function, param: u16) -> Option<Value> {
    let entry = func.entry?;
    func.blocks[entry.index()].params.get(usize::from(param)).copied()
}

fn param_deref_ok(func: &Function, param: u16, offset: i64, width: u32) -> bool {
    let deref_bytes = func.signature.params.get(usize::from(param)).map_or(0, |abi| abi.deref_bytes);
    offset >= 0 && (offset as u64).checked_add(u64::from(width)).is_some_and(|end| end <= u64::from(deref_bytes))
}

impl LoopMem {
    pub(crate) fn analyze(func: &Function, each: &Loop, prov: &Provenance) -> Self {
        let mut fx = Self {
            writes: false,
            unknown_mem_writes: false,
            calls: false,
            try_calls: false,
            atomics: false,
            block_copies: false,
            slot_writes: WriteRanges::new(func.slots.len()),
            param_writes: WriteRanges::new(func.signature.params.len()),
            copy_locs: Vec::new(),
        };
        for inst in (0..each.body.len()).filter(|b| each.body[*b]).flat_map(|b| &func.blocks[b].insts) {
            let op = &func.insts[inst.index()].op;
            let is_atomic = matches!(op, Op::AtomicLoad(_) | Op::AtomicStore(..) | Op::AtomicRmw(..) | Op::AtomicCas(..) | Op::Fence);
            let is_try_call = matches!(op, Op::TryCall { .. });
            let is_call = is_try_call || matches!(op, Op::Call { .. });
            if is_atomic || is_call || matches!(op, Op::Store(..) | Op::StackStore(..) | Op::Copy(..) | Op::Move(..)) {
                fx.writes = true;
            }
            fx.calls |= is_call;
            fx.try_calls |= is_try_call;
            fx.atomics |= is_atomic;
            if is_atomic || is_call {
                fx.unknown_mem_writes = true;
            }
            match *op {
                Op::StackStore(val, slot, off) => fx.slot_writes.record(slot.index(), i64::from(off), func.ty(val).bytes()),
                Op::Store(val, addr, off) => fx.record_loc(func, prov.loc(addr, i64::from(off), func.ty(val).bytes())),
                Op::Move(dst, src, bytes) => {
                    fx.block_copies = true;
                    let dloc = prov.loc(dst, 0, bytes);
                    fx.record_clobber_loc(dloc);
                    fx.copy_locs.push(dloc);
                    fx.copy_locs.push(prov.loc(src, 0, bytes));
                }
                Op::Copy(dst, _, _) => {
                    fx.block_copies = true;
                    let dloc = prov.loc(dst, 0, u32::MAX);
                    fx.record_clobber_loc(dloc);
                    fx.copy_locs.push(dloc);
                }
                _ => {}
            }
        }
        fx
    }

    fn record_loc(&mut self, func: &Function, loc: MemLoc) {
        if let MemBase::Slot(s) = loc.base {
            self.slot_writes.record(s.index(), loc.offset, loc.size);
        } else if let MemRoot::NoAliasParam(p) = loc.root
            && entry_param(func, p).is_some_and(|ep| loc.base == MemBase::Addr(ep))
        {
            self.param_writes.record(usize::from(p), loc.offset, loc.size);
        } else {
            self.record_clobber_loc(loc);
        }
    }

    fn record_clobber_loc(&mut self, loc: MemLoc) {
        match loc.root {
            MemRoot::Slot(s) => self.slot_writes.clobber(s.index()),
            MemRoot::NoAliasParam(p) => self.param_writes.clobber(usize::from(p)),
            _ => self.unknown_mem_writes = true,
        }
    }

    fn slot_load_ok(&self, func: &Function, prov: &Provenance, slot: StackSlot, total: i64, width: u32) -> bool {
        let end_ok = total >= 0 && (total as u64).checked_add(u64::from(width)).is_some_and(|end| end <= u64::from(func.slots[slot.index()].size));
        let mut_ok = prov.slot_unescaped(slot)
            || (prov.slot_call_readonly(slot) && !self.unknown_mem_writes)
            || (!self.calls && !self.unknown_mem_writes);
        end_ok && mut_ok && !self.slot_writes.overlaps(slot.index(), total, width)
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
                let deref_ok = offset >= 0
                    && (offset as u64)
                        .checked_add(u64::from(width))
                        .is_some_and(|e| e <= u64::from(prov.ro_bytes.get(addr.index()).copied().unwrap_or(0)));
                let mut trap_ok = head_prefix || deref_ok;
                let mut readonly_ok = false;
                let root_ok = match loc.root {
                    MemRoot::Slot(slot) => {
                        if loc.offset >= 0
                            && (loc.offset as u64).checked_add(u64::from(width)).is_some_and(|e| e <= u64::from(func.slots[slot.index()].size))
                        {
                            trap_ok = true;
                        }
                        (prov.slot_unescaped(slot) || (prov.slot_call_readonly(slot) && !self.unknown_mem_writes))
                            && !self.slot_writes.any(slot.index())
                    }
                    MemRoot::NoAliasParam(p) => {
                        let is_entry = entry_param(func, p).is_some_and(|ep| loc.base == MemBase::Addr(ep));
                        if is_entry && param_deref_ok(func, p, loc.offset, width) {
                            trap_ok = true;
                        }
                        let no_param_write = if is_entry {
                            !self.param_writes.overlaps(usize::from(p), loc.offset, width)
                        } else {
                            !self.param_writes.any(usize::from(p))
                        };
                        let param_ro = func.signature.params.get(usize::from(p)).is_some_and(|abi| abi.readonly);
                        if param_ro && no_param_write {
                            readonly_ok = true;
                        }
                        no_param_write && (prov.noalias_unescaped(p) || !self.unknown_mem_writes)
                    }
                    MemRoot::ReadonlyParam(p) => {
                        if entry_param(func, p).is_some_and(|ep| loc.base == MemBase::Addr(ep)) && param_deref_ok(func, p, loc.offset, width) {
                            trap_ok = true;
                        }
                        readonly_ok = true;
                        true
                    }
                    _ => false,
                };
                if self.atomics || !trap_ok {
                    return false;
                }
                if readonly_ok {
                    return true;
                }
                !self.calls && (!self.writes || root_ok)
            }
            _ => false,
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

fn check_promote_key(mem: &LoopMem, cx: &PromoteCx<'_>, accesses: &[LoopAccess], sample: LoopAccess) -> Option<PromoteCand> {
    let (loc, ty) = (sample.loc, sample.ty);
    if !(ty.is_small_int() || ty.is_float()) {
        return None;
    }
    let safe_across_calls = match loc.root {
        MemRoot::Slot(s) => cx.prov.slot_unescaped(s),
        MemRoot::NoAliasParam(p) => cx.prov.noalias_unescaped(p),
        _ => false,
    };
    if (mem.calls && !safe_across_calls) || mem.copy_locs.iter().any(|&cloc| may_alias(loc, cloc, cx.prov)) {
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

use rustc_hash::FxHashMap as HashMap;

use super::{MemBase, MemLoc, MemRoot, Provenance};
use crate::ir::{AbiParam, AtomicOrd, Binary, Function, Inst, Op, StackSlot, Type, Value};
use crate::reloc::RuntimeSymbol;

pub(crate) fn may_alias(a: MemLoc, b: MemLoc, prov: &Provenance) -> bool {
    may_alias_at(a, b, prov, None)
}

/// Whether `a` and `b` may alias at instruction `at`, using slot escape frontiers when `at` is known.
pub(crate) fn may_alias_at(a: MemLoc, b: MemLoc, prov: &Provenance, at: Option<Inst>) -> bool {
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
        (
            MemRoot::ThreadCell(_),
            MemRoot::Slot(_) | MemRoot::Global(_) | MemRoot::UniqueHeap(_),
        )
        | (
            MemRoot::Slot(_) | MemRoot::Global(_) | MemRoot::UniqueHeap(_),
            MemRoot::ThreadCell(_),
        ) => false,
        (MemRoot::Slot(sa), MemRoot::Slot(sb)) => sa == sb,
        (MemRoot::NoAliasParam(pa), MemRoot::NoAliasParam(pb)) => pa == pb,
        (MemRoot::Global(ga), MemRoot::Global(gb)) => ga == gb,
        (
            MemRoot::NoAliasParam(_),
            MemRoot::Slot(_) | MemRoot::Global(_) | MemRoot::ReadonlyParam(_),
        )
        | (
            MemRoot::Slot(_) | MemRoot::Global(_) | MemRoot::ReadonlyParam(_),
            MemRoot::NoAliasParam(_),
        )
        | (
            MemRoot::UniqueHeap(_),
            MemRoot::Slot(_)
            | MemRoot::NoAliasParam(_)
            | MemRoot::Global(_)
            | MemRoot::ReadonlyParam(_),
        )
        | (
            MemRoot::Slot(_)
            | MemRoot::NoAliasParam(_)
            | MemRoot::Global(_)
            | MemRoot::ReadonlyParam(_),
            MemRoot::UniqueHeap(_),
        ) => false,
        (MemRoot::UniqueHeap(Some(owner)), _) if b.base == MemBase::Addr(owner) => false,
        (_, MemRoot::UniqueHeap(Some(owner))) if a.base == MemBase::Addr(owner) => false,
        (MemRoot::NoAliasParam(p), MemRoot::Unknown)
        | (MemRoot::Unknown, MemRoot::NoAliasParam(p)) => !prov.noalias_unescaped(p),
        (MemRoot::Slot(_), _) if prov.is_incoming(b.base) => false,
        (_, MemRoot::Slot(_)) if prov.is_incoming(a.base) => false,
        (MemRoot::Slot(s), _) | (_, MemRoot::Slot(s)) => {
            !prov.slot_addr_unescaped(s) && !prov.slot_return_only(s) && prov.slot_escaped_at(s, at)
        }
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
    off >= 0
        && (off as u64)
            .checked_add(u64::from(bytes))
            .is_some_and(|end| end <= u64::from(size))
}

/// Whether `op` is a `StackStore` whose byte range in `slot` overlaps `[off, off + bytes)`.
pub(crate) fn stack_store_overlaps(
    func: &Function,
    op: &Op,
    slot: StackSlot,
    (off, bytes): (i32, u32),
) -> Option<bool> {
    let Op::StackStore(val, wslot, woff) = *op else {
        return None;
    };
    Some(
        wslot == slot
            && MemLoc::slot(slot, i64::from(off), bytes).overlaps(MemLoc::slot(
                wslot,
                i64::from(woff),
                func.ty(val).bytes(),
            )),
    )
}

/// Whether `op` is a volatile or atomic memory op or fence.
pub(crate) fn is_atomic_barrier(op: &Op) -> bool {
    matches!(
        op,
        Op::VolatileLoad(..)
            | Op::VolatileStore(..)
            | Op::AtomicLoad(..)
            | Op::AtomicStore(..)
            | Op::AtomicRmw(..)
            | Op::AtomicCas(..)
            | Op::Fence(..)
    )
}

/// Whether `op` prevents hoisting, sinking, or rematerializing memory accesses across it.
pub(crate) fn is_hard_barrier(op: &Op) -> bool {
    is_atomic_barrier(op)
        || matches!(
            op,
            Op::Call { .. }
                | Op::TryCall { .. }
                | Op::Trap
                | Op::Trapz(_)
                | Op::Assume(_)
                | Op::Binary(
                    Binary::Sdiv | Binary::Udiv | Binary::Srem | Binary::Urem,
                    ..
                )
        )
}

fn stack_store_clobbers(
    func: &Function,
    op: &Op,
    (val, slot, off): (Value, StackSlot, i32),
    loc: MemLoc,
    prov: &Provenance,
) -> bool {
    if let MemBase::Slot(lslot) = loc.base
        && loc.root == MemRoot::Slot(lslot)
        && let Ok(loff) = i32::try_from(loc.offset)
        && let Some(hit) = stack_store_overlaps(func, op, lslot, (loff, loc.size))
    {
        return hit;
    }
    may_alias(
        MemLoc::slot(slot, i64::from(off), func.ty(val).bytes()),
        loc,
        prov,
    )
}

pub(crate) fn clobbers(func: &Function, op: &Op, loc: MemLoc, prov: &Provenance) -> bool {
    clobbers_op(func, (op, None), loc, prov)
}

fn block_clobbers(addr: Value, loc: MemLoc, prov: &Provenance, at: Option<Inst>) -> bool {
    let w = prov.loc(addr, 0, u32::MAX);
    w.base == loc.base
        || (w.root != MemRoot::Unknown && w.root == loc.root)
        || may_alias_at(w, loc, prov, at)
}

fn slot_unescaped_at(loc: MemLoc, at: Option<Inst>, prov: &Provenance) -> bool {
    matches!(loc.root, MemRoot::Slot(s) if !prov.slot_is_escaped_at(s, at))
}

fn noalias_immune_to_atomic(op: &Op, loc: MemLoc, prov: &Provenance) -> bool {
    let MemRoot::NoAliasParam(p) = loc.root else {
        return false;
    };
    match *op {
        Op::AtomicLoad(..) | Op::Fence(_) => true,
        Op::AtomicStore(_, addr, _)
        | Op::AtomicRmw(_, _, addr, _)
        | Op::AtomicCas(_, addr, _, _) => {
            prov.noalias_unescaped(p) && prov.loc(addr, 0, 1).root != MemRoot::NoAliasParam(p)
        }
        _ => false,
    }
}

/// Whether `op`, run as instruction `at` when known, may overwrite `loc`; `at` rules out slots whose address leaves only later.
pub(crate) fn clobbers_op(
    func: &Function,
    (op, at): (&Op, Option<Inst>),
    loc: MemLoc,
    prov: &Provenance,
) -> bool {
    match *op {
        Op::StackStore(val, slot, off) => {
            stack_store_clobbers(func, op, (val, slot, off), loc, prov)
        }
        Op::Store(val, addr, off) => may_alias_at(
            prov.loc(addr, i64::from(off), func.ty(val).bytes()),
            loc,
            prov,
            at,
        ),
        Op::Move(dst, _, bytes) => may_alias_at(prov.loc(dst, 0, bytes), loc, prov, at),
        Op::Copy(dst, _, _) => block_clobbers(dst, loc, prov, at),
        Op::Call { callee, .. } if func.is_fresh_alloc(callee) => false,
        Op::Call {
            callee, ref args, ..
        } if let Some(rt) = prov.frame_runtime(callee) => {
            loc.root != MemRoot::CallerLoc
                && rt == RuntimeSymbol::Free
                && args
                    .first()
                    .is_none_or(|&freed| block_clobbers(freed, loc, prov, at))
        }
        Op::Call { sig, ref args, .. } | Op::TryCall { sig, ref args, .. } => match loc.root {
            MemRoot::Slot(s) => {
                if matches!(*op, Op::TryCall { .. }) {
                    !prov.slot_unescaped(s) && !prov.slot_call_readonly(s)
                } else {
                    !prov.slot_addr_unescaped(s)
                        && prov.slot_escaped_at(s, at)
                        && !prov.slot_call_readonly(s)
                        && (!prov.slot_return_only(s)
                            || args
                                .iter()
                                .any(|&arg| prov.loc(arg, 0, 1).root == MemRoot::Slot(s)))
                }
            }
            MemRoot::NoAliasParam(p) => {
                !prov.noalias_unescaped(p)
                    || call_writes_noalias(&func.sigs[sig.index()].params, args, prov, p)
            }
            _ => matches!(*op, Op::TryCall { .. }) || !prov.private_frame(loc),
        },
        _ => {
            is_atomic_barrier(op)
                && loc.root != MemRoot::CallerLoc
                && !(prov.private_frame(loc) && !prov.touches_frame(op))
                && !slot_unescaped_at(loc, at, prov)
                && !noalias_immune_to_atomic(op, loc, prov)
        }
    }
}

/// Whether `op` may overwrite `loc` for load forwarding; `Relaxed` and `Release` atomics cannot acquire writes from other threads.
pub(crate) fn clobbers_load(
    func: &Function,
    op: (&Op, Option<Inst>),
    loc: MemLoc,
    prov: &Provenance,
) -> bool {
    match *op.0 {
        Op::AtomicLoad(_, AtomicOrd::Relaxed) => false,
        Op::AtomicStore(val, addr, AtomicOrd::Relaxed | AtomicOrd::Release)
        | Op::AtomicRmw(_, AtomicOrd::Relaxed | AtomicOrd::Release, addr, val)
        | Op::AtomicCas(AtomicOrd::Relaxed | AtomicOrd::Release, addr, _, val) => {
            !slot_unescaped_at(loc, op.1, prov)
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
        Self {
            by_base: HashMap::default(),
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.by_base.is_empty()
    }

    pub(crate) fn get(&self, base: MemBase, offset: i64, ty: Type) -> Option<Value> {
        self.by_base
            .get(&base)?
            .1
            .iter()
            .find(|&&(off, t, _)| off == offset && t == ty)
            .map(|&(_, _, val)| val)
    }

    pub(crate) fn entries_for(&self, base: MemBase) -> &[(i64, Type, Value)] {
        self.by_base
            .get(&base)
            .map_or(&[], |entry| entry.1.as_slice())
    }

    pub(crate) fn remember(&mut self, loc: MemLoc, ty: Type, value: Value) {
        let (_, list) = self
            .by_base
            .entry(loc.base)
            .or_insert_with(|| (loc.root, Vec::new()));
        if let Some(entry) = list
            .iter_mut()
            .find(|(off, t, _)| *off == loc.offset && *t == ty)
        {
            entry.2 = value;
        } else {
            list.push((loc.offset, ty, value));
        }
    }

    pub(crate) fn write(&mut self, loc: MemLoc, ty: Type, value: Value, prov: &Provenance) {
        self.write_at(loc, ty, value, prov, None);
    }

    pub(crate) fn write_at(
        &mut self,
        loc: MemLoc,
        ty: Type,
        value: Value,
        prov: &Provenance,
        at: Option<Inst>,
    ) {
        self.by_base.retain(|&base, (root, entries)| {
            entries.retain(|&(off, t, _)| {
                !may_alias_at(loc, MemLoc::new(*root, base, off, t.bytes()), prov, at)
            });
            !entries.is_empty()
        });
        self.remember(loc, ty, value);
    }

    pub(crate) fn invalidate(
        &mut self,
        func: &Function,
        op: (&Op, Option<Inst>),
        prov: &Provenance,
    ) {
        self.by_base.retain(|&base, (root, entries)| {
            entries.retain(|&(off, t, _)| {
                !clobbers_load(func, op, MemLoc::new(*root, base, off, t.bytes()), prov)
            });
            !entries.is_empty()
        });
    }
}

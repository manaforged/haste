use crate::cfg::Loop;
use crate::ir::{Binary, Block, Function, Op, StackSlot, Type, Value};

use super::super::{MemBase, MemLoc, MemRoot, Provenance, may_alias};
use super::LoopMem;

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

impl LoopMem {
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
        let cx = PromoteCx {
            func,
            each,
            prov,
            home,
        };
        let accesses = collect_loop_accesses(&cx);
        let mut cands = Vec::new();
        for &acc in &accesses {
            let key = (acc.loc.base, acc.loc.offset, acc.ty);
            if cands
                .iter()
                .any(|c: &PromoteCand| (c.loc.base, c.loc.offset, c.ty) == key)
            {
                continue;
            }
            let Some(cand) = check_promote_key(self, &cx, &accesses, acc) else {
                continue;
            };
            cands.push(cand);
        }
        cands
    }
}

fn loop_inst_access(
    func: &Function,
    op: &Op,
    res: Option<Value>,
    prov: &Provenance,
    head_prefix: bool,
) -> Option<LoopAccess> {
    match (op, res) {
        (&Op::StackLoad(slot, off), Some(r)) => {
            let ty = func.ty(r);
            Some(LoopAccess {
                loc: MemLoc::slot(slot, i64::from(off), ty.bytes()),
                ty,
                is_write: false,
                head_prefix,
            })
        }
        (&Op::Load(addr, off), Some(r)) => {
            let ty = func.ty(r);
            Some(LoopAccess {
                loc: prov.loc(addr, i64::from(off), ty.bytes()),
                ty,
                is_write: false,
                head_prefix,
            })
        }
        (&Op::StackStore(val, slot, off), _) => {
            let ty = func.ty(val);
            Some(LoopAccess {
                loc: MemLoc::slot(slot, i64::from(off), ty.bytes()),
                ty,
                is_write: true,
                head_prefix,
            })
        }
        (&Op::Store(val, addr, off), _) => {
            let ty = func.ty(val);
            Some(LoopAccess {
                loc: prov.loc(addr, i64::from(off), ty.bytes()),
                ty,
                is_write: true,
                head_prefix,
            })
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
            if let Some(acc) =
                loop_inst_access(cx.func, &data.op, data.result, cx.prov, head_prefix)
            {
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
                    | Op::Binary(
                        Binary::Sdiv | Binary::Udiv | Binary::Srem | Binary::Urem,
                        ..
                    )
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

fn check_promote_key(
    mem: &LoopMem,
    cx: &PromoteCx<'_>,
    accesses: &[LoopAccess],
    sample: LoopAccess,
) -> Option<PromoteCand> {
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
    Some(PromoteCand {
        loc,
        base,
        ty,
        offset,
    })
}

fn promote_base_eligible(
    cx: &PromoteCx<'_>,
    loc: MemLoc,
    ty: Type,
    head_accessed: bool,
) -> Option<PromoteBase> {
    let end = loc.offset.checked_add(i64::from(ty.bytes()))?;
    if loc.offset < 0 {
        return None;
    }
    match loc.base {
        MemBase::Slot(slot) => (cx.prov.slot_unescaped(slot)
            && end as u64 <= u64::from(cx.func.slots[slot.index()].size))
        .then_some(PromoteBase::Slot(slot)),
        MemBase::Addr(addr) => {
            let MemRoot::NoAliasParam(p) = loc.root else {
                return None;
            };
            let abi = cx.func.signature.params.get(usize::from(p))?;
            if abi.readonly || !cx.prov.noalias_unescaped(p) {
                return None;
            }
            let invariant = cx
                .home
                .get(addr.index())
                .copied()
                .flatten()
                .is_none_or(|at| !cx.each.body.get(at.index()).copied().unwrap_or(false));
            if !invariant {
                return None;
            }
            let deref_ok =
                end as u64 <= u64::from(cx.prov.ro_bytes.get(addr.index()).copied().unwrap_or(0));
            (head_accessed || deref_ok).then_some(PromoteBase::Addr(addr))
        }
        MemBase::Global(_) => None,
    }
}

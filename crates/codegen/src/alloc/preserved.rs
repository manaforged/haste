use crate::ir::{Function, Op, Value};
use crate::reloc::{FRAME_CACHE_LIMIT, RuntimeSymbol};

use super::{CallTarget, Location, Regs};

/// Returns `(lookup, [slot, size, init])` when a non-try `ThreadCell` call has static arguments eligible for inline TLS with register-preserving cold miss stub.
pub(crate) fn cold_thread_cell<R>(
    func: &Function,
    (callee, args): (Value, &[Value]),
    through: &[Option<CallTarget>],
    locations: &[Location<R>],
    elided: &[bool],
) -> Option<(u32, [Value; 3])> {
    let Some(CallTarget::Symbol(lookup)) = through[callee.index()] else { return None };
    if !func.is_thread_cell(callee) {
        return None;
    }
    let [slot, size, init] = *args else { return None };
    let static_ok = |v: Value| !elided[v.index()] && matches!(locations[v.index()], Location::Imm(_) | Location::Symbol(_) | Location::SlotAddr(_));
    if !(static_ok(slot) && static_ok(size) && static_ok(init)) {
        return None;
    }
    matches!(locations[slot.index()], Location::Symbol(_) | Location::Imm(0)).then_some((lookup, [slot, size, init]))
}

pub(crate) fn cold_thread_cell_op<R>(
    func: &Function,
    op: &Op,
    through: &[Option<CallTarget>],
    locations: &[Location<R>],
    elided: &[bool],
) -> Option<(u32, [Value; 3])> {
    let Op::Call { callee, args, .. } = op else { return None };
    cold_thread_cell(func, (*callee, args), through, locations, elided)
}

/// Returns `(rt, bytes)` when a non-try `Malloc`/`Free` frame-cache call is eligible for inline TLS with register-preserving cold miss stub.
pub(crate) fn cold_cached_frame<R>(
    (callee, args): (Value, &[Value]),
    through: &[Option<CallTarget>],
    locations: &[Location<R>],
) -> Option<(RuntimeSymbol, u32)> {
    let Some(CallTarget::Runtime(rt)) = through[callee.index()] else { return None };
    if !matches!(rt, RuntimeSymbol::Malloc | RuntimeSymbol::Free) {
        return None;
    }
    if rt == RuntimeSymbol::Free && args.len() < 2 {
        return None;
    }
    let Some(&Location::Imm(bytes)) = args.last().map(|&b| &locations[b.index()]) else { return None };
    ((0..i64::from(FRAME_CACHE_LIMIT)).contains(&bytes) && bytes % 16 == 0).then_some((rt, bytes as u32))
}

pub(crate) fn cold_cached_frame_op<R>(
    op: &Op,
    through: &[Option<CallTarget>],
    locations: &[Location<R>],
) -> Option<(RuntimeSymbol, u32)> {
    let Op::Call { callee, args, .. } = op else { return None };
    cold_cached_frame((*callee, args), through, locations)
}

/// Reports whether `op` is eligible for register-preserving inline TLS (`ThreadCell` or `Malloc`/`Free` frame cache).
pub(crate) fn is_preserved_call_op<R>(
    func: &Function,
    op: &Op,
    through: &[Option<CallTarget>],
    locations: &[Location<R>],
    elided: &[bool],
) -> bool {
    cold_thread_cell_op(func, op, through, locations, elided).is_some()
        || cold_cached_frame_op(op, through, locations).is_some()
}

/// Returns the caller-saved leaf registers actually allocated in `locations` (plus optional dict reg) that a preserved cold stub must save.
pub(crate) fn used_preserved_leaf_regs<R: Copy + PartialEq>(regs: &Regs<R>, locations: &[Location<R>], include_dict: bool) -> Vec<R> {
    let mut eligible = [false; 32];
    let mut need_mask = 0u32;
    for (i, &r) in regs.leaf.iter().enumerate() {
        if !regs.preserved_clobbers.contains(&r) {
            eligible[i] = true;
            need_mask |= 1u32 << i;
        }
    }
    let mut used_mask = 0u32;
    if include_dict
        && let Some(dict) = regs.dict
        && let Some(i) = regs.leaf.iter().position(|&r| r == dict)
        && eligible[i]
    {
        used_mask |= 1u32 << i;
    }
    if used_mask != need_mask {
        for loc in locations {
            if let Location::Reg(at) = *loc
                && let Some(i) = regs.leaf.iter().position(|&r| r == at)
                && eligible[i]
            {
                used_mask |= 1u32 << i;
                if used_mask == need_mask {
                    break;
                }
            }
        }
    }
    regs.leaf.iter().enumerate().filter(|&(i, _)| (used_mask & (1u32 << i)) != 0).map(|(_, &r)| r).collect()
}

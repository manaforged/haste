use crate::ir::{Block, Function, Inst, Op, StackSlot, Value};
use crate::mem::{Provenance, may_alias};

use super::{Access, accesses_at, collect_sret_info, resolve_slot, retarget_slot};

/// Builds each frame slot of a coroutine that one block fills and then moves out whole straight into the move's target, so no copy stays to move through the coroutine's frame.
pub(crate) fn build_in_place(func: &mut Function) -> bool {
    let Some(entry) = func.entry else {
        return false;
    };
    let Some(keep) = func
        .coroutine
        .as_ref()
        .map(|co| [co.resumes.as_slice(), co.done.as_slice()].concat())
    else {
        return false;
    };
    let mut changed = crate::merge::join_jumps(func, &keep);
    if changed {
        func.bump_cfg();
    }
    while let Some((slot, to, inst)) = plan(func, entry) {
        hoist_addr(func, inst, to);
        retarget_slot(func, slot, to, &[]);
        let gone = func
            .blocks
            .iter()
            .position(|data| data.insts.contains(&inst));
        if let Some(block) = gone {
            func.blocks[block].insts.retain(|&other| other != inst);
        }
        func.insts[inst.index()].op = Op::Trap;
        changed = true;
    }
    if changed {
        func.bump_cfg();
    }
    changed
}

/// A slot that only stores and moves in one block write before one whole move copies it out, with the move's target address and the move, when nothing between the first write and the move can see that target.
fn plan(func: &Function, entry: Block) -> Option<(StackSlot, Value, Inst)> {
    let info = collect_sret_info(func, entry);
    let cfg = func.cfg();
    let prov = Provenance::new(func, &cfg);
    for (at, data) in func.blocks.iter().enumerate() {
        for (pos, &inst) in data.insts.iter().enumerate() {
            let Op::Move(to, from, bytes) = func.insts[inst.index()].op else {
                continue;
            };
            let Some(slot) = resolve_slot(func, &info.defs, from) else {
                continue;
            };
            if func.slots[slot.index()].size != bytes || resolve_base(func, &info, to) == Some(slot)
            {
                continue;
            }
            let Some(first) = writes_before(func, &info, (Block(at as u32), pos), slot) else {
                continue;
            };
            let target = prov.loc(to, 0, bytes);
            let clear = data.insts[first..pos].iter().all(|&other| {
                let op = &func.insts[other.index()].op;
                touches(func, &info, other, slot)
                    || op.is_speculable()
                    || matches!(*op, Op::Load(addr, offset) if !may_alias(prov.loc(addr, i64::from(offset), func.ty(func.insts[other.index()].result.unwrap_or(addr)).bytes()), target, &prov))
            });
            if clear && addr_ready(func, &info, (Block(at as u32), first), to) {
                return Some((slot, to, inst));
            }
        }
    }
    None
}

/// The position of the first instruction of `block` that touches `slot`, when every touch of `slot` lies in `block` before `at`, writes it, and the move at `at` is its only read.
fn writes_before(
    func: &Function,
    info: &super::SretUseInfo,
    (block, at): (Block, usize),
    slot: StackSlot,
) -> Option<usize> {
    let insts = &func.blocks[block.index()].insts;
    let mut roots: Vec<Value> = info.slot_addrs[slot.index()]
        .iter()
        .map(|&(_, _, value)| value)
        .collect();
    roots.extend(func.insts.iter().filter_map(|data| match data.op {
        Op::StackAddr(found, offset) if found == slot && offset != 0 => data.result,
        _ => None,
    }));
    let accesses = accesses_at(func, info, roots, &[(block, insts[at])], Access::Write)?;
    let direct = func
        .blocks
        .iter()
        .enumerate()
        .flat_map(|(index, data)| {
            data.insts
                .iter()
                .enumerate()
                .map(move |(pos, &inst)| (Block(index as u32), pos, inst))
        })
        .filter(|&(_, _, inst)| match func.insts[inst.index()].op {
            Op::StackLoad(found, _) => found == slot,
            Op::StackStore(_, found, _) => found == slot,
            _ => false,
        });
    let mut first = at;
    for (other, pos, inst) in direct
        .map(|(b, p, i)| (b, p, Some(i)))
        .chain(accesses.into_iter().map(|(b, p)| (b, p, None)))
    {
        let op = &func.insts[func.blocks[other.index()].insts[pos].index()].op;
        let read = inst
            .is_some_and(|inst| matches!(func.insts[inst.index()].op, Op::StackLoad(..)))
            || matches!(op, Op::Call { .. } | Op::TryCall { .. });
        if other != block || pos >= at || read {
            return None;
        }
        first = first.min(pos);
    }
    (first < at).then_some(first)
}

/// Whether `inst` is a write of `slot` through its address or a direct store.
fn touches(func: &Function, info: &super::SretUseInfo, inst: Inst, slot: StackSlot) -> bool {
    let data = &func.insts[inst.index()];
    let of_slot = |value: Value| resolve_base(func, info, value) == Some(slot);
    match data.op {
        Op::StackStore(_, found, _) | Op::StackAddr(found, _) => found == slot,
        Op::Store(_, addr, _) | Op::Move(addr, _, _) => of_slot(addr),
        Op::Binary(_, left, right) => of_slot(left) || of_slot(right),
        _ => false,
    }
}

/// The slot whose address `value` is, at any offset.
fn resolve_base(func: &Function, info: &super::SretUseInfo, value: Value) -> Option<StackSlot> {
    let def = (*info.defs.get(value.index())?)?;
    match func.insts[def.index()].op {
        Op::StackAddr(slot, _) => Some(slot),
        Op::Binary(crate::ir::Binary::Iadd, base, _) => resolve_base(func, info, base),
        _ => resolve_slot(func, &info.defs, value),
    }
}

/// Whether `to` is defined before position `first` of `block`, or is a slot address `hoist_addr` can move there.
fn addr_ready(
    func: &Function,
    info: &super::SretUseInfo,
    (block, first): (Block, usize),
    to: Value,
) -> bool {
    let Some(def) = info.defs.get(to.index()).copied().flatten() else {
        return true;
    };
    let insts = &func.blocks[block.index()].insts;
    match insts.iter().position(|&inst| inst == def) {
        Some(pos) => pos < first || matches!(func.insts[def.index()].op, Op::StackAddr(..)),
        None => true,
    }
}

/// Moves the slot address `to` that the move `inst` targets to the start of its block when it is defined there after other uses would need it.
fn hoist_addr(func: &mut Function, inst: Inst, to: Value) {
    let Some(block) = func
        .blocks
        .iter()
        .position(|data| data.insts.contains(&inst))
    else {
        return;
    };
    let Some(def) = func.blocks[block]
        .insts
        .iter()
        .copied()
        .find(|&other| func.insts[other.index()].result == Some(to))
    else {
        return;
    };
    if matches!(func.insts[def.index()].op, Op::StackAddr(..)) {
        func.blocks[block].insts.retain(|&other| other != def);
        func.blocks[block].insts.insert(0, def);
    }
}

use super::join_slots::resolve_addrs;
use super::subword::eval_consts;
use crate::copy::MOVE_INLINE;
use crate::ir::{Block, Function, Inst, Op, StackSlot, Type, Value};

/// Gives each call on a path that ends in a trap its own copy of every local slot it borrows, so a slot that only those calls let escape can still be promoted.
pub(super) fn run(func: &mut Function) -> bool {
    let cold = cold_calls(func);
    if cold.is_empty() {
        return false;
    }
    let consts = eval_consts(func);
    let (addrs, _) = resolve_addrs(func, &consts);
    let slot_of = |v: Value| addrs.get(v.index()).copied().flatten();
    let eligible = eligible_slots(func, &cold, slot_of);
    let mut changed = false;
    for &(block, call) in &cold {
        let Op::Call { args, .. } = &func.insts[call.index()].op else { continue };
        let mut args = args.clone();
        let mut before = Vec::new();
        let mut copies: Vec<(StackSlot, StackSlot)> = Vec::new();
        for arg in &mut args {
            let Some((slot, offset)) = slot_of(*arg).filter(|(slot, _)| eligible[slot.index()]) else { continue };
            let Ok(offset) = i32::try_from(offset) else { continue };
            let ty = func.ty(*arg);
            let copy = match copies.iter().find(|&&(from, _)| from == slot) {
                Some(&(_, copy)) => copy,
                None => {
                    let data = func.slots[slot.index()].clone();
                    let size = data.size;
                    func.slots.push(data);
                    let copy = StackSlot(func.slots.len() as u32 - 1);
                    let from = stack_addr(func, (slot, 0, ty), &mut before);
                    let to = stack_addr(func, (copy, 0, ty), &mut before);
                    before.push(func.add(Op::Move(to, from, size), None));
                    copies.push((slot, copy));
                    copy
                }
            };
            *arg = stack_addr(func, (copy, offset, ty), &mut before);
        }
        if before.is_empty() {
            continue;
        }
        if let Op::Call { args: old, .. } = &mut func.insts[call.index()].op {
            *old = args;
        }
        let insts = &mut func.blocks[block.index()].insts;
        let at = insts.len() - 2;
        insts.splice(at..at, before);
        changed = true;
    }
    changed
}

/// Every plain call that is the last instruction before a block's trap.
fn cold_calls(func: &Function) -> Vec<(Block, Inst)> {
    let mut out = Vec::new();
    for (index, block) in func.blocks.iter().enumerate() {
        if let [.., call, trap] = block.insts[..]
            && matches!(func.insts[trap.index()].op, Op::Trap)
            && matches!(func.insts[call.index()].op, Op::Call { .. })
        {
            out.push((Block(index as u32), call));
        }
    }
    out
}

/// Which slots are read, borrowed by a cold call, small enough to copy, and otherwise only loaded, stored, moved or offset.
fn eligible_slots(func: &Function, cold: &[(Block, Inst)], slot_of: impl Fn(Value) -> Option<(StackSlot, i64)>) -> Vec<bool> {
    let count = func.slots.len();
    let (mut read, mut borrowed, mut bad) = (vec![false; count], vec![false; count], vec![false; count]);
    for &inst in func.blocks.iter().flat_map(|block| &block.insts) {
        let data = &func.insts[inst.index()];
        if let Op::StackLoad(slot, _) = data.op {
            read[slot.index()] = true;
        }
        let derived = data.result.is_some_and(|result| slot_of(result).is_some());
        let in_cold = cold.iter().any(|&(_, call)| call == inst);
        data.op.for_operands(|v| {
            let Some((slot, _)) = slot_of(v) else { return };
            let s = slot.index();
            let ok = match &data.op {
                Op::Load(..) => {
                    read[s] = true;
                    true
                }
                Op::Store(value, ..) => *value != v,
                Op::Move(..) => true,
                Op::Copy(_, _, len) => *len != v,
                Op::Call { callee, .. } if in_cold => {
                    borrowed[s] = true;
                    *callee != v
                }
                _ => derived,
            };
            bad[s] |= !ok;
        });
    }
    (0..count).map(|s| read[s] && borrowed[s] && !bad[s] && func.slots[s].size <= MOVE_INLINE).collect()
}

fn stack_addr(func: &mut Function, (slot, offset, ty): (StackSlot, i32, Type), out: &mut Vec<Inst>) -> Value {
    let addr = func.value(ty);
    out.push(func.add(Op::StackAddr(slot, offset), Some(addr)));
    addr
}

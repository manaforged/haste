use super::join_slots::resolve_addrs;
use super::subword::eval_consts;
use crate::copy::MOVE_INLINE;
use crate::ir::{Block, Function, Inst, Op, StackSlot, Type, Value};

type SlotOf<'a> = &'a dyn Fn(Value) -> Option<(StackSlot, i64)>;

/// Gives each block that ends in a call and a trap its own copy of every local slot whose address it passes on, so a slot that only such blocks let escape can still be promoted.
pub(super) fn run(func: &mut Function) -> bool {
    let cold = cold_blocks(func);
    if cold.is_empty() {
        return false;
    }
    let consts = eval_consts(func);
    let (addrs, _) = resolve_addrs(func, &consts);
    let slot_of = |v: Value| addrs.get(v.index()).copied().flatten();
    let eligible = eligible_slots(func, &cold, &slot_of);
    let mut changed = false;
    for &block in &cold {
        changed |= copy_in(func, block, &eligible, &slot_of);
    }
    changed
}

/// Every block whose last two instructions are a plain call and a trap.
fn cold_blocks(func: &Function) -> Vec<Block> {
    let mut out = Vec::new();
    for (index, block) in func.blocks.iter().enumerate() {
        if let [.., call, trap] = block.insts[..]
            && matches!(func.insts[trap.index()].op, Op::Trap)
            && matches!(func.insts[call.index()].op, Op::Call { .. })
        {
            out.push(Block(index as u32));
        }
    }
    out
}

/// The operands of `op` through which an address leaves the function: call arguments and stored values.
fn passes_on(op: &Op, v: Value) -> bool {
    match op {
        Op::Call { args, .. } => args.contains(&v),
        Op::Store(value, ..) => *value == v,
        _ => false,
    }
}

/// Which slots are read, passed on by a cold block, small enough to copy, and otherwise only loaded, stored, moved or offset.
fn eligible_slots(func: &Function, cold: &[Block], slot_of: SlotOf<'_>) -> Vec<bool> {
    let count = func.slots.len();
    let (mut read, mut borrowed, mut bad) =
        (vec![false; count], vec![false; count], vec![false; count]);
    for (index, block) in func.blocks.iter().enumerate() {
        let in_cold = cold.contains(&Block(index as u32));
        for &inst in &block.insts {
            let data = &func.insts[inst.index()];
            if let Op::StackLoad(slot, _) = data.op {
                read[slot.index()] = true;
            }
            let derived = data.result.is_some_and(|result| slot_of(result).is_some());
            data.op.for_operands(|v| {
                let Some((slot, _)) = slot_of(v) else { return };
                let s = slot.index();
                if in_cold && passes_on(&data.op, v) {
                    borrowed[s] = true;
                    return;
                }
                let ok = match &data.op {
                    Op::Load(..) => {
                        read[s] = true;
                        true
                    }
                    Op::Store(value, ..) => *value != v,
                    Op::Move(..) => true,
                    Op::Copy(_, _, len) => *len != v,
                    _ => derived,
                };
                bad[s] |= !ok;
            });
        }
    }
    (0..count)
        .map(|s| read[s] && borrowed[s] && !bad[s] && func.slots[s].size <= MOVE_INLINE)
        .collect()
}

/// Whether `op` writes `slot` other than through a call, whose borrows of it get the copy.
fn writes(op: &Op, slot: StackSlot, slot_of: SlotOf<'_>) -> bool {
    let hits = |addr: &Value| slot_of(*addr).is_some_and(|(s, _)| s == slot);
    match op {
        Op::StackStore(_, s, _) => *s == slot,
        Op::Store(_, addr, _) | Op::Move(addr, ..) | Op::Copy(addr, ..) => hits(addr),
        _ => false,
    }
}

/// Copies each eligible slot that `block` passes on, right before the first instruction that passes it, and passes the copy instead.
fn copy_in(func: &mut Function, block: Block, eligible: &[bool], slot_of: SlotOf<'_>) -> bool {
    let insts = func.blocks[block.index()].insts.clone();
    let mut copies: Vec<(StackSlot, StackSlot, usize)> = Vec::new();
    for (pos, &inst) in insts.iter().enumerate() {
        let op = &func.insts[inst.index()].op;
        let mut found = Vec::new();
        op.for_operands(|v| {
            if let Some((slot, _)) =
                slot_of(v).filter(|(slot, _)| eligible[slot.index()] && passes_on(op, v))
            {
                found.push(slot);
            }
        });
        for slot in found {
            if copies.iter().any(|&(from, ..)| from == slot)
                || insts[pos + 1..]
                    .iter()
                    .any(|&later| writes(&func.insts[later.index()].op, slot, slot_of))
            {
                continue;
            }
            let data = func.slots[slot.index()].clone();
            func.slots.push(data);
            copies.push((slot, StackSlot(func.slots.len() as u32 - 1), pos));
        }
    }
    if copies.is_empty() {
        return false;
    }
    let mut out = Vec::with_capacity(insts.len() + copies.len() * 4);
    for (pos, &inst) in insts.iter().enumerate() {
        for &(from, to, _) in copies.iter().filter(|&&(.., at)| at == pos) {
            let size = func.slots[from.index()].size;
            let src = stack_addr(func, (from, 0, Type::I64), &mut out);
            let dst = stack_addr(func, (to, 0, Type::I64), &mut out);
            out.push(func.add(Op::Move(dst, src, size), None));
        }
        let mut op = func.insts[inst.index()].op.clone();
        let mut swaps = Vec::new();
        op.for_operands(|v| {
            if passes_on(&op, v)
                && let Some((slot, offset)) = slot_of(v)
                && let Some(&(_, to, at)) = copies.iter().find(|&&(from, ..)| from == slot)
                && at <= pos
                && let Ok(offset) = i32::try_from(offset)
            {
                swaps.push((v, to, offset));
            }
        });
        for (v, to, offset) in swaps {
            let ty = func.ty(v);
            let addr = stack_addr(func, (to, offset, ty), &mut out);
            replace_passed(&mut op, v, addr);
        }
        func.insts[inst.index()].op = op;
        out.push(inst);
    }
    func.blocks[block.index()].insts = out;
    true
}

/// Replaces `old` with `new` where `op` passes it on.
fn replace_passed(op: &mut Op, old: Value, new: Value) {
    match op {
        Op::Call { args, .. } => args
            .iter_mut()
            .filter(|arg| **arg == old)
            .for_each(|arg| *arg = new),
        Op::Store(value, ..) if *value == old => *value = new,
        _ => {}
    }
}

fn stack_addr(
    func: &mut Function,
    (slot, offset, ty): (StackSlot, i32, Type),
    out: &mut Vec<Inst>,
) -> Value {
    let addr = func.value(ty);
    out.push(func.add(Op::StackAddr(slot, offset), Some(addr)));
    addr
}

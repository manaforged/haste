use crate::frame::MAX_ALIGN_SHIFT as FRAME_ALIGN_SHIFT;
use crate::ir::{Binary, Function, Op, StackSlot, Type, Value};

/// Gives each frame slot that asks for more alignment than the frame keeps a larger slot and an address rounded up inside it, computed once at entry, which its accesses then go through.
pub(crate) fn lower(func: &mut Function) {
    let Some(entry) = func.entry else { return };
    if func.slots.iter().all(|slot| slot.align_shift <= FRAME_ALIGN_SHIFT) {
        return;
    }
    let mut used = vec![false; func.slots.len()];
    for block in &func.blocks {
        for inst in &block.insts {
            if let Op::StackAddr(slot, _) | Op::StackLoad(slot, _) | Op::StackStore(_, slot, _) = func.insts[inst.index()].op {
                used[slot.index()] = true;
            }
        }
    }
    let wide: Vec<usize> = (0..func.slots.len()).filter(|&slot| used[slot] && func.slots[slot].align_shift > FRAME_ALIGN_SHIFT).collect();
    if wide.is_empty() {
        return;
    }
    let mut bases: Vec<Option<Value>> = vec![None; func.slots.len()];
    let mut prelude = Vec::new();
    for &slot in &wide {
        let align = 1_i64 << func.slots[slot].align_shift;
        func.slots[slot].size += (align as u32) - (1 << FRAME_ALIGN_SHIFT);
        func.slots[slot].align_shift = FRAME_ALIGN_SHIFT;
        let raw = func.value(Type::I64);
        prelude.push(func.add(Op::StackAddr(StackSlot(slot as u32), 0), Some(raw)));
        let bump = constant(func, &mut prelude, align - 1);
        let up = func.value(Type::I64);
        prelude.push(func.add(Op::Binary(Binary::Iadd, raw, bump), Some(up)));
        let mask = constant(func, &mut prelude, -align);
        let base = func.value(Type::I64);
        prelude.push(func.add(Op::Binary(Binary::Band, up, mask), Some(base)));
        bases[slot] = Some(base);
    }
    for block in 0..func.blocks.len() {
        let mut at = 0;
        while at < func.blocks[block].insts.len() {
            let inst = func.blocks[block].insts[at];
            let op = func.insts[inst.index()].op.clone();
            match op {
                Op::StackAddr(slot, offset) if let Some(base) = bases[slot.index()] => {
                    let mut added = Vec::new();
                    let off = constant(func, &mut added, i64::from(offset));
                    func.insts[inst.index()].op = Op::Binary(Binary::Iadd, base, off);
                    let count = added.len();
                    func.blocks[block].insts.splice(at..at, added);
                    at += count;
                }
                Op::StackLoad(slot, offset) if let Some(base) = bases[slot.index()] => func.insts[inst.index()].op = Op::Load(base, offset),
                Op::StackStore(value, slot, offset) if let Some(base) = bases[slot.index()] => func.insts[inst.index()].op = Op::Store(value, base, offset),
                _ => {}
            }
            at += 1;
        }
    }
    let entry = entry.index();
    func.blocks[entry].insts.splice(0..0, prelude);
}

fn constant(func: &mut Function, out: &mut Vec<crate::ir::Inst>, imm: i64) -> Value {
    let value = func.value(Type::I64);
    out.push(func.add(Op::Const(imm), Some(value)));
    value
}

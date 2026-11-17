use crate::ir::{Binary, Function, Inst, Op, Type, Value};

/// Puts each 64-bit high product of a block just before the `Imul` of the same operands there, so one x86-64 `mul` yields both halves.
pub(super) fn adjoin(func: &mut Function) {
    for block in 0..func.blocks.len() {
        let mut insts = std::mem::take(&mut func.blocks[block].insts);
        let mut at = 0;
        while at < insts.len() {
            let pair = operands(func, insts[at], &[Binary::Imul])
                .and_then(|low| insts.iter().position(|&inst| operands(func, inst, &[Binary::Umulhi, Binary::Smulhi]).is_some_and(|high| high == low || high == (low.1, low.0))));
            match pair {
                Some(high) if high > at => {
                    let moved = insts.remove(high);
                    insts.insert(at, moved);
                    at += 2;
                }
                Some(high) if high + 1 < at => {
                    let moved = insts.remove(at);
                    insts.insert(high + 1, moved);
                    at += 1;
                }
                _ => at += 1,
            }
        }
        func.blocks[block].insts = insts;
    }
}

fn operands(func: &Function, inst: Inst, ops: &[Binary]) -> Option<(Value, Value)> {
    match func.insts[inst.index()].op {
        Op::Binary(op, x, y) if ops.contains(&op) && func.ty(x) == Type::I64 => Some((x, y)),
        _ => None,
    }
}

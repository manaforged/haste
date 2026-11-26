use crate::ir::{Binary, Function, Inst, Op, TargetOpId, TinyArgs, Type, Value};

/// Puts each 64-bit high product of a block just before the `Imul` of the same operands, and each carry out just after the sum of the same operands, so one x86-64 instruction yields both.
pub(super) fn adjoin(func: &mut Function) {
    for block in 0..func.blocks.len() {
        let mut insts = std::mem::take(&mut func.blocks[block].insts);
        let mut at = 0;
        while at < insts.len() {
            let pair = operands(func, insts[at], &[Binary::Imul]).and_then(|low| {
                insts.iter().position(|&inst| {
                    operands(func, inst, &[Binary::Umulhi, Binary::Smulhi])
                        .is_some_and(|high| high == low || high == (low.1, low.0))
                })
            });
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
        follow_carries(func, &mut insts);
        func.blocks[block].insts = insts;
    }
}

fn operands(func: &Function, inst: Inst, ops: &[Binary]) -> Option<(Value, Value)> {
    match func.insts[inst.index()].op {
        Op::Binary(op, x, y) if ops.contains(&op) && func.ty(x) == Type::I64 => Some((x, y)),
        _ => None,
    }
}

fn follow_carries(func: &Function, insts: &mut Vec<Inst>) {
    let carry = |inst: Inst, id: TargetOpId| match &func.insts[inst.index()].op {
        Op::TargetOp(op, args) if *op == id => Some(*args),
        _ => None,
    };
    for (sum, out) in [
        (TargetOpId::X64AddCarry, TargetOpId::X64AddCarryOut),
        (TargetOpId::X64SubBorrow, TargetOpId::X64SubBorrowOut),
    ] {
        let mut at = 0;
        while at < insts.len() {
            let args: Option<TinyArgs> = carry(insts[at], sum);
            match args.and_then(|args| {
                insts
                    .iter()
                    .position(|&inst| carry(inst, out) == Some(args))
            }) {
                Some(after) if after > at + 1 => {
                    let moved = insts.remove(after);
                    insts.insert(at + 1, moved);
                }
                Some(after) if after < at => {
                    let moved = insts.remove(at);
                    insts.insert(after, moved);
                }
                _ => {}
            }
            at += 1;
        }
    }
}

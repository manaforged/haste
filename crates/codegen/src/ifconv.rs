use crate::ir::{Binary, Block, Function, Inst, InstData, Op, Type, Value};
use crate::merge::pred_counts;

fn scalar_select(ty: Type) -> bool {
    matches!(ty, Type::I8 | Type::I16 | Type::I32 | Type::I64 | Type::F32 | Type::F64)
}

fn speculable(op: &Op) -> bool {
    match op {
        Op::Binary(bin, ..) => !matches!(bin, Binary::Sdiv | Binary::Udiv | Binary::Srem | Binary::Urem),
        Op::Const(_)
        | Op::SlotAddr(_)
        | Op::Symbol(_)
        | Op::StackAddr(..)
        | Op::Unary(..)
        | Op::Convert(..)
        | Op::Icmp(..)
        | Op::Fits(..)
        | Op::Fcmp(..)
        | Op::Select(..)
        | Op::ByteCmp(..)
        | Op::Splat(_)
        | Op::Movemask(_) => true,
        _ => false,
    }
}

fn pure_arm(func: &Function, block: Block, join: Block) -> Option<Vec<Value>> {
    let data = &func.blocks[block.index()];
    if !data.params.is_empty() || data.insts.len() > 5 {
        return None;
    }
    let (&last, body) = data.insts.split_last()?;
    let Op::Jump(target, args) = &func.insts[last.index()].op else {
        return None;
    };
    if *target != join || !body.iter().all(|inst| speculable(&func.insts[inst.index()].op)) {
        return None;
    }
    Some(args.clone())
}

fn try_head(func: &mut Function, head: Block, preds: &mut [u32]) -> bool {
    let Some(&brif_inst) = func.blocks[head.index()].insts.last() else {
        return false;
    };
    let Op::Brif(cond, then_b, then_args, else_b, else_args) = func.insts[brif_inst.index()].op.clone() else {
        return false;
    };
    if then_b == head || else_b == head || func.entry == Some(then_b) || func.entry == Some(else_b) {
        return false;
    }
    let (join_b, t_args, e_args, clear) = if then_b == else_b {
        (then_b, then_args, else_args, [None, None])
    } else if then_args.is_empty()
        && else_args.is_empty()
        && preds[then_b.index()] == 1
        && preds[else_b.index()] == 1
        && let Some(Op::Jump(join_b, _)) = func.last(then_b).cloned()
        && join_b != head
        && join_b != then_b
        && join_b != else_b
        && func.entry != Some(join_b)
        && preds[join_b.index()] == 2
        && let (Some(t), Some(e)) = (pure_arm(func, then_b, join_b), pure_arm(func, else_b, join_b))
        && func.blocks[then_b.index()].insts.len() + func.blocks[else_b.index()].insts.len() <= 6
    {
        (join_b, t, e, [Some(then_b), Some(else_b)])
    } else if then_args.is_empty()
        && preds[then_b.index()] == 1
        && preds[else_b.index()] == 2
        && let Some(t) = pure_arm(func, then_b, else_b)
    {
        (else_b, t, else_args, [Some(then_b), None])
    } else if else_args.is_empty()
        && preds[else_b.index()] == 1
        && preds[then_b.index()] == 2
        && let Some(e) = pure_arm(func, else_b, then_b)
    {
        (then_b, then_args, e, [Some(else_b), None])
    } else {
        return false;
    };

    let join_params = &func.blocks[join_b.index()].params;
    if !(1..=4).contains(&join_params.len())
        || t_args.len() != join_params.len()
        || e_args.len() != join_params.len()
        || !join_params.iter().all(|p| scalar_select(func.ty(*p)))
    {
        return false;
    }
    let param_tys: Vec<Type> = join_params.iter().map(|p| func.ty(*p)).collect();

    if let Some(old_brif) = func.blocks[head.index()].insts.pop() {
        func.insts[old_brif.index()].op = Op::Trap;
    }
    for side in clear.into_iter().flatten() {
        let mut moved = std::mem::take(&mut func.blocks[side.index()].insts);
        if let Some(old_jump) = moved.pop() {
            func.insts[old_jump.index()].op = Op::Trap;
        }
        func.blocks[head.index()].insts.extend(moved);
        preds[side.index()] = 0;
    }

    let mut selected = Vec::with_capacity(param_tys.len());
    for ((ty, t_val), e_val) in param_tys.into_iter().zip(t_args).zip(e_args) {
        if t_val == e_val {
            selected.push(t_val);
            continue;
        }
        let val = Value(func.types.len() as u32);
        func.types.push(ty);
        let inst = Inst(func.insts.len() as u32);
        func.insts.push(InstData { op: Op::Select(cond, t_val, e_val), result: Some(val) });
        func.blocks[head.index()].insts.push(inst);
        selected.push(val);
    }

    let jump = Inst(func.insts.len() as u32);
    func.insts.push(InstData { op: Op::Jump(join_b, selected), result: None });
    func.blocks[head.index()].insts.push(jump);
    preds[join_b.index()] = preds[join_b.index()].saturating_sub(1);
    true
}

/// Converts pure diamond and triangle conditional branches into branchless `Op::Select`.
pub(crate) fn run(func: &mut Function) {
    if func.entry.is_none() || func.coroutine.is_some() {
        return;
    }
    let mut preds = pred_counts(func);
    loop {
        let mut changed = false;
        for index in 0..func.blocks.len() {
            let block = Block(index as u32);
            if preds[index] > 0 || func.entry == Some(block) {
                changed |= try_head(func, block, &mut preds);
            }
        }
        if !changed {
            break;
        }
    }
}

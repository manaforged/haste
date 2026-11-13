use crate::cfg::Cfg;
use crate::ir::{Block, Function, Inst, InstData, Op, Type, Value};

fn scalar_select(ty: Type) -> bool {
    matches!(ty, Type::I8 | Type::I16 | Type::I32 | Type::I64 | Type::F32 | Type::F64)
}

struct ArmMatch {
    join: Block,
    t_args: Vec<Value>,
    e_args: Vec<Value>,
    clear: [Option<Block>; 2],
}

fn real_op_count(func: &Function, body: &[Inst]) -> usize {
    body.iter().filter(|inst| !matches!(func.insts[inst.index()].op, Op::Const(_))).count()
}

fn pure_arm(func: &Function, block: Block, join: Block) -> Option<(Vec<Value>, usize)> {
    let data = &func.blocks[block.index()];
    if !data.params.is_empty() || data.insts.len() > 12 {
        return None;
    }
    let (&last, body) = data.insts.split_last()?;
    let Op::Jump(target, args) = &func.insts[last.index()].op else {
        return None;
    };
    let ops = real_op_count(func, body);
    if *target != join || ops > 6 || !body.iter().all(|inst| func.insts[inst.index()].op.is_speculable()) {
        return None;
    }
    Some((args.clone(), ops))
}

fn match_diamond(func: &Function, cfg: &Cfg, head: Block, (then_b, else_b): (Block, Block)) -> Option<ArmMatch> {
    if cfg.pred_count(then_b) != 1 || cfg.pred_count(else_b) != 1 {
        return None;
    }
    let Some(Op::Jump(join_b, _)) = func.last(then_b).cloned() else { return None };
    if join_b == head || join_b == then_b || join_b == else_b || func.entry == Some(join_b) || cfg.pred_count(join_b) < 2 {
        return None;
    }
    let ((t, t_ops), (e, e_ops)) = (pure_arm(func, then_b, join_b)?, pure_arm(func, else_b, join_b)?);
    (t_ops + e_ops <= 8).then_some(ArmMatch { join: join_b, t_args: t, e_args: e, clear: [Some(then_b), Some(else_b)] })
}

fn match_triangle(func: &Function, cfg: &Cfg, arm: Block, join: Block) -> Option<Vec<Value>> {
    (cfg.pred_count(arm) == 1 && cfg.pred_count(join) >= 2)
        .then(|| pure_arm(func, arm, join).map(|(args, _)| args))
        .flatten()
}

fn match_arms(func: &Function, cfg: &Cfg, head: Block) -> Option<(Value, ArmMatch)> {
    let &brif_inst = func.blocks[head.index()].insts.last()?;
    let Op::Brif(cond, then_b, then_args, else_b, else_args) = func.insts[brif_inst.index()].op.clone() else {
        return None;
    };
    if then_b == head || else_b == head || func.entry == Some(then_b) || func.entry == Some(else_b) {
        return None;
    }
    let matched = if then_b == else_b {
        ArmMatch { join: then_b, t_args: then_args, e_args: else_args, clear: [None, None] }
    } else if then_args.is_empty() && else_args.is_empty() && let Some(m) = match_diamond(func, cfg, head, (then_b, else_b)) {
        m
    } else if then_args.is_empty() && let Some(t) = match_triangle(func, cfg, then_b, else_b) {
        ArmMatch { join: else_b, t_args: t, e_args: else_args, clear: [Some(then_b), None] }
    } else if else_args.is_empty() && let Some(e) = match_triangle(func, cfg, else_b, then_b) {
        ArmMatch { join: then_b, t_args: then_args, e_args: e, clear: [Some(else_b), None] }
    } else {
        return None;
    };
    Some((cond, matched))
}

fn try_head(func: &mut Function, cfg: &Cfg, head: Block, cleared: &mut [bool]) -> bool {
    let Some((cond, m)) = match_arms(func, cfg, head) else {
        return false;
    };
    let join_params = &func.blocks[m.join.index()].params;
    if !(1..=4).contains(&join_params.len())
        || m.t_args.len() != join_params.len()
        || m.e_args.len() != join_params.len()
        || !join_params.iter().all(|p| scalar_select(func.ty(*p)))
    {
        return false;
    }
    let param_tys: Vec<Type> = join_params.iter().map(|p| func.ty(*p)).collect();
    if let Some(old_brif) = func.blocks[head.index()].insts.pop() {
        func.insts[old_brif.index()].op = Op::Trap;
    }
    for side in m.clear.into_iter().flatten() {
        let mut moved = std::mem::take(&mut func.blocks[side.index()].insts);
        if let Some(old_jump) = moved.pop() {
            func.insts[old_jump.index()].op = Op::Trap;
        }
        func.blocks[head.index()].insts.extend(moved);
        cleared[side.index()] = true;
    }
    let mut selected = Vec::with_capacity(param_tys.len());
    for ((ty, t_val), e_val) in param_tys.into_iter().zip(m.t_args).zip(m.e_args) {
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
    func.insts.push(InstData { op: Op::Jump(m.join, selected), result: None });
    func.blocks[head.index()].insts.push(jump);
    true
}

/// Converts pure diamond and triangle conditional branches into branchless `Op::Select`.
pub(crate) fn run(func: &mut Function) -> bool {
    if func.entry.is_none() || func.coroutine.is_some() {
        return false;
    }
    let mut any = false;
    loop {
        let cfg = Cfg::new(func);
        let mut cleared = vec![false; func.blocks.len()];
        let mut changed = false;
        for &block in cfg.rpo.iter().rev() {
            if !cleared[block.index()] {
                changed |= try_head(func, &cfg, block, &mut cleared);
            }
        }
        if !changed {
            return any;
        }
        any = true;
        crate::merge::run(func);
    }
}

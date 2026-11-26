mod join;
mod ssa;

use self::join::{collapse_convergent_join, split_correlated_join, thread_repaired_joins};
pub(super) use self::ssa::{SsaRepairPlan, apply_ssa_repair, plan_ssa_joins};
use crate::ir::{Block, BlockData, Function, Inst, IntCC, Op, Value};

const MAX_COPY: usize = 8;

#[derive(Clone, Copy)]
enum Edge {
    Jump(Block),
    Brif(Block, bool),
}

impl Edge {
    fn pred(self) -> Block {
        match self {
            Edge::Jump(b) | Edge::Brif(b, _) => b,
        }
    }
}

fn pred_edge_to(func: &Function, pred: Block, target: Block) -> Option<(Edge, &[Value])> {
    match func.last(pred) {
        Some(Op::Jump(t, args)) if *t == target => Some((Edge::Jump(pred), args.as_slice())),
        Some(Op::Brif(_, then, then_args, other, other_args)) if *then != *other => {
            if *then == target {
                Some((Edge::Brif(pred, true), then_args.as_slice()))
            } else if *other == target {
                Some((Edge::Brif(pred, false), other_args.as_slice()))
            } else {
                None
            }
        }
        _ => None,
    }
}

enum Branch {
    Brif {
        test: Test,
        then: Block,
        other: Block,
    },
    Table {
        index: Value,
        default: Block,
        targets: Vec<Block>,
    },
}

enum Test {
    Cmp(IntCC, [Value; 2]),
    Cond(Value),
}

/// Copies a short block that branches on a compare, condition, or jump table into each predecessor edge whose jump arguments decide that branch, and sends the copy straight to the taken target; returns whether it copied any block.
pub(super) fn run(func: &mut Function) -> bool {
    if func.blocks.len() < 2 {
        return false;
    }
    let cfg = func.cfg();
    let mut defs_cache = Some(crate::fold::defs(func));
    let mut consts_cache = Some(func.const_values());
    let (defs, consts) = (
        defs_cache.as_deref().expect("invariant: defs initialized"),
        consts_cache
            .as_deref()
            .expect("invariant: consts initialized"),
    );
    let mut copies: Vec<(Edge, Block, Block)> = Vec::new();
    for at in 0..func.blocks.len() {
        let block = Block(at as u32);
        if !cfg.reachable[at] {
            continue;
        }
        let Some(found) = branch(func, defs, block) else {
            continue;
        };
        let params_len = func.blocks[block.index()].params.len();
        for &pred in cfg.preds_of(block) {
            if pred == block
                || copies
                    .iter()
                    .any(|&(e, b, _)| e.pred() == pred || b == pred || e.pred() == block)
            {
                continue;
            }
            let Some((edge, args)) = pred_edge_to(func, pred, block) else {
                continue;
            };
            if args.len() != params_len {
                continue;
            }
            let Some(to) = decide(func, defs, consts, pred, block, args, &found) else {
                continue;
            };
            if to != block && stays_under(func, &cfg, block, to) {
                copies.push((edge, block, to));
            }
        }
    }
    for &(edge, block, to) in &copies {
        localize_consts(func, block);
        match edge {
            Edge::Jump(pred) => {
                copy_into(func, pred, block, to);
            }
            Edge::Brif(pred, on_then) => {
                if let Some(split) = split_brif_arm(func, pred, on_then, block) {
                    copy_into(func, split, block, to);
                }
            }
        }
    }
    if !copies.is_empty() {
        func.bump_cfg();
        defs_cache = None;
        consts_cache = None;
    }
    let split = split_correlated_join(func, &mut defs_cache, &mut consts_cache);
    let collapsed = collapse_convergent_join(func, &mut defs_cache, &mut consts_cache);
    let repaired = thread_repaired_joins(func, &mut defs_cache, &mut consts_cache);
    !copies.is_empty() || split || collapsed || repaired
}

fn split_brif_arm(func: &mut Function, pred: Block, on_then: bool, target: Block) -> Option<Block> {
    let last = *func.blocks[pred.index()].insts.last()?;
    let split = Block(func.blocks.len() as u32);
    let args = match &mut func.insts[last.index()].op {
        Op::Brif(_, then, then_args, other, other_args) => {
            let (arm_target, arm_args) = if on_then {
                (then, then_args)
            } else {
                (other, other_args)
            };
            if *arm_target != target {
                return None;
            }
            *arm_target = split;
            std::mem::take(arm_args)
        }
        _ => return None,
    };
    let jump = func.add(Op::Jump(target, args), None);
    func.blocks.push(BlockData {
        params: Vec::new(),
        insts: vec![jump],
    });
    Some(split)
}

fn is_body_small_load(func: &Function, defs: &[Option<Inst>], body: &[Inst], val: Value) -> bool {
    defs[val.index()].is_some_and(|i| {
        body.contains(&i)
            && matches!(func.insts[i.index()].op, Op::Load(..) | Op::StackLoad(..))
            && func.ty(val).is_small_int()
    })
}

fn branch(func: &Function, defs: &[Option<Inst>], block: Block) -> Option<Branch> {
    let (&last, body) = func.blocks[block.index()].insts.split_last()?;
    if body.len() > MAX_COPY
        || !body.iter().all(|inst| {
            matches!(
                func.insts[inst.index()].op,
                Op::Store(..) | Op::Load(..) | Op::StackStore(..) | Op::StackLoad(..)
            ) || func.insts[inst.index()].op.is_speculable()
        })
    {
        return None;
    }
    match func.insts[last.index()].op {
        Op::Brif(cond, then, _, other, _) if then != other => {
            let test = if is_body_small_load(func, defs, body, cond)
                || func.blocks[block.index()].params.contains(&cond)
            {
                Test::Cond(cond)
            } else if let Some(def) = defs[cond.index()].filter(|inst| body.contains(inst)) {
                let (cc, a, b) = crate::algebra::branch_cmp(&func.insts[def.index()].op)?;
                if !func.ty(a).is_small_int() {
                    return None;
                }
                Test::Cmp(cc, [a, b])
            } else {
                return None;
            };
            Some(Branch::Brif { test, then, other })
        }
        Op::BrTable(index, default, ref targets)
            if func.blocks[block.index()].params.contains(&index)
                || is_body_small_load(func, defs, body, index) =>
        {
            Some(Branch::Table {
                index,
                default,
                targets: targets.clone(),
            })
        }
        _ => None,
    }
}

fn decide(
    func: &Function,
    defs: &[Option<Inst>],
    consts: &[Option<i64>],
    pred: Block,
    block: Block,
    args: &[Value],
    found: &Branch,
) -> Option<Block> {
    let params = &func.blocks[block.index()].params;
    let body = &func.blocks[block.index()].insts
        [..func.blocks[block.index()].insts.len().saturating_sub(1)];
    let subst = |v: Value| params.iter().position(|&p| p == v).map_or(v, |i| args[i]);
    let eval_val = |v: Value| -> Option<i64> {
        let mapped = subst(v);
        if let Some(imm) = consts.get(mapped.index()).copied().flatten() {
            return Some(imm);
        }
        let def = defs[v.index()]?;
        let pos = body.iter().position(|&i| i == def)?;
        if !matches!(func.insts[def.index()].op, Op::Load(..) | Op::StackLoad(..)) {
            return None;
        }
        if body[..pos]
            .iter()
            .any(|&i| matches!(func.insts[i.index()].op, Op::Store(..) | Op::StackStore(..)))
        {
            return None;
        }
        let mut local_ops = Vec::with_capacity(pos);
        for &i in &body[..pos] {
            let data = &func.insts[i.index()];
            if let Some(res) = data.result
                && data.op.is_speculable()
            {
                let mut op = data.op.clone();
                op.map_operands(&subst);
                local_ops.push((res, op));
            }
        }
        let mut op = func.insts[def.index()].op.clone();
        op.map_operands(&subst);
        super::stored::pred_load_const_raw(
            func,
            defs,
            consts,
            Some((&local_ops, &[])),
            pred,
            &op,
            func.ty(v),
        )
    };
    match found {
        Branch::Brif { test, then, other } => {
            let taken = match *test {
                Test::Cond(cond) => {
                    eval_val(cond).map(|c| crate::fold::extend(c, func.ty(cond), false) != 0)?
                }
                Test::Cmp(cc, sides) => {
                    let ty = func.ty(sides[0]);
                    if let (Some(ca), Some(cb)) = (eval_val(sides[0]), eval_val(sides[1])) {
                        crate::fold::holds(cc, ty, ca, cb)
                    } else if same(func, defs, block, args, sides) {
                        crate::fold::holds(cc, ty, 0, 0)
                    } else {
                        return None;
                    }
                }
            };
            Some(if taken { *then } else { *other })
        }
        Branch::Table {
            index,
            default,
            targets,
        } => {
            let imm = eval_val(*index)?;
            let u = crate::fold::extend(imm, func.ty(*index), false) as u64;
            Some(crate::fold::table_target(u, *default, targets))
        }
    }
}

fn same(
    func: &Function,
    defs: &[Option<Inst>],
    block: Block,
    args: &[Value],
    sides: [Value; 2],
) -> bool {
    let params = &func.blocks[block.index()].params;
    let local = |v: Value| {
        defs[v.index()].is_some_and(|inst| func.blocks[block.index()].insts.contains(&inst))
    };
    let subst = |v: Value| params.iter().position(|&p| p == v).map_or(v, |i| args[i]);
    let constant = |v: Value| match defs[v.index()].map(|inst| &func.insts[inst.index()].op) {
        Some(&Op::Const(imm)) => Some(imm),
        _ => None,
    };
    let leaf = |u: Value, v: Value| u == v || constant(u).is_some_and(|c| constant(v) == Some(c));
    let shape = |v: Value| {
        let mapped = subst(v);
        let op = match defs[mapped.index()].map(|inst| &func.insts[inst.index()].op) {
            Some(&Op::Binary(bin, x, y)) if local(mapped) => Some((bin, subst(x), subst(y))),
            Some(&Op::Binary(bin, x, y)) => Some((bin, x, y)),
            _ => None,
        };
        (mapped, op)
    };
    let ((a, a_op), (b, b_op)) = (shape(sides[0]), shape(sides[1]));
    a == b
        || matches!(
            (a_op, b_op),
            (Some((p, x1, y1)), Some((q, x2, y2))) if p == q && leaf(x1, x2) && leaf(y1, y2)
        )
}

fn stays_under(func: &Function, cfg: &crate::cfg::Cfg, block: Block, to: Block) -> bool {
    let data = &func.blocks[block.index()];
    let mut own = vec![false; func.types.len()];
    data.params.iter().for_each(|&p| own[p.index()] = true);
    data.insts
        .iter()
        .filter(|inst| !matches!(func.insts[inst.index()].op, Op::Const(_)))
        .filter_map(|inst| func.insts[inst.index()].result)
        .for_each(|r| own[r.index()] = true);
    let kept: Vec<Block> = func
        .successors(block)
        .filter(|&s| s != to && s != block && cfg.single_pred(s) == Some(block))
        .collect();
    func.blocks.iter().enumerate().all(|(at, other)| {
        let here = Block(at as u32);
        here == block
            || kept.iter().any(|&s| cfg.dominates(s, here))
            || other.insts.iter().all(|inst| {
                let mut reads = false;
                func.insts[inst.index()]
                    .op
                    .for_operands(|v| reads |= own[v.index()]);
                !reads
            })
    })
}

/// Gives each other block that reads a constant of `block` its own copy of that constant.
pub(super) fn localize_consts(func: &mut Function, block: Block) {
    let consts: Vec<(Value, i64)> = func.blocks[block.index()]
        .insts
        .iter()
        .filter_map(|inst| {
            match (
                &func.insts[inst.index()].op,
                func.insts[inst.index()].result,
            ) {
                (&Op::Const(imm), Some(res)) => Some((res, imm)),
                _ => None,
            }
        })
        .collect();
    for at in 0..func.blocks.len() {
        if at == block.index() {
            continue;
        }
        let mut local: Vec<(Value, Value)> = Vec::new();
        for i in 0..func.blocks[at].insts.len() {
            let inst = func.blocks[at].insts[i];
            let mut reads = Vec::new();
            func.insts[inst.index()].op.for_operands(|v| {
                reads.extend(consts.iter().filter(|&&(c, _)| c == v).map(|&(c, _)| c))
            });
            for c in reads {
                if local.iter().all(|&(from, _)| from != c) {
                    let fresh = func.value(func.ty(c));
                    local.push((c, fresh));
                }
            }
            let mut op = func.insts[inst.index()].op.clone();
            op.map_operands(|v| {
                local
                    .iter()
                    .find(|&&(from, _)| from == v)
                    .map_or(v, |&(_, into)| into)
            });
            func.insts[inst.index()].op = op;
        }
        let defs: Vec<Inst> = local
            .iter()
            .map(|&(from, into)| {
                let imm = consts
                    .iter()
                    .find(|&&(c, _)| c == from)
                    .map_or(0, |&(_, imm)| imm);
                func.add(Op::Const(imm), Some(into))
            })
            .collect();
        func.blocks[at].insts.splice(0..0, defs);
    }
}

fn copy_into(func: &mut Function, pred: Block, block: Block, to: Block) -> Vec<(Value, Value)> {
    let Some(&jump) = func.blocks[pred.index()].insts.last() else {
        return Vec::new();
    };
    let Op::Jump(_, ref args) = func.insts[jump.index()].op else {
        return Vec::new();
    };
    let mut map: Vec<(Value, Value)> = func.blocks[block.index()]
        .params
        .iter()
        .copied()
        .zip(args.iter().copied())
        .collect();
    let (&last, body) = func.blocks[block.index()]
        .insts
        .split_last()
        .expect("invariant: branch() saw a terminator");
    let body = body.to_vec();
    let mut copied = Vec::with_capacity(body.len());
    for inst in body {
        let data = func.insts[inst.index()].clone();
        let mut op = data.op;
        op.map_operands(|v| {
            map.iter()
                .find(|&&(from, _)| from == v)
                .map_or(v, |&(_, into)| into)
        });
        let result = data.result.map(|r| {
            let fresh = func.value(func.ty(r));
            map.push((r, fresh));
            fresh
        });
        let new = func.add(op, result);
        func.insts[new.index()].loc = data.loc;
        copied.push(new);
    }
    let mut to_args = match &func.insts[last.index()].op {
        Op::Brif(_, then, then_args, _, _) if to == *then => then_args.clone(),
        Op::Brif(_, _, _, other, other_args) if to == *other => other_args.clone(),
        Op::BrTable(..) => Vec::new(),
        _ => return map,
    };
    to_args.iter_mut().for_each(|v| {
        *v = map
            .iter()
            .find(|&&(from, _)| from == *v)
            .map_or(*v, |&(_, into)| into)
    });
    func.insts[jump.index()].op = Op::Jump(to, to_args);
    let insts = &mut func.blocks[pred.index()].insts;
    let at = insts.len() - 1;
    insts.splice(at..at, copied);
    map
}

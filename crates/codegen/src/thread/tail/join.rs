use super::ssa::{SsaRepairPlan, apply_ssa_repair, plan_ssa_joins};
use super::{Edge, branch, copy_into, decide, localize_consts, pred_edge_to, split_brif_arm};
use crate::ir::{Block, BlockData, Function, Inst, Op, Value};

pub(super) fn thread_repaired_joins(
    func: &mut Function,
    defs_cache: &mut Option<Vec<Option<Inst>>>,
    consts_cache: &mut Option<Vec<Option<i64>>>,
) -> bool {
    let mut any = false;
    for _ in 0..8 {
        let found_plan = {
            let cfg = func.cfg();
            if !(0..func.blocks.len())
                .any(|at| cfg.reachable[at] && cfg.preds_of(Block(at as u32)).len() >= 2)
            {
                break;
            }
            let defs = defs_cache.get_or_insert_with(|| crate::fold::defs(func));
            let consts = consts_cache.get_or_insert_with(|| func.const_values());
            let mut picked = None;
            'blocks: for at in 0..func.blocks.len() {
                let block = Block(at as u32);
                if !cfg.reachable[at]
                    || cfg.preds_of(block).len() < 2
                    || cfg.preds_of(block).iter().any(|&p| cfg.dominates(block, p))
                {
                    continue;
                }
                let Some(found) = branch(func, defs, block) else {
                    continue;
                };
                let params_len = func.blocks[at].params.len();
                let mut own_vals: Vec<Value> = func.blocks[at].params.clone();
                for &inst in &func.blocks[at].insts {
                    if !matches!(func.insts[inst.index()].op, Op::Const(_))
                        && let Some(r) = func.insts[inst.index()].result
                    {
                        own_vals.push(r);
                    }
                }
                for &pred in cfg.preds_of(block) {
                    if pred == block || cfg.dominates(block, pred) {
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
                    if to == block || !cfg.dominates(block, to) {
                        continue;
                    }
                    let is_brif = matches!(edge, Edge::Brif(..));
                    if is_brif && cfg.preds_of(to).iter().any(|&p| cfg.dominates(to, p)) {
                        continue;
                    }
                    let Some(ssa) = plan_ssa_joins(func, &cfg, block, to, &[], &own_vals) else {
                        continue;
                    };
                    if ssa.need.is_empty() || (is_brif && ssa.joins.len() > 1) {
                        continue;
                    }
                    picked = Some((edge, block, to, ssa));
                    break 'blocks;
                }
            }
            picked
        };
        let Some((edge, block, to, ssa)) = found_plan else {
            break;
        };
        localize_consts(func, block);
        let new_pred = match edge {
            Edge::Jump(pred) => pred,
            Edge::Brif(pred, on_then) => {
                let Some(split) = split_brif_arm(func, pred, on_then, block) else {
                    break;
                };
                split
            }
        };
        let map = copy_into(func, new_pred, block, to);
        let pairs: Vec<(Value, Value)> = ssa
            .need
            .iter()
            .filter_map(|&orig| {
                map.iter()
                    .find(|&&(from, _)| from == orig)
                    .map(|&(_, into)| (orig, into))
            })
            .collect();
        apply_ssa_repair(func, &ssa, to, new_pred, &pairs);
        func.bump_cfg();
        *defs_cache = None;
        *consts_cache = None;
        any = true;
    }
    any
}

fn defs_local_only(func: &Function, blocks: &[Block]) -> bool {
    let mut own = vec![false; func.types.len()];
    for &b in blocks {
        for &inst in &func.blocks[b.index()].insts {
            if let Some(res) = func.insts[inst.index()].result {
                own[res.index()] = true;
            }
        }
    }
    func.blocks.iter().enumerate().all(|(bi, bdata)| {
        let cur = Block(bi as u32);
        blocks.contains(&cur)
            || bdata.insts.iter().all(|&inst| {
                let mut hit = false;
                func.insts[inst.index()]
                    .op
                    .for_operands(|v| hit |= own[v.index()]);
                !hit
            })
    })
}

fn clone_block(
    func: &mut Function,
    src: Block,
    map: &mut Vec<(Value, Value)>,
    term_override: Option<Op>,
) -> Block {
    let fresh_block = Block(func.blocks.len() as u32);
    let insts_src = func.blocks[src.index()].insts.clone();
    let mut new_insts = Vec::with_capacity(insts_src.len());
    for (idx, inst) in insts_src.iter().copied().enumerate() {
        let is_last = idx + 1 == insts_src.len();
        let data = func.insts[inst.index()].clone();
        let mut op = if is_last && let Some(over) = term_override.clone() {
            over
        } else {
            data.op
        };
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
        let added = func.add(op, result);
        func.insts[added.index()].loc = data.loc;
        new_insts.push(added);
    }
    func.blocks.push(BlockData {
        params: Vec::new(),
        insts: new_insts,
    });
    fresh_block
}

struct CorrelatedPlan {
    pred: Block,
    disp: Block,
    to: Block,
    mid: Block,
    mid_is_then: bool,
    exit: Block,
    exit_args: Vec<Value>,
    args: Vec<Value>,
    ssa: SsaRepairPlan,
}

fn find_correlated_join(
    func: &Function,
    defs_cache: &mut Option<Vec<Option<Inst>>>,
    consts_cache: &mut Option<Vec<Option<i64>>>,
) -> Option<CorrelatedPlan> {
    if !func
        .blocks
        .iter()
        .any(|b| !b.params.is_empty() && b.insts.len() == 1)
    {
        return None;
    }
    let cfg = func.cfg();
    let defs = defs_cache.get_or_insert_with(|| crate::fold::defs(func));
    let consts = consts_cache.get_or_insert_with(|| func.const_values());
    for at in 0..func.blocks.len() {
        let disp = Block(at as u32);
        if !cfg.reachable[at]
            || func.blocks[at].params.is_empty()
            || func.blocks[at].insts.len() != 1
        {
            continue;
        }
        let Some(found) = branch(func, defs, disp) else {
            continue;
        };
        let disp_params = &func.blocks[at].params;
        for &pred in cfg.preds_of(disp) {
            if pred == disp {
                continue;
            }
            let Some(Op::Jump(target, args)) = func.last(pred) else {
                continue;
            };
            if *target != disp || args.len() != disp_params.len() {
                continue;
            }
            let Some(to) = decide(func, defs, consts, pred, disp, args, &found) else {
                continue;
            };
            if to == disp
                || !func.blocks[to.index()].params.is_empty()
                || cfg.preds_of(to).len() < 2
                || !cfg.dominates(disp, to)
                || cfg.dominates(to, pred)
                || func.blocks[to.index()].insts.len() > 6
                || func.block_has_call(to)
            {
                continue;
            }
            let Some(Op::Brif(_, succ_a, a_args, succ_b, b_args)) = func.last(to) else {
                continue;
            };
            if *succ_a == *succ_b || *succ_a == to || *succ_b == to {
                continue;
            }
            let candidates = [
                (*succ_a, a_args.is_empty(), true, *succ_b),
                (*succ_b, b_args.is_empty(), false, *succ_a),
            ];
            for (mid, mid_empty, mid_is_then, other_succ) in candidates {
                if !mid_empty
                    || cfg.single_pred(mid) != Some(to)
                    || !func.blocks[mid.index()].params.is_empty()
                    || func.blocks[mid.index()].insts.len() > 6
                    || func.block_has_call(mid)
                    || !defs_local_only(func, &[to, mid])
                {
                    continue;
                }
                if cfg.dominates(disp, other_succ) {
                    continue;
                }
                let Some(Op::Brif(mid_cond, m_then, m_then_args, m_else, m_else_args)) =
                    func.last(mid)
                else {
                    continue;
                };
                if *m_then == *m_else {
                    continue;
                }
                let Some(pidx) = disp_params.iter().position(|&p| p == *mid_cond) else {
                    continue;
                };
                let Some(&Op::Const(imm)) =
                    defs[args[pidx].index()].map(|i| &func.insts[i.index()].op)
                else {
                    continue;
                };
                let taken = crate::fold::extend(imm, func.ty(*mid_cond), false) != 0;
                let (exit, exit_args) = if taken {
                    (*m_then, m_then_args.clone())
                } else {
                    (*m_else, m_else_args.clone())
                };
                if exit == disp || exit == to || exit == mid || !cfg.dominates(disp, exit) {
                    continue;
                }
                let Some(ssa) = plan_ssa_joins(func, &cfg, disp, exit, &[to, mid], disp_params)
                else {
                    continue;
                };
                if ssa.need.len() > 2 {
                    continue;
                }
                return Some(CorrelatedPlan {
                    pred,
                    disp,
                    to,
                    mid,
                    mid_is_then,
                    exit,
                    exit_args,
                    args: args.clone(),
                    ssa,
                });
            }
        }
    }
    None
}

pub(super) fn split_correlated_join(
    func: &mut Function,
    defs_cache: &mut Option<Vec<Option<Inst>>>,
    consts_cache: &mut Option<Vec<Option<i64>>>,
) -> bool {
    let mut any = false;
    for _ in 0..4 {
        let Some(plan) = find_correlated_join(func, defs_cache, consts_cache) else {
            break;
        };
        let disp_params = func.blocks[plan.disp.index()].params.clone();
        let mut map: Vec<(Value, Value)> = disp_params
            .iter()
            .copied()
            .zip(plan.args.iter().copied())
            .collect();
        let to_clone = clone_block(func, plan.to, &mut map, None);
        let mid_term = Op::Jump(plan.exit, plan.exit_args.clone());
        let mid_clone = clone_block(func, plan.mid, &mut map, Some(mid_term));
        let to_last = *func.blocks[to_clone.index()]
            .insts
            .last()
            .expect("invariant: to_clone has Brif");
        if let Op::Brif(_, ref mut then, _, ref mut other, _) = func.insts[to_last.index()].op {
            if plan.mid_is_then {
                *then = mid_clone;
            } else {
                *other = mid_clone;
            }
        }
        let pred_last = *func.blocks[plan.pred.index()]
            .insts
            .last()
            .expect("invariant: pred has Jump");
        func.insts[pred_last.index()].op = Op::Jump(to_clone, Vec::new());
        let pairs: Vec<(Value, Value)> = plan
            .ssa
            .need
            .iter()
            .filter_map(|&orig| {
                disp_params
                    .iter()
                    .position(|&p| p == orig)
                    .map(|pi| (orig, plan.args[pi]))
            })
            .collect();
        apply_ssa_repair(func, &plan.ssa, plan.exit, mid_clone, &pairs);
        func.bump_cfg();
        *defs_cache = None;
        *consts_cache = None;
        any = true;
    }
    any
}

fn find_convergent_join(
    func: &Function,
    consts_cache: &mut Option<Vec<Option<i64>>>,
) -> Option<(Block, Block, Block, Block, Vec<Value>, SsaRepairPlan)> {
    if !func
        .blocks
        .iter()
        .any(|b| !b.params.is_empty() && b.insts.len() == 1)
    {
        return None;
    }
    let cfg = func.cfg();
    let consts = consts_cache.get_or_insert_with(|| func.const_values());
    for at in 0..func.blocks.len() {
        let disp = Block(at as u32);
        let disp_params = &func.blocks[at].params;
        if !cfg.reachable[at] || disp_params.is_empty() || func.blocks[at].insts.len() != 1 {
            continue;
        }
        let Some(&Op::Brif(_, then, ref ta, other, ref oa)) = func.last(disp) else {
            continue;
        };
        if !ta.is_empty() || !oa.is_empty() || then == other {
            continue;
        }
        for (arm, exit) in [(then, other), (other, then)] {
            if arm == disp || exit == disp || arm == exit || !cfg.dominates(disp, exit) {
                continue;
            }
            if !func.blocks[arm.index()].params.is_empty()
                || func.blocks[arm.index()].insts.len() != 1
            {
                continue;
            }
            let Some(&Op::Brif(arm_cond, a_then, ref ata, a_other, ref aoa)) = func.last(arm)
            else {
                continue;
            };
            if !ata.is_empty() || !aoa.is_empty() || a_then == a_other {
                continue;
            }
            let Some(pidx) = disp_params.iter().position(|&p| p == arm_cond) else {
                continue;
            };
            for &pred in cfg.preds_of(disp) {
                if pred == disp || cfg.dominates(disp, pred) {
                    continue;
                }
                let Some(Op::Jump(target, args)) = func.last(pred) else {
                    continue;
                };
                if *target != disp || args.len() != disp_params.len() {
                    continue;
                }
                let Some(imm) = consts.get(args[pidx].index()).copied().flatten() else {
                    continue;
                };
                let taken = crate::fold::extend(imm, func.ty(arm_cond), false) != 0;
                if (if taken { a_then } else { a_other }) != exit {
                    continue;
                }
                if let Some(ssa) = plan_ssa_joins(func, &cfg, disp, exit, &[arm], disp_params) {
                    return Some((pred, disp, arm, exit, args.clone(), ssa));
                }
            }
        }
    }
    None
}

pub(super) fn collapse_convergent_join(
    func: &mut Function,
    defs_cache: &mut Option<Vec<Option<Inst>>>,
    consts_cache: &mut Option<Vec<Option<i64>>>,
) -> bool {
    let mut any = false;
    for _ in 0..4 {
        let Some((pred, disp, _, exit, args, ssa)) = find_convergent_join(func, consts_cache)
        else {
            break;
        };
        let disp_params = func.blocks[disp.index()].params.clone();
        let pred_last = *func.blocks[pred.index()]
            .insts
            .last()
            .expect("invariant: pred has Jump");
        func.insts[pred_last.index()].op = Op::Jump(exit, Vec::new());
        let pairs: Vec<(Value, Value)> = ssa
            .need
            .iter()
            .filter_map(|&o| {
                disp_params
                    .iter()
                    .position(|&p| p == o)
                    .map(|i| (o, args[i]))
            })
            .collect();
        apply_ssa_repair(func, &ssa, exit, pred, &pairs);
        func.bump_cfg();
        *defs_cache = None;
        *consts_cache = None;
        any = true;
    }
    any
}

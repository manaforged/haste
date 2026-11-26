use crate::cfg::Cfg;
use crate::ir::{Block, Function, Op, Value};

pub(in crate::thread) struct SsaRepairPlan {
    pub(in crate::thread) need: Vec<Value>,
    pub(in crate::thread) from_exit: Vec<bool>,
    pub(in crate::thread) joins: Vec<Block>,
    join_of: Vec<Option<usize>>,
}

fn empty_plan() -> SsaRepairPlan {
    SsaRepairPlan {
        need: Vec::new(),
        from_exit: Vec::new(),
        joins: Vec::new(),
        join_of: Vec::new(),
    }
}

fn reachable_from_exit(
    func: &Function,
    cfg: &Cfg,
    disp: Block,
    exit: Block,
    skip: &[Block],
) -> (Vec<bool>, Vec<Block>) {
    let mut from_exit = vec![false; func.blocks.len()];
    let mut visited = Vec::new();
    let mut stack = vec![exit];
    while let Some(b) = stack.pop() {
        if !cfg.dominates(disp, b)
            || b == disp
            || skip.contains(&b)
            || std::mem::replace(&mut from_exit[b.index()], true)
        {
            continue;
        }
        visited.push(b);
        for s in func.successors(b) {
            stack.push(s);
        }
    }
    (from_exit, visited)
}

fn used_in_region(
    func: &Function,
    visited: &[Block],
    vals: &[Value],
) -> Option<(Vec<Value>, Vec<Block>)> {
    let vals_mask = vals
        .iter()
        .fold(0_u128, |m, &v| m | (1_u128 << (v.0 & 127)));
    let val_map = (vals.len() > 8).then(|| {
        let mut map = vec![u32::MAX; func.types.len()];
        for (idx, &v) in vals.iter().enumerate() {
            map[v.index()] = idx as u32;
        }
        map
    });
    let mut seen_val = vec![false; vals.len()];
    let mut need_count = 0usize;
    let mut back = Vec::new();
    for &b in visited {
        let mut used_here = false;
        for &inst in &func.blocks[b.index()].insts {
            func.insts[inst.index()].op.for_operands(|op| {
                if vals_mask & (1_u128 << (op.0 & 127)) == 0 {
                    return;
                }
                let pos = match &val_map {
                    Some(map) => map
                        .get(op.index())
                        .copied()
                        .filter(|&p| p != u32::MAX)
                        .map(|p| p as usize),
                    None => vals.iter().position(|&v| v == op),
                };
                if let Some(pos) = pos {
                    used_here = true;
                    if !std::mem::replace(&mut seen_val[pos], true) {
                        need_count += 1;
                    }
                }
            });
            if need_count > 6 {
                return None;
            }
        }
        if used_here {
            back.push(b);
        }
    }
    let need = vals
        .iter()
        .zip(&seen_val)
        .filter_map(|(&v, &s)| s.then_some(v))
        .collect();
    Some((need, back))
}

fn backward_live(
    func: &Function,
    cfg: &Cfg,
    from_exit: &[bool],
    mut back: Vec<Block>,
) -> (Vec<bool>, Vec<Block>) {
    let mut live = vec![false; func.blocks.len()];
    let mut live_list = Vec::new();
    while let Some(b) = back.pop() {
        if !from_exit[b.index()] || std::mem::replace(&mut live[b.index()], true) {
            continue;
        }
        live_list.push(b);
        for &pr in cfg.preds_of(b) {
            if from_exit[pr.index()] && !live[pr.index()] {
                back.push(pr);
            }
        }
    }
    live_list.sort_unstable_by_key(|b| b.0);
    (live, live_list)
}

fn compute_joins(
    func: &Function,
    cfg: &Cfg,
    exit: Block,
    from_exit: &[bool],
    live_list: &mut [Block],
) -> Option<(Vec<Block>, Vec<Option<usize>>)> {
    let mut joins = Vec::new();
    for &b in live_list.iter() {
        if b == exit || cfg.preds_of(b).iter().any(|&pr| !from_exit[pr.index()]) {
            joins.push(b);
        }
    }
    let mut join_of = vec![None; func.blocks.len()];
    let mut ranked = false;
    loop {
        let mut added = false;
        if !ranked && (joins.len() > 1 || live_list.iter().any(|&b| !cfg.dominates(joins[0], b))) {
            let rank = Cfg::ranks(&cfg.rpo, func.blocks.len());
            live_list.sort_unstable_by_key(|b| rank[b.index()]);
            ranked = true;
        }
        for &b in live_list.iter() {
            let bi = b.index();
            let deepest = (0..joins.len())
                .filter(|&ji| cfg.dominates(joins[ji], b))
                .find(|&ji| {
                    (0..joins.len()).all(|other| {
                        !cfg.dominates(joins[other], b) || cfg.dominates(joins[other], joins[ji])
                    })
                });
            match deepest {
                Some(ji) => join_of[bi] = Some(ji),
                None => {
                    if joins.len() >= 8 {
                        return None;
                    }
                    joins.push(b);
                    added = true;
                    break;
                }
            }
        }
        if !added {
            break;
        }
    }
    for &j in &joins {
        for &pr in cfg.preds_of(j) {
            if !matches!(func.last(pr), Some(Op::Jump(..) | Op::Brif(..))) {
                return None;
            }
        }
    }
    Some((joins, join_of))
}

pub(in crate::thread) fn plan_ssa_joins(
    func: &Function,
    cfg: &Cfg,
    disp: Block,
    exit: Block,
    skip: &[Block],
    vals: &[Value],
) -> Option<SsaRepairPlan> {
    if vals.is_empty() {
        return Some(empty_plan());
    }
    let (from_exit, visited) = reachable_from_exit(func, cfg, disp, exit, skip);
    if visited.is_empty() {
        return Some(empty_plan());
    }
    let (need, back) = used_in_region(func, &visited, vals)?;
    if need.is_empty() {
        return Some(SsaRepairPlan {
            need,
            from_exit,
            joins: Vec::new(),
            join_of: Vec::new(),
        });
    }
    let (from_exit, mut live_list) = backward_live(func, cfg, &from_exit, back);
    let (joins, join_of) = compute_joins(func, cfg, exit, &from_exit, &mut live_list)?;
    Some(SsaRepairPlan {
        need,
        from_exit,
        joins,
        join_of,
    })
}

pub(in crate::thread) fn apply_ssa_repair(
    func: &mut Function,
    plan: &SsaRepairPlan,
    exit: Block,
    new_pred: Block,
    pairs: &[(Value, Value)],
) {
    if pairs.is_empty() {
        return;
    }
    let cfg = func.cfg();
    let orig_preds: Vec<Vec<Block>> = plan
        .joins
        .iter()
        .map(|&j| {
            cfg.preds_of(j)
                .iter()
                .copied()
                .filter(|&pr| pr != new_pred)
                .collect()
        })
        .collect();
    for &(orig_v, new_v) in pairs {
        let mut j_params = Vec::with_capacity(plan.joins.len());
        for &j in &plan.joins {
            let fresh = func.value(func.ty(orig_v));
            func.blocks[j.index()].params.push(fresh);
            j_params.push(fresh);
        }
        for (bi, &in_fe) in plan.from_exit.iter().enumerate() {
            if !in_fe {
                continue;
            }
            let Some(ji) = plan.join_of[bi] else { continue };
            let rep = j_params[ji];
            let insts = func.blocks[bi].insts.clone();
            for inst in insts {
                func.insts[inst.index()]
                    .op
                    .map_operands(|v| if v == orig_v { rep } else { v });
            }
        }
        for (ji, &j) in plan.joins.iter().enumerate() {
            if j == exit {
                func.push_edge_arg(new_pred, j, new_v);
            }
            let mut seen_preds = Vec::new();
            for &pr in &orig_preds[ji] {
                if seen_preds.contains(&pr) {
                    continue;
                }
                seen_preds.push(pr);
                let val = if plan.from_exit[pr.index()] {
                    plan.join_of[pr.index()].map_or(orig_v, |pji| j_params[pji])
                } else {
                    orig_v
                };
                func.push_edge_arg(pr, j, val);
            }
        }
    }
}

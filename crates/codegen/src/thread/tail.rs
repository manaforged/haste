use crate::ir::{Block, BlockData, Function, Inst, IntCC, Op, Value};

/// The most instructions a copied block may run before its branch.
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

/// The branch a block ends with: an integer compare in the block, a condition parameter, or a jump-table parameter.
enum Branch {
    Brif { test: Test, then: Block, other: Block },
    Table { index: Value, default: Block, targets: Vec<Block> },
}

enum Test {
    Cmp(IntCC, [Value; 2]),
    Cond(Value),
}

/// Copies a short block that branches on a compare, condition, or jump table into each predecessor edge whose jump arguments decide that branch, and sends the copy straight to the taken target; returns whether it copied any block.
pub(super) fn run(func: &mut Function) -> bool {
    let cfg = func.cfg();
    let defs = crate::fold::defs(func);
    let mut copies: Vec<(Edge, Block, Block)> = Vec::new();
    for at in 0..func.blocks.len() {
        let block = Block(at as u32);
        if !cfg.reachable[at] {
            continue;
        }
        let Some(found) = branch(func, &defs, block) else { continue };
        let params_len = func.blocks[block.index()].params.len();
        for &pred in cfg.preds_of(block) {
            if pred == block
                || copies.iter().any(|&(e, b, _)| e.pred() == pred || b == pred || e.pred() == block)
            {
                continue;
            }
            let edge_args = match func.last(pred) {
                Some(Op::Jump(target, args)) if *target == block => Some((Edge::Jump(pred), args.as_slice())),
                Some(Op::Brif(_, then, then_args, other, other_args)) if *then != *other => {
                    if *then == block {
                        Some((Edge::Brif(pred, true), then_args.as_slice()))
                    } else if *other == block {
                        Some((Edge::Brif(pred, false), other_args.as_slice()))
                    } else {
                        None
                    }
                }
                _ => None,
            };
            let Some((edge, args)) = edge_args else { continue };
            if args.len() != params_len {
                continue;
            }
            let Some(to) = decide(func, &defs, block, args, &found) else {
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
                let Some(split) = split_brif_arm(func, pred, on_then, block) else { continue };
                copy_into(func, split, block, to);
            }
        }
    }
    if !copies.is_empty() {
        func.bump_cfg();
    }
    let split = split_correlated_join(func);
    let repaired = thread_repaired_joins(func);
    !copies.is_empty() || split || repaired
}

struct SsaRepairPlan {
    need: Vec<Value>,
    from_exit: Vec<bool>,
    joins: Vec<Block>,
    join_of: Vec<Option<usize>>,
}

fn plan_ssa_joins(func: &Function, cfg: &crate::cfg::Cfg, disp: Block, exit: Block, skip: &[Block], vals: &[Value]) -> Option<SsaRepairPlan> {
    let mut from_exit = vec![false; func.blocks.len()];
    let mut stack = vec![exit];
    while let Some(b) = stack.pop() {
        if !cfg.dominates(disp, b) || b == disp || skip.contains(&b) || std::mem::replace(&mut from_exit[b.index()], true) {
            continue;
        }
        for s in func.successors(b) {
            stack.push(s);
        }
    }
    let mut need = Vec::new();
    for &v in vals {
        let used_in_exit = from_exit.iter().enumerate().any(|(bi, &in_fe)| {
            in_fe
                && func.blocks[bi].insts.iter().any(|&inst| {
                    let mut hit = false;
                    func.insts[inst.index()].op.for_operands(|op| hit |= op == v);
                    hit
                })
        });
        if used_in_exit {
            need.push(v);
        }
    }
    if need.len() > 4 {
        return None;
    }
    let mut joins = Vec::new();
    if !need.is_empty() {
        for (bi, &in_fe) in from_exit.iter().enumerate() {
            if !in_fe {
                continue;
            }
            let b = Block(bi as u32);
            if b == exit || cfg.preds_of(b).iter().any(|&pr| !from_exit[pr.index()]) {
                joins.push(b);
            }
        }
    }
    let mut join_of = vec![None; func.blocks.len()];
    if !need.is_empty() {
        loop {
            let mut added = false;
            for &b in &cfg.rpo {
                let bi = b.index();
                if !from_exit[bi] {
                    continue;
                }
                let doms: Vec<usize> = joins.iter().enumerate().filter(|&(_, &j)| cfg.dominates(j, b)).map(|(ji, _)| ji).collect();
                let deepest = doms.iter().copied().find(|&ji| doms.iter().all(|&other| cfg.dominates(joins[other], joins[ji])));
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
    }
    Some(SsaRepairPlan { need, from_exit, joins, join_of })
}

fn apply_ssa_repair(func: &mut Function, plan: &SsaRepairPlan, exit: Block, new_pred: Block, pairs: &[(Value, Value)]) {
    if pairs.is_empty() {
        return;
    }
    let cfg = func.cfg();
    let orig_preds: Vec<Vec<Block>> = plan
        .joins
        .iter()
        .map(|&j| cfg.preds_of(j).iter().copied().filter(|&pr| pr != new_pred).collect())
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
                func.insts[inst.index()].op.map_operands(|v| if v == orig_v { rep } else { v });
            }
        }
        for (ji, &j) in plan.joins.iter().enumerate() {
            if j == exit {
                push_edge_arg(func, new_pred, j, new_v);
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
                push_edge_arg(func, pr, j, val);
            }
        }
    }
}

fn thread_repaired_joins(func: &mut Function) -> bool {
    let mut any = false;
    for _ in 0..8 {
        let found_plan = {
            let cfg = func.cfg();
            let defs = crate::fold::defs(func);
            let mut picked = None;
            'blocks: for at in 0..func.blocks.len() {
                let block = Block(at as u32);
                if !cfg.reachable[at] || cfg.preds_of(block).len() < 2 {
                    continue;
                }
                let Some(found) = branch(func, &defs, block) else { continue };
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
                    let Some(Op::Jump(target, args)) = func.last(pred) else { continue };
                    if *target != block || args.len() != params_len {
                        continue;
                    }
                    let Some(to) = decide(func, &defs, block, args, &found) else { continue };
                    if to == block || !cfg.dominates(block, to) {
                        continue;
                    }
                    let Some(ssa) = plan_ssa_joins(func, &cfg, block, to, &[], &own_vals) else { continue };
                    if ssa.need.is_empty() {
                        continue;
                    }
                    picked = Some((pred, block, to, ssa));
                    break 'blocks;
                }
            }
            picked
        };
        let Some((pred, block, to, ssa)) = found_plan else { break };
        localize_consts(func, block);
        let map = copy_into(func, pred, block, to);
        let pairs: Vec<(Value, Value)> = ssa
            .need
            .iter()
            .filter_map(|&orig| map.iter().find(|&&(from, _)| from == orig).map(|&(_, into)| (orig, into)))
            .collect();
        apply_ssa_repair(func, &ssa, to, pred, &pairs);
        func.bump_cfg();
        any = true;
    }
    any
}

fn has_call(func: &Function, block: Block) -> bool {
    func.blocks[block.index()].insts.iter().any(|&inst| matches!(func.insts[inst.index()].op, Op::Call { .. } | Op::TryCall { .. }))
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
                func.insts[inst.index()].op.for_operands(|v| hit |= own[v.index()]);
                !hit
            })
    })
}

fn clone_block(func: &mut Function, src: Block, map: &mut Vec<(Value, Value)>, term_override: Option<Op>) -> Block {
    let fresh_block = Block(func.blocks.len() as u32);
    let insts_src = func.blocks[src.index()].insts.clone();
    let mut new_insts = Vec::with_capacity(insts_src.len());
    for (idx, inst) in insts_src.iter().copied().enumerate() {
        let is_last = idx + 1 == insts_src.len();
        let data = func.insts[inst.index()].clone();
        let mut op = if is_last && let Some(over) = term_override.clone() { over } else { data.op };
        op.map_operands(|v| map.iter().find(|&&(from, _)| from == v).map_or(v, |&(_, into)| into));
        let result = data.result.map(|r| {
            let fresh = func.value(func.ty(r));
            map.push((r, fresh));
            fresh
        });
        let added = func.add(op, result);
        func.insts[added.index()].loc = data.loc;
        new_insts.push(added);
    }
    func.blocks.push(BlockData { params: Vec::new(), insts: new_insts });
    fresh_block
}

fn push_edge_arg(func: &mut Function, src: Block, dst: Block, arg: Value) {
    let Some(&last) = func.blocks[src.index()].insts.last() else { return };
    match &mut func.insts[last.index()].op {
        Op::Jump(target, args) if *target == dst => args.push(arg),
        Op::Brif(_, then, then_args, other, other_args) => {
            if *then == dst {
                then_args.push(arg);
            }
            if *other == dst {
                other_args.push(arg);
            }
        }
        _ => {}
    }
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

fn find_correlated_join(func: &Function) -> Option<CorrelatedPlan> {
    let cfg = func.cfg();
    let defs = crate::fold::defs(func);
    for at in 0..func.blocks.len() {
        let disp = Block(at as u32);
        if !cfg.reachable[at] || func.blocks[at].params.is_empty() || func.blocks[at].insts.len() != 1 {
            continue;
        }
        let Some(found) = branch(func, &defs, disp) else { continue };
        let disp_params = &func.blocks[at].params;
        for &pred in cfg.preds_of(disp) {
            if pred == disp {
                continue;
            }
            let Some(Op::Jump(target, args)) = func.last(pred) else { continue };
            if *target != disp || args.len() != disp_params.len() {
                continue;
            }
            let Some(to) = decide(func, &defs, disp, args, &found) else { continue };
            if to == disp
                || !func.blocks[to.index()].params.is_empty()
                || cfg.preds_of(to).len() < 2
                || !cfg.dominates(disp, to)
                || cfg.dominates(to, pred)
                || func.blocks[to.index()].insts.len() > 6
                || has_call(func, to)
            {
                continue;
            }
            let Some(Op::Brif(_, succ_a, a_args, succ_b, b_args)) = func.last(to) else { continue };
            if *succ_a == *succ_b || *succ_a == to || *succ_b == to {
                continue;
            }
            let candidates = [(*succ_a, a_args.is_empty(), true, *succ_b), (*succ_b, b_args.is_empty(), false, *succ_a)];
            for (mid, mid_empty, mid_is_then, other_succ) in candidates {
                if !mid_empty
                    || cfg.single_pred(mid) != Some(to)
                    || !func.blocks[mid.index()].params.is_empty()
                    || func.blocks[mid.index()].insts.len() > 6
                    || has_call(func, mid)
                    || !defs_local_only(func, &[to, mid])
                {
                    continue;
                }
                if cfg.dominates(disp, other_succ) {
                    continue;
                }
                let Some(Op::Brif(mid_cond, m_then, m_then_args, m_else, m_else_args)) = func.last(mid) else { continue };
                if *m_then == *m_else {
                    continue;
                }
                let Some(pidx) = disp_params.iter().position(|&p| p == *mid_cond) else { continue };
                let Some(&Op::Const(imm)) = defs[args[pidx].index()].map(|i| &func.insts[i.index()].op) else { continue };
                let taken = crate::fold::extend(imm, func.ty(*mid_cond), false) != 0;
                let (exit, exit_args) = if taken { (*m_then, m_then_args.clone()) } else { (*m_else, m_else_args.clone()) };
                if exit == disp || exit == to || exit == mid || !cfg.dominates(disp, exit) {
                    continue;
                }
                let Some(ssa) = plan_ssa_joins(func, &cfg, disp, exit, &[to, mid], disp_params) else {
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

/// Duplicates a short non-call join block and its correlated single-pred continuation onto a predecessor edge that decides both branches.
fn split_correlated_join(func: &mut Function) -> bool {
    let mut any = false;
    for _ in 0..4 {
        let Some(plan) = find_correlated_join(func) else { break };
        let disp_params = func.blocks[plan.disp.index()].params.clone();
        let mut map: Vec<(Value, Value)> = disp_params.iter().copied().zip(plan.args.iter().copied()).collect();
        let to_clone = clone_block(func, plan.to, &mut map, None);
        let mid_term = Op::Jump(plan.exit, plan.exit_args.clone());
        let mid_clone = clone_block(func, plan.mid, &mut map, Some(mid_term));
        let to_last = *func.blocks[to_clone.index()].insts.last().expect("invariant: to_clone has Brif");
        if let Op::Brif(_, ref mut then, _, ref mut other, _) = func.insts[to_last.index()].op {
            if plan.mid_is_then {
                *then = mid_clone;
            } else {
                *other = mid_clone;
            }
        }
        let pred_last = *func.blocks[plan.pred.index()].insts.last().expect("invariant: pred has Jump");
        func.insts[pred_last.index()].op = Op::Jump(to_clone, Vec::new());
        let pairs: Vec<(Value, Value)> = plan
            .ssa
            .need
            .iter()
            .filter_map(|&orig| disp_params.iter().position(|&p| p == orig).map(|pi| (orig, plan.args[pi])))
            .collect();
        apply_ssa_repair(func, &plan.ssa, plan.exit, mid_clone, &pairs);
        func.bump_cfg();
        any = true;
    }
    any
}

/// Splits one arm of `pred`'s `Brif` into a fresh block jumping to `target` with that arm's arguments.
fn split_brif_arm(func: &mut Function, pred: Block, on_then: bool, target: Block) -> Option<Block> {
    let last = *func.blocks[pred.index()].insts.last()?;
    let split = Block(func.blocks.len() as u32);
    let args = match &mut func.insts[last.index()].op {
        Op::Brif(_, then, then_args, other, other_args) => {
            let (arm_target, arm_args) = if on_then { (then, then_args) } else { (other, other_args) };
            if *arm_target != target {
                return None;
            }
            *arm_target = split;
            std::mem::take(arm_args)
        }
        _ => return None,
    };
    let jump = func.add(Op::Jump(target, args), None);
    func.blocks.push(BlockData { params: Vec::new(), insts: vec![jump] });
    Some(split)
}

/// The branch ending `block` when its body is a short run of pure instructions, loads and stores and its terminator is a decidable `Brif` or `BrTable`.
fn branch(func: &Function, defs: &[Option<Inst>], block: Block) -> Option<Branch> {
    let (&last, body) = func.blocks[block.index()].insts.split_last()?;
    if body.len() > MAX_COPY
        || !body.iter().all(|inst| {
            matches!(func.insts[inst.index()].op, Op::Store(..) | Op::Load(..) | Op::StackStore(..) | Op::StackLoad(..))
                || func.insts[inst.index()].op.is_speculable()
        })
    {
        return None;
    }
    match func.insts[last.index()].op {
        Op::Brif(cond, then, _, other, _) if then != other => {
            let test = if let Some(def) = defs[cond.index()].filter(|inst| body.contains(inst)) {
                let (cc, a, b) = crate::algebra::branch_cmp(&func.insts[def.index()].op)?;
                if !func.ty(a).is_small_int() {
                    return None;
                }
                Test::Cmp(cc, [a, b])
            } else if func.blocks[block.index()].params.contains(&cond) {
                Test::Cond(cond)
            } else {
                return None;
            };
            Some(Branch::Brif { test, then, other })
        }
        Op::BrTable(index, default, ref targets) if func.blocks[block.index()].params.contains(&index) => {
            Some(Branch::Table { index, default, targets: targets.clone() })
        }
        _ => None,
    }
}

/// Evaluates `found` in `block` under incoming `args` when its branch target is statically known.
fn decide(func: &Function, defs: &[Option<Inst>], block: Block, args: &[Value], found: &Branch) -> Option<Block> {
    let params = &func.blocks[block.index()].params;
    let subst = |v: Value| params.iter().position(|&p| p == v).map_or(v, |i| args[i]);
    let constant = |v: Value| match defs[v.index()].map(|inst| &func.insts[inst.index()].op) {
        Some(&Op::Const(imm)) => Some(imm),
        _ => None,
    };
    match found {
        Branch::Brif { test, then, other } => {
            let taken = match *test {
                Test::Cond(cond) => constant(subst(cond)).map(|c| crate::fold::extend(c, func.ty(cond), false) != 0)?,
                Test::Cmp(cc, sides) => {
                    let ty = func.ty(sides[0]);
                    if let (Some(ca), Some(cb)) = (constant(subst(sides[0])), constant(subst(sides[1]))) {
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
        Branch::Table { index, default, targets } => {
            let imm = constant(subst(*index))?;
            let u = crate::fold::extend(imm, func.ty(*index), false) as u64;
            Some(crate::fold::table_target(u, *default, targets))
        }
    }
}

/// Whether both `sides` of the compare in `block` compute one expression once its parameters take `args`.
fn same(func: &Function, defs: &[Option<Inst>], block: Block, args: &[Value], sides: [Value; 2]) -> bool {
    let params = &func.blocks[block.index()].params;
    let local = |v: Value| defs[v.index()].is_some_and(|inst| func.blocks[block.index()].insts.contains(&inst));
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
    a == b || matches!((a_op, b_op), (Some((p, x1, y1)), Some((q, x2, y2))) if p == q && leaf(x1, x2) && leaf(y1, y2))
}

/// Whether every read of a parameter or computed value of `block` outside `block` sits under a single-pred successor of `block` other than `to`.
fn stays_under(func: &Function, cfg: &crate::cfg::Cfg, block: Block, to: Block) -> bool {
    let data = &func.blocks[block.index()];
    let mut own = vec![false; func.types.len()];
    data.params.iter().for_each(|&p| own[p.index()] = true);
    data.insts.iter().filter(|inst| !matches!(func.insts[inst.index()].op, Op::Const(_))).filter_map(|inst| func.insts[inst.index()].result).for_each(|r| own[r.index()] = true);
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
                func.insts[inst.index()].op.for_operands(|v| reads |= own[v.index()]);
                !reads
            })
    })
}

/// Gives each other block that reads a constant of `block` its own copy of that constant.
fn localize_consts(func: &mut Function, block: Block) {
    let consts: Vec<(Value, i64)> = func.blocks[block.index()]
        .insts
        .iter()
        .filter_map(|inst| match (&func.insts[inst.index()].op, func.insts[inst.index()].result) {
            (&Op::Const(imm), Some(res)) => Some((res, imm)),
            _ => None,
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
            func.insts[inst.index()].op.for_operands(|v| reads.extend(consts.iter().filter(|&&(c, _)| c == v).map(|&(c, _)| c)));
            for c in reads {
                if local.iter().all(|&(from, _)| from != c) {
                    let fresh = func.value(func.ty(c));
                    local.push((c, fresh));
                }
            }
            let mut op = func.insts[inst.index()].op.clone();
            op.map_operands(|v| local.iter().find(|&&(from, _)| from == v).map_or(v, |&(_, into)| into));
            func.insts[inst.index()].op = op;
        }
        let defs: Vec<Inst> = local.iter().map(|&(from, into)| {
            let imm = consts.iter().find(|&&(c, _)| c == from).map_or(0, |&(_, imm)| imm);
            func.add(Op::Const(imm), Some(into))
        }).collect();
        func.blocks[at].insts.splice(0..0, defs);
    }
}

/// Puts a copy of the body of `block` before the jump that ends `pred` and points that jump at `to` with the arguments `block` passes it; returns the value mapping from `block` to the copy.
fn copy_into(func: &mut Function, pred: Block, block: Block, to: Block) -> Vec<(Value, Value)> {
    let Some(&jump) = func.blocks[pred.index()].insts.last() else { return Vec::new() };
    let Op::Jump(_, ref args) = func.insts[jump.index()].op else { return Vec::new() };
    let mut map: Vec<(Value, Value)> = func.blocks[block.index()].params.iter().copied().zip(args.iter().copied()).collect();
    let (&last, body) = func.blocks[block.index()].insts.split_last().expect("invariant: branch() saw a terminator");
    let body = body.to_vec();
    let mut copied = Vec::with_capacity(body.len());
    for inst in body {
        let data = func.insts[inst.index()].clone();
        let mut op = data.op;
        op.map_operands(|v| map.iter().find(|&&(from, _)| from == v).map_or(v, |&(_, into)| into));
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
    to_args.iter_mut().for_each(|v| *v = map.iter().find(|&&(from, _)| from == *v).map_or(*v, |&(_, into)| into));
    func.insts[jump.index()].op = Op::Jump(to, to_args);
    let insts = &mut func.blocks[pred.index()].insts;
    let at = insts.len() - 1;
    insts.splice(at..at, copied);
    map
}

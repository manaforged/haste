use crate::fold::extend;
use crate::ir::{Binary, Block, BlockData, Function, Inst, IntCC, Op, Type, Value};

use super::carry;
use super::eval::{
    EdgeSeed, EvalState, ThreadCx, bind, cmp_implies, eval_cond_truth, fact_of, is_bool_val,
    lookup, lookup_const, step_inst,
};
use super::stored;
use super::{
    Eval, MAX_DISP_INSTS, emit_needed, emit_select, needed_ops, note_use, redefines, select_arms,
    select_split, single_pred_brif_seed,
};

const MAX_CHAIN: usize = 8;
const MAX_BRIF_SPEC_OPS: usize = 4;

fn step_block(
    func: &Function,
    cx: &ThreadCx<'_>,
    pred: Option<Block>,
    cur: Block,
    cur_args: &[Value],
    state: &mut EvalState,
) -> Option<(Block, Vec<Value>, bool)> {
    let block = &func.blocks[cur.index()];
    if cur_args.len() != block.params.len() || !(1..=MAX_DISP_INSTS).contains(&block.insts.len()) {
        return None;
    }
    let (&last, body) = block.insts.split_last()?;
    if !matches!(
        func.insts[last.index()].op,
        Op::Brif(..) | Op::Jump(..) | Op::BrTable(..)
    ) {
        return None;
    }
    bind(cx, state, &block.params, cur_args);
    for &inst in body {
        step_inst(func, cx, pred, inst, state)?;
    }
    let (next_b, raw_args, resolved_branch): (Block, &[Value], bool) = match &func.insts
        [last.index()]
    .op
    {
        Op::Brif(cond, then_b, then_args, else_b, else_args) => {
            let cond_val = lookup(&state.map, *cond);
            let (nb, args) = if eval_cond_truth(func, cx, state, cond_val)? {
                (*then_b, then_args)
            } else {
                (*else_b, else_args)
            };
            (nb, args, true)
        }
        Op::Jump(target, args) => (*target, args, false),
        Op::BrTable(index, default, targets) => {
            let idx_val = lookup(&state.map, *index);
            let next_b = if let Some(c) = lookup_const(&state.local_consts, cx.consts, idx_val) {
                crate::fold::table_target(
                    extend(c, func.ty(idx_val), false) as u64,
                    *default,
                    targets,
                )
            } else if let Some(u) = fact_of(&state.local_facts, cx.ranges, cx.ranges_len, idx_val)
                .and_then(|f| f.exact())
            {
                crate::fold::table_target(u, *default, targets)
            } else if targets.iter().all(|&t| t == *default) {
                *default
            } else {
                let ty = func.ty(idx_val);
                let fits_default = ty.is_small_int()
                    && (targets.len() as u64) <= ty.unsigned_max()
                    && state.edge_cmps.iter().any(|&(v, v_ty, cc1, imm1)| {
                        v == idx_val
                            && cmp_implies(
                                v_ty,
                                cc1,
                                imm1,
                                IntCC::UnsignedGreaterThanOrEqual,
                                targets.len() as i64,
                            ) == Some(true)
                    });
                if fits_default { *default } else { return None }
            };
            (next_b, &[], true)
        }
        _ => return None,
    };
    if next_b == cur {
        return None;
    }
    let next_args: Vec<Value> = raw_args.iter().map(|&v| lookup(&state.map, v)).collect();
    Some((next_b, next_args, resolved_branch))
}

fn marks(state: &EvalState) -> (usize, usize, usize) {
    (
        state.map.len(),
        state.local_ops.len(),
        state.local_consts.len(),
    )
}

fn eval_dispatcher(
    func: &Function,
    cx: &ThreadCx<'_>,
    single: &[Option<Block>],
    (pred, disp): (Option<Block>, Block),
    incoming: &[Value],
    seed: EdgeSeed,
) -> Option<Eval> {
    if func.blocks[disp.index()].params.is_empty()
        && matches!(seed, EdgeSeed::None)
        && !(pred.is_some() && stored::body_has_load(func, disp))
    {
        return None;
    }
    let mut state = EvalState::default();
    state.seed(func, cx, disp, seed);
    if let Some(p) = pred {
        state.seed_pred(func, cx, p, disp);
    }
    let seeded = state.seeded();
    if redefines(func, disp, &seeded) {
        return None;
    }
    let first = step_block(func, cx, pred, disp, incoming, &mut state)?;
    let mut visited = vec![disp];
    let mut stops = vec![(first, marks(&state))];
    while let Some(((next_b, next_args, resolved), _)) = stops.last() {
        if visited.len() >= MAX_CHAIN
            || cx.pinned[next_b.index()]
            || visited.contains(next_b)
            || redefines(func, *next_b, &seeded)
        {
            break;
        }
        let cur = *next_b;
        let Some((nb, na, step_resolved)) = step_block(func, cx, pred, cur, next_args, &mut state)
        else {
            break;
        };
        let resolved = *resolved | step_resolved;
        visited.push(cur);
        stops.push(((nb, na, resolved), marks(&state)));
    }
    while let Some(((chosen_target, mut chosen_args, resolved_branch), kept)) = stops.pop() {
        let chain = &visited[..=stops.len()];
        if !resolved_branch
            && (chain.len() == 1
                || crate::cfg::dominates(cx.idom, chosen_target, chain[chain.len() - 1]))
        {
            continue;
        }
        let Some(carry) = carry::plan(func, cx, single, (chain, incoming), chosen_target) else {
            continue;
        };
        state.map.truncate(kept.0);
        state.local_ops.truncate(kept.1);
        state.local_consts.truncate(kept.2);
        chosen_args.extend(carry.values.iter().map(|&v| lookup(&state.map, v)));
        let needed = needed_ops(&state, &chosen_args);
        return Some(Eval {
            chosen_target,
            chosen_args,
            local_ops: state.local_ops,
            local_consts: state.local_consts,
            needed,
            resolved_branch,
            carry,
        });
    }
    None
}

fn non_const_needed(eval: &Eval) -> usize {
    eval.local_ops
        .iter()
        .zip(&eval.needed)
        .filter(|&((res, _), &need)| need && !eval.local_consts.iter().any(|&(r, _)| r == *res))
        .count()
}

fn finish_split_branch(
    func: &mut Function,
    cx: &mut ThreadCx<'_>,
    pred: Block,
    cond: Value,
    et: Eval,
    ef: Eval,
) -> Op {
    note_use(cx.uses, pred, cond);
    let (t_b, t_a) = emit_needed(func, pred, (&mut *cx.consts, &mut *cx.uses), et);
    let (f_b, f_a) = emit_needed(func, pred, (&mut *cx.consts, &mut *cx.uses), ef);
    if t_b == f_b && t_a == f_a {
        return Op::Jump(t_b, t_a);
    }
    if t_b == f_b && t_a.len() == f_a.len() {
        let diffs: Vec<usize> = (0..t_a.len()).filter(|&i| t_a[i] != f_a[i]).collect();
        if diffs.len() == 1 && func.ty(t_a[diffs[0]]) == func.ty(f_a[diffs[0]]) {
            let i = diffs[0];
            let new_v = emit_select(func, cx, pred, cond, t_a[i], f_a[i]);
            let mut merged = t_a;
            merged[i] = new_v;
            return Op::Jump(t_b, merged);
        }
    }
    Op::Brif(cond, t_b, t_a, f_b, f_a)
}

fn eval_pair_ok(func: &Function, disp: Block, et: &Eval, ef: &Eval) -> bool {
    if et.chosen_target == disp
        || ef.chosen_target == disp
        || non_const_needed(et) + non_const_needed(ef) > MAX_BRIF_SPEC_OPS
    {
        return false;
    }
    if et.resolved_branch || ef.resolved_branch {
        return true;
    }
    if et.chosen_target != ef.chosen_target || et.chosen_args.len() != ef.chosen_args.len() {
        return false;
    }
    let mut diffs = et
        .chosen_args
        .iter()
        .zip(&ef.chosen_args)
        .filter(|(a, b)| a != b);
    match (diffs.next(), diffs.next()) {
        (None, _) => true,
        (Some((a, b)), None) => func.ty(*a) == func.ty(*b),
        _ => false,
    }
}

/// Splits each branch on a select with one constant arm into a branch on the select's condition, whose constant side goes straight to its target, and a new block that branches on the other arm when that arm is a comparison, a load, a block parameter or a logic op on flags.
pub(super) fn split_select_branches(func: &mut Function) -> bool {
    let uses = func.use_counts();
    let defs = crate::fold::defs(func);
    let konst = |v: Value| {
        defs[v.index()].and_then(|inst| match func.insts[inst.index()].op {
            Op::Const(k) => Some(k),
            _ => None,
        })
    };
    let compare = |v: Value| {
        defs[v.index()].is_none_or(|inst| match func.insts[inst.index()].op {
            Op::Icmp(..) | Op::Load(..) => true,
            Op::Binary(Binary::Band | Binary::Bor | Binary::Bxor, ..) => func.ty(v) == Type::I8,
            _ => false,
        })
    };
    let mut splits = Vec::new();
    for block in &func.blocks {
        let Some(&term) = block.insts.last() else {
            continue;
        };
        let Op::Brif(sel, ..) = func.insts[term.index()].op else {
            continue;
        };
        let Some((cond, then, otherwise)) =
            select_arms(func, &defs, sel).filter(|_| uses[sel.index()] == 1)
        else {
            continue;
        };
        match (konst(then), konst(otherwise)) {
            (Some(k), None) if compare(otherwise) => {
                splits.push((term, cond, k != 0, otherwise, true))
            }
            (None, Some(k)) if compare(then) => splits.push((term, cond, k != 0, then, false)),
            _ => {}
        }
    }
    for &(term, cond, fixed, other, on_true) in &splits {
        let Op::Brif(_, then, then_args, otherwise, else_args) =
            func.insts[term.index()].op.clone()
        else {
            continue;
        };
        let direct = if fixed {
            (then, then_args.clone())
        } else {
            (otherwise, else_args.clone())
        };
        let inner = func.add(Op::Brif(other, then, then_args, otherwise, else_args), None);
        let rest = Block(func.blocks.len() as u32);
        func.blocks.push(BlockData {
            params: Vec::new(),
            insts: vec![inner],
        });
        func.insts[term.index()].op = if on_true {
            Op::Brif(cond, direct.0, direct.1, rest, Vec::new())
        } else {
            Op::Brif(cond, rest, Vec::new(), direct.0, direct.1)
        };
    }
    if !splits.is_empty() {
        func.bump_cfg();
    }
    !splits.is_empty()
}

fn thread_jump_select(
    func: &mut Function,
    cx: &mut ThreadCx<'_>,
    pred: Block,
    disp: Block,
    term: Inst,
    args: &[Value],
) -> bool {
    for at in 0..args.len() {
        let Some((sel_c, args_t, args_f)) = select_split(func, cx.defs, args, at) else {
            continue;
        };
        let (Some(et), Some(ef)) = (
            eval_dispatcher(
                func,
                cx,
                &[],
                (Some(pred), disp),
                &args_t,
                EdgeSeed::Brif(sel_c, true),
            ),
            eval_dispatcher(
                func,
                cx,
                &[],
                (Some(pred), disp),
                &args_f,
                EdgeSeed::Brif(sel_c, false),
            ),
        ) else {
            continue;
        };
        if !eval_pair_ok(func, disp, &et, &ef) {
            continue;
        }
        func.insts[term.index()].op = finish_split_branch(func, cx, pred, sel_c, et, ef);
        return true;
    }
    for &arg in args {
        if lookup_const(&[], cx.consts, arg).is_some()
            || !is_bool_val(func, cx.defs, cx.consts, arg)
        {
            continue;
        }
        let (Some(et), Some(ef)) = (
            eval_dispatcher(
                func,
                cx,
                &[],
                (Some(pred), disp),
                args,
                EdgeSeed::Const(arg, 1),
            ),
            eval_dispatcher(
                func,
                cx,
                &[],
                (Some(pred), disp),
                args,
                EdgeSeed::Const(arg, 0),
            ),
        ) else {
            continue;
        };
        if !eval_pair_ok(func, disp, &et, &ef) {
            continue;
        }
        func.insts[term.index()].op = finish_split_branch(func, cx, pred, arg, et, ef);
        return true;
    }
    false
}

pub(super) fn thread_jump(
    func: &mut Function,
    cx: &mut ThreadCx<'_>,
    single: &mut [Option<Block>],
    pred: Block,
    disp: Block,
) -> bool {
    let Some(&term) = func.blocks[pred.index()].insts.last() else {
        return false;
    };
    let Op::Jump(target, ref args) = func.insts[term.index()].op else {
        return false;
    };
    if target != disp {
        return false;
    }
    let args = args.clone();
    if let Some(eval) = eval_dispatcher(func, cx, single, (Some(pred), disp), &args, EdgeSeed::None)
    {
        let (new_target, new_args) = carry::commit(func, cx, single, pred, eval);
        func.insts[term.index()].op = Op::Jump(new_target, new_args);
        return true;
    }
    if let Some((cond, taken)) = single_pred_brif_seed(func, single, pred)
        && let Some(eval) = eval_dispatcher(
            func,
            cx,
            single,
            (Some(pred), disp),
            &args,
            EdgeSeed::Brif(cond, taken),
        )
        && eval.chosen_target != disp
        && non_const_needed(&eval) <= MAX_BRIF_SPEC_OPS
    {
        let (new_target, new_args) = carry::commit(func, cx, single, pred, eval);
        func.insts[term.index()].op = Op::Jump(new_target, new_args);
        return true;
    }
    thread_jump_select(func, cx, pred, disp, term, &args)
}

fn brif_arm(
    func: &mut Function,
    cx: &mut ThreadCx<'_>,
    single: &mut [Option<Block>],
    pred: Block,
    disp: Block,
    taken: bool,
) -> Option<(Block, Vec<Value>)> {
    let Some(Op::Brif(cond, then_b, then_args, else_b, else_args)) = func.last(pred) else {
        return None;
    };
    let (target, args) = if taken {
        (*then_b, then_args)
    } else {
        (*else_b, else_args)
    };
    if target != disp {
        return None;
    }
    let ok = |eval: &Eval, strict: bool| {
        let needed = non_const_needed(eval);
        eval.chosen_target != target
            && needed <= MAX_BRIF_SPEC_OPS
            && (eval.resolved_branch || needed == 0)
            && (eval.carry.values.is_empty()
                || !func.successors(pred).any(|s| s == eval.chosen_target))
            && (!strict || (eval.resolved_branch && needed == 0 && eval.carry.values.is_empty()))
    };
    let eval = eval_dispatcher(
        func,
        cx,
        single,
        (Some(pred), disp),
        args,
        EdgeSeed::Brif(*cond, taken),
    )
    .filter(|eval| ok(eval, false))
    .or_else(|| {
        eval_dispatcher(
            func,
            cx,
            single,
            (Some(pred), disp),
            args,
            EdgeSeed::Equal(*cond, taken),
        )
        .filter(|eval| ok(eval, true))
    })?;
    Some(carry::commit(func, cx, single, pred, eval))
}

pub(super) fn thread_brif(
    func: &mut Function,
    cx: &mut ThreadCx<'_>,
    single: &mut [Option<Block>],
    pred: Block,
    disp: Block,
) -> bool {
    let mut changed = false;
    for taken in [true, false] {
        let Some(arm) = brif_arm(func, cx, single, pred, disp, taken) else {
            continue;
        };
        let Some(&term) = func.blocks[pred.index()].insts.last() else {
            continue;
        };
        let Op::Brif(_, then_b, then_args, else_b, else_args) = &mut func.insts[term.index()].op
        else {
            continue;
        };
        if taken {
            (*then_b, *then_args) = arm;
        } else {
            (*else_b, *else_args) = arm;
        }
        changed = true;
    }
    changed
}

pub(super) fn thread_br_table(
    func: &mut Function,
    cx: &ThreadCx<'_>,
    single: &[Option<Block>],
    pred: Block,
    disp: Block,
) -> bool {
    let Some(&term) = func.blocks[pred.index()].insts.last() else {
        return false;
    };
    let Op::BrTable(index, default, ref targets) = func.insts[term.index()].op else {
        return false;
    };
    if default != disp && !targets.contains(&disp) {
        return false;
    }
    let mut new_default = default;
    let mut new_targets = targets.clone();
    let mut changed = false;
    for (k, target) in new_targets.iter_mut().enumerate() {
        if *target != disp {
            continue;
        }
        let Some(eval) = eval_dispatcher(
            func,
            cx,
            &[],
            (Some(pred), disp),
            &[],
            EdgeSeed::Const(index, k as i64),
        ) else {
            continue;
        };
        if eval.chosen_args.is_empty()
            && non_const_needed(&eval) == 0
            && eval.chosen_target != *target
        {
            *target = eval.chosen_target;
            changed = true;
        }
    }
    if new_default == disp
        && let Some((cond, taken)) = single_pred_brif_seed(func, single, pred)
        && let Some(eval) = eval_dispatcher(
            func,
            cx,
            &[],
            (Some(pred), disp),
            &[],
            EdgeSeed::Brif(cond, taken),
        )
        && eval.chosen_args.is_empty()
        && non_const_needed(&eval) == 0
        && eval.chosen_target != new_default
    {
        new_default = eval.chosen_target;
        changed = true;
    }
    if changed {
        func.insts[term.index()].op = Op::BrTable(index, new_default, new_targets);
    }
    changed
}

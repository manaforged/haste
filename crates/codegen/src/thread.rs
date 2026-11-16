mod carry;
mod eval;
mod sink;
mod widen;

use crate::fold::extend;
use crate::ir::{Block, Domain, Function, Inst, InstData, IntCC, Op, SigRef, Value};

use self::eval::{
    EdgeSeed, EvalState, ThreadCx, bind, cmp_implies, eval_cond_truth, eval_op, fact_of, is_bool_val, lookup, lookup_const, seed_edge,
    step_inst,
};

const MAX_DISP_INSTS: usize = 10;
const MAX_JUMP_DISP_INSTS: usize = 5;
const MAX_CHAIN: usize = 8;
const MAX_BRIF_SPEC_OPS: usize = 4;

struct Eval {
    chosen_target: Block,
    chosen_args: Vec<Value>,
    local_ops: Vec<(Value, Op)>,
    local_consts: Vec<(Value, i64)>,
    needed: Vec<bool>,
    resolved_branch: bool,
    carry: carry::Carry,
}

fn block_bit(block: Block) -> u64 {
    1 << (block.index() % 64)
}

fn note_use(uses: &mut Vec<u64>, block: Block, value: Value) {
    if uses.len() <= value.index() {
        uses.resize(value.index() + 1, 0);
    }
    uses[value.index()] |= block_bit(block);
}

fn outside_uses(func: &Function) -> Vec<u64> {
    let mut uses = vec![0; func.types.len()];
    let mut home = vec![u32::MAX; func.types.len()];
    for (index, block) in func.blocks.iter().enumerate() {
        let id = index as u32;
        block.params.iter().for_each(|param| home[param.index()] = id);
        for &inst in &block.insts {
            let data = &func.insts[inst.index()];
            data.op.for_operands(|v| {
                if home[v.index()] != id {
                    uses[v.index()] |= block_bit(Block(id));
                }
            });
            if let Some(res) = data.result {
                home[res.index()] = id;
            }
        }
    }
    uses
}

fn step_block(
    func: &Function,
    cx: &ThreadCx<'_>,
    cur: Block,
    cur_args: &[Value],
    state: &mut EvalState,
) -> Option<(Block, Vec<Value>, bool)> {
    let block = &func.blocks[cur.index()];
    if cur_args.len() != block.params.len() || !(1..=MAX_DISP_INSTS).contains(&block.insts.len()) {
        return None;
    }
    let (&last, body) = block.insts.split_last()?;
    if !matches!(func.insts[last.index()].op, Op::Brif(..) | Op::Jump(..) | Op::BrTable(..)) {
        return None;
    }
    bind(cx, state, &block.params, cur_args);
    for &inst in body {
        step_inst(func, cx, inst, state)?;
    }
    let (next_b, raw_args, resolved_branch): (Block, &[Value], bool) = match &func.insts[last.index()].op {
        Op::Brif(cond, then_b, then_args, else_b, else_args) => {
            let cond_val = lookup(&state.map, *cond);
            let (nb, args) = if eval_cond_truth(func, cx, state, cond_val)? { (*then_b, then_args) } else { (*else_b, else_args) };
            (nb, args, true)
        }
        Op::Jump(target, args) => (*target, args, false),
        Op::BrTable(index, default, targets) => {
            let idx_val = lookup(&state.map, *index);
            let next_b = if let Some(c) = lookup_const(&state.local_consts, cx.consts, idx_val) {
                let u = extend(c, func.ty(idx_val), false) as u64;
                usize::try_from(u).ok().and_then(|at| targets.get(at).copied()).unwrap_or(*default)
            } else if let Some(u) = fact_of(&state.local_facts, cx.ranges, cx.ranges_len, idx_val).and_then(|f| f.exact()) {
                usize::try_from(u).ok().and_then(|at| targets.get(at).copied()).unwrap_or(*default)
            } else if targets.iter().all(|&t| t == *default) {
                *default
            } else {
                let ty = func.ty(idx_val);
                let fits_default = ty.is_small_int()
                    && (targets.len() as u64) <= ty.unsigned_max()
                    && state.edge_cmps.iter().any(|&(v, v_ty, cc1, imm1)| {
                        v == idx_val && cmp_implies(v_ty, cc1, imm1, IntCC::UnsignedGreaterThanOrEqual, targets.len() as i64) == Some(true)
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

fn redefines(func: &Function, block: Block, held: &[Value]) -> bool {
    !held.is_empty() && func.blocks[block.index()].insts.iter().filter_map(|inst| func.insts[inst.index()].result).any(|result| held.contains(&result))
}

fn marks(state: &EvalState) -> (usize, usize, usize) {
    (state.map.len(), state.local_ops.len(), state.local_consts.len())
}

fn eval_dispatcher(func: &Function, cx: &ThreadCx<'_>, single: &[Option<Block>], disp: Block, incoming: &[Value], seed: EdgeSeed) -> Option<Eval> {
    if func.blocks[disp.index()].params.is_empty() && matches!(seed, EdgeSeed::None) {
        return None;
    }
    let mut state = EvalState::default();
    match seed {
        EdgeSeed::None => {}
        EdgeSeed::Brif(cond, taken) => seed_edge(func, cx, cond, taken, 0, (&mut state.local_consts, &mut state.edge_cmps)),
        EdgeSeed::Const(val, imm) => state.local_consts.push((val, imm)),
    }
    let seeded = state.seeded();
    if redefines(func, disp, &seeded) {
        return None;
    }
    let first = step_block(func, cx, disp, incoming, &mut state)?;
    let mut visited = vec![disp];
    let mut stops = vec![(first, marks(&state))];
    while let Some(((next_b, next_args, resolved), _)) = stops.last() {
        if visited.len() >= MAX_CHAIN || cx.pinned[next_b.index()] || visited.contains(next_b) || redefines(func, *next_b, &seeded) {
            break;
        }
        let cur = *next_b;
        let Some((nb, na, step_resolved)) = step_block(func, cx, cur, next_args, &mut state) else {
            break;
        };
        let resolved = *resolved | step_resolved;
        visited.push(cur);
        stops.push(((nb, na, resolved), marks(&state)));
    }
    while let Some(((chosen_target, mut chosen_args, resolved_branch), kept)) = stops.pop() {
        let chain = &visited[..=stops.len()];
        if !resolved_branch && (chain.len() == 1 || crate::cfg::dominates(cx.idom, chosen_target, chain[chain.len() - 1])) {
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
        return Some(Eval { chosen_target, chosen_args, local_ops: state.local_ops, local_consts: state.local_consts, needed, resolved_branch, carry });
    }
    None
}

fn needed_ops(state: &EvalState, args: &[Value]) -> Vec<bool> {
    let mut needed = vec![false; state.local_ops.len()];
    for &arg in args {
        if let Some(idx) = state.local_ops.iter().position(|&(r, _)| r == arg) {
            needed[idx] = true;
        }
    }
    for i in (0..state.local_ops.len()).rev() {
        if !needed[i] || state.local_consts.iter().any(|&(r, _)| r == state.local_ops[i].0) {
            continue;
        }
        state.local_ops[i].1.for_operands(|v| {
            if let Some(idx) = state.local_ops.iter().position(|&(r, _)| r == v) {
                needed[idx] = true;
            }
        });
    }
    needed
}

fn non_const_needed(eval: &Eval) -> usize {
    eval.local_ops
        .iter()
        .zip(&eval.needed)
        .filter(|&((res, _), &need)| need && !eval.local_consts.iter().any(|&(r, _)| r == *res))
        .count()
}

fn resolve_emit_val(
    func: &mut Function,
    consts: &mut Vec<Option<i64>>,
    local_consts: &[(Value, i64)],
    emit_map: &mut Vec<(Value, Value)>,
    inserted: &mut Vec<Inst>,
    val: Value,
) -> Value {
    let mapped = lookup(emit_map, val);
    if mapped != val {
        return mapped;
    }
    let Some(&(_, imm)) = local_consts.iter().rev().find(|&&(r, _)| r == val) else {
        return val;
    };
    let c_val = Value(func.types.len() as u32);
    func.types.push(func.ty(val));
    consts.push(Some(imm));
    let c_inst = Inst(func.insts.len() as u32);
    func.insts.push(InstData { op: Op::Const(imm), result: Some(c_val) });
    inserted.push(c_inst);
    emit_map.push((val, c_val));
    c_val
}

fn emit_needed(func: &mut Function, pred: Block, (consts, uses): (&mut Vec<Option<i64>>, &mut Vec<u64>), eval: Eval) -> (Block, Vec<Value>) {
    let mut emit_map: Vec<(Value, Value)> = Vec::new();
    let mut inserted: Vec<Inst> = Vec::new();
    for (idx, (res, op)) in eval.local_ops.into_iter().enumerate() {
        if !eval.needed[idx] {
            continue;
        }
        let (new_op, imm_opt) = if let Some(&(_, imm)) = eval.local_consts.iter().find(|&&(r, _)| r == res) {
            (Op::Const(imm), Some(imm))
        } else {
            let mut mapped = op;
            mapped.map_operands(|v| resolve_emit_val(func, consts, &eval.local_consts, &mut emit_map, &mut inserted, v));
            (mapped, None)
        };
        let ty = func.ty(res);
        let new_val = Value(func.types.len() as u32);
        func.types.push(ty);
        consts.push(imm_opt);
        let new_inst = Inst(func.insts.len() as u32);
        func.insts.push(InstData { op: new_op, result: Some(new_val) });
        inserted.push(new_inst);
        emit_map.push((res, new_val));
    }
    let new_args: Vec<Value> = eval
        .chosen_args
        .into_iter()
        .map(|v| resolve_emit_val(func, consts, &eval.local_consts, &mut emit_map, &mut inserted, v))
        .collect();
    for &inst in &inserted {
        func.insts[inst.index()].op.for_operands(|v| note_use(uses, pred, v));
    }
    new_args.iter().for_each(|&v| note_use(uses, pred, v));
    if !inserted.is_empty() {
        let insts = &mut func.blocks[pred.index()].insts;
        let at = insts.len() - 1;
        insts.splice(at..at, inserted);
    }
    (eval.chosen_target, new_args)
}

fn emit_select(func: &mut Function, cx: &mut ThreadCx<'_>, pred: Block, cond: Value, then: Value, otherwise: Value) -> Value {
    let value = func.value(func.ty(then));
    cx.consts.push(None);
    let inst = func.add(Op::Select(cond, then, otherwise), Some(value));
    let insts = &mut func.blocks[pred.index()].insts;
    let at = insts.len() - 1;
    insts.insert(at, inst);
    value
}

fn finish_split_branch(func: &mut Function, cx: &mut ThreadCx<'_>, pred: Block, cond: Value, et: Eval, ef: Eval) -> Op {
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
    if et.chosen_target == disp || ef.chosen_target == disp || non_const_needed(et) + non_const_needed(ef) > MAX_BRIF_SPEC_OPS {
        return false;
    }
    if et.resolved_branch || ef.resolved_branch {
        return true;
    }
    if et.chosen_target != ef.chosen_target || et.chosen_args.len() != ef.chosen_args.len() {
        return false;
    }
    let mut diffs = et.chosen_args.iter().zip(&ef.chosen_args).filter(|(a, b)| a != b);
    match (diffs.next(), diffs.next()) {
        (None, _) => true,
        (Some((a, b)), None) => func.ty(*a) == func.ty(*b),
        _ => false,
    }
}

fn select_arms(func: &Function, defs: &[Option<Inst>], value: Value) -> Option<(Value, Value, Value)> {
    if func.is_unpredictable(value) {
        return None;
    }
    match func.insts[defs.get(value.index()).copied().flatten()?.index()].op {
        Op::Select(cond, then, otherwise) => Some((cond, then, otherwise)),
        _ => None,
    }
}

fn select_split(func: &Function, defs: &[Option<Inst>], args: &[Value], at: usize) -> Option<(Value, Vec<Value>, Vec<Value>)> {
    let (sel_c, sel_t, sel_f) = select_arms(func, defs, args[at])?;
    if sel_t == sel_f || args[..at].iter().any(|&a| select_arms(func, defs, a).is_some_and(|(c, t, f)| c == sel_c && t != f)) {
        return None;
    }
    let mut args_t = args.to_vec();
    let mut args_f = args.to_vec();
    for (j, &a) in args.iter().enumerate() {
        if let Some((c2, t2, f2)) = select_arms(func, defs, a)
            && c2 == sel_c
        {
            args_t[j] = t2;
            args_f[j] = f2;
        }
    }
    Some((sel_c, args_t, args_f))
}

fn thread_jump_select(func: &mut Function, cx: &mut ThreadCx<'_>, pred: Block, disp: Block, term: Inst, args: &[Value]) -> bool {
    for at in 0..args.len() {
        let Some((sel_c, args_t, args_f)) = select_split(func, cx.defs, args, at) else { continue };
        let (Some(et), Some(ef)) = (
            eval_dispatcher(func, cx, &[], disp, &args_t, EdgeSeed::Brif(sel_c, true)),
            eval_dispatcher(func, cx, &[], disp, &args_f, EdgeSeed::Brif(sel_c, false)),
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
        if lookup_const(&[], cx.consts, arg).is_some() || !is_bool_val(func, cx.defs, cx.consts, arg) {
            continue;
        }
        let (Some(et), Some(ef)) = (
            eval_dispatcher(func, cx, &[], disp, args, EdgeSeed::Const(arg, 1)),
            eval_dispatcher(func, cx, &[], disp, args, EdgeSeed::Const(arg, 0)),
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

fn single_pred_brif_seed(func: &Function, single: &[Option<Block>], pred: Block) -> Option<(Value, bool)> {
    let mut cur = pred;
    for _ in 0..2 {
        let parent = single[cur.index()]?;
        match func.last(parent) {
            Some(Op::Brif(cond, t_blk, _, e_blk, _)) if *t_blk != *e_blk && (*t_blk == cur || *e_blk == cur) => {
                return Some((*cond, *t_blk == cur));
            }
            Some(Op::Jump(target, _)) if *target == cur => cur = parent,
            _ => break,
        }
    }
    None
}

fn thread_jump(func: &mut Function, cx: &mut ThreadCx<'_>, single: &mut [Option<Block>], pred: Block, disp: Block) -> bool {
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
    if let Some(eval) = eval_dispatcher(func, cx, single, disp, &args, EdgeSeed::None) {
        let (new_target, new_args) = carry::commit(func, cx, single, pred, eval);
        func.insts[term.index()].op = Op::Jump(new_target, new_args);
        return true;
    }
    if let Some((cond, taken)) = single_pred_brif_seed(func, single, pred)
        && let Some(eval) = eval_dispatcher(func, cx, single, disp, &args, EdgeSeed::Brif(cond, taken))
        && eval.chosen_target != disp
        && non_const_needed(&eval) <= MAX_BRIF_SPEC_OPS
    {
        let (new_target, new_args) = carry::commit(func, cx, single, pred, eval);
        func.insts[term.index()].op = Op::Jump(new_target, new_args);
        return true;
    }
    thread_jump_select(func, cx, pred, disp, term, &args)
}

fn brif_arm(func: &mut Function, cx: &mut ThreadCx<'_>, single: &mut [Option<Block>], pred: Block, disp: Block, taken: bool) -> Option<(Block, Vec<Value>)> {
    let Some(Op::Brif(cond, then_b, then_args, else_b, else_args)) = func.last(pred) else {
        return None;
    };
    let (target, args) = if taken { (*then_b, then_args) } else { (*else_b, else_args) };
    if target != disp {
        return None;
    }
    let eval = eval_dispatcher(func, cx, single, disp, args, EdgeSeed::Brif(*cond, taken))?;
    if eval.chosen_target == target
        || non_const_needed(&eval) > MAX_BRIF_SPEC_OPS
        || (!eval.resolved_branch && non_const_needed(&eval) > 0)
        || (!eval.carry.values.is_empty() && func.successors(pred).any(|s| s == eval.chosen_target))
    {
        return None;
    }
    Some(carry::commit(func, cx, single, pred, eval))
}

fn thread_brif(func: &mut Function, cx: &mut ThreadCx<'_>, single: &mut [Option<Block>], pred: Block, disp: Block) -> bool {
    let mut changed = false;
    for taken in [true, false] {
        let Some(arm) = brif_arm(func, cx, single, pred, disp, taken) else {
            continue;
        };
        let Some(&term) = func.blocks[pred.index()].insts.last() else {
            continue;
        };
        let Op::Brif(_, then_b, then_args, else_b, else_args) = &mut func.insts[term.index()].op else {
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

fn thread_br_table(func: &mut Function, cx: &ThreadCx<'_>, single: &[Option<Block>], pred: Block, disp: Block) -> bool {
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
        let Some(eval) = eval_dispatcher(func, cx, &[], disp, &[], EdgeSeed::Const(index, k as i64)) else {
            continue;
        };
        if eval.chosen_args.is_empty() && non_const_needed(&eval) == 0 && eval.chosen_target != *target {
            *target = eval.chosen_target;
            changed = true;
        }
    }
    if new_default == disp
        && let Some((cond, taken)) = single_pred_brif_seed(func, single, pred)
        && let Some(eval) = eval_dispatcher(func, cx, &[], disp, &[], EdgeSeed::Brif(cond, taken))
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

fn known_consts(func: &Function) -> Vec<Option<i64>> {
    let mut consts = vec![None; func.types.len()];
    for data in &func.insts {
        if let (Op::Const(_), Some(res)) = (&data.op, data.result) {
            consts[res.index()] = eval_op(func, &data.op, func.ty(res), &|_| None);
        }
    }
    for block in &func.blocks {
        for &inst in &block.insts {
            let data = &func.insts[inst.index()];
            if let Some(res) = data.result && consts[res.index()].is_none() {
                consts[res.index()] = eval_op(func, &data.op, func.ty(res), &|v| consts[v.index()]);
            }
        }
    }
    consts
}

fn body(func: &Function, block: Block) -> &[Inst] {
    func.blocks[block.index()].insts.split_last().map_or(&[][..], |(_, body)| body)
}

fn steppable(func: &Function, block: Block) -> bool {
    let insts = &func.blocks[block.index()].insts;
    let Some((&last, body)) = insts.split_last() else {
        return false;
    };
    insts.len() <= MAX_DISP_INSTS
        && matches!(func.insts[last.index()].op, Op::Brif(..) | Op::Jump(..) | Op::BrTable(..))
        && body.iter().all(|inst| func.insts[inst.index()].result.is_some() && func.insts[inst.index()].op.is_speculable())
}

fn dispatches(func: &Function, pinned: &[bool], disp: Block) -> bool {
    if pinned[disp.index()] || !steppable(func, disp) {
        return false;
    }
    match func.last(disp) {
        Some(Op::Jump(next, _)) => {
            func.blocks[disp.index()].insts.len() <= MAX_JUMP_DISP_INSTS && *next != disp && !pinned[next.index()] && steppable(func, *next)
        }
        _ => true,
    }
}

fn threadable(func: &Function, pinned: &[bool]) -> bool {
    (0..func.blocks.len() as u32).map(Block).any(|pred| func.successors(pred).any(|disp| disp != pred && dispatches(func, pinned, disp)))
}

/// The blocks below the immediate dominator of each block holding an unpredictable select that reach that block, which together decide its condition.
pub(crate) fn unpredictable_region(func: &Function, cfg: &crate::cfg::Cfg) -> Vec<bool> {
    let mut region = vec![false; func.blocks.len()];
    if !func.has_unpredictable() {
        return region;
    }
    for (index, block) in func.blocks.iter().enumerate() {
        let marked = block.insts.iter().any(|inst| func.insts[inst.index()].result.is_some_and(|res| func.is_unpredictable(res)));
        if !marked || !cfg.reachable[index] {
            continue;
        }
        let here = Block(index as u32);
        let wall = cfg.idom[index].unwrap_or(here);
        let body = crate::cfg::natural_loop(cfg, wall, vec![here]).body;
        for (at, (inside, hit)) in region.iter_mut().zip(body).enumerate() {
            *inside |= hit && (at != wall.index() || at == index);
        }
    }
    region
}

/// Threads correlated constant- and range-condition jump chains through small pure dispatcher blocks.
pub(crate) fn run(func: &mut Function) -> Domain {
    let Some(entry) = func.entry else {
        return Domain::NONE;
    };
    if func.coroutine.is_some() {
        return Domain::NONE;
    }
    let mut pinned = vec![false; func.blocks.len()];
    pinned[entry.index()] = true;
    for data in &func.insts {
        if let Op::TryCall { normal, handler, .. } = data.op {
            pinned[normal.index()] = true;
            pinned[handler.index()] = true;
        }
    }
    if func.has_unpredictable() {
        let region = unpredictable_region(func, &func.cfg());
        pinned.iter_mut().zip(region).for_each(|(pin, inside)| *pin |= inside);
    }
    let mut consts = Vec::new();
    let mut any = false;
    for round in 0..16 {
        let mut defs = vec![None; func.types.len()];
        for block in &func.blocks {
            for &inst in &block.insts {
                let data = &func.insts[inst.index()];
                if let Some(res) = data.result {
                    defs[res.index()] = Some(inst);
                }
                if let Op::Call { results, .. } = &data.op {
                    for &res in results {
                        defs[res.index()] = Some(inst);
                    }
                }
            }
        }
        if !threadable(func, &pinned) && !sink::wanted(func, &defs, &pinned) {
            break;
        }
        if round == 0 {
            consts = known_consts(func);
        }
        let cfg = func.cfg();
        let single: Vec<Option<Block>> = (0..func.blocks.len()).map(|b| cfg.single_pred(Block(b as u32))).collect();
        let ranges = crate::facts::Ranges::new(func, (&defs, &consts), (&cfg.rpo, &single, &cfg.idom));
        let ranges_len = func.types.len();
        let mut uses = outside_uses(func);
        let mut cx = ThreadCx { defs: &defs, consts: &mut consts, uses: &mut uses, ranges: &ranges, ranges_len, pinned: &pinned, idom: &cfg.idom };
        let mut changed = sink::run(func, &mut cx, (&*cfg, &single));
        let mut stale = false;
        let mut preds = cfg.preds.clone();
        let mut live = single.clone();
        for disp_idx in 0..func.blocks.len() {
            let disp = Block(disp_idx as u32);
            if !dispatches(func, cx.pinned, disp) {
                continue;
            }
            let mut into = std::mem::take(&mut preds[disp_idx]);
            into.sort_unstable();
            into.dedup();
            for pred in into {
                if pred == disp {
                    continue;
                }
                let mut pred_changed = false;
                pred_changed |= thread_jump(func, &mut cx, &mut live, pred, disp);
                pred_changed |= thread_brif(func, &mut cx, &mut live, pred, disp);
                pred_changed |= thread_br_table(func, &cx, &live, pred, disp);
                if pred_changed {
                    changed = true;
                    func.successors(pred).for_each(|s| preds[s.index()].push(pred));
                    if func.successors(pred).any(|s| {
                        s != disp
                            && single[s.index()] == live[s.index()]
                            && (single[s.index()].is_some()
                                || cfg.idom[s.index()].is_some_and(|d| d != s && !crate::cfg::dominates(&cfg.idom, d, pred)))
                    }) {
                        stale = true;
                        break;
                    }
                }
            }
            if stale {
                break;
            }
        }
        if !changed {
            break;
        }
        func.bump_cfg();
        any = true;
    }
    let widened = widen::run(func);
    let mut dirty = if any || widened {
        crate::fold::prune(func, entry);
        Domain::CFG | Domain::VALUES
    } else {
        Domain::NONE
    };
    dirty |= sink::sink_cold(func);
    dirty
}

/// Renumbers the instructions and values the blocks hold and drops the rest, so a body kept or copied whole carries no dead entries.
pub(crate) fn compact(func: &mut Function) {
    if func.coroutine.is_some() {
        return;
    }
    let mut map: Vec<Option<Value>> = vec![None; func.types.len()];
    let mut types = Vec::new();
    let old_flags = std::mem::take(&mut func.value_flags);
    let mut flags = Vec::new();
    let mut take = |value: Value| -> Value {
        *map[value.index()].get_or_insert_with(|| {
            types.push(func.types[value.index()]);
            if !old_flags.is_empty() {
                flags.push(old_flags.get(value.index()).copied().unwrap_or(0));
            }
            Value(types.len() as u32 - 1)
        })
    };
    let live = func.blocks.iter().map(|block| block.insts.len()).sum();
    let mut insts = Vec::with_capacity(live);
    let mut blocks = std::mem::take(&mut func.blocks);
    for block in &mut blocks {
        block.params.iter_mut().for_each(|param| *param = take(*param));
        for inst in &mut block.insts {
            let slot = &mut func.insts[inst.index()];
            let mut data = InstData { op: std::mem::replace(&mut slot.op, Op::Trap), result: slot.result };
            data.result = data.result.map(&mut take);
            if let Op::Call { results, .. } = &mut data.op {
                results.iter_mut().for_each(|result| *result = take(*result));
            }
            data.op.map_operands(&mut take);
            insts.push(data);
            *inst = Inst(insts.len() as u32 - 1);
        }
    }
    func.blocks = blocks;
    func.insts = insts;
    types.shrink_to_fit();
    func.types = types;
    if flags.iter().all(|&bits| bits == 0) {
        flags.clear();
    }
    func.value_flags = flags;
    compact_blocks(func);
    compact_sigs(func);
}

fn compact_blocks(func: &mut Function) {
    let mut map: Vec<Option<Block>> = vec![None; func.blocks.len()];
    let mut blocks = Vec::new();
    for (index, mut block) in std::mem::take(&mut func.blocks).into_iter().enumerate() {
        if block.insts.is_empty() && func.entry != Some(Block(index as u32)) {
            continue;
        }
        block.params.shrink_to_fit();
        block.insts.shrink_to_fit();
        map[index] = Some(Block(blocks.len() as u32));
        blocks.push(block);
    }
    let renumber = |block: Block| map[block.index()].unwrap_or(Block(0));
    func.insts.iter_mut().for_each(|data| data.op.map_targets(renumber));
    func.entry = func.entry.map(renumber);
    func.blocks = blocks;
    func.bump_cfg();
}

fn compact_sigs(func: &mut Function) {
    let mut map: Vec<Option<SigRef>> = vec![None; func.sigs.len()];
    let mut sigs = Vec::new();
    for data in &mut func.insts {
        if let Op::Call { sig, .. } | Op::TryCall { sig, .. } = &mut data.op {
            *sig = *map[sig.index()].get_or_insert_with(|| {
                sigs.push(func.sigs[sig.index()].clone());
                SigRef(sigs.len() as u32 - 1)
            });
        }
    }
    func.sigs = sigs;
}

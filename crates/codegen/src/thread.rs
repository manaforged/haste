mod widen;

use crate::facts::{Fact, Ranges, decide};
use crate::fold::{evaluate_wrapped, extend};
use crate::ir::{Binary, Block, Convert, Domain, Function, Inst, InstData, IntCC, Op, SigRef, Type, Value};

const MAX_DISP_INSTS: usize = 10;
const MAX_CHAIN: usize = 8;
const MAX_BRIF_SPEC_OPS: usize = 2;

#[derive(Clone, Copy)]
enum EdgeSeed {
    None,
    Brif(Value, bool),
    Const(Value, i64),
}

struct ThreadCx<'a> {
    defs: &'a [Option<Inst>],
    consts: &'a mut Vec<Option<i64>>,
    ranges: &'a Ranges<'a>,
    ranges_len: usize,
    pinned: &'a [bool],
}

fn lookup(map: &[(Value, Value)], value: Value) -> Value {
    map.iter().find(|&&(k, _)| k == value).map_or(value, |&(_, v)| v)
}

fn lookup_const(local: &[(Value, i64)], global: &[Option<i64>], value: Value) -> Option<i64> {
    local.iter().find(|&&(k, _)| k == value).map(|&(_, imm)| imm).or_else(|| global.get(value.index()).copied().flatten())
}

fn fact_of(local_facts: &[(Value, Fact)], ranges: &Ranges<'_>, ranges_len: usize, value: Value) -> Option<Fact> {
    local_facts
        .iter()
        .rev()
        .find(|&&(k, _)| k == value)
        .map(|(_, f)| f.clone())
        .or_else(|| (value.index() < ranges_len).then(|| ranges.of(value)))
}

fn swap_int_cc(cc: IntCC) -> IntCC {
    match cc {
        IntCC::Equal | IntCC::NotEqual => cc,
        IntCC::SignedLessThan => IntCC::SignedGreaterThan,
        IntCC::SignedGreaterThanOrEqual => IntCC::SignedLessThanOrEqual,
        IntCC::SignedGreaterThan => IntCC::SignedLessThan,
        IntCC::SignedLessThanOrEqual => IntCC::SignedGreaterThanOrEqual,
        IntCC::UnsignedLessThan => IntCC::UnsignedGreaterThan,
        IntCC::UnsignedGreaterThanOrEqual => IntCC::UnsignedLessThanOrEqual,
        IntCC::UnsignedGreaterThan => IntCC::UnsignedLessThan,
        IntCC::UnsignedLessThanOrEqual => IntCC::UnsignedGreaterThanOrEqual,
    }
}

fn bounds_of_cmp(ty: Type, cc: IntCC, imm: i64) -> Option<(bool, i128, i128)> {
    if !cc.is_signed() {
        let (u_max, u) = (i128::from(ty.unsigned_max()), i128::from(extend(imm, ty, false) as u64));
        let (lo, hi) = match cc {
            IntCC::Equal => (u, u),
            IntCC::NotEqual if u == 0 && u_max > 0 => (1, u_max),
            IntCC::NotEqual if u == u_max && u_max > 0 => (0, u_max - 1),
            IntCC::UnsignedLessThan if u > 0 => (0, u - 1),
            IntCC::UnsignedLessThanOrEqual => (0, u),
            IntCC::UnsignedGreaterThan if u < u_max => (u + 1, u_max),
            IntCC::UnsignedGreaterThanOrEqual => (u, u_max),
            _ => return None,
        };
        return Some((false, lo, hi));
    }
    let (s_min, s_max) = if ty == Type::I64 {
        (i128::from(i64::MIN), i128::from(i64::MAX))
    } else {
        let h = 1_i128 << (ty.bits() - 1);
        (-h, h - 1)
    };
    let s = i128::from(extend(imm, ty, true));
    let (lo, hi) = match cc {
        IntCC::SignedLessThan if s > s_min => (s_min, s - 1),
        IntCC::SignedLessThanOrEqual => (s_min, s),
        IntCC::SignedGreaterThan if s < s_max => (s + 1, s_max),
        IntCC::SignedGreaterThanOrEqual => (s, s_max),
        _ => return None,
    };
    Some((true, lo, hi))
}

fn cmp_implies(ty: Type, cc1: IntCC, imm1: i64, cc2: IntCC, imm2: i64) -> Option<bool> {
    if cc1 == IntCC::NotEqual && matches!(cc2, IntCC::Equal | IntCC::NotEqual)
        && extend(imm1, ty, false) == extend(imm2, ty, false)
    {
        return Some(cc2 == IntCC::NotEqual);
    }
    let (signed, lo, hi) = bounds_of_cmp(ty, cc1, imm1)?;
    let c2 = if signed {
        if !cc2.is_signed() && !matches!(cc2, IntCC::Equal | IntCC::NotEqual) {
            return None;
        }
        i128::from(extend(imm2, ty, true))
    } else {
        let half = i128::from(ty.unsigned_max() >> 1);
        let c = i128::from(extend(imm2, ty, false) as u64);
        if cc2.is_signed() && (hi > half || c > half) {
            return None;
        }
        c
    };
    match cc2 {
        IntCC::Equal => (lo == hi).then_some(lo == c2).or_else(|| (c2 < lo || c2 > hi).then_some(false)),
        IntCC::NotEqual => (lo == hi).then_some(lo != c2).or_else(|| (c2 < lo || c2 > hi).then_some(true)),
        IntCC::UnsignedLessThan | IntCC::SignedLessThan => {
            if hi < c2 { Some(true) } else if lo >= c2 { Some(false) } else { None }
        }
        IntCC::UnsignedLessThanOrEqual | IntCC::SignedLessThanOrEqual => {
            if hi <= c2 { Some(true) } else if lo > c2 { Some(false) } else { None }
        }
        IntCC::UnsignedGreaterThan | IntCC::SignedGreaterThan => {
            if lo > c2 { Some(true) } else if hi <= c2 { Some(false) } else { None }
        }
        IntCC::UnsignedGreaterThanOrEqual | IntCC::SignedGreaterThanOrEqual => {
            if lo >= c2 { Some(true) } else if hi < c2 { Some(false) } else { None }
        }
    }
}

fn seed_edge(
    func: &Function,
    cx: &ThreadCx<'_>,
    cond: Value,
    taken: bool,
    depth: u8,
    (local_consts, edge_cmps): (&mut Vec<(Value, i64)>, &mut Vec<(Value, Type, IntCC, i64)>),
) {
    if depth > 4 {
        return;
    }
    let Some(inst) = cx.defs.get(cond.index()).copied().flatten() else {
        if !taken {
            local_consts.push((cond, 0));
        }
        return;
    };
    let op = &func.insts[inst.index()].op;
    if !taken {
        local_consts.push((cond, 0));
    } else if matches!(op, Op::Icmp(..) | Op::Fcmp(..) | Op::Fits(..)) {
        local_consts.push((cond, 1));
    }
    match *op {
        Op::Binary(Binary::Band, a, b) if taken => {
            seed_edge(func, cx, a, true, depth + 1, (local_consts, edge_cmps));
            seed_edge(func, cx, b, true, depth + 1, (local_consts, edge_cmps));
        }
        Op::Binary(Binary::Bor, a, b) if !taken => {
            seed_edge(func, cx, a, false, depth + 1, (local_consts, edge_cmps));
            seed_edge(func, cx, b, false, depth + 1, (local_consts, edge_cmps));
        }
        Op::Icmp(cc, x, y) if func.ty(x).is_small_int() => {
            let eff = if taken { cc } else { cc.inverse() };
            let (lhs, ord, imm) = match (lookup_const(local_consts, cx.consts, x), lookup_const(local_consts, cx.consts, y)) {
                (None, Some(k)) => (x, eff, k),
                (Some(k), None) => (y, swap_int_cc(eff), k),
                _ => return,
            };
            if ord == IntCC::Equal {
                local_consts.push((lhs, imm));
            } else {
                edge_cmps.push((lhs, func.ty(lhs), ord, imm));
            }
        }
        _ => {}
    }
}

fn eval_op(func: &Function, op: &Op, ty: Type, known: &dyn Fn(Value) -> Option<i64>) -> Option<i64> {
    if let Some(imm) = evaluate_wrapped(func, op, ty, known) {
        return Some(imm);
    }
    if !ty.is_small_int() {
        return None;
    }
    let is_zero = |v: Value| known(v).is_some_and(|imm| extend(imm, ty, false) == 0);
    match *op {
        Op::Convert(Convert::Bitcast, x) if func.ty(x).is_small_int() && func.ty(x).bits() == ty.bits() => {
            known(x).map(|imm| extend(imm, ty, true))
        }
        Op::Select(c, x, y) => {
            if x == y {
                return known(x).map(|imm| extend(imm, ty, true));
            }
            let chosen = if extend(known(c)?, func.ty(c), false) != 0 { x } else { y };
            known(chosen).map(|imm| extend(imm, ty, true))
        }
        Op::Binary(Binary::Imul | Binary::Umulhi | Binary::Smulhi | Binary::Band, x, y) if is_zero(x) || is_zero(y) => Some(0),
        Op::Binary(Binary::Isub | Binary::Bxor, x, y) if x == y => Some(0),
        Op::Icmp(cc, x, y) if x == y && func.ty(x).is_small_int() => Some(i64::from(matches!(
            cc,
            IntCC::Equal | IntCC::SignedGreaterThanOrEqual | IntCC::SignedLessThanOrEqual | IntCC::UnsignedGreaterThanOrEqual | IntCC::UnsignedLessThanOrEqual
        ))),
        _ => None,
    }
}

fn eval_cond_truth(func: &Function, cx: &ThreadCx<'_>, state: &EvalState, cond_val: Value) -> Option<bool> {
    if let Some(c) = lookup_const(&state.local_consts, cx.consts, cond_val) {
        return Some(extend(c, func.ty(cond_val), false) != 0);
    }
    if let Some(u) = fact_of(&state.local_facts, cx.ranges, cx.ranges_len, cond_val).and_then(|f| f.exact()) {
        return Some(u != 0);
    }
    state.edge_cmps.iter().find_map(|&(v, v_ty, cc1, imm1)| (v == cond_val).then(|| cmp_implies(v_ty, cc1, imm1, IntCC::NotEqual, 0)).flatten())
}

fn is_bool_val(func: &Function, defs: &[Option<Inst>], consts: &[Option<i64>], v: Value) -> bool {
    let Some(inst) = defs.get(v.index()).copied().flatten() else { return false };
    let is_one = |u: Value| consts.get(u.index()).copied().flatten() == Some(1);
    match func.insts[inst.index()].op {
        Op::Icmp(..) | Op::Fcmp(..) | Op::Fits(..) => true,
        Op::Convert(Convert::Uextend, inner) => is_bool_val(func, defs, consts, inner),
        Op::Binary(Binary::Band, a, b) => is_one(a) || is_one(b) || is_bool_val(func, defs, consts, a) || is_bool_val(func, defs, consts, b),
        Op::Binary(Binary::Bxor, a, b) => (is_one(b) && is_bool_val(func, defs, consts, a)) || (is_one(a) && is_bool_val(func, defs, consts, b)),
        _ => false,
    }
}

fn eval_cmp_facts(
    func: &Function,
    cx: &ThreadCx<'_>,
    op: &Op,
    (local_consts, local_facts, edge_cmps): (&[(Value, i64)], &[(Value, Fact)], &[(Value, Type, IntCC, i64)]),
) -> Option<i64> {
    let &Op::Icmp(cc, x, y) = op else { return None };
    let ty = func.ty(x);
    if !ty.is_small_int() {
        return None;
    }
    if let (Some(fx), Some(fy)) = (
        fact_of(local_facts, cx.ranges, cx.ranges_len, x),
        fact_of(local_facts, cx.ranges, cx.ranges_len, y),
    ) && let Some(res) = decide(cc, ty, &fx, &fy)
    {
        return Some(i64::from(res));
    }
    let (lhs, ord, imm2) = match (lookup_const(local_consts, cx.consts, x), lookup_const(local_consts, cx.consts, y)) {
        (None, Some(k)) => (x, cc, k),
        (Some(k), None) => (y, swap_int_cc(cc), k),
        _ => return None,
    };
    for &(v, v_ty, cc1, imm1) in edge_cmps {
        if v == lhs && v_ty == ty && let Some(res) = cmp_implies(ty, cc1, imm1, ord, imm2) {
            return Some(i64::from(res));
        }
    }
    None
}

struct Eval {
    chosen_target: Block,
    chosen_args: Vec<Value>,
    local_ops: Vec<(Value, Op)>,
    local_consts: Vec<(Value, i64)>,
    needed: Vec<bool>,
}

fn escapes_from(func: &Function, start: Block, def_block: Block, defs: &[Value]) -> bool {
    if defs.is_empty() || start == def_block {
        return false;
    }
    let mut seen = vec![false; func.blocks.len()];
    seen[def_block.index()] = true;
    let mut stack = vec![start];
    while let Some(b) = stack.pop() {
        if std::mem::replace(&mut seen[b.index()], true) {
            continue;
        }
        for &inst in &func.blocks[b.index()].insts {
            let mut hit = false;
            func.insts[inst.index()].op.for_operands(|v| hit |= defs.contains(&v));
            if hit {
                return true;
            }
        }
        stack.extend(func.successors(b));
    }
    false
}

#[derive(Default)]
struct EvalState {
    map: Vec<(Value, Value)>,
    local_ops: Vec<(Value, Op)>,
    local_consts: Vec<(Value, i64)>,
    local_facts: Vec<(Value, Fact)>,
    edge_cmps: Vec<(Value, Type, IntCC, i64)>,
}

fn step_block(
    func: &Function,
    cx: &ThreadCx<'_>,
    cur: Block,
    cur_args: &[Value],
    state: &mut EvalState,
) -> Option<(Block, Vec<Value>, Vec<Value>)> {
    let block = &func.blocks[cur.index()];
    if cur_args.len() != block.params.len() || !(1..=MAX_DISP_INSTS).contains(&block.insts.len()) {
        return None;
    }
    let (&last, body) = block.insts.split_last()?;
    if !matches!(func.insts[last.index()].op, Op::Brif(..) | Op::Jump(..) | Op::BrTable(..)) {
        return None;
    }
    let defs_here: Vec<Value> = block.params.iter().chain(body.iter().filter_map(|&i| func.insts[i.index()].result.as_ref())).copied().collect();
    for (&p, &a) in block.params.iter().zip(cur_args) {
        state.map.retain(|&(k, _)| k != p);
        state.map.push((p, a));
        if let Some(f) = fact_of(&state.local_facts, cx.ranges, cx.ranges_len, a) {
            state.local_facts.retain(|&(k, _)| k != p);
            state.local_facts.push((p, f));
        }
    }
    for &inst in body {
        let data = &func.insts[inst.index()];
        let res = data.result?;
        if !data.op.is_speculable() {
            return None;
        }
        let mut op = data.op.clone();
        op.map_operands(|v| lookup(&state.map, v));
        if let Some(imm) = eval_op(func, &op, func.ty(res), &|v| lookup_const(&state.local_consts, cx.consts, v))
            .or_else(|| eval_cmp_facts(func, cx, &op, (&state.local_consts, &state.local_facts, &state.edge_cmps)))
        {
            state.local_consts.push((res, imm));
        } else if let Op::Select(c, x, y) = op
            && let Some(taken) = eval_cond_truth(func, cx, state, c)
        {
            let chosen = if taken { x } else { y };
            state.map.retain(|&(k, _)| k != res);
            state.map.push((res, chosen));
            if let Some(imm) = lookup_const(&state.local_consts, cx.consts, chosen) {
                state.local_consts.push((res, imm));
            }
            if let Some(f) = fact_of(&state.local_facts, cx.ranges, cx.ranges_len, chosen) {
                state.local_facts.retain(|&(k, _)| k != res);
                state.local_facts.push((res, f));
            }
            continue;
        }
        state.local_ops.push((res, op));
        state.map.retain(|&(k, _)| k != res);
    }
    let (next_b, raw_args): (Block, &[Value]) = match &func.insts[last.index()].op {
        Op::Brif(cond, then_b, then_args, else_b, else_args) => {
            let cond_val = lookup(&state.map, *cond);
            if eval_cond_truth(func, cx, state, cond_val)? { (*then_b, then_args) } else { (*else_b, else_args) }
        }
        Op::Jump(target, args) => (*target, args),
        Op::BrTable(index, default, targets) => {
            let idx_val = lookup(&state.map, *index);
            let next_b = if let Some(c) = lookup_const(&state.local_consts, cx.consts, idx_val) {
                let u = extend(c, func.ty(idx_val), false) as u64;
                usize::try_from(u).ok().and_then(|at| targets.get(at).copied()).unwrap_or(*default)
            } else {
                let ty = func.ty(idx_val);
                let fits_default = ty.is_small_int()
                    && (targets.len() as u64) <= ty.unsigned_max()
                    && state.edge_cmps.iter().any(|&(v, v_ty, cc1, imm1)| {
                        v == idx_val && cmp_implies(v_ty, cc1, imm1, IntCC::UnsignedGreaterThanOrEqual, targets.len() as i64) == Some(true)
                    });
                if fits_default { *default } else { return None }
            };
            (next_b, &[])
        }
        _ => return None,
    };
    if next_b == cur {
        return None;
    }
    let next_args: Vec<Value> = raw_args.iter().map(|&v| lookup(&state.map, v)).collect();
    Some((next_b, next_args, defs_here))
}

fn eval_dispatcher(func: &Function, cx: &ThreadCx<'_>, disp: Block, incoming: &[Value], seed: EdgeSeed) -> Option<Eval> {
    if func.blocks[disp.index()].params.is_empty() && matches!(seed, EdgeSeed::None) {
        return None;
    }
    let mut state = EvalState::default();
    match seed {
        EdgeSeed::None => {}
        EdgeSeed::Brif(cond, taken) => seed_edge(func, cx, cond, taken, 0, (&mut state.local_consts, &mut state.edge_cmps)),
        EdgeSeed::Const(val, imm) => state.local_consts.push((val, imm)),
    }
    let mut visited: Vec<(Block, Vec<Value>)> = Vec::new();
    let (mut next_b, mut next_args, first_defs) = step_block(func, cx, disp, incoming, &mut state)?;
    visited.push((disp, first_defs));
    while visited.len() < MAX_CHAIN && !cx.pinned[next_b.index()] && !visited.iter().any(|&(b, _)| b == next_b) {
        let marks = (state.map.len(), state.local_ops.len(), state.local_consts.len(), state.local_facts.len());
        let cur = next_b;
        if let Some((nb, na, defs)) = step_block(func, cx, cur, &next_args, &mut state) {
            visited.push((cur, defs));
            next_b = nb;
            next_args = na;
        } else {
            state.map.truncate(marks.0);
            state.local_ops.truncate(marks.1);
            state.local_consts.truncate(marks.2);
            state.local_facts.truncate(marks.3);
            break;
        }
    }
    if visited.len() == 1 && func.last(disp).is_some_and(|op| matches!(op, Op::Jump(..))) {
        return None;
    }
    let (chosen_target, chosen_args) = (next_b, next_args);
    for (def_block, defs) in &visited {
        if escapes_from(func, chosen_target, *def_block, defs) {
            return None;
        }
    }
    let mut needed = vec![false; state.local_ops.len()];
    for &arg in &chosen_args {
        if let Some(idx) = state.local_ops.iter().position(|&(r, _)| r == arg) {
            needed[idx] = true;
        }
    }
    for i in (0..state.local_ops.len()).rev() {
        if !needed[i] || state.local_consts.iter().any(|&(r, _)| r == state.local_ops[i].0) {
            continue;
        }
        let (_, ref op) = state.local_ops[i];
        let mut refs = Vec::new();
        op.for_operands(|v| refs.push(v));
        for v in refs {
            if let Some(idx) = state.local_ops.iter().position(|&(r, _)| r == v) {
                needed[idx] = true;
            }
        }
    }
    Some(Eval { chosen_target, chosen_args, local_ops: state.local_ops, local_consts: state.local_consts, needed })
}

fn non_const_needed(eval: &Eval) -> usize {
    eval.local_ops
        .iter()
        .zip(&eval.needed)
        .filter(|&((res, _), &need)| need && !eval.local_consts.iter().any(|&(r, _)| r == *res))
        .count()
}

fn emit_needed(func: &mut Function, pred: Block, consts: &mut Vec<Option<i64>>, eval: Eval) -> (Block, Vec<Value>) {
    let mut emit_map: Vec<(Value, Value)> = Vec::new();
    let mut inserted: Vec<Inst> = Vec::new();
    for (idx, (res, op)) in eval.local_ops.into_iter().enumerate() {
        if !eval.needed[idx] {
            continue;
        }
        let ty = func.ty(res);
        let new_val = Value(func.types.len() as u32);
        func.types.push(ty);
        let (new_op, imm_opt) = if let Some(&(_, imm)) = eval.local_consts.iter().find(|&&(r, _)| r == res) {
            (Op::Const(imm), Some(imm))
        } else {
            let mut mapped = op;
            mapped.map_operands(|v| lookup(&emit_map, v));
            (mapped, None)
        };
        consts.push(imm_opt);
        let new_inst = Inst(func.insts.len() as u32);
        func.insts.push(InstData { op: new_op, result: Some(new_val) });
        inserted.push(new_inst);
        emit_map.push((res, new_val));
    }
    let new_args: Vec<Value> = eval.chosen_args.into_iter().map(|v| lookup(&emit_map, v)).collect();
    if !inserted.is_empty() {
        let insts = &mut func.blocks[pred.index()].insts;
        let at = insts.len() - 1;
        insts.splice(at..at, inserted);
    }
    (eval.chosen_target, new_args)
}

fn finish_split_branch(func: &mut Function, cx: &mut ThreadCx<'_>, pred: Block, cond: Value, et: Eval, ef: Eval) -> Op {
    let (t_b, t_a) = emit_needed(func, pred, cx.consts, et);
    let (f_b, f_a) = emit_needed(func, pred, cx.consts, ef);
    if t_b == f_b && t_a == f_a {
        return Op::Jump(t_b, t_a);
    }
    if t_b == f_b && t_a.len() == f_a.len() {
        let diffs: Vec<usize> = (0..t_a.len()).filter(|&i| t_a[i] != f_a[i]).collect();
        if diffs.len() == 1 && func.ty(t_a[diffs[0]]) == func.ty(f_a[diffs[0]]) {
            let i = diffs[0];
            let new_v = Value(func.types.len() as u32);
            func.types.push(func.ty(t_a[i]));
            cx.consts.push(None);
            let sel_inst = Inst(func.insts.len() as u32);
            func.insts.push(InstData { op: Op::Select(cond, t_a[i], f_a[i]), result: Some(new_v) });
            let insts = &mut func.blocks[pred.index()].insts;
            let at = insts.len() - 1;
            insts.insert(at, sel_inst);
            let mut merged = t_a;
            merged[i] = new_v;
            return Op::Jump(t_b, merged);
        }
    }
    Op::Brif(cond, t_b, t_a, f_b, f_a)
}

fn thread_jump_select(func: &mut Function, cx: &mut ThreadCx<'_>, pred: Block, disp: Block, term: Inst, args: &[Value]) -> bool {
    for (k, &arg) in args.iter().enumerate() {
        let Some(def) = cx.defs.get(arg.index()).copied().flatten() else { continue };
        let Op::Select(sel_c, sel_t, sel_f) = func.insts[def.index()].op else { continue };
        if sel_t == sel_f {
            continue;
        }
        let mut args_t = args.to_vec();
        args_t[k] = sel_t;
        let mut args_f = args.to_vec();
        args_f[k] = sel_f;
        let (Some(et), Some(ef)) = (
            eval_dispatcher(func, cx, disp, &args_t, EdgeSeed::None),
            eval_dispatcher(func, cx, disp, &args_f, EdgeSeed::None),
        ) else {
            continue;
        };
        if non_const_needed(&et) + non_const_needed(&ef) > MAX_BRIF_SPEC_OPS {
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
            eval_dispatcher(func, cx, disp, args, EdgeSeed::Const(arg, 1)),
            eval_dispatcher(func, cx, disp, args, EdgeSeed::Const(arg, 0)),
        ) else {
            continue;
        };
        if et.chosen_target == disp || ef.chosen_target == disp || non_const_needed(&et) + non_const_needed(&ef) > MAX_BRIF_SPEC_OPS {
            continue;
        }
        func.insts[term.index()].op = finish_split_branch(func, cx, pred, arg, et, ef);
        return true;
    }
    false
}

fn thread_jump(func: &mut Function, cx: &mut ThreadCx<'_>, pred: Block, disp: Block) -> bool {
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
    if let Some(eval) = eval_dispatcher(func, cx, disp, &args, EdgeSeed::None) {
        let (new_target, new_args) = emit_needed(func, pred, cx.consts, eval);
        func.insts[term.index()].op = Op::Jump(new_target, new_args);
        return true;
    }
    thread_jump_select(func, cx, pred, disp, term, &args)
}

fn thread_brif(func: &mut Function, cx: &mut ThreadCx<'_>, pred: Block, disp: Block) -> bool {
    let Some(&term) = func.blocks[pred.index()].insts.last() else {
        return false;
    };
    let Op::Brif(cond, then_b, then_args, else_b, else_args) = func.insts[term.index()].op.clone() else {
        return false;
    };
    let mut changed = false;
    let mut rewrite = |target: &mut Block, args: &mut Vec<Value>, taken: bool| {
        if *target != disp {
            return;
        }
        let Some(eval) = eval_dispatcher(func, cx, disp, args, EdgeSeed::Brif(cond, taken)) else {
            return;
        };
        if eval.chosen_target == *target || non_const_needed(&eval) > MAX_BRIF_SPEC_OPS {
            return;
        }
        (*target, *args) = emit_needed(func, pred, cx.consts, eval);
        changed = true;
    };
    let (mut t_b, mut t_a, mut e_b, mut e_a) = (then_b, then_args, else_b, else_args);
    rewrite(&mut t_b, &mut t_a, true);
    rewrite(&mut e_b, &mut e_a, false);
    if changed {
        func.insts[term.index()].op = Op::Brif(cond, t_b, t_a, e_b, e_a);
    }
    changed
}

fn thread_br_table(func: &mut Function, cx: &ThreadCx<'_>, pred: Block, disp: Block) -> bool {
    let Some(&term) = func.blocks[pred.index()].insts.last() else {
        return false;
    };
    let Op::BrTable(index, default, ref targets) = func.insts[term.index()].op else {
        return false;
    };
    if !targets.contains(&disp) {
        return false;
    }
    let mut new_targets = targets.clone();
    let mut changed = false;
    for (k, target) in new_targets.iter_mut().enumerate() {
        if *target != disp {
            continue;
        }
        let Some(eval) = eval_dispatcher(func, cx, disp, &[], EdgeSeed::Const(index, k as i64)) else {
            continue;
        };
        if eval.chosen_args.is_empty() && non_const_needed(&eval) == 0 && eval.chosen_target != *target {
            *target = eval.chosen_target;
            changed = true;
        }
    }
    if changed {
        func.insts[term.index()].op = Op::BrTable(index, default, new_targets);
    }
    changed
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
    let mut any = false;
    for _ in 0..8 {
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
        let cfg = func.cfg();
        let single: Vec<Option<Block>> = (0..func.blocks.len()).map(|b| cfg.single_pred(Block(b as u32))).collect();
        let ranges = Ranges::new(func, (&defs, &consts), (&cfg.rpo, &single, &cfg.idom));
        let ranges_len = func.types.len();
        let mut cx = ThreadCx { defs: &defs, consts: &mut consts, ranges: &ranges, ranges_len, pinned: &pinned };
        let mut changed = false;
        for disp_idx in 0..func.blocks.len() {
            if cx.pinned[disp_idx] {
                continue;
            }
            let disp = Block(disp_idx as u32);
            let block = &func.blocks[disp_idx];
            if !(1..=MAX_DISP_INSTS).contains(&block.insts.len()) {
                continue;
            }
            let Some(&last) = block.insts.last() else {
                continue;
            };
            if !matches!(func.insts[last.index()].op, Op::Brif(..) | Op::BrTable(..) | Op::Jump(..))
                || (matches!(func.insts[last.index()].op, Op::Jump(..)) && block.insts.len() > 5)
            {
                continue;
            }
            for pred_idx in 0..func.blocks.len() {
                if pred_idx == disp_idx {
                    continue;
                }
                let pred = Block(pred_idx as u32);
                changed |= thread_jump(func, &mut cx, pred, disp);
                changed |= thread_brif(func, &mut cx, pred, disp);
                changed |= thread_br_table(func, &cx, pred, disp);
            }
        }
        if !changed {
            break;
        }
        func.bump_cfg();
        any = true;
    }
    let widened = widen::run(func);
    if any || widened {
        crate::fold::prune(func, entry);
        Domain::CFG | Domain::VALUES
    } else {
        Domain::NONE
    }
}

/// Renumbers the instructions and values the blocks hold and drops the rest, so a body kept or copied whole carries no dead entries.
pub(crate) fn compact(func: &mut Function) {
    if func.coroutine.is_some() {
        return;
    }
    let mut map: Vec<Option<Value>> = vec![None; func.types.len()];
    let mut types = Vec::new();
    let mut take = |value: Value| -> Value {
        *map[value.index()].get_or_insert_with(|| {
            types.push(func.types[value.index()]);
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

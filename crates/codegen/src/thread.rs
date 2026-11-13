use crate::fold::{evaluate_wrapped, extend};
use crate::ir::{Binary, Block, Convert, Function, Inst, InstData, IntCC, Op, SigRef, Type, Value};

const MAX_DISP_INSTS: usize = 10;
const MAX_CHAIN: usize = 8;
const MAX_BRIF_SPEC_OPS: usize = 2;

fn lookup(map: &[(Value, Value)], value: Value) -> Value {
    map.iter().find(|&&(k, _)| k == value).map_or(value, |&(_, v)| v)
}

fn lookup_const(local: &[(Value, i64)], global: &[Option<i64>], value: Value) -> Option<i64> {
    local.iter().find(|&&(k, _)| k == value).map(|&(_, imm)| imm).or_else(|| global.get(value.index()).copied().flatten())
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
            IntCC::Equal
                | IntCC::SignedGreaterThanOrEqual
                | IntCC::SignedLessThanOrEqual
                | IntCC::UnsignedGreaterThanOrEqual
                | IntCC::UnsignedLessThanOrEqual
        ))),
        _ => None,
    }
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

fn step_block(
    func: &Function,
    cur: Block,
    cur_args: &[Value],
    consts: &[Option<i64>],
    (map, local_ops, local_consts): (&mut Vec<(Value, Value)>, &mut Vec<(Value, Op)>, &mut Vec<(Value, i64)>),
) -> Option<(Block, Vec<Value>, Vec<Value>)> {
    let block = &func.blocks[cur.index()];
    if cur_args.len() != block.params.len() || !(1..=MAX_DISP_INSTS).contains(&block.insts.len()) {
        return None;
    }
    let (&last, body) = block.insts.split_last()?;
    if !matches!(func.insts[last.index()].op, Op::Brif(..) | Op::Jump(..)) {
        return None;
    }
    let mut defs_here: Vec<Value> = block.params.clone();
    for (&p, &a) in block.params.iter().zip(cur_args) {
        map.retain(|&(k, _)| k != p);
        map.push((p, a));
    }
    for &inst in body {
        let data = &func.insts[inst.index()];
        let res = data.result?;
        if !data.op.is_speculable() {
            return None;
        }
        let mut op = data.op.clone();
        op.map_operands(|v| lookup(map, v));
        if let Some(imm) = eval_op(func, &op, func.ty(res), &|v| lookup_const(local_consts, consts, v)) {
            local_consts.push((res, imm));
        }
        local_ops.push((res, op));
        defs_here.push(res);
        map.retain(|&(k, _)| k != res);
    }
    let (next_b, raw_args) = match &func.insts[last.index()].op {
        Op::Brif(cond, then_b, then_args, else_b, else_args) => {
            let cond_val = lookup(map, *cond);
            let c = lookup_const(local_consts, consts, cond_val)?;
            if extend(c, func.ty(cond_val), false) != 0 { (*then_b, then_args) } else { (*else_b, else_args) }
        }
        Op::Jump(target, args) => (*target, args),
        _ => return None,
    };
    if next_b == cur {
        return None;
    }
    let next_args: Vec<Value> = raw_args.iter().map(|&v| lookup(map, v)).collect();
    Some((next_b, next_args, defs_here))
}

fn eval_dispatcher(
    func: &Function,
    disp: Block,
    incoming: &[Value],
    consts: &[Option<i64>],
    pinned: &[bool],
) -> Option<Eval> {
    if func.blocks[disp.index()].params.is_empty() {
        return None;
    }
    let mut map: Vec<(Value, Value)> = Vec::new();
    let mut local_ops: Vec<(Value, Op)> = Vec::new();
    let mut local_consts: Vec<(Value, i64)> = Vec::new();
    let mut visited: Vec<(Block, Vec<Value>)> = Vec::new();
    let (mut next_b, mut next_args, first_defs) = step_block(func, disp, incoming, consts, (&mut map, &mut local_ops, &mut local_consts))?;
    visited.push((disp, first_defs));
    while visited.len() < MAX_CHAIN && !pinned[next_b.index()] && !visited.iter().any(|&(b, _)| b == next_b) {
        let marks = (map.len(), local_ops.len(), local_consts.len());
        let cur = next_b;
        if let Some((nb, na, defs)) = step_block(func, cur, &next_args, consts, (&mut map, &mut local_ops, &mut local_consts)) {
            visited.push((cur, defs));
            next_b = nb;
            next_args = na;
        } else {
            map.truncate(marks.0);
            local_ops.truncate(marks.1);
            local_consts.truncate(marks.2);
            break;
        }
    }
    let (chosen_target, chosen_args) = (next_b, next_args);

    for (def_block, defs) in &visited {
        if escapes_from(func, chosen_target, *def_block, defs) {
            return None;
        }
    }

    let mut needed = vec![false; local_ops.len()];
    for &arg in &chosen_args {
        if let Some(idx) = local_ops.iter().position(|&(r, _)| r == arg) {
            needed[idx] = true;
        }
    }
    for i in (0..local_ops.len()).rev() {
        if !needed[i] || local_consts.iter().any(|&(r, _)| r == local_ops[i].0) {
            continue;
        }
        let (_, ref op) = local_ops[i];
        let mut refs = Vec::new();
        op.for_operands(|v| refs.push(v));
        for v in refs {
            if let Some(idx) = local_ops.iter().position(|&(r, _)| r == v) {
                needed[idx] = true;
            }
        }
    }
    Some(Eval { chosen_target, chosen_args, local_ops, local_consts, needed })
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

fn thread_jump(func: &mut Function, pred: Block, disp: Block, consts: &mut Vec<Option<i64>>, pinned: &[bool]) -> bool {
    let Some(&term) = func.blocks[pred.index()].insts.last() else {
        return false;
    };
    let Op::Jump(target, ref args) = func.insts[term.index()].op else {
        return false;
    };
    if target != disp {
        return false;
    }
    let Some(eval) = eval_dispatcher(func, disp, args, consts, pinned) else {
        return false;
    };
    let (new_target, new_args) = emit_needed(func, pred, consts, eval);
    func.insts[term.index()].op = Op::Jump(new_target, new_args);
    true
}

fn thread_brif(func: &mut Function, pred: Block, disp: Block, consts: &mut Vec<Option<i64>>, pinned: &[bool]) -> bool {
    let Some(&term) = func.blocks[pred.index()].insts.last() else {
        return false;
    };
    let Op::Brif(cond, then_b, then_args, else_b, else_args) = func.insts[term.index()].op.clone() else {
        return false;
    };
    let mut changed = false;
    let mut rewrite = |target: &mut Block, args: &mut Vec<Value>| {
        if *target != disp {
            return;
        }
        let Some(eval) = eval_dispatcher(func, disp, args, consts, pinned) else {
            return;
        };
        if non_const_needed(&eval) > MAX_BRIF_SPEC_OPS {
            return;
        }
        (*target, *args) = emit_needed(func, pred, consts, eval);
        changed = true;
    };
    let (mut t_b, mut t_a, mut e_b, mut e_a) = (then_b, then_args, else_b, else_args);
    rewrite(&mut t_b, &mut t_a);
    rewrite(&mut e_b, &mut e_a);
    if changed {
        func.insts[term.index()].op = Op::Brif(cond, t_b, t_a, e_b, e_a);
    }
    changed
}

/// Threads correlated constant-condition jump chains through small pure dispatcher blocks.
pub(crate) fn run(func: &mut Function) -> bool {
    let Some(entry) = func.entry else {
        return false;
    };
    if func.coroutine.is_some() {
        return false;
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
        let mut changed = false;
        for disp_idx in 0..func.blocks.len() {
            if pinned[disp_idx] {
                continue;
            }
            let disp = Block(disp_idx as u32);
            let block = &func.blocks[disp_idx];
            if block.params.is_empty() || !(1..=MAX_DISP_INSTS).contains(&block.insts.len()) {
                continue;
            }
            let Some(&last) = block.insts.last() else {
                continue;
            };
            if !matches!(func.insts[last.index()].op, Op::Brif(..)) {
                continue;
            }
            for pred_idx in 0..func.blocks.len() {
                if pred_idx == disp_idx {
                    continue;
                }
                let pred = Block(pred_idx as u32);
                changed |= thread_jump(func, pred, disp, &mut consts, &pinned);
                changed |= thread_brif(func, pred, disp, &mut consts, &pinned);
            }
        }
        if !changed {
            break;
        }
        any = true;
    }
    if any {
        crate::fold::prune(func, entry);
    }
    any
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

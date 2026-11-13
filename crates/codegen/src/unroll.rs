use crate::cfg::{Cfg, Loop};
use crate::fold::{evaluate_wrapped, extend};
use crate::ir::{Binary, Block, BlockData, Function, Inst, IntCC, Op, Type, Value};

fn remap_val(map: &[(Value, Value)], v: Value) -> Value {
    map.iter().rev().find(|&&(old, _)| old == v).map_or(v, |&(_, new)| new)
}

fn has_bad_non_term(func: &Function, block_idx: usize) -> bool {
    let insts = &func.blocks[block_idx].insts;
    insts[..insts.len() - 1]
        .iter()
        .any(|&i| matches!(func.insts[i.index()].op, Op::Call { .. } | Op::TryCall { .. } | Op::BrTable(..) | Op::Jump(..) | Op::Brif(..)))
}

fn eval_block(func: &Function, block_idx: usize, params: &[Option<i64>], env: &mut [Option<i64>]) {
    let block = &func.blocks[block_idx];
    for (p, val) in block.params.iter().zip(params) {
        env[p.index()] = *val;
    }
    for &inst in &block.insts[..block.insts.len() - 1] {
        let data = &func.insts[inst.index()];
        let Some(res) = data.result else { continue };
        let ty = func.ty(res);
        env[res.index()] = match data.op {
            Op::Select(c, x, y) if ty.is_small_int() => env[c.index()].and_then(|cv| {
                let chosen = if cv != 0 { x } else { y };
                env[chosen.index()].map(|imm| extend(imm, ty, true))
            }),
            _ => evaluate_wrapped(func, &data.op, ty, &|v| env[v.index()]),
        };
    }
}

fn clone_body(func: &mut Function, src_non_term: &[Inst], dst: Block, map: &mut Vec<(Value, Value)>) {
    for &orig_inst in src_non_term {
        let mut op = func.insts[orig_inst.index()].op.clone();
        op.map_operands(|v| remap_val(map, v));
        let new_res = func.insts[orig_inst.index()].result.map(|r| {
            let nr = func.value(func.ty(r));
            map.push((r, nr));
            nr
        });
        let ni = func.add(op, new_res);
        func.blocks[dst.index()].insts.push(ni);
    }
}

fn remap_other_blocks(func: &mut Function, skip_a: usize, skip_b: Option<usize>, map: &[(Value, Value)]) {
    if map.is_empty() {
        return;
    }
    for ob in 0..func.blocks.len() {
        if ob == skip_a || skip_b == Some(ob) {
            continue;
        }
        for idx in 0..func.blocks[ob].insts.len() {
            let inst = func.blocks[ob].insts[idx];
            func.insts[inst.index()].op.map_operands(|v| remap_val(map, v));
        }
    }
}

fn pop_terminator(func: &mut Function, blk_idx: usize) {
    if let Some(popped) = func.blocks[blk_idx].insts.pop() {
        func.insts[popped.index()].op = Op::Trap;
    }
}

fn retire_block(func: &mut Function, blk_idx: usize) {
    for inst in std::mem::take(&mut func.blocks[blk_idx].insts) {
        if func.insts[inst.index()].op.is_terminator() {
            func.insts[inst.index()].op = Op::Trap;
        }
    }
    func.blocks[blk_idx].params.clear();
}

fn single_outside_pred(func: &Function, head: Block, back_pred: Block) -> Option<Vec<Value>> {
    let mut found = None;
    for (p, block) in func.blocks.iter().enumerate() {
        if p == head.index() || p == back_pred.index() || !func.successors(Block(p as u32)).any(|t| t == head) {
            continue;
        }
        if found.is_some() {
            return None;
        }
        let Some(&last) = block.insts.last() else { return None };
        let args = match &func.insts[last.index()].op {
            Op::Jump(target, args) if *target == head => args.clone(),
            Op::Brif(_, t_blk, t_args, f_blk, _) if *t_blk == head && *f_blk != head => t_args.clone(),
            Op::Brif(_, t_blk, _, f_blk, f_args) if *f_blk == head && *t_blk != head => f_args.clone(),
            _ => return None,
        };
        found = Some(args);
    }
    found
}

fn unroll_self(
    func: &mut Function,
    cfg: &Cfg,
    b: usize,
    cond: Value,
    back_on_true: bool,
    back_args: Vec<Value>,
    exit_blk: Block,
    exit_args: Vec<Value>,
    global_consts: &[Option<i64>],
) -> bool {
    let head = Block(b as u32);
    let inst_count = func.blocks[b].insts.len();
    if inst_count > 48 {
        return false;
    }
    let Some(init_args) = single_outside_pred(func, head, head) else { return false };
    if init_args.len() != func.blocks[b].params.len() || back_args.len() != func.blocks[b].params.len() {
        return false;
    }
    let mut cur_params: Vec<Option<i64>> = init_args.iter().map(|v| global_consts[v.index()]).collect();
    let mut proven = None;
    for step in 0..=16_usize {
        let mut env = global_consts.to_vec();
        eval_block(func, b, &cur_params, &mut env);
        let Some(cond_val) = env[cond.index()] else { break };
        if (cond_val != 0) == back_on_true {
            cur_params = back_args.iter().map(|v| env[v.index()]).collect();
        } else {
            proven = Some(step);
            break;
        }
    }
    let Some(back_steps) = proven else {
        return unroll_self_dynamic(func, cfg, b, cond, back_on_true, &back_args, exit_blk, global_consts);
    };
    let last = *func.blocks[b].insts.last().unwrap_or(&Inst(0));
    if back_steps == 0 {
        func.insts[last.index()].op = Op::Jump(exit_blk, exit_args);
        return true;
    }
    if (back_steps + 1) * inst_count > 192 {
        return false;
    }
    let body_insts = func.blocks[b].insts[..inst_count - 1].to_vec();
    pop_terminator(func, b);
    let mut cur_back_args = back_args.clone();
    let mut last_map = Vec::new();
    for _ in 0..back_steps {
        let mut map: Vec<(Value, Value)> = func.blocks[b].params.iter().copied().zip(cur_back_args.iter().copied()).collect();
        clone_body(func, &body_insts, head, &mut map);
        cur_back_args = back_args.iter().map(|&v| remap_val(&map, v)).collect();
        last_map = map;
    }
    let final_exit_args: Vec<Value> = exit_args.iter().map(|&v| remap_val(&last_map, v)).collect();
    let jump = func.add(Op::Jump(exit_blk, final_exit_args), None);
    func.blocks[b].insts.push(jump);
    remap_other_blocks(func, b, None, &last_map);
    true
}

fn match_iv_step(func: &Function, b: usize, body_insts: &[Inst], back_args: &[Value], iv_next: Value) -> Option<(usize, Value)> {
    let iv_inst = body_insts.iter().copied().find(|&i| func.insts[i.index()].result == Some(iv_next))?;
    let Op::Binary(Binary::Iadd, p, q) = func.insts[iv_inst.index()].op else { return None };
    for (cand_iv, cand_stride) in [(p, q), (q, p)] {
        if let Some(idx) = func.blocks[b].params.iter().position(|&param| param == cand_iv)
            && back_args[idx] == iv_next
        {
            return Some((idx, cand_stride));
        }
    }
    None
}

fn unroll_self_dynamic(
    func: &mut Function,
    cfg: &Cfg,
    b: usize,
    cond: Value,
    back_on_true: bool,
    back_args: &[Value],
    exit_blk: Block,
    global_consts: &[Option<i64>],
) -> bool {
    let head = Block(b as u32);
    let inst_count = func.blocks[b].insts.len();
    if inst_count > 36
        || func.blocks[b].params.len() > 8
        || back_args.len() != func.blocks[b].params.len()
        || func.successors(exit_blk).any(|t| t == exit_blk)
    {
        return false;
    }
    let body_insts = func.blocks[b].insts[..inst_count - 1].to_vec();
    if body_insts.iter().any(|&i| matches!(func.insts[i.index()].op, Op::Trap | Op::Trapz(..) | Op::AtomicLoad(..) | Op::AtomicStore(..) | Op::AtomicRmw(..) | Op::AtomicCas(..) | Op::Fence)) {
        return false;
    }
    let mut head_defs = vec![false; func.types.len()];
    for &p in &func.blocks[b].params {
        head_defs[p.index()] = true;
    }
    for &i in &body_insts {
        if let Some(r) = func.insts[i.index()].result {
            head_defs[r.index()] = true;
        }
    }
    let outside: Vec<usize> = (0..func.blocks.len())
        .filter(|&p| p != b && func.successors(Block(p as u32)).any(|t| t == head))
        .collect();
    if outside.len() != 1 {
        return false;
    }
    let out_blk = outside[0];
    if !func.blocks[out_blk].insts.last().is_some_and(|&last| matches!(func.insts[last.index()].op, Op::Jump(..) | Op::Brif(..))) {
        return false;
    }
    for ob in 0..func.blocks.len() {
        if ob == b {
            continue;
        }
        let mut escapes = false;
        for &inst in &func.blocks[ob].insts {
            func.insts[inst.index()].op.for_operands(|v| escapes |= head_defs[v.index()]);
        }
        if escapes && (ob == out_blk || !cfg.dominates(head, Block(ob as u32))) {
            return false;
        }
    }
    let cond_inst = body_insts.iter().copied().find(|&i| func.insts[i.index()].result == Some(cond));
    let Some(cond_inst) = cond_inst else { return false };
    let Op::Icmp(cc, lhs, rhs) = func.insts[cond_inst.index()].op else { return false };
    let eff_cc = if back_on_true { cc } else { cc.inverse() };
    let (cmp_cc, iv_cmp, limit) = match eff_cc {
        IntCC::UnsignedLessThan | IntCC::UnsignedLessThanOrEqual => (eff_cc, lhs, rhs),
        IntCC::UnsignedGreaterThan => (IntCC::UnsignedLessThan, rhs, lhs),
        IntCC::UnsignedGreaterThanOrEqual => (IntCC::UnsignedLessThanOrEqual, rhs, lhs),
        _ => return false,
    };
    if func.ty(iv_cmp) != Type::I64 || head_defs[limit.index()] {
        return false;
    }
    let mut matched = match_iv_step(func, b, &body_insts, back_args, iv_cmp)
        .filter(|&(_, stride)| !head_defs[stride.index()])
        .map(|(idx, stride)| (idx, stride, 0_i64));
    if matched.is_none()
        && let Some(cmp_add) = body_insts.iter().copied().find(|&i| func.insts[i.index()].result == Some(iv_cmp))
        && let Op::Binary(Binary::Iadd, p, q) = func.insts[cmp_add.index()].op
    {
        for (cand_next, cand_bias) in [(p, q), (q, p)] {
            if let Some(bias) = global_consts[cand_bias.index()]
                && (0..(1_i64 << 30)).contains(&bias)
                && let Some((idx, stride)) = match_iv_step(func, b, &body_insts, back_args, cand_next)
                && !head_defs[stride.index()]
            {
                matched = Some((idx, stride, bias));
                break;
            }
        }
    }
    let Some((iv_idx, stride, bias)) = matched else { return false };
    let factor = if inst_count <= 8 { 4_i64 } else { 2_i64 };
    let param_tys: Vec<Type> = func.blocks[b].params.iter().map(|&p| func.ty(p)).collect();
    let guard_params: Vec<Value> = param_tys.iter().map(|&ty| func.value(ty)).collect();
    let guard = Block(func.blocks.len() as u32);
    let unrolled = Block((func.blocks.len() + 1) as u32);
    let unrolled_params: Vec<Value> = param_tys.iter().map(|&ty| func.value(ty)).collect();
    func.blocks.push(BlockData { params: guard_params.clone(), insts: Vec::new() });
    func.blocks.push(BlockData { params: unrolled_params.clone(), insts: Vec::new() });

    let mut emit_guard = |op: Op, ty: Type| -> Value {
        let v = func.value(ty);
        let i = func.add(op, Some(v));
        func.blocks[guard.index()].insts.push(i);
        v
    };
    let zero = emit_guard(Op::Const(0), Type::I64);
    let sh61 = emit_guard(Op::Const(61), Type::I64);
    let cf = emit_guard(Op::Const(factor), Type::I64);
    let stride_pos = emit_guard(Op::Icmp(IntCC::UnsignedGreaterThan, stride, zero), Type::I8);
    let or_lim = emit_guard(Op::Binary(Binary::Bor, limit, stride), Type::I64);
    let hi_bits = emit_guard(Op::Binary(Binary::Ushr, or_lim, sh61), Type::I64);
    let no_wrap = emit_guard(Op::Icmp(IntCC::Equal, hi_bits, zero), Type::I8);
    let sf = emit_guard(Op::Binary(Binary::Imul, stride, cf), Type::I64);
    let s_need = if bias > 0 {
        let b_val = emit_guard(Op::Const(bias), Type::I64);
        emit_guard(Op::Binary(Binary::Iadd, sf, b_val), Type::I64)
    } else {
        sf
    };
    let lim_ok = emit_guard(Op::Icmp(IntCC::UnsignedLessThanOrEqual, s_need, limit), Type::I8);
    let safe_limit = emit_guard(Op::Binary(Binary::Isub, limit, s_need), Type::I64);
    let init_ok = emit_guard(Op::Icmp(cmp_cc, guard_params[iv_idx], safe_limit), Type::I8);
    let g1 = emit_guard(Op::Binary(Binary::Band, stride_pos, no_wrap), Type::I8);
    let g2 = emit_guard(Op::Binary(Binary::Band, lim_ok, init_ok), Type::I8);
    let guard_cond = emit_guard(Op::Binary(Binary::Band, g1, g2), Type::I8);
    let guard_br = func.add(Op::Brif(guard_cond, unrolled, guard_params.clone(), head, guard_params), None);
    func.blocks[guard.index()].insts.push(guard_br);

    let mut cur_args = unrolled_params;
    for _ in 0..factor {
        let mut map: Vec<(Value, Value)> = func.blocks[b].params.iter().copied().zip(cur_args.iter().copied()).collect();
        clone_body(func, &body_insts, unrolled, &mut map);
        cur_args = back_args.iter().map(|&v| remap_val(&map, v)).collect();
    }
    let loop_cond = func.value(Type::I8);
    let loop_icmp = func.add(Op::Icmp(cmp_cc, cur_args[iv_idx], safe_limit), Some(loop_cond));
    let loop_br = func.add(Op::Brif(loop_cond, unrolled, cur_args.clone(), head, cur_args), None);
    func.blocks[unrolled.index()].insts.push(loop_icmp);
    func.blocks[unrolled.index()].insts.push(loop_br);

    if let Some(&last) = func.blocks[out_blk].insts.last() {
        match &mut func.insts[last.index()].op {
            Op::Jump(target, _) if *target == head => *target = guard,
            Op::Brif(_, then, _, otherwise, _) => {
                if *then == head {
                    *then = guard;
                }
                if *otherwise == head {
                    *otherwise = guard;
                }
            }
            _ => {}
        }
    }
    true
}

fn unroll_two_block(
    func: &mut Function,
    cfg: &Cfg,
    each: &Loop,
    cond: Value,
    (t_blk, t_args, f_blk, f_args): (Block, Vec<Value>, Block, Vec<Value>),
    global_consts: &[Option<i64>],
) -> bool {
    let head = each.head;
    let b = head.index();
    let body_blk = each.latches[0];
    let (body_on_true, body_args, exit_blk, exit_args) = if t_blk == body_blk {
        (true, t_args, f_blk, f_args)
    } else if f_blk == body_blk {
        (false, f_args, t_blk, t_args)
    } else {
        return false;
    };
    if func.entry == Some(body_blk)
        || each.body.get(exit_blk.index()).copied().unwrap_or(false)
        || cfg.pred_count(body_blk) != 1
        || func.blocks[body_blk.index()].insts.is_empty()
        || func.blocks[b].insts.len() + func.blocks[body_blk.index()].insts.len() > 48
        || body_args.len() != func.blocks[body_blk.index()].params.len()
        || has_bad_non_term(func, body_blk.index())
    {
        return false;
    }
    let Some(&body_last) = func.blocks[body_blk.index()].insts.last() else { return false };
    let Op::Jump(back_target, back_args) = &func.insts[body_last.index()].op else { return false };
    if *back_target != head || back_args.len() != func.blocks[b].params.len() {
        return false;
    }
    let back_args = back_args.clone();
    let Some(init_args) = single_outside_pred(func, head, body_blk) else { return false };
    if init_args.len() != func.blocks[b].params.len() {
        return false;
    }
    let mut cur_head_params: Vec<Option<i64>> = init_args.iter().map(|v| global_consts[v.index()]).collect();
    let mut proven = None;
    for step in 0..=8_usize {
        let mut env = global_consts.to_vec();
        eval_block(func, b, &cur_head_params, &mut env);
        let Some(cond_val) = env[cond.index()] else { break };
        if (cond_val != 0) != body_on_true {
            proven = Some(step);
            break;
        }
        if step == 8 {
            break;
        }
        let body_params: Vec<Option<i64>> = body_args.iter().map(|v| env[v.index()]).collect();
        eval_block(func, body_blk.index(), &body_params, &mut env);
        cur_head_params = back_args.iter().map(|v| env[v.index()]).collect();
    }
    let Some(trips) = proven else { return false };
    let head_len = func.blocks[b].insts.len();
    let body_len = func.blocks[body_blk.index()].insts.len();
    if trips == 0 {
        let last = *func.blocks[b].insts.last().unwrap_or(&Inst(0));
        func.insts[last.index()].op = Op::Jump(exit_blk, exit_args);
        retire_block(func, body_blk.index());
        return true;
    }
    if head_len + trips * (head_len + body_len) > 192 {
        return false;
    }
    let head_non_term = func.blocks[b].insts[..head_len - 1].to_vec();
    let body_non_term = func.blocks[body_blk.index()].insts[..body_len - 1].to_vec();
    pop_terminator(func, b);
    let mut cur_body_args = body_args.clone();
    let mut head_map: Vec<(Value, Value)> = Vec::new();
    for _ in 0..trips {
        let mut body_map = head_map.clone();
        for (bp, ca) in func.blocks[body_blk.index()].params.iter().copied().zip(cur_body_args.iter().copied()) {
            body_map.push((bp, ca));
        }
        clone_body(func, &body_non_term, head, &mut body_map);
        let cur_back_args: Vec<Value> = back_args.iter().map(|&v| remap_val(&body_map, v)).collect();
        let mut next_head_map: Vec<(Value, Value)> = func.blocks[b].params.iter().copied().zip(cur_back_args).collect();
        clone_body(func, &head_non_term, head, &mut next_head_map);
        head_map = next_head_map;
        cur_body_args = body_args.iter().map(|&v| remap_val(&head_map, v)).collect();
    }
    let final_exit_args: Vec<Value> = exit_args.iter().map(|&v| remap_val(&head_map, v)).collect();
    let jump = func.add(Op::Jump(exit_blk, final_exit_args), None);
    func.blocks[b].insts.push(jump);
    remap_other_blocks(func, b, Some(body_blk.index()), &head_map);
    retire_block(func, body_blk.index());
    true
}

/// Unrolls 1-block and 2-block natural loops with a statically provable constant trip count (`0..=16` / `0..=8`) or dynamic counted loops.
pub(crate) fn run(func: &mut Function) -> bool {
    let Some(entry) = func.entry else { return false };
    let mut cfg = Cfg::new(func);
    let loops = cfg.loops(func);
    let mut global_consts = vec![None; func.types.len()];
    for &blk in &cfg.rpo {
        for &inst in &func.blocks[blk.index()].insts {
            let data = &func.insts[inst.index()];
            let Some(res) = data.result else { continue };
            if let Some(imm) = evaluate_wrapped(func, &data.op, func.ty(res), &|v| global_consts[v.index()]) {
                global_consts[res.index()] = Some(imm);
            }
        }
    }
    let mut changed = false;
    let mut dirty = false;
    for each in &loops.list {
        let head = each.head;
        let b = head.index();
        if head == entry || each.size > 2 || each.latches.len() != 1 || func.blocks[b].insts.is_empty() || has_bad_non_term(func, b) {
            continue;
        }
        if dirty {
            cfg = Cfg::new(func);
            dirty = false;
        }
        if cfg.pred_count(head) != 2 {
            continue;
        }
        let Some(&last) = func.blocks[b].insts.last() else { continue };
        let Op::Brif(cond, t_blk, t_args, f_blk, f_args) = &func.insts[last.index()].op else { continue };
        if t_blk == f_blk {
            continue;
        }
        let (cond, t_blk, t_args, f_blk, f_args) = (*cond, *t_blk, t_args.clone(), *f_blk, f_args.clone());
        global_consts.resize(func.types.len(), None);
        let did = if each.size == 1 && each.latches[0] == head {
            if t_blk == head {
                unroll_self(func, &cfg, b, cond, true, t_args, f_blk, f_args, &global_consts)
            } else if f_blk == head {
                unroll_self(func, &cfg, b, cond, false, f_args, t_blk, t_args, &global_consts)
            } else {
                false
            }
        } else if each.size == 2 && each.latches[0] != head {
            unroll_two_block(func, &cfg, each, cond, (t_blk, t_args, f_blk, f_args), &global_consts)
        } else {
            false
        };
        changed |= did;
        dirty |= did;
    }
    changed
}

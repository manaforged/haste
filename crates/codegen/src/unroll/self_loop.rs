use super::count::{
    counted, emit_dynamic_guard, guarded_trapz, head_defs, outside_preds, stride_const,
};
use super::{
    MAX_CHAIN_TRIPS, MAX_CHAIN_UNROLLED_INSTS, MAX_FP_FULL_INSTS, MAX_FP_FULL_TRIPS,
    MAX_FULL_INSTS, MAX_FULL_TRIPS, MAX_VEC_FULL_TRIPS, block_has_float, block_has_libcall_mem,
    block_has_vec, clone_body, eval_block, is_libcall_mem, outside_const_params, pop_terminator,
    remap_other_blocks, remap_val,
};
use crate::cfg::Cfg;
use crate::fold::evaluate_wrapped;
use crate::ir::{Binary, Block, BlockData, Function, Inst, Op, Type, Value};

fn self_unroll_fits(func: &Function, b: usize, inst_count: usize, trips: usize) -> bool {
    if trips > 4 && block_has_libcall_mem(func, b) {
        return false;
    }
    let body_count = inst_count.saturating_sub(1);
    if block_has_float(func, b) {
        return trips <= MAX_FP_FULL_TRIPS && trips * body_count <= MAX_FP_FULL_INSTS;
    }
    if block_has_vec(func, b) && trips > MAX_VEC_FULL_TRIPS {
        return false;
    }
    let fits_legacy = inst_count <= 64 && trips * inst_count <= MAX_FULL_INSTS;
    fits_legacy || (trips <= MAX_CHAIN_TRIPS && trips * body_count <= MAX_CHAIN_UNROLLED_INSTS)
}

pub(super) fn unroll_self(
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
    if inst_count > MAX_CHAIN_UNROLLED_INSTS + 1 {
        return false;
    }
    let Some(mut cur_params) = outside_const_params(func, cfg, head, &[head], global_consts) else {
        return unroll_self_dynamic(
            func,
            cfg,
            (b, cond, back_on_true),
            (&back_args, exit_blk),
            global_consts,
        );
    };
    if back_args.len() != func.blocks[b].params.len() {
        return false;
    }
    let mut proven = None;
    for step in 0..=MAX_FULL_TRIPS {
        let mut env = global_consts.to_vec();
        eval_block(func, b, &cur_params, &mut env, evaluate_wrapped);
        let Some(cond_val) = env[cond.index()] else {
            break;
        };
        if (cond_val != 0) == back_on_true {
            cur_params = back_args.iter().map(|v| env[v.index()]).collect();
        } else {
            proven = Some(step);
            break;
        }
    }
    let Some(back_steps) = proven else {
        return unroll_self_dynamic(
            func,
            cfg,
            (b, cond, back_on_true),
            (&back_args, exit_blk),
            global_consts,
        );
    };
    let last = *func.blocks[b].insts.last().unwrap_or(&Inst(0));
    if back_steps == 0 {
        func.insts[last.index()].op = Op::Jump(exit_blk, exit_args);
        return true;
    }
    let trips = back_steps + 1;
    let partial = (!self_unroll_fits(func, b, inst_count, trips)).then(|| {
        let body = inst_count.saturating_sub(1);
        let ok = inst_count <= 64
            && !block_has_float(func, b)
            && !block_has_vec(func, b)
            && !block_has_libcall_mem(func, b);
        ok.then(|| {
            [8_usize, 4, 2].into_iter().find(|&f| {
                trips > f && trips.is_multiple_of(f) && f * body <= MAX_CHAIN_UNROLLED_INSTS
            })
        })
        .flatten()
    });
    if matches!(partial, Some(None)) {
        return unroll_self_dynamic(
            func,
            cfg,
            (b, cond, back_on_true),
            (&back_args, exit_blk),
            global_consts,
        );
    }
    let steps = partial.flatten().map_or(back_steps, |f| f - 1);
    let body_insts = func.blocks[b].insts[..inst_count - 1].to_vec();
    pop_terminator(func, b);
    let mut cur_back_args = back_args.clone();
    let mut last_map = Vec::new();
    for _ in 0..steps {
        let mut map: Vec<(Value, Value)> = func.blocks[b]
            .params
            .iter()
            .copied()
            .zip(cur_back_args.iter().copied())
            .collect();
        clone_body(func, &body_insts, head, &mut map);
        cur_back_args = back_args.iter().map(|&v| remap_val(&map, v)).collect();
        last_map = map;
    }
    let final_exit_args: Vec<Value> = exit_args.iter().map(|&v| remap_val(&last_map, v)).collect();
    let term = match (partial.is_some(), back_on_true) {
        (true, true) => Op::Brif(
            remap_val(&last_map, cond),
            head,
            cur_back_args,
            exit_blk,
            final_exit_args,
        ),
        (true, false) => Op::Brif(
            remap_val(&last_map, cond),
            exit_blk,
            final_exit_args,
            head,
            cur_back_args,
        ),
        (false, _) => Op::Jump(exit_blk, final_exit_args),
    };
    let jump = func.add(term, None);
    func.blocks[b].insts.push(jump);
    remap_other_blocks(func, &[b], &last_map);
    true
}

fn blocks_dynamic_unroll(op: &Op) -> bool {
    is_libcall_mem(op) || matches!(*op, Op::Trap) || crate::mem::is_atomic_barrier(op)
}

fn head_defs_dominated(
    func: &Function,
    cfg: &Cfg,
    (b, out_blk): (usize, usize),
    head_defs: &[bool],
) -> bool {
    let head = Block(b as u32);
    for ob in 0..func.blocks.len() {
        if ob == b {
            continue;
        }
        let mut escapes = false;
        for &inst in &func.blocks[ob].insts {
            func.insts[inst.index()]
                .op
                .for_operands(|v| escapes |= head_defs[v.index()]);
        }
        if escapes && (ob == out_blk || !cfg.dominates(head, Block(ob as u32))) {
            return false;
        }
    }
    true
}

fn retarget_outside_pred(func: &mut Function, out_blk: usize, head: Block, guard: Block) {
    let Some(&last) = func.blocks[out_blk].insts.last() else {
        return;
    };
    func.insts[last.index()]
        .op
        .map_targets(|b| if b == head { guard } else { b });
}

fn peel_jumps(func: &Function, mut blk: Block) -> Block {
    for _ in 0..4 {
        if func.blocks[blk.index()].insts.len() != 1 {
            break;
        }
        match func.last(blk) {
            Some(Op::Jump(next, _)) if *next != blk => blk = *next,
            _ => break,
        }
    }
    blk
}

fn is_preloop_for_remainder(
    func: &Function,
    cfg: &Cfg,
    head: Block,
    out_blk: usize,
    exit_blk: Block,
) -> bool {
    let rem = peel_jumps(func, exit_blk);
    if rem == head
        || !cfg
            .loops()
            .list
            .iter()
            .any(|lp| lp.head == rem && !lp.body.get(head.index()).copied().unwrap_or(false))
    {
        return false;
    }
    match func.last(Block(out_blk as u32)) {
        Some(Op::Brif(_, t, _, f, _)) => peel_jumps(func, if *t == head { *f } else { *t }) == rem,
        _ => false,
    }
}

fn unrolled_skips(
    func: &Function,
    (body_insts, trap_insts): (&[Inst], &[Inst]),
    back_args: &[Value],
    replaced_iv: Option<Value>,
) -> Vec<bool> {
    let mut needed = vec![false; func.types.len()];
    for &v in back_args {
        needed[v.index()] = true;
    }
    let mut skip = vec![false; body_insts.len()];
    for (idx, &inst) in body_insts.iter().enumerate().rev() {
        if trap_insts.contains(&inst) {
            skip[idx] = true;
            continue;
        }
        let data = &func.insts[inst.index()];
        if replaced_iv.is_some() && data.result == replaced_iv {
            continue;
        }
        if data.op.is_speculable() && data.result.is_some_and(|r| !needed[r.index()]) {
            skip[idx] = true;
            continue;
        }
        data.op.for_operands(|v| needed[v.index()] = true);
    }
    skip
}

fn unroll_self_dynamic(
    func: &mut Function,
    cfg: &Cfg,
    (b, cond, back_on_true): (usize, Value, bool),
    (back_args, exit_blk): (&[Value], Block),
    global_consts: &[Option<i64>],
) -> bool {
    let head = Block(b as u32);
    let inst_count = func.blocks[b].insts.len();
    if inst_count > 48
        || func.blocks[b].params.len() > 8
        || back_args.len() != func.blocks[b].params.len()
        || func.successors(exit_blk).any(|t| t == exit_blk)
    {
        return false;
    }
    let body_insts = func.blocks[b].insts[..inst_count - 1].to_vec();
    if body_insts
        .iter()
        .any(|&i| blocks_dynamic_unroll(&func.insts[i.index()].op))
    {
        return false;
    }
    let head_defs = head_defs(func, b, &body_insts);
    let outside = outside_preds(func, b);
    let [out_blk] = outside.as_slice() else {
        return false;
    };
    let out_blk = *out_blk;
    if is_preloop_for_remainder(func, cfg, head, out_blk, exit_blk)
        || !func.blocks[out_blk]
            .insts
            .last()
            .is_some_and(|&last| matches!(func.insts[last.index()].op, Op::Jump(..) | Op::Brif(..)))
        || !head_defs_dominated(func, cfg, (b, out_blk), &head_defs)
    {
        return false;
    }
    let Some(counted_info) = counted(
        func,
        b,
        &body_insts,
        (cond, back_on_true),
        back_args,
        &head_defs,
        global_consts,
    ) else {
        return false;
    };
    let iv_param = func.blocks[b].params[counted_info.iv_idx];
    let Some((trap_insts, trap_lens)) =
        guarded_trapz(func, &body_insts, iv_param, counted_info.iv_ty, &head_defs)
    else {
        return false;
    };
    let const_stride = stride_const(global_consts, counted_info.stride);
    let factor = if inst_count.saturating_sub(trap_insts.len() * 2) <= 16 {
        4_i64
    } else {
        2_i64
    };
    let param_tys: Vec<Type> = func.blocks[b].params.iter().map(|&p| func.ty(p)).collect();
    let guard_params: Vec<Value> = param_tys.iter().map(|&ty| func.value(ty)).collect();
    let (guard, unrolled) = (
        Block(func.blocks.len() as u32),
        Block((func.blocks.len() + 1) as u32),
    );
    let unrolled_params: Vec<Value> = param_tys.iter().map(|&ty| func.value(ty)).collect();
    func.blocks.push(BlockData {
        params: guard_params.clone(),
        insts: Vec::new(),
    });
    func.blocks.push(BlockData {
        params: unrolled_params.clone(),
        insts: Vec::new(),
    });
    let iv_back = back_args[counted_info.iv_idx];
    let parallel_strides = body_insts
        .iter()
        .any(|&i| {
            let data = &func.insts[i.index()];
            data.result == Some(iv_back)
                && matches!(
                    data.op,
                    Op::Binary(Binary::Iadd, a, b)
                        if (a == iv_param && b == counted_info.stride)
                            || (a == counted_info.stride && b == iv_param)
                )
        })
        .then(|| {
            if let Some(c) = const_stride {
                (1..=(factor as usize))
                    .map(|k| {
                        let v = func.value(counted_info.iv_ty);
                        let imm = func.add(Op::Const(c * k as i64), Some(v));
                        func.blocks[guard.index()].insts.push(imm);
                        v
                    })
                    .collect()
            } else {
                let mut strides = vec![counted_info.stride];
                for k in 2..=(factor as usize) {
                    let prev = strides[k - 2];
                    let next = func.value(counted_info.iv_ty);
                    let add = func.add(
                        Op::Binary(Binary::Iadd, prev, counted_info.stride),
                        Some(next),
                    );
                    func.blocks[guard.index()].insts.push(add);
                    strides.push(next);
                }
                strides
            }
        });
    let pre_sf = parallel_strides.as_ref().map(|s| s[factor as usize - 1]);
    let safe_limit = emit_dynamic_guard(
        func,
        guard,
        &counted_info,
        (const_stride, factor, pre_sf),
        (head, unrolled, guard_params),
        &trap_lens,
    );
    let skip_unrolled = unrolled_skips(
        func,
        (&body_insts, &trap_insts),
        back_args,
        parallel_strides.is_some().then_some(iv_back),
    );
    let iv_base = unrolled_params[counted_info.iv_idx];
    let mut cur_args = unrolled_params;
    for step in 0..(factor as usize) {
        let mut map: Vec<(Value, Value)> = func.blocks[b]
            .params
            .iter()
            .copied()
            .zip(cur_args.iter().copied())
            .collect();
        for (idx, &orig_inst) in body_insts.iter().enumerate() {
            if skip_unrolled[idx] {
                continue;
            }
            if let Some(strides) = &parallel_strides {
                let data = &func.insts[orig_inst.index()];
                if data.result == Some(iv_back) {
                    let nr = func.value_like(iv_back);
                    map.push((iv_back, nr));
                    let ni = func.add(Op::Binary(Binary::Iadd, iv_base, strides[step]), Some(nr));
                    func.blocks[unrolled.index()].insts.push(ni);
                    continue;
                }
            }
            clone_body(func, &[orig_inst], unrolled, &mut map);
        }
        cur_args = back_args.iter().map(|&v| remap_val(&map, v)).collect();
    }
    let loop_cond = func.value(Type::I8);
    let loop_icmp = func.add(
        Op::Icmp(
            counted_info.cmp_cc,
            cur_args[counted_info.iv_idx],
            safe_limit,
        ),
        Some(loop_cond),
    );
    let loop_br = func.add(
        Op::Brif(loop_cond, unrolled, cur_args.clone(), head, cur_args),
        None,
    );
    func.blocks[unrolled.index()]
        .insts
        .extend([loop_icmp, loop_br]);
    retarget_outside_pred(func, out_blk, head, guard);
    true
}

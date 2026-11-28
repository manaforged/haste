use super::count::{counted, emit_dynamic_guard, stride_const};
use super::{clone_body, has_bad_non_term, is_libcall_mem, remap_val};
use crate::cfg::{Cfg, Loop};
use crate::ir::{Binary, Block, BlockData, Function, Op, Type, Value};

const MAX_BLOCKS: usize = 4;
const MAX_INSTS: usize = 36;
const FACTOR: i64 = 4;

fn blocks_region_unroll(op: &Op) -> bool {
    is_libcall_mem(op)
        || matches!(*op, Op::Trap | Op::Trapz(..))
        || crate::mem::is_atomic_barrier(op)
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

fn is_preloop(func: &Function, cfg: &Cfg, head: Block, out_blk: usize, exit_blk: Block) -> bool {
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

fn region_defs_dominated(
    func: &Function,
    cfg: &Cfg,
    head: Block,
    out_blk: usize,
    in_loop: &[bool],
    loop_defs: &[bool],
) -> bool {
    for ob in 0..func.blocks.len() {
        if in_loop.get(ob).copied().unwrap_or(false) {
            continue;
        }
        let mut escapes = false;
        for &inst in &func.blocks[ob].insts {
            func.insts[inst.index()]
                .op
                .for_operands(|v| escapes |= loop_defs[v.index()]);
        }
        if escapes && (ob == out_blk || !cfg.dominates(head, Block(ob as u32))) {
            return false;
        }
    }
    true
}

fn has_iv_mask(
    func: &Function,
    blocks: &[Block],
    iv_param: Value,
    global_consts: &[Option<i64>],
) -> bool {
    for &blk in blocks {
        for &inst in &func.blocks[blk.index()].insts {
            let Op::Binary(Binary::Band, a, b) = func.insts[inst.index()].op else {
                continue;
            };
            let mask = if a == iv_param {
                global_consts[b.index()]
            } else if b == iv_param {
                global_consts[a.index()]
            } else {
                None
            };
            if mask.is_some_and(|m| (1..=7).contains(&m)) {
                return true;
            }
        }
    }
    false
}

struct Plan {
    blocks: Vec<Block>,
    latch: Block,
    out_blk: usize,
    counted_info: super::count::Counted,
    const_stride: i64,
    back_args: Vec<Value>,
    iv_back: Value,
}

fn analyze(func: &Function, cfg: &Cfg, each: &Loop, global_consts: &[Option<i64>]) -> Option<Plan> {
    if each.latches.len() != 1 || !(2..=MAX_BLOCKS).contains(&each.size) {
        return None;
    }
    let head = each.head;
    let latch = each.latches[0];
    if head == latch || func.blocks[head.index()].params.len() > 8 {
        return None;
    }
    let blocks: Vec<Block> = cfg
        .rpo
        .iter()
        .copied()
        .filter(|b| each.body.get(b.index()).copied().unwrap_or(false))
        .collect();
    if blocks.len() != each.size || blocks.first() != Some(&head) {
        return None;
    }
    let mut total_insts = 0usize;
    let mut loop_defs = vec![false; func.types.len()];
    for &blk in &blocks {
        let data = &func.blocks[blk.index()];
        if data.insts.is_empty()
            || has_bad_non_term(func, blk.index())
            || !matches!(func.last(blk), Some(Op::Jump(..) | Op::Brif(..)))
        {
            return None;
        }
        let non_term = &data.insts[..data.insts.len() - 1];
        total_insts += non_term.len();
        if total_insts > MAX_INSTS
            || non_term
                .iter()
                .any(|&i| blocks_region_unroll(&func.insts[i.index()].op))
        {
            return None;
        }
        if blk != head && !cfg.preds_of(blk).iter().all(|p| each.body[p.index()]) {
            return None;
        }
        if blk != latch
            && !func
                .successors(blk)
                .all(|s| s != head && each.body.get(s.index()).copied().unwrap_or(false))
        {
            return None;
        }
        for &p in &data.params {
            loop_defs[p.index()] = true;
        }
        for &i in non_term {
            if let Some(r) = func.insts[i.index()].result {
                loop_defs[r.index()] = true;
            }
        }
    }
    let outside: Vec<usize> = cfg
        .preds_of(head)
        .iter()
        .map(|p| p.index())
        .filter(|&p| !each.body.get(p).copied().unwrap_or(false))
        .collect();
    let [out_blk] = outside.as_slice() else {
        return None;
    };
    let out_blk = *out_blk;
    if !matches!(
        func.last(Block(out_blk as u32)),
        Some(Op::Jump(..) | Op::Brif(..))
    ) {
        return None;
    }
    let Op::Brif(cond, t_blk, t_args, f_blk, f_args) = func.last(latch)? else {
        return None;
    };
    if t_blk == f_blk {
        return None;
    }
    let (cond, back_on_true, back_args, exit_blk) = if *t_blk == head && !each.body[f_blk.index()] {
        (*cond, true, t_args.clone(), *f_blk)
    } else if *f_blk == head && !each.body[t_blk.index()] {
        (*cond, false, f_args.clone(), *t_blk)
    } else {
        return None;
    };
    if back_args.len() != func.blocks[head.index()].params.len()
        || func.successors(exit_blk).any(|t| t == exit_blk)
        || is_preloop(func, cfg, head, out_blk, exit_blk)
        || !region_defs_dominated(func, cfg, head, out_blk, &each.body, &loop_defs)
    {
        return None;
    }
    let mut always_insts = Vec::new();
    for blk in [head, latch] {
        let insts = &func.blocks[blk.index()].insts;
        always_insts.extend_from_slice(&insts[..insts.len() - 1]);
    }
    let counted_info = counted(
        func,
        head.index(),
        &always_insts,
        (cond, back_on_true),
        &back_args,
        &loop_defs,
        global_consts,
    )?;
    let const_stride = stride_const(global_consts, counted_info.stride)?;
    let iv_param = func.blocks[head.index()].params[counted_info.iv_idx];
    let iv_back = back_args[counted_info.iv_idx];
    let is_direct_step = always_insts.iter().any(|&i| {
        let data = &func.insts[i.index()];
        data.result == Some(iv_back)
            && matches!(
                data.op,
                Op::Binary(Binary::Iadd, a, b)
                    if (a == iv_param && b == counted_info.stride)
                        || (a == counted_info.stride && b == iv_param)
            )
    });
    if !is_direct_step || const_stride != 1 || !has_iv_mask(func, &blocks, iv_param, global_consts)
    {
        return None;
    }
    Some(Plan {
        blocks,
        latch,
        out_blk,
        counted_info,
        const_stride,
        back_args,
        iv_back,
    })
}

pub(super) fn unroll(
    func: &mut Function,
    cfg: &Cfg,
    each: &Loop,
    global_consts: &[Option<i64>],
) -> bool {
    let Some(plan) = analyze(func, cfg, each, global_consts) else {
        return false;
    };
    let head = plan.blocks[0];
    let head_tys: Vec<Type> = func.blocks[head.index()]
        .params
        .iter()
        .map(|&p| func.ty(p))
        .collect();
    let guard_params: Vec<Value> = head_tys.iter().map(|&ty| func.value(ty)).collect();
    let guard = Block(func.blocks.len() as u32);
    func.blocks.push(BlockData {
        params: guard_params.clone(),
        insts: Vec::new(),
    });
    let mut clones: Vec<Vec<Block>> = Vec::with_capacity(FACTOR as usize);
    for _ in 0..(FACTOR as usize) {
        let mut round_blks = Vec::with_capacity(plan.blocks.len());
        for &blk in &plan.blocks {
            let tys: Vec<Type> = func.blocks[blk.index()]
                .params
                .iter()
                .map(|&p| func.ty(p))
                .collect();
            let params: Vec<Value> = tys.into_iter().map(|ty| func.value(ty)).collect();
            let nb = Block(func.blocks.len() as u32);
            func.blocks.push(BlockData {
                params,
                insts: Vec::new(),
            });
            round_blks.push(nb);
        }
        clones.push(round_blks);
    }
    let strides: Vec<Value> = (1..=(FACTOR as usize))
        .map(|k| {
            let v = func.value(plan.counted_info.iv_ty);
            let imm = func.add(Op::Const(plan.const_stride * k as i64), Some(v));
            func.blocks[guard.index()].insts.push(imm);
            v
        })
        .collect();
    let unrolled_head = clones[0][0];
    let safe_limit = emit_dynamic_guard(
        func,
        guard,
        &plan.counted_info,
        (
            Some(plan.const_stride),
            FACTOR,
            Some(strides[FACTOR as usize - 1]),
        ),
        (head, unrolled_head, guard_params),
        &[],
    );
    let iv_base = func.blocks[unrolled_head.index()].params[plan.counted_info.iv_idx];
    for r in 0..(FACTOR as usize) {
        let mut map: Vec<(Value, Value)> = Vec::new();
        for (at, &orig_blk) in plan.blocks.iter().enumerate() {
            let dst_blk = clones[r][at];
            for (&old_p, &new_p) in func.blocks[orig_blk.index()]
                .params
                .iter()
                .zip(func.blocks[dst_blk.index()].params.iter())
            {
                map.push((old_p, new_p));
            }
        }
        for (at, &orig_blk) in plan.blocks.iter().enumerate() {
            let dst_blk = clones[r][at];
            let insts = func.blocks[orig_blk.index()].insts.clone();
            let non_term = &insts[..insts.len() - 1];
            for &orig_inst in non_term {
                if func.insts[orig_inst.index()].result == Some(plan.iv_back) {
                    let nr = func.value_like(plan.iv_back);
                    map.push((plan.iv_back, nr));
                    let ni = func.add(Op::Binary(Binary::Iadd, iv_base, strides[r]), Some(nr));
                    func.blocks[dst_blk.index()].insts.push(ni);
                    continue;
                }
                clone_body(func, &[orig_inst], dst_blk, &mut map);
            }
            if orig_blk != plan.latch {
                let mut op = func.insts[insts[insts.len() - 1].index()].op.clone();
                op.map_operands(|v| remap_val(&map, v));
                op.map_targets(|t| {
                    plan.blocks
                        .iter()
                        .position(|&b| b == t)
                        .map_or(t, |idx| clones[r][idx])
                });
                let ti = func.add(op, None);
                func.blocks[dst_blk.index()].insts.push(ti);
            } else {
                let remapped_back: Vec<Value> =
                    plan.back_args.iter().map(|&v| remap_val(&map, v)).collect();
                if r + 1 < FACTOR as usize {
                    let jmp = func.add(Op::Jump(clones[r + 1][0], remapped_back), None);
                    func.blocks[dst_blk.index()].insts.push(jmp);
                } else {
                    let cond = func.value(Type::I8);
                    let icmp = func.add(
                        Op::Icmp(
                            plan.counted_info.cmp_cc,
                            remapped_back[plan.counted_info.iv_idx],
                            safe_limit,
                        ),
                        Some(cond),
                    );
                    let br = func.add(
                        Op::Brif(
                            cond,
                            unrolled_head,
                            remapped_back.clone(),
                            head,
                            remapped_back,
                        ),
                        None,
                    );
                    func.blocks[dst_blk.index()].insts.extend([icmp, br]);
                }
            }
        }
    }
    if let Some(&last) = func.blocks[plan.out_blk].insts.last() {
        func.insts[last.index()]
            .op
            .map_targets(|b| if b == head { guard } else { b });
    }
    true
}

mod count;
mod memset;
mod region;
mod symbolic;
mod vector;

use crate::cfg::{Cfg, Loop};
use crate::fold::{evaluate_wrapped, extend};
use crate::ir::{Binary, Block, BlockData, Domain, Function, Inst, IntCC, Op, Type, Value};
use crate::{Arch, Target};
pub(crate) use memset::Emit;
pub(crate) use vector::splat_in;
use count::{Counted, counted, head_defs, outside_preds, stride_const};

const MAX_FULL_TRIPS: usize = 64;
const MAX_FULL_INSTS: usize = 1024;
pub(super) const MAX_FP_FULL_TRIPS: usize = 8;
pub(super) const MAX_FP_FULL_INSTS: usize = 96;
pub(super) const MAX_VEC_FULL_TRIPS: usize = 20;
const MAX_CHAIN_BLOCKS: usize = 8;
const MAX_CHAIN_TRIPS: usize = 64;
const MAX_CHAIN_UNROLLED_INSTS: usize = 384;

type Eval = fn(&Function, &Op, Type, &dyn Fn(Value) -> Option<i64>) -> Option<i64>;

/// The value the latest entry of `map` renames `v` to, or `v` when none does.
pub(crate) fn remap_val(map: &[(Value, Value)], v: Value) -> Value {
    map.iter().rev().find(|&&(old, _)| old == v).map_or(v, |&(_, new)| new)
}

fn is_libcall_mem(op: &Op) -> bool {
    matches!(*op, Op::Copy(..)) || matches!(*op, Op::Move(_, _, bytes) if bytes > crate::copy::MOVE_INLINE)
}

pub(super) fn block_has_libcall_mem(func: &Function, block_idx: usize) -> bool {
    let insts = &func.blocks[block_idx].insts;
    insts[..insts.len().saturating_sub(1)].iter().any(|&i| is_libcall_mem(&func.insts[i.index()].op))
}

fn has_bad_non_term(func: &Function, block_idx: usize) -> bool {
    let insts = &func.blocks[block_idx].insts;
    insts[..insts.len() - 1]
        .iter()
        .any(|&i| matches!(func.insts[i.index()].op, Op::Call { .. } | Op::TryCall { .. } | Op::BrTable(..) | Op::Jump(..) | Op::Brif(..)))
}

fn eval_block(func: &Function, block_idx: usize, params: &[Option<i64>], env: &mut [Option<i64>], eval: Eval) {
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
            _ => eval(func, &data.op, ty, &|v| env[v.index()]),
        };
    }
}

fn clone_body(func: &mut Function, src_non_term: &[Inst], dst: Block, map: &mut Vec<(Value, Value)>) {
    for &orig_inst in src_non_term {
        let mut op = func.insts[orig_inst.index()].op.clone();
        op.map_operands(|v| remap_val(map, v));
        let new_res = func.insts[orig_inst.index()].result.map(|r| {
            let nr = func.value_like(r);
            map.push((r, nr));
            nr
        });
        let ni = func.add(op, new_res);
        func.blocks[dst.index()].insts.push(ni);
    }
}

fn remap_other_blocks(func: &mut Function, skip: &[usize], map: &[(Value, Value)]) {
    if map.is_empty() {
        return;
    }
    for ob in 0..func.blocks.len() {
        if skip.contains(&ob) {
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

fn edge_args(func: &Function, pred: Block, head: Block) -> Option<&[Value]> {
    match func.last(pred)? {
        Op::Jump(target, args) if *target == head => Some(args.as_slice()),
        Op::Brif(_, t_blk, t_args, f_blk, _) if *t_blk == head && *f_blk != head => Some(t_args.as_slice()),
        Op::Brif(_, t_blk, _, f_blk, f_args) if *f_blk == head && *t_blk != head => Some(f_args.as_slice()),
        _ => None,
    }
}

fn meet(known: &mut Option<Vec<Option<i64>>>, args: &[Value], env: &[Option<i64>]) {
    match known {
        None => *known = Some(args.iter().map(|arg| env[arg.index()]).collect()),
        Some(known) => {
            for (slot, arg) in known.iter_mut().zip(args) {
                *slot = slot.filter(|&imm| env[arg.index()] == Some(imm));
            }
        }
    }
}

fn outside_const_params(func: &Function, cfg: &Cfg, head: Block, latches: &[Block], global_consts: &[Option<i64>]) -> Option<Vec<Option<i64>>> {
    let param_len = func.blocks[head.index()].params.len();
    let mut acc: Option<Vec<Option<i64>>> = None;
    for p in 0..func.blocks.len() {
        let blk = Block(p as u32);
        if p == head.index() || latches.contains(&blk) || !cfg.reachable.get(p).copied().unwrap_or(false) || !func.successors(blk).any(|t| t == head) {
            continue;
        }
        let args = edge_args(func, blk, head).filter(|a| a.len() == param_len)?;
        meet(&mut acc, args, global_consts);
    }
    acc
}

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
    if inst_count > MAX_CHAIN_UNROLLED_INSTS + 1 {
        return false;
    }
    let Some(mut cur_params) = outside_const_params(func, cfg, head, &[head], global_consts) else {
        return unroll_self_dynamic(func, cfg, (b, cond, back_on_true), (&back_args, exit_blk), global_consts);
    };
    if back_args.len() != func.blocks[b].params.len() {
        return false;
    }
    let mut proven = None;
    for step in 0..=MAX_FULL_TRIPS {
        let mut env = global_consts.to_vec();
        eval_block(func, b, &cur_params, &mut env, evaluate_wrapped);
        let Some(cond_val) = env[cond.index()] else { break };
        if (cond_val != 0) == back_on_true {
            cur_params = back_args.iter().map(|v| env[v.index()]).collect();
        } else {
            proven = Some(step);
            break;
        }
    }
    let Some(back_steps) = proven else {
        return unroll_self_dynamic(func, cfg, (b, cond, back_on_true), (&back_args, exit_blk), global_consts);
    };
    let last = *func.blocks[b].insts.last().unwrap_or(&Inst(0));
    if back_steps == 0 {
        func.insts[last.index()].op = Op::Jump(exit_blk, exit_args);
        return true;
    }
    if !self_unroll_fits(func, b, inst_count, back_steps + 1) {
        return unroll_self_dynamic(func, cfg, (b, cond, back_on_true), (&back_args, exit_blk), global_consts);
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
    remap_other_blocks(func, &[b], &last_map);
    true
}

fn blocks_dynamic_unroll(op: &Op) -> bool {
    is_libcall_mem(op) || matches!(*op, Op::Trap | Op::Trapz(..) | Op::AtomicLoad(..) | Op::AtomicStore(..) | Op::AtomicRmw(..) | Op::AtomicCas(..) | Op::Fence)
}

fn head_defs_dominated(func: &Function, cfg: &Cfg, (b, out_blk): (usize, usize), head_defs: &[bool]) -> bool {
    let head = Block(b as u32);
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
    true
}

fn emit_dynamic_guard(
    func: &mut Function,
    guard: Block,
    counted_info: &Counted,
    (const_stride, factor): (Option<i64>, i64),
    (head, unrolled, guard_params): (Block, Block, Vec<Value>),
) -> Value {
    let &Counted { cmp_cc, is_ne, iv_ty, limit, iv_idx, stride, bias } = counted_info;
    let mut emit_guard = |op: Op, ty: Type| -> Value {
        let v = func.value(ty);
        let i = func.add(op, Some(v));
        func.blocks[guard.index()].insts.push(i);
        v
    };
    let (s_need, g1) = if let Some(c) = const_stride {
        (emit_guard(Op::Const(c * factor + bias), iv_ty), None)
    } else {
        let zero = emit_guard(Op::Const(0), iv_ty);
        let sh = emit_guard(Op::Const(i64::from(iv_ty.bits() - 3)), iv_ty);
        let cf = emit_guard(Op::Const(factor), iv_ty);
        let stride_ok = if is_ne {
            let one = emit_guard(Op::Const(1), iv_ty);
            emit_guard(Op::Icmp(IntCC::Equal, stride, one), Type::I8)
        } else {
            emit_guard(Op::Icmp(IntCC::UnsignedGreaterThan, stride, zero), Type::I8)
        };
        let or_lim = emit_guard(Op::Binary(Binary::Bor, limit, stride), iv_ty);
        let hi_bits = emit_guard(Op::Binary(Binary::Ushr, or_lim, sh), iv_ty);
        let no_wrap = emit_guard(Op::Icmp(IntCC::Equal, hi_bits, zero), Type::I8);
        let sf = emit_guard(Op::Binary(Binary::Imul, stride, cf), iv_ty);
        let need = if bias > 0 {
            let b_val = emit_guard(Op::Const(bias), iv_ty);
            emit_guard(Op::Binary(Binary::Iadd, sf, b_val), iv_ty)
        } else {
            sf
        };
        (need, Some(emit_guard(Op::Binary(Binary::Band, stride_ok, no_wrap), Type::I8)))
    };
    let lim_ok = emit_guard(Op::Icmp(IntCC::UnsignedLessThanOrEqual, s_need, limit), Type::I8);
    let safe_limit = emit_guard(Op::Binary(Binary::Isub, limit, s_need), iv_ty);
    let init_ok = emit_guard(Op::Icmp(cmp_cc, guard_params[iv_idx], safe_limit), Type::I8);
    let g2 = emit_guard(Op::Binary(Binary::Band, lim_ok, init_ok), Type::I8);
    let guard_cond = match g1 {
        Some(g1) => emit_guard(Op::Binary(Binary::Band, g1, g2), Type::I8),
        None => g2,
    };
    let guard_br = func.add(Op::Brif(guard_cond, unrolled, guard_params.clone(), head, guard_params), None);
    func.blocks[guard.index()].insts.push(guard_br);
    safe_limit
}

fn retarget_outside_pred(func: &mut Function, out_blk: usize, head: Block, guard: Block) {
    let Some(&last) = func.blocks[out_blk].insts.last() else { return };
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

fn peel_jumps(func: &Function, mut blk: Block) -> Block {
    for _ in 0..4 {
        match func.last(blk) {
            Some(Op::Jump(next, _)) if *next != blk => blk = *next,
            _ => break,
        }
    }
    blk
}

pub(super) fn block_has_vec(func: &Function, blk: usize) -> bool {
    func.blocks[blk].insts.iter().any(|&i| func.insts[i.index()].result.is_some_and(|r| func.ty(r).is_vector()))
}

pub(super) fn block_has_float(func: &Function, blk: usize) -> bool {
    func.blocks[blk].insts.iter().any(|&i| func.insts[i.index()].result.is_some_and(|r| func.ty(r).is_float()))
}

fn is_preloop_for_remainder(func: &Function, cfg: &Cfg, head: Block, out_blk: usize, exit_blk: Block) -> bool {
    let rem = peel_jumps(func, exit_blk);
    if rem == head || !cfg.loops().list.iter().any(|lp| lp.head == rem) {
        return false;
    }
    match func.last(Block(out_blk as u32)) {
        Some(Op::Brif(_, t, _, f, _)) => peel_jumps(func, if *t == head { *f } else { *t }) == rem,
        _ => false,
    }
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
    if body_insts.iter().any(|&i| blocks_dynamic_unroll(&func.insts[i.index()].op)) {
        return false;
    }
    let head_defs = head_defs(func, b, &body_insts);
    let outside = outside_preds(func, b);
    let [out_blk] = outside.as_slice() else { return false };
    let out_blk = *out_blk;
    if is_preloop_for_remainder(func, cfg, head, out_blk, exit_blk)
        || !func.blocks[out_blk].insts.last().is_some_and(|&last| matches!(func.insts[last.index()].op, Op::Jump(..) | Op::Brif(..)))
        || !head_defs_dominated(func, cfg, (b, out_blk), &head_defs)
    {
        return false;
    }
    let Some(counted_info) = counted(func, b, &body_insts, (cond, back_on_true), back_args, &head_defs, global_consts) else {
        return false;
    };
    let const_stride = stride_const(global_consts, counted_info.stride, counted_info.is_ne);
    let factor = if inst_count <= 16 { 4_i64 } else { 2_i64 };
    let param_tys: Vec<Type> = func.blocks[b].params.iter().map(|&p| func.ty(p)).collect();
    let guard_params: Vec<Value> = param_tys.iter().map(|&ty| func.value(ty)).collect();
    let (guard, unrolled) = (Block(func.blocks.len() as u32), Block((func.blocks.len() + 1) as u32));
    let unrolled_params: Vec<Value> = param_tys.iter().map(|&ty| func.value(ty)).collect();
    func.blocks.push(BlockData { params: guard_params.clone(), insts: Vec::new() });
    func.blocks.push(BlockData { params: unrolled_params.clone(), insts: Vec::new() });
    let safe_limit = emit_dynamic_guard(func, guard, &counted_info, (const_stride, factor), (head, unrolled, guard_params));
    let mut cur_args = unrolled_params;
    for _ in 0..factor {
        let mut map: Vec<(Value, Value)> = func.blocks[b].params.iter().copied().zip(cur_args.iter().copied()).collect();
        clone_body(func, &body_insts, unrolled, &mut map);
        cur_args = back_args.iter().map(|&v| remap_val(&map, v)).collect();
    }
    let loop_cond = func.value(Type::I8);
    let loop_icmp = func.add(Op::Icmp(counted_info.cmp_cc, cur_args[counted_info.iv_idx], safe_limit), Some(loop_cond));
    let loop_br = func.add(Op::Brif(loop_cond, unrolled, cur_args.clone(), head, cur_args), None);
    func.blocks[unrolled.index()].insts.extend([loop_icmp, loop_br]);
    retarget_outside_pred(func, out_blk, head, guard);
    true
}

struct ChainSeg {
    blk: Block,
    params: Vec<Value>,
    non_term: Vec<Inst>,
    edge_args: Vec<Value>,
}

enum LoopChainKind {
    TopTested {
        cond: Value,
        body_on_true: bool,
        exit_blk: Block,
        exit_args: Vec<Value>,
    },
    BottomTested {
        cond: Value,
        back_on_true: bool,
        exit_blk: Block,
        exit_args: Vec<Value>,
    },
}

struct LoopChain {
    segs: Vec<ChainSeg>,
    kind: LoopChainKind,
}

fn seg_from_block(func: &Function, blk: Block, edge_args: Vec<Value>) -> ChainSeg {
    let data = &func.blocks[blk.index()];
    ChainSeg {
        blk,
        params: data.params.clone(),
        non_term: data.insts[..data.insts.len() - 1].to_vec(),
        edge_args,
    }
}

fn clean_chain_block(func: &Function, cfg: &Cfg, each: &Loop, blk: Block) -> bool {
    let idx = blk.index();
    cfg.reachable.get(idx).copied().unwrap_or(false)
        && each.body.get(idx).copied().unwrap_or(false)
        && func.entry != Some(blk)
        && (blk == each.head || cfg.pred_count(blk) == 1)
        && !func.blocks[idx].insts.is_empty()
        && !has_bad_non_term(func, idx)
}

fn push_params_map(map: &mut Vec<(Value, Value)>, params: &[Value], args: &[Value]) {
    for (&p, &a) in params.iter().zip(args.iter()) {
        map.push((p, a));
    }
}

fn collect_chain(func: &Function, cfg: &Cfg, each: &Loop) -> Option<LoopChain> {
    let head = each.head;
    let latch = each.latches[0];
    if head == latch || !clean_chain_block(func, cfg, each, head) {
        return None;
    }
    let head_last = *func.blocks[head.index()].insts.last()?;
    let (mut cur, mut cur_args, mut segs, is_top, top_meta) = match &func.insts[head_last.index()].op {
        Op::Jump(next_blk, next_args) if *next_blk != head => (
            *next_blk,
            next_args.clone(),
            vec![seg_from_block(func, head, next_args.clone())],
            false,
            None,
        ),
        Op::Brif(cond, t_blk, t_args, f_blk, f_args) if t_blk != f_blk => {
            let t_in = each.body.get(t_blk.index()).copied().unwrap_or(false);
            let f_in = each.body.get(f_blk.index()).copied().unwrap_or(false);
            let (body_on_true, body_blk, body_args, exit_blk, exit_args) = if t_in && !f_in && *t_blk != head {
                (true, *t_blk, t_args.clone(), *f_blk, f_args.clone())
            } else if f_in && !t_in && *f_blk != head {
                (false, *f_blk, f_args.clone(), *t_blk, t_args.clone())
            } else {
                return None;
            };
            (
                body_blk,
                body_args.clone(),
                vec![seg_from_block(func, head, body_args)],
                true,
                Some((*cond, body_on_true, exit_blk, exit_args)),
            )
        }
        _ => return None,
    };

    loop {
        if segs.len() >= each.size
            || segs.iter().any(|s| s.blk == cur)
            || !clean_chain_block(func, cfg, each, cur)
            || cur_args.len() != func.blocks[cur.index()].params.len()
        {
            return None;
        }
        let last = *func.blocks[cur.index()].insts.last()?;
        if cur != latch {
            let Op::Jump(nxt, nxt_args) = &func.insts[last.index()].op else { return None };
            let nxt_args = nxt_args.clone();
            segs.push(seg_from_block(func, cur, nxt_args.clone()));
            cur = *nxt;
            cur_args = nxt_args;
            continue;
        }
        if segs.len() + 1 != each.size {
            return None;
        }
        if is_top {
            let Op::Jump(target, back_args) = &func.insts[last.index()].op else { return None };
            if *target != head || back_args.len() != func.blocks[head.index()].params.len() {
                return None;
            }
            segs.push(seg_from_block(func, cur, back_args.clone()));
            let (cond, body_on_true, exit_blk, exit_args) = top_meta?;
            return Some(LoopChain {
                segs,
                kind: LoopChainKind::TopTested {
                    cond,
                    body_on_true,
                    exit_blk,
                    exit_args,
                },
            });
        }
        let Op::Brif(cond, t_blk, t_args, f_blk, f_args) = &func.insts[last.index()].op else { return None };
        if t_blk == f_blk {
            return None;
        }
        let (back_on_true, back_args, exit_blk, exit_args) = if *t_blk == head {
            (true, t_args.clone(), *f_blk, f_args.clone())
        } else if *f_blk == head {
            (false, f_args.clone(), *t_blk, t_args.clone())
        } else {
            return None;
        };
        if each.body.get(exit_blk.index()).copied().unwrap_or(false)
            || back_args.len() != func.blocks[head.index()].params.len()
        {
            return None;
        }
        segs.push(seg_from_block(func, cur, back_args));
        return Some(LoopChain {
            segs,
            kind: LoopChainKind::BottomTested {
                cond: *cond,
                back_on_true,
                exit_blk,
                exit_args,
            },
        });
    }
}

fn eval_chain(
    func: &Function,
    chain: &LoopChain,
    init_params: Vec<Option<i64>>,
    global_consts: &[Option<i64>],
    max_trips: usize,
) -> Option<usize> {
    let mut cur_head_params = init_params;
    match &chain.kind {
        LoopChainKind::TopTested {
            cond, body_on_true, ..
        } => {
            for step in 0..=max_trips {
                let mut env = global_consts.to_vec();
                eval_block(func, chain.segs[0].blk.index(), &cur_head_params, &mut env, evaluate_wrapped);
                let cond_val = env[cond.index()]?;
                if (cond_val != 0) != *body_on_true {
                    return Some(step);
                }
                if step == max_trips {
                    return None;
                }
                let mut cur_args: Vec<Option<i64>> = chain.segs[0].edge_args.iter().map(|v| env[v.index()]).collect();
                for seg in &chain.segs[1..] {
                    eval_block(func, seg.blk.index(), &cur_args, &mut env, evaluate_wrapped);
                    cur_args = seg.edge_args.iter().map(|v| env[v.index()]).collect();
                }
                cur_head_params = cur_args;
            }
            None
        }
        LoopChainKind::BottomTested {
            cond, back_on_true, ..
        } => {
            for step in 0..max_trips {
                let mut env = global_consts.to_vec();
                let mut cur_args = cur_head_params;
                for seg in &chain.segs {
                    eval_block(func, seg.blk.index(), &cur_args, &mut env, evaluate_wrapped);
                    cur_args = seg.edge_args.iter().map(|v| env[v.index()]).collect();
                }
                let cond_val = env[cond.index()]?;
                if (cond_val != 0) != *back_on_true {
                    return Some(step);
                }
                cur_head_params = cur_args;
            }
            None
        }
    }
}

fn finish_chain_exit(func: &mut Function, b: usize, segs: &[ChainSeg], (exit_blk, exit_args): (Block, &[Value]), skip: &[usize], map: &[(Value, Value)]) -> bool {
    let final_exit_args: Vec<Value> = exit_args.iter().map(|&v| remap_val(map, v)).collect();
    let jump = func.add(Op::Jump(exit_blk, final_exit_args), None);
    func.blocks[b].insts.push(jump);
    remap_other_blocks(func, skip, map);
    for seg in &segs[1..] {
        retire_block(func, seg.blk.index());
    }
    true
}

fn emit_top_chain(func: &mut Function, segs: &[ChainSeg], trips: usize, (exit_blk, exit_args): (Block, Vec<Value>), skip: &[usize]) -> bool {
    let head = segs[0].blk;
    let b = head.index();
    if trips == 0 {
        let last = *func.blocks[b].insts.last().unwrap_or(&Inst(0));
        func.insts[last.index()].op = Op::Jump(exit_blk, exit_args);
        segs[1..].iter().for_each(|seg| retire_block(func, seg.blk.index()));
        return true;
    }
    pop_terminator(func, b);
    let mut head_map = Vec::new();
    let mut cur_body_args = segs[0].edge_args.clone();
    for _ in 0..trips {
        let mut iter_map = head_map.clone();
        for seg in &segs[1..] {
            push_params_map(&mut iter_map, &seg.params, &cur_body_args);
            clone_body(func, &seg.non_term, head, &mut iter_map);
            cur_body_args = seg.edge_args.iter().map(|&v| remap_val(&iter_map, v)).collect();
        }
        let mut next_head_map = Vec::new();
        push_params_map(&mut next_head_map, &segs[0].params, &cur_body_args);
        clone_body(func, &segs[0].non_term, head, &mut next_head_map);
        head_map = next_head_map;
        cur_body_args = segs[0].edge_args.iter().map(|&v| remap_val(&head_map, v)).collect();
    }
    finish_chain_exit(func, b, segs, (exit_blk, &exit_args), skip, &head_map)
}

fn emit_bottom_chain(func: &mut Function, segs: &[ChainSeg], proven: usize, (exit_blk, exit_args): (Block, Vec<Value>), skip: &[usize]) -> bool {
    let head = segs[0].blk;
    pop_terminator(func, head.index());
    let mut last_map = Vec::new();
    let mut cur_edge_args = segs[0].edge_args.clone();
    for seg in &segs[1..] {
        push_params_map(&mut last_map, &seg.params, &cur_edge_args);
        clone_body(func, &seg.non_term, head, &mut last_map);
        cur_edge_args = seg.edge_args.iter().map(|&v| remap_val(&last_map, v)).collect();
    }
    for _ in 0..proven {
        let mut step_map = Vec::new();
        for seg in segs {
            push_params_map(&mut step_map, &seg.params, &cur_edge_args);
            clone_body(func, &seg.non_term, head, &mut step_map);
            cur_edge_args = seg.edge_args.iter().map(|&v| remap_val(&step_map, v)).collect();
        }
        last_map = step_map;
    }
    finish_chain_exit(func, head.index(), segs, (exit_blk, &exit_args), skip, &last_map)
}

fn chain_fits(func: &Function, segs: &[ChainSeg], body_insts: usize, trips: usize) -> bool {
    if trips == 0 {
        return true;
    }
    if trips > 4 && segs.iter().any(|s| block_has_libcall_mem(func, s.blk.index())) {
        return false;
    }
    if segs.iter().any(|s| block_has_float(func, s.blk.index())) {
        return trips <= MAX_FP_FULL_TRIPS && trips * body_insts <= MAX_FP_FULL_INSTS;
    }
    if trips > MAX_VEC_FULL_TRIPS && segs.iter().any(|s| block_has_vec(func, s.blk.index())) {
        return false;
    }
    let is_two_seg = segs.len() == 2;
    let total_insts = body_insts + segs.len();
    let head_len = segs[0].non_term.len() + 1;
    let fits_legacy = is_two_seg && total_insts <= 64 && head_len + trips * total_insts <= MAX_FULL_INSTS;
    fits_legacy || (trips <= MAX_CHAIN_TRIPS && trips * body_insts <= MAX_CHAIN_UNROLLED_INSTS)
}

fn unroll_chain(func: &mut Function, cfg: &Cfg, each: &Loop, global_consts: &[Option<i64>]) -> bool {
    let Some(chain) = collect_chain(func, cfg, each) else { return false };
    let head = chain.segs[0].blk;
    let body_insts: usize = chain.segs.iter().map(|s| s.non_term.len()).sum();
    let is_two_seg = chain.segs.len() == 2;
    if body_insts > MAX_CHAIN_UNROLLED_INSTS {
        return false;
    }
    let Some(init_params) = outside_const_params(func, cfg, head, &each.latches, global_consts) else { return false };
    let max_trips = if is_two_seg { MAX_FULL_TRIPS } else { MAX_CHAIN_TRIPS };
    let Some(proven) = eval_chain(func, &chain, init_params, global_consts, max_trips) else { return false };
    let trips = match &chain.kind {
        LoopChainKind::TopTested { .. } => proven,
        LoopChainKind::BottomTested { .. } => proven + 1,
    };
    if !chain_fits(func, &chain.segs, body_insts, trips) {
        return false;
    }
    let skip: Vec<usize> = chain.segs.iter().map(|s| s.blk.index()).collect();
    match chain.kind {
        LoopChainKind::TopTested { exit_blk, exit_args, .. } => emit_top_chain(func, &chain.segs, trips, (exit_blk, exit_args), &skip),
        LoopChainKind::BottomTested { exit_blk, exit_args, .. } => emit_bottom_chain(func, &chain.segs, proven, (exit_blk, exit_args), &skip),
    }
}

fn collect_global_consts(func: &Function, cfg: &Cfg, eval: Eval) -> Vec<Option<i64>> {
    let mut global_consts = vec![None; func.types.len()];
    for &blk in &cfg.rpo {
        for &inst in &func.blocks[blk.index()].insts {
            let data = &func.insts[inst.index()];
            let Some(res) = data.result else { continue };
            if let Some(imm) = eval(func, &data.op, func.ty(res), &|v| global_consts[v.index()]) {
                global_consts[res.index()] = Some(imm);
            }
        }
    }
    global_consts
}

fn unroll_single_block(func: &mut Function, cfg: &Cfg, b: usize, global_consts: &[Option<i64>], sad: bool) -> bool {
    if has_bad_non_term(func, b) {
        return false;
    }
    let Some(&last) = func.blocks[b].insts.last() else { return false };
    let Op::Brif(cond, t_blk, t_args, f_blk, f_args) = &func.insts[last.index()].op else { return false };
    if t_blk == f_blk {
        return false;
    }
    let head = Block(b as u32);
    let (cond, back_on_true, back_args, exit_blk, exit_args) = if *t_blk == head {
        (*cond, true, t_args.clone(), *f_blk, f_args.clone())
    } else if *f_blk == head {
        (*cond, false, f_args.clone(), *t_blk, t_args.clone())
    } else {
        return false;
    };
    memset::run(func, b, (cond, back_on_true), &back_args, (exit_blk, &exit_args), global_consts)
        || vector::run(func, b, (cond, back_on_true), &back_args, (exit_blk, &exit_args), (global_consts, sad))
        || unroll_self(func, cfg, b, cond, back_on_true, back_args, exit_blk, exit_args, global_consts)
}

fn unroll_loop_step(func: &mut Function, cfg: &Cfg, each: &Loop, global_consts: &[Option<i64>], sad: bool) -> bool {
    let head = each.head;
    if each.latches.len() != 1 {
        region::unroll(func, cfg, each)
    } else if each.size == 1 && each.latches[0] == head {
        unroll_single_block(func, cfg, head.index(), global_consts, sad)
    } else if each.size >= 2 && each.latches[0] != head {
        vector::flatten(func, each, (global_consts, sad)) || unroll_chain(func, cfg, each, global_consts) || region::unroll(func, cfg, each)
    } else {
        false
    }
}

/// Unrolls clean 1-block and straight-line inner loops with constant or dynamic trip counts, and constant-trip loops whose bodies branch or leave early.
pub(crate) fn run(func: &mut Function, target: &Target) -> Domain {
    let sad = target.arch == Arch::X64;
    let Some(entry) = func.entry.filter(|_| func.coroutine.is_none()) else { return Domain::NONE };
    let initial = func.cfg();
    let loops = initial.loops();
    let mut cfg = std::sync::Arc::clone(&initial);
    let mut global_consts = collect_global_consts(func, &cfg, evaluate_wrapped);
    let (mut changed, mut dirty) = (false, false);
    for each in &loops.list {
        let head = each.head;
        let b = head.index();
        if head == entry || each.size > MAX_CHAIN_BLOCKS || func.blocks[b].insts.is_empty() {
            continue;
        }
        if dirty {
            func.bump_cfg();
            cfg = func.cfg();
            dirty = false;
        }
        if !cfg.reachable.get(b).copied().unwrap_or(false) || cfg.pred_count(head) < 2 {
            continue;
        }
        global_consts.resize(func.types.len(), None);
        let did = unroll_loop_step(func, &cfg, each, &global_consts, sad);
        changed |= did;
        dirty |= did;
    }
    if !changed {
        return Domain::NONE;
    }
    func.bump_cfg();
    Domain::CFG | Domain::VALUES | Domain::MEM
}

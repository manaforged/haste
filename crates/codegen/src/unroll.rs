mod count;
mod memset;
mod region;
mod symbolic;
mod vector;

use crate::cfg::{Cfg, Loop};
use crate::fold::{evaluate_wrapped, extend};
use crate::ir::{Binary, Block, BlockData, Domain, Function, Inst, IntCC, Op, Type, Value};
use count::{Counted, counted, head_defs, outside_preds, stride_const};

const MAX_FULL_TRIPS: usize = 64;
const MAX_FULL_INSTS: usize = 1024;
const MAX_CHAIN_BLOCKS: usize = 8;
const MAX_CHAIN_TRIPS: usize = 64;
const MAX_CHAIN_UNROLLED_INSTS: usize = 384;

/// Evaluates one operation from the known values of its operands.
type Eval = fn(&Function, &Op, Type, &dyn Fn(Value) -> Option<i64>) -> Option<i64>;

/// The value the latest entry of `map` renames `v` to, or `v` when none does.
pub(crate) fn remap_val(map: &[(Value, Value)], v: Value) -> Value {
    map.iter().rev().find(|&&(old, _)| old == v).map_or(v, |&(_, new)| new)
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

fn outside_const_params(
    func: &Function,
    cfg: &Cfg,
    head: Block,
    latches: &[Block],
    global_consts: &[Option<i64>],
) -> Option<Vec<Option<i64>>> {
    let param_len = func.blocks[head.index()].params.len();
    let mut acc: Option<Vec<Option<i64>>> = None;
    for p in 0..func.blocks.len() {
        if p == head.index()
            || latches.contains(&Block(p as u32))
            || !cfg.reachable.get(p).copied().unwrap_or(false)
            || !func.successors(Block(p as u32)).any(|t| t == head)
        {
            continue;
        }
        let args = edge_args(func, Block(p as u32), head)?;
        if args.len() != param_len {
            return None;
        }
        meet(&mut acc, args, global_consts);
    }
    acc
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
        return unroll_self_dynamic(func, cfg, b, cond, back_on_true, &back_args, exit_blk, global_consts);
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
        return unroll_self_dynamic(func, cfg, b, cond, back_on_true, &back_args, exit_blk, global_consts);
    };
    let last = *func.blocks[b].insts.last().unwrap_or(&Inst(0));
    if back_steps == 0 {
        func.insts[last.index()].op = Op::Jump(exit_blk, exit_args);
        return true;
    }
    let trips = back_steps + 1;
    let body_count = inst_count.saturating_sub(1);
    let fits_legacy = inst_count <= 64 && trips * inst_count <= MAX_FULL_INSTS;
    let fits_small = trips <= MAX_CHAIN_TRIPS && trips * body_count <= MAX_CHAIN_UNROLLED_INSTS;
    if !fits_legacy && !fits_small {
        return unroll_self_dynamic(func, cfg, b, cond, back_on_true, &back_args, exit_blk, global_consts);
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
    if inst_count > 48
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
    let head_defs = head_defs(func, b, &body_insts);
    let outside = outside_preds(func, b);
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
    let Some(Counted { cmp_cc, is_ne, iv_ty, limit, iv_idx, stride, bias }) =
        counted(func, b, &body_insts, (cond, back_on_true), back_args, &head_defs, global_consts)
    else {
        return false;
    };
    let const_stride = stride_const(global_consts, stride, is_ne);
    let factor = if inst_count <= 16 { 4_i64 } else { 2_i64 };
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

fn unroll_chain(func: &mut Function, cfg: &Cfg, each: &Loop, global_consts: &[Option<i64>]) -> bool {
    let Some(chain) = collect_chain(func, cfg, each) else { return false };
    let head = chain.segs[0].blk;
    let b = head.index();
    let body_insts: usize = chain.segs.iter().map(|s| s.non_term.len()).sum();
    let total_insts = body_insts + chain.segs.len();
    let is_two_seg = chain.segs.len() == 2;
    if (is_two_seg && total_insts > 64 && body_insts > MAX_CHAIN_UNROLLED_INSTS)
        || (!is_two_seg && body_insts > MAX_CHAIN_UNROLLED_INSTS)
    {
        return false;
    }
    let Some(init_params) = outside_const_params(func, cfg, head, &each.latches, global_consts) else { return false };
    let max_trips = if is_two_seg { MAX_FULL_TRIPS } else { MAX_CHAIN_TRIPS };
    let Some(proven) = eval_chain(func, &chain, init_params, global_consts, max_trips) else {
        return false;
    };
    let trips = match &chain.kind {
        LoopChainKind::TopTested { .. } => proven,
        LoopChainKind::BottomTested { .. } => proven + 1,
    };
    let head_len = chain.segs[0].non_term.len() + 1;
    let fits_legacy = is_two_seg && total_insts <= 64 && head_len + trips * total_insts <= MAX_FULL_INSTS;
    let fits_chain = trips <= MAX_CHAIN_TRIPS && trips * body_insts <= MAX_CHAIN_UNROLLED_INSTS;
    if trips > 0 && !fits_legacy && !fits_chain {
        return false;
    }
    let skip: Vec<usize> = chain.segs.iter().map(|s| s.blk.index()).collect();
    match chain.kind {
        LoopChainKind::TopTested {
            exit_blk, exit_args, ..
        } => {
            if trips == 0 {
                let last = *func.blocks[b].insts.last().unwrap_or(&Inst(0));
                func.insts[last.index()].op = Op::Jump(exit_blk, exit_args);
                for seg in &chain.segs[1..] {
                    retire_block(func, seg.blk.index());
                }
                return true;
            }
            pop_terminator(func, b);
            let mut head_map = Vec::new();
            let mut cur_body_args = chain.segs[0].edge_args.clone();
            for _ in 0..trips {
                let mut iter_map = head_map.clone();
                for seg in &chain.segs[1..] {
                    push_params_map(&mut iter_map, &seg.params, &cur_body_args);
                    clone_body(func, &seg.non_term, head, &mut iter_map);
                    cur_body_args = seg.edge_args.iter().map(|&v| remap_val(&iter_map, v)).collect();
                }
                let mut next_head_map = Vec::new();
                push_params_map(&mut next_head_map, &chain.segs[0].params, &cur_body_args);
                clone_body(func, &chain.segs[0].non_term, head, &mut next_head_map);
                head_map = next_head_map;
                cur_body_args = chain.segs[0].edge_args.iter().map(|&v| remap_val(&head_map, v)).collect();
            }
            let final_exit_args: Vec<Value> = exit_args.iter().map(|&v| remap_val(&head_map, v)).collect();
            let jump = func.add(Op::Jump(exit_blk, final_exit_args), None);
            func.blocks[b].insts.push(jump);
            remap_other_blocks(func, &skip, &head_map);
            for seg in &chain.segs[1..] {
                retire_block(func, seg.blk.index());
            }
            true
        }
        LoopChainKind::BottomTested {
            exit_blk, exit_args, ..
        } => {
            pop_terminator(func, b);
            let mut last_map = Vec::new();
            let mut cur_edge_args = chain.segs[0].edge_args.clone();
            for seg in &chain.segs[1..] {
                push_params_map(&mut last_map, &seg.params, &cur_edge_args);
                clone_body(func, &seg.non_term, head, &mut last_map);
                cur_edge_args = seg.edge_args.iter().map(|&v| remap_val(&last_map, v)).collect();
            }
            for _ in 0..proven {
                let mut step_map = Vec::new();
                for seg in &chain.segs {
                    push_params_map(&mut step_map, &seg.params, &cur_edge_args);
                    clone_body(func, &seg.non_term, head, &mut step_map);
                    cur_edge_args = seg.edge_args.iter().map(|&v| remap_val(&step_map, v)).collect();
                }
                last_map = step_map;
            }
            let final_exit_args: Vec<Value> = exit_args.iter().map(|&v| remap_val(&last_map, v)).collect();
            let jump = func.add(Op::Jump(exit_blk, final_exit_args), None);
            func.blocks[b].insts.push(jump);
            remap_other_blocks(func, &skip, &last_map);
            for seg in &chain.segs[1..] {
                retire_block(func, seg.blk.index());
            }
            true
        }
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

/// Unrolls clean 1-block and straight-line inner loops with constant or dynamic trip counts, and constant-trip loops whose bodies branch or leave early.
pub(crate) fn run(func: &mut Function) -> Domain {
    if func.coroutine.is_some() {
        return Domain::NONE;
    }
    let Some(entry) = func.entry else { return Domain::NONE };
    let initial = func.cfg();
    let loops = initial.loops();
    let mut cfg = std::sync::Arc::clone(&initial);
    let mut global_consts = collect_global_consts(func, &cfg, evaluate_wrapped);
    let mut changed = false;
    let mut dirty = false;
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
        let did = if each.latches.len() != 1 {
            region::unroll(func, &cfg, each)
        } else if each.size == 1 && each.latches[0] == head {
            if has_bad_non_term(func, b) {
                false
            } else if let Some(&last) = func.blocks[b].insts.last()
                && let Op::Brif(cond, t_blk, t_args, f_blk, f_args) = &func.insts[last.index()].op
                && t_blk != f_blk
            {
                let (cond, t_blk, t_args, f_blk, f_args) = (*cond, *t_blk, t_args.clone(), *f_blk, f_args.clone());
                if t_blk == head {
                    memset::run(func, b, (cond, true), &t_args, (f_blk, &f_args), &global_consts)
                        || vector::run(func, b, (cond, true), &t_args, (f_blk, &f_args), &global_consts)
                        || unroll_self(func, &cfg, b, cond, true, t_args, f_blk, f_args, &global_consts)
                } else if f_blk == head {
                    memset::run(func, b, (cond, false), &f_args, (t_blk, &t_args), &global_consts)
                        || vector::run(func, b, (cond, false), &f_args, (t_blk, &t_args), &global_consts)
                        || unroll_self(func, &cfg, b, cond, false, f_args, t_blk, t_args, &global_consts)
                } else {
                    false
                }
            } else {
                false
            }
        } else if each.size >= 2 && each.latches[0] != head {
            vector::flatten(func, each, &global_consts) || unroll_chain(func, &cfg, each, &global_consts) || region::unroll(func, &cfg, each)
        } else {
            false
        };
        changed |= did;
        dirty |= did;
    }
    if changed {
        func.bump_cfg();
        Domain::CFG | Domain::VALUES | Domain::MEM
    } else {
        Domain::NONE
    }
}

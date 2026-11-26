mod chain;
mod count;
mod memset;
mod region;
mod self_loop;
mod symbolic;
mod vector;
mod vector_shape;

use crate::cfg::{Cfg, Loop};
use crate::fold::{evaluate_wrapped, extend};
use crate::ir::{Block, Domain, Function, Inst, Op, Type, Value};
use crate::{Arch, Target};
pub(crate) use memset::Emit;
pub(crate) use vector::splat_in;

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
    map.iter()
        .rev()
        .find(|&&(old, _)| old == v)
        .map_or(v, |&(_, new)| new)
}

fn is_libcall_mem(op: &Op) -> bool {
    matches!(*op, Op::Copy(..))
        || matches!(*op, Op::Move(_, _, bytes) if bytes > crate::copy::MOVE_INLINE)
}

pub(super) fn block_has_libcall_mem(func: &Function, block_idx: usize) -> bool {
    let insts = &func.blocks[block_idx].insts;
    insts[..insts.len().saturating_sub(1)]
        .iter()
        .any(|&i| is_libcall_mem(&func.insts[i.index()].op))
}

fn has_bad_non_term(func: &Function, block_idx: usize) -> bool {
    let insts = &func.blocks[block_idx].insts;
    insts[..insts.len() - 1].iter().any(|&i| {
        matches!(
            func.insts[i.index()].op,
            Op::Call { .. } | Op::TryCall { .. } | Op::BrTable(..) | Op::Jump(..) | Op::Brif(..)
        )
    })
}

fn eval_block(
    func: &Function,
    block_idx: usize,
    params: &[Option<i64>],
    env: &mut [Option<i64>],
    eval: Eval,
) {
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

fn clone_body(
    func: &mut Function,
    src_non_term: &[Inst],
    dst: Block,
    map: &mut Vec<(Value, Value)>,
) {
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
            func.insts[inst.index()]
                .op
                .map_operands(|v| remap_val(map, v));
        }
    }
}

fn pop_terminator(func: &mut Function, blk_idx: usize) {
    if let Some(popped) = func.blocks[blk_idx].insts.pop() {
        func.insts[popped.index()].op = Op::Trap;
    }
}

fn edge_args(func: &Function, pred: Block, head: Block) -> Option<&[Value]> {
    match func.last(pred)? {
        Op::Jump(target, args) if *target == head => Some(args.as_slice()),
        Op::Brif(_, t_blk, t_args, f_blk, _) if *t_blk == head && *f_blk != head => {
            Some(t_args.as_slice())
        }
        Op::Brif(_, t_blk, _, f_blk, f_args) if *f_blk == head && *t_blk != head => {
            Some(f_args.as_slice())
        }
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
        let blk = Block(p as u32);
        if p == head.index()
            || latches.contains(&blk)
            || !cfg.reachable.get(p).copied().unwrap_or(false)
            || !func.successors(blk).any(|t| t == head)
        {
            continue;
        }
        let args = edge_args(func, blk, head).filter(|a| a.len() == param_len)?;
        meet(&mut acc, args, global_consts);
    }
    acc
}

pub(super) fn block_has_vec(func: &Function, blk: usize) -> bool {
    func.blocks[blk].insts.iter().any(|&i| {
        func.insts[i.index()]
            .result
            .is_some_and(|r| func.ty(r).is_vector())
    })
}

pub(super) fn block_has_float(func: &Function, blk: usize) -> bool {
    func.blocks[blk].insts.iter().any(|&i| {
        func.insts[i.index()]
            .result
            .is_some_and(|r| func.ty(r).is_float())
    })
}

fn push_params_map(map: &mut Vec<(Value, Value)>, params: &[Value], args: &[Value]) {
    for (&p, &a) in params.iter().zip(args.iter()) {
        map.push((p, a));
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

fn unroll_single_block(
    func: &mut Function,
    cfg: &Cfg,
    b: usize,
    global_consts: &[Option<i64>],
    sad: bool,
) -> bool {
    if has_bad_non_term(func, b) {
        return false;
    }
    let Some(&last) = func.blocks[b].insts.last() else {
        return false;
    };
    let Op::Brif(cond, t_blk, t_args, f_blk, f_args) = &func.insts[last.index()].op else {
        return false;
    };
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
    memset::run(
        func,
        b,
        (cond, back_on_true),
        &back_args,
        (exit_blk, &exit_args),
        global_consts,
    ) || vector::run(
        func,
        b,
        (cond, back_on_true),
        &back_args,
        (exit_blk, &exit_args),
        (global_consts, sad),
    ) || self_loop::unroll_self(
        func,
        cfg,
        b,
        cond,
        back_on_true,
        back_args,
        exit_blk,
        exit_args,
        global_consts,
    )
}

fn unroll_loop_step(
    func: &mut Function,
    cfg: &Cfg,
    each: &Loop,
    global_consts: &[Option<i64>],
    sad: bool,
) -> bool {
    let head = each.head;
    if each.latches.len() != 1 {
        region::unroll(func, cfg, each)
    } else if each.size == 1 && each.latches[0] == head {
        unroll_single_block(func, cfg, head.index(), global_consts, sad)
    } else if each.size >= 2 && each.latches[0] != head {
        vector::flatten(func, each, (global_consts, sad))
            || chain::unroll_chain(func, cfg, each, global_consts)
            || region::unroll(func, cfg, each)
    } else {
        false
    }
}

/// Unrolls clean 1-block and straight-line inner loops with constant or dynamic trip counts, and constant-trip loops whose bodies branch or leave early.
pub(crate) fn run(func: &mut Function, target: &Target) -> Domain {
    let sad = target.arch == Arch::X64;
    let Some(entry) = func.entry.filter(|_| func.coroutine.is_none()) else {
        return Domain::NONE;
    };
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

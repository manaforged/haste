mod entry_zeros;
mod frontier;

use crate::cfg::Cfg;
use crate::ir::{Block, Function, Inst, Op, StackSlot};

pub(super) use self::entry_zeros::{entry_first_zeros, sink_entry_zeros};
pub(crate) use self::frontier::sink_frontier;
use self::frontier::{
    call_bounded_blocks, dominated_clean_region, prune_redundant_frontier, sink_frontier_or_solo,
    solve_liveness,
};
use super::{CalleeKind, FlagOp, classify_flag_op, store_clears_active};

pub(super) fn flag_call_liveness(
    func: &Function,
    cfg: &Cfg,
    defs: &[Option<Inst>],
    kinds: &[CalleeKind],
    removed: &[bool],
    flag: StackSlot,
) -> (Vec<bool>, Vec<bool>) {
    let n = func.blocks.len();
    let (mut gen_b, mut kill_b) = (vec![false; n], vec![false; n]);
    for &b in &cfg.rpo {
        let bi = b.index();
        for &inst in &func.blocks[bi].insts {
            if removed[inst.index()] {
                continue;
            }
            match classify_flag_op(func, defs, kinds, flag, inst) {
                Some(FlagOp::MayUnwindCall | FlagOp::TryCall | FlagOp::Load) => {
                    gen_b[bi] = true;
                    break;
                }
                Some(FlagOp::Exit | FlagOp::Clear) => {
                    kill_b[bi] = true;
                    break;
                }
                Some(FlagOp::Set) | None => {}
            }
        }
    }
    solve_liveness(func, cfg, &gen_b, &kill_b)
}

fn block_is_flag_clean(func: &Function, kinds: &[CalleeKind], b: Block, flag: StackSlot) -> bool {
    func.blocks[b.index()]
        .insts
        .iter()
        .all(|&inst| match func.insts[inst.index()].op {
            Op::Call { .. } if kinds[inst.index()] == CalleeKind::MayUnwind => false,
            Op::TryCall { .. } => false,
            Op::StackLoad(s, _) | Op::StackStore(_, s, _) if s == flag => false,
            _ => true,
        })
}

pub(super) fn plan_flag_sink(
    func: &Function,
    cfg: &Cfg,
    kinds: &[CalleeKind],
    depths: &[u32],
    (live_in, always_active, b, sinst, flag): (&[bool], (&[bool], &[bool]), Block, Inst, StackSlot),
) -> Option<Vec<Block>> {
    let insts = &func.blocks[b.index()].insts;
    let pos = insts.iter().position(|&i| i == sinst)?;
    if insts[pos + 1..]
        .iter()
        .any(|&i| match func.insts[i.index()].op {
            Op::Call { .. } if kinds[i.index()] == CalleeKind::MayUnwind => true,
            Op::TryCall { .. } | Op::Return(..) | Op::Trap => true,
            Op::StackLoad(s, _) | Op::StackStore(_, s, _) if s == flag => true,
            _ => false,
        })
    {
        return None;
    }
    let call_bounded = call_bounded_blocks(func, cfg, kinds);
    let mut in_r = dominated_clean_region(func, cfg, b, live_in, |s| {
        block_is_flag_clean(func, kinds, s, flag)
    });
    let mut frontier = sink_frontier_or_solo(
        func,
        cfg,
        depths,
        (live_in, Some(always_active), Some(&call_bounded), b, 8),
        &mut in_r,
    )?;
    prune_redundant_frontier(
        func,
        cfg,
        live_in,
        |s| block_is_flag_clean(func, kinds, s, flag),
        &mut frontier,
    );
    Some(frontier)
}

fn range_covers(w_off: i32, w_bytes: u32, off: i32, bytes: u32) -> bool {
    let (w0, s0) = (i64::from(w_off), i64::from(off));
    w0 <= s0 && w0 + i64::from(w_bytes) >= s0 + i64::from(bytes)
}

fn range_overlaps(a_off: i32, a_bytes: u32, b_off: i32, b_bytes: u32) -> bool {
    let (a0, b0) = (i64::from(a_off), i64::from(b_off));
    a0 < b0 + i64::from(b_bytes) && b0 < a0 + i64::from(a_bytes)
}

enum LocalOp {
    Use,
    CoverStore,
    OverlapStore,
    Exit,
}

fn sets_flag_op(func: &Function, defs: &[Option<Inst>], flag: StackSlot, op: &Op) -> bool {
    matches!(*op, Op::StackStore(v, s, 0) if s == flag && !store_clears_active(func, defs, v))
}

fn classify_local_op(
    func: &Function,
    defs: &[Option<Inst>],
    (local, flag): (StackSlot, StackSlot),
    (store_off, store_bytes): (i32, u32),
    inst: Inst,
) -> Option<LocalOp> {
    let data = &func.insts[inst.index()];
    match data.op {
        Op::StackLoad(s, roff)
            if s == local
                && data.result.is_none_or(|r| {
                    range_overlaps(roff, func.ty(r).bytes(), store_off, store_bytes)
                }) =>
        {
            Some(LocalOp::Use)
        }
        Op::StackStore(v, s, woff)
            if s == local && range_covers(woff, func.ty(v).bytes(), store_off, store_bytes) =>
        {
            Some(LocalOp::CoverStore)
        }
        Op::StackStore(v, s, woff)
            if s == local && range_overlaps(woff, func.ty(v).bytes(), store_off, store_bytes) =>
        {
            Some(LocalOp::OverlapStore)
        }
        Op::Return(..) | Op::Trap => Some(LocalOp::Exit),
        ref op if sets_flag_op(func, defs, flag, op) => Some(LocalOp::Use),
        _ => None,
    }
}

pub(super) fn plan_local_sink(
    func: &Function,
    cfg: &Cfg,
    defs: &[Option<Inst>],
    depths: &[u32],
    (flag_visited, local, flag): (&[u8], StackSlot, StackSlot),
    (b, sinst, store_off, store_bytes): (Block, Inst, i32, u32),
) -> Option<Vec<Block>> {
    if flag_visited[b.index()] != 1 {
        return None;
    }
    let insts = &func.blocks[b.index()].insts;
    let pos = insts.iter().position(|&i| i == sinst)?;
    if insts[..pos]
        .iter()
        .any(|&i| sets_flag_op(func, defs, flag, &func.insts[i.index()].op))
    {
        return None;
    }
    for &i in &insts[pos + 1..] {
        match classify_local_op(func, defs, (local, flag), (store_off, store_bytes), i) {
            Some(LocalOp::Use | LocalOp::OverlapStore) => return None,
            Some(LocalOp::CoverStore | LocalOp::Exit) => return Some(Vec::new()),
            None => {}
        }
    }
    let n = func.blocks.len();
    let (mut gen_b, mut kill_b, mut clean_b) = (vec![false; n], vec![false; n], vec![true; n]);
    for &blk in &cfg.rpo {
        let bi = blk.index();
        if flag_visited[bi] != 1 {
            gen_b[bi] = true;
            clean_b[bi] = false;
            continue;
        }
        for &inst in &func.blocks[bi].insts {
            match classify_local_op(func, defs, (local, flag), (store_off, store_bytes), inst) {
                Some(LocalOp::Use) => {
                    gen_b[bi] = true;
                    clean_b[bi] = false;
                    break;
                }
                Some(LocalOp::CoverStore) => {
                    kill_b[bi] = true;
                    clean_b[bi] = false;
                    break;
                }
                Some(LocalOp::OverlapStore) => clean_b[bi] = false,
                Some(LocalOp::Exit) => {
                    kill_b[bi] = true;
                    break;
                }
                None => {}
            }
        }
    }
    let (live_in, _) = solve_liveness(func, cfg, &gen_b, &kill_b);
    if !func.successors(b).any(|s| live_in[s.index()]) {
        return Some(Vec::new());
    }
    let mut in_r = dominated_clean_region(func, cfg, b, &live_in, |s| clean_b[s.index()]);
    sink_frontier(func, cfg, depths, (&live_in, None, b), &mut in_r)
}

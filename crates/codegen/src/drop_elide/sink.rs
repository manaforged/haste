use crate::cfg::Cfg;
use crate::ir::{Block, Function, Inst, Op, StackSlot};

use super::{CalleeKind, store_clears_active};

pub(super) fn flag_call_liveness(func: &Function, cfg: &Cfg, defs: &[Option<Inst>], kinds: &[CalleeKind], flag: StackSlot) -> (Vec<bool>, Vec<bool>) {
    let n = func.blocks.len();
    let (mut gen_b, mut kill_b) = (vec![false; n], vec![false; n]);
    for &b in &cfg.rpo {
        let bi = b.index();
        for &inst in &func.blocks[bi].insts {
            match func.insts[inst.index()].op {
                Op::Call { .. } if kinds[inst.index()] == CalleeKind::MayUnwind => {
                    gen_b[bi] = true;
                    break;
                }
                Op::TryCall { .. } => {
                    gen_b[bi] = true;
                    break;
                }
                Op::StackLoad(s, _) if s == flag => {
                    gen_b[bi] = true;
                    break;
                }
                Op::Return(..) | Op::Trap => {
                    kill_b[bi] = true;
                    break;
                }
                Op::StackStore(v, s, 0) if s == flag && store_clears_active(func, defs, v) => {
                    kill_b[bi] = true;
                    break;
                }
                _ => {}
            }
        }
    }
    let (mut live_in, mut live_out, mut changed) = (gen_b.clone(), vec![false; n], true);
    while changed {
        changed = false;
        for &b in cfg.rpo.iter().rev() {
            let bi = b.index();
            let out = func.successors(b).any(|s| live_in[s.index()]);
            let inp = gen_b[bi] || (!kill_b[bi] && out);
            if out != live_out[bi] || inp != live_in[bi] {
                (live_out[bi], live_in[bi], changed) = (out, inp, true);
            }
        }
    }
    (live_in, live_out)
}

fn block_is_flag_clean(func: &Function, kinds: &[CalleeKind], b: Block, flag: StackSlot) -> bool {
    func.blocks[b.index()].insts.iter().all(|&inst| match func.insts[inst.index()].op {
        Op::Call { .. } if kinds[inst.index()] == CalleeKind::MayUnwind => false,
        Op::TryCall { .. } => false,
        Op::StackLoad(s, _) | Op::StackStore(_, s, _) if s == flag => false,
        _ => true,
    })
}

fn block_has_unwind_call(func: &Function, kinds: &[CalleeKind], b: Block) -> bool {
    func.blocks[b.index()].insts.iter().any(|&inst| match func.insts[inst.index()].op {
        Op::Call { .. } => kinds[inst.index()] == CalleeKind::MayUnwind,
        Op::TryCall { .. } => true,
        _ => false,
    })
}

fn target_ok(cfg: &Cfg, depths: &[u32], always_active: Option<&[bool]>, call_in_loop: Option<&[bool]>, b: Block, in_r: &[bool], s: Block) -> bool {
    let si = s.index();
    let depth_ok = depths[si] <= depths[b.index()] || call_in_loop.is_some_and(|c| c[si]);
    depth_ok
        && (cfg.preds_of(s).iter().all(|&p| !cfg.reachable[p.index()] || in_r[p.index()])
            || (always_active.is_some_and(|aa| aa[si]) && cfg.dominates(b, s)))
}

/// Prunes a clean dominated region `in_r` starting at `b` to blocks reaching a `!live_in` fast exit and returns the `<= 4` cold frontier blocks.
pub(crate) fn sink_frontier(
    func: &Function,
    cfg: &Cfg,
    depths: &[u32],
    (live_in, always_active, b): (&[bool], Option<&[bool]>, Block),
    in_r: &mut [bool],
) -> Option<Vec<Block>> {
    sink_frontier_with_calls(func, cfg, depths, (live_in, always_active, None, b), in_r)
}

fn sink_frontier_with_calls(
    func: &Function,
    cfg: &Cfg,
    depths: &[u32],
    (live_in, always_active, call_in_loop, b): (&[bool], Option<&[bool]>, Option<&[bool]>, Block),
    in_r: &mut [bool],
) -> Option<Vec<Block>> {
    let n = func.blocks.len();
    let mut members: Vec<Block> = in_r.iter().enumerate().filter_map(|(bi, &m)| m.then_some(Block(bi as u32))).collect();
    loop {
        let mut pruned = false;
        members.retain(|&blk| {
            let bi = blk.index();
            if bi == b.index() {
                return true;
            }
            let bad_pred = cfg.preds_of(blk).iter().any(|&p| {
                cfg.reachable[p.index()]
                    && !in_r[p.index()]
                    && !(always_active.is_some_and(|aa| aa[p.index()]) && cfg.dominates(b, p))
            });
            let bad_succ = func.successors(blk).any(|s| !in_r[s.index()] && live_in[s.index()] && !target_ok(cfg, depths, always_active, call_in_loop, b, in_r, s));
            if bad_pred || bad_succ {
                (in_r[bi], pruned) = (false, true);
                false
            } else {
                true
            }
        });
        if !pruned {
            break;
        }
    }
    let (mut useful, mut q) = (vec![false; n], Vec::new());
    for &blk in &members {
        if func.successors(blk).any(|s| !in_r[s.index()] && !live_in[s.index()]) {
            useful[blk.index()] = true;
            q.push(blk);
        }
    }
    if q.is_empty() {
        return None;
    }
    while let Some(u) = q.pop() {
        for &p in cfg.preds_of(u) {
            if in_r[p.index()] && !useful[p.index()] {
                useful[p.index()] = true;
                q.push(p);
            }
        }
    }
    members.retain(|&blk| {
        let bi = blk.index();
        if bi != b.index() && !useful[bi] {
            in_r[bi] = false;
            false
        } else {
            true
        }
    });
    let (mut frontier, mut skipped_fast) = (Vec::new(), false);
    for &blk in &members {
        for s in func.successors(blk) {
            if in_r[s.index()] {
                continue;
            }
            if !live_in[s.index()] {
                skipped_fast = true;
            } else if !frontier.contains(&s) {
                frontier.push(s);
            }
        }
    }
    if !skipped_fast || frontier.is_empty() || frontier.len() > 4 {
        return None;
    }
    frontier.iter().all(|&t| target_ok(cfg, depths, always_active, call_in_loop, b, in_r, t)).then_some(frontier)
}

pub(super) fn plan_flag_sink(
    func: &Function,
    cfg: &Cfg,
    kinds: &[CalleeKind],
    depths: &[u32],
    (live_in, always_active, b, sinst, flag): (&[bool], &[bool], Block, Inst, StackSlot),
) -> Option<Vec<Block>> {
    let insts = &func.blocks[b.index()].insts;
    let pos = insts.iter().position(|&i| i == sinst)?;
    if insts[pos + 1..].iter().any(|&i| match func.insts[i.index()].op {
        Op::Call { .. } if kinds[i.index()] == CalleeKind::MayUnwind => true,
        Op::TryCall { .. } | Op::Return(..) | Op::Trap => true,
        Op::StackLoad(s, _) | Op::StackStore(_, s, _) if s == flag => true,
        _ => false,
    }) {
        return None;
    }
    let n = func.blocks.len();
    let has_call: Vec<bool> = (0..n).map(|bi| block_has_unwind_call(func, kinds, Block(bi as u32))).collect();
    let (mut in_r, mut work) = (vec![false; n], vec![b]);
    in_r[b.index()] = true;
    while let Some(u) = work.pop() {
        for s in func.successors(u) {
            let si = s.index();
            if !in_r[si] && cfg.reachable[si] && cfg.dominates(b, s) && live_in[si] && block_is_flag_clean(func, kinds, s, flag) {
                in_r[si] = true;
                work.push(s);
            }
        }
    }
    sink_frontier_with_calls(func, cfg, depths, (live_in, Some(always_active), Some(&has_call), b), &mut in_r).or_else(|| {
        let mut solo = vec![false; n];
        solo[b.index()] = true;
        sink_frontier_with_calls(func, cfg, depths, (live_in, Some(always_active), Some(&has_call), b), &mut solo)
    })
}

fn range_covers(w_off: i32, w_bytes: u32, off: i32, bytes: u32) -> bool {
    let (w0, s0) = (i64::from(w_off), i64::from(off));
    w0 <= s0 && w0 + i64::from(w_bytes) >= s0 + i64::from(bytes)
}

fn range_overlaps(a_off: i32, a_bytes: u32, b_off: i32, b_bytes: u32) -> bool {
    let (a0, b0) = (i64::from(a_off), i64::from(b_off));
    a0 < b0 + i64::from(b_bytes) && b0 < a0 + i64::from(a_bytes)
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
    let sets_flag = |op: &Op| matches!(*op, Op::StackStore(v, s, 0) if s == flag && !store_clears_active(func, defs, v));
    if insts[..pos].iter().any(|&i| sets_flag(&func.insts[i.index()].op)) {
        return None;
    }
    for &i in &insts[pos + 1..] {
        let data = &func.insts[i.index()];
        match data.op {
            Op::StackLoad(s, roff) if s == local && data.result.is_none_or(|r| range_overlaps(roff, func.ty(r).bytes(), store_off, store_bytes)) => {
                return None;
            }
            Op::StackStore(v, s, woff) if s == local && range_covers(woff, func.ty(v).bytes(), store_off, store_bytes) => {
                return Some(Vec::new());
            }
            Op::StackStore(v, s, woff) if s == local && range_overlaps(woff, func.ty(v).bytes(), store_off, store_bytes) => {
                return None;
            }
            Op::Return(..) | Op::Trap => return Some(Vec::new()),
            ref op if sets_flag(op) => return None,
            _ => {}
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
            let data = &func.insts[inst.index()];
            match data.op {
                Op::StackLoad(s, roff) if s == local && data.result.is_none_or(|r| range_overlaps(roff, func.ty(r).bytes(), store_off, store_bytes)) => {
                    gen_b[bi] = true;
                    clean_b[bi] = false;
                    break;
                }
                Op::StackStore(v, s, woff) if s == local && range_covers(woff, func.ty(v).bytes(), store_off, store_bytes) => {
                    kill_b[bi] = true;
                    clean_b[bi] = false;
                    break;
                }
                Op::StackStore(v, s, woff) if s == local && range_overlaps(woff, func.ty(v).bytes(), store_off, store_bytes) => {
                    clean_b[bi] = false;
                }
                Op::Return(..) | Op::Trap => {
                    kill_b[bi] = true;
                    break;
                }
                ref op if sets_flag(op) => {
                    gen_b[bi] = true;
                    clean_b[bi] = false;
                    break;
                }
                _ => {}
            }
        }
    }
    let (mut live_in, mut changed) = (gen_b.clone(), true);
    while changed {
        changed = false;
        for &blk in cfg.rpo.iter().rev() {
            let bi = blk.index();
            let out = func.successors(blk).any(|s| live_in[s.index()]);
            let inp = gen_b[bi] || (!kill_b[bi] && out);
            if inp != live_in[bi] {
                (live_in[bi], changed) = (inp, true);
            }
        }
    }
    if !func.successors(b).any(|s| live_in[s.index()]) {
        return Some(Vec::new());
    }
    let (mut in_r, mut work) = (vec![false; n], vec![b]);
    in_r[b.index()] = true;
    while let Some(u) = work.pop() {
        for s in func.successors(u) {
            let si = s.index();
            if !in_r[si] && cfg.reachable[si] && cfg.dominates(b, s) && live_in[si] && clean_b[si] {
                in_r[si] = true;
                work.push(s);
            }
        }
    }
    sink_frontier(func, cfg, depths, (&live_in, None, b), &mut in_r)
}

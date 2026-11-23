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

fn sink_frontier(func: &Function, cfg: &Cfg, depths: &[u32], (live_in, b): (&[bool], Block), in_r: &mut [bool]) -> Option<Vec<Block>> {
    let n = func.blocks.len();
    loop {
        let mut pruned = false;
        for bi in 0..n {
            if !in_r[bi] || bi == b.index() {
                continue;
            }
            let blk = Block(bi as u32);
            let bad_pred = cfg.preds_of(blk).iter().any(|&p| cfg.reachable[p.index()] && !in_r[p.index()]);
            let bad_succ = func.successors(blk).any(|s| {
                let si = s.index();
                !in_r[si]
                    && live_in[si]
                    && (depths[si] > depths[b.index()] || cfg.preds_of(s).iter().any(|&p| cfg.reachable[p.index()] && !in_r[p.index()]))
            });
            if bad_pred || bad_succ {
                (in_r[bi], pruned) = (false, true);
            }
        }
        if !pruned {
            break;
        }
    }
    let (mut useful, mut q) = (vec![false; n], Vec::new());
    for bi in 0..n {
        if in_r[bi] && func.successors(Block(bi as u32)).any(|s| !in_r[s.index()] && !live_in[s.index()]) {
            useful[bi] = true;
            q.push(Block(bi as u32));
        }
    }
    while let Some(u) = q.pop() {
        for &p in cfg.preds_of(u) {
            if in_r[p.index()] && !useful[p.index()] {
                useful[p.index()] = true;
                q.push(p);
            }
        }
    }
    for bi in 0..n {
        if in_r[bi] && bi != b.index() && !useful[bi] {
            in_r[bi] = false;
        }
    }
    let (mut frontier, mut skipped_fast) = (Vec::new(), false);
    for bi in 0..n {
        if !in_r[bi] {
            continue;
        }
        for s in func.successors(Block(bi as u32)) {
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
    frontier
        .iter()
        .all(|&t| depths[t.index()] <= depths[b.index()] && cfg.preds_of(t).iter().all(|&p| !cfg.reachable[p.index()] || in_r[p.index()]))
        .then_some(frontier)
}

pub(super) fn plan_flag_sink(
    func: &Function,
    cfg: &Cfg,
    kinds: &[CalleeKind],
    depths: &[u32],
    (live_in, b, sinst, flag): (&[bool], Block, Inst, StackSlot),
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
    sink_frontier(func, cfg, depths, (live_in, b), &mut in_r).or_else(|| {
        let mut solo = vec![false; n];
        solo[b.index()] = true;
        sink_frontier(func, cfg, depths, (live_in, b), &mut solo)
    })
}

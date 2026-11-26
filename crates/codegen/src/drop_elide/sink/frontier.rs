use crate::cfg::Cfg;
use crate::ir::{Block, Function, Op};

use super::super::CalleeKind;

pub(super) fn solve_liveness(
    func: &Function,
    cfg: &Cfg,
    gen_b: &[bool],
    kill_b: &[bool],
) -> (Vec<bool>, Vec<bool>) {
    let n = func.blocks.len();
    let (mut live_in, mut live_out, mut changed) = (gen_b.to_vec(), vec![false; n], true);
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

pub(super) fn dominated_clean_region(
    func: &Function,
    cfg: &Cfg,
    root: Block,
    live_in: &[bool],
    mut is_clean: impl FnMut(Block) -> bool,
) -> Vec<bool> {
    let n = func.blocks.len();
    let (mut in_r, mut work) = (vec![false; n], vec![root]);
    in_r[root.index()] = true;
    while let Some(u) = work.pop() {
        for s in func.successors(u) {
            let si = s.index();
            if !in_r[si]
                && cfg.reachable[si]
                && cfg.dominates(root, s)
                && live_in[si]
                && is_clean(s)
            {
                in_r[si] = true;
                work.push(s);
            }
        }
    }
    in_r
}

fn block_has_unwind_call(func: &Function, kinds: &[CalleeKind], b: Block) -> bool {
    func.blocks[b.index()]
        .insts
        .iter()
        .any(|&inst| match func.insts[inst.index()].op {
            Op::Call { .. } => kinds[inst.index()] == CalleeKind::MayUnwind,
            Op::TryCall { .. } => true,
            _ => false,
        })
}

pub(super) fn call_bounded_blocks(func: &Function, cfg: &Cfg, kinds: &[CalleeKind]) -> Vec<bool> {
    let n = func.blocks.len();
    let has_call: Vec<bool> = (0..n)
        .map(|bi| block_has_unwind_call(func, kinds, Block(bi as u32)))
        .collect();
    let mut outdeg = vec![0u32; n];
    let mut ok = has_call.clone();
    let mut q = Vec::new();
    for bi in 0..n {
        if ok[bi] || !cfg.reachable[bi] {
            continue;
        }
        let deg = func
            .successors(Block(bi as u32))
            .filter(|s| cfg.reachable[s.index()] && !has_call[s.index()])
            .count() as u32;
        outdeg[bi] = deg;
        if deg == 0 {
            ok[bi] = true;
            q.push(Block(bi as u32));
        }
    }
    while let Some(u) = q.pop() {
        for &p in cfg.preds_of(u) {
            let pi = p.index();
            if !ok[pi] && cfg.reachable[pi] {
                outdeg[pi] = outdeg[pi].saturating_sub(1);
                if outdeg[pi] == 0 {
                    ok[pi] = true;
                    q.push(p);
                }
            }
        }
    }
    ok
}

fn target_ok(
    cfg: &Cfg,
    depths: &[u32],
    always_in: Option<&[bool]>,
    call_in_loop: Option<&[bool]>,
    b: Block,
    in_r: &[bool],
    s: Block,
) -> bool {
    let si = s.index();
    let depth_ok = depths[si] <= depths[b.index()] || call_in_loop.is_some_and(|c| c[si]);
    depth_ok
        && (cfg
            .preds_of(s)
            .iter()
            .all(|&p| !cfg.reachable[p.index()] || in_r[p.index()])
            || (always_in.is_some_and(|aa| aa[si]) && cfg.dominates(b, s)))
}

/// Prunes a clean dominated region `in_r` starting at `b` to blocks reaching a `!live_in` fast exit and returns the `<= 4` cold frontier blocks.
pub(crate) fn sink_frontier(
    func: &Function,
    cfg: &Cfg,
    depths: &[u32],
    (live_in, always_active, b): (&[bool], Option<&[bool]>, Block),
    in_r: &mut [bool],
) -> Option<Vec<Block>> {
    sink_frontier_with_calls(
        func,
        cfg,
        depths,
        (live_in, always_active.map(|aa| (aa, aa)), None, b, 4),
        in_r,
    )
}

pub(super) fn cycle_depths(func: &Function, cfg: &Cfg) -> Vec<u32> {
    let n = func.blocks.len();
    let mut depths = cfg.loops().depths.clone();
    let mut seen = vec![false; n];
    let mut order = Vec::with_capacity(n);
    let mut stack = Vec::new();
    for start in 0..n {
        if !cfg.reachable[start] || seen[start] {
            continue;
        }
        stack.push((start, false));
        while let Some((u, done)) = stack.pop() {
            if done {
                order.push(u);
                continue;
            }
            if std::mem::replace(&mut seen[u], true) {
                continue;
            }
            stack.push((u, true));
            for s in func.successors(Block(u as u32)) {
                let si = s.index();
                if cfg.reachable[si] && !seen[si] {
                    stack.push((si, false));
                }
            }
        }
    }
    let mut visited = vec![false; n];
    let mut comp = Vec::new();
    let mut rev_stack = Vec::new();
    for &start in order.iter().rev() {
        if visited[start] {
            continue;
        }
        comp.clear();
        visited[start] = true;
        rev_stack.push(start);
        while let Some(u) = rev_stack.pop() {
            comp.push(u);
            for &p in cfg.preds_of(Block(u as u32)) {
                let pi = p.index();
                if cfg.reachable[pi] && !visited[pi] {
                    visited[pi] = true;
                    rev_stack.push(pi);
                }
            }
        }
        let cyclic = comp.len() > 1
            || func
                .successors(Block(start as u32))
                .any(|s| s.index() == start);
        if cyclic {
            for &u in &comp {
                depths[u] = depths[u].max(1);
            }
        }
    }
    depths
}

pub(super) fn prune_redundant_frontier(
    func: &Function,
    cfg: &Cfg,
    live_in: &[bool],
    mut is_clean: impl FnMut(Block) -> bool,
    frontier: &mut Vec<Block>,
) {
    loop {
        let Some(pos) = frontier.iter().position(|&t| {
            is_clean(t)
                && func.successors(t).all(|s| {
                    !cfg.reachable[s.index()]
                        || !live_in[s.index()]
                        || (s != t && frontier.contains(&s))
                })
        }) else {
            break;
        };
        frontier.remove(pos);
    }
}

pub(super) fn sink_frontier_or_solo(
    func: &Function,
    cfg: &Cfg,
    depths: &[u32],
    args: (
        &[bool],
        Option<(&[bool], &[bool])>,
        Option<&[bool]>,
        Block,
        usize,
    ),
    in_r: &mut [bool],
) -> Option<Vec<Block>> {
    sink_frontier_with_calls(func, cfg, depths, args, in_r).or_else(|| {
        let mut solo = vec![false; func.blocks.len()];
        solo[args.3.index()] = true;
        sink_frontier_with_calls(func, cfg, depths, args, &mut solo)
    })
}

fn prune_sink_region(
    func: &Function,
    cfg: &Cfg,
    depths: &[u32],
    (live_in, always_ready, call_in_loop, b): (
        &[bool],
        Option<(&[bool], &[bool])>,
        Option<&[bool]>,
        Block,
    ),
    in_r: &mut [bool],
) -> Option<Vec<Block>> {
    let always_in = always_ready.map(|(ai, _)| ai);
    let always_out = always_ready.map(|(_, ao)| ao);
    let n = func.blocks.len();
    let mut members: Vec<Block> = in_r
        .iter()
        .enumerate()
        .filter_map(|(bi, &m)| m.then_some(Block(bi as u32)))
        .collect();
    loop {
        let before = members.len();
        members.retain(|&blk| {
            let bi = blk.index();
            if bi == b.index() {
                return true;
            }
            let bad_pred = cfg.preds_of(blk).iter().any(|&p| {
                cfg.reachable[p.index()]
                    && !in_r[p.index()]
                    && !(always_out.is_some_and(|ao| ao[p.index()]) && cfg.dominates(b, p))
            });
            let bad_succ = func.successors(blk).any(|s| {
                !in_r[s.index()]
                    && live_in[s.index()]
                    && !target_ok(cfg, depths, always_in, call_in_loop, b, in_r, s)
            });
            if bad_pred || bad_succ {
                in_r[bi] = false;
                false
            } else {
                true
            }
        });
        if members.len() == before {
            break;
        }
    }
    let (mut useful, mut q) = (vec![false; n], Vec::new());
    for &blk in &members {
        if func
            .successors(blk)
            .any(|s| !in_r[s.index()] && !live_in[s.index()])
        {
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
    if !skipped_fast || frontier.is_empty() {
        return None;
    }
    frontier
        .iter()
        .all(|&t| target_ok(cfg, depths, always_in, call_in_loop, b, in_r, t))
        .then_some(frontier)
}

fn bfs_prefix_contract(
    func: &Function,
    cfg: &Cfg,
    depths: &[u32],
    (live_in, always_ready, call_in_loop, b, max_frontier): (
        &[bool],
        Option<(&[bool], &[bool])>,
        Option<&[bool]>,
        Block,
        usize,
    ),
    in_r: &[bool],
) -> Option<(Vec<bool>, Vec<Block>)> {
    let n = func.blocks.len();
    let mut dist = vec![u32::MAX; n];
    dist[b.index()] = 0;
    let mut q = Vec::with_capacity(n.min(64));
    q.push(b);
    let mut head = 0;
    let mut max_d = 0u32;
    let mut cand_ds = Vec::new();
    while head < q.len() {
        let u = q[head];
        head += 1;
        let d = dist[u.index()];
        max_d = max_d.max(d);
        let mut has_fast_exit = false;
        for s in func.successors(u) {
            let si = s.index();
            if in_r[si] {
                if dist[si] == u32::MAX {
                    dist[si] = d + 1;
                    q.push(s);
                }
            } else if !live_in[si] {
                has_fast_exit = true;
            }
        }
        if has_fast_exit {
            cand_ds.push(d);
            cand_ds.push(d + 1);
        }
    }
    cand_ds.sort_unstable();
    cand_ds.dedup();
    let mut best: Option<(Vec<bool>, Vec<Block>)> = None;
    for &d in cand_ds.iter().rev().filter(|&&d| d < max_d).take(24) {
        let mut trial = vec![false; n];
        for &u in &q {
            if dist[u.index()] <= d {
                trial[u.index()] = true;
            }
        }
        if let Some(tf) = prune_sink_region(
            func,
            cfg,
            depths,
            (live_in, always_ready, call_in_loop, b),
            &mut trial,
        ) {
            if tf.len() <= max_frontier {
                return Some((trial, tf));
            }
            if best.as_ref().is_none_or(|(_, bf)| tf.len() < bf.len()) {
                best = Some((trial, tf));
            }
        }
    }
    best
}

fn sink_frontier_with_calls(
    func: &Function,
    cfg: &Cfg,
    depths: &[u32],
    (live_in, always_ready, call_in_loop, b, max_frontier): (
        &[bool],
        Option<(&[bool], &[bool])>,
        Option<&[bool]>,
        Block,
        usize,
    ),
    in_r: &mut [bool],
) -> Option<Vec<Block>> {
    let mut frontier = prune_sink_region(
        func,
        cfg,
        depths,
        (live_in, always_ready, call_in_loop, b),
        in_r,
    )?;
    if frontier.len() <= max_frontier {
        return Some(frontier);
    }
    let mut cur = in_r.to_vec();
    if let Some((pref_r, pref_f)) = bfs_prefix_contract(
        func,
        cfg,
        depths,
        (live_in, always_ready, call_in_loop, b, max_frontier),
        &cur,
    ) {
        if pref_f.len() <= max_frontier {
            in_r.copy_from_slice(&pref_r);
            return Some(pref_f);
        }
        if pref_f.len() < frontier.len() {
            cur = pref_r;
            frontier = pref_f;
        }
    }
    if cur.iter().filter(|&&m| m).count() > 64 {
        return None;
    }
    while frontier.len() > max_frontier {
        let mut best: Option<(Vec<bool>, Vec<Block>)> = None;
        for bi in 0..cur.len() {
            if !cur[bi] || bi == b.index() {
                continue;
            }
            let mut trial = cur.clone();
            trial[bi] = false;
            if let Some(tf) = prune_sink_region(
                func,
                cfg,
                depths,
                (live_in, always_ready, call_in_loop, b),
                &mut trial,
            ) && tf.len() < best.as_ref().map_or(frontier.len(), |(_, bf)| bf.len())
            {
                best = Some((trial, tf));
            }
        }
        let Some((next_r, next_f)) = best else {
            return None;
        };
        cur = next_r;
        frontier = next_f;
    }
    in_r.copy_from_slice(&cur);
    Some(frontier)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::FunctionBuilder;
    use crate::ir::{AbiParam, CallConv, Signature, Type};

    #[test]
    fn cycle_depths_does_not_merge_acyclic_diamond() {
        let mut sig = Signature::new(CallConv::Native);
        sig.params = vec![AbiParam::new(Type::I8)];
        let mut func = Function::new(sig);
        let mut b = FunctionBuilder::new(&mut func);
        let b0 = b.create_block();
        let b1 = b.create_block();
        let b2 = b.create_block();
        b.append_block_params_for_function_params(b0);
        b.switch_to_block(b0);
        let cond = b.block_params(b0)[0];
        b.ins().brif(cond, b1, &[], b2, &[]);
        b.switch_to_block(b2);
        b.ins().jump(b1, &[]);
        b.switch_to_block(b1);
        b.ins().return_(&[]);
        let cfg = func.cfg();
        assert_eq!(cycle_depths(&func, &cfg), vec![0, 0, 0]);
    }
}

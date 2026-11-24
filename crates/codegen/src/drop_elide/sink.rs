use crate::cfg::Cfg;
use crate::ir::{Block, Function, Inst, Op, StackSlot, Value};

use super::{CalleeKind, FlagOp, classify_callee, classify_flag_op, store_clears_active};

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

fn call_bounded_blocks(func: &Function, cfg: &Cfg, kinds: &[CalleeKind]) -> Vec<bool> {
    let n = func.blocks.len();
    let has_call: Vec<bool> = (0..n).map(|bi| block_has_unwind_call(func, kinds, Block(bi as u32))).collect();
    let mut outdeg = vec![0u32; n];
    let mut ok = has_call.clone();
    let mut q = Vec::new();
    for bi in 0..n {
        if ok[bi] || !cfg.reachable[bi] {
            continue;
        }
        let deg = func.successors(Block(bi as u32)).filter(|s| cfg.reachable[s.index()] && !has_call[s.index()]).count() as u32;
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

fn target_ok(cfg: &Cfg, depths: &[u32], always_in: Option<&[bool]>, call_in_loop: Option<&[bool]>, b: Block, in_r: &[bool], s: Block) -> bool {
    let si = s.index();
    let depth_ok = depths[si] <= depths[b.index()] || call_in_loop.is_some_and(|c| c[si]);
    depth_ok
        && (cfg.preds_of(s).iter().all(|&p| !cfg.reachable[p.index()] || in_r[p.index()])
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
    sink_frontier_with_calls(func, cfg, depths, (live_in, always_active.map(|aa| (aa, aa)), None, b, 4), in_r)
}

fn prune_sink_region(
    func: &Function,
    cfg: &Cfg,
    depths: &[u32],
    (live_in, always_ready, call_in_loop, b): (&[bool], Option<(&[bool], &[bool])>, Option<&[bool]>, Block),
    in_r: &mut [bool],
) -> Option<Vec<Block>> {
    let always_in = always_ready.map(|(ai, _)| ai);
    let always_out = always_ready.map(|(_, ao)| ao);
    let n = func.blocks.len();
    let mut members: Vec<Block> = in_r.iter().enumerate().filter_map(|(bi, &m)| m.then_some(Block(bi as u32))).collect();
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
            let bad_succ = func.successors(blk).any(|s| !in_r[s.index()] && live_in[s.index()] && !target_ok(cfg, depths, always_in, call_in_loop, b, in_r, s));
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
    if !skipped_fast || frontier.is_empty() {
        return None;
    }
    frontier.iter().all(|&t| target_ok(cfg, depths, always_in, call_in_loop, b, in_r, t)).then_some(frontier)
}

fn bfs_prefix_contract(
    func: &Function,
    cfg: &Cfg,
    depths: &[u32],
    (live_in, always_ready, call_in_loop, b, max_frontier): (&[bool], Option<(&[bool], &[bool])>, Option<&[bool]>, Block, usize),
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
        if let Some(tf) = prune_sink_region(func, cfg, depths, (live_in, always_ready, call_in_loop, b), &mut trial) {
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
    (live_in, always_ready, call_in_loop, b, max_frontier): (&[bool], Option<(&[bool], &[bool])>, Option<&[bool]>, Block, usize),
    in_r: &mut [bool],
) -> Option<Vec<Block>> {
    let mut frontier = prune_sink_region(func, cfg, depths, (live_in, always_ready, call_in_loop, b), in_r)?;
    if frontier.len() <= max_frontier {
        return Some(frontier);
    }
    let mut cur = in_r.to_vec();
    if let Some((pref_r, pref_f)) = bfs_prefix_contract(func, cfg, depths, (live_in, always_ready, call_in_loop, b, max_frontier), &cur) {
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
            if let Some(tf) = prune_sink_region(func, cfg, depths, (live_in, always_ready, call_in_loop, b), &mut trial)
                && tf.len() < best.as_ref().map_or(frontier.len(), |(_, bf)| bf.len())
            {
                best = Some((trial, tf));
            }
        }
        let Some((next_r, next_f)) = best else { return None };
        cur = next_r;
        frontier = next_f;
    }
    in_r.copy_from_slice(&cur);
    Some(frontier)
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
    if insts[pos + 1..].iter().any(|&i| match func.insts[i.index()].op {
        Op::Call { .. } if kinds[i.index()] == CalleeKind::MayUnwind => true,
        Op::TryCall { .. } | Op::Return(..) | Op::Trap => true,
        Op::StackLoad(s, _) | Op::StackStore(_, s, _) if s == flag => true,
        _ => false,
    }) {
        return None;
    }
    let n = func.blocks.len();
    let call_bounded = call_bounded_blocks(func, cfg, kinds);
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
    sink_frontier_with_calls(func, cfg, depths, (live_in, Some(always_active), Some(&call_bounded), b, 8), &mut in_r).or_else(|| {
        let mut solo = vec![false; n];
        solo[b.index()] = true;
        sink_frontier_with_calls(func, cfg, depths, (live_in, Some(always_active), Some(&call_bounded), b, 8), &mut solo)
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

fn flag_always_zero(func: &Function, defs: &[Option<Inst>], entry: Block, flag: StackSlot) -> (Vec<bool>, Vec<bool>) {
    let n = func.blocks.len();
    let (mut in_state, mut out_state) = (vec![0u8; n], vec![0u8; n]);
    in_state[entry.index()] = 1;
    let mut work = vec![(entry, false)];
    while let Some((block, mut active)) = work.pop() {
        let mut stopped = false;
        for &inst in &func.blocks[block.index()].insts {
            match func.insts[inst.index()].op {
                Op::StackStore(v, s, 0) if s == flag => active = !store_clears_active(func, defs, v),
                Op::Return(..) | Op::Trap => {
                    stopped = true;
                    break;
                }
                _ => {}
            }
        }
        if stopped {
            continue;
        }
        let bit = if active { 2 } else { 1 };
        out_state[block.index()] |= bit;
        for target in func.successors(block) {
            if in_state[target.index()] & bit == 0 {
                in_state[target.index()] |= bit;
                work.push((target, active));
            }
        }
    }
    (in_state.iter().map(|&v| v == 1).collect(), out_state.iter().map(|&v| v == 1).collect())
}

pub(super) fn entry_first_zeros(func: &Function, defs: &[Option<Inst>], entry: Block, nslots: usize) -> Vec<Option<Inst>> {
    let mut first = vec![None; nslots];
    let mut seen_call = false;
    for &inst in &func.blocks[entry.index()].insts {
        match func.insts[inst.index()].op {
            Op::Call { .. } | Op::TryCall { .. } => seen_call = true,
            Op::StackStore(v, slot, 0) if !seen_call && slot.index() < nslots && first[slot.index()].is_none() && store_clears_active(func, defs, v) => {
                first[slot.index()] = Some(inst);
            }
            _ => {}
        }
    }
    first
}

pub(super) fn sink_entry_zeros(
    func: &mut Function,
    entry: Block,
    slot_escaped: &[bool],
    sym_may_unwind: &mut impl FnMut(u32) -> bool,
) -> bool {
    let nslots = func.slots.len();
    let defs = crate::fold::defs(func);
    let kinds: Vec<CalleeKind> = func
        .insts
        .iter()
        .map(|data| match data.op {
            Op::Call { callee, .. } => classify_callee(func, &defs, callee, sym_may_unwind),
            _ => CalleeKind::MayUnwind,
        })
        .collect();
    let first = entry_first_zeros(func, &defs, entry, nslots);
    let mut is_local = vec![false; nslots];
    let mut is_flag = vec![false; nslots];
    for u in &func.unwind_drops {
        if u.local.index() < nslots { is_local[u.local.index()] = true; }
        if u.flag.index() < nslots { is_flag[u.flag.index()] = true; }
    }
    let cfg = func.cfg();
    let depths = &cfg.loops().depths;
    let mut remove = Vec::new();
    let mut insert: Vec<(Block, Value, StackSlot)> = Vec::new();
    for fi in 0..nslots {
        if !is_flag[fi] || is_local[fi] || slot_escaped[fi] {
            continue;
        }
        let Some(zinst) = first[fi] else { continue };
        let Op::StackStore(val, flag, 0) = func.insts[zinst.index()].op else { continue };
        let Some(frontier) = plan_entry_zero_sink(func, &cfg, &defs, &kinds, depths, (entry, flag, zinst)) else { continue };
        remove.push(zinst);
        for t in frontier {
            insert.push((t, val, flag));
        }
    }
    if remove.is_empty() {
        return false;
    }
    for &i in &remove {
        func.insts[i.index()].op = Op::Trap;
    }
    func.blocks[entry.index()].insts.retain(|i| !remove.contains(i));
    for (t, val, flag) in insert {
        let store = func.add(Op::StackStore(val, flag, 0), None);
        func.blocks[t.index()].insts.insert(0, store);
    }
    true
}

fn plan_entry_zero_sink(
    func: &Function,
    cfg: &Cfg,
    defs: &[Option<Inst>],
    kinds: &[CalleeKind],
    depths: &[u32],
    (entry, flag, zinst): (Block, StackSlot, Inst),
) -> Option<Vec<Block>> {
    if cfg.pred_count(entry) != 0 {
        return None;
    }
    let insts = &func.blocks[entry.index()].insts;
    let pos = insts.iter().position(|&i| i == zinst)?;
    for &i in &insts[pos + 1..] {
        match func.insts[i.index()].op {
            Op::Call { .. } if kinds[i.index()] == CalleeKind::MayUnwind => return None,
            Op::TryCall { .. } => return None,
            Op::StackLoad(s, _) if s == flag => return None,
            Op::StackStore(_, s, 0) if s == flag => return Some(Vec::new()),
            Op::Return(..) | Op::Trap => return Some(Vec::new()),
            _ => {}
        }
    }
    let n = func.blocks.len();
    let (mut gen_b, mut kill_b, mut clean_b) = (vec![false; n], vec![false; n], vec![true; n]);
    for &blk in &cfg.rpo {
        if blk == entry {
            continue;
        }
        let bi = blk.index();
        for &inst in &func.blocks[bi].insts {
            match func.insts[inst.index()].op {
                Op::Call { .. } if kinds[inst.index()] == CalleeKind::MayUnwind => {
                    gen_b[bi] = true;
                    clean_b[bi] = false;
                    break;
                }
                Op::TryCall { .. } => {
                    gen_b[bi] = true;
                    clean_b[bi] = false;
                    break;
                }
                Op::StackLoad(s, _) if s == flag => {
                    gen_b[bi] = true;
                    clean_b[bi] = false;
                    break;
                }
                Op::StackStore(_, s, 0) if s == flag => {
                    kill_b[bi] = true;
                    clean_b[bi] = false;
                    break;
                }
                Op::Return(..) | Op::Trap => {
                    kill_b[bi] = true;
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
    if !func.successors(entry).any(|s| live_in[s.index()]) {
        return Some(Vec::new());
    }
    let (always_in, always_out) = flag_always_zero(func, defs, entry, flag);
    let call_bounded = call_bounded_blocks(func, cfg, kinds);
    let (mut in_r, mut work) = (vec![false; n], vec![entry]);
    in_r[entry.index()] = true;
    while let Some(u) = work.pop() {
        for s in func.successors(u) {
            let si = s.index();
            if !in_r[si] && cfg.reachable[si] && cfg.dominates(entry, s) && live_in[si] && clean_b[si] {
                in_r[si] = true;
                work.push(s);
            }
        }
    }
    sink_frontier_with_calls(func, cfg, depths, (&live_in, Some((&always_in, &always_out)), Some(&call_bounded), entry, 8), &mut in_r).or_else(|| {
        let mut solo = vec![false; n];
        solo[entry.index()] = true;
        sink_frontier_with_calls(func, cfg, depths, (&live_in, Some((&always_in, &always_out)), Some(&call_bounded), entry, 8), &mut solo)
    })
}

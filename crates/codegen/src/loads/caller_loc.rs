use crate::cfg::Cfg;
use crate::ir::{Block, Function, Inst, Op, SigRef, Type, Value};
use crate::mem::{MemRoot, Provenance};

use super::caller_loc_sink::{drop_dead_stores, drop_silent_stores, sink_cell_calls};

pub(super) const MAX_NARROW_SITES: usize = 8;

#[derive(Clone, Copy)]
pub(super) enum ClEvent {
    None,
    Read,
    Write(Value),
    Observe,
    Trap,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum ClVal {
    Uninit,
    Known(Value),
    Unknown,
}

impl ClVal {
    pub(super) fn join(self, other: Self) -> Self {
        match (self, other) {
            (Self::Uninit, x) | (x, Self::Uninit) => x,
            (Self::Known(a), Self::Known(b)) if a == b => Self::Known(a),
            _ => Self::Unknown,
        }
    }
}

pub(super) fn cl_event(func: &Function, prov: &Provenance, zeros: &[bool], is_cl: &[bool], op: &Op) -> ClEvent {
    match *op {
        Op::Store(val, addr, 0) if is_cl.get(addr.index()) == Some(&true) => ClEvent::Write(val),
        Op::Load(addr, 0) if is_cl.get(addr.index()) == Some(&true) => ClEvent::Read,
        Op::Call { callee, .. } | Op::TryCall { callee, .. } => {
            if prov.frame_runtime(callee).is_some()
                || func.is_fresh_alloc(callee)
                || (!zeros.is_empty() && func.pure_thread_cell_result(op, |v| zeros.get(v.index()) == Some(&true)).is_some())
            {
                ClEvent::None
            } else {
                ClEvent::Observe
            }
        }
        Op::Return(..) => ClEvent::Observe,
        Op::Trap => ClEvent::Trap,
        _ => ClEvent::None,
    }
}

pub(super) type ClDef = (SigRef, Value, Vec<Value>);

pub(super) fn collect_cl_list(func: &Function, cfg: &Cfg, zeros: &[bool]) -> Vec<(Value, Block, Inst, ClDef)> {
    let mut out = Vec::new();
    if zeros.is_empty() { return out; }
    for &b in &cfg.rpo {
        for &inst in &func.blocks[b.index()].insts {
            let data = &func.insts[inst.index()];
            if data.result.is_none()
                && let Some(res) = func.caller_loc_result(&data.op, |v| zeros.get(v.index()) == Some(&true))
                && let Op::Call { sig, callee, args, results } = &data.op
                && results.len() == 1
            {
                out.push((res, b, inst, (*sig, *callee, args.clone())));
            }
        }
    }
    out
}

fn collect_cl_defs(func: &Function, cfg: &Cfg, zeros: &[bool]) -> Vec<Option<(Block, Inst, ClDef)>> {
    let list = collect_cl_list(func, cfg, zeros);
    if list.is_empty() { return Vec::new(); }
    let mut defs = vec![None; func.types.len()];
    for (res, b, inst, def) in list { defs[res.index()] = Some((b, inst, def)); }
    defs
}

pub(super) fn push_cl_value(func: &mut Function, zeros: &mut Vec<bool>, is_cl: &mut Vec<bool>, mark_cl: bool) -> Value {
    let val = func.value(Type::I64);
    is_cl.push(mark_cl);
    if !zeros.is_empty() {
        zeros.push(false);
    }
    val
}

fn tail_is_cl_clean(func: &Function, prov: &Provenance, zeros: &[bool], is_cl: &[bool], block: Block, after_pos: usize) -> bool {
    !func.blocks[block.index()].insts[after_pos + 1..]
        .iter()
        .any(|&i| !matches!(cl_event(func, prov, zeros, is_cl, &func.insts[i.index()].op), ClEvent::None))
}

fn push_unvisited_succs(func: &Function, block: Block, visited: &mut [bool], work: &mut Vec<Block>) {
    for succ in func.successors(block) {
        if !std::mem::replace(&mut visited[succ.index()], true) {
            work.push(succ);
        }
    }
}

fn has_fast_path(func: &Function, start: Block, pass: &[bool], targets: &[bool]) -> bool {
    let mut fast_vis = vec![false; func.blocks.len()];
    let mut fast_work: Vec<Block> = func.successors(start).collect();
    while let Some(u) = fast_work.pop() {
        if targets[u.index()] {
            return true;
        }
        if pass[u.index()] && !std::mem::replace(&mut fast_vis[u.index()], true) {
            fast_work.extend(func.successors(u));
        }
    }
    false
}

enum ClWrap {
    TrapStore(Value),
    SaveStore(Value),
    SaveLocal(Value),
}

fn rewrite_cl_addr(func: &mut Function, insts: &[Inst], old_addr: Value, new_addr: Value) {
    for &inst in insts {
        match &mut func.insts[inst.index()].op {
            Op::Store(_, a, 0) | Op::Load(a, 0) if *a == old_addr => *a = new_addr,
            _ => {}
        }
    }
}

fn wrap_cl_block(
    func: &mut Function,
    (zeros, is_cl): (&mut Vec<bool>, &mut Vec<bool>),
    (sig, callee, args): &ClDef,
    (u, first, last): (Block, usize, usize),
    mode: ClWrap,
) {
    let cl_u = push_cl_value(func, zeros, is_cl, true);
    let cl_inst = func.add(Op::Call { sig: *sig, callee: *callee, args: args.clone(), results: vec![cl_u] }, None);
    let orig = std::mem::take(&mut func.blocks[u.index()].insts);
    let mut rebuilt = Vec::with_capacity(orig.len() + 4);
    rebuilt.extend_from_slice(&orig[..first]);
    rebuilt.push(cl_inst);
    match mode {
        ClWrap::TrapStore(val) => {
            rebuilt.push(func.add(Op::Store(val, cl_u, 0), None));
            rebuilt.extend_from_slice(&orig[first..]);
        }
        ClWrap::SaveStore(val) => {
            let prev_u = push_cl_value(func, zeros, is_cl, false);
            rebuilt.push(func.add(Op::Load(cl_u, 0), Some(prev_u)));
            rebuilt.push(func.add(Op::Store(val, cl_u, 0), None));
            rebuilt.extend_from_slice(&orig[first..=last]);
            rebuilt.push(func.add(Op::Store(prev_u, cl_u, 0), None));
            rebuilt.extend_from_slice(&orig[last + 1..]);
        }
        ClWrap::SaveLocal(old_addr) => {
            rewrite_cl_addr(func, &orig[first..=last], old_addr, cl_u);
            let prev_u = push_cl_value(func, zeros, is_cl, false);
            rebuilt.push(func.add(Op::Load(cl_u, 0), Some(prev_u)));
            rebuilt.extend_from_slice(&orig[first..=last]);
            rebuilt.push(func.add(Op::Store(prev_u, cl_u, 0), None));
            rebuilt.extend_from_slice(&orig[last + 1..]);
        }
    }
    func.blocks[u.index()].insts = rebuilt;
}

fn wrap_cl_region(
    func: &mut Function,
    (zeros, is_cl): (&mut Vec<bool>, &mut Vec<bool>),
    (sig, callee, args): &ClDef,
    (root, first, old_addr): (Block, usize, Value),
    (region, exits): (Vec<Block>, Vec<Block>),
) {
    let cl_r = push_cl_value(func, zeros, is_cl, true);
    let cl_inst = func.add(Op::Call { sig: *sig, callee: *callee, args: args.clone(), results: vec![cl_r] }, None);
    let prev_r = (!exits.is_empty()).then(|| push_cl_value(func, zeros, is_cl, false));
    let orig = std::mem::take(&mut func.blocks[root.index()].insts);
    rewrite_cl_addr(func, &orig[first..], old_addr, cl_r);
    let mut rebuilt = Vec::with_capacity(orig.len() + 2);
    rebuilt.extend_from_slice(&orig[..first]);
    rebuilt.push(cl_inst);
    if let Some(prev) = prev_r {
        rebuilt.push(func.add(Op::Load(cl_r, 0), Some(prev)));
    }
    rebuilt.extend_from_slice(&orig[first..]);
    func.blocks[root.index()].insts = rebuilt;
    for b in region {
        if b != root {
            let insts = std::mem::take(&mut func.blocks[b.index()].insts);
            rewrite_cl_addr(func, &insts, old_addr, cl_r);
            func.blocks[b.index()].insts = insts;
        }
    }
    if let Some(prev) = prev_r {
        for exit in exits {
            let restore = func.add(Op::Store(prev, cl_r, 0), None);
            let insts = &mut func.blocks[exit.index()].insts;
            let at = insts.len().saturating_sub(1);
            insts.insert(at, restore);
        }
    }
}

fn first_restore_inst(func: &Function, prov: &Provenance, zeros: &[bool], is_cl: &[bool], b: Block, saved: Value) -> Option<Inst> {
    for &inst in &func.blocks[b.index()].insts {
        match cl_event(func, prov, zeros, is_cl, &func.insts[inst.index()].op) {
            ClEvent::None => {}
            ClEvent::Write(w) => return (w == saved).then_some(inst),
            _ => return None,
        }
    }
    None
}

fn starts_with_trap(func: &Function, prov: &Provenance, zeros: &[bool], is_cl: &[bool], b: Block) -> bool {
    for &inst in &func.blocks[b.index()].insts {
        match cl_event(func, prov, zeros, is_cl, &func.insts[inst.index()].op) {
            ClEvent::None => {}
            ClEvent::Trap => return true,
            _ => return false,
        }
    }
    false
}

fn collect_dirty_region(
    func: &Function,
    cfg: &Cfg,
    loops: &crate::cfg::Loops,
    prov: &Provenance,
    (zeros, is_cl): (&[bool], &[bool]),
    (b_l, saved, root, depth_l): (Block, Value, Block, u32),
    restores: &mut [bool],
    hit_restores: &mut Vec<(Block, Inst)>,
) -> Option<(Vec<Block>, Vec<Block>)> {
    if matches!(func.last(root), Some(Op::Return(..) | Op::TryCall { .. }))
        || cfg.preds_of(root).iter().any(|&p| cfg.dominates(root, p))
    {
        return None;
    }
    let (mut region, mut exits, mut stack) = (vec![root], Vec::new(), vec![root]);
    let mut touched_restores: Vec<(Block, Inst)> = Vec::new();
    while let Some(cur) = stack.pop() {
        if matches!(func.last(cur), Some(Op::Trap)) {
            continue;
        }
        for s in func.successors(cur) {
            if s != root && cfg.dominates(b_l, s) && let Some(r_inst) = first_restore_inst(func, prov, zeros, is_cl, s, saved) {
                if !matches!(func.last(cur), Some(&Op::Jump(t, _)) if t == s) {
                    return None;
                }
                touched_restores.push((s, r_inst));
                if !exits.contains(&cur) {
                    exits.push(cur);
                }
                continue;
            }
            if starts_with_trap(func, prov, zeros, is_cl, s) {
                continue;
            }
            if !cfg.dominates(root, s) || loops.depth_of(s) > depth_l {
                return None;
            }
            if region.contains(&s) {
                continue;
            }
            if region.len() >= 16 || matches!(func.last(s), Some(Op::Return(..) | Op::TryCall { .. })) {
                return None;
            }
            if func.blocks[s.index()].insts.iter().any(|&i| matches!(cl_event(func, prov, zeros, is_cl, &func.insts[i.index()].op), ClEvent::Write(w) if w == saved)) {
                return None;
            }
            region.push(s);
            stack.push(s);
        }
    }
    for (s, r_inst) in touched_restores {
        restores[s.index()] = true;
        if !hit_restores.iter().any(|&(_, ri)| ri == r_inst) {
            hit_restores.push((s, r_inst));
        }
    }
    Some((region, exits))
}

/// Narrows `#[track_caller]` TLS stores and save/restore spans onto cold call sites, drops silent or unobservable stores, and sinks cold cell lookups.
pub(super) fn optimize(func: &mut Function, cfg: &Cfg, prov: &Provenance) -> bool {
    if !prov.has_caller_loc {
        return false;
    }
    let mut zeros = func.thread_cell_zeros();
    let mut is_cl: Vec<bool> = prov.param_roots.iter().map(|&r| r == MemRoot::CallerLoc).collect();
    let mut store_cands: Vec<(Block, Inst)> = Vec::new();
    let mut load_cands: Vec<(Block, Inst)> = Vec::new();
    let mut defs_list: Vec<(Value, Block, Inst, ClDef)> = Vec::new();
    let mut cl_defs: Vec<Option<(Block, Inst, ClDef)>> = Vec::new();
    for &b in &cfg.rpo {
        for &i in &func.blocks[b.index()].insts {
            let d = &func.insts[i.index()];
            match (&d.op, d.result) {
                (Op::Store(_, a, 0), _) if is_cl.get(a.index()) == Some(&true) => store_cands.push((b, i)),
                (Op::Load(a, 0), Some(_)) if is_cl.get(a.index()) == Some(&true) => load_cands.push((b, i)),
                (Op::Call { sig, callee, args, results }, None)
                    if !zeros.is_empty()
                        && results.len() == 1
                        && let Some(res) = func.caller_loc_result(&d.op, |v| zeros.get(v.index()) == Some(&true)) =>
                {
                    let def = (*sig, *callee, args.clone());
                    defs_list.push((res, b, i, def.clone()));
                    if cl_defs.is_empty() {
                        cl_defs.resize(func.types.len(), None);
                    }
                    cl_defs[res.index()] = Some((b, i, def));
                }
                _ => {}
            }
        }
    }
    let has_load = !load_cands.is_empty();
    let mut changed = false;
    if !store_cands.is_empty() {
        let ns = narrow_stores(func, cfg, prov, &mut zeros, &mut is_cl, &cl_defs, store_cands);
        changed |= ns;
        if has_load || ns {
            if ns {
                cl_defs = collect_cl_defs(func, cfg, &zeros);
            }
            let loads_arg = if ns { None } else { Some(load_cands) };
            changed |= narrow_saves(func, cfg, prov, &mut zeros, &mut is_cl, &cl_defs, loads_arg);
        }
        changed |= drop_silent_stores(func, cfg, prov, &zeros, &is_cl);
        changed |= drop_dead_stores(func, cfg, prov, &zeros, &is_cl);
    }
    if changed || has_load || !defs_list.is_empty() {
        let pre_defs = (!changed).then_some(defs_list);
        changed |= sink_cell_calls(func, cfg, &mut zeros, &mut is_cl, has_load || changed, pre_defs);
    }
    changed
}

fn narrow_stores(
    func: &mut Function,
    cfg: &Cfg,
    prov: &Provenance,
    zeros: &mut Vec<bool>,
    is_cl: &mut Vec<bool>,
    cl_defs: &[Option<(Block, Inst, ClDef)>],
    candidates: Vec<(Block, Inst)>,
) -> bool {
    if candidates.is_empty() || cl_defs.is_empty() {
        return false;
    }
    let n = func.blocks.len();
    let mut loops = None;
    let (mut visited, mut pass, mut kills) = (Vec::new(), Vec::new(), Vec::new());
    let mut changed = false;
    for (b_s, store_inst) in candidates {
        let Some(pos_s) = func.blocks[b_s.index()].insts.iter().position(|&i| i == store_inst) else {
            continue;
        };
        let Op::Store(val, addr, 0) = func.insts[store_inst.index()].op else {
            continue;
        };
        let Some((_, _, def)) = cl_defs.get(addr.index()).and_then(Clone::clone) else {
            continue;
        };
        if !tail_is_cl_clean(func, prov, zeros, is_cl, b_s, pos_s) {
            continue;
        }
        let loops = loops.get_or_insert_with(|| cfg.loops());
        if visited.is_empty() {
            (visited, pass, kills) = (vec![false; n], vec![false; n], vec![false; n]);
        } else {
            visited.fill(false);
            pass.fill(false);
            kills.fill(false);
        }
        let mut obs: Vec<(Block, usize, usize, bool)> = Vec::new();
        let mut work: Vec<Block> = Vec::new();
        push_unvisited_succs(func, b_s, &mut visited, &mut work);
        let mut ok = !work.is_empty();
        while let Some(u) = work.pop() {
            if u == b_s {
                ok = false;
                break;
            }
            let insts = &func.blocks[u.index()].insts;
            let mut first_obs = None;
            let mut last_obs = 0;
            let mut hit_kill = false;
            let mut write_after_obs = false;
            for (idx, &inst) in insts.iter().enumerate() {
                match cl_event(func, prov, zeros, is_cl, &func.insts[inst.index()].op) {
                    ClEvent::None => {}
                    ClEvent::Write(..) => {
                        if first_obs.is_none() {
                            hit_kill = true;
                        } else {
                            write_after_obs = true;
                        }
                        break;
                    }
                    ClEvent::Trap => {
                        if first_obs.is_none() {
                            hit_kill = true;
                        }
                        break;
                    }
                    ClEvent::Observe | ClEvent::Read => {
                        if first_obs.is_none() {
                            first_obs = Some(idx);
                        }
                        last_obs = idx;
                    }
                }
            }
            if hit_kill {
                kills[u.index()] = true;
                continue;
            }
            if !cfg.dominates(b_s, u) {
                ok = false;
                break;
            }
            if let Some(first) = first_obs {
                let last_op = func.last(u);
                if write_after_obs
                    || matches!(last_op, Some(Op::Return(..) | Op::TryCall { .. }))
                    || loops.depth_of(u) > loops.depth_of(b_s)
                    || obs.len() >= MAX_NARROW_SITES
                {
                    ok = false;
                    break;
                }
                let ends_in_trap = matches!(last_op, Some(Op::Trap));
                obs.push((u, first, last_obs, ends_in_trap));
                if !ends_in_trap {
                    push_unvisited_succs(func, u, &mut visited, &mut work);
                }
            } else {
                pass[u.index()] = true;
                push_unvisited_succs(func, u, &mut visited, &mut work);
            }
        }
        if !ok || obs.is_empty() || !has_fast_path(func, b_s, &pass, &kills) {
            continue;
        }
        let removed = func.blocks[b_s.index()].insts.remove(pos_s);
        func.insts[removed.index()].op = Op::Trap;
        for (u, first, last, ends_in_trap) in obs {
            let mode = if ends_in_trap { ClWrap::TrapStore(val) } else { ClWrap::SaveStore(val) };
            wrap_cl_block(func, (zeros, is_cl), &def, (u, first, last), mode);
        }
        changed = true;
    }
    changed
}

fn narrow_saves(
    func: &mut Function,
    cfg: &Cfg,
    prov: &Provenance,
    zeros: &mut Vec<bool>,
    is_cl: &mut Vec<bool>,
    cl_defs: &[Option<(Block, Inst, ClDef)>],
    pre_cands: Option<Vec<(Block, Inst)>>,
) -> bool {
    if cl_defs.is_empty() {
        return false;
    }
    let candidates = pre_cands.unwrap_or_else(|| {
        let mut out = Vec::new();
        for &b in &cfg.rpo {
            for &inst in &func.blocks[b.index()].insts {
                let data = &func.insts[inst.index()];
                if let Op::Load(addr, 0) = data.op
                    && data.result.is_some()
                    && is_cl.get(addr.index()) == Some(&true)
                {
                    out.push((b, inst));
                }
            }
        }
        out
    });
    if candidates.is_empty() {
        return false;
    }
    let saved_vals: Vec<Value> = candidates.iter().filter_map(|&(_, i)| func.insts[i.index()].result).collect();
    let saved_mask = saved_vals.iter().fold(0_u128, |m, &v| m | (1_u128 << (v.0 & 127)));
    let mut non_restore = vec![false; saved_vals.len()];
    let mut restore_counts = vec![0_u32; saved_vals.len()];
    for &b in &cfg.rpo {
        for &inst in &func.blocks[b.index()].insts {
            let data = &func.insts[inst.index()];
            if let Op::Store(val, addr, 0) = data.op
                && is_cl.get(addr.index()) == Some(&true)
                && val != addr
            {
                if saved_mask & (1_u128 << (val.0 & 127)) != 0 {
                    for (idx, &sv) in saved_vals.iter().enumerate() {
                        if sv == val { restore_counts[idx] += 1; }
                    }
                }
                if saved_mask & (1_u128 << (addr.0 & 127)) != 0 {
                    for (idx, &sv) in saved_vals.iter().enumerate() {
                        if sv == addr { non_restore[idx] = true; }
                    }
                }
            } else {
                data.op.for_operands(|v| {
                    if saved_mask & (1_u128 << (v.0 & 127)) != 0 {
                        for (idx, &sv) in saved_vals.iter().enumerate() {
                            if sv == v { non_restore[idx] = true; }
                        }
                    }
                });
            }
        }
    }
    let n = func.blocks.len();
    let mut loops = None;
    let (mut visited, mut pass, mut restores) = (Vec::new(), Vec::new(), Vec::new());
    let mut changed = false;
    for (ci, (b_l, load_inst)) in candidates.into_iter().enumerate() {
        let Some(pos_l) = func.blocks[b_l.index()].insts.iter().position(|&i| i == load_inst) else {
            continue;
        };
        let data = &func.insts[load_inst.index()];
        let (Op::Load(addr, 0), Some(saved)) = (data.op.clone(), data.result) else {
            continue;
        };
        if non_restore.get(ci) == Some(&true) {
            continue;
        }
        let Some((_, _, def)) = cl_defs.get(addr.index()).and_then(Clone::clone) else {
            continue;
        };
        if !tail_is_cl_clean(func, prov, zeros, is_cl, b_l, pos_l) {
            continue;
        }
        let loops = loops.get_or_insert_with(|| cfg.loops());
        if visited.is_empty() {
            (visited, pass, restores) = (vec![false; n], vec![false; n], vec![false; n]);
        } else {
            visited.fill(false);
            pass.fill(false);
            restores.fill(false);
        }
        let mut dirty: Vec<(Block, usize, usize)> = Vec::new();
        let mut regions: Vec<(Block, usize, Vec<Block>, Vec<Block>)> = Vec::new();
        let mut hit_restores: Vec<(Block, Inst)> = Vec::new();
        let mut work: Vec<Block> = Vec::new();
        push_unvisited_succs(func, b_l, &mut visited, &mut work);
        let mut ok = !work.is_empty();
        while let Some(u) = work.pop() {
            if u == b_l { ok = false; break; }
            let insts = &func.blocks[u.index()].insts;
            let (mut first_write, mut last_event, mut starts_restore, mut starts_trap, mut bad_event) = (None, 0, None, false, false);
            for (idx, &inst) in insts.iter().enumerate() {
                match cl_event(func, prov, zeros, is_cl, &func.insts[inst.index()].op) {
                    ClEvent::None => {}
                    ClEvent::Write(w) => {
                        if first_write.is_none() {
                            if w == saved { starts_restore = Some(inst); break; }
                            (first_write, last_event) = (Some(idx), idx);
                        } else if w == saved { bad_event = true; break; } else { last_event = idx; }
                    }
                    ClEvent::Trap => { if first_write.is_none() { starts_trap = true; } break; }
                    ClEvent::Read | ClEvent::Observe => { if first_write.is_none() { bad_event = true; break; } last_event = idx; }
                }
            }
            if let Some(r_inst) = starts_restore {
                if !cfg.dominates(b_l, u) { ok = false; break; }
                restores[u.index()] = true;
                if !hit_restores.iter().any(|&(_, ri)| ri == r_inst) {
                    hit_restores.push((u, r_inst));
                }
                continue;
            }
            if starts_trap { continue; }
            if bad_event || !cfg.dominates(b_l, u) { ok = false; break; }
            if let Some(fw) = first_write {
                let last_op = func.last(u);
                if matches!(last_op, Some(Op::Return(..) | Op::TryCall { .. }))
                    || loops.depth_of(u) > loops.depth_of(b_l)
                    || dirty.len() + regions.len() >= MAX_NARROW_SITES
                {
                    ok = false;
                    break;
                }
                if let Some((region, exits)) = collect_dirty_region(func, cfg, loops, prov, (zeros, is_cl), (b_l, saved, u, loops.depth_of(b_l)), &mut restores, &mut hit_restores) {
                    regions.push((u, fw, region, exits));
                } else if !matches!(last_op, Some(Op::Trap)) {
                    dirty.push((u, fw, last_event));
                    push_unvisited_succs(func, u, &mut visited, &mut work);
                }
            } else {
                pass[u.index()] = true;
                push_unvisited_succs(func, u, &mut visited, &mut work);
            }
        }
        if !ok
            || (dirty.is_empty() && regions.is_empty())
            || hit_restores.len() != restore_counts[ci] as usize
            || !has_fast_path(func, b_l, &pass, &restores)
        {
            continue;
        }
        func.blocks[b_l.index()].insts.remove(pos_l);
        func.insts[load_inst.index()].op = Op::Trap;
        for (rb, r_inst) in hit_restores {
            if let Some(rpos) = func.blocks[rb.index()].insts.iter().position(|&i| i == r_inst) {
                func.blocks[rb.index()].insts.remove(rpos);
                func.insts[r_inst.index()].op = Op::Trap;
            }
        }
        for (u, first, last) in dirty {
            wrap_cl_block(func, (zeros, is_cl), &def, (u, first, last), ClWrap::SaveLocal(addr));
        }
        for (root, fw, region, exits) in regions {
            wrap_cl_region(func, (zeros, is_cl), &def, (root, fw, addr), (region, exits));
        }
        changed = true;
    }
    changed
}

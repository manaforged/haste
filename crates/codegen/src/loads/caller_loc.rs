use crate::cfg::Cfg;
use crate::ir::{Block, Function, Inst, Op, SigRef, Type, Value};
use crate::mem::{MemRoot, Provenance};

const MAX_NARROW_SITES: usize = 8;

#[derive(Clone, Copy)]
enum ClEvent {
    None,
    Read,
    Write(Value),
    Observe,
    Trap,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ClVal {
    Uninit,
    Known(Value),
    Unknown,
}

impl ClVal {
    fn join(self, other: Self) -> Self {
        match (self, other) {
            (Self::Uninit, x) | (x, Self::Uninit) => x,
            (Self::Known(a), Self::Known(b)) if a == b => Self::Known(a),
            _ => Self::Unknown,
        }
    }
}

fn cl_event(func: &Function, prov: &Provenance, zeros: &[bool], is_cl: &[bool], op: &Op) -> ClEvent {
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

/// Narrows `#[track_caller]` TLS stores onto cold call sites and drops silent or unobservable stores.
pub(super) fn optimize(func: &mut Function, cfg: &Cfg, prov: &Provenance) -> bool {
    if !prov.param_roots.contains(&MemRoot::CallerLoc) {
        return false;
    }
    let mut zeros = func.thread_cell_zeros();
    let mut is_cl: Vec<bool> = prov.param_roots.iter().map(|&r| r == MemRoot::CallerLoc).collect();
    let mut changed = narrow_stores(func, cfg, prov, &mut zeros, &mut is_cl);
    changed |= drop_silent_stores(func, cfg, prov, &zeros, &is_cl);
    changed |= drop_dead_stores(func, cfg, prov, &zeros, &is_cl);
    changed
}

fn narrow_stores(
    func: &mut Function,
    cfg: &Cfg,
    prov: &Provenance,
    zeros: &mut Vec<bool>,
    is_cl: &mut Vec<bool>,
) -> bool {
    let mut cl_defs: Vec<Option<(SigRef, Value, Vec<Value>)>> = vec![None; func.types.len()];
    let mut candidates: Vec<(Block, Inst)> = Vec::new();
    for &b in &cfg.rpo {
        for &inst in &func.blocks[b.index()].insts {
            let data = &func.insts[inst.index()];
            if data.result.is_none()
                && !zeros.is_empty()
                && let Some(res) = func.caller_loc_result(&data.op, |v| zeros.get(v.index()) == Some(&true))
                && let Op::Call { sig, callee, args, results } = &data.op
                && results.len() == 1
            {
                cl_defs[res.index()] = Some((*sig, *callee, args.clone()));
            }
            if let Op::Store(_, addr, 0) = data.op
                && is_cl.get(addr.index()) == Some(&true)
            {
                candidates.push((b, inst));
            }
        }
    }
    if candidates.is_empty() {
        return false;
    }
    let loops = cfg.loops();
    let n = func.blocks.len();
    let mut changed = false;
    for (b_s, store_inst) in candidates {
        let Some(pos_s) = func.blocks[b_s.index()].insts.iter().position(|&i| i == store_inst) else {
            continue;
        };
        let Op::Store(val, addr, 0) = func.insts[store_inst.index()].op else {
            continue;
        };
        let Some((sig, callee, args)) = cl_defs.get(addr.index()).and_then(Clone::clone) else {
            continue;
        };
        if func.blocks[b_s.index()].insts[pos_s + 1..]
            .iter()
            .any(|&i| !matches!(cl_event(func, prov, zeros, is_cl, &func.insts[i.index()].op), ClEvent::None))
        {
            continue;
        }
        let mut visited = vec![false; n];
        let mut pass = vec![false; n];
        let mut kills = vec![false; n];
        let mut obs: Vec<(Block, usize, usize, bool)> = Vec::new();
        let mut work: Vec<Block> = Vec::new();
        for succ in func.successors(b_s) {
            if !visited[succ.index()] {
                visited[succ.index()] = true;
                work.push(succ);
            }
        }
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
                    for succ in func.successors(u) {
                        if !visited[succ.index()] {
                            visited[succ.index()] = true;
                            work.push(succ);
                        }
                    }
                }
            } else {
                pass[u.index()] = true;
                for succ in func.successors(u) {
                    if !visited[succ.index()] {
                        visited[succ.index()] = true;
                        work.push(succ);
                    }
                }
            }
        }
        if !ok || obs.is_empty() {
            continue;
        }
        let mut fast_vis = vec![false; n];
        let mut fast_work: Vec<Block> = func.successors(b_s).collect();
        let mut has_fast = false;
        while let Some(u) = fast_work.pop() {
            if kills[u.index()] {
                has_fast = true;
                break;
            }
            if pass[u.index()] && !fast_vis[u.index()] {
                fast_vis[u.index()] = true;
                fast_work.extend(func.successors(u));
            }
        }
        if !has_fast {
            continue;
        }
        let removed = func.blocks[b_s.index()].insts.remove(pos_s);
        func.insts[removed.index()].op = Op::Trap;
        for (u, first, last, ends_in_trap) in obs {
            let cl_u = func.value(Type::I64);
            is_cl.push(true);
            if !zeros.is_empty() {
                zeros.push(false);
            }
            let cl_inst = func.add(Op::Call { sig, callee, args: args.clone(), results: vec![cl_u] }, None);
            let orig = std::mem::take(&mut func.blocks[u.index()].insts);
            let mut rebuilt = Vec::with_capacity(orig.len() + 4);
            rebuilt.extend_from_slice(&orig[..first]);
            rebuilt.push(cl_inst);
            if ends_in_trap {
                rebuilt.push(func.add(Op::Store(val, cl_u, 0), None));
                rebuilt.extend_from_slice(&orig[first..]);
            } else {
                let prev_u = func.value(Type::I64);
                is_cl.push(false);
                if !zeros.is_empty() {
                    zeros.push(false);
                }
                rebuilt.push(func.add(Op::Load(cl_u, 0), Some(prev_u)));
                rebuilt.push(func.add(Op::Store(val, cl_u, 0), None));
                rebuilt.extend_from_slice(&orig[first..=last]);
                rebuilt.push(func.add(Op::Store(prev_u, cl_u, 0), None));
                rebuilt.extend_from_slice(&orig[last + 1..]);
            }
            func.blocks[u.index()].insts = rebuilt;
        }
        changed = true;
    }
    changed
}

fn step_forward(
    func: &Function,
    prov: &Provenance,
    zeros: &[bool],
    is_cl: &[bool],
    block: Block,
    entry_state: ClVal,
    mut on_silent: impl FnMut(usize),
) -> ClVal {
    let mut state = entry_state;
    let mut local_loads: Vec<(Value, Value)> = Vec::new();
    for (pos, &inst) in func.blocks[block.index()].insts.iter().enumerate() {
        let data = &func.insts[inst.index()];
        match cl_event(func, prov, zeros, is_cl, &data.op) {
            ClEvent::None => {}
            ClEvent::Read => {
                if let Some(res) = data.result {
                    match state {
                        ClVal::Known(k) => local_loads.push((res, k)),
                        _ => state = ClVal::Known(res),
                    }
                }
            }
            ClEvent::Write(val) => {
                let eff = local_loads.iter().find(|&&(r, _)| r == val).map_or(val, |&(_, k)| k);
                if state == ClVal::Known(eff) {
                    on_silent(pos);
                } else {
                    state = ClVal::Known(eff);
                }
            }
            ClEvent::Observe => state = ClVal::Unknown,
            ClEvent::Trap => break,
        }
    }
    state
}

fn drop_silent_stores(func: &mut Function, cfg: &Cfg, prov: &Provenance, zeros: &[bool], is_cl: &[bool]) -> bool {
    let Some(entry) = func.entry else { return false };
    let n = func.blocks.len();
    let mut in_val = vec![ClVal::Uninit; n];
    let mut out_val = vec![ClVal::Uninit; n];
    in_val[entry.index()] = ClVal::Unknown;
    loop {
        let mut changed = false;
        for &b in &cfg.rpo {
            let state = if b == entry {
                ClVal::Unknown
            } else {
                cfg.preds_of(b).iter().fold(ClVal::Uninit, |acc, &p| acc.join(out_val[p.index()]))
            };
            if state != in_val[b.index()] {
                in_val[b.index()] = state;
                changed = true;
            }
            let out = step_forward(func, prov, zeros, is_cl, b, in_val[b.index()], |_| {});
            if out != out_val[b.index()] {
                out_val[b.index()] = out;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let mut dropped = false;
    for &b in &cfg.rpo {
        let mut kills = Vec::new();
        step_forward(func, prov, zeros, is_cl, b, in_val[b.index()], |pos| kills.push(pos));
        for pos in kills.into_iter().rev() {
            let inst = func.blocks[b.index()].insts.remove(pos);
            func.insts[inst.index()].op = Op::Trap;
            dropped = true;
        }
    }
    dropped
}

fn drop_dead_stores(func: &mut Function, cfg: &Cfg, prov: &Provenance, zeros: &[bool], is_cl: &[bool]) -> bool {
    let step = |op: &Op, live: bool| match cl_event(func, prov, zeros, is_cl, op) {
        ClEvent::Write(..) => false,
        ClEvent::Read | ClEvent::Observe => true,
        ClEvent::Trap | ClEvent::None => live,
    };
    let n = func.blocks.len();
    let mut live_in = vec![false; n];
    loop {
        let mut changed = false;
        for &b in cfg.rpo.iter().rev() {
            let mut live = match func.last(b) {
                Some(Op::Trap) => false,
                Some(Op::Return(..)) => true,
                _ => func.successors(b).any(|s| live_in[s.index()]),
            };
            for &inst in func.blocks[b.index()].insts.iter().rev() {
                live = step(&func.insts[inst.index()].op, live);
            }
            if live && !live_in[b.index()] {
                live_in[b.index()] = true;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let mut kills = Vec::new();
    for &b in &cfg.rpo {
        let mut live = match func.last(b) {
            Some(Op::Trap) => false,
            Some(Op::Return(..)) => true,
            _ => func.successors(b).any(|s| live_in[s.index()]),
        };
        let len = func.blocks[b.index()].insts.len();
        for pos in (0..len).rev() {
            let inst = func.blocks[b.index()].insts[pos];
            let op = &func.insts[inst.index()].op;
            if matches!(cl_event(func, prov, zeros, is_cl, op), ClEvent::Write(..)) && !live {
                kills.push((b.index(), pos));
            }
            live = step(op, live);
        }
    }
    let dropped = !kills.is_empty();
    for (b_idx, pos) in kills {
        let inst = func.blocks[b_idx].insts.remove(pos);
        func.insts[inst.index()].op = Op::Trap;
    }
    dropped
}

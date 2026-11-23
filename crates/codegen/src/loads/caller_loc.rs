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

type ClDef = (SigRef, Value, Vec<Value>);

fn collect_cl_defs(func: &Function, cfg: &Cfg, zeros: &[bool]) -> Vec<Option<(Block, Inst, ClDef)>> {
    let mut defs = vec![None; func.types.len()];
    if zeros.is_empty() {
        return defs;
    }
    for &b in &cfg.rpo {
        for &inst in &func.blocks[b.index()].insts {
            let data = &func.insts[inst.index()];
            if data.result.is_none()
                && let Some(res) = func.caller_loc_result(&data.op, |v| zeros.get(v.index()) == Some(&true))
                && let Op::Call { sig, callee, args, results } = &data.op
                && results.len() == 1
            {
                defs[res.index()] = Some((b, inst, (*sig, *callee, args.clone())));
            }
        }
    }
    defs
}

fn push_cl_value(func: &mut Function, zeros: &mut Vec<bool>, is_cl: &mut Vec<bool>, mark_cl: bool) -> Value {
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
            for &inst in &orig[first..=last] {
                match &mut func.insts[inst.index()].op {
                    Op::Store(_, a, 0) | Op::Load(a, 0) if *a == old_addr => *a = cl_u,
                    _ => {}
                }
            }
            let prev_u = push_cl_value(func, zeros, is_cl, false);
            rebuilt.push(func.add(Op::Load(cl_u, 0), Some(prev_u)));
            rebuilt.extend_from_slice(&orig[first..=last]);
            rebuilt.push(func.add(Op::Store(prev_u, cl_u, 0), None));
            rebuilt.extend_from_slice(&orig[last + 1..]);
        }
    }
    func.blocks[u.index()].insts = rebuilt;
}

/// Narrows `#[track_caller]` TLS stores and save/restore spans onto cold call sites, drops silent or unobservable stores, and sinks cold cell lookups.
pub(super) fn optimize(func: &mut Function, cfg: &Cfg, prov: &Provenance) -> bool {
    if !prov.has_caller_loc {
        return false;
    }
    let mut zeros = func.thread_cell_zeros();
    let mut is_cl: Vec<bool> = prov.param_roots.iter().map(|&r| r == MemRoot::CallerLoc).collect();
    let (mut has_store, mut has_load) = (false, false);
    for &b in &cfg.rpo {
        for &i in &func.blocks[b.index()].insts {
            let d = &func.insts[i.index()];
            match (&d.op, d.result) {
                (Op::Store(_, a, 0), _) if is_cl.get(a.index()) == Some(&true) => has_store = true,
                (Op::Load(a, 0), Some(_)) if is_cl.get(a.index()) == Some(&true) => has_load = true,
                _ => {}
            }
        }
    }
    let mut changed = false;
    if has_store {
        let mut cl_defs = collect_cl_defs(func, cfg, &zeros);
        let ns = narrow_stores(func, cfg, prov, &mut zeros, &mut is_cl, &cl_defs);
        changed |= ns;
        if has_load || ns {
            if ns { cl_defs = collect_cl_defs(func, cfg, &zeros); }
            changed |= narrow_saves(func, cfg, prov, &mut zeros, &mut is_cl, &cl_defs);
        }
        changed |= drop_silent_stores(func, cfg, prov, &zeros, &is_cl);
        changed |= drop_dead_stores(func, cfg, prov, &zeros, &is_cl);
    }
    changed |= sink_cell_calls(func, cfg, &mut zeros, &mut is_cl, has_load || changed);
    changed
}

fn narrow_stores(
    func: &mut Function,
    cfg: &Cfg,
    prov: &Provenance,
    zeros: &mut Vec<bool>,
    is_cl: &mut Vec<bool>,
    cl_defs: &[Option<(Block, Inst, ClDef)>],
) -> bool {
    let mut candidates: Vec<(Block, Inst)> = Vec::new();
    for &b in &cfg.rpo {
        for &inst in &func.blocks[b.index()].insts {
            if let Op::Store(_, addr, 0) = func.insts[inst.index()].op
                && is_cl.get(addr.index()) == Some(&true)
            {
                candidates.push((b, inst));
            }
        }
    }
    if candidates.is_empty() {
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
) -> bool {
    let mut non_restore = vec![false; func.types.len()];
    let mut candidates: Vec<(Block, Inst)> = Vec::new();
    for &b in &cfg.rpo {
        for &inst in &func.blocks[b.index()].insts {
            let data = &func.insts[inst.index()];
            if let Op::Store(val, addr, 0) = data.op
                && is_cl.get(addr.index()) == Some(&true)
                && val != addr
            {
                non_restore[addr.index()] = true;
            } else {
                data.op.for_operands(|v| non_restore[v.index()] = true);
            }
            if let Op::Load(addr, 0) = data.op
                && data.result.is_some()
                && is_cl.get(addr.index()) == Some(&true)
            {
                candidates.push((b, inst));
            }
        }
    }
    if candidates.is_empty() {
        return false;
    }
    let n = func.blocks.len();
    let mut loops = None;
    let (mut visited, mut pass, mut restores) = (Vec::new(), Vec::new(), Vec::new());
    let mut changed = false;
    for (b_l, load_inst) in candidates {
        let Some(pos_l) = func.blocks[b_l.index()].insts.iter().position(|&i| i == load_inst) else {
            continue;
        };
        let data = &func.insts[load_inst.index()];
        let (Op::Load(addr, 0), Some(saved)) = (data.op.clone(), data.result) else {
            continue;
        };
        if non_restore.get(saved.index()) == Some(&true) {
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
        let mut work: Vec<Block> = Vec::new();
        push_unvisited_succs(func, b_l, &mut visited, &mut work);
        let mut ok = !work.is_empty();
        while let Some(u) = work.pop() {
            if u == b_l {
                ok = false;
                break;
            }
            let insts = &func.blocks[u.index()].insts;
            let mut first_write = None;
            let mut last_event = 0;
            let mut starts_restore = false;
            let mut starts_trap = false;
            let mut bad_event = false;
            for (idx, &inst) in insts.iter().enumerate() {
                match cl_event(func, prov, zeros, is_cl, &func.insts[inst.index()].op) {
                    ClEvent::None => {}
                    ClEvent::Write(w) => {
                        if first_write.is_none() {
                            if w == saved {
                                starts_restore = true;
                                break;
                            }
                            first_write = Some(idx);
                            last_event = idx;
                        } else if w == saved {
                            bad_event = true;
                            break;
                        } else {
                            last_event = idx;
                        }
                    }
                    ClEvent::Trap => {
                        if first_write.is_none() {
                            starts_trap = true;
                        }
                        break;
                    }
                    ClEvent::Read | ClEvent::Observe => {
                        if first_write.is_none() {
                            bad_event = true;
                            break;
                        }
                        last_event = idx;
                    }
                }
            }
            if starts_restore {
                if !cfg.dominates(b_l, u) {
                    ok = false;
                    break;
                }
                restores[u.index()] = true;
                continue;
            }
            if starts_trap {
                continue;
            }
            if bad_event || !cfg.dominates(b_l, u) {
                ok = false;
                break;
            }
            if let Some(fw) = first_write {
                let last_op = func.last(u);
                if matches!(last_op, Some(Op::Return(..) | Op::TryCall { .. }))
                    || loops.depth_of(u) > loops.depth_of(b_l)
                    || dirty.len() >= MAX_NARROW_SITES
                {
                    ok = false;
                    break;
                }
                if !matches!(last_op, Some(Op::Trap)) {
                    dirty.push((u, fw, last_event));
                    push_unvisited_succs(func, u, &mut visited, &mut work);
                }
            } else {
                pass[u.index()] = true;
                push_unvisited_succs(func, u, &mut visited, &mut work);
            }
        }
        if !ok || dirty.is_empty() || !has_fast_path(func, b_l, &pass, &restores) {
            continue;
        }
        for (u, first, last) in dirty {
            wrap_cl_block(func, (zeros, is_cl), &def, (u, first, last), ClWrap::SaveLocal(addr));
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

#[derive(Clone, Copy)]
enum BlockFwd { PassThrough, Const(ClVal), Dynamic }

fn drop_silent_stores(func: &mut Function, cfg: &Cfg, prov: &Provenance, zeros: &[bool], is_cl: &[bool]) -> bool {
    let Some(entry) = func.entry else { return false };
    let n = func.blocks.len();
    let (mut fwd, mut has_write, mut any_write) = (vec![BlockFwd::PassThrough; n], vec![false; n], false);
    for &b in &cfg.rpo {
        let (mut saw_read, mut saw_effect, mut out) = (false, false, ClVal::Unknown);
        for &inst in &func.blocks[b.index()].insts {
            match cl_event(func, prov, zeros, is_cl, &func.insts[inst.index()].op) {
                ClEvent::None => {}
                ClEvent::Read => saw_read = true,
                ClEvent::Write(val) => { (has_write[b.index()], any_write, saw_effect, out) = (true, true, true, ClVal::Known(val)); }
                ClEvent::Observe => (saw_effect, out) = (true, ClVal::Unknown),
                ClEvent::Trap => break,
            }
        }
        fwd[b.index()] = if saw_read { BlockFwd::Dynamic } else if saw_effect { BlockFwd::Const(out) } else { BlockFwd::PassThrough };
    }
    if !any_write { return false; }
    let (mut in_val, mut out_val) = (vec![ClVal::Uninit; n], vec![ClVal::Uninit; n]);
    in_val[entry.index()] = ClVal::Unknown;
    loop {
        let mut changed = false;
        for &b in &cfg.rpo {
            let bi = b.index();
            let state = if b == entry { ClVal::Unknown } else { cfg.preds_of(b).iter().fold(ClVal::Uninit, |acc, &p| acc.join(out_val[p.index()])) };
            if state != in_val[bi] { (in_val[bi], changed) = (state, true); }
            let out = match fwd[bi] {
                BlockFwd::PassThrough => state,
                BlockFwd::Const(v) => v,
                BlockFwd::Dynamic => step_forward(func, prov, zeros, is_cl, b, state, |_| {}),
            };
            if out != out_val[bi] { (out_val[bi], changed) = (out, true); }
        }
        if !changed { break; }
    }
    let mut dropped = false;
    for &b in &cfg.rpo {
        if !has_write[b.index()] { continue; }
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
    let (mut gen_b, mut trans, mut has_write, mut any_write) = (vec![false; n], vec![false; n], vec![false; n], false);
    for &b in &cfg.rpo {
        let bi = b.index();
        let (mut g, mut t) = match func.last(b) { Some(Op::Trap) => (false, false), Some(Op::Return(..)) => (true, false), _ => (false, true) };
        for &inst in func.blocks[bi].insts.iter().rev() {
            match cl_event(func, prov, zeros, is_cl, &func.insts[inst.index()].op) {
                ClEvent::Write(..) => (g, t, has_write[bi], any_write) = (false, false, true, true),
                ClEvent::Read | ClEvent::Observe => (g, t) = (true, false),
                ClEvent::Trap | ClEvent::None => {}
            }
        }
        (gen_b[bi], trans[bi]) = (g, t);
    }
    if !any_write { return false; }
    let mut live_in = gen_b;
    loop {
        let mut changed = false;
        for &b in cfg.rpo.iter().rev() {
            let bi = b.index();
            if !live_in[bi] && trans[bi] && func.successors(b).any(|s| live_in[s.index()]) { (live_in[bi], changed) = (true, true); }
        }
        if !changed { break; }
    }
    let mut kills = Vec::new();
    for &b in &cfg.rpo {
        if !has_write[b.index()] { continue; }
        let mut live = match func.last(b) { Some(Op::Trap) => false, Some(Op::Return(..)) => true, _ => func.successors(b).any(|s| live_in[s.index()]) };
        for pos in (0..func.blocks[b.index()].insts.len()).rev() {
            let op = &func.insts[func.blocks[b.index()].insts[pos].index()].op;
            if matches!(cl_event(func, prov, zeros, is_cl, op), ClEvent::Write(..)) && !live { kills.push((b.index(), pos)); }
            live = step(op, live);
        }
    }
    let dropped = !kills.is_empty();
    for (b_idx, pos) in kills { let inst = func.blocks[b_idx].insts.remove(pos); func.insts[inst.index()].op = Op::Trap; }
    dropped
}

fn sink_cell_calls(func: &mut Function, cfg: &Cfg, zeros: &mut Vec<bool>, is_cl: &mut Vec<bool>, check_loads: bool) -> bool {
    let mut changed = false;
    if check_loads {
        let mut cl_loads: Vec<(Block, usize, Value)> = Vec::new();
        for &b in &cfg.rpo {
            for (pos, &inst) in func.blocks[b.index()].insts.iter().enumerate() {
                let d = &func.insts[inst.index()];
                if let (Op::Load(addr, 0), Some(res)) = (&d.op, d.result) && is_cl.get(addr.index()) == Some(&true) { cl_loads.push((b, pos, res)); }
            }
        }
        if !cl_loads.is_empty() {
            let (mut used, mut rem) = (vec![false; cl_loads.len()], cl_loads.len());
            for &b in &cfg.rpo {
                if rem == 0 { break; }
                for &inst in &func.blocks[b.index()].insts {
                    func.insts[inst.index()].op.for_operands(|v| {
                        for (idx, &(_, _, r)) in cl_loads.iter().enumerate() { if r == v && !used[idx] { used[idx] = true; rem -= 1; } }
                    });
                }
            }
            if rem > 0 {
                for (idx, &(b, pos, _)) in cl_loads.iter().enumerate().rev() {
                    if !used[idx] { let inst = func.blocks[b.index()].insts.remove(pos); func.insts[inst.index()].op = Op::Trap; changed = true; }
                }
            }
        }
    }
    let mut defs_list: Vec<(Value, Block, Inst, ClDef)> = Vec::new();
    if !zeros.is_empty() {
        for &b in &cfg.rpo {
            for &inst in &func.blocks[b.index()].insts {
                let data = &func.insts[inst.index()];
                if data.result.is_none()
                    && let Some(res) = func.caller_loc_result(&data.op, |v| zeros.get(v.index()) == Some(&true))
                    && let Op::Call { sig, callee, args, results } = &data.op
                    && results.len() == 1
                {
                    defs_list.push((res, b, inst, (*sig, *callee, args.clone())));
                }
            }
        }
    }
    if defs_list.is_empty() {
        return changed;
    }
    let defs_mask = defs_list.iter().fold(0_u128, |m, &(res, ..)| m | (1_u128 << (res.0 & 127)));
    let cl_slot = (defs_list.len() > 8).then(|| {
        let mut map = vec![u32::MAX; func.types.len()];
        for (idx, &(res, ..)) in defs_list.iter().enumerate() {
            map[res.index()] = idx as u32;
        }
        map
    });
    let mut all_use_blocks: Vec<Vec<Block>> = vec![Vec::new(); defs_list.len()];
    let mut all_ok = vec![true; defs_list.len()];
    for &b in &cfg.rpo {
        for &inst in &func.blocks[b.index()].insts {
            let op = &func.insts[inst.index()].op;
            op.for_operands(|v| {
                if defs_mask & (1_u128 << (v.0 & 127)) == 0 {
                    return;
                }
                let slot = match &cl_slot {
                    Some(map) => map.get(v.index()).copied().filter(|&s| s != u32::MAX).map(|s| s as usize),
                    None => defs_list.iter().position(|&(res, ..)| res == v),
                };
                if let Some(si) = slot {
                    if all_ok[si] {
                        let valid = matches!(*op, Op::Load(a, 0) if a == v)
                            || matches!(*op, Op::Store(val, a, 0) if a == v && val != v);
                        if b == defs_list[si].1 || !valid {
                            all_ok[si] = false;
                        } else if all_use_blocks[si].last() != Some(&b) {
                            all_use_blocks[si].push(b);
                        }
                    }
                }
            });
        }
    }
    let loops = cfg.loops();
    let n = func.blocks.len();
    for (si, (res, b_def, def_inst, (sig, callee, args))) in defs_list.into_iter().enumerate() {
        if !all_ok[si] {
            continue;
        }
        let use_blocks = std::mem::take(&mut all_use_blocks[si]);
        if use_blocks.is_empty() {
            if let Some(pos) = func.blocks[b_def.index()].insts.iter().position(|&i| i == def_inst) {
                func.blocks[b_def.index()].insts.remove(pos);
                func.insts[def_inst.index()].op = Op::Trap;
                changed = true;
            }
            continue;
        }
        if use_blocks.len() > MAX_NARROW_SITES || use_blocks.iter().any(|&u| loops.depth_of(u) > loops.depth_of(b_def)) {
            continue;
        }
        let mut is_use = vec![false; n];
        for &u in &use_blocks {
            is_use[u.index()] = true;
        }
        let mut visited = vec![false; n];
        visited[b_def.index()] = true;
        let mut work: Vec<Block> = func.successors(b_def).filter(|s| !is_use[s.index()]).collect();
        let mut has_bypass = false;
        while let Some(u) = work.pop() {
            if matches!(func.last(u), Some(Op::Return(..) | Op::Trap)) {
                has_bypass = true;
                break;
            }
            if std::mem::replace(&mut visited[u.index()], true) {
                has_bypass = true;
                break;
            }
            for succ in func.successors(u) {
                if !is_use[succ.index()] {
                    work.push(succ);
                }
            }
        }
        if !has_bypass {
            continue;
        }
        let Some(pos) = func.blocks[b_def.index()].insts.iter().position(|&i| i == def_inst) else {
            continue;
        };
        func.blocks[b_def.index()].insts.remove(pos);
        func.insts[def_inst.index()].op = Op::Trap;
        for u in use_blocks {
            let cl_u = push_cl_value(func, zeros, is_cl, true);
            let cl_inst = func.add(Op::Call { sig, callee, args: args.clone(), results: vec![cl_u] }, None);
            let insts = &mut func.blocks[u.index()].insts;
            let mut first = None;
            for (idx, &inst) in insts.iter().enumerate() {
                match &mut func.insts[inst.index()].op {
                    Op::Load(a, 0) | Op::Store(_, a, 0) if *a == res => {
                        *a = cl_u;
                        first.get_or_insert(idx);
                    }
                    _ => {}
                }
            }
            if let Some(at) = first {
                insts.insert(at, cl_inst);
            }
        }
        changed = true;
    }
    changed
}

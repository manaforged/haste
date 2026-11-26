use crate::cfg::Cfg;
use crate::ir::{Block, Function, Op, Value};
use crate::mem::Provenance;

use super::caller_loc::{ClDef, ClEvent, ClVal, MAX_NARROW_SITES, cl_event, collect_cl_list, push_cl_value};

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

pub(super) fn drop_silent_stores(func: &mut Function, cfg: &Cfg, prov: &Provenance, zeros: &[bool], is_cl: &[bool]) -> bool {
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

pub(super) fn drop_dead_stores(func: &mut Function, cfg: &Cfg, prov: &Provenance, zeros: &[bool], is_cl: &[bool]) -> bool {
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

pub(super) fn sink_cell_calls(
    func: &mut Function,
    cfg: &Cfg,
    zeros: &mut Vec<bool>,
    is_cl: &mut Vec<bool>,
    check_loads: bool,
    pre_defs: Option<Vec<(Value, Block, crate::ir::Inst, ClDef)>>,
) -> bool {
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
            let load_mask = cl_loads.iter().fold(0_u128, |m, &(_, _, r)| m | (1_u128 << (r.0 & 127)));
            let (mut used, mut rem) = (vec![false; cl_loads.len()], cl_loads.len());
            for &b in &cfg.rpo {
                if rem == 0 { break; }
                for &inst in &func.blocks[b.index()].insts {
                    func.insts[inst.index()].op.for_operands(|v| {
                        if load_mask & (1_u128 << (v.0 & 127)) == 0 { return; }
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
    let defs_list = if !changed && let Some(pre) = pre_defs { pre } else { collect_cl_list(func, cfg, zeros) };
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

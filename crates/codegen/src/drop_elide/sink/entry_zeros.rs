use crate::cfg::Cfg;
use crate::ir::{Block, Function, Inst, Op, StackSlot, Value};

use super::super::{CalleeKind, FlagOp, classify_callee, classify_flag_op, store_clears_active};
use super::frontier::{
    call_bounded_blocks, cycle_depths, dominated_clean_region, prune_redundant_frontier,
    sink_frontier_or_solo, solve_liveness,
};

fn flag_always_zero(
    func: &Function,
    defs: &[Option<Inst>],
    entry: Block,
    flag: StackSlot,
) -> (Vec<bool>, Vec<bool>) {
    let n = func.blocks.len();
    let (mut in_state, mut out_state) = (vec![0u8; n], vec![0u8; n]);
    in_state[entry.index()] = 1;
    let mut work = vec![(entry, false)];
    while let Some((block, mut active)) = work.pop() {
        let mut stopped = false;
        for &inst in &func.blocks[block.index()].insts {
            match func.insts[inst.index()].op {
                Op::StackStore(v, s, 0) if s == flag => {
                    active = !store_clears_active(func, defs, v)
                }
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
    (
        in_state.iter().map(|&v| v == 1).collect(),
        out_state.iter().map(|&v| v == 1).collect(),
    )
}

pub(in super::super) fn entry_first_zeros(
    func: &Function,
    defs: &[Option<Inst>],
    entry: Block,
    nslots: usize,
) -> Vec<Option<Inst>> {
    let mut first = vec![None; nslots];
    let mut seen_call = false;
    for &inst in &func.blocks[entry.index()].insts {
        match func.insts[inst.index()].op {
            Op::Call { .. } | Op::TryCall { .. } => seen_call = true,
            Op::StackStore(v, slot, 0)
                if !seen_call
                    && slot.index() < nslots
                    && first[slot.index()].is_none()
                    && store_clears_active(func, defs, v) =>
            {
                first[slot.index()] = Some(inst);
            }
            _ => {}
        }
    }
    first
}

fn elide_redundant_flag_zeros(
    func: &mut Function,
    entry: Block,
    defs: &[Option<Inst>],
    (is_flag, is_local, slot_escaped): (&[bool], &[bool], &[bool]),
) -> bool {
    let n = func.blocks.len();
    let nslots = func.slots.len();
    let mut dead = vec![false; func.insts.len()];
    let mut any = false;
    for fi in 0..nslots {
        if !is_flag[fi] || is_local[fi] || slot_escaped[fi] || func.slots[fi].size != 1 {
            continue;
        }
        let flag = StackSlot(fi as u32);
        let mut in_state = vec![0u8; n];
        in_state[entry.index()] = 2;
        let mut work = vec![(entry, false)];
        while let Some((block, mut known_zero)) = work.pop() {
            let mut stopped = false;
            for &inst in &func.blocks[block.index()].insts {
                match func.insts[inst.index()].op {
                    Op::StackStore(v, s, off) if s == flag => {
                        known_zero = off == 0 && store_clears_active(func, defs, v);
                    }
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
            let bit = if known_zero { 1 } else { 2 };
            for target in func.successors(block) {
                if in_state[target.index()] & bit == 0 {
                    in_state[target.index()] |= bit;
                    work.push((target, known_zero));
                }
            }
        }
        for (bi, &st) in in_state.iter().enumerate() {
            if st == 0 {
                continue;
            }
            let mut known_zero = st == 1;
            for &inst in &func.blocks[bi].insts {
                if let Op::StackStore(v, s, off) = func.insts[inst.index()].op
                    && s == flag
                {
                    if off == 0 && store_clears_active(func, defs, v) {
                        if known_zero {
                            dead[inst.index()] = true;
                            any = true;
                        }
                        known_zero = true;
                    } else {
                        known_zero = false;
                    }
                }
            }
        }
    }
    if !any {
        return false;
    }
    for (idx, &rm) in dead.iter().enumerate() {
        if rm {
            func.insts[idx].op = Op::Trap;
        }
    }
    for block in &mut func.blocks {
        block.insts.retain(|i| !dead[i.index()]);
    }
    true
}

pub(in super::super) fn sink_entry_zeros(
    func: &mut Function,
    entry: Block,
    slot_escaped: &[bool],
    sym_may_unwind: &mut impl FnMut(u32) -> bool,
) -> bool {
    let nslots = func.slots.len();
    let defs = crate::fold::defs(func);
    let mut is_local = vec![false; nslots];
    let mut is_flag = vec![false; nslots];
    for u in &func.unwind_drops {
        if u.local.index() < nslots {
            is_local[u.local.index()] = true;
        }
        if u.flag.index() < nslots {
            is_flag[u.flag.index()] = true;
        }
    }
    let elided =
        elide_redundant_flag_zeros(func, entry, &defs, (&is_flag, &is_local, slot_escaped));
    let kinds: Vec<CalleeKind> = func
        .insts
        .iter()
        .map(|data| match &data.op {
            Op::Call { callee, args, .. } => {
                classify_callee(func, &defs, *callee, args, sym_may_unwind)
            }
            _ => CalleeKind::MayUnwind,
        })
        .collect();
    let first = entry_first_zeros(func, &defs, entry, nslots);
    let cfg = func.cfg();
    let depths = cycle_depths(func, &cfg);
    let cold = crate::live::cold(func);
    let call_bounded: Vec<bool> = call_bounded_blocks(func, &cfg, &kinds)
        .into_iter()
        .zip(&cold)
        .map(|(bounded, &is_cold)| bounded && is_cold)
        .collect();
    let mut remove = Vec::new();
    let mut insert: Vec<(Block, Value, StackSlot)> = Vec::new();
    for fi in 0..nslots {
        if !is_flag[fi] || is_local[fi] || slot_escaped[fi] {
            continue;
        }
        let Some(zinst) = first[fi] else {
            continue;
        };
        let Op::StackStore(val, flag, 0) = func.insts[zinst.index()].op else {
            continue;
        };
        let Some(frontier) = plan_entry_zero_sink(
            func,
            &cfg,
            &defs,
            (&kinds, &depths, &call_bounded),
            (entry, flag, zinst),
        ) else {
            continue;
        };
        remove.push(zinst);
        for t in frontier {
            insert.push((t, val, flag));
        }
    }
    if remove.is_empty() {
        return elided;
    }
    for &i in &remove {
        func.insts[i.index()].op = Op::Trap;
    }
    func.blocks[entry.index()]
        .insts
        .retain(|i| !remove.contains(i));
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
    (kinds, depths, call_bounded): (&[CalleeKind], &[u32], &[bool]),
    (entry, flag, zinst): (Block, StackSlot, Inst),
) -> Option<Vec<Block>> {
    if cfg.pred_count(entry) != 0 {
        return None;
    }
    let insts = &func.blocks[entry.index()].insts;
    let pos = insts.iter().position(|&i| i == zinst)?;
    for &i in &insts[pos + 1..] {
        match classify_flag_op(func, defs, kinds, flag, i) {
            Some(FlagOp::MayUnwindCall | FlagOp::TryCall | FlagOp::Load) => return None,
            Some(FlagOp::Clear | FlagOp::Set | FlagOp::Exit) => return Some(Vec::new()),
            None => {}
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
            match classify_flag_op(func, defs, kinds, flag, inst) {
                Some(FlagOp::MayUnwindCall | FlagOp::TryCall | FlagOp::Load) => {
                    gen_b[bi] = true;
                    clean_b[bi] = false;
                    break;
                }
                Some(FlagOp::Clear | FlagOp::Set) => {
                    kill_b[bi] = true;
                    clean_b[bi] = false;
                    break;
                }
                Some(FlagOp::Exit) => {
                    kill_b[bi] = true;
                    break;
                }
                None => {}
            }
        }
    }
    let (live_in, _) = solve_liveness(func, cfg, &gen_b, &kill_b);
    if !func.successors(entry).any(|s| live_in[s.index()]) {
        return Some(Vec::new());
    }
    let (always_in, always_out) = flag_always_zero(func, defs, entry, flag);
    let mut in_r = dominated_clean_region(func, cfg, entry, &live_in, |s| clean_b[s.index()]);
    let mut frontier = sink_frontier_or_solo(
        func,
        cfg,
        depths,
        (
            &live_in,
            Some((&always_in, &always_out)),
            Some(call_bounded),
            entry,
            8,
        ),
        &mut in_r,
    )?;
    prune_redundant_frontier(func, cfg, &live_in, |s| clean_b[s.index()], &mut frontier);
    Some(frontier)
}

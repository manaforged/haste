use crate::cfg::{Cfg, Loops};
use crate::ir::{Block, Function, Inst, Op, Value};
use crate::mem::Provenance;

use super::{
    ClDef, ClEvent, ClWrap, MAX_NARROW_SITES, cl_event, has_fast_path, push_unvisited_succs,
    tail_is_cl_clean, wrap_cl_block, wrap_cl_region,
};

fn first_restore_inst(
    func: &Function,
    prov: &Provenance,
    zeros: &[bool],
    is_cl: &[bool],
    b: Block,
    saved: Value,
) -> Option<Inst> {
    for &inst in &func.blocks[b.index()].insts {
        match cl_event(func, prov, zeros, is_cl, &func.insts[inst.index()].op) {
            ClEvent::None => {}
            ClEvent::Write(w) => return (w == saved).then_some(inst),
            _ => return None,
        }
    }
    None
}

fn starts_with_trap(
    func: &Function,
    prov: &Provenance,
    zeros: &[bool],
    is_cl: &[bool],
    b: Block,
) -> bool {
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
    loops: &Loops,
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
            if s != root
                && cfg.dominates(b_l, s)
                && let Some(r_inst) = first_restore_inst(func, prov, zeros, is_cl, s, saved)
            {
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
            if region.len() >= 16
                || matches!(func.last(s), Some(Op::Return(..) | Op::TryCall { .. }))
            {
                return None;
            }
            if func.blocks[s.index()].insts.iter().any(|&i| {
                matches!(
                    cl_event(func, prov, zeros, is_cl, &func.insts[i.index()].op),
                    ClEvent::Write(w) if w == saved
                )
            }) {
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

fn collect_load_cands(func: &Function, cfg: &Cfg, is_cl: &[bool]) -> Vec<(Block, Inst)> {
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
}

fn mark_matching_saved(val: Value, saved_mask: u128, saved_vals: &[Value], flags: &mut [bool]) {
    if saved_mask & (1_u128 << (val.0 & 127)) != 0 {
        for (idx, &sv) in saved_vals.iter().enumerate() {
            if sv == val {
                flags[idx] = true;
            }
        }
    }
}

fn analyze_saved_uses(
    func: &Function,
    cfg: &Cfg,
    is_cl: &[bool],
    candidates: &[(Block, Inst)],
) -> (Vec<bool>, Vec<u32>) {
    let saved_vals: Vec<Value> = candidates
        .iter()
        .filter_map(|&(_, i)| func.insts[i.index()].result)
        .collect();
    let saved_mask = saved_vals
        .iter()
        .fold(0_u128, |m, &v| m | (1_u128 << (v.0 & 127)));
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
                        if sv == val {
                            restore_counts[idx] += 1;
                        }
                    }
                }
                mark_matching_saved(addr, saved_mask, &saved_vals, &mut non_restore);
            } else {
                data.op.for_operands(|v| {
                    mark_matching_saved(v, saved_mask, &saved_vals, &mut non_restore);
                });
            }
        }
    }
    (non_restore, restore_counts)
}

enum SaveBlockScan {
    Restore(Inst),
    Trap,
    Bad,
    Clean,
    Dirty {
        first_write: usize,
        last_event: usize,
    },
}

fn scan_save_block(
    func: &Function,
    prov: &Provenance,
    zeros: &[bool],
    is_cl: &[bool],
    u: Block,
    saved: Value,
) -> SaveBlockScan {
    let mut first_write = None;
    let mut last_event = 0;
    for (idx, &inst) in func.blocks[u.index()].insts.iter().enumerate() {
        match cl_event(func, prov, zeros, is_cl, &func.insts[inst.index()].op) {
            ClEvent::None => {}
            ClEvent::Write(w) => {
                if first_write.is_none() {
                    if w == saved {
                        return SaveBlockScan::Restore(inst);
                    }
                    (first_write, last_event) = (Some(idx), idx);
                } else if w == saved {
                    return SaveBlockScan::Bad;
                } else {
                    last_event = idx;
                }
            }
            ClEvent::Trap => {
                if first_write.is_none() {
                    return SaveBlockScan::Trap;
                }
                break;
            }
            ClEvent::Read | ClEvent::Observe => {
                if first_write.is_none() {
                    return SaveBlockScan::Bad;
                }
                last_event = idx;
            }
        }
    }
    match first_write {
        Some(first_write) => SaveBlockScan::Dirty {
            first_write,
            last_event,
        },
        None => SaveBlockScan::Clean,
    }
}

struct SavePlan {
    dirty: Vec<(Block, usize, usize)>,
    regions: Vec<(Block, usize, Vec<Block>, Vec<Block>)>,
    hit_restores: Vec<(Block, Inst)>,
}

#[derive(Default)]
struct SaveScratch {
    visited: Vec<bool>,
    pass: Vec<bool>,
    restores: Vec<bool>,
}

impl SaveScratch {
    fn reset(&mut self, n: usize) {
        if self.visited.is_empty() {
            self.visited = vec![false; n];
            self.pass = vec![false; n];
            self.restores = vec![false; n];
        } else {
            self.visited.fill(false);
            self.pass.fill(false);
            self.restores.fill(false);
        }
    }
}

fn plan_save_narrow(
    func: &Function,
    cfg: &Cfg,
    loops: &Loops,
    prov: &Provenance,
    (zeros, is_cl): (&[bool], &[bool]),
    (b_l, saved, restore_count): (Block, Value, u32),
    scratch: &mut SaveScratch,
) -> Option<SavePlan> {
    scratch.reset(func.blocks.len());
    let mut dirty: Vec<(Block, usize, usize)> = Vec::new();
    let mut regions: Vec<(Block, usize, Vec<Block>, Vec<Block>)> = Vec::new();
    let mut hit_restores: Vec<(Block, Inst)> = Vec::new();
    let mut work: Vec<Block> = Vec::new();
    push_unvisited_succs(func, b_l, &mut scratch.visited, &mut work);
    if work.is_empty() {
        return None;
    }
    let depth_l = loops.depth_of(b_l);
    while let Some(u) = work.pop() {
        if u == b_l {
            return None;
        }
        match scan_save_block(func, prov, zeros, is_cl, u, saved) {
            SaveBlockScan::Restore(r_inst) => {
                if !cfg.dominates(b_l, u) {
                    return None;
                }
                scratch.restores[u.index()] = true;
                if !hit_restores.iter().any(|&(_, ri)| ri == r_inst) {
                    hit_restores.push((u, r_inst));
                }
            }
            SaveBlockScan::Trap => {}
            SaveBlockScan::Bad => return None,
            SaveBlockScan::Dirty {
                first_write,
                last_event,
            } => {
                if !cfg.dominates(b_l, u) {
                    return None;
                }
                let last_op = func.last(u);
                if matches!(last_op, Some(Op::Return(..) | Op::TryCall { .. }))
                    || loops.depth_of(u) > depth_l
                    || dirty.len() + regions.len() >= MAX_NARROW_SITES
                {
                    return None;
                }
                if let Some((region, exits)) = collect_dirty_region(
                    func,
                    cfg,
                    loops,
                    prov,
                    (zeros, is_cl),
                    (b_l, saved, u, depth_l),
                    &mut scratch.restores,
                    &mut hit_restores,
                ) {
                    regions.push((u, first_write, region, exits));
                } else if !matches!(last_op, Some(Op::Trap)) {
                    dirty.push((u, first_write, last_event));
                    push_unvisited_succs(func, u, &mut scratch.visited, &mut work);
                }
            }
            SaveBlockScan::Clean => {
                if !cfg.dominates(b_l, u) {
                    return None;
                }
                scratch.pass[u.index()] = true;
                push_unvisited_succs(func, u, &mut scratch.visited, &mut work);
            }
        }
    }
    if (dirty.is_empty() && regions.is_empty())
        || hit_restores.len() != restore_count as usize
        || !has_fast_path(func, b_l, &scratch.pass, &scratch.restores)
    {
        return None;
    }
    Some(SavePlan {
        dirty,
        regions,
        hit_restores,
    })
}

pub(super) fn narrow_saves(
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
    let candidates = pre_cands.unwrap_or_else(|| collect_load_cands(func, cfg, is_cl));
    if candidates.is_empty() {
        return false;
    }
    let (non_restore, restore_counts) = analyze_saved_uses(func, cfg, is_cl, &candidates);
    let mut loops = None;
    let mut scratch = SaveScratch::default();
    let mut changed = false;
    for (ci, (b_l, load_inst)) in candidates.into_iter().enumerate() {
        let Some(pos_l) = func.blocks[b_l.index()]
            .insts
            .iter()
            .position(|&i| i == load_inst)
        else {
            continue;
        };
        let data = &func.insts[load_inst.index()];
        let (&Op::Load(addr, 0), Some(saved)) = (&data.op, data.result) else {
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
        let Some(plan) = plan_save_narrow(
            func,
            cfg,
            loops,
            prov,
            (zeros, is_cl),
            (b_l, saved, restore_counts[ci]),
            &mut scratch,
        ) else {
            continue;
        };
        func.blocks[b_l.index()].insts.remove(pos_l);
        func.insts[load_inst.index()].op = Op::Trap;
        for (rb, r_inst) in plan.hit_restores {
            if let Some(rpos) = func.blocks[rb.index()]
                .insts
                .iter()
                .position(|&i| i == r_inst)
            {
                func.blocks[rb.index()].insts.remove(rpos);
                func.insts[r_inst.index()].op = Op::Trap;
            }
        }
        for (u, first, last) in plan.dirty {
            wrap_cl_block(
                func,
                (zeros, is_cl),
                &def,
                (u, first, last),
                ClWrap::SaveLocal(addr),
            );
        }
        for (root, fw, region, exits) in plan.regions {
            wrap_cl_region(
                func,
                (zeros, is_cl),
                &def,
                (root, fw, addr),
                (region, exits),
            );
        }
        changed = true;
    }
    changed
}

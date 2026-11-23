use crate::cfg::Cfg;
use crate::ir::{Block, Function, Inst, Op, StackSlot, Value};
use crate::mem::{MemLoc, MemRoot, Provenance, clobbers};

/// The most blocks one group of loads may sink through.
const MAX_REGION: usize = 1024;

/// The fewest loads from one address that sink without a call on the path, since holding that many words at once spills them.
const WIDE_GROUP: usize = 8;

/// A load that may move: its position in its block, its address, offset and result.
#[derive(Clone, Copy)]
struct Candidate {
    pos: usize,
    addr: Value,
    offset: i32,
    result: Value,
}

/// Moves each load whose every use is in later blocks into those blocks when no path between them may write the loaded bytes and some path passes a call or the load is one of a wide group from one address, so the value no longer lives across that call or beside the rest of the group.
pub(super) fn sink_loads(func: &mut Function, prov: &Provenance) -> bool {
    let cfg = func.cfg().clone();
    let mut changed = sink_stack_stores(func, &cfg, prov);
    let users = user_blocks(func);
    let mut rank = vec![usize::MAX; func.blocks.len()];
    for (index, block) in cfg.rpo.iter().enumerate() {
        rank[block.index()] = index;
    }
    for &block in &cfg.rpo {
        changed |= sink_block(func, (&cfg, &rank), prov, &users, block);
    }
    changed
}

fn user_blocks(func: &Function) -> Vec<Vec<Block>> {
    let mut users = vec![Vec::new(); func.types.len()];
    for (index, data) in func.blocks.iter().enumerate() {
        for inst in &data.insts {
            func.insts[inst.index()].op.for_operands(|v| {
                let list: &mut Vec<Block> = &mut users[v.index()];
                if list.last() != Some(&Block(index as u32)) {
                    list.push(Block(index as u32));
                }
            });
        }
    }
    users
}

fn candidates(func: &Function, users: &[Vec<Block>], block: Block) -> Vec<Candidate> {
    let insts = &func.blocks[block.index()].insts;
    insts.iter().enumerate().filter_map(|(pos, inst)| {
        let data = &func.insts[inst.index()];
        let Op::Load(addr, offset) = data.op else { return None };
        let result = data.result?;
        let blocks = users.get(result.index())?;
        (!blocks.is_empty() && !blocks.contains(&block)).then_some(Candidate { pos, addr, offset, result })
    }).collect()
}

fn sink_block(func: &mut Function, (cfg, rank): (&Cfg, &[usize]), prov: &Provenance, users: &[Vec<Block>], block: Block) -> bool {
    let found = candidates(func, users, block);
    let mut addrs: Vec<Value> = found.iter().map(|c| c.addr).collect();
    addrs.sort_unstable();
    addrs.dedup();
    let mut moved: Vec<Inst> = Vec::new();
    for addr in addrs {
        let group: Vec<Candidate> = found.iter().copied().filter(|c| c.addr == addr).collect();
        let Some(loc) = span(func, prov, &group) else { continue };
        let first = group.iter().map(|c| c.pos).min().unwrap_or(0);
        let insts = &func.blocks[block.index()].insts;
        if insts[first + 1..].iter().any(|inst| clobbers(func, &func.insts[inst.index()].op, loc, prov)) {
            continue;
        }
        let region = clean_region(func, cfg, prov, block, loc);
        let depth = cfg.loops().depth_of(block);
        let wide = group.len() >= WIDE_GROUP;
        for load in group {
            let targets = &users[load.result.index()];
            let fits = targets.iter().all(|&target| {
                region[target.index()].is_some_and(|first_write| {
                    cfg.loops().depth_of(target) <= depth && first_use(func, target, load.result).is_some_and(|pos| pos < first_write)
                })
            }) && (wide || crosses_call(func, cfg, (block, load.pos), &region, (targets, load.result)));
            if fits {
                moved.push(func.blocks[block.index()].insts[load.pos]);
                let mut ordered = targets.clone();
                ordered.sort_unstable_by_key(|target| rank[target.index()]);
                let mut clones: Vec<(Block, Value)> = Vec::new();
                for target in ordered {
                    let shared = clones.iter().find(|&&(at, _)| region[at.index()] == Some(usize::MAX) && cfg.dominates(at, target)).map(|&(_, value)| value);
                    let value = clone_into(func, target, load, shared);
                    clones.push((target, value));
                }
            }
        }
    }
    if moved.is_empty() {
        return false;
    }
    func.blocks[block.index()].insts.retain(|inst| !moved.contains(inst));
    true
}

fn span(func: &Function, prov: &Provenance, group: &[Candidate]) -> Option<MemLoc> {
    let lo = group.iter().map(|c| i64::from(c.offset)).min()?;
    let hi = group.iter().map(|c| i64::from(c.offset) + i64::from(func.ty(c.result).bytes())).max()?;
    Some(prov.loc(group[0].addr, lo, u32::try_from(hi - lo).ok()?))
}

/// For each block `from` dominates that every path from `from` reaches without passing an instruction that may write `loc`, the position of its first such instruction, or `usize::MAX`.
fn clean_region(func: &Function, cfg: &Cfg, prov: &Provenance, from: Block, loc: MemLoc) -> Vec<Option<usize>> {
    let mut region: Vec<Option<usize>> = vec![None; func.blocks.len()];
    let mut members: Vec<Block> = Vec::new();
    let mut inside = vec![false; func.blocks.len()];
    for &block in &cfg.rpo {
        if members.len() >= MAX_REGION {
            break;
        }
        let dominated = block != from && cfg.idom[block.index()].is_some_and(|idom| idom == from || inside[idom.index()]);
        if dominated {
            inside[block.index()] = true;
            members.push(block);
            region[block.index()] = Some(func.blocks[block.index()].insts.iter().position(|inst| clobbers(func, &func.insts[inst.index()].op, loc, prov)).unwrap_or(usize::MAX));
        }
    }
    let mut changed = true;
    while changed {
        changed = false;
        for &block in &members {
            if region[block.index()].is_some() && !cfg.preds_of(block).iter().all(|&pred| pred == from || region[pred.index()] == Some(usize::MAX)) {
                region[block.index()] = None;
                changed = true;
            }
        }
    }
    region
}

fn is_call(func: &Function, inst: Inst) -> bool {
    matches!(func.insts[inst.index()].op, Op::Call { .. } | Op::TryCall { .. })
}

/// Whether a path from the load at `at` to a use of `result` passes a call, so the loaded value would live across it.
fn crosses_call(func: &Function, cfg: &Cfg, (from, pos): (Block, usize), region: &[Option<usize>], (targets, result): (&[Block], Value)) -> bool {
    if func.blocks[from.index()].insts[pos + 1..].iter().any(|&inst| is_call(func, inst)) {
        return true;
    }
    let mut seen = vec![false; func.blocks.len()];
    let mut work: Vec<Block> = Vec::new();
    for &target in targets {
        let before = first_use(func, target, result).unwrap_or(0);
        if func.blocks[target.index()].insts[..before].iter().any(|&inst| is_call(func, inst)) {
            return true;
        }
        work.extend(cfg.preds_of(target).iter().copied());
    }
    while let Some(block) = work.pop() {
        if block == from || seen[block.index()] || region[block.index()].is_none() {
            continue;
        }
        seen[block.index()] = true;
        if func.blocks[block.index()].insts.iter().any(|&inst| is_call(func, inst)) {
            return true;
        }
        work.extend(cfg.preds_of(block).iter().copied());
    }
    false
}

fn first_use(func: &Function, block: Block, value: Value) -> Option<usize> {
    func.blocks[block.index()].insts.iter().position(|inst| {
        let mut uses = false;
        func.insts[inst.index()].op.for_operands(|v| uses |= v == value);
        uses
    })
}

/// Points the uses of the load's result in `target` at `shared`, a copy loaded in a dominating block, or at a new copy loaded before the first use; returns the value they now read.
fn clone_into(func: &mut Function, target: Block, load: Candidate, shared: Option<Value>) -> Value {
    let pos = first_use(func, target, load.result).unwrap_or(0);
    let value = shared.unwrap_or_else(|| func.value(func.ty(load.result)));
    let insts = func.blocks[target.index()].insts.clone();
    for &user in &insts[pos..] {
        func.insts[user.index()].op.map_operands(|v| if v == load.result { value } else { v });
    }
    if shared.is_none() {
        let inst = func.add(Op::Load(load.addr, load.offset), Some(value));
        func.blocks[target.index()].insts.insert(pos, inst);
    }
    value
}

fn is_full_slot_store(func: &Function, op: &Op, slot: StackSlot) -> bool {
    matches!(*op, Op::StackStore(v, s, 0) if s == slot && func.ty(v).bytes() == func.slots[slot.index()].size)
}

fn inst_reads_slot(func: &Function, inst: Inst, slot: StackSlot, prov: &Provenance) -> bool {
    if prov.slot_escaped_at(slot, Some(inst)) {
        return true;
    }
    let op = &func.insts[inst.index()].op;
    match *op {
        Op::StackLoad(s, _) | Op::StackAddr(s, _) => s == slot,
        Op::Load(addr, _) | Op::AtomicLoad(addr, _) | Op::AtomicRmw(_, _, addr, _) | Op::AtomicCas(_, addr, _, _) => {
            prov.loc(addr, 0, 1).root == MemRoot::Slot(slot)
        }
        Op::Move(_, src, _) | Op::Copy(_, src, _) => prov.loc(src, 0, 1).root == MemRoot::Slot(slot),
        _ => false,
    }
}

fn slot_liveness(func: &Function, cfg: &Cfg, prov: &Provenance, slot: StackSlot) -> (Vec<bool>, Vec<bool>) {
    let n = func.blocks.len();
    let (mut reads, mut kill) = (vec![false; n], vec![false; n]);
    for &b in &cfg.rpo {
        let bi = b.index();
        for &inst in &func.blocks[bi].insts {
            if inst_reads_slot(func, inst, slot, prov) {
                reads[bi] = true;
                break;
            }
            if is_full_slot_store(func, &func.insts[inst.index()].op, slot) {
                kill[bi] = true;
                break;
            }
        }
    }
    let (mut live_in, mut live_out) = (reads.clone(), vec![false; n]);
    let mut changed = true;
    while changed {
        changed = false;
        for &b in cfg.rpo.iter().rev() {
            let bi = b.index();
            let out = func.successors(b).any(|s| live_in[s.index()]);
            let inp = reads[bi] || (out && !kill[bi]);
            if out != live_out[bi] || inp != live_in[bi] {
                live_out[bi] = out;
                live_in[bi] = inp;
                changed = true;
            }
        }
    }
    (live_in, live_out)
}

fn sink_pure_defs_into(func: &mut Function, from: Block, to: Block) {
    let mut uses = func.use_counts();
    let mut to_move = Vec::new();
    for &inst in &func.blocks[to.index()].insts {
        func.insts[inst.index()].op.for_operands(|v| {
            if uses[v.index()] > 0 {
                uses[v.index()] -= 1;
            }
        });
    }
    let from_insts = &func.blocks[from.index()].insts;
    for &inst in from_insts.iter().rev() {
        let data = &func.insts[inst.index()];
        let Some(res) = data.result else { continue };
        if !data.op.is_speculable() || matches!(data.op, Op::Const(_)) {
            continue;
        }
        if uses[res.index()] != 0 {
            continue;
        }
        let used_in_to = func.blocks[to.index()].insts.iter().chain(to_move.iter()).any(|&u| {
            let mut hit = false;
            func.insts[u.index()].op.for_operands(|v| hit |= v == res);
            hit
        });
        if !used_in_to {
            continue;
        }
        data.op.for_operands(|v| {
            if uses[v.index()] > 0 {
                uses[v.index()] -= 1;
            }
        });
        to_move.push(inst);
    }
    if to_move.is_empty() {
        return;
    }
    to_move.reverse();
    func.blocks[from.index()].insts.retain(|i| !to_move.contains(i));
    func.blocks[to.index()].insts.splice(0..0, to_move);
}

fn sink_stack_stores(func: &mut Function, cfg: &Cfg, prov: &Provenance) -> bool {
    let mut cand_slots: Vec<StackSlot> = Vec::new();
    for &b in &cfg.rpo {
        for &inst in &func.blocks[b.index()].insts {
            if let Op::StackStore(_, slot, _) = func.insts[inst.index()].op
                && !func.is_unwind_slot(slot)
                && !prov.slot_escaped_at(slot, Some(inst))
                && !cand_slots.contains(&slot)
            {
                cand_slots.push(slot);
            }
        }
    }
    if cand_slots.is_empty() {
        return false;
    }
    let mut changed = false;
    for slot in cand_slots {
        let (live_in, live_out) = slot_liveness(func, cfg, prov, slot);
        for &b in &cfg.rpo {
            let bi = b.index();
            let insts = func.blocks[bi].insts.clone();
            let mut dead = Vec::new();
            let mut sink_to: Option<(Block, Vec<Inst>)> = None;
            let brif_arms = match func.last(b) {
                Some(&Op::Brif(_, then, _, other, _)) if then != other => Some((then, other)),
                _ => None,
            };
            let mut live_after = live_out[bi];
            let mut any_access_after = false;
            for &inst in insts.iter().rev() {
                let op = &func.insts[inst.index()].op;
                if let Op::StackStore(_, s, _) = *op
                    && s == slot
                    && !prov.slot_escaped_at(slot, Some(inst))
                {
                    if !live_after {
                        dead.push(inst);
                        continue;
                    }
                    if !any_access_after
                        && let Some((then, other)) = brif_arms
                        && live_in[then.index()] != live_in[other.index()]
                    {
                        let target = if live_in[then.index()] { then } else { other };
                        if cfg.single_pred(target) == Some(b) && cfg.loops().depth_of(target) <= cfg.loops().depth_of(b) {
                            sink_to.get_or_insert_with(|| (target, Vec::new())).1.push(inst);
                            if is_full_slot_store(func, op, slot) {
                                live_after = false;
                            }
                            continue;
                        }
                    }
                    any_access_after = true;
                    if is_full_slot_store(func, op, slot) {
                        live_after = false;
                    }
                } else {
                    if inst_reads_slot(func, inst, slot, prov) {
                        live_after = true;
                        any_access_after = true;
                    } else if matches!(*op, Op::StackStore(_, s, _) if s == slot) {
                        any_access_after = true;
                    }
                }
            }
            if !dead.is_empty() {
                for &inst in &dead {
                    func.insts[inst.index()].op = Op::Trap;
                }
                func.blocks[bi].insts.retain(|i| !dead.contains(i));
                changed = true;
            }
            if let Some((target, mut sunk)) = sink_to {
                sunk.reverse();
                func.blocks[bi].insts.retain(|i| !sunk.contains(i));
                func.blocks[target.index()].insts.splice(0..0, sunk.iter().copied());
                sink_pure_defs_into(func, b, target);
                changed = true;
            }
        }
    }
    changed
}

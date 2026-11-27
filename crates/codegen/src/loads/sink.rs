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
    let Some(users) = load_user_blocks(func) else {
        return changed;
    };
    let mut rank = vec![usize::MAX; func.blocks.len()];
    for (index, block) in cfg.rpo.iter().enumerate() {
        rank[block.index()] = index;
    }
    for &block in &cfg.rpo {
        changed |= sink_block(func, (&cfg, &rank), prov, &users, block);
    }
    changed
}

fn load_user_blocks(func: &Function) -> Option<Vec<Vec<Block>>> {
    let mut is_load = vec![false; func.types.len()];
    let mut any_load = false;
    for data in &func.blocks {
        for &inst in &data.insts {
            let inst_data = &func.insts[inst.index()];
            if matches!(inst_data.op, Op::Load(..))
                && let Some(res) = inst_data.result
            {
                is_load[res.index()] = true;
                any_load = true;
            }
        }
    }
    if !any_load {
        return None;
    }
    let mut users = vec![Vec::new(); func.types.len()];
    for (index, data) in func.blocks.iter().enumerate() {
        for inst in &data.insts {
            func.insts[inst.index()].op.for_operands(|v| {
                if is_load[v.index()] {
                    let list: &mut Vec<Block> = &mut users[v.index()];
                    if list.last() != Some(&Block(index as u32)) {
                        list.push(Block(index as u32));
                    }
                }
            });
        }
    }
    Some(users)
}

fn candidates(func: &Function, users: &[Vec<Block>], block: Block) -> Vec<Candidate> {
    let insts = &func.blocks[block.index()].insts;
    insts
        .iter()
        .enumerate()
        .filter_map(|(pos, inst)| {
            let data = &func.insts[inst.index()];
            let Op::Load(addr, offset) = data.op else {
                return None;
            };
            let result = data.result?;
            let blocks = users.get(result.index())?;
            (!blocks.is_empty() && !blocks.contains(&block)).then_some(Candidate {
                pos,
                addr,
                offset,
                result,
            })
        })
        .collect()
}

fn sink_block(
    func: &mut Function,
    (cfg, rank): (&Cfg, &[usize]),
    prov: &Provenance,
    users: &[Vec<Block>],
    block: Block,
) -> bool {
    let found = candidates(func, users, block);
    let mut addrs: Vec<Value> = found.iter().map(|c| c.addr).collect();
    addrs.sort_unstable();
    addrs.dedup();
    let mut moved: Vec<Inst> = Vec::new();
    for addr in addrs {
        let group: Vec<Candidate> = found.iter().copied().filter(|c| c.addr == addr).collect();
        let Some(loc) = span(func, prov, &group) else {
            continue;
        };
        let first = group.iter().map(|c| c.pos).min().unwrap_or(0);
        let insts = &func.blocks[block.index()].insts;
        if insts[first + 1..]
            .iter()
            .any(|inst| clobbers(func, &func.insts[inst.index()].op, loc, prov))
        {
            continue;
        }
        let region = clean_region(func, cfg, prov, block, loc);
        let depth = cfg.loops().depth_of(block);
        let wide = group.len() >= WIDE_GROUP;
        for load in group {
            let targets = &users[load.result.index()];
            let fits = targets.iter().all(|&target| {
                region[target.index()].is_some_and(|first_write| {
                    cfg.loops().depth_of(target) <= depth
                        && first_use(func, target, load.result).is_some_and(|pos| pos < first_write)
                })
            }) && (wide
                || crosses_call(
                    func,
                    cfg,
                    (block, load.pos),
                    &region,
                    (targets, load.result),
                ));
            if fits {
                moved.push(func.blocks[block.index()].insts[load.pos]);
                let mut ordered = targets.clone();
                ordered.sort_unstable_by_key(|target| rank[target.index()]);
                let mut clones: Vec<(Block, Value)> = Vec::new();
                for target in ordered {
                    let shared = clones
                        .iter()
                        .find(|&&(at, _)| {
                            region[at.index()] == Some(usize::MAX) && cfg.dominates(at, target)
                        })
                        .map(|&(_, value)| value);
                    let value = clone_into(func, target, load, shared);
                    clones.push((target, value));
                }
            }
        }
    }
    if moved.is_empty() {
        return false;
    }
    func.blocks[block.index()]
        .insts
        .retain(|inst| !moved.contains(inst));
    true
}

fn span(func: &Function, prov: &Provenance, group: &[Candidate]) -> Option<MemLoc> {
    let lo = group.iter().map(|c| i64::from(c.offset)).min()?;
    let hi = group
        .iter()
        .map(|c| i64::from(c.offset) + i64::from(func.ty(c.result).bytes()))
        .max()?;
    Some(prov.loc(group[0].addr, lo, u32::try_from(hi - lo).ok()?))
}

/// For each block `from` dominates that every path from `from` reaches without passing an instruction that may write `loc`, the position of its first such instruction, or `usize::MAX`.
fn clean_region(
    func: &Function,
    cfg: &Cfg,
    prov: &Provenance,
    from: Block,
    loc: MemLoc,
) -> Vec<Option<usize>> {
    let mut region: Vec<Option<usize>> = vec![None; func.blocks.len()];
    let mut members: Vec<Block> = Vec::new();
    let mut inside = vec![false; func.blocks.len()];
    for &block in &cfg.rpo {
        if members.len() >= MAX_REGION {
            break;
        }
        let dominated = block != from
            && cfg.idom[block.index()].is_some_and(|idom| idom == from || inside[idom.index()]);
        if dominated {
            inside[block.index()] = true;
            members.push(block);
            region[block.index()] = Some(
                func.blocks[block.index()]
                    .insts
                    .iter()
                    .position(|inst| clobbers(func, &func.insts[inst.index()].op, loc, prov))
                    .unwrap_or(usize::MAX),
            );
        }
    }
    let mut changed = true;
    while changed {
        changed = false;
        for &block in &members {
            if region[block.index()].is_some()
                && !cfg
                    .preds_of(block)
                    .iter()
                    .all(|&pred| pred == from || region[pred.index()] == Some(usize::MAX))
            {
                region[block.index()] = None;
                changed = true;
            }
        }
    }
    region
}

fn is_call(func: &Function, inst: Inst) -> bool {
    matches!(
        func.insts[inst.index()].op,
        Op::Call { .. } | Op::TryCall { .. }
    )
}

/// Whether a path from the load at `at` to a use of `result` passes a call, so the loaded value would live across it.
fn crosses_call(
    func: &Function,
    cfg: &Cfg,
    (from, pos): (Block, usize),
    region: &[Option<usize>],
    (targets, result): (&[Block], Value),
) -> bool {
    if func.blocks[from.index()].insts[pos + 1..]
        .iter()
        .any(|&inst| is_call(func, inst))
    {
        return true;
    }
    let mut seen = vec![false; func.blocks.len()];
    let mut work: Vec<Block> = Vec::new();
    for &target in targets {
        let before = first_use(func, target, result).unwrap_or(0);
        if func.blocks[target.index()].insts[..before]
            .iter()
            .any(|&inst| is_call(func, inst))
        {
            return true;
        }
        work.extend(cfg.preds_of(target).iter().copied());
    }
    while let Some(block) = work.pop() {
        if block == from || seen[block.index()] || region[block.index()].is_none() {
            continue;
        }
        seen[block.index()] = true;
        if func.blocks[block.index()]
            .insts
            .iter()
            .any(|&inst| is_call(func, inst))
        {
            return true;
        }
        work.extend(cfg.preds_of(block).iter().copied());
    }
    false
}

fn first_use(func: &Function, block: Block, value: Value) -> Option<usize> {
    func.blocks[block.index()].insts.iter().position(|inst| {
        let mut uses = false;
        func.insts[inst.index()]
            .op
            .for_operands(|v| uses |= v == value);
        uses
    })
}

/// Points the uses of the load's result in `target` at `shared`, a copy loaded in a dominating block, or at a new copy loaded before the first use; returns the value they now read.
fn clone_into(func: &mut Function, target: Block, load: Candidate, shared: Option<Value>) -> Value {
    let pos = first_use(func, target, load.result).unwrap_or(0);
    let value = shared.unwrap_or_else(|| func.value(func.ty(load.result)));
    let insts = func.blocks[target.index()].insts.clone();
    for &user in &insts[pos..] {
        func.insts[user.index()]
            .op
            .map_operands(|v| if v == load.result { value } else { v });
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
    if prov.slot_unwind_target(slot) || prov.slot_is_escaped_at(slot, Some(inst)) {
        return true;
    }
    let op = &func.insts[inst.index()].op;
    match *op {
        Op::StackLoad(s, _) | Op::StackAddr(s, _) => s == slot,
        Op::Load(addr, _)
        | Op::VolatileLoad(addr, _)
        | Op::AtomicLoad(addr, _)
        | Op::AtomicRmw(_, _, addr, _)
        | Op::AtomicCas(_, addr, _, _) => prov.loc(addr, 0, 1).root == MemRoot::Slot(slot),
        Op::Move(_, src, _) | Op::Copy(_, src, _) => {
            prov.loc(src, 0, 1).root == MemRoot::Slot(slot)
        }
        _ => false,
    }
}

fn slot_liveness_batch(
    func: &Function,
    cfg: &Cfg,
    prov: &Provenance,
    batch: &[StackSlot],
) -> (Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>) {
    let n = func.blocks.len();
    let mut slot_bit = vec![0_u64; func.slots.len()];
    let mut always_esc = 0_u64;
    let mut frontiers = Vec::new();
    for (i, &s) in batch.iter().enumerate() {
        let bit = 1_u64 << i;
        slot_bit[s.index()] = bit;
        if prov.slot_unwind_target(s) {
            always_esc |= bit;
        } else if !prov.slot_addr_unescaped(s) {
            match prov.slot_frontier(s) {
                Some(f) => frontiers.push((bit, f)),
                None => always_esc |= bit,
            }
        }
    }
    let (mut reads, mut kill, mut writes) = (vec![0_u64; n], vec![0_u64; n], vec![0_u64; n]);
    let mut local_esc: Vec<(u32, u64)> = Vec::new();
    let mut block_rem: Vec<(u64, &crate::mem::escape::Frontier)> = Vec::new();
    for &b in &cfg.rpo {
        let bi = b.index();
        let insts = &func.blocks[bi].insts;
        if insts.is_empty() {
            continue;
        }
        let mut r_mask = always_esc;
        let (mut k_mask, mut w_mask) = (0_u64, 0_u64);
        local_esc.clear();
        block_rem.clear();
        let mut rem_mask = 0_u64;
        for &(bit, f) in &frontiers {
            if f.block_seen(b) {
                r_mask |= bit;
            } else {
                block_rem.push((bit, f));
                rem_mask |= bit;
                if let Some(p) = f.first_point_in(b) {
                    local_esc.push((p, bit));
                }
            }
        }
        if local_esc.len() > 1 {
            local_esc.sort_unstable_by_key(|&(p, _)| p);
        }
        let mut esc_idx = 0usize;
        for (pos, &inst) in insts.iter().enumerate() {
            if rem_mask & !(r_mask | k_mask) != 0 {
                match prov.inst_loc(inst) {
                    None => {
                        r_mask |= rem_mask & !k_mask;
                        rem_mask = 0;
                    }
                    Some((eb, epos)) if eb == b.0 && epos == pos as u32 => {
                        while esc_idx < local_esc.len() && local_esc[esc_idx].0 <= epos {
                            r_mask |= local_esc[esc_idx].1 & !k_mask;
                            esc_idx += 1;
                        }
                    }
                    Some(loc) => {
                        for &(bit, f) in &block_rem {
                            if (r_mask | k_mask) & bit == 0 && f.reached(loc) {
                                r_mask |= bit;
                            }
                        }
                    }
                }
            }
            let op = &func.insts[inst.index()].op;
            match *op {
                Op::StackLoad(s, _) | Op::StackAddr(s, _) => {
                    let bit = slot_bit[s.index()];
                    if bit != 0 && k_mask & bit == 0 {
                        r_mask |= bit;
                    }
                }
                Op::StackStore(_, s, _) => {
                    let bit = slot_bit[s.index()];
                    if bit != 0 {
                        w_mask |= bit;
                        if r_mask & bit == 0 && is_full_slot_store(func, op, s) {
                            k_mask |= bit;
                        }
                    }
                }
                Op::Load(addr, _)
                | Op::VolatileLoad(addr, _)
                | Op::AtomicLoad(addr, _)
                | Op::AtomicRmw(_, _, addr, _)
                | Op::AtomicCas(_, addr, ..) => {
                    if let MemRoot::Slot(s) = prov.loc(addr, 0, 1).root {
                        let bit = slot_bit[s.index()];
                        if bit != 0 && k_mask & bit == 0 {
                            r_mask |= bit;
                        }
                    }
                }
                Op::Copy(_, src, _) | Op::Move(_, src, _) => {
                    if let MemRoot::Slot(s) = prov.loc(src, 0, 1).root {
                        let bit = slot_bit[s.index()];
                        if bit != 0 && k_mask & bit == 0 {
                            r_mask |= bit;
                        }
                    }
                }
                _ => {}
            }
        }
        reads[bi] = r_mask;
        kill[bi] = k_mask;
        writes[bi] = w_mask;
    }
    let (mut live_in, mut live_out) = (reads.clone(), vec![0_u64; n]);
    let mut changed = true;
    while changed {
        changed = false;
        for &b in cfg.rpo.iter().rev() {
            let bi = b.index();
            let mut out = 0_u64;
            for s in func.successors(b) {
                out |= live_in[s.index()];
            }
            let inp = reads[bi] | (out & !kill[bi]);
            if out != live_out[bi] || inp != live_in[bi] {
                live_out[bi] = out;
                live_in[bi] = inp;
                changed = true;
            }
        }
    }
    (live_in, live_out, reads, writes)
}

fn sink_pure_defs_into(func: &mut Function, from: Block, to: Block) {
    let mut cand_mask = 0_u128;
    let mut any_cand = false;
    for &inst in &func.blocks[from.index()].insts {
        let data = &func.insts[inst.index()];
        if let Some(res) = data.result
            && data.op.is_speculable()
            && !matches!(data.op, Op::Const(_))
        {
            cand_mask |= 1_u128 << (res.0 & 127);
            any_cand = true;
        }
    }
    if !any_cand {
        return;
    }
    let mut used_in_to_any = false;
    for &inst in &func.blocks[to.index()].insts {
        func.insts[inst.index()].op.for_operands(|v| {
            if cand_mask & (1_u128 << (v.0 & 127)) != 0 {
                used_in_to_any = true;
            }
        });
        if used_in_to_any {
            break;
        }
    }
    if !used_in_to_any {
        return;
    }
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
        let used_in_to = func.blocks[to.index()]
            .insts
            .iter()
            .chain(to_move.iter())
            .any(|&u| {
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
    func.blocks[from.index()]
        .insts
        .retain(|i| !to_move.contains(i));
    func.blocks[to.index()].insts.splice(0..0, to_move);
}

fn plan_store_frontier(
    func: &Function,
    cfg: &Cfg,
    (live_in, reads, writes): (&[u64], &[u64], &[u64]),
    (b, bit): (Block, u64),
) -> Option<Vec<Block>> {
    let (mut has_fast, mut has_clean_dom) = (false, false);
    for s in func.successors(b) {
        let si = s.index();
        if live_in[si] & bit == 0 {
            has_fast = true;
        } else if cfg.reachable[si] && cfg.dominates(b, s) && (reads[si] | writes[si]) & bit == 0 {
            has_clean_dom = true;
        }
    }
    if !has_fast && !has_clean_dom {
        return None;
    }
    let n = func.blocks.len();
    let (mut in_r, mut work, mut saw_fast) = (vec![false; n], vec![b], has_fast);
    in_r[b.index()] = true;
    while let Some(u) = work.pop() {
        for s in func.successors(u) {
            let si = s.index();
            if live_in[si] & bit == 0 {
                saw_fast = true;
            } else if !in_r[si]
                && cfg.reachable[si]
                && cfg.dominates(b, s)
                && (reads[si] | writes[si]) & bit == 0
            {
                in_r[si] = true;
                work.push(s);
            }
        }
    }
    if !saw_fast {
        return None;
    }
    let live: Vec<bool> = live_in.iter().map(|m| m & bit != 0).collect();
    crate::drop_elide::sink_frontier(func, cfg, &cfg.loops().depths, (&live, None, b), &mut in_r)
}

fn sink_stack_stores(func: &mut Function, cfg: &Cfg, prov: &Provenance) -> bool {
    let mut cand_slots: Vec<StackSlot> = Vec::new();
    let mut slot_pos: Vec<u32> = vec![u32::MAX; func.slots.len()];
    let mut store_blocks: Vec<Vec<Block>> = Vec::new();
    for &b in &cfg.rpo {
        for &inst in &func.blocks[b.index()].insts {
            if let Op::StackStore(_, slot, _) = func.insts[inst.index()].op
                && !func.is_unwind_slot(slot)
                && !prov.slot_unwind_target(slot)
                && !prov.slot_is_escaped_at(slot, Some(inst))
            {
                let idx = if slot_pos[slot.index()] == u32::MAX {
                    let next = cand_slots.len();
                    slot_pos[slot.index()] = next as u32;
                    cand_slots.push(slot);
                    store_blocks.push(vec![b]);
                    next
                } else {
                    let cur = slot_pos[slot.index()] as usize;
                    if store_blocks[cur].last() != Some(&b) {
                        store_blocks[cur].push(b);
                    }
                    cur
                };
                let _ = idx;
            }
        }
    }
    if cand_slots.is_empty() {
        return false;
    }
    let mut changed = false;
    for (chunk_idx, batch) in cand_slots.chunks(64).enumerate() {
        let (live_in, live_out, reads, writes) = slot_liveness_batch(func, cfg, prov, batch);
        for (bit_i, &slot) in batch.iter().enumerate() {
            let bit = 1_u64 << bit_i;
            let blocks = &store_blocks[chunk_idx * 64 + bit_i];
            for &b in blocks {
                let bi = b.index();
                let mut dead = Vec::new();
                let mut sink_targets: Option<(Vec<Block>, Vec<Inst>)> = None;
                let brif_arms = match func.last(b) {
                    Some(&Op::Brif(_, then, _, other, _)) if then != other => Some((then, other)),
                    _ => None,
                };
                let mut live_after = live_out[bi] & bit != 0;
                let mut any_access_after = false;
                for &inst in func.blocks[bi].insts.iter().rev() {
                    let op = &func.insts[inst.index()].op;
                    if let Op::StackStore(_, s, _) = *op
                        && s == slot
                        && !prov.slot_is_escaped_at(slot, Some(inst))
                    {
                        if !live_after {
                            dead.push(inst);
                            continue;
                        }
                        if !any_access_after {
                            if let Some((_, list)) = sink_targets.as_mut() {
                                list.push(inst);
                                if is_full_slot_store(func, op, slot) {
                                    live_after = false;
                                }
                                continue;
                            }
                            let targets = if let Some((then, other)) = brif_arms
                                && (live_in[then.index()] & bit != 0)
                                    != (live_in[other.index()] & bit != 0)
                            {
                                let target = if live_in[then.index()] & bit != 0 {
                                    then
                                } else {
                                    other
                                };
                                (cfg.single_pred(target) == Some(b)
                                    && cfg.loops().depth_of(target) <= cfg.loops().depth_of(b))
                                .then(|| vec![target])
                            } else {
                                plan_store_frontier(
                                    func,
                                    cfg,
                                    (&live_in, &reads, &writes),
                                    (b, bit),
                                )
                            };
                            if let Some(targets) = targets {
                                sink_targets = Some((targets, vec![inst]));
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
                    } else if inst_reads_slot(func, inst, slot, prov) {
                        live_after = true;
                        any_access_after = true;
                    } else if matches!(*op, Op::StackStore(_, s, _) if s == slot) {
                        any_access_after = true;
                    }
                }
                if !dead.is_empty() {
                    for &inst in &dead {
                        func.insts[inst.index()].op = Op::Trap;
                    }
                    func.blocks[bi].insts.retain(|i| !dead.contains(i));
                    changed = true;
                }
                if let Some((targets, mut sunk)) = sink_targets {
                    sunk.reverse();
                    func.blocks[bi].insts.retain(|i| !sunk.contains(i));
                    if targets.len() == 1 {
                        let target = targets[0];
                        func.blocks[target.index()]
                            .insts
                            .splice(0..0, sunk.iter().copied());
                        sink_pure_defs_into(func, b, target);
                    } else {
                        let ops: Vec<Op> = sunk
                            .iter()
                            .map(|&i| func.insts[i.index()].op.clone())
                            .collect();
                        for &i in &sunk {
                            func.insts[i.index()].op = Op::Trap;
                        }
                        for target in targets {
                            let clones: Vec<Inst> =
                                ops.iter().map(|op| func.add(op.clone(), None)).collect();
                            func.blocks[target.index()].insts.splice(0..0, clones);
                        }
                    }
                    changed = true;
                }
            }
        }
    }
    changed
}

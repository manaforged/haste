use crate::cfg::Cfg;
use crate::ir::{ArgumentPurpose, Binary, Block, Convert, Function, Inst, Op, StackSlot, Value};
use crate::mem::{MemLoc, MemRoot, Provenance, clobbers};

/// A place in the function: a block and a position in it.
type Site = (Block, usize);

/// Removes each move of a whole stack slot into another of the same size when the source is never touched after the move and the move dominates every touch of the target, by giving both one slot.
pub(super) fn elide_moves(func: &mut Function, prov: &Provenance) -> bool {
    let cfg = func.cfg().clone();
    let slots = func.slots.len();
    let mut refs: Vec<Vec<Site>> = vec![Vec::new(); slots];
    let mut moves: Vec<(Site, StackSlot, StackSlot)> = Vec::new();
    for &block in &cfg.rpo {
        for (pos, &inst) in func.blocks[block.index()].insts.iter().enumerate() {
            let touched = touched_slots(func, prov, inst);
            if let Some((to, from)) = whole_move(func, prov, inst) {
                moves.push(((block, pos), to, from));
            }
            for slot in touched {
                refs[slot.index()].push((block, pos));
            }
        }
    }
    let mut owner: Vec<StackSlot> = (0..slots).map(|slot| StackSlot(slot as u32)).collect();
    let mut sealed = sealed_slots(func, prov);
    let mut dead: Vec<Site> = undefined_moves(func, &cfg, prov, &sealed);
    for (site, to, from) in moves {
        let (to, from) = (find(&owner, to), find(&owner, from));
        if dead.contains(&site) || to == from || !sealed[from.index()] || !elidable(func, &cfg, site, &refs[from.index()], &refs[to.index()]) {
            continue;
        }
        owner[to.index()] = from;
        sealed[from.index()] &= sealed[to.index()];
        let moved = std::mem::take(&mut refs[to.index()]);
        refs[from.index()].extend(moved);
        refs[from.index()].retain(|&other| other != site);
        let align = func.slots[to.index()].align_shift.max(func.slots[from.index()].align_shift);
        func.slots[from.index()].align_shift = align;
        dead.push(site);
    }
    if dead.is_empty() {
        return false;
    }
    for data in &mut func.insts {
        match &mut data.op {
            Op::StackAddr(slot, _) | Op::StackLoad(slot, _) | Op::StackStore(_, slot, _) => *slot = find(&owner, *slot),
            _ => {}
        }
    }
    let gone: Vec<Inst> = dead.iter().map(|&(block, pos)| func.blocks[block.index()].insts[pos]).collect();
    for block in &mut func.blocks {
        block.insts.retain(|inst| !gone.contains(inst));
    }
    true
}

/// The entry-block moves whose sealed source slot nothing touched before them, so they copy undefined bytes.
fn undefined_moves(func: &Function, cfg: &Cfg, prov: &Provenance, sealed: &[bool]) -> Vec<Site> {
    let Some(entry) = cfg.entry.filter(|&entry| cfg.preds_of(entry).is_empty()) else { return Vec::new() };
    let mut touched = vec![false; func.slots.len()];
    let mut out = Vec::new();
    for (pos, &inst) in func.blocks[entry.index()].insts.iter().enumerate() {
        if let Op::Move(_, from, bytes) = func.insts[inst.index()].op
            && let MemRoot::Slot(slot) = prov.loc(from, 0, bytes).root
            && sealed[slot.index()]
            && !touched[slot.index()]
        {
            out.push((entry, pos));
        }
        for slot in touched_slots(func, prov, inst) {
            touched[slot.index()] = true;
        }
    }
    out
}

fn find(owner: &[StackSlot], mut slot: StackSlot) -> StackSlot {
    while owner[slot.index()] != slot {
        slot = owner[slot.index()];
    }
    slot
}

fn slot_of(prov: &Provenance, addr: Value) -> Option<StackSlot> {
    match prov.loc(addr, 0, 1).root {
        MemRoot::Slot(slot) => Some(slot),
        _ => None,
    }
}

fn touched_slots(func: &Function, prov: &Provenance, inst: Inst) -> Vec<StackSlot> {
    let op = &func.insts[inst.index()].op;
    let mut out = Vec::new();
    if let Op::StackLoad(slot, _) | Op::StackStore(_, slot, _) = op {
        out.push(*slot);
    }
    op.for_operands(|value| out.extend(slot_of(prov, value)));
    out.sort_unstable();
    out.dedup();
    out
}

fn whole_move(func: &Function, prov: &Provenance, inst: Inst) -> Option<(StackSlot, StackSlot)> {
    let Op::Move(to, from, bytes) = func.insts[inst.index()].op else { return None };
    let (target, source) = (prov.loc(to, 0, bytes), prov.loc(from, 0, bytes));
    let (MemRoot::Slot(to), MemRoot::Slot(from)) = (target.root, source.root) else { return None };
    let whole = |slot: StackSlot| func.slots[slot.index()].size == bytes;
    (target.offset == 0 && source.offset == 0 && whole(to) && whole(from)).then_some((to, from))
}

/// Whether no touch of the source follows the move on any path, the move itself included, and the move comes before every touch of the target.
fn elidable(func: &Function, cfg: &Cfg, (block, pos): Site, from_refs: &[Site], to_refs: &[Site]) -> bool {
    let after = reachable_after(func, block);
    let source_later = from_refs.iter().any(|&(at, p)| (at, p) != (block, pos) && ((at == block && p > pos) || after[at.index()]));
    if source_later || after[block.index()] {
        return false;
    }
    to_refs.iter().all(|&(at, p)| (at, p) == (block, pos) || if at == block { p > pos } else { cfg.dominates(block, at) })
}

fn reachable_after(func: &Function, block: Block) -> Vec<bool> {
    let mut seen = vec![false; func.blocks.len()];
    let mut work: Vec<Block> = func.successors(block).collect();
    while let Some(next) = work.pop() {
        if !std::mem::replace(&mut seen[next.index()], true) {
            work.extend(func.successors(next));
        }
    }
    seen
}

/// For each slot, whether every use of its address is a load, a store to it, a move, a compare, an offset that stays in it, or the result memory of a call, so no instruction outside those reaches it.
fn sealed_slots(func: &Function, prov: &Provenance) -> Vec<bool> {
    let mut sealed = vec![true; func.slots.len()];
    for block in &func.blocks {
        for &inst in &block.insts {
            let data = &func.insts[inst.index()];
            let stays = |value: Value| data.result.is_some_and(|result| slot_of(prov, result) == slot_of(prov, value));
            let allowed = |value: Value| match &data.op {
                Op::Load(..) | Op::Move(..) | Op::Copy(..) | Op::Icmp(..) => true,
                Op::Store(stored, ..) => *stored != value,
                Op::Binary(Binary::Iadd | Binary::Isub, ..) | Op::Convert(Convert::Bitcast, _) => stays(value),
                Op::Call { sig, callee, args, .. } | Op::TryCall { sig, callee, args, .. } => {
                    *callee != value && args.iter().zip(&func.sigs[sig.index()].params).all(|(&arg, param)| arg != value || param.purpose == ArgumentPurpose::StructReturn)
                }
                _ => false,
            };
            data.op.for_operands(|value| {
                if let Some(slot) = slot_of(prov, value)
                    && !allowed(value)
                {
                    sealed[slot.index()] = false;
                }
            });
        }
    }
    sealed
}

/// Points each load of a move's target that follows the move in its block, before anything may write either range, at the same bytes of the source.
fn forward_moved_loads(func: &mut Function, prov: &Provenance) -> bool {
    let mut changed = false;
    for block in 0..func.blocks.len() {
        let insts = func.blocks[block].insts.clone();
        for (pos, &inst) in insts.iter().enumerate() {
            let Op::Move(to, from, bytes) = func.insts[inst.index()].op else { continue };
            let (target, source) = (prov.loc(to, 0, bytes), prov.loc(from, 0, bytes));
            if !matches!(target.root, MemRoot::Slot(_)) {
                continue;
            }
            for &later in &insts[pos + 1..] {
                let data = &func.insts[later.index()];
                let read = match (&data.op, data.result) {
                    (&Op::Load(addr, offset), Some(result)) => Some(prov.loc(addr, i64::from(offset), func.ty(result).bytes())),
                    (&Op::StackLoad(slot, offset), Some(result)) => Some(MemLoc::slot(slot, i64::from(offset), func.ty(result).bytes())),
                    _ => None,
                };
                if let Some(read) = read {
                    let delta = read.offset - target.offset;
                    if read.base == target.base && delta >= 0 && delta + i64::from(read.size) <= i64::from(bytes)
                        && let Ok(delta) = i32::try_from(delta)
                    {
                        func.insts[later.index()].op = Op::Load(from, delta);
                        changed = true;
                    }
                    continue;
                }
                if clobbers(func, &data.op, target, prov) || clobbers(func, &data.op, source, prov) {
                    break;
                }
            }
        }
    }
    changed
}

/// Deletes each move into a slot that nothing reads, whose address only ever names the memory a store or move writes.
fn drop_unread_moves(func: &mut Function, prov: &Provenance) -> bool {
    let mut read = vec![false; func.slots.len()];
    for block in &func.blocks {
        for &inst in &block.insts {
            let op = &func.insts[inst.index()].op;
            if let Op::StackLoad(slot, _) = op {
                read[slot.index()] = true;
            }
            op.for_operands(|value| {
                let writes = match *op {
                    Op::Store(stored, addr, _) => addr == value && stored != value,
                    Op::Move(to, from, _) => to == value && from != value,
                    _ => false,
                };
                if let Some(slot) = slot_of(prov, value)
                    && !writes
                {
                    read[slot.index()] = true;
                }
            });
        }
    }
    let mut changed = false;
    for index in 0..func.blocks.len() {
        let before = func.blocks[index].insts.len();
        let insts = std::mem::take(&mut func.blocks[index].insts);
        func.blocks[index].insts = insts.into_iter().filter(|inst| !matches!(func.insts[inst.index()].op, Op::Move(to, ..) if slot_of(prov, to).is_some_and(|slot| !read[slot.index()]))).collect();
        changed |= func.blocks[index].insts.len() != before;
    }
    changed
}

/// Reads moved bytes from the source where the target is read next to the move, then deletes the moves no read needs.
pub(super) fn forward_moves(func: &mut Function, prov: &Provenance) -> bool {
    let forwarded = forward_moved_loads(func, prov);
    forwarded | drop_unread_moves(func, prov)
}

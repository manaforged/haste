use crate::cfg::Cfg;
use crate::ir::{ArgumentPurpose, Binary, Block, Convert, Function, Inst, Op, StackSlot, Value};
use crate::mem::{MemRoot, Provenance};

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
    let mut dead: Vec<Site> = Vec::new();
    for (site, to, from) in moves {
        let (to, from) = (find(&owner, to), find(&owner, from));
        if to == from || !sealed[from.index()] || !elidable(func, &cfg, site, &refs[from.index()], &refs[to.index()]) {
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

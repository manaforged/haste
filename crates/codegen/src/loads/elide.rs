use crate::cfg::Cfg;
use crate::ir::{
    ArgumentPurpose, AtomicOrd, AtomicRmwOp, Binary, Block, Convert, Function, Inst, Op, StackSlot,
    Type, Value,
};
use crate::mem::{KnownLoads, MemBase, MemLoc, MemRoot, Provenance, clobbers, may_alias};
use crate::merge::find as find_val;

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
        if dead.contains(&site)
            || to == from
            || !sealed[from.index()]
            || func.is_unwind_slot(to)
            || func.is_unwind_slot(from)
            || !elidable(func, &cfg, site, &refs[from.index()], &refs[to.index()])
        {
            continue;
        }
        owner[to.index()] = from;
        sealed[from.index()] &= sealed[to.index()];
        let moved = std::mem::take(&mut refs[to.index()]);
        refs[from.index()].extend(moved);
        refs[from.index()].retain(|&other| other != site);
        let align = func.slots[to.index()]
            .align_shift
            .max(func.slots[from.index()].align_shift);
        func.slots[from.index()].align_shift = align;
        let poll_local = func.slots[to.index()].poll_local;
        func.slots[from.index()].poll_local &= poll_local;
        dead.push(site);
    }
    if dead.is_empty() {
        return false;
    }
    for data in &mut func.insts {
        match &mut data.op {
            Op::StackAddr(slot, _) | Op::StackLoad(slot, _) | Op::StackStore(_, slot, _) => {
                *slot = find(&owner, *slot)
            }
            _ => {}
        }
    }
    let gone: Vec<Inst> = dead
        .iter()
        .map(|&(block, pos)| func.blocks[block.index()].insts[pos])
        .collect();
    for block in &mut func.blocks {
        block.insts.retain(|inst| !gone.contains(inst));
    }
    true
}

/// The entry-block moves whose sealed source slot nothing touched before them, so they copy undefined bytes.
fn undefined_moves(func: &Function, cfg: &Cfg, prov: &Provenance, sealed: &[bool]) -> Vec<Site> {
    let Some(entry) = cfg.entry.filter(|&entry| cfg.preds_of(entry).is_empty()) else {
        return Vec::new();
    };
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
    let Op::Move(to, from, bytes) = func.insts[inst.index()].op else {
        return None;
    };
    let (target, source) = (prov.loc(to, 0, bytes), prov.loc(from, 0, bytes));
    let (MemRoot::Slot(to), MemRoot::Slot(from)) = (target.root, source.root) else {
        return None;
    };
    let whole = |slot: StackSlot| func.slots[slot.index()].size == bytes;
    (target.offset == 0 && source.offset == 0 && whole(to) && whole(from)).then_some((to, from))
}

/// Whether no touch of the source follows the move on any path, the move itself included, and the move comes before every touch of the target.
fn elidable(
    func: &Function,
    cfg: &Cfg,
    (block, pos): Site,
    from_refs: &[Site],
    to_refs: &[Site],
) -> bool {
    let after = reachable_after(func, block);
    let source_later = from_refs
        .iter()
        .any(|&(at, p)| (at, p) != (block, pos) && ((at == block && p > pos) || after[at.index()]));
    if source_later || after[block.index()] {
        return false;
    }
    to_refs.iter().all(|&(at, p)| {
        (at, p) == (block, pos)
            || if at == block {
                p > pos
            } else {
                cfg.dominates(block, at)
            }
    })
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
            let stays = |value: Value| {
                data.result
                    .is_some_and(|result| slot_of(prov, result) == slot_of(prov, value))
            };
            let allowed =
                |value: Value| match &data.op {
                    Op::Load(..) | Op::Move(..) | Op::Copy(..) | Op::Icmp(..) => true,
                    Op::Store(stored, ..) => *stored != value,
                    Op::Binary(Binary::Iadd | Binary::Isub, ..)
                    | Op::Convert(Convert::Bitcast, _) => stays(value),
                    Op::Call {
                        sig, callee, args, ..
                    }
                    | Op::TryCall {
                        sig, callee, args, ..
                    } => {
                        *callee != value
                            && args.iter().zip(&func.sigs[sig.index()].params).all(
                                |(&arg, param)| {
                                    arg != value || param.purpose == ArgumentPurpose::StructReturn
                                },
                            )
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

fn try_forward_move_chain(
    func: &mut Function,
    later: Inst,
    (target, source, from, bytes): (MemLoc, MemLoc, Value, u32),
    prov: &Provenance,
) -> bool {
    let Op::Move(dst2, src2, bytes2) = func.insts[later.index()].op else {
        return false;
    };
    if bytes2 > bytes || src2 == from {
        return false;
    }
    let (read_loc, dst2_loc) = (prov.loc(src2, 0, bytes2), prov.loc(dst2, 0, bytes2));
    if read_loc.base != target.base
        || read_loc.offset != target.offset
        || may_alias(dst2_loc, source, prov)
        || may_alias(dst2_loc, target, prov)
    {
        return false;
    }
    func.insts[later.index()].op = Op::Move(dst2, from, bytes2);
    true
}

fn read_loc_at(func: &Function, data: &crate::ir::InstData, prov: &Provenance) -> Option<MemLoc> {
    let result = data.result?;
    match data.op {
        Op::Load(addr, offset) => Some(prov.loc(addr, i64::from(offset), func.ty(result).bytes())),
        Op::StackLoad(slot, offset) => Some(MemLoc::slot(
            slot,
            i64::from(offset),
            func.ty(result).bytes(),
        )),
        _ => None,
    }
}

/// Points each load or chained move of a move's target that follows the move in its block, before anything may write either range, at the same bytes of the source.
fn forward_moved_loads(func: &mut Function, prov: &Provenance) -> bool {
    let mut changed = false;
    for block in 0..func.blocks.len() {
        let insts = func.blocks[block].insts.clone();
        for (pos, &inst) in insts.iter().enumerate() {
            let Op::Move(to, from, bytes) = func.insts[inst.index()].op else {
                continue;
            };
            let (target, source) = (prov.loc(to, 0, bytes), prov.loc(from, 0, bytes));
            if !matches!(target.root, MemRoot::Slot(_)) {
                continue;
            }
            for &later in &insts[pos + 1..] {
                if try_forward_move_chain(func, later, (target, source, from, bytes), prov) {
                    changed = true;
                    continue;
                }
                let data = &func.insts[later.index()];
                if let Some(read) = read_loc_at(func, data, prov) {
                    let delta = read.offset - target.offset;
                    if read.base == target.base
                        && delta >= 0
                        && delta + i64::from(read.size) <= i64::from(bytes)
                        && let Ok(delta) = i32::try_from(delta)
                    {
                        func.insts[later.index()].op = Op::Load(from, delta);
                        changed = true;
                    }
                    continue;
                }
                if clobbers(func, &data.op, target, prov) || clobbers(func, &data.op, source, prov)
                {
                    break;
                }
            }
        }
    }
    changed
}

/// Deletes each move into a slot that nothing reads, whose address only ever names the memory a store or move writes.
fn drop_unread_moves(func: &mut Function, prov: &Provenance) -> bool {
    let mut read: Vec<bool> = (0..func.slots.len())
        .map(|s| func.is_unwind_slot(StackSlot(s as u32)))
        .collect();
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

fn op_keeps_alloc_private(op: &Op, addr: Value, subst: &[Value]) -> bool {
    let mut used = false;
    op.for_operands(|v| used |= find_val(subst, v) == addr);
    if !used {
        return true;
    }
    match *op {
        Op::Load(a, _) => find_val(subst, a) == addr,
        Op::Store(v, a, _) => find_val(subst, a) == addr && find_val(subst, v) != addr,
        Op::Move(dst, src, _) => find_val(subst, dst) == addr && find_val(subst, src) != addr,
        Op::Icmp(..) | Op::Trapz(_) => true,
        Op::Brif(_, _, ref ta, _, ref ea) => {
            !ta.iter().chain(ea).any(|&v| find_val(subst, v) == addr)
        }
        _ => false,
    }
}

fn scan_fresh_alloc(
    func: &Function,
    insts: impl Iterator<Item = Inst>,
    subst: &[Value],
    addr: Value,
) -> Option<bool> {
    for inst in insts {
        let op = &func.insts[inst.index()].op;
        if let Op::Call {
            callee, results, ..
        } = op
            && func.is_fresh_alloc(*callee)
            && results.first().is_some_and(|&r| find_val(subst, r) == addr)
        {
            return Some(true);
        }
        if !op_keeps_alloc_private(op, addr, subst) {
            return Some(false);
        }
    }
    None
}

fn is_unescaped_fresh_alloc(
    func: &Function,
    cfg: &Cfg,
    block: Block,
    kept: &[Option<Inst>],
    subst: &[Value],
    addr: Value,
) -> bool {
    if let Some(found) = scan_fresh_alloc(func, kept.iter().rev().flatten().copied(), subst, addr) {
        return found;
    }
    let mut cur = block;
    for _ in 0..4 {
        let Some(pred) = cfg.single_pred(cur) else {
            return false;
        };
        if pred == block {
            return false;
        }
        if let Some(found) = scan_fresh_alloc(
            func,
            func.blocks[pred.index()].insts.iter().rev().copied(),
            subst,
            addr,
        ) {
            return found;
        }
        cur = pred;
    }
    false
}

/// Folds a `Relaxed` `Add` RMW on a freshly allocated, unescaped heap pointer whose previous constant value is known into a plain constant store.
pub(super) fn try_fold_private_rmw(
    func: &mut Function,
    (cfg, prov, block, inst): (&Cfg, &Provenance, Block, Inst),
    (consts, subst): (&mut Vec<Option<i64>>, &mut Vec<Value>),
    (known, kept): (&mut KnownLoads, &mut Vec<Option<Inst>>),
) -> Option<(MemLoc, Type, Value)> {
    let (Op::AtomicRmw(AtomicRmwOp::Add, AtomicOrd::Relaxed, raw_addr, raw_delta), Some(res)) = (
        func.insts[inst.index()].op.clone(),
        func.insts[inst.index()].result,
    ) else {
        return None;
    };
    let addr = find_val(subst, raw_addr);
    let delta = find_val(subst, raw_delta);
    let ty = func.ty(res);
    if !ty.is_small_int() {
        return None;
    }
    let d = consts.get(delta.index()).copied().flatten()?;
    let loc = prov.loc(addr, 0, ty.bytes());
    let prev_val = known.get(loc.base, 0, ty)?;
    let prev = consts
        .get(find_val(subst, prev_val).index())
        .copied()
        .flatten()?;
    if !is_unescaped_fresh_alloc(func, cfg, block, kept, subst, addr) {
        return None;
    }
    let prev_ext = crate::fold::extend(prev, ty, false);
    let next_ext = crate::fold::extend(prev.wrapping_add(d), ty, false);
    func.insts[inst.index()].op = Op::Const(prev_ext);
    consts[res.index()] = Some(prev_ext);
    kept.push(Some(inst));
    let c_new = func.value(ty);
    subst.push(c_new);
    consts.push(Some(next_ext));
    kept.push(Some(func.add(Op::Const(next_ext), Some(c_new))));
    kept.push(Some(func.add(Op::Store(c_new, addr, 0), None)));
    known.write(loc, ty, c_new, prov);
    Some((loc, ty, c_new))
}

fn slot_op_sealed(
    func: &Function,
    prov: &Provenance,
    inst: Inst,
    other: Inst,
    slot: StackSlot,
) -> bool {
    if other == inst {
        return true;
    }
    let data = &func.insts[other.index()];
    let stays = |v: Value| {
        data.result
            .is_some_and(|r| slot_of(prov, r) == Some(slot) && slot_of(prov, v) == Some(slot))
    };
    let mut ok = true;
    data.op.for_operands(|v| {
        if slot_of(prov, v) != Some(slot) {
            return;
        }
        ok &= match &data.op {
            Op::Load(..) | Op::Icmp(..) => true,
            Op::Store(stored, ..) => *stored != v,
            Op::Binary(Binary::Iadd | Binary::Isub, ..) | Op::Convert(Convert::Bitcast, _) => {
                stays(v)
            }
            _ => false,
        };
    });
    ok
}

fn slot_writes_covered(
    func: &Function,
    cfg: &Cfg,
    prov: &Provenance,
    (block, inst): (Block, Inst),
    slot: StackSlot,
    entries: &[(i64, Type, Value)],
) -> bool {
    let pred = cfg.single_pred(block);
    let covered = |off: i64, size: u32| -> bool {
        let Some(end) = off.checked_add(i64::from(size)) else {
            return false;
        };
        (off..end).all(|b| {
            entries.iter().any(|&(eo, et, _)| {
                eo <= b
                    && eo
                        .checked_add(i64::from(et.bytes()))
                        .is_some_and(|ee| b < ee)
            })
        })
    };
    for (bi, bdata) in func.blocks.iter().enumerate() {
        let cur = Block(bi as u32);
        let in_cur = cur == block;
        let allowed_block = in_cur || pred == Some(cur);
        let mut past_inst = false;
        for &other in &bdata.insts {
            if !slot_op_sealed(func, prov, inst, other, slot) {
                return false;
            }
            if in_cur && other == inst {
                past_inst = true;
                continue;
            }
            let op = &func.insts[other.index()].op;
            let write = match *op {
                Op::StackStore(v, s, off) if s == slot => {
                    Some((i64::from(off), func.ty(v).bytes()))
                }
                Op::Store(v, a, off) => {
                    let l = prov.loc(a, i64::from(off), func.ty(v).bytes());
                    (l.root == MemRoot::Slot(slot)).then_some((l.offset, l.size))
                }
                _ => None,
            };
            if let Some((off, size)) = write
                && (!allowed_block || (in_cur && past_inst) || !covered(off, size))
            {
                return false;
            }
        }
    }
    true
}

/// Replaces a whole-slot `Move` into a fresh heap allocation with direct scalar stores for every initialized field of the slot, dropping copies of uninitialized padding.
pub(super) fn try_scalarize_slot_move(
    func: &mut Function,
    (cfg, prov, block, inst): (&Cfg, &Provenance, Block, Inst),
    (dst, src, bytes): (Value, Value, u32),
    subst: &[Value],
    known: &mut KnownLoads,
) -> Option<Vec<(MemLoc, Inst)>> {
    if bytes == 0 {
        return None;
    }
    let dst = find_val(subst, dst);
    let src = find_val(subst, src);
    let src_loc = prov.loc(src, 0, bytes);
    let dst_loc = prov.loc(dst, 0, bytes);
    let MemRoot::Slot(slot) = src_loc.root else {
        return None;
    };
    if src_loc.offset != 0
        || func.slots[slot.index()].size != bytes
        || !matches!(dst_loc.root, MemRoot::UniqueHeap(_))
        || may_alias(dst_loc, src_loc, prov)
    {
        return None;
    }
    let mut entries: Vec<(i64, Type, Value)> = known
        .entries_for(MemBase::Slot(slot))
        .iter()
        .copied()
        .filter(|&(off, ty, _)| {
            off >= 0
                && off
                    .checked_add(i64::from(ty.bytes()))
                    .is_some_and(|e| e <= i64::from(bytes))
        })
        .collect();
    let move_ops = 2 * crate::copy::vec_words(bytes).len();
    if entries.is_empty() || entries.len() > 8 || entries.len() >= move_ops {
        return None;
    }
    entries.sort_by_key(|&(off, ..)| off);
    if entries
        .iter()
        .any(|&(off, ty, _)| i32::try_from(off).is_err() || !super::fits_mem_imm(ty, off))
    {
        return None;
    }
    if !slot_writes_covered(func, cfg, prov, (block, inst), slot, &entries) {
        return None;
    }
    func.insts[inst.index()].op = Op::Trap;
    let mut emitted = Vec::with_capacity(entries.len());
    for (off, ty, raw_val) in entries {
        let val = find_val(subst, raw_val);
        let store = func.add(Op::Store(val, dst, off as i32), None);
        let wloc = prov.loc(dst, off, ty.bytes());
        known.write(wloc, ty, val, prov);
        emitted.push((wloc, store));
    }
    Some(emitted)
}

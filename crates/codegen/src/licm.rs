//! Moves out of each loop what every trip computes alike: pure instructions that cannot trap and whose operands the loop does not change, and loads at the head of a loop that writes no memory.

use crate::dom::{dominates, dominators};
use crate::ir::{Block, BlockData, Function, Inst, Op, StackSlot, Value};
use crate::loads::{classify_slots, clobbers, slot_addresses};

/// One natural loop: its head, the blocks of its body including the head, and the one block outside it that enters the head by a jump, when there is one.
struct Loop {
    head: Block,
    body: Vec<bool>,
    size: usize,
    entry: Option<Block>,
}

/// The natural loop that the back edges from `latches` into `head` close.
fn natural(func: &Function, preds: &[Vec<Block>], head: Block, latches: &[Block]) -> Loop {
    let mut body = vec![false; func.blocks.len()];
    body[head.index()] = true;
    let mut todo = latches.to_vec();
    while let Some(block) = todo.pop() {
        if !std::mem::replace(&mut body[block.index()], true) {
            todo.extend(preds[block.index()].iter().copied());
        }
    }
    let mut outside = preds[head.index()].iter().copied().filter(|pred| !body[pred.index()]);
    let first = outside.next();
    let single = first.filter(|entry| outside.all(|other| other == *entry) && matches!(func.last(*entry), Some(Op::Jump(target, _)) if *target == head));
    Loop { head, size: body.iter().filter(|inside| **inside).count(), body, entry: single }
}

/// The natural loops of `func`, innermost first.
fn loops(func: &Function, entry: Block) -> Vec<Loop> {
    let (order, preds, idom) = dominators(func, entry);
    let mut found: Vec<Loop> = order
        .iter()
        .filter_map(|&head| {
            let latches: Vec<Block> = preds[head.index()].iter().copied().filter(|pred| dominates(&idom, head, *pred)).collect();
            (!latches.is_empty()).then(|| natural(func, &preds, head, &latches))
        })
        .collect();
    found.sort_by_key(|each| each.size);
    found
}

/// The number of natural loops each block lies in.
pub(crate) fn depths(func: &Function) -> Vec<u32> {
    let mut depth = vec![0; func.blocks.len()];
    if let Some(entry) = func.entry {
        for each in loops(func, entry) {
            depth.iter_mut().zip(&each.body).filter(|(_, inside)| **inside).for_each(|(depth, _)| *depth += 1);
        }
    }
    depth
}

/// Synthesizes a preheader block for `each` when `each.head` is not the entry and every outside predecessor jumps or branches to it.
fn preheader(func: &mut Function, each: &Loop, home: &mut Vec<Option<Block>>) -> Option<Block> {
    if Some(each.head) == func.entry {
        return None;
    }
    let outside: Vec<Block> = (0..func.blocks.len())
        .filter(|&pred| !each.body.get(pred).copied().unwrap_or(false))
        .map(|pred| Block(pred as u32))
        .filter(|&pred| func.successors(pred).any(|target| target == each.head))
        .collect();
    if outside.is_empty() || !outside.iter().all(|&pred| matches!(func.last(pred), Some(Op::Jump(..) | Op::Brif(..)))) {
        return None;
    }
    let param_types: Vec<_> = func.blocks[each.head.index()].params.iter().map(|&p| func.ty(p)).collect();
    let pre_params: Vec<Value> = param_types.into_iter().map(|ty| func.value(ty)).collect();
    let jump = func.add(Op::Jump(each.head, pre_params.clone()), None);
    let pre = Block(func.blocks.len() as u32);
    func.blocks.push(BlockData { params: pre_params, insts: vec![jump] });
    home.resize(func.types.len(), Some(pre));
    for pred in outside {
        let Some(&last) = func.blocks[pred.index()].insts.last() else { continue };
        match &mut func.insts[last.index()].op {
            Op::Jump(target, _) if *target == each.head => *target = pre,
            Op::Brif(_, then, _, otherwise, _) => {
                if *then == each.head {
                    *then = pre;
                }
                if *otherwise == each.head {
                    *otherwise = pre;
                }
            }
            _ => {}
        }
    }
    Some(pre)
}

/// Moves into the loop's entry block, before its jump, each instruction of the loop that may move, in order, and records their new home.
fn hoist(
    func: &mut Function,
    each: &Loop,
    entry: Block,
    (readonly, ro_bytes): (&[bool], &[u32]),
    (slot_ro, slot_addrs, has_stack_addr): (&[bool], &[Option<(StackSlot, i32)>], &[bool]),
    home: &mut [Option<Block>],
) {
    let mut writes = false;
    let mut unknown_mem_writes = false;
    let mut calls = false;
    let mut slot_clobbered = vec![false; func.slots.len()];
    let mut slot_write_ranges: Vec<Vec<(i64, i64)>> = vec![Vec::new(); func.slots.len()];
    for inst in (0..each.body.len()).filter(|block| each.body[*block]).flat_map(|block| &func.blocks[block].insts) {
        let data = &func.insts[inst.index()];
        let op = &data.op;
        if clobbers(op) || matches!(op, Op::Store(..) | Op::StackStore(..)) {
            writes = true;
        }
        if matches!(op, Op::Call { .. } | Op::TryCall { .. }) {
            calls = true;
        }
        match *op {
            Op::StackStore(val, slot, off) => {
                if off < 0 {
                    slot_clobbered[slot.index()] = true;
                } else {
                    let lo = i64::from(off);
                    slot_write_ranges[slot.index()].push((lo, lo + i64::from(func.ty(val).bytes())));
                }
            }
            Op::Store(val, addr, off) => {
                if let Some((slot, base_off)) = slot_addrs.get(addr.index()).copied().flatten() {
                    let lo = i64::from(base_off) + i64::from(off);
                    if lo < 0 {
                        slot_clobbered[slot.index()] = true;
                    } else {
                        slot_write_ranges[slot.index()].push((lo, lo + i64::from(func.ty(val).bytes())));
                    }
                } else {
                    unknown_mem_writes = true;
                }
            }
            Op::Move(addr, _, _) | Op::Copy(addr, _, _) => {
                if let Some((slot, _)) = slot_addrs.get(addr.index()).copied().flatten() {
                    slot_clobbered[slot.index()] = true;
                } else {
                    unknown_mem_writes = true;
                }
            }
            _ if clobbers(op) => unknown_mem_writes = true,
            _ => {}
        }
    }
    let slot_written = |slot: StackSlot, lo: i64, hi: i64| {
        slot_clobbered[slot.index()] || slot_write_ranges[slot.index()].iter().any(|&(wlo, whi)| lo < whi && wlo < hi)
    };
    let slot_ok = |slot: StackSlot, lo: i64, hi: i64| {
        (slot_ro.get(slot.index()).copied().unwrap_or(false) || (!calls && !unknown_mem_writes)) && !slot_written(slot, lo, hi)
    };
    let mut moved: Vec<Inst> = Vec::new();
    for block in (0..each.body.len()).filter(|block| each.body[*block]) {
        let head = block == each.head.index();
        let insts = std::mem::take(&mut func.blocks[block].insts);
        let mut kept = Vec::with_capacity(insts.len());
        let mut head_prefix = head;
        for inst in insts {
            let data = &func.insts[inst.index()];
            if matches!(data.op, Op::Call { .. } | Op::TryCall { .. } | Op::Trap | Op::Trapz(_) | Op::Brif(..)) {
                head_prefix = false;
            }
            let static_loc = matches!(data.op, Op::StackAddr(..) | Op::SlotAddr(_));
            let load_ok = match (&data.op, data.result) {
                (&Op::StackLoad(slot, offset), Some(res)) => {
                    let width = u64::from(func.ty(res).bytes());
                    let lo = i64::from(offset);
                    !slot_written(slot, lo, lo + width as i64)
                        && (!has_stack_addr[slot.index()] || (!calls && !unknown_mem_writes))
                        && offset >= 0
                        && (offset as u64).checked_add(width).is_some_and(|end| end <= u64::from(func.slots[slot.index()].size))
                }
                (&Op::Load(addr, offset), Some(res)) => {
                    let width = func.ty(res).bytes();
                    let slot_load_ok = slot_addrs.get(addr.index()).copied().flatten().is_some_and(|(slot, base_off)| {
                        i64::from(base_off).checked_add(i64::from(offset)).is_some_and(|total| {
                            total >= 0
                                && (total as u64)
                                    .checked_add(u64::from(width))
                                    .is_some_and(|end| end <= u64::from(func.slots[slot.index()].size))
                                && slot_ok(slot, total, total + i64::from(width))
                        })
                    });
                    let deref_ok = offset >= 0 && (offset as u64) + u64::from(width) <= u64::from(ro_bytes.get(addr.index()).copied().unwrap_or(0));
                    slot_load_ok || (!calls && ((head && !writes) || (head_prefix && readonly.get(addr.index()).copied().unwrap_or(false)) || deref_ok))
                }
                _ => false,
            };
            let movable = (data.op.is_speculable() && !(calls && static_loc)) || load_ok;
            let invariant = |value: Value| home.get(value.index()).copied().flatten().is_none_or(|at| !each.body.get(at.index()).copied().unwrap_or(false));
            let mut operands_invariant = true;
            data.op.for_operands(|value| operands_invariant &= invariant(value));
            if movable && operands_invariant && data.result.is_some() {
                data.result.iter().for_each(|result| home[result.index()] = Some(entry));
                moved.push(inst);
            } else {
                kept.push(inst);
            }
        }
        func.blocks[block].insts = kept;
    }
    let insts = &mut func.blocks[entry.index()].insts;
    let at = insts.len().saturating_sub(1);
    insts.splice(at..at, moved);
}

/// Moves loop-invariant work of every natural loop that one jump enters out to the block that enters it.
pub(crate) fn run(func: &mut Function) {
    let Some(entry) = func.entry else { return };
    if func.coroutine.is_some() {
        return;
    }
    let mut found = loops(func, entry);
    if found.is_empty() {
        return;
    }
    let mut readonly = vec![false; func.types.len()];
    let mut ro_bytes = vec![0_u32; func.types.len()];
    for (i, &param) in func.blocks[entry.index()].params.iter().enumerate() {
        if i < 64 && (func.signature.readonly_params >> i) & 1 != 0 {
            readonly[param.index()] = true;
        }
        if i < 16 {
            ro_bytes[param.index()] = func.signature.readonly_bytes[i];
        }
    }
    let slot_addrs = slot_addresses(func);
    let (_, slot_ro) = classify_slots(func, &slot_addrs);
    let mut has_stack_addr = vec![false; func.slots.len()];
    for inst in func.blocks.iter().flat_map(|b| &b.insts) {
        if let Op::StackAddr(slot, _) = func.insts[inst.index()].op {
            has_stack_addr[slot.index()] = true;
        }
    }
    let mut home = func.def_blocks();
    for idx in 0..found.len() {
        let pre = found[idx].entry.or_else(|| {
            let created = preheader(func, &found[idx], &mut home)?;
            let head_idx = found[idx].head.index();
            for later in &mut found[idx + 1..] {
                let inside = later.body.get(head_idx).copied().unwrap_or(false);
                later.body.resize(created.index(), false);
                later.body.push(inside);
            }
            Some(created)
        });
        if let Some(entry) = pre {
            hoist(func, &found[idx], entry, (&readonly, &ro_bytes), (&slot_ro, &slot_addrs, &has_stack_addr), &mut home);
        }
    }
}

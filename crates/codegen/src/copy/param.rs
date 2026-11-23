use crate::ir::{Block, Function, Inst, Op, StackSlot, Value};

use super::{Access, accesses_at, collect_sret_info, resolve_slot, retarget_slot};

/// Reads each frame slot that only holds an entry copy of a read-only, unaliased parameter's memory from that memory instead, so each field loads where it is used.
pub(crate) fn forward_param_slots(func: &mut Function) -> bool {
    let Some(entry) = func.entry else { return false };
    let mut changed = false;
    while let Some((slot, param, copies)) = param_slot_plan(func, entry) {
        retarget_slot(func, slot, param, &copies);
        changed = true;
    }
    changed
}

fn has_entry_param_copy(func: &Function, entry: Block, params: &[(Value, u32)]) -> bool {
    func.blocks[entry.index()].insts.iter().any(|&inst| match func.insts[inst.index()].op {
        Op::Load(from, _) | Op::Move(_, from, _) => params.iter().any(|&(p, _)| p == from),
        _ => false,
    })
}

/// A slot that nothing writes but stores in the entry block that copy each byte from the same offset of one read-only, unaliased parameter, with that parameter and those stores.
fn param_slot_plan(func: &Function, entry: Block) -> Option<(StackSlot, Value, Vec<(Block, Inst)>)> {
    let params: Vec<(Value, u32)> = func.blocks[entry.index()].params.iter().zip(&func.signature.params).filter(|(_, abi)| abi.noalias && abi.readonly && abi.deref_bytes > 0).map(|(&value, abi)| (value, abi.deref_bytes)).collect();
    if params.is_empty() || !has_entry_param_copy(func, entry, &params) {
        return None;
    }
    let info = collect_sret_info(func, entry);
    let copies: Vec<(StackSlot, Value, Inst)> = func.blocks[entry.index()].insts.iter().filter_map(|&inst| param_copy(func, &info.defs, &params, inst).map(|(slot, param)| (slot, param, inst))).collect();
    let mut tried = Vec::new();
    for &(slot, param, _) in &copies {
        if tried.contains(&slot) {
            continue;
        }
        tried.push(slot);
        let index = slot.index() as u32;
        let bytes = params.iter().find(|&&(value, _)| value == param).map_or(0, |&(_, bytes)| bytes);
        if func.slots[slot.index()].size > bytes || func.self_slots.contains(&index) || func.vtable_slots.iter().any(|&(at, ..)| at == index) || copies.iter().any(|copy| copy.0 == slot && copy.1 != param) {
            continue;
        }
        let stores: Vec<(Block, Inst)> = copies.iter().filter(|copy| copy.0 == slot).map(|copy| (entry, copy.2)).collect();
        if !func.is_unwind_slot(slot) && slot_written(func, &info, slot, &stores) == Some(false) {
            return Some((slot, param, stores));
        }
    }
    None
}

/// The slot and parameter of `inst` when it stores into a slot the value loaded from the same offset of one of `params`, or moves a whole parameter into a slot.
fn param_copy(func: &Function, defs: &[Option<Inst>], params: &[(Value, u32)], inst: Inst) -> Option<(StackSlot, Value)> {
    let loaded = |value: Value, at: i64| -> Option<Value> {
        let def = (*defs.get(value.index())?)?;
        match func.insts[def.index()].op {
            Op::Load(from, offset) if i64::from(offset) == at && params.iter().any(|&(param, _)| param == from) => Some(from),
            _ => None,
        }
    };
    match func.insts[inst.index()].op {
        Op::StackStore(value, slot, offset) => Some((slot, loaded(value, i64::from(offset))?)),
        Op::Store(value, addr, offset) => {
            let def = (*defs.get(addr.index())?)?;
            let Op::StackAddr(slot, base) = func.insts[def.index()].op else { return None };
            Some((slot, loaded(value, i64::from(base) + i64::from(offset))?))
        }
        Op::Move(to, from, _) => Some((resolve_slot(func, defs, to)?, params.iter().find(|&&(param, _)| param == from)?.0)),
        _ => None,
    }
}

/// Whether anything but `stores` may write `slot`, or none when one of its addresses escapes.
fn slot_written(func: &Function, info: &super::SretUseInfo, slot: StackSlot, stores: &[(Block, Inst)]) -> Option<bool> {
    let direct = func.blocks.iter().flat_map(|block| &block.insts).any(|&inst| matches!(func.insts[inst.index()].op, Op::StackStore(_, found, _) if found == slot) && !stores.iter().any(|&(_, store)| store == inst));
    let mut roots: Vec<Value> = info.slot_addrs[slot.index()].iter().map(|&(_, _, value)| value).collect();
    roots.extend(func.insts.iter().filter_map(|data| match data.op {
        Op::StackAddr(found, offset) if found == slot && offset != 0 => data.result,
        _ => None,
    }));
    Some(direct || !accesses_at(func, info, roots, stores, Access::Written)?.is_empty())
}

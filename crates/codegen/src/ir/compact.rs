use super::{Block, Function, Inst, InstData, Op, SigRef, SlotData, StackSlot, Value};

/// Renumbers the instructions, values, blocks, live stack slots, and signatures `func` holds and drops the rest.
pub(crate) fn compact(func: &mut Function) {
    if func.coroutine.is_some() {
        return;
    }
    let mut map: Vec<Option<Value>> = vec![None; func.types.len()];
    let mut types = Vec::new();
    let old_flags = std::mem::take(&mut func.value_flags);
    let mut flags = Vec::new();
    let mut take = |value: Value| -> Value {
        *map[value.index()].get_or_insert_with(|| {
            types.push(func.types[value.index()]);
            if !old_flags.is_empty() {
                flags.push(old_flags.get(value.index()).copied().unwrap_or(0));
            }
            Value(types.len() as u32 - 1)
        })
    };
    let live = func.blocks.iter().map(|block| block.insts.len()).sum();
    let mut insts = Vec::with_capacity(live);
    let mut blocks = std::mem::take(&mut func.blocks);
    for block in &mut blocks {
        block
            .params
            .iter_mut()
            .for_each(|param| *param = take(*param));
        for inst in &mut block.insts {
            let slot = &mut func.insts[inst.index()];
            let mut data = InstData {
                op: std::mem::replace(&mut slot.op, Op::Trap),
                result: slot.result,
                loc: slot.loc,
            };
            data.result = data.result.map(&mut take);
            if let Op::Call { results, .. } = &mut data.op {
                results
                    .iter_mut()
                    .for_each(|result| *result = take(*result));
            }
            data.op.map_operands(&mut take);
            insts.push(data);
            *inst = Inst(insts.len() as u32 - 1);
        }
    }
    func.blocks = blocks;
    func.insts = insts;
    types.shrink_to_fit();
    func.types = types;
    if flags.iter().all(|&bits| bits == 0) {
        flags.clear();
    }
    func.value_flags = flags;
    compact_blocks(func);
    compact_slots(func);
    compact_sigs(func);
}

/// Drops unreferenced stack slots and remaps all live stack-slot ops and unwind-drop entries densely.
pub(crate) fn compact_slots(func: &mut Function) {
    if func.coroutine.is_some() || func.slots.is_empty() {
        return;
    }
    let mut used = vec![false; func.slots.len()];
    for block in &func.blocks {
        for &inst in &block.insts {
            match func.insts[inst.index()].op {
                Op::StackAddr(s, _) | Op::StackLoad(s, _) | Op::StackStore(_, s, _) => {
                    used[s.index()] = true
                }
                _ => {}
            }
        }
    }
    for u in &func.unwind_drops {
        used[u.local.index()] = true;
        used[u.flag.index()] = true;
    }
    if used.iter().all(|&live| live) {
        return;
    }
    let mut map: Vec<Option<StackSlot>> = vec![None; func.slots.len()];
    let mut slots: Vec<SlotData> = Vec::new();
    for (idx, data) in std::mem::take(&mut func.slots).into_iter().enumerate() {
        if used[idx] {
            map[idx] = Some(StackSlot(slots.len() as u32));
            slots.push(data);
        }
    }
    for data in &mut func.insts {
        match &mut data.op {
            Op::StackAddr(s, _) | Op::StackLoad(s, _) | Op::StackStore(_, s, _) => {
                match map[s.index()] {
                    Some(new_s) => *s = new_s,
                    None => data.op = Op::Trap,
                }
            }
            _ => {}
        }
    }
    for u in &mut func.unwind_drops {
        u.local = map[u.local.index()].expect("invariant: unwind drop local slot is marked used");
        u.flag = map[u.flag.index()].expect("invariant: unwind drop flag slot is marked used");
    }
    slots.shrink_to_fit();
    func.slots = slots;
}

fn compact_blocks(func: &mut Function) {
    let mut map: Vec<Option<Block>> = vec![None; func.blocks.len()];
    let mut blocks = Vec::new();
    for (index, mut block) in std::mem::take(&mut func.blocks).into_iter().enumerate() {
        if block.insts.is_empty() && func.entry != Some(Block(index as u32)) {
            continue;
        }
        block.params.shrink_to_fit();
        block.insts.shrink_to_fit();
        map[index] = Some(Block(blocks.len() as u32));
        blocks.push(block);
    }
    let renumber = |block: Block| map[block.index()].unwrap_or(Block(0));
    func.insts
        .iter_mut()
        .for_each(|data| data.op.map_targets(renumber));
    func.entry = func.entry.map(renumber);
    func.blocks = blocks;
    func.bump_cfg();
}

fn compact_sigs(func: &mut Function) {
    let mut map: Vec<Option<SigRef>> = vec![None; func.sigs.len()];
    let mut sigs = Vec::new();
    for data in &mut func.insts {
        if let Op::Call { sig, .. } | Op::TryCall { sig, .. } = &mut data.op {
            *sig = *map[sig.index()].get_or_insert_with(|| {
                sigs.push(func.sigs[sig.index()].clone());
                SigRef(sigs.len() as u32 - 1)
            });
        }
    }
    func.sigs = sigs;
}

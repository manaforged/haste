use crate::ir::{Binary, Block, Convert, Function, Inst, Op, StackSlot, Type, Value};

pub(super) struct SretUseInfo {
    pub(super) defs: Vec<Option<Inst>>,
    uses_off: Vec<u32>,
    uses_flat: Vec<(Block, usize, Inst)>,
    preds_off: Vec<u32>,
    preds_flat: Vec<Block>,
    pub(super) reachable: Vec<bool>,
    pub(super) bad_slots: Vec<bool>,
    pub(super) slot_addrs: Vec<Vec<(Block, Inst, Value)>>,
}

impl SretUseInfo {
    pub(super) fn uses_of(&self, v: Value) -> &[(Block, usize, Inst)] {
        let i = v.index();
        &self.uses_flat[self.uses_off[i] as usize..self.uses_off[i + 1] as usize]
    }

    pub(super) fn preds_of(&self, b: Block) -> &[Block] {
        let i = b.index();
        &self.preds_flat[self.preds_off[i] as usize..self.preds_off[i + 1] as usize]
    }
}

/// Which memory accesses a walk over derived addresses accepts.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Access {
    /// Only writes.
    Write,
    /// Reads and writes.
    ReadWrite,
    /// Only writes, letting reads and read-only call arguments pass.
    Written,
}

pub(super) fn remove_inst_in_block(func: &mut Function, block: Block, inst: Inst) {
    func.blocks[block.index()].insts.retain(|&i| i != inst);
}

/// Points every access of `slot` at the `StructReturn` memory at `sret_ptr` and deletes the `moves` that copied it out.
pub(super) fn retarget_slot(
    func: &mut Function,
    slot: StackSlot,
    sret_ptr: Value,
    moves: &[(Block, Inst)],
) {
    for &(block, inst) in moves {
        remove_inst_in_block(func, block, inst);
        func.insts[inst.index()].op = Op::Trap;
    }
    let mut subst = crate::merge::identity(func.types.len());
    for block in 0..func.blocks.len() {
        let mut at = 0;
        while at < func.blocks[block].insts.len() {
            let inst = func.blocks[block].insts[at];
            match func.insts[inst.index()].op {
                Op::StackAddr(found, 0) if found == slot => {
                    if let Some(result) = func.insts[inst.index()].result {
                        subst[result.index()] = sret_ptr;
                    }
                    func.insts[inst.index()].op = Op::Trap;
                    func.blocks[block].insts.remove(at);
                    continue;
                }
                Op::StackAddr(found, offset) if found == slot => {
                    let constant = func.value(Type::I64);
                    let def = func.add(Op::Const(i64::from(offset)), Some(constant));
                    func.insts[inst.index()].op = Op::Binary(Binary::Iadd, sret_ptr, constant);
                    func.blocks[block].insts.insert(at, def);
                    at += 1;
                }
                Op::StackLoad(found, offset) if found == slot => {
                    func.insts[inst.index()].op = Op::Load(sret_ptr, offset)
                }
                Op::StackStore(value, found, offset) if found == slot => {
                    func.insts[inst.index()].op = Op::Store(value, sret_ptr, offset)
                }
                _ => {}
            }
            at += 1;
        }
    }
    crate::merge::rename(func, &subst);
}

pub(super) fn collect_sret_info(func: &Function, entry: Block) -> SretUseInfo {
    let nblocks = func.blocks.len();
    let nvals = func.types.len();
    let mut reachable = vec![false; nblocks];
    let mut work = vec![entry];
    reachable[entry.index()] = true;
    while let Some(b) = work.pop() {
        for s in func.successors(b) {
            if !reachable[s.index()] {
                reachable[s.index()] = true;
                work.push(s);
            }
        }
    }
    let mut defs = vec![None; nvals];
    let mut uses_off = vec![0u32; nvals + 1];
    let mut preds_off = vec![0u32; nblocks + 1];
    let mut bad_slots = vec![false; func.slots.len()];
    let mut slot_addrs = vec![Vec::new(); func.slots.len()];
    for (bi, block) in func.blocks.iter().enumerate() {
        if !reachable[bi] {
            continue;
        }
        let b = Block(bi as u32);
        func.successors(b)
            .for_each(|s| preds_off[s.index() + 1] += 1);
        for &inst in &block.insts {
            let data = &func.insts[inst.index()];
            if let Some(res) = data.result {
                defs[res.index()] = Some(inst);
            }
            match &data.op {
                Op::StackLoad(slot, _) | Op::StackStore(_, slot, _) => {
                    bad_slots[slot.index()] = true
                }
                Op::StackAddr(slot, off) if *off != 0 => bad_slots[slot.index()] = true,
                Op::StackAddr(slot, _) => {
                    if let Some(res) = data.result {
                        slot_addrs[slot.index()].push((b, inst, res));
                    }
                }
                _ => {}
            }
            data.op.for_operands(|v| uses_off[v.index() + 1] += 1);
        }
    }
    for i in 1..=nvals {
        uses_off[i] += uses_off[i - 1];
    }
    for i in 1..=nblocks {
        preds_off[i] += preds_off[i - 1];
    }
    let mut uses_flat = vec![(Block(0), 0usize, Inst(0)); uses_off[nvals] as usize];
    let mut preds_flat = vec![Block(0); preds_off[nblocks] as usize];
    let (mut uses_cur, mut preds_cur) = (uses_off[..nvals].to_vec(), preds_off[..nblocks].to_vec());
    for (bi, block) in func.blocks.iter().enumerate() {
        if !reachable[bi] {
            continue;
        }
        let b = Block(bi as u32);
        for succ in func.successors(b) {
            let idx = preds_cur[succ.index()] as usize;
            preds_flat[idx] = b;
            preds_cur[succ.index()] = (idx + 1) as u32;
        }
        for (pos, &inst) in block.insts.iter().enumerate() {
            func.insts[inst.index()].op.for_operands(|v| {
                let idx = uses_cur[v.index()] as usize;
                uses_flat[idx] = (b, pos, inst);
                uses_cur[v.index()] = (idx + 1) as u32;
            });
        }
    }
    SretUseInfo {
        defs,
        uses_off,
        uses_flat,
        preds_off,
        preds_flat,
        reachable,
        bad_slots,
        slot_addrs,
    }
}

pub(super) fn move_to_sret_src(
    func: &Function,
    defs: &[Option<Inst>],
    inst: Inst,
    sret_ptr: Value,
    bytes: u32,
) -> Option<Value> {
    match func.insts[inst.index()].op {
        Op::Move(dst, src, move_bytes)
            if dst == sret_ptr && src != sret_ptr && move_bytes == bytes =>
        {
            Some(src)
        }
        Op::Copy(dst, src, len_val)
            if dst == sret_ptr && src != sret_ptr && len_val != sret_ptr =>
        {
            let len_inst = (*defs.get(len_val.index())?)?;
            match func.insts[len_inst.index()].op {
                Op::Const(imm) if imm == i64::from(bytes) => Some(src),
                _ => None,
            }
        }
        _ => None,
    }
}

pub(super) fn resolve_slot(
    func: &Function,
    defs: &[Option<Inst>],
    mut val: Value,
) -> Option<StackSlot> {
    for _ in 0..8 {
        let inst = (*defs.get(val.index())?)?;
        match func.insts[inst.index()].op {
            Op::StackAddr(slot, 0) => return Some(slot),
            Op::Convert(Convert::Bitcast, inner) => val = inner,
            _ => return None,
        }
    }
    None
}

/// Where code reads or writes slot `slot` through loads, stores, moves, copies and call arguments at its addresses, besides the `moves` that copy it out, or none when its address escapes some other way.
pub(super) fn slot_accesses(
    func: &Function,
    info: &SretUseInfo,
    slot: StackSlot,
    moves: &[(Block, Inst)],
) -> Option<Vec<(Block, usize)>> {
    let mut writes: Vec<(Block, usize)> = Vec::new();
    for (at, block) in func.blocks.iter().enumerate() {
        for (pos, inst) in block.insts.iter().enumerate() {
            match func.insts[inst.index()].op {
                Op::StackLoad(found, _) | Op::StackStore(_, found, _) if found == slot => {
                    writes.push((Block(at as u32), pos))
                }
                _ => {}
            }
        }
    }
    let mut roots: Vec<Value> = info.slot_addrs[slot.index()]
        .iter()
        .map(|&(_, _, value)| value)
        .collect();
    roots.extend(func.insts.iter().filter_map(|data| match data.op {
        Op::StackAddr(found, offset) if found == slot && offset != 0 => data.result,
        _ => None,
    }));
    writes.extend(accesses_at(func, info, roots, moves, Access::ReadWrite)?);
    Some(writes)
}

/// Where code accesses memory through `roots` and the addresses derived from them by adds, bitcasts and block arguments, skipping the instructions of `skip`, or none when an address escapes or `access` does not allow a read.
pub(super) fn accesses_at(
    func: &Function,
    info: &SretUseInfo,
    mut addrs: Vec<Value>,
    skip: &[(Block, Inst)],
    access: Access,
) -> Option<Vec<(Block, usize)>> {
    let mut writes = Vec::new();
    let mut at = 0;
    while let Some(&addr) = addrs.get(at) {
        at += 1;
        if addrs[..at - 1].contains(&addr) {
            continue;
        }
        for &(block, pos, user) in info.uses_of(addr) {
            if skip.iter().any(|&(_, inst)| inst == user) {
                continue;
            }
            let data = &func.insts[user.index()];
            match data.op {
                Op::Binary(Binary::Iadd, ..) | Op::Convert(Convert::Bitcast, _) => {
                    addrs.extend(data.result)
                }
                Op::Jump(target, ref args) => addrs.extend(edge_params(func, target, args, addr)),
                Op::Brif(cond, then, ref then_args, other, ref other_args) if cond != addr => {
                    addrs.extend(edge_params(func, then, then_args, addr));
                    addrs.extend(edge_params(func, other, other_args, addr));
                }
                ref op
                    if access == Access::Written
                        && (reads_through(op, addr) || reads_through_call(func, op, addr)) => {}
                ref op
                    if writes_through(op, addr)
                        || (access == Access::ReadWrite && reads_through(op, addr)) =>
                {
                    writes.push((block, pos))
                }
                _ => return None,
            }
        }
    }
    Some(writes)
}

fn writes_through(op: &Op, ptr: Value) -> bool {
    match *op {
        Op::Store(value, to, _) => to == ptr && value != ptr,
        Op::Move(to, from, _) => to == ptr && from != ptr,
        Op::Copy(to, from, len) => to == ptr && from != ptr && len != ptr,
        Op::Call { callee, .. } | Op::TryCall { callee, .. } => callee != ptr,
        _ => false,
    }
}

fn reads_through_call(func: &Function, op: &Op, ptr: Value) -> bool {
    let (Op::Call {
        sig,
        callee,
        ref args,
        ..
    }
    | Op::TryCall {
        sig,
        callee,
        ref args,
        ..
    }) = *op
    else {
        return false;
    };
    let params = &func.sigs[sig.index()].params;
    callee != ptr
        && args
            .iter()
            .enumerate()
            .all(|(at, &arg)| arg != ptr || params.get(at).is_some_and(|abi| abi.readonly))
}

fn reads_through(op: &Op, ptr: Value) -> bool {
    match *op {
        Op::Load(from, _) => from == ptr,
        Op::Move(to, from, _) => from == ptr && to != ptr,
        Op::Copy(to, from, len) => from == ptr && to != ptr && len != ptr,
        _ => false,
    }
}

/// Whether the instruction at `later` can run after the one at `first`, in the same block or in a block reached from it.
pub(super) fn runs_after(
    func: &Function,
    (first, first_pos): (Block, usize),
    (later, later_pos): (Block, usize),
) -> bool {
    if first == later && later_pos > first_pos {
        return true;
    }
    let mut seen = vec![false; func.blocks.len()];
    let mut work: Vec<Block> = func.successors(first).collect();
    while let Some(block) = work.pop() {
        if std::mem::replace(&mut seen[block.index()], true) {
            continue;
        }
        if block == later {
            return true;
        }
        work.extend(func.successors(block));
    }
    false
}

/// Whether a path from the instruction at `from` reaches a return or one of `bad` before it passes one of `stops`.
pub(super) fn escapes(
    func: &Function,
    (block, pos): (Block, usize),
    stops: &[(Block, usize)],
    bad: &[(Block, usize)],
) -> bool {
    let mut seen = vec![false; func.blocks.len()];
    let mut work = vec![(block, pos + 1)];
    while let Some((block, start)) = work.pop() {
        let count = func.blocks[block.index()].insts.len();
        match (start..count).find(|&at| stops.contains(&(block, at)) || bad.contains(&(block, at)))
        {
            Some(at) if stops.contains(&(block, at)) => continue,
            Some(_) => return true,
            None if matches!(func.last(block), Some(Op::Return(_))) => return true,
            None => {}
        }
        for next in func.successors(block) {
            if !std::mem::replace(&mut seen[next.index()], true) {
                work.push((next, 0));
            }
        }
    }
    false
}

fn edge_params(func: &Function, target: Block, args: &[Value], value: Value) -> Vec<Value> {
    func.blocks[target.index()]
        .params
        .iter()
        .zip(args)
        .filter(|&(_, &arg)| arg == value)
        .map(|(&param, _)| param)
        .collect()
}

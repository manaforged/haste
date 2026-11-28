use super::analysis::{
    Access, SretUseInfo, accesses_at, collect_sret_info, escapes, move_to_sret_src,
    remove_inst_in_block, resolve_slot, retarget_slot, runs_after, slot_accesses,
};
use super::join;
use crate::ir::{ArgumentPurpose, Block, BlockData, Convert, Function, Inst, Op, StackSlot, Value};

fn has_sret_move(func: &Function, sret_ptr: Value) -> bool {
    crate::schedule::has_live_op(
        func,
        |op| matches!(*op, Op::Move(dst, _, _) | Op::Copy(dst, _, _) if dst == sret_ptr),
    )
}

fn sret_param(func: &Function) -> Option<(Block, Value, u32)> {
    if func.coroutine.is_some() || func.slots.is_empty() {
        return None;
    }
    let entry = func.entry?;
    let mut sret_params = func
        .signature
        .params
        .iter()
        .enumerate()
        .filter(|(_, p)| p.purpose == ArgumentPurpose::StructReturn);
    let (Some((sret_idx, sret_abi)), None) = (sret_params.next(), sret_params.next()) else {
        return None;
    };
    if !sret_abi.noalias || sret_abi.deref_bytes == 0 {
        return None;
    }
    let sret_ptr = *func.blocks[entry.index()].params.get(sret_idx)?;
    has_sret_move(func, sret_ptr).then_some((entry, sret_ptr, sret_abi.deref_bytes))
}

/// Forwards a function's incoming `StructReturn` pointer directly into a non-aliased callee `StructReturn` temporary slot when the slot is copied out intact.
pub(super) fn forward_sret_calls(func: &mut Function) -> bool {
    let Some((entry, sret_ptr, bytes)) = sret_param(func) else {
        return false;
    };
    let mut changed = false;
    while forward_one_sret_call(func, entry, sret_ptr, bytes) {
        changed = true;
    }
    changed
}

/// Builds the return value in the incoming `StructReturn` memory, first for call results copied out intact, then for each slot that the function only writes and copies out whole right before it returns; returns whether it changed the body.
pub(crate) fn forward_sret_slot(func: &mut Function) -> bool {
    let Some((entry, sret_ptr, bytes)) = sret_param(func) else {
        return false;
    };
    let calls = forward_sret_calls(func);
    let joins = slot_return_joins(func, entry, sret_ptr, bytes);
    let split = split_returns(func, sret_ptr, bytes, &joins);
    let mut forwarded = forward_sret_slots(func, entry, sret_ptr, bytes);
    let kept = rejoin_returns(func, &split);
    loop {
        let hoisted = join::hoist(func, entry, sret_ptr, bytes);
        if hoisted.is_empty() {
            break;
        }
        let more = forward_sret_slots(func, entry, sret_ptr, bytes);
        join::settle(func, hoisted);
        if !more {
            break;
        }
        forwarded = true;
    }
    calls || forwarded || kept
}

fn forward_sret_slots(func: &mut Function, entry: Block, sret_ptr: Value, bytes: u32) -> bool {
    let mut changed = false;
    while let Some((slot, moves)) = sret_slot_plan(func, entry, sret_ptr, bytes) {
        retarget_slot(func, slot, sret_ptr, &moves);
        changed = true;
    }
    changed
}

/// The edge of a block's last instruction that leads to a copy-and-return block.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Edge {
    /// The target of a jump.
    Jump,
    /// The taken target of a branch.
    Then,
    /// The fall-through target of a branch.
    Else,
}

fn slot_return_joins(
    func: &Function,
    entry: Block,
    sret_ptr: Value,
    bytes: u32,
) -> Vec<(Block, Vec<(Block, Edge, Value)>)> {
    let info = collect_sret_info(func, entry);
    let mut splits = Vec::new();
    for &(join, pos, copy) in info.uses_of(sret_ptr) {
        let data = &func.blocks[join.index()];
        let ([param], [_, ret], 0) = (&data.params[..], &data.insts[..], pos) else {
            continue;
        };
        if !matches!(func.insts[ret.index()].op, Op::Return(ref values) if values.is_empty())
            || move_to_sret_src(func, &info.defs, copy, sret_ptr, bytes) != Some(*param)
        {
            continue;
        }
        let slot_arg = |args: &[Value]| match args {
            [arg] if resolve_slot(func, &info.defs, *arg).is_some() => Some(*arg),
            _ => None,
        };
        let mut edges = Vec::new();
        let mut all = true;
        for &pred in info.preds_of(join) {
            for (edge, args) in incoming_edges(func, pred, join) {
                match args.and_then(slot_arg) {
                    Some(addr)
                        if !edges
                            .iter()
                            .any(|&(seen, kind, _)| seen == pred && kind == edge) =>
                    {
                        edges.push((pred, edge, addr))
                    }
                    Some(_) => {}
                    None => all = false,
                }
            }
        }
        if all && !edges.is_empty() && !splits.iter().any(|(seen, _)| *seen == join) {
            splits.push((join, edges));
        }
    }
    splits
}

pub(super) fn incoming_edges<'a>(
    func: &'a Function,
    pred: Block,
    join: Block,
) -> Vec<(Edge, Option<&'a [Value]>)> {
    match func.last(pred) {
        Some(Op::Jump(target, args)) if *target == join => {
            vec![(Edge::Jump, Some(args.as_slice()))]
        }
        Some(Op::Brif(_, then, then_args, other, other_args)) => {
            let mut found = Vec::new();
            if *then == join {
                found.push((Edge::Then, Some(then_args.as_slice())));
            }
            if *other == join {
                found.push((Edge::Else, Some(other_args.as_slice())));
            }
            found
        }
        _ => vec![(Edge::Jump, None)],
    }
}

struct SplitReturn {
    block: Block,
    copy: Inst,
    ret: Inst,
    branch: Option<(Block, Edge)>,
    join: Block,
    addr: Value,
}

/// Points branch edge `edge` of the last instruction of `pred` at `target` with `args`.
pub(super) fn retarget_edge(
    func: &mut Function,
    pred: Block,
    edge: Edge,
    target: Block,
    args: Vec<Value>,
) {
    let Some(&last) = func.blocks[pred.index()].insts.last() else {
        return;
    };
    match (&mut func.insts[last.index()].op, edge) {
        (Op::Brif(_, to, to_args, ..), Edge::Then)
        | (Op::Brif(_, _, _, to, to_args), Edge::Else) => {
            *to = target;
            *to_args = args;
        }
        _ => {}
    }
}

fn split_returns(
    func: &mut Function,
    sret_ptr: Value,
    bytes: u32,
    joins: &[(Block, Vec<(Block, Edge, Value)>)],
) -> Vec<SplitReturn> {
    let mut split = Vec::new();
    for &(join, ref edges) in joins {
        for &(pred, edge, addr) in edges {
            let ret = func.add(Op::Return(Vec::new()), None);
            let (block, copy, branch) = if edge == Edge::Jump {
                let Some(&copy) = func.blocks[pred.index()].insts.last() else {
                    continue;
                };
                func.insts[copy.index()].op = Op::Move(sret_ptr, addr, bytes);
                func.blocks[pred.index()].insts.push(ret);
                (pred, copy, None)
            } else {
                let copy = func.add(Op::Move(sret_ptr, addr, bytes), None);
                let block = Block(func.blocks.len() as u32);
                func.blocks.push(BlockData {
                    params: Vec::new(),
                    insts: vec![copy, ret],
                });
                retarget_edge(func, pred, edge, block, Vec::new());
                (block, copy, Some((pred, edge)))
            };
            split.push(SplitReturn {
                block,
                copy,
                ret,
                branch,
                join,
                addr,
            });
        }
    }
    if !split.is_empty() {
        func.bump_cfg();
    }
    split
}

fn rejoin_returns(func: &mut Function, split: &[SplitReturn]) -> bool {
    let mut kept = false;
    for one in split {
        if !matches!(func.insts[one.copy.index()].op, Op::Move(..)) {
            kept = true;
            continue;
        }
        match one.branch {
            None => {
                func.insts[one.copy.index()].op = Op::Jump(one.join, vec![one.addr]);
                func.insts[one.ret.index()].op = Op::Trap;
                func.blocks[one.block.index()]
                    .insts
                    .retain(|&inst| inst != one.ret);
            }
            Some((pred, edge)) => {
                retarget_edge(func, pred, edge, one.join, vec![one.addr]);
                func.blocks[one.block.index()]
                    .insts
                    .drain(..)
                    .for_each(|inst| func.insts[inst.index()].op = Op::Trap);
            }
        }
    }
    if !split.is_empty() {
        func.bump_cfg();
    }
    kept
}

fn move_to_sret_prefix_src(
    func: &Function,
    defs: &[Option<Inst>],
    inst: Inst,
    sret_ptr: Value,
    bytes: u32,
) -> Option<(Value, u32)> {
    match func.insts[inst.index()].op {
        Op::Move(dst, src, move_bytes)
            if dst == sret_ptr && src != sret_ptr && move_bytes > 0 && move_bytes <= bytes =>
        {
            Some((src, move_bytes))
        }
        Op::Copy(dst, src, len_val)
            if dst == sret_ptr && src != sret_ptr && len_val != sret_ptr =>
        {
            let len_inst = (*defs.get(len_val.index())?)?;
            match func.insts[len_inst.index()].op {
                Op::Const(imm) if imm > 0 && imm <= i64::from(bytes) => Some((src, imm as u32)),
                _ => None,
            }
        }
        _ => None,
    }
}

fn sret_slot_plan(
    func: &Function,
    entry: Block,
    sret_ptr: Value,
    bytes: u32,
) -> Option<(StackSlot, Vec<(Block, Inst)>)> {
    let info = collect_sret_info(func, entry);
    let mut copies = Vec::new();
    for &(block, pos, inst) in info.uses_of(sret_ptr) {
        if let Some((src, move_bytes)) =
            move_to_sret_prefix_src(func, &info.defs, inst, sret_ptr, bytes)
            && let Some(slot) = resolve_slot(func, &info.defs, src)
            && func.slots[slot.index()].size <= move_bytes
        {
            copies.push((slot, block, pos, inst));
        }
    }
    let copied: Vec<(Block, Inst)> = copies
        .iter()
        .map(|&(_, block, _, inst)| (block, inst))
        .collect();
    let stops: Vec<(Block, usize)> = copies
        .iter()
        .map(|&(_, block, pos, _)| (block, pos))
        .collect();
    let writes = accesses_at(func, &info, vec![sret_ptr], &copied, Access::ReadWrite)?;
    let mut tried = Vec::new();
    for &(slot, ..) in &copies {
        if tried.contains(&slot) {
            continue;
        }
        tried.push(slot);
        let index = slot.index() as u32;
        if func.slots[slot.index()].size > bytes
            || func.self_slots.contains(&index)
            || func.vtable_slots.iter().any(|&(at, ..)| at == index)
            || func.is_unwind_slot(slot)
        {
            continue;
        }
        let moves: Vec<(Block, Inst)> = copies
            .iter()
            .filter(|copy| copy.0 == slot)
            .map(|&(_, block, _, inst)| (block, inst))
            .collect();
        let spots: Vec<(Block, usize)> = copies
            .iter()
            .filter(|copy| copy.0 == slot)
            .map(|&(_, block, pos, _)| (block, pos))
            .collect();
        let Some(accesses) = slot_accesses(func, &info, slot, &moves) else {
            continue;
        };
        if spots.iter().any(|&spot| {
            accesses
                .iter()
                .chain(&spots)
                .any(|&access| runs_after(func, spot, access))
        }) {
            continue;
        }
        let others: Vec<(Block, usize)> = copies
            .iter()
            .filter(|copy| copy.0 != slot)
            .map(|&(_, block, pos, _)| (block, pos))
            .chain(writes.iter().copied())
            .collect();
        let live: Vec<(Block, usize)> = accesses.iter().chain(&spots).copied().collect();
        let before =
            |other: (Block, usize)| live.iter().any(|&access| runs_after(func, other, access));
        if others.iter().any(|&other| {
            live.contains(&other)
                || (before(other) && live.iter().any(|&access| runs_after(func, access, other)))
        }) {
            continue;
        }
        if others.iter().any(|&other| before(other))
            && accesses
                .iter()
                .any(|&access| escapes(func, access, &stops, &others))
        {
            continue;
        }
        return Some((slot, moves));
    }
    None
}

struct SretCandidate {
    block: Block,
    call_inst: Inst,
    call_sret_idx: usize,
    move_inst: Inst,
    dead_insts: Vec<(Block, Inst)>,
}

fn forward_one_sret_call(func: &mut Function, entry: Block, sret_ptr: Value, bytes: u32) -> bool {
    let info = collect_sret_info(func, entry);
    let Some(cand) = find_sret_candidate(func, &info, sret_ptr, bytes) else {
        return false;
    };
    if let Op::Call { args, .. } = &mut func.insts[cand.call_inst.index()].op {
        args[cand.call_sret_idx] = sret_ptr;
    }
    remove_inst_in_block(func, cand.block, cand.move_inst);
    func.insts[cand.move_inst.index()].op = Op::Trap;
    for (dead_b, dead_inst) in cand.dead_insts {
        remove_inst_in_block(func, dead_b, dead_inst);
        func.insts[dead_inst.index()].op = Op::Trap;
    }
    true
}

fn find_sret_candidate(
    func: &Function,
    info: &SretUseInfo,
    sret_ptr: Value,
    bytes: u32,
) -> Option<SretCandidate> {
    let sret_uses = info.uses_of(sret_ptr);
    for &(b, move_pos, move_inst) in sret_uses {
        if !info.reachable[b.index()] {
            continue;
        }
        let Some(src) = move_to_sret_src(func, &info.defs, move_inst, sret_ptr, bytes) else {
            continue;
        };
        let Some(slot) = resolve_slot(func, &info.defs, src) else {
            continue;
        };
        if info.bad_slots[slot.index()] || !sret_ptr_clean_before(info, sret_uses, b, move_pos) {
            continue;
        }
        for &call_inst in &func.blocks[b.index()].insts[..move_pos] {
            let Some(call_sret_idx) = call_sret_param_idx(func, call_inst, bytes) else {
                continue;
            };
            if let Some(dead_insts) =
                exclusive_slot_insts(func, info, slot, call_inst, call_sret_idx, move_inst)
            {
                return Some(SretCandidate {
                    block: b,
                    call_inst,
                    call_sret_idx,
                    move_inst,
                    dead_insts,
                });
            }
        }
    }
    None
}

fn sret_ptr_clean_before(
    info: &SretUseInfo,
    sret_uses: &[(Block, usize, Inst)],
    block: Block,
    move_pos: usize,
) -> bool {
    let mut can_reach = vec![false; info.reachable.len()];
    let mut stack = info.preds_of(block).to_vec();
    for &p in &stack {
        can_reach[p.index()] = true;
    }
    while let Some(cur) = stack.pop() {
        for &p in info.preds_of(cur) {
            if !can_reach[p.index()] {
                can_reach[p.index()] = true;
                stack.push(p);
            }
        }
    }
    sret_uses
        .iter()
        .all(|&(ub, upos, _)| !can_reach[ub.index()] && (ub != block || upos >= move_pos))
}

fn call_sret_param_idx(func: &Function, call_inst: Inst, bytes: u32) -> Option<usize> {
    let Op::Call { sig, args, .. } = &func.insts[call_inst.index()].op else {
        return None;
    };
    let params = &func.sigs[sig.index()].params;
    let mut sret_params = params
        .iter()
        .enumerate()
        .filter(|(_, p)| p.purpose == ArgumentPurpose::StructReturn);
    let (Some((idx, abi)), None) = (sret_params.next(), sret_params.next()) else {
        return None;
    };
    (abi.noalias && abi.deref_bytes == bytes && idx < args.len()).then_some(idx)
}

fn exclusive_slot_insts(
    func: &Function,
    info: &SretUseInfo,
    slot: StackSlot,
    call_inst: Inst,
    call_sret_idx: usize,
    move_inst: Inst,
) -> Option<Vec<(Block, Inst)>> {
    let addrs = &info.slot_addrs[slot.index()];
    if addrs.is_empty() {
        return None;
    }
    let mut derived = Vec::with_capacity(addrs.len() + 2);
    let mut dead_insts = Vec::with_capacity(addrs.len() + 2);
    for &(ab, ainst, aval) in addrs {
        derived.push(aval);
        dead_insts.push((ab, ainst));
    }
    let mut i = 0;
    let mut saw_call = false;
    let mut saw_move = false;
    while i < derived.len() {
        let v = derived[i];
        for &(ub, _, uinst) in info.uses_of(v) {
            if uinst == call_inst {
                let Op::Call { callee, args, .. } = &func.insts[call_inst.index()].op else {
                    return None;
                };
                if *callee == v
                    || args
                        .iter()
                        .enumerate()
                        .any(|(ai, &a)| a == v && ai != call_sret_idx)
                {
                    return None;
                }
                if args.get(call_sret_idx) == Some(&v) {
                    saw_call = true;
                }
            } else if uinst == move_inst {
                let is_move_src = match func.insts[move_inst.index()].op {
                    Op::Move(dst, src, _) => dst != v && src == v,
                    Op::Copy(dst, src, len_val) => dst != v && len_val != v && src == v,
                    _ => false,
                };
                if !is_move_src {
                    return None;
                }
                saw_move = true;
            } else if let Op::Convert(Convert::Bitcast, inner) = func.insts[uinst.index()].op
                && inner == v
                && let Some(res) = func.insts[uinst.index()].result
            {
                if !derived.contains(&res) {
                    derived.push(res);
                    dead_insts.push((ub, uinst));
                }
            } else {
                return None;
            }
        }
        i += 1;
    }
    (saw_call && saw_move).then_some(dead_insts)
}

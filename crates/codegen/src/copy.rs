mod chain;
mod join;
mod param;

pub(crate) use chain::build_in_place;
pub(crate) use param::forward_param_slots;

use crate::ir::{ArgumentPurpose, Binary, Block, BlockData, Convert, Function, Inst, Lane, Op, StackSlot, Type, Value};

/// The most bytes a move is emitted inline with; a larger one calls `memmove`.
pub(crate) const MOVE_INLINE: u32 = 128;

/// The most chunks a copy of a constant size is emitted inline with, one scratch register each.
const CHUNKS: usize = 4;

/// The integer type and offset of each load and store that copy `bytes` bytes word by word, widest first.
pub(crate) fn words(bytes: u32) -> Vec<(Type, u32)> {
    let mut words = Vec::new();
    let mut at = 0;
    while at < bytes {
        let ty = [Type::I64, Type::I32, Type::I16].into_iter().find(|ty| ty.bytes() <= bytes - at).unwrap_or(Type::I8);
        words.push((ty, at));
        at += ty.bytes();
    }
    words
}

/// The vector-first types and offsets of each load and store that copy `bytes` bytes, using 16-byte vectors before `words`.
pub(crate) fn vec_words(bytes: u32) -> Vec<(Type, u32)> {
    let mut out = Vec::new();
    let mut at = 0;
    let vec_ty = Type::Vec { lane: Lane::I8, count: 16 };
    while bytes - at >= 16 {
        out.push((vec_ty, at));
        at += 16;
    }
    if at < bytes {
        out.extend(words(bytes - at).into_iter().map(|(ty, off)| (ty, at + off)));
    }
    out
}

/// The integer types and offsets of the loads and stores that copy `bytes` bytes inline, widest first, up to `MOVE_INLINE`.
pub(crate) fn copy_words(bytes: i64) -> Option<Vec<(Type, u32)>> {
    u32::try_from(bytes).ok().filter(|&bytes| bytes <= MOVE_INLINE).map(words)
}

/// The widths and offsets of the loads that copy `bytes` bytes inline, widest first, when there are at most `CHUNKS` of them.
pub(crate) fn chunks(bytes: i64) -> Option<Vec<(u32, u32)>> {
    let words = copy_words(bytes).filter(|words| words.len() <= CHUNKS)?;
    Some(words.into_iter().map(|(ty, at)| (ty.bytes(), at)).collect())
}

/// A move that `expand` split into the loads and stores in `pairs`, which `rejoin` can turn back into it.
pub(crate) struct Split {
    block: usize,
    original: Inst,
    pairs: Vec<Inst>,
}

/// The most words a single-ended move is expanded into for scalar promotion.
pub(crate) const EXPAND_WORDS: usize = 16;

/// Replaces each move for which `split` gives pieces, from its target, source and size, with a load and a store per piece, in order, and returns what it split.
pub(crate) fn expand(func: &mut Function, split: impl Fn(Value, Value, u32) -> Option<Vec<(Type, u32)>>) -> Vec<Split> {
    let mut splits = Vec::new();
    for block in 0..func.blocks.len() {
        let insts = std::mem::take(&mut func.blocks[block].insts);
        let mut out = Vec::with_capacity(insts.len());
        for inst in insts {
            let pieces = match func.insts[inst.index()].op {
                Op::Move(to, from, bytes) => split(to, from, bytes).map(|pieces| (to, from, pieces)),
                _ => None,
            };
            let Some((to, from, pieces)) = pieces else {
                out.push(inst);
                continue;
            };
            let start = out.len();
            pairs(func, (to, from, pieces), &mut out);
            splits.push(Split { block, original: inst, pairs: out[start..].to_vec() });
        }
        func.blocks[block].insts = out;
    }
    splits
}

/// Appends to `out` a load from `from` and a store to `to` for each piece of a move.
fn pairs(func: &mut Function, (to, from, pieces): (Value, Value, Vec<(Type, u32)>), out: &mut Vec<Inst>) {
    for (ty, at) in pieces {
        let word = func.value(ty);
        out.push(func.add(Op::Load(from, at as i32), Some(word)));
        out.push(func.add(Op::Store(word, to, at as i32), None));
    }
}

/// Turns back into its move each split whose loads and stores all still stand together, each store writing the word its load read.
pub(crate) fn rejoin(func: &mut Function, splits: &[Split], forward_sret: bool) {
    for split in splits {
        let insts = &func.blocks[split.block].insts;
        let Some(at) = insts.iter().position(|inst| Some(inst) == split.pairs.first()) else { continue };
        let Some(ends) = insts.get(at..at + split.pairs.len()).filter(|window| *window == split.pairs).and_then(|_| bases(func, &split.pairs)) else { continue };
        let Op::Move(_, _, bytes) = func.insts[split.original.index()].op else { continue };
        func.insts[split.original.index()].op = Op::Move(ends.0, ends.1, bytes);
        func.blocks[split.block].insts.splice(at..at + split.pairs.len(), [split.original]);
    }
    if forward_sret {
        forward_sret_calls(func);
    }
}

fn has_sret_move(func: &Function, sret_ptr: Value) -> bool {
    crate::schedule::has_live_op(func, |op| matches!(*op, Op::Move(dst, _, _) | Op::Copy(dst, _, _) if dst == sret_ptr))
}

/// The entry block, incoming `StructReturn` pointer and its size, when the function has one noalias return pointer and slots that could stand for it.
fn sret_param(func: &Function) -> Option<(Block, Value, u32)> {
    if func.coroutine.is_some() || func.slots.is_empty() {
        return None;
    }
    let entry = func.entry?;
    let mut sret_params = func.signature.params.iter().enumerate().filter(|(_, p)| p.purpose == ArgumentPurpose::StructReturn);
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
fn forward_sret_calls(func: &mut Function) -> bool {
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

/// Builds in the `StructReturn` memory at `sret_ptr` each slot that `sret_slot_plan` finds, one at a time; returns whether it built any.
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
enum Edge {
    /// The target of a jump.
    Jump,
    /// The taken target of a branch.
    Then,
    /// The fall-through target of a branch.
    Else,
}

/// The blocks that only copy their one parameter to `sret_ptr` and return, each with the edges that reach it and the slot address each passes.
fn slot_return_joins(func: &Function, entry: Block, sret_ptr: Value, bytes: u32) -> Vec<(Block, Vec<(Block, Edge, Value)>)> {
    let info = collect_sret_info(func, entry);
    let mut splits = Vec::new();
    for &(join, pos, copy) in info.uses_of(sret_ptr) {
        let data = &func.blocks[join.index()];
        let ([param], [_, ret], 0) = (&data.params[..], &data.insts[..], pos) else { continue };
        if !matches!(func.insts[ret.index()].op, Op::Return(ref values) if values.is_empty()) || move_to_sret_src(func, &info.defs, copy, sret_ptr, bytes) != Some(*param) {
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
                    Some(addr) if !edges.iter().any(|&(seen, kind, _)| seen == pred && kind == edge) => edges.push((pred, edge, addr)),
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

fn incoming_edges<'a>(func: &'a Function, pred: Block, join: Block) -> Vec<(Edge, Option<&'a [Value]>)> {
    match func.last(pred) {
        Some(Op::Jump(target, args)) if *target == join => vec![(Edge::Jump, Some(args.as_slice()))],
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

/// One edge that `split_returns` gave its own copy to the return memory: the block holding the copy and its return, the branch edge sent to that block when it is new, the join it reached and the address it passed.
struct SplitReturn {
    block: Block,
    copy: Inst,
    ret: Inst,
    branch: Option<(Block, Edge)>,
    join: Block,
    addr: Value,
}

/// Points branch edge `edge` of the last instruction of `pred` at `target` with `args`.
fn retarget_edge(func: &mut Function, pred: Block, edge: Edge, target: Block, args: Vec<Value>) {
    let Some(&last) = func.blocks[pred.index()].insts.last() else { return };
    match (&mut func.insts[last.index()].op, edge) {
        (Op::Brif(_, to, to_args, ..), Edge::Then) | (Op::Brif(_, _, _, to, to_args), Edge::Else) => {
            *to = target;
            *to_args = args;
        }
        _ => {}
    }
}

/// Gives each edge of `joins` that reaches a copy-and-return block its own copy to `sret_ptr` and return, in place of a jump or in a new block for a branch edge, so each slot has its own copy-out.
fn split_returns(func: &mut Function, sret_ptr: Value, bytes: u32, joins: &[(Block, Vec<(Block, Edge, Value)>)]) -> Vec<SplitReturn> {
    let mut split = Vec::new();
    for &(join, ref edges) in joins {
        for &(pred, edge, addr) in edges {
            let ret = func.add(Op::Return(Vec::new()), None);
            let (block, copy, branch) = if edge == Edge::Jump {
                let Some(&copy) = func.blocks[pred.index()].insts.last() else { continue };
                func.insts[copy.index()].op = Op::Move(sret_ptr, addr, bytes);
                func.blocks[pred.index()].insts.push(ret);
                (pred, copy, None)
            } else {
                let copy = func.add(Op::Move(sret_ptr, addr, bytes), None);
                let block = Block(func.blocks.len() as u32);
                func.blocks.push(BlockData { params: Vec::new(), insts: vec![copy, ret] });
                retarget_edge(func, pred, edge, block, Vec::new());
                (block, copy, Some((pred, edge)))
            };
            split.push(SplitReturn { block, copy, ret, branch, join, addr });
        }
    }
    if !split.is_empty() {
        func.bump_cfg();
    }
    split
}

/// Points each edge of `split` whose copy still stands back at its join, so only forwarded slots keep their own return; returns whether any edge kept its own return.
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
                func.blocks[one.block.index()].insts.retain(|&inst| inst != one.ret);
            }
            Some((pred, edge)) => {
                retarget_edge(func, pred, edge, one.join, vec![one.addr]);
                func.blocks[one.block.index()].insts.drain(..).for_each(|inst| func.insts[inst.index()].op = Op::Trap);
            }
        }
    }
    if !split.is_empty() {
        func.bump_cfg();
    }
    kept
}

/// A slot that can live in the `StructReturn` memory at `sret_ptr`, with the moves that copy it out: no access or copy of it can run after one of its copies or after another write of the return memory.
fn sret_slot_plan(func: &Function, entry: Block, sret_ptr: Value, bytes: u32) -> Option<(StackSlot, Vec<(Block, Inst)>)> {
    let info = collect_sret_info(func, entry);
    let mut copies = Vec::new();
    for &(block, pos, inst) in info.uses_of(sret_ptr) {
        if let Some(slot) = move_to_sret_src(func, &info.defs, inst, sret_ptr, bytes).and_then(|src| resolve_slot(func, &info.defs, src)) {
            copies.push((slot, block, pos, inst));
        }
    }
    let copied: Vec<(Block, Inst)> = copies.iter().map(|&(_, block, _, inst)| (block, inst)).collect();
    let stops: Vec<(Block, usize)> = copies.iter().map(|&(_, block, pos, _)| (block, pos)).collect();
    let writes = accesses_at(func, &info, vec![sret_ptr], &copied, Access::ReadWrite)?;
    let mut tried = Vec::new();
    for &(slot, ..) in &copies {
        if tried.contains(&slot) {
            continue;
        }
        tried.push(slot);
        let index = slot.index() as u32;
        if func.slots[slot.index()].size > bytes || func.self_slots.contains(&index) || func.vtable_slots.iter().any(|&(at, ..)| at == index) || func.is_unwind_slot(slot) {
            continue;
        }
        let moves: Vec<(Block, Inst)> = copies.iter().filter(|copy| copy.0 == slot).map(|&(_, block, _, inst)| (block, inst)).collect();
        let spots: Vec<(Block, usize)> = copies.iter().filter(|copy| copy.0 == slot).map(|&(_, block, pos, _)| (block, pos)).collect();
        let Some(accesses) = slot_accesses(func, &info, slot, &moves) else { continue };
        if spots.iter().any(|&spot| accesses.iter().chain(&spots).any(|&access| runs_after(func, spot, access))) {
            continue;
        }
        let others: Vec<(Block, usize)> = copies.iter().filter(|copy| copy.0 != slot).map(|&(_, block, pos, _)| (block, pos)).chain(writes.iter().copied()).collect();
        let live: Vec<(Block, usize)> = accesses.iter().chain(&spots).copied().collect();
        let before = |other: (Block, usize)| live.iter().any(|&access| runs_after(func, other, access));
        if others.iter().any(|&other| live.contains(&other) || (before(other) && live.iter().any(|&access| runs_after(func, access, other)))) {
            continue;
        }
        if others.iter().any(|&other| before(other)) && accesses.iter().any(|&access| escapes(func, access, &stops, &others)) {
            continue;
        }
        return Some((slot, moves));
    }
    None
}

/// Points every access of `slot` at the `StructReturn` memory at `sret_ptr` and deletes the `moves` that copied it out.
fn retarget_slot(func: &mut Function, slot: StackSlot, sret_ptr: Value, moves: &[(Block, Inst)]) {
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
                Op::StackLoad(found, offset) if found == slot => func.insts[inst.index()].op = Op::Load(sret_ptr, offset),
                Op::StackStore(value, found, offset) if found == slot => func.insts[inst.index()].op = Op::Store(value, sret_ptr, offset),
                _ => {}
            }
            at += 1;
        }
    }
    crate::merge::rename(func, &subst);
}

struct SretUseInfo {
    defs: Vec<Option<Inst>>,
    uses_off: Vec<u32>,
    uses_flat: Vec<(Block, usize, Inst)>,
    preds_off: Vec<u32>,
    preds_flat: Vec<Block>,
    reachable: Vec<bool>,
    bad_slots: Vec<bool>,
    slot_addrs: Vec<Vec<(Block, Inst, Value)>>,
}

impl SretUseInfo {
    fn uses_of(&self, v: Value) -> &[(Block, usize, Inst)] {
        let i = v.index();
        &self.uses_flat[self.uses_off[i] as usize..self.uses_off[i + 1] as usize]
    }

    fn preds_of(&self, b: Block) -> &[Block] {
        let i = b.index();
        &self.preds_flat[self.preds_off[i] as usize..self.preds_off[i + 1] as usize]
    }
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

fn remove_inst_in_block(func: &mut Function, block: Block, inst: Inst) {
    func.blocks[block.index()].insts.retain(|&i| i != inst);
}

fn collect_sret_info(func: &Function, entry: Block) -> SretUseInfo {
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
        if !reachable[bi] { continue; }
        let b = Block(bi as u32);
        func.successors(b).for_each(|s| preds_off[s.index() + 1] += 1);
        for &inst in &block.insts {
            let data = &func.insts[inst.index()];
            if let Some(res) = data.result { defs[res.index()] = Some(inst); }
            match &data.op {
                Op::StackLoad(slot, _) | Op::StackStore(_, slot, _) => bad_slots[slot.index()] = true,
                Op::StackAddr(slot, off) if *off != 0 => bad_slots[slot.index()] = true,
                Op::StackAddr(slot, _) => if let Some(res) = data.result { slot_addrs[slot.index()].push((b, inst, res)); }
                _ => {}
            }
            data.op.for_operands(|v| uses_off[v.index() + 1] += 1);
        }
    }
    for i in 1..=nvals { uses_off[i] += uses_off[i - 1]; }
    for i in 1..=nblocks { preds_off[i] += preds_off[i - 1]; }
    let mut uses_flat = vec![(Block(0), 0usize, Inst(0)); uses_off[nvals] as usize];
    let mut preds_flat = vec![Block(0); preds_off[nblocks] as usize];
    let (mut uses_cur, mut preds_cur) = (uses_off[..nvals].to_vec(), preds_off[..nblocks].to_vec());
    for (bi, block) in func.blocks.iter().enumerate() {
        if !reachable[bi] { continue; }
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
    SretUseInfo { defs, uses_off, uses_flat, preds_off, preds_flat, reachable, bad_slots, slot_addrs }
}

fn find_sret_candidate(func: &Function, info: &SretUseInfo, sret_ptr: Value, bytes: u32) -> Option<SretCandidate> {
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
            if let Some(dead_insts) = exclusive_slot_insts(func, info, slot, call_inst, call_sret_idx, move_inst) {
                return Some(SretCandidate { block: b, call_inst, call_sret_idx, move_inst, dead_insts });
            }
        }
    }
    None
}

fn move_to_sret_src(func: &Function, defs: &[Option<Inst>], inst: Inst, sret_ptr: Value, bytes: u32) -> Option<Value> {
    match func.insts[inst.index()].op {
        Op::Move(dst, src, move_bytes) if dst == sret_ptr && src != sret_ptr && move_bytes == bytes => Some(src),
        Op::Copy(dst, src, len_val) if dst == sret_ptr && src != sret_ptr && len_val != sret_ptr => {
            let len_inst = (*defs.get(len_val.index())?)?;
            match func.insts[len_inst.index()].op {
                Op::Const(imm) if imm == i64::from(bytes) => Some(src),
                _ => None,
            }
        }
        _ => None,
    }
}

fn resolve_slot(func: &Function, defs: &[Option<Inst>], mut val: Value) -> Option<StackSlot> {
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

fn sret_ptr_clean_before(info: &SretUseInfo, sret_uses: &[(Block, usize, Inst)], block: Block, move_pos: usize) -> bool {
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
    sret_uses.iter().all(|&(ub, upos, _)| !can_reach[ub.index()] && (ub != block || upos >= move_pos))
}

fn call_sret_param_idx(func: &Function, call_inst: Inst, bytes: u32) -> Option<usize> {
    let Op::Call { sig, args, .. } = &func.insts[call_inst.index()].op else {
        return None;
    };
    let params = &func.sigs[sig.index()].params;
    let mut sret_params = params.iter().enumerate().filter(|(_, p)| p.purpose == ArgumentPurpose::StructReturn);
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
                if *callee == v || args.iter().enumerate().any(|(ai, &a)| a == v && ai != call_sret_idx) {
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

/// Where code reads or writes slot `slot` through loads, stores, moves, copies and call arguments at its addresses, besides the `moves` that copy it out, or none when its address escapes some other way.
fn slot_accesses(func: &Function, info: &SretUseInfo, slot: StackSlot, moves: &[(Block, Inst)]) -> Option<Vec<(Block, usize)>> {
    let mut writes: Vec<(Block, usize)> = Vec::new();
    for (at, block) in func.blocks.iter().enumerate() {
        for (pos, inst) in block.insts.iter().enumerate() {
            match func.insts[inst.index()].op {
                Op::StackLoad(found, _) | Op::StackStore(_, found, _) if found == slot => writes.push((Block(at as u32), pos)),
                _ => {}
            }
        }
    }
    let mut roots: Vec<Value> = info.slot_addrs[slot.index()].iter().map(|&(_, _, value)| value).collect();
    roots.extend(func.insts.iter().filter_map(|data| match data.op {
        Op::StackAddr(found, offset) if found == slot && offset != 0 => data.result,
        _ => None,
    }));
    writes.extend(accesses_at(func, info, roots, moves, Access::ReadWrite)?);
    Some(writes)
}

/// Which memory accesses a walk over derived addresses accepts.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Access {
    /// Only writes.
    Write,
    /// Reads and writes.
    ReadWrite,
    /// Only writes, letting reads and read-only call arguments pass.
    Written,
}

/// Where code accesses memory through `roots` and the addresses derived from them by adds, bitcasts and block arguments, skipping the instructions of `skip`, or none when an address escapes or `access` does not allow a read.
fn accesses_at(func: &Function, info: &SretUseInfo, mut addrs: Vec<Value>, skip: &[(Block, Inst)], access: Access) -> Option<Vec<(Block, usize)>> {
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
                Op::Binary(Binary::Iadd, ..) | Op::Convert(Convert::Bitcast, _) => addrs.extend(data.result),
                Op::Jump(target, ref args) => addrs.extend(edge_params(func, target, args, addr)),
                Op::Brif(cond, then, ref then_args, other, ref other_args) if cond != addr => {
                    addrs.extend(edge_params(func, then, then_args, addr));
                    addrs.extend(edge_params(func, other, other_args, addr));
                }
                ref op if access == Access::Written && (reads_through(op, addr) || reads_through_call(func, op, addr)) => {}
                ref op if writes_through(op, addr) || (access == Access::ReadWrite && reads_through(op, addr)) => writes.push((block, pos)),
                _ => return None,
            }
        }
    }
    Some(writes)
}

/// Whether `op` only writes memory at `ptr`: a store to it, a move or copy into it, or a call that takes it as an argument.
fn writes_through(op: &Op, ptr: Value) -> bool {
    match *op {
        Op::Store(value, to, _) => to == ptr && value != ptr,
        Op::Move(to, from, _) => to == ptr && from != ptr,
        Op::Copy(to, from, len) => to == ptr && from != ptr && len != ptr,
        Op::Call { callee, .. } | Op::TryCall { callee, .. } => callee != ptr,
        _ => false,
    }
}

/// Whether `op` is a call that passes `ptr` only as arguments its callee marks read-only.
fn reads_through_call(func: &Function, op: &Op, ptr: Value) -> bool {
    let (Op::Call { sig, callee, ref args, .. } | Op::TryCall { sig, callee, ref args, .. }) = *op else { return false };
    let params = &func.sigs[sig.index()].params;
    callee != ptr && args.iter().enumerate().all(|(at, &arg)| arg != ptr || params.get(at).is_some_and(|abi| abi.readonly))
}

/// Whether `op` only reads memory at `ptr`: a load from it, or a move or copy out of it.
fn reads_through(op: &Op, ptr: Value) -> bool {
    match *op {
        Op::Load(from, _) => from == ptr,
        Op::Move(to, from, _) => from == ptr && to != ptr,
        Op::Copy(to, from, len) => from == ptr && to != ptr && len != ptr,
        _ => false,
    }
}

/// Whether the instruction at `later` can run after the one at `first`, in the same block or in a block reached from it.
fn runs_after(func: &Function, (first, first_pos): (Block, usize), (later, later_pos): (Block, usize)) -> bool {
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
fn escapes(func: &Function, (block, pos): (Block, usize), stops: &[(Block, usize)], bad: &[(Block, usize)]) -> bool {
    let mut seen = vec![false; func.blocks.len()];
    let mut work = vec![(block, pos + 1)];
    while let Some((block, start)) = work.pop() {
        let count = func.blocks[block.index()].insts.len();
        match (start..count).find(|&at| stops.contains(&(block, at)) || bad.contains(&(block, at))) {
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

/// The parameters of `target` that the edge arguments `args` bind to `value`.
fn edge_params(func: &Function, target: Block, args: &[Value], value: Value) -> Vec<Value> {
    func.blocks[target.index()].params.iter().zip(args).filter(|&(_, &arg)| arg == value).map(|(&param, _)| param).collect()
}

/// The target and source address of loads and stores that copy word by word, when every store writes the word the load before it read.
fn bases(func: &Function, pairs: &[Inst]) -> Option<(Value, Value)> {
    let mut ends = None;
    for pair in pairs.chunks(2) {
        let [load, store] = pair else { return None };
        let (Op::Load(from, _), Some(word)) = (&func.insts[load.index()].op, func.insts[load.index()].result) else { return None };
        let Op::Store(value, to, _) = func.insts[store.index()].op else { return None };
        if value != word || ends.is_some_and(|ends| ends != (to, *from)) {
            return None;
        }
        ends = Some((to, *from));
    }
    ends
}

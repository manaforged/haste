use crate::ir::{ArgumentPurpose, Binary, Block, Convert, Function, Inst, Lane, Op, StackSlot, Type, Value};

/// The most bytes a move is emitted inline with; a larger one calls `memmove`.
pub(crate) const MOVE_INLINE: u32 = 64;

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
pub(crate) fn rejoin(func: &mut Function, splits: &[Split]) {
    for split in splits {
        let insts = &func.blocks[split.block].insts;
        let Some(at) = insts.iter().position(|inst| Some(inst) == split.pairs.first()) else { continue };
        let Some(ends) = insts.get(at..at + split.pairs.len()).filter(|window| *window == split.pairs).and_then(|_| bases(func, &split.pairs)) else { continue };
        let Op::Move(_, _, bytes) = func.insts[split.original.index()].op else { continue };
        func.insts[split.original.index()].op = Op::Move(ends.0, ends.1, bytes);
        func.blocks[split.block].insts.splice(at..at + split.pairs.len(), [split.original]);
    }
    forward_sret_calls(func);
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
    Some((entry, sret_ptr, sret_abi.deref_bytes))
}

/// Forwards a function's incoming `StructReturn` pointer directly into a non-aliased callee `StructReturn` temporary slot when the slot is copied out intact.
fn forward_sret_calls(func: &mut Function) {
    let Some((entry, sret_ptr, bytes)) = sret_param(func) else {
        return;
    };
    while forward_one_sret_call(func, entry, sret_ptr, bytes) {}
}

/// Builds the return value in the incoming `StructReturn` memory when the function fills one slot and only ever copies that slot out whole right before it returns.
pub(crate) fn forward_sret_slot(func: &mut Function) -> bool {
    let Some((entry, sret_ptr, bytes)) = sret_param(func) else {
        return false;
    };
    let info = collect_sret_info(func, entry);
    let mut slot = None;
    let mut moves = Vec::new();
    let mut others = Vec::new();
    let mut copies = Vec::new();
    for &(block, pos, inst) in &info.uses_by_val[sret_ptr.index()] {
        let insts = &func.blocks[block.index()].insts;
        let returns_next = pos + 2 == insts.len() && matches!(func.insts[insts[pos + 1].index()].op, Op::Return(_));
        match move_to_sret_src(func, &info.defs, inst, sret_ptr, bytes).and_then(|src| resolve_slot(func, &info.defs, src)) {
            Some(found) if returns_next && slot.is_none_or(|slot| slot == found) => {
                slot = Some(found);
                moves.push((block, inst));
                copies.push((block, pos));
            }
            _ if writes_through(&func.insts[inst.index()].op, sret_ptr) => others.push((block, pos)),
            _ => return false,
        }
    }
    let Some(slot) = slot else {
        return false;
    };
    let index = slot.index() as u32;
    if func.slots[slot.index()].size > bytes || func.self_slots.contains(&index) || func.vtable_slots.iter().any(|&(at, ..)| at == index) {
        return false;
    }
    let Some(accesses) = written_only(func, &info, slot, &moves) else {
        return false;
    };
    if others.iter().any(|&other| accesses.iter().chain(&copies).any(|&access| access == other || runs_after(func, other, access))) {
        return false;
    }
    for (block, inst) in moves {
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
    true
}

struct SretUseInfo {
    defs: Vec<Option<Inst>>,
    uses_by_val: Vec<Vec<(Block, usize, Inst)>>,
    preds: Vec<Vec<Block>>,
    reachable: Vec<bool>,
    bad_slots: Vec<bool>,
    slot_addrs: Vec<Vec<(Block, Inst, Value)>>,
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
    let mut defs = vec![None; func.types.len()];
    let mut uses_by_val = vec![Vec::new(); func.types.len()];
    let mut preds = vec![Vec::new(); nblocks];
    let mut bad_slots = vec![false; func.slots.len()];
    let mut slot_addrs = vec![Vec::new(); func.slots.len()];
    for (bi, block) in func.blocks.iter().enumerate() {
        if !reachable[bi] {
            continue;
        }
        let b = Block(bi as u32);
        for succ in func.successors(b) {
            preds[succ.index()].push(b);
        }
        for (pos, &inst) in block.insts.iter().enumerate() {
            let data = &func.insts[inst.index()];
            if let Some(res) = data.result {
                defs[res.index()] = Some(inst);
            }
            match &data.op {
                Op::StackLoad(slot, _) | Op::StackStore(_, slot, _) => bad_slots[slot.index()] = true,
                Op::StackAddr(slot, off) => {
                    if *off != 0 {
                        bad_slots[slot.index()] = true;
                    } else if let Some(res) = data.result {
                        slot_addrs[slot.index()].push((b, inst, res));
                    }
                }
                _ => {}
            }
            data.op.for_operands(|v| uses_by_val[v.index()].push((b, pos, inst)));
        }
    }
    SretUseInfo { defs, uses_by_val, preds, reachable, bad_slots, slot_addrs }
}

fn find_sret_candidate(func: &Function, info: &SretUseInfo, sret_ptr: Value, bytes: u32) -> Option<SretCandidate> {
    let sret_uses = &info.uses_by_val[sret_ptr.index()];
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
    let mut can_reach = vec![false; info.preds.len()];
    let mut stack = info.preds[block.index()].clone();
    for &p in &stack {
        can_reach[p.index()] = true;
    }
    while let Some(cur) = stack.pop() {
        for &p in &info.preds[cur.index()] {
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
        for &(ub, _, uinst) in &info.uses_by_val[v.index()] {
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

/// Where code accesses slot `slot` when it only writes it, through stores, move and copy targets and call arguments at its addresses, besides the `moves` that copy it out.
fn written_only(func: &Function, info: &SretUseInfo, slot: StackSlot, moves: &[(Block, Inst)]) -> Option<Vec<(Block, usize)>> {
    if info.bad_slots[slot.index()] && func.insts.iter().any(|data| matches!(data.op, Op::StackLoad(found, _) if found == slot)) {
        return None;
    }
    let mut accesses: Vec<(Block, usize)> = func.blocks.iter().enumerate().flat_map(|(at, block)| {
        block.insts.iter().enumerate().filter(|&(_, inst)| matches!(func.insts[inst.index()].op, Op::StackStore(_, found, _) if found == slot)).map(move |(pos, _)| (Block(at as u32), pos))
    }).collect();
    let mut addrs: Vec<Value> = info.slot_addrs[slot.index()].iter().map(|&(_, _, value)| value).collect();
    addrs.extend(func.insts.iter().filter_map(|data| match data.op {
        Op::StackAddr(found, offset) if found == slot && offset != 0 => data.result,
        _ => None,
    }));
    let mut at = 0;
    while let Some(&addr) = addrs.get(at) {
        at += 1;
        if addrs[..at - 1].contains(&addr) {
            continue;
        }
        for &(block, pos, user) in &info.uses_by_val[addr.index()] {
            if moves.iter().any(|&(_, inst)| inst == user) {
                continue;
            }
            let data = &func.insts[user.index()];
            let fine = match data.op {
                Op::Binary(Binary::Iadd, ..) | Op::Convert(Convert::Bitcast, _) => {
                    addrs.extend(data.result);
                    true
                }
                Op::Jump(target, ref args) => {
                    addrs.extend(edge_params(func, target, args, addr));
                    true
                }
                Op::Brif(cond, then, ref then_args, other, ref other_args) => {
                    addrs.extend(edge_params(func, then, then_args, addr));
                    addrs.extend(edge_params(func, other, other_args, addr));
                    cond != addr
                }
                ref op => writes_through(op, addr),
            };
            if !fine {
                return None;
            }
            accesses.push((block, pos));
        }
    }
    Some(accesses)
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

//! Reuses what memory holds along straight-line code: a load of the bytes an earlier load or store on the same path read or wrote, with no write of unknown reach between them, takes that value.

mod coalesce;
mod dom;
mod pack;

use coalesce::{ScanBuf, slot_unescaped, try_coalesce_store};
use dom::{BlockWrites, record_base};

use crate::cfg::Cfg;
use crate::ir::{Binary, Block, Convert, Domain, Function, Inst, Lane, Op, ShuffleMask, StackSlot, Type, Value};
use crate::mem::{KnownLoads, MemBase, MemLoc, MemRoot, Provenance, clobbers, compute_affine, may_alias};
use crate::merge::{find, identity, rename};

enum Assembled {
    Int(i64),
    FloatBits(i64),
    VecZero,
    VecByteSplat(u8),
}

fn assemble_const(known: &KnownLoads, loc: MemLoc, ty: Type, consts: &[Option<i64>]) -> Option<Assembled> {
    let entries = known.entries_for(loc.base);
    let byte_at = |byte: i64| -> Option<u8> {
        let at = loc.offset.checked_add(byte)?;
        let (koff, imm) = entries.iter().find_map(|&(off, kty, v)| {
            let end = off.checked_add(i64::from(kty.bytes()))?;
            (kty.is_small_int() && off <= at && at < end).then(|| consts[v.index()].map(|imm| (off, imm))).flatten()
        })?;
        let shift = 8 * (at - koff);
        Some((((imm as u64) >> shift) & 0xFF) as u8)
    };
    if ty.is_vector() {
        let len = i64::from(ty.bytes());
        let first = byte_at(0)?;
        for byte in 1..len {
            if byte_at(byte)? != first {
                return None;
            }
        }
        return Some(if first == 0 { Assembled::VecZero } else { Assembled::VecByteSplat(first) });
    }
    let width = i64::from(ty.bytes());
    if !((ty.is_small_int() && matches!(width, 1 | 2 | 4 | 8)) || matches!(ty, Type::F32 | Type::F64)) {
        return None;
    }
    let mut assembled = 0_u64;
    for byte in 0..width {
        assembled |= u64::from(byte_at(byte)?) << (8 * byte);
    }
    Some(if ty.is_small_int() { Assembled::Int(assembled as i64) } else { Assembled::FloatBits(assembled as i64) })
}

/// What one instruction does to memory.
enum Effect {
    None,
    Read(MemLoc, Type, Value),
    Write(MemLoc, Type, Value),
    Unknown,
}

struct State<'a> {
    prov: &'a Provenance,
    selects: &'a [Option<(Value, Value, Value)>],
    consts: &'a mut Vec<Option<i64>>,
    subst: &'a mut Vec<Value>,
}

fn addr_loc(prov: &Provenance, subst: &[Value], addr: Value, offset: i32, size: u32) -> MemLoc {
    let loc = prov.loc(find(subst, addr), i64::from(offset), size);
    if let MemBase::Addr(base) = loc.base {
        let canonical = find(subst, base);
        if canonical != base {
            return prov.loc(canonical, loc.offset, size);
        }
    }
    loc
}

fn effect(func: &Function, state: &State<'_>, inst: Inst) -> Effect {
    let data = &func.insts[inst.index()];
    match (&data.op, data.result) {
        (&Op::Load(addr, offset), Some(result)) => {
            let ty = func.ty(result);
            Effect::Read(addr_loc(state.prov, state.subst, addr, offset, ty.bytes()), ty, result)
        }
        (&Op::StackLoad(from, offset), Some(result)) => {
            let ty = func.ty(result);
            Effect::Read(MemLoc::slot(from, i64::from(offset), ty.bytes()), ty, result)
        }
        (&Op::Store(value, addr, offset), _) => {
            let ty = func.ty(value);
            Effect::Write(addr_loc(state.prov, state.subst, addr, offset, ty.bytes()), ty, find(state.subst, value))
        }
        (&Op::StackStore(value, to, offset), _) => {
            let ty = func.ty(value);
            Effect::Write(MemLoc::slot(to, i64::from(offset), ty.bytes()), ty, find(state.subst, value))
        }
        (op, _)
            if matches!(
                op,
                Op::AtomicLoad(_)
                    | Op::AtomicStore(..)
                    | Op::AtomicRmw(..)
                    | Op::AtomicCas(..)
                    | Op::Fence
                    | Op::Copy(..)
                    | Op::Move(..)
                    | Op::Call { .. }
                    | Op::TryCall { .. }
            ) =>
        {
            Effect::Unknown
        }
        _ => Effect::None,
    }
}

fn emit_op(func: &mut Function, state: &mut State<'_>, kept: &mut Vec<Option<Inst>>, ty: Type, op: Op) -> Value {
    let val = func.value(ty);
    state.subst.push(val);
    state.consts.push(match op {
        Op::Const(imm) => Some(imm),
        _ => None,
    });
    kept.push(Some(func.add(op, Some(val))));
    val
}

fn int_of(ty: Type) -> Option<Type> {
    match ty {
        Type::F32 => Some(Type::I32),
        Type::F64 => Some(Type::I64),
        t if t.is_small_int() => Some(t),
        _ => None,
    }
}

fn forward_subword(func: &mut Function, state: &mut State<'_>, kept: &mut Vec<Option<Inst>>, known: &KnownLoads, inst: Inst, loc: MemLoc, ty: Type) -> bool {
    let Some(target_int) = int_of(ty) else { return false };
    let w = ty.bytes();
    let Some(cell_end) = loc.offset.checked_add(i64::from(w)) else { return false };
    let Some((koff, kty, kval)) = known
        .entries_for(loc.base)
        .iter()
        .filter(|&&(off, kty, _)| {
            int_of(kty).is_some()
                && off <= loc.offset
                && off.checked_add(i64::from(kty.bytes())).is_some_and(|end| cell_end <= end)
        })
        .min_by_key(|&&(off, kty, _)| (kty.bytes(), !kty.is_small_int(), loc.offset - off))
        .copied()
    else {
        return false;
    };
    let Some(k_int) = int_of(kty) else { return false };
    let shift = (loc.offset - koff) * 8;
    let need_shift = shift > 0;
    let need_reduce = w < kty.bytes();
    let bitcast_out = ty.is_float();
    let mut cur = kval;
    if kty.is_float() {
        let op = Op::Convert(Convert::Bitcast, cur);
        if !need_shift && !need_reduce && !bitcast_out {
            func.insts[inst.index()].op = op;
            return true;
        }
        cur = emit_op(func, state, kept, k_int, op);
    }
    if need_shift {
        let sh = emit_op(func, state, kept, k_int, Op::Const(shift));
        let op = Op::Binary(Binary::Ushr, cur, sh);
        if !need_reduce && !bitcast_out {
            func.insts[inst.index()].op = op;
            return true;
        }
        cur = emit_op(func, state, kept, k_int, op);
    }
    if need_reduce {
        let op = Op::Convert(Convert::Ireduce, cur);
        if !bitcast_out {
            func.insts[inst.index()].op = op;
            return true;
        }
        cur = emit_op(func, state, kept, target_int, op);
    }
    if bitcast_out {
        func.insts[inst.index()].op = Op::Convert(Convert::Bitcast, cur);
        return true;
    }
    false
}

enum ConcatHalf {
    Val(Value, bool),
    Const(i64),
}

fn resolve_concat_half(known: &KnownLoads, loc: MemLoc, offset: i64, consts: &[Option<i64>]) -> Option<ConcatHalf> {
    if let Some(v) = known.get(loc.base, offset, Type::I32) {
        return Some(ConcatHalf::Val(v, false));
    }
    if let Some(v) = known.get(loc.base, offset, Type::F32) {
        return Some(ConcatHalf::Val(v, true));
    }
    match assemble_const(known, MemLoc { offset, size: 4, ..loc }, Type::I32, consts) {
        Some(Assembled::Int(imm)) => Some(ConcatHalf::Const(imm)),
        _ => None,
    }
}

fn materialize_concat_half(func: &mut Function, state: &mut State<'_>, kept: &mut Vec<Option<Inst>>, half: ConcatHalf) -> Value {
    match half {
        ConcatHalf::Val(v, true) => emit_op(func, state, kept, Type::I32, Op::Convert(Convert::Bitcast, v)),
        ConcatHalf::Val(v, false) => v,
        ConcatHalf::Const(imm) => emit_op(func, state, kept, Type::I32, Op::Const(imm)),
    }
}

fn forward_concat(func: &mut Function, state: &mut State<'_>, kept: &mut Vec<Option<Inst>>, known: &KnownLoads, inst: Inst, loc: MemLoc, ty: Type) -> bool {
    if !matches!(ty, Type::I64 | Type::F64) {
        return false;
    }
    let Some(hi_off) = loc.offset.checked_add(4) else { return false };
    let Some(lo_half) = resolve_concat_half(known, loc, loc.offset, state.consts) else { return false };
    let Some(hi_half) = resolve_concat_half(known, loc, hi_off, state.consts) else { return false };
    let lo32 = materialize_concat_half(func, state, kept, lo_half);
    let hi32 = materialize_concat_half(func, state, kept, hi_half);
    let lo64 = emit_op(func, state, kept, Type::I64, Op::Convert(Convert::Uextend, lo32));
    let hi64 = emit_op(func, state, kept, Type::I64, Op::Convert(Convert::Uextend, hi32));
    let c32 = emit_op(func, state, kept, Type::I64, Op::Const(32));
    let hi_shifted = emit_op(func, state, kept, Type::I64, Op::Binary(Binary::Ishl, hi64, c32));
    let combined = Op::Binary(Binary::Bor, lo64, hi_shifted);
    if ty == Type::I64 {
        func.insts[inst.index()].op = combined;
    } else {
        let word = emit_op(func, state, kept, Type::I64, combined);
        func.insts[inst.index()].op = Op::Convert(Convert::Bitcast, word);
    }
    true
}

const MAX_JOIN_PHIS_PER_BLOCK: usize = 4;

enum BlockClobber {
    Loc(MemLoc),
    CopyLoc(MemLoc),
    Op(Inst),
}

struct JoinCtx<'a> {
    cfg: &'a Cfg,
    known_out: &'a [KnownLoads],
    def_blocks: &'a mut Vec<Option<Block>>,
    preds: Option<Vec<Block>>,
    added_phis: usize,
}

fn const_bits_eq(ty: Type, ca: i64, cb: i64) -> bool {
    match ty {
        t if t.is_small_int() => (ca as u64 & t.unsigned_max()) == (cb as u64 & t.unsigned_max()),
        Type::F32 => (ca as u32) == (cb as u32),
        Type::F64 => ca == cb,
        _ => false,
    }
}

fn const_eq_for_ty(consts: &[Option<i64>], ty: Type, a: Value, b: Value) -> bool {
    consts.get(a.index()).copied().flatten().zip(consts.get(b.index()).copied().flatten()).is_some_and(|(ca, cb)| const_bits_eq(ty, ca, cb))
}

fn is_silent_store(known: &KnownLoads, loc: MemLoc, ty: Type, value: Value, consts: &[Option<i64>]) -> bool {
    if let Some(prev) = known.get(loc.base, loc.offset, ty)
        && (prev == value || const_eq_for_ty(consts, ty, prev, value))
    {
        return true;
    }
    let Some(imm) = consts.get(value.index()).copied().flatten() else { return false };
    match assemble_const(known, loc, ty, consts) {
        Some(Assembled::Int(kimm) | Assembled::FloatBits(kimm)) => const_bits_eq(ty, kimm, imm),
        Some(Assembled::VecZero) => ty.is_vector() && imm == 0,
        _ => false,
    }
}

fn for_edge_args(op: &Op, target: Block, mut f: impl FnMut(&[Value])) -> bool {
    match op {
        Op::Jump(b, args) if *b == target => f(args),
        Op::Brif(_, t, ta, e, ea) => {
            if *t == target {
                f(ta);
            }
            if *e == target {
                f(ea);
            }
        }
        _ if op.targets().any(|t| t == target) => return false,
        _ => {}
    }
    true
}

fn eligible_join_preds(func: &Function, cfg: &Cfg, visited: &[bool], block: Block) -> Option<Vec<Block>> {
    if Some(block) == func.entry {
        return None;
    }
    let param_len = func.blocks[block.index()].params.len();
    let mut unique = Vec::new();
    for &pred in cfg.preds_of(block) {
        if !cfg.reachable[pred.index()] || !visited[pred.index()] {
            return None;
        }
        if !unique.contains(&pred) {
            let (mut seen, mut ok) = (false, true);
            if !func.last(pred).is_some_and(|op| for_edge_args(op, block, |a| { seen = true; ok &= a.len() == param_len; }))
                || !seen
                || !ok
            {
                return None;
            }
            unique.push(pred);
        }
    }
    (2..=4).contains(&unique.len()).then_some(unique)
}

fn for_edge_args_mut(func: &mut Function, pred: Block, target: Block, mut f: impl FnMut(&mut Vec<Value>)) {
    let Some(&last) = func.blocks[pred.index()].insts.last() else { return };
    match &mut func.insts[last.index()].op {
        Op::Jump(b, args) if *b == target => f(args),
        Op::Brif(_, t, ta, e, ea) => {
            if *t == target {
                f(ta);
            }
            if *e == target {
                f(ea);
            }
        }
        _ => {}
    }
}

fn try_join_phi_forward(
    func: &mut Function,
    (block, inst, loc, ty, result): (Block, Inst, MemLoc, Type, Value),
    state: &mut State<'_>,
    known: &mut KnownLoads,
    join: &mut JoinCtx<'_>,
    clobbers_seen: &[BlockClobber],
) -> Option<Option<Value>> {
    let preds = join.preds.as_ref()?;
    if matches!(loc.base, MemBase::Addr(b) if !join.def_blocks.get(b.index()).copied().flatten().is_some_and(|d| d != block && join.cfg.dominates(d, block)))
        || clobbers_seen.iter().any(|c| match *c {
            BlockClobber::Loc(w) => may_alias(w, loc, state.prov),
            BlockClobber::CopyLoc(w) => w.base == loc.base || (w.root != MemRoot::Unknown && w.root == loc.root) || may_alias(w, loc, state.prov),
            BlockClobber::Op(c_inst) => clobbers(func, &func.insts[c_inst.index()].op, loc, state.prov),
        })
    {
        return None;
    }
    let mut vals: Vec<(Block, Value)> = Vec::with_capacity(preds.len());
    for &p in preds {
        let v = find(state.subst, join.known_out[p.index()].get(loc.base, loc.offset, ty)?);
        if !join.def_blocks.get(v.index()).copied().flatten().is_some_and(|d| join.cfg.dominates(d, p)) {
            return None;
        }
        vals.push((p, v));
    }
    if vals.iter().all(|&(_, v)| v == vals[0].1) {
        known.remember(loc, ty, vals[0].1);
        return Some(Some(vals[0].1));
    }
    if let Some(imm0) = state.consts.get(vals[0].1.index()).copied().flatten()
        && vals.iter().all(|&(_, v)| const_eq_for_ty(state.consts, ty, vals[0].1, v))
    {
        func.insts[inst.index()].op = Op::Const(imm0);
        state.consts[result.index()] = Some(imm0);
        known.remember(loc, ty, result);
        return Some(None);
    }
    if join.added_phis >= MAX_JOIN_PHIS_PER_BLOCK {
        return None;
    }
    let phi = func.value(ty);
    func.blocks[block.index()].params.push(phi);
    state.subst.push(phi);
    state.consts.push(None);
    join.def_blocks.resize(func.types.len(), Some(block));
    join.added_phis += 1;
    vals.into_iter().for_each(|(p, v)| for_edge_args_mut(func, p, block, |a| a.push(v)));
    known.remember(loc, ty, phi);
    Some(Some(phi))
}

/// Drops from `block` each load whose bytes `known` holds, recording its replacement in `subst`, eliminates dead overwritten stores, and leaves in `known` what memory holds at the block's end; returns `(changed, renamed)`.
fn scan(
    func: &mut Function,
    block: Block,
    state: &mut State<'_>,
    known: &mut KnownLoads,
    bases: &mut Vec<(MemBase, MemRoot)>,
    join: &mut JoinCtx<'_>,
) -> (bool, bool) {
    let insts = std::mem::take(&mut func.blocks[block.index()].insts);
    let mut bufs = ScanBuf {
        kept: Vec::with_capacity(insts.len()),
        pending_stores: Vec::new(),
        const_stores: Vec::new(),
    };
    let mut clobbers_seen = Vec::new();
    let (mut changed, mut renamed) = (false, false);
    for inst in insts {
        match func.insts[inst.index()].op {
            Op::Move(_, from, bytes) => {
                let read_loc = addr_loc(state.prov, state.subst, from, 0, bytes);
                bufs.pending_stores.retain(|&(ploc, _)| !may_alias(read_loc, ploc, state.prov));
                bufs.const_stores.retain(|cs| !may_alias(read_loc, cs.loc, state.prov));
            }
            Op::Trap | Op::Trapz(_) => {
                bufs.pending_stores.retain(|&(ploc, _)| slot_unescaped(state.prov, ploc));
                bufs.const_stores.retain(|cs| slot_unescaped(state.prov, cs.loc));
            }
            Op::TryCall { .. } => {
                bufs.pending_stores.clear();
                bufs.const_stores.clear();
            }
            _ => {}
        }
        match effect(func, state, inst) {
            Effect::Read(loc, ty, result) => {
                let mut forwarded = known.get(loc.base, loc.offset, ty);
                if forwarded.is_none() {
                    if let Some(assembled) = assemble_const(known, loc, ty, state.consts) {
                        match assembled {
                            Assembled::Int(imm) => {
                                func.insts[inst.index()].op = Op::Const(imm);
                                state.consts[result.index()] = Some(imm);
                            }
                            Assembled::FloatBits(imm) => func.insts[inst.index()].op = Op::Const(imm),
                            Assembled::VecZero => func.insts[inst.index()].op = Op::Const(0),
                            Assembled::VecByteSplat(b) => {
                                let byte_ty = Type::Vec { lane: Lane::I8, count: ty.bytes() as u8 };
                                let b_val = emit_op(func, state, &mut bufs.kept, Type::I8, Op::Const(i64::from(b)));
                                let zero = emit_op(func, state, &mut bufs.kept, byte_ty, Op::Const(0));
                                let first = emit_op(func, state, &mut bufs.kept, byte_ty, Op::Insert(zero, b_val, 0));
                                let splat = Op::Shuffle(first, first, ShuffleMask::Const([0; 16]));
                                if ty == byte_ty {
                                    func.insts[inst.index()].op = splat;
                                } else {
                                    let vec = emit_op(func, state, &mut bufs.kept, byte_ty, splat);
                                    func.insts[inst.index()].op = Op::Convert(Convert::Bitcast, vec);
                                }
                            }
                        }
                        known.remember(loc, ty, result);
                        changed = true;
                    } else if forward_subword(func, state, &mut bufs.kept, known, inst, loc, ty)
                        || forward_concat(func, state, &mut bufs.kept, known, inst, loc, ty)
                    {
                        known.remember(loc, ty, result);
                        changed = true;
                    } else if let Op::Load(addr, offset) = func.insts[inst.index()].op
                        && let Some(&(c, ptr_a, ptr_b)) = state.selects.get(find(state.subst, addr).index()).and_then(Option::as_ref)
                        && let (loc_a, loc_b) = (
                            addr_loc(state.prov, state.subst, ptr_a, offset, ty.bytes()),
                            addr_loc(state.prov, state.subst, ptr_b, offset, ty.bytes()),
                        )
                        && let (Some(val_a), Some(val_b)) = (known.get(loc_a.base, loc_a.offset, ty), known.get(loc_b.base, loc_b.offset, ty))
                    {
                        func.insts[inst.index()].op = Op::Select(find(state.subst, c), val_a, val_b);
                        known.remember(loc, ty, result);
                        changed = true;
                    } else if let Some(fwd) = try_join_phi_forward(func, (block, inst, loc, ty, result), state, known, join, &clobbers_seen) {
                        forwarded = fwd;
                        changed = true;
                    } else {
                        known.remember(loc, ty, result);
                        bufs.pending_stores.retain(|&(ploc, _)| !may_alias(loc, ploc, state.prov));
                        let safe_read = slot_unescaped(state.prov, loc);
                        bufs.const_stores.retain(|cs| {
                            !may_alias(loc, cs.loc, state.prov)
                                && (slot_unescaped(state.prov, cs.loc) || safe_read || cs.loc.base == loc.base)
                        });
                    }
                    record_base(bases, loc);
                }
                if let Some(value) = forwarded {
                    state.subst[result.index()] = value;
                    if let Some(imm) = state.consts.get(value.index()).copied().flatten() {
                        state.consts[result.index()] = Some(imm);
                    }
                    renamed = true;
                    changed = true;
                    continue;
                }
            }
            Effect::Write(loc, ty, value) => {
                if is_silent_store(known, loc, ty, value, state.consts) {
                    func.insts[inst.index()].op = Op::Trap;
                    changed = true;
                    continue;
                }
                clobbers_seen.push(BlockClobber::Loc(loc));
                known.write(loc, ty, value, state.prov);
                record_base(bases, loc);
                let safe_write = slot_unescaped(state.prov, loc);
                bufs.const_stores.retain(|cs| {
                    !may_alias(loc, cs.loc, state.prov)
                        && (slot_unescaped(state.prov, cs.loc) || safe_write || cs.loc.base == loc.base)
                });
                let track = !matches!(loc.root, MemRoot::Slot(s) if !state.prov.slot_unescaped(s));
                let mut push_loc = loc;
                if track {
                    let end = i128::from(loc.offset) + i128::from(loc.size);
                    bufs.pending_stores.retain(|&(ploc, kept_idx)| {
                        let pend = i128::from(ploc.offset) + i128::from(ploc.size);
                        if ploc.base == loc.base && ploc.offset >= loc.offset && pend <= end {
                            if let Some(dead_store) = bufs.kept[kept_idx].take() {
                                func.insts[dead_store.index()].op = Op::Trap;
                                changed = true;
                            }
                            false
                        } else {
                            true
                        }
                    });
                    if let Some(wide_loc) = try_coalesce_store(func, state, known, &mut bufs, (inst, loc, ty, value)) {
                        push_loc = wide_loc;
                        changed = true;
                    }
                    bufs.pending_stores.push((push_loc, bufs.kept.len()));
                }
            }
            Effect::Unknown => {
                clobbers_seen.push(match func.insts[inst.index()].op {
                    Op::Move(dst, _, bytes) => BlockClobber::Loc(addr_loc(state.prov, state.subst, dst, 0, bytes)),
                    Op::Copy(dst, _, _) => BlockClobber::CopyLoc(addr_loc(state.prov, state.subst, dst, 0, u32::MAX)),
                    _ => BlockClobber::Op(inst),
                });
                let op = &func.insts[inst.index()].op;
                known.invalidate(func, op, state.prov);
                let keep_fn = |ploc: MemLoc| match ploc.root {
                    MemRoot::Slot(s) => state.prov.slot_unescaped(s),
                    MemRoot::NoAliasParam(p) => !clobbers(func, op, ploc, state.prov) && state.prov.noalias_unescaped(p),
                    _ => false,
                };
                bufs.pending_stores.retain(|&(ploc, _)| keep_fn(ploc));
                bufs.const_stores.retain(|cs| keep_fn(cs.loc));
            }
            Effect::None => {}
        }
        bufs.kept.push(Some(inst));
    }
    bases.retain(|&(base, _)| !known.entries_for(base).is_empty());
    func.blocks[block.index()].insts = bufs.kept.into_iter().flatten().collect();
    (changed, renamed)
}

fn eliminate_affine_block_params(func: &mut Function) -> bool {
    let Some(entry) = func.entry else { return false };
    let affine = compute_affine(func);
    let mut subst = identity(func.types.len());
    let mut any_changed = false;
    for b_idx in 0..func.blocks.len() {
        let target = Block(b_idx as u32);
        if target == entry || func.blocks[b_idx].params.len() < 2 {
            continue;
        }
        let mut incoming: Vec<Vec<Value>> = Vec::new();
        let mut non_branch_pred = false;
        for pred in &func.blocks {
            let Some(&last) = pred.insts.last() else { continue };
            if !for_edge_args(&func.insts[last.index()].op, target, |a| incoming.push(a.to_vec())) {
                non_branch_pred = true;
                break;
            }
        }
        let params = func.blocks[b_idx].params.clone();
        if non_branch_pred || incoming.is_empty() || incoming.iter().any(|args| args.len() != params.len()) {
            continue;
        }
        let mut eliminated = vec![false; params.len()];
        let mut prepended = Vec::new();
        for i in 1..params.len() {
            if func.ty(params[i]) != Type::I64 {
                continue;
            }
            for j in 0..i {
                if eliminated[j] || func.ty(params[j]) != Type::I64 {
                    continue;
                }
                let (r0_i, o0_i) = affine[incoming[0][i].index()];
                let (r0_j, o0_j) = affine[incoming[0][j].index()];
                if r0_i != r0_j {
                    continue;
                }
                let delta = o0_i.wrapping_sub(o0_j);
                let all_match = incoming.iter().skip(1).all(|args| {
                    let (ri, oi) = affine[args[i].index()];
                    let (rj, oj) = affine[args[j].index()];
                    ri == rj && oi.wrapping_sub(oj) == delta
                });
                if !all_match {
                    continue;
                }
                eliminated[i] = true;
                if delta == 0 {
                    subst[params[i].index()] = params[j];
                } else {
                    let c_val = func.value(Type::I64);
                    let c_inst = func.add(Op::Const(delta), Some(c_val));
                    let derived = func.value(Type::I64);
                    let d_inst = func.add(Op::Binary(Binary::Iadd, params[j], c_val), Some(derived));
                    subst.push(c_val);
                    subst.push(derived);
                    subst[params[i].index()] = derived;
                    prepended.push(c_inst);
                    prepended.push(d_inst);
                }
                break;
            }
        }
        if !eliminated.iter().any(|&e| e) {
            continue;
        }
        any_changed = true;
        if !prepended.is_empty() {
            prepended.append(&mut func.blocks[b_idx].insts);
            func.blocks[b_idx].insts = prepended;
        }
        let retain_mask = |args: &mut Vec<Value>| {
            let mut k = 0;
            args.retain(|_| {
                k += 1;
                !eliminated[k - 1]
            });
        };
        retain_mask(&mut func.blocks[b_idx].params);
        for pred_idx in 0..func.blocks.len() {
            for_edge_args_mut(func, Block(pred_idx as u32), target, retain_mask);
        }
    }
    if any_changed {
        rename(func, &subst);
    }
    any_changed
}

fn fits_mem_imm(ty: Type, off: i64) -> bool {
    let fits_one = |align: i64, o: i64| (-256..=255).contains(&o) || (o >= 0 && o % align == 0 && o <= align * 4095);
    if ty == Type::I128 {
        return off.checked_add(8).is_some_and(|hi| fits_one(8, off) && fits_one(8, hi));
    }
    let align = i64::from(ty.bytes().max(1));
    fits_one(align, off)
}

fn fold_affine_mem(func: &mut Function, slots: &[Option<(StackSlot, i32)>], affine: &[(Value, i64)]) -> bool {
    let mut is_stack_addr = vec![false; func.types.len()];
    for data in &func.insts {
        if let (Op::StackAddr(..), Some(res)) = (&data.op, data.result) {
            is_stack_addr[res.index()] = true;
        }
    }
    let mut changed = false;
    for &inst in func.blocks.iter().flat_map(|b| &b.insts) {
        let Some(ty) = (match func.insts[inst.index()].op {
            Op::Load(..) => func.insts[inst.index()].result.map(|r| func.ty(r)),
            Op::Store(val, ..) => Some(func.ty(val)),
            _ => None,
        }) else {
            continue;
        };
        let (Op::Load(addr, off) | Op::Store(_, addr, off)) = &mut func.insts[inst.index()].op else { continue };
        if slots[addr.index()].is_some() {
            continue;
        }
        let (root, base_off) = affine[addr.index()];
        if root != *addr
            && func.types[root.index()] == Type::I64
            && !is_stack_addr[root.index()]
            && let Some(new_off) = i64::from(*off).checked_add(base_off)
            && fits_mem_imm(ty, new_off)
        {
            *addr = root;
            *off = new_off as i32;
            changed = true;
        }
    }
    changed
}

fn for_term_args(op: &Op, mut f: impl FnMut(Value)) -> bool {
    match op {
        Op::Jump(_, args) => args.iter().for_each(|&v| f(v)),
        Op::Brif(_, _, t_args, _, e_args) => t_args.iter().chain(e_args).for_each(|&v| f(v)),
        _ => return false,
    }
    true
}

fn sink_terminator_defs(func: &mut Function) -> bool {
    let uses = func.use_counts();
    let mut fits_used = vec![false; func.types.len()];
    for inst in func.blocks.iter().flat_map(|b| &b.insts) {
        if let Op::Fits(_, _, x, y) = func.insts[inst.index()].op {
            fits_used[x.index()] = true;
            fits_used[y.index()] = true;
        }
    }
    let mut term_uses = vec![0_u32; func.types.len()];
    let mut changed = false;
    for b in 0..func.blocks.len() {
        if func.blocks[b].insts.len() < 2 {
            continue;
        }
        let Some(&last) = func.blocks[b].insts.last() else { continue };
        if !for_term_args(&func.insts[last.index()].op, |v| term_uses[v.index()] += 1) {
            continue;
        }
        let insts = std::mem::take(&mut func.blocks[b].insts);
        let len = insts.len();
        let mut kept = Vec::with_capacity(len);
        let mut sunk = Vec::new();
        for &inst in &insts[..len - 1] {
            let data = &func.insts[inst.index()];
            if let Some(res) = data.result
                && data.op.is_speculable()
                && !fits_used[res.index()]
                && term_uses[res.index()] > 0
                && uses[res.index()] == term_uses[res.index()]
            {
                sunk.push(inst);
            } else {
                kept.push(inst);
            }
        }
        if !sunk.is_empty() && sunk.iter().enumerate().any(|(idx, &inst)| insts[len - 1 - sunk.len() + idx] != inst) {
            changed = true;
        }
        kept.extend(sunk);
        kept.push(last);
        func.blocks[b].insts = kept;
        for_term_args(&func.insts[last.index()].op, |v| term_uses[v.index()] = 0);
    }
    changed
}

/// Replaces each load of bytes that an earlier access on every path to it read or wrote with no write of unknown reach between; returns the effect domains modified.
pub(crate) fn run(func: &mut Function) -> Domain {
    if func.entry.is_none() || func.coroutine.is_some() {
        return Domain::NONE;
    }
    let mut changed = eliminate_affine_block_params(func);
    let cfg = func.cfg();
    let prov = Provenance::new(func, &cfg);
    changed |= fold_affine_mem(func, &prov.slot_offsets, &prov.affine);
    changed |= pack::coalesce_loads(func, &prov);
    changed |= sink_terminator_defs(func);
    let mut consts = vec![None; func.types.len()];
    let mut selects = vec![None; func.types.len()];
    for data in &func.insts {
        match (&data.op, data.result) {
            (&Op::Const(imm), Some(result)) => consts[result.index()] = Some(imm),
            (&Op::Select(c, a, b), Some(result)) => selects[result.index()] = Some((c, a, b)),
            _ => {}
        }
    }
    let mut bw = BlockWrites::new(func, &cfg, &prov);
    let mut subst = identity(func.types.len());
    let mut state = State { prov: &prov, selects: &selects, consts: &mut consts, subst: &mut subst };
    let mut renamed = false;
    let mut def_blocks = func.def_blocks();
    let mut visited = vec![false; func.blocks.len()];
    let mut known_out: Vec<KnownLoads> = (0..func.blocks.len()).map(|_| KnownLoads::new()).collect();
    let mut block_bases: Vec<Vec<(MemBase, MemRoot)>> = vec![Vec::new(); func.blocks.len()];
    for &block in &cfg.rpo {
        let (mut known, mut bases) = bw.entry_known(func, state.subst, &known_out, &block_bases, block);
        let preds = eligible_join_preds(func, &cfg, &visited, block);
        let mut join = JoinCtx { cfg: &cfg, known_out: &known_out, def_blocks: &mut def_blocks, preds, added_phis: 0 };
        let (block_changed, block_renamed) = scan(func, block, &mut state, &mut known, &mut bases, &mut join);
        def_blocks.resize(func.types.len(), Some(block));
        visited[block.index()] = true;
        changed |= block_changed;
        renamed |= block_renamed;
        known_out[block.index()] = known;
        block_bases[block.index()] = bases;
    }
    if renamed {
        rename(func, state.subst);
    }
    if changed { Domain::MEM | Domain::VALUES } else { Domain::NONE }
}

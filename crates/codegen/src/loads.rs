//! Reuses what memory holds along straight-line code: a load of the bytes an earlier load or store on the same path read or wrote, with no write of unknown reach between them, takes that value.

mod coalesce;
mod dom;
mod join;
mod pack;

use coalesce::{ScanBuf, slot_unescaped, try_coalesce_store};
use dom::{BlockWrites, invalidate_preserving, record_base, write_preserving};
use join::{BlockClobber, JoinCtx, SubwordPlan, const_eq_for_ty, eligible_join_preds, find_subword_source, for_edge_args, for_edge_args_mut, resolve_select_load, rodata_const, try_join_phi_forward};

use crate::ir::{Binary, Block, Convert, Domain, Function, Inst, Lane, Op, ShuffleMask, StackSlot, TargetOpId, TinyArgs, Type, Value};
use crate::mem::{KnownLoads, MemBase, MemLoc, MemRoot, Provenance, compute_affine, may_alias};
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
        if ty.bytes() > 16 && first != 0 {
            return None;
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
    target: &'a crate::Target,
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

fn forward_subword(func: &mut Function, state: &mut State<'_>, kept: &mut Vec<Option<Inst>>, known: &KnownLoads, inst: Inst, loc: MemLoc, ty: Type) -> bool {
    let Some((koff, kty, kval)) = find_subword_source(known, loc, ty) else {
        return false;
    };
    let Some(plan) = SubwordPlan::new(loc.offset, ty, koff, kty, kval) else {
        return false;
    };
    func.insts[inst.index()].op = plan.build_op(|op_ty, op| emit_op(func, state, kept, op_ty, op));
    true
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

enum Word64Plan {
    Val(Value),
    Bitcast(Value),
    Const(i64),
    Pair32(ConcatHalf, ConcatHalf),
}

fn plan_word64(func: &Function, known: &KnownLoads, loc: MemLoc, offset: i64, consts: &[Option<i64>]) -> Option<Word64Plan> {
    if let Some(v) = known.get(loc.base, offset, Type::I64) {
        return Some(Word64Plan::Val(v));
    }
    if let Some(v) = known.get(loc.base, offset, Type::F64) {
        return Some(Word64Plan::Bitcast(v));
    }
    if let Some(&(_, _, v)) = known.entries_for(loc.base).iter().find(|&&(off, kty, _)| off == offset && kty.is_vector() && kty.bytes() == 8) {
        return Some(Word64Plan::Bitcast(v));
    }
    let slice = MemLoc { offset, size: 8, ..loc };
    if let Some(imm) = rodata_const(func, slice, Type::I64) {
        return Some(Word64Plan::Const(imm));
    }
    if let Some(Assembled::Int(imm)) = assemble_const(known, slice, Type::I64, consts) {
        return Some(Word64Plan::Const(imm));
    }
    let hi_off = offset.checked_add(4)?;
    let lo_half = resolve_concat_half(known, loc, offset, consts)?;
    let hi_half = resolve_concat_half(known, loc, hi_off, consts)?;
    Some(Word64Plan::Pair32(lo_half, hi_half))
}

fn emit_pack32_op(func: &mut Function, state: &mut State<'_>, kept: &mut Vec<Option<Inst>>, lo_half: ConcatHalf, hi_half: ConcatHalf) -> Op {
    let lo32 = materialize_concat_half(func, state, kept, lo_half);
    let hi32 = materialize_concat_half(func, state, kept, hi_half);
    let lo64 = emit_op(func, state, kept, Type::I64, Op::Convert(Convert::Uextend, lo32));
    let hi64 = emit_op(func, state, kept, Type::I64, Op::Convert(Convert::Uextend, hi32));
    let c32 = emit_op(func, state, kept, Type::I64, Op::Const(32));
    let hi_shifted = emit_op(func, state, kept, Type::I64, Op::Binary(Binary::Ishl, hi64, c32));
    Op::Binary(Binary::Bor, lo64, hi_shifted)
}

fn materialize_word64(func: &mut Function, state: &mut State<'_>, kept: &mut Vec<Option<Inst>>, plan: Word64Plan) -> Value {
    match plan {
        Word64Plan::Val(v) => find(state.subst, v),
        Word64Plan::Bitcast(v) => emit_op(func, state, kept, Type::I64, Op::Convert(Convert::Bitcast, find(state.subst, v))),
        Word64Plan::Const(imm) => emit_op(func, state, kept, Type::I64, Op::Const(imm)),
        Word64Plan::Pair32(lo, hi) => {
            let op = emit_pack32_op(func, state, kept, lo, hi);
            emit_op(func, state, kept, Type::I64, op)
        }
    }
}

const QUAD: Type = Type::Vec { lane: Lane::I64, count: 2 };
const WIDE_QUADS: Type = Type::Vec { lane: Lane::I64, count: 4 };

fn quad_op(func: &mut Function, state: &mut State<'_>, kept: &mut Vec<Option<Inst>>, (lo, hi): (Word64Plan, Word64Plan)) -> Op {
    let lo64 = materialize_word64(func, state, kept, lo);
    let hi64 = materialize_word64(func, state, kept, hi);
    let zero = emit_op(func, state, kept, QUAD, Op::Const(0));
    let half = emit_op(func, state, kept, QUAD, Op::Insert(zero, lo64, 0));
    Op::Insert(half, hi64, 1)
}

fn set_load(func: &mut Function, state: &mut State<'_>, kept: &mut Vec<Option<Inst>>, (inst, ty): (Inst, Type), (built, op): (Type, Op)) {
    if ty == built {
        func.insts[inst.index()].op = op;
    } else {
        let value = emit_op(func, state, kept, built, op);
        func.insts[inst.index()].op = Op::Convert(Convert::Bitcast, value);
    }
}

fn forward_concat(func: &mut Function, state: &mut State<'_>, kept: &mut Vec<Option<Inst>>, known: &KnownLoads, inst: Inst, loc: MemLoc, ty: Type) -> bool {
    if ty.is_vector() && (ty.bytes() == 16 || (ty.bytes() == 32 && ty.is_register_vec(state.target))) {
        let Some(plans) = (0..i64::from(ty.bytes()) / 8).map(|word| plan_word64(func, known, loc, loc.offset.checked_add(8 * word)?, state.consts)).collect::<Option<Vec<_>>>() else { return false };
        let mut plans = plans.into_iter();
        let (Some(lo), Some(hi)) = (plans.next(), plans.next()) else { return false };
        let low = quad_op(func, state, kept, (lo, hi));
        let built = match (plans.next(), plans.next()) {
            (Some(lo), Some(hi)) => {
                let low = emit_op(func, state, kept, QUAD, low);
                let high = quad_op(func, state, kept, (lo, hi));
                let high = emit_op(func, state, kept, QUAD, high);
                (WIDE_QUADS, Op::TargetOp(TargetOpId::X64Concat128, TinyArgs::two(low, high)))
            }
            _ => (QUAD, low),
        };
        set_load(func, state, kept, (inst, ty), built);
        return true;
    }
    if !(matches!(ty, Type::I64 | Type::F64) || (ty.is_vector() && ty.bytes() == 8)) {
        return false;
    }
    let Some(hi_off) = loc.offset.checked_add(4) else { return false };
    let Some(lo_half) = resolve_concat_half(known, loc, loc.offset, state.consts) else { return false };
    let Some(hi_half) = resolve_concat_half(known, loc, hi_off, state.consts) else { return false };
    let combined = emit_pack32_op(func, state, kept, lo_half, hi_half);
    set_load(func, state, kept, (inst, ty), (Type::I64, combined));
    true
}

fn copy_len(op: &Op, consts: &[Option<i64>]) -> Option<(Value, Value, u32)> {
    match *op {
        Op::Move(dst, src, bytes) => Some((dst, src, bytes)),
        Op::Copy(dst, src, len) => consts.get(len.index()).copied().flatten().and_then(|n| u32::try_from(n).ok()).map(|b| (dst, src, b)),
        _ => None,
    }
}

fn is_silent_store(known: &KnownLoads, loc: MemLoc, ty: Type, value: Value, consts: &[Option<i64>]) -> bool {
    if let Some(prev) = known.get(loc.base, loc.offset, ty)
        && (prev == value || const_eq_for_ty(consts, ty, prev, value))
    {
        return true;
    }
    let Some(imm) = consts.get(value.index()).copied().flatten() else { return false };
    match assemble_const(known, loc, ty, consts) {
        Some(Assembled::Int(kimm) | Assembled::FloatBits(kimm)) => join::const_bits_eq(ty, kimm, imm),
        Some(Assembled::VecZero) => ty.is_vector() && imm == 0,
        _ => false,
    }
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
        if let Some((_, from, bytes)) = copy_len(&func.insts[inst.index()].op, state.consts) {
            if bytes > 0 {
                let read_loc = addr_loc(state.prov, state.subst, from, 0, bytes);
                bufs.pending_stores.retain(|&(ploc, _)| !may_alias(read_loc, ploc, state.prov));
                bufs.const_stores.retain(|cs| !may_alias(read_loc, cs.loc, state.prov));
            }
        } else {
            match func.insts[inst.index()].op {
                Op::Copy(_, from, _) => {
                    let r = addr_loc(state.prov, state.subst, from, 0, u32::MAX);
                    let ok = |l: MemLoc| l.base != r.base && (l.root == MemRoot::Unknown || l.root != r.root) && !may_alias(r, l, state.prov);
                    bufs.pending_stores.retain(|&(p, _)| ok(p));
                    bufs.const_stores.retain(|cs| ok(cs.loc));
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
        }
        match effect(func, state, inst) {
            Effect::Read(loc, ty, result) => {
                let mut forwarded = known.get(loc.base, loc.offset, ty);
                if forwarded.is_none() {
                    if let Some(imm) = rodata_const(func, loc, ty) {
                        func.insts[inst.index()].op = Op::Const(imm);
                        state.consts[result.index()] = Some(imm);
                        known.remember(loc, ty, result);
                        changed = true;
                    } else if let Some(assembled) = assemble_const(known, loc, ty, state.consts) {
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
                        && ty != Type::I128
                        && let Some(&(c, ptr_a, ptr_b)) = state.selects.get(find(state.subst, addr).index()).and_then(Option::as_ref)
                        && !func.ty(c).is_vector()
                        && let (loc_a, loc_b) = (
                            addr_loc(state.prov, state.subst, ptr_a, offset, ty.bytes()),
                            addr_loc(state.prov, state.subst, ptr_b, offset, ty.bytes()),
                        )
                        && let Some((val_a, val_b)) = resolve_select_load(func, state, &mut bufs.kept, known, ty, (loc_a, loc_b))
                    {
                        if val_a == val_b {
                            forwarded = Some(val_a);
                        } else {
                            func.insts[inst.index()].op = Op::Select(find(state.subst, c), val_a, val_b);
                            known.remember(loc, ty, result);
                        }
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
                write_preserving(func, known, bases, loc, ty, value, state.prov);
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
                    if let Some(wide_loc) = try_coalesce_store(func, state, known, bases, &mut bufs, (inst, loc, ty, value)) {
                        push_loc = wide_loc;
                        changed = true;
                    }
                    bufs.pending_stores.push((push_loc, bufs.kept.len()));
                }
            }
            Effect::Unknown => {
                let mut copy_remaps: Option<(MemLoc, Vec<(i64, Type, Value)>)> = None;
                let canon_move = copy_len(&func.insts[inst.index()].op, state.consts).map(|(dst, src, bytes)| {
                    let dst_loc = addr_loc(state.prov, state.subst, dst, 0, bytes);
                    if bytes > 0 {
                        clobbers_seen.push(BlockClobber::Loc(dst_loc));
                        let src_loc = addr_loc(state.prov, state.subst, src, 0, bytes);
                        if !may_alias(dst_loc, src_loc, state.prov)
                            && let Some(src_end) = src_loc.offset.checked_add(i64::from(bytes))
                        {
                            let copied = known
                                .entries_for(src_loc.base)
                                .iter()
                                .filter_map(|&(off, kty, kval)| {
                                    (off >= src_loc.offset && off.checked_add(i64::from(kty.bytes())).is_some_and(|e| e <= src_end))
                                        .then(|| dst_loc.offset.checked_add(off - src_loc.offset).map(|d_off| (d_off, kty, find(state.subst, kval))))
                                        .flatten()
                                })
                                .collect::<Vec<_>>();
                            if !copied.is_empty() {
                                copy_remaps = Some((dst_loc, copied));
                            }
                        }
                    }
                    (bytes, Op::Move(find(state.subst, dst), find(state.subst, src), bytes))
                });
                if let Some((0, _)) = canon_move {
                    bufs.kept.push(Some(inst));
                    continue;
                }
                let clobber_op = match canon_move {
                    Some((_, ref m)) => m,
                    None => {
                        clobbers_seen.push(match func.insts[inst.index()].op {
                            Op::Copy(dst, _, _) => BlockClobber::CopyLoc(addr_loc(state.prov, state.subst, dst, 0, u32::MAX)),
                            _ => BlockClobber::Op(inst),
                        });
                        &func.insts[inst.index()].op
                    }
                };
                invalidate_preserving(func, known, bases, clobber_op, state.prov);
                if let Some((bytes, _)) = canon_move
                    && let Some((dst_loc, copied)) = copy_remaps
                    && let Some(dst_end) = dst_loc.offset.checked_add(i64::from(bytes))
                    && known.entries_for(dst_loc.base).iter().all(|&(o, t, _)| {
                        o.checked_add(i64::from(t.bytes())).is_some_and(|e| e <= dst_loc.offset) || o >= dst_end
                    })
                {
                    for (dst_off, kty, kval) in copied {
                        known.remember(MemLoc::new(dst_loc.root, dst_loc.base, dst_off, kty.bytes()), kty, kval);
                    }
                    record_base(bases, dst_loc);
                }
                let keep_fn = |ploc: MemLoc| match ploc.root {
                    MemRoot::Slot(s) => state.prov.slot_unescaped(s),
                    MemRoot::NoAliasParam(p) => !dom::clobbers_loc(func, clobber_op, ploc, state.prov) && state.prov.noalias_unescaped(p),
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

fn eliminate_affine_block_params(func: &mut Function) -> (bool, Option<Vec<(Value, i64)>>) {
    let Some(entry) = func.entry else { return (false, None) };
    let mergeable = |func: &Function, b_idx: usize| Block(b_idx as u32) != entry && func.blocks[b_idx].params.len() >= 2;
    if !(0..func.blocks.len()).any(|b_idx| mergeable(func, b_idx)) {
        return (false, None);
    }
    let affine = compute_affine(func);
    let mut subst = identity(func.types.len());
    let mut any_changed = false;
    for b_idx in 0..func.blocks.len() {
        let target = Block(b_idx as u32);
        if !mergeable(func, b_idx) {
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
    (any_changed, (!any_changed).then_some(affine))
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
pub(crate) fn run(func: &mut Function, target: &crate::Target) -> Domain {
    if func.entry.is_none() || func.coroutine.is_some() {
        return Domain::NONE;
    }
    let domain = |changed: bool| if changed { Domain::MEM | Domain::VALUES } else { Domain::NONE };
    let (mut changed, affine) = eliminate_affine_block_params(func);
    let is_access = |op: &Op| matches!(op, Op::Load(..) | Op::StackLoad(..) | Op::Store(..) | Op::StackStore(..));
    if !func.insts.iter().any(|data| is_access(&data.op)) {
        changed |= sink_terminator_defs(func);
        return domain(changed);
    }
    let cfg = func.cfg();
    let prov = Provenance::with_affine(func, &cfg, affine);
    changed |= fold_affine_mem(func, &prov.slot_offsets, &prov.affine);
    changed |= pack::coalesce_loads(func, &prov, target);
    changed |= sink_terminator_defs(func);
    let mut consts = vec![None; func.types.len()];
    let mut selects = vec![None; func.types.len()];
    let mut accesses = false;
    for data in &func.insts {
        match (&data.op, data.result) {
            (&Op::Const(imm), Some(result)) => consts[result.index()] = Some(imm),
            (&Op::Select(c, a, b), Some(result)) => selects[result.index()] = Some((c, a, b)),
            (op, _) if is_access(op) => accesses = true,
            _ => {}
        }
    }
    if !accesses {
        return domain(changed);
    }
    let mut bw = BlockWrites::new(func, &cfg, &prov);
    let mut subst = identity(func.types.len());
    let mut state = State { prov: &prov, selects: &selects, consts: &mut consts, subst: &mut subst, target };
    let (mut renamed, mut scan_changed) = (false, false);
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
        scan_changed |= block_changed;
        changed |= block_changed;
        renamed |= block_renamed;
        known_out[block.index()] = known;
        block_bases[block.index()] = bases;
    }
    if renamed {
        rename(func, state.subst);
    }
    if scan_changed {
        changed |= pack::coalesce_vec_rescan(func, &cfg, target);
    }
    domain(changed)
}

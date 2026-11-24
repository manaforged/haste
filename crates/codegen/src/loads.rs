//! Reuses what memory holds along straight-line code: a load of the bytes an earlier load or store on the same path read or wrote, with no write of unknown reach between them, takes that value.

mod caller_loc;
mod caller_loc_sink;
mod coalesce;
mod dom;
mod elide;
mod join;
mod pack;
mod regroup;
mod remat;
mod sink;

use coalesce::{ScanBuf, slot_unescaped, try_coalesce_store};
use dom::{BlockWrites, invalidate_preserving, record_base, write_preserving};
use join::{BlockClobber, JoinCtx, const_eq_for_ty, eligible_join_preds, for_edge_args, resolve_select_load, rodata_const, try_join_phi_forward};
use remat::{FarCseGuard, SubwordAction, SubwordSlices, drop_covered_stores, drop_stores_if, resolve_subword_plan, sink_terminator_defs};

use crate::ir::{Binary, Block, Convert, Domain, Function, Inst, Lane, Op, ShuffleMask, StackSlot, TargetOpId, TinyArgs, Type, Value};
use crate::mem::{KnownLoads, MemBase, MemLoc, MemRoot, Provenance, compute_affine, may_alias, may_alias_at};
use crate::merge::{find, identity, rename};
use crate::reloc::RuntimeSymbol;

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
    defs: &'a [Option<Inst>],
    selects: &'a [Option<(Value, Value, Value)>],
    consts: &'a mut Vec<Option<i64>>,
    subst: &'a mut Vec<Value>,
    target: &'a crate::Target,
    post_inline: bool,
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
        (&Op::Call { callee, .. }, _)
            if state.prov.frame_runtime(callee) == Some(RuntimeSymbol::Malloc)
                || func.is_fresh_alloc(callee)
                || func.pure_thread_cell_result(&data.op, |v| state.consts[v.index()] == Some(0)).is_some() => Effect::None,
        (op, _) if crate::mem::is_atomic_barrier(op) => Effect::Unknown,
        (op, _) if matches!(op, Op::Copy(..) | Op::Move(..) | Op::Call { .. } | Op::TryCall { .. }) => Effect::Unknown,
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

fn forward_subword(
    func: &mut Function,
    state: &mut State<'_>,
    kept: &mut Vec<Option<Inst>>,
    ctx: (&KnownLoads, &SubwordSlices, &FarCseGuard),
    (inst, loc, ty, result): (Inst, MemLoc, Type, Value),
) -> Option<Option<Value>> {
    let action = resolve_subword_plan(func, (inst, loc, ty), ctx, (state.post_inline, state.defs, state.subst, state.consts))?;
    match action {
        SubwordAction::Forwarded(val) => Some(Some(val)),
        SubwordAction::Const(imm) => {
            func.insts[inst.index()].op = Op::Const(imm);
            if ty.is_small_int() {
                state.consts[result.index()] = Some(imm);
            }
            Some(None)
        }
        SubwordAction::Plan(plan) => {
            func.insts[inst.index()].op = plan.build_op(|op_ty, op| emit_op(func, state, kept, op_ty, op));
            Some(None)
        }
    }
}

enum ConcatHalf {
    Val(Value, bool),
    Const(i64),
}

fn resolve_concat_half(known: &KnownLoads, loc: MemLoc, offset: i64, consts: &[Option<i64>]) -> Option<ConcatHalf> {
    if let Some(v) = known.get(loc.base, offset, Type::I32) { return Some(ConcatHalf::Val(v, false)); }
    if let Some(v) = known.get(loc.base, offset, Type::F32) { return Some(ConcatHalf::Val(v, true)); }
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

enum Word64Plan { Val(Value), Bitcast(Value), Const(i64), Pair32(ConcatHalf, ConcatHalf) }

fn plan_word64(func: &Function, known: &KnownLoads, loc: MemLoc, offset: i64, consts: &[Option<i64>]) -> Option<Word64Plan> {
    if let Some(v) = known.get(loc.base, offset, Type::I64) { return Some(Word64Plan::Val(v)); }
    if let Some(v) = known.get(loc.base, offset, Type::F64) { return Some(Word64Plan::Bitcast(v)); }
    if let Some(&(_, _, v)) = known.entries_for(loc.base).iter().find(|&&(off, kty, _)| off == offset && kty.is_vector() && kty.bytes() == 8) {
        return Some(Word64Plan::Bitcast(v));
    }
    let slice = MemLoc { offset, size: 8, ..loc };
    if let Some(imm) = rodata_const(func, slice, Type::I64) { return Some(Word64Plan::Const(imm)); }
    if let Some(Assembled::Int(imm)) = assemble_const(known, slice, Type::I64, consts) { return Some(Word64Plan::Const(imm)); }
    let hi_off = offset.checked_add(4)?;
    Some(Word64Plan::Pair32(resolve_concat_half(known, loc, offset, consts)?, resolve_concat_half(known, loc, hi_off, consts)?))
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
    if ty == built { func.insts[inst.index()].op = op; } else {
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

fn call_passes_noalias(op: &Op, prov: &Provenance, p: u16) -> bool {
    match *op {
        Op::Call { ref args, .. } | Op::TryCall { ref args, .. } => args
            .iter()
            .any(|arg| prov.param_roots.get(arg.index()).copied() == Some(MemRoot::NoAliasParam(p))),
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
    (bufs, clobbers_seen, guard): (&mut ScanBuf, &mut Vec<BlockClobber>, &mut FarCseGuard),
) -> (bool, bool) {
    let mut insts = std::mem::take(&mut func.blocks[block.index()].insts);
    bufs.kept.clear();
    bufs.pending_stores.clear();
    bufs.const_stores.clear();
    clobbers_seen.clear();
    let (mut changed, mut renamed) = (false, false);
    guard.reset(insts.len(), func.types.len());
    let mut slices = SubwordSlices::new();
    for &inst in &insts {
        let kept_before = bufs.kept.len();
        if coalesce::is_pure_scan_op(&func.insts[inst.index()].op) { bufs.kept.push(Some(inst)); guard.sync_kept(func, state.subst, &bufs.kept[kept_before..]); continue; }
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
                Op::Trap | Op::Trapz(_) | Op::Assume(_) => {
                    bufs.pending_stores.retain(|&(ploc, _)| slot_unescaped(state.prov, ploc));
                    bufs.const_stores.retain(|cs| slot_unescaped(state.prov, cs.loc));
                }
                Op::TryCall { .. } => {
                    bufs.pending_stores.clear();
                    bufs.const_stores.clear();
                }
                Op::Return(..) if state.post_inline => {
                    changed |= drop_stores_if(func, bufs, guard, |ploc| slot_unescaped(state.prov, ploc));
                }
                _ => {}
            }
            changed |= remat::drop_freed_stores(func, state, (bufs, guard), inst);
        }
        match effect(func, state, inst) {
            Effect::Read(loc, ty, result) => {
                let (mut forwarded, skip_far) = guard.check_cse(&func.insts[inst.index()].op, ty, known.get(loc.base, loc.offset, ty));
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
                    } else if !skip_far
                        && let Some(sub_fwd) = forward_subword(func, state, &mut bufs.kept, (known, &slices, &guard), (inst, loc, ty, result))
                    {
                        if let Some(val) = sub_fwd {
                            forwarded = Some(val);
                        } else {
                            known.remember(loc, ty, result);
                        }
                        changed = true;
                    } else if !skip_far && forward_concat(func, state, &mut bufs.kept, known, inst, loc, ty) {
                        known.remember(loc, ty, result);
                        changed = true;
                    } else if !skip_far
                        && let Op::Load(addr, offset) = func.insts[inst.index()].op
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
                    } else if !skip_far
                        && let Some(fwd) = try_join_phi_forward(func, (block, inst, loc, ty, result), state, known, join, (&clobbers_seen, &bufs.kept))
                    {
                        forwarded = fwd;
                        changed = true;
                    } else {
                        known.remember(loc, ty, result);
                        bufs.pending_stores.retain(|&(ploc, _)| !may_alias_at(loc, ploc, state.prov, Some(inst)));
                        let safe_read = slot_unescaped(state.prov, loc);
                        bufs.const_stores.retain(|cs| {
                            !may_alias_at(loc, cs.loc, state.prov, Some(inst))
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
                    guard.sync_kept(func, state.subst, &bufs.kept[kept_before..]);
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
                if state.post_inline {
                    slices.prune_on_write(func, state.prov, loc, Some(inst));
                    slices.save_split_slices(known, state.subst, loc);
                }
                write_preserving(func, known, bases, loc, ty, value, state.prov, Some(inst));
                record_base(bases, loc);
                let track = state.post_inline || !matches!(loc.root, MemRoot::Slot(s) if !state.prov.slot_unescaped(s));
                if track && !state.post_inline {
                    changed |= drop_covered_stores(func, bufs, guard, (loc, false));
                }
                let mut push_loc = loc;
                if let Some(wide_loc) = try_coalesce_store(func, state, known, bases, bufs, (inst, loc, ty, value, track)) {
                    push_loc = wide_loc;
                    changed = true;
                }
                if track {
                    if state.post_inline {
                        changed |= drop_covered_stores(func, bufs, guard, (push_loc, true));
                    }
                    bufs.pending_stores.push((push_loc, bufs.kept.len()));
                }
            }
            Effect::Unknown => {
                if let Some((wloc, ..)) = elide::try_fold_private_rmw(func, (join.cfg, state.prov, block, inst), (state.consts, state.subst), (known, &mut bufs.kept)) {
                    clobbers_seen.push(BlockClobber::Loc(wloc));
                    record_base(bases, wloc);
                    if state.post_inline {
                        slices.prune_on_write(func, state.prov, wloc, Some(inst));
                        drop_covered_stores(func, bufs, guard, (wloc, true));
                    }
                    bufs.pending_stores.retain(|&(ploc, _)| !may_alias_at(wloc, ploc, state.prov, Some(inst)));
                    bufs.const_stores.retain(|cs| !may_alias_at(wloc, cs.loc, state.prov, Some(inst)));
                    changed = true;
                    guard.sync_kept(func, state.subst, &bufs.kept[kept_before..]);
                    continue;
                }
                if state.post_inline
                    && let Some(op) = copy_len(&func.insts[inst.index()].op, state.consts)
                    && let Some(stores) = elide::try_scalarize_slot_move(func, (join.cfg, state.prov, block, inst), op, state.subst, known)
                {
                    for (wloc, store) in stores {
                        clobbers_seen.push(BlockClobber::Loc(wloc));
                        record_base(bases, wloc);
                        slices.prune_on_write(func, state.prov, wloc, Some(inst));
                        bufs.const_stores.retain(|cs| !may_alias_at(wloc, cs.loc, state.prov, Some(inst)));
                        bufs.pending_stores.push((wloc, bufs.kept.len()));
                        bufs.kept.push(Some(store));
                    }
                    changed = true;
                    guard.sync_kept(func, state.subst, &bufs.kept[kept_before..]);
                    continue;
                }
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
                    (bytes, dst_loc, Op::Move(find(state.subst, dst), find(state.subst, src), bytes))
                });
                if let Some((0, ..)) = canon_move {
                    bufs.kept.push(Some(inst));
                    guard.sync_kept(func, state.subst, &bufs.kept[kept_before..]);
                    continue;
                }
                let clobber_op = match canon_move {
                    Some((_, _, ref m)) => m,
                    None => {
                        clobbers_seen.push(match func.insts[inst.index()].op {
                            Op::Copy(dst, _, _) => BlockClobber::CopyLoc(addr_loc(state.prov, state.subst, dst, 0, u32::MAX)),
                            _ => BlockClobber::Op(inst),
                        });
                        &func.insts[inst.index()].op
                    }
                };
                if state.post_inline {
                    slices.on_unknown(func, clobber_op, state.prov);
                }
                invalidate_preserving(func, known, bases, (clobber_op, Some(inst)), state.prov);
                if let Some((bytes, dst_loc, _)) = canon_move {
                    if state.post_inline && bytes > 0 {
                        changed |= drop_covered_stores(func, bufs, guard, (dst_loc, true));
                    }
                    if let Some((dst_loc, copied)) = copy_remaps
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
                }
                let keep_fn = |ploc: MemLoc| match ploc.root {
                    MemRoot::Slot(s) => state.prov.slot_unescaped(s),
                    MemRoot::CallerLoc => !matches!(canon_move.as_ref().map_or(&func.insts[inst.index()].op, |(_, _, m)| m), Op::Call { .. } | Op::TryCall { .. }),
                    MemRoot::NoAliasParam(p) => {
                        let cop = canon_move.as_ref().map_or(&func.insts[inst.index()].op, |(_, _, m)| m);
                        !call_passes_noalias(cop, state.prov, p)
                            && !dom::clobbers_loc(func, (cop, Some(inst)), ploc, state.prov)
                            && state.prov.noalias_unescaped(p)
                    }
                    _ => false,
                };
                bufs.pending_stores.retain(|&(ploc, _)| keep_fn(ploc));
                bufs.const_stores.retain(|cs| keep_fn(cs.loc));
            }
            Effect::None => {}
        }
        bufs.kept.push(Some(inst));
        guard.sync_kept(func, state.subst, &bufs.kept[kept_before..]);
    }
    bases.retain(|&(base, _)| !known.entries_for(base).is_empty());
    if changed { insts.clear(); insts.extend(bufs.kept.drain(..).flatten()); }
    func.blocks[block.index()].insts = insts;
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
            func.for_edge_args_mut(Block(pred_idx as u32), target, retain_mask);
        }
    }
    if any_changed {
        rename(func, &subst);
    }
    (any_changed, (!any_changed).then_some(affine))
}

fn fits_mem_imm(ty: Type, off: i64) -> bool {
    let fits_one = |align: i64, o: i64| (-256..=255).contains(&o) || (o >= 0 && o % align == 0 && o <= align * 4095);
    if ty == Type::I128 { return off.checked_add(8).is_some_and(|hi| fits_one(8, off) && fits_one(8, hi)); }
    fits_one(i64::from(ty.bytes().max(1)), off)
}

fn fold_affine_mem(func: &mut Function, slots: &[Option<(StackSlot, i32)>], affine: &[(Value, i64)]) -> bool {
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
        if let Some((slot, base_off)) = slots[addr.index()] {
            if let Some(total) = i64::from(base_off).checked_add(i64::from(*off))
                && crate::mem::in_range(total, ty.bytes(), func.slots[slot.index()].size)
                && let Ok(disp) = i32::try_from(total)
            {
                func.insts[inst.index()].op = match func.insts[inst.index()].op { Op::Load(..) => Op::StackLoad(slot, disp), Op::Store(v, ..) => Op::StackStore(v, slot, disp), _ => unreachable!() };
                changed = true;
            }
            continue;
        }
        let (root, base_off) = affine[addr.index()];
        if root != *addr
            && func.types[root.index()] == Type::I64
            && slots[root.index()].is_none()
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

fn collect_scan_meta(func: &Function) -> Option<(Vec<Option<Inst>>, Vec<Option<i64>>, Vec<Option<(Value, Value, Value)>>)> {
    let n = func.types.len();
    let (mut defs, mut consts, mut selects, mut accesses) = (vec![None; n], vec![None; n], Vec::new(), false);
    for &inst in func.blocks.iter().flat_map(|block| &block.insts) {
        let data = &func.insts[inst.index()];
        if let Some(result) = data.result { defs[result.index()] = Some(inst); }
        match (&data.op, data.result) {
            (&Op::Const(imm), Some(result)) => consts[result.index()] = Some(imm),
            (&Op::Select(c, a, b), Some(result)) => { if selects.is_empty() { selects.resize(n, None); } selects[result.index()] = Some((c, a, b)); }
            (op, _) if matches!(op, Op::Load(..) | Op::StackLoad(..) | Op::Store(..) | Op::StackStore(..)) => accesses = true,
            _ => {}
        }
    }
    accesses.then_some((defs, consts, selects))
}

/// Elides whole-slot moves once, before promotion splits the slots into values.
pub(crate) fn elide_early(func: &mut Function) -> bool {
    if func.entry.is_none() || func.coroutine.is_some() || !crate::schedule::has_live_op(func, |op| matches!(op, Op::Move(..))) {
        return false;
    }
    let cfg = func.cfg();
    elide::elide_moves(func, &Provenance::new(func, &cfg))
}

/// Replaces each load of bytes that an earlier access on every path to it read or wrote with no write of unknown reach between; returns the effect domains modified.
pub(crate) fn run(func: &mut Function, post_inline: bool, target: &crate::Target) -> Domain {
    if func.entry.is_none() || func.coroutine.is_some() {
        return Domain::NONE;
    }
    let domain = |changed: bool| if changed { Domain::MEM | Domain::VALUES } else { Domain::NONE };
    let (mut changed, affine) = eliminate_affine_block_params(func);
    if !crate::schedule::has_live_op(func, |op| matches!(op, Op::Load(..) | Op::StackLoad(..) | Op::Store(..) | Op::StackStore(..))) {
        changed |= sink_terminator_defs(func);
        return domain(changed);
    }
    let cfg = func.cfg();
    let mut prov = Provenance::with_affine(func, &cfg, affine);
    if caller_loc::optimize(func, &cfg, &prov) {
        changed = true;
        if func.types.len() > prov.param_roots.len() { prov = Provenance::new(func, &cfg); }
    }
    changed |= crate::mem::drop_unread_frame_stores(func, &cfg, &prov);
    changed |= fold_affine_mem(func, &prov.slot_offsets, &prov.affine);
    changed |= pack::coalesce_loads(func, &cfg, &prov, target);
    changed |= sink_terminator_defs(func);
    let Some((defs, mut consts, selects)) = collect_scan_meta(func) else { return domain(changed) };
    let mut bw = BlockWrites::new(func, &cfg, &prov);
    let mut subst = identity(func.types.len());
    let mut state = State { prov: &prov, defs: &defs, selects: &selects, consts: &mut consts, subst: &mut subst, target, post_inline };
    let has_join = cfg.rpo.len() > 1 && cfg.rpo.iter().any(|&b| cfg.pred_count(b) >= 2);
    let (mut renamed, mut scan_changed, mut def_blocks) = (false, false, if has_join { func.def_blocks() } else { Vec::new() });
    let (mut visited, mut added_phis) = (vec![false; func.blocks.len()], vec![0u8; func.blocks.len()]);
    let mut known_out: Vec<KnownLoads> = (0..func.blocks.len()).map(|_| KnownLoads::new()).collect();
    let mut block_bases: Vec<Vec<(MemBase, MemRoot)>> = vec![Vec::new(); func.blocks.len()];
    let mut bufs = ScanBuf { kept: Vec::new(), pending_stores: Vec::new(), const_stores: Vec::new() };
    let (mut clobbers_seen, mut guard) = (Vec::new(), FarCseGuard::empty());
    for &block in &cfg.rpo {
        let (mut known, mut bases) = bw.entry_known(func, state.subst, &known_out, &block_bases, block);
        let preds = eligible_join_preds(func, &cfg, &visited, block);
        let mut join = JoinCtx { cfg: &cfg, bw: &bw, known_out: &known_out, block_bases: &block_bases, visited: &visited, def_blocks: &mut def_blocks, preds, added_phis: &mut added_phis };
        let (bc, br) = scan(func, block, &mut state, &mut known, &mut bases, &mut join, (&mut bufs, &mut clobbers_seen, &mut guard));
        if has_join { def_blocks.resize(func.types.len(), Some(block)); }
        visited[block.index()] = true;
        (scan_changed, changed, renamed) = (scan_changed | bc, changed | bc, renamed | br);
        (known_out[block.index()], block_bases[block.index()]) = (known, bases);
    }
    if renamed { rename(func, state.subst); }
    if scan_changed { changed |= pack::coalesce_vec_rescan(func, &cfg, target); }
    changed |= remat::split_long_loads(func, &prov) | sink::sink_loads(func, &prov);
    if post_inline {
        if elide::elide_moves(func, &prov) {
            changed = true;
            let cfg = func.cfg();
            changed |= elide::forward_moves(func, &Provenance::new(func, &cfg));
        } else {
            changed |= elide::forward_moves(func, &prov);
        }
    }
    changed |= regroup::run(func, (!changed).then_some(&prov));
    domain(changed)
}

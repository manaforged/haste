//! Reuses what memory holds along straight-line code: a load of the bytes an earlier load or store on the same path read or wrote, with no write of unknown reach between them, takes that value.

mod coalesce;

use coalesce::{ScanBuf, slot_unescaped, try_coalesce_store};

use crate::cfg::Cfg;
use crate::ir::{Binary, Block, Convert, Function, Inst, Lane, Op, ShuffleMask, StackSlot, Type, Value};
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

fn forward_concat(func: &mut Function, state: &mut State<'_>, kept: &mut Vec<Option<Inst>>, known: &KnownLoads, inst: Inst, loc: MemLoc, ty: Type) -> bool {
    if !matches!(ty, Type::I64 | Type::F64) {
        return false;
    }
    let Some(hi_off) = loc.offset.checked_add(4) else { return false };
    let half_at = |offset: i64| {
        known
            .get(loc.base, offset, Type::I32)
            .map(|v| (v, false))
            .or_else(|| known.get(loc.base, offset, Type::F32).map(|v| (v, true)))
    };
    let Some((lo, lo_float)) = half_at(loc.offset) else { return false };
    let Some((hi, hi_float)) = half_at(hi_off) else { return false };
    let lo32 = if lo_float { emit_op(func, state, kept, Type::I32, Op::Convert(Convert::Bitcast, lo)) } else { lo };
    let hi32 = if hi_float { emit_op(func, state, kept, Type::I32, Op::Convert(Convert::Bitcast, hi)) } else { hi };
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

/// Drops from `block` each load whose bytes `known` holds, recording its replacement in `subst`, eliminates dead overwritten stores, and leaves in `known` what memory holds at the block's end; returns `(changed, renamed)`.
fn scan(func: &mut Function, block: Block, state: &mut State<'_>, known: &mut KnownLoads) -> (bool, bool) {
    let insts = std::mem::take(&mut func.blocks[block.index()].insts);
    let mut bufs = ScanBuf {
        kept: Vec::with_capacity(insts.len()),
        pending_stores: Vec::new(),
        const_stores: Vec::new(),
    };
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
            Effect::Read(loc, ty, result) => match known.get(loc.base, loc.offset, ty) {
                Some(value) => {
                    state.subst[result.index()] = value;
                    renamed = true;
                    changed = true;
                    continue;
                }
                None => {
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
                    } else {
                        known.remember(loc, ty, result);
                        bufs.pending_stores.retain(|&(ploc, _)| !may_alias(loc, ploc, state.prov));
                        let safe_read = slot_unescaped(state.prov, loc);
                        bufs.const_stores.retain(|cs| {
                            !may_alias(loc, cs.loc, state.prov)
                                && (slot_unescaped(state.prov, cs.loc) || safe_read || cs.loc.base == loc.base)
                        });
                    }
                }
            },
            Effect::Write(loc, ty, value) => {
                known.write(loc, ty, value, state.prov);
                let safe_write = slot_unescaped(state.prov, loc);
                bufs.const_stores.retain(|cs| {
                    !may_alias(loc, cs.loc, state.prov)
                        && (slot_unescaped(state.prov, cs.loc) || safe_write || cs.loc.base == loc.base)
                });
                let track = match loc.root {
                    MemRoot::Slot(s) => state.prov.slot_unescaped(s),
                    _ => true,
                };
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
            let op = &func.insts[last.index()].op;
            match op {
                Op::Jump(b, args) if *b == target => incoming.push(args.clone()),
                Op::Brif(_, t, t_args, e, e_args) => {
                    if *t == target {
                        incoming.push(t_args.clone());
                    }
                    if *e == target {
                        incoming.push(e_args.clone());
                    }
                }
                _ if op.targets().any(|t| t == target) => {
                    non_branch_pred = true;
                    break;
                }
                _ => {}
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
        let mut idx = 0;
        func.blocks[b_idx].params.retain(|_| {
            let keep = !eliminated[idx];
            idx += 1;
            keep
        });
        for pred_idx in 0..func.blocks.len() {
            let Some(&last) = func.blocks[pred_idx].insts.last() else { continue };
            let retain_args = |args: &mut Vec<Value>| {
                let mut k = 0;
                args.retain(|_| {
                    let keep = !eliminated[k];
                    k += 1;
                    keep
                });
            };
            match &mut func.insts[last.index()].op {
                Op::Jump(b, args) if *b == target => retain_args(args),
                Op::Brif(_, t, t_args, e, e_args) => {
                    if *t == target {
                        retain_args(t_args);
                    }
                    if *e == target {
                        retain_args(e_args);
                    }
                }
                _ => {}
            }
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
    for block in &func.blocks {
        for &inst in &block.insts {
            let res_ty = func.insts[inst.index()].result.map(|r| func.ty(r));
            let val_ty = match func.insts[inst.index()].op {
                Op::Load(..) => res_ty,
                Op::Store(val, ..) => Some(func.ty(val)),
                _ => None,
            };
            let Some(ty) = val_ty else { continue };
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
    }
    changed
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
        match &func.insts[last.index()].op {
            Op::Jump(_, args) => args.iter().for_each(|v| term_uses[v.index()] += 1),
            Op::Brif(_, _, t_args, _, e_args) => {
                t_args.iter().for_each(|v| term_uses[v.index()] += 1);
                e_args.iter().for_each(|v| term_uses[v.index()] += 1);
            }
            _ => continue,
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
        match &func.insts[last.index()].op {
            Op::Jump(_, args) => args.iter().for_each(|v| term_uses[v.index()] = 0),
            Op::Brif(_, _, t_args, _, e_args) => {
                t_args.iter().for_each(|v| term_uses[v.index()] = 0);
                e_args.iter().for_each(|v| term_uses[v.index()] = 0);
            }
            _ => {}
        }
    }
    changed
}

struct BlockWrites<'a> {
    cfg: &'a Cfg,
    prov: &'a Provenance,
    writes: &'a [Vec<Inst>],
    trycall: &'a [bool],
}

fn collect_block_writes(func: &Function) -> (Vec<Vec<Inst>>, Vec<bool>) {
    let mut writes = vec![Vec::new(); func.blocks.len()];
    let mut trycall = vec![false; func.blocks.len()];
    for (b_idx, block) in func.blocks.iter().enumerate() {
        trycall[b_idx] = func.last(Block(b_idx as u32)).is_some_and(|op| matches!(op, Op::TryCall { .. }));
        for &inst in &block.insts {
            if matches!(
                func.insts[inst.index()].op,
                Op::Store(..)
                    | Op::StackStore(..)
                    | Op::Copy(..)
                    | Op::Move(..)
                    | Op::AtomicLoad(_)
                    | Op::AtomicStore(..)
                    | Op::AtomicRmw(..)
                    | Op::AtomicCas(..)
                    | Op::Fence
                    | Op::Call { .. }
                    | Op::TryCall { .. }
            ) {
                writes[b_idx].push(inst);
            }
        }
    }
    (writes, trycall)
}

fn entry_known(func: &Function, bw: &BlockWrites<'_>, known_out: &[KnownLoads], block: Block) -> KnownLoads {
    if Some(block) == bw.cfg.entry {
        return KnownLoads::new();
    }
    if let Some(pred) = bw.cfg.single_pred(block) {
        return if bw.trycall[pred.index()] { KnownLoads::new() } else { known_out[pred.index()].clone() };
    }
    let Some(dom) = bw.cfg.idom[block.index()] else {
        return KnownLoads::new();
    };
    if bw.trycall[dom.index()] {
        return KnownLoads::new();
    }
    let mut known = known_out[dom.index()].clone();
    if known.is_empty() {
        return known;
    }
    let mut in_region = vec![false; func.blocks.len()];
    let mut work = Vec::new();
    for &pred in bw.cfg.preds_of(block) {
        if bw.cfg.reachable[pred.index()] && pred != dom && !in_region[pred.index()] {
            in_region[pred.index()] = true;
            work.push(pred);
        }
    }
    while let Some(m) = work.pop() {
        if bw.trycall[m.index()] {
            return KnownLoads::new();
        }
        for &w in &bw.writes[m.index()] {
            known.invalidate(func, &func.insts[w.index()].op, bw.prov);
            if known.is_empty() {
                return known;
            }
        }
        for &pred in bw.cfg.preds_of(m) {
            if bw.cfg.reachable[pred.index()] && pred != dom && !in_region[pred.index()] {
                in_region[pred.index()] = true;
                work.push(pred);
            }
        }
    }
    known
}

/// Replaces each load of bytes that an earlier access on every path to it read or wrote with no write of unknown reach between; returns whether `func` was mutated.
pub(crate) fn run(func: &mut Function) -> bool {
    let Some(entry) = func.entry else { return false };
    if func.coroutine.is_some() {
        return false;
    }
    let mut changed = eliminate_affine_block_params(func);
    let cfg = Cfg::from_entry(func, entry);
    let prov = Provenance::new(func, &cfg);
    changed |= fold_affine_mem(func, &prov.slot_offsets, &prov.affine);
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
    let (writes, trycall) = collect_block_writes(func);
    let bw = BlockWrites { cfg: &cfg, prov: &prov, writes: &writes, trycall: &trycall };
    let mut subst = identity(func.types.len());
    let mut state = State { prov: &prov, selects: &selects, consts: &mut consts, subst: &mut subst };
    let mut renamed = false;
    let mut known_out: Vec<KnownLoads> = (0..func.blocks.len()).map(|_| KnownLoads::new()).collect();
    for &block in &cfg.rpo {
        let mut known = entry_known(func, &bw, &known_out, block);
        let (block_changed, block_renamed) = scan(func, block, &mut state, &mut known);
        changed |= block_changed;
        renamed |= block_renamed;
        known_out[block.index()] = known;
    }
    if renamed {
        rename(func, state.subst);
    }
    changed
}

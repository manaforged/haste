use super::{Kind, Scan};
use crate::ir::{Binary, Convert, Function, Inst, Op, StackSlot, Type, Value};

fn int_for_bytes(bytes: u32) -> Option<Type> {
    match bytes {
        1 => Some(Type::I8),
        2 => Some(Type::I16),
        4 => Some(Type::I32),
        8 => Some(Type::I64),
        _ => None,
    }
}

fn is_scalar(ty: Type) -> bool {
    ty.is_small_int() || ty.is_float()
}

fn is_splittable(ty: Type) -> bool {
    is_scalar(ty) || ty == Type::I128
}

pub(super) fn eval_consts(func: &Function) -> Vec<Option<i64>> {
    let mut consts = vec![None; func.types.len()];
    for inst in func.blocks.iter().flat_map(|b| &b.insts) {
        let data = &func.insts[inst.index()];
        if let (&Op::Const(imm), Some(res)) = (&data.op, data.result) {
            consts[res.index()] = Some(imm);
        }
    }
    for _ in 0..4 {
        let mut changed = false;
        for inst in func.blocks.iter().flat_map(|b| &b.insts) {
            let data = &func.insts[inst.index()];
            let Some(res) = data.result else { continue };
            if consts[res.index()].is_some() {
                continue;
            }
            let val = match data.op {
                Op::Convert(Convert::Uextend, x) => consts[x.index()].map(|imm| crate::fold::extend(imm, func.ty(x), false)),
                Op::Convert(Convert::Sextend, x) => consts[x.index()].map(|imm| crate::fold::extend(imm, func.ty(x), true)),
                Op::Convert(Convert::Ireduce, x) => consts[x.index()].map(|imm| crate::fold::extend(imm, func.ty(res), false)),
                _ => None,
            };
            if val.is_some() {
                consts[res.index()] = val;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    consts
}

pub(super) const MAX_COPY_BYTES: u32 = crate::copy::MOVE_INLINE * 2;
pub(super) const PROMOTE_BOTH_BYTES: u32 = 1024;
pub(super) const PROMOTE_BOTH_PIECES: usize = 256;

pub(super) fn copy_bytes_limit(consts: &[Option<i64>], len: Value, limit: u32) -> Option<u32> {
    consts.get(len.index()).copied().flatten().and_then(|n| u32::try_from(n).ok()).filter(|&n| n <= limit)
}

pub(super) fn copy_open_slots(
    func: &Function,
    addrs: &[Option<(StackSlot, i64)>],
    consts: &[Option<i64>],
    param_roots: &[crate::mem::MemRoot],
) -> Vec<bool> {
    let mut open = vec![true; func.slots.len()];
    let close_addr = |open: &mut [bool], v: Value| {
        if let Some((s, _)) = addrs[v.index()] {
            open[s.index()] = false;
        } else if let Some(&crate::mem::MemRoot::Slot(s)) = param_roots.get(v.index()) {
            open[s.index()] = false;
        }
    };
    for inst in func.blocks.iter().flat_map(|b| &b.insts) {
        let data = &func.insts[inst.index()];
        match data.op {
            Op::Load(addr, _) => {
                if addrs[addr.index()].is_none()
                    && let Some(&crate::mem::MemRoot::Slot(s)) = param_roots.get(addr.index())
                {
                    open[s.index()] = false;
                }
            }
            Op::Store(val, addr, _) => {
                close_addr(&mut open, val);
                if addrs[addr.index()].is_none()
                    && let Some(&crate::mem::MemRoot::Slot(s)) = param_roots.get(addr.index())
                {
                    open[s.index()] = false;
                }
            }
            Op::Move(to, from, _) => {
                for v in [to, from] {
                    if addrs[v.index()].is_none()
                        && let Some(&crate::mem::MemRoot::Slot(s)) = param_roots.get(v.index())
                    {
                        open[s.index()] = false;
                    }
                }
            }
            Op::Copy(to, from, len) => {
                if let Some((s, _)) = addrs[len.index()] {
                    open[s.index()] = false;
                }
                let limit = if addrs[to.index()].is_some() && addrs[from.index()].is_some() {
                    PROMOTE_BOTH_BYTES
                } else {
                    MAX_COPY_BYTES
                };
                if copy_bytes_limit(consts, len, limit).is_some() {
                    for v in [to, from] {
                        if addrs[v.index()].is_none()
                            && let Some(&crate::mem::MemRoot::Slot(s)) = param_roots.get(v.index())
                        {
                            open[s.index()] = false;
                        }
                    }
                } else {
                    close_addr(&mut open, to);
                    close_addr(&mut open, from);
                }
            }
            Op::Binary(Binary::Iadd | Binary::Isub, a, b) if data.result.is_some_and(|r| addrs[r.index()].is_some()) => {
                if addrs[a.index()].is_some() && addrs[b.index()].is_some() {
                    close_addr(&mut open, a);
                    close_addr(&mut open, b);
                }
            }
            _ => data.op.for_operands(|v| {
                if let Some((s, _)) = addrs[v.index()] {
                    open[s.index()] = false;
                }
            }),
        }
    }
    open
}

pub(super) struct CopySplit {
    block: usize,
    original: Inst,
    len: Value,
    loads: Vec<Inst>,
    stores: Vec<Inst>,
}

pub(super) fn expand_copies(
    func: &mut Function,
    can_expand: impl Fn(Value, Value, Value) -> Option<Vec<(Type, u32)>>,
) -> (Vec<CopySplit>, bool) {
    let mut splits = Vec::new();
    let mut dropped_zero = false;
    for block in 0..func.blocks.len() {
        let insts = std::mem::take(&mut func.blocks[block].insts);
        let mut out = Vec::with_capacity(insts.len());
        for inst in insts {
            let Op::Copy(to, from, len) = func.insts[inst.index()].op else {
                out.push(inst);
                continue;
            };
            let Some(pieces) = can_expand(to, from, len) else {
                out.push(inst);
                continue;
            };
            if pieces.is_empty() {
                dropped_zero = true;
                continue;
            }
            let mut loads = Vec::with_capacity(pieces.len());
            let mut stores = Vec::with_capacity(pieces.len());
            let mut vals = Vec::with_capacity(pieces.len());
            for (ty, at) in pieces {
                let word = func.value(ty);
                loads.push(func.add(Op::Load(from, at as i32), Some(word)));
                vals.push((word, at));
            }
            for (word, at) in vals {
                stores.push(func.add(Op::Store(word, to, at as i32), None));
            }
            out.extend_from_slice(&loads);
            out.extend_from_slice(&stores);
            splits.push(CopySplit { block, original: inst, len, loads, stores });
        }
        func.blocks[block].insts = out;
    }
    (splits, dropped_zero)
}

fn copy_bases(func: &Function, loads: &[Inst], stores: &[Inst]) -> Option<(Value, Value)> {
    if loads.len() != stores.len() || loads.is_empty() {
        return None;
    }
    let mut ends = None;
    for (&load, &store) in loads.iter().zip(stores) {
        let (Op::Load(from, lo), Some(word)) = (&func.insts[load.index()].op, func.insts[load.index()].result) else { return None };
        let Op::Store(value, to, so) = func.insts[store.index()].op else { return None };
        if value != word || *lo != so || ends.is_some_and(|pair| pair != (to, *from)) {
            return None;
        }
        ends = Some((to, *from));
    }
    ends
}

pub(super) fn rejoin_copies(func: &mut Function, splits: &[CopySplit]) -> bool {
    let mut all_rejoined = true;
    for split in splits {
        let total = split.loads.len() + split.stores.len();
        let insts = &func.blocks[split.block].insts;
        let Some(first) = split.loads.first() else { continue };
        let Some(at) = insts.iter().position(|inst| inst == first) else {
            all_rejoined = false;
            continue;
        };
        let intact = insts.get(at..at + split.loads.len()) == Some(split.loads.as_slice())
            && insts.get(at + split.loads.len()..at + total) == Some(split.stores.as_slice());
        let Some((to, from)) = intact.then(|| copy_bases(func, &split.loads, &split.stores)).flatten() else {
            all_rejoined = false;
            continue;
        };
        func.insts[split.original.index()].op = Op::Copy(to, from, split.len);
        func.blocks[split.block].insts.splice(at..at + total, [split.original]);
    }
    all_rejoined
}

fn splittable_slices(scan: &Scan) -> Vec<Vec<(i64, Type)>> {
    let mut per_slot: Vec<Vec<(i64, i64, Type, usize)>> = vec![Vec::new(); scan.escaped.len()];
    for access in &scan.accesses {
        let var = &scan.vars[access.var];
        if scan.escaped[var.slot.index()] {
            continue;
        }
        let item = (var.offset, var.offset + i64::from(access.ty.bytes()), access.ty, access.var);
        let slot_list = &mut per_slot[var.slot.index()];
        if !slot_list.contains(&item) {
            slot_list.push(item);
        }
    }
    let mut out = vec![Vec::new(); scan.escaped.len()];
    for (slot, mut items) in per_slot.into_iter().enumerate() {
        if items.is_empty() {
            continue;
        }
        items.sort_unstable_by_key(|&(off, end, _, _)| (off, end));
        let mut idx = 0;
        while idx < items.len() {
            let start_idx = idx;
            let mut comp_end = items[idx].1;
            idx += 1;
            while idx < items.len() && items[idx].0 < comp_end {
                comp_end = comp_end.max(items[idx].1);
                idx += 1;
            }
            let comp = &items[start_idx..idx];
            let same_start = comp.iter().all(|&(o, _, _, _)| o == comp[0].0);
            let all_sound = comp.iter().all(|&(_, _, _, v)| scan.vars[v].sound());
            if (same_start && all_sound) || !comp.iter().all(|&(_, _, ty, _)| is_splittable(ty)) {
                continue;
            }
            let mut bounds = Vec::with_capacity(comp.len() * 3);
            for &(o, e, ty, _) in comp {
                bounds.push(o);
                if ty == Type::I128 {
                    bounds.push(o + 8);
                }
                bounds.push(e);
            }
            bounds.sort_unstable();
            bounds.dedup();
            let mut comp_slices = Vec::new();
            let mut valid = true;
            for w in bounds.windows(2) {
                let Ok(len) = u32::try_from(w[1] - w[0]) else {
                    valid = false;
                    break;
                };
                for (slice_ty, rel) in crate::copy::words(len) {
                    let slice_off = w[0] + i64::from(rel);
                    if i32::try_from(slice_off).is_err() {
                        valid = false;
                        break;
                    }
                    comp_slices.push((slice_off, slice_ty));
                }
                if !valid {
                    break;
                }
            }
            if valid {
                out[slot].extend(comp_slices);
            }
        }
    }
    out
}

fn emit_word_store_pieces(
    func: &mut Function,
    mut reuse: Option<Inst>,
    (src, base_ty): (Value, Type),
    (slot, off): (StackSlot, i64),
    pieces: &[(i64, Type)],
    out: &mut Vec<Inst>,
) {
    for &(s_off, s_ty) in pieces {
        let shift = (s_off - off) * 8;
        let shifted = if shift == 0 {
            src
        } else {
            let c = func.value(base_ty);
            out.push(func.add(Op::Const(shift), Some(c)));
            let s = func.value(base_ty);
            out.push(func.add(Op::Binary(Binary::Ushr, src, c), Some(s)));
            s
        };
        let piece = if s_ty == base_ty {
            shifted
        } else {
            let red = func.value(s_ty);
            out.push(func.add(Op::Convert(Convert::Ireduce, shifted), Some(red)));
            red
        };
        let op = Op::StackStore(piece, slot, s_off as i32);
        if let Some(inst) = reuse.take() {
            func.insts[inst.index()].op = op;
            out.push(inst);
        } else {
            out.push(func.add(op, None));
        }
    }
}

fn emit_split_store(func: &mut Function, (inst, val, ty): (Inst, Value, Type), (slot, off): (StackSlot, i64), pieces: &[(i64, Type)], out: &mut Vec<Inst>) {
    if ty == Type::I128 {
        let mid = pieces.partition_point(|&(s_off, _)| s_off < off + 8);
        let (lo_pieces, hi_pieces) = pieces.split_at(mid);
        if lo_pieces.is_empty() || hi_pieces.is_empty() {
            out.push(inst);
            return;
        }
        let lo64 = func.value(Type::I64);
        out.push(func.add(Op::Convert(Convert::Ireduce, val), Some(lo64)));
        let c64 = func.value(Type::I8);
        out.push(func.add(Op::Const(64), Some(c64)));
        let shifted = func.value(Type::I128);
        out.push(func.add(Op::Binary(Binary::Ushr, val, c64), Some(shifted)));
        let hi64 = func.value(Type::I64);
        out.push(func.add(Op::Convert(Convert::Ireduce, shifted), Some(hi64)));
        emit_word_store_pieces(func, Some(inst), (lo64, Type::I64), (slot, off), lo_pieces, out);
        emit_word_store_pieces(func, None, (hi64, Type::I64), (slot, off + 8), hi_pieces, out);
        return;
    }
    let Some(base_ty) = int_for_bytes(ty.bytes()) else {
        out.push(inst);
        return;
    };
    let src = if ty.is_float() {
        let cast = func.value(base_ty);
        out.push(func.add(Op::Convert(Convert::Bitcast, val), Some(cast)));
        cast
    } else {
        val
    };
    emit_word_store_pieces(func, Some(inst), (src, base_ty), (slot, off), pieces, out);
}

fn build_word_load(
    func: &mut Function,
    reuse: Option<(Inst, bool)>,
    base_ty: Type,
    (slot, off): (StackSlot, i64),
    pieces: &[(i64, Type)],
    out: &mut Vec<Inst>,
) -> Option<Value> {
    if pieces.len() == 1 && pieces[0] == (off, base_ty) {
        if let Some((inst, false)) = reuse {
            func.insts[inst.index()].op = Op::StackLoad(slot, off as i32);
            out.push(inst);
            return func.insts[inst.index()].result;
        }
        let raw = func.value(base_ty);
        out.push(func.add(Op::StackLoad(slot, off as i32), Some(raw)));
        return Some(raw);
    }
    let mut acc: Option<Value> = None;
    for (idx, &(s_off, s_ty)) in pieces.iter().enumerate() {
        let loaded = func.value(s_ty);
        out.push(func.add(Op::StackLoad(slot, s_off as i32), Some(loaded)));
        let ext = if s_ty == base_ty {
            loaded
        } else {
            let e = func.value(base_ty);
            out.push(func.add(Op::Convert(Convert::Uextend, loaded), Some(e)));
            e
        };
        let shift = (s_off - off) * 8;
        let term = if shift == 0 {
            ext
        } else {
            let c = func.value(base_ty);
            out.push(func.add(Op::Const(shift), Some(c)));
            let sh = func.value(base_ty);
            out.push(func.add(Op::Binary(Binary::Ishl, ext, c), Some(sh)));
            sh
        };
        let last = idx + 1 == pieces.len();
        acc = match acc {
            None => Some(term),
            Some(prev) => {
                if last && let Some((inst, false)) = reuse {
                    func.insts[inst.index()].op = Op::Binary(Binary::Bor, prev, term);
                    out.push(inst);
                    return func.insts[inst.index()].result;
                }
                let b = func.value(base_ty);
                out.push(func.add(Op::Binary(Binary::Bor, prev, term), Some(b)));
                Some(b)
            }
        };
    }
    acc
}

fn emit_split_load(func: &mut Function, (inst, ty): (Inst, Type), (slot, off): (StackSlot, i64), pieces: &[(i64, Type)], out: &mut Vec<Inst>) {
    if ty == Type::I128 {
        let mid = pieces.partition_point(|&(s_off, _)| s_off < off + 8);
        let (lo_pieces, hi_pieces) = pieces.split_at(mid);
        let Some(lo64) = (!lo_pieces.is_empty()).then(|| build_word_load(func, None, Type::I64, (slot, off), lo_pieces, out)).flatten() else {
            out.push(inst);
            return;
        };
        let Some(hi64) = (!hi_pieces.is_empty()).then(|| build_word_load(func, None, Type::I64, (slot, off + 8), hi_pieces, out)).flatten() else {
            out.push(inst);
            return;
        };
        func.insts[inst.index()].op = Op::Binary(Binary::Iconcat, lo64, hi64);
        out.push(inst);
        return;
    }
    let Some(base_ty) = int_for_bytes(ty.bytes()) else {
        out.push(inst);
        return;
    };
    if !ty.is_float() {
        if build_word_load(func, Some((inst, false)), base_ty, (slot, off), pieces, out).is_none() {
            out.push(inst);
        }
        return;
    }
    if let Some(packed) = build_word_load(func, Some((inst, true)), base_ty, (slot, off), pieces, out) {
        func.insts[inst.index()].op = Op::Convert(Convert::Bitcast, packed);
        out.push(inst);
    } else {
        out.push(inst);
    }
}

pub(super) fn split(func: &mut Function, scan: &Scan) -> bool {
    let slot_slices = splittable_slices(scan);
    if slot_slices.iter().all(Vec::is_empty) {
        return false;
    }
    let mut changed = false;
    let mut pieces = Vec::new();
    for block in 0..func.blocks.len() {
        let insts = std::mem::take(&mut func.blocks[block].insts);
        let mut out = Vec::with_capacity(insts.len());
        for inst in insts {
            let Some(touch) = scan.touch(&func.insts[inst.index()]) else {
                out.push(inst);
                continue;
            };
            let ty = func.ty(touch.kind.value());
            let Some(off) = touch.offset.filter(|_| is_splittable(ty) && !slot_slices[touch.slot.index()].is_empty()) else {
                out.push(inst);
                continue;
            };
            let end = off + i64::from(ty.bytes());
            pieces.clear();
            pieces.extend(
                slot_slices[touch.slot.index()]
                    .iter()
                    .copied()
                    .filter(|&(s_off, s_ty)| s_off >= off && s_off + i64::from(s_ty.bytes()) <= end),
            );
            let mut cursor = off;
            let covered = !pieces.is_empty()
                && pieces.iter().all(|&(s_off, s_ty)| {
                    let ok = s_off == cursor;
                    cursor += i64::from(s_ty.bytes());
                    ok
                })
                && cursor == end;
            if !covered || (pieces.len() == 1 && pieces[0] == (off, ty)) {
                out.push(inst);
                continue;
            }
            match touch.kind {
                Kind::Store(val) => emit_split_store(func, (inst, val, ty), (touch.slot, off), &pieces, &mut out),
                Kind::Load(_) => emit_split_load(func, (inst, ty), (touch.slot, off), &pieces, &mut out),
            }
            changed = true;
        }
        func.blocks[block].insts = out;
    }
    changed
}

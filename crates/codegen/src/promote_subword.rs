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
            let same_span = comp.iter().all(|&(o, e, _, _)| o == comp[0].0 && e == comp[0].1);
            let all_sound = comp.iter().all(|&(_, _, _, v)| scan.vars[v].sound());
            if (same_span && all_sound) || !comp.iter().all(|&(_, _, ty, _)| is_scalar(ty)) {
                continue;
            }
            let mut bounds = Vec::with_capacity(comp.len() * 2);
            for &(o, e, _, _) in comp {
                bounds.push(o);
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

fn emit_split_store(func: &mut Function, (inst, val, ty): (Inst, Value, Type), (slot, off): (StackSlot, i64), pieces: &[(i64, Type)], out: &mut Vec<Inst>) {
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
    for (idx, &(s_off, s_ty)) in pieces.iter().enumerate() {
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
        if idx == 0 {
            func.insts[inst.index()].op = op;
            out.push(inst);
        } else {
            out.push(func.add(op, None));
        }
    }
}

fn emit_split_load(func: &mut Function, (inst, ty): (Inst, Type), (slot, off): (StackSlot, i64), pieces: &[(i64, Type)], out: &mut Vec<Inst>) {
    let Some(base_ty) = int_for_bytes(ty.bytes()) else {
        out.push(inst);
        return;
    };
    if pieces.len() == 1 {
        let raw = func.value(base_ty);
        out.push(func.add(Op::StackLoad(slot, off as i32), Some(raw)));
        func.insts[inst.index()].op = Op::Convert(Convert::Bitcast, raw);
        out.push(inst);
        return;
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
            Some(prev) if last && !ty.is_float() => {
                func.insts[inst.index()].op = Op::Binary(Binary::Bor, prev, term);
                out.push(inst);
                return;
            }
            Some(prev) => {
                let b = func.value(base_ty);
                out.push(func.add(Op::Binary(Binary::Bor, prev, term), Some(b)));
                Some(b)
            }
        };
    }
    if let Some(packed) = acc {
        func.insts[inst.index()].op = Op::Convert(Convert::Bitcast, packed);
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
            let Some(off) = touch.offset.filter(|_| is_scalar(ty) && !slot_slices[touch.slot.index()].is_empty()) else {
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
            let covered = pieces.first().is_some_and(|p| p.0 == off) && pieces.last().is_some_and(|p| p.0 + i64::from(p.1.bytes()) == end);
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

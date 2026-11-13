use super::asm::Reg;
use crate::alloc::{Location, Plan};
use crate::ir::{Binary, Convert, Function, Inst, IntCC, Op, Type, Value};
use crate::live::{Sib, SibBase};

pub(super) fn shift_parts(op: &Op, plan: &Plan<Reg>) -> Option<(Binary, Value, i64)> {
    match *op {
        Op::Binary(sop @ (Binary::Ishl | Binary::Ushr | Binary::Sshr | Binary::Rotl | Binary::Rotr), b, c) => Some((sop, b, plan.const_of(c)?)),
        Op::Binary(Binary::Imul, a, b) => {
            if let Some(m) = plan.const_of(b).filter(|&m| m > 1 && m.count_ones() == 1) {
                Some((Binary::Ishl, a, i64::from(m.trailing_zeros())))
            } else if let Some(m) = plan.const_of(a).filter(|&m| m > 1 && m.count_ones() == 1) {
                Some((Binary::Ishl, b, i64::from(m.trailing_zeros())))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn preserved(func: &Function, plan: &Plan<Reg>, block_idx: usize, start: usize, end: usize, skip_pos: Option<usize>, val: Value) -> bool {
    if start + 1 >= end || matches!(plan.location(val), Location::Imm(_) | Location::SlotAddr(_) | Location::Frame(..)) {
        return true;
    }
    let (reg, slot_id) = match plan.location(val) {
        Location::Reg(reg) => (Some(reg), None),
        Location::Slot => {
            let Some(slot_id) = plan.shared.get(val.index()).copied().flatten() else {
                return true;
            };
            (None, Some(slot_id))
        }
        _ => return false,
    };
    let clobbers = |r: Value| {
        reg.is_some_and(|reg| plan.reg_of(r) == Some(reg))
            || slot_id.is_some_and(|id| matches!(plan.location(r), Location::Slot) && plan.shared.get(r.index()).copied().flatten() == Some(id))
    };
    for pos in start + 1..end {
        if skip_pos == Some(pos) {
            continue;
        }
        let data = &func.insts[func.blocks[block_idx].insts[pos].index()];
        if data.result.is_some_and(|r| plan.elided[r.index()] || plan.deferred[r.index()]) {
            continue;
        }
        if (reg.is_some() && matches!(data.op, Op::Call { .. } | Op::TryCall { .. }))
            || data.result.and_then(|r| plan.pairs[r.index()]).is_some_and(clobbers)
            || crate::live::results(data).any(clobbers)
        {
            return false;
        }
    }
    true
}

fn fuse_ireduce_ushr(func: &Function, plan: &mut Plan<Reg>, (uses, pos_of): (&[u32], &[Option<(usize, usize, Inst)>]), (b_idx, pos, res): (usize, usize, Value), shr: Value) {
    let src_ty = func.ty(shr);
    if !matches!(src_ty, Type::I32 | Type::I64) || uses[shr.index()] != 1 || plan.elided[shr.index()] {
        return;
    }
    let Some((sb, spos, sinst)) = pos_of[shr.index()] else { return };
    if sb != b_idx || spos >= pos {
        return;
    }
    let Op::Binary(Binary::Ushr, val, amt) = func.insts[sinst.index()].op else { return };
    let Some(shift) = plan.const_of(amt) else { return };
    let src_bits = src_ty.bits();
    let shift = (shift as u32) & (src_bits - 1);
    if shift > 0 && src_bits - shift <= func.ty(res).bits() && preserved(func, plan, b_idx, spos, pos, None, val) {
        plan.elided[shr.index()] = true;
    }
}

fn can_peel_uextend(
    func: &Function,
    plan: &Plan<Reg>,
    (uses, pos_of): (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, limit, end, skip, res): (usize, usize, usize, Option<usize>, Value),
    (val, ok): (Value, &dyn Fn(Value) -> bool),
) -> bool {
    if uses[val.index()] != 1 || plan.elided[val.index()] {
        return false;
    }
    let Some((ub, upos, uinst)) = pos_of[val.index()] else { return false };
    if ub != b_idx || upos >= limit {
        return false;
    }
    let Op::Convert(Convert::Uextend, inner) = func.insts[uinst.index()].op else { return false };
    inner != res && ok(inner) && preserved(func, plan, b_idx, upos, end, skip, inner)
}

fn try_fuse_shift(
    func: &Function,
    plan: &mut Plan<Reg>,
    (uses, pos_of): (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos, res, alu_op): (usize, usize, Value, Binary),
    (shift_val, other): (Value, Value),
) -> bool {
    if uses[shift_val.index()] != 1 || plan.elided[shift_val.index()] {
        return false;
    }
    let Some((sb, spos, sinst)) = pos_of[shift_val.index()] else { return false };
    if sb != b_idx || spos >= pos {
        return false;
    }
    let Some((sop, b, imm)) = shift_parts(&func.insts[sinst.index()].op, plan) else { return false };
    let logical = matches!(alu_op, Binary::Band | Binary::Bor | Binary::Bxor);
    if b == res || !(matches!(sop, Binary::Ishl | Binary::Ushr | Binary::Sshr) || (logical && matches!(sop, Binary::Rotl | Binary::Rotr))) {
        return false;
    }
    let ty_bits = func.ty(res).bits();
    let shift_ok = |inner: Value| func.ty(inner).is_small_int() && ((imm as u32) & (ty_bits - 1)) >= ty_bits - func.ty(inner).bits();
    let peeled = sop == Binary::Ishl && can_peel_uextend(func, plan, (uses, pos_of), (b_idx, spos, pos, Some(spos), res), (b, &shift_ok));
    if !peeled && !preserved(func, plan, b_idx, spos, pos, None, b) {
        return false;
    }
    plan.elided[shift_val.index()] = true;
    if peeled {
        plan.elided[b.index()] = true;
    }
    let other_ok = |inner: Value| plan.zero_ext[inner.index()] && matches!(plan.location(inner), Location::Reg(_) | Location::Slot);
    if can_peel_uextend(func, plan, (uses, pos_of), (b_idx, pos, pos, None, res), (other, &other_ok)) {
        plan.elided[other.index()] = true;
    }
    true
}

fn fuse_shifted_alu(func: &Function, plan: &mut Plan<Reg>, info: (&[u32], &[Option<(usize, usize, Inst)>]), (b_idx, pos, res): (usize, usize, Value), (alu_op, x, y): (Binary, Value, Value)) {
    let candidates = if alu_op == Binary::Isub { [Some((y, x)), None] } else { [Some((y, x)), Some((x, y))] };
    for pair in candidates.into_iter().flatten() {
        if try_fuse_shift(func, plan, info, (b_idx, pos, res, alu_op), pair) {
            break;
        }
    }
}

pub(super) fn fuse(func: &Function, plan: &mut Plan<Reg>) {
    let mut uses = vec![0_u32; func.types.len()];
    let mut pos_of = vec![None; func.types.len()];
    for (b_idx, block) in func.blocks.iter().enumerate() {
        for (pos, &inst) in block.insts.iter().enumerate() {
            let data = &func.insts[inst.index()];
            data.op.for_operands(|v| uses[v.index()] += 1);
            if let Some(res) = data.result {
                pos_of[res.index()] = Some((b_idx, pos, inst));
            }
        }
    }
    for (b_idx, block) in func.blocks.iter().enumerate() {
        for (pos, &inst) in block.insts.iter().enumerate() {
            let data = &func.insts[inst.index()];
            let Some(res) = data.result else { continue };
            if plan.elided[res.index()] {
                continue;
            }
            if let Op::Convert(Convert::Ireduce, shr) = data.op {
                fuse_ireduce_ushr(func, plan, (&uses, &pos_of), (b_idx, pos, res), shr);
                continue;
            }
            if let Op::Icmp(_, a, _) = data.op
                && func.ty(a).is_small_int()
                && (1..=2).contains(&uses[res.index()])
                && (1..=uses[res.index()] as usize).all(|k| {
                    block.insts.get(pos + k).is_some_and(|&ni| matches!(func.insts[ni.index()].op, Op::Select(c, x, y) if c == res && x != res && y != res))
                })
            {
                plan.elided[res.index()] = true;
                continue;
            }
            if let Op::Binary(alu_op @ (Binary::Band | Binary::Bor | Binary::Bxor | Binary::Iadd | Binary::Isub), x, y) = data.op
                && matches!(func.ty(res), Type::I32 | Type::I64)
            {
                let mut done = false;
                if alu_op == Binary::Iadd {
                    for flag_val in [y, x] {
                        if uses[flag_val.index()] == 1
                            && !plan.elided[flag_val.index()]
                            && let Some((fb, fpos, finst)) = pos_of[flag_val.index()]
                            && fb == b_idx
                            && fpos < pos
                        {
                            match func.insts[finst.index()].op {
                                Op::Icmp(_, a, b)
                                    if func.ty(a).is_small_int()
                                        && preserved(func, plan, b_idx, fpos, pos, None, a)
                                        && preserved(func, plan, b_idx, fpos, pos, None, b) =>
                                {
                                    plan.elided[flag_val.index()] = true;
                                    done = true;
                                    break;
                                }
                                Op::Convert(Convert::Uextend, inner)
                                    if uses[inner.index()] == 1
                                        && !plan.elided[inner.index()]
                                        && let Some((ib, ipos, iinst)) = pos_of[inner.index()]
                                        && ib == b_idx
                                        && ipos < fpos
                                        && matches!(
                                            func.insts[iinst.index()].op,
                                            Op::Icmp(_, a, b)
                                                if func.ty(a).is_small_int()
                                                    && preserved(func, plan, b_idx, ipos, pos, Some(fpos), a)
                                                    && preserved(func, plan, b_idx, ipos, pos, Some(fpos), b)
                                        ) =>
                                {
                                    plan.elided[flag_val.index()] = true;
                                    plan.elided[inner.index()] = true;
                                    done = true;
                                    break;
                                }
                                _ => {}
                            }
                        }
                    }
                }
                if !done && alu_op == Binary::Iadd && plan.sib[res.index()].is_none() && uses[res.index()] >= 2 {
                    let try_sib = |plan: &mut Plan<Reg>, base: Value, idx: Value, shift: i64, spos: usize, shl_elide: Option<Value>| -> bool {
                        if plan.reg_of(base).is_none() || plan.reg_of(idx).is_none() {
                            return false;
                        }
                        let mut seen_uses = 0_u32;
                        let mut last_pos = pos;
                        for (u_pos, &u_inst) in block.insts.iter().enumerate().skip(pos + 1) {
                            let u_data = &func.insts[u_inst.index()];
                            let ok = match (&u_data.op, u_data.result) {
                                (&Op::Load(addr, 0), Some(r)) if addr == res => {
                                    let ty = func.ty(r);
                                    ty != Type::I128 && (shift == 0 || ty.bytes() == (1 << shift))
                                }
                                (&Op::Store(val, addr, 0), _) if addr == res && val != res => {
                                    let ty = func.ty(val);
                                    ty != Type::I128 && (shift == 0 || ty.bytes() == (1 << shift))
                                }
                                _ => false,
                            };
                            if ok {
                                seen_uses += 1;
                                last_pos = u_pos;
                            }
                        }
                        if seen_uses == uses[res.index()]
                            && preserved(func, plan, b_idx, pos, last_pos, None, base)
                            && preserved(func, plan, b_idx, spos, last_pos, None, idx)
                        {
                            plan.sib[res.index()] = Some(Sib { base: SibBase::Value(base), index: idx, shift: shift as u8, disp: 0 });
                            plan.elided[res.index()] = true;
                            if let Some(shl) = shl_elide
                                && uses[shl.index()] == 1
                                && !plan.elided[shl.index()]
                            {
                                plan.elided[shl.index()] = true;
                            }
                            return true;
                        }
                        false
                    };
                    for (base, shl) in [(x, y), (y, x)] {
                        if let Some((sb, spos, sinst)) = pos_of[shl.index()]
                            && sb == b_idx
                            && spos < pos
                            && let Some((Binary::Ishl, idx, shift)) = shift_parts(&func.insts[sinst.index()].op, plan)
                            && (1..=3).contains(&shift)
                            && try_sib(plan, base, idx, shift, spos, Some(shl))
                        {
                            done = true;
                            break;
                        }
                    }
                    if !done {
                        done = try_sib(plan, x, y, 0, pos, None);
                    }
                }
                if !done {
                    fuse_shifted_alu(func, plan, (&uses, &pos_of), (b_idx, pos, res), (alu_op, x, y));
                }
            }
        }
    }
    if func.size() <= 2000 {
        for (b_idx, block) in func.blocks.iter().enumerate() {
            let Some(&last) = block.insts.last() else { continue };
            let Op::Brif(cond_val, ..) = func.insts[last.index()].op else { continue };
            let pos = block.insts.len() - 1;
            if uses[cond_val.index()] != 1 || plan.elided[cond_val.index()] {
                continue;
            }
            let Some((cb, cpos, cinst)) = pos_of[cond_val.index()] else { continue };
            if cb != b_idx {
                continue;
            }
            match func.insts[cinst.index()].op {
                Op::Binary(Binary::Band, x, y) if cpos + 1 == pos => {
                    if single_bit_mask(func, plan, x, y).is_some() {
                        plan.elided[cond_val.index()] = true;
                    }
                }
                Op::Icmp(IntCC::Equal | IntCC::NotEqual, lhs, zero) => {
                    let masked = if plan.const_of(zero) == Some(0) {
                        Some(lhs)
                    } else if plan.const_of(lhs) == Some(0) {
                        Some(zero)
                    } else {
                        None
                    };
                    if let Some(m) = masked
                        && uses[m.index()] == 1
                        && !plan.elided[m.index()]
                        && let Some((mb, mpos, minst)) = pos_of[m.index()]
                        && mb == b_idx
                        && mpos + 1 == cpos
                        && cpos + 1 == pos
                        && let Op::Binary(Binary::Band, x, y) = func.insts[minst.index()].op
                        && single_bit_mask(func, plan, x, y).is_some()
                    {
                        plan.elided[cond_val.index()] = true;
                        plan.elided[m.index()] = true;
                    }
                }
                _ => {}
            }
        }
    }
}

pub(super) fn single_bit_mask(func: &Function, plan: &Plan<Reg>, x: Value, y: Value) -> Option<(Value, u8)> {
    for (val, mask_val) in [(x, y), (y, x)] {
        let ty = func.ty(val);
        if !ty.is_small_int() {
            continue;
        }
        if let Some(imm) = plan.const_of(mask_val) {
            let bits = ty.bits();
            let u = (imm as u64) & if bits == 64 { u64::MAX } else { (1_u64 << bits) - 1 };
            if u.is_power_of_two() {
                return Some((val, u.trailing_zeros() as u8));
            }
        }
    }
    None
}

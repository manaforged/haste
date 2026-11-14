mod mem;

use super::asm::{Alu, ExtOp, Reg, Shift};
use crate::alloc::{Location, Plan};
use crate::ir::{Binary, Convert, Function, Inst, IntCC, Op, Type, Unary, Value};
pub(super) use mem::madd_parts;

pub(super) fn const_of(func: &Function, plan: &Plan<Reg>, val: Value, def: impl Fn(Value) -> Option<Inst>) -> Option<i64> {
    plan.const_of(val).or_else(|| match func.insts[def(val)?.index()].op {
        Op::Const(imm) => Some(imm),
        _ => None,
    })
}

fn zero_ext_loc(plan: &Plan<Reg>, val: Value) -> bool {
    plan.zero_ext[val.index()] && matches!(plan.location(val), Location::Reg(_) | Location::Slot)
}

fn local_single(
    plan: &Plan<Reg>,
    (uses, pos_of): (&[u32], &[Option<(usize, usize, Inst)>]),
    b_idx: usize,
    limit: usize,
    val: Value,
) -> Option<(usize, Inst)> {
    (uses[val.index()] == 1 && !plan.elided[val.index()])
        .then(|| pos_of[val.index()].filter(|&(b, p, _)| b == b_idx && p < limit).map(|(_, p, i)| (p, i)))
        .flatten()
}

fn shift_uext_ok(func: &Function, plan: &Plan<Reg>, ty_bits: u32, (sop, imm): (Binary, i64), inner: Value) -> bool {
    func.ty(inner).is_small_int()
        && (zero_ext_loc(plan, inner) || (sop == Binary::Ishl && ((imm as u32) & (ty_bits - 1)) >= ty_bits - func.ty(inner).bits()))
}

pub(super) fn alu_shift(logical: bool, ty: Type, op: Binary, imm: i64) -> Option<(Shift, u8)> {
    if !matches!(ty, Type::I32 | Type::I64) {
        return None;
    }
    let bits = ty.bits();
    let mask = bits - 1;
    let amt = (imm as u32) & mask;
    let (shift, count) = match op {
        Binary::Ishl => (Shift::Lsl, amt),
        Binary::Ushr => (Shift::Lsr, amt),
        Binary::Sshr => (Shift::Asr, amt),
        Binary::Rotr if logical => (Shift::Ror, amt),
        Binary::Rotl if logical => (Shift::Ror, (bits - amt) & mask),
        _ => return None,
    };
    Some((shift, count as u8))
}

fn ext_op_of(dst_ty: Type, src_ty: Type, signed: bool) -> Option<ExtOp> {
    match (dst_ty, src_ty, signed) {
        (Type::I64 | Type::I32, Type::I8, false) => Some(ExtOp::Uxtb),
        (Type::I64 | Type::I32, Type::I16, false) => Some(ExtOp::Uxth),
        (Type::I64, Type::I32, false) => Some(ExtOp::Uxtw),
        (Type::I64 | Type::I32, Type::I8, true) => Some(ExtOp::Sxtb),
        (Type::I64 | Type::I32, Type::I16, true) => Some(ExtOp::Sxth),
        (Type::I64, Type::I32, true) => Some(ExtOp::Sxtw),
        _ => None,
    }
}

pub(super) fn ext_parts(func: &Function, plan: &Plan<Reg>, dst_ty: Type, val: Value, def: impl Fn(Value) -> Option<Inst>) -> Option<(ExtOp, u8, Value)> {
    if !matches!(dst_ty, Type::I32 | Type::I64) || !plan.elided[val.index()] {
        return None;
    }
    let op = &func.insts[def(val)?.index()].op;
    if let Op::Convert(conv @ (Convert::Uextend | Convert::Sextend), inner) = *op {
        return ext_op_of(dst_ty, func.ty(inner), conv == Convert::Sextend).map(|ext| (ext, 0, inner));
    }
    let (Binary::Ishl, mid, imm) = shift_parts(op, plan)? else { return None };
    let shift = (imm as u32) & (dst_ty.bits() - 1);
    if shift > 4 || !plan.elided[mid.index()] {
        return None;
    }
    let Op::Convert(conv @ (Convert::Uextend | Convert::Sextend), inner) = func.insts[def(mid)?.index()].op else {
        return None;
    };
    ext_op_of(dst_ty, func.ty(inner), conv == Convert::Sextend).map(|ext| (ext, shift as u8, inner))
}

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

pub(super) fn extr_parts(
    func: &Function,
    plan: &Plan<Reg>,
    ty: Type,
    x: Value,
    y: Value,
    def: impl Fn(Value) -> Option<Inst>,
) -> Option<(Value, Value, u8)> {
    if !matches!(ty, Type::I32 | Type::I64) {
        return None;
    }
    let bits = ty.bits();
    let (sx, a, ax) = shift_parts(&func.insts[def(x)?.index()].op, plan)?;
    let (sy, b, ay) = shift_parts(&func.insts[def(y)?.index()].op, plan)?;
    let (hi, l_amt, lo, r_amt) = match (sx, sy) {
        (Binary::Ishl, Binary::Ushr) => (a, ax, b, ay),
        (Binary::Ushr, Binary::Ishl) => (b, ay, a, ax),
        _ => return None,
    };
    let kl = (l_amt as u32) & (bits - 1);
    let kr = (r_amt as u32) & (bits - 1);
    (kl > 0 && kr > 0 && kl + kr == bits).then_some((hi, lo, kr as u8))
}

fn try_fuse_extr(
    func: &Function,
    plan: &mut Plan<Reg>,
    uses: &[u32],
    pos_of: &[Option<(usize, usize, Inst)>],
    (b_idx, pos, res): (usize, usize, Value),
    (op, x, y): (Binary, Value, Value),
) -> bool {
    if !matches!(op, Binary::Bor | Binary::Bxor | Binary::Iadd)
        || uses[x.index()] != 1
        || uses[y.index()] != 1
        || plan.elided[x.index()]
        || plan.elided[y.index()]
    {
        return false;
    }
    let Some((xb, xpos, xinst)) = pos_of[x.index()] else { return false };
    let Some((yb, ypos, yinst)) = pos_of[y.index()] else { return false };
    if xb != b_idx || yb != b_idx || xpos >= pos || ypos >= pos {
        return false;
    }
    let def = |v: Value| if v == x { Some(xinst) } else if v == y { Some(yinst) } else { None };
    let Some((hi, lo, kr)) = extr_parts(func, plan, func.ty(res), x, y, def) else {
        return false;
    };
    let (lpos, rpos) = if matches!(shift_parts(&func.insts[xinst.index()].op, plan), Some((Binary::Ishl, ..))) {
        (xpos, ypos)
    } else {
        (ypos, xpos)
    };
    let hi_ok = if plan.elided[hi.index()]
        && let Some((hb, hpos, hinst)) = pos_of[hi.index()]
        && hb == b_idx && uses[hi.index()] == 1
        && let Op::Convert(Convert::Uextend, inner) = func.insts[hinst.index()].op
        && u32::from(kr) <= func.ty(inner).bits()
    {
        preserved(func, plan, b_idx, hpos, pos, Some(rpos), inner)
    } else {
        preserved(func, plan, b_idx, lpos, pos, Some(rpos), hi)
    };
    if !hi_ok || !preserved(func, plan, b_idx, rpos, pos, Some(lpos), lo) {
        return false;
    }
    plan.elided[x.index()] = true;
    plan.elided[y.index()] = true;
    true
}

fn can_elide_band(func: &Function, plan: &Plan<Reg>, b_idx: usize, (mpos, end_pos, skip_pos): (usize, usize, Option<usize>), (x, y): (Value, Value)) -> bool {
    if let Some((val, _)) = single_bit_mask(func, plan, x, y) {
        return preserved(func, plan, b_idx, mpos, end_pos, skip_pos, val);
    }
    func.ty(x).is_small_int()
        && preserved(func, plan, b_idx, mpos, end_pos, skip_pos, x)
        && preserved(func, plan, b_idx, mpos, end_pos, skip_pos, y)
}

fn zero_cmp_band(
    func: &Function,
    plan: &Plan<Reg>,
    uses: &[u32],
    pos_of: &[Option<(usize, usize, Inst)>],
    b_idx: usize,
    lhs: Value,
    rhs: Value,
) -> Option<(Value, usize, Value, Value)> {
    let masked = (plan.const_of(rhs) == Some(0)).then_some(lhs).or_else(|| (plan.const_of(lhs) == Some(0)).then_some(rhs))?;
    if uses[masked.index()] != 1 || plan.elided[masked.index()] {
        return None;
    }
    let (mb, mpos, minst) = pos_of[masked.index()]?;
    if mb != b_idx {
        return None;
    }
    let Op::Binary(Binary::Band, x, y) = func.insts[minst.index()].op else {
        return None;
    };
    Some((masked, mpos, x, y))
}

fn preserved(func: &Function, plan: &Plan<Reg>, block_idx: usize, start: usize, end: usize, skip_pos: Option<usize>, val: Value) -> bool {
    if plan.elided[val.index()] {
        return false;
    }
    if start + 1 >= end || plan.location(val).is_static() {
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
    if uses[val.index()] != 1 || (plan.elided[val.index()] && skip.is_none()) {
        return false;
    }
    let Some((ub, upos, uinst)) = pos_of[val.index()] else { return false };
    if ub != b_idx || upos >= limit {
        return false;
    }
    let Op::Convert(Convert::Uextend, inner) = func.insts[uinst.index()].op else { return false };
    inner != res && ok(inner) && preserved(func, plan, b_idx, upos, end, skip, inner)
}

fn peel_other_uextend(
    func: &Function,
    plan: &mut Plan<Reg>,
    info: (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos, res): (usize, usize, Value),
    other: Value,
) {
    let other_ok = |inner: Value| zero_ext_loc(plan, inner);
    if can_peel_uextend(func, plan, info, (b_idx, pos, pos, None, res), (other, &other_ok)) {
        plan.elided[other.index()] = true;
    }
}

fn bnot_alu(op: Binary) -> Option<Alu> {
    match op {
        Binary::Band => Some(Alu::Bic),
        Binary::Bor => Some(Alu::Orn),
        Binary::Bxor => Some(Alu::Eon),
        _ => None,
    }
}

fn bfx_match(
    func: &Function,
    plan: &Plan<Reg>,
    ty: Type,
    op: Binary,
    (x, y): (Value, Value),
    def: impl Fn(Value) -> Option<Inst>,
    mid_ok: impl Fn(Value) -> bool,
) -> Option<(bool, bool, Value, Value, u8, u8)> {
    if !matches!(ty, Type::I32 | Type::I64) { return None; }
    let bits = ty.bits();
    let ty_mask = if bits == 64 { u64::MAX } else { (1_u64 << bits) - 1 };
    let low_mask_width = |m: u64| (m != 0 && m != ty_mask && (m & (m + 1)) == 0).then(|| m.trailing_ones());
    match op {
        Binary::Band => {
            for (mid, m_val) in [(x, y), (y, x)] {
                if !mid_ok(mid) { continue; }
                let Some(m) = const_of(func, plan, m_val, &def).map(|imm| (imm as u64) & ty_mask) else { continue };
                match func.insts[def(mid)?.index()].op {
                    Op::Binary(sop @ (Binary::Ushr | Binary::Sshr), src, amt_val) => {
                        if let Some(w) = low_mask_width(m) && let Some(amt) = const_of(func, plan, amt_val, &def) {
                            let lsb = (amt as u32) & (bits - 1);
                            if sop == Binary::Ushr || w <= bits - lsb {
                                return Some((false, false, mid, src, lsb as u8, w.min(bits - lsb) as u8));
                            }
                        }
                    }
                    Op::Binary(Binary::Ishl, src, amt_val) => {
                        if let Some(amt) = const_of(func, plan, amt_val, &def) {
                            let lsb = (amt as u32) & (bits - 1);
                            let eff = (m | ((1_u64 << lsb) - 1)) & ty_mask;
                            let top = if eff == ty_mask { Some(bits) } else { low_mask_width(eff) };
                            if lsb > 0 && let Some(top) = top && top > lsb {
                                return Some((true, false, mid, src, lsb as u8, (top - lsb) as u8));
                            }
                        }
                    }
                    _ => {}
                }
            }
            None
        }
        Binary::Ushr | Binary::Sshr => {
            if !mid_ok(x) { return None; }
            let shr_amt = (const_of(func, plan, y, &def)? as u32) & (bits - 1);
            match func.insts[def(x)?.index()].op {
                Op::Binary(Binary::Band, a, b) if op == Binary::Ushr => [(a, b), (b, a)].into_iter().find_map(|(src, m_val)| {
                    let imm = const_of(func, plan, m_val, &def)?;
                    let width = low_mask_width(((imm as u64) & ty_mask) >> shr_amt)?;
                    Some((false, false, x, src, shr_amt as u8, width as u8))
                }),
                Op::Binary(Binary::Ishl, src, l_amt_val) => {
                    let top = (const_of(func, plan, l_amt_val, &def)? as u32) & (bits - 1);
                    (shr_amt >= top).then(|| (false, op == Binary::Sshr, x, src, (shr_amt - top) as u8, (bits - shr_amt) as u8))
                }
                Op::Convert(Convert::Sextend, inner) if op == Binary::Sshr && func.ty(inner).is_small_int() && func.ty(inner).bits() < bits => {
                    let ib = func.ty(inner).bits();
                    let lsb = shr_amt.min(ib - 1);
                    Some((false, true, x, inner, lsb as u8, (ib - lsb) as u8))
                }
                _ => None,
            }
        }
        Binary::Ishl => {
            if !mid_ok(x) { return None; }
            let shl_amt = (const_of(func, plan, y, &def)? as u32) & (bits - 1);
            if shl_amt == 0 { return None; }
            match func.insts[def(x)?.index()].op {
                Op::Binary(Binary::Band, a, b) => [(a, b), (b, a)].into_iter().find_map(|(src, m_val)| {
                    let imm = const_of(func, plan, m_val, &def)?;
                    let w = low_mask_width((imm as u64) & ty_mask)?;
                    Some((true, false, x, src, shl_amt as u8, w.min(bits - shl_amt) as u8))
                }),
                Op::Convert(conv @ (Convert::Uextend | Convert::Sextend), inner) if func.ty(inner).is_small_int() && func.ty(inner).bits() < bits => {
                    Some((true, conv == Convert::Sextend, x, inner, shl_amt as u8, func.ty(inner).bits().min(bits - shl_amt) as u8))
                }
                _ => None,
            }
        }
        _ => None,
    }
}

pub(super) fn bfx_parts(func: &Function, plan: &Plan<Reg>, ty: Type, op: Binary, x: Value, y: Value, def: impl Fn(Value) -> Option<Inst>) -> Option<(bool, bool, Value, u8, u8)> {
    let (insert, signed, _, src, lsb, width) = bfx_match(func, plan, ty, op, (x, y), def, |v| plan.elided[v.index()])?;
    Some((insert, signed, src, lsb, width))
}

fn try_fuse_bfx(func: &Function, plan: &mut Plan<Reg>, (uses, pos_of): (&[u32], &[Option<(usize, usize, Inst)>]), (b_idx, pos, res): (usize, usize, Value), (op, x, y): (Binary, Value, Value)) -> bool {
    let def = |v: Value| pos_of[v.index()].map(|(_, _, i)| i);
    let mid_ok = |v: Value| uses[v.index()] == 1 && !plan.elided[v.index()] && pos_of[v.index()].is_some_and(|(sb, spos, _)| sb == b_idx && spos < pos);
    let Some((insert, _, mid, src, lsb, width)) = bfx_match(func, plan, func.ty(res), op, (x, y), def, mid_ok) else {
        return false;
    };
    let Some((_, spos, min_inst)) = pos_of[mid.index()] else { return false };
    let is_conv = matches!(func.insts[min_inst.index()].op, Op::Convert(..));
    let need = if insert { u32::from(width) } else { u32::from(lsb) + u32::from(width) };
    let ext_ok = |inner: Value| need <= func.ty(inner).bits() || zero_ext_loc(plan, inner);
    let peeled = !is_conv && can_peel_uextend(func, plan, (uses, pos_of), (b_idx, spos, pos, Some(spos), res), (src, &ext_ok));
    if !peeled && !preserved(func, plan, b_idx, spos, pos, None, src) {
        return false;
    }
    plan.elided[mid.index()] = true;
    if peeled { plan.elided[src.index()] = true; }
    true
}

pub(super) fn bnot_parts(
    func: &Function,
    plan: &Plan<Reg>,
    ty: Type,
    op: Binary,
    x: Value,
    y: Value,
    def: impl Fn(Value) -> Option<Inst>,
) -> Option<(Alu, Value, Value, Option<(Shift, u8)>)> {
    let not_alu = bnot_alu(op)?;
    if !matches!(ty, Type::I32 | Type::I64) {
        return None;
    }
    for (other, neg) in [(x, y), (y, x)] {
        if plan.elided[neg.index()]
            && let Some(ninst) = def(neg)
            && let Op::Unary(Unary::Bnot, inner) = func.insts[ninst.index()].op
        {
            if plan.elided[inner.index()]
                && let Some(sinst) = def(inner)
                && let Some((sop, b, imm)) = shift_parts(&func.insts[sinst.index()].op, plan)
                && let Some((shift, count)) = alu_shift(true, ty, sop, imm)
            {
                return Some((not_alu, other, b, Some((shift, count))));
            }
            return Some((not_alu, other, inner, None));
        }
    }
    None
}

fn try_fuse_bnot(
    func: &Function,
    plan: &mut Plan<Reg>,
    info: (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos, res): (usize, usize, Value),
    (op, x, y): (Binary, Value, Value),
) -> bool {
    if bnot_alu(op).is_none() {
        return false;
    }
    let (ty, ty_bits) = (func.ty(res), func.ty(res).bits());
    for (other, neg) in [(x, y), (y, x)] {
        let Some((npos, ninst)) = local_single(plan, info, b_idx, pos, neg) else { continue };
        let Op::Unary(Unary::Bnot, inner) = func.insts[ninst.index()].op else { continue };
        if let Some((spos, sinst)) = local_single(plan, info, b_idx, npos, inner)
            && let Some((sop, b, imm)) = shift_parts(&func.insts[sinst.index()].op, plan)
            && alu_shift(true, ty, sop, imm).is_some()
            && b != res
        {
            let peel_b = local_single(plan, info, b_idx, spos, b).is_some_and(|(upos, uinst)| {
                matches!(
                    func.insts[uinst.index()].op,
                    Op::Convert(Convert::Uextend, u_inner)
                        if u_inner != res
                            && shift_uext_ok(func, plan, ty_bits, (sop, imm), u_inner)
                            && preserved(func, plan, b_idx, upos, npos, Some(spos), u_inner)
                            && preserved(func, plan, b_idx, npos, pos, None, u_inner)
                )
            });
            if peel_b || preserved(func, plan, b_idx, spos, pos, Some(npos), b) {
                plan.elided[neg.index()] = true;
                plan.elided[inner.index()] = true;
                if peel_b { plan.elided[b.index()] = true; }
                peel_other_uextend(func, plan, info, (b_idx, pos, res), other);
                return true;
            }
        }
        if preserved(func, plan, b_idx, npos, pos, None, inner) {
            plan.elided[neg.index()] = true;
            peel_other_uextend(func, plan, info, (b_idx, pos, res), other);
            return true;
        }
    }
    false
}

pub(super) fn elided_use_end(func: &Function, plan: &Plan<Reg>, insts: &[Inst], pos: usize, mut target: Value) -> Option<usize> {
    let mut last = None;
    for (u_pos, &ui) in insts.iter().enumerate().skip(pos + 1) {
        let ud = &func.insts[ui.index()];
        let mut used = false;
        ud.op.for_operands(|v| used |= v == target);
        if !used { continue; }
        if matches!(ud.op, Op::Convert(Convert::Uextend, _)) && let Some(ur) = ud.result && plan.elided[ur.index()] { target = ur; } else { last = Some(u_pos); }
    }
    last
}

pub(super) fn eon_inner(func: &Function, plan: &Plan<Reg>, x: Value, def: impl Fn(Value) -> Option<Inst>) -> Option<(Value, Value)> {
    if !plan.elided[x.index()] { return None; }
    let Op::Binary(Binary::Bxor, a, b) = func.insts[def(x)?.index()].op else { return None };
    Some((a, b))
}

fn try_fuse_eon(func: &Function, plan: &mut Plan<Reg>, info @ (uses, pos_of): (&[u32], &[Option<(usize, usize, Inst)>]), (b_idx, pos, res): (usize, usize, Value), inner: Value) -> bool {
    let ty = func.ty(res);
    if !matches!(ty, Type::I32 | Type::I64) { return false; }
    let Some((ipos, iinst)) = local_single(plan, info, b_idx, pos, inner) else { return false };
    let Op::Binary(Binary::Bxor, a, b) = func.insts[iinst.index()].op else { return false };
    let shift_ok = [(a, b), (b, a)].into_iter().any(|(other, cand)| {
        plan.elided[cand.index()] && uses[cand.index()] == 1 && preserved(func, plan, b_idx, ipos, pos, None, other)
            && pos_of[cand.index()].is_some_and(|(sb, spos, sinst)| {
                sb == b_idx && spos < ipos && shift_parts(&func.insts[sinst.index()].op, plan)
                    .is_some_and(|(sop, src, imm)| alu_shift(true, ty, sop, imm).is_some() && preserved(func, plan, b_idx, spos, pos, Some(ipos), src))
            })
    });
    if (!preserved(func, plan, b_idx, ipos, pos, None, a) || !preserved(func, plan, b_idx, ipos, pos, None, b)) && !shift_ok { return false; }
    plan.elided[inner.index()] = true;
    true
}

fn try_fuse_shift(
    func: &Function,
    plan: &mut Plan<Reg>,
    info: (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos, res, alu_op): (usize, usize, Value, Binary),
    (shift_val, other): (Value, Value),
) -> bool {
    let Some((spos, sinst)) = local_single(plan, info, b_idx, pos, shift_val) else { return false };
    let Some((sop, b, imm)) = shift_parts(&func.insts[sinst.index()].op, plan) else { return false };
    let logical = matches!(alu_op, Binary::Band | Binary::Bor | Binary::Bxor);
    if b == res || alu_shift(logical, func.ty(res), sop, imm).is_none() {
        return false;
    }
    let ty_bits = func.ty(res).bits();
    let shift_ok = |inner: Value| shift_uext_ok(func, plan, ty_bits, (sop, imm), inner);
    let peeled = can_peel_uextend(func, plan, info, (b_idx, spos, pos, Some(spos), res), (b, &shift_ok));
    if !peeled && !preserved(func, plan, b_idx, spos, pos, None, b) {
        return false;
    }
    plan.elided[shift_val.index()] = true;
    if peeled { plan.elided[b.index()] = true; }
    peel_other_uextend(func, plan, info, (b_idx, pos, res), other);
    true
}

fn fuse_shifted_alu(func: &Function, plan: &mut Plan<Reg>, info: (&[u32], &[Option<(usize, usize, Inst)>]), (b_idx, pos, res): (usize, usize, Value), (alu_op, x, y): (Binary, Value, Value)) {
    let candidates = if alu_op == Binary::Isub { [Some((y, x)), None] } else { [Some((y, x)), Some((x, y))] };
    for pair in candidates.into_iter().flatten() {
        if try_fuse_shift(func, plan, info, (b_idx, pos, res, alu_op), pair) { break; }
    }
}

fn fuse_shift_mask(func: &Function, plan: &mut Plan<Reg>, info @ (_, pos_of): (&[u32], &[Option<(usize, usize, Inst)>]), (b_idx, pos): (usize, usize), x: Value, y: Value) {
    let ty = func.ty(x);
    if !matches!(ty, Type::I32 | Type::I64) { return; }
    let Some((ypos, yinst)) = local_single(plan, info, b_idx, pos, y) else { return };
    let Op::Binary(Binary::Band, a, b) = func.insts[yinst.index()].op else { return };
    let (mask, def) = (u64::from(ty.bits() - 1), |v: Value| pos_of[v.index()].map(|(_, _, i)| i));
    for (raw, m) in [(a, b), (b, a)] {
        if const_of(func, plan, m, def).is_some_and(|imm| (imm as u64 & mask) == mask) && preserved(func, plan, b_idx, ypos, pos, None, raw) {
            plan.elided[y.index()] = true;
            return;
        }
    }
}

fn try_fuse_ext_operand(func: &Function, plan: &mut Plan<Reg>, info: (&[u32], &[Option<(usize, usize, Inst)>]), (b_idx, pos, dst_ty): (usize, usize, Type), ext_val: Value) -> bool {
    let Some((epos, einst)) = local_single(plan, info, b_idx, pos, ext_val) else { return false };
    let (mid, spos, ext_pos, inner, signed) = match func.insts[einst.index()].op {
        Op::Convert(conv @ (Convert::Uextend | Convert::Sextend), inner) => (None, None, epos, inner, conv == Convert::Sextend),
        ref op => {
            let Some((Binary::Ishl, mid, imm)) = shift_parts(op, plan) else { return false };
            if ((imm as u32) & (dst_ty.bits() - 1)) > 4 { return false; }
            let Some((upos, uinst)) = local_single(plan, info, b_idx, epos, mid) else { return false };
            let Op::Convert(conv @ (Convert::Uextend | Convert::Sextend), inner) = func.insts[uinst.index()].op else { return false };
            (Some(mid), Some(epos), upos, inner, conv == Convert::Sextend)
        }
    };
    if ext_op_of(dst_ty, func.ty(inner), signed).is_none() || !preserved(func, plan, b_idx, ext_pos, pos, spos, inner) {
        return false;
    }
    plan.elided[ext_val.index()] = true;
    if let Some(m) = mid { plan.elided[m.index()] = true; }
    true
}

fn cmp_operand_preserved(func: &Function, plan: &Plan<Reg>, pos_of: &[Option<(usize, usize, Inst)>], (b_idx, start, pos, skip): (usize, usize, usize, Option<usize>), (cmp_ty, v): (Type, Value)) -> bool {
    let actual = if plan.elided[v.index()] {
        let Some((_, _, inner)) = ext_parts(func, plan, cmp_ty, v, |val| pos_of[val.index()].map(|(_, _, i)| i)) else { return false };
        inner
    } else {
        v
    };
    preserved(func, plan, b_idx, start, pos, skip, actual)
}

fn try_fuse_flag_operand(func: &Function, plan: &mut Plan<Reg>, info @ (_, pos_of): (&[u32], &[Option<(usize, usize, Inst)>]), (b_idx, pos): (usize, usize), flag_val: Value) -> bool {
    let Some((fpos, finst)) = local_single(plan, info, b_idx, pos, flag_val) else { return false };
    let cmp_ok = |start, skip, a, b| {
        func.ty(a).is_small_int()
            && cmp_operand_preserved(func, plan, pos_of, (b_idx, start, pos, skip), (func.ty(a), a))
            && cmp_operand_preserved(func, plan, pos_of, (b_idx, start, pos, skip), (func.ty(a), b))
    };
    match func.insts[finst.index()].op {
        Op::Icmp(_, a, b) if cmp_ok(fpos, None, a, b) => { plan.elided[flag_val.index()] = true; true }
        Op::Convert(Convert::Uextend, inner)
            if let Some((ipos, iinst)) = local_single(plan, info, b_idx, fpos, inner)
                && matches!(func.insts[iinst.index()].op, Op::Icmp(_, a, b) if cmp_ok(ipos, Some(fpos), a, b)) =>
        {
            plan.elided[flag_val.index()] = true;
            plan.elided[inner.index()] = true;
            true
        }
        _ => false,
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
            if let Op::Binary(Binary::Ishl | Binary::Ushr | Binary::Sshr | Binary::Rotl | Binary::Rotr, x, y) = data.op {
                fuse_shift_mask(func, plan, (&uses, &pos_of), (b_idx, pos), x, y);
            }
            if let Op::Unary(Unary::Bnot, inner) = data.op {
                if !try_fuse_eon(func, plan, (&uses, &pos_of), (b_idx, pos, res), inner)
                    && matches!(func.ty(res), Type::I32 | Type::I64)
                {
                    try_fuse_shift(func, plan, (&uses, &pos_of), (b_idx, pos, res, Binary::Bxor), (inner, res));
                }
                continue;
            }
            if let Op::Unary(Unary::Ineg, inner) = data.op
                && matches!(func.ty(res), Type::I32 | Type::I64)
            {
                if !try_fuse_flag_operand(func, plan, (&uses, &pos_of), (b_idx, pos), inner) {
                    try_fuse_shift(func, plan, (&uses, &pos_of), (b_idx, pos, res, Binary::Isub), (inner, res));
                }
                continue;
            }
            if let Op::Binary(op @ (Binary::Ishl | Binary::Ushr | Binary::Sshr), x, y) = data.op
                && matches!(func.ty(res), Type::I32 | Type::I64)
            {
                try_fuse_bfx(func, plan, (&uses, &pos_of), (b_idx, pos, res), (op, x, y));
                continue;
            }
            if let Op::Convert(Convert::Ireduce, shr) = data.op {
                fuse_ireduce_ushr(func, plan, (&uses, &pos_of), (b_idx, pos, res), shr);
                continue;
            }
            if let Op::Icmp(cc, a, b) = data.op
                && func.ty(a).is_small_int()
            {
                let mut select_elided = false;
                if (1..=2).contains(&uses[res.index()]) {
                    let mut found = 0_u32;
                    let mut last_sel = pos;
                    for (s_pos, &ni) in block.insts.iter().enumerate().skip(pos + 1) {
                        if matches!(func.insts[ni.index()].op, Op::Select(c, x, y) if c == res && x != res && y != res) {
                            found += 1;
                            last_sel = s_pos;
                            if found == uses[res.index()] {
                                break;
                            }
                        }
                    }
                    if found == uses[res.index()]
                        && preserved(func, plan, b_idx, pos, last_sel, None, a)
                        && preserved(func, plan, b_idx, pos, last_sel, None, b)
                    {
                        plan.elided[res.index()] = true;
                        select_elided = true;
                    }
                }
                if !select_elided && matches!(func.ty(a), Type::I32 | Type::I64) {
                    let b_imm = plan
                        .const_of(b)
                        .and_then(|imm| super::int::add_imm_parts(true, crate::fold::extend(imm, func.ty(b), cc.is_signed())))
                        .is_some();
                    if !b_imm && !try_fuse_ext_operand(func, plan, (&uses, &pos_of), (b_idx, pos, func.ty(a)), b) {
                        try_fuse_ext_operand(func, plan, (&uses, &pos_of), (b_idx, pos, func.ty(a)), a);
                    }
                }
                continue;
            }
            if let Op::Binary(alu_op @ (Binary::Band | Binary::Bor | Binary::Bxor | Binary::Iadd | Binary::Isub), x, y) = data.op
                && matches!(func.ty(res), Type::I32 | Type::I64)
            {
                let mut done = false;
                if alu_op == Binary::Iadd {
                    for flag_val in [y, x] {
                        if try_fuse_flag_operand(func, plan, (&uses, &pos_of), (b_idx, pos), flag_val) {
                            done = true;
                            break;
                        }
                    }
                } else if alu_op == Binary::Isub && plan.const_of(x) == Some(0) {
                    done = try_fuse_flag_operand(func, plan, (&uses, &pos_of), (b_idx, pos), y);
                }
                if !done && alu_op == Binary::Iadd {
                    done = mem::try_fuse_sib(func, plan, (&uses, &pos_of), (b_idx, pos, res), (x, y));
                }
                if !done { done = try_fuse_extr(func, plan, &uses, &pos_of, (b_idx, pos, res), (alu_op, x, y)); }
                if !done && matches!(alu_op, Binary::Iadd | Binary::Isub) {
                    done = mem::try_fuse_madd(func, plan, (&uses, &pos_of), (b_idx, pos, res), (alu_op, x, y));
                }
                if !done && alu_op == Binary::Band { done = try_fuse_bfx(func, plan, (&uses, &pos_of), (b_idx, pos, res), (alu_op, x, y)); }
                if !done { done = try_fuse_bnot(func, plan, (&uses, &pos_of), (b_idx, pos, res), (alu_op, x, y)); }
                if !done && matches!(alu_op, Binary::Iadd | Binary::Isub) {
                    done = try_fuse_ext_operand(func, plan, (&uses, &pos_of), (b_idx, pos, func.ty(res)), y)
                        || (alu_op == Binary::Iadd && try_fuse_ext_operand(func, plan, (&uses, &pos_of), (b_idx, pos, func.ty(res)), x));
                }
                if !done { fuse_shifted_alu(func, plan, (&uses, &pos_of), (b_idx, pos, res), (alu_op, x, y)); }
            }
        }
    }
    if func.size() <= 2000 {
        let (mut branch_tst, mut cmp_tst) = (Vec::new(), Vec::new());
        for (b_idx, block) in func.blocks.iter().enumerate() {
            for (pos, &inst) in block.insts.iter().enumerate() {
                match func.insts[inst.index()].op {
                    Op::Brif(cond_val, ..) | Op::Trapz(cond_val) => {
                        let Some((cpos, cinst)) = local_single(plan, (&uses, &pos_of), b_idx, pos, cond_val) else { continue };
                        match func.insts[cinst.index()].op {
                            Op::Binary(Binary::Band, x, y) if can_elide_band(func, plan, b_idx, (cpos, pos, None), (x, y)) => branch_tst.push((cond_val, None)),
                            Op::Icmp(IntCC::Equal | IntCC::NotEqual, lhs, rhs) => {
                                if let Some((m, mpos, x, y)) = zero_cmp_band(func, plan, &uses, &pos_of, b_idx, lhs, rhs)
                                    && mpos < cpos && can_elide_band(func, plan, b_idx, (mpos, pos, Some(cpos)), (x, y))
                                {
                                    branch_tst.push((cond_val, Some(m)));
                                }
                            }
                            _ => {}
                        }
                    }
                    Op::Icmp(IntCC::Equal | IntCC::NotEqual, lhs, rhs) => {
                        if let Some(res) = func.insts[inst.index()].result
                            && let Some(end_pos) = if !plan.elided[res.index()] { Some(pos) } else { elided_use_end(func, plan, &block.insts, pos, res) }
                        {
                            cmp_tst.push((b_idx, pos, end_pos, lhs, rhs));
                        }
                    }
                    _ => {}
                }
            }
        }
        for (cond_val, m) in branch_tst {
            plan.elided[cond_val.index()] = true;
            if let Some(m) = m { plan.elided[m.index()] = true; }
        }
        for (b_idx, cpos, end_pos, lhs, rhs) in cmp_tst {
            if let Some((m, mpos, x, y)) = zero_cmp_band(func, plan, &uses, &pos_of, b_idx, lhs, rhs)
                && mpos < cpos && can_elide_band(func, plan, b_idx, (mpos, end_pos, None), (x, y))
            {
                plan.elided[m.index()] = true;
            }
        }
    }
}

pub(super) fn single_bit_mask(func: &Function, plan: &Plan<Reg>, x: Value, y: Value) -> Option<(Value, u8)> {
    for (val, mask_val) in [(x, y), (y, x)] {
        let ty = func.ty(val);
        if ty.is_small_int() && let Some(imm) = plan.const_of(mask_val) {
            let u = (imm as u64) & ty.unsigned_max();
            if u.is_power_of_two() {
                return Some((val, u.trailing_zeros() as u8));
            }
        }
    }
    None
}

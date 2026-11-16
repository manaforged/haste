mod bfi;
mod bits;
mod ccmp;
mod flag;
mod mem;
mod shift;
mod sign;

use super::asm::{Alu, ExtOp, Reg, Shift};
use crate::alloc::{Location, Plan};
use crate::ir::{Binary, Convert, Function, Inst, IntCC, Op, Type, Unary, Value};
pub(super) use bfi::bfi_parts;
pub(super) use bits::{Bfx, bfx_parts, conv_bfx_parts, extr_parts, single_bit_mask, unwrap_elided_tst};
use flag::{fuse_fits, fuse_icmp_flags, fuse_icmp_selects, try_fuse_flag_operand, try_fuse_icmp_select};
pub(super) use mem::{encode_neon_movi, madd_parts, store_source};
use mem::try_fuse_vec_op;
pub(super) use sign::sign_bit_test;

pub(super) fn def_entry(pos_of: &[Option<(usize, usize, Inst)>], val: Value) -> Option<(usize, usize, Inst)> {
    pos_of.get(val.index()).and_then(|&entry| entry)
}

pub(super) fn def_inst(pos_of: &[Option<(usize, usize, Inst)>], val: Value) -> Option<Inst> {
    def_entry(pos_of, val).map(|(_, _, inst)| inst)
}

pub(super) fn const_of(func: &Function, plan: &Plan<Reg>, val: Value, def: impl Fn(Value) -> Option<Inst>) -> Option<i64> {
    plan.const_of(val).or_else(|| match func.insts[def(val)?.index()].op {
        Op::Const(imm) => Some(imm),
        _ => None,
    })
}

pub(super) fn zero_ext_loc(plan: &Plan<Reg>, val: Value) -> bool {
    plan.zero_ext[val.index()] && matches!(plan.location(val), Location::Reg(_) | Location::Slot)
}

pub(super) fn local_single(
    plan: &Plan<Reg>,
    (uses, pos_of): (&[u32], &[Option<(usize, usize, Inst)>]),
    b_idx: usize,
    limit: usize,
    val: Value,
) -> Option<(usize, Inst)> {
    (uses[val.index()] == 1 && !plan.elided[val.index()])
        .then(|| def_entry(pos_of, val).filter(|&(b, p, _)| b == b_idx && p < limit).map(|(_, p, i)| (p, i)))
        .flatten()
}

fn shift_uext_ok(func: &Function, plan: &Plan<Reg>, ty_bits: u32, (sop, imm): (Binary, i64), inner: Value) -> bool {
    func.ty(inner).is_small_int()
        && (zero_ext_loc(plan, inner) || (sop == Binary::Ishl && ((imm as u32) & (ty_bits - 1)) >= ty_bits - func.ty(inner).bits()))
}

pub(super) fn alu_shift(logical: bool, ty: Type, op: Binary, imm: i64) -> Option<(Shift, u8)> {
    if !ty.is_small_int() {
        return None;
    }
    let bits = ty.bits();
    let mask = bits - 1;
    let amt = (imm as u32) & mask;
    if !matches!(ty, Type::I32 | Type::I64) {
        return (op == Binary::Ishl).then_some((Shift::Lsl, amt as u8));
    }
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

pub(in crate::arm64) const fn width_mask(bits: u32) -> u64 {
    if bits >= 64 { u64::MAX } else { (1_u64 << bits) - 1 }
}

pub(in crate::arm64) const fn narrow_mask_bits(mask: i64) -> Option<u32> {
    match mask {
        0xFF => Some(8),
        0xFFFF => Some(16),
        0xFFFF_FFFF => Some(32),
        _ => None,
    }
}

pub(in crate::arm64) const fn fits_width(val: i64, bits: u32, signed: bool) -> bool {
    if bits == 0 || bits > 64 {
        return false;
    }
    if bits == 64 {
        return true;
    }
    if signed {
        let shift = 64 - bits;
        (val << shift) >> shift == val
    } else {
        (val as u64) <= width_mask(bits)
    }
}

pub(super) fn ext_band_of(
    func: &Function,
    plan: &Plan<Reg>,
    dst_ty: Type,
    (a, b): (Value, Value),
    def: &dyn Fn(Value) -> Option<Inst>,
) -> Option<(ExtOp, Value)> {
    if plan.elided[a.index()] || plan.elided[b.index()] {
        return None;
    }
    let mask = dst_ty.unsigned_max();
    for (inner, m_val) in [(a, b), (b, a)] {
        let Some(imm) = const_of(func, plan, m_val, def).map(|v| (v as u64) & mask) else {
            continue;
        };
        let ext = match (dst_ty, narrow_mask_bits(imm as i64)) {
            (Type::I32 | Type::I64, Some(8)) => Some(ExtOp::Uxtb),
            (Type::I32 | Type::I64, Some(16)) => Some(ExtOp::Uxth),
            (Type::I64, Some(32)) => Some(ExtOp::Uxtw),
            _ => None,
        };
        if let Some(ext) = ext {
            return Some((ext, inner));
        }
    }
    None
}

fn ext_from_op(
    func: &Function,
    plan: &Plan<Reg>,
    dst_ty: Type,
    op: &Op,
    def: &dyn Fn(Value) -> Option<Inst>,
) -> Option<(ExtOp, Value)> {
    match *op {
        Op::Convert(conv @ (Convert::Uextend | Convert::Sextend), inner) => {
            ext_op_of(dst_ty, func.ty(inner), conv == Convert::Sextend).map(|ext| (ext, inner))
        }
        Op::Binary(Binary::Band, a, b) => ext_band_of(func, plan, dst_ty, (a, b), def),
        _ => None,
    }
}

pub(super) fn ext_parts(func: &Function, plan: &Plan<Reg>, dst_ty: Type, val: Value, def: impl Fn(Value) -> Option<Inst>) -> Option<(ExtOp, u8, Value)> {
    if !matches!(dst_ty, Type::I32 | Type::I64) || !plan.elided[val.index()] {
        return None;
    }
    let op = &func.insts[def(val)?.index()].op;
    if let Some((ext, inner)) = ext_from_op(func, plan, dst_ty, op, &def) {
        return Some((ext, 0, inner));
    }
    let (Binary::Ishl, mid, imm) = shift_parts(op, plan)? else { return None };
    let shift = (imm as u32) & (dst_ty.bits() - 1);
    if shift > 4 || !plan.elided[mid.index()] {
        return None;
    }
    ext_from_op(func, plan, dst_ty, &func.insts[def(mid)?.index()].op, &def).map(|(ext, inner)| (ext, shift as u8, inner))
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

pub(super) fn zero_cmp_band(
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
    let (mb, mpos, minst) = def_entry(pos_of, masked)?;
    if mb != b_idx {
        return None;
    }
    let Op::Binary(Binary::Band, x, y) = func.insts[minst.index()].op else {
        return None;
    };
    Some((masked, mpos, x, y))
}

pub(super) fn preserved(func: &Function, plan: &Plan<Reg>, block_idx: usize, start: usize, end: usize, skip_pos: Option<usize>, val: Value) -> bool {
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
            || reg.is_some_and(|reg| super::REGS.scratches(func, data, reg))
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
    let Some((sb, spos, sinst)) = def_entry(pos_of, shr) else { return };
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

pub(super) fn can_peel_uextend(
    func: &Function,
    plan: &Plan<Reg>,
    (uses, pos_of): (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, limit, end, skip, res): (usize, usize, usize, Option<usize>, Value),
    (val, ok): (Value, &dyn Fn(Value) -> bool),
) -> bool {
    if uses[val.index()] != 1 || (plan.elided[val.index()] && skip.is_none()) {
        return false;
    }
    let Some((ub, upos, uinst)) = def_entry(pos_of, val) else { return false };
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
    if !ty.is_small_int() {
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

fn eon_arg_preserved(
    func: &Function,
    plan: &Plan<Reg>,
    pos_of: &[Option<(usize, usize, Inst)>],
    (b_idx, ipos, pos): (usize, usize, usize),
    val: Value,
) -> bool {
    if !plan.elided[val.index()] {
        return preserved(func, plan, b_idx, ipos, pos, None, val);
    }
    def_entry(pos_of, val).is_some_and(|(ub, upos, uinst)| {
        ub == b_idx && upos < ipos
            && matches!(func.insts[uinst.index()].op, Op::Convert(Convert::Uextend, inner) if preserved(func, plan, b_idx, upos, pos, Some(ipos), inner))
    })
}

fn try_fuse_eon(func: &Function, plan: &mut Plan<Reg>, info @ (uses, pos_of): (&[u32], &[Option<(usize, usize, Inst)>]), (b_idx, pos, res): (usize, usize, Value), inner: Value) -> bool {
    let ty = func.ty(res);
    if !ty.is_small_int() { return false; }
    let Some((ipos, iinst)) = local_single(plan, info, b_idx, pos, inner) else { return false };
    let Op::Binary(Binary::Bxor, a, b) = func.insts[iinst.index()].op else { return false };
    let shift_ok = [(a, b), (b, a)].into_iter().any(|(other, cand)| {
        plan.elided[cand.index()] && uses[cand.index()] == 1 && eon_arg_preserved(func, plan, pos_of, (b_idx, ipos, pos), other)
            && def_entry(pos_of, cand).is_some_and(|(sb, spos, sinst)| {
                sb == b_idx && spos < ipos && shift_parts(&func.insts[sinst.index()].op, plan)
                    .is_some_and(|(sop, src, imm)| alu_shift(true, ty, sop, imm).is_some() && eon_arg_preserved(func, plan, pos_of, (b_idx, spos, pos), src))
            })
    });
    if (!eon_arg_preserved(func, plan, pos_of, (b_idx, ipos, pos), a) || !eon_arg_preserved(func, plan, pos_of, (b_idx, ipos, pos), b)) && !shift_ok { return false; }
    plan.elided[inner.index()] = true;
    true
}

fn try_fuse_shift_ty(
    func: &Function,
    plan: &mut Plan<Reg>,
    info: (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos, res, ty, logical): (usize, usize, Value, Type, bool),
    (shift_val, other): (Value, Value),
) -> bool {
    let Some((spos, sinst)) = local_single(plan, info, b_idx, pos, shift_val) else { return false };
    let Some((sop, b, imm)) = shift_parts(&func.insts[sinst.index()].op, plan) else { return false };
    if b == res || alu_shift(logical, ty, sop, imm).is_none() {
        return false;
    }
    let shift_ok = |inner: Value| shift_uext_ok(func, plan, ty.bits(), (sop, imm), inner);
    let peeled = can_peel_uextend(func, plan, info, (b_idx, spos, pos, Some(spos), res), (b, &shift_ok));
    if !peeled && !preserved(func, plan, b_idx, spos, pos, None, b) {
        return false;
    }
    plan.elided[shift_val.index()] = true;
    if peeled { plan.elided[b.index()] = true; }
    peel_other_uextend(func, plan, info, (b_idx, pos, res), other);
    true
}

fn try_fuse_shift(func: &Function, plan: &mut Plan<Reg>, info: (&[u32], &[Option<(usize, usize, Inst)>]), (b_idx, pos, res, alu_op): (usize, usize, Value, Binary), pair: (Value, Value)) -> bool {
    let logical = matches!(alu_op, Binary::Band | Binary::Bor | Binary::Bxor);
    try_fuse_shift_ty(func, plan, info, (b_idx, pos, res, func.ty(res), logical), pair)
}

fn fuse_shifted_alu(func: &Function, plan: &mut Plan<Reg>, info: (&[u32], &[Option<(usize, usize, Inst)>]), (b_idx, pos, res): (usize, usize, Value), (alu_op, x, y): (Binary, Value, Value)) {
    if plan.elided[x.index()] || plan.elided[y.index()] {
        return;
    }
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
    let (mask, def) = (u64::from(ty.bits() - 1), |v| def_inst(pos_of, v));
    for (raw, m) in [(a, b), (b, a)] {
        if const_of(func, plan, m, def).is_some_and(|imm| (imm as u64 & mask) == mask) && preserved(func, plan, b_idx, ypos, pos, None, raw) {
            plan.elided[y.index()] = true;
            return;
        }
    }
}

fn try_fuse_ext_operand(func: &Function, plan: &mut Plan<Reg>, info @ (uses, pos_of): (&[u32], &[Option<(usize, usize, Inst)>]), (b_idx, pos, dst_ty): (usize, usize, Type), ext_val: Value) -> bool {
    let Some((epos, einst)) = local_single(plan, info, b_idx, pos, ext_val) else { return false };
    let def = |v| def_inst(pos_of, v);
    let (mid, spos, ext_pos, inner) = if let Some((_, inner)) = ext_from_op(func, plan, dst_ty, &func.insts[einst.index()].op, &def) {
        (None, None, epos, inner)
    } else {
        let Some((Binary::Ishl, mid, imm)) = shift_parts(&func.insts[einst.index()].op, plan) else { return false };
        if ((imm as u32) & (dst_ty.bits() - 1)) > 4 || uses[mid.index()] != 1 {
            return false;
        }
        let Some((mb, upos, uinst)) = def_entry(pos_of, mid) else { return false };
        if mb != b_idx || upos >= epos {
            return false;
        }
        let Some((_, inner)) = ext_from_op(func, plan, dst_ty, &func.insts[uinst.index()].op, &def) else { return false };
        (Some(mid), Some(epos), upos, inner)
    };
    if !preserved(func, plan, b_idx, ext_pos, pos, spos, inner) {
        return false;
    }
    plan.elided[ext_val.index()] = true;
    if let Some(m) = mid { plan.elided[m.index()] = true; }
    true
}

pub(super) fn mneg_parts(func: &Function, plan: &Plan<Reg>, ty: Type, val: Value, def: impl Fn(Value) -> Option<Inst>) -> Option<(Value, Value)> {
    if !matches!(ty, Type::I32 | Type::I64) || !plan.elided[val.index()] { return None; }
    let Op::Binary(Binary::Imul, a, b) = func.insts[def(val)?.index()].op else { return None };
    (!plan.elided[a.index()] && !plan.elided[b.index()]).then_some((a, b))
}

fn neg_inner(func: &Function, plan: &Plan<Reg>, op: &Op, def: &dyn Fn(Value) -> Option<Inst>) -> Option<Value> {
    match *op {
        Op::Unary(Unary::Ineg, inner) => Some(inner),
        Op::Binary(Binary::Isub, zero, inner) if const_of(func, plan, zero, def) == Some(0) => Some(inner),
        _ => None,
    }
}

pub(super) fn cmn_parts(func: &Function, plan: &Plan<Reg>, ty: Type, val: Value, def: &dyn Fn(Value) -> Option<Inst>) -> Option<(Value, Option<(Shift, u8)>)> {
    if !matches!(ty, Type::I32 | Type::I64) || !plan.elided[val.index()] { return None; }
    let inner = neg_inner(func, plan, &func.insts[def(val)?.index()].op, def)?;
    if plan.elided[inner.index()]
        && let Some(sinst) = def(inner)
        && let Some((sop, root, imm)) = shift_parts(&func.insts[sinst.index()].op, plan)
        && let Some((shift, count)) = alu_shift(false, ty, sop, imm)
    {
        return (!plan.elided[root.index()]).then_some((root, Some((shift, count))));
    }
    (!plan.elided[inner.index()]).then_some((inner, None))
}

pub(super) fn cmn_sum_parts(func: &Function, plan: &Plan<Reg>, (x, y): (Value, Value), def: &dyn Fn(Value) -> Option<Inst>) -> Option<(Value, Value)> {
    let sum = (const_of(func, plan, y, def) == Some(0)).then_some(x).or_else(|| (const_of(func, plan, x, def) == Some(0)).then_some(y))?;
    if !plan.elided[sum.index()] { return None; }
    let Op::Binary(Binary::Iadd, a, b) = func.insts[def(sum)?.index()].op else { return None };
    (!plan.elided[a.index()] && !plan.elided[b.index()]).then_some((a, b))
}

fn try_fuse_cmn_neg(func: &Function, plan: &mut Plan<Reg>, info @ (_, pos_of): (&[u32], &[Option<(usize, usize, Inst)>]), (b_idx, pos, cmp_ty): (usize, usize, Type), cand: Value) -> bool {
    let Some((npos, ninst)) = local_single(plan, info, b_idx, pos, cand) else { return false };
    let def = |v| def_inst(pos_of, v);
    let Some(inner) = neg_inner(func, plan, &func.insts[ninst.index()].op, &def) else { return false };
    if !plan.elided[inner.index()] {
        if preserved(func, plan, b_idx, npos, pos, None, inner) { plan.elided[cand.index()] = true; return true; }
        return false;
    }
    if let Some((sb, spos, sinst)) = def_entry(pos_of, inner)
        && sb == b_idx && spos < npos
        && let Some((sop, root, imm)) = shift_parts(&func.insts[sinst.index()].op, plan)
        && alu_shift(false, cmp_ty, sop, imm).is_some()
        && !plan.elided[root.index()] && preserved(func, plan, b_idx, spos, pos, Some(npos), root)
    {
        plan.elided[cand.index()] = true;
        return true;
    }
    false
}

fn try_fuse_cmn_sum(func: &Function, plan: &mut Plan<Reg>, info @ (_, pos_of): (&[u32], &[Option<(usize, usize, Inst)>]), (b_idx, pos): (usize, usize), (lhs, rhs): (Value, Value)) -> bool {
    let def = |v| def_inst(pos_of, v);
    let Some(sum) = (const_of(func, plan, rhs, &def) == Some(0)).then_some(lhs).or_else(|| (const_of(func, plan, lhs, &def) == Some(0)).then_some(rhs)) else { return false };
    let Some((spos, sinst)) = local_single(plan, info, b_idx, pos, sum) else { return false };
    let Op::Binary(Binary::Iadd, a, b) = func.insts[sinst.index()].op else { return false };
    if plan.elided[a.index()] || plan.elided[b.index()] || const_of(func, plan, a, &def).is_some() || const_of(func, plan, b, &def).is_some()
        || !preserved(func, plan, b_idx, spos, pos, None, a) || !preserved(func, plan, b_idx, spos, pos, None, b)
    {
        return false;
    }
    plan.elided[sum.index()] = true;
    true
}

fn fuse_icmp_operands(func: &Function, plan: &mut Plan<Reg>, info: (&[u32], &[Option<(usize, usize, Inst)>]), (b_idx, pos, res): (usize, usize, Value), (cc, a, b): (IntCC, Value, Value)) {
    if plan.elided[a.index()] || plan.elided[b.index()] {
        return;
    }
    let ty = func.ty(a);
    let cmn_eq = matches!(cc, IntCC::Equal | IntCC::NotEqual);
    if cmn_eq && try_fuse_cmn_sum(func, plan, info, (b_idx, pos), (a, b)) { return; }
    let imm_check = |p: &Plan<Reg>, v: Value| p.const_of(v).and_then(|imm| super::int::add_imm_parts(true, crate::fold::extend(imm, func.ty(v), cc.is_signed()))).is_some();
    if imm_check(plan, b) || try_fuse_ext_operand(func, plan, info, (b_idx, pos, ty), b) || try_fuse_ext_operand(func, plan, info, (b_idx, pos, ty), a) || imm_check(plan, a) {
        return;
    }
    if cmn_eq && (try_fuse_cmn_neg(func, plan, info, (b_idx, pos, ty), b) || try_fuse_cmn_neg(func, plan, info, (b_idx, pos, ty), a)) {
        return;
    }
    if !try_fuse_shift_ty(func, plan, info, (b_idx, pos, res, ty, false), (b, a)) {
        try_fuse_shift_ty(func, plan, info, (b_idx, pos, res, ty, false), (a, b));
    }
}

fn fuse_int_flag_pair(
    func: &Function,
    plan: &mut Plan<Reg>,
    info: (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos): (usize, usize),
    (alu_op, x, y): (Binary, Value, Value),
) -> bool {
    if alu_op == Binary::Iadd {
        for (flag_val, other) in [(y, x), (x, y)] {
            if !plan.elided[other.index()] && try_fuse_flag_operand(func, plan, info, (b_idx, pos), flag_val) {
                return true;
            }
        }
        return false;
    }
    alu_op == Binary::Isub && plan.const_of(x) == Some(0) && try_fuse_flag_operand(func, plan, info, (b_idx, pos), y)
}

fn fuse_int_binary_wide(
    func: &Function,
    plan: &mut Plan<Reg>,
    info: (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos, res): (usize, usize, Value),
    (alu_op, x, y): (Binary, Value, Value),
) -> bool {
    fuse_int_flag_pair(func, plan, info, (b_idx, pos), (alu_op, x, y))
        || (alu_op == Binary::Iadd && mem::try_fuse_sib(func, plan, info, (b_idx, pos, res), (x, y)))
        || bits::try_fuse_extr(func, plan, info, (b_idx, pos, res), (alu_op, x, y))
        || bfi::try_fuse_bfi(func, plan, info, (b_idx, pos, res), (alu_op, x, y))
        || (matches!(alu_op, Binary::Iadd | Binary::Isub) && mem::try_fuse_madd(func, plan, info, (b_idx, pos, res), (alu_op, x, y)))
}

fn fuse_int_binary_ext(
    func: &Function,
    plan: &mut Plan<Reg>,
    info: (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos, res): (usize, usize, Value),
    (alu_op, x, y): (Binary, Value, Value),
) -> bool {
    let ty = func.ty(res);
    if !matches!(ty, Type::I32 | Type::I64) || !matches!(alu_op, Binary::Iadd | Binary::Isub) {
        return false;
    }
    if !plan.elided[x.index()] && try_fuse_ext_operand(func, plan, info, (b_idx, pos, ty), y) {
        peel_other_uextend(func, plan, info, (b_idx, pos, res), x);
        if alu_op == Binary::Iadd && !plan.elided[x.index()] {
            swap_ext_operand(func, plan, info, (b_idx, pos, ty), (x, y));
        }
        return true;
    }
    alu_op == Binary::Iadd && !plan.elided[y.index()] && try_fuse_ext_operand(func, plan, info, (b_idx, pos, ty), x)
}

fn uext_inner(func: &Function, val: Value, def: &dyn Fn(Value) -> Option<Inst>) -> Option<Value> {
    match func.insts[def(val)?.index()].op {
        Op::Convert(Convert::Uextend, inner) => Some(inner),
        _ => None,
    }
}

fn swap_parts(func: &Function, plan: &Plan<Reg>, (x, y): (Value, Value), def: &dyn Fn(Value) -> Option<Inst>) -> bool {
    uext_inner(func, x, def).is_some_and(|a| !zero_ext_loc(plan, a)) && uext_inner(func, y, def).is_some_and(|b| zero_ext_loc(plan, b))
}

/// Reports whether an add extends `x` in its operand and reads `y` as an already zero-extended register.
pub(super) fn ext_swapped(func: &Function, plan: &Plan<Reg>, (x, y): (Value, Value), def: &dyn Fn(Value) -> Option<Inst>) -> bool {
    plan.elided[x.index()] && plan.elided[y.index()] && swap_parts(func, plan, (x, y), def)
}

fn swap_ext_operand(func: &Function, plan: &mut Plan<Reg>, info @ (_, pos_of): (&[u32], &[Option<(usize, usize, Inst)>]), (b_idx, pos, ty): (usize, usize, Type), (x, y): (Value, Value)) {
    if swap_parts(func, plan, (x, y), &|v| def_inst(pos_of, v)) {
        try_fuse_ext_operand(func, plan, info, (b_idx, pos, ty), x);
    }
}

fn fuse_int_binary(
    func: &Function,
    plan: &mut Plan<Reg>,
    info: (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos, res): (usize, usize, Value),
    (alu_op, x, y): (Binary, Value, Value),
) {
    let wide = matches!(func.ty(res), Type::I32 | Type::I64);
    if (wide && fuse_int_binary_wide(func, plan, info, (b_idx, pos, res), (alu_op, x, y)))
        || (alu_op == Binary::Band && bits::try_fuse_bfx(func, plan, info, (b_idx, pos, res), (alu_op, x, y)))
        || shift::try_fuse_bnot(func, plan, info, (b_idx, pos, res), (alu_op, x, y))
        || fuse_int_binary_ext(func, plan, info, (b_idx, pos, res), (alu_op, x, y))
    {
        return;
    }
    fuse_shifted_alu(func, plan, info, (b_idx, pos, res), (alu_op, x, y));
}

fn fuse_block_unary(
    func: &Function,
    plan: &mut Plan<Reg>,
    info: (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos, res): (usize, usize, Value),
    (unary_op, inner): (Unary, Value),
) {
    match unary_op {
        Unary::Bnot => {
            if !try_fuse_eon(func, plan, info, (b_idx, pos, res), inner) && func.ty(res).is_small_int() {
                try_fuse_shift(func, plan, info, (b_idx, pos, res, Binary::Bxor), (inner, res));
            }
        }
        Unary::Ineg if matches!(func.ty(res), Type::I32 | Type::I64) => {
            if !try_fuse_flag_operand(func, plan, info, (b_idx, pos), inner)
                && !mem::try_fuse_madd(func, plan, info, (b_idx, pos, res), (Binary::Isub, res, inner))
            {
                try_fuse_shift(func, plan, info, (b_idx, pos, res, Binary::Isub), (inner, res));
            }
        }
        _ => {}
    }
}

fn fuse_block_shift(
    func: &Function,
    plan: &mut Plan<Reg>,
    info: (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos, res): (usize, usize, Value),
    (sop, x, y): (Binary, Value, Value),
) {
    fuse_shift_mask(func, plan, info, (b_idx, pos), x, y);
    if matches!(sop, Binary::Ishl | Binary::Ushr | Binary::Sshr) && func.ty(res).is_small_int() {
        bits::try_fuse_bfx(func, plan, info, (b_idx, pos, res), (sop, x, y));
    }
}

fn fuse_block_icmp(
    func: &Function,
    plan: &mut Plan<Reg>,
    info: (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos, res): (usize, usize, Value),
    (cc, a, b): (IntCC, Value, Value),
) {
    try_fuse_icmp_select(func, plan, info, (b_idx, pos, res), (a, b));
    if plan.elided[res.index()] {
        return;
    }
    if matches!(func.ty(a), Type::I32 | Type::I64) {
        fuse_icmp_operands(func, plan, info, (b_idx, pos, res), (cc, a, b));
    } else if cc.is_signed() {
        mem::try_fuse_icmp_sload(func, plan, info, (b_idx, pos), (cc, a, b));
    }
}

fn fuse_block_inst(
    func: &Function,
    plan: &mut Plan<Reg>,
    info: (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos, res): (usize, usize, Value),
    op: &Op,
) {
    if try_fuse_vec_op(func, plan, info, (b_idx, pos, res), op)
        || shift::try_fuse_multi_shift(func, plan, info, (b_idx, pos, res), op)
    {
        return;
    }
    match *op {
        Op::Binary(sop @ (Binary::Ishl | Binary::Ushr | Binary::Sshr | Binary::Rotl | Binary::Rotr), x, y) => {
            fuse_block_shift(func, plan, info, (b_idx, pos, res), (sop, x, y));
        }
        Op::Unary(uop, inner) => fuse_block_unary(func, plan, info, (b_idx, pos, res), (uop, inner)),
        Op::Convert(conv @ (Convert::Uextend | Convert::Sextend), inner) => {
            if !bits::try_fuse_conv_bfx(func, plan, info, (b_idx, pos, res), (conv, inner)) && conv == Convert::Sextend {
                mem::try_fuse_sload_extend(func, plan, info, (b_idx, pos, res), inner);
            }
        }
        Op::Convert(Convert::Ireduce, shr) => fuse_ireduce_ushr(func, plan, info, (b_idx, pos, res), shr),
        Op::Icmp(cc, a, b) if func.ty(a).is_small_int() => fuse_block_icmp(func, plan, info, (b_idx, pos, res), (cc, a, b)),
        Op::Binary(alu_op @ (Binary::Band | Binary::Bor | Binary::Bxor | Binary::Iadd | Binary::Isub), x, y)
            if func.ty(res).is_small_int() =>
        {
            fuse_int_binary(func, plan, info, (b_idx, pos, res), (alu_op, x, y));
        }
        _ => {}
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
            if let Some(res) = data.result && !plan.elided[res.index()] {
                fuse_block_inst(func, plan, (&uses, &pos_of), (b_idx, pos, res), &data.op);
            }
        }
    }
    fuse_icmp_selects(func, plan, (&uses, &pos_of));
    fuse_icmp_flags(func, plan, &uses);
    bits::fuse_tst(func, plan, (&uses, &pos_of));
    fuse_fits(func, plan, &uses);
    ccmp::fuse_ccmp(func, plan, (&uses, &pos_of));
    mem::fuse_store_ireduce(func, plan, &uses);
}

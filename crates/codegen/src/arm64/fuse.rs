mod alu;
mod bfi;
mod bits;
mod ccmp;
mod cmp;
mod flag;
mod mem;
mod shift;
mod sign;
mod vconst;

use super::asm::{Alu, ExtOp, Reg, Shift};
use crate::alloc::{Location, Plan};
use crate::ir::{Binary, Convert, Function, Inst, Op, Type, Unary, Value};
pub(super) use alu::ext_swapped;
pub(super) use bfi::bfi_parts;
pub(super) use bits::{
    Bfx, bfx_parts, conv_bfx_parts, extr_parts, single_bit_mask, unwrap_elided_tst,
};
pub(super) use cmp::{cmn_parts, cmn_sum_parts};
use flag::{fuse_fits, fuse_icmp_flags, fuse_icmp_selects};
pub(super) use flag::{preserves_flags, zero_cmp_operand};
use mem::try_fuse_vec_op;
pub(super) use mem::{encode_neon_movi, madd_parts, store_source};
pub(super) use sign::sign_bit_test;

pub(super) fn def_entry(
    pos_of: &[Option<(usize, usize, Inst)>],
    val: Value,
) -> Option<(usize, usize, Inst)> {
    pos_of.get(val.index()).and_then(|&entry| entry)
}

pub(super) fn def_inst(pos_of: &[Option<(usize, usize, Inst)>], val: Value) -> Option<Inst> {
    def_entry(pos_of, val).map(|(_, _, inst)| inst)
}

pub(super) fn const_of(
    func: &Function,
    plan: &Plan<Reg>,
    val: Value,
    def: impl Fn(Value) -> Option<Inst>,
) -> Option<i64> {
    plan.const_of(val)
        .or_else(|| match func.insts[def(val)?.index()].op {
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
        .then(|| {
            def_entry(pos_of, val)
                .filter(|&(b, p, _)| b == b_idx && p < limit)
                .map(|(_, p, i)| (p, i))
        })
        .flatten()
}

fn shift_uext_ok(
    func: &Function,
    plan: &Plan<Reg>,
    ty_bits: u32,
    (sop, imm): (Binary, i64),
    inner: Value,
) -> bool {
    func.ty(inner).is_small_int()
        && (zero_ext_loc(plan, inner)
            || (sop == Binary::Ishl
                && ((imm as u32) & (ty_bits - 1)) >= ty_bits - func.ty(inner).bits()))
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
    if bits >= 64 {
        u64::MAX
    } else {
        (1_u64 << bits) - 1
    }
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

pub(super) fn ext_parts(
    func: &Function,
    plan: &Plan<Reg>,
    dst_ty: Type,
    val: Value,
    def: impl Fn(Value) -> Option<Inst>,
) -> Option<(ExtOp, u8, Value)> {
    if !matches!(dst_ty, Type::I32 | Type::I64) || !plan.elided[val.index()] {
        return None;
    }
    let op = &func.insts[def(val)?.index()].op;
    if let Some((ext, inner)) = ext_from_op(func, plan, dst_ty, op, &def) {
        return Some((ext, 0, inner));
    }
    let (Binary::Ishl, mid, imm) = shift_parts(op, plan)? else {
        return None;
    };
    let shift = (imm as u32) & (dst_ty.bits() - 1);
    if shift > 4 || !plan.elided[mid.index()] {
        return None;
    }
    ext_from_op(func, plan, dst_ty, &func.insts[def(mid)?.index()].op, &def)
        .map(|(ext, inner)| (ext, shift as u8, inner))
}

pub(super) fn shift_parts(op: &Op, plan: &Plan<Reg>) -> Option<(Binary, Value, i64)> {
    match *op {
        Op::Binary(
            sop @ (Binary::Ishl | Binary::Ushr | Binary::Sshr | Binary::Rotl | Binary::Rotr),
            b,
            c,
        ) => Some((sop, b, plan.const_of(c)?)),
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
    crate::emit::zero_cmp_band(func, plan, (uses, pos_of), b_idx, (lhs, rhs), |v| {
        plan.const_of(v) == Some(0)
    })
}

pub(super) fn preserved(
    func: &Function,
    plan: &Plan<Reg>,
    block_idx: usize,
    start: usize,
    end: usize,
    skip_pos: Option<usize>,
    val: Value,
) -> bool {
    crate::emit::preserved(
        func,
        (plan, &super::REGS),
        (block_idx, start, end, usize::MAX),
        |pos| skip_pos == Some(pos),
        val,
    )
}

pub(super) fn shrn_shift(src_ty: Type, dst_ty: Type, imm: i64) -> Option<u32> {
    let (
        Type::Vec {
            lane: sl,
            count: sc,
        },
        Type::Vec {
            lane: dl,
            count: dc,
        },
    ) = (src_ty, dst_ty)
    else {
        return None;
    };
    if sc != dc || !sl.is_int() || !dl.is_int() || sl.bits() != dl.bits() * 2 {
        return None;
    }
    let shift = (imm as u32) & (sl.bits() - 1);
    (1..=dl.bits()).contains(&shift).then_some(shift)
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
    let Some((ub, upos, uinst)) = def_entry(pos_of, val) else {
        return false;
    };
    if ub != b_idx || upos >= limit {
        return false;
    }
    let Op::Convert(Convert::Uextend, inner) = func.insts[uinst.index()].op else {
        return false;
    };
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
    if can_peel_uextend(
        func,
        plan,
        info,
        (b_idx, pos, pos, None, res),
        (other, &other_ok),
    ) {
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

pub(super) fn elided_use_end(
    func: &Function,
    plan: &Plan<Reg>,
    insts: &[Inst],
    pos: usize,
    mut target: Value,
) -> Option<usize> {
    let mut last = None;
    for (u_pos, &ui) in insts.iter().enumerate().skip(pos + 1) {
        let ud = &func.insts[ui.index()];
        let mut used = false;
        ud.op.for_operands(|v| used |= v == target);
        if !used {
            continue;
        }
        if matches!(ud.op, Op::Convert(Convert::Uextend, _))
            && let Some(ur) = ud.result
            && plan.elided[ur.index()]
        {
            target = ur;
        } else {
            last = Some(u_pos);
        }
    }
    last
}

pub(super) fn eon_inner(
    func: &Function,
    plan: &Plan<Reg>,
    x: Value,
    def: impl Fn(Value) -> Option<Inst>,
) -> Option<(Value, Value)> {
    if !plan.elided[x.index()] {
        return None;
    }
    let Op::Binary(Binary::Bxor, a, b) = func.insts[def(x)?.index()].op else {
        return None;
    };
    Some((a, b))
}

pub(super) fn mneg_parts(
    func: &Function,
    plan: &Plan<Reg>,
    ty: Type,
    val: Value,
    def: impl Fn(Value) -> Option<Inst>,
) -> Option<(Value, Value)> {
    if !matches!(ty, Type::I32 | Type::I64) || !plan.elided[val.index()] {
        return None;
    }
    let Op::Binary(Binary::Imul, a, b) = func.insts[def(val)?.index()].op else {
        return None;
    };
    (!plan.elided[a.index()] && !plan.elided[b.index()]).then_some((a, b))
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
        Op::Binary(
            sop @ (Binary::Ishl | Binary::Ushr | Binary::Sshr | Binary::Rotl | Binary::Rotr),
            x,
            y,
        ) => {
            alu::fuse_block_shift(func, plan, info, (b_idx, pos, res), (sop, x, y));
        }
        Op::Unary(uop, inner) => {
            alu::fuse_block_unary(func, plan, info, (b_idx, pos, res), (uop, inner))
        }
        Op::Convert(conv @ (Convert::Uextend | Convert::Sextend), inner) => {
            if !bits::try_fuse_conv_bfx(func, plan, info, (b_idx, pos, res), (conv, inner))
                && conv == Convert::Sextend
            {
                mem::try_fuse_sload_extend(func, plan, info, (b_idx, pos, res), inner);
            }
        }
        Op::Convert(Convert::Ireduce, shr) => {
            alu::fuse_ireduce_ushr(func, plan, info, (b_idx, pos, res), shr)
        }
        Op::Icmp(cc, a, b) if func.ty(a).is_small_int() => {
            cmp::fuse_block_icmp(func, plan, info, (b_idx, pos, res), (cc, a, b));
        }
        Op::Binary(
            alu_op @ (Binary::Band | Binary::Bor | Binary::Bxor | Binary::Iadd | Binary::Isub),
            x,
            y,
        ) if func.ty(res).is_small_int() => {
            alu::fuse_int_binary(func, plan, info, (b_idx, pos, res), (alu_op, x, y));
        }
        _ => {}
    }
}

pub(super) fn fuse(func: &Function, plan: &mut Plan<Reg>, lse: bool) {
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
            if let Some(res) = data.result
                && !plan.elided[res.index()]
            {
                if lse
                    && uses[res.index()] == 0
                    && func.ty(res).is_small_int()
                    && matches!(data.op, Op::AtomicRmw(op, ..) if op != crate::ir::AtomicRmwOp::Nand)
                {
                    plan.fused[res.index()] = true;
                    continue;
                }
                fuse_block_inst(func, plan, (&uses, &pos_of), (b_idx, pos, res), &data.op);
            }
        }
    }
    vconst::elide_vec_consts(func, plan, &pos_of);
    fuse_icmp_selects(func, plan, (&uses, &pos_of));
    fuse_icmp_flags(func, plan, &uses);
    bits::fuse_tst(func, plan, (&uses, &pos_of));
    fuse_fits(func, plan, &uses);
    ccmp::fuse_ccmp(func, plan, (&uses, &pos_of));
    mem::fuse_atomic_addrs(func, plan, lse, (&uses, &pos_of));
    mem::fuse_store_ireduce(func, plan, &uses);
}

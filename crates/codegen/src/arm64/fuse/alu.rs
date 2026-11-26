use super::super::asm::Reg;
use super::super::int::{add_imm_parts, add_imm_split_parts, encode_bitmask_imm};
use super::flag::try_fuse_flag_operand;
use super::{
    alu_shift, bfi, bits, can_peel_uextend, const_of, def_entry, def_inst, ext_from_op,
    local_single, mem, peel_other_uextend, preserved, shift, shift_parts, shift_uext_ok,
    shrn_shift, zero_ext_loc,
};
use crate::alloc::Plan;
use crate::ir::{Binary, Convert, Function, Inst, Op, Type, Unary, Value};

pub(super) fn fuse_ireduce_ushr(
    func: &Function,
    plan: &mut Plan<Reg>,
    (uses, pos_of): (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos, res): (usize, usize, Value),
    shr: Value,
) {
    let src_ty = func.ty(shr);
    if uses[shr.index()] != 1 || plan.elided[shr.index()] {
        return;
    }
    let Some((sb, spos, sinst)) = def_entry(pos_of, shr) else {
        return;
    };
    if sb != b_idx || spos >= pos {
        return;
    }
    match func.insts[sinst.index()].op {
        Op::Binary(Binary::Ushr, val, amt) if matches!(src_ty, Type::I32 | Type::I64) => {
            let Some(shift) = plan.const_of(amt) else {
                return;
            };
            let src_bits = src_ty.bits();
            let shift = (shift as u32) & (src_bits - 1);
            if shift > 0
                && src_bits - shift <= func.ty(res).bits()
                && preserved(func, plan, b_idx, spos, pos, None, val)
            {
                plan.elided[shr.index()] = true;
            }
        }
        Op::Binary(Binary::Ushr | Binary::Sshr, val, amt) if src_ty.is_vector() => {
            if mem::const_splat(func, plan, pos_of, amt)
                .is_some_and(|(_, imm)| shrn_shift(src_ty, func.ty(res), imm).is_some())
                && preserved(func, plan, b_idx, spos, pos, None, val)
            {
                plan.elided[shr.index()] = true;
            }
        }
        _ => {}
    }
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
        ub == b_idx
            && upos < ipos
            && matches!(
                func.insts[uinst.index()].op,
                Op::Convert(Convert::Uextend, inner) if preserved(func, plan, b_idx, upos, pos, Some(ipos), inner)
            )
    })
}

fn try_fuse_eon(
    func: &Function,
    plan: &mut Plan<Reg>,
    info @ (uses, pos_of): (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos, res): (usize, usize, Value),
    inner: Value,
) -> bool {
    let ty = func.ty(res);
    if !ty.is_small_int() {
        return false;
    }
    let Some((ipos, iinst)) = local_single(plan, info, b_idx, pos, inner) else {
        return false;
    };
    let Op::Binary(Binary::Bxor, a, b) = func.insts[iinst.index()].op else {
        return false;
    };
    let shift_ok = [(a, b), (b, a)].into_iter().any(|(other, cand)| {
        plan.elided[cand.index()]
            && uses[cand.index()] == 1
            && eon_arg_preserved(func, plan, pos_of, (b_idx, ipos, pos), other)
            && def_entry(pos_of, cand).is_some_and(|(sb, spos, sinst)| {
                sb == b_idx
                    && spos < ipos
                    && shift_parts(&func.insts[sinst.index()].op, plan).is_some_and(
                        |(sop, src, imm)| {
                            alu_shift(true, ty, sop, imm).is_some()
                                && eon_arg_preserved(func, plan, pos_of, (b_idx, spos, pos), src)
                        },
                    )
            })
    });
    if (!eon_arg_preserved(func, plan, pos_of, (b_idx, ipos, pos), a)
        || !eon_arg_preserved(func, plan, pos_of, (b_idx, ipos, pos), b))
        && !shift_ok
    {
        return false;
    }
    plan.elided[inner.index()] = true;
    true
}

pub(super) fn try_fuse_shift_ty(
    func: &Function,
    plan: &mut Plan<Reg>,
    info: (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos, res, ty, logical): (usize, usize, Value, Type, bool),
    (shift_val, other): (Value, Value),
) -> bool {
    let Some((spos, sinst)) = local_single(plan, info, b_idx, pos, shift_val) else {
        return false;
    };
    let Some((sop, b, imm)) = shift_parts(&func.insts[sinst.index()].op, plan) else {
        return false;
    };
    if b == res || alu_shift(logical, ty, sop, imm).is_none() {
        return false;
    }
    let shift_ok = |inner: Value| shift_uext_ok(func, plan, ty.bits(), (sop, imm), inner);
    let peeled = can_peel_uextend(
        func,
        plan,
        info,
        (b_idx, spos, pos, Some(spos), res),
        (b, &shift_ok),
    );
    if !peeled && !preserved(func, plan, b_idx, spos, pos, None, b) {
        return false;
    }
    plan.elided[shift_val.index()] = true;
    if peeled {
        plan.elided[b.index()] = true;
    }
    peel_other_uextend(func, plan, info, (b_idx, pos, res), other);
    true
}

fn try_fuse_shift(
    func: &Function,
    plan: &mut Plan<Reg>,
    info: (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos, res, alu_op): (usize, usize, Value, Binary),
    pair: (Value, Value),
) -> bool {
    let logical = matches!(alu_op, Binary::Band | Binary::Bor | Binary::Bxor);
    try_fuse_shift_ty(
        func,
        plan,
        info,
        (b_idx, pos, res, func.ty(res), logical),
        pair,
    )
}

fn fuse_shifted_alu(
    func: &Function,
    plan: &mut Plan<Reg>,
    info: (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos, res): (usize, usize, Value),
    (alu_op, x, y): (Binary, Value, Value),
) {
    if plan.elided[x.index()] || plan.elided[y.index()] {
        return;
    }
    let candidates = if alu_op == Binary::Isub {
        [Some((y, x)), None]
    } else {
        [Some((y, x)), Some((x, y))]
    };
    for pair in candidates.into_iter().flatten() {
        if try_fuse_shift(func, plan, info, (b_idx, pos, res, alu_op), pair) {
            break;
        }
    }
}

fn fuse_shift_mask(
    func: &Function,
    plan: &mut Plan<Reg>,
    info @ (_, pos_of): (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos): (usize, usize),
    x: Value,
    y: Value,
) {
    let ty = func.ty(x);
    if !matches!(ty, Type::I32 | Type::I64) {
        return;
    }
    let Some((ypos, yinst)) = local_single(plan, info, b_idx, pos, y) else {
        return;
    };
    let Op::Binary(Binary::Band, a, b) = func.insts[yinst.index()].op else {
        return;
    };
    let (mask, def) = (u64::from(ty.bits() - 1), |v| def_inst(pos_of, v));
    for (raw, m) in [(a, b), (b, a)] {
        if const_of(func, plan, m, def).is_some_and(|imm| (imm as u64 & mask) == mask)
            && preserved(func, plan, b_idx, ypos, pos, None, raw)
        {
            plan.elided[y.index()] = true;
            return;
        }
    }
}

pub(super) fn try_fuse_ext_operand(
    func: &Function,
    plan: &mut Plan<Reg>,
    info @ (uses, pos_of): (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos, dst_ty): (usize, usize, Type),
    ext_val: Value,
) -> bool {
    let Some((epos, einst)) = local_single(plan, info, b_idx, pos, ext_val) else {
        return false;
    };
    let def = |v| def_inst(pos_of, v);
    let (mid, spos, ext_pos, inner) = if let Some((_, inner)) =
        ext_from_op(func, plan, dst_ty, &func.insts[einst.index()].op, &def)
    {
        (None, None, epos, inner)
    } else {
        let Some((Binary::Ishl, mid, imm)) = shift_parts(&func.insts[einst.index()].op, plan)
        else {
            return false;
        };
        if ((imm as u32) & (dst_ty.bits() - 1)) > 4 || uses[mid.index()] != 1 {
            return false;
        }
        let Some((mb, upos, uinst)) = def_entry(pos_of, mid) else {
            return false;
        };
        if mb != b_idx || upos >= epos {
            return false;
        }
        let Some((_, inner)) = ext_from_op(func, plan, dst_ty, &func.insts[uinst.index()].op, &def)
        else {
            return false;
        };
        (Some(mid), Some(epos), upos, inner)
    };
    if !preserved(func, plan, b_idx, ext_pos, pos, spos, inner) {
        return false;
    }
    plan.elided[ext_val.index()] = true;
    if let Some(m) = mid {
        plan.elided[m.index()] = true;
    }
    true
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
            if !plan.elided[other.index()]
                && try_fuse_flag_operand(func, plan, info, (b_idx, pos), flag_val)
            {
                return true;
            }
        }
        return false;
    }
    alu_op == Binary::Isub
        && plan.const_of(x) == Some(0)
        && try_fuse_flag_operand(func, plan, info, (b_idx, pos), y)
}

fn fuse_int_binary_wide(
    func: &Function,
    plan: &mut Plan<Reg>,
    info: (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos, res): (usize, usize, Value),
    (alu_op, x, y): (Binary, Value, Value),
) -> bool {
    fuse_int_flag_pair(func, plan, info, (b_idx, pos), (alu_op, x, y))
        || (alu_op == Binary::Iadd
            && mem::try_fuse_sib(func, plan, info, (b_idx, pos, res), (x, y)))
        || bits::try_fuse_extr(func, plan, info, (b_idx, pos, res), (alu_op, x, y))
        || bfi::try_fuse_bfi(func, plan, info, (b_idx, pos, res), (alu_op, x, y))
        || (matches!(alu_op, Binary::Iadd | Binary::Isub)
            && mem::try_fuse_madd(func, plan, info, (b_idx, pos, res), (alu_op, x, y)))
}

fn fits_add_sub_imm(plan: &Plan<Reg>, sub: bool, ty: Type, val: Value) -> bool {
    let Some(raw) = plan.const_of(val) else {
        return false;
    };
    let simm = crate::fold::extend(raw, ty, true);
    add_imm_parts(sub, simm).is_some() || add_imm_split_parts(sub, simm).is_some()
}

fn has_zext_uextend_imm_skip(
    func: &Function,
    plan: &Plan<Reg>,
    def: &dyn Fn(Value) -> Option<Inst>,
    (alu_op, ty, imm_val, ext_val): (Binary, Type, Value, Value),
) -> bool {
    fits_add_sub_imm(plan, alu_op == Binary::Isub, ty, imm_val)
        && uext_inner(func, ext_val, def).is_some_and(|inner| zero_ext_loc(plan, inner))
}

fn fuse_int_binary_ext(
    func: &Function,
    plan: &mut Plan<Reg>,
    info @ (_, pos_of): (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos, res): (usize, usize, Value),
    (alu_op, x, y): (Binary, Value, Value),
) -> bool {
    let ty = func.ty(res);
    if !matches!(ty, Type::I32 | Type::I64) || !matches!(alu_op, Binary::Iadd | Binary::Isub) {
        return false;
    }
    let def = |v| def_inst(pos_of, v);
    if !plan.elided[x.index()]
        && !has_zext_uextend_imm_skip(func, plan, &def, (alu_op, ty, x, y))
        && try_fuse_ext_operand(func, plan, info, (b_idx, pos, ty), y)
    {
        peel_other_uextend(func, plan, info, (b_idx, pos, res), x);
        if alu_op == Binary::Iadd && !plan.elided[x.index()] {
            swap_ext_operand(func, plan, info, (b_idx, pos, ty), (x, y));
        }
        return true;
    }
    alu_op == Binary::Iadd
        && !plan.elided[y.index()]
        && !has_zext_uextend_imm_skip(func, plan, &def, (alu_op, ty, y, x))
        && try_fuse_ext_operand(func, plan, info, (b_idx, pos, ty), x)
}

fn uext_inner(func: &Function, val: Value, def: &dyn Fn(Value) -> Option<Inst>) -> Option<Value> {
    match func.insts[def(val)?.index()].op {
        Op::Convert(Convert::Uextend, inner) => Some(inner),
        _ => None,
    }
}

fn swap_parts(
    func: &Function,
    plan: &Plan<Reg>,
    (x, y): (Value, Value),
    def: &dyn Fn(Value) -> Option<Inst>,
) -> bool {
    uext_inner(func, x, def).is_some_and(|a| !zero_ext_loc(plan, a))
        && uext_inner(func, y, def).is_some_and(|b| zero_ext_loc(plan, b))
}

/// Reports whether an add extends `x` in its operand and reads `y` as an already zero-extended register.
pub(in super::super) fn ext_swapped(
    func: &Function,
    plan: &Plan<Reg>,
    (x, y): (Value, Value),
    def: &dyn Fn(Value) -> Option<Inst>,
) -> bool {
    plan.elided[x.index()] && plan.elided[y.index()] && swap_parts(func, plan, (x, y), def)
}

fn swap_ext_operand(
    func: &Function,
    plan: &mut Plan<Reg>,
    info @ (_, pos_of): (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos, ty): (usize, usize, Type),
    (x, y): (Value, Value),
) {
    if swap_parts(func, plan, (x, y), &|v| def_inst(pos_of, v)) {
        try_fuse_ext_operand(func, plan, info, (b_idx, pos, ty), x);
    }
}

fn peel_binary_uextend(
    func: &Function,
    plan: &mut Plan<Reg>,
    info: (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos, res): (usize, usize, Value),
    (alu_op, x, y): (Binary, Value, Value),
) {
    let ty = func.ty(res);
    if !matches!(ty, Type::I32 | Type::I64) {
        return;
    }
    let ok = match alu_op {
        Binary::Iadd | Binary::Isub | Binary::Bor | Binary::Bxor => true,
        Binary::Band => [x, y].into_iter().any(|v| {
            plan.const_of(v)
                .is_some_and(|imm| encode_bitmask_imm(imm as u64, ty == Type::I64).is_some())
        }),
        _ => false,
    };
    if !ok {
        return;
    }
    peel_other_uextend(func, plan, info, (b_idx, pos, res), x);
    if !plan.elided[x.index()] || alu_op != Binary::Isub {
        peel_other_uextend(func, plan, info, (b_idx, pos, res), y);
    }
}

pub(super) fn fuse_int_binary(
    func: &Function,
    plan: &mut Plan<Reg>,
    info: (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos, res): (usize, usize, Value),
    (alu_op, x, y): (Binary, Value, Value),
) {
    let wide = matches!(func.ty(res), Type::I32 | Type::I64);
    if (wide && fuse_int_binary_wide(func, plan, info, (b_idx, pos, res), (alu_op, x, y)))
        || (alu_op == Binary::Band
            && bits::try_fuse_bfx(func, plan, info, (b_idx, pos, res), (alu_op, x, y)))
        || shift::try_fuse_bnot(func, plan, info, (b_idx, pos, res), (alu_op, x, y))
        || fuse_int_binary_ext(func, plan, info, (b_idx, pos, res), (alu_op, x, y))
    {
        return;
    }
    fuse_shifted_alu(func, plan, info, (b_idx, pos, res), (alu_op, x, y));
    if !plan.elided[x.index()] && !plan.elided[y.index()] {
        peel_binary_uextend(func, plan, info, (b_idx, pos, res), (alu_op, x, y));
    }
}

pub(super) fn fuse_block_unary(
    func: &Function,
    plan: &mut Plan<Reg>,
    info: (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos, res): (usize, usize, Value),
    (unary_op, inner): (Unary, Value),
) {
    match unary_op {
        Unary::Bnot => {
            if !try_fuse_eon(func, plan, info, (b_idx, pos, res), inner)
                && func.ty(res).is_small_int()
            {
                try_fuse_shift(
                    func,
                    plan,
                    info,
                    (b_idx, pos, res, Binary::Bxor),
                    (inner, res),
                );
            }
        }
        Unary::Ineg if matches!(func.ty(res), Type::I32 | Type::I64) => {
            if !try_fuse_flag_operand(func, plan, info, (b_idx, pos), inner)
                && !mem::try_fuse_madd(
                    func,
                    plan,
                    info,
                    (b_idx, pos, res),
                    (Binary::Isub, res, inner),
                )
            {
                try_fuse_shift(
                    func,
                    plan,
                    info,
                    (b_idx, pos, res, Binary::Isub),
                    (inner, res),
                );
            }
        }
        _ => {}
    }
}

pub(super) fn fuse_block_shift(
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
    if matches!(sop, Binary::Ishl | Binary::Ushr)
        && matches!(func.ty(res), Type::I32 | Type::I64)
        && !plan.elided[x.index()]
    {
        peel_other_uextend(func, plan, info, (b_idx, pos, res), x);
    }
}

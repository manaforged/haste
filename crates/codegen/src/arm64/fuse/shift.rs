use super::super::asm::Reg;
use super::bits::UseInfo;
use super::{
    alu_shift, can_peel_uextend, const_of, def_entry, def_inst, fits_width, local_single, peel_other_uextend,
    preserved, shift_parts, shift_uext_ok, zero_ext_loc,
};
use crate::alloc::{Location, Plan};
use crate::ir::{Binary, Convert, Function, Inst, InstData, Op, Type, Unary, Value};

fn multi_shift_binary(
    func: &Function,
    plan: &Plan<Reg>,
    pos_of: &[Option<(usize, usize, Inst)>],
    (ty, sop, imm): (Type, Binary, i64),
    (shift_val, alu_op, x, y): (Value, Binary, Value, Value),
) -> Option<Option<Value>> {
    if alu_op == Binary::Isub && y != shift_val {
        return None;
    }
    let logical = matches!(alu_op, Binary::Band | Binary::Bor | Binary::Bxor);
    alu_shift(logical, ty, sop, imm)?;
    let other = if y == shift_val { x } else { y };
    let def = |v| def_inst(pos_of, v);
    (!plan.elided[other.index()] && const_of(func, plan, other, &def).is_none()).then_some(Some(other))
}

fn multi_shift_icmp(
    func: &Function,
    plan: &Plan<Reg>,
    pos_of: &[Option<(usize, usize, Inst)>],
    (ty, sop, imm): (Type, Binary, i64),
    (shift_val, a, b): (Value, Value, Value),
) -> Option<Option<Value>> {
    if !matches!(ty, Type::I32 | Type::I64) {
        return None;
    }
    alu_shift(false, ty, sop, imm)?;
    let other = if a == shift_val { b } else { a };
    let def = |v| def_inst(pos_of, v);
    (!plan.elided[other.index()] && const_of(func, plan, other, &def).is_none()).then_some(None)
}

fn multi_shift_consumer(
    func: &Function,
    plan: &Plan<Reg>,
    pos_of: &[Option<(usize, usize, Inst)>],
    (ty, sop, imm): (Type, Binary, i64),
    (shift_val, ud): (Value, &InstData),
) -> Option<Option<Value>> {
    let u_res = ud.result?;
    if plan.elided[u_res.index()] {
        return None;
    }
    match ud.op {
        Op::Binary(alu_op @ (Binary::Band | Binary::Bor | Binary::Bxor | Binary::Iadd | Binary::Isub), x, y) => {
            (func.ty(u_res) == ty)
                .then(|| multi_shift_binary(func, plan, pos_of, (ty, sop, imm), (shift_val, alu_op, x, y)))
                .flatten()
        }
        Op::Unary(Unary::Bnot, x) if x == shift_val && func.ty(u_res) == ty => alu_shift(true, ty, sop, imm).map(|_| None),
        Op::Unary(Unary::Ineg, x) if x == shift_val && matches!(ty, Type::I32 | Type::I64) && func.ty(u_res) == ty => {
            alu_shift(false, ty, sop, imm).map(|_| None)
        }
        Op::Icmp(_, a, b) => multi_shift_icmp(func, plan, pos_of, (ty, sop, imm), (shift_val, a, b)),
        _ => None,
    }
}

fn find_two_shift_uses(
    func: &Function,
    plan: &Plan<Reg>,
    pos_of: &[Option<(usize, usize, Inst)>],
    (b_idx, spos, shift_val): (usize, usize, Value),
    sig: (Type, Binary, i64),
) -> Option<[(usize, Value, Option<Value>); 2]> {
    let mut found = [None, None];
    let mut count = 0_usize;
    for (u_pos, &ui) in func.blocks[b_idx].insts.iter().enumerate().skip(spos + 1) {
        let ud = &func.insts[ui.index()];
        let mut occ = 0_u32;
        ud.op.for_operands(|v| occ += u32::from(v == shift_val));
        if occ == 0 {
            continue;
        }
        if occ > 1 || count >= 2 {
            return None;
        }
        let other = multi_shift_consumer(func, plan, pos_of, sig, (shift_val, ud))?;
        found[count] = Some((u_pos, ud.result?, other));
        count += 1;
    }
    if count == 2 { Some([found[0]?, found[1]?]) } else { None }
}

pub(super) fn try_fuse_multi_shift(
    func: &Function,
    plan: &mut Plan<Reg>,
    info @ (uses, pos_of): UseInfo<'_>,
    (b_idx, spos, shift_val): (usize, usize, Value),
    op: &Op,
) -> bool {
    if uses[shift_val.index()] != 2 || plan.elided[shift_val.index()] || plan.deferred[shift_val.index()] || plan.is_static(shift_val) {
        return false;
    }
    let Some((sop, b, imm)) = shift_parts(op, plan) else { return false };
    let ty = func.ty(shift_val);
    if !ty.is_small_int() {
        return false;
    }
    let Some([(u1_pos, u1_res, o1), (u2_pos, u2_res, o2)]) = find_two_shift_uses(func, plan, pos_of, (b_idx, spos, shift_val), (ty, sop, imm)) else {
        return false;
    };
    let shift_ok = |inner: Value| shift_uext_ok(func, plan, ty.bits(), (sop, imm), inner);
    let peeled = can_peel_uextend(func, plan, info, (b_idx, spos, u2_pos, Some(spos), shift_val), (b, &shift_ok));
    if !peeled && (plan.elided[b.index()] || !preserved(func, plan, b_idx, spos, u2_pos, None, b)) {
        return false;
    }
    plan.elided[shift_val.index()] = true;
    if peeled {
        plan.elided[b.index()] = true;
    }
    if let Some(other) = o1 {
        peel_other_uextend(func, plan, info, (b_idx, u1_pos, u1_res), other);
    }
    if let Some(other) = o2 {
        peel_other_uextend(func, plan, info, (b_idx, u2_pos, u2_res), other);
    }
    true
}

fn high_bits_zero(
    func: &Function,
    plan: &Plan<Reg>,
    pos_of: &[Option<(usize, usize, Inst)>],
    val: Value,
    u_bits: u32,
) -> bool {
    let def = |v| def_inst(pos_of, v);
    let ty = func.ty(val);
    let (ty_bits, ty_mask) = (ty.bits(), ty.unsigned_max());
    let fits_u = |c: i64| fits_width(((c as u64) & ty_mask) as i64, u_bits, false);
    if let Some(c) = const_of(func, plan, val, &def) {
        return fits_u(c);
    }
    let Some(inst) = def(val) else { return false };
    match func.insts[inst.index()].op {
        Op::Convert(Convert::Uextend, inner) => func.ty(inner).is_small_int() && func.ty(inner).bits() <= u_bits,
        Op::Binary(Binary::Band, a, b) => [a, b].into_iter().any(|mv| {
            const_of(func, plan, mv, &def).is_some_and(fits_u)
        }),
        Op::Binary(Binary::Ushr, _, amt) => const_of(func, plan, amt, &def)
            .is_some_and(|c| ((c as u32) & (ty_bits - 1)) >= ty_bits - u_bits),
        Op::Icmp(..) | Op::Fcmp(..) | Op::Fits(..) => true,
        _ => false,
    }
}

fn try_fuse_bnot_uext(
    func: &Function,
    plan: &mut Plan<Reg>,
    info @ (_, pos_of): UseInfo<'_>,
    (b_idx, npos, pos, res, op): (usize, usize, usize, Value, Binary),
    (other, neg, inner): (Value, Value, Value),
) -> bool {
    let Some((upos, uinst)) = local_single(plan, info, b_idx, npos, inner) else { return false };
    let Op::Convert(Convert::Uextend, u_inner) = func.insts[uinst.index()].op else { return false };
    if u_inner == res || plan.elided[u_inner.index()] || !matches!(plan.location(u_inner), Location::Reg(_) | Location::Slot) {
        return false;
    }
    let ok = zero_ext_loc(plan, u_inner)
        || (op == Binary::Band && high_bits_zero(func, plan, pos_of, other, func.ty(u_inner).bits()));
    if !ok || !preserved(func, plan, b_idx, upos, pos, Some(npos), u_inner) {
        return false;
    }
    plan.elided[neg.index()] = true;
    plan.elided[inner.index()] = true;
    peel_other_uextend(func, plan, info, (b_idx, pos, res), other);
    true
}

fn try_fuse_bnot_elided_shift(
    func: &Function,
    plan: &mut Plan<Reg>,
    info @ (_, pos_of): UseInfo<'_>,
    (b_idx, npos, pos, res): (usize, usize, usize, Value),
    (other, neg, inner): (Value, Value, Value),
) -> bool {
    let ty = func.ty(res);
    let Some((sb, spos, sinst)) = def_entry(pos_of, inner) else { return false };
    if sb != b_idx || spos >= npos {
        return false;
    }
    let Some((sop, b, imm)) = shift_parts(&func.insts[sinst.index()].op, plan) else { return false };
    if alu_shift(true, ty, sop, imm).is_none() {
        return false;
    }
    let root = if !plan.elided[b.index()] {
        b
    } else {
        let Some((ub, upos, uinst)) = def_entry(pos_of, b) else { return false };
        let Op::Convert(Convert::Uextend, u_inner) = func.insts[uinst.index()].op else { return false };
        if ub != b_idx || upos >= spos || plan.elided[u_inner.index()] || !shift_uext_ok(func, plan, ty.bits(), (sop, imm), u_inner) {
            return false;
        }
        u_inner
    };
    if root == res || !preserved(func, plan, b_idx, npos, pos, None, root) {
        return false;
    }
    plan.elided[neg.index()] = true;
    peel_other_uextend(func, plan, info, (b_idx, pos, res), other);
    true
}

fn try_fuse_bnot_shift(
    func: &Function,
    plan: &mut Plan<Reg>,
    info: UseInfo<'_>,
    (b_idx, npos, pos, res, ty): (usize, usize, usize, Value, Type),
    (other, neg, inner): (Value, Value, Value),
) -> bool {
    let Some((spos, sinst)) = local_single(plan, info, b_idx, npos, inner) else { return false };
    let Some((sop, b, imm)) = shift_parts(&func.insts[sinst.index()].op, plan) else { return false };
    if alu_shift(true, ty, sop, imm).is_none() || b == res {
        return false;
    }
    let ty_bits = ty.bits();
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
    if !peel_b && !preserved(func, plan, b_idx, spos, pos, Some(npos), b) {
        return false;
    }
    plan.elided[neg.index()] = true;
    plan.elided[inner.index()] = true;
    if peel_b {
        plan.elided[b.index()] = true;
    }
    peel_other_uextend(func, plan, info, (b_idx, pos, res), other);
    true
}

pub(super) fn try_fuse_bnot(
    func: &Function,
    plan: &mut Plan<Reg>,
    info: UseInfo<'_>,
    (b_idx, pos, res): (usize, usize, Value),
    (op, x, y): (Binary, Value, Value),
) -> bool {
    if !matches!(op, Binary::Band | Binary::Bor | Binary::Bxor) {
        return false;
    }
    let ty = func.ty(res);
    for (other, neg) in [(x, y), (y, x)] {
        if plan.elided[other.index()] {
            continue;
        }
        let Some((npos, ninst)) = local_single(plan, info, b_idx, pos, neg) else { continue };
        let Op::Unary(Unary::Bnot, inner) = func.insts[ninst.index()].op else { continue };
        if plan.elided[inner.index()] {
            if try_fuse_bnot_elided_shift(func, plan, info, (b_idx, npos, pos, res), (other, neg, inner)) {
                return true;
            }
            continue;
        }
        if try_fuse_bnot_shift(func, plan, info, (b_idx, npos, pos, res, ty), (other, neg, inner))
            || try_fuse_bnot_uext(func, plan, info, (b_idx, npos, pos, res, op), (other, neg, inner))
        {
            return true;
        }
        if preserved(func, plan, b_idx, npos, pos, None, inner) {
            plan.elided[neg.index()] = true;
            peel_other_uextend(func, plan, info, (b_idx, pos, res), other);
            return true;
        }
    }
    false
}

use super::super::asm::{Reg, Shift};
use super::super::int::add_imm_parts;
use super::alu::{try_fuse_ext_operand, try_fuse_shift_ty};
use super::flag::try_fuse_icmp_select;
use super::mem::try_fuse_icmp_sload;
use super::{alu_shift, const_of, def_entry, def_inst, local_single, preserved, shift_parts};
use crate::alloc::Plan;
use crate::ir::{Binary, Function, Inst, IntCC, Op, Type, Unary, Value};

fn neg_inner(
    func: &Function,
    plan: &Plan<Reg>,
    op: &Op,
    def: &dyn Fn(Value) -> Option<Inst>,
) -> Option<Value> {
    match *op {
        Op::Unary(Unary::Ineg, inner) => Some(inner),
        Op::Binary(Binary::Isub, zero, inner) if const_of(func, plan, zero, def) == Some(0) => {
            Some(inner)
        }
        _ => None,
    }
}

pub(in super::super) fn cmn_parts(
    func: &Function,
    plan: &Plan<Reg>,
    ty: Type,
    val: Value,
    def: &dyn Fn(Value) -> Option<Inst>,
) -> Option<(Value, Option<(Shift, u8)>)> {
    if !matches!(ty, Type::I32 | Type::I64) || !plan.elided[val.index()] {
        return None;
    }
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

pub(in super::super) fn cmn_sum_parts(
    func: &Function,
    plan: &Plan<Reg>,
    (x, y): (Value, Value),
    def: &dyn Fn(Value) -> Option<Inst>,
) -> Option<(Value, Value)> {
    let sum = (const_of(func, plan, y, def) == Some(0))
        .then_some(x)
        .or_else(|| (const_of(func, plan, x, def) == Some(0)).then_some(y))?;
    if !plan.elided[sum.index()] {
        return None;
    }
    let Op::Binary(Binary::Iadd, a, b) = func.insts[def(sum)?.index()].op else {
        return None;
    };
    (!plan.elided[a.index()] && !plan.elided[b.index()]).then_some((a, b))
}

fn try_fuse_cmn_neg(
    func: &Function,
    plan: &mut Plan<Reg>,
    info @ (_, pos_of): (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos, cmp_ty): (usize, usize, Type),
    cand: Value,
) -> bool {
    let Some((npos, ninst)) = local_single(plan, info, b_idx, pos, cand) else {
        return false;
    };
    let def = |v| def_inst(pos_of, v);
    let Some(inner) = neg_inner(func, plan, &func.insts[ninst.index()].op, &def) else {
        return false;
    };
    if !plan.elided[inner.index()] {
        if preserved(func, plan, b_idx, npos, pos, None, inner) {
            plan.elided[cand.index()] = true;
            return true;
        }
        return false;
    }
    if let Some((sb, spos, sinst)) = def_entry(pos_of, inner)
        && sb == b_idx
        && spos < npos
        && let Some((sop, root, imm)) = shift_parts(&func.insts[sinst.index()].op, plan)
        && alu_shift(false, cmp_ty, sop, imm).is_some()
        && !plan.elided[root.index()]
        && preserved(func, plan, b_idx, spos, pos, Some(npos), root)
    {
        plan.elided[cand.index()] = true;
        return true;
    }
    false
}

fn try_fuse_cmn_sum(
    func: &Function,
    plan: &mut Plan<Reg>,
    info @ (_, pos_of): (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos): (usize, usize),
    (lhs, rhs): (Value, Value),
) -> bool {
    let def = |v| def_inst(pos_of, v);
    let Some(sum) = (const_of(func, plan, rhs, &def) == Some(0))
        .then_some(lhs)
        .or_else(|| (const_of(func, plan, lhs, &def) == Some(0)).then_some(rhs))
    else {
        return false;
    };
    let Some((spos, sinst)) = local_single(plan, info, b_idx, pos, sum) else {
        return false;
    };
    let Op::Binary(Binary::Iadd, a, b) = func.insts[sinst.index()].op else {
        return false;
    };
    if plan.elided[a.index()]
        || plan.elided[b.index()]
        || const_of(func, plan, a, &def).is_some()
        || const_of(func, plan, b, &def).is_some()
        || !preserved(func, plan, b_idx, spos, pos, None, a)
        || !preserved(func, plan, b_idx, spos, pos, None, b)
    {
        return false;
    }
    plan.elided[sum.index()] = true;
    true
}

fn fuse_icmp_operands(
    func: &Function,
    plan: &mut Plan<Reg>,
    info: (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos, res): (usize, usize, Value),
    (cc, a, b): (IntCC, Value, Value),
) {
    if plan.elided[a.index()] || plan.elided[b.index()] {
        return;
    }
    let ty = func.ty(a);
    let cmn_eq = matches!(cc, IntCC::Equal | IntCC::NotEqual);
    if cmn_eq && try_fuse_cmn_sum(func, plan, info, (b_idx, pos), (a, b)) {
        return;
    }
    let imm_check = |p: &Plan<Reg>, v: Value| {
        p.const_of(v)
            .and_then(|imm| {
                add_imm_parts(true, crate::fold::extend(imm, func.ty(v), cc.is_signed()))
            })
            .is_some()
    };
    if imm_check(plan, b)
        || try_fuse_ext_operand(func, plan, info, (b_idx, pos, ty), b)
        || try_fuse_ext_operand(func, plan, info, (b_idx, pos, ty), a)
        || imm_check(plan, a)
    {
        return;
    }
    if cmn_eq
        && (try_fuse_cmn_neg(func, plan, info, (b_idx, pos, ty), b)
            || try_fuse_cmn_neg(func, plan, info, (b_idx, pos, ty), a))
    {
        return;
    }
    if !try_fuse_shift_ty(func, plan, info, (b_idx, pos, res, ty, false), (b, a)) {
        try_fuse_shift_ty(func, plan, info, (b_idx, pos, res, ty, false), (a, b));
    }
}

pub(super) fn fuse_block_icmp(
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
        try_fuse_icmp_sload(func, plan, info, (b_idx, pos), (cc, a, b));
    }
}

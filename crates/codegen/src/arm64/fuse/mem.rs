use super::{local_single, preserved, shift_parts, zero_ext_loc};
use crate::alloc::Plan;
use crate::ir::{Binary, Convert, Function, Inst, Op, Type, Value};
use crate::live::{Sib, SibBase};
use super::super::asm::Reg;

pub(super) fn try_fuse_sib(
    func: &Function,
    plan: &mut Plan<Reg>,
    (uses, pos_of): (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos, res): (usize, usize, Value),
    (x, y): (Value, Value),
) -> bool {
    if func.ty(res) != Type::I64 || plan.sib[res.index()].is_some() || uses[res.index()] == 0 {
        return false;
    }
    let block = &func.blocks[b_idx];
    let try_sib = |plan: &mut Plan<Reg>, base: Value, idx: Value, shift: i64, idx_pos: usize, elide: [Option<Value>; 2]| -> bool {
        if plan.reg_of(base).is_none() || plan.reg_of(idx).is_none() {
            return false;
        }
        let (mut seen_uses, mut last_pos) = (0_u32, pos);
        for (u_pos, &u_inst) in block.insts.iter().enumerate().skip(pos + 1) {
            let u_data = &func.insts[u_inst.index()];
            let ty = match (&u_data.op, u_data.result) {
                (&Op::Load(addr, 0), Some(r)) if addr == res => Some(func.ty(r)),
                (&Op::Store(val, addr, 0), _) if addr == res && val != res => Some(func.ty(val)),
                _ => None,
            };
            if ty.is_some_and(|t| t != Type::I128 && (shift == 0 || t.bytes() == (1 << shift))) {
                seen_uses += 1;
                last_pos = u_pos;
            }
        }
        if seen_uses != uses[res.index()] {
            return false;
        }
        plan.elided[res.index()] = true;
        let mut extra = [None; 2];
        for (slot, v) in extra.iter_mut().zip(elide) {
            if let Some(ev) = v
                && uses[ev.index()] == 1
                && !plan.elided[ev.index()]
            {
                plan.elided[ev.index()] = true;
                *slot = Some(ev);
            }
        }
        if preserved(func, plan, b_idx, pos, last_pos, None, base)
            && preserved(func, plan, b_idx, idx_pos, last_pos, None, idx)
        {
            plan.sib[res.index()] = Some(Sib { base: SibBase::Value(base), index: idx, shift: shift as u8, disp: 0 });
            return true;
        }
        plan.elided[res.index()] = false;
        for ev in extra.into_iter().flatten() {
            plan.elided[ev.index()] = false;
        }
        false
    };
    let unwrap_uext = |plan: &Plan<Reg>, v: Value, limit: usize| match local_single(plan, (uses, pos_of), b_idx, limit, v) {
        Some((epos, einst))
            if let Op::Convert(Convert::Uextend, inner) = func.insts[einst.index()].op
                && func.ty(inner).is_small_int()
                && zero_ext_loc(plan, inner) =>
        {
            (inner, epos, Some(v))
        }
        _ => (v, limit, None),
    };
    for (base, rhs) in [(x, y), (y, x)] {
        if let Some((sb, spos, sinst)) = pos_of[rhs.index()]
            && sb == b_idx
            && spos < pos
            && let Some((Binary::Ishl, mid, shift)) = shift_parts(&func.insts[sinst.index()].op, plan)
            && (1..=3).contains(&shift)
        {
            let (idx, idx_pos, uext) = unwrap_uext(plan, mid, spos);
            if try_sib(plan, base, idx, shift, idx_pos, [Some(rhs), uext]) {
                return true;
            }
        }
        let (idx, idx_pos, uext) = unwrap_uext(plan, rhs, pos);
        if uext.is_some() && try_sib(plan, base, idx, 0, idx_pos, [uext, None]) {
            return true;
        }
    }
    try_sib(plan, x, y, 0, pos, [None, None])
}

fn trivial_mul_imm(func: &Function, plan: &Plan<Reg>, ty: Type, val: Value) -> bool {
    let Some(raw) = plan.const_of(val) else {
        return false;
    };
    let uimm = crate::fold::extend(raw, ty, false) as u64;
    let simm = crate::fold::extend(raw, ty, true);
    let max_shift = if ty == Type::I64 { 63 } else { 31 };
    uimm <= 1
        || simm == -1
        || uimm.is_power_of_two()
        || (uimm - 1).is_power_of_two()
        || uimm.checked_add(1).is_some_and(|n| n.is_power_of_two() && n.trailing_zeros() as u8 <= max_shift)
        || simm
            .checked_neg()
            .filter(|&n| n > 2)
            .and_then(|n| (n as u64).checked_add(1))
            .is_some_and(|n| n.is_power_of_two() && n.trailing_zeros() as u8 <= max_shift)
}

pub(super) fn try_fuse_madd(
    func: &Function,
    plan: &mut Plan<Reg>,
    info: (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos, res): (usize, usize, Value),
    (alu_op, x, y): (Binary, Value, Value),
) -> bool {
    if !matches!(alu_op, Binary::Iadd | Binary::Isub) {
        return false;
    }
    let ty = func.ty(res);
    let candidates = if alu_op == Binary::Isub { [Some((y, x)), None] } else { [Some((y, x)), Some((x, y))] };
    for (mul, acc) in candidates.into_iter().flatten() {
        if plan.elided[acc.index()] {
            continue;
        }
        let Some((mpos, minst)) = local_single(plan, info, b_idx, pos, mul) else {
            continue;
        };
        let Op::Binary(Binary::Imul, a, b) = func.insts[minst.index()].op else {
            continue;
        };
        if trivial_mul_imm(func, plan, ty, a) || trivial_mul_imm(func, plan, ty, b) {
            continue;
        }
        plan.elided[mul.index()] = true;
        let ok = preserved(func, plan, b_idx, mpos, pos, None, a) && preserved(func, plan, b_idx, mpos, pos, None, b);
        if ok {
            return true;
        }
        plan.elided[mul.index()] = false;
    }
    false
}

/// Returns `(sub, a, b, acc)` when `x` and `y` under `op` emit `madd`/`msub` with an elided `Imul`.
pub(in super::super) fn madd_parts(
    func: &Function,
    plan: &Plan<Reg>,
    ty: Type,
    op: Binary,
    x: Value,
    y: Value,
    def: impl Fn(Value) -> Option<Inst>,
) -> Option<(bool, Value, Value, Value)> {
    if !matches!(op, Binary::Iadd | Binary::Isub) || !matches!(ty, Type::I32 | Type::I64) {
        return None;
    }
    let sub = op == Binary::Isub;
    let candidates = if sub { [Some((y, x)), None] } else { [Some((y, x)), Some((x, y))] };
    for (mul, acc) in candidates.into_iter().flatten() {
        if !plan.elided[mul.index()] || plan.elided[acc.index()] {
            continue;
        }
        let Some(inst) = def(mul) else { continue };
        if let Op::Binary(Binary::Imul, a, b) = func.insts[inst.index()].op
            && !plan.elided[a.index()]
            && !plan.elided[b.index()]
            && !trivial_mul_imm(func, plan, ty, a)
            && !trivial_mul_imm(func, plan, ty, b)
        {
            return Some((sub, a, b, acc));
        }
    }
    None
}

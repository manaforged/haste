use super::super::{
    const_of, def_inst, elided_use_end, local_single, preserved, sign, zero_cmp_band,
};
use super::{PosMap, UseInfo};
use crate::alloc::Plan;
use crate::arm64::asm::Reg;
use crate::arm64::int::encode_bitmask_imm;
use crate::ir::{Binary, Convert, Function, Inst, IntCC, Op, Type, Value};

fn find_def(func: &Function, val: Value) -> Option<Inst> {
    func.blocks
        .iter()
        .flat_map(|b| &b.insts)
        .copied()
        .find(|&i| func.insts[i.index()].result == Some(val))
}

fn func_const(func: &Function, plan: &Plan<Reg>, val: Value) -> Option<i64> {
    const_of(func, plan, val, |v| find_def(func, v))
}

fn peel_bit_op(
    func: &Function,
    plan: &Plan<Reg>,
    val: Value,
    bit: u8,
    def: &dyn Fn(Value) -> Option<Inst>,
) -> Option<(Value, u8)> {
    let ty = func.ty(val);
    let bits = ty.bits();
    match func.insts[def(val)?.index()].op {
        Op::Binary(Binary::Ushr, inner, amt) => {
            let k = (const_of(func, plan, amt, def)? as u32) & (bits - 1);
            (u32::from(bit) + k < bits).then_some((inner, bit + k as u8))
        }
        Op::Binary(Binary::Sshr, inner, amt) => {
            let k = (const_of(func, plan, amt, def)? as u32) & (bits - 1);
            Some((inner, (u32::from(bit) + k).min(bits - 1) as u8))
        }
        Op::Binary(Binary::Ishl, inner, amt) => {
            let k = (const_of(func, plan, amt, def)? as u32) & (bits - 1);
            (u32::from(bit) >= k).then_some((inner, bit - k as u8))
        }
        Op::Convert(Convert::Uextend, inner) => {
            (u32::from(bit) < func.ty(inner).bits()).then_some((inner, bit))
        }
        Op::Convert(Convert::Sextend, inner) => {
            Some((inner, bit.min(func.ty(inner).bits() as u8 - 1)))
        }
        Op::Convert(Convert::Ireduce, inner) => (u32::from(bit) < bits).then_some((inner, bit)),
        _ => None,
    }
}

pub(in crate::arm64) fn single_bit_raw(
    func: &Function,
    plan: &Plan<Reg>,
    x: Value,
    y: Value,
) -> Option<(Value, u8)> {
    for (val, mask_val) in [(x, y), (y, x)] {
        let ty = func.ty(val);
        if ty.is_small_int()
            && let Some(imm) = func_const(func, plan, mask_val)
        {
            let u = (imm as u64) & ty.unsigned_max();
            if u.is_power_of_two() {
                return Some((val, u.trailing_zeros() as u8));
            }
        }
    }
    None
}

pub(in crate::arm64::fuse) fn peel_elided(
    func: &Function,
    plan: &Plan<Reg>,
    (mut val, mut bit): (Value, u8),
    def: &dyn Fn(Value) -> Option<Inst>,
) -> Option<(Value, u8)> {
    while plan.elided[val.index()] {
        (val, bit) = peel_bit_op(func, plan, val, bit, def)?;
    }
    Some((val, bit))
}

pub(in crate::arm64) fn single_bit_mask(
    func: &Function,
    plan: &Plan<Reg>,
    x: Value,
    y: Value,
) -> Option<(Value, u8)> {
    peel_elided(
        func,
        plan,
        single_bit_raw(func, plan, x, y)?,
        &|v: Value| find_def(func, v),
    )
}

fn can_elide_bit(
    func: &Function,
    plan: &Plan<Reg>,
    info @ (_, pos_of): UseInfo<'_>,
    (b_idx, mpos, end_pos, skip_pos): (usize, usize, usize, Option<usize>),
    (val, bit): (Value, u8),
) -> Option<Option<Value>> {
    let def = |v| def_inst(pos_of, v);
    if plan.elided[val.index()] {
        let (root, _) = peel_elided(func, plan, (val, bit), &def)?;
        return preserved(func, plan, b_idx, mpos, end_pos, skip_pos, root).then_some(None);
    }
    if let Some((spos, _)) = local_single(plan, info, b_idx, mpos, val)
        && let Some((inner, _)) = peel_bit_op(func, plan, val, bit, &def)
        && !plan.elided[inner.index()]
        && preserved(func, plan, b_idx, spos, mpos, None, inner)
        && preserved(func, plan, b_idx, mpos, end_pos, skip_pos, inner)
    {
        return Some(Some(val));
    }
    preserved(func, plan, b_idx, mpos, end_pos, skip_pos, val).then_some(None)
}

pub(in crate::arm64::fuse) fn can_elide_tst_band(
    func: &Function,
    plan: &Plan<Reg>,
    pos_of: PosMap<'_>,
    (b_idx, mpos, end_pos, skip_pos): (usize, usize, usize, Option<usize>),
    (x, y): (Value, Value),
) -> bool {
    let ty = func.ty(x);
    if !ty.is_small_int() {
        return false;
    }
    if preserved(func, plan, b_idx, mpos, end_pos, skip_pos, x)
        && preserved(func, plan, b_idx, mpos, end_pos, skip_pos, y)
    {
        return true;
    }
    let def = |v| def_inst(pos_of, v);
    let ty_mask = ty.unsigned_max();
    for (val, mask_val) in [(x, y), (y, x)] {
        if plan.elided[val.index()]
            && let Some(imm) = const_of(func, plan, mask_val, &def).map(|v| (v as u64) & ty_mask)
            && let Some((root, eff)) = unwrap_elided_tst(func, plan, val, imm, &def)
            && encode_bitmask_imm(eff, ty == Type::I64).is_some()
        {
            return preserved(func, plan, b_idx, mpos, end_pos, skip_pos, root);
        }
    }
    false
}

pub(in crate::arm64) fn unwrap_elided_tst(
    func: &Function,
    plan: &Plan<Reg>,
    mut val: Value,
    mut imm: u64,
    def: &dyn Fn(Value) -> Option<Inst>,
) -> Option<(Value, u64)> {
    while plan.elided[val.index()] {
        let ty = func.ty(val);
        let bits = ty.bits();
        match func.insts[def(val)?.index()].op {
            Op::Binary(Binary::Ushr, inner, amt) => {
                let k = (const_of(func, plan, amt, def)? as u32) & (bits - 1);
                if imm.leading_zeros() < (64 - bits) + k {
                    return None;
                }
                val = inner;
                imm <<= k;
            }
            Op::Binary(Binary::Ishl, inner, amt) => {
                let k = (const_of(func, plan, amt, def)? as u32) & (bits - 1);
                if k > 0 && (imm & ((1_u64 << k) - 1)) != 0 {
                    return None;
                }
                val = inner;
                imm >>= k;
            }
            Op::Convert(Convert::Uextend, inner) => {
                if imm > func.ty(inner).unsigned_max() {
                    return None;
                }
                val = inner;
            }
            _ => return None,
        }
    }
    Some((val, imm))
}

fn check_branch_tst(
    func: &Function,
    plan: &Plan<Reg>,
    info @ (uses, pos_of): UseInfo<'_>,
    (b_idx, pos, cond_val): (usize, usize, Value),
) -> Option<(Value, Option<Value>, Option<Value>)> {
    let (cpos, cinst) = local_single(plan, info, b_idx, pos, cond_val)?;
    if func.size() > 2000 && pos - cpos > 64 {
        return None;
    }
    match func.insts[cinst.index()].op {
        Op::Binary(Binary::Band, x, y) => {
            if let Some(extra) = single_bit_raw(func, plan, x, y)
                .and_then(|b| can_elide_bit(func, plan, info, (b_idx, cpos, pos, None), b))
            {
                return Some((cond_val, None, extra));
            }
            can_elide_tst_band(func, plan, pos_of, (b_idx, cpos, pos, None), (x, y))
                .then_some((cond_val, None, None))
        }
        Op::Convert(Convert::Ireduce, inner) => {
            let (mpos, minst) = local_single(plan, info, b_idx, cpos, inner)?;
            let Op::Binary(Binary::Band, x, y) = func.insts[minst.index()].op else {
                return None;
            };
            let (val, bit) = single_bit_raw(func, plan, x, y)
                .filter(|&(_, b)| u32::from(b) < func.ty(cond_val).bits())?;
            can_elide_bit(func, plan, info, (b_idx, mpos, pos, Some(cpos)), (val, bit))
                .map(|extra| (cond_val, Some(inner), extra))
        }
        Op::Icmp(IntCC::Equal | IntCC::NotEqual, lhs, rhs) => {
            let (m, mpos, x, y) = zero_cmp_band(func, plan, uses, pos_of, b_idx, lhs, rhs)?;
            if mpos >= cpos || (func.size() > 2000 && pos - mpos > 64) {
                return None;
            }
            if let Some(extra) = single_bit_raw(func, plan, x, y)
                .and_then(|b| can_elide_bit(func, plan, info, (b_idx, mpos, pos, Some(cpos)), b))
            {
                return Some((cond_val, Some(m), extra));
            }
            can_elide_tst_band(func, plan, pos_of, (b_idx, mpos, pos, Some(cpos)), (x, y))
                .then_some((cond_val, Some(m), None))
        }
        Op::Icmp(cc, lhs, rhs) => {
            let (val, bit, _) =
                sign::sign_bit_raw(func, plan, (cc, lhs, rhs), &|v: Value| def_inst(pos_of, v))?;
            can_elide_bit(func, plan, info, (b_idx, cpos, pos, None), (val, bit))
                .map(|extra| (cond_val, None, extra))
        }
        _ => None,
    }
}

type BranchTstHit = (Value, Option<Value>, Option<Value>);
type CmpTstHit = (usize, usize, usize, Value, Value);

fn scan_block_tst(
    func: &Function,
    plan: &Plan<Reg>,
    info: UseInfo<'_>,
    (b_idx, small): (usize, bool),
    (branches, cmps): (&mut Vec<BranchTstHit>, &mut Vec<CmpTstHit>),
) {
    let block = &func.blocks[b_idx];
    for (pos, &inst) in block.insts.iter().enumerate() {
        match func.insts[inst.index()].op {
            Op::Brif(cond_val, ..) | Op::Trapz(cond_val) => {
                if let Some(hit) = check_branch_tst(func, plan, info, (b_idx, pos, cond_val)) {
                    branches.push(hit);
                }
            }
            Op::Icmp(IntCC::Equal | IntCC::NotEqual, lhs, rhs) => {
                if let Some(res) = func.insts[inst.index()].result
                    && let Some(end_pos) = if !plan.elided[res.index()] {
                        Some(pos)
                    } else if small || block.insts.len() - pos <= 64 {
                        elided_use_end(func, plan, &block.insts, pos, res)
                    } else {
                        None
                    }
                {
                    cmps.push((b_idx, pos, end_pos, lhs, rhs));
                }
            }
            _ => {}
        }
    }
}

pub(in crate::arm64::fuse) fn fuse_tst(
    func: &Function,
    plan: &mut Plan<Reg>,
    info @ (uses, pos_of): UseInfo<'_>,
) {
    let small = func.size() <= 2000;
    let (mut branch_tst, mut cmp_tst) = (Vec::new(), Vec::new());
    for b_idx in 0..func.blocks.len() {
        scan_block_tst(
            func,
            plan,
            info,
            (b_idx, small),
            (&mut branch_tst, &mut cmp_tst),
        );
    }
    for (cond_val, m, extra) in branch_tst {
        plan.elided[cond_val.index()] = true;
        if let Some(m) = m {
            plan.elided[m.index()] = true;
        }
        if let Some(e) = extra {
            plan.elided[e.index()] = true;
        }
    }
    for (b_idx, cpos, end_pos, lhs, rhs) in cmp_tst {
        if let Some((m, mpos, x, y)) = zero_cmp_band(func, plan, uses, pos_of, b_idx, lhs, rhs)
            && mpos < cpos
            && (small || end_pos - mpos <= 64)
            && can_elide_tst_band(func, plan, pos_of, (b_idx, mpos, end_pos, None), (x, y))
        {
            plan.elided[m.index()] = true;
        }
    }
}

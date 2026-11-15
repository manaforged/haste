use super::super::asm::Reg;
use super::super::Location;
use super::bits::can_elide_tst_band;
use super::{alu_shift, cmn_parts, cmn_sum_parts, const_of, ext_parts, local_single, preserved, shift_parts, unwrap_elided_tst, zero_cmp_band};
use crate::alloc::Plan;
use crate::ir::{Convert, Function, Inst, IntCC, Op, Type, Value};

fn peel_uext(func: &Function, plan: &Plan<Reg>, pos_of: &[Option<(usize, usize, Inst)>], val: Value) -> Value {
    if plan.elided[val.index()]
        && let Some((_, _, inst)) = pos_of[val.index()]
        && let Op::Convert(Convert::Uextend, inner) = func.insts[inst.index()].op
    {
        return inner;
    }
    val
}

fn cmp_shift_preserved(
    func: &Function,
    plan: &Plan<Reg>,
    pos_of: &[Option<(usize, usize, Inst)>],
    (b_idx, start, pos, skip): (usize, usize, usize, Option<usize>),
    (cmp_ty, v, other): (Type, Value, Value),
) -> bool {
    if !matches!(cmp_ty, Type::I32 | Type::I64) || !plan.elided[v.index()] {
        return false;
    }
    let Some((_, _, inst)) = pos_of[v.index()] else {
        return false;
    };
    let Some((sop, mid, imm)) = shift_parts(&func.insts[inst.index()].op, plan) else {
        return false;
    };
    if alu_shift(false, cmp_ty, sop, imm).is_none() {
        return false;
    }
    let root = peel_uext(func, plan, pos_of, mid);
    let other_root = peel_uext(func, plan, pos_of, other);
    preserved(func, plan, b_idx, start, pos, skip, root) && preserved(func, plan, b_idx, start, pos, skip, other_root)
}

fn cmp_operand_preserved(
    func: &Function,
    plan: &Plan<Reg>,
    pos_of: &[Option<(usize, usize, Inst)>],
    (b_idx, start, pos, skip): (usize, usize, usize, Option<usize>),
    (cmp_ty, cc, v): (Type, IntCC, Value),
) -> bool {
    if !plan.elided[v.index()] {
        return preserved(func, plan, b_idx, start, pos, skip, v);
    }
    let def = |val: Value| pos_of[val.index()].map(|(_, _, i)| i);
    if let Some((_, _, inner)) = ext_parts(func, plan, cmp_ty, v, def) {
        return preserved(func, plan, b_idx, start, pos, skip, inner);
    }
    if matches!(cc, IntCC::Equal | IntCC::NotEqual)
        && matches!(cmp_ty, Type::I32 | Type::I64)
        && let Some((root, _)) = cmn_parts(func, plan, cmp_ty, v, &def)
    {
        return preserved(func, plan, b_idx, start, pos, skip, root);
    }
    false
}

fn tst_single_insn(func: &Function, plan: &Plan<Reg>, pos_of: &[Option<(usize, usize, Inst)>], (x, y): (Value, Value)) -> bool {
    let def = |v: Value| pos_of[v.index()].map(|(_, _, i)| i);
    let ty = func.ty(x);
    let ty_mask = ty.unsigned_max();
    for (raw, mask) in [(x, y), (y, x)] {
        if let Some(imm) = const_of(func, plan, mask, &def) {
            return unwrap_elided_tst(func, plan, raw, (imm as u64) & ty_mask, &def)
                .is_some_and(|(_, eff)| super::super::int::encode_bitmask_imm(eff, ty == Type::I64).is_some());
        }
    }
    true
}

fn cmp_band_tst(
    func: &Function,
    plan: &Plan<Reg>,
    (uses, pos_of): (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, start, pos, skip): (usize, usize, usize, Option<usize>),
    (cc, a, b, cmp_uses): (IntCC, Value, Value, u32),
) -> Option<Value> {
    if !matches!(cc, IntCC::Equal | IntCC::NotEqual) {
        return None;
    }
    let (m, mpos, bx, by) = zero_cmp_band(func, plan, uses, pos_of, b_idx, a, b)?;
    let ok = mpos < start
        && can_elide_tst_band(func, plan, pos_of, (b_idx, mpos, pos, skip), (bx, by))
        && (cmp_uses == 1 || tst_single_insn(func, plan, pos_of, (bx, by)));
    ok.then_some(m)
}

fn try_cmp_preserved(
    func: &Function,
    plan: &Plan<Reg>,
    info @ (_, pos_of): (&[u32], &[Option<(usize, usize, Inst)>]),
    span @ (b_idx, start, pos, skip): (usize, usize, usize, Option<usize>),
    (cc, a, b, cmp_uses): (IntCC, Value, Value, u32),
) -> Option<Option<Value>> {
    let cmp_ty = func.ty(a);
    if !cmp_ty.is_small_int() {
        return None;
    }
    let def = |val: Value| pos_of[val.index()].map(|(_, _, i)| i);
    if matches!(cc, IntCC::Equal | IntCC::NotEqual)
        && let Some((sa, sb)) = cmn_sum_parts(func, plan, (a, b), &def)
        && preserved(func, plan, b_idx, start, pos, skip, sa)
        && preserved(func, plan, b_idx, start, pos, skip, sb)
    {
        return Some(None);
    }
    if !(plan.elided[a.index()] && plan.elided[b.index()])
        && cmp_operand_preserved(func, plan, pos_of, span, (cmp_ty, cc, a))
        && cmp_operand_preserved(func, plan, pos_of, span, (cmp_ty, cc, b))
    {
        return Some(None);
    }
    if cmp_shift_preserved(func, plan, pos_of, span, (cmp_ty, b, a)) || cmp_shift_preserved(func, plan, pos_of, span, (cmp_ty, a, b)) {
        return Some(None);
    }
    cmp_band_tst(func, plan, info, span, (cc, a, b, cmp_uses)).map(Some)
}

pub(super) fn try_fuse_flag_operand(
    func: &Function,
    plan: &mut Plan<Reg>,
    info: (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos): (usize, usize),
    flag_val: Value,
) -> bool {
    let Some((fpos, finst)) = local_single(plan, info, b_idx, pos, flag_val) else {
        return false;
    };
    match func.insts[finst.index()].op {
        Op::Icmp(cc, a, b) => {
            let Some(band) = try_cmp_preserved(func, plan, info, (b_idx, fpos, pos, None), (cc, a, b, 1)) else {
                return false;
            };
            plan.elided[flag_val.index()] = true;
            if let Some(m) = band {
                plan.elided[m.index()] = true;
            }
            true
        }
        Op::Convert(Convert::Uextend, inner)
            if let Some((ipos, iinst)) = local_single(plan, info, b_idx, fpos, inner)
                && let Op::Icmp(cc, a, b) = func.insts[iinst.index()].op
                && let Some(band) = try_cmp_preserved(func, plan, info, (b_idx, ipos, pos, Some(fpos)), (cc, a, b, 1)) =>
        {
            plan.elided[flag_val.index()] = true;
            plan.elided[inner.index()] = true;
            if let Some(m) = band {
                plan.elided[m.index()] = true;
            }
            true
        }
        _ => false,
    }
}

pub(super) fn try_fuse_icmp_select(
    func: &Function,
    plan: &mut Plan<Reg>,
    info @ (uses, _): (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos, res): (usize, usize, Value),
    (a, b): (Value, Value),
) {
    let cmp_uses = uses[res.index()];
    if !(1..=2).contains(&cmp_uses) {
        return;
    }
    let Op::Icmp(cc, _, _) = func.insts[func.blocks[b_idx].insts[pos].index()].op else {
        return;
    };
    let (mut found, mut last_sel) = (0_u32, pos);
    for (s_pos, &ni) in func.blocks[b_idx].insts.iter().enumerate().skip(pos + 1) {
        if matches!(func.insts[ni.index()].op, Op::Select(c, x, y) if c == res && x != res && y != res) {
            found += 1;
            last_sel = s_pos;
            if found == cmp_uses {
                break;
            }
        }
    }
    if found == cmp_uses
        && let Some(band) = try_cmp_preserved(func, plan, info, (b_idx, pos, last_sel, None), (cc, a, b, cmp_uses))
    {
        plan.elided[res.index()] = true;
        if let Some(m) = band {
            plan.elided[m.index()] = true;
        }
    }
}

pub(super) fn fuse_icmp_selects(func: &Function, plan: &mut Plan<Reg>, info: (&[u32], &[Option<(usize, usize, Inst)>])) {
    for (b_idx, block) in func.blocks.iter().enumerate() {
        for (pos, &inst) in block.insts.iter().enumerate() {
            let data = &func.insts[inst.index()];
            if let Some(res) = data.result
                && !plan.elided[res.index()]
                && let Op::Icmp(_, a, b) = data.op
                && func.ty(a).is_small_int()
            {
                try_fuse_icmp_select(func, plan, info, (b_idx, pos, res), (a, b));
            }
        }
    }
}

fn fits_consumer(func: &Function, plan: &Plan<Reg>, tail: &[Inst], res: Value) -> bool {
    for &ni in tail {
        let nd = &func.insts[ni.index()];
        if !matches!(nd.op, Op::Call { .. }) && nd.result.is_some_and(|r| plan.elided[r.index()]) {
            continue;
        }
        let is_use = match &nd.op {
            Op::Select(c, x, y) => *c == res && *x != res && *y != res,
            Op::Brif(c, _, ta, _, ea) => *c == res && !ta.contains(&res) && !ea.contains(&res),
            Op::Trapz(c) => *c == res,
            _ => false,
        };
        if is_use {
            return true;
        }
        if !matches!(
            nd.op,
            Op::Const(_) | Op::StackAddr(..) | Op::FramePointer | Op::Load(..) | Op::Store(..) | Op::StackLoad(..) | Op::StackStore(..)
        ) {
            return false;
        }
    }
    false
}

pub(super) fn fuse_fits(func: &Function, plan: &mut Plan<Reg>, uses: &[u32]) {
    for block in &func.blocks {
        for (pos, &inst) in block.insts.iter().enumerate() {
            let data = &func.insts[inst.index()];
            let Some(res) = data.result else { continue };
            let Op::Fits(_, _, a, _) = data.op else { continue };
            if uses[res.index()] == 1
                && !plan.elided[res.index()]
                && !plan.fused[res.index()]
                && matches!(plan.location(res), Location::Reg(_) | Location::Slot)
                && func.ty(a).is_small_int()
                && fits_consumer(func, plan, &block.insts[pos + 1..], res)
            {
                plan.fused[res.index()] = true;
            }
        }
    }
}

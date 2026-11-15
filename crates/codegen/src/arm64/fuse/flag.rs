use super::super::asm::Reg;
use super::{ext_parts, local_single, preserved};
use crate::alloc::{Location, Plan};
use crate::ir::{Convert, Function, Inst, Op, Type, Value};

fn cmp_operand_preserved(
    func: &Function,
    plan: &Plan<Reg>,
    pos_of: &[Option<(usize, usize, Inst)>],
    (b_idx, start, pos, skip): (usize, usize, usize, Option<usize>),
    (cmp_ty, v): (Type, Value),
) -> bool {
    let actual = if plan.elided[v.index()] {
        let Some((_, _, inner)) = ext_parts(func, plan, cmp_ty, v, |val| pos_of[val.index()].map(|(_, _, i)| i)) else {
            return false;
        };
        inner
    } else {
        v
    };
    preserved(func, plan, b_idx, start, pos, skip, actual)
}

pub(super) fn try_fuse_flag_operand(
    func: &Function,
    plan: &mut Plan<Reg>,
    info @ (_, pos_of): (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos): (usize, usize),
    flag_val: Value,
) -> bool {
    let Some((fpos, finst)) = local_single(plan, info, b_idx, pos, flag_val) else {
        return false;
    };
    let cmp_ok = |start, skip, a, b| {
        func.ty(a).is_small_int()
            && cmp_operand_preserved(func, plan, pos_of, (b_idx, start, pos, skip), (func.ty(a), a))
            && cmp_operand_preserved(func, plan, pos_of, (b_idx, start, pos, skip), (func.ty(a), b))
    };
    match func.insts[finst.index()].op {
        Op::Icmp(_, a, b) if cmp_ok(fpos, None, a, b) => {
            plan.elided[flag_val.index()] = true;
            true
        }
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

pub(super) fn try_fuse_icmp_select(
    func: &Function,
    plan: &mut Plan<Reg>,
    (uses, pos_of): (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos, res): (usize, usize, Value),
    (a, b): (Value, Value),
) {
    if !(1..=2).contains(&uses[res.index()]) {
        return;
    }
    let (mut found, mut last_sel) = (0_u32, pos);
    for (s_pos, &ni) in func.blocks[b_idx].insts.iter().enumerate().skip(pos + 1) {
        if matches!(func.insts[ni.index()].op, Op::Select(c, x, y) if c == res && x != res && y != res) {
            found += 1;
            last_sel = s_pos;
            if found == uses[res.index()] {
                break;
            }
        }
    }
    let cmp_ty = func.ty(a);
    if found == uses[res.index()]
        && cmp_operand_preserved(func, plan, pos_of, (b_idx, pos, last_sel, None), (cmp_ty, a))
        && cmp_operand_preserved(func, plan, pos_of, (b_idx, pos, last_sel, None), (cmp_ty, b))
    {
        plan.elided[res.index()] = true;
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
        let is_use = match nd.op {
            Op::Select(c, x, y) => c == res && x != res && y != res,
            Op::Brif(c, ..) | Op::Trapz(c) => c == res,
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

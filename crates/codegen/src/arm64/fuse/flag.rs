mod select;

pub(super) use self::select::{fuse_icmp_selects, try_fuse_icmp_select};
use super::super::Location;
use super::super::asm::Reg;
use super::bits::can_elide_tst_band;
use super::{
    alu_shift, cmn_parts, cmn_sum_parts, const_of, def_entry, def_inst, elided_or_deferred,
    ext_parts, local_single, preserved, shift_parts, unwrap_elided_tst, zero_cmp_band,
};
use crate::alloc::Plan;
use crate::ir::{Binary, Convert, Function, Inst, IntCC, Op, Type, Unary, Value};

fn peel_uext(
    func: &Function,
    plan: &Plan<Reg>,
    pos_of: &[Option<(usize, usize, Inst)>],
    val: Value,
) -> Value {
    if plan.elided[val.index()]
        && let Some((_, _, inst)) = def_entry(pos_of, val)
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
    let Some((_, _, inst)) = def_entry(pos_of, v) else {
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
    preserved(func, plan, b_idx, start, pos, skip, root)
        && preserved(func, plan, b_idx, start, pos, skip, other_root)
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
    let def = |val| def_inst(pos_of, val);
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

fn tst_single_insn(
    func: &Function,
    plan: &Plan<Reg>,
    pos_of: &[Option<(usize, usize, Inst)>],
    (x, y): (Value, Value),
) -> bool {
    let def = |v| def_inst(pos_of, v);
    let ty = func.ty(x);
    let ty_mask = ty.unsigned_max();
    for (raw, mask) in [(x, y), (y, x)] {
        if let Some(imm) = const_of(func, plan, mask, &def) {
            return unwrap_elided_tst(func, plan, raw, (imm as u64) & ty_mask, &def).is_some_and(
                |(_, eff)| super::super::int::encode_bitmask_imm(eff, ty == Type::I64).is_some(),
            );
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

pub(super) fn try_cmp_preserved(
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
    let def = |val| def_inst(pos_of, val);
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
    if cmp_shift_preserved(func, plan, pos_of, span, (cmp_ty, b, a))
        || cmp_shift_preserved(func, plan, pos_of, span, (cmp_ty, a, b))
    {
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
            let Some(band) =
                try_cmp_preserved(func, plan, info, (b_idx, fpos, pos, None), (cc, a, b, 1))
            else {
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
                && let Some(band) = try_cmp_preserved(
                    func,
                    plan,
                    info,
                    (b_idx, ipos, pos, Some(fpos)),
                    (cc, a, b, 1),
                ) =>
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

fn fits_consumer(
    func: &Function,
    plan: &Plan<Reg>,
    def: &dyn Fn(Value) -> Option<Inst>,
    tail: &[Inst],
    res: Value,
) -> bool {
    for &ni in tail {
        let nd = &func.insts[ni.index()];
        if elided_or_deferred(plan, nd) {
            continue;
        }
        if consumes_flag_only(&nd.op, res) {
            return true;
        }
        if !preserves_flags(func, plan, def, &nd.op, nd.result, false) {
            return false;
        }
    }
    false
}

pub(super) fn fuse_fits(
    func: &Function,
    plan: &mut Plan<Reg>,
    (uses, pos_of): (&[u32], &[Option<(usize, usize, Inst)>]),
) {
    let def = |v| def_inst(pos_of, v);
    for block in &func.blocks {
        for (pos, &inst) in block.insts.iter().enumerate() {
            let data = &func.insts[inst.index()];
            let Some(res) = data.result else { continue };
            let Op::Fits(_, _, a, _) = data.op else {
                continue;
            };
            if uses[res.index()] == 1
                && !plan.elided[res.index()]
                && !plan.fused[res.index()]
                && matches!(plan.location(res), Location::Reg(_) | Location::Slot)
                && func.ty(a).is_small_int()
                && fits_consumer(func, plan, &def, &block.insts[pos + 1..], res)
            {
                plan.fused[res.index()] = true;
            }
        }
    }
}

pub(super) fn consumes_flag_only(op: &Op, res: Value) -> bool {
    match op {
        Op::Select(c, x, y) => *c == res && *x != res && *y != res,
        Op::Brif(c, _, ta, _, ea) => *c == res && !ta.contains(&res) && !ea.contains(&res),
        Op::Trapz(c) => *c == res,
        _ => false,
    }
}

pub(in super::super) fn cinc_operand(
    func: &Function,
    plan: &Plan<Reg>,
    val: Value,
    def: &dyn Fn(Value) -> Option<Inst>,
) -> Option<(IntCC, Value, Value)> {
    if !plan.elided[val.index()] {
        return None;
    }
    let cmp_val = match func.insts[def(val)?.index()].op {
        Op::Convert(Convert::Uextend, inner) if plan.elided[inner.index()] => inner,
        _ => val,
    };
    match func.insts[def(cmp_val)?.index()].op {
        Op::Icmp(cc, a, b) => Some((cc, a, b)),
        _ => None,
    }
}

fn is_elided_select(
    func: &Function,
    plan: &Plan<Reg>,
    val: Value,
    def: &dyn Fn(Value) -> Option<Inst>,
) -> bool {
    plan.elided[val.index()]
        && def(val).is_some_and(|inst| matches!(func.insts[inst.index()].op, Op::Select(..)))
}

pub(in super::super) fn preserves_flags(
    func: &Function,
    plan: &Plan<Reg>,
    def: &dyn Fn(Value) -> Option<Inst>,
    op: &Op,
    result: Option<Value>,
    allow_i8_bitwise: bool,
) -> bool {
    matches!(
        *op,
        Op::Load(..)
            | Op::Store(..)
            | Op::StackAddr(..)
            | Op::StackLoad(..)
            | Op::StackStore(..)
            | Op::Const(_)
            | Op::SlotAddr(_)
            | Op::Symbol(_)
            | Op::Runtime(_)
            | Op::FramePointer
    ) || match *op {
        Op::Binary(b, x, y) => {
            !matches!(func.ty(x), Type::I128)
                && !func.ty(x).is_vector()
                && match b {
                    Binary::Iadd | Binary::Isub => {
                        cinc_operand(func, plan, x, def).is_none()
                            && cinc_operand(func, plan, y, def).is_none()
                            && !is_elided_select(func, plan, x, def)
                            && !is_elided_select(func, plan, y, def)
                    }
                    Binary::Imul
                    | Binary::Bxor
                    | Binary::Ishl
                    | Binary::Ushr
                    | Binary::Sshr
                    | Binary::Rotl
                    | Binary::Rotr => true,
                    Binary::Band | Binary::Bor => {
                        (allow_i8_bitwise || func.ty(x) != Type::I8)
                            && !plan.elided[x.index()]
                            && !plan.elided[y.index()]
                            && result.is_none_or(|r| !plan.fused[r.index()])
                    }
                    _ => false,
                }
        }
        Op::Unary(u, x) => {
            !matches!(func.ty(x), Type::I128)
                && !func.ty(x).is_vector()
                && match u {
                    Unary::Ineg => cinc_operand(func, plan, x, def).is_none(),
                    Unary::Bnot | Unary::Clz | Unary::Ctz | Unary::Popcnt | Unary::Bswap => true,
                    _ => false,
                }
        }
        Op::Convert(c, x) => {
            !matches!(func.ty(x), Type::I128)
                && !func.ty(x).is_vector()
                && result.is_some_and(|r| !func.ty(r).is_vector() && !func.ty(r).is_float())
                && matches!(
                    c,
                    Convert::Uextend | Convert::Sextend | Convert::Ireduce | Convert::Bitcast
                )
        }
        _ => false,
    }
}

fn flag_consumer(
    func: &Function,
    plan: &Plan<Reg>,
    def: &dyn Fn(Value) -> Option<Inst>,
    tail: &[Inst],
    res: Value,
    allow_clobber_branch: bool,
) -> Option<(usize, bool, bool)> {
    let mut flags_ok = true;
    let capped = func.size() > 2000;
    for (off, &ni) in tail.iter().enumerate() {
        let nd = &func.insts[ni.index()];
        if elided_or_deferred(plan, nd) {
            continue;
        }
        if consumes_flag_only(&nd.op, res) {
            let is_branch = !matches!(nd.op, Op::Select(..));
            return (flags_ok || (allow_clobber_branch && is_branch))
                .then_some((off, is_branch, flags_ok));
        }
        if !preserves_flags(func, plan, def, &nd.op, nd.result, false) {
            if !allow_clobber_branch {
                return None;
            }
            flags_ok = false;
        }
        if !flags_ok && capped && off >= 64 {
            return None;
        }
    }
    None
}

fn multi_flag_consumers(
    func: &Function,
    plan: &Plan<Reg>,
    def: &dyn Fn(Value) -> Option<Inst>,
    tail: &[Inst],
    res: Value,
    total_uses: u32,
) -> bool {
    let mut found = 0_u32;
    for &ni in tail {
        let nd = &func.insts[ni.index()];
        if elided_or_deferred(plan, nd) {
            continue;
        }
        if consumes_flag_only(&nd.op, res) {
            found += 1;
            if found == total_uses {
                return true;
            }
            continue;
        }
        if !preserves_flags(func, plan, def, &nd.op, nd.result, false) {
            return false;
        }
    }
    false
}

pub(super) fn icmp_consumer(
    func: &Function,
    plan: &Plan<Reg>,
    def: &dyn Fn(Value) -> Option<Inst>,
    tail: &[Inst],
    res: Value,
) -> bool {
    flag_consumer(func, plan, def, tail, res, false).is_some()
}

pub(in super::super) fn zero_cmp_operand(
    plan: &Plan<Reg>,
    ty: Type,
    a: Value,
    b: Value,
) -> Option<(Value, bool)> {
    if !ty.is_small_int() {
        return None;
    }
    let is_imm = |v: Value, k: i64| {
        plan.const_of(v)
            .is_some_and(|imm| crate::fold::extend(imm, ty, false) == k)
    };
    let (val, one) = if is_imm(b, 0) {
        (a, false)
    } else if is_imm(a, 0) {
        (b, false)
    } else if is_imm(b, 1) && plan.clean_bits[a.index()] <= 1 {
        (a, true)
    } else if is_imm(a, 1) && plan.clean_bits[b.index()] <= 1 {
        (b, true)
    } else {
        return None;
    };
    (!plan.elided[val.index()] && plan.const_of(val).is_none()).then_some((val, one))
}

fn can_elide_cbz(
    func: &Function,
    plan: &Plan<Reg>,
    (uses, pos_of): (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos, c_pos): (usize, usize, usize),
    val: Value,
    one: bool,
) -> bool {
    if !preserved(func, plan, b_idx, pos, c_pos, None, val) {
        return false;
    }
    if !one
        && uses[val.index()] == 1
        && let Some((vb, vpos, vinst)) = def_entry(pos_of, val)
        && vb == b_idx
        && let Op::Binary(Binary::Band, bx, by) = func.insts[vinst.index()].op
        && (super::bits::single_bit_raw(func, plan, bx, by).is_some()
            || can_elide_tst_band(
                func,
                plan,
                pos_of,
                (b_idx, vpos, c_pos, Some(pos)),
                (bx, by),
            ))
    {
        return false;
    }
    true
}

pub(super) fn fuse_icmp_flags(
    func: &Function,
    plan: &mut Plan<Reg>,
    (uses, pos_of): (&[u32], &[Option<(usize, usize, Inst)>]),
) {
    let def = |v| def_inst(pos_of, v);
    for (b_idx, block) in func.blocks.iter().enumerate() {
        for (pos, &inst) in block.insts.iter().enumerate() {
            let data = &func.insts[inst.index()];
            let Some(res) = data.result else { continue };
            let Op::Icmp(cc, a, b) = data.op else {
                continue;
            };
            if plan.elided[res.index()] || plan.is_static(res) || func.ty(a).is_vector() {
                continue;
            }
            let use_count = uses[res.index()];
            if use_count > 1 {
                if !plan.fused[res.index()]
                    && multi_flag_consumers(
                        func,
                        plan,
                        &def,
                        &block.insts[pos + 1..],
                        res,
                        use_count,
                    )
                {
                    plan.fused[res.index()] = true;
                }
                continue;
            }
            if use_count != 1 {
                continue;
            }
            let was_fused = plan.fused[res.index()];
            let zero_val = matches!(cc, IntCC::Equal | IntCC::NotEqual)
                .then(|| zero_cmp_operand(plan, func.ty(a), a, b))
                .flatten();
            if was_fused && zero_val.is_none() {
                continue;
            }
            let Some((off, is_branch, flags_ok)) = flag_consumer(
                func,
                plan,
                &def,
                &block.insts[pos + 1..],
                res,
                zero_val.is_some(),
            ) else {
                continue;
            };
            if is_branch
                && let Some((val, one)) = zero_val
                && can_elide_cbz(
                    func,
                    plan,
                    (uses, pos_of),
                    (b_idx, pos, pos + 1 + off),
                    val,
                    one,
                )
            {
                plan.fused[res.index()] = false;
                plan.elided[res.index()] = true;
            } else if !was_fused && flags_ok {
                plan.fused[res.index()] = true;
            }
        }
    }
}

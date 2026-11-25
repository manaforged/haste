use super::is_zero_const;
use crate::alloc::Plan;
use crate::ir::{Binary, Convert, Function, Inst, IntCC, Op, Type, Value};
use crate::x64::asm::Reg;

fn contiguous_ones(v: u64) -> Option<u8> {
    if v == 0 {
        return None;
    }
    let len = v.trailing_ones();
    (len < 64 && v == (1_u64 << len) - 1).then_some(len as u8)
}

pub(in crate::x64) fn shld_parts(
    func: &Function,
    plan: &Plan<Reg>,
    ty: Type,
    (x, y): (Value, Value),
    child_ok: impl Fn(Value) -> bool,
    def: impl Fn(Value) -> Option<Inst>,
) -> Option<(Value, Value, Value, u8, u8)> {
    if !matches!(ty, Type::I32 | Type::I64) || x == y || !child_ok(x) || !child_ok(y) {
        return None;
    }
    let bits = i64::from(ty.bits());
    for (a, b) in [(x, y), (y, x)] {
        let (ai, bi) = (def(a)?, def(b)?);
        if let (Op::Binary(Binary::Ishl, hi, cl), Op::Binary(Binary::Ushr, lo, cr)) =
            (&func.insts[ai.index()].op, &func.insts[bi.index()].op)
            && let (Some(l), Some(r)) = (plan.const_of(*cl), plan.const_of(*cr))
            && (1..bits).contains(&l)
            && (1..bits).contains(&r)
            && l + r == bits
        {
            return Some((*hi, *lo, *cl, l as u8, r as u8));
        }
    }
    None
}

fn bextr_band(
    func: &Function,
    plan: &Plan<Reg>,
    ty: Type,
    (x, y): (Value, Value),
    child_ok: impl Fn(Value) -> bool,
    def: impl Fn(Value) -> Option<Inst>,
) -> Option<(Option<Value>, Value, u8, u8)> {
    let bits = i64::from(ty.bits());
    for (a, b) in [(x, y), (y, x)] {
        let Some(mask_raw) = plan.const_of(b) else {
            continue;
        };
        let mask = crate::fold::extend(mask_raw, ty, false) as u64;
        if child_ok(a)
            && let Some(ai) = def(a)
            && let Op::Binary(Binary::Ushr, src, cnt) = func.insts[ai.index()].op
            && let Some(start) = plan.const_of(cnt)
            && (1..bits).contains(&start)
            && let Some(len) = contiguous_ones(mask)
            && i64::from(len) < bits - start
        {
            return Some((Some(a), src, start as u8, len));
        }
        if ty == Type::I64
            && i32::try_from(mask_raw).is_err()
            && mask != 0xFFFF_FFFF
            && let Some(len) = contiguous_ones(mask)
            && len < 64
        {
            return Some((None, a, 0, len));
        }
    }
    None
}

fn bextr_ushr(
    func: &Function,
    plan: &Plan<Reg>,
    ty: Type,
    (x, y): (Value, Value),
    child_ok: impl Fn(Value) -> bool,
    def: impl Fn(Value) -> Option<Inst>,
) -> Option<(Option<Value>, Value, u8, u8)> {
    let bits = i64::from(ty.bits());
    let start = plan.const_of(y)?;
    if !(1..bits).contains(&start) || !child_ok(x) {
        return None;
    }
    let xi = def(x)?;
    let Op::Binary(Binary::Band, a, b) = func.insts[xi.index()].op else {
        return None;
    };
    for (src, mask_v) in [(a, b), (b, a)] {
        let Some(mask_raw) = plan.const_of(mask_v) else {
            continue;
        };
        let mask = (crate::fold::extend(mask_raw, ty, false) as u64) >> (start as u32);
        if let Some(len) = contiguous_ones(mask)
            && i64::from(len) < bits - start
        {
            return Some((Some(x), src, start as u8, len));
        }
    }
    None
}

pub(in crate::x64) fn bextr_parts(
    func: &Function,
    plan: &Plan<Reg>,
    ty: Type,
    op: Binary,
    (x, y): (Value, Value),
    child_ok: impl Fn(Value) -> bool,
    def: impl Fn(Value) -> Option<Inst>,
) -> Option<(Option<Value>, Value, u8, u8)> {
    if !matches!(ty, Type::I32 | Type::I64) {
        return None;
    }
    match op {
        Binary::Band => bextr_band(func, plan, ty, (x, y), child_ok, def),
        Binary::Ushr => bextr_ushr(func, plan, ty, (x, y), child_ok, def),
        _ => None,
    }
}

pub(in crate::x64) fn adc_plan(
    func: &Function,
    plan: &Plan<Reg>,
    ty: Type,
    op: Binary,
    (x, y): (Value, Value),
    child_ok: impl Fn(Value) -> bool,
    def: impl Fn(Value) -> Option<Inst>,
) -> Option<(Value, Option<Value>, Value, Value, Option<Value>, bool)> {
    let (cand, inner, cc, a, b) = adc_parts(func, plan, ty, op, (x, y), child_ok, def)?;
    let (invert, left, right) = adc_sides(func, plan, cc, a, b)?;
    Some((
        cand,
        inner,
        left,
        if cand == y { x } else { y },
        right,
        invert,
    ))
}

/// Whether the carry `a cc b` enters an `adc` or `sbb` inverted, and the compare it reads as `left <u right`, or as `left != 0` with no `right`.
pub(in crate::x64::location) fn adc_sides(
    func: &Function,
    plan: &Plan<Reg>,
    cc: IntCC,
    a: Value,
    b: Value,
) -> Option<(bool, Value, Option<Value>)> {
    Some(match cc {
        IntCC::UnsignedLessThan => (false, a, Some(b)),
        IntCC::UnsignedGreaterThan if is_zero_const(func, plan, b) => (true, a, None),
        IntCC::UnsignedGreaterThan => (false, b, Some(a)),
        IntCC::UnsignedGreaterThanOrEqual => (true, a, Some(b)),
        IntCC::UnsignedLessThanOrEqual => (true, b, Some(a)),
        IntCC::Equal => (
            false,
            if is_zero_const(func, plan, b) { a } else { b },
            None,
        ),
        IntCC::NotEqual => (true, if is_zero_const(func, plan, b) { a } else { b }, None),
        _ => return None,
    })
}

/// The operands of `other` when it is the add or subtract of the same kind as `op` that an `adc` or `sbb` adding an uninverted carry can compute in the same instruction.
pub(in crate::x64) fn adc_fold(
    func: &Function,
    op: Binary,
    invert: bool,
    other: Value,
    def: impl Fn(Value) -> Option<Inst>,
) -> Option<(Value, Value)> {
    match func.insts[def(other)?.index()].op {
        Op::Binary(inner, p, q) if !invert && inner == op => Some((p, q)),
        _ => None,
    }
}

pub(in crate::x64) fn unwrap_shift_mask(
    func: &Function,
    plan: &Plan<Reg>,
    defs: &[Option<Inst>],
    y: Value,
    shift_res: Value,
    bits: u32,
) -> Value {
    if !matches!(bits, 32 | 64) {
        return y;
    }
    let Some(y_inst) = defs[y.index()] else {
        return y;
    };
    let Op::Binary(Binary::Band, a, b) = func.insts[y_inst.index()].op else {
        return y;
    };
    let mask = u64::from(bits - 1);
    let Some(raw) = [(a, b), (b, a)].into_iter().find_map(|(raw, m)| {
        let imm = plan.const_or_def(func, |x| defs[x.index()], m)?;
        ((imm as u64 & mask) == mask).then_some(raw)
    }) else {
        return y;
    };
    if plan.elided[y.index()] {
        return raw;
    }
    let Some(r_raw) = plan.reg_of(raw) else {
        return y;
    };
    if plan.elided[raw.index()] || plan.reg_of(y) == Some(r_raw) {
        return y;
    }
    let Some(shift_inst) = defs[shift_res.index()] else {
        return y;
    };
    for block in &func.blocks {
        let Some(ypos) = block.insts.iter().position(|&i| i == y_inst) else {
            continue;
        };
        let Some(spos) = block.insts.iter().position(|&i| i == shift_inst) else {
            return y;
        };
        if ypos >= spos {
            return y;
        }
        let ok = block.insts[ypos + 1..spos].iter().all(|&inst| {
            let data = &func.insts[inst.index()];
            if data
                .result
                .is_some_and(|r| plan.elided[r.index()] || plan.deferred[r.index()])
            {
                return true;
            }
            !matches!(data.op, Op::Call { .. } | Op::TryCall { .. })
                && !data
                    .result
                    .and_then(|r| plan.pairs[r.index()])
                    .is_some_and(|p| plan.reg_of(p) == Some(r_raw))
                && !crate::live::results(data).any(|r| plan.reg_of(r) == Some(r_raw))
        });
        return if ok { raw } else { y };
    }
    y
}

fn adc_cc_ok(func: &Function, plan: &Plan<Reg>, cc: IntCC, a: Value, b: Value) -> bool {
    func.ty(a).is_small_int()
        && (cc.is_unsigned()
            || (matches!(cc, IntCC::Equal | IntCC::NotEqual)
                && (is_zero_const(func, plan, a) || is_zero_const(func, plan, b))))
}

pub(in crate::x64::location) fn adc_parts(
    func: &Function,
    plan: &Plan<Reg>,
    ty: Type,
    op: Binary,
    (x, y): (Value, Value),
    child_ok: impl Fn(Value) -> bool,
    def: impl Fn(Value) -> Option<Inst>,
) -> Option<(Value, Option<Value>, IntCC, Value, Value)> {
    if !matches!(op, Binary::Iadd | Binary::Isub) || !matches!(ty, Type::I32 | Type::I64) {
        return None;
    }
    let both = [(y, x), (x, y)];
    let one = [(y, x)];
    let pairs = if op == Binary::Iadd {
        &both[..]
    } else {
        &one[..]
    };
    for &(cand, _) in pairs {
        if !child_ok(cand) {
            continue;
        }
        let Some(ci) = def(cand) else { continue };
        match func.insts[ci.index()].op {
            Op::Icmp(cc, a, b) if adc_cc_ok(func, plan, cc, a, b) => {
                return Some((cand, None, cc, a, b));
            }
            Op::Convert(Convert::Uextend, inner) if child_ok(inner) => {
                if let Some(ii) = def(inner)
                    && let Op::Icmp(cc, a, b) = func.insts[ii.index()].op
                    && adc_cc_ok(func, plan, cc, a, b)
                {
                    return Some((cand, Some(inner), cc, a, b));
                }
            }
            _ => {}
        }
    }
    None
}

/// The signedness and factors of the 64-bit product whose high and low halves are `x` and `y` in either order.
pub(in crate::x64) fn product_halves(
    func: &Function,
    def: &dyn Fn(Value) -> Option<Inst>,
    (x, y): (Value, Value),
) -> Option<(bool, Value, Value)> {
    let op = |v: Value| def(v).map(|inst| &func.insts[inst.index()].op);
    let ((
        Some(&Op::Binary(high @ (Binary::Umulhi | Binary::Smulhi), a, b)),
        Some(&Op::Binary(Binary::Imul, c, d)),
    )
    | (
        Some(&Op::Binary(Binary::Imul, c, d)),
        Some(&Op::Binary(high @ (Binary::Umulhi | Binary::Smulhi), a, b)),
    )) = (op(x), op(y))
    else {
        return None;
    };
    (func.ty(x) == Type::I64 && func.ty(y) == Type::I64 && ((a, b) == (c, d) || (a, b) == (d, c)))
        .then_some((high == Binary::Smulhi, a, b))
}

use crate::fold::{extend, holds, known_bits, non_null_at};
use crate::ir::{Binary, Convert, Function, Inst, IntCC, Op, Type, Unary, Value};

use super::{RewriteCx, const_u128, op_of};

mod addr;
mod fuse;
#[path = "ord.rs"]
mod ord;
pub(super) use addr::{ADDR_SPAN, before, null_test, offset_bound};
pub(crate) use addr::{is_non_null_addr, non_null_cmp};
pub(super) use fuse::fuse_cmp_ops;
pub(super) use ord::three_way_non_zero;

use addr::MAX_DEPTH;

#[derive(Clone, Copy)]
enum Step {
    Conv(Convert, Type, Type),
    Mask(Type, u64),
}

#[derive(Clone, Copy)]
struct Cmp {
    cc: IntCC,
    ty: Type,
    k: i64,
}

fn apply_step(c: i64, step: Step) -> i64 {
    match step {
        Step::Conv(Convert::Uextend, src, _) => extend(c, src, false),
        Step::Conv(Convert::Sextend, src, _) => extend(c, src, true),
        Step::Conv(Convert::Ireduce, _, dst) => extend(c, dst, false),
        Step::Conv(_, _, _) => c,
        Step::Mask(ty, mask) => extend(((c as u64) & mask) as i64, ty, false),
    }
}

fn apply_steps(mut c: i64, steps: &[Step]) -> i64 {
    for &step in steps.iter().rev() {
        c = apply_step(c, step);
    }
    c
}

fn step_bits((z, o): (u64, u64), step: Step) -> (u64, u64) {
    match step {
        Step::Conv(Convert::Uextend, src, dst) => {
            let (sw, dw) = (src.unsigned_max(), dst.unsigned_max());
            ((z | !sw) & dw, o & sw)
        }
        Step::Conv(Convert::Ireduce, _, dst) => {
            let dw = dst.unsigned_max();
            (z & dw, o & dw)
        }
        Step::Conv(Convert::Sextend, src, dst) => {
            let (sw, dw) = (src.unsigned_max(), dst.unsigned_max());
            let sign = 1_u64 << (src.bits() - 1);
            let ext = !sw & dw;
            let nz = if (z & sign) != 0 { z | ext } else { z & sw };
            let no = if (o & sign) != 0 { o | ext } else { o & sw };
            (nz & dw, no & dw)
        }
        Step::Conv(_, _, _) => (0, 0),
        Step::Mask(ty, mask) => {
            let w = ty.unsigned_max();
            ((z | !mask) & w, o & mask & w)
        }
    }
}

fn steps_preserve_non_null(src: Type, steps: &[Step]) -> bool {
    src.bits() >= 32
        && steps.iter().all(|s| match *s {
            Step::Conv(Convert::Uextend | Convert::Sextend, from, _) => from.bits() >= 32,
            Step::Mask(ty, mask) => mask >= ty.unsigned_max(),
            _ => false,
        })
}

fn peel_steps(cx: &RewriteCx, mut val: Value) -> (Value, Vec<Step>) {
    let mut steps = Vec::new();
    for _ in 0..MAX_DEPTH {
        let Some(inst) = cx.def(val) else { break };
        match cx.func.insts[inst.index()].op {
            Op::Convert(kind @ (Convert::Uextend | Convert::Sextend | Convert::Ireduce), inner)
                if cx.func.ty(inner).is_small_int() =>
            {
                steps.push(Step::Conv(kind, cx.func.ty(inner), cx.func.ty(val)));
                val = inner;
            }
            Op::Binary(Binary::Band, a, b) => {
                let ty = cx.func.ty(val);
                let masked = [(a, b), (b, a)]
                    .into_iter()
                    .find_map(|(v, m)| cx.uconst(m, ty).map(|mask| (v, mask)));
                let Some((inner, mask)) = masked else { break };
                steps.push(Step::Mask(ty, mask));
                val = inner;
            }
            _ => break,
        }
    }
    (val, steps)
}

fn leaf_holds(cx: &RewriteCx, v: Value, steps: &[Step], cmp: Cmp) -> Option<bool> {
    if let Some(imm) = cx.sconst(v) {
        return Some(holds(cmp.cc, cmp.ty, apply_steps(imm, steps), cmp.k));
    }
    let ty = cx.func.ty(v);
    if steps_preserve_non_null(ty, steps)
        && let Some(bit) = null_test(cmp.cc, cmp.ty, cmp.k)
        && non_null_at(cx.func, &|x| cx.def(x), &|x| cx.sconst(x), before(cx), v)
    {
        return Some(bit);
    }
    if matches!(cmp.cc, IntCC::Equal | IntCC::NotEqual) {
        let mut bits = known_bits(cx.func, &|x| cx.def(x), &|x| cx.sconst(x), v, 0);
        for &step in steps.iter().rev() {
            bits = step_bits(bits, step);
        }
        let u = (extend(cmp.k, cmp.ty, false) as u64) & cmp.ty.unsigned_max();
        if (u & bits.0) != 0 || (!u & bits.1) != 0 {
            return Some(cmp.cc == IntCC::NotEqual);
        }
    }
    None
}

fn arm_holds(
    cx: &RewriteCx,
    v: Value,
    steps: &mut Vec<Step>,
    cmp: Cmp,
    budget: &mut u8,
) -> Option<bool> {
    if *budget == 0 || steps.len() > 8 {
        return None;
    }
    *budget -= 1;
    if let Some(bit) = leaf_holds(cx, v, steps, cmp) {
        return Some(bit);
    }
    let inst = cx.def(v)?;
    match cx.func.insts[inst.index()].op {
        Op::Convert(kind @ (Convert::Uextend | Convert::Sextend | Convert::Ireduce), inner)
            if cx.func.ty(inner).is_small_int() =>
        {
            steps.push(Step::Conv(kind, cx.func.ty(inner), cx.func.ty(v)));
            let out = arm_holds(cx, inner, steps, cmp, budget);
            steps.pop();
            out
        }
        Op::Binary(Binary::Band, a, b) => {
            let ty = cx.func.ty(v);
            let (inner, mask) = [(a, b), (b, a)]
                .into_iter()
                .find_map(|(u, m)| cx.uconst(m, ty).map(|mask| (u, mask)))?;
            steps.push(Step::Mask(ty, mask));
            let out = arm_holds(cx, inner, steps, cmp, budget);
            steps.pop();
            out
        }
        Op::Select(cond, a, b) if cx.func.ty(cond) == Type::I8 => {
            let ha = arm_holds(cx, a, steps, cmp, budget)?;
            let hb = arm_holds(cx, b, steps, cmp, budget)?;
            (ha == hb).then_some(ha)
        }
        _ => None,
    }
}

fn cc_mask(cc: IntCC) -> u8 {
    match cc {
        IntCC::Equal => 0b010,
        IntCC::NotEqual => 0b101,
        IntCC::SignedLessThan | IntCC::UnsignedLessThan => 0b001,
        IntCC::SignedLessThanOrEqual | IntCC::UnsignedLessThanOrEqual => 0b011,
        IntCC::SignedGreaterThan | IntCC::UnsignedGreaterThan => 0b100,
        IntCC::SignedGreaterThanOrEqual | IntCC::UnsignedGreaterThanOrEqual => 0b110,
    }
}

fn merge_icmps(
    cx: &RewriteCx,
    is_and: bool,
    (v1, neg1): (Value, bool),
    (v2, neg2): (Value, bool),
) -> Option<Op> {
    let Op::Icmp(raw1, x1, y1) = cx.func.insts[cx.def(v1)?.index()].op else {
        return None;
    };
    let Op::Icmp(raw2, x2, y2) = cx.func.insts[cx.def(v2)?.index()].op else {
        return None;
    };
    let cc1 = if neg1 { raw1.inverse() } else { raw1 };
    let cc2 = if neg2 { raw2.inverse() } else { raw2 };
    let aligned2 = if (x2, y2) == (x1, y1) {
        cc2
    } else if (y2, x2) == (x1, y1) {
        cc2.swapped()
    } else {
        return None;
    };
    let cmp_cc = |c: IntCC| !matches!(c, IntCC::Equal | IntCC::NotEqual);
    if cmp_cc(cc1) && cmp_cc(aligned2) && cc1.is_signed() != aligned2.is_signed() {
        return None;
    }
    let signed = (cmp_cc(cc1) && cc1.is_signed()) || (cmp_cc(aligned2) && aligned2.is_signed());
    let mask = if is_and {
        cc_mask(cc1) & cc_mask(aligned2)
    } else {
        cc_mask(cc1) | cc_mask(aligned2)
    };
    let out = match mask {
        0b000 => return Some(Op::Const(0)),
        0b111 => return Some(Op::Const(1)),
        0b010 => IntCC::Equal,
        0b101 => IntCC::NotEqual,
        0b001 if signed => IntCC::SignedLessThan,
        0b001 => IntCC::UnsignedLessThan,
        0b011 if signed => IntCC::SignedLessThanOrEqual,
        0b011 => IntCC::UnsignedLessThanOrEqual,
        0b100 if signed => IntCC::SignedGreaterThan,
        0b100 => IntCC::UnsignedGreaterThan,
        0b110 if signed => IntCC::SignedGreaterThanOrEqual,
        0b110 => IntCC::UnsignedGreaterThanOrEqual,
        _ => return None,
    };
    Some(Op::Icmp(out, x1, y1))
}

fn emit_combo(
    cx: &mut RewriteCx,
    res: Option<Value>,
    is_and: bool,
    a: (Value, bool),
    b: (Value, bool),
) -> Option<Op> {
    if a.0 == b.0 {
        return match (is_and, a.1 == b.1) {
            (true, false) => Some(Op::Const(0)),
            (false, false) => Some(Op::Const(1)),
            (_, true) if !a.1 => {
                cx.alias(res, a.0);
                None
            }
            (_, true) => {
                let one = cx.emit(Type::I8, Op::Const(1));
                Some(Op::Binary(Binary::Bxor, a.0, one))
            }
        };
    }
    if let Some(op) = merge_icmps(cx, is_and, a, b) {
        return Some(op);
    }
    let mut lit = |(v, neg): (Value, bool)| {
        if neg {
            let one = cx.emit(Type::I8, Op::Const(1));
            cx.emit(Type::I8, Op::Binary(Binary::Bxor, v, one))
        } else {
            v
        }
    };
    let (lhs, rhs) = (lit(a), lit(b));
    Some(Op::Binary(
        if is_and { Binary::Band } else { Binary::Bor },
        lhs,
        rhs,
    ))
}

fn eval_mixed(
    cx: &mut RewriteCx,
    res: Option<Value>,
    (c1, outer_arm, c1_is_then): (Value, bool, bool),
    arm: Value,
    (steps, cmp): (&[Step], Cmp),
) -> Option<Option<Op>> {
    let (inner, extra) = peel_steps(cx, arm);
    let inst = cx.def(inner)?;
    let Op::Select(c2, a2, b2) = cx.func.insts[inst.index()].op else {
        return None;
    };
    if cx.func.ty(c2) != Type::I8 || !cx.is_unit_bool(c2) {
        return None;
    }
    let mut combined = Vec::with_capacity(steps.len() + extra.len());
    combined.extend_from_slice(steps);
    combined.extend_from_slice(&extra);
    let (mut b_a, mut b_b) = (16, 16);
    let ha = arm_holds(cx, a2, &mut combined, cmp, &mut b_a)?;
    let hb = arm_holds(cx, b2, &mut combined, cmp, &mut b_b)?;
    if ha == hb {
        return None;
    }
    let is_and = !outer_arm;
    let neg1 = c1_is_then ^ outer_arm;
    let neg2 = !ha;
    Some(emit_combo(cx, res, is_and, (c1, neg1), (c2, neg2)))
}

pub(super) fn emit_cond_truth(
    cx: &mut RewriteCx,
    res: Option<Value>,
    raw_cond: Value,
    truth: bool,
) -> Option<Option<Op>> {
    let cond = cx.unwrap_uextend(raw_cond);
    let cond_ty = cx.func.ty(cond);
    if cond_ty == Type::I8 && cx.is_unit_bool(cond) {
        if truth {
            cx.alias(res, cond);
            return Some(None);
        }
        let one = cx.emit(Type::I8, Op::Const(1));
        return Some(Some(Op::Binary(Binary::Bxor, cond, one)));
    }
    if !cond_ty.is_small_int() {
        return None;
    }
    let zero = cx.emit(cond_ty, Op::Const(0));
    Some(Some(Op::Icmp(
        if truth { IntCC::NotEqual } else { IntCC::Equal },
        cond,
        zero,
    )))
}

fn eval_one(
    cx: &mut RewriteCx,
    res: Option<Value>,
    cc: IntCC,
    val: Value,
    k: i64,
) -> Option<Option<Op>> {
    let cmp = Cmp {
        cc,
        ty: cx.func.ty(val),
        k,
    };
    let (core, mut steps) = peel_steps(cx, val);
    let inst = cx.def(core)?;
    let Op::Select(cond, a, b) = cx.func.insts[inst.index()].op else {
        return None;
    };
    let (mut ba, mut bb) = (16, 16);
    let ha = arm_holds(cx, a, &mut steps, cmp, &mut ba);
    let hb = arm_holds(cx, b, &mut steps, cmp, &mut bb);
    if let (Some(ta), Some(tb)) = (ha, hb)
        && ta == tb
    {
        return Some(Some(Op::Const(i64::from(ta))));
    }
    match (ha, hb) {
        (Some(true), Some(false)) => return emit_cond_truth(cx, res, cond, true),
        (Some(false), Some(true)) => return emit_cond_truth(cx, res, cond, false),
        _ => {}
    }
    if cx.func.ty(cond) != Type::I8 || !cx.is_unit_bool(cond) {
        return None;
    }
    match (ha, hb) {
        (Some(ta), None) => eval_mixed(cx, res, (cond, ta, true), b, (&steps, cmp)),
        (None, Some(tb)) => eval_mixed(cx, res, (cond, tb, false), a, (&steps, cmp)),
        _ => None,
    }
}

fn eval_unit_cmp(
    cx: &mut RewriteCx,
    res: Option<Value>,
    cc: IntCC,
    val: Value,
    k: i64,
) -> Option<Option<Op>> {
    let ty = cx.func.ty(val);
    let (zeros, ones) = cx.known(val);
    if (zeros | ones) == ty.unsigned_max() || (zeros | 1) != ty.unsigned_max() {
        return None;
    }
    let (h0, h1) = (holds(cc, ty, 0, k), holds(cc, ty, 1, k));
    if h0 == h1 {
        return Some(Some(Op::Const(i64::from(h0))));
    }
    let base = cx.unwrap_uextend(val);
    let bty = cx.func.ty(base);
    if !h0 {
        if bty == Type::I8 {
            cx.alias(res, base);
            return Some(None);
        }
        return Some(Some(Op::Convert(Convert::Ireduce, base)));
    }
    let low = if bty == Type::I8 {
        base
    } else {
        cx.emit(Type::I8, Op::Convert(Convert::Ireduce, base))
    };
    let one = cx.emit(Type::I8, Op::Const(1));
    Some(Some(Op::Binary(Binary::Bxor, low, one)))
}

fn unwrap_strict(
    func: &Function,
    def_of: &dyn Fn(Value) -> Option<Inst>,
    mut v: Value,
) -> Option<(Value, Value, bool, bool)> {
    for _ in 0..4 {
        match func.insts[def_of(v)?.index()].op {
            Op::Icmp(cc, a, b) if !func.ty(a).in_float_register() => {
                return match cc {
                    IntCC::SignedGreaterThan => Some((a, b, true, true)),
                    IntCC::SignedLessThan => Some((a, b, true, false)),
                    IntCC::UnsignedGreaterThan => Some((a, b, false, true)),
                    IntCC::UnsignedLessThan => Some((a, b, false, false)),
                    _ => None,
                };
            }
            Op::Convert(Convert::Uextend | Convert::Sextend | Convert::Ireduce, inner)
                if !func.ty(inner).in_float_register() =>
            {
                v = inner;
            }
            _ => return None,
        }
    }
    None
}

pub(super) fn complementary_pair(
    func: &Function,
    def_of: &dyn Fn(Value) -> Option<Inst>,
    x: Value,
    y: Value,
) -> Option<(Value, Value, bool)> {
    let (a, b, s1, g1) = unwrap_strict(func, def_of, x)?;
    let (na, nb, s2, g2) = unwrap_strict(func, def_of, y)?;
    if s1 != s2 || func.ty(a) != func.ty(na) {
        return None;
    }
    ((na == a && nb == b && g2 == !g1) || (na == b && nb == a && g2 == g1)).then_some((a, b, s1))
}

pub(super) fn complementary_eq_pair(
    func: &Function,
    def_of: &dyn Fn(Value) -> Option<Inst>,
    x: Value,
    y: Value,
) -> Option<(Value, Value)> {
    if let Some((a, b, _)) = complementary_pair(func, def_of, x, y) {
        return Some((a, b));
    }
    for (expr, z) in [(x, y), (y, x)] {
        if const_u128(func, def_of, func.ty(z), z) == Some(0)
            && let Some(&Op::Binary(Binary::Isub | Binary::Bxor, p, q)) = op_of(func, def_of, expr)
            && let Some((a, b, _)) = complementary_pair(func, def_of, p, q)
        {
            return Some((a, b));
        }
    }
    None
}

pub(super) fn eval_select(
    cx: &mut RewriteCx,
    res: Option<Value>,
    cc: IntCC,
    x: Value,
    y: Value,
) -> Option<Option<Op>> {
    if !res.is_some_and(|r| cx.func.ty(r) == Type::I8) {
        return None;
    }
    if matches!(cc, IntCC::Equal | IntCC::NotEqual)
        && let Some((a, b)) = complementary_eq_pair(cx.func, &|v| cx.def(v), x, y)
    {
        return Some(Some(Op::Icmp(cc, a, b)));
    }
    if !cx.func.ty(x).is_small_int() {
        return None;
    }
    if let Some(bit) = non_null_cmp(
        cx.func,
        &|v| cx.def(v),
        &|v| cx.sconst(v),
        before(cx),
        (cc, x, y),
    ) {
        return Some(Some(Op::Const(i64::from(bit))));
    }
    if let Some(op) = ord::eval_three_way(cx, cc, x, y)
        .or_else(|| ord::eval_complementary_cmp(cx, cc, x, y))
        .or_else(|| ord::eval_unary_cmp(cx, cc, x, y))
    {
        return Some(Some(op));
    }
    for (norm, val, c_val) in [(cc, x, y), (cc.swapped(), y, x)] {
        if let Some(k) = cx.sconst(c_val)
            && let Some(out) =
                eval_one(cx, res, norm, val, k).or_else(|| eval_unit_cmp(cx, res, norm, val, k))
        {
            return Some(out);
        }
    }
    None
}

/// Peels zero-preserving ops (`x * odd`, exact `x >> k` / `x << k`, `-x`) from a small-int value tested only against zero.
pub(super) fn strip_zero_test(cx: &RewriteCx, v: Value) -> Option<Value> {
    let ty = cx.func.ty(v);
    if !ty.is_small_int() {
        return None;
    }
    let (bits, w) = (ty.bits(), ty.unsigned_max());
    let mut cur = v;
    for _ in 0..4 {
        let next = match cx.op(cur)? {
            Op::Binary(Binary::Imul, a, b) => [(*a, *b), (*b, *a)]
                .into_iter()
                .find_map(|(x, k)| cx.uconst(k, ty).filter(|&c| (c & 1) == 1).map(|_| x)),
            Op::Binary(Binary::Ushr, a, sh) => cx
                .count(*sh, ty)
                .filter(|&k| k > 0 && k < bits)
                .and_then(|k| {
                    let lo = (1_u64 << k) - 1;
                    ((cx.known(*a).0 & lo) == lo).then_some(*a)
                }),
            Op::Binary(Binary::Ishl, a, sh) => cx
                .count(*sh, ty)
                .filter(|&k| k > 0 && k < bits)
                .and_then(|k| {
                    let hi = w & !((1_u64 << (bits - k)) - 1);
                    ((cx.known(*a).0 & hi) == hi).then_some(*a)
                }),
            Op::Unary(Unary::Ineg, a) => Some(*a),
            _ => None,
        };
        let Some(inner) = next else { break };
        cur = inner;
    }
    (cur != v).then_some(cur)
}

/// Simplifies an equality compare of `x + y` or `x - y` against `x`: folds constant `y` and rewrites variable `y` to `y == 0` / `y != 0`.
pub(super) fn offset_self(cx: &mut RewriteCx, cc: IntCC, x: Value, y: Value) -> Option<Op> {
    if !matches!(cc, IntCC::Equal | IntCC::NotEqual) {
        return None;
    }
    let ty = cx.func.ty(x);
    if !ty.is_small_int() {
        return None;
    }
    let step = [(x, y), (y, x)]
        .into_iter()
        .find_map(|(moved, base)| match *cx.op(moved)? {
            Op::Binary(Binary::Iadd, p, q) if p == base => Some(q),
            Op::Binary(Binary::Iadd, p, q) if q == base => Some(p),
            Op::Binary(Binary::Isub, p, q) if p == base => Some(q),
            _ => None,
        })?;
    if let Some(c) = cx.uconst(step, ty) {
        return Some(Op::Const(i64::from((c != 0) == (cc == IntCC::NotEqual))));
    }
    let zero = cx.emit(ty, Op::Const(0));
    Some(Op::Icmp(cc, step, zero))
}

/// Rewrites an unsigned compare of a single-use `x + c1` against `c2` into `x` against `c2 - c1` when `x + c1` cannot wrap.
pub(super) fn strip_const_add(cx: &mut RewriteCx, cc: IntCC, x: Value, y: Value) -> Option<Op> {
    if cc.is_signed() {
        return None;
    }
    let ty = cx.func.ty(x);
    for (norm, sum, c_v) in [(cc, x, y), (cc.swapped(), y, x)] {
        let Some(c2) = cx.uconst(c_v, ty) else {
            continue;
        };
        let Some(&Op::Binary(Binary::Iadd, p, q)) = cx.op(sum) else {
            continue;
        };
        let Some((base, c1)) = [(p, q), (q, p)]
            .into_iter()
            .find_map(|(b, k)| cx.uconst(k, ty).map(|c| (b, c)))
        else {
            continue;
        };
        if c1 == 0 || c2 < c1 || !cx.single_use(sum) {
            continue;
        }
        let hi = offset_bound(cx.func, &|v| cx.def(v), &|v| cx.sconst(v), base, (None, 0));
        if hi.checked_add(c1).is_none_or(|top| top > ty.unsigned_max()) {
            continue;
        }
        return Some(fuse::emit_cmp_const(cx, norm, base, (c2 - c1) as i64));
    }
    None
}

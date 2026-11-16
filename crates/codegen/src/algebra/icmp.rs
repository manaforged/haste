use crate::cse::flip_int_cc;
use crate::fold::{Before, extend, holds, known_bits, non_null_at};
use crate::ir::{Binary, Convert, Function, Inst, IntCC, Op, Type, Value};

use super::{RewriteCx, const_u128, op_of};

#[path = "ord.rs"]
mod ord;
pub(super) use ord::three_way_non_zero;

const MAX_DEPTH: u8 = 4;
pub(super) const ADDR_SPAN: i64 = 0x1000_0000;

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

fn stack_base(func: &Function, def_of: &dyn Fn(Value) -> Option<Inst>, mut v: Value) -> bool {
    for _ in 0..MAX_DEPTH {
        match def_of(v).map(|inst| &func.insts[inst.index()].op) {
            Some(Op::StackAddr(..) | Op::SlotAddr(_)) => return true,
            Some(&Op::Convert(Convert::Bitcast, a)) if func.ty(a) == func.ty(v) => v = a,
            _ => return false,
        }
    }
    false
}

pub(super) fn offset_bound(
    func: &Function,
    def_of: &dyn Fn(Value) -> Option<Inst>,
    known: &dyn Fn(Value) -> Option<i64>,
    v: Value,
    (bound, depth): (Option<(Value, u64)>, u8),
) -> u64 {
    let ty = func.ty(v);
    if !ty.is_small_int() { return u64::MAX; }
    let w = ty.unsigned_max();
    let mut hi = match known(v) {
        Some(c) => extend(c, ty, false) as u64,
        None => w & !known_bits(func, def_of, known, v, 0).0,
    };
    if let Some((bv, bmax)) = bound && bv == v { hi = hi.min(bmax); }
    if depth >= MAX_DEPTH || known(v).is_some() { return hi; }
    let next = (bound, depth + 1);
    let sub = match def_of(v).map(|inst| &func.insts[inst.index()].op) {
        Some(&Op::Convert(Convert::Uextend, a)) => offset_bound(func, def_of, known, a, next),
        Some(&Op::Binary(Binary::Band, a, b)) => offset_bound(func, def_of, known, a, next).min(offset_bound(func, def_of, known, b, next)),
        Some(&Op::Binary(Binary::Ishl, a, sh)) => known(sh)
            .filter(|&k| (0..i64::from(ty.bits())).contains(&k))
            .and_then(|k| offset_bound(func, def_of, known, a, next).checked_mul(1_u64 << (k as u32)))
            .filter(|&n| n <= w)
            .unwrap_or(u64::MAX),
        Some(&Op::Binary(Binary::Imul, a, b)) => [(a, b), (b, a)]
            .into_iter()
            .find_map(|(x, k)| {
                let c = known(k).map(|imm| extend(imm, ty, false) as u64)?;
                offset_bound(func, def_of, known, x, next).checked_mul(c).filter(|&n| n <= w)
            })
            .unwrap_or(u64::MAX),
        Some(&Op::Binary(Binary::Iadd, a, b)) => offset_bound(func, def_of, known, a, next)
            .checked_add(offset_bound(func, def_of, known, b, next))
            .filter(|&n| n <= w)
            .unwrap_or(u64::MAX),
        _ => u64::MAX,
    };
    hi.min(sub)
}

fn sym_base_disp(func: &Function, def_of: &dyn Fn(Value) -> Option<Inst>, known: &dyn Fn(Value) -> Option<i64>, v: Value, depth: u8) -> Option<u64> {
    if depth >= MAX_DEPTH { return None; }
    match def_of(v).map(|inst| &func.insts[inst.index()].op) {
        Some(&Op::Symbol(_)) => Some(0),
        Some(&Op::Convert(Convert::Bitcast, a)) if func.ty(a) == func.ty(v) => sym_base_disp(func, def_of, known, a, depth + 1),
        Some(&Op::Binary(Binary::Iadd, a, b)) => [(a, b), (b, a)].into_iter().find_map(|(base, off)| {
            sym_base_disp(func, def_of, known, base, depth + 1)?
                .checked_add(offset_bound(func, def_of, known, off, (None, depth + 1)))
                .filter(|&sum| sum <= ADDR_SPAN as u64)
        }),
        _ => None,
    }
}

/// Returns whether `v` denotes a guaranteed non-null stack, symbol, or slot address.
pub(crate) fn is_non_null_addr(func: &Function, def_of: &dyn Fn(Value) -> Option<Inst>, known: &dyn Fn(Value) -> Option<i64>, v: Value, depth: u8) -> bool {
    if depth >= MAX_DEPTH || func.ty(v).bits() < 32 { return false; }
    let Some(inst) = def_of(v) else { return false };
    let stack_disp = |base: Value, off: Value| stack_base(func, def_of, base) && known(off).is_some_and(|c| (-ADDR_SPAN..=ADDR_SPAN).contains(&c));
    match func.insts[inst.index()].op {
        Op::StackAddr(..) | Op::Symbol(_) | Op::SlotAddr(_) => true,
        Op::Binary(Binary::Iadd, a, b) => stack_disp(a, b) || stack_disp(b, a) || sym_base_disp(func, def_of, known, v, depth).is_some(),
        Op::Binary(Binary::Isub, a, b) => stack_disp(a, b),
        Op::Select(_, a, b) => is_non_null_addr(func, def_of, known, a, depth + 1) && is_non_null_addr(func, def_of, known, b, depth + 1),
        Op::Convert(Convert::Bitcast, a) if func.ty(a) == func.ty(v) => is_non_null_addr(func, def_of, known, a, depth + 1),
        _ => false,
    }
}

pub(super) fn null_test(cc: IntCC, ty: Type, k: i64) -> Option<bool> {
    match (cc, extend(k, ty, false) as u64) {
        (IntCC::Equal, 0) | (IntCC::UnsignedLessThanOrEqual, 0) | (IntCC::UnsignedLessThan, 1) => Some(false),
        (IntCC::NotEqual, 0) | (IntCC::UnsignedGreaterThan, 0) | (IntCC::UnsignedGreaterThanOrEqual, 1) => Some(true),
        _ => None,
    }
}

pub(super) fn before<'a>(cx: &'a RewriteCx) -> Before<'a> { Before { prefix: &cx.out, block: None } }

/// Folds `Icmp(cc, x, y)` against `0` or `1` when one side is a non-null address where `at` reads it.
pub(crate) fn non_null_cmp(func: &Function, def_of: &dyn Fn(Value) -> Option<Inst>, known: &dyn Fn(Value) -> Option<i64>, at: Before, (cc, x, y): (IntCC, Value, Value)) -> Option<bool> {
    let ty = func.ty(x);
    if !ty.is_small_int() || ty.bits() < 32 { return None; }
    [(cc, x, y), (flip_int_cc(cc), y, x)].into_iter().find_map(|(norm, val, rhs)| {
        null_test(norm, ty, known(rhs)?).filter(|_| non_null_at(func, def_of, known, at, val))
    })
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

fn arm_holds(cx: &RewriteCx, v: Value, steps: &mut Vec<Step>, cmp: Cmp, budget: &mut u8) -> Option<bool> {
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

fn merge_icmps(cx: &RewriteCx, is_and: bool, (v1, neg1): (Value, bool), (v2, neg2): (Value, bool)) -> Option<Op> {
    let Op::Icmp(raw1, x1, y1) = cx.func.insts[cx.def(v1)?.index()].op else { return None };
    let Op::Icmp(raw2, x2, y2) = cx.func.insts[cx.def(v2)?.index()].op else { return None };
    let cc1 = if neg1 { raw1.inverse() } else { raw1 };
    let cc2 = if neg2 { raw2.inverse() } else { raw2 };
    let aligned2 = if (x2, y2) == (x1, y1) {
        cc2
    } else if (y2, x2) == (x1, y1) {
        crate::cse::flip_int_cc(cc2)
    } else {
        return None;
    };
    let cmp_cc = |c: IntCC| !matches!(c, IntCC::Equal | IntCC::NotEqual);
    if cmp_cc(cc1) && cmp_cc(aligned2) && cc1.is_signed() != aligned2.is_signed() {
        return None;
    }
    let signed = (cmp_cc(cc1) && cc1.is_signed()) || (cmp_cc(aligned2) && aligned2.is_signed());
    let mask = if is_and { cc_mask(cc1) & cc_mask(aligned2) } else { cc_mask(cc1) | cc_mask(aligned2) };
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

fn emit_combo(cx: &mut RewriteCx, res: Option<Value>, is_and: bool, a: (Value, bool), b: (Value, bool)) -> Option<Op> {
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
    Some(Op::Binary(if is_and { Binary::Band } else { Binary::Bor }, lhs, rhs))
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
    let Op::Select(c2, a2, b2) = cx.func.insts[inst.index()].op else { return None };
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

pub(super) fn emit_cond_truth(cx: &mut RewriteCx, res: Option<Value>, raw_cond: Value, truth: bool) -> Option<Option<Op>> {
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
    Some(Some(Op::Icmp(if truth { IntCC::NotEqual } else { IntCC::Equal }, cond, zero)))
}

fn eval_one(cx: &mut RewriteCx, res: Option<Value>, cc: IntCC, val: Value, k: i64) -> Option<Option<Op>> {
    let cmp = Cmp { cc, ty: cx.func.ty(val), k };
    let (core, mut steps) = peel_steps(cx, val);
    let inst = cx.def(core)?;
    let Op::Select(cond, a, b) = cx.func.insts[inst.index()].op else {
        return None;
    };
    let (mut ba, mut bb) = (16, 16);
    let ha = arm_holds(cx, a, &mut steps, cmp, &mut ba);
    let hb = arm_holds(cx, b, &mut steps, cmp, &mut bb);
    if let (Some(ta), Some(tb)) = (ha, hb) && ta == tb {
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

fn eval_unit_cmp(cx: &mut RewriteCx, res: Option<Value>, cc: IntCC, val: Value, k: i64) -> Option<Option<Op>> {
    let ty = cx.func.ty(val);
    let (zeros, ones) = cx.known(val);
    if (zeros | ones) == ty.unsigned_max() || (zeros | 1) != ty.unsigned_max() { return None; }
    let (h0, h1) = (holds(cc, ty, 0, k), holds(cc, ty, 1, k));
    if h0 == h1 { return Some(Some(Op::Const(i64::from(h0)))); }
    let base = cx.unwrap_uextend(val);
    let bty = cx.func.ty(base);
    if !h0 {
        if bty == Type::I8 { cx.alias(res, base); return Some(None); }
        return Some(Some(Op::Convert(Convert::Ireduce, base)));
    }
    let low = if bty == Type::I8 { base } else { cx.emit(Type::I8, Op::Convert(Convert::Ireduce, base)) };
    let one = cx.emit(Type::I8, Op::Const(1));
    Some(Some(Op::Binary(Binary::Bxor, low, one)))
}

fn unwrap_strict(func: &Function, def_of: &dyn Fn(Value) -> Option<Inst>, mut v: Value) -> Option<(Value, Value, bool, bool)> {
    for _ in 0..4 {
        match func.insts[def_of(v)?.index()].op {
            Op::Icmp(cc, a, b) if !func.ty(a).in_float_register() => return match cc {
                IntCC::SignedGreaterThan => Some((a, b, true, true)),
                IntCC::SignedLessThan => Some((a, b, true, false)),
                IntCC::UnsignedGreaterThan => Some((a, b, false, true)),
                IntCC::UnsignedLessThan => Some((a, b, false, false)),
                _ => None,
            },
            Op::Convert(Convert::Uextend | Convert::Sextend | Convert::Ireduce, inner) if !func.ty(inner).in_float_register() => v = inner,
            _ => return None,
        }
    }
    None
}

pub(super) fn complementary_pair(func: &Function, def_of: &dyn Fn(Value) -> Option<Inst>, x: Value, y: Value) -> Option<(Value, Value, bool)> {
    let (a, b, s1, g1) = unwrap_strict(func, def_of, x)?;
    let (na, nb, s2, g2) = unwrap_strict(func, def_of, y)?;
    if s1 != s2 || func.ty(a) != func.ty(na) { return None; }
    ((na == a && nb == b && g2 == !g1) || (na == b && nb == a && g2 == g1)).then_some((a, b, s1))
}

pub(super) fn complementary_eq_pair(func: &Function, def_of: &dyn Fn(Value) -> Option<Inst>, x: Value, y: Value) -> Option<(Value, Value)> {
    if let Some((a, b, _)) = complementary_pair(func, def_of, x, y) { return Some((a, b)); }
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

pub(super) fn eval_select(cx: &mut RewriteCx, res: Option<Value>, cc: IntCC, x: Value, y: Value) -> Option<Option<Op>> {
    if !res.is_some_and(|r| cx.func.ty(r) == Type::I8) { return None; }
    if matches!(cc, IntCC::Equal | IntCC::NotEqual)
        && let Some((a, b)) = complementary_eq_pair(cx.func, &|v| cx.def(v), x, y)
    {
        return Some(Some(Op::Icmp(cc, a, b)));
    }
    if !cx.func.ty(x).is_small_int() { return None; }
    if let Some(bit) = non_null_cmp(cx.func, &|v| cx.def(v), &|v| cx.sconst(v), before(cx), (cc, x, y)) {
        return Some(Some(Op::Const(i64::from(bit))));
    }
    if let Some(op) = ord::eval_three_way(cx, cc, x, y).or_else(|| ord::eval_complementary_cmp(cx, cc, x, y)).or_else(|| ord::eval_unary_cmp(cx, cc, x, y)) {
        return Some(Some(op));
    }
    for (norm, val, c_val) in [(cc, x, y), (flip_int_cc(cc), y, x)] {
        if let Some(k) = cx.sconst(c_val)
            && let Some(out) = eval_one(cx, res, norm, val, k).or_else(|| eval_unit_cmp(cx, res, norm, val, k))
        {
            return Some(out);
        }
    }
    None
}

enum FusedCmp {
    Const(i64),
    Alias(Value),
    Op(Op),
}

struct RangeTest {
    x: Value,
    lo: u64,
    span: u64,
    in_range: bool,
    sub_val: Option<Value>,
}

fn peel_uext_i8(cx: &RewriteCx, v: Value) -> (Value, bool) {
    if let Some(inst) = cx.def(v)
        && let Op::Convert(Convert::Uextend, inner) = cx.func.insts[inst.index()].op
        && cx.func.ty(inner) == Type::I8
    {
        (inner, true)
    } else {
        (v, false)
    }
}

fn cmp_const(cx: &RewriteCx, v: Value) -> Option<(IntCC, Value, i64)> {
    let Op::Icmp(cc, l, r) = *cx.op(v)? else { return None };
    if !cx.func.ty(l).is_small_int() { return None; }
    cx.sconst(r).map(|c| (cc, l, c)).or_else(|| cx.sconst(l).map(|c| (flip_int_cc(cc), r, c)))
}

fn emit_cmp_const(cx: &mut RewriteCx, cc: IntCC, lhs: Value, c: i64) -> Op {
    let ty = cx.func.ty(lhs);
    let rhs = cx.emit(ty, Op::Const(extend(c, ty, true)));
    Op::Icmp(cc, lhs, rhs)
}

/// Rewrites an unsigned compare of a single-use `x + c1` against `c2` into `x` against `c2 - c1` when `x + c1` cannot wrap.
pub(super) fn strip_const_add(cx: &mut RewriteCx, cc: IntCC, x: Value, y: Value) -> Option<Op> {
    if cc.is_signed() { return None; }
    let ty = cx.func.ty(x);
    for (norm, sum, c_v) in [(cc, x, y), (flip_int_cc(cc), y, x)] {
        let Some(c2) = cx.uconst(c_v, ty) else { continue };
        let Some(&Op::Binary(Binary::Iadd, p, q)) = cx.op(sum) else { continue };
        let Some((base, c1)) = [(p, q), (q, p)].into_iter().find_map(|(b, k)| cx.uconst(k, ty).map(|c| (b, c))) else { continue };
        if c1 == 0 || c2 < c1 || !cx.single_use(sum) { continue; }
        let hi = offset_bound(cx.func, &|v| cx.def(v), &|v| cx.sconst(v), base, (None, 0));
        if hi.checked_add(c1).is_none_or(|top| top > ty.unsigned_max()) { continue; }
        return Some(emit_cmp_const(cx, norm, base, (c2 - c1) as i64));
    }
    None
}

fn emit_sub_cmp(cx: &mut RewriteCx, x: Value, lo: u64, cc: IntCC, bound: u64) -> Op {
    let ty = cx.func.ty(x);
    let lhs = if lo == 0 { x } else {
        let lo_c = cx.emit(ty, Op::Const(extend(lo as i64, ty, true)));
        cx.emit(ty, Op::Binary(Binary::Isub, x, lo_c))
    };
    emit_cmp_const(cx, cc, lhs, bound as i64)
}

fn extract_range(cx: &RewriteCx, v: Value) -> Option<RangeTest> {
    let (cc, lhs, c_imm) = cmp_const(cx, v)?;
    let ty = cx.func.ty(lhs);
    let umax = ty.unsigned_max();
    let uc = extend(c_imm, ty, false) as u64;
    let (in_range, span) = match cc {
        IntCC::UnsignedLessThanOrEqual => (true, uc),
        IntCC::UnsignedLessThan if uc > 0 => (true, uc - 1),
        IntCC::UnsignedGreaterThan => (false, uc),
        IntCC::UnsignedGreaterThanOrEqual if uc > 0 => (false, uc - 1),
        _ => return None,
    };
    let (x, lo, sub_val) = match cx.op(lhs) {
        Some(&Op::Binary(Binary::Isub, base, k)) if cx.uconst(k, ty).is_some() => (base, cx.uconst(k, ty).unwrap_or(0), Some(lhs)),
        Some(&Op::Binary(Binary::Iadd, p, q)) => [(p, q), (q, p)]
            .into_iter()
            .find_map(|(base, k)| cx.uconst(k, ty).map(|neg| (base, 0_u64.wrapping_sub(neg) & umax, Some(lhs))))
            .unwrap_or((lhs, 0, None)),
        _ => (lhs, 0, None),
    };
    (lo <= umax - span).then_some(RangeTest { x, lo, span, in_range, sub_val })
}

fn dual_range_diff(cx: &RewriteCx, pair_single: bool, r1: &RangeTest, r2: &RangeTest) -> Option<(u64, u64)> {
    if !pair_single || !r1.sub_val.is_none_or(|s| cx.single_use(s)) || !r2.sub_val.is_none_or(|s| cx.single_use(s)) { return None; }
    let (lo_min, lo_max) = if r1.lo < r2.lo { (r1.lo, r2.lo) } else { (r2.lo, r1.lo) };
    let diff = lo_min ^ lo_max;
    (diff.is_power_of_two() && (lo_min & diff) == 0 && r1.span < diff && (lo_min & (diff - 1)) < diff - r1.span).then_some((diff, lo_max))
}

fn fuse_dual_ranges(cx: &mut RewriteCx, pair_single: bool, is_and: bool, a: Value, b: Value) -> Option<FusedCmp> {
    let (r1, r2) = (extract_range(cx, a)?, extract_range(cx, b)?);
    if r1.x != r2.x || r1.in_range == is_and || r2.in_range == is_and || r1.span != r2.span || r1.lo == r2.lo { return None; }
    let (diff, lo_max) = dual_range_diff(cx, pair_single, &r1, &r2)?;
    let ty = cx.func.ty(r1.x);
    let diff_c = cx.emit(ty, Op::Const(extend(diff as i64, ty, true)));
    let folded = cx.emit(ty, Op::Binary(Binary::Bor, r1.x, diff_c));
    let cc = if is_and { IntCC::UnsignedGreaterThan } else { IntCC::UnsignedLessThanOrEqual };
    Some(FusedCmp::Op(emit_sub_cmp(cx, folded, lo_max, cc, r1.span)))
}

fn fuse_range_adj_eq(cx: &mut RewriteCx, pair_single: bool, is_and: bool, a: Value, b: Value) -> Option<FusedCmp> {
    let want_in = !is_and;
    let eq_cc = if is_and { IntCC::NotEqual } else { IntCC::Equal };
    for (r_v, eq_v) in [(a, b), (b, a)] {
        let Some(r) = extract_range(cx, r_v) else { continue };
        let Some((cc, eq_x, eq_c)) = cmp_const(cx, eq_v) else { continue };
        if r.in_range != want_in || cc != eq_cc || eq_x != r.x { continue; }
        let ty = cx.func.ty(r.x);
        let umax = ty.unsigned_max();
        let eu = extend(eq_c, ty, false) as u64;
        let new_lo = if r.lo + r.span < umax && eu == r.lo + r.span + 1 { r.lo } else if r.lo > 0 && eu == r.lo - 1 { r.lo - 1 } else { continue };
        let new_span = r.span + 1;
        if new_span >= umax || (new_lo != 0 && (!pair_single || !r.sub_val.is_none_or(|s| cx.single_use(s)))) { continue; }
        let out_cc = if is_and { IntCC::UnsignedGreaterThan } else { IntCC::UnsignedLessThanOrEqual };
        return Some(FusedCmp::Op(emit_sub_cmp(cx, r.x, new_lo, out_cc, new_span)));
    }
    None
}

fn fuse_same_eq(cx: &mut RewriteCx, pair_single: bool, is_and: bool, (a, x1, c1, c2): (Value, Value, i64, i64)) -> Option<FusedCmp> {
    let ty = cx.func.ty(x1);
    let (u1, u2) = (extend(c1, ty, false) as u64, extend(c2, ty, false) as u64);
    let (lo, hi) = if u1 <= u2 { (u1, u2) } else { (u2, u1) };
    if lo == hi { return Some(FusedCmp::Alias(a)); }
    let out_cc = if is_and { IntCC::UnsignedGreaterThan } else { IntCC::UnsignedLessThanOrEqual };
    if lo == 0 && hi == 1 { return Some(FusedCmp::Op(emit_sub_cmp(cx, x1, 0, out_cc, 1))); }
    if !pair_single { return None; }
    let diff = hi - lo;
    if diff.is_power_of_two() && (lo & diff) == 0 {
        let mask_c = cx.emit(ty, Op::Const(extend(diff as i64, ty, true)));
        let masked = cx.emit(ty, Op::Binary(Binary::Bor, x1, mask_c));
        return Some(FusedCmp::Op(emit_cmp_const(cx, if is_and { IntCC::NotEqual } else { IntCC::Equal }, masked, hi as i64)));
    }
    (hi == lo + 1 && ty.unsigned_max() >= 2).then(|| FusedCmp::Op(emit_sub_cmp(cx, x1, lo, out_cc, 1)))
}

fn fuse_eq_pairs(cx: &mut RewriteCx, pair_single: bool, is_and: bool, a: Value, b: Value, cmp1: (IntCC, Value, i64), cmp2: (IntCC, Value, i64)) -> Option<FusedCmp> {
    let ((cc1, x1, c1), (cc2, _, c2)) = (cmp1, cmp2);
    let eq_cc = if is_and { IntCC::NotEqual } else { IntCC::Equal };
    if cc1 == eq_cc && cc2 == eq_cc {
        return fuse_same_eq(cx, pair_single, is_and, (a, x1, c1, c2));
    }
    let ty = cx.func.ty(x1);
    for ((p_cc, p_v, p_c), (q_cc, q_v, q_c)) in [((cc1, a, c1), (cc2, b, c2)), ((cc2, b, c2), (cc1, a, c1))] {
        if !matches!(p_cc, IntCC::Equal | IntCC::NotEqual) { continue; }
        let eval = holds(q_cc, ty, p_c, q_c);
        if (is_and && p_cc == IntCC::Equal) || (!is_and && p_cc == IntCC::NotEqual) {
            return Some(if eval == is_and { FusedCmp::Alias(p_v) } else { FusedCmp::Const(i64::from(!is_and)) });
        }
        if eval != is_and { return Some(FusedCmp::Alias(q_v)); }
    }
    None
}

fn cmp_interval(ty: Type, cc: IntCC, c: i64, signed: bool) -> Option<((i128, i128), i128, i128)> {
    let (d_lo, d_hi, v) = if signed {
        let half = 1_i128 << (ty.bits() - 1);
        (-half, half - 1, i128::from(extend(c, ty, true)))
    } else {
        (0_i128, i128::from(ty.unsigned_max()), i128::from(extend(c, ty, false) as u64))
    };
    let iv = match cc {
        IntCC::SignedLessThan | IntCC::UnsignedLessThan => (d_lo, v - 1),
        IntCC::SignedLessThanOrEqual | IntCC::UnsignedLessThanOrEqual => (d_lo, v),
        IntCC::SignedGreaterThan | IntCC::UnsignedGreaterThan => (v + 1, d_hi),
        IntCC::SignedGreaterThanOrEqual | IntCC::UnsignedGreaterThanOrEqual => (v, d_hi),
        _ => return None,
    };
    Some((iv, d_lo, d_hi))
}

fn fuse_interval_bounds(is_and: bool, (a, b): (Value, Value), (l1, h1): (i128, i128), (l2, h2): (i128, i128), (d_lo, d_hi): (i128, i128)) -> Option<FusedCmp> {
    for ((lo, hi), other) in [((l1, h1), b), ((l2, h2), a)] {
        if lo > hi { return Some(if is_and { FusedCmp::Const(0) } else { FusedCmp::Alias(other) }); }
        if lo == d_lo && hi == d_hi { return Some(if is_and { FusedCmp::Alias(other) } else { FusedCmp::Const(1) }); }
    }
    if l1 <= l2 && h2 <= h1 { return Some(FusedCmp::Alias(if is_and { b } else { a })); }
    (l2 <= l1 && h1 <= h2).then(|| FusedCmp::Alias(if is_and { a } else { b }))
}

fn fuse_or_intervals(cx: &mut RewriteCx, pair_single: bool, x: Value, ivs: ((i128, i128), (i128, i128)), (d_lo, d_hi): (i128, i128)) -> Option<FusedCmp> {
    let ((l1, h1), (l2, h2)) = if ivs.0.0 <= ivs.1.0 { ivs } else { (ivs.1, ivs.0) };
    if h1 + 1 >= l2 { return (l1 == d_lo && h1.max(h2) == d_hi).then_some(FusedCmp::Const(1)); }
    if l1 != d_lo || h2 != d_hi { return None; }
    let (g_lo, g_hi) = (h1 + 1, l2 - 1);
    if g_lo == g_hi { return Some(FusedCmp::Op(emit_cmp_const(cx, IntCC::NotEqual, x, g_lo as i64))); }
    let umax = cx.func.ty(x).unsigned_max();
    let span = (g_hi - g_lo) as u64;
    (pair_single && span < umax).then(|| FusedCmp::Op(emit_sub_cmp(cx, x, (g_lo as u64) & umax, IntCC::UnsignedGreaterThan, span)))
}

fn fuse_intervals(cx: &mut RewriteCx, pair_single: bool, is_and: bool, a: Value, b: Value, cmp1: (IntCC, Value, i64), cmp2: (IntCC, Value, i64)) -> Option<FusedCmp> {
    let ((cc1, x, c1), (cc2, _, c2)) = (cmp1, cmp2);
    let signed = cc1.is_signed();
    if signed != cc2.is_signed() { return None; }
    let ty = cx.func.ty(x);
    let ((l1, h1), d_lo, d_hi) = cmp_interval(ty, cc1, c1, signed)?;
    let ((l2, h2), _, _) = cmp_interval(ty, cc2, c2, signed)?;
    if let Some(trivial) = fuse_interval_bounds(is_and, (a, b), (l1, h1), (l2, h2), (d_lo, d_hi)) { return Some(trivial); }
    if !is_and { return fuse_or_intervals(cx, pair_single, x, ((l1, h1), (l2, h2)), (d_lo, d_hi)); }
    let (i_lo, i_hi) = (l1.max(l2), h1.min(h2));
    if i_lo > i_hi { return Some(FusedCmp::Const(0)); }
    if i_lo == i_hi { return Some(FusedCmp::Op(emit_cmp_const(cx, IntCC::Equal, x, i_lo as i64))); }
    let (umax, span) = (ty.unsigned_max(), (i_hi - i_lo) as u64);
    (pair_single && span < umax).then(|| FusedCmp::Op(emit_sub_cmp(cx, x, (i_lo as u64) & umax, IntCC::UnsignedLessThanOrEqual, span)))
}

fn finish_fused(cx: &mut RewriteCx, res: Option<Value>, ext_a: bool, fused: FusedCmp) -> Option<Op> {
    match fused {
        FusedCmp::Const(c) => Some(Op::Const(c)),
        FusedCmp::Alias(v) if ext_a => Some(Op::Convert(Convert::Uextend, v)),
        FusedCmp::Alias(v) => { cx.alias(res, v); None }
        FusedCmp::Op(op) if ext_a => { let inner = cx.emit(Type::I8, op); Some(Op::Convert(Convert::Uextend, inner)) }
        FusedCmp::Op(op) => Some(op),
    }
}

pub(super) fn fuse_cmp_ops(cx: &mut RewriteCx, res: Option<Value>, is_and: bool, raw_a: Value, raw_b: Value) -> Option<Op> {
    let ((a, ext_a), (b, ext_b)) = (peel_uext_i8(cx, raw_a), peel_uext_i8(cx, raw_b));
    if ext_a != ext_b { return None; }
    let pair_single = cx.single_use(raw_a) && cx.single_use(raw_b) && (!ext_a || (cx.single_use(a) && cx.single_use(b)));
    let fused = fuse_dual_ranges(cx, pair_single, is_and, a, b)
        .or_else(|| fuse_range_adj_eq(cx, pair_single, is_and, a, b))
        .or_else(|| {
            let (cmp1, cmp2) = (cmp_const(cx, a)?, cmp_const(cx, b)?);
            (cmp1.1 == cmp2.1).then(|| fuse_eq_pairs(cx, pair_single, is_and, a, b, cmp1, cmp2).or_else(|| fuse_intervals(cx, pair_single, is_and, a, b, cmp1, cmp2))).flatten()
        })?;
    finish_fused(cx, res, ext_a, fused)
}

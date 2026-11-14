use crate::cse::flip_int_cc;
use crate::fold::{constant, extend, holds};
use crate::ir::{Binary, Convert, IntCC, Op, Type, Unary, Value};

use crate::algebra::RewriteCx;

const MAX_ORD_DEPTH: u8 = 4;

struct ThreeWay {
    a: Value,
    b: Value,
    signed: bool,
    vals: [i64; 3],
}

fn unwrap_bool_cmp(cx: &RewriteCx, mut v: Value) -> Option<(IntCC, Value, Value)> {
    for _ in 0..2 {
        let inst = cx.def(v)?;
        match cx.func.insts[inst.index()].op {
            Op::Icmp(cc, x, y) if cx.func.ty(x).is_small_int() => return Some((cc, x, y)),
            Op::Convert(Convert::Uextend | Convert::Sextend | Convert::Ireduce, inner)
                if cx.func.ty(inner).is_small_int() =>
            {
                v = inner;
            }
            _ => return None,
        }
    }
    None
}

fn norm_strict(cc: IntCC, x: Value, y: Value) -> Option<(Value, Value, bool, bool)> {
    match cc {
        IntCC::SignedGreaterThan => Some((x, y, true, true)),
        IntCC::SignedLessThan => Some((x, y, true, false)),
        IntCC::UnsignedGreaterThan => Some((x, y, false, true)),
        IntCC::UnsignedLessThan => Some((x, y, false, false)),
        _ => None,
    }
}

fn match_base_ord(cx: &RewriteCx, v: Value) -> Option<ThreeWay> {
    let ty = cx.func.ty(v);
    if !ty.is_small_int() {
        return None;
    }
    let inst = cx.def(v)?;
    let Op::Binary(Binary::Isub, pos, neg) = cx.func.insts[inst.index()].op else {
        return None;
    };
    let (p_cc, p_x, p_y) = unwrap_bool_cmp(cx, pos)?;
    let (n_cc, n_x, n_y) = unwrap_bool_cmp(cx, neg)?;
    let (a, b, signed, pos_is_gt) = norm_strict(p_cc, p_x, p_y)?;
    let (na, nb, n_signed, neg_is_gt) = norm_strict(n_cc, n_x, n_y)?;
    if signed != n_signed || cx.func.ty(a) != cx.func.ty(na) {
        return None;
    }
    let same_opposite = na == a && nb == b && neg_is_gt == !pos_is_gt;
    let swapped_same = na == b && nb == a && neg_is_gt == pos_is_gt;
    if !same_opposite && !swapped_same {
        return None;
    }
    let neg_one = extend(-1, ty, true);
    let vals = if pos_is_gt { [neg_one, 0, 1] } else { [1, 0, neg_one] };
    Some(ThreeWay { a, b, signed, vals })
}

fn match_three_way(cx: &RewriteCx, v: Value, depth: u8) -> Option<ThreeWay> {
    if depth > MAX_ORD_DEPTH {
        return None;
    }
    if let Some(tw) = match_base_ord(cx, v) {
        return Some(tw);
    }
    let ty = cx.func.ty(v);
    if !ty.is_small_int() {
        return None;
    }
    let inst = cx.def(v)?;
    match cx.func.insts[inst.index()].op {
        Op::Convert(kind @ (Convert::Ireduce | Convert::Sextend | Convert::Uextend), inner)
            if cx.func.ty(inner).is_small_int() =>
        {
            let src_ty = cx.func.ty(inner);
            let mut tw = match_three_way(cx, inner, depth + 1)?;
            tw.vals = tw.vals.map(|x| match kind {
                Convert::Uextend => extend(x, src_ty, false),
                Convert::Sextend => extend(x, src_ty, true),
                Convert::Ireduce => extend(x, ty, false),
                _ => x,
            });
            Some(tw)
        }
        Op::Binary(op @ (Binary::Iadd | Binary::Isub | Binary::Bxor | Binary::Band), x, y) => {
            if let Some(imm) = cx.sconst(y) {
                let mut tw = match_three_way(cx, x, depth + 1)?;
                tw.vals = tw.vals.map(|val| extend(constant(op, val, imm).unwrap_or(val), ty, false));
                return Some(tw);
            }
            if let Some(imm) = cx.sconst(x) {
                let mut tw = match_three_way(cx, y, depth + 1)?;
                tw.vals = tw.vals.map(|val| extend(constant(op, imm, val).unwrap_or(val), ty, false));
                return Some(tw);
            }
            None
        }
        _ => None,
    }
}

pub(super) fn eval_three_way(cx: &RewriteCx, cc: IntCC, x: Value, y: Value) -> Option<Op> {
    let ty = cx.func.ty(x);
    for (norm, expr_val, c_val) in [(cc, x, y), (flip_int_cc(cc), y, x)] {
        let Some(imm) = cx.sconst(c_val) else { continue };
        let Some(tw) = match_three_way(cx, expr_val, 0) else { continue };
        let b_lt = holds(norm, ty, tw.vals[0], imm);
        let b_eq = holds(norm, ty, tw.vals[1], imm);
        let b_gt = holds(norm, ty, tw.vals[2], imm);
        let (lt_cc, gt_cc, ge_cc, le_cc) = if tw.signed {
            (
                IntCC::SignedLessThan,
                IntCC::SignedGreaterThan,
                IntCC::SignedGreaterThanOrEqual,
                IntCC::SignedLessThanOrEqual,
            )
        } else {
            (
                IntCC::UnsignedLessThan,
                IntCC::UnsignedGreaterThan,
                IntCC::UnsignedGreaterThanOrEqual,
                IntCC::UnsignedLessThanOrEqual,
            )
        };
        let op = match (b_lt, b_eq, b_gt) {
            (false, false, false) => Op::Const(0),
            (true, true, true) => Op::Const(1),
            (true, false, false) => Op::Icmp(lt_cc, tw.a, tw.b),
            (false, true, false) => Op::Icmp(IntCC::Equal, tw.a, tw.b),
            (false, false, true) => Op::Icmp(gt_cc, tw.a, tw.b),
            (false, true, true) => Op::Icmp(ge_cc, tw.a, tw.b),
            (true, true, false) => Op::Icmp(le_cc, tw.a, tw.b),
            (true, false, true) => Op::Icmp(IntCC::NotEqual, tw.a, tw.b),
        };
        return Some(op);
    }
    None
}

fn peel_lossless_convs(cx: &RewriteCx, mut v: Value) -> Value {
    for _ in 0..2 {
        let Some(inst) = cx.def(v) else { break };
        if let Op::Convert(Convert::Uextend | Convert::Sextend | Convert::Ireduce, inner) = cx.func.insts[inst.index()].op
            && cx.func.ty(inner).is_small_int()
        {
            v = inner;
        } else {
            break;
        }
    }
    v
}

fn eval_count_cmp(cx: &mut RewriteCx, norm: IntCC, expr_val: Value, imm: i64) -> Option<Op> {
    let cmp_ty = cx.func.ty(expr_val);
    let core = peel_lossless_convs(cx, expr_val);
    let inst = cx.def(core)?;
    let Op::Unary(uop @ (Unary::Popcnt | Unary::Clz | Unary::Ctz), s) = cx.func.insts[inst.index()].op else {
        return None;
    };
    let src_ty = cx.func.ty(s);
    if !src_ty.is_small_int() {
        return None;
    }
    let w = i64::from(src_ty.bits());
    let mut sat_count = 0_i64;
    let (mut sat_0, mut sat_w, mut sat_wm1) = (false, false, false);
    for v in 0..=w {
        if holds(norm, cmp_ty, v, imm) {
            sat_count += 1;
            if v == 0 {
                sat_0 = true;
            }
            if v == w {
                sat_w = true;
            }
            if v == w - 1 {
                sat_wm1 = true;
            }
        }
    }
    if sat_count == 0 {
        return Some(Op::Const(0));
    }
    if sat_count == w + 1 {
        return Some(Op::Const(1));
    }
    let cmp_const = |cx: &mut RewriteCx, cc: IntCC, k: i64| {
        let c = cx.emit(src_ty, Op::Const(k));
        Op::Icmp(cc, s, c)
    };
    match uop {
        Unary::Popcnt => {
            if sat_count == 1 && sat_0 {
                return Some(cmp_const(cx, IntCC::Equal, 0));
            }
            if sat_count == w && !sat_0 {
                return Some(cmp_const(cx, IntCC::NotEqual, 0));
            }
            let all_ones = extend(-1, src_ty, true);
            if sat_count == 1 && sat_w {
                return Some(cmp_const(cx, IntCC::Equal, all_ones));
            }
            if sat_count == w && !sat_w {
                return Some(cmp_const(cx, IntCC::NotEqual, all_ones));
            }
        }
        Unary::Clz | Unary::Ctz => {
            if sat_count == 1 && sat_w {
                return Some(cmp_const(cx, IntCC::Equal, 0));
            }
            if sat_count == w && !sat_w {
                return Some(cmp_const(cx, IntCC::NotEqual, 0));
            }
            if uop == Unary::Clz {
                if sat_count == 1 && sat_wm1 {
                    return Some(cmp_const(cx, IntCC::Equal, 1));
                }
                if sat_count == w && !sat_wm1 {
                    return Some(cmp_const(cx, IntCC::NotEqual, 1));
                }
                if sat_count == 1 && sat_0 {
                    return Some(cmp_const(cx, IntCC::SignedLessThan, 0));
                }
                if sat_count == w && !sat_0 {
                    return Some(cmp_const(cx, IntCC::SignedGreaterThanOrEqual, 0));
                }
            }
        }
        _ => {}
    }
    None
}

fn is_involution(uop: Unary, ty: Type) -> bool {
    match uop {
        Unary::Bswap => ty.bits() >= 16,
        Unary::Bitrev | Unary::Bnot | Unary::Ineg => true,
        _ => false,
    }
}

fn apply_involution(uop: Unary, ty: Type, a: i64) -> i64 {
    let unused = 64 - ty.bits();
    let u = extend(a, ty, false) as u64;
    match uop {
        Unary::Ineg => a.wrapping_neg(),
        Unary::Bnot => !a,
        Unary::Bswap => (u.swap_bytes() >> unused) as i64,
        Unary::Bitrev => (u.reverse_bits() >> unused) as i64,
        _ => a,
    }
}

pub(super) fn eval_unary_cmp(cx: &mut RewriteCx, cc: IntCC, x: Value, y: Value) -> Option<Op> {
    for (norm, expr_val, c_val) in [(cc, x, y), (flip_int_cc(cc), y, x)] {
        if let Some(imm) = cx.sconst(c_val)
            && let Some(op) = eval_count_cmp(cx, norm, expr_val, imm)
        {
            return Some(op);
        }
    }
    if !matches!(cc, IntCC::Equal | IntCC::NotEqual) {
        return None;
    }
    let ty = cx.func.ty(x);
    if let (Some(xi), Some(yi)) = (cx.def(x), cx.def(y))
        && let Op::Unary(ux, a) = cx.func.insts[xi.index()].op
        && let Op::Unary(uy, b) = cx.func.insts[yi.index()].op
        && ux == uy
        && is_involution(ux, ty)
    {
        return Some(Op::Icmp(cc, a, b));
    }
    for (expr_val, c_val) in [(x, y), (y, x)] {
        if let Some(imm) = cx.sconst(c_val)
            && let Some(inst) = cx.def(expr_val)
            && let Op::Unary(uop, a) = cx.func.insts[inst.index()].op
            && is_involution(uop, ty)
        {
            let inv = apply_involution(uop, ty, imm);
            let c2 = cx.emit(ty, Op::Const(inv));
            return Some(Op::Icmp(cc, a, c2));
        }
    }
    None
}

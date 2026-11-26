mod nonzero;
mod ord;

pub(super) use nonzero::{is_non_zero, tested_select};
pub(super) use ord::{build_ord_select, match_ord_select};

use crate::fold::extend;
use crate::ir::{Binary, Convert, IntCC, Op, Type, Value};

use super::RewriteCx;
use super::icmp::{emit_cond_truth, null_test};

pub(super) fn narrow_arm(cx: &RewriteCx, dst: Type, v: Value) -> Option<Result<Value, i64>> {
    if let Some(&Op::Convert(Convert::Uextend | Convert::Sextend, inner)) = cx.op(v)
        && cx.func.ty(inner) == dst
    {
        return Some(Ok(inner));
    }
    cx.sconst(v).map(|c| Err(extend(c, dst, true)))
}

pub(super) fn simplify_select(
    cx: &mut RewriteCx,
    res: Option<Value>,
    cond: Value,
    t: Value,
    f: Value,
) -> Option<Op> {
    let given = (t, f);
    if t == f {
        cx.alias(res, t);
        return None;
    }
    if is_non_zero(cx, cond, 0) {
        cx.alias(res, t);
        return None;
    }
    let inner = cx.unwrap_uextend(cond);
    if let Some(arm) = tested_arm(cx, inner, t, f) {
        cx.alias(res, arm);
        return None;
    }
    let (t, f) = (
        refuted_arm(cx, inner, t, true).unwrap_or(t),
        refuted_arm(cx, inner, f, false).unwrap_or(f),
    );
    if t == f {
        cx.alias(res, t);
        return None;
    }
    if let Some(op) = simplify_unit_bool_select(cx, res, inner, t, f) {
        return op;
    }
    ord::fold_ord_select_cmp(cx, inner, t, f)
        .or_else(|| factor_select_add(cx, inner, t, f))
        .or_else(|| factor_select_binary(cx, inner, t, f))
        .or_else(|| (inner != cond || (t, f) != given).then_some(Op::Select(inner, t, f)))
}

fn factor_select_binary(cx: &mut RewriteCx, cond: Value, t: Value, f: Value) -> Option<Op> {
    if !cx.single_use(t) || !cx.single_use(f) {
        return None;
    }
    let (&Op::Binary(op_t, tl, tr), &Op::Binary(op_f, fl, fr)) = (cx.op(t)?, cx.op(f)?) else {
        return None;
    };
    if op_t != op_f {
        return None;
    }
    let ty = cx.func.ty(t);
    let comm = matches!(
        op_t,
        Binary::Iadd
            | Binary::Imul
            | Binary::Band
            | Binary::Bor
            | Binary::Bxor
            | Binary::Fadd
            | Binary::Fmul
    );
    if !comm && !matches!(op_t, Binary::Isub | Binary::Fsub) {
        return None;
    }
    if cx.func.ty(tl) != ty || cx.func.ty(tr) != ty {
        return None;
    }
    let (common, a, b, lhs_common) = if tl == fl {
        (tl, tr, fr, true)
    } else if tr == fr {
        (tr, tl, fl, false)
    } else if comm && tl == fr {
        (tl, tr, fl, true)
    } else if comm && tr == fl {
        (tr, tl, fr, true)
    } else {
        return None;
    };
    let sel_op = factor_select_binary(cx, cond, a, b).unwrap_or(Op::Select(cond, a, b));
    let sel = cx.emit(ty, sel_op);
    Some(if lhs_common {
        Op::Binary(op_t, common, sel)
    } else {
        Op::Binary(op_t, sel, common)
    })
}

fn match_acc_delta(cx: &RewriteCx, ty: Type, cand: Value, acc: Value) -> Option<(Binary, Value)> {
    if !cx.single_use(cand) {
        return None;
    }
    match *cx.op(cand)? {
        Op::Binary(Binary::Iadd, a, b) => {
            let delta = if a == acc {
                b
            } else if b == acc {
                a
            } else {
                return None;
            };
            (cx.uconst(delta, ty) != Some(1)).then_some((Binary::Iadd, delta))
        }
        Op::Binary(Binary::Isub, a, b) if a == acc => Some((Binary::Isub, b)),
        _ => None,
    }
}

fn factor_select_add(cx: &mut RewriteCx, cond: Value, t: Value, f: Value) -> Option<Op> {
    let ty = cx.func.ty(t);
    if !matches!(ty, Type::I32 | Type::I64) {
        return None;
    }
    let (acc, bin, t_arm, f_arm) = if let Some((bin, delta)) = match_acc_delta(cx, ty, t, f) {
        let zero = cx.emit(ty, Op::Const(0));
        (f, bin, delta, zero)
    } else if let Some((bin, delta)) = match_acc_delta(cx, ty, f, t) {
        let zero = cx.emit(ty, Op::Const(0));
        (t, bin, zero, delta)
    } else {
        return None;
    };
    let sel = cx.emit(ty, Op::Select(cond, t_arm, f_arm));
    Some(Op::Binary(bin, acc, sel))
}

fn simplify_unit_bool_select(
    cx: &mut RewriteCx,
    res: Option<Value>,
    inner: Value,
    t: Value,
    f: Value,
) -> Option<Option<Op>> {
    if !res.is_some_and(|r| cx.func.ty(r) == Type::I8)
        || cx.func.ty(inner) != Type::I8
        || !cx.is_unit_bool(inner)
    {
        return None;
    }
    match (cx.uconst(t, Type::I8), cx.uconst(f, Type::I8)) {
        (Some(1), Some(0)) => {
            cx.alias(res, inner);
            Some(None)
        }
        (Some(1), _) if cx.is_unit_bool(f) => Some(Some(Op::Binary(Binary::Bor, inner, f))),
        (_, Some(0)) if cx.is_unit_bool(t) => Some(Some(Op::Binary(Binary::Band, inner, t))),
        (Some(0), Some(1)) => Some(Some(Op::Binary(Binary::Bxor, inner, f))),
        (Some(0), _) if cx.is_unit_bool(f) => band_without(cx, inner, f).map(Some),
        _ => None,
    }
}

fn band_without(cx: &mut RewriteCx, cond: Value, kept: Value) -> Option<Op> {
    let &Op::Binary(Binary::Band, x, y) = cx.op(cond)? else {
        return None;
    };
    let other = if x == kept {
        y
    } else if y == kept {
        x
    } else {
        return None;
    };
    if cx.func.ty(other) != Type::I8 || !cx.is_unit_bool(other) {
        return None;
    }
    let not = emit_bool_not(cx, other);
    Some(Op::Binary(Binary::Band, kept, not))
}

fn opposite(cx: &RewriteCx, a: Value, b: Value) -> bool {
    let (Some(&Op::Icmp(ca, xa, ya)), Some(&Op::Icmp(cb, xb, yb))) = (cx.op(a), cx.op(b)) else {
        return false;
    };
    cx.func.ty(xa).is_small_int()
        && ((xa == xb && ya == yb && cb == ca.inverse())
            || (xa == yb && ya == xb && cb == ca.inverse().swapped()))
}

fn refuted_arm(cx: &RewriteCx, cond: Value, arm: Value, taken: bool) -> Option<Value> {
    let &Op::Select(test, x, y) = cx.op(arm)? else {
        return None;
    };
    opposite(cx, cond, cx.unwrap_uextend(test)).then_some(if taken { y } else { x })
}

fn tested_arm(cx: &RewriteCx, cond: Value, t: Value, f: Value) -> Option<Value> {
    let ty = cx.func.ty(t);
    if !ty.is_small_int() {
        return None;
    }
    let key = |v: Value| cx.uconst(v, ty).map_or(Ok(v), Err);
    let (pair, differ) = match *cx.op(cond)? {
        Op::Icmp(cc @ (IntCC::Equal | IntCC::NotEqual), a, b) if cx.func.ty(a) == ty => {
            ([key(a), key(b)], cc == IntCC::NotEqual)
        }
        Op::Binary(Binary::Isub | Binary::Bxor, a, b) if cx.func.ty(cond) == ty => {
            ([key(a), key(b)], true)
        }
        Op::Binary(Binary::Iadd, a, b) if cx.func.ty(cond) == ty => {
            let (v, k) = [(a, b), (b, a)]
                .into_iter()
                .find_map(|(v, k)| Some((v, cx.uconst(k, ty)?)))?;
            ([key(v), Err(k.wrapping_neg() & ty.unsigned_max())], true)
        }
        _ => return None,
    };
    let arms = [key(t), key(f)];
    (pair == arms || pair == [arms[1], arms[0]]).then_some(if differ { t } else { f })
}

fn bool_cond(cx: &RewriteCx, raw: Value) -> Option<Value> {
    let c = cx.unwrap_uextend(raw);
    (cx.func.ty(c) == Type::I8 && cx.is_unit_bool(c)).then_some(c)
}

fn simplify_1_bool(
    cx: &mut RewriteCx,
    res: Option<Value>,
    c: Value,
    b1: bool,
    b2: bool,
) -> Option<Op> {
    match (b1, b2) {
        (true, true) => Some(Op::Const(1)),
        (false, false) => Some(Op::Const(0)),
        (true, false) => {
            cx.alias(res, c);
            None
        }
        (false, true) => {
            let one = cx.emit(Type::I8, Op::Const(1));
            Some(Op::Binary(Binary::Bxor, c, one))
        }
    }
}

fn emit_bool_not(cx: &mut RewriteCx, v: Value) -> Value {
    let one = cx.emit(Type::I8, Op::Const(1));
    cx.emit(Type::I8, Op::Binary(Binary::Bxor, v, one))
}

fn invert_op(cx: &mut RewriteCx, bin: Binary, a: Value, b: Value) -> Op {
    let inner = cx.emit(Type::I8, Op::Binary(bin, a, b));
    let one = cx.emit(Type::I8, Op::Const(1));
    Op::Binary(Binary::Bxor, inner, one)
}

fn eval_const_select_cmp(
    cx: &mut RewriteCx,
    res: Option<Value>,
    cc: IntCC,
    x: Value,
    y: Value,
) -> Option<Option<Op>> {
    let ty = cx.func.ty(x);
    for (norm, sel_val, c_val) in [(cc, x, y), (cc.swapped(), y, x)] {
        let (Some(c), Some(&Op::Select(raw_c1, t1, f1))) = (cx.sconst(c_val), cx.op(sel_val))
        else {
            continue;
        };
        let Some(c1) = bool_cond(cx, raw_c1) else {
            continue;
        };
        let eval = |k: i64| crate::fold::holds(norm, ty, k, c);
        if let (Some(k1), Some(k2)) = (cx.sconst(t1), cx.sconst(f1)) {
            return Some(simplify_1_bool(cx, res, c1, eval(k1), eval(k2)));
        }
        if let (Some(k1), Some(&Op::Select(raw_c2, t2, f2))) = (cx.sconst(t1), cx.op(f1))
            && let (Some(k2), Some(k3)) = (cx.sconst(t2), cx.sconst(f2))
        {
            let (b1, b2, b3) = (eval(k1), eval(k2), eval(k3));
            if b2 == b3 {
                return Some(simplify_1_bool(cx, res, c1, b1, b2));
            }
            let Some(c2) = bool_cond(cx, raw_c2) else {
                continue;
            };
            return Some(Some(match (b1, b2, b3) {
                (false, false, true) => invert_op(cx, Binary::Bor, c1, c2),
                (true, true, false) => Op::Binary(Binary::Bor, c1, c2),
                (false, true, false) => Op::Binary(Binary::Band, emit_bool_not(cx, c1), c2),
                _ => Op::Binary(Binary::Bor, c1, emit_bool_not(cx, c2)),
            }));
        }
        if let (Some(&Op::Select(raw_c2, t2, f2)), Some(k3)) = (cx.op(t1), cx.sconst(f1))
            && let (Some(k1), Some(k2)) = (cx.sconst(t2), cx.sconst(f2))
        {
            let (b1, b2, b3) = (eval(k1), eval(k2), eval(k3));
            if b1 == b2 {
                return Some(simplify_1_bool(cx, res, c1, b1, b3));
            }
            let Some(c2) = bool_cond(cx, raw_c2) else {
                continue;
            };
            return Some(Some(match (b1, b2, b3) {
                (true, false, false) => Op::Binary(Binary::Band, c1, c2),
                (false, true, false) => Op::Binary(Binary::Band, c1, emit_bool_not(cx, c2)),
                (true, false, true) => Op::Binary(Binary::Bor, emit_bool_not(cx, c1), c2),
                _ => invert_op(cx, Binary::Band, c1, c2),
            }));
        }
    }
    None
}

#[derive(Clone, Copy)]
enum ArmRel {
    EqK,
    NeqK,
    Unknown,
}

fn cmp_target(cc: IntCC, ty: Type, k_s: i64) -> Option<(u64, bool)> {
    match cc {
        IntCC::Equal | IntCC::NotEqual => Some((extend(k_s, ty, false) as u64, cc == IntCC::Equal)),
        _ => null_test(cc, ty, k_s).map(|bit| (0, !bit)),
    }
}

fn classify_arm(
    cx: &RewriteCx,
    v: Value,
    ty: Type,
    k_u: u64,
    cmp: Option<(IntCC, Value, Value)>,
) -> ArmRel {
    if let Some(c) = cx.uconst(v, ty) {
        return if c == k_u { ArmRel::EqK } else { ArmRel::NeqK };
    }
    let (zeros, ones) = cx.known(v);
    if (zeros | ones) == ty.unsigned_max() {
        return if ones == k_u {
            ArmRel::EqK
        } else {
            ArmRel::NeqK
        };
    }
    if (ones & !k_u) != 0
        || (zeros & k_u) != 0
        || (k_u == 0 && nonzero::is_non_zero_under(cx, v, cmp, 0))
    {
        return ArmRel::NeqK;
    }
    ArmRel::Unknown
}

pub(super) fn eval_select_zero_cmp(
    cx: &mut RewriteCx,
    res: Option<Value>,
    cc: IntCC,
    x: Value,
    y: Value,
) -> Option<Option<Op>> {
    let ty = cx.func.ty(x);
    if !ty.is_small_int() || !res.is_some_and(|r| cx.func.ty(r) == Type::I8) {
        return None;
    }
    if let Some(folded) = eval_const_select_cmp(cx, res, cc, x, y) {
        return Some(folded);
    }
    for (norm, sel_val, c_val) in [(cc, x, y), (cc.swapped(), y, x)] {
        let Some((k_u, want_match)) = cx.sconst(c_val).and_then(|k_s| cmp_target(norm, ty, k_s))
        else {
            continue;
        };
        let Some(&Op::Select(raw_cond, t_val, f_val)) = cx.op(sel_val) else {
            continue;
        };
        let t_cmp = (k_u == 0)
            .then(|| nonzero::unwrap_cond_cmp(cx, raw_cond, true))
            .flatten();
        let f_cmp = (k_u == 0)
            .then(|| nonzero::unwrap_cond_cmp(cx, raw_cond, false))
            .flatten();
        let out = match (
            classify_arm(cx, t_val, ty, k_u, t_cmp),
            classify_arm(cx, f_val, ty, k_u, f_cmp),
        ) {
            (ArmRel::EqK, ArmRel::EqK) => Some(Some(Op::Const(i64::from(want_match)))),
            (ArmRel::NeqK, ArmRel::NeqK) => Some(Some(Op::Const(i64::from(!want_match)))),
            (ArmRel::EqK, ArmRel::NeqK) => emit_cond_truth(cx, res, raw_cond, want_match),
            (ArmRel::NeqK, ArmRel::EqK) => emit_cond_truth(cx, res, raw_cond, !want_match),
            _ => None,
        };
        if let Some(folded) = out {
            return Some(folded);
        }
    }
    None
}

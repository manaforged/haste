use crate::cse::flip_int_cc;
use crate::fold::{extend, known_bits, non_null_at};
use crate::ir::{Binary, Convert, Function, Inst, IntCC, Op, Type, Value};

use super::icmp::{ADDR_SPAN, before, emit_cond_truth, null_test, offset_bound, three_way_non_zero};
use super::RewriteCx;

pub(super) fn simplify_select(cx: &mut RewriteCx, res: Option<Value>, cond: Value, t: Value, f: Value) -> Option<Op> {
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
    let (t, f) = (refuted_arm(cx, inner, t, true).unwrap_or(t), refuted_arm(cx, inner, f, false).unwrap_or(f));
    if t == f {
        cx.alias(res, t);
        return None;
    }
    if let Some(op) = simplify_unit_bool_select(cx, res, inner, t, f) {
        return op;
    }
    fold_ord_select_cmp(cx, inner, t, f).or_else(|| (inner != cond || (t, f) != given).then_some(Op::Select(inner, t, f)))
}

fn simplify_unit_bool_select(cx: &mut RewriteCx, res: Option<Value>, inner: Value, t: Value, f: Value) -> Option<Option<Op>> {
    if !res.is_some_and(|r| cx.func.ty(r) == Type::I8) || cx.func.ty(inner) != Type::I8 || !cx.is_unit_bool(inner) {
        return None;
    }
    match (cx.uconst(t, Type::I8), cx.uconst(f, Type::I8)) {
        (Some(1), Some(0)) => { cx.alias(res, inner); Some(None) }
        (Some(1), _) if cx.is_unit_bool(f) => Some(Some(Op::Binary(Binary::Bor, inner, f))),
        (_, Some(0)) if cx.is_unit_bool(t) => Some(Some(Op::Binary(Binary::Band, inner, t))),
        (Some(0), Some(1)) => Some(Some(Op::Binary(Binary::Bxor, inner, f))),
        _ => None,
    }
}

fn peel_uext_icmp(cx: &RewriteCx, ty: Type, v: Value) -> Option<(IntCC, Value, Value)> {
    let target = if ty == Type::I8 { v } else if let Some(&Op::Convert(Convert::Uextend, i)) = cx.op(v) { i } else { return None };
    if cx.func.ty(target) != Type::I8 { return None; }
    let &Op::Icmp(cc, a, b) = cx.op(target)? else { return None };
    cx.func.ty(a).is_small_int().then_some((cc, a, b))
}

fn peel_base_icmp(cx: &RewriteCx, ty: Type, v: Value) -> Option<(Option<Value>, IntCC, Value, Value)> {
    if let Some((cc, a, b)) = peel_uext_icmp(cx, ty, v) { return Some((None, cc, a, b)); }
    let &Op::Binary(Binary::Iadd, lhs, rhs) = cx.op(v)? else { return None };
    [(lhs, rhs), (rhs, lhs)].into_iter().find_map(|(base, cv)| peel_uext_icmp(cx, ty, cv).map(|(cc, a, b)| (Some(base), cc, a, b)))
}

fn peel_base_add_01(cx: &RewriteCx, ty: Type, base_opt: Option<Value>, v: Value) -> Option<u64> {
    let Some(base) = base_opt else { return cx.uconst(v, ty).filter(|&k| k <= 1) };
    if v == base { return Some(0); }
    let &Op::Binary(Binary::Iadd, p, q) = cx.op(v)? else { return None };
    [(p, q), (q, p)].into_iter().find_map(|(b, k)| (b == base).then(|| cx.uconst(k, ty).filter(|&u| u <= 1)).flatten())
}

fn collapse_ord_cc(cc: IntCC, eq_add: u64) -> Option<IntCC> {
    match (cc, eq_add) {
        (IntCC::SignedLessThan | IntCC::SignedLessThanOrEqual, 1) => Some(IntCC::SignedLessThanOrEqual),
        (IntCC::UnsignedLessThan | IntCC::UnsignedLessThanOrEqual, 1) => Some(IntCC::UnsignedLessThanOrEqual),
        (IntCC::SignedGreaterThan | IntCC::SignedGreaterThanOrEqual, 1) => Some(IntCC::SignedGreaterThanOrEqual),
        (IntCC::UnsignedGreaterThan | IntCC::UnsignedGreaterThanOrEqual, 1) => Some(IntCC::UnsignedGreaterThanOrEqual),
        (IntCC::SignedLessThan | IntCC::SignedLessThanOrEqual, 0) => Some(IntCC::SignedLessThan),
        (IntCC::UnsignedLessThan | IntCC::UnsignedLessThanOrEqual, 0) => Some(IntCC::UnsignedLessThan),
        (IntCC::SignedGreaterThan | IntCC::SignedGreaterThanOrEqual, 0) => Some(IntCC::SignedGreaterThan),
        (IntCC::UnsignedGreaterThan | IntCC::UnsignedGreaterThanOrEqual, 0) => Some(IntCC::UnsignedGreaterThan),
        _ => None,
    }
}

fn fold_ord_select_cmp(cx: &mut RewriteCx, cond: Value, t: Value, f: Value) -> Option<Op> {
    let ty = cx.func.ty(t);
    let &Op::Icmp(cond_cc @ (IntCC::Equal | IntCC::NotEqual), x, y) = cx.op(cond)? else { return None };
    if !ty.is_small_int() || !cx.func.ty(x).is_small_int() { return None; }
    let (eq_arm, ne_arm) = if cond_cc == IntCC::Equal { (t, f) } else { (f, t) };
    let (base_opt, cmp_cc, cx0, cy0) = peel_base_icmp(cx, ty, ne_arm)?;
    if !((cx0 == x && cy0 == y) || (cx0 == y && cy0 == x)) { return None; }
    let new_cc = collapse_ord_cc(cmp_cc, peel_base_add_01(cx, ty, base_opt, eq_arm)?)?;
    let cmp_op = Op::Icmp(new_cc, cx0, cy0);
    if base_opt.is_none() && ty == Type::I8 { return Some(cmp_op); }
    let cmp_val = cx.emit(Type::I8, cmp_op);
    let Some(base) = base_opt else { return Some(Op::Convert(Convert::Uextend, cmp_val)) };
    let uext = if ty == Type::I8 { cmp_val } else { cx.emit(ty, Op::Convert(Convert::Uextend, cmp_val)) };
    Some(Op::Binary(Binary::Iadd, base, uext))
}

fn opposite(cx: &RewriteCx, a: Value, b: Value) -> bool {
    let (Some(&Op::Icmp(ca, xa, ya)), Some(&Op::Icmp(cb, xb, yb))) = (cx.op(a), cx.op(b)) else { return false };
    cx.func.ty(xa).is_small_int() && ((xa == xb && ya == yb && cb == ca.inverse()) || (xa == yb && ya == xb && cb == flip_int_cc(ca.inverse())))
}

fn refuted_arm(cx: &RewriteCx, cond: Value, arm: Value, taken: bool) -> Option<Value> {
    let &Op::Select(test, x, y) = cx.op(arm)? else { return None };
    opposite(cx, cond, cx.unwrap_uextend(test)).then_some(if taken { y } else { x })
}

pub(super) fn tested_select(cx: &RewriteCx, cond: Value) -> Option<(Value, bool)> {
    let &Op::Select(test, t, f) = cx.op(cx.unwrap_uextend(cond))? else { return None };
    let ty = cx.func.ty(t);
    if !ty.is_small_int() || !cx.func.ty(test).is_small_int() {
        return None;
    }
    if cx.uconst(f, ty) == Some(0) && is_non_zero_under(cx, t, unwrap_cond_cmp(cx, test, true), 0) {
        Some((test, true))
    } else if cx.uconst(t, ty) == Some(0) && is_non_zero_under(cx, f, unwrap_cond_cmp(cx, test, false), 0) {
        Some((test, false))
    } else {
        None
    }
}

fn unwrap_cond_cmp(cx: &RewriteCx, mut raw: Value, mut taken: bool) -> Option<(IntCC, Value, Value)> {
    for _ in 0..4 {
        raw = cx.unwrap_uextend(raw);
        if let Some(&Op::Binary(Binary::Bxor, a, b)) = cx.op(raw)
            && cx.func.ty(raw).is_small_int()
        {
            let next = [(a, b), (b, a)].into_iter().find_map(|(inner, mask)| {
                (cx.uconst(mask, cx.func.ty(mask)) == Some(1) && cx.is_unit_bool(inner)).then_some(inner)
            });
            if let Some(inner) = next {
                (raw, taken) = (inner, !taken);
                continue;
            }
        }
        break;
    }
    let &Op::Icmp(cc, a, b) = cx.op(raw)? else { return None };
    cx.func.ty(a).is_small_int().then_some((if taken { cc } else { cc.inverse() }, a, b))
}

fn cond_upper_bound(cx: &RewriteCx, (eff, a, b): (IntCC, Value, Value)) -> Option<(Value, u64)> {
    let ty = cx.func.ty(a);
    let (w, sign) = (ty.unsigned_max(), 1_u64 << (ty.bits() - 1));
    let non_neg = |v: Value| (cx.known(v).0 & sign) != 0;
    for (norm, mut var, lim) in [(eff, a, b), (flip_int_cc(eff), b, a)] {
        if cx.sconst(var).is_some() {
            continue;
        }
        let lim_hi = w & !cx.known(lim).0;
        let both_nn = non_neg(var) && non_neg(lim);
        let bound = match norm {
            IntCC::UnsignedLessThan => lim_hi.checked_sub(1),
            IntCC::UnsignedLessThanOrEqual | IntCC::Equal => Some(lim_hi),
            IntCC::SignedLessThan if both_nn => lim_hi.checked_sub(1),
            IntCC::SignedLessThanOrEqual if both_nn => Some(lim_hi),
            _ => None,
        };
        if let Some(mut hi) = bound {
            while let Some(&Op::Convert(Convert::Uextend, inner)) = cx.op(var) {
                if !cx.func.ty(inner).is_small_int() {
                    break;
                }
                var = inner;
                hi = hi.min(cx.func.ty(var).unsigned_max());
            }
            return Some((var, hi));
        }
    }
    None
}

fn cond_proves_nonzero(cx: &RewriteCx, (eff, a, b): (IntCC, Value, Value), v: Value) -> bool {
    let ty = cx.func.ty(a);
    let sign = 1_u64 << (ty.bits() - 1);
    [(eff, a, b), (flip_int_cc(eff), b, a)].into_iter().any(|(norm, var, lim)| {
        let same = var == v || matches!(cx.op(var), Some(&Op::Convert(Convert::Uextend, inner)) if inner == v);
        if !same {
            return false;
        }
        match norm {
            IntCC::UnsignedGreaterThan => true,
            IntCC::NotEqual => cx.uconst(lim, ty) == Some(0),
            IntCC::Equal | IntCC::UnsignedGreaterThanOrEqual => cx.known(lim).1 != 0,
            IntCC::SignedGreaterThan => (cx.known(var).0 & sign) != 0 && (cx.known(lim).0 & sign) != 0,
            _ => false,
        }
    })
}

fn peel_const_base(cx: &RewriteCx, mut v: Value) -> (Value, i64) {
    let mut disp = 0_i64;
    for _ in 0..4 {
        match cx.op(v) {
            Some(&Op::Convert(Convert::Bitcast, a)) if cx.func.ty(a) == cx.func.ty(v) => v = a,
            Some(&Op::Binary(Binary::Iadd, a, b)) => {
                let step = [(a, b), (b, a)].into_iter().find_map(|(base, k)| Some((base, disp.checked_add(cx.sconst(k)?)?)));
                let Some((base, next)) = step else { break };
                (v, disp) = (base, next);
            }
            Some(&Op::Binary(Binary::Isub, a, b)) => {
                let Some(next) = cx.sconst(b).and_then(i64::checked_neg).and_then(|c| disp.checked_add(c)) else { break };
                (v, disp) = (a, next);
            }
            _ => break,
        }
    }
    (v, disp)
}

fn deref_span_covers(cx: &RewriteCx, root: Value, lo_disp: i64, hi_disp: i64) -> bool {
    if cx.func.ty(root) != Type::I64 {
        return false;
    }
    cx.out.iter().rev().take(64).any(|&inst| {
        let data = &cx.func.insts[inst.index()];
        let (addr, off, bytes) = match data.op {
            Op::Load(addr, off) => (addr, off, data.result.map_or(0, |r| cx.func.ty(r).bytes())),
            Op::Store(val, addr, off) => (addr, off, cx.func.ty(val).bytes()),
            _ => return false,
        };
        let (aroot, adisp) = peel_const_base(cx, addr);
        bytes != 0
            && aroot == root
            && adisp
                .checked_add(i64::from(off))
                .and_then(|lo| Some((lo, lo.checked_add(i64::from(bytes))?)))
                .is_some_and(|(lo, hi)| lo <= lo_disp && hi_disp < hi)
    })
}

fn valid_ptr_span(cx: &RewriteCx, root: Value, base_disp: i64, off_max: u64) -> bool {
    let Ok(off_s) = i64::try_from(off_max) else { return false };
    let Some(hi_disp) = base_disp.checked_add(off_s) else { return false };
    match cx.op(root) {
        Some(Op::StackAddr(..) | Op::SlotAddr(_)) => (-ADDR_SPAN..=ADDR_SPAN).contains(&base_disp) && (-ADDR_SPAN..=ADDR_SPAN).contains(&hi_disp),
        Some(Op::Symbol(_)) => base_disp >= 0 && hi_disp <= ADDR_SPAN,
        _ => cx.func.ty(root) == Type::I64
            && ((base_disp >= 0 && (hi_disp as u64) < u64::from(entry_deref_bytes(cx, root)))
                || deref_span_covers(cx, root, base_disp, hi_disp)),
    }
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
        Op::Binary(Binary::Isub | Binary::Bxor, a, b) if cx.func.ty(cond) == ty => ([key(a), key(b)], true),
        Op::Binary(Binary::Iadd, a, b) if cx.func.ty(cond) == ty => {
            let (v, k) = [(a, b), (b, a)].into_iter().find_map(|(v, k)| Some((v, cx.uconst(k, ty)?)))?;
            ([key(v), Err(k.wrapping_neg() & ty.unsigned_max())], true)
        }
        _ => return None,
    };
    let arms = [key(t), key(f)];
    (pair == arms || pair == [arms[1], arms[0]]).then_some(if differ { t } else { f })
}

fn entry_deref_bytes(cx: &RewriteCx, v: Value) -> u32 {
    let Some(entry) = cx.func.entry else { return 0 };
    cx.func.blocks[entry.index()]
        .params
        .iter()
        .position(|&p| p == v)
        .and_then(|pos| cx.func.signature.params.get(pos))
        .map_or(0, |abi| abi.deref_bytes)
}

pub(super) fn is_non_zero(cx: &RewriteCx, v: Value, depth: u8) -> bool {
    is_non_zero_under(cx, v, None, depth)
}

fn add_non_zero(cx: &RewriteCx, (a, b): (Value, Value), outer_disp: i64, cmp: Option<(IntCC, Value, Value)>) -> bool {
    let bound = cmp.and_then(|c| cond_upper_bound(cx, c));
    [(a, b), (b, a)].into_iter().any(|(base, off_v)| {
        let (root, base_disp) = peel_const_base(cx, base);
        base_disp
            .checked_add(outer_disp)
            .is_some_and(|disp| valid_ptr_span(cx, root, disp, offset_bound(cx.func, &|x| cx.def(x), &|x| cx.sconst(x), off_v, (bound, 0))))
    })
}

fn known_non_zero(cx: &RewriteCx, v: Value, cmp: Option<(IntCC, Value, Value)>) -> bool {
    non_null_at(cx.func, &|x| cx.def(x), &|x| cx.sconst(x), before(cx), v)
        || cx.known(v).1 != 0
        || entry_deref_bytes(cx, v) > 0
        || three_way_non_zero(cx, v)
        || cmp.is_some_and(|c| cond_proves_nonzero(cx, c, v))
}

fn is_non_zero_under(cx: &RewriteCx, v: Value, cmp: Option<(IntCC, Value, Value)>, depth: u8) -> bool {
    let ty = cx.func.ty(v);
    if depth >= 4 || !ty.is_small_int() {
        return false;
    }
    if known_non_zero(cx, v, cmp) {
        return true;
    }
    let (core, outer_disp) = peel_const_base(cx, v);
    if outer_disp != 0 && valid_ptr_span(cx, core, outer_disp, 0) {
        return true;
    }
    let Some(inst) = cx.def(core) else { return false };
    match cx.func.insts[inst.index()].op {
        Op::Binary(Binary::Iadd, a, b) => add_non_zero(cx, (a, b), outer_disp, cmp),
        Op::Convert(Convert::Uextend | Convert::Sextend, a) if outer_disp == 0 && cx.func.ty(a).is_small_int() => is_non_zero_under(cx, a, cmp, depth + 1),
        Op::Select(_, a, b) if outer_disp == 0 => is_non_zero_under(cx, a, cmp, depth + 1) && is_non_zero_under(cx, b, cmp, depth + 1),
        _ => false,
    }
}

fn bool_cond(cx: &RewriteCx, raw: Value) -> Option<Value> {
    let c = cx.unwrap_uextend(raw);
    (cx.func.ty(c) == Type::I8 && cx.is_unit_bool(c)).then_some(c)
}

fn simplify_1_bool(cx: &mut RewriteCx, res: Option<Value>, c: Value, b1: bool, b2: bool) -> Option<Op> {
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

fn eval_const_select_cmp(cx: &mut RewriteCx, res: Option<Value>, cc: IntCC, x: Value, y: Value) -> Option<Option<Op>> {
    let ty = cx.func.ty(x);
    for (norm, sel_val, c_val) in [(cc, x, y), (flip_int_cc(cc), y, x)] {
        let Some(c) = cx.sconst(c_val) else { continue };
        let Some(sel_inst) = cx.def(sel_val) else { continue };
        let Op::Select(raw_c1, t1, f1) = cx.func.insts[sel_inst.index()].op else { continue };
        let Some(c1) = bool_cond(cx, raw_c1) else { continue };
        let eval = |k: i64| crate::fold::holds(norm, ty, k, c);
        if let (Some(k1), Some(k2)) = (cx.sconst(t1), cx.sconst(f1)) {
            return Some(simplify_1_bool(cx, res, c1, eval(k1), eval(k2)));
        }
        if let (Some(k1), Some(f_inst)) = (cx.sconst(t1), cx.def(f1))
            && let Op::Select(raw_c2, t2, f2) = cx.func.insts[f_inst.index()].op
            && let (Some(k2), Some(k3)) = (cx.sconst(t2), cx.sconst(f2))
        {
            let (b1, b2, b3) = (eval(k1), eval(k2), eval(k3));
            if b2 == b3 { return Some(simplify_1_bool(cx, res, c1, b1, b2)); }
            let Some(c2) = bool_cond(cx, raw_c2) else { continue };
            return Some(Some(match (b1, b2, b3) {
                (false, false, true) => { let or = cx.emit(Type::I8, Op::Binary(Binary::Bor, c1, c2)); let one = cx.emit(Type::I8, Op::Const(1)); Op::Binary(Binary::Bxor, or, one) }
                (true, true, false) => Op::Binary(Binary::Bor, c1, c2),
                (false, true, false) => { let nc1 = emit_bool_not(cx, c1); Op::Binary(Binary::Band, nc1, c2) }
                _ => { let nc2 = emit_bool_not(cx, c2); Op::Binary(Binary::Bor, c1, nc2) }
            }));
        }
        if let (Some(t_inst), Some(k3)) = (cx.def(t1), cx.sconst(f1))
            && let Op::Select(raw_c2, t2, f2) = cx.func.insts[t_inst.index()].op
            && let (Some(k1), Some(k2)) = (cx.sconst(t2), cx.sconst(f2))
        {
            let (b1, b2, b3) = (eval(k1), eval(k2), eval(k3));
            if b1 == b2 { return Some(simplify_1_bool(cx, res, c1, b1, b3)); }
            let Some(c2) = bool_cond(cx, raw_c2) else { continue };
            return Some(Some(match (b1, b2, b3) {
                (true, false, false) => Op::Binary(Binary::Band, c1, c2),
                (false, true, false) => { let nc2 = emit_bool_not(cx, c2); Op::Binary(Binary::Band, c1, nc2) }
                (true, false, true) => { let nc1 = emit_bool_not(cx, c1); Op::Binary(Binary::Bor, nc1, c2) }
                _ => { let and = cx.emit(Type::I8, Op::Binary(Binary::Band, c1, c2)); let one = cx.emit(Type::I8, Op::Const(1)); Op::Binary(Binary::Bxor, and, one) }
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

fn classify_arm(cx: &RewriteCx, v: Value, ty: Type, k_u: u64, cmp: Option<(IntCC, Value, Value)>) -> ArmRel {
    if let Some(c) = cx.uconst(v, ty) {
        return if c == k_u { ArmRel::EqK } else { ArmRel::NeqK };
    }
    let (zeros, ones) = cx.known(v);
    if (zeros | ones) == ty.unsigned_max() {
        return if ones == k_u { ArmRel::EqK } else { ArmRel::NeqK };
    }
    if (ones & !k_u) != 0 || (zeros & k_u) != 0 {
        return ArmRel::NeqK;
    }
    if k_u == 0 && is_non_zero_under(cx, v, cmp, 0) {
        return ArmRel::NeqK;
    }
    ArmRel::Unknown
}

pub(super) fn eval_select_zero_cmp(cx: &mut RewriteCx, res: Option<Value>, cc: IntCC, x: Value, y: Value) -> Option<Option<Op>> {
    let ty = cx.func.ty(x);
    if !ty.is_small_int() || !res.is_some_and(|r| cx.func.ty(r) == Type::I8) {
        return None;
    }
    if let Some(folded) = eval_const_select_cmp(cx, res, cc, x, y) {
        return Some(folded);
    }
    for (norm, sel_val, c_val) in [(cc, x, y), (flip_int_cc(cc), y, x)] {
        let Some((k_u, want_match)) = cx.sconst(c_val).and_then(|k_s| cmp_target(norm, ty, k_s)) else { continue };
        let Some(sel_inst) = cx.def(sel_val) else { continue };
        let Op::Select(raw_cond, t_val, f_val) = cx.func.insts[sel_inst.index()].op else { continue };
        let t_cmp = (k_u == 0).then(|| unwrap_cond_cmp(cx, raw_cond, true)).flatten();
        let f_cmp = (k_u == 0).then(|| unwrap_cond_cmp(cx, raw_cond, false)).flatten();
        let out = match (classify_arm(cx, t_val, ty, k_u, t_cmp), classify_arm(cx, f_val, ty, k_u, f_cmp)) {
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

struct BoolFacts {
    compares: Vec<Option<(IntCC, Value, Value)>>,
    consts: Vec<Option<i64>>,
    xors: Vec<Option<(Value, Value)>>,
    uextends: Vec<Option<Value>>,
    bools: Vec<bool>,
}

impl BoolFacts {
    fn collect(func: &Function) -> Self {
        let n = func.types.len();
        let mut f = Self { compares: vec![None; n], consts: vec![None; n], xors: vec![None; n], uextends: vec![None; n], bools: vec![false; n] };
        for inst in func.blocks.iter().flat_map(|block| &block.insts) {
            let d = &func.insts[inst.index()];
            match (&d.op, d.result) {
                (&Op::Icmp(cc, x, y), Some(r)) if func.ty(r) == Type::I8 => { f.compares[r.index()] = Some((cc, x, y)); f.bools[r.index()] = true; }
                (&Op::Fcmp(..) | &Op::Fits(..), Some(r)) if func.ty(r) == Type::I8 => f.bools[r.index()] = true,
                (&Op::Const(imm), Some(r)) if func.ty(r).is_small_int() => f.consts[r.index()] = Some(imm),
                (&Op::Binary(Binary::Bxor, x, y), Some(r)) => f.xors[r.index()] = Some((x, y)),
                (&Op::Convert(Convert::Uextend, inner), Some(r)) => f.uextends[r.index()] = Some(inner),
                _ => {}
            }
        }
        f.propagate(func);
        f
    }

    fn propagate(&mut self, func: &Function) {
        let is_01 = |v: Value, bools: &[bool], consts: &[Option<i64>]| bools[v.index()] || matches!(consts[v.index()], Some(0 | 1));
        for _ in 0..2 {
            for inst in func.blocks.iter().flat_map(|block| &block.insts) {
                let d = &func.insts[inst.index()];
                let Some(r) = d.result else { continue };
                self.bools[r.index()] |= match d.op {
                    Op::Binary(Binary::Band, x, y) => matches!(self.consts[x.index()], Some(0 | 1)) || matches!(self.consts[y.index()], Some(0 | 1)) || (is_01(x, &self.bools, &self.consts) && is_01(y, &self.bools, &self.consts)),
                    Op::Binary(Binary::Bor | Binary::Bxor, x, y) | Op::Select(_, x, y) => is_01(x, &self.bools, &self.consts) && is_01(y, &self.bools, &self.consts),
                    Op::Convert(Convert::Uextend, inner) => self.bools[inner.index()],
                    _ => false,
                };
            }
        }
    }

    fn unwrap_uext(&self, mut v: Value) -> Value {
        while let Some(inner) = self.uextends.get(v.index()).copied().flatten() { v = inner; }
        v
    }

    fn xor_one_src(&self, x: Value, y: Value) -> Option<Value> {
        if self.consts[y.index()] == Some(1) && self.bools[x.index()] { Some(x) }
        else if self.consts[x.index()] == Some(1) && self.bools[y.index()] { Some(y) }
        else { None }
    }
}

fn negated(op: &Op, facts: &BoolFacts) -> Option<(Value, Option<Value>)> {
    let is = |v: Value, want: i64| facts.consts[v.index()] == Some(want);
    match *op {
        Op::Binary(Binary::Bxor, x, y) | Op::Icmp(IntCC::NotEqual, x, y) if is(y, 1) => Some((x, Some(y))),
        Op::Binary(Binary::Bxor, y, x) | Op::Icmp(IntCC::NotEqual, y, x) if is(y, 1) => Some((x, Some(y))),
        Op::Icmp(IntCC::Equal, x, y) if is(y, 0) => Some((x, None)),
        Op::Icmp(IntCC::Equal, y, x) if is(y, 0) => Some((x, None)),
        Op::Select(c, x, y) if is(x, 0) && is(y, 1) && facts.bools[c.index()] => Some((c, Some(y))),
        _ => None,
    }
}

fn invert_brif(op: &mut Op, facts: &BoolFacts, types: &[Type]) {
    let Op::Brif(cond, then, then_args, otherwise, else_args) = op else { return };
    if let Some((x, y)) = facts.xors[cond.index()] && let Some(inner) = facts.xor_one_src(x, y) {
        *cond = inner; std::mem::swap(then, otherwise); std::mem::swap(then_args, else_args); return;
    }
    let Some((cc, x, y)) = facts.compares[cond.index()] else { return };
    let (val, flipped) = if facts.consts[y.index()] == Some(0) && types[x.index()].is_small_int() { (x, false) }
        else if facts.consts[x.index()] == Some(0) && types[y.index()].is_small_int() { (y, true) } else { return };
    let invert_arms = matches!((flipped, cc), (_, IntCC::Equal) | (false, IntCC::UnsignedLessThanOrEqual) | (true, IntCC::UnsignedGreaterThanOrEqual));
    if invert_arms || matches!((flipped, cc), (_, IntCC::NotEqual) | (false, IntCC::UnsignedGreaterThan) | (true, IntCC::UnsignedLessThan)) {
        *cond = val;
        if invert_arms { std::mem::swap(then, otherwise); std::mem::swap(then_args, else_args); }
    }
}

fn invert_inst(op: &mut Op, res: Option<Value>, facts: &BoolFacts, types: &[Type]) {
    if let (Some(res), Some((inner, opt_one))) = (res, negated(op, facts)) {
        let res_ty = types[res.index()];
        if res_ty == Type::I8 && let Some((cc, a, b)) = facts.compares[inner.index()] { *op = Op::Icmp(cc.inverse(), a, b); return; }
        if facts.bools[inner.index()] && let Some(one) = opt_one && types[inner.index()] == res_ty && types[one.index()] == res_ty && !matches!(*op, Op::Binary(Binary::Bxor, ..)) {
            *op = Op::Binary(Binary::Bxor, inner, one); return;
        }
    }
    invert_brif(op, facts, types);
}

/// Rewrites a negation of a compare, `x ^ 1`, `x == 0`, `x != 1`, or `select(x, 0, 1)`, as the inverse compare or `x ^ 1`, and simplifies branches on `x == 0` / `x != 0`.
pub(crate) fn invert(func: &mut Function) {
    let facts = BoolFacts::collect(func);
    for inst in func.blocks.iter().flat_map(|block| &block.insts) {
        let res = func.insts[inst.index()].result;
        invert_inst(&mut func.insts[inst.index()].op, res, &facts, &func.types);
    }
    crate::schedule::sink_compares(func);
}

fn xor_ne_pair(func: &Function, facts: &BoolFacts, x: Value, y: Value, allow_const: bool) -> Option<(Value, Value)> {
    if allow_const && let Some(pair) = [(x, y), (y, x)].into_iter().find(|&(_, c)| facts.consts[c.index()].is_some()) {
        return Some(pair);
    }
    if facts.bools[x.index()] && facts.bools[y.index()] {
        let (a, b) = (facts.unwrap_uext(x), facts.unwrap_uext(y));
        return (func.ty(a) == func.ty(b) && func.ty(a).is_small_int()).then_some((a, b));
    }
    None
}

fn fold_bxor_select(func: &mut Function, facts: &BoolFacts, out: &mut Vec<Inst>, inst: Inst) -> bool {
    let Op::Select(cond, t_val, f_val) = func.insts[inst.index()].op else { return false };
    let inner = facts.unwrap_uext(cond);
    if let Some((x, y)) = facts.xors[inner.index()] {
        if let Some((lhs, _)) = [(x, y), (y, x)].into_iter().find(|&(_, c)| facts.consts[c.index()] == Some(1))
            && let bool_src = facts.unwrap_uext(lhs) && facts.bools[bool_src.index()]
        {
            func.insts[inst.index()].op = Op::Select(bool_src, f_val, t_val);
            return true;
        }
        if let Some((a, b)) = xor_ne_pair(func, facts, x, y, false) {
            let cmp_val = func.value(Type::I8);
            out.push(func.add(Op::Icmp(IntCC::NotEqual, a, b), Some(cmp_val)));
            func.insts[inst.index()].op = Op::Select(cmp_val, t_val, f_val);
            return true;
        }
    }
    if inner != cond { func.insts[inst.index()].op = Op::Select(inner, t_val, f_val); return true; }
    false
}

fn fold_bxor_brif(func: &mut Function, facts: &BoolFacts, out: &mut Vec<Inst>, inst: Inst) -> bool {
    let Op::Brif(cond, ..) = func.insts[inst.index()].op else { return false };
    let inner = facts.unwrap_uext(cond);
    if let Some((x, y)) = facts.xors[inner.index()]
        && let Some((a, b)) = xor_ne_pair(func, facts, x, y, true) && func.ty(a) == func.ty(b) && func.ty(a).is_small_int()
    {
        let cmp_val = func.value(Type::I8);
        out.push(func.add(Op::Icmp(IntCC::NotEqual, a, b), Some(cmp_val)));
        if let Op::Brif(c_mut, ..) = &mut func.insts[inst.index()].op { *c_mut = cmp_val; }
        return true;
    }
    if inner != cond && let Op::Brif(c_mut, ..) = &mut func.insts[inst.index()].op { *c_mut = inner; return true; }
    false
}

fn tag_cond_info(facts: &BoolFacts, func: &Function, param_of: &[Option<(usize, usize)>], b_idx: usize, cond: Value) -> Option<(usize, Value, u64, bool)> {
    let inner = facts.unwrap_uext(cond);
    if let Some((cc @ (IntCC::Equal | IntCC::NotEqual), x, y)) = facts.compares[inner.index()] {
        return [(x, y), (y, x)].into_iter().find_map(|(p, c)| {
            let (pb, ti) = param_of.get(p.index()).copied().flatten()?;
            let imm = facts.consts[c.index()]?;
            (pb == b_idx && func.ty(p).is_small_int()).then(|| (ti, p, extend(imm, func.ty(p), false) as u64, cc == IntCC::Equal))
        });
    }
    let (pb, ti) = param_of.get(inner.index()).copied().flatten()?;
    (pb == b_idx && func.ty(inner).is_small_int()).then_some((ti, inner, 0, false))
}

fn push_edge(edges: &mut Vec<(Value, Value)>, args: &[Value], (tag_idx, val_idx): (usize, usize)) -> bool {
    let (Some(&ta), Some(&va)) = (args.get(tag_idx), args.get(val_idx)) else { return false };
    edges.push((ta, va));
    true
}

fn push_edge_pair(edges: &mut Vec<(Value, Value)>, b_idx: usize, op: &Op, idx: (usize, usize)) -> bool {
    match op {
        Op::Jump(t, args) if t.index() == b_idx => push_edge(edges, args, idx),
        Op::Brif(_, tb, ta, eb, ea) => (tb.index() != b_idx || push_edge(edges, ta, idx)) && (eb.index() != b_idx || push_edge(edges, ea, idx)),
        Op::TryCall { normal, handler, .. } => normal.index() != b_idx && handler.index() != b_idx,
        Op::BrTable(_, d, ts) => d.index() != b_idx && ts.iter().all(|t| t.index() != b_idx),
        _ => true,
    }
}

fn incoming_pair_edges(func: &Function, b_idx: usize, self_last: Option<Inst>, tag_idx: usize, val_idx: usize) -> Option<Vec<(Value, Value)>> {
    let mut edges = Vec::new();
    for (pb, block) in func.blocks.iter().enumerate() {
        let Some(last) = (if pb == b_idx { self_last } else { block.insts.last().copied() }) else { continue };
        if !push_edge_pair(&mut edges, b_idx, &func.insts[last.index()].op, (tag_idx, val_idx)) { return None; }
    }
    (edges.len() >= 2).then_some(edges)
}

fn edge_matches_correlated(func: &Function, facts: &BoolFacts, defs: &[Option<Inst>], (tag_ty, val_ty, match_u, k_u, val_takes_eq): (Type, Type, u64, u64, bool), (tag_arg, val_arg): (Value, Value)) -> bool {
    let tag_const = facts.consts[tag_arg.index()].map(|c| extend(c, tag_ty, false) as u64);
    let proven_val_arm = if val_takes_eq {
        tag_const == Some(match_u)
    } else if tag_const.is_some_and(|u| u != match_u) {
        true
    } else {
        let (kz, ko) = known_bits(func, &|v| defs.get(v.index()).copied().flatten(), &|v| facts.consts.get(v.index()).copied().flatten(), tag_arg, 0);
        let mask = tag_ty.unsigned_max() as u64;
        ((ko & mask) & !match_u) != 0 || ((kz & mask) & match_u) != 0
    };
    proven_val_arm || (tag_const.is_some_and(|u| (u == match_u) == !val_takes_eq) && facts.consts[val_arg.index()].is_some_and(|vc| extend(vc, val_ty, false) as u64 == k_u))
}

fn correlated_select_target(func: &Function, facts: &BoolFacts, param_of: &[Option<(usize, usize)>], defs: &mut Option<Vec<Option<Inst>>>, (b_idx, self_last, cond, t, f): (usize, Option<Inst>, Value, Value, Value)) -> Option<Value> {
    if Some(b_idx) == func.entry.map(|e| e.index()) { return None; }
    let (tag_idx, tag_phi, match_u, eq_means_true) = tag_cond_info(facts, func, param_of, b_idx, cond)?;
    let (val_phi, val_idx, k_val, val_on_true) = [(t, f, true), (f, t, false)].into_iter().find_map(|(v, k, on_t)| {
        let (pb, vi) = param_of.get(v.index()).copied().flatten()?;
        facts.consts[k.index()]?;
        (pb == b_idx && vi != tag_idx).then_some((v, vi, k, on_t))
    })?;
    let val_ty = func.ty(val_phi);
    if !val_ty.is_small_int() { return None; }
    let k_u = extend(facts.consts[k_val.index()]?, val_ty, false) as u64;
    let edges = incoming_pair_edges(func, b_idx, self_last, tag_idx, val_idx)?;
    let defs_ref = defs.get_or_insert_with(|| crate::fold::defs(func));
    let spec = (func.ty(tag_phi), val_ty, match_u, k_u, val_on_true == eq_means_true);
    edges.iter().all(|&edge| edge_matches_correlated(func, facts, defs_ref, spec, edge)).then_some(val_phi)
}

fn fold_correlated_select(func: &Function, facts: &BoolFacts, param_of: &[Option<(usize, usize)>], (defs, subst): (&mut Option<Vec<Option<Inst>>>, &mut Vec<Value>), (b_idx, self_last, inst): (usize, Option<Inst>, Inst)) -> bool {
    let Op::Select(c, t, f) = func.insts[inst.index()].op else { return false };
    let Some(res) = func.insts[inst.index()].result else { return false };
    let (c, t, f) = if subst.is_empty() { (c, t, f) } else { (crate::merge::find(subst, c), crate::merge::find(subst, t), crate::merge::find(subst, f)) };
    let Some(target) = correlated_select_target(func, facts, param_of, defs, (b_idx, self_last, c, t, f)) else { return false };
    if subst.is_empty() { *subst = (0..func.types.len()).map(|i| Value(i as u32)).collect(); }
    subst[res.index()] = crate::merge::find(subst, target);
    true
}

fn fold_shifted_select(func: &mut Function, facts: &BoolFacts, seen: &mut Vec<(Value, Type, Value, u64, u64)>, out: &mut Vec<Inst>, inst: Inst) -> bool {
    let Op::Select(cond, t_val, f_val) = func.insts[inst.index()].op else { return false };
    let Some(res) = func.insts[inst.index()].result else { return false };
    let ty = func.ty(res);
    if !ty.is_small_int() { return false; }
    let (Some(ca), Some(cb)) = (facts.consts.get(t_val.index()).copied().flatten(), facts.consts.get(f_val.index()).copied().flatten()) else { return false };
    let (a_u, b_u) = (extend(ca, ty, false) as u64, extend(cb, ty, false) as u64);
    if a_u == b_u { return false; }
    let w = ty.unsigned_max();
    if let Some((base_sel, shift)) = seen.iter().find_map(|&(c, st, r1, a1, b1)| {
        (c == cond && st == ty).then(|| (1_u32..=4).find(|&k| ((a1 << k) & w) == a_u && ((b1 << k) & w) == b_u).map(|k| (r1, k))).flatten()
    }) {
        let shift_val = func.value(ty);
        out.push(func.add(Op::Const(i64::from(shift)), Some(shift_val)));
        func.insts[inst.index()].op = Op::Binary(Binary::Ishl, base_sel, shift_val);
        return true;
    }
    seen.push((cond, ty, res, a_u, b_u));
    false
}

/// Folds `x ^ 1` / `x ^ y` in `select` and `brif` conditions, eliminates correlated join-param selects, and sinks single-use compares toward their branch.
pub(crate) fn fold_bxor_conds(func: &mut Function) -> bool {
    let facts = BoolFacts::collect(func);
    let mut param_of = vec![None; func.types.len()];
    for (b_idx, block) in func.blocks.iter().enumerate() {
        if Some(b_idx) != func.entry.map(|e| e.index()) { block.params.iter().enumerate().for_each(|(p_idx, &p)| param_of[p.index()] = Some((b_idx, p_idx))); }
    }
    let (mut changed, mut defs, mut subst) = (false, None, Vec::new());
    for b_idx in 0..func.blocks.len() {
        let insts = std::mem::take(&mut func.blocks[b_idx].insts);
        let self_last = insts.last().copied();
        let (mut out, mut seen_sels) = (Vec::with_capacity(insts.len()), Vec::new());
        for inst in insts {
            changed |= fold_bxor_select(func, &facts, &mut out, inst) || fold_bxor_brif(func, &facts, &mut out, inst);
            if !fold_correlated_select(func, &facts, &param_of, (&mut defs, &mut subst), (b_idx, self_last, inst)) {
                changed |= fold_shifted_select(func, &facts, &mut seen_sels, &mut out, inst);
                out.push(inst);
            }
        }
        func.blocks[b_idx].insts = out;
    }
    if !subst.is_empty() {
        func.blocks.iter().flat_map(|b| &b.insts).for_each(|inst| func.insts[inst.index()].op.map_operands(|v| crate::merge::find(&subst, v)));
        changed = true;
    }
    if changed { crate::schedule::sink_compares(func); }
    changed
}

fn difference(func: &Function, defs: &[Option<Inst>], uses: &[u32], v: Value) -> Option<(Inst, Value, Value)> {
    let inst = defs.get(v.index()).copied().flatten().filter(|_| uses.get(v.index()) == Some(&1))?;
    match func.insts[inst.index()].op {
        Op::Binary(Binary::Isub | Binary::Bxor, a, b) if func.ty(a).is_small_int() => Some((inst, a, b)),
        _ => None,
    }
}

fn offset_compare(func: &mut Function, defs: &[Option<Inst>], uses: &[u32], out: &mut Vec<Inst>, inst: Inst) -> Option<Inst> {
    let konst = |func: &Function, v: Value| match defs.get(v.index()).copied().flatten().map(|def| &func.insts[def.index()].op) {
        Some(&Op::Const(imm)) => Some(imm),
        _ => None,
    };
    let (cc, diff, rhs) = match func.insts[inst.index()].op {
        Op::Brif(cond, ..) | Op::Select(cond, ..) => (None, cond, 0),
        Op::Icmp(cc @ (IntCC::Equal | IntCC::NotEqual), x, y) => match (konst(func, x), konst(func, y)) {
            (None, Some(k)) => (Some(cc), x, k),
            (Some(k), None) => (Some(cc), y, k),
            _ => return None,
        },
        _ => return None,
    };
    let (gone, a, b) = difference(func, defs, uses, diff)?;
    let ty = func.ty(a);
    let (lhs, bound) = if crate::fold::extend(rhs, ty, false) == 0 {
        (a, b)
    } else {
        let sub = matches!(func.insts[gone.index()].op, Op::Binary(Binary::Isub, ..));
        let (base, imm) = match (konst(func, a), konst(func, b), sub) {
            (None, Some(c), true) => (a, c.wrapping_add(rhs)),
            (Some(c), None, true) => (b, c.wrapping_sub(rhs)),
            (None, Some(c), false) => (a, c ^ rhs),
            (Some(c), None, false) => (b, c ^ rhs),
            _ => return None,
        };
        let val = func.value(ty);
        out.push(func.add(Op::Const(crate::fold::extend(imm, ty, true)), Some(val)));
        (base, val)
    };
    match cc {
        Some(cc) => func.insts[inst.index()].op = Op::Icmp(cc, lhs, bound),
        None => {
            let test = func.value(Type::I8);
            out.push(func.add(Op::Icmp(IntCC::NotEqual, lhs, bound), Some(test)));
            if let Op::Brif(cond, ..) | Op::Select(cond, ..) = &mut func.insts[inst.index()].op {
                *cond = test;
            }
        }
    }
    Some(gone)
}

/// Rewrites a branch, a select, or an equality with a constant that is the only reader of `a - b` or `a ^ b` as a compare of `a`, and drops the difference; it runs after the passes that read ranges from the difference.
pub(crate) fn compare_offsets(func: &mut Function) -> bool {
    let (defs, uses) = (crate::fold::defs(func), func.use_counts());
    let mut dead = Vec::new();
    for b in 0..func.blocks.len() {
        let insts = std::mem::take(&mut func.blocks[b].insts);
        let mut out = Vec::with_capacity(insts.len());
        for inst in insts {
            dead.extend(offset_compare(func, &defs, &uses, &mut out, inst));
            out.push(inst);
        }
        func.blocks[b].insts = out;
    }
    for block in &mut func.blocks {
        block.insts.retain(|inst| !dead.contains(inst));
    }
    for &inst in &dead {
        func.insts[inst.index()].op = Op::Trap;
    }
    !dead.is_empty()
}

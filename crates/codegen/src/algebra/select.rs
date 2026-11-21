use crate::cse::flip_int_cc;
use crate::fold::{extend, non_null_at};
use crate::ir::{Binary, Convert, Function, Inst, IntCC, Op, Type, Value};

use super::icmp::{ADDR_SPAN, before, complementary_eq_pair, emit_cond_truth, null_test, offset_bound, three_way_non_zero};
use super::{RewriteCx, const_u128, op_of, peel_uext};

/// `v` narrowed to `dst` for free: the `dst` value it extends, or the constant it holds at that width.
pub(super) fn narrow_arm(cx: &RewriteCx, dst: Type, v: Value) -> Option<Result<Value, i64>> {
    if let Some(&Op::Convert(Convert::Uextend | Convert::Sextend, inner)) = cx.op(v)
        && cx.func.ty(inner) == dst
    {
        return Some(Ok(inner));
    }
    cx.sconst(v).map(|c| Err(extend(c, dst, true)))
}

pub(super) fn simplify_select(cx: &mut RewriteCx, res: Option<Value>, cond: Value, t: Value, f: Value) -> Option<Op> {
    let given = (t, f);
    if t == f { cx.alias(res, t); return None; }
    if is_non_zero(cx, cond, 0) { cx.alias(res, t); return None; }
    let inner = cx.unwrap_uextend(cond);
    if let Some(arm) = tested_arm(cx, inner, t, f) { cx.alias(res, arm); return None; }
    let (t, f) = (refuted_arm(cx, inner, t, true).unwrap_or(t), refuted_arm(cx, inner, f, false).unwrap_or(f));
    if t == f { cx.alias(res, t); return None; }
    if let Some(op) = simplify_unit_bool_select(cx, res, inner, t, f) { return op; }
    fold_ord_select_cmp(cx, inner, t, f).or_else(|| (inner != cond || (t, f) != given).then_some(Op::Select(inner, t, f)))
}

fn simplify_unit_bool_select(cx: &mut RewriteCx, res: Option<Value>, inner: Value, t: Value, f: Value) -> Option<Option<Op>> {
    if !res.is_some_and(|r| cx.func.ty(r) == Type::I8) || cx.func.ty(inner) != Type::I8 || !cx.is_unit_bool(inner) { return None; }
    match (cx.uconst(t, Type::I8), cx.uconst(f, Type::I8)) {
        (Some(1), Some(0)) => { cx.alias(res, inner); Some(None) }
        (Some(1), _) if cx.is_unit_bool(f) => Some(Some(Op::Binary(Binary::Bor, inner, f))),
        (_, Some(0)) if cx.is_unit_bool(t) => Some(Some(Op::Binary(Binary::Band, inner, t))),
        (Some(0), Some(1)) => Some(Some(Op::Binary(Binary::Bxor, inner, f))),
        (Some(0), _) if cx.is_unit_bool(f) => band_without(cx, inner, f).map(Some),
        _ => None,
    }
}

fn band_without(cx: &mut RewriteCx, cond: Value, kept: Value) -> Option<Op> {
    let &Op::Binary(Binary::Band, x, y) = cx.op(cond)? else { return None };
    let other = if x == kept { y } else if y == kept { x } else { return None };
    if cx.func.ty(other) != Type::I8 || !cx.is_unit_bool(other) { return None; }
    let not = emit_bool_not(cx, other);
    Some(Op::Binary(Binary::Band, kept, not))
}

pub(super) struct OrdSelectMatch {
    base: Option<Value>,
    cc: IntCC,
    lhs: Value,
    rhs: Value,
}

fn peel_uext_icmp(func: &Function, def_of: &dyn Fn(Value) -> Option<Inst>, v: Value) -> Option<(IntCC, Value, Value)> {
    let target = peel_uext(func, def_of, v);
    if func.ty(target) != Type::I8 { return None; }
    let &Op::Icmp(cc, a, b) = op_of(func, def_of, target)? else { return None };
    (!func.ty(a).in_float_register()).then_some((cc, a, b))
}

fn peel_base_icmp(func: &Function, def_of: &dyn Fn(Value) -> Option<Inst>, v: Value) -> Option<(Option<Value>, IntCC, Value, Value)> {
    if let Some((cc, a, b)) = peel_uext_icmp(func, def_of, v) { return Some((None, cc, a, b)); }
    let &Op::Binary(Binary::Iadd, lhs, rhs) = op_of(func, def_of, v)? else { return None };
    [(lhs, rhs), (rhs, lhs)].into_iter().find_map(|(base, cv)| peel_uext_icmp(func, def_of, cv).map(|(cc, a, b)| (Some(base), cc, a, b)))
}

fn peel_base_add_01(func: &Function, def_of: &dyn Fn(Value) -> Option<Inst>, ty: Type, base_opt: Option<Value>, v: Value) -> Option<u64> {
    let c01 = |u: Value| const_u128(func, def_of, ty, u).and_then(|x| u64::try_from(x).ok()).filter(|&x| x <= 1);
    let Some(base) = base_opt else { return c01(v) };
    if v == base { return Some(0); }
    let &Op::Binary(Binary::Iadd, p, q) = op_of(func, def_of, v)? else { return None };
    [(p, q), (q, p)].into_iter().find_map(|(b, k)| (b == base).then(|| c01(k)).flatten())
}

fn collapse_ord_cc(cc: IntCC, eq_add: u64) -> Option<IntCC> {
    if eq_add > 1 { return None; }
    let le = eq_add == 1;
    Some(match cc {
        IntCC::SignedLessThan | IntCC::SignedLessThanOrEqual => if le { IntCC::SignedLessThanOrEqual } else { IntCC::SignedLessThan },
        IntCC::UnsignedLessThan | IntCC::UnsignedLessThanOrEqual => if le { IntCC::UnsignedLessThanOrEqual } else { IntCC::UnsignedLessThan },
        IntCC::SignedGreaterThan | IntCC::SignedGreaterThanOrEqual => if le { IntCC::SignedGreaterThanOrEqual } else { IntCC::SignedGreaterThan },
        IntCC::UnsignedGreaterThan | IntCC::UnsignedGreaterThanOrEqual => if le { IntCC::UnsignedGreaterThanOrEqual } else { IntCC::UnsignedGreaterThan },
        _ => return None,
    })
}

fn cond_eq_ab(func: &Function, def_of: &dyn Fn(Value) -> Option<Inst>, cond: Value) -> Option<(bool, Value, Value)> {
    match *op_of(func, def_of, peel_uext(func, def_of, cond))? {
        Op::Icmp(cc @ (IntCC::Equal | IntCC::NotEqual), x, y) => complementary_eq_pair(func, def_of, x, y)
            .map(|(a, b)| (cc == IntCC::Equal, a, b))
            .or_else(|| (!func.ty(x).in_float_register()).then_some((cc == IntCC::Equal, x, y))),
        Op::Binary(Binary::Isub | Binary::Bxor, p, q) => complementary_eq_pair(func, def_of, p, q).map(|(a, b)| (false, a, b)),
        _ => None,
    }
}

pub(super) fn match_ord_select(func: &Function, def_of: &dyn Fn(Value) -> Option<Inst>, (cond, t, f): (Value, Value, Value)) -> Option<OrdSelectMatch> {
    let ty = func.ty(t);
    if ty.in_float_register() { return None; }
    let (eq_on_true, x, y) = cond_eq_ab(func, def_of, cond)?;
    let (eq_arm, ne_arm) = if eq_on_true { (t, f) } else { (f, t) };
    let (base, cmp_cc, lhs, rhs) = peel_base_icmp(func, def_of, ne_arm)?;
    if !((lhs == x && rhs == y) || (lhs == y && rhs == x)) { return None; }
    let cc = collapse_ord_cc(cmp_cc, peel_base_add_01(func, def_of, ty, base, eq_arm)?)?;
    Some(OrdSelectMatch { base, cc, lhs, rhs })
}

pub(super) fn build_ord_select(ty: Type, m: OrdSelectMatch, emit: &mut dyn FnMut(Type, Op) -> Value) -> Op {
    let cmp_op = Op::Icmp(m.cc, m.lhs, m.rhs);
    if m.base.is_none() && ty == Type::I8 { return cmp_op; }
    let cmp_val = emit(Type::I8, cmp_op);
    let Some(base) = m.base else { return Op::Convert(Convert::Uextend, cmp_val) };
    let uext = if ty == Type::I8 { cmp_val } else { emit(ty, Op::Convert(Convert::Uextend, cmp_val)) };
    Op::Binary(Binary::Iadd, base, uext)
}

fn fold_ord_select_cmp(cx: &mut RewriteCx, cond: Value, t: Value, f: Value) -> Option<Op> {
    let ty = cx.func.ty(t);
    let m = match_ord_select(cx.func, &|v| cx.def(v), (cond, t, f))?;
    Some(build_ord_select(ty, m, &mut |uty, op| cx.emit(uty, op)))
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
    if !ty.is_small_int() || !cx.func.ty(test).is_small_int() { return None; }
    if cx.uconst(f, ty) == Some(0) && is_non_zero_under(cx, t, unwrap_cond_cmp(cx, test, true), 0) { Some((test, true)) }
    else if cx.uconst(t, ty) == Some(0) && is_non_zero_under(cx, f, unwrap_cond_cmp(cx, test, false), 0) { Some((test, false)) }
    else { None }
}

fn unwrap_cond_cmp(cx: &RewriteCx, mut raw: Value, mut taken: bool) -> Option<(IntCC, Value, Value)> {
    for _ in 0..4 {
        raw = cx.unwrap_uextend(raw);
        if let Some(&Op::Binary(Binary::Bxor, a, b)) = cx.op(raw) && cx.func.ty(raw).is_small_int() {
            let next = [(a, b), (b, a)].into_iter().find_map(|(inner, mask)| (cx.uconst(mask, cx.func.ty(mask)) == Some(1) && cx.is_unit_bool(inner)).then_some(inner));
            if let Some(inner) = next { (raw, taken) = (inner, !taken); continue; }
        }
        break;
    }
    let &Op::Icmp(cc, a, b) = cx.op(raw)? else { return None };
    cx.func.ty(a).is_small_int().then_some((if taken { cc } else { cc.inverse() }, a, b))
}

fn cond_upper_bound(cx: &RewriteCx, (eff, a, b): (IntCC, Value, Value)) -> Option<(Value, u64)> {
    let (ty, sign) = (cx.func.ty(a), 1_u64 << (cx.func.ty(a).bits() - 1));
    let non_neg = |v: Value| (cx.known(v).0 & sign) != 0;
    for (norm, mut var, lim) in [(eff, a, b), (flip_int_cc(eff), b, a)] {
        if cx.sconst(var).is_some() { continue; }
        let (lim_hi, both_nn) = (ty.unsigned_max() & !cx.known(lim).0, non_neg(var) && non_neg(lim));
        let bound = match norm {
            IntCC::UnsignedLessThan => lim_hi.checked_sub(1),
            IntCC::UnsignedLessThanOrEqual | IntCC::Equal => Some(lim_hi),
            IntCC::SignedLessThan if both_nn => lim_hi.checked_sub(1),
            IntCC::SignedLessThanOrEqual if both_nn => Some(lim_hi),
            _ => None,
        };
        if let Some(mut hi) = bound {
            while let Some(&Op::Convert(Convert::Uextend, inner)) = cx.op(var) && cx.func.ty(inner).is_small_int() {
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
        same && match norm {
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
                let Some((base, next)) = [(a, b), (b, a)].into_iter().find_map(|(base, k)| Some((base, disp.checked_add(cx.sconst(k)?)?))) else { break };
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
    cx.func.ty(root) == Type::I64 && cx.out.iter().rev().take(64).any(|&inst| {
        let data = &cx.func.insts[inst.index()];
        let (addr, off, bytes) = match data.op {
            Op::Load(addr, off) => (addr, off, data.result.map_or(0, |r| cx.func.ty(r).bytes())),
            Op::Store(val, addr, off) => (addr, off, cx.func.ty(val).bytes()),
            _ => return false,
        };
        let (aroot, adisp) = peel_const_base(cx, addr);
        bytes != 0 && aroot == root && adisp.checked_add(i64::from(off)).and_then(|lo| Some((lo, lo.checked_add(i64::from(bytes))?))).is_some_and(|(lo, hi)| lo <= lo_disp && hi_disp < hi)
    })
}

fn valid_ptr_span(cx: &RewriteCx, root: Value, base_disp: i64, off_max: u64) -> bool {
    let Some(hi_disp) = i64::try_from(off_max).ok().and_then(|o| base_disp.checked_add(o)) else { return false };
    match cx.op(root) {
        Some(Op::StackAddr(..) | Op::SlotAddr(_)) => (-ADDR_SPAN..=ADDR_SPAN).contains(&base_disp) && (-ADDR_SPAN..=ADDR_SPAN).contains(&hi_disp),
        Some(Op::Symbol(_)) => base_disp >= 0 && hi_disp <= ADDR_SPAN,
        _ => cx.func.ty(root) == Type::I64 && ((base_disp >= 0 && (hi_disp as u64) < u64::from(entry_deref_bytes(cx, root))) || deref_span_covers(cx, root, base_disp, hi_disp)),
    }
}

fn tested_arm(cx: &RewriteCx, cond: Value, t: Value, f: Value) -> Option<Value> {
    let ty = cx.func.ty(t);
    if !ty.is_small_int() { return None; }
    let key = |v: Value| cx.uconst(v, ty).map_or(Ok(v), Err);
    let (pair, differ) = match *cx.op(cond)? {
        Op::Icmp(cc @ (IntCC::Equal | IntCC::NotEqual), a, b) if cx.func.ty(a) == ty => ([key(a), key(b)], cc == IntCC::NotEqual),
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
    cx.func.blocks[entry.index()].params.iter().position(|&p| p == v).and_then(|pos| cx.func.signature.params.get(pos)).map_or(0, |abi| abi.deref_bytes)
}

pub(super) fn is_non_zero(cx: &RewriteCx, v: Value, depth: u8) -> bool { is_non_zero_under(cx, v, None, depth) }

fn add_non_zero(cx: &RewriteCx, (a, b): (Value, Value), outer_disp: i64, cmp: Option<(IntCC, Value, Value)>) -> bool {
    let bound = cmp.and_then(|c| cond_upper_bound(cx, c));
    [(a, b), (b, a)].into_iter().any(|(base, off_v)| {
        let (root, base_disp) = peel_const_base(cx, base);
        base_disp.checked_add(outer_disp).is_some_and(|disp| valid_ptr_span(cx, root, disp, offset_bound(cx.func, &|x| cx.def(x), &|x| cx.sconst(x), off_v, (bound, 0))))
    })
}

fn known_non_zero(cx: &RewriteCx, v: Value, cmp: Option<(IntCC, Value, Value)>) -> bool {
    non_null_at(cx.func, &|x| cx.def(x), &|x| cx.sconst(x), before(cx), v)
        || cx.known(v).1 != 0 || entry_deref_bytes(cx, v) > 0 || three_way_non_zero(cx, v) || cmp.is_some_and(|c| cond_proves_nonzero(cx, c, v))
}

fn is_non_zero_under(cx: &RewriteCx, v: Value, cmp: Option<(IntCC, Value, Value)>, depth: u8) -> bool {
    let ty = cx.func.ty(v);
    if depth >= 4 || !ty.is_small_int() { return false; }
    if known_non_zero(cx, v, cmp) { return true; }
    let (core, outer_disp) = peel_const_base(cx, v);
    if outer_disp != 0 && valid_ptr_span(cx, core, outer_disp, 0) { return true; }
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
        (true, false) => { cx.alias(res, c); None }
        (false, true) => { let one = cx.emit(Type::I8, Op::Const(1)); Some(Op::Binary(Binary::Bxor, c, one)) }
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

fn eval_const_select_cmp(cx: &mut RewriteCx, res: Option<Value>, cc: IntCC, x: Value, y: Value) -> Option<Option<Op>> {
    let ty = cx.func.ty(x);
    for (norm, sel_val, c_val) in [(cc, x, y), (flip_int_cc(cc), y, x)] {
        let (Some(c), Some(&Op::Select(raw_c1, t1, f1))) = (cx.sconst(c_val), cx.op(sel_val)) else { continue };
        let Some(c1) = bool_cond(cx, raw_c1) else { continue };
        let eval = |k: i64| crate::fold::holds(norm, ty, k, c);
        if let (Some(k1), Some(k2)) = (cx.sconst(t1), cx.sconst(f1)) { return Some(simplify_1_bool(cx, res, c1, eval(k1), eval(k2))); }
        if let (Some(k1), Some(&Op::Select(raw_c2, t2, f2))) = (cx.sconst(t1), cx.op(f1))
            && let (Some(k2), Some(k3)) = (cx.sconst(t2), cx.sconst(f2))
        {
            let (b1, b2, b3) = (eval(k1), eval(k2), eval(k3));
            if b2 == b3 { return Some(simplify_1_bool(cx, res, c1, b1, b2)); }
            let Some(c2) = bool_cond(cx, raw_c2) else { continue };
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
            if b1 == b2 { return Some(simplify_1_bool(cx, res, c1, b1, b3)); }
            let Some(c2) = bool_cond(cx, raw_c2) else { continue };
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
enum ArmRel { EqK, NeqK, Unknown }

fn cmp_target(cc: IntCC, ty: Type, k_s: i64) -> Option<(u64, bool)> {
    match cc {
        IntCC::Equal | IntCC::NotEqual => Some((extend(k_s, ty, false) as u64, cc == IntCC::Equal)),
        _ => null_test(cc, ty, k_s).map(|bit| (0, !bit)),
    }
}

fn classify_arm(cx: &RewriteCx, v: Value, ty: Type, k_u: u64, cmp: Option<(IntCC, Value, Value)>) -> ArmRel {
    if let Some(c) = cx.uconst(v, ty) { return if c == k_u { ArmRel::EqK } else { ArmRel::NeqK }; }
    let (zeros, ones) = cx.known(v);
    if (zeros | ones) == ty.unsigned_max() { return if ones == k_u { ArmRel::EqK } else { ArmRel::NeqK }; }
    if (ones & !k_u) != 0 || (zeros & k_u) != 0 || (k_u == 0 && is_non_zero_under(cx, v, cmp, 0)) { return ArmRel::NeqK; }
    ArmRel::Unknown
}

pub(super) fn eval_select_zero_cmp(cx: &mut RewriteCx, res: Option<Value>, cc: IntCC, x: Value, y: Value) -> Option<Option<Op>> {
    let ty = cx.func.ty(x);
    if !ty.is_small_int() || !res.is_some_and(|r| cx.func.ty(r) == Type::I8) { return None; }
    if let Some(folded) = eval_const_select_cmp(cx, res, cc, x, y) { return Some(folded); }
    for (norm, sel_val, c_val) in [(cc, x, y), (flip_int_cc(cc), y, x)] {
        let Some((k_u, want_match)) = cx.sconst(c_val).and_then(|k_s| cmp_target(norm, ty, k_s)) else { continue };
        let Some(&Op::Select(raw_cond, t_val, f_val)) = cx.op(sel_val) else { continue };
        let t_cmp = (k_u == 0).then(|| unwrap_cond_cmp(cx, raw_cond, true)).flatten();
        let f_cmp = (k_u == 0).then(|| unwrap_cond_cmp(cx, raw_cond, false)).flatten();
        let out = match (classify_arm(cx, t_val, ty, k_u, t_cmp), classify_arm(cx, f_val, ty, k_u, f_cmp)) {
            (ArmRel::EqK, ArmRel::EqK) => Some(Some(Op::Const(i64::from(want_match)))),
            (ArmRel::NeqK, ArmRel::NeqK) => Some(Some(Op::Const(i64::from(!want_match)))),
            (ArmRel::EqK, ArmRel::NeqK) => emit_cond_truth(cx, res, raw_cond, want_match),
            (ArmRel::NeqK, ArmRel::EqK) => emit_cond_truth(cx, res, raw_cond, !want_match),
            _ => None,
        };
        if let Some(folded) = out { return Some(folded); }
    }
    None
}

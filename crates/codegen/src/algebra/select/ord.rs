use crate::ir::{Binary, Convert, Function, Inst, IntCC, Op, Type, Value};

use super::super::icmp::complementary_eq_pair;
use super::super::{RewriteCx, const_u128, op_of, peel_uext};

pub(in super::super) struct OrdSelectMatch {
    base: Option<Value>,
    cc: IntCC,
    lhs: Value,
    rhs: Value,
}

fn peel_uext_icmp(
    func: &Function,
    def_of: &dyn Fn(Value) -> Option<Inst>,
    v: Value,
) -> Option<(IntCC, Value, Value)> {
    let target = peel_uext(func, def_of, v);
    if func.ty(target) != Type::I8 {
        return None;
    }
    let &Op::Icmp(cc, a, b) = op_of(func, def_of, target)? else {
        return None;
    };
    (!func.ty(a).in_float_register()).then_some((cc, a, b))
}

fn peel_base_icmp(
    func: &Function,
    def_of: &dyn Fn(Value) -> Option<Inst>,
    v: Value,
) -> Option<(Option<Value>, IntCC, Value, Value)> {
    if let Some((cc, a, b)) = peel_uext_icmp(func, def_of, v) {
        return Some((None, cc, a, b));
    }
    let &Op::Binary(Binary::Iadd, lhs, rhs) = op_of(func, def_of, v)? else {
        return None;
    };
    [(lhs, rhs), (rhs, lhs)].into_iter().find_map(|(base, cv)| {
        peel_uext_icmp(func, def_of, cv).map(|(cc, a, b)| (Some(base), cc, a, b))
    })
}

fn peel_base_add_01(
    func: &Function,
    def_of: &dyn Fn(Value) -> Option<Inst>,
    ty: Type,
    base_opt: Option<Value>,
    v: Value,
) -> Option<u64> {
    let c01 = |u: Value| {
        const_u128(func, def_of, ty, u)
            .and_then(|x| u64::try_from(x).ok())
            .filter(|&x| x <= 1)
    };
    let Some(base) = base_opt else {
        return c01(v);
    };
    if v == base {
        return Some(0);
    }
    let &Op::Binary(Binary::Iadd, p, q) = op_of(func, def_of, v)? else {
        return None;
    };
    [(p, q), (q, p)]
        .into_iter()
        .find_map(|(b, k)| (b == base).then(|| c01(k)).flatten())
}

fn collapse_ord_cc(cc: IntCC, eq_add: u64) -> Option<IntCC> {
    if eq_add > 1 {
        return None;
    }
    let le = eq_add == 1;
    Some(match cc {
        IntCC::SignedLessThan | IntCC::SignedLessThanOrEqual => {
            if le {
                IntCC::SignedLessThanOrEqual
            } else {
                IntCC::SignedLessThan
            }
        }
        IntCC::UnsignedLessThan | IntCC::UnsignedLessThanOrEqual => {
            if le {
                IntCC::UnsignedLessThanOrEqual
            } else {
                IntCC::UnsignedLessThan
            }
        }
        IntCC::SignedGreaterThan | IntCC::SignedGreaterThanOrEqual => {
            if le {
                IntCC::SignedGreaterThanOrEqual
            } else {
                IntCC::SignedGreaterThan
            }
        }
        IntCC::UnsignedGreaterThan | IntCC::UnsignedGreaterThanOrEqual => {
            if le {
                IntCC::UnsignedGreaterThanOrEqual
            } else {
                IntCC::UnsignedGreaterThan
            }
        }
        _ => return None,
    })
}

fn cond_eq_ab(
    func: &Function,
    def_of: &dyn Fn(Value) -> Option<Inst>,
    cond: Value,
) -> Option<(bool, Value, Value)> {
    match *op_of(func, def_of, peel_uext(func, def_of, cond))? {
        Op::Icmp(cc @ (IntCC::Equal | IntCC::NotEqual), x, y) => {
            complementary_eq_pair(func, def_of, x, y)
                .map(|(a, b)| (cc == IntCC::Equal, a, b))
                .or_else(|| (!func.ty(x).in_float_register()).then_some((cc == IntCC::Equal, x, y)))
        }
        Op::Binary(Binary::Isub | Binary::Bxor, p, q) => {
            complementary_eq_pair(func, def_of, p, q).map(|(a, b)| (false, a, b))
        }
        _ => None,
    }
}

pub(in super::super) fn match_ord_select(
    func: &Function,
    def_of: &dyn Fn(Value) -> Option<Inst>,
    (cond, t, f): (Value, Value, Value),
) -> Option<OrdSelectMatch> {
    let ty = func.ty(t);
    if ty.in_float_register() {
        return None;
    }
    let (eq_on_true, x, y) = cond_eq_ab(func, def_of, cond)?;
    let (eq_arm, ne_arm) = if eq_on_true { (t, f) } else { (f, t) };
    let (base, cmp_cc, lhs, rhs) = peel_base_icmp(func, def_of, ne_arm)?;
    if !((lhs == x && rhs == y) || (lhs == y && rhs == x)) {
        return None;
    }
    let cc = collapse_ord_cc(cmp_cc, peel_base_add_01(func, def_of, ty, base, eq_arm)?)?;
    Some(OrdSelectMatch { base, cc, lhs, rhs })
}

pub(in super::super) fn build_ord_select(
    ty: Type,
    m: OrdSelectMatch,
    emit: &mut dyn FnMut(Type, Op) -> Value,
) -> Op {
    let cmp_op = Op::Icmp(m.cc, m.lhs, m.rhs);
    if m.base.is_none() && ty == Type::I8 {
        return cmp_op;
    }
    let cmp_val = emit(Type::I8, cmp_op);
    let Some(base) = m.base else {
        return Op::Convert(Convert::Uextend, cmp_val);
    };
    let uext = if ty == Type::I8 {
        cmp_val
    } else {
        emit(ty, Op::Convert(Convert::Uextend, cmp_val))
    };
    Op::Binary(Binary::Iadd, base, uext)
}

pub(super) fn fold_ord_select_cmp(
    cx: &mut RewriteCx,
    cond: Value,
    t: Value,
    f: Value,
) -> Option<Op> {
    let ty = cx.func.ty(t);
    let m = match_ord_select(cx.func, &|v| cx.def(v), (cond, t, f))?;
    Some(build_ord_select(ty, m, &mut |uty, op| cx.emit(uty, op)))
}

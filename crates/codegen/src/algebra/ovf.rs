use super::RewriteCx;
use crate::fold::extend;
use crate::ir::{Binary, IntCC, Op, Type, Value};

fn scaled(cx: &RewriteCx, x: Value, y: Value) -> Option<(Value, u64)> {
    let ty = cx.func.ty(x);
    if !ty.is_small_int() {
        return None;
    }
    [(x, y), (y, x)]
        .into_iter()
        .find_map(|(n, c)| cx.uconst(c, ty).filter(|&c| c != 0).map(|c| (n, c)))
}

fn upper(cx: &RewriteCx, cc: IntCC, limit: Value, ty: Type) -> Option<u64> {
    let limit = cx.uconst(limit, ty)?;
    match cc {
        IntCC::UnsignedLessThanOrEqual => Some(limit),
        IntCC::UnsignedLessThan => limit.checked_sub(1),
        _ => None,
    }
}

fn fits_bound(cx: &RewriteCx, cond: Value, (n, c): (Value, u64)) -> Option<u64> {
    let ty = cx.func.ty(n);
    match *cx.op(cond)? {
        Op::Fits(Binary::Imul, false, x, y) => {
            (scaled(cx, x, y)? == (n, c)).then(|| ty.unsigned_max() / c)
        }
        Op::Icmp(cc, x, limit) if x == n => {
            upper(cx, cc, limit, ty).filter(|&m| m <= ty.unsigned_max() / c)
        }
        _ => None,
    }
}

fn at_most(cx: &mut RewriteCx, n: Value, bound: u64) -> Op {
    let ty = cx.func.ty(n);
    let limit = cx.emit(ty, Op::Const(extend(bound as i64, ty, true)));
    Op::Icmp(IntCC::UnsignedLessThanOrEqual, n, limit)
}

/// Unsigned `x * c` fits for a nonzero constant `c` exactly when the other factor is at most `MAX / c`.
pub(super) fn simplify_mul_fits(cx: &mut RewriteCx, x: Value, y: Value) -> Option<Op> {
    let (n, c) = scaled(cx, x, y)?;
    let ty = cx.func.ty(n);
    if c == 1 {
        return Some(Op::Const(1));
    }
    Some(at_most(cx, n, ty.unsigned_max() / c))
}

/// `fits(n * c) ? (n * c <= l ? a : z) : z` as `n <= l / c ? a : z`, the bound that also proves the product fits.
pub(super) fn gated_product(cx: &mut RewriteCx, cond: Value, inner: Value, z: Value) -> Option<Op> {
    let &Op::Select(test, a, other) = cx.op(inner)? else {
        return None;
    };
    let same = other == z
        || (cx.func.ty(other) == cx.func.ty(z)
            && cx.sconst(other).is_some_and(|v| cx.sconst(z) == Some(v)));
    let &Op::Icmp(cc, product, limit) = cx.op(test)? else {
        return None;
    };
    let &Op::Binary(Binary::Imul, px, py) = cx.op(product)? else {
        return None;
    };
    let (n, c) = scaled(cx, px, py)?;
    let bound = upper(cx, cc, limit, cx.func.ty(n))? / c;
    if !same || fits_bound(cx, cond, (n, c))? < bound {
        return None;
    }
    let test = at_most(cx, n, bound);
    let tight = cx.emit(Type::I8, test);
    Some(Op::Select(tight, a, z))
}

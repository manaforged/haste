use crate::ir::{Binary, Convert, IntCC, Op, Type, Value};

use super::{RewriteCx, const_u128};

/// The value `v` widened or narrowed to a 64-bit integer.
fn word(cx: &mut RewriteCx, v: Value) -> Value {
    match cx.func.ty(v).bits() {
        64 => v,
        bits if bits > 64 => cx.emit(Type::I64, Op::Convert(Convert::Ireduce, v)),
        _ => cx.emit(Type::I64, Op::Convert(Convert::Uextend, v)),
    }
}

/// Whether no instruction placed after `x`'s definition in the block so far writes memory or calls.
fn calm_since(cx: &RewriteCx, x: Value) -> bool {
    let Some(def) = cx.def(x) else { return false };
    let Some(at) = cx.out.iter().position(|&inst| inst == def) else { return false };
    cx.out[at + 1..].iter().all(|inst| cx.func.insts[inst.index()].op.is_removable())
}

/// Tests a variable bit `k` of a 128-bit load in the 64-bit word that holds it: `load128(p) & (1 << k) != 0` becomes `load64(p + (k >> 6 & 1) * 8) & (1 << (k & 63)) != 0`.
pub(super) fn narrow_bit_test(cx: &mut RewriteCx, cc: IntCC, x: Value, y: Value) -> Option<Op> {
    let cc = match cc {
        IntCC::Equal | IntCC::NotEqual => cc,
        IntCC::UnsignedGreaterThan => IntCC::NotEqual,
        _ => return None,
    };
    let wide = |v: Value| const_u128(cx.func, |u| cx.def(u), Type::I128, v);
    if cx.func.ty(x) != Type::I128 || wide(y) != Some(0) || !cx.single_use(x) {
        return None;
    }
    let Some(&Op::Binary(Binary::Band, a, b)) = cx.op(x) else { return None };
    let (loaded, mask) = [(a, b), (b, a)].into_iter().find(|&(l, m)| matches!(cx.op(l), Some(Op::Load(..))) && matches!(cx.op(m), Some(&Op::Binary(Binary::Ishl, one, _)) if wide(one) == Some(1)))?;
    let (Some(&Op::Load(addr, off)), Some(&Op::Binary(Binary::Ishl, _, k))) = (cx.op(loaded), cx.op(mask)) else { return None };
    if !cx.single_use(loaded) || !cx.single_use(mask) || !calm_since(cx, loaded) || cx.sconst(k).is_some() {
        return None;
    }
    let k = word(cx, k);
    let six = cx.emit(Type::I64, Op::Const(6));
    let half = cx.emit(Type::I64, Op::Binary(Binary::Ushr, k, six));
    let one = cx.emit(Type::I64, Op::Const(1));
    let half = cx.emit(Type::I64, Op::Binary(Binary::Band, half, one));
    let three = cx.emit(Type::I64, Op::Const(3));
    let bytes = cx.emit(Type::I64, Op::Binary(Binary::Ishl, half, three));
    let at = cx.emit(cx.func.ty(addr), Op::Binary(Binary::Iadd, addr, bytes));
    let bits = cx.emit(Type::I64, Op::Load(at, off));
    let low = cx.emit(Type::I64, Op::Const(63));
    let shift = cx.emit(Type::I64, Op::Binary(Binary::Band, k, low));
    let bit = cx.emit(Type::I64, Op::Binary(Binary::Ishl, one, shift));
    let hit = cx.emit(Type::I64, Op::Binary(Binary::Band, bits, bit));
    let zero = cx.emit(Type::I64, Op::Const(0));
    Some(Op::Icmp(cc, hit, zero))
}

/// Keeps the low bits of `x` directly: `x - ((x >> k) << k)` and `x - (x >> k) * 2^k` become `x & (2^k - 1)`.
pub(super) fn low_bits(cx: &mut RewriteCx, x: Value, y: Value) -> Option<Op> {
    let ty = cx.func.ty(x);
    let bits = ty.bytes() * 8;
    let (q, k) = match cx.func.insts[cx.def(y)?.index()].op {
        Op::Binary(Binary::Ishl, q, k) => (q, cx.uconst(k, ty)?),
        Op::Binary(Binary::Imul, q, m) => (q, cx.uconst(m, ty).filter(|m| m.is_power_of_two())?.trailing_zeros().into()),
        _ => return None,
    };
    let Op::Binary(Binary::Ushr, inner, shift) = cx.func.insts[cx.def(q)?.index()].op else { return None };
    if inner != x || cx.uconst(shift, ty)? != k || k == 0 || k >= u64::from(bits) {
        return None;
    }
    let mask = cx.emit(ty, Op::Const(((1_u64 << k) - 1) as i64));
    Some(Op::Binary(Binary::Band, x, mask))
}

/// Tests a variable bit `k` by shifting the word instead of building a mask: `w & (1 << k) != 0` and `(w & (1 << k)) == (1 << k)` become `(w >> k) & 1 != 0`.
pub(super) fn shifted_bit_test(cx: &mut RewriteCx, cc: IntCC, x: Value, y: Value) -> Option<Op> {
    let ty = cx.func.ty(x);
    if !matches!(cc, IntCC::Equal | IntCC::NotEqual) || !ty.is_small_int() {
        return None;
    }
    let is_bit = |m: Value| matches!(cx.op(m), Some(&Op::Binary(Binary::Ishl, one, k)) if cx.uconst(one, ty) == Some(1) && cx.sconst(k).is_none());
    let (word, k, out_cc, zero) = if cx.uconst(y, ty) == Some(0) && cx.single_use(x)
        && let Some(&Op::Binary(Binary::Band, a, b)) = cx.op(x)
        && let Some((w, mask)) = [(a, b), (b, a)].into_iter().find(|&(_, m)| is_bit(m) && cx.single_use(m))
        && let Some(&Op::Binary(Binary::Ishl, _, k)) = cx.op(mask)
    {
        (w, k, cc, Some(y))
    } else if matches!(ty, Type::I32 | Type::I64)
        && let Some((w, k)) = [(x, y), (y, x)].into_iter().find_map(|(band, mask)| {
            if !cx.single_use(band) || !is_bit(mask) || cx.uses.get(mask.index()).copied().unwrap_or(3) > 2 {
                return None;
            }
            let &Op::Binary(Binary::Band, a, b) = cx.op(band)? else { return None };
            let w = if a == mask { b } else if b == mask { a } else { return None };
            let &Op::Binary(Binary::Ishl, _, k) = cx.op(mask)? else { return None };
            Some((w, k))
        })
    {
        (w, k, if cc == IntCC::Equal { IntCC::NotEqual } else { IntCC::Equal }, None)
    } else {
        return None;
    };
    let shifted = cx.emit(ty, Op::Binary(Binary::Ushr, word, k));
    let one = cx.emit(ty, Op::Const(1));
    let hit = cx.emit(ty, Op::Binary(Binary::Band, shifted, one));
    let z = zero.unwrap_or_else(|| cx.emit(ty, Op::Const(0)));
    Some(Op::Icmp(out_cc, hit, z))
}

/// Rewrites `(x >> k) == 0` / `!= 0` (logical right shift by constant `k`) into a sign test or unsigned range compare against `1 << k`.
fn shr_zero_compare(cx: &mut RewriteCx, ty: Type, cc: IntCC, shifted: Value, zero: Value) -> Option<Op> {
    if !cx.single_use(shifted) {
        return None;
    }
    let &Op::Binary(Binary::Ushr, inner, shift) = cx.op(shifted)? else { return None };
    let k = cx.uconst(shift, ty)?;
    let bits = u64::from(ty.bits());
    if k == 0 || k >= bits {
        return None;
    }
    let eq = cc == IntCC::Equal;
    if k == bits - 1 {
        return Some(Op::Icmp(if eq { IntCC::SignedGreaterThanOrEqual } else { IntCC::SignedLessThan }, inner, zero));
    }
    if k <= 23 {
        let bound = cx.emit(ty, Op::Const((1_u64 << k) as i64));
        return Some(Op::Icmp(if eq { IntCC::UnsignedLessThan } else { IntCC::UnsignedGreaterThanOrEqual }, inner, bound));
    }
    None
}

/// Normalizes integer compares against 0/1, signed-range boundaries, shifted-zero tests, or single-bit known-zero masks.
pub(super) fn zero_compare(cx: &mut RewriteCx, cc: IntCC, x: Value, y: Value) -> Option<Op> {
    let ty = cx.func.ty(x);
    if !ty.is_small_int() {
        return None;
    }
    let umax = ty.unsigned_max();
    let smin = 1_u64 << (ty.bits() - 1);
    let smax = smin - 1;
    let (cx_u, cy_u) = (cx.uconst(x, ty), cx.uconst(y, ty));
    if matches!(cc, IntCC::Equal | IntCC::NotEqual) {
        if cy_u == Some(0) && let Some(op) = shr_zero_compare(cx, ty, cc, x, y) {
            return Some(op);
        }
        if cx_u == Some(0) && let Some(op) = shr_zero_compare(cx, ty, cc, y, x) {
            return Some(op);
        }
    }
    let (val, new_cc, existing_zero) = match (cc, cx_u, cy_u) {
        (IntCC::UnsignedLessThanOrEqual, _, Some(0)) => (x, IntCC::Equal, Some(y)),
        (IntCC::UnsignedGreaterThanOrEqual, Some(0), _) => (y, IntCC::Equal, Some(x)),
        (IntCC::UnsignedGreaterThan, _, Some(0)) => (x, IntCC::NotEqual, Some(y)),
        (IntCC::UnsignedLessThan, Some(0), _) => (y, IntCC::NotEqual, Some(x)),
        (IntCC::UnsignedLessThan, _, Some(1)) => (x, IntCC::Equal, None),
        (IntCC::UnsignedGreaterThan, Some(1), _) => (y, IntCC::Equal, None),
        (IntCC::UnsignedGreaterThanOrEqual, _, Some(1)) => (x, IntCC::NotEqual, None),
        (IntCC::UnsignedLessThanOrEqual, Some(1), _) => (y, IntCC::NotEqual, None),
        (IntCC::UnsignedGreaterThan, _, Some(c)) if c == smax => (x, IntCC::SignedLessThan, None),
        (IntCC::UnsignedGreaterThanOrEqual, _, Some(c)) if c == smin => (x, IntCC::SignedLessThan, None),
        (IntCC::UnsignedLessThan, Some(c), _) if c == smax => (y, IntCC::SignedLessThan, None),
        (IntCC::UnsignedLessThanOrEqual, Some(c), _) if c == smin => (y, IntCC::SignedLessThan, None),
        (IntCC::UnsignedLessThanOrEqual, _, Some(c)) if c == smax => (x, IntCC::SignedGreaterThanOrEqual, None),
        (IntCC::UnsignedLessThan, _, Some(c)) if c == smin => (x, IntCC::SignedGreaterThanOrEqual, None),
        (IntCC::UnsignedGreaterThanOrEqual, Some(c), _) if c == smax => (y, IntCC::SignedGreaterThanOrEqual, None),
        (IntCC::UnsignedGreaterThan, Some(c), _) if c == smin => (y, IntCC::SignedGreaterThanOrEqual, None),
        (IntCC::SignedLessThanOrEqual, _, Some(c)) if c == umax => (x, IntCC::SignedLessThan, None),
        (IntCC::SignedGreaterThanOrEqual, Some(c), _) if c == umax => (y, IntCC::SignedLessThan, None),
        (IntCC::SignedGreaterThan, _, Some(c)) if c == umax => (x, IntCC::SignedGreaterThanOrEqual, None),
        (IntCC::SignedLessThan, Some(c), _) if c == umax => (y, IntCC::SignedGreaterThanOrEqual, None),
        (IntCC::Equal | IntCC::NotEqual, _, Some(k)) if k.is_power_of_two() && ((cx.known(x).0 & umax) | k) == umax => {
            (x, if cc == IntCC::Equal { IntCC::NotEqual } else { IntCC::Equal }, None)
        }
        (IntCC::Equal | IntCC::NotEqual, Some(k), _) if k.is_power_of_two() && ((cx.known(y).0 & umax) | k) == umax => {
            (y, if cc == IntCC::Equal { IntCC::NotEqual } else { IntCC::Equal }, None)
        }
        _ => return None,
    };
    Some(Op::Icmp(new_cc, val, existing_zero.unwrap_or_else(|| cx.emit(ty, Op::Const(0)))))
}

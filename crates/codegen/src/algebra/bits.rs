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

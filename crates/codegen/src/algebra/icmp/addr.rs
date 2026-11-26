use crate::fold::{Before, extend, known_bits, non_null_at};
use crate::ir::{Binary, Convert, Function, Inst, IntCC, Op, Type, Value};

use crate::algebra::RewriteCx;

pub(super) const MAX_DEPTH: u8 = 4;
pub(in crate::algebra) const ADDR_SPAN: i64 = 0x1000_0000;

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

pub(in crate::algebra) fn offset_bound(
    func: &Function,
    def_of: &dyn Fn(Value) -> Option<Inst>,
    known: &dyn Fn(Value) -> Option<i64>,
    v: Value,
    (bound, depth): (Option<(Value, u64)>, u8),
) -> u64 {
    let ty = func.ty(v);
    if !ty.is_small_int() {
        return u64::MAX;
    }
    let w = ty.unsigned_max();
    let mut hi = match known(v) {
        Some(c) => extend(c, ty, false) as u64,
        None => w & !known_bits(func, def_of, known, v, 0).0,
    };
    if let Some((bv, bmax)) = bound
        && bv == v
    {
        hi = hi.min(bmax);
    }
    if depth >= MAX_DEPTH || known(v).is_some() {
        return hi;
    }
    let next = (bound, depth + 1);
    let sub = match def_of(v).map(|inst| &func.insts[inst.index()].op) {
        Some(&Op::Convert(Convert::Uextend, a)) => offset_bound(func, def_of, known, a, next),
        Some(&Op::Binary(Binary::Band, a, b)) => offset_bound(func, def_of, known, a, next)
            .min(offset_bound(func, def_of, known, b, next)),
        Some(&Op::Binary(Binary::Ishl, a, sh)) => known(sh)
            .filter(|&k| (0..i64::from(ty.bits())).contains(&k))
            .and_then(|k| {
                offset_bound(func, def_of, known, a, next).checked_mul(1_u64 << (k as u32))
            })
            .filter(|&n| n <= w)
            .unwrap_or(u64::MAX),
        Some(&Op::Binary(Binary::Imul, a, b)) => [(a, b), (b, a)]
            .into_iter()
            .find_map(|(x, k)| {
                let c = known(k).map(|imm| extend(imm, ty, false) as u64)?;
                offset_bound(func, def_of, known, x, next)
                    .checked_mul(c)
                    .filter(|&n| n <= w)
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

fn sym_base_disp(
    func: &Function,
    def_of: &dyn Fn(Value) -> Option<Inst>,
    known: &dyn Fn(Value) -> Option<i64>,
    v: Value,
    depth: u8,
) -> Option<u64> {
    if depth >= MAX_DEPTH {
        return None;
    }
    match def_of(v).map(|inst| &func.insts[inst.index()].op) {
        Some(&Op::Symbol(_)) => Some(0),
        Some(&Op::Convert(Convert::Bitcast, a)) if func.ty(a) == func.ty(v) => {
            sym_base_disp(func, def_of, known, a, depth + 1)
        }
        Some(&Op::Binary(Binary::Iadd, a, b)) => {
            [(a, b), (b, a)].into_iter().find_map(|(base, off)| {
                sym_base_disp(func, def_of, known, base, depth + 1)?
                    .checked_add(offset_bound(func, def_of, known, off, (None, depth + 1)))
                    .filter(|&sum| sum <= ADDR_SPAN as u64)
            })
        }
        _ => None,
    }
}

/// Returns whether `v` denotes a guaranteed non-null stack, symbol, or slot address.
pub(crate) fn is_non_null_addr(
    func: &Function,
    def_of: &dyn Fn(Value) -> Option<Inst>,
    known: &dyn Fn(Value) -> Option<i64>,
    v: Value,
    depth: u8,
) -> bool {
    if depth >= MAX_DEPTH || func.ty(v).bits() < 32 {
        return false;
    }
    let Some(inst) = def_of(v) else { return false };
    let stack_disp = |base: Value, off: Value| {
        stack_base(func, def_of, base)
            && known(off).is_some_and(|c| (-ADDR_SPAN..=ADDR_SPAN).contains(&c))
    };
    match func.insts[inst.index()].op {
        Op::StackAddr(..) | Op::Symbol(_) | Op::SlotAddr(_) => true,
        Op::Binary(Binary::Iadd, a, b) => {
            stack_disp(a, b)
                || stack_disp(b, a)
                || sym_base_disp(func, def_of, known, v, depth).is_some()
        }
        Op::Binary(Binary::Isub, a, b) => stack_disp(a, b),
        Op::Select(_, a, b) => {
            is_non_null_addr(func, def_of, known, a, depth + 1)
                && is_non_null_addr(func, def_of, known, b, depth + 1)
        }
        Op::Convert(Convert::Bitcast, a) if func.ty(a) == func.ty(v) => {
            is_non_null_addr(func, def_of, known, a, depth + 1)
        }
        _ => false,
    }
}

pub(in crate::algebra) fn null_test(cc: IntCC, ty: Type, k: i64) -> Option<bool> {
    match (cc, extend(k, ty, false) as u64) {
        (IntCC::Equal, 0) | (IntCC::UnsignedLessThanOrEqual, 0) | (IntCC::UnsignedLessThan, 1) => {
            Some(false)
        }
        (IntCC::NotEqual, 0)
        | (IntCC::UnsignedGreaterThan, 0)
        | (IntCC::UnsignedGreaterThanOrEqual, 1) => Some(true),
        _ => None,
    }
}

pub(in crate::algebra) fn before<'a>(cx: &'a RewriteCx) -> Before<'a> {
    Before {
        prefix: &cx.out,
        block: None,
    }
}

/// Folds `Icmp(cc, x, y)` against `0` or `1` when one side is a non-null address where `at` reads it.
pub(crate) fn non_null_cmp(
    func: &Function,
    def_of: &dyn Fn(Value) -> Option<Inst>,
    known: &dyn Fn(Value) -> Option<i64>,
    at: Before,
    (cc, x, y): (IntCC, Value, Value),
) -> Option<bool> {
    let ty = func.ty(x);
    if !ty.is_small_int() || ty.bits() < 32 {
        return None;
    }
    [(cc, x, y), (cc.swapped(), y, x)]
        .into_iter()
        .find_map(|(norm, val, rhs)| {
            null_test(norm, ty, known(rhs)?).filter(|_| non_null_at(func, def_of, known, at, val))
        })
}

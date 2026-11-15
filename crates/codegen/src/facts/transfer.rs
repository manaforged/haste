use crate::ir::{Binary, Convert, Function, IntCC, Op, Type, Value};

use super::{Fact, Ranges, compared_range, swapped, unsigned_order};

/// The fact on an operand: its definition's range, or what one path knows about it.
pub(super) type Of<'a> = &'a dyn Fn(Value) -> Fact;

/// The range of the result of `op`, of type `ty`, from the facts `of` gives on its operands.
pub(super) fn transfer(func: &Function, ranges: &Ranges, op: &Op, ty: Type, of: Of<'_>) -> Fact {
    match *op {
        Op::Const(imm) => Fact::constant(ty, imm),
        Op::Icmp(..) | Op::Fcmp(..) | Op::Fits(..) => Fact::range(0, 1),
        Op::StackAddr(..) if ty == Type::I64 => Fact::range(1, u64::MAX - 0x1000_0000),
        Op::Symbol(_) | Op::SlotAddr(_) => Fact::range(1, ty.unsigned_max()),
        Op::Convert(Convert::Uextend, x) => of(x),
        Op::Convert(Convert::Ireduce, x) => Some(of(x)).filter(|inner| inner.hi <= ty.unsigned_max()).unwrap_or_else(|| Fact::full(ty)),
        Op::Binary(op, x, y) if func.ty(x).is_small_int() => binary(func, ranges, op, (x, y), of),
        Op::Select(cond, x, y) => match (arm(func, ranges, (cond, x, true), of), arm(func, ranges, (cond, y, false), of)) {
            (Some(a), Some(b)) => Fact::range(a.lo.min(b.lo), a.hi.max(b.hi)),
            (Some(only), None) | (None, Some(only)) => Fact::range(only.lo, only.hi),
            (None, None) => Fact::full(ty),
        },
        _ => Fact::full(ty),
    }
}

pub(super) fn valid(fact: &Fact) -> bool {
    fact.lo < fact.hi || (fact.lo == fact.hi && fact.admits(fact.lo))
}

fn unwrap_cond(func: &Function, ranges: &Ranges, mut cond: Value, mut taken: bool) -> (Value, bool) {
    for _ in 0..4 {
        match ranges.op(func, cond) {
            Some(&Op::Convert(Convert::Uextend, inner)) if func.ty(inner).is_small_int() => cond = inner,
            Some(&Op::Binary(Binary::Bxor, a, b)) => {
                let flipped = [(a, b), (b, a)].into_iter().find_map(|(inner, mask)| {
                    (ranges.of(mask).exact() == Some(1) && ranges.of(inner).hi <= 1).then_some(inner)
                });
                let Some(inner) = flipped else { break };
                cond = inner;
                taken = !taken;
            }
            _ => break,
        }
    }
    (cond, taken)
}

fn cond_bounds(
    func: &Function,
    (cc, p, _): (IntCC, Value, Value),
    (left, right): (&Fact, &Fact),
    taken: bool,
) -> Option<Option<(Fact, Fact)>> {
    let ty = func.ty(p);
    let Some(order) = unsigned_order(cc, ty, left, right) else { return Some(None) };
    let np = compared_range(order, ty, right, taken).map_or_else(|| left.clone(), |b| left.clone().meet(&b));
    let nq = compared_range(swapped(order), ty, left, taken).map_or_else(|| right.clone(), |b| right.clone().meet(&b));
    (valid(&np) && valid(&nq)).then_some(Some((np, nq)))
}

fn eval_under(
    func: &Function,
    ranges: &Ranges,
    v: Value,
    of: Of<'_>,
    ((p, np), (q, nq), depth): ((Value, &Fact), (Value, &Fact), u8),
) -> Fact {
    if v == p {
        return np.clone();
    }
    if v == q {
        return nq.clone();
    }
    let base = of(v);
    if depth >= 4 || !func.ty(v).is_small_int() {
        return base;
    }
    let Some(op) = ranges.op(func, v).filter(|op| matches!(op, Op::Binary(..) | Op::Convert(..))) else {
        return base;
    };
    let sub = |u: Value| eval_under(func, ranges, u, of, ((p, np), (q, nq), depth + 1));
    base.meet(&transfer(func, ranges, op, func.ty(v), &sub))
}

/// The fact on `value` as the arm a select on `cond` yields when `cond` is `taken`: the arm's own fact, met with what the condition says of it on that side. `None` when the select never yields the arm.
fn arm(func: &Function, ranges: &Ranges, (raw_cond, value, raw_taken): (Value, Value, bool), of: Of<'_>) -> Option<Fact> {
    let mut fact = of(value);
    let (cond, taken) = unwrap_cond(func, ranges, raw_cond, raw_taken);
    match ranges.op(func, cond) {
        Some(&Op::Icmp(cc, p, q)) if func.ty(p).is_small_int() => {
            let (left, right) = (of(p), of(q));
            if let Some((np, nq)) = cond_bounds(func, (cc, p, q), (&left, &right), taken)? {
                fact = if value == p {
                    np
                } else if value == q {
                    nq
                } else if matches!(ranges.op(func, value), Some(Op::Binary(..) | Op::Convert(..))) {
                    fact.meet(&eval_under(func, ranges, value, of, ((p, &np), (q, &nq), 0)))
                } else {
                    fact
                };
            }
        }
        Some(&Op::Fits(op, false, p, q)) if taken && computes(ranges.op(func, value), op, p, q) => {
            fact = fact.meet(&unwrapped(op, func.ty(p), &of(p), &of(q))?);
        }
        _ => {}
    }
    valid(&fact).then_some(fact)
}

/// Whether `def` is `p op q`, in either operand order when `op` commutes.
fn computes(def: Option<&Op>, op: Binary, p: Value, q: Value) -> bool {
    matches!(def, Some(&Op::Binary(found, a, b)) if found == op && ((a, b) == (p, q) || (op != Binary::Isub && (a, b) == (q, p))))
}

/// The range of `x op y` on the executions where it does not wrap at the width of `ty`; `None` when it always wraps.
pub(super) fn unwrapped(op: Binary, ty: Type, a: &Fact, b: &Fact) -> Option<Fact> {
    let max = ty.unsigned_max();
    let (lo, hi) = match op {
        Binary::Iadd => (a.lo.checked_add(b.lo)?, a.hi.saturating_add(b.hi)),
        Binary::Isub => (a.lo.saturating_sub(b.hi), a.hi.checked_sub(b.lo)?),
        Binary::Imul => (a.lo.checked_mul(b.lo)?, a.hi.saturating_mul(b.hi)),
        _ => return Some(Fact::full(ty)),
    };
    (lo <= max).then(|| Fact::range(lo, hi.min(max)))
}

/// The count a shift by `count` moves at the width of `ty`, when it is one constant.
fn shift_count(ty: Type, count: &Fact) -> Option<u32> {
    count.exact().map(|count| (count & u64::from(ty.bits() - 1)) as u32)
}

/// The range of `x op y` from the facts on its operands.
fn binary(func: &Function, ranges: &Ranges, op: Binary, (x, y): (Value, Value), of: Of<'_>) -> Fact {
    let ty = func.ty(x);
    let full = || Fact::full(ty);
    match op {
        Binary::Band => Fact::range(0, of(x).hi.min(of(y).hi)),
        Binary::Ushr => shift_count(ty, &of(y)).map_or_else(full, |shift| {
            let inner = of(x);
            Fact::range(inner.lo >> shift, inner.hi >> shift)
        }),
        Binary::Ishl => shift_count(ty, &of(y)).map_or_else(full, |shift| {
            let inner = of(x);
            if inner.hi <= (ty.unsigned_max() >> shift) { Fact::range(inner.lo << shift, inner.hi << shift) } else { full() }
        }),
        Binary::Udiv => match of(y).exact() {
            Some(divisor) if divisor > 0 => {
                let inner = of(x);
                Fact::range(inner.lo / divisor, inner.hi / divisor)
            }
            _ => full(),
        },
        Binary::Urem => match of(y).exact() {
            Some(divisor) if divisor > 0 => Fact::range(0, divisor - 1),
            _ => full(),
        },
        Binary::Imul => {
            let (a, b) = (of(x), of(y));
            a.hi.checked_mul(b.hi).filter(|&hi| hi <= ty.unsigned_max()).map_or_else(full, |hi| Fact::range(a.lo * b.lo, hi))
        }
        Binary::Iadd => {
            let (a, b) = (of(x), of(y));
            added(ty, &a, &b).or_else(|| wrapped(ty, &a, &b)).or_else(|| wrapped(ty, &b, &a)).unwrap_or_else(full)
        }
        Binary::Isub => {
            if let Some(&Op::Binary(Binary::Ushr, base, count)) = ranges.op(func, y)
                && base == x
            {
                let inner = of(x);
                let (lo, hi) = shift_count(ty, &of(count)).map_or((0, inner.hi), |k| (inner.lo - (inner.lo >> k), inner.hi - (inner.hi >> k)));
                return Fact::range(lo, hi);
            }
            let (a, b) = (of(x), of(y));
            if a.lo >= b.hi { Fact::range(a.lo - b.hi, a.hi - b.lo) } else { full() }
        }
        _ => full(),
    }
}

/// The range of `a + b` when no pair of values in the two ranges wraps at the width of `ty`.
fn added(ty: Type, a: &Fact, b: &Fact) -> Option<Fact> {
    a.hi.checked_add(b.hi).filter(|&hi| hi <= ty.unsigned_max()).map(|hi| Fact::range(a.lo + b.lo, hi))
}

/// The range of `a + k` for a constant `k` when every value of `a` wraps past the width of `ty`.
fn wrapped(ty: Type, a: &Fact, k: &Fact) -> Option<Fact> {
    let back = (ty.unsigned_max() - k.exact().filter(|&k| k > 0)?) + 1;
    (a.lo >= back).then(|| Fact::range(a.lo - back, a.hi - back))
}

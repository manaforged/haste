use crate::ir::{Binary, Block, Convert, Function, IntCC, Op, Type, Value};

use super::{Fact, Ranges, compared_range, unsigned_order};

/// The fact on an operand: its definition's range, or what one path knows about it.
pub(super) type Of<'a> = &'a dyn Fn(Value) -> Fact;

/// Returns the base pointer dereferenced within page zero by `op`, proving it nonzero when `op` runs.
#[inline]
pub(crate) fn deref_ptr(op: &Op) -> Option<Value> {
    match *op {
        Op::Load(addr, off) | Op::Store(_, addr, off) if (0..4096).contains(&off) => Some(addr),
        Op::AtomicLoad(addr, _)
        | Op::AtomicStore(_, addr, _)
        | Op::AtomicRmw(_, _, addr, _)
        | Op::AtomicCas(_, addr, ..) => Some(addr),
        _ => None,
    }
}

/// The way a branch goes when exactly one of its targets assumes a constant zero before any call or trap: control that enters that target breaks the assumption, so it never does.
pub(super) fn avoided(
    func: &Function,
    ranges: &Ranges,
    then: Block,
    otherwise: Block,
) -> Option<bool> {
    let barred = |block: Block| {
        func.blocks[block.index()]
            .insts
            .iter()
            .map(|inst| &func.insts[inst.index()].op)
            .take_while(|op| !matches!(op, Op::Call { .. } | Op::Trapz(_)))
            .any(|op| matches!(op, Op::Assume(value) if ranges.constant(*value) == Some(0)))
    };
    match (barred(then), barred(otherwise)) {
        (true, false) => Some(false),
        (false, true) => Some(true),
        _ => None,
    }
}

#[inline]
pub(super) fn note_deref_addr(
    func: &Function,
    addr: Value,
    falls_through: &[u32],
    epoch: u32,
    table: &mut [(u64, u64)],
) -> bool {
    let idx = addr.index();
    let (lo, hi) = &mut table[idx];
    if *lo == 0 && *hi != 0 && func.ty(addr) == Type::I64 && falls_through[idx] == epoch {
        *lo = 1;
        return true;
    }
    false
}

/// Raises the lower bound of a dereferenced pointer marked at `epoch` in `falls_through` to 1, returning it when narrowed.
#[inline]
pub(super) fn note_deref(
    func: &Function,
    op: &Op,
    falls_through: &[u32],
    epoch: u32,
    table: &mut [(u64, u64)],
) -> Option<Value> {
    let addr = deref_ptr(op)?;
    note_deref_addr(func, addr, falls_through, epoch, table).then_some(addr)
}

#[inline]
pub(super) fn known_range(op: &Op, ty: Type) -> Option<(u64, u64)> {
    Some(match *op {
        Op::Const(imm) => {
            let u = super::unsigned(imm, ty);
            (u, u)
        }
        Op::Icmp(..) | Op::Fcmp(..) | Op::Fits(..) => (0, 1),
        Op::StackAddr(..) if ty == Type::I64 => (1, u64::MAX - 0x1000_0000),
        Op::Symbol(_) | Op::SlotAddr(_) => (1, ty.unsigned_max()),
        _ => return None,
    })
}

/// Fast `(lo, hi)` range for `Ranges::propagate`, returning `None` when `op` always produces the full range (leaving `table` intact).
#[inline]
pub(super) fn fast_range(
    func: &Function,
    ranges: &Ranges,
    op: &Op,
    ty: Type,
) -> Option<(u64, u64)> {
    if let Some(pair) = known_range(op, ty) {
        return Some(pair);
    }
    match *op {
        Op::Convert(Convert::Uextend, x) => Some(ranges.table[x.index()]),
        Op::Convert(Convert::Ireduce, x) => {
            let inner = ranges.table[x.index()];
            (inner.1 <= ty.unsigned_max()).then_some(inner)
        }
        Op::Binary(bin, x, y) if func.ty(x).is_small_int() => {
            binary_bounds(func, ranges, bin, (x, y), |v| ranges.table[v.index()])
        }
        Op::Select(..) => {
            let fact = transfer(func, ranges, op, ty, &|value| ranges.of(value));
            Some((fact.lo, fact.hi))
        }
        _ => None,
    }
}

/// The range of the result of `op`, of type `ty`, from the facts `of` gives on its operands.
pub(super) fn transfer(func: &Function, ranges: &Ranges, op: &Op, ty: Type, of: Of<'_>) -> Fact {
    if let Some((lo, hi)) = known_range(op, ty) {
        return Fact::range(lo, hi);
    }
    match *op {
        Op::Convert(Convert::Uextend, x) => of(x),
        Op::Convert(Convert::Ireduce, x) => Some(of(x))
            .filter(|inner| inner.hi <= ty.unsigned_max())
            .unwrap_or_else(|| Fact::full(ty)),
        Op::Binary(op, x, y) if func.ty(x).is_small_int() => {
            binary_bounds(func, ranges, op, (x, y), |v| {
                let f = of(v);
                (f.lo, f.hi)
            })
            .map_or_else(|| Fact::full(ty), |(lo, hi)| Fact::range(lo, hi))
        }
        Op::Select(cond, x, y) => match (
            arm(func, ranges, (cond, x, true), of),
            arm(func, ranges, (cond, y, false), of),
        ) {
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

fn unwrap_cond(
    func: &Function,
    ranges: &Ranges,
    mut cond: Value,
    mut taken: bool,
) -> (Value, bool) {
    for _ in 0..4 {
        match ranges.op(func, cond) {
            Some(&Op::Convert(Convert::Uextend, inner)) if func.ty(inner).is_small_int() => {
                cond = inner
            }
            Some(&Op::Binary(Binary::Bxor, a, b)) => {
                let flipped = [(a, b), (b, a)].into_iter().find_map(|(inner, mask)| {
                    (ranges.of(mask).exact() == Some(1) && ranges.of(inner).hi <= 1)
                        .then_some(inner)
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
    let Some(order) = unsigned_order(cc, ty, left, right) else {
        return Some(None);
    };
    let np = compared_range(order, ty, right, taken)
        .map_or_else(|| left.clone(), |b| left.clone().meet(&b));
    let nq = compared_range(order.swapped(), ty, left, taken)
        .map_or_else(|| right.clone(), |b| right.clone().meet(&b));
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
    let Some(op) = ranges
        .op(func, v)
        .filter(|op| matches!(op, Op::Binary(..) | Op::Convert(..)))
    else {
        return base;
    };
    let sub = |u: Value| eval_under(func, ranges, u, of, ((p, np), (q, nq), depth + 1));
    base.meet(&transfer(func, ranges, op, func.ty(v), &sub))
}

/// The fact on `value` as the arm a select on `cond` yields when `cond` is `taken`: the arm's own fact, met with what the condition says of it on that side. `None` when the select never yields the arm.
fn arm(
    func: &Function,
    ranges: &Ranges,
    (raw_cond, value, raw_taken): (Value, Value, bool),
    of: Of<'_>,
) -> Option<Fact> {
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
                } else if matches!(
                    ranges.op(func, value),
                    Some(Op::Binary(..) | Op::Convert(..))
                ) {
                    fact.meet(&eval_under(
                        func,
                        ranges,
                        value,
                        of,
                        ((p, &np), (q, &nq), 0),
                    ))
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

#[inline]
fn shift_count_bounds(ty: Type, (lo, hi): (u64, u64)) -> Option<u32> {
    (lo == hi).then(|| (lo & u64::from(ty.bits() - 1)) as u32)
}

#[inline]
fn binary_bounds(
    func: &Function,
    ranges: &Ranges,
    op: Binary,
    (x, y): (Value, Value),
    bounds: impl Fn(Value) -> (u64, u64),
) -> Option<(u64, u64)> {
    let ty = func.ty(x);
    match op {
        Binary::Band => Some((0, bounds(x).1.min(bounds(y).1))),
        Binary::Bor => {
            let (a, b) = (bounds(x), bounds(y));
            added_bounds(ty, a, b).map(|(_, hi)| (a.0.max(b.0), hi))
        }
        Binary::Ushr => shift_count_bounds(ty, bounds(y)).map(|shift| {
            let (lo, hi) = bounds(x);
            (lo >> shift, hi >> shift)
        }),
        Binary::Ishl => shift_count_bounds(ty, bounds(y)).and_then(|shift| {
            let (lo, hi) = bounds(x);
            (hi <= (ty.unsigned_max() >> shift)).then_some((lo << shift, hi << shift))
        }),
        Binary::Udiv => {
            let (dlo, dhi) = bounds(y);
            (dlo == dhi && dlo > 0).then(|| {
                let (lo, hi) = bounds(x);
                (lo / dlo, hi / dlo)
            })
        }
        Binary::Urem => {
            let (dlo, dhi) = bounds(y);
            (dlo == dhi && dlo > 0).then_some((0, dlo - 1))
        }
        Binary::Imul => {
            let (a, b) = (bounds(x), bounds(y));
            a.1.checked_mul(b.1)
                .filter(|&hi| hi <= ty.unsigned_max())
                .map(|hi| (a.0 * b.0, hi))
        }
        Binary::Iadd => {
            let (a, b) = (bounds(x), bounds(y));
            added_bounds(ty, a, b)
                .or_else(|| wrapped_bounds(ty, a, b))
                .or_else(|| wrapped_bounds(ty, b, a))
        }
        Binary::Isub => {
            if let Some(&Op::Binary(Binary::Ushr, base, count)) = ranges.op(func, y)
                && base == x
            {
                let (lo, hi) = bounds(x);
                return Some(
                    shift_count_bounds(ty, bounds(count))
                        .map_or((0, hi), |k| (lo - (lo >> k), hi - (hi >> k))),
                );
            }
            let (a, b) = (bounds(x), bounds(y));
            if a.0 >= b.1 {
                return Some((a.0 - b.1, a.1 - b.0));
            }
            (b.0 == b.1).then(|| {
                let k = super::unsigned(b.0.wrapping_neg() as i64, ty);
                added_bounds(ty, a, (k, k))
            })?
        }
        _ => None,
    }
}

#[inline]
fn added_bounds(ty: Type, a: (u64, u64), b: (u64, u64)) -> Option<(u64, u64)> {
    a.1.checked_add(b.1)
        .filter(|&hi| hi <= ty.unsigned_max())
        .map(|hi| (a.0 + b.0, hi))
}

#[inline]
fn wrapped_bounds(ty: Type, a: (u64, u64), k: (u64, u64)) -> Option<(u64, u64)> {
    let val = (k.0 == k.1 && k.0 > 0).then_some(k.0)?;
    let back = (ty.unsigned_max() - val) + 1;
    (a.0 >= back).then_some((a.0 - back, a.1 - back))
}

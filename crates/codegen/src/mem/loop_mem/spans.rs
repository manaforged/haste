use crate::cfg::Cfg;
use crate::ir::{Binary, Convert, Function, Op, StackSlot, Type, Value};

use super::super::MemBase;
use super::SPAN_LIMIT;

fn mask_hi(hi: i64) -> Option<i64> {
    let u = u64::try_from(hi).ok()?;
    let m = if u == 0 {
        0
    } else {
        u64::MAX >> u.leading_zeros()
    };
    i64::try_from(m).ok().filter(|&v| v <= SPAN_LIMIT)
}

fn shift_amt(ty: Type, ry: Option<(i64, i64)>) -> Option<u32> {
    let (_, sh) = ry.filter(|&(lo, hi)| lo == hi)?;
    Some((sh as u32) & (ty.bits() - 1))
}

fn eval_band(
    is_64: bool,
    umax: i64,
    rx: Option<(i64, i64)>,
    ry: Option<(i64, i64)>,
) -> Option<(i64, i64)> {
    if !is_64 {
        return Some((0, rx.map_or(umax, |r| r.1).min(ry.map_or(umax, |r| r.1))));
    }
    match (rx.filter(|r| r.0 >= 0), ry.filter(|r| r.0 >= 0)) {
        (Some(a), Some(b)) => Some((0, a.1.min(b.1))),
        (Some(a), None) => Some((0, a.1)),
        (None, Some(b)) => Some((0, b.1)),
        (None, None) => None,
    }
}

fn eval_bor(
    is_64: bool,
    umax: i64,
    rx: Option<(i64, i64)>,
    ry: Option<(i64, i64)>,
) -> Option<(i64, i64)> {
    let (Some((la, ha)), Some((lb, hb))) = (rx, ry) else {
        return None;
    };
    if la < 0 || lb < 0 {
        return None;
    }
    let hi = mask_hi(ha)? | mask_hi(hb)?;
    (is_64 || hi <= umax).then_some((la.max(lb), hi))
}

fn eval_ushr(ty: Type, rx: Option<(i64, i64)>, ry: Option<(i64, i64)>) -> Option<(i64, i64)> {
    let k = shift_amt(ty, ry)?;
    if k == 0 {
        return rx;
    }
    if ty != Type::I64 {
        let (lo, hi) = rx.unwrap_or((0, ty.unsigned_max() as i64));
        return Some(((lo as u64 >> k) as i64, (hi as u64 >> k) as i64));
    }
    match rx.filter(|r| r.0 >= 0) {
        Some((lo, hi)) => Some(((lo as u64 >> k) as i64, (hi as u64 >> k) as i64)),
        None => Some((0, (u64::MAX >> k) as i64)),
    }
}

fn eval_ishl(ty: Type, rx: Option<(i64, i64)>, ry: Option<(i64, i64)>) -> Option<(i64, i64)> {
    let k = shift_amt(ty, ry)?;
    let (lo, hi) = rx?;
    if k == 0 {
        return Some((lo, hi));
    }
    let fits = if ty == Type::I64 {
        lo >= (i64::MIN >> k) && hi <= (i64::MAX >> k)
    } else {
        hi <= (ty.unsigned_max() >> k) as i64
    };
    fits.then_some((lo << k, hi << k))
}

fn eval_bit_range(
    ty: Type,
    bin: Binary,
    rx: Option<(i64, i64)>,
    ry: Option<(i64, i64)>,
) -> Option<(i64, i64)> {
    let (umax, is_64) = (ty.unsigned_max() as i64, ty == Type::I64);
    match bin {
        Binary::Band => eval_band(is_64, umax, rx, ry),
        Binary::Bor => eval_bor(is_64, umax, rx, ry),
        Binary::Ushr => eval_ushr(ty, rx, ry),
        Binary::Ishl => eval_ishl(ty, rx, ry),
        _ => None,
    }
}

fn eval_udiv_urem(
    is_rem: bool,
    ty: Type,
    rx: Option<(i64, i64)>,
    ry: Option<(i64, i64)>,
) -> Option<(i64, i64)> {
    let (_, d) = ry.filter(|&(lo, hi)| lo == hi && lo > 0)?;
    if is_rem {
        let hi = rx.filter(|r| r.0 >= 0).map_or(d - 1, |r| r.1.min(d - 1));
        return Some((0, hi));
    }
    let (lo, hi) = if ty == Type::I64 {
        rx.filter(|r| r.0 >= 0)?
    } else {
        rx.unwrap_or((0, ty.unsigned_max() as i64))
    };
    Some((lo / d, hi / d))
}

fn eval_imul(is_64: bool, umax: i64, a: (i64, i64), b: (i64, i64)) -> Option<(i64, i64)> {
    let (la, ha) = a;
    let (lb, hb) = b;
    let ps = [
        la.checked_mul(lb)?,
        la.checked_mul(hb)?,
        ha.checked_mul(lb)?,
        ha.checked_mul(hb)?,
    ];
    let lo = *ps.iter().min()?;
    let hi = *ps.iter().max()?;
    (is_64 || hi <= umax).then_some((lo, hi))
}

fn eval_arith_range(
    ty: Type,
    bin: Binary,
    rx: Option<(i64, i64)>,
    ry: Option<(i64, i64)>,
) -> Option<(i64, i64)> {
    let (umax, is_64) = (ty.unsigned_max() as i64, ty == Type::I64);
    match bin {
        Binary::Urem => eval_udiv_urem(true, ty, rx, ry),
        Binary::Udiv => eval_udiv_urem(false, ty, rx, ry),
        Binary::Iadd | Binary::Isub | Binary::Imul => {
            let (Some(a), Some(b)) = (rx, ry) else {
                return None;
            };
            match bin {
                Binary::Iadd => {
                    a.0.checked_add(b.0)
                        .zip(a.1.checked_add(b.1))
                        .filter(|&(_, hi)| is_64 || hi <= umax)
                }
                Binary::Isub if is_64 || a.0 >= b.1 => {
                    a.0.checked_sub(b.1).zip(a.1.checked_sub(b.0))
                }
                Binary::Imul => eval_imul(is_64, umax, a, b),
                _ => None,
            }
        }
        _ => None,
    }
}

fn eval_convert_range(
    func: &Function,
    kind: Convert,
    x: Value,
    ty: Type,
    ranges: &[Option<(i64, i64)>],
) -> Option<(i64, i64)> {
    let umax = ty.unsigned_max() as i64;
    match kind {
        Convert::Uextend if func.ty(x).is_small_int() => {
            Some(ranges[x.index()].unwrap_or((0, func.ty(x).unsigned_max() as i64)))
        }
        Convert::Sextend => {
            let (lo, hi) = ranges[x.index()]?;
            let lo_ok = ty == Type::I64 || lo >= 0;
            let hi_ok = func.ty(x) == Type::I64 || hi <= (func.ty(x).unsigned_max() >> 1) as i64;
            (lo_ok && hi_ok).then_some((lo, hi))
        }
        Convert::Ireduce => match ranges[x.index()] {
            Some((lo, hi)) if lo >= 0 && hi <= umax => Some((lo, hi)),
            _ => Some((0, umax)),
        },
        Convert::Bitcast if func.ty(x) == ty => ranges[x.index()],
        _ => None,
    }
}

fn eval_range(
    func: &Function,
    op: &Op,
    ty: Type,
    ranges: &[Option<(i64, i64)>],
) -> Option<(i64, i64)> {
    if !ty.is_small_int() {
        return None;
    }
    let r = match *op {
        Op::Const(imm) => {
            let v = if ty == Type::I64 {
                imm
            } else {
                ((imm as u64) & ty.unsigned_max()) as i64
            };
            (v, v)
        }
        Op::Icmp(..) | Op::Fcmp(..) | Op::Fits(..) => (0, 1),
        Op::Convert(kind, x) => eval_convert_range(func, kind, x, ty, ranges)?,
        Op::Select(_, a, b) => match (ranges[a.index()], ranges[b.index()]) {
            (Some((la, ha)), Some((lb, hb))) => (la.min(lb), ha.max(hb)),
            _ => return None,
        },
        Op::Binary(bin @ (Binary::Band | Binary::Bor | Binary::Ushr | Binary::Ishl), x, y) => {
            eval_bit_range(ty, bin, ranges[x.index()], ranges[y.index()])?
        }
        Op::Binary(bin, x, y) => eval_arith_range(ty, bin, ranges[x.index()], ranges[y.index()])?,
        _ => return None,
    };
    (r.0 >= -SPAN_LIMIT && r.1 <= SPAN_LIMIT && r.0 <= r.1).then_some(r)
}

fn ptr_base(
    v: Value,
    spans: &[Option<(MemBase, i64, i64)>],
    slots: &[Option<(StackSlot, i32)>],
    symbols: &[Option<u32>],
) -> (MemBase, i64, i64) {
    if let Some(s) = spans[v.index()] {
        return s;
    }
    if let Some((slot, off)) = slots[v.index()] {
        return (MemBase::Slot(slot), i64::from(off), i64::from(off));
    }
    if let Some(sym) = symbols.get(v.index()).copied().flatten() {
        return (MemBase::Global(sym), 0, 0);
    }
    (MemBase::Addr(v), 0, 0)
}

fn eval_ptr_offset(
    is_sub: bool,
    a: Value,
    b: Value,
    ranges: &[Option<(i64, i64)>],
    ptrs: (
        &[Option<(MemBase, i64, i64)>],
        &[Option<(StackSlot, i32)>],
        &[Option<u32>],
    ),
) -> Option<(MemBase, i64, i64)> {
    let (ptr, off, sign) = match (is_sub, ranges[a.index()], ranges[b.index()]) {
        (true, None, Some(off)) => (a, off, true),
        (false, None, Some(off)) => (a, off, false),
        (false, Some(off), None) => (b, off, false),
        _ => return None,
    };
    let (base, plo, phi) = ptr_base(ptr, ptrs.0, ptrs.1, ptrs.2);
    if sign {
        plo.checked_sub(off.1)
            .zip(phi.checked_sub(off.0))
            .map(|(lo, hi)| (base, lo, hi))
    } else {
        plo.checked_add(off.0)
            .zip(phi.checked_add(off.1))
            .map(|(lo, hi)| (base, lo, hi))
    }
}

fn eval_span(
    func: &Function,
    op: &Op,
    ty: Type,
    ranges: &[Option<(i64, i64)>],
    ptrs: (
        &[Option<(MemBase, i64, i64)>],
        &[Option<(StackSlot, i32)>],
        &[Option<u32>],
    ),
) -> Option<(MemBase, i64, i64)> {
    if ty != Type::I64 {
        return None;
    }
    let span = match *op {
        Op::StackAddr(slot, off) => (MemBase::Slot(slot), i64::from(off), i64::from(off)),
        Op::Symbol(sym) | Op::SlotAddr(sym) => (MemBase::Global(sym), 0, 0),
        Op::Convert(Convert::Bitcast, x) if func.ty(x) == Type::I64 => ptrs.0[x.index()]?,
        Op::Select(_, a, b) => {
            let ((ba, la, ha), (bb, lb, hb)) = (
                ptr_base(a, ptrs.0, ptrs.1, ptrs.2),
                ptr_base(b, ptrs.0, ptrs.1, ptrs.2),
            );
            if ba != bb {
                return None;
            }
            (ba, la.min(lb), ha.max(hb))
        }
        Op::Binary(bin @ (Binary::Iadd | Binary::Isub), a, b) => {
            eval_ptr_offset(bin == Binary::Isub, a, b, ranges, ptrs)?
        }
        _ => return None,
    };
    (span.1 >= -SPAN_LIMIT && span.2 <= SPAN_LIMIT && span.1 <= span.2).then_some(span)
}

pub(in crate::mem) fn compute_spans(
    func: &Function,
    cfg: &Cfg,
    slots: &[Option<(StackSlot, i32)>],
    symbols: &[Option<u32>],
) -> Vec<Option<(MemBase, i64, i64)>> {
    let n = func.types.len();
    let mut ranges = vec![None; n];
    let mut spans = vec![None; n];
    for &b in &cfg.rpo {
        for &inst in &func.blocks[b.index()].insts {
            let data = &func.insts[inst.index()];
            let Some(res) = data.result else { continue };
            let ty = func.ty(res);
            ranges[res.index()] = eval_range(func, &data.op, ty, &ranges);
            spans[res.index()] = eval_span(func, &data.op, ty, &ranges, (&spans, slots, symbols));
        }
    }
    spans
}

use super::RewriteCx;
use crate::fold::extend;
use crate::ir::{Binary, Op, Type, Unary, Value};

const DEPTH: u8 = 2;

struct Sum {
    terms: Vec<(Value, bool)>,
    imm: i64,
    freed: usize,
}

fn children(cx: &RewriteCx, v: Value, neg: bool) -> Option<[Option<(Value, bool)>; 2]> {
    match *cx.op(v)? {
        Op::Binary(Binary::Iadd, x, y) => Some([Some((x, neg)), Some((y, neg))]),
        Op::Binary(Binary::Isub, x, y) => Some([Some((x, neg)), Some((y, !neg))]),
        Op::Unary(Unary::Ineg, x) => Some([Some((x, !neg)), None]),
        _ => None,
    }
}

fn collect(cx: &RewriteCx, sum: &mut Sum, (v, neg): (Value, bool), depth: u8, dead_parent: bool) {
    if let Some(c) = cx.sconst(v) {
        sum.imm = if neg { sum.imm.wrapping_sub(c) } else { sum.imm.wrapping_add(c) };
        return;
    }
    let Some(kids) = children(cx, v, neg).filter(|_| depth < DEPTH) else {
        sum.terms.push((v, neg));
        return;
    };
    let dead = dead_parent && cx.single_use(v);
    sum.freed += usize::from(dead);
    for kid in kids.into_iter().flatten() {
        collect(cx, sum, kid, depth + 1, dead);
    }
}

fn cancel(terms: &mut Vec<(Value, bool)>) -> bool {
    let (mut i, len) = (0, terms.len());
    while i < terms.len() {
        let (v, neg) = terms[i];
        match terms[i + 1..].iter().position(|&(w, n)| w == v && n != neg) {
            Some(j) => {
                terms.remove(i + 1 + j);
                terms.remove(i);
            }
            None => i += 1,
        }
    }
    terms.len() < len
}

/// Rewrites an `Iadd`/`Isub` tree of up to four terms with `x` and `-x` cancelled and constants folded, when that takes fewer ops.
pub(super) fn cancel_terms(cx: &mut RewriteCx, fits_used: &[bool], res: Option<Value>, (bin, a, b): (Binary, Value, Value)) -> Option<Op> {
    let ty = cx.func.ty(a);
    if !ty.is_small_int() || [a, b].iter().any(|v| fits_used.get(v.index()).copied().unwrap_or(false)) { return None; }
    let mut sum = Sum { terms: Vec::with_capacity(4), imm: 0, freed: 1 };
    collect(cx, &mut sum, (a, false), 1, true);
    collect(cx, &mut sum, (b, bin == Binary::Isub), 1, true);
    if !cancel(&mut sum.terms) { return None; }
    let imm = extend(sum.imm, ty, true);
    let mut steps: Vec<(Binary, Value)> = sum.terms.iter().map(|&(v, n)| (if n { Binary::Isub } else { Binary::Iadd }, v)).collect();
    steps.sort_by_key(|&(op, _)| op == Binary::Isub);
    let cost = match steps.first() {
        None => 0,
        Some(&(Binary::Isub, _)) => steps.len(),
        Some(_) => steps.len() - 1 + usize::from(imm != 0),
    };
    if cost >= sum.freed { return None; }
    if steps.is_empty() { return Some(Op::Const(imm)); }
    if imm != 0 {
        let c = cx.emit(ty, Op::Const(imm));
        if steps[0].0 == Binary::Isub { steps.insert(0, (Binary::Iadd, c)); } else { steps.push((Binary::Iadd, c)); }
    }
    let (mut acc, mut pending) = (None, None);
    for (op, v) in steps {
        if let Some(prev) = pending.take() { acc = Some(cx.emit(ty, prev)); }
        pending = Some(match acc {
            Some(lhs) => Op::Binary(op, lhs, v),
            None if op == Binary::Isub => Op::Unary(Unary::Ineg, v),
            None => { acc = Some(v); continue; }
        });
    }
    if pending.is_none() { cx.alias(res, acc?); }
    pending
}

/// Rewrites `(v + c) * k` with a single-use sum as `v * k + c * k`, so the constant can join an address offset.
pub(super) fn scale_sum(cx: &mut RewriteCx, a: Value, b: Value) -> Option<Op> {
    let (x, scale) = [(a, b), (b, a)].into_iter().find(|&(_, k)| cx.sconst(k).is_some())?;
    let k = cx.sconst(scale)?;
    let ty = cx.func.ty(x);
    if !ty.is_small_int() || !cx.single_use(x) {
        return None;
    }
    let &Op::Binary(Binary::Iadd, p, q) = cx.op(x)? else { return None };
    let (v, c) = super::const_addend(cx, p, q)?;
    let scaled = cx.emit(ty, Op::Binary(Binary::Imul, v, scale));
    let addend = cx.emit(ty, Op::Const(extend(c.wrapping_mul(k), ty, true)));
    Some(Op::Binary(Binary::Iadd, scaled, addend))
}

/// The value `v` scales and the power of two it scales by, as a left shift amount, when `v` is a shift left by a constant or a multiply by a power of two.
fn scaled(cx: &RewriteCx, ty: Type, v: Value) -> Option<(Value, i64)> {
    match cx.func.insts[cx.def(v)?.index()].op {
        Op::Binary(Binary::Ishl, x, k) => Some((x, cx.sconst(k)?)),
        Op::Binary(Binary::Imul, m1, m2) => {
            [(m1, m2), (m2, m1)].into_iter().find_map(|(x, cv)| cx.uconst(cv, ty).filter(|&m| m > 1 && m.is_power_of_two()).map(|m| (x, i64::from(m.trailing_zeros()))))
        }
        _ => None,
    }
}

/// `base + -((idx << k) + c)` as `(base - (idx << k)) + -c`, so the scaled difference matches the same address computed without the offset.
pub(super) fn negated_offset_add(cx: &mut RewriteCx, ty: Type, a: Value, b: Value) -> Option<Op> {
    if !ty.is_small_int() {
        return None;
    }
    for (base, rhs) in [(a, b), (b, a)] {
        let Some(Op::Unary(Unary::Ineg, sum)) = cx.def(rhs).map(|inst| cx.func.insts[inst.index()].op.clone()) else { continue };
        let Some(Op::Binary(Binary::Iadd, p, q)) = cx.def(sum).map(|inst| cx.func.insts[inst.index()].op.clone()) else { continue };
        let Some((term, c)) = super::const_addend(cx, p, q) else { continue };
        let Some((idx, shift)) = scaled(cx, ty, term) else { continue };
        let k = cx.emit(ty, Op::Const(shift));
        let shifted = cx.emit(ty, Op::Binary(Binary::Ishl, idx, k));
        let diff = cx.emit(ty, Op::Binary(Binary::Isub, base, shifted));
        let offset = cx.emit(ty, Op::Const(extend(c.wrapping_neg(), ty, true)));
        return Some(Op::Binary(Binary::Iadd, diff, offset));
    }
    None
}

pub(super) fn negated_shift_add(cx: &RewriteCx, ty: Type, a: Value, b: Value) -> Option<(Value, Value, i64)> {
    for (base, rhs) in [(a, b), (b, a)] {
        let Some((neg, shift)) = scaled(cx, ty, rhs) else { continue };
        let Some(neg_inst) = cx.def(neg) else { continue };
        match cx.func.insts[neg_inst.index()].op {
            Op::Binary(Binary::Isub, zero, idx) if cx.uconst(zero, ty) == Some(0) => return Some((base, idx, shift)),
            Op::Unary(Unary::Ineg, idx) => return Some((base, idx, shift)),
            _ => {}
        }
    }
    None
}

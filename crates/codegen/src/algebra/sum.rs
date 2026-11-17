use super::RewriteCx;
use crate::fold::extend;
use crate::ir::{Binary, Op, Unary, Value};

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

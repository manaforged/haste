use crate::alloc::Plan;
use crate::ir::{Binary, Convert, Function, Inst, Op, Value};

pub(crate) fn half_score((lo, hi): (Option<i64>, Option<i64>), zero_only: bool) -> u8 {
    let hit = |v: Option<i64>| matches!(v, Some(0)) || (!zero_only && matches!(v, Some(-1)));
    u8::from(hit(lo)) + u8::from(hit(hi))
}

pub(crate) fn const_val<R: Copy>(func: &Function, plan: &Plan<R>, defs: &[Option<Inst>], v: Value) -> Option<i64> {
    plan.const_of(v).or_else(|| match func.insts[defs[v.index()]?.index()].op {
        Op::Const(imm) => Some(imm),
        _ => None,
    })
}

pub(crate) fn const_halves<R: Copy>(func: &Function, plan: &Plan<R>, defs: &[Option<Inst>], v: Value) -> (Option<i64>, Option<i64>) {
    if let Some(imm) = const_val(func, plan, defs, v) {
        return (Some(imm), Some(imm >> 63));
    }
    let Some(inst) = defs[v.index()] else { return (None, None) };
    match func.insts[inst.index()].op {
        Op::Binary(Binary::Iconcat, lo, hi) => (const_val(func, plan, defs, lo), const_val(func, plan, defs, hi)),
        _ => (None, None),
    }
}

pub(crate) fn shift_const<R: Copy>(func: &Function, plan: &Plan<R>, defs: &[Option<Inst>], v: Value) -> Option<u8> {
    const_halves(func, plan, defs, v).0.map(|imm| (imm & 127) as u8)
}

pub(crate) fn has_zero_high<R: Copy>(func: &Function, plan: &Plan<R>, defs: &[Option<Inst>], v: Value) -> bool {
    if const_halves(func, plan, defs, v).1 == Some(0) {
        return true;
    }
    let Some(inst) = defs[v.index()] else { return false };
    match func.insts[inst.index()].op {
        Op::Convert(Convert::Uextend, inner) => func.ty(inner).bits() <= 64,
        Op::Binary(Binary::Band, a, b) => const_halves(func, plan, defs, a).1 == Some(0) || const_halves(func, plan, defs, b).1 == Some(0),
        Op::Binary(Binary::Ushr, _, k) => shift_const(func, plan, defs, k).is_some_and(|s| s >= 64),
        _ => false,
    }
}

pub(crate) fn known_halves<R: Copy>(func: &Function, plan: &Plan<R>, defs: &[Option<Inst>], v: Value) -> (Option<i64>, Option<i64>) {
    let (lo, hi) = const_halves(func, plan, defs, v);
    (lo, hi.or_else(|| has_zero_high(func, plan, defs, v).then_some(0)))
}

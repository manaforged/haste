use crate::alloc::Plan;
use crate::ir::{Binary, Function, Inst, Op, Value};

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

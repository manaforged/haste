use super::{RangeState, const_val, def_inst};
use crate::fold::{extend, known_bits};
use crate::ir::{Binary, Convert, Function, Inst, Op, Type, Unary, Value};

fn signed_max(ty: Type) -> u64 {
    (1_u64 << (ty.bits() - 1)) - 1
}

fn shift_count(state: &RangeState, ty: Type, sh: Value) -> Option<u32> {
    const_val(state, sh).map(|c| (c as u32) & (ty.bits() - 1))
}

fn pos_const(state: &RangeState, v: Value) -> Option<u64> {
    const_val(state, v).map(|d| d as u64).filter(|&d| d > 0)
}

fn smax_u_bounds(sm: u64, w: u64, (l1, h1): (u64, u64), (l2, h2): (u64, u64)) -> (u64, u64) {
    match (h1 <= sm, h2 <= sm) {
        (true, true) => (l1.max(l2), h1.max(h2)),
        (true, false) => (l1, sm),
        (false, true) => (l2, sm),
        (false, false) => (0, w),
    }
}

fn shift_div_u_bounds(
    func: &Function,
    state: &RangeState,
    ty: Type,
    bin: Binary,
    (a, b): (Value, Value),
    depth: u8,
) -> (u64, u64) {
    let w = ty.unsigned_max();
    let (l1, h1) = u_bounds(func, state, a, depth + 1);
    match bin {
        Binary::Ushr => shift_count(state, ty, b).map_or((0, h1), |k| (l1 >> k, h1 >> k)),
        Binary::Sshr => {
            if h1 <= signed_max(ty) {
                shift_count(state, ty, b).map_or((0, h1), |k| (l1 >> k, h1 >> k))
            } else {
                (0, w)
            }
        }
        Binary::Ishl => shift_count(state, ty, b)
            .filter(|&k| h1 <= (w >> k))
            .map_or((0, w), |k| (l1 << k, h1 << k)),
        Binary::Urem => (0, h1.min(pos_const(state, b).map_or(w, |d| d - 1))),
        Binary::Udiv => pos_const(state, b).map_or((0, h1), |d| (l1 / d, h1 / d)),
        Binary::Srem if h1 <= signed_max(ty) => {
            let rem = const_val(state, b)
                .map(|c| extend(c, ty, true).unsigned_abs())
                .filter(|&d| d > 0)
                .map_or(w, |d| d - 1);
            (0, h1.min(rem))
        }
        Binary::Sdiv if h1 <= signed_max(ty) => {
            let (l2, h2) = u_bounds(func, state, b, depth + 1);
            if h2 <= signed_max(ty) && l2 >= 1 {
                (l1 / h2, h1 / l2)
            } else {
                (0, w)
            }
        }
        _ => (0, w),
    }
}

fn bin_u_bounds(
    func: &Function,
    state: &RangeState,
    ty: Type,
    bin: Binary,
    (a, b): (Value, Value),
    depth: u8,
) -> (u64, u64) {
    if matches!(
        bin,
        Binary::Ushr
            | Binary::Sshr
            | Binary::Ishl
            | Binary::Urem
            | Binary::Udiv
            | Binary::Srem
            | Binary::Sdiv
    ) {
        return shift_div_u_bounds(func, state, ty, bin, (a, b), depth);
    }
    let w = ty.unsigned_max();
    let ((l1, h1), (l2, h2)) = (
        u_bounds(func, state, a, depth + 1),
        u_bounds(func, state, b, depth + 1),
    );
    let sm = signed_max(ty);
    match bin {
        Binary::Iadd => h1
            .checked_add(h2)
            .filter(|&h| h <= w)
            .map_or((0, w), |h| (l1 + l2, h)),
        Binary::Isub if l1 >= h2 => (l1 - h2, h1 - l2),
        Binary::Imul => h1
            .checked_mul(h2)
            .filter(|&h| h <= w)
            .map_or((0, w), |h| (l1 * l2, h)),
        Binary::Band => (0, h1.min(h2)),
        Binary::Umin => (l1.min(l2), h1.min(h2)),
        Binary::Umax => (l1.max(l2), h1.max(h2)),
        Binary::Smin if h1 <= sm && h2 <= sm => (l1.min(l2), h1.min(h2)),
        Binary::Smax => smax_u_bounds(sm, w, (l1, h1), (l2, h2)),
        _ => (0, w),
    }
}

fn op_u_bounds(func: &Function, state: &RangeState, v: Value, depth: u8) -> (u64, u64) {
    let ty = func.ty(v);
    let w = ty.unsigned_max();
    let Some(inst) = def_inst(state, v) else {
        return (0, w);
    };
    let sub = |u: Value| u_bounds(func, state, u, depth + 1);
    match func.insts[inst.index()].op {
        Op::Unary(Unary::Bnot, a) if func.ty(a).is_small_int() => {
            let (l, h) = sub(a);
            (w - h.min(w), w - l.min(w))
        }
        Op::Unary(Unary::Clz | Unary::Ctz | Unary::Popcnt, _) => (0, u64::from(ty.bits())),
        Op::Select(_, a, b) => {
            let ((l1, h1), (l2, h2)) = (sub(a), sub(b));
            (l1.min(l2), h1.max(h2))
        }
        Op::Convert(Convert::Uextend, a) if func.ty(a).is_small_int() => {
            let (l, h) = sub(a);
            let aw = func.ty(a).unsigned_max();
            (l.min(aw), h.min(aw))
        }
        Op::Convert(Convert::Sextend, a) if func.ty(a).is_small_int() => {
            let (l, h) = sub(a);
            if h <= signed_max(func.ty(a)) {
                (l, h)
            } else {
                (0, w)
            }
        }
        Op::Convert(Convert::Ireduce, a) if func.ty(a).is_small_int() => {
            let (l, h) = sub(a);
            if h <= w { (l, h) } else { (0, w) }
        }
        Op::Binary(bin, a, b) => bin_u_bounds(func, state, ty, bin, (a, b), depth),
        _ => (0, w),
    }
}

pub(super) fn u_bounds(func: &Function, state: &RangeState, v: Value, depth: u8) -> (u64, u64) {
    let ty = func.ty(v);
    let w = ty.unsigned_max();
    if !ty.is_small_int() {
        return (0, w);
    }
    let (z, o) = known_bits(
        func,
        &|u| def_inst(state, u),
        &|u| const_val(state, u),
        v,
        depth,
    );
    let (bmin, bmax) = (o & w, !z & w);
    if depth >= 6 || bmin == bmax {
        return (bmin, bmax);
    }
    let (omin, omax) = op_u_bounds(func, state, v, depth);
    (bmin.max(omin), bmax.min(omax))
}

pub(super) fn sext_candidate(func: &Function, inst: Inst) -> Option<(Value, Value, u64)> {
    let data = &func.insts[inst.index()];
    let (Op::Convert(Convert::Sextend, inner), Some(res)) = (&data.op, data.result) else {
        return None;
    };
    let (in_ty, out_ty) = (func.ty(*inner), func.ty(res));
    if !in_ty.is_small_int()
        || in_ty == Type::I64
        || !out_ty.is_small_int()
        || out_ty.bits() <= in_ty.bits()
    {
        return None;
    }
    Some((res, *inner, 1_u64 << (in_ty.bits() - 1)))
}

pub(super) fn try_fold_sext_to_uext(func: &mut Function, inst: Inst, state: &RangeState) -> bool {
    let Some((_, inner, sign_bit)) = sext_candidate(func, inst) else {
        return false;
    };
    if u_bounds(func, state, inner, 0).1 >= sign_bit {
        return false;
    }
    func.insts[inst.index()].op = Op::Convert(Convert::Uextend, inner);
    true
}

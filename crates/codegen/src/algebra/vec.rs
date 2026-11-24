mod lanes;

use super::RewriteCx;
use crate::ir::{
    Binary, Convert, IntCC, Lane, Op, ShuffleMask, TargetOpId, TinyArgs, Type, Unary, Value,
};

pub(super) use lanes::{simplify_bitcast, simplify_extract, simplify_insert, simplify_shuffle};

fn is_zero_vec(cx: &RewriteCx, v: Value) -> bool {
    cx.sconst(v) == Some(0) || cx.func.is_zero_vec(&|value| cx.def(value), v)
}

fn sign_source(cx: &RewriteCx, v: Value) -> Option<Value> {
    match *cx.op(v)? {
        Op::Icmp(IntCC::SignedLessThan, x, zero)
            if cx.func.ty(x).lane().is_some_and(Lane::is_int) && is_zero_vec(cx, zero) =>
        {
            Some(x)
        }
        Op::TargetOp(TargetOpId::X64Pcmpgtb256, args) => match *args.as_slice() {
            [zero, x] if is_zero_vec(cx, zero) => Some(x),
            _ => None,
        },
        _ => None,
    }
}

/// `select(x < 0, -x, x)` on sixteen bytes as `umin(x, -x)`, the same unsigned magnitude in one byte minimum.
pub(super) fn byte_abs(cx: &RewriteCx, cond: Value, t: Value, f: Value) -> Option<Op> {
    let bytes = Type::Vec {
        lane: Lane::I8,
        count: 16,
    };
    let negated = matches!(cx.op(t)?, &Op::Unary(Unary::Ineg, x) if x == f);
    (cx.func.ty(f) == bytes && negated && sign_source(cx, cond) == Some(f)).then_some(Op::Binary(
        Binary::Umin,
        f,
        t,
    ))
}

/// Reads the lane sign bits of a compare against zero from the compared vector itself.
pub(super) fn simplify_movemask(cx: &RewriteCx, op: &Op) -> Option<Op> {
    match *op {
        Op::Reduce(crate::ir::ReduceOp::Movemask, v) => {
            sign_source(cx, v).map(|x| Op::Reduce(crate::ir::ReduceOp::Movemask, x))
        }
        Op::TargetOp(TargetOpId::X64Pmovmskb256, args) => match *args.as_slice() {
            [v] => sign_source(cx, v)
                .map(|x| Op::TargetOp(TargetOpId::X64Pmovmskb256, TinyArgs::one(x))),
            _ => None,
        },
        _ => None,
    }
}

fn movemask_of(cx: &RewriteCx, v: Value) -> Option<(bool, Value)> {
    match *cx.op(v)? {
        Op::Reduce(crate::ir::ReduceOp::Movemask, x) => Some((false, x)),
        Op::TargetOp(TargetOpId::X64Pmovmskb256, args) => match *args.as_slice() {
            [x] => Some((true, x)),
            _ => None,
        },
        _ => None,
    }
}

fn merge_movemasks(cx: &mut RewriteCx, a: Value, b: Value) -> Option<Op> {
    let ((wide, x), (other, y)) = (movemask_of(cx, a)?, movemask_of(cx, b)?);
    let ty = cx.func.ty(x);
    if wide != other || ty != cx.func.ty(y) || cx.func.ty(a) != cx.func.ty(b) {
        return None;
    }
    let both = cx.emit(ty, Op::Binary(Binary::Bor, x, y));
    Some(if wide {
        Op::TargetOp(TargetOpId::X64Pmovmskb256, TinyArgs::one(both))
    } else {
        Op::Reduce(crate::ir::ReduceOp::Movemask, both)
    })
}

fn merge_vec_bitwise_direct(cx: &mut RewriteCx, bin: Binary, a: Value, b: Value) -> Option<Op> {
    if bin == Binary::Bor
        && let Some(op) = merge_movemasks(cx, a, b)
    {
        return Some(op);
    }
    if a == b || !cx.single_use(a) || !cx.single_use(b) {
        return None;
    }
    let (Some(&Op::Extract(va, la)), Some(&Op::Extract(vb, lb))) = (cx.op(a), cx.op(b)) else {
        return None;
    };
    let ty = cx.func.ty(va);
    if la != lb || ty != cx.func.ty(vb) || !ty.lane().is_some_and(Lane::is_int) {
        return None;
    }
    let inner = emit_vec_bitwise(cx, ty, bin, (va, vb), 6);
    Some(Op::Extract(inner, la))
}

fn band_common(cx: &RewriteCx, a: Value, b: Value) -> Option<(Value, Value, Value)> {
    if !cx.single_use(a) || !cx.single_use(b) {
        return None;
    }
    let (Some(&Op::Binary(Binary::Band, a0, a1)), Some(&Op::Binary(Binary::Band, b0, b1))) =
        (cx.op(a), cx.op(b))
    else {
        return None;
    };
    let ty = cx.func.ty(a);
    for (xa, ca) in [(a0, a1), (a1, a0)] {
        for (xb, cb) in [(b0, b1), (b1, b0)] {
            if ca == cb
                || cx
                    .uconst(ca, ty)
                    .is_some_and(|ka| cx.uconst(cb, ty) == Some(ka))
            {
                return Some((xa, xb, ca));
            }
        }
    }
    None
}

/// Fuses scalar bitwise ops over movemasks or single-use lane extracts (directly or under a shared `Band` mask) into one vector op followed by one extract/movemask.
pub(super) fn merge_vec_bitwise(cx: &mut RewriteCx, bin: Binary, a: Value, b: Value) -> Option<Op> {
    if let Some(op) = merge_vec_bitwise_direct(cx, bin, a, b) {
        return Some(op);
    }
    if matches!(bin, Binary::Bor | Binary::Bxor)
        && let Some((xa, xb, common)) = band_common(cx, a, b)
        && let Some(inner_op) = merge_vec_bitwise_direct(cx, bin, xa, xb)
    {
        let merged = cx.emit(cx.func.ty(a), inner_op);
        return Some(Op::Binary(Binary::Band, merged, common));
    }
    None
}

fn same_vec_shift(cx: &RewriteCx, ty: Type, sh1: Value, sh2: Value) -> bool {
    sh1 == sh2
        || ty.vec_shape().is_some_and(|(lane, count)| {
            splat_int_imm(cx, sh1, lane, usize::from(count))
                .is_some_and(|k| splat_int_imm(cx, sh2, lane, usize::from(count)) == Some(k))
        })
}

fn sink_vec_bitwise_inner(
    cx: &mut RewriteCx,
    res_ty: Type,
    bin: Binary,
    (a, b): (Value, Value),
    depth: u8,
) -> Option<Op> {
    if a == b {
        return None;
    }
    if let (Some(&Op::Convert(Convert::Bitcast, x)), Some(&Op::Convert(Convert::Bitcast, y))) =
        (cx.op(a), cx.op(b))
    {
        let ty = cx.func.ty(x);
        if ty == cx.func.ty(y)
            && ty.lane().is_some_and(Lane::is_int)
            && ty.bytes() == res_ty.bytes()
        {
            return Some(Op::Convert(
                Convert::Bitcast,
                emit_vec_bitwise(cx, ty, bin, (x, y), depth),
            ));
        }
    }
    if !cx.single_use(a) || !cx.single_use(b) {
        return None;
    }
    match (cx.op(a), cx.op(b)) {
        (Some(&Op::Convert(Convert::Ireduce, x)), Some(&Op::Convert(Convert::Ireduce, y))) => {
            let ty = cx.func.ty(x);
            (ty == cx.func.ty(y) && ty.lane().is_some_and(Lane::is_int)).then(|| {
                Op::Convert(
                    Convert::Ireduce,
                    emit_vec_bitwise(cx, ty, bin, (x, y), depth),
                )
            })
        }
        (
            Some(&Op::Binary(sop @ (Binary::Ushr | Binary::Ishl | Binary::Sshr), x, sh1)),
            Some(&Op::Binary(sop2, y, sh2)),
        ) if sop == sop2 => {
            let ty = cx.func.ty(x);
            (ty == cx.func.ty(y)
                && ty.lane().is_some_and(Lane::is_int)
                && same_vec_shift(cx, ty, sh1, sh2))
            .then(|| Op::Binary(sop, emit_vec_bitwise(cx, ty, bin, (x, y), depth), sh1))
        }
        _ => None,
    }
}

fn emit_vec_bitwise(
    cx: &mut RewriteCx,
    ty: Type,
    bin: Binary,
    (a, b): (Value, Value),
    depth: u8,
) -> Value {
    if a == b {
        return if bin == Binary::Bxor {
            cx.emit(ty, Op::Const(0))
        } else {
            a
        };
    }
    let op = if depth > 0 {
        sink_vec_bitwise_inner(cx, ty, bin, (a, b), depth - 1)
    } else {
        None
    }
    .unwrap_or(Op::Binary(bin, a, b));
    cx.emit(ty, op)
}

fn sink_bitcast(
    cx: &mut RewriteCx,
    res_v: Value,
    bin: Binary,
    (a, b): (Value, Value),
) -> Option<Op> {
    sink_vec_bitwise_inner(cx, cx.func.ty(res_v), bin, (a, b), 6)
}

fn splat_scalar(cx: &RewriteCx, v: Value, count: usize) -> Option<Value> {
    let inst = cx.def(v)?;
    match &cx.func.insts[inst.index()].op {
        &Op::Insert(_, scalar, 0) if count == 1 => Some(scalar),
        Op::Shuffle(a, b, ShuffleMask::Const(mask)) => {
            let head = mask[0];
            if usize::from(head) >= 2 * count || mask[..count].iter().any(|&i| i != head) {
                return None;
            }
            let src = if usize::from(head) < count { *a } else { *b };
            let target = head % (count as u8);
            match cx.op(src)? {
                &Op::Insert(_, scalar, at) if at == target => Some(scalar),
                _ => None,
            }
        }
        _ => None,
    }
}

fn splat_int_imm(cx: &RewriteCx, v: Value, lane: Lane, count: usize) -> Option<u64> {
    if !lane.is_int() {
        return None;
    }
    if is_zero_vec(cx, v) {
        return Some(0);
    }
    if let Some(&Op::Unary(Unary::Bnot, z)) = cx.op(v)
        && is_zero_vec(cx, z)
    {
        return Some(lane.to_type().unsigned_max());
    }
    cx.uconst(splat_scalar(cx, v, count)?, lane.to_type())
}

fn is_all_ones_vec(cx: &RewriteCx, mut v: Value) -> bool {
    for _ in 0..4 {
        if let Some(&Op::Unary(Unary::Bnot, z)) = cx.op(v)
            && is_zero_vec(cx, z)
        {
            return true;
        }
        if let Some((lane, count)) = cx.func.ty(v).vec_shape()
            && lane.is_int()
            && splat_int_imm(cx, v, lane, usize::from(count)) == Some(lane.to_type().unsigned_max())
        {
            return true;
        }
        let Some(&Op::Convert(Convert::Bitcast, inner)) = cx.op(v) else {
            return false;
        };
        v = inner;
    }
    false
}

fn alias_if_diff(cx: &mut RewriteCx, res: Option<Value>, res_v: Value, target: Value) {
    if target != res_v {
        cx.alias(res, target);
    }
}

fn simplify_vec_band_bor(
    cx: &mut RewriteCx,
    res: Option<Value>,
    res_v: Value,
    is_band: bool,
    (a, b): (Value, Value),
) -> Option<Op> {
    let (za, zb) = (is_zero_vec(cx, a), is_zero_vec(cx, b));
    let (oa, ob) = (is_all_ones_vec(cx, a), is_all_ones_vec(cx, b));
    let bnot_pair = cx.op(b) == Some(&Op::Unary(Unary::Bnot, a))
        || cx.op(a) == Some(&Op::Unary(Unary::Bnot, b));
    if is_band {
        if za || zb || bnot_pair {
            return Some(Op::Const(0));
        }
        if a == b || ob {
            alias_if_diff(cx, res, res_v, a);
        } else if oa {
            alias_if_diff(cx, res, res_v, b);
        }
    } else if zb || a == b || oa {
        alias_if_diff(cx, res, res_v, a);
    } else if za || ob {
        alias_if_diff(cx, res, res_v, b);
    }
    None
}

fn simplify_vec_bxor(
    cx: &mut RewriteCx,
    res: Option<Value>,
    res_v: Value,
    a: Value,
    b: Value,
) -> Option<Op> {
    if a == b {
        return Some(Op::Const(0));
    }
    if is_zero_vec(cx, b) {
        alias_if_diff(cx, res, res_v, a);
    } else if is_zero_vec(cx, a) {
        alias_if_diff(cx, res, res_v, b);
    } else if is_all_ones_vec(cx, b) {
        return Some(Op::Unary(Unary::Bnot, a));
    } else if is_all_ones_vec(cx, a) {
        return Some(Op::Unary(Unary::Bnot, b));
    }
    None
}

fn simplify_vec_minmax(
    cx: &mut RewriteCx,
    res: Option<Value>,
    res_v: Value,
    bin: Binary,
    (a, b): (Value, Value),
) -> Option<Op> {
    if a == b {
        alias_if_diff(cx, res, res_v, a);
        return None;
    }
    let (za, zb) = (is_zero_vec(cx, a), is_zero_vec(cx, b));
    let (oa, ob) = (is_all_ones_vec(cx, a), is_all_ones_vec(cx, b));
    if bin == Binary::Umin {
        if za || zb {
            return Some(Op::Const(0));
        }
        if ob {
            alias_if_diff(cx, res, res_v, a);
        } else if oa {
            alias_if_diff(cx, res, res_v, b);
        }
    } else if bin == Binary::Umax {
        if zb || oa {
            alias_if_diff(cx, res, res_v, a);
        } else if za || ob {
            alias_if_diff(cx, res, res_v, b);
        }
    }
    None
}

fn simplify_vec_sat(
    cx: &mut RewriteCx,
    res: Option<Value>,
    res_v: Value,
    bin: Binary,
    (a, b): (Value, Value),
) -> Option<Op> {
    let (za, zb) = (is_zero_vec(cx, a), is_zero_vec(cx, b));
    if matches!(bin, Binary::SaddSat | Binary::UaddSat) {
        let u = bin == Binary::UaddSat;
        if zb || (u && is_all_ones_vec(cx, a)) {
            alias_if_diff(cx, res, res_v, a);
        } else if za || (u && is_all_ones_vec(cx, b)) {
            alias_if_diff(cx, res, res_v, b);
        }
        return None;
    }
    let u = bin == Binary::UsubSat;
    if a == b || (u && (za || is_all_ones_vec(cx, b))) {
        return Some(Op::Const(0));
    }
    if zb {
        alias_if_diff(cx, res, res_v, a);
    }
    None
}

fn simplify_vec_add_sub(
    cx: &mut RewriteCx,
    res: Option<Value>,
    res_v: Value,
    bin: Binary,
    (a, b): (Value, Value),
) -> Option<Op> {
    let (za, zb) = (is_zero_vec(cx, a), is_zero_vec(cx, b));
    if bin == Binary::Iadd {
        if zb {
            alias_if_diff(cx, res, res_v, a);
        } else if za {
            alias_if_diff(cx, res, res_v, b);
        } else if let Some(&Op::Unary(Unary::Ineg, x)) = cx.op(b) {
            return Some(if x == a {
                Op::Const(0)
            } else {
                Op::Binary(Binary::Isub, a, x)
            });
        } else if let Some(&Op::Unary(Unary::Ineg, x)) = cx.op(a) {
            return Some(if x == b {
                Op::Const(0)
            } else {
                Op::Binary(Binary::Isub, b, x)
            });
        }
        return None;
    }
    if zb {
        alias_if_diff(cx, res, res_v, a);
        return None;
    }
    if a == b {
        return Some(Op::Const(0));
    }
    if za {
        return Some(Op::Unary(Unary::Ineg, b));
    }
    match cx.op(b) {
        Some(&Op::Unary(Unary::Ineg, x)) => Some(Op::Binary(Binary::Iadd, a, x)),
        _ => None,
    }
}

fn simplify_vec_mul_shift(
    cx: &mut RewriteCx,
    res: Option<Value>,
    res_v: Value,
    bin: Binary,
    (a, b): (Value, Value),
) -> Option<Op> {
    let (lane, count) = cx.func.ty(res_v).vec_shape()?;
    let c = usize::from(count);
    let (za, zb) = (is_zero_vec(cx, a), is_zero_vec(cx, b));
    if bin == Binary::Imul {
        if za
            || zb
            || splat_int_imm(cx, a, lane, c) == Some(0)
            || splat_int_imm(cx, b, lane, c) == Some(0)
        {
            return Some(Op::Const(0));
        }
        if splat_int_imm(cx, b, lane, c) == Some(1) {
            alias_if_diff(cx, res, res_v, a);
        } else if splat_int_imm(cx, a, lane, c) == Some(1) {
            alias_if_diff(cx, res, res_v, b);
        }
        return None;
    }
    if za {
        return Some(Op::Const(0));
    }
    if zb || splat_int_imm(cx, b, lane, c).is_some_and(|imm| (imm as u32) & (lane.bits() - 1) == 0)
    {
        alias_if_diff(cx, res, res_v, a);
    }
    None
}

/// Simplifies vector binary algebraic, zero, all-ones, shift, and saturation identities.
pub(super) fn simplify_vec_binary(
    cx: &mut RewriteCx,
    res: Option<Value>,
    bin: Binary,
    a: Value,
    b: Value,
) -> Option<Op> {
    let res_v = res?;
    if cx.func.ty(res_v).vec_shape().is_none() {
        return None;
    }
    match bin {
        Binary::Band | Binary::Bor | Binary::Bxor
            if a != b
                && let Some(op) = sink_bitcast(cx, res_v, bin, (a, b)) =>
        {
            Some(op)
        }
        Binary::Band | Binary::Bor => {
            simplify_vec_band_bor(cx, res, res_v, bin == Binary::Band, (a, b))
        }
        Binary::Bxor => simplify_vec_bxor(cx, res, res_v, a, b),
        Binary::Iadd | Binary::Isub => simplify_vec_add_sub(cx, res, res_v, bin, (a, b)),
        Binary::Imul | Binary::Ishl | Binary::Ushr | Binary::Sshr => {
            simplify_vec_mul_shift(cx, res, res_v, bin, (a, b))
        }
        Binary::SaddSat | Binary::UaddSat | Binary::SsubSat | Binary::UsubSat => {
            simplify_vec_sat(cx, res, res_v, bin, (a, b))
        }
        Binary::Umin | Binary::Umax | Binary::Smin | Binary::Smax => {
            simplify_vec_minmax(cx, res, res_v, bin, (a, b))
        }
        _ => None,
    }
}

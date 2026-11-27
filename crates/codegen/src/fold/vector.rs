mod bytes;
mod canon;

use super::{Fold, extend};
use crate::ir::{
    Binary, Convert, Function, Inst, IntCC, Op, ReduceOp, ShuffleMask, Type, Unary, Value,
};

pub(super) fn rotate(ty: Type, value: u64, left: u32) -> i64 {
    match ty {
        Type::I8 => i64::from((value as u8).rotate_left(left)),
        Type::I16 => i64::from((value as u16).rotate_left(left)),
        Type::I32 => i64::from((value as u32).rotate_left(left)),
        _ => value.rotate_left(left) as i64,
    }
}

pub(super) fn divide(op: Binary, ty: Type, a: i64, b: i64) -> Option<i64> {
    let signed = matches!(op, Binary::Sdiv | Binary::Srem);
    let (x, y) = (extend(a, ty, signed), extend(b, ty, signed));
    if y == 0 || (signed && y == -1 && x == -1_i64 << (ty.bits() - 1)) {
        return None;
    }
    Some(match op {
        Binary::Sdiv => x / y,
        Binary::Srem => x % y,
        Binary::Udiv => (x as u64 / y as u64) as i64,
        _ => (x as u64 % y as u64) as i64,
    })
}

pub(super) fn sat_binary(op: Binary, ty: Type, sa: i64, sb: i64, ua: u64, ub: u64) -> Option<i64> {
    let bits = ty.bits();
    let (smin, smax) = (-(1_i128 << (bits - 1)), (1_i128 << (bits - 1)) - 1);
    Some(match op {
        Binary::SaddSat => (i128::from(sa) + i128::from(sb)).clamp(smin, smax) as i64,
        Binary::SsubSat => (i128::from(sa) - i128::from(sb)).clamp(smin, smax) as i64,
        Binary::UaddSat => {
            (u128::from(ua) + u128::from(ub)).min(u128::from(ty.unsigned_max())) as i64
        }
        Binary::UsubSat => ua.saturating_sub(ub) as i64,
        Binary::AvgRound => ((u128::from(ua) + u128::from(ub) + 1) >> 1) as i64,
        _ => return None,
    })
}

fn clamp_narrow(src: Type, dst: Type, signed: bool, raw: i64) -> i64 {
    let val = extend(raw, src, true);
    let clamped = if signed {
        let max = (1_i64 << (dst.bits() - 1)) - 1;
        val.clamp(-max - 1, max)
    } else {
        val.clamp(0, dst.unsigned_max() as i64)
    };
    extend(clamped, dst, true)
}

pub(super) fn narrow_sat(
    func: &Function,
    dst: Type,
    signed: bool,
    x: Value,
    known: &dyn Fn(Value) -> Option<i64>,
) -> Option<i64> {
    let src = func.ty(x);
    if !src.is_small_int() || dst.bytes() >= src.bytes() {
        return None;
    }
    Some(clamp_narrow(src, dst, signed, known(x)?))
}

fn recast(func: &Function, inst: Inst) -> Option<(Value, Value)> {
    match (
        &func.insts[inst.index()].op,
        func.insts[inst.index()].result,
    ) {
        (&Op::Convert(Convert::Bitcast, src), Some(res))
            if func.ty(src).bytes() == func.ty(res).bytes() && func.ty(src) != func.ty(res) =>
        {
            Some((src, res))
        }
        _ => None,
    }
}

pub(super) fn fold_bitcast_loads(func: &mut Function, defs: &[Option<Inst>]) -> bool {
    if !func.blocks.iter().flat_map(|b| &b.insts).any(|&inst| {
        recast(func, inst).is_some_and(|(src, _)| {
            defs.get(src.index())
                .copied()
                .flatten()
                .is_some_and(|src_inst| matches!(func.insts[src_inst.index()].op, Op::Load(..)))
        })
    }) {
        return canon::fold_const_vectors(func);
    }
    let uses = func.use_counts();
    let mut remap: Vec<Option<Value>> = vec![None; func.types.len()];
    let mut dropped = vec![false; func.insts.len()];
    for inst in func.blocks.iter().flat_map(|b| &b.insts) {
        if let Some((src, res)) = recast(func, *inst)
            && uses[src.index()] == 1
            && let Some(src_inst) = defs.get(src.index()).copied().flatten()
            && matches!(func.insts[src_inst.index()].op, Op::Load(..))
        {
            func.types[src.index()] = func.ty(res);
            remap[res.index()] = Some(src);
            dropped[inst.index()] = true;
        }
    }
    let mut changed = !remap.iter().all(Option::is_none);
    if changed {
        for block in &mut func.blocks {
            block.insts.retain(|inst| !dropped[inst.index()]);
            for &inst in &block.insts {
                func.insts[inst.index()]
                    .op
                    .map_operands(|v| remap[v.index()].unwrap_or(v));
            }
        }
    }
    changed |= canon::fold_const_vectors(func);
    changed
}

fn trace_extract(fold: &Fold, func: &Function, mut vec: Value, mut idx: u8) -> Option<Value> {
    for _ in 0..16 {
        match *fold.def(func, vec)? {
            Op::Convert(Convert::Bitcast, inner)
                if idx == 0
                    && func
                        .ty(vec)
                        .vec_shape()
                        .is_some_and(|(l, _)| func.ty(inner) == l.to_type()) =>
            {
                return Some(inner);
            }
            Op::Insert(_, val, at) if at == idx => return Some(val),
            Op::Insert(prev, _, _) => vec = prev,
            Op::Shuffle(a, b, ShuffleMask::Const(mask)) => {
                let count = func.ty(a).vec_shape()?.1;
                let sel = *mask.get(usize::from(idx))?;
                if sel < count {
                    (vec, idx) = (a, sel);
                } else if sel < 2 * count {
                    (vec, idx) = (b, sel - count);
                } else {
                    return None;
                }
            }
            _ => return None,
        }
    }
    None
}

fn splat_shuffle_ident(fold: &Fold, func: &Function, v: Value) -> bool {
    let Some((&Op::Shuffle(a, b, ShuffleMask::Const(mask)), Some((_, count)))) =
        fold.def(func, v).map(|op| (op, func.ty(v).vec_shape()))
    else {
        return false;
    };
    let c = usize::from(count);
    let head = mask[0];
    let same = fold.find(a) == fold.find(b);
    usize::from(head) < 2 * c
        && mask[..c]
            .iter()
            .all(|&i| i == head || (same && i % count == head % count))
}

fn shuffle_identity(
    fold: &Fold,
    func: &Function,
    a: Value,
    b: Value,
    mask: [u8; 16],
) -> Option<Value> {
    let (_, count) = func.ty(a).vec_shape()?;
    let c = usize::from(count);
    let same_ab = fold.find(a) == fold.find(b);
    if (0..count)
        .all(|i| mask[usize::from(i)] == i || (same_ab && mask[usize::from(i)] == count + i))
    {
        return Some(a);
    }
    if !same_ab && (0..count).all(|i| mask[usize::from(i)] == count + i) {
        return Some(b);
    }
    if mask[..c]
        .iter()
        .all(|&i| i < count || (same_ab && i < 2 * count))
        && splat_shuffle_ident(fold, func, a)
    {
        return Some(a);
    }
    (!same_ab
        && mask[..c].iter().all(|&i| i >= count && i < 2 * count)
        && splat_shuffle_ident(fold, func, b))
    .then_some(b)
}

pub(super) fn vec_identity(fold: &Fold, func: &Function, op: &Op) -> Option<Value> {
    match *op {
        Op::Extract(vec, idx) => trace_extract(fold, func, vec, idx),
        Op::Shuffle(a, b, ShuffleMask::Const(mask)) => shuffle_identity(fold, func, a, b, mask),
        _ => None,
    }
}

fn vec_zero_binary(func: &Function, fold: &Fold, kind: Binary, x: Value, y: Value) -> Option<Op> {
    match kind {
        Binary::SaddSat
        | Binary::UaddSat
        | Binary::SsubSat
        | Binary::UsubSat
        | Binary::AvgRound
            if is_zero_bits(func, fold, x) && is_zero_bits(func, fold, y) =>
        {
            Some(Op::Const(0))
        }
        _ if !func.ty(x).is_vector() => None,
        Binary::Band | Binary::Imul | Binary::Umin
            if is_zero_bits(func, fold, x) || is_zero_bits(func, fold, y) =>
        {
            Some(Op::Const(0))
        }
        Binary::Bxor | Binary::Isub | Binary::SsubSat | Binary::UsubSat
            if fold.find(x) == fold.find(y) =>
        {
            Some(Op::Const(0))
        }
        Binary::UsubSat | Binary::Ishl | Binary::Ushr | Binary::Sshr
            if is_zero_bits(func, fold, x) =>
        {
            Some(Op::Const(0))
        }
        _ => None,
    }
}

fn vec_zero_icmp(func: &Function, fold: &Fold, cc: IntCC, x: Value, y: Value) -> Option<Op> {
    if !func.ty(x).is_vector() {
        return None;
    }
    let eq_false = fold.find(x) == fold.find(y)
        && matches!(
            cc,
            IntCC::NotEqual
                | IntCC::SignedLessThan
                | IntCC::SignedGreaterThan
                | IntCC::UnsignedLessThan
                | IntCC::UnsignedGreaterThan
        );
    let u_false = (cc == IntCC::UnsignedLessThan && is_zero_bits(func, fold, y))
        || (cc == IntCC::UnsignedGreaterThan && is_zero_bits(func, fold, x));
    (eq_false || u_false).then_some(Op::Const(0))
}

fn vec_zero_other(func: &Function, fold: &Fold, op: &Op) -> Option<Op> {
    match *op {
        Op::Unary(Unary::Ineg | Unary::Bswap | Unary::Bitrev | Unary::Popcnt | Unary::Fabs, x)
            if func.ty(x).is_vector() && is_zero_bits(func, fold, x) =>
        {
            Some(Op::Const(0))
        }
        Op::Select(_, x, y)
            if func.ty(x).is_vector()
                && is_zero_bits(func, fold, x)
                && is_zero_bits(func, fold, y) =>
        {
            Some(Op::Const(0))
        }
        Op::Icmp(cc, x, y) => vec_zero_icmp(func, fold, cc, x, y),
        Op::Convert(
            Convert::Sextend
            | Convert::Uextend
            | Convert::Ireduce
            | Convert::NarrowSat { .. }
            | Convert::FromSint
            | Convert::FromUint
            | Convert::ToSintSat
            | Convert::ToUintSat,
            x,
        ) if func.ty(x).is_vector() && is_zero_bits(func, fold, x) => Some(Op::Const(0)),
        _ => None,
    }
}

fn vec_zero_bitcast(func: &Function, fold: &Fold, x: Value, res: Option<Value>) -> Option<Op> {
    let ty = res.map(|r| func.ty(r))?;
    if (ty.is_vector() || ty.is_small_int() || ty.is_float()) && is_zero_bits(func, fold, x) {
        return Some(Op::Const(0));
    }
    match fold.def(func, x) {
        Some(&Op::Convert(Convert::Bitcast, v)) if func.ty(fold.find(v)) != ty => {
            Some(Op::Convert(Convert::Bitcast, fold.find(v)))
        }
        _ => None,
    }
}

pub(super) fn vec_zero_fold(
    func: &Function,
    fold: &Fold,
    op: &Op,
    res: Option<Value>,
) -> Option<Op> {
    match *op {
        Op::Convert(Convert::Bitcast, x) => vec_zero_bitcast(func, fold, x, res),
        Op::Shuffle(a, b, ShuffleMask::Const(mask)) => {
            let count = func.ty(a).vec_shape()?.1;
            let (za, zb) = (is_zero_bits(func, fold, a), is_zero_bits(func, fold, b));
            mask[..usize::from(count)]
                .iter()
                .all(|&i| if i < count { za } else { zb })
                .then_some(Op::Const(0))
        }
        Op::Insert(vec, scalar, _) => (is_zero_bits(func, fold, vec)
            && is_zero_bits(func, fold, scalar))
        .then_some(Op::Const(0)),
        Op::Extract(vec, _) => is_zero_bits(func, fold, vec).then_some(Op::Const(0)),
        Op::Reduce(kind, vec) if !matches!(kind, ReduceOp::Fmin | ReduceOp::Fmax) => {
            is_zero_bits(func, fold, vec).then_some(Op::Const(0))
        }
        Op::Binary(kind, x, y) => vec_zero_binary(func, fold, kind, x, y),
        _ => vec_zero_other(func, fold, op),
    }
}

fn is_zero_bits(func: &Function, fold: &Fold, val: Value) -> bool {
    let val = fold.find(val);
    fold.known(val) == Some(0)
        || matches!(fold.def(func, val), Some(&Op::Const(0)))
        || matches!(
            fold.def(func, val),
            Some(&Op::Convert(Convert::Bitcast, z))
                if fold.known(z) == Some(0) || matches!(fold.def(func, fold.find(z)), Some(&Op::Const(0)))
        )
}

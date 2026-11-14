use super::{Fold, defs, extend};
use crate::ir::{Binary, Convert, Function, Op, ReduceOp, ShuffleMask, Type, Value};

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
        Binary::UaddSat => (u128::from(ua) + u128::from(ub)).min(u128::from(ty.unsigned_max())) as i64,
        Binary::UsubSat => ua.saturating_sub(ub) as i64,
        Binary::AvgRound => ((u128::from(ua) + u128::from(ub) + 1) >> 1) as i64,
        _ => return None,
    })
}

pub(super) fn narrow_sat(func: &Function, dst: Type, signed: bool, x: Value, known: &dyn Fn(Value) -> Option<i64>) -> Option<i64> {
    let src = func.ty(x);
    if !src.is_small_int() || dst.bytes() >= src.bytes() {
        return None;
    }
    let val = extend(known(x)?, src, true);
    let clamped = if signed {
        let max = (1_i64 << (dst.bits() - 1)) - 1;
        val.clamp(-max - 1, max)
    } else {
        val.clamp(0, dst.unsigned_max() as i64)
    };
    Some(extend(clamped, dst, true))
}

pub(super) fn fold_bitcast_loads(func: &mut Function) -> bool {
    let uses = func.use_counts();
    let defs = defs(func);
    let mut remap: Vec<Option<Value>> = vec![None; func.types.len()];
    let mut dropped = vec![false; func.insts.len()];
    for inst in func.blocks.iter().flat_map(|b| &b.insts) {
        let data = &func.insts[inst.index()];
        if let (Op::Convert(Convert::Bitcast, src), Some(res)) = (&data.op, data.result)
            && uses[src.index()] == 1
            && func.ty(*src).bytes() == func.ty(res).bytes()
            && func.ty(*src) != func.ty(res)
            && let Some(src_inst) = defs[src.index()]
            && matches!(func.insts[src_inst.index()].op, Op::Load(..))
        {
            func.types[src.index()] = func.ty(res);
            remap[res.index()] = Some(*src);
            dropped[inst.index()] = true;
        }
    }
    if remap.iter().all(Option::is_none) {
        return false;
    }
    for block in &mut func.blocks {
        block.insts.retain(|inst| !dropped[inst.index()]);
        for &inst in &block.insts {
            func.insts[inst.index()].op.map_operands(|v| remap[v.index()].unwrap_or(v));
        }
    }
    true
}

pub(super) fn vec_identity(fold: &Fold, func: &Function, op: &Op) -> Option<Value> {
    match *op {
        Op::Extract(vec, idx) => match *fold.def(func, vec)? {
            Op::Insert(_, val, at) if at == idx => Some(val),
            _ => None,
        },
        Op::Shuffle(a, b, ShuffleMask::Const(mask)) => {
            let (_, count) = func.ty(a).vec_shape()?;
            if (0..count).all(|i| mask[usize::from(i)] == i) {
                Some(a)
            } else if (0..count).all(|i| mask[usize::from(i)] == count + i) {
                Some(b)
            } else {
                None
            }
        }
        _ => None,
    }
}

pub(super) fn vec_zero_fold(func: &Function, fold: &Fold, op: &Op, res: Option<Value>) -> Option<Op> {
    match *op {
        Op::Convert(Convert::Bitcast, x) => match (res.map(|r| func.ty(r)), fold.def(func, x)) {
            (Some(ty), Some(&Op::Convert(Convert::Bitcast, v))) if func.ty(fold.find(v)) != ty => {
                Some(Op::Convert(Convert::Bitcast, fold.find(v)))
            }
            _ => None,
        },
        Op::Shuffle(a, b, ShuffleMask::Const(mask)) => {
            let count = func.ty(a).vec_shape()?.1;
            let (za, zb) = (is_zero_bits(func, fold, a), is_zero_bits(func, fold, b));
            mask[..usize::from(count)].iter().all(|&i| if i < count { za } else { zb }).then_some(Op::Const(0))
        }
        Op::Insert(vec, scalar, _) => (is_zero_bits(func, fold, vec) && is_zero_bits(func, fold, scalar)).then_some(Op::Const(0)),
        Op::Extract(vec, _) => is_zero_bits(func, fold, vec).then_some(Op::Const(0)),
        Op::Reduce(kind, vec) if !matches!(kind, ReduceOp::Fmin | ReduceOp::Fmax) => is_zero_bits(func, fold, vec).then_some(Op::Const(0)),
        Op::Binary(Binary::SaddSat | Binary::UaddSat | Binary::SsubSat | Binary::UsubSat | Binary::AvgRound, x, y) => {
            (is_zero_bits(func, fold, x) && is_zero_bits(func, fold, y)).then_some(Op::Const(0))
        }
        Op::Convert(Convert::NarrowSat { .. }, x) if func.ty(x).is_vector() => is_zero_bits(func, fold, x).then_some(Op::Const(0)),
        _ => None,
    }
}

fn is_zero_bits(func: &Function, fold: &Fold, val: Value) -> bool {
    let val = fold.find(val);
    fold.known(val) == Some(0)
        || matches!(fold.def(func, val), Some(&Op::Const(0)))
        || matches!(fold.def(func, val), Some(&Op::Convert(Convert::Bitcast, z)) if fold.known(z) == Some(0) || matches!(fold.def(func, fold.find(z)), Some(&Op::Const(0))))
}

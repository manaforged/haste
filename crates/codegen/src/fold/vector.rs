use super::Fold;
use crate::ir::{Convert, Function, Op, ReduceOp, ShuffleMask, Value};

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

pub(super) fn vec_zero_fold(func: &Function, fold: &Fold, op: &Op) -> Option<Op> {
    match *op {
        Op::Shuffle(a, b, ShuffleMask::Const(mask)) => {
            let count = func.ty(a).vec_shape()?.1;
            let (za, zb) = (is_zero_bits(func, fold, a), is_zero_bits(func, fold, b));
            mask[..usize::from(count)].iter().all(|&i| if i < count { za } else { zb }).then_some(Op::Const(0))
        }
        Op::Insert(vec, scalar, _) => (is_zero_bits(func, fold, vec) && is_zero_bits(func, fold, scalar)).then_some(Op::Const(0)),
        Op::Extract(vec, _) => is_zero_bits(func, fold, vec).then_some(Op::Const(0)),
        Op::Reduce(kind, vec) if !matches!(kind, ReduceOp::Fmin | ReduceOp::Fmax) => is_zero_bits(func, fold, vec).then_some(Op::Const(0)),
        _ => None,
    }
}

fn is_zero_bits(func: &Function, fold: &Fold, val: Value) -> bool {
    let val = fold.find(val);
    fold.known(val) == Some(0)
        || matches!(fold.def(func, val), Some(&Op::Const(0)))
        || matches!(fold.def(func, val), Some(&Op::Convert(Convert::Bitcast, z)) if fold.known(z) == Some(0) || matches!(fold.def(func, fold.find(z)), Some(&Op::Const(0))))
}

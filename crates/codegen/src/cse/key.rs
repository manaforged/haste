use crate::fold::extend;
use crate::ir::{
    Binary, Convert, FloatCC, IntCC, Op, ReduceOp, ShuffleMask, StackSlot, TargetOpId, Type, Unary,
    Value,
};
use crate::merge::find;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum Operand {
    Val(Value),
    Imm(Type, i64),
}

pub(super) fn order_pair(a: Operand, b: Operand) -> (Operand, Operand) {
    if a > b { (b, a) } else { (a, b) }
}

fn is_commutative_binary(op: Binary) -> bool {
    matches!(
        op,
        Binary::Iadd
            | Binary::Imul
            | Binary::Umulhi
            | Binary::Smulhi
            | Binary::Smin
            | Binary::Smax
            | Binary::Umin
            | Binary::Umax
            | Binary::SaddSat
            | Binary::UaddSat
            | Binary::AvgRound
            | Binary::Fmin
            | Binary::Fmax
            | Binary::Band
            | Binary::Bor
            | Binary::Bxor
    )
}

fn canonical_bound(
    cc: IntCC,
    v: i128,
    min: i128,
    max: i128,
    (lt, le, gt, ge): (IntCC, IntCC, IntCC, IntCC),
) -> (IntCC, i128) {
    if cc == lt {
        if v == min + 1 {
            return (IntCC::Equal, min);
        }
        if v == max {
            return (IntCC::NotEqual, max);
        }
    } else if cc == le {
        if v == min {
            return (IntCC::Equal, min);
        }
        if v < max - 1 {
            return (lt, v + 1);
        }
        if v == max - 1 {
            return (IntCC::NotEqual, max);
        }
    } else if cc == gt {
        if v == min {
            return (IntCC::NotEqual, min);
        }
        if v < max - 1 {
            return (ge, v + 1);
        }
        if v == max - 1 {
            return (IntCC::Equal, max);
        }
    } else if cc == ge {
        if v == min + 1 {
            return (IntCC::NotEqual, min);
        }
        if v == max {
            return (IntCC::Equal, max);
        }
    }
    (cc, v)
}

pub(super) fn canonical_icmp(cc: IntCC, x: Operand, y: Operand) -> (IntCC, Operand, Operand) {
    let (cc, a, b) = if x > y {
        (cc.swapped(), y, x)
    } else {
        (cc, x, y)
    };
    let (Operand::Val(_), Operand::Imm(ity, imm)) = (a, b) else {
        return (cc, a, b);
    };
    if !ity.is_small_int() {
        return (cc, a, b);
    }
    let bits = ity.bits();
    let (new_cc, new_v) = if cc.is_signed() {
        let min = -(1_i128 << (bits - 1));
        let max = (1_i128 << (bits - 1)) - 1;
        let v = i128::from(extend(imm, ity, true));
        canonical_bound(
            cc,
            v,
            min,
            max,
            (
                IntCC::SignedLessThan,
                IntCC::SignedLessThanOrEqual,
                IntCC::SignedGreaterThan,
                IntCC::SignedGreaterThanOrEqual,
            ),
        )
    } else {
        let min = 0_i128;
        let max = (1_i128 << bits) - 1;
        let v = i128::from(extend(imm, ity, false) as u64);
        canonical_bound(
            cc,
            v,
            min,
            max,
            (
                IntCC::UnsignedLessThan,
                IntCC::UnsignedLessThanOrEqual,
                IntCC::UnsignedGreaterThan,
                IntCC::UnsignedGreaterThanOrEqual,
            ),
        )
    };
    (
        new_cc,
        a,
        Operand::Imm(ity, extend(new_v as i64, ity, false)),
    )
}

fn flip_float_cc(cc: FloatCC) -> FloatCC {
    match cc {
        FloatCC::Equal | FloatCC::NotEqual => cc,
        FloatCC::LessThan => FloatCC::GreaterThan,
        FloatCC::GreaterThan => FloatCC::LessThan,
        FloatCC::LessThanOrEqual => FloatCC::GreaterThanOrEqual,
        FloatCC::GreaterThanOrEqual => FloatCC::LessThanOrEqual,
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum ShuffleKey {
    Const([u8; 16]),
    Dyn(Operand),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum Key {
    Const(Type, i64),
    Unary(Unary, Operand),
    Binary(Binary, Operand, Operand),
    Convert(Convert, Type, Operand),
    Icmp(IntCC, Type, Operand, Operand),
    Fits(Binary, bool, Operand, Operand),
    Fcmp(FloatCC, Type, Operand, Operand),
    Shuffle(Type, Operand, Operand, ShuffleKey),
    Extract(Operand, u8),
    Insert(Operand, Operand, u8),
    Reduce(ReduceOp, Operand),
    TargetOp(TargetOpId, [Option<Operand>; 3]),
    Select(Operand, Operand, Operand),
    Fma(Operand, Operand, Operand),
    StackAddr(StackSlot, i32),
    SlotAddr(u32),
    Symbol(u32),
    ThreadCell(Operand, Operand, Operand),
}

pub(super) fn operand(
    consts: &[Option<i64>],
    types: &[Type],
    subst: &[Value],
    value: Value,
) -> Operand {
    let v = find(subst, value);
    match consts[v.index()] {
        Some(imm) => Operand::Imm(types[v.index()], imm),
        None => Operand::Val(v),
    }
}

pub(super) fn unwrap_bool_neg(
    bool_neg: &[Option<Value>],
    subst: &[Value],
    c: Value,
    x: Value,
    y: Value,
) -> (Value, Value, Value) {
    let mut cond = find(subst, c);
    let mut inv = false;
    while let Some(inner) = bool_neg[cond.index()] {
        cond = find(subst, inner);
        inv = !inv;
    }
    if inv { (cond, y, x) } else { (cond, x, y) }
}

pub(super) fn key(
    op: &Op,
    consts: &[Option<i64>],
    bool_neg: &[Option<Value>],
    types: &[Type],
    ty: Type,
    subst: &[Value],
) -> Option<Key> {
    let opnd = |value: Value| operand(consts, types, subst, value);
    Some(match *op {
        Op::Const(imm) => Key::Const(
            ty,
            if ty.is_small_int() {
                extend(imm, ty, false)
            } else {
                imm
            },
        ),
        Op::Unary(op, x) => Key::Unary(op, opnd(x)),
        Op::Binary(Binary::Isub, x, y) => match (opnd(x), opnd(y)) {
            (a, Operand::Imm(sub_ty, imm)) => {
                let neg = extend(imm.wrapping_neg(), sub_ty, false);
                let (a, b) = order_pair(a, Operand::Imm(sub_ty, neg));
                Key::Binary(Binary::Iadd, a, b)
            }
            (a, b) => Key::Binary(Binary::Isub, a, b),
        },
        Op::Binary(op, x, y) => {
            let (a, b) = (opnd(x), opnd(y));
            let (a, b) = if is_commutative_binary(op) {
                order_pair(a, b)
            } else {
                (a, b)
            };
            Key::Binary(op, a, b)
        }
        Op::Convert(op, x) => Key::Convert(op, ty, opnd(x)),
        Op::Icmp(cc, x, y) => {
            let x_ty = types[find(subst, x).index()];
            let (cc, a, b) = canonical_icmp(cc, opnd(x), opnd(y));
            Key::Icmp(cc, x_ty, a, b)
        }
        Op::Fits(op, signed, x, y) => {
            let (a, b) = (opnd(x), opnd(y));
            let (a, b) = if matches!(op, Binary::Iadd | Binary::Imul) {
                order_pair(a, b)
            } else {
                (a, b)
            };
            Key::Fits(op, signed, a, b)
        }
        Op::Fcmp(cc, x, y) => {
            let x_ty = types[find(subst, x).index()];
            let (a, b) = (opnd(x), opnd(y));
            if a > b {
                Key::Fcmp(flip_float_cc(cc), x_ty, b, a)
            } else {
                Key::Fcmp(cc, x_ty, a, b)
            }
        }
        Op::Shuffle(a, b, mask) => {
            let mkey = match mask {
                ShuffleMask::Const(bytes) => ShuffleKey::Const(bytes),
                ShuffleMask::Dyn(idx) => ShuffleKey::Dyn(opnd(idx)),
            };
            Key::Shuffle(ty, opnd(a), opnd(b), mkey)
        }
        Op::Extract(v, lane) => Key::Extract(opnd(v), lane),
        Op::Insert(vec, scalar, lane) => Key::Insert(opnd(vec), opnd(scalar), lane),
        Op::Reduce(kind, v) => Key::Reduce(kind, opnd(v)),
        Op::TargetOp(id, args) => {
            let s = args.as_slice();
            Key::TargetOp(
                id,
                [
                    s.first().map(|v| opnd(*v)),
                    s.get(1).map(|v| opnd(*v)),
                    s.get(2).map(|v| opnd(*v)),
                ],
            )
        }
        Op::Select(c, x, y) => {
            let (cond, sx, sy) = unwrap_bool_neg(bool_neg, subst, c, x, y);
            Key::Select(opnd(cond), opnd(sx), opnd(sy))
        }
        Op::Fma(a, b, c) => Key::Fma(opnd(a), opnd(b), opnd(c)),
        Op::StackAddr(slot, offset) => Key::StackAddr(slot, offset),
        Op::SlotAddr(sym) => Key::SlotAddr(sym),
        Op::Symbol(sym) => Key::Symbol(sym),
        _ => return None,
    })
}

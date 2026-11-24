use crate::fold::extend;
use crate::ir::{Binary, Convert, Op, Type, Unary, Value};

use super::RewriteCx;

pub(super) enum Demand {
    Const(u64),
    Bits(Value),
}

fn shift_demand(op: Binary, w: u64, k: u32, mask: u64) -> u64 {
    if op == Binary::Ushr {
        (mask << k) & w
    } else {
        mask >> k
    }
}

pub(super) fn demanded(cx: &RewriteCx, v: Value, mask: u64, depth: u8) -> Demand {
    let (kz, ko) = cx.known(v);
    if ((kz | ko) & mask) == mask {
        return Demand::Const(ko & mask);
    }
    let Some(inst) = cx.def(v).filter(|_| depth < 6) else {
        return Demand::Bits(v);
    };
    let ty = cx.func.ty(v);
    let next = |u: Value, m: u64| demanded(cx, u, m, depth + 1);
    match cx.func.insts[inst.index()].op {
        Op::Binary(bin @ (Binary::Bor | Binary::Bxor | Binary::Band), a, b) => [(a, b), (b, a)]
            .into_iter()
            .find_map(|(keep, drop)| {
                let bits = if bin == Binary::Band {
                    cx.known(drop).1
                } else {
                    cx.known(drop).0
                };
                ((bits & mask) == mask).then(|| next(keep, mask))
            })
            .unwrap_or(Demand::Bits(v)),
        Op::Unary(Unary::Bnot, inner) => match cx.op(inner) {
            Some(&Op::Binary(Binary::Bxor, x, y)) => [(x, y), (y, x)]
                .into_iter()
                .find_map(|(src, cv)| {
                    cx.uconst(cv, ty)
                        .filter(|c| (c & mask) == mask)
                        .map(|_| next(src, mask))
                })
                .unwrap_or(Demand::Bits(v)),
            _ => Demand::Bits(v),
        },
        Op::Binary(op @ (Binary::Ishl | Binary::Ushr), a, sh) => {
            let Some(k) = cx.count(sh, ty) else {
                return Demand::Bits(v);
            };
            let (w, inner) = (
                ty.unsigned_max(),
                shift_demand(op, ty.unsigned_max(), k, mask),
            );
            match next(a, inner) {
                Demand::Const(c) => Demand::Const(if op == Binary::Ushr {
                    c >> k
                } else {
                    (c << k) & mask
                }),
                Demand::Bits(s) => match cx.op(s) {
                    _ if k == 0 => Demand::Bits(s),
                    Some(&Op::Binary(inv @ (Binary::Ishl | Binary::Ushr), t, sh2))
                        if inv != op
                            && cx.count(sh2, ty) == Some(k)
                            && shift_demand(inv, w, k, inner) == mask =>
                    {
                        next(t, mask)
                    }
                    _ => Demand::Bits(v),
                },
            }
        }
        Op::Convert(kind @ (Convert::Uextend | Convert::Ireduce), a)
            if cx.func.ty(a).is_small_int() =>
        {
            let inner = mask & cx.func.ty(a).unsigned_max();
            match next(a, inner) {
                Demand::Const(c) => Demand::Const(c),
                Demand::Bits(s) => match cx.op(s) {
                    Some(&Op::Convert(back @ (Convert::Uextend | Convert::Ireduce), t))
                        if back != kind && inner == mask && cx.func.ty(t) == ty =>
                    {
                        next(t, mask)
                    }
                    _ => Demand::Bits(v),
                },
            }
        }
        _ => Demand::Bits(v),
    }
}

fn field_of(cx: &RewriteCx, v: Value, depth: u8) -> (Value, u64) {
    let ty = cx.func.ty(v);
    let w = ty.unsigned_max();
    let part = |s: Value, m: u64| {
        let (src, sm) = field_of(cx, s, depth + 1);
        (src, sm & m)
    };
    match cx.op(v).filter(|_| depth < 6 && ty.is_small_int()) {
        Some(&Op::Binary(Binary::Band, a, b)) => [(a, b), (b, a)]
            .into_iter()
            .find_map(|(s, c)| cx.uconst(c, ty).map(|m| part(s, m)))
            .unwrap_or((v, w)),
        Some(&Op::Convert(Convert::Uextend, n)) => match cx.op(n) {
            Some(&Op::Convert(Convert::Ireduce, s)) if cx.func.ty(s) == ty => {
                part(s, cx.func.ty(n).unsigned_max())
            }
            _ => (v, w),
        },
        Some(&Op::Binary(op @ (Binary::Ishl | Binary::Ushr), f, sh)) => {
            let (s1, m1) = field_of(cx, f, depth + 1);
            match (cx.count(sh, ty), cx.op(s1)) {
                (Some(k), Some(&Op::Binary(inv @ (Binary::Ishl | Binary::Ushr), s, sh2)))
                    if inv != op && cx.count(sh2, ty) == Some(k) =>
                {
                    part(s, shift_demand(inv, w, k, m1))
                }
                _ => (v, w),
            }
        }
        _ => (v, w),
    }
}

pub(super) fn merge_fields(
    cx: &mut RewriteCx,
    res: Option<Value>,
    a: Value,
    b: Value,
) -> Option<Op> {
    let ty = cx.func.ty(a);
    let ((src, ma), (other, mb)) = (field_of(cx, a, 0), field_of(cx, b, 0));
    if src != other || !ty.is_small_int() {
        return None;
    }
    if cx.redundant_mask(src, ty, ma | mb) {
        cx.alias(res, src);
        return None;
    }
    let mask = cx.emit(ty, Op::Const(extend((ma | mb) as i64, ty, true)));
    Some(Op::Binary(Binary::Band, src, mask))
}

pub(super) fn match_swar_msb_broadcast(cx: &RewriteCx, v: Value) -> Option<Value> {
    if cx.func.ty(v) != Type::I64 {
        return None;
    }
    let Op::Binary(Binary::Imul, m1, m2) = cx.func.insts[cx.def(v)?.index()].op else {
        return None;
    };
    [(m1, m2), (m2, m1)]
        .into_iter()
        .find_map(|(shifted, mul_c)| {
            (cx.sconst(mul_c) == Some(0xFF))
                .then(|| {
                    let Op::Binary(Binary::Ushr, x, sh) =
                        cx.func.insts[cx.def(shifted)?.index()].op
                    else {
                        return None;
                    };
                    (cx.sconst(sh) == Some(7)
                        && (cx.known(x).0 & 0x7f7f_7f7f_7f7f_7f7f) == 0x7f7f_7f7f_7f7f_7f7f)
                        .then_some(x)
                })
                .flatten()
        })
}

fn fold_nested_shift(
    cx: &mut RewriteCx,
    res: Option<Value>,
    (op, ty): (Binary, Type),
    (cur, sh): (Value, Value),
    k: u32,
) -> Option<Op> {
    let Op::Binary(inner_op, inner, sh1) = cx.func.insts[cx.def(cur)?.index()].op else {
        return None;
    };
    let k1 = cx.count(sh1, ty)?;
    let inv_op = if op == Binary::Ushr {
        Binary::Ishl
    } else {
        Binary::Ushr
    };
    if inner_op == op {
        if k1 == 0 {
            return Some(Op::Binary(op, inner, sh));
        }
        if k1 + k < ty.bits() {
            let sum = cx.emit(ty, Op::Const(i64::from(k1 + k)));
            return Some(Op::Binary(op, inner, sum));
        }
        return Some(Op::Const(0));
    }
    if inner_op == inv_op && k1 == k {
        let keep = if op == Binary::Ushr {
            ty.unsigned_max() >> k
        } else {
            ty.unsigned_max() & !((1_u64 << k) - 1)
        };
        if cx.redundant_mask(inner, ty, keep) {
            cx.alias(res, inner);
            return None;
        }
        let mv = cx.emit(ty, Op::Const(extend(keep as i64, ty, true)));
        return Some(Op::Binary(Binary::Band, inner, mv));
    }
    None
}

pub(super) fn simplify_shift(
    cx: &mut RewriteCx,
    res: Option<Value>,
    op: Binary,
    x: Value,
    sh: Value,
) -> Option<Op> {
    if op == Binary::Ushr
        && cx.func.ty(x) == Type::I64
        && cx.sconst(sh) == Some(3)
        && let Some(inst) = cx.def(x)
        && let Op::Unary(Unary::Ctz, m) = cx.func.insts[inst.index()].op
        && let Some(word) = match_swar_msb_broadcast(cx, m)
    {
        return Some(Op::Binary(
            Binary::Ushr,
            cx.emit(Type::I64, Op::Unary(Unary::Ctz, word)),
            sh,
        ));
    }
    let ty = cx.func.ty(x);
    if matches!(ty, Type::I32 | Type::I64)
        && cx.sconst(sh).is_none()
        && let count = strip_rot_mask(cx, cx.func.ty(sh), sh, u64::from(ty.bits() - 1))
        && count != sh
    {
        return Some(Op::Binary(op, x, count));
    }
    let raw_k = cx.sconst(sh).filter(|_| ty.is_small_int())?;
    let k = (raw_k as u32) & (ty.bits() - 1);
    if k == 0 {
        cx.alias(res, x);
        return None;
    }
    let cur = match demanded(
        cx,
        x,
        shift_demand(op, ty.unsigned_max(), k, ty.unsigned_max()),
        0,
    ) {
        Demand::Const(c) => {
            return crate::fold::binary(op, ty, c as i64, raw_k)
                .map(|v| Op::Const(extend(v, ty, true)));
        }
        Demand::Bits(v) => v,
    };
    fold_nested_shift(cx, res, (op, ty), (cur, sh), k)
        .or_else(|| (cur != x).then_some(Op::Binary(op, cur, sh)))
}

fn unmask_shift(cx: &RewriteCx, mut v: Value, bits: u32) -> Value {
    let m = u64::from(bits - 1);
    for _ in 0..4 {
        let Some(inst) = cx.def(v) else { break };
        let ty = cx.func.ty(v);
        v = match cx.func.insts[inst.index()].op {
            Op::Binary(Binary::Band, x, y) if cx.uconst(y, ty).is_some_and(|c| (c & m) == m) => x,
            Op::Binary(Binary::Band, x, y) if cx.uconst(x, ty).is_some_and(|c| (c & m) == m) => y,
            Op::Convert(Convert::Uextend | Convert::Sextend | Convert::Ireduce, x)
                if cx.func.ty(x).is_small_int() =>
            {
                x
            }
            _ => break,
        };
    }
    v
}

fn unwrap_shift_arg(cx: &RewriteCx, v: Value, bits: u32) -> (Value, bool) {
    let v = unmask_shift(cx, v, bits);
    let mod_zero = |c: Value| {
        cx.uconst(c, cx.func.ty(c))
            .is_some_and(|k| (k & u64::from(bits - 1)) == 0)
    };
    if let Some(inst) = cx.def(v) {
        match cx.func.insts[inst.index()].op {
            Op::Unary(Unary::Ineg, inner) => return (unmask_shift(cx, inner, bits), true),
            Op::Binary(Binary::Isub, c, inner) if mod_zero(c) => {
                return (unmask_shift(cx, inner, bits), true);
            }
            Op::Binary(Binary::Iadd, p, q) => {
                for (c, neg) in [(p, q), (q, p)] {
                    if mod_zero(c)
                        && let Some(ni) = cx.def(neg)
                        && let Op::Unary(Unary::Ineg, inner) = cx.func.insts[ni.index()].op
                    {
                        return (unmask_shift(cx, inner, bits), true);
                    }
                }
            }
            _ => {}
        }
    }
    (v, false)
}

fn match_shift_pair(
    cx: &RewriteCx,
    a: Value,
    b: Value,
) -> Option<((Value, Value), (Value, Value))> {
    let (ia, ib) = (cx.def(a)?, cx.def(b)?);
    [(ia, ib), (ib, ia)].into_iter().find_map(|(l, r)| {
        if let (Op::Binary(Binary::Ishl, x1, sh1), Op::Binary(Binary::Ushr, x2, sh2)) =
            (&cx.func.insts[l.index()].op, &cx.func.insts[r.index()].op)
        {
            Some(((*x1, *sh1), (*x2, *sh2)))
        } else {
            None
        }
    })
}

fn strip_rot_mask(cx: &RewriteCx, ty: Type, v: Value, keep: u64) -> Value {
    let Some(inst) = cx.def(v) else { return v };
    let Op::Binary(Binary::Band, x, y) = cx.func.insts[inst.index()].op else {
        return v;
    };
    if cx.uconst(y, ty).is_some_and(|m| (m & keep) == keep) {
        x
    } else if cx.uconst(x, ty).is_some_and(|m| (m & keep) == keep) {
        y
    } else {
        v
    }
}

pub(super) fn ireduce_rotate(cx: &mut RewriteCx, dst: Type, v: Value) -> Option<Op> {
    if dst != Type::I32 {
        return None;
    }
    let Op::Binary(Binary::Bor | Binary::Bxor | Binary::Iadd, a, b) =
        cx.func.insts[cx.def(v)?.index()].op
    else {
        return None;
    };
    let ((x1, sh1), (x2, sh2)) = match_shift_pair(cx, a, b)?;
    let (c1, c2) = (cx.sconst(sh1)?, cx.sconst(sh2)?);
    if c1 <= 0 || c2 <= 0 || c1 + c2 != 32 {
        return None;
    }
    let (k1, k2) = (c1 as u32, c2 as u32);
    let hi_mask = ((1_u64 << k2) - 1) << 32;
    if (cx.known(x2).0 & hi_mask) != hi_mask {
        return None;
    }
    let ty = cx.func.ty(x1);
    let (s1, s2) = (
        strip_rot_mask(cx, ty, x1, dst.unsigned_max() >> k1),
        strip_rot_mask(cx, ty, x2, dst.unsigned_max() ^ ((1_u64 << k2) - 1)),
    );
    let peel = |val: Value| match cx.def(val).map(|i| &cx.func.insts[i.index()].op) {
        Some(&Op::Convert(Convert::Uextend, inner)) => {
            if cx.func.ty(inner) == dst {
                Ok(inner)
            } else if let Some(ii) = cx.def(inner)
                && let Op::Convert(Convert::Ireduce, raw) = cx.func.insts[ii.index()].op
                && cx.func.ty(raw) == ty
            {
                Err(raw)
            } else {
                Err(val)
            }
        }
        _ => Err(val),
    };
    let base = match (peel(s1), peel(s2)) {
        (Ok(u1), Ok(u2)) if u1 == u2 => u1,
        (Err(r1), Err(r2)) if r1 == r2 => cx.emit(dst, Op::Convert(Convert::Ireduce, r1)),
        _ => return None,
    };
    let sh = cx.emit(dst, Op::Const(i64::from(k2)));
    Some(Op::Binary(Binary::Rotr, base, sh))
}

pub(super) fn simplify_rotate(cx: &mut RewriteCx, bin: Binary, a: Value, b: Value) -> Option<Op> {
    let ty = cx.func.ty(a);
    if !matches!(ty, Type::I32 | Type::I64) {
        return None;
    }
    let bits = ty.bits();
    let ((x1, sh1), (x2, sh2)) = match_shift_pair(cx, a, b)?;
    if let (Some(c1), Some(c2)) = (cx.sconst(sh1), cx.sconst(sh2))
        && let (k1, k2) = ((c1 as u32) & (bits - 1), (c2 as u32) & (bits - 1))
        && k1 > 0
        && k2 > 0
        && k1 + k2 == bits
        && let (ux1, ux2) = (
            strip_rot_mask(cx, ty, x1, ty.unsigned_max() >> k1),
            strip_rot_mask(cx, ty, x2, ty.unsigned_max() ^ ((1_u64 << k2) - 1)),
        )
        && ux1 == ux2
    {
        let sh_val = if c2 as u32 == k2 {
            sh2
        } else {
            cx.emit(ty, Op::Const(i64::from(k2)))
        };
        return Some(Op::Binary(Binary::Rotr, ux1, sh_val));
    }
    if bin == Binary::Bor
        && let (ux1, ux2) = (
            strip_rot_mask(cx, ty, x1, ty.unsigned_max()),
            strip_rot_mask(cx, ty, x2, ty.unsigned_max()),
        )
        && ux1 == ux2
    {
        let ((b1, n1), (b2, n2)) = (
            unwrap_shift_arg(cx, sh1, bits),
            unwrap_shift_arg(cx, sh2, bits),
        );
        if b1 == b2 && n1 != n2 {
            return Some(Op::Binary(
                if n1 { Binary::Rotr } else { Binary::Rotl },
                ux1,
                if n1 { sh2 } else { sh1 },
            ));
        }
    }
    None
}

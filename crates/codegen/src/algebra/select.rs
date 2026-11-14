use crate::ir::{Binary, Convert, IntCC, Op, Type, Value};

use super::{RewriteCx, is_non_null_addr, swap_cc};

pub(super) fn simplify_select(cx: &mut RewriteCx, res: Option<Value>, cond: Value, t: Value, f: Value) -> Option<Op> {
    if t == f {
        cx.alias(res, t);
        return None;
    }
    let inner = cx.unwrap_uextend(cond);
    if res.is_some_and(|r| cx.func.ty(r) == Type::I8) && cx.func.ty(inner) == Type::I8 && cx.is_unit_bool(inner) {
        match (cx.uconst(t, Type::I8), cx.uconst(f, Type::I8)) {
            (Some(1), Some(0)) => {
                cx.alias(res, inner);
                return None;
            }
            (Some(1), _) if cx.is_unit_bool(f) => return Some(Op::Binary(Binary::Bor, inner, f)),
            (_, Some(0)) if cx.is_unit_bool(t) => return Some(Op::Binary(Binary::Band, inner, t)),
            (Some(0), Some(1)) => return Some(Op::Binary(Binary::Bxor, inner, f)),
            _ => {}
        }
    }
    (inner != cond).then_some(Op::Select(inner, t, f))
}

fn entry_deref_bytes(cx: &RewriteCx, v: Value) -> u32 {
    let Some(entry) = cx.func.entry else { return 0 };
    cx.func.blocks[entry.index()]
        .params
        .iter()
        .position(|&p| p == v)
        .and_then(|pos| cx.func.signature.params.get(pos))
        .map_or(0, |abi| abi.deref_bytes)
}

pub(super) fn is_non_zero(cx: &RewriteCx, v: Value, depth: u8) -> bool {
    let ty = cx.func.ty(v);
    if depth >= 4 || !ty.is_small_int() {
        return false;
    }
    if is_non_null_addr(cx.func, &|x| cx.def(x), &|x| cx.sconst(x), v, 0) || cx.known(v).1 != 0 {
        return true;
    }
    if entry_deref_bytes(cx, v) > 0 {
        return true;
    }
    let Some(inst) = cx.def(v) else { return false };
    match cx.func.insts[inst.index()].op {
        Op::Binary(Binary::Iadd, a, b) => [(a, b), (b, a)].into_iter().any(|(base, off_v)| {
            cx.sconst(off_v).is_some_and(|c| {
                let deref = u64::from(entry_deref_bytes(cx, base));
                if deref > 0 && c >= 0 && (c as u64) < deref {
                    return true;
                }
                cx.def(base).is_some_and(|b_inst| {
                    matches!(cx.func.insts[b_inst.index()].op, Op::StackAddr(..) | Op::SlotAddr(_))
                }) && (0..=0x1000_0000).contains(&c)
            })
        }),
        Op::Convert(Convert::Bitcast, a) if cx.func.ty(a).bytes() == ty.bytes() => is_non_zero(cx, a, depth + 1),
        Op::Convert(Convert::Uextend | Convert::Sextend, a) if cx.func.ty(a).is_small_int() => {
            is_non_zero(cx, a, depth + 1)
        }
        Op::Select(_, a, b) => is_non_zero(cx, a, depth + 1) && is_non_zero(cx, b, depth + 1),
        _ => false,
    }
}

fn bool_cond(cx: &RewriteCx, raw: Value) -> Option<Value> {
    let c = cx.unwrap_uextend(raw);
    (cx.func.ty(c) == Type::I8 && cx.is_unit_bool(c)).then_some(c)
}

fn simplify_1_bool(cx: &mut RewriteCx, res: Option<Value>, c: Value, b1: bool, b2: bool) -> Option<Op> {
    match (b1, b2) {
        (true, true) => Some(Op::Const(1)),
        (false, false) => Some(Op::Const(0)),
        (true, false) => {
            cx.alias(res, c);
            None
        }
        (false, true) => {
            let one = cx.emit(Type::I8, Op::Const(1));
            Some(Op::Binary(Binary::Bxor, c, one))
        }
    }
}

fn eval_const_select_cmp(cx: &mut RewriteCx, res: Option<Value>, cc: IntCC, x: Value, y: Value) -> Option<Option<Op>> {
    let ty = cx.func.ty(x);
    for (norm, sel_val, c_val) in [(cc, x, y), (swap_cc(cc), y, x)] {
        let Some(c) = cx.sconst(c_val) else { continue };
        let Some(sel_inst) = cx.def(sel_val) else { continue };
        let Op::Select(raw_c1, t1, f1) = cx.func.insts[sel_inst.index()].op else { continue };
        let Some(c1) = bool_cond(cx, raw_c1) else { continue };
        let eval = |k: i64| crate::fold::holds(norm, ty, k, c);
        if let (Some(k1), Some(k2)) = (cx.sconst(t1), cx.sconst(f1)) {
            return Some(simplify_1_bool(cx, res, c1, eval(k1), eval(k2)));
        }
        if let (Some(k1), Some(f_inst)) = (cx.sconst(t1), cx.def(f1))
            && let Op::Select(raw_c2, t2, f2) = cx.func.insts[f_inst.index()].op
            && let (Some(k2), Some(k3)) = (cx.sconst(t2), cx.sconst(f2))
        {
            let (b1, b2, b3) = (eval(k1), eval(k2), eval(k3));
            if b2 == b3 {
                return Some(simplify_1_bool(cx, res, c1, b1, b2));
            }
            let Some(c2) = bool_cond(cx, raw_c2) else { continue };
            return Some(Some(match (b1, b2, b3) {
                (false, false, true) => {
                    let or = cx.emit(Type::I8, Op::Binary(Binary::Bor, c1, c2));
                    let one = cx.emit(Type::I8, Op::Const(1));
                    Op::Binary(Binary::Bxor, or, one)
                }
                (true, true, false) => Op::Binary(Binary::Bor, c1, c2),
                (false, true, false) => {
                    let one = cx.emit(Type::I8, Op::Const(1));
                    let nc1 = cx.emit(Type::I8, Op::Binary(Binary::Bxor, c1, one));
                    Op::Binary(Binary::Band, nc1, c2)
                }
                _ => {
                    let one = cx.emit(Type::I8, Op::Const(1));
                    let nc2 = cx.emit(Type::I8, Op::Binary(Binary::Bxor, c2, one));
                    Op::Binary(Binary::Bor, c1, nc2)
                }
            }));
        }
        if let (Some(t_inst), Some(k3)) = (cx.def(t1), cx.sconst(f1))
            && let Op::Select(raw_c2, t2, f2) = cx.func.insts[t_inst.index()].op
            && let (Some(k1), Some(k2)) = (cx.sconst(t2), cx.sconst(f2))
        {
            let (b1, b2, b3) = (eval(k1), eval(k2), eval(k3));
            if b1 == b2 {
                return Some(simplify_1_bool(cx, res, c1, b1, b3));
            }
            let Some(c2) = bool_cond(cx, raw_c2) else { continue };
            return Some(Some(match (b1, b2, b3) {
                (true, false, false) => Op::Binary(Binary::Band, c1, c2),
                (false, true, false) => {
                    let one = cx.emit(Type::I8, Op::Const(1));
                    let nc2 = cx.emit(Type::I8, Op::Binary(Binary::Bxor, c2, one));
                    Op::Binary(Binary::Band, c1, nc2)
                }
                (true, false, true) => {
                    let one = cx.emit(Type::I8, Op::Const(1));
                    let nc1 = cx.emit(Type::I8, Op::Binary(Binary::Bxor, c1, one));
                    Op::Binary(Binary::Bor, nc1, c2)
                }
                _ => {
                    let and = cx.emit(Type::I8, Op::Binary(Binary::Band, c1, c2));
                    let one = cx.emit(Type::I8, Op::Const(1));
                    Op::Binary(Binary::Bxor, and, one)
                }
            }));
        }
    }
    None
}

pub(super) fn eval_select_zero_cmp(cx: &mut RewriteCx, res: Option<Value>, cc: IntCC, x: Value, y: Value) -> Option<Option<Op>> {
    let ty = cx.func.ty(x);
    if !ty.is_small_int() || !res.is_some_and(|r| cx.func.ty(r) == Type::I8) {
        return None;
    }
    if let Some(folded) = eval_const_select_cmp(cx, res, cc, x, y) {
        return Some(folded);
    }
    for (norm, sel_val, c_val) in [(cc, x, y), (swap_cc(cc), y, x)] {
        let Some(k) = cx.uconst(c_val, ty) else { continue };
        let want_nonzero = match (norm, k) {
            (IntCC::NotEqual | IntCC::UnsignedGreaterThan, 0) | (IntCC::UnsignedGreaterThanOrEqual, 1) => true,
            (IntCC::Equal | IntCC::UnsignedLessThanOrEqual, 0) | (IntCC::UnsignedLessThan, 1) => false,
            _ => continue,
        };
        let Some(sel_inst) = cx.def(sel_val) else { continue };
        let Op::Select(raw_cond, t_val, f_val) = cx.func.insts[sel_inst.index()].op else { continue };
        let non_zero_arm = if cx.uconst(f_val, ty) == Some(0) && is_non_zero(cx, t_val, 0) {
            Some(true)
        } else if cx.uconst(t_val, ty) == Some(0) && is_non_zero(cx, f_val, 0) {
            Some(false)
        } else {
            None
        };
        let Some(true_is_nonzero) = non_zero_arm else { continue };
        let cond = cx.unwrap_uextend(raw_cond);
        let truth = want_nonzero == true_is_nonzero;
        if cx.func.ty(cond) == Type::I8 && cx.is_unit_bool(cond) {
            if truth {
                cx.alias(res, cond);
                return Some(None);
            }
            let one = cx.emit(Type::I8, Op::Const(1));
            return Some(Some(Op::Binary(Binary::Bxor, cond, one)));
        }
        let cond_ty = cx.func.ty(cond);
        if !cond_ty.is_small_int() {
            continue;
        }
        let zero = cx.emit(cond_ty, Op::Const(0));
        return Some(Some(Op::Icmp(if truth { IntCC::NotEqual } else { IntCC::Equal }, cond, zero)));
    }
    None
}

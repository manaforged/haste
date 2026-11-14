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

pub(super) fn eval_select_zero_cmp(cx: &mut RewriteCx, res: Option<Value>, cc: IntCC, x: Value, y: Value) -> Option<Option<Op>> {
    let ty = cx.func.ty(x);
    if !ty.is_small_int() || !res.is_some_and(|r| cx.func.ty(r) == Type::I8) {
        return None;
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

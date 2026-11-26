use crate::fold::non_null_at;
use crate::ir::{Binary, Convert, IntCC, Op, Type, Value};

use super::super::RewriteCx;
use super::super::icmp::{ADDR_SPAN, before, offset_bound, three_way_non_zero};

pub(in super::super) fn tested_select(cx: &RewriteCx, cond: Value) -> Option<(Value, bool)> {
    match *cx.op(cx.unwrap_uextend(cond))? {
        Op::Select(test, t, f) => {
            let ty = cx.func.ty(t);
            if !ty.is_small_int() || !cx.func.ty(test).is_small_int() {
                return None;
            }
            if cx.uconst(f, ty) == Some(0)
                && is_non_zero_under(cx, t, unwrap_cond_cmp(cx, test, true), 0)
            {
                Some((test, true))
            } else if cx.uconst(t, ty) == Some(0)
                && is_non_zero_under(cx, f, unwrap_cond_cmp(cx, test, false), 0)
            {
                Some((test, false))
            } else {
                None
            }
        }
        Op::Binary(Binary::Isub | Binary::Bxor, a, b) => {
            let ty = cx.func.ty(a);
            if !ty.is_small_int() {
                return None;
            }
            for (sel_v, k_v) in [(a, b), (b, a)] {
                let (Some(k), Some(&Op::Select(test, t, f))) = (cx.uconst(k_v, ty), cx.op(sel_v))
                else {
                    continue;
                };
                if !cx.func.ty(test).is_small_int() {
                    continue;
                }
                let (ct, cf) = (cx.uconst(t, ty), cx.uconst(f, ty));
                if cf == Some(k) && ct.is_some_and(|v| v != k) {
                    return Some((test, true));
                }
                if ct == Some(k) && cf.is_some_and(|v| v != k) {
                    return Some((test, false));
                }
            }
            None
        }
        _ => None,
    }
}

pub(super) fn unwrap_cond_cmp(
    cx: &RewriteCx,
    mut raw: Value,
    mut taken: bool,
) -> Option<(IntCC, Value, Value)> {
    for _ in 0..4 {
        raw = cx.unwrap_uextend(raw);
        if let Some(&Op::Binary(Binary::Bxor, a, b)) = cx.op(raw)
            && cx.func.ty(raw).is_small_int()
        {
            let next = [(a, b), (b, a)].into_iter().find_map(|(inner, mask)| {
                (cx.uconst(mask, cx.func.ty(mask)) == Some(1) && cx.is_unit_bool(inner))
                    .then_some(inner)
            });
            if let Some(inner) = next {
                (raw, taken) = (inner, !taken);
                continue;
            }
        }
        break;
    }
    let &Op::Icmp(cc, a, b) = cx.op(raw)? else {
        return None;
    };
    cx.func
        .ty(a)
        .is_small_int()
        .then_some((if taken { cc } else { cc.inverse() }, a, b))
}

fn cond_upper_bound(cx: &RewriteCx, (eff, a, b): (IntCC, Value, Value)) -> Option<(Value, u64)> {
    let (ty, sign) = (cx.func.ty(a), 1_u64 << (cx.func.ty(a).bits() - 1));
    let non_neg = |v: Value| (cx.known(v).0 & sign) != 0;
    for (norm, mut var, lim) in [(eff, a, b), (eff.swapped(), b, a)] {
        if cx.sconst(var).is_some() {
            continue;
        }
        let (lim_hi, both_nn) = (
            ty.unsigned_max() & !cx.known(lim).0,
            non_neg(var) && non_neg(lim),
        );
        let bound = match norm {
            IntCC::UnsignedLessThan => lim_hi.checked_sub(1),
            IntCC::UnsignedLessThanOrEqual | IntCC::Equal => Some(lim_hi),
            IntCC::SignedLessThan if both_nn => lim_hi.checked_sub(1),
            IntCC::SignedLessThanOrEqual if both_nn => Some(lim_hi),
            _ => None,
        };
        if let Some(mut hi) = bound {
            while let Some(&Op::Convert(Convert::Uextend, inner)) = cx.op(var)
                && cx.func.ty(inner).is_small_int()
            {
                var = inner;
                hi = hi.min(cx.func.ty(var).unsigned_max());
            }
            return Some((var, hi));
        }
    }
    None
}

fn cond_proves_nonzero(cx: &RewriteCx, (eff, a, b): (IntCC, Value, Value), v: Value) -> bool {
    let ty = cx.func.ty(a);
    let sign = 1_u64 << (ty.bits() - 1);
    [(eff, a, b), (eff.swapped(), b, a)]
        .into_iter()
        .any(|(norm, var, lim)| {
            let same = var == v
                || matches!(cx.op(var), Some(&Op::Convert(Convert::Uextend, inner)) if inner == v);
            same && match norm {
                IntCC::UnsignedGreaterThan => true,
                IntCC::NotEqual => cx.uconst(lim, ty) == Some(0),
                IntCC::Equal | IntCC::UnsignedGreaterThanOrEqual => cx.known(lim).1 != 0,
                IntCC::SignedGreaterThan => {
                    (cx.known(var).0 & sign) != 0 && (cx.known(lim).0 & sign) != 0
                }
                _ => false,
            }
        })
}

fn peel_const_base(cx: &RewriteCx, mut v: Value) -> (Value, i64) {
    let mut disp = 0_i64;
    for _ in 0..4 {
        match cx.op(v) {
            Some(&Op::Convert(Convert::Bitcast, a)) if cx.func.ty(a) == cx.func.ty(v) => v = a,
            Some(&Op::Binary(Binary::Iadd, a, b)) => {
                let Some((base, next)) = [(a, b), (b, a)]
                    .into_iter()
                    .find_map(|(base, k)| Some((base, disp.checked_add(cx.sconst(k)?)?)))
                else {
                    break;
                };
                (v, disp) = (base, next);
            }
            Some(&Op::Binary(Binary::Isub, a, b)) => {
                let Some(next) = cx
                    .sconst(b)
                    .and_then(i64::checked_neg)
                    .and_then(|c| disp.checked_add(c))
                else {
                    break;
                };
                (v, disp) = (a, next);
            }
            _ => break,
        }
    }
    (v, disp)
}

fn deref_span_covers(cx: &RewriteCx, root: Value, lo_disp: i64, hi_disp: i64) -> bool {
    cx.func.ty(root) == Type::I64
        && cx.out.iter().rev().take(64).any(|&inst| {
            let data = &cx.func.insts[inst.index()];
            let (addr, off, bytes) = match data.op {
                Op::Load(addr, off) => {
                    (addr, off, data.result.map_or(0, |r| cx.func.ty(r).bytes()))
                }
                Op::Store(val, addr, off) => (addr, off, cx.func.ty(val).bytes()),
                _ => return false,
            };
            let (aroot, adisp) = peel_const_base(cx, addr);
            bytes != 0
                && aroot == root
                && adisp
                    .checked_add(i64::from(off))
                    .and_then(|lo| Some((lo, lo.checked_add(i64::from(bytes))?)))
                    .is_some_and(|(lo, hi)| lo <= lo_disp && hi_disp < hi)
        })
}

fn valid_ptr_span(cx: &RewriteCx, root: Value, base_disp: i64, off_max: u64) -> bool {
    let Some(hi_disp) = i64::try_from(off_max)
        .ok()
        .and_then(|o| base_disp.checked_add(o))
    else {
        return false;
    };
    match cx.op(root) {
        Some(Op::StackAddr(..) | Op::SlotAddr(_)) => {
            (-ADDR_SPAN..=ADDR_SPAN).contains(&base_disp)
                && (-ADDR_SPAN..=ADDR_SPAN).contains(&hi_disp)
        }
        Some(Op::Symbol(_)) => base_disp >= 0 && hi_disp <= ADDR_SPAN,
        _ => {
            cx.func.ty(root) == Type::I64
                && ((base_disp >= 0 && (hi_disp as u64) < u64::from(entry_deref_bytes(cx, root)))
                    || deref_span_covers(cx, root, base_disp, hi_disp))
        }
    }
}

fn entry_deref_bytes(cx: &RewriteCx, v: Value) -> u32 {
    let Some(entry) = cx.func.entry else {
        return 0;
    };
    cx.func.blocks[entry.index()]
        .params
        .iter()
        .position(|&p| p == v)
        .and_then(|pos| cx.func.signature.params.get(pos))
        .map_or(0, |abi| abi.deref_bytes)
}

pub(in super::super) fn is_non_zero(cx: &RewriteCx, v: Value, depth: u8) -> bool {
    is_non_zero_under(cx, v, None, depth)
}

fn add_non_zero(
    cx: &RewriteCx,
    (a, b): (Value, Value),
    outer_disp: i64,
    cmp: Option<(IntCC, Value, Value)>,
) -> bool {
    let bound = cmp.and_then(|c| cond_upper_bound(cx, c));
    [(a, b), (b, a)].into_iter().any(|(base, off_v)| {
        let (root, base_disp) = peel_const_base(cx, base);
        base_disp.checked_add(outer_disp).is_some_and(|disp| {
            valid_ptr_span(
                cx,
                root,
                disp,
                offset_bound(
                    cx.func,
                    &|x| cx.def(x),
                    &|x| cx.sconst(x),
                    off_v,
                    (bound, 0),
                ),
            )
        })
    })
}

fn known_non_zero(cx: &RewriteCx, v: Value, cmp: Option<(IntCC, Value, Value)>) -> bool {
    non_null_at(cx.func, &|x| cx.def(x), &|x| cx.sconst(x), before(cx), v)
        || cx.known(v).1 != 0
        || entry_deref_bytes(cx, v) > 0
        || three_way_non_zero(cx, v)
        || cmp.is_some_and(|c| cond_proves_nonzero(cx, c, v))
}

pub(super) fn is_non_zero_under(
    cx: &RewriteCx,
    v: Value,
    cmp: Option<(IntCC, Value, Value)>,
    depth: u8,
) -> bool {
    let ty = cx.func.ty(v);
    if depth >= 4 || !ty.is_small_int() {
        return false;
    }
    if known_non_zero(cx, v, cmp) {
        return true;
    }
    let (core, outer_disp) = peel_const_base(cx, v);
    if outer_disp != 0 && valid_ptr_span(cx, core, outer_disp, 0) {
        return true;
    }
    let Some(inst) = cx.def(core) else {
        return false;
    };
    match cx.func.insts[inst.index()].op {
        Op::Binary(Binary::Iadd, a, b) => add_non_zero(cx, (a, b), outer_disp, cmp),
        Op::Convert(Convert::Uextend | Convert::Sextend, a)
            if outer_disp == 0 && cx.func.ty(a).is_small_int() =>
        {
            is_non_zero_under(cx, a, cmp, depth + 1)
        }
        Op::Select(_, a, b) if outer_disp == 0 => {
            is_non_zero_under(cx, a, cmp, depth + 1) && is_non_zero_under(cx, b, cmp, depth + 1)
        }
        _ => false,
    }
}

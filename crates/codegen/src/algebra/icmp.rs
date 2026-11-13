use crate::fold::{extend, holds};
use crate::ir::{Binary, Convert, IntCC, Op, Type, Value};

use super::{RewriteCx, swap_cc};

const MAX_DEPTH: u8 = 4;
const MAX_LEAVES: usize = 16;

fn apply_convert(c: i64, kind: Convert, src: Type, dst: Type) -> i64 {
    match kind {
        Convert::Uextend => extend(c, src, false),
        Convert::Sextend => extend(c, src, true),
        Convert::Ireduce => extend(c, dst, false),
        _ => c,
    }
}

fn collect_leaves(cx: &RewriteCx, v: Value, depth: u8, out: &mut Vec<i64>) -> bool {
    if depth > MAX_DEPTH || out.len() >= MAX_LEAVES {
        return false;
    }
    if let Some(imm) = cx.sconst(v) {
        out.push(imm);
        return true;
    }
    let Some(inst) = cx.def(v) else {
        return false;
    };
    match cx.func.insts[inst.index()].op {
        Op::Convert(kind @ (Convert::Uextend | Convert::Sextend | Convert::Ireduce), inner) => {
            let src = cx.func.ty(inner);
            let dst = cx.func.ty(v);
            if !src.is_small_int() {
                return false;
            }
            let start = out.len();
            if !collect_leaves(cx, inner, depth + 1, out) {
                return false;
            }
            for slot in &mut out[start..] {
                *slot = apply_convert(*slot, kind, src, dst);
            }
            true
        }
        Op::Select(cond, a, b) if cx.func.ty(cond) == Type::I8 => {
            collect_leaves(cx, a, depth + 1, out) && collect_leaves(cx, b, depth + 1, out)
        }
        _ => false,
    }
}

fn uniform_holds(leaves: &[i64], convs: &[(Convert, Type, Type)], cc: IntCC, cmp_ty: Type, k: i64) -> Option<bool> {
    let mut expected = None;
    for &leaf in leaves {
        let mut val = leaf;
        for &(kind, src, dst) in convs.iter().rev() {
            val = apply_convert(val, kind, src, dst);
        }
        let bit = holds(cc, cmp_ty, val, k);
        match expected {
            None => expected = Some(bit),
            Some(prev) if prev != bit => return None,
            _ => {}
        }
    }
    expected
}

fn eval_one(cx: &mut RewriteCx, res: Option<Value>, cc: IntCC, mut val: Value, k: i64) -> Option<Option<Op>> {
    let cmp_ty = cx.func.ty(val);
    let mut convs = Vec::new();
    for _ in 0..MAX_DEPTH {
        let Some(inst) = cx.def(val) else { break };
        if let Op::Convert(kind @ (Convert::Uextend | Convert::Sextend | Convert::Ireduce), inner) = cx.func.insts[inst.index()].op
            && cx.func.ty(inner).is_small_int()
        {
            convs.push((kind, cx.func.ty(inner), cx.func.ty(val)));
            val = inner;
        } else {
            break;
        }
    }
    let inst = cx.def(val)?;
    let Op::Select(cond, a, b) = cx.func.insts[inst.index()].op else {
        return None;
    };
    let (mut la, mut lb) = (Vec::new(), Vec::new());
    if !collect_leaves(cx, a, 0, &mut la) || !collect_leaves(cx, b, 0, &mut lb) {
        return None;
    }
    let ha = uniform_holds(&la, &convs, cc, cmp_ty, k)?;
    let hb = uniform_holds(&lb, &convs, cc, cmp_ty, k)?;
    if ha == hb {
        return Some(Some(Op::Const(i64::from(ha))));
    }
    if cx.func.ty(cond) != Type::I8 || !cx.is_unit_bool(cond) {
        return None;
    }
    if ha {
        cx.alias(res, cond);
        Some(None)
    } else {
        let one = cx.emit(Type::I8, Op::Const(1));
        Some(Some(Op::Binary(Binary::Bxor, cond, one)))
    }
}

pub(super) fn eval_select(cx: &mut RewriteCx, res: Option<Value>, cc: IntCC, x: Value, y: Value) -> Option<Option<Op>> {
    if !cx.func.ty(x).is_small_int() || !res.is_some_and(|r| cx.func.ty(r) == Type::I8) {
        return None;
    }
    for (norm, val, c_val) in [(cc, x, y), (swap_cc(cc), y, x)] {
        if let Some(k) = cx.sconst(c_val)
            && let Some(out) = eval_one(cx, res, norm, val, k)
        {
            return Some(out);
        }
    }
    None
}

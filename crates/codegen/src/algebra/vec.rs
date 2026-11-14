use super::RewriteCx;
use crate::ir::{Binary, Convert, Lane, Op, ShuffleMask, Type, Value};

fn is_zero_vec(cx: &RewriteCx, mut v: Value) -> bool {
    for _ in 0..4 {
        if cx.sconst(v) == Some(0) {
            return true;
        }
        let Some(inst) = cx.def(v) else { return false };
        match cx.func.insts[inst.index()].op {
            Op::Const(0) => return true,
            Op::Convert(Convert::Bitcast, inner) => v = inner,
            _ => return false,
        }
    }
    false
}

fn splat_scalar(cx: &RewriteCx, v: Value, count: usize) -> Option<Value> {
    let inst = cx.def(v)?;
    let Op::Shuffle(a, _, ShuffleMask::Const(mask)) = &cx.func.insts[inst.index()].op else {
        return None;
    };
    let head = mask[0];
    if head >= count as u8 || mask[..count].iter().any(|&i| i != head) {
        return None;
    }
    let a_inst = cx.def(*a)?;
    match cx.func.insts[a_inst.index()].op {
        Op::Insert(_, scalar, at) if at == head => Some(scalar),
        _ => None,
    }
}

fn extract_source(cx: &RewriteCx, mut v: Value, lane_bytes: u32) -> Option<(Value, u8)> {
    if let Some(inst) = cx.def(v)
        && let Op::Convert(Convert::Bitcast, inner) = cx.func.insts[inst.index()].op
        && cx.func.ty(inner).bytes() == lane_bytes
    {
        v = inner;
    }
    let inst = cx.def(v)?;
    let Op::Extract(src, idx) = cx.func.insts[inst.index()].op else {
        return None;
    };
    let (src_lane, src_count) = cx.func.ty(src).vec_shape()?;
    (src_lane.bytes() == lane_bytes && idx < src_count).then_some((src, idx))
}

fn match_pack32_half(cx: &RewriteCx, v: Value) -> Option<(Value, u8)> {
    let inst = cx.def(v)?;
    let Op::Binary(Binary::Bor, a, b) = cx.func.insts[inst.index()].op else {
        return None;
    };
    for (lo_v, hi_v) in [(a, b), (b, a)] {
        let Some(li) = cx.def(lo_v) else { continue };
        let Some(hi) = cx.def(hi_v) else { continue };
        let Op::Convert(Convert::Uextend, e0) = cx.func.insts[li.index()].op else { continue };
        let Op::Binary(Binary::Ishl, hi_ext, sh) = cx.func.insts[hi.index()].op else { continue };
        if cx.sconst(sh) != Some(32) {
            continue;
        }
        let Some(hei) = cx.def(hi_ext) else { continue };
        let Op::Convert(Convert::Uextend, e1) = cx.func.insts[hei.index()].op else { continue };
        if cx.func.ty(e0) != Type::I32 || cx.func.ty(e1) != Type::I32 {
            continue;
        }
        if let (Some((s0, i0)), Some((s1, i1))) = (extract_source(cx, e0, 4), extract_source(cx, e1, 4))
            && s0 == s1
            && cx.func.ty(s0).bytes() == 16
            && i0 % 2 == 0
            && i1 == i0 + 1
        {
            return Some((s0, i0 / 2));
        }
    }
    None
}

/// Simplifies vector and scalar `Bitcast` chains and zero bitcasts.
pub(super) fn simplify_bitcast(cx: &mut RewriteCx, res: Option<Value>, x: Value) -> Option<Op> {
    let res_v = res?;
    let ty = cx.func.ty(res_v);
    if cx.func.ty(x) == ty {
        if x != res_v {
            cx.alias(res, x);
        }
        return None;
    }
    if (ty.is_vector() || ty.is_small_int() || ty.is_float()) && is_zero_vec(cx, x) {
        return Some(Op::Const(0));
    }
    let inst = cx.def(x)?;
    let Op::Convert(Convert::Bitcast, inner) = cx.func.insts[inst.index()].op else {
        return None;
    };
    if cx.func.ty(inner) == ty {
        if inner != res_v {
            cx.alias(res, inner);
        }
        return None;
    }
    (cx.func.ty(inner).bytes() == ty.bytes()).then_some(Op::Convert(Convert::Bitcast, inner))
}

/// Simplifies `Extract` through `Insert`, `Shuffle`, and zero vectors.
pub(super) fn simplify_extract(cx: &mut RewriteCx, res: Option<Value>, mut vec: Value, mut idx: u8) -> Option<Op> {
    let res_v = res?;
    let res_ty = cx.func.ty(res_v);
    let orig = (vec, idx);
    for _ in 0..16 {
        let Some(inst) = cx.def(vec) else { break };
        match cx.func.insts[inst.index()].op {
            Op::Const(0) if res_ty.is_small_int() || res_ty.is_float() || res_ty.is_vector() => {
                return Some(Op::Const(0));
            }
            Op::Insert(prev, val, at) => {
                if at == idx {
                    if val != res_v {
                        cx.alias(res, val);
                    }
                    return None;
                }
                vec = prev;
            }
            Op::Shuffle(a, b, ShuffleMask::Const(mask)) => {
                let (_, count) = cx.func.ty(a).vec_shape()?;
                let sel = *mask.get(usize::from(idx))?;
                if sel < count {
                    (vec, idx) = (a, sel);
                } else if sel < 2 * count {
                    (vec, idx) = (b, sel - count);
                } else if res_ty.is_small_int() || res_ty.is_float() || res_ty.is_vector() {
                    return Some(Op::Const(0));
                } else {
                    break;
                }
            }
            _ => break,
        }
    }
    ((vec, idx) != orig).then_some(Op::Extract(vec, idx))
}

/// Simplifies identity, single-source, and all-zero constant `Shuffle` ops.
pub(super) fn simplify_shuffle(cx: &mut RewriteCx, res: Option<Value>, a: Value, b: Value, mask: [u8; 16]) -> Option<Op> {
    let res_v = res?;
    let ty = cx.func.ty(res_v);
    if cx.func.ty(a) != ty || cx.func.ty(b) != ty {
        return None;
    }
    let (_, count) = ty.vec_shape()?;
    let c = usize::from(count);
    if (0..count).all(|i| mask[usize::from(i)] == i || (a == b && mask[usize::from(i)] == count + i)) {
        if a != res_v {
            cx.alias(res, a);
        }
        return None;
    }
    if a != b && (0..count).all(|i| mask[usize::from(i)] == count + i) {
        if b != res_v {
            cx.alias(res, b);
        }
        return None;
    }
    let (za, zb) = (is_zero_vec(cx, a), is_zero_vec(cx, b));
    if mask[..c].iter().all(|&i| if i < count { za } else if i < 2 * count { zb } else { true }) {
        return Some(Op::Const(0));
    }
    if a != b {
        if mask[..c].iter().all(|&i| i < count) {
            return Some(Op::Shuffle(a, a, ShuffleMask::Const(mask)));
        }
        if mask[..c].iter().all(|&i| i >= count && i < 2 * count) {
            let mut m = mask;
            for x in &mut m[..c] {
                *x -= count;
            }
            return Some(Op::Shuffle(b, b, ShuffleMask::Const(m)));
        }
    }
    None
}

/// Folds `Insert` chains reconstructing vectors from `Extract`s, overwritten lanes, and all-zero vectors.
pub(super) fn simplify_insert(cx: &mut RewriteCx, res: Option<Value>, vec: Value, val: Value, lane: u8) -> Option<Op> {
    let res_v = res?;
    let ty = cx.func.ty(res_v);
    let (dst_lane, dst_count) = ty.vec_shape()?;
    let c = usize::from(dst_count);
    if lane >= dst_count {
        return None;
    }
    if extract_source(cx, val, dst_lane.bytes()) == Some((vec, lane)) {
        if vec != res_v {
            cx.alias(res, vec);
        }
        return None;
    }
    if let Some(prev_inst) = cx.def(vec)
        && let Op::Insert(prev, _, at) = cx.func.insts[prev_inst.index()].op
        && at == lane
    {
        return Some(Op::Insert(prev, val, lane));
    }
    let mut slots: [Option<Value>; 16] = [None; 16];
    slots[usize::from(lane)] = Some(val);
    let mut cur = vec;
    for _ in 0..32 {
        if slots[..c].iter().all(Option::is_some) {
            break;
        }
        let Some(inst) = cx.def(cur) else { break };
        let Op::Insert(prev, elem, at) = cx.func.insts[inst.index()].op else {
            break;
        };
        if at < dst_count && slots[usize::from(at)].is_none() {
            slots[usize::from(at)] = Some(elem);
        }
        cur = prev;
    }
    let base = cur;
    let base_zero = is_zero_vec(cx, base);
    if slots[..c].iter().any(Option::is_none)
        && let Some(sp) = splat_scalar(cx, base, c)
    {
        for s in &mut slots[..c] {
            if s.is_none() {
                *s = Some(sp);
            }
        }
    }
    if slots[..c].iter().all(|s| match s {
        Some(v) => is_zero_vec(cx, *v),
        None => base_zero,
    }) {
        return Some(Op::Const(0));
    }
    if dst_lane == Lane::I64
        && dst_count == 2
        && let (Some(v0), Some(v1)) = (slots[0], slots[1])
        && let (Some((s0, 0)), Some((s1, 1))) = (match_pack32_half(cx, v0), match_pack32_half(cx, v1))
        && s0 == s1
    {
        return Some(Op::Convert(Convert::Bitcast, s0));
    }
    let cand = slots[..c].iter().flatten().find_map(|&v| extract_source(cx, v, dst_lane.bytes())).map(|(s, _)| s);
    if let Some(src) = cand
        && cx.func.ty(src).bytes() == ty.bytes()
        && (0..c).all(|i| match slots[i] {
            Some(v) => extract_source(cx, v, dst_lane.bytes()) == Some((src, i as u8)),
            None => base == src && cx.func.ty(src) == ty,
        })
    {
        if cx.func.ty(src) == ty {
            if src != res_v {
                cx.alias(res, src);
            }
            return None;
        }
        return Some(Op::Convert(Convert::Bitcast, src));
    }
    if slots[..c].iter().all(Option::is_some) {
        let mut s0: Option<Value> = None;
        let mut s1: Option<Value> = None;
        let mut mask = [0_u8; 16];
        let mut ok = true;
        for (i, slot) in slots[..c].iter().enumerate() {
            let Some(v) = *slot else { ok = false; break };
            let Some((src, idx)) = extract_source(cx, v, dst_lane.bytes()) else { ok = false; break };
            if cx.func.ty(src) != ty {
                ok = false;
                break;
            }
            if s0.is_none() || s0 == Some(src) {
                s0 = Some(src);
                mask[i] = idx;
            } else if s1.is_none() || s1 == Some(src) {
                s1 = Some(src);
                mask[i] = dst_count + idx;
            } else {
                ok = false;
                break;
            }
        }
        if ok && let Some(a) = s0 {
            return Some(Op::Shuffle(a, s1.unwrap_or(a), ShuffleMask::Const(mask)));
        }
    }
    None
}

/// Simplifies vector binary zero and self identities.
pub(super) fn simplify_vec_binary(cx: &mut RewriteCx, res: Option<Value>, bin: Binary, a: Value, b: Value) -> Option<Op> {
    let res_v = res?;
    let (_, count) = cx.func.ty(res_v).vec_shape()?;
    let c = usize::from(count);
    let (za, zb) = (is_zero_vec(cx, a), is_zero_vec(cx, b));
    let alias = |cx: &mut RewriteCx, target: Value| {
        if target != res_v {
            cx.alias(res, target);
        }
    };
    match bin {
        Binary::Iadd | Binary::Bor | Binary::Bxor => {
            if zb {
                alias(cx, a);
                return None;
            }
            if za {
                alias(cx, b);
                return None;
            }
            if a == b {
                if bin == Binary::Bxor {
                    return Some(Op::Const(0));
                }
                if bin == Binary::Bor {
                    alias(cx, a);
                    return None;
                }
            }
        }
        Binary::Isub => {
            if zb {
                alias(cx, a);
                return None;
            }
            if a == b {
                return Some(Op::Const(0));
            }
        }
        Binary::Band => {
            if za || zb {
                return Some(Op::Const(0));
            }
            if a == b {
                alias(cx, a);
                return None;
            }
        }
        Binary::Ishl | Binary::Ushr | Binary::Sshr => {
            if zb || splat_scalar(cx, b, c).is_some_and(|s| cx.sconst(s) == Some(0)) {
                alias(cx, a);
                return None;
            }
        }
        _ => {}
    }
    None
}

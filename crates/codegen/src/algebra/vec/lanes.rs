use super::super::RewriteCx;
use super::{alias_if_diff, is_zero_vec, splat_scalar};
use crate::ir::{Binary, Convert, Lane, Op, ShuffleMask, Type, Value};

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
        let Op::Convert(Convert::Uextend, e0) = cx.func.insts[li.index()].op else {
            continue;
        };
        let Op::Binary(Binary::Ishl, hi_ext, sh) = cx.func.insts[hi.index()].op else {
            continue;
        };
        if cx.sconst(sh) != Some(32) {
            continue;
        }
        let Some(hei) = cx.def(hi_ext) else { continue };
        let Op::Convert(Convert::Uextend, e1) = cx.func.insts[hei.index()].op else {
            continue;
        };
        if cx.func.ty(e0) != Type::I32 || cx.func.ty(e1) != Type::I32 {
            continue;
        }
        if let (Some((s0, i0)), Some((s1, i1))) =
            (extract_source(cx, e0, 4), extract_source(cx, e1, 4))
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
pub(in super::super) fn simplify_bitcast(
    cx: &mut RewriteCx,
    res: Option<Value>,
    x: Value,
) -> Option<Op> {
    let res_v = res?;
    let ty = cx.func.ty(res_v);
    if cx.func.ty(x) == ty {
        alias_if_diff(cx, res, res_v, x);
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
        alias_if_diff(cx, res, res_v, inner);
        return None;
    }
    (cx.func.ty(inner).bytes() == ty.bytes()).then_some(Op::Convert(Convert::Bitcast, inner))
}

fn extract_shuffle_step(
    cx: &RewriteCx,
    res_ty: Type,
    (a, b, mask): (Value, Value, [u8; 16]),
    vec: &mut Value,
    idx: &mut u8,
) -> Result<Option<Op>, bool> {
    let Some((_, count)) = cx.func.ty(a).vec_shape() else {
        return Err(false);
    };
    let Some(&sel) = mask.get(usize::from(*idx)) else {
        return Err(false);
    };
    if sel < count {
        (*vec, *idx) = (a, sel);
        Err(true)
    } else if sel < 2 * count {
        (*vec, *idx) = (b, sel - count);
        Err(true)
    } else if res_ty.is_small_int() || res_ty.is_float() || res_ty.is_vector() {
        Ok(Some(Op::Const(0)))
    } else {
        Err(false)
    }
}

fn extract_step(
    cx: &mut RewriteCx,
    res: Option<Value>,
    res_v: Value,
    vec: &mut Value,
    idx: &mut u8,
) -> Result<Option<Op>, bool> {
    let res_ty = cx.func.ty(res_v);
    let Some(inst) = cx.def(*vec) else {
        return Err(false);
    };
    match cx.func.insts[inst.index()].op {
        Op::Const(0) if res_ty.is_small_int() || res_ty.is_float() || res_ty.is_vector() => {
            Ok(Some(Op::Const(0)))
        }
        Op::Convert(Convert::Bitcast, inner) if *idx == 0 && cx.func.ty(inner) == res_ty => {
            alias_if_diff(cx, res, res_v, inner);
            Ok(None)
        }
        Op::Insert(_, val, at) if at == *idx => {
            alias_if_diff(cx, res, res_v, val);
            Ok(None)
        }
        Op::Insert(prev, _, _) => {
            *vec = prev;
            Err(true)
        }
        Op::Shuffle(a, b, ShuffleMask::Const(mask)) => {
            extract_shuffle_step(cx, res_ty, (a, b, mask), vec, idx)
        }
        _ => Err(false),
    }
}

/// Simplifies `Extract` through `Insert`, `Shuffle`, scalar `Bitcast`, and zero vectors.
pub(in super::super) fn simplify_extract(
    cx: &mut RewriteCx,
    res: Option<Value>,
    mut vec: Value,
    mut idx: u8,
) -> Option<Op> {
    let res_v = res?;
    let orig = (vec, idx);
    for _ in 0..16 {
        match extract_step(cx, res, res_v, &mut vec, &mut idx) {
            Ok(done) => return done,
            Err(true) => continue,
            Err(false) => break,
        }
    }
    ((vec, idx) != orig).then_some(Op::Extract(vec, idx))
}

fn compose_shuffle(
    cx: &mut RewriteCx,
    res: Option<Value>,
    src: Value,
    m2: &[u8; 16],
    count: u8,
) -> Option<Op> {
    let res_v = res?;
    let c = usize::from(count);
    let &Op::Shuffle(x, y, ShuffleMask::Const(m1)) = cx.op(src)? else {
        return None;
    };
    let mut composed = [0_u8; 16];
    let (mut uses_x, mut uses_y) = (false, false);
    for i in 0..c {
        let sel = m2[i];
        if sel >= count {
            return None;
        }
        let inner = m1[usize::from(sel)];
        if inner < count {
            uses_x = true;
            composed[i] = inner;
        } else if inner < 2 * count {
            uses_y = true;
            composed[i] = inner;
        } else {
            return None;
        }
    }
    let (nx, ny) = match (uses_x, uses_y) {
        (true, false) => (x, x),
        (false, true) => {
            for item in &mut composed[..c] {
                *item -= count;
            }
            (y, y)
        }
        _ => (x, y),
    };
    let is_id = (0..count).all(|i| {
        composed[usize::from(i)] == i || (nx == ny && composed[usize::from(i)] == count + i)
    });
    if is_id {
        alias_if_diff(cx, res, res_v, nx);
        return None;
    }
    let is_splat = composed[..c].iter().all(|&v| v == composed[0]);
    (is_splat || cx.single_use(src)).then_some(Op::Shuffle(nx, ny, ShuffleMask::Const(composed)))
}

fn normalize_single_shuffle(
    a: Value,
    b: Value,
    mask: [u8; 16],
    count: u8,
) -> Option<(Value, [u8; 16])> {
    let c = usize::from(count);
    if mask[..c].iter().all(|&i| i < count) {
        return Some((a, mask));
    }
    if a != b && mask[..c].iter().all(|&i| i >= count && i < 2 * count) {
        let mut m = mask;
        for x in &mut m[..c] {
            *x -= count;
        }
        return Some((b, m));
    }
    if a == b {
        let mut m = mask;
        for x in &mut m[..c] {
            if *x >= count && *x < 2 * count {
                *x -= count;
            }
        }
        return Some((a, m));
    }
    None
}

fn shuffle_trivial(
    cx: &mut RewriteCx,
    res: Option<Value>,
    res_v: Value,
    (a, b): (Value, Value),
    mask: [u8; 16],
) -> Result<Option<Op>, (u8, usize)> {
    let ty = cx.func.ty(res_v);
    let (_, count) = ty.vec_shape().ok_or((0, 0))?;
    let c = usize::from(count);
    if (0..count)
        .all(|i| mask[usize::from(i)] == i || (a == b && mask[usize::from(i)] == count + i))
    {
        alias_if_diff(cx, res, res_v, a);
        return Ok(None);
    }
    if a != b && (0..count).all(|i| mask[usize::from(i)] == count + i) {
        alias_if_diff(cx, res, res_v, b);
        return Ok(None);
    }
    let (za, zb) = (is_zero_vec(cx, a), is_zero_vec(cx, b));
    if mask[..c].iter().all(|&i| {
        if i < count {
            za
        } else if i < 2 * count {
            zb
        } else {
            true
        }
    }) {
        return Ok(Some(Op::Const(0)));
    }
    Err((count, c))
}

/// Simplifies identity, single-source, composed, and all-zero constant `Shuffle` ops.
pub(in super::super) fn simplify_shuffle(
    cx: &mut RewriteCx,
    res: Option<Value>,
    a: Value,
    b: Value,
    mask: [u8; 16],
) -> Option<Op> {
    let res_v = res?;
    let ty = cx.func.ty(res_v);
    if cx.func.ty(a) != ty || cx.func.ty(b) != ty {
        return None;
    }
    let count = match shuffle_trivial(cx, res, res_v, (a, b), mask) {
        Ok(done) => return done,
        Err((0, _)) => return None,
        Err((count, _)) => count,
    };
    let (src, norm_m) = normalize_single_shuffle(a, b, mask, count)?;
    compose_shuffle(cx, res, src, &norm_m, count).or_else(|| {
        (a != b || norm_m != mask).then_some(Op::Shuffle(src, src, ShuffleMask::Const(norm_m)))
    })
}

fn collect_insert_slots(
    cx: &RewriteCx,
    mut cur: Value,
    lane: u8,
    val: Value,
    dst_count: u8,
) -> ([Option<Value>; 16], Value) {
    let c = usize::from(dst_count);
    let mut slots: [Option<Value>; 16] = [None; 16];
    slots[usize::from(lane)] = Some(val);
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
    if slots[..c].iter().any(Option::is_none)
        && let Some(sp) = splat_scalar(cx, cur, c)
    {
        for s in &mut slots[..c] {
            if s.is_none() {
                *s = Some(sp);
            }
        }
    }
    (slots, cur)
}

fn insert_as_shuffle(
    cx: &RewriteCx,
    slots: &[Option<Value>; 16],
    (ty, dst_lane, dst_count): (Type, Lane, u8),
) -> Option<Op> {
    let c = usize::from(dst_count);
    let mut s0: Option<Value> = None;
    let mut s1: Option<Value> = None;
    let mut mask = [0_u8; 16];
    for (i, slot) in slots[..c].iter().enumerate() {
        let v = (*slot)?;
        let (src, idx) = extract_source(cx, v, dst_lane.bytes())?;
        if cx.func.ty(src) != ty {
            return None;
        }
        if s0.is_none() || s0 == Some(src) {
            s0 = Some(src);
            mask[i] = idx;
        } else if s1.is_none() || s1 == Some(src) {
            s1 = Some(src);
            mask[i] = dst_count + idx;
        } else {
            return None;
        }
    }
    s0.map(|a| Op::Shuffle(a, s1.unwrap_or(a), ShuffleMask::Const(mask)))
}

fn peel_bitcast(cx: &RewriteCx, mut v: Value, bytes: u32) -> Value {
    for _ in 0..4 {
        if let Some(&Op::Convert(Convert::Bitcast, inner)) = cx.op(v)
            && cx.func.ty(inner).bytes() == bytes
        {
            v = inner;
        } else {
            break;
        }
    }
    v
}

fn half64_source(cx: &RewriteCx, v: Value) -> Option<(Value, u8)> {
    let inner = peel_bitcast(cx, v, 8);
    let &Op::Extract(src, idx) = cx.op(inner)? else {
        return None;
    };
    let (lane, count) = cx.func.ty(src).vec_shape()?;
    (lane.bytes() == 8 && count == 2 && idx < 2).then(|| (peel_bitcast(cx, src, 16), idx))
}

fn merge_half_mulhi(cx: &mut RewriteCx, v0: Value, v1: Value) -> Option<Op> {
    let (lo, hi) = (peel_bitcast(cx, v0, 8), peel_bitcast(cx, v1, 8));
    let (&Op::Binary(op @ (Binary::Umulhi | Binary::Smulhi), a0, b0), &Op::Binary(op2, a1, b1)) =
        (cx.op(lo)?, cx.op(hi)?)
    else {
        return None;
    };
    let half_ty = Type::Vec {
        lane: Lane::I16,
        count: 4,
    };
    if op != op2 || cx.func.ty(lo) != half_ty || cx.func.ty(hi) != half_ty {
        return None;
    }
    let ((sa, 0), (sb, 0)) = (half64_source(cx, a0)?, half64_source(cx, b0)?) else {
        return None;
    };
    let (s1a, s1b) = (half64_source(cx, a1)?, half64_source(cx, b1)?);
    if !((s1a == (sa, 1) && s1b == (sb, 1)) || (s1a == (sb, 1) && s1b == (sa, 1))) {
        return None;
    }
    let wide_ty = Type::Vec {
        lane: Lane::I16,
        count: 8,
    };
    let wa = if cx.func.ty(sa) == wide_ty {
        sa
    } else {
        cx.emit(wide_ty, Op::Convert(Convert::Bitcast, sa))
    };
    let wb = if cx.func.ty(sb) == wide_ty {
        sb
    } else {
        cx.emit(wide_ty, Op::Convert(Convert::Bitcast, sb))
    };
    let out = cx.emit(wide_ty, Op::Binary(op, wa, wb));
    Some(Op::Convert(Convert::Bitcast, out))
}

fn insert_as_bitcast(
    cx: &mut RewriteCx,
    res: Option<Value>,
    slots: &[Option<Value>; 16],
    base: Value,
    (ty, dst_lane, c): (Type, Lane, usize),
) -> Result<Option<Op>, ()> {
    let res_v = res.ok_or(())?;
    if dst_lane == Lane::I64
        && c == 2
        && let (Some(v0), Some(v1)) = (slots[0], slots[1])
    {
        if let (Some((s0, 0)), Some((s1, 1))) =
            (match_pack32_half(cx, v0), match_pack32_half(cx, v1))
            && s0 == s1
        {
            return Ok(Some(Op::Convert(Convert::Bitcast, s0)));
        }
        if let Some(op) = merge_half_mulhi(cx, v0, v1) {
            return Ok(Some(op));
        }
    }
    let cand = slots[..c]
        .iter()
        .flatten()
        .find_map(|&v| extract_source(cx, v, dst_lane.bytes()))
        .map(|(s, _)| s);
    let Some(src) = cand else { return Err(()) };
    let ok = cx.func.ty(src).bytes() == ty.bytes()
        && (0..c).all(|i| match slots[i] {
            Some(v) => extract_source(cx, v, dst_lane.bytes()) == Some((src, i as u8)),
            None => base == src && cx.func.ty(src) == ty,
        });
    if !ok {
        return Err(());
    }
    if cx.func.ty(src) == ty {
        alias_if_diff(cx, res, res_v, src);
        return Ok(None);
    }
    Ok(Some(Op::Convert(Convert::Bitcast, src)))
}

/// Folds `Insert` chains reconstructing vectors from `Extract`s, overwritten lanes, and all-zero vectors.
pub(in super::super) fn simplify_insert(
    cx: &mut RewriteCx,
    res: Option<Value>,
    vec: Value,
    val: Value,
    lane: u8,
) -> Option<Op> {
    let res_v = res?;
    let ty = cx.func.ty(res_v);
    let (dst_lane, dst_count) = ty.vec_shape()?;
    let c = usize::from(dst_count);
    if lane >= dst_count {
        return None;
    }
    if extract_source(cx, val, dst_lane.bytes()) == Some((vec, lane)) {
        alias_if_diff(cx, res, res_v, vec);
        return None;
    }
    if let Some(&Op::Insert(prev, _, at)) = cx.op(vec)
        && at == lane
    {
        return Some(Op::Insert(prev, val, lane));
    }
    let (slots, base) = collect_insert_slots(cx, vec, lane, val, dst_count);
    let base_zero = is_zero_vec(cx, base);
    if slots[..c]
        .iter()
        .all(|s| s.map_or(base_zero, |v| is_zero_vec(cx, v)))
    {
        return Some(Op::Const(0));
    }
    if let Ok(done) = insert_as_bitcast(cx, res, &slots, base, (ty, dst_lane, c)) {
        return done;
    }
    insert_as_shuffle(cx, &slots, (ty, dst_lane, dst_count))
}

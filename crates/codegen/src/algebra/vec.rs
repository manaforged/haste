use super::RewriteCx;
use crate::ir::{Binary, Convert, IntCC, Lane, Op, ShuffleMask, TargetOpId, TinyArgs, Type, Unary, Value};

fn is_zero_vec(cx: &RewriteCx, v: Value) -> bool {
    cx.sconst(v) == Some(0) || cx.func.is_zero_vec(&|value| cx.def(value), v)
}

/// The vector whose lane sign bits equal those of `v`, when `v` only spreads each sign bit of it across its lane.
fn sign_source(cx: &RewriteCx, v: Value) -> Option<Value> {
    match *cx.op(v)? {
        Op::Icmp(IntCC::SignedLessThan, x, zero) if cx.func.ty(x).lane().is_some_and(Lane::is_int) && is_zero_vec(cx, zero) => Some(x),
        Op::TargetOp(TargetOpId::X64Pcmpgtb256, args) => match *args.as_slice() {
            [zero, x] if is_zero_vec(cx, zero) => Some(x),
            _ => None,
        },
        _ => None,
    }
}

/// `select(x < 0, -x, x)` on sixteen bytes as `umin(x, -x)`, the same unsigned magnitude in one byte minimum.
pub(super) fn byte_abs(cx: &RewriteCx, cond: Value, t: Value, f: Value) -> Option<Op> {
    let bytes = Type::Vec { lane: Lane::I8, count: 16 };
    let negated = matches!(cx.op(t)?, &Op::Unary(Unary::Ineg, x) if x == f);
    (cx.func.ty(f) == bytes && negated && sign_source(cx, cond) == Some(f)).then_some(Op::Binary(Binary::Umin, f, t))
}

/// Reads the lane sign bits of a compare against zero from the compared vector itself.
pub(super) fn simplify_movemask(cx: &RewriteCx, op: &Op) -> Option<Op> {
    match *op {
        Op::Reduce(crate::ir::ReduceOp::Movemask, v) => sign_source(cx, v).map(|x| Op::Reduce(crate::ir::ReduceOp::Movemask, x)),
        Op::TargetOp(TargetOpId::X64Pmovmskb256, args) => match *args.as_slice() {
            [v] => sign_source(cx, v).map(|x| Op::TargetOp(TargetOpId::X64Pmovmskb256, TinyArgs::one(x))),
            _ => None,
        },
        _ => None,
    }
}

fn movemask_of(cx: &RewriteCx, v: Value) -> Option<(bool, Value)> {
    match *cx.op(v)? {
        Op::Reduce(crate::ir::ReduceOp::Movemask, x) => Some((false, x)),
        Op::TargetOp(TargetOpId::X64Pmovmskb256, args) => match *args.as_slice() {
            [x] => Some((true, x)),
            _ => None,
        },
        _ => None,
    }
}

/// Or-s two masks of lane sign bits as one mask of the or of their vectors, which a mask of that or elsewhere then shares.
fn merge_movemasks(cx: &mut RewriteCx, a: Value, b: Value) -> Option<Op> {
    let ((wide, x), (other, y)) = (movemask_of(cx, a)?, movemask_of(cx, b)?);
    let ty = cx.func.ty(x);
    if wide != other || ty != cx.func.ty(y) || cx.func.ty(a) != cx.func.ty(b) {
        return None;
    }
    let both = cx.emit(ty, Op::Binary(Binary::Bor, x, y));
    Some(if wide { Op::TargetOp(TargetOpId::X64Pmovmskb256, TinyArgs::one(both)) } else { Op::Reduce(crate::ir::ReduceOp::Movemask, both) })
}

fn merge_vec_bitwise_direct(cx: &mut RewriteCx, bin: Binary, a: Value, b: Value) -> Option<Op> {
    if bin == Binary::Bor && let Some(op) = merge_movemasks(cx, a, b) {
        return Some(op);
    }
    if a == b || !cx.single_use(a) || !cx.single_use(b) {
        return None;
    }
    let (Some(&Op::Extract(va, la)), Some(&Op::Extract(vb, lb))) = (cx.op(a), cx.op(b)) else { return None };
    let ty = cx.func.ty(va);
    if la != lb || ty != cx.func.ty(vb) || !ty.lane().is_some_and(Lane::is_int) {
        return None;
    }
    let inner = emit_vec_bitwise(cx, ty, bin, (va, vb), 6);
    Some(Op::Extract(inner, la))
}

fn band_common(cx: &RewriteCx, a: Value, b: Value) -> Option<(Value, Value, Value)> {
    if !cx.single_use(a) || !cx.single_use(b) {
        return None;
    }
    let (Some(&Op::Binary(Binary::Band, a0, a1)), Some(&Op::Binary(Binary::Band, b0, b1))) = (cx.op(a), cx.op(b)) else { return None };
    let ty = cx.func.ty(a);
    for (xa, ca) in [(a0, a1), (a1, a0)] {
        for (xb, cb) in [(b0, b1), (b1, b0)] {
            if ca == cb || cx.uconst(ca, ty).is_some_and(|ka| cx.uconst(cb, ty) == Some(ka)) {
                return Some((xa, xb, ca));
            }
        }
    }
    None
}

/// Fuses scalar bitwise ops over movemasks or single-use lane extracts (directly or under a shared `Band` mask) into one vector op followed by one extract/movemask.
pub(super) fn merge_vec_bitwise(cx: &mut RewriteCx, bin: Binary, a: Value, b: Value) -> Option<Op> {
    if let Some(op) = merge_vec_bitwise_direct(cx, bin, a, b) {
        return Some(op);
    }
    if matches!(bin, Binary::Bor | Binary::Bxor)
        && let Some((xa, xb, common)) = band_common(cx, a, b)
        && let Some(inner_op) = merge_vec_bitwise_direct(cx, bin, xa, xb)
    {
        let merged = cx.emit(cx.func.ty(a), inner_op);
        return Some(Op::Binary(Binary::Band, merged, common));
    }
    None
}

fn same_vec_shift(cx: &RewriteCx, ty: Type, sh1: Value, sh2: Value) -> bool {
    sh1 == sh2
        || ty.vec_shape().is_some_and(|(lane, count)| {
            splat_int_imm(cx, sh1, lane, usize::from(count)).is_some_and(|k| splat_int_imm(cx, sh2, lane, usize::from(count)) == Some(k))
        })
}

fn sink_vec_bitwise_inner(cx: &mut RewriteCx, res_ty: Type, bin: Binary, (a, b): (Value, Value), depth: u8) -> Option<Op> {
    if a == b {
        return None;
    }
    if let (Some(&Op::Convert(Convert::Bitcast, x)), Some(&Op::Convert(Convert::Bitcast, y))) = (cx.op(a), cx.op(b)) {
        let ty = cx.func.ty(x);
        if ty == cx.func.ty(y) && ty.lane().is_some_and(Lane::is_int) && ty.bytes() == res_ty.bytes() {
            return Some(Op::Convert(Convert::Bitcast, emit_vec_bitwise(cx, ty, bin, (x, y), depth)));
        }
    }
    if !cx.single_use(a) || !cx.single_use(b) {
        return None;
    }
    match (cx.op(a), cx.op(b)) {
        (Some(&Op::Convert(Convert::Ireduce, x)), Some(&Op::Convert(Convert::Ireduce, y))) => {
            let ty = cx.func.ty(x);
            (ty == cx.func.ty(y) && ty.lane().is_some_and(Lane::is_int)).then(|| Op::Convert(Convert::Ireduce, emit_vec_bitwise(cx, ty, bin, (x, y), depth)))
        }
        (Some(&Op::Binary(sop @ (Binary::Ushr | Binary::Ishl | Binary::Sshr), x, sh1)), Some(&Op::Binary(sop2, y, sh2))) if sop == sop2 => {
            let ty = cx.func.ty(x);
            (ty == cx.func.ty(y) && ty.lane().is_some_and(Lane::is_int) && same_vec_shift(cx, ty, sh1, sh2))
                .then(|| Op::Binary(sop, emit_vec_bitwise(cx, ty, bin, (x, y), depth), sh1))
        }
        _ => None,
    }
}

fn emit_vec_bitwise(cx: &mut RewriteCx, ty: Type, bin: Binary, (a, b): (Value, Value), depth: u8) -> Value {
    if a == b {
        return if bin == Binary::Bxor { cx.emit(ty, Op::Const(0)) } else { a };
    }
    let op = if depth > 0 { sink_vec_bitwise_inner(cx, ty, bin, (a, b), depth - 1) } else { None }.unwrap_or(Op::Binary(bin, a, b));
    cx.emit(ty, op)
}

/// Moves a vector bitwise op inside matching bitcasts, truncations, or identical shifts so common prefixes collapse.
fn sink_bitcast(cx: &mut RewriteCx, res_v: Value, bin: Binary, (a, b): (Value, Value)) -> Option<Op> {
    sink_vec_bitwise_inner(cx, cx.func.ty(res_v), bin, (a, b), 6)
}

fn splat_scalar(cx: &RewriteCx, v: Value, count: usize) -> Option<Value> {
    let inst = cx.def(v)?;
    match &cx.func.insts[inst.index()].op {
        &Op::Insert(_, scalar, 0) if count == 1 => Some(scalar),
        Op::Shuffle(a, b, ShuffleMask::Const(mask)) => {
            let head = mask[0];
            if usize::from(head) >= 2 * count || mask[..count].iter().any(|&i| i != head) {
                return None;
            }
            let src = if usize::from(head) < count { *a } else { *b };
            let target = head % (count as u8);
            match cx.op(src)? {
                &Op::Insert(_, scalar, at) if at == target => Some(scalar),
                _ => None,
            }
        }
        _ => None,
    }
}

fn splat_int_imm(cx: &RewriteCx, v: Value, lane: Lane, count: usize) -> Option<u64> {
    if !lane.is_int() {
        return None;
    }
    if is_zero_vec(cx, v) {
        return Some(0);
    }
    if let Some(&Op::Unary(Unary::Bnot, z)) = cx.op(v)
        && is_zero_vec(cx, z)
    {
        return Some(lane.to_type().unsigned_max());
    }
    cx.uconst(splat_scalar(cx, v, count)?, lane.to_type())
}

fn is_all_ones_vec(cx: &RewriteCx, mut v: Value) -> bool {
    for _ in 0..4 {
        if let Some(&Op::Unary(Unary::Bnot, z)) = cx.op(v)
            && is_zero_vec(cx, z)
        {
            return true;
        }
        if let Some((lane, count)) = cx.func.ty(v).vec_shape()
            && lane.is_int()
            && splat_int_imm(cx, v, lane, usize::from(count)) == Some(lane.to_type().unsigned_max())
        {
            return true;
        }
        let Some(&Op::Convert(Convert::Bitcast, inner)) = cx.op(v) else { return false };
        v = inner;
    }
    false
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

fn extract_shuffle_step(cx: &RewriteCx, res_ty: Type, (a, b, mask): (Value, Value, [u8; 16]), vec: &mut Value, idx: &mut u8) -> Result<Option<Op>, bool> {
    let Some((_, count)) = cx.func.ty(a).vec_shape() else { return Err(false) };
    let Some(&sel) = mask.get(usize::from(*idx)) else { return Err(false) };
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

fn extract_step(cx: &mut RewriteCx, res: Option<Value>, res_v: Value, vec: &mut Value, idx: &mut u8) -> Result<Option<Op>, bool> {
    let res_ty = cx.func.ty(res_v);
    let Some(inst) = cx.def(*vec) else { return Err(false) };
    match cx.func.insts[inst.index()].op {
        Op::Const(0) if res_ty.is_small_int() || res_ty.is_float() || res_ty.is_vector() => Ok(Some(Op::Const(0))),
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
        Op::Shuffle(a, b, ShuffleMask::Const(mask)) => extract_shuffle_step(cx, res_ty, (a, b, mask), vec, idx),
        _ => Err(false),
    }
}

/// Simplifies `Extract` through `Insert`, `Shuffle`, scalar `Bitcast`, and zero vectors.
pub(super) fn simplify_extract(cx: &mut RewriteCx, res: Option<Value>, mut vec: Value, mut idx: u8) -> Option<Op> {
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

fn compose_shuffle(cx: &mut RewriteCx, res: Option<Value>, src: Value, m2: &[u8; 16], count: u8) -> Option<Op> {
    let res_v = res?;
    let c = usize::from(count);
    let &Op::Shuffle(x, y, ShuffleMask::Const(m1)) = cx.op(src)? else { return None };
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
    let is_id = (0..count).all(|i| composed[usize::from(i)] == i || (nx == ny && composed[usize::from(i)] == count + i));
    if is_id {
        if nx != res_v {
            cx.alias(res, nx);
        }
        return None;
    }
    let is_splat = composed[..c].iter().all(|&v| v == composed[0]);
    (is_splat || cx.single_use(src)).then_some(Op::Shuffle(nx, ny, ShuffleMask::Const(composed)))
}

fn normalize_single_shuffle(a: Value, b: Value, mask: [u8; 16], count: u8) -> Option<(Value, [u8; 16])> {
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

fn shuffle_trivial(cx: &mut RewriteCx, res: Option<Value>, res_v: Value, (a, b): (Value, Value), mask: [u8; 16]) -> Result<Option<Op>, (u8, usize)> {
    let ty = cx.func.ty(res_v);
    let (_, count) = ty.vec_shape().ok_or((0, 0))?;
    let c = usize::from(count);
    if (0..count).all(|i| mask[usize::from(i)] == i || (a == b && mask[usize::from(i)] == count + i)) {
        alias_if_diff(cx, res, res_v, a);
        return Ok(None);
    }
    if a != b && (0..count).all(|i| mask[usize::from(i)] == count + i) {
        alias_if_diff(cx, res, res_v, b);
        return Ok(None);
    }
    let (za, zb) = (is_zero_vec(cx, a), is_zero_vec(cx, b));
    if mask[..c].iter().all(|&i| if i < count { za } else if i < 2 * count { zb } else { true }) {
        return Ok(Some(Op::Const(0)));
    }
    Err((count, c))
}

/// Simplifies identity, single-source, composed, and all-zero constant `Shuffle` ops.
pub(super) fn simplify_shuffle(cx: &mut RewriteCx, res: Option<Value>, a: Value, b: Value, mask: [u8; 16]) -> Option<Op> {
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
    compose_shuffle(cx, res, src, &norm_m, count).or_else(|| (a != b || norm_m != mask).then_some(Op::Shuffle(src, src, ShuffleMask::Const(norm_m))))
}

fn collect_insert_slots(cx: &RewriteCx, mut cur: Value, lane: u8, val: Value, dst_count: u8) -> ([Option<Value>; 16], Value) {
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

fn insert_as_shuffle(cx: &RewriteCx, slots: &[Option<Value>; 16], (ty, dst_lane, dst_count): (Type, Lane, u8)) -> Option<Op> {
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

fn insert_as_bitcast(cx: &mut RewriteCx, res: Option<Value>, slots: &[Option<Value>; 16], base: Value, (ty, dst_lane, c): (Type, Lane, usize)) -> Result<Option<Op>, ()> {
    let res_v = res.ok_or(())?;
    if dst_lane == Lane::I64
        && c == 2
        && let (Some(v0), Some(v1)) = (slots[0], slots[1])
        && let (Some((s0, 0)), Some((s1, 1))) = (match_pack32_half(cx, v0), match_pack32_half(cx, v1))
        && s0 == s1
    {
        return Ok(Some(Op::Convert(Convert::Bitcast, s0)));
    }
    let cand = slots[..c].iter().flatten().find_map(|&v| extract_source(cx, v, dst_lane.bytes())).map(|(s, _)| s);
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
pub(super) fn simplify_insert(cx: &mut RewriteCx, res: Option<Value>, vec: Value, val: Value, lane: u8) -> Option<Op> {
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
    if slots[..c].iter().all(|s| s.map_or(base_zero, |v| is_zero_vec(cx, v))) {
        return Some(Op::Const(0));
    }
    if let Ok(done) = insert_as_bitcast(cx, res, &slots, base, (ty, dst_lane, c)) {
        return done;
    }
    insert_as_shuffle(cx, &slots, (ty, dst_lane, dst_count))
}

fn alias_if_diff(cx: &mut RewriteCx, res: Option<Value>, res_v: Value, target: Value) {
    if target != res_v {
        cx.alias(res, target);
    }
}

fn simplify_vec_band_bor(cx: &mut RewriteCx, res: Option<Value>, res_v: Value, is_band: bool, (a, b): (Value, Value)) -> Option<Op> {
    let (za, zb) = (is_zero_vec(cx, a), is_zero_vec(cx, b));
    let (oa, ob) = (is_all_ones_vec(cx, a), is_all_ones_vec(cx, b));
    let bnot_pair = cx.op(b) == Some(&Op::Unary(Unary::Bnot, a)) || cx.op(a) == Some(&Op::Unary(Unary::Bnot, b));
    if is_band {
        if za || zb || bnot_pair {
            return Some(Op::Const(0));
        }
        if a == b || ob {
            alias_if_diff(cx, res, res_v, a);
        } else if oa {
            alias_if_diff(cx, res, res_v, b);
        }
    } else if zb || a == b || oa {
        alias_if_diff(cx, res, res_v, a);
    } else if za || ob {
        alias_if_diff(cx, res, res_v, b);
    }
    None
}

fn simplify_vec_bxor(cx: &mut RewriteCx, res: Option<Value>, res_v: Value, a: Value, b: Value) -> Option<Op> {
    if a == b {
        return Some(Op::Const(0));
    }
    if is_zero_vec(cx, b) {
        alias_if_diff(cx, res, res_v, a);
    } else if is_zero_vec(cx, a) {
        alias_if_diff(cx, res, res_v, b);
    } else if is_all_ones_vec(cx, b) {
        return Some(Op::Unary(Unary::Bnot, a));
    } else if is_all_ones_vec(cx, a) {
        return Some(Op::Unary(Unary::Bnot, b));
    }
    None
}

fn simplify_vec_minmax(cx: &mut RewriteCx, res: Option<Value>, res_v: Value, bin: Binary, (a, b): (Value, Value)) -> Option<Op> {
    if a == b {
        alias_if_diff(cx, res, res_v, a);
        return None;
    }
    let (za, zb) = (is_zero_vec(cx, a), is_zero_vec(cx, b));
    let (oa, ob) = (is_all_ones_vec(cx, a), is_all_ones_vec(cx, b));
    if bin == Binary::Umin {
        if za || zb {
            return Some(Op::Const(0));
        }
        if ob {
            alias_if_diff(cx, res, res_v, a);
        } else if oa {
            alias_if_diff(cx, res, res_v, b);
        }
    } else if bin == Binary::Umax {
        if zb || oa {
            alias_if_diff(cx, res, res_v, a);
        } else if za || ob {
            alias_if_diff(cx, res, res_v, b);
        }
    }
    None
}

fn simplify_vec_sat(cx: &mut RewriteCx, res: Option<Value>, res_v: Value, bin: Binary, (a, b): (Value, Value)) -> Option<Op> {
    let (za, zb) = (is_zero_vec(cx, a), is_zero_vec(cx, b));
    if matches!(bin, Binary::SaddSat | Binary::UaddSat) {
        let u = bin == Binary::UaddSat;
        if zb || (u && is_all_ones_vec(cx, a)) {
            alias_if_diff(cx, res, res_v, a);
        } else if za || (u && is_all_ones_vec(cx, b)) {
            alias_if_diff(cx, res, res_v, b);
        }
        return None;
    }
    let u = bin == Binary::UsubSat;
    if a == b || (u && (za || is_all_ones_vec(cx, b))) {
        return Some(Op::Const(0));
    }
    if zb {
        alias_if_diff(cx, res, res_v, a);
    }
    None
}

fn simplify_vec_add_sub(cx: &mut RewriteCx, res: Option<Value>, res_v: Value, bin: Binary, (a, b): (Value, Value)) -> Option<Op> {
    let (za, zb) = (is_zero_vec(cx, a), is_zero_vec(cx, b));
    if bin == Binary::Iadd {
        if zb {
            alias_if_diff(cx, res, res_v, a);
        } else if za {
            alias_if_diff(cx, res, res_v, b);
        } else if let Some(&Op::Unary(Unary::Ineg, x)) = cx.op(b) {
            return Some(if x == a { Op::Const(0) } else { Op::Binary(Binary::Isub, a, x) });
        } else if let Some(&Op::Unary(Unary::Ineg, x)) = cx.op(a) {
            return Some(if x == b { Op::Const(0) } else { Op::Binary(Binary::Isub, b, x) });
        }
        return None;
    }
    if zb {
        alias_if_diff(cx, res, res_v, a);
        return None;
    }
    if a == b {
        return Some(Op::Const(0));
    }
    if za {
        return Some(Op::Unary(Unary::Ineg, b));
    }
    match cx.op(b) {
        Some(&Op::Unary(Unary::Ineg, x)) => Some(Op::Binary(Binary::Iadd, a, x)),
        _ => None,
    }
}

fn simplify_vec_mul_shift(cx: &mut RewriteCx, res: Option<Value>, res_v: Value, bin: Binary, (a, b): (Value, Value)) -> Option<Op> {
    let (lane, count) = cx.func.ty(res_v).vec_shape()?;
    let c = usize::from(count);
    let (za, zb) = (is_zero_vec(cx, a), is_zero_vec(cx, b));
    if bin == Binary::Imul {
        if za || zb || splat_int_imm(cx, a, lane, c) == Some(0) || splat_int_imm(cx, b, lane, c) == Some(0) {
            return Some(Op::Const(0));
        }
        if splat_int_imm(cx, b, lane, c) == Some(1) {
            alias_if_diff(cx, res, res_v, a);
        } else if splat_int_imm(cx, a, lane, c) == Some(1) {
            alias_if_diff(cx, res, res_v, b);
        }
        return None;
    }
    if za {
        return Some(Op::Const(0));
    }
    if zb || splat_int_imm(cx, b, lane, c).is_some_and(|imm| (imm as u32) & (lane.bits() - 1) == 0) {
        alias_if_diff(cx, res, res_v, a);
    }
    None
}

/// Simplifies vector binary algebraic, zero, all-ones, shift, and saturation identities.
pub(super) fn simplify_vec_binary(cx: &mut RewriteCx, res: Option<Value>, bin: Binary, a: Value, b: Value) -> Option<Op> {
    let res_v = res?;
    if cx.func.ty(res_v).vec_shape().is_none() {
        return None;
    }
    match bin {
        Binary::Band | Binary::Bor | Binary::Bxor if a != b && let Some(op) = sink_bitcast(cx, res_v, bin, (a, b)) => Some(op),
        Binary::Band | Binary::Bor => simplify_vec_band_bor(cx, res, res_v, bin == Binary::Band, (a, b)),
        Binary::Bxor => simplify_vec_bxor(cx, res, res_v, a, b),
        Binary::Iadd | Binary::Isub => simplify_vec_add_sub(cx, res, res_v, bin, (a, b)),
        Binary::Imul | Binary::Ishl | Binary::Ushr | Binary::Sshr => simplify_vec_mul_shift(cx, res, res_v, bin, (a, b)),
        Binary::SaddSat | Binary::UaddSat | Binary::SsubSat | Binary::UsubSat => simplify_vec_sat(cx, res, res_v, bin, (a, b)),
        Binary::Umin | Binary::Umax | Binary::Smin | Binary::Smax => simplify_vec_minmax(cx, res, res_v, bin, (a, b)),
        _ => None,
    }
}

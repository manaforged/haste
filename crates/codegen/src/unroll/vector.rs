use super::count::{counted, head_defs, outside_preds};
use super::memset::{Emit, outside_uses, steps};
use super::vector_shape::{Access, Kind, LANES, Shape, analyze};
use crate::cfg::Loop;
use crate::ir::{Binary, Block, BlockData, Convert, Function, Inst, IntCC, Lane, Op, ReduceOp, ShuffleMask, TargetOpId, TinyArgs, Type, Unary, Value};

const VEC: Type = Type::Vec { lane: Lane::I8, count: LANES as u8 };
const WORDS: Type = Type::Vec { lane: Lane::I16, count: (LANES / 2) as u8 };
const QUADS: Type = Type::Vec { lane: Lane::I64, count: 2 };
const MAX_BODY: usize = 96;
const MAX_BLOCKS: usize = 6;
const LO_BYTE_WEIGHTS: i64 = 0x0706_0504_0302_0100_u64 as i64;
const HI_BYTE_WEIGHTS: i64 = 0x0F0E_0D0C_0B0A_0908_u64 as i64;

/// Whether every load may run ahead of the store of the same iteration: `None` when the distance is unknown until run time.
fn static_safe(load: Access, store: Access) -> Option<bool> {
    (load.idx == store.idx && load.base == store.base).then(|| {
        let d = i64::from(store.off) - i64::from(load.off);
        !(1..LANES).contains(&d)
    })
}

fn splat(e: &mut Emit<'_>, scalar: Value) -> Value {
    splat_in(e, VEC, scalar)
}

pub(crate) fn splat_in(e: &mut Emit<'_>, ty: Type, scalar: Value) -> Value {
    let zero = e.op(Op::Const(0), ty);
    let first = e.op(Op::Insert(zero, scalar, 0), ty);
    e.op(Op::Shuffle(first, first, ShuffleMask::Const([0; 16])), ty)
}

fn byte_weights(e: &mut Emit<'_>) -> Value {
    let lo = e.konst(LO_BYTE_WEIGHTS, Type::I64);
    let hi = e.konst(HI_BYTE_WEIGHTS, Type::I64);
    let zero = e.op(Op::Const(0), QUADS);
    let first = e.op(Op::Insert(zero, lo, 0), QUADS);
    let pair = e.op(Op::Insert(first, hi, 1), QUADS);
    e.op(Op::Convert(Convert::Bitcast, pair), VEC)
}

/// The sum of the sixteen unsigned bytes of `bytes`: adjacent byte pairs into 16-bit lanes, then one horizontal add.
fn byte_sum(e: &mut Emit<'_>, bytes: Value) -> Value {
    let words = e.op(Op::Convert(Convert::Bitcast, bytes), WORDS);
    let low = e.konst(0xFF, Type::I16);
    let low = splat_in(e, WORDS, low);
    let even = e.op(Op::Binary(Binary::Band, words, low), WORDS);
    let eight = e.konst(8, Type::I16);
    let eight = splat_in(e, WORDS, eight);
    let odd = e.op(Op::Binary(Binary::Ushr, words, eight), WORDS);
    let pairs = e.op(Op::Binary(Binary::Iadd, even, odd), WORDS);
    e.op(Op::Reduce(ReduceOp::Add, pairs), Type::I16)
}

/// The two `I64` partial sums of the sixteen unsigned bytes of `bytes`: `psadbw` against zero.
fn sad_halves(e: &mut Emit<'_>, bytes: Value) -> Value {
    let zero = e.op(Op::Const(0), VEC);
    e.op(Op::TargetOp(TargetOpId::X64Psadbw, TinyArgs::two(bytes, zero)), QUADS)
}

/// Converts `total` to `ty` and adds it to `into`, wrapping.
fn add_resized(e: &mut Emit<'_>, into: Value, total: Value) -> Value {
    let ty = e.func.ty(into);
    let wide = match e.func.ty(total).bytes().cmp(&ty.bytes()) {
        std::cmp::Ordering::Less => e.op(Op::Convert(Convert::Uextend, total), ty),
        std::cmp::Ordering::Equal => total,
        std::cmp::Ordering::Greater => e.op(Op::Convert(Convert::Ireduce, total), ty),
    };
    e.op(Op::Binary(Binary::Iadd, into, wide), ty)
}

fn byte_shift(e: &mut Emit<'_>, op: Binary, x: Value, by: i64) -> Value {
    let words = e.op(Op::Convert(Convert::Bitcast, x), WORDS);
    let amount = e.konst(by, Type::I16);
    let count = splat_in(e, WORDS, amount);
    let shifted = e.op(Op::Binary(op, words, count), WORDS);
    let bytes = e.op(Op::Convert(Convert::Bitcast, shifted), VEC);
    let keep = if op == Binary::Ushr { 0xFF >> by } else { (0xFF << by) & 0xFF };
    let keep = e.konst(keep, Type::I8);
    let mask = splat(e, keep);
    e.op(Op::Binary(Binary::Band, bytes, mask), VEC)
}

fn new_block(func: &mut Function, params: Vec<Value>, insts: Vec<Inst>) -> Block {
    func.blocks.push(BlockData { params, insts });
    Block(func.blocks.len() as u32 - 1)
}

fn fresh(func: &mut Function, like: &[Value]) -> Vec<Value> {
    like.iter().map(|&v| func.ty(v)).collect::<Vec<_>>().into_iter().map(|ty| func.value(ty)).collect()
}

/// The minimum or maximum that `select(cmp, x, y)` computes when `cmp` orders `x` and `y` themselves.
fn min_max(func: &Function, def: Option<&Op>, x: Value, y: Value) -> Option<Binary> {
    let Some(&Op::Icmp(cc, a, b)) = def else { return None };
    let low_first = match cc {
        IntCC::UnsignedLessThan | IntCC::UnsignedLessThanOrEqual => (Binary::Umin, Binary::Umax),
        IntCC::UnsignedGreaterThan | IntCC::UnsignedGreaterThanOrEqual => (Binary::Umax, Binary::Umin),
        IntCC::SignedLessThan | IntCC::SignedLessThanOrEqual => (Binary::Smin, Binary::Smax),
        IntCC::SignedGreaterThan | IntCC::SignedGreaterThanOrEqual => (Binary::Smax, Binary::Smin),
        _ => return None,
    };
    if func.ty(a) != Type::I8 {
        return None;
    }
    if (x, y) == (a, b) {
        Some(low_first.0)
    } else if (x, y) == (b, a) {
        Some(low_first.1)
    } else {
        None
    }
}

/// An unsigned compare as an equality against the minimum or maximum, which byte vectors compute directly.
fn unsigned_cmp(e: &mut Emit<'_>, cc: IntCC, x: Value, y: Value) -> Option<Value> {
    let (pick, equal) = match cc {
        IntCC::UnsignedLessThanOrEqual => (Binary::Umin, true),
        IntCC::UnsignedGreaterThanOrEqual => (Binary::Umax, true),
        IntCC::UnsignedLessThan => (Binary::Umax, false),
        IntCC::UnsignedGreaterThan => (Binary::Umin, false),
        _ => return None,
    };
    let bound = e.op(Op::Binary(pick, x, y), VEC);
    Some(e.op(Op::Icmp(if equal { IntCC::Equal } else { IntCC::NotEqual }, bound, x), VEC))
}

struct Loopy<'a> {
    params: &'a [Value],
    back_args: &'a [Value],
    steps: &'a [Option<i64>],
    defs: &'a [bool],
    iv: usize,
    limit: Value,
}

fn lane_body(e: &mut Emit<'_>, shape: &Shape, (q, splats): (&[Value], &[(Value, Value)])) -> Vec<(Value, Value)> {
    let mut map: Vec<(Value, Value)> = splats.to_vec();
    let get = |map: &[(Value, Value)], v: Value| map.iter().rev().find(|&&(s, _)| s == v).map(|&(_, d)| d).expect("lane operand mapped");
    let defs_of = |func: &Function, v: Value| shape.lanes.iter().map(|&i| &func.insts[i.index()]).find(|d| d.result == Some(v)).map(|d| d.op.clone());
    for &inst in &shape.lanes {
        let op = e.func.insts[inst.index()].op.clone();
        let result = e.func.insts[inst.index()].result;
        let made = match op {
            Op::Load(..) => {
                let at = address(e, shape, inst, q);
                e.op(Op::Load(at.0, at.1), VEC)
            }
            Op::Store(value, ..) => {
                let at = address(e, shape, inst, q);
                let stored = get(&map, value);
                let store = e.func.add(Op::Store(stored, at.0, at.1), None);
                e.out.push(store);
                continue;
            }
            Op::Binary(op @ (Binary::Ushr | Binary::Ishl), x, y) => {
                let by = shape.amounts.iter().find(|&&(v, _)| v == y).map(|&(_, by)| by).expect("shift amount recorded");
                let vx = get(&map, x);
                byte_shift(e, op, vx, by)
            }
            Op::Binary(op, x, y) => {
                let (vx, vy) = (get(&map, x), get(&map, y));
                e.op(Op::Binary(op, vx, vy), VEC)
            }
            Op::Unary(Unary::Ineg, x) => {
                let vx = get(&map, x);
                let zero = e.op(Op::Const(0), VEC);
                e.op(Op::Binary(Binary::Isub, zero, vx), VEC)
            }
            Op::Icmp(cc, x, y) => {
                let (vx, vy) = (get(&map, x), get(&map, y));
                let data = shape.kinds[x.index()] != Some(Kind::Mask);
                match unsigned_cmp(e, cc, vx, vy).filter(|_| data) {
                    Some(v) => v,
                    None => e.op(Op::Icmp(cc, vx, vy), VEC),
                }
            }
            Op::Select(c, x, y) => {
                let def = defs_of(e.func, c);
                let (vx, vy) = (get(&map, x), get(&map, y));
                match min_max(e.func, def.as_ref(), x, y) {
                    Some(pick) => e.op(Op::Binary(pick, vx, vy), VEC),
                    None => {
                        let vc = get(&map, c);
                        e.op(Op::Select(vc, vx, vy), VEC)
                    }
                }
            }
            _ => unreachable!("shape admits only lane operations"),
        };
        if let Some(result) = result {
            map.push((result, made));
        }
    }
    map
}

/// Adds each summed byte vector into its sum; with `psadbw` (`accs` given) into a vector accumulator per sum, returning the next accumulators, else into the scalar sum.
fn add_sums(e: &mut Emit<'_>, shape: &Shape, (q, map): (&[Value], &[(Value, Value)]), next: &mut [Value], accs: Option<&[Value]>) -> Vec<Value> {
    let mut out = Vec::new();
    for (i, sum) in shape.sums.iter().enumerate() {
        let bytes = map.iter().rev().find(|&&(s, _)| s == sum.lane).map(|&(_, d)| d).expect("summed lane mapped");
        match accs {
            Some(accs) => {
                let halves = sad_halves(e, bytes);
                out.push(e.op(Op::Binary(Binary::Iadd, accs[i], halves), QUADS));
            }
            None => {
                let total = byte_sum(e, bytes);
                next[sum.idx] = add_resized(e, q[sum.idx], total);
            }
        }
    }
    out
}

/// Adds `(cnt * iv) + weighted_lane_offsets` for each masked induction-variable sum, accumulating weighted lane offsets in `accs` when `psadbw` is used.
fn add_iv_sums(
    e: &mut Emit<'_>,
    shape: &Shape,
    (q, map, weights): (&[Value], &[(Value, Value)], Value),
    next: &mut [Value],
    accs: Option<&[Value]>,
) -> Vec<Value> {
    let mut out = Vec::new();
    for (i, iv_sum) in shape.iv_sums.iter().enumerate() {
        let raw_cond = map.iter().rev().find(|&&(s, _)| s == iv_sum.cond).map(|&(_, d)| d).expect("invariant: iv sum cond mapped");
        let mut mask = if shape.kinds[iv_sum.cond.index()] == Some(Kind::Data) {
            let zero = e.op(Op::Const(0), VEC);
            e.op(Op::Icmp(IntCC::NotEqual, raw_cond, zero), VEC)
        } else {
            raw_cond
        };
        if iv_sum.invert {
            mask = e.op(Op::Unary(Unary::Bnot, mask), VEC);
        }
        let ones = e.op(Op::Unary(Unary::Ineg, mask), VEC);
        let w_vec = e.op(Op::Binary(Binary::Band, mask, weights), VEC);
        let iv_val = if iv_sum.bias == 0 {
            q[iv_sum.iv_idx]
        } else {
            let bias = e.konst(iv_sum.bias, Type::I64);
            e.op(Op::Binary(Binary::Iadd, q[iv_sum.iv_idx], bias), Type::I64)
        };
        match accs {
            Some(accs) => {
                let cnt_halves = sad_halves(e, ones);
                let cnt = e.op(Op::Reduce(ReduceOp::Add, cnt_halves), Type::I64);
                let term = e.op(Op::Binary(Binary::Imul, cnt, iv_val), Type::I64);
                next[iv_sum.idx] = add_resized(e, q[iv_sum.idx], term);
                let w_halves = sad_halves(e, w_vec);
                out.push(e.op(Op::Binary(Binary::Iadd, accs[i], w_halves), QUADS));
            }
            None => {
                let cnt8 = e.op(Op::Reduce(ReduceOp::Add, ones), Type::I8);
                let cnt = e.op(Op::Convert(Convert::Uextend, cnt8), Type::I64);
                let w8 = e.op(Op::Reduce(ReduceOp::Add, w_vec), Type::I8);
                let w = e.op(Op::Convert(Convert::Uextend, w8), Type::I64);
                let mul = e.op(Op::Binary(Binary::Imul, cnt, iv_val), Type::I64);
                let term = e.op(Op::Binary(Binary::Iadd, mul, w), Type::I64);
                next[iv_sum.idx] = add_resized(e, q[iv_sum.idx], term);
            }
        }
    }
    out
}

fn address(e: &mut Emit<'_>, shape: &Shape, inst: Inst, at: &[Value]) -> (Value, i32) {
    let access = shape.access.iter().find(|&&(i, _)| i == inst).map(|&(_, a)| a).expect("lane access recorded");
    (base_plus(e, access, at), access.off)
}

fn base_plus(e: &mut Emit<'_>, access: Access, at: &[Value]) -> Value {
    match access.base {
        Some(base) => e.op(Op::Binary(Binary::Iadd, base, at[access.idx]), Type::I64),
        None => at[access.idx],
    }
}

fn advance_stepped(e: &mut Emit<'_>, q: &[Value], strides: &[Option<Value>], next: &mut [Value]) {
    for ((&v, s), slot) in q.iter().zip(strides).zip(next) {
        if let Some(by) = *s {
            let ty = e.func.ty(v);
            *slot = e.op(Op::Binary(Binary::Iadd, v, by), ty);
        }
    }
}

/// Whether `at` through `at + LANES - 1` all stay below `bound`.
fn fits(e: &mut Emit<'_>, at: Value, bound: Value) -> Value {
    let ty = e.func.ty(at);
    let left = e.op(Op::Binary(Binary::Isub, bound, at), ty);
    let lanes = e.konst(LANES, ty);
    let wide = e.op(Op::Icmp(IntCC::UnsignedGreaterThanOrEqual, left, lanes), Type::I8);
    let below = e.op(Op::Icmp(IntCC::UnsignedLessThan, at, bound), Type::I8);
    e.op(Op::Binary(Binary::Band, below, wide), Type::I8)
}

/// The trip count and every bounds check of the loop, as `(index, bound)` pairs without repeats.
fn limits(shape: &Shape, lp: &Loopy<'_>) -> Vec<(usize, Value)> {
    std::iter::once((lp.iv, lp.limit)).chain(shape.bounds.iter().copied().filter(|&pair| pair != (lp.iv, lp.limit))).collect()
}

/// Whether one vector iteration from `at` stays inside the trip count and every bounds check.
fn room(e: &mut Emit<'_>, shape: &Shape, lp: &Loopy<'_>, at: &[Value]) -> Value {
    let mut ok = None;
    for (idx, bound) in limits(shape, lp) {
        let inside = fits(e, at[idx], bound);
        ok = Some(ok.map_or(inside, |acc| e.op(Op::Binary(Binary::Band, acc, inside), Type::I8)));
    }
    ok.expect("the trip count is always a limit")
}

/// The number of whole vector iterations from `at` that every limit admits, once `room` holds there.
fn trips(e: &mut Emit<'_>, shape: &Shape, lp: &Loopy<'_>, at: &[Value]) -> Value {
    let mut n = None;
    for (idx, bound) in limits(shape, lp) {
        let ty = e.func.ty(at[idx]);
        let left = e.op(Op::Binary(Binary::Isub, bound, at[idx]), ty);
        let left = if ty == Type::I64 { left } else { e.op(Op::Convert(Convert::Uextend, left), Type::I64) };
        let shift = e.konst(LANES.trailing_zeros().into(), Type::I64);
        let whole = e.op(Op::Binary(Binary::Ushr, left, shift), Type::I64);
        n = Some(n.map_or(whole, |acc| e.op(Op::Binary(Binary::Umin, acc, whole), Type::I64)));
    }
    n.expect("the trip count is always a limit")
}

fn guard(e: &mut Emit<'_>, shape: &Shape, lp: &Loopy<'_>, p: &[Value]) -> Value {
    let mut ok = room(e, shape, lp, p);
    let Some(store) = shape.store else { return ok };
    for &load in &shape.loads {
        if static_safe(load, store).is_some() {
            continue;
        }
        let to = base_plus(e, store, p);
        let from = base_plus(e, load, p);
        let gap = e.op(Op::Binary(Binary::Isub, to, from), Type::I64);
        let bias = e.konst(i64::from(store.off) - i64::from(load.off) - 1, Type::I64);
        let dist = e.op(Op::Binary(Binary::Iadd, gap, bias), Type::I64);
        let span = e.konst(LANES - 1, Type::I64);
        let safe = e.op(Op::Icmp(IntCC::UnsignedGreaterThanOrEqual, dist, span), Type::I8);
        ok = e.op(Op::Binary(Binary::Band, ok, safe), Type::I8);
    }
    ok
}

fn exit_values(e: &mut Emit<'_>, lp: &Loopy<'_>, r: &[Value], exit_args: &[Value]) -> Vec<Value> {
    let mut out = Vec::with_capacity(exit_args.len());
    for &v in exit_args {
        let made = if let Some(i) = lp.params.iter().position(|&p| p == v) {
            match lp.steps[i] {
                Some(0) | None => r[i],
                Some(step) => {
                    let ty = e.func.ty(v);
                    let by = e.konst(step, ty);
                    e.op(Op::Binary(Binary::Isub, r[i], by), ty)
                }
            }
        } else if let Some(i) = lp.back_args.iter().position(|&n| n == v).filter(|&i| lp.steps[i] != Some(0)) {
            r[i]
        } else {
            v
        };
        out.push(made);
    }
    out
}

fn exit_ok(lp: &Loopy<'_>, exit_args: &[Value]) -> bool {
    let param_ok = |v: Value| lp.params.iter().position(|&p| p == v).is_some_and(|i| lp.steps[i].is_some());
    exit_args.iter().all(|&v| !lp.defs[v.index()] || param_ok(v) || lp.back_args.iter().zip(lp.steps).any(|(&n, &s)| n == v && s != Some(0)))
}

/// Runs the one-block byte loop `b` sixteen lanes at a time while sixteen or more iterations remain and no store can feed a later load within one vector, then finishes in `b`.
pub(super) fn run(
    func: &mut Function,
    b: usize,
    exit_test: (Value, bool),
    back_args: &[Value],
    (exit_blk, exit_args): (Block, &[Value]),
    (global_consts, sad): (&[Option<i64>], bool),
) -> bool {
    let insts = &func.blocks[b].insts;
    let preds = outside_preds(func, b);
    if insts.len() > MAX_BODY + 1 || preds.len() != 1 {
        return false;
    }
    let body = insts[..insts.len() - 1].to_vec();
    let params = func.blocks[b].params.clone();
    if back_args.len() != params.len() {
        return false;
    }
    let defs = head_defs(func, b, &body);
    let Some(count) = counted(func, b, &body, exit_test, back_args, &defs, global_consts) else { return false };
    if count.cmp_cc != IntCC::UnsignedLessThan || count.bias != 0 || global_consts[count.stride.index()] != Some(1) {
        return false;
    }
    let steps = steps(func, &body, &params, back_args, global_consts);
    if steps[count.iv_idx] != Some(1) {
        return false;
    }
    let Some(shape) = analyze(func, &body, (&params, back_args), (&steps, &defs), global_consts) else { return false };
    if shape.store.is_some_and(|store| shape.loads.iter().any(|&l| static_safe(l, store) == Some(false))) {
        return false;
    }
    let Some(wanted) = outside_uses(func, b, &defs, (&params, back_args), &[]) else { return false };
    let lp = Loopy { params: &params, back_args, steps: &steps, defs: &defs, iv: count.iv_idx, limit: count.limit };
    if wanted.iter().any(|&w| w) || !exit_ok(&lp, exit_args) {
        return false;
    }
    let head = Block(b as u32);
    let p = fresh(func, &params);
    let q = fresh(func, &params);
    let r = fresh(func, &params);
    let mut e = Emit { func, out: Vec::new() };
    let ok = guard(&mut e, &shape, &lp, &p);
    let n = trips(&mut e, &shape, &lp, &p);
    let mut splats = Vec::new();
    for &v in &shape.splats {
        let source = match e.func.insts.iter().find(|d| d.result == Some(v)).map(|d| d.op.clone()) {
            Some(Op::Const(imm)) if defs[v.index()] => e.konst(imm, Type::I8),
            _ => params.iter().position(|&x| x == v).map_or(v, |i| p[i]),
        };
        let wide = splat(&mut e, source);
        splats.push((v, wide));
    }
    let strides: Vec<Option<Value>> = params
        .iter()
        .zip(&steps)
        .map(|(&v, &s)| s.filter(|&s| s != 0).map(|s| { let ty = e.func.ty(v); e.konst(s * LANES, ty) }))
        .collect();
    let weights = (!shape.iv_sums.is_empty()).then(|| byte_weights(&mut e));
    let quad_count = if sad { shape.sums.len() + shape.iv_sums.len() } else { 0 };
    let zeros: Vec<Value> = (0..quad_count).map(|_| e.op(Op::Const(0), QUADS)).collect();
    let pre_insts = std::mem::take(&mut e.out);
    let left = e.func.value(Type::I64);
    let accs: Vec<Value> = zeros.iter().map(|_| e.func.value(QUADS)).collect();
    let ends: Vec<Value> = zeros.iter().map(|_| e.func.value(QUADS)).collect();
    let map = lane_body(&mut e, &shape, (&q, &splats));
    let mut next = q.clone();
    let mut accs_next = add_sums(&mut e, &shape, (&q, &map), &mut next, sad.then_some(&accs[..]));
    if let Some(weights) = weights {
        let iv_accs = sad.then(|| &accs[shape.sums.len()..]);
        accs_next.extend(add_iv_sums(&mut e, &shape, (&q, &map, weights), &mut next, iv_accs));
    }
    advance_stepped(&mut e, &q, &strides, &mut next);
    let one = e.konst(1, Type::I64);
    let rest = e.op(Op::Binary(Binary::Isub, left, one), Type::I64);
    let zero = e.konst(0, Type::I64);
    let more = e.op(Op::Icmp(IntCC::NotEqual, rest, zero), Type::I8);
    let body_insts = std::mem::take(&mut e.out);
    let mut r_sum = r.clone();
    for (sum, &end) in shape.sums.iter().zip(&ends) {
        let total = e.op(Op::Reduce(ReduceOp::Add, end), Type::I64);
        r_sum[sum.idx] = add_resized(&mut e, r_sum[sum.idx], total);
    }
    if sad {
        for (iv_sum, &end) in shape.iv_sums.iter().zip(&ends[shape.sums.len()..]) {
            let total = e.op(Op::Reduce(ReduceOp::Add, end), Type::I64);
            r_sum[iv_sum.idx] = add_resized(&mut e, r_sum[iv_sum.idx], total);
        }
    }
    let done = e.op(Op::Icmp(IntCC::Equal, r_sum[lp.iv], lp.limit), Type::I8);
    let leave = exit_values(&mut e, &lp, &r_sum, exit_args);
    let tail_insts = std::mem::take(&mut e.out);
    let Emit { func, .. } = e;
    let pre = new_block(func, p.clone(), pre_insts);
    let vbody = new_block(func, q.into_iter().chain([left]).chain(accs).collect(), body_insts);
    let tail = new_block(func, r.iter().copied().chain(ends).collect(), tail_insts);
    let enter = func.add(Op::Brif(ok, vbody, p.iter().copied().chain([n]).chain(zeros).collect(), head, p), None);
    func.blocks[pre.index()].insts.push(enter);
    let back = func.add(Op::Brif(more, vbody, next.iter().copied().chain([rest]).chain(accs_next.iter().copied()).collect(), tail, next.iter().copied().chain(accs_next).collect()), None);
    func.blocks[vbody.index()].insts.push(back);
    let finish = func.add(Op::Brif(done, exit_blk, leave, head, r_sum), None);
    func.blocks[tail.index()].insts.push(finish);
    let from = preds[0];
    if let Some(&last) = func.blocks[from].insts.last() {
        func.insts[last.index()].op.map_targets(|t| if t == head { pre } else { t });
    }
    true
}

fn lane_blocks(func: &Function, each: &Loop) -> bool {
    let mut stores = 0;
    let ok = func.blocks.iter().enumerate().filter(|&(i, _)| each.body.get(i).copied().unwrap_or(false)).all(|(_, block)| {
        block.insts.iter().all(|&inst| {
            let data = &func.insts[inst.index()];
            match data.op {
                Op::Store(value, ..) => {
                    stores += 1;
                    func.ty(value) == Type::I8
                }
                Op::Load(..) => data.result.is_some_and(|r| func.ty(r) == Type::I8),
                Op::Const(_) | Op::Binary(..) | Op::Unary(..) | Op::Convert(..) | Op::Icmp(..) | Op::Select(..) | Op::Brif(..) | Op::Jump(..) | Op::Trapz(_) | Op::Assume(_) => true,
                _ => false,
            }
        })
    });
    ok && stores <= 1
}

/// Vectorizes the byte loop `each` after turning its inner branches into selects, keeping that conversion only when the loop then vectorizes.
pub(super) fn flatten(func: &mut Function, each: &Loop, (global_consts, sad): (&[Option<i64>], bool)) -> bool {
    if each.size > MAX_BLOCKS || !lane_blocks(func, each) {
        return false;
    }
    let mut trial = func.clone();
    if !crate::ifconv::flatten(&mut trial, &each.body) {
        return false;
    }
    let head = each.head;
    let Some(Op::Brif(cond, t_blk, t_args, f_blk, f_args)) = trial.last(head).cloned() else { return false };
    let mut consts = global_consts.to_vec();
    consts.resize(trial.types.len(), None);
    let did = if t_blk == head && f_blk != head {
        run(&mut trial, head.index(), (cond, true), &t_args, (f_blk, &f_args), (&consts, sad))
    } else if f_blk == head && t_blk != head {
        run(&mut trial, head.index(), (cond, false), &f_args, (t_blk, &t_args), (&consts, sad))
    } else {
        false
    };
    if did {
        *func = trial;
    }
    did
}

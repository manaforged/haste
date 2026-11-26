use crate::fold::extend;
use crate::ir::{Binary, Block, BlockData, Convert, Function, Inst, IntCC, Op, Type, Value};

use super::bitmask::try_emit_bitmask_runs;
use super::{Arm, Edge, Info, Run};

pub(super) type Cache = Vec<((Block, Vec<Value>), Block)>;

pub(super) fn forward_cached(func: &mut Function, cache: &mut Cache, target: Block, args: Vec<Value>) -> Block {
    let target = if args.is_empty() { func.jump_target(target) } else { target };
    if args.is_empty() && func.blocks[target.index()].params.is_empty() {
        return target;
    }
    if let Some(&(_, blk)) = cache.iter().find(|((t, a), _)| *t == target && *a == args) {
        return blk;
    }
    let block = Block(func.blocks.len() as u32);
    let inst = func.add(Op::Jump(target, args.clone()), None);
    func.blocks.push(BlockData { params: Vec::new(), insts: vec![inst] });
    cache.push(((target, args), block));
    block
}

pub(super) fn push_node(func: &mut Function, at: Option<(usize, Inst)>, insts: Vec<Inst>) -> Block {
    match at {
        Some((b, last)) => {
            func.insts[last.index()].op = Op::Trap;
            let dst = &mut func.blocks[b].insts;
            dst.pop();
            dst.extend(insts);
            Block(b as u32)
        }
        None => {
            let b = Block(func.blocks.len() as u32);
            func.blocks.push(BlockData { params: Vec::new(), insts });
            b
        }
    }
}

const GUARD_MIN: u64 = 0x80;
const GUARD_RUNS: usize = 4;

fn widen_sub(func: &mut Function, insts: &mut Vec<Inst>, (v, ty): (Value, Type), (to, signed): (Type, bool), base: i64) -> Value {
    let wide = if ty == to {
        v
    } else {
        let conv = if signed { Convert::Sextend } else { Convert::Uextend };
        let res = func.value(to);
        insts.push(func.add(Op::Convert(conv, v), Some(res)));
        res
    };
    if base == 0 {
        return wide;
    }
    let (base_c, sub) = (func.value(to), func.value(to));
    insts.push(func.add(Op::Const(extend(base, to, true)), Some(base_c)));
    insts.push(func.add(Op::Binary(Binary::Isub, wide, base_c), Some(sub)));
    sub
}

fn finish_br_table(func: &mut Function, at: Option<(usize, Inst)>, (v, ty): (Value, Type), (min, signed): (i64, bool), (default_blk, table): (Block, Vec<Block>)) -> Block {
    let mut insts = Vec::with_capacity(4);
    let idx = if min == 0 && !signed { v } else { widen_sub(func, &mut insts, (v, ty), (Type::I64, signed), min) };
    insts.push(func.add(Op::BrTable(idx, default_blk, table), None));
    push_node(func, at, insts)
}

fn dense_enough(runs_len: usize, covered: u64, table_len: u64) -> bool {
    if runs_len < 3 || covered < 4 || table_len > 256 {
        return false;
    }
    if runs_len >= 5 { covered * 3 >= table_len } else { (runs_len as u64) * 5 >= table_len * 2 }
}

fn signed_table_bounds(arms: &[Arm]) -> Option<(i64, usize)> {
    if arms.iter().any(|a| !a.range.signed_ok || a.range.s_hi.abs_diff(a.range.s_lo) >= 256) {
        return None;
    }
    let (mut s_min, mut s_max, mut u_min, mut u_max, mut wrap) = (i64::MAX, i64::MIN, u64::MAX, 0_u64, false);
    for a in arms {
        (s_min, s_max) = (s_min.min(a.range.s_lo), s_max.max(a.range.s_hi));
        if a.range.lo > a.range.hi { wrap = true; } else { (u_min, u_max) = (u_min.min(a.range.lo), u_max.max(a.range.hi)); }
    }
    let s_span = s_max.abs_diff(s_min);
    usize::try_from(s_span).ok().and_then(|s| s.checked_add(1)).filter(|&l| l <= 256 && (wrap || s_span < u_max - u_min)).map(|l| (s_min, l))
}

pub(super) fn can_signed_br_table(arms: &[Arm]) -> bool {
    signed_table_bounds(arms).is_some()
}

pub(super) fn try_signed_br_table(
    func: &mut Function,
    at: (usize, Inst),
    (v, ty): (Value, Type),
    arms: &[Arm],
    (default_blk, cache): (Block, &mut Cache),
) -> bool {
    let Some((s_min, s_len)) = signed_table_bounds(arms) else { return false };
    let (mut table, mut filled, mut covered) = (vec![default_blk; s_len], vec![false; s_len], 0_u64);
    for a in arms {
        let target = forward_cached(func, cache, a.hit.block, a.hit.args.clone());
        let (start, end) = ((a.range.s_lo as i128 - s_min as i128) as usize, (a.range.s_hi as i128 - s_min as i128) as usize);
        for idx in start..=end {
            if !std::mem::replace(&mut filled[idx], true) {
                table[idx] = target;
                covered += 1;
            }
        }
    }
    let (mut s_runs, mut prev) = (0_usize, None);
    for (&blk, &is_hit) in table.iter().zip(&filled) {
        if is_hit && prev != Some(blk) { s_runs += 1; prev = Some(blk); } else if !is_hit { prev = None; }
    }
    if !dense_enough(s_runs, covered, s_len as u64) {
        return false;
    }
    finish_br_table(func, Some(at), (v, ty), (s_min, true), (default_blk, table));
    true
}

pub(super) fn can_br_table(runs: &[Run]) -> Option<(u64, usize)> {
    let (min_lo, max_hi) = (runs.first()?.lo, runs.last()?.hi);
    let table_len = usize::try_from(max_hi - min_lo).ok().and_then(|s| s.checked_add(1))?;
    let covered: u64 = runs.iter().map(|r| r.hi - r.lo + 1).sum();
    dense_enough(runs.len(), covered, table_len as u64).then_some((min_lo, table_len))
}

fn find_cluster(runs: &[Run]) -> Option<(usize, usize)> {
    let mut best: Option<(u64, usize, usize)> = None;
    for i in 0..runs.len() {
        let (min_lo, mut covered) = (runs[i].lo, 0_u64);
        for j in (i + 1)..=runs.len() {
            let last = &runs[j - 1];
            covered += last.hi - last.lo + 1;
            let span = last.hi - min_lo;
            if span >= 256 {
                break;
            }
            if dense_enough(j - i, covered, span + 1) && best.is_none_or(|(bc, bj, bi)| (covered, j - i) > (bc, bj - bi)) {
                best = Some((covered, j, i));
            }
        }
    }
    best.map(|(_, j, i)| (i, j))
}

pub(super) fn emit_sub_range(func: &mut Function, at: Option<(usize, Inst)>, (v, ty): (Value, Type), (lo, span): (i64, u64), (hit, otherwise): (&Edge, Block)) -> Block {
    let lo_u = extend(lo, ty, false) as u64;
    let to = if ty.bits() < 32 && lo_u.checked_add(span).is_some_and(|top| top <= ty.unsigned_max()) { Type::I32 } else { ty };
    let mut insts = Vec::with_capacity(6);
    let diff = widen_sub(func, &mut insts, (v, ty), (to, false), lo_u as i64);
    let (span_c, le) = (func.value(to), func.value(Type::I8));
    insts.push(func.add(Op::Const(extend(span as i64, to, true)), Some(span_c)));
    insts.push(func.add(Op::Icmp(IntCC::UnsignedLessThanOrEqual, diff, span_c), Some(le)));
    insts.push(func.add(Op::Brif(le, hit.block, hit.args.clone(), otherwise, Vec::new()), None));
    push_node(func, at, insts)
}

fn emit_run_check(func: &mut Function, at: Option<(usize, Inst)>, (v, ty, (lo, hi)): (Value, Type, (u64, u64)), run: &Run, (otherwise, cache): (Block, &mut Cache)) -> Block {
    if run.lo <= lo && run.hi >= hi {
        if at.is_none() {
            return forward_cached(func, cache, run.hit.block, run.hit.args.clone());
        }
        let jump = func.add(Op::Jump(run.hit.block, run.hit.args.clone()), None);
        return push_node(func, at, vec![jump]);
    }
    let (floor, ceil) = (lo, hi);
    if run.lo > floor && run.hi < ceil && run.lo != run.hi {
        return emit_sub_range(func, at, (v, ty), (run.lo as i64, run.hi - run.lo), (&run.hit, otherwise));
    }
    let (cc, imm) = match (run.lo == run.hi, run.lo <= floor) {
        (true, _) => (IntCC::Equal, run.lo),
        (false, true) => (IntCC::UnsignedLessThanOrEqual, run.hi),
        (false, false) => (IntCC::UnsignedGreaterThanOrEqual, run.lo),
    };
    let (c, cond) = (func.value(ty), func.value(Type::I8));
    let insts = vec![
        func.add(Op::Const(extend(imm as i64, ty, true)), Some(c)),
        func.add(Op::Icmp(cc, v, c), Some(cond)),
        func.add(Op::Brif(cond, run.hit.block, run.hit.args.clone(), otherwise, Vec::new()), None),
    ];
    push_node(func, at, insts)
}

pub(super) fn emit_node(func: &mut Function, info: &Info, at: Option<(usize, Inst)>, (v, ty): (Value, Type), runs: &[Run], (default_blk, cache): (Block, &mut Cache)) -> Block {
    emit_within(func, info, at, (v, ty, (0, ty.unsigned_max())), runs, (default_blk, cache))
}

pub(super) fn emit_within(func: &mut Function, info: &Info, at: Option<(usize, Inst)>, (v, ty, (lo, hi)): (Value, Type, (u64, u64)), runs: &[Run], (default_blk, cache): (Block, &mut Cache)) -> Block {
    if let Some(blk) = try_emit_bitmask_runs(func, info, at, (v, ty, (lo, hi)), runs, (default_blk, cache)) {
        return blk;
    }
    if let Some((min, table_len)) = can_br_table(runs) {
        let mut table = vec![default_blk; table_len];
        for r in runs {
            let target = forward_cached(func, cache, r.hit.block, r.hit.args.clone());
            table[(r.lo - min) as usize..=(r.hi - min) as usize].fill(target);
        }
        return finish_br_table(func, at, (v, ty), (min as i64, false), (default_blk, table));
    }
    if lo == 0 && runs.len() >= GUARD_RUNS && runs[0].lo >= GUARD_MIN {
        let min = runs[0].lo;
        let rest = emit_tree(func, info, None, (v, ty, (min, hi)), runs, (default_blk, cache));
        return emit_split(func, at, (v, ty), min, (default_blk, rest));
    }
    emit_tree(func, info, at, (v, ty, (lo, hi)), runs, (default_blk, cache))
}

fn emit_tree(func: &mut Function, info: &Info, at: Option<(usize, Inst)>, (v, ty, (lo, hi)): (Value, Type, (u64, u64)), runs: &[Run], (default_blk, cache): (Block, &mut Cache)) -> Block {
    if runs.len() <= 2 {
        let rest = if runs[0].lo <= lo { runs[0].hi.saturating_add(1) } else { lo };
        let otherwise = if runs.len() == 2 { emit_within(func, info, None, (v, ty, (rest, hi)), &runs[1..], (default_blk, cache)) } else { default_blk };
        return emit_run_check(func, at, (v, ty, (lo, hi)), &runs[0], (otherwise, cache));
    }
    let mid = match find_cluster(runs) {
        Some((start, _)) if start > 0 => start,
        Some((_, end)) => end,
        None => runs.len() / 2,
    };
    let split = runs[mid].lo;
    let left = emit_within(func, info, None, (v, ty, (lo, split - 1)), &runs[..mid], (default_blk, cache));
    let right = emit_within(func, info, None, (v, ty, (split, hi)), &runs[mid..], (default_blk, cache));
    emit_split(func, at, (v, ty), split, (left, right))
}

pub(super) fn emit_split(func: &mut Function, at: Option<(usize, Inst)>, (v, ty): (Value, Type), split: u64, (left, right): (Block, Block)) -> Block {
    let (pivot, lt) = (func.value(ty), func.value(Type::I8));
    let insts = vec![
        func.add(Op::Const(extend(split as i64, ty, true)), Some(pivot)),
        func.add(Op::Icmp(IntCC::UnsignedLessThan, v, pivot), Some(lt)),
        func.add(Op::Brif(lt, left, Vec::new(), right, Vec::new()), None),
    ];
    push_node(func, at, insts)
}

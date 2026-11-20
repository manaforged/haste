use crate::ir::{Binary, Block, Convert, Function, Inst, IntCC, Op, Type, Value};

use super::emit::{Cache, can_br_table, emit_node, push_node};
use super::{Arm, Edge, Info, Range, Run, intersect_ranges, match_test, range_fits, range_of_test, same_edge};

#[derive(Clone, Copy)]
pub(super) struct BitmaskPlan {
    pub(super) base: i64,
    pub(super) span: u64,
    pub(super) words: [u64; 2],
    pub(super) two_words: bool,
    pub(super) signed: bool,
}

#[derive(Clone, Copy)]
pub(super) struct TableLayout {
    pub(super) base: i64,
    pub(super) span: u64,
    pub(super) words: [u64; 2],
    pub(super) two_words: bool,
    pub(super) signed: bool,
    pub(super) per_word: u64,
    pub(super) bounds_elide_max: u64,
}

pub(super) struct ExprCond {
    pub(super) val: Value,
    pub(super) want_or: bool,
    pub(super) ranges: Vec<Range>,
    pub(super) used: Vec<Inst>,
}

pub(super) fn known_upper_bound(func: &Function, info: &Info, v: Value) -> u64 {
    let (known_zero, _) = crate::fold::known_bits(
        func,
        &|u| info.def_inst.get(u.index()).copied().flatten(),
        &|u| info.consts.get(u.index()).copied().flatten(),
        v,
        0,
    );
    func.ty(v).unsigned_max() & !known_zero
}

fn insert_interval(runs: &mut Vec<Run>, mut cur: u64, hi: u64, hit: Edge) {
    let (mut added, mut done) = (Vec::new(), false);
    for r in runs.iter() {
        if r.hi < cur {
            continue;
        }
        if r.lo > hi {
            break;
        }
        if cur < r.lo {
            added.push(Run { lo: cur, hi: r.lo - 1, hit: hit.clone() });
        }
        if r.hi >= hi {
            done = true;
            break;
        }
        cur = r.hi + 1;
    }
    if !done && cur <= hi {
        added.push(Run { lo: cur, hi, hit });
    }
    if !added.is_empty() {
        runs.extend(added);
        runs.sort_unstable_by_key(|r| r.lo);
    }
}

pub(super) fn build_runs(func: &Function, info: &Info, ty: Type, arms: &[Arm]) -> Vec<Run> {
    let (mut runs, u_max) = (Vec::new(), ty.unsigned_max());
    for arm in arms {
        if arm.range.lo <= arm.range.hi {
            insert_interval(&mut runs, arm.range.lo, arm.range.hi, arm.hit.clone());
        } else {
            insert_interval(&mut runs, arm.range.lo, u_max, arm.hit.clone());
            insert_interval(&mut runs, 0, arm.range.hi, arm.hit.clone());
        }
    }
    let mut merged: Vec<Run> = Vec::with_capacity(runs.len());
    for r in runs {
        if let Some(prev) = merged.last_mut()
            && prev.hi < u64::MAX
            && prev.hi + 1 == r.lo
            && same_edge(func, info, &prev.hit, &r.hit)
        {
            prev.hi = r.hi;
        } else {
            merged.push(r);
        }
    }
    merged
}

fn set_mask_range(words: &mut [u64; 2], start: u64, end: u64) {
    for bit in start..=end {
        words[(bit >> 6) as usize] |= 1_u64 << (bit & 63);
    }
}

fn plan_unsigned(runs: &[Run], max_bits: u64) -> Option<BitmaskPlan> {
    let (min_lo, max_hi) = (runs.first()?.lo, runs.last()?.hi);
    if max_hi - min_lo >= max_bits {
        return None;
    }
    let two_words = max_bits > 64;
    let base = if max_hi < 64 || (two_words && max_hi < 128 && max_hi - min_lo >= 64) { 0 } else { min_lo };
    let span = max_hi - base;
    let mut words = [0_u64; 2];
    for r in runs {
        set_mask_range(&mut words, r.lo - base, r.hi - base);
    }
    Some(BitmaskPlan { base: base as i64, span, words, two_words: span >= 64, signed: false })
}

fn plan_signed(ty: Type, runs: &[Run], max_bits: u64) -> Option<BitmaskPlan> {
    let sign_bit = 1_u64 << (ty.bits() - 1);
    let mut segs: Vec<(i64, i64)> = Vec::with_capacity(runs.len() + 1);
    for r in runs {
        if r.lo < sign_bit && r.hi >= sign_bit {
            segs.push((crate::fold::extend(r.lo as i64, ty, true), crate::fold::extend((sign_bit - 1) as i64, ty, true)));
            segs.push((crate::fold::extend(sign_bit as i64, ty, true), crate::fold::extend(r.hi as i64, ty, true)));
        } else {
            segs.push((crate::fold::extend(r.lo as i64, ty, true), crate::fold::extend(r.hi as i64, ty, true)));
        }
    }
    let s_min = segs.iter().map(|&(lo, _)| lo).min()?;
    let s_max = segs.iter().map(|&(_, hi)| hi).max()?;
    let span = s_max.abs_diff(s_min);
    if span >= max_bits {
        return None;
    }
    let mut words = [0_u64; 2];
    for (lo, hi) in segs {
        set_mask_range(&mut words, lo.abs_diff(s_min), hi.abs_diff(s_min));
    }
    Some(BitmaskPlan { base: s_min, span, words, two_words: span >= 64, signed: true })
}

pub(super) fn build_bitmask_plan_with(ty: Type, runs: &[Run], min_runs_two_words: usize) -> Option<BitmaskPlan> {
    if runs.len() < 3 {
        return None;
    }
    if let Some(plan) = plan_unsigned(runs, 64).or_else(|| plan_signed(ty, runs, 64)) {
        return Some(plan);
    }
    if runs.len() >= min_runs_two_words {
        return plan_unsigned(runs, 128).or_else(|| plan_signed(ty, runs, 128));
    }
    None
}

fn build_bitmask_plan(ty: Type, runs: &[Run]) -> Option<BitmaskPlan> {
    build_bitmask_plan_with(ty, runs, 4)
}

fn best_bitmask_window(ty: Type, subset: &[Run]) -> Option<(BitmaskPlan, usize, usize)> {
    if let Some(plan) = build_bitmask_plan(ty, subset) {
        return Some((plan, 0, subset.len()));
    }
    if subset.len() < 4 {
        return None;
    }
    let mut best: Option<(BitmaskPlan, usize, usize)> = None;
    for i in 0..subset.len() {
        for j in (i + 3..=subset.len()).rev() {
            if best.as_ref().is_some_and(|&(_, bi, bj)| j - i <= bj - bi) {
                break;
            }
            if let Some(plan) = build_bitmask_plan(ty, &subset[i..j]) {
                best = Some((plan, i, j));
                break;
            }
        }
    }
    best
}

fn find_bitmask_target(func: &Function, info: &Info, ty: Type, runs: &[Run]) -> Option<(BitmaskPlan, Edge, Vec<Run>)> {
    if runs.len() < 3 {
        return None;
    }
    let mut groups: Vec<(Edge, Vec<Run>)> = Vec::new();
    for r in runs {
        if let Some((_, list)) = groups.iter_mut().find(|(edge, _)| same_edge(func, info, edge, &r.hit)) {
            list.push(r.clone());
        } else {
            groups.push((r.hit.clone(), vec![r.clone()]));
        }
    }
    groups.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
    let can_tbl = groups.len() > 1 && can_br_table(runs).is_some();
    for (hit, subset) in groups {
        if subset.len() < 3 {
            break;
        }
        let Some((plan, wi, wj)) = best_bitmask_window(ty, &subset) else { continue };
        let (w_lo, w_hi) = (subset[wi].lo, subset[wj - 1].hi);
        let rest: Vec<Run> = runs
            .iter()
            .filter(|r| !(same_edge(func, info, &hit, &r.hit) && r.lo >= w_lo && r.hi <= w_hi))
            .cloned()
            .collect();
        if can_tbl && rest.len() > 2 {
            continue;
        }
        return Some((plan, hit, rest));
    }
    None
}

pub(super) fn can_bitmask_runs(func: &Function, info: &Info, ty: Type, runs: &[Run]) -> bool {
    find_bitmask_target(func, info, ty, runs).is_some()
}

pub(super) fn emit_table_word(
    func: &mut Function,
    (v, ty): (Value, Type),
    layout: TableLayout,
    known_u_max: u64,
    need_bounds: bool,
    insts: &mut Vec<Inst>,
) -> (Value, Option<Value>, Value) {
    let v64 = if ty == Type::I64 {
        v
    } else {
        let conv = if layout.signed { Convert::Sextend } else { Convert::Uextend };
        let res = func.value(Type::I64);
        insts.push(func.add(Op::Convert(conv, v), Some(res)));
        res
    };
    let diff = if layout.base == 0 {
        v64
    } else {
        let (base_c, sub) = (func.value(Type::I64), func.value(Type::I64));
        insts.push(func.add(Op::Const(layout.base), Some(base_c)));
        insts.push(func.add(Op::Binary(Binary::Isub, v64, base_c), Some(sub)));
        sub
    };
    let bounds_known = !layout.signed && layout.base == 0 && known_u_max <= layout.bounds_elide_max;
    let in_range = if need_bounds && !bounds_known {
        let (span_c, cond) = (func.value(Type::I64), func.value(Type::I8));
        insts.push(func.add(Op::Const(layout.span as i64), Some(span_c)));
        insts.push(func.add(Op::Icmp(IntCC::UnsignedLessThanOrEqual, diff, span_c), Some(cond)));
        Some(cond)
    } else {
        None
    };
    let word = if layout.two_words {
        let (split_c, in_lo) = (func.value(Type::I64), func.value(Type::I8));
        insts.push(func.add(Op::Const(layout.per_word as i64), Some(split_c)));
        insts.push(func.add(Op::Icmp(IntCC::UnsignedLessThan, diff, split_c), Some(in_lo)));
        let (lo_c, hi_c, chosen) = (func.value(Type::I64), func.value(Type::I64), func.value(Type::I64));
        insts.push(func.add(Op::Const(layout.words[0] as i64), Some(lo_c)));
        insts.push(func.add(Op::Const(layout.words[1] as i64), Some(hi_c)));
        insts.push(func.add(Op::Select(in_lo, lo_c, hi_c), Some(chosen)));
        chosen
    } else {
        let mask_c = func.value(Type::I64);
        insts.push(func.add(Op::Const(layout.words[0] as i64), Some(mask_c)));
        mask_c
    };
    (diff, in_range, word)
}

fn bitmask_layout(plan: BitmaskPlan) -> TableLayout {
    TableLayout {
        base: plan.base,
        span: plan.span,
        words: plan.words,
        two_words: plan.two_words,
        signed: plan.signed,
        per_word: 64,
        bounds_elide_max: if plan.two_words { 127 } else { 63 },
    }
}

fn emit_bit(func: &mut Function, (word, diff): (Value, Value), insts: &mut Vec<Inst>) -> Value {
    let shifted = func.value(Type::I64);
    insts.push(func.add(Op::Binary(Binary::Ushr, word, diff), Some(shifted)));
    let (shifted8, one8, bit8) = (func.value(Type::I8), func.value(Type::I8), func.value(Type::I8));
    insts.push(func.add(Op::Convert(Convert::Ireduce, shifted), Some(shifted8)));
    insts.push(func.add(Op::Const(1), Some(one8)));
    insts.push(func.add(Op::Binary(Binary::Band, shifted8, one8), Some(bit8)));
    bit8
}

pub(super) fn emit_bitmask_test(
    func: &mut Function,
    (v, ty): (Value, Type),
    plan: BitmaskPlan,
    known_u_max: u64,
    need_bounds: bool,
    insts: &mut Vec<Inst>,
) -> Value {
    let (diff, in_range, word) = emit_table_word(func, (v, ty), bitmask_layout(plan), known_u_max, need_bounds, insts);
    let bit8 = emit_bit(func, (word, diff), insts);
    let Some(in_range) = in_range else { return bit8 };
    let cond = func.value(Type::I8);
    insts.push(func.add(Op::Binary(Binary::Band, in_range, bit8), Some(cond)));
    cond
}

fn emit_bitmask_branch(
    func: &mut Function,
    (v, ty): (Value, Type),
    (plan, known_u_max): (BitmaskPlan, u64),
    (hit, miss): ((Block, Vec<Value>), (Block, Vec<Value>)),
) -> Vec<Inst> {
    let mut insts = Vec::with_capacity(14);
    let (diff, in_range, word) = emit_table_word(func, (v, ty), bitmask_layout(plan), known_u_max, true, &mut insts);
    let Some(in_range) = in_range else {
        let bit8 = emit_bit(func, (word, diff), &mut insts);
        insts.push(func.add(Op::Brif(bit8, hit.0, hit.1, miss.0, miss.1), None));
        return insts;
    };
    let guard_end = insts.iter().position(|&inst| func.insts[inst.index()].result == Some(in_range)).map_or(insts.len(), |at| at + 1);
    let mut test = insts.split_off(guard_end);
    let bit8 = emit_bit(func, (word, diff), &mut test);
    test.push(func.add(Op::Brif(bit8, hit.0, hit.1, miss.0, miss.1.clone()), None));
    let test_blk = push_node(func, None, test);
    insts.push(func.add(Op::Brif(in_range, test_blk, Vec::new(), miss.0, miss.1), None));
    insts
}

pub(super) fn try_emit_bitmask_runs(
    func: &mut Function,
    info: &Info,
    at: Option<(usize, Inst)>,
    (v, ty): (Value, Type),
    runs: &[Run],
    (default_blk, cache): (Block, &mut Cache),
) -> Option<Block> {
    let (plan, hit, rest) = find_bitmask_target(func, info, ty, runs)?;
    let otherwise = if rest.is_empty() {
        default_blk
    } else {
        emit_node(func, info, None, (v, ty), &rest, (default_blk, cache))
    };
    let known_u_max = known_upper_bound(func, info, v);
    let insts = emit_bitmask_branch(func, (v, ty), (plan, known_u_max), ((hit.block, hit.args), (otherwise, Vec::new())));
    Some(push_node(func, at, insts))
}

fn collect_expr_leaf(
    func: &Function,
    info: &Info,
    b0: usize,
    node: Value,
    want_true: bool,
    (val, ranges, dead): (&mut Option<Value>, &mut Vec<Range>, &mut Vec<Inst>),
) -> bool {
    let Some(test) = match_test(func, info, node, *val, Some(b0)) else { return false };
    let ty = func.ty(test.val);
    let Some(r) = range_of_test(ty, test.kind, want_true).filter(|&r| range_fits(r)) else { return false };
    *val = Some(test.val);
    ranges.push(r);
    for inst in test.used.into_iter().flatten() {
        if !dead.contains(&inst) {
            dead.push(inst);
        }
    }
    true
}

fn collect_expr_tree(
    func: &Function,
    info: &Info,
    b0: usize,
    (node, want_or, depth): (Value, bool, u8),
    (val, ranges, dead): (&mut Option<Value>, &mut Vec<Range>, &mut Vec<Inst>),
) -> bool {
    if depth > 16 || info.def_block.get(node.index()).copied().flatten() != Some(b0) || info.uses.get(node.index()).copied() != Some(1) {
        return false;
    }
    let Some(inst) = info.def_inst.get(node.index()).copied().flatten() else { return false };
    match func.insts[inst.index()].op {
        Op::Binary(bin, a, b) if (want_or && bin == Binary::Bor) || (!want_or && bin == Binary::Band) => {
            dead.push(inst);
            collect_expr_tree(func, info, b0, (a, want_or, depth + 1), (val, ranges, dead))
                && collect_expr_tree(func, info, b0, (b, want_or, depth + 1), (val, ranges, dead))
        }
        Op::Binary(Binary::Band, a, b) if want_or => {
            let mut sub_ranges = Vec::with_capacity(2);
            if !collect_expr_leaf(func, info, b0, a, true, (val, &mut sub_ranges, dead))
                || !collect_expr_leaf(func, info, b0, b, true, (val, &mut sub_ranges, dead))
            {
                return false;
            }
            let Some(v) = *val else { return false };
            let ty = func.ty(v);
            let Some(merged) = intersect_ranges(ty, sub_ranges[0], sub_ranges[1]).filter(|&r| range_fits(r)) else {
                return false;
            };
            dead.push(inst);
            ranges.push(merged);
            true
        }
        _ => collect_expr_leaf(func, info, b0, node, want_or, (val, ranges, dead)),
    }
}

pub(super) fn match_expr_cond(func: &Function, info: &Info, b0: usize, cond: Value, expected_val: Option<Value>, allow_single_band: bool) -> Option<ExprCond> {
    let inst = info.def_inst.get(cond.index()).copied().flatten()?;
    if !matches!(func.insts[inst.index()].op, Op::Binary(Binary::Band | Binary::Bor, ..)) {
        return None;
    }
    for want_or in [true, false] {
        let (mut val, mut ranges, mut dead) = (expected_val, Vec::new(), Vec::new());
        if !collect_expr_tree(func, info, b0, (cond, want_or, 0), (&mut val, &mut ranges, &mut dead)) {
            continue;
        }
        let Some(v) = val else { continue };
        let min_len = if allow_single_band && want_or { 1 } else { 2 };
        if ranges.len() < min_len {
            continue;
        }
        return Some(ExprCond { val: v, want_or, ranges, used: dead });
    }
    None
}

pub(super) fn try_rewrite_expr(func: &mut Function, info: &Info, (b0, last0): (usize, Inst), cleared: &mut [bool]) -> bool {
    let Op::Brif(cond, t_blk, t_args, f_blk, f_args) = func.insts[last0.index()].op.clone() else {
        return false;
    };
    for want_or in [true, false] {
        let (mut val, mut ranges, mut dead) = (None, Vec::new(), Vec::new());
        if !collect_expr_tree(func, info, b0, (cond, want_or, 0), (&mut val, &mut ranges, &mut dead)) || ranges.len() < 3 {
            continue;
        }
        let Some(v) = val else { continue };
        let ty = func.ty(v);
        let dummy = Edge { raw: t_blk, block: t_blk, args: Vec::new(), synth_bool: None };
        let arms: Vec<Arm> = ranges.iter().copied().map(|range| Arm { range, hit: dummy.clone() }).collect();
        let runs = build_runs(func, info, ty, &arms);
        let min_runs_two_words = if ranges.len() >= 4 || dead.len() >= 5 { 3 } else { 4 };
        let Some(plan) = build_bitmask_plan_with(ty, &runs, min_runs_two_words) else { continue };
        func.blocks[b0].insts.retain(|i| !dead.contains(i));
        for inst in dead {
            func.insts[inst.index()].op = Op::Trap;
        }
        let known_u_max = known_upper_bound(func, info, v);
        let edges = if want_or { ((t_blk, t_args), (f_blk, f_args)) } else { ((f_blk, f_args), (t_blk, t_args)) };
        let insts = emit_bitmask_branch(func, (v, ty), (plan, known_u_max), edges);
        push_node(func, Some((b0, last0)), insts);
        cleared[b0] = true;
        return true;
    }
    false
}

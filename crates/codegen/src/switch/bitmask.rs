mod emit;
mod expr;

pub(super) use self::emit::{emit_bitmask_test, emit_table_word, try_emit_bitmask_runs};
pub(super) use self::expr::{match_expr_cond, try_rewrite_expr};
use super::emit::can_br_table;
use super::{Arm, Edge, Info, Range, Run, same_edge};
use crate::ir::{Function, Inst, Type, Value};

const ONE_WORD_BITS: u64 = 64;
const TWO_WORD_BITS: u64 = 128;
const ONE_WORD_MAX_SPAN: u64 = ONE_WORD_BITS - 1;
const TWO_WORD_MAX_SPAN: u64 = TWO_WORD_BITS - 1;

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
            added.push(Run {
                lo: cur,
                hi: r.lo - 1,
                hit: hit.clone(),
            });
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
    let base = if max_hi < 64 || (two_words && max_hi < 128 && max_hi - min_lo >= 64) {
        0
    } else {
        min_lo
    };
    let span = max_hi - base;
    let mut words = [0_u64; 2];
    for r in runs {
        set_mask_range(&mut words, r.lo - base, r.hi - base);
    }
    Some(BitmaskPlan {
        base: base as i64,
        span,
        words,
        two_words: span >= 64,
        signed: false,
    })
}

fn plan_signed(ty: Type, runs: &[Run], max_bits: u64) -> Option<BitmaskPlan> {
    let sign_bit = 1_u64 << (ty.bits() - 1);
    let mut segs: Vec<(i64, i64)> = Vec::with_capacity(runs.len() + 1);
    for r in runs {
        if r.lo < sign_bit && r.hi >= sign_bit {
            segs.push((
                crate::fold::extend(r.lo as i64, ty, true),
                crate::fold::extend((sign_bit - 1) as i64, ty, true),
            ));
            segs.push((
                crate::fold::extend(sign_bit as i64, ty, true),
                crate::fold::extend(r.hi as i64, ty, true),
            ));
        } else {
            segs.push((
                crate::fold::extend(r.lo as i64, ty, true),
                crate::fold::extend(r.hi as i64, ty, true),
            ));
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
    Some(BitmaskPlan {
        base: s_min,
        span,
        words,
        two_words: span >= 64,
        signed: true,
    })
}

fn plan_width(ty: Type, runs: &[Run], two_words: bool) -> Option<BitmaskPlan> {
    let max_bits = if two_words {
        TWO_WORD_BITS
    } else {
        ONE_WORD_BITS
    };
    plan_unsigned(runs, max_bits).or_else(|| plan_signed(ty, runs, max_bits))
}

fn build_bitmask_plan_with(
    ty: Type,
    runs: &[Run],
    min_runs_two_words: usize,
) -> Option<BitmaskPlan> {
    if runs.len() < 3 {
        return None;
    }
    plan_width(ty, runs, false).or_else(|| {
        (runs.len() >= min_runs_two_words)
            .then(|| plan_width(ty, runs, true))
            .flatten()
    })
}

fn plan_rank(
    plan: BitmaskPlan,
    wi: usize,
    wj: usize,
) -> ((usize, bool), (bool, std::cmp::Reverse<u64>)) {
    (
        (wj - wi, !plan.two_words),
        (plan.base == 0, std::cmp::Reverse(plan.span)),
    )
}

fn partition_rest(rest: &[Run], base_u: u64, top_u: u64) -> (Vec<Run>, Vec<Run>) {
    let (mut in_runs, mut out_runs) = (Vec::new(), Vec::new());
    for r in rest {
        if r.lo < base_u {
            out_runs.push(Run {
                lo: r.lo,
                hi: r.hi.min(base_u - 1),
                hit: r.hit.clone(),
            });
        }
        let lo_in = r.lo.max(base_u);
        let hi_in = r.hi.min(top_u);
        if lo_in <= hi_in {
            in_runs.push(Run {
                lo: lo_in,
                hi: hi_in,
                hit: r.hit.clone(),
            });
        }
        if r.hi > top_u {
            out_runs.push(Run {
                lo: r.lo.max(top_u + 1),
                hi: r.hi,
                hit: r.hit.clone(),
            });
        }
    }
    (in_runs, out_runs)
}

fn target_rank(
    plan: BitmaskPlan,
    wi: usize,
    wj: usize,
    rest: &[Run],
) -> (
    (usize, bool),
    (std::cmp::Reverse<usize>, bool),
    (bool, std::cmp::Reverse<u64>),
) {
    let (max_miss, zero_hit) = if !plan.signed && plan.base >= 0 {
        let base_u = plan.base as u64;
        let (in_runs, out_runs) = partition_rest(rest, base_u, base_u + plan.span);
        let zero = in_runs
            .first()
            .is_some_and(|r| r.lo == r.hi && r.lo == base_u);
        (in_runs.len().max(out_runs.len()), zero)
    } else {
        (rest.len(), false)
    };
    let (head, tail) = plan_rank(plan, wi, wj);
    (head, (std::cmp::Reverse(max_miss), zero_hit), tail)
}

fn best_window_for(
    ty: Type,
    subset: &[Run],
    two_words: bool,
    min_len: usize,
) -> Option<(BitmaskPlan, usize, usize)> {
    if subset.len() < min_len {
        return None;
    }
    let mut best: Option<(BitmaskPlan, usize, usize)> = None;
    for i in 0..subset.len() {
        if i + min_len > subset.len() {
            break;
        }
        for j in (i + min_len..=subset.len()).rev() {
            let best_len = best.map_or(0, |(_, bi, bj)| bj - bi);
            if j - i < best_len {
                break;
            }
            if let Some(plan) = plan_width(ty, &subset[i..j], two_words) {
                if best.is_none_or(|(bp, bi, bj)| plan_rank(plan, i, j) > plan_rank(bp, bi, bj)) {
                    best = Some((plan, i, j));
                }
                break;
            }
        }
    }
    best
}

fn find_bitmask_target(
    func: &Function,
    info: &Info,
    ty: Type,
    runs: &[Run],
) -> Option<(BitmaskPlan, Edge, Vec<Run>, u64, u64)> {
    if runs.len() < 3 {
        return None;
    }
    let mut groups: Vec<(Edge, Vec<Run>)> = Vec::new();
    for r in runs {
        if let Some((_, list)) = groups
            .iter_mut()
            .find(|(edge, _)| same_edge(func, info, edge, &r.hit))
        {
            list.push(r.clone());
        } else {
            groups.push((r.hit.clone(), vec![r.clone()]));
        }
    }
    groups.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
    let can_tbl = groups.len() > 1 && can_br_table(runs).is_some();
    let mut best: Option<(BitmaskPlan, Edge, Vec<Run>, usize, usize, u64, u64)> = None;
    for (hit, subset) in groups {
        if subset.len() < 3
            || best
                .as_ref()
                .is_some_and(|(_, _, _, wi, wj, _, _)| subset.len() < wj - wi)
        {
            break;
        }
        let mut consider = |plan: BitmaskPlan, wi: usize, wj: usize| {
            let (w_lo, w_hi) = (subset[wi].lo, subset[wj - 1].hi);
            let rest: Vec<Run> = runs
                .iter()
                .filter(|r| !(same_edge(func, info, &hit, &r.hit) && r.lo >= w_lo && r.hi <= w_hi))
                .cloned()
                .collect();
            if can_tbl && rest.len() > 2 {
                return;
            }
            if best.as_ref().is_none_or(|(bp, _, brest, bi, bj, _, _)| {
                target_rank(plan, wi, wj, &rest) > target_rank(*bp, *bi, *bj, brest)
            }) {
                best = Some((plan, hit.clone(), rest, wi, wj, w_lo, w_hi));
            }
        };
        let mut best_1w_len = 0_usize;
        for i in 0..subset.len() {
            if i + 3 > subset.len() {
                break;
            }
            for j in (i + 3..=subset.len()).rev() {
                if j - i < best_1w_len {
                    break;
                }
                if let Some(plan) = plan_width(ty, &subset[i..j], false) {
                    best_1w_len = best_1w_len.max(j - i);
                    consider(plan, i, j);
                    if !plan.signed && plan.base > 0 {
                        let w_hi = subset[j - 1].hi;
                        for q in runs {
                            if q.lo == q.hi
                                && q.lo < plan.base as u64
                                && w_hi - q.lo <= ONE_WORD_MAX_SPAN
                            {
                                let shift = plan.base as u64 - q.lo;
                                let shifted = BitmaskPlan {
                                    base: q.lo as i64,
                                    span: w_hi - q.lo,
                                    words: [plan.words[0] << shift, 0],
                                    two_words: false,
                                    signed: false,
                                };
                                consider(shifted, i, j);
                            }
                        }
                    }
                    break;
                }
            }
        }
        let min_2w = if best_1w_len == 0 { 4 } else { best_1w_len + 3 };
        if let Some((plan, wi, wj)) = best_window_for(ty, &subset, true, min_2w) {
            consider(plan, wi, wj);
        }
    }
    best.map(|(plan, hit, rest, _, _, w_lo, w_hi)| (plan, hit, rest, w_lo, w_hi))
}

pub(super) fn can_bitmask_runs(func: &Function, info: &Info, ty: Type, runs: &[Run]) -> bool {
    find_bitmask_target(func, info, ty, runs).is_some()
}

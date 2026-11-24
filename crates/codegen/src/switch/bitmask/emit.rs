use super::super::emit::{Cache, emit_split, emit_within, push_node};
use super::super::{Edge, Info, Run};
use super::{
    BitmaskPlan, ONE_WORD_BITS, ONE_WORD_MAX_SPAN, TWO_WORD_MAX_SPAN, TableLayout,
    find_bitmask_target, known_upper_bound, partition_rest,
};
use crate::ir::{Binary, Block, Convert, Function, Inst, IntCC, Op, Type, Value};

const MASKED_QUAD_BASE_MAX: u64 = 4096;
const MASKED_QUAD_BIT_MAX: u64 = 1 << 31;

fn bounds_elidable(layout: TableLayout, lo: u64, known_u_max: u64) -> bool {
    !layout.signed
        && layout.base >= 0
        && lo >= layout.base as u64
        && known_u_max.saturating_sub(layout.base as u64) <= layout.bounds_elide_max
}

fn emit_entry_const(func: &mut Function, imm: i64, fallback: &mut Vec<Inst>) -> Value {
    if let Some(val) = func.entry_const(imm, Type::I64) {
        return val;
    }
    let val = func.value(Type::I64);
    fallback.push(func.add(Op::Const(imm), Some(val)));
    val
}

fn emit_word_select(
    func: &mut Function,
    diff: Value,
    layout: TableLayout,
    insts: &mut Vec<Inst>,
) -> Value {
    if layout.two_words && layout.words[0] != layout.words[1] {
        let (split_c, in_lo) = (func.value(Type::I64), func.value(Type::I8));
        insts.push(func.add(Op::Const(layout.per_word as i64), Some(split_c)));
        insts.push(func.add(
            Op::Icmp(IntCC::UnsignedLessThan, diff, split_c),
            Some(in_lo),
        ));
        let lo_c = emit_entry_const(func, layout.words[0] as i64, insts);
        let hi_c = emit_entry_const(func, layout.words[1] as i64, insts);
        let chosen = func.value(Type::I64);
        insts.push(func.add(Op::Select(in_lo, lo_c, hi_c), Some(chosen)));
        chosen
    } else {
        emit_entry_const(func, layout.words[0] as i64, insts)
    }
}

fn emit_v64_diff(
    func: &mut Function,
    (v, ty): (Value, Type),
    (signed, base): (bool, i64),
    insts: &mut Vec<Inst>,
) -> Value {
    let v64 = if ty == Type::I64 {
        v
    } else {
        let conv = if signed {
            Convert::Sextend
        } else {
            Convert::Uextend
        };
        let res = func.value(Type::I64);
        insts.push(func.add(Op::Convert(conv, v), Some(res)));
        res
    };
    if base == 0 {
        v64
    } else {
        let (base_c, sub) = (func.value(Type::I64), func.value(Type::I64));
        insts.push(func.add(Op::Const(base), Some(base_c)));
        insts.push(func.add(Op::Binary(Binary::Isub, v64, base_c), Some(sub)));
        sub
    }
}

fn emit_table_word_in(
    func: &mut Function,
    (v, ty): (Value, Type),
    layout: TableLayout,
    (lo, known_u_max): (u64, u64),
    need_bounds: bool,
    insts: &mut Vec<Inst>,
) -> (Value, Option<Value>, Value) {
    let diff = emit_v64_diff(func, (v, ty), (layout.signed, layout.base), insts);
    let in_range = if need_bounds && !bounds_elidable(layout, lo, known_u_max) {
        let (span_c, cond) = (func.value(Type::I64), func.value(Type::I8));
        insts.push(func.add(Op::Const(layout.span as i64), Some(span_c)));
        insts.push(func.add(
            Op::Icmp(IntCC::UnsignedLessThanOrEqual, diff, span_c),
            Some(cond),
        ));
        Some(cond)
    } else {
        None
    };
    let word = emit_word_select(func, diff, layout, insts);
    (diff, in_range, word)
}

pub(in crate::switch) fn emit_table_word(
    func: &mut Function,
    (v, ty): (Value, Type),
    layout: TableLayout,
    known_u_max: u64,
    need_bounds: bool,
    insts: &mut Vec<Inst>,
) -> (Value, Option<Value>, Value) {
    emit_table_word_in(func, (v, ty), layout, (0, known_u_max), need_bounds, insts)
}

fn bitmask_layout(plan: BitmaskPlan) -> TableLayout {
    TableLayout {
        base: plan.base,
        span: plan.span,
        words: plan.words,
        two_words: plan.two_words,
        signed: plan.signed,
        per_word: ONE_WORD_BITS,
        bounds_elide_max: if plan.two_words {
            TWO_WORD_MAX_SPAN
        } else {
            ONE_WORD_MAX_SPAN
        },
    }
}

fn emit_bit(func: &mut Function, (word, diff): (Value, Value), insts: &mut Vec<Inst>) -> Value {
    let shifted = func.value(Type::I64);
    insts.push(func.add(Op::Binary(Binary::Ushr, word, diff), Some(shifted)));
    let (shifted8, one8, bit8) = (
        func.value(Type::I8),
        func.value(Type::I8),
        func.value(Type::I8),
    );
    insts.push(func.add(Op::Convert(Convert::Ireduce, shifted), Some(shifted8)));
    insts.push(func.add(Op::Const(1), Some(one8)));
    insts.push(func.add(Op::Binary(Binary::Band, shifted8, one8), Some(bit8)));
    bit8
}

fn emit_bit_branch_insts(
    func: &mut Function,
    (word, diff): (Value, Value),
    (hit, miss): ((Block, Vec<Value>), (Block, Vec<Value>)),
    insts: &mut Vec<Inst>,
) {
    let shifted = func.value(Type::I64);
    insts.push(func.add(Op::Binary(Binary::Ushr, word, diff), Some(shifted)));
    let (one, masked, zero, cond) = (
        func.value(Type::I64),
        func.value(Type::I64),
        func.value(Type::I64),
        func.value(Type::I8),
    );
    insts.push(func.add(Op::Const(1), Some(one)));
    insts.push(func.add(Op::Binary(Binary::Band, shifted, one), Some(masked)));
    insts.push(func.add(Op::Const(0), Some(zero)));
    insts.push(func.add(Op::Icmp(IntCC::NotEqual, masked, zero), Some(cond)));
    insts.push(func.add(Op::Brif(cond, hit.0, hit.1, miss.0, miss.1), None));
}

pub(in crate::switch) fn emit_bitmask_test(
    func: &mut Function,
    (v, ty): (Value, Type),
    plan: BitmaskPlan,
    known_u_max: u64,
    need_bounds: bool,
    insts: &mut Vec<Inst>,
) -> Value {
    let (diff, in_range, word) = emit_table_word(
        func,
        (v, ty),
        bitmask_layout(plan),
        known_u_max,
        need_bounds,
        insts,
    );
    let bit8 = emit_bit(func, (word, diff), insts);
    let Some(in_range) = in_range else {
        return bit8;
    };
    let cond = func.value(Type::I8);
    insts.push(func.add(Op::Binary(Binary::Band, in_range, bit8), Some(cond)));
    cond
}

fn zero_diff_miss(
    func: &mut Function,
    diff: Value,
    zero_arm: Option<(Edge, Block)>,
    fallback: (Block, Vec<Value>),
) -> (Block, Vec<Value>) {
    let Some((z_hit, z_miss)) = zero_arm else {
        return fallback;
    };
    let (zero_c, is_zero) = (func.value(Type::I64), func.value(Type::I8));
    let insts = vec![
        func.add(Op::Const(0), Some(zero_c)),
        func.add(Op::Icmp(IntCC::Equal, diff, zero_c), Some(is_zero)),
        func.add(
            Op::Brif(is_zero, z_hit.block, z_hit.args, z_miss, Vec::new()),
            None,
        ),
    ];
    (push_node(func, None, insts), Vec::new())
}

pub(super) fn emit_bitmask_branch(
    func: &mut Function,
    (v, ty): (Value, Type),
    (plan, lo, known_u_max): (BitmaskPlan, u64, u64),
    (hit, in_miss, out_miss): (
        (Block, Vec<Value>),
        (Block, Vec<Value>),
        (Block, Vec<Value>),
    ),
    zero_arm: Option<(Edge, Block)>,
) -> Vec<Inst> {
    let layout = bitmask_layout(plan);
    let mut insts = Vec::with_capacity(14);
    let (diff, in_range, word) = if !layout.signed
        && layout.base == 0
        && ty != Type::I64
        && layout.span <= ty.unsigned_max()
        && !bounds_elidable(layout, lo, known_u_max)
    {
        let (span_c, in_range) = (func.value(ty), func.value(Type::I8));
        insts.push(func.add(
            Op::Const(crate::fold::extend(layout.span as i64, ty, true)),
            Some(span_c),
        ));
        insts.push(func.add(
            Op::Icmp(IntCC::UnsignedLessThanOrEqual, v, span_c),
            Some(in_range),
        ));
        let diff = emit_v64_diff(func, (v, ty), (false, 0), &mut insts);
        let word = emit_word_select(func, diff, layout, &mut insts);
        (diff, Some(in_range), word)
    } else {
        emit_table_word_in(func, (v, ty), layout, (lo, known_u_max), true, &mut insts)
    };
    let mut test = if let Some(in_range) = in_range {
        let guard_end = insts
            .iter()
            .position(|&inst| func.insts[inst.index()].result == Some(in_range))
            .map_or(insts.len(), |at| at + 1);
        insts.split_off(guard_end)
    } else {
        Vec::new()
    };
    let body = if in_range.is_some() {
        &mut test
    } else {
        &mut insts
    };
    let miss = zero_diff_miss(func, diff, zero_arm, in_miss);
    emit_bit_branch_insts(func, (word, diff), (hit, miss), body);
    if let Some(in_range) = in_range {
        let test_blk = push_node(func, None, test);
        insts.push(func.add(
            Op::Brif(in_range, test_blk, Vec::new(), out_miss.0, out_miss.1),
            None,
        ));
    }
    insts
}

fn masked_quad_params(ty: Type, runs: &[Run]) -> Option<(u64, u64, u64)> {
    if !ty.is_small_int() || runs.len() != 4 || runs.iter().any(|r| r.lo != r.hi) {
        return None;
    }
    let [a, b, c, d] = [runs[0].lo, runs[1].lo, runs[2].lo, runs[3].lo];
    if a >= MASKED_QUAD_BASE_MAX || d > ty.unsigned_max() {
        return None;
    }
    for (hi_bit, delta, mid_hi) in [(a ^ c, b - a, b), (a ^ b, c - a, c)] {
        let other_hi = if mid_hi == b { c } else { b };
        if hi_bit.is_power_of_two()
            && delta.is_power_of_two()
            && hi_bit < MASKED_QUAD_BIT_MAX
            && delta < MASKED_QUAD_BIT_MAX
            && (a & hi_bit) == 0
            && (mid_hi & hi_bit) == 0
            && other_hi == (a | hi_bit)
            && d == (mid_hi | hi_bit)
        {
            return Some((a, hi_bit, delta));
        }
    }
    None
}

fn emit_masked_quad(
    func: &mut Function,
    at: Option<(usize, Inst)>,
    (v, ty): (Value, Type),
    (base, hi_bit, delta): (u64, u64, u64),
    (hit, otherwise): (Edge, Block),
) -> Block {
    let to = if ty.bits() < 32 { Type::I32 } else { ty };
    let mut insts = Vec::with_capacity(10);
    let vw = if ty == to {
        v
    } else {
        let res = func.value(to);
        insts.push(func.add(Op::Convert(Convert::Uextend, v), Some(res)));
        res
    };
    let (hi_mask_c, cleared) = (func.value(to), func.value(to));
    insts.push(func.add(
        Op::Const(crate::fold::extend((!hi_bit) as i64, to, true)),
        Some(hi_mask_c),
    ));
    insts.push(func.add(Op::Binary(Binary::Band, vw, hi_mask_c), Some(cleared)));
    let diff = if base == 0 {
        cleared
    } else {
        let (base_c, sub) = (func.value(to), func.value(to));
        insts.push(func.add(
            Op::Const(crate::fold::extend(base as i64, to, true)),
            Some(base_c),
        ));
        insts.push(func.add(Op::Binary(Binary::Isub, cleared, base_c), Some(sub)));
        sub
    };
    let (delta_mask_c, masked, zero_c, cond) = (
        func.value(to),
        func.value(to),
        func.value(to),
        func.value(Type::I8),
    );
    insts.push(func.add(
        Op::Const(crate::fold::extend((!delta) as i64, to, true)),
        Some(delta_mask_c),
    ));
    insts.push(func.add(Op::Binary(Binary::Band, diff, delta_mask_c), Some(masked)));
    insts.push(func.add(Op::Const(0), Some(zero_c)));
    insts.push(func.add(Op::Icmp(IntCC::Equal, masked, zero_c), Some(cond)));
    insts.push(func.add(
        Op::Brif(cond, hit.block, hit.args, otherwise, Vec::new()),
        None,
    ));
    push_node(func, at, insts)
}

fn miss_with_zero(
    func: &mut Function,
    info: &Info,
    (v, ty, (in_lo, in_hi)): (Value, Type, (u64, u64)),
    base_u: u64,
    runs: &[Run],
    (default_blk, cache): (Block, &mut Cache),
) -> (Block, Option<(Edge, Block)>) {
    if runs.is_empty() {
        (default_blk, None)
    } else if runs[0].lo == base_u && runs[0].hi == base_u {
        let after_zero = if runs.len() == 1 {
            default_blk
        } else {
            emit_within(
                func,
                info,
                None,
                (v, ty, (in_lo.max(base_u + 1), in_hi)),
                &runs[1..],
                (default_blk, cache),
            )
        };
        (default_blk, Some((runs[0].hit.clone(), after_zero)))
    } else {
        (
            emit_within(
                func,
                info,
                None,
                (v, ty, (in_lo, in_hi)),
                runs,
                (default_blk, cache),
            ),
            None,
        )
    }
}

pub(in crate::switch) fn try_emit_bitmask_runs(
    func: &mut Function,
    info: &Info,
    at: Option<(usize, Inst)>,
    (v, ty, (lo, hi)): (Value, Type, (u64, u64)),
    runs: &[Run],
    (default_blk, cache): (Block, &mut Cache),
) -> Option<Block> {
    let (plan, hit, rest, _, _) = find_bitmask_target(func, info, ty, runs)?;
    if rest.is_empty()
        && let Some((base, hi_bit, delta)) = masked_quad_params(ty, runs)
    {
        return Some(emit_masked_quad(
            func,
            at,
            (v, ty),
            (base, hi_bit, delta),
            (hit, default_blk),
        ));
    }
    let known_u_max = known_upper_bound(func, info, v).min(hi);
    let (in_miss_blk, out_miss_blk, zero_arm) = if rest.is_empty() {
        (default_blk, default_blk, None)
    } else if !plan.signed && plan.base >= 0 {
        let base_u = plan.base as u64;
        let top_u = base_u.saturating_add(plan.span);
        let (in_runs, out_runs) = partition_rest(&rest, base_u, top_u);
        if in_runs.is_empty() && out_runs.len() > 2 {
            let split = if out_runs.last().is_some_and(|r| r.hi < base_u) {
                Some(base_u)
            } else if out_runs.first().is_some_and(|r| r.lo > top_u) {
                Some(out_runs[0].lo)
            } else {
                None
            };
            if let Some(split) = split
                && let Some(split_idx) = runs.iter().position(|r| r.lo >= split)
                && split_idx > 0
                && split_idx < runs.len()
            {
                let left = emit_within(
                    func,
                    info,
                    None,
                    (v, ty, (lo, split - 1)),
                    &runs[..split_idx],
                    (default_blk, cache),
                );
                let right = emit_within(
                    func,
                    info,
                    None,
                    (v, ty, (split, hi)),
                    &runs[split_idx..],
                    (default_blk, cache),
                );
                return Some(emit_split(func, at, (v, ty), split, (left, right)));
            }
        }
        if bounds_elidable(bitmask_layout(plan), lo, known_u_max) {
            let (in_miss_blk, zero_arm) = miss_with_zero(
                func,
                info,
                (v, ty, (lo, hi)),
                base_u,
                &rest,
                (default_blk, cache),
            );
            (in_miss_blk, default_blk, zero_arm)
        } else {
            let out_lo = if lo >= base_u {
                top_u.saturating_add(1).max(lo)
            } else {
                lo
            };
            let out_hi = if hi <= top_u {
                base_u.saturating_sub(1).min(hi)
            } else {
                hi
            };
            let out_miss_blk = if out_runs.is_empty() {
                default_blk
            } else {
                emit_within(
                    func,
                    info,
                    None,
                    (v, ty, (out_lo, out_hi)),
                    &out_runs,
                    (default_blk, cache),
                )
            };
            let (in_miss_blk, zero_arm) = miss_with_zero(
                func,
                info,
                (v, ty, (base_u.max(lo), top_u.min(hi))),
                base_u,
                &in_runs,
                (default_blk, cache),
            );
            (in_miss_blk, out_miss_blk, zero_arm)
        }
    } else {
        let otherwise = emit_within(
            func,
            info,
            None,
            (v, ty, (lo, hi)),
            &rest,
            (default_blk, cache),
        );
        (otherwise, otherwise, None)
    };
    let insts = emit_bitmask_branch(
        func,
        (v, ty),
        (plan, lo, known_u_max),
        (
            (hit.block, hit.args),
            (in_miss_blk, Vec::new()),
            (out_miss_blk, Vec::new()),
        ),
        zero_arm,
    );
    Some(push_node(func, at, insts))
}

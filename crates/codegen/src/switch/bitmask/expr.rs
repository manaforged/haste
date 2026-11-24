use super::super::emit::push_node;
use super::super::{
    Arm, Edge, Info, Range, intersect_ranges, match_test, range_fits, range_of_test,
};
use super::emit::emit_bitmask_branch;
use super::{ExprCond, build_bitmask_plan_with, build_runs, known_upper_bound};
use crate::ir::{Binary, Function, Inst, Op, Value};

fn collect_expr_leaf(
    func: &Function,
    info: &Info,
    b0: usize,
    node: Value,
    want_true: bool,
    (val, ranges, dead): (&mut Option<Value>, &mut Vec<Range>, &mut Vec<Inst>),
) -> bool {
    let Some(test) = match_test(func, info, node, *val, Some(b0)) else {
        return false;
    };
    let ty = func.ty(test.val);
    let Some(r) = range_of_test(ty, test.kind, want_true).filter(|&r| range_fits(r)) else {
        return false;
    };
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
    if depth > 16
        || info.def_block.get(node.index()).copied().flatten() != Some(b0)
        || info.uses.get(node.index()).copied() != Some(1)
    {
        return false;
    }
    let Some(inst) = info.def_inst.get(node.index()).copied().flatten() else {
        return false;
    };
    match func.insts[inst.index()].op {
        Op::Binary(bin, a, b)
            if (want_or && bin == Binary::Bor) || (!want_or && bin == Binary::Band) =>
        {
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
            let Some(v) = *val else {
                return false;
            };
            let ty = func.ty(v);
            let Some(merged) =
                intersect_ranges(ty, sub_ranges[0], sub_ranges[1]).filter(|&r| range_fits(r))
            else {
                return false;
            };
            dead.push(inst);
            ranges.push(merged);
            true
        }
        _ => collect_expr_leaf(func, info, b0, node, want_or, (val, ranges, dead)),
    }
}

pub(in crate::switch) fn match_expr_cond(
    func: &Function,
    info: &Info,
    b0: usize,
    cond: Value,
    expected_val: Option<Value>,
    allow_single_band: bool,
) -> Option<ExprCond> {
    let inst = info.def_inst.get(cond.index()).copied().flatten()?;
    if !matches!(
        func.insts[inst.index()].op,
        Op::Binary(Binary::Band | Binary::Bor, ..)
    ) {
        return None;
    }
    for want_or in [true, false] {
        let (mut val, mut ranges, mut dead) = (expected_val, Vec::new(), Vec::new());
        if !collect_expr_tree(
            func,
            info,
            b0,
            (cond, want_or, 0),
            (&mut val, &mut ranges, &mut dead),
        ) {
            continue;
        }
        let Some(v) = val else {
            continue;
        };
        let min_len = if allow_single_band && want_or { 1 } else { 2 };
        if ranges.len() < min_len {
            continue;
        }
        return Some(ExprCond {
            val: v,
            want_or,
            ranges,
            used: dead,
        });
    }
    None
}

pub(in crate::switch) fn try_rewrite_expr(
    func: &mut Function,
    info: &Info,
    (b0, last0): (usize, Inst),
    cleared: &mut [bool],
) -> bool {
    let Op::Brif(cond, t_blk, t_args, f_blk, f_args) = func.insts[last0.index()].op.clone() else {
        return false;
    };
    for want_or in [true, false] {
        let (mut val, mut ranges, mut dead) = (None, Vec::new(), Vec::new());
        if !collect_expr_tree(
            func,
            info,
            b0,
            (cond, want_or, 0),
            (&mut val, &mut ranges, &mut dead),
        ) || ranges.len() < 3
        {
            continue;
        }
        let Some(v) = val else {
            continue;
        };
        let ty = func.ty(v);
        let dummy = Edge {
            raw: t_blk,
            block: t_blk,
            args: Vec::new(),
            synth_bool: None,
        };
        let arms: Vec<Arm> = ranges
            .iter()
            .copied()
            .map(|range| Arm {
                range,
                hit: dummy.clone(),
            })
            .collect();
        let runs = build_runs(func, info, ty, &arms);
        let min_runs_two_words = if dead.len() >= 8 { 3 } else { 4 };
        let Some(plan) = build_bitmask_plan_with(ty, &runs, min_runs_two_words) else {
            continue;
        };
        func.blocks[b0].insts.retain(|i| !dead.contains(i));
        for inst in dead {
            func.insts[inst.index()].op = Op::Trap;
        }
        let known_u_max = known_upper_bound(func, info, v);
        let edges = if want_or {
            ((t_blk, t_args), (f_blk, f_args))
        } else {
            ((f_blk, f_args), (t_blk, t_args))
        };
        let insts = emit_bitmask_branch(
            func,
            (v, ty),
            (plan, 0, known_u_max),
            (edges.0, edges.1.clone(), edges.1),
            None,
        );
        push_node(func, Some((b0, last0)), insts);
        cleared[b0] = true;
        return true;
    }
    false
}

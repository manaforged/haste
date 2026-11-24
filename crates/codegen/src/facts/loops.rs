mod collect;
mod guards;
mod query;

use crate::cfg::Cfg;
use crate::ir::{Binary, Block, Function, IntCC, Op, Type, Value};

use super::Ranges;
use super::rel::{Path, RelKind};

#[derive(Clone, Copy)]
struct SumInv {
    head: Block,
    p: Value,
    q: Value,
    bound: Value,
    ty: Type,
}

#[derive(Clone, Copy)]
struct OrderInv {
    head: Block,
    lo: Value,
    hi: Value,
    bound: Value,
    ty: Type,
}

#[derive(Clone, Copy)]
struct PtrInv {
    head: Block,
    ptr: Value,
    base: Value,
    end: Value,
    len: Value,
    shift: u32,
    stride: u64,
}

#[derive(Clone, Copy)]
struct SubBound {
    idx: Value,
    end: Value,
    min_diff: u64,
    ty: Type,
}

#[derive(Clone, Copy)]
pub(super) struct RelInv {
    pub(super) head: Block,
    pub(super) param: Value,
    pub(super) bound: Value,
    pub(super) ty: Type,
    pub(super) kind: RelKind,
}

#[derive(Default)]
pub(super) struct Env {
    sums: Vec<SumInv>,
    orders: Vec<OrderInv>,
    ptrs: Vec<PtrInv>,
    pub(super) rel_invs: Vec<RelInv>,
    param_lo: Vec<(Block, Value, u64)>,
}

#[derive(Clone, Default)]
pub(super) struct ExtraPath {
    subs: Vec<SubBound>,
    diffs: Vec<(Value, Value, u64)>,
}

fn op_is_const_val(ranges: &Ranges, path: &Path, v: Value, exact: u64) -> bool {
    path.fact(ranges, v).exact() == Some(exact)
}

fn edge_ule(func: &Function, ranges: &Ranges, path: &Path, ty: Type, a: Value, b: Value) -> bool {
    a == b
        || path.fact(ranges, a).hi <= path.fact(ranges, b).lo
        || path.decide_rel(func, ranges, IntCC::UnsignedLessThanOrEqual, ty, a, b) == Some(true)
}

fn edge_min_lo(ranges: &Ranges, env: &Env, cfg: &Cfg, path: &Path, block: Block, v: Value) -> u64 {
    let mut lo = path.fact(ranges, v).lo;
    for &(head, p, min_lo) in &env.param_lo {
        if p == v && cfg.dominates(head, block) {
            lo = lo.max(min_lo);
        }
    }
    lo
}

fn is_ushr_of(func: &Function, ranges: &Ranges, val: Value, base: Value) -> bool {
    matches!(ranges.op(func, val), Some(&Op::Binary(Binary::Ushr, x, _)) if x == base)
}

fn is_nonzero_ushr_of(func: &Function, ranges: &Ranges, ty: Type, val: Value, base: Value) -> bool {
    matches!(
        ranges.op(func, val),
        Some(&Op::Binary(Binary::Ushr, x, sh))
            if x == base && ranges.constant(sh).is_some_and(|k| (k & u64::from(ty.bits() - 1)) >= 1)
    )
}

fn is_shrunk(
    func: &Function,
    ranges: &Ranges,
    path: &Path,
    ty: Type,
    expr: Value,
    base: Value,
) -> bool {
    if expr == base || is_ushr_of(func, ranges, expr, base) {
        return true;
    }
    match ranges.op(func, expr) {
        Some(&Op::Binary(Binary::Band, a, b)) if a == base || b == base => true,
        Some(&Op::Binary(Binary::Isub, x, d)) if x == base => {
            is_ushr_of(func, ranges, d, base) || edge_ule(func, ranges, path, ty, d, base)
        }
        _ => false,
    }
}

fn add_delta_of_depth(
    func: &Function,
    ranges: &Ranges,
    expr: Value,
    base: Value,
    depth: u8,
) -> Option<(Value, u64)> {
    let &Op::Binary(Binary::Iadd, p, q) = ranges.op(func, expr)? else {
        return None;
    };
    if p == base {
        return Some((q, 0));
    }
    if q == base {
        return Some((p, 0));
    }
    if depth >= 3 {
        return None;
    }
    let max = func.ty(expr).unsigned_max();
    for (inner, k_val) in [(p, q), (q, p)] {
        if let Some(k) = ranges.constant(k_val)
            && let Some((d, c)) = add_delta_of_depth(func, ranges, inner, base, depth + 1)
            && let Some(sum) = c.checked_add(k).filter(|&s| s <= max)
        {
            return Some((d, sum));
        }
    }
    None
}

fn add_delta_of(
    func: &Function,
    ranges: &Ranges,
    expr: Value,
    base: Value,
) -> Option<(Value, u64)> {
    add_delta_of_depth(func, ranges, expr, base, 0)
}

fn sub_delta_of(
    func: &Function,
    ranges: &Ranges,
    expr: Value,
    base: Value,
) -> Option<(Value, u64)> {
    match ranges.op(func, expr) {
        Some(&Op::Binary(Binary::Isub, x, d)) if x == base => Some((d, 0)),
        Some(&Op::Binary(Binary::Isub, inner, k)) if ranges.constant(k) == Some(1) => {
            sub_delta_of(func, ranges, inner, base)
                .filter(|&(_, c)| c == 0)
                .map(|(d, _)| (d, 1))
        }
        _ => None,
    }
}

fn is_halved_mid(
    func: &Function,
    ranges: &Ranges,
    ty: Type,
    expr: Value,
    lo: Value,
    hi: Value,
) -> bool {
    let Some((d, 0)) = add_delta_of(func, ranges, expr, lo) else {
        return false;
    };
    let Some(&Op::Binary(Binary::Ushr, diff, _)) = ranges.op(func, d) else {
        return false;
    };
    is_nonzero_ushr_of(func, ranges, ty, d, diff)
        && matches!(ranges.op(func, diff), Some(&Op::Binary(Binary::Isub, a, b)) if a == hi && b == lo)
}

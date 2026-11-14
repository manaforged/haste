use crate::ir::{Function, Inst, Op, Type, Value};

use super::bitmask::{build_runs, can_bitmask_runs, match_expr_cond};
use super::emit::{can_br_table, can_signed_br_table, emit_node, forward_cached, push_node, try_signed_br_table};
use super::lut::try_emit_runs_lut;
use super::{
    Arm, Edge, Info, Range, TestKind, body_clean, can_enter, can_follow_into, clean_dead_head_insts,
    collect_hoisted_consts, edge_of, make_range, match_test, push_before_last, range_of_test, same_edge,
};

pub(super) struct Tree {
    val: Value,
    arms: Vec<Arm>,
    default_edge: Edge,
    tree_blocks: Vec<usize>,
    head_dead: Vec<Inst>,
    has_fork: bool,
}

impl Tree {
    pub(super) fn is_forked(&self) -> bool {
        self.has_fork
    }
}

pub(super) struct CondIntervals {
    pub(super) val: Value,
    pub(super) true_ivs: Vec<(u64, u64)>,
    pub(super) false_ivs: Vec<(u64, u64)>,
    pub(super) used: Vec<Inst>,
}

pub(super) fn merge_intervals(mut ivs: Vec<(u64, u64)>) -> Vec<(u64, u64)> {
    if ivs.len() <= 1 {
        return ivs;
    }
    ivs.sort_unstable_by_key(|&(lo, _)| lo);
    let mut out: Vec<(u64, u64)> = Vec::with_capacity(ivs.len());
    for (lo, hi) in ivs {
        if let Some(prev) = out.last_mut()
            && lo <= prev.1.saturating_add(1)
        {
            prev.1 = prev.1.max(hi);
        } else {
            out.push((lo, hi));
        }
    }
    out
}

pub(super) fn intervals_of_range(ty: Type, r: Range) -> Vec<(u64, u64)> {
    if r.lo <= r.hi {
        vec![(r.lo, r.hi)]
    } else {
        vec![(0, r.hi), (r.lo, ty.unsigned_max())]
    }
}

pub(super) fn complement_intervals(u_max: u64, set: &[(u64, u64)]) -> Vec<(u64, u64)> {
    let mut out = Vec::new();
    let mut cur = 0_u64;
    for &(lo, hi) in set {
        if cur < lo {
            out.push((cur, lo - 1));
        }
        if hi == u_max {
            return out;
        }
        cur = hi + 1;
    }
    if cur <= u_max {
        out.push((cur, u_max));
    }
    out
}

pub(super) fn intersect_intervals(a: &[(u64, u64)], b: &[(u64, u64)]) -> Vec<(u64, u64)> {
    let mut out = Vec::new();
    let (mut i, mut j) = (0_usize, 0_usize);
    while i < a.len() && j < b.len() {
        let lo = a[i].0.max(b[j].0);
        let hi = a[i].1.min(b[j].1);
        if lo <= hi {
            out.push((lo, hi));
        }
        if a[i].1 < b[j].1 {
            i += 1;
        } else {
            j += 1;
        }
    }
    out
}

pub(super) fn intervals_of_test(ty: Type, kind: TestKind, want_true: bool) -> Option<Vec<(u64, u64)>> {
    if let Some(r) = range_of_test(ty, kind, want_true) {
        return Some(intervals_of_range(ty, r));
    }
    let r = range_of_test(ty, kind, !want_true)?;
    let comp = complement_intervals(ty.unsigned_max(), &intervals_of_range(ty, r));
    (!comp.is_empty()).then_some(comp)
}

pub(super) fn match_cond_intervals(
    func: &Function,
    info: &Info,
    b: usize,
    cond: Value,
    expected_val: Option<Value>,
    strict_uses: bool,
    check_body: bool,
) -> Option<CondIntervals> {
    if let Some(test) = match_test(func, info, cond, expected_val, strict_uses.then_some(b)) {
        let used: Vec<Inst> = test.used.into_iter().flatten().collect();
        if check_body && !body_clean(func, info, b, &|i| used.contains(&i), None) {
            return None;
        }
        let ty = func.ty(test.val);
        let true_ivs = intervals_of_test(ty, test.kind, true)?;
        let false_ivs = intervals_of_test(ty, test.kind, false)?;
        return Some(CondIntervals { val: test.val, true_ivs, false_ivs, used });
    }
    let expr = match_expr_cond(func, info, b, cond, expected_val, true)?;
    if check_body && !body_clean(func, info, b, &|i| expr.used.contains(&i), None) {
        return None;
    }
    let ty = func.ty(expr.val);
    let mut raw = Vec::new();
    for r in expr.ranges {
        raw.extend(intervals_of_range(ty, r));
    }
    let base = merge_intervals(raw);
    let comp = complement_intervals(ty.unsigned_max(), &base);
    let (true_ivs, false_ivs) = if expr.want_or { (base, comp) } else { (comp, base) };
    if true_ivs.is_empty() || false_ivs.is_empty() {
        return None;
    }
    Some(CondIntervals { val: expr.val, true_ivs, false_ivs, used: expr.used })
}

struct TreeWalk<'a> {
    func: &'a Function,
    info: &'a Info,
    b0: usize,
    cleared: &'a [bool],
    val: Option<Value>,
    tree_blocks: Vec<usize>,
    leaves: Vec<(u64, u64, Edge)>,
    head_dead: Vec<Inst>,
    has_fork: bool,
}

impl TreeWalk<'_> {
    fn visit(&mut self, b: usize, domain: Vec<(u64, u64)>) -> bool {
        if domain.is_empty() || domain.len() > 64 || self.leaves.len() > 256 || self.tree_blocks.len() > 128 {
            return false;
        }
        let is_head = b == self.b0;
        let Some(&last) = self.func.blocks[b].insts.last() else { return false };
        let Op::Brif(cond, t_blk, ref t_args, f_blk, ref f_args) = self.func.insts[last.index()].op else {
            return false;
        };
        let Some(ci) = match_cond_intervals(self.func, self.info, b, cond, self.val, !is_head, !is_head) else {
            return false;
        };
        self.val = Some(ci.val);
        if is_head {
            self.head_dead = ci.used;
        }
        let (t_dom, f_dom) = (intersect_intervals(&domain, &ci.true_ivs), intersect_intervals(&domain, &ci.false_ivs));
        let (t_edge, f_edge) = (edge_of(self.func, t_blk, t_args), edge_of(self.func, f_blk, f_args));
        let mut child_subtrees = 0_u8;
        for (sub_dom, edge) in [(t_dom, t_edge), (f_dom, f_edge)] {
            if sub_dom.is_empty() {
                continue;
            }
            if let Some((next_b, hops)) = can_follow_into(self.func, self.info, (self.b0, self.cleared, &self.tree_blocks), &edge)
                && can_enter(self.func, self.info, (self.b0, self.cleared, &self.tree_blocks), next_b)
                && self.tree_blocks.len() + hops.len() + 1 <= 128
            {
                let saved_tb = self.tree_blocks.len();
                let saved_leaves = self.leaves.len();
                self.tree_blocks.extend(hops);
                self.tree_blocks.push(next_b);
                if self.visit(next_b, sub_dom.clone()) {
                    child_subtrees += 1;
                    continue;
                }
                self.tree_blocks.truncate(saved_tb);
                self.leaves.truncate(saved_leaves);
            }
            for (lo, hi) in sub_dom {
                self.leaves.push((lo, hi, edge.clone()));
            }
        }
        if child_subtrees >= 2 {
            self.has_fork = true;
        }
        self.leaves.len() <= 256
    }
}

pub(super) fn collect_tree(func: &Function, info: &Info, b0: usize, cleared: &[bool]) -> Option<Tree> {
    let Some(&last0) = func.blocks[b0].insts.last() else { return None };
    let Op::Brif(cond0, ..) = func.insts[last0.index()].op else { return None };
    let ci0 = match_cond_intervals(func, info, b0, cond0, None, false, false)?;
    let u_max = func.ty(ci0.val).unsigned_max();
    let mut walk = TreeWalk {
        func,
        info,
        b0,
        cleared,
        val: Some(ci0.val),
        tree_blocks: Vec::new(),
        leaves: Vec::new(),
        head_dead: Vec::new(),
        has_fork: false,
    };
    if !walk.visit(b0, vec![(0, u_max)]) || walk.tree_blocks.is_empty() || walk.leaves.is_empty() {
        return None;
    }
    let val = walk.val?;
    let ty = func.ty(val);
    let mut groups: Vec<(Edge, u128, usize)> = Vec::new();
    for &(lo, hi, ref edge) in &walk.leaves {
        let count = u128::from(hi - lo) + 1;
        if let Some((_, tot, cnt)) = groups.iter_mut().find(|(e, _, _)| same_edge(func, info, e, edge)) {
            *tot += count;
            *cnt += 1;
        } else {
            groups.push((edge.clone(), count, 1));
        }
    }
    let default_edge = groups.into_iter().max_by_key(|&(_, tot, cnt)| (tot, cnt))?.0;
    let arms: Vec<Arm> = walk
        .leaves
        .into_iter()
        .filter(|(_, _, e)| !same_edge(func, info, e, &default_edge))
        .map(|(lo, hi, hit)| Arm { range: make_range(ty, lo, hi), hit })
        .collect();
    Some(Tree { val, arms, default_edge, tree_blocks: walk.tree_blocks, head_dead: walk.head_dead, has_fork: walk.has_fork })
}

pub(super) fn rewrite_tree(func: &mut Function, info: &Info, (b0, last0): (usize, Inst), tree: Tree, cleared: &mut [bool]) -> bool {
    let ty = func.ty(tree.val);
    let runs = build_runs(func, info, ty, &tree.arms);
    let hoisted = collect_hoisted_consts(func, info, &tree.tree_blocks);
    let finish = |func: &mut Function, cleared: &mut [bool]| {
        hoisted.iter().for_each(|&inst| push_before_last(func, b0, inst));
        for b in tree.tree_blocks {
            for inst in func.blocks[b].insts.drain(..) {
                if !hoisted.contains(&inst) {
                    func.insts[inst.index()].op = Op::Trap;
                }
            }
            cleared[b] = true;
        }
        cleared[b0] = true;
        true
    };
    if runs.is_empty() {
        clean_dead_head_insts(func, info, b0, &tree.head_dead);
        let jump = func.add(Op::Jump(tree.default_edge.block, tree.default_edge.args), None);
        push_node(func, Some((b0, last0)), vec![jump]);
        return finish(func, cleared);
    }
    if try_emit_runs_lut(func, info, (b0, last0), tree.val, &runs, &tree.default_edge, &tree.head_dead) {
        return finish(func, cleared);
    }
    let can_simplify = runs.len() <= 2
        || can_bitmask_runs(func, info, ty, &runs)
        || can_br_table(&runs).is_some()
        || can_signed_br_table(&tree.arms);
    if !can_simplify {
        return false;
    }
    clean_dead_head_insts(func, info, b0, &tree.head_dead);
    let mut cache = Vec::new();
    let default_blk = forward_cached(func, &mut cache, tree.default_edge.block, tree.default_edge.args);
    if !try_signed_br_table(func, (b0, last0), (tree.val, ty), &tree.arms, (default_blk, &mut cache)) {
        emit_node(func, info, Some((b0, last0)), (tree.val, ty), &runs, (default_blk, &mut cache));
    }
    finish(func, cleared)
}

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::ir::{Binary, Function, Inst, Op, Type, Value};
use crate::live::results;

use super::{add_use, fill_block_depths, is_fits_paired, set_bin_depth};

fn max_opt(a: Option<usize>, b: Option<usize>) -> Option<usize> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.max(y)),
        (Some(x), None) | (None, Some(x)) => Some(x),
        (None, None) => None,
    }
}

fn small_iadd(func: &Function, inst: Inst, pinned: bool) -> Option<(Value, Value, Value)> {
    if pinned {
        return None;
    }
    let data = &func.insts[inst.index()];
    let (Op::Binary(Binary::Iadd, a, b), Some(res)) = (data.op.clone(), data.result) else {
        return None;
    };
    func.ty(res).is_small_int().then_some((a, b, res))
}

fn add_scan(
    func: &Function,
    insts: &[Inst],
    uses: &[u32],
) -> (Vec<bool>, Vec<Option<usize>>, Vec<usize>) {
    let n = insts.len();
    let pinned: Vec<bool> = (0..n)
        .map(|p| p + 1 < n && is_fits_paired(func, insts[p], insts[p + 1]))
        .collect();
    let mut def_pos = vec![None; func.types.len()];
    for (pos, &inst) in insts.iter().enumerate() {
        let p = if pinned[pos] { pos + 1 } else { pos };
        for r in results(&func.insts[inst.index()]) {
            def_pos[r.index()] = Some(p);
        }
    }
    let mut has_parent = vec![false; n];
    for (pos, &inst) in insts.iter().enumerate() {
        let Some((a, b, res)) = small_iadd(func, inst, pinned[pos]) else {
            continue;
        };
        if a == b {
            continue;
        }
        for c in [a, b] {
            if let Some(cp) = def_pos[c.index()]
                && uses[c.index()] == 1
                && func.ty(c) == func.ty(res)
                && small_iadd(func, insts[cp], pinned[cp]).is_some()
            {
                has_parent[cp] = true;
            }
        }
    }
    let roots = (0..n)
        .filter(|&p| !has_parent[p] && small_iadd(func, insts[p], pinned[p]).is_some())
        .collect();
    (pinned, def_pos, roots)
}

fn flatten_add(
    func: &Function,
    (insts, def_pos, pinned, uses): (&[Inst], &[Option<usize>], &[bool], &[u32]),
    root_pos: usize,
) -> (Vec<Value>, Vec<Inst>) {
    let Some((a, b, root_res)) = small_iadd(func, insts[root_pos], pinned[root_pos]) else {
        return (Vec::new(), Vec::new());
    };
    let ty = func.ty(root_res);
    let (mut leaves, mut interiors, mut stack) = (Vec::new(), Vec::new(), vec![b, a]);
    while let Some(v) = stack.pop() {
        if let Some(cp) = def_pos[v.index()]
            && uses[v.index()] == 1
            && func.ty(v) == ty
            && let Some((x, y, _)) = small_iadd(func, insts[cp], pinned[cp])
            && x != y
        {
            interiors.push(insts[cp]);
            stack.push(y);
            stack.push(x);
        } else {
            leaves.push(v);
        }
    }
    (leaves, interiors)
}

fn pop_pair(
    heap: &mut BinaryHeap<Reverse<(u32, usize, Value)>>,
    two_use: bool,
) -> ((u32, usize, Value), (u32, usize, Value)) {
    if two_use && heap.len() == 3 {
        let (Some(Reverse(e0)), Some(Reverse(e1)), Some(Reverse(e2))) =
            (heap.pop(), heap.pop(), heap.pop())
        else {
            return ((0, 0, Value(0)), (0, 0, Value(0)));
        };
        if e0.0 + 1 < e1.0 && e1.0 == e2.0 {
            heap.push(Reverse(e0));
            return (e1, e2);
        }
        heap.push(Reverse(e2));
        return (e0, e1);
    }
    let (Some(Reverse(a)), Some(Reverse(b))) = (heap.pop(), heap.pop()) else {
        return ((0, 0, Value(0)), (0, 0, Value(0)));
    };
    (a, b)
}

fn plan_add_tree(
    func: &Function,
    (leaves, interiors, root_res, two_use): (&[Value], &[Inst], Value, bool),
    (def_pos, depth): (&[Option<usize>], &[u32]),
) -> Option<(Vec<(Inst, Value, Value, Option<usize>)>, Value, Value, u32)> {
    if leaves.len() < 3 || interiors.len() + 2 != leaves.len() {
        return None;
    }
    let mut heap: BinaryHeap<Reverse<(u32, usize, Value)>> = leaves
        .iter()
        .enumerate()
        .map(|(i, &v)| Reverse((depth[v.index()], i, v)))
        .collect();
    let mut local_pos: Vec<(Value, Option<usize>)> = Vec::with_capacity(interiors.len());
    let get_pos = |v: Value, lp: &[(Value, Option<usize>)]| {
        lp.iter()
            .find(|&&(x, _)| x == v)
            .map_or_else(|| def_pos[v.index()], |&(_, p)| p)
    };
    let mut ops = Vec::with_capacity(interiors.len());
    let mut sum_d = 0u32;
    for &inst in interiors {
        let val = func.insts[inst.index()].result?;
        let ((da, ra, va), (db, rb, vb)) = pop_pair(&mut heap, two_use);
        let d_new = da.max(db) + 1;
        let p_new = max_opt(get_pos(va, &local_pos), get_pos(vb, &local_pos));
        ops.push((inst, va, vb, p_new));
        local_pos.push((val, p_new));
        sum_d = sum_d.saturating_add(d_new);
        heap.push(Reverse((d_new, ra.min(rb), val)));
    }
    let ((da, _, va), (db, _, vb)) = pop_pair(&mut heap, false);
    let new_root_d = da.max(db) + 1;
    let old_root_d = depth[root_res.index()];
    let old_int_sum: u32 = interiors
        .iter()
        .filter_map(|&i| func.insts[i.index()].result)
        .map(|v| depth[v.index()])
        .sum();
    if new_root_d > old_root_d || (new_root_d == old_root_d && sum_d >= old_int_sum) {
        return None;
    }
    Some((ops, va, vb, new_root_d))
}

fn balance_add_trees_block(
    func: &mut Function,
    uses: &[u32],
    depth: &mut Vec<u32>,
    bi: usize,
) -> bool {
    if func.blocks[bi].insts.len() < 8 {
        return false;
    }
    let orig = func.blocks[bi].insts.clone();
    fill_block_depths(func, bi, depth);
    let (pinned, def_pos, roots) = add_scan(func, &orig, uses);
    let (mut killed, mut at_start) = (vec![false; func.insts.len()], Vec::new());
    let mut after: Vec<Vec<Inst>> = vec![Vec::new(); orig.len()];
    let mut changed = false;
    for rp in roots {
        let root_inst = orig[rp];
        let Some(root_res) = func.insts[root_inst.index()].result else {
            continue;
        };
        let (leaves, interiors) = flatten_add(func, (&orig, &def_pos, &pinned, uses), rp);
        let two_use = uses[root_res.index()] == 2;
        let Some((ops, ra, rb, _)) = plan_add_tree(
            func,
            (&leaves, &interiors, root_res, two_use),
            (&def_pos, depth),
        ) else {
            continue;
        };
        for &inst in &interiors {
            killed[inst.index()] = true;
        }
        for (inst, va, vb, p_new) in ops {
            func.insts[inst.index()].op = Op::Binary(Binary::Iadd, va, vb);
            if let Some(p) = p_new {
                after[p].push(inst);
            } else {
                at_start.push(inst);
            }
        }
        func.insts[root_inst.index()].op = Op::Binary(Binary::Iadd, ra, rb);
        changed = true;
    }
    if !changed {
        return false;
    }
    let mut rebuilt = Vec::with_capacity(orig.len());
    rebuilt.extend(at_start);
    for (pos, &inst) in orig.iter().enumerate() {
        if !killed[inst.index()] {
            rebuilt.push(inst);
        }
        rebuilt.extend(after[pos].iter().copied());
    }
    func.blocks[bi].insts = rebuilt;
    true
}

fn block_peak_gpr(func: &Function, insts: &[Inst], total_uses: &[u32]) -> usize {
    let mut in_bi = vec![0u32; func.types.len()];
    let mut tracked = vec![false; func.types.len()];
    for &inst in insts {
        let data = &func.insts[inst.index()];
        let is_stat = matches!(
            data.op,
            Op::Const(_) | Op::SlotAddr(_) | Op::Symbol(_) | Op::StackAddr(..) | Op::FramePointer
        );
        for r in results(data) {
            tracked[r.index()] = func.ty(r).is_small_int() && !is_stat;
        }
        data.op.for_operands(|v| in_bi[v.index()] += 1);
    }
    let used_other: Vec<bool> = (0..in_bi.len()).map(|i| total_uses[i] > in_bi[i]).collect();
    let (mut live, mut peak) = (0usize, 0usize);
    for &inst in insts {
        let data = &func.insts[inst.index()];
        data.op.for_operands(|v| {
            in_bi[v.index()] -= 1;
            if tracked[v.index()] && in_bi[v.index()] == 0 && !used_other[v.index()] {
                live = live.saturating_sub(1);
            }
        });
        for r in results(data) {
            if tracked[r.index()] && (in_bi[r.index()] > 0 || used_other[r.index()]) {
                live += 1;
                peak = peak.max(live);
            }
        }
    }
    peak
}

fn find_two_use_bypass(
    func: &Function,
    (orig, def_pos, pinned): (&[Inst], &[Option<usize>], &[bool]),
    (uses, depth): (&[u32], &[u32]),
    (lhs, rhs, ty): (Value, Value, Type),
) -> Option<(Value, Value, Value, Value)> {
    for (t1, d) in [(lhs, rhs), (rhs, lhs)] {
        if uses[t1.index()] != 2 || t1 == d {
            continue;
        }
        let Some(t1_pos) = def_pos[t1.index()] else {
            continue;
        };
        let Some((x, y, _)) =
            small_iadd(func, orig[t1_pos], pinned[t1_pos]).filter(|_| func.ty(t1) == ty)
        else {
            continue;
        };
        let (late, early) = if depth[x.index()] >= depth[y.index()] {
            (x, y)
        } else {
            (y, x)
        };
        if depth[late.index()] >= depth[early.index()] + 3
            && depth[late.index()] >= depth[d.index()] + 2
        {
            return Some((t1, d, late, early));
        }
    }
    None
}

fn bypass_two_use_adds_block(
    func: &mut Function,
    uses: &mut Vec<u32>,
    depth: &mut Vec<u32>,
    bi: usize,
) {
    let orig = func.blocks[bi].insts.clone();
    if orig.len() < 16 || block_peak_gpr(func, &orig, uses) >= 22 {
        return;
    }
    fill_block_depths(func, bi, depth);
    let (pinned, mut def_pos, _) = add_scan(func, &orig, uses);
    let (mut at_start, mut after, mut changed) = (Vec::new(), vec![Vec::new(); orig.len()], false);
    for (pos, &e_inst) in orig.iter().enumerate() {
        let Some((lhs, rhs, e_res)) = small_iadd(func, e_inst, pinned[pos]) else {
            continue;
        };
        let ty = func.ty(e_res);
        let Some((t1, d, late, early)) = find_two_use_bypass(
            func,
            (&orig, &def_pos, &pinned),
            (uses, depth),
            (lhs, rhs, ty),
        ) else {
            continue;
        };
        let d_sum = func.value(ty);
        let sum_inst = func.add(Op::Binary(Binary::Iadd, early, d), Some(d_sum));
        let p_new = max_opt(def_pos[early.index()], def_pos[d.index()]);
        if let Some(p) = p_new {
            after[p].push(sum_inst);
        } else {
            at_start.push(sum_inst);
        }
        func.insts[e_inst.index()].op = Op::Binary(Binary::Iadd, late, d_sum);
        def_pos.resize(func.types.len(), None);
        def_pos[d_sum.index()] = p_new;
        depth.resize(func.types.len(), 0);
        set_bin_depth(depth, d_sum, early, d);
        set_bin_depth(depth, e_res, late, d_sum);
        add_use(uses, t1, -1);
        add_use(uses, early, 1);
        add_use(uses, late, 1);
        add_use(uses, d_sum, 1);
        changed = true;
    }
    if !changed {
        return;
    }
    let mut rebuilt = Vec::with_capacity(orig.len() + at_start.len());
    rebuilt.extend(at_start);
    for (pos, inst) in orig.into_iter().enumerate() {
        rebuilt.push(inst);
        rebuilt.extend(after[pos].iter().copied());
    }
    func.blocks[bi].insts = rebuilt;
}

pub(super) fn balance_add_trees(func: &mut Function, uses: &mut Vec<u32>) -> bool {
    if !func.blocks.iter().any(|b| b.insts.len() >= 8) {
        return false;
    }
    let mut balanced = false;
    let mut depth = Vec::new();
    for bi in 0..func.blocks.len() {
        balanced |= balance_add_trees_block(func, uses, &mut depth, bi);
    }
    for bi in 0..func.blocks.len() {
        bypass_two_use_adds_block(func, uses, &mut depth, bi);
    }
    balanced
}

mod trap;

use crate::cfg::Cfg;
use crate::fold::extend;
use crate::ir::{Binary, Block, Convert, Domain, Function, Inst, IntCC, Op, Type, Value};

#[derive(Clone, Copy)]
struct MaskedRange {
    mask: u64,
    lo: u64,
    hi: u64,
}

enum BinaryPlan {
    Jump(Block),
    Ranges { idx: Value, ty: Type, hit: Block, default: Block, items: Vec<MaskedRange> },
}

/// Folds every block that only one jump enters into the block that jumps, so its values stay block-local.
pub(crate) fn run(func: &mut Function) -> Domain {
    let mut changed = trap::merge_leaves(func) | trap::merge_trap_calls(func);
    if changed {
        func.bump_cfg();
    }
    if forward_trampolines(func) {
        func.bump_cfg();
        changed = true;
    }
    if lower_binary_br_tables(func) {
        func.bump_cfg();
        changed = true;
    }
    changed |= join_jumps(func, &[]);
    if changed {
        func.bump_cfg();
        Domain::CFG | Domain::VALUES
    } else {
        Domain::NONE
    }
}

fn emit_const(func: &mut Function, insts: &mut Vec<Inst>, ty: Type, val: u64) -> Value {
    let v = func.value(ty);
    insts.push(func.add(Op::Const(extend(val as i64, ty, true)), Some(v)));
    v
}

fn emit_bin(func: &mut Function, insts: &mut Vec<Inst>, ty: Type, op: Binary, (lhs, rhs): (Value, Value)) -> Value {
    let v = func.value(ty);
    insts.push(func.add(Op::Binary(op, lhs, rhs), Some(v)));
    v
}

fn emit_icmp(func: &mut Function, insts: &mut Vec<Inst>, cc: IntCC, (lhs, rhs): (Value, Value)) -> Value {
    let v = func.value(Type::I8);
    insts.push(func.add(Op::Icmp(cc, lhs, rhs), Some(v)));
    v
}

fn collect_hit_intervals(default: Block, targets: &[Block], u_max: u64) -> Option<(Option<Block>, Vec<(u64, u64)>)> {
    let mut hit: Option<Block> = None;
    let mut intervals: Vec<(u64, u64)> = Vec::new();
    for (i, &target) in targets.iter().enumerate() {
        let u = i as u64;
        if u > u_max {
            break;
        }
        if target == default {
            continue;
        }
        match hit {
            None => hit = Some(target),
            Some(prev) if prev != target => return None,
            Some(_) => {}
        }
        if let Some(last) = intervals.last_mut().filter(|last| last.1 + 1 == u) {
            last.1 = u;
        } else {
            intervals.push((u, u));
        }
    }
    Some((hit, intervals))
}

fn try_coalesce_pair(a: MaskedRange, b: MaskedRange) -> Option<MaskedRange> {
    if a.mask != b.mask || a.hi - a.lo != b.hi - b.lo {
        return None;
    }
    let d = b.lo - a.lo;
    if d == 0 || !d.is_power_of_two() || (d & a.mask) != d || (a.lo & d) != 0 || (a.lo ^ a.hi) >= d {
        return None;
    }
    Some(MaskedRange { mask: a.mask & !d, lo: a.lo, hi: a.hi })
}

fn coalesce_pass(items: &mut Vec<MaskedRange>) -> bool {
    let mut changed = false;
    let mut i = 0;
    while i < items.len() {
        if let Some((j, merged)) = ((i + 1)..items.len()).find_map(|j| try_coalesce_pair(items[i], items[j]).map(|m| (j, m))) {
            items[i] = merged;
            items.remove(j);
            changed = true;
        }
        i += 1;
    }
    changed
}

fn coalesce_masked_ranges(intervals: &[(u64, u64)], u_max: u64) -> Vec<MaskedRange> {
    let mut items: Vec<MaskedRange> = intervals.iter().map(|&(lo, hi)| MaskedRange { mask: u_max, lo, hi }).collect();
    if items.len() <= 512 {
        while coalesce_pass(&mut items) {}
    }
    items
}

fn local_def_op(func: &Function, b: usize, val: Value) -> Option<&Op> {
    func.blocks[b].insts.iter().find_map(|&inst| {
        let item = &func.insts[inst.index()];
        (item.result == Some(val)).then_some(&item.op)
    })
}

fn unshifted_br_index(func: &Function, b: usize, idx: Value, table_len: usize) -> Option<(Value, u64)> {
    if table_len == 0 {
        return None;
    }
    let &Op::Binary(Binary::Isub, base_v, min_v) = local_def_op(func, b, idx)? else {
        return None;
    };
    let &Op::Const(min) = local_def_op(func, b, min_v)? else {
        return None;
    };
    if min <= 0 {
        return None;
    }
    let orig = match local_def_op(func, b, base_v) {
        Some(&Op::Convert(Convert::Uextend, narrow)) if func.ty(narrow).is_small_int() => narrow,
        _ => base_v,
    };
    let shift = min as u64;
    let end = shift.checked_add(table_len as u64)?;
    if end > i64::MAX as u64 || end - 1 > func.ty(orig).unsigned_max() {
        return None;
    }
    Some((orig, shift))
}

fn profitable_masked_ranges(items: &[MaskedRange], u_max: u64) -> bool {
    items.len() == 1 || (items.len() <= 2 && items.iter().any(|it| it.mask != u_max))
}

fn binary_br_table_plan(func: &Function, b: usize) -> Option<BinaryPlan> {
    let inst = func.blocks[b].insts.last().copied()?;
    let Op::BrTable(idx, default, ref targets) = func.insts[inst.index()].op else {
        return None;
    };
    let ty = func.ty(idx);
    if !ty.is_small_int() {
        return None;
    }
    let u_max = ty.unsigned_max();
    let (hit_opt, intervals) = collect_hit_intervals(default, targets, u_max)?;
    let Some(hit) = hit_opt else {
        return Some(BinaryPlan::Jump(default));
    };
    if let Some((orig_idx, base)) = unshifted_br_index(func, b, idx, targets.len()) {
        let orig_ty = func.ty(orig_idx);
        let orig_umax = orig_ty.unsigned_max();
        let shifted: Vec<(u64, u64)> = intervals.iter().map(|&(lo, hi)| (lo + base, hi + base)).collect();
        let items = coalesce_masked_ranges(&shifted, orig_umax);
        if profitable_masked_ranges(&items, orig_umax) {
            return Some(BinaryPlan::Ranges { idx: orig_idx, ty: orig_ty, hit, default, items });
        }
    }
    let items = coalesce_masked_ranges(&intervals, u_max);
    profitable_masked_ranges(&items, u_max).then_some(BinaryPlan::Ranges { idx, ty, hit, default, items })
}

fn emit_masked_range(
    func: &mut Function,
    insts: &mut Vec<Inst>,
    (idx, ty, u_max): (Value, Type, u64),
    item: MaskedRange,
) -> Value {
    let v = if item.mask == u_max {
        idx
    } else {
        let mask_c = emit_const(func, insts, ty, item.mask);
        emit_bin(func, insts, ty, Binary::Band, (idx, mask_c))
    };
    if item.lo == item.hi {
        let lo_c = emit_const(func, insts, ty, item.lo);
        emit_icmp(func, insts, IntCC::Equal, (v, lo_c))
    } else if item.lo == 0 {
        let hi_c = emit_const(func, insts, ty, item.hi);
        emit_icmp(func, insts, IntCC::UnsignedLessThanOrEqual, (v, hi_c))
    } else {
        let lo_c = emit_const(func, insts, ty, item.lo);
        let diff = emit_bin(func, insts, ty, Binary::Isub, (v, lo_c));
        let span_c = emit_const(func, insts, ty, item.hi - item.lo);
        emit_icmp(func, insts, IntCC::UnsignedLessThanOrEqual, (diff, span_c))
    }
}

fn prune_dead_local(func: &mut Function, b: usize) {
    loop {
        let uses = func.use_counts();
        let before = func.blocks[b].insts.len();
        func.blocks[b].insts.retain(|&inst| {
            let item = &func.insts[inst.index()];
            let Some(res) = item.result else { return true };
            uses[res.index()] > 0 || !item.op.is_speculable()
        });
        if func.blocks[b].insts.len() == before {
            break;
        }
    }
}

fn apply_binary_plan(func: &mut Function, b: usize, plan: BinaryPlan) {
    let mut insts = Vec::new();
    let new_term = match plan {
        BinaryPlan::Jump(default) => Op::Jump(default, Vec::new()),
        BinaryPlan::Ranges { idx, ty, hit, default, items } => {
            let u_max = ty.unsigned_max();
            let mut acc: Option<Value> = None;
            for item in items {
                let cmp = emit_masked_range(func, &mut insts, (idx, ty, u_max), item);
                acc = Some(match acc {
                    None => cmp,
                    Some(prev) => emit_bin(func, &mut insts, Type::I8, Binary::Bor, (prev, cmp)),
                });
            }
            let Some(cond) = acc else { return };
            Op::Brif(cond, hit, Vec::new(), default, Vec::new())
        }
    };
    let Some(term) = func.blocks[b].insts.pop() else {
        return;
    };
    func.insts[term.index()].op = new_term;
    func.blocks[b].insts.extend(insts);
    func.blocks[b].insts.push(term);
    prune_dead_local(func, b);
}

pub(crate) fn lower_binary_br_tables(func: &mut Function) -> bool {
    let has_br_table = (0..func.blocks.len()).any(|b| matches!(func.last(Block(b as u32)), Some(Op::BrTable(..))));
    if !has_br_table {
        return false;
    }
    let cfg = func.cfg();
    let mut changed = false;
    for b in 0..func.blocks.len() {
        if !cfg.reachable[b] {
            continue;
        }
        if let Some(plan) = binary_br_table_plan(func, b) {
            apply_binary_plan(func, b, plan);
            changed = true;
        }
    }
    changed
}

fn jump_only_target(func: &Function, block: Block) -> Option<(Block, &[Value])> {
    let [inst] = func.blocks[block.index()].insts[..] else { return None };
    let Op::Jump(target, ref args) = func.insts[inst.index()].op else { return None };
    (target != block).then_some((target, args.as_slice()))
}

fn trampoline_target<'a>(func: &'a Function, block: Block, is_trampoline: &[bool]) -> Option<(Block, &'a [Value])> {
    is_trampoline[block.index()].then(|| jump_only_target(func, block)).flatten()
}

fn forward_edge(
    func: &Function,
    is_trampoline: &[bool],
    target: &mut Block,
    args: &mut Vec<Value>,
    require_empty: bool,
) -> bool {
    let (mut cur, mut cur_args) = (*target, args.clone());
    let mut visited = [usize::MAX; 16];
    let mut changed = false;
    for step in 0..16 {
        let Some((next, jump_args)) = trampoline_target(func, cur, is_trampoline) else { break };
        if visited[..step].contains(&next.index()) {
            break;
        }
        let params = &func.blocks[cur.index()].params;
        if params.len() != cur_args.len() {
            break;
        }
        let composed: Vec<Value> = jump_args
            .iter()
            .map(|&v| params.iter().position(|&p| p == v).map_or(v, |idx| cur_args[idx]))
            .collect();
        if require_empty && !composed.is_empty() {
            break;
        }
        visited[step] = cur.index();
        cur = next;
        cur_args = composed;
        changed = true;
    }
    if changed {
        *target = cur;
        *args = cur_args;
    }
    changed
}

/// Whether block `b` is not the entry and holds only a jump to another block.
fn jumps_onward(func: &Function, entry: Option<usize>, b: usize) -> bool {
    Some(b) != entry && jump_only_target(func, Block(b as u32)).is_some()
}

fn outside_param_uses(func: &Function, cfg: &Cfg, param_owner: &[usize]) -> Vec<bool> {
    let mut used_outside = vec![false; func.types.len()];
    for (b, data) in func.blocks.iter().enumerate() {
        if !cfg.reachable[b] {
            continue;
        }
        for &inst in &data.insts {
            func.insts[inst.index()].op.for_operands(|v| {
                let owner = param_owner[v.index()];
                if owner != usize::MAX && owner != b {
                    used_outside[v.index()] = true;
                }
            });
        }
    }
    used_outside
}

fn compute_trampoline_mask(func: &Function, cfg: &Cfg, entry: Option<usize>) -> Option<Vec<bool>> {
    let n = func.blocks.len();
    if !(0..n).any(|b| cfg.reachable[b] && jumps_onward(func, entry, b)) {
        return None;
    }
    let mut param_owner = vec![usize::MAX; func.types.len()];
    let mut has_params = false;
    for (b, data) in func.blocks.iter().enumerate() {
        if cfg.reachable[b] {
            for &p in &data.params {
                param_owner[p.index()] = b;
                has_params = true;
            }
        }
    }
    let used_outside = if has_params { outside_param_uses(func, cfg, &param_owner) } else { Vec::new() };
    let mask: Vec<bool> = (0..n)
        .map(|b| cfg.reachable[b] && jumps_onward(func, entry, b) && func.blocks[b].params.iter().all(|p| !used_outside[p.index()]))
        .collect();
    mask.iter().any(|&t| t).then_some(mask)
}

fn collapse_identical_brifs(func: &mut Function, cfg: &Cfg) -> bool {
    let mut changed = false;
    for b in 0..func.blocks.len() {
        if !cfg.reachable[b] {
            continue;
        }
        let Some(&term) = func.blocks[b].insts.last() else { continue };
        if let Op::Brif(_, then, ref then_args, otherwise, ref else_args) = func.insts[term.index()].op
            && then == otherwise
            && then_args == else_args
        {
            let args = then_args.clone();
            func.insts[term.index()].op = Op::Jump(then, args);
            changed = true;
        }
    }
    changed
}

fn forward_term_op(func: &Function, is_trampoline: &[bool], op: &mut Op) -> bool {
    match op {
        Op::Jump(target, args) => forward_edge(func, is_trampoline, target, args, false),
        Op::Brif(_, then, then_args, otherwise, else_args) => {
            let mut did = forward_edge(func, is_trampoline, then, then_args, false);
            did |= forward_edge(func, is_trampoline, otherwise, else_args, false);
            if then == otherwise && then_args == else_args {
                *op = Op::Jump(*then, std::mem::take(then_args));
                did = true;
            }
            did
        }
        Op::BrTable(_, default, targets) => {
            let mut empty = Vec::new();
            let mut did = forward_edge(func, is_trampoline, default, &mut empty, true);
            for target in targets {
                let mut empty = Vec::new();
                did |= forward_edge(func, is_trampoline, target, &mut empty, true);
            }
            did
        }
        Op::TryCall { normal, .. } => {
            let mut empty = Vec::new();
            forward_edge(func, is_trampoline, normal, &mut empty, true)
        }
        _ => false,
    }
}

fn forward_trampolines(func: &mut Function) -> bool {
    if func.coroutine.is_some() {
        return false;
    }
    let cfg = func.cfg();
    let collapsed = collapse_identical_brifs(func, &cfg);
    let Some(is_trampoline) = compute_trampoline_mask(func, &cfg, func.entry.map(Block::index)) else {
        return collapsed;
    };
    let terms: Vec<Option<(usize, Inst)>> = func.blocks.iter().enumerate().map(|(b, data)| data.insts.last().copied().map(|i| (b, i))).collect();
    let mut forwarded = false;
    for &(b, term) in terms.iter().flatten() {
        if !cfg.reachable[b] {
            continue;
        }
        let mut op = func.insts[term.index()].op.clone();
        if forward_term_op(func, &is_trampoline, &mut op) {
            func.insts[term.index()].op = op;
            forwarded = true;
        }
    }
    if !forwarded {
        return collapsed;
    }
    let mut referenced = vec![false; func.blocks.len()];
    for (_, term) in terms.into_iter().flatten() {
        func.insts[term.index()].op.targets().for_each(|b| referenced[b.index()] = true);
    }
    for (b, is_t) in is_trampoline.into_iter().enumerate() {
        if is_t && !referenced[b] {
            if let Some(inst) = func.blocks[b].insts.pop() {
                func.insts[inst.index()].op = Op::Trap;
            }
            func.blocks[b].params.clear();
        }
    }
    true
}

#[derive(Clone, Copy)]
enum FoldedCmp {
    Const(bool),
    Single(IntCC, u64),
    Range(u64, u64),
    MaskEq(u64, u64, bool),
}

fn swap_int_cc(cc: IntCC) -> IntCC {
    match cc {
        IntCC::SignedLessThan => IntCC::SignedGreaterThan,
        IntCC::SignedGreaterThanOrEqual => IntCC::SignedLessThanOrEqual,
        IntCC::SignedGreaterThan => IntCC::SignedLessThan,
        IntCC::SignedLessThanOrEqual => IntCC::SignedGreaterThanOrEqual,
        IntCC::UnsignedLessThan => IntCC::UnsignedGreaterThan,
        IntCC::UnsignedGreaterThanOrEqual => IntCC::UnsignedLessThanOrEqual,
        IntCC::UnsignedGreaterThan => IntCC::UnsignedLessThan,
        IntCC::UnsignedLessThanOrEqual => IntCC::UnsignedGreaterThanOrEqual,
        other => other,
    }
}

fn def_const(func: &Function, defs: &[Option<Inst>], val: Value) -> Option<i64> {
    match func.insts[defs[val.index()]?.index()].op { Op::Const(imm) => Some(imm), _ => None }
}

fn parse_const_cmp(func: &Function, defs: &[Option<Inst>], mut cond: Value) -> Option<(Value, IntCC, i64)> {
    let mut invert = false;
    for _ in 0..2 {
        match func.insts[defs[cond.index()]?.index()].op {
            Op::Binary(Binary::Bxor, a, b) if def_const(func, defs, b) == Some(1) => { cond = a; invert = !invert; }
            Op::Binary(Binary::Bxor, a, b) if def_const(func, defs, a) == Some(1) => { cond = b; invert = !invert; }
            Op::Icmp(cc, a, b) if func.ty(a).is_small_int() => {
                let (x, eff_cc, imm) = if let Some(k) = def_const(func, defs, b) { (a, cc, k) } else { (b, swap_int_cc(cc), def_const(func, defs, a)?) };
                return Some((x, if invert { eff_cc.inverse() } else { eff_cc }, imm));
            }
            _ => return None,
        }
    }
    None
}

fn range_bounds(cc: IntCC, raw_k: i64, (u_max, sign_bit): (u64, u64), signed: bool) -> Option<(u64, u64)> {
    let key = ((raw_k as u64) & u_max) ^ sign_bit;
    if cc == IntCC::Equal { return Some((key, key)); }
    if cc == IntCC::NotEqual { return (key == 0).then_some((1, u_max)).or_else(|| (key == u_max).then_some((0, u_max - 1))); }
    if cc.is_signed() != signed { return None; }
    match cc {
        IntCC::SignedGreaterThanOrEqual | IntCC::UnsignedGreaterThanOrEqual => Some((key, u_max)),
        IntCC::SignedGreaterThan | IntCC::UnsignedGreaterThan => (key < u_max).then_some((key + 1, u_max)),
        IntCC::SignedLessThanOrEqual | IntCC::UnsignedLessThanOrEqual => Some((0, key)),
        IntCC::SignedLessThan | IntCC::UnsignedLessThan => (key > 0).then_some((0, key - 1)),
        _ => None,
    }
}

fn fold_interval_pair(ty: Type, (cc1, k1): (IntCC, i64), (cc2, k2): (IntCC, i64), signed: bool) -> Option<FoldedCmp> {
    let (u_max, sign_bit) = (ty.unsigned_max(), if signed { 1_u64 << (ty.bits() - 1) } else { 0 });
    let (a_lo, a_hi) = range_bounds(cc1, k1, (u_max, sign_bit), signed)?;
    let (b_lo, b_hi) = range_bounds(cc2, k2, (u_max, sign_bit), signed)?;
    let (lo, hi) = (a_lo.max(b_lo), a_hi.min(b_hi));
    if lo > hi { return Some(FoldedCmp::Const(false)); }
    if lo == 0 && hi == u_max { return Some(FoldedCmp::Const(true)); }
    let span = hi - lo;
    if span >= u_max { return None; }
    let (lo_raw, hi_raw) = (lo ^ sign_bit, hi ^ sign_bit);
    if lo == hi { return Some(FoldedCmp::Single(IntCC::Equal, lo_raw)); }
    if hi == u_max { return Some(FoldedCmp::Single(if signed { IntCC::SignedGreaterThanOrEqual } else { IntCC::UnsignedGreaterThanOrEqual }, lo_raw)); }
    if lo == 0 { return Some(FoldedCmp::Single(if signed { IntCC::SignedLessThanOrEqual } else { IntCC::UnsignedLessThanOrEqual }, hi_raw)); }
    if lo_raw == 0 { return Some(FoldedCmp::Single(IntCC::UnsignedLessThanOrEqual, span)); }
    Some(FoldedCmp::Range(lo_raw, span))
}

fn fold_and_cmps(ty: Type, (cc1, k1): (IntCC, i64), (cc2, k2): (IntCC, i64)) -> Option<FoldedCmp> {
    if cc1 == IntCC::NotEqual && cc2 == IntCC::NotEqual {
        let u_max = ty.unsigned_max();
        let (u1, u2) = ((k1 as u64) & u_max, (k2 as u64) & u_max);
        if u1 == u2 { return Some(FoldedCmp::Single(IntCC::NotEqual, u1)); }
        let d = u1 ^ u2;
        if d.is_power_of_two() { return Some(FoldedCmp::MaskEq(u_max & !d, u1 & !d, true)); }
    }
    fold_interval_pair(ty, (cc1, k1), (cc2, k2), false).or_else(|| fold_interval_pair(ty, (cc1, k1), (cc2, k2), true))
}

fn mid_self_contained(func: &Function, uses: &[u32], mid: usize) -> bool {
    let insts = &func.blocks[mid].insts;
    let Some((&term, body)) = insts.split_last() else { return false };
    if insts.len() > 5 || !body.iter().all(|&i| func.insts[i.index()].op.is_speculable()) { return false; }
    let Op::Brif(cond, _, _, _, _) = func.insts[term.index()].op else { return false };
    body.iter().all(|&i| {
        let Some(res) = func.insts[i.index()].result else { return true };
        let internal: u32 = body.iter().map(|&j| { let mut n = 0; func.insts[j.index()].op.for_operands(|v| if v == res { n += 1; }); n }).sum::<u32>() + u32::from(cond == res);
        uses[res.index()] == internal
    })
}

fn emit_folded_cmp(func: &mut Function, insts: &mut Vec<Inst>, x: Value, ty: Type, folded: FoldedCmp) -> Option<Value> {
    match folded {
        FoldedCmp::Const(_) => None,
        FoldedCmp::Single(cc, k) => { let c = emit_const(func, insts, ty, k); Some(emit_icmp(func, insts, cc, (x, c))) }
        FoldedCmp::Range(lo_raw, span) => {
            let lo_c = emit_const(func, insts, ty, lo_raw);
            let diff = emit_bin(func, insts, ty, Binary::Isub, (x, lo_c));
            let span_c = emit_const(func, insts, ty, span);
            Some(emit_icmp(func, insts, IntCC::UnsignedLessThanOrEqual, (diff, span_c)))
        }
        FoldedCmp::MaskEq(mask, expect, ne) => {
            let mask_c = emit_const(func, insts, ty, mask);
            let masked = emit_bin(func, insts, ty, Binary::Band, (x, mask_c));
            let exp_c = emit_const(func, insts, ty, expect);
            Some(emit_icmp(func, insts, if ne { IntCC::NotEqual } else { IntCC::Equal }, (masked, exp_c)))
        }
    }
}

type TriangleMatch = (usize, Value, FoldedCmp, (Block, Vec<Value>), (Block, Vec<Value>));

fn try_mid_arm(
    func: &Function,
    (cfg, uses, defs): (&Cfg, &[u32], &[Option<Inst>]),
    (b, x1, eff1, k1): (usize, Value, IntCC, i64),
    (mid_blk, other_blk, other_args): (Block, Block, &[Value]),
) -> Option<TriangleMatch> {
    let mid = mid_blk.index();
    let eligible = mid != b && Some(mid) != func.entry.map(Block::index) && cfg.pred_count(mid_blk) == 1 && func.blocks[mid].params.is_empty() && mid_self_contained(func, uses, mid);
    if !eligible { return None; }
    let Op::Brif(c2, t2, ta2, e2, ea2) = func.last(mid_blk)?.clone() else { return None };
    let (x2, cc2, k2) = parse_const_cmp(func, defs, c2)?;
    if x1 != x2 { return None; }
    let (true_tgt, inv2) = if e2 == other_blk && ea2.as_slice() == other_args { ((t2, ta2), false) } else if t2 == other_blk && ta2.as_slice() == other_args { ((e2, ea2), true) } else { return None };
    let eff2 = if inv2 { cc2.inverse() } else { cc2 };
    fold_and_cmps(func.ty(x1), (eff1, k1), (eff2, k2)).map(|f| (mid, x1, f, true_tgt, (other_blk, other_args.to_vec())))
}

fn match_triangle_arm(func: &Function, cfg: &Cfg, uses: &[u32], defs: &[Option<Inst>], b: usize) -> Option<TriangleMatch> {
    let Op::Brif(c1, t1, ta1, e1, ea1) = func.last(Block(b as u32))?.clone() else { return None };
    if t1 == e1 { return None; }
    let (x1, cc1, k1) = parse_const_cmp(func, defs, c1)?;
    if ta1.is_empty() && let Some(m) = try_mid_arm(func, (cfg, uses, defs), (b, x1, cc1, k1), (t1, e1, &ea1)) { return Some(m); }
    if ea1.is_empty() { return try_mid_arm(func, (cfg, uses, defs), (b, x1, cc1.inverse(), k1), (e1, t1, &ta1)); }
    None
}

fn collect_triangle_round(func: &Function) -> Vec<(usize, TriangleMatch)> {
    if func.blocks.len() < 3 { return Vec::new(); }
    let has_brif_pair = func.blocks.iter().enumerate().any(|(b, _)| {
        matches!(func.last(Block(b as u32)), Some(&Op::Brif(_, t1, _, e1, _)) if t1 != e1 && (
            matches!(func.last(t1), Some(&Op::Brif(_, t2, _, e2, _)) if t2 != e2)
            || matches!(func.last(e1), Some(&Op::Brif(_, t2, _, e2, _)) if t2 != e2)
        ))
    });
    if !has_brif_pair { return Vec::new(); }
    let (cfg, uses) = (func.cfg(), func.use_counts());
    let mut defs = vec![None; func.types.len()];
    for inst in func.blocks.iter().flat_map(|blk| &blk.insts) {
        if let Some(res) = func.insts[inst.index()].result { defs[res.index()] = Some(*inst); }
    }
    let mut busy = vec![false; func.blocks.len()];
    let mut out = Vec::new();
    for b in 0..func.blocks.len() {
        if !cfg.reachable[b] || busy[b] { continue; }
        let Some(m) = match_triangle_arm(func, &cfg, &uses, &defs, b) else { continue };
        let (mid, t_b, f_b) = (m.0, (m.3).0.index(), (m.4).0.index());
        if busy[mid] { continue; }
        for idx in [b, mid, t_b, f_b] { busy[idx] = true; }
        out.push((b, m));
    }
    out
}

/// Folds load-free single-operand interval and power-of-two bit-pair short-circuit triangles into one compare.
pub(crate) fn fold_short_circuit_compares(func: &mut Function) -> bool {
    let mut changed = false;
    for _ in 0..8 {
        let round = collect_triangle_round(func);
        if round.is_empty() { break; }
        for (b, (mid, x, folded, t_tgt, f_tgt)) in round {
            let (ty, mut extra) = (func.ty(x), Vec::new());
            let new_term = match emit_folded_cmp(func, &mut extra, x, ty, folded) {
                Some(cond) => Op::Brif(cond, t_tgt.0, t_tgt.1, f_tgt.0, f_tgt.1),
                None => Op::Jump(if matches!(folded, FoldedCmp::Const(true)) { t_tgt.0 } else { f_tgt.0 }, if matches!(folded, FoldedCmp::Const(true)) { t_tgt.1 } else { f_tgt.1 }),
            };
            let Some(term_inst) = func.blocks[b].insts.pop() else { continue };
            func.insts[term_inst.index()].op = new_term;
            func.blocks[b].insts.extend(extra);
            func.blocks[b].insts.push(term_inst);
            if let Some(m_term) = func.blocks[mid].insts.pop() {
                func.insts[m_term.index()].op = Op::Trap;
                func.blocks[mid].insts.clear();
                func.blocks[mid].insts.push(m_term);
            }
            prune_dead_local(func, b);
        }
        func.bump_cfg();
        changed = true;
    }
    changed
}

/// Builds an identity replacement table mapping each of `count` values to itself.
pub(crate) fn identity(count: usize) -> Vec<Value> {
    (0..count as u32).map(Value).collect()
}

/// Folds every block that only one jump enters into the block that jumps, except the blocks in `keep`.
pub(crate) fn join_jumps(func: &mut Function, keep: &[Block]) -> bool {
    let cfg = func.cfg();
    let mut subst = identity(func.types.len());
    let mut merged = false;
    for block in 0..func.blocks.len() {
        while let Some(next) = single_jump(func, &cfg, block).filter(|&next| !keep.contains(&Block(next as u32))) {
            absorb(func, block, next, &mut subst);
            merged = true;
        }
    }
    if merged {
        rename(func, &subst);
    }
    merged
}

fn single_jump(func: &Function, cfg: &Cfg, block: usize) -> Option<usize> {
    let Op::Jump(next, _) = func.last(Block(block as u32))? else {
        return None;
    };
    let next_idx = next.index();
    let entry = func.entry.map(Block::index);
    (next_idx != block && cfg.pred_count(*next) == 1 && Some(next_idx) != entry && !func.blocks[next_idx].insts.is_empty()).then_some(next_idx)
}

fn absorb(func: &mut Function, block: usize, next: usize, subst: &mut [Value]) {
    let Some(last) = func.blocks[block].insts.pop() else {
        return;
    };
    if let Op::Jump(_, args) = std::mem::replace(&mut func.insts[last.index()].op, Op::Trap) {
        for (param, arg) in func.blocks[next].params.iter().zip(args) {
            subst[param.index()] = arg;
        }
    }
    let moved = std::mem::take(&mut func.blocks[next]);
    func.blocks[block].insts.extend(moved.insts);
}

/// Replaces every operand by the value it stands for in `subst`.
pub(crate) fn rename(func: &mut Function, subst: &[Value]) {
    for data in &mut func.insts {
        data.op.map_operands(|value| find(subst, value));
    }
}

/// The value that `value` stands for once every replacement in `subst` is followed.
pub(crate) fn find(subst: &[Value], mut value: Value) -> Value {
    while let Some(&next) = subst.get(value.index()).filter(|&&next| next != value) {
        value = next;
    }
    value
}

/// Follows every replacement in `subst` from `value` to its root and points the chain straight at it.
pub(crate) fn find_compress(subst: &mut [Value], value: Value) -> Value {
    let root = find(subst, value);
    let mut at = value;
    while let Some(slot) = subst.get_mut(at.index()).filter(|slot| **slot != root) {
        at = std::mem::replace(slot, root);
    }
    root
}

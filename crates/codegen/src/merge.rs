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
    let forwarded = forward_trampolines(func);
    if forwarded {
        func.bump_cfg();
    }
    let lowered = lower_binary_br_tables(func);
    if lowered {
        func.bump_cfg();
    }
    let cfg = func.cfg();
    let mut subst = identity(func.types.len());
    let mut merged = false;
    for block in 0..func.blocks.len() {
        while let Some(next) = single_jump(func, &cfg, block) {
            absorb(func, block, next, &mut subst);
            merged = true;
        }
    }
    if merged {
        rename(func, &subst);
    }
    let traps = merge_traps(func);
    if forwarded || lowered || merged || traps {
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

fn compute_trampoline_mask(func: &Function, entry: Option<usize>) -> Option<Vec<bool>> {
    let n = func.blocks.len();
    if !(0..n).any(|b| jumps_onward(func, entry, b)) {
        return None;
    }
    let mut param_owner = vec![usize::MAX; func.types.len()];
    let mut has_params = false;
    for (b, data) in func.blocks.iter().enumerate() {
        for &p in &data.params {
            param_owner[p.index()] = b;
            has_params = true;
        }
    }
    let mut used_outside = vec![false; if has_params { func.types.len() } else { 0 }];
    if has_params {
        for (b, data) in func.blocks.iter().enumerate() {
            for &inst in &data.insts {
                func.insts[inst.index()].op.for_operands(|v| {
                    if param_owner[v.index()] != usize::MAX && param_owner[v.index()] != b {
                        used_outside[v.index()] = true;
                    }
                });
            }
        }
    }
    let mask: Vec<bool> = (0..n).map(|b| jumps_onward(func, entry, b) && func.blocks[b].params.iter().all(|p| !used_outside[p.index()])).collect();
    mask.iter().any(|&t| t).then_some(mask)
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
    let Some(is_trampoline) = compute_trampoline_mask(func, func.entry.map(Block::index)) else {
        return false;
    };
    let cfg = func.cfg();
    let terms: Vec<Option<(usize, Inst)>> = func.blocks.iter().enumerate().map(|(b, data)| data.insts.last().copied().map(|i| (b, i))).collect();
    let mut changed = false;
    for &(b, term) in terms.iter().flatten() {
        if !cfg.reachable[b] {
            continue;
        }
        let mut op = func.insts[term.index()].op.clone();
        if forward_term_op(func, &is_trampoline, &mut op) {
            func.insts[term.index()].op = op;
            changed = true;
        }
    }
    if !changed {
        return false;
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

fn is_trap_block(func: &Function, block: usize) -> bool {
    let data = &func.blocks[block];
    data.params.is_empty() && matches!(data.insts[..], [inst] if func.insts[inst.index()].op == Op::Trap)
}

fn redirect_trap_target(target: &mut Block, empty_args: bool, is_dup: &[bool], canon: Block, kept_dup: &mut [bool]) -> bool {
    let idx = target.index();
    if !is_dup[idx] || *target == canon {
        return false;
    }
    if empty_args {
        *target = canon;
        return true;
    }
    kept_dup[idx] = true;
    false
}

fn redirect_trap_op(op: &mut Op, is_dup: &[bool], canon: Block, kept_dup: &mut [bool]) -> bool {
    match op {
        Op::Jump(target, args) => redirect_trap_target(target, args.is_empty(), is_dup, canon, kept_dup),
        Op::Brif(_, then, then_args, otherwise, else_args) => {
            let a = redirect_trap_target(then, then_args.is_empty(), is_dup, canon, kept_dup);
            let b = redirect_trap_target(otherwise, else_args.is_empty(), is_dup, canon, kept_dup);
            a || b
        }
        Op::BrTable(_, default, targets) => {
            let mut did = redirect_trap_target(default, true, is_dup, canon, kept_dup);
            for target in targets {
                did |= redirect_trap_target(target, true, is_dup, canon, kept_dup);
            }
            did
        }
        Op::TryCall { normal, .. } => redirect_trap_target(normal, true, is_dup, canon, kept_dup),
        _ => false,
    }
}

fn merge_traps(func: &mut Function) -> bool {
    let entry = func.entry.map(Block::index);
    let mut canonical = None;
    let mut is_dup = vec![false; func.blocks.len()];
    let mut has_dup = false;
    for block in 0..func.blocks.len() {
        if Some(block) != entry && is_trap_block(func, block) {
            if canonical.is_none() {
                canonical = Some(Block(block as u32));
            } else {
                is_dup[block] = true;
                has_dup = true;
            }
        }
    }
    let Some(canon) = canonical.filter(|_| has_dup) else {
        return false;
    };
    let mut changed = false;
    let mut kept_dup = vec![false; func.blocks.len()];
    for inst in &mut func.insts {
        changed |= redirect_trap_op(&mut inst.op, &is_dup, canon, &mut kept_dup);
    }
    for (block, dup) in is_dup.into_iter().enumerate() {
        if dup && !kept_dup[block] {
            func.blocks[block].insts.clear();
            changed = true;
        }
    }
    changed
}

/// Builds an identity replacement table mapping each of `count` values to itself.
pub(crate) fn identity(count: usize) -> Vec<Value> {
    (0..count as u32).map(Value).collect()
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

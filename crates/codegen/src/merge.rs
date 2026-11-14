use crate::cfg::Cfg;
use crate::ir::{Block, Domain, Function, Op, Value};

/// Folds every block that only one jump enters into the block that jumps, so its values stay block-local.
pub(crate) fn run(func: &mut Function) -> Domain {
    let forwarded = forward_trampolines(func);
    if forwarded {
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
    if forwarded || merged || traps {
        func.bump_cfg();
        Domain::CFG | Domain::VALUES
    } else {
        Domain::NONE
    }
}

fn trampoline_target<'a>(func: &'a Function, block: Block, is_trampoline: &[bool]) -> Option<(Block, &'a [Value])> {
    if !is_trampoline[block.index()] {
        return None;
    }
    let data = &func.blocks[block.index()];
    let [inst] = data.insts[..] else { return None };
    let Op::Jump(target, ref args) = func.insts[inst.index()].op else { return None };
    (target != block).then_some((target, args.as_slice()))
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

fn forward_trampolines(func: &mut Function) -> bool {
    if func.coroutine.is_some() {
        return false;
    }
    let entry = func.entry.map(Block::index);
    let n = func.blocks.len();
    let has_cand = (0..n).any(|b| {
        Some(b) != entry
            && matches!(func.blocks[b].insts[..], [inst] if matches!(func.insts[inst.index()].op, Op::Jump(t, _) if t.index() != b))
    });
    if !has_cand {
        return false;
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
    let is_trampoline: Vec<bool> = (0..n)
        .map(|b| {
            Some(b) != entry
                && matches!(func.blocks[b].insts[..], [inst] if matches!(func.insts[inst.index()].op, Op::Jump(t, _) if t.index() != b))
                && func.blocks[b].params.iter().all(|p| !used_outside[p.index()])
        })
        .collect();
    if !is_trampoline.iter().any(|&t| t) {
        return false;
    }
    let cfg = func.cfg();
    let mut changed = false;
    let terms: Vec<Option<(usize, crate::ir::Inst)>> = func
        .blocks
        .iter()
        .enumerate()
        .map(|(b, data)| data.insts.last().copied().map(|i| (b, i)))
        .collect();
    for &(b, term) in terms.iter().flatten() {
        if !cfg.reachable[b] {
            continue;
        }
        let mut op = func.insts[term.index()].op.clone();
        match &mut op {
            Op::Jump(target, args) => {
                changed |= forward_edge(func, &is_trampoline, target, args, false);
            }
            Op::Brif(_, then, then_args, otherwise, else_args) => {
                changed |= forward_edge(func, &is_trampoline, then, then_args, false);
                changed |= forward_edge(func, &is_trampoline, otherwise, else_args, false);
                if then == otherwise && then_args == else_args {
                    op = Op::Jump(*then, std::mem::take(then_args));
                    changed = true;
                }
            }
            Op::BrTable(_, default, targets) => {
                let mut empty = Vec::new();
                changed |= forward_edge(func, &is_trampoline, default, &mut empty, true);
                for target in targets {
                    let mut empty = Vec::new();
                    changed |= forward_edge(func, &is_trampoline, target, &mut empty, true);
                }
            }
            Op::TryCall { normal, .. } => {
                let mut empty = Vec::new();
                changed |= forward_edge(func, &is_trampoline, normal, &mut empty, true);
            }
            _ => {}
        }
        func.insts[term.index()].op = op;
    }
    if !changed {
        return false;
    }
    let mut referenced = vec![false; n];
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
        let mut redirect = |target: &mut Block, empty_args: bool| {
            let idx = target.index();
            if is_dup[idx] && *target != canon {
                if empty_args {
                    *target = canon;
                    changed = true;
                } else {
                    kept_dup[idx] = true;
                }
            }
        };
        match &mut inst.op {
            Op::Jump(target, args) => redirect(target, args.is_empty()),
            Op::Brif(_, then, then_args, otherwise, else_args) => {
                redirect(then, then_args.is_empty());
                redirect(otherwise, else_args.is_empty());
            }
            Op::BrTable(_, default, targets) => {
                redirect(default, true);
                for target in targets {
                    redirect(target, true);
                }
            }
            Op::TryCall { normal, .. } => redirect(normal, true),
            _ => {}
        }
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

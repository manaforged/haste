use crate::cfg::Cfg;
use crate::ir::{Block, Function, Op, Value};

/// Folds every block that only one jump enters into the block that jumps, so its values stay block-local.
pub(crate) fn run(func: &mut Function) -> bool {
    let cfg = Cfg::new(func);
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
    merged | merge_traps(func)
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

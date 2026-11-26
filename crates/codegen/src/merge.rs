mod br_table;
mod trampoline;
mod trap;
mod triangle;

use crate::cfg::Cfg;
use crate::fold::extend;
use crate::ir::{Binary, Block, Domain, Function, Inst, IntCC, Op, Type, Value};

pub(crate) use br_table::lower_binary_br_tables;
pub(crate) use triangle::fold_short_circuit_compares;

/// Folds every block that only one jump enters into the block that jumps, so its values stay block-local.
pub(crate) fn run(func: &mut Function) -> Domain {
    let mut changed = trap::merge_leaves(func) | trap::merge_trap_calls(func);
    if changed {
        func.bump_cfg();
    }
    if trampoline::forward_trampolines(func) {
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

pub(super) fn emit_const(func: &mut Function, insts: &mut Vec<Inst>, ty: Type, val: u64) -> Value {
    let v = func.value(ty);
    insts.push(func.add(Op::Const(extend(val as i64, ty, true)), Some(v)));
    v
}

pub(super) fn emit_bin(
    func: &mut Function,
    insts: &mut Vec<Inst>,
    ty: Type,
    op: Binary,
    (lhs, rhs): (Value, Value),
) -> Value {
    let v = func.value(ty);
    insts.push(func.add(Op::Binary(op, lhs, rhs), Some(v)));
    v
}

pub(super) fn emit_icmp(
    func: &mut Function,
    insts: &mut Vec<Inst>,
    cc: IntCC,
    (lhs, rhs): (Value, Value),
) -> Value {
    let v = func.value(Type::I8);
    insts.push(func.add(Op::Icmp(cc, lhs, rhs), Some(v)));
    v
}

pub(super) fn prune_dead_local(func: &mut Function, b: usize) {
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
        while let Some(next) =
            single_jump(func, &cfg, block).filter(|&next| !keep.contains(&Block(next as u32)))
        {
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
    (next_idx != block
        && cfg.pred_count(*next) == 1
        && Some(next_idx) != entry
        && !func.blocks[next_idx].insts.is_empty())
    .then_some(next_idx)
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

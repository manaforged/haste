use crate::ir::{Block, Function, Op, Value};

use super::for_edges;

pub(super) fn prune_changed(func: &mut Function, entry: Block, cached: bool) -> (Vec<bool>, bool) {
    let reachable = if cached && Some(entry) == func.entry {
        func.cfg().reachable.clone()
    } else {
        crate::cfg::reverse_postorder(func, entry).1
    };
    let mut changed = false;
    for (block, live) in func.blocks.iter_mut().zip(&reachable) {
        if !live && !block.insts.is_empty() {
            for inst in block.insts.drain(..) {
                func.insts[inst.index()].op = Op::Trap;
            }
            changed = true;
        }
    }
    if changed {
        func.bump_cfg();
    }
    (reachable, changed)
}

/// Empties the blocks no path from `entry` reaches and returns which blocks a path reaches.
pub(crate) fn prune(func: &mut Function, entry: Block) -> Vec<bool> {
    prune_changed(func, entry, true).0
}

/// Drops pure instructions whose results nothing reads, in one backward pass that also frees the operands they read.
pub(super) fn dead(func: &mut Function) -> bool {
    let mut uses = func.use_counts();
    let mut keep = Vec::new();
    let mut dropped = false;
    for block in func.blocks.iter_mut().rev() {
        keep.clear();
        keep.resize(block.insts.len(), true);
        for (at, inst) in block.insts.iter().enumerate().rev() {
            let data = &func.insts[inst.index()];
            if data.op.is_removable() && data.result.is_some_and(|result| uses[result.index()] == 0) {
                keep[at] = false;
                dropped = true;
                data.op.for_operands(|value| uses[value.index()] -= 1);
            }
        }
        let mut at = 0;
        block.insts.retain(|_| {
            at += 1;
            keep[at - 1]
        });
    }
    dropped
}

/// Drops each block parameter that nothing reads but edge arguments of other dropped parameters, with its argument on every edge; parameters of `pinned` blocks stay.
pub(super) fn run(func: &mut Function, pinned: &[bool]) -> bool {
    let mut live = vec![false; func.types.len()];
    for (block, data) in func.blocks.iter().enumerate() {
        if pinned[block] {
            data.params.iter().for_each(|param| live[param.index()] = true);
        }
        for inst in &data.insts {
            match &func.insts[inst.index()].op {
                Op::Jump(..) => {}
                Op::Brif(cond, ..) => live[cond.index()] = true,
                op => op.for_operands(|value| live[value.index()] = true),
            }
        }
    }
    let mut changed = true;
    while changed {
        changed = false;
        for block in 0..func.blocks.len() {
            for_edges(func, block, |target, args| {
                for (param, arg) in func.blocks[target.index()].params.iter().zip(args) {
                    if live[param.index()] && !live[arg.index()] {
                        live[arg.index()] = true;
                        changed = true;
                    }
                }
            });
        }
    }
    let dead = |param: &Value| !live[param.index()];
    if !func.blocks.iter().any(|block| block.params.iter().any(dead)) {
        return false;
    }
    let blocks = &func.blocks;
    let keep = |args: &mut Vec<Value>, target: crate::ir::Block| {
        let mut params = blocks[target.index()].params.iter();
        args.retain(|_| params.next().is_none_or(|param| live[param.index()]));
    };
    for block in blocks {
        let Some(&last) = block.insts.last() else { continue };
        match &mut func.insts[last.index()].op {
            Op::Jump(target, args) => keep(args, *target),
            Op::Brif(_, then, then_args, otherwise, else_args) => {
                keep(then_args, *then);
                keep(else_args, *otherwise);
            }
            _ => {}
        }
    }
    for block in &mut func.blocks {
        block.params.retain(|param| live[param.index()]);
    }
    true
}

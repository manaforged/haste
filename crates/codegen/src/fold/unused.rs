use crate::ir::{AtomicOrd, Block, Function, Op, Value};

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

fn removable_result(func: &Function, zeros: &[bool], data: &crate::ir::InstData) -> Option<Value> {
    if (data.op.is_removable() || matches!(data.op, Op::AtomicLoad(_, AtomicOrd::Relaxed)))
        && let Some(res) = data.result
    {
        return Some(res);
    }
    (!zeros.is_empty()).then(|| func.pure_thread_cell_result(&data.op, |v| zeros[v.index()])).flatten()
}

/// Drops pure instructions whose results nothing reads, in one backward pass that also frees the operands they read.
pub(super) fn dead(func: &mut Function) -> bool {
    let mut uses = func.use_counts();
    let zeros = func.thread_cell_zeros();
    let mut keep = Vec::new();
    let mut dropped = false;
    for b in (0..func.blocks.len()).rev() {
        let len = func.blocks[b].insts.len();
        keep.clear();
        keep.resize(len, true);
        for at in (0..len).rev() {
            let data = &func.insts[func.blocks[b].insts[at].index()];
            let dead = removable_result(func, &zeros, data).is_some_and(|result| uses[result.index()] == 0);
            if dead {
                keep[at] = false;
                dropped = true;
                data.op.for_operands(|value| uses[value.index()] -= 1);
            }
        }
        let mut at = 0;
        func.blocks[b].insts.retain(|_| {
            at += 1;
            keep[at - 1]
        });
    }
    dropped
}

/// Drops each block parameter that nothing side-effecting or control-flow reads (including dead pure phi cycles), with its argument on every edge; parameters of `pinned` blocks stay.
pub(super) fn run(func: &mut Function, pinned: &[bool]) -> bool {
    if !func.blocks.iter().enumerate().any(|(b, d)| !pinned[b] && !d.params.is_empty()) {
        return false;
    }
    let nvals = func.types.len();
    let zeros = func.thread_cell_zeros();
    let mut def_inst: Vec<Option<crate::ir::Inst>> = vec![None; nvals];
    let mut param_id: Vec<u32> = vec![u32::MAX; nvals];
    let mut live = vec![false; nvals];
    let mut work: Vec<Value> = Vec::new();
    let mut num_params = 0_u32;
    for (block, data) in func.blocks.iter().enumerate() {
        if pinned[block] {
            for &param in &data.params {
                if !std::mem::replace(&mut live[param.index()], true) {
                    work.push(param);
                }
            }
        } else {
            for &param in &data.params {
                param_id[param.index()] = num_params;
                num_params += 1;
            }
        }
        for &inst in &data.insts {
            let idata = &func.insts[inst.index()];
            if let Some(res) = removable_result(func, &zeros, idata) {
                def_inst[res.index()] = Some(inst);
                continue;
            }
            match &idata.op {
                Op::Jump(..) => {}
                Op::Brif(cond, ..) => {
                    if !std::mem::replace(&mut live[cond.index()], true) {
                        work.push(*cond);
                    }
                }
                op => op.for_operands(|value| {
                    if !std::mem::replace(&mut live[value.index()], true) {
                        work.push(value);
                    }
                }),
            }
        }
    }
    let mut incoming: Vec<Vec<Value>> = vec![Vec::new(); num_params as usize];
    for block in 0..func.blocks.len() {
        for_edges(func, block, |target, args| {
            for (param, &arg) in func.blocks[target.index()].params.iter().zip(args) {
                let id = param_id[param.index()];
                if id != u32::MAX {
                    incoming[id as usize].push(arg);
                }
            }
        });
    }
    while let Some(v) = work.pop() {
        if let Some(inst) = def_inst[v.index()] {
            func.insts[inst.index()].op.for_operands(|opnd| {
                if !std::mem::replace(&mut live[opnd.index()], true) {
                    work.push(opnd);
                }
            });
        } else {
            let id = param_id[v.index()];
            if id != u32::MAX {
                for &arg in &incoming[id as usize] {
                    if !std::mem::replace(&mut live[arg.index()], true) {
                        work.push(arg);
                    }
                }
            }
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

pub(super) fn pinned(func: &Function, entry: Block) -> Vec<bool> {
    let mut pinned = vec![false; func.blocks.len()];
    pinned[entry.index()] = true;
    for block in 0..func.blocks.len() as u32 {
        match func.last(Block(block)) {
            Some(Op::TryCall { normal, handler, .. }) => (pinned[normal.index()], pinned[handler.index()]) = (true, true),
            Some(Op::BrTable(_, d, t)) => {
                pinned[d.index()] = true;
                t.iter().for_each(|x| pinned[x.index()] = true);
            }
            _ => {}
        }
    }
    pinned
}

pub(super) fn drop_params(func: &mut Function, subst: &[Value]) -> bool {
    let gone = |param: &Value| subst[param.index()] != *param;
    if !func.blocks.iter().any(|block| block.params.iter().any(gone)) {
        return false;
    }
    let blocks = &func.blocks;
    let keep = |args: &mut Vec<Value>, target: Block| {
        let mut i = 0;
        let p = &blocks[target.index()].params;
        args.retain(|_| {
            i += 1;
            !p.get(i - 1).is_some_and(gone)
        });
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
        block.params.retain(|param| !gone(param));
    }
    true
}

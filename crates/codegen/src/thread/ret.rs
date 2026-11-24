use crate::ir::{Block, Function, Op, Value};

/// A return block given parameters: the values it returned and the parameters that now carry them.
pub(super) struct Opened {
    block: Block,
    outside: Vec<Value>,
    fresh: Vec<Value>,
}

/// Gives each return-only block that returns a dispatching block's parameters its own parameters, so threading carries the returned values along each edge into it.
pub(super) fn open_returns(func: &mut Function, pinned: &[bool]) -> Vec<Opened> {
    let cfg = func.cfg();
    let mut owner = vec![None; func.types.len()];
    for (index, data) in func.blocks.iter().enumerate() {
        for &param in &data.params {
            owner[param.index()] = Some(Block(index as u32));
        }
    }
    let mut opened = Vec::new();
    for index in 0..func.blocks.len() {
        let ret = Block(index as u32);
        let [inst] = func.blocks[index].insts[..] else { continue };
        let Op::Return(ref vals) = func.insts[inst.index()].op else { continue };
        let mut outside: Vec<Value> = Vec::new();
        for &value in vals {
            if owner[value.index()] != Some(ret) && !outside.contains(&value) {
                outside.push(value);
            }
        }
        let fed = outside.iter().any(|value| owner[value.index()].is_some_and(|block| super::dispatches(func, pinned, block)));
        let mut preds = cfg.preds_of(ret).to_vec();
        preds.sort_unstable();
        preds.dedup();
        if pinned[index] || !fed || preds.is_empty() || preds.iter().any(|&pred| !matches!(func.last(pred), Some(Op::Jump(..) | Op::Brif(..)))) {
            continue;
        }
        let fresh: Vec<Value> = outside.iter().map(|&value| func.value(func.ty(value))).collect();
        func.blocks[index].params.extend(&fresh);
        if let Op::Return(vals) = &mut func.insts[inst.index()].op {
            for value in vals.iter_mut() {
                if let Some(at) = outside.iter().position(|other| other == value) {
                    *value = fresh[at];
                }
            }
        }
        for pred in preds {
            func.for_edge_args_mut(pred, ret, |args| args.extend(&outside));
        }
        opened.push(Opened { block: ret, outside, fresh });
    }
    if !opened.is_empty() {
        func.bump_cfg();
    }
    opened
}

/// Removes each parameter `open_returns` added that every edge still feeds its original value, so a return block threading never reached keeps its old shape.
pub(super) fn close_returns(func: &mut Function, opened: &[Opened]) -> bool {
    let cfg = func.cfg();
    let mut changed = false;
    for open in opened {
        let index = open.block.index();
        let [inst] = func.blocks[index].insts[..] else { continue };
        let mut preds = cfg.preds_of(open.block).to_vec();
        preds.sort_unstable();
        preds.dedup();
        let base = func.blocks[index].params.len() - open.fresh.len();
        for (at, (&fresh, &outside)) in open.fresh.iter().zip(&open.outside).enumerate().rev() {
            let slot = base + at;
            if func.blocks[index].params.get(slot) != Some(&fresh) || !preds.iter().all(|&pred| edge_args(func, pred, open.block).iter().all(|args| args.get(slot) == Some(&outside))) {
                continue;
            }
            func.blocks[index].params.remove(slot);
            func.insts[inst.index()].op.map_operands(|value| if value == fresh { outside } else { value });
            for &pred in &preds {
                func.for_edge_args_mut(pred, open.block, |args| {
                    args.remove(slot);
                });
            }
            changed = true;
        }
    }
    if changed {
        func.bump_cfg();
    }
    changed
}

fn edge_args(func: &Function, pred: Block, target: Block) -> Vec<&[Value]> {
    match func.last(pred) {
        Some(Op::Jump(to, args)) if *to == target => vec![args.as_slice()],
        Some(Op::Brif(_, then, then_args, otherwise, otherwise_args)) => {
            let mut out = Vec::new();
            if *then == target {
                out.push(then_args.as_slice());
            }
            if *otherwise == target {
                out.push(otherwise_args.as_slice());
            }
            out
        }
        _ => vec![&[]],
    }
}

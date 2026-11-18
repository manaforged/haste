use crate::cfg::Loop;
use crate::ir::{Block, Function, Op, Value};

/// Passes each register-sized value that `wanted` names for a loop header into each loop whose own blocks read it but do not define it as a header parameter, outermost loop first, so its range inside the loop is its own and can keep a register there; returns each new parameter with the value it carries and its loop header.
pub(crate) fn run(func: &mut Function, wanted: impl Fn(Block, Value) -> bool) -> Vec<(Value, Value, Block)> {
    let mut made = Vec::new();
    let cfg = func.cfg();
    let depths = &cfg.loops().depths;
    let mut loops: Vec<&Loop> = cfg.loops().list.iter().collect();
    loops.sort_by_key(|each| (each.depth, each.head.index()));
    loops.dedup_by_key(|each| each.head);
    for each in loops {
        if Some(each.head) == func.entry || !cfg.preds_of(each.head).iter().all(|&pred| matches!(func.last(pred), Some(Op::Jump(..) | Op::Brif(..)))) {
            continue;
        }
        let outside_preds: Vec<Block> = cfg.preds_of(each.head).iter().copied().filter(|pred| !inside(each, *pred)).collect();
        for value in reads(func, (each, depths), &|v| wanted(each.head, v)) {
            let param = func.value(func.ty(value));
            made.push((param, value, each.head));
            func.blocks[each.head.index()].params.push(param);
            for (at, _) in each.body.iter().enumerate().filter(|(_, is)| **is) {
                for inst in func.blocks[at].insts.clone() {
                    func.insts[inst.index()].op.map_operands(|v| if v == value { param } else { v });
                }
            }
            for &pred in &outside_preds {
                pass(func, pred, each.head, value);
            }
            for &latch in &each.latches {
                pass(func, latch, each.head, param);
            }
        }
    }
    made
}

fn inside(each: &Loop, block: Block) -> bool {
    each.body.get(block.index()).copied().unwrap_or(false)
}

/// The values `wanted` names that the loop's own blocks, outside its inner loops, read, that are defined outside it and could live in a register, in value order.
fn reads(func: &Function, (each, depths): (&Loop, &[u32]), wanted: &impl Fn(Value) -> bool) -> Vec<Value> {
    let mut defined_inside = vec![false; func.types.len()];
    let mut outside_def = vec![false; func.types.len()];
    for (at, data) in func.blocks.iter().enumerate() {
        let local = inside(each, Block(at as u32));
        for &param in &data.params {
            defined_inside[param.index()] = local;
            outside_def[param.index()] = !local;
        }
        for inst in &data.insts {
            let inst = &func.insts[inst.index()];
            if let Some(result) = inst.result {
                defined_inside[result.index()] = local;
                outside_def[result.index()] = !local && !matches!(inst.op, Op::Const(_) | Op::SlotAddr(_) | Op::Symbol(_) | Op::Runtime(_) | Op::StackAddr(..) | Op::Icmp(..) | Op::Fcmp(..) | Op::Fits(..) | Op::FramePointer);
            }
        }
    }
    let mut found: Vec<Value> = Vec::new();
    for data in func.blocks.iter().enumerate().filter(|&(at, _)| inside(each, Block(at as u32)) && depths.get(at) == Some(&each.depth)).map(|(_, data)| data) {
        for inst in &data.insts {
            func.insts[inst.index()].op.for_operands(|v| {
                let ty = func.ty(v);
                if outside_def[v.index()] && !defined_inside[v.index()] && (ty.is_small_int() || ty.is_float()) && wanted(v) {
                    found.push(v);
                }
            });
        }
    }
    found.sort_unstable();
    found.dedup();
    found
}

/// Appends `arg` to every edge from `from` to `head`.
fn pass(func: &mut Function, from: Block, head: Block, arg: Value) {
    let Some(&last) = func.blocks[from.index()].insts.last() else { return };
    match &mut func.insts[last.index()].op {
        Op::Jump(target, args) if *target == head => args.push(arg),
        Op::Brif(_, then, then_args, otherwise, else_args) => {
            if *then == head {
                then_args.push(arg);
            }
            if *otherwise == head {
                else_args.push(arg);
            }
        }
        _ => {}
    }
}

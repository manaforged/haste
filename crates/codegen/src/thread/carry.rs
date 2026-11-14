use crate::ir::{Block, Function, Op, Value};

use super::eval::{ThreadCx, lookup};
use super::{Eval, block_bit, body, emit_needed, note_use, redefines};

const MAX_CARRY: usize = 8;

#[derive(Default)]
pub(super) struct Carry {
    pub(super) values: Vec<Value>,
    blocks: Vec<Block>,
}

fn block_defs(func: &Function, block: Block) -> impl Iterator<Item = Value> + '_ {
    func.blocks[block.index()].params.iter().copied().chain(body(func, block).iter().filter_map(move |&inst| func.insts[inst.index()].result))
}

fn flood(func: &Function, start: Block, wall: Block, mut visit: impl FnMut(Block) -> bool) -> Vec<bool> {
    let mut seen = vec![false; func.blocks.len()];
    seen[wall.index()] = true;
    let mut stack = vec![start];
    while let Some(block) = stack.pop() {
        if std::mem::replace(&mut seen[block.index()], true) {
            continue;
        }
        if !visit(block) {
            break;
        }
        stack.extend(func.successors(block));
    }
    seen
}

fn below(free: &[bool], target: Block, block: Block) -> bool {
    block == target || !free[block.index()]
}

fn reads(func: &Function, uses: &[u64], start: Block, def_block: Block, every: bool, carry: &mut Carry) {
    let defs: Vec<Value> = block_defs(func, def_block).collect();
    let users = defs.iter().fold(0, |all, v| all | uses.get(v.index()).copied().unwrap_or(0));
    if users == 0 || start == def_block {
        return;
    }
    let known = carry.values.len();
    flood(func, start, def_block, |block| {
        if users & block_bit(block) != 0 {
            for &inst in &func.blocks[block.index()].insts {
                func.insts[inst.index()].op.for_operands(|v| {
                    if defs.contains(&v) {
                        if !carry.values.contains(&v) {
                            carry.values.push(v);
                        }
                        if !carry.blocks.contains(&block) {
                            carry.blocks.push(block);
                        }
                    }
                });
            }
        }
        every || carry.values.len() == known
    });
}

fn rebinds(func: &Function, chain: &[Block], incoming: &[Value]) -> bool {
    let mut read = incoming.to_vec();
    chain.iter().any(|&block| {
        let hit = redefines(func, block, &read);
        func.blocks[block.index()].insts.iter().for_each(|inst| func.insts[inst.index()].op.for_operands(|v| read.push(v)));
        hit
    })
}

pub(super) fn plan(func: &Function, cx: &ThreadCx<'_>, single: &[Option<Block>], (chain, incoming): (&[Block], &[Value]), target: Block) -> Option<Carry> {
    let last = *chain.last()?;
    let mut linked = !chain.contains(&target)
        && !cx.pinned[target.index()]
        && single.get(target.index()) == Some(&Some(last))
        && matches!(func.last(last), Some(Op::Jump(..) | Op::Brif(..)));
    let mut carry = Carry::default();
    for at in (0..chain.len()).rev() {
        let known = carry.values.len();
        reads(func, cx.uses, target, chain[at], linked, &mut carry);
        if carry.values.len() > known && !linked {
            return None;
        }
        linked &= at > 0 && single.get(chain[at].index()) == Some(&Some(chain[at - 1]));
    }
    if carry.values.is_empty() {
        return Some(carry);
    }
    if carry.values.len() > MAX_CARRY || rebinds(func, chain, incoming) {
        return None;
    }
    let free = flood(func, func.entry?, target, |_| true);
    carry.blocks.iter().all(|&block| below(&free, target, block)).then_some(carry)
}

fn apply(func: &mut Function, cx: &mut ThreadCx<'_>, single: &mut [Option<Block>], pred: Block, target: Block, carry: Carry, args: &mut [Value]) {
    let (Some(entry), Some(last)) = (func.entry, single.get(target.index()).copied().flatten()) else {
        return;
    };
    let Carry { values, mut blocks } = carry;
    let inside = below(&flood(func, entry, target, |_| true), target, pred);
    if inside && !blocks.contains(&pred) {
        blocks.push(pred);
    }
    let mut fresh: Vec<(Value, Value)> = Vec::with_capacity(values.len());
    for &value in &values {
        let param = func.value(func.ty(value));
        cx.consts.push(None);
        func.blocks[target.index()].params.push(param);
        if !block_defs(func, last).any(|def| def == value) {
            note_use(cx.uses, last, value);
        }
        fresh.push((value, param));
    }
    if let Some(&term) = func.blocks[last.index()].insts.last() {
        match &mut func.insts[term.index()].op {
            Op::Jump(_, to) => to.extend(&values),
            Op::Brif(_, then, then_args, otherwise, else_args) => {
                if *then == target {
                    then_args.extend(&values);
                }
                if *otherwise == target {
                    else_args.extend(&values);
                }
            }
            _ => {}
        }
    }
    for &block in &blocks {
        for at in 0..func.blocks[block.index()].insts.len() {
            let inst = func.blocks[block.index()].insts[at];
            func.insts[inst.index()].op.map_operands(|v| lookup(&fresh, v));
        }
        if block != target {
            fresh.iter().for_each(|&(_, param)| note_use(cx.uses, block, param));
        }
    }
    if inside {
        args.iter_mut().for_each(|arg| *arg = lookup(&fresh, *arg));
    }
    single[target.index()] = None;
}

pub(super) fn commit(func: &mut Function, cx: &mut ThreadCx<'_>, single: &mut [Option<Block>], pred: Block, mut eval: Eval) -> (Block, Vec<Value>) {
    let carry = std::mem::take(&mut eval.carry);
    let (target, mut args) = emit_needed(func, pred, (&mut *cx.consts, &mut *cx.uses), eval);
    if !carry.values.is_empty() {
        apply(func, cx, single, pred, target, carry, &mut args);
    }
    (target, args)
}

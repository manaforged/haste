//! Moves out of each loop what every trip computes alike: pure instructions that cannot trap and whose operands the loop does not change, and loads at the head of a loop that writes no memory.

use crate::dom::dominators;
use crate::ir::{Binary, Block, Function, Inst, Op, Value};
use crate::loads::clobbers;
use crate::live::results;

/// One natural loop: its head, the blocks of its body including the head, and the one block outside it that enters the head by a jump, when there is one.
struct Loop {
    head: Block,
    body: Vec<bool>,
    size: usize,
    entry: Option<Block>,
}

/// Whether `op` computes its result from its operands alone, cannot trap, and may run on a path that did not run it.
fn speculable(op: &Op) -> bool {
    match op {
        Op::Binary(op, ..) => !matches!(op, Binary::Sdiv | Binary::Udiv | Binary::Srem | Binary::Urem),
        Op::Const(_) | Op::SlotAddr(_) | Op::Symbol(_) | Op::Unary(..) | Op::Convert(..) | Op::Icmp(..) | Op::Fits(..) | Op::Fcmp(..) | Op::ByteCmp(..) | Op::Splat(_) | Op::Movemask(_) | Op::Select(..) | Op::StackAddr(..) => true,
        _ => false,
    }
}

/// Whether `ancestor` dominates `block`, by walking up `idom`.
fn dominates(idom: &[Option<Block>], ancestor: Block, mut block: Block) -> bool {
    loop {
        if block == ancestor {
            return true;
        }
        match idom[block.index()] {
            Some(parent) if parent != block => block = parent,
            _ => return false,
        }
    }
}

/// The natural loop that the back edges from `latches` into `head` close.
fn natural(func: &Function, preds: &[Vec<Block>], head: Block, latches: &[Block]) -> Loop {
    let mut body = vec![false; func.blocks.len()];
    body[head.index()] = true;
    let mut todo = latches.to_vec();
    while let Some(block) = todo.pop() {
        if !std::mem::replace(&mut body[block.index()], true) {
            todo.extend(preds[block.index()].iter().copied());
        }
    }
    let mut outside = preds[head.index()].iter().copied().filter(|pred| !body[pred.index()]);
    let first = outside.next();
    let single = first.filter(|entry| outside.all(|other| other == *entry) && matches!(func.last(*entry), Some(Op::Jump(target, _)) if *target == head));
    Loop { head, size: body.iter().filter(|inside| **inside).count(), body, entry: single }
}

/// The natural loops of `func`, innermost first.
fn loops(func: &Function, entry: Block) -> Vec<Loop> {
    let (order, preds, idom) = dominators(func, entry);
    let mut found: Vec<Loop> = order
        .iter()
        .filter_map(|&head| {
            let latches: Vec<Block> = preds[head.index()].iter().copied().filter(|pred| dominates(&idom, head, *pred)).collect();
            (!latches.is_empty()).then(|| natural(func, &preds, head, &latches))
        })
        .collect();
    found.sort_by_key(|each| each.size);
    found
}

/// For each value, the block that defines it.
fn homes(func: &Function) -> Vec<Option<Block>> {
    let mut home = vec![None; func.types.len()];
    for (index, block) in func.blocks.iter().enumerate() {
        block.params.iter().for_each(|param| home[param.index()] = Some(Block(index as u32)));
        for inst in &block.insts {
            results(&func.insts[inst.index()]).for_each(|value| home[value.index()] = Some(Block(index as u32)));
        }
    }
    home
}

/// The number of natural loops each block lies in.
pub(crate) fn depths(func: &Function) -> Vec<u32> {
    let mut depth = vec![0; func.blocks.len()];
    if let Some(entry) = func.entry {
        for each in loops(func, entry) {
            depth.iter_mut().zip(&each.body).filter(|(_, inside)| **inside).for_each(|(depth, _)| *depth += 1);
        }
    }
    depth
}

/// Moves into the loop's entry block, before its jump, each instruction of the loop that may move, in order, and records their new home.
fn hoist(func: &mut Function, each: &Loop, entry: Block, home: &mut [Option<Block>]) {
    let writes = (0..func.blocks.len()).filter(|block| each.body[*block]).flat_map(|block| &func.blocks[block].insts).any(|inst| {
        let op = &func.insts[inst.index()].op;
        clobbers(op) || matches!(op, Op::Store(..) | Op::StackStore(..))
    });
    let calls = (0..func.blocks.len()).filter(|block| each.body[*block]).flat_map(|block| &func.blocks[block].insts).any(|inst| matches!(func.insts[inst.index()].op, Op::Call { .. } | Op::TryCall { .. }));
    if calls {
        return;
    }
    let mut moved: Vec<Inst> = Vec::new();
    for block in (0..func.blocks.len()).filter(|block| each.body[*block]) {
        let head = block == each.head.index();
        let insts = std::mem::take(&mut func.blocks[block].insts);
        let mut kept = Vec::with_capacity(insts.len());
        for inst in insts {
            let data = &func.insts[inst.index()];
            let movable = speculable(&data.op) || (head && !writes && matches!(data.op, Op::Load(..)));
            let invariant = |value: Value| home[value.index()].is_none_or(|at| !each.body[at.index()]);
            let mut operands_invariant = true;
            data.op.for_operands(|value| operands_invariant &= invariant(value));
            if movable && operands_invariant && data.result.is_some() {
                data.result.iter().for_each(|result| home[result.index()] = Some(entry));
                moved.push(inst);
            } else {
                kept.push(inst);
            }
        }
        func.blocks[block].insts = kept;
    }
    let insts = &mut func.blocks[entry.index()].insts;
    let at = insts.len().saturating_sub(1);
    insts.splice(at..at, moved);
}

/// Moves loop-invariant work of every natural loop that one jump enters out to the block that enters it.
pub(crate) fn run(func: &mut Function) {
    let Some(entry) = func.entry else { return };
    if func.coroutine.is_some() {
        return;
    }
    let found = loops(func, entry);
    if found.is_empty() {
        return;
    }
    let mut home = homes(func);
    for each in &found {
        if let Some(entry) = each.entry {
            hoist(func, each, entry, &mut home);
        }
    }
}

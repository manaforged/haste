use crate::builder::FunctionBuilder;
use crate::ir::{Block, Function, IntCC, Op, SlotData, StackSlot, Type};
use crate::reloc::RuntimeSymbol;

use super::{Coroutine, Frame, Home, call, push};

/// The largest frame a first poll keeps on the native stack.
const MAX_STACK_FRAME: u32 = 4096;

/// A frame that starts in a native stack buffer and moves to the heap at the first suspension.
#[derive(Clone, Copy)]
pub(super) struct Lazy {
    pub(super) buf: StackSlot,
    pub(super) bytes: u32,
}

/// Adds the stack buffer when no framed slot has its address taken, so a byte copy can move the frame; the anchor slot grows to hold the object address at offset 8.
pub(super) fn plan(func: &Function, frame: &Frame, stack: &mut Vec<SlotData>) -> Option<Lazy> {
    let bytes = frame.bytes();
    if bytes > MAX_STACK_FRAME {
        return None;
    }
    let addressed = func.blocks.iter().flat_map(|block| &block.insts).any(|inst| {
        matches!(func.insts[inst.index()].op, Op::StackAddr(slot, _) if matches!(frame.slots[slot.index()], Home::Frame(_)))
    });
    if addressed {
        return None;
    }
    stack[0].size = 16;
    stack.push(SlotData { size: bytes, align_shift: 4 });
    Some(Lazy { buf: StackSlot(stack.len() as u32 - 1), bytes })
}

/// Before each return, moves a frame still in the stack buffer to the heap when the coroutine suspends, and frees a heap frame when it ends.
pub(super) fn settle(func: &mut Function, spec: &Coroutine, anchor: StackSlot, lazy: Lazy, blocks: usize) {
    for idx in 0..blocks {
        let block = Block(idx as u32);
        let done = spec.done.contains(&block);
        let Some(&last) = func.blocks[idx].insts.last() else { continue };
        let op = &func.insts[last.index()].op;
        if !(op.is_terminator() && (done || matches!(op, Op::Return(_)))) {
            continue;
        }
        let term = func.blocks[idx].insts.pop().expect("a block with a terminator");
        let mut b = FunctionBuilder::new(func);
        let (cold, tail) = (b.create_block(), b.create_block());
        b.switch_to_block(block);
        let fp = b.ins().stack_load(Type::I64, anchor, 0);
        let buf = b.ins().stack_addr(Type::I64, lazy.buf, 0);
        let on_stack = b.ins().icmp(IntCC::Equal, fp, buf);
        if done {
            b.ins().brif(on_stack, tail, &[], cold, &[]);
        } else {
            b.ins().brif(on_stack, cold, &[], tail, &[]);
        }
        let mut insts = Vec::new();
        let size = push(b.func, &mut insts, Op::Const(i64::from(lazy.bytes)), Some(Type::I64)).expect("a constant has a result");
        if done {
            call(b.func, &mut insts, RuntimeSymbol::Free, &[fp, size], false);
        } else {
            let heap = call(b.func, &mut insts, RuntimeSymbol::Malloc, &[size], true).expect("malloc returns");
            push(b.func, &mut insts, Op::Move(heap, buf, lazy.bytes), None);
            let object = push(b.func, &mut insts, Op::StackLoad(anchor, 8), Some(Type::I64)).expect("a load has a result");
            push(b.func, &mut insts, Op::Store(heap, object, spec.header.1), None);
        }
        b.func.blocks[cold.index()].insts = insts;
        b.switch_to_block(cold);
        b.ins().jump(tail, &[]);
        b.func.blocks[tail.index()].insts.push(term);
    }
}

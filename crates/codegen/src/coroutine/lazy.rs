use crate::builder::FunctionBuilder;
use crate::ir::{Binary, Block, Function, IntCC, Op, SlotData, StackSlot, Type, Value};
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

/// Adds the stack buffer when no address into the frame leaves the function or crosses a suspension, so a byte copy can move the frame; the anchor slot grows to hold the object address at offset 8.
pub(super) fn plan(func: &Function, frame: &Frame, is_spill: &[bool], stack: &mut Vec<SlotData>) -> Option<Lazy> {
    let bytes = frame.bytes();
    if bytes > MAX_STACK_FRAME {
        return None;
    }
    let framed = |slot: StackSlot| matches!(frame.slots[slot.index()], Home::Frame(_));
    let addressed = func.blocks.iter().flat_map(|block| &block.insts).any(|inst| matches!(func.insts[inst.index()].op, Op::StackAddr(slot, _) if framed(slot)));
    if addressed && frame_escapes(func, &framed, is_spill) {
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
        b.func.mark_frame_ptr(fp);
        b.func.mark_frame_ptr(buf);
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
            b.func.mark_frame_ptr(heap);
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

/// Whether an address into a framed slot, followed through local slots that hold it, reaches anything but a load, a store address, a copy or a local slot, or lives across a suspension.
fn frame_escapes(func: &Function, framed: &dyn Fn(StackSlot) -> bool, is_spill: &[bool]) -> bool {
    let (addrs, _) = func.slot_offsets(|_| None);
    let slot_of = |v: Value| addrs.get(v.index()).copied().flatten().map(|(slot, _)| slot);
    let mut tainted: Vec<bool> = (0..func.types.len()).map(|v| slot_of(Value(v as u32)).is_some_and(framed)).collect();
    let mut holders = vec![false; func.slots.len()];
    let insts: Vec<_> = func.blocks.iter().flat_map(|block| &block.insts).map(|inst| &func.insts[inst.index()]).collect();
    let mut changed = true;
    while changed {
        changed = false;
        for data in &insts {
            let held = match data.op {
                Op::StackStore(value, slot, _) if tainted[value.index()] => Some(slot),
                Op::Store(value, addr, _) if tainted[value.index()] => slot_of(addr),
                _ => None,
            };
            if let Some(slot) = held.filter(|&slot| !framed(slot) && !holders[slot.index()]) {
                holders[slot.index()] = true;
                changed = true;
            }
            let from = match data.op {
                Op::StackLoad(slot, _) => holders[slot.index()],
                Op::Load(addr, _) => slot_of(addr).is_some_and(|slot| holders[slot.index()]),
                Op::Binary(Binary::Iadd | Binary::Isub, a, _) => tainted[a.index()],
                _ => false,
            };
            if let Some(result) = data.result.filter(|r| from && !tainted[r.index()]) {
                tainted[result.index()] = true;
                changed = true;
            }
        }
    }
    let local = |v: Value| slot_of(v).is_some_and(|slot| !framed(slot));
    insts.iter().any(|data| {
        let mut bad = false;
        data.op.for_operands(|v| {
            let holder_addr = slot_of(v).is_some_and(|slot| holders[slot.index()]);
            if !tainted[v.index()] && !holder_addr {
                return;
            }
            bad |= match &data.op {
                Op::Load(..) => false,
                Op::Move(..) => holder_addr,
                Op::Copy(_, _, len) => holder_addr || *len == v,
                Op::Store(value, addr, _) => *value == v && (holder_addr || !local(*addr)),
                Op::StackStore(_, slot, _) => holder_addr || framed(*slot),
                Op::Binary(Binary::Iadd | Binary::Isub, a, b) => *a != v || tainted[b.index()] || (holder_addr && data.result.and_then(|r| slot_of(r)).is_none()),
                _ => true,
            };
        });
        bad || data.result.is_some_and(|r| tainted[r.index()] && is_spill.get(r.index()).copied().unwrap_or(false))
    })
}

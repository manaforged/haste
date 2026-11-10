//! Frame layout shared by the encoders: where arguments travel, and frame offsets from the frame pointer for values, stack slots, edge scratch and outgoing arguments.

use crate::alloc::Location;
use crate::ir::{Function, Op, Signature, Type};

const WORD: u32 = 8;
const MAX_ALIGN_SHIFT: u8 = 4;

/// Where one argument or return value travels.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Loc<R> {
    Int(R),
    Pair(R, R),
    Float(u8),
    Stack(u32),
    Struct(u32, u32),
}

/// Where the arguments and return values of one signature travel, and the bytes of stack its arguments take.
pub(crate) struct Abi<R> {
    pub params: Vec<Loc<R>>,
    pub returns: Vec<Loc<R>>,
    pub stack: u32,
}

pub(crate) fn round(value: u32, to: u32) -> u32 {
    value.div_ceil(to) * to
}

/// Frame offsets from the frame pointer for every value, stack slot and the edge scratch area.
pub(crate) struct Frame<R> {
    pub values: Vec<i32>,
    pub slots: Vec<i32>,
    pub scratch: i32,
    pub size: u32,
    pub saves: Vec<R>,
}

/// The operations of the instructions in blocks.
fn live_ops(func: &Function) -> impl Iterator<Item = &Op> {
    func.blocks.iter().flat_map(|block| &block.insts).map(|inst| &func.insts[inst.index()].op)
}

/// The registers a frame saves: all of `all` for a function with a handler, which restores what the frames it unwinds clobbered, else the pool registers it uses.
pub(crate) fn saved<R: Copy>(func: &Function, used: &[R], all: &[R]) -> Vec<R> {
    let handles = live_ops(func).any(|op| matches!(op, Op::TryCall { .. }));
    if handles { all.to_vec() } else { used.to_vec() }
}

fn needs<R>(func: &Function, classify: impl Fn(&Signature) -> Result<Abi<R>, String>) -> Result<(u32, u32), String> {
    let (mut outgoing, mut scratch) = (0, 0);
    for op in live_ops(func) {
        match op {
            Op::Call { sig, .. } | Op::TryCall { sig, .. } => outgoing = outgoing.max(classify(&func.sigs[sig.index()])?.stack),
            Op::Jump(_, args) => scratch = scratch.max(args.len() as u32 * 16),
            Op::Brif(_, _, then, _, otherwise) => scratch = scratch.max(then.len().max(otherwise.len()) as u32 * 16),
            _ => {}
        }
    }
    Ok((outgoing, scratch))
}

/// The values the blocks define or read, and the stack slots they address.
fn present(func: &Function) -> (Vec<bool>, Vec<bool>) {
    let (mut values, mut slots) = (vec![false; func.types.len()], vec![false; func.slots.len()]);
    for block in &func.blocks {
        block.params.iter().for_each(|param| values[param.index()] = true);
        for inst in &block.insts {
            let data = &func.insts[inst.index()];
            data.op.for_operands(|value| values[value.index()] = true);
            crate::live::results(data).for_each(|value| values[value.index()] = true);
            if let Op::StackAddr(slot, _) | Op::StackLoad(slot, _) | Op::StackStore(_, slot, _) = data.op {
                slots[slot.index()] = true;
            }
        }
    }
    (values, slots)
}

/// Lays out the frame below the frame pointer: saved registers, the values that live in memory, the stack slots in use, edge scratch, then outgoing arguments at the stack pointer.
pub(crate) fn layout<R>(func: &Function, saves: Vec<R>, locations: &[Location<R>], classify: impl Fn(&Signature) -> Result<Abi<R>, String>) -> Result<Frame<R>, String> {
    let (outgoing, scratch) = needs(func, classify)?;
    let (present, addressed) = present(func);
    let mut cursor = saves.len() as u32 * WORD;
    let mut values = Vec::with_capacity(func.types.len());
    for (value, ty) in func.types.iter().enumerate() {
        if !present[value] || matches!(locations[value], Location::Reg(_)) {
            values.push(0);
            continue;
        }
        cursor += if *ty == Type::I128 { 16 } else { WORD };
        values.push(-(cursor as i32));
    }
    let mut slots = Vec::with_capacity(func.slots.len());
    for (slot, data) in func.slots.iter().enumerate() {
        if !addressed[slot] {
            slots.push(0);
            continue;
        }
        let align = 1 << data.align_shift.min(MAX_ALIGN_SHIFT);
        cursor = round(cursor + data.size, align);
        slots.push(-(cursor as i32));
    }
    cursor = round(cursor + scratch, 16);
    let scratch = -(cursor as i32);
    let size = round(cursor + outgoing, 16);
    Ok(Frame { values, slots, scratch, size, saves })
}

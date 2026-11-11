//! Frame layout shared by the encoders: where arguments travel, and frame offsets from the frame pointer for values, stack slots, edge scratch and outgoing arguments.

use crate::alloc::Location;
use crate::ir::{Function, Op, Signature, Type, Value};

const WORD: u32 = 8;
const MAX_ALIGN_SHIFT: u8 = 4;
/// The bytes of a page of stack, which a larger frame touches one at a time.
pub(crate) const PAGE: u32 = 4096;
/// The offset from the frame pointer of the first argument that arrived on the stack.
pub(crate) const INCOMING: i32 = 16;
/// The bytes of scratch that one move of a block edge takes.
const EDGE_SLOT: u32 = 16;

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

/// Places the return values of `signature`: integers in `ints`, an i128 in both of them, and floats in the first `floats` float registers.
pub(crate) fn returns<R: Copy>(signature: &Signature, ints: [R; 2], floats: u8) -> Result<Vec<Loc<R>>, String> {
    let (mut int, mut float) = (0, 0);
    let place = |ty: Type| match ty {
        Type::F32 | Type::F64 if float < floats => {
            float += 1;
            Ok(Loc::Float(float - 1))
        }
        Type::I128 if int == 0 => {
            int = ints.len();
            Ok(Loc::Pair(ints[0], ints[1]))
        }
        Type::I8 | Type::I16 | Type::I32 | Type::I64 if int < ints.len() => {
            int += 1;
            Ok(Loc::Int(ints[int - 1]))
        }
        _ => Err("too many return values".to_owned()),
    };
    signature.returns.iter().map(|param| param.value_type).map(place).collect()
}

/// The argument and parameter of each move an edge into a block with `params` makes, and whether one move writes the place another reads, when `same_home` tells that two values share a place.
pub(crate) fn edge_moves(args: &[Value], params: &[Value], same_home: impl Fn(Value, Value) -> bool) -> (Vec<(Value, Value)>, bool) {
    let pairs: Vec<(Value, Value)> = args.iter().copied().zip(params.iter().copied()).filter(|(arg, param)| !same_home(*arg, *param)).collect();
    let clash = pairs.iter().any(|(arg, _)| pairs.iter().any(|(_, param)| same_home(*arg, *param)));
    (pairs, clash)
}

/// Frame offsets from the frame pointer for every value, stack slot and the edge scratch area.
pub(crate) struct Frame<R> {
    pub values: Vec<i32>,
    pub slots: Vec<i32>,
    pub scratch: i32,
    pub size: u32,
    pub saves: Vec<R>,
}

impl<R> Frame<R> {
    /// The offset from the frame pointer of the scratch that move `index` of a block edge uses.
    pub(crate) fn edge_slot(&self, index: usize) -> i32 {
        self.scratch + (EDGE_SLOT as usize * index) as i32
    }
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
            Op::Jump(_, args) => scratch = scratch.max(args.len() as u32 * EDGE_SLOT),
            Op::Brif(_, _, then, _, otherwise) => scratch = scratch.max(then.len().max(otherwise.len()) as u32 * EDGE_SLOT),
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
        if !present[value] || matches!(locations[value], Location::Reg(_) | Location::Float(_)) {
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

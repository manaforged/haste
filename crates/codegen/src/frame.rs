//! Frame layout shared by the encoders: where arguments travel, and frame offsets from the frame pointer for values, stack slots, edge scratch and outgoing arguments.

use crate::alloc::{Location, Plan};
use crate::ir::{Function, Op, Signature, Type, Value};

/// The bytes of a machine word.
pub(crate) const WORD: u32 = 8;
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

/// One step of the moves an edge into a block makes, in an order that reads every place before any step writes it.
#[derive(Clone, Copy)]
pub(crate) enum Step {
    /// Writes the argument into the place of the parameter.
    Move(Value, Value),
    /// Saves the argument into edge scratch `index`, which breaks a cycle of moves.
    Save(Value, usize),
    /// Loads the parameter from edge scratch `index`.
    Restore(usize, Value),
}

/// Where a pending edge move reads from: an argument's place, or edge scratch.
#[derive(Clone, Copy)]
enum Source {
    Value(Value),
    Scratch(usize),
}

/// The steps of the moves an edge into a block with `params` makes, when `same_home` tells that two values share a place: each move runs once no other pending move still reads its target, and a cycle goes through edge scratch.
pub(crate) fn edge_steps(args: &[Value], params: &[Value], same_home: impl Fn(Value, Value) -> bool) -> Vec<Step> {
    let mut pending: Vec<(Source, Value)> = args.iter().zip(params).filter(|(arg, param)| !same_home(**arg, **param)).map(|(arg, param)| (Source::Value(*arg), *param)).collect();
    let mut steps = Vec::with_capacity(pending.len());
    let mut saved = 0;
    while !pending.is_empty() {
        let read = |at: usize| pending.iter().enumerate().any(|(other, (source, _))| other != at && matches!(source, Source::Value(arg) if same_home(*arg, pending[at].1)));
        if let Some(at) = (0..pending.len()).find(|at| !read(*at)) {
            steps.push(match pending.remove(at) {
                (Source::Value(arg), param) => Step::Move(arg, param),
                (Source::Scratch(index), param) => Step::Restore(index, param),
            });
            continue;
        }
        let Some((at, arg)) = pending.iter().enumerate().find_map(|(at, (source, _))| if let Source::Value(arg) = source { Some((at, *arg)) } else { None }) else { break };
        steps.push(Step::Save(arg, saved));
        pending[at].0 = Source::Scratch(saved);
        saved += 1;
    }
    steps
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

    /// Whether `a` and `b` both live in memory at one shared home, so the bytes of one are the bytes of the other.
    pub(crate) fn shares(&self, plan: &Plan<R>, a: Value, b: Value) -> bool
    where
        R: Copy,
    {
        let memory = |value: Value| matches!(plan.location(value), Location::Slot);
        memory(a) && memory(b) && self.values[a.index()] == self.values[b.index()]
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

/// Whether a function needs a frame of its own: one that saves registers, keeps anything in memory, calls, reads its frame pointer or takes arguments on the stack.
pub(crate) fn framed<R>(func: &Function, frame: &Frame<R>, stack_params: bool) -> bool {
    stack_params || frame.size > 0 || !frame.saves.is_empty() || live_ops(func).any(|op| matches!(op, Op::Call { .. } | Op::TryCall { .. } | Op::FramePointer))
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

/// Lays out the frame below the frame pointer: saved registers, the values that live in memory (one place per shared home) or are `homed` there around calls, the stack slots in use, edge scratch, then outgoing arguments at the stack pointer.
pub(crate) fn layout<R>(func: &Function, saves: Vec<R>, (locations, shared, homed): (&[Location<R>], &[Option<u32>], &[bool]), classify: impl Fn(&Signature) -> Result<Abi<R>, String>) -> Result<Frame<R>, String> {
    let (outgoing, scratch) = needs(func, classify)?;
    let (present, addressed) = present(func);
    let mut cursor = saves.len() as u32 * WORD;
    let mut values = Vec::with_capacity(func.types.len());
    let mut homes: Vec<i32> = Vec::new();
    for (value, ty) in func.types.iter().enumerate() {
        if !present[value] || (matches!(locations[value], Location::Reg(_) | Location::Float(_)) && homed.get(value) != Some(&true)) {
            values.push(0);
            continue;
        }
        let Some(home) = shared.get(value).copied().flatten().map(|home| home as usize) else {
            cursor += round(ty.bytes(), WORD);
            values.push(-(cursor as i32));
            continue;
        };
        if homes.len() <= home {
            homes.resize(home + 1, 0);
        }
        if homes[home] == 0 {
            cursor += round(ty.bytes(), WORD);
            homes[home] = -(cursor as i32);
        }
        values.push(homes[home]);
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

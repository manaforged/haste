use super::asm::Reg;
use super::location::Location;
use crate::ir::{ArgumentPurpose, Function, Op, Signature, Type};

const INT_ARGS: [Reg; 6] = [Reg::Rdi, Reg::Rsi, Reg::Rdx, Reg::Rcx, Reg::R8, Reg::R9];
const FLOAT_ARGS: u8 = 8;
const INT_RETURNS: [Reg; 2] = [Reg::Rax, Reg::Rdx];
const FLOAT_RETURNS: u8 = 2;
const WORD: u32 = 8;
const MAX_ALIGN_SHIFT: u8 = 4;

/// The registers every function keeps for its caller, with rbp kept by the frame chain.
pub(super) const SAVED: [Reg; 5] = [Reg::Rbx, Reg::R12, Reg::R13, Reg::R14, Reg::R15];

/// Where one argument or return value travels.
#[derive(Clone, Copy, Debug)]
pub(super) enum Loc {
    Int(Reg),
    Pair(Reg, Reg),
    Float(u8),
    Stack(u32),
    Struct(u32, u32),
}

pub(super) struct Abi {
    pub params: Vec<Loc>,
    pub returns: Vec<Loc>,
    pub stack: u32,
}

fn round(value: u32, to: u32) -> u32 {
    value.div_ceil(to) * to
}

struct Counter {
    int: usize,
    float: u8,
    stack: u32,
}

impl Counter {
    fn stack(&mut self, bytes: u32, align: u32) -> Loc {
        self.stack = round(self.stack, align);
        let at = self.stack;
        self.stack += bytes;
        Loc::Stack(at)
    }

    fn param(&mut self, purpose: ArgumentPurpose, ty: Type) -> Loc {
        if let ArgumentPurpose::StructArgument(size) = purpose {
            let at = self.stack;
            self.stack += round(size, WORD);
            return Loc::Struct(at, size);
        }
        match ty {
            Type::I128 if self.int + 2 <= INT_ARGS.len() => {
                self.int += 2;
                Loc::Pair(INT_ARGS[self.int - 2], INT_ARGS[self.int - 1])
            }
            Type::I128 => self.stack(16, 16),
            Type::F32 | Type::F64 if self.float < FLOAT_ARGS => {
                self.float += 1;
                Loc::Float(self.float - 1)
            }
            Type::F32 | Type::F64 => self.stack(WORD, WORD),
            _ if self.int < INT_ARGS.len() => {
                self.int += 1;
                Loc::Int(INT_ARGS[self.int - 1])
            }
            _ => self.stack(WORD, WORD),
        }
    }

    fn ret(&mut self, ty: Type) -> Result<Loc, String> {
        match ty {
            Type::F32 | Type::F64 if self.float < FLOAT_RETURNS => {
                self.float += 1;
                Ok(Loc::Float(self.float - 1))
            }
            Type::I128 if self.int == 0 => {
                self.int = INT_RETURNS.len();
                Ok(Loc::Pair(INT_RETURNS[0], INT_RETURNS[1]))
            }
            Type::I8 | Type::I16 | Type::I32 | Type::I64 if self.int < INT_RETURNS.len() => {
                self.int += 1;
                Ok(Loc::Int(INT_RETURNS[self.int - 1]))
            }
            _ => Err("too many return values".to_owned()),
        }
    }
}

/// Places the parameters and return values of `signature` by the System V rules.
pub(super) fn classify(signature: &Signature) -> Result<Abi, String> {
    let mut params = Counter { int: 0, float: 0, stack: 0 };
    let placed = signature.params.iter().map(|param| params.param(param.purpose, param.value_type)).collect();
    let mut returns = Counter { int: 0, float: 0, stack: 0 };
    let mut rets = Vec::with_capacity(signature.returns.len());
    for param in &signature.returns {
        rets.push(returns.ret(param.value_type)?);
    }
    Ok(Abi { params: placed, returns: rets, stack: round(params.stack, 16) })
}

/// Frame offsets from rbp for every value, stack slot and the edge scratch area.
pub(super) struct Frame {
    pub values: Vec<i32>,
    pub slots: Vec<i32>,
    pub scratch: i32,
    pub size: u32,
    pub saves: Vec<Reg>,
}

/// The operations of the instructions in blocks.
fn live_ops(func: &Function) -> impl Iterator<Item = &Op> {
    func.blocks.iter().flat_map(|block| &block.insts).map(|inst| &func.insts[inst.index()].op)
}

/// The registers a frame saves: all of `SAVED` for a function with a handler, which restores what the frames it unwinds clobbered, else the pool registers it uses.
pub(super) fn saved(func: &Function, used: &[Reg]) -> Vec<Reg> {
    let handles = live_ops(func).any(|op| matches!(op, Op::TryCall { .. }));
    if handles { SAVED.to_vec() } else { used.to_vec() }
}

fn needs(func: &Function) -> Result<(u32, u32), String> {
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
            super::location::results(data).for_each(|value| values[value.index()] = true);
            if let Op::StackAddr(slot, _) | Op::StackLoad(slot, _) | Op::StackStore(_, slot, _) = data.op {
                slots[slot.index()] = true;
            }
        }
    }
    (values, slots)
}

/// Lays out the frame: saved registers, the values that live in memory, the stack slots in use, edge scratch, then outgoing arguments at rsp.
pub(super) fn layout(func: &Function, saves: Vec<Reg>, locations: &[Location]) -> Result<Frame, String> {
    let (outgoing, scratch) = needs(func)?;
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

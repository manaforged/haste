use super::asm::Reg;
use crate::frame::{WORD, returns, round};
use crate::ir::{ArgumentPurpose, CallConv, Signature, Type};
use crate::target::CallAbi;

const INT_ARGS: [Reg; 6] = [Reg::Rdi, Reg::Rsi, Reg::Rdx, Reg::Rcx, Reg::R8, Reg::R9];
const WIN64_INT_ARGS: [Reg; 4] = [Reg::Rcx, Reg::Rdx, Reg::R8, Reg::R9];
const FLOAT_ARGS: u8 = 8;
const INT_RETURNS: [Reg; 2] = [Reg::Rax, Reg::Rdx];
const FLOAT_RETURNS: u8 = 2;

pub(super) const SAVED: [Reg; 5] = [Reg::Rbx, Reg::R12, Reg::R13, Reg::R14, Reg::R15];

pub(super) type Loc = crate::frame::Loc<Reg>;
pub(super) type Abi = crate::frame::Abi<Reg>;

struct Counter {
    int: usize,
    float: u8,
    stack: u32,
    win64: bool,
}

impl Counter {
    fn stack(&mut self, bytes: u32, align: u32) -> Loc {
        self.stack = round(self.stack, align);
        let at = self.stack;
        self.stack += bytes;
        Loc::Stack(at)
    }

    fn param(&mut self, purpose: ArgumentPurpose, ty: Type) -> Loc {
        if purpose == ArgumentPurpose::Dict {
            return Loc::Int(Reg::R10);
        }
        if let ArgumentPurpose::StructArgument(size) = purpose {
            let at = self.stack;
            self.stack += round(size, WORD);
            return Loc::Struct(at, size);
        }
        if self.win64 {
            let slot = self.int;
            self.int += 1;
            return match ty {
                Type::F32 | Type::F64 if slot < WIN64_INT_ARGS.len() => Loc::Float(slot as u8),
                Type::F32 | Type::F64 => self.stack(WORD, WORD),
                Type::I128 => self.stack(16, 16),
                _ if slot < WIN64_INT_ARGS.len() => Loc::Int(WIN64_INT_ARGS[slot]),
                _ => self.stack(WORD, WORD),
            };
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
}

pub(super) fn classify(signature: &Signature, abi: CallAbi) -> Result<Abi, String> {
    let win64 = abi == CallAbi::Win64 && signature.call_conv == CallConv::Native;
    let mut params = Counter { int: 0, float: 0, stack: if win64 { 32 } else { 0 }, win64 };
    let placed = signature.params.iter().map(|param| params.param(param.purpose, param.value_type)).collect();
    Ok(Abi { params: placed, returns: returns(signature, INT_RETURNS, FLOAT_RETURNS)?, stack: round(params.stack, 16) })
}

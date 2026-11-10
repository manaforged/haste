use super::asm::Reg;
use crate::frame::round;
use crate::ir::{ArgumentPurpose, Signature, Type};

const INT_ARGS: [Reg; 6] = [Reg::Rdi, Reg::Rsi, Reg::Rdx, Reg::Rcx, Reg::R8, Reg::R9];
const FLOAT_ARGS: u8 = 8;
const INT_RETURNS: [Reg; 2] = [Reg::Rax, Reg::Rdx];
const FLOAT_RETURNS: u8 = 2;
const WORD: u32 = 8;

/// The registers every function keeps for its caller, with rbp kept by the frame chain.
pub(super) const SAVED: [Reg; 5] = [Reg::Rbx, Reg::R12, Reg::R13, Reg::R14, Reg::R15];

pub(super) type Loc = crate::frame::Loc<Reg>;
pub(super) type Abi = crate::frame::Abi<Reg>;

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


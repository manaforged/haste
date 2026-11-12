use super::asm::Reg;
use crate::frame::{returns, round};
use crate::ir::{ArgumentPurpose, Signature, Type};

const INT_ARGS: [Reg; 8] = [Reg(0), Reg(1), Reg(2), Reg(3), Reg(4), Reg(5), Reg(6), Reg(7)];
const FLOAT_ARGS: u8 = 8;
const INT_RETURNS: [Reg; 2] = [INT_ARGS[0], INT_ARGS[1]];
const FLOAT_RETURNS: u8 = 4;
/// The register that carries the address of memory for a returned aggregate.
const INDIRECT: Reg = Reg(8);

/// The registers every function keeps for its caller besides the frame pointer and the link register.
pub(super) const SAVED: [Reg; 10] = [Reg(19), Reg(20), Reg(21), Reg(22), Reg(23), Reg(24), Reg(25), Reg(26), Reg(27), Reg(28)];
pub(super) const SAVED_FLOAT: [u8; 8] = [8, 9, 10, 11, 12, 13, 14, 15];

pub(super) type Loc = crate::frame::Loc<Reg>;
pub(super) type Abi = crate::frame::Abi<Reg>;

/// How stack arguments are packed: every one in an 8-byte slot, or by its natural size and alignment as Apple's convention does.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Packing {
    Slots,
    Natural,
}

struct Counter {
    int: usize,
    float: u8,
    stack: u32,
    packing: Packing,
}

impl Counter {
    fn stack(&mut self, bytes: u32) -> Loc {
        let size = match self.packing {
            Packing::Slots => bytes.max(8),
            Packing::Natural => bytes,
        };
        self.stack = round(self.stack, size);
        let at = self.stack;
        self.stack += size;
        Loc::Stack(at)
    }

    fn param(&mut self, purpose: ArgumentPurpose, ty: Type) -> Result<Loc, String> {
        match purpose {
            ArgumentPurpose::StructReturn => return Ok(Loc::Int(INDIRECT)),
            ArgumentPurpose::Dict => return Ok(Loc::Int(Reg(9))),
            ArgumentPurpose::StructArgument(_) => return Err("a struct passed by value on the stack has no arm64 lowering yet".to_owned()),
            ArgumentPurpose::Variadic if self.packing == Packing::Natural => {
                self.int = INT_ARGS.len();
                self.float = FLOAT_ARGS;
                let size = ty.bytes().max(8);
                self.stack = round(self.stack, size);
                let at = self.stack;
                self.stack += size;
                return Ok(Loc::Stack(at));
            }
            ArgumentPurpose::Normal | ArgumentPurpose::Variadic => {}
        }
        Ok(match ty {
            Type::I128 => {
                self.int = self.int.next_multiple_of(2);
                if self.int + 2 <= INT_ARGS.len() {
                    self.int += 2;
                    Loc::Pair(INT_ARGS[self.int - 2], INT_ARGS[self.int - 1])
                } else {
                    self.int = INT_ARGS.len();
                    self.stack(16)
                }
            }
            Type::F32 | Type::F64 if self.float < FLOAT_ARGS => {
                self.float += 1;
                Loc::Float(self.float - 1)
            }
            _ if !ty.is_float() && self.int < INT_ARGS.len() => {
                self.int += 1;
                Loc::Int(INT_ARGS[self.int - 1])
            }
            _ => self.stack(ty.bytes()),
        })
    }
}

/// Places the parameters and return values of `signature` by the AAPCS64 rules, with stack arguments packed by `packing`.
pub(super) fn classify(signature: &Signature, packing: Packing) -> Result<Abi, String> {
    let mut params = Counter { int: 0, float: 0, stack: 0, packing };
    let placed = signature.params.iter().map(|param| params.param(param.purpose, param.value_type)).collect::<Result<_, _>>()?;
    Ok(Abi { params: placed, returns: returns(signature, INT_RETURNS, FLOAT_RETURNS)?, stack: round(params.stack, 16) })
}

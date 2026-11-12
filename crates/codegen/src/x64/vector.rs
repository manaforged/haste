use super::asm::{Reg, Xmm};
use super::{Emit, Ext};
use super::float::X1;
use crate::ir::{Binary, ByteCC, Op, Value};

/// The operand-size prefix every packed integer SSE2 instruction takes.
const PACKED: [u8; 1] = [0x66];

/// The bytes of a 32-bit integer whose four bytes are each one.
const BYTES: i64 = 0x0101_0101;

/// The opcode of the packed integer bitwise `op`.
fn bitwise(op: Binary) -> Option<u8> {
    match op {
        Binary::Band => Some(0xDB),
        Binary::Bor => Some(0xEB),
        Binary::Bxor => Some(0xEF),
        _ => None,
    }
}

impl Emit<'_> {
    pub(super) fn vector_op(&mut self, op: &Op, result: Value) -> Result<(), String> {
        match *op {
            Op::ByteCmp(cc, x, y) => {
                let opcode = match cc {
                    ByteCC::Equal => 0x74,
                    ByteCC::SignedGreaterThan => 0x64,
                };
                self.packed(opcode, (x, y), result);
            }
            Op::Splat(x) => self.splat(x, result),
            Op::Movemask(x) => {
                let source = self.fin(x, Xmm(0));
                let dst = self.dst(result);
                self.asm.sse(&PACKED, &[0x0F, 0xD7], dst as u8, source);
                self.put(result, dst);
            }
            _ => return Err(format!("unexpected {op:?}")),
        }
        Ok(())
    }

    /// Computes the packed bitwise `op` of vectors `x` and `y` into `result`.
    pub(super) fn vector_binary(&mut self, op: Binary, (x, y): (Value, Value), result: Value) -> Result<(), String> {
        let opcode = bitwise(op).ok_or_else(|| format!("{op:?} of a vector"))?;
        self.packed(opcode, (x, y), result);
        Ok(())
    }

    /// `dst = x op y` for the two-operand packed SSE2 instruction `opcode`, reading `y` from a register so no operand needs alignment.
    fn packed(&mut self, opcode: u8, (x, y): (Value, Value), result: Value) {
        let dst = self.fdst(result);
        let source = self.fin(y, X1);
        self.fget(dst, x);
        self.asm.sse(&PACKED, &[0x0F, opcode], dst.0, source);
        self.fput(result, dst);
    }

    /// Fills every byte of `result` with the byte `x`: its pattern in the low word, then `pshufd` copies that word to all four.
    fn splat(&mut self, x: Value, result: Value) {
        let dst = self.fdst(result);
        match self.const_of(x) {
            Some(0) => self.asm.sse(&PACKED, &[0x0F, 0xEF], dst.0, dst),
            Some(byte) => {
                self.asm.mov_imm(Reg::Rax, (byte & 0xFF) * BYTES);
                self.word_splat(dst);
            }
            None => {
                self.get(Reg::Rax, x, Ext::Zero);
                self.asm.imul_imm(Reg::Rax, Reg::Rax, BYTES as i32);
                self.word_splat(dst);
            }
        }
        self.fput(result, dst);
    }

    /// Copies the 32-bit word in eax to all four words of `dst`.
    fn word_splat(&mut self, dst: Xmm) {
        self.asm.sse(&PACKED, &[0x0F, 0x6E], dst.0, Reg::Rax);
        self.asm.sse(&PACKED, &[0x0F, 0x70], dst.0, dst);
        self.asm.bytes(&[0]);
    }

    /// Zeroes `result`, the one vector constant there is.
    pub(super) fn vector_zero(&mut self, result: Value) {
        let dst = self.fdst(result);
        self.asm.sse(&PACKED, &[0x0F, 0xEF], dst.0, dst);
        self.fput(result, dst);
    }
}

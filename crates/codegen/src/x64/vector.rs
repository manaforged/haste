use super::asm::{Reg, Xmm};
use super::Emit;
use super::float::{X0, X1, X2};
use crate::fill::BYTES;
use crate::frame::Ext;
use crate::ir::{Binary, ByteCC, Op, Value, VectorOp};

/// The operand-size prefix every packed integer SSE2 instruction takes.
const PACKED: [u8; 1] = [0x66];

/// The prefix and opcode bytes of the vector instruction `op`.
fn encoding(op: VectorOp) -> (&'static [u8], [u8; 3]) {
    match op {
        VectorOp::ByteShuffle => (&PACKED, [0x0F, 0x38, 0x00]),
        VectorOp::AddBytes => (&PACKED, [0x0F, 0xFC, 0]),
        VectorOp::AddWords => (&PACKED, [0x0F, 0xFD, 0]),
        VectorOp::AddDwords => (&PACKED, [0x0F, 0xFE, 0]),
        VectorOp::AddQwords => (&PACKED, [0x0F, 0xD4, 0]),
        VectorOp::SubBytes => (&PACKED, [0x0F, 0xF8, 0]),
        VectorOp::SubWords => (&PACKED, [0x0F, 0xF9, 0]),
        VectorOp::SubDwords => (&PACKED, [0x0F, 0xFA, 0]),
        VectorOp::SubQwords => (&PACKED, [0x0F, 0xFB, 0]),
        VectorOp::Sha1NextE => (&[], [0x0F, 0x38, 0xC8]),
        VectorOp::Sha1Msg1 => (&[], [0x0F, 0x38, 0xC9]),
        VectorOp::Sha1Msg2 => (&[], [0x0F, 0x38, 0xCA]),
        VectorOp::Sha256Msg1 => (&[], [0x0F, 0x38, 0xCC]),
        VectorOp::Sha256Msg2 => (&[], [0x0F, 0x38, 0xCD]),
        VectorOp::AesEnc => (&PACKED, [0x0F, 0x38, 0xDC]),
        VectorOp::AesEncLast => (&PACKED, [0x0F, 0x38, 0xDD]),
        VectorOp::AesDec => (&PACKED, [0x0F, 0x38, 0xDE]),
        VectorOp::AesDecLast => (&PACKED, [0x0F, 0x38, 0xDF]),
    }
}

/// The opcode bytes of `encoding`: two for a 0F map instruction, three for a 0F 38 one.
fn opcode(bytes: &[u8; 3]) -> &[u8] {
    if bytes[1] == 0x38 { bytes } else { &bytes[..2] }
}

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
                self.packed((&PACKED, &[0x0F, opcode]), (x, y), result);
            }
            Op::Vector(kind, x, y) => {
                let (prefix, bytes) = encoding(kind);
                self.packed((prefix, opcode(&bytes)), (x, y), result);
            }
            Op::Sha256Rounds(x, y, k) => self.sha256_rounds((x, y, k), result),
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
        self.packed((&PACKED, &[0x0F, opcode]), (x, y), result);
        Ok(())
    }

    fn packed(&mut self, (prefix, opcode): (&[u8], &[u8]), (x, y): (Value, Value), result: Value) {
        let dst = self.fdst(result);
        let mut source = self.fin(y, X1);
        if source == dst && x != y {
            self.asm.movaps(X1, source);
            source = X1;
        }
        self.fget(dst, x);
        self.asm.sse(prefix, opcode, dst.0, source);
        self.fput(result, dst);
    }

    fn sha256_rounds(&mut self, (x, y, k): (Value, Value, Value), result: Value) {
        let dst = self.xmm_of(result).unwrap_or(X2);
        let mut source = self.fin(y, X1);
        if source == dst && x != y {
            self.asm.movaps(X1, source);
            source = X1;
        }
        self.fget(X0, k);
        self.fget(dst, x);
        self.asm.sse(&[], &[0x0F, 0x38, 0xCB], dst.0, source);
        self.fput(result, dst);
    }

    /// Fills every byte of `result` with the byte `x`: its pattern in the low word, then `pshufd` copies that word to all four.
    fn splat(&mut self, x: Value, result: Value) {
        let dst = self.fdst(result);
        match self.const_of(x) {
            Some(0) => self.asm.sse(&PACKED, &[0x0F, 0xEF], dst.0, dst),
            Some(byte) => {
                self.asm.mov_imm(Reg::Rax, (byte & 0xFF).wrapping_mul(BYTES));
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

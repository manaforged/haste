use super::Emit;
use super::asm::{Alu, Cond, Reg, Rm, Shift, Size};
use crate::ir::{Binary, Type};

/// The masks and shifts of the bit-parallel steps that reverse and count the bits of a word.
const BIT_STEPS: [(u64, u8); 3] = [(0x5555_5555_5555_5555, 1), (0x3333_3333_3333_3333, 2), (0x0F0F_0F0F_0F0F_0F0F, 4)];

impl Emit<'_> {
    pub(super) fn emit_sat_op(&mut self, op: Binary, size: Size, dst: Reg, src: Reg) {
        let (umax, smin, smax) = match size {
            Size::B => (0xFF_i64, 0x80_i64, 0x7F_i64),
            Size::W => (0xFFFF_i64, 0x8000_i64, 0x7FFF_i64),
            Size::D => (0xFFFF_FFFF_i64, 0x8000_0000_i64, 0x7FFF_FFFF_i64),
            Size::Q => (-1_i64, i64::MIN, i64::MAX),
        };
        match op {
            Binary::UaddSat => {
                self.asm.alu_sized(size, Alu::Add, dst, src);
                self.asm.mov_imm(Reg::Rdx, umax);
                self.asm.cmov(Cond::B, dst, Reg::Rdx);
            }
            Binary::UsubSat => {
                self.asm.alu32(Alu::Xor, Reg::Rdx, Reg::Rdx);
                self.asm.alu_sized(size, Alu::Sub, dst, src);
                self.asm.cmov(Cond::B, dst, Reg::Rdx);
            }
            Binary::SaddSat | Binary::SsubSat => {
                let alu = if op == Binary::SaddSat { Alu::Add } else { Alu::Sub };
                self.asm.alu_sized(size, alu, dst, src);
                self.asm.mov_imm(Reg::Rdx, smin);
                self.asm.mov_imm(Reg::R11, smax);
                self.asm.cmov(Cond::S, Reg::Rdx, Reg::R11);
                self.asm.cmov(Cond::O, dst, Reg::Rdx);
            }
            _ => {}
        }
        self.clobber_regs(&[dst, Reg::Rdx, Reg::R11]);
    }

    pub(super) fn popcount(&mut self, dst: Reg, src: Rm) {
        let (part, mask) = (Reg::R11, Reg::Rsi);
        match src {
            Rm::Reg(r) if r == dst as u8 => {}
            _ => self.asm.load(Size::Q, dst, src),
        }
        for (bits, shift) in BIT_STEPS {
            self.asm.mov_imm(mask, bits as i64);
            self.asm.mov(part, dst);
            self.asm.shift_imm(Size::Q, Shift::Shr, part, shift);
            self.asm.alu(Alu::And, part, mask);
            self.asm.alu(Alu::And, dst, mask);
            self.asm.alu(Alu::Add, dst, part);
        }
        self.asm.mov_imm(mask, 0x0101_0101_0101_0101);
        self.asm.imul(dst, mask);
        self.asm.shift_imm(Size::Q, Shift::Shr, dst, 56);
        self.clobber_regs(&[dst, Reg::R11, Reg::Rsi]);
    }

    pub(super) fn reverse_bits(&mut self) {
        for (mask, shift) in BIT_STEPS {
            self.asm.mov_imm(Reg::Rcx, mask as i64);
            self.asm.mov(Reg::Rdx, Reg::Rax);
            self.asm.shift_imm(Size::Q, Shift::Shr, Reg::Rdx, shift);
            self.asm.alu(Alu::And, Reg::Rdx, Reg::Rcx);
            self.asm.alu(Alu::And, Reg::Rax, Reg::Rcx);
            self.asm.shift_imm(Size::Q, Shift::Shl, Reg::Rax, shift);
            self.asm.alu(Alu::Or, Reg::Rax, Reg::Rdx);
        }
        self.asm.bswap(Size::Q, Reg::Rax);
        self.clobber_regs(&[Reg::Rax, Reg::Rcx, Reg::Rdx]);
    }

    pub(super) fn clamp_signed(&mut self, reg: Reg, to: Type, signed: bool) {
        let bits = to.bits();
        if bits >= 64 {
            return;
        }
        let (lo, hi) = if signed {
            (-(1_i64 << (bits - 1)), (1_i64 << (bits - 1)) - 1)
        } else {
            (0, (1_i64 << bits) - 1)
        };
        self.asm.mov_imm(Reg::Rcx, lo);
        self.asm.alu(Alu::Cmp, reg, Reg::Rcx);
        self.asm.cmov(Cond::L, reg, Reg::Rcx);
        self.asm.mov_imm(Reg::Rcx, hi);
        self.asm.alu(Alu::Cmp, reg, Reg::Rcx);
        self.asm.cmov(Cond::G, reg, Reg::Rcx);
        self.clobber_regs(&[reg, Reg::Rcx]);
    }
}

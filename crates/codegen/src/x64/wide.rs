use super::Emit;
use super::asm::{Alu, Cond, Count, Group3, Reg, Shift, Size};
use super::int::alu_of;
use crate::ir::{Binary, IntCC, Unary, Value};

/// The registers 128-bit rotates use as scratch besides rax, rcx and rdx.
pub(super) const WIDE_SCRATCH: [Reg; 3] = [Reg::R8, Reg::R9, Reg::R10];

impl Emit<'_> {
    fn halves(&mut self, x: Value) {
        self.raw(Reg::Rax, x);
        self.high(Reg::Rdx, x);
    }

    fn put_halves(&mut self, result: Value) {
        self.put(result, Reg::Rax);
        self.put_high(result, Reg::Rdx);
    }

    fn put_count(&mut self, result: Value) {
        self.put(result, Reg::Rax);
        self.asm.alu32(Alu::Xor, Reg::Rax, Reg::Rax);
        self.put_high(result, Reg::Rax);
    }

    pub(super) fn wide_unary(&mut self, op: Unary, x: Value, result: Value) -> Result<(), String> {
        match op {
            Unary::Ineg => {
                self.halves(x);
                self.asm.group3(Group3::Neg, Reg::Rax);
                self.asm.alu_imm(Alu::Adc, Reg::Rdx, 0);
                self.asm.group3(Group3::Neg, Reg::Rdx);
            }
            Unary::Bnot => {
                self.halves(x);
                self.asm.group3(Group3::Not, Reg::Rax);
                self.asm.group3(Group3::Not, Reg::Rdx);
            }
            Unary::Bswap => {
                self.high(Reg::Rax, x);
                self.raw(Reg::Rdx, x);
                self.asm.bswap(Size::Q, Reg::Rax);
                self.asm.bswap(Size::Q, Reg::Rdx);
            }
            Unary::Bitrev => {
                self.raw(Reg::R11, x);
                self.high(Reg::Rax, x);
                self.reverse_bits();
                self.put(result, Reg::Rax);
                self.asm.mov(Reg::Rax, Reg::R11);
                self.reverse_bits();
                self.put_high(result, Reg::Rax);
                return Ok(());
            }
            Unary::Clz | Unary::Ctz | Unary::Popcnt => {
                self.wide_count(op, x);
                self.put_count(result);
                return Ok(());
            }
            other => return Err(format!("{other:?} of a 128-bit integer")),
        }
        self.put_halves(result);
        Ok(())
    }

    fn wide_count(&mut self, op: Unary, x: Value) {
        self.raw(Reg::Rcx, x);
        self.high(Reg::Rdx, x);
        let (count, first, second) = match op {
            Unary::Clz => (Count::Lzcnt, Reg::Rdx, Reg::Rcx),
            Unary::Ctz => (Count::Tzcnt, Reg::Rcx, Reg::Rdx),
            _ => (Count::Popcnt, Reg::Rcx, Reg::Rdx),
        };
        self.count(count, Reg::Rax, first);
        self.count(count, second, second);
        if op == Unary::Popcnt {
            self.asm.alu(Alu::Add, Reg::Rax, second);
            return;
        }
        self.asm.alu_imm(Alu::Add, second, 64);
        self.asm.test(first, first);
        self.asm.cmov(Cond::E, Reg::Rax, second);
    }

    pub(super) fn wide_binary(&mut self, op: Binary, x: Value, y: Value, result: Value) -> Result<(), String> {
        if let Some(alu) = alu_of(op) {
            let high = match alu {
                Alu::Add => Alu::Adc,
                Alu::Sub => Alu::Sbb,
                same => same,
            };
            self.halves(x);
            self.asm.alu(alu, Reg::Rax, self.at(y));
            self.asm.alu(high, Reg::Rdx, self.at(y).offset(8));
            self.put_halves(result);
            return Ok(());
        }
        match op {
            Binary::Imul => self.wide_multiply(x, y),
            Binary::Ishl | Binary::Ushr | Binary::Sshr => {
                self.halves(x);
                self.raw(Reg::Rcx, y);
                self.wide_shift(op, (Reg::Rax, Reg::Rdx), Reg::Rsi);
            }
            Binary::Rotl | Binary::Rotr => self.wide_rotate(op == Binary::Rotl, x, y),
            other => return Err(format!("{other:?} of a 128-bit integer")),
        }
        self.put_halves(result);
        Ok(())
    }

    fn wide_multiply(&mut self, x: Value, y: Value) {
        self.raw(Reg::Rax, x);
        self.asm.group3(Group3::Mul, self.at(y));
        self.raw(Reg::Rcx, x);
        self.asm.imul(Reg::Rcx, self.at(y).offset(8));
        self.asm.alu(Alu::Add, Reg::Rdx, Reg::Rcx);
        self.high(Reg::Rcx, x);
        self.asm.imul(Reg::Rcx, self.at(y));
        self.asm.alu(Alu::Add, Reg::Rdx, Reg::Rcx);
    }

    /// Shifts the pair (low, high) by cl modulo 128, using `spare` for the fill word.
    fn wide_shift(&mut self, op: Binary, (low, high): (Reg, Reg), spare: Reg) {
        match op {
            Binary::Ishl => {
                self.asm.alu32(Alu::Xor, spare, spare);
                self.asm.shld(high, low);
                self.asm.shift(Size::Q, Shift::Shl, low);
                self.asm.test_imm8(Reg::Rcx, 64);
                self.asm.cmov(Cond::Ne, high, low);
                self.asm.cmov(Cond::Ne, low, spare);
            }
            Binary::Ushr => {
                self.asm.alu32(Alu::Xor, spare, spare);
                self.asm.shrd(low, high);
                self.asm.shift(Size::Q, Shift::Shr, high);
                self.asm.test_imm8(Reg::Rcx, 64);
                self.asm.cmov(Cond::Ne, low, high);
                self.asm.cmov(Cond::Ne, high, spare);
            }
            _ => {
                self.asm.mov(spare, high);
                self.asm.shift_imm(Size::Q, Shift::Sar, spare, 63);
                self.asm.shrd(low, high);
                self.asm.shift(Size::Q, Shift::Sar, high);
                self.asm.test_imm8(Reg::Rcx, 64);
                self.asm.cmov(Cond::Ne, low, high);
                self.asm.cmov(Cond::Ne, high, spare);
            }
        }
    }

    fn wide_rotate(&mut self, left: bool, x: Value, y: Value) {
        let (first, second) = if left { (Binary::Ishl, Binary::Ushr) } else { (Binary::Ushr, Binary::Ishl) };
        let [low, high, spare] = WIDE_SCRATCH;
        self.halves(x);
        self.asm.mov(low, Reg::Rax);
        self.asm.mov(high, Reg::Rdx);
        self.raw(Reg::Rcx, y);
        self.asm.alu_imm32(Alu::And, Reg::Rcx, 127);
        self.wide_shift(first, (Reg::Rax, Reg::Rdx), spare);
        self.asm.group3(Group3::Neg, Reg::Rcx);
        self.asm.alu_imm32(Alu::And, Reg::Rcx, 127);
        self.wide_shift(second, (low, high), spare);
        self.asm.alu(Alu::Or, Reg::Rax, low);
        self.asm.alu(Alu::Or, Reg::Rdx, high);
    }

    pub(super) fn wide_icmp(&mut self, cc: IntCC, x: Value, y: Value) -> Cond {
        if matches!(cc, IntCC::Equal | IntCC::NotEqual) {
            self.halves(x);
            self.asm.alu(Alu::Xor, Reg::Rax, self.at(y));
            self.asm.alu(Alu::Xor, Reg::Rdx, self.at(y).offset(8));
            self.asm.alu(Alu::Or, Reg::Rax, Reg::Rdx);
            return if cc == IntCC::Equal { Cond::E } else { Cond::Ne };
        }
        let (swap, cond) = match cc {
            IntCC::SignedLessThan => (false, Cond::L),
            IntCC::SignedGreaterThanOrEqual => (false, Cond::Ge),
            IntCC::SignedGreaterThan => (true, Cond::L),
            IntCC::SignedLessThanOrEqual => (true, Cond::Ge),
            IntCC::UnsignedLessThan => (false, Cond::B),
            IntCC::UnsignedGreaterThanOrEqual => (false, Cond::Ae),
            IntCC::UnsignedGreaterThan => (true, Cond::B),
            _ => (true, Cond::Ae),
        };
        let (a, b) = if swap { (y, x) } else { (x, y) };
        self.halves(a);
        self.asm.alu(Alu::Cmp, Reg::Rax, self.at(b));
        self.asm.alu(Alu::Sbb, Reg::Rdx, self.at(b).offset(8));
        cond
    }
}

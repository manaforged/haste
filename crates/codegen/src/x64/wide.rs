use super::Emit;
use super::asm::{Alu, Cond, Count, Group3, Reg, Shift, Size};
use crate::ir::{Binary, IntCC, Op, Unary, Value};

/// The registers 128-bit rotates use as scratch besides rax, rcx and rdx.
pub(super) const WIDE_SCRATCH: [Reg; 3] = [Reg::R8, Reg::R9, Reg::R10];

fn half_score((lo, hi): (Option<i64>, Option<i64>), zero_only: bool) -> u8 {
    let hit = |v: Option<i64>| matches!(v, Some(0)) || (!zero_only && matches!(v, Some(-1)));
    u8::from(hit(lo)) + u8::from(hit(hi))
}

impl Emit<'_> {
    fn const_val(&self, v: Value) -> Option<i64> {
        self.const_of(v).or_else(|| match self.func.insts[self.defs[v.index()]?.index()].op {
            Op::Const(imm) => Some(imm),
            _ => None,
        })
    }

    fn const_halves(&self, v: Value) -> (Option<i64>, Option<i64>) {
        if let Some(imm) = self.const_val(v) {
            return (Some(imm), Some(imm >> 63));
        }
        let Some(inst) = self.defs[v.index()] else { return (None, None) };
        match self.func.insts[inst.index()].op {
            Op::Binary(Binary::Iconcat, lo, hi) => (self.const_val(lo), self.const_val(hi)),
            _ => (None, None),
        }
    }

    fn shift_const(&self, y: Value) -> Option<u8> {
        self.const_halves(y).0.map(|imm| (imm & 127) as u8)
    }

    fn load_half(&mut self, high: bool, reg: Reg, value: Value) {
        if !high {
            self.raw(reg, value);
        } else if let Some(imm) = self.const_val(value) {
            self.asm.mov_imm(reg, imm >> 63);
        } else {
            self.high(reg, value);
        }
    }

    fn halves(&mut self, x: Value) {
        self.load_half(false, Reg::Rax, x);
        self.load_half(true, Reg::Rdx, x);
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
        match op {
            Binary::Iadd | Binary::Isub => {
                let (low, high) = if op == Binary::Iadd { (Alu::Add, Alu::Adc) } else { (Alu::Sub, Alu::Sbb) };
                self.halves(x);
                self.asm.alu(low, Reg::Rax, self.at(y));
                self.asm.alu(high, Reg::Rdx, self.at(y).offset(8));
            }
            Binary::Band | Binary::Bor | Binary::Bxor => self.wide_bitwise(op, x, y),
            Binary::Imul => self.wide_multiply(x, y),
            Binary::Ishl | Binary::Ushr | Binary::Sshr => {
                if let Some(k) = self.shift_const(y) {
                    if k < 64 || op == Binary::Ishl {
                        self.load_half(false, Reg::Rax, x);
                    }
                    if k < 64 || op != Binary::Ishl {
                        self.load_half(true, Reg::Rdx, x);
                    }
                    self.wide_shift_const(op, k);
                } else {
                    self.halves(x);
                    self.raw(Reg::Rcx, y);
                    self.wide_shift(op, (Reg::Rax, Reg::Rdx), Reg::Rsi);
                }
            }
            Binary::Rotl | Binary::Rotr => self.wide_rotate(op == Binary::Rotl, x, y),
            other => return Err(format!("{other:?} of a 128-bit integer")),
        }
        self.put_halves(result);
        Ok(())
    }

    fn wide_bitwise(&mut self, op: Binary, x: Value, y: Value) {
        let (cx, cy) = (self.const_halves(x), self.const_halves(y));
        let ((lhs, rhs), (rhs_lo, rhs_hi)) = if half_score(cx, false) > half_score(cy, false) {
            ((y, x), cx)
        } else {
            ((x, y), cy)
        };
        self.wide_bitwise_half((op, false), (lhs, rhs, rhs_lo), Reg::Rax);
        self.wide_bitwise_half((op, true), (lhs, rhs, rhs_hi), Reg::Rdx);
    }

    fn alu_rhs_half(&mut self, alu: Alu, (high, reg): (bool, Reg), (rhs, rhs_c): (Value, Option<i64>)) {
        if let Some(imm) = rhs_c.and_then(|v| i8::try_from(v).ok()) {
            self.asm.alu_imm(alu, reg, i32::from(imm));
        } else if let Some(imm) = rhs_c.filter(|_| self.const_val(rhs).is_some()) {
            self.asm.mov_imm(Reg::Rcx, imm);
            self.asm.alu(alu, reg, Reg::Rcx);
        } else {
            let mem = if high { self.at(rhs).offset(8) } else { self.at(rhs) };
            self.asm.alu(alu, reg, mem);
        }
    }

    fn wide_bitwise_half(&mut self, (op, high): (Binary, bool), (lhs, rhs, rhs_c): (Value, Value, Option<i64>), reg: Reg) {
        match (op, rhs_c) {
            (Binary::Band, Some(0)) => return self.asm.alu32(Alu::Xor, reg, reg),
            (Binary::Bor, Some(-1)) => return self.asm.mov_imm(reg, -1),
            (Binary::Bor | Binary::Bxor, Some(0)) | (Binary::Band, Some(-1)) => return self.load_half(high, reg, lhs),
            (Binary::Bxor, Some(-1)) => {
                self.load_half(high, reg, lhs);
                return self.asm.group3(Group3::Not, reg);
            }
            _ => {}
        }
        self.load_half(high, reg, lhs);
        let alu = match op {
            Binary::Band => Alu::And,
            Binary::Bor => Alu::Or,
            _ => Alu::Xor,
        };
        self.alu_rhs_half(alu, (high, reg), (rhs, rhs_c));
    }

    fn wide_shift_const(&mut self, op: Binary, k: u8) {
        if k == 0 {
            return;
        }
        if k >= 64 {
            let s = k - 64;
            match op {
                Binary::Ishl => {
                    self.asm.mov(Reg::Rdx, Reg::Rax);
                    if s > 0 {
                        self.asm.shift_imm(Size::Q, Shift::Shl, Reg::Rdx, s);
                    }
                    self.asm.alu32(Alu::Xor, Reg::Rax, Reg::Rax);
                }
                _ => {
                    self.asm.mov(Reg::Rax, Reg::Rdx);
                    if s > 0 {
                        let kind = if op == Binary::Sshr { Shift::Sar } else { Shift::Shr };
                        self.asm.shift_imm(Size::Q, kind, Reg::Rax, s);
                    }
                    if op == Binary::Sshr {
                        self.asm.shift_imm(Size::Q, Shift::Sar, Reg::Rdx, 63);
                    } else {
                        self.asm.alu32(Alu::Xor, Reg::Rdx, Reg::Rdx);
                    }
                }
            }
            return;
        }
        match op {
            Binary::Ishl => {
                self.asm.shld_imm(Size::Q, Reg::Rdx, Reg::Rax, k);
                self.asm.shift_imm(Size::Q, Shift::Shl, Reg::Rax, k);
            }
            _ => {
                self.asm.shrd_imm(Size::Q, Reg::Rax, Reg::Rdx, k);
                let kind = if op == Binary::Sshr { Shift::Sar } else { Shift::Shr };
                self.asm.shift_imm(Size::Q, kind, Reg::Rdx, k);
            }
        }
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
        self.halves(x);
        if let Some(k) = self.shift_const(y) {
            let r = if left { (128 - k) & 127 } else { k };
            if r == 0 {
                return;
            }
            self.asm.mov(Reg::Rcx, Reg::Rax);
            if r < 64 {
                self.asm.shrd_imm(Size::Q, Reg::Rax, Reg::Rdx, r);
                self.asm.shrd_imm(Size::Q, Reg::Rdx, Reg::Rcx, r);
                return;
            }
            self.asm.mov(Reg::Rax, Reg::Rdx);
            if r > 64 {
                self.asm.shrd_imm(Size::Q, Reg::Rax, Reg::Rcx, r - 64);
                self.asm.shrd_imm(Size::Q, Reg::Rcx, Reg::Rdx, r - 64);
            }
            self.asm.mov(Reg::Rdx, Reg::Rcx);
            return;
        }
        let (first, second) = if left { (Binary::Ishl, Binary::Ushr) } else { (Binary::Ushr, Binary::Ishl) };
        let [low, high, spare] = WIDE_SCRATCH;
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

    fn xor_icmp_half(&mut self, high: bool, reg: Reg, rhs: (Value, Option<i64>)) {
        if rhs.1 != Some(0) {
            self.alu_rhs_half(Alu::Xor, (high, reg), rhs);
        }
    }

    pub(super) fn wide_icmp(&mut self, cc: IntCC, x: Value, y: Value) -> Cond {
        if matches!(cc, IntCC::Equal | IntCC::NotEqual) {
            let (cx, cy) = (self.const_halves(x), self.const_halves(y));
            let ((lhs, rhs), (rhs_lo, rhs_hi)) = if half_score(cx, true) > half_score(cy, true) {
                ((y, x), cx)
            } else {
                ((x, y), cy)
            };
            self.halves(lhs);
            self.xor_icmp_half(false, Reg::Rax, (rhs, rhs_lo));
            self.xor_icmp_half(true, Reg::Rdx, (rhs, rhs_hi));
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

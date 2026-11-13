use super::asm::{Alu, Bits, Cond, Reg, Size, ZR};
use super::{A, C, D, Emit, T, V0};
use crate::frame::Ext;
use crate::ir::{Binary, Convert, IntCC, Type, Unary, Value};

pub(super) fn cond(cc: IntCC) -> Cond {
    match cc {
        IntCC::Equal => Cond::Eq,
        IntCC::NotEqual => Cond::Ne,
        IntCC::SignedLessThan => Cond::Lt,
        IntCC::SignedGreaterThanOrEqual => Cond::Ge,
        IntCC::SignedGreaterThan => Cond::Gt,
        IntCC::SignedLessThanOrEqual => Cond::Le,
        IntCC::UnsignedLessThan => Cond::Lo,
        IntCC::UnsignedGreaterThanOrEqual => Cond::Hs,
        IntCC::UnsignedGreaterThan => Cond::Hi,
        IntCC::UnsignedLessThanOrEqual => Cond::Ls,
    }
}

pub(super) fn alu_of(op: Binary) -> Option<Alu> {
    match op {
        Binary::Iadd => Some(Alu::Add),
        Binary::Isub => Some(Alu::Sub),
        Binary::Band => Some(Alu::And),
        Binary::Bor => Some(Alu::Orr),
        Binary::Bxor => Some(Alu::Eor),
        _ => None,
    }
}

impl Emit<'_> {
    pub(super) fn unary(&mut self, op: Unary, x: Value, result: Value) -> Result<(), String> {
        let ty = self.ty(x);
        if ty.is_float() {
            return self.float_unary(op, x, result);
        }
        if ty == Type::I128 {
            return self.wide_unary(op, x, result);
        }
        let dst = self.dst(result);
        self.int_unary(op, x, ty.bits() as u8, dst)?;
        self.put(result, dst);
        Ok(())
    }

    fn int_unary(&mut self, op: Unary, x: Value, bits: u8, dst: Reg) -> Result<(), String> {
        match op {
            Unary::Ineg => {
                let reg = self.bits(x, A);
                self.asm.alu(Alu::Sub, Size::X, dst, ZR, reg);
            }
            Unary::Bnot => {
                let reg = self.bits(x, A);
                self.asm.alu(Alu::Orn, Size::X, dst, ZR, reg);
            }
            Unary::Bswap => {
                let reg = self.bits(x, A);
                let op = match bits {
                    16 => Bits::Rev16,
                    32 => Bits::Rev32,
                    64 => Bits::Rev,
                    _ => return Ok(self.asm.mov(dst, reg)),
                };
                self.asm.bits(op, dst, reg);
            }
            Unary::Bitrev => {
                let reg = self.bits(x, A);
                self.asm.bits(Bits::Rbit, dst, reg);
                if bits < 64 {
                    self.asm.lsr_imm(dst, dst, 64 - bits);
                }
            }
            Unary::Clz => {
                let reg = self.src(x, A, Ext::Zero);
                self.asm.bits(Bits::Clz, dst, reg);
                if bits < 64 {
                    self.asm.add_imm(true, false, Size::X, dst, dst, u32::from(64 - bits));
                }
            }
            Unary::Ctz => {
                let reg = self.bits(x, A);
                let source = if bits < 64 {
                    self.asm.mov_imm(C, 1 << bits);
                    self.asm.alu(Alu::Orr, Size::X, A, reg, C);
                    A
                } else {
                    reg
                };
                self.asm.bits(Bits::Rbit, dst, source);
                self.asm.bits(Bits::Clz, dst, dst);
            }
            Unary::Popcnt => {
                let reg = self.src(x, A, Ext::Zero);
                self.popcount(dst, reg);
            }
            other => return Err(format!("{other:?} of an integer")),
        }
        Ok(())
    }

    pub(super) fn popcount(&mut self, dst: Reg, src: Reg) {
        self.asm.fmov_to_float(V0, src);
        self.asm.popcount_bytes(V0, V0);
        self.asm.fmov_to_int(dst, V0);
    }

    pub(super) fn binary(&mut self, op: Binary, x: Value, y: Value, result: Value) -> Result<(), String> {
        let ty = self.ty(x);
        if self.plan.is_static(result) || self.plan.deferred[result.index()] {
            return Ok(());
        }
        if op == Binary::Iconcat {
            self.raw(A, x);
            self.put(result, A);
            self.raw(A, y);
            self.put_high(result, A);
            return Ok(());
        }
        if ty.is_float() {
            return self.float_binary(op, x, y, result);
        }
        if ty == Type::I128 {
            return self.wide_binary(op, x, y, result);
        }
        let dst = self.dst(result);
        self.int_binary(op, (x, y), ty, dst)?;
        self.put(result, dst);
        Ok(())
    }

    fn add_imm_args(&self, sub: bool, value: Value, ty: Type) -> Option<(bool, u32)> {
        let imm = crate::fold::extend(self.plan.const_of(value)?, ty, true);
        if (0..4096).contains(&imm) {
            Some((sub, imm as u32))
        } else if (-4095..0).contains(&imm) {
            Some((!sub, (-imm) as u32))
        } else {
            None
        }
    }

    fn int_binary(&mut self, op: Binary, (x, y): (Value, Value), ty: Type, dst: Reg) -> Result<(), String> {
        if let Some(alu) = alu_of(op) {
            let left = self.bits(x, A);
            if matches!(alu, Alu::Add | Alu::Sub)
                && let Some((sub, imm)) = self.add_imm_args(alu == Alu::Sub, y, ty)
            {
                self.asm.add_imm(sub, false, Size::X, dst, left, imm);
            } else {
                let right = self.bits(y, C);
                self.asm.alu(alu, Size::X, dst, left, right);
            }
            return Ok(());
        }
        let bits = ty.bits() as u8;
        match op {
            Binary::Imul => self.multiply(dst, x, y),
            Binary::Udiv | Binary::Urem if self.divide_by_power(op, (x, y), dst) => {}
            Binary::Sdiv | Binary::Srem | Binary::Udiv | Binary::Urem => self.divide(op, (x, y), dst),
            Binary::Umulhi | Binary::Smulhi => self.multiply_high(op == Binary::Smulhi, (x, y), bits, dst),
            Binary::Smin | Binary::Smax => {
                let left = self.src(x, A, Ext::Sign);
                let right = self.src(y, C, Ext::Sign);
                self.asm.alu(Alu::Subs, Size::X, ZR, left, right);
                self.asm.csel(if op == Binary::Smin { Cond::Lt } else { Cond::Gt }, dst, left, right);
            }
            Binary::Ishl | Binary::Ushr | Binary::Sshr => self.shift_by(op, (x, y), bits, dst),
            Binary::Rotl | Binary::Rotr => self.rotate(op == Binary::Rotl, (x, y), bits, dst),
            other => return Err(format!("{other:?} of an integer")),
        }
        Ok(())
    }

    fn multiply(&mut self, dst: Reg, x: Value, y: Value) {
        let (value, imm) = match (self.plan.const_of(x), self.plan.const_of(y)) {
            (_, Some(imm)) => (x, imm),
            (Some(imm), None) => (y, imm),
            (None, None) => (x, 0),
        };
        if imm > 0 && imm.count_ones() == 1 {
            let reg = self.bits(value, A);
            self.asm.lsl_imm(dst, reg, imm.trailing_zeros() as u8);
            return;
        }
        let left = self.bits(x, A);
        let right = self.bits(y, C);
        self.asm.madd(false, dst, (left, right), ZR);
    }

    fn divide_by_power(&mut self, op: Binary, (x, y): (Value, Value), dst: Reg) -> bool {
        let ty = self.ty(y);
        let Some(divisor) = self.plan.const_of(y).map(|imm| crate::fold::extend(imm, ty, false)).filter(|imm| *imm > 0 && imm.count_ones() == 1) else { return false };
        let left = self.src(x, A, Ext::Zero);
        match op {
            Binary::Udiv => self.asm.lsr_imm(dst, left, divisor.trailing_zeros() as u8),
            _ => {
                self.asm.mov_imm(C, divisor - 1);
                self.asm.alu(Alu::And, Size::X, dst, left, C);
            }
        }
        true
    }

    fn divide(&mut self, op: Binary, (x, y): (Value, Value), dst: Reg) {
        let signed = matches!(op, Binary::Sdiv | Binary::Srem);
        let left = self.src(x, A, Ext::of(signed));
        let right = self.src(y, C, Ext::of(signed));
        let div = if signed { Alu::Sdiv } else { Alu::Udiv };
        if matches!(op, Binary::Sdiv | Binary::Udiv) {
            self.asm.alu(div, Size::X, dst, left, right);
            return;
        }
        self.asm.alu(div, Size::X, D, left, right);
        self.asm.madd(true, dst, (D, right), left);
    }

    fn multiply_high(&mut self, signed: bool, (x, y): (Value, Value), bits: u8, dst: Reg) {
        let left = self.src(x, A, Ext::of(signed));
        let right = self.src(y, C, Ext::of(signed));
        if bits == 64 {
            self.asm.alu(if signed { Alu::Smulh } else { Alu::Umulh }, Size::X, dst, left, right);
            return;
        }
        self.asm.madd(false, dst, (left, right), ZR);
        if signed {
            self.asm.asr_imm(dst, dst, bits);
        } else {
            self.asm.lsr_imm(dst, dst, bits);
        }
    }

    fn shift_by(&mut self, op: Binary, (x, y): (Value, Value), bits: u8, dst: Reg) {
        let (ext, alu) = match op {
            Binary::Ishl => (Ext::Zero, Alu::Lslv),
            Binary::Ushr => (Ext::Zero, Alu::Lsrv),
            _ => (Ext::Sign, Alu::Asrv),
        };
        let value = if op == Binary::Ishl { self.bits(x, A) } else { self.src(x, A, ext) };
        if let Some(count) = self.plan.const_of(y) {
            let count = (count as u8) & (bits - 1);
            match alu {
                Alu::Lslv => self.asm.lsl_imm(dst, value, count),
                Alu::Lsrv => self.asm.lsr_imm(dst, value, count),
                _ => self.asm.asr_imm(dst, value, count),
            }
            return;
        }
        let mut count = self.bits(y, C);
        if bits < 64 {
            self.asm.mov_imm(D, i64::from(bits - 1));
            self.asm.alu(Alu::And, Size::X, C, count, D);
            count = C;
        }
        self.asm.alu(alu, Size::X, dst, value, count);
    }

    fn rotate(&mut self, left: bool, (x, y): (Value, Value), bits: u8, dst: Reg) {
        let count = self.bits(y, C);
        if left {
            self.asm.alu(Alu::Sub, Size::X, C, ZR, count);
        } else {
            self.asm.mov(C, count);
        }
        if bits >= 32 {
            let value = self.bits(x, A);
            self.asm.alu(Alu::Rorv, Size::of(u32::from(bits) / 8), dst, value, C);
            return;
        }
        let value = self.src(x, A, Ext::Zero);
        self.asm.mov_imm(D, i64::from(bits - 1));
        self.asm.alu(Alu::And, Size::X, C, C, D);
        self.asm.alu(Alu::Lsrv, Size::X, T, value, C);
        self.asm.alu(Alu::Sub, Size::X, C, ZR, C);
        self.asm.alu(Alu::And, Size::X, C, C, D);
        self.asm.alu(Alu::Lslv, Size::X, dst, value, C);
        self.asm.alu(Alu::Orr, Size::X, dst, dst, T);
    }

    fn compare(&mut self, cc: IntCC, x: Value, y: Value) -> Cond {
        let bytes = self.ty(x).bytes();
        let size = if bytes >= 8 { Size::X } else { Size::W };
        let ext = Ext::of(cc.is_signed());
        let left = if bytes >= 4 { self.bits(x, A) } else { self.src(x, A, ext) };
        let imm = self.plan.const_of(y).map(|imm| crate::fold::extend(imm, self.ty(y), cc.is_signed()));
        match imm {
            Some(imm) if (0..4096).contains(&imm) => self.asm.add_imm(true, true, size, ZR, left, imm as u32),
            Some(imm) if (-4095..0).contains(&imm) => self.asm.add_imm(false, true, size, ZR, left, (-imm) as u32),
            _ => {
                let right = if bytes >= 4 { self.bits(y, C) } else { self.src(y, C, ext) };
                self.asm.alu(Alu::Subs, size, ZR, left, right);
            }
        }
        cond(cc)
    }

    pub(super) fn icmp(&mut self, cc: IntCC, x: Value, y: Value, result: Value) -> Result<(), String> {
        if self.plan.is_static(result) {
            return Ok(());
        }
        let flags = if self.ty(x) == Type::I128 { self.wide_icmp(cc, x, y) } else { self.compare(cc, x, y) };
        self.set_flag(result, flags);
        Ok(())
    }

    pub(super) fn set_flag(&mut self, result: Value, flags: Cond) {
        if self.plan.fused[result.index()] {
            self.flags = Some((result, flags));
            return;
        }
        let dst = self.dst(result);
        self.asm.cset(flags, dst);
        self.put(result, dst);
    }

    pub(super) fn fits(&mut self, op: Binary, signed: bool, (x, y): (Value, Value), result: Value) -> Result<(), String> {
        let pair = self.plan.pairs[result.index()];
        if self.plan.is_static(result) && pair.is_none() {
            return Ok(());
        }
        let bits = self.ty(x).bits() as u8;
        let reg = pair.and_then(|sum| self.plan.reg_of(sum)).unwrap_or(D);
        let ok = match (op, bits) {
            (Binary::Iadd | Binary::Isub, 32 | 64) => {
                let size = if bits == 64 { Size::X } else { Size::W };
                let (left, right) = (self.bits(x, A), self.bits(y, C));
                let sub = op == Binary::Isub;
                self.asm.alu(if sub { Alu::Subs } else { Alu::Adds }, size, reg, left, right);
                match (signed, sub) {
                    (true, _) => Cond::Vc,
                    (false, false) => Cond::Lo,
                    (false, true) => Cond::Hs,
                }
            }
            (Binary::Iadd | Binary::Isub | Binary::Imul, _) if bits < 64 => {
                let (left, right) = (self.src(x, A, Ext::of(signed)), self.src(y, C, Ext::of(signed)));
                match op {
                    Binary::Iadd => self.asm.alu(Alu::Add, Size::X, reg, left, right),
                    Binary::Isub => self.asm.alu(Alu::Sub, Size::X, reg, left, right),
                    _ => self.asm.madd(false, reg, (left, right), ZR),
                }
                self.asm.extend(signed, u32::from(bits) / 8, T, reg);
                self.asm.alu(Alu::Subs, Size::X, ZR, reg, T);
                Cond::Eq
            }
            (Binary::Imul, _) => {
                let (left, right) = (self.bits(x, A), self.bits(y, C));
                self.asm.alu(if signed { Alu::Smulh } else { Alu::Umulh }, Size::X, T, left, right);
                self.asm.madd(false, reg, (left, right), ZR);
                let expected = if signed {
                    self.asm.asr_imm(A, reg, 63);
                    A
                } else {
                    ZR
                };
                self.asm.alu(Alu::Subs, Size::X, ZR, T, expected);
                Cond::Eq
            }
            (other, _) => return Err(format!("{other:?} has no overflow check")),
        };
        if let Some(sum) = pair {
            self.put(sum, reg);
        }
        self.set_flag(result, ok);
        Ok(())
    }

    pub(super) fn convert(&mut self, op: Convert, x: Value, result: Value) -> Result<(), String> {
        if self.plan.is_static(result) {
            return Ok(());
        }
        let wide = self.ty(result) == Type::I128;
        match op {
            Convert::Uextend | Convert::Sextend => {
                let signed = op == Convert::Sextend;
                let dst = self.dst(result);
                self.get(dst, x, Ext::of(signed));
                self.put(result, dst);
                if wide && signed {
                    self.asm.asr_imm(C, dst, 63);
                    self.put_high(result, C);
                } else if wide {
                    self.put_high(result, ZR);
                }
                Ok(())
            }
            Convert::Ireduce => {
                let dst = self.dst(result);
                self.raw(dst, x);
                self.put(result, dst);
                Ok(())
            }
            _ => self.float_convert(op, x, result),
        }
    }
}

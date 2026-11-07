use super::asm::{Alu, Cond, Count, Group3, Reg, Shift, Size};
use super::{Emit, Ext};
use crate::ir::{Binary, Convert, IntCC, Type, Unary, Value};

const REVERSE_STEPS: [(u64, u8); 3] = [(0x5555_5555_5555_5555, 1), (0x3333_3333_3333_3333, 2), (0x0F0F_0F0F_0F0F_0F0F, 4)];

pub(super) fn cond(cc: IntCC) -> Cond {
    match cc {
        IntCC::Equal => Cond::E,
        IntCC::NotEqual => Cond::Ne,
        IntCC::SignedLessThan => Cond::L,
        IntCC::SignedGreaterThanOrEqual => Cond::Ge,
        IntCC::SignedGreaterThan => Cond::G,
        IntCC::SignedLessThanOrEqual => Cond::Le,
        IntCC::UnsignedLessThan => Cond::B,
        IntCC::UnsignedGreaterThanOrEqual => Cond::Ae,
        IntCC::UnsignedGreaterThan => Cond::A,
        IntCC::UnsignedLessThanOrEqual => Cond::Be,
    }
}

pub(super) fn alu_of(op: Binary) -> Option<Alu> {
    match op {
        Binary::Iadd => Some(Alu::Add),
        Binary::Isub => Some(Alu::Sub),
        Binary::Band => Some(Alu::And),
        Binary::Bor => Some(Alu::Or),
        Binary::Bxor => Some(Alu::Xor),
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
        self.int_unary(op, x, ty.bits() as u8)?;
        self.put(result, Reg::Rax);
        Ok(())
    }

    fn int_unary(&mut self, op: Unary, x: Value, bits: u8) -> Result<(), String> {
        match op {
            Unary::Ineg | Unary::Bnot => {
                self.raw(Reg::Rax, x);
                self.asm.group3(if op == Unary::Ineg { Group3::Neg } else { Group3::Not }, Reg::Rax);
            }
            Unary::Bswap => self.swap_bytes(x, bits),
            Unary::Bitrev => {
                self.get(Reg::Rax, x, Ext::Zero);
                self.reverse_bits();
                if bits < 64 {
                    self.asm.shift_imm(Size::Q, Shift::Shr, Reg::Rax, 64 - bits);
                }
            }
            Unary::Clz => {
                self.get(Reg::Rax, x, Ext::Zero);
                self.asm.count(Count::Lzcnt, Reg::Rax, Reg::Rax);
                if bits < 64 {
                    self.asm.alu_imm(Alu::Sub, Reg::Rax, i32::from(64 - bits));
                }
            }
            Unary::Ctz => {
                self.get(Reg::Rax, x, Ext::Zero);
                if bits < 64 {
                    self.asm.bts(Reg::Rax, bits);
                }
                self.asm.count(Count::Tzcnt, Reg::Rax, Reg::Rax);
            }
            Unary::Popcnt => {
                self.get(Reg::Rax, x, Ext::Zero);
                self.asm.count(Count::Popcnt, Reg::Rax, Reg::Rax);
            }
            other => return Err(format!("{other:?} of an integer")),
        }
        Ok(())
    }

    fn swap_bytes(&mut self, x: Value, bits: u8) {
        self.raw(Reg::Rax, x);
        match bits {
            16 => self.asm.shift_imm(Size::W, Shift::Rol, Reg::Rax, 8),
            32 => self.asm.bswap(Size::D, Reg::Rax),
            64 => self.asm.bswap(Size::Q, Reg::Rax),
            _ => {}
        }
    }

    /// Reverses the 64 bits of rax in place, using rcx and rdx.
    pub(super) fn reverse_bits(&mut self) {
        for (mask, shift) in REVERSE_STEPS {
            self.asm.mov_imm(Reg::Rcx, mask as i64);
            self.asm.mov(Reg::Rdx, Reg::Rax);
            self.asm.shift_imm(Size::Q, Shift::Shr, Reg::Rdx, shift);
            self.asm.alu(Alu::And, Reg::Rdx, Reg::Rcx);
            self.asm.alu(Alu::And, Reg::Rax, Reg::Rcx);
            self.asm.shift_imm(Size::Q, Shift::Shl, Reg::Rax, shift);
            self.asm.alu(Alu::Or, Reg::Rax, Reg::Rdx);
        }
        self.asm.bswap(Size::Q, Reg::Rax);
    }

    pub(super) fn binary(&mut self, op: Binary, x: Value, y: Value, result: Value) -> Result<(), String> {
        let ty = self.ty(x);
        if self.is_static(result) {
            return Ok(());
        }
        if op == Binary::Iconcat {
            self.raw(Reg::Rax, x);
            self.put(result, Reg::Rax);
            self.raw(Reg::Rax, y);
            self.put_high(result, Reg::Rax);
            return Ok(());
        }
        if ty.is_float() {
            return self.float_binary(op, x, y, result);
        }
        if ty == Type::I128 {
            return self.wide_binary(op, x, y, result);
        }
        self.int_binary(op, x, y, ty)?;
        self.put(result, Reg::Rax);
        Ok(())
    }

    fn int_binary(&mut self, op: Binary, x: Value, y: Value, ty: Type) -> Result<(), String> {
        if let Some(alu) = alu_of(op) {
            self.raw(Reg::Rax, x);
            match self.imm_of(y) {
                Some(imm) => self.asm.alu_imm(alu, Reg::Rax, imm),
                None => {
                    let source = self.operand(y);
                    self.asm.alu(alu, Reg::Rax, source);
                }
            }
            return Ok(());
        }
        match op {
            Binary::Imul => {
                self.raw(Reg::Rax, x);
                let source = self.operand(y);
                self.asm.imul(Reg::Rax, source);
            }
            Binary::Sdiv | Binary::Srem | Binary::Udiv | Binary::Urem => self.divide(op, x, y),
            Binary::Umulhi | Binary::Smulhi => self.multiply_high(op == Binary::Smulhi, x, y, ty.bits() as u8),
            Binary::Smin | Binary::Smax => {
                self.get(Reg::Rax, x, Ext::Sign);
                self.get(Reg::Rcx, y, Ext::Sign);
                self.asm.alu(Alu::Cmp, Reg::Rax, Reg::Rcx);
                self.asm.cmov(if op == Binary::Smin { Cond::G } else { Cond::L }, Reg::Rax, Reg::Rcx);
            }
            Binary::Ishl | Binary::Ushr | Binary::Sshr | Binary::Rotl | Binary::Rotr => self.shift_by(op, x, y, ty),
            other => return Err(format!("{other:?} of an integer")),
        }
        Ok(())
    }

    fn divide(&mut self, op: Binary, x: Value, y: Value) {
        let signed = matches!(op, Binary::Sdiv | Binary::Srem);
        let ext = if signed { Ext::Sign } else { Ext::Zero };
        self.get(Reg::Rax, x, ext);
        self.get(Reg::Rcx, y, ext);
        if signed {
            self.asm.cqo();
            self.asm.group3(Group3::Idiv, Reg::Rcx);
        } else {
            self.asm.alu32(Alu::Xor, Reg::Rdx, Reg::Rdx);
            self.asm.group3(Group3::Div, Reg::Rcx);
        }
        if matches!(op, Binary::Srem | Binary::Urem) {
            self.asm.mov(Reg::Rax, Reg::Rdx);
        }
    }

    fn multiply_high(&mut self, signed: bool, x: Value, y: Value, bits: u8) {
        let ext = if signed { Ext::Sign } else { Ext::Zero };
        self.get(Reg::Rax, x, ext);
        self.get(Reg::Rcx, y, ext);
        if bits == 64 {
            self.asm.group3(if signed { Group3::Imul } else { Group3::Mul }, Reg::Rcx);
            self.asm.mov(Reg::Rax, Reg::Rdx);
        } else {
            self.asm.imul(Reg::Rax, Reg::Rcx);
            self.asm.shift_imm(Size::Q, if signed { Shift::Sar } else { Shift::Shr }, Reg::Rax, bits);
        }
    }

    fn shift_by(&mut self, op: Binary, x: Value, y: Value, ty: Type) {
        let (ext, shift) = match op {
            Binary::Ishl => (Ext::Zero, Shift::Shl),
            Binary::Ushr => (Ext::Zero, Shift::Shr),
            Binary::Sshr => (Ext::Sign, Shift::Sar),
            Binary::Rotl => (Ext::Zero, Shift::Rol),
            _ => (Ext::Zero, Shift::Ror),
        };
        self.get(Reg::Rax, x, ext);
        self.raw(Reg::Rcx, y);
        if matches!(op, Binary::Rotl | Binary::Rotr) {
            self.asm.shift(Size::of(ty.bytes()), shift, Reg::Rax);
        } else {
            self.asm.alu_imm32(Alu::And, Reg::Rcx, ty.bits() as i32 - 1);
            self.asm.shift(Size::Q, shift, Reg::Rax);
        }
    }

    pub(super) fn icmp(&mut self, cc: IntCC, x: Value, y: Value, result: Value) -> Result<(), String> {
        let flags = if self.ty(x) == Type::I128 {
            self.wide_icmp(cc, x, y)
        } else {
            let ext = if cc.is_signed() { Ext::Sign } else { Ext::Zero };
            self.get(Reg::Rax, x, ext);
            match self.imm_extended(y, ext) {
                Some(imm) => self.asm.alu_imm(Alu::Cmp, Reg::Rax, imm),
                None => {
                    self.get(Reg::Rcx, y, ext);
                    self.asm.alu(Alu::Cmp, Reg::Rax, Reg::Rcx);
                }
            }
            cond(cc)
        };
        if self.plan.fused[result.index()] {
            self.flags = Some((result, flags));
            return Ok(());
        }
        self.asm.setcc(flags, Reg::Rax);
        self.asm.load(Size::B, Reg::Rax, Reg::Rax);
        self.put(result, Reg::Rax);
        Ok(())
    }

    pub(super) fn convert(&mut self, op: Convert, x: Value, result: Value) -> Result<(), String> {
        let wide = self.ty(result) == Type::I128;
        match op {
            Convert::Uextend | Convert::Sextend => {
                let signed = op == Convert::Sextend;
                self.get(Reg::Rax, x, if signed { Ext::Sign } else { Ext::Zero });
                self.put(result, Reg::Rax);
                if wide && signed {
                    self.asm.mov(Reg::Rcx, Reg::Rax);
                    self.asm.shift_imm(Size::Q, Shift::Sar, Reg::Rcx, 63);
                    self.put_high(result, Reg::Rcx);
                } else if wide {
                    self.asm.alu32(Alu::Xor, Reg::Rcx, Reg::Rcx);
                    self.put_high(result, Reg::Rcx);
                }
                Ok(())
            }
            Convert::Ireduce => {
                self.raw(Reg::Rax, x);
                self.put(result, Reg::Rax);
                Ok(())
            }
            _ => self.float_convert(op, x, result),
        }
    }
}

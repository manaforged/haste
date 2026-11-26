mod binary;
mod cmp;

use super::Emit;
use super::asm::{Alu, Cond, Count, Group3, Reg, Rm, Shift, Size};
use crate::alloc::Location;
use crate::frame::Ext;
use crate::ir::{Binary, Convert, IntCC, Type, Unary, Value};

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
        self.flags = None;
        let ty = self.ty(x);
        if ty.is_vector() {
            self.invalidate_spills();
            let res = self.vector_unary(op, x, result);
            self.invalidate_spills();
            return res;
        }
        if ty.is_float() {
            return self.float_unary(op, x, result);
        }
        if ty == Type::I128 {
            self.invalidate_spills();
            let res = self.wide_unary(op, x, result);
            self.invalidate_spills();
            return res;
        }
        let dst = self.dst(result);
        let reg = self.int_unary(op, x, ty.bits() as u8, dst)?;
        self.put(result, reg);
        Ok(())
    }

    fn int_unary(&mut self, op: Unary, x: Value, bits: u8, dst: Reg) -> Result<Reg, String> {
        match op {
            Unary::Ineg | Unary::Bnot => {
                self.raw(dst, x);
                self.asm.group3(
                    if op == Unary::Ineg {
                        Group3::Neg
                    } else {
                        Group3::Not
                    },
                    dst,
                );
            }
            Unary::Bswap => {
                self.raw(dst, x);
                match bits {
                    16 => self.asm.shift_imm(Size::W, Shift::Rol, dst, 8),
                    32 => self.asm.bswap(Size::D, dst),
                    64 => self.asm.bswap(Size::Q, dst),
                    _ => {}
                }
            }
            Unary::Bitrev => {
                self.raw(Reg::Rax, x);
                self.reverse_bits();
                if bits < 64 {
                    self.asm.shift_imm(Size::Q, Shift::Shr, Reg::Rax, 64 - bits);
                }
                return Ok(Reg::Rax);
            }
            Unary::Clz | Unary::Ctz | Unary::Popcnt => {
                let (count, fast) = match op {
                    Unary::Clz => (Count::Lzcnt, self.features.lzcnt),
                    Unary::Ctz => (Count::Tzcnt, self.features.bmi1),
                    _ => (Count::Popcnt, self.features.popcnt),
                };
                self.unary_count(count, fast, bits, x, dst);
            }
            other => return Err(format!("{other:?} of an integer")),
        }
        Ok(dst)
    }

    fn unary_count(&mut self, op: Count, fast: bool, bits: u8, x: Value, dst: Reg) {
        if fast && bits >= 32 {
            let src = self.operand(x);
            return self
                .asm
                .count(op, if bits == 32 { Size::D } else { Size::Q }, dst, src);
        }
        if op == Count::Tzcnt && bits == 64 {
            let src = self.operand(x);
            return self.count(op, dst, src);
        }
        if op == Count::Tzcnt {
            self.raw(dst, x);
            self.asm.bt_imm(Size::Q, 5, dst, bits);
        } else {
            self.get(dst, x, Ext::Zero);
        }
        self.count(op, dst, dst);
        if op == Count::Lzcnt && bits < 64 {
            self.asm.alu_imm(Alu::Sub, dst, i32::from(64 - bits));
        }
    }

    /// `op` of the 64 bits of `src` into `dst`, with 64 leading or trailing zeros for zero; without the host instruction a population count uses r11 and rsi, which the allocator keeps free across it.
    pub(super) fn count(&mut self, op: Count, dst: Reg, src: impl Into<Rm>) {
        let src = src.into();
        let fast = match op {
            Count::Lzcnt => self.features.lzcnt,
            Count::Tzcnt => self.features.bmi1,
            Count::Popcnt => self.features.popcnt,
        };
        if fast {
            return self.asm.count(op, Size::Q, dst, src);
        }
        let reverse = match op {
            Count::Popcnt => return self.popcount(dst, src),
            Count::Lzcnt => true,
            Count::Tzcnt => false,
        };
        self.asm.bit_scan(reverse, dst, src);
        let found = self.asm.label();
        self.asm.jcc(Cond::Ne, found);
        self.asm.mov_imm(dst, if reverse { 127 } else { 64 });
        self.asm.bind(found);
        if reverse {
            self.asm.alu_imm(Alu::Xor, dst, 63);
        }
        self.clobber_reg(dst);
    }

    /// The cached spill register of `value` when it survives reading `other`: not the rcx that reading `other` may write.
    pub(super) fn cached_beside(&self, value: Value, other: Value) -> Option<Reg> {
        self.cached_spill(value)
            .filter(|&reg| !(reg == Reg::Rcx && self.reads_through_rcx(other)))
    }

    /// `dst op= y` on the low `size` bytes, with an immediate when `y` is a constant that fits one at that width.
    pub(super) fn alu_at(&mut self, size: Size, alu: Alu, dst: Reg, y: Value) {
        match self.imm_sized(y, size) {
            Some(imm) => self.asm.alu_imm_sized(size, alu, dst, imm),
            None => {
                let source = self.operand(y);
                self.asm.alu_sized(size, alu, dst, source);
            }
        }
    }

    pub(super) fn convert(&mut self, op: Convert, x: Value, result: Value) -> Result<(), String> {
        let saved_flags = self.flags.take();
        if self.is_static(result) {
            self.flags = saved_flags;
            return Ok(());
        }
        if self.ty(x).is_vector() || self.ty(result).is_vector() {
            self.invalidate_spills();
            let res = self.vector_convert(op, x, result);
            self.invalidate_spills();
            return res;
        }
        let wide = self.ty(result) == Type::I128;
        match op {
            Convert::Uextend | Convert::Sextend => {
                let signed = op == Convert::Sextend;
                if !wide {
                    self.flags = saved_flags;
                }
                if !wide
                    && matches!(self.location(result), Location::Slot)
                    && let Some(rx) = self.reg_of(x).or_else(|| self.cached_spill(x))
                    && (self.ty(x).bytes() >= 8 || (!signed && self.plan.zero_ext[x.index()]))
                {
                    self.put(result, rx);
                    return Ok(());
                }
                let dst = self.dst(result);
                self.get(dst, x, Ext::of(signed));
                self.put(result, dst);
                if wide && signed {
                    self.asm.mov(Reg::Rcx, Reg::Rax);
                    self.asm.shift_imm(Size::Q, Shift::Sar, Reg::Rcx, 63);
                    self.put_high(result, Reg::Rcx);
                    self.clobber_reg(Reg::Rcx);
                } else if wide {
                    self.asm.alu32(Alu::Xor, Reg::Rcx, Reg::Rcx);
                    self.put_high(result, Reg::Rcx);
                    self.clobber_reg(Reg::Rcx);
                }
                Ok(())
            }
            Convert::Ireduce => {
                let exact = saved_flags.is_some_and(|(value, _)| value == x)
                    && super::location::narrows_exactly(
                        self.func,
                        &self.plan,
                        &|v| self.defs[v.index()],
                        x,
                        self.ty(result),
                    );
                self.flags =
                    saved_flags.map(|(value, cond)| (if exact { result } else { value }, cond));
                if matches!(self.location(result), Location::Slot)
                    && let Some(rx) = self.reg_of(x).or_else(|| self.cached_spill(x))
                {
                    self.put(result, rx);
                    return Ok(());
                }
                let dst = self.dst(result);
                self.raw(dst, x);
                self.put(result, dst);
                Ok(())
            }
            Convert::NarrowSat { signed } => {
                let dst = self.dst(result);
                self.get(dst, x, Ext::Sign);
                self.clamp_signed(dst, self.ty(result), signed);
                self.put(result, dst);
                Ok(())
            }
            _ => self.float_convert(op, x, result),
        }
    }
}

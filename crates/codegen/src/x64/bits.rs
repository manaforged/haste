use super::Emit;
use super::asm::{Alu, Cond, Group3, Reg, Rm, Shift, Size};
use crate::frame::Ext;
use crate::ir::{AtomicRmwOp, Binary, Type, Value};

/// The masks and shifts of the bit-parallel steps that reverse and count the bits of a word.
const BIT_STEPS: [(u64, u8); 3] = [(0x5555_5555_5555_5555, 1), (0x3333_3333_3333_3333, 2), (0x0F0F_0F0F_0F0F_0F0F, 4)];

impl Emit<'_> {
    pub(super) fn lock_rmw(&mut self, op: AtomicRmwOp, size: Size, addr: Value, value: Value) {
        let ty = self.ty(value);
        let imm = self.const_of(value).map(|c| crate::fold::extend(c, ty, false) as u64);
        let mask = ty.unsigned_max();
        let inc_dec = match (op, imm) {
            (AtomicRmwOp::Add, Some(1)) => Some(false),
            (AtomicRmwOp::Add, Some(u)) if u == mask => Some(true),
            (AtomicRmwOp::Sub, Some(1)) => Some(true),
            (AtomicRmwOp::Sub, Some(u)) if u == mask => Some(false),
            _ => None,
        };
        let bt_ext = match (op, size != Size::B, imm) {
            (AtomicRmwOp::Or, true, Some(u)) if u.count_ones() == 1 => Some((5, u.trailing_zeros() as u8)),
            (AtomicRmwOp::And, true, Some(u)) if ((!u) & mask).count_ones() == 1 => Some((6, ((!u) & mask).trailing_zeros() as u8)),
            (AtomicRmwOp::Xor, true, Some(u)) if u.count_ones() == 1 => Some((7, u.trailing_zeros() as u8)),
            _ => None,
        };
        let alu = match op {
            AtomicRmwOp::Add => Alu::Add,
            AtomicRmwOp::Sub => Alu::Sub,
            AtomicRmwOp::Or => Alu::Or,
            AtomicRmwOp::And => Alu::And,
            _ => Alu::Xor,
        };
        if let Some(dec) = inc_dec {
            let target = self.atomic_target(addr, Reg::Rcx, &[]);
            self.asm.lock_inc_dec(size, dec, target);
        } else if let Some(imm32) = self.imm_sized(value, size) {
            let target = self.atomic_target(addr, Reg::Rcx, &[]);
            self.asm.lock_alu_imm(size, alu, target, imm32);
        } else if let Some((ext, bit)) = bt_ext {
            let target = self.atomic_target(addr, Reg::Rcx, &[]);
            self.asm.lock_bt_imm(size, ext, target, bit);
        } else if let Some(vr) = self.reg_of(value) {
            let target = self.atomic_target(addr, if vr == Reg::Rcx { Reg::Rax } else { Reg::Rcx }, &[]);
            self.asm.lock_alu_reg(size, alu, target, vr);
        } else {
            let target = self.atomic_target(addr, Reg::Rcx, &[Reg::Rax]);
            self.raw(Reg::Rax, value);
            self.asm.lock_alu_reg(size, alu, target, Reg::Rax);
        }
        self.clobber_regs(&[Reg::Rax, Reg::Rcx]);
    }

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

    /// Divides or takes the remainder by a power of two into `dst` with a shift or a mask, when `op` is unsigned and `y` is one.
    pub(super) fn divide_by_power(&mut self, op: Binary, (x, y): (Value, Value), dst: Reg) -> bool {
        let ty = self.ty(y);
        let Some(divisor) = self.const_of(y).map(|imm| crate::fold::extend(imm, ty, false) as u64).filter(|d| d.is_power_of_two()) else {
            return false;
        };
        match op {
            Binary::Udiv => {
                self.get(dst, x, Ext::Zero);
                let shift = divisor.trailing_zeros() as u8;
                if shift != 0 {
                    self.asm.shift_imm(Size::Q, Shift::Shr, dst, shift);
                }
            }
            Binary::Urem => {
                let Ok(mask) = i32::try_from(divisor - 1) else { return false };
                self.raw(dst, x);
                self.asm.alu_imm(Alu::And, dst, mask);
            }
            _ => return false,
        }
        true
    }

    /// Divides `x` by `y` at 32-bit width for sub-64-bit operands and 64-bit otherwise, returning `Rdx` for remainders and `Rax` for quotients.
    pub(super) fn divide(&mut self, op: Binary, x: Value, y: Value) -> Reg {
        let ty = self.ty(x);
        let signed = matches!(op, Binary::Sdiv | Binary::Srem);
        let ext = Ext::of(signed);
        let narrow = ty.bytes() <= 4;
        self.get(Reg::Rax, x, ext);
        let divisor = self
            .reg_of(y)
            .filter(|&r| !matches!(r, Reg::Rax | Reg::Rdx | Reg::Rsp) && (ty.bytes() >= 4 || (!signed && self.plan.zero_ext[y.index()])))
            .unwrap_or_else(|| {
                self.get(Reg::Rcx, y, ext);
                Reg::Rcx
            });
        self.asm.flags_overwritten();
        if signed {
            if narrow {
                self.asm.bytes(&[0x99]);
                self.asm.group3_sized(Size::D, Group3::Idiv, divisor);
            } else {
                self.asm.cqo();
                self.asm.group3(Group3::Idiv, divisor);
            }
        } else {
            self.asm.alu32(Alu::Xor, Reg::Rdx, Reg::Rdx);
            if narrow {
                self.asm.group3_sized(Size::D, Group3::Div, divisor);
            } else {
                self.asm.group3(Group3::Div, divisor);
            }
        }
        self.clobber_regs(&[Reg::Rax, Reg::Rdx]);
        if matches!(op, Binary::Srem | Binary::Urem) { Reg::Rdx } else { Reg::Rax }
    }

    /// Divides the 128-bit value `high:low` by `by` with one `div`, putting the quotient, or the remainder when `rem` is set, in `result`.
    pub(super) fn divide_wide(&mut self, rem: bool, [high, low, by]: [Value; 3], result: Value) {
        self.get(Reg::Rdx, high, Ext::Zero);
        self.get(Reg::Rax, low, Ext::Zero);
        self.get(Reg::Rcx, by, Ext::Zero);
        self.asm.group3(Group3::Div, Reg::Rcx);
        self.clobber_regs(&[Reg::Rax, Reg::Rcx, Reg::Rdx]);
        self.flags = None;
        self.put(result, if rem { Reg::Rdx } else { Reg::Rax });
    }

    /// Computes the high half of `x * y` and returns its register; a 64-bit product leaves its low half in rax for the `Imul` that follows.
    pub(super) fn multiply_high(&mut self, signed: bool, x: Value, y: Value, bits: u8) -> Reg {
        let ext = Ext::of(signed);
        self.get(Reg::Rax, x, ext);
        if bits == 64 {
            let source = self.reg_of(y).filter(|&reg| !matches!(reg, Reg::Rax | Reg::Rdx | Reg::Rsp)).unwrap_or_else(|| {
                self.get(Reg::Rcx, y, ext);
                Reg::Rcx
            });
            self.clobber_reg(Reg::Rax);
            self.asm.group3(if signed { Group3::Imul } else { Group3::Mul }, source);
            self.clobber_reg(Reg::Rdx);
            self.low_product = Some((x, y));
            return Reg::Rdx;
        }
        self.get(Reg::Rcx, y, ext);
        self.clobber_reg(Reg::Rax);
        self.asm.imul(Reg::Rax, Reg::Rcx);
        self.asm.shift_imm(Size::Q, if signed { Shift::Sar } else { Shift::Shr }, Reg::Rax, bits);
        Reg::Rax
    }
}

use super::asm::{Alu, Asm, Cond, Count, Group3, Mem, Reg, Rm, Shift, Size};
use super::Emit;
use crate::alloc::Location;
use crate::frame::Ext;
use crate::ir::{Binary, Convert, IntCC, Op, Type, Unary, Value};

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
                self.asm.group3(if op == Unary::Ineg { Group3::Neg } else { Group3::Not }, dst);
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
            return self.asm.count(op, if bits == 32 { Size::D } else { Size::Q }, dst, src);
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

    pub(super) fn binary(&mut self, op: Binary, x: Value, y: Value, result: Value) -> Result<(), String> {
        let saved_flags = self.flags.take();
        let ty = self.ty(x);
        if self.is_static(result) || self.plan.deferred[result.index()] {
            self.flags = saved_flags;
            return Ok(());
        }
        if op == Binary::Band && self.pure_test[result.index()] {
            let cond = self.emit_test_band(x, y);
            self.flags = Some((result, cond));
            return Ok(());
        }
        if op == Binary::Iconcat {
            self.flags = saved_flags;
            self.raw(Reg::Rax, x);
            self.put(result, Reg::Rax);
            self.raw(Reg::Rax, y);
            self.put_high(result, Reg::Rax);
            return Ok(());
        }
        if ty.is_vector() {
            self.invalidate_spills();
            let res = self.vector_binary(op, (x, y), result);
            self.invalidate_spills();
            return res;
        }
        if ty.is_float() {
            return self.float_binary(op, x, y, result);
        }
        if ty == Type::I128 {
            self.invalidate_spills();
            let res = self.wide_binary(op, x, y, result);
            self.invalidate_spills();
            return res;
        }
        let dst = self.dst(result);
        let reg = self.int_binary(op, (x, y), ty, dst, result, saved_flags)?;
        self.put(result, reg);
        Ok(())
    }

    /// Computes `x op y` into `dst` for the operations that can, or into rax, and returns the register that holds it.
    fn int_binary(&mut self, op: Binary, (x, y): (Value, Value), ty: Type, dst: Reg, result: Value, saved_flags: Option<(Value, Cond)>) -> Result<Reg, String> {
        if matches!(op, Binary::Iadd | Binary::Isub)
            && ty == Type::I64
            && let Some(mode) = self.plan.sib[result.index()]
        {
            self.flags = saved_flags;
            let mem = self.sib_mem(mode, 0, Reg::Rcx, Some(dst));
            self.asm.lea(dst, mem);
            return Ok(dst);
        }
        if let Some(reg) = self.folded_product(op, (x, y), result) {
            return Ok(reg);
        }
        if matches!(op, Binary::Bor | Binary::Bxor | Binary::Iadd)
            && self.plan.elided[x.index()]
            && self.plan.elided[y.index()]
            && let Some((hi, lo, cl, l, r)) = super::location::shld_parts(self.func, &self.plan, ty, (x, y), |v| self.plan.elided[v.index()], |v| self.defs[v.index()])
        {
            if hi == lo {
                return Ok(self.shift_by(Binary::Rotl, (hi, cl), ty, dst, result));
            }
            let size = if ty == Type::I32 { Size::D } else { Size::Q };
            let is_shrd = self.reg_of(lo) == Some(dst);
            let src = if is_shrd { hi } else { lo };
            let held = self.reg_of(src).or_else(|| self.cached_beside(src, hi)).filter(|&reg| reg != dst);
            if is_shrd {
                let src_reg = held.unwrap_or_else(|| self.beside(dst, src, |_| {}));
                self.asm.shrd_imm(size, dst, src_reg, r);
            } else {
                let src_reg = match held {
                    Some(reg) => {
                        self.raw(dst, hi);
                        reg
                    }
                    None => self.beside(dst, src, |this| this.raw(dst, hi)),
                };
                self.asm.shld_imm(size, dst, src_reg, l);
            }
            return Ok(dst);
        }
        if let Some((_, _, left, other, right, invert)) = super::location::adc_plan(self.func, &self.plan, ty, op, (x, y), |v| self.plan.elided[v.index()], |v| self.defs[v.index()]) {
            self.carry_add(op, ty, dst, (left, other, right, invert))?;
            return Ok(dst);
        }
        if self.features.bmi1
            && let Some((_, src, start, len)) = super::location::bextr_parts(self.func, &self.plan, ty, op, (x, y), |v| self.plan.elided[v.index()], |v| self.defs[v.index()])
        {
            let size = if ty == Type::I32 { Size::D } else { Size::Q };
            let ctrl = i64::from(start) | (i64::from(len) << 8);
            let source = if (matches!(self.location(src), Location::Reg(_) | Location::Slot) && !self.plan.elided[src.index()]) || self.has_elided_mem(src) {
                self.operand(src)
            } else {
                self.raw(dst, src);
                dst.into()
            };
            let (source, held) = self.spare(Some(dst), src, source);
            self.asm.mov_imm(held, ctrl);
            self.clobber_reg(held);
            self.asm.bextr(size, dst, source, held);
            if self.plan.fused[result.index()] {
                self.flags = Some((result, Cond::Ne));
            }
            return Ok(dst);
        }
        if let Some(alu) = alu_of(op) {
            let size = if ty == Type::I32 { Size::D } else { Size::Q };
            let (x, y) = if !matches!(alu, Alu::Sub)
                && x != y
                && ((self.imm_sized(y, size).is_none()
                    && (self.reg_of(y) == Some(dst) || (self.has_elided_mem(x) && !self.has_elided_mem(y))))
                    || (self.imm_sized(x, size).is_some() && self.imm_sized(y, size).is_none()))
            {
                (y, x)
            } else {
                (x, y)
            };
            if matches!(alu, Alu::And)
                && self.features.bmi1
                && let Some((not_val, other)) = [(x, y), (y, x)].into_iter().find_map(|(a, b)| {
                    self.plan.elided[a.index()].then(|| self.defs[a.index()]).flatten().and_then(|inst| match self.func.insts[inst.index()].op {
                        Op::Unary(Unary::Bnot, inner) => Some((inner, b)),
                        _ => None,
                    })
                })
            {
                self.vex_gpr(not_val, other, dst, |asm, not_reg, src| asm.andn(size, dst, not_reg, src));
                if self.plan.fused[result.index()] {
                    self.flags = Some((result, Cond::Ne));
                }
                return Ok(dst);
            }
            if matches!(alu, Alu::And)
                && let Some((src_v, narrow)) = [(x, y), (y, x)].into_iter().find_map(|(v, m)| {
                    self.const_of(m).and_then(|mask| match mask {
                        0xFF => Some((v, Size::B)),
                        0xFFFF => Some((v, Size::W)),
                        0xFFFF_FFFF => Some((v, Size::D)),
                        _ => None,
                    })
                })
            {
                self.flags = saved_flags;
                let source = self.operand(src_v);
                self.asm.load(narrow, dst, source);
                return Ok(dst);
            }
            let disp = match alu {
                Alu::Add => self.imm_sized(y, size),
                Alu::Sub if size == Size::D => self.const_of(y).map(|c| (c as i32).wrapping_neg()),
                Alu::Sub => self.const_of(y).and_then(|c| c.checked_neg()).and_then(|c| i32::try_from(c).ok()),
                _ => None,
            };
            if size == Size::Q
                && !self.plan.fused[result.index()]
                && let Some(sym) = self.image_sym(x)
                && let Some(imm) = self.const_of(y).and_then(|c| if matches!(alu, Alu::Add) { Some(c) } else { c.checked_neg() }).and_then(|c| i32::try_from(c).ok())
            {
                self.flags = saved_flags;
                self.asm.lea(dst, Mem::rip(sym, imm));
                return Ok(dst);
            }
            if matches!(size, Size::D | Size::Q)
                && matches!(alu, Alu::Add | Alu::Sub)
                && !self.plan.fused[result.index()]
                && !(self.adc_sum[result.index()] && self.asm.held_carry().is_none())
                && let Some(rx) = self.reg_of(x).or_else(|| self.cached_spill(x)).filter(|&rx| (rx != dst || self.asm.held_carry().is_some()) && rx != Reg::Rsp)
            {
                if let Some(imm) = disp {
                    self.flags = saved_flags;
                    self.asm.lea_sized(size, dst, Mem::new(rx, imm));
                    return Ok(dst);
                }
                if matches!(alu, Alu::Add)
                    && let Some(ry) = self.reg_of(y).or_else(|| self.cached_spill(y)).filter(|&ry| ry != Reg::Rsp)
                {
                    self.flags = saved_flags;
                    self.asm.lea_sized(size, dst, Mem::sib(rx, ry, 0, 0));
                    return Ok(dst);
                }
            }
            if size == Size::Q && !self.plan.fused[result.index()] && self.imm_sized(y, size).is_none()
                && let Some(c) = self.const_of(y)
                && let Some((ext, bit)) = match alu {
                    Alu::Or if (c as u64).is_power_of_two() => Some((5, (c as u64).trailing_zeros() as u8)),
                    Alu::Xor if (c as u64).is_power_of_two() => Some((7, (c as u64).trailing_zeros() as u8)),
                    Alu::And if (!(c as u64)).is_power_of_two() => Some((6, (!(c as u64)).trailing_zeros() as u8)),
                    _ => None,
                }
            {
                self.raw(dst, x);
                self.asm.bt_imm(Size::Q, ext, dst, bit);
                return Ok(dst);
            }
            if self.imm_sized(y, size).is_none() && self.reg_of(y) == Some(dst) && x != y {
                self.asm.mov(Reg::Rax, dst);
                self.clobber_reg(Reg::Rax);
                self.raw(dst, x);
                self.asm.alu_sized(size, alu, dst, Reg::Rax);
            } else {
                self.raw(dst, x);
                match self.imm_sized(y, size) {
                    Some(imm) => self.asm.alu_imm_sized(size, alu, dst, imm),
                    None => {
                        let source = self.operand(y);
                        self.asm.alu_sized(size, alu, dst, source);
                    }
                }
            }
            if matches!(alu, Alu::Add | Alu::Sub) && matches!(ty, Type::I32 | Type::I64) && (self.overflow_read[result.index()] || self.adc_sum[result.index()]) {
                self.asm.hold_carry(super::wide::overflow_tag(result));
            }
            if self.plan.fused[result.index()] {
                self.flags = Some((result, Cond::Ne));
            }
            return Ok(dst);
        }
        match op {
            Binary::Imul => {
                self.multiply(dst, x, y, saved_flags);
                return Ok(dst);
            }
            Binary::Udiv | Binary::Urem if self.divide_by_power(op, (x, y), dst) => return Ok(dst),
            Binary::Sdiv | Binary::Srem | Binary::Udiv | Binary::Urem => return Ok(self.divide(op, x, y)),
            Binary::Umulhi | Binary::Smulhi => return Ok(self.multiply_high(op == Binary::Smulhi, x, y, ty.bits() as u8)),
            Binary::Smin | Binary::Smax | Binary::Umin | Binary::Umax => {
                let signed = matches!(op, Binary::Smin | Binary::Smax);
                self.get(Reg::Rax, x, Ext::of(signed));
                self.get(Reg::Rcx, y, Ext::of(signed));
                self.asm.alu(Alu::Cmp, Reg::Rax, Reg::Rcx);
                let take_y = match op {
                    Binary::Smin => Cond::G,
                    Binary::Smax => Cond::L,
                    Binary::Umin => Cond::A,
                    _ => Cond::B,
                };
                self.asm.cmov(take_y, Reg::Rax, Reg::Rcx);
            }
            Binary::SaddSat | Binary::UaddSat | Binary::SsubSat | Binary::UsubSat => {
                self.raw(Reg::Rcx, y);
                self.raw(dst, x);
                self.emit_sat_op(op, Size::of(ty.bytes()), dst, Reg::Rcx);
                return Ok(dst);
            }
            Binary::AvgRound => {
                self.get(Reg::Rcx, y, Ext::Zero);
                self.get(dst, x, Ext::Zero);
                self.asm.mov(Reg::Rdx, dst);
                self.asm.alu(Alu::Or, dst, Reg::Rcx);
                self.asm.alu(Alu::Xor, Reg::Rdx, Reg::Rcx);
                self.asm.shift_imm(Size::Q, Shift::Shr, Reg::Rdx, 1);
                self.asm.alu(Alu::Sub, dst, Reg::Rdx);
                self.clobber_reg(Reg::Rdx);
                return Ok(dst);
            }
            Binary::Ishl | Binary::Ushr | Binary::Sshr | Binary::Rotl | Binary::Rotr => return Ok(self.shift_by(op, (x, y), ty, dst, result)),
            other => return Err(format!("{other:?} of an integer")),
        }
        Ok(Reg::Rax)
    }

    /// Multiplies `x` by `y` into `dst`, with a shift, `lea`, or an immediate when either factor is a constant.
    fn multiply(&mut self, dst: Reg, x: Value, y: Value, saved_flags: Option<(Value, Cond)>) {
        let size = if self.ty(x).bits() <= 32 { Size::D } else { Size::Q };
        let (x, c_val, raw_imm) = match (self.const_of(x), self.const_of(y)) {
            (_, Some(imm)) => (x, y, Some(imm)),
            (Some(imm), None) => (y, x, Some(imm)),
            (None, None) => (x, y, None),
        };
        if let Some(raw) = raw_imm {
            let uimm = if size == Size::D { u64::from(raw as u32) } else { raw as u64 };
            if uimm == 0 {
                self.flags = saved_flags;
                self.asm.mov_imm(dst, 0);
                return;
            }
            if uimm == 1 {
                self.flags = saved_flags;
                if size == Size::D { self.get(dst, x, Ext::Zero); } else { self.raw(dst, x); }
                return;
            }
            let shift = uimm.trailing_zeros() as u8;
            let odd = uimm >> shift;
            if odd == 1 {
                self.raw(dst, x);
                self.asm.shift_imm(size, Shift::Shl, dst, shift);
                return;
            }
            if matches!(size, Size::D | Size::Q) && matches!(odd, 3 | 5 | 9) {
                if shift == 0 {
                    self.flags = saved_flags;
                }
                let rx = match self.reg_of(x).or_else(|| self.cached_spill(x)) {
                    Some(r) if r != Reg::Rsp => r,
                    _ => { self.raw(dst, x); dst }
                };
                self.asm.lea_sized(size, dst, Mem::sib(rx, rx, (odd - 1).trailing_zeros() as u8, 0));
                if shift > 0 {
                    self.asm.shift_imm(size, Shift::Shl, dst, shift);
                }
                return;
            }
            if let Some(imm) = if size == Size::D { Some(raw as i32) } else { i32::try_from(raw).ok() } {
                let source = self.operand(x);
                self.asm.imul_imm_sized(size, dst, source, imm);
                return;
            }
        }
        let (x, y) = if (self.reg_of(c_val) == Some(dst) || (self.has_elided_mem(x) && !self.has_elided_mem(c_val))) && x != c_val { (c_val, x) } else { (x, c_val) };
        self.raw(dst, x);
        let source = self.operand(y);
        self.asm.imul_sized(size, dst, source);
    }

    /// Shifts or rotates `x` by `y`: into `dst` with BMI2 or an immediate `y`, else into rax by cl; returns the register that holds the result.
    fn shift_by(&mut self, op: Binary, (x, y): (Value, Value), ty: Type, dst: Reg, result: Value) -> Reg {
        let (ext, shift) = match op {
            Binary::Ishl => (Ext::Zero, Shift::Shl),
            Binary::Ushr => (Ext::Zero, Shift::Shr),
            Binary::Sshr => (Ext::Sign, Shift::Sar),
            Binary::Rotl => (Ext::Zero, Shift::Rol),
            _ => (Ext::Zero, Shift::Ror),
        };
        let size = match op {
            Binary::Rotl | Binary::Rotr => Size::of(ty.bytes()),
            _ if ty == Type::I32 => Size::D,
            _ => Size::Q,
        };
        let need_ext = op != Binary::Ishl && ty != Type::I32;
        let load_x = |this: &mut Self, reg, flag| if flag { this.get(reg, x, ext); } else { this.raw(reg, x); };
        if let Some(imm) = self.const_of(y) {
            let count = (imm & i64::from(ty.bits() - 1)) as u8;
            if self.features.bmi2 && count != 0 && matches!(op, Binary::Rotl | Binary::Rotr) && matches!(ty, Type::I32 | Type::I64) {
                let imm = if op == Binary::Rotl { ty.bits() as u8 - count } else { count };
                let source = self.operand(x);
                self.asm.rorx(size, dst, source, imm);
                return dst;
            }
            load_x(self, dst, need_ext || (count == 0 && ty == Type::I32 && matches!(op, Binary::Ushr | Binary::Rotl | Binary::Rotr) && !self.plan.zero_ext[x.index()]));
            if count != 0 {
                self.asm.shift_imm(size, shift, dst, count);
                if matches!(op, Binary::Ishl | Binary::Ushr | Binary::Sshr) && self.plan.fused[result.index()] {
                    self.flags = Some((result, Cond::Ne));
                }
            }
            return dst;
        }
        let y = super::location::unwrap_shift_mask(self.func, &self.plan, &self.defs, y, result, ty.bits());
        if self.features.bmi2 && matches!(op, Binary::Ishl | Binary::Ushr | Binary::Sshr) {
            if matches!(ty, Type::I8 | Type::I16) {
                let count = self.beside(dst, y, |this| load_x(this, dst, need_ext));
                self.asm.alu_imm32(Alu::And, count, ty.bits() as i32 - 1);
                self.clobber_reg(count);
                self.asm.shiftx(Size::Q, shift, dst, dst, count);
                return dst;
            }
            self.vex_gpr(y, x, dst, |asm, cnt, src| asm.shiftx(size, shift, dst, src, cnt));
            return dst;
        }
        load_x(self, Reg::Rax, need_ext);
        self.raw(Reg::Rcx, y);
        if matches!(ty, Type::I8 | Type::I16) && !matches!(op, Binary::Rotl | Binary::Rotr) {
            self.asm.alu_imm32(Alu::And, Reg::Rcx, ty.bits() as i32 - 1);
            self.clobber_reg(Reg::Rcx);
        }
        self.asm.shift(size, shift, Reg::Rax);
        self.clobber_reg(Reg::Rax);
        Reg::Rax
    }

    fn vex_gpr(&mut self, vvvv: Value, rm_val: Value, dst: Reg, emit: impl FnOnce(&mut Asm, Reg, Rm)) {
        if let Some(reg) = self.reg_of(vvvv).or_else(|| self.cached_beside(vvvv, rm_val)) {
            let source = self.operand(rm_val);
            emit(&mut self.asm, reg, source);
        } else if self.has_elided_mem(rm_val) {
            let source = self.operand(rm_val);
            self.raw(Reg::Rax, vvvv);
            emit(&mut self.asm, Reg::Rax, source);
        } else if !matches!(self.location(rm_val), Location::Reg(_) | Location::Slot) {
            let count = self.beside(dst, vvvv, |this| this.raw(dst, rm_val));
            emit(&mut self.asm, count, dst.into());
        } else {
            self.raw(Reg::Rax, vvvv);
            let source = self.operand(rm_val);
            emit(&mut self.asm, Reg::Rax, source);
        }
    }

    /// The cached spill register of `value` when it survives reading `other`: not the rcx that reading `other` may write.
    pub(super) fn cached_beside(&self, value: Value, other: Value) -> Option<Reg> {
        self.cached_spill(value).filter(|&reg| !(reg == Reg::Rcx && self.reads_through_rcx(other)))
    }

    pub(super) fn elided_band_cmp(&self, x: Value, y: Value) -> Option<(Value, Value)> {
        let is_zero = |v: Value| self.const_of(v).is_some_and(|imm| crate::fold::extend(imm, self.ty(v), false) == 0);
        let masked = is_zero(y).then_some(x).or_else(|| is_zero(x).then_some(y))?;
        if !self.plan.elided[masked.index()] {
            return None;
        }
        let inst = self.defs[masked.index()]?;
        let Op::Binary(Binary::Band, a, b) = self.func.insts[inst.index()].op else {
            return None;
        };
        Some((a, b))
    }

    pub(super) fn emit_cmp_pair(&mut self, size: Size, x: Value, y: Option<Value>, zero_test: bool, unsigned: bool) {
        let (size, imm) = match y {
            Some(yv) => match self.imm_sized(yv, size) {
                Some(imm) => (size, Some(imm)),
                None if unsigned && size == Size::Q && self.narrow_u32_cmp(x, yv) => {
                    (Size::D, self.const_of(yv).map(|c| c as u32 as i32))
                }
                None => (size, None),
            },
            None => (size, Some(1)),
        };
        if (matches!(self.location(x), Location::Slot) && self.cached_spill(x).is_none() || self.has_elided_mem(x))
            && let Some(imm) = imm
        {
            let rm = self.operand(x);
            self.asm.alu_imm_sized(size, Alu::Cmp, rm, imm);
            return;
        }
        let left = self.reg_of(x).or_else(|| y.filter(|&yv| imm.is_none() && self.imm_sized(yv, size).is_none()).map_or_else(|| self.cached_spill(x), |yv| self.cached_beside(x, yv))).unwrap_or_else(|| {
            self.raw(Reg::Rax, x);
            Reg::Rax
        });
        match y {
            Some(yv) if zero_test && self.const_of(yv).is_some_and(|c| crate::fold::extend(c, self.ty(yv), false) == 0) => {
                self.asm.test_sized(size, left, left);
            }
            Some(_) if size == Size::D && imm.is_some() && self.ty(x) == Type::I64 => {
                self.asm.alu_imm_sized(Size::D, Alu::Cmp, left, imm.expect("invariant: checked Some"));
            }
            Some(yv) => self.alu_at(size, Alu::Cmp, left, yv),
            None => self.asm.alu_imm_sized(size, Alu::Cmp, left, 1),
        }
    }

    fn narrow_u32_cmp(&self, x: Value, y: Value) -> bool {
        let Some(c) = self.const_of(y) else { return false };
        if u32::try_from(c as u64).is_err() || i32::try_from(c).is_ok() { return false; }
        let Some(inst) = self.defs[x.index()] else { return false };
        match self.func.insts[inst.index()].op {
            Op::Convert(crate::ir::Convert::Uextend, a) => self.ty(a).bits() <= 32,
            Op::Binary(Binary::Ushr, _, sh) => self.const_of(sh).is_some_and(|k| (k & 63) >= 32),
            Op::Binary(Binary::Band, a, b) => [a, b].into_iter().any(|v| self.const_of(v).is_some_and(|m| u32::try_from(m as u64).is_ok())),
            _ => false,
        }
    }

    pub(super) fn icmp(&mut self, cc: IntCC, x: Value, y: Value, result: Value) -> Result<(), String> {
        let live_flags = self.flags.take();
        if self.is_static(result) {
            return Ok(());
        }
        if self.ty(x).is_vector() {
            self.invalidate_spills();
            let res = self.vector_icmp(cc, x, y, result);
            self.invalidate_spills();
            return res;
        }
        let flags = self.cmp_flags(cc, x, y, live_flags);
        if self.plan.fused[result.index()] {
            self.flags = Some((result, flags));
            return Ok(());
        }
        let dst = self.dst(result);
        self.asm.setcc(flags, dst);
        self.asm.load(Size::B, dst, dst);
        self.put(result, dst);
        if self.ty(x) != Type::I128 {
            self.flags = Some((result, flags));
        }
        Ok(())
    }

    /// Sets `result` to whether `x op y` fits its type, from the carry or overflow flag of the operation at that width; computes the paired operation's result with the same instruction.
    pub(super) fn fits(&mut self, op: Binary, signed: bool, (x, y): (Value, Value), result: Value) -> Result<(), String> {
        self.flags = None;
        let pair = self.plan.pairs[result.index()];
        if self.is_static(result) && pair.is_none() {
            return Ok(());
        }
        let size = Size::of(self.ty(x).bytes());
        let reg = match op {
            Binary::Iadd | Binary::Isub => {
                let alu = if op == Binary::Iadd { Alu::Add } else { Alu::Sub };
                let reg = pair.and_then(|sum| self.reg_of(sum)).filter(|reg| Some(*reg) != self.reg_of(y)).unwrap_or(Reg::Rax);
                self.raw(reg, x);
                self.alu_at(size, alu, reg, y);
                self.clobber_reg(reg);
                reg
            }
            Binary::Imul => {
                self.raw(Reg::Rax, x);
                self.raw(Reg::Rcx, y);
                self.asm.group3_sized(size, if signed { Group3::Imul } else { Group3::Mul }, Reg::Rcx);
                self.clobber_regs(&[Reg::Rax, Reg::Rdx]);
                Reg::Rax
            }
            other => return Err(format!("{other:?} has no overflow flag")),
        };
        if let Some(sum) = pair {
            self.put(sum, reg);
        }
        let ok = if signed { Cond::No } else { Cond::Ae };
        if self.plan.fused[result.index()] {
            self.flags = Some((result, ok));
            return Ok(());
        }
        let dst = self.dst(result);
        self.asm.setcc(ok, dst);
        self.asm.load(Size::B, dst, dst);
        self.put(result, dst);
        Ok(())
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
                let exact = saved_flags.is_some_and(|(value, _)| value == x) && super::location::narrows_exactly(self.func, &self.plan, &|v| self.defs[v.index()], x, self.ty(result));
                self.flags = saved_flags.map(|(value, cond)| (if exact { result } else { value }, cond));
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

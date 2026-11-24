use super::alu_of;
use crate::alloc::Location;
use crate::frame::Ext;
use crate::ir::{Binary, Op, Type, Unary, Value};
use crate::x64::Emit;
use crate::x64::asm::{Alu, Asm, Cond, Mem, Reg, Rm, Shift, Size};

impl Emit<'_> {
    pub(in crate::x64) fn binary(
        &mut self,
        op: Binary,
        x: Value,
        y: Value,
        result: Value,
    ) -> Result<(), String> {
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

    fn int_binary(
        &mut self,
        op: Binary,
        (x, y): (Value, Value),
        ty: Type,
        dst: Reg,
        result: Value,
        saved_flags: Option<(Value, Cond)>,
    ) -> Result<Reg, String> {
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
        if let Some(reg) = self.try_shld(op, (x, y), ty, dst, result) {
            return Ok(reg);
        }
        if let Some((_, _, left, other, right, invert)) = crate::x64::location::adc_plan(
            self.func,
            &self.plan,
            ty,
            op,
            (x, y),
            |v| self.plan.elided[v.index()],
            |v| self.defs[v.index()],
        ) {
            self.carry_add(op, ty, dst, (left, other, right, invert))?;
            return Ok(dst);
        }
        if self.try_bextr(op, (x, y), ty, dst, result) {
            return Ok(dst);
        }
        if let Some(alu) = alu_of(op) {
            return Ok(self.int_alu(alu, (x, y), ty, dst, result, saved_flags));
        }
        match op {
            Binary::Imul => {
                self.multiply(dst, x, y, saved_flags);
                Ok(dst)
            }
            Binary::Udiv | Binary::Urem if self.divide_by_power(op, (x, y), dst) => Ok(dst),
            Binary::Sdiv | Binary::Srem | Binary::Udiv | Binary::Urem => Ok(self.divide(op, x, y)),
            Binary::Umulhi | Binary::Smulhi => {
                Ok(self.multiply_high(op == Binary::Smulhi, x, y, ty.bits() as u8))
            }
            Binary::Smin | Binary::Smax | Binary::Umin | Binary::Umax => {
                Ok(self.int_minmax(op, x, y))
            }
            Binary::SaddSat | Binary::UaddSat | Binary::SsubSat | Binary::UsubSat => {
                self.raw(Reg::Rcx, y);
                self.raw(dst, x);
                self.emit_sat_op(op, Size::of(ty.bytes()), dst, Reg::Rcx);
                Ok(dst)
            }
            Binary::AvgRound => {
                self.int_avg_round(dst, x, y);
                Ok(dst)
            }
            Binary::Ishl | Binary::Ushr | Binary::Sshr | Binary::Rotl | Binary::Rotr => {
                Ok(self.shift_by(op, (x, y), ty, dst, result))
            }
            other => Err(format!("{other:?} of an integer")),
        }
    }

    fn try_shld(
        &mut self,
        op: Binary,
        (x, y): (Value, Value),
        ty: Type,
        dst: Reg,
        result: Value,
    ) -> Option<Reg> {
        if !matches!(op, Binary::Bor | Binary::Bxor | Binary::Iadd)
            || !self.plan.elided[x.index()]
            || !self.plan.elided[y.index()]
        {
            return None;
        }
        let (hi, lo, cl, l, r) = crate::x64::location::shld_parts(
            self.func,
            &self.plan,
            ty,
            (x, y),
            |v| self.plan.elided[v.index()],
            |v| self.defs[v.index()],
        )?;
        if hi == lo {
            return Some(self.shift_by(Binary::Rotl, (hi, cl), ty, dst, result));
        }
        let size = if ty == Type::I32 { Size::D } else { Size::Q };
        let is_shrd = self.reg_of(lo) == Some(dst);
        let src = if is_shrd { hi } else { lo };
        let held = self
            .reg_of(src)
            .or_else(|| self.cached_beside(src, hi))
            .filter(|&reg| reg != dst);
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
        Some(dst)
    }

    fn try_bextr(
        &mut self,
        op: Binary,
        (x, y): (Value, Value),
        ty: Type,
        dst: Reg,
        result: Value,
    ) -> bool {
        if !self.features.bmi1 {
            return false;
        }
        let Some((_, src, start, len)) = crate::x64::location::bextr_parts(
            self.func,
            &self.plan,
            ty,
            op,
            (x, y),
            |v| self.plan.elided[v.index()],
            |v| self.defs[v.index()],
        ) else {
            return false;
        };
        if ty == Type::I64 && start > 0 && u32::from(start) + u32::from(len) == 32 {
            self.raw(dst, src);
            self.asm.shift_imm(Size::D, Shift::Shr, dst, start);
            if self.plan.fused[result.index()] {
                self.flags = Some((result, Cond::Ne));
            }
            return true;
        }
        let size = if ty == Type::I32 { Size::D } else { Size::Q };
        let ctrl = i64::from(start) | (i64::from(len) << 8);
        let source = if (matches!(self.location(src), Location::Reg(_) | Location::Slot)
            && !self.plan.elided[src.index()])
            || self.has_elided_mem(src)
        {
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
        true
    }

    fn order_alu_operands(
        &self,
        alu: Alu,
        size: Size,
        dst: Reg,
        (x, y): (Value, Value),
    ) -> (Value, Value) {
        if !matches!(alu, Alu::Sub)
            && x != y
            && ((self.imm_sized(y, size).is_none()
                && (self.reg_of(y) == Some(dst)
                    || (self.has_elided_mem(x) && !self.has_elided_mem(y))))
                || (self.imm_sized(x, size).is_some() && self.imm_sized(y, size).is_none()))
        {
            (y, x)
        } else {
            (x, y)
        }
    }

    fn try_andn(
        &mut self,
        alu: Alu,
        size: Size,
        (x, y): (Value, Value),
        dst: Reg,
        result: Value,
    ) -> bool {
        if !matches!(alu, Alu::And) || !self.features.bmi1 {
            return false;
        }
        let Some((not_val, other)) = [(x, y), (y, x)].into_iter().find_map(|(a, b)| {
            self.plan.elided[a.index()]
                .then(|| self.defs[a.index()])
                .flatten()
                .and_then(|inst| match self.func.insts[inst.index()].op {
                    Op::Unary(Unary::Bnot, inner) => Some((inner, b)),
                    _ => None,
                })
        }) else {
            return false;
        };
        self.vex_gpr(not_val, other, dst, |asm, not_reg, src| {
            asm.andn(size, dst, not_reg, src)
        });
        if self.plan.fused[result.index()] {
            self.flags = Some((result, Cond::Ne));
        }
        true
    }

    fn try_mask_load(
        &mut self,
        alu: Alu,
        (x, y): (Value, Value),
        dst: Reg,
        saved_flags: Option<(Value, Cond)>,
    ) -> bool {
        if !matches!(alu, Alu::And) {
            return false;
        }
        let Some((src_v, narrow)) = [(x, y), (y, x)].into_iter().find_map(|(v, m)| {
            self.const_of(m).and_then(|mask| match mask {
                0xFF => Some((v, Size::B)),
                0xFFFF => Some((v, Size::W)),
                0xFFFF_FFFF => Some((v, Size::D)),
                _ => None,
            })
        }) else {
            return false;
        };
        self.flags = saved_flags;
        let source = self.operand(src_v);
        self.asm.load(narrow, dst, source);
        true
    }

    fn try_alu_lea(
        &mut self,
        alu: Alu,
        size: Size,
        (x, y): (Value, Value),
        dst: Reg,
        result: Value,
        saved_flags: Option<(Value, Cond)>,
    ) -> bool {
        let disp = match alu {
            Alu::Add => self.imm_sized(y, size),
            Alu::Sub if size == Size::D => self.const_of(y).map(|c| (c as i32).wrapping_neg()),
            Alu::Sub => self
                .const_of(y)
                .and_then(|c| c.checked_neg())
                .and_then(|c| i32::try_from(c).ok()),
            _ => None,
        };
        if size == Size::Q
            && !self.plan.fused[result.index()]
            && let Some(sym) = self.image_sym(x)
            && let Some(imm) = self
                .const_of(y)
                .and_then(|c| {
                    if matches!(alu, Alu::Add) {
                        Some(c)
                    } else {
                        c.checked_neg()
                    }
                })
                .and_then(|c| i32::try_from(c).ok())
        {
            self.flags = saved_flags;
            self.asm.lea(dst, Mem::rip(sym, imm));
            return true;
        }
        if matches!(size, Size::D | Size::Q)
            && matches!(alu, Alu::Add | Alu::Sub)
            && !self.plan.fused[result.index()]
            && !(self.adc_sum[result.index()] && self.asm.held_carry().is_none())
            && let Some(rx) = self
                .reg_of(x)
                .or_else(|| self.cached_spill(x))
                .filter(|&rx| (rx != dst || self.asm.held_carry().is_some()) && rx != Reg::Rsp)
        {
            if let Some(imm) = disp {
                self.flags = saved_flags;
                self.asm.lea_sized(size, dst, Mem::new(rx, imm));
                return true;
            }
            if matches!(alu, Alu::Add)
                && let Some(ry) = self
                    .reg_of(y)
                    .or_else(|| self.cached_spill(y))
                    .filter(|&ry| ry != Reg::Rsp)
            {
                self.flags = saved_flags;
                self.asm.lea_sized(size, dst, Mem::sib(rx, ry, 0, 0));
                return true;
            }
        }
        false
    }

    fn try_bt_imm(
        &mut self,
        alu: Alu,
        size: Size,
        (x, y): (Value, Value),
        dst: Reg,
        result: Value,
    ) -> bool {
        if size != Size::Q || self.plan.fused[result.index()] || self.imm_sized(y, size).is_some() {
            return false;
        }
        let Some(c) = self.const_of(y) else {
            return false;
        };
        let (ext, bit) = match alu {
            Alu::Or if (c as u64).is_power_of_two() => (5, (c as u64).trailing_zeros() as u8),
            Alu::Xor if (c as u64).is_power_of_two() => (7, (c as u64).trailing_zeros() as u8),
            Alu::And if (!(c as u64)).is_power_of_two() => {
                (6, (!(c as u64)).trailing_zeros() as u8)
            }
            _ => return false,
        };
        self.raw(dst, x);
        self.asm.bt_imm(Size::Q, ext, dst, bit);
        true
    }

    fn int_alu(
        &mut self,
        alu: Alu,
        (x, y): (Value, Value),
        ty: Type,
        dst: Reg,
        result: Value,
        saved_flags: Option<(Value, Cond)>,
    ) -> Reg {
        let size = if ty == Type::I32 { Size::D } else { Size::Q };
        let (x, y) = self.order_alu_operands(alu, size, dst, (x, y));
        if self.try_andn(alu, size, (x, y), dst, result)
            || self.try_mask_load(alu, (x, y), dst, saved_flags)
            || self.try_alu_lea(alu, size, (x, y), dst, result, saved_flags)
            || self.try_bt_imm(alu, size, (x, y), dst, result)
        {
            return dst;
        }
        if self.imm_sized(y, size).is_none() && self.reg_of(y) == Some(dst) && x != y {
            self.asm.mov(Reg::Rax, dst);
            self.clobber_reg(Reg::Rax);
            self.raw(dst, x);
            self.asm.alu_sized(size, alu, dst, Reg::Rax);
        } else {
            self.raw(dst, x);
            self.alu_at(size, alu, dst, y);
        }
        if matches!(alu, Alu::Add | Alu::Sub)
            && matches!(ty, Type::I32 | Type::I64)
            && (self.overflow_read[result.index()] || self.adc_sum[result.index()])
        {
            self.asm.hold_carry(crate::x64::wide::overflow_tag(result));
        }
        if self.plan.fused[result.index()] {
            self.flags = Some((result, Cond::Ne));
        }
        dst
    }

    fn int_minmax(&mut self, op: Binary, x: Value, y: Value) -> Reg {
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
        Reg::Rax
    }

    fn int_avg_round(&mut self, dst: Reg, x: Value, y: Value) {
        self.get(Reg::Rcx, y, Ext::Zero);
        self.get(dst, x, Ext::Zero);
        self.asm.mov(Reg::Rdx, dst);
        self.asm.alu(Alu::Or, dst, Reg::Rcx);
        self.asm.alu(Alu::Xor, Reg::Rdx, Reg::Rcx);
        self.asm.shift_imm(Size::Q, Shift::Shr, Reg::Rdx, 1);
        self.asm.alu(Alu::Sub, dst, Reg::Rdx);
        self.clobber_reg(Reg::Rdx);
    }

    fn multiply(&mut self, dst: Reg, x: Value, y: Value, saved_flags: Option<(Value, Cond)>) {
        let size = if self.ty(x).bits() <= 32 {
            Size::D
        } else {
            Size::Q
        };
        let (x, c_val, raw_imm) = match (self.const_of(x), self.const_of(y)) {
            (_, Some(imm)) => (x, y, Some(imm)),
            (Some(imm), None) => (y, x, Some(imm)),
            (None, None) => (x, y, None),
        };
        if let Some(raw) = raw_imm {
            let uimm = if size == Size::D {
                u64::from(raw as u32)
            } else {
                raw as u64
            };
            if uimm == 0 {
                self.flags = saved_flags;
                self.asm.mov_imm(dst, 0);
                return;
            }
            if uimm == 1 {
                self.flags = saved_flags;
                if size == Size::D {
                    self.get(dst, x, Ext::Zero);
                } else {
                    self.raw(dst, x);
                }
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
                    _ => {
                        self.raw(dst, x);
                        dst
                    }
                };
                self.asm.lea_sized(
                    size,
                    dst,
                    Mem::sib(rx, rx, (odd - 1).trailing_zeros() as u8, 0),
                );
                if shift > 0 {
                    self.asm.shift_imm(size, Shift::Shl, dst, shift);
                }
                return;
            }
            if let Some(imm) = if size == Size::D {
                Some(raw as i32)
            } else {
                i32::try_from(raw).ok()
            } {
                let source = self.operand(x);
                self.asm.imul_imm_sized(size, dst, source, imm);
                return;
            }
        }
        let (x, y) = if (self.reg_of(c_val) == Some(dst)
            || (self.has_elided_mem(x) && !self.has_elided_mem(c_val)))
            && x != c_val
        {
            (c_val, x)
        } else {
            (x, c_val)
        };
        self.raw(dst, x);
        let source = self.operand(y);
        self.asm.imul_sized(size, dst, source);
    }

    fn shift_by(
        &mut self,
        op: Binary,
        (x, y): (Value, Value),
        ty: Type,
        dst: Reg,
        result: Value,
    ) -> Reg {
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
        let load_x = |this: &mut Self, reg, flag| {
            if flag {
                this.get(reg, x, ext);
            } else {
                this.raw(reg, x);
            }
        };
        if let Some(imm) = self.const_of(y) {
            let count = (imm & i64::from(ty.bits() - 1)) as u8;
            if self.features.bmi2
                && count != 0
                && matches!(op, Binary::Rotl | Binary::Rotr)
                && matches!(ty, Type::I32 | Type::I64)
            {
                let imm = if op == Binary::Rotl {
                    ty.bits() as u8 - count
                } else {
                    count
                };
                let source = self.operand(x);
                self.asm.rorx(size, dst, source, imm);
                return dst;
            }
            load_x(
                self,
                dst,
                need_ext
                    || (count == 0
                        && ty == Type::I32
                        && matches!(op, Binary::Ushr | Binary::Rotl | Binary::Rotr)
                        && !self.plan.zero_ext[x.index()]),
            );
            if count != 0 {
                self.asm.shift_imm(size, shift, dst, count);
                if matches!(op, Binary::Ishl | Binary::Ushr | Binary::Sshr)
                    && self.plan.fused[result.index()]
                {
                    self.flags = Some((result, Cond::Ne));
                }
            }
            return dst;
        }
        let y = crate::x64::location::unwrap_shift_mask(
            self.func,
            &self.plan,
            &self.defs,
            y,
            result,
            ty.bits(),
        );
        if self.features.bmi2 && matches!(op, Binary::Ishl | Binary::Ushr | Binary::Sshr) {
            if matches!(ty, Type::I8 | Type::I16) {
                let count = self.beside(dst, y, |this| load_x(this, dst, need_ext));
                self.asm.alu_imm32(Alu::And, count, ty.bits() as i32 - 1);
                self.clobber_reg(count);
                self.asm.shiftx(Size::Q, shift, dst, dst, count);
                return dst;
            }
            self.vex_gpr(y, x, dst, |asm, cnt, src| {
                asm.shiftx(size, shift, dst, src, cnt)
            });
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

    fn vex_gpr(
        &mut self,
        vvvv: Value,
        rm_val: Value,
        dst: Reg,
        emit: impl FnOnce(&mut Asm, Reg, Rm),
    ) {
        if let Some(reg) = self
            .reg_of(vvvv)
            .or_else(|| self.cached_beside(vvvv, rm_val))
        {
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
}

use super::super::asm::{Alu, Cond, Reg, Shift, Size, ZR};
use super::super::{A, C, D, Emit, IP, T};
use super::{alu_of, encode_bitmask_imm};
use crate::frame::Ext;
use crate::ir::{Binary, Op, Type, Value};

impl Emit<'_> {
    pub(in crate::arm64) fn binary(
        &mut self,
        op: Binary,
        x: Value,
        y: Value,
        result: Value,
    ) -> Result<(), String> {
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
        if ty.is_vector() {
            self.invalidate_spills();
            return self.vec_binary(op, x, y, result);
        }
        if ty.is_float() {
            self.invalidate_spills();
            return self.float_binary(op, x, y, result);
        }
        if ty == Type::I128 {
            self.invalidate_spills();
            return self.wide_binary(op, x, y, result);
        }
        if matches!(op, Binary::Band | Binary::Bor) && self.try_emit_ccmp(op, (x, y), result) {
            return Ok(());
        }
        let dst = self.dst(result);
        self.int_binary(op, (x, y), ty, dst)?;
        self.put(result, dst);
        Ok(())
    }

    fn emit_bit_insert(&mut self, ty: Type, dst: Reg, (a, b, lsb, width): (Value, Value, u8, u8)) {
        let size = if ty == Type::I64 { Size::X } else { Size::W };
        let reg_bits: u8 = if size == Size::X { 64 } else { 32 };
        let b_src = self.peel_uextend(b);
        let rhs = if self.plan.reg_of(a) == Some(dst) {
            self.bits(b_src, C)
        } else if self.plan.reg_of(b_src) == Some(dst) {
            self.asm.mov(C, dst);
            self.clobber_reg(C);
            self.raw(dst, a);
            C
        } else {
            self.raw(dst, a);
            self.bits(b_src, C)
        };
        let base = if size == Size::X {
            0xB340_0000
        } else {
            0x3300_0000
        };
        self.asm.emit_gp(
            dst,
            base | u32::from((reg_bits - lsb) % reg_bits) << 16
                | u32::from(width - 1) << 10
                | u32::from(rhs.0) << 5
                | u32::from(dst.0),
        );
    }

    fn try_alu_bit_or_cinc(
        &mut self,
        op: Binary,
        alu: Alu,
        (x, y): (Value, Value),
        (ty, size): (Type, Size),
        dst: Reg,
    ) -> bool {
        if matches!(op, Binary::Bor | Binary::Bxor | Binary::Iadd)
            && self.plan.elided[x.index()]
            && self.plan.elided[y.index()]
            && let Some((hi, lo, kr)) =
                super::super::fuse::extr_parts(self.func, &self.plan, ty, (x, y), &|v| {
                    self.defs[v.index()]
                })
        {
            let (l, r) = self.bits_pair(self.peel_uextend(hi), self.peel_uextend(lo));
            self.asm.extr(size, dst, l, r, kr);
            return true;
        }
        if matches!(op, Binary::Bor | Binary::Bxor | Binary::Iadd)
            && let Some(bfi) =
                super::super::fuse::bfi_parts(self.func, &self.plan, ty, (x, y), &|v| {
                    self.defs[v.index()]
                })
        {
            self.emit_bit_insert(ty, dst, bfi);
            return true;
        }
        if alu == Alu::Add
            && matches!(ty, Type::I32 | Type::I64)
            && let Some((other, (cc, a, b))) = self
                .cinc_operand(y)
                .map(|cmp| (x, cmp))
                .or_else(|| self.cinc_operand(x).map(|cmp| (y, cmp)))
        {
            let cond = self.compare(cc, a, b);
            let reg = self.bits(other, A);
            self.asm.csinc(size, cond.invert(), dst, reg, reg);
            return true;
        }
        if alu == Alu::Sub
            && matches!(ty, Type::I32 | Type::I64)
            && self.const_imm(x) == Some(0)
            && let Some((cc, a, b)) = self.cinc_operand(y)
        {
            let cond = self.compare(cc, a, b);
            self.asm.csinv(size, cond.invert(), dst, ZR, ZR);
            return true;
        }
        false
    }

    fn try_alu_ext(
        &mut self,
        alu: Alu,
        (x, y): (Value, Value),
        (ty, size): (Type, Size),
        dst: Reg,
    ) -> bool {
        if !matches!(alu, Alu::Add | Alu::Sub) || !matches!(ty, Type::I32 | Type::I64) {
            return false;
        }
        let ext_of = |v: Value| {
            super::super::fuse::ext_parts(self.func, &self.plan, ty, v, |d| self.defs[d.index()])
        };
        let other_ok =
            |other: Value| !self.plan.elided[other.index()] || self.peel_uextend(other) != other;
        let ext = if alu == Alu::Add
            && super::super::fuse::ext_swapped(self.func, &self.plan, (x, y), &|v| {
                self.defs[v.index()]
            }) {
            ext_of(x).map(|e| (y, e))
        } else {
            ext_of(y)
                .filter(|_| other_ok(x))
                .map(|e| (x, e))
                .or_else(|| {
                    (alu == Alu::Add)
                        .then(|| ext_of(x).filter(|_| other_ok(y)).map(|e| (y, e)))
                        .flatten()
                })
        };
        let Some((other, (ext_op, shift, inner))) = ext else {
            return false;
        };
        let (left, right) = self.bits_pair(self.peel_uextend(other), inner);
        self.asm
            .alu_extended(alu, size, ext_op, shift, dst, left, right);
        true
    }

    fn try_alu_shifted_or_madd(
        &mut self,
        op: Binary,
        alu: Alu,
        (x, y): (Value, Value),
        (ty, size): (Type, Size),
        dst: Reg,
    ) -> bool {
        if let Some((not_alu, lhs, rhs, shifted)) =
            super::super::fuse::bnot_parts(self.func, &self.plan, ty, op, x, y, |v| {
                self.defs[v.index()]
            })
        {
            let (left, right) = self.bits_pair(self.peel_uextend(lhs), self.peel_uextend(rhs));
            if let Some((shift, imm)) = shifted
                && imm != 0
            {
                self.asm
                    .alu_shifted(not_alu, size, shift, imm, dst, left, right);
            } else {
                self.asm.alu(not_alu, size, dst, left, right);
            }
            return true;
        }
        let shifted = self
            .shifted_operand(alu, ty, y)
            .map(|s| (x, s))
            .or_else(|| {
                (alu != Alu::Sub)
                    .then(|| self.shifted_operand(alu, ty, x).map(|s| (y, s)))
                    .flatten()
            });
        if let Some((other, (b, shift, imm))) = shifted {
            let (left, right) = if alu == Alu::Sub && self.const_imm(other) == Some(0) {
                (ZR, self.bits(b, C))
            } else {
                self.bits_pair(self.peel_uextend(other), b)
            };
            self.emit_shifted(alu, size, dst, left, (right, shift, imm));
            return true;
        }
        if matches!(alu, Alu::Add | Alu::Sub)
            && matches!(ty, Type::I32 | Type::I64)
            && let Some((sub, a, b, c)) =
                super::super::fuse::madd_parts(self.func, &self.plan, ty, op, x, y, |v| {
                    self.defs[v.index()]
                })
        {
            let (left, right) = self.bits_pair(a, b);
            let acc = if self.plan.const_of(c) == Some(0) {
                ZR
            } else {
                let r = self.bits(c, IP);
                if r == IP {
                    self.clobber_reg(IP);
                }
                r
            };
            self.asm.madd(sub, size, dst, (left, right), acc);
            return true;
        }
        false
    }

    fn int_alu(
        &mut self,
        op: Binary,
        alu: Alu,
        (x, y): (Value, Value),
        (ty, size): (Type, Size),
        dst: Reg,
    ) {
        if self.try_alu_bit_or_cinc(op, alu, (x, y), (ty, size), dst)
            || self.try_alu_ext(alu, (x, y), (ty, size), dst)
            || self.try_alu_shifted_or_madd(op, alu, (x, y), (ty, size), dst)
        {
            return;
        }
        if matches!(alu, Alu::And | Alu::Orr | Alu::Eor) && dst != ZR {
            let imm_pair = match (self.const_imm(y), self.const_imm(x)) {
                (Some(imm), _) => Some((x, imm)),
                (None, Some(imm)) => Some((y, imm)),
                _ => None,
            };
            if let Some((val, imm)) = imm_pair
                && let Some(bitmask) = encode_bitmask_imm(imm as u64, size == Size::X)
            {
                let left = self.bits(self.peel_uextend(val), A);
                self.asm.alu_bitmask(alu, size, bitmask, dst, left);
                return;
            }
        }
        if matches!(alu, Alu::Add | Alu::Sub)
            && self.emit_add_sub_imm(alu == Alu::Sub, (x, y), (ty, size), dst)
        {
            return;
        }
        if alu == Alu::Sub && self.const_imm(x) == Some(0) {
            let right = self.bits(self.peel_uextend(y), C);
            self.asm.alu(alu, size, dst, ZR, right);
            return;
        }
        let (left, right) = self.bits_pair(self.peel_uextend(x), self.peel_uextend(y));
        self.asm.alu(alu, size, dst, left, right);
    }

    fn int_binary(
        &mut self,
        op: Binary,
        (x, y): (Value, Value),
        ty: Type,
        dst: Reg,
    ) -> Result<(), String> {
        let size = if ty == Type::I64 { Size::X } else { Size::W };
        if matches!(
            op,
            Binary::Band | Binary::Ishl | Binary::Ushr | Binary::Sshr
        ) && let Some(bfx) =
            super::super::fuse::bfx_parts(self.func, &self.plan, ty, op, (x, y), &|v| {
                self.defs[v.index()]
            })
        {
            self.emit_bfm(bfx, ty, dst);
            return Ok(());
        }
        if let Some(alu) = alu_of(op) {
            self.int_alu(op, alu, (x, y), (ty, size), dst);
            return Ok(());
        }
        let bits = ty.bits() as u8;
        match op {
            Binary::Imul => self.multiply(dst, x, y),
            Binary::Udiv | Binary::Urem if self.divide_by_power(op, (x, y), dst) => {}
            Binary::Sdiv | Binary::Srem | Binary::Udiv | Binary::Urem => {
                self.divide(op, (x, y), dst)
            }
            Binary::Umulhi | Binary::Smulhi => {
                self.multiply_high(op == Binary::Smulhi, (x, y), bits, dst)
            }
            Binary::Smin | Binary::Smax | Binary::Umin | Binary::Umax => {
                let signed = matches!(op, Binary::Smin | Binary::Smax);
                let left = self.src_opt(x, A, signed);
                let cond = match op {
                    Binary::Smin => Cond::Lt,
                    Binary::Smax => Cond::Gt,
                    Binary::Umin => Cond::Lo,
                    _ => Cond::Hi,
                };
                let right = if self.plan.const_of(y) == Some(0) {
                    self.asm.add_imm(true, true, Size::X, ZR, left, 0);
                    ZR
                } else {
                    let r = self.src_opt(y, C, signed);
                    self.asm.alu(Alu::Subs, Size::X, ZR, left, r);
                    r
                };
                self.asm.csel(cond, dst, left, right);
            }
            Binary::SaddSat | Binary::UaddSat | Binary::SsubSat | Binary::UsubSat => {
                self.sat_add_sub(op, (x, y), bits, dst)
            }
            Binary::AvgRound => {
                let (left, right) = self.zext_pair(x, y);
                self.asm.alu(Alu::Orr, Size::X, D, left, right);
                self.asm.alu(Alu::Eor, Size::X, A, left, right);
                self.asm
                    .alu_shifted(Alu::Sub, Size::X, Shift::Lsr, 1, dst, D, A);
                self.clobber_regs(&[A, D]);
            }
            Binary::Ishl | Binary::Ushr | Binary::Sshr => self.shift_by(op, (x, y), bits, dst),
            Binary::Rotl | Binary::Rotr => self.rotate(op == Binary::Rotl, (x, y), bits, dst),
            other => return Err(format!("{other:?} of an integer")),
        }
        Ok(())
    }

    pub(super) fn clamp_range(&mut self, val: Reg, dst: Reg, lo: i64, hi: i64) {
        let low_src = if lo == 0 {
            ZR
        } else {
            self.asm.mov_imm(C, lo);
            C
        };
        self.asm.alu(Alu::Subs, Size::X, ZR, val, low_src);
        self.asm.csel(Cond::Lt, D, low_src, val);
        self.asm.mov_imm(C, hi);
        self.asm.alu(Alu::Subs, Size::X, ZR, D, C);
        self.asm.csel(Cond::Gt, dst, C, D);
        self.clobber_regs(&[C, D]);
    }

    fn sat_add_sub(&mut self, op: Binary, (x, y): (Value, Value), bits: u8, dst: Reg) {
        let signed = matches!(op, Binary::SaddSat | Binary::SsubSat);
        let sub = matches!(op, Binary::SsubSat | Binary::UsubSat);
        if sub && !signed {
            let (left, right) = self.zext_pair(x, y);
            self.asm.alu(Alu::Subs, Size::X, D, left, right);
            self.asm.csel(Cond::Hs, dst, D, ZR);
            self.clobber_reg(D);
            return;
        }
        if bits < 64 {
            let (left, right) = self.pair_opt(x, y, signed);
            self.asm.alu(
                if sub { Alu::Sub } else { Alu::Add },
                Size::X,
                D,
                left,
                right,
            );
            if signed {
                let bound = 1i64 << (bits - 1);
                self.clamp_range(D, dst, -bound, bound - 1);
            } else {
                self.asm.mov_imm(C, (1i64 << bits) - 1);
                self.asm.alu(Alu::Subs, Size::X, ZR, D, C);
                self.asm.csel(Cond::Lo, dst, D, C);
                self.clobber_regs(&[C, D]);
            }
            return;
        }
        let (left, right) = self.bits_pair(x, y);
        self.asm.alu(
            if sub { Alu::Subs } else { Alu::Adds },
            Size::X,
            D,
            left,
            right,
        );
        if signed {
            self.asm.mov_imm(A, i64::MIN);
            self.asm
                .alu_shifted(Alu::Eor, Size::X, Shift::Asr, 63, C, A, D);
            self.asm.csel(Cond::Vs, dst, C, D);
        } else {
            self.asm.mov_imm(A, -1);
            self.asm.csel(Cond::Hs, dst, A, D);
        }
        self.clobber_regs(&[A, C, D]);
    }

    fn emit_mul_one(&mut self, size: Size, dst: Reg, val: Value, reg: Reg) {
        if size == Size::W && !self.is_zext_reg(val) {
            self.asm.extend(false, 4, dst, reg);
        } else {
            self.asm.mov(dst, reg);
        }
    }

    fn multiply(&mut self, dst: Reg, x: Value, y: Value) {
        let ty = self.ty(x);
        let size = if ty.bits() <= 32 { Size::W } else { Size::X };
        let lsl = |asm: &mut super::super::asm::Asm, d, s, k| {
            if size == Size::W {
                asm.shift32_imm(true, d, s, k)
            } else {
                asm.lsl_imm(d, s, k)
            }
        };
        let (value, raw_imm) = match (self.plan.const_of(x), self.plan.const_of(y)) {
            (_, Some(imm)) => (x, Some(imm)),
            (Some(imm), None) => (y, Some(imm)),
            (None, None) => (x, None),
        };
        if let Some(raw) = raw_imm {
            let uimm = crate::fold::extend(raw, ty, false) as u64;
            let simm = crate::fold::extend(raw, ty, true);
            if uimm == 0 {
                return self.asm.mov_imm(dst, 0);
            }
            let reg = self.bits(value, A);
            if uimm == 1 {
                return self.emit_mul_one(size, dst, value, reg);
            }
            if simm == -1 {
                return self.asm.alu(Alu::Sub, size, dst, ZR, reg);
            }
            if uimm.is_power_of_two() {
                return lsl(&mut self.asm, dst, reg, uimm.trailing_zeros() as u8);
            }
            if (uimm - 1).is_power_of_two() {
                return self.asm.alu_shifted(
                    Alu::Add,
                    size,
                    Shift::Lsl,
                    (uimm - 1).trailing_zeros() as u8,
                    dst,
                    reg,
                    reg,
                );
            }
            let max_shift = if size == Size::W { 31 } else { 63 };
            if let Some(next) = uimm
                .checked_add(1)
                .filter(|n| n.is_power_of_two() && n.trailing_zeros() as u8 <= max_shift)
            {
                lsl(&mut self.asm, C, reg, next.trailing_zeros() as u8);
                self.clobber_reg(C);
                return self.asm.alu(Alu::Sub, size, dst, C, reg);
            }
            if let Some(next) = simm
                .checked_neg()
                .filter(|&n| n > 2)
                .and_then(|n| (n as u64).checked_add(1))
                .filter(|n| n.is_power_of_two() && n.trailing_zeros() as u8 <= max_shift)
            {
                return self.asm.alu_shifted(
                    Alu::Sub,
                    size,
                    Shift::Lsl,
                    next.trailing_zeros() as u8,
                    dst,
                    reg,
                    reg,
                );
            }
            let shift = uimm.trailing_zeros() as u8;
            let odd = uimm >> shift;
            if shift > 0 && matches!(odd, 3 | 5 | 9) {
                self.asm.alu_shifted(
                    Alu::Add,
                    size,
                    Shift::Lsl,
                    (odd - 1).trailing_zeros() as u8,
                    dst,
                    reg,
                    reg,
                );
                return lsl(&mut self.asm, dst, dst, shift);
            }
        }
        let (left, right) = self.bits_pair(x, y);
        self.asm.madd(false, size, dst, (left, right), ZR);
    }

    fn divide_by_power(&mut self, op: Binary, (x, y): (Value, Value), dst: Reg) -> bool {
        let ty = self.ty(y);
        let Some(divisor) = self
            .plan
            .const_of(y)
            .map(|imm| crate::fold::extend(imm, ty, false) as u64)
            .filter(|imm| imm.is_power_of_two())
        else {
            return false;
        };
        if divisor == 1 && op == Binary::Urem {
            self.asm.mov_imm(dst, 0);
            return true;
        }
        if op == Binary::Udiv {
            let left = self.zext_src(x, A);
            if divisor == 1 {
                self.asm.mov(dst, left);
            } else {
                self.asm.lsr_imm(dst, left, divisor.trailing_zeros() as u8);
            }
            return true;
        }
        let left = self.bits(x, A);
        self.asm.mov_imm(C, (divisor - 1) as i64);
        self.asm.alu(Alu::And, Size::X, dst, left, C);
        self.clobber_reg(C);
        true
    }

    fn divide(&mut self, op: Binary, (x, y): (Value, Value), dst: Reg) {
        let signed = matches!(op, Binary::Sdiv | Binary::Srem);
        let size = if self.ty(x).bits() <= 32 {
            Size::W
        } else {
            Size::X
        };
        let (left, right) = if self.ty(x) == Type::I32 {
            self.bits_pair(x, y)
        } else {
            self.pair_opt(x, y, signed)
        };
        let div = if signed { Alu::Sdiv } else { Alu::Udiv };
        if matches!(op, Binary::Sdiv | Binary::Udiv) {
            return self.asm.alu(div, size, dst, left, right);
        }
        self.asm.alu(div, size, IP, left, right);
        self.asm.madd(true, size, dst, (IP, right), left);
        self.clobber_reg(IP);
    }

    fn multiply_high(&mut self, signed: bool, (x, y): (Value, Value), bits: u8, dst: Reg) {
        let (left, right) = self.pair_opt(x, y, signed);
        if bits == 64 {
            return self.asm.alu(
                if signed { Alu::Smulh } else { Alu::Umulh },
                Size::X,
                dst,
                left,
                right,
            );
        }
        self.asm.madd(false, Size::X, dst, (left, right), ZR);
        if signed {
            self.asm.asr_imm(dst, dst, bits);
        } else {
            self.asm.lsr_imm(dst, dst, bits);
        }
    }

    fn unwrap_shift_mask(&self, y: Value, bits: u8) -> Value {
        if bits < 32 || !self.plan.elided[y.index()] {
            return y;
        }
        let Some(Op::Binary(Binary::Band, a, b)) =
            self.defs[y.index()].map(|i| self.func.insts[i.index()].op.clone())
        else {
            return y;
        };
        let mask = u64::from(bits - 1);
        [(a, b), (b, a)]
            .into_iter()
            .find_map(|(raw, m)| {
                self.const_imm(m)
                    .filter(|&i| (i as u64 & mask) == mask)
                    .map(|_| raw)
            })
            .unwrap_or(y)
    }

    fn shift_src(&mut self, op: Binary, x: Value) -> Reg {
        match op {
            Binary::Ishl => self.bits(x, A),
            Binary::Ushr => self.zext_src(x, A),
            _ => self.src(x, A, Ext::Sign),
        }
    }

    pub(super) fn shift_by(&mut self, op: Binary, (x, y): (Value, Value), bits: u8, dst: Reg) {
        let alu = match op {
            Binary::Ishl => Alu::Lslv,
            Binary::Ushr => Alu::Lsrv,
            _ => Alu::Asrv,
        };
        let peeled = if bits >= 32 && matches!(op, Binary::Ishl | Binary::Ushr) {
            self.peel_uextend(x)
        } else {
            x
        };
        if let Some(count) = self.plan.const_of(y) {
            let count = (count as u8) & (bits - 1);
            if bits == 32 && matches!(op, Binary::Ishl | Binary::Ushr) {
                let value = if peeled != x {
                    self.zext_src(peeled, A)
                } else {
                    self.bits(x, A)
                };
                if count == 0 {
                    self.asm.extend(false, 4, dst, value);
                } else {
                    self.asm.shift32_imm(op == Binary::Ishl, dst, value, count);
                }
                return;
            }
            let value = if peeled != x {
                self.zext_src(peeled, A)
            } else {
                self.shift_src(op, x)
            };
            if count == 0 {
                self.asm.mov(dst, value);
                return;
            }
            match alu {
                Alu::Lslv => self.asm.lsl_imm(dst, value, count),
                Alu::Lsrv => self.asm.lsr_imm(dst, value, count),
                _ => self.asm.asr_imm(dst, value, count),
            }
            return;
        }
        let y = self.unwrap_shift_mask(y, bits);
        if bits >= 32 {
            let (value, count) = if peeled != x {
                (self.zext_src(peeled, A), self.bits(y, C))
            } else {
                self.bits_pair(x, y)
            };
            return self.asm.alu(
                alu,
                if bits == 32 { Size::W } else { Size::X },
                dst,
                value,
                count,
            );
        }
        let (value, count) = (self.shift_src(op, x), self.bits(y, C));
        self.asm.mov_imm(D, i64::from(bits - 1));
        self.asm.alu(Alu::And, Size::X, C, count, D);
        self.asm.alu(alu, Size::X, dst, value, C);
        self.clobber_regs(&[C, D]);
    }

    fn rotate(&mut self, left: bool, (x, y): (Value, Value), bits: u8, dst: Reg) {
        if bits >= 32
            && let Some(c) = self.plan.const_of(y)
        {
            let mask = u32::from(bits - 1);
            let c = (c as u32) & mask;
            let imm = if left {
                (u32::from(bits) - c) & mask
            } else {
                c
            } as u8;
            let value = self.bits(x, A);
            let size = Size::of(u32::from(bits) / 8);
            if imm == 0 {
                self.emit_mul_one(size, dst, x, value);
            } else {
                self.asm.ror_imm(size, dst, value, imm);
            }
            return;
        }
        let y = self.unwrap_shift_mask(y, bits);
        let count = self.bits(y, C);
        if left {
            self.asm.alu(Alu::Sub, Size::X, C, ZR, count);
        } else {
            self.asm.mov(C, count);
        }
        self.clobber_reg(C);
        if bits >= 32 {
            let value = self.bits(x, A);
            self.asm
                .alu(Alu::Rorv, Size::of(u32::from(bits) / 8), dst, value, C);
            return;
        }
        let value = self.zext_src(x, A);
        self.asm.mov_imm(D, i64::from(bits - 1));
        self.asm.alu(Alu::And, Size::X, C, C, D);
        self.asm.alu(Alu::Lsrv, Size::X, T, value, C);
        self.asm.alu(Alu::Sub, Size::X, C, ZR, C);
        self.asm.alu(Alu::And, Size::X, C, C, D);
        self.asm.alu(Alu::Lslv, Size::X, dst, value, C);
        self.asm.alu(Alu::Orr, Size::X, dst, dst, T);
        self.clobber_regs(&[C, D, T]);
    }
}

use super::asm::{Alu, Bits, Cond, Reg, Shift, Size, ZR};
use super::{A, C, D, Emit, IP, T, V0};
use crate::frame::Ext;
use crate::ir::{Binary, Convert, IntCC, Op, Type, Unary, Value};

pub(super) fn cond(cc: IntCC) -> Cond {
    match cc {
        IntCC::Equal => Cond::Eq, IntCC::NotEqual => Cond::Ne,
        IntCC::SignedLessThan => Cond::Lt, IntCC::SignedGreaterThanOrEqual => Cond::Ge,
        IntCC::SignedGreaterThan => Cond::Gt, IntCC::SignedLessThanOrEqual => Cond::Le,
        IntCC::UnsignedLessThan => Cond::Lo, IntCC::UnsignedGreaterThanOrEqual => Cond::Hs,
        IntCC::UnsignedGreaterThan => Cond::Hi, IntCC::UnsignedLessThanOrEqual => Cond::Ls,
    }
}

fn swap_cond(c: Cond) -> Cond {
    match c {
        Cond::Lt => Cond::Gt, Cond::Le => Cond::Ge, Cond::Gt => Cond::Lt, Cond::Ge => Cond::Le,
        Cond::Lo => Cond::Hi, Cond::Ls => Cond::Hs, Cond::Hi => Cond::Lo, Cond::Hs => Cond::Ls,
        other => other,
    }
}

pub(super) fn alu_of(op: Binary) -> Option<Alu> {
    match op {
        Binary::Iadd => Some(Alu::Add), Binary::Isub => Some(Alu::Sub),
        Binary::Band => Some(Alu::And), Binary::Bor => Some(Alu::Orr), Binary::Bxor => Some(Alu::Eor),
        _ => None,
    }
}

impl Emit<'_> {
    pub(super) fn unary(&mut self, op: Unary, x: Value, result: Value) -> Result<(), String> {
        let ty = self.ty(x);
        if ty.is_vector() { self.invalidate_spills(); return self.vec_unary(op, x, result); }
        if ty.is_float() { self.invalidate_spills(); return self.float_unary(op, x, result); }
        if ty == Type::I128 { self.invalidate_spills(); return self.wide_unary(op, x, result); }
        let dst = self.dst(result);
        self.int_unary(op, x, ty.bits() as u8, dst)?;
        self.put(result, dst);
        Ok(())
    }

    fn int_unary(&mut self, op: Unary, x: Value, bits: u8, dst: Reg) -> Result<(), String> {
        let size = if bits == 32 { Size::W } else { Size::X };
        match op {
            Unary::Ineg => {
                if matches!(bits, 32 | 64)
                    && let Some((cc, a, b)) = self.cinc_operand(x)
                {
                    let cond = self.compare(cc, a, b);
                    self.asm.csinv(size, cond.invert(), dst, ZR, ZR);
                } else {
                    let reg = self.bits(x, A);
                    self.asm.alu(Alu::Sub, size, dst, ZR, reg);
                }
            }
            Unary::Bnot => {
                if let Some((a, b)) = super::fuse::eon_inner(self.func, &self.plan, x, |v| self.defs[v.index()]) {
                    let left = self.bits(a, A);
                    let right = self.bits(b, C);
                    self.asm.alu(Alu::Eon, size, dst, left, right);
                } else {
                    let reg = self.bits(x, A);
                    self.asm.alu(Alu::Orn, size, dst, ZR, reg);
                }
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
                    self.clobber_regs(&[A, C]);
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
                self.clobber_vreg(V0);
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
        if ty.is_vector() { self.invalidate_spills(); return self.vec_binary(op, x, y, result); }
        if ty.is_float() { self.invalidate_spills(); return self.float_binary(op, x, y, result); }
        if ty == Type::I128 { self.invalidate_spills(); return self.wide_binary(op, x, y, result); }
        let dst = self.dst(result);
        self.int_binary(op, (x, y), ty, dst)?;
        self.put(result, dst);
        Ok(())
    }

    fn add_imm_args(&self, sub: bool, value: Value, ty: Type) -> Option<(bool, u32)> {
        add_imm_parts(sub, crate::fold::extend(self.plan.const_of(value)?, ty, true))
    }

    fn peel_uextend(&self, val: Value) -> Value {
        if self.plan.elided[val.index()]
            && let Some(inst) = self.defs[val.index()]
            && let Op::Convert(Convert::Uextend, inner) = self.func.insts[inst.index()].op
        {
            return inner;
        }
        val
    }

    fn shifted_operand(&self, alu: Alu, ty: Type, val: Value) -> Option<(Value, Shift, u8)> {
        if !self.plan.elided[val.index()] {
            return None;
        }
        let inst = self.defs[val.index()]?;
        let (op, b, imm) = super::fuse::shift_parts(&self.func.insts[inst.index()].op, &self.plan)?;
        let logical = matches!(alu, Alu::And | Alu::Orr | Alu::Eor);
        let (shift, count) = super::fuse::alu_shift(logical, ty, op, imm)?;
        Some((self.peel_uextend(b), shift, count))
    }

    fn cinc_operand(&self, val: Value) -> Option<(IntCC, Value, Value)> {
        if !self.plan.elided[val.index()] {
            return None;
        }
        let inst = self.defs[val.index()]?;
        match self.func.insts[inst.index()].op {
            Op::Icmp(cc, a, b) => Some((cc, a, b)),
            Op::Convert(Convert::Uextend, inner) if self.plan.elided[inner.index()] => {
                let inner_inst = self.defs[inner.index()]?;
                match self.func.insts[inner_inst.index()].op {
                    Op::Icmp(cc, a, b) => Some((cc, a, b)),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn const_imm(&self, val: Value) -> Option<i64> {
        super::fuse::const_of(self.func, &self.plan, val, |v| self.defs[v.index()])
    }

    fn int_binary(&mut self, op: Binary, (x, y): (Value, Value), ty: Type, dst: Reg) -> Result<(), String> {
        let size = if ty == Type::I64 { Size::X } else { Size::W };
        if matches!(op, Binary::Band | Binary::Ushr | Binary::Sshr)
            && let Some((signed, src, lsb, width)) = super::fuse::bfx_parts(self.func, &self.plan, ty, op, x, y, |v| self.defs[v.index()])
        {
            let reg = self.bits(self.peel_uextend(src), A);
            if lsb == 0 && u32::from(width) == ty.bits() {
                if size == Size::W { self.asm.extend(false, 4, dst, reg); } else { self.asm.mov(dst, reg); }
            } else {
                self.asm.bfx(signed, size, dst, reg, lsb, width);
            }
            return Ok(());
        }
        if let Some(alu) = alu_of(op) {
            if matches!(op, Binary::Bor | Binary::Bxor | Binary::Iadd)
                && self.plan.elided[x.index()]
                && self.plan.elided[y.index()]
                && let Some((hi, lo, kr)) = super::fuse::extr_parts(self.func, &self.plan, ty, x, y, |v| self.defs[v.index()])
            {
                let left = self.bits(hi, A);
                let right = self.bits(lo, C);
                self.asm.extr(size, dst, left, right, kr);
                return Ok(());
            }
            if alu == Alu::Add && matches!(ty, Type::I32 | Type::I64) {
                let cinc = self.cinc_operand(y).map(|cmp| (x, cmp)).or_else(|| self.cinc_operand(x).map(|cmp| (y, cmp)));
                if let Some((other, (cc, a, b))) = cinc {
                    let cond = self.compare(cc, a, b);
                    let reg = self.bits(other, A);
                    self.asm.csinc(size, cond.invert(), dst, reg, reg);
                    return Ok(());
                }
            }
            if alu == Alu::Sub && matches!(ty, Type::I32 | Type::I64) && self.const_imm(x) == Some(0)
                && let Some((cc, a, b)) = self.cinc_operand(y)
            {
                let cond = self.compare(cc, a, b);
                self.asm.csinv(size, cond.invert(), dst, ZR, ZR);
                return Ok(());
            }
            if matches!(alu, Alu::Add | Alu::Sub) && matches!(ty, Type::I32 | Type::I64) {
                let ext = super::fuse::ext_parts(self.func, &self.plan, ty, y, |v| self.defs[v.index()])
                    .map(|e| (x, e))
                    .or_else(|| {
                        (alu == Alu::Add)
                            .then(|| super::fuse::ext_parts(self.func, &self.plan, ty, x, |v| self.defs[v.index()]).map(|e| (y, e)))
                            .flatten()
                    });
                if let Some((other, (ext_op, shift, inner))) = ext {
                    let left = self.bits(other, A);
                    let right = self.bits(inner, C);
                    self.asm.alu_extended(alu, size, ext_op, shift, dst, left, right);
                    return Ok(());
                }
            }
            if let Some((not_alu, lhs, rhs, shifted)) = super::fuse::bnot_parts(self.func, &self.plan, ty, op, x, y, |v| self.defs[v.index()]) {
                let left = self.bits(self.peel_uextend(lhs), A);
                let right = self.bits(self.peel_uextend(rhs), C);
                if let Some((shift, imm)) = shifted && imm != 0 {
                    self.asm.alu_shifted(not_alu, size, shift, imm, dst, left, right);
                } else {
                    self.asm.alu(not_alu, size, dst, left, right);
                }
                return Ok(());
            }
            let shifted = self.shifted_operand(alu, ty, y).map(|s| (x, s)).or_else(|| (alu != Alu::Sub).then(|| self.shifted_operand(alu, ty, x).map(|s| (y, s))).flatten());
            if let Some((other, (b, shift, imm))) = shifted {
                let left = self.bits(self.peel_uextend(other), A);
                let right = self.bits(b, C);
                if imm == 0 {
                    self.asm.alu(alu, size, dst, left, right);
                } else {
                    self.asm.alu_shifted(alu, size, shift, imm, dst, left, right);
                }
                return Ok(());
            }
            if matches!(alu, Alu::And | Alu::Orr | Alu::Eor) && dst != ZR {
                let imm_pair = match (self.const_imm(y), self.const_imm(x)) {
                    (Some(imm), _) => Some((x, imm)),
                    (None, Some(imm)) => Some((y, imm)),
                    (None, None) => None,
                };
                if let Some((val, imm)) = imm_pair
                    && let Some(bitmask) = encode_bitmask_imm(imm as u64, size == Size::X)
                {
                    let left = self.bits(val, A);
                    self.asm.alu_bitmask(alu, size, bitmask, dst, left);
                    return Ok(());
                }
            }
            let left = self.bits(x, A);
            if matches!(alu, Alu::Add | Alu::Sub)
                && let Some((sub, imm)) = self.add_imm_args(alu == Alu::Sub, y, ty)
            {
                self.asm.add_imm(sub, false, size, dst, left, imm);
            } else {
                let right = self.bits(y, C);
                self.asm.alu(alu, size, dst, left, right);
            }
            return Ok(());
        }
        let bits = ty.bits() as u8;
        match op {
            Binary::Imul => self.multiply(dst, x, y),
            Binary::Udiv | Binary::Urem if self.divide_by_power(op, (x, y), dst) => {}
            Binary::Sdiv | Binary::Srem | Binary::Udiv | Binary::Urem => self.divide(op, (x, y), dst),
            Binary::Umulhi | Binary::Smulhi => self.multiply_high(op == Binary::Smulhi, (x, y), bits, dst),
            Binary::Smin | Binary::Smax | Binary::Umin | Binary::Umax => {
                let signed = matches!(op, Binary::Smin | Binary::Smax);
                let left = self.src(x, A, Ext::of(signed));
                let right = self.src(y, C, Ext::of(signed));
                self.asm.alu(Alu::Subs, Size::X, ZR, left, right);
                let cond = match op {
                    Binary::Smin => Cond::Lt,
                    Binary::Smax => Cond::Gt,
                    Binary::Umin => Cond::Lo,
                    _ => Cond::Hi,
                };
                self.asm.csel(cond, dst, left, right);
            }
            Binary::SaddSat | Binary::UaddSat | Binary::SsubSat | Binary::UsubSat => self.sat_add_sub(op, (x, y), bits, dst),
            Binary::AvgRound => {
                let left = self.src(x, A, Ext::Zero);
                let right = self.src(y, C, Ext::Zero);
                self.asm.alu(Alu::Orr, Size::X, D, left, right);
                self.asm.alu(Alu::Eor, Size::X, A, left, right);
                self.asm.alu_shifted(Alu::Sub, Size::X, Shift::Lsr, 1, dst, D, A);
                self.clobber_regs(&[A, D]);
            }
            Binary::Ishl | Binary::Ushr | Binary::Sshr => self.shift_by(op, (x, y), bits, dst),
            Binary::Rotl | Binary::Rotr => self.rotate(op == Binary::Rotl, (x, y), bits, dst),
            other => return Err(format!("{other:?} of an integer")),
        }
        Ok(())
    }

    fn clamp_range(&mut self, val: Reg, dst: Reg, lo: i64, hi: i64) {
        let low_src = if lo == 0 { ZR } else { self.asm.mov_imm(C, lo); C };
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
            let (left, right) = (self.src(x, A, Ext::Zero), self.src(y, C, Ext::Zero));
            self.asm.alu(Alu::Subs, Size::X, D, left, right);
            self.asm.csel(Cond::Hs, dst, D, ZR);
            self.clobber_reg(D);
            return;
        }
        if bits < 64 {
            let (left, right) = (self.src(x, A, Ext::of(signed)), self.src(y, C, Ext::of(signed)));
            self.asm.alu(if sub { Alu::Sub } else { Alu::Add }, Size::X, D, left, right);
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
        let (left, right) = (self.bits(x, A), self.bits(y, C));
        self.asm.alu(if sub { Alu::Subs } else { Alu::Adds }, Size::X, D, left, right);
        if signed {
            self.asm.mov_imm(A, i64::MIN);
            self.asm.alu_shifted(Alu::Eor, Size::X, Shift::Asr, 63, C, A, D);
            self.asm.csel(Cond::Vs, dst, C, D);
        } else {
            self.asm.mov_imm(A, -1);
            self.asm.csel(Cond::Hs, dst, A, D);
        }
        self.clobber_regs(&[A, C, D]);
    }

    fn multiply(&mut self, dst: Reg, x: Value, y: Value) {
        let ty = self.ty(x);
        let size = if ty.bits() <= 32 { Size::W } else { Size::X };
        let lsl = |asm: &mut super::asm::Asm, d, s, k| {
            if size == Size::W { asm.lsl32_imm(d, s, k) } else { asm.lsl_imm(d, s, k) }
        };
        let (value, raw_imm) = match (self.plan.const_of(x), self.plan.const_of(y)) {
            (_, Some(imm)) => (x, Some(imm)),
            (Some(imm), None) => (y, Some(imm)),
            (None, None) => (x, None),
        };
        if let Some(raw) = raw_imm {
            let uimm = crate::fold::extend(raw, ty, false) as u64;
            let simm = crate::fold::extend(raw, ty, true);
            if uimm == 0 { return self.asm.mov_imm(dst, 0); }
            let reg = self.bits(value, A);
            if uimm == 1 {
                return if size == Size::W { self.asm.extend(false, 4, dst, reg) } else { self.asm.mov(dst, reg) };
            }
            if simm == -1 { return self.asm.alu(Alu::Sub, size, dst, ZR, reg); }
            if uimm.is_power_of_two() { return lsl(&mut self.asm, dst, reg, uimm.trailing_zeros() as u8); }
            if (uimm - 1).is_power_of_two() {
                return self.asm.alu_shifted(Alu::Add, size, Shift::Lsl, (uimm - 1).trailing_zeros() as u8, dst, reg, reg);
            }
            let max_shift = if size == Size::W { 31 } else { 63 };
            if let Some(next) = uimm.checked_add(1).filter(|n| n.is_power_of_two() && n.trailing_zeros() as u8 <= max_shift) {
                lsl(&mut self.asm, C, reg, next.trailing_zeros() as u8);
                self.clobber_reg(C);
                return self.asm.alu(Alu::Sub, size, dst, C, reg);
            }
            if let Some(next) = simm.checked_neg().filter(|&n| n > 2).and_then(|n| (n as u64).checked_add(1))
                .filter(|n| n.is_power_of_two() && n.trailing_zeros() as u8 <= max_shift)
            {
                return self.asm.alu_shifted(Alu::Sub, size, Shift::Lsl, next.trailing_zeros() as u8, dst, reg, reg);
            }
        }
        let left = self.bits(x, A);
        let right = self.bits(y, C);
        self.asm.madd(false, size, dst, (left, right), ZR);
    }

    fn divide_by_power(&mut self, op: Binary, (x, y): (Value, Value), dst: Reg) -> bool {
        let ty = self.ty(y);
        let Some(divisor) = self.plan.const_of(y).map(|imm| crate::fold::extend(imm, ty, false)).filter(|imm| *imm > 0 && imm.count_ones() == 1) else { return false };
        let left = self.src(x, A, Ext::Zero);
        if divisor == 1 {
            if op == Binary::Udiv { self.asm.mov(dst, left); } else { self.asm.mov_imm(dst, 0); }
            return true;
        }
        match op {
            Binary::Udiv => self.asm.lsr_imm(dst, left, divisor.trailing_zeros() as u8),
            _ => {
                self.asm.mov_imm(C, divisor - 1);
                self.asm.alu(Alu::And, Size::X, dst, left, C);
                self.clobber_reg(C);
            }
        }
        true
    }

    fn divide(&mut self, op: Binary, (x, y): (Value, Value), dst: Reg) {
        let signed = matches!(op, Binary::Sdiv | Binary::Srem);
        let (left, right) = (self.src(x, A, Ext::of(signed)), self.src(y, C, Ext::of(signed)));
        let div = if signed { Alu::Sdiv } else { Alu::Udiv };
        if matches!(op, Binary::Sdiv | Binary::Udiv) {
            return self.asm.alu(div, Size::X, dst, left, right);
        }
        let size = if self.ty(x).bits() <= 32 { Size::W } else { Size::X };
        self.asm.alu(div, Size::X, IP, left, right);
        self.asm.madd(true, size, dst, (IP, right), left);
        self.clobber_reg(IP);
    }

    fn multiply_high(&mut self, signed: bool, (x, y): (Value, Value), bits: u8, dst: Reg) {
        let left = self.src(x, A, Ext::of(signed));
        let right = self.src(y, C, Ext::of(signed));
        if bits == 64 {
            self.asm.alu(if signed { Alu::Smulh } else { Alu::Umulh }, Size::X, dst, left, right);
            return;
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
        let Some(inst) = self.defs[y.index()] else { return y };
        let Op::Binary(Binary::Band, a, b) = self.func.insts[inst.index()].op else { return y };
        let mask = u64::from(bits - 1);
        for (raw, m) in [(a, b), (b, a)] {
            if let Some(imm) = self.const_imm(m)
                && (imm as u64 & mask) == mask
            {
                return raw;
            }
        }
        y
    }

    fn shift_by(&mut self, op: Binary, (x, y): (Value, Value), bits: u8, dst: Reg) {
        let (ext, alu) = match op {
            Binary::Ishl => (Ext::Zero, Alu::Lslv),
            Binary::Ushr => (Ext::Zero, Alu::Lsrv),
            _ => (Ext::Sign, Alu::Asrv),
        };
        if let Some(count) = self.plan.const_of(y) {
            let value = if op == Binary::Ishl { self.bits(x, A) } else { self.src(x, A, ext) };
            let count = (count as u8) & (bits - 1);
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
            let value = self.bits(x, A);
            let count = self.bits(y, C);
            self.asm.alu(alu, if bits == 32 { Size::W } else { Size::X }, dst, value, count);
            return;
        }
        let value = if op == Binary::Ishl { self.bits(x, A) } else { self.src(x, A, ext) };
        let count = self.bits(y, C);
        self.asm.mov_imm(D, i64::from(bits - 1));
        self.asm.alu(Alu::And, Size::X, C, count, D);
        self.asm.alu(alu, Size::X, dst, value, C);
        self.clobber_regs(&[C, D]);
    }

    fn rotate(&mut self, left: bool, (x, y): (Value, Value), bits: u8, dst: Reg) {
        if bits >= 32 && let Some(c) = self.plan.const_of(y) {
            let mask = u32::from(bits - 1);
            let c = (c as u32) & mask;
            let imm = if left { (u32::from(bits) - c) & mask } else { c } as u8;
            let value = self.bits(x, A);
            let size = Size::of(u32::from(bits) / 8);
            if imm == 0 {
                if size == Size::W { self.asm.extend(false, 4, dst, value); } else { self.asm.mov(dst, value); }
            } else {
                self.asm.ror_imm(size, dst, value, imm);
            }
            return;
        }
        let y = self.unwrap_shift_mask(y, bits);
        let count = self.bits(y, C);
        if left { self.asm.alu(Alu::Sub, Size::X, C, ZR, count); } else { self.asm.mov(C, count); }
        self.clobber_reg(C);
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
        self.clobber_regs(&[C, D, T]);
    }

    pub(super) fn elided_band_cmp(&self, x: Value, y: Value) -> Option<(Value, Value)> {
        let masked = (self.const_imm(y) == Some(0)).then_some(x).or_else(|| (self.const_imm(x) == Some(0)).then_some(y))?;
        if !self.plan.elided[masked.index()] {
            return None;
        }
        let inst = self.defs[masked.index()]?;
        let Op::Binary(Binary::Band, a, b) = self.func.insts[inst.index()].op else {
            return None;
        };
        Some((a, b))
    }

    pub(super) fn emit_tst(&mut self, x: Value, y: Value) {
        let ty = self.ty(x);
        let bits = ty.bits();
        let size = if bits >= 64 { Size::X } else { Size::W };
        let ty_mask = if bits == 64 { u64::MAX } else { (1_u64 << bits) - 1 };
        let imm_pair = match (self.const_imm(y), self.const_imm(x)) {
            (Some(imm), _) => Some((x, (imm as u64) & ty_mask)),
            (None, Some(imm)) => Some((y, (imm as u64) & ty_mask)),
            (None, None) => None,
        };
        if let Some((val, imm)) = imm_pair {
            let left = self.bits(val, A);
            if let Some(bitmask) = encode_bitmask_imm(imm, size == Size::X) {
                self.asm.alu_bitmask(Alu::Ands, size, bitmask, ZR, left);
            } else {
                self.asm.mov_imm(C, imm as i64);
                self.asm.alu(Alu::Ands, size, ZR, left, C);
                self.clobber_reg(C);
            }
            return;
        }
        let left = if bits >= 32 { self.bits(x, A) } else { self.src(x, A, Ext::Zero) };
        let right = if bits >= 32 { self.bits(y, C) } else { self.src(y, C, Ext::Zero) };
        self.asm.alu(Alu::Ands, size, ZR, left, right);
    }

    pub(super) fn compare(&mut self, cc: IntCC, x: Value, y: Value) -> Cond {
        if matches!(cc, IntCC::Equal | IntCC::NotEqual)
            && let Some((a, b)) = self.elided_band_cmp(x, y)
        {
            self.emit_tst(a, b);
            self.last_cmp = None;
            return cond(cc);
        }
        let bytes = self.ty(x).bytes();
        let size = if bytes >= 8 { Size::X } else { Size::W };
        let signed = cc.is_signed();
        if self.last_cmp == Some((size, x, y, signed)) {
            return cond(cc);
        }
        let ext = Ext::of(signed);
        let x_ty = self.ty(x);
        if matches!(x_ty, Type::I32 | Type::I64)
            && let Some((ext_op, shift, inner)) = super::fuse::ext_parts(self.func, &self.plan, x_ty, y, |v| self.defs[v.index()])
        {
            let (left, right) = (self.bits(x, A), self.bits(inner, C));
            self.asm.alu_extended(Alu::Subs, size, ext_op, shift, ZR, left, right);
            self.last_cmp = Some((size, x, y, signed));
            return cond(cc);
        }
        let left = if bytes >= 4 { self.bits(x, A) } else { self.src(x, A, ext) };
        let imm = self.plan.const_of(y).map(|imm| crate::fold::extend(imm, self.ty(y), signed));
        match imm.and_then(|imm| add_imm_parts(true, imm)) {
            Some((sub, imm)) => self.asm.add_imm(sub, true, size, ZR, left, imm),
            None => {
                if matches!(x_ty, Type::I32 | Type::I64)
                    && let Some((ext_op, shift, inner)) = super::fuse::ext_parts(self.func, &self.plan, x_ty, x, |v| self.defs[v.index()])
                {
                    let (lhs, rhs) = (self.bits(y, A), self.bits(inner, C));
                    self.asm.alu_extended(Alu::Subs, size, ext_op, shift, ZR, lhs, rhs);
                    self.last_cmp = None;
                    return swap_cond(cond(cc));
                }
                let right = if bytes >= 4 { self.bits(y, C) } else { self.src(y, C, ext) };
                self.asm.alu(Alu::Subs, size, ZR, left, right);
            }
        }
        self.last_cmp = Some((size, x, y, signed));
        cond(cc)
    }

    pub(super) fn icmp(&mut self, cc: IntCC, x: Value, y: Value, result: Value) -> Result<(), String> {
        if self.plan.is_static(result) {
            return Ok(());
        }
        if self.ty(x).is_vector() { self.invalidate_spills(); return self.vec_icmp(cc, x, y, result); }
        let flags = if self.ty(x) == Type::I128 {
            self.last_cmp = None;
            self.invalidate_spills();
            self.wide_icmp(cc, x, y)
        } else {
            self.compare(cc, x, y)
        };
        self.set_flag(result, flags);
        Ok(())
    }

    pub(super) fn set_flag(&mut self, result: Value, flags: Cond) {
        self.flags = Some((result, flags));
        if self.plan.fused[result.index()] {
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
        self.invalidate_spills();
        let bits = self.ty(x).bits() as u8;
        let reg = pair.and_then(|sum| self.plan.reg_of(sum)).unwrap_or(IP);
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
                    _ => self.asm.madd(false, Size::X, reg, (left, right), ZR),
                }
                self.asm.extend(signed, u32::from(bits) / 8, A, reg);
                self.asm.alu(Alu::Subs, Size::X, ZR, reg, A);
                Cond::Eq
            }
            (Binary::Imul, _) => {
                let (left, right) = (self.bits(x, A), self.bits(y, C));
                self.asm.alu(if signed { Alu::Smulh } else { Alu::Umulh }, Size::X, T, left, right);
                self.asm.madd(false, Size::X, reg, (left, right), ZR);
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
        if self.ty(x).is_vector() || self.ty(result).is_vector() {
            self.invalidate_spills();
            return self.vec_convert(op, x, result);
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
                    self.clobber_reg(C);
                } else if wide {
                    self.put_high(result, ZR);
                }
                Ok(())
            }
            Convert::Ireduce => {
                let dst = self.dst(result);
                if self.plan.elided[x.index()]
                    && let Some(inst) = self.defs[x.index()]
                    && let Op::Binary(Binary::Ushr, val, amt) = self.func.insts[inst.index()].op
                {
                    self.shift_by(Binary::Ushr, (val, amt), self.ty(x).bits() as u8, dst);
                } else {
                    self.raw(dst, x);
                }
                self.put(result, dst);
                Ok(())
            }
            Convert::NarrowSat { signed } => {
                let src = self.src(x, A, Ext::Sign);
                let dst = self.dst(result);
                let to_bits = self.ty(result).bits();
                let lo = if signed { -(1i64 << (to_bits - 1)) } else { 0 };
                let hi = if signed { (1i64 << (to_bits - 1)) - 1 } else { (1i64 << to_bits) - 1 };
                self.clamp_range(src, dst, lo, hi);
                self.put(result, dst);
                Ok(())
            }
            _ => {
                self.invalidate_spills();
                self.float_convert(op, x, result)
            }
        }
    }
}

pub(crate) fn add_imm_parts(sub: bool, imm: i64) -> Option<(bool, u32)> {
    let abs = imm.unsigned_abs();
    (abs < 4096 || (abs & 0xFFF == 0 && abs < (1 << 24))).then(|| (if imm < 0 { !sub } else { sub }, abs as u32))
}

pub(crate) fn encode_bitmask_imm(val: u64, is_64: bool) -> Option<(u32, u32, u32)> {
    let w = if is_64 { val } else { let low = (val as u32) as u64; low | (low << 32) };
    if w == 0 || w == u64::MAX {
        return None;
    }
    let mut esize = 64u32;
    while esize > 2 && w == w.rotate_right(esize / 2) {
        esize /= 2;
    }
    let mask = if esize == 64 { u64::MAX } else { (1u64 << esize) - 1 };
    let elem = w & mask;
    let ones = elem.count_ones();
    if ones == 0 || ones == esize {
        return None;
    }
    let tz = elem.trailing_zeros();
    let immr = if tz > 0 { (esize - tz) & (esize - 1) } else { ones - (!elem & mask).trailing_zeros() };
    let (n, imms) = (u32::from(esize == 64), ((!(esize - 1) << 1) | (ones - 1)) & 0x3F);
    let pat = (1u64 << ones) - 1;
    let mut decoded = if esize == 64 { pat.rotate_right(immr) } else { ((pat >> immr) | (pat << (esize - immr))) & mask };
    let mut shift = esize;
    while shift < 64 {
        decoded |= decoded << shift;
        shift *= 2;
    }
    (decoded == w).then_some((n, immr, imms))
}

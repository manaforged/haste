use super::super::asm::{Alu, Cond, Reg, Size, ZR};
use super::super::{A, Emit, IP, T};
use super::{add_imm_parts, cond, swap_cond};
use crate::frame::Ext;
use crate::ir::{Binary, Convert, IntCC, Op, Type, Value};

impl Emit<'_> {
    pub(in crate::arm64) fn elided_band_cmp(&self, x: Value, y: Value) -> Option<(Value, Value)> {
        let masked = (self.const_imm(y) == Some(0))
            .then_some(x)
            .or_else(|| (self.const_imm(x) == Some(0)).then_some(y))?;
        if !self.plan.elided[masked.index()] {
            return None;
        }
        let Op::Binary(Binary::Band, a, b) = self.func.insts[self.defs[masked.index()]?.index()].op
        else {
            return None;
        };
        Some((a, b))
    }

    pub(in crate::arm64) fn cmp_arg(
        &mut self,
        val: Value,
        scratch: Reg,
        bytes: u32,
        signed: bool,
    ) -> Reg {
        if bytes >= 4 {
            self.bits(val, scratch)
        } else {
            self.src_opt(val, scratch, signed)
        }
    }

    fn cmp_imm(
        &mut self,
        (lhs, rhs): (Value, Value),
        (bytes, size, cc): (u32, Size, IntCC),
    ) -> bool {
        if self.plan.elided[lhs.index()] {
            return false;
        }
        let Some(raw) = self.plan.const_of(rhs) else {
            return false;
        };
        let rhs_ty = self.ty(rhs);
        let parts = |signed: bool| {
            add_imm_parts(true, crate::fold::extend(raw, rhs_ty, signed)).map(|p| (signed, p))
        };
        let eq = matches!(cc, IntCC::Equal | IntCC::NotEqual);
        let Some((signed, (sub, imm))) =
            parts(cc.is_signed()).or_else(|| eq.then(|| parts(true)).flatten())
        else {
            return false;
        };
        if imm == 0 && eq && self.tst_small_zero(lhs) {
            self.last_cmp = None;
            return true;
        }
        let reg = if self.is_zext_sub_cmp(lhs, cc, raw) {
            self.bits(lhs, A)
        } else {
            self.cmp_arg(lhs, A, bytes, signed)
        };
        self.asm.add_imm(sub, true, size, ZR, reg, imm);
        self.last_cmp = Some((size, lhs, rhs, signed));
        true
    }

    fn cmp_fused(
        &mut self,
        cc: IntCC,
        (x, y): (Value, Value),
        (x_ty, size, signed): (Type, Size, bool),
    ) -> Option<Cond> {
        if matches!(cc, IntCC::Equal | IntCC::NotEqual)
            && let Some((a, b)) =
                super::super::fuse::cmn_sum_parts(self.func, &self.plan, (x, y), &|v| {
                    self.defs[v.index()]
                })
        {
            let (left, right) = self.bits_pair(a, b);
            self.asm.alu(Alu::Adds, size, ZR, left, right);
            self.last_cmp = None;
            return Some(cond(cc));
        }
        if matches!(x_ty, Type::I32 | Type::I64)
            && let Some((ext_op, shift, inner)) =
                super::super::fuse::ext_parts(self.func, &self.plan, x_ty, y, |v| {
                    self.defs[v.index()]
                })
        {
            let (left, right) = self.bits_pair(self.peel_uextend(x), inner);
            self.asm
                .alu_extended(Alu::Subs, size, ext_op, shift, ZR, left, right);
            self.last_cmp = Some((size, x, y, signed));
            return Some(cond(cc));
        }
        let bytes = x_ty.bytes();
        if self.cmp_imm((x, y), (bytes, size, cc)) {
            return Some(cond(cc));
        }
        if self.cmp_imm((y, x), (bytes, size, cc)) {
            return Some(swap_cond(cond(cc)));
        }
        if let Some(c) = self.cmp_high_bits(cc, (x, y), size) {
            return Some(c);
        }
        if matches!(x_ty, Type::I32 | Type::I64)
            && let Some((ext_op, shift, inner)) =
                super::super::fuse::ext_parts(self.func, &self.plan, x_ty, x, |v| {
                    self.defs[v.index()]
                })
        {
            let (lhs, rhs) = self.bits_pair(self.peel_uextend(y), inner);
            self.asm
                .alu_extended(Alu::Subs, size, ext_op, shift, ZR, lhs, rhs);
            self.last_cmp = Some((size, y, x, signed));
            return Some(swap_cond(cond(cc)));
        }
        None
    }

    fn cmp_shift_cmn(
        &mut self,
        cc: IntCC,
        (x, y): (Value, Value),
        (x_ty, size, signed): (Type, Size, bool),
    ) -> Option<Cond> {
        if !matches!(x_ty, Type::I32 | Type::I64) {
            return None;
        }
        if matches!(cc, IntCC::Equal | IntCC::NotEqual)
            && let Some((other, (rhs, shift))) =
                super::super::fuse::cmn_parts(self.func, &self.plan, x_ty, y, &|v: Value| {
                    self.defs[v.index()]
                })
                .map(|p| (x, p))
                .or_else(|| {
                    super::super::fuse::cmn_parts(self.func, &self.plan, x_ty, x, &|v: Value| {
                        self.defs[v.index()]
                    })
                    .map(|p| (y, p))
                })
        {
            let (left, right) = self.bits_pair(other, rhs);
            match shift {
                Some((s, imm)) => self.emit_shifted(Alu::Adds, size, ZR, left, (right, s, imm)),
                None => self.asm.alu(Alu::Adds, size, ZR, left, right),
            }
            self.last_cmp = None;
            return Some(cond(cc));
        }
        if let Some((b, shift, imm)) = self.shifted_operand(Alu::Sub, x_ty, y) {
            let (left, right) = self.bits_pair(self.peel_uextend(x), b);
            self.emit_shifted(Alu::Subs, size, ZR, left, (right, shift, imm));
            self.last_cmp = Some((size, x, y, signed));
            return Some(cond(cc));
        }
        let (b, shift, imm) = self.shifted_operand(Alu::Sub, x_ty, x)?;
        let (lhs, rhs) = self.bits_pair(self.peel_uextend(y), b);
        self.emit_shifted(Alu::Subs, size, ZR, lhs, (rhs, shift, imm));
        self.last_cmp = Some((size, y, x, signed));
        Some(swap_cond(cond(cc)))
    }

    pub(in crate::arm64) fn compare(&mut self, cc: IntCC, x: Value, y: Value) -> Cond {
        if matches!(cc, IntCC::Equal | IntCC::NotEqual)
            && let Some((a, b)) = self.elided_band_cmp(x, y)
        {
            self.emit_tst(a, b);
            self.last_cmp = None;
            return cond(cc);
        }
        let x_ty = self.ty(x);
        let (bytes, signed) = (x_ty.bytes(), cc.is_signed());
        let size = if bytes >= 8 { Size::X } else { Size::W };
        if self.last_cmp == Some((size, x, y, signed)) {
            return cond(cc);
        }
        if self.last_cmp == Some((size, y, x, signed)) {
            return swap_cond(cond(cc));
        }
        if let Some(c) = self
            .cmp_sload(cc, x, y)
            .or_else(|| self.cmp_fused(cc, (x, y), (x_ty, size, signed)))
            .or_else(|| self.cmp_shift_cmn(cc, (x, y), (x_ty, size, signed)))
        {
            return c;
        }
        let (left, right) = if bytes >= 4 {
            self.bits_pair(x, y)
        } else {
            self.pair_opt(x, y, signed)
        };
        self.asm.alu(Alu::Subs, size, ZR, left, right);
        self.last_cmp = Some((size, x, y, signed));
        cond(cc)
    }

    pub(in crate::arm64) fn icmp(
        &mut self,
        cc: IntCC,
        x: Value,
        y: Value,
        result: Value,
    ) -> Result<(), String> {
        if self.plan.is_static(result) {
            return Ok(());
        }
        if self.ty(x).is_vector() {
            self.invalidate_spills();
            return self.vec_icmp(cc, x, y, result);
        }
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

    pub(in crate::arm64) fn set_flag(&mut self, result: Value, flags: Cond) {
        self.flags = Some((result, flags));
        if self.plan.fused[result.index()] {
            return;
        }
        let dst = self.dst(result);
        self.asm.cset(flags, dst);
        self.put(result, dst);
    }

    pub(in crate::arm64) fn fits(
        &mut self,
        op: Binary,
        signed: bool,
        (x, y): (Value, Value),
        result: Value,
    ) -> Result<(), String> {
        let pair = self.plan.pairs[result.index()];
        if self.plan.is_static(result) && pair.is_none() {
            return Ok(());
        }
        let bits = self.ty(x).bits() as u8;
        let reg = pair.and_then(|sum| self.plan.reg_of(sum)).unwrap_or(IP);
        let ok = match (op, bits) {
            (Binary::Iadd | Binary::Isub, 32 | 64) => {
                let (ty, sub) = (self.ty(x), op == Binary::Isub);
                let size = if bits == 64 { Size::X } else { Size::W };
                let imm_pair = self.add_imm_args(sub, y, ty).map(|p| (x, p)).or_else(|| {
                    (!sub)
                        .then(|| self.add_imm_args(false, x, ty).map(|p| (y, p)))
                        .flatten()
                });
                if let Some((lhs, (eff_sub, imm))) = imm_pair {
                    let left = self.bits(lhs, A);
                    self.asm.add_imm(eff_sub, true, size, reg, left, imm);
                } else {
                    let (left, right) = self.bits_pair(x, y);
                    self.asm.alu(
                        if sub { Alu::Subs } else { Alu::Adds },
                        size,
                        reg,
                        left,
                        right,
                    );
                }
                match (signed, sub) {
                    (true, _) => Cond::Vc,
                    (false, false) => Cond::Lo,
                    (false, true) => Cond::Hs,
                }
            }
            (Binary::Iadd | Binary::Isub | Binary::Imul, _) if bits < 64 => {
                let (left, right) = self.pair_opt(x, y, signed);
                match op {
                    Binary::Iadd => self.asm.alu(Alu::Add, Size::X, reg, left, right),
                    Binary::Isub => self.asm.alu(Alu::Sub, Size::X, reg, left, right),
                    _ => self.asm.madd(false, Size::X, reg, (left, right), ZR),
                }
                self.asm.extend(signed, u32::from(bits) / 8, A, reg);
                self.asm.alu(Alu::Subs, Size::X, ZR, reg, A);
                self.clobber_reg(A);
                Cond::Eq
            }
            (Binary::Imul, _) => {
                let (left, right) = self.bits_pair(x, y);
                self.asm.alu(
                    if signed { Alu::Smulh } else { Alu::Umulh },
                    Size::X,
                    T,
                    left,
                    right,
                );
                self.asm.madd(false, Size::X, reg, (left, right), ZR);
                let expected = if signed {
                    self.asm.asr_imm(A, reg, 63);
                    A
                } else {
                    ZR
                };
                self.asm.alu(Alu::Subs, Size::X, ZR, T, expected);
                self.clobber_regs(&[A, T]);
                Cond::Eq
            }
            (other, _) => return Err(format!("{other:?} has no overflow check")),
        };
        if let Some(sum) = pair {
            self.put(sum, reg);
        } else {
            self.clobber_reg(reg);
        }
        self.set_flag(result, ok);
        Ok(())
    }

    pub(in crate::arm64) fn convert(
        &mut self,
        op: Convert,
        x: Value,
        result: Value,
    ) -> Result<(), String> {
        if self.plan.is_static(result) {
            return Ok(());
        }
        if self.ty(x).is_vector() || self.ty(result).is_vector() {
            self.invalidate_spills();
            return self.vec_convert(op, x, result);
        }
        match op {
            Convert::Uextend if self.ty(result) != Type::I128 && self.is_zext_reg(x) => {
                let dst = self.dst(result);
                self.raw(dst, x);
                self.put(result, dst);
                Ok(())
            }
            Convert::Uextend | Convert::Sextend => self.emit_extend_conv(op, x, result),
            Convert::Ireduce => {
                let dst = self.dst(result);
                if self.plan.elided[x.index()]
                    && let Some(inst) = self.defs[x.index()]
                    && let Op::Binary(Binary::Ushr, val, amt) = self.func.insts[inst.index()].op
                {
                    self.shift_by(Binary::Ushr, (val, amt), self.ty(x).bits() as u8, dst);
                } else if self.plan.zero_ext[result.index()]
                    && self.ty(result).bits() < 32
                    && !self.is_zext_within(x, self.ty(result).bits(), 4)
                {
                    let src = self.bits(x, A);
                    self.asm.extend(false, self.ty(result).bytes(), dst, src);
                } else {
                    self.raw(dst, x);
                }
                self.put(result, dst);
                Ok(())
            }
            Convert::NarrowSat { signed } => {
                let (src, dst, to_bits) = (
                    self.src(x, A, Ext::Sign),
                    self.dst(result),
                    self.ty(result).bits(),
                );
                let (lo, hi) = if signed {
                    (-(1i64 << (to_bits - 1)), (1i64 << (to_bits - 1)) - 1)
                } else {
                    (0, (1i64 << to_bits) - 1)
                };
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

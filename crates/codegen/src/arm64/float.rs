use super::asm::{Alu, Cond, Fconv, Fop, Funary, Size, Vreg, ZR};
use super::vector::{V2, V3};
use super::{A, C, Emit, V0, V1};
use crate::frame::Ext;
use crate::ir::{Binary, Convert, FloatCC, Type, Unary, Value};

pub(super) const POOL: [u8; 20] = [
    4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
];
pub(super) const BRANCHES: [FloatCC; 6] = [
    FloatCC::Equal,
    FloatCC::NotEqual,
    FloatCC::LessThan,
    FloatCC::LessThanOrEqual,
    FloatCC::GreaterThan,
    FloatCC::GreaterThanOrEqual,
];

fn swap_float_cc(cc: FloatCC) -> FloatCC {
    match cc {
        FloatCC::LessThan => FloatCC::GreaterThan,
        FloatCC::LessThanOrEqual => FloatCC::GreaterThanOrEqual,
        FloatCC::GreaterThan => FloatCC::LessThan,
        FloatCC::GreaterThanOrEqual => FloatCC::LessThanOrEqual,
        other => other,
    }
}

fn fmov_imm8(double: bool, bits: u64) -> Option<u8> {
    let (frac_drop, exp_bits, sign_shift) = if double { (48, 8, 63) } else { (19, 5, 31) };
    let u = if double { bits } else { bits & 0xFFFF_FFFF };
    if (u & ((1_u64 << frac_drop) - 1)) != 0 {
        return None;
    }
    let b = ((u >> (sign_shift - 1)) & 1) ^ 1;
    let exp_mask = (1_u64 << exp_bits) - 1;
    if ((u >> (frac_drop + 6)) & exp_mask) != if b == 1 { exp_mask } else { 0 } {
        return None;
    }
    let a = (u >> sign_shift) & 1;
    let cdefgh = (u >> frac_drop) & 0x3F;
    Some(((a << 7) | (b << 6) | cdefgh) as u8)
}

impl Emit<'_> {
    pub(super) fn float_unary(&mut self, op: Unary, x: Value, result: Value) -> Result<(), String> {
        let double = self.ty(x) == Type::F64;
        let funary = match op {
            Unary::Fneg => Funary::Neg,
            Unary::Fabs => Funary::Abs,
            Unary::Sqrt => Funary::Sqrt,
            Unary::Floor => Funary::Floor,
            Unary::Ceil => Funary::Ceil,
            Unary::Trunc => Funary::Trunc,
            other => return Err(format!("{other:?} of a float")),
        };
        let dst = self.fdst(result);
        let src = self.fin(x, V0);
        self.asm.funary(funary, double, dst, src);
        self.fput(result, dst);
        Ok(())
    }

    pub(super) fn float_binary(
        &mut self,
        op: Binary,
        x: Value,
        y: Value,
        result: Value,
    ) -> Result<(), String> {
        let ty = self.ty(x);
        let fop = match op {
            Binary::Fadd => Fop::Add,
            Binary::Fsub => Fop::Sub,
            Binary::Fmul => Fop::Mul,
            Binary::Fdiv => Fop::Div,
            Binary::Fmin | Binary::Fmax => {
                let dst = self.fdst(result);
                let left = self.fin(x, V0);
                let right = self.fin(y, V1);
                let base = if op == Binary::Fmin {
                    0x1E20_7800
                } else {
                    0x1E20_6800
                };
                self.asm.word(
                    base | u32::from(ty == Type::F64) << 22
                        | u32::from(right.0) << 16
                        | u32::from(left.0) << 5
                        | u32::from(dst.0),
                );
                self.fput(result, dst);
                return Ok(());
            }
            Binary::Fcopysign => {
                self.copysign(ty, x, y, result);
                return Ok(());
            }
            other => return Err(format!("{other:?} of a float")),
        };
        let dst = self.fdst(result);
        let left = self.fin(x, V0);
        let right = self.fin(y, V1);
        self.asm.fop(fop, ty == Type::F64, dst, left, right);
        self.fput(result, dst);
        Ok(())
    }

    fn is_float_zero(&self, v: Value, ty: Type) -> bool {
        let mag_mask = (ty.sign_bit() as u64).wrapping_sub(1);
        self.const_bits(v)
            .is_some_and(|bits| (bits as u64 & mag_mask) == 0)
    }

    pub(super) fn try_emit_float_imm_bits(&mut self, to: Type, raw: i64, dst: Vreg) -> bool {
        if raw == 0 {
            self.asm.veor_zero(dst);
            self.clobber_vreg(dst);
            return true;
        }
        if !to.is_float() {
            return false;
        }
        let double = to == Type::F64;
        let u = if double {
            raw as u64
        } else {
            raw as u32 as u64
        };
        if u == 0 {
            self.asm.veor_zero(dst);
            self.clobber_vreg(dst);
            return true;
        }
        if let Some(imm8) = fmov_imm8(double, u) {
            self.clobber_vreg(dst);
            self.asm.word(
                0x1E20_1000
                    | (u32::from(double) << 22)
                    | (u32::from(imm8) << 13)
                    | u32::from(dst.0),
            );
            return true;
        }
        let pat = if double { u } else { u | (u << 32) };
        if let Some(word) = super::fuse::encode_neon_movi(0, pat) {
            self.clobber_vreg(dst);
            self.asm.word(word | u32::from(dst.0));
            return true;
        }
        false
    }

    fn try_emit_float_imm(&mut self, to: Type, x: Value, dst: Vreg) -> bool {
        self.const_bits(x)
            .is_some_and(|raw| self.try_emit_float_imm_bits(to, raw, dst))
    }

    fn copysign(&mut self, ty: Type, x: Value, y: Value, result: Value) {
        let vb = self.fin(y, V1);
        let va = self.fin(x, if vb == V0 { V1 } else { V0 });
        let dst = self.fdst(result);
        let out = if dst == vb {
            [V0, V1, V2, V3]
                .into_iter()
                .find(|&r| r != vb && r != va)
                .unwrap_or(V3)
        } else {
            dst
        };
        let mask = [V0, V1, V2, V3]
            .into_iter()
            .find(|&r| r != va && r != vb && r != out)
            .unwrap_or(V2);
        self.clobber_vreg(mask);
        self.asm.word(0x0F04_6400 | u32::from(mask.0));
        if ty == Type::F64 {
            self.asm
                .word(0x5F60_5400 | (u32::from(mask.0) << 5) | u32::from(mask.0));
        }
        if out != va {
            self.asm.vmov(out, va);
        }
        self.clobber_vreg(out);
        self.asm.vbit(0x2EA0_1C00, out, vb, mask);
        self.fput(result, out);
    }

    pub(super) fn fma(
        &mut self,
        a: Value,
        b: Value,
        c: Value,
        result: Value,
    ) -> Result<(), String> {
        if self.plan.is_static(result) || self.plan.deferred[result.index()] {
            return Ok(());
        }
        let ty = self.ty(a);
        if ty.is_vector() {
            return self.vec_fma(a, b, c, result);
        }
        let va = self.fin(a, V0);
        let vb = self.fin(b, V1);
        let vc = self.fin(c, V2);
        let dst = self.fdst(result);
        let word = 0x1F00_0000
            | u32::from(ty == Type::F64) << 22
            | u32::from(vb.0) << 16
            | u32::from(vc.0) << 10
            | u32::from(va.0) << 5
            | u32::from(dst.0);
        self.asm.word(word);
        self.fput(result, dst);
        Ok(())
    }

    pub(super) fn fcmp(&mut self, cc: FloatCC, x: Value, y: Value, result: Value) {
        let ty = self.ty(x);
        if ty.is_vector() {
            self.vec_fcmp(cc, x, y, result);
            return;
        }
        let double = ty == Type::F64;
        let (eff_cc, zero_arg) = if self.is_float_zero(y, ty) {
            (cc, Some(x))
        } else if self.is_float_zero(x, ty) {
            (swap_float_cc(cc), Some(y))
        } else {
            (cc, None)
        };
        if let Some(arg) = zero_arg {
            let reg = self.fin(arg, V0);
            self.asm
                .word(0x1E20_2008 | (u32::from(double) << 22) | (u32::from(reg.0) << 5));
        } else {
            let left = self.fin(x, V0);
            let right = self.fin(y, V1);
            self.asm.fcmp(double, left, right);
        }
        let cond = match eff_cc {
            FloatCC::Equal => Cond::Eq,
            FloatCC::NotEqual => Cond::Ne,
            FloatCC::LessThan => Cond::Mi,
            FloatCC::LessThanOrEqual => Cond::Ls,
            FloatCC::GreaterThan => Cond::Gt,
            FloatCC::GreaterThanOrEqual => Cond::Ge,
        };
        self.set_flag(result, cond);
    }

    pub(super) fn float_convert(
        &mut self,
        op: Convert,
        x: Value,
        result: Value,
    ) -> Result<(), String> {
        let (from, to) = (self.ty(x), self.ty(result));
        if from == Type::I128 || to == Type::I128 {
            return Err(format!("{op:?} from {from:?} to {to:?}"));
        }
        match op {
            Convert::Fpromote | Convert::Fdemote => {
                let dst = self.fdst(result);
                let src = self.fin(x, V0);
                self.asm.fcvt(op == Convert::Fdemote, dst, src);
                self.fput(result, dst);
            }
            Convert::FromSint | Convert::FromUint => {
                let signed = op == Convert::FromSint;
                let src = self.src(x, A, Ext::of(signed));
                let dst = self.fdst(result);
                self.asm.fconv(
                    if signed { Fconv::Scvtf } else { Fconv::Ucvtf },
                    to == Type::F64,
                    true,
                    src,
                    dst,
                );
                self.fput(result, dst);
            }
            Convert::ToSintSat | Convert::ToUintSat => {
                let signed = op == Convert::ToSintSat;
                let src = self.fin(x, V0);
                let dst = self.dst(result);
                let conv = if signed { Fconv::Fcvtzs } else { Fconv::Fcvtzu };
                self.asm
                    .fconv(conv, from == Type::F64, to.bits() > 32, dst, src);
                self.clamp(to.bits(), signed, dst);
                self.put(result, dst);
            }
            Convert::Bitcast if to.is_float() => {
                let dst = self.fdst(result);
                if !self.try_emit_float_imm(to, x, dst) {
                    let src = self.src(x, A, Ext::Zero);
                    self.asm.fmov_in(to == Type::F64, dst, src);
                    self.clobber_vreg(dst);
                }
                self.fput(result, dst);
            }
            Convert::Bitcast => {
                let src = self.fin(x, V0);
                let dst = self.dst(result);
                self.asm.fmov_out(from == Type::F64, dst, src);
                self.put(result, dst);
            }
            other => return Err(format!("{other:?} from {from:?} to {to:?}")),
        }
        Ok(())
    }

    fn clamp(&mut self, bits: u32, signed: bool, dst: super::asm::Reg) {
        if bits >= 32 {
            return;
        }
        if signed {
            self.asm.extend(true, 4, dst, dst);
            self.asm.mov_imm(C, -(1_i64 << (bits - 1)));
            self.asm.alu(Alu::Subs, Size::X, ZR, dst, C);
            self.asm.csel(Cond::Lt, dst, C, dst);
            self.asm.mov_imm(C, (1_i64 << (bits - 1)) - 1);
            self.asm.alu(Alu::Subs, Size::X, ZR, dst, C);
            self.asm.csel(Cond::Gt, dst, C, dst);
        } else {
            self.asm.mov_imm(C, (1_i64 << bits) - 1);
            self.asm.alu(Alu::Subs, Size::X, ZR, dst, C);
            self.asm.csel(Cond::Hi, dst, C, dst);
        }
    }
}

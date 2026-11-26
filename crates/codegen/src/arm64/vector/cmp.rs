use super::super::asm::{Cond, Size, Vreg, ZR};
use super::super::vector_ops::{NEON_BSL, NEON_DUP_GPR_2D};
use super::super::{A, Emit, V0, V1};
use super::{V2, V3, VecSpec};
use crate::ir::{FloatCC, IntCC, Value};

impl Emit<'_> {
    fn vec_icmp_zero(&mut self, spec: VecSpec, val: Value, zcc: IntCC, dst: Vreg) {
        match zcc {
            IntCC::UnsignedLessThan => self.asm.veor_zero(dst),
            IntCC::UnsignedGreaterThanOrEqual => {
                self.asm
                    .word(0x0F07_E7E0 | (spec.q << 30) | u32::from(dst.0));
            }
            IntCC::NotEqual | IntCC::UnsignedGreaterThan => {
                let a = self.fin(val, V0);
                let base = if spec.is_1d() {
                    0x5EE0_8C00
                } else {
                    0x0E20_8C00 | (spec.q << 30) | (spec.size << 22)
                };
                self.asm.vbit(base, dst, a, a);
            }
            _ => {
                let a = self.fin(val, V0);
                let base = match (spec.is_1d(), zcc) {
                    (true, IntCC::Equal | IntCC::UnsignedLessThanOrEqual) => 0x5EE0_9800,
                    (true, IntCC::SignedGreaterThan) => 0x5EE0_8800,
                    (true, IntCC::SignedGreaterThanOrEqual) => 0x7EE0_8800,
                    (true, IntCC::SignedLessThan) => 0x5EE0_A800,
                    (true, IntCC::SignedLessThanOrEqual) => 0x7EE0_9800,
                    (false, IntCC::Equal | IntCC::UnsignedLessThanOrEqual) => 0x0E20_9800,
                    (false, IntCC::SignedGreaterThan) => 0x0E20_8800,
                    (false, IntCC::SignedGreaterThanOrEqual) => 0x2E20_8800,
                    (false, IntCC::SignedLessThan) => 0x0E20_A800,
                    (false, IntCC::SignedLessThanOrEqual) => 0x2E20_9800,
                    _ => unreachable!(),
                };
                let word = if spec.is_1d() {
                    base
                } else {
                    base | (spec.q << 30) | (spec.size << 22)
                };
                self.v_two(word, dst, a);
            }
        }
    }

    pub(in crate::arm64) fn vec_icmp(
        &mut self,
        cc: IntCC,
        x: Value,
        y: Value,
        result: Value,
    ) -> Result<(), String> {
        let spec = VecSpec::of(self.ty(x))?;
        let dst = self.fdst(result);
        let zero_cmp = if self.is_vec_zero(y) {
            Some((x, cc))
        } else if self.is_vec_zero(x) {
            Some((y, cc.swapped()))
        } else {
            None
        };
        if let Some((val, zcc)) = zero_cmp {
            self.vec_icmp_zero(spec, val, zcc, dst);
            self.fput(result, dst);
            return Ok(());
        }
        let a = self.fin(x, V0);
        let b = self.fin(y, V1);
        let (eq_base, gt_base, ge_base, hi_base, hs_base) = if spec.is_1d() {
            (
                0x7EE0_8C00,
                0x5EE0_3400,
                0x5EE0_3C00,
                0x7EE0_3400,
                0x7EE0_3C00,
            )
        } else {
            let tail = (spec.q << 30) | (spec.size << 22);
            (
                0x2E20_8C00 | tail,
                0x0E20_3400 | tail,
                0x0E20_3C00 | tail,
                0x2E20_3400 | tail,
                0x2E20_3C00 | tail,
            )
        };
        match cc {
            IntCC::Equal => self.asm.vbit(eq_base, dst, a, b),
            IntCC::NotEqual => {
                self.asm.vbit(eq_base, dst, a, b);
                self.v_not(spec.q, dst, dst);
            }
            IntCC::SignedGreaterThan => self.asm.vbit(gt_base, dst, a, b),
            IntCC::SignedGreaterThanOrEqual => self.asm.vbit(ge_base, dst, a, b),
            IntCC::SignedLessThan => self.asm.vbit(gt_base, dst, b, a),
            IntCC::SignedLessThanOrEqual => self.asm.vbit(ge_base, dst, b, a),
            IntCC::UnsignedGreaterThan => self.asm.vbit(hi_base, dst, a, b),
            IntCC::UnsignedGreaterThanOrEqual => self.asm.vbit(hs_base, dst, a, b),
            IntCC::UnsignedLessThan => self.asm.vbit(hi_base, dst, b, a),
            IntCC::UnsignedLessThanOrEqual => self.asm.vbit(hs_base, dst, b, a),
        }
        self.fput(result, dst);
        Ok(())
    }

    pub(in crate::arm64) fn vec_fcmp(&mut self, cc: FloatCC, x: Value, y: Value, result: Value) {
        let Ok(spec) = VecSpec::of(self.ty(x)) else {
            return;
        };
        let dst = self.fdst(result);
        let a = self.fin(x, V0);
        let b = self.fin(y, V1);
        let (eq_base, ge_base, gt_base) = if spec.is_1d() {
            (0x5E60_E400, 0x7E60_E400, 0x7EE0_E400)
        } else {
            let tail = (spec.q << 30) | (spec.fsz() << 22);
            (0x0E20_E400 | tail, 0x2E20_E400 | tail, 0x2EA0_E400 | tail)
        };
        match cc {
            FloatCC::Equal => self.asm.vbit(eq_base, dst, a, b),
            FloatCC::NotEqual => {
                self.asm.vbit(eq_base, dst, a, b);
                self.v_not(spec.q, dst, dst);
            }
            FloatCC::GreaterThan => self.asm.vbit(gt_base, dst, a, b),
            FloatCC::GreaterThanOrEqual => self.asm.vbit(ge_base, dst, a, b),
            FloatCC::LessThan => self.asm.vbit(gt_base, dst, b, a),
            FloatCC::LessThanOrEqual => self.asm.vbit(ge_base, dst, b, a),
        }
        self.fput(result, dst);
    }

    pub(in crate::arm64) fn vec_select_mask(
        &mut self,
        condition: Value,
        x: Value,
        y: Value,
        result: Value,
    ) {
        let q = u32::from(self.ty(result).bytes() == 16);
        let dst = self.fdst(result);
        let c = self.fin(condition, V0);
        let a = self.fin(x, V1);
        let b = self.fin(y, V2);
        if dst == a && dst != c {
            self.asm.vbit(0x2EE0_1C00 | (q << 30), dst, b, c);
        } else if dst == b && dst != c {
            self.asm.vbit(0x2EA0_1C00 | (q << 30), dst, a, c);
        } else {
            self.bsl(q, c, a, b, dst);
        }
        self.fput(result, dst);
    }

    pub(in crate::arm64) fn vec_select_cond(
        &mut self,
        cond: Cond,
        x: Value,
        y: Value,
        result: Value,
    ) {
        let q = u32::from(self.ty(result).bytes() == 16);
        let dst = self.fdst(result);
        let a = self.fin(x, V0);
        let b = self.fin(y, V1);
        self.asm.csinv(Size::X, cond.invert(), A, ZR, ZR);
        self.clobber_reg(A);
        let target = if dst != a && dst != b { dst } else { V3 };
        if target == V3 {
            self.clobber_vreg(V3);
        }
        self.asm
            .word(NEON_DUP_GPR_2D | u32::from(A.0) << 5 | u32::from(target.0));
        self.asm.vbit(NEON_BSL | (q << 30), target, a, b);
        self.asm.vmov(dst, target);
        self.fput(result, dst);
    }
}

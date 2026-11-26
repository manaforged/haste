use super::super::asm::{Fop, Funary, Size, Vreg, ZR};
use super::super::vector_ops::{
    NEON_AND, NEON_BSL, NEON_EOR, NEON_ORR, NEON_REV16, NEON_REV32, NEON_REV64, NEON_SMAX,
    NEON_SMIN, NEON_SUB, NEON_SUB_1D, NEON_UMAX, NEON_UMIN, NEON_USHL, NEON_USHL_1D,
};
use super::super::{A, C, D, Emit, S, V0, V1};
use super::{V2, V3, VecSpec};
use crate::ir::{Binary, Lane, Unary, Value};

impl Emit<'_> {
    pub(in crate::arm64) fn vec_unary(
        &mut self,
        op: Unary,
        x: Value,
        result: Value,
    ) -> Result<(), String> {
        let spec = VecSpec::of(self.ty(x))?;
        let dst = self.fdst(result);
        let src = self.fin(x, V0);
        match op {
            Unary::Bnot => self.v_not(spec.q, dst, src),
            Unary::Ineg => self.v_neg(spec, dst, src),
            Unary::Bswap => self.vec_bswap(spec, dst, src),
            Unary::Bitrev => self.vec_bitrev(spec, dst, src),
            Unary::Clz => self.vec_clz(spec, dst, src),
            Unary::Ctz => {
                if spec.lane == Lane::I64 {
                    self.vec_clz_or_ctz_i64(spec, true, dst, src);
                } else {
                    self.vec_bitrev(spec, dst, src);
                    self.v_two(0x2E20_4800 | (spec.q << 30) | (spec.size << 22), dst, dst);
                }
            }
            Unary::Popcnt => {
                self.v_two(0x0E20_5800 | (spec.q << 30), dst, src);
                for step in 0..spec.size {
                    self.v_two(0x2E20_2800 | (spec.q << 30) | (step << 22), dst, dst);
                }
            }
            Unary::Fneg | Unary::Fabs | Unary::Sqrt | Unary::Floor | Unary::Ceil | Unary::Trunc => {
                self.vec_float_unary(spec, op, dst, src)
            }
        }
        self.fput(result, dst);
        Ok(())
    }

    fn vec_bitrev(&mut self, spec: VecSpec, dst: Vreg, src: Vreg) {
        self.v_two(0x2E60_5800 | (spec.q << 30), dst, src);
        if spec.lane != Lane::I8 {
            self.vec_bswap(spec, dst, dst);
        }
    }

    fn vec_float_unary(&mut self, spec: VecSpec, op: Unary, dst: Vreg, src: Vreg) {
        if spec.is_1d() {
            let funary = match op {
                Unary::Fneg => Funary::Neg,
                Unary::Fabs => Funary::Abs,
                Unary::Sqrt => Funary::Sqrt,
                Unary::Floor => Funary::Floor,
                Unary::Ceil => Funary::Ceil,
                Unary::Trunc => Funary::Trunc,
                _ => unreachable!(),
            };
            self.asm.funary(funary, true, dst, src);
            return;
        }
        let base = match op {
            Unary::Fabs => 0x0EA0_F800,
            Unary::Fneg => 0x2EA0_F800,
            Unary::Sqrt => 0x2EA1_F800,
            Unary::Floor => 0x0EA1_8800,
            Unary::Ceil => 0x0E21_9800,
            Unary::Trunc => 0x0EA1_9800,
            _ => unreachable!(),
        };
        self.v_two(base | (spec.q << 30) | (spec.fsz() << 22), dst, src);
    }

    fn vec_bswap(&mut self, spec: VecSpec, dst: Vreg, src: Vreg) {
        let base = match spec.lane {
            Lane::I8 => {
                self.asm.vmov(dst, src);
                return;
            }
            Lane::I16 => NEON_REV16,
            Lane::I32 | Lane::F32 => NEON_REV32,
            Lane::I64 | Lane::F64 => NEON_REV64,
        };
        self.v_two(base | (spec.q << 30), dst, src);
    }

    fn vec_clz(&mut self, spec: VecSpec, dst: Vreg, src: Vreg) {
        if spec.lane != Lane::I64 {
            self.v_two(0x2E20_4800 | (spec.q << 30) | (spec.size << 22), dst, src);
        } else {
            self.vec_clz_or_ctz_i64(spec, false, dst, src);
        }
    }

    pub(in crate::arm64) fn vec_binary(
        &mut self,
        op: Binary,
        x: Value,
        y: Value,
        result: Value,
    ) -> Result<(), String> {
        let spec = VecSpec::of(self.ty(x))?;
        let dst = self.fdst(result);
        let a = self.fin(x, V0);
        if matches!(op, Binary::Ishl | Binary::Ushr | Binary::Sshr)
            && let Some(imm) = self.splat_const_lane(y)
        {
            self.vec_shift_imm(spec, op, imm, dst, a);
            self.fput(result, dst);
            return Ok(());
        }
        let b = self.fin(y, V1);
        match op {
            Binary::Band | Binary::Bor | Binary::Bxor => self.vec_bitwise(spec, op, dst, a, b),
            Binary::Iadd | Binary::Isub => self.vec_add_sub(spec, op == Binary::Isub, dst, a, b),
            Binary::Imul => self.vec_imul(spec, dst, a, b),
            Binary::Umulhi | Binary::Smulhi if spec.lane == Lane::I16 => {
                self.vec_mulhi_i16(spec, op == Binary::Umulhi, dst, a, b);
            }
            Binary::Smin | Binary::Smax | Binary::Umin | Binary::Umax => {
                self.vec_minmax_int(spec, op, dst, a, b);
            }
            Binary::Ishl | Binary::Ushr | Binary::Sshr => self.vec_shift_var(spec, op, dst, a, b),
            Binary::Fcopysign => self.vec_fcopysign(spec, dst, a, b),
            Binary::Fadd
            | Binary::Fsub
            | Binary::Fmul
            | Binary::Fdiv
            | Binary::Fmin
            | Binary::Fmax => self.vec_float_binary(spec, op, dst, a, b),
            Binary::SaddSat | Binary::UaddSat | Binary::SsubSat | Binary::UsubSat => {
                self.vec_sat_arith(spec, op, dst, a, b);
            }
            Binary::AvgRound => self.vec_avg_round(spec, dst, a, b),
            other => return Err(format!("{other:?} of a vector")),
        }
        self.fput(result, dst);
        Ok(())
    }

    fn vec_bitwise(&mut self, spec: VecSpec, op: Binary, dst: Vreg, a: Vreg, b: Vreg) {
        let base = match op {
            Binary::Band => NEON_AND,
            Binary::Bor => NEON_ORR,
            _ => NEON_EOR,
        };
        self.asm.vbit(base | (spec.q << 30), dst, a, b);
    }

    fn vec_add_sub(&mut self, spec: VecSpec, sub: bool, dst: Vreg, a: Vreg, b: Vreg) {
        let base = match (spec.is_1d(), sub) {
            (true, false) => 0x5EE0_8400,
            (true, true) => NEON_SUB_1D,
            (false, false) => 0x0E20_8400 | (spec.q << 30) | (spec.size << 22),
            (false, true) => NEON_SUB | (spec.q << 30) | (spec.size << 22),
        };
        self.asm.vbit(base, dst, a, b);
    }

    fn vec_imul(&mut self, spec: VecSpec, dst: Vreg, a: Vreg, b: Vreg) {
        if spec.lane != Lane::I64 {
            self.asm
                .vbit(0x0E20_9C00 | (spec.q << 30) | (spec.size << 22), dst, a, b);
            return;
        }
        self.asm.fmov_out(true, A, a);
        self.asm.fmov_out(true, C, b);
        self.asm.madd(false, Size::X, D, (A, C), ZR);
        if spec.count == 2 {
            self.asm.umov_d1(A, a);
            self.asm.umov_d1(C, b);
            self.asm.madd(false, Size::X, S, (A, C), ZR);
        }
        self.clobber_regs(&[A, C, D, S]);
        self.asm.fmov_in(true, dst, D);
        if spec.count == 2 {
            self.ins_gpr(dst, 3, 1, S);
        }
    }

    fn vec_mulhi_i16(&mut self, spec: VecSpec, unsigned: bool, dst: Vreg, a: Vreg, b: Vreg) {
        let u = u32::from(unsigned) << 29;
        self.clobber_vreg(V2);
        self.asm.vbit(0x0E60_C000 | u, V2, a, b);
        if spec.q == 1 {
            self.clobber_vreg(V3);
            self.asm.vbit(0x4E60_C000 | u, V3, a, b);
            self.asm.vbit(0x4E40_5800, dst, V2, V3);
        } else {
            self.asm.vbit(0x4E40_5800, dst, V2, V2);
        }
    }

    fn vec_minmax_int(&mut self, spec: VecSpec, op: Binary, dst: Vreg, a: Vreg, b: Vreg) {
        if spec.lane != Lane::I64 {
            let base = match op {
                Binary::Smax => NEON_SMAX,
                Binary::Smin => NEON_SMIN,
                Binary::Umax => NEON_UMAX,
                _ => NEON_UMIN,
            };
            self.asm
                .vbit(base | (spec.q << 30) | (spec.size << 22), dst, a, b);
            return;
        }
        let signed = matches!(op, Binary::Smin | Binary::Smax);
        let cmp = match (spec.is_1d(), signed) {
            (true, true) => 0x5EE0_3400,
            (true, false) => 0x7EE0_3400,
            (false, true) => 0x4EE0_3400,
            (false, false) => 0x6EE0_3400,
        };
        let (t, f) = if matches!(op, Binary::Smax | Binary::Umax) {
            (a, b)
        } else {
            (b, a)
        };
        if dst != a && dst != b {
            self.asm.vbit(cmp, dst, a, b);
            self.asm.vbit(NEON_BSL | (spec.q << 30), dst, t, f);
        } else {
            self.clobber_vreg(V2);
            self.asm.vbit(cmp, V2, a, b);
            self.asm.vbit(NEON_BSL | (spec.q << 30), V2, t, f);
            self.asm.vmov(dst, V2);
        }
    }

    fn vec_shift_var(&mut self, spec: VecSpec, op: Binary, dst: Vreg, a: Vreg, b: Vreg) {
        if op == Binary::Ishl {
            let base = if spec.is_1d() {
                NEON_USHL_1D
            } else {
                NEON_USHL | (spec.q << 30) | (spec.size << 22)
            };
            self.asm.vbit(base, dst, a, b);
            return;
        }
        self.clobber_vreg(V2);
        self.v_neg(spec, V2, b);
        let base = match (spec.is_1d(), op == Binary::Sshr) {
            (true, false) => NEON_USHL_1D,
            (true, true) => 0x5EE0_4400,
            (false, false) => NEON_USHL | (spec.q << 30) | (spec.size << 22),
            (false, true) => 0x0E20_4400 | (spec.q << 30) | (spec.size << 22),
        };
        self.asm.vbit(base, dst, a, V2);
    }

    fn vec_float_binary(&mut self, spec: VecSpec, op: Binary, dst: Vreg, a: Vreg, b: Vreg) {
        if spec.is_1d() {
            match op {
                Binary::Fadd => self.asm.fop(Fop::Add, true, dst, a, b),
                Binary::Fsub => self.asm.fop(Fop::Sub, true, dst, a, b),
                Binary::Fmul => self.asm.fop(Fop::Mul, true, dst, a, b),
                Binary::Fdiv => self.asm.fop(Fop::Div, true, dst, a, b),
                Binary::Fmin => self.asm.vbit(0x1E60_7800, dst, a, b),
                Binary::Fmax => self.asm.vbit(0x1E60_6800, dst, a, b),
                _ => unreachable!(),
            }
            return;
        }
        let base = match op {
            Binary::Fadd => 0x0E20_D400,
            Binary::Fsub => 0x0EA0_D400,
            Binary::Fmul => 0x2E20_DC00,
            Binary::Fdiv => 0x2E20_FC00,
            Binary::Fmax => 0x0E20_C400,
            Binary::Fmin => 0x0EA0_C400,
            _ => unreachable!(),
        };
        self.asm
            .vbit(base | (spec.q << 30) | (spec.fsz() << 22), dst, a, b);
    }

    fn vec_sat_arith(&mut self, spec: VecSpec, op: Binary, dst: Vreg, a: Vreg, b: Vreg) {
        let base = match (spec.is_1d(), op) {
            (true, Binary::SaddSat) => 0x5EE0_0C00,
            (true, Binary::UaddSat) => 0x7EE0_0C00,
            (true, Binary::SsubSat) => 0x5EE0_2C00,
            (true, Binary::UsubSat) => 0x7EE0_2C00,
            (false, Binary::SaddSat) => 0x0E20_0C00 | (spec.q << 30) | (spec.size << 22),
            (false, Binary::UaddSat) => 0x2E20_0C00 | (spec.q << 30) | (spec.size << 22),
            (false, Binary::SsubSat) => 0x0E20_2C00 | (spec.q << 30) | (spec.size << 22),
            (false, Binary::UsubSat) => 0x2E20_2C00 | (spec.q << 30) | (spec.size << 22),
            _ => unreachable!(),
        };
        self.asm.vbit(base, dst, a, b);
    }

    fn vec_avg_round(&mut self, spec: VecSpec, dst: Vreg, a: Vreg, b: Vreg) {
        if spec.lane != Lane::I64 {
            self.asm
                .vbit(0x2E20_1400 | (spec.q << 30) | (spec.size << 22), dst, a, b);
            return;
        }
        self.clobber_vreg(V2);
        self.clobber_vreg(V3);
        self.asm.vbit(NEON_ORR | (spec.q << 30), V2, a, b);
        self.asm.vbit(NEON_EOR | (spec.q << 30), V3, a, b);
        self.vec_shift_imm(spec, Binary::Ushr, 1, V3, V3);
        self.vec_add_sub(spec, true, dst, V2, V3);
    }

    pub(in crate::arm64) fn vec_fma(
        &mut self,
        a: Value,
        b: Value,
        c: Value,
        result: Value,
    ) -> Result<(), String> {
        let spec = VecSpec::of(self.ty(a))?;
        let va = self.fin(a, V0);
        let vb = self.fin(b, V1);
        let vc = self.fin(c, V2);
        let dst = self.fdst(result);
        if spec.is_1d() {
            let word = 0x1F40_0000
                | u32::from(vb.0) << 16
                | u32::from(vc.0) << 10
                | u32::from(va.0) << 5
                | u32::from(dst.0);
            self.asm.word(word);
        } else {
            let acc = if dst == va || dst == vb { V3 } else { dst };
            if acc == V3 {
                self.clobber_vreg(V3);
            }
            self.asm.vmov(acc, vc);
            self.asm.vbit(
                0x0E20_CC00 | (spec.q << 30) | (spec.fsz() << 22),
                acc,
                va,
                vb,
            );
            self.asm.vmov(dst, acc);
        }
        self.fput(result, dst);
        Ok(())
    }

    pub(in crate::arm64) fn vec_shift_imm(
        &mut self,
        spec: VecSpec,
        op: Binary,
        imm: i64,
        dst: Vreg,
        src: Vreg,
    ) {
        let esize = spec.lane.bits();
        let shift = (imm as u32) & (esize - 1);
        if shift == 0 {
            self.asm.vmov(dst, src);
            return;
        }
        let (base, enc_imm) = match (op, spec.is_1d()) {
            (Binary::Ishl, true) => (0x5F00_5400, esize + shift),
            (Binary::Ishl, false) => (0x0F00_5400 | (spec.q << 30), esize + shift),
            (Binary::Sshr, true) => (0x5F00_0400, 2 * esize - shift),
            (Binary::Sshr, false) => (0x0F00_0400 | (spec.q << 30), 2 * esize - shift),
            (Binary::Ushr, true) => (0x7F00_0400, 2 * esize - shift),
            (Binary::Ushr, false) => (0x2F00_0400 | (spec.q << 30), 2 * esize - shift),
            _ => unreachable!(),
        };
        self.v_two(base | (enc_imm << 16), dst, src);
    }
}

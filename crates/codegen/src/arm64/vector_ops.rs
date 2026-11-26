mod reduce;
mod shuffle;

use super::asm::{Bits, Vreg, ZR};
use super::vector::{V2, V3, VecSpec};
use super::{A, C, Emit, V0, V1};
use crate::ir::{Binary, Lane, TargetOpId, TinyArgs, Value};

pub(super) const NEON_EXT: u32 = 0x2E00_0000;
pub(super) const NEON_AND: u32 = 0x0E20_1C00;
pub(super) const NEON_ORR: u32 = 0x0EA0_1C00;
pub(super) const NEON_EOR: u32 = 0x2E20_1C00;
pub(super) const NEON_BSL: u32 = 0x2E60_1C00;
pub(super) const NEON_DUP_GPR_2D: u32 = 0x4E08_0C00;
pub(super) const NEON_SMAX: u32 = 0x0E20_6400;
pub(super) const NEON_SMIN: u32 = 0x0E20_6C00;
pub(super) const NEON_UMAX: u32 = 0x2E20_6400;
pub(super) const NEON_UMIN: u32 = 0x2E20_6C00;
pub(super) const NEON_REV64: u32 = 0x0E20_0800;
pub(super) const NEON_REV32: u32 = 0x2E20_0800;
pub(super) const NEON_REV16: u32 = 0x0E20_1800;
pub(super) const NEON_SUB: u32 = 0x2E20_8400;
pub(super) const NEON_SUB_1D: u32 = 0x7EE0_8400;
pub(super) const NEON_USHL: u32 = 0x2E20_4400;
pub(super) const NEON_USHL_1D: u32 = 0x7EE0_4400;

#[derive(Clone, Copy)]
enum TargetKind {
    TwoInPlace,
    ThreeInPlace,
    OneReg,
}

struct TargetDesc {
    id: TargetOpId,
    base: u32,
    kind: TargetKind,
}

const ARM64_TARGET_OPS: &[TargetDesc] = &[
    TargetDesc {
        id: TargetOpId::Arm64Sha256H,
        base: 0x5E00_4000,
        kind: TargetKind::ThreeInPlace,
    },
    TargetDesc {
        id: TargetOpId::Arm64Sha256H2,
        base: 0x5E00_5000,
        kind: TargetKind::ThreeInPlace,
    },
    TargetDesc {
        id: TargetOpId::Arm64Sha256Su1,
        base: 0x5E00_6000,
        kind: TargetKind::ThreeInPlace,
    },
    TargetDesc {
        id: TargetOpId::Arm64Sha256Su0,
        base: 0x5E28_2800,
        kind: TargetKind::TwoInPlace,
    },
    TargetDesc {
        id: TargetOpId::Arm64AesE,
        base: 0x4E28_4800,
        kind: TargetKind::TwoInPlace,
    },
    TargetDesc {
        id: TargetOpId::Arm64AesD,
        base: 0x4E28_5800,
        kind: TargetKind::TwoInPlace,
    },
    TargetDesc {
        id: TargetOpId::Arm64AesMc,
        base: 0x4E28_6800,
        kind: TargetKind::OneReg,
    },
    TargetDesc {
        id: TargetOpId::Arm64AesImc,
        base: 0x4E28_7800,
        kind: TargetKind::OneReg,
    },
];

impl Emit<'_> {
    pub(super) fn vec_clz_or_ctz_i64(&mut self, spec: VecSpec, ctz: bool, dst: Vreg, src: Vreg) {
        self.asm.fmov_out(true, A, src);
        self.clobber_reg(A);
        if ctz {
            self.asm.bits(Bits::Rbit, A, A);
        }
        self.asm.bits(Bits::Clz, A, A);
        if spec.count == 2 {
            self.asm.umov_d1(C, src);
            self.clobber_reg(C);
            if ctz {
                self.asm.bits(Bits::Rbit, C, C);
            }
            self.asm.bits(Bits::Clz, C, C);
        }
        self.asm.fmov_in(true, dst, A);
        if spec.count == 2 {
            self.ins_gpr(dst, 3, 1, C);
        }
    }

    pub(super) fn vec_extract(&mut self, vec: Value, idx: u8, result: Value) -> Result<(), String> {
        let spec = VecSpec::of(self.ty(vec))?;
        let src = self.fin(vec, V0);
        if spec.lane.is_float() {
            let dst = self.fdst(result);
            if idx == 0 {
                self.asm.fmov_reg(spec.lane == Lane::F64, dst, src);
            } else {
                self.dup_scalar_elem(spec.size, idx, dst, src);
            }
            self.fput(result, dst);
        } else {
            let dst = self.dst(result);
            self.umov_gpr(spec.size, idx, dst, src);
            self.put(result, dst);
        }
        Ok(())
    }

    pub(super) fn vec_insert(
        &mut self,
        vec: Value,
        val: Value,
        idx: u8,
        result: Value,
    ) -> Result<(), String> {
        let spec = VecSpec::of(self.ty(vec))?;
        let dst = self.fdst(result);
        let src = self.fin(vec, dst);
        if spec.lane.is_float() {
            if self.const_bits(val) == Some(0) {
                self.asm.vmov(dst, src);
                self.ins_gpr(dst, spec.size, idx, ZR);
            } else {
                let mut s = self.fin(val, V1);
                if s == dst && src != dst {
                    self.asm.vmov(V1, s);
                    self.clobber_vreg(V1);
                    s = V1;
                }
                self.asm.vmov(dst, src);
                self.ins_elem(dst, spec.size, idx, s, 0);
            }
        } else {
            let s = if self.plan.const_of(val) == Some(0) {
                ZR
            } else {
                self.bits(val, A)
            };
            self.asm.vmov(dst, src);
            self.ins_gpr(dst, spec.size, idx, s);
        }
        self.fput(result, dst);
        Ok(())
    }

    pub(super) fn vec_target_op(
        &mut self,
        id: TargetOpId,
        args: TinyArgs,
        result: Value,
    ) -> Result<(), String> {
        if id == TargetOpId::Arm64Sqdmulh {
            let spec = VecSpec::of(self.ty(result))?;
            let args = args.as_slice();
            let dst = self.fdst(result);
            let a = self.fin(args[0], V0);
            let b = self.fin(args[1], V1);
            self.asm
                .vbit(0x0E20_B400 | (spec.q << 30) | (spec.size << 22), dst, a, b);
            self.fput(result, dst);
            return Ok(());
        }
        let Some(desc) = ARM64_TARGET_OPS.iter().find(|d| d.id == id) else {
            return Err(format!("unsupported ARM64 target op {id:?}"));
        };
        let args = args.as_slice();
        let base = desc.base;
        let dst = self.fdst(result);
        match desc.kind {
            TargetKind::OneReg => {
                let a = self.fin(args[0], V0);
                self.v_two(base, dst, a);
            }
            TargetKind::TwoInPlace => {
                let a = self.fin(args[0], V0);
                let mut b = self.fin(args[1], V1);
                if b == dst && a != dst {
                    self.asm.vmov(V1, b);
                    self.clobber_vreg(V1);
                    b = V1;
                }
                self.asm.vmov(dst, a);
                self.v_two(base, dst, b);
            }
            TargetKind::ThreeInPlace => {
                let a = self.fin(args[0], V0);
                let mut b = self.fin(args[1], V1);
                let mut c = self.fin(args[2], V2);
                if dst != a {
                    if b == dst {
                        self.asm.vmov(V1, b);
                        self.clobber_vreg(V1);
                        b = V1;
                    }
                    if c == dst {
                        self.asm.vmov(V2, c);
                        self.clobber_vreg(V2);
                        c = V2;
                    }
                    self.asm.vmov(dst, a);
                }
                self.asm.vbit(base, dst, b, c);
            }
        }
        self.fput(result, dst);
        Ok(())
    }

    pub(super) fn emit_neon_imm_only(&mut self, dst: Vreg, full: bool, lo: u64, hi: u64) -> bool {
        if lo == 0 && (!full || hi == 0) {
            self.asm.veor_zero(dst);
            return true;
        }
        if (!full || lo == hi)
            && let Some(word) = encode_neon_imm(u32::from(full), lo)
        {
            self.clobber_vreg(dst);
            self.asm.word(word | u32::from(dst.0));
            return true;
        }
        if full
            && hi == 0
            && let Some(word) = encode_neon_imm(0, lo)
        {
            self.clobber_vreg(dst);
            self.asm.word(word | u32::from(dst.0));
            return true;
        }
        false
    }

    fn emit_ext_pair_imm(&mut self, dst: Vreg, tmp: Vreg, lo: u64, hi: u64) -> bool {
        let Some(w_hi) = (dst != tmp && can_emit_ext_pair_imm(lo, hi))
            .then(|| encode_neon_imm(0, hi))
            .flatten()
        else {
            return false;
        };
        if lo == 0 {
            self.asm.veor_zero(dst);
        } else if let Some(w_lo) = encode_neon_imm(1, lo) {
            self.asm.word(w_lo | u32::from(dst.0));
        }
        self.clobber_vreg(tmp);
        self.asm.word(w_hi | u32::from(tmp.0));
        self.asm
            .vbit(NEON_EXT | (1 << 30) | (8 << 11), dst, dst, tmp);
        true
    }

    fn emit_dup_i64(&mut self, dst: Vreg, lo: u64) {
        self.asm.mov_imm(A, lo as i64);
        self.clobber_reg(A);
        self.asm
            .word(NEON_DUP_GPR_2D | u32::from(A.0) << 5 | u32::from(dst.0));
    }

    pub(super) fn try_emit_const_vec_imm(&mut self, result: Value) -> bool {
        let Some((full, lo, hi)) = self.eval_vec_full(result) else {
            return false;
        };
        let dst = self.fdst(result);
        self.emit_neon_imm_only(dst, full, lo, hi)
            .then(|| self.fput(result, dst))
            .is_some()
    }

    pub(super) fn emit_insert_i64_pair(&mut self, dst: Vreg, lo: Value, hi: Value) {
        if let (Some(c_lo), Some(c_hi)) = (self.const_bits(lo), self.const_bits(hi)) {
            let (u_lo, u_hi) = (c_lo as u64, c_hi as u64);
            if self.emit_neon_imm_only(dst, true, u_lo, u_hi)
                || self.emit_ext_pair_imm(dst, V2, u_lo, u_hi)
            {
                return;
            }
            if u_lo == u_hi {
                let s = self.bits(lo, A);
                self.asm
                    .word(NEON_DUP_GPR_2D | u32::from(s.0) << 5 | u32::from(dst.0));
                return;
            }
        }
        if !self
            .const_bits(lo)
            .is_some_and(|c| self.emit_neon_imm_only(dst, false, c as u64, 0))
        {
            let s_lo = self.bits(lo, A);
            self.asm.fmov_in(true, dst, s_lo);
        }
        if self.const_bits(hi) != Some(0) {
            let s_hi = self.bits(hi, C);
            self.ins_gpr(dst, 3, 1, s_hi);
        }
    }

    pub(super) fn try_vec_insert_const(
        &mut self,
        vec: Value,
        val: Value,
        idx: u8,
        spec: VecSpec,
        result: Value,
    ) -> bool {
        if self.try_emit_const_vec_imm(result) {
            return true;
        }
        if idx != 0 || !self.is_vec_zero(vec) {
            return false;
        }
        if matches!(spec.lane, Lane::I32 | Lane::I64) {
            let (dst, s) = (self.fdst(result), self.bits(val, A));
            self.asm.fmov_in(spec.lane == Lane::I64, dst, s);
            self.fput(result, dst);
            return true;
        }
        if matches!(spec.lane, Lane::I8 | Lane::I16)
            && let Some(imm) = self.const_bits(val)
        {
            let dst = self.fdst(result);
            self.asm
                .mov_imm(A, imm & if spec.lane == Lane::I8 { 0xFF } else { 0xFFFF });
            self.clobber_reg(A);
            self.asm.fmov_in(false, dst, A);
            self.fput(result, dst);
            return true;
        }
        false
    }

    pub(super) fn emit_vec_from_i64(&mut self, dst: Vreg, x: Value) {
        if !self
            .const_bits(x)
            .is_some_and(|imm| self.emit_neon_imm_only(dst, false, imm as u64, 0))
        {
            let src = self.bits(x, A);
            self.asm.fmov_in(true, dst, src);
        }
    }

    pub(super) fn vec_fcopysign(&mut self, spec: VecSpec, dst: Vreg, a: Vreg, b: Vreg) {
        self.clobber_vreg(V2);
        self.asm.word(
            (if spec.lane == Lane::F64 {
                0x0F00_E420
            } else {
                0x0F04_6400
            }) | (spec.q << 30)
                | u32::from(V2.0),
        );
        if spec.lane == Lane::F64 {
            self.vec_shift_imm(spec, Binary::Ishl, 63, V2, V2);
        }
        let out = if dst == b {
            self.clobber_vreg(V3);
            V3
        } else {
            dst
        };
        self.asm.vmov(out, a);
        self.asm.vbit(0x2EA0_1C00 | (spec.q << 30), out, b, V2);
        self.asm.vmov(dst, out);
    }
}

fn can_emit_ext_pair_imm(lo: u64, hi: u64) -> bool {
    (lo == 0 || encode_neon_imm(1, lo).is_some()) && encode_neon_imm(0, hi).is_some()
}

fn neon_fmov_imm8(bits: u64, is_f64: bool) -> Option<u8> {
    let (low_mask, mant_shift, exp_bits, total) = if is_f64 {
        ((1_u64 << 48) - 1, 48, 8, 64)
    } else {
        ((1_u64 << 19) - 1, 19, 5, 32)
    };
    let (exp_mask, b) = ((1_u64 << exp_bits) - 1, ((bits >> (total - 2)) & 1) ^ 1);
    if (bits & low_mask) != 0
        || ((bits >> (mant_shift + 6)) & exp_mask) != if b == 1 { exp_mask } else { 0 }
    {
        return None;
    }
    Some((((bits >> (total - 1)) & 1) << 7 | (b << 6) | ((bits >> mant_shift) & 0x3F)) as u8)
}

pub(super) fn encode_neon_imm(q: u32, pattern: u64) -> Option<u32> {
    if let Some(word) = super::fuse::encode_neon_movi(q, pattern) {
        return Some(word);
    }
    let v32 = pattern as u32;
    if v32 == (pattern >> 32) as u32
        && let Some(i) = neon_fmov_imm8(u64::from(v32), false).map(u32::from)
    {
        return Some(0x0F00_F400 | (q << 30) | ((i >> 5) << 16) | ((i & 0x1F) << 5));
    }
    if let Some(i) = neon_fmov_imm8(pattern, true).map(u32::from) {
        return Some(if q == 1 {
            0x6F00_F400 | ((i >> 5) << 16) | ((i & 0x1F) << 5)
        } else {
            0x1E60_1000 | (i << 13)
        });
    }
    (q == 0 && (pattern >> 32) == 0)
        .then(|| neon_fmov_imm8(pattern, false).map(|i| 0x1E20_1000 | (u32::from(i) << 13)))
        .flatten()
}

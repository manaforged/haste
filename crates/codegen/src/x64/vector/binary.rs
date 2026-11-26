use super::{float_vec_opcode, is_commutative, shift_count_opcode, vec_alu_opcode, vec_prefix};
use crate::frame::Ext;
use crate::ir::{Binary, IntCC, Lane, Op, ShuffleMask, Value};
use crate::x64::Emit;
use crate::x64::asm::{Alu, Reg, Xmm};
use crate::x64::float::{X0, X1, X2};
use crate::x64::vector_narrow::PMULUDQ;

impl Emit<'_> {
    pub(in crate::x64) fn vector_binary(
        &mut self,
        op: Binary,
        (x, y): (Value, Value),
        result: Value,
    ) -> Result<(), String> {
        let Some((lane, count)) = self.ty(x).vec_shape() else {
            return Err(format!("non-vector type {:?} in vector_binary", self.ty(x)));
        };
        if self.ty(x).is_wide_vec() {
            return self.ymm_binary(op, (x, y), result);
        }
        if let Some(opcode) = vec_alu_opcode(op, lane) {
            self.vex_packed_op(&[0x66], &[0x0F, opcode], (x, y), result, is_commutative(op));
            return Ok(());
        }
        if let Some(opcode) = float_vec_opcode(op) {
            if !lane.is_float() {
                return Err(format!("{op:?} of integer vector {lane:?}"));
            }
            self.vex_packed_op(
                vec_prefix(lane),
                &[0x0F, opcode],
                (x, y),
                result,
                is_commutative(op),
            );
            return Ok(());
        }
        match op {
            Binary::Fmin | Binary::Fmax if lane.is_float() => {
                self.vector_fminmax(op, lane, x, y, result)
            }
            Binary::Fcopysign if lane.is_float() => {
                let dst = self.xmm_of(result).unwrap_or(X2);
                self.fget(X0, y);
                self.fget(dst, x);
                self.sse_or_vex(&[0x66], &[0x0F, 0x76], (X1, X1, X1));
                self.asm
                    .pshift_imm(6, lane.bytes(), X1, lane.bits() as u8 - 1);
                self.asm.sse(&[0x66], &[0x0F, 0xDB], X0.0, X1);
                self.asm.sse(&[0x66], &[0x0F, 0xDF], X1.0, dst);
                self.asm.sse(&[0x66], &[0x0F, 0xEB], X1.0, X0);
                self.asm.movaps(dst, X1);
                self.fput(result, dst);
                Ok(())
            }
            Binary::Imul if lane.is_int() => self.vector_imul(lane, count, x, y, result),
            Binary::Smin | Binary::Smax | Binary::Umin | Binary::Umax if lane.is_int() => {
                self.vector_iminmax(op, lane, count, x, y, result)
            }
            Binary::SaddSat
            | Binary::UaddSat
            | Binary::SsubSat
            | Binary::UsubSat
            | Binary::AvgRound
                if lane.is_int() =>
            {
                self.vector_sat_avg(op, lane, count, x, y, result)
            }
            Binary::Ishl | Binary::Ushr | Binary::Sshr if lane.is_int() => {
                self.vector_shift(op, lane, count, x, y, result)
            }
            _ => Err(format!("{op:?} of vector {lane:?}")),
        }
    }

    fn vector_sat_avg(
        &mut self,
        op: Binary,
        lane: Lane,
        count: u8,
        x: Value,
        y: Value,
        result: Value,
    ) -> Result<(), String> {
        let sse_opc = match (op, lane) {
            (Binary::SaddSat, Lane::I8) => Some(0xEC),
            (Binary::SaddSat, Lane::I16) => Some(0xED),
            (Binary::UaddSat, Lane::I8) => Some(0xDC),
            (Binary::UaddSat, Lane::I16) => Some(0xDD),
            (Binary::SsubSat, Lane::I8) => Some(0xE8),
            (Binary::SsubSat, Lane::I16) => Some(0xE9),
            (Binary::UsubSat, Lane::I8) => Some(0xD8),
            (Binary::UsubSat, Lane::I16) => Some(0xD9),
            (Binary::AvgRound, Lane::I8) => Some(0xE0),
            (Binary::AvgRound, Lane::I16) => Some(0xE3),
            _ => None,
        };
        if let Some(opc) = sse_opc {
            self.vex_packed_op(&[0x66], &[0x0F, opc], (x, y), result, is_commutative(op));
            return Ok(());
        }
        let dst = self.xmm_of(result).unwrap_or(X2);
        if op == Binary::AvgRound {
            self.fget(X1, y);
            self.fget(dst, x);
            self.asm.movaps(X0, dst);
            self.asm.sse(&[0x66], &[0x0F, 0xEB], dst.0, X1);
            self.asm.sse(&[0x66], &[0x0F, 0xEF], X0.0, X1);
            self.asm.pshift_imm(2, lane.bytes(), X0, 1);
            let sub = vec_alu_opcode(Binary::Isub, lane)
                .ok_or_else(|| format!("AvgRound of {lane:?}"))?;
            self.asm.sse(&[0x66], &[0x0F, sub], dst.0, X0);
        } else if lane == Lane::I32
            && self.features.sse41
            && matches!(op, Binary::UaddSat | Binary::UsubSat)
        {
            self.fget(X1, y);
            self.fget(dst, x);
            if op == Binary::UaddSat {
                self.asm.sse(&[0x66], &[0x0F, 0x76], X0.0, X0);
                self.asm.sse(&[0x66], &[0x0F, 0xEF], X0.0, X1);
                self.asm.sse(&[0x66], &[0x0F, 0x38, 0x3B], dst.0, X0);
                self.asm.sse(&[0x66], &[0x0F, 0xFE], dst.0, X1);
            } else {
                self.asm.sse(&[0x66], &[0x0F, 0x38, 0x3F], dst.0, X1);
                self.asm.sse(&[0x66], &[0x0F, 0xFA], dst.0, X1);
            }
        } else {
            self.scalarize_int_binary(op, lane, count, x, y, dst);
        }
        self.fput(result, dst);
        Ok(())
    }

    fn vector_fminmax(
        &mut self,
        op: Binary,
        lane: Lane,
        x: Value,
        y: Value,
        result: Value,
    ) -> Result<(), String> {
        let dst = self.xmm_of(result).unwrap_or(X2);
        self.fget(X0, x);
        self.fget(X1, y);
        self.fminmax_regs(op, lane, dst);
        self.fput(result, dst);
        Ok(())
    }

    pub(in crate::x64) fn fminmax_regs(&mut self, op: Binary, lane: Lane, dst: Xmm) {
        let opcode = if op == Binary::Fmin { 0x5D } else { 0x5F };
        let pf = vec_prefix(lane);
        self.asm.movaps(dst, X0);
        self.asm.sse(pf, &[0x0F, opcode], dst.0, X1);
        self.asm.sse_imm(pf, &[0x0F, 0xC2], X1.0, X1, 3);
        self.asm.sse(&[0x66], &[0x0F, 0xDB], X0.0, X1);
        self.asm.sse(&[0x66], &[0x0F, 0xDF], X1.0, dst);
        self.asm.sse(&[0x66], &[0x0F, 0xEB], X1.0, X0);
        self.asm.movaps(dst, X1);
    }

    fn vector_imul(
        &mut self,
        lane: Lane,
        count: u8,
        x: Value,
        y: Value,
        result: Value,
    ) -> Result<(), String> {
        let dst = self.xmm_of(result).unwrap_or(X2);
        match lane {
            Lane::I16 => self.vex_packed_regs(&[0x66], &[0x0F, 0xD5], (x, y), dst, true),
            Lane::I32 if self.features.sse41 => {
                self.vex_packed_regs(&[0x66], &[0x0F, 0x38, 0x40], (x, y), dst, true)
            }
            Lane::I32 => {
                self.fget(X0, x);
                self.fget(X1, y);
                self.asm.movaps(dst, X0);
                self.asm.sse(&[0x66], PMULUDQ, dst.0, X1);
                self.asm.pshift_imm(2, 8, X0, 32);
                self.asm.pshift_imm(2, 8, X1, 32);
                self.asm.sse(&[0x66], PMULUDQ, X0.0, X1);
                self.asm.sse_imm(&[0x66], &[0x0F, 0x70], dst.0, dst, 0x08);
                self.asm.sse_imm(&[0x66], &[0x0F, 0x70], X0.0, X0, 0x08);
                self.asm.sse(&[0x66], &[0x0F, 0x62], dst.0, X0);
            }
            Lane::I64 => self.imul_quads((x, y), dst),
            Lane::I8 => self.scalarize_int_binary(Binary::Imul, lane, count, x, y, dst),
            _ => return Err(format!("Imul of {lane:?}")),
        }
        self.fput(result, dst);
        Ok(())
    }

    pub(in crate::x64) fn iminmax_opc(&self, op: Binary, lane: Lane) -> Option<&'static [u8]> {
        match (op, lane) {
            (Binary::Umin, Lane::I8) => Some(&[0x0F, 0xDA]),
            (Binary::Umax, Lane::I8) => Some(&[0x0F, 0xDE]),
            (Binary::Smin, Lane::I16) => Some(&[0x0F, 0xEA]),
            (Binary::Smax, Lane::I16) => Some(&[0x0F, 0xEE]),
            (Binary::Smin, Lane::I8) if self.features.sse41 => Some(&[0x0F, 0x38, 0x38]),
            (Binary::Smax, Lane::I8) if self.features.sse41 => Some(&[0x0F, 0x38, 0x3C]),
            (Binary::Umin, Lane::I16) if self.features.sse41 => Some(&[0x0F, 0x38, 0x3A]),
            (Binary::Umax, Lane::I16) if self.features.sse41 => Some(&[0x0F, 0x38, 0x3E]),
            (Binary::Smin, Lane::I32) if self.features.sse41 => Some(&[0x0F, 0x38, 0x39]),
            (Binary::Smax, Lane::I32) if self.features.sse41 => Some(&[0x0F, 0x38, 0x3D]),
            (Binary::Umin, Lane::I32) if self.features.sse41 => Some(&[0x0F, 0x38, 0x3B]),
            (Binary::Umax, Lane::I32) if self.features.sse41 => Some(&[0x0F, 0x38, 0x3F]),
            _ => None,
        }
    }

    fn vector_iminmax(
        &mut self,
        op: Binary,
        lane: Lane,
        count: u8,
        x: Value,
        y: Value,
        result: Value,
    ) -> Result<(), String> {
        if let Some(bytes) = self.iminmax_opc(op, lane) {
            self.vex_packed_op(&[0x66], bytes, (x, y), result, true);
            return Ok(());
        }
        let cc = match op {
            Binary::Smin => IntCC::SignedLessThan,
            Binary::Smax => IntCC::SignedGreaterThan,
            Binary::Umin => IntCC::UnsignedLessThan,
            _ => IntCC::UnsignedGreaterThan,
        };
        let dst = self.xmm_of(result).unwrap_or(X2);
        self.fget(X0, x);
        self.fget(X1, y);
        if lane == Lane::I64 && !self.features.sse41 {
            self.scalarize_int_binary(op, lane, count, x, y, dst);
        } else {
            self.emit_icmp_regs(cc, lane, X0, X1, X2);
            self.fget(X0, x);
            self.fget(X1, y);
            self.blend_regs(X2, X0, X1, dst);
        }
        self.fput(result, dst);
        Ok(())
    }

    pub(in crate::x64) fn splat_source(&self, v: Value, count: u8) -> Option<Value> {
        let inst = self.defs.get(v.index()).copied().flatten()?;
        let Op::Shuffle(a, _, ShuffleMask::Const(m)) = &self.func.insts[inst.index()].op else {
            return None;
        };
        let c = usize::from(count);
        if !m[..c]
            .iter()
            .all(|&idx| idx == m[0] && usize::from(idx) < c)
        {
            return None;
        }
        let a_inst = self.defs.get(a.index()).copied().flatten()?;
        match self.func.insts[a_inst.index()].op {
            Op::Insert(_, scalar, idx) if idx == m[0] => Some(scalar),
            _ => None,
        }
    }

    fn vector_shift(
        &mut self,
        op: Binary,
        lane: Lane,
        count: u8,
        x: Value,
        y: Value,
        result: Value,
    ) -> Result<(), String> {
        let dst = self.xmm_of(result).unwrap_or(X2);
        if self.const_of(y) == Some(0) {
            let src = self.fin(x, dst);
            if dst != src {
                self.asm.movaps(dst, src);
            }
            self.fput(result, dst);
            return Ok(());
        }
        if matches!(lane, Lane::I16 | Lane::I32) || (lane == Lane::I64 && op != Binary::Sshr) {
            if let Some(scalar_v) = self.splat_source(y, count) {
                if let Some(imm) = self.const_of(scalar_v) {
                    let k = ((imm as u32) & (lane.bits() - 1)) as u8;
                    let ext = match op {
                        Binary::Ushr => 2,
                        Binary::Sshr => 4,
                        _ => 6,
                    };
                    let sx = self.fin(x, if self.features.avx { X0 } else { dst });
                    self.pshift_into(ext, lane.bytes(), dst, sx, k);
                    self.fput(result, dst);
                    return Ok(());
                }
                let sx = if self.features.avx {
                    self.fin(x, X0)
                } else {
                    let src = self.fin(x, dst);
                    if dst != src {
                        self.asm.movaps(dst, src);
                    }
                    dst
                };
                self.get(Reg::Rax, scalar_v, Ext::Zero);
                self.asm
                    .alu_imm32(Alu::And, Reg::Rax, (lane.bits() - 1) as i32);
                self.asm.movd_in(X1, Reg::Rax);
                self.sse_or_vex(
                    &[0x66],
                    &[0x0F, shift_count_opcode(op, lane)],
                    (dst, sx, X1),
                );
                self.fput(result, dst);
                return Ok(());
            }
        }
        self.scalarize_int_binary(op, lane, count, x, y, dst);
        self.fput(result, dst);
        Ok(())
    }
}

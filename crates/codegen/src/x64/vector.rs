mod binary;
mod cmp;
mod convert;

use super::Emit;
use super::asm::{Mem, Reg, Rm, Size, Xmm};
use super::float::{X0, X1, X2};
use crate::alloc::Location;
use crate::ir::{Binary, Lane, Type, Unary, Value};

pub(super) fn vec_alu_opcode(op: Binary, lane: Lane) -> Option<u8> {
    match (op, lane) {
        (Binary::Iadd, Lane::I8) => Some(0xFC),
        (Binary::Iadd, Lane::I16) => Some(0xFD),
        (Binary::Iadd, Lane::I32) => Some(0xFE),
        (Binary::Iadd, Lane::I64) => Some(0xD4),
        (Binary::Isub, Lane::I8) => Some(0xF8),
        (Binary::Isub, Lane::I16) => Some(0xF9),
        (Binary::Isub, Lane::I32) => Some(0xFA),
        (Binary::Isub, Lane::I64) => Some(0xFB),
        (Binary::Umulhi, Lane::I16) => Some(0xE4),
        (Binary::Smulhi, Lane::I16) => Some(0xE5),
        (Binary::Band, _) => Some(0xDB),
        (Binary::Bor, _) => Some(0xEB),
        (Binary::Bxor, _) => Some(0xEF),
        _ => None,
    }
}

/// The `0F` opcode that shifts every lane of a `lane` vector by the count in the low quadword of its second operand.
pub(super) fn shift_count_opcode(op: Binary, lane: Lane) -> u8 {
    match (op, lane) {
        (Binary::Ishl, Lane::I16) => 0xF1,
        (Binary::Ishl, Lane::I32) => 0xF2,
        (Binary::Ishl, Lane::I64) => 0xF3,
        (Binary::Ushr, Lane::I16) => 0xD1,
        (Binary::Ushr, Lane::I32) => 0xD2,
        (Binary::Ushr, Lane::I64) => 0xD3,
        (Binary::Sshr, Lane::I16) => 0xE1,
        _ => 0xE2,
    }
}

pub(super) fn float_vec_opcode(op: Binary) -> Option<u8> {
    match op {
        Binary::Fadd => Some(0x58),
        Binary::Fmul => Some(0x59),
        Binary::Fsub => Some(0x5C),
        Binary::Fdiv => Some(0x5E),
        _ => None,
    }
}

pub(super) fn vec_prefix(lane: Lane) -> &'static [u8] {
    if lane == Lane::F64 { &[0x66] } else { &[] }
}

pub(super) fn is_commutative(op: Binary) -> bool {
    matches!(
        op,
        Binary::Iadd
            | Binary::Imul
            | Binary::Umulhi
            | Binary::Smulhi
            | Binary::Band
            | Binary::Bor
            | Binary::Bxor
            | Binary::Fadd
            | Binary::Fmul
            | Binary::Smin
            | Binary::Smax
            | Binary::Umin
            | Binary::Umax
            | Binary::SaddSat
            | Binary::UaddSat
            | Binary::AvgRound
    )
}

impl Emit<'_> {
    pub(super) fn vector_zero(&mut self, result: Value) {
        if !self.is_static(result) {
            let dst = self.fdst(result);
            self.zero_xmm(dst);
            self.fput(result, dst);
        }
    }

    pub(super) fn is_zero_vec(&self, v: Value) -> bool {
        self.func
            .is_zero_vec(&|value| self.defs.get(value.index()).copied().flatten(), v)
    }

    pub(super) fn packed_regs(
        &mut self,
        prefix: &[u8],
        opcode: &[u8],
        (x, y): (Value, Value),
        dst: Xmm,
    ) {
        let [_, source] = self.fins(dst, true, [x, y], &[]);
        self.asm.sse(prefix, opcode, dst.0, source);
    }

    pub(super) fn sse_or_vex(
        &mut self,
        prefix: &[u8],
        opcode: &[u8],
        (dst, sx, sy): (Xmm, Xmm, impl Into<Rm>),
    ) {
        let sy = sy.into();
        let pp = match prefix {
            [] => Some(0),
            [0x66] => Some(1),
            [0xF3] => Some(2),
            [0xF2] => Some(3),
            _ => None,
        };
        let map_opc = match opcode {
            [0x0F, o] => Some((1, *o)),
            [0x0F, 0x38, o] => Some((2, *o)),
            _ => None,
        };
        if self.features.avx
            && let (Some(pp), Some((map, opc))) = (pp, map_opc)
        {
            self.asm.vex_xmm(map, pp, opc, (dst, sx), sy);
            return;
        }
        if dst != sx {
            self.asm.movaps(dst, sx);
        }
        self.asm.sse(prefix, opcode, dst.0, sy);
    }

    pub(super) fn vex_packed_op(
        &mut self,
        prefix: &[u8],
        opcode: &[u8],
        (x, y): (Value, Value),
        result: Value,
        commute: bool,
    ) {
        let dst = self.fdst(result);
        self.vex_packed_regs(prefix, opcode, (x, y), dst, commute);
        self.fput(result, dst);
    }

    pub(super) fn vex_packed_regs(
        &mut self,
        prefix: &[u8],
        opcode: &[u8],
        (x, y): (Value, Value),
        dst: Xmm,
        commute: bool,
    ) {
        if self.features.avx {
            let (x, y) = if commute
                && self.xmm_of(x).is_none()
                && (self.xmm_of(y).is_some()
                    || (self.ty(x).bytes() == 16
                        && self.elided_mem(x).is_some()
                        && self.elided_mem(y).is_none()))
            {
                (y, x)
            } else {
                (x, y)
            };
            let sx = self.fin(x, X0);
            let cached_y = self
                .spills
                .cached_fspill(self.frame.values[y.index()], self.ty(y))
                .is_some();
            let sy: Rm = if let Some(mem) = (self.ty(y).bytes() == 16)
                .then(|| self.elided_mem(y))
                .flatten()
            {
                mem.into()
            } else if self.ty(y).bytes() == 16
                && matches!(self.location(y), Location::Slot)
                && !cached_y
            {
                self.at(y).into()
            } else {
                self.fin(y, X1).into()
            };
            self.sse_or_vex(prefix, opcode, (dst, sx, sy));
            return;
        }
        let pair = if commute && self.xmm_of(x) != Some(dst) && self.xmm_of(y) == Some(dst) {
            (y, x)
        } else {
            (x, y)
        };
        self.packed_regs(prefix, opcode, pair, dst);
    }

    pub(super) fn splat_scalar(&mut self, lane: Lane, dst: Xmm, reg: Reg) {
        match lane {
            Lane::I8 => {
                self.asm.load(Size::B, Reg::Rax, reg);
                self.asm.imul_imm(Reg::Rax, Reg::Rax, 0x0101_0101);
                self.asm.movd_in(dst, Reg::Rax);
                self.asm.sse_imm(&[0x66], &[0x0F, 0x70], dst.0, dst, 0);
            }
            Lane::I16 => {
                self.asm.movd_in(dst, reg);
                self.asm.sse_imm(&[0xF2], &[0x0F, 0x70], dst.0, dst, 0);
                self.asm.sse_imm(&[0x66], &[0x0F, 0x70], dst.0, dst, 0);
            }
            Lane::I32 | Lane::F32 => {
                self.asm.movd_in(dst, reg);
                self.asm.sse_imm(&[0x66], &[0x0F, 0x70], dst.0, dst, 0);
            }
            Lane::I64 | Lane::F64 => {
                self.asm.movq_in(dst, reg);
                self.asm.sse(&[0x66], &[0x0F, 0x6C], dst.0, dst);
            }
        }
    }

    pub(super) fn splat_imm(&mut self, lane: Lane, dst: Xmm, imm: i64) {
        if imm == 0 {
            self.zero_xmm(dst);
            return;
        }
        if imm == -1
            || (lane.bytes() < 8
                && (imm as u64 & ((1_u64 << lane.bits()) - 1)) == ((1_u64 << lane.bits()) - 1))
        {
            self.sse_or_vex(&[0x66], &[0x0F, 0x76], (dst, dst, dst));
            return;
        }
        let (packed, emit_lane) = match lane {
            Lane::I8 => (i64::from((imm as u8 as u32) * 0x0101_0101), Lane::I32),
            Lane::I16 => (i64::from((imm as u16 as u32) * 0x0001_0001), Lane::I32),
            Lane::I32 | Lane::F32 => (i64::from(imm as u32), Lane::I32),
            Lane::I64 | Lane::F64 => (imm, Lane::I64),
        };
        self.asm.mov_imm(Reg::Rax, packed);
        self.splat_scalar(emit_lane, dst, Reg::Rax);
    }

    pub(super) fn vector_unary(
        &mut self,
        op: Unary,
        x: Value,
        result: Value,
    ) -> Result<(), String> {
        let Some((lane, count)) = self.ty(x).vec_shape() else {
            return Err(format!("non-vector type {:?} in vector_unary", self.ty(x)));
        };
        if self.ty(x).is_wide_vec() {
            return self.ymm_unary(op, x, result);
        }
        let dst = self.xmm_of(result).unwrap_or(X2);
        match op {
            Unary::Bnot => {
                let sx = if self.features.avx {
                    self.fin(x, X0)
                } else {
                    self.fget(dst, x);
                    dst
                };
                self.sse_or_vex(&[0x66], &[0x0F, 0x76], (X1, X1, X1));
                self.sse_or_vex(&[0x66], &[0x0F, 0xEF], (dst, sx, X1));
            }
            Unary::Ineg if lane.is_int() => {
                let sx = if self.xmm_of(x) != Some(dst) {
                    self.fin(x, X0)
                } else {
                    self.fget(X0, x);
                    X0
                };
                self.zero_xmm(dst);
                let sub = vec_alu_opcode(Binary::Isub, lane)
                    .ok_or_else(|| format!("{op:?} of {lane:?}"))?;
                self.sse_or_vex(&[0x66], &[0x0F, sub], (dst, dst, sx));
            }
            Unary::Fneg | Unary::Fabs if lane.is_float() => {
                self.fget(dst, x);
                self.sse_or_vex(&[0x66], &[0x0F, 0x76], (X1, X1, X1));
                let (ext, shift) = if op == Unary::Fneg {
                    (6, lane.bits() as u8 - 1)
                } else {
                    (2, 1)
                };
                self.asm.pshift_imm(ext, lane.bytes(), X1, shift);
                self.asm.sse(
                    &[0x66],
                    &[0x0F, if op == Unary::Fneg { 0xEF } else { 0xDB }],
                    dst.0,
                    X1,
                );
            }
            Unary::Sqrt if lane.is_float() => {
                let src = self.fin(x, dst);
                self.asm.sse(vec_prefix(lane), &[0x0F, 0x51], dst.0, src);
            }
            Unary::Floor | Unary::Ceil | Unary::Trunc if lane.is_float() => {
                let mode = match op {
                    Unary::Floor => 9,
                    Unary::Ceil => 10,
                    _ => 11,
                };
                if self.features.sse41 {
                    let src = self.fin(x, dst);
                    let opc = if lane == Lane::F64 { 0x09 } else { 0x08 };
                    self.asm
                        .sse_imm(&[0x66], &[0x0F, 0x3A, opc], dst.0, src, mode);
                } else {
                    self.scalarize_float_unary(op, lane, count, x, dst);
                }
            }
            Unary::Popcnt if lane.is_int() => self.vector_popcnt(lane, x, dst),
            Unary::Clz if lane.is_int() => self.scalarize_int_clz(lane, count, x, dst),
            _ => return Err(format!("{op:?} of a vector")),
        }
        self.fput(result, dst);
        Ok(())
    }

    fn vector_popcnt(&mut self, lane: Lane, x: Value, dst: Xmm) {
        self.fget(dst, x);
        self.asm.movaps(X0, dst);
        self.asm.pshift_imm(2, 2, X0, 1);
        self.splat_imm(Lane::I8, X1, 0x55);
        self.asm.sse(&[0x66], &[0x0F, 0xDB], X0.0, X1);
        self.asm.sse(&[0x66], &[0x0F, 0xF8], dst.0, X0);
        self.asm.movaps(X0, dst);
        self.asm.pshift_imm(2, 2, X0, 2);
        self.splat_imm(Lane::I8, X1, 0x33);
        self.asm.sse(&[0x66], &[0x0F, 0xDB], dst.0, X1);
        self.asm.sse(&[0x66], &[0x0F, 0xDB], X0.0, X1);
        self.asm.sse(&[0x66], &[0x0F, 0xFC], dst.0, X0);
        self.asm.movaps(X0, dst);
        self.asm.pshift_imm(2, 2, X0, 4);
        self.asm.sse(&[0x66], &[0x0F, 0xFC], dst.0, X0);
        self.splat_imm(Lane::I8, X1, 0x0F);
        self.asm.sse(&[0x66], &[0x0F, 0xDB], dst.0, X1);
        match lane {
            Lane::I8 => {}
            Lane::I16 => {
                self.asm.movaps(X0, dst);
                self.asm.pshift_imm(2, 2, X0, 8);
                self.asm.sse(&[0x66], &[0x0F, 0xFD], dst.0, X0);
                self.asm.pshift_imm(6, 2, dst, 8);
                self.asm.pshift_imm(2, 2, dst, 8);
            }
            Lane::I32 => {
                self.asm.movaps(X0, dst);
                self.asm.pshift_imm(2, 2, X0, 8);
                self.asm.sse(&[0x66], &[0x0F, 0xFD], dst.0, X0);
                self.asm.movaps(X0, dst);
                self.asm.pshift_imm(2, 4, X0, 16);
                self.asm.sse(&[0x66], &[0x0F, 0xFE], dst.0, X0);
                self.asm.pshift_imm(6, 4, dst, 24);
                self.asm.pshift_imm(2, 4, dst, 24);
            }
            Lane::I64 => {
                self.zero_xmm(X0);
                self.asm.sse(&[0x66], &[0x0F, 0xF6], dst.0, X0);
            }
            _ => {}
        }
    }

    fn scalarize_float_unary(&mut self, op: Unary, lane: Lane, count: u8, x: Value, dst: Xmm) {
        let fl_ty = lane.to_type();
        let stride = lane.bytes() as i32;
        self.fget(X0, x);
        self.reserve_scratch(16);
        self.xmm_store(Type::Vec { lane, count }, Mem::new(Reg::Rsp, 0), X0);
        for i in 0..count {
            let slot = Mem::new(Reg::Rsp, i32::from(i) * stride);
            self.xmm_load(fl_ty, X0, slot);
            self.round_baseline(op, fl_ty, X0);
            self.xmm_store(fl_ty, slot, X0);
        }
        self.xmm_load(Type::Vec { lane, count }, dst, Mem::new(Reg::Rsp, 0));
        self.release_scratch(16);
    }
}

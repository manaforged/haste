use super::asm::{Alu, Cond, Mem, Reg, Shift, Size, Xmm};
use super::Emit;
use super::float::{X0, X1, X2, scalar};
use crate::frame::Ext;
use crate::ir::{Binary, Convert, FloatCC, IntCC, Lane, Op, ShuffleMask, Type, Unary, Value};

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
        (Binary::Band, _) => Some(0xDB),
        (Binary::Bor, _) => Some(0xEB),
        (Binary::Bxor, _) => Some(0xEF),
        _ => None,
    }
}

fn float_vec_opcode(op: Binary) -> Option<u8> {
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

impl Emit<'_> {
    pub(super) fn vector_zero(&mut self, result: Value) {
        if !self.is_static(result) {
            let dst = self.fdst(result);
            self.zero_xmm(dst);
            self.fput(result, dst);
        }
    }

    pub(super) fn packed_op(&mut self, prefix: &[u8], opcode: &[u8], (x, y): (Value, Value), result: Value) {
        let dst = self.fdst(result);
        self.packed_regs(prefix, opcode, (x, y), dst);
        self.fput(result, dst);
    }

    pub(super) fn packed_regs(&mut self, prefix: &[u8], opcode: &[u8], (x, y): (Value, Value), dst: Xmm) {
        let mut source = self.fin(y, X1);
        if source == dst && x != y {
            self.asm.movaps(X1, source);
            source = X1;
        }
        self.fget(dst, x);
        self.asm.sse(prefix, opcode, dst.0, source);
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
        self.asm.mov_imm(Reg::Rax, imm);
        self.splat_scalar(lane, dst, Reg::Rax);
    }

    pub(super) fn vector_unary(&mut self, op: Unary, x: Value, result: Value) -> Result<(), String> {
        let Some((lane, count)) = self.ty(x).vec_shape() else {
            return Err(format!("non-vector type {:?} in vector_unary", self.ty(x)));
        };
        let dst = self.xmm_of(result).unwrap_or(X2);
        match op {
            Unary::Bnot => {
                self.fget(dst, x);
                self.asm.sse(&[0x66], &[0x0F, 0x76], X1.0, X1);
                self.asm.sse(&[0x66], &[0x0F, 0xEF], dst.0, X1);
            }
            Unary::Ineg if lane.is_int() => {
                self.fget(X0, x);
                self.zero_xmm(dst);
                let sub = vec_alu_opcode(Binary::Isub, lane).ok_or_else(|| format!("{op:?} of {lane:?}"))?;
                self.asm.sse(&[0x66], &[0x0F, sub], dst.0, X0);
            }
            Unary::Fneg | Unary::Fabs if lane.is_float() => {
                self.fget(dst, x);
                let sign = if lane == Lane::F64 { i64::MIN } else { 0x8000_0000_u32 as i64 };
                self.splat_imm(lane, X1, if op == Unary::Fneg { sign } else { !sign });
                let bitwise = if op == Unary::Fneg { 0xEF } else { 0xDB };
                self.asm.sse(&[0x66], &[0x0F, bitwise], dst.0, X1);
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
                    self.asm.sse_imm(&[0x66], &[0x0F, 0x3A, opc], dst.0, src, mode);
                } else {
                    self.scalarize_float_unary(op, lane, count, x, dst);
                }
            }
            _ => return Err(format!("{op:?} of a vector")),
        }
        self.fput(result, dst);
        Ok(())
    }

    fn scalarize_float_unary(&mut self, op: Unary, lane: Lane, count: u8, x: Value, dst: Xmm) {
        let fl_ty = lane.to_type();
        let stride = lane.bytes() as i32;
        self.fget(X0, x);
        self.asm.alu_imm(Alu::Sub, Reg::Rsp, 16);
        self.xmm_store(Type::Vec { lane, count }, Mem::new(Reg::Rsp, 0), X0);
        for i in 0..count {
            let slot = Mem::new(Reg::Rsp, i32::from(i) * stride);
            self.xmm_load(fl_ty, X0, slot);
            self.round_baseline(op, fl_ty, X0);
            self.xmm_store(fl_ty, slot, X0);
        }
        self.xmm_load(Type::Vec { lane, count }, dst, Mem::new(Reg::Rsp, 0));
        self.asm.alu_imm(Alu::Add, Reg::Rsp, 16);
    }

    pub(super) fn vector_binary(&mut self, op: Binary, (x, y): (Value, Value), result: Value) -> Result<(), String> {
        let Some((lane, count)) = self.ty(x).vec_shape() else {
            return Err(format!("non-vector type {:?} in vector_binary", self.ty(x)));
        };
        if let Some(opcode) = vec_alu_opcode(op, lane) {
            self.packed_op(&[0x66], &[0x0F, opcode], (x, y), result);
            return Ok(());
        }
        if let Some(opcode) = float_vec_opcode(op) {
            if !lane.is_float() {
                return Err(format!("{op:?} of integer vector {lane:?}"));
            }
            self.packed_op(vec_prefix(lane), &[0x0F, opcode], (x, y), result);
            return Ok(());
        }
        match op {
            Binary::Fmin | Binary::Fmax if lane.is_float() => self.vector_fminmax(op, lane, x, y, result),
            Binary::Fcopysign if lane.is_float() => {
                let dst = self.xmm_of(result).unwrap_or(X2);
                self.fget(dst, x);
                self.fget(X0, y);
                let sign = if lane == Lane::F64 { i64::MIN } else { 0x8000_0000_u32 as i64 };
                self.splat_imm(lane, X1, sign);
                self.asm.sse(&[0x66], &[0x0F, 0xDB], X0.0, X1);
                self.asm.sse(&[0x66], &[0x0F, 0xDF], X1.0, dst);
                self.asm.sse(&[0x66], &[0x0F, 0xEB], X1.0, X0);
                self.asm.movaps(dst, X1);
                self.fput(result, dst);
                Ok(())
            }
            Binary::Imul if lane.is_int() => self.vector_imul(lane, count, x, y, result),
            Binary::Smin | Binary::Smax | Binary::Umin | Binary::Umax if lane.is_int() => self.vector_iminmax(op, lane, count, x, y, result),
            Binary::Ishl | Binary::Ushr | Binary::Sshr if lane.is_int() => self.vector_shift(op, lane, count, x, y, result),
            _ => Err(format!("{op:?} of vector {lane:?}")),
        }
    }

    fn vector_fminmax(&mut self, op: Binary, lane: Lane, x: Value, y: Value, result: Value) -> Result<(), String> {
        let dst = self.xmm_of(result).unwrap_or(X2);
        self.fget(X0, x);
        self.fget(X1, y);
        self.fminmax_regs(op, lane, dst);
        self.fput(result, dst);
        Ok(())
    }

    pub(super) fn fminmax_regs(&mut self, op: Binary, lane: Lane, dst: Xmm) {
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

    fn vector_imul(&mut self, lane: Lane, count: u8, x: Value, y: Value, result: Value) -> Result<(), String> {
        let dst = self.xmm_of(result).unwrap_or(X2);
        match lane {
            Lane::I16 => self.packed_regs(&[0x66], &[0x0F, 0xD5], (x, y), dst),
            Lane::I32 if self.features.sse41 => self.packed_regs(&[0x66], &[0x0F, 0x38, 0x40], (x, y), dst),
            Lane::I32 => {
                self.fget(X0, x);
                self.fget(X1, y);
                self.asm.movaps(dst, X0);
                self.asm.sse(&[0x66], &[0x0F, 0xF4], dst.0, X1);
                self.asm.pshift_imm(2, 8, X0, 32);
                self.asm.pshift_imm(2, 8, X1, 32);
                self.asm.sse(&[0x66], &[0x0F, 0xF4], X0.0, X1);
                self.asm.sse_imm(&[0x66], &[0x0F, 0x70], dst.0, dst, 0x08);
                self.asm.sse_imm(&[0x66], &[0x0F, 0x70], X0.0, X0, 0x08);
                self.asm.sse(&[0x66], &[0x0F, 0x62], dst.0, X0);
            }
            Lane::I64 => {
                self.fget(X0, x);
                self.fget(X1, y);
                self.asm.movq_out(Reg::Rax, X0);
                self.asm.movq_out(Reg::Rcx, X1);
                self.asm.imul(Reg::Rax, Reg::Rcx);
                self.asm.movq_in(dst, Reg::Rax);
                if count == 2 {
                    self.asm.sse_imm(&[0x66], &[0x0F, 0x70], X0.0, X0, 0xEE);
                    self.asm.sse_imm(&[0x66], &[0x0F, 0x70], X1.0, X1, 0xEE);
                    self.asm.movq_out(Reg::Rax, X0);
                    self.asm.movq_out(Reg::Rcx, X1);
                    self.asm.imul(Reg::Rax, Reg::Rcx);
                    self.asm.movq_in(X0, Reg::Rax);
                    self.asm.sse(&[0x66], &[0x0F, 0x6C], dst.0, X0);
                }
            }
            Lane::I8 => self.scalarize_int_binary(Binary::Imul, lane, count, x, y, dst),
            _ => return Err(format!("Imul of {lane:?}")),
        }
        self.fput(result, dst);
        Ok(())
    }

    pub(super) fn iminmax_opc(&self, op: Binary, lane: Lane) -> Option<&'static [u8]> {
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

    fn vector_iminmax(&mut self, op: Binary, lane: Lane, count: u8, x: Value, y: Value, result: Value) -> Result<(), String> {
        if let Some(bytes) = self.iminmax_opc(op, lane) {
            self.packed_op(&[0x66], bytes, (x, y), result);
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
            self.emit_icmp_regs(cc, lane, X0, X1, dst);
            self.fget(X0, x);
            self.fget(X1, y);
            self.blend_regs(dst, X0, X1, dst);
        }
        self.fput(result, dst);
        Ok(())
    }

    fn splat_source(&self, v: Value, count: u8) -> Option<Value> {
        let inst = self.defs.get(v.index()).copied().flatten()?;
        let Op::Shuffle(a, _, ShuffleMask::Const(m)) = &self.func.insts[inst.index()].op else {
            return None;
        };
        let c = usize::from(count);
        if !m[..c].iter().all(|&idx| idx == m[0] && usize::from(idx) < c) {
            return None;
        }
        let a_inst = self.defs.get(a.index()).copied().flatten()?;
        match self.func.insts[a_inst.index()].op {
            Op::Insert(_, scalar, idx) if idx == m[0] => Some(scalar),
            _ => None,
        }
    }

    fn vector_shift(&mut self, op: Binary, lane: Lane, count: u8, x: Value, y: Value, result: Value) -> Result<(), String> {
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
                    let src = self.fin(x, dst);
                    if dst != src {
                        self.asm.movaps(dst, src);
                    }
                    self.asm.pshift_imm(ext, lane.bytes(), dst, k);
                    self.fput(result, dst);
                    return Ok(());
                }
                let src = self.fin(x, dst);
                if dst != src {
                    self.asm.movaps(dst, src);
                }
                self.get(Reg::Rax, scalar_v, Ext::Zero);
                self.asm.alu_imm32(Alu::And, Reg::Rax, (lane.bits() - 1) as i32);
                self.asm.movd_in(X1, Reg::Rax);
                let opc = match (op, lane) {
                    (Binary::Ishl, Lane::I16) => 0xF1,
                    (Binary::Ishl, Lane::I32) => 0xF2,
                    (Binary::Ishl, Lane::I64) => 0xF3,
                    (Binary::Ushr, Lane::I16) => 0xD1,
                    (Binary::Ushr, Lane::I32) => 0xD2,
                    (Binary::Ushr, Lane::I64) => 0xD3,
                    (Binary::Sshr, Lane::I16) => 0xE1,
                    _ => 0xE2,
                };
                self.asm.sse(&[0x66], &[0x0F, opc], dst.0, X1);
                self.fput(result, dst);
                return Ok(());
            }
        }
        self.scalarize_int_binary(op, lane, count, x, y, dst);
        self.fput(result, dst);
        Ok(())
    }

    pub(super) fn vector_icmp(&mut self, cc: IntCC, x: Value, y: Value, result: Value) -> Result<(), String> {
        let Some((lane, count)) = self.ty(x).vec_shape() else {
            return Err(format!("non-vector type {:?} in vector_icmp", self.ty(x)));
        };
        let dst = self.xmm_of(result).unwrap_or(X2);
        if lane == Lane::I64 && !self.features.sse41 {
            self.scalarize_int_icmp(cc, lane, count, x, y, dst);
        } else {
            self.fget(X0, x);
            self.fget(X1, y);
            self.emit_icmp_regs(cc, lane, X0, X1, dst);
        }
        self.fput(result, dst);
        Ok(())
    }

    pub(super) fn emit_icmp_regs(&mut self, cc: IntCC, lane: Lane, a: Xmm, b: Xmm, dst: Xmm) {
        let unsigned = matches!(
            cc,
            IntCC::UnsignedLessThan | IntCC::UnsignedLessThanOrEqual | IntCC::UnsignedGreaterThan | IntCC::UnsignedGreaterThanOrEqual
        );
        if unsigned {
            let sign = match lane {
                Lane::I8 => 0x80_i64,
                Lane::I16 => 0x8000_i64,
                Lane::I32 => 0x8000_0000_u32 as i64,
                _ => i64::MIN,
            };
            self.splat_imm(lane, dst, sign);
            self.asm.sse(&[0x66], &[0x0F, 0xEF], a.0, dst);
            self.asm.sse(&[0x66], &[0x0F, 0xEF], b.0, dst);
        }
        let eq_opc: &[u8] = match lane {
            Lane::I8 => &[0x0F, 0x74],
            Lane::I16 => &[0x0F, 0x75],
            Lane::I32 => &[0x0F, 0x76],
            _ => &[0x0F, 0x38, 0x29],
        };
        let gt_opc: &[u8] = match lane {
            Lane::I8 => &[0x0F, 0x64],
            Lane::I16 => &[0x0F, 0x65],
            Lane::I32 => &[0x0F, 0x66],
            _ => &[0x0F, 0x38, 0x37],
        };
        let (opc, left, right, invert) = match cc {
            IntCC::Equal => (eq_opc, a, b, false),
            IntCC::NotEqual => (eq_opc, a, b, true),
            IntCC::SignedGreaterThan | IntCC::UnsignedGreaterThan => (gt_opc, a, b, false),
            IntCC::SignedLessThan | IntCC::UnsignedLessThan => (gt_opc, b, a, false),
            IntCC::SignedLessThanOrEqual | IntCC::UnsignedLessThanOrEqual => (gt_opc, a, b, true),
            IntCC::SignedGreaterThanOrEqual | IntCC::UnsignedGreaterThanOrEqual => (gt_opc, b, a, true),
        };
        if dst != left {
            self.asm.movaps(dst, left);
        }
        self.asm.sse(&[0x66], opc, dst.0, right);
        if invert {
            self.asm.sse(&[0x66], &[0x0F, 0x76], X1.0, X1);
            self.asm.sse(&[0x66], &[0x0F, 0xEF], dst.0, X1);
        }
    }

    pub(super) fn vector_fcmp(&mut self, cc: FloatCC, x: Value, y: Value, result: Value) {
        let Some((lane, _)) = self.ty(x).vec_shape() else { return };
        let (left, right, imm) = match cc {
            FloatCC::Equal => (x, y, 0),
            FloatCC::LessThan => (x, y, 1),
            FloatCC::LessThanOrEqual => (x, y, 2),
            FloatCC::NotEqual => (x, y, 4),
            FloatCC::GreaterThan => (y, x, 1),
            FloatCC::GreaterThanOrEqual => (y, x, 2),
        };
        let dst = self.fdst(result);
        let mut src = self.fin(right, X1);
        if src == dst && left != right {
            self.asm.movaps(X1, src);
            src = X1;
        }
        self.fget(dst, left);
        self.asm.sse_imm(vec_prefix(lane), &[0x0F, 0xC2], dst.0, src, imm);
        self.fput(result, dst);
    }

    pub(super) fn blend_regs(&mut self, mask: Xmm, a: Xmm, b: Xmm, dst: Xmm) {
        self.asm.movaps(X0, a);
        self.asm.sse(&[0x66], &[0x0F, 0xDB], X0.0, mask);
        self.asm.sse(&[0x66], &[0x0F, 0xDF], mask.0, b);
        self.asm.sse(&[0x66], &[0x0F, 0xEB], mask.0, X0);
        if dst != mask {
            self.asm.movaps(dst, mask);
        }
    }

    pub(super) fn vector_select(&mut self, cond: Value, x: Value, y: Value, result: Value) {
        let dst = self.xmm_of(result).unwrap_or(X2);
        if self.features.sse41 {
            self.fget(X0, cond);
            self.fget(dst, y);
            let src = self.fin(x, X1);
            self.asm.sse(&[0x66], &[0x0F, 0x38, 0x10], dst.0, src);
        } else {
            self.fget(X1, cond);
            self.fget(dst, y);
            self.fget(X0, x);
            self.asm.sse(&[0x66], &[0x0F, 0xDB], X0.0, X1);
            self.asm.sse(&[0x66], &[0x0F, 0xDF], X1.0, dst);
            self.asm.sse(&[0x66], &[0x0F, 0xEB], X1.0, X0);
            self.asm.movaps(dst, X1);
        }
        self.fput(result, dst);
    }

    pub(super) fn vector_convert(&mut self, op: Convert, x: Value, result: Value) -> Result<(), String> {
        let from_ty = self.ty(x);
        let to_ty = self.ty(result);
        if op == Convert::Bitcast {
            return self.float_convert(op, x, result);
        }
        let (Some((fl, fc)), Some((tl, tc))) = (from_ty.vec_shape(), to_ty.vec_shape()) else {
            return Err(format!("{op:?} between {from_ty:?} and {to_ty:?}"));
        };
        if fc != tc {
            return Err(format!("{op:?} lane count mismatch {fc} != {tc}"));
        }
        let dst = self.fdst(result);
        match (op, fl, tl) {
            (Convert::FromSint, Lane::I32, Lane::F32) => {
                let src = self.fin(x, dst);
                self.asm.sse(&[], &[0x0F, 0x5B], dst.0, src);
            }
            (Convert::Fpromote, Lane::F32, Lane::F64) => {
                let src = self.fin(x, dst);
                self.asm.sse(&[], &[0x0F, 0x5A], dst.0, src);
            }
            (Convert::Fdemote, Lane::F64, Lane::F32) => {
                let src = self.fin(x, dst);
                self.asm.sse(&[0x66], &[0x0F, 0x5A], dst.0, src);
            }
            (Convert::Uextend | Convert::Sextend, _, _) if self.features.sse41 && fl.is_int() && tl.is_int() => {
                let code = match (fl, tl) {
                    (Lane::I8, Lane::I16) => 0x00,
                    (Lane::I8, Lane::I32) => 0x01,
                    (Lane::I8, Lane::I64) => 0x02,
                    (Lane::I16, Lane::I32) => 0x03,
                    (Lane::I16, Lane::I64) => 0x04,
                    (Lane::I32, Lane::I64) => 0x05,
                    _ => return Err(format!("{op:?} from {fl:?} to {tl:?}")),
                };
                let base = if op == Convert::Sextend { 0x20 } else { 0x30 };
                let src = self.fin(x, dst);
                self.asm.sse(&[0x66], &[0x0F, 0x38, base + code], dst.0, src);
            }
            _ => self.scalarize_convert(op, (fl, tl, fc), (from_ty, to_ty), x, dst)?,
        }
        self.fput(result, dst);
        Ok(())
    }

    fn scalarize_convert(&mut self, op: Convert, (fl, tl, count): (Lane, Lane, u8), (from_ty, to_ty): (Type, Type), x: Value, dst: Xmm) -> Result<(), String> {
        self.fget(X0, x);
        self.asm.alu_imm(Alu::Sub, Reg::Rsp, 32);
        self.xmm_store(from_ty, Mem::new(Reg::Rsp, 16), X0);
        for i in 0..count {
            let src_slot = Mem::new(Reg::Rsp, 16 + i32::from(i) * fl.bytes() as i32);
            let dst_slot = Mem::new(Reg::Rsp, i32::from(i) * tl.bytes() as i32);
            match op {
                Convert::Uextend | Convert::Ireduce => {
                    self.asm.load(Size::of(fl.bytes()), Reg::Rax, src_slot);
                    self.asm.store(Size::of(tl.bytes()), dst_slot, Reg::Rax);
                }
                Convert::Sextend => {
                    self.asm.load_signed(Size::of(fl.bytes()), Reg::Rax, src_slot);
                    self.asm.store(Size::of(tl.bytes()), dst_slot, Reg::Rax);
                }
                Convert::FromSint => {
                    self.asm.load_signed(Size::of(fl.bytes()), Reg::Rax, src_slot);
                    self.asm.sse_wide(scalar(tl.to_type()), 0x2A, X0.0, Reg::Rax);
                    self.xmm_store(tl.to_type(), dst_slot, X0);
                }
                Convert::FromUint => {
                    self.asm.load(Size::of(fl.bytes()), Reg::Rax, src_slot);
                    self.asm.sse_wide(scalar(tl.to_type()), 0x2A, X0.0, Reg::Rax);
                    if fl == Lane::I64 {
                        let (big, done) = (self.asm.label(), self.asm.label());
                        self.asm.test(Reg::Rax, Reg::Rax);
                        self.asm.jcc(Cond::S, big);
                        self.asm.jmp(done);
                        self.asm.bind(big);
                        self.asm.mov(Reg::Rcx, Reg::Rax);
                        self.asm.shift_imm(Size::Q, Shift::Shr, Reg::Rcx, 1);
                        self.asm.alu_imm32(Alu::And, Reg::Rax, 1);
                        self.asm.alu(Alu::Or, Reg::Rcx, Reg::Rax);
                        self.asm.sse_wide(scalar(tl.to_type()), 0x2A, X0.0, Reg::Rcx);
                        self.asm.sse(&[scalar(tl.to_type())], &[0x0F, 0x58], X0.0, X0);
                        self.asm.bind(done);
                    }
                    self.xmm_store(tl.to_type(), dst_slot, X0);
                }
                Convert::ToSintSat | Convert::ToUintSat => {
                    let fl_t = fl.to_type();
                    if fl_t == Type::F32 {
                        self.xmm_load(Type::F32, X0, src_slot);
                        self.asm.sse(&[0xF3], &[0x0F, 0x5A], X0.0, X0);
                    } else {
                        self.xmm_load(Type::F64, X0, src_slot);
                    }
                    let signed = op == Convert::ToSintSat;
                    if signed { self.to_signed() } else { self.to_unsigned() }
                    self.clamp(tl.to_type(), signed);
                    self.asm.store(Size::of(tl.bytes()), dst_slot, Reg::Rax);
                }
                _ => {
                    self.asm.alu_imm(Alu::Add, Reg::Rsp, 32);
                    return Err(format!("unsupported vector convert {op:?} {fl:?}->{tl:?}"));
                }
            }
        }
        self.xmm_load(to_ty, dst, Mem::new(Reg::Rsp, 0));
        self.asm.alu_imm(Alu::Add, Reg::Rsp, 32);
        Ok(())
    }
}

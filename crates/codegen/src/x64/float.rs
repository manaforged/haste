use super::Emit;
use super::asm::{Alu, Cond, Group3, Reg, Rm, Shift, Size, Xmm};
use super::vector::vec_prefix;
use crate::frame::Ext;
use crate::ir::{Binary, Convert, FloatCC, Lane, Type, Unary, Value};

pub(super) const X0: Xmm = Xmm(0);
pub(super) const X1: Xmm = Xmm(1);
pub(super) const X2: Xmm = Xmm(2);
const TWO_63: i64 = 0x43E0_0000_0000_0000;
const TWO_64: i64 = 0x43F0_0000_0000_0000;
/// The bits of 2^52, the least magnitude at which every f64 is whole.
const F64_WHOLE: i64 = 0x4330_0000_0000_0000;
/// The bits of 2^23, the least magnitude at which every f32 is whole.
const F32_WHOLE: i64 = 0x4B00_0000;
const F64_ONE: i64 = 0x3FF0_0000_0000_0000;
const F32_ONE: i64 = 0x3F80_0000;

/// The float registers planning hands out: past xmm0-xmm2, which conversions use as scratch and calls return in.
pub(super) const POOL: [u8; 13] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

/// The float compares whose one condition a branch can read from the flags `ucomis` sets; equality needs two.
pub(super) const BRANCHES: [FloatCC; 4] = [
    FloatCC::LessThan,
    FloatCC::LessThanOrEqual,
    FloatCC::GreaterThan,
    FloatCC::GreaterThanOrEqual,
];

pub(super) fn scalar(ty: Type) -> u8 {
    if ty == Type::F64 { 0xF2 } else { 0xF3 }
}

impl Emit<'_> {
    fn ucomi(&mut self, ty: Type, a: Xmm, b: impl Into<Rm>) {
        let prefix: &[u8] = if ty == Type::F64 { &[0x66] } else { &[] };
        self.asm.sse(prefix, &[0x0F, 0x2E], a.0, b);
    }

    pub(super) fn zero_xmm(&mut self, xmm: Xmm) {
        self.sse_or_vex(&[0x66], &[0x0F, 0x57], (xmm, xmm, xmm));
        self.clobber_xmm(xmm);
    }

    fn bit_sign(&mut self, ty: Type, toggle: bool, reg: Reg) {
        self.asm.bt_imm(
            if ty == Type::F64 { Size::Q } else { Size::D },
            if toggle { 7 } else { 6 },
            reg,
            ty.bits() as u8 - 1,
        );
    }

    pub(super) fn float_unary(&mut self, op: Unary, x: Value, result: Value) -> Result<(), String> {
        let ty = self.ty(x);
        let mode = match op {
            Unary::Fneg | Unary::Fabs => {
                self.raw(Reg::Rax, x);
                self.bit_sign(ty, op == Unary::Fneg, Reg::Rax);
                self.clobber_reg(Reg::Rax);
                self.put(result, Reg::Rax);
                return Ok(());
            }
            Unary::Sqrt => None,
            Unary::Floor => Some(9),
            Unary::Ceil => Some(10),
            Unary::Trunc => Some(11),
            other => return Err(format!("{other:?} of a float")),
        };
        let dst = self.fdst(result);
        self.fget(dst, x);
        match mode {
            None => self.asm.sse(&[scalar(ty)], &[0x0F, 0x51], dst.0, dst),
            Some(mode) if self.features.sse41 => {
                let opcode = if ty == Type::F64 { 0x0B } else { 0x0A };
                self.asm.sse(&[0x66], &[0x0F, 0x3A, opcode], dst.0, dst);
                self.asm.bytes(&[mode]);
            }
            Some(_) => self.round_baseline(op, ty, dst),
        }
        self.clobber_xmm(dst);
        self.fput(result, dst);
        Ok(())
    }

    /// Rounds the float in `dst` by `op` without SSE4.1: through a 64-bit integer below the magnitude where every float is whole, then the sign of the input, using rax, rcx, rdx, xmm1 and xmm2.
    pub(super) fn round_baseline(&mut self, op: Unary, ty: Type, dst: Xmm) {
        let (size, whole, one) = if ty == Type::F64 {
            (Size::Q, F64_WHOLE, F64_ONE)
        } else {
            (Size::D, F32_WHOLE, F32_ONE)
        };
        let done = self.asm.label();
        self.asm.movq_out(Reg::Rax, dst);
        self.asm.mov(Reg::Rcx, Reg::Rax);
        self.asm.shift_imm(size, Shift::Shl, Reg::Rcx, 1);
        self.asm.mov_imm(Reg::Rdx, whole << 1);
        self.asm.alu_sized(size, Alu::Cmp, Reg::Rcx, Reg::Rdx);
        self.asm.jcc(Cond::Ae, done);
        self.asm.sse_wide(scalar(ty), 0x2C, Reg::Rcx as u8, dst);
        self.asm.sse_wide(scalar(ty), 0x2A, X1.0, Reg::Rcx);
        if op != Unary::Trunc {
            self.step(op == Unary::Floor, ty, dst, one);
        }
        self.asm.movq_out(Reg::Rcx, X1);
        self.asm.mov_imm(Reg::Rdx, ty.sign_bit());
        self.asm.alu_sized(size, Alu::And, Reg::Rax, Reg::Rdx);
        self.asm.alu_sized(size, Alu::Or, Reg::Rcx, Reg::Rax);
        self.asm.movq_in(dst, Reg::Rcx);
        self.asm.bind(done);
        self.clobber_regs(&[Reg::Rax, Reg::Rcx, Reg::Rdx]);
        self.clobber_xmms(&[X1, X2, dst]);
    }

    /// Moves the truncated xmm1 one down for a floor, or up for a ceiling, when truncating `x` moved it the other way; `one` holds the bits of 1.0.
    fn step(&mut self, down: bool, ty: Type, x: Xmm, one: i64) {
        let skip = self.asm.label();
        let (above, below, opcode) = if down { (X1, x, 0x5C) } else { (x, X1, 0x58) };
        self.ucomi(ty, above, below);
        self.asm.jcc(Cond::Be, skip);
        self.asm.mov_imm(Reg::Rdx, one);
        self.asm.movq_in(X2, Reg::Rdx);
        self.asm.sse(&[scalar(ty)], &[0x0F, opcode], X1.0, X2);
        self.asm.bind(skip);
    }

    fn fminmax(&mut self, op: Binary, ty: Type, x: Value, y: Value, result: Value) {
        let opcode = if op == Binary::Fmin { 0x5D } else { 0x5F };
        let dst = self.xmm_of(result).unwrap_or(X2);
        self.fget(X0, x);
        self.fget(X1, y);
        self.asm.movaps(dst, X0);
        self.asm.sse(&[scalar(ty)], &[0x0F, opcode], dst.0, X1);
        let done = self.asm.label();
        self.ucomi(ty, X1, X1);
        self.asm.jcc(Cond::Np, done);
        self.asm.movaps(dst, X0);
        self.asm.bind(done);
        self.clobber_xmms(&[X0, X1, dst]);
        self.fput(result, dst);
    }

    pub(super) fn float_binary(
        &mut self,
        op: Binary,
        x: Value,
        y: Value,
        result: Value,
    ) -> Result<(), String> {
        let ty = self.ty(x);
        let opcode = match op {
            Binary::Fadd => 0x58,
            Binary::Fmul => 0x59,
            Binary::Fsub => 0x5C,
            Binary::Fdiv => 0x5E,
            Binary::Fmin | Binary::Fmax => {
                self.fminmax(op, ty, x, y, result);
                return Ok(());
            }
            Binary::Fcopysign => {
                self.copysign(ty, x, y, result);
                return Ok(());
            }
            other => return Err(format!("{other:?} of a float")),
        };
        let dst = self.fdst(result);
        match self.xmm_of(x).filter(|_| self.features.avx) {
            Some(src) => {
                let source = self.foperand(y, X1);
                self.asm.vex_scalar(scalar(ty), opcode, (dst, src), source);
            }
            None => {
                let source = if self.xmm_of(y) == Some(dst) && x != y {
                    self.asm.movaps(X1, dst);
                    self.clobber_xmm(X1);
                    X1.into()
                } else {
                    self.foperand(y, X1)
                };
                self.fget(dst, x);
                self.asm.sse(&[scalar(ty)], &[0x0F, opcode], dst.0, source);
            }
        }
        self.clobber_xmm(dst);
        self.fput(result, dst);
        Ok(())
    }

    fn fma_reg(&mut self, value: Value, dst: Xmm, scratch: Xmm) -> Xmm {
        match self.xmm_of(value) {
            Some(r) if r != dst => r,
            _ => {
                self.fget(scratch, value);
                scratch
            }
        }
    }

    fn fma_vex(&mut self, ops: [Value; 3], result: Value, is_vec: bool, is_f64: bool) {
        let [a, b, c] = ops;
        let dst = self.fdst(result);
        let src_b = self.fma_reg(b, dst, X1);
        let src_c: Rm = if !is_vec && self.xmm_of(c) != Some(dst) {
            self.foperand(c, X2)
        } else {
            self.fma_reg(c, dst, X2).into()
        };
        self.fget(dst, a);
        self.asm.vex_fma(is_vec, is_f64, (dst, src_b), src_c);
        self.clobber_xmms(&[X1, X2, dst]);
        self.fput(result, dst);
    }

    pub(super) fn fma(
        &mut self,
        a: Value,
        b: Value,
        c: Value,
        result: Value,
    ) -> Result<(), String> {
        let ty = self.ty(a);
        let (is_vec, is_f64, lane) = match ty {
            Type::F32 => (false, false, Lane::F32),
            Type::F64 => (false, true, Lane::F64),
            Type::Vec {
                lane: Lane::F32,
                count: 2 | 4,
            } => (true, false, Lane::F32),
            Type::Vec {
                lane: Lane::F64,
                count: 2,
            } => (true, true, Lane::F64),
            other => return Err(format!("Fma of {other:?}")),
        };
        if self.features.fma && self.features.avx {
            self.fma_vex([a, b, c], result, is_vec, is_f64);
            return Ok(());
        }
        let prefix: &[u8] = if is_vec {
            vec_prefix(lane)
        } else {
            &[scalar(ty)]
        };
        self.fget(X0, a);
        let sse_arg = |this: &mut Self, v| {
            if is_vec {
                this.fin(v, X1).into()
            } else {
                this.foperand(v, X1)
            }
        };
        let mul_src = sse_arg(self, b);
        self.asm.sse(prefix, &[0x0F, 0x59], X0.0, mul_src);
        let add_src = sse_arg(self, c);
        self.asm.sse(prefix, &[0x0F, 0x58], X0.0, add_src);
        self.clobber_xmms(&[X0, X1]);
        self.fput(result, X0);
        Ok(())
    }

    fn copysign(&mut self, ty: Type, x: Value, y: Value, result: Value) {
        self.raw(Reg::Rax, x);
        self.raw(Reg::Rcx, y);
        self.bit_sign(ty, false, Reg::Rax);
        if ty == Type::F64 {
            self.asm.shift_imm(Size::Q, Shift::Shr, Reg::Rcx, 63);
            self.asm.shift_imm(Size::Q, Shift::Shl, Reg::Rcx, 63);
            self.asm.alu(Alu::Or, Reg::Rax, Reg::Rcx);
        } else {
            self.asm.alu_imm32(Alu::And, Reg::Rcx, i32::MIN);
            self.asm.alu32(Alu::Or, Reg::Rax, Reg::Rcx);
        }
        self.clobber_regs(&[Reg::Rax, Reg::Rcx]);
        self.put(result, Reg::Rax);
    }

    /// Compares by `ucomis`, the less-than forms with swapped operands; a compare fused with the next branch leaves its one condition in the flags.
    pub(super) fn fcmp(&mut self, cc: FloatCC, x: Value, y: Value, result: Value) {
        self.flags = None;
        if self.ty(x).is_vector() {
            self.invalidate_spills();
            self.vector_fcmp(cc, x, y, result);
            self.invalidate_spills();
            return;
        }
        let (a, b, cond) = match cc {
            FloatCC::Equal => (x, y, Cond::E),
            FloatCC::NotEqual => (x, y, Cond::Ne),
            FloatCC::LessThan => (y, x, Cond::A),
            FloatCC::LessThanOrEqual => (y, x, Cond::Ae),
            FloatCC::GreaterThan => (x, y, Cond::A),
            FloatCC::GreaterThanOrEqual => (x, y, Cond::Ae),
        };
        let left = self.fin(a, X0);
        let right = self.foperand(b, X1);
        self.ucomi(self.ty(x), left, right);
        if self.plan.fused[result.index()] {
            self.flags = Some((result, cond));
            return;
        }
        self.asm.setcc(cond, Reg::Rax);
        if matches!(cc, FloatCC::Equal | FloatCC::NotEqual) {
            let equal = cc == FloatCC::Equal;
            self.asm
                .setcc(if equal { Cond::Np } else { Cond::P }, Reg::Rcx);
            self.asm
                .alu32(if equal { Alu::And } else { Alu::Or }, Reg::Rax, Reg::Rcx);
        }
        self.asm.load(Size::B, Reg::Rax, Reg::Rax);
        self.clobber_regs(&[Reg::Rax, Reg::Rcx]);
        self.put(result, Reg::Rax);
    }

    pub(super) fn float_convert(
        &mut self,
        op: Convert,
        x: Value,
        result: Value,
    ) -> Result<(), String> {
        let to = self.ty(result);
        match op {
            Convert::Fpromote | Convert::Fdemote => {
                let (dst, source) = (self.fdst(result), self.foperand(x, X1));
                self.asm
                    .sse(&[scalar(self.ty(x))], &[0x0F, 0x5A], dst.0, source);
                self.clobber_xmm(dst);
                self.fput(result, dst);
            }
            Convert::FromSint | Convert::FromUint if self.ty(x) == Type::I128 => {
                return Err(format!("{op:?} from I128 to {to:?}"));
            }
            Convert::FromSint => {
                self.get(Reg::Rax, x, Ext::Sign);
                let dst = self.fdst(result);
                self.asm.sse_wide(scalar(to), 0x2A, dst.0, Reg::Rax);
                self.clobber_xmm(dst);
                self.fput(result, dst);
            }
            Convert::FromUint => {
                let dst = self.fdst(result);
                self.from_unsigned(x, to, dst);
                self.clobber_regs(&[Reg::Rax, Reg::Rcx]);
                self.clobber_xmm(dst);
                self.fput(result, dst);
            }
            Convert::ToSintSat | Convert::ToUintSat if to != Type::I128 => {
                self.widen_to_double(x);
                let signed = op == Convert::ToSintSat;
                if signed {
                    self.to_signed()
                } else {
                    self.to_unsigned()
                }
                self.clamp(to, signed);
                self.clobber_regs(&[Reg::Rax, Reg::Rcx]);
                self.clobber_xmms(&[X0, X1, X2]);
                self.put(result, Reg::Rax);
            }
            Convert::Bitcast => self.bitcast(x, result),
            other => return Err(format!("{other:?} from {:?} to {to:?}", self.ty(x))),
        }
        Ok(())
    }

    fn bitcast(&mut self, x: Value, result: Value) {
        let to = self.ty(result);
        if to.in_float_register() && self.ty(x).in_float_register() {
            let dst = self.fdst(result);
            let src = self.fin(x, dst);
            if dst != src {
                self.fmov(to, dst, src);
            }
            self.fput(result, dst);
        } else if to.in_float_register() {
            let dst = self.fdst(result);
            if self.const_of(x) == Some(0) {
                self.zero_xmm(dst);
            } else {
                self.get(Reg::Rax, x, Ext::Zero);
                let wide = to != Type::F32
                    && self
                        .const_of(x)
                        .is_none_or(|c| (c as u64) > u64::from(u32::MAX));
                self.emit_mov_xmm_gpr(dst, Reg::Rax, wide);
                self.clobber_xmm(dst);
            }
            self.fput(result, dst);
        } else {
            let src = self.fin(x, X0);
            let dst = self.dst(result);
            if self.ty(x) == Type::F32 {
                self.asm.movd_out(dst, src);
            } else {
                self.asm.movq_out(dst, src);
            }
            self.put(result, dst);
        }
    }

    /// Converts unsigned `x` to a float of type `to` in `dst`.
    fn from_unsigned(&mut self, x: Value, to: Type, dst: Xmm) {
        self.get(Reg::Rax, x, Ext::Zero);
        let prefix = scalar(to);
        if self.ty(x).bits() < 64 {
            self.asm.sse_wide(prefix, 0x2A, dst.0, Reg::Rax);
            return;
        }
        let (big, done) = (self.asm.label(), self.asm.label());
        self.asm.test(Reg::Rax, Reg::Rax);
        self.asm.jcc(Cond::S, big);
        self.asm.sse_wide(prefix, 0x2A, dst.0, Reg::Rax);
        self.asm.jmp(done);
        self.asm.bind(big);
        self.asm.mov(Reg::Rcx, Reg::Rax);
        self.asm.shift_imm(Size::Q, Shift::Shr, Reg::Rcx, 1);
        self.asm.alu_imm32(Alu::And, Reg::Rax, 1);
        self.asm.alu(Alu::Or, Reg::Rcx, Reg::Rax);
        self.asm.sse_wide(prefix, 0x2A, dst.0, Reg::Rcx);
        self.asm.sse(&[prefix], &[0x0F, 0x58], dst.0, dst);
        self.asm.bind(done);
    }

    fn widen_to_double(&mut self, x: Value) {
        if self.ty(x) == Type::F32 {
            let source = self.foperand(x, X1);
            self.asm.sse(&[0xF3], &[0x0F, 0x5A], X0.0, source);
        } else {
            self.fget(X0, x);
        }
    }

    /// Converts the double in xmm0 to a saturated i64 in rax, with NaN as zero.
    pub(super) fn to_signed(&mut self) {
        let (nan, done) = (self.asm.label(), self.asm.label());
        self.asm.sse_wide(0xF2, 0x2C, Reg::Rax as u8, X0);
        self.asm.alu_imm(Alu::Cmp, Reg::Rax, 1);
        self.asm.jcc(Cond::No, done);
        self.ucomi(Type::F64, X0, X0);
        self.asm.jcc(Cond::P, nan);
        self.zero_xmm(X1);
        self.ucomi(Type::F64, X0, X1);
        self.asm.jcc(Cond::Be, done);
        self.asm.group3(Group3::Not, Reg::Rax);
        self.asm.jmp(done);
        self.asm.bind(nan);
        self.asm.alu32(Alu::Xor, Reg::Rax, Reg::Rax);
        self.asm.bind(done);
    }

    /// Converts the double in xmm0 to a saturated u64 in rax, with NaN as zero.
    pub(super) fn to_unsigned(&mut self) {
        let (positive, big, max, done) = (
            self.asm.label(),
            self.asm.label(),
            self.asm.label(),
            self.asm.label(),
        );
        self.zero_xmm(X1);
        self.ucomi(Type::F64, X0, X1);
        self.asm.jcc(Cond::A, positive);
        self.asm.alu32(Alu::Xor, Reg::Rax, Reg::Rax);
        self.asm.jmp(done);
        self.asm.bind(positive);
        self.asm.mov_imm(Reg::Rcx, TWO_63);
        self.asm.movq_in(X1, Reg::Rcx);
        self.ucomi(Type::F64, X0, X1);
        self.asm.jcc(Cond::Ae, big);
        self.asm.sse_wide(0xF2, 0x2C, Reg::Rax as u8, X0);
        self.asm.jmp(done);
        self.asm.bind(big);
        self.asm.mov_imm(Reg::Rcx, TWO_64);
        self.asm.movq_in(X2, Reg::Rcx);
        self.ucomi(Type::F64, X0, X2);
        self.asm.jcc(Cond::Ae, max);
        self.asm.sse(&[0xF2], &[0x0F, 0x5C], X0.0, X1);
        self.asm.sse_wide(0xF2, 0x2C, Reg::Rax as u8, X0);
        self.asm.bt_imm(Size::Q, 5, Reg::Rax, 63);
        self.asm.jmp(done);
        self.asm.bind(max);
        self.asm.mov_imm(Reg::Rax, -1);
        self.asm.bind(done);
    }

    pub(super) fn clamp(&mut self, to: Type, signed: bool) {
        let bits = to.bits();
        if bits >= 64 {
            return;
        }
        if signed {
            self.asm.mov_imm(Reg::Rcx, -(1_i64 << (bits - 1)));
            self.asm.alu(Alu::Cmp, Reg::Rax, Reg::Rcx);
            self.asm.cmov(Cond::L, Reg::Rax, Reg::Rcx);
            self.asm.mov_imm(Reg::Rcx, (1_i64 << (bits - 1)) - 1);
            self.asm.alu(Alu::Cmp, Reg::Rax, Reg::Rcx);
            self.asm.cmov(Cond::G, Reg::Rax, Reg::Rcx);
        } else {
            self.asm.mov_imm(Reg::Rcx, (1_i64 << bits) - 1);
            self.asm.alu(Alu::Cmp, Reg::Rax, Reg::Rcx);
            self.asm.cmov(Cond::A, Reg::Rax, Reg::Rcx);
        }
    }
}

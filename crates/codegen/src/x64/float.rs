use super::asm::{Alu, Cond, Group3, Reg, Rm, Shift, Size, Xmm};
use super::{Emit, Ext};
use crate::ir::{Binary, Convert, FloatCC, Type, Unary, Value};

const X0: Xmm = Xmm(0);
const X1: Xmm = Xmm(1);
const X2: Xmm = Xmm(2);
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
pub(super) const BRANCHES: [FloatCC; 4] = [FloatCC::LessThan, FloatCC::LessThanOrEqual, FloatCC::GreaterThan, FloatCC::GreaterThanOrEqual];

pub(super) fn scalar(ty: Type) -> u8 {
    if ty == Type::F64 { 0xF2 } else { 0xF3 }
}

fn sign(ty: Type) -> i64 {
    if ty == Type::F64 { i64::MIN } else { 0x8000_0000 }
}

impl Emit<'_> {
    fn ucomi(&mut self, ty: Type, a: Xmm, b: impl Into<Rm>) {
        let prefix: &[u8] = if ty == Type::F64 { &[0x66] } else { &[] };
        self.asm.sse(prefix, &[0x0F, 0x2E], a.0, b);
    }

    fn zero_xmm(&mut self, xmm: Xmm) {
        self.asm.sse(&[0x66], &[0x0F, 0x57], xmm.0, xmm);
    }

    pub(super) fn float_unary(&mut self, op: Unary, x: Value, result: Value) -> Result<(), String> {
        let ty = self.ty(x);
        let mode = match op {
            Unary::Fneg | Unary::Fabs => {
                self.raw(Reg::Rax, x);
                let neg = op == Unary::Fneg;
                self.asm.mov_imm(Reg::Rcx, if neg { sign(ty) } else { !sign(ty) });
                self.asm.alu(if neg { Alu::Xor } else { Alu::And }, Reg::Rax, Reg::Rcx);
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
        self.fput(result, dst);
        Ok(())
    }

    /// Rounds the float in `dst` by `op` without SSE4.1: through a 64-bit integer below the magnitude where every float is whole, then the sign of the input, using rax, rcx, rdx, xmm1 and xmm2.
    fn round_baseline(&mut self, op: Unary, ty: Type, dst: Xmm) {
        let (size, whole, one) = if ty == Type::F64 { (Size::Q, F64_WHOLE, F64_ONE) } else { (Size::D, F32_WHOLE, F32_ONE) };
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
        self.asm.mov_imm(Reg::Rdx, sign(ty));
        self.asm.alu_sized(size, Alu::And, Reg::Rax, Reg::Rdx);
        self.asm.alu_sized(size, Alu::Or, Reg::Rcx, Reg::Rax);
        self.asm.movq_in(dst, Reg::Rcx);
        self.asm.bind(done);
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

    pub(super) fn float_binary(&mut self, op: Binary, x: Value, y: Value, result: Value) -> Result<(), String> {
        let ty = self.ty(x);
        let opcode = match op {
            Binary::Fadd => 0x58,
            Binary::Fmul => 0x59,
            Binary::Fsub => 0x5C,
            Binary::Fdiv => 0x5E,
            Binary::Fcopysign => {
                self.copysign(ty, x, y, result);
                return Ok(());
            }
            other => return Err(format!("{other:?} of a float")),
        };
        let dst = self.fdst(result);
        match self.xmm_of(x).filter(|_| self.features.avx) {
            Some(src) => {
                let source = self.foperand(y);
                self.asm.vex_scalar(scalar(ty), opcode, (dst, src), source);
            }
            None => {
                self.fget(dst, x);
                let source = self.foperand(y);
                self.asm.sse(&[scalar(ty)], &[0x0F, opcode], dst.0, source);
            }
        }
        self.fput(result, dst);
        Ok(())
    }

    fn copysign(&mut self, ty: Type, x: Value, y: Value, result: Value) {
        self.raw(Reg::Rax, x);
        self.raw(Reg::Rcx, y);
        self.asm.mov_imm(Reg::Rdx, sign(ty));
        self.asm.alu(Alu::And, Reg::Rcx, Reg::Rdx);
        self.asm.group3(Group3::Not, Reg::Rdx);
        self.asm.alu(Alu::And, Reg::Rax, Reg::Rdx);
        self.asm.alu(Alu::Or, Reg::Rax, Reg::Rcx);
        self.put(result, Reg::Rax);
    }

    /// Compares by `ucomis`, the less-than forms with swapped operands; a compare fused with the next branch leaves its one condition in the flags.
    pub(super) fn fcmp(&mut self, cc: FloatCC, x: Value, y: Value, result: Value) {
        let (a, b, cond) = match cc {
            FloatCC::Equal => (x, y, Cond::E),
            FloatCC::NotEqual => (x, y, Cond::Ne),
            FloatCC::LessThan => (y, x, Cond::A),
            FloatCC::LessThanOrEqual => (y, x, Cond::Ae),
            FloatCC::GreaterThan => (x, y, Cond::A),
            FloatCC::GreaterThanOrEqual => (x, y, Cond::Ae),
        };
        let left = self.fin(a, X0);
        let right = self.foperand(b);
        self.ucomi(self.ty(x), left, right);
        if self.plan.fused[result.index()] {
            self.flags = Some((result, cond));
            return;
        }
        self.asm.setcc(cond, Reg::Rax);
        if matches!(cc, FloatCC::Equal | FloatCC::NotEqual) {
            let equal = cc == FloatCC::Equal;
            self.asm.setcc(if equal { Cond::Np } else { Cond::P }, Reg::Rcx);
            self.asm.alu32(if equal { Alu::And } else { Alu::Or }, Reg::Rax, Reg::Rcx);
        }
        self.asm.load(Size::B, Reg::Rax, Reg::Rax);
        self.put(result, Reg::Rax);
    }

    pub(super) fn float_convert(&mut self, op: Convert, x: Value, result: Value) -> Result<(), String> {
        let to = self.ty(result);
        match op {
            Convert::Fpromote | Convert::Fdemote => {
                let (dst, source) = (self.fdst(result), self.foperand(x));
                self.asm.sse(&[scalar(self.ty(x))], &[0x0F, 0x5A], dst.0, source);
                self.fput(result, dst);
            }
            Convert::FromSint | Convert::FromUint if self.ty(x) == Type::I128 => {
                return Err(format!("{op:?} from I128 to {to:?}"));
            }
            Convert::FromSint => {
                self.get(Reg::Rax, x, Ext::Sign);
                let dst = self.fdst(result);
                self.asm.sse_wide(scalar(to), 0x2A, dst.0, Reg::Rax);
                self.fput(result, dst);
            }
            Convert::FromUint => {
                let dst = self.fdst(result);
                self.from_unsigned(x, to, dst);
                self.fput(result, dst);
            }
            Convert::ToSintSat | Convert::ToUintSat if to != Type::I128 => {
                self.widen_to_double(x);
                let signed = op == Convert::ToSintSat;
                if signed { self.to_signed() } else { self.to_unsigned() }
                self.clamp(to, signed);
                self.put(result, Reg::Rax);
            }
            other => return Err(format!("{other:?} from {:?} to {to:?}", self.ty(x))),
        }
        Ok(())
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
            let source = self.foperand(x);
            self.asm.sse(&[0xF3], &[0x0F, 0x5A], X0.0, source);
        } else {
            self.fget(X0, x);
        }
    }

    /// Converts the double in xmm0 to a saturated i64 in rax, with NaN as zero.
    fn to_signed(&mut self) {
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
    fn to_unsigned(&mut self) {
        let (positive, big, max, done) = (self.asm.label(), self.asm.label(), self.asm.label(), self.asm.label());
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
        self.asm.bts(Reg::Rax, 63);
        self.asm.jmp(done);
        self.asm.bind(max);
        self.asm.mov_imm(Reg::Rax, -1);
        self.asm.bind(done);
    }

    fn clamp(&mut self, to: Type, signed: bool) {
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


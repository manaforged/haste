use super::asm::{Alu, Cond, Group3, Reg, Shift, Size, Xmm};
use super::{Emit, Ext};
use crate::ir::{Binary, Convert, FloatCC, Type, Unary, Value};

const X0: Xmm = Xmm(0);
const X1: Xmm = Xmm(1);
const X2: Xmm = Xmm(2);
const TWO_63: i64 = 0x43E0_0000_0000_0000;
const TWO_64: i64 = 0x43F0_0000_0000_0000;

fn scalar(ty: Type) -> u8 {
    if ty == Type::F64 { 0xF2 } else { 0xF3 }
}

fn sign(ty: Type) -> i64 {
    if ty == Type::F64 { i64::MIN } else { 0x8000_0000 }
}

impl Emit<'_> {
    fn fload(&mut self, xmm: Xmm, value: Value) {
        let prefix = scalar(self.ty(value));
        let at = self.at(value);
        self.asm.sse(&[prefix], &[0x0F, 0x10], xmm.0, at);
    }

    fn fstore(&mut self, value: Value, xmm: Xmm) {
        let prefix = scalar(self.ty(value));
        let at = self.at(value);
        self.asm.sse(&[prefix], &[0x0F, 0x11], xmm.0, at);
    }

    fn movq_in(&mut self, xmm: Xmm, reg: Reg) {
        self.asm.encode(&[0x66], true, &[0x0F, 0x6E], xmm.0, reg.into(), false);
    }

    fn ucomi(&mut self, ty: Type, a: Xmm, b: Xmm) {
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
        self.fload(X0, x);
        match mode {
            None => self.asm.sse(&[scalar(ty)], &[0x0F, 0x51], X0.0, X0),
            Some(mode) => {
                let opcode = if ty == Type::F64 { 0x0B } else { 0x0A };
                self.asm.sse(&[0x66], &[0x0F, 0x3A, opcode], X0.0, X0);
                self.asm.bytes(&[mode]);
            }
        }
        self.fstore(result, X0);
        Ok(())
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
        self.fload(X0, x);
        let at = self.at(y);
        self.asm.sse(&[scalar(ty)], &[0x0F, opcode], X0.0, at);
        self.fstore(result, X0);
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

    pub(super) fn fcmp(&mut self, cc: FloatCC, x: Value, y: Value, result: Value) {
        let ty = self.ty(x);
        self.fload(X0, x);
        self.fload(X1, y);
        match cc {
            FloatCC::Equal | FloatCC::NotEqual => {
                self.ucomi(ty, X0, X1);
                let equal = cc == FloatCC::Equal;
                self.asm.setcc(if equal { Cond::E } else { Cond::Ne }, Reg::Rax);
                self.asm.setcc(if equal { Cond::Np } else { Cond::P }, Reg::Rcx);
                self.asm.alu32(if equal { Alu::And } else { Alu::Or }, Reg::Rax, Reg::Rcx);
            }
            FloatCC::LessThan | FloatCC::LessThanOrEqual => {
                self.ucomi(ty, X1, X0);
                self.asm.setcc(if cc == FloatCC::LessThan { Cond::A } else { Cond::Ae }, Reg::Rax);
            }
            FloatCC::GreaterThan | FloatCC::GreaterThanOrEqual => {
                self.ucomi(ty, X0, X1);
                self.asm.setcc(if cc == FloatCC::GreaterThan { Cond::A } else { Cond::Ae }, Reg::Rax);
            }
        }
        self.asm.load(Size::B, Reg::Rax, Reg::Rax);
        self.put(result, Reg::Rax);
    }

    pub(super) fn float_convert(&mut self, op: Convert, x: Value, result: Value) -> Result<(), String> {
        let to = self.ty(result);
        match op {
            Convert::Fpromote | Convert::Fdemote => {
                let at = self.at(x);
                self.asm.sse(&[scalar(self.ty(x))], &[0x0F, 0x5A], X0.0, at);
                self.fstore(result, X0);
            }
            Convert::FromSint => {
                self.get(Reg::Rax, x, Ext::Sign);
                self.asm.sse_wide(scalar(to), 0x2A, X0.0, Reg::Rax);
                self.fstore(result, X0);
            }
            Convert::FromUint => {
                self.from_unsigned(x, to);
                self.fstore(result, X0);
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

    fn from_unsigned(&mut self, x: Value, to: Type) {
        self.get(Reg::Rax, x, Ext::Zero);
        let prefix = scalar(to);
        if self.ty(x).bits() < 64 {
            self.asm.sse_wide(prefix, 0x2A, X0.0, Reg::Rax);
            return;
        }
        let (big, done) = (self.asm.label(), self.asm.label());
        self.asm.test(Reg::Rax, Reg::Rax);
        self.asm.jcc(Cond::S, big);
        self.asm.sse_wide(prefix, 0x2A, X0.0, Reg::Rax);
        self.asm.jmp(done);
        self.asm.bind(big);
        self.asm.mov(Reg::Rcx, Reg::Rax);
        self.asm.shift_imm(Size::Q, Shift::Shr, Reg::Rcx, 1);
        self.asm.alu_imm32(Alu::And, Reg::Rax, 1);
        self.asm.alu(Alu::Or, Reg::Rcx, Reg::Rax);
        self.asm.sse_wide(prefix, 0x2A, X0.0, Reg::Rcx);
        self.asm.sse(&[prefix], &[0x0F, 0x58], X0.0, X0);
        self.asm.bind(done);
    }

    fn widen_to_double(&mut self, x: Value) {
        if self.ty(x) == Type::F32 {
            let at = self.at(x);
            self.asm.sse(&[0xF3], &[0x0F, 0x5A], X0.0, at);
        } else {
            self.fload(X0, x);
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
        self.movq_in(X1, Reg::Rcx);
        self.ucomi(Type::F64, X0, X1);
        self.asm.jcc(Cond::Ae, big);
        self.asm.sse_wide(0xF2, 0x2C, Reg::Rax as u8, X0);
        self.asm.jmp(done);
        self.asm.bind(big);
        self.asm.mov_imm(Reg::Rcx, TWO_64);
        self.movq_in(X2, Reg::Rcx);
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


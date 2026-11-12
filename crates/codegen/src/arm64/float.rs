use super::asm::{Alu, Cond, Fconv, Fop, Funary, Size, ZR};
use super::{A, C, D, Emit, V0, V1};
use crate::frame::Ext;
use crate::ir::{Binary, Convert, FloatCC, Type, Unary, Value};

pub(super) const POOL: [u8; 20] = [4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31];
pub(super) const BRANCHES: [FloatCC; 6] = [FloatCC::Equal, FloatCC::NotEqual, FloatCC::LessThan, FloatCC::LessThanOrEqual, FloatCC::GreaterThan, FloatCC::GreaterThanOrEqual];

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

    pub(super) fn float_binary(&mut self, op: Binary, x: Value, y: Value, result: Value) -> Result<(), String> {
        let ty = self.ty(x);
        let fop = match op {
            Binary::Fadd => Fop::Add,
            Binary::Fsub => Fop::Sub,
            Binary::Fmul => Fop::Mul,
            Binary::Fdiv => Fop::Div,
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

    fn copysign(&mut self, ty: Type, x: Value, y: Value, result: Value) {
        self.raw(A, x);
        self.raw(C, y);
        self.asm.mov_imm(D, ty.sign_bit());
        self.asm.alu(Alu::And, Size::X, C, C, D);
        self.asm.alu(Alu::Bic, Size::X, A, A, D);
        self.asm.alu(Alu::Orr, Size::X, A, A, C);
        self.put(result, A);
    }

    pub(super) fn fcmp(&mut self, cc: FloatCC, x: Value, y: Value, result: Value) {
        let left = self.fin(x, V0);
        let right = self.fin(y, V1);
        self.asm.fcmp(self.ty(x) == Type::F64, left, right);
        let cond = match cc {
            FloatCC::Equal => Cond::Eq,
            FloatCC::NotEqual => Cond::Ne,
            FloatCC::LessThan => Cond::Mi,
            FloatCC::LessThanOrEqual => Cond::Ls,
            FloatCC::GreaterThan => Cond::Gt,
            FloatCC::GreaterThanOrEqual => Cond::Ge,
        };
        self.set_flag(result, cond);
    }

    pub(super) fn float_convert(&mut self, op: Convert, x: Value, result: Value) -> Result<(), String> {
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
                self.asm.fconv(if signed { Fconv::Scvtf } else { Fconv::Ucvtf }, to == Type::F64, true, src, dst);
                self.fput(result, dst);
            }
            Convert::ToSintSat | Convert::ToUintSat => {
                let signed = op == Convert::ToSintSat;
                let src = self.fin(x, V0);
                let dst = self.dst(result);
                let conv = if signed { Fconv::Fcvtzs } else { Fconv::Fcvtzu };
                self.asm.fconv(conv, from == Type::F64, to.bits() > 32, dst, src);
                self.clamp(to.bits(), signed, dst);
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

use super::asm::{Alu, Cond, Fconv, Fop, Funary, Size, Vreg, ZR};
use super::{A, C, D, Emit, Ext, V0, V1};
use crate::ir::{Binary, Convert, FloatCC, Type, Unary, Value};

fn sign(ty: Type) -> i64 {
    if ty == Type::F64 { i64::MIN } else { 0x8000_0000 }
}

impl Emit<'_> {
    fn fload(&mut self, reg: Vreg, value: Value) {
        let at = self.at(value);
        self.asm.vload(self.ty(value).bytes(), reg, at);
    }

    fn fstore(&mut self, value: Value, reg: Vreg) {
        let at = self.at(value);
        self.asm.vstore(self.ty(value).bytes(), at, reg);
    }

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
        self.fload(V0, x);
        self.asm.funary(funary, double, V0, V0);
        self.fstore(result, V0);
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
        self.fload(V0, x);
        self.fload(V1, y);
        self.asm.fop(fop, ty == Type::F64, V0, V0, V1);
        self.fstore(result, V0);
        Ok(())
    }

    fn copysign(&mut self, ty: Type, x: Value, y: Value, result: Value) {
        self.raw(A, x);
        self.raw(C, y);
        self.asm.mov_imm(D, sign(ty));
        self.asm.alu(Alu::And, Size::X, C, C, D);
        self.asm.alu(Alu::Bic, Size::X, A, A, D);
        self.asm.alu(Alu::Orr, Size::X, A, A, C);
        self.put(result, A);
    }

    pub(super) fn fcmp(&mut self, cc: FloatCC, x: Value, y: Value, result: Value) {
        self.fload(V0, x);
        self.fload(V1, y);
        self.asm.fcmp(self.ty(x) == Type::F64, V0, V1);
        let cond = match cc {
            FloatCC::Equal => Cond::Eq,
            FloatCC::NotEqual => Cond::Ne,
            FloatCC::LessThan => Cond::Mi,
            FloatCC::LessThanOrEqual => Cond::Ls,
            FloatCC::GreaterThan => Cond::Gt,
            FloatCC::GreaterThanOrEqual => Cond::Ge,
        };
        let dst = self.dst(result);
        self.asm.cset(cond, dst);
        self.put(result, dst);
    }

    pub(super) fn float_convert(&mut self, op: Convert, x: Value, result: Value) -> Result<(), String> {
        let (from, to) = (self.ty(x), self.ty(result));
        if from == Type::I128 || to == Type::I128 {
            return Err(format!("{op:?} from {from:?} to {to:?}"));
        }
        match op {
            Convert::Fpromote | Convert::Fdemote => {
                self.fload(V0, x);
                self.asm.fcvt(op == Convert::Fdemote, V0, V0);
                self.fstore(result, V0);
            }
            Convert::FromSint | Convert::FromUint => {
                let signed = op == Convert::FromSint;
                self.get(A, x, if signed { Ext::Sign } else { Ext::Zero });
                self.asm.fconv(if signed { Fconv::Scvtf } else { Fconv::Ucvtf }, to == Type::F64, true, A, V0);
                self.fstore(result, V0);
            }
            Convert::ToSintSat | Convert::ToUintSat => {
                let signed = op == Convert::ToSintSat;
                self.fload(V0, x);
                let dst = self.dst(result);
                let conv = if signed { Fconv::Fcvtzs } else { Fconv::Fcvtzu };
                self.asm.fconv(conv, from == Type::F64, to.bits() > 32, dst, V0);
                self.clamp(to.bits(), signed, dst);
                self.put(result, dst);
            }
            other => return Err(format!("{other:?} from {from:?} to {to:?}")),
        }
        Ok(())
    }

    /// Clamps the 32-bit conversion in `dst` to an 8- or 16-bit range.
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

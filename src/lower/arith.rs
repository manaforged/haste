use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{InstBuilder, Value, types};
use hir_ty::PointerCast;
use hir_ty::mir::{BinOp, CastKind, Operand};

use super::{Lower, Place, Scalar, intrinsic};

impl<'db> Lower<'_, '_, 'db> {
    pub(super) fn cast(&mut self, dest: Place<'db>, kind: &CastKind, op: &Operand, size: u64) -> Result<(), String> {
        if matches!(kind, CastKind::PtrToPtr | CastKind::FnPtrToPtr | CastKind::PointerExposeAddress | CastKind::PointerFromExposedAddress) {
            let from = self.operand(op)?;
            self.copy(dest.addr, from.addr, size);
            return Ok(());
        }
        if let CastKind::PointerCoercion(kind) = kind {
            return self.pointer_cast(dest, kind, op, size);
        }
        let (value, from) = self.read(op)?;
        let to = self.scalar(dest.ty.ty)?;
        if matches!(kind, CastKind::FloatToInt) && to.ty.bits() < 32 {
            return self.float_to_narrow(dest, value, to);
        }
        let ins = self.builder.ins();
        let value = match kind {
            CastKind::IntToInt if to.ty.bits() > from.ty.bits() && from.signed => ins.sextend(to.ty, value),
            CastKind::IntToInt if to.ty.bits() > from.ty.bits() => ins.uextend(to.ty, value),
            CastKind::IntToInt if to.ty.bits() < from.ty.bits() => ins.ireduce(to.ty, value),
            CastKind::IntToInt => value,
            CastKind::FloatToFloat if to.ty.bits() > from.ty.bits() => ins.fpromote(to.ty, value),
            CastKind::FloatToFloat if to.ty.bits() < from.ty.bits() => ins.fdemote(to.ty, value),
            CastKind::FloatToFloat => value,
            CastKind::IntToFloat if from.ty.bits() < 32 => {
                let wide = if from.signed { ins.sextend(types::I32, value) } else { ins.uextend(types::I32, value) };
                let ins = self.builder.ins();
                if from.signed { ins.fcvt_from_sint(to.ty, wide) } else { ins.fcvt_from_uint(to.ty, wide) }
            }
            CastKind::IntToFloat if from.signed => ins.fcvt_from_sint(to.ty, value),
            CastKind::IntToFloat => ins.fcvt_from_uint(to.ty, value),
            CastKind::FloatToInt if to.signed => ins.fcvt_to_sint_sat(to.ty, value),
            CastKind::FloatToInt => ins.fcvt_to_uint_sat(to.ty, value),
            other => return Err(format!("cast {other:?}")),
        };
        self.store(value, dest.addr, 0);
        Ok(())
    }

    pub(super) fn pointer_cast(&mut self, dest: Place<'db>, kind: &PointerCast, op: &Operand, size: u64) -> Result<(), String> {
        match kind {
            PointerCast::Unsize => {
                let from = self.operand(op)?;
                self.coerce(from, dest)
            }
            PointerCast::MutToConstPointer | PointerCast::UnsafeFnPointer | PointerCast::ArrayToPointer => {
                let from = self.operand(op)?;
                self.copy(dest.addr, from.addr, size);
                Ok(())
            }
            PointerCast::ReifyFnPointer | PointerCast::ClosureFnPointer(_) => {
                let source = crate::backend::operand_ty(self.db, self.env, self.body, op);
                let cell = self.reify(source)?;
                let cell = self.builder.ins().iconst(types::I64, cell);
                self.store(cell, dest.addr, 0);
                Ok(())
            }
        }
    }

    pub(super) fn float_to_int(&mut self, dest: Place<'db>, from: Place<'db>) -> Result<(), String> {
        let source = self.scalar(from.ty.ty)?;
        let value = self.load(source.ty, from.addr, 0);
        let to = self.scalar(dest.ty.ty)?;
        if to.ty.bits() < 32 {
            return self.float_to_narrow(dest, value, to);
        }
        let ins = self.builder.ins();
        let value = if to.signed { ins.fcvt_to_sint_sat(to.ty, value) } else { ins.fcvt_to_uint_sat(to.ty, value) };
        self.store(value, dest.addr, 0);
        Ok(())
    }

    pub(super) fn float_to_narrow(&mut self, dest: Place<'db>, value: Value, to: Scalar) -> Result<(), String> {
        let bits = to.ty.bits();
        let (low, high) = if to.signed {
            (-(1_i64 << (bits - 1)), (1_i64 << (bits - 1)) - 1)
        } else {
            (0, (1_i64 << bits) - 1)
        };
        let wide = self.builder.ins().fcvt_to_sint_sat(types::I32, value);
        let low = self.builder.ins().iconst(types::I32, low);
        let high = self.builder.ins().iconst(types::I32, high);
        let raised = self.builder.ins().smax(wide, low);
        let clamped = self.builder.ins().smin(raised, high);
        let narrow = self.builder.ins().ireduce(to.ty, clamped);
        self.store(narrow, dest.addr, 0);
        Ok(())
    }

    pub(super) fn binary(&mut self, op: &BinOp, left: Value, right: Value, scalar: Scalar) -> Result<Value, String> {
        if let Some(code) = intrinsic::wide_call(op, scalar.ty, scalar.signed) {
            return Ok(self.scalar_call(code, scalar.ty, &[left, right]));
        }
        let ins = self.builder.ins();
        if scalar.ty.is_float() {
            return Ok(match op {
                BinOp::Add => ins.fadd(left, right),
                BinOp::Sub => ins.fsub(left, right),
                BinOp::Mul => ins.fmul(left, right),
                BinOp::Div => ins.fdiv(left, right),
                BinOp::Eq => ins.fcmp(FloatCC::Equal, left, right),
                BinOp::Ne => ins.fcmp(FloatCC::NotEqual, left, right),
                BinOp::Lt => ins.fcmp(FloatCC::LessThan, left, right),
                BinOp::Le => ins.fcmp(FloatCC::LessThanOrEqual, left, right),
                BinOp::Gt => ins.fcmp(FloatCC::GreaterThan, left, right),
                BinOp::Ge => ins.fcmp(FloatCC::GreaterThanOrEqual, left, right),
                other => return Err(format!("float {other:?}")),
            });
        }
        let ordered = |signed: IntCC, unsigned: IntCC| if scalar.signed { signed } else { unsigned };
        Ok(match op {
            BinOp::Add => ins.iadd(left, right),
            BinOp::Sub => ins.isub(left, right),
            BinOp::Mul => ins.imul(left, right),
            BinOp::Div if scalar.signed => ins.sdiv(left, right),
            BinOp::Div => ins.udiv(left, right),
            BinOp::Rem if scalar.signed => ins.srem(left, right),
            BinOp::Rem => ins.urem(left, right),
            BinOp::BitXor => ins.bxor(left, right),
            BinOp::BitAnd => ins.band(left, right),
            BinOp::BitOr => ins.bor(left, right),
            BinOp::Shl => ins.ishl(left, right),
            BinOp::Shr if scalar.signed => ins.sshr(left, right),
            BinOp::Shr => ins.ushr(left, right),
            BinOp::Eq => ins.icmp(IntCC::Equal, left, right),
            BinOp::Ne => ins.icmp(IntCC::NotEqual, left, right),
            BinOp::Lt => ins.icmp(ordered(IntCC::SignedLessThan, IntCC::UnsignedLessThan), left, right),
            BinOp::Le => ins.icmp(ordered(IntCC::SignedLessThanOrEqual, IntCC::UnsignedLessThanOrEqual), left, right),
            BinOp::Gt => ins.icmp(ordered(IntCC::SignedGreaterThan, IntCC::UnsignedGreaterThan), left, right),
            BinOp::Ge => {
                ins.icmp(ordered(IntCC::SignedGreaterThanOrEqual, IntCC::UnsignedGreaterThanOrEqual), left, right)
            }
            BinOp::Offset => return Err("pointer offset".to_owned()),
        })
    }
}

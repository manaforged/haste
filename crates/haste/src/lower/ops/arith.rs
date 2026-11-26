use anyhow::{Result, bail};
use haste_codegen::{IntCC, Type, Value};

use crate::body::{BinOp, ExprId};
use crate::host::{HostSymbol, MathSymbol};
use crate::names::ItemRef;
use crate::sema::infer::{Adjust, Res};
use crate::sema::substs::Substs;
use crate::sema::ty::Mut;

use super::super::Lower;
use super::{Cmp, Num, compare, float_cc, int_cc};

impl Lower<'_, '_> {
    pub(in crate::lower) fn binary(
        &mut self,
        expr: ExprId,
        op: BinOp,
        lhs: ExprId,
        rhs: ExprId,
    ) -> Result<Option<Value>> {
        if matches!(op, BinOp::And | BinOp::Or) {
            return self.logical(expr, op, lhs, rhs);
        }
        if let Some(&Res::Fn(item, substs)) = self.typed.operators.get(&expr) {
            return self.overloaded(expr, op, (lhs, rhs), (item, substs));
        }
        if self.is_wide(self.ty_of(lhs))? {
            let (Some(x), Some(y)) = (self.eval_operand(lhs)?, self.eval_operand(rhs)?) else {
                return Ok(None);
            };
            return self.wide_compare(op, (x, y)).map(Some);
        }
        let num = self.num(self.ty_of(lhs))?;
        let shift = self.num(self.ty_of(rhs))?;
        let (Some(x), Some(y)) = (self.expr(lhs)?, self.expr(rhs)?) else {
            return Ok(None);
        };
        self.apply(op, num, (x, y), shift).map(Some)
    }

    fn wide_compare(&mut self, op: BinOp, (x, y): (Value, Value)) -> Result<Value> {
        let Some(cmp) = compare(op) else {
            bail!("no arithmetic on wide pointers");
        };
        let (a, b) = (
            self.ins().load(Type::I64, x, 0),
            self.ins().load(Type::I64, y, 0),
        );
        let (c, d) = (
            self.ins().load(Type::I64, x, 8),
            self.ins().load(Type::I64, y, 8),
        );
        let same = self.ins().icmp(IntCC::Equal, a, b);
        let tail = self.ins().icmp(int_cc(cmp, false), c, d);
        Ok(match cmp {
            Cmp::Eq => self.ins().band(same, tail),
            Cmp::Ne => {
                let apart = self.ins().icmp(IntCC::NotEqual, a, b);
                self.ins().bor(apart, tail)
            }
            Cmp::Lt | Cmp::Le | Cmp::Gt | Cmp::Ge => {
                let strict = if matches!(cmp, Cmp::Lt | Cmp::Le) {
                    IntCC::UnsignedLessThan
                } else {
                    IntCC::UnsignedGreaterThan
                };
                let head = self.ins().icmp(strict, a, b);
                let tied = self.ins().band(same, tail);
                self.ins().bor(head, tied)
            }
        })
    }

    fn overloaded(
        &mut self,
        expr: ExprId,
        op: BinOp,
        operands: (ExprId, ExprId),
        (item, substs): (ItemRef, Substs),
    ) -> Result<Option<Value>> {
        let by_ref = Adjust {
            derefs: 0,
            autoref: Some(Mut::Not),
        };
        let mut values = Vec::with_capacity(2);
        for operand in [operands.0, operands.1] {
            let value = if compare(op).is_some() {
                self.receiver((expr, operand), by_ref)?
            } else {
                self.eval_operand(operand)?
            };
            if !self.open {
                return Ok(None);
            }
            values.extend(value);
        }
        self.invoke(
            super::super::reloc::SiteOrigin::Table(crate::sema::infer::SiteKey::Operator(expr)),
            item,
            substs,
            values,
        )
    }

    pub(in crate::lower) fn apply(
        &mut self,
        op: BinOp,
        num: Num,
        (x, y): (Value, Value),
        rhs: Num,
    ) -> Result<Value> {
        if let Some(cc) = compare(op) {
            return Ok(match num {
                Num::Float(_) => self.ins().fcmp(float_cc(cc), x, y),
                Num::Int { signed, .. } => self.ins().icmp(int_cc(cc, signed), x, y),
                Num::Bool => self.ins().icmp(int_cc(cc, false), x, y),
            });
        }
        match num {
            Num::Float(_) => self.float_op(op, x, y),
            Num::Int { ty, signed } => self.int_op(op, (ty, signed), (x, y), rhs),
            Num::Bool => self.bits_op(op, x, y),
        }
    }

    fn bits_op(&mut self, op: BinOp, x: Value, y: Value) -> Result<Value> {
        Ok(match op {
            BinOp::BitAnd => self.ins().band(x, y),
            BinOp::BitOr => self.ins().bor(x, y),
            BinOp::BitXor => self.ins().bxor(x, y),
            other => bail!("cannot lower {other:?} on this type"),
        })
    }

    fn float_op(&mut self, op: BinOp, x: Value, y: Value) -> Result<Value> {
        Ok(match op {
            BinOp::Add => self.ins().fadd(x, y),
            BinOp::Sub => self.ins().fsub(x, y),
            BinOp::Mul => self.ins().fmul(x, y),
            BinOp::Div => self.ins().fdiv(x, y),
            BinOp::Rem => {
                let ty = self.b.func.ty(x);
                let fmod = if ty == Type::F32 {
                    MathSymbol::Fmodf32
                } else {
                    MathSymbol::Fmodf64
                };
                let Some(value) = self.call_host(HostSymbol::Math(fmod), &[x, y], Some(ty)) else {
                    bail!("float remainder returned no value");
                };
                value
            }
            other => bail!("cannot lower float {other:?} yet"),
        })
    }

    fn int_op(
        &mut self,
        op: BinOp,
        (ty, signed): (Type, bool),
        (x, y): (Value, Value),
        rhs: Num,
    ) -> Result<Value> {
        match op {
            BinOp::Add | BinOp::Sub | BinOp::Mul => self.checked(op, (ty, signed), x, y),
            BinOp::Div | BinOp::Rem => self.divide(op, (ty, signed), x, y),
            BinOp::Shl | BinOp::Shr => self.shift(op, (ty, signed), x, (y, rhs)),
            _ => self.bits_op(op, x, y),
        }
    }

    fn checked(&mut self, op: BinOp, num: (Type, bool), x: Value, y: Value) -> Result<Value> {
        if !self.overflow_checks {
            return Ok(match op {
                BinOp::Add => self.ins().iadd(x, y),
                BinOp::Sub => self.ins().isub(x, y),
                _ => self.ins().imul(x, y),
            });
        }
        let (result, ok) = self.overflowing(op, num, x, y)?;
        let msg = match op {
            BinOp::Add => "attempt to add with overflow",
            BinOp::Sub => "attempt to subtract with overflow",
            _ => "attempt to multiply with overflow",
        };
        self.check(ok, msg)?;
        Ok(result)
    }

    /// `op` on `x` and `y` with wrapping, and whether the result did not overflow.
    pub(in crate::lower) fn overflowing(
        &mut self,
        op: BinOp,
        (ty, signed): (Type, bool),
        x: Value,
        y: Value,
    ) -> Result<(Value, Value)> {
        if ty != Type::I128 {
            return Ok(match op {
                BinOp::Add => (self.ins().iadd(x, y), self.ins().add_fits(signed, x, y)),
                BinOp::Sub => (self.ins().isub(x, y), self.ins().sub_fits(signed, x, y)),
                _ => (self.ins().imul(x, y), self.ins().mul_fits(signed, x, y)),
            });
        }
        let result = match op {
            BinOp::Add => self.ins().iadd(x, y),
            BinOp::Sub => self.ins().isub(x, y),
            _ => return self.multiply(signed, x, y),
        };
        let ok = match (op, signed) {
            (BinOp::Add, false) => self
                .ins()
                .icmp(IntCC::UnsignedGreaterThanOrEqual, result, x),
            (_, false) => self.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, x, y),
            (_, true) => {
                let first = self.ins().bxor(x, result);
                let other = if op == BinOp::Add { y } else { x };
                let second = if op == BinOp::Add {
                    self.ins().bxor(other, result)
                } else {
                    self.ins().bxor(other, y)
                };
                let both = self.ins().band(first, second);
                let zero = self.int_const(ty, 0);
                self.ins().icmp(IntCC::SignedGreaterThanOrEqual, both, zero)
            }
        };
        Ok((result, ok))
    }

    fn multiply(&mut self, signed: bool, x: Value, y: Value) -> Result<(Value, Value)> {
        let result = self.ins().imul(x, y);
        Ok((result, self.wide_product_fits(signed, x, y)))
    }

    fn wide_product_fits(&mut self, signed: bool, x: Value, y: Value) -> Value {
        if !signed {
            return self.unsigned_wide_product_fits(x, y);
        }
        let shift = self.int_const(Type::I64, 127);
        let (x_sign, y_sign) = (self.ins().sshr(x, shift), self.ins().sshr(y, shift));
        let magnitude = |lower: &mut Self, value: Value, sign: Value| {
            let flipped = lower.ins().bxor(value, sign);
            lower.ins().isub(flipped, sign)
        };
        let (x_magnitude, y_magnitude) = (magnitude(self, x, x_sign), magnitude(self, y, y_sign));
        let fits = self.unsigned_wide_product_fits(x_magnitude, y_magnitude);
        let product = self.ins().imul(x_magnitude, y_magnitude);
        let negative = self.ins().bxor(x_sign, y_sign);
        let max = self.int_const(Type::I128, i128::MAX as u128);
        let limit = self.ins().isub(max, negative);
        let in_range = self
            .ins()
            .icmp(IntCC::UnsignedLessThanOrEqual, product, limit);
        self.ins().band(fits, in_range)
    }

    fn unsigned_wide_product_fits(&mut self, x: Value, y: Value) -> Value {
        let shift = self.int_const(Type::I64, 64);
        let halves = |lower: &mut Self, value: Value| {
            let high = lower.ins().ushr(value, shift);
            (
                lower.ins().ireduce(Type::I64, value),
                lower.ins().ireduce(Type::I64, high),
            )
        };
        let ((x_low, x_high), (y_low, y_high)) = (halves(self, x), halves(self, y));
        let both_high = {
            let x_wide = self.ins().icmp_imm(IntCC::NotEqual, x_high, 0);
            let y_wide = self.ins().icmp_imm(IntCC::NotEqual, y_high, 0);
            self.ins().band(x_wide, y_wide)
        };
        let mut overflow = both_high;
        let mut sum = self.ins().umulhi(x_low, y_low);
        for (high, low) in [(x_high, y_low), (x_low, y_high)] {
            let spill = self.ins().umulhi(high, low);
            let spilled = self.ins().icmp_imm(IntCC::NotEqual, spill, 0);
            let part = self.ins().imul(high, low);
            let next = self.ins().iadd(sum, part);
            let carried = self.ins().icmp(IntCC::UnsignedLessThan, next, part);
            let either = self.ins().bor(spilled, carried);
            overflow = self.ins().bor(overflow, either);
            sum = next;
        }
        self.ins().icmp_imm(IntCC::Equal, overflow, 0)
    }

    fn divide(
        &mut self,
        op: BinOp,
        (ty, signed): (Type, bool),
        x: Value,
        y: Value,
    ) -> Result<Value> {
        let nonzero = self.ins().icmp_imm(IntCC::NotEqual, y, 0);
        let zero_msg = if op == BinOp::Div {
            "attempt to divide by zero"
        } else {
            "attempt to calculate the remainder with a divisor of zero"
        };
        self.check(nonzero, zero_msg)?;
        if signed {
            let min = self.min_value(ty);
            let not_min = self.ins().icmp(IntCC::NotEqual, x, min);
            let not_minus_one = self.ins().icmp_imm(IntCC::NotEqual, y, -1);
            let ok = self.ins().bor(not_min, not_minus_one);
            let overflow_msg = if op == BinOp::Div {
                "attempt to divide with overflow"
            } else {
                "attempt to calculate the remainder with overflow"
            };
            self.check(ok, overflow_msg)?;
        }
        Ok(match (op, signed) {
            (BinOp::Div, true) => self.ins().sdiv(x, y),
            (BinOp::Div, false) => self.ins().udiv(x, y),
            (_, true) => self.ins().srem(x, y),
            (_, false) => self.ins().urem(x, y),
        })
    }

    fn shift(
        &mut self,
        op: BinOp,
        (ty, signed): (Type, bool),
        x: Value,
        (y, rhs): (Value, Num),
    ) -> Result<Value> {
        let Num::Int { ty: by, .. } = rhs else {
            bail!("shift by a non-integer");
        };
        let target = if ty == Type::I128 { Type::I64 } else { ty };
        let y = if self.overflow_checks {
            let bits = self.int_const(by, u128::from(ty.bits()));
            let ok = self.ins().icmp(IntCC::UnsignedLessThan, y, bits);
            let msg = if op == BinOp::Shl {
                "attempt to shift left with overflow"
            } else {
                "attempt to shift right with overflow"
            };
            self.check(ok, msg)?;
            if by == Type::I128 {
                self.ins().ireduce(target, y)
            } else {
                y
            }
        } else {
            let (y, mask_ty) = if by == Type::I128 {
                (self.ins().ireduce(target, y), target)
            } else {
                (y, by)
            };
            let mask = self.int_const(mask_ty, u128::from(ty.bits() - 1));
            self.ins().band(y, mask)
        };
        Ok(match (op, signed) {
            (BinOp::Shl, _) => self.ins().ishl(x, y),
            (_, true) => self.ins().sshr(x, y),
            (_, false) => self.ins().ushr(x, y),
        })
    }

    fn logical(
        &mut self,
        expr: ExprId,
        op: BinOp,
        lhs: ExprId,
        rhs: ExprId,
    ) -> Result<Option<Value>> {
        let Some(left) = self.expr(lhs)? else {
            return Ok(None);
        };
        let (right_block, join) = (self.b.create_block(), self.b.create_block());
        let machine = super::super::machine(self.cx, self.ty_of(expr))?.unwrap_or(Type::I8);
        let result = self.b.append_block_param(join, machine);
        let short = self.ins().iconst(Type::I8, i64::from(op == BinOp::Or));
        match op {
            BinOp::Or => self.brif(left, (join, &[short]), (right_block, &[])),
            _ => self.brif(left, (right_block, &[]), (join, &[short])),
        };
        self.enter(right_block);
        let right = self.expr(rhs)?;
        if self.open
            && let Some(right) = right
        {
            self.jump(join, &[right]);
        }
        self.enter(join);
        Ok(Some(result))
    }
}

use anyhow::{Result, bail};
use haste_codegen::{IntCC, Type, Value};

use crate::body::ExprId;
use crate::sema::ty::Kind;

use super::super::Lower;
use super::Num;

impl Lower<'_, '_> {
    pub(in crate::lower) fn cast(&mut self, expr: ExprId, inner: ExprId) -> Result<Option<Value>> {
        if self.ty_of(expr) == self.ty_of(inner)
            || self.is_wide(self.ty_of(expr))?
            || self.typed.coercions.contains_key(&inner)
        {
            return self.expr(inner);
        }
        let to = self.num(self.ty_of(expr))?;
        let Some((value, from)) = self.cast_source(inner)? else {
            return Ok(None);
        };
        self.convert(value, from, to).map(Some)
    }

    pub(in crate::lower) fn convert(&mut self, value: Value, from: Num, to: Num) -> Result<Value> {
        Ok(match (from, to) {
            (Num::Int { ty: a, signed }, Num::Int { ty: b, .. }) => {
                self.resize(value, (a, signed), b)
            }
            (Num::Bool, Num::Int { ty: b, .. }) => self.resize(value, (Type::I8, false), b),
            (
                Num::Int {
                    signed,
                    ty: Type::I128,
                },
                Num::Float(b),
            ) => self.int128_to_float(value, signed, b)?,
            (Num::Int { signed, ty: a }, Num::Float(b)) => {
                let wide = self.resize(value, (a, signed), Type::I64);
                if signed {
                    self.ins().fcvt_from_sint(b, wide)
                } else {
                    self.ins().fcvt_from_uint(b, wide)
                }
            }
            (
                Num::Float(_),
                Num::Int {
                    ty: b,
                    signed: true,
                },
            ) => self.ins().fcvt_to_sint_sat(b, value),
            (
                Num::Float(_),
                Num::Int {
                    ty: b,
                    signed: false,
                },
            ) => self.ins().fcvt_to_uint_sat(b, value),
            (Num::Float(a), Num::Float(b)) if a == b => value,
            (Num::Float(Type::F32), Num::Float(b)) => self.ins().fpromote(b, value),
            (Num::Float(_), Num::Float(b)) => self.ins().fdemote(b, value),
            _ => bail!("cannot lower this cast yet"),
        })
    }

    fn pow2_float(&mut self, dst: Type, shift: Value) -> Result<Value> {
        let (int_ty, bias, mant_bits, size, align) = match dst {
            Type::F32 => (Type::I32, 127i64, 23i64, 4u32, 2u8),
            Type::F64 => (Type::I64, 1023i64, 52i64, 8u32, 3u8),
            other => bail!("no float power of {other:?}"),
        };
        let shift = if int_ty == Type::I32 {
            self.ins().ireduce(Type::I32, shift)
        } else {
            shift
        };
        let exp = self.ins().iadd_imm(shift, bias);
        let mant = self.ins().iconst(int_ty, mant_bits);
        let bits = self.ins().ishl(exp, mant);
        let slot = self.b.create_stack_slot(size, align);
        self.ins().stack_store(bits, slot, 0);
        Ok(self.ins().stack_load(dst, slot, 0))
    }

    fn int128_to_float(&mut self, value: Value, signed: bool, dst: Type) -> Result<Value> {
        let (neg, abs) = if signed {
            let zero = self.ins().iconst(Type::I128, 0);
            let neg = self.ins().icmp(IntCC::SignedLessThan, value, zero);
            let flipped = self.ins().ineg(value);
            (Some(neg), self.ins().select(neg, flipped, value))
        } else {
            (None, value)
        };
        let lo = self.ins().ireduce(Type::I64, abs);
        let shift64 = self.ins().iconst(Type::I128, 64);
        let hi_wide = self.ins().ushr(abs, shift64);
        let hi = self.ins().ireduce(Type::I64, hi_wide);
        let hi_zero = self.ins().icmp_imm(IntCC::Equal, hi, 0);
        let lz = self.ins().clz(hi);
        let lz_zero = self.ins().icmp_imm(IntCC::Equal, lz, 0);
        let c64 = self.ins().iconst(Type::I64, 64);
        let shift = self.ins().isub(c64, lz);
        let zero64 = self.ins().iconst(Type::I64, 0);
        let top = self.ins().ishl(hi, lz);
        let mask63 = self.ins().iconst(Type::I64, 63);
        let safe_shift = self.ins().band(shift, mask63);
        let shifted_lo = self.ins().ushr(lo, safe_shift);
        let low_part = self.ins().select(lz_zero, zero64, shifted_lo);
        let rem_shifted = self.ins().ishl(lo, lz);
        let rem = self.ins().select(lz_zero, lo, rem_shifted);
        let rem_nz = self.ins().icmp_imm(IntCC::NotEqual, rem, 0);
        let sticky = self.ins().uextend(Type::I64, rem_nz);
        let top_low = self.ins().bor(top, low_part);
        let mag = self.ins().bor(top_low, sticky);
        let u64_val = self.ins().select(hi_zero, lo, mag);
        let float_val = self.ins().fcvt_from_uint(dst, u64_val);
        let scale = self.pow2_float(dst, shift)?;
        let scaled = self.ins().fmul(float_val, scale);
        let unsigned = self.ins().select(hi_zero, float_val, scaled);
        Ok(match neg {
            Some(neg) => {
                let negated = self.ins().fneg(unsigned);
                self.ins().select(neg, negated, unsigned)
            }
            None => unsigned,
        })
    }

    fn cast_source(&mut self, inner: ExprId) -> Result<Option<(Value, Num)>> {
        let source = self.ty_of(inner);
        if let Kind::Adt(..) = self.cx.types.kind(source) {
            let place = self.place_or_temp(inner)?;
            return self.discriminant(place, source).map(Some);
        }
        if self.is_wide(source)? {
            let Some(pair) = self.expr(inner)? else {
                return Ok(None);
            };
            return Ok(Some((
                self.ins().load(Type::I64, pair, 0),
                Num::Int {
                    ty: Type::I64,
                    signed: false,
                },
            )));
        }
        let from = self.num(source)?;
        Ok(self.expr(inner)?.map(|value| (value, from)))
    }

    pub(in crate::lower) fn resize(
        &mut self,
        value: Value,
        (from, signed): (Type, bool),
        to: Type,
    ) -> Value {
        match from.bits().cmp(&to.bits()) {
            std::cmp::Ordering::Equal => value,
            std::cmp::Ordering::Greater => self.ins().ireduce(to, value),
            std::cmp::Ordering::Less if signed => self.ins().sextend(to, value),
            std::cmp::Ordering::Less => self.ins().uextend(to, value),
        }
    }
}

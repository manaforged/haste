use anyhow::{Result, bail};
use haste_codegen::{FloatCC, IntCC, Type, Value};

use crate::body::literal::{byte_value, char_value, string_bytes};
use crate::body::{BinOp, Expr, ExprId, Lit, UnOp};
use crate::host::{HostSymbol, MathSymbol};
use crate::names::{ItemRef, Primitive};
use crate::sema::infer::parse_int;
use crate::sema::infer::{Adjust, Res};
use crate::sema::substs::Substs;
use crate::sema::ty::{Kind, Mut, Ty, int_info};

use super::stencil::SymbolSource;
use super::{Lower, Parts, Repr};

/// What arithmetic means for one operand type.
#[derive(Clone, Copy)]
pub(super) enum Num {
    Int { ty: Type, signed: bool },
    Float(Type),
    Bool,
}

const WIDE_SIZE: u32 = 16;
const WIDE_ALIGN: u8 = 3;

/// How a reference to a slice or `str` is held: its address, then its length.
pub(super) const WIDE: Repr = Repr::Mem { size: WIDE_SIZE, align: WIDE_ALIGN, parts: Parts::Pair(Type::I64, Type::I64, 8) };

impl Lower<'_, '_> {
    pub(super) fn num(&mut self, ty: Ty) -> Result<Num> {
        let machine = super::machine(self.cx, ty)?;
        Ok(match (self.cx.types.kind(ty), machine) {
            (Kind::Prim(Primitive::Bool), _) => Num::Bool,
            (Kind::Prim(Primitive::F32 | Primitive::F64), Some(ty)) => Num::Float(ty),
            (Kind::Prim(Primitive::Char), Some(ty)) => Num::Int { ty, signed: false },
            (Kind::Prim(primitive), Some(ty)) if int_info(*primitive).is_some() => Num::Int { ty, signed: int_info(*primitive).is_some_and(|(_, signed)| signed) },
            (Kind::Ref(..) | Kind::Ptr(..) | Kind::FnPtr(..), Some(Type::I64)) => Num::Int { ty: Type::I64, signed: false },
            (other, _) => bail!("no arithmetic on {other:?}"),
        })
    }

    pub(super) fn lit(&mut self, expr: ExprId, lit: Lit, at: u32) -> Result<Value> {
        let text = self.src.text(at);
        let ty = super::machine(self.cx, self.ty_of(expr))?.unwrap_or(Type::I8);
        match lit {
            Lit::Bool(value) => Ok(self.ins().iconst(Type::I8, i64::from(value))),
            Lit::Int => Ok(self.int_const(ty, parse_int(text)?.0)),
            Lit::Float => self.float_const(ty, parse_float(text)?),
            Lit::Char => Ok(self.ins().iconst(Type::I32, i64::from(u32::from(char_value(text)?)))),
            Lit::Byte => Ok(self.ins().iconst(Type::I8, i64::from(byte_value(text)?))),
            Lit::Str | Lit::ByteStr | Lit::CStr => self.string_lit(lit, text),
        }
    }

    fn string_lit(&mut self, lit: Lit, text: &str) -> Result<Value> {
        let mut bytes = string_bytes(text)?;
        if lit == Lit::CStr {
            bytes.push(0);
        }
        let wide = lit != Lit::ByteStr;
        let id = self.slots.rodata(&bytes, &[]);
        let data = self.emit_symbol(SymbolSource::Rodata(id));
        Ok(if wide { self.wide_pointer(data, bytes.len() as i64) } else { data })
    }

    /// A new 16-byte slot that holds the pointer `data` and the length `len`, as a reference to a slice or `str`.
    pub(super) fn wide_pointer(&mut self, data: Value, len: i64) -> Value {
        let len = self.ins().iconst(Type::I64, len);
        self.wide(data, len)
    }

    /// A wide pointer in a new frame slot: the address `data`, then the metadata `len`.
    pub(super) fn wide(&mut self, data: Value, len: Value) -> Value {
        let slot = self.b.create_stack_slot(WIDE_SIZE, WIDE_ALIGN);
        self.ins().stack_store(data, slot, 0);
        self.ins().stack_store(len, slot, 8);
        self.ins().stack_addr(Type::I64, slot, 0)
    }

    pub(super) fn int_const(&mut self, ty: Type, value: u128) -> Value {
        if ty != Type::I128 {
            return self.ins().iconst(ty, value as i64);
        }
        let low = self.ins().iconst(Type::I64, value as u64 as i64);
        let high = self.ins().iconst(Type::I64, (value >> 64) as u64 as i64);
        self.ins().iconcat(low, high)
    }

    pub(super) fn float_const(&mut self, ty: Type, value: f64) -> Result<Value> {
        let bits = match ty {
            Type::F32 => i64::from((value as f32).to_bits()),
            Type::F64 => value.to_bits() as i64,
            other => bail!("no float constant of {other:?}"),
        };
        Ok(self.ins().iconst(ty, bits))
    }

    pub(super) fn unary(&mut self, expr: ExprId, op: UnOp, operand: ExprId) -> Result<Option<Value>> {
        if let Some(&Res::Fn(item, substs)) = self.typed.operators.get(&expr) {
            let Some(value) = self.expr(operand)? else { return Ok(None) };
            return self.invoke(super::reloc::SiteOrigin::Table(crate::sema::infer::SiteKey::Operator(expr)), item, substs, vec![value]);
        }
        let num = self.num(self.ty_of(operand))?;
        if let (UnOp::Neg, Num::Int { ty, .. }, Some(at)) = (op, num, self.int_literal(operand)) {
            let value = parse_int(self.src.text(at))?.0;
            return Ok(Some(self.int_const(ty, value.wrapping_neg())));
        }
        let Some(value) = self.expr(operand)? else {
            return Ok(None);
        };
        Ok(Some(match (op, num) {
            (UnOp::Not, Num::Bool) => self.ins().bxor_imm(value, 1),
            (UnOp::Not, Num::Int { .. }) => self.ins().bnot(value),
            (UnOp::Neg, Num::Float(_)) => self.ins().fneg(value),
            (UnOp::Neg, Num::Int { ty, signed: true }) => {
                let min = self.min_value(ty);
                let ok = self.ins().icmp(IntCC::NotEqual, value, min);
                self.overflow_check(ok);
                self.ins().ineg(value)
            }
            _ => bail!("cannot lower {op:?} on this type"),
        }))
    }

    fn int_literal(&self, expr: ExprId) -> Option<u32> {
        match *self.body.expr(expr) {
            Expr::Paren(inner) => self.int_literal(inner),
            Expr::Lit(Lit::Int, at) => Some(at),
            _ => None,
        }
    }

    pub(super) fn min_value(&mut self, ty: Type) -> Value {
        let bits = ty.bits();
        self.int_const(ty, 1u128 << (bits - 1))
    }

    /// Traps unless `ok` is nonzero, where rustc's debug build panics.
    fn check(&mut self, ok: Value) {
        self.ins().trapz(ok);
    }

    /// Checks `ok` where the body's crate checks overflow, as rustc's MIR for that crate does.
    fn overflow_check(&mut self, ok: Value) {
        if self.overflow_checks {
            self.check(ok);
        }
    }

    pub(super) fn binary(&mut self, expr: ExprId, op: BinOp, lhs: ExprId, rhs: ExprId) -> Result<Option<Value>> {
        if matches!(op, BinOp::And | BinOp::Or) {
            return self.logical(expr, op, lhs, rhs);
        }
        if let Some(&Res::Fn(item, substs)) = self.typed.operators.get(&expr) {
            return self.overloaded(expr, op, (lhs, rhs), (item, substs));
        }
        if self.is_wide(self.ty_of(lhs))? {
            let (Some(x), Some(y)) = (self.expr(lhs)?, self.expr(rhs)?) else {
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

    /// Compares two wide pointers held in memory by address, then by metadata, as rustc compares them.
    fn wide_compare(&mut self, op: BinOp, (x, y): (Value, Value)) -> Result<Value> {
        let Some(cmp) = compare(op) else { bail!("no arithmetic on wide pointers") };
        let (a, b) = (self.ins().load(Type::I64, x, 0), self.ins().load(Type::I64, y, 0));
        let (c, d) = (self.ins().load(Type::I64, x, 8), self.ins().load(Type::I64, y, 8));
        let same = self.ins().icmp(IntCC::Equal, a, b);
        let tail = self.ins().icmp(int_cc(cmp, false), c, d);
        Ok(match cmp {
            Cmp::Eq => self.ins().band(same, tail),
            Cmp::Ne => {
                let apart = self.ins().icmp(IntCC::NotEqual, a, b);
                self.ins().bor(apart, tail)
            }
            Cmp::Lt | Cmp::Le | Cmp::Gt | Cmp::Ge => {
                let strict = if matches!(cmp, Cmp::Lt | Cmp::Le) { IntCC::UnsignedLessThan } else { IntCC::UnsignedGreaterThan };
                let head = self.ins().icmp(strict, a, b);
                let tied = self.ins().band(same, tail);
                self.ins().bor(head, tied)
            }
        })
    }

    fn overloaded(&mut self, expr: ExprId, op: BinOp, operands: (ExprId, ExprId), (item, substs): (ItemRef, Substs)) -> Result<Option<Value>> {
        let by_ref = Adjust { derefs: 0, autoref: Some(Mut::Not) };
        let mut values = Vec::with_capacity(2);
        for operand in [operands.0, operands.1] {
            let value = if compare(op).is_some() { self.receiver((expr, operand), by_ref)? } else { self.expr(operand)? };
            let Some(value) = value else { return Ok(None) };
            values.push(value);
        }
        self.invoke(super::reloc::SiteOrigin::Table(crate::sema::infer::SiteKey::Operator(expr)), item, substs, values)
    }

    fn overloaded_assign(&mut self, (expr, lhs, rhs): (ExprId, ExprId, ExprId), (item, substs): (ItemRef, Substs)) -> Result<Option<Value>> {
        let by_mut = Adjust { derefs: 0, autoref: Some(Mut::Mut) };
        let Some(place) = self.receiver((expr, lhs), by_mut)? else {
            return Ok(None);
        };
        let mut values = vec![place];
        values.extend(self.expr(rhs)?);
        self.invoke(super::reloc::SiteOrigin::Table(crate::sema::infer::SiteKey::Operator(expr)), item, substs, values).map(|_| None)
    }

    pub(super) fn apply(&mut self, op: BinOp, num: Num, (x, y): (Value, Value), rhs: Num) -> Result<Value> {
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
                let fmod = if ty == Type::F32 { MathSymbol::Fmodf32 } else { MathSymbol::Fmodf64 };
                let Some(value) = self.call_host(HostSymbol::Math(fmod), &[x, y], Some(ty)) else { bail!("float remainder returned no value") };
                value
            }
            other => bail!("cannot lower float {other:?} yet"),
        })
    }

    fn int_op(&mut self, op: BinOp, (ty, signed): (Type, bool), (x, y): (Value, Value), rhs: Num) -> Result<Value> {
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
        self.check(ok);
        Ok(result)
    }

    /// `op` on `x` and `y` with wrapping, and whether the result did not overflow.
    pub(super) fn overflowing(&mut self, op: BinOp, (ty, signed): (Type, bool), x: Value, y: Value) -> Result<(Value, Value)> {
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
            (BinOp::Add, false) => self.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, result, x),
            (_, false) => self.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, x, y),
            (_, true) => {
                let first = self.ins().bxor(x, result);
                let other = if op == BinOp::Add { y } else { x };
                let second = if op == BinOp::Add { self.ins().bxor(other, result) } else { self.ins().bxor(other, y) };
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
        let in_range = self.ins().icmp(IntCC::UnsignedLessThanOrEqual, product, limit);
        self.ins().band(fits, in_range)
    }

    fn unsigned_wide_product_fits(&mut self, x: Value, y: Value) -> Value {
        let shift = self.int_const(Type::I64, 64);
        let halves = |lower: &mut Self, value: Value| {
            let high = lower.ins().ushr(value, shift);
            (lower.ins().ireduce(Type::I64, value), lower.ins().ireduce(Type::I64, high))
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

    fn divide(&mut self, op: BinOp, (ty, signed): (Type, bool), x: Value, y: Value) -> Result<Value> {
        let nonzero = self.ins().icmp_imm(IntCC::NotEqual, y, 0);
        self.check(nonzero);
        if signed {
            let min = self.min_value(ty);
            let not_min = self.ins().icmp(IntCC::NotEqual, x, min);
            let not_minus_one = self.ins().icmp_imm(IntCC::NotEqual, y, -1);
            let ok = self.ins().bor(not_min, not_minus_one);
            self.check(ok);
        }
        Ok(match (op, signed) {
            (BinOp::Div, true) => self.ins().sdiv(x, y),
            (BinOp::Div, false) => self.ins().udiv(x, y),
            (_, true) => self.ins().srem(x, y),
            (_, false) => self.ins().urem(x, y),
        })
    }

    fn shift(&mut self, op: BinOp, (ty, signed): (Type, bool), x: Value, (y, rhs): (Value, Num)) -> Result<Value> {
        let Num::Int { ty: by, .. } = rhs else { bail!("shift by a non-integer") };
        let y = if self.overflow_checks {
            let bits = self.int_const(by, u128::from(ty.bits()));
            let ok = self.ins().icmp(IntCC::UnsignedLessThan, y, bits);
            self.check(ok);
            y
        } else {
            let mask = self.int_const(by, u128::from(ty.bits() - 1));
            self.ins().band(y, mask)
        };
        Ok(match (op, signed) {
            (BinOp::Shl, _) => self.ins().ishl(x, y),
            (_, true) => self.ins().sshr(x, y),
            (_, false) => self.ins().ushr(x, y),
        })
    }

    fn logical(&mut self, expr: ExprId, op: BinOp, lhs: ExprId, rhs: ExprId) -> Result<Option<Value>> {
        let Some(left) = self.expr(lhs)? else {
            return Ok(None);
        };
        let (right_block, join) = (self.b.create_block(), self.b.create_block());
        let machine = super::machine(self.cx, self.ty_of(expr))?.unwrap_or(Type::I8);
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

    pub(super) fn assign(&mut self, expr: ExprId, (lhs, rhs): (ExprId, ExprId), op: Option<BinOp>) -> Result<Option<Value>> {
        if let Some(op) = op {
            return self.compound(expr, (lhs, rhs), op);
        }
        if matches!(self.body.expr(lhs), Expr::Tuple(_) | Expr::Underscore) {
            let value = self.expr(rhs)?;
            return self.assign_parts(lhs, value).map(|_| None);
        }
        let Some(value) = self.expr(rhs)? else {
            return Ok(None);
        };
        let Some(place) = self.place_of(lhs)? else { bail!("cannot lower assignment to this expression yet") };
        if !self.replace(lhs, place, value)? {
            self.write(place, value);
        }
        Ok(None)
    }

    fn compound(&mut self, expr: ExprId, (lhs, rhs): (ExprId, ExprId), op: BinOp) -> Result<Option<Value>> {
        if let Some(&Res::Fn(item, substs)) = self.typed.operators.get(&expr) {
            return self.overloaded_assign((expr, lhs, rhs), (item, substs));
        }
        let rhs_num = self.num(self.ty_of(rhs)).ok();
        let Some(value) = self.expr(rhs)? else {
            return Ok(None);
        };
        let Some(place) = self.place_of(lhs)? else { bail!("cannot lower assignment to this expression yet") };
        let Some(current) = self.read(place) else {
            return Ok(None);
        };
        let num = self.num(self.ty_of(lhs))?;
        let value = self.apply(op, num, (current, value), rhs_num.unwrap_or(num))?;
        self.write(place, value);
        Ok(None)
    }

    fn assign_parts(&mut self, lhs: ExprId, value: Option<Value>) -> Result<()> {
        match *self.body.expr(lhs) {
            Expr::Underscore => Ok(()),
            Expr::Paren(inner) => self.assign_parts(inner, value),
            Expr::Tuple(items) => {
                let ty = self.ty_of(lhs);
                let whole = self.spill(value, ty)?;
                for (index, &item) in self.body.list(items).to_vec().iter().enumerate() {
                    let part = match whole {
                        Some(whole) => self.part_place(whole, ty, (0, index as u32)).map(|part| self.read(part))?,
                        None => None,
                    };
                    self.assign_parts(ExprId(item), part)?;
                }
                Ok(())
            }
            _ => {
                let Some(place) = self.place_of(lhs)? else { bail!("cannot lower assignment to this expression yet") };
                let Some(value) = value else { return Ok(()) };
                if !self.replace(lhs, place, value)? {
                    self.write(place, value);
                }
                Ok(())
            }
        }
    }

    pub(super) fn cast(&mut self, expr: ExprId, inner: ExprId) -> Result<Option<Value>> {
        if self.is_wide(self.ty_of(expr))? || self.typed.coercions.contains_key(&inner) {
            return self.expr(inner);
        }
        let to = self.num(self.ty_of(expr))?;
        let Some((value, from)) = self.cast_source(inner)? else {
            return Ok(None);
        };
        self.convert(value, from, to).map(Some)
    }

    pub(super) fn convert(&mut self, value: Value, from: Num, to: Num) -> Result<Value> {
        Ok(match (from, to) {
            (Num::Int { ty: a, signed }, Num::Int { ty: b, .. }) => self.resize(value, (a, signed), b),
            (Num::Bool, Num::Int { ty: b, .. }) => self.resize(value, (Type::I8, false), b),
            (Num::Int { signed, ty: a }, Num::Float(b)) => {
                let wide = self.resize(value, (a, signed), Type::I64);
                if signed { self.ins().fcvt_from_sint(b, wide) } else { self.ins().fcvt_from_uint(b, wide) }
            }
            (Num::Float(_), Num::Int { ty: b, signed: true }) => self.ins().fcvt_to_sint_sat(b, value),
            (Num::Float(_), Num::Int { ty: b, signed: false }) => self.ins().fcvt_to_uint_sat(b, value),
            (Num::Float(a), Num::Float(b)) if a == b => value,
            (Num::Float(Type::F32), Num::Float(b)) => self.ins().fpromote(b, value),
            (Num::Float(_), Num::Float(b)) => self.ins().fdemote(b, value),
            _ => bail!("cannot lower this cast yet"),
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
            return Ok(Some((self.ins().load(Type::I64, pair, 0), Num::Int { ty: Type::I64, signed: false })));
        }
        let from = self.num(source)?;
        Ok(self.expr(inner)?.map(|value| (value, from)))
    }

    /// Whether `ty` is a pointer that carries a length after its address.
    fn is_wide(&mut self, ty: Ty) -> Result<bool> {
        match self.cx.types.kind(ty).clone() {
            Kind::Ref(_, pointee) | Kind::Ptr(_, pointee) => self.cx.is_unsized(pointee),
            _ => Ok(false),
        }
    }

    pub(super) fn resize(&mut self, value: Value, (from, signed): (Type, bool), to: Type) -> Value {
        match from.bits().cmp(&to.bits()) {
            std::cmp::Ordering::Equal => value,
            std::cmp::Ordering::Greater => self.ins().ireduce(to, value),
            std::cmp::Ordering::Less if signed => self.ins().sextend(to, value),
            std::cmp::Ordering::Less => self.ins().uextend(to, value),
        }
    }
}

/// An ordering or equality test.
#[derive(Clone, Copy)]
pub(super) enum Cmp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// The comparison that `op` is, when it compares.
pub(super) fn compare(op: BinOp) -> Option<Cmp> {
    Some(match op {
        BinOp::Eq => Cmp::Eq,
        BinOp::Ne => Cmp::Ne,
        BinOp::Lt => Cmp::Lt,
        BinOp::Le => Cmp::Le,
        BinOp::Gt => Cmp::Gt,
        BinOp::Ge => Cmp::Ge,
        _ => return None,
    })
}

fn int_cc(cmp: Cmp, signed: bool) -> IntCC {
    match (cmp, signed) {
        (Cmp::Eq, _) => IntCC::Equal,
        (Cmp::Ne, _) => IntCC::NotEqual,
        (Cmp::Lt, true) => IntCC::SignedLessThan,
        (Cmp::Le, true) => IntCC::SignedLessThanOrEqual,
        (Cmp::Gt, true) => IntCC::SignedGreaterThan,
        (Cmp::Ge, true) => IntCC::SignedGreaterThanOrEqual,
        (Cmp::Lt, false) => IntCC::UnsignedLessThan,
        (Cmp::Le, false) => IntCC::UnsignedLessThanOrEqual,
        (Cmp::Gt, false) => IntCC::UnsignedGreaterThan,
        (Cmp::Ge, false) => IntCC::UnsignedGreaterThanOrEqual,
    }
}

fn float_cc(cmp: Cmp) -> FloatCC {
    match cmp {
        Cmp::Eq => FloatCC::Equal,
        Cmp::Ne => FloatCC::NotEqual,
        Cmp::Lt => FloatCC::LessThan,
        Cmp::Le => FloatCC::LessThanOrEqual,
        Cmp::Gt => FloatCC::GreaterThan,
        Cmp::Ge => FloatCC::GreaterThanOrEqual,
    }
}

fn parse_float(text: &str) -> Result<f64> {
    let digits: String = text.chars().filter(|ch| *ch != '_').collect();
    let digits = ["f16", "f32", "f64", "f128"].iter().find_map(|suffix| digits.strip_suffix(suffix)).unwrap_or(&digits);
    Ok(digits.parse()?)
}

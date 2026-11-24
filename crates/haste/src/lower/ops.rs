mod arith;
mod assign;
mod cast;

use anyhow::{Result, bail};
use haste_codegen::{FloatCC, IntCC, Type, Value};

use crate::body::literal::{byte_value, char_value, string_bytes};
use crate::body::{BinOp, Expr, ExprId, Lit, UnOp};
use crate::names::{ItemRef, Primitive};
use crate::sema::infer::{Res, parse_int};
use crate::sema::substs::Substs;
use crate::sema::ty::{Kind, Ty, int_info};

use super::{Lower, Parts, Repr};

/// What arithmetic means for one operand type.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Num {
    Int { ty: Type, signed: bool },
    Float(Type),
    Bool,
}

const WIDE_SIZE: u32 = 16;
const WIDE_ALIGN: u8 = 3;

/// How a reference to a slice or `str` is held: its address, then its length.
pub(super) const WIDE: Repr = Repr::Mem {
    size: WIDE_SIZE,
    align: WIDE_ALIGN,
    parts: Parts::Pair(Type::I64, Type::I64, 8),
};

impl Lower<'_, '_> {
    pub(super) fn num(&mut self, ty: Ty) -> Result<Num> {
        let machine = super::machine(self.cx, ty)?;
        Ok(match (self.cx.types.kind(ty), machine) {
            (Kind::Prim(Primitive::Bool), _) => Num::Bool,
            (Kind::Prim(Primitive::F32 | Primitive::F64), Some(ty)) => Num::Float(ty),
            (Kind::Prim(Primitive::Char), Some(ty)) => Num::Int { ty, signed: false },
            (Kind::Prim(primitive), Some(ty)) if int_info(*primitive).is_some() => Num::Int {
                ty,
                signed: int_info(*primitive).is_some_and(|(_, signed)| signed),
            },
            (Kind::Ref(..) | Kind::Ptr(..) | Kind::FnPtr(..), Some(Type::I64)) => Num::Int {
                ty: Type::I64,
                signed: false,
            },
            (other, _) => bail!("no arithmetic on {other:?}"),
        })
    }

    pub(super) fn lit(&mut self, expr: ExprId, lit: Lit, at: u32) -> Result<Value> {
        let text = self.src.text(at);
        let ty = super::machine(self.cx, self.ty_of(expr))?.unwrap_or(Type::I8);
        match lit {
            Lit::Bool(value) => Ok(self.ins().iconst(Type::I8, i64::from(value))),
            Lit::Int => Ok(self.int_const(ty, parse_int(text)?.0)),
            Lit::Float => Ok(self.ins().iconst(ty, parse_float(ty, text)?)),
            Lit::Char => Ok(self
                .ins()
                .iconst(Type::I32, i64::from(u32::from(char_value(text)?)))),
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
        let data = self.emit_const_rodata(&bytes);
        Ok(if wide {
            self.wide_pointer(data, bytes.len() as i64)
        } else {
            data
        })
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

    pub(super) fn unary(
        &mut self,
        expr: ExprId,
        op: UnOp,
        operand: ExprId,
    ) -> Result<Option<Value>> {
        if let Some(&Res::Fn(item, substs)) = self.typed.operators.get(&expr) {
            let values: Vec<Value> = self.expr(operand)?.into_iter().collect();
            if !self.open {
                return Ok(None);
            }
            return self.invoke(
                super::reloc::SiteOrigin::Table(crate::sema::infer::SiteKey::Operator(expr)),
                item,
                substs,
                values,
            );
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
                self.overflow_check(ok, "attempt to negate with overflow")?;
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

    fn is_wide(&mut self, ty: Ty) -> Result<bool> {
        match self.cx.types.kind(ty).clone() {
            Kind::Ref(_, pointee) | Kind::Ptr(_, pointee) => self.cx.is_unsized(pointee),
            _ => Ok(false),
        }
    }

    /// Traps or unwinds with `msg` unless `ok` is nonzero.
    pub(super) fn check(&mut self, ok: Value, msg: &str) -> Result<()> {
        self.check_with(ok, |lower| lower.panic_str(msg))
    }

    /// Traps or runs `on_fail` unless `ok` is nonzero.
    pub(super) fn check_with(
        &mut self,
        ok: Value,
        on_fail: impl FnOnce(&mut Self) -> Result<()>,
    ) -> Result<()> {
        if !self.open {
            return Ok(());
        }
        if !self.should_unwind() {
            self.ins().trapz(ok);
            return Ok(());
        }
        let (pass, fail) = (self.b.create_block(), self.b.create_block());
        self.brif(ok, (pass, &[]), (fail, &[]));
        self.enter(fail);
        on_fail(self)?;
        self.enter(pass);
        Ok(())
    }

    /// Returns the reached index of `(item, substs)` if inferred, else requests it in lazy JIT or returns `Ok(None)`.
    pub(super) fn try_reached_index(&self, item: ItemRef, substs: Substs) -> Result<Option<usize>> {
        let instance = (item, substs);
        if let Some(&index) = self.reached.index.get(&instance)
            && !self.reached.is_untyped(index)
        {
            return Ok(Some(index));
        }
        let before = self.slots.wanted_len();
        self.slots.want(instance);
        if self.slots.wanted_len() > before {
            bail!("a use of `{}` that was not reached", self.cx.name(item));
        }
        Ok(None)
    }

    fn trap_and_close(&mut self) {
        if self.open {
            self.ins().trap();
            self.close();
        }
    }

    fn panic_lang_call(
        &mut self,
        lang: &str,
        args: impl FnOnce(&mut Self) -> Vec<Value>,
    ) -> Result<bool> {
        let Some(item) = self.cx.lang_item(lang) else {
            return Ok(false);
        };
        if self.try_reached_index(item, Substs::EMPTY)?.is_none() {
            return Ok(false);
        }
        let values = args(self);
        self.invoke(
            super::reloc::SiteOrigin::Known(item, Substs::EMPTY),
            item,
            Substs::EMPTY,
            values,
        )?;
        self.trap_and_close();
        Ok(true)
    }

    pub(super) fn panic_str(&mut self, msg: &str) -> Result<()> {
        if !self.panic_lang_call("panic", |lower| {
            let data = lower.emit_const_rodata(msg.as_bytes());
            vec![lower.wide_pointer(data, msg.len() as i64)]
        })? {
            self.trap_and_close();
        }
        Ok(())
    }

    pub(super) fn panic_bounds(&mut self, at: Value, len: Value) -> Result<()> {
        if !self.panic_lang_call(crate::sema::reach::PANIC_BOUNDS_CHECK, |_| vec![at, len])? {
            self.panic_str("index out of bounds")?;
        }
        Ok(())
    }

    fn overflow_check(&mut self, ok: Value, msg: &str) -> Result<()> {
        if self.overflow_checks {
            self.check(ok, msg)?;
        }
        Ok(())
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

pub(super) fn parse_float(ty: Type, text: &str) -> Result<i64> {
    let digits: String = text.chars().filter(|ch| *ch != '_').collect();
    let digits = ["f16", "f32", "f64", "f128"]
        .iter()
        .find_map(|suffix| digits.strip_suffix(suffix))
        .unwrap_or(&digits);
    match ty {
        Type::F32 => Ok(i64::from(digits.parse::<f32>()?.to_bits())),
        Type::F64 => Ok(digits.parse::<f64>()?.to_bits() as i64),
        other => bail!("no float constant of {other:?}"),
    }
}

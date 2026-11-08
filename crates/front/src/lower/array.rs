use anyhow::{Result, bail};
use haste_codegen::{IntCC, Type, Value};

use crate::body::{ExprId, List};
use crate::sema::ty::Kind;

use super::place::{Base, Place};
use super::{Lower, repr};

impl Lower<'_, '_> {
    /// Lowers `[items]` into a new frame slot and returns its address.
    pub fn array(&mut self, expr: ExprId, items: List) -> Result<Option<Value>> {
        let ty = self.ty_of(expr);
        let whole = repr(self.cx, ty)?;
        let slot = self.slot(whole);
        let Kind::Array(element, _) = *self.cx.types.kind(ty) else { bail!("array literal of a type that is not an array") };
        let (stride, part) = (self.cx.layout(element)?.size.bytes() as i32, repr(self.cx, element)?);
        for (at, item) in self.body.list(items).to_vec().into_iter().enumerate() {
            let value = self.expr(ExprId(item))?;
            if let (Some(slot), Some(value)) = (slot, value) {
                self.write(Place::slot(slot, whole).at(at as i32 * stride, part), value);
            }
        }
        Ok(slot.and_then(|slot| self.read(Place::slot(slot, whole))))
    }

    /// Lowers `[value; len]` into a new frame slot and returns its address.
    pub fn repeat(&mut self, expr: ExprId, value: ExprId) -> Result<Option<Value>> {
        let ty = self.ty_of(expr);
        let whole = repr(self.cx, ty)?;
        let Kind::Array(element, count) = *self.cx.types.kind(ty) else { bail!("repeat of a type that is not an array") };
        let (stride, part) = (self.cx.layout(element)?.size.bytes() as i32, repr(self.cx, element)?);
        let value = self.expr(value)?;
        let Some(slot) = self.slot(whole) else { return Ok(None) };
        if let Some(value) = value {
            for at in 0..count as i32 {
                self.write(Place::slot(slot, whole).at(at * stride, part), value);
            }
        }
        Ok(self.read(Place::slot(slot, whole)))
    }

    /// The place of the element `base[index]` of an array or slice, through references, after a bounds check.
    pub fn index_place(&mut self, expr: ExprId, base: ExprId, index: ExprId) -> Result<Option<Place>> {
        let mut place = self.place_or_temp(base)?;
        let mut ty = self.ty_of(base);
        let (data, len) = loop {
            match *self.cx.types.kind(ty) {
                Kind::Array(_, count) => {
                    let Some(at) = place else { return Ok(None) };
                    let data = self.address(at);
                    break (data, self.ins().iconst(Type::I64, count as i64));
                }
                _ if self.is_unsized(ty) => {
                    let Some(at) = place else { bail!("slice with no wide pointer") };
                    let wide = self.address(at);
                    break (self.ins().load(Type::I64, wide, 0), self.ins().load(Type::I64, wide, 8));
                }
                Kind::Ref(_, inner) | Kind::Ptr(_, inner) => {
                    let Some(pointer) = place.and_then(|at| self.read(at)) else { bail!("reference with no value") };
                    place = Some(self.pointee_place(pointer, inner)?);
                    ty = inner;
                }
                _ => bail!("builtin index of a type that is not an array or slice"),
            }
        };
        let Some(at) = self.expr(index)? else { return Ok(None) };
        let inside = self.ins().icmp(IntCC::UnsignedLessThan, at, len);
        self.ins().trapz(inside);
        let element = self.ty_of(expr);
        let stride = self.cx.layout(element)?.size.bytes() as i64;
        let offset = self.ins().imul_imm(at, stride);
        let address = self.ins().iadd(data, offset);
        Ok(Some(Place { base: Base::Addr(address), offset: 0, repr: repr(self.cx, element)? }))
    }
}

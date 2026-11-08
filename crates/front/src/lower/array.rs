use anyhow::{Result, bail};
use haste_codegen::{IntCC, Type, Value};

use crate::body::{ExprId, List};
use crate::sema::infer::Res;
use crate::sema::ty::{Kind, Ty};

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
            for at in 0..self.len(count)? as i32 {
                self.write(Place::slot(slot, whole).at(at * stride, part), value);
            }
        }
        Ok(self.read(Place::slot(slot, whole)))
    }

    /// The place that `base[index]` names after the autoderef steps of `base`: an element of an array or slice after a bounds check, or the target of `Index::index`.
    pub fn index_place(&mut self, expr: ExprId, base: ExprId, index: ExprId) -> Result<Option<Place>> {
        let Some((place, ty)) = self.autoderef_place(expr, base)? else { return Ok(None) };
        if let Some(Res::Fn(item, substs)) = self.typed.operators.get(&expr).cloned() {
            let receiver = self.address(place);
            let values = self.expr(index)?.into_iter();
            let Some(pointer) = self.invoke(item, &substs, std::iter::once(receiver).chain(values).collect())? else { return Ok(None) };
            return self.pointee_place(pointer, self.ty_of(expr)).map(Some);
        }
        let Some((data, len)) = self.data_len(Some(place), ty)? else { return Ok(None) };
        let Some(at) = self.expr(index)? else { return Ok(None) };
        let inside = self.ins().icmp(IntCC::UnsignedLessThan, at, len);
        self.ins().trapz(inside);
        self.element_place(data, at, self.ty_of(expr)).map(Some)
    }

    /// The data pointer and length of the array or slice of type `ty` at `place`; `None` when it has no bytes.
    pub(super) fn data_len(&mut self, place: Option<Place>, ty: Ty) -> Result<Option<(Value, Value)>> {
        if let Kind::Array(_, count) = *self.cx.types.kind(ty) {
            let Some(at) = place else { return Ok(None) };
            let data = self.address(at);
            let count = self.len(count)?;
            return Ok(Some((data, self.ins().iconst(Type::I64, count))));
        }
        let Some(at) = place else { bail!("slice with no wide pointer") };
        let wide = self.address(at);
        Ok(Some((self.ins().load(Type::I64, wide, 0), self.ins().load(Type::I64, wide, 8))))
    }

    /// The place of element `index` of type `element` in the array data at `data`.
    pub(super) fn element_place(&mut self, data: Value, index: Value, element: Ty) -> Result<Place> {
        let stride = self.cx.layout(element)?.size.bytes() as i64;
        let offset = self.ins().imul_imm(index, stride);
        let address = self.ins().iadd(data, offset);
        Ok(Place { base: Base::Addr(address), offset: 0, repr: repr(self.cx, element)? })
    }
}

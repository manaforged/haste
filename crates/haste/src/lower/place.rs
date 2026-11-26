mod deref;
mod eval;

use anyhow::{Result, bail};
use haste_codegen::{StackSlot, Type, Value};

use crate::body::{BlockKind, Expr, ExprId, UnOp};
use crate::sema::infer::{Coercion, Res};
use crate::sema::ty::{Kind, Ty};

use super::{Lower, Parts, Repr, repr};

/// Where a place's bytes start: a frame slot, or an address held in a value.
#[derive(Clone, Copy, Debug)]
pub enum Base {
    Slot(StackSlot),
    Addr(Value),
}

/// A place in memory that holds a value of one representation.
#[derive(Clone, Copy, Debug)]
pub struct Place {
    pub base: Base,
    pub offset: i32,
    pub repr: Repr,
}

impl Place {
    /// The start of frame slot `slot`.
    pub fn slot(slot: StackSlot, repr: Repr) -> Place {
        Place {
            base: Base::Slot(slot),
            offset: 0,
            repr,
        }
    }

    /// The place `offset` bytes further on, holding a value of `repr`.
    pub fn at(self, offset: i32, repr: Repr) -> Place {
        Place {
            base: self.base,
            offset: self.offset + offset,
            repr,
        }
    }
}

impl Lower<'_, '_> {
    /// The value in `place`: a scalar loaded from it, or its address for a memory value.
    pub fn read(&mut self, place: Place) -> Option<Value> {
        match (place.repr, place.base) {
            (Repr::Zst, _) => None,
            (Repr::Scalar(ty), Base::Slot(slot)) => {
                Some(self.ins().stack_load(ty, slot, place.offset))
            }
            (Repr::Scalar(ty), Base::Addr(addr)) => Some(self.ins().load(ty, addr, place.offset)),
            (Repr::Mem { .. }, _) => Some(self.address(place)),
        }
    }

    /// Reads `place` of type `ty`; a sized `Unique` comes back in a fresh frame slot, its pointer marked as owning a heap allocation.
    pub fn read_typed(&mut self, place: Place, ty: Ty) -> Option<Value> {
        if !matches!(
            place.repr,
            Repr::Mem {
                size: 8,
                parts: Parts::One(Type::I64),
                ..
            }
        ) || !self.cx.is_unique(ty)
        {
            return self.read(place);
        }
        let pointer = self.read(place.at(0, Repr::Scalar(Type::I64)))?;
        self.b.func.mark_unique_ptr(pointer);
        let slot = self.slot(place.repr)?;
        self.write(Place::slot(slot, Repr::Scalar(Type::I64)), pointer);
        Some(self.address(Place::slot(slot, place.repr)))
    }

    /// Stores `value` into `place`, copying the bytes of a memory value.
    pub fn write(&mut self, place: Place, value: Value) {
        match (place.repr, place.base) {
            (Repr::Zst, _) => {}
            (Repr::Scalar(_), Base::Slot(slot)) => {
                self.ins().stack_store(value, slot, place.offset);
            }
            (Repr::Scalar(_), Base::Addr(addr)) => {
                self.ins().store(value, addr, place.offset);
            }
            (Repr::Mem { .. }, _) => {
                let to = self.address(place);
                self.copy(to, value, place.repr);
            }
        }
    }

    /// The address of `place`.
    pub fn address(&mut self, place: Place) -> Value {
        match place.base {
            Base::Slot(slot) => self.ins().stack_addr(Type::I64, slot, place.offset),
            Base::Addr(addr) if place.offset == 0 => addr,
            Base::Addr(addr) => self.ins().iadd_imm(addr, i64::from(place.offset)),
        }
    }

    /// Copies a value of `repr` from address `from` to address `to`: scalar by scalar at the widths and offsets its layout names, else as one move of its bytes.
    pub fn copy(&mut self, to: Value, from: Value, repr: Repr) {
        let (size, parts) = match repr {
            Repr::Zst => return,
            Repr::Scalar(ty) => (ty.bytes(), Parts::One(ty)),
            Repr::Mem { size, parts, .. } => (size, parts),
        };
        if parts != Parts::Bytes {
            for (ty, at) in parts.scalars() {
                let part = self.ins().load(ty, from, at);
                self.ins().store(part, to, at);
            }
            return;
        }
        self.ins().move_value(to, from, size);
    }

    /// The place that the place expression `expr` names, or none when `expr` is a value.
    pub fn place_of(&mut self, expr: ExprId) -> Result<Option<Place>> {
        if let Some(inner) = self.transparent(expr) {
            return self.place_of(inner);
        }
        match *self.body.expr(expr) {
            Expr::Path(_) => match self.typed.res.get(&expr) {
                Some(Res::Local(local)) => Ok(self.locals[*local as usize]),
                Some(&Res::Fn(item, substs)) if super::is_static(self.cx, item) => self
                    .static_place((item, substs), self.ty_of(expr))
                    .map(Some),
                _ => Ok(None),
            },
            Expr::Unary(UnOp::Deref, operand) => self.deref_place(expr, operand),
            Expr::TupleField(base, index) => self.field_place(expr, base, index),
            Expr::Field(base, _) => match self.typed.fields.get(&expr) {
                Some(&index) => self.field_place(expr, base, index),
                None => bail!("field read has no field"),
            },
            Expr::Index(base, index) => self.index_place(expr, base, index),
            _ => Ok(None),
        }
    }

    pub(super) fn transparent(&self, expr: ExprId) -> Option<ExprId> {
        transparent_in(self.body, expr)
    }

    fn coerced_repr(&mut self, expr: ExprId, coerced: Option<&Coercion>) -> Result<Repr> {
        Ok(match coerced {
            Some(Coercion::Field { target, .. }) => repr(self.cx, *target)?,
            Some(coercion) if coercion.widens() => super::ops::WIDE,
            Some(Coercion::Deref(_, to)) if self.cx.is_unsized(*to)? => super::ops::WIDE,
            Some(Coercion::Deref(..) | Coercion::Reify(_)) => Repr::Scalar(Type::I64),
            _ => self.repr_of(expr)?,
        })
    }

    /// The place of `expr`, holding its value in a new frame slot when it is not a place already.
    pub fn place_or_temp(&mut self, expr: ExprId) -> Result<Option<Place>> {
        let coerced = self.typed.coercions.get(&expr);
        if coerced.is_none() {
            if let Some(place) = self.place_of(expr)? {
                return Ok(Some(place));
            }
            if self.is_uncaptured_local(expr) && self.repr_of(expr)? == Repr::Zst {
                return Ok(None);
            }
        }
        self.temp(expr)
    }

    fn is_uncaptured_local(&self, mut expr: ExprId) -> bool {
        while let Some(inner) = self.transparent(expr) {
            expr = inner;
        }
        matches!(self.body.expr(expr), Expr::Path(_))
            && matches!(self.typed.res.get(&expr), Some(&Res::Local(local)) if self.locals[local as usize].is_none())
    }

    /// The value of `expr` in a new temporary that the statement owns.
    pub fn temp(&mut self, expr: ExprId) -> Result<Option<Place>> {
        let coerced = self.typed.coercions.get(&expr);
        let repr = self.coerced_repr(expr, coerced)?;
        let value = self.expr(expr)?;
        let copy_mem = matches!(self.cx.types.kind(self.ty_of(expr)), Kind::Ref(..));
        let expr_ty = self.ty_of(expr);
        let place = match (repr, value) {
            (Repr::Mem { .. }, Some(addr)) if !copy_mem => {
                if let Some((slot, offset)) = self.b.func.stack_addr_of(addr) {
                    Some(Place {
                        base: Base::Slot(slot),
                        offset,
                        repr,
                    })
                } else if crate::sema::infer::needs_drop(self.cx, expr_ty)? {
                    self.slot(repr).map(|slot| {
                        let place = Place::slot(slot, repr);
                        self.write(place, addr);
                        place
                    })
                } else {
                    Some(Place {
                        base: Base::Addr(addr),
                        offset: 0,
                        repr,
                    })
                }
            }
            (_, Some(value)) => self.slot(repr).map(|slot| {
                let place = Place::slot(slot, repr);
                self.write(place, value);
                place
            }),
            _ => None,
        };
        if self.open && (place.is_some() || (repr == Repr::Zst && !self.is_uncaptured_local(expr)))
        {
            self.own(super::moves::Root::Temp(expr), expr_ty, place, true)?;
        }
        Ok(place)
    }

    pub(super) fn field_offset(&mut self, tuple: Ty, index: u32) -> Result<i32> {
        let layout = self.cx.layout(tuple)?;
        Ok(layout.fields.offset(index as usize).bytes() as i32)
    }

    fn field_place(&mut self, expr: ExprId, base: ExprId, index: u32) -> Result<Option<Place>> {
        if let Some((local, path)) = self.expr_capture_path(expr)
            && let Some((_, place)) = self.capture_by_path(local, &path)
        {
            return Ok(place);
        }
        let Some((place, ty)) = self.autoderef_place(expr, base)? else {
            return Ok(None);
        };
        self.part_place(place, ty, (0, index)).map(Some)
    }

    pub(super) fn autoderef_place(
        &mut self,
        expr: ExprId,
        base: ExprId,
    ) -> Result<Option<(Place, Ty)>> {
        let Some(mut place) = self.place_or_temp(base)? else {
            return Ok(None);
        };
        let mut ty = self.ty_of(base);
        for step in 0..self.typed.autoderefs.get(&expr).copied().unwrap_or(0) {
            let Some(next) = self.deref_step((expr, step), Some(place), ty)? else {
                return Ok(None);
            };
            (place, ty) = next;
        }
        Ok(Some((place, ty)))
    }

    pub(super) fn part_place(
        &mut self,
        place: Place,
        ty: Ty,
        (variant, index): (u32, u32),
    ) -> Result<Place> {
        match *self.cx.types.kind(ty) {
            Kind::Array(element, _) => {
                let stride = self.cx.layout(element)?.size.bytes() as i32;
                Ok(place.at(stride * index as i32, repr(self.cx, element)?))
            }
            Kind::Tuple(items) => {
                let item_ty = self.cx.types.substs(items)[index as usize];
                let offset = self.field_offset(ty, index)?;
                Ok(place.at(offset, repr(self.cx, item_ty)?))
            }
            _ => self.variant_field(place, ty, (variant, index)),
        }
    }

    /// Lowers `&inner` to the address of `inner`'s place, and tells the optimizer that a reference made from a raw pointer is not null unless the borrow is `raw`.
    pub fn reference(&mut self, inner: ExprId, raw: bool) -> Result<Option<Value>> {
        let place = self.place_or_temp(inner)?;
        let ty = self.ty_of(inner);
        let address = self.address_or_dangling(place, ty)?;
        if !raw && place.is_some() && self.derefs_raw(inner) {
            let pointer = if self.cx.is_unsized(ty)? {
                self.ins().load(Type::I64, address, 0)
            } else {
                address
            };
            self.ins().assume(pointer);
        }
        Ok(Some(address))
    }

    fn derefs_raw(&self, expr: ExprId) -> bool {
        match *self.body.expr(expr) {
            Expr::Unary(UnOp::Deref, operand) => {
                self.typed.derefs.get(&(expr, 0)).is_none()
                    && matches!(self.cx.types.kind(self.ty_of(operand)), Kind::Ptr(..))
            }
            _ => self
                .transparent(expr)
                .is_some_and(|inner| self.derefs_raw(inner)),
        }
    }

    /// The address of `place`, or for a value with no bytes the alignment of `ty` as a dangling address.
    pub fn address_or_dangling(&mut self, place: Option<Place>, ty: Ty) -> Result<Value> {
        match place {
            Some(place) => Ok(self.address(place)),
            None => {
                let layout = self.cx.layout(ty)?;
                Ok(self.ins().iconst(Type::I64, layout.align.bytes() as i64))
            }
        }
    }
}

pub(super) fn transparent_in(body: &crate::body::Body, expr: ExprId) -> Option<ExprId> {
    match *body.expr(expr) {
        Expr::Paren(inner) => Some(inner),
        Expr::Block {
            stmts,
            tail,
            label: None,
            kind: BlockKind::Expansion,
        } if body.list(stmts).is_empty() => tail,
        _ => None,
    }
}

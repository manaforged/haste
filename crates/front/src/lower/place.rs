use anyhow::{Result, bail};
use haste_codegen::{StackSlot, Type, Value};

use crate::body::{Expr, ExprId, List, UnOp};
use crate::sema::infer::{Adjust, Res};
use crate::sema::ty::{Kind, Ty};

use super::{Lower, Repr, repr};

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
        Place { base: Base::Slot(slot), offset: 0, repr }
    }

    /// The place `offset` bytes further on, holding a value of `repr`.
    pub fn at(self, offset: i32, repr: Repr) -> Place {
        Place { base: self.base, offset: self.offset + offset, repr }
    }
}

impl Lower<'_, '_> {
    /// The value in `place`: a scalar loaded from it, or its address for a memory value.
    pub fn read(&mut self, place: Place) -> Option<Value> {
        match (place.repr, place.base) {
            (Repr::Zst, _) => None,
            (Repr::Scalar(ty), Base::Slot(slot)) => Some(self.ins().stack_load(ty, slot, place.offset)),
            (Repr::Scalar(ty), Base::Addr(addr)) => Some(self.ins().load(ty, addr, place.offset)),
            (Repr::Mem { .. }, _) => Some(self.address(place)),
        }
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
            (Repr::Mem { size, .. }, _) => {
                let to = self.address(place);
                self.copy(to, value, size);
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

    /// Copies `size` bytes from address `from` to address `to`.
    pub fn copy(&mut self, to: Value, from: Value, size: u32) {
        let mut at = 0;
        for (ty, step) in [(Type::I64, 8), (Type::I32, 4), (Type::I16, 2), (Type::I8, 1)] {
            while size - at >= step {
                let word = self.ins().load(ty, from, at as i32);
                self.ins().store(word, to, at as i32);
                at += step;
            }
        }
    }

    /// The place that the place expression `expr` names, or none when `expr` is a value.
    pub fn place_of(&mut self, expr: ExprId) -> Result<Option<Place>> {
        match *self.body.expr(expr) {
            Expr::Paren(inner) => self.place_of(inner),
            Expr::Path(_) => match self.typed.res.get(&expr) {
                Some(Res::Local(local)) => Ok(self.locals[*local as usize].map(|(slot, repr)| Place::slot(slot, repr))),
                _ => Ok(None),
            },
            Expr::Unary(UnOp::Deref, pointer) => {
                let repr = self.repr_of(expr)?;
                Ok(self.expr(pointer)?.map(|addr| Place { base: Base::Addr(addr), offset: 0, repr }))
            }
            Expr::TupleField(base, index) => self.field_place(expr, base, index).map(Some),
            Expr::Field(base, _) => self.field_read_place(expr, base),
            Expr::Index(base, index) => self.index_place(expr, base, index),
            _ => Ok(None),
        }
    }

    /// The place of `expr`, holding its value in a new frame slot when it is not a place already.
    pub fn place_or_temp(&mut self, expr: ExprId) -> Result<Option<Place>> {
        if let Some(place) = self.place_of(expr)? {
            return Ok(Some(place));
        }
        let repr = self.repr_of(expr)?;
        let value = self.expr(expr)?;
        Ok(match (repr, value) {
            (Repr::Mem { .. }, Some(addr)) => Some(Place { base: Base::Addr(addr), offset: 0, repr }),
            (Repr::Scalar(_), Some(value)) => {
                let slot = self.slot(repr).expect("a scalar has a slot");
                let place = Place::slot(slot, repr);
                self.write(place, value);
                Some(place)
            }
            _ => None,
        })
    }

    pub(super) fn field_offset(&mut self, tuple: Ty, index: u32) -> Result<i32> {
        let layout = self.cx.layout(tuple)?;
        Ok(layout.fields.offset(index as usize).bytes() as i32)
    }

    fn field_place(&mut self, expr: ExprId, base: ExprId, index: u32) -> Result<Place> {
        let Some(mut place) = self.place_or_temp(base)? else { bail!("field of a value with no place") };
        let mut ty = self.ty_of(base);
        while let Kind::Ref(_, inner) | Kind::Ptr(_, inner) = *self.cx.types.kind(ty) {
            let Some(pointer) = self.read(place) else { bail!("reference with no value") };
            place = Place { base: Base::Addr(pointer), offset: 0, repr: repr(self.cx, inner)? };
            ty = inner;
        }
        let offset = self.field_offset(ty, index)?;
        let field = self.repr_of(expr)?;
        Ok(place.at(offset, field))
    }

    /// Lowers the tuple expression `expr` into a new frame slot and returns its address.
    pub fn tuple(&mut self, expr: ExprId, list: List) -> Result<Option<Value>> {
        let repr = self.repr_of(expr)?;
        let ty = self.ty_of(expr);
        let slot = self.slot(repr);
        for (index, &item) in self.body.list(list).to_vec().iter().enumerate() {
            let value = self.expr(ExprId(item))?;
            let field = self.repr_of(ExprId(item))?;
            let offset = self.field_offset(ty, index as u32)?;
            if let (Some(slot), Some(value)) = (slot, value) {
                self.write(Place::slot(slot, repr).at(offset, field), value);
            }
        }
        Ok(slot.map(|slot| self.read(Place::slot(slot, repr))).flatten())
    }

    /// Lowers `&inner` to the address of `inner`'s place.
    pub fn reference(&mut self, inner: ExprId) -> Result<Option<Value>> {
        let place = self.place_or_temp(inner)?;
        self.address_or_dangling(place, self.ty_of(inner)).map(Some)
    }

    /// The address of `place`, or for a value with no bytes the alignment of `ty` as a dangling address.
    pub fn address_or_dangling(&mut self, place: Option<Place>, ty: Ty) -> Result<Value> {
        match place {
            Some(place) => Ok(self.address(place)),
            None => {
                let layout = self.cx.layout(ty)?;
                Ok(self.ins().iconst(Type::I64, layout.align.abi.bytes() as i64))
            }
        }
    }

    /// The receiver of a method call after `adjust`: read through its references, then borrowed or read.
    pub fn receiver(&mut self, receiver: ExprId, adjust: Adjust) -> Result<Option<Value>> {
        if adjust == (Adjust { derefs: 0, autoref: None }) {
            return self.expr(receiver);
        }
        let mut place = self.place_or_temp(receiver)?;
        let mut ty = self.ty_of(receiver);
        for _ in 0..adjust.derefs {
            if let Kind::Array(_, len) = *self.cx.types.kind(ty) {
                let Some(_) = adjust.autoref else { bail!("a slice receiver by value") };
                let data = self.address_or_dangling(place, ty)?;
                return Ok(Some(self.wide_pointer(data, len as i64)));
            }
            let Kind::Ref(_, inner) = *self.cx.types.kind(ty) else { bail!("receiver deref through {:?}", self.cx.types.kind(ty)) };
            let Some(pointer) = place.and_then(|place| self.read(place)) else { bail!("reference with no value") };
            place = Some(Place { base: Base::Addr(pointer), offset: 0, repr: repr(self.cx, inner)? });
            ty = inner;
        }
        match adjust.autoref {
            Some(_) => self.address_or_dangling(place, ty).map(Some),
            None => Ok(place.and_then(|place| self.read(place))),
        }
    }
}

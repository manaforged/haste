use anyhow::{Result, bail};
use haste_codegen::{StackSlot, Type, Value};

use crate::body::{BlockKind, Expr, ExprId, List, UnOp};
use crate::sema::infer::{Adjust, Coercion, Res};
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
            (Repr::Mem { .. }, _) => {
                let to = self.address(place);
                self.copy(to, value, place.repr);
            }
        }
    }

    /// The place that the dereference `expr` of `operand` names: through a pointer, through the pointer that a `Box` holds, or through the reference that `Deref::deref` returns.
    fn deref_place(&mut self, expr: ExprId, operand: ExprId) -> Result<Option<Place>> {
        let place = self.place_or_temp(operand)?;
        Ok(self.deref_step((expr, 0), place, self.ty_of(operand))?.map(|(place, _)| place))
    }

    /// The place and type that dereference step `step` of `expr` reaches from `place` of type `ty`: through the `Deref` call inference picked, or the builtin dereference.
    pub(super) fn deref_step(&mut self, (expr, step): (ExprId, u32), place: Option<Place>, ty: Ty) -> Result<Option<(Place, Ty)>> {
        match self.typed.derefs.get(&(expr, step)).copied() {
            Some(Res::Fn(item, substs)) => {
                let receiver = self.address_or_dangling(place, ty)?;
                let Some(pointer) = self.invoke(item, substs, vec![receiver])? else { return Ok(None) };
                let index = self.reached_index(item, substs)?;
                let ret = self.reached.list[index].1.as_ref().ok().and_then(|callee| callee.ret);
                let Some(Kind::Ref(_, target)) = ret.map(|ret| self.cx.types.kind(ret).clone()) else { bail!("`{}` returns no reference", self.cx.name(item)) };
                Ok(Some((self.pointee_place(pointer, target)?, target)))
            }
            Some(_) => bail!("a dereference through something that is not a function"),
            None => place.map(|place| self.step_in(place, ty)).transpose(),
        }
    }

    /// The pointer that the value `value` of the reference, raw pointer or `Box` type `ty` holds, and its pointee.
    pub(super) fn builtin_pointer(&mut self, value: Value, ty: Ty) -> Result<(Value, Ty)> {
        match self.cx.types.kind(ty).clone() {
            Kind::Ref(_, pointee) | Kind::Ptr(_, pointee) => Ok((value, pointee)),
            _ if let Some(pointee) = self.cx.boxed(ty) => {
                let pointer = if self.cx.is_unsized(pointee)? { value } else { self.box_data(value) };
                Ok((pointer, pointee))
            }
            other => bail!("dereference of {other:?}"),
        }
    }

    /// The reference that the deref coercion `steps` reaches from the reference `pointer`, one builtin dereference or `Deref::deref` call per step.
    pub(super) fn deref_coerced(&mut self, mut pointer: Value, steps: Vec<(Ty, Option<Res>)>) -> Result<Option<Value>> {
        for (ty, res) in steps {
            pointer = match res {
                Some(Res::Fn(item, substs)) => match self.invoke(item, substs, vec![pointer])? {
                    Some(next) => next,
                    None => return Ok(None),
                },
                Some(_) => bail!("a deref coercion through something that is not a function"),
                None => {
                    let place = self.pointee_place(pointer, ty)?;
                    let Some(value) = self.read(place) else { bail!("a pointer with no value") };
                    self.builtin_pointer(value, ty)?.0
                }
            };
        }
        Ok(Some(pointer))
    }

    /// The address that the value `value` of the sized pointer type `ty` holds, whether a reference, a raw pointer, or a smart pointer such as `Box` or `Rc`.
    pub(super) fn sized_pointer(&mut self, value: Value, ty: Ty) -> Value {
        match self.cx.types.kind(ty) {
            Kind::Ref(..) | Kind::Ptr(..) => value,
            _ => self.box_data(value),
        }
    }

    /// The data pointer that the value `boxed` of a `Box` of a sized type holds.
    pub(super) fn box_data(&mut self, boxed: Value) -> Value {
        self.ins().load(Type::I64, boxed, 0)
    }

    /// The place that `pointer` to `pointee` names, which for a slice or `str` is the wide pointer itself.
    pub(super) fn pointee_place(&mut self, pointer: Value, pointee: Ty) -> Result<Place> {
        let repr = if self.cx.is_unsized(pointee)? { super::ops::WIDE } else { repr(self.cx, pointee)? };
        Ok(Place { base: Base::Addr(pointer), offset: 0, repr })
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
                Some(&Res::Fn(item, substs)) if super::is_static(self.cx, item) => self.static_place((item, substs), self.ty_of(expr)).map(Some),
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

    /// The expression that `expr` only wraps: the inside of parentheses, or the tail of a macro expansion with no statements.
    pub(super) fn transparent(&self, expr: ExprId) -> Option<ExprId> {
        match *self.body.expr(expr) {
            Expr::Paren(inner) => Some(inner),
            Expr::Block { stmts, tail, label: None, kind: BlockKind::Expansion } if self.body.list(stmts).is_empty() => tail,
            _ => None,
        }
    }

    fn coerced_repr(&mut self, expr: ExprId, coerced: Option<&Coercion>) -> Result<Repr> {
        Ok(match coerced {
            Some(coercion) if coercion.widens() => super::ops::WIDE,
            Some(Coercion::Deref(_, to)) if self.cx.is_unsized(*to)? => super::ops::WIDE,
            Some(Coercion::Deref(..) | Coercion::Reify(_)) => Repr::Scalar(Type::I64),
            _ => self.repr_of(expr)?,
        })
    }

    /// The place of `expr`, holding its value in a new frame slot when it is not a place already.
    pub fn place_or_temp(&mut self, expr: ExprId) -> Result<Option<Place>> {
        let coerced = self.typed.coercions.get(&expr);
        if coerced.is_none()
            && let Some(place) = self.place_of(expr)?
        {
            return Ok(Some(place));
        }
        self.temp(expr)
    }

    /// The value of `expr` in a new temporary that the statement owns.
    pub fn temp(&mut self, expr: ExprId) -> Result<Option<Place>> {
        let coerced = self.typed.coercions.get(&expr);
        let repr = self.coerced_repr(expr, coerced)?;
        let value = self.expr(expr)?;
        let place = match (repr, value) {
            (Repr::Mem { .. }, Some(addr)) => Some(Place { base: Base::Addr(addr), offset: 0, repr }),
            (Repr::Scalar(_), Some(value)) => {
                let slot = self.slot(repr).expect("a scalar has a slot");
                let place = Place::slot(slot, repr);
                self.write(place, value);
                Some(place)
            }
            _ => None,
        };
        if self.open {
            self.own(super::moves::Root::Temp(expr), self.ty_of(expr), place, true)?;
        }
        Ok(place)
    }

    pub(super) fn field_offset(&mut self, tuple: Ty, index: u32) -> Result<i32> {
        let layout = self.cx.layout(tuple)?;
        Ok(layout.fields.offset(index as usize).bytes() as i32)
    }

    fn field_place(&mut self, expr: ExprId, base: ExprId, index: u32) -> Result<Option<Place>> {
        let Some((place, ty)) = self.autoderef_place(expr, base)? else {
            return Ok(None);
        };
        self.part_place(place, ty, (0, index)).map(Some)
    }

    /// The place of `base` and its type, after the dereferences that inference recorded on `expr`.
    pub(super) fn autoderef_place(&mut self, expr: ExprId, base: ExprId) -> Result<Option<(Place, Ty)>> {
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

    /// The place that the reference, raw pointer or `Box` of type `ty` at `place` points at, and its pointee type.
    fn step_in(&mut self, place: Place, ty: Ty) -> Result<(Place, Ty)> {
        let Some(value) = self.read(place) else { bail!("a pointer with no value") };
        let (pointer, inner) = self.builtin_pointer(value, ty)?;
        Ok((self.pointee_place(pointer, inner)?, inner))
    }

    /// The place of field `index` of variant `variant` of the tuple or ADT `ty`, or of element `index` of the array `ty`, at `place`.
    pub(super) fn part_place(&mut self, place: Place, ty: Ty, (variant, index): (u32, u32)) -> Result<Place> {
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

    /// Lowers the tuple expression `expr` into a new frame slot and returns its address.
    pub fn tuple(&mut self, expr: ExprId, list: List) -> Result<Option<Value>> {
        self.tuple_of(self.ty_of(expr), list)
    }

    /// Lowers the expressions `list` into a new frame slot as a tuple of type `ty`, and returns its value.
    pub(super) fn tuple_of(&mut self, ty: Ty, list: List) -> Result<Option<Value>> {
        let mut values = Vec::new();
        for &item in self.body.list(list).to_vec().iter() {
            values.push(self.expr(ExprId(item))?);
        }
        self.tuple_from(ty, values)
    }

    /// Lowers the compiler-made `Clone::clone` of a tuple: each field of the referenced tuple cloned through its call in `calls`, into a new tuple.
    pub fn clone_body(&mut self, calls: Vec<Res>) -> Result<Option<Value>> {
        let (Some(ty), Some(local)) = (self.typed.ret, self.typed.params.first().copied()) else { bail!("a compiler-made clone with no parameter or result") };
        let Some(source) = self.locals[local as usize].and_then(|place| self.read(place)) else { bail!("a compiler-made clone with no source") };
        let mut values = Vec::new();
        for (index, call) in calls.into_iter().enumerate() {
            let Res::Fn(item, substs) = call else { bail!("a field clone that is not a call") };
            let offset = self.field_offset(ty, index as u32)?;
            let field = self.ins().iadd_imm(source, i64::from(offset));
            values.push(self.invoke(item, substs, vec![field])?);
        }
        match crate::sema::infer::cloned_tuple(self.cx, ty) {
            Some(tuple) => self.tuple_from(tuple, values),
            None => {
                let whole = repr(self.cx, ty)?;
                Ok(self.read(Place { base: Base::Addr(source), offset: 0, repr: whole }))
            }
        }
    }

    fn tuple_from(&mut self, ty: Ty, values: Vec<Option<Value>>) -> Result<Option<Value>> {
        let whole = repr(self.cx, ty)?;
        let Kind::Tuple(items) = *self.cx.types.kind(ty) else { bail!("a tuple of a type that is not a tuple") };
        let items = self.cx.types.substs(items).to_vec();
        let slot = self.slot(whole);
        for (index, value) in values.into_iter().enumerate() {
            let field = repr(self.cx, items[index])?;
            let offset = self.field_offset(ty, index as u32)?;
            if let (Some(slot), Some(value)) = (slot, value) {
                self.write(Place::slot(slot, whole).at(offset, field), value);
            }
        }
        Ok(slot.map(|slot| self.read(Place::slot(slot, whole))).flatten())
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
                Ok(self.ins().iconst(Type::I64, layout.align.bytes() as i64))
            }
        }
    }

    /// The receiver of a method call after `adjust`: read through its references, then borrowed or read.
    pub fn receiver(&mut self, (expr, receiver): (ExprId, ExprId), adjust: Adjust) -> Result<Option<Value>> {
        if adjust == (Adjust { derefs: 0, autoref: None }) {
            return self.expr(receiver);
        }
        let mut place = self.place_or_temp(receiver)?;
        let mut ty = self.ty_of(receiver);
        for step in 0..adjust.derefs {
            if let Kind::Array(_, len) = *self.cx.types.kind(ty) {
                let Some(_) = adjust.autoref else { bail!("a slice receiver by value") };
                let data = self.address_or_dangling(place, ty)?;
                let len = self.len(len)?;
                return Ok(Some(self.wide_pointer(data, len)));
            }
            let Some((inner_place, inner)) = self.deref_step((expr, step), place, ty)? else {
                return Ok(None);
            };
            (place, ty) = (Some(inner_place), inner);
        }
        match adjust.autoref {
            Some(_) => self.address_or_dangling(place, ty).map(Some),
            None => Ok(place.and_then(|place| self.read(place))),
        }
    }
}

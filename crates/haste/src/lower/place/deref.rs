use anyhow::{Result, bail};
use haste_codegen::{Type, Value};

use crate::body::ExprId;
use crate::sema::infer::{Adjust, Coercion, Res, SiteKey, field_types};
use crate::sema::ty::{Kind, Ty};

use super::super::ops::WIDE;
use super::super::reloc::SiteOrigin;
use super::super::{Lower, Repr, repr};
use super::{Base, Place};

impl Lower<'_, '_> {
    pub(super) fn deref_place(&mut self, expr: ExprId, operand: ExprId) -> Result<Option<Place>> {
        let place = self.place_or_temp(operand)?;
        Ok(self
            .deref_step((expr, 0), place, self.ty_of(operand))?
            .map(|(place, _)| place))
    }

    pub(in super::super) fn deref_step(
        &mut self,
        (expr, step): (ExprId, u32),
        place: Option<Place>,
        ty: Ty,
    ) -> Result<Option<(Place, Ty)>> {
        match self.typed.derefs.get(&(expr, step)).copied() {
            Some(Res::Fn(item, substs)) => {
                let receiver = self.address_or_dangling(place, ty)?;
                let Some(pointer) = self.invoke(
                    SiteOrigin::Table(SiteKey::Deref(expr, step)),
                    item,
                    substs,
                    vec![receiver],
                )?
                else {
                    return Ok(None);
                };
                let index = self.reached_index(item, substs)?;
                let ret = self.reached.list[index]
                    .1
                    .as_ref()
                    .ok()
                    .and_then(|callee| callee.ret);
                let Some(Kind::Ref(_, target)) = ret.map(|ret| self.cx.types.kind(ret).clone())
                else {
                    bail!("`{}` returns no reference", self.cx.name(item));
                };
                Ok(Some((self.pointee_place(pointer, target)?, target)))
            }
            Some(_) => bail!("a dereference through something that is not a function"),
            None => place.map(|place| self.step_in(place, ty)).transpose(),
        }
    }

    pub(in super::super) fn builtin_pointer(
        &mut self,
        value: Value,
        ty: Ty,
    ) -> Result<(Value, Ty)> {
        match self.cx.types.kind(ty).clone() {
            Kind::Ref(_, pointee) | Kind::Ptr(_, pointee) => Ok((value, pointee)),
            _ if let Some(pointee) = self.cx.boxed(ty) => {
                let pointer = if self.cx.is_unsized(pointee)? {
                    value
                } else {
                    let data = self.box_data(value);
                    self.b.func.mark_unique_ptr(data);
                    data
                };
                Ok((pointer, pointee))
            }
            other => bail!("dereference of {other:?}"),
        }
    }

    pub(in super::super) fn deref_coerced(
        &mut self,
        expr: ExprId,
        mut pointer: Value,
        steps: Vec<(Ty, Option<Res>)>,
    ) -> Result<Option<Value>> {
        for (idx, (ty, res)) in steps.into_iter().enumerate() {
            pointer = match res {
                Some(Res::Fn(item, substs)) => match self.invoke(
                    SiteOrigin::Table(SiteKey::CoercionDeref(expr, idx as u16)),
                    item,
                    substs,
                    vec![pointer],
                )? {
                    Some(next) => next,
                    None => return Ok(None),
                },
                Some(_) => bail!("a deref coercion through something that is not a function"),
                None => {
                    let place = self.pointee_place(pointer, ty)?;
                    let Some(value) = self.read(place) else {
                        bail!("a pointer with no value")
                    };
                    self.builtin_pointer(value, ty)?.0
                }
            };
        }
        Ok(Some(pointer))
    }

    pub(in super::super) fn sized_pointer(&mut self, value: Value, ty: Ty) -> Value {
        let is_scalar = matches!(self.cx.types.kind(ty), Kind::Ref(..) | Kind::Ptr(..))
            || matches!(repr(self.cx, ty), Ok(Repr::Scalar(_)));
        let data = if is_scalar {
            value
        } else {
            self.box_data(value)
        };
        if self.cx.boxed(ty).is_some() {
            self.b.func.mark_unique_ptr(data);
        }
        data
    }

    pub(in super::super) fn box_data(&mut self, boxed: Value) -> Value {
        self.ins().load(Type::I64, boxed, 0)
    }

    pub(in super::super) fn pointee_place(&mut self, pointer: Value, pointee: Ty) -> Result<Place> {
        let repr = if self.cx.is_unsized(pointee)? {
            WIDE
        } else {
            repr(self.cx, pointee)?
        };
        Ok(Place {
            base: Base::Addr(pointer),
            offset: 0,
            repr,
        })
    }

    fn step_in(&mut self, place: Place, ty: Ty) -> Result<(Place, Ty)> {
        let Some(value) = self.read(place) else {
            bail!("a pointer with no value")
        };
        let (pointer, inner) = self.builtin_pointer(value, ty)?;
        Ok((self.pointee_place(pointer, inner)?, inner))
    }

    /// The receiver of a method call after `adjust`: read through its references, then borrowed or read.
    pub fn receiver(
        &mut self,
        (expr, receiver): (ExprId, ExprId),
        adjust: Adjust,
    ) -> Result<Option<Value>> {
        if adjust
            == (Adjust {
                derefs: 0,
                autoref: None,
            })
        {
            return self.eval_operand(receiver);
        }
        let mut place = self.place_or_temp(receiver)?;
        let mut ty = self.ty_of(receiver);
        for step in 0..adjust.derefs {
            if let Kind::Array(_, len) = *self.cx.types.kind(ty) {
                let Some(_) = adjust.autoref else {
                    bail!("a slice receiver by value")
                };
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
            None => Ok(place.and_then(|place| {
                self.read(place).map(|val| {
                    if matches!(place.repr, Repr::Mem { .. }) {
                        self.snapshot_repr(place.repr, val)
                    } else {
                        val
                    }
                })
            })),
        }
    }

    pub(in super::super) fn apply_leaf_coercion(
        &mut self,
        expr: ExprId,
        value: Value,
        from_ty: Ty,
        coercion: &Coercion,
    ) -> Result<Option<Value>> {
        match coercion {
            Coercion::Len(len) => {
                let data = self.sized_pointer(value, from_ty);
                let len = self.len(*len)?;
                Ok(Some(self.wide_pointer(data, len)))
            }
            Coercion::Vtable(vtable) => {
                let data = self.sized_pointer(value, from_ty);
                let table = self.vtable(expr, vtable)?;
                Ok(Some(self.wide(data, table)))
            }
            Coercion::Upcast(slot) => Ok(Some(self.upcast(value, *slot, from_ty)?)),
            _ => Ok(Some(value)),
        }
    }

    pub(in super::super) fn coerce_struct(
        &mut self,
        expr: ExprId,
        value: Value,
        from_ty: Ty,
        to_ty: Ty,
        inner: &Coercion,
    ) -> Result<Option<Value>> {
        if from_ty == to_ty {
            return Ok(Some(value));
        }
        let from_repr = repr(self.cx, from_ty)?;
        if matches!(from_repr, Repr::Scalar(Type::I64)) {
            return self.apply_leaf_coercion(expr, value, from_ty, inner);
        }
        let (Kind::Adt(item, from_s), Kind::Adt(item2, to_s)) = (
            self.cx.types.kind(from_ty).clone(),
            self.cx.types.kind(to_ty).clone(),
        ) else {
            return self.apply_leaf_coercion(expr, value, from_ty, inner);
        };
        if item != item2 {
            return self.apply_leaf_coercion(expr, value, from_ty, inner);
        }
        let from_args = self.cx.types.substs(from_s).to_vec();
        let to_args = self.cx.types.substs(to_s).to_vec();
        let from_fields = field_types(self.cx, item, &from_args)?;
        let to_fields = field_types(self.cx, item, &to_args)?;
        let (Some(ff), Some(tf)) = (from_fields.first(), to_fields.first()) else {
            return self.apply_leaf_coercion(expr, value, from_ty, inner);
        };
        let to_repr = repr(self.cx, to_ty)?;
        let Some(slot) = self.slot(to_repr) else {
            return Ok(None);
        };
        let src_place = Place {
            base: Base::Addr(value),
            offset: 0,
            repr: from_repr,
        };
        let dst_place = Place::slot(slot, to_repr);
        for (i, (&f_ty, &t_ty)) in ff.iter().zip(tf.iter()).enumerate() {
            let f_repr = repr(self.cx, f_ty)?;
            let t_repr = repr(self.cx, t_ty)?;
            if matches!(t_repr, Repr::Zst) {
                continue;
            }
            let f_off = self.field_offset(from_ty, i as u32)?;
            let t_off = self.field_offset(to_ty, i as u32)?;
            let field_val = self.read(src_place.at(f_off, f_repr));
            let out_val = if f_ty == t_ty {
                field_val
            } else if let Some(v) = field_val {
                self.coerce_struct(expr, v, f_ty, t_ty, inner)?
            } else {
                None
            };
            if let Some(v) = out_val {
                self.write(dst_place.at(t_off, t_repr), v);
            }
        }
        Ok(self.read(dst_place))
    }
}

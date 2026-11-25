use anyhow::{Result, bail};
use haste_codegen::{IntCC, StackSlot, Type, Value};

use crate::names::{ItemRef, Primitive};
use crate::sema::infer::{Fields, Res, Shape, needs_drop};
use crate::sema::substs::Substs;
use crate::sema::ty::{Kind, Mut, Ty};

use super::super::moves::BOX_STEP;
use super::super::ops::WIDE;
use super::super::place::{Base, Place};
use super::super::reloc::{SiteOrigin, TyOrigin};
use super::super::{Lower, Repr, repr};

impl Lower<'_, '_> {
    fn invoke_glue(&mut self, ty: Ty, target: Value, origin: TyOrigin) -> Result<()> {
        let Some(glue) = self.cx.lang_item("drop_glue") else {
            bail!("core has no drop glue")
        };
        let substs = self.cx.types.intern_substs(&[ty]);
        self.invoke(SiteOrigin::DropGlue(origin), glue, substs, vec![target])
            .map(drop)
    }

    /// Runs the drop glue of `ty` on the value at `place`.
    pub(in crate::lower) fn glue(
        &mut self,
        ty: Ty,
        place: Option<Place>,
        origin: TyOrigin,
    ) -> Result<()> {
        if !needs_drop(self.cx, ty)? {
            return Ok(());
        }
        let target = self.address_or_dangling(place, ty)?;
        self.invoke_glue(ty, target, origin)
    }

    /// Lowers the body of the drop glue for `ty` whose target is `target`: `Drop::drop`, then the parts.
    pub fn glue_body(&mut self, ty: Ty, target: Option<Value>) -> Result<()> {
        let Some(target) = target else { return Ok(()) };
        let glue = match &self.typed.shape {
            Shape::Glue(glue) => glue.clone(),
            _ => Default::default(),
        };
        if let Some(resume) = glue.resume {
            return self.coroutine_glue(ty, target, resume);
        }
        let drop_fn = match glue.drop {
            Some(Res::Fn(item, substs)) => Some((item, substs)),
            _ => None,
        };
        let origin = TyOrigin::Subst(0);
        if let Some(owned) = self.cx.boxed(ty) {
            self.box_contents(owned, target, origin.step(BOX_STEP.0, 0))?;
            if let Some((item, substs)) = drop_fn {
                self.invoke(SiteOrigin::ShapeGlueDrop, item, substs, vec![target])?;
            }
            return Ok(());
        }
        if let Some((item, substs)) = drop_fn {
            return self.drop_with_parts(ty, target, origin, (item, substs));
        }
        self.drop_parts(ty, target, origin)
    }

    pub(in crate::lower) fn glue_place(&mut self, ty: Ty, target: Value) -> Result<Place> {
        let repr = if self.cx.is_unsized(ty)? {
            WIDE
        } else {
            repr(self.cx, ty)?
        };
        Ok(Place {
            base: Base::Addr(target),
            offset: 0,
            repr,
        })
    }

    fn push_variant_unwind(
        &mut self,
        origin: &TyOrigin,
        variant: u32,
        fields: &[(u32, Ty, Place)],
        slots: &mut Vec<StackSlot>,
    ) -> Result<()> {
        for &(field, field_ty, part) in fields.iter().rev() {
            slots.push(self.push_unwind_origin(
                origin.clone().step(variant, field),
                field_ty,
                Some(part),
                true,
            )?);
        }
        Ok(())
    }

    fn droppable_variants(
        &mut self,
        ty: Ty,
        place: Place,
        variants: &Fields,
    ) -> Result<Vec<(u32, Vec<(u32, Ty, Place)>)>> {
        let mut per_variant = Vec::new();
        for (variant, fields) in variants.iter().enumerate() {
            let mut droppable = Vec::new();
            for (field, &field_ty) in fields.iter().enumerate() {
                if needs_drop(self.cx, field_ty)? {
                    let part = self.part_place(place, ty, (variant as u32, field as u32))?;
                    droppable.push((field as u32, field_ty, part));
                }
            }
            if !droppable.is_empty() {
                per_variant.push((variant as u32, droppable));
            }
        }
        Ok(per_variant)
    }

    fn drop_with_parts(
        &mut self,
        ty: Ty,
        target: Value,
        origin: TyOrigin,
        (item, substs): (ItemRef, Substs),
    ) -> Result<()> {
        let want_unwind = self.open && self.should_unwind() && self.body_may_unwind()?;
        let variants = if want_unwind {
            self.adt_variants(ty)?
        } else {
            None
        };
        let Some(variants) = variants else {
            self.invoke(SiteOrigin::ShapeGlueDrop, item, substs, vec![target])?;
            return self.drop_parts(ty, target, origin);
        };
        let place = self.glue_place(ty, target)?;
        let mut per_variant = self.droppable_variants(ty, place, &variants)?;
        if per_variant.is_empty() {
            self.invoke(SiteOrigin::ShapeGlueDrop, item, substs, vec![target])?;
            return Ok(());
        }
        let mut slots = Vec::new();
        if variants.len() == 1
            && let Some((_, fields)) = per_variant.pop()
        {
            self.push_variant_unwind(&origin, 0, &fields, &mut slots)?;
            self.invoke(SiteOrigin::ShapeGlueDrop, item, substs, vec![target])?;
            if !self.open {
                return Ok(());
            }
            for ((field, field_ty, part), slot) in fields.into_iter().zip(slots.into_iter().rev()) {
                self.disarm_unwind(slot, true);
                self.glue(field_ty, Some(part), origin.clone().step(0, field))?;
            }
            return Ok(());
        }
        for (variant, fields) in per_variant {
            self.when_variant(Some(place), ty, variant, &mut |lower| {
                lower.push_variant_unwind(&origin, variant, &fields, &mut slots)
            })?;
        }
        self.invoke(SiteOrigin::ShapeGlueDrop, item, substs, vec![target])?;
        if !self.open {
            return Ok(());
        }
        for slot in slots {
            self.disarm_unwind(slot, true);
        }
        self.drop_fields(ty, place, origin)
    }

    fn box_contents(&mut self, inner: Ty, target: Value, origin: TyOrigin) -> Result<()> {
        let reference = self.cx.types.intern(Kind::Ref(Mut::Mut, inner));
        let contents = match repr(self.cx, reference)? {
            Repr::Scalar(_) => self.ins().load(Type::I64, target, 0),
            _ => target,
        };
        let place = Place {
            base: Base::Addr(contents),
            offset: 0,
            repr: repr(self.cx, inner).unwrap_or(Repr::Zst),
        };
        match self.cx.types.kind(inner) {
            Kind::Slice(_) | Kind::Prim(Primitive::Str) | Kind::Dyn(..) => {
                self.glue_wide(inner, contents, origin)
            }
            _ => self.glue(inner, Some(place), origin),
        }
    }

    fn glue_wide(&mut self, ty: Ty, wide: Value, origin: TyOrigin) -> Result<()> {
        if !needs_drop(self.cx, ty)? {
            return Ok(());
        }
        self.invoke_glue(ty, wide, origin)
    }

    pub(in crate::lower) fn drop_parts(
        &mut self,
        ty: Ty,
        target: Value,
        origin: TyOrigin,
    ) -> Result<()> {
        match self.cx.types.kind(ty).clone() {
            Kind::Array(inner, count) => {
                let count = self.len(count)?;
                let count = self.ins().iconst(Type::I64, count);
                self.drop_elements(inner, target, count, origin.step(0, 0))
            }
            Kind::Slice(inner) => {
                let (data, len) = (
                    self.ins().load(Type::I64, target, 0),
                    self.ins().load(Type::I64, target, 8),
                );
                self.drop_elements(inner, data, len, origin.step(0, 0))
            }
            Kind::Tuple(_) | Kind::Adt(..) => {
                let place = self.glue_place(ty, target)?;
                self.drop_fields(ty, place, origin)
            }
            Kind::Closure(..) => self.drop_closure_parts(ty, target, origin),
            Kind::Dyn(..) => self.drop_object(target),
            _ => Ok(()),
        }
    }

    pub(in crate::lower) fn drop_object(&mut self, wide: Value) -> Result<()> {
        let data = self.ins().load(Type::I64, wide, 0);
        let vtable = self.ins().load(Type::I64, wide, 8);
        let code = self.ins().load(Type::I64, vtable, 0);
        let unit = self.cx.types.unit();
        let pointer = self.cx.types.intern(Kind::Ptr(Mut::Mut, unit));
        self.when(code, |lower| {
            lower
                .call_code(code, (vec![pointer], None), vec![data])
                .map(drop)
        })
    }

    pub(in crate::lower) fn drop_guarded_list(
        &mut self,
        origin: &TyOrigin,
        variant: u32,
        fields: &[(u32, Ty, Place)],
    ) -> Result<()> {
        if fields.is_empty() {
            return Ok(());
        }
        let want_unwind = fields.len() > 1 && self.wants_field_unwind()?;
        let mut slots = Vec::new();
        if want_unwind {
            self.push_variant_unwind(origin, variant, &fields[1..], &mut slots)?;
        }
        for (idx, &(field, field_ty, part)) in fields.iter().enumerate() {
            if !self.open {
                return Ok(());
            }
            if want_unwind && idx > 0 {
                let slot = slots[fields.len() - 1 - idx];
                self.disarm_unwind(slot, true);
            }
            self.glue(field_ty, Some(part), origin.clone().step(variant, field))?;
        }
        Ok(())
    }

    pub(in crate::lower) fn drop_fields(
        &mut self,
        ty: Ty,
        place: Place,
        origin: TyOrigin,
    ) -> Result<()> {
        let variants: Fields = match *self.cx.types.kind(ty) {
            Kind::Tuple(items) => vec![self.cx.types.substs(items).to_vec()].into(),
            Kind::Adt(..) => match self.adt_variants(ty)? {
                Some(variants) => variants,
                None => return Ok(()),
            },
            _ => return Ok(()),
        };
        let per_variant = self.droppable_variants(ty, place, &variants)?;
        if per_variant.is_empty() {
            return Ok(());
        }
        let enumerated = variants.len() > 1;
        for (variant, fields) in per_variant {
            let mut run = |lower: &mut Self| lower.drop_guarded_list(&origin, variant, &fields);
            if enumerated {
                self.when_variant(Some(place), ty, variant, &mut run)?;
            } else {
                run(self)?;
            }
        }
        Ok(())
    }

    fn drop_elements(
        &mut self,
        inner: Ty,
        data: Value,
        count: Value,
        origin: TyOrigin,
    ) -> Result<()> {
        if !needs_drop(self.cx, inner)? {
            return Ok(());
        }
        let size = self.cx.layout(inner)?.size.bytes() as i64;
        let element = repr(self.cx, inner)?;
        let want_unwind = self.wants_field_unwind()?;
        let guard = if want_unwind {
            let slice_ty = self.cx.types.intern(Kind::Slice(inner));
            let rem_slot = self.b.create_stack_slot(16, 3);
            let slice_origin = origin.clone().step(super::super::reloc::SLICE_WRAP_STEP, 0);
            let rem_flag = self.push_unwind_origin(
                slice_origin,
                slice_ty,
                Some(Place::slot(rem_slot, WIDE)),
                false,
            )?;
            Some((rem_slot, rem_flag))
        } else {
            None
        };
        let mut failed = None;
        self.each(count, |lower, index| {
            if let Some((rem_slot, rem_flag)) = guard {
                let next = lower.ins().iadd_imm(index, 1);
                let rem_len = lower.ins().isub(count, next);
                let next_off = lower.ins().imul_imm(next, size);
                let next_ptr = lower.ins().iadd(data, next_off);
                lower.ins().stack_store(next_ptr, rem_slot, 0);
                lower.ins().stack_store(rem_len, rem_slot, 8);
                let has_rem = lower.ins().icmp_imm(IntCC::NotEqual, rem_len, 0);
                lower.set_dynamic_active(rem_flag, true, has_rem);
            }
            let offset = lower.ins().imul_imm(index, size);
            let at = lower.ins().iadd(data, offset);
            if let Err(error) = lower.glue(
                inner,
                Some(Place {
                    base: Base::Addr(at),
                    offset: 0,
                    repr: element,
                }),
                origin.clone(),
            ) {
                failed = Some(error);
            }
        });
        if let Some((_, rem_flag)) = guard
            && self.open
        {
            self.disarm_unwind(rem_flag, true);
        }
        failed.map_or(Ok(()), Err)
    }
}

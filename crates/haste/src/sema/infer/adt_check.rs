use anyhow::{Result, bail};

use crate::names::ItemRef;

use super::super::Context;
use super::super::ty::{Kind, Ty};
use super::{Infer, detached, scope};

impl Infer<'_, '_> {
    fn union_field_allowed(&mut self, ty: Ty) -> Result<bool> {
        let ty = self.resolve(ty);
        if let Kind::Adt(adt, _) = *self.cx.types.kind(ty)
            && self.cx.lang(adt) == Some("manually_drop")
        {
            return Ok(true);
        }
        self.is_copy(ty)
    }
}

/// Validates that every field of the union `item` is `Copy` or `ManuallyDrop<T>` (E0740).
pub(in crate::sema) fn check_union_fields(cx: &mut Context<'_>, item: ItemRef) -> Result<()> {
    if cx.outline.is_sysroot(item.krate) || cx.foreign(item) {
        return Ok(());
    }
    let def = cx.adt_def(item)?;
    let Some(variant) = def.variants.first() else {
        return Ok(());
    };
    if variant.fields.is_empty() {
        return Ok(());
    }
    let param_count = scope::type_params(&def.generics.params).count();
    if param_count > u16::MAX as usize {
        return Ok(());
    }
    detached(cx, item, |infer| {
        let before = infer.cx.quiet_mark();
        let args: Vec<Ty> = (0..param_count as u16)
            .map(|idx| infer.cx.types.intern(Kind::Param(idx)))
            .collect();
        if param_count > 0 {
            infer.poly = Some(Box::new(super::poly::PolyState::default()));
        }
        let frame = infer.adt_frame(item, &def.generics.params, &args)?;
        infer.swap_frame(frame);
        if param_count > 0 {
            infer.init_poly_env(item)?;
        }
        for field in &variant.fields {
            let ty = infer.lower_ty(&def.body, field.ty)?;
            if infer.cx.quiet_mark() != before {
                return Ok(());
            }
            let ty = infer.resolve(ty);
            if !infer.union_field_allowed(ty)? {
                if infer.cx.quiet_mark() != before {
                    return Ok(());
                }
                let field_name = field.name.map_or("_", |tok| infer.src.text(tok));
                bail!(
                    "field `{field_name}` of union `{}` must be `Copy` or `ManuallyDrop<T>`: type `{}` does not implement `Copy`",
                    infer.cx.name(item),
                    infer.show(ty),
                );
            }
        }
        Ok(())
    })
}

/// Validates concrete field types of union `item` during layout computation (E0740).
pub(in crate::sema) fn check_union_concrete_fields(
    cx: &mut Context<'_>,
    item: ItemRef,
    fields: &[Ty],
) -> Result<()> {
    if cx.outline.is_sysroot(item.krate) || cx.foreign(item) || fields.is_empty() {
        return Ok(());
    }
    detached(cx, item, |infer| {
        let before = infer.cx.quiet_mark();
        for &ty in fields {
            let ty = infer.resolve(ty);
            if infer.has_vars(ty) || infer.has_params(ty) {
                continue;
            }
            if !infer.union_field_allowed(ty)? {
                if infer.cx.quiet_mark() != before {
                    return Ok(());
                }
                bail!(
                    "union `{}` field of type `{}` must be `Copy` or `ManuallyDrop<T>`",
                    infer.cx.name(item),
                    infer.show(ty),
                );
            }
        }
        Ok(())
    })
}

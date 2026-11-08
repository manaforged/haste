use anyhow::{Result, bail};

use crate::outline::item::ItemKind;

use super::super::ty::{Kind, Ty};
use super::Infer;
use super::select::Picked;

impl Infer<'_, '_> {
    /// Whether dropping a value of `ty` runs code: a `Drop` impl of its own or of a part.
    pub fn needs_drop(&mut self, ty: Ty) -> Result<bool> {
        let ty = self.resolve(ty);
        if let Some(&known) = self.cx.drops.get(&ty) {
            return Ok(known);
        }
        self.cx.drops.insert(ty, false);
        let needs = self.parts_need_drop(ty)?;
        self.cx.drops.insert(ty, needs);
        Ok(needs)
    }

    /// The `Drop` impl of `ty` itself.
    pub fn drop_impl(&mut self, ty: Ty) -> Result<Option<Picked>> {
        let Some(drop) = self.cx.lang_item("drop") else { bail!("core has no `Drop` trait") };
        self.select(drop, ty, &[])
    }

    fn parts_need_drop(&mut self, ty: Ty) -> Result<bool> {
        match self.cx.types.kind(ty).clone() {
            Kind::Adt(item, args) => {
                if self.drop_impl(ty)?.is_some() {
                    return Ok(true);
                }
                if self.cx.lang(item) == Some("manually_drop") || self.cx.item(item).1.kind == ItemKind::Union {
                    return Ok(false);
                }
                let fields = self.adt_fields(item, &args)?;
                self.any_needs_drop(fields.into_iter().flatten())
            }
            Kind::Tuple(items) => self.any_needs_drop(items.into_iter()),
            Kind::Array(inner, len) => Ok(len > 0 && self.needs_drop(inner)?),
            Kind::Slice(inner) => self.needs_drop(inner),
            _ => Ok(false),
        }
    }

    fn any_needs_drop(&mut self, mut tys: impl Iterator<Item = Ty>) -> Result<bool> {
        tys.try_fold(false, |found, ty| Ok(found || self.needs_drop(ty)?))
    }
}

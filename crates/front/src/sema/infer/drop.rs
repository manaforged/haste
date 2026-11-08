use anyhow::{Result, bail};

use crate::names::ItemRef;
use crate::outline::item::ItemKind;

use super::super::Context;
use super::super::ty::{Kind, Mut, Ty};
use super::select::Picked;
use super::{Infer, Local, Res, Typed, detached};

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
            Kind::Array(_, 0) => return Ok(false),
            Kind::Adt(..) if self.drop_impl(ty)?.is_some() => return Ok(true),
            _ => {}
        }
        let parts = self.drop_parts(ty)?;
        self.any_needs_drop(parts.into_iter())
    }

    fn any_needs_drop(&mut self, mut tys: impl Iterator<Item = Ty>) -> Result<bool> {
        tys.try_fold(false, |found, ty| Ok(found || self.needs_drop(ty)?))
    }
}

/// The calls that the drop glue of one type makes: its own `Drop::drop`, then the glue of each part that needs drop.
#[derive(Clone, Debug, Default)]
pub struct Glue {
    pub drop: Option<Res>,
    pub parts: Vec<Res>,
}

/// The typed form of the drop glue `drop_glue::<ty>`: one `&mut ty` parameter, no body, and the calls the glue makes.
pub fn glue_typed(cx: &mut Context<'_>, item: ItemRef, ty: Ty) -> Result<Typed> {
    let target = cx.types.intern(Kind::Ref(Mut::Mut, ty));
    let unit = cx.types.unit();
    let glue = detached(cx, item, |infer| infer.glue(item, ty))?;
    Ok(Typed { locals: vec![Local { ty: target, mutable: false }], params: vec![0], ret: Some(unit), glue: Some(glue), ..Typed::default() })
}

/// The drop glue calls for every type that a local or an expression of `typed` holds and that needs drop.
pub fn drop_calls(cx: &mut Context<'_>, typed: &Typed) -> Result<Vec<Res>> {
    let Some(glue) = cx.lang_item("drop_glue") else { return Ok(Vec::new()) };
    let mut seen = rustc_hash::FxHashSet::default();
    let mut calls = Vec::new();
    for ty in typed.locals.iter().map(|local| local.ty).chain(typed.exprs.iter().copied()) {
        if ty.0 != u32::MAX && seen.insert(ty) && super::needs_drop(cx, ty)? {
            calls.push(Res::Fn(glue, vec![ty]));
        }
    }
    Ok(calls)
}

impl Infer<'_, '_> {
    fn glue(&mut self, glue: ItemRef, ty: Ty) -> Result<Glue> {
        let drop = match self.drop_impl(ty)? {
            Some((block, args)) => {
                let args = args.into_iter().map(|arg| self.zonk(arg)).collect();
                self.cx.member(block, "drop").map(|member| Res::Fn(member, args))
            }
            None => None,
        };
        let mut parts = Vec::new();
        for part in self.drop_parts(ty)? {
            if self.needs_drop(part)? && !parts.contains(&Res::Fn(glue, vec![part])) {
                parts.push(Res::Fn(glue, vec![part]));
            }
        }
        Ok(Glue { drop, parts })
    }

    /// The parts of `ty` that its drop glue drops after its own `Drop::drop`: fields, elements, or the value a `Box` owns.
    pub fn drop_parts(&mut self, ty: Ty) -> Result<Vec<Ty>> {
        Ok(match self.cx.types.kind(ty).clone() {
            Kind::Adt(item, args) if self.cx.lang(item) == Some("owned_box") => vec![args[0]],
            Kind::Adt(item, _) if self.cx.lang(item) == Some("manually_drop") || self.cx.item(item).1.kind == ItemKind::Union => Vec::new(),
            Kind::Adt(item, args) => self.adt_fields(item, &args)?.into_iter().flatten().collect(),
            Kind::Tuple(items) => items,
            Kind::Array(inner, _) | Kind::Slice(inner) => vec![inner],
            _ => Vec::new(),
        })
    }
}

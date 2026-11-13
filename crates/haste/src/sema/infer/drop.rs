use anyhow::{Result, bail};

use crate::names::ItemRef;
use crate::outline::item::ItemKind;

use super::super::Context;
use super::super::ty::{Kind, Mut, Ty};
use super::select::Picked;
use super::{Infer, Local, Res, Shape, Typed, detached};

impl Infer<'_, '_> {
    /// Whether dropping a value of `ty` runs code: a `Drop` impl of its own or of a part.
    pub fn needs_drop(&mut self, ty: Ty) -> Result<bool> {
        let ty = self.resolve(ty);
        if self.has_vars(ty) || self.has_params(ty) {
            return self.parts_need_drop(ty);
        }
        let ty = self.substitute(ty);
        if let Some(known) = self.cx.store.lookups.drops.get(&ty) {
            return Ok(known);
        }
        if self.cx.drop_stack.contains(&ty) {
            return Ok(false);
        }
        let before = self.cx.quiet_mark();
        self.cx.drop_stack.push(ty);
        let result = self.parts_need_drop(ty);
        self.cx.drop_stack.pop();
        let needs = result?;
        if self.cx.quiet_mark() == before && !self.has_vars(ty) {
            self.cx.store.lookups.drops.insert(ty, needs);
        }
        Ok(needs)
    }

    /// Whether `ty` is `Copy`: every part of a tuple, array or closure is, and a struct or enum has a `Copy` impl.
    pub fn is_copy(&mut self, ty: Ty) -> Result<bool> {
        let ty = self.resolve(ty);
        Ok(match self.cx.types.kind(ty).clone() {
            Kind::Ref(Mut::Mut, _) | Kind::Slice(_) | Kind::Dyn(..) | Kind::Prim(crate::names::Primitive::Str) => false,
            Kind::Tuple(items) => {
                let items = self.cx.types.substs(items).to_vec();
                self.all_copy(items.into_iter())?
            }
            Kind::Array(inner, _) => self.is_copy(inner)?,
            Kind::Closure(..) => self.closure_is_copy(ty)?,
            Kind::Adt(..) => {
                let Some(copy) = self.cx.lang_item("copy") else { bail!("core has no `Copy` trait") };
                self.select(copy, ty, &[])?.is_some()
            }
            Kind::Param(_) | Kind::Proj(_) => self.poly_is_copy(ty)?,
            _ => true,
        })
    }

    fn closure_is_copy(&mut self, ty: Ty) -> Result<bool> {
        if self.coroutine_output(ty).is_some() {
            return Ok(false);
        }
        match self.cx.ensure_closure(ty).map(|info| info.env) {
            Some(env) => self.is_copy(env),
            None => Ok(true),
        }
    }

    fn all_copy(&mut self, mut tys: impl Iterator<Item = Ty>) -> Result<bool> {
        tys.try_fold(true, |all, ty| Ok(all && self.is_copy(ty)?))
    }

    /// The `Drop` impl of `ty` itself.
    pub fn drop_impl(&mut self, ty: Ty) -> Result<Option<Picked>> {
        let Some(drop) = self.cx.lang_item("drop") else { bail!("core has no `Drop` trait") };
        self.select(drop, ty, &[])
    }

    fn parts_need_drop(&mut self, ty: Ty) -> Result<bool> {
        match self.cx.types.kind(ty).clone() {
            Kind::Array(_, len) if self.cx.types.len(len) == Some(0) => return Ok(false),
            Kind::Adt(..) if self.drop_impl(ty)?.is_some() => return Ok(true),
            Kind::Dyn(..) => return Ok(true),
            Kind::Closure(..) if self.coroutine_output(ty).is_some() => return Ok(true),
            Kind::Param(_) | Kind::Proj(_) => return Ok(!self.is_copy(ty)?),
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
    /// The `Future::poll` that a suspended coroutine runs in its drop mode.
    pub resume: Option<Res>,
}

/// The typed form of the drop glue `drop_glue::<ty>`: one `&mut ty` parameter, no body, and the calls the glue makes.
pub fn glue_typed(cx: &mut Context<'_>, item: ItemRef, ty: Ty) -> Result<Typed> {
    let target = cx.types.intern(Kind::Ref(Mut::Mut, ty));
    let unit = cx.types.unit();
    let glue = detached(cx, item, |infer| infer.glue(item, ty))?;
    Ok(Typed { locals: vec![Local { ty: target }], params: vec![0], ret: Some(unit), shape: Shape::Glue(glue), ..Typed::default() })
}

/// The call of the drop glue of `ty`, when dropping a value of `ty` runs code.
pub fn glue_call(cx: &mut Context<'_>, ty: Ty) -> Result<Option<Res>> {
    if let Some(cached) = cx.store.lookups.glues.get(&ty) {
        return Ok(cached);
    }
    let before = cx.quiet_mark();
    let call = if needs_drop(cx, ty)? {
        let Some(glue) = cx.lang_item("drop_glue") else { bail!("core has no drop glue") };
        Some(Res::fn_of(&cx.types, glue, &[ty]))
    } else {
        None
    };
    if cx.quiet_mark() == before && !cx.types.has_params(ty) {
        cx.store.lookups.glues.insert(ty, call);
    }
    Ok(call)
}

/// The drop glue calls for every type that a local or an expression of `typed` holds and that needs drop.
pub fn drop_calls(cx: &mut Context<'_>, typed: &Typed) -> Result<Vec<Res>> {
    let scoped = typed.scoped(&cx.types);
    let tables = scoped.ensure_tables(&cx.types);
    let mut seen = rustc_hash::FxHashSet::default();
    let mut calls = Vec::new();
    for ty in typed.locals.iter().map(|local| local.ty).chain(tables.owned_tys()) {
        if ty != Ty::NONE && seen.insert(ty) {
            calls.extend(glue_call(cx, ty)?);
        }
    }
    Ok(calls)
}

impl Infer<'_, '_> {
    /// The `Drop::drop` of `ty`'s own `Drop` impl, without the drops of its parts.
    pub(in crate::sema) fn own_drop(&mut self, ty: Ty) -> Result<Option<Res>> {
        let Some((block, args)) = self.drop_impl(ty)? else { return Ok(None) };
        let args: Vec<Ty> = args.into_iter().map(|arg| self.zonk(arg)).collect();
        let substs = self.cx.types.intern_substs(&args);
        Ok(self.cx.member(block, "drop").map(|member| Res::Fn(member, substs)))
    }

    fn glue(&mut self, glue: ItemRef, ty: Ty) -> Result<Glue> {
        let drop = self.own_drop(ty)?;
        let mut parts = Vec::new();
        for part in self.drop_parts(ty)? {
            let call = Res::fn_of(&self.cx.types, glue, &[part]);
            if self.needs_drop(part)? && !parts.contains(&call) {
                parts.push(call);
            }
        }
        let resume = match self.coroutine_output(ty) {
            Some(_) => self.cx.lang_item("poll").map(|poll| Res::fn_of(&self.cx.types, poll, &[ty])),
            None => None,
        };
        Ok(Glue { drop, parts, resume })
    }

    /// The parts of `ty` that its drop glue drops after its own `Drop::drop`: fields, elements, or the value a `Box` owns.
    pub fn drop_parts(&mut self, ty: Ty) -> Result<Vec<Ty>> {
        Ok(match *self.cx.types.kind(ty) {
            _ if let Some(owned) = self.cx.boxed(ty) => vec![owned],
            Kind::Adt(item, _) if self.cx.lang(item) == Some("manually_drop") || self.cx.item(item).1.kind == ItemKind::Union => Vec::new(),
            Kind::Adt(item, args) => {
                let args = self.cx.types.substs(args).to_vec();
                self.adt_fields(item, &args)?.iter().flatten().copied().collect()
            }
            Kind::Tuple(items) => self.cx.types.substs(items).to_vec(),
            Kind::Closure(..) => self.closure_drop_parts(ty)?,
            Kind::Array(inner, _) | Kind::Slice(inner) => vec![inner],
            _ => Vec::new(),
        })
    }

    fn closure_drop_parts(&mut self, ty: Ty) -> Result<Vec<Ty>> {
        let Some(&Kind::Tuple(items)) = self.cx.ensure_closure(ty).map(|info| self.cx.types.kind(info.env)) else {
            bail!("drop of a closure whose defining body was not inferred");
        };
        if self.coroutine_output(ty).is_none() {
            return Ok(self.cx.types.substs(items).to_vec());
        }
        let Some(&Kind::Tuple(captures)) = self.cx.types.substs(items).last().map(|&env| self.cx.types.kind(env)) else {
            bail!("a coroutine with no captures tuple");
        };
        Ok(self.cx.types.substs(captures).to_vec())
    }
}

fn with_drop<T>(cx: &mut Context<'_>, run: impl FnOnce(&mut Infer<'_, '_>) -> Result<T>) -> Result<T> {
    let Some(anchor) = cx.lang_item("drop") else { bail!("core has no `Drop` trait") };
    detached(cx, anchor, run)
}

/// Whether dropping a value of `ty` runs code.
pub fn needs_drop(cx: &mut Context<'_>, ty: Ty) -> Result<bool> {
    if ty == Ty::NONE {
        return Ok(false);
    }
    match cx.types.kind(ty) {
        Kind::Prim(_) | Kind::Ref(..) | Kind::Ptr(..) | Kind::FnDef(..) | Kind::FnPtr(..) | Kind::Never | Kind::Value(_) => return Ok(false),
        Kind::Dyn(..) => return Ok(true),
        _ => {}
    }
    if !cx.types.has_var_flag(ty)
        && !cx.types.has_params(ty)
        && let Some(known) = cx.store.lookups.drops.get(&ty)
    {
        return Ok(known);
    }
    with_drop(cx, |infer| infer.needs_drop(ty))
}

/// Whether `ty` has a `Drop` impl of its own, which forbids moving out of its parts.
pub fn has_drop_impl(cx: &mut Context<'_>, ty: Ty) -> Result<bool> {
    if ty == Ty::NONE {
        return Ok(false);
    }
    if !matches!(cx.types.kind(ty), Kind::Adt(..) | Kind::Param(_) | Kind::Proj(_) | Kind::Var(_)) {
        return Ok(false);
    }
    with_drop(cx, |infer| infer.drop_impl(ty).map(|found| found.is_some()))
}

/// The `Drop::drop` of `ty`'s own `Drop` impl, which for a `Box` frees the allocation without dropping what it owns.
pub fn drop_fn(cx: &mut Context<'_>, ty: Ty) -> Result<Option<Res>> {
    with_drop(cx, |infer| infer.own_drop(ty))
}

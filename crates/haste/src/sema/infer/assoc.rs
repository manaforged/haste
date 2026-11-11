use anyhow::{Result, bail};

use crate::body::{Body, PathId};
use crate::names::path::Path as NamePath;
use crate::names::{ItemRef, plain};
use crate::outline::item::ItemKind;

use super::super::substs::Substs;
use super::super::ty::{Kind, Ty};
use super::scope::type_params;
use super::{Infer, Res};

impl Infer<'_, '_> {
    /// Resolves a path to an associated function or constant: `Type::name`, `Trait::name`, `<T>::name` or `<T as Trait>::name`.
    pub(super) fn assoc_value(&mut self, body: &Body, path: PathId) -> Result<Option<(Ty, Res)>> {
        let entry = body.path(path);
        let Some((&last, prefix)) = body.list(entry.segments).split_last() else { return Ok(None) };
        let name = plain(self.src.text(body.segments[last as usize].name));
        let given = self.segment_args(body, path, prefix.len() + 1)?;
        let owner = match entry.qself {
            Some((ty, Some(trait_path))) if prefix.is_empty() => {
                let self_ty = self.lower_ty(body, ty)?;
                let Some(of) = self.trait_named(body, trait_path, self.home) else { bail!("`as` names something that is not a trait") };
                let args = self.bound_args(of, body, trait_path, self_ty)?;
                return self.trait_value(of, (Some(self_ty), Some(args)), (name, given)).map(Some);
            }
            Some((ty, None)) if prefix.is_empty() => self.lower_ty(body, ty)?,
            Some((ty, Some(trait_path))) if prefix.len() == 1 => self.projection(body, (ty, trait_path), prefix[0])?,
            None if !prefix.is_empty() => return self.unqualified_value(body, (path, prefix), (name, given)),
            _ => return Ok(None),
        };
        self.type_value(owner, (name, given)).map(Some)
    }

    /// Resolves `T::name`, `Trait::name` or `Type::name` with no `<..>` qualifier, its owner named by the segments `prefix` of `path`.
    fn unqualified_value(&mut self, body: &Body, (path, prefix): (PathId, &[u32]), (name, given): (&str, Vec<Ty>)) -> Result<Option<(Ty, Res)>> {
        if let [first] = prefix
            && let Some(found) = self.param_value(plain(self.src.text(body.segments[*first as usize].name)), (name, given.clone()))?
        {
            return Ok(Some(found));
        }
        if let Some(of) = self.prefix_trait(body, path, prefix.len()) {
            return self.trait_value(of, (None, None), (name, given)).map(Some);
        }
        let Ok(owner) = self.named_ty(body, path, prefix.len(), true) else { return Ok(None) };
        self.type_value(owner, (name, given)).map(Some)
    }

    fn type_value(&mut self, owner: Ty, (name, given): (&str, Vec<Ty>)) -> Result<(Ty, Res)> {
        if let Some((member, args)) = self.inherent_member(owner, name)? {
            return self.member_value(member, args, given);
        }
        if let Kind::Dyn(of, args, members, _) = self.cx.types.kind(owner).clone()
            && self.cx.trait_member(of, name).is_some()
        {
            let args = super::super::ty::object_parts(self.cx.types.substs(args), &members).0.to_vec();
            return self.trait_value(of, (Some(owner), Some(args)), (name, given));
        }
        let current = self.current_trait().into_iter().collect();
        let current = self.implied_traits(current)?;
        let scope = current.into_iter().chain(self.cx.traits_in_scope(self.home)).chain(self.body_traits());
        let traits: Vec<ItemRef> = scope.filter(|&of| self.cx.trait_member(of, name).is_some()).collect();
        for &of in &traits {
            if let Some(args) = self.own_trait_args(of, owner)? {
                return self.trait_value(of, (Some(owner), Some(args)), (name, given));
            }
            if self.implements(of, owner)? {
                return self.trait_value(of, (Some(owner), None), (name, given));
            }
        }
        match traits[..] {
            [of] if matches!(self.cx.types.kind(self.resolve(owner)), Kind::Var(_)) => self.trait_value(of, (Some(owner), None), (name, given)),
            _ => bail!("no associated item `{name}` on {}", self.show(owner)),
        }
    }

    /// The arguments of trait `of` when this body is a default method of `of` and `owner` is its `Self`, which `Self::NAME` in that body reads at.
    fn own_trait_args(&mut self, of: ItemRef, owner: Ty) -> Result<Option<Vec<Ty>>> {
        let Some((item, substs)) = self.instance else { return Ok(None) };
        let first = self.cx.types.substs(substs).first().copied();
        let default = self.block.is_none() && self.cx.owner_trait(item) == Some(of);
        if !default || first.map(|ty| self.resolve(ty)) != Some(self.resolve(owner)) {
            return Ok(None);
        }
        let count = type_params(&self.cx.trait_generics(of)?.1.params).count();
        Ok(self.cx.types.substs(substs).get(1..1 + count).map(<[Ty]>::to_vec))
    }

    fn prefix_trait(&mut self, body: &Body, path: PathId, count: usize) -> Option<ItemRef> {
        let (mut segments, absolute, dollar) = self.segments(body, path);
        segments.truncate(count);
        self.trait_at(&NamePath { segments: &segments, absolute, dollar }, self.home)
    }

    fn implements(&mut self, of: ItemRef, ty: Ty) -> Result<bool> {
        self.probe(|infer| {
            let args = infer.fresh_trait_args(of)?;
            infer.may_implement(of, ty, &args)
        })
    }

    /// A fresh variable for each type parameter of trait `of`.
    pub(super) fn fresh_trait_args(&mut self, of: ItemRef) -> Result<Vec<Ty>> {
        let count = type_params(&self.cx.trait_generics(of)?.1.params).count();
        Ok((0..count).map(|_| self.fresh(Kind::Var)).collect())
    }

    fn param_value(&mut self, param: &str, (name, given): (&str, Vec<Ty>)) -> Result<Option<(Ty, Res)>> {
        let Some(ty) = self.param_ty(param).filter(|_| param != "Self") else { return Ok(None) };
        if let Some((of, args)) = self.bound_trait((param, ty), name, Self::has_member)? {
            return self.trait_value(of, (Some(ty), Some(args)), (name, given)).map(Some);
        }
        let Some((of, _)) = self.bound_trait((param, ty), name, Self::reaches_member)? else { return Ok(None) };
        let Some(owner) = self.implied_traits(vec![of])?.into_iter().find(|&owner| self.cx.trait_member(owner, name).is_some()) else { return Ok(None) };
        self.trait_value(owner, (Some(ty), None), (name, given)).map(Some)
    }

    fn trait_value(&mut self, of: ItemRef, (self_ty, args): (Option<Ty>, Option<Vec<Ty>>), (name, given): (&str, Vec<Ty>)) -> Result<(Ty, Res)> {
        let Some(member) = self.cx.trait_member(of, name) else { bail!("trait `{}` has no item `{name}`", self.cx.name(of)) };
        let self_ty = self_ty.unwrap_or_else(|| self.fresh(Kind::Var));
        let mut outer = vec![self_ty];
        match args {
            Some(args) => outer.extend(args),
            None => outer.extend(self.fresh_trait_args(of)?),
        }
        self.obligations.push((of, outer.clone()));
        self.member_value(member, outer, given)
    }

    fn member_value(&mut self, member: ItemRef, outer: Vec<Ty>, given: Vec<Ty>) -> Result<(Ty, Res)> {
        match self.cx.item(member).1.kind {
            ItemKind::Fn => self.fn_item(member, outer, given),
            ItemKind::Const => self.const_item(member, outer),
            other => bail!("associated {other:?} values are not supported yet"),
        }
    }

    /// The impl member that a call of trait item `item` at `substs` runs, once `Self` is known, or the trait item itself when `Self` is a trait object that dispatches it.
    pub(super) fn trait_call(&mut self, item: ItemRef, substs: Substs) -> Result<Res> {
        let Some(of) = self.cx.owner_trait(item) else { return Ok(Res::Fn(item, substs)) };
        let slice = self.cx.types.substs(substs).to_vec();
        let count = 1 + type_params(&self.cx.trait_generics(of)?.1.params).count();
        if slice.len() < count || self.object_holds(of, slice[0], &slice[1..count]) == Some(true) {
            return Ok(Res::Fn(item, substs));
        }
        let Some(selected) = self.select(of, slice[0], &slice[1..count])? else { return Ok(Res::Fn(item, substs)) };
        Ok(match self.impl_member(of, (slice[0], &slice[1..count]), selected, self.cx.name(item))? {
            Some((member, (_, mut args))) => {
                args.extend(slice[count..].iter().map(|ty| self.zonk(*ty)));
                let zonked: Vec<Ty> = args.into_iter().map(|ty| self.zonk(ty)).collect();
                Res::fn_of(&self.cx.types, member, &zonked)
            }
            None => Res::Fn(item, substs),
        })
    }
}

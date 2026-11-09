use anyhow::{Result, bail};

use crate::body::{Body, PathId};
use crate::names::path::Path as NamePath;
use crate::names::{Def, ItemRef, plain};
use crate::outline::item::ItemKind;

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
            None if !prefix.is_empty() => {
                if let Some(of) = self.prefix_trait(body, path, prefix.len()) {
                    return self.trait_value(of, (None, None), (name, given)).map(Some);
                }
                if let [first] = prefix
                    && let Some(found) = self.param_value(plain(self.src.text(body.segments[*first as usize].name)), (name, given.clone()))?
                {
                    return Ok(Some(found));
                }
                let Ok(ty) = self.named_ty(body, path, prefix.len(), true) else { return Ok(None) };
                ty
            }
            _ => return Ok(None),
        };
        self.type_value(owner, (name, given)).map(Some)
    }

    fn type_value(&mut self, owner: Ty, (name, given): (&str, Vec<Ty>)) -> Result<(Ty, Res)> {
        if let Some((member, args)) = self.inherent_member(owner, name)? {
            return self.member_value(member, args, given);
        }
        if let Kind::Dyn(of, args, members) = self.cx.types.kind(owner).clone()
            && self.cx.trait_member(of, name).is_some()
        {
            let args = super::super::ty::object_parts(&args, &members).0.to_vec();
            return self.trait_value(of, (Some(owner), Some(args)), (name, given));
        }
        for of in self.cx.traits_in_scope(self.home).into_iter().chain(self.body_traits()) {
            if self.cx.trait_member(of, name).is_some() && self.implements(of, owner)? {
                return self.trait_value(of, (Some(owner), None), (name, given));
            }
        }
        bail!("no associated item `{name}` on {}", self.show(owner))
    }

    fn prefix_trait(&mut self, body: &Body, path: PathId, count: usize) -> Option<ItemRef> {
        let (mut segments, absolute, dollar) = self.segments(body, path);
        segments.truncate(count);
        match self.cx.paths(self.home.krate).resolve(&NamePath { segments: &segments, absolute, dollar }, self.home)? {
            Def::Item(item) if self.cx.item(item).1.kind == ItemKind::Trait => Some(item),
            _ => None,
        }
    }

    fn implements(&mut self, of: ItemRef, ty: Ty) -> Result<bool> {
        let saved = self.vars.clone();
        let args = self.fresh_trait_args(of)?;
        let found = self.may_implement(of, ty, &args)?;
        self.vars = saved;
        Ok(found)
    }

    fn fresh_trait_args(&mut self, of: ItemRef) -> Result<Vec<Ty>> {
        let count = type_params(&self.cx.trait_generics(of)?.1.params).count();
        Ok((0..count).map(|_| self.fresh(Kind::Var)).collect())
    }

    fn param_value(&mut self, param: &str, (name, given): (&str, Vec<Ty>)) -> Result<Option<(Ty, Res)>> {
        let Some(ty) = self.param_ty(param).filter(|_| param != "Self") else { return Ok(None) };
        let Some((of, args)) = self.bound_trait((param, ty), name, |infer, of, name| infer.cx.trait_member(of, name).is_some())? else { return Ok(None) };
        self.trait_value(of, (Some(ty), Some(args)), (name, given)).map(Some)
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

    /// The impl member that a call of trait item `item` at `substs` runs, once `Self` is known, or the trait's own default.
    pub(super) fn trait_call(&mut self, item: ItemRef, substs: Vec<Ty>) -> Result<Res> {
        let Some(of) = self.cx.owner_trait(item) else { return Ok(Res::Fn(item, substs)) };
        let count = 1 + type_params(&self.cx.trait_generics(of)?.1.params).count();
        if substs.len() < count {
            return Ok(Res::Fn(item, substs));
        }
        let Some((block, mut args)) = self.select(of, substs[0], &substs[1..count])? else { return Ok(Res::Fn(item, substs)) };
        Ok(match self.cx.member(block, self.cx.name(item)) {
            Some(member) => {
                args.extend(substs[count..].iter().map(|ty| self.zonk(*ty)));
                Res::Fn(member, args.into_iter().map(|ty| self.zonk(ty)).collect())
            }
            None => Res::Fn(item, substs),
        })
    }
}

mod bounds;

use anyhow::{Result, bail};

use crate::body::{Body, PathId};
use crate::names::{ItemRef, Primitive};
use crate::outline::item::ItemKind;

use super::super::impls::ImplRef;
use super::super::substs::Substs;
use super::super::ty::{Kind, Ty};
use super::scope::type_params;
use super::unify::{CacheState, replay_obligations};
use super::{Infer, LocalProjectionKey};

pub(super) type Projection = (ItemRef, Ty, Vec<Ty>, String, Vec<Ty>, Ty);
type ProjectionKey = (ItemRef, Ty, Substs, crate::names::Sym, Substs);

const SUPER_DEPTH: u32 = 8;

impl Infer<'_, '_> {
    /// The type that `param::name` names, when `param` is `Self` or a type parameter of the frame and a trait that bounds it has the associated type `name`.
    pub(super) fn param_projection(
        &mut self,
        param: &str,
        (name, own): (&str, &[Ty]),
    ) -> Result<Option<Ty>> {
        if param == "Self"
            && let Some(block) = self.block
            && let Some(ty) = self.impl_self_alias(block, (name, own))?
        {
            return Ok(Some(ty));
        }
        let Some(ty) = self.param_ty(param) else {
            return Ok(None);
        };
        let owner_trait = self.frame_item.and_then(|item| {
            (self.cx.item(item).1.kind == ItemKind::Trait)
                .then_some(item)
                .or_else(|| self.cx.owner_trait(item))
        });
        if param == "Self"
            && let Some(of) = owner_trait
            && self.reaches_assoc_ty(of, name)?
        {
            let count = type_params(&self.cx.trait_generics(of)?.1.params).count();
            let args: Vec<Ty> = self
                .generics
                .iter()
                .take(count)
                .map(|&(_, ty)| ty)
                .collect();
            if let Some((owner, args)) = self.assoc_owner((of, args), ty, name, 0)? {
                return self.project_own(owner, ty, args, (name, own)).map(Some);
            }
        }
        let Some(bound) = self.bound_trait((param, ty), name, Self::reaches_assoc_ty)? else {
            return Ok(None);
        };
        match self.assoc_owner(bound, ty, name, 0)? {
            Some((of, args)) => self.project_own(of, ty, args, (name, own)).map(Some),
            None => Ok(None),
        }
    }

    fn impl_self_alias(
        &mut self,
        block: ImplRef,
        (name, own): (&str, &[Ty]),
    ) -> Result<Option<Ty>> {
        if let Some(ty) = self.impl_alias(block, (name, own))? {
            return Ok(Some(ty));
        }
        let (Some(of), Some(self_ty)) = (self.cx.impl_trait(block), self.self_ty) else {
            return Ok(None);
        };
        if !self.reaches_assoc_ty(of, name)? {
            return Ok(None);
        }
        let count = self.impl_params(block)?;
        let args = if self.generics.len() >= count {
            let impl_args: Vec<Ty> = self.generics[..count].iter().map(|&(_, ty)| ty).collect();
            self.impl_trait_args(block, &impl_args, self_ty)?
        } else {
            self.fresh_trait_args(of)?
        };
        match self.assoc_owner((of, args), self_ty, name, 0)? {
            Some((owner, args)) => self
                .project_own(owner, self_ty, args, (name, own))
                .map(Some),
            None => Ok(None),
        }
    }

    fn object_super_alias(
        &mut self,
        of: ItemRef,
        ty: Ty,
        (principal, args): (ItemRef, Vec<Ty>),
        name: &str,
    ) -> Result<Option<Ty>> {
        let mut stack = vec![(
            principal,
            std::iter::once(ty).chain(args).collect::<Vec<Ty>>(),
            0,
        )];
        while let Some((trait_of, substs, depth)) = stack.pop() {
            if depth > SUPER_DEPTH {
                continue;
            }
            for ((parent, parent_substs), bindings) in self.super_bindings(trait_of, &substs)? {
                if let Some((_, found)) = bindings.into_iter().find(|(bound, _)| bound == name)
                    && self
                        .assoc_owner((parent, parent_substs[1..].to_vec()), ty, name, 0)?
                        .is_some_and(|(owner, _)| owner == of)
                {
                    return Ok(Some(found));
                }
                stack.push((parent, parent_substs, depth + 1));
            }
        }
        Ok(None)
    }

    /// The associated type `name` of trait `of` at `args` for `ty`: read from the impl that selection picks, or a variable until `ty` is known.
    pub(in super::super) fn project(
        &mut self,
        of: ItemRef,
        ty: Ty,
        args: Vec<Ty>,
        name: &str,
    ) -> Result<Ty> {
        self.project_own(of, ty, args, (name, &[]))
    }

    /// The generic associated type `name` of trait `of` at `args` for `ty`, at its own arguments `own`.
    pub(super) fn project_own(
        &mut self,
        of: ItemRef,
        ty: Ty,
        args: Vec<Ty>,
        (name, own): (&str, &[Ty]),
    ) -> Result<Ty> {
        if let Some(found) = self.picked_alias(of, ty, &args, (name, own))? {
            return Ok(found);
        }
        if !self.obligation_open(ty, &args) {
            bail!(
                "no one impl of `{}` gives `{name}` for {}",
                self.cx.name(of),
                self.show(ty)
            );
        }
        let result = self.fresh(Kind::Var);
        self.projections
            .push((of, ty, args, name.to_string(), own.to_vec(), result));
        Ok(result)
    }

    fn projection_keys(
        &mut self,
        of: ItemRef,
        ty: Ty,
        args: &[Ty],
        (name, own): (&str, &[Ty]),
    ) -> (Option<ProjectionKey>, Option<(LocalProjectionKey, usize)>) {
        self.sync_local_cache();
        if self.closed(ty)
            && let Some(args) = self.closed_substs(args)
            && let Some(own) = self.closed_substs(own)
        {
            let sym = self.cx.names.symbols.intern(name);
            return (Some((of, self.substitute(ty), args, sym, own)), None);
        }
        let Some((sub_ty, sub_args, sub_own, krate, var_limit)) =
            self.local_key_parts(ty, args, own)
        else {
            return (None, None);
        };
        let sym = self.cx.names.symbols.intern(name);
        (
            None,
            Some(((of, sub_ty, sub_args, sym, sub_own, krate), var_limit)),
        )
    }

    pub(in crate::sema) fn picked_alias(
        &mut self,
        of: ItemRef,
        ty: Ty,
        args: &[Ty],
        (name, own): (&str, &[Ty]),
    ) -> Result<Option<Ty>> {
        if self.has_params(ty) || self.any_has_params(args) || self.any_has_params(own) {
            return match self.poly.is_some() {
                true => self.poly_picked_alias(of, ty, args, (name, own)),
                false => Ok(None),
            };
        }
        let (key, local_key) = self.projection_keys(of, ty, args, (name, own));
        if let Some(found) = key
            .as_ref()
            .and_then(|k| self.cx.store.lookups.projections.get(k))
        {
            return Ok(found);
        }
        if let Some((ref k, key_limit)) = local_key
            && let Some(found) = self.cached_local_projection(k, key_limit)
        {
            return Ok(found);
        }
        let state = self.begin_cache_state();
        let res = self.picked_alias_uncached(of, ty, args, (name, own));
        let min_var = self.end_cache_state();
        let res = res?;
        self.store_projection(key, local_key, res, state, min_var);
        Ok(res)
    }

    fn cached_local_projection(
        &mut self,
        k: &LocalProjectionKey,
        key_limit: usize,
    ) -> Option<Option<Ty>> {
        if key_limit == 0
            && self.local_closed_cache_epoch == self.capture_epoch
            && let Some((found, obs)) = self.local_closed_projections.get(k)
        {
            let found = *found;
            replay_obligations(&mut self.obligations, obs);
            return Some(found);
        }
        if self.local_cache_epoch == self.var_epoch
            && let Some((found, obs)) = self.local_projections.get(k)
        {
            let found = *found;
            replay_obligations(&mut self.obligations, obs);
            return Some(found);
        }
        None
    }

    fn store_projection(
        &mut self,
        key: Option<ProjectionKey>,
        local_key: Option<(LocalProjectionKey, usize)>,
        res: Option<Ty>,
        state: CacheState,
        min_var: usize,
    ) {
        let (global_ok, local_ok) = self.cache_check(state, min_var);
        if global_ok && let Some(k) = key {
            if res.is_none_or(|t| self.closed(t)) {
                let cached = res.map(|t| self.substitute(t));
                self.cx.store.lookups.projections.insert(k, cached);
            }
            return;
        }
        if !local_ok {
            return;
        }
        let Some((k, key_limit)) = local_key else {
            return;
        };
        let single;
        let val_tys: &[Ty] = match res {
            Some(t) => {
                single = [t];
                &single
            }
            None => &[],
        };
        let Some((obs, var_limit)) = self.prepare_local_store(key_limit, val_tys, state) else {
            return;
        };
        let cached = res.map(|t| self.substitute(t));
        if var_limit == 0 {
            self.local_closed_projections.insert(k, (cached, obs));
        } else {
            self.local_projections.insert(k, (cached, obs));
        }
    }

    fn picked_alias_uncached(
        &mut self,
        of: ItemRef,
        ty: Ty,
        args: &[Ty],
        (name, own): (&str, &[Ty]),
    ) -> Result<Option<Ty>> {
        if self.cx.lang(of) == Some("discriminant_kind") {
            return self.discriminant_ty(ty);
        }
        if let Some(found) = self.builtin_alias(of, ty, name)? {
            return Ok(Some(found));
        }
        let Some(selected) = self.select(of, ty, args)? else {
            return Ok(None);
        };
        let (block, impl_args) = self
            .impl_member(of, (ty, args), selected.clone(), name)?
            .map_or(selected, |(_, picked)| picked);
        let frame = self.impl_frame_of(block, &impl_args, Some(ty))?;
        match self.with_frame(frame, |infer| infer.impl_alias(block, (name, own)))? {
            Some(found) => Ok(Some(found)),
            None => bail!(
                "the impl of `{}` for {} has no type `{name}`",
                self.cx.name(of),
                self.show(ty)
            ),
        }
    }

    pub(super) fn builtin_alias(&mut self, of: ItemRef, ty: Ty, name: &str) -> Result<Option<Ty>> {
        if self.cx.lang(of) == Some("pointee_trait") && name == "Metadata" {
            return self.pointee_metadata(ty);
        }
        let kind = *self.cx.types.kind(self.resolve(ty));
        if let Kind::Dyn(principal, args, members, _) = kind {
            let member = self.cx.trait_member(of, name);
            let trait_args = {
                let (trait_args, mut bound) = super::super::ty::object_parts(
                    self.cx.types.substs(args),
                    self.cx.types.items(members),
                );
                if let Some((_, found)) = bound.find(|(bound, _)| Some(*bound) == member) {
                    return Ok(Some(found));
                }
                trait_args.to_vec()
            };
            return self.object_super_alias(of, ty, (principal, trait_args), name);
        }
        if let Some(found) = self.callable_assoc_alias(of, ty, name)? {
            return Ok(Some(found));
        }
        if self.cx.lang(of) == Some("future_trait")
            && name == "Output"
            && let Some(output) = self
                .coroutine_output(ty)
                .or_else(|| self.opaque_output(self.resolve(ty)))
        {
            return Ok(Some(output));
        }
        self.opaque_binding(of, self.resolve(ty), name)
    }

    /// The `<ty as Pointee>::Metadata` type once the unsized tail of `ty` is known, or none while it is an inference variable.
    pub(super) fn pointee_metadata(&mut self, ty: Ty) -> Result<Option<Ty>> {
        if matches!(self.cx.types.kind(self.resolve(ty)), Kind::Var(_)) {
            return Ok(None);
        }
        let tailed = self.tail_ty(ty)?;
        let tail = self.resolve(tailed);
        match self.cx.types.kind(tail).clone() {
            Kind::Var(_) => Ok(None),
            Kind::Slice(_) | Kind::Prim(Primitive::Str) => {
                Ok(Some(self.cx.types.prim(Primitive::Usize)))
            }
            Kind::Dyn(..) => {
                let Some(item) = self.cx.lang_item("dyn_metadata") else {
                    bail!("core has no `DynMetadata`");
                };
                Ok(Some(self.cx.types.adt(item, &[tail])))
            }
            _ if self.has_params(tail) => self.param_metadata(tail).map(Some),
            _ => Ok(Some(self.cx.types.unit())),
        }
    }

    pub(super) fn discriminant_ty(&mut self, ty: Ty) -> Result<Option<Ty>> {
        match *self.cx.types.kind(self.resolve(ty)) {
            Kind::Adt(item, _) if self.cx.item(item).1.kind == ItemKind::Enum => Ok(Some(
                self.cx.types.prim(self.cx.discriminant_primitive(item)),
            )),
            Kind::Var(_) => Ok(None),
            Kind::Param(_) | Kind::Proj(_) if self.poly.is_some() => {
                self.poly_fallback();
                Ok(None)
            }
            _ => bail!(
                "the discriminant type of {} is not supported yet",
                self.show(ty)
            ),
        }
    }

    pub(super) fn assoc_owner(
        &mut self,
        (of, args): (ItemRef, Vec<Ty>),
        ty: Ty,
        name: &str,
        depth: u32,
    ) -> Result<Option<(ItemRef, Vec<Ty>)>> {
        if self.has_assoc_ty(of, name) {
            return Ok(Some((of, args)));
        }
        let def = self.cx.trait_generics(of)?;
        if depth > SUPER_DEPTH || def.1.supers.len == 0 {
            return Ok(None);
        }
        let params: Vec<_> = type_params(&def.1.params).copied().collect();
        let frame = self.item_frame(of, (&params, &args), (Some(ty), Some(of)))?;
        let home = frame.home;
        let supers = self.with_frame(frame, |infer| {
            let mut found = Vec::new();
            for &bound in def.0.list(def.1.supers) {
                let Some((of, path)) = infer.bound_trait_at(&def.0, bound, home) else {
                    continue;
                };
                if infer.reaches(of, name, Self::has_assoc_ty, depth + 1, &mut Vec::new())? {
                    found.push((of, infer.bound_args(of, &def.0, path, ty)?));
                }
            }
            Ok(found)
        })?;
        for parent in supers {
            if let Some(owner) = self.assoc_owner(parent, ty, name, depth + 1)? {
                return Ok(Some(owner));
            }
        }
        Ok(None)
    }

    /// Resolves each deferred associated type whose impl can now be picked.
    pub(in crate::sema) fn settle_projections(&mut self) -> Result<()> {
        let mut at = 0;
        while at < self.projections.len() {
            let (of, ty, args, name, own, result) = self.projections[at].clone();
            match self.picked_alias(of, ty, &args, (&name, &own))? {
                Some(found) => {
                    self.projections.swap_remove(at);
                    self.unify(found, result)?;
                }
                None => at += 1,
            }
        }
        Ok(())
    }
}

/// The angle-bracketed arguments of the last segment of the bound path `path`, as in `FnOnce<A, Output = R>`.
pub(super) fn angle_args(arena: &Body, path: PathId) -> Option<crate::body::List> {
    let &last = arena.list(arena.path(path).segments).last()?;
    match arena.segments[last as usize].args {
        Some(crate::body::Args::Angle(list)) => Some(list),
        _ => None,
    }
}

/// The associated type bounds that the last segment of the bound path `path` puts, as in `Iterator<Item: Try<Output = T>>`: each name and its bounds.
pub(super) fn path_constraints(arena: &Body, path: PathId) -> Vec<(u32, crate::body::List)> {
    let Some(list) = angle_args(arena, path) else {
        return Vec::new();
    };
    arena
        .list(list)
        .iter()
        .filter_map(|&at| match arena.args[at as usize] {
            crate::body::GenericArg::Constraint { name, bounds } => Some((name, bounds)),
            _ => None,
        })
        .collect()
}

pub(super) fn has_assoc_bounds(arena: &Body, path: PathId) -> bool {
    let Some(&last) = arena.list(arena.path(path).segments).last() else {
        return false;
    };
    match arena.segments[last as usize].args {
        Some(crate::body::Args::Paren { .. }) => true,
        Some(crate::body::Args::Angle(list)) => arena.list(list).iter().any(|&at| {
            matches!(
                arena.args[at as usize],
                crate::body::GenericArg::Binding { .. }
                    | crate::body::GenericArg::Constraint { .. }
            )
        }),
        Some(crate::body::Args::ReturnType) | None => false,
    }
}

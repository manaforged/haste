use anyhow::{Result, bail};

use crate::body::{Body, Bound, GenericArg, PathId};
use crate::names::{ItemRef, ModRef};
use crate::outline::item::ItemKind;

use super::super::ty::{Kind, Ty};
use super::Infer;
use super::scope::{named_bounds, type_params};

/// An associated type whose `Self` is not known yet: trait, `Self`, trait arguments, name, and the variable that stands for it.
pub(super) type Projection = (ItemRef, Ty, Vec<Ty>, String, Ty);

const SUPER_DEPTH: u32 = 8;

impl Infer<'_, '_> {
    /// The type that `param::name` names, when `param` is `Self` or a type parameter of the frame and a trait that bounds it has the associated type `name`.
    pub(super) fn param_projection(&mut self, param: &str, name: &str) -> Result<Option<Ty>> {
        if param == "Self"
            && let Some(block) = self.block
            && let Some(ty) = self.impl_alias(block, name)?
        {
            return Ok(Some(ty));
        }
        let Some(ty) = self.param_ty(param) else { return Ok(None) };
        if param == "Self"
            && let Some(of) = self.frame_item.and_then(|item| self.cx.owner_trait(item))
        {
            let count = type_params(&self.cx.trait_generics(of)?.1.params).count();
            let args: Vec<Ty> = self.generics.iter().take(count).map(|&(_, ty)| ty).collect();
            if let Some((owner, args)) = self.assoc_owner((of, args), ty, name, 0)? {
                return self.project(owner, ty, args, name).map(Some);
            }
        }
        let Some(bound) = self.bound_trait((param, ty), name, Self::reaches_assoc_ty)? else { return Ok(None) };
        match self.assoc_owner(bound, ty, name, 0)? {
            Some((of, args)) => self.project(of, ty, args, name).map(Some),
            None => Ok(None),
        }
    }

    /// The type that `param` stands for in the frame: `Self` or one of its type parameters.
    pub(super) fn param_ty(&self, param: &str) -> Option<Ty> {
        match param {
            "Self" => self.self_ty,
            _ => self.generics.iter().find(|(generic, _)| *generic == param).map(|&(_, ty)| ty),
        }
    }

    /// The first trait that bounds `param`, standing for `ty`, in the frame's headers and has a member `name` that `wants` accepts, with its arguments.
    pub(super) fn bound_trait(&mut self, (param, ty): (&str, Ty), name: &str, wants: fn(&mut Self, ItemRef, &str) -> Result<bool>) -> Result<Option<(ItemRef, Vec<Ty>)>> {
        for header in self.headers()? {
            let (arena, generics, predicates) = header.parts();
            for bound in named_bounds(arena, self.src, (generics, predicates), param) {
                let Bound::Trait { path, maybe: false, .. } = arena.bounds[bound as usize] else { continue };
                let Some(of) = self.trait_named(arena, path, self.home) else { continue };
                if !wants(self, of, name)? {
                    continue;
                }
                return Ok(Some((of, self.bound_args(of, arena, path, ty)?)));
            }
        }
        Ok(None)
    }

    /// Whether trait `of` has a member `name`.
    pub(super) fn has_member(&mut self, of: ItemRef, name: &str) -> Result<bool> {
        Ok(self.cx.trait_member(of, name).is_some())
    }

    fn has_assoc_ty(&self, of: ItemRef, name: &str) -> bool {
        self.cx.trait_member(of, name).is_some_and(|member| self.cx.item(member).1.kind == ItemKind::Type)
    }

    fn reaches_assoc_ty(&mut self, of: ItemRef, name: &str) -> Result<bool> {
        self.declares_assoc_ty(of, name, 0)
    }

    fn declares_assoc_ty(&mut self, of: ItemRef, name: &str, depth: u32) -> Result<bool> {
        if self.has_assoc_ty(of, name) {
            return Ok(true);
        }
        let def = self.cx.trait_generics(of)?;
        if depth > SUPER_DEPTH || def.1.supers.len == 0 {
            return Ok(false);
        }
        let params: Vec<_> = type_params(&def.1.params).copied().collect();
        let args: Vec<Ty> = params.iter().map(|_| self.fresh(Kind::Var)).collect();
        let frame = self.item_frame(of, (&params, &args), (None, None))?;
        let home = frame.home;
        let parents = self.with_frame(frame, |infer| {
            let bounds = def.0.list(def.1.supers).iter().map(|&bound| def.0.bounds[bound as usize]);
            let paths: Vec<PathId> = bounds.filter_map(|bound| if let Bound::Trait { path, maybe: false, .. } = bound { Some(path) } else { None }).collect();
            Ok(paths.into_iter().filter_map(|path| infer.trait_named(&def.0, path, home)).collect::<Vec<_>>())
        })?;
        for parent in parents {
            if self.declares_assoc_ty(parent, name, depth + 1)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// The associated type `name` of trait `of` at `args` for `ty`: read from the impl that selection picks, or a variable until `ty` is known.
    pub(super) fn project(&mut self, of: ItemRef, ty: Ty, args: Vec<Ty>, name: &str) -> Result<Ty> {
        if let Some(found) = self.picked_alias(of, ty, &args, name)? {
            return Ok(found);
        }
        if !self.has_vars(ty) && !args.iter().any(|arg| self.has_vars(*arg)) {
            bail!("no one impl of `{}` gives `{name}` for {}", self.cx.name(of), self.show(ty));
        }
        let result = self.fresh(Kind::Var);
        self.projections.push((of, ty, args, name.to_string(), result));
        Ok(result)
    }

    fn picked_alias(&mut self, of: ItemRef, ty: Ty, args: &[Ty], name: &str) -> Result<Option<Ty>> {
        if self.cx.lang(of) == Some("discriminant_kind") {
            return self.discriminant_ty(ty);
        }
        if let Some(found) = self.builtin_alias(of, ty, name)? {
            return Ok(Some(found));
        }
        let Some((block, impl_args)) = self.select(of, ty, args)? else { return Ok(None) };
        let frame = self.impl_frame_of(block, &impl_args, Some(ty))?;
        match self.with_frame(frame, |infer| infer.impl_alias(block, name))? {
            Some(found) => Ok(Some(found)),
            None => bail!("the impl of `{}` for {} has no type `{name}`", self.cx.name(of), self.show(ty)),
        }
    }

    fn builtin_alias(&mut self, of: ItemRef, ty: Ty, name: &str) -> Result<Option<Ty>> {
        let kind = self.cx.types.kind(self.resolve(ty)).clone();
        if let Kind::Dyn(_, args, members) = &kind {
            let member = self.cx.trait_member(of, name);
            return Ok(super::super::ty::object_parts(args, members).1.find(|(bound, _)| Some(*bound) == member).map(|(_, found)| found));
        }
        if self.cx.lang(of) == Some("fn_once") && name == "Output" {
            return Ok(self.callable_sig(ty)?.map(|(_, ret)| ret));
        }
        Ok(None)
    }

    fn discriminant_ty(&mut self, ty: Ty) -> Result<Option<Ty>> {
        match *self.cx.types.kind(self.resolve(ty)) {
            Kind::Adt(item, _) if self.cx.item(item).1.kind == ItemKind::Enum => Ok(Some(self.cx.types.prim(self.cx.discriminant_primitive(item)))),
            Kind::Var(_) => Ok(None),
            _ => bail!("the discriminant type of {} is not supported yet", self.show(ty)),
        }
    }

    /// Ties each associated type that a bound of function `item` binds, as in `I: Iterator<Item = A>`, to the bound type at `substs`, so the call's arguments infer it.
    pub(super) fn bind_bounds(&mut self, item: ItemRef, substs: &[Ty]) -> Result<()> {
        let sig = self.cx.signature(item)?;
        if !sig.body.args.iter().any(|arg| matches!(arg, GenericArg::Binding { args: None, .. })) {
            return Ok(());
        }
        let frame = self.frame(item, substs)?;
        let (src, home) = (frame.src, frame.home);
        let params: Vec<(String, Ty)> = sig.generics.iter().filter_map(|generic| frame.generics.iter().find(|(name, _)| *name == src.text(generic.name)).map(|&(name, ty)| (name.to_owned(), ty))).collect();
        self.with_frame(frame, |infer| {
            for (name, ty) in params {
                for bound in named_bounds(&sig.body, src, (&sig.generics, &sig.predicates), &name) {
                    infer.bind_bound(&sig.body, home, (bound, ty))?;
                }
            }
            Ok(())
        })
    }

    fn bind_bound(&mut self, arena: &Body, home: ModRef, (bound, ty): (u32, Ty)) -> Result<()> {
        let Bound::Trait { path, maybe: false, .. } = arena.bounds[bound as usize] else { return Ok(()) };
        let Some(of) = self.trait_named(arena, path, home) else { return Ok(()) };
        let args = self.bound_args(of, arena, path, ty)?;
        if !self.bindings_hold(arena, path, (of, ty, &args))? {
            bail!("{} does not give the associated types that a bound of `{}` binds", self.show(ty), self.cx.name(of));
        }
        Ok(())
    }

    /// Whether each associated type that the bound path `path` binds, as in `Iterator<Item = X>`, is its type for `ty` under trait `of` at `args`; unifies the two.
    pub(super) fn bindings_hold(&mut self, arena: &Body, path: PathId, (of, ty, args): (ItemRef, Ty, &[Ty])) -> Result<bool> {
        for (name, bound_ty) in self.path_bindings(arena, path)? {
            if !self.binding_holds((of, args), ty, (&name, bound_ty))? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Whether the associated type `name` of trait `of` at `args`, found on it or a supertrait, is `bound` for `ty`.
    pub(super) fn binding_holds(&mut self, (of, args): (ItemRef, &[Ty]), ty: Ty, (name, bound): (&str, Ty)) -> Result<bool> {
        let Some((owner, owner_args)) = self.assoc_owner((of, args.to_vec()), ty, name, 0)? else { bail!("`{}` has no associated type `{name}`", self.cx.name(of)) };
        let found = self.project(owner, ty, owner_args, name)?;
        Ok(self.unify(found, bound).is_ok())
    }

    pub(super) fn assoc_owner(&mut self, (of, args): (ItemRef, Vec<Ty>), ty: Ty, name: &str, depth: u32) -> Result<Option<(ItemRef, Vec<Ty>)>> {
        if self.has_assoc_ty(of, name) {
            return Ok(Some((of, args)));
        }
        let def = self.cx.trait_generics(of)?;
        if depth > SUPER_DEPTH || def.1.supers.len == 0 {
            return Ok(None);
        }
        let params: Vec<_> = type_params(&def.1.params).copied().collect();
        let frame = self.item_frame(of, (&params, &args), (Some(ty), None))?;
        let home = frame.home;
        let supers = self.with_frame(frame, |infer| {
            let mut found = Vec::new();
            for &bound in def.0.list(def.1.supers) {
                let Bound::Trait { path, maybe: false, .. } = def.0.bounds[bound as usize] else { continue };
                let Some(of) = infer.trait_named(&def.0, path, home) else { continue };
                found.push((of, infer.bound_args(of, &def.0, path, ty)?));
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
    pub(super) fn settle_projections(&mut self) -> Result<()> {
        let mut at = 0;
        while at < self.projections.len() {
            let (of, ty, args, name, result) = self.projections[at].clone();
            match self.picked_alias(of, ty, &args, &name)? {
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

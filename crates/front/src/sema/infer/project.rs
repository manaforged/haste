use anyhow::{Result, bail};

use crate::body::Bound;
use crate::names::ItemRef;
use crate::outline::item::ItemKind;

use super::super::ty::{Kind, Ty};
use super::Infer;
use super::scope::{named_bounds, type_params};

/// An associated type whose `Self` is not known yet: trait, `Self`, trait arguments, name, and the variable that stands for it.
pub(super) type Projection = (ItemRef, Ty, Vec<Ty>, String, Ty);

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
            && self.has_assoc_ty(of, name)
        {
            let count = type_params(&self.cx.trait_generics(of)?.1.params).count();
            let args: Vec<Ty> = self.generics.iter().take(count).map(|&(_, ty)| ty).collect();
            return self.project(of, ty, args, name).map(Some);
        }
        match self.bound_trait((param, ty), name, Self::has_assoc_ty)? {
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
    pub(super) fn bound_trait(&mut self, (param, ty): (&str, Ty), name: &str, wants: fn(&Self, ItemRef, &str) -> bool) -> Result<Option<(ItemRef, Vec<Ty>)>> {
        for header in self.headers()? {
            let (arena, generics, predicates) = header.parts();
            for bound in named_bounds(arena, self.src, (generics, predicates), param) {
                let Bound::Trait { path, maybe: false, .. } = arena.bounds[bound as usize] else { continue };
                let Some(of) = self.trait_named(arena, path, self.home) else { continue };
                if !wants(self, of, name) {
                    continue;
                }
                let count = arena.list(arena.path(path).segments).len();
                let given = self.segment_args(arena, path, count)?;
                return Ok(Some((of, self.trait_args(of, given, ty)?)));
            }
        }
        Ok(None)
    }

    fn has_assoc_ty(&self, of: ItemRef, name: &str) -> bool {
        self.cx.trait_member(of, name).is_some_and(|member| self.cx.item(member).1.kind == ItemKind::Type)
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
        let Some((block, impl_args)) = self.select(of, ty, args)? else { return Ok(None) };
        let frame = self.impl_frame_of(block, &impl_args, Some(ty))?;
        match self.with_frame(frame, |infer| infer.impl_alias(block, name))? {
            Some(found) => Ok(Some(found)),
            None => bail!("the impl of `{}` for {} has no type `{name}`", self.cx.name(of), self.show(ty)),
        }
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

use anyhow::{Result, bail};

use crate::body::sig::Predicate;
use crate::body::{Body, PathId, TyId};
use crate::names::{ItemRef, ModRef};
use crate::outline::item::ItemKind;
use crate::outline::source::Source;
use crate::sema::impls::ImplRef;
use crate::sema::ty::{Kind, Ty};

use super::super::Infer;
use super::super::scope::{named_bounds, one_name, same_param, type_params};
use super::{SUPER_DEPTH, angle_args, path_constraints};

const MAX_ASSUMPTIONS: usize = 256;

impl Infer<'_, '_> {
    /// The type that `param` stands for in the frame: `Self` or one of its type parameters.
    pub(in super::super) fn param_ty(&self, param: &str) -> Option<Ty> {
        match param {
            "Self" => self.self_ty,
            _ => self
                .generics
                .iter()
                .find(|(generic, _)| same_param(generic, param))
                .map(|&(_, ty)| ty),
        }
    }

    /// The first trait that bounds `param`, standing for `ty`, in the frame's headers and has a member `name` that `wants` accepts, with its arguments.
    pub(in super::super) fn bound_trait(
        &mut self,
        (param, ty): (&str, Ty),
        name: &str,
        wants: fn(&mut Self, ItemRef, &str) -> Result<bool>,
    ) -> Result<Option<(ItemRef, Vec<Ty>)>> {
        Ok(self
            .header_bounds((param, ty), &mut |infer, of| wants(infer, of, name), true)?
            .pop())
    }

    /// The trait bounds that the frame's headers put on its type parameters, each as the trait and its types with `Self` first: what the body may assume holds.
    pub(in super::super) fn assumptions(&mut self) -> Result<Vec<(ItemRef, Vec<Ty>)>> {
        if let Some(poly) = &self.poly
            && poly.phase != super::super::poly::PolyPhase::HeaderInit
        {
            return Ok(poly.assumptions.clone());
        }
        let pending = self.obligations.len();
        let found = self.assumed_bounds();
        self.obligations.truncate(pending);
        found
    }

    fn assumed_bounds(&mut self) -> Result<Vec<(ItemRef, Vec<Ty>)>> {
        let mut found = self.param_bounds()?;
        found.extend(self.own_trait()?);
        found.extend(self.type_bounds()?);
        let mut at = 0;
        while at < found.len() && found.len() < MAX_ASSUMPTIONS {
            let (of, substs) = found[at].clone();
            let mut implied = self.super_bounds(of, &substs)?;
            implied.extend(self.alias_bounds(of, &substs)?);
            for implied in implied {
                if !found.contains(&implied) {
                    found.push(implied);
                }
            }
            at += 1;
        }
        Ok(found)
    }

    fn param_bounds(&mut self) -> Result<Vec<(ItemRef, Vec<Ty>)>> {
        let mut found = Vec::new();
        for (param, ty) in self.generics.clone() {
            for (of, args) in self.header_bounds((param, ty), &mut |_, _| Ok(true), false)? {
                found.push((of, std::iter::once(ty).chain(args).collect()));
            }
        }
        Ok(found)
    }

    /// The supertrait bounds that `substs[0]: of<substs[1..]>` implies, each with its arguments.
    pub(in super::super) fn super_bounds(
        &mut self,
        of: ItemRef,
        substs: &[Ty],
    ) -> Result<Vec<(ItemRef, Vec<Ty>)>> {
        Ok(self
            .super_bindings(of, substs)?
            .into_iter()
            .map(|(bound, _)| bound)
            .collect())
    }

    /// The supertrait bounds that `substs[0]: of<substs[1..]>` implies, each with its arguments and the associated types its path binds, as `Iterator<Item = T>`.
    pub(in super::super) fn super_bindings(
        &mut self,
        of: ItemRef,
        substs: &[Ty],
    ) -> Result<Vec<((ItemRef, Vec<Ty>), Vec<(String, Ty)>)>> {
        let def = self.cx.trait_generics(of)?;
        let params: Vec<_> = type_params(&def.1.params).copied().collect();
        if def.1.supers.len == 0 || substs.len() != params.len() + 1 {
            return Ok(Vec::new());
        }
        let frame = self.item_frame(of, (&params, &substs[1..]), (Some(substs[0]), Some(of)))?;
        let home = frame.home;
        self.with_frame(frame, |infer| {
            let mut found = Vec::new();
            for &bound in def.0.list(def.1.supers) {
                let Some((parent, path)) = infer.bound_trait_at(&def.0, bound, home) else {
                    continue;
                };
                if let Ok(args) = infer.bound_args(parent, &def.0, path, substs[0]) {
                    let bindings = infer.path_bindings(&def.0, path)?;
                    found.push((
                        (parent, std::iter::once(substs[0]).chain(args).collect()),
                        bindings,
                    ));
                }
            }
            Ok(found)
        })
    }

    /// The trait whose default method the frame types, with its `Self` and arguments, which the method may assume holds.
    pub(in super::super) fn own_trait(&mut self) -> Result<Option<(ItemRef, Vec<Ty>)>> {
        let (Some(of), Some(self_ty), None) = (
            self.frame_item.and_then(|item| self.cx.owner_trait(item)),
            self.self_ty,
            self.block,
        ) else {
            return Ok(None);
        };
        let def = self.cx.trait_generics(of)?;
        let src = &self.cx.file(of).source;
        let args: Option<Vec<Ty>> = type_params(&def.1.params)
            .map(|param| {
                self.generics
                    .iter()
                    .find(|(name, _)| *name == src.text(param.name))
                    .map(|&(_, ty)| ty)
            })
            .collect();
        Ok(args.map(|args| (of, std::iter::once(self_ty).chain(args).collect())))
    }

    fn alias_bounds(&mut self, of: ItemRef, substs: &[Ty]) -> Result<Vec<(ItemRef, Vec<Ty>)>> {
        let def = self.cx.trait_generics(of)?;
        let params: Vec<_> = type_params(&def.1.params).copied().collect();
        let mut found = Vec::new();
        for member in self.cx.children(of) {
            if self.cx.item(member).1.kind != ItemKind::Type {
                continue;
            }
            let bounds = self.cx.trait_generics(member)?;
            if bounds.1.supers.len == 0 || substs.len() != params.len() + 1 {
                continue;
            }
            let Some(value) =
                self.picked_alias(of, substs[0], &substs[1..], (self.cx.name(member), &[]))?
            else {
                continue;
            };
            let frame =
                self.item_frame(of, (&params, &substs[1..]), (Some(substs[0]), Some(of)))?;
            let home = frame.home;
            self.with_frame(frame, |infer| {
                for &bound in bounds.0.list(bounds.1.supers) {
                    let Some((trait_of, path)) = infer.bound_trait_at(&bounds.0, bound, home)
                    else {
                        continue;
                    };
                    if let Ok(args) = infer.bound_args(trait_of, &bounds.0, path, value) {
                        found.push((trait_of, std::iter::once(value).chain(args).collect()));
                    }
                }
                Ok(())
            })?;
        }
        Ok(found)
    }

    pub(in super::super) fn header_bounds(
        &mut self,
        (param, ty): (&str, Ty),
        keep: &mut dyn FnMut(&mut Self, ItemRef) -> Result<bool>,
        first: bool,
    ) -> Result<Vec<(ItemRef, Vec<Ty>)>> {
        let mut found = Vec::new();
        for (header, src, home) in self.headers()? {
            let (arena, generics, predicates) = header.parts();
            let done = self.in_source_home(src, home, |infer| {
                for bound in named_bounds(arena, src, (generics, predicates), param) {
                    let Some((of, path)) = infer.bound_trait_at(arena, bound, home) else {
                        continue;
                    };
                    if keep(infer, of)? {
                        found.push((of, infer.bound_args(of, arena, path, ty)?));
                        if first {
                            return Ok(true);
                        }
                    }
                }
                Ok(false)
            })?;
            if done {
                break;
            }
        }
        Ok(found)
    }

    /// The trait bounds that the where-clauses of the frame's headers put on types other than one parameter, such as `<I::Item as Edge>::Id: Into<Ix>`, each with `Self` first.
    pub(in super::super) fn type_bounds(&mut self) -> Result<Vec<(ItemRef, Vec<Ty>)>> {
        let mut found = Vec::new();
        for (header, src, home) in self.headers()? {
            let (arena, _, predicates) = header.parts();
            self.in_source_home(src, home, |infer| {
                for (ty, bound) in infer.typed_predicates(arena, src, predicates) {
                    let Some((of, path)) = infer.bound_trait_at(arena, bound, home) else {
                        continue;
                    };
                    found.push((
                        of,
                        std::iter::once(ty)
                            .chain(infer.bound_args(of, arena, path, ty)?)
                            .collect(),
                    ));
                }
                Ok(())
            })?;
        }
        Ok(found)
    }

    /// Each bound of the where-clauses in `predicates` whose bounded type is not one parameter, with that type lowered in the current frame.
    pub(in super::super) fn typed_predicates(
        &mut self,
        arena: &Body,
        src: &Source,
        predicates: &[Predicate],
    ) -> Vec<(Ty, u32)> {
        let mut found = Vec::new();
        for predicate in predicates {
            let Some(ty) = predicate
                .ty
                .filter(|&ty| !self.is_frame_param(arena, src, ty))
            else {
                continue;
            };
            let Ok(ty) = self.lower_ty(arena, ty) else {
                continue;
            };
            found.extend(
                arena
                    .list(predicate.bounds)
                    .iter()
                    .map(|&bound| (ty, bound)),
            );
        }
        found
    }

    fn is_frame_param(&self, arena: &Body, src: &Source, ty: TyId) -> bool {
        let Some(name) = one_name(arena, src, ty) else {
            return false;
        };
        (name == "Self" && self.self_ty.is_some())
            || self
                .generics
                .iter()
                .any(|&(param, _)| same_param(param, name))
    }

    /// Whether trait `of` has a member `name`.
    pub(in super::super) fn has_member(&mut self, of: ItemRef, name: &str) -> Result<bool> {
        Ok(self.cx.trait_member(of, name).is_some())
    }

    pub(super) fn has_assoc_ty(&self, of: ItemRef, name: &str) -> bool {
        self.cx
            .trait_member(of, name)
            .is_some_and(|member| self.cx.item(member).1.kind == ItemKind::Type)
    }

    pub(super) fn reaches_assoc_ty(&mut self, of: ItemRef, name: &str) -> Result<bool> {
        self.reaches(of, name, Self::has_assoc_ty, 0, &mut Vec::new())
    }

    /// Whether trait `of` or one of its supertraits has a member `name`.
    pub(in super::super) fn reaches_member(&mut self, of: ItemRef, name: &str) -> Result<bool> {
        self.reaches(
            of,
            name,
            |infer, of, name| infer.cx.trait_member(of, name).is_some(),
            0,
            &mut Vec::new(),
        )
    }

    pub(super) fn reaches(
        &mut self,
        of: ItemRef,
        name: &str,
        owns: fn(&Self, ItemRef, &str) -> bool,
        depth: u32,
        seen: &mut Vec<ItemRef>,
    ) -> Result<bool> {
        if seen.contains(&of) {
            return Ok(false);
        }
        seen.push(of);
        if owns(self, of, name) {
            return Ok(true);
        }
        if depth > SUPER_DEPTH {
            return Ok(false);
        }
        for parent in self.parent_traits(of)? {
            if self.reaches(parent, name, owns, depth + 1, seen)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// The traits that trait `of` names as its supertraits.
    pub(in super::super) fn parent_traits(&mut self, of: ItemRef) -> Result<Vec<ItemRef>> {
        let def = self.cx.trait_generics(of)?;
        if def.1.supers.len == 0 {
            return Ok(Vec::new());
        }
        let params: Vec<_> = type_params(&def.1.params).copied().collect();
        let args: Vec<Ty> = params.iter().map(|_| self.fresh(Kind::Var)).collect();
        let frame = self.item_frame(of, (&params, &args), (None, None))?;
        let home = frame.home;
        self.with_frame(frame, |infer| {
            Ok(def
                .0
                .list(def.1.supers)
                .iter()
                .filter_map(|&bound| infer.bound_trait_at(&def.0, bound, home))
                .map(|(of, _)| of)
                .collect())
        })
    }

    /// Whether the active inference root belongs to a user crate rather than a fixed sysroot crate.
    pub(in super::super) fn root_is_user(&self) -> bool {
        self.root
            .is_some_and(|root| !self.cx.outline.is_fixed(root.krate))
    }

    /// Whether trait `of` should be checked as an obligation on call-site and impl bounds.
    pub(in super::super) fn checks_bound_trait(&self, of: ItemRef) -> bool {
        !self.compiler_trait(of)
            || self.is_thread_auto_trait(of)
            || matches!(self.cx.lang(of), Some("copy" | "clone"))
    }

    /// Holds the call of function `item` at `substs` to the bounds its header puts on any type parameter of its frame, its impl's included: each trait bound whose types are not known yet becomes an obligation, and each associated type a bound binds, as in `I: Iterator<Item = A>`, is tied to the bound type.
    pub(in super::super) fn bind_bounds(&mut self, item: ItemRef, substs: &[Ty]) -> Result<()> {
        let enforce = self.root_is_user();
        let sig = self.cx.signature(item)?;
        if sig.generics.iter().any(|generic| generic.bounds.len > 0) || !sig.predicates.is_empty() {
            let frame = self.frame(item, substs)?;
            let (src, home) = (frame.src, frame.home);
            let params: Vec<(&str, Ty)> = frame
                .generics
                .iter()
                .copied()
                .chain(frame.self_ty.map(|ty| ("Self", ty)))
                .collect();
            self.with_frame(frame, |infer| {
                for (name, ty) in params {
                    for bound in
                        named_bounds(&sig.body, src, (&sig.generics, &sig.predicates), name)
                    {
                        infer.bind_bound(&sig.body, home, (bound, ty), enforce)?;
                    }
                }
                for (ty, bound) in infer.typed_predicates(&sig.body, src, &sig.predicates) {
                    if infer
                        .bound_trait_at(&sig.body, bound, home)
                        .is_some_and(|(of, _)| !infer.checks_bound_trait(of))
                    {
                        continue;
                    }
                    infer.bind_bound(&sig.body, home, (bound, ty), enforce)?;
                }
                Ok(())
            })?;
        }
        self.bind_impl_bounds(item, substs, enforce)
    }

    fn bind_impl_bounds(&mut self, item: ItemRef, substs: &[Ty], enforce: bool) -> Result<()> {
        let Some(block) = self.cx.owner_impl(item) else {
            return Ok(());
        };
        let count = self.impl_params(block)?;
        let Some(args) = substs.get(..count) else {
            return Ok(());
        };
        if self.cx.impl_trait(block).is_some() {
            return Ok(());
        }
        if !enforce
            && !args
                .iter()
                .any(|&arg| matches!(self.cx.types.kind(self.resolve(arg)), Kind::Var(_)))
        {
            return Ok(());
        }
        for (of, ty, args, bindings) in self.impl_param_bounds(block, args)? {
            if enforce || self.obligation_open(ty, &args) {
                self.note_obligation_site(of, ty);
                self.obligations.push((
                    of,
                    std::iter::once(ty).chain(args.iter().copied()).collect(),
                ));
            }
            for (name, bound) in bindings {
                self.binding_holds((of, &args), ty, (&name, bound))?;
            }
        }
        Ok(())
    }

    fn impl_param_bounds(
        &mut self,
        block: ImplRef,
        args: &[Ty],
    ) -> Result<Vec<(ItemRef, Ty, Vec<Ty>, Vec<(String, Ty)>)>> {
        let header = self.cx.impl_header(block)?;
        let (_, self_ty) = self.impl_frame(block, args)?;
        let frame = self.impl_frame_of(block, args, Some(self_ty))?;
        let (src, home, arena) = (frame.src, frame.home, &header.body);
        let params = frame.generics.clone();
        self.with_frame(frame, |infer| {
            let mut found = Vec::new();
            let mut push_bound = |infer: &mut Self, ty: Ty, bound: u32| -> Result<()> {
                let Some((of, path)) =
                    infer.bound_trait_at(arena, bound, home).filter(|&(of, _)| {
                        infer.checks_bound_trait(of) && !infer.is_callable_trait(of)
                    })
                else {
                    return Ok(());
                };
                found.push((
                    of,
                    ty,
                    infer.bound_args(of, arena, path, ty)?,
                    infer.path_bindings(arena, path)?,
                ));
                Ok(())
            };
            for (name, ty) in params {
                for bound in named_bounds(arena, src, (&header.generics, &header.predicates), name)
                {
                    push_bound(infer, ty, bound)?;
                }
            }
            for (ty, bound) in infer.typed_predicates(arena, src, &header.predicates) {
                push_bound(infer, ty, bound)?;
            }
            Ok(found)
        })
    }

    fn bind_bound(
        &mut self,
        arena: &Body,
        home: ModRef,
        (bound, ty): (u32, Ty),
        enforce: bool,
    ) -> Result<()> {
        let Some((of, path)) = self.bound_trait_at(arena, bound, home) else {
            return Ok(());
        };
        if self.is_callable_trait(of) {
            return self.bind_callable(arena, (of, path), ty);
        }
        let args = self.bound_args(of, arena, path, ty)?;
        if !self.compiler_trait(of)
            && !self.has_vars(ty)
            && !self.has_params(ty)
            && args.iter().any(|&a| self.has_vars(a))
        {
            let _ = self.select(of, ty, &args)?;
        }
        let push = if enforce {
            self.checks_bound_trait(of)
        } else {
            (!self.compiler_trait(of) && self.obligation_open(ty, &args))
                || self.is_thread_auto_trait(of)
        };
        if push {
            self.note_obligation_site(of, ty);
            self.obligations.push((
                of,
                std::iter::once(ty).chain(args.iter().copied()).collect(),
            ));
        }
        if !self.bindings_hold(arena, path, (of, ty, &args))? {
            bail!(
                "{} does not give the associated types that a bound of `{}` binds",
                self.show(ty),
                self.cx.name(of)
            );
        }
        self.bind_constraints((arena, home), (of, path, &args), ty, enforce)
    }

    fn bind_callable(&mut self, arena: &Body, (of, path): (ItemRef, PathId), ty: Ty) -> Result<()> {
        if angle_args(arena, path).is_some() {
            let args = self.bound_args(of, arena, path, ty)?;
            self.bindings_hold(arena, path, (of, ty, &args))?;
        }
        Ok(())
    }

    fn bind_constraints(
        &mut self,
        (arena, home): (&Body, ModRef),
        (of, path, args): (ItemRef, PathId, &[Ty]),
        ty: Ty,
        enforce: bool,
    ) -> Result<()> {
        for (name, bounds) in path_constraints(arena, path) {
            let name = self.src.text(name).to_string();
            let Some((owner, owner_args)) = self.assoc_owner((of, args.to_vec()), ty, &name, 0)?
            else {
                bail!("`{}` has no associated type `{name}`", self.cx.name(of));
            };
            let assoc = self.project(owner, ty, owner_args, &name)?;
            for &bound in arena.list(bounds) {
                self.bind_bound(arena, home, (bound, assoc), enforce)?;
            }
        }
        Ok(())
    }

    /// Whether each associated type that the bound path `path` binds, as in `Iterator<Item = X>`, is its type for `ty` under trait `of` at `args`; unifies the two.
    pub(in super::super) fn assoc_bounds_hold(
        &mut self,
        (arena, home, path): (&Body, ModRef, PathId),
        (of, ty, args): (ItemRef, Ty, &[Ty]),
        depth: u32,
    ) -> Result<bool> {
        let key = (of, self.resolve(ty), path);
        if self.assoc_stack.len() >= 16 || self.assoc_stack.contains(&key) {
            return Ok(false);
        }
        self.assoc_stack.push(key);
        let res = self.assoc_bounds_hold_inner((arena, home, path), (of, ty, args), depth);
        self.assoc_stack.pop();
        res
    }

    fn assoc_bounds_hold_inner(
        &mut self,
        (arena, home, path): (&Body, ModRef, PathId),
        (of, ty, args): (ItemRef, Ty, &[Ty]),
        depth: u32,
    ) -> Result<bool> {
        if !self.bindings_hold(arena, path, (of, ty, args))? {
            return Ok(false);
        }
        let Some(&last) = arena.list(arena.path(path).segments).last() else {
            return Ok(true);
        };
        let Some(crate::body::Args::Angle(list)) = arena.segments[last as usize].args else {
            return Ok(true);
        };
        for &at in arena.list(list) {
            let crate::body::GenericArg::Constraint { name, bounds } = arena.args[at as usize]
            else {
                continue;
            };
            let name = self.src.text(name);
            let Some((owner, owner_args)) = self.assoc_owner((of, args.to_vec()), ty, name, 0)?
            else {
                bail!("`{}` has no associated type `{name}`", self.cx.name(of));
            };
            let assoc = self.project(owner, ty, owner_args, name)?;
            for &bound in arena.list(bounds) {
                if !self.bound_holds(arena, home, (assoc, bound), depth + 1)? {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    pub(in super::super) fn bindings_hold(
        &mut self,
        arena: &Body,
        path: PathId,
        (of, ty, args): (ItemRef, Ty, &[Ty]),
    ) -> Result<bool> {
        for (name, bound_ty) in self.path_bindings(arena, path)? {
            if !self.binding_holds((of, args), ty, (&name, bound_ty))? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Whether the associated type `name` of trait `of` at `args`, found on it or a supertrait, is `bound` for `ty`.
    pub(in super::super) fn binding_holds(
        &mut self,
        (of, args): (ItemRef, &[Ty]),
        ty: Ty,
        (name, bound): (&str, Ty),
    ) -> Result<bool> {
        let Some((owner, owner_args)) = self.assoc_owner((of, args.to_vec()), ty, name, 0)? else {
            bail!("`{}` has no associated type `{name}`", self.cx.name(of));
        };
        let found = self.project(owner, ty, owner_args, name)?;
        Ok(self.unifies(found, bound))
    }
}

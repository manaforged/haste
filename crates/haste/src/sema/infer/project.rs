use anyhow::{Result, bail};

use crate::body::sig::Predicate;
use crate::body::{Body, PathId};
use crate::names::{ItemRef, ModRef, Primitive};
use crate::outline::item::ItemKind;
use crate::outline::source::Source;

use super::super::impls::ImplRef;
use super::super::ty::{Kind, Ty};
use super::Infer;
use super::scope::{named_bounds, one_name, type_params};

/// An associated type whose `Self` is not known yet: trait, `Self`, trait arguments, name, and the variable that stands for it.
pub(super) type Projection = (ItemRef, Ty, Vec<Ty>, String, Ty);

const SUPER_DEPTH: u32 = 8;
const MAX_ASSUMPTIONS: usize = 256;

impl Infer<'_, '_> {
    /// The type that `param::name` names, when `param` is `Self` or a type parameter of the frame and a trait that bounds it has the associated type `name`.
    pub(super) fn param_projection(&mut self, param: &str, name: &str) -> Result<Option<Ty>> {
        if param == "Self"
            && let Some(block) = self.block
            && let Some(ty) = self.impl_self_alias(block, name)?
        {
            return Ok(Some(ty));
        }
        let Some(ty) = self.param_ty(param) else {
            return Ok(None);
        };
        if param == "Self"
            && let Some(of) = self.frame_item.and_then(|item| self.cx.owner_trait(item))
        {
            let count = type_params(&self.cx.trait_generics(of)?.1.params).count();
            let args: Vec<Ty> = self.generics.iter().take(count).map(|&(_, ty)| ty).collect();
            if let Some((owner, args)) = self.assoc_owner((of, args), ty, name, 0)? {
                return self.project(owner, ty, args, name).map(Some);
            }
        }
        let Some(bound) = self.bound_trait((param, ty), name, Self::reaches_assoc_ty)? else {
            return Ok(None);
        };
        match self.assoc_owner(bound, ty, name, 0)? {
            Some((of, args)) => self.project(of, ty, args, name).map(Some),
            None => Ok(None),
        }
    }

    /// The associated type `name` of `Self` inside impl `block`: the impl's own alias, or the one a supertrait of the implemented trait gives `Self`.
    fn impl_self_alias(&mut self, block: ImplRef, name: &str) -> Result<Option<Ty>> {
        if let Some(ty) = self.impl_alias(block, (name, &[]))? {
            return Ok(Some(ty));
        }
        let (Some(of), Some(self_ty)) = (self.cx.impl_trait(block), self.self_ty) else {
            return Ok(None);
        };
        let args = self.fresh_trait_args(of)?;
        match self.assoc_owner((of, args), self_ty, name, 0)? {
            Some((owner, args)) => self.project(owner, self_ty, args, name).map(Some),
            None => Ok(None),
        }
    }

    /// The type that `param` stands for in the frame: `Self` or one of its type parameters.
    pub(super) fn param_ty(&self, param: &str) -> Option<Ty> {
        match param {
            "Self" => self.self_ty,
            _ => self.generics.iter().find(|(generic, _)| super::scope::same_param(generic, param)).map(|&(_, ty)| ty),
        }
    }

    /// The first trait that bounds `param`, standing for `ty`, in the frame's headers and has a member `name` that `wants` accepts, with its arguments.
    pub(super) fn bound_trait(&mut self, (param, ty): (&str, Ty), name: &str, wants: fn(&mut Self, ItemRef, &str) -> Result<bool>) -> Result<Option<(ItemRef, Vec<Ty>)>> {
        Ok(self.header_bounds((param, ty), &mut |infer, of| wants(infer, of, name), true)?.pop())
    }

    /// The trait bounds that the frame's headers put on its type parameters, each as the trait and its types with `Self` first: what the body may assume holds.
    pub(super) fn assumptions(&mut self) -> Result<Vec<(ItemRef, Vec<Ty>)>> {
        if let Some(poly) = &self.poly
            && poly.env_ready
        {
            return Ok(poly.assumptions.clone());
        }
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

    /// The trait bounds that the frame's headers put on its type parameters, each with `Self` first.
    fn param_bounds(&mut self) -> Result<Vec<(ItemRef, Vec<Ty>)>> {
        let mut found = Vec::new();
        for (param, ty) in self.generics.clone() {
            for (of, args) in self.header_bounds((param, ty), &mut |_, _| Ok(true), false)? {
                found.push((of, std::iter::once(ty).chain(args).collect()));
            }
        }
        Ok(found)
    }

    /// The bounds that the associated types of the traits bounding the frame's type parameters declare, as `type S: Source` gives `<T as Logos>::S`.
    pub(super) fn param_alias_bounds(&mut self) -> Result<Vec<(ItemRef, Vec<Ty>)>> {
        let mut found = Vec::new();
        for (of, substs) in self.param_bounds()? {
            found.extend(self.alias_bounds(of, &substs)?);
        }
        Ok(found)
    }

    /// The supertrait bounds that `substs[0]: of<substs[1..]>` implies, each with its arguments.
    pub(super) fn super_bounds(&mut self, of: ItemRef, substs: &[Ty]) -> Result<Vec<(ItemRef, Vec<Ty>)>> {
        Ok(self.super_bindings(of, substs)?.into_iter().map(|(bound, _)| bound).collect())
    }

    /// The supertrait bounds that `substs[0]: of<substs[1..]>` implies, each with its arguments and the associated types its path binds, as `Iterator<Item = T>`.
    pub(super) fn super_bindings(&mut self, of: ItemRef, substs: &[Ty]) -> Result<Vec<((ItemRef, Vec<Ty>), Vec<(String, Ty)>)>> {
        let def = self.cx.trait_generics(of)?;
        let params: Vec<_> = type_params(&def.1.params).copied().collect();
        if def.1.supers.len == 0 || substs.len() != params.len() + 1 {
            return Ok(Vec::new());
        }
        let frame = self.item_frame(of, (&params, &substs[1..]), (Some(substs[0]), None))?;
        let home = frame.home;
        self.with_frame(frame, |infer| {
            let mut found = Vec::new();
            for &bound in def.0.list(def.1.supers) {
                let Some((parent, path)) = infer.bound_trait_at(&def.0, bound, home) else {
                    continue;
                };
                if let Ok(args) = infer.bound_args(parent, &def.0, path, substs[0]) {
                    let bindings = infer.path_bindings(&def.0, path)?;
                    found.push(((parent, std::iter::once(substs[0]).chain(args).collect()), bindings));
                }
            }
            Ok(found)
        })
    }

    /// The associated type `name` of trait `of` that a supertrait path of the trait object `ty`'s principal binds, as `trait Walk: Iterator<Item = u8>`.
    fn object_super_alias(&mut self, of: ItemRef, ty: Ty, (principal, args): (ItemRef, Vec<Ty>), name: &str) -> Result<Option<Ty>> {
        let mut stack = vec![(principal, std::iter::once(ty).chain(args).collect::<Vec<Ty>>(), 0)];
        while let Some((trait_of, substs, depth)) = stack.pop() {
            if depth > SUPER_DEPTH {
                continue;
            }
            for ((parent, parent_substs), bindings) in self.super_bindings(trait_of, &substs)? {
                if let Some((_, found)) = bindings.into_iter().find(|(bound, _)| bound == name)
                    && self.assoc_owner((parent, parent_substs[1..].to_vec()), ty, name, 0)?.is_some_and(|(owner, _)| owner == of)
                {
                    return Ok(Some(found));
                }
                stack.push((parent, parent_substs, depth + 1));
            }
        }
        Ok(None)
    }

    /// The trait whose default method the frame types, with its `Self` and arguments, which the method may assume holds.
    pub(super) fn own_trait(&mut self) -> Result<Option<(ItemRef, Vec<Ty>)>> {
        let (Some(of), Some(self_ty), None) = (self.frame_item.and_then(|item| self.cx.owner_trait(item)), self.self_ty, self.block) else { return Ok(None) };
        let def = self.cx.trait_generics(of)?;
        let src = &self.cx.file(of).source;
        let args: Option<Vec<Ty>> = type_params(&def.1.params).map(|param| self.generics.iter().find(|(name, _)| *name == src.text(param.name)).map(|&(_, ty)| ty)).collect();
        Ok(args.map(|args| (of, std::iter::once(self_ty).chain(args).collect())))
    }

    /// The bounds that the associated types of trait `of` declare, as in `type Word: Into<u64>`, on their values for `substs[0]: of<substs[1..]>`.
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
            let Some(value) = self.picked_alias(of, substs[0], &substs[1..], (self.cx.name(member), &[]))? else {
                continue;
            };
            let frame = self.item_frame(of, (&params, &substs[1..]), (Some(substs[0]), None))?;
            let home = frame.home;
            self.with_frame(frame, |infer| {
                for &bound in bounds.0.list(bounds.1.supers) {
                    let Some((trait_of, path)) = infer.bound_trait_at(&bounds.0, bound, home) else {
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

    fn header_bounds(&mut self, (param, ty): (&str, Ty), keep: &mut dyn FnMut(&mut Self, ItemRef) -> Result<bool>, first: bool) -> Result<Vec<(ItemRef, Vec<Ty>)>> {
        let mut found = Vec::new();
        for (header, src) in self.headers()? {
            let (arena, generics, predicates) = header.parts();
            let done = self.in_source(src, |infer| {
                for bound in named_bounds(arena, src, (generics, predicates), param) {
                    let Some((of, path)) = infer.bound_trait_at(arena, bound, infer.home) else {
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
    pub(super) fn type_bounds(&mut self) -> Result<Vec<(ItemRef, Vec<Ty>)>> {
        let mut found = Vec::new();
        for (header, src) in self.headers()? {
            let (arena, _, predicates) = header.parts();
            self.in_source(src, |infer| {
                for (ty, bound) in infer.typed_predicates(arena, src, predicates) {
                    let Some((of, path)) = infer.bound_trait_at(arena, bound, infer.home) else {
                        continue;
                    };
                    found.push((of, std::iter::once(ty).chain(infer.bound_args(of, arena, path, ty)?).collect()));
                }
                Ok(())
            })?;
        }
        Ok(found)
    }

    /// Each bound of the where-clauses in `predicates` whose bounded type is not one parameter, with that type lowered in the current frame.
    pub(super) fn typed_predicates(&mut self, arena: &Body, src: &Source, predicates: &[Predicate]) -> Vec<(Ty, u32)> {
        let mut found = Vec::new();
        for predicate in predicates {
            let Some(ty) = predicate.ty.filter(|&ty| one_name(arena, src, ty).is_none()) else {
                continue;
            };
            let Ok(ty) = self.lower_ty(arena, ty) else {
                continue;
            };
            found.extend(arena.list(predicate.bounds).iter().map(|&bound| (ty, bound)));
        }
        found
    }

    /// Whether trait `of` has a member `name`.
    pub(super) fn has_member(&mut self, of: ItemRef, name: &str) -> Result<bool> {
        Ok(self.cx.trait_member(of, name).is_some())
    }

    fn has_assoc_ty(&self, of: ItemRef, name: &str) -> bool {
        self.cx.trait_member(of, name).is_some_and(|member| self.cx.item(member).1.kind == ItemKind::Type)
    }

    fn reaches_assoc_ty(&mut self, of: ItemRef, name: &str) -> Result<bool> {
        self.reaches(of, name, Self::has_assoc_ty, 0)
    }

    /// Whether trait `of` or one of its supertraits has a member `name`.
    pub(super) fn reaches_member(&mut self, of: ItemRef, name: &str) -> Result<bool> {
        self.reaches(of, name, |infer, of, name| infer.cx.trait_member(of, name).is_some(), 0)
    }

    fn reaches(&mut self, of: ItemRef, name: &str, owns: fn(&Self, ItemRef, &str) -> bool, depth: u32) -> Result<bool> {
        if owns(self, of, name) {
            return Ok(true);
        }
        if depth > SUPER_DEPTH {
            return Ok(false);
        }
        for parent in self.parent_traits(of)? {
            if self.reaches(parent, name, owns, depth + 1)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// The traits that trait `of` names as its supertraits.
    pub(super) fn parent_traits(&mut self, of: ItemRef) -> Result<Vec<ItemRef>> {
        let def = self.cx.trait_generics(of)?;
        if def.1.supers.len == 0 {
            return Ok(Vec::new());
        }
        let params: Vec<_> = type_params(&def.1.params).copied().collect();
        let args: Vec<Ty> = params.iter().map(|_| self.fresh(Kind::Var)).collect();
        let frame = self.item_frame(of, (&params, &args), (None, None))?;
        let home = frame.home;
        self.with_frame(frame, |infer| Ok(def.0.list(def.1.supers).iter().filter_map(|&bound| infer.bound_trait_at(&def.0, bound, home)).map(|(of, _)| of).collect()))
    }

    /// The associated type `name` of trait `of` at `args` for `ty`: read from the impl that selection picks, or a variable until `ty` is known.
    pub(super) fn project(&mut self, of: ItemRef, ty: Ty, args: Vec<Ty>, name: &str) -> Result<Ty> {
        self.project_own(of, ty, args, (name, &[]))
    }

    /// The generic associated type `name` of trait `of` at `args` for `ty`, at its own arguments `own`.
    pub(super) fn project_own(&mut self, of: ItemRef, ty: Ty, args: Vec<Ty>, (name, own): (&str, &[Ty])) -> Result<Ty> {
        if let Some(found) = self.picked_alias(of, ty, &args, (name, own))? {
            return Ok(found);
        }
        if !self.obligation_open(ty, &args) {
            bail!("no one impl of `{}` gives `{name}` for {}", self.cx.name(of), self.show(ty));
        }
        let result = self.fresh(Kind::Var);
        self.projections.push((of, ty, args, name.to_string(), result));
        Ok(result)
    }

    pub(in crate::sema) fn picked_alias(&mut self, of: ItemRef, ty: Ty, args: &[Ty], (name, own): (&str, &[Ty])) -> Result<Option<Ty>> {
        if self.has_params(ty) || self.any_has_params(args) || self.any_has_params(own) {
            return match self.poly.is_some() {
                true => self.poly_picked_alias(of, ty, args, (name, own)),
                false => Ok(None),
            };
        }
        if self.cx.lang(of) == Some("discriminant_kind") {
            return self.discriminant_ty(ty);
        }
        if let Some(found) = self.builtin_alias(of, ty, name)? {
            return Ok(Some(found));
        }
        let Some(selected) = self.select(of, ty, args)? else {
            return Ok(None);
        };
        let (block, impl_args) = self.impl_member(of, (ty, args), selected.clone(), name)?.map_or(selected, |(_, picked)| picked);
        let frame = self.impl_frame_of(block, &impl_args, Some(ty))?;
        match self.with_frame(frame, |infer| infer.impl_alias(block, (name, own)))? {
            Some(found) => Ok(Some(found)),
            None => bail!("the impl of `{}` for {} has no type `{name}`", self.cx.name(of), self.show(ty)),
        }
    }

    pub(super) fn builtin_alias(&mut self, of: ItemRef, ty: Ty, name: &str) -> Result<Option<Ty>> {
        if self.cx.lang(of) == Some("pointee_trait") && name == "Metadata" {
            return self.pointee_metadata(ty);
        }
        let kind = self.cx.types.kind(self.resolve(ty)).clone();
        if let Kind::Dyn(principal, args, members, _) = &kind {
            let member = self.cx.trait_member(of, name);
            let trait_args = {
                let (trait_args, mut bound) = super::super::ty::object_parts(self.cx.types.substs(*args), members);
                if let Some((_, found)) = bound.find(|(bound, _)| Some(*bound) == member) {
                    return Ok(Some(found));
                }
                trait_args.to_vec()
            };
            return self.object_super_alias(of, ty, (*principal, trait_args), name);
        }
        if self.cx.lang(of) == Some("fn_once") && name == "Output" {
            return Ok(self.callable_sig(ty)?.map(|(_, ret)| ret));
        }
        if self.cx.lang(of) == Some("future_trait")
            && name == "Output"
            && let Some(output) = self.coroutine_output(ty)
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
            Kind::Slice(_) | Kind::Prim(Primitive::Str) => Ok(Some(self.cx.types.prim(Primitive::Usize))),
            Kind::Dyn(..) => {
                let Some(item) = self.cx.lang_item("dyn_metadata") else { bail!("core has no `DynMetadata`") };
                Ok(Some(self.cx.types.adt(item, &[tail])))
            }
            _ if self.has_params(tail) => self.param_metadata(tail).map(Some),
            _ => Ok(Some(self.cx.types.unit())),
        }
    }

    pub(super) fn discriminant_ty(&mut self, ty: Ty) -> Result<Option<Ty>> {
        match *self.cx.types.kind(self.resolve(ty)) {
            Kind::Adt(item, _) if self.cx.item(item).1.kind == ItemKind::Enum => Ok(Some(self.cx.types.prim(self.cx.discriminant_primitive(item)))),
            Kind::Var(_) => Ok(None),
            Kind::Param(_) | Kind::Proj(_) if self.poly.is_some() => {
                self.poly_fallback();
                Ok(None)
            }
            _ => bail!("the discriminant type of {} is not supported yet", self.show(ty)),
        }
    }

    /// Holds the call of function `item` at `substs` to the bounds its header puts on any type parameter of its frame, its impl's included: each trait bound whose types are not known yet becomes an obligation, and each associated type a bound binds, as in `I: Iterator<Item = A>`, is tied to the bound type.
    pub(super) fn bind_bounds(&mut self, item: ItemRef, substs: &[Ty]) -> Result<()> {
        let sig = self.cx.signature(item)?;
        if sig.generics.iter().all(|generic| generic.bounds.len == 0) && sig.predicates.is_empty() {
            return Ok(());
        }
        let frame = self.frame(item, substs)?;
        let (src, home) = (frame.src, frame.home);
        let params = frame.generics.clone();
        self.with_frame(frame, |infer| {
            for (name, ty) in params {
                for bound in named_bounds(&sig.body, src, (&sig.generics, &sig.predicates), name) {
                    infer.bind_bound(&sig.body, home, (bound, ty))?;
                }
            }
            for (ty, bound) in infer.typed_predicates(&sig.body, src, &sig.predicates) {
                if infer.bound_trait_at(&sig.body, bound, home).is_some_and(|(of, _)| infer.compiler_trait(of)) {
                    continue;
                }
                infer.bind_bound(&sig.body, home, (bound, ty))?;
            }
            Ok(())
        })
    }

    /// Binds the bounds on the parameters of the inherent impl that owns `item` at its arguments in `substs` while one is unknown, as `C: Deref<Target = T>` names `T`.
    pub(super) fn bind_impl_bounds(&mut self, item: ItemRef, substs: &[Ty]) -> Result<()> {
        let Some(block) = self.cx.owner_impl(item) else { return Ok(()) };
        let count = self.impl_params(block)?;
        let Some(args) = substs.get(..count) else { return Ok(()) };
        if self.cx.impl_trait(block).is_some() || !args.iter().any(|&arg| matches!(self.cx.types.kind(self.resolve(arg)), Kind::Var(_))) {
            return Ok(());
        }
        for (of, ty, args, bindings) in self.impl_param_bounds(block, args)? {
            if self.obligation_open(ty, &args) {
                self.obligations.push((of, std::iter::once(ty).chain(args.iter().copied()).collect()));
            }
            for (name, bound) in bindings {
                self.binding_holds((of, &args), ty, (&name, bound))?;
            }
        }
        Ok(())
    }

    /// Each trait bound on a parameter of impl `block` at `args`, lowered in the impl's frame: the trait, the bounded type, the trait's arguments and the associated types it binds.
    fn impl_param_bounds(&mut self, block: ImplRef, args: &[Ty]) -> Result<Vec<(ItemRef, Ty, Vec<Ty>, Vec<(String, Ty)>)>> {
        let header = self.cx.impl_header(block)?;
        let (_, self_ty) = self.impl_frame(block, args)?;
        let frame = self.impl_frame_of(block, args, Some(self_ty))?;
        let (src, home, arena) = (frame.src, frame.home, &header.body);
        let params = frame.generics.clone();
        self.with_frame(frame, |infer| {
            let mut found = Vec::new();
            for (name, ty) in params {
                for bound in named_bounds(arena, src, (&header.generics, &header.predicates), name) {
                    let Some((of, path)) = infer.bound_trait_at(arena, bound, home).filter(|&(of, _)| !infer.compiler_trait(of) && !infer.is_callable_trait(of)) else { continue };
                    found.push((of, ty, infer.bound_args(of, arena, path, ty)?, infer.path_bindings(arena, path)?));
                }
            }
            Ok(found)
        })
    }

    fn bind_bound(&mut self, arena: &Body, home: ModRef, (bound, ty): (u32, Ty)) -> Result<()> {
        let Some((of, path)) = self.bound_trait_at(arena, bound, home) else {
            return Ok(());
        };
        if self.is_callable_trait(of) {
            return self.bind_callable(arena, (of, path), ty);
        }
        let args = self.bound_args(of, arena, path, ty)?;
        if !self.compiler_trait(of) && self.obligation_open(ty, &args) {
            self.obligations.push((of, std::iter::once(ty).chain(args.iter().copied()).collect()));
        }
        if !self.bindings_hold(arena, path, (of, ty, &args))? {
            bail!("{} does not give the associated types that a bound of `{}` binds", self.show(ty), self.cx.name(of));
        }
        self.bind_constraints((arena, home), (of, path, &args), ty)
    }

    fn bind_callable(&mut self, arena: &Body, (of, path): (ItemRef, PathId), ty: Ty) -> Result<()> {
        if angle_args(arena, path).is_some() {
            let args = self.bound_args(of, arena, path, ty)?;
            self.bindings_hold(arena, path, (of, ty, &args))?;
        }
        Ok(())
    }

    fn bind_constraints(&mut self, (arena, home): (&Body, ModRef), (of, path, args): (ItemRef, PathId, &[Ty]), ty: Ty) -> Result<()> {
        for (name, bounds) in path_constraints(arena, path) {
            let name = self.src.text(name).to_string();
            let Some((owner, owner_args)) = self.assoc_owner((of, args.to_vec()), ty, &name, 0)? else { bail!("`{}` has no associated type `{name}`", self.cx.name(of)) };
            let assoc = self.project(owner, ty, owner_args, &name)?;
            for &bound in arena.list(bounds) {
                self.bind_bound(arena, home, (bound, assoc))?;
            }
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
        Ok(self.unifies(found, bound))
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
                let Some((of, path)) = infer.bound_trait_at(&def.0, bound, home) else {
                    continue;
                };
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
    pub(in crate::sema) fn settle_projections(&mut self) -> Result<()> {
        let mut at = 0;
        while at < self.projections.len() {
            let (of, ty, args, name, result) = self.projections[at].clone();
            match self.picked_alias(of, ty, &args, (&name, &[]))? {
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
    let Some(list) = angle_args(arena, path) else { return Vec::new() };
    arena
        .list(list)
        .iter()
        .filter_map(|&at| match arena.args[at as usize] {
            crate::body::GenericArg::Constraint { name, bounds } => Some((name, bounds)),
            _ => None,
        })
        .collect()
}

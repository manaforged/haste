use anyhow::{Result, bail};

use crate::body::{Body, Bound, PathId, Ty as SyntaxTy};
use crate::names::path::Path as NamePath;
use crate::names::{Def, ItemRef, ModRef};
use crate::outline::item::ItemKind;

use super::super::impls::{ImplRef, head};
use super::super::ty::{Kind, Ty};
use super::Infer;
use super::scope::{Frame, named_args, type_params};

/// An impl that a trait obligation picked, with its type arguments.
pub type Picked = (ImplRef, Vec<Ty>);

type Vars = Vec<Option<Ty>>;

const DEPTH: u32 = 8;

/// The traits that closures, function items and function pointers implement by the compiler, with no impl to select.
const CALLABLE: [&str; 3] = ["fn", "fn_mut", "fn_once"];

const BUILTIN: [&str; 12] = ["sized", "meta_sized", "pointee_sized", "copy", "clone", "destruct", "tuple_trait", "freeze", "pointee_trait", "unsize", "sync", "unpin"];

impl<'a> Infer<'_, 'a> {
    /// The one impl of trait `of` at trait arguments `args` for `self_ty`, or none when no impl or several apply.
    pub fn select(&mut self, of: ItemRef, self_ty: Ty, args: &[Ty]) -> Result<Option<Picked>> {
        self.select_at(of, self_ty, args, 0)
    }

    fn select_at(&mut self, of: ItemRef, self_ty: Ty, args: &[Ty], depth: u32) -> Result<Option<Picked>> {
        let found = self.applicable(of, self_ty, args, depth)?;
        let Ok([(picked, vars)]) = <[_; 1]>::try_from(found) else { return Ok(None) };
        self.vars = vars;
        Ok(Some(picked))
    }

    /// Whether some impl of trait `of` at `args` may apply to `self_ty`, even when inference cannot pick one yet.
    pub fn may_implement(&mut self, of: ItemRef, self_ty: Ty, args: &[Ty]) -> Result<bool> {
        Ok(!self.applicable(of, self_ty, args, 0)?.is_empty())
    }

    fn applicable(&mut self, of: ItemRef, self_ty: Ty, args: &[Ty], depth: u32) -> Result<Vec<(Picked, Vars)>> {
        let self_ty = self.resolve(self_ty);
        if depth > DEPTH || matches!(self.cx.types.kind(self_ty), Kind::Var(_)) {
            return Ok(Vec::new());
        }
        let holders = self.holders(of, self_ty, args);
        let blocks = self.cx.trait_impls(of, head(self.cx.types.kind(self_ty)), &holders);
        let mut found = Vec::new();
        for block in blocks {
            let saved = self.vars.clone();
            let applies = self.impl_applies(block, self_ty, args, depth);
            let vars = std::mem::replace(&mut self.vars, saved);
            if let Some(impl_args) = applies? {
                found.push(((block, impl_args), vars));
            }
        }
        if found.len() > 1 {
            let mut kept = Vec::new();
            for entry in &found {
                if !self.blanket(entry.0.0)? {
                    kept.push(entry.clone());
                }
            }
            if !kept.is_empty() {
                found = kept;
            }
        }
        if found.len() > 1 {
            found = self.most_specialized(found)?;
        }
        Ok(found)
    }

    fn most_specialized(&mut self, found: Vec<(Picked, Vars)>) -> Result<Vec<(Picked, Vars)>> {
        let mut facts = Vec::new();
        for ((block, impl_args), _) in &found {
            let direct = self.impl_bound_traits(*block, impl_args)?;
            let implied = self.implied_traits(direct.clone())?;
            facts.push(Specialty { default: self.cx.specializable(*block), direct, implied });
        }
        let kept = (0..found.len()).map(|at| !facts.iter().enumerate().any(|(by, other)| by != at && other.specializes(&facts[at])));
        Ok(found.into_iter().zip(kept.collect::<Vec<_>>()).filter_map(|(entry, keep)| keep.then_some(entry)).collect())
    }

    fn impl_bound_traits(&mut self, block: ImplRef, impl_args: &[Ty]) -> Result<Vec<ItemRef>> {
        let header = self.cx.impl_header(block)?;
        let frame = self.impl_frame_of(block, impl_args, None)?;
        let home = frame.home;
        let on_params = type_params(&header.generics).flat_map(|generic| header.body.list(generic.bounds).iter().copied());
        let bounds: Vec<u32> = on_params.chain(header.predicates.iter().flat_map(|predicate| header.body.list(predicate.bounds).iter().copied())).collect();
        self.with_frame(frame, |infer| Ok(bounds.into_iter().filter_map(|bound| infer.bound_trait_at(&header.body, bound, home)).map(|(of, _)| of).collect()))
    }

    /// The traits in `stack` with every supertrait they imply.
    pub(crate) fn implied_traits(&mut self, mut stack: Vec<ItemRef>) -> Result<Vec<ItemRef>> {
        let mut found = Vec::new();
        while let Some(of) = stack.pop() {
            if !found.contains(&of) {
                found.push(of);
                stack.extend(self.parent_traits(of)?);
            }
        }
        Ok(found)
    }
    fn blanket(&mut self, block: ImplRef) -> Result<bool> {
        let count = self.impl_params(block)?;
        let args: Vec<Ty> = (0..count).map(|_| self.fresh(Kind::Var)).collect();
        let (_, self_ty) = self.impl_frame(block, &args)?;
        Ok(args.contains(&self_ty))
    }

    fn holders(&mut self, of: ItemRef, self_ty: Ty, args: &[Ty]) -> Vec<u32> {
        let mut crates = vec![of.krate];
        let mut stack: Vec<Ty> = std::iter::once(self_ty).chain(args.iter().copied()).collect();
        while let Some(ty) = stack.pop() {
            let kind = self.cx.types.kind(self.resolve(ty));
            if let Kind::Adt(item, _) | Kind::Dyn(item, ..) = kind
                && !crates.contains(&item.krate)
            {
                crates.push(item.krate);
            }
            stack.extend(kind.children());
        }
        crates
    }

    fn impl_applies(&mut self, block: ImplRef, self_ty: Ty, args: &[Ty], depth: u32) -> Result<Option<Vec<Ty>>> {
        let Some(impl_args) = self.match_trait_impl(block, self_ty, args)? else { return Ok(None) };
        Ok(self.bounds_hold(block, &impl_args, depth)?.then_some(impl_args))
    }

    fn match_trait_impl(&mut self, block: ImplRef, self_ty: Ty, args: &[Ty]) -> Result<Option<Vec<Ty>>> {
        let Some(impl_args) = self.match_impl(block, self_ty) else { return Ok(None) };
        let given = self.impl_trait_args(block, &impl_args, self_ty)?;
        if given.len() != args.len() {
            bail!("an impl gives {} trait arguments where {} are wanted", given.len(), args.len());
        }
        for (found, wanted) in given.iter().zip(args) {
            if self.unify(*found, *wanted).is_err() {
                return Ok(None);
            }
        }
        Ok(Some(impl_args))
    }

    pub(super) fn impl_frame_of(&mut self, block: ImplRef, impl_args: &[Ty], self_ty: Option<Ty>) -> Result<Frame<'a>> {
        let header = self.cx.impl_header(block)?;
        let file = self.cx.impl_file(block);
        let generics = named_args(&file.source, &header.generics, impl_args);
        Ok(Frame { src: &file.source, home: self.cx.impl_module(block)?, generics, self_ty, block: Some(block), item: None, root: block.root() })
    }

    /// The trait arguments of impl `block` at `impl_args`, with the trait's defaults filled in.
    pub fn impl_trait_args(&mut self, block: ImplRef, impl_args: &[Ty], self_ty: Ty) -> Result<Vec<Ty>> {
        let header = self.cx.impl_header(block)?;
        let Some(SyntaxTy::Path(path)) = header.trait_ref.map(|at| *header.body.ty(at)) else { bail!("impl has no trait") };
        let Some(of) = self.cx.impl_trait(block) else { bail!("impl of something that is not a trait") };
        let frame = self.impl_frame_of(block, impl_args, Some(self_ty))?;
        self.with_frame(frame, |infer| infer.bound_args(of, &header.body, path, self_ty))
    }

    /// The arguments that the bound path `path` gives trait `of` for `ty`, completed with the trait's defaults.
    pub(super) fn bound_args(&mut self, of: ItemRef, arena: &Body, path: PathId, ty: Ty) -> Result<Vec<Ty>> {
        let count = arena.list(arena.path(path).segments).len();
        let given = self.segment_args(arena, path, count)?;
        self.trait_args(of, given, ty)
    }

    /// The arguments `given` to trait `of` for `self_ty`, completed with the trait's defaults.
    pub fn trait_args(&mut self, of: ItemRef, mut given: Vec<Ty>, self_ty: Ty) -> Result<Vec<Ty>> {
        let def = self.cx.trait_generics(of)?;
        let params: Vec<_> = type_params(&def.1.params).copied().collect();
        for param in params.iter().skip(given.len()) {
            let Some(default) = param.default else { bail!("`{}` needs {} type arguments", self.cx.name(of), params.len()) };
            let frame = self.item_frame(of, (&params, &given), (Some(self_ty), None))?;
            given.push(self.with_frame(frame, |infer| infer.lower_ty(&def.0, default))?);
        }
        Ok(given)
    }

    fn bounds_hold(&mut self, block: ImplRef, impl_args: &[Ty], depth: u32) -> Result<bool> {
        let header = self.cx.impl_header(block)?;
        let frame = self.impl_frame_of(block, impl_args, None)?;
        let home = frame.home;
        let mut wanted = Vec::new();
        for (generic, &arg) in type_params(&header.generics).zip(impl_args) {
            wanted.extend(header.body.list(generic.bounds).iter().map(|&bound| (arg, bound)));
        }
        self.with_frame(frame, |infer| {
            for (ty, bound) in wanted {
                if !infer.bound_holds(&header.body, home, (ty, bound), depth)? {
                    return Ok(false);
                }
            }
            for predicate in &header.predicates {
                let Some(ty) = predicate.ty else { continue };
                let ty = infer.lower_ty(&header.body, ty)?;
                for &bound in header.body.list(predicate.bounds) {
                    if !infer.bound_holds(&header.body, home, (ty, bound), depth)? {
                        return Ok(false);
                    }
                }
            }
            Ok(true)
        })
    }

    fn bound_holds(&mut self, arena: &Body, home: ModRef, (ty, bound): (Ty, u32), depth: u32) -> Result<bool> {
        let Some((of, path)) = self.bound_trait_at(arena, bound, home) else { return Ok(true) };
        if self.compiler_trait(of) {
            return Ok(true);
        }
        let kind = self.cx.types.kind(self.resolve(ty));
        if matches!(kind, Kind::Var(_) | Kind::IntVar(_) | Kind::FloatVar(_)) {
            return Ok(true);
        }
        if self.is_callable_trait(of) {
            if !matches!(kind, Kind::Closure(..) | Kind::FnDef(..) | Kind::FnPtr(..)) {
                return Ok(false);
            }
            let Some(sig) = self.fn_bound(arena, home, bound)? else { return Ok(true) };
            return Ok(self.tie_callable(ty, sig).is_ok());
        }
        let args = self.bound_args(of, arena, path, ty)?;
        if let Some(held) = self.object_holds(of, ty, &args) {
            return Ok(held);
        }
        Ok(self.select_at(of, ty, &args, depth + 1)?.is_some() && self.bindings_hold(arena, path, (of, ty, &args))?)
    }

    /// Whether the trait object `ty` implements trait `of` at `args`, by being an object of it, or none when `ty` is not a trait object.
    pub(super) fn object_holds(&mut self, of: ItemRef, ty: Ty, args: &[Ty]) -> Option<bool> {
        let Kind::Dyn(object, parts, members, _) = self.cx.types.kind(self.resolve(ty)).clone() else { return None };
        let given = super::super::ty::object_parts(&parts, &members).0;
        if object != of {
            return Some(args.is_empty() && self.implied_traits(vec![object]).is_ok_and(|traits| traits.contains(&of)));
        }
        if given.len() != args.len() {
            return Some(false);
        }
        let held = self.trial(|infer| Ok(given.iter().zip(args).all(|(found, wanted)| infer.unify(*found, *wanted).is_ok()).then_some(())));
        Some(matches!(held, Ok(Some(()))))
    }

    /// Whether trait `of` is one the compiler proves for every type that reaches selection: a marker such as `Sized` or `Copy`, or an auto trait.
    pub(super) fn compiler_trait(&self, of: ItemRef) -> bool {
        self.cx.lang(of).is_some_and(|lang| BUILTIN.contains(&lang)) || self.cx.is_auto_trait(of)
    }

    /// Whether trait `of` is `Fn`, `FnMut` or `FnOnce`, which callables implement with no impl to select.
    pub(super) fn is_callable_trait(&self, of: ItemRef) -> bool {
        self.cx.lang(of).is_some_and(|lang| CALLABLE.contains(&lang))
    }

    /// The trait that bound `bound` in `arena` requires from module `home`, with its path, or none for a lifetime, a `?Sized` or a name that is not a trait.
    pub(super) fn bound_trait_at(&mut self, arena: &Body, bound: u32, home: ModRef) -> Option<(ItemRef, PathId)> {
        let Bound::Trait { path, maybe: false, .. } = arena.bounds[bound as usize] else { return None };
        Some((self.trait_named(arena, path, home)?, path))
    }

    /// The trait that `path` in `arena` names from module `home`.
    pub fn trait_named(&mut self, arena: &Body, path: PathId, home: ModRef) -> Option<ItemRef> {
        let (segments, absolute, dollar) = self.segments(arena, path);
        self.trait_at(&NamePath { segments: &segments, absolute, dollar }, home)
    }

    /// The trait that `path` names from module `home`, through the body's own items and `use` declarations when `home` is the frame's module.
    pub(super) fn trait_at(&mut self, path: &NamePath<'_>, home: ModRef) -> Option<ItemRef> {
        let def = if home == self.home { self.type_def(path) } else { self.cx.paths(home.krate).resolve(path, home) };
        match def? {
            Def::Item(item) if self.cx.item(item).1.kind == ItemKind::Trait => Some(item),
            _ => None,
        }
    }
}

/// What ranks one applicable impl against another under specialization: whether it has `default` items, the traits its bounds name, and those with every supertrait.
struct Specialty {
    default: bool,
    direct: Vec<ItemRef>,
    implied: Vec<ItemRef>,
}

impl Specialty {
    /// Whether this impl specializes `other`: `other` has `default` items, and this one has none or its bounds imply the bounds of `other` and not the reverse.
    fn specializes(&self, other: &Specialty) -> bool {
        other.default && (!self.default || (self.covers(other) && !other.covers(self)))
    }

    fn covers(&self, other: &Specialty) -> bool {
        other.direct.iter().all(|of| self.implied.contains(of))
    }
}

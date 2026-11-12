use std::hash::{Hash, Hasher};

use anyhow::{Result, bail};

use crate::body::{Body, Bound, PathId, Ty as SyntaxTy};
use crate::names::path::Path as NamePath;
use crate::names::{Def, ItemRef, ModRef};
use crate::outline::item::ItemKind;

use super::super::impls::{ImplRef, head};
use super::super::substs::Substs;
use super::super::ty::{Kind, Mut, Ty};
use super::Infer;
use super::scope::{Frame, named_args, type_params};

/// An impl that a trait obligation picked, with its type arguments.
pub type Picked = (ImplRef, Vec<Ty>);

/// What trying one impl left: its variable bindings, and the projections and obligations it deferred.
#[derive(Clone)]
struct Vars {
    bound: Vec<Option<Ty>>,
    projections: Vec<super::project::Projection>,
    obligations: Vec<(ItemRef, Vec<Ty>)>,
}

/// The traits that closures, function items and function pointers implement by the compiler, with no impl to select.
pub(super) const CALLABLE: [&str; 3] = ["fn", "fn_mut", "fn_once"];

const BUILTIN: [&str; 12] = ["sized", "meta_sized", "pointee_sized", "copy", "clone", "destruct", "tuple_trait", "freeze", "pointee_trait", "unsize", "sync", "unpin"];

/// The sizedness traits weaker than `Sized`, whose bound on a parameter drops its implicit `Sized` bound.
const WEAKER_THAN_SIZED: [&str; 2] = ["meta_sized", "pointee_sized"];

/// The compiler traits that no unsized type implements.
const SIZED_ONLY: [&str; 3] = ["sized", "copy", "clone"];

impl<'a> Infer<'_, 'a> {
    /// The one impl of trait `of` at trait arguments `args` for `self_ty`, or none when no impl or several apply.
    pub fn select(&mut self, of: ItemRef, self_ty: Ty, args: &[Ty]) -> Result<Option<Picked>> {
        let key = self.selection_key(of, self_ty, args);
        if let Some(found) = key.as_ref().and_then(|key| self.cx.store.lookups.selections.get(key)) {
            return Ok(found.map(|(block, impl_args)| (block, self.cx.types.substs(impl_args).to_vec())));
        }
        let before = self.unsettled();
        let picked = self.select_at(of, self_ty, args, 0)?;
        let Some(key) = key else { return Ok(picked) };
        let settled = self.unsettled() == before;
        let kept = match &picked {
            None => Some(None),
            Some((block, impl_args)) => {
                let filled: Vec<Ty> = impl_args.iter().map(|&ty| self.substitute(ty)).collect();
                filled.iter().all(|&ty| self.closed(ty)).then(|| Some((*block, self.cx.types.intern_substs(&filled))))
            }
        };
        if let Some(kept) = kept.filter(|_| settled) {
            self.cx.store.lookups.selections.insert(key, kept);
        }
        Ok(picked)
    }

    /// The key that caches selecting `of<args>` for `self_ty` across bodies: the query with its variables filled in, when it is made at the top level and every type in it is known and holds no closure.
    fn selection_key(&mut self, of: ItemRef, self_ty: Ty, args: &[Ty]) -> Option<(ItemRef, Ty, Substs)> {
        if !self.selecting.is_empty() || !self.closed(self_ty) || !args.iter().all(|&arg| self.closed(arg)) {
            return None;
        }
        let substituted: Vec<Ty> = args.iter().map(|&arg| self.substitute(arg)).collect();
        Some((of, self.substitute(self_ty), self.cx.types.intern_substs(&substituted)))
    }

    /// Whether `ty` holds no inference variable and no closure, whose signature may still hold one, so selecting with it gives the same answer in every body.
    fn closed(&self, ty: Ty) -> bool {
        let ty = self.resolve(ty);
        if self.has_params(ty) {
            return false;
        }
        let kind = self.cx.types.kind(ty);
        !matches!(kind, Kind::Var(_) | Kind::IntVar(_) | Kind::FloatVar(_) | Kind::Closure(..)) && kind.children(&self.cx.types).all(|item| self.closed(item))
    }

    fn select_at(&mut self, of: ItemRef, self_ty: Ty, args: &[Ty], depth: u32) -> Result<Option<Picked>> {
        let found = self.applicable(of, self_ty, args, depth)?;
        let Ok([(picked, vars)]) = <[_; 1]>::try_from(found) else {
            return Ok(None);
        };
        self.adopt(vars);
        Ok(Some(picked))
    }

    /// Whether some impl of trait `of` at `args` may apply to `self_ty`, even when inference cannot pick one yet.
    pub fn may_implement(&mut self, of: ItemRef, self_ty: Ty, args: &[Ty]) -> Result<bool> {
        Ok(!self.applicable(of, self_ty, args, 0)?.is_empty() || self.poly_may_implement(of, self_ty, args)?)
    }

    fn applicable(&mut self, of: ItemRef, self_ty: Ty, args: &[Ty], depth: u32) -> Result<Vec<(Picked, Vars)>> {
        let self_ty = self.resolve(self_ty);
        if depth as usize > crate::RECURSION_LIMIT || matches!(self.cx.types.kind(self_ty), Kind::Var(_)) {
            return Ok(Vec::new());
        }
        let mut found = self.impl_candidates(of, self_ty, args, depth)?;
        if found.len() > 1 && !self.obligation_open(self_ty, args) {
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
        if found.len() > 1 && self.same_trait_args(&found, args) {
            found = self.most_specialized(found)?;
        }
        Ok(found)
    }

    fn impl_candidates(&mut self, of: ItemRef, self_ty: Ty, args: &[Ty], depth: u32) -> Result<Vec<(Picked, Vars)>> {
        let holders = self.holders(of, self_ty, args);
        let blocks = self.cx.trait_impls(of, head(self.cx.types.kind(self_ty)), &holders);
        let mut found = Vec::new();
        let blocks: Vec<ImplRef> = blocks.into_iter().filter(|&block| !self.cx.reserved(block)).collect();
        for block in blocks {
            let (saved, deferred) = (self.vars.clone(), (self.projections.len(), self.obligations.len()));
            let applies = self.impl_applies(block, self_ty, args, depth);
            let bound = std::mem::replace(&mut self.vars, saved);
            let projections = self.projections.split_off(deferred.0.min(self.projections.len()));
            let obligations = self.obligations.split_off(deferred.1.min(self.obligations.len()));
            if let Some(impl_args) = applies? {
                found.push(((block, impl_args), Vars { bound, projections, obligations }));
            }
        }
        Ok(found)
    }

    /// Takes on the bindings and deferred work of the impl trial `vars`.
    fn adopt(&mut self, vars: Vars) {
        self.vars = vars.bound;
        self.projections.extend(vars.projections);
        self.obligations.extend(vars.obligations);
    }

    /// The most specialized impl of trait `of` at `args` for `self_ty` that defines the member `name`, among those that apply, as a specializing impl inherits it from the impl it specializes.
    fn defining_impl(&mut self, of: ItemRef, self_ty: Ty, args: &[Ty], name: &str) -> Result<Option<Picked>> {
        let self_ty = self.resolve(self_ty);
        let mut found = self.impl_candidates(of, self_ty, args, 0)?;
        found.retain(|((block, _), _)| self.cx.member(*block, name).is_some());
        if found.len() > 1 {
            found = self.most_specialized(found)?;
        }
        let Some((picked, vars)) = found.into_iter().next() else {
            return Ok(None);
        };
        self.adopt(vars);
        Ok(Some(picked))
    }

    /// The member `name` of the selected impl `picked` of trait `of`, or else of the impl it specializes, with the impl that defines it.
    pub(super) fn impl_member(&mut self, of: ItemRef, (self_ty, args): (Ty, &[Ty]), picked: Picked, name: &str) -> Result<Option<(ItemRef, Picked)>> {
        if let Some(member) = self.cx.member(picked.0, name) {
            return Ok(Some((member, picked)));
        }
        let Some(general) = self.defining_impl(of, self_ty, args, name)? else {
            return Ok(None);
        };
        Ok(self.cx.member(general.0, name).map(|member| (member, general)))
    }

    /// Whether every impl in `found` sets the trait arguments `args` alike, so that specialization, not inference, picks among them.
    fn same_trait_args(&mut self, found: &[(Picked, Vars)], args: &[Ty]) -> bool {
        let saved = self.vars.clone();
        let mut each = found.iter().map(|(_, vars)| {
            self.vars = vars.bound.clone();
            args.iter().map(|&arg| self.substitute(arg)).collect::<Vec<Ty>>()
        });
        let first = each.next();
        let same = each.all(|other| Some(&other) == first.as_ref());
        self.vars = saved;
        same
    }

    fn most_specialized(&mut self, found: Vec<(Picked, Vars)>) -> Result<Vec<(Picked, Vars)>> {
        let mut facts = Vec::new();
        for ((block, impl_args), _) in &found {
            let direct = self.impl_bound_traits(*block, impl_args)?;
            let implied = self.implied_traits(direct.clone())?;
            facts.push(Specialty { default: self.cx.specializable(*block), direct, implied });
        }
        let blocks: Vec<ImplRef> = found.iter().map(|((block, _), _)| *block).collect();
        let mut kept = Vec::new();
        for at in 0..found.len() {
            kept.push(!self.specialized(&blocks, &facts, at)?);
        }
        Ok(found.into_iter().zip(kept).filter_map(|(entry, keep)| keep.then_some(entry)).collect())
    }

    /// Whether another impl in `blocks` specializes impl `at`: its specialty ranks above, and its header is an instance of the header of `at`.
    fn specialized(&mut self, blocks: &[ImplRef], facts: &[Specialty], at: usize) -> Result<bool> {
        for (by, (&block, fact)) in blocks.iter().zip(facts).enumerate() {
            if by != at && fact.specializes(&facts[at]) && self.instance_of(block, blocks[at])? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Whether the header of impl `special`, its self type and trait arguments, is an instance of the header of impl `general`: some arguments of `general` make it equal while every type parameter of `special` stays distinct and unbound, as rustc requires of an impl that specializes another.
    fn instance_of(&mut self, special: ImplRef, general: ImplRef) -> Result<bool> {
        let deferred = (self.projections.len(), self.obligations.len());
        let found: Result<bool> = self.probe(|infer| {
            let (params, own) = infer.fresh_header(special)?;
            let (_, other) = infer.fresh_header(general)?;
            let unified = own.len() == other.len() && own.iter().zip(&other).all(|(&a, &b)| infer.unifies(a, b));
            let roots: Vec<Ty> = params.iter().map(|&param| infer.resolve(param)).collect();
            let rigid = roots.iter().enumerate().all(|(at, root)| matches!(infer.cx.types.kind(*root), Kind::Var(_)) && !roots[..at].contains(root));
            Ok(unified && rigid)
        });
        self.projections.truncate(deferred.0);
        self.obligations.truncate(deferred.1);
        found
    }

    /// Fresh type arguments for impl `block`, with its self type and then its trait arguments at them.
    fn fresh_header(&mut self, block: ImplRef) -> Result<(Vec<Ty>, Vec<Ty>)> {
        let (args, self_ty) = self.fresh_impl(block)?;
        let header = std::iter::once(self_ty).chain(self.impl_trait_args(block, &args, self_ty)?).collect();
        Ok((args, header))
    }

    /// Fresh type arguments for impl `block`, with its self type at them.
    fn fresh_impl(&mut self, block: ImplRef) -> Result<(Vec<Ty>, Ty)> {
        let count = self.impl_params(block)?;
        let args: Vec<Ty> = (0..count).map(|_| self.fresh(Kind::Var)).collect();
        let (_, self_ty) = self.impl_frame(block, &args)?;
        Ok((args, self_ty))
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
        let (args, self_ty) = self.fresh_impl(block)?;
        Ok(args.contains(&self_ty))
    }

    /// The crates whose impls of `of` may apply to `self_ty` at `args`: the trait's, each named type's, and the body's own while a type is still unknown, since an impl there may name the type it becomes.
    fn holders(&mut self, of: ItemRef, self_ty: Ty, args: &[Ty]) -> Vec<u32> {
        let mut crates = vec![of.krate];
        let mut stack: Vec<Ty> = std::iter::once(self_ty).chain(args.iter().copied()).collect();
        while let Some(ty) = stack.pop() {
            let kind = self.cx.types.kind(self.resolve(ty));
            let named = match kind {
                Kind::Adt(item, _) | Kind::Dyn(item, ..) => Some(item.krate),
                Kind::Var(_) => Some(self.home.krate),
                _ => None,
            };
            if let Some(krate) = named.filter(|krate| !crates.contains(krate)) {
                crates.push(krate);
            }
            stack.extend(kind.children(&self.cx.types));
        }
        crates
    }

    fn impl_applies(&mut self, block: ImplRef, self_ty: Ty, args: &[Ty], depth: u32) -> Result<Option<Vec<Ty>>> {
        let Some(impl_args) = self.match_trait_impl(block, self_ty, args)? else {
            return Ok(None);
        };
        Ok(self.bounds_hold(block, &impl_args, depth)?.then_some(impl_args))
    }

    fn match_trait_impl(&mut self, block: ImplRef, self_ty: Ty, args: &[Ty]) -> Result<Option<Vec<Ty>>> {
        let Some(impl_args) = self.match_impl(block, self_ty) else {
            return Ok(None);
        };
        let given = self.impl_trait_args(block, &impl_args, self_ty)?;
        if given.len() != args.len() {
            bail!("an impl gives {} trait arguments where {} are wanted", given.len(), args.len());
        }
        for (found, wanted) in given.iter().zip(args) {
            if !self.unifies(*found, *wanted) {
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
    pub fn trait_args(&mut self, of: ItemRef, given: Vec<Ty>, self_ty: Ty) -> Result<Vec<Ty>> {
        self.fill_trait_args(of, given, self_ty, false)
    }

    pub(super) fn fill_trait_args(&mut self, of: ItemRef, mut given: Vec<Ty>, self_ty: Ty, fresh: bool) -> Result<Vec<Ty>> {
        let def = self.cx.trait_generics(of)?;
        let params: Vec<_> = type_params(&def.1.params).copied().collect();
        if given.len() > params.len() {
            bail!("`{}` takes {} type arguments, {} given", self.cx.name(of), params.len(), given.len());
        }
        for param in params.iter().skip(given.len()) {
            let arg = match param.default {
                Some(default) => {
                    let frame = self.item_frame(of, (&params, &given), (Some(self_ty), None))?;
                    self.with_frame(frame, |infer| infer.ty_or_const_arg(&def.0, default))?
                }
                None if fresh => self.fresh(Kind::Var),
                None => bail!("`{}` needs {} type arguments", self.cx.name(of), params.len()),
            };
            given.push(arg);
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
            if infer.sets_unsized_param(block, &header, (impl_args, home))? {
                return Ok(false);
            }
            infer.self_ty = Some(infer.lower_ty(&header.body, header.self_ty)?);
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

    /// Whether impl `block` at `impl_args` sets a type parameter to a slice, `str` or trait object while the parameter keeps its implicit `Sized` bound, which rules the impl out.
    fn sets_unsized_param(&mut self, block: ImplRef, header: &crate::body::sig::ImplHeader, (impl_args, home): (&[Ty], ModRef)) -> Result<bool> {
        let src = &self.cx.impl_file(block).source;
        for (generic, &arg) in type_params(&header.generics).zip(impl_args) {
            if generic.kind != crate::body::sig::GenericKind::Type || !self.cx.types.kind(self.resolve(arg)).is_unsized() {
                continue;
            }
            let bounds = super::scope::generic_bounds(&header.body, src, &header.predicates, generic);
            if !bounds.into_iter().any(|bound| self.relaxes_sized(&header.body, bound, home)) {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Whether `bound` drops the implicit `Sized` bound of the parameter it is on: `?Sized`, or a bound on a sizedness trait weaker than `Sized`.
    fn relaxes_sized(&mut self, arena: &Body, bound: u32, home: ModRef) -> bool {
        matches!(arena.bounds[bound as usize], Bound::Trait { maybe: true, .. }) || self.bound_trait_at(arena, bound, home).and_then(|(of, _)| self.cx.lang(of)).is_some_and(|lang| WEAKER_THAN_SIZED.contains(&lang))
    }

    fn bound_holds(&mut self, arena: &Body, home: ModRef, (ty, bound): (Ty, u32), depth: u32) -> Result<bool> {
        let Some((of, path)) = self.bound_trait_at(arena, bound, home) else {
            return Ok(true);
        };
        if self.compiler_decides(of, ty) {
            return Ok(self.compiler_holds(of, ty));
        }
        let kind = self.cx.types.kind(self.resolve(ty));
        if matches!(kind, Kind::Var(_) | Kind::IntVar(_) | Kind::FloatVar(_)) {
            return self.var_holds(arena, (of, path), ty, depth);
        }
        if self.compiler_impl(of, ty) {
            let args = self.bound_args(of, arena, path, ty)?;
            return self.assoc_bounds_hold((arena, home, path), (of, ty, &args), depth);
        }
        if self.is_callable_trait(of) && matches!(kind, Kind::Closure(..) | Kind::FnDef(..) | Kind::FnPtr(..)) {
            let Some(sig) = self.fn_bound(arena, home, bound)? else {
                return Ok(true);
            };
            return Ok(self.tie_callable(ty, sig).is_ok());
        }
        if self.poly.is_some()
            && let Some(held) = self.poly_bound_check((arena, home, bound), (of, path, ty), depth)?
        {
            return Ok(held);
        }
        let args = self.bound_args(of, arena, path, ty)?;
        if self.object_holds(of, ty, &args) == Some(true) {
            return Ok(true);
        }
        self.impl_holds((arena, home, path), (of, ty, &args), depth)
    }

    /// Whether the compiler, not an impl, decides that `ty` implements trait `of`: for its marker and auto traits, except `Copy` and `Clone` of a struct, enum or union, which derives and std write impls for.
    pub(in crate::sema) fn compiler_decides(&self, of: ItemRef, ty: Ty) -> bool {
        let by_impl = matches!(self.cx.types.kind(self.resolve(ty)), Kind::Adt(..) | Kind::Param(_) | Kind::Proj(_)) && self.cx.lang(of).is_some_and(|lang| matches!(lang, "copy" | "clone"));
        self.compiler_trait(of) && !by_impl
    }

    /// Whether an impl proves `ty: of<args>` for the bound at `path`, where an obligation that repeats an enclosing one is a cycle.
    fn impl_holds(&mut self, (arena, home, path): (&Body, ModRef, PathId), (of, ty, args): (ItemRef, Ty, &[Ty]), depth: u32) -> Result<bool> {
        let query = self.query(of, ty, args);
        if self.selecting.contains(&query) {
            return Ok(self.obligation_open(ty, args));
        }
        self.selecting.push(query);
        let found = self.applicable(of, ty, args, depth + 1);
        self.selecting.pop();
        match <[_; 1]>::try_from(found?) {
            Ok([(_, vars)]) => {
                self.adopt(vars);
                self.assoc_bounds_hold((arena, home, path), (of, ty, args), depth)
            }
            Err(found) => Ok(!found.is_empty()),
        }
    }

    /// Whether `ty` holds an unbound inference variable.
    pub(in crate::sema) fn has_vars(&self, ty: Ty) -> bool {
        if !self.cx.types.has_var_flag(ty) {
            return false;
        }
        let mut seen = rustc_hash::FxHashSet::default();
        self.has_vars_seen(ty, &mut seen)
    }

    pub(in crate::sema) fn has_vars_seen(&self, ty: Ty, seen: &mut rustc_hash::FxHashSet<Ty>) -> bool {
        let ty = self.resolve(ty);
        if !self.cx.types.has_var_flag(ty) || !seen.insert(ty) {
            return false;
        }
        let kind = self.cx.types.kind(ty);
        matches!(kind, Kind::Var(_) | Kind::IntVar(_) | Kind::FloatVar(_)) || kind.children(&self.cx.types).any(|item| self.has_vars_seen(item, seen))
    }

    pub(in crate::sema) fn has_params(&self, ty: Ty) -> bool {
        if !self.cx.types.has_var_flag(ty) {
            return self.cx.types.has_params(ty);
        }
        let mut seen = rustc_hash::FxHashSet::default();
        self.has_params_seen(ty, &mut seen)
    }

    fn has_params_seen(&self, ty: Ty, seen: &mut rustc_hash::FxHashSet<Ty>) -> bool {
        let ty = self.resolve(ty);
        if self.cx.types.has_params(ty) {
            return true;
        }
        if !self.cx.types.has_var_flag(ty) || !seen.insert(ty) {
            return false;
        }
        match self.cx.types.kind(ty) {
            Kind::Closure(_, args, _) => self.cx.types.substs(*args).iter().any(|&item| self.has_params_seen(item, seen)),
            kind => kind.children(&self.cx.types).any(|item| self.has_params_seen(item, seen)),
        }
    }

    pub(in crate::sema) fn any_has_params(&self, tys: &[Ty]) -> bool {
        tys.iter().any(|&ty| self.has_params(ty))
    }

    pub(in crate::sema) fn substs_have_params(&self, substs: Substs) -> bool {
        self.any_has_params(self.cx.types.substs(substs))
    }

    /// Whether the obligation `ty: _<args>` holds an unbound inference variable.
    pub(super) fn obligation_open(&self, ty: Ty, args: &[Ty]) -> bool {
        self.has_vars(ty) || args.iter().any(|&arg| self.has_vars(arg))
    }

    /// The obligation `ty: of<args>` hashed with every unbound inference variable of a kind alike, so a nested obligation that repeats an enclosing one is a cycle.
    fn query(&self, of: ItemRef, ty: Ty, args: &[Ty]) -> u64 {
        let mut hasher = rustc_hash::FxHasher::default();
        of.hash(&mut hasher);
        self.hash_canonical(ty, &mut hasher);
        self.hash_list(args, &mut hasher);
        hasher.finish()
    }

    fn hash_canonical(&self, ty: Ty, hasher: &mut rustc_hash::FxHasher) {
        let kind = self.cx.types.kind(self.resolve(ty));
        std::mem::discriminant(kind).hash(hasher);
        match kind {
            Kind::Var(_) | Kind::IntVar(_) | Kind::FloatVar(_) | Kind::Never => {}
            Kind::Prim(prim) => prim.hash(hasher),
            Kind::Value(bits) => bits.hash(hasher),
            Kind::Param(index) => index.hash(hasher),
            Kind::Proj(id) => id.hash(hasher),
            Kind::Ref(mutable, inner) | Kind::Ptr(mutable, inner) => {
                mutable.hash(hasher);
                self.hash_canonical(*inner, hasher);
            }
            Kind::Slice(inner) => self.hash_canonical(*inner, hasher),
            Kind::Array(inner, len) => self.hash_list(&[*inner, *len], hasher),
            Kind::Tuple(items) => self.hash_list(self.cx.types.substs(*items), hasher),
            Kind::Adt(item, items) | Kind::FnDef(item, items) => {
                item.hash(hasher);
                self.hash_list(self.cx.types.substs(*items), hasher);
            }
            Kind::FnPtr(params, ret) => {
                self.hash_list(self.cx.types.substs(*params), hasher);
                self.hash_canonical(*ret, hasher);
            }
            Kind::Closure(item, items, id) => {
                (item, id).hash(hasher);
                self.hash_list(self.cx.types.substs(*items), hasher);
            }
            Kind::Dyn(of, items, members, autos) => {
                (of, members, autos).hash(hasher);
                self.hash_list(self.cx.types.substs(*items), hasher);
            }
            Kind::Pat(base, start, end) => {
                (start, end).hash(hasher);
                self.hash_canonical(*base, hasher);
            }
        }
    }

    fn hash_list(&self, items: &[Ty], hasher: &mut rustc_hash::FxHasher) {
        items.len().hash(hasher);
        for &item in items {
            self.hash_canonical(item, hasher);
        }
    }

    /// Whether the associated types of `of` for `ty` meet the bound path's bindings, as `Item = T`, and its constraints, as `Item: Trait`.
    pub(super) fn assoc_bounds_hold(&mut self, (arena, home, path): (&Body, ModRef, PathId), (of, ty, args): (ItemRef, Ty, &[Ty]), depth: u32) -> Result<bool> {
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
            let crate::body::GenericArg::Constraint { name, bounds } = arena.args[at as usize] else {
                continue;
            };
            let name = self.src.text(name);
            let Some((owner, owner_args)) = self.assoc_owner((of, args.to_vec()), ty, name, 0)? else { bail!("`{}` has no associated type `{name}`", self.cx.name(of)) };
            let assoc = self.project(owner, ty, owner_args, name)?;
            for &bound in arena.list(bounds) {
                if !self.bound_holds(arena, home, (assoc, bound), depth + 1)? {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    /// Whether the bound `of` at `path` can hold for the unknown type `ty`: for an integer or float variable, whether an impl for some number type applies, kept as an obligation that picks the type when one impl alone applies, as rustc selects with one; else false only when `ty` is a pending `impl Trait` value whose own bound of that trait does not unify.
    fn var_holds(&mut self, arena: &Body, (of, path): (ItemRef, PathId), ty: Ty, depth: u32) -> Result<bool> {
        let args = self.bound_args(of, arena, path, ty)?;
        let substs: Vec<Ty> = std::iter::once(ty).chain(args.iter().copied()).collect();
        if !matches!(self.cx.types.kind(self.resolve(ty)), Kind::Var(_)) {
            let held = !self.applicable(of, ty, &args, depth + 1)?.is_empty() || self.poly_may_implement(of, ty, &args)?;
            if held {
                self.obligations.push((of, substs));
            }
            return Ok(held);
        }
        let held = self.opaque_holds(of, &substs)?;
        if held.is_none() {
            self.obligations.push((of, substs));
        }
        Ok(held != Some(false))
    }

    /// Whether the one impl of `of<args>` that the function pointer of the `Fn` bound on the pending `impl Fn` value `ty` selects applies, tying `args` to that signature as an impl over any `F: Fn` does for the value it reveals.
    pub(super) fn callable_opaque_holds(&mut self, of: ItemRef, ty: Ty, args: &[Ty]) -> Result<bool> {
        let Some(sig) = self.opaque_sig(self.resolve(ty)) else { return Ok(false) };
        let Ok([(_, vars)]) = <[_; 1]>::try_from(self.applicable(of, sig, args, 0)?) else { return Ok(false) };
        self.adopt(vars);
        Ok(true)
    }

    /// Whether the trait object `ty` implements trait `of` at `args` by being an object of it, or none when `ty` is not a trait object; an impl written for the object type can still hold when this is false.
    pub(in crate::sema) fn object_holds(&mut self, of: ItemRef, ty: Ty, args: &[Ty]) -> Option<bool> {
        let Kind::Dyn(object, parts, members, _) = self.cx.types.kind(self.resolve(ty)).clone() else { return None };
        let given = super::super::ty::object_parts(self.cx.types.substs(parts), &members).0.to_vec();
        if object != of && self.is_callable_trait(object) && self.is_callable_trait(of) {
            if !self.implied_traits(vec![object]).is_ok_and(|traits| traits.contains(&of)) || given.len() != args.len() {
                return Some(false);
            }
            let held = self.trial(|infer| Ok(given.iter().zip(args).all(|(found, wanted)| infer.unifies(*found, *wanted)).then_some(())));
            return Some(matches!(held, Ok(Some(()))));
        }
        let bounds = self.expand_super_bounds(vec![(object, std::iter::once(ty).chain(given).collect())]).ok()?;
        let matching: Vec<Vec<Ty>> = bounds.into_iter().filter(|(bound, substs)| *bound == of && substs.len() == args.len() + 1).map(|(_, substs)| substs).collect();
        if matching.is_empty() {
            return Some(false);
        }
        if matching.len() == 1 {
            let saved = self.vars.clone();
            if matching[0][1..].iter().zip(args).all(|(&found, &wanted)| self.unifies(found, wanted)) {
                return Some(true);
            }
            self.vars = saved;
            return Some(false);
        }
        let held = matching.iter().any(|substs| matches!(self.trial(|infer| Ok(substs[1..].iter().zip(args).all(|(&found, &wanted)| infer.unifies(found, wanted)).then_some(()))), Ok(Some(()))));
        Some(held)
    }

    /// Whether trait `of` is one the compiler proves for every type that reaches selection: a marker such as `Sized` or `Copy`, or an auto trait.
    pub(super) fn compiler_trait(&self, of: ItemRef) -> bool {
        self.cx.lang(of).is_some_and(|lang| BUILTIN.contains(&lang)) || self.cx.is_auto_trait(of)
    }

    /// Whether the compiler trait `of` holds for `ty`: for every type, except that an unsized one, such as a struct that ends in a slice, is neither `Sized`, `Copy` nor `Clone`, and `&mut T` is neither `Copy` nor `Clone`.
    pub(in crate::sema) fn compiler_holds(&mut self, of: ItemRef, ty: Ty) -> bool {
        let Some(lang) = self.cx.lang(of) else { return true };
        let ty = self.resolve(ty);
        match self.cx.types.kind(ty) {
            Kind::Slice(_) | Kind::Prim(crate::names::Primitive::Str) | Kind::Dyn(..) => !SIZED_ONLY.contains(&lang),
            Kind::Ref(Mut::Mut, _) => !matches!(lang, "copy" | "clone"),
            Kind::Adt(..) | Kind::Tuple(..) | Kind::Param(_) | Kind::Proj(_) if SIZED_ONLY.contains(&lang) && self.poly.is_some() && self.has_params(ty) => self.poly_sized(ty),
            Kind::Adt(item, _) if SIZED_ONLY.contains(&lang) => !self.cx.foreign(*item) && !self.is_unsized(ty).unwrap_or(false),
            Kind::Tuple(_) | Kind::Array(..) if matches!(lang, "copy" | "clone") => self.parts_hold(of, ty),
            _ => true,
        }
    }

    /// Whether `Copy` or `Clone`, as `of`, holds for every element of the tuple or array `ty`, as rustc's built-in impls require.
    fn parts_hold(&mut self, of: ItemRef, ty: Ty) -> bool {
        let parts = match *self.cx.types.kind(ty) {
            Kind::Tuple(items) => self.cx.types.substs(items).to_vec(),
            Kind::Array(inner, _) => vec![inner],
            _ => return true,
        };
        parts.into_iter().all(|part| if self.compiler_decides(of, part) { self.compiler_holds(of, part) } else { self.select(of, part, &[]).is_ok_and(|picked| picked.is_some()) })
    }

    /// Whether trait `of` is `Fn`, `FnMut` or `FnOnce`, which callables implement with no impl to select.
    pub(in crate::sema) fn is_callable_trait(&self, of: ItemRef) -> bool {
        self.cx.lang(of).is_some_and(|lang| CALLABLE.contains(&lang))
    }

    /// The trait that bound `bound` in `arena` requires from module `home`, with its path, or none for a lifetime, a `?Sized` or a name that is not a trait.
    pub(super) fn bound_trait_at(&mut self, arena: &Body, bound: u32, home: ModRef) -> Option<(ItemRef, PathId)> {
        let Bound::Trait { path, maybe: false, .. } = arena.bounds[bound as usize] else {
            return None;
        };
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

    fn collect_header_traits(&mut self, item: ItemRef) -> Result<Vec<ItemRef>> {
        self.frame_item = Some(item);
        self.block = self.cx.owner_impl(item);
        let mut direct = Vec::new();
        if let Some(of) = self.cx.owner_trait(item) {
            direct.push(of);
        }
        if let Some(block) = self.block
            && let Some(of) = self.cx.impl_trait(block)
        {
            direct.push(of);
        }
        let home = self.home;
        for (header, src) in self.headers()? {
            let (arena, generics, predicates) = header.parts();
            let bounds: Vec<u32> = type_params(generics).flat_map(|g| arena.list(g.bounds).iter().copied()).chain(predicates.iter().flat_map(|p| arena.list(p.bounds).iter().copied())).collect();
            let found = self.in_source(src, |inner| Ok(bounds.into_iter().filter_map(|b| inner.bound_trait_at(arena, b, home)).map(|(of, _)| of).collect::<Vec<_>>()))?;
            direct.extend(found);
        }
        self.implied_traits(direct)
    }
}

/// The traits declared in `item`'s header and any enclosing impl or trait header, with every supertrait they imply.
pub fn header_bound_traits(cx: &mut super::super::Context<'_>, item: ItemRef) -> Vec<ItemRef> {
    if !matches!(cx.item(item).1.kind, ItemKind::Fn | ItemKind::Struct | ItemKind::Enum | ItemKind::Union | ItemKind::Type) {
        return Vec::new();
    }
    super::detached(cx, item, |infer| infer.collect_header_traits(item)).unwrap_or_default()
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

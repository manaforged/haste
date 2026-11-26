mod bounds;
mod cache;

use anyhow::{Result, bail};

use crate::body::{Body, PathId, Ty as SyntaxTy};
use crate::names::ItemRef;

use super::super::impls::{ImplRef, head};
use super::super::ty::{Kind, Ty};
use super::Infer;
use super::scope::{Frame, named_args, type_params};

/// An impl that a trait obligation picked, with its type arguments.
pub type Picked = (ImplRef, Vec<Ty>);

#[derive(Clone)]
pub(super) struct Vars {
    base_len: usize,
    updates: Vec<(u32, Option<Ty>)>,
    added: Vec<Option<Ty>>,
    diverges: Vec<u32>,
    projections: Vec<super::project::Projection>,
    obligations: Vec<(ItemRef, Vec<Ty>)>,
}

impl<'a> Infer<'_, 'a> {
    /// The one impl of trait `of` at trait arguments `args` for `self_ty`, or none when no impl or several apply.
    pub fn select(&mut self, of: ItemRef, self_ty: Ty, args: &[Ty]) -> Result<Option<Picked>> {
        let (key, local_key) = self.selection_keys(of, self_ty, args);
        if let Some(found) = self.cached_selection(key.as_ref(), local_key.as_ref()) {
            return Ok(
                found.map(|(block, impl_args)| (block, self.cx.types.substs(impl_args).to_vec()))
            );
        }
        let state = self.cache_state();
        let picked = self.select_at(of, self_ty, args, 0)?;
        self.store_selection(key, local_key, picked.as_ref(), state);
        Ok(picked)
    }

    fn select_at(
        &mut self,
        of: ItemRef,
        self_ty: Ty,
        args: &[Ty],
        depth: u32,
    ) -> Result<Option<Picked>> {
        let query = self.query(of, self_ty, args);
        if self.selecting.len() > crate::RECURSION_LIMIT || self.selecting.contains(&query) {
            self.cycle_hits += 1;
            return Ok(None);
        }
        self.selecting.push(query);
        let found = self.applicable(of, self_ty, args, depth);
        self.selecting.pop();
        let Ok([(picked, vars)]) = <[_; 1]>::try_from(found?) else {
            return Ok(None);
        };
        let mark = self.mark_vars();
        self.apply_var_delta(vars.base_len, &vars.updates, &vars.added, &vars.diverges);
        let open_arg = args
            .iter()
            .any(|&a| matches!(self.cx.types.kind(self.resolve(a)), Kind::Var(_)));
        self.rollback_vars(mark);
        if open_arg
            && of.krate != self.home.krate
            && head(self.cx.types.kind(self.resolve(self_ty))).is_some()
            && self.blanket(picked.0)?
        {
            return Ok(None);
        }
        self.adopt(vars);
        Ok(Some(picked))
    }

    /// Whether some impl of trait `of` at `args` may apply to `self_ty`, even when inference cannot pick one yet.
    pub fn may_implement(&mut self, of: ItemRef, self_ty: Ty, args: &[Ty]) -> Result<bool> {
        Ok(!self.applicable(of, self_ty, args, 0)?.is_empty()
            || self.poly_may_implement(of, self_ty, args)?)
    }

    pub(super) fn applicable(
        &mut self,
        of: ItemRef,
        self_ty: Ty,
        args: &[Ty],
        depth: u32,
    ) -> Result<Vec<(Picked, Vars)>> {
        let self_ty = self.resolve(self_ty);
        if depth as usize + self.selecting.len() > crate::RECURSION_LIMIT
            || (matches!(self.cx.types.kind(self_ty), Kind::Var(_))
                && !self.is_pending_opaque(self_ty))
        {
            return Ok(Vec::new());
        }
        let mut found = self.impl_candidates(of, self_ty, args, depth)?;
        let open = found.len() > 1 && self.obligation_open(self_ty, args);
        let outline = self.cx.outline;
        let non_sysroot = |b: ImplRef| {
            !outline.is_sysroot(match b {
                ImplRef::Module { krate, .. } => krate,
                ImplRef::Body(i) => i.krate,
            })
        };
        if found.len() > 1 && !open {
            let mut kept = Vec::new();
            for entry in &found {
                if non_sysroot(entry.0.0) || !self.blanket(entry.0.0)? {
                    kept.push(entry.clone());
                }
            }
            if !kept.is_empty() {
                found = kept;
            }
        }
        if found.len() > 1 && self.same_trait_args(&found, args) {
            found = self.most_specialized(found)?;
            if found.len() > 1 && !open && found.iter().any(|((b, _), _)| non_sysroot(*b)) {
                bail!(
                    "conflicting implementations of trait `{}`",
                    self.cx.name(of)
                );
            }
        }
        Ok(found)
    }

    fn impl_candidates(
        &mut self,
        of: ItemRef,
        self_ty: Ty,
        args: &[Ty],
        depth: u32,
    ) -> Result<Vec<(Picked, Vars)>> {
        let holders = self.holders(of, self_ty, args);
        let blocks = self
            .cx
            .trait_impls(of, head(self.cx.types.kind(self_ty)), &holders);
        let opaque_var = matches!(self.cx.types.kind(self_ty), Kind::Var(_));
        let mut found = Vec::new();
        for block in blocks {
            if opaque_var && !self.blanket(block)? {
                continue;
            }
            let mark = (
                self.mark_vars(),
                self.projections.len(),
                self.obligations.len(),
            );
            let applies = self.impl_applies(block, self_ty, args, depth);
            let projections = self
                .projections
                .split_off(mark.1.min(self.projections.len()));
            let obligations = self
                .obligations
                .split_off(mark.2.min(self.obligations.len()));
            match applies {
                Ok(Some(impl_args)) if !opaque_var || obligations.is_empty() => {
                    let base_len = mark.0.0;
                    let (updates, added, diverges) = self.capture_var_delta(mark.0);
                    self.rollback_vars(mark.0);
                    found.push((
                        (block, impl_args),
                        Vars {
                            base_len,
                            updates,
                            added,
                            diverges,
                            projections,
                            obligations,
                        },
                    ));
                }
                Ok(Some(_)) | Ok(None) => self.rollback_vars(mark.0),
                Err(err) => {
                    self.rollback_vars(mark.0);
                    return Err(err);
                }
            }
        }
        Ok(found)
    }

    pub(super) fn adopt(&mut self, vars: Vars) {
        self.apply_var_delta(vars.base_len, &vars.updates, &vars.added, &vars.diverges);
        self.projections.extend(vars.projections);
        self.obligations.extend(vars.obligations);
    }

    fn defining_impl(
        &mut self,
        of: ItemRef,
        self_ty: Ty,
        args: &[Ty],
        name: &str,
    ) -> Result<Option<Picked>> {
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

    pub(super) fn impl_member(
        &mut self,
        of: ItemRef,
        (self_ty, args): (Ty, &[Ty]),
        picked: Picked,
        name: &str,
    ) -> Result<Option<(ItemRef, Picked)>> {
        if let Some(member) = self.cx.member(picked.0, name) {
            return Ok(Some((member, picked)));
        }
        let Some(general) = self.defining_impl(of, self_ty, args, name)? else {
            return Ok(None);
        };
        Ok(self
            .cx
            .member(general.0, name)
            .map(|member| (member, general)))
    }

    fn same_trait_args(&mut self, found: &[(Picked, Vars)], args: &[Ty]) -> bool {
        let mut first: Option<Vec<Ty>> = None;
        for (_, vars) in found {
            let mark = self.mark_vars();
            self.apply_var_delta(vars.base_len, &vars.updates, &vars.added, &vars.diverges);
            let sub: Vec<Ty> = args.iter().map(|&arg| self.substitute(arg)).collect();
            self.rollback_vars(mark);
            match &first {
                None => first = Some(sub),
                Some(prev) if *prev != sub => return false,
                _ => {}
            }
        }
        true
    }

    fn most_specialized(&mut self, found: Vec<(Picked, Vars)>) -> Result<Vec<(Picked, Vars)>> {
        let mut facts = Vec::new();
        for ((block, impl_args), _) in &found {
            let direct = self.impl_bound_traits(*block, impl_args)?;
            let implied = self.implied_traits(direct.clone())?;
            facts.push(Specialty {
                default: self.cx.specializable(*block),
                direct,
                implied,
            });
        }
        let blocks: Vec<ImplRef> = found.iter().map(|((block, _), _)| *block).collect();
        let mut kept = Vec::new();
        for at in 0..found.len() {
            kept.push(!self.specialized(&blocks, &facts, at)?);
        }
        Ok(found
            .into_iter()
            .zip(kept)
            .filter_map(|(entry, keep)| keep.then_some(entry))
            .collect())
    }

    fn specialized(&mut self, blocks: &[ImplRef], facts: &[Specialty], at: usize) -> Result<bool> {
        for (by, (&block, fact)) in blocks.iter().zip(facts).enumerate() {
            if by != at && fact.specializes(&facts[at]) && self.instance_of(block, blocks[at])? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn instance_of(&mut self, special: ImplRef, general: ImplRef) -> Result<bool> {
        let deferred = (self.projections.len(), self.obligations.len());
        let found: Result<bool> = self.probe(|infer| {
            let (params, own) = infer.fresh_header(special)?;
            let (_, other) = infer.fresh_header(general)?;
            let unified = own.len() == other.len()
                && own.iter().zip(&other).all(|(&a, &b)| infer.unifies(a, b));
            let roots: Vec<Ty> = params.iter().map(|&param| infer.resolve(param)).collect();
            let rigid = roots.iter().enumerate().all(|(at, root)| {
                matches!(infer.cx.types.kind(*root), Kind::Var(_)) && !roots[..at].contains(root)
            });
            Ok(unified && rigid)
        });
        self.projections.truncate(deferred.0);
        self.obligations.truncate(deferred.1);
        found
    }

    fn fresh_header(&mut self, block: ImplRef) -> Result<(Vec<Ty>, Vec<Ty>)> {
        let (args, self_ty) = self.fresh_impl(block)?;
        let header = std::iter::once(self_ty)
            .chain(self.impl_trait_args(block, &args, self_ty)?)
            .collect();
        Ok((args, header))
    }

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
        let on_params = type_params(&header.generics)
            .flat_map(|generic| header.body.list(generic.bounds).iter().copied());
        let bounds: Vec<u32> = on_params
            .chain(
                header
                    .predicates
                    .iter()
                    .flat_map(|predicate| header.body.list(predicate.bounds).iter().copied()),
            )
            .collect();
        self.with_frame(frame, |infer| {
            Ok(bounds
                .into_iter()
                .filter_map(|bound| infer.bound_trait_at(&header.body, bound, home))
                .map(|(of, _)| of)
                .collect())
        })
    }

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
        self.probe(|infer| {
            let (args, self_ty) = infer.fresh_impl(block)?;
            Ok(args.contains(&self_ty))
        })
    }

    fn holders(&mut self, of: ItemRef, self_ty: Ty, args: &[Ty]) -> Vec<u32> {
        let mut crates = vec![of.krate];
        let mut stack: Vec<Ty> = std::iter::once(self_ty)
            .chain(args.iter().copied())
            .collect();
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

    fn impl_applies(
        &mut self,
        block: ImplRef,
        self_ty: Ty,
        args: &[Ty],
        depth: u32,
    ) -> Result<Option<Vec<Ty>>> {
        let Some(impl_args) = self.match_trait_impl(block, self_ty, args)? else {
            return Ok(None);
        };
        Ok(self
            .bounds_hold(block, &impl_args, depth)?
            .then_some(impl_args))
    }

    fn match_trait_impl(
        &mut self,
        block: ImplRef,
        self_ty: Ty,
        args: &[Ty],
    ) -> Result<Option<Vec<Ty>>> {
        let Some(impl_args) = self.match_impl(block, self_ty) else {
            return Ok(None);
        };
        let given = self.impl_trait_args(block, &impl_args, self_ty)?;
        if given.len() != args.len() {
            bail!(
                "an impl gives {} trait arguments where {} are wanted",
                given.len(),
                args.len()
            );
        }
        for (found, wanted) in given.iter().zip(args) {
            if !self.unifies(*found, *wanted) {
                return Ok(None);
            }
        }
        Ok(Some(impl_args))
    }

    pub(super) fn impl_frame_of(
        &mut self,
        block: ImplRef,
        impl_args: &[Ty],
        self_ty: Option<Ty>,
    ) -> Result<Frame<'a>> {
        let header = self.cx.impl_header(block)?;
        let file = self.cx.impl_file(block);
        let generics = named_args(&file.source, &header.generics, impl_args);
        Ok(Frame {
            src: &file.source,
            home: self.cx.impl_module(block)?,
            generics,
            self_ty,
            block: Some(block),
            item: None,
            root: block.root(),
        })
    }

    /// The trait arguments of impl `block` at `impl_args`, with the trait's defaults filled in.
    pub fn impl_trait_args(
        &mut self,
        block: ImplRef,
        impl_args: &[Ty],
        self_ty: Ty,
    ) -> Result<Vec<Ty>> {
        let header = self.cx.impl_header(block)?;
        let Some(SyntaxTy::Path(path)) = header.trait_ref.map(|at| *header.body.ty(at)) else {
            bail!("impl has no trait")
        };
        let Some(of) = self.cx.impl_trait(block) else {
            bail!("impl of something that is not a trait")
        };
        let frame = self.impl_frame_of(block, impl_args, Some(self_ty))?;
        self.with_frame(frame, |infer| {
            infer.bound_args(of, &header.body, path, self_ty)
        })
    }

    pub(super) fn bound_args(
        &mut self,
        of: ItemRef,
        arena: &Body,
        path: PathId,
        ty: Ty,
    ) -> Result<Vec<Ty>> {
        let count = arena.list(arena.path(path).segments).len();
        let given = self.segment_args(arena, path, count)?;
        self.trait_args(of, given, ty)
    }

    /// The arguments `given` to trait `of` for `self_ty`, completed with the trait's defaults.
    pub fn trait_args(&mut self, of: ItemRef, given: Vec<Ty>, self_ty: Ty) -> Result<Vec<Ty>> {
        self.fill_trait_args(of, given, self_ty, false)
    }

    pub(super) fn fill_trait_args(
        &mut self,
        of: ItemRef,
        mut given: Vec<Ty>,
        self_ty: Ty,
        fresh: bool,
    ) -> Result<Vec<Ty>> {
        let def = self.cx.trait_generics(of)?;
        let params: Vec<_> = type_params(&def.1.params).copied().collect();
        if given.len() > params.len() {
            bail!(
                "`{}` takes {} type arguments, {} given",
                self.cx.name(of),
                params.len(),
                given.len()
            );
        }
        for param in params.iter().skip(given.len()) {
            let arg = match param.default {
                Some(default) => {
                    let frame = self.item_frame(of, (&params, &given), (Some(self_ty), None))?;
                    self.with_frame(frame, |infer| infer.ty_or_const_arg(&def.0, default))?
                }
                None if fresh => self.fresh(Kind::Var),
                None => bail!(
                    "`{}` needs {} type arguments",
                    self.cx.name(of),
                    params.len()
                ),
            };
            given.push(arg);
        }
        Ok(given)
    }
}

struct Specialty {
    default: bool,
    direct: Vec<ItemRef>,
    implied: Vec<ItemRef>,
}

impl Specialty {
    fn specializes(&self, other: &Specialty) -> bool {
        other.default && (!self.default || (self.covers(other) && !other.covers(self)))
    }

    fn covers(&self, other: &Specialty) -> bool {
        other.direct.iter().all(|of| self.implied.contains(of))
    }
}

use anyhow::Result;

use crate::body::{Body, Bound, PathId};
use crate::names::{ItemRef, ModRef};

use super::super::super::impls::ImplRef;
use super::super::super::ty::{Kind, Mut, Ty};
use super::super::Infer;
use super::super::scope::type_params;

const BUILTIN: [&str; 12] = [
    "sized",
    "meta_sized",
    "pointee_sized",
    "copy",
    "clone",
    "destruct",
    "tuple_trait",
    "freeze",
    "pointee_trait",
    "unsize",
    "sync",
    "unpin",
];

const WEAKER_THAN_SIZED: [&str; 2] = ["meta_sized", "pointee_sized"];

const SIZED_ONLY: [&str; 3] = ["sized", "copy", "clone"];

impl<'a> Infer<'_, 'a> {
    pub(super) fn bounds_hold(
        &mut self,
        block: ImplRef,
        impl_args: &[Ty],
        depth: u32,
    ) -> Result<bool> {
        let header = self.cx.impl_header(block)?;
        let frame = self.impl_frame_of(block, impl_args, None)?;
        let home = frame.home;
        let mut wanted = Vec::new();
        for (generic, &arg) in type_params(&header.generics).zip(impl_args) {
            wanted.extend(header.body.list(generic.bounds).iter().map(|&b| (arg, b)));
        }
        self.with_frame(frame, |infer| {
            if infer.sets_unsized_param(block, &header, (impl_args, home))? {
                return Ok(false);
            }
            infer.self_ty = Some(match infer.cached_impl_self(block, impl_args) {
                Some(ty) => ty,
                None => infer.lower_ty(&header.body, header.self_ty)?,
            });
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
            infer.copy_impl_holds(block)
        })
    }

    fn sets_unsized_param(
        &mut self,
        block: ImplRef,
        header: &crate::body::sig::ImplHeader,
        (impl_args, home): (&[Ty], ModRef),
    ) -> Result<bool> {
        let src = &self.cx.impl_file(block).source;
        for (generic, &arg) in type_params(&header.generics).zip(impl_args) {
            if generic.kind != crate::body::sig::GenericKind::Type
                || !self.cx.types.kind(self.resolve(arg)).is_unsized()
            {
                continue;
            }
            let bounds =
                super::super::scope::generic_bounds(&header.body, src, &header.predicates, generic);
            if !bounds
                .into_iter()
                .any(|bound| self.relaxes_sized(&header.body, bound, home))
            {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn relaxes_sized(&mut self, arena: &Body, bound: u32, home: ModRef) -> bool {
        matches!(
            arena.bounds[bound as usize],
            Bound::Trait { maybe: true, .. }
        ) || self
            .bound_trait_at(arena, bound, home)
            .and_then(|(of, _)| self.cx.lang(of))
            .is_some_and(|lang| WEAKER_THAN_SIZED.contains(&lang))
    }

    pub(in super::super) fn bound_holds(
        &mut self,
        arena: &Body,
        home: ModRef,
        (ty, bound): (Ty, u32),
        depth: u32,
    ) -> Result<bool> {
        let Some((of, path)) = self.bound_trait_at(arena, bound, home) else {
            return Ok(true);
        };
        if self.cx.lang(of) == Some("unsize") {
            return self.unsize_bound_holds(arena, (of, path), ty);
        }
        if self.compiler_decides(of, ty) {
            let held = self.compiler_holds(of, ty);
            let defer_closure = !self.captures_settled
                && matches!(self.cx.types.kind(self.resolve(ty)), Kind::Closure(..));
            if held && self.is_thread_auto_trait(of) && (self.has_vars(ty) || defer_closure) {
                self.note_obligation_site(of, ty);
                self.obligations.push((of, vec![ty]));
            }
            return Ok(held);
        }
        let kind = self.cx.types.kind(self.resolve(ty));
        let is_var = matches!(kind, Kind::Var(_) | Kind::IntVar(_) | Kind::FloatVar(_));
        let is_closure = matches!(kind, Kind::Closure(..));
        let is_callable = matches!(kind, Kind::Closure(..) | Kind::FnDef(..) | Kind::FnPtr(..));
        if is_var {
            return self.var_holds(arena, home, (of, path), ty, depth);
        }
        if is_closure && self.cx.lang(of) == Some("future_trait") {
            self.ensure_coroutine(ty);
        }
        if self.compiler_impl(of, ty) {
            let args = self.bound_args(of, arena, path, ty)?;
            return self.assoc_bounds_hold((arena, home, path), (of, ty, &args), depth);
        }
        if self.is_callable_trait(of) && is_callable {
            return self.callable_bound_holds((arena, home, bound), (of, path, ty), depth);
        }
        if self.poly.is_some()
            && let Some(held) =
                self.poly_bound_check((arena, home, bound), (of, path, ty), depth)?
        {
            return Ok(held);
        }
        let args = self.bound_args(of, arena, path, ty)?;
        if self.object_holds(of, ty, &args) == Some(true) {
            return Ok(true);
        }
        let held = self.impl_holds((arena, home, path), (of, ty, &args), depth)?;
        if held && self.is_callable_trait(of) {
            return self.callable_bound_holds((arena, home, bound), (of, path, ty), depth);
        }
        Ok(held)
    }

    pub(in crate::sema) fn compiler_decides(&self, of: ItemRef, ty: Ty) -> bool {
        let lang = self.cx.lang(of);
        if !(lang.is_some_and(|l| BUILTIN.contains(&l)) || self.cx.is_auto_trait(of)) {
            return false;
        }
        if !lang.is_some_and(|l| matches!(l, "copy" | "clone")) {
            return true;
        }
        let resolved = self.resolve(ty);
        let user_concrete_copy = lang == Some("copy")
            && matches!(self.cx.types.kind(resolved), Kind::Adt(item, _) if !self.cx.outline.is_sysroot(item.krate))
            && !self.has_vars(resolved)
            && !self.has_params(resolved);
        let by_impl = !user_concrete_copy
            && matches!(
                self.cx.types.kind(resolved),
                Kind::Adt(..) | Kind::Param(_) | Kind::Proj(_)
            );
        !by_impl
    }

    fn impl_holds(
        &mut self,
        (arena, home, path): (&Body, ModRef, PathId),
        (of, ty, args): (ItemRef, Ty, &[Ty]),
        depth: u32,
    ) -> Result<bool> {
        let (key, local_key) = if super::super::project::has_assoc_bounds(arena, path) {
            (None, None)
        } else {
            self.selection_keys(of, ty, args, true)
        };
        if let Some(found) = self.cached_selection(key.as_ref(), local_key.as_ref()) {
            return Ok(found.is_some());
        }
        let query = self.query(of, ty, args);
        if self.selecting.len() > crate::RECURSION_LIMIT || self.selecting.contains(&query) {
            self.cycle_hits += 1;
            return Ok(self.obligation_open(ty, args));
        }
        let state = self.begin_cache_state();
        self.selecting.push(query);
        let found = self.applicable(of, ty, args, depth + 1);
        self.selecting.pop();
        let found = match found {
            Ok(found) => found,
            Err(err) => {
                self.end_cache_state();
                return Err(err);
            }
        };
        match <[_; 1]>::try_from(found) {
            Ok([(picked, vars)]) => {
                self.adopt(vars);
                let holds = self.assoc_bounds_hold((arena, home, path), (of, ty, args), depth);
                let min_var = self.end_cache_state();
                let holds = holds?;
                if holds {
                    self.store_selection(key, local_key, Some(&picked), state, min_var);
                }
                Ok(holds)
            }
            Err(found) => {
                let min_var = self.end_cache_state();
                let held = !found.is_empty();
                if !held {
                    self.store_selection(key, local_key, None, state, min_var);
                }
                if held && self.obligation_open(ty, args) {
                    self.note_obligation_site(of, ty);
                    let list: Vec<Ty> = std::iter::once(ty).chain(args.iter().copied()).collect();
                    self.obligations.push((of, list));
                }
                Ok(held)
            }
        }
    }

    fn var_holds(
        &mut self,
        arena: &Body,
        home: ModRef,
        (of, path): (ItemRef, PathId),
        ty: Ty,
        depth: u32,
    ) -> Result<bool> {
        let args = self.bound_args(of, arena, path, ty)?;
        let substs: Vec<Ty> = std::iter::once(ty).chain(args.iter().copied()).collect();
        let resolved = self.resolve(ty);
        if !matches!(self.cx.types.kind(resolved), Kind::Var(_)) {
            let held = (!self.applicable(of, ty, &args, depth + 1)?.is_empty()
                || self.poly_may_implement(of, ty, &args)?)
                && self.assoc_bounds_hold((arena, home, path), (of, ty, &args), depth)?;
            if held {
                self.note_obligation_site(of, substs[0]);
                self.obligations.push((of, substs));
            }
            return Ok(held);
        }
        if let Some(held) = self.opaque_holds(of, &substs)? {
            return Ok(held && self.assoc_bounds_hold((arena, home, path), (of, ty, &args), depth)?);
        }
        if self.is_pending_opaque(resolved) {
            return self.opaque_var_holds(
                arena,
                home,
                (of, path),
                (ty, resolved, &args, substs),
                depth,
            );
        }
        if !self.assoc_bounds_hold((arena, home, path), (of, ty, &args), depth)? {
            return Ok(false);
        }
        self.note_obligation_site(of, substs[0]);
        self.obligations.push((of, substs));
        Ok(true)
    }

    fn opaque_var_holds(
        &mut self,
        arena: &Body,
        home: ModRef,
        (of, path): (ItemRef, PathId),
        (ty, resolved, args, substs): (Ty, Ty, &[Ty], Vec<Ty>),
        depth: u32,
    ) -> Result<bool> {
        let matches: Vec<Vec<Ty>> = self
            .opaque_bounds(resolved)?
            .into_iter()
            .filter(|(bound, _)| *bound == of)
            .map(|(_, types)| types)
            .collect();
        if !matches.is_empty() {
            let mut any = false;
            for types in &matches {
                if self
                    .trial(|infer| {
                        Ok(substs
                            .iter()
                            .zip(types)
                            .all(|(&a, &b)| infer.unifies(a, b))
                            .then_some(()))
                    })?
                    .is_some()
                {
                    any = true;
                    break;
                }
            }
            if !any || !self.assoc_bounds_hold((arena, home, path), (of, ty, args), depth)? {
                return Ok(false);
            }
            self.note_obligation_site(of, substs[0]);
            self.obligations.push((of, substs));
            return Ok(true);
        }
        if self.is_callable_trait(of)
            && self.opaque_sig(resolved).is_some()
            && (self.callable_opaque_holds(of, resolved, args)?
                || (self.is_sync_callable_trait(of)
                    && args.first().is_some_and(|&arg| {
                        matches!(self.cx.types.kind(self.resolve(arg)), Kind::Var(_))
                    })))
        {
            if !self.assoc_bounds_hold((arena, home, path), (of, ty, args), depth)? {
                return Ok(false);
            }
            self.note_obligation_site(of, substs[0]);
            self.obligations.push((of, substs));
            return Ok(true);
        }
        self.impl_holds((arena, home, path), (of, ty, args), depth)
    }

    pub(in crate::sema) fn object_holds(
        &mut self,
        of: ItemRef,
        ty: Ty,
        args: &[Ty],
    ) -> Option<bool> {
        let Kind::Dyn(object, parts, members, _) = *self.cx.types.kind(self.resolve(ty)) else {
            return None;
        };
        let (p, m) = (self.cx.types.substs(parts), self.cx.types.items(members));
        let given = super::super::super::ty::object_parts(p, m).0.to_vec();
        let same_family = (self.is_sync_callable_trait(object) && self.is_sync_callable_trait(of))
            || (self.is_async_callable_trait(object) && self.is_async_callable_trait(of));
        if object != of && same_family {
            if !self
                .implied_traits(vec![object])
                .is_ok_and(|t| t.contains(&of))
                || given.len() != args.len()
            {
                return Some(false);
            }
            let held = self.trial(|infer| {
                let ok = given.iter().zip(args).all(|(a, b)| infer.unifies(*a, *b));
                Ok(ok.then_some(()))
            });
            return Some(matches!(held, Ok(Some(()))));
        }
        let bounds = self
            .expand_super_bounds(vec![(object, std::iter::once(ty).chain(given).collect())])
            .ok()?;
        let matching: Vec<Vec<Ty>> = bounds
            .into_iter()
            .filter(|(b, s)| *b == of && s.len() == args.len() + 1)
            .map(|(_, s)| s)
            .collect();
        if matching.is_empty() {
            return Some(false);
        }
        if matching.len() == 1 {
            let mark = self.mark_vars();
            if matching[0][1..]
                .iter()
                .zip(args)
                .all(|(&a, &b)| self.unifies(a, b))
            {
                self.commit_vars(mark);
                return Some(true);
            }
            self.rollback_vars(mark);
            return Some(false);
        }
        let held = matching.iter().any(|substs| {
            let ok = self.trial(|infer| {
                Ok(substs[1..]
                    .iter()
                    .zip(args)
                    .all(|(&a, &b)| infer.unifies(a, b))
                    .then_some(()))
            });
            matches!(ok, Ok(Some(())))
        });
        Some(held)
    }

    pub(in super::super) fn is_thread_auto_trait(&self, of: ItemRef) -> bool {
        matches!(self.cx.lang(of), Some("sync" | "unpin"))
            || (self.cx.is_auto_trait(of)
                && self.cx.outline.is_sysroot(of.krate)
                && self.cx.name(of) == "Send")
    }

    pub(in super::super) fn compiler_trait(&self, of: ItemRef) -> bool {
        self.cx.lang(of).is_some_and(|lang| BUILTIN.contains(&lang)) || self.cx.is_auto_trait(of)
    }

    pub(in crate::sema) fn compiler_holds(&mut self, of: ItemRef, ty: Ty) -> bool {
        let lang = self.cx.lang(of);
        let is_thread_auto = matches!(lang, Some("sync" | "unpin"))
            || (self.cx.is_auto_trait(of)
                && self.cx.outline.is_sysroot(of.krate)
                && self.cx.name(of) == "Send");
        if is_thread_auto {
            return self.thread_auto_holds(of, ty);
        }
        let Some(lang) = lang else {
            return true;
        };
        let ty = self.resolve(ty);
        match self.cx.types.kind(ty) {
            Kind::Slice(_) | Kind::Prim(crate::names::Primitive::Str) | Kind::Dyn(..) => {
                !SIZED_ONLY.contains(&lang)
            }
            Kind::Ref(Mut::Mut, _) => !matches!(lang, "copy" | "clone"),
            Kind::Adt(..) | Kind::Tuple(..) | Kind::Param(_) | Kind::Proj(_)
                if SIZED_ONLY.contains(&lang) && self.poly.is_some() && self.has_params(ty) =>
            {
                self.poly_sized(ty)
            }
            Kind::Adt(item, _)
                if lang == "copy"
                    && !self.cx.outline.is_sysroot(item.krate)
                    && !self.has_vars(ty)
                    && !self.has_params(ty) =>
            {
                !self.cx.foreign(*item)
                    && !self.is_unsized(ty).unwrap_or(false)
                    && self.is_copy(ty).unwrap_or(false)
            }
            Kind::Adt(item, _) if SIZED_ONLY.contains(&lang) => {
                !self.cx.foreign(*item) && !self.is_unsized(ty).unwrap_or(false)
            }
            Kind::Tuple(_) | Kind::Array(..) if matches!(lang, "copy" | "clone") => {
                self.parts_hold(of, ty)
            }
            _ => true,
        }
    }

    fn parts_hold(&mut self, of: ItemRef, ty: Ty) -> bool {
        self.parts(ty).into_iter().all(|part| {
            if self.compiler_decides(of, part) {
                self.compiler_holds(of, part)
            } else {
                self.select(of, part, &[]).is_ok_and(|p| p.is_some())
            }
        })
    }

    pub(super) fn parts(&mut self, ty: Ty) -> Vec<Ty> {
        match self.cx.types.kind(ty).clone() {
            Kind::Tuple(items) => self.cx.types.substs(items).to_vec(),
            Kind::Array(inner, _) => vec![inner],
            Kind::Adt(item, args) => {
                let args = self.cx.types.substs(args).to_vec();
                self.adt_fields(item, &args)
                    .map(|f| f.iter().flatten().copied().collect())
                    .unwrap_or_default()
            }
            _ => Vec::new(),
        }
    }
}

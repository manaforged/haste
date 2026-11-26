use anyhow::Result;

use crate::body::{Body, Bound, PathId};
use crate::names::path::Path as NamePath;
use crate::names::{Def, ItemRef, ModRef};
use crate::outline::item::ItemKind;

use super::super::super::impls::{Head, ImplRef};
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
            return self.var_holds(arena, (of, path), ty, depth);
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
        let resolved = self.resolve(ty);
        let lang = self.cx.lang(of);
        let user_concrete_copy = lang == Some("copy")
            && matches!(self.cx.types.kind(resolved), Kind::Adt(item, _) if !self.cx.outline.is_sysroot(item.krate))
            && !self.has_vars(resolved)
            && !self.has_params(resolved);
        let by_impl = !user_concrete_copy
            && matches!(
                self.cx.types.kind(resolved),
                Kind::Adt(..) | Kind::Param(_) | Kind::Proj(_)
            )
            && lang.is_some_and(|l| matches!(l, "copy" | "clone"));
        self.compiler_trait(of) && !by_impl
    }

    fn impl_holds(
        &mut self,
        (arena, home, path): (&Body, ModRef, PathId),
        (of, ty, args): (ItemRef, Ty, &[Ty]),
        depth: u32,
    ) -> Result<bool> {
        let query = self.query(of, ty, args);
        if self.selecting.len() > crate::RECURSION_LIMIT || self.selecting.contains(&query) {
            self.cycle_hits += 1;
            return Ok(self.obligation_open(ty, args));
        }
        let (key, local_key) = if super::super::project::has_assoc_bounds(arena, path) {
            (None, None)
        } else {
            self.selection_keys(of, ty, args, true)
        };
        if let Some(found) = self.cached_selection(key.as_ref(), local_key.as_ref()) {
            return Ok(found.is_some());
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
        (of, path): (ItemRef, PathId),
        ty: Ty,
        depth: u32,
    ) -> Result<bool> {
        let args = self.bound_args(of, arena, path, ty)?;
        let substs: Vec<Ty> = std::iter::once(ty).chain(args.iter().copied()).collect();
        if !matches!(self.cx.types.kind(self.resolve(ty)), Kind::Var(_)) {
            let held = !self.applicable(of, ty, &args, depth + 1)?.is_empty()
                || self.poly_may_implement(of, ty, &args)?;
            if held {
                self.note_obligation_site(of, substs[0]);
                self.obligations.push((of, substs));
            }
            return Ok(held);
        }
        let held = self.opaque_holds(of, &substs)?;
        if held.is_none() {
            self.note_obligation_site(of, substs[0]);
            self.obligations.push((of, substs));
        }
        Ok(held != Some(false))
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
        if self.is_thread_auto_trait(of) {
            return self.thread_auto_holds(of, ty);
        }
        let Some(lang) = self.cx.lang(of) else {
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

    fn parts(&mut self, ty: Ty) -> Vec<Ty> {
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

    fn dyn_holds_auto(
        &mut self,
        of: ItemRef,
        object: ItemRef,
        autos: super::super::super::substs::ItemList,
    ) -> bool {
        object == of
            || self.cx.types.items(autos).contains(&of)
            || self
                .implied_traits(vec![object])
                .is_ok_and(|t| t.contains(&of))
    }

    fn unpin_holds(&mut self, of: ItemRef, ty: Ty, seen: &mut Vec<Ty>) -> bool {
        let ty = self.resolve(ty);
        if seen.contains(&ty) {
            return true;
        }
        match *self.cx.types.kind(ty) {
            Kind::Var(_) | Kind::IntVar(_) | Kind::FloatVar(_) => return true,
            Kind::Param(_) | Kind::Proj(_) => {
                return !self.root_is_user()
                    || (self.poly.is_some()
                        && self.poly_holds_in_env(of, ty, &[]).unwrap_or(false));
            }
            Kind::Slice(inner) | Kind::Pat(inner, _) => {
                seen.push(ty);
                return self.unpin_holds(of, inner, seen);
            }
            Kind::Dyn(object, _, _, autos) => return self.dyn_holds_auto(of, object, autos),
            Kind::Closure(parent, substs, at) => {
                return !self.ensure_coroutine(ty)
                    && self.closure_unpin_holds(of, ty, (parent, substs, at), seen);
            }
            Kind::Adt(item, args) => {
                if let Some(explicit) = self.adt_explicit_unpin(of, ty, item, args, seen) {
                    return explicit;
                }
            }
            Kind::Tuple(_) | Kind::Array(..) => {}
            _ => return true,
        }
        seen.push(ty);
        self.parts(ty)
            .into_iter()
            .all(|part| self.unpin_holds(of, part, seen))
    }

    fn adt_explicit_unpin(
        &mut self,
        of: ItemRef,
        ty: Ty,
        item: ItemRef,
        args: super::super::super::substs::Substs,
        seen: &mut Vec<Ty>,
    ) -> Option<bool> {
        let holders = self.holders(of, ty, &[]);
        if self.cx.lang(item) == Some("unsafe_pinned")
            || (self.cx.outline.is_sysroot(item.krate) && self.cx.name(item) == "PhantomPinned")
            || self.cx.negative_impl(of, Head::Adt(item), &holders)
        {
            return Some(false);
        }
        if self.cx.boxed(ty).is_some() {
            return Some(true);
        }
        if self
            .cx
            .has_specific_trait_impl_for_head(of, Head::Adt(item), &holders)
        {
            let holds = self.has_vars(ty)
                || self.probe(|infer| infer.select(of, ty, &[]).is_ok_and(|p| p.is_some()));
            return Some(holds);
        }
        if self.cx.lang(item) == Some("phantom_data") {
            seen.push(ty);
            let first = self.cx.types.substs(args).first().copied();
            return Some(first.is_none_or(|t| self.unpin_holds(of, t, seen)));
        }
        None
    }

    fn closure_unpin_holds(
        &mut self,
        of: ItemRef,
        ty: Ty,
        (parent, substs, at): (ItemRef, super::super::super::substs::Substs, u32),
        seen: &mut Vec<Ty>,
    ) -> bool {
        seen.push(ty);
        if self.instance == Some((parent, substs)) {
            if !self.captures_settled {
                return true;
            }
            let Some(captures) = self
                .out
                .closures
                .get(&crate::body::ExprId(at))
                .map(|c| c.captures.clone())
            else {
                return true;
            };
            return captures.into_iter().all(|cap| {
                cap.by != super::super::By::Value || {
                    let local_ty = self.resolve(self.out.locals[cap.local as usize].ty);
                    let cap_ty = cap.ty(self.cx, local_ty).unwrap_or(local_ty);
                    self.unpin_holds(of, cap_ty, seen)
                }
            });
        }
        self.cx
            .ensure_closure(ty)
            .is_none_or(|info| self.unpin_holds(of, info.env, seen))
    }

    pub(in super::super) fn thread_auto_holds(&mut self, of: ItemRef, ty: Ty) -> bool {
        if self.cx.lang(of) == Some("unpin") {
            return self.unpin_holds(of, ty, &mut Vec::new());
        }
        let (holds, dep) = self.thread_auto_holds_dep(of, ty);
        self.auto_dep = self.auto_dep.min(dep);
        if self.auto_stack.is_empty() {
            self.auto_provisional.clear();
            self.auto_dep = usize::MAX;
        }
        holds
    }

    fn finish_auto_node(
        &mut self,
        key: (ItemRef, Ty),
        cacheable: bool,
        my_depth: usize,
        holds: bool,
        dep: usize,
    ) -> (bool, usize) {
        if !holds {
            self.auto_provisional.clear();
            if cacheable {
                self.cx.set_auto_trait(key, false);
            }
            return (false, usize::MAX);
        }
        if dep >= my_depth {
            if cacheable {
                self.cx.set_auto_trait(key, true);
            }
            let promoted: Vec<(ItemRef, Ty)> = self
                .auto_provisional
                .iter()
                .filter_map(|(&k, &d)| (d >= my_depth).then_some(k))
                .collect();
            for k in promoted {
                self.auto_provisional.remove(&k);
                self.cx.set_auto_trait(k, true);
            }
            (true, usize::MAX)
        } else {
            if cacheable {
                self.auto_provisional.insert(key, dep);
            }
            (true, dep)
        }
    }

    fn thread_auto_holds_dep(&mut self, of: ItemRef, ty: Ty) -> (bool, usize) {
        let ty = self.resolve(ty);
        if let Some(pos) = self.auto_stack.iter().position(|&e| e == (of, ty)) {
            self.cycle_hits += 1;
            return (true, pos);
        }
        if self.auto_stack.len() >= crate::RECURSION_LIMIT {
            self.cycle_hits += 1;
            return (true, 0);
        }
        let cacheable = self.cx.types.is_ground(ty);
        if cacheable {
            if let Some(cached) = self.cx.auto_trait(&(of, ty)) {
                return (cached, usize::MAX);
            }
            if let Some(&dep) = self.auto_provisional.get(&(of, ty)) {
                return (true, dep);
            }
        }
        match *self.cx.types.kind(ty) {
            Kind::Prim(_)
            | Kind::Never
            | Kind::FnDef(..)
            | Kind::FnPtr(..)
            | Kind::Value(_)
            | Kind::Var(_)
            | Kind::IntVar(_)
            | Kind::FloatVar(_)
            | Kind::Param(_)
            | Kind::Proj(_) => (true, usize::MAX),
            Kind::Ptr(..) => (false, usize::MAX),
            Kind::Ref(Mut::Not, inner) => match self.cx.lang_item("sync") {
                Some(sync_of) => self.thread_auto_holds_dep(sync_of, inner),
                None => (true, usize::MAX),
            },
            Kind::Ref(Mut::Mut, inner)
            | Kind::Array(inner, _)
            | Kind::Slice(inner)
            | Kind::Pat(inner, _) => self.thread_auto_holds_dep(of, inner),
            Kind::Dyn(object, _, _, autos) => (self.dyn_holds_auto(of, object, autos), usize::MAX),
            Kind::Tuple(_) => {
                let my_depth = self.auto_stack.len();
                self.auto_stack.push((of, ty));
                let (mut dep, mut holds) = (usize::MAX, true);
                for part in self.parts(ty) {
                    let (ok, child_dep) = self.thread_auto_holds_dep(of, part);
                    if !ok {
                        holds = false;
                        break;
                    }
                    dep = dep.min(child_dep);
                }
                self.auto_stack.pop();
                self.finish_auto_node((of, ty), cacheable, my_depth, holds, dep)
            }
            Kind::Closure(parent, substs, at) => {
                self.closure_auto_holds(of, ty, (parent, substs, at))
            }
            Kind::Adt(item, args) => self.adt_auto_holds(of, ty, item, args, cacheable),
        }
    }

    fn closure_auto_holds(
        &mut self,
        of: ItemRef,
        ty: Ty,
        (parent, substs, at): (ItemRef, super::super::super::substs::Substs, u32),
    ) -> (bool, usize) {
        self.auto_stack.push((of, ty));
        let mut dep = usize::MAX;
        let (mut holds, extra_tys) = if self.instance == Some((parent, substs)) {
            let Some(c) = self.out.closures.get(&crate::body::ExprId(at)) else {
                self.auto_stack.pop();
                return (true, usize::MAX);
            };
            let (captures, is_cor, cor_body) = (c.captures.clone(), c.coroutine, c.body);
            let mut ok = true;
            for cap in captures {
                let local_ty = self.resolve(self.out.locals[cap.local as usize].ty);
                let cap_ty = cap.ty(self.cx, local_ty).unwrap_or(local_ty);
                let held = match cap.by {
                    super::super::By::Ref => self.cx.types.intern(Kind::Ref(Mut::Not, cap_ty)),
                    super::super::By::Mut => self.cx.types.intern(Kind::Ref(Mut::Mut, cap_ty)),
                    super::super::By::Value => cap_ty,
                };
                let (h, child_dep) = self.thread_auto_holds_dep(of, held);
                if !h {
                    ok = false;
                    break;
                }
                dep = dep.min(child_dep);
            }
            let extra = if ok && is_cor {
                super::super::drop::coroutine_body_tys(
                    self.body,
                    super::super::drop::CoroutineSource::Infer(&self.out),
                    cor_body,
                )
            } else {
                Vec::new()
            };
            (ok, extra)
        } else if let Some(info) = self.cx.ensure_closure(ty) {
            let (h, child_dep) = self.thread_auto_holds_dep(of, info.env);
            dep = child_dep;
            let extra = if h && info.closure.coroutine {
                if !matches!(info.typed.tables, super::super::LazyTables::CheckOnly(_))
                    && let Ok(body) = self.cx.body(info.parent)
                {
                    let scoped = info.typed.scoped(&self.cx.types);
                    super::super::drop::coroutine_body_tys(
                        &body,
                        super::super::drop::CoroutineSource::Typed(&**scoped, &scoped.locals),
                        info.closure.body,
                    )
                } else {
                    info.typed.locals.iter().map(|l| l.ty).collect()
                }
            } else {
                Vec::new()
            };
            (h, extra)
        } else {
            (true, Vec::new())
        };
        if holds {
            for raw in extra_tys {
                let lt = self.resolve(raw);
                if lt == Ty::NONE {
                    continue;
                }
                let (lh, ldep) = self.thread_auto_holds_dep(of, lt);
                if !lh {
                    holds = false;
                    break;
                }
                dep = dep.min(ldep);
            }
        }
        self.auto_stack.pop();
        (holds, if holds { dep } else { usize::MAX })
    }

    fn adt_auto_holds(
        &mut self,
        of: ItemRef,
        ty: Ty,
        item: ItemRef,
        args: super::super::super::substs::Substs,
        cacheable: bool,
    ) -> (bool, usize) {
        let holders = self.holders(of, ty, &[]);
        if self.cx.negative_impl(of, Head::Adt(item), &holders) {
            if cacheable {
                self.cx.set_auto_trait((of, ty), false);
            }
            return (false, usize::MAX);
        }
        let my_depth = self.auto_stack.len();
        self.auto_stack.push((of, ty));
        let (holds, dep) =
            if self
                .cx
                .has_specific_trait_impl_for_head(of, Head::Adt(item), &holders)
            {
                if self.has_vars(ty) || self.has_params(ty) {
                    (true, usize::MAX)
                } else {
                    let saved_dep = std::mem::replace(&mut self.auto_dep, usize::MAX);
                    let ok =
                        self.probe(|infer| infer.select(of, ty, &[]).is_ok_and(|p| p.is_some()));
                    (ok, std::mem::replace(&mut self.auto_dep, saved_dep))
                }
            } else if self.cx.lang(item) == Some("phantom_data") {
                match self.cx.types.substs(args).first().copied() {
                    Some(t) => self.thread_auto_holds_dep(of, t),
                    None => (true, usize::MAX),
                }
            } else {
                let (mut dep, mut ok) = (usize::MAX, true);
                for part in self.parts(ty) {
                    let (h, child_dep) = self.thread_auto_holds_dep(of, part);
                    if !h {
                        ok = false;
                        break;
                    }
                    dep = dep.min(child_dep);
                }
                (ok, dep)
            };
        self.auto_stack.pop();
        self.finish_auto_node((of, ty), cacheable, my_depth, holds, dep)
    }

    pub(in crate::sema) fn bound_trait_at(
        &mut self,
        arena: &Body,
        bound: u32,
        home: ModRef,
    ) -> Option<(ItemRef, PathId)> {
        let Bound::Trait {
            path, maybe: false, ..
        } = arena.bounds[bound as usize]
        else {
            return None;
        };
        Some((self.trait_named(arena, path, home)?, path))
    }

    /// The trait that `path` in `arena` names from module `home`.
    pub fn trait_named(&mut self, arena: &Body, path: PathId, home: ModRef) -> Option<ItemRef> {
        let (segments, absolute, dollar) = self.segments(arena, path);
        self.trait_at(
            &NamePath {
                segments: &segments,
                absolute,
                dollar,
            },
            home,
        )
    }

    pub(in super::super) fn trait_at(
        &mut self,
        path: &NamePath<'_>,
        home: ModRef,
    ) -> Option<ItemRef> {
        let def = if home == self.home {
            self.type_def(path)
        } else {
            self.cx.paths(home.krate).resolve(path, home)
        };
        match def? {
            Def::Item(item) if self.cx.item(item).1.kind == ItemKind::Trait => Some(item),
            _ => None,
        }
    }
}

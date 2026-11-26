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
            wanted.extend(
                header
                    .body
                    .list(generic.bounds)
                    .iter()
                    .map(|&bound| (arg, bound)),
            );
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
            if held && self.is_thread_auto_trait(of) && self.has_vars(ty) {
                self.note_obligation_site(of, ty);
                self.obligations.push((of, vec![ty]));
            }
            return Ok(held);
        }
        let kind = self.cx.types.kind(self.resolve(ty));
        if matches!(kind, Kind::Var(_) | Kind::IntVar(_) | Kind::FloatVar(_)) {
            return self.var_holds(arena, (of, path), ty, depth);
        }
        if self.compiler_impl(of, ty) {
            let args = self.bound_args(of, arena, path, ty)?;
            return self.assoc_bounds_hold((arena, home, path), (of, ty, &args), depth);
        }
        if self.is_callable_trait(of)
            && matches!(kind, Kind::Closure(..) | Kind::FnDef(..) | Kind::FnPtr(..))
        {
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
        let by_impl = matches!(
            self.cx.types.kind(self.resolve(ty)),
            Kind::Adt(..) | Kind::Param(_) | Kind::Proj(_)
        ) && self
            .cx
            .lang(of)
            .is_some_and(|lang| matches!(lang, "copy" | "clone"));
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
        let state = self.cache_state();
        self.selecting.push(query);
        let found = self.applicable(of, ty, args, depth + 1);
        self.selecting.pop();
        match <[_; 1]>::try_from(found?) {
            Ok([(picked, vars)]) => {
                self.adopt(vars);
                let holds = self.assoc_bounds_hold((arena, home, path), (of, ty, args), depth)?;
                if holds {
                    self.store_selection(key, local_key, Some(&picked), state);
                }
                Ok(holds)
            }
            Err(found) => {
                let held = !found.is_empty();
                if !held {
                    self.store_selection(key, local_key, None, state);
                }
                if held && self.obligation_open(ty, args) {
                    self.note_obligation_site(of, ty);
                    self.obligations.push((
                        of,
                        std::iter::once(ty).chain(args.iter().copied()).collect(),
                    ));
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
        let given = super::super::super::ty::object_parts(
            self.cx.types.substs(parts),
            self.cx.types.items(members),
        )
        .0
        .to_vec();
        let same_family = (self.is_sync_callable_trait(object) && self.is_sync_callable_trait(of))
            || (self.is_async_callable_trait(object) && self.is_async_callable_trait(of));
        if object != of && same_family {
            if !self
                .implied_traits(vec![object])
                .is_ok_and(|traits| traits.contains(&of))
                || given.len() != args.len()
            {
                return Some(false);
            }
            let held = self.trial(|infer| {
                Ok(given
                    .iter()
                    .zip(args)
                    .all(|(found, wanted)| infer.unifies(*found, *wanted))
                    .then_some(()))
            });
            return Some(matches!(held, Ok(Some(()))));
        }
        let bounds = self
            .expand_super_bounds(vec![(object, std::iter::once(ty).chain(given).collect())])
            .ok()?;
        let matching: Vec<Vec<Ty>> = bounds
            .into_iter()
            .filter(|(bound, substs)| *bound == of && substs.len() == args.len() + 1)
            .map(|(_, substs)| substs)
            .collect();
        if matching.is_empty() {
            return Some(false);
        }
        if matching.len() == 1 {
            let mark = self.mark_vars();
            if matching[0][1..]
                .iter()
                .zip(args)
                .all(|(&found, &wanted)| self.unifies(found, wanted))
            {
                self.commit_vars(mark);
                return Some(true);
            }
            self.rollback_vars(mark);
            return Some(false);
        }
        let held = matching.iter().any(|substs| {
            matches!(
                self.trial(|infer| {
                    Ok(substs[1..]
                        .iter()
                        .zip(args)
                        .all(|(&found, &wanted)| infer.unifies(found, wanted))
                        .then_some(()))
                }),
                Ok(Some(()))
            )
        });
        Some(held)
    }

    pub(in super::super) fn is_thread_auto_trait(&self, of: ItemRef) -> bool {
        self.cx.lang(of) == Some("sync")
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
            Kind::Adt(item, _) if SIZED_ONLY.contains(&lang) => {
                !self.cx.foreign(*item) && !self.is_unsized(ty).unwrap_or(false)
            }
            Kind::Tuple(_) | Kind::Array(..) if matches!(lang, "copy" | "clone") => {
                self.parts_hold(of, ty)
            }
            _ if lang == "unpin" => self.unpin_holds(of, ty, &mut Vec::new()),
            _ => true,
        }
    }

    fn parts_hold(&mut self, of: ItemRef, ty: Ty) -> bool {
        self.parts(ty).into_iter().all(|part| {
            if self.compiler_decides(of, part) {
                self.compiler_holds(of, part)
            } else {
                self.select(of, part, &[])
                    .is_ok_and(|picked| picked.is_some())
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
                    .map(|fields| fields.iter().flatten().copied().collect())
                    .unwrap_or_default()
            }
            _ => Vec::new(),
        }
    }

    fn unpin_holds(&mut self, of: ItemRef, ty: Ty, seen: &mut Vec<Ty>) -> bool {
        let ty = self.resolve(ty);
        if seen.contains(&ty) {
            return true;
        }
        match *self.cx.types.kind(ty) {
            Kind::Closure(..) => return self.coroutine_output(ty).is_none(),
            Kind::Adt(item, _) => {
                let holders = self.holders(of, ty, &[]);
                if self.cx.negative_impl(of, Head::Adt(item), &holders) {
                    return false;
                }
                if self
                    .select(of, ty, &[])
                    .is_ok_and(|picked| picked.is_some())
                {
                    return true;
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

    pub(in super::super) fn thread_auto_holds(&mut self, of: ItemRef, ty: Ty) -> bool {
        let ty = self.resolve(ty);
        if self.auto_stack.contains(&(of, ty)) {
            self.cycle_hits += 1;
            return true;
        }
        if self.auto_stack.len() >= crate::RECURSION_LIMIT {
            self.cycle_hits += 1;
            return true;
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
            | Kind::Proj(_) => true,
            Kind::Ptr(..) => false,
            Kind::Ref(Mut::Not, inner) => match self.cx.lang_item("sync") {
                Some(sync_of) => self.thread_auto_holds(sync_of, inner),
                None => true,
            },
            Kind::Ref(Mut::Mut, inner)
            | Kind::Array(inner, _)
            | Kind::Slice(inner)
            | Kind::Pat(inner, _) => self.thread_auto_holds(of, inner),
            Kind::Dyn(object, _, _, autos) => {
                object == of
                    || self.cx.types.items(autos).contains(&of)
                    || self
                        .implied_traits(vec![object])
                        .is_ok_and(|traits| traits.contains(&of))
            }
            Kind::Tuple(_) => {
                self.auto_stack.push((of, ty));
                let holds = self
                    .parts(ty)
                    .into_iter()
                    .all(|part| self.thread_auto_holds(of, part));
                self.auto_stack.pop();
                holds
            }
            Kind::Closure(parent, _, at) => self.closure_auto_holds(of, ty, parent, at),
            Kind::Adt(item, args) => self.adt_auto_holds(of, ty, item, args),
        }
    }

    fn closure_auto_holds(&mut self, of: ItemRef, ty: Ty, parent: ItemRef, at: u32) -> bool {
        self.auto_stack.push((of, ty));
        let holds = if self.instance.is_some_and(|(cur, _)| cur == parent) {
            let Some(captures) = self
                .out
                .closures
                .get(&crate::body::ExprId(at))
                .map(|c| c.captures.clone())
            else {
                self.auto_stack.pop();
                return true;
            };
            captures.into_iter().all(|cap| {
                let local_ty = self.resolve(self.out.locals[cap.local as usize].ty);
                let cap_ty = cap.ty(self.cx, local_ty).unwrap_or(local_ty);
                let held = match cap.by {
                    super::super::By::Ref => self.cx.types.intern(Kind::Ref(Mut::Not, cap_ty)),
                    super::super::By::Mut => self.cx.types.intern(Kind::Ref(Mut::Mut, cap_ty)),
                    super::super::By::Value => cap_ty,
                };
                self.thread_auto_holds(of, held)
            })
        } else if let Some(info) = self.cx.ensure_closure(ty) {
            self.thread_auto_holds(of, info.env)
        } else {
            true
        };
        self.auto_stack.pop();
        holds
    }

    fn adt_auto_holds(
        &mut self,
        of: ItemRef,
        ty: Ty,
        item: ItemRef,
        args: super::super::super::substs::Substs,
    ) -> bool {
        let holders = self.holders(of, ty, &[]);
        if self.cx.negative_impl(of, Head::Adt(item), &holders) {
            return false;
        }
        self.auto_stack.push((of, ty));
        let holds = if self
            .cx
            .has_specific_trait_impl_for_head(of, Head::Adt(item), &holders)
        {
            if self.has_vars(ty) || self.has_params(ty) {
                true
            } else {
                self.select(of, ty, &[])
                    .is_ok_and(|picked| picked.is_some())
            }
        } else if self.cx.lang(item) == Some("phantom_data") {
            let inner = self.cx.types.substs(args).first().copied();
            inner.is_none_or(|t| self.thread_auto_holds(of, t))
        } else {
            self.parts(ty)
                .into_iter()
                .all(|part| self.thread_auto_holds(of, part))
        };
        self.auto_stack.pop();
        holds
    }

    pub(in super::super) fn bound_trait_at(
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

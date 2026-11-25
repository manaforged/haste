use anyhow::Result;

use crate::body::{Body, Bound};
use crate::names::{ItemRef, ModRef};
use crate::outline::item::ItemKind;

use super::super::super::substs::Substs;
use super::super::super::ty::{Kind, Ty};
use super::super::Infer;
use super::super::project::path_constraints;
use super::super::scope::{named_bounds, predicate_bounds, type_params};
use super::PolyPhase;

const MAX_ENV: usize = 256;
const MAX_CONSTRAINT_DEPTH: u32 = 6;

impl Infer<'_, '_> {
    pub(in super::super) fn poly_fallback(&mut self) {
        if let Some(poly) = &mut self.poly {
            poly.phase = PolyPhase::Fallback;
        }
    }

    pub(in super::super) fn poly_fell_back(&self) -> bool {
        self.poly
            .as_ref()
            .is_none_or(|poly| poly.phase == PolyPhase::Fallback)
    }

    pub(in super::super) fn init_poly_env(&mut self, item: ItemRef) -> Result<()> {
        if self.cx.owner_trait(item).is_some()
            && self.cx.owner_impl(item).is_none()
            && let Some(poly) = &mut self.poly
        {
            poly.unsized_params.insert(0);
        }
        if let Some(own) = self.own_trait()? {
            self.push_assumption(own);
        }
        self.collect_header_env()?;
        if let Some(poly) = &mut self.poly
            && poly.phase == PolyPhase::HeaderInit
        {
            poly.phase = PolyPhase::Active;
        }
        self.close_poly_env()?;
        self.prune_sized_params();
        Ok(())
    }

    fn collect_header_env(&mut self) -> Result<()> {
        let params = self.generics.clone();
        let self_ty = self.self_ty;
        for (header, src, home) in self.headers()? {
            let (arena, generics, predicates) = header.parts();
            self.in_source_home(src, home, |infer| {
                for &(name, ty) in &params {
                    for bound in named_bounds(arena, src, (generics, predicates), name) {
                        infer.note_maybe_unsized(arena, home, bound, ty);
                        infer.collect_env_bound(arena, home, (ty, bound), 0)?;
                    }
                }
                if let Some(self_ty) = self_ty {
                    for bound in predicate_bounds(arena, src, predicates, "Self") {
                        infer.note_maybe_unsized(arena, home, bound, self_ty);
                        infer.collect_env_bound(arena, home, (self_ty, bound), 0)?;
                    }
                }
                for (ty, bound) in infer.typed_predicates(arena, src, predicates) {
                    infer.note_maybe_unsized(arena, home, bound, ty);
                    infer.collect_env_bound(arena, home, (ty, bound), 0)?;
                }
                Ok(())
            })?;
        }
        Ok(())
    }

    pub(super) fn bound_relaxes_sized(&mut self, arena: &Body, home: ModRef, bound: u32) -> bool {
        match arena.bounds[bound as usize] {
            Bound::Trait { maybe: true, .. } => true,
            Bound::Trait {
                path, maybe: false, ..
            } => self
                .trait_named(arena, path, home)
                .and_then(|of| self.cx.lang(of))
                .is_some_and(|lang| matches!(lang, "pointee_sized" | "meta_sized")),
            _ => false,
        }
    }

    fn note_maybe_unsized(&mut self, arena: &Body, home: ModRef, bound: u32, ty: Ty) {
        if self.bound_relaxes_sized(arena, home, bound)
            && let Kind::Param(idx) = *self.cx.types.kind(self.resolve(ty))
            && let Some(poly) = &mut self.poly
        {
            poly.unsized_params.insert(idx);
        }
    }

    fn collect_env_bound(
        &mut self,
        arena: &Body,
        home: ModRef,
        (ty, bound): (Ty, u32),
        depth: u32,
    ) -> Result<()> {
        if depth > MAX_CONSTRAINT_DEPTH {
            return Ok(());
        }
        let Some((of, path)) = self.bound_trait_at(arena, bound, home) else {
            return Ok(());
        };
        let args = self.bound_args(of, arena, path, ty)?;
        let full: Vec<Ty> = std::iter::once(ty).chain(args.iter().copied()).collect();
        self.push_assumption((of, full));
        if self.is_callable_trait(of)
            && let Some((_, sig)) = self.callable_bound(arena, home, bound)?
        {
            self.push_fn_bound(ty, of, sig);
        }
        for (assoc_name, assoc_ty) in self.path_bindings(arena, path)? {
            self.record_assoc_eq((of, args.clone()), ty, assoc_name, assoc_ty)?;
        }
        for (name_tok, sub_bounds) in path_constraints(arena, path) {
            let assoc_name = self.src.text(name_tok).to_string();
            let Some((owner, owner_args)) =
                self.assoc_owner((of, args.clone()), ty, &assoc_name, 0)?
            else {
                continue;
            };
            let assoc = self.project(owner, ty, owner_args, &assoc_name)?;
            for &sub in arena.list(sub_bounds) {
                self.collect_env_bound(arena, home, (assoc, sub), depth + 1)?;
            }
        }
        Ok(())
    }

    fn close_poly_env(&mut self) -> Result<()> {
        let mut at = 0;
        while let Some(pair) = self.poly.as_ref().and_then(|poly| {
            (at < poly.assumptions.len() && poly.assumptions.len() < MAX_ENV)
                .then(|| poly.assumptions[at].clone())
        }) {
            let (of, substs) = pair;
            for ((parent, parent_substs), bindings) in self.super_bindings(of, &substs)? {
                self.push_assumption((parent, parent_substs.clone()));
                if self.is_callable_trait(of) && self.is_callable_trait(parent) {
                    self.propagate_fn_bound(substs[0], of, parent);
                }
                for (assoc_name, assoc_ty) in bindings {
                    self.record_assoc_eq(
                        (parent, parent_substs[1..].to_vec()),
                        substs[0],
                        assoc_name,
                        assoc_ty,
                    )?;
                }
            }
            self.collect_alias_decl_bounds(of, &substs)?;
            at += 1;
        }
        Ok(())
    }

    fn collect_alias_decl_bounds(&mut self, of: ItemRef, substs: &[Ty]) -> Result<()> {
        let def = self.cx.trait_generics(of)?;
        let params: Vec<_> = type_params(&def.1.params).copied().collect();
        if substs.len() != params.len() + 1 {
            return Ok(());
        }
        let members: Vec<ItemRef> = self
            .cx
            .children(of)
            .into_iter()
            .filter(|&m| self.cx.item(m).1.kind == ItemKind::Type)
            .collect();
        for member in members {
            let bounds = self.cx.trait_generics(member)?;
            if bounds.1.supers.len == 0 {
                continue;
            }
            let member_name = self.cx.name(member).to_string();
            let Some(value) =
                self.picked_alias(of, substs[0], &substs[1..], (&member_name, &[]))?
            else {
                continue;
            };
            let frame =
                self.item_frame(of, (&params, &substs[1..]), (Some(substs[0]), Some(of)))?;
            let (home, src) = (frame.home, frame.src);
            self.with_frame(frame, |infer| {
                infer.in_source(src, |infer| {
                    for &bound in bounds.0.list(bounds.1.supers) {
                        infer.collect_env_bound(&bounds.0, home, (value, bound), 1)?;
                    }
                    Ok(())
                })
            })?;
        }
        Ok(())
    }

    fn prune_sized_params(&mut self) {
        let Some(sized) = self.cx.lang_item("sized") else {
            return;
        };
        let Some(poly) = &mut self.poly else {
            return;
        };
        let relaxed_params: Vec<u16> = poly.unsized_params.iter().copied().collect();
        for idx in relaxed_params {
            let param_ty = self.cx.types.intern(Kind::Param(idx));
            if poly
                .assumptions
                .iter()
                .any(|(of, tys)| *of == sized && tys.as_slice() == [param_ty])
            {
                poly.unsized_params.remove(&idx);
            }
        }
    }

    fn push_assumption(&mut self, entry: (ItemRef, Vec<Ty>)) {
        if let Some(poly) = &mut self.poly
            && poly.assumptions.len() < MAX_ENV
            && !poly.assumptions.contains(&entry)
        {
            poly.assumptions.push(entry);
        }
    }

    fn push_fn_bound(&mut self, ty: Ty, of: ItemRef, sig: Ty) {
        if let Kind::FnPtr(inputs, output) = *self.cx.types.kind(sig)
            && let Some(once) = self.callable_output_trait(of)
        {
            let tuple = self.cx.types.intern(Kind::Tuple(inputs));
            let trait_args = self.cx.types.intern_substs(&[tuple]);
            self.push_assoc_eq_raw(once, ty, trait_args, "Output".to_string(), output);
        }
        if let Some(poly) = &mut self.poly
            && !poly.fn_bounds.contains(&(ty, of, sig))
        {
            poly.fn_bounds.push((ty, of, sig));
        }
    }

    fn propagate_fn_bound(&mut self, ty: Ty, from: ItemRef, to: ItemRef) {
        let sigs: Vec<Ty> = self
            .poly
            .as_ref()
            .map(|p| {
                p.fn_bounds
                    .iter()
                    .filter(|&&(t, o, _)| t == ty && o == from)
                    .map(|&(_, _, s)| s)
                    .collect()
            })
            .unwrap_or_default();
        for sig in sigs {
            self.push_fn_bound(ty, to, sig);
        }
    }

    fn record_assoc_eq(
        &mut self,
        (of, args): (ItemRef, Vec<Ty>),
        ty: Ty,
        name: String,
        eq_ty: Ty,
    ) -> Result<()> {
        let Some((owner, owner_args)) = self.assoc_owner((of, args), ty, &name, 0)? else {
            return Ok(());
        };
        let interned = self.cx.types.intern_substs(&owner_args);
        self.push_assoc_eq_raw(owner, ty, interned, name, eq_ty);
        Ok(())
    }

    fn push_assoc_eq_raw(&mut self, owner: ItemRef, ty: Ty, args: Substs, name: String, eq_ty: Ty) {
        if let Some(poly) = &mut self.poly
            && !poly
                .assoc_eqs
                .iter()
                .any(|(o, t, a, n, _)| *o == owner && *t == ty && *a == args && *n == name)
        {
            poly.assoc_eqs.push((owner, ty, args, name, eq_ty));
        }
    }
}

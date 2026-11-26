use anyhow::{Result, bail};
use rustc_hash::FxHashSet as HashSet;

use crate::body::{Body, ExprId, List, PathId};
use crate::names::{ItemRef, ModRef, Primitive, plain};
use crate::outline::item::ItemKind;

use super::super::super::substs::{ProjDef, ProjId, Substs};
use super::super::super::summary::{ConstHole, PolyObligation};
use super::super::super::ty::{Kind, Ty};
use super::super::select::Picked;
use super::super::{Infer, Res};
use super::push_proj_topo;

impl Infer<'_, '_> {
    pub(in super::super) fn poly_sized(&mut self, ty: Ty) -> bool {
        let tail = self.tail_ty(ty).unwrap_or_else(|_| self.resolve(ty));
        let tail = self.resolve(tail);
        match *self.cx.types.kind(tail) {
            Kind::Slice(_) | Kind::Prim(Primitive::Str) | Kind::Dyn(..) => false,
            Kind::Param(idx) => self
                .poly
                .as_ref()
                .is_none_or(|poly| !poly.unsized_params.contains(&idx)),
            Kind::Proj(id) => self.proj_is_sized(tail, id),
            _ => true,
        }
    }

    fn proj_is_sized(&mut self, proj_ty: Ty, id: ProjId) -> bool {
        if let Some(sized) = self.cx.lang_item("sized")
            && self.poly.as_ref().is_some_and(|p| {
                p.assumptions
                    .iter()
                    .any(|(of, tys)| *of == sized && tys.as_slice() == [proj_ty])
            })
        {
            return true;
        }
        let def = self.cx.types.proj(id);
        let (Ok(bounds), Ok(assoc_home)) = (
            self.cx.trait_generics(def.assoc),
            self.cx.module_of(def.assoc),
        ) else {
            return false;
        };
        let assoc_src = &self.cx.file(def.assoc).source;
        let relaxed = self.in_source(assoc_src, |infer| {
            Ok(bounds
                .0
                .list(bounds.1.supers)
                .iter()
                .any(|&b| infer.bound_relaxes_sized(&bounds.0, assoc_home, b)))
        });
        !relaxed.unwrap_or(true)
    }

    pub(in super::super) fn poly_is_copy(&mut self, ty: Ty) -> Result<bool> {
        let Some(copy) = self.cx.lang_item("copy") else {
            return Ok(false);
        };
        self.poly_holds_in_env(copy, ty, &[])
    }

    pub(in super::super) fn poly_fn_sig(&self, ty: Ty) -> Option<(Vec<Ty>, Ty)> {
        let ty = self.resolve(ty);
        let poly = self.poly.as_ref()?;
        let &(_, _, sig) = poly.fn_bounds.iter().find(|&&(bound_ty, of, _)| {
            self.resolve(bound_ty) == ty && self.is_sync_callable_trait(of)
        })?;
        let Kind::FnPtr(params, ret) = *self.cx.types.kind(sig) else {
            return None;
        };
        Some((self.cx.types.substs(params).to_vec(), ret))
    }

    pub(in super::super) fn poly_may_implement(
        &mut self,
        of: ItemRef,
        self_ty: Ty,
        args: &[Ty],
    ) -> Result<bool> {
        let Some(assumptions) = self.poly.as_ref().map(|poly| poly.assumptions.clone()) else {
            return Ok(false);
        };
        let wanted: Vec<Ty> = std::iter::once(self_ty)
            .chain(args.iter().copied())
            .collect();
        for (held, types) in assumptions {
            if held == of
                && types.len() == wanted.len()
                && self.probe(|infer| {
                    wanted
                        .iter()
                        .zip(&types)
                        .all(|(&a, &b)| infer.unifies(a, b))
                })
            {
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub(in super::super) fn poly_holds_in_env(
        &mut self,
        of: ItemRef,
        self_ty: Ty,
        args: &[Ty],
    ) -> Result<bool> {
        let Some(assumptions) = self.poly.as_ref().map(|poly| poly.assumptions.clone()) else {
            return Ok(false);
        };
        let wanted: Vec<Ty> = std::iter::once(self_ty)
            .chain(args.iter().copied())
            .collect();
        if !self.obligation_open(self_ty, args) {
            let resolved: Vec<Ty> = wanted.into_iter().map(|t| self.substitute(t)).collect();
            return Ok(assumptions.iter().any(|(held, types)| {
                *held == of
                    && types.len() == resolved.len()
                    && types
                        .iter()
                        .zip(&resolved)
                        .all(|(&a, &b)| self.substitute(a) == b)
            }));
        }
        let mut matched = Vec::new();
        for (held, types) in assumptions {
            if held == of
                && types.len() == wanted.len()
                && !matched.contains(&types)
                && self.probe(|infer| {
                    wanted
                        .iter()
                        .zip(&types)
                        .all(|(&a, &b)| infer.unifies(a, b))
                })
            {
                matched.push(types);
            }
        }
        if let [unique] = &matched[..] {
            wanted
                .iter()
                .zip(unique)
                .try_for_each(|(&a, &b)| self.unify(a, b))?;
        }
        Ok(!matched.is_empty())
    }

    pub(in super::super) fn poly_bound_check(
        &mut self,
        (arena, home, bound): (&Body, ModRef, u32),
        (of, path, ty): (ItemRef, PathId, Ty),
        depth: u32,
    ) -> Result<Option<bool>> {
        let kind = self.cx.types.kind(self.resolve(ty));
        if self.is_callable_trait(of) && matches!(kind, Kind::Param(_) | Kind::Proj(_)) {
            return self
                .poly_callable_bound_holds((arena, home, bound), (of, path, ty), depth)
                .map(Some);
        }
        let args = self.bound_args(of, arena, path, ty)?;
        if self.has_params(ty) || self.any_has_params(&args) {
            let ty = self.substitute(ty);
            let args: Vec<Ty> = args.into_iter().map(|a| self.substitute(a)).collect();
            if self.poly_env_holds((arena, home, path), (of, ty, &args), depth)? {
                return Ok(Some(true));
            }
        }
        Ok(None)
    }

    pub(in super::super) fn poly_env_holds(
        &mut self,
        (arena, home, path): (&Body, ModRef, PathId),
        (of, ty, args): (ItemRef, Ty, &[Ty]),
        depth: u32,
    ) -> Result<bool> {
        if !self.poly_holds_in_env(of, ty, args)? {
            return Ok(false);
        }
        self.assoc_bounds_hold((arena, home, path), (of, ty, args), depth)
    }

    pub(in super::super) fn poly_picked_alias(
        &mut self,
        of: ItemRef,
        ty: Ty,
        args: &[Ty],
        (name, own): (&str, &[Ty]),
    ) -> Result<Option<Ty>> {
        if self.has_vars(ty)
            || args.iter().any(|&a| self.has_vars(a))
            || own.iter().any(|&o| self.has_vars(o))
        {
            return Ok(None);
        }
        let ty = self.zonk(ty);
        let args: Vec<Ty> = args.iter().map(|&a| self.zonk(a)).collect();
        let own: Vec<Ty> = own.iter().map(|&o| self.zonk(o)).collect();
        if self.cx.lang(of) == Some("discriminant_kind") {
            return self.discriminant_ty(ty);
        }
        let interned_args = self.cx.types.intern_substs(&args);
        if let Some(fast) = self.poly_fast_alias(of, ty, interned_args, (name, &own))? {
            return Ok(Some(fast));
        }
        self.poly_selected_or_proj(of, ty, (&args, interned_args), (name, &own))
    }

    fn poly_fast_alias(
        &mut self,
        of: ItemRef,
        ty: Ty,
        interned_args: Substs,
        (name, own): (&str, &[Ty]),
    ) -> Result<Option<Ty>> {
        if own.is_empty()
            && let Some(eq_ty) = self.lookup_assoc_eq(of, ty, interned_args, name)
        {
            return Ok(Some(eq_ty));
        }
        self.builtin_alias(of, ty, name)
    }

    fn poly_selected_or_proj(
        &mut self,
        of: ItemRef,
        ty: Ty,
        (args, interned_args): (&[Ty], Substs),
        (name, own): (&str, &[Ty]),
    ) -> Result<Option<Ty>> {
        let selected = self.select(of, ty, args)?;
        if let Some(picked) = &selected
            && let Some(concrete) =
                self.non_default_impl_alias(of, ty, args, picked.clone(), (name, own))?
        {
            return Ok(Some(concrete));
        }
        if selected.is_none() && !self.poly_holds_in_env(of, ty, args)? {
            return Ok(None);
        }
        let Some(assoc) = self.cx.trait_member(of, name) else {
            return Ok(None);
        };
        self.poly_prove(of, ty, args);
        let own_args = self.cx.types.intern_substs(own);
        let proj_id = self.cx.types.intern_proj(ProjDef {
            of,
            self_ty: ty,
            args: interned_args,
            assoc,
            own: own_args,
        });
        if let Some(poly) = &mut self.poly {
            let mut seen = HashSet::default();
            push_proj_topo(&self.cx.types, poly, &mut seen, proj_id);
        }
        Ok(Some(self.cx.types.intern(Kind::Proj(proj_id))))
    }

    fn non_default_impl_alias(
        &mut self,
        of: ItemRef,
        ty: Ty,
        args: &[Ty],
        picked: Picked,
        (name, own): (&str, &[Ty]),
    ) -> Result<Option<Ty>> {
        let (block, impl_args) = self
            .impl_member(of, (ty, args), picked.clone(), name)?
            .map_or(picked, |(_, found)| found);
        if self.cx.specializable(block) {
            return Ok(None);
        }
        let frame = self.impl_frame_of(block, &impl_args, Some(ty))?;
        match self.with_frame(frame, |infer| infer.impl_alias(block, (name, own)))? {
            Some(found) => Ok(Some(found)),
            None => bail!(
                "the impl of `{}` for {} has no type `{name}`",
                self.cx.name(of),
                self.show(ty)
            ),
        }
    }

    fn lookup_assoc_eq(&mut self, of: ItemRef, ty: Ty, args: Substs, name: &str) -> Option<Ty> {
        let len = self.poly.as_ref()?.assoc_eqs.len();
        for idx in 0..len {
            let (o, t, a, eq, same_name) = {
                let (o, t, a, n, eq) = &self.poly.as_ref()?.assoc_eqs[idx];
                (*o, *t, *a, *eq, n == name)
            };
            if o == of && a == args && same_name && self.substitute(t) == ty {
                return Some(eq);
            }
        }
        None
    }

    pub(in super::super) fn poly_prove(&mut self, of: ItemRef, self_ty: Ty, args: &[Ty]) {
        let self_ty = self.resolve(self_ty);
        let resolved: Vec<Ty> = args.iter().map(|&a| self.resolve(a)).collect();
        let interned = self.cx.types.intern_substs(&resolved);
        let entry = PolyObligation::Prove {
            of,
            self_ty,
            args: interned,
        };
        if let Some(poly) = &mut self.poly
            && !poly.obligations.contains(&entry)
        {
            poly.obligations.push(entry);
        }
    }

    pub(in super::super) fn poly_settle_obligation(
        &mut self,
        of: ItemRef,
        self_ty: Ty,
        args: &[Ty],
    ) -> Result<bool> {
        if self.cx.lang(of) == Some("unsize") {
            if !args
                .first()
                .copied()
                .is_some_and(|to| self.unsize_holds(of, self_ty, to).unwrap_or(false))
            {
                self.poly_fallback();
            }
            return Ok(true);
        }
        if self.compiler_decides(of, self_ty) {
            if !self.compiler_holds(of, self_ty) {
                self.poly_fallback();
            }
            return Ok(true);
        }
        if self.compiler_impl(of, self_ty) || self.object_holds(of, self_ty, args) == Some(true) {
            return Ok(true);
        }
        if self.poly_holds_in_env(of, self_ty, args)? || self.select(of, self_ty, args)?.is_some() {
            self.poly_prove(of, self_ty, args);
            return Ok(true);
        }
        if self.poly_strict_param_obligation(of, self_ty, args) {
            return Ok(false);
        }
        self.poly_fallback();
        Ok(true)
    }

    fn poly_strict_param_obligation(&self, of: ItemRef, self_ty: Ty, args: &[Ty]) -> bool {
        self.root_is_user()
            && !self.poly_fell_back()
            && self.has_params(self_ty)
            && !self.has_vars(self_ty)
            && !self.cx.types.has_projs(self_ty)
            && self.checks_bound_trait(of)
            && !args
                .iter()
                .any(|&a| self.has_vars(a) || self.cx.types.has_projs(self.resolve(a)))
    }

    pub(in super::super) fn select_or_assume(
        &mut self,
        of: ItemRef,
        self_ty: Ty,
        args: &[Ty],
    ) -> Result<bool> {
        if self.select(of, self_ty, args)?.is_some() {
            return Ok(true);
        }
        if self.poly.is_some() {
            let full: Vec<Ty> = std::iter::once(self_ty)
                .chain(args.iter().copied())
                .collect();
            return self.assume(of, &full);
        }
        Ok(false)
    }

    pub(in super::super) fn select_or_prove(
        &mut self,
        of: ItemRef,
        self_ty: Ty,
        args: &[Ty],
    ) -> Result<bool> {
        let has_params =
            self.poly.is_some() && (self.has_params(self_ty) || self.any_has_params(args));
        if self
            .trial(|infer| infer.select(of, self_ty, args))?
            .is_some()
        {
            if has_params {
                self.poly_prove(of, self_ty, args);
            }
            return Ok(true);
        }
        if has_params {
            let full: Vec<Ty> = std::iter::once(self_ty)
                .chain(args.iter().copied())
                .collect();
            self.assume(of, &full)?;
            if self.poly_holds_in_env(of, self_ty, args)? {
                self.poly_prove(of, self_ty, args);
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub(in super::super) fn poly_only_right(&mut self, of: ItemRef, left: Ty) -> Option<Ty> {
        let candidates: Vec<(Ty, Ty)> = self
            .poly
            .as_ref()?
            .assumptions
            .iter()
            .filter(|(held, types)| *held == of && types.len() == 2)
            .map(|(_, types)| (types[0], types[1]))
            .collect();
        let mut found = None;
        for (lhs, rhs) in candidates {
            if self.substitute(lhs) == left {
                let mapped = self.substitute(rhs);
                if found.replace(mapped).is_some_and(|prev| prev != mapped) {
                    return None;
                }
            }
        }
        found
    }

    pub(in super::super) fn poly_call(
        &mut self,
        expr: ExprId,
        found: Ty,
        callee_ty: Ty,
        args: List,
    ) -> Result<Ty> {
        let Some((method, params, ret)) = self.poly_callable_target(callee_ty) else {
            self.poly_fallback();
            bail!(
                "calling {} requires concrete callee type",
                self.show(callee_ty)
            );
        };
        let arg_ids = self.body.list(args).to_vec();
        if arg_ids.len() != params.len() {
            return Err(self.diag_arity(Some(expr), params.len(), arg_ids.len()));
        }
        for (&arg, &param) in arg_ids.iter().zip(&params) {
            self.call_arg(None, ExprId(arg), (param, None), &mut None)?;
        }
        let tuple = self.cx.types.tuple(&params);
        self.out.res.insert(
            expr,
            Res::fn_of(&self.cx.types, method, &[callee_ty, tuple]),
        );
        if let Some(poly) = &mut self.poly {
            poly.obligations
                .push(PolyObligation::Call { expr, found, tuple });
        }
        Ok(ret)
    }

    pub(in super::super) fn poly_offset_of(
        &mut self,
        expr: ExprId,
        mut current: Ty,
        fields: List,
    ) -> Result<Ty> {
        let mut steps = Vec::new();
        for &at in self.body.list(fields) {
            for name in self.src.text(at).split('.').map(plain) {
                let (index, field) = self.poly_offset_step(current, name)?;
                steps.push((current, index));
                current = field;
            }
        }
        if let Some(poly) = &mut self.poly {
            poly.offset_holes.push(ConstHole::OffsetOf {
                expr,
                steps: steps.into_boxed_slice(),
            });
        }
        Ok(self.cx.types.prim(Primitive::Usize))
    }

    fn poly_offset_step(&mut self, current: Ty, name: &str) -> Result<(u32, Ty)> {
        let kind = self.cx.types.kind(self.resolve(current)).clone();
        if matches!(kind, Kind::Param(_) | Kind::Proj(_)) {
            self.poly_fallback();
            bail!("offset_of on open parameter");
        }
        if let Kind::Adt(item, _) = kind
            && self.cx.item(item).1.kind == ItemKind::Enum
        {
            self.poly_fallback();
            bail!("offset_of on enum variant");
        }
        self.field_at(current, name)
    }
}

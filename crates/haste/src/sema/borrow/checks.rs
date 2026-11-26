mod args;
pub(super) mod closure;
mod types;

use anyhow::Result;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::body::sig::{Generic, Predicate, Signature};
use crate::body::{Body, Bound, Expr, ExprId, List, Pat, PatId, UnOp};
use crate::names::ItemRef;
use crate::outline::item::ItemKind;
use crate::outline::source::Source;
use crate::sema::Context;
use crate::sema::infer::Res;
use crate::sema::ty::{Kind, Ty};

use super::bind::push_source_loans;
use super::places::{FieldKey, Place, PlaceStep, push_step};
use super::propagate::BorrowSource;
use super::{Checker, PartialMove, State, is_temp_holder, loan_conflicts_borrow, merge_states};

pub(super) const TEMP_TARGET_BASE: u32 = u32::MAX - 262_144;

pub(super) use closure::ClosureEffects;
pub(in crate::sema) use types::is_copy_ty;

fn trait_has_static_supertrait(
    cx: &mut Context<'_>,
    tr: ItemRef,
    seen: &mut HashSet<ItemRef>,
    cache: &mut HashMap<ItemRef, bool>,
) -> bool {
    if let Some(&cached) = cache.get(&tr) {
        return cached;
    }
    if !seen.insert(tr) {
        return false;
    }
    let Ok(tg) = cx.trait_generics(tr) else {
        return false;
    };
    let (tbody, tgen) = (&tg.0, &tg.1);
    let tsrc = &cx.file(tr).source;
    let supers = tbody.list(tgen.supers);
    if supers.iter().any(
        |&b| matches!(tbody.bounds[b as usize], Bound::Lifetime(lt) if tsrc.text(lt) == "'static"),
    ) {
        cache.insert(tr, true);
        return true;
    }
    let Ok(thome) = cx.module_of(tr) else {
        return false;
    };
    let sub_traits = super::super::infer::detached(cx, tr, |infer| {
        Ok(supers
            .iter()
            .filter_map(|&b| infer.bound_trait_at(tbody, b, thome).map(|(sub, _)| sub))
            .collect::<Vec<_>>())
    })
    .unwrap_or_default();
    let hit = sub_traits
        .into_iter()
        .any(|sub| trait_has_static_supertrait(cx, sub, seen, cache));
    if hit {
        cache.insert(tr, true);
    }
    hit
}

fn param_bounds_static(
    cx: &mut Context<'_>,
    scope: (ItemRef, &Body, &Source),
    (generics, preds): (&[Generic], &[Predicate]),
    name: &str,
    cache: &mut HashMap<ItemRef, bool>,
) -> bool {
    let (home_item, body, src) = scope;
    let bounds = crate::sema::infer::scope::named_bounds(body, src, (generics, preds), name);
    if bounds.iter().any(
        |&b| matches!(body.bounds[b as usize], Bound::Lifetime(lt) if src.text(lt) == "'static"),
    ) {
        return true;
    }
    let trait_bounds: Vec<u32> = bounds
        .into_iter()
        .filter(|&b| matches!(body.bounds[b as usize], Bound::Trait { maybe: false, .. }))
        .collect();
    if trait_bounds.is_empty() {
        return false;
    }
    let Ok(home) = cx.module_of(home_item) else {
        return false;
    };
    let traits = super::super::infer::detached(cx, home_item, |infer| {
        Ok(trait_bounds
            .into_iter()
            .filter_map(|b| infer.bound_trait_at(body, b, home).map(|(tr, _)| tr))
            .collect::<Vec<_>>())
    })
    .unwrap_or_default();
    traits
        .into_iter()
        .any(|tr| trait_has_static_supertrait(cx, tr, &mut HashSet::default(), cache))
}

/// Returns `(static_param_indices, sig_static_param_names)` for type parameters bounded by `'static` directly or via supertraits.
pub(super) fn compute_static_type_params(
    cx: &mut Context<'_>,
    item: ItemRef,
    sig: &Signature,
    src: &Source,
) -> (Vec<u16>, Vec<String>) {
    let mut cache = HashMap::default();
    let mut out = Vec::new();
    let mut sig_names = Vec::new();
    let mut idx = 0u16;
    if let Some(impl_ref) = cx.owner_impl(item) {
        if let Ok(header) = cx.impl_header(impl_ref) {
            let impl_src = &cx.impl_file(impl_ref).source;
            for g in crate::sema::infer::scope::type_params(&header.generics) {
                let name = impl_src.text(g.name);
                if param_bounds_static(
                    cx,
                    (item, &header.body, impl_src),
                    (&header.generics, &header.predicates),
                    name,
                    &mut cache,
                ) || param_bounds_static(
                    cx,
                    (item, &sig.body, src),
                    (&sig.generics, &sig.predicates),
                    name,
                    &mut cache,
                ) {
                    out.push(idx);
                }
                idx += 1;
            }
        }
    } else if let Some(trait_ref) = cx.owner_trait(item) {
        if trait_has_static_supertrait(cx, trait_ref, &mut HashSet::default(), &mut cache)
            || param_bounds_static(
                cx,
                (item, &sig.body, src),
                (&sig.generics, &sig.predicates),
                "Self",
                &mut cache,
            )
        {
            out.push(idx);
        }
        idx += 1;
        if let Ok(tg) = cx.trait_generics(trait_ref) {
            let (tbody, tgen) = &*tg;
            let tsrc = &cx.file(trait_ref).source;
            for g in crate::sema::infer::scope::type_params(&tgen.params) {
                let name = tsrc.text(g.name);
                if param_bounds_static(
                    cx,
                    (trait_ref, tbody, tsrc),
                    (&tgen.params, &tgen.predicates),
                    name,
                    &mut cache,
                ) || param_bounds_static(
                    cx,
                    (item, &sig.body, src),
                    (&sig.generics, &sig.predicates),
                    name,
                    &mut cache,
                ) {
                    out.push(idx);
                }
                idx += 1;
            }
        }
    }
    for g in crate::sema::infer::scope::type_params(&sig.generics) {
        let name = src.text(g.name);
        if param_bounds_static(
            cx,
            (item, &sig.body, src),
            (&sig.generics, &sig.predicates),
            name,
            &mut cache,
        ) {
            out.push(idx);
            sig_names.push(name.to_owned());
        }
        idx += 1;
    }
    (out, sig_names)
}

pub(super) fn is_temp_target(target: u32) -> Option<u32> {
    (target >= TEMP_TARGET_BASE && target < super::ITER_HOLDER_BASE)
        .then(|| target - TEMP_TARGET_BASE)
}

pub(super) fn temp_target_for(depth: u32) -> u32 {
    TEMP_TARGET_BASE + depth.min(65_000)
}

impl Checker<'_, '_> {
    pub(super) fn check_stmt_temp_escape(
        &self,
        state: &State,
        min_depth: u32,
        check_locals: &[u32],
        outer_live: &HashSet<u32>,
    ) -> Result<()> {
        for l in &state.loans {
            let Some(holder) = l.holder else { continue };
            let Some(t_depth) = is_temp_target(l.target) else {
                continue;
            };
            if t_depth >= min_depth
                && !(self.is_const_or_static_item()
                    && self.temp_target_allowed(l.mutable)
                    && !l.site.is_some_and(|e| self.is_non_promotable_temp(e)))
                && (check_locals.contains(&holder) || outer_live.contains(&holder))
            {
                return Err(self.borrow_err(l.site, "temporary value dropped while borrowed"));
            }
        }
        Ok(())
    }

    pub(super) fn check_assign_escape(
        &mut self,
        p: &Place,
        sources: &[BorrowSource],
        state: &mut State,
    ) -> Result<()> {
        if sources.is_empty() {
            return Ok(());
        }
        let mut targets: Vec<u32> = if p.through_ref {
            state
                .loans
                .iter()
                .filter(|l| l.holder == Some(p.local) && l.mutable)
                .map(|l| l.target)
                .collect()
        } else {
            Vec::new()
        };
        if p.through_ref
            && (!self.is_param_local(p.local) || self.closure_depth > 0)
            && !targets.contains(&p.local)
        {
            targets.push(p.local);
        }
        if (p.through_ref || p.through_deref)
            && (self.is_param_local(p.local) || targets.iter().any(|&t| self.is_param_local(t)))
        {
            for s in sources {
                let allowed_ref = self.is_ref_ty(self.local_ty(s.target))
                    && (s.through_ref || self.is_closure_param(s.target));
                if is_temp_target(s.target).is_some()
                    || (!self.from_caller(s.target, s.through_ref) && !allowed_ref)
                {
                    return Err(self.does_not_live_long_enough(None, s.target));
                }
            }
        }
        if self.closure_depth > 0
            && self.scope_depth.get(p.local as usize).copied().unwrap_or(0) < self.closure_depth
        {
            for s in sources {
                let is_cparam = self.is_closure_param(s.target);
                let cparam_borrow = is_cparam && self.ty_holds_borrow(self.local_ty(s.target));
                let allowed_ref =
                    self.is_ref_ty(self.local_ty(s.target)) && (s.through_ref || is_cparam);
                if is_temp_target(s.target).is_some()
                    || (!s.through_ref
                        && !self.is_ref_ty(self.local_ty(s.target))
                        && self.closure_by_val_caps.contains(&s.target))
                    || (!self.from_caller(s.target, s.through_ref)
                        && !allowed_ref
                        && !cparam_borrow
                        && self
                            .scope_depth
                            .get(s.target as usize)
                            .copied()
                            .unwrap_or(0)
                            >= self.closure_depth)
                {
                    return Err(self.borrow_err(None, "borrowed data escapes outside of closure"));
                }
            }
        }
        if p.through_ref {
            for holder in targets {
                state.set_tuple_sources(holder, None);
                if self.ty_holds_borrow(self.local_ty(holder)) {
                    super::bind::push_source_loans_filtered(
                        &mut state.loans,
                        Some(holder),
                        sources,
                        false,
                        self.current_expr,
                        |s| s.target != holder,
                        |s| s.mutable,
                    );
                }
            }
        }
        Ok(())
    }

    pub(super) fn propagate_assign_sources(
        &mut self,
        p: &Place,
        sources: &[BorrowSource],
        state: &mut State,
    ) -> Result<()> {
        if !sources.is_empty() {
            self.check_assign_escape(p, sources, state)?;
            if !p.through_ref {
                self.push_sources_for_local(p.local, sources, self.current_expr, state);
            }
        }
        Ok(())
    }

    pub(super) fn check_mut_ref_local_move(
        &self,
        dst_local: u32,
        src: ExprId,
        state: &State,
    ) -> Result<()> {
        if let Some(src_local) = self.expr_plain_local(src)
            && src_local != dst_local
            && self.is_mut_ref_ty(self.local_ty(dst_local))
            && self.is_mut_ref_ty(self.local_ty(src_local))
            && let Some(loan) = state
                .loans
                .iter()
                .find(|l| loan_conflicts_borrow(l, src_local, &[]))
        {
            return Err(self.borrow_err_labeled(
                Some(src),
                format!(
                    "cannot move out of `{}` because it is borrowed",
                    self.local_name(src_local)
                ),
                "move out of borrowed value occurs here",
                loan.site,
                "borrow occurs here",
            ));
        }
        Ok(())
    }

    pub(super) fn check_mut_ref_reborrow(
        &self,
        dst: ExprId,
        src: ExprId,
        state: &State,
    ) -> Result<()> {
        if let Some(dst_local) = self.expr_plain_local(dst) {
            self.check_mut_ref_local_move(dst_local, src, state)?;
        }
        Ok(())
    }

    pub(super) fn check_adt_drop_move(&mut self, ty: Ty) -> Result<()> {
        if self.ty_has_own_drop(ty) {
            return Err(self.borrow_err(
                None,
                "cannot move out of type which implements the `Drop` trait",
            ));
        }
        Ok(())
    }

    pub(super) fn check_fru_base(
        &mut self,
        expr: ExprId,
        fields: List,
        b: ExprId,
        state: &mut State,
        outer_live: &HashSet<u32>,
        depth: u32,
    ) -> Result<()> {
        let deref_base = matches!(*self.body.expr(b), Expr::Unary(UnOp::Deref, _));
        let Some(p) = self.root_place(b) else {
            return self.check_expr(b, state, deref_base, outer_live, true, depth);
        };
        let Some((adt_item, variant, args)) = self.struct_adt(expr) else {
            return self.check_expr(b, state, deref_base, outer_live, true, depth);
        };
        let args_vec = self.cx.types.substs(args).to_vec();
        let Ok(field_tys) = super::super::infer::detached(self.cx, self.item, |infer| {
            infer.adt_fields(adt_item, &args_vec)
        }) else {
            return self.check_expr(b, state, deref_base, outer_live, true, depth);
        };
        let pending_fields: Vec<(FieldKey, Ty)> = {
            let Ok(def) = self.cx.adt_def(adt_item) else {
                return self.check_expr(b, state, deref_base, outer_live, true, depth);
            };
            let Some(var) = def.variants.get(variant) else {
                return self.check_expr(b, state, deref_base, outer_live, true, depth);
            };
            let overridden: HashSet<&str> = self
                .body
                .list(fields)
                .iter()
                .map(|&id| self.src.text(self.body.fields[id as usize].name))
                .collect();
            let adt_src = &self.cx.file(adt_item).source;
            var.fields
                .iter()
                .enumerate()
                .filter_map(|(idx, fdef)| {
                    let (fname, key) = match fdef.name {
                        Some(n) => {
                            let s = adt_src.text(n);
                            (s.to_owned(), FieldKey::Named(s.to_owned()))
                        }
                        None => (idx.to_string(), FieldKey::Tuple(idx as u32)),
                    };
                    if overridden.contains(fname.as_str()) {
                        return None;
                    }
                    let fty = field_tys
                        .get(variant)
                        .and_then(|fs| fs.get(idx))
                        .copied()
                        .unwrap_or(Ty::NONE);
                    Some((key, fty))
                })
                .collect()
        };
        self.check_place_subexprs(b, state, outer_live, depth)?;
        let behind_ref = self.place_behind_ref(b, false);
        let mut drop_checked = false;
        let adt_ty = self.expr_ty(expr);
        for (key, fty) in pending_fields {
            let mut steps = p.steps.clone();
            push_step(&mut steps, PlaceStep::Field(key));
            if self.ty_is_copy(fty) {
                self.check_can_read(p.local, &steps, state)?;
            } else {
                if !drop_checked {
                    drop_checked = true;
                    self.check_adt_drop_move(adt_ty)?;
                }
                if behind_ref {
                    return Err(self
                        .borrow_err(Some(b), "cannot move out of a shared or mutable reference"));
                }
                self.check_can_move(p.local, &steps, state)?;
                if !p.through_ref {
                    self.push_partial_move(state, p.local, steps, Some(b));
                }
            }
        }
        Ok(())
    }

    pub(super) fn struct_adt(
        &self,
        expr: ExprId,
    ) -> Option<(crate::names::ItemRef, usize, super::super::substs::Substs)> {
        match self.typed.res.get(&expr).copied() {
            Some(Res::Ctor(item, v, args)) => Some((item, v as usize, args)),
            _ => match *self.cx.types.kind(self.expr_ty(expr)) {
                Kind::Adt(item, args) => Some((item, 0, args)),
                _ => None,
            },
        }
    }

    pub(super) fn check_place_subexprs(
        &mut self,
        expr: ExprId,
        state: &mut State,
        outer_live: &HashSet<u32>,
        depth: u32,
    ) -> Result<()> {
        match *self.body.expr(expr) {
            Expr::Paren(inner)
            | Expr::Field(inner, _)
            | Expr::TupleField(inner, _)
            | Expr::Unary(_, inner) => self.check_place_subexprs(inner, state, outer_live, depth),
            Expr::Index(base, idx) => {
                self.check_place_subexprs(base, state, outer_live, depth)?;
                self.check_expr(idx, state, false, outer_live, true, depth)
            }
            _ => Ok(()),
        }
    }

    pub(super) fn has_new_moves(
        before_moved: &[bool],
        before_partial: &[PartialMove],
        after: &State,
        locals: &[u32],
    ) -> bool {
        locals.iter().any(|&l| {
            let idx = l as usize;
            (after.moved.get(idx).copied().unwrap_or(false)
                && !before_moved.get(idx).copied().unwrap_or(false))
                || after
                    .partial_moves
                    .iter()
                    .any(|m| m.local == l && !before_partial.contains(m))
        })
    }

    pub(super) fn check_union_access(&self, expr: ExprId, base: ExprId) -> Result<()> {
        if self.unsafe_depth == 0 {
            let cont = self.field_container_ty(expr, base);
            if cont != Ty::NONE
                && matches!(self.cx.types.kind(cont), Kind::Adt(item, _) if self.cx.item(*item).1.kind == ItemKind::Union)
            {
                return Err(self.unsafe_err(
                    Some(expr),
                    "access to union field is unsafe and requires unsafe function or block",
                ));
            }
        }
        Ok(())
    }

    pub(super) fn check_union_place(&self, mut expr: ExprId) -> Result<()> {
        if self.unsafe_depth > 0 {
            return Ok(());
        }
        loop {
            match *self.body.expr(expr) {
                Expr::Paren(inner) => expr = inner,
                Expr::Field(base, _) => {
                    self.check_union_access(expr, base)?;
                    expr = base;
                }
                Expr::TupleField(base, _) | Expr::Index(base, _) => expr = base,
                _ => return Ok(()),
            }
        }
    }

    pub(super) fn check_coercion_vtables(&self) -> Result<()> {
        for (&coercion_expr, coercion) in &self.typed.coercions {
            let Some(vtable) = coercion.vtable() else {
                continue;
            };
            let Kind::Closure(parent, _, at) = *self.cx.types.kind(vtable.ty) else {
                continue;
            };
            if parent != self.item {
                continue;
            }
            let Kind::Dyn(of, ..) = *self.cx.types.kind(vtable.object) else {
                continue;
            };
            let Some(eff) = self.closure_effects.get(&ExprId(at)) else {
                continue;
            };
            let lang = self.cx.lang(of);
            self.check_closure_effects(
                Some(coercion_expr),
                eff,
                lang == Some("fn"),
                lang == Some("fn_mut"),
                false,
            )?;
        }
        Ok(())
    }

    pub(super) fn check_const_block_locals(
        &self,
        stmts: List,
        tail: Option<ExprId>,
        block_depth: u32,
    ) -> Result<()> {
        let ids_check: Vec<u32> = self.body.list(stmts).to_vec();
        let (c_before, _, c_tail) = self.block_liveness(&ids_check, tail, &HashSet::default());
        let live_in = c_before.first().unwrap_or(&c_tail);
        if let Some(&bad) = live_in.iter().find(|&&loc| {
            self.scope_depth
                .get(loc as usize)
                .is_some_and(|&d| d < block_depth)
        }) {
            return Err(self.const_err(
                None,
                format!(
                    "attempt to use a non-constant value `{}` in a constant",
                    self.local_name(bad)
                ),
            ));
        }
        Ok(())
    }

    pub(super) fn hold_drop_temp(&mut self, expr: ExprId, state: &mut State) -> Option<u32> {
        let ty = self.expr_ty(expr);
        if !self.ty_holds_borrow(ty) || !self.ty_has_borrow_drop(ty) {
            return None;
        }
        let sources = self.expr_borrow_sources(expr, state);
        if sources.is_empty() {
            return None;
        }
        let id = self.next_temp_holder();
        push_source_loans(&mut state.loans, Some(id), &sources, Some(expr));
        Some(id)
    }

    pub(super) fn pat_inspects_scrut(&self, pat: PatId) -> bool {
        !matches!(*self.body.pat(pat), Pat::Wild | Pat::Bind { sub: None, .. })
    }

    pub(super) fn base_goes_through_ref(&self, mut expr: ExprId) -> bool {
        loop {
            if self.is_ref_ty(self.expr_ty(expr)) {
                return true;
            }
            match *self.body.expr(expr) {
                Expr::Paren(inner)
                | Expr::Cast(inner, _)
                | Expr::Field(inner, _)
                | Expr::TupleField(inner, _)
                | Expr::Index(inner, _)
                | Expr::Unary(UnOp::Deref, inner) => expr = inner,
                _ => return false,
            }
        }
    }

    pub(super) fn peeled_tuple_elems(&self, mut ty: Ty) -> Option<Vec<Ty>> {
        if ty == Ty::NONE {
            return None;
        }
        while let Kind::Pat(inner, ..) = *self.cx.types.kind(ty) {
            ty = inner;
        }
        match *self.cx.types.kind(ty) {
            Kind::Tuple(elems) => Some(self.cx.types.substs(elems).to_vec()),
            _ => None,
        }
    }

    pub(super) fn mask_infer_tuple_ty(&mut self, mut ast_ty: crate::body::TyId, ty: Ty) -> Ty {
        while let crate::body::Ty::Paren(inner) = *self.body.ty(ast_ty) {
            ast_ty = inner;
        }
        match *self.body.ty(ast_ty) {
            crate::body::Ty::Infer => Ty::NONE,
            crate::body::Ty::Tuple(list) if ty != Ty::NONE => {
                let Kind::Tuple(elems) = *self.cx.types.kind(ty) else {
                    return ty;
                };
                let ast_elems: Vec<u32> = self.body.list(list).to_vec();
                let ty_elems: Vec<Ty> = self.cx.types.substs(elems).to_vec();
                if ast_elems.len() != ty_elems.len() {
                    return ty;
                }
                let masked: Vec<Ty> = ast_elems
                    .into_iter()
                    .zip(ty_elems)
                    .map(|(a, t)| self.mask_infer_tuple_ty(crate::body::TyId(a), t))
                    .collect();
                let substs = self.cx.types.intern_substs(&masked);
                self.cx.types.intern(Kind::Tuple(substs))
            }
            _ => ty,
        }
    }
}

pub(super) fn retain_non_loop_state(state: &mut State, loop_locals: &HashSet<u32>) {
    let keep = |h| is_temp_holder(Some(h)) || !loop_locals.contains(&h);
    state.loans.retain(|l| l.holder.is_none_or(keep));
    state
        .tuple_elem_sources
        .retain(|k, _| !loop_locals.contains(k));
}

pub(super) fn merge_break_states(mut merged: Option<State>, states: Vec<State>) -> Option<State> {
    for bs in states {
        match &mut merged {
            Some(acc) => merge_states(acc, &bs),
            None => merged = Some(bs),
        }
    }
    merged
}

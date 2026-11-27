use anyhow::Result;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::body::sig::Signature;
use crate::body::{Body, Expr, ExprId, PatId, Ty as AstTy, TyId};
use crate::outline::source::Source;
use crate::sema::infer::Res;
use crate::sema::ty::{Kind, Ty};

use super::super::calls::lifetimes::{LT_STATIC, lifetime_is_static};
use super::super::calls::return_lt::{
    ast_ref_dyn_requires_static, ast_ty_is_provably_static_in, lifetime_outlives_in_item,
};
use super::super::places::PlaceStep;
use super::super::propagate::BorrowSource;
use super::super::{Checker, State};
use super::is_temp_target;
use super::variance_ast::{
    ContraLifetimes, InvariantLifetimes, Variance, VarianceLifetimes, collect_ast_variance,
    collect_ast_variance_mode, collect_fn_item_variance,
};

pub(in super::super) struct VarianceMeta {
    pub(in super::super) decl_seq: Vec<u32>,
    decl_counter: u32,
    pub(in super::super) local_static_inv: Vec<bool>,
    pub(in super::super) local_static_ann: Vec<bool>,
    pub(in super::super) local_inv_lts: HashMap<u32, Vec<String>>,
    pub(in super::super) local_ann_ast_ty: HashMap<u32, TyId>,
    pub(in super::super) local_contra_lts: HashMap<u32, ContraLifetimes>,
    pub(in super::super) param_inv_lts: Vec<InvariantLifetimes>,
    pub(in super::super) param_inner_inv_lts: Vec<InvariantLifetimes>,
    pub(in super::super) param_contra_lts: Vec<ContraLifetimes>,
    pub(in super::super) param_inner_contra_lts: Vec<ContraLifetimes>,
    pub(in super::super) interior_mut_cache: HashMap<Ty, bool>,
}

impl VarianceMeta {
    pub(in super::super) fn new(n: usize) -> Self {
        Self {
            decl_seq: vec![0; n],
            decl_counter: 0,
            local_static_inv: vec![false; n],
            local_static_ann: vec![false; n],
            local_inv_lts: HashMap::default(),
            local_ann_ast_ty: HashMap::default(),
            local_contra_lts: HashMap::default(),
            param_inv_lts: vec![InvariantLifetimes::default(); n],
            param_inner_inv_lts: vec![InvariantLifetimes::default(); n],
            param_contra_lts: vec![ContraLifetimes::default(); n],
            param_inner_contra_lts: vec![ContraLifetimes::default(); n],
            interior_mut_cache: HashMap::default(),
        }
    }
}

impl Checker<'_, '_> {
    pub(in super::super) fn next_decl_seq(&mut self) -> u32 {
        self.variance.decl_counter += 1;
        self.variance.decl_counter
    }

    pub(in super::super) fn record_param_invariant(&mut self, sig: &Signature, src: &Source) {
        let mut cursor = 0usize;
        if let Some((kind, _)) = sig.receiver {
            if let Some(&local) = self.typed.params.get(cursor)
                && (local as usize) < self.variance.param_inv_lts.len()
                && let crate::body::sig::SelfKind::Typed(ty) = kind
            {
                self.record_param_inv_slot(local, &sig.body, src, ty);
            }
            cursor += 1;
        }
        for &(_, param_ty) in &sig.params {
            if let Some(&local) = self.typed.params.get(cursor)
                && (local as usize) < self.variance.param_inv_lts.len()
            {
                self.record_param_inv_slot(local, &sig.body, src, param_ty);
            }
            cursor += 1;
        }
    }

    fn record_param_inv_slot(&mut self, local: u32, body: &Body, src: &Source, param_ty: TyId) {
        let lty = self.local_ty(local);
        let var = collect_ast_variance(self.cx, Some(self.item), body, src, param_ty, Some(lty));
        self.variance.param_inv_lts[local as usize] = var.inv;
        self.variance.param_contra_lts[local as usize] = var.contra;
        if let AstTy::Ref {
            mutable, ty: inner, ..
        } = *body.ty(param_ty)
        {
            let inner_hint = (lty != Ty::NONE)
                .then(|| match *self.cx.types.kind(lty) {
                    Kind::Ref(_, p) => Some(p),
                    _ => None,
                })
                .flatten();
            let init_var = if mutable {
                Variance::Invariant
            } else {
                Variance::Covariant
            };
            let inner_var = collect_ast_variance_mode(
                self.cx,
                Some(self.item),
                body,
                src,
                inner,
                inner_hint,
                init_var,
            );
            self.variance.param_inner_inv_lts[local as usize] = inner_var.inv;
            self.variance.param_inner_contra_lts[local as usize] = inner_var.contra;
        }
    }

    pub(in super::super) fn param_inv_slot(
        &self,
        local: u32,
        peel: bool,
    ) -> Option<&InvariantLifetimes> {
        let slice = if peel {
            &self.variance.param_inner_inv_lts
        } else {
            &self.variance.param_inv_lts
        };
        slice.get(local as usize)
    }

    pub(in super::super) fn param_contra_slot(
        &self,
        local: u32,
        peel: bool,
    ) -> Option<&ContraLifetimes> {
        let slice = if peel {
            &self.variance.param_inner_contra_lts
        } else {
            &self.variance.param_contra_lts
        };
        slice.get(local as usize)
    }

    pub(in super::super) fn param_has_invariant_lt(&self, local: u32, peel: bool) -> bool {
        self.param_inv_slot(local, peel)
            .is_some_and(|inv| !inv.is_empty())
    }

    pub(in super::super) fn record_let_annotation(
        &mut self,
        pat: PatId,
        ann_ty: Option<TyId>,
        init: Option<ExprId>,
        state: &mut State,
    ) -> Result<()> {
        let mut bound = HashSet::default();
        self.collect_pat_locals(pat, &mut bound);
        if let Some(e) = init
            && bound.len() == 1
            && let Some(&local) = bound.iter().next()
            && self.is_mut_ptr_ty(self.local_ty(local))
        {
            self.record_ptr_targets(local, e, state);
        }
        let has_concrete_ann =
            ann_ty.is_some_and(|ty_id| !matches!(*self.body.ty(ty_id), AstTy::Infer));
        if !has_concrete_ann {
            if let Some(e) = init
                && bound.len() == 1
                && let Some(&local) = bound.iter().next()
            {
                let src_var = self.expr_variance_lts(e);
                if !src_var.contra.is_empty() {
                    self.variance.local_contra_lts.insert(local, src_var.contra);
                }
                if src_var.inv.is_only_static()
                    && let Some(slot) = self.variance.local_static_inv.get_mut(local as usize)
                {
                    *slot = true;
                } else if !src_var.inv.named.is_empty() {
                    self.variance.local_inv_lts.insert(local, src_var.inv.named);
                }
            }
            return Ok(());
        }
        let Some(ty_id) = ann_ty else {
            return Ok(());
        };
        if bound.len() != 1 {
            return Ok(());
        }
        let Some(&local) = bound.iter().next() else {
            return Ok(());
        };
        self.variance.local_ann_ast_ty.insert(local, ty_id);
        let Ok(sig) = self.cx.signature(self.item) else {
            return Ok(());
        };
        let lty = self.local_ty(local);
        let var = collect_ast_variance(
            self.cx,
            Some(self.item),
            self.body,
            self.src,
            ty_id,
            Some(lty),
        );
        let inv = var.inv;
        let contra = var.contra;
        let full_static =
            ast_ty_is_provably_static_in(self.body, self.src, Some((&sig, self.src)), ty_id);
        let ref_dyn_static =
            ast_ref_dyn_requires_static(self.body, self.src, Some((&sig, self.src)), ty_id);
        let mut inner_static_ref = false;
        let check_inv = if let AstTy::Ref {
            mutable,
            lifetime,
            ty: ann_inner,
        } = *self.body.ty(ty_id)
            && !lifetime.is_some_and(|lt| self.src.text(lt) == LT_STATIC)
        {
            let inner_hint = (lty != Ty::NONE)
                .then(|| match *self.cx.types.kind(lty) {
                    Kind::Ref(_, p) => Some(p),
                    _ => None,
                })
                .flatten();
            let init_var = if mutable {
                Variance::Invariant
            } else {
                Variance::Covariant
            };
            let inner_inv = collect_ast_variance_mode(
                self.cx,
                Some(self.item),
                self.body,
                self.src,
                ann_inner,
                inner_hint,
                init_var,
            )
            .inv;
            inner_static_ref = inner_inv.is_only_static();
            inner_inv
        } else {
            inv.clone()
        };
        if full_static && let Some(slot) = self.variance.local_static_ann.get_mut(local as usize) {
            *slot = true;
        }
        if inv.is_only_static() || inner_static_ref {
            if let Some(slot) = self.variance.local_static_inv.get_mut(local as usize) {
                *slot = true;
            }
        } else if !check_inv.named.is_empty() {
            self.variance
                .local_inv_lts
                .insert(local, check_inv.named.clone());
        }
        if !contra.is_empty() {
            self.variance.local_contra_lts.insert(local, contra.clone());
        }
        if let Some(e) = init {
            if ref_dyn_static {
                let sources = self.expr_borrow_sources_let(e, state);
                self.check_ref_dyn_static_pointee(e, &sources)?;
            }
            self.check_let_init_variance(ty_id, e, full_static, &check_inv, &contra, state)?;
        }
        Ok(())
    }

    fn check_let_init_variance(
        &mut self,
        ann_ty: TyId,
        init: ExprId,
        full_static: bool,
        inv: &InvariantLifetimes,
        contra: &ContraLifetimes,
        state: &State,
    ) -> Result<()> {
        if !contra.is_empty() {
            self.check_expr_contra_fit(init, contra)?;
        }
        let unwrapped = self.unwrap_parens(init);
        if let (
            AstTy::Ref { lifetime, .. },
            Expr::Ref {
                raw: false,
                expr: inner,
                ..
            },
        ) = (*self.body.ty(ann_ty), *self.body.expr(unwrapped))
            && !lifetime.is_some_and(|lt| self.src.text(lt) == LT_STATIC)
        {
            if let Some(p) = self.root_place(inner) {
                let direct = p.steps.is_empty() && !p.through_ref;
                let deref_reborrow =
                    p.through_ref && p.steps.len() == 1 && matches!(p.steps[0], PlaceStep::Deref);
                if direct || deref_reborrow {
                    if direct
                        && inv.is_only_static()
                        && let Some(slot) = self.variance.local_static_inv.get_mut(p.local as usize)
                    {
                        *slot = true;
                    }
                    let inner_loans: Vec<BorrowSource> = state
                        .loans
                        .iter()
                        .filter(|l| l.holder == Some(p.local))
                        .map(BorrowSource::from_loan)
                        .collect();
                    self.check_sources_fit_invariant_lts(
                        Some(init),
                        &inner_loans,
                        inv.is_only_static(),
                        &inv.named,
                    )?;
                    self.check_local_place_invariant_fit(
                        Some(init),
                        p.local,
                        deref_reborrow,
                        inv.is_only_static(),
                        &inv.named,
                    )?;
                }
            }
            return Ok(());
        }
        let is_ptr = matches!(*self.body.ty(ann_ty), AstTy::Ptr { .. });
        if is_ptr && inv.is_empty() {
            return Ok(());
        }
        let req_static = full_static || inv.is_only_static();
        if !req_static && inv.named.is_empty() {
            return Ok(());
        }
        let source_expr = if is_ptr {
            match *self.body.expr(self.unwrap_parens(init)) {
                Expr::Ref { expr: inner, .. } => inner,
                _ => init,
            }
        } else {
            init
        };
        let source_unwrapped = self.unwrap_parens(source_expr);
        let sources = self.expr_borrow_sources_let(source_expr, state);
        self.check_sources_fit_invariant_lts(Some(init), &sources, req_static, &inv.named)?;
        if let Some(src_local) = self.expr_plain_local(source_unwrapped) {
            let peel = self.is_ref_ty(self.local_ty(src_local));
            self.check_local_place_invariant_fit(
                Some(init),
                src_local,
                peel,
                req_static,
                &inv.named,
            )?;
        } else {
            let src_var = self.expr_variance_lts(source_expr);
            if !src_var.inv.is_empty() {
                self.check_inv_against_req(Some(init), 0, &src_var.inv, req_static, &inv.named)?;
            }
        }
        Ok(())
    }

    pub(in super::super) fn target_contra_req(
        &self,
        target_local: u32,
        peel_target_ref: bool,
    ) -> ContraLifetimes {
        if !peel_target_ref && let Some(c) = self.variance.local_contra_lts.get(&target_local) {
            return c.clone();
        }
        if self.is_param_local(target_local)
            && let Some(c) = self.param_contra_slot(target_local, peel_target_ref)
        {
            return c.clone();
        }
        ContraLifetimes::default()
    }

    pub(in super::super) fn expr_variance_lts(&mut self, expr: ExprId) -> VarianceLifetimes {
        let unwrapped = self.unwrap_parens_and_empty_blocks(expr);
        match *self.body.expr(unwrapped) {
            Expr::Cast(inner, _) => self.expr_variance_lts(inner),
            Expr::Path(_) => {
                if let Some(local) = self.expr_plain_local(unwrapped) {
                    let mut out = VarianceLifetimes::default();
                    if self.is_param_local(local) {
                        if let Some(inv) = self.param_inv_slot(local, false) {
                            out.inv = inv.clone();
                        }
                        if let Some(contra) = self.param_contra_slot(local, false) {
                            out.contra = contra.clone();
                        }
                    } else {
                        out.inv.saw_static = self
                            .variance
                            .local_static_inv
                            .get(local as usize)
                            .copied()
                            .unwrap_or(false);
                        if let Some(lts) = self.variance.local_inv_lts.get(&local) {
                            out.inv.named = lts.clone();
                        }
                        if let Some(contra) = self.variance.local_contra_lts.get(&local) {
                            out.contra = contra.clone();
                        }
                    }
                    return out;
                }
                if let Some(Res::Fn(fn_item, _)) = self.typed.res.get(&unwrapped).copied() {
                    return collect_fn_item_variance(self.cx, fn_item);
                }
                VarianceLifetimes::default()
            }
            _ => VarianceLifetimes::default(),
        }
    }

    pub(in super::super) fn check_contra_against_req(
        &mut self,
        site: Option<ExprId>,
        err_local: Option<u32>,
        src: &ContraLifetimes,
        dst: &ContraLifetimes,
    ) -> Result<()> {
        if src.is_empty() || src.is_only_hrtb() || dst.is_empty() {
            return Ok(());
        }
        let Ok(sig) = self.cx.signature(self.item) else {
            return Ok(());
        };
        let reject = |this: &Self| {
            match err_local {
            Some(local) => this.does_not_live_long_enough(site, local),
            None => this.borrow_err(
                site,
                "lifetime may not live long enough: function pointer parameter lifetime is contravariant",
            ),
        }
        };
        if src.saw_static {
            let ok = if src.is_only_static() {
                !dst.has_hrtb
                    && dst
                        .named
                        .iter()
                        .all(|lt| lifetime_is_static(&sig, self.src, lt))
            } else {
                dst.saw_static
                    || dst
                        .named
                        .iter()
                        .any(|lt| lifetime_is_static(&sig, self.src, lt))
            };
            if !ok {
                return Err(reject(self));
            }
        }
        for src_lt in &src.named {
            let ok = dst.saw_static
                || dst.named.iter().any(|dst_lt| {
                    lifetime_outlives_in_item(self.cx, self.item, &sig, self.src, dst_lt, src_lt)
                });
            if !ok {
                return Err(reject(self));
            }
        }
        Ok(())
    }

    pub(in super::super) fn check_expr_contra_fit(
        &mut self,
        expr: ExprId,
        dst: &ContraLifetimes,
    ) -> Result<()> {
        if dst.is_empty() {
            return Ok(());
        }
        let src_var = self.expr_variance_lts(expr);
        let err_local = self.expr_plain_local(self.unwrap_parens_and_empty_blocks(expr));
        self.check_contra_against_req(Some(expr), err_local, &src_var.contra, dst)
    }

    pub(in super::super) fn check_assign_variance(
        &mut self,
        lhs: ExprId,
        rhs: ExprId,
        state: &State,
    ) -> Result<()> {
        let Some(p) = self.root_place(lhs) else {
            return Ok(());
        };
        let has_field = p
            .steps
            .iter()
            .any(|st| matches!(st, PlaceStep::Field(..) | PlaceStep::SliceElem { .. }));
        if has_field {
            return Ok(());
        }
        let dst_contra = self.target_contra_req(p.local, p.through_ref);
        if !dst_contra.is_empty() {
            self.check_expr_contra_fit(rhs, &dst_contra)?;
        }
        if p.through_ref {
            let targets: Vec<u32> = state
                .loans
                .iter()
                .filter(|l| l.holder == Some(p.local) && l.mutable)
                .map(|l| l.target)
                .collect();
            for t in targets {
                let t_contra = self.target_contra_req(t, false);
                if !t_contra.is_empty() {
                    self.check_expr_contra_fit(rhs, &t_contra)?;
                }
            }
        } else if p.steps.is_empty() && dst_contra.is_empty() && !self.is_param_local(p.local) {
            let src_var = self.expr_variance_lts(rhs);
            if !src_var.contra.is_empty() {
                self.variance
                    .local_contra_lts
                    .insert(p.local, src_var.contra);
            }
        }
        Ok(())
    }

    pub(in super::super) fn check_inv_against_req(
        &mut self,
        site: Option<ExprId>,
        err_local: u32,
        inv: &InvariantLifetimes,
        req_static: bool,
        req_named: &[String],
    ) -> Result<()> {
        if inv.is_empty() {
            return Ok(());
        }
        let Ok(sig) = self.cx.signature(self.item) else {
            return Ok(());
        };
        if req_static {
            let ok = inv.is_only_static()
                || (!inv.has_elided
                    && inv
                        .named
                        .iter()
                        .all(|lt| lifetime_is_static(&sig, self.src, lt)));
            if !ok {
                return Err(self.does_not_live_long_enough(site, err_local));
            }
        }
        for req in req_named {
            if inv.is_only_static() && !lifetime_is_static(&sig, self.src, req) {
                return Err(self.does_not_live_long_enough(site, err_local));
            }
            if inv.has_elided
                || !inv.named.iter().all(|p_lt| {
                    lifetime_outlives_in_item(self.cx, self.item, &sig, self.src, p_lt, req)
                        && lifetime_outlives_in_item(self.cx, self.item, &sig, self.src, req, p_lt)
                })
            {
                return Err(self.does_not_live_long_enough(site, err_local));
            }
        }
        Ok(())
    }

    pub(in super::super) fn check_local_place_invariant_fit(
        &mut self,
        site: Option<ExprId>,
        src_local: u32,
        peel_src: bool,
        req_static: bool,
        req_named: &[String],
    ) -> Result<()> {
        if !req_static && req_named.is_empty() {
            return Ok(());
        }
        if self.is_param_local(src_local) {
            if let Some(inv) = self.param_inv_slot(src_local, peel_src).cloned() {
                return self.check_inv_against_req(site, src_local, &inv, req_static, req_named);
            }
            return Ok(());
        }
        let mut local_inv = InvariantLifetimes::default();
        local_inv.saw_static = self
            .variance
            .local_static_inv
            .get(src_local as usize)
            .copied()
            .unwrap_or(false);
        if let Some(lts) = self.variance.local_inv_lts.get(&src_local) {
            local_inv.named = lts.clone();
        }
        self.check_inv_against_req(site, src_local, &local_inv, req_static, req_named)
    }

    pub(in super::super) fn target_invariant_req(
        &self,
        target_local: u32,
        peel_target_ref: bool,
        has_field_step: bool,
    ) -> (bool, Vec<String>) {
        let local_static = self
            .variance
            .local_static_inv
            .get(target_local as usize)
            .copied()
            .unwrap_or(false)
            || self
                .variance
                .local_static_ann
                .get(target_local as usize)
                .copied()
                .unwrap_or(false);
        if !has_field_step && let Some(named) = self.variance.local_inv_lts.get(&target_local) {
            return (local_static, named.clone());
        }
        if self.is_param_local(target_local) {
            if let Some(inv) = self.param_inv_slot(target_local, peel_target_ref) {
                let param_static = inv.is_only_static()
                    || (peel_target_ref && self.param_inner_is_static(target_local))
                    || (!peel_target_ref && !self.is_non_static_param(target_local));
                let named = if has_field_step {
                    Vec::new()
                } else {
                    inv.named.clone()
                };
                return (local_static || param_static, named);
            }
        }
        (local_static, Vec::new())
    }

    pub(in super::super) fn check_sources_fit_target(
        &mut self,
        site: Option<ExprId>,
        target_local: u32,
        peel_target_ref: bool,
        has_field_step: bool,
        sources: &[BorrowSource],
    ) -> Result<()> {
        if sources.is_empty() {
            return Ok(());
        }
        let (req_static, req_named) =
            self.target_invariant_req(target_local, peel_target_ref, has_field_step);
        let target_ty = self.local_ty(target_local);
        self.check_sources_fit_invariant_lts_with_target(
            site,
            sources,
            req_static,
            &req_named,
            Some(super::super::calls::bounds::ShareTarget::Local(
                target_local,
                target_ty,
            )),
        )
    }

    pub(in super::super) fn check_sources_fit_invariant_lts(
        &mut self,
        site: Option<ExprId>,
        sources: &[BorrowSource],
        req_static: bool,
        req_named: &[String],
    ) -> Result<()> {
        self.check_sources_fit_invariant_lts_with_target(site, sources, req_static, req_named, None)
    }

    fn check_sources_fit_invariant_lts_with_target(
        &mut self,
        site: Option<ExprId>,
        sources: &[BorrowSource],
        req_static: bool,
        req_named: &[String],
        dest: Option<super::super::calls::bounds::ShareTarget>,
    ) -> Result<()> {
        if (!req_static && req_named.is_empty()) || sources.is_empty() {
            return Ok(());
        }
        let in_const = self.is_const_or_static_item();
        let req_refs: Vec<&str> = req_named.iter().map(String::as_str).collect();
        for s in sources {
            if is_temp_target(s.target).is_some() {
                if in_const && self.temp_target_allowed(s.mutable) {
                    continue;
                }
                return Err(self.does_not_live_long_enough(site, s.target));
            }
            let target_ty = self.local_ty(s.target);
            let target_kind = self.cx.types.kind(target_ty).clone();
            let static_reborrow = s.through_ref
                && !self
                    .is_param
                    .get(s.target as usize)
                    .copied()
                    .unwrap_or(false)
                && matches!(target_kind, Kind::Ref(crate::sema::ty::Mut::Not, _));
            if !self.from_caller(s.target, s.through_ref) && !static_reborrow {
                return Err(self.does_not_live_long_enough(site, s.target));
            }
            let peel = (s.origin_only || s.steps.iter().any(|st| matches!(st, PlaceStep::Deref)))
                && matches!(
                    target_kind,
                    Kind::Ref(_, pointee)
                        if self.ty_holds_non_dyn_borrow(pointee)
                            || (self.is_param_local(s.target)
                                && !self
                                    .param_lts
                                    .inner_named_lts
                                    .get(s.target as usize)
                                    .is_none_or(Vec::is_empty))
                );
            if req_static
                && self.is_non_static_param(s.target)
                && !(peel && self.param_inner_is_static(s.target))
            {
                return Err(self.does_not_live_long_enough(site, s.target));
            }
            if !req_refs.is_empty() {
                let target_pointee = match *self.cx.types.kind(target_ty) {
                    Kind::Ref(_, p) => p,
                    _ => target_ty,
                };
                let invariant_src = (s.steps.is_empty()
                    && self.param_has_invariant_lt(s.target, false))
                    || s.mutable
                    || self.ty_has_borrow_interior_mut(if s.steps.is_empty() {
                        target_ty
                    } else {
                        target_pointee
                    });
                if invariant_src && self.param_has_invariant_lt(s.target, peel) {
                    self.check_local_place_invariant_fit(
                        site, s.target, peel, req_static, req_named,
                    )?;
                } else if !self.param_satisfies_lt_mode(s.target, &req_refs, peel, false, dest) {
                    return Err(self.does_not_live_long_enough(site, s.target));
                }
            }
        }
        Ok(())
    }
}

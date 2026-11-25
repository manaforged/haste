use crate::body::{Expr, ExprId, Ty as AstTy, TyId};
use crate::outline::source::Source;
use crate::sema::ty::Kind;

use super::lifetimes::{Lifetimes, collect_ty_lifetimes, lifetime_outlives_in_headers, owned_lts};

fn lifetime_outlives_in_item(
    cx: &mut crate::sema::Context<'_>,
    item: crate::names::ItemRef,
    sig: &crate::body::sig::Signature,
    src: &Source,
    a: &str,
    b: &str,
) -> bool {
    let impl_hdr = cx
        .owner_impl(item)
        .and_then(|block| cx.impl_header(block).ok().map(|hdr| (block, hdr)));
    match impl_hdr {
        Some((block, ref hdr)) => {
            let impl_src = &cx.impl_file(block).source;
            lifetime_outlives_in_headers(
                &[
                    (&sig.body, src, &sig.generics, &sig.predicates),
                    (&hdr.body, impl_src, &hdr.generics, &hdr.predicates),
                ],
                a,
                b,
            )
        }
        None => {
            lifetime_outlives_in_headers(&[(&sig.body, src, &sig.generics, &sig.predicates)], a, b)
        }
    }
}

impl super::super::Checker<'_, '_> {
    pub(in super::super) fn param_satisfies_caller_lt(
        &mut self,
        local: u32,
        req_lts: &[&str],
        peel_outer_ref: bool,
    ) -> bool {
        self.param_satisfies_lt_mode(local, req_lts, peel_outer_ref, true)
    }

    fn param_satisfies_lt_mode(
        &mut self,
        local: u32,
        req_lts: &[&str],
        peel_outer_ref: bool,
        check_invariant: bool,
    ) -> bool {
        if !self.is_non_static_param(local) {
            return true;
        }
        let slot = local as usize;
        if peel_outer_ref && self.param_inner_is_static(local) {
            return true;
        }
        let (has_elided, declared) = self.param_lts.slot(slot, peel_outer_ref);
        if has_elided {
            return false;
        }
        let Some(declared) = declared.map(<[String]>::to_vec) else {
            return false;
        };
        if declared.is_empty() {
            return false;
        }
        let Ok(sig) = self.cx.signature(self.item) else {
            return false;
        };
        let local_ty = self.local_ty(local);
        let invariant = check_invariant && peel_outer_ref && self.is_mut_ref_ty(local_ty);
        let allow_any = !check_invariant && (peel_outer_ref || !self.is_ref_ty(local_ty));
        let mut satisfies = |p_lt: &str| {
            req_lts.iter().all(|&req| {
                lifetime_outlives_in_item(self.cx, self.item, &sig, self.src, p_lt, req)
                    && (!invariant
                        || lifetime_outlives_in_item(self.cx, self.item, &sig, self.src, req, p_lt))
            })
        };
        if allow_any {
            declared.iter().any(|p_lt| satisfies(p_lt))
        } else {
            declared.iter().all(|p_lt| satisfies(p_lt))
        }
    }

    fn check_return_named_lt_against(
        &mut self,
        expr: ExprId,
        src: &super::super::propagate::BorrowSource,
        req: &[&str],
        ret_mut_inner: Option<&[&str]>,
    ) -> anyhow::Result<()> {
        if self.closure_depth != 0
            || req.is_empty()
            || src.env_only
            || !self.is_non_static_param(src.target)
        {
            return Ok(());
        }
        let expr_ty = self.expr_ty(expr);
        let mut_ret_from_mut = ret_mut_inner.is_some()
            && self.is_mut_ref_ty(self.local_ty(src.target))
            && self.is_mut_ref_ty(expr_ty);
        if mut_ret_from_mut
            && let Some(inner_req) = ret_mut_inner
            && !inner_req.is_empty()
            && !self.param_satisfies_lt_mode(src.target, inner_req, true, true)
        {
            return Err(self.borrow_err(
                Some(expr),
                format!(
                    "lifetime may not live long enough: mutable reference `{}` is invariant over its inner lifetime",
                    self.local_name(src.target)
                ),
            ));
        }
        if req
            .iter()
            .any(|&r| self.param_satisfies_lt_mode(src.target, &[r], false, false))
        {
            return Ok(());
        }
        let inner_ty = match *self.cx.types.kind(self.local_ty(src.target)) {
            Kind::Ref(_, inner) => Some(inner),
            _ => None,
        };
        let borrows_owned_field = matches!(
            *self.body.expr(expr),
            Expr::Ref { expr: inner, .. }
                if self.ty_is_borrow_free(self.expr_ty(inner))
                    && match *self.body.expr(inner) {
                        Expr::Field(base, _) | Expr::TupleField(base, _) => {
                            self.expr_plain_local(base) == Some(src.target)
                        }
                        _ => false,
                    }
        );
        let can_peel = !mut_ret_from_mut
            && !borrows_owned_field
            && self.expr_plain_local(expr) != Some(src.target)
            && inner_ty.is_some_and(|inner| {
                self.ty_holds_borrow(inner)
                    && (!self.is_ref_ty(inner)
                        || self.ty_max_ref_depth(expr_ty) <= self.ty_max_ref_depth(inner))
            });
        if can_peel
            && req
                .iter()
                .any(|&r| self.param_satisfies_lt_mode(src.target, &[r], true, false))
        {
            return Ok(());
        }
        Err(self.borrow_err(
            Some(expr),
            format!(
                "lifetime may not live long enough: `{}` does not satisfy return lifetime",
                self.local_name(src.target)
            ),
        ))
    }

    pub(in super::super) fn check_return_named_lt(
        &mut self,
        expr: ExprId,
        src: &super::super::propagate::BorrowSource,
    ) -> anyhow::Result<()> {
        if self.closure_depth != 0 || self.ret_named_lts.is_empty() {
            return Ok(());
        }
        let req = self.ret_named_lts.clone();
        let ret_mut_inner = self.ret_mut_inner_named_lts();
        let inner_slice: Option<Vec<&str>> = ret_mut_inner
            .as_ref()
            .map(|v| v.iter().map(String::as_str).collect());
        self.check_return_named_lt_against(expr, src, &req, inner_slice.as_deref())
    }

    fn ret_mut_inner_named_lts(&mut self) -> Option<Vec<String>> {
        let sig = self.cx.signature(self.item).ok()?;
        let ret_ast = sig.ret?;
        let AstTy::Ref {
            mutable: true,
            ty: inner,
            ..
        } = *sig.body.ty(ret_ast)
        else {
            return None;
        };
        let mut lts = Lifetimes::default();
        collect_ty_lifetimes(&sig.body, self.src, inner, &mut lts);
        Some(owned_lts(lts.named))
    }

    fn check_tuple_elem_lts(
        &mut self,
        sig: &crate::body::sig::Signature,
        site: ExprId,
        ty: TyId,
        sources: &[super::super::propagate::BorrowSource],
    ) -> anyhow::Result<()> {
        let mut elem_lts = Lifetimes::default();
        collect_ty_lifetimes(&sig.body, self.src, ty, &mut elem_lts);
        if elem_lts.has_elided || elem_lts.named.is_empty() {
            return Ok(());
        }
        for src in sources {
            self.check_return_named_lt_against(site, src, &elem_lts.named, None)?;
        }
        Ok(())
    }

    fn check_tuple_sources_lts(
        &mut self,
        sig: &crate::body::sig::Signature,
        site: ExprId,
        ty: TyId,
        ts: &super::super::propagate::TupleSources,
    ) -> anyhow::Result<()> {
        match (sig.body.ty(ty), ts) {
            (AstTy::Tuple(ret_elems), super::super::propagate::TupleSources::Tuple(elems)) => {
                let ret_ids: Vec<TyId> = sig
                    .body
                    .list(*ret_elems)
                    .iter()
                    .copied()
                    .map(TyId)
                    .collect();
                if elems.len() == ret_ids.len() {
                    for (elem_ts, &t_id) in elems.iter().zip(&ret_ids) {
                        self.check_tuple_sources_lts(sig, site, t_id, elem_ts)?;
                    }
                }
                Ok(())
            }
            (_, super::super::propagate::TupleSources::Leaf { sources, .. }) => {
                self.check_tuple_elem_lts(sig, site, ty, sources)
            }
            _ => Ok(()),
        }
    }

    pub(in super::super) fn check_return_tuple_named_lt(
        &mut self,
        expr: ExprId,
        state: &super::super::State,
    ) -> anyhow::Result<()> {
        if self.closure_depth != 0 || self.ret_named_lts.is_empty() {
            return Ok(());
        }
        let Ok(sig) = self.cx.signature(self.item) else {
            return Ok(());
        };
        let Some(ret_ast) = sig.ret else {
            return Ok(());
        };
        let AstTy::Tuple(ret_elems) = *sig.body.ty(ret_ast) else {
            return Ok(());
        };
        let ret_ids: Vec<TyId> = sig.body.list(ret_elems).iter().copied().map(TyId).collect();
        if let Expr::Tuple(expr_elems) = *self.body.expr(expr) {
            let expr_ids = self.body.list(expr_elems).to_vec();
            if expr_ids.len() == ret_ids.len() {
                for (&e_id, &t_id) in expr_ids.iter().zip(&ret_ids) {
                    let elem_expr = ExprId(e_id);
                    let sources = self.expr_borrow_sources(elem_expr, state);
                    self.check_tuple_elem_lts(&sig, elem_expr, t_id, &sources)?;
                }
            }
            return Ok(());
        }
        if let Some(local) = self.expr_plain_local(expr)
            && let Some(ts) = state.tuple_elem_sources.get(&local).cloned()
        {
            self.check_tuple_sources_lts(&sig, expr, ret_ast, &ts)?;
        }
        Ok(())
    }
}

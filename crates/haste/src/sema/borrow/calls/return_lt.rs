use anyhow::Result;

use crate::body::{Args, BlockKind, Body, Bound, Expr, ExprId, GenericArg, Ty as AstTy, TyId};
use crate::outline::source::Source;
use crate::sema::infer::Res;
use crate::sema::ty::Kind;

use super::super::checks::{InvariantLifetimes, collect_ast_invariant_lifetimes, is_temp_target};
use super::super::places::PlaceStep;
use super::super::propagate::BorrowSource;
use super::lifetimes::{
    LT_STATIC, Lifetimes, collect_ty_lifetimes, is_owned_container, is_prim_name,
    lifetime_is_static, lifetime_outlives_in_headers, owned_lts,
};

pub(in super::super) fn lifetime_outlives_in_item(
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

#[derive(Clone, Copy, PartialEq, Eq)]
enum DynDefault {
    Inferred,
    InSig,
    InSigOwned,
}

impl DynDefault {
    fn enter_owned(self) -> Self {
        match self {
            Self::Inferred => Self::Inferred,
            Self::InSig | Self::InSigOwned => Self::InSigOwned,
        }
    }

    fn reset_owned(self) -> Self {
        match self {
            Self::Inferred => Self::Inferred,
            Self::InSig | Self::InSigOwned => Self::InSig,
        }
    }
}

pub(super) fn ast_ty_is_provably_static(
    sig: &crate::body::sig::Signature,
    src: &Source,
    ty: TyId,
) -> bool {
    ast_ty_is_provably_static_in_mode(&sig.body, src, Some((sig, src)), ty, DynDefault::InSig)
}

pub(in super::super) fn ast_ty_is_provably_static_in(
    body: &Body,
    src: &Source,
    sig: Option<(&crate::body::sig::Signature, &Source)>,
    ty: TyId,
) -> bool {
    ast_ty_is_provably_static_in_mode(body, src, sig, ty, DynDefault::Inferred)
}

fn path_args_are_static(
    body: &Body,
    src: &Source,
    sig: Option<(&crate::body::sig::Signature, &Source)>,
    path: crate::body::PathId,
    lt_static: &impl Fn(&str) -> bool,
    mode: DynDefault,
) -> bool {
    let base = mode.reset_owned();
    body.list(body.path(path).segments).iter().all(|&s_id| {
        match body.segments[s_id as usize].args {
            None => true,
            Some(Args::Angle(args)) => {
                body.list(args)
                    .iter()
                    .all(|&a| match body.args[a as usize] {
                        GenericArg::Ty(t) | GenericArg::Binding { ty: t, .. } => {
                            ast_ty_is_provably_static_in_mode(body, src, sig, t, base)
                        }
                        GenericArg::Lifetime(lt) => lt_static(src.text(lt)),
                        GenericArg::Const(_) => true,
                        GenericArg::Constraint { .. } => false,
                    })
            }
            Some(Args::Paren { inputs, output }) => {
                body.list(inputs)
                    .iter()
                    .all(|&t| fn_param_is_provably_static(body, src, sig, TyId(t), lt_static, base))
                    && output
                        .is_none_or(|o| ast_ty_is_provably_static_in_mode(body, src, sig, o, base))
            }
            Some(Args::ReturnType) => false,
        }
    })
}

fn fn_param_is_provably_static(
    body: &Body,
    src: &Source,
    sig: Option<(&crate::body::sig::Signature, &Source)>,
    ty: TyId,
    lt_static: &impl Fn(&str) -> bool,
    mode: DynDefault,
) -> bool {
    match *body.ty(ty) {
        AstTy::Ref {
            lifetime: None,
            ty: inner,
            ..
        } => ast_ty_is_provably_static_in_mode(body, src, sig, inner, mode.reset_owned()),
        AstTy::Ref {
            lifetime: Some(lt),
            ty: inner,
            ..
        } => {
            lt_static(src.text(lt))
                && ast_ty_is_provably_static_in_mode(body, src, sig, inner, mode.enter_owned())
        }
        _ => ast_ty_is_provably_static_in_mode(body, src, sig, ty, mode.reset_owned()),
    }
}

fn ast_ty_is_provably_static_in_mode(
    body: &Body,
    src: &Source,
    sig: Option<(&crate::body::sig::Signature, &Source)>,
    ty: TyId,
    mode: DynDefault,
) -> bool {
    let lt_static = |lt: &str| match sig {
        Some((s, s_src)) => lifetime_is_static(s, s_src, lt),
        None => lt == LT_STATIC,
    };
    match *body.ty(ty) {
        AstTy::Never => true,
        AstTy::Ptr { ty: t, .. } => {
            ast_ty_is_provably_static_in_mode(body, src, sig, t, mode.enter_owned())
        }
        AstTy::Paren(t) | AstTy::Slice(t) | AstTy::Array(t, _) | AstTy::Pat(t, _) => {
            ast_ty_is_provably_static_in_mode(body, src, sig, t, mode)
        }
        AstTy::Tuple(list) => body
            .list(list)
            .iter()
            .all(|&t| ast_ty_is_provably_static_in_mode(body, src, sig, TyId(t), mode)),
        AstTy::Ref {
            lifetime: Some(lt),
            ty: inner,
            ..
        } => {
            lt_static(src.text(lt))
                && ast_ty_is_provably_static_in_mode(body, src, sig, inner, mode.enter_owned())
        }
        AstTy::ImplTrait(_, list) | AstTy::DynTrait(list) => {
            let is_dyn = matches!(*body.ty(ty), AstTy::DynTrait(_));
            let bounds = body.list(list);
            let has_lt = bounds
                .iter()
                .any(|&b_id| matches!(body.bounds[b_id as usize], Bound::Lifetime(_)));
            let lifetime_ok = if has_lt {
                bounds.iter().any(|&b_id| {
                    matches!(body.bounds[b_id as usize], Bound::Lifetime(lt) if lt_static(src.text(lt)))
                })
            } else {
                is_dyn && mode == DynDefault::InSigOwned
            };
            lifetime_ok
                && bounds.iter().all(|&b_id| match body.bounds[b_id as usize] {
                    Bound::Lifetime(lt) => lt_static(src.text(lt)),
                    Bound::Trait { path, .. } => {
                        path_args_are_static(body, src, sig, path, &lt_static, mode)
                    }
                    Bound::Use(_) => true,
                })
        }
        AstTy::Path(path_id) => {
            let p = body.path(path_id);
            let segs = body.list(p.segments);
            if p.qself.is_some() || segs.is_empty() {
                return false;
            }
            if segs[..segs.len() - 1]
                .iter()
                .any(|&s| body.segments[s as usize].args.is_some())
            {
                return false;
            }
            let seg = body.segments[segs[segs.len() - 1] as usize];
            let name = crate::names::plain(src.text(seg.name));
            if seg.args.is_none() && is_prim_name(name) {
                return true;
            }
            if is_owned_container(name)
                && let Some(Args::Angle(args)) = seg.args
            {
                return body
                    .list(args)
                    .iter()
                    .all(|&a| match body.args[a as usize] {
                        GenericArg::Ty(t) | GenericArg::Binding { ty: t, .. } => {
                            ast_ty_is_provably_static_in_mode(body, src, sig, t, mode.enter_owned())
                        }
                        GenericArg::Lifetime(lt) => lt_static(src.text(lt)),
                        GenericArg::Const(_) => true,
                        GenericArg::Constraint { .. } => false,
                    });
            }
            false
        }
        _ => false,
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

    pub(in super::super) fn param_satisfies_lt_mode(
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
    ) -> Result<()> {
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
    ) -> Result<()> {
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

    fn ret_invariant_lts(&mut self) -> Option<InvariantLifetimes> {
        let sig = self.cx.signature(self.item).ok()?;
        let ret_ast = sig.ret?;
        Some(collect_ast_invariant_lifetimes(
            self.cx,
            &sig.body,
            self.src,
            ret_ast,
            self.typed.ret,
        ))
    }

    fn check_tuple_elem_lts(
        &mut self,
        sig: &crate::body::sig::Signature,
        site: ExprId,
        ty: TyId,
        sources: &[BorrowSource],
    ) -> Result<()> {
        let mut elem_lts = Lifetimes::default();
        collect_ty_lifetimes(&sig.body, self.src, ty, &mut elem_lts);
        if elem_lts.named.is_empty() {
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
    ) -> Result<()> {
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
    ) -> Result<()> {
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

    pub(in super::super) fn check_return_escape(
        &mut self,
        expr: ExprId,
        state: &super::super::State,
    ) -> Result<()> {
        let had_cached = if let Some(cached) = self.temp_sources.get(&expr).cloned() {
            let site = self.return_diag_site(expr);
            self.check_return_sources(site, &cached)?;
            true
        } else {
            false
        };
        match *self.body.expr(expr) {
            Expr::Block {
                kind: BlockKind::Const,
                ..
            } => Ok(()),
            Expr::Loop { .. } | Expr::Block { tail: None, .. } if had_cached => Ok(()),
            Expr::Paren(inner)
            | Expr::Block {
                tail: Some(inner), ..
            } => self.check_return_escape(inner, state),
            Expr::If(_, then, otherwise) => {
                self.check_return_escape(then, state)?;
                if let Some(alt) = otherwise {
                    self.check_return_escape(alt, state)?;
                }
                Ok(())
            }
            Expr::Match(_, arms) => {
                for id in self.body.list(arms).to_vec() {
                    self.check_return_escape(self.body.arms[id as usize].body, state)?;
                }
                Ok(())
            }
            _ => {
                let sources = self.expr_borrow_sources(expr, state);
                self.check_return_sources(expr, &sources)?;
                self.check_return_tuple_named_lt(expr, state)
            }
        }
    }

    fn return_diag_site(&self, mut expr: ExprId) -> ExprId {
        loop {
            match *self.body.expr(expr) {
                Expr::Paren(inner)
                | Expr::Block {
                    tail: Some(inner), ..
                } => expr = inner,
                _ => return expr,
            }
        }
    }

    fn ret_ast_requires_static(&mut self) -> bool {
        let Ok(sig) = self.cx.signature(self.item) else {
            return false;
        };
        !sig.is_async
            && sig
                .ret
                .is_some_and(|ret_ast| ast_ty_is_provably_static(&sig, self.src, ret_ast))
    }

    pub(in super::super) fn check_return_sources(
        &mut self,
        expr: ExprId,
        sources: &[BorrowSource],
    ) -> Result<()> {
        let expr_depth = self.ty_max_ref_depth(self.expr_ty(expr));
        for src in sources {
            let target_ty = self.local_ty(src.target);
            let inner = if target_ty != crate::sema::ty::Ty::NONE {
                match *self.cx.types.kind(target_ty) {
                    Kind::Ref(_, inner) => Some(inner),
                    _ => None,
                }
            } else {
                None
            };
            let past_outer = src.origin_only
                && inner.is_some_and(|i| {
                    self.ty_holds_non_dyn_borrow(i) && expr_depth <= self.ty_max_ref_depth(i)
                });
            if let Some(active) = self.active_closures.last_mut() {
                if !past_outer
                    && let Some(idx) = active.params.iter().position(|&p| p == src.target)
                    && !active.ret_params.contains(&idx)
                {
                    active.ret_params.push(idx);
                }
                if src.origin_only {
                    active.ret_orig.insert(src.target);
                } else {
                    active.ret_real.insert(src.target);
                }
            }
        }
        let returns_only_static = self.returns_only_static || self.ret_ast_requires_static();
        for src in sources {
            if self.is_escaping_return(src) {
                if is_temp_target(src.target).is_some() {
                    return Err(self.does_not_live_long_enough(Some(expr), src.target));
                }
                return Err(self.borrow_err(
                    Some(expr),
                    format!(
                        "cannot return value referencing local variable `{}`",
                        self.local_name(src.target)
                    ),
                ));
            }
            if self.closure_depth == 0
                && returns_only_static
                && self.is_non_static_param(src.target)
            {
                return Err(self.borrow_err(
                    Some(expr),
                    format!(
                        "borrowed data `{}` escapes outside of function: return type requires `'static` lifetime",
                        self.local_name(src.target)
                    ),
                ));
            }
            self.check_return_named_lt(expr, src)?;
        }
        if self.closure_depth == 0 && returns_only_static {
            self.check_return_static_local_args(expr)?;
        }
        if self.closure_depth == 0 && self.returns_invariant {
            self.check_return_invariant_expr(expr, sources)?;
        }
        Ok(())
    }

    fn check_return_static_local_args(&mut self, mut expr: ExprId) -> Result<()> {
        expr = self.unwrap_parens(expr);
        match *self.body.expr(expr) {
            Expr::Cast(inner, _) | Expr::Ref { expr: inner, .. } => {
                self.check_return_static_local_args(inner)
            }
            Expr::Block {
                kind,
                tail: Some(tail),
                ..
            } if kind != BlockKind::Const => self.check_return_static_local_args(tail),
            Expr::Tuple(list) | Expr::Array(list) => {
                for id in self.body.list(list).to_vec() {
                    self.check_return_static_local_args(ExprId(id))?;
                }
                Ok(())
            }
            Expr::Struct { fields, base, .. } => {
                for id in self.body.list(fields).to_vec() {
                    self.check_return_static_local_args(self.body.fields[id as usize].value)?;
                }
                if let Some(b) = base {
                    self.check_return_static_local_args(b)?;
                }
                Ok(())
            }
            Expr::Call(callee, args) => {
                let expr_ty = self.expr_ty(expr);
                let is_owned_new = (self.cx.boxed(expr_ty).is_some()
                    || matches!(
                        *self.cx.types.kind(expr_ty),
                        Kind::Adt(item, _) if is_owned_container(self.cx.name(item))
                    ))
                    && self
                        .call_sig_fn_item(expr, callee)
                        .is_some_and(|it| self.cx.name(it) == "new");
                let is_wrapper = matches!(
                    self.typed
                        .res
                        .get(&callee)
                        .or_else(|| self.typed.res.get(&expr)),
                    Some(Res::Ctor(..))
                ) || is_owned_new;
                if is_wrapper {
                    for id in self.body.list(args).to_vec() {
                        self.check_return_static_local_args(ExprId(id))?;
                    }
                }
                Ok(())
            }
            Expr::Path(_) => {
                if let Some(local) = self.expr_plain_local(expr)
                    && self.is_non_static_param(local)
                {
                    return Err(self.borrow_err(
                        Some(expr),
                        format!(
                            "borrowed data `{}` escapes outside of function: return type requires `'static` lifetime",
                            self.local_name(local)
                        ),
                    ));
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn check_return_invariant_expr(
        &mut self,
        expr: ExprId,
        sources: &[BorrowSource],
    ) -> Result<()> {
        let Some(ret_inv) = self.ret_invariant_lts() else {
            return Ok(());
        };
        let req_static = ret_inv.is_only_static();
        if !req_static && ret_inv.named.is_empty() {
            return Ok(());
        }
        let whole_sources: Vec<BorrowSource> = sources
            .iter()
            .filter(|s| {
                !s.steps
                    .iter()
                    .any(|st| matches!(st, PlaceStep::Field(..) | PlaceStep::SliceElem { .. }))
            })
            .cloned()
            .collect();
        for s in &whole_sources {
            if self.is_param_local(s.target) {
                let target_ty = self.local_ty(s.target);
                let is_inv_src = self.param_has_invariant_lt(s.target, false)
                    || (self.is_mut_ref_ty(target_ty)
                        && self.expr_plain_local(expr) == Some(s.target));
                if is_inv_src {
                    self.check_local_place_invariant_fit(
                        Some(expr),
                        s.target,
                        false,
                        req_static,
                        &ret_inv.named,
                    )?;
                } else if req_static {
                    self.check_sources_fit_invariant_lts(
                        Some(expr),
                        std::slice::from_ref(s),
                        true,
                        &[],
                    )?;
                }
            }
        }
        if let Some(local) = self.expr_plain_local(expr) {
            self.check_local_place_invariant_fit(
                Some(expr),
                local,
                false,
                req_static,
                &ret_inv.named,
            )?;
        }
        if let Some(inner_ref) = match *self.body.expr(self.unwrap_parens(expr)) {
            Expr::Ref { expr: inner, .. } => self.root_place(inner),
            _ => None,
        } && inner_ref.through_ref
            && inner_ref.steps.len() == 1
            && matches!(inner_ref.steps[0], PlaceStep::Deref)
        {
            let target_ty = self.local_ty(inner_ref.local);
            let inner_pointee = match *self.cx.types.kind(target_ty) {
                Kind::Ref(_, p) => p,
                _ => crate::sema::ty::Ty::NONE,
            };
            if self.is_mut_ref_ty(target_ty) || self.ty_has_borrow_interior_mut(inner_pointee) {
                self.check_local_place_invariant_fit(
                    Some(expr),
                    inner_ref.local,
                    false,
                    req_static,
                    &ret_inv.named,
                )?;
            }
        }
        Ok(())
    }
}

use crate::body::sig::{SelfKind, Signature};
use crate::body::{Expr, ExprId, List, Ty as AstTy, TyId};
use crate::names::ItemRef;
use crate::outline::item::Item;
use crate::outline::source::Source;
use crate::sema::infer::Res;
use crate::sema::ty::{Kind, Mut, Ty};

use super::super::calls::{
    Lifetimes, collect_ty_lifetimes, receiver_has_any_named, recv_is_by_ref, recv_ref_mut,
    sig_returns_only_static, ty_shares_name,
};
use super::super::{Checker, State};
use super::{BorrowSource, dedup_sources, unwrap_container_loans};

pub(super) struct SigCall<'a, 'b> {
    pub(super) expr: ExprId,
    pub(super) fn_item: ItemRef,
    pub(super) sig: &'b Signature,
    pub(super) sig_src: &'a Source,
    pub(super) sig_item: &'a Item,
    pub(super) receiver: Option<ExprId>,
    pub(super) arg_ids: &'b [u32],
}

struct SigFnFlags<'a> {
    fn_name: &'a str,
    is_iter_fn: bool,
    is_opt_fn: bool,
    is_owned_copy_fn: bool,
    is_by_ref: bool,
}

struct SigCallCtx<'a, 'b> {
    expr: ExprId,
    sig: &'b Signature,
    sig_src: &'a Source,
    sig_item: &'a Item,
    ret_ast: TyId,
    ret_lts: &'b Lifetimes<'a>,
    flags: SigFnFlags<'a>,
}

impl<'a> Checker<'_, 'a> {
    pub(in super::super) fn call_sig_fn_item(
        &mut self,
        expr: ExprId,
        callee: ExprId,
    ) -> Option<ItemRef> {
        let res = if self.body.names_callee(callee) {
            self.typed
                .res
                .get(&callee)
                .or_else(|| self.typed.res.get(&expr))
                .copied()
        } else {
            self.typed.res.get(&expr).copied()
        };
        match res {
            Some(Res::Fn(fn_item, _))
                if !matches!(self.cx.types.kind(self.expr_ty(callee)), Kind::Closure(..))
                    && !self
                        .cx
                        .owner_trait(fn_item)
                        .and_then(|tr| self.cx.lang(tr))
                        .is_some_and(crate::sema::infer::is_callable_lang) =>
            {
                Some(fn_item)
            }
            _ => None,
        }
    }

    pub(super) fn call_borrow_sources(
        &mut self,
        expr: ExprId,
        callee: ExprId,
        args: List,
        state: &State,
        extend_depth: Option<u32>,
    ) -> Vec<BorrowSource> {
        let ctor_res = if self.body.names_callee(callee) {
            self.typed
                .res
                .get(&callee)
                .or_else(|| self.typed.res.get(&expr))
                .copied()
        } else {
            self.typed.res.get(&expr).copied()
        };
        if matches!(ctor_res, Some(Res::Ctor(..))) {
            let arg_ids: Vec<u32> = self.body.list(args).to_vec();
            let mut out = Vec::new();
            for id in arg_ids {
                out.extend(self.expr_borrow_sources_inner(ExprId(id), state, true, extend_depth));
            }
            return dedup_sources(out);
        }
        if let Some(fn_item) = self.call_sig_fn_item(expr, callee) {
            return self.sig_borrow_sources(expr, fn_item, None, args, state);
        }
        if matches!(self.cx.types.kind(self.expr_ty(callee)), Kind::FnPtr(..))
            && !self.ty_holds_non_dyn_borrow(self.expr_ty(expr))
        {
            return Vec::new();
        }
        let mut out = self.expr_borrow_sources_inner(callee, state, true, None);
        let ret_has_borrow = self.ty_holds_borrow(self.expr_ty(expr));
        let hrtb_non_ret = self.callee_hrtb_non_ret_params(callee, ret_has_borrow);
        let copies_inner_arg = self
            .closure_expr_of(callee)
            .is_some_and(|c| self.closure_copies_inner_ref(c));
        let arg_ids: Vec<u32> = self.body.list(args).to_vec();
        for (idx, id) in arg_ids.into_iter().enumerate() {
            if hrtb_non_ret.contains(&idx) {
                continue;
            }
            let arg = ExprId(id);
            if copies_inner_arg && idx == 0 {
                out.extend(self.copied_arg_borrow_sources(expr, arg, state));
            } else {
                out.extend(self.expr_borrow_sources_inner(arg, state, true, None));
            }
        }
        self.filter_owned_closure_param_sources(self.expr_ty(expr), &mut out);
        self.finish_sig_sources(expr, out, false)
    }

    pub(super) fn method_borrow_sources(
        &mut self,
        expr: ExprId,
        receiver: ExprId,
        params: List,
        state: &State,
    ) -> Vec<BorrowSource> {
        let Some(Res::Fn(fn_item, _)) = self.typed.res.get(&expr).copied() else {
            return Vec::new();
        };
        if self.body.list(params).iter().any(|&id| {
            matches!(*self.body.expr(ExprId(id)), Expr::Closure { body, .. } if self.is_raw_ptr_deref(body))
        }) {
            return Vec::new();
        }
        self.sig_borrow_sources(expr, fn_item, Some(receiver), params, state)
    }

    fn method_recv_borrow_sources(
        &mut self,
        call: ExprId,
        recv: ExprId,
        mutable: bool,
        state: &State,
    ) -> Vec<BorrowSource> {
        let Some(mut p) = self.root_place(recv) else {
            return self.non_place_borrow_sources(recv, mutable, state, None);
        };
        let derefs = self.typed.adjusts.get(&call).map_or(0, |a| a.derefs);
        if derefs > 0 {
            self.apply_deref_steps(call, self.expr_ty(recv), derefs, &mut p);
        }
        self.place_to_sources(&p, mutable, state)
    }

    pub(super) fn normalize_sig_args(
        &self,
        receiver: Option<ExprId>,
        args: List,
        sig: &Signature,
    ) -> (Option<ExprId>, Vec<u32>) {
        let raw_args = self.body.list(args);
        if receiver.is_none() && sig.receiver.is_some() && !raw_args.is_empty() {
            (Some(ExprId(raw_args[0])), raw_args[1..].to_vec())
        } else {
            (receiver, raw_args.to_vec())
        }
    }

    fn sig_borrow_sources(
        &mut self,
        expr: ExprId,
        fn_item: ItemRef,
        receiver: Option<ExprId>,
        args: List,
        state: &State,
    ) -> Vec<BorrowSource> {
        let Ok(sig) = self.cx.signature(fn_item) else {
            return Vec::new();
        };
        let (receiver, arg_ids) = self.normalize_sig_args(receiver, args, &sig);
        if sig.is_async {
            let mut out = Vec::new();
            if let Some(recv) = receiver {
                if let Some(mutability) = self.typed.adjusts.get(&expr).and_then(|a| a.autoref) {
                    out.extend(self.method_recv_borrow_sources(
                        expr,
                        recv,
                        mutability == Mut::Mut,
                        state,
                    ));
                } else {
                    out.extend(self.expr_borrow_sources(recv, state));
                }
            }
            for &arg in &arg_ids {
                out.extend(self.expr_borrow_sources(ExprId(arg), state));
            }
            return self.finish_sig_sources(expr, out, false);
        }
        let Some(ret_ast) = sig.ret else {
            return Vec::new();
        };
        let (sig_src, sig_item) = self.cx.item(fn_item);
        let call = SigCall {
            expr,
            fn_item,
            sig: &sig,
            sig_src,
            sig_item,
            receiver,
            arg_ids: &arg_ids,
        };
        self.sig_sources_for_ret(&call, ret_ast, self.expr_ty(expr), state)
    }

    pub(super) fn sig_sources_for_ret(
        &mut self,
        call: &SigCall<'a, '_>,
        ret_ast: TyId,
        ret_ty: Ty,
        state: &State,
    ) -> Vec<BorrowSource> {
        if ret_ty != Ty::NONE && !self.ty_holds_borrow(ret_ty) {
            return Vec::new();
        }
        let ret_lts = self.collect_ret_lifetimes(call.fn_item, call.sig, call.sig_src, ret_ast);
        let flags = self.sig_fn_flags(call.fn_item, call.sig);
        let ctx = SigCallCtx {
            expr: call.expr,
            sig: call.sig,
            sig_src: call.sig_src,
            sig_item: call.sig_item,
            ret_ast,
            ret_lts: &ret_lts,
            flags,
        };
        if !ret_lts.has_elided && ret_lts.named.is_empty() {
            return self.sig_sources_without_ret_lifetimes(
                &ctx,
                call.receiver,
                call.arg_ids,
                ret_ty,
                state,
            );
        }
        let mut srcs = self.sig_arg_sources_with_lifetimes(&ctx, call.arg_ids, state);
        if let Some(recv) = call.receiver {
            srcs.extend(self.sig_recv_sources_with_lifetimes(&ctx, recv, state));
        }
        self.finish_sources_for_ty(ret_ty, srcs, ctx.flags.is_by_ref)
    }

    fn collect_ret_lifetimes(
        &mut self,
        fn_item: ItemRef,
        sig: &Signature,
        sig_src: &'a Source,
        ret_ast: TyId,
    ) -> Lifetimes<'a> {
        let mut ret_lts = Lifetimes::default();
        collect_ty_lifetimes(&sig.body, sig_src, ret_ast, &mut ret_lts);
        if ret_lts.names.contains(&"Self")
            && let Some(block) = self.cx.owner_impl(fn_item)
            && let Ok(hdr) = self.cx.impl_header(block)
        {
            let impl_src = &self.cx.impl_file(block).source;
            collect_ty_lifetimes(&hdr.body, impl_src, hdr.self_ty, &mut ret_lts);
        }
        ret_lts
    }

    fn sig_fn_flags(&mut self, fn_item: ItemRef, sig: &Signature) -> SigFnFlags<'a> {
        let fn_name = self.cx.name(fn_item);
        let is_iter_fn = self.is_iter_mod(fn_item);
        let is_opt_fn = self.item_in_std_mod(fn_item, "option");
        let is_owned_copy_fn = recv_is_by_ref(sig)
            && ((fn_name == "clone"
                && self
                    .cx
                    .owner_trait(fn_item)
                    .or_else(|| {
                        self.cx
                            .owner_impl(fn_item)
                            .and_then(|b| self.cx.impl_trait(b))
                    })
                    .is_some_and(|tr| self.item_in_std_mod(tr, "clone")))
                || (fn_name == "to_vec" && self.item_in_std_mod(fn_item, "slice"))
                || sig
                    .ret
                    .is_some_and(|ret| self.sig_returns_type_param(fn_item, sig, ret)));
        let is_by_ref = is_iter_fn && fn_name == "by_ref";
        SigFnFlags {
            fn_name,
            is_iter_fn,
            is_opt_fn,
            is_owned_copy_fn,
            is_by_ref,
        }
    }

    fn sig_sources_without_ret_lifetimes(
        &mut self,
        ctx: &SigCallCtx<'a, '_>,
        receiver: Option<ExprId>,
        arg_ids: &[u32],
        ret_ty: Ty,
        state: &State,
    ) -> Vec<BorrowSource> {
        let (sig, sig_src, ret_lts) = (ctx.sig, ctx.sig_src, ctx.ret_lts);
        if sig_returns_only_static(sig, sig_src, ctx.ret_ast, ret_lts) {
            return Vec::new();
        }
        let mut out = Vec::new();
        let mut param_matched_own_generic = false;
        for (idx, &arg) in arg_ids.iter().enumerate() {
            let Some(&(_, param_ty)) = sig.params.get(idx) else {
                continue;
            };
            let matches_ret = ty_shares_name(&sig.body, sig_src, param_ty, &ret_lts.names)
                || ret_lts.names.contains(&"Self");
            if !matches_ret {
                continue;
            }
            if sig
                .generics
                .iter()
                .any(|g| ret_lts.names.contains(&sig_src.text(g.name)))
                && ty_shares_name(&sig.body, sig_src, param_ty, &ret_lts.names)
            {
                param_matched_own_generic = true;
            }
            if let AstTy::Ref { .. } = *sig.body.ty(param_ty) {
                let mut arg_srcs = self.copied_arg_borrow_sources(ctx.expr, ExprId(arg), state);
                self.filter_owned_closure_param_sources(ret_ty, &mut arg_srcs);
                out.extend(arg_srcs);
                continue;
            }
            let mut arg_srcs = self.expr_borrow_sources(ExprId(arg), state);
            if !self.ty_holds_closure(ret_ty)
                && let Some(c_expr) = self.closure_expr_of(ExprId(arg))
                && let Some((ret_real, ret_orig)) = self
                    .closure_effects
                    .get(&c_expr)
                    .and_then(|e| e.ret_targets.as_ref())
                    .cloned()
            {
                arg_srcs.retain(|s| ret_real.contains(&s.target) || ret_orig.contains(&s.target));
            }
            out.extend(arg_srcs);
        }
        if let Some(recv) = receiver
            && (!param_matched_own_generic || ret_lts.names.contains(&"Self"))
        {
            out.extend(self.sig_no_lt_recv_sources(
                (ctx.expr, recv, ret_ty),
                sig,
                arg_ids,
                &ctx.flags,
                state,
            ));
        }
        self.finish_sources_for_ty(ret_ty, out, ctx.flags.is_by_ref)
    }

    fn sig_no_lt_recv_sources(
        &mut self,
        (call_expr, recv, ret_ty): (ExprId, ExprId, Ty),
        sig: &Signature,
        arg_ids: &[u32],
        flags: &SigFnFlags<'a>,
        state: &State,
    ) -> Vec<BorrowSource> {
        let mut recv_sources =
            self.expr_borrow_sources_inner(recv, state, !recv_is_by_ref(sig), None);
        if recv_is_by_ref(sig) {
            self.filter_owned_closure_param_sources(ret_ty, &mut recv_sources);
        }
        let maps_inner_copy = (flags.is_iter_fn || flags.is_opt_fn)
            && matches!(
                flags.fn_name,
                "map" | "filter_map" | "flat_map" | "find_map"
            )
            && arg_ids.first().is_some_and(|&a| {
                self.closure_copies_inner_ref(ExprId(a)) || self.closure_peels_outer_ref(ExprId(a))
            });
        let owns_dyn_only =
            recv_is_by_ref(sig) && ret_ty != Ty::NONE && !self.ty_holds_non_dyn_borrow(ret_ty);
        if ((flags.is_iter_fn || flags.is_opt_fn) && matches!(flags.fn_name, "copied" | "cloned"))
            || flags.is_owned_copy_fn
            || maps_inner_copy
            || owns_dyn_only
        {
            return self.copied_recv_sources(
                call_expr,
                recv,
                recv_sources,
                state,
                flags.is_opt_fn || flags.is_owned_copy_fn || owns_dyn_only,
            );
        }
        if flags.is_opt_fn
            && flags.fn_name == "take"
            && self
                .typed
                .adjusts
                .get(&call_expr)
                .is_some_and(|a| a.derefs > 0)
        {
            return self.guard_take_recv_sources(call_expr, recv, recv_sources, state);
        }
        let mut closure_arg = |idx: usize| {
            arg_ids
                .get(idx)
                .is_some_and(|&a| self.iter_closure_ignores_param(ExprId(a), state))
        };
        if flags.is_opt_fn
            && ((matches!(flags.fn_name, "map" | "and_then") && closure_arg(0))
                || (matches!(flags.fn_name, "map_or" | "map_or_else") && closure_arg(1)))
        {
            return Vec::new();
        }
        if flags.is_iter_fn
            && matches!(flags.fn_name, "map" | "filter_map" | "flat_map")
            && closure_arg(0)
        {
            for s in &mut recv_sources {
                s.env_only = true;
            }
        }
        if flags.is_iter_fn && flags.fn_name == "collect" {
            recv_sources.retain(|s| !s.env_only);
        }
        recv_sources
    }

    fn sig_arg_sources_with_lifetimes(
        &mut self,
        ctx: &SigCallCtx<'a, '_>,
        arg_ids: &[u32],
        state: &State,
    ) -> Vec<BorrowSource> {
        let (sig, sig_src, ret_lts) = (ctx.sig, ctx.sig_src, ctx.ret_lts);
        let allow_elided_param = !recv_is_by_ref(sig);
        let mut out = Vec::new();
        for (idx, &arg) in arg_ids.iter().enumerate() {
            let Some(&(_, param_ty)) = sig.params.get(idx) else {
                continue;
            };
            let pairs: Vec<(TyId, ExprId)> =
                match (sig.body.ty(param_ty), *self.body.expr(ExprId(arg))) {
                    (AstTy::Tuple(ptys), Expr::Tuple(aexprs)) if ptys.len == aexprs.len => sig
                        .body
                        .list(*ptys)
                        .iter()
                        .zip(self.body.list(aexprs))
                        .map(|(&pt, &ae)| (TyId(pt), ExprId(ae)))
                        .collect(),
                    _ => vec![(param_ty, ExprId(arg))],
                };
            for (pt, ae) in pairs {
                let mut param_lts = Lifetimes::default();
                collect_ty_lifetimes(&sig.body, sig_src, pt, &mut param_lts);
                let matched = (ret_lts.has_elided
                    && allow_elided_param
                    && (param_lts.has_elided || !param_lts.named.is_empty()))
                    || ret_lts.named.iter().any(|n| param_lts.named.contains(n))
                    || (!matches!(sig.body.ty(pt), AstTy::Ref { .. })
                        && ty_shares_name(&sig.body, sig_src, pt, &ret_lts.names));
                if matched {
                    let inner_only = !ret_lts.has_elided
                        && matches!(sig.body.ty(pt), AstTy::Ref { lifetime: None, .. });
                    let mut arg_sources = if inner_only {
                        self.copied_arg_borrow_sources(ctx.expr, ae, state)
                    } else {
                        self.expr_borrow_sources_inner(ae, state, true, None)
                    };
                    if matches!(sig.body.ty(pt), AstTy::Ref { mutable: false, .. }) {
                        arg_sources = super::clamp_mut(arg_sources, false);
                    }
                    out.extend(arg_sources);
                }
            }
        }
        out
    }

    fn sig_recv_sources_with_lifetimes(
        &mut self,
        ctx: &SigCallCtx<'a, '_>,
        recv: ExprId,
        state: &State,
    ) -> Vec<BorrowSource> {
        let (sig, sig_src, ret_lts) = (ctx.sig, ctx.sig_src, ctx.ret_lts);
        let recv_creates_borrow = match sig.receiver {
            Some((SelfKind::Ref | SelfKind::RefMut, _)) => {
                ret_lts.has_elided
                    || (!ret_lts.named.is_empty()
                        && receiver_has_any_named(sig_src, ctx.sig_item, &ret_lts.named))
            }
            Some((SelfKind::Typed(self_ty), _)) => {
                let mut self_lts = Lifetimes::default();
                collect_ty_lifetimes(&sig.body, sig_src, self_ty, &mut self_lts);
                (ret_lts.has_elided && (self_lts.has_elided || !self_lts.named.is_empty()))
                    || ret_lts.named.iter().any(|n| self_lts.named.contains(n))
            }
            Some((SelfKind::Value, _)) => true,
            None => false,
        };
        let recv_is_mut = recv_ref_mut(sig) == Some(Mut::Mut);
        if ctx.flags.is_by_ref {
            self.by_ref_recv_sources(ctx.expr, recv, state)
        } else if recv_creates_borrow
            && !recv_is_mut
            && self.is_mut_ref_to_shared_ref(self.expr_ty(recv))
        {
            let inner_srcs = if let Some(mut p) = self.root_place(recv) {
                p.reborrows_shared_field = true;
                self.place_to_sources(&p, false, state)
            } else {
                self.expr_borrow_sources_inner(recv, state, false, None)
            };
            unwrap_container_loans(inner_srcs, state, |s| {
                self.is_ref_ty(self.local_ty(s.target))
            })
        } else if recv_creates_borrow {
            let srcs = if let Some(mutability) =
                self.typed.adjusts.get(&ctx.expr).and_then(|a| a.autoref)
            {
                self.method_recv_borrow_sources(ctx.expr, recv, mutability == Mut::Mut, state)
            } else {
                self.expr_borrow_sources(recv, state)
            };
            let inner_only = !ret_lts.has_elided
                && matches!(sig.receiver, Some((SelfKind::Ref | SelfKind::RefMut, _)))
                && sig.receiver_lifetime.is_none();
            if inner_only {
                self.copied_recv_sources(ctx.expr, recv, srcs, state, true)
            } else {
                srcs
            }
        } else {
            if recv_is_by_ref(sig) {
                let adj = self.typed.adjusts.get(&ctx.expr);
                let mut pointee = self.expr_ty(recv);
                for _ in 0..adj.map_or(0, |a| a.derefs) {
                    if let Kind::Ref(_, inner) = *self.cx.types.kind(pointee) {
                        pointee = inner;
                    }
                }
                if adj.and_then(|a| a.autoref).is_none()
                    && let Kind::Ref(_, inner) = *self.cx.types.kind(pointee)
                {
                    pointee = inner;
                }
                if self.ty_is_borrow_free(pointee) {
                    return Vec::new();
                }
            }
            let srcs = self.expr_borrow_sources_inner(recv, state, !recv_is_by_ref(sig), None);
            if ctx.flags.is_owned_copy_fn {
                self.copied_recv_sources(ctx.expr, recv, srcs, state, true)
            } else {
                srcs
            }
        }
    }

    pub(super) fn copied_recv_sources(
        &mut self,
        call_expr: ExprId,
        recv: ExprId,
        mut recv_sources: Vec<BorrowSource>,
        state: &State,
        unwrap: bool,
    ) -> Vec<BorrowSource> {
        if unwrap && self.expr_is_any_downcast(recv, state) {
            return Vec::new();
        }
        let (root, indirect) = self.copied_recv_root(call_expr, recv);
        let call_ty = self.expr_ty(call_expr);
        let call_owned = !self.ty_holds_non_dyn_borrow(call_ty);
        let static_field = unwrap
            && !self.is_ref_ty(call_ty)
            && call_owned
            && self.recv_is_static_field_read(recv);
        let targets: Vec<u32> = recv_sources.iter().map(|s| s.target).collect();
        let mut container_targets = Vec::new();
        for s in &recv_sources {
            if (static_field && Some(s.target) == root)
                || self.is_copied_container_target(
                    s,
                    (call_ty, call_owned, unwrap),
                    (root, indirect, &targets),
                    state,
                )
            {
                container_targets.push(s.target);
            }
        }
        if unwrap {
            let out = unwrap_container_loans(recv_sources, state, |s| {
                container_targets.contains(&s.target)
            });
            let recv_is_mut_ref = self.is_mut_ref_ty(self.expr_ty(recv));
            self.finish_unwrapped_copied_sources(call_ty, recv_is_mut_ref, root, out, state)
        } else {
            let call_depth = self.ty_max_ref_depth(call_ty);
            for s in &mut recv_sources {
                if container_targets.contains(&s.target) {
                    s.env_only = true;
                    if self.is_param_local(s.target)
                        && self.param_inner_is_static(s.target)
                        && let Kind::Ref(_, pointee) = *self.cx.types.kind(self.local_ty(s.target))
                        && call_depth <= self.ty_max_ref_depth(pointee)
                    {
                        s.origin_only = true;
                    }
                }
            }
            recv_sources
        }
    }

    fn by_ref_recv_sources(
        &mut self,
        expr: ExprId,
        recv: ExprId,
        state: &State,
    ) -> Vec<BorrowSource> {
        if let Some(p) = self.root_place(recv) {
            let mut srcs =
                if let Some(mutability) = self.typed.adjusts.get(&expr).and_then(|a| a.autoref) {
                    self.method_recv_borrow_sources(expr, recv, mutability == Mut::Mut, state)
                } else {
                    self.expr_borrow_sources(recv, state)
                };
            for s in &mut srcs {
                if s.target == p.local
                    || (p.through_ref
                        && self.ty_holds_borrow(self.local_ty(s.target))
                        && state
                            .loans
                            .iter()
                            .any(|l| l.holder == Some(p.local) && l.target == s.target))
                {
                    s.env_only = true;
                }
            }
            srcs
        } else {
            let mut srcs = self.expr_borrow_sources_inner(recv, state, true, None);
            if self.is_non_promotable_borrow_temp(recv, true) {
                srcs.push(BorrowSource::temp(self.temp_scope_depth, true, true));
            }
            srcs
        }
    }

    pub(in super::super) fn copied_arg_borrow_sources(
        &mut self,
        call_expr: ExprId,
        arg: ExprId,
        state: &State,
    ) -> Vec<BorrowSource> {
        let cur = self.unwrap_parens(arg);
        if let Expr::Ref {
            raw: false,
            expr: inner,
            ..
        } = *self.body.expr(cur)
        {
            return self.expr_borrow_sources(inner, state);
        }
        let arg_sources = self.expr_borrow_sources_inner(arg, state, false, None);
        self.copied_recv_sources(call_expr, arg, arg_sources, state, true)
    }

    fn iter_closure_ignores_param(&mut self, arg: ExprId, state: &State) -> bool {
        if self.fn_arg_returns_static(arg) {
            return true;
        }
        let cur = self.unwrap_parens(arg);
        let Some(closure) = self.typed.closures.get(&cur).cloned() else {
            return false;
        };
        if self
            .closure_effects
            .get(&cur)
            .is_some_and(|eff| !eff.ret_params.is_empty())
        {
            return false;
        }
        let Some(targets) = self.closure_ret_targets(cur, closure.ret, state) else {
            return false;
        };
        targets.iter().all(|t| {
            closure.captures.iter().any(|c| c.local == *t)
                || state.loans.iter().any(|l| l.target == *t)
        })
    }

    fn finish_sig_sources(
        &mut self,
        expr: ExprId,
        out: Vec<BorrowSource>,
        keep_env_only: bool,
    ) -> Vec<BorrowSource> {
        self.finish_sources_for_ty(self.expr_ty(expr), out, keep_env_only)
    }

    fn finish_sources_for_ty(
        &mut self,
        ret_ty: Ty,
        mut out: Vec<BorrowSource>,
        keep_env_only: bool,
    ) -> Vec<BorrowSource> {
        if !keep_env_only && ret_ty != Ty::NONE && !self.ty_holds_closure(ret_ty) {
            out.retain(|s| !s.env_only);
            for s in &mut out {
                if self.is_ref_ty(self.local_ty(s.target)) {
                    s.through_ref = true;
                }
            }
        }
        dedup_sources(out)
    }
}

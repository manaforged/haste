use crate::body::sig::{SelfKind, Signature};
use crate::body::{Expr, ExprId, List, Ty as AstTy, TyId};
use crate::names::ItemRef;
use crate::outline::item::Item;
use crate::outline::source::Source;
use crate::sema::infer::Res;
use crate::sema::ty::{Kind, Mut, Ty};

use super::super::calls::{
    Lifetimes, collect_ty_lifetimes, receiver_has_any_named, recv_is_by_ref, recv_ref_mut,
    ty_shares_name,
};
use super::super::checks;
use super::super::{Checker, State};
use super::{BorrowSource, dedup_sources, unwrap_container_loans};

struct SigFnFlags<'a> {
    fn_name: &'a str,
    is_iter_fn: bool,
    is_opt_fn: bool,
    is_res_fn: bool,
    is_clone_fn: bool,
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
    pub(super) fn call_borrow_sources(
        &mut self,
        expr: ExprId,
        callee: ExprId,
        args: List,
        state: &State,
        extend_depth: Option<u32>,
    ) -> Vec<BorrowSource> {
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
            Some(Res::Ctor(..)) => {
                let arg_ids: Vec<u32> = self.body.list(args).to_vec();
                let mut out = Vec::new();
                for id in arg_ids {
                    out.extend(self.expr_borrow_sources_inner(
                        ExprId(id),
                        state,
                        true,
                        extend_depth,
                    ));
                }
                dedup_sources(out)
            }
            Some(Res::Fn(fn_item, _))
                if !matches!(self.cx.types.kind(self.expr_ty(callee)), Kind::Closure(..))
                    && !self
                        .cx
                        .owner_trait(fn_item)
                        .and_then(|tr| self.cx.lang(tr))
                        .is_some_and(crate::sema::infer::is_callable_lang) =>
            {
                self.sig_borrow_sources(expr, fn_item, None, args, state)
            }
            _ => {
                let mut out = self.expr_borrow_sources_inner(callee, state, true, None);
                let ret_has_borrow = self.ty_holds_borrow(self.expr_ty(expr));
                let hrtb_non_ret = self.callee_hrtb_non_ret_params(callee, ret_has_borrow);
                let arg_ids: Vec<u32> = self.body.list(args).to_vec();
                for (idx, id) in arg_ids.into_iter().enumerate() {
                    if hrtb_non_ret.contains(&idx) {
                        continue;
                    }
                    out.extend(self.expr_borrow_sources_inner(ExprId(id), state, true, None));
                }
                self.finish_sig_sources(expr, out, false)
            }
        }
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
        let Some(ret_ast) = sig.ret else {
            return Vec::new();
        };
        let (sig_src, sig_item) = self.cx.item(fn_item);
        let ret_lts = self.collect_ret_lifetimes(fn_item, &sig, sig_src, ret_ast);
        let raw_args = self.body.list(args);
        let (receiver, arg_ids): (Option<ExprId>, Vec<u32>) =
            if receiver.is_none() && sig.receiver.is_some() && !raw_args.is_empty() {
                (Some(ExprId(raw_args[0])), raw_args[1..].to_vec())
            } else {
                (receiver, raw_args.to_vec())
            };
        let flags = self.sig_fn_flags(fn_item, &sig);
        let ctx = SigCallCtx {
            expr,
            sig: &sig,
            sig_src,
            sig_item,
            ret_ast,
            ret_lts: &ret_lts,
            flags,
        };
        if !ret_lts.has_elided && ret_lts.named.is_empty() {
            return self.sig_sources_without_ret_lifetimes(&ctx, receiver, &arg_ids, state);
        }
        let mut out = self.sig_arg_sources_with_lifetimes(&ctx, &arg_ids, state);
        if let Some(recv) = receiver {
            out.extend(self.sig_recv_sources_with_lifetimes(&ctx, recv, state));
        }
        self.finish_sig_sources(expr, out, ctx.flags.is_by_ref)
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
        let is_res_fn = self.item_in_std_mod(fn_item, "result");
        let is_clone_fn = fn_name == "clone"
            && recv_is_by_ref(sig)
            && self
                .cx
                .owner_trait(fn_item)
                .or_else(|| {
                    self.cx
                        .owner_impl(fn_item)
                        .and_then(|b| self.cx.impl_trait(b))
                })
                .is_some_and(|tr| self.item_in_std_mod(tr, "clone"));
        let is_by_ref = is_iter_fn && fn_name == "by_ref";
        SigFnFlags {
            fn_name,
            is_iter_fn,
            is_opt_fn,
            is_res_fn,
            is_clone_fn,
            is_by_ref,
        }
    }

    fn sig_sources_without_ret_lifetimes(
        &mut self,
        ctx: &SigCallCtx<'a, '_>,
        receiver: Option<ExprId>,
        arg_ids: &[u32],
        state: &State,
    ) -> Vec<BorrowSource> {
        let (sig, sig_src, ret_lts) = (ctx.sig, ctx.sig_src, ctx.ret_lts);
        if checks::closure::sig_returns_only_static(sig, sig_src, ctx.ret_ast, ret_lts) {
            return Vec::new();
        }
        let mut out = Vec::new();
        let mut param_matched_own_generic = false;
        for (idx, &arg) in arg_ids.iter().enumerate() {
            if let Some(&(_, param_ty)) = sig.params.get(idx)
                && !matches!(sig.body.ty(param_ty), AstTy::Ref { .. })
                && (ty_shares_name(&sig.body, sig_src, param_ty, &ret_lts.names)
                    || ret_lts.names.contains(&"Self"))
            {
                if sig
                    .generics
                    .iter()
                    .any(|g| ret_lts.names.contains(&sig_src.text(g.name)))
                    && ty_shares_name(&sig.body, sig_src, param_ty, &ret_lts.names)
                {
                    param_matched_own_generic = true;
                }
                out.extend(self.expr_borrow_sources(ExprId(arg), state));
            }
        }
        if let Some(recv) = receiver
            && (!param_matched_own_generic || ret_lts.names.contains(&"Self"))
        {
            out.extend(self.sig_no_lt_recv_sources(sig, recv, arg_ids, &ctx.flags, state));
        }
        self.finish_sig_sources(ctx.expr, out, ctx.flags.is_by_ref)
    }

    fn sig_no_lt_recv_sources(
        &mut self,
        sig: &Signature,
        recv: ExprId,
        arg_ids: &[u32],
        flags: &SigFnFlags<'a>,
        state: &State,
    ) -> Vec<BorrowSource> {
        let mut recv_sources =
            self.expr_borrow_sources_inner(recv, state, !recv_is_by_ref(sig), None);
        if ((flags.is_iter_fn || flags.is_opt_fn) && matches!(flags.fn_name, "copied" | "cloned"))
            || flags.is_clone_fn
        {
            return self.copied_recv_sources(
                recv,
                recv_sources,
                state,
                flags.is_opt_fn || flags.is_clone_fn,
            );
        }
        let ignores_elem = (flags.is_iter_fn
            && matches!(flags.fn_name, "map" | "filter_map" | "flat_map")
            && arg_ids
                .first()
                .is_some_and(|&a| self.iter_closure_ignores_param(ExprId(a), state)))
            || ((flags.is_opt_fn || flags.is_res_fn)
                && matches!(flags.fn_name, "map" | "and_then")
                && arg_ids
                    .first()
                    .is_some_and(|&a| self.fn_arg_returns_static(ExprId(a))));
        if ignores_elem {
            for s in &mut recv_sources {
                s.env_only = true;
            }
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
                    let mut arg_sources =
                        self.expr_borrow_sources_inner(ae, state, !inner_only, None);
                    if matches!(sig.body.ty(pt), AstTy::Ref { mutable: false, .. }) {
                        for s in &mut arg_sources {
                            s.mutable = false;
                        }
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
            if let Some(mutability) = self.typed.adjusts.get(&ctx.expr).and_then(|a| a.autoref) {
                self.method_recv_borrow_sources(ctx.expr, recv, mutability == Mut::Mut, state)
            } else {
                self.expr_borrow_sources(recv, state)
            }
        } else {
            self.expr_borrow_sources_inner(recv, state, !recv_is_by_ref(sig), None)
        }
    }

    fn is_mut_ref_to_shared_ref(&self, ty: Ty) -> bool {
        ty != Ty::NONE
            && matches!(
                *self.cx.types.kind(ty),
                Kind::Ref(Mut::Mut, inner) if matches!(*self.cx.types.kind(inner), Kind::Ref(Mut::Not, _))
            )
    }

    fn copied_recv_sources(
        &mut self,
        recv: ExprId,
        mut recv_sources: Vec<BorrowSource>,
        state: &State,
        unwrap: bool,
    ) -> Vec<BorrowSource> {
        let mut cur = recv;
        loop {
            match *self.body.expr(cur) {
                Expr::Paren(inner)
                | Expr::Method {
                    receiver: inner, ..
                } => cur = inner,
                _ => break,
            }
        }
        let root = (!matches!(*self.body.expr(cur), Expr::Ref { .. }))
            .then(|| self.root_place(cur).map(|p| p.local))
            .flatten();
        let indirect = root.is_some_and(|r| {
            let r_ty = self.local_ty(r);
            cur == recv || self.is_ref_ty(r_ty) || self.ty_holds_closure(r_ty)
        });
        let targets: Vec<u32> = recv_sources.iter().map(|s| s.target).collect();
        let container_targets: Vec<u32> = recv_sources
            .iter()
            .filter(|s| {
                if checks::is_temp_target(s.target).is_some() || Some(s.target) == root {
                    return true;
                }
                let Some(r) = root.filter(|_| indirect) else {
                    return false;
                };
                if !state
                    .loans
                    .iter()
                    .any(|l| l.holder == Some(r) && l.target == s.target)
                {
                    return false;
                }
                if !self.ty_holds_borrow(self.local_ty(s.target)) {
                    return false;
                }
                !targets.iter().any(|&other| {
                    other != s.target
                        && state
                            .loans
                            .iter()
                            .any(|l| l.holder == Some(other) && l.target == s.target)
                })
            })
            .map(|s| s.target)
            .collect();
        if unwrap {
            unwrap_container_loans(recv_sources, state, |s| {
                container_targets.contains(&s.target)
            })
        } else {
            for s in &mut recv_sources {
                if container_targets.contains(&s.target) {
                    s.env_only = true;
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
            if self.is_non_promotable_temp(recv) {
                srcs.push(BorrowSource::temp(self.temp_scope_depth, true, true));
            }
            srcs
        }
    }

    fn iter_closure_ignores_param(&mut self, arg: ExprId, state: &State) -> bool {
        if self.fn_arg_returns_static(arg) {
            return true;
        }
        let cur = self.unwrap_parens(arg);
        let Some(closure) = self.typed.closures.get(&cur).cloned() else {
            return false;
        };
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
        mut out: Vec<BorrowSource>,
        keep_env_only: bool,
    ) -> Vec<BorrowSource> {
        let ret_ty = self.expr_ty(expr);
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

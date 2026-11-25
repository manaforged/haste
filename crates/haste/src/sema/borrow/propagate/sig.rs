use crate::body::sig::{SelfKind, Signature};
use crate::body::{Expr, ExprId, List, Pat, Ty as AstTy, TyId, UnOp};
use crate::names::ItemRef;
use crate::outline::item::Item;
use crate::outline::source::Source;
use crate::sema::infer::Res;
use crate::sema::ty::{Kind, Mut, Ty};

use super::super::calls::{
    Lifetimes, collect_ty_lifetimes, receiver_has_any_named, recv_is_by_ref, recv_ref_mut,
    sig_returns_only_static, ty_shares_name,
};
use super::super::checks;
use super::super::places::{PlaceStep, push_step};
use super::super::{Checker, State};
use super::{BorrowSource, dedup_sources, state_holds_other_target, unwrap_container_loans};

struct SigFnFlags<'a> {
    fn_name: &'a str,
    is_iter_fn: bool,
    is_opt_fn: bool,
    is_res_fn: bool,
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
        let raw_args = self.body.list(args);
        let (receiver, arg_ids): (Option<ExprId>, Vec<u32>) =
            if receiver.is_none() && sig.receiver.is_some() && !raw_args.is_empty() {
                (Some(ExprId(raw_args[0])), raw_args[1..].to_vec())
            } else {
                (receiver, raw_args.to_vec())
            };
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
        let ret_lts = self.collect_ret_lifetimes(fn_item, &sig, sig_src, ret_ast);
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
                || (fn_name == "to_vec" && self.item_in_std_mod(fn_item, "slice")));
        let is_by_ref = is_iter_fn && fn_name == "by_ref";
        SigFnFlags {
            fn_name,
            is_iter_fn,
            is_opt_fn,
            is_res_fn,
            is_owned_copy_fn,
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
            if let AstTy::Ref { .. } = *sig.body.ty(param_ty) {
                out.extend(self.copied_arg_borrow_sources(ctx.expr, ExprId(arg), state));
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
            out.extend(self.expr_borrow_sources(ExprId(arg), state));
        }
        if let Some(recv) = receiver
            && (!param_matched_own_generic || ret_lts.names.contains(&"Self"))
        {
            out.extend(
                self.sig_no_lt_recv_sources(ctx.expr, sig, recv, arg_ids, &ctx.flags, state),
            );
        }
        self.finish_sig_sources(ctx.expr, out, ctx.flags.is_by_ref)
    }

    fn sig_no_lt_recv_sources(
        &mut self,
        call_expr: ExprId,
        sig: &Signature,
        recv: ExprId,
        arg_ids: &[u32],
        flags: &SigFnFlags<'a>,
        state: &State,
    ) -> Vec<BorrowSource> {
        let mut recv_sources =
            self.expr_borrow_sources_inner(recv, state, !recv_is_by_ref(sig), None);
        let maps_inner_copy = (flags.is_iter_fn || flags.is_opt_fn)
            && flags.fn_name == "map"
            && arg_ids
                .first()
                .is_some_and(|&a| self.closure_copies_inner_ref(ExprId(a)));
        if ((flags.is_iter_fn || flags.is_opt_fn) && matches!(flags.fn_name, "copied" | "cloned"))
            || flags.is_owned_copy_fn
            || maps_inner_copy
        {
            return self.copied_recv_sources(
                call_expr,
                recv,
                recv_sources,
                state,
                flags.is_opt_fn || flags.is_owned_copy_fn,
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
                    .is_some_and(|&a| self.iter_closure_ignores_param(ExprId(a), state)));
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
                    let mut arg_sources = if inner_only {
                        self.copied_arg_borrow_sources(ctx.expr, ae, state)
                    } else {
                        self.expr_borrow_sources_inner(ae, state, true, None)
                    };
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
        call_expr: ExprId,
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
        let call_ty = self.expr_ty(call_expr);
        let targets: Vec<u32> = recv_sources.iter().map(|s| s.target).collect();
        let container_targets: Vec<u32> = recv_sources
            .iter()
            .filter(|s| {
                let held_by_other = state_holds_other_target(&targets, s.target, state);
                if self.is_non_static_param(s.target) {
                    return !held_by_other && self.can_unwrap_inner_static_param(s, call_ty, false);
                }
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
                !held_by_other
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

    fn copied_arg_borrow_sources(
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

    fn closure_copies_inner_ref(&mut self, arg: ExprId) -> bool {
        let cur = self.unwrap_parens(arg);
        let Expr::Closure { params, body, .. } = *self.body.expr(cur) else {
            return false;
        };
        let Some((param_local, ret_ty)) = self
            .typed
            .closures
            .get(&cur)
            .filter(|c| c.params.len() == 1 && c.captures.is_empty())
            .map(|c| (c.params[0], c.ret))
        else {
            return false;
        };
        let param_ty = self.local_ty(param_local);
        let Kind::Ref(_, inner_ty) = *self.cx.types.kind(param_ty) else {
            return false;
        };
        let body = self.unwrap_parens_and_empty_blocks(body);
        if matches!(*self.cx.types.kind(inner_ty), Kind::Ref(Mut::Not, _)) {
            if let Expr::Unary(UnOp::Deref, inner) = *self.body.expr(body)
                && self.expr_plain_local(inner) == Some(param_local)
            {
                return true;
            }
            let param_ids = self.body.list(params);
            if param_ids.len() == 1
                && let Pat::Ref { pat: inner_pat, .. } =
                    *self.body.pat(self.body.params[param_ids[0] as usize].pat)
                && let Some(&bound) = self.typed.binds.get(&inner_pat)
                && self.expr_plain_local(body) == Some(bound)
            {
                return true;
            }
        }
        if matches!(
            *self.body.expr(body),
            Expr::Field(base, _) | Expr::TupleField(base, _) | Expr::Index(base, _)
                if self.expr_plain_local(base) == Some(param_local)
        ) && self.ty_is_copy(ret_ty)
        {
            return true;
        }
        false
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

    fn leading_ref_depth(&self, mut ty: Ty) -> u32 {
        let mut depth = 0;
        while ty != Ty::NONE
            && let Kind::Ref(_, inner) = *self.cx.types.kind(ty)
        {
            depth += 1;
            ty = inner;
        }
        depth
    }

    pub(super) fn ty_max_ref_depth(&self, ty: Ty) -> u32 {
        if ty == Ty::NONE {
            return 0;
        }
        match *self.cx.types.kind(ty) {
            Kind::Ref(_, inner) => 1 + self.ty_max_ref_depth(inner),
            Kind::Slice(inner) | Kind::Array(inner, _) | Kind::Pat(inner, ..) => {
                self.ty_max_ref_depth(inner)
            }
            Kind::Tuple(items) | Kind::Adt(_, items) => self
                .cx
                .types
                .substs(items)
                .to_vec()
                .into_iter()
                .map(|t| self.ty_max_ref_depth(t))
                .max()
                .unwrap_or(0),
            _ => 0,
        }
    }

    pub(in super::super) fn can_unwrap_inner_static_param(
        &mut self,
        s: &BorrowSource,
        res_ty: Ty,
        is_deref: bool,
    ) -> bool {
        if !s.through_ref
            || !self.is_non_static_param(s.target)
            || !self.param_inner_is_static(s.target)
            || !s.steps.iter().all(|st| matches!(st, PlaceStep::Deref))
        {
            return false;
        }
        let local_ty = self.local_ty(s.target);
        if res_ty == local_ty {
            return false;
        }
        let Kind::Ref(_, pointee) = *self.cx.types.kind(local_ty) else {
            return false;
        };
        if !self.ty_holds_borrow(pointee) {
            return false;
        }
        if is_deref && self.leading_ref_depth(res_ty) >= self.leading_ref_depth(local_ty) {
            return false;
        }
        self.ty_max_ref_depth(res_ty) <= self.ty_max_ref_depth(pointee)
    }

    pub(in super::super) fn promote_param_behind_deref(
        &self,
        s: &BorrowSource,
        is_cparam: bool,
    ) -> BorrowSource {
        let mut steps = s.steps.clone();
        if self.is_ref_ty(self.local_ty(s.target)) {
            push_step(&mut steps, PlaceStep::Deref);
        }
        BorrowSource {
            steps,
            through_ref: true,
            origin_only: s.origin_only || !is_cparam,
            ..s.clone()
        }
    }
}

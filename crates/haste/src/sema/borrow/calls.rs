mod arg_refs;
pub(super) mod bounds;
mod closure;
pub(super) mod lifetimes;
pub(super) mod return_lt;
mod static_ty;

use std::sync::Arc;

use anyhow::Result;
use rustc_hash::FxHashSet as HashSet;

use crate::body::sig::{SelfKind, Signature};
use crate::body::{ExprId, List, TyId};
use crate::sema::infer::{By, Res};
use crate::sema::ty::{Kind, Mut, Ty};

use super::bind::push_loan;
use super::checks::is_temp_target;
use super::places::PlaceStep;
use super::propagate::BorrowSource;
use super::{Checker, Loan, State, prune_moved_loans};

pub(super) use arg_refs::{recv_is_by_ref, recv_ref_mut, sig_param_idx};
pub(super) use closure::ActiveClosure;
pub(super) use lifetimes::{
    Lifetimes, ParamLifetimes, ast_ty_has_type_param, ast_ty_may_carry_elided_lt,
    callable_param_inputs, collect_ty_lifetimes, compute_param_lifetimes,
    for_each_param_bound_list, lifetime_outlives, pred_matches_name, receiver_has_any_named,
    sig_return_named_lifetimes, sig_returns_only_static, sig_type_params, ty_shares_name,
};
use static_ty::non_env_sources;

#[derive(Default)]
pub(super) struct ArgListCtx<'a> {
    pub(super) sig: Option<&'a Signature>,
    pub(super) fn_item: Option<crate::names::ItemRef>,
    pub(super) recv_sources: Option<&'a [BorrowSource]>,
    pub(super) call_expr: Option<ExprId>,
    pub(super) callee_ty: Option<Ty>,
    pub(super) callee_expr: Option<ExprId>,
    pub(super) ctor_res: Option<(crate::names::ItemRef, u32)>,
    pub(super) call_res: Option<Res>,
    pub(super) param_refs: Option<&'a [Option<Mut>]>,
    pub(super) arg_tys: Option<&'a [Ty]>,
    pub(super) expected_from_arg_tys: bool,
    pub(super) has_self_arg: bool,
}

impl Checker<'_, '_> {
    pub(super) fn check_call(
        &mut self,
        expr: ExprId,
        callee: ExprId,
        args: List,
        state: &mut State,
        outer_live: &HashSet<u32>,
        depth: u32,
    ) -> Result<()> {
        self.check_expr(callee, state, true, outer_live, false, depth)?;
        if let Some(c_local) = self.expr_plain_local(callee) {
            let c_ty = self.local_ty(c_local);
            if let Kind::Closure(parent, _, at) = *self.cx.types.kind(c_ty) {
                let closure = if parent == self.item {
                    self.typed.closures.get(&ExprId(at)).cloned()
                } else {
                    self.cx.closure(c_ty).map(|info| info.closure.clone())
                };
                if let Some(c) = closure {
                    if c.consumes {
                        self.check_can_move(c_local, &[], state)?;
                        self.mark_local_moved(state, c_local, Some(expr));
                    } else if parent == self.item
                        && c.captures.iter().any(|cap| {
                            cap.by == By::Mut
                                && self
                                    .closure_capture_info(ExprId(at), cap.by, cap.local)
                                    .iter()
                                    .any(|(_, m, ..)| *m)
                        })
                    {
                        self.check_borrow(c_local, false, false, &[], true, state)?;
                    }
                }
            }
        }
        let arg_ids: Vec<u32> = self.body.list(args).to_vec();
        if self.is_mem_drop(expr, callee)
            && let Some(&first) = arg_ids.first()
            && self.is_mut_ref_ty(self.expr_ty(ExprId(first)))
        {
            if let Some(local) = self.expr_plain_local(ExprId(first)) {
                self.check_can_move(local, &[], state)?;
                self.mark_local_moved(state, local, Some(expr));
                return Ok(());
            }
            if self.place_behind_ref_or_ptr(ExprId(first)) {
                return Err(self.borrow_err(Some(ExprId(first)), "cannot move out of a reference"));
            }
        }
        let fn_item = match self.callee_res(expr, Some(callee)) {
            Some(&Res::Fn(item, _)) => Some(item),
            _ => None,
        };
        let sig = self.callee_sig(expr, Some(callee));
        self.check_call_unsafe(fn_item, sig.as_deref())?;
        let callee_ty = self.expr_ty(callee);
        let call_res = self.callee_res(expr, Some(callee)).copied();
        let (is_ctor, ctor_res) = match call_res {
            Some(Res::Ctor(item, variant, _)) => (true, Some((item, variant))),
            _ => (false, None),
        };
        let call_extend = is_ctor.then_some(self.let_extend_depth).flatten();
        let has_self_arg = sig.as_deref().is_some_and(|s| s.receiver.is_some());
        let is_callable_method = fn_item.is_some_and(|it| self.is_callable_trait_method(it));
        self.with_let_extend(call_extend, |this| {
            let arg_sources = this.check_arg_list_sources(
                &arg_ids,
                &ArgListCtx {
                    sig: (!is_callable_method).then(|| sig.as_deref()).flatten(),
                    fn_item: fn_item.filter(|_| !is_callable_method),
                    call_expr: Some(expr),
                    callee_ty: Some(callee_ty),
                    callee_expr: Some(callee),
                    ctor_res,
                    call_res,
                    has_self_arg: !is_callable_method && has_self_arg,
                    ..ArgListCtx::default()
                },
                state,
                outer_live,
                depth,
            )?;
            this.check_call_bounds(
                &arg_ids,
                &arg_sources,
                sig.as_deref(),
                fn_item,
                has_self_arg,
            )?;
            this.check_callee_bound_args(callee, &arg_ids, &arg_sources, state)?;
            this.attach_call_out_sources(
                expr,
                &arg_ids,
                &arg_sources,
                sig.as_deref(),
                fn_item,
                has_self_arg,
                state,
            )?;
            this.apply_closure_call_loans(callee, &arg_ids, &arg_sources, state)
        })
    }

    fn callee_res(&self, expr: ExprId, callee: Option<ExprId>) -> Option<&Res> {
        if let Some(c) = callee
            && self.body.names_callee(c)
        {
            self.typed.res.get(&c).or_else(|| self.typed.res.get(&expr))
        } else {
            self.typed.res.get(&expr)
        }
    }

    fn callee_sig(&mut self, expr: ExprId, callee: Option<ExprId>) -> Option<Arc<Signature>> {
        let Some(&Res::Fn(item, _)) = self.callee_res(expr, callee) else {
            return None;
        };
        self.cx.signature(item).ok()
    }

    fn check_call_unsafe(
        &self,
        fn_item: Option<crate::names::ItemRef>,
        sig: Option<&Signature>,
    ) -> Result<()> {
        if self.unsafe_depth == 0 {
            let safe_before_2024 = sig.is_some_and(|s| s.deprecated_safe_2024)
                && self
                    .cx
                    .crate_edition(self.item.krate)
                    .is_some_and(|e| e < "2024");
            let is_unsafe = (sig.is_some_and(|s| s.is_unsafe) && !safe_before_2024)
                || fn_item.is_some_and(|it| self.cx.foreign(it) && !sig.is_some_and(|s| s.is_safe));
            if is_unsafe {
                let name = fn_item.map_or("function", |it| self.cx.name(it));
                return Err(self.unsafe_err(
                    None,
                    format!(
                        "call to unsafe function `{name}` is unsafe and requires unsafe function or block"
                    ),
                ));
            }
        }
        Ok(())
    }

    fn is_mem_drop(&mut self, expr: ExprId, callee: ExprId) -> bool {
        let Some(&Res::Fn(item, _)) = self.callee_res(expr, Some(callee)) else {
            return false;
        };
        self.item_in_std_mod(item, "mem") && self.cx.name(item) == "drop"
    }

    pub(super) fn check_method(
        &mut self,
        expr: ExprId,
        receiver: ExprId,
        params: List,
        state: &mut State,
        outer_live: &HashSet<u32>,
        depth: u32,
    ) -> Result<()> {
        self.with_let_extend(None, |this| {
            this.check_method_inner(expr, receiver, params, state, outer_live, depth)
        })
    }

    fn method_recv_borrow(&mut self, expr: ExprId, receiver: ExprId) -> Option<Mut> {
        let adjust = self.typed.adjusts.get(&expr).copied();
        adjust.and_then(|m| m.autoref).or_else(|| {
            self.infer_method_recv_borrow(expr, receiver, adjust.map_or(0, |a| a.derefs))
        })
    }

    fn check_borrowed_receiver(
        &mut self,
        expr: ExprId,
        receiver: ExprId,
        mutability: Mut,
        temp_id: u32,
        state: &mut State,
        outer_live: &HashSet<u32>,
        depth: u32,
    ) -> Result<Option<(u32, bool, bool, Vec<PlaceStep>)>> {
        let derefs = self.typed.adjusts.get(&expr).map_or(0, |a| a.derefs);
        if mutability == Mut::Mut {
            self.check_static_mut(receiver, derefs > 0)?;
        }
        let Some(mut p) = self.root_place(receiver) else {
            self.check_expr(receiver, state, true, outer_live, true, depth)?;
            if self.ty_holds_borrow(self.expr_ty(receiver)) {
                let recv_sources = self.expr_borrow_sources(receiver, state);
                super::bind::push_source_loans_filtered(
                    &mut state.loans,
                    Some(temp_id),
                    &recv_sources,
                    false,
                    Some(expr),
                    |_| true,
                    |s| s.mutable && mutability == Mut::Mut,
                );
            }
            return Ok(None);
        };
        if derefs > 0 {
            self.apply_deref_steps(expr, self.expr_ty(receiver), derefs, &mut p);
        }
        if mutability == Mut::Mut && p.behind_shared {
            return Err(self.borrow_err(
                Some(expr),
                "cannot borrow data in a `&` reference or `*const` pointer as mutable",
            ));
        }
        if let Some(prefix_len) = p.raw_ptr_prefix {
            self.check_can_read(p.local, &p.steps[..prefix_len], state)?;
            self.check_expr(receiver, state, true, outer_live, true, depth)?;
            return Ok(None);
        }
        let recv_is_mut_ref = self.is_mut_ref_ty(self.expr_ty(receiver));
        if mutability == Mut::Mut {
            state.set_tuple_sources(p.local, None);
        }
        let deferred_recv = if mutability == Mut::Mut {
            self.check_can_read_inner(p.local, &p.steps, state, true)?;
            Some((
                p.local,
                p.through_ref,
                p.mut_ref_root || recv_is_mut_ref,
                p.steps.clone(),
            ))
        } else {
            self.check_borrow(
                p.local,
                p.through_ref,
                p.mut_ref_root,
                &p.steps,
                false,
                state,
            )?;
            None
        };
        if !(deferred_recv.is_some() && self.expr_plain_local(receiver).is_some()) {
            self.check_expr(receiver, state, true, outer_live, true, depth)?;
        }
        if mutability == Mut::Not || !p.through_ref || p.mut_ref_root || recv_is_mut_ref {
            let mut loan = Loan::new(
                p.local,
                p.steps,
                Some(temp_id),
                mutability == Mut::Mut,
                p.through_ref,
                Some(expr),
            );
            loan.two_phase = mutability == Mut::Mut;
            push_loan(&mut state.loans, loan);
        }
        Ok(deferred_recv)
    }

    fn check_method_inner(
        &mut self,
        expr: ExprId,
        receiver: ExprId,
        params: List,
        state: &mut State,
        outer_live: &HashSet<u32>,
        depth: u32,
    ) -> Result<()> {
        let recv_borrow = self.method_recv_borrow(expr, receiver);
        let derefs = self.typed.adjusts.get(&expr).map_or(0, |a| a.derefs);
        if recv_borrow == Some(Mut::Mut)
            && (self.place_behind_shared_ref(receiver)
                || (derefs > 0 && self.is_shared_ref_or_ptr(self.expr_ty(receiver))))
        {
            return Err(self.borrow_err(
                Some(expr),
                "cannot borrow data in a `&` reference as mutable",
            ));
        }
        let temp_id = self.next_temp_holder();
        let deferred_recv = if let Some(mutability) = recv_borrow {
            self.check_borrowed_receiver(
                expr, receiver, mutability, temp_id, state, outer_live, depth,
            )?
        } else {
            let derefs = self.typed.adjusts.get(&expr).map_or(0, |a| a.derefs);
            let by_val_deref = derefs > 0
                && self
                    .callee_sig(expr, None)
                    .is_some_and(|s| matches!(s.receiver, Some((SelfKind::Value, _))));
            if by_val_deref {
                self.check_method_by_val_derefs(expr, receiver, derefs, state, outer_live, depth)?;
            } else {
                self.check_by_val_receiver(receiver, state, outer_live, depth)?;
            }
            None
        };
        if state.diverged {
            self.release_temp_holder(state, temp_id);
            return Ok(());
        }
        let param_ids: Vec<u32> = self.body.list(params).to_vec();
        let recv_closure_sources =
            if !param_ids.is_empty() && self.ty_holds_borrow(self.expr_ty(receiver)) {
                self.expr_borrow_sources(receiver, state)
            } else {
                Vec::new()
            };
        let mut arg_live = outer_live.clone();
        if self.ty_holds_borrow(self.expr_ty(expr)) {
            self.collect_live_expr(receiver, &mut arg_live);
        }
        for &id in &param_ids {
            self.collect_live_expr(ExprId(id), &mut arg_live);
        }
        super::prune_dead_loans(state, &arg_live);
        let call_res = self.typed.res.get(&expr).copied();
        let fn_item = match call_res {
            Some(Res::Fn(item, _)) => Some(item),
            _ => None,
        };
        let sig = self.callee_sig(expr, None);
        self.check_call_unsafe(fn_item, sig.as_deref())?;
        let res = self.check_arg_list_sources(
            &param_ids,
            &ArgListCtx {
                sig: sig.as_deref(),
                fn_item,
                recv_sources: Some(&recv_closure_sources),
                call_expr: Some(expr),
                call_res,
                ..ArgListCtx::default()
            },
            state,
            outer_live,
            depth,
        );
        self.release_temp_holder(state, temp_id);
        let arg_sources = res?;
        if !state.diverged
            && let Some((local, through_ref, mut_ref_root, steps)) = deferred_recv
        {
            let mut check_state = state.clone();
            super::prune_dead_loans(&mut check_state, outer_live);
            self.check_borrow(local, through_ref, mut_ref_root, &steps, true, &check_state)?;
        }
        if !state.diverged {
            self.check_call_bounds(&param_ids, &arg_sources, sig.as_deref(), fn_item, false)?;
            self.attach_method_arg_sources(
                expr,
                receiver,
                &param_ids,
                &arg_sources,
                recv_borrow == Some(Mut::Mut),
                (sig.as_deref(), fn_item),
                state,
            )?;
            self.attach_closure_arg_param_loans(
                Some(&recv_closure_sources),
                &param_ids,
                &arg_sources,
                state,
            )?;
        }
        Ok(())
    }

    fn check_method_by_val_derefs(
        &mut self,
        expr: ExprId,
        receiver: ExprId,
        derefs: u32,
        state: &mut State,
        outer_live: &HashSet<u32>,
        depth: u32,
    ) -> Result<()> {
        let mut cur = self.expr_ty(receiver);
        let mut crossed_ref = false;
        for step in 0..derefs {
            if self.typed.derefs.contains_key(&(expr, step)) {
                crossed_ref = true;
                cur = self.deref_target_ty(expr, step).unwrap_or(Ty::NONE);
            } else if cur != Ty::NONE {
                if matches!(self.cx.types.kind(cur), Kind::Ref(..) | Kind::Ptr(..)) {
                    crossed_ref = true;
                }
                cur = self.builtin_deref_ty(cur).unwrap_or(Ty::NONE);
            }
        }
        if cur != Ty::NONE && !self.ty_is_copy(cur) {
            if crossed_ref || self.place_behind_ref_or_ptr(receiver) {
                return Err(self.borrow_err(Some(expr), "cannot move out of a reference or deref"));
            }
            if let Some(mut p) = self.root_place(receiver) {
                self.apply_deref_steps(expr, self.expr_ty(receiver), derefs, &mut p);
                self.check_expr(receiver, state, true, outer_live, false, depth)?;
                self.check_can_move(p.local, &p.steps, state)?;
                self.push_partial_move(state, p.local, p.steps, Some(expr));
                return Ok(());
            }
        }
        self.check_expr(receiver, state, crossed_ref, outer_live, true, depth)
    }

    pub(super) fn check_arg_list_with_refs(
        &mut self,
        arg_ids: &[u32],
        param_refs: Option<&[Option<Mut>]>,
        arg_tys: Option<&[Ty]>,
        state: &mut State,
        outer_live: &HashSet<u32>,
        depth: u32,
    ) -> Result<()> {
        self.check_arg_list_sources(
            arg_ids,
            &ArgListCtx {
                param_refs,
                arg_tys,
                ..ArgListCtx::default()
            },
            state,
            outer_live,
            depth,
        )
        .map(|_| ())
    }

    pub(super) fn check_arg_list_expected_tys(
        &mut self,
        arg_ids: &[u32],
        param_refs: &[Option<Mut>],
        arg_tys: &[Ty],
        state: &mut State,
        outer_live: &HashSet<u32>,
        depth: u32,
    ) -> Result<()> {
        self.check_arg_list_sources(
            arg_ids,
            &ArgListCtx {
                param_refs: Some(param_refs),
                arg_tys: Some(arg_tys),
                expected_from_arg_tys: true,
                ..ArgListCtx::default()
            },
            state,
            outer_live,
            depth,
        )
        .map(|_| ())
    }

    fn check_call_arg(
        &mut self,
        arg: ExprId,
        effective_ref: Option<Mut>,
        expected_ty: Option<Ty>,
        deferred_mut_args: &mut Vec<(ExprId, u32, Vec<PlaceStep>)>,
        state: &mut State,
        outer_live: &HashSet<u32>,
        depth: u32,
    ) -> Result<bool> {
        let prev_expr = self.current_expr;
        self.current_expr = Some(arg);
        let res = self.check_call_arg_inner(
            arg,
            effective_ref,
            expected_ty,
            deferred_mut_args,
            state,
            outer_live,
            depth,
        );
        self.current_expr = prev_expr;
        res
    }

    fn check_call_arg_inner(
        &mut self,
        arg: ExprId,
        effective_ref: Option<Mut>,
        expected_ty: Option<Ty>,
        deferred_mut_args: &mut Vec<(ExprId, u32, Vec<PlaceStep>)>,
        state: &mut State,
        outer_live: &HashSet<u32>,
        depth: u32,
    ) -> Result<bool> {
        let shared_reborrow = effective_ref == Some(Mut::Not);
        let mut_reborrow = effective_ref == Some(Mut::Mut);
        if self.is_mut_ref_ty(self.expr_ty(arg))
            && let Some(local) = self.expr_plain_local(arg)
        {
            if shared_reborrow {
                self.check_can_read(local, &[], state)?;
                return Ok(false);
            }
            if mut_reborrow {
                self.check_can_read_inner(local, &[], state, true)?;
                deferred_mut_args.push((arg, local, Vec::new()));
                return Ok(true);
            }
            self.check_can_move(local, &[], state)?;
            self.mark_local_moved(state, local, Some(arg));
            return Ok(false);
        }
        if self.is_mut_ref_ty(self.expr_ty(arg))
            && let Some(p) = self.root_place(arg)
            && (shared_reborrow || (!p.behind_shared && mut_reborrow))
        {
            let mut is_deferred_mut = false;
            if let Some(prefix_len) = p.raw_ptr_prefix {
                self.check_can_read(p.local, &p.steps[..prefix_len], state)?;
            } else if shared_reborrow {
                self.check_can_read(p.local, &p.steps, state)?;
            } else {
                self.check_can_read_inner(p.local, &p.steps, state, true)?;
                deferred_mut_args.push((arg, p.local, p.steps.clone()));
                is_deferred_mut = true;
            }
            self.check_expr(arg, state, true, outer_live, true, depth)?;
            return Ok(is_deferred_mut);
        }
        if let crate::body::Expr::Tuple(list) = *self.body.expr(self.unwrap_parens(arg))
            && let Some(elem_tys) = expected_ty.and_then(|t| self.peeled_tuple_elems(t))
        {
            self.check_tuple_with_expected_tys(arg, list, &elem_tys, state, outer_live, depth)?;
            return Ok(false);
        }
        let reborrow_mut_expr = self.is_mut_ref_ty(self.expr_ty(arg))
            && (shared_reborrow || (mut_reborrow && !self.place_behind_shared_ref(arg)));
        self.check_expr(arg, state, reborrow_mut_expr, outer_live, true, depth)?;
        Ok(false)
    }

    fn push_closure_arg_param_origins(
        &mut self,
        arg: ExprId,
        idx: usize,
        ctx: &ArgListCtx<'_>,
        arg_sources: &[Vec<BorrowSource>],
    ) -> Vec<u32> {
        let cur = self.unwrap_parens(arg);
        let Some(params) = self.typed.closures.get(&cur).map(|c| c.params.clone()) else {
            return Vec::new();
        };
        let mut added = Vec::new();
        if params.len() == 1
            && let Some(recv) = ctx.recv_sources
            && ctx.fn_item.is_some_and(|it| {
                self.is_iter_mod(it)
                    || self.item_in_std_mod(it, "option")
                    || self.item_in_std_mod(it, "result")
                    || self.item_in_std_mod(it, "array")
            })
        {
            let origins = non_env_sources(recv);
            if !origins.is_empty() || self.is_by_val_map_fn(ctx.fn_item) {
                self.closure_param_origins.insert(params[0], origins);
                added.push(params[0]);
            }
            return added;
        }
        let Some(sig) = ctx.sig else {
            return self.mark_closure_arg_static_params(ctx.callee_expr, idx, &params);
        };
        let Some(param_idx) = sig_param_idx(idx, ctx.has_self_arg) else {
            return added;
        };
        let Some(&(_, param_ty)) = sig.params.get(param_idx) else {
            return added;
        };
        let sig_src = ctx.fn_item.map_or(self.src, |it| self.cx.item(it).0);
        let inputs = callable_param_inputs(sig, sig_src, param_ty);
        for (c_idx, in_ty) in inputs.into_iter().enumerate() {
            let Some(&c_local) = params.get(c_idx) else {
                continue;
            };
            let mut in_lts = Lifetimes::default();
            collect_ty_lifetimes(&sig.body, sig_src, in_ty, &mut in_lts);
            if in_lts.has_elided || in_lts.named.is_empty() {
                continue;
            }
            let ty_ties_to_in = |other_ty: TyId| -> bool {
                let mut other_lts = Lifetimes::default();
                collect_ty_lifetimes(&sig.body, sig_src, other_ty, &mut other_lts);
                other_lts.named.iter().any(|&n| {
                    in_lts
                        .named
                        .iter()
                        .any(|&in_lt| n == in_lt || lifetime_outlives(sig, sig_src, n, in_lt))
                }) || self.generic_param_ties_to_self(sig, sig_src, &other_lts, &in_lts)
            };
            let later_shares = sig
                .params
                .iter()
                .skip(param_idx + 1)
                .any(|&(_, later_ty)| ty_ties_to_in(later_ty));
            if later_shares {
                continue;
            }
            let mut origins = Vec::new();
            if let Some(recv) = ctx.recv_sources
                && ctx.fn_item.is_some_and(|it| {
                    receiver_has_any_named(sig_src, self.cx.item(it).1, &in_lts.named)
                })
            {
                origins.extend(non_env_sources(recv));
            }
            for (other_idx, srcs) in arg_sources.iter().enumerate() {
                let Some(other_p) = sig_param_idx(other_idx, ctx.has_self_arg) else {
                    continue;
                };
                let Some(&(_, other_ty)) = sig.params.get(other_p) else {
                    continue;
                };
                if ty_ties_to_in(other_ty) {
                    origins.extend(non_env_sources(srcs));
                }
            }
            if !origins.is_empty() {
                self.closure_param_origins.insert(c_local, origins);
                added.push(c_local);
            }
        }
        added
    }

    fn check_arg_list_sources(
        &mut self,
        arg_ids: &[u32],
        ctx: &ArgListCtx<'_>,
        state: &mut State,
        outer_live: &HashSet<u32>,
        depth: u32,
    ) -> Result<Vec<Vec<BorrowSource>>> {
        let mut after_arg = vec![HashSet::default(); arg_ids.len()];
        let mut live = outer_live.clone();
        for pos in (0..arg_ids.len()).rev() {
            after_arg[pos] = live.clone();
            self.collect_live_expr(ExprId(arg_ids[pos]), &mut live);
        }
        let temp_id = self.next_temp_holder();
        let mut deferred_mut_args: Vec<(ExprId, u32, Vec<PlaceStep>)> = Vec::new();
        let mut arg_sources = Vec::with_capacity(arg_ids.len());
        let res = (|| -> Result<()> {
            for (idx, &id) in arg_ids.iter().enumerate() {
                let arg = ExprId(id);
                let cache_expr = self.unwrap_parens(arg);
                let plain_local = self.expr_plain_local(cache_expr);
                if plain_local.is_some() {
                    self.temp_sources.remove(&cache_expr);
                }
                let effective_ref = self.resolve_call_arg_ref(ctx, idx, arg);
                let expected_ty = self.resolve_call_arg_ty(ctx, idx, arg);
                let shared_reborrow = effective_ref == Some(Mut::Not);
                let added_origins =
                    self.push_closure_arg_param_origins(arg, idx, ctx, &arg_sources);
                let arg_res = self.check_call_arg(
                    arg,
                    effective_ref,
                    expected_ty,
                    &mut deferred_mut_args,
                    state,
                    &after_arg[idx],
                    depth,
                );
                for local in added_origins {
                    self.closure_param_origins.remove(&local);
                }
                let is_deferred_mut = arg_res?;
                if state.diverged {
                    break;
                }
                let mut sources = self.expr_borrow_sources(arg, state);
                if shared_reborrow {
                    sources
                        .iter_mut()
                        .for_each(|s| s.mutable &= is_temp_target(s.target).is_some());
                }
                if effective_ref.is_none()
                    && let Some(local) = plain_local
                    && state.moved.get(local as usize).copied().unwrap_or(false)
                    && !sources.is_empty()
                {
                    self.temp_sources.insert(cache_expr, sources.clone());
                }
                super::bind::push_source_loans_filtered(
                    &mut state.loans,
                    Some(temp_id),
                    &sources,
                    is_deferred_mut,
                    Some(arg),
                    |_| true,
                    |s| s.mutable,
                );
                arg_sources.push(sources);
                prune_moved_loans(state);
                if ctx.call_expr.is_none() && idx + 1 < arg_ids.len() {
                    super::prune_dead_loans(state, &after_arg[idx]);
                }
            }
            if !state.diverged && !deferred_mut_args.is_empty() {
                super::prune_dead_loans(state, outer_live);
            }
            if !state.diverged {
                for (arg, local, steps) in deferred_mut_args {
                    if let Some(loan) = state.loans.iter().find(|l| {
                        !(l.holder == Some(temp_id) && l.two_phase)
                            && super::loan_conflicts_borrow(l, local, &steps)
                    }) {
                        let name = self.local_name(local);
                        let (msg, primary, sec) = if loan.mutable {
                            (
                                format!("cannot use `{name}` because it was mutably borrowed"),
                                "use of borrowed value occurs here",
                                "mutable borrow occurs here",
                            )
                        } else {
                            (
                                format!(
                                    "cannot borrow `{name}` as mutable because it is also borrowed as immutable"
                                ),
                                "mutable borrow occurs here",
                                "immutable borrow occurs here",
                            )
                        };
                        return Err(self.borrow_err_labeled(
                            Some(arg),
                            msg,
                            primary,
                            loan.site,
                            sec,
                        ));
                    }
                }
            }
            Ok(())
        })();
        self.release_temp_holder(state, temp_id);
        res.map(|()| arg_sources)
    }
}

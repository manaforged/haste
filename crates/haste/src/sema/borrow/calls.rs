mod closure;
mod lifetimes;

use std::sync::Arc;

use anyhow::Result;
use rustc_hash::FxHashSet as HashSet;

use crate::body::sig::{SelfKind, Signature};
use crate::body::{Body, ExprId, List, Ty as AstTy, TyId};
use crate::sema::infer::{By, Res};
use crate::sema::ty::{Kind, Mut, Ty};

use super::bind::push_loan;
use super::places::PlaceStep;
use super::propagate::BorrowSource;
use super::{Checker, State, prune_moved_loans};

pub(super) use lifetimes::{
    Lifetimes, collect_ty_lifetimes, compute_param_is_static, receiver_has_any_named,
    ty_shares_name,
};

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
        let is_ctor = matches!(self.callee_res(expr, Some(callee)), Some(Res::Ctor(..)));
        let call_extend = if is_ctor { self.let_extend_depth } else { None };
        let has_self_arg = sig.as_deref().is_some_and(|s| s.receiver.is_some());
        self.with_let_extend(call_extend, |this| {
            let arg_sources = this.check_arg_list_sources(
                &arg_ids,
                sig.as_deref(),
                Some(callee_ty),
                has_self_arg,
                state,
                outer_live,
                depth,
            )?;
            this.check_call_bounds(&arg_ids, sig.as_deref(), fn_item, has_self_arg, state)?;
            this.check_callee_bound_args(callee, &arg_ids, state)?;
            this.attach_call_out_sources(
                expr,
                &arg_ids,
                sig.as_deref(),
                fn_item,
                has_self_arg,
                state,
            )?;
            this.apply_closure_call_loans(callee, &arg_ids, &arg_sources, state)
        })
    }

    fn callee_param_ref_mut(&self, callee_ty: Option<Ty>, idx: usize) -> Option<Mut> {
        let mut ty = match callee_ty {
            Some(t) if t != Ty::NONE => t,
            _ => return None,
        };
        while let Kind::Ref(_, inner) = *self.cx.types.kind(ty) {
            ty = inner;
        }
        let param_ty = match *self.cx.types.kind(ty) {
            Kind::FnPtr(params, _) => self.cx.types.substs(params).get(idx).copied(),
            Kind::Dyn(_, args, ..) => self.cx.types.substs(args).first().copied().and_then(|tup| {
                match *self.cx.types.kind(tup) {
                    Kind::Tuple(elems) => self.cx.types.substs(elems).get(idx).copied(),
                    _ => None,
                }
            }),
            Kind::Closure(_, _, at) => self
                .typed
                .closures
                .get(&ExprId(at))
                .and_then(|c| c.params.get(idx))
                .map(|&l| self.local_ty(l)),
            _ => None,
        }?;
        match *self.cx.types.kind(param_ty) {
            Kind::Ref(m, _) => Some(m),
            _ => None,
        }
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
            let is_unsafe = sig.is_some_and(|s| s.is_unsafe)
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
            let sig = self.callee_sig(expr, None);
            let is_by_val = adjust.map_or(0, |a| a.derefs) == 0
                && sig
                    .as_deref()
                    .is_some_and(|s| matches!(s.receiver, Some((SelfKind::Value, _))));
            (!is_by_val && self.is_mut_ref_ty(self.expr_ty(receiver))).then(|| {
                if sig
                    .as_deref()
                    .is_some_and(|s| recv_ref_mut(s) == Some(Mut::Not))
                {
                    Mut::Not
                } else {
                    Mut::Mut
                }
            })
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
            self.check_expr(receiver, state, true, outer_live, false, depth)?;
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
            self.check_expr(receiver, state, true, outer_live, false, depth)?;
            return Ok(None);
        }
        let recv_is_mut_ref = self.is_mut_ref_ty(self.expr_ty(receiver));
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
            self.check_expr(receiver, state, true, outer_live, false, depth)?;
        }
        if mutability == Mut::Not || !p.through_ref || p.mut_ref_root || recv_is_mut_ref {
            push_loan(
                &mut state.loans,
                p.local,
                p.steps,
                Some(temp_id),
                mutability == Mut::Mut,
                p.through_ref,
                mutability == Mut::Mut,
                false,
                Some(expr),
            );
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
        if recv_borrow == Some(Mut::Mut)
            && (self.place_behind_shared_ref(receiver)
                || self.is_shared_ref_or_ptr(self.expr_ty(receiver)))
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
            let sig = self.callee_sig(expr, None);
            if derefs > 0
                && sig
                    .as_deref()
                    .is_some_and(|s| matches!(s.receiver, Some((SelfKind::Value, _))))
            {
                self.check_method_by_val_derefs(expr, receiver, derefs, state, outer_live, depth)?;
            } else {
                self.check_expr(receiver, state, false, outer_live, true, depth)?;
            }
            None
        };
        if state.diverged {
            state.loans.retain(|l| l.holder != Some(temp_id));
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
        let fn_item = match self.typed.res.get(&expr).copied() {
            Some(Res::Fn(item, _)) => Some(item),
            _ => None,
        };
        let sig = self.callee_sig(expr, None);
        self.check_call_unsafe(fn_item, sig.as_deref())?;
        let res = self.check_arg_list_sources(
            &param_ids,
            sig.as_deref(),
            None,
            false,
            state,
            outer_live,
            depth,
        );
        state.loans.retain(|l| l.holder != Some(temp_id));
        let arg_sources = res?;
        if !state.diverged
            && let Some((local, through_ref, mut_ref_root, steps)) = deferred_recv
        {
            let mut check_state = state.clone();
            super::prune_dead_loans(&mut check_state, outer_live);
            self.check_borrow(local, through_ref, mut_ref_root, &steps, true, &check_state)?;
        }
        if !state.diverged {
            self.check_call_bounds(&param_ids, sig.as_deref(), fn_item, false, state)?;
            self.attach_method_arg_sources(
                expr,
                receiver,
                &param_ids,
                recv_borrow == Some(Mut::Mut),
                sig.as_deref(),
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
        self.check_expr(receiver, state, false, outer_live, true, depth)
    }

    pub(super) fn check_arg_list(
        &mut self,
        arg_ids: &[u32],
        sig: Option<&Signature>,
        callee_ty: Option<Ty>,
        has_self_arg: bool,
        state: &mut State,
        outer_live: &HashSet<u32>,
        depth: u32,
    ) -> Result<()> {
        self.check_arg_list_sources(
            arg_ids,
            sig,
            callee_ty,
            has_self_arg,
            state,
            outer_live,
            depth,
        )
        .map(|_| ())
    }

    fn check_call_arg(
        &mut self,
        arg: ExprId,
        param_ref: Option<Mut>,
        deferred_mut_args: &mut Vec<(ExprId, u32, Vec<PlaceStep>)>,
        state: &mut State,
        outer_live: &HashSet<u32>,
        depth: u32,
    ) -> Result<bool> {
        let prev_expr = self.current_expr;
        self.current_expr = Some(arg);
        let res =
            self.check_call_arg_inner(arg, param_ref, deferred_mut_args, state, outer_live, depth);
        self.current_expr = prev_expr;
        res
    }

    fn check_call_arg_inner(
        &mut self,
        arg: ExprId,
        param_ref: Option<Mut>,
        deferred_mut_args: &mut Vec<(ExprId, u32, Vec<PlaceStep>)>,
        state: &mut State,
        outer_live: &HashSet<u32>,
        depth: u32,
    ) -> Result<bool> {
        let shared_reborrow = param_ref == Some(Mut::Not);
        let mut_reborrow = param_ref == Some(Mut::Mut)
            || (param_ref.is_none() && self.typed.coercions.contains_key(&arg));
        if self.is_mut_ref_ty(self.expr_ty(arg))
            && let Some(local) = self.expr_plain_local(arg)
        {
            if shared_reborrow {
                self.check_can_read(local, &[], state)?;
                return Ok(false);
            }
            self.check_can_read_inner(local, &[], state, true)?;
            deferred_mut_args.push((arg, local, Vec::new()));
            return Ok(true);
        }
        if self.is_mut_ref_ty(self.expr_ty(arg))
            && let Some(p) = self.root_place(arg)
            && (shared_reborrow || (!p.behind_shared && (mut_reborrow || param_ref.is_none())))
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
        let reborrow_mut_expr = self.is_mut_ref_ty(self.expr_ty(arg))
            && (shared_reborrow
                || ((mut_reborrow || param_ref.is_none()) && !self.place_behind_shared_ref(arg)));
        self.check_expr(arg, state, reborrow_mut_expr, outer_live, true, depth)?;
        Ok(false)
    }

    fn check_arg_list_sources(
        &mut self,
        arg_ids: &[u32],
        sig: Option<&Signature>,
        callee_ty: Option<Ty>,
        has_self_arg: bool,
        state: &mut State,
        outer_live: &HashSet<u32>,
        depth: u32,
    ) -> Result<Vec<Vec<BorrowSource>>> {
        let temp_id = self.next_temp_holder();
        let mut deferred_mut_args: Vec<(ExprId, u32, Vec<PlaceStep>)> = Vec::new();
        let mut arg_sources = Vec::with_capacity(arg_ids.len());
        let res = (|| -> Result<()> {
            for (idx, &id) in arg_ids.iter().enumerate() {
                let arg = ExprId(id);
                let param_ref = sig
                    .and_then(|s| param_ref_mut(s, idx, has_self_arg))
                    .or_else(|| self.callee_param_ref_mut(callee_ty, idx));
                let shared_reborrow = param_ref == Some(Mut::Not);
                let is_deferred_mut = self.check_call_arg(
                    arg,
                    param_ref,
                    &mut deferred_mut_args,
                    state,
                    outer_live,
                    depth,
                )?;
                if state.diverged {
                    break;
                }
                let sources = self.expr_borrow_sources(arg, state);
                super::bind::push_source_loans_filtered(
                    &mut state.loans,
                    Some(temp_id),
                    &sources,
                    is_deferred_mut,
                    Some(arg),
                    |_| true,
                    |s| s.mutable && !shared_reborrow,
                );
                arg_sources.push(sources);
                prune_moved_loans(state);
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
                        let (msg, primary, sec) = if loan.mutable {
                            (
                                format!(
                                    "cannot use `{}` because it was mutably borrowed",
                                    self.local_name(local)
                                ),
                                "use of borrowed value occurs here",
                                "mutable borrow occurs here",
                            )
                        } else {
                            (
                                format!(
                                    "cannot borrow `{}` as mutable because it is also borrowed as immutable",
                                    self.local_name(local)
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
        state.loans.retain(|l| l.holder != Some(temp_id));
        res.map(|()| arg_sources)
    }
}

pub(super) fn ast_ref_mut(body: &Body, mut ty: TyId) -> Option<Mut> {
    while let AstTy::Paren(inner) = *body.ty(ty) {
        ty = inner;
    }
    match *body.ty(ty) {
        AstTy::Ref { mutable, .. } => Some(if mutable { Mut::Mut } else { Mut::Not }),
        _ => None,
    }
}

pub(super) fn recv_ref_mut(sig: &Signature) -> Option<Mut> {
    match sig.receiver? {
        (SelfKind::Ref, _) => Some(Mut::Not),
        (SelfKind::RefMut, _) => Some(Mut::Mut),
        (SelfKind::Typed(ty), _) => ast_ref_mut(&sig.body, ty),
        (SelfKind::Value, _) => None,
    }
}

fn param_ref_mut(sig: &Signature, idx: usize, has_self_arg: bool) -> Option<Mut> {
    if has_self_arg {
        if idx == 0 {
            return recv_ref_mut(sig);
        }
        return sig
            .params
            .get(idx - 1)
            .and_then(|&(_, ty)| ast_ref_mut(&sig.body, ty));
    }
    sig.params
        .get(idx)
        .and_then(|&(_, ty)| ast_ref_mut(&sig.body, ty))
}

pub(super) fn recv_is_by_ref(sig: &Signature) -> bool {
    recv_ref_mut(sig).is_some()
}

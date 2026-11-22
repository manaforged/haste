use std::sync::Arc;

use anyhow::{Result, bail};
use rustc_hash::FxHashSet as HashSet;

use crate::body::sig::{SelfKind, Signature};
use crate::body::{Args, BlockKind, Body, Bound, Expr, ExprId, GenericArg, List, Stmt, Ty as AstTy, TyId};
use crate::outline::item::Item;
use crate::outline::source::Source;
use crate::sema::infer::{By, Res};
use crate::sema::ty::{Kind, Mut, Ty};

use super::bind::push_loan;
use super::places::{FieldKey, PlaceStep};
use super::propagate::{BorrowSource, dedup_sources};
use super::{Checker, State, prune_moved_loans};

#[derive(Default)]
pub(super) struct Lifetimes<'s> {
    pub(super) has_elided: bool,
    pub(super) saw_static: bool,
    pub(super) named: Vec<&'s str>,
    pub(super) names: Vec<&'s str>,
}

impl Checker<'_, '_> {
    pub(super) fn check_call(&mut self, expr: ExprId, callee: ExprId, args: List, state: &mut State, outer_live: &HashSet<u32>, depth: u32) -> Result<()> {
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
                        if let Some(slot) = state.moved.get_mut(c_local as usize) {
                            *slot = true;
                        }
                    } else if parent == self.item
                        && c.captures.iter().any(|cap| cap.by == By::Mut && self.closure_capture_info(ExprId(at), cap.by, cap.local).iter().any(|(_, m, ..)| *m))
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
                if let Some(slot) = state.moved.get_mut(local as usize) {
                    *slot = true;
                }
                return Ok(());
            }
            if self.place_behind_ref_or_ptr(ExprId(first)) {
                bail!("cannot move out of a reference");
            }
        }
        let fn_item = match self.callee_res(expr, Some(callee)) {
            Some(&Res::Fn(item, _)) => Some(item),
            _ => None,
        };
        let sig = self.callee_sig(expr, Some(callee));
        let callee_ty = self.expr_ty(callee);
        let is_ctor = matches!(self.callee_res(expr, Some(callee)), Some(Res::Ctor(..)));
        let call_extend = if is_ctor { self.let_extend_depth } else { None };
        let has_self_arg = sig.as_deref().is_some_and(|s| s.receiver.is_some());
        self.with_let_extend(call_extend, |this| {
            let arg_sources = this.check_arg_list_sources(&arg_ids, sig.as_deref(), Some(callee_ty), has_self_arg, state, outer_live, depth)?;
            this.check_call_bounds(&arg_ids, sig.as_deref(), fn_item, has_self_arg, state)?;
            this.attach_call_out_sources(expr, &arg_ids, sig.as_deref(), fn_item, has_self_arg, state)?;
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
            Kind::Dyn(_, args, ..) => self.cx.types.substs(args).first().copied().and_then(|tup| match *self.cx.types.kind(tup) {
                Kind::Tuple(elems) => self.cx.types.substs(elems).get(idx).copied(),
                _ => None,
            }),
            Kind::Closure(_, _, at) => self.typed.closures.get(&ExprId(at)).and_then(|c| c.params.get(idx)).map(|&l| self.local_ty(l)),
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

    fn is_mem_drop(&mut self, expr: ExprId, callee: ExprId) -> bool {
        let Some(&Res::Fn(item, _)) = self.callee_res(expr, Some(callee)) else { return false };
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
        self.with_let_extend(None, |this| this.check_method_inner(expr, receiver, params, state, outer_live, depth))
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
        let adjust = self.typed.adjusts.get(&expr).copied();
        let recv_borrow = adjust.and_then(|m| m.autoref).or_else(|| {
            let sig = self.callee_sig(expr, None);
            let is_by_val = adjust.map_or(0, |a| a.derefs) == 0
                && sig.as_deref().is_some_and(|s| matches!(s.receiver, Some((SelfKind::Value, _))));
            (!is_by_val && self.is_mut_ref_ty(self.expr_ty(receiver))).then(|| {
                if sig.as_deref().is_some_and(|s| recv_ref_mut(s) == Some(Mut::Not)) {
                    Mut::Not
                } else {
                    Mut::Mut
                }
            })
        });
        if recv_borrow == Some(Mut::Mut)
            && (self.place_behind_shared_ref(receiver) || self.is_shared_ref_or_ptr(self.expr_ty(receiver)))
        {
            bail!("cannot borrow data in a `&` reference as mutable");
        }
        let temp_id = self.next_temp_holder();
        let mut deferred_recv = None;
        if let Some(mutability) = recv_borrow {
            if mutability == Mut::Mut {
                self.check_static_mut(receiver, self.typed.adjusts.get(&expr).map_or(0, |a| a.derefs) > 0)?;
            }
            if let Some(mut p) = self.root_place(receiver) {
                let derefs = self.typed.adjusts.get(&expr).map_or(0, |a| a.derefs);
                if derefs > 0 {
                    self.apply_deref_steps(expr, self.expr_ty(receiver), derefs, &mut p);
                }
                if mutability == Mut::Mut && p.behind_shared {
                    bail!("cannot borrow data in a `&` reference or `*const` pointer as mutable");
                }
                if let Some(prefix_len) = p.raw_ptr_prefix {
                    self.check_can_read(p.local, &p.steps[..prefix_len], state)?;
                    self.check_expr(receiver, state, true, outer_live, false, depth)?;
                } else {
                    if mutability == Mut::Mut {
                        self.check_can_read_inner(p.local, &p.steps, state, true)?;
                        deferred_recv = Some((p.local, p.through_ref, p.mut_ref_root || self.is_mut_ref_ty(self.expr_ty(receiver)), p.steps.clone()));
                    } else {
                        self.check_borrow(p.local, p.through_ref, p.mut_ref_root, &p.steps, false, state)?;
                    }
                    if !(deferred_recv.is_some() && self.expr_plain_local(receiver).is_some()) {
                        self.check_expr(receiver, state, true, outer_live, false, depth)?;
                    }
                    if mutability == Mut::Not || !p.through_ref || p.mut_ref_root || self.is_mut_ref_ty(self.expr_ty(receiver)) {
                        push_loan(&mut state.loans, p.local, p.steps, Some(temp_id), mutability == Mut::Mut, p.through_ref, mutability == Mut::Mut, false);
                    }
                }
            } else {
                self.check_expr(receiver, state, true, outer_live, false, depth)?;
                if self.ty_holds_borrow(self.expr_ty(receiver)) {
                    let recv_sources = self.expr_borrow_sources(receiver, state);
                    super::bind::push_source_loans_filtered(
                        &mut state.loans,
                        Some(temp_id),
                        &recv_sources,
                        false,
                        |_| true,
                        |s| s.mutable && mutability == Mut::Mut,
                    );
                }
            }
        } else {
            let derefs = self.typed.adjusts.get(&expr).map_or(0, |a| a.derefs);
            let sig = self.callee_sig(expr, None);
            if derefs > 0 && sig.as_deref().is_some_and(|s| matches!(s.receiver, Some((SelfKind::Value, _)))) {
                self.check_method_by_val_derefs(expr, receiver, derefs, state, outer_live, depth)?;
            } else {
                self.check_expr(receiver, state, false, outer_live, true, depth)?;
            }
        }
        if state.diverged {
            state.loans.retain(|l| l.holder != Some(temp_id));
            return Ok(());
        }
        let param_ids: Vec<u32> = self.body.list(params).to_vec();
        let recv_closure_sources = if !param_ids.is_empty() && self.ty_holds_borrow(self.expr_ty(receiver)) {
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
        let res = self.check_arg_list_sources(&param_ids, sig.as_deref(), None, false, state, outer_live, depth);
        state.loans.retain(|l| l.holder != Some(temp_id));
        let arg_sources = res?;
        if !state.diverged && let Some((local, through_ref, mut_ref_root, steps)) = deferred_recv {
            let mut check_state = state.clone();
            super::prune_dead_loans(&mut check_state, outer_live);
            self.check_borrow(local, through_ref, mut_ref_root, &steps, true, &check_state)?;
        }
        if !state.diverged {
            self.check_call_bounds(&param_ids, sig.as_deref(), fn_item, false, state)?;
            self.attach_method_arg_sources(expr, receiver, &param_ids, recv_borrow == Some(Mut::Mut), sig.as_deref(), state)?;
            self.attach_closure_arg_param_loans(Some(&recv_closure_sources), &param_ids, &arg_sources, state)?;
        }
        Ok(())
    }

    fn check_method_by_val_derefs(&mut self, expr: ExprId, receiver: ExprId, derefs: u32, state: &mut State, outer_live: &HashSet<u32>, depth: u32) -> Result<()> {
        let mut cur = self.expr_ty(receiver);
        let mut crossed_ref = false;
        for step in 0..derefs {
            if self.typed.derefs.contains_key(&(expr, step)) {
                crossed_ref = true;
                cur = self.deref_target_ty(expr, step).unwrap_or(Ty::NONE);
            } else if cur != Ty::NONE {
                if matches!(self.cx.types.kind(cur), Kind::Ref(..) | Kind::Ptr(..)) { crossed_ref = true; }
                cur = self.builtin_deref_ty(cur).unwrap_or(Ty::NONE);
            }
        }
        if cur != Ty::NONE && !self.ty_is_copy(cur) {
            if crossed_ref || self.place_behind_ref_or_ptr(receiver) { bail!("cannot move out of a reference or deref"); }
            if let Some(mut p) = self.root_place(receiver) {
                self.apply_deref_steps(expr, self.expr_ty(receiver), derefs, &mut p);
                self.check_expr(receiver, state, true, outer_live, false, depth)?;
                self.check_can_move(p.local, &p.steps, state)?;
                self.push_partial_move(state, p.local, p.steps);
                return Ok(());
            }
        }
        self.check_expr(receiver, state, false, outer_live, true, depth)
    }

    pub(super) fn check_arg_list(&mut self, arg_ids: &[u32], sig: Option<&Signature>, callee_ty: Option<Ty>, has_self_arg: bool, state: &mut State, outer_live: &HashSet<u32>, depth: u32) -> Result<()> {
        self.check_arg_list_sources(arg_ids, sig, callee_ty, has_self_arg, state, outer_live, depth).map(|_| ())
    }

    fn check_arg_list_sources(&mut self, arg_ids: &[u32], sig: Option<&Signature>, callee_ty: Option<Ty>, has_self_arg: bool, state: &mut State, outer_live: &HashSet<u32>, depth: u32) -> Result<Vec<Vec<BorrowSource>>> {
        let temp_id = self.next_temp_holder();
        let mut deferred_mut_args: Vec<(u32, Vec<PlaceStep>)> = Vec::new();
        let mut arg_sources = Vec::with_capacity(arg_ids.len());
        let res = (|| -> Result<()> {
            for (idx, &id) in arg_ids.iter().enumerate() {
                let arg = ExprId(id);
                let param_ref = sig.and_then(|s| param_ref_mut(s, idx, has_self_arg)).or_else(|| self.callee_param_ref_mut(callee_ty, idx));
                let shared_reborrow = param_ref == Some(Mut::Not);
                let mut_reborrow = param_ref == Some(Mut::Mut) || (param_ref.is_none() && self.typed.coercions.contains_key(&arg));
                let mut is_deferred_mut = false;
                if self.is_mut_ref_ty(self.expr_ty(arg))
                    && let Some(local) = self.expr_plain_local(arg)
                {
                    if shared_reborrow {
                        self.check_not_moved(local, &[], state)?;
                        if state.loans.iter().any(|l| super::loan_conflicts_borrow(l, local, &[]) && l.mutable && !l.two_phase) {
                            bail!("cannot use `{}` because it was mutably borrowed", self.local_name(local));
                        }
                    } else {
                        self.check_can_read_inner(local, &[], state, true)?;
                        deferred_mut_args.push((local, Vec::new()));
                        is_deferred_mut = true;
                    }
                } else if self.is_mut_ref_ty(self.expr_ty(arg))
                    && let Some(p) = self.root_place(arg)
                    && (shared_reborrow || (!p.behind_shared && (mut_reborrow || param_ref.is_none())))
                {
                    if let Some(prefix_len) = p.raw_ptr_prefix {
                        self.check_can_read(p.local, &p.steps[..prefix_len], state)?;
                    } else if shared_reborrow {
                        self.check_can_read(p.local, &p.steps, state)?;
                    } else {
                        self.check_can_read_inner(p.local, &p.steps, state, true)?;
                        deferred_mut_args.push((p.local, p.steps.clone()));
                        is_deferred_mut = true;
                    }
                    self.check_expr(arg, state, true, outer_live, true, depth)?;
                } else {
                    let reborrow_mut_expr = self.is_mut_ref_ty(self.expr_ty(arg))
                        && (shared_reborrow || ((mut_reborrow || param_ref.is_none()) && !self.place_behind_shared_ref(arg)));
                    self.check_expr(arg, state, reborrow_mut_expr, outer_live, true, depth)?;
                }
                if state.diverged {
                    break;
                }
                let sources = self.expr_borrow_sources(arg, state);
                super::bind::push_source_loans_filtered(
                    &mut state.loans,
                    Some(temp_id),
                    &sources,
                    is_deferred_mut,
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
                for (local, steps) in deferred_mut_args {
                    if state.loans.iter().any(|l| !(l.holder == Some(temp_id) && l.two_phase) && super::loan_conflicts_borrow(l, local, &steps)) {
                        bail!("cannot use `{}` because it was mutably borrowed", self.local_name(local));
                    }
                }
            }
            Ok(())
        })();
        state.loans.retain(|l| l.holder != Some(temp_id));
        res.map(|()| arg_sources)
    }

    pub(super) fn check_closure(&mut self, expr: ExprId, params: Option<List>, body: ExprId, state: &mut State, depth: u32) -> Result<()> {
        let Some(closure) = self.typed.closures.get(&expr).cloned() else { return Ok(()) };
        let mut body_state = State {
            init: state.init.clone(), maybe_init: state.maybe_init.clone(), moved: state.moved.clone(),
            partial_moves: state.partial_moves.clone(), loans: Vec::new(), diverged: false,
        };
        let saved_no_move = self.no_move_captures.clone();
        if let Err(err) = self.check_closure_captures(expr, &closure, state, &mut body_state) {
            self.no_move_captures = saved_no_move;
            return Err(err);
        }
        let closure_depth = depth + 1;
        if let Some(params) = params {
            let param_ids: Vec<u32> = self.body.list(params).to_vec();
            for (idx, p_id) in param_ids.into_iter().enumerate() {
                let mut fallback = Vec::new();
                if let Some(&local) = closure.params.get(idx)
                    && (local as usize) < body_state.init.len()
                {
                    self.scope_depth[local as usize] = closure_depth;
                    body_state.init[local as usize] = true;
                    body_state.maybe_init[local as usize] = true;
                    body_state.moved[local as usize] = false;
                    fallback.push(self.param_fallback_source(local));
                }
                if let Err(err) = self.bind_pat(self.body.params[p_id as usize].pat, None, None, &[], &fallback, &mut body_state, (true, true, closure_depth)) {
                    self.no_move_captures = saved_no_move;
                    return Err(err);
                }
            }
        }
        let saved_returns = (self.returns_borrow, self.return_min_depth, self.closure_depth);
        let next_by_val = closure
            .captures
            .iter()
            .filter(|c| c.by == By::Value && !self.is_ref_ty(self.local_ty(c.local)))
            .map(|c| c.local)
            .collect();
        let saved_by_val = std::mem::replace(&mut self.closure_by_val_caps, next_by_val);
        self.returns_borrow = self.ty_holds_borrow(closure.ret);
        self.return_min_depth = closure_depth;
        self.closure_depth = closure_depth;
        let init_moved = body_state.moved.clone();
        let init_partial = body_state.partial_moves.clone();
        let empty = HashSet::default();
        let res = self.with_let_extend(None, |this| {
            if let Expr::Block { kind, stmts, tail, label } = *this.body.expr(body)
                && matches!(kind, BlockKind::Async | BlockKind::AsyncMove)
            {
                this.check_block(expr, kind, stmts, tail, label, &mut body_state, (false, true), &empty, depth)
            } else {
                this.check_expr(body, &mut body_state, false, &empty, true, closure_depth)
            }
        });
        self.no_move_captures = saved_no_move;
        let escape_res = res.and_then(|()| {
            if self.returns_borrow && !body_state.diverged {
                self.check_return_escape(body, &body_state)?;
            }
            Ok(())
        });
        (self.returns_borrow, self.return_min_depth, self.closure_depth) = saved_returns;
        self.closure_by_val_caps = saved_by_val;
        escape_res?;
        let cap_locals: Vec<u32> = closure.captures.iter().map(|c| c.local).collect();
        let moves_capture = Self::has_new_moves(&init_moved, &init_partial, &body_state, &cap_locals);
        let mutates_capture = closure.captures.iter().any(|c| {
            matches!(c.by, By::Mut | By::Value)
                && self.closure_capture_info(expr, c.by, c.local).iter().any(|(_, m, ..)| *m)
        });
        let borrows_local = self
            .closure_borrow_sources(expr, state)
            .iter()
            .any(|s| super::checks::is_temp_target(s.target).is_some() || !self.from_caller(s.target, s.through_ref));
        self.record_closure_body_loans(
            expr,
            &closure.params,
            closure_depth,
            &body_state,
            state,
            (moves_capture, mutates_capture, borrows_local),
        );
        Ok(())
    }

    fn check_closure_captures(&mut self, expr: ExprId, closure: &super::super::infer::Closure, state: &mut State, body_state: &mut State) -> Result<()> {
        let mut seen_whole = Vec::new();
        for cap in &closure.captures {
            let local = cap.local;
            if (local as usize) >= self.is_copy.len() { continue; }
            if cap.path.is_empty() {
                if seen_whole.contains(&local) { continue; }
                seen_whole.push(local);
            }
            let cap_steps: Vec<PlaceStep> = cap.path.iter().map(|&(_, f)| PlaceStep::Field(FieldKey::Tuple(f))).collect();
            let cap_ty = if cap.path.is_empty() { self.local_ty(local) } else { let r = self.local_ty(local); cap.ty(self.cx, r).unwrap_or(r) };
            let cap_copy = if cap.path.is_empty() { self.is_copy[local as usize] } else { self.ty_is_copy(cap_ty) };
            match cap.by {
                By::Ref | By::Mut => {
                    let is_mut_ref = self.is_mut_ref_ty(self.local_ty(local));
                    for (steps, is_mut, through_ref, mut_ref_root) in self.closure_capture_info(expr, cap.by, local) {
                        if steps.starts_with(&cap_steps) {
                            let ref_deref = is_mut_ref && steps.first() == Some(&PlaceStep::Deref);
                            self.check_borrow(local, through_ref || ref_deref, (is_mut && mut_ref_root) || ref_deref, &steps, is_mut, state)?;
                            if cap.by == By::Ref && !is_mut && !self.is_mut_ref_ty(cap_ty) && !mut_ref_root && !matches!(self.cx.types.kind(cap_ty), Kind::Closure(..)) {
                                push_loan(&mut body_state.loans, local, steps, None, false, false, false, false);
                            } else if cap.by == By::Mut && !closure.consumes {
                                self.no_move_captures.push((local, steps));
                            }
                        }
                    }
                }
                By::Value => {
                    if self.is_mut_ref_ty(cap_ty) {
                        self.check_borrow(local, false, true, &cap_steps, true, state)?;
                    } else if cap_copy {
                        self.check_can_read(local, &cap_steps, state)?;
                    } else if cap_steps.is_empty() {
                        let paths = self.closure_capture_info(expr, cap.by, local);
                        if !paths.is_empty() && paths.iter().all(|(s, ..)| !s.is_empty() && s.first() != Some(&PlaceStep::Deref)) {
                            for (steps, ..) in paths {
                                self.check_can_move(local, &steps, state)?;
                                self.push_partial_move(state, local, steps);
                            }
                        } else {
                            self.check_can_move(local, &[], state)?;
                            if let Some(slot) = state.moved.get_mut(local as usize) { *slot = true; }
                            state.partial_moves.retain(|m| m.local != local);
                        }
                    } else {
                        self.check_can_move(local, &cap_steps, state)?;
                        self.push_partial_move(state, local, cap_steps);
                    }
                }
            }
        }
        Ok(())
    }

    pub(super) fn closure_borrow_sources(&mut self, expr: ExprId, state: &State) -> Vec<BorrowSource> {
        let Some(closure) = self.typed.closures.get(&expr).cloned() else {
            return Vec::new();
        };
        let ret_targets = self.closure_ret_targets(expr, closure.ret, state);
        let capture_env_only = |target: u32| ret_targets.as_ref().is_some_and(|set| !set.contains(&target));
        let mut out = Vec::new();
        for cap in &closure.captures {
            let cap_steps: Vec<PlaceStep> = cap.path.iter().map(|&(_, f)| PlaceStep::Field(FieldKey::Tuple(f))).collect();
            if cap.by == By::Value {
                let cap_ty = cap.ty(self.cx, self.local_ty(cap.local)).unwrap_or_else(|_| self.local_ty(cap.local));
                if self.is_mut_ref_ty(cap_ty) {
                    let mut steps = cap_steps;
                    steps.push(PlaceStep::Deref);
                    out.push(BorrowSource { target: cap.local, steps, mutable: true, through_ref: true, env_only: capture_env_only(cap.local) });
                }
            } else {
                for (steps, is_mut, through_ref, _) in self.closure_capture_info(expr, cap.by, cap.local) {
                    if steps.starts_with(&cap_steps) {
                        out.push(BorrowSource { target: cap.local, steps, mutable: is_mut, through_ref, env_only: capture_env_only(cap.local) });
                    }
                }
            }
            for l in state.loans.iter().filter(|l| l.holder == Some(cap.local)) {
                out.push(BorrowSource { env_only: l.env_only || capture_env_only(l.target), ..BorrowSource::from_loan(l) });
            }
        }
        dedup_sources(out)
    }

    pub(super) fn closure_ret_targets(&mut self, expr: ExprId, ret_ty: Ty, state: &State) -> Option<HashSet<u32>> {
        if !self.ty_holds_borrow(ret_ty) {
            return Some(HashSet::default());
        }
        let Expr::Closure { body, .. } = *self.body.expr(expr) else {
            return None;
        };
        if let Some(cached) = self.temp_sources.get(&body)
            && !self.expr_has_return(body)
        {
            return Some(cached.iter().map(|s| s.target).collect());
        }
        let mut inner_locals = HashSet::default();
        if !self.can_analyze_closure_ret(body, &mut inner_locals) {
            return None;
        }
        let targets = self.expr_borrow_sources(body, state).into_iter().map(|s| s.target).collect();
        Some(targets)
    }

    fn can_analyze_closure_ret(&mut self, expr: ExprId, inner_locals: &mut HashSet<u32>) -> bool {
        match *self.body.expr(expr) {
            Expr::Paren(inner) | Expr::Cast(inner, _) | Expr::Try(inner) => {
                self.can_analyze_closure_ret(inner, inner_locals)
            }
            Expr::Block { kind, stmts, tail, .. } => {
                if matches!(kind, BlockKind::Async | BlockKind::AsyncMove) {
                    return false;
                }
                let stm_ids: Vec<u32> = self.body.list(stmts).to_vec();
                for id in stm_ids {
                    match self.body.stmts[id as usize] {
                        Stmt::Let { pat, init, otherwise, .. } => {
                            self.collect_pat_locals(pat, inner_locals);
                            if init.is_some_and(|e| self.expr_has_return(e))
                                || otherwise.is_some_and(|e| self.expr_has_return(e))
                            {
                                return false;
                            }
                        }
                        Stmt::Expr(e, _) => {
                            if self.expr_has_return(e) {
                                return false;
                            }
                        }
                        Stmt::Item(..) | Stmt::Macro { .. } | Stmt::Empty => {}
                    }
                }
                tail.is_none_or(|t| self.can_analyze_closure_ret(t, inner_locals))
            }
            Expr::If(c, t, e) => {
                !self.expr_has_return(c)
                    && self.can_analyze_closure_ret(t, inner_locals)
                    && e.is_none_or(|alt| self.can_analyze_closure_ret(alt, inner_locals))
            }
            Expr::Match(scrut, arms) => {
                if self.expr_has_return(scrut) {
                    return false;
                }
                let arm_ids: Vec<u32> = self.body.list(arms).to_vec();
                for id in arm_ids {
                    let arm = self.body.arms[id as usize];
                    let mut arm_locals = inner_locals.clone();
                    self.collect_pat_locals(arm.pat, &mut arm_locals);
                    if arm.guard.is_some_and(|g| self.expr_has_return(g))
                        || !self.can_analyze_closure_ret(arm.body, &mut arm_locals)
                    {
                        return false;
                    }
                }
                true
            }
            _ => !self.expr_has_return(expr) && !self.expr_uses_inner_borrow_local(expr, inner_locals),
        }
    }
}

pub(super) fn collect_ty_lifetimes<'s>(body: &Body, src: &'s Source, ty: TyId, out: &mut Lifetimes<'s>) {
    match *body.ty(ty) {
        AstTy::Ref { lifetime, ty: inner, .. } => {
            match lifetime {
                None => out.has_elided = true,
                Some(lt) => push_lt(src.text(lt), out),
            }
            collect_ty_lifetimes(body, src, inner, out);
        }
        AstTy::Path(path) => {
            let p = body.path(path);
            let segs = body.list(p.segments);
            if let Some(&first) = segs.first() {
                let name = src.text(body.segments[first as usize].name);
                if !out.names.contains(&name) {
                    out.names.push(name);
                }
            }
            for &seg_id in segs {
                match body.segments[seg_id as usize].args {
                    Some(Args::Angle(list)) => {
                        for &arg_id in body.list(list) {
                            match body.args[arg_id as usize] {
                                GenericArg::Lifetime(lt) => push_lt(src.text(lt), out),
                                GenericArg::Ty(t) | GenericArg::Binding { ty: t, .. } => {
                                    collect_ty_lifetimes(body, src, t, out);
                                }
                                GenericArg::Constraint { .. } | GenericArg::Const(_) => {}
                            }
                        }
                    }
                    Some(Args::Paren { inputs, output }) => {
                        for &t in body.list(inputs) {
                            collect_ty_lifetimes(body, src, TyId(t), out);
                        }
                        if let Some(o) = output {
                            collect_ty_lifetimes(body, src, o, out);
                        }
                    }
                    Some(Args::ReturnType) | None => {}
                }
            }
        }
        AstTy::Tuple(list) => {
            for &t in body.list(list) {
                collect_ty_lifetimes(body, src, TyId(t), out);
            }
        }
        AstTy::ImplTrait(_, list) | AstTy::DynTrait(list) => {
            for &b_id in body.list(list) {
                match body.bounds[b_id as usize] {
                    Bound::Lifetime(lt) => push_lt(src.text(lt), out),
                    Bound::Trait { path, .. } => {
                        for &seg_id in body.list(body.path(path).segments) {
                            if let Some(Args::Angle(args)) = body.segments[seg_id as usize].args {
                                for &arg_id in body.list(args) {
                                    match body.args[arg_id as usize] {
                                        GenericArg::Lifetime(lt) => push_lt(src.text(lt), out),
                                        GenericArg::Ty(t) | GenericArg::Binding { ty: t, .. } => {
                                            collect_ty_lifetimes(body, src, t, out);
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }
                    }
                    Bound::Use(_) => {}
                }
            }
        }
        AstTy::Slice(t) | AstTy::Array(t, _) | AstTy::Ptr { ty: t, .. } | AstTy::Paren(t) | AstTy::Pat(t, _) => {
            collect_ty_lifetimes(body, src, t, out);
        }
        _ => {}
    }
}

fn push_lt<'s>(text: &'s str, out: &mut Lifetimes<'s>) {
    if text == "'_" {
        out.has_elided = true;
    } else if text == "'static" {
        out.saw_static = true;
    } else if !out.named.contains(&text) {
        out.named.push(text);
    }
}

pub(super) fn ty_shares_name(body: &Body, src: &Source, ty: TyId, ret_names: &[&str]) -> bool {
    if ret_names.is_empty() {
        return false;
    }
    let mut param_lts = Lifetimes::default();
    collect_ty_lifetimes(body, src, ty, &mut param_lts);
    param_lts.names.iter().any(|n| ret_names.contains(n))
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
        return sig.params.get(idx - 1).and_then(|&(_, ty)| ast_ref_mut(&sig.body, ty));
    }
    sig.params.get(idx).and_then(|&(_, ty)| ast_ref_mut(&sig.body, ty))
}

pub(super) fn recv_is_by_ref(sig: &Signature) -> bool {
    recv_ref_mut(sig).is_some()
}

pub(super) fn receiver_has_any_named(src: &Source, item: &Item, named: &[&str]) -> bool {
    let mut at = item.start;
    while at < item.end && src.kind(at) != rustc_lexer::TokenKind::OpenParen {
        at += 1;
    }
    let end = (at + 12).min(item.end);
    for tok in at + 1..end {
        if src.text(tok) == "self" {
            return (at + 1..tok).any(|t| named.contains(&src.text(t)));
        }
    }
    false
}

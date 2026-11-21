use std::sync::Arc;

use rustc_hash::FxHashSet as HashSet;

use anyhow::{Result, bail};

use crate::body::sig::Signature;
use crate::body::{Expr, ExprId, List, Pat, PatId};
use crate::sema::infer::{By, Res};
use crate::sema::ty::{Kind, Mut, Ty};

use super::places::{FieldKey, Place, PlaceStep, push_step, steps_conflict};
use super::propagate::BorrowSource;
use super::{Checker, Loan, PartialMove, State, merge_states, prune_moved_loans};

impl Checker<'_, '_> {
    pub(super) fn push_sources_for_local(&mut self, local: u32, sources: &[BorrowSource], state: &mut State) {
        if sources.is_empty() {
            return;
        }
        let ty = self.local_ty(local);
        if ty != Ty::NONE && !self.ty_holds_borrow(ty) {
            return;
        }
        for s in sources {
            if s.target == local {
                continue;
            }
            let loan = Loan {
                target: s.target,
                steps: s.steps.clone(),
                holder: Some(local),
                mutable: s.mutable && (!s.through_ref || !self.is_shared_ref_or_ptr(self.local_ty(s.target))),
                through_ref: s.through_ref,
                two_phase: false,
                env_only: s.env_only,
            };
            if !state.loans.contains(&loan) {
                state.loans.push(loan);
            }
        }
    }

    pub(super) fn bind_pat(
        &mut self,
        pat: PatId,
        init: Option<ExprId>,
        scrut_place: Option<&Place>,
        pat_steps: &[PlaceStep],
        fallback_sources: &[BorrowSource],
        state: &mut State,
        options: (bool, bool, u32),
    ) -> Result<()> {
        let (init_val, move_on_val, scope_depth) = options;
        let unwrapped_init = init.map(|e| self.unwrap_parens(e));
        match *self.body.pat(pat) {
            Pat::Bind { sub, .. } => {
                if let Some(&local) = self.typed.binds.get(&pat)
                    && (local as usize) < state.init.len()
                {
                    self.scope_depth[local as usize] = scope_depth;
                    state.init[local as usize] = init_val;
                    state.moved[local as usize] = false;
                    state.partial_moves.retain(|m| m.local != local);
                    clear_rebound_loans(local, state);
                    if init_val {
                        self.bind_local_init(pat, local, unwrapped_init, scrut_place, pat_steps, fallback_sources, state, (move_on_val, sub.is_none()))?;
                    }
                }
                if let Some(sub) = sub {
                    self.bind_pat(sub, init, scrut_place, pat_steps, fallback_sources, state, options)?;
                }
            }
            Pat::Paren(inner) | Pat::Box(inner) => {
                self.bind_pat(inner, unwrapped_init, scrut_place, pat_steps, fallback_sources, state, options)?;
            }
            Pat::Ref { pat: inner, .. } => {
                let inner_init = unwrapped_init.and_then(|e| match *self.body.expr(e) {
                    Expr::Ref { expr: inner_e, .. } => Some(inner_e),
                    _ => None,
                });
                let next_fallback = if inner_init.is_none() {
                    if let Some(e) = unwrapped_init {
                        self.expr_borrow_sources(e, state)
                    } else {
                        fallback_sources.to_vec()
                    }
                } else {
                    Vec::new()
                };
                let mut next_steps = pat_steps.to_vec();
                push_step(&mut next_steps, PlaceStep::Deref);
                self.bind_pat(inner, inner_init, scrut_place, &next_steps, &next_fallback, state, options)?;
            }
            Pat::Tuple(list) => {
                let pat_ids: Vec<u32> = self.body.list(list).to_vec();
                let elem_exprs = unwrapped_init.and_then(|e| match *self.body.expr(e) {
                    Expr::Tuple(elist) => {
                        let es: Vec<u32> = self.body.list(elist).to_vec();
                        (es.len() == pat_ids.len()).then_some(es)
                    }
                    _ => None,
                });
                let next_fallback = self.fallback_or_empty(elem_exprs.is_none(), unwrapped_init, fallback_sources, state);
                self.bind_positional_pat(&pat_ids, elem_exprs.as_deref(), scrut_place, pat_steps, &next_fallback, state, options)?;
            }
            Pat::TupleStruct(_, list) => {
                let pat_ids: Vec<u32> = self.body.list(list).to_vec();
                let elem_exprs = unwrapped_init.and_then(|e| match *self.body.expr(e) {
                    Expr::Call(callee, args) if matches!(self.typed.res.get(&callee), Some(Res::Ctor(..))) => {
                        let es: Vec<u32> = self.body.list(args).to_vec();
                        (es.len() == pat_ids.len()).then_some(es)
                    }
                    _ => None,
                });
                let next_fallback = self.fallback_or_empty(elem_exprs.is_none(), unwrapped_init, fallback_sources, state);
                self.bind_positional_pat(&pat_ids, elem_exprs.as_deref(), scrut_place, pat_steps, &next_fallback, state, options)?;
            }
            Pat::Slice(list) => {
                let pat_ids: Vec<u32> = self.body.list(list).to_vec();
                let next_fallback = self.fallback_or_empty(true, unwrapped_init, fallback_sources, state);
                let mut next_steps = pat_steps.to_vec();
                push_step(&mut next_steps, PlaceStep::Opaque);
                for id in pat_ids {
                    self.bind_pat(PatId(id), None, scrut_place, &next_steps, &next_fallback, state, options)?;
                }
            }
            Pat::Struct { fields, .. } => {
                let field_pat_ids: Vec<u32> = self.body.list(fields).to_vec();
                let struct_fields = unwrapped_init.and_then(|e| match *self.body.expr(e) {
                    Expr::Struct { fields: ef, .. } => Some(self.body.list(ef).to_vec()),
                    _ => None,
                });
                let next_fallback = self.fallback_or_empty(struct_fields.is_none(), unwrapped_init, fallback_sources, state);
                for fp_id in field_pat_ids {
                    let fp = self.body.field_pats[fp_id as usize];
                    let fname = self.src.text(fp.name).to_owned();
                    let f_init = struct_fields.as_ref().and_then(|ef| {
                        ef.iter()
                            .find(|&&id| self.src.text(self.body.fields[id as usize].name) == fname)
                            .map(|&id| self.body.fields[id as usize].value)
                    });
                    let mut next_steps = pat_steps.to_vec();
                    push_step(&mut next_steps, PlaceStep::Field(FieldKey::Named(fname)));
                    self.bind_pat(fp.pat, f_init, scrut_place, &next_steps, &next_fallback, state, options)?;
                }
            }
            Pat::Or(list) => {
                let alts: Vec<u32> = self.body.list(list).to_vec();
                let base = state.clone();
                let mut merged: Option<State> = None;
                for alt in alts {
                    let mut alt_state = base.clone();
                    self.bind_pat(PatId(alt), init, scrut_place, pat_steps, fallback_sources, &mut alt_state, options)?;
                    match &mut merged {
                        Some(acc) => merge_states(acc, &alt_state),
                        None => merged = Some(alt_state),
                    }
                }
                if let Some(acc) = merged {
                    *state = acc;
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn fallback_or_empty(
        &mut self,
        needed: bool,
        unwrapped_init: Option<ExprId>,
        fallback_sources: &[BorrowSource],
        state: &State,
    ) -> Vec<BorrowSource> {
        if !needed {
            return Vec::new();
        }
        unwrapped_init
            .map(|e| self.expr_borrow_sources(e, state))
            .unwrap_or_else(|| fallback_sources.to_vec())
    }

    fn bind_local_init(
        &mut self,
        pat: PatId,
        local: u32,
        unwrapped_init: Option<ExprId>,
        scrut_place: Option<&Place>,
        pat_steps: &[PlaceStep],
        fallback_sources: &[BorrowSource],
        state: &mut State,
        (move_on_val, leaf_bind): (bool, bool),
    ) -> Result<()> {
        if let Some(&ref_mut) = self.typed.ref_binds.get(&pat) {
            let is_mut = ref_mut == Mut::Mut;
            let mut sources = match unwrapped_init {
                Some(e) => self.expr_borrow_sources(e, state),
                None => fallback_sources.to_vec(),
            };
            if sources.is_empty()
                && let Some(place) = scrut_place
            {
                let mut full_steps = place.steps.clone();
                for s in pat_steps {
                    push_step(&mut full_steps, s.clone());
                }
                self.check_borrow(
                    place.local,
                    place.through_ref,
                    place.mut_ref_root,
                    &full_steps,
                    is_mut,
                    state,
                )?;
                sources.push(BorrowSource {
                    target: place.local,
                    steps: full_steps,
                    mutable: is_mut,
                    through_ref: place.through_ref,
                    env_only: false,
                });
            } else if is_mut {
                for s in &mut sources {
                    s.mutable = true;
                }
            }
            self.push_sources_for_local(local, &sources, state);
            return Ok(());
        }
        if let Some(e) = unwrapped_init {
            let sources = self.expr_borrow_sources(e, state);
            self.push_sources_for_local(local, &sources, state);
        } else if !fallback_sources.is_empty() {
            self.push_sources_for_local(local, fallback_sources, state);
        }
        if move_on_val
            && leaf_bind
            && !self.is_copy[local as usize]
            && let Some(place) = scrut_place
            && !place.through_ref
        {
            let mut full_steps = place.steps.clone();
            for s in pat_steps {
                push_step(&mut full_steps, s.clone());
            }
            self.check_can_move(place.local, &full_steps, state)?;
            if full_steps.is_empty() {
                state.moved[place.local as usize] = true;
            } else {
                state.partial_moves.push(PartialMove {
                    local: place.local,
                    steps: full_steps,
                });
            }
        }
        Ok(())
    }

    fn bind_positional_pat(
        &mut self,
        pat_ids: &[u32],
        elem_exprs: Option<&[u32]>,
        scrut_place: Option<&Place>,
        pat_steps: &[PlaceStep],
        fallback_sources: &[BorrowSource],
        state: &mut State,
        options: (bool, bool, u32),
    ) -> Result<()> {
        let mut saw_rest = false;
        for (idx, &id) in pat_ids.iter().enumerate() {
            let p = PatId(id);
            if matches!(self.body.pat(p), Pat::Rest) {
                saw_rest = true;
                continue;
            }
            let sub_init = elem_exprs.filter(|_| !saw_rest).map(|es| ExprId(es[idx]));
            let mut next_steps = pat_steps.to_vec();
            if !saw_rest {
                push_step(&mut next_steps, PlaceStep::Field(FieldKey::Tuple(idx as u32)));
            } else {
                push_step(&mut next_steps, PlaceStep::Opaque);
            }
            self.bind_pat(p, sub_init, scrut_place, &next_steps, fallback_sources, state, options)?;
        }
        Ok(())
    }

    fn unwrap_parens(&self, mut expr: ExprId) -> ExprId {
        while let Expr::Paren(inner) = *self.body.expr(expr) {
            expr = inner;
        }
        expr
    }

    pub(super) fn assign_place(
        &mut self,
        lhs: ExprId,
        sources: &[BorrowSource],
        state: &mut State,
        outer_live: &HashSet<u32>,
        depth: u32,
    ) -> Result<()> {
        if self.place_behind_shared_ref(lhs) {
            bail!("cannot assign to data in a `&` reference");
        }
        match *self.body.expr(lhs) {
            Expr::Paren(inner) => return self.assign_place(inner, sources, state, outer_live, depth),
            Expr::Tuple(list) | Expr::Array(list) => {
                let ids: Vec<u32> = self.body.list(list).to_vec();
                for elem in ids {
                    self.assign_place(ExprId(elem), &[], state, outer_live, depth)?;
                }
                return Ok(());
            }
            Expr::Struct { fields, .. } => {
                let fids: Vec<u32> = self.body.list(fields).to_vec();
                for f in fids {
                    self.assign_place(self.body.fields[f as usize].value, &[], state, outer_live, depth)?;
                }
                return Ok(());
            }
            _ => {}
        }
        if let Some(p) = self.root_place(lhs) {
            self.check_can_mutate(p.local, p.through_ref, p.mut_ref_root, &p.steps, state, true)?;
            if !p.through_ref && (p.local as usize) < state.init.len() {
                if p.steps.is_empty() {
                    state.init[p.local as usize] = true;
                    state.moved[p.local as usize] = false;
                    state.partial_moves.retain(|m| m.local != p.local);
                    clear_rebound_loans(p.local, state);
                    let rebound: Vec<BorrowSource> = sources
                        .iter()
                        .filter(|s| !(s.target == p.local && s.steps.first() == Some(&PlaceStep::Deref)))
                        .cloned()
                        .collect();
                    self.push_sources_for_local(p.local, &rebound, state);
                } else {
                    state.partial_moves.retain(|m| !(m.local == p.local && steps_conflict(&m.steps, &p.steps)));
                }
            }
        } else {
            self.check_expr(lhs, state, true, outer_live, false, depth)?;
        }
        Ok(())
    }

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
                        state.moved[c_local as usize] = true;
                    } else if parent == self.item
                        && c.captures.iter().any(|cap| cap.by == By::Mut && self.closure_capture_info(ExprId(at), cap.by, cap.local).1)
                    {
                        self.check_borrow(c_local, false, false, &[], true, state)?;
                    }
                }
            }
        }
        let arg_ids: Vec<u32> = self.body.list(args).to_vec();
        if self.is_mem_drop(expr, callee)
            && let Some(&first) = arg_ids.first()
            && let Some(local) = self.expr_plain_local(ExprId(first))
            && self.is_mut_ref_ty(self.local_ty(local))
        {
            self.check_can_move(local, &[], state)?;
            state.moved[local as usize] = true;
            return Ok(());
        }
        let sig = self.callee_sig(expr, Some(callee));
        let callee_ty = self.expr_ty(callee);
        self.check_arg_list(&arg_ids, sig.as_deref(), Some(callee_ty), state, outer_live, depth)
    }

    fn callee_param_is_shared_ref(&self, callee_ty: Option<Ty>, idx: usize) -> bool {
        let mut ty = match callee_ty {
            Some(t) if t != Ty::NONE => t,
            _ => return false,
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
            _ => None,
        };
        param_ty.is_some_and(|t| matches!(*self.cx.types.kind(t), Kind::Ref(Mut::Not, _)))
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
        self.cx.name(item) == "drop" && matches!(self.cx.qualified(item).as_str(), "core::mem::drop" | "std::mem::drop")
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
        let recv_borrow = self.typed.adjusts.get(&expr).and_then(|m| m.autoref).or_else(|| {
            let sig = self.callee_sig(expr, None);
            (self.is_mut_ref_ty(self.expr_ty(receiver))
                && sig.as_deref().is_none_or(|s| !matches!(s.receiver, Some((crate::body::sig::SelfKind::Ref, _)))))
            .then_some(Mut::Mut)
        });
        if recv_borrow == Some(Mut::Mut)
            && (self.place_behind_shared_ref(receiver) || self.is_shared_ref_or_ptr(self.expr_ty(receiver)))
        {
            bail!("cannot borrow data in a `&` reference as mutable");
        }
        let temp_id = self.next_temp_holder();
        let mut deferred_recv = None;
        if let Some(mutability) = recv_borrow {
            if let Some(mut p) = self.root_place(receiver) {
                let derefs = self.typed.adjusts.get(&expr).map_or(0, |a| a.derefs);
                if derefs > 0 {
                    self.apply_deref_steps(expr, self.expr_ty(receiver), derefs, &mut p);
                }
                if mutability == Mut::Mut {
                    self.check_can_read_inner(p.local, &p.steps, state, true)?;
                    deferred_recv = Some((p.local, p.through_ref, p.mut_ref_root || self.is_mut_ref_ty(self.expr_ty(receiver)), p.steps.clone()));
                } else {
                    self.check_borrow(p.local, p.through_ref, p.mut_ref_root, &p.steps, false, state)?;
                }
                if !(deferred_recv.is_some() && self.expr_plain_local(receiver).is_some()) {
                    self.check_expr(receiver, state, true, outer_live, false, depth)?;
                }
                if mutability == Mut::Mut && (!p.through_ref || self.is_mut_ref_ty(self.local_ty(p.local))) {
                    state.loans.push(Loan {
                        target: p.local,
                        steps: p.steps,
                        holder: Some(temp_id),
                        mutable: true,
                        through_ref: p.through_ref,
                        two_phase: true,
                        env_only: false,
                    });
                }
            } else {
                if !(deferred_recv.is_some() && self.expr_plain_local(receiver).is_some()) {
                    self.check_expr(receiver, state, true, outer_live, false, depth)?;
                }
            }
        } else {
            self.check_expr(receiver, state, false, outer_live, true, depth)?;
        }
        let param_ids: Vec<u32> = self.body.list(params).to_vec();
        let sig = self.callee_sig(expr, None);
        let res = self.check_arg_list(&param_ids, sig.as_deref(), None, state, outer_live, depth);
        state.loans.retain(|l| l.holder != Some(temp_id));
        res?;
        if let Some((local, through_ref, mut_ref_root, steps)) = deferred_recv {
            super::prune_dead_loans(state, outer_live);
            self.check_borrow(local, through_ref, mut_ref_root, &steps, true, state)?;
        }
        Ok(())
    }

    pub(super) fn check_arg_list(
        &mut self,
        arg_ids: &[u32],
        sig: Option<&Signature>,
        callee_ty: Option<Ty>,
        state: &mut State,
        outer_live: &HashSet<u32>,
        depth: u32,
    ) -> Result<()> {
        let temp_id = self.next_temp_holder();
        let mut deferred_mut_args = Vec::new();
        let res = (|| -> Result<()> {
            for (idx, &id) in arg_ids.iter().enumerate() {
                let arg = ExprId(id);
                let shared_reborrow = sig.is_some_and(|s| super::propagate::param_is_shared_ref(s, idx))
                    || self.callee_param_is_shared_ref(callee_ty, idx);
                let mut is_deferred_mut = false;
                if let Some(local) = self.expr_plain_local(arg)
                    && self.is_mut_ref_ty(self.local_ty(local))
                {
                    if shared_reborrow {
                        self.check_not_moved(local, &[], state)?;
                        if state.loans.iter().any(|l| super::loan_conflicts_borrow(l, local, &[]) && l.mutable && !l.two_phase) {
                            bail!("cannot use `{}` because it was mutably borrowed", self.local_name(local));
                        }
                    } else {
                        self.check_can_read_inner(local, &[], state, true)?;
                        deferred_mut_args.push(local);
                        is_deferred_mut = true;
                    }
                } else {
                    self.check_expr(arg, state, false, outer_live, true, depth)?;
                }
                for s in self.expr_borrow_sources(arg, state) {
                    state.loans.push(Loan {
                        target: s.target,
                        steps: s.steps,
                        holder: Some(temp_id),
                        mutable: s.mutable && !shared_reborrow,
                        through_ref: s.through_ref,
                        two_phase: is_deferred_mut,
                        env_only: s.env_only,
                    });
                }
                prune_moved_loans(state);
            }
            if !deferred_mut_args.is_empty() {
                super::prune_dead_loans(state, outer_live);
            }
            for local in deferred_mut_args {
                if state.loans.iter().any(|l| {
                    !(l.holder == Some(temp_id) && l.two_phase)
                        && super::loan_conflicts_borrow(l, local, &[])
                }) {
                    bail!("cannot use `{}` because it was mutably borrowed", self.local_name(local));
                }
            }
            Ok(())
        })();
        state.loans.retain(|l| l.holder != Some(temp_id));
        res
    }

    pub(super) fn check_closure(
        &mut self,
        expr: ExprId,
        params: Option<List>,
        body: ExprId,
        state: &mut State,
        depth: u32,
    ) -> Result<()> {
        let Some(closure) = self.typed.closures.get(&expr).cloned() else {
            return Ok(());
        };
        let mut body_state = State {
            init: state.init.clone(),
            moved: state.moved.clone(),
            partial_moves: state.partial_moves.clone(),
            loans: Vec::new(),
            diverged: false,
        };
        for cap in &closure.captures {
            let local = cap.local;
            if (local as usize) >= self.is_copy.len() {
                continue;
            }
            match cap.by {
                By::Ref | By::Mut => {
                    let is_ref = self.is_ref_ty(self.local_ty(local));
                    let is_mut_ref = self.is_mut_ref_ty(self.local_ty(local));
                    let (paths, is_mut) = self.closure_capture_info(expr, cap.by, local);
                    for steps in paths {
                        let through_ref = steps.first() == Some(&PlaceStep::Deref) && is_ref;
                        self.check_borrow(local, through_ref, is_mut && is_mut_ref, &steps, is_mut, state)?;
                    }
                }
                By::Value => {
                    if self.is_copy[local as usize] {
                        self.check_can_read(local, &[], state)?;
                    } else {
                        let (paths, _) = self.closure_capture_info(expr, cap.by, local);
                        if !paths.is_empty() && paths.iter().all(|s| !s.is_empty() && s.first() != Some(&PlaceStep::Deref)) {
                            for steps in paths {
                                self.check_can_move(local, &steps, state)?;
                                let pm = PartialMove { local, steps };
                                if !state.partial_moves.contains(&pm) {
                                    state.partial_moves.push(pm);
                                }
                            }
                        } else {
                            self.check_can_move(local, &[], state)?;
                            state.moved[local as usize] = true;
                            state.partial_moves.retain(|m| m.local != local);
                        }
                    }
                }
            }
        }
        let closure_depth = depth + 1;
        if let Some(params) = params {
            let param_ids: Vec<u32> = self.body.list(params).to_vec();
            for (idx, p_id) in param_ids.into_iter().enumerate() {
                if let Some(&local) = closure.params.get(idx)
                    && (local as usize) < body_state.init.len()
                {
                    self.scope_depth[local as usize] = closure_depth;
                    body_state.init[local as usize] = true;
                    body_state.moved[local as usize] = false;
                }
                let pat = self.body.params[p_id as usize].pat;
                self.bind_pat(pat, None, None, &[], &[], &mut body_state, (true, false, closure_depth))?;
            }
        }
        let saved_returns = (self.returns_borrow, self.return_min_depth);
        self.returns_borrow = self.ty_holds_borrow(closure.ret);
        self.return_min_depth = closure_depth;
        let empty = HashSet::default();
        self.check_expr(body, &mut body_state, false, &empty, true, closure_depth)?;
        if self.returns_borrow && !body_state.diverged {
            self.check_return_escape(body, &body_state)?;
        }
        (self.returns_borrow, self.return_min_depth) = saved_returns;
        Ok(())
    }
}

fn clear_rebound_loans(local: u32, state: &mut State) {
    state.loans.retain(|l| l.holder != Some(local) && l.target != local);
}

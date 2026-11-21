use std::sync::Arc;

use rustc_hash::FxHashSet as HashSet;

use anyhow::{Result, bail};

use crate::body::sig::Signature;
use crate::body::{BlockKind, Expr, ExprId, List, Pat, PatId};
use crate::sema::infer::{By, Res};
use crate::sema::ty::{Kind, Mut, Ty};

use super::places::{FieldKey, Place, PlaceStep, push_step, steps_conflict};
use super::propagate::BorrowSource;
use super::{Checker, Loan, PartialMove, State, merge_states, prune_moved_loans};

#[derive(Clone, Copy)]
struct PatCtx {
    init_val: bool,
    move_on_val: bool,
    scope_depth: u32,
    behind_ref: bool,
    in_drop_adt: bool,
    is_param: bool,
}

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

    pub(super) fn bind_pat_param(&mut self, pat: PatId, state: &mut State) -> Result<()> {
        let ctx = PatCtx { init_val: true, move_on_val: true, scope_depth: 0, behind_ref: false, in_drop_adt: false, is_param: true };
        self.bind_pat_ctx(pat, None, None, &[], &[], state, ctx)
    }

    pub(super) fn bind_pat(
        &mut self, pat: PatId, init: Option<ExprId>, scrut_place: Option<&Place>,
        pat_steps: &[PlaceStep], fallback_sources: &[BorrowSource], state: &mut State, options: (bool, bool, u32),
    ) -> Result<()> {
        let ctx = PatCtx { init_val: options.0, move_on_val: options.1, scope_depth: options.2, behind_ref: false, in_drop_adt: false, is_param: false };
        self.bind_pat_ctx(pat, init, scrut_place, pat_steps, fallback_sources, state, ctx)
    }

    fn bind_pat_ctx(
        &mut self, pat: PatId, init: Option<ExprId>, scrut_place: Option<&Place>,
        pat_steps: &[PlaceStep], fallback_sources: &[BorrowSource], state: &mut State, mut ctx: PatCtx,
    ) -> Result<()> {
        let unwrapped_init = init.map(|e| self.unwrap_parens(e));
        let (adjusted_place, adjusted_steps, crossed_ref) = self.apply_pat_derefs(pat, unwrapped_init, scrut_place, pat_steps);
        if crossed_ref {
            ctx.behind_ref = true;
        }
        let scrut_place = adjusted_place.as_ref();
        let pat_steps = adjusted_steps.as_slice();
        match *self.body.pat(pat) {
            Pat::Bind { sub, .. } => {
                if let Some(&local) = self.typed.binds.get(&pat)
                    && (local as usize) < state.init.len()
                {
                    if ctx.is_param && (ctx.behind_ref || !self.typed.ref_binds.contains_key(&pat)) {
                        self.is_param[local as usize] = true;
                    }
                    self.scope_depth[local as usize] = ctx.scope_depth;
                    state.init[local as usize] = ctx.init_val;
                    state.maybe_init[local as usize] = ctx.init_val;
                    state.moved[local as usize] = false;
                    state.partial_moves.retain(|m| m.local != local);
                    clear_rebound_loans(local, state);
                    if ctx.init_val {
                        let is_leaf = sub.is_none_or(|s| matches!(*self.body.pat(s), Pat::Rest));
                        self.bind_local_init(pat, local, unwrapped_init, scrut_place, pat_steps, fallback_sources, state, (ctx.move_on_val, is_leaf, ctx.behind_ref, ctx.in_drop_adt))?;
                    }
                }
                if let Some(sub) = sub {
                    self.bind_pat_ctx(sub, init, scrut_place, pat_steps, fallback_sources, state, ctx)?;
                }
            }
            Pat::Paren(inner) | Pat::Box(inner) => {
                self.bind_pat_ctx(inner, unwrapped_init, scrut_place, pat_steps, fallback_sources, state, ctx)?;
            }
            Pat::Ref { mutable, pat: inner } => {
                let inner_init = unwrapped_init.and_then(|e| match *self.body.expr(e) {
                    Expr::Ref { expr: inner_e, .. } => Some(inner_e),
                    _ => None,
                });
                let next_fallback = if inner_init.is_none() {
                    unwrapped_init.map(|e| self.expr_borrow_sources(e, state)).unwrap_or_else(|| fallback_sources.to_vec())
                } else {
                    Vec::new()
                };
                let next_place = if let Some(inner_e) = inner_init {
                    self.root_place(inner_e)
                } else {
                    scrut_place.cloned().map(|mut p| {
                        if !mutable && !p.reborrows_shared_field {
                            p.reborrows_shared_field = true;
                            p.behind_shared = true;
                            p.mut_ref_root = false;
                        } else if mutable && !p.reborrows_shared_field && !p.behind_shared {
                            p.mut_ref_root = true;
                        }
                        p.through_ref = true;
                        p
                    })
                };
                let mut next_steps = pat_steps.to_vec();
                if inner_init.is_none() {
                    push_step(&mut next_steps, PlaceStep::Deref);
                }
                ctx.behind_ref = true;
                self.bind_pat_ctx(inner, inner_init, next_place.as_ref(), &next_steps, &next_fallback, state, ctx)?;
            }
            Pat::Tuple(list) | Pat::TupleStruct(_, list) => {
                let is_ctor = matches!(*self.body.pat(pat), Pat::TupleStruct(..));
                if is_ctor && self.typed.pats.get(&pat).copied().is_some_and(|ty| self.ty_has_own_drop(ty)) {
                    ctx.in_drop_adt = true;
                }
                let pat_ids: Vec<u32> = self.body.list(list).to_vec();
                let elem_exprs = unwrapped_init.and_then(|e| match *self.body.expr(e) {
                    Expr::Tuple(elist) if !is_ctor => {
                        let es: Vec<u32> = self.body.list(elist).to_vec();
                        (es.len() == pat_ids.len()).then_some(es)
                    }
                    Expr::Call(callee, args) if is_ctor && matches!(self.typed.res.get(&callee), Some(Res::Ctor(..))) => {
                        let es: Vec<u32> = self.body.list(args).to_vec();
                        (es.len() == pat_ids.len()).then_some(es)
                    }
                    _ => None,
                });
                let next_fallback = self.fallback_or_empty(elem_exprs.is_none(), unwrapped_init, fallback_sources, state);
                self.bind_positional_pat(&pat_ids, elem_exprs.as_deref(), scrut_place, pat_steps, &next_fallback, state, ctx)?;
            }
            Pat::Slice(list) => {
                let steps = self.slice_pat_steps(pat, list);
                let next_fallback = self.fallback_or_empty(true, unwrapped_init, fallback_sources, state);
                for (id, step) in steps {
                    let mut next_steps = pat_steps.to_vec();
                    push_step(&mut next_steps, step);
                    self.bind_pat_ctx(id, None, scrut_place, &next_steps, &next_fallback, state, ctx)?;
                }
            }
            Pat::Struct { fields, .. } => {
                if self.typed.pats.get(&pat).copied().is_some_and(|ty| self.ty_has_own_drop(ty)) {
                    ctx.in_drop_adt = true;
                }
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
                    self.bind_pat_ctx(fp.pat, f_init, scrut_place, &next_steps, &next_fallback, state, ctx)?;
                }
            }
            Pat::Or(list) => {
                let base = state.clone();
                let mut merged: Option<State> = None;
                for alt in self.body.list(list).to_vec() {
                    let mut alt_state = base.clone();
                    self.bind_pat_ctx(PatId(alt), init, scrut_place, pat_steps, fallback_sources, &mut alt_state, ctx)?;
                    match &mut merged { Some(acc) => merge_states(acc, &alt_state), None => merged = Some(alt_state) }
                }
                if let Some(acc) = merged { *state = acc; }
            }
            _ => {}
        }
        Ok(())
    }

    fn apply_pat_derefs(&self, pat: PatId, unwrapped_init: Option<ExprId>, scrut_place: Option<&Place>, pat_steps: &[PlaceStep]) -> (Option<Place>, Vec<PlaceStep>, bool) {
        let derefs = self.typed.pat_derefs(pat);
        let mut adjusted_place = scrut_place.cloned();
        let mut adjusted_steps = pat_steps.to_vec();
        if derefs == 0 { return (adjusted_place, adjusted_steps, false); }
        let mut crossed_ref = false;
        let mut cur = unwrapped_init.map_or(Ty::NONE, |e| self.expr_ty(e));
        for _ in 0..derefs {
            if let Some(ref mut place) = adjusted_place
                && let Some(e) = unwrapped_init
                && self.place_behind_shared_ref(e)
            {
                place.reborrows_shared_field = true;
                place.behind_shared = true;
                place.mut_ref_root = false;
            }
            if cur == Ty::NONE || matches!(self.cx.types.kind(cur), Kind::Ref(..)) {
                crossed_ref = true;
                if let Some(ref mut place) = adjusted_place {
                    if cur != Ty::NONE && matches!(self.cx.types.kind(cur), Kind::Ref(Mut::Not, _)) {
                        place.reborrows_shared_field = true;
                        place.behind_shared = true;
                        place.mut_ref_root = false;
                    } else if !place.reborrows_shared_field && !place.behind_shared {
                        place.mut_ref_root = true;
                    }
                    place.through_ref = true;
                }
            }
            cur = if cur != Ty::NONE { self.builtin_deref_ty(cur).unwrap_or(Ty::NONE) } else { Ty::NONE };
            push_step(&mut adjusted_steps, PlaceStep::Deref);
        }
        (adjusted_place, adjusted_steps, crossed_ref)
    }

    pub(super) fn is_rest_pat(&self, pat: PatId) -> bool {
        match *self.body.pat(pat) {
            Pat::Rest => true,
            Pat::Bind { sub: Some(sub), .. } => matches!(*self.body.pat(sub), Pat::Rest),
            _ => false,
        }
    }

    fn fallback_or_empty(&mut self, needed: bool, unwrapped_init: Option<ExprId>, fallback_sources: &[BorrowSource], state: &State) -> Vec<BorrowSource> {
        if !needed { return Vec::new(); }
        unwrapped_init.map(|e| self.expr_borrow_sources(e, state)).unwrap_or_else(|| fallback_sources.to_vec())
    }

    fn bind_local_init(
        &mut self, pat: PatId, local: u32, unwrapped_init: Option<ExprId>, scrut_place: Option<&Place>,
        pat_steps: &[PlaceStep], fallback_sources: &[BorrowSource], state: &mut State,
        (move_on_val, leaf_bind, behind_ref, in_drop_adt): (bool, bool, bool, bool),
    ) -> Result<()> {
        if let Some(&ref_mut) = self.typed.ref_binds.get(&pat) {
            let is_mut = ref_mut == Mut::Mut;
            let sources = if let Some(place) = scrut_place {
                let mut full_place = place.clone();
                for s in pat_steps {
                    push_step(&mut full_place.steps, s.clone());
                }
                if is_mut && full_place.behind_shared {
                    bail!("cannot borrow data in a `&` reference or `*const` pointer as mutable");
                }
                if let Some(prefix_len) = full_place.raw_ptr_prefix {
                    self.check_can_read(full_place.local, &full_place.steps[..prefix_len], state)?;
                } else {
                    self.check_borrow(full_place.local, full_place.through_ref, full_place.mut_ref_root, &full_place.steps, is_mut, state)?;
                }
                self.place_to_sources(&full_place, is_mut, state)
            } else {
                let mut sources = match unwrapped_init {
                    Some(e) => self.expr_borrow_sources(e, state),
                    None => fallback_sources.to_vec(),
                };
                if is_mut {
                    for s in &mut sources {
                        s.mutable = true;
                    }
                }
                sources
            };
            self.push_sources_for_local(local, &sources, state);
            return Ok(());
        }
        if let Some(e) = unwrapped_init {
            if let Some(src_local) = self.expr_plain_local(e)
                && self.is_mut_ref_ty(self.local_ty(src_local))
                && self.is_mut_ref_ty(self.local_ty(local))
                && state.loans.iter().any(|l| super::loan_conflicts_borrow(l, src_local, &[]))
            {
                bail!("cannot move out of `{}` because it is borrowed", self.local_name(src_local));
            }
            let sources = self.expr_borrow_sources(e, state);
            self.push_sources_for_local(local, &sources, state);
        } else if !fallback_sources.is_empty() {
            self.push_sources_for_local(local, fallback_sources, state);
        }
        if move_on_val && leaf_bind && !self.is_copy[local as usize] {
            if behind_ref {
                bail!("cannot move out of a shared or mutable reference");
            }
            if in_drop_adt {
                bail!("cannot move out of type which implements the `Drop` trait");
            }
            if let Some(place) = scrut_place {
                if place.through_ref {
                    bail!("cannot move out of `{}` which is behind a reference", self.local_name(place.local));
                }
                let mut full_steps = place.steps.clone();
                for s in pat_steps {
                    push_step(&mut full_steps, s.clone());
                }
                self.check_can_move(place.local, &full_steps, state)?;
                if full_steps.is_empty() {
                    if let Some(slot) = state.moved.get_mut(place.local as usize) {
                        *slot = true;
                    }
                } else {
                    state.partial_moves.push(PartialMove { local: place.local, steps: full_steps });
                }
            }
        }
        Ok(())
    }

    fn bind_positional_pat(&mut self, pat_ids: &[u32], elem_exprs: Option<&[u32]>, scrut_place: Option<&Place>, pat_steps: &[PlaceStep], fallback_sources: &[BorrowSource], state: &mut State, ctx: PatCtx) -> Result<()> {
        let mut saw_rest = false;
        for (idx, &id) in pat_ids.iter().enumerate() {
            let p = PatId(id);
            if matches!(self.body.pat(p), Pat::Rest) { saw_rest = true; continue; }
            let sub_init = elem_exprs.filter(|_| !saw_rest).map(|es| ExprId(es[idx]));
            let mut next_steps = pat_steps.to_vec();
            push_step(&mut next_steps, if !saw_rest { PlaceStep::Field(FieldKey::Tuple(idx as u32)) } else { PlaceStep::Opaque });
            self.bind_pat_ctx(p, sub_init, scrut_place, &next_steps, fallback_sources, state, ctx)?;
        }
        Ok(())
    }

    fn unwrap_parens(&self, mut expr: ExprId) -> ExprId {
        while let Expr::Paren(inner) = *self.body.expr(expr) { expr = inner; }
        expr
    }

    pub(super) fn assign_place(&mut self, lhs: ExprId, sources: &[BorrowSource], state: &mut State, outer_live: &HashSet<u32>, depth: u32) -> Result<()> {
        if self.place_behind_shared_ref(lhs) { bail!("cannot assign to data in a `&` reference"); }
        self.check_static_mut(lhs, false)?;
        match *self.body.expr(lhs) {
            Expr::Paren(inner) => return self.assign_place(inner, sources, state, outer_live, depth),
            Expr::Tuple(list) | Expr::Array(list) => {
                for elem in self.body.list(list).to_vec() { self.assign_place(ExprId(elem), &[], state, outer_live, depth)?; }
                return Ok(());
            }
            Expr::Struct { fields, .. } => {
                for f in self.body.list(fields).to_vec() { self.assign_place(self.body.fields[f as usize].value, &[], state, outer_live, depth)?; }
                return Ok(());
            }
            _ => {}
        }
        if let Some(p) = self.root_place(lhs) {
            self.check_can_mutate(p.local, p.through_ref, p.mut_ref_root, p.behind_shared, &p.steps, state, p.steps.is_empty())?;
            if !p.through_ref && (p.local as usize) < state.init.len() {
                if p.steps.is_empty() {
                    state.init[p.local as usize] = true;
                    state.maybe_init[p.local as usize] = true;
                    state.moved[p.local as usize] = false;
                    state.partial_moves.retain(|m| m.local != p.local);
                    clear_rebound_loans(p.local, state);
                    let rebound: Vec<BorrowSource> = sources.iter().filter(|s| !(s.target == p.local && s.steps.first() == Some(&PlaceStep::Deref))).cloned().collect();
                    self.push_sources_for_local(p.local, &rebound, state);
                } else {
                    if !state.init[p.local as usize] || state.moved[p.local as usize] {
                        bail!("assign to part of moved or uninitialized value `{}`", self.local_name(p.local));
                    }
                    state.partial_moves.retain(|m| !(m.local == p.local && steps_conflict(&m.steps, &p.steps)));
                }
            }
        } else {
            self.check_expr(lhs, state, true, outer_live, false, depth)?;
        }
        Ok(())
    }

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
            && let Some(local) = self.expr_plain_local(ExprId(first))
            && self.is_mut_ref_ty(self.local_ty(local))
        {
            self.check_can_move(local, &[], state)?;
            if let Some(slot) = state.moved.get_mut(local as usize) {
                *slot = true;
            }
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
            Kind::Closure(_, _, at) => self.typed.closures.get(&ExprId(at)).and_then(|c| c.params.get(idx)).map(|&l| self.local_ty(l)),
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
                    for s in self.expr_borrow_sources(receiver, state) {
                        push_loan(&mut state.loans, s.target, s.steps, Some(temp_id), s.mutable && mutability == Mut::Mut, s.through_ref, false, s.env_only);
                    }
                }
            }
        } else {
            let derefs = self.typed.adjusts.get(&expr).map_or(0, |a| a.derefs);
            let sig = self.callee_sig(expr, None);
            if derefs > 0 && sig.as_deref().is_some_and(|s| matches!(s.receiver, Some((crate::body::sig::SelfKind::Value, _)))) {
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
        let mut arg_live = outer_live.clone();
        for &id in &param_ids {
            self.collect_live_expr(ExprId(id), &mut arg_live);
        }
        super::prune_dead_loans(state, &arg_live);
        let sig = self.callee_sig(expr, None);
        let res = self.check_arg_list(&param_ids, sig.as_deref(), None, state, outer_live, depth);
        state.loans.retain(|l| l.holder != Some(temp_id));
        res?;
        if !state.diverged && let Some((local, through_ref, mut_ref_root, steps)) = deferred_recv {
            super::prune_dead_loans(state, outer_live);
            self.check_borrow(local, through_ref, mut_ref_root, &steps, true, state)?;
        }
        Ok(())
    }

    fn check_method_by_val_derefs(&mut self, expr: ExprId, receiver: ExprId, derefs: u32, state: &mut State, outer_live: &HashSet<u32>, depth: u32) -> Result<()> {
        let mut cur = self.expr_ty(receiver);
        let mut crossed_ref = false;
        for step in 0..derefs {
            if let Some(Res::Fn(d_fn, d_args)) = self.typed.derefs.get(&(expr, step)).cloned() {
                crossed_ref = true;
                let args_vec = self.cx.types.substs(d_args).to_vec();
                cur = super::super::infer::detached(self.cx, self.item, |infer| infer.fn_types(d_fn, &args_vec))
                    .ok()
                    .and_then(|(_, ret)| match *self.cx.types.kind(ret) { Kind::Ref(_, inner) => Some(inner), _ => None })
                    .unwrap_or(Ty::NONE);
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

    pub(super) fn check_arg_list(&mut self, arg_ids: &[u32], sig: Option<&Signature>, callee_ty: Option<Ty>, state: &mut State, outer_live: &HashSet<u32>, depth: u32) -> Result<()> {
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
                if state.diverged {
                    break;
                }
                for s in self.expr_borrow_sources(arg, state) {
                    push_loan(&mut state.loans, s.target, s.steps, Some(temp_id), s.mutable && !shared_reborrow, s.through_ref, is_deferred_mut, s.env_only);
                }
                prune_moved_loans(state);
            }
            if !state.diverged && !deferred_mut_args.is_empty() {
                super::prune_dead_loans(state, outer_live);
            }
            if !state.diverged {
                for local in deferred_mut_args {
                    if state.loans.iter().any(|l| !(l.holder == Some(temp_id) && l.two_phase) && super::loan_conflicts_borrow(l, local, &[])) {
                        bail!("cannot use `{}` because it was mutably borrowed", self.local_name(local));
                    }
                }
            }
            Ok(())
        })();
        state.loans.retain(|l| l.holder != Some(temp_id));
        res
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
                if let Some(&local) = closure.params.get(idx)
                    && (local as usize) < body_state.init.len()
                {
                    self.scope_depth[local as usize] = closure_depth;
                    body_state.init[local as usize] = true;
                    body_state.maybe_init[local as usize] = true;
                    body_state.moved[local as usize] = false;
                }
                if let Err(err) = self.bind_pat(self.body.params[p_id as usize].pat, None, None, &[], &[], &mut body_state, (true, true, closure_depth)) {
                    self.no_move_captures = saved_no_move;
                    return Err(err);
                }
            }
        }
        let saved_returns = (self.returns_borrow, self.return_min_depth);
        self.returns_borrow = self.ty_holds_borrow(closure.ret);
        self.return_min_depth = closure_depth;
        let empty = HashSet::default();
        let res = if let Expr::Block { kind, stmts, tail, label } = *self.body.expr(body)
            && matches!(kind, BlockKind::Async | BlockKind::AsyncMove)
        {
            self.check_block(kind, stmts, tail, label, &mut body_state, (false, true), &empty, depth)
        } else {
            self.check_expr(body, &mut body_state, false, &empty, true, closure_depth)
        };
        self.no_move_captures = saved_no_move;
        res?;
        if self.returns_borrow && !body_state.diverged {
            self.check_return_escape(body, &body_state)?;
        }
        (self.returns_borrow, self.return_min_depth) = saved_returns;
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
}

fn push_loan(loans: &mut Vec<Loan>, target: u32, steps: Vec<PlaceStep>, holder: Option<u32>, mutable: bool, through_ref: bool, two_phase: bool, env_only: bool) {
    loans.push(Loan { target, steps, holder, mutable, through_ref, two_phase, env_only });
}

fn clear_rebound_loans(local: u32, state: &mut State) {
    state.loans.retain(|l| l.holder != Some(local) && l.target != local);
}

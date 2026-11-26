use rustc_hash::FxHashSet as HashSet;

use anyhow::{Result, bail};

use crate::body::{Expr, ExprId, Pat, PatId};
use crate::sema::infer::Res;
use crate::sema::ty::{Kind, Mut, Ty};

use super::places::{FieldKey, Place, PlaceStep, push_step, steps_conflict};
use super::propagate::BorrowSource;
use super::{Checker, Loan, PartialMove, State, merge_states};

#[derive(Clone, Copy)]
struct PatCtx {
    init_val: bool,
    move_on_val: bool,
    scope_depth: u32,
    behind_ref: bool,
    in_drop_adt: bool,
    is_param: bool,
    move_only: bool,
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
        let is_shared_target = |target: u32| self.is_shared_ref_or_ptr(self.local_ty(target));
        push_source_loans_filtered(
            &mut state.loans,
            Some(local),
            sources,
            false,
            |s| s.target != local,
            |s| s.mutable && (!s.through_ref || !is_shared_target(s.target)),
        );
    }

    pub(super) fn bind_pat_param(&mut self, pat: PatId, fallback_sources: &[BorrowSource], state: &mut State) -> Result<()> {
        let ctx = PatCtx { init_val: true, move_on_val: true, scope_depth: 0, behind_ref: false, in_drop_adt: false, is_param: true, move_only: false };
        self.bind_pat_ctx(pat, None, None, &[], fallback_sources, state, ctx)
    }

    pub(super) fn bind_pat(
        &mut self, pat: PatId, init: Option<ExprId>, scrut_place: Option<&Place>,
        pat_steps: &[PlaceStep], fallback_sources: &[BorrowSource], state: &mut State, options: (bool, bool, u32),
    ) -> Result<()> {
        let ctx = PatCtx { init_val: options.0, move_on_val: options.1, scope_depth: options.2, behind_ref: false, in_drop_adt: false, is_param: false, move_only: false };
        self.bind_pat_ctx(pat, init, scrut_place, pat_steps, fallback_sources, state, ctx)
    }

    pub(super) fn bind_pat_moves(
        &mut self, pat: PatId, init: Option<ExprId>, scrut_place: Option<&Place>, state: &mut State, scope_depth: u32,
    ) -> Result<()> {
        let ctx = PatCtx { init_val: true, move_on_val: true, scope_depth, behind_ref: false, in_drop_adt: false, is_param: false, move_only: true };
        self.bind_pat_ctx(pat, init, scrut_place, &[], &[], state, ctx)
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
                    if !ctx.move_only {
                        if ctx.is_param && (ctx.behind_ref || !self.typed.ref_binds.contains_key(&pat)) {
                            self.is_param[local as usize] = true;
                        }
                        self.scope_depth[local as usize] = ctx.scope_depth;
                        state.init[local as usize] = ctx.init_val;
                        state.maybe_init[local as usize] = ctx.init_val;
                        state.moved[local as usize] = false;
                        state.partial_moves.retain(|m| m.local != local);
                        clear_rebound_loans(local, state);
                    }
                    if ctx.init_val {
                        let is_leaf = sub.is_none_or(|s| matches!(*self.body.pat(s), Pat::Rest));
                        self.bind_local_init(pat, local, unwrapped_init, scrut_place, pat_steps, fallback_sources, state, (ctx.move_on_val, is_leaf, ctx.behind_ref, ctx.in_drop_adt, ctx.move_only))?;
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
                let (struct_fields, has_base) = unwrapped_init.map_or((None, false), |e| match *self.body.expr(e) {
                    Expr::Struct { fields: ef, base, .. } => (Some(self.body.list(ef).to_vec()), base.is_some()),
                    _ => (None, false),
                });
                let saved_extend = if has_base { self.let_extend_depth.take() } else { None };
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
                    if let Err(err) = self.bind_pat_ctx(fp.pat, f_init, scrut_place, &next_steps, &next_fallback, state, ctx) {
                        if has_base { self.let_extend_depth = saved_extend; }
                        return Err(err);
                    }
                }
                if has_base { self.let_extend_depth = saved_extend; }
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
        unwrapped_init.map(|e| self.expr_borrow_sources_let(e, state)).unwrap_or_else(|| fallback_sources.to_vec())
    }

    fn bind_local_init(
        &mut self, pat: PatId, local: u32, unwrapped_init: Option<ExprId>, scrut_place: Option<&Place>,
        pat_steps: &[PlaceStep], fallback_sources: &[BorrowSource], state: &mut State,
        (move_on_val, leaf_bind, behind_ref, in_drop_adt, move_only): (bool, bool, bool, bool, bool),
    ) -> Result<()> {
        if self.typed.ref_binds.contains_key(&pat) && move_only {
            return Ok(());
        }
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
                let is_ref_init = unwrapped_init.is_some_and(|e| matches!(*self.body.expr(self.unwrap_parens(e)), Expr::Ref { .. }));
                let mut sources = match unwrapped_init {
                    Some(e) => self.place_borrow_sources(e, is_mut, state, self.let_extend_depth),
                    None => fallback_sources
                        .iter()
                        .map(|s| {
                            let mut s = s.clone();
                            if !behind_ref && !self.is_ref_ty(self.local_ty(s.target)) {
                                s.through_ref = false;
                            }
                            s
                        })
                        .collect(),
                };
                if is_mut && !is_ref_init {
                    for s in &mut sources {
                        s.mutable = true;
                    }
                }
                sources
            };
            self.push_sources_for_local(local, &sources, state);
            return Ok(());
        }
        if !move_only {
            if let Some(e) = unwrapped_init {
                self.check_mut_ref_local_move(local, e, state)?;
                let sources = self.expr_borrow_sources_let(e, state);
                self.push_sources_for_local(local, &sources, state);
            } else if !fallback_sources.is_empty() {
                if behind_ref && self.is_copy[local as usize] {
                    let mut inner: Vec<BorrowSource> = fallback_sources
                        .iter()
                        .flat_map(|s| {
                            state.loans.iter().filter(move |l| l.holder == Some(s.target)).map(BorrowSource::from_loan)
                        })
                        .collect();
                    if inner.is_empty() && self.ty_holds_borrow(self.local_ty(local)) {
                        for s in fallback_sources {
                            if self.is_closure_param(s.target) && self.is_ref_ty(self.local_ty(s.target)) {
                                let mut steps = s.steps.clone();
                                push_step(&mut steps, PlaceStep::Deref);
                                inner.push(BorrowSource { steps, ..s.clone() });
                            }
                        }
                    }
                    if !inner.is_empty() {
                        self.push_sources_for_local(local, &inner, state);
                    }
                } else {
                    self.push_sources_for_local(local, fallback_sources, state);
                }
            }
        }
        if move_on_val && leaf_bind && !self.is_copy[local as usize] {
            if behind_ref {
                bail!("cannot move out of a shared or mutable reference");
            }
            if in_drop_adt {
                bail!("cannot move out of type which implements the `Drop` trait");
            }
            if let Some(place) = scrut_place {
                if place.through_ref || place.through_deref {
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
            push_step(&mut next_steps, if !saw_rest { PlaceStep::Field(FieldKey::Tuple(idx as u32)) } else { PlaceStep::Opaque(None) });
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
            let idx_temp = if let Expr::Index(base, _) = *self.body.expr(lhs)
                && self.typed.operators.contains_key(&lhs)
                && let Some(mut bp) = self.root_place(base)
            {
                let autoderefs = self.typed.autoderefs.get(&lhs).copied().unwrap_or(0);
                if autoderefs > 0 {
                    self.apply_deref_steps(lhs, self.expr_ty(base), autoderefs, &mut bp);
                }
                if bp.behind_shared {
                    bail!("cannot borrow data in a `&` reference or `*const` pointer as mutable");
                }
                let tid = self.next_temp_holder();
                if let Some(prefix_len) = bp.raw_ptr_prefix {
                    self.check_can_read(bp.local, &bp.steps[..prefix_len], state)?;
                } else {
                    self.check_borrow(bp.local, bp.through_ref, bp.mut_ref_root, &bp.steps, true, state)?;
                    push_loan(&mut state.loans, bp.local, bp.steps, Some(tid), true, bp.through_ref, false, false);
                }
                Some(tid)
            } else {
                None
            };
            let sub_res = self.check_place_subexprs(lhs, state, outer_live, depth);
            if let Some(tid) = idx_temp {
                state.loans.retain(|l| l.holder != Some(tid));
            }
            sub_res?;
            if p.through_ref || p.through_deref {
                let prefix_len = p.steps.iter().position(|s| matches!(s, PlaceStep::Deref | PlaceStep::Opaque(_))).unwrap_or(0);
                self.check_not_moved(p.local, &p.steps[..prefix_len], state)?;
            }
            let is_union = p.steps.last() == Some(&PlaceStep::Union);
            let union_prefix = if is_union { &p.steps[..p.steps.len() - 1] } else { &p.steps[..] };
            self.check_can_mutate(p.local, p.through_ref, p.mut_ref_root, p.behind_shared, union_prefix, state, union_prefix.is_empty())?;
            self.check_assign_escape(&p, sources, state)?;
            if !p.through_ref && !p.through_deref && (p.local as usize) < state.init.len() {
                if union_prefix.is_empty() {
                    state.init[p.local as usize] = true;
                    state.maybe_init[p.local as usize] = true;
                    state.moved[p.local as usize] = false;
                    state.partial_moves.retain(|m| m.local != p.local);
                    clear_rebound_loans(p.local, state);
                    let rebound: Vec<BorrowSource> = sources.iter().filter(|s| !(s.target == p.local && s.steps.first() == Some(&PlaceStep::Deref))).cloned().collect();
                    self.push_sources_for_local(p.local, &rebound, state);
                } else {
                    if !state.init[p.local as usize]
                        || state.moved[p.local as usize]
                        || state.partial_moves.iter().any(|m| m.local == p.local && m.steps.len() < union_prefix.len() && union_prefix.starts_with(&m.steps))
                    {
                        bail!("assign to part of moved or uninitialized value `{}`", self.local_name(p.local));
                    }
                    state.partial_moves.retain(|m| !(m.local == p.local && steps_conflict(&m.steps, union_prefix)));
                }
            }
        } else {
            self.check_expr(lhs, state, true, outer_live, false, depth)?;
        }
        Ok(())
    }
}

pub(super) fn push_loan(loans: &mut Vec<Loan>, target: u32, steps: Vec<PlaceStep>, holder: Option<u32>, mutable: bool, through_ref: bool, two_phase: bool, env_only: bool) {
    let loan = Loan { target, steps, holder, mutable, through_ref, two_phase, env_only };
    if !loans.contains(&loan) {
        loans.push(loan);
    }
}

pub(super) fn push_source_loans(loans: &mut Vec<Loan>, holder: Option<u32>, sources: &[BorrowSource]) {
    push_source_loans_filtered(loans, holder, sources, false, |_| true, |s| s.mutable);
}

pub(super) fn push_source_loans_filtered(
    loans: &mut Vec<Loan>,
    holder: Option<u32>,
    sources: &[BorrowSource],
    two_phase: bool,
    mut include: impl FnMut(&BorrowSource) -> bool,
    mut mutable_of: impl FnMut(&BorrowSource) -> bool,
) {
    for s in sources {
        if include(s) {
            push_loan(loans, s.target, s.steps.clone(), holder, mutable_of(s), s.through_ref, two_phase, s.env_only);
        }
    }
}

fn clear_rebound_loans(local: u32, state: &mut State) {
    state.loans.retain(|l| l.holder != Some(local) && l.target != local);
}

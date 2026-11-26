mod assign;
mod local;

use anyhow::Result;

use crate::body::{Expr, ExprId, List, Pat, PatId};
use crate::sema::infer::Res;
use crate::sema::ty::{Kind, Mut, Ty};

use super::places::{FieldKey, Place, PlaceStep, push_step};
use super::propagate::BorrowSource;
use super::{Checker, Loan, State, merge_states};

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

#[derive(Clone, Copy)]
struct PatTarget<'a> {
    init: Option<ExprId>,
    scrut_place: Option<&'a Place>,
    pat_steps: &'a [PlaceStep],
    fallback_sources: &'a [BorrowSource],
}

impl Checker<'_, '_> {
    pub(super) fn push_sources_for_local(
        &mut self,
        local: u32,
        sources: &[BorrowSource],
        state: &mut State,
    ) {
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

    pub(super) fn bind_pat_param(
        &mut self,
        pat: PatId,
        fallback_sources: &[BorrowSource],
        state: &mut State,
    ) -> Result<()> {
        let ctx = PatCtx {
            init_val: true,
            move_on_val: true,
            scope_depth: 0,
            behind_ref: false,
            in_drop_adt: false,
            is_param: true,
            move_only: false,
        };
        let target = PatTarget {
            init: None,
            scrut_place: None,
            pat_steps: &[],
            fallback_sources,
        };
        self.bind_pat_ctx(pat, target, state, ctx)
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
        let ctx = PatCtx {
            init_val: options.0,
            move_on_val: options.1,
            scope_depth: options.2,
            behind_ref: false,
            in_drop_adt: false,
            is_param: false,
            move_only: false,
        };
        let target = PatTarget {
            init,
            scrut_place,
            pat_steps,
            fallback_sources,
        };
        self.bind_pat_ctx(pat, target, state, ctx)
    }

    pub(super) fn bind_pat_moves(
        &mut self,
        pat: PatId,
        init: Option<ExprId>,
        scrut_place: Option<&Place>,
        state: &mut State,
        scope_depth: u32,
    ) -> Result<()> {
        let ctx = PatCtx {
            init_val: true,
            move_on_val: true,
            scope_depth,
            behind_ref: false,
            in_drop_adt: false,
            is_param: false,
            move_only: true,
        };
        let target = PatTarget {
            init,
            scrut_place,
            pat_steps: &[],
            fallback_sources: &[],
        };
        self.bind_pat_ctx(pat, target, state, ctx)
    }

    fn bind_pat_ctx(
        &mut self,
        pat: PatId,
        target: PatTarget<'_>,
        state: &mut State,
        mut ctx: PatCtx,
    ) -> Result<()> {
        let unwrapped_init = target.init.map(|e| self.unwrap_parens(e));
        let (adjusted_place, adjusted_steps, crossed_ref) =
            self.apply_pat_derefs(pat, unwrapped_init, target.scrut_place, target.pat_steps);
        if crossed_ref {
            ctx.behind_ref = true;
        }
        let next = PatTarget {
            init: unwrapped_init,
            scrut_place: adjusted_place.as_ref(),
            pat_steps: adjusted_steps.as_slice(),
            fallback_sources: target.fallback_sources,
        };
        match *self.body.pat(pat) {
            Pat::Bind { sub, .. } => self.bind_pat_bind(pat, sub, next, state, ctx),
            Pat::Paren(inner) | Pat::Box(inner) => self.bind_pat_ctx(inner, next, state, ctx),
            Pat::Ref {
                mutable,
                pat: inner,
            } => self.bind_pat_ref(mutable, inner, next, state, ctx),
            Pat::Tuple(list) | Pat::TupleStruct(_, list) => {
                self.bind_pat_tuple(pat, list, next, state, ctx)
            }
            Pat::Slice(list) => self.bind_pat_slice(pat, list, next, state, ctx),
            Pat::Struct { fields, .. } => self.bind_pat_struct(pat, fields, next, state, ctx),
            Pat::Or(list) => self.bind_pat_or(list, next, state, ctx),
            _ => Ok(()),
        }
    }

    fn bind_pat_bind(
        &mut self,
        pat: PatId,
        sub: Option<PatId>,
        target: PatTarget<'_>,
        state: &mut State,
        ctx: PatCtx,
    ) -> Result<()> {
        if let Some(&local) = self.typed.binds.get(&pat)
            && (local as usize) < state.init.len()
        {
            if !ctx.move_only {
                if ctx.is_param && (ctx.behind_ref || !self.typed.ref_binds.contains_key(&pat)) {
                    self.is_param[local as usize] = true;
                }
                self.scope_depth[local as usize] = ctx.scope_depth;
                reset_local_binding(local, ctx.init_val, state);
            }
            if ctx.init_val {
                let is_leaf = sub.is_none_or(|s| matches!(*self.body.pat(s), Pat::Rest));
                self.bind_local_init(pat, local, target, state, ctx, is_leaf)?;
            }
        }
        if let Some(sub) = sub {
            self.bind_pat_ctx(sub, target, state, ctx)?;
        }
        Ok(())
    }

    fn bind_pat_ref(
        &mut self,
        mutable: bool,
        inner: PatId,
        target: PatTarget<'_>,
        state: &mut State,
        mut ctx: PatCtx,
    ) -> Result<()> {
        let inner_init = target.init.and_then(|e| match *self.body.expr(e) {
            Expr::Ref { expr: inner_e, .. } => Some(inner_e),
            _ => None,
        });
        let next_fallback = if inner_init.is_none() {
            target
                .init
                .map(|e| self.expr_borrow_sources(e, state))
                .unwrap_or_else(|| target.fallback_sources.to_vec())
        } else {
            Vec::new()
        };
        let next_place = if let Some(inner_e) = inner_init {
            self.root_place(inner_e)
        } else {
            target.scrut_place.cloned().map(|mut p| {
                if !mutable && !p.reborrows_shared_field {
                    mark_place_behind_shared(&mut p);
                } else if mutable && !p.reborrows_shared_field && !p.behind_shared {
                    p.mut_ref_root = true;
                }
                p.through_ref = true;
                p
            })
        };
        let mut next_steps = target.pat_steps.to_vec();
        if inner_init.is_none() {
            push_step(&mut next_steps, PlaceStep::Deref);
        }
        ctx.behind_ref = true;
        let inner_target = PatTarget {
            init: inner_init,
            scrut_place: next_place.as_ref(),
            pat_steps: &next_steps,
            fallback_sources: &next_fallback,
        };
        self.bind_pat_ctx(inner, inner_target, state, ctx)
    }

    fn bind_pat_tuple(
        &mut self,
        pat: PatId,
        list: List,
        target: PatTarget<'_>,
        state: &mut State,
        mut ctx: PatCtx,
    ) -> Result<()> {
        let is_ctor = matches!(*self.body.pat(pat), Pat::TupleStruct(..));
        if is_ctor
            && self
                .typed
                .pats
                .get(&pat)
                .copied()
                .is_some_and(|ty| self.ty_has_own_drop(ty))
        {
            ctx.in_drop_adt = true;
        }
        let pat_ids: Vec<u32> = self.body.list(list).to_vec();
        let elem_exprs = target.init.and_then(|e| match *self.body.expr(e) {
            Expr::Tuple(elist) if !is_ctor => {
                let es: Vec<u32> = self.body.list(elist).to_vec();
                (es.len() == pat_ids.len()).then_some(es)
            }
            Expr::Call(callee, args)
                if is_ctor && matches!(self.typed.res.get(&callee), Some(Res::Ctor(..))) =>
            {
                let es: Vec<u32> = self.body.list(args).to_vec();
                (es.len() == pat_ids.len()).then_some(es)
            }
            _ => None,
        });
        let next_fallback = self.fallback_or_empty(
            elem_exprs.is_none(),
            target.init,
            target.fallback_sources,
            state,
        );
        let pos_target = PatTarget {
            fallback_sources: &next_fallback,
            ..target
        };
        self.bind_positional_pat(&pat_ids, elem_exprs.as_deref(), pos_target, state, ctx)
    }

    fn bind_pat_slice(
        &mut self,
        pat: PatId,
        list: List,
        target: PatTarget<'_>,
        state: &mut State,
        ctx: PatCtx,
    ) -> Result<()> {
        let steps = self.slice_pat_steps(pat, list);
        let next_fallback =
            self.fallback_or_empty(true, target.init, target.fallback_sources, state);
        for (id, step) in steps {
            let mut next_steps = target.pat_steps.to_vec();
            push_step(&mut next_steps, step);
            let elem_target = PatTarget {
                init: None,
                scrut_place: target.scrut_place,
                pat_steps: &next_steps,
                fallback_sources: &next_fallback,
            };
            self.bind_pat_ctx(id, elem_target, state, ctx)?;
        }
        Ok(())
    }

    fn bind_pat_struct(
        &mut self,
        pat: PatId,
        fields: List,
        target: PatTarget<'_>,
        state: &mut State,
        mut ctx: PatCtx,
    ) -> Result<()> {
        if self
            .typed
            .pats
            .get(&pat)
            .copied()
            .is_some_and(|ty| self.ty_has_own_drop(ty))
        {
            ctx.in_drop_adt = true;
        }
        let field_pat_ids: Vec<u32> = self.body.list(fields).to_vec();
        let (struct_fields, has_base) =
            target
                .init
                .map_or((None, false), |e| match *self.body.expr(e) {
                    Expr::Struct {
                        fields: ef, base, ..
                    } => (Some(self.body.list(ef).to_vec()), base.is_some()),
                    _ => (None, false),
                });
        let saved_extend = if has_base {
            self.let_extend_depth.take()
        } else {
            None
        };
        let next_fallback = self.fallback_or_empty(
            struct_fields.is_none(),
            target.init,
            target.fallback_sources,
            state,
        );
        for fp_id in field_pat_ids {
            let fp = self.body.field_pats[fp_id as usize];
            let fname = self.src.text(fp.name).to_owned();
            let f_init = struct_fields.as_ref().and_then(|ef| {
                ef.iter()
                    .find(|&&id| self.src.text(self.body.fields[id as usize].name) == fname)
                    .map(|&id| self.body.fields[id as usize].value)
            });
            let mut next_steps = target.pat_steps.to_vec();
            push_step(&mut next_steps, PlaceStep::Field(FieldKey::Named(fname)));
            let field_target = PatTarget {
                init: f_init,
                scrut_place: target.scrut_place,
                pat_steps: &next_steps,
                fallback_sources: &next_fallback,
            };
            if let Err(err) = self.bind_pat_ctx(fp.pat, field_target, state, ctx) {
                if has_base {
                    self.let_extend_depth = saved_extend;
                }
                return Err(err);
            }
        }
        if has_base {
            self.let_extend_depth = saved_extend;
        }
        Ok(())
    }

    fn bind_pat_or(
        &mut self,
        list: List,
        target: PatTarget<'_>,
        state: &mut State,
        ctx: PatCtx,
    ) -> Result<()> {
        let base = state.clone();
        let mut merged: Option<State> = None;
        for alt in self.body.list(list).to_vec() {
            let mut alt_state = base.clone();
            self.bind_pat_ctx(PatId(alt), target, &mut alt_state, ctx)?;
            match &mut merged {
                Some(acc) => merge_states(acc, &alt_state),
                None => merged = Some(alt_state),
            }
        }
        if let Some(acc) = merged {
            *state = acc;
        }
        Ok(())
    }

    fn apply_pat_derefs(
        &self,
        pat: PatId,
        unwrapped_init: Option<ExprId>,
        scrut_place: Option<&Place>,
        pat_steps: &[PlaceStep],
    ) -> (Option<Place>, Vec<PlaceStep>, bool) {
        let derefs = self.typed.pat_derefs(pat);
        let mut adjusted_place = scrut_place.cloned();
        let mut adjusted_steps = pat_steps.to_vec();
        if derefs == 0 {
            return (adjusted_place, adjusted_steps, false);
        }
        let mut crossed_ref = false;
        let mut cur = unwrapped_init.map_or(Ty::NONE, |e| self.expr_ty(e));
        for _ in 0..derefs {
            if let Some(ref mut place) = adjusted_place
                && let Some(e) = unwrapped_init
                && self.place_behind_shared_ref(e)
            {
                mark_place_behind_shared(place);
            }
            if cur == Ty::NONE || matches!(self.cx.types.kind(cur), Kind::Ref(..)) {
                crossed_ref = true;
                if let Some(ref mut place) = adjusted_place {
                    if cur != Ty::NONE && matches!(self.cx.types.kind(cur), Kind::Ref(Mut::Not, _))
                    {
                        mark_place_behind_shared(place);
                    } else if !place.reborrows_shared_field && !place.behind_shared {
                        place.mut_ref_root = true;
                    }
                    place.through_ref = true;
                }
            }
            cur = if cur != Ty::NONE {
                self.builtin_deref_ty(cur).unwrap_or(Ty::NONE)
            } else {
                Ty::NONE
            };
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
            .map(|e| self.expr_borrow_sources_let(e, state))
            .unwrap_or_else(|| fallback_sources.to_vec())
    }

    fn bind_positional_pat(
        &mut self,
        pat_ids: &[u32],
        elem_exprs: Option<&[u32]>,
        target: PatTarget<'_>,
        state: &mut State,
        ctx: PatCtx,
    ) -> Result<()> {
        let mut saw_rest = false;
        for (idx, &id) in pat_ids.iter().enumerate() {
            let p = PatId(id);
            if matches!(self.body.pat(p), Pat::Rest) {
                saw_rest = true;
                continue;
            }
            let sub_init = elem_exprs.filter(|_| !saw_rest).map(|es| ExprId(es[idx]));
            let mut next_steps = target.pat_steps.to_vec();
            push_step(
                &mut next_steps,
                if !saw_rest {
                    PlaceStep::Field(FieldKey::Tuple(idx as u32))
                } else {
                    PlaceStep::Opaque(None)
                },
            );
            let elem_target = PatTarget {
                init: sub_init,
                scrut_place: target.scrut_place,
                pat_steps: &next_steps,
                fallback_sources: target.fallback_sources,
            };
            self.bind_pat_ctx(p, elem_target, state, ctx)?;
        }
        Ok(())
    }

    pub(super) fn unwrap_parens(&self, mut expr: ExprId) -> ExprId {
        while let Expr::Paren(inner) = *self.body.expr(expr) {
            expr = inner;
        }
        expr
    }
}

fn mark_place_behind_shared(place: &mut Place) {
    place.reborrows_shared_field = true;
    place.behind_shared = true;
    place.mut_ref_root = false;
}

fn reset_local_binding(local: u32, init_val: bool, state: &mut State) {
    state.init[local as usize] = init_val;
    state.maybe_init[local as usize] = init_val;
    state.moved[local as usize] = false;
    state.partial_moves.retain(|m| m.local != local);
    clear_rebound_loans(local, state);
}

pub(super) fn push_loan(
    loans: &mut Vec<Loan>,
    target: u32,
    steps: Vec<PlaceStep>,
    holder: Option<u32>,
    mutable: bool,
    through_ref: bool,
    two_phase: bool,
    env_only: bool,
) {
    let loan = Loan {
        target,
        steps,
        holder,
        mutable,
        through_ref,
        two_phase,
        env_only,
    };
    if !loans.contains(&loan) {
        loans.push(loan);
    }
}

pub(super) fn push_source_loans(
    loans: &mut Vec<Loan>,
    holder: Option<u32>,
    sources: &[BorrowSource],
) {
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
            push_loan(
                loans,
                s.target,
                s.steps.clone(),
                holder,
                mutable_of(s),
                s.through_ref,
                two_phase,
                s.env_only,
            );
        }
    }
}

fn clear_rebound_loans(local: u32, state: &mut State) {
    state
        .loans
        .retain(|l| l.holder != Some(local) && l.target != local);
}

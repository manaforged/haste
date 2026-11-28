use anyhow::Result;

use crate::body::{Expr, ExprId, PatId};
use crate::sema::ty::{Mut, Ty};

use super::super::places::push_step;
use super::super::propagate::BorrowSource;
use super::super::{Checker, State};
use super::{PatCtx, PatTarget};

impl Checker<'_, '_> {
    pub(in super::super) fn bind_fn_params(
        &mut self,
        sig: &crate::body::sig::Signature,
        state: &mut State,
    ) -> Result<()> {
        let param_offset = usize::from(sig.receiver.is_some());
        for (idx, &pat) in self.body.fn_params.iter().enumerate() {
            let fallback = self
                .typed
                .params
                .get(param_offset + idx)
                .map(|&pl| vec![self.param_fallback_source(pl)]);
            if let crate::body::Pat::Tuple(plist) = *self.body.pat(pat)
                && let Some(&(_, param_ty)) = sig.params.get(idx)
                && let crate::body::Ty::Tuple(tlist) = *sig.body.ty(param_ty)
                && plist.len == tlist.len
            {
                for (&p_id, &t_id) in self.body.list(plist).iter().zip(sig.body.list(tlist)) {
                    let mut lts = super::super::calls::Lifetimes::default();
                    super::super::calls::collect_ty_lifetimes(
                        &sig.body,
                        self.src,
                        crate::body::TyId(t_id),
                        &mut lts,
                    );
                    let elem_fb = if !lts.has_elided && lts.named.is_empty() && lts.saw_static {
                        &[][..]
                    } else {
                        fallback.as_deref().unwrap_or(&[])
                    };
                    self.bind_pat_param(PatId(p_id), elem_fb, state)?;
                }
            } else {
                self.bind_pat_param(pat, fallback.as_deref().unwrap_or(&[]), state)?;
            }
        }
        Ok(())
    }

    pub(super) fn bind_local_init(
        &mut self,
        pat: PatId,
        local: u32,
        target: PatTarget<'_>,
        state: &mut State,
        ctx: PatCtx,
        leaf_bind: bool,
    ) -> Result<()> {
        if self.typed.ref_binds.contains_key(&pat) && ctx.move_only {
            return Ok(());
        }
        if let Some(&ref_mut) = self.typed.ref_binds.get(&pat) {
            return self.bind_ref_local(local, ref_mut == Mut::Mut, target, state, ctx);
        }
        if ctx.move_on_val && leaf_bind && !self.is_copy[local as usize] {
            self.record_pat_move(local, target, state, ctx.behind_ref, ctx.in_drop_adt)?;
        }
        if !ctx.move_only {
            self.bind_val_sources(local, target, state, ctx)?;
        }
        Ok(())
    }

    fn fallback_sources_with_steps(
        &self,
        target: PatTarget<'_>,
        steps_match_fallback: bool,
    ) -> Vec<BorrowSource> {
        let mut sources = target.fallback_sources.to_vec();
        if steps_match_fallback && !target.pat_steps.is_empty() {
            for s in &mut sources {
                for step in target.pat_steps {
                    push_step(&mut s.steps, step.clone());
                }
            }
        }
        sources
    }

    fn bind_ref_local(
        &mut self,
        local: u32,
        is_mut: bool,
        target: PatTarget<'_>,
        state: &mut State,
        ctx: PatCtx,
    ) -> Result<()> {
        let has_field_projection = target.pat_steps.iter().any(|st| {
            matches!(
                st,
                super::super::places::PlaceStep::Field(..)
                    | super::super::places::PlaceStep::SliceElem { .. }
                    | super::super::places::PlaceStep::Opaque(..)
            )
        });
        if ctx.outer_temp_only && has_field_projection {
            self.peelable_ref_field_locals.insert(local);
        }
        let mut sources = if let Some(place) = target.scrut_place {
            let mut full_place = place.clone();
            for s in target.pat_steps {
                push_step(&mut full_place.steps, s.clone());
            }
            if is_mut && full_place.behind_shared {
                return Err(self.borrow_err(
                    None,
                    "cannot borrow data in a `&` reference or `*const` pointer as mutable",
                ));
            }
            if let Some(prefix_len) = full_place.raw_ptr_prefix {
                self.check_can_read(full_place.local, &full_place.steps[..prefix_len], state)?;
            } else {
                self.check_borrow(
                    full_place.local,
                    full_place.through_ref,
                    full_place.mut_ref_root,
                    &full_place.steps,
                    is_mut,
                    state,
                )?;
            }
            if is_mut {
                state.set_tuple_sources(full_place.local, None);
            }
            self.place_to_sources(&full_place, is_mut, state)
        } else {
            let is_ref_init = target.init.is_some_and(|e| {
                matches!(*self.body.expr(self.unwrap_parens(e)), Expr::Ref { .. })
            });
            let mut sources = match target.init {
                Some(e) => self.place_borrow_sources(e, is_mut, state, self.let_extend_depth),
                None => self
                    .fallback_sources_with_steps(target, ctx.steps_match_fallback)
                    .into_iter()
                    .map(|mut s| {
                        if !ctx.behind_ref && !self.is_ref_ty(self.local_ty(s.target)) {
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
        strip_temp_if(&mut sources, ctx.strip_outer_temp);
        self.push_sources_for_local(local, &sources, target.init, state);
        Ok(())
    }

    fn bind_val_sources(
        &mut self,
        local: u32,
        target: PatTarget<'_>,
        state: &mut State,
        ctx: PatCtx,
    ) -> Result<()> {
        if let Some(e) = target.init {
            self.check_mut_ref_local_move(local, e, state)?;
            if self.is_outer_temp_only_ref_expr(e, state) {
                self.outer_temp_only_locals.insert(local);
            } else {
                let unwrapped = self.unwrap_parens(e);
                if let Some(src_local) = self.expr_plain_local(unwrapped)
                    && !self.typed.coercions.contains_key(&unwrapped)
                    && self.peelable_ref_field_locals.contains(&src_local)
                    && self.is_ref_ty(self.local_ty(local))
                {
                    self.peelable_ref_field_locals.insert(local);
                }
            }
            let sources = self.expr_borrow_sources_let(e, state);
            self.push_sources_for_local(local, &sources, Some(e), state);
            let ts = self.expr_tuple_sources(e, state, true);
            state.set_tuple_sources(local, ts);
            if self.is_mut_ptr_ty(self.local_ty(local)) {
                self.record_ptr_targets(local, e, state);
            }
        } else {
            let nested_ts = target
                .elem_tuple
                .filter(|ts| matches!(ts, super::super::propagate::TupleSources::Tuple(_)))
                .cloned();
            state.set_tuple_sources(local, nested_ts);
            if target.fallback_sources.is_empty() {
                return Ok(());
            }
            if ctx.behind_ref
                && self.is_copy[local as usize]
                && let Some(place) = target.scrut_place
            {
                let fallback = self.fallback_sources_with_steps(target, ctx.steps_match_fallback);
                self.bind_copy_behind_ref_place(
                    local,
                    &fallback,
                    place,
                    ctx.strip_outer_temp,
                    state,
                );
            } else {
                let mut sources =
                    self.fallback_sources_with_steps(target, ctx.steps_match_fallback);
                if ctx.behind_ref && self.is_copy[local as usize] && !ctx.is_param {
                    let local_ty = self.local_ty(local);
                    sources =
                        self.unwrap_outer_container_loans_with_res(sources, state, true, local_ty);
                }
                strip_temp_if(&mut sources, ctx.strip_outer_temp);
                self.push_sources_for_local(local, &sources, None, state);
            }
        }
        Ok(())
    }

    fn bind_copy_behind_ref_place(
        &mut self,
        local: u32,
        fallback_sources: &[BorrowSource],
        place: &super::super::places::Place,
        strip_outer_temp: bool,
        state: &mut State,
    ) {
        let local_ty = self.local_ty(local);
        let holds_borrow = self.ty_holds_borrow(local_ty);
        let mut inner = Vec::new();
        for s in fallback_sources {
            let is_deref_container = (s.target == place.local && !place.through_ref)
                || (place.through_ref
                    && (s.target == place.local
                        || state
                            .loans
                            .iter()
                            .any(|l| l.holder == Some(place.local) && l.target == s.target)));
            if !is_deref_container {
                inner.push(s.clone());
                continue;
            }
            for l in state.loans.iter().filter(|l| l.holder == Some(s.target)) {
                inner.push(BorrowSource::from_loan(l));
            }
            if holds_borrow {
                let target_ty = self.local_ty(s.target);
                let is_cparam = self.is_closure_param(s.target);
                if (is_cparam || self.is_non_static_param(s.target))
                    && self.ty_holds_borrow(target_ty)
                    && !self.can_unwrap_inner_static_param(s, local_ty, false)
                {
                    inner.push(self.promote_param_behind_deref(s, is_cparam));
                }
            }
        }
        strip_temp_if(&mut inner, strip_outer_temp);
        if !inner.is_empty() {
            self.push_sources_for_local(local, &inner, None, state);
        }
    }

    fn record_pat_move(
        &self,
        local: u32,
        target: PatTarget<'_>,
        state: &mut State,
        behind_ref: bool,
        in_drop_adt: bool,
    ) -> Result<()> {
        if behind_ref {
            return Err(self.borrow_err(None, "cannot move out of a shared or mutable reference"));
        }
        if in_drop_adt {
            return Err(self.borrow_err(
                None,
                "cannot move out of type which implements the `Drop` trait",
            ));
        }
        let Some(place) = target.scrut_place else {
            return Ok(());
        };
        if place.through_ref || place.through_deref {
            return Err(self.borrow_err(
                None,
                format!(
                    "cannot move out of `{}` which is behind a reference",
                    self.local_name(place.local)
                ),
            ));
        }
        let mut full_steps = place.steps.clone();
        for s in target.pat_steps {
            push_step(&mut full_steps, s.clone());
        }
        self.check_can_move_except(place.local, &full_steps, state, Some(local))?;
        if full_steps.is_empty() {
            self.mark_local_moved(state, place.local, target.init);
        } else {
            self.push_partial_move(state, place.local, full_steps, target.init);
        }
        Ok(())
    }

    pub(super) fn is_outer_temp_only_ref_expr(&mut self, expr: ExprId, state: &State) -> bool {
        let Expr::Ref {
            raw: false,
            expr: inner,
            ..
        } = *self.body.expr(self.unwrap_parens(expr))
        else {
            return false;
        };
        self.is_outer_temp_only_container(inner, state)
    }

    pub(super) fn is_outer_temp_only_expr(&mut self, expr: ExprId, state: &State) -> bool {
        let unwrapped = self.unwrap_parens(expr);
        match *self.body.expr(unwrapped) {
            Expr::Ref {
                raw: false,
                expr: inner,
                ..
            } => self.is_outer_temp_only_container(inner, state),
            _ => self.is_outer_temp_only_container(unwrapped, state),
        }
    }

    fn is_outer_temp_only_container(&mut self, inner: ExprId, state: &State) -> bool {
        let inner_ty = self.expr_ty(inner);
        inner_ty != Ty::NONE
            && !self.is_ref_ty(inner_ty)
            && self.ty_holds_borrow(inner_ty)
            && !self
                .expr_borrow_sources_inner(inner, state, false, None)
                .iter()
                .any(|s| super::super::checks::is_temp_target(s.target).is_some())
    }
}

pub(super) fn mark_place_behind_shared(place: &mut super::super::places::Place) {
    place.reborrows_shared_field = true;
    place.behind_shared = true;
    place.mut_ref_root = false;
}

fn strip_temp_if(sources: &mut Vec<BorrowSource>, strip: bool) {
    if strip {
        sources.retain(|s| super::super::checks::is_temp_target(s.target).is_none());
    }
}

pub(super) fn reset_local_binding(local: u32, init_val: bool, state: &mut State) {
    state.init[local as usize] = init_val;
    state.maybe_init[local as usize] = init_val;
    state.moved[local as usize] = false;
    state.moved_at[local as usize] = None;
    state.partial_moves.retain(|m| m.local != local);
    state.set_tuple_sources(local, None);
    state.ptr_targets.remove(&local);
    clear_rebound_loans(local, state);
}

fn clear_rebound_loans(local: u32, state: &mut State) {
    state.loans.retain_mut(|l| {
        if l.holder == Some(local) {
            return false;
        }
        if l.target == local {
            if l.origin_only || l.through_ref {
                l.origin_only = true;
                return true;
            }
            return false;
        }
        true
    });
}

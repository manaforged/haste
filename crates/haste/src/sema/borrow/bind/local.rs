use anyhow::Result;

use crate::body::{Expr, PatId};
use crate::sema::ty::Mut;

use super::super::places::{PlaceStep, push_step};
use super::super::propagate::BorrowSource;
use super::super::{Checker, State};
use super::{PatCtx, PatTarget};

impl Checker<'_, '_> {
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
            return self.bind_ref_local(local, ref_mut == Mut::Mut, target, state, ctx.behind_ref);
        }
        if !ctx.move_only {
            self.bind_val_sources(local, target, state, ctx.behind_ref)?;
        }
        if ctx.move_on_val && leaf_bind && !self.is_copy[local as usize] {
            self.record_pat_move(target, state, ctx.behind_ref, ctx.in_drop_adt)?;
        }
        Ok(())
    }

    fn bind_ref_local(
        &mut self,
        local: u32,
        is_mut: bool,
        target: PatTarget<'_>,
        state: &mut State,
        behind_ref: bool,
    ) -> Result<()> {
        let sources = if let Some(place) = target.scrut_place {
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
            self.place_to_sources(&full_place, is_mut, state)
        } else {
            let is_ref_init = target.init.is_some_and(|e| {
                matches!(*self.body.expr(self.unwrap_parens(e)), Expr::Ref { .. })
            });
            let mut sources = match target.init {
                Some(e) => self.place_borrow_sources(e, is_mut, state, self.let_extend_depth),
                None => target
                    .fallback_sources
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
        self.push_sources_for_local(local, &sources, target.init, state);
        Ok(())
    }

    fn bind_val_sources(
        &mut self,
        local: u32,
        target: PatTarget<'_>,
        state: &mut State,
        behind_ref: bool,
    ) -> Result<()> {
        if let Some(e) = target.init {
            self.check_mut_ref_local_move(local, e, state)?;
            let sources = self.expr_borrow_sources_let(e, state);
            self.push_sources_for_local(local, &sources, Some(e), state);
        } else if !target.fallback_sources.is_empty() {
            if behind_ref && self.is_copy[local as usize] {
                let mut inner: Vec<BorrowSource> = target
                    .fallback_sources
                    .iter()
                    .flat_map(|s| {
                        state
                            .loans
                            .iter()
                            .filter(move |l| l.holder == Some(s.target))
                            .map(BorrowSource::from_loan)
                    })
                    .collect();
                if inner.is_empty() && self.ty_holds_borrow(self.local_ty(local)) {
                    for s in target.fallback_sources {
                        if self.is_closure_param(s.target)
                            && self.is_ref_ty(self.local_ty(s.target))
                        {
                            let mut steps = s.steps.clone();
                            push_step(&mut steps, PlaceStep::Deref);
                            inner.push(BorrowSource { steps, ..s.clone() });
                        }
                    }
                }
                if !inner.is_empty() {
                    self.push_sources_for_local(local, &inner, None, state);
                }
            } else {
                self.push_sources_for_local(local, target.fallback_sources, None, state);
            }
        }
        Ok(())
    }

    fn record_pat_move(
        &self,
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
        self.check_can_move(place.local, &full_steps, state)?;
        if full_steps.is_empty() {
            self.mark_local_moved(state, place.local, target.init);
        } else {
            self.push_partial_move(state, place.local, full_steps, target.init);
        }
        Ok(())
    }
}

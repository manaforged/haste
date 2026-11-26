use anyhow::Result;
use rustc_hash::FxHashSet as HashSet;

use crate::body::{Expr, ExprId};

use super::super::places::{Place, PlaceStep, steps_conflict};
use super::super::propagate::BorrowSource;
use super::super::{Checker, Loan, State};
use super::{push_loan, reset_local_binding};

impl Checker<'_, '_> {
    pub(in super::super) fn assign_place(
        &mut self,
        lhs: ExprId,
        sources: &[BorrowSource],
        state: &mut State,
        outer_live: &HashSet<u32>,
        depth: u32,
    ) -> Result<()> {
        if self.place_behind_shared_ref(lhs) {
            return Err(self.borrow_err(Some(lhs), "cannot assign to data in a `&` reference"));
        }
        self.check_static_mut(lhs, false)?;
        match *self.body.expr(lhs) {
            Expr::Paren(inner) => {
                return self.assign_place(inner, sources, state, outer_live, depth);
            }
            Expr::Tuple(list) | Expr::Array(list) => {
                for elem in self.body.list(list).to_vec() {
                    self.assign_place(ExprId(elem), &[], state, outer_live, depth)?;
                }
                return Ok(());
            }
            Expr::Struct { fields, .. } => {
                for f in self.body.list(fields).to_vec() {
                    self.assign_place(
                        self.body.fields[f as usize].value,
                        &[],
                        state,
                        outer_live,
                        depth,
                    )?;
                }
                return Ok(());
            }
            _ => {}
        }
        if let Some(p) = self.root_place(lhs) {
            self.assign_root_place(lhs, p, sources, state, outer_live, depth)
        } else {
            if !sources.is_empty()
                && let Expr::Unary(crate::body::UnOp::Deref, ptr_expr) =
                    *self.body.expr(self.unwrap_parens(lhs))
                && self.is_mut_ptr_ty(self.expr_ty(ptr_expr))
            {
                let mut targets = Vec::new();
                self.collect_ptr_targets_expr(ptr_expr, state, &mut targets);
                self.check_expr(lhs, state, true, outer_live, false, depth)?;
                if !targets.is_empty() {
                    self.check_ptr_deref_write(&targets, sources, state)?;
                }
                return Ok(());
            }
            self.check_expr(lhs, state, true, outer_live, false, depth)
        }
    }

    fn assign_root_place(
        &mut self,
        lhs: ExprId,
        p: Place,
        sources: &[BorrowSource],
        state: &mut State,
        outer_live: &HashSet<u32>,
        depth: u32,
    ) -> Result<()> {
        let idx_temp = self.lock_index_base(lhs, state)?;
        let sub_res = self.check_place_subexprs(lhs, state, outer_live, depth);
        if let Some(tid) = idx_temp {
            self.release_temp_holder(state, tid);
        }
        sub_res?;
        if p.through_ref || p.through_deref {
            let prefix_len = p
                .steps
                .iter()
                .position(|s| matches!(s, PlaceStep::Deref | PlaceStep::Opaque(_)))
                .unwrap_or(0);
            self.check_not_moved(p.local, &p.steps[..prefix_len], state)?;
        }
        let is_union = p.steps.last() == Some(&PlaceStep::Union);
        let union_prefix = if is_union {
            &p.steps[..p.steps.len() - 1]
        } else {
            &p.steps[..]
        };
        self.check_can_mutate(
            p.local,
            p.through_ref,
            p.mut_ref_root,
            p.behind_shared,
            union_prefix,
            state,
            union_prefix.is_empty(),
        )?;
        self.check_assign_escape(&p, sources, state)?;
        state.set_tuple_sources(p.local, None);
        if !p.through_ref && !p.through_deref && (p.local as usize) < state.init.len() {
            if union_prefix.is_empty() {
                reset_local_binding(p.local, true, state);
                let rebound: Vec<BorrowSource> = sources
                    .iter()
                    .filter(|s| {
                        !(s.target == p.local && s.steps.first() == Some(&PlaceStep::Deref))
                    })
                    .cloned()
                    .collect();
                self.push_sources_for_local(p.local, &rebound, self.current_expr, state);
            } else {
                if !state.init[p.local as usize]
                    || state.moved[p.local as usize]
                    || state.partial_moves.iter().any(|m| {
                        m.local == p.local
                            && m.steps.len() < union_prefix.len()
                            && union_prefix.starts_with(&m.steps)
                    })
                {
                    return Err(self.borrow_err(
                        Some(lhs),
                        format!(
                            "assign to part of moved or uninitialized value `{}`",
                            self.local_name(p.local)
                        ),
                    ));
                }
                state
                    .partial_moves
                    .retain(|m| !(m.local == p.local && steps_conflict(&m.steps, union_prefix)));
                self.push_sources_for_local(p.local, sources, self.current_expr, state);
            }
        } else if !p.through_ref && (p.local as usize) < state.init.len() {
            self.push_sources_for_local(p.local, sources, self.current_expr, state);
        }
        Ok(())
    }

    fn lock_index_base(&mut self, lhs: ExprId, state: &mut State) -> Result<Option<u32>> {
        let Expr::Index(base, _) = *self.body.expr(lhs) else {
            return Ok(None);
        };
        if !self.typed.operators.contains_key(&lhs) {
            return Ok(None);
        }
        let Some(mut bp) = self.root_place(base) else {
            return Ok(None);
        };
        let autoderefs = self.typed.autoderefs.get(&lhs).copied().unwrap_or(0);
        if autoderefs > 0 {
            self.apply_deref_steps(lhs, self.expr_ty(base), autoderefs, &mut bp);
        }
        if bp.behind_shared {
            return Err(self.borrow_err(
                Some(lhs),
                "cannot borrow data in a `&` reference or `*const` pointer as mutable",
            ));
        }
        let tid = self.next_temp_holder();
        if let Some(prefix_len) = bp.raw_ptr_prefix {
            self.check_can_read(bp.local, &bp.steps[..prefix_len], state)?;
        } else {
            self.check_borrow(
                bp.local,
                bp.through_ref,
                bp.mut_ref_root,
                &bp.steps,
                true,
                state,
            )?;
            push_loan(
                &mut state.loans,
                Loan::new(
                    bp.local,
                    bp.steps,
                    Some(tid),
                    true,
                    bp.through_ref,
                    Some(lhs),
                ),
            );
        }
        Ok(Some(tid))
    }
}

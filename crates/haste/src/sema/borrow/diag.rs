//! Borrow-checker diagnostics and conflict checks with primary and secondary source labels.

use anyhow::Result;

use crate::body::ExprId;
use crate::diag::{Diagnostic, DiagnosticKind, Span};

use super::checks::is_temp_target;
use super::places::{PlaceStep, steps_conflict};
use super::{Checker, State, loan_conflicts_assign, loan_conflicts_borrow};

impl Checker<'_, '_> {
    pub(super) fn span_of(&self, expr: Option<ExprId>) -> Option<Span> {
        expr.and_then(|id| Span::from_expr(self.src, self.body, id))
            .or_else(|| {
                self.current_pat
                    .and_then(|p| Span::from_pat(self.src, self.body, p))
            })
            .or_else(|| {
                self.current_expr
                    .and_then(|id| Span::from_expr(self.src, self.body, id))
            })
    }

    fn diag_with_span(
        &self,
        kind: DiagnosticKind,
        expr: Option<ExprId>,
        msg: impl Into<String>,
    ) -> Diagnostic {
        Diagnostic::new(kind, msg).with_span(self.span_of(expr))
    }

    pub(super) fn borrow_err(&self, expr: Option<ExprId>, msg: impl Into<String>) -> anyhow::Error {
        self.diag_with_span(DiagnosticKind::Borrow, expr, msg)
            .into()
    }

    pub(super) fn unsafe_err(&self, expr: Option<ExprId>, msg: impl Into<String>) -> anyhow::Error {
        self.diag_with_span(DiagnosticKind::Unsafe, expr, msg)
            .with_note("requires an `unsafe` function or block")
            .into()
    }

    pub(super) fn const_err(&self, expr: Option<ExprId>, msg: impl Into<String>) -> anyhow::Error {
        self.diag_with_span(DiagnosticKind::Type, expr, msg).into()
    }

    pub(super) fn does_not_live_long_enough(
        &self,
        expr: Option<ExprId>,
        target: u32,
    ) -> anyhow::Error {
        let msg = if is_temp_target(target).is_some() {
            "temporary value dropped while borrowed".to_owned()
        } else {
            format!("`{}` does not live long enough", self.local_name(target))
        };
        self.borrow_err(expr, msg)
    }

    pub(super) fn borrow_err_labeled(
        &self,
        expr: Option<ExprId>,
        msg: impl Into<String>,
        primary: &str,
        prev: Option<ExprId>,
        secondary: &str,
    ) -> anyhow::Error {
        let mut diag = self
            .diag_with_span(DiagnosticKind::Borrow, expr, msg)
            .with_primary_label(primary);
        if let Some(prev_expr) = prev {
            diag = diag.with_label(Span::from_expr(self.src, self.body, prev_expr), secondary);
        }
        diag.into()
    }

    fn borrow_mut_help(&self, local: u32, msg: String) -> anyhow::Error {
        let name = self.local_name(local);
        self.diag_with_span(DiagnosticKind::Borrow, None, msg)
            .with_help(format!(
                "consider declaring `{name}` as mutable: `mut {name}`"
            ))
            .into()
    }

    pub(super) fn check_not_moved(
        &self,
        local: u32,
        steps: &[PlaceStep],
        state: &State,
    ) -> Result<()> {
        let uninit = !state.init.get(local as usize).copied().unwrap_or(true);
        let partial = state
            .partial_moves
            .iter()
            .find(|m| m.local == local && steps_conflict(&m.steps, steps));
        let moved = state.moved.get(local as usize).copied().unwrap_or(false) || partial.is_some();
        if uninit && !moved {
            return Err(self.borrow_err_labeled(
                None,
                format!(
                    "use of possibly-uninitialized variable `{}`",
                    self.local_name(local)
                ),
                "use of uninitialized value",
                None,
                "",
            ));
        }
        if moved {
            let prev = partial
                .and_then(|pm| pm.site)
                .or_else(|| state.moved_at.get(local as usize).copied().flatten());
            return Err(self.borrow_err_labeled(
                None,
                format!("use of moved value `{}`", self.local_name(local)),
                "value used here after move",
                prev,
                "value moved here",
            ));
        }
        Ok(())
    }

    pub(super) fn check_can_read(
        &self,
        local: u32,
        steps: &[PlaceStep],
        state: &State,
    ) -> Result<()> {
        self.check_can_read_inner(local, steps, state, false)
    }

    pub(super) fn check_can_read_inner(
        &self,
        local: u32,
        steps: &[PlaceStep],
        state: &State,
        allow_shared: bool,
    ) -> Result<()> {
        self.check_not_moved(local, steps, state)?;
        if let Some(loan) = state.loans.iter().find(|l| {
            loan_conflicts_borrow(l, local, steps) && l.mutable && (allow_shared || !l.two_phase)
        }) {
            return Err(self.borrow_err_labeled(
                None,
                format!(
                    "cannot use `{}` because it was mutably borrowed",
                    self.local_name(local)
                ),
                "use of borrowed value occurs here",
                loan.site,
                "mutable borrow occurs here",
            ));
        }
        Ok(())
    }

    pub(super) fn check_can_move(
        &self,
        local: u32,
        steps: &[PlaceStep],
        state: &State,
    ) -> Result<()> {
        self.check_can_move_except(local, steps, state, None)
    }

    pub(super) fn check_can_move_except(
        &self,
        local: u32,
        steps: &[PlaceStep],
        state: &State,
        except_holder: Option<u32>,
    ) -> Result<()> {
        self.check_not_moved(local, steps, state)?;
        if self
            .no_move_captures
            .iter()
            .any(|(l, s)| *l == local && steps_conflict(s, steps))
        {
            return Err(self.borrow_err(
                None,
                format!(
                    "cannot move out of `{}` in a captured closure",
                    self.local_name(local)
                ),
            ));
        }
        if let Some(loan) = state.loans.iter().find(|l| {
            except_holder.is_none_or(|h| l.holder != Some(h))
                && loan_conflicts_borrow(l, local, steps)
        }) {
            return Err(self.borrow_err_labeled(
                None,
                format!(
                    "cannot move out of `{}` because it is borrowed",
                    self.local_name(local)
                ),
                "move out of borrowed value occurs here",
                loan.site,
                "borrow occurs here",
            ));
        }
        Ok(())
    }

    pub(super) fn check_can_mutate(
        &self,
        local: u32,
        through_ref: bool,
        mut_ref_root: bool,
        behind_shared: bool,
        steps: &[PlaceStep],
        state: &State,
        init_ok: bool,
    ) -> Result<()> {
        if behind_shared {
            return Err(self.borrow_err(
                None,
                "cannot assign to data in a `&` reference or `*const` pointer",
            ));
        }
        if !mut_ref_root
            && !self.mutable.get(local as usize).copied().unwrap_or(true)
            && !(init_ok
                && !state
                    .maybe_init
                    .get(local as usize)
                    .copied()
                    .unwrap_or(true))
        {
            return Err(self.borrow_mut_help(
                local,
                format!(
                    "cannot assign to immutable variable `{}`",
                    self.local_name(local)
                ),
            ));
        }
        let is_ref_local = self.is_ref_ty(self.local_ty(local));
        if let Some(loan) = state
            .loans
            .iter()
            .find(|l| loan_conflicts_assign(l, local, through_ref, steps, is_ref_local))
        {
            return Err(self.borrow_err_labeled(
                None,
                format!(
                    "cannot assign to `{}` because it is borrowed",
                    self.local_name(local)
                ),
                "assignment to borrowed value occurs here",
                loan.site,
                "borrow occurs here",
            ));
        }
        Ok(())
    }

    pub(super) fn check_borrow(
        &self,
        local: u32,
        _through_ref: bool,
        mut_ref_root: bool,
        steps: &[PlaceStep],
        mutable: bool,
        state: &State,
    ) -> Result<()> {
        self.check_not_moved(local, steps, state)?;
        if mutable && !mut_ref_root && !self.mutable.get(local as usize).copied().unwrap_or(true) {
            return Err(self.borrow_mut_help(
                local,
                format!(
                    "cannot borrow `{}` as mutable, as it is not declared as mutable",
                    self.local_name(local)
                ),
            ));
        }
        if mutable
            && let Some(loan) = state
                .loans
                .iter()
                .find(|l| loan_conflicts_borrow(l, local, steps) && l.mutable)
        {
            return Err(self.borrow_err_labeled(
                None,
                format!(
                    "cannot borrow `{}` as mutable more than once at a time",
                    self.local_name(local)
                ),
                "second mutable borrow occurs here",
                loan.site,
                "first mutable borrow occurs here",
            ));
        }
        if mutable
            && let Some(loan) = state
                .loans
                .iter()
                .find(|l| loan_conflicts_borrow(l, local, steps))
        {
            return Err(self.borrow_err_labeled(
                None,
                format!(
                    "cannot borrow `{}` as mutable because it is also borrowed as immutable",
                    self.local_name(local)
                ),
                "mutable borrow occurs here",
                loan.site,
                "immutable borrow occurs here",
            ));
        }
        if !mutable
            && let Some(loan) = state
                .loans
                .iter()
                .find(|l| loan_conflicts_borrow(l, local, steps) && l.mutable && !l.two_phase)
        {
            return Err(self.borrow_err_labeled(
                None,
                format!(
                    "cannot borrow `{}` as immutable because it is also borrowed as mutable",
                    self.local_name(local)
                ),
                "immutable borrow occurs here",
                loan.site,
                "mutable borrow occurs here",
            ));
        }
        Ok(())
    }
}

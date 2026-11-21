use anyhow::{Result, bail};
use rustc_hash::FxHashSet as HashSet;

use crate::body::{BinOp, BlockKind, Expr, ExprId, List, Pat, PatId, Stmt};

use super::{BreakFrame, Checker, Loan, State, is_temp_holder, merge_branch, merge_states, prune_dead_loans, prune_moved_loans};

impl Checker<'_, '_> {
    pub(super) fn check_block(
        &mut self,
        kind: BlockKind,
        stmts: List,
        tail: Option<ExprId>,
        label: Option<u32>,
        state: &mut State,
        (place_ctx, value_used): (bool, bool),
        outer_live: &HashSet<u32>,
        depth: u32,
    ) -> Result<()> {
        let block_depth = depth + 1;
        let saved_stmt_depth = self.stmt_depth;
        let update_stmt_depth = !matches!(kind, BlockKind::Expansion | BlockKind::Unsafe);
        if label.is_some() {
            self.breaks.push(BreakFrame {
                label,
                is_block: true,
                depth: block_depth,
                states: Vec::new(),
            });
        }
        let ids: Vec<u32> = self.body.list(stmts).to_vec();
        let (before_stmt, after_stmt, tail_live) = self.block_liveness(&ids, tail, outer_live);
        for (pos, &stmt_id) in ids.iter().enumerate() {
            if state.diverged {
                break;
            }
            prune_dead_loans(state, &before_stmt[pos]);
            match self.body.stmts[stmt_id as usize] {
                Stmt::Let { pat, init, otherwise, extend, .. } => {
                    if update_stmt_depth && !extend {
                        self.stmt_depth = block_depth;
                    }
                    let let_stmt_depth = self.stmt_depth;
                    let is_wild = matches!(*self.body.pat(pat), Pat::Wild);
                    let destructures = !matches!(
                        *self.body.pat(pat),
                        Pat::Bind { by_ref: false, sub: None, .. } | Pat::Wild
                    );
                    if let Some(e) = init {
                        self.check_expr(e, state, destructures || is_wild, &after_stmt[pos], !is_wild, block_depth)?;
                    }
                    if let Some(e) = otherwise {
                        let mut alt = state.clone();
                        self.check_expr(e, &mut alt, false, &after_stmt[pos], false, block_depth)?;
                    }
                    self.stmt_depth = let_stmt_depth;
                    let pat_depth = if extend { let_stmt_depth } else { block_depth };
                    let scrut_place = init.and_then(|e| self.root_place(e));
                    self.bind_pat(
                        pat,
                        init,
                        scrut_place.as_ref(),
                        &[],
                        &[],
                        state,
                        (init.is_some(), destructures, pat_depth),
                    )?;
                    prune_moved_loans(state);
                }
                Stmt::Expr(e, semi) => {
                    if update_stmt_depth {
                        self.stmt_depth = block_depth;
                    }
                    self.check_expr(e, state, false, &after_stmt[pos], !semi, block_depth)?;
                    if semi {
                        prune_moved_loans(state);
                    }
                }
                Stmt::Item(..) | Stmt::Macro { .. } | Stmt::Empty => {}
            }
        }
        if let Some(t) = tail
            && !state.diverged
        {
            if update_stmt_depth {
                self.stmt_depth = block_depth;
            }
            prune_dead_loans(state, &tail_live);
            self.check_expr(t, state, place_ctx, outer_live, value_used, block_depth)?;
            if value_used && !state.diverged {
                for src in self.expr_borrow_sources(t, state) {
                    if self.escapes_depth(src.target, src.through_ref, block_depth) {
                        bail!("`{}` does not live long enough", self.local_name(src.target));
                    }
                }
            }
        }
        self.stmt_depth = saved_stmt_depth;
        if !state.diverged {
            self.check_block_escape(block_depth, state, outer_live)?;
            prune_dead_loans(state, outer_live);
        }
        if label.is_some()
            && let Some(frame) = self.breaks.pop()
            && !frame.states.is_empty()
        {
            let mut merged: Option<State> = (!state.diverged).then(|| state.clone());
            for bs in frame.states {
                match &mut merged {
                    Some(acc) => merge_states(acc, &bs),
                    None => merged = Some(bs),
                }
            }
            if let Some(acc) = merged {
                *state = acc;
            }
        }
        Ok(())
    }

    pub(super) fn check_if(
        &mut self,
        cond: ExprId,
        then: ExprId,
        otherwise: Option<ExprId>,
        state: &mut State,
        (place_ctx, value_used): (bool, bool),
        outer_live: &HashSet<u32>,
        depth: u32,
    ) -> Result<()> {
        let mut then_live = outer_live.clone();
        self.collect_live_expr(then, &mut then_live);
        let mut else_live = outer_live.clone();
        if let Some(alt) = otherwise {
            self.collect_live_expr(alt, &mut else_live);
        }
        let mut then_state = state.clone();
        let mut else_state = state.clone();
        self.check_cond_for_branches(cond, &mut then_state, &mut else_state, &then_live, &else_live, depth + 1)?;
        prune_dead_loans(&mut then_state, &then_live);
        self.check_expr(then, &mut then_state, place_ctx, outer_live, value_used, depth)?;
        if let Some(alt) = otherwise {
            prune_dead_loans(&mut else_state, &else_live);
            self.check_expr(alt, &mut else_state, place_ctx, outer_live, value_used, depth)?;
        }
        merge_branch(state, &then_state, &else_state);
        Ok(())
    }

    pub(super) fn check_cond_for_branches(
        &mut self,
        cond: ExprId,
        then_state: &mut State,
        else_state: &mut State,
        then_live: &HashSet<u32>,
        else_live: &HashSet<u32>,
        scope_depth: u32,
    ) -> Result<()> {
        match *self.body.expr(cond) {
            Expr::Paren(inner) => {
                self.check_cond_for_branches(inner, then_state, else_state, then_live, else_live, scope_depth)
            }
            Expr::Let(pat, init) => {
                self.check_expr(init, then_state, true, then_live, true, scope_depth)?;
                let scrut_place = self.root_place(init);
                self.bind_pat(pat, Some(init), scrut_place.as_ref(), &[], &[], then_state, (true, true, scope_depth))?;
                self.check_expr(init, else_state, true, else_live, false, scope_depth)
            }
            Expr::Binary(BinOp::And, a, b) => {
                self.check_cond_for_branches(a, then_state, else_state, then_live, else_live, scope_depth)?;
                let mut dummy_else = then_state.clone();
                self.check_cond_for_branches(b, then_state, &mut dummy_else, then_live, else_live, scope_depth)
            }
            _ => {
                self.check_expr(cond, then_state, false, then_live, false, scope_depth)?;
                *else_state = then_state.clone();
                Ok(())
            }
        }
    }

    pub(super) fn check_match(
        &mut self,
        scrut: ExprId,
        arms: List,
        state: &mut State,
        (place_ctx, value_used): (bool, bool),
        outer_live: &HashSet<u32>,
        depth: u32,
    ) -> Result<()> {
        let arm_ids: Vec<u32> = self.body.list(arms).to_vec();
        let mut match_live = outer_live.clone();
        for &arm_id in &arm_ids {
            let arm = self.body.arms[arm_id as usize];
            self.collect_live_expr(arm.body, &mut match_live);
            if let Some(g) = arm.guard {
                self.collect_live_expr(g, &mut match_live);
            }
        }
        self.check_expr(scrut, state, true, &match_live, true, depth)?;
        let scrut_place = self.root_place(scrut);
        let arm_depth = depth + 1;
        let mut merged: Option<State> = None;
        for arm_id in arm_ids {
            let arm = self.body.arms[arm_id as usize];
            let mut arm_state = state.clone();
            self.bind_pat(
                arm.pat,
                Some(scrut),
                scrut_place.as_ref(),
                &[],
                &[],
                &mut arm_state,
                (true, true, arm_depth),
            )?;
            let mut arm_live = outer_live.clone();
            self.collect_live_expr(arm.body, &mut arm_live);
            if let Some(g) = arm.guard {
                self.collect_live_expr(g, &mut arm_live);
            }
            prune_dead_loans(&mut arm_state, &arm_live);
            if let Some(g) = arm.guard {
                let mut guard_live = outer_live.clone();
                self.collect_live_expr(arm.body, &mut guard_live);
                self.check_expr(g, &mut arm_state, false, &guard_live, false, arm_depth)?;
            }
            self.check_expr(arm.body, &mut arm_state, place_ctx, outer_live, value_used, arm_depth)?;
            if !arm_state.diverged {
                self.check_block_escape(arm_depth, &arm_state, outer_live)?;
                prune_dead_loans(&mut arm_state, outer_live);
                match &mut merged {
                    Some(acc) => merge_states(acc, &arm_state),
                    None => merged = Some(arm_state),
                }
            }
        }
        if let Some(acc) = merged {
            *state = acc;
        } else {
            state.diverged = true;
        }
        Ok(())
    }

    pub(super) fn check_for(
        &mut self,
        label: Option<u32>,
        pat: PatId,
        iter: ExprId,
        body: ExprId,
        state: &mut State,
        outer_live: &HashSet<u32>,
        depth: u32,
    ) -> Result<()> {
        let mut loop_live = outer_live.clone();
        self.collect_live_expr(body, &mut loop_live);
        self.check_expr(iter, state, false, &loop_live, true, depth)?;
        let iter_sources = self.expr_borrow_sources(iter, state);
        let iter_temp = if iter_sources.is_empty() {
            None
        } else {
            let id = self.next_iter_holder();
            for s in iter_sources {
                state.loans.push(Loan {
                    target: s.target,
                    steps: s.steps,
                    holder: Some(id),
                    mutable: s.mutable,
                    through_ref: s.through_ref,
                    two_phase: false,
                    env_only: s.env_only,
                });
            }
            Some(id)
        };
        self.check_loop(label, None, Some(pat), iter_temp, body, state, outer_live, depth)?;
        if let Some(id) = iter_temp {
            state.loans.retain(|l| l.holder != Some(id));
        }
        Ok(())
    }

    pub(super) fn check_loop(
        &mut self,
        label: Option<u32>,
        cond: Option<ExprId>,
        pat: Option<PatId>,
        iter_holder: Option<u32>,
        body: ExprId,
        state: &mut State,
        outer_live: &HashSet<u32>,
        depth: u32,
    ) -> Result<()> {
        let loop_depth = depth + 1;
        let infinite = cond.is_none() && pat.is_none();
        let mut loop_locals = HashSet::default();
        if let Some(c) = cond {
            self.collect_declared_locals_expr(c, &mut loop_locals);
        }
        if let Some(p) = pat {
            self.collect_pat_locals(p, &mut loop_locals);
        }
        self.collect_declared_locals_expr(body, &mut loop_locals);
        let mut loop_live = outer_live.clone();
        if let Some(c) = cond {
            self.collect_live_expr(c, &mut loop_live);
        }
        self.collect_live_expr(body, &mut loop_live);
        for l in &loop_locals {
            loop_live.remove(l);
        }
        if let Some(h) = iter_holder {
            loop_live.insert(h);
        }
        self.breaks.push(BreakFrame {
            label,
            is_block: false,
            depth: loop_depth,
            states: Vec::new(),
        });
        let mut iter_state = state.clone();
        for _ in 0..2 {
            iter_state.diverged = false;
            if let Some(c) = cond {
                let mut cond_dummy = iter_state.clone();
                self.check_cond_for_branches(c, &mut iter_state, &mut cond_dummy, &loop_live, outer_live, loop_depth)?;
            }
            if let Some(p) = pat {
                self.bind_pat(p, None, None, &[], &[], &mut iter_state, (true, false, loop_depth))?;
            }
            self.check_expr(body, &mut iter_state, false, &loop_live, false, loop_depth)?;
            iter_state
                .loans
                .retain(|l| l.holder.is_none_or(|h| is_temp_holder(Some(h)) || !loop_locals.contains(&h)));
            if iter_state.diverged {
                break;
            }
            self.check_block_escape(loop_depth, &iter_state, outer_live)?;
        }
        let frame = self.breaks.pop().expect("loop pushed break frame");
        if infinite {
            let mut merged: Option<State> = None;
            for bs in frame.states {
                match &mut merged {
                    Some(acc) => merge_states(acc, &bs),
                    None => merged = Some(bs),
                }
            }
            match merged {
                Some(acc) => *state = acc,
                None => state.diverged = true,
            }
        } else {
            if !iter_state.diverged {
                merge_states(state, &iter_state);
            }
            for bs in frame.states {
                merge_states(state, &bs);
            }
        }
        state
            .loans
            .retain(|l| l.holder.is_none_or(|h| is_temp_holder(Some(h)) || !loop_locals.contains(&h)));
        Ok(())
    }
}

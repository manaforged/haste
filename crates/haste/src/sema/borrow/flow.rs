use anyhow::Result;
use rustc_hash::FxHashSet as HashSet;

use crate::body::{BinOp, BlockKind, Expr, ExprId, List, Pat, PatId, Stmt};

use super::bind::{push_loan, push_source_loans};
use super::checks::{is_temp_target, merge_break_states, retain_non_loop_state};
use super::places::PlaceStep;
use super::propagate::BorrowSource;
use super::{
    BreakFrame, Checker, Loan, STMT_TEMP_OFFSET, State, join_moved, merge_branch, merge_states,
    prune_dead_loans, prune_moved_loans,
};

impl Checker<'_, '_> {
    pub(super) fn check_block(
        &mut self,
        expr_id: ExprId,
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
        let is_const = kind == BlockKind::Const;
        let is_unsafe = kind == BlockKind::Unsafe;
        if is_unsafe {
            self.unsafe_depth += 1;
        }
        if is_const {
            self.check_const_block_locals(stmts, tail, block_depth)?;
            self.const_depth += 1;
        }
        let update_stmt_depth = !matches!(kind, BlockKind::Expansion | BlockKind::Unsafe);
        if label.is_some() {
            self.breaks.push(BreakFrame {
                label,
                is_block: true,
                depth: block_depth,
                states: Vec::new(),
                continues: Vec::new(),
                sources: Vec::new(),
                break_live: outer_live.clone(),
                continue_live: None,
            });
        }
        let ids: Vec<u32> = self.body.list(stmts).to_vec();
        let (before_stmt, after_stmt, tail_live) = self.block_liveness(&ids, tail, outer_live);
        self.with_let_extend(None, |this| -> Result<()> {
            for (pos, &stmt_id) in ids.iter().enumerate() {
                if state.diverged {
                    break;
                }
                prune_dead_loans(state, &before_stmt[pos]);
                this.check_block_stmt(
                    stmt_id,
                    state,
                    &after_stmt[pos],
                    block_depth,
                    update_stmt_depth,
                )?;
            }
            Ok(())
        })?;
        if let Some(t) = tail
            && !state.diverged
        {
            self.check_block_tail(
                expr_id,
                t,
                state,
                (place_ctx, value_used),
                (outer_live, &tail_live),
                (block_depth, update_stmt_depth, is_const),
            )?;
        }
        if is_const {
            self.const_depth -= 1;
        }
        if is_unsafe {
            self.unsafe_depth -= 1;
        }
        self.stmt_depth = saved_stmt_depth;
        if !state.diverged {
            self.check_block_escape(block_depth, state, outer_live)?;
            prune_dead_loans(state, outer_live);
        }
        if label.is_some()
            && let Some(frame) = self.breaks.pop()
        {
            if !frame.sources.is_empty() && !is_const {
                let mut merged = self.temp_sources.remove(&expr_id).unwrap_or_default();
                merged.extend(frame.sources);
                self.temp_sources
                    .insert(expr_id, super::propagate::dedup_sources(merged));
            }
            if !frame.states.is_empty()
                && let Some(acc) =
                    merge_break_states((!state.diverged).then(|| state.clone()), frame.states)
            {
                *state = acc;
            }
        }
        Ok(())
    }

    fn check_block_stmt(
        &mut self,
        stmt_id: u32,
        state: &mut State,
        after_live: &HashSet<u32>,
        block_depth: u32,
        update_stmt_depth: bool,
    ) -> Result<()> {
        match self.body.stmts[stmt_id as usize] {
            Stmt::Let {
                pat,
                ty,
                init,
                otherwise,
                extend,
                ..
            } => self.check_let_stmt(
                (pat, ty, init, otherwise, extend),
                state,
                after_live,
                block_depth,
                update_stmt_depth,
            ),
            Stmt::Expr(e, semi) => {
                self.check_expr_stmt(e, semi, state, after_live, block_depth, update_stmt_depth)
            }
            Stmt::Item(..) | Stmt::Macro { .. } | Stmt::Empty => Ok(()),
        }
    }

    fn check_let_stmt(
        &mut self,
        (pat, ann_ty, init, otherwise, extend): (
            PatId,
            Option<crate::body::TyId>,
            Option<ExprId>,
            Option<ExprId>,
            bool,
        ),
        state: &mut State,
        after_live: &HashSet<u32>,
        block_depth: u32,
        update_stmt_depth: bool,
    ) -> Result<()> {
        if update_stmt_depth && !extend {
            self.stmt_depth = block_depth;
        }
        let let_stmt_depth = self.stmt_depth;
        let saved_temp_scope = self.temp_scope_depth;
        let stmt_temp_depth = block_depth + STMT_TEMP_OFFSET;
        if init.is_some() {
            self.temp_scope_depth = if extend {
                let_stmt_depth
            } else {
                stmt_temp_depth
            };
        }
        let is_wild = matches!(*self.body.pat(pat), Pat::Wild);
        let destructures = !matches!(
            *self.body.pat(pat),
            Pat::Bind {
                by_ref: false,
                sub: None,
                ..
            } | Pat::Wild
        );
        let pat_depth = if extend { let_stmt_depth } else { block_depth };
        if let Some(e) = init {
            self.with_let_extend((!is_wild).then_some(pat_depth), |inner| {
                inner.check_let_init(
                    pat,
                    ann_ty,
                    e,
                    (destructures || is_wild, !is_wild),
                    state,
                    after_live,
                    block_depth,
                )
            })?;
        }
        if let Some(e) = otherwise {
            let mut alt = state.clone();
            self.with_let_extend(None, |inner| {
                inner.check_expr(e, &mut alt, false, after_live, false, block_depth)
            })?;
        }
        self.stmt_depth = let_stmt_depth;
        let scrut_place = init.and_then(|e| self.root_place(e));
        let bind_res = self.with_let_extend(init.map(|_| pat_depth), |inner| {
            inner.bind_pat(
                pat,
                init,
                scrut_place.as_ref(),
                &[],
                &[],
                state,
                (init.is_some(), destructures, pat_depth),
            )
        });
        self.temp_scope_depth = saved_temp_scope;
        bind_res?;
        let mut bound = HashSet::default();
        self.collect_pat_locals(pat, &mut bound);
        let bound_vec: Vec<u32> = bound.into_iter().collect();
        self.check_stmt_temp_escape(state, stmt_temp_depth, &bound_vec, after_live)?;
        prune_moved_loans(state);
        Ok(())
    }

    fn check_expr_stmt(
        &mut self,
        expr: ExprId,
        semi: bool,
        state: &mut State,
        after_live: &HashSet<u32>,
        block_depth: u32,
        update_stmt_depth: bool,
    ) -> Result<()> {
        if update_stmt_depth {
            self.stmt_depth = block_depth;
        }
        let saved_temp_scope = self.temp_scope_depth;
        let stmt_temp_depth = block_depth + STMT_TEMP_OFFSET;
        if semi {
            self.temp_scope_depth = stmt_temp_depth;
        }
        self.check_expr(expr, state, false, after_live, !semi, block_depth)?;
        self.temp_scope_depth = saved_temp_scope;
        if semi {
            self.check_stmt_temp_escape(state, stmt_temp_depth, &[], after_live)?;
            prune_moved_loans(state);
        }
        Ok(())
    }

    fn check_block_tail(
        &mut self,
        expr_id: ExprId,
        tail: ExprId,
        state: &mut State,
        (place_ctx, value_used): (bool, bool),
        (outer_live, tail_live): (&HashSet<u32>, &HashSet<u32>),
        (block_depth, update_stmt_depth, is_const): (u32, bool, bool),
    ) -> Result<()> {
        if update_stmt_depth {
            self.stmt_depth = block_depth;
        }
        prune_dead_loans(state, tail_live);
        self.check_expr(tail, state, place_ctx, outer_live, value_used, block_depth)?;
        if (value_used || is_const) && !state.diverged {
            let tail_srcs = self.expr_borrow_sources_let(tail, state);
            for src in &tail_srcs {
                if (is_const
                    && is_temp_target(src.target).is_some()
                    && !self.temp_target_allowed(src.mutable))
                    || (value_used
                        && self.escapes_depth(
                            src.target,
                            src.through_ref,
                            src.mutable,
                            block_depth,
                        ))
                {
                    return Err(self.does_not_live_long_enough(Some(tail), src.target));
                }
            }
            if value_used && !tail_srcs.is_empty() && !is_const {
                self.temp_sources.insert(expr_id, tail_srcs);
            }
        }
        Ok(())
    }

    pub(super) fn check_if(
        &mut self,
        expr_id: ExprId,
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
        let mut drop_temps = Vec::new();
        self.with_let_extend(None, |this| {
            this.check_cond_for_branches(
                cond,
                &mut then_state,
                &mut else_state,
                &then_live,
                &else_live,
                depth + 1,
                &mut drop_temps,
            )
        })?;
        prune_dead_loans(&mut then_state, &then_live);
        self.check_expr(
            then,
            &mut then_state,
            place_ctx,
            outer_live,
            value_used,
            depth,
        )?;
        let mut out_sources = Vec::new();
        if value_used && !then_state.diverged {
            out_sources.extend(self.expr_borrow_sources_let(then, &then_state));
        }
        for id in &drop_temps {
            then_state.loans.retain(|l| l.holder != Some(*id));
            self.release_temp_holder(&mut else_state, *id);
        }
        if let Some(alt) = otherwise {
            prune_dead_loans(&mut else_state, &else_live);
            self.check_expr(
                alt,
                &mut else_state,
                place_ctx,
                outer_live,
                value_used,
                depth,
            )?;
            if value_used && !else_state.diverged {
                for s in self.expr_borrow_sources_let(alt, &else_state) {
                    if !out_sources.contains(&s) {
                        out_sources.push(s);
                    }
                }
            }
        }
        if value_used && !out_sources.is_empty() {
            self.temp_sources.insert(expr_id, out_sources);
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
        drop_temps: &mut Vec<u32>,
    ) -> Result<()> {
        match *self.body.expr(cond) {
            Expr::Paren(inner) => self.check_cond_for_branches(
                inner,
                then_state,
                else_state,
                then_live,
                else_live,
                scope_depth,
                drop_temps,
            ),
            Expr::Let(pat, init) => {
                let mut both_live = then_live.clone();
                both_live.extend(else_live);
                self.check_expr(init, then_state, true, &both_live, true, scope_depth)?;
                *else_state = then_state.clone();
                if let Some(temp_id) = self.hold_drop_temp(init, then_state) {
                    drop_temps.push(temp_id);
                }
                let scrut_place = self.root_place(init);
                self.bind_pat(
                    pat,
                    Some(init),
                    scrut_place.as_ref(),
                    &[],
                    &[],
                    then_state,
                    (true, true, scope_depth),
                )
            }
            Expr::Binary(BinOp::And, a, b) => {
                self.check_cond_for_branches(
                    a,
                    then_state,
                    else_state,
                    then_live,
                    else_live,
                    scope_depth,
                    drop_temps,
                )?;
                let mut rhs_else = then_state.clone();
                self.check_cond_for_branches(
                    b,
                    then_state,
                    &mut rhs_else,
                    then_live,
                    else_live,
                    scope_depth,
                    drop_temps,
                )?;
                if !rhs_else.diverged && !else_state.diverged {
                    merge_states(else_state, &rhs_else);
                } else if else_state.diverged && !rhs_else.diverged {
                    *else_state = rhs_else;
                }
                Ok(())
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
        expr_id: ExprId,
        scrut: ExprId,
        arms: List,
        state: &mut State,
        (place_ctx, value_used): (bool, bool),
        outer_live: &HashSet<u32>,
        depth: u32,
    ) -> Result<()> {
        let arm_ids: Vec<u32> = self.body.list(arms).to_vec();
        let mut match_live = outer_live.clone();
        self.collect_match_arms_live(&arm_ids, outer_live, &mut match_live);
        self.with_let_extend(None, |this| {
            this.check_expr(scrut, state, true, &match_live, true, depth)
        })?;
        let scrut_temp = self.hold_drop_temp(scrut, state);
        let scrut_place = self.root_place(scrut);
        let arm_depth = depth + 1;
        let mut fallthrough_state = state.clone();
        let mut merged: Option<State> = None;
        let mut out_sources = Vec::new();
        for arm_id in arm_ids {
            let arm = self.body.arms[arm_id as usize];
            let mut arm_state = fallthrough_state.clone();
            self.bind_pat(
                arm.pat,
                Some(scrut),
                scrut_place.as_ref(),
                &[],
                &[],
                &mut arm_state,
                (true, arm.guard.is_none(), arm_depth),
            )?;
            let mut arm_live = outer_live.clone();
            self.collect_live_expr(arm.body, &mut arm_live);
            if let Some(g) = arm.guard {
                self.collect_live_expr(g, &mut arm_live);
            }
            prune_dead_loans(&mut arm_state, &arm_live);
            if let Some(g) = arm.guard {
                let guard_temp = if let Some(ref p) = scrut_place
                    && p.raw_ptr_prefix.is_none()
                    && self.pat_inspects_scrut(arm.pat)
                {
                    let id = self.next_temp_holder();
                    push_loan(
                        &mut arm_state.loans,
                        Loan::new(
                            p.local,
                            p.steps.clone(),
                            Some(id),
                            false,
                            p.through_ref,
                            Some(scrut),
                        ),
                    );
                    Some(id)
                } else {
                    None
                };
                let mut guard_live = outer_live.clone();
                self.collect_live_expr(arm.body, &mut guard_live);
                let mut guard_else = arm_state.clone();
                let mut g_temps = Vec::new();
                self.with_let_extend(None, |this| {
                    this.check_cond_for_branches(
                        g,
                        &mut arm_state,
                        &mut guard_else,
                        &guard_live,
                        &match_live,
                        arm_depth,
                        &mut g_temps,
                    )
                })?;
                if let Some(gt) = guard_temp {
                    arm_state.loans.retain(|l| l.holder != Some(gt));
                    self.release_temp_holder(&mut guard_else, gt);
                }
                if !guard_else.diverged {
                    join_moved(&mut fallthrough_state, &guard_else);
                }
                if !arm_state.diverged {
                    self.bind_pat_moves(
                        arm.pat,
                        Some(scrut),
                        scrut_place.as_ref(),
                        &mut arm_state,
                        arm_depth,
                    )?;
                }
                self.check_expr(
                    arm.body,
                    &mut arm_state,
                    place_ctx,
                    outer_live,
                    value_used,
                    arm_depth,
                )?;
                for id in g_temps {
                    self.release_temp_holder(&mut arm_state, id);
                }
            } else {
                self.check_expr(
                    arm.body,
                    &mut arm_state,
                    place_ctx,
                    outer_live,
                    value_used,
                    arm_depth,
                )?;
            }
            if !arm_state.diverged {
                if value_used {
                    for s in self.expr_borrow_sources_let(arm.body, &arm_state) {
                        if self.escapes_depth(s.target, s.through_ref, s.mutable, arm_depth) {
                            return Err(self.does_not_live_long_enough(Some(arm.body), s.target));
                        }
                        if !out_sources.contains(&s) {
                            out_sources.push(s);
                        }
                    }
                }
                self.check_block_escape(arm_depth, &arm_state, outer_live)?;
                prune_dead_loans(&mut arm_state, outer_live);
                match &mut merged {
                    Some(acc) => merge_states(acc, &arm_state),
                    None => merged = Some(arm_state),
                }
            }
        }
        if value_used && !out_sources.is_empty() {
            self.temp_sources.insert(expr_id, out_sources);
        }
        if let Some(acc) = merged {
            *state = acc;
        } else {
            state.diverged = true;
        }
        if let Some(st) = scrut_temp {
            self.release_temp_holder(state, st);
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
        let targets = self.active_live_targets();
        let loop_live = self.compute_loop_live(label, None, Some(pat), body, outer_live, &targets);
        self.check_expr(iter, state, false, &loop_live, true, depth)?;
        let iter_sources = self.expr_borrow_sources(iter, state);
        let iter_temp = if iter_sources.is_empty() {
            None
        } else {
            let id = self.next_iter_holder();
            push_source_loans(&mut state.loans, Some(id), &iter_sources, Some(iter));
            Some(id)
        };
        let item_sources: Vec<BorrowSource> = iter_sources
            .iter()
            .filter(|s| !s.env_only)
            .cloned()
            .collect();
        self.check_loop(
            None,
            label,
            None,
            Some((pat, &item_sources)),
            iter_temp,
            body,
            state,
            outer_live,
            depth,
        )?;
        if let Some(id) = iter_temp {
            self.release_temp_holder(state, id);
        }
        Ok(())
    }

    pub(super) fn check_loop(
        &mut self,
        expr_id: Option<ExprId>,
        label: Option<u32>,
        cond: Option<ExprId>,
        pat: Option<(PatId, &[BorrowSource])>,
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
        if let Some((p, _)) = pat {
            self.collect_pat_locals(p, &mut loop_locals);
        }
        self.collect_declared_locals_expr(body, &mut loop_locals);
        let targets = self.active_live_targets();
        let mut loop_live =
            self.compute_loop_live(label, cond, pat.map(|(p, _)| p), body, outer_live, &targets);
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
            continues: Vec::new(),
            sources: Vec::new(),
            break_live: outer_live.clone(),
            continue_live: Some(loop_live.clone()),
        });
        let mut iter_state = state.clone();
        let mut exit_cond_states = Vec::new();
        let saved_loop_depth = self.loop_depth;
        self.loop_depth += 1;
        let loop_res = self.with_let_extend(None, |this| -> Result<()> {
            for _ in 0..2 {
                iter_state.diverged = false;
                let mut drop_temps = Vec::new();
                if let Some(c) = cond {
                    let mut cond_dummy = iter_state.clone();
                    this.check_cond_for_branches(
                        c,
                        &mut iter_state,
                        &mut cond_dummy,
                        &loop_live,
                        outer_live,
                        loop_depth,
                        &mut drop_temps,
                    )?;
                    if !cond_dummy.diverged {
                        retain_non_loop_state(&mut cond_dummy, &loop_locals);
                        exit_cond_states.push(cond_dummy);
                    }
                }
                if let Some((p, sources)) = pat {
                    this.bind_pat(
                        p,
                        None,
                        None,
                        &[],
                        sources,
                        &mut iter_state,
                        (true, true, loop_depth),
                    )?;
                }
                this.check_expr(body, &mut iter_state, false, &loop_live, false, loop_depth)?;
                let Some(next_state) = this.finish_loop_iter(
                    &iter_state,
                    drop_temps,
                    &mut exit_cond_states,
                    &loop_locals,
                    outer_live,
                    loop_depth,
                )?
                else {
                    iter_state.diverged = true;
                    break;
                };
                iter_state = next_state;
            }
            Ok(())
        });
        self.loop_depth = saved_loop_depth;
        loop_res?;
        let frame = self.breaks.pop().expect("loop pushed break frame");
        if let Some(id) = expr_id
            && !frame.sources.is_empty()
            && self.const_depth == 0
        {
            self.temp_sources
                .insert(id, super::propagate::dedup_sources(frame.sources));
        }
        if infinite || cond.is_some() {
            let mut exits = exit_cond_states;
            exits.extend(frame.states);
            match merge_break_states(None, exits) {
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
        self.strip_loop_locals(state, &loop_locals);
        Ok(())
    }

    fn finish_loop_iter(
        &mut self,
        iter_state: &State,
        drop_temps: Vec<u32>,
        exit_cond_states: &mut Vec<State>,
        loop_locals: &HashSet<u32>,
        outer_live: &HashSet<u32>,
        loop_depth: u32,
    ) -> Result<Option<State>> {
        let continues = self
            .breaks
            .last_mut()
            .map(|f| std::mem::take(&mut f.continues))
            .unwrap_or_default();
        let fallthrough = (!iter_state.diverged).then(|| iter_state.clone());
        let mut merged_next = merge_break_states(fallthrough, continues);
        for id in drop_temps {
            if let Some(ref mut next) = merged_next {
                self.release_temp_holder(next, id);
            } else {
                self.release_temp_in_breaks(id);
            }
            Checker::release_temp_states(exit_cond_states, id);
        }
        let Some(mut next_state) = merged_next else {
            return Ok(None);
        };
        retain_non_loop_state(&mut next_state, loop_locals);
        self.check_block_escape(loop_depth, &next_state, outer_live)?;
        for loan in &mut next_state.loans {
            for step in &mut loan.steps {
                if matches!(step, PlaceStep::Opaque(Some(_))) {
                    *step = PlaceStep::Opaque(None);
                }
            }
        }
        Ok(Some(next_state))
    }

    fn strip_loop_locals(&mut self, state: &mut State, loop_locals: &HashSet<u32>) {
        retain_non_loop_state(state, loop_locals);
        for f in &mut self.breaks {
            for s in &mut f.states {
                retain_non_loop_state(s, loop_locals);
            }
            for s in &mut f.continues {
                retain_non_loop_state(s, loop_locals);
            }
        }
    }
}

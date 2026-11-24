use anyhow::Result;
use rustc_hash::FxHashSet as HashSet;

use crate::body::{BlockKind, Expr, ExprId, List, Stmt};
use crate::sema::infer::{By, Closure};
use crate::sema::ty::{Kind, Ty};

use super::super::bind::push_loan;
use super::super::places::{FieldKey, PlaceStep};
use super::super::propagate::{BorrowSource, dedup_sources};
use super::super::{Checker, State};

impl Checker<'_, '_> {
    pub(in super::super) fn check_closure(
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
            maybe_init: state.maybe_init.clone(),
            moved: state.moved.clone(),
            moved_at: state.moved_at.clone(),
            partial_moves: state.partial_moves.clone(),
            loans: Vec::new(),
            diverged: false,
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
                let mut fallback = Vec::new();
                if let Some(&local) = closure.params.get(idx)
                    && (local as usize) < body_state.init.len()
                {
                    self.scope_depth[local as usize] = closure_depth;
                    body_state.init[local as usize] = true;
                    body_state.maybe_init[local as usize] = true;
                    body_state.moved[local as usize] = false;
                    body_state.moved_at[local as usize] = None;
                    fallback.push(self.param_fallback_source(local));
                }
                if let Err(err) = self.bind_pat(
                    self.body.params[p_id as usize].pat,
                    None,
                    None,
                    &[],
                    &fallback,
                    &mut body_state,
                    (true, true, closure_depth),
                ) {
                    self.no_move_captures = saved_no_move;
                    return Err(err);
                }
            }
        }
        let saved_returns = (
            self.returns_borrow,
            self.return_min_depth,
            self.closure_depth,
        );
        let next_by_val = closure
            .captures
            .iter()
            .filter(|c| c.by == By::Value && !self.is_ref_ty(self.local_ty(c.local)))
            .map(|c| c.local)
            .collect();
        let saved_by_val = std::mem::replace(&mut self.closure_by_val_caps, next_by_val);
        self.returns_borrow = self.ty_holds_borrow(closure.ret);
        self.return_min_depth = closure_depth;
        self.closure_depth = closure_depth;
        let init_moved = body_state.moved.clone();
        let init_partial = body_state.partial_moves.clone();
        let empty = HashSet::default();
        let res = self.with_let_extend(None, |this| {
            if let Expr::Block {
                kind,
                stmts,
                tail,
                label,
            } = *this.body.expr(body)
                && matches!(kind, BlockKind::Async | BlockKind::AsyncMove)
            {
                this.check_block(
                    expr,
                    kind,
                    stmts,
                    tail,
                    label,
                    &mut body_state,
                    (false, true),
                    &empty,
                    depth,
                )
            } else {
                this.check_expr(body, &mut body_state, false, &empty, true, closure_depth)
            }
        });
        self.no_move_captures = saved_no_move;
        let escape_res = res.and_then(|()| {
            if self.returns_borrow && !body_state.diverged {
                self.check_return_escape(body, &body_state)?;
            }
            Ok(())
        });
        (
            self.returns_borrow,
            self.return_min_depth,
            self.closure_depth,
        ) = saved_returns;
        self.closure_by_val_caps = saved_by_val;
        escape_res?;
        let cap_locals: Vec<u32> = closure.captures.iter().map(|c| c.local).collect();
        let moves_capture =
            Self::has_new_moves(&init_moved, &init_partial, &body_state, &cap_locals);
        let mutates_capture = closure.captures.iter().any(|c| {
            matches!(c.by, By::Mut | By::Value)
                && self
                    .closure_capture_info(expr, c.by, c.local)
                    .iter()
                    .any(|(_, m, ..)| *m)
        });
        let borrows_local = self.closure_borrow_sources(expr, state).iter().any(|s| {
            super::super::checks::is_temp_target(s.target).is_some()
                || !self.from_caller(s.target, s.through_ref)
        });
        let ret_params = if self.ty_holds_borrow(closure.ret) {
            self.closure_ret_param_indices(body, &closure.params, &body_state)
        } else {
            Vec::new()
        };
        self.record_closure_body_loans(
            expr,
            &closure.params,
            closure_depth,
            &body_state,
            state,
            (moves_capture, mutates_capture, borrows_local),
            ret_params,
        );
        Ok(())
    }

    fn push_probe_loan(probe_state: &mut State, target: u32, holder: u32) {
        push_loan(
            &mut probe_state.loans,
            target,
            Vec::new(),
            Some(holder),
            false,
            true,
            false,
            false,
            None,
        );
    }

    fn record_ret_param_sources(
        sources: &[BorrowSource],
        closure_params: &[u32],
        out: &mut Vec<usize>,
    ) {
        for src in sources {
            if let Some(idx) = closure_params.iter().position(|&p| p == src.target)
                && !out.contains(&idx)
            {
                out.push(idx);
            }
        }
    }

    fn bind_probe_pat(
        &mut self,
        pat: crate::body::PatId,
        init: ExprId,
        probe_state: &mut State,
        closure_params: &[u32],
    ) {
        let mut bound = HashSet::default();
        self.collect_pat_locals(pat, &mut bound);
        if bound.is_empty() {
            return;
        }
        let srcs = self.expr_borrow_sources(init, probe_state);
        for src in srcs {
            if closure_params.contains(&src.target) {
                for &local in &bound {
                    Self::push_probe_loan(probe_state, src.target, local);
                }
            }
        }
    }

    fn closure_ret_param_indices(
        &mut self,
        body: ExprId,
        closure_params: &[u32],
        body_state: &State,
    ) -> Vec<usize> {
        let mut probe_state = body_state.clone();
        for &p in closure_params {
            if self.ty_holds_borrow(self.local_ty(p)) {
                Self::push_probe_loan(&mut probe_state, p, p);
            }
        }
        let mut ret_exprs = vec![body];
        self.collect_closure_return_exprs(
            body,
            (true, false),
            &mut ret_exprs,
            &mut probe_state,
            closure_params,
        );
        let mut out = Vec::new();
        for ret_expr in ret_exprs {
            if let Some(cached) = self.temp_sources.get(&ret_expr) {
                Self::record_ret_param_sources(cached, closure_params, &mut out);
            }
            let srcs = self.expr_borrow_sources(ret_expr, &probe_state);
            Self::record_ret_param_sources(&srcs, closure_params, &mut out);
        }
        out
    }

    fn collect_closure_return_exprs(
        &mut self,
        expr: ExprId,
        (in_tail, break_returns): (bool, bool),
        out: &mut Vec<ExprId>,
        probe_state: &mut State,
        closure_params: &[u32],
    ) {
        let non_tail = (false, false);
        match *self.body.expr(expr) {
            Expr::Lit(..)
            | Expr::Path(_)
            | Expr::Closure { .. }
            | Expr::Continue(_)
            | Expr::Macro { .. }
            | Expr::Underscore
            | Expr::FormatArgs(_)
            | Expr::OffsetOf { .. } => {}
            Expr::Return(val) => {
                if let Some(val) = val {
                    out.push(val);
                    self.collect_closure_return_exprs(
                        val,
                        non_tail,
                        out,
                        probe_state,
                        closure_params,
                    );
                }
            }
            Expr::Become(val) => {
                out.push(val);
                self.collect_closure_return_exprs(val, non_tail, out, probe_state, closure_params);
            }
            Expr::Break(_, val) => {
                if let Some(val) = val {
                    if in_tail || break_returns {
                        out.push(val);
                    }
                    self.collect_closure_return_exprs(
                        val,
                        non_tail,
                        out,
                        probe_state,
                        closure_params,
                    );
                }
            }
            Expr::Yield(val) => {
                if let Some(val) = val {
                    self.collect_closure_return_exprs(
                        val,
                        non_tail,
                        out,
                        probe_state,
                        closure_params,
                    );
                }
            }
            Expr::Block {
                stmts, tail, label, ..
            } => {
                let stmt_break = break_returns || (in_tail && label.is_some());
                for id in self.body.list(stmts).to_vec() {
                    match self.body.stmts[id as usize] {
                        Stmt::Let {
                            pat,
                            init,
                            otherwise,
                            ..
                        } => {
                            if let Some(e) = init {
                                self.collect_closure_return_exprs(
                                    e,
                                    non_tail,
                                    out,
                                    probe_state,
                                    closure_params,
                                );
                                self.bind_probe_pat(pat, e, probe_state, closure_params);
                            }
                            if let Some(e) = otherwise {
                                self.collect_closure_return_exprs(
                                    e,
                                    (false, stmt_break),
                                    out,
                                    probe_state,
                                    closure_params,
                                );
                            }
                        }
                        Stmt::Expr(e, _) => self.collect_closure_return_exprs(
                            e,
                            (false, stmt_break),
                            out,
                            probe_state,
                            closure_params,
                        ),
                        Stmt::Item(..) | Stmt::Macro { .. } | Stmt::Empty => {}
                    }
                }
                if let Some(t) = tail {
                    self.collect_closure_return_exprs(
                        t,
                        (in_tail, stmt_break),
                        out,
                        probe_state,
                        closure_params,
                    );
                }
            }
            Expr::If(c, t, e) => {
                self.collect_closure_return_exprs(c, non_tail, out, probe_state, closure_params);
                self.collect_closure_return_exprs(
                    t,
                    (in_tail, break_returns),
                    out,
                    probe_state,
                    closure_params,
                );
                if let Some(alt) = e {
                    self.collect_closure_return_exprs(
                        alt,
                        (in_tail, break_returns),
                        out,
                        probe_state,
                        closure_params,
                    );
                }
            }
            Expr::Let(pat, init) => {
                self.collect_closure_return_exprs(init, non_tail, out, probe_state, closure_params);
                self.bind_probe_pat(pat, init, probe_state, closure_params);
            }
            Expr::Match(scrut, arms) => {
                self.collect_closure_return_exprs(
                    scrut,
                    non_tail,
                    out,
                    probe_state,
                    closure_params,
                );
                for id in self.body.list(arms).to_vec() {
                    let arm = self.body.arms[id as usize];
                    self.bind_probe_pat(arm.pat, scrut, probe_state, closure_params);
                    if let Some(g) = arm.guard {
                        self.collect_closure_return_exprs(
                            g,
                            non_tail,
                            out,
                            probe_state,
                            closure_params,
                        );
                    }
                    self.collect_closure_return_exprs(
                        arm.body,
                        (in_tail, break_returns),
                        out,
                        probe_state,
                        closure_params,
                    );
                }
            }
            Expr::Loop { body, .. } => self.collect_closure_return_exprs(
                body,
                (false, in_tail),
                out,
                probe_state,
                closure_params,
            ),
            Expr::While { cond, body, .. } => {
                self.collect_closure_return_exprs(cond, non_tail, out, probe_state, closure_params);
                self.collect_closure_return_exprs(body, non_tail, out, probe_state, closure_params);
            }
            Expr::For {
                pat, iter, body, ..
            } => {
                self.collect_closure_return_exprs(iter, non_tail, out, probe_state, closure_params);
                self.bind_probe_pat(pat, iter, probe_state, closure_params);
                self.collect_closure_return_exprs(body, non_tail, out, probe_state, closure_params);
            }
            Expr::Assign(lhs, rhs) | Expr::AssignOp(_, lhs, rhs) => {
                self.collect_closure_return_exprs(lhs, non_tail, out, probe_state, closure_params);
                self.collect_closure_return_exprs(rhs, non_tail, out, probe_state, closure_params);
                if let Some(local) = self.expr_plain_local(lhs) {
                    for src in self.expr_borrow_sources(rhs, probe_state) {
                        if closure_params.contains(&src.target) {
                            Self::push_probe_loan(probe_state, src.target, local);
                        }
                    }
                }
            }
            Expr::Paren(inner)
            | Expr::Cast(inner, _)
            | Expr::Try(inner)
            | Expr::Await(inner)
            | Expr::Unary(_, inner)
            | Expr::Ref { expr: inner, .. }
            | Expr::Field(inner, _)
            | Expr::TupleField(inner, _) => self.collect_closure_return_exprs(
                inner,
                (in_tail, break_returns),
                out,
                probe_state,
                closure_params,
            ),
            Expr::Binary(_, a, b) | Expr::Index(a, b) | Expr::Repeat(a, b) => {
                self.collect_closure_return_exprs(a, non_tail, out, probe_state, closure_params);
                self.collect_closure_return_exprs(b, non_tail, out, probe_state, closure_params);
            }
            Expr::Call(callee, args) => {
                self.collect_closure_return_exprs(
                    callee,
                    non_tail,
                    out,
                    probe_state,
                    closure_params,
                );
                for id in self.body.list(args).to_vec() {
                    self.collect_closure_return_exprs(
                        ExprId(id),
                        non_tail,
                        out,
                        probe_state,
                        closure_params,
                    );
                }
            }
            Expr::Method {
                receiver, params, ..
            } => {
                self.collect_closure_return_exprs(
                    receiver,
                    non_tail,
                    out,
                    probe_state,
                    closure_params,
                );
                for id in self.body.list(params).to_vec() {
                    self.collect_closure_return_exprs(
                        ExprId(id),
                        non_tail,
                        out,
                        probe_state,
                        closure_params,
                    );
                }
            }
            Expr::Tuple(list) | Expr::Array(list) => {
                for id in self.body.list(list).to_vec() {
                    self.collect_closure_return_exprs(
                        ExprId(id),
                        non_tail,
                        out,
                        probe_state,
                        closure_params,
                    );
                }
            }
            Expr::Struct { fields, base, .. } => {
                for id in self.body.list(fields).to_vec() {
                    let val = self.body.fields[id as usize].value;
                    self.collect_closure_return_exprs(
                        val,
                        non_tail,
                        out,
                        probe_state,
                        closure_params,
                    );
                }
                if let Some(b) = base {
                    self.collect_closure_return_exprs(
                        b,
                        non_tail,
                        out,
                        probe_state,
                        closure_params,
                    );
                }
            }
            Expr::Range { from, to, .. } => {
                if let Some(f) = from {
                    self.collect_closure_return_exprs(
                        f,
                        non_tail,
                        out,
                        probe_state,
                        closure_params,
                    );
                }
                if let Some(t) = to {
                    self.collect_closure_return_exprs(
                        t,
                        non_tail,
                        out,
                        probe_state,
                        closure_params,
                    );
                }
            }
        }
    }

    fn check_closure_captures(
        &mut self,
        expr: ExprId,
        closure: &Closure,
        state: &mut State,
        body_state: &mut State,
    ) -> Result<()> {
        let mut seen_whole = Vec::new();
        for cap in &closure.captures {
            let local = cap.local;
            if (local as usize) >= self.is_copy.len() {
                continue;
            }
            if cap.path.is_empty() {
                if seen_whole.contains(&local) {
                    continue;
                }
                seen_whole.push(local);
            }
            let cap_steps: Vec<PlaceStep> = cap
                .path
                .iter()
                .map(|&(_, f)| PlaceStep::Field(FieldKey::Tuple(f)))
                .collect();
            let cap_ty = if cap.path.is_empty() {
                self.local_ty(local)
            } else {
                let r = self.local_ty(local);
                cap.ty(self.cx, r).unwrap_or(r)
            };
            let cap_copy = if cap.path.is_empty() {
                self.is_copy[local as usize]
            } else {
                self.ty_is_copy(cap_ty)
            };
            match cap.by {
                By::Ref | By::Mut => {
                    let is_mut_ref = self.is_mut_ref_ty(self.local_ty(local));
                    for (steps, is_mut, through_ref, mut_ref_root) in
                        self.closure_capture_info(expr, cap.by, local)
                    {
                        if steps.starts_with(&cap_steps) {
                            let ref_deref = is_mut_ref && steps.first() == Some(&PlaceStep::Deref);
                            self.check_borrow(
                                local,
                                through_ref || ref_deref,
                                (is_mut && mut_ref_root) || ref_deref,
                                &steps,
                                is_mut,
                                state,
                            )?;
                            if cap.by == By::Ref
                                && !is_mut
                                && !self.is_mut_ref_ty(cap_ty)
                                && !mut_ref_root
                                && !matches!(self.cx.types.kind(cap_ty), Kind::Closure(..))
                            {
                                push_loan(
                                    &mut body_state.loans,
                                    local,
                                    steps,
                                    None,
                                    false,
                                    false,
                                    false,
                                    false,
                                    self.current_expr,
                                );
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
                        if !paths.is_empty()
                            && paths.iter().all(|(s, ..)| {
                                !s.is_empty() && s.first() != Some(&PlaceStep::Deref)
                            })
                        {
                            for (steps, ..) in paths {
                                self.check_can_move(local, &steps, state)?;
                                self.push_partial_move(state, local, steps, Some(expr));
                            }
                        } else {
                            self.check_can_move(local, &[], state)?;
                            self.mark_local_moved(state, local, Some(expr));
                        }
                    } else {
                        self.check_can_move(local, &cap_steps, state)?;
                        self.push_partial_move(state, local, cap_steps, Some(expr));
                    }
                }
            }
        }
        Ok(())
    }

    pub(in super::super) fn closure_borrow_sources(
        &mut self,
        expr: ExprId,
        state: &State,
    ) -> Vec<BorrowSource> {
        let Some(closure) = self.typed.closures.get(&expr).cloned() else {
            return Vec::new();
        };
        let ret_targets = self.closure_ret_targets(expr, closure.ret, state);
        let capture_env_only = |target: u32| {
            ret_targets
                .as_ref()
                .is_some_and(|set| !set.contains(&target))
        };
        let mut out = Vec::new();
        for cap in &closure.captures {
            let cap_steps: Vec<PlaceStep> = cap
                .path
                .iter()
                .map(|&(_, f)| PlaceStep::Field(FieldKey::Tuple(f)))
                .collect();
            if cap.by == By::Value {
                let cap_ty = cap
                    .ty(self.cx, self.local_ty(cap.local))
                    .unwrap_or_else(|_| self.local_ty(cap.local));
                if self.is_mut_ref_ty(cap_ty) {
                    let mut steps = cap_steps;
                    steps.push(PlaceStep::Deref);
                    out.push(BorrowSource {
                        target: cap.local,
                        steps,
                        mutable: true,
                        through_ref: true,
                        env_only: capture_env_only(cap.local),
                    });
                }
            } else {
                for (steps, is_mut, through_ref, _) in
                    self.closure_capture_info(expr, cap.by, cap.local)
                {
                    if steps.starts_with(&cap_steps) {
                        out.push(BorrowSource {
                            target: cap.local,
                            steps,
                            mutable: is_mut,
                            through_ref,
                            env_only: capture_env_only(cap.local),
                        });
                    }
                }
            }
            for l in state.loans.iter().filter(|l| l.holder == Some(cap.local)) {
                out.push(BorrowSource {
                    env_only: l.env_only || capture_env_only(l.target),
                    ..BorrowSource::from_loan(l)
                });
            }
        }
        dedup_sources(out)
    }

    pub(in super::super) fn closure_ret_targets(
        &mut self,
        expr: ExprId,
        ret_ty: Ty,
        state: &State,
    ) -> Option<HashSet<u32>> {
        if !self.ty_holds_borrow(ret_ty) {
            return Some(HashSet::default());
        }
        let Expr::Closure { body, .. } = *self.body.expr(expr) else {
            return None;
        };
        if let Some(cached) = self.temp_sources.get(&body)
            && !self.expr_has_return(body)
        {
            return Some(cached.iter().map(|s| s.target).collect());
        }
        let mut inner_locals = HashSet::default();
        if !self.can_analyze_closure_ret(body, &mut inner_locals) {
            return None;
        }
        let targets = self
            .expr_borrow_sources(body, state)
            .into_iter()
            .map(|s| s.target)
            .collect();
        Some(targets)
    }

    fn can_analyze_closure_ret(&mut self, expr: ExprId, inner_locals: &mut HashSet<u32>) -> bool {
        match *self.body.expr(expr) {
            Expr::Paren(inner) | Expr::Cast(inner, _) | Expr::Try(inner) => {
                self.can_analyze_closure_ret(inner, inner_locals)
            }
            Expr::Block {
                kind, stmts, tail, ..
            } => {
                if matches!(kind, BlockKind::Async | BlockKind::AsyncMove) {
                    return false;
                }
                let stm_ids: Vec<u32> = self.body.list(stmts).to_vec();
                for id in stm_ids {
                    match self.body.stmts[id as usize] {
                        Stmt::Let {
                            pat,
                            init,
                            otherwise,
                            ..
                        } => {
                            self.collect_pat_locals(pat, inner_locals);
                            if init.is_some_and(|e| self.expr_has_return(e))
                                || otherwise.is_some_and(|e| self.expr_has_return(e))
                            {
                                return false;
                            }
                        }
                        Stmt::Expr(e, _) => {
                            if self.expr_has_return(e) {
                                return false;
                            }
                        }
                        Stmt::Item(..) | Stmt::Macro { .. } | Stmt::Empty => {}
                    }
                }
                tail.is_none_or(|t| self.can_analyze_closure_ret(t, inner_locals))
            }
            Expr::If(c, t, e) => {
                !self.expr_has_return(c)
                    && self.can_analyze_closure_ret(t, inner_locals)
                    && e.is_none_or(|alt| self.can_analyze_closure_ret(alt, inner_locals))
            }
            Expr::Match(scrut, arms) => {
                if self.expr_has_return(scrut) {
                    return false;
                }
                let arm_ids: Vec<u32> = self.body.list(arms).to_vec();
                for id in arm_ids {
                    let arm = self.body.arms[id as usize];
                    let mut arm_locals = inner_locals.clone();
                    self.collect_pat_locals(arm.pat, &mut arm_locals);
                    if arm.guard.is_some_and(|g| self.expr_has_return(g))
                        || !self.can_analyze_closure_ret(arm.body, &mut arm_locals)
                    {
                        return false;
                    }
                }
                true
            }
            _ => {
                !self.expr_has_return(expr)
                    && !self.expr_uses_inner_borrow_local(expr, inner_locals)
            }
        }
    }
}

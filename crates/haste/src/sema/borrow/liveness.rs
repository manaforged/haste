use rustc_hash::FxHashSet as HashSet;

use crate::body::{BinOp, BlockKind, Expr, ExprId, List, Pat, PatId, Stmt, jumps_to};
use crate::sema::infer::Res;
use crate::sema::ty::Ty;

use super::Checker;

#[derive(Clone, Copy)]
pub(super) struct LiveTarget<'a> {
    pub(super) label: Option<u32>,
    pub(super) is_block: bool,
    pub(super) break_live: &'a HashSet<u32>,
    pub(super) continue_live: Option<&'a HashSet<u32>>,
}

impl Checker<'_, '_> {
    pub(super) fn active_live_targets(&self) -> Vec<LiveTarget<'_>> {
        self.breaks
            .iter()
            .map(|f| LiveTarget {
                label: f.label,
                is_block: f.is_block,
                break_live: &f.break_live,
                continue_live: f.continue_live.as_ref(),
            })
            .collect()
    }
    pub(super) fn collect_pat_locals(&self, pat: PatId, out: &mut HashSet<u32>) {
        match *self.body.pat(pat) {
            Pat::Bind { sub, .. } => {
                if let Some(&local) = self.typed.binds.get(&pat) {
                    out.insert(local);
                }
                if let Some(sub) = sub {
                    self.collect_pat_locals(sub, out);
                }
            }
            Pat::Paren(inner)
            | Pat::Ref { pat: inner, .. }
            | Pat::Box(inner)
            | Pat::Deref(inner) => {
                self.collect_pat_locals(inner, out);
            }
            Pat::Tuple(list) | Pat::TupleStruct(_, list) | Pat::Slice(list) | Pat::Or(list) => {
                for &at in self.body.list(list) {
                    self.collect_pat_locals(PatId(at), out);
                }
            }
            Pat::Struct { fields, .. } => {
                for &at in self.body.list(fields) {
                    self.collect_pat_locals(self.body.field_pats[at as usize].pat, out);
                }
            }
            _ => {}
        }
    }

    pub(super) fn expr_plain_local(&self, expr: ExprId) -> Option<u32> {
        match *self.body.expr(expr) {
            Expr::Paren(inner) => self.expr_plain_local(inner),
            Expr::Path(_) => match self.typed.res.get(&expr) {
                Some(Res::Local(local)) => Some(*local),
                _ => None,
            },
            _ => None,
        }
    }

    pub(super) fn block_liveness(
        &self,
        ids: &[u32],
        tail: Option<ExprId>,
        outer_live: &HashSet<u32>,
    ) -> (Vec<HashSet<u32>>, Vec<HashSet<u32>>, HashSet<u32>) {
        let targets = self.active_live_targets();
        let mut live = outer_live.clone();
        if let Some(t) = tail {
            self.collect_live_expr_with(t, &mut live, &targets);
        }
        for &id in ids {
            if let Stmt::Let {
                pat, init: Some(_), ..
            } = self.body.stmts[id as usize]
                && !matches!(*self.body.pat(pat), Pat::Wild)
            {
                let mut bound = HashSet::default();
                self.collect_pat_locals(pat, &mut bound);
                for local in bound {
                    if self.drop_locals.contains(&local) {
                        live.insert(local);
                    }
                }
            }
        }
        let tail_live = live.clone();
        let mut after_stmt = vec![HashSet::default(); ids.len()];
        let mut before_stmt = vec![HashSet::default(); ids.len()];
        for pos in (0..ids.len()).rev() {
            after_stmt[pos] = live.clone();
            self.collect_live_stmt_with(self.body.stmts[ids[pos] as usize], &mut live, &targets);
            before_stmt[pos] = live.clone();
        }
        (before_stmt, after_stmt, tail_live)
    }

    pub(super) fn collect_match_arms_live(
        &self,
        arm_ids: &[u32],
        outer_live: &HashSet<u32>,
        live: &mut HashSet<u32>,
    ) {
        let targets = self.active_live_targets();
        for &arm_id in arm_ids {
            let arm = self.body.arms[arm_id as usize];
            let mut arm_live = outer_live.clone();
            self.collect_live_expr_with(arm.body, &mut arm_live, &targets);
            if let Some(g) = arm.guard {
                self.collect_live_expr_with(g, &mut arm_live, &targets);
            }
            live.extend(arm_live);
        }
    }

    fn collect_live_stmt_with(
        &self,
        stmt: Stmt,
        live: &mut HashSet<u32>,
        targets: &[LiveTarget<'_>],
    ) {
        match stmt {
            Stmt::Let {
                pat,
                init,
                otherwise,
                ..
            } => {
                let mut bound = HashSet::default();
                self.collect_pat_locals(pat, &mut bound);
                for local in bound {
                    live.remove(&local);
                }
                if let Some(e) = otherwise {
                    self.collect_live_expr_with(e, live, targets);
                }
                if let Some(e) = init {
                    self.collect_live_expr_with(e, live, targets);
                }
            }
            Stmt::Expr(e, _) => self.collect_live_expr_with(e, live, targets),
            Stmt::Item(..) | Stmt::Macro { .. } | Stmt::Empty => {}
        }
    }

    pub(super) fn collect_live_expr(&self, expr: ExprId, live: &mut HashSet<u32>) {
        let targets = self.active_live_targets();
        self.collect_live_expr_with(expr, live, &targets);
    }

    fn collect_live_expr_with(
        &self,
        expr: ExprId,
        live: &mut HashSet<u32>,
        targets: &[LiveTarget<'_>],
    ) {
        match *self.body.expr(expr) {
            Expr::Path(_) => {
                if let Some(Res::Local(local)) = self.typed.res.get(&expr) {
                    live.insert(*local);
                }
            }
            Expr::Unary(_, e)
            | Expr::Field(e, _)
            | Expr::TupleField(e, _)
            | Expr::Cast(e, _)
            | Expr::Ref { expr: e, .. }
            | Expr::Try(e)
            | Expr::Await(e)
            | Expr::Paren(e)
            | Expr::Become(e) => self.collect_live_expr_with(e, live, targets),
            Expr::Assign(lhs, rhs) => {
                if let Some(local) = self.expr_plain_local(lhs) {
                    if !self.drop_locals.contains(&local) {
                        live.remove(&local);
                    }
                    self.collect_live_expr_with(rhs, live, targets);
                } else {
                    self.collect_live_expr_with(rhs, live, targets);
                    self.collect_live_expr_with(lhs, live, targets);
                }
            }
            Expr::Binary(op, a, b) => {
                if matches!(op, BinOp::And | BinOp::Or) {
                    let mut b_live = live.clone();
                    self.collect_live_expr_with(b, &mut b_live, targets);
                    live.extend(b_live);
                } else {
                    self.collect_live_expr_with(b, live, targets);
                }
                self.collect_live_expr_with(a, live, targets);
            }
            Expr::AssignOp(_, a, b) | Expr::Index(a, b) | Expr::Repeat(a, b) => {
                self.collect_live_expr_with(b, live, targets);
                self.collect_live_expr_with(a, live, targets);
            }
            Expr::Call(callee, args) => {
                for &a in self.body.list(args).iter().rev() {
                    self.collect_live_expr_with(ExprId(a), live, targets);
                }
                self.collect_live_expr_with(callee, live, targets);
            }
            Expr::Method {
                receiver, params, ..
            } => {
                for &a in self.body.list(params).iter().rev() {
                    self.collect_live_expr_with(ExprId(a), live, targets);
                }
                self.collect_live_expr_with(receiver, live, targets);
            }
            Expr::Block {
                stmts, tail, label, ..
            } => self.collect_live_block(label, stmts, tail, live, targets),
            Expr::If(c, t, e) => self.collect_live_if(c, t, e, live, targets),
            Expr::Let(pat, init) => {
                let mut bound = HashSet::default();
                self.collect_pat_locals(pat, &mut bound);
                for local in bound {
                    live.remove(&local);
                }
                self.collect_live_expr_with(init, live, targets);
            }
            Expr::While {
                cond, body, label, ..
            } => *live = self.compute_loop_live(label, Some(cond), None, body, live, targets),
            Expr::Loop { body, label, .. } => {
                *live = self.compute_loop_live(label, None, None, body, live, targets)
            }
            Expr::Closure { params, body, .. } => {
                self.collect_live_closure(expr, params, body, live);
            }
            Expr::For {
                pat,
                iter,
                body,
                label,
                ..
            } => {
                *live = self.compute_loop_live(label, None, Some(pat), body, live, targets);
                self.collect_live_expr_with(iter, live, targets);
            }
            Expr::Match(scrut, arms) => self.collect_live_match(scrut, arms, live, targets),
            Expr::Tuple(list) | Expr::Array(list) => self.collect_live_list(list, live, targets),
            Expr::Struct { fields, base, .. } => {
                self.collect_live_struct(fields, base, live, targets);
            }
            Expr::Range { from, to, .. } => {
                if let Some(t) = to {
                    self.collect_live_expr_with(t, live, targets);
                }
                if let Some(f) = from {
                    self.collect_live_expr_with(f, live, targets);
                }
            }
            Expr::Return(e) => {
                live.clear();
                if let Some(e) = e {
                    self.collect_live_expr_with(e, live, targets);
                }
            }
            Expr::Break(label, e) => {
                if let Some(target) = targets
                    .iter()
                    .rev()
                    .find(|t| jumps_to(self.src, (t.label, t.is_block), label))
                {
                    *live = target.break_live.clone();
                }
                if let Some(e) = e {
                    self.collect_live_expr_with(e, live, targets);
                }
            }
            Expr::Continue(label) => {
                if let Some(cont) = targets
                    .iter()
                    .rev()
                    .find(|t| !t.is_block && jumps_to(self.src, (t.label, false), label))
                    .and_then(|t| t.continue_live)
                {
                    *live = cont.clone();
                }
            }
            Expr::Yield(e) => {
                if let Some(e) = e {
                    self.collect_live_expr_with(e, live, targets);
                }
            }
            _ => {}
        }
    }

    fn collect_live_closure(
        &self,
        expr: ExprId,
        params: List,
        body: ExprId,
        live: &mut HashSet<u32>,
    ) {
        let mut body_live = HashSet::default();
        self.collect_live_expr_with(body, &mut body_live, &[]);
        if let Some(closure) = self.typed.closures.get(&expr) {
            for &local in &closure.params {
                body_live.remove(&local);
            }
        }
        for &p in self.body.list(params) {
            let param = self.body.params[p as usize];
            let mut bound = HashSet::default();
            self.collect_pat_locals(param.pat, &mut bound);
            for local in bound {
                body_live.remove(&local);
            }
        }
        live.extend(body_live);
    }

    fn collect_live_list(&self, list: List, live: &mut HashSet<u32>, targets: &[LiveTarget<'_>]) {
        for &a in self.body.list(list).iter().rev() {
            self.collect_live_expr_with(ExprId(a), live, targets);
        }
    }

    fn collect_live_struct(
        &self,
        fields: List,
        base: Option<ExprId>,
        live: &mut HashSet<u32>,
        targets: &[LiveTarget<'_>],
    ) {
        if let Some(b) = base {
            self.collect_live_expr_with(b, live, targets);
        }
        for &f in self.body.list(fields).iter().rev() {
            self.collect_live_expr_with(self.body.fields[f as usize].value, live, targets);
        }
    }

    fn collect_live_block(
        &self,
        label: Option<u32>,
        stmts: List,
        tail: Option<ExprId>,
        live: &mut HashSet<u32>,
        targets: &[LiveTarget<'_>],
    ) {
        let after_block = label.map(|_| live.clone());
        let mut block_targets;
        let cur = if let Some(ref break_live) = after_block {
            block_targets = targets.to_vec();
            block_targets.push(LiveTarget {
                label,
                is_block: true,
                break_live,
                continue_live: None,
            });
            &block_targets
        } else {
            targets
        };
        if let Some(t) = tail {
            self.collect_live_expr_with(t, live, cur);
        }
        for &s in self.body.list(stmts).iter().rev() {
            self.collect_live_stmt_with(self.body.stmts[s as usize], live, cur);
        }
    }

    fn collect_live_if(
        &self,
        c: ExprId,
        t: ExprId,
        e: Option<ExprId>,
        live: &mut HashSet<u32>,
        targets: &[LiveTarget<'_>],
    ) {
        let mut then_live = live.clone();
        self.collect_live_expr_with(t, &mut then_live, targets);
        if let Some(alt) = e {
            let mut else_live = live.clone();
            self.collect_live_expr_with(alt, &mut else_live, targets);
            *live = then_live;
            live.extend(else_live);
        } else {
            live.extend(then_live);
        }
        self.collect_live_expr_with(c, live, targets);
    }

    fn collect_live_match(
        &self,
        scrut: ExprId,
        arms: List,
        live: &mut HashSet<u32>,
        targets: &[LiveTarget<'_>],
    ) {
        let arm_ids = self.body.list(arms);
        if !arm_ids.is_empty() {
            let mut arms_live = HashSet::default();
            for &a in arm_ids {
                let arm = self.body.arms[a as usize];
                let mut arm_live = live.clone();
                self.collect_live_expr_with(arm.body, &mut arm_live, targets);
                if let Some(g) = arm.guard {
                    self.collect_live_expr_with(g, &mut arm_live, targets);
                }
                let mut bound = HashSet::default();
                self.collect_pat_locals(arm.pat, &mut bound);
                for local in bound {
                    arm_live.remove(&local);
                }
                arms_live.extend(arm_live);
            }
            *live = arms_live;
        }
        self.collect_live_expr_with(scrut, live, targets);
    }

    pub(super) fn compute_loop_live(
        &self,
        label: Option<u32>,
        cond: Option<ExprId>,
        pat: Option<PatId>,
        body: ExprId,
        after_live: &HashSet<u32>,
        targets: &[LiveTarget<'_>],
    ) -> HashSet<u32> {
        let mut loop_locals = HashSet::default();
        if let Some(c) = cond {
            self.collect_declared_locals_expr(c, &mut loop_locals);
        }
        if let Some(p) = pat {
            self.collect_pat_locals(p, &mut loop_locals);
        }
        self.collect_declared_locals_expr(body, &mut loop_locals);
        let mut head_live = HashSet::default();
        loop {
            let mut body_live = head_live.clone();
            let mut next_targets = targets.to_vec();
            next_targets.push(LiveTarget {
                label,
                is_block: false,
                break_live: after_live,
                continue_live: Some(&head_live),
            });
            self.collect_live_expr_with(body, &mut body_live, &next_targets);
            if let Some(p) = pat {
                let mut bound = HashSet::default();
                self.collect_pat_locals(p, &mut bound);
                for l in bound {
                    body_live.remove(&l);
                }
            }
            let mut next_head = if cond.is_some() || pat.is_some() {
                let mut h = after_live.clone();
                h.extend(body_live);
                h
            } else {
                body_live
            };
            if let Some(c) = cond {
                self.collect_live_expr_with(c, &mut next_head, &next_targets);
            }
            for l in &loop_locals {
                next_head.remove(l);
            }
            if next_head == head_live {
                return head_live;
            }
            head_live = next_head;
        }
    }

    pub(super) fn collect_declared_locals_expr(&self, expr: ExprId, out: &mut HashSet<u32>) {
        match *self.body.expr(expr) {
            Expr::Let(pat, init) => {
                self.collect_pat_locals(pat, out);
                self.collect_declared_locals_expr(init, out);
            }
            Expr::Paren(e)
            | Expr::Unary(_, e)
            | Expr::Field(e, _)
            | Expr::TupleField(e, _)
            | Expr::Cast(e, _)
            | Expr::Ref { expr: e, .. }
            | Expr::Try(e)
            | Expr::Await(e) => {
                self.collect_declared_locals_expr(e, out);
            }
            Expr::Binary(_, a, b)
            | Expr::Assign(a, b)
            | Expr::AssignOp(_, a, b)
            | Expr::Index(a, b)
            | Expr::Repeat(a, b)
            | Expr::While {
                cond: a, body: b, ..
            } => {
                self.collect_declared_locals_expr(a, out);
                self.collect_declared_locals_expr(b, out);
            }
            Expr::Loop { body, .. } => self.collect_declared_locals_expr(body, out),
            Expr::For {
                pat, iter, body, ..
            } => {
                self.collect_pat_locals(pat, out);
                self.collect_declared_locals_expr(iter, out);
                self.collect_declared_locals_expr(body, out);
            }
            Expr::Call(callee, list)
            | Expr::Method {
                receiver: callee,
                params: list,
                ..
            } => {
                self.collect_declared_locals_expr(callee, out);
                for &id in self.body.list(list) {
                    self.collect_declared_locals_expr(ExprId(id), out);
                }
            }
            Expr::Tuple(list) | Expr::Array(list) => {
                for &id in self.body.list(list) {
                    self.collect_declared_locals_expr(ExprId(id), out);
                }
            }
            Expr::Struct { fields, base, .. } => {
                for &id in self.body.list(fields) {
                    self.collect_declared_locals_expr(self.body.fields[id as usize].value, out);
                }
                if let Some(b) = base {
                    self.collect_declared_locals_expr(b, out);
                }
            }
            Expr::Range { from, to, .. } => {
                if let Some(f) = from {
                    self.collect_declared_locals_expr(f, out);
                }
                if let Some(t) = to {
                    self.collect_declared_locals_expr(t, out);
                }
            }
            Expr::Return(e) | Expr::Break(_, e) | Expr::Yield(e) => {
                if let Some(e) = e {
                    self.collect_declared_locals_expr(e, out);
                }
            }
            Expr::Block { stmts, tail, .. } => {
                self.collect_declared_locals_block(stmts, tail, out);
            }
            Expr::If(c, t, e) => {
                self.collect_declared_locals_expr(c, out);
                self.collect_declared_locals_expr(t, out);
                if let Some(alt) = e {
                    self.collect_declared_locals_expr(alt, out);
                }
            }
            Expr::Match(scrut, arms) => {
                self.collect_declared_locals_expr(scrut, out);
                for &a in self.body.list(arms) {
                    let arm = self.body.arms[a as usize];
                    self.collect_pat_locals(arm.pat, out);
                    if let Some(g) = arm.guard {
                        self.collect_declared_locals_expr(g, out);
                    }
                    self.collect_declared_locals_expr(arm.body, out);
                }
            }
            _ => {}
        }
    }

    fn collect_declared_locals_block(
        &self,
        stmts: crate::body::List,
        tail: Option<ExprId>,
        out: &mut HashSet<u32>,
    ) {
        for &s in self.body.list(stmts) {
            match self.body.stmts[s as usize] {
                Stmt::Let {
                    pat,
                    init,
                    otherwise,
                    ..
                } => {
                    self.collect_pat_locals(pat, out);
                    if let Some(e) = init {
                        self.collect_declared_locals_expr(e, out);
                    }
                    if let Some(e) = otherwise {
                        self.collect_declared_locals_expr(e, out);
                    }
                }
                Stmt::Expr(e, _) => self.collect_declared_locals_expr(e, out),
                _ => {}
            }
        }
        if let Some(t) = tail {
            self.collect_declared_locals_expr(t, out);
        }
    }

    pub(super) fn expr_has_return(&self, expr: ExprId) -> bool {
        match *self.body.expr(expr) {
            Expr::Return(_) => true,
            Expr::Closure { .. } => false,
            Expr::Paren(e)
            | Expr::Unary(_, e)
            | Expr::Field(e, _)
            | Expr::TupleField(e, _)
            | Expr::Cast(e, _)
            | Expr::Ref { expr: e, .. }
            | Expr::Try(e)
            | Expr::Await(e)
            | Expr::Become(e)
            | Expr::Let(_, e)
            | Expr::Loop { body: e, .. } => self.expr_has_return(e),
            Expr::Binary(_, a, b)
            | Expr::Assign(a, b)
            | Expr::AssignOp(_, a, b)
            | Expr::Index(a, b)
            | Expr::Repeat(a, b)
            | Expr::While {
                cond: a, body: b, ..
            }
            | Expr::For {
                iter: a, body: b, ..
            } => self.expr_has_return(a) || self.expr_has_return(b),
            Expr::Call(a, list)
            | Expr::Method {
                receiver: a,
                params: list,
                ..
            } => {
                self.expr_has_return(a)
                    || self
                        .body
                        .list(list)
                        .iter()
                        .any(|&id| self.expr_has_return(ExprId(id)))
            }
            Expr::Tuple(list) | Expr::Array(list) => self
                .body
                .list(list)
                .iter()
                .any(|&id| self.expr_has_return(ExprId(id))),
            Expr::Struct { fields, base, .. } => {
                self.body
                    .list(fields)
                    .iter()
                    .any(|&id| self.expr_has_return(self.body.fields[id as usize].value))
                    || base.is_some_and(|b| self.expr_has_return(b))
            }
            Expr::Block {
                kind, stmts, tail, ..
            } => {
                !matches!(kind, BlockKind::Async | BlockKind::AsyncMove)
                    && (self.body.list(stmts).iter().any(|&id| {
                        match self.body.stmts[id as usize] {
                            Stmt::Let {
                                init, otherwise, ..
                            } => {
                                init.is_some_and(|e| self.expr_has_return(e))
                                    || otherwise.is_some_and(|e| self.expr_has_return(e))
                            }
                            Stmt::Expr(e, _) => self.expr_has_return(e),
                            Stmt::Item(..) | Stmt::Macro { .. } | Stmt::Empty => false,
                        }
                    }) || tail.is_some_and(|t| self.expr_has_return(t)))
            }
            Expr::If(c, t, e) => {
                self.expr_has_return(c)
                    || self.expr_has_return(t)
                    || e.is_some_and(|alt| self.expr_has_return(alt))
            }
            Expr::Match(s, arms) => {
                self.expr_has_return(s)
                    || self.body.list(arms).iter().any(|&id| {
                        let arm = self.body.arms[id as usize];
                        arm.guard.is_some_and(|g| self.expr_has_return(g))
                            || self.expr_has_return(arm.body)
                    })
            }
            Expr::Range { from, to, .. } => {
                from.is_some_and(|f| self.expr_has_return(f))
                    || to.is_some_and(|t| self.expr_has_return(t))
            }
            Expr::Break(_, e) | Expr::Yield(e) => e.is_some_and(|e| self.expr_has_return(e)),
            Expr::Lit(..)
            | Expr::Path(_)
            | Expr::Continue(_)
            | Expr::Macro { .. }
            | Expr::Underscore
            | Expr::FormatArgs(_)
            | Expr::OffsetOf { .. } => false,
        }
    }

    pub(super) fn expr_uses_inner_borrow_local(
        &mut self,
        expr: ExprId,
        inner_locals: &HashSet<u32>,
    ) -> bool {
        if inner_locals.is_empty() {
            return false;
        }
        match *self.body.expr(expr) {
            Expr::Path(_) => {
                if let Some(&Res::Local(local)) = self.typed.res.get(&expr)
                    && inner_locals.contains(&local)
                {
                    let ty = self.local_ty(local);
                    return ty == Ty::NONE || self.ty_holds_borrow(ty);
                }
                false
            }
            Expr::Closure { .. } => true,
            Expr::Paren(e)
            | Expr::Unary(_, e)
            | Expr::Field(e, _)
            | Expr::TupleField(e, _)
            | Expr::Cast(e, _)
            | Expr::Ref { expr: e, .. }
            | Expr::Try(e)
            | Expr::Await(e)
            | Expr::Become(e) => self.expr_uses_inner_borrow_local(e, inner_locals),
            Expr::Binary(_, a, b) | Expr::Index(a, b) | Expr::Repeat(a, b) => {
                self.expr_uses_inner_borrow_local(a, inner_locals)
                    || self.expr_uses_inner_borrow_local(b, inner_locals)
            }
            Expr::Call(a, list)
            | Expr::Method {
                receiver: a,
                params: list,
                ..
            } => {
                let ids: Vec<u32> = self.body.list(list).to_vec();
                self.expr_uses_inner_borrow_local(a, inner_locals)
                    || ids
                        .into_iter()
                        .any(|id| self.expr_uses_inner_borrow_local(ExprId(id), inner_locals))
            }
            Expr::Tuple(list) | Expr::Array(list) => {
                let ids: Vec<u32> = self.body.list(list).to_vec();
                ids.into_iter()
                    .any(|id| self.expr_uses_inner_borrow_local(ExprId(id), inner_locals))
            }
            Expr::Struct { fields, base, .. } => {
                let vals: Vec<ExprId> = self
                    .body
                    .list(fields)
                    .iter()
                    .map(|&id| self.body.fields[id as usize].value)
                    .collect();
                vals.into_iter()
                    .any(|v| self.expr_uses_inner_borrow_local(v, inner_locals))
                    || base.is_some_and(|b| self.expr_uses_inner_borrow_local(b, inner_locals))
            }
            Expr::Lit(..)
            | Expr::Continue(_)
            | Expr::Underscore
            | Expr::FormatArgs(_)
            | Expr::OffsetOf { .. } => false,
            _ => true,
        }
    }
}

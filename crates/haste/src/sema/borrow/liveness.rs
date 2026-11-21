use rustc_hash::FxHashSet as HashSet;

use crate::body::{BlockKind, Expr, ExprId, Pat, PatId, Stmt};
use crate::sema::infer::Res;
use crate::sema::ty::Ty;

use super::Checker;

impl Checker<'_, '_> {
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
            Pat::Paren(inner) | Pat::Ref { pat: inner, .. } | Pat::Box(inner) | Pat::Deref(inner) => {
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

    pub(super) fn block_liveness(&self, ids: &[u32], tail: Option<ExprId>, outer_live: &HashSet<u32>) -> (Vec<HashSet<u32>>, Vec<HashSet<u32>>, HashSet<u32>) {
        let mut live = outer_live.clone();
        if let Some(t) = tail {
            self.collect_live_expr(t, &mut live);
        }
        for &id in ids {
            if let Stmt::Let { pat, init: Some(_), .. } = self.body.stmts[id as usize]
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
            self.collect_live_stmt(self.body.stmts[ids[pos] as usize], &mut live);
            before_stmt[pos] = live.clone();
        }
        (before_stmt, after_stmt, tail_live)
    }

    pub(super) fn collect_live_stmt(&self, stmt: Stmt, live: &mut HashSet<u32>) {
        match stmt {
            Stmt::Let { pat, init, otherwise, .. } => {
                let mut bound = HashSet::default();
                self.collect_pat_locals(pat, &mut bound);
                for local in bound {
                    live.remove(&local);
                }
                if let Some(e) = otherwise {
                    self.collect_live_expr(e, live);
                }
                if let Some(e) = init {
                    self.collect_live_expr(e, live);
                }
            }
            Stmt::Expr(e, _) => self.collect_live_expr(e, live),
            Stmt::Item(..) | Stmt::Macro { .. } | Stmt::Empty => {}
        }
    }

    pub(super) fn collect_live_expr(&self, expr: ExprId, live: &mut HashSet<u32>) {
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
            | Expr::Become(e) => self.collect_live_expr(e, live),
            Expr::Assign(lhs, rhs) => {
                if let Some(local) = self.expr_plain_local(lhs) {
                    if !self.drop_locals.contains(&local) {
                        live.remove(&local);
                    }
                    self.collect_live_expr(rhs, live);
                } else {
                    self.collect_live_expr(rhs, live);
                    self.collect_live_expr(lhs, live);
                }
            }
            Expr::Binary(_, a, b) | Expr::AssignOp(_, a, b) | Expr::Index(a, b) | Expr::Repeat(a, b) => {
                self.collect_live_expr(b, live);
                self.collect_live_expr(a, live);
            }
            Expr::Call(callee, args) => {
                for &a in self.body.list(args).iter().rev() {
                    self.collect_live_expr(ExprId(a), live);
                }
                self.collect_live_expr(callee, live);
            }
            Expr::Method { receiver, params, .. } => {
                for &a in self.body.list(params).iter().rev() {
                    self.collect_live_expr(ExprId(a), live);
                }
                self.collect_live_expr(receiver, live);
            }
            Expr::Block { stmts, tail, .. } => {
                if let Some(t) = tail {
                    self.collect_live_expr(t, live);
                }
                for &s in self.body.list(stmts).iter().rev() {
                    self.collect_live_stmt(self.body.stmts[s as usize], live);
                }
            }
            Expr::If(c, t, e) => {
                let mut then_live = HashSet::default();
                self.collect_live_expr(t, &mut then_live);
                live.extend(then_live);
                if let Some(alt) = e {
                    let mut else_live = HashSet::default();
                    self.collect_live_expr(alt, &mut else_live);
                    live.extend(else_live);
                }
                self.collect_live_expr(c, live);
            }
            Expr::Let(pat, init) => {
                let mut bound = HashSet::default();
                self.collect_pat_locals(pat, &mut bound);
                for local in bound {
                    live.remove(&local);
                }
                self.collect_live_expr(init, live);
            }
            Expr::While { cond, body, .. } => {
                let mut loop_live = HashSet::default();
                self.collect_live_expr(body, &mut loop_live);
                self.collect_live_expr(cond, &mut loop_live);
                live.extend(loop_live);
            }
            Expr::Loop { body, .. } => {
                let mut loop_live = HashSet::default();
                self.collect_live_expr(body, &mut loop_live);
                live.extend(loop_live);
            }
            Expr::Closure { params, body, .. } => {
                let mut body_live = HashSet::default();
                self.collect_live_expr(body, &mut body_live);
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
            Expr::For { pat, iter, body, .. } => {
                let mut loop_live = HashSet::default();
                self.collect_live_expr(body, &mut loop_live);
                let mut bound = HashSet::default();
                self.collect_pat_locals(pat, &mut bound);
                for local in bound {
                    loop_live.remove(&local);
                }
                live.extend(loop_live);
                self.collect_live_expr(iter, live);
            }
            Expr::Match(scrut, arms) => {
                let mut arms_live = HashSet::default();
                for &a in self.body.list(arms) {
                    let arm = self.body.arms[a as usize];
                    let mut arm_live = HashSet::default();
                    self.collect_live_expr(arm.body, &mut arm_live);
                    if let Some(g) = arm.guard {
                        self.collect_live_expr(g, &mut arm_live);
                    }
                    let mut bound = HashSet::default();
                    self.collect_pat_locals(arm.pat, &mut bound);
                    for local in bound {
                        arm_live.remove(&local);
                    }
                    arms_live.extend(arm_live);
                }
                live.extend(arms_live);
                self.collect_live_expr(scrut, live);
            }
            Expr::Tuple(list) | Expr::Array(list) => {
                for &a in self.body.list(list).iter().rev() {
                    self.collect_live_expr(ExprId(a), live);
                }
            }
            Expr::Struct { fields, base, .. } => {
                if let Some(b) = base {
                    self.collect_live_expr(b, live);
                }
                for &f in self.body.list(fields).iter().rev() {
                    self.collect_live_expr(self.body.fields[f as usize].value, live);
                }
            }
            Expr::Range { from, to, .. } => {
                if let Some(t) = to {
                    self.collect_live_expr(t, live);
                }
                if let Some(f) = from {
                    self.collect_live_expr(f, live);
                }
            }
            Expr::Return(e) | Expr::Break(_, e) => {
                live.clear();
                if let Some(e) = e {
                    self.collect_live_expr(e, live);
                }
            }
            Expr::Continue(_) => {
                live.clear();
            }
            Expr::Yield(e) => {
                if let Some(e) = e {
                    self.collect_live_expr(e, live);
                }
            }
            _ => {}
        }
    }

    pub(super) fn collect_declared_locals_expr(&self, expr: ExprId, out: &mut HashSet<u32>) {
        match *self.body.expr(expr) {
            Expr::Let(pat, init) => {
                self.collect_pat_locals(pat, out);
                self.collect_declared_locals_expr(init, out);
            }
            Expr::Paren(e) | Expr::Unary(_, e) | Expr::Field(e, _) | Expr::TupleField(e, _) | Expr::Cast(e, _) | Expr::Ref { expr: e, .. } | Expr::Try(e) | Expr::Await(e) => {
                self.collect_declared_locals_expr(e, out);
            }
            Expr::Binary(_, a, b) | Expr::Assign(a, b) | Expr::AssignOp(_, a, b) | Expr::Index(a, b) => {
                self.collect_declared_locals_expr(a, out);
                self.collect_declared_locals_expr(b, out);
            }
            Expr::Block { stmts, tail, .. } => {
                for &s in self.body.list(stmts) {
                    if let Stmt::Let { pat, init, otherwise, .. } = self.body.stmts[s as usize] {
                        self.collect_pat_locals(pat, out);
                        if let Some(e) = init {
                            self.collect_declared_locals_expr(e, out);
                        }
                        if let Some(e) = otherwise {
                            self.collect_declared_locals_expr(e, out);
                        }
                    } else if let Stmt::Expr(e, _) = self.body.stmts[s as usize] {
                        self.collect_declared_locals_expr(e, out);
                    }
                }
                if let Some(t) = tail {
                    self.collect_declared_locals_expr(t, out);
                }
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

    pub(super) fn expr_has_return(&self, expr: ExprId) -> bool {
        match *self.body.expr(expr) {
            Expr::Return(_) => true,
            Expr::Closure { .. } => false,
            Expr::Paren(e) | Expr::Unary(_, e) | Expr::Field(e, _) | Expr::TupleField(e, _) | Expr::Cast(e, _) | Expr::Ref { expr: e, .. } | Expr::Try(e) | Expr::Await(e) | Expr::Become(e) | Expr::Let(_, e) | Expr::Loop { body: e, .. } => {
                self.expr_has_return(e)
            }
            Expr::Binary(_, a, b) | Expr::Assign(a, b) | Expr::AssignOp(_, a, b) | Expr::Index(a, b) | Expr::Repeat(a, b) | Expr::While { cond: a, body: b, .. } | Expr::For { iter: a, body: b, .. } => {
                self.expr_has_return(a) || self.expr_has_return(b)
            }
            Expr::Call(a, list) | Expr::Method { receiver: a, params: list, .. } => {
                self.expr_has_return(a) || self.body.list(list).iter().any(|&id| self.expr_has_return(ExprId(id)))
            }
            Expr::Tuple(list) | Expr::Array(list) => {
                self.body.list(list).iter().any(|&id| self.expr_has_return(ExprId(id)))
            }
            Expr::Struct { fields, base, .. } => {
                self.body.list(fields).iter().any(|&id| self.expr_has_return(self.body.fields[id as usize].value))
                    || base.is_some_and(|b| self.expr_has_return(b))
            }
            Expr::Block { kind, stmts, tail, .. } => {
                !matches!(kind, BlockKind::Async | BlockKind::AsyncMove)
                    && (self.body.list(stmts).iter().any(|&id| match self.body.stmts[id as usize] {
                        Stmt::Let { init, otherwise, .. } => {
                            init.is_some_and(|e| self.expr_has_return(e))
                                || otherwise.is_some_and(|e| self.expr_has_return(e))
                        }
                        Stmt::Expr(e, _) => self.expr_has_return(e),
                        Stmt::Item(..) | Stmt::Macro { .. } | Stmt::Empty => false,
                    }) || tail.is_some_and(|t| self.expr_has_return(t)))
            }
            Expr::If(c, t, e) => {
                self.expr_has_return(c) || self.expr_has_return(t) || e.is_some_and(|alt| self.expr_has_return(alt))
            }
            Expr::Match(s, arms) => {
                self.expr_has_return(s)
                    || self.body.list(arms).iter().any(|&id| {
                        let arm = self.body.arms[id as usize];
                        arm.guard.is_some_and(|g| self.expr_has_return(g)) || self.expr_has_return(arm.body)
                    })
            }
            Expr::Range { from, to, .. } => {
                from.is_some_and(|f| self.expr_has_return(f)) || to.is_some_and(|t| self.expr_has_return(t))
            }
            Expr::Break(_, e) | Expr::Yield(e) => e.is_some_and(|e| self.expr_has_return(e)),
            Expr::Lit(..) | Expr::Path(_) | Expr::Continue(_) | Expr::Macro { .. } | Expr::Underscore | Expr::FormatArgs(_) | Expr::OffsetOf { .. } => false,
        }
    }

    pub(super) fn expr_uses_inner_borrow_local(&mut self, expr: ExprId, inner_locals: &HashSet<u32>) -> bool {
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
            Expr::Paren(e) | Expr::Unary(_, e) | Expr::Field(e, _) | Expr::TupleField(e, _) | Expr::Cast(e, _) | Expr::Ref { expr: e, .. } | Expr::Try(e) | Expr::Await(e) | Expr::Become(e) => {
                self.expr_uses_inner_borrow_local(e, inner_locals)
            }
            Expr::Binary(_, a, b) | Expr::Index(a, b) | Expr::Repeat(a, b) => {
                self.expr_uses_inner_borrow_local(a, inner_locals) || self.expr_uses_inner_borrow_local(b, inner_locals)
            }
            Expr::Call(a, list) | Expr::Method { receiver: a, params: list, .. } => {
                let ids: Vec<u32> = self.body.list(list).to_vec();
                self.expr_uses_inner_borrow_local(a, inner_locals)
                    || ids.into_iter().any(|id| self.expr_uses_inner_borrow_local(ExprId(id), inner_locals))
            }
            Expr::Tuple(list) | Expr::Array(list) => {
                let ids: Vec<u32> = self.body.list(list).to_vec();
                ids.into_iter().any(|id| self.expr_uses_inner_borrow_local(ExprId(id), inner_locals))
            }
            Expr::Struct { fields, base, .. } => {
                let vals: Vec<ExprId> = self.body.list(fields).iter().map(|&id| self.body.fields[id as usize].value).collect();
                vals.into_iter().any(|v| self.expr_uses_inner_borrow_local(v, inner_locals))
                    || base.is_some_and(|b| self.expr_uses_inner_borrow_local(b, inner_locals))
            }
            Expr::Lit(..) | Expr::Continue(_) | Expr::Underscore | Expr::FormatArgs(_) | Expr::OffsetOf { .. } => false,
            _ => true,
        }
    }
}

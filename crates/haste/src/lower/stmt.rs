use anyhow::{Result, bail};
use haste_codegen::Value;

use crate::body::{Expr, ExprId, List, Pat, PatId, Stmt, UnOp};
use crate::sema::infer::Res;

use super::moves::Root;
use super::place::{Base, Place};
use super::reloc::TyOrigin;
use super::{Lower, Repr, repr};

impl Lower<'_, '_> {
    pub(super) fn block(&mut self, stmts: List, tail: Option<ExprId>) -> Result<Option<Value>> {
        let mark = self.owned.len();
        for &stmt in self.body.list(stmts) {
            self.stmt(self.body.stmts[stmt as usize])?;
        }
        let value = match tail {
            Some(tail) => {
                let value = self.expr(tail)?;
                self.snapshot_before_drops(mark, tail, value)?
            }
            None => None,
        };
        self.end_scope(mark)?;
        Ok(value)
    }

    pub(super) fn expansion(&mut self, stmts: List, tail: Option<ExprId>) -> Result<Option<Value>> {
        for &stmt in self.body.list(stmts) {
            self.stmt(self.body.stmts[stmt as usize])?;
        }
        tail.map_or(Ok(None), |tail| self.expr(tail))
    }

    pub(super) fn stmt(&mut self, stmt: Stmt) -> Result<()> {
        let mark = self.owned.len();
        match stmt {
            Stmt::Let {
                pat,
                init,
                otherwise: None,
                ..
            } => self.let_stmt(pat, init)?,
            Stmt::Let {
                pat,
                init: Some(init),
                otherwise: Some(otherwise),
                ..
            } => {
                self.extend_let(pat, init);
                self.let_else(pat, init, otherwise)?;
            }
            Stmt::Expr(expr, _) => self.discard(expr)?,
            Stmt::Item(..) | Stmt::Empty => {}
            other => bail!("cannot lower statement {other:?} yet"),
        }
        self.end_statement(mark)?;
        if let Stmt::Let { extend: true, .. } = stmt {
            self.extend_owned(mark);
        }
        Ok(())
    }

    fn discard(&mut self, expr: ExprId) -> Result<()> {
        let value = self.expr(expr)?;
        if !self.open {
            return Ok(());
        }
        let ty = self.ty_of(expr);
        let repr = repr(self.cx, ty)?;
        let place = value.map(|value| Place {
            base: Base::Addr(value),
            offset: 0,
            repr,
        });
        match (repr, place) {
            (Repr::Scalar(_), _) => Ok(()),
            (_, place) => self.glue(ty, place, TyOrigin::Expr(expr)),
        }
    }

    fn let_stmt(&mut self, pat: PatId, init: Option<ExprId>) -> Result<()> {
        let Some(init) = init else {
            return self.declare_binding(pat, false);
        };
        if self.open
            && self.uninit_call(init)
            && self.can_declare_uninit_val(pat)
            && !self.typed.coercions.contains_key(&init)
        {
            return self.declare_binding(pat, true);
        }
        self.extend_let(pat, init);
        let unboxed = self.moves_whole_box(pat, init)?;
        let place = if unboxed {
            self.temp(init)?
        } else {
            self.place_or_temp(init)?
        };
        if !self.open {
            return Ok(());
        }
        let path = if unboxed {
            Some((Root::Temp(init), Vec::new()))
        } else {
            self.scrutinee_path(init)
        };
        self.bind_matched(pat, place, path)?;
        self.commit_moves()
    }

    fn can_declare_uninit_val(&self, pat: PatId) -> bool {
        match *self.body.pat(pat) {
            Pat::Bind {
                sub: None,
                by_ref: false,
                ..
            } => !self.typed.pat_res.contains_key(&pat) && !self.typed.ref_binds.contains_key(&pat),
            Pat::Wild => true,
            Pat::Paren(inner) => self.can_declare_uninit_val(inner),
            _ => false,
        }
    }

    fn declare_binding(&mut self, pat: PatId, init: bool) -> Result<()> {
        match *self.body.pat(pat) {
            Pat::Bind { sub: None, .. } => match self.typed.binds.get(&pat) {
                Some(&local) => self.declare_as(local, None, init),
                None => bail!("an uninitialized binding with no local"),
            },
            Pat::Wild => Ok(()),
            Pat::Paren(inner) => self.declare_binding(inner, init),
            Pat::Tuple(items) => self
                .body
                .list(items)
                .to_vec()
                .into_iter()
                .try_for_each(|item| self.declare_binding(PatId(item), init)),
            _ => bail!("uninitialized `let` with a pattern"),
        }
    }

    fn extend_let(&mut self, pat: PatId, init: ExprId) {
        self.extend(init);
        if self.pat_binds_ref(pat) {
            self.extend_place(init);
        }
    }

    fn pat_binds_ref(&self, pat: PatId) -> bool {
        match *self.body.pat(pat) {
            Pat::Bind { by_ref, sub, .. } => {
                by_ref
                    || self.typed.ref_binds.contains_key(&pat)
                    || sub.is_some_and(|sub| self.pat_binds_ref(sub))
            }
            Pat::Paren(inner) => self.pat_binds_ref(inner),
            Pat::Tuple(list) | Pat::TupleStruct(_, list) | Pat::Slice(list) | Pat::Or(list) => self
                .body
                .list(list)
                .iter()
                .any(|&at| self.pat_binds_ref(PatId(at))),
            Pat::Struct { fields, .. } => self
                .body
                .list(fields)
                .iter()
                .any(|&at| self.pat_binds_ref(self.body.field_pats[at as usize].pat)),
            _ => false,
        }
    }

    fn extend_place(&mut self, mut expr: ExprId) {
        loop {
            self.extended.insert(expr);
            match *self.body.expr(expr) {
                Expr::Paren(inner)
                | Expr::Field(inner, _)
                | Expr::TupleField(inner, _)
                | Expr::Index(inner, _)
                | Expr::Unary(UnOp::Deref, inner) => expr = inner,
                Expr::Block {
                    stmts,
                    tail: Some(tail),
                    ..
                } if self.body.list(stmts).is_empty() => {
                    expr = tail;
                }
                _ => break,
            }
        }
    }

    fn extend(&mut self, expr: ExprId) {
        match *self.body.expr(expr) {
            Expr::Ref { expr: inner, .. } => {
                self.extend_place(inner);
                self.extend(inner);
            }
            Expr::Paren(inner) | Expr::Cast(inner, _) => self.extend(inner),
            Expr::Block {
                tail: Some(tail), ..
            } => self.extend(tail),
            Expr::If(_, then, otherwise) => {
                self.extend(then);
                if let Some(otherwise) = otherwise {
                    self.extend(otherwise);
                }
            }
            Expr::Match(_, arms) => self
                .body
                .list(arms)
                .to_vec()
                .into_iter()
                .for_each(|at| self.extend(self.body.arms[at as usize].body)),
            Expr::Call(callee, args)
                if self.body.names_callee(callee)
                    && matches!(self.typed.res.get(&callee), Some(Res::Ctor(..))) =>
            {
                self.body
                    .list(args)
                    .to_vec()
                    .into_iter()
                    .for_each(|arg| self.extend(ExprId(arg)));
            }
            Expr::Tuple(list) | Expr::Array(list) => self
                .body
                .list(list)
                .to_vec()
                .into_iter()
                .for_each(|item| self.extend(ExprId(item))),
            Expr::Struct { fields, .. } => self
                .body
                .list(fields)
                .to_vec()
                .into_iter()
                .for_each(|at| self.extend(self.body.fields[at as usize].value)),
            _ => {}
        }
    }
}

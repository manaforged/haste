use anyhow::{Result, bail};

use crate::body::{Expr, ExprId, List, PatId};
use crate::names::plain;

use super::super::ty::{Kind, Ty};
use super::Infer;

impl Infer<'_, '_> {
    /// Infers struct literals, field reads, `match` and `let` conditions.
    pub fn data(&mut self, expr: ExprId, expected: Option<Ty>) -> Result<Ty> {
        match *self.body.expr(expr) {
            Expr::Struct { path, fields, base, .. } => self.struct_expr(expr, path, fields, base),
            Expr::Field(base, name) => self.field_expr(expr, base, plain(self.src.text(name))),
            Expr::Match(scrutinee, arms) => self.match_expr(scrutinee, arms, expected),
            Expr::Let(pat, init) => self.let_expr(pat, init),
            Expr::Array(items) => self.array_expr(items, expected),
            Expr::Repeat(value, len) => self.repeat_expr(value, len, expected),
            Expr::Index(base, index) => self.index_expr(expr, base, index),
            Expr::For { label, pat, iter, body } => self.for_expr(expr, pat, iter, (label, body)),
            Expr::Range { from, to, inclusive } => self.range_expr(expr, (from, to), inclusive),
            Expr::OffsetOf { ty, fields } => self.offset_of(expr, ty, fields),
            ref other => bail!("unsupported expression {other:?}"),
        }
    }

    fn match_expr(&mut self, scrutinee: ExprId, arms: List, expected: Option<Ty>) -> Result<Ty> {
        let ty = self.expr(scrutinee, None)?;
        let before = std::mem::replace(&mut self.diverges, false);
        let result = self.known(expected).unwrap_or_else(|| self.fresh(Kind::Var));
        let mut all_diverge = true;
        for &at in self.body.list(arms) {
            let arm = self.body.arms[at as usize];
            let depth = self.scopes.len();
            self.bind_pat(arm.pat, ty)?;
            if let Some(guard) = arm.guard {
                let bool_ty = self.cx.types.bool();
                let found = self.expr(guard, Some(bool_ty))?;
                self.coerce(found, bool_ty)?;
            }
            self.diverges = false;
            let found = self.expr(arm.body, Some(result))?;
            self.coerce_expr(arm.body, found, result)?;
            all_diverge &= self.diverges;
            self.scopes.truncate(depth);
        }
        self.diverges = before || all_diverge;
        if all_diverge {
            return Ok(self.cx.types.intern(Kind::Never));
        }
        Ok(result)
    }

    fn let_expr(&mut self, pat: PatId, init: ExprId) -> Result<Ty> {
        let ty = self.expr(init, None)?;
        self.bind_pat(pat, ty)?;
        Ok(self.cx.types.bool())
    }
}

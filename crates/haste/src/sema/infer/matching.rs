use anyhow::{Result, bail};

use crate::body::{Arm, Expr, ExprId, List, PatId};
use crate::names::plain;

use super::super::ty::{Kind, Ty};
use super::Infer;

impl Infer<'_, '_> {
    /// Infers struct literals, field reads, `match` and `let` conditions.
    pub fn data(&mut self, expr: ExprId, expected: Option<Ty>) -> Result<Ty> {
        match *self.body.expr(expr) {
            Expr::Struct { path, fields, base, .. } => self.struct_expr(expr, path, (fields, base), expected),
            Expr::Field(base, name) => self.field_expr(expr, base, plain(self.src.text(name))),
            Expr::Match(scrutinee, arms) => self.match_expr(scrutinee, arms, expected),
            Expr::Let(pat, init) => self.let_expr(pat, init),
            Expr::Array(items) => self.array_expr(items, expected),
            Expr::Repeat(value, len) => self.repeat_expr(value, len, expected),
            Expr::Index(base, index) => self.index_expr(expr, base, index),
            Expr::For { label, pat, iter, body } => self.for_expr(expr, pat, iter, (label, body)),
            Expr::Range { from, to, inclusive } => self.range_expr(expr, (from, to), inclusive),
            Expr::OffsetOf { ty, fields } => self.offset_of(expr, ty, fields),
            Expr::Underscore => Ok(expected.unwrap_or_else(|| self.fresh(Kind::Var))),
            ref other => bail!("unsupported expression {other:?}"),
        }
    }

    fn match_expr(&mut self, scrutinee: ExprId, arms: List, expected: Option<Ty>) -> Result<Ty> {
        let scrutinee_ty = self.expr(scrutinee, None)?;
        let before = std::mem::replace(&mut self.diverges, false);
        if let Some(target) = self.known(expected) {
            return self.match_known(scrutinee_ty, arms, target, before);
        }
        let (found_arms, all_diverge) = self.infer_arms(scrutinee_ty, arms, expected)?;
        self.diverges = before || all_diverge;
        if all_diverge {
            return Ok(self.cx.types.intern(Kind::Never));
        }
        let target = self.merge_arms(&found_arms, expected)?;
        for &(body, found) in &found_arms {
            self.coerce_expr(body, found, target)?;
        }
        if let Some(exp) = expected {
            self.unify(exp, target)?;
        }
        Ok(target)
    }

    fn match_known(&mut self, scrutinee_ty: Ty, arms: List, target: Ty, before: bool) -> Result<Ty> {
        let mut all_diverge = true;
        for &at in self.body.list(arms) {
            let arm = self.body.arms[at as usize];
            let (found, diverges) = self.match_arm(arm, scrutinee_ty, Some(target))?;
            self.coerce_expr(arm.body, found, target)?;
            all_diverge &= diverges;
        }
        self.diverges = before || all_diverge;
        Ok(if all_diverge { self.cx.types.intern(Kind::Never) } else { target })
    }

    fn infer_arms(&mut self, scrutinee_ty: Ty, arms: List, expected: Option<Ty>) -> Result<(Vec<(ExprId, Ty)>, bool)> {
        let mut hint = None;
        let mut all_diverge = true;
        let mut found_arms = Vec::new();
        for &at in self.body.list(arms) {
            let arm = self.body.arms[at as usize];
            let (found, diverges) = self.match_arm(arm, scrutinee_ty, expected.or(hint))?;
            all_diverge &= diverges;
            if !self.is_never(found) {
                hint = Some(self.next_arm_hint(hint, found)?);
            }
            found_arms.push((arm.body, found));
        }
        Ok((found_arms, all_diverge))
    }

    fn next_arm_hint(&mut self, hint: Option<Ty>, found: Ty) -> Result<Ty> {
        let Some(prev) = hint else { return Ok(found) };
        Ok(self.branch_target(prev, found)?.or_else(|| self.probe(|infer| infer.join(prev, found).ok())).unwrap_or(prev))
    }

    fn merge_arms(&mut self, found_arms: &[(ExprId, Ty)], expected: Option<Ty>) -> Result<Ty> {
        let mut target = None;
        for &(_, found) in found_arms {
            if !self.is_never(found) {
                target = Some(match target {
                    None => found,
                    Some(acc) => self.merge_branch(acc, found)?,
                });
            }
        }
        Ok(target.or(expected).unwrap_or_else(|| self.fresh(Kind::Var)))
    }

    fn merge_branch(&mut self, a: Ty, b: Ty) -> Result<Ty> {
        match self.branch_target(a, b)? {
            Some(common) => Ok(common),
            None => self.join(a, b),
        }
    }

    fn match_arm(&mut self, arm: Arm, scrutinee_ty: Ty, expected: Option<Ty>) -> Result<(Ty, bool)> {
        let depth = self.scopes.len();
        self.bind_pat(arm.pat, scrutinee_ty)?;
        if let Some(guard) = arm.guard {
            let bool_ty = self.cx.types.bool();
            let found = self.expr(guard, Some(bool_ty))?;
            self.coerce(found, bool_ty)?;
        }
        self.diverges = false;
        let found = self.expr(arm.body, expected);
        self.scopes.truncate(depth);
        Ok((found?, self.diverges))
    }

    fn let_expr(&mut self, pat: PatId, init: ExprId) -> Result<Ty> {
        let ty = self.expr(init, None)?;
        self.bind_pat(pat, ty)?;
        Ok(self.cx.types.bool())
    }
}

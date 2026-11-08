use anyhow::{Result, bail};

use crate::body::{ExprId, List, Pat, PatId, Stmt};

use super::super::ty::{Kind, Ty};
use super::Infer;

impl Infer<'_, '_> {
    /// Infers a block's statements and tail in a new scope.
    pub fn block(&mut self, stmts: List, tail: Option<ExprId>, expected: Option<Ty>) -> Result<Ty> {
        let depth = self.scopes.len();
        let diverges = std::mem::replace(&mut self.diverges, false);
        let result = self.block_inner(stmts, tail, expected);
        self.scopes.truncate(depth);
        let ty = result?;
        let ty = if tail.is_none() && self.diverges { self.cx.types.intern(Kind::Never) } else { ty };
        self.diverges |= diverges;
        Ok(ty)
    }

    fn block_inner(&mut self, stmts: List, tail: Option<ExprId>, expected: Option<Ty>) -> Result<Ty> {
        for &stmt in self.body.list(stmts) {
            self.stmt(self.body.stmts[stmt as usize])?;
        }
        match tail {
            Some(tail) => self.expr(tail, expected),
            None => Ok(self.cx.types.unit()),
        }
    }

    fn stmt(&mut self, stmt: Stmt) -> Result<()> {
        match stmt {
            Stmt::Let { pat, ty, init, otherwise: None } => self.let_stmt(pat, ty, init),
            Stmt::Expr(expr, semi) => {
                let ty = self.expr(expr, None)?;
                if !semi {
                    let unit = self.cx.types.unit();
                    self.coerce(ty, unit)?;
                }
                Ok(())
            }
            Stmt::Item(..) | Stmt::Empty => Ok(()),
            other => bail!("unsupported statement {other:?}"),
        }
    }

    fn let_stmt(&mut self, pat: PatId, ty: Option<crate::body::TyId>, init: Option<ExprId>) -> Result<()> {
        let declared = match ty {
            Some(ty) => Some(self.lower_ty(self.body, ty)?),
            None => None,
        };
        let ty = match (init, declared) {
            (Some(init), _) => {
                let found = self.expr(init, declared)?;
                match declared {
                    Some(declared) => self.coerce(found, declared).map(|_| declared)?,
                    None => found,
                }
            }
            (None, Some(declared)) => declared,
            (None, None) => self.fresh(Kind::Var),
        };
        self.bind_pat(pat, ty)
    }

    /// Binds the names in pattern `pat` of the body to parts of a value of type `ty`.
    pub fn bind_pat(&mut self, pat: PatId, ty: Ty) -> Result<()> {
        self.out.pats.insert(pat, ty);
        match *self.body.pat(pat) {
            Pat::Bind { name, mutable, by_ref: false, sub: None } => {
                let local = self.new_local(ty, mutable);
                self.out.binds.insert(pat, local);
                self.scopes.push((crate::names::plain(self.src.text(name)), local));
                Ok(())
            }
            Pat::Wild => Ok(()),
            Pat::Paren(inner) => self.bind_pat(inner, ty),
            Pat::Tuple(list) => self.tuple_pat(list, ty),
            ref other => bail!("unsupported pattern {other:?}"),
        }
    }

    fn tuple_pat(&mut self, list: crate::body::List, ty: Ty) -> Result<()> {
        let pats = self.body.list(list).to_vec();
        if pats.iter().any(|&pat| matches!(self.body.pat(PatId(pat)), Pat::Rest)) {
            bail!("`..` in tuple patterns is not supported yet");
        }
        let items: Vec<Ty> = (0..pats.len()).map(|_| self.fresh(Kind::Var)).collect();
        let tuple = self.cx.types.intern(Kind::Tuple(items.clone()));
        self.unify(ty, tuple)?;
        for (pat, item) in pats.into_iter().zip(items) {
            self.bind_pat(PatId(pat), item)?;
        }
        Ok(())
    }
}

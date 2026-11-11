use anyhow::{Result, bail};

use crate::body::{ExprId, List, PatId, Stmt};

use super::super::ty::{Kind, Ty};
use super::Infer;

impl Infer<'_, '_> {
    /// Infers a block's statements and tail in a new scope.
    pub fn block(&mut self, stmts: List, tail: Option<ExprId>, expected: Option<Ty>) -> Result<Ty> {
        let (depth, items) = (self.scopes.len(), self.items.len());
        let diverges = std::mem::replace(&mut self.diverges, false);
        self.declare_items(stmts);
        let result = self.block_inner(stmts, tail, expected);
        self.scopes.truncate(depth);
        self.items.truncate(items);
        let ty = result?;
        let ty = if tail.is_none() && self.diverges { self.cx.types.intern(Kind::Never) } else { ty };
        self.diverges |= diverges;
        Ok(ty)
    }

    fn declare_items(&mut self, stmts: List) {
        let Some((root, found)) = self.root.and_then(|root| Some((root, self.cx.expansion(root)?))) else {
            return;
        };
        let (src, items) = (&found.file.source, &found.file.items.items);
        for &stmt in self.body.list(stmts) {
            let Stmt::Item(start, end) = self.body.stmts[stmt as usize] else {
                continue;
            };
            for at in super::super::expand::stmt_items(items, (start, end)) {
                if let Some(name) = items[at].name {
                    self.items.push((crate::names::plain(src.text(name)), root.inner(at as u32)));
                }
            }
        }
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
            Stmt::Let { pat, ty, init, otherwise, .. } => self.let_stmt(pat, ty, init, otherwise),
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

    fn let_stmt(&mut self, pat: PatId, ty: Option<crate::body::TyId>, init: Option<ExprId>, otherwise: Option<ExprId>) -> Result<()> {
        let declared = match ty {
            Some(ty) => Some(self.lower_ty(self.body, ty)?),
            None => None,
        };
        let ty = match (init, declared) {
            (Some(init), _) => {
                let found = self.expr(init, declared)?;
                match declared {
                    Some(declared) => self.coerce_expr(init, found, declared).map(|_| declared)?,
                    None => found,
                }
            }
            (None, Some(declared)) => declared,
            (None, None) => self.fresh(Kind::Var),
        };
        if let Some(otherwise) = otherwise {
            let before = self.diverges;
            self.expr(otherwise, None)?;
            self.diverges = before;
        }
        self.bind_pat(pat, ty)
    }
}

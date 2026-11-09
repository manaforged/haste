use anyhow::Result;
use rustc_hash::FxHashMap;

use crate::body::{Expr, ExprId, Pat, PatId, Stmt, UnOp};
use crate::sema::infer::{Res, needs_drop};
use crate::sema::ty::Kind;

use super::Lower;

/// What owns a value that the body drops: a local, or the temporary that holds the value of an expression.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Root {
    Local(u32),
    Temp(ExprId),
}

/// The fields from a root down to a part of its value, each as a variant and a field index.
pub type MovePath = Vec<(u32, u32)>;

impl Lower<'_, '_> {
    /// The root and path of the place expression `expr`, or none when it reads through a pointer or names no local.
    pub fn move_path(&self, expr: ExprId) -> Option<(Root, MovePath)> {
        let (base, index) = match *self.body.expr(expr) {
            Expr::Paren(inner) => return self.move_path(inner),
            Expr::Path(_) => return match self.typed.res.get(&expr) {
                Some(Res::Local(local)) => Some((Root::Local(*local), Vec::new())),
                _ => None,
            },
            Expr::Field(base, _) => (base, *self.typed.fields.get(&expr)?),
            Expr::TupleField(base, index) => (base, index),
            _ => return None,
        };
        if self.typed.exprs[base.0 as usize].0 == u32::MAX || matches!(self.cx.types.kind(self.ty_of(base)), Kind::Ref(..) | Kind::Ptr(..)) {
            return None;
        }
        let (root, mut path) = self.move_path(base).or_else(|| (!self.is_place(base)).then(|| (Root::Temp(base), Vec::new())))?;
        path.push((0, index));
        Some((root, path))
    }

    /// The root and path whose value a pattern matched against `scrutinee` takes parts of: the temporary that holds a coerced value.
    pub fn scrutinee_path(&self, scrutinee: ExprId) -> Option<(Root, MovePath)> {
        if self.typed.coercions.contains_key(&scrutinee) {
            return Some((Root::Temp(scrutinee), Vec::new()));
        }
        self.move_path(scrutinee).or_else(|| (!self.is_place(scrutinee)).then(|| (Root::Temp(scrutinee), Vec::new())))
    }

    fn is_place(&self, expr: ExprId) -> bool {
        match *self.body.expr(expr) {
            Expr::Paren(inner) => self.is_place(inner),
            Expr::Path(_) => match self.typed.res.get(&expr) {
                Some(Res::Local(_)) => true,
                Some(Res::Fn(item, _)) => super::is_static(self.cx, *item),
                _ => false,
            },
            Expr::Field(..) | Expr::TupleField(..) | Expr::Index(..) | Expr::Unary(UnOp::Deref, _) => true,
            _ => false,
        }
    }

    /// Every path below a root that the body can move out of: fields read by value and parts that patterns bind by value.
    pub fn moved_paths(&mut self) -> Result<FxHashMap<Root, Vec<MovePath>>> {
        let mut found: FxHashMap<Root, Vec<MovePath>> = FxHashMap::default();
        for at in 0..self.body.exprs.len() as u32 {
            let expr = ExprId(at);
            let mut each = |lower: &mut Self, path: Option<(Root, MovePath)>| lower.add_path(&mut found, path);
            match *self.body.expr(expr) {
                Expr::Field(..) | Expr::TupleField(..) if self.typed.exprs[at as usize].0 != u32::MAX && needs_drop(self.cx, self.ty_of(expr))? => each(self, self.move_path(expr)),
                Expr::Match(scrutinee, arms) => {
                    for arm in self.body.list(arms).to_vec() {
                        let pat = self.body.arms[arm as usize].pat;
                        self.pattern_paths(pat, self.scrutinee_path(scrutinee), &mut found)?;
                    }
                }
                Expr::Let(pat, init) => self.pattern_paths(pat, self.scrutinee_path(init), &mut found)?,
                _ => {}
            }
        }
        self.binding_paths(&mut found)?;
        Ok(found)
    }

    fn binding_paths(&mut self, found: &mut FxHashMap<Root, Vec<MovePath>>) -> Result<()> {
        for stmt in self.body.stmts.clone() {
            if let Stmt::Let { pat, init: Some(init), .. } = stmt {
                self.pattern_paths(pat, self.scrutinee_path(init), found)?;
            }
        }
        for (&local, &pat) in &self.typed.unpacks.clone() {
            self.pattern_paths(pat, Some((Root::Local(local), Vec::new())), found)?;
        }
        Ok(())
    }

    fn add_path(&self, found: &mut FxHashMap<Root, Vec<MovePath>>, path: Option<(Root, MovePath)>) {
        if let Some((root, path)) = path.filter(|(_, path)| !path.is_empty()) {
            let paths = found.entry(root).or_default();
            if !paths.contains(&path) {
                paths.push(path);
            }
        }
    }

    fn pattern_paths(&mut self, pat: PatId, at: Option<(Root, MovePath)>, found: &mut FxHashMap<Root, Vec<MovePath>>) -> Result<()> {
        let Some(at) = at.filter(|_| self.typed.pats.contains_key(&pat)) else { return Ok(()) };
        if self.typed.pat_derefs.get(&pat).copied().unwrap_or(0) > 0 {
            return Ok(());
        }
        match *self.body.pat(pat) {
            Pat::Bind { by_ref: false, .. } if !self.typed.pat_res.contains_key(&pat) && !self.typed.ref_binds.contains_key(&pat) => {
                if needs_drop(self.cx, self.pat_ty(pat)?)? {
                    self.add_path(found, Some(at));
                }
                Ok(())
            }
            Pat::Paren(inner) => self.pattern_paths(inner, Some(at), found),
            Pat::Or(list) => self.body.list(list).to_vec().into_iter().try_for_each(|alt| self.pattern_paths(PatId(alt), Some(at.clone()), found)),
            Pat::Tuple(_) | Pat::TupleStruct(..) | Pat::Struct { .. } => {
                for (sub, step) in self.part_indices(pat)? {
                    let mut path = at.1.clone();
                    path.push(step);
                    self.pattern_paths(sub, Some((at.0, path)), found)?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

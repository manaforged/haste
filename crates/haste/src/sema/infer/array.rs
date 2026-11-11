use anyhow::{Result, bail};

use crate::body::{ExprId, List};
use crate::names::Primitive;

use super::super::ty::{Kind, Ty};
use super::{Infer, Res};

impl Infer<'_, '_> {
    /// Infers the array expression `[items]`, coercing each item to the element type of an expected array or slice.
    pub fn array_expr(&mut self, items: List, expected: Option<Ty>) -> Result<Ty> {
        let element = match expected.map(|ty| self.cx.types.kind(self.resolve(ty)).clone()) {
            Some(Kind::Array(element, _) | Kind::Slice(element)) => element,
            _ => self.fresh(Kind::Var),
        };
        let items = self.body.list(items).to_vec();
        for &item in &items {
            let found = self.expr(ExprId(item), Some(element))?;
            self.coerce_expr(ExprId(item), found, element)?;
        }
        Ok(self.cx.types.array(element, items.len() as u64))
    }

    /// Infers the repeat expression `[value; len]`.
    pub fn repeat_expr(&mut self, value: ExprId, len: ExprId, expected: Option<Ty>) -> Result<Ty> {
        let element = match expected.map(|ty| self.cx.types.kind(self.resolve(ty)).clone()) {
            Some(Kind::Array(element, _)) => Some(element),
            _ => None,
        };
        let found = self.expr(value, element)?;
        let element = match element {
            Some(element) => {
                self.coerce_expr(value, found, element)?;
                element
            }
            None => found,
        };
        let count = self.const_arg(self.body, len)?;
        Ok(self.cx.types.intern(Kind::Array(element, count)))
    }

    /// Infers the index `expr` of `base` by `index`, through the autoderef steps of `base`: the builtin index of an array or slice by an integer, or `Index::index`.
    pub fn index_expr(&mut self, expr: ExprId, base: ExprId, index: ExprId) -> Result<Ty> {
        let found = self.expr(base, None)?;
        let at = self.expr(index, None)?;
        let (mut step, mut steps) = (self.resolve(found), 0);
        loop {
            if let Some(ty) = self.index_step(expr, step, at)? {
                if steps > 0 {
                    self.out.autoderefs.insert(expr, steps);
                }
                return Ok(ty);
            }
            let next = match self.cx.types.kind(step) {
                Kind::Ptr(..) | Kind::Var(_) => None,
                _ => match self.builtin_deref(step) {
                    Some(inner) => Some(inner),
                    None => self.deref_step(expr, steps, step)?,
                },
            };
            let Some(next) = next else { bail!("cannot index {} by {}", self.show(found), self.show(at)) };
            (step, steps) = (self.resolve(next), steps + 1);
            if steps as usize == crate::RECURSION_LIMIT {
                bail!("reached the recursion limit while dereferencing {}", self.show(found));
            }
        }
    }

    fn index_step(&mut self, expr: ExprId, step: Ty, at: Ty) -> Result<Option<Ty>> {
        let usize_ty = self.cx.types.prim(Primitive::Usize);
        if let Kind::Array(element, _) | Kind::Slice(element) = *self.cx.types.kind(step)
            && matches!(self.cx.types.kind(self.resolve(at)), Kind::IntVar(_) | Kind::Prim(Primitive::Usize))
        {
            self.coerce(at, usize_ty)?;
            return Ok(Some(element));
        }
        let Some(of) = self.cx.lang_item("index") else { bail!("core has no `Index` trait") };
        if self.trial(|infer| infer.select(of, step, &[at]))?.is_none() {
            return Ok(None);
        }
        let Some(method) = self.cx.trait_member(of, "index") else { bail!("`Index` has no `index`") };
        self.out.operators.insert(expr, Res::fn_of(&self.cx.types, method, &[step, at]));
        self.project(of, step, vec![at], "Output").map(Some)
    }
}

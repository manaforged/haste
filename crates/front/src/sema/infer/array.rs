use anyhow::{Result, bail};

use crate::body::{ExprId, List};
use crate::names::Primitive;

use super::super::ty::{Kind, Ty};
use super::Infer;

impl Infer<'_, '_> {
    /// Infers the array expression `[items]`.
    pub fn array_expr(&mut self, items: List, expected: Option<Ty>) -> Result<Ty> {
        let element = match expected.map(|ty| self.cx.types.kind(self.resolve(ty)).clone()) {
            Some(Kind::Array(element, _)) => element,
            _ => self.fresh(Kind::Var),
        };
        let items = self.body.list(items).to_vec();
        for &item in &items {
            let found = self.expr(ExprId(item), Some(element))?;
            self.coerce(found, element)?;
        }
        Ok(self.cx.types.intern(Kind::Array(element, items.len() as u64)))
    }

    /// Infers the repeat expression `[value; len]`.
    pub fn repeat_expr(&mut self, value: ExprId, len: ExprId, expected: Option<Ty>) -> Result<Ty> {
        let element = match expected.map(|ty| self.cx.types.kind(self.resolve(ty)).clone()) {
            Some(Kind::Array(element, _)) => Some(element),
            _ => None,
        };
        let found = self.expr(value, element)?;
        let count = self.const_len(self.body, len)?;
        Ok(self.cx.types.intern(Kind::Array(found, count)))
    }

    /// Infers the builtin index `base[index]` of an array or slice, through references.
    pub fn index_expr(&mut self, base: ExprId, index: ExprId) -> Result<Ty> {
        let found = self.expr(base, None)?;
        let usize_ty = self.cx.types.prim(Primitive::Usize);
        let at = self.expr(index, Some(usize_ty))?;
        let element = match self.cx.types.kind(self.autoderef(found)).clone() {
            Kind::Array(element, _) | Kind::Slice(element) => element,
            _ => bail!("indexing {} through the Index trait is not supported yet", self.show(found)),
        };
        self.coerce(at, usize_ty)?;
        Ok(element)
    }
}

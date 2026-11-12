use anyhow::{Result, bail};

use crate::body::{BinOp, ExprId};

use super::super::ty::{Kind, Ty};
use super::expr::operator_trait;
use super::{Infer, Res};

impl Infer<'_, '_> {
    /// Types `lhs = rhs`, or `lhs op= rhs` when `op` is given.
    pub(super) fn assign(&mut self, expr: ExprId, (lhs, rhs): (ExprId, ExprId), op: Option<BinOp>) -> Result<Ty> {
        let place = self.expr(lhs, None)?;
        self.note_mut(lhs)?;
        match op {
            Some(bin) if matches!(self.cx.types.kind(self.resolve(place)), Kind::Var(_)) => {
                let value = self.expr(rhs, None)?;
                let unit = self.cx.types.unit();
                self.operands.push((expr, bin, (place, value), unit));
            }
            Some(bin) if !self.builtin_compound(bin, place) => self.overloaded_assign(expr, bin, (place, rhs))?,
            Some(BinOp::Shl | BinOp::Shr) => drop(self.expr(rhs, None)?),
            Some(bin) => {
                let value = self.expr(rhs, Some(place))?;
                self.compound(expr, bin, (place, rhs, value))?;
            }
            None => {
                let value = self.expr(rhs, Some(place))?;
                self.coerce_expr(rhs, value, place)?;
            }
        }
        Ok(self.cx.types.unit())
    }

    /// Types `place op= rhs`, where `rhs` has type `value`, as the built-in operator or through the impl of the operator's assign trait.
    pub(super) fn compound(&mut self, expr: ExprId, op: BinOp, (place, rhs, value): (Ty, ExprId, Ty)) -> Result<()> {
        if !self.builtin_compound(op, place) {
            let right = self.assign_right(op, place)?;
            return self.overloaded_value(expr, op, (place, rhs, value), right);
        }
        match (op, self.cx.types.kind(self.resolve(value))) {
            (BinOp::Shl | BinOp::Shr, _) => Ok(()),
            (_, Kind::Ref(..) | Kind::Adt(..)) => self.assign_impl(expr, op, (place, value)),
            _ => self.coerce_expr(rhs, value, place),
        }
    }

    /// Types `place op= rhs` through the operator's assign trait, coercing `rhs` to the right operand type when one impl fixes it, as rustc does.
    fn overloaded_assign(&mut self, expr: ExprId, op: BinOp, (place, rhs): (Ty, ExprId)) -> Result<()> {
        let right = self.assign_right(op, place)?;
        let found = self.expr(rhs, right)?;
        self.overloaded_value(expr, op, (place, rhs, found), right)
    }

    /// The right operand type of the assign trait of `op` when one impl of it applies to `place`.
    fn assign_right(&mut self, op: BinOp, place: Ty) -> Result<Option<Ty>> {
        let (lang, _) = assign_trait(op)?;
        Ok(self.cx.lang_item(&lang).and_then(|of| self.only_right(of, place)))
    }

    /// Types `place op= rhs`, where `rhs` has type `found`, through the operator's assign trait, coercing `rhs` to `right` when one impl fixes it.
    fn overloaded_value(&mut self, expr: ExprId, op: BinOp, (place, rhs, found): (Ty, ExprId, Ty), right: Option<Ty>) -> Result<()> {
        let value = match right {
            Some(want) => self.coerce_expr(rhs, found, want).map(|()| want)?,
            None => found,
        };
        self.assign_impl(expr, op, (place, value))
    }

    /// Types `place op= value` through the impl of the operator's assign trait for `place` and `value`.
    fn assign_impl(&mut self, expr: ExprId, op: BinOp, (place, value): (Ty, Ty)) -> Result<()> {
        let (lang, name) = assign_trait(op)?;
        let Some((of, method)) = self.operator_impl((&lang, &name), place, &[value])? else { bail!("unsupported compound assignment on {}", self.show(place)) };
        if self.has_vars(value) {
            self.obligations.push((of, vec![place, value]));
        }
        self.out.operators.insert(expr, Res::fn_of(&self.cx.types, method, &[place, value]));
        Ok(())
    }

    /// Whether `place op= ..` is the built-in operator on a scalar place.
    pub(super) fn builtin_compound(&mut self, op: BinOp, place: Ty) -> bool {
        let logical = matches!(op, BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor);
        self.is_int(place) || self.is_float(place) || logical && self.resolve(place) == self.cx.types.bool()
    }
}

/// The lang item of the trait that overloads `op=`, and its method.
fn assign_trait(op: BinOp) -> Result<(String, String)> {
    let Some((lang, name)) = operator_trait(op) else { bail!("{op:?} has no operator trait") };
    Ok((format!("{lang}_assign"), format!("{name}_assign")))
}

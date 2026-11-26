use anyhow::{Result, bail};
use haste_codegen::{Type, Value};

use crate::body::{BinOp, Expr, ExprId};
use crate::names::ItemRef;
use crate::sema::infer::{Adjust, Res};
use crate::sema::substs::Substs;
use crate::sema::ty::Mut;

use super::super::Lower;

impl Lower<'_, '_> {
    pub(in crate::lower) fn assign(
        &mut self,
        expr: ExprId,
        (lhs, rhs): (ExprId, ExprId),
        op: Option<BinOp>,
    ) -> Result<Option<Value>> {
        if let Some(op) = op {
            return self.compound(expr, (lhs, rhs), op);
        }
        if matches!(self.body.expr(lhs), Expr::Tuple(_) | Expr::Underscore) {
            let value = self.expr(rhs)?;
            return self.assign_parts(lhs, value).map(|_| None);
        }
        let value = self.eval_operand(rhs)?;
        if !self.open {
            return Ok(None);
        }
        let mut guards = Vec::new();
        if self.wants_field_unwind()? {
            self.push_arg_guard(rhs, value, &mut guards)?;
        }
        let Some(place) = self.place_of(lhs)? else {
            if value.is_some() {
                bail!("cannot lower assignment to this expression yet");
            }
            self.disarm_owned_guards(guards);
            return Ok(None);
        };
        if !self.open {
            return Ok(None);
        }
        match value {
            Some(value) => {
                if !self.replace(lhs, place, value)? {
                    self.write(place, value);
                }
            }
            None if crate::sema::infer::needs_drop(self.cx, self.ty_of(lhs))? => {
                let dummy = self.ins().iconst(Type::I8, 0);
                let _ = self.replace(lhs, place, dummy)?;
            }
            None => {}
        }
        self.disarm_owned_guards(guards);
        Ok(None)
    }

    fn compound(
        &mut self,
        expr: ExprId,
        (lhs, rhs): (ExprId, ExprId),
        op: BinOp,
    ) -> Result<Option<Value>> {
        if let Some(&Res::Fn(item, substs)) = self.typed.operators.get(&expr) {
            return self.overloaded_assign((expr, lhs, rhs), (item, substs));
        }
        let rhs_num = self.num(self.ty_of(rhs)).ok();
        let Some(value) = self.expr(rhs)? else {
            return Ok(None);
        };
        let Some(place) = self.place_of(lhs)? else {
            bail!("cannot lower assignment to this expression yet");
        };
        let Some(current) = self.read(place) else {
            return Ok(None);
        };
        let num = self.num(self.ty_of(lhs))?;
        let value = self.apply(op, num, (current, value), rhs_num.unwrap_or(num))?;
        self.write(place, value);
        Ok(None)
    }

    fn overloaded_assign(
        &mut self,
        (expr, lhs, rhs): (ExprId, ExprId, ExprId),
        (item, substs): (ItemRef, Substs),
    ) -> Result<Option<Value>> {
        let by_mut = Adjust {
            derefs: 0,
            autoref: Some(Mut::Mut),
        };
        let Some(place) = self.receiver((expr, lhs), by_mut)? else {
            return Ok(None);
        };
        let mut values = vec![place];
        values.extend(self.expr(rhs)?);
        self.invoke(
            super::super::reloc::SiteOrigin::Table(crate::sema::infer::SiteKey::Operator(expr)),
            item,
            substs,
            values,
        )
        .map(|_| None)
    }

    fn assign_parts(&mut self, lhs: ExprId, value: Option<Value>) -> Result<()> {
        match *self.body.expr(lhs) {
            Expr::Underscore => Ok(()),
            Expr::Paren(inner) => self.assign_parts(inner, value),
            Expr::Tuple(items) => {
                let ty = self.ty_of(lhs);
                let whole = self.spill(value, ty)?;
                for (index, &item) in self.body.list(items).to_vec().iter().enumerate() {
                    let part = match whole {
                        Some(whole) => self
                            .part_place(whole, ty, (0, index as u32))
                            .map(|part| self.read(part))?,
                        None => None,
                    };
                    self.assign_parts(ExprId(item), part)?;
                }
                Ok(())
            }
            _ => {
                let Some(value) = value else {
                    return Ok(());
                };
                let Some(place) = self.place_of(lhs)? else {
                    bail!("cannot lower assignment to this expression yet");
                };
                if !self.replace(lhs, place, value)? {
                    self.write(place, value);
                }
                Ok(())
            }
        }
    }
}

use anyhow::{Result, bail};
use haste_codegen::{Block, Signature, Type, Value};

use crate::body::{Expr, ExprId, List};
use crate::sema::infer::Res;

use super::{Loop, Lower, Repr, repr};

impl Lower<'_, '_> {
    /// Lowers `expr` and returns its value, or none for `()`, `!` and code after a jump.
    pub fn expr(&mut self, expr: ExprId) -> Result<Option<Value>> {
        match *self.body.expr(expr) {
            Expr::Lit(lit, at) => self.lit(expr, lit, at).map(Some),
            Expr::Path(_) => self.path(expr),
            Expr::Paren(inner) => self.expr(inner),
            Expr::Unary(op, operand) => self.unary(op, operand),
            Expr::Binary(op, lhs, rhs) => self.binary(expr, op, lhs, rhs),
            Expr::Assign(lhs, rhs) => self.assign(lhs, rhs, None),
            Expr::AssignOp(op, lhs, rhs) => self.assign(lhs, rhs, Some(op)),
            Expr::Cast(inner, _) => self.cast(expr, inner),
            Expr::Call(callee, args) => self.call(callee, args),
            _ => self.memory(expr),
        }
    }

    fn memory(&mut self, expr: ExprId) -> Result<Option<Value>> {
        match *self.body.expr(expr) {
            Expr::Tuple(list) => self.tuple(expr, list),
            Expr::Ref { expr: inner, .. } => self.reference(inner),
            Expr::TupleField(..) => Ok(self.place_of(expr)?.and_then(|place| self.read(place))),
            _ => self.flow(expr),
        }
    }

    fn flow(&mut self, expr: ExprId) -> Result<Option<Value>> {
        match *self.body.expr(expr) {
            Expr::Block { stmts, tail, label: None, .. } => self.block(stmts, tail),
            Expr::If(cond, then, otherwise) => self.if_expr(expr, cond, then, otherwise),
            Expr::While { cond, body, label: None } => self.while_expr(cond, body),
            Expr::Loop { body, label: None } => self.loop_expr(expr, body),
            Expr::Break(None, value) => self.break_expr(value),
            Expr::Continue(None) => self.continue_expr(),
            Expr::Return(value) => self.return_expr(value),
            ref other => bail!("cannot lower expression {other:?} yet"),
        }
    }

    fn path(&mut self, expr: ExprId) -> Result<Option<Value>> {
        match self.typed.res.get(&expr) {
            Some(Res::Local(_)) => Ok(self.place_of(expr)?.and_then(|place| self.read(place))),
            Some(Res::Fn(..)) => Ok(None),
            None => bail!("path has no resolution"),
        }
    }

    /// The register type that carries a value of `repr` across a block edge.
    pub fn carrier(repr: Repr) -> Option<Type> {
        match repr {
            Repr::Zst => None,
            Repr::Scalar(ty) => Some(ty),
            Repr::Mem { .. } => Some(Type::I64),
        }
    }

    fn call(&mut self, callee: ExprId, args: List) -> Result<Option<Value>> {
        let Some(Res::Fn(item, substs)) = self.typed.res.get(&callee) else { bail!("cannot lower calls through values yet") };
        let Some(&index) = self.reached.index.get(&(*item, substs.clone())) else { bail!("call to a function that was not reached") };
        let Ok(callee_typed) = &self.reached.list[index].1 else { bail!("{} did not type-check", self.cx.name(*item)) };
        let signature: Signature = super::signature(self.cx, callee_typed)?;
        let ret = callee_typed.ret.map(|ret| repr(self.cx, ret)).transpose()?.unwrap_or(Repr::Zst);
        let out = match ret {
            Repr::Mem { .. } => self.slot(ret),
            _ => None,
        };
        let mut values = Vec::new();
        if let Some(out) = out {
            values.push(self.ins().stack_addr(Type::I64, out, 0));
        }
        for &arg in self.body.list(args) {
            let value = self.expr(ExprId(arg))?;
            values.extend(value);
        }
        if !self.open {
            return Ok(None);
        }
        let returns = signature.returns.first().map(|param| param.value_type);
        let sig = self.b.import_signature(signature);
        let entry = self.slots.entry(index);
        let table = self.ins().iconst(Type::I64, entry);
        let code = self.ins().load(Type::I64, table, 0);
        let call = self.ins().call_indirect(sig, code, &values);
        if let Some(out) = out {
            return Ok(Some(self.ins().stack_addr(Type::I64, out, 0)));
        }
        Ok(returns.map(|_| self.b.inst_results(call)[0]))
    }

    fn branch_value(&mut self, value: Option<Value>, join: Block) {
        if self.open {
            let args: Vec<Value> = value.into_iter().collect();
            let args = if self.b.block_params(join).is_empty() { Vec::new() } else { args };
            self.ins().jump(join, &args);
            self.close();
        }
    }

    fn if_expr(&mut self, expr: ExprId, cond: ExprId, then: ExprId, otherwise: Option<ExprId>) -> Result<Option<Value>> {
        let Some(test) = self.expr(cond)? else { return Ok(None) };
        let (then_block, else_block, join) = (self.b.create_block(), self.b.create_block(), self.b.create_block());
        let result = match otherwise {
            Some(_) => Self::carrier(self.repr_of(expr)?),
            None => None,
        };
        let joined = result.map(|ty| self.b.append_block_param(join, ty));
        self.ins().brif(test, then_block, &[], else_block, &[]);
        self.b.switch_to_block(then_block);
        self.open = true;
        let value = self.expr(then)?;
        self.branch_value(value, join);
        self.b.switch_to_block(else_block);
        self.open = true;
        let value = match otherwise {
            Some(otherwise) => self.expr(otherwise)?,
            None => None,
        };
        self.branch_value(value, join);
        self.b.switch_to_block(join);
        self.open = true;
        Ok(joined)
    }

    fn while_expr(&mut self, cond: ExprId, body: ExprId) -> Result<Option<Value>> {
        let (head, inner, exit) = (self.b.create_block(), self.b.create_block(), self.b.create_block());
        self.ins().jump(head, &[]);
        self.b.switch_to_block(head);
        let test = self.expr(cond)?;
        if let Some(test) = test {
            self.ins().brif(test, inner, &[], exit, &[]);
        }
        self.b.switch_to_block(inner);
        self.open = true;
        self.loops.push(Loop { head, exit, value: None });
        let result = self.expr(body);
        self.loops.pop();
        result?;
        self.branch_value(None, head);
        self.b.switch_to_block(exit);
        self.open = true;
        Ok(None)
    }

    fn loop_expr(&mut self, expr: ExprId, body: ExprId) -> Result<Option<Value>> {
        let (head, exit) = (self.b.create_block(), self.b.create_block());
        let value = match Self::carrier(self.repr_of(expr)?) {
            Some(ty) => Some((self.b.create_stack_slot(ty.bytes(), ty.bytes().trailing_zeros() as u8), ty)),
            None => None,
        };
        self.ins().jump(head, &[]);
        self.b.switch_to_block(head);
        self.loops.push(Loop { head, exit, value });
        let result = self.expr(body);
        self.loops.pop();
        result?;
        self.branch_value(None, head);
        self.b.switch_to_block(exit);
        self.open = true;
        Ok(value.map(|(slot, ty)| self.ins().stack_load(ty, slot, 0)))
    }

    fn break_expr(&mut self, value: Option<ExprId>) -> Result<Option<Value>> {
        let value = match value {
            Some(value) => self.expr(value)?,
            None => None,
        };
        let Some(&Loop { exit, value: slot, .. }) = self.loops.last() else { bail!("`break` outside a loop") };
        if let (Some(value), Some((slot, _))) = (value, slot) {
            self.ins().stack_store(value, slot, 0);
        }
        self.branch_value(None, exit);
        Ok(None)
    }

    fn continue_expr(&mut self) -> Result<Option<Value>> {
        let Some(head) = self.loops.last().map(|lp| lp.head) else { bail!("`continue` outside a loop") };
        self.branch_value(None, head);
        Ok(None)
    }

    fn return_expr(&mut self, value: Option<ExprId>) -> Result<Option<Value>> {
        let value = match value {
            Some(value) => self.expr(value)?,
            None => None,
        };
        self.finish(value)?;
        Ok(None)
    }
}

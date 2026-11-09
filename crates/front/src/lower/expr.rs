use anyhow::{Result, anyhow, bail};
use haste_codegen::{Block, Signature, StackSlot, Type, Value};

use crate::body::{Expr, ExprId, List, UnOp};
use crate::names::ItemRef;
use crate::outline::item::ItemKind;
use crate::sema::infer::{Coercion, Res};
use crate::sema::ty::{Kind, Ty};

use super::{Loop, Lower, Repr, repr};

impl Lower<'_, '_> {
    /// Lowers `expr` and returns its value, or none for `()`, `!` and code after a jump.
    pub fn expr(&mut self, expr: ExprId) -> Result<Option<Value>> {
        if !self.open {
            return Ok(None);
        }
        let value = self.expr_kind(expr)?;
        let typed = self.typed;
        match (typed.coercions.get(&expr), value) {
            _ if !self.open => Ok(None),
            (Some(Coercion::Len(len)), Some(value)) => {
                let data = self.sized_pointer(value, self.ty_of(expr));
                let len = self.len(*len)?;
                Ok(Some(self.wide_pointer(data, len)))
            }
            (Some(Coercion::Vtable(vtable)), Some(value)) => {
                let data = self.sized_pointer(value, self.ty_of(expr));
                let table = self.vtable(vtable)?;
                Ok(Some(self.wide_pointer(data, table)))
            }
            (Some(Coercion::Deref(steps, _)), Some(value)) => self.deref_coerced(value, steps.clone()),
            (Some(&Coercion::Upcast(slot)), Some(value)) => Ok(Some(self.upcast(value, slot))),
            (Some(Coercion::Reify(Res::Fn(item, substs))), _) => {
                let index = self.reached_index(*item, substs)?;
                Ok(Some(self.code_of(index)))
            }
            _ => Ok(value),
        }
    }

    fn expr_kind(&mut self, expr: ExprId) -> Result<Option<Value>> {
        match *self.body.expr(expr) {
            Expr::Lit(lit, at) => self.lit(expr, lit, at).map(Some),
            Expr::Path(_) => self.path(expr),
            Expr::Paren(inner) => self.expr(inner),
            Expr::Unary(UnOp::Deref, operand) => self.deref_read(expr, operand),
            Expr::Unary(op, operand) => self.unary(op, operand),
            Expr::Binary(op, lhs, rhs) => self.binary(expr, op, lhs, rhs),
            Expr::Assign(lhs, rhs) => self.assign(lhs, rhs, None),
            Expr::AssignOp(op, lhs, rhs) => self.assign(lhs, rhs, Some(op)),
            Expr::Cast(inner, _) => self.cast(expr, inner),
            Expr::Call(callee, args) => self.call(expr, callee, args),
            Expr::Method { receiver, params, .. } => self.method(expr, receiver, params),
            Expr::OffsetOf { .. } => {
                let Some(&offset) = self.typed.offsets.get(&expr) else { bail!("offset_of with no offset") };
                Ok(Some(self.int_const(haste_codegen::Type::I64, u128::from(offset))))
            }
            _ => self.memory(expr),
        }
    }

    fn deref_read(&mut self, expr: ExprId, operand: ExprId) -> Result<Option<Value>> {
        if self.cx.boxed(self.ty_of(operand)).is_some() && crate::sema::infer::needs_drop(self.cx, self.ty_of(expr))? {
            bail!("moving a value that needs dropping out of a `Box` is not supported yet");
        }
        let place = self.place_of(expr)?;
        Ok(place.and_then(|place| self.read(place)))
    }

    fn memory(&mut self, expr: ExprId) -> Result<Option<Value>> {
        match *self.body.expr(expr) {
            Expr::Tuple(list) => self.tuple(expr, list),
            Expr::Closure { .. } => self.closure(expr),
            Expr::Ref { expr: inner, mutable: false, raw: false } if self.promotable(inner) => self.promoted(inner),
            Expr::Ref { expr: inner, .. } => self.reference(inner),
            Expr::TupleField(..) => self.field_value(expr),
            _ => self.flow(expr),
        }
    }

    fn flow(&mut self, expr: ExprId) -> Result<Option<Value>> {
        match *self.body.expr(expr) {
            Expr::Block { stmts, tail, label: None, kind: crate::body::BlockKind::Try } => self.try_block(expr, stmts, tail),
            Expr::Block { stmts, tail, label: None, .. } => self.block(stmts, tail),
            Expr::Block { stmts, tail, label: Some(label), .. } => self.labeled_block(expr, (stmts, tail), label),
            Expr::If(cond, then, otherwise) => self.if_expr(expr, cond, then, otherwise),
            Expr::While { cond, body, label } => self.while_expr(cond, body, label),
            Expr::Loop { body, label } => self.loop_expr(expr, body, label),
            Expr::Break(label, value) => self.break_expr(label, value),
            Expr::Continue(label) => self.continue_expr(label),
            Expr::Return(value) => self.return_expr(value),
            _ => self.data(expr),
        }
    }

    fn data(&mut self, expr: ExprId) -> Result<Option<Value>> {
        match *self.body.expr(expr) {
            Expr::Struct { fields, base, .. } => self.struct_literal(expr, fields, base),
            Expr::Field(..) => self.field_value(expr),
            Expr::Match(scrutinee, arms) => self.match_expr(expr, scrutinee, arms),
            Expr::Let(pat, init) => self.let_expr(pat, init),
            Expr::Array(items) => self.array(expr, items),
            Expr::Repeat(value, _) => self.repeat(expr, value),
            Expr::Index(..) => Ok(self.place_of(expr)?.and_then(|place| self.read(place))),
            Expr::For { label, pat, iter, body } => self.for_expr(expr, pat, iter, (label, body)),
            Expr::Try(inner) => self.try_expr(expr, inner),
            Expr::Range { from, to, .. } => self.range(expr, from, to),
            ref other => bail!("cannot lower expression {other:?} yet"),
        }
    }

    fn field_value(&mut self, expr: ExprId) -> Result<Option<Value>> {
        let value = self.place_of(expr)?.and_then(|place| self.read(place));
        if let Some((root, path)) = self.move_path(expr)
            && crate::sema::infer::needs_drop(self.cx, self.ty_of(expr))?
        {
            self.moved_out(root, &path)?;
        }
        Ok(value)
    }

    fn path(&mut self, expr: ExprId) -> Result<Option<Value>> {
        match self.typed.res.get(&expr) {
            Some(&Res::Local(local)) => {
                let value = self.place_of(expr)?.and_then(|place| self.read(place));
                if crate::sema::infer::needs_drop(self.cx, self.ty_of(expr))? {
                    self.moved_out(super::moves::Root::Local(local), &[])?;
                }
                Ok(value)
            }
            Some(Res::Fn(item, substs)) => {
                let (item, substs) = (*item, substs.clone());
                self.item_path(expr, item, &substs)
            }
            Some(&Res::Const(value)) => {
                let (Kind::Value(bits), Repr::Scalar(machine)) = (self.cx.types.kind(value).clone(), self.repr_of(expr)?) else { bail!("a const parameter whose value is not known") };
                Ok(Some(self.int_const(machine, bits)))
            }
            Some(Res::Ctor(..)) if self.typed.coercions.contains_key(&expr) => Ok(None),
            Some(Res::Ctor(_, variant, _)) => {
                let variant = *variant;
                self.construct(self.ty_of(expr), variant, Vec::new(), None)
            }
            None => bail!("path has no resolution"),
        }
    }

    fn item_path(&mut self, expr: ExprId, item: ItemRef, substs: &[Ty]) -> Result<Option<Value>> {
        match self.cx.item(item).1.kind {
            ItemKind::Static => Ok(self.place_of(expr)?.and_then(|place| self.read(place))),
            ItemKind::Const => self.invoke(item, substs, Vec::new()),
            _ => Ok(None),
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

    fn call(&mut self, expr: ExprId, callee: ExprId, args: List) -> Result<Option<Value>> {
        if let Some(Res::Fn(item, substs)) = self.typed.res.get(&expr).cloned() {
            return self.call_value((expr, callee), (item, substs), args);
        }
        let (item, substs) = match self.typed.res.get(&callee).cloned() {
            Some(Res::Fn(item, _)) if matches!(self.cx.item(item).1.kind, ItemKind::Const | ItemKind::Static) => return self.pointer_call(expr, callee, args),
            Some(Res::Fn(item, substs)) => (item, substs),
            Some(Res::Ctor(_, variant, _)) => return self.ctor_call(expr, variant, args),
            _ => return self.pointer_call(expr, callee, args),
        };
        if self.tupled_closure_call(item, &substs) {
            bail!("a `call_once` of a closure with a tupled argument is not lowered yet");
        }
        let values = self.args(args, Vec::new())?;
        self.invoke(item, &substs, values)
    }

    fn method(&mut self, expr: ExprId, receiver: ExprId, params: List) -> Result<Option<Value>> {
        let Some(Res::Fn(item, substs)) = self.typed.res.get(&expr).cloned() else { bail!("method call has no resolution") };
        let Some(&adjust) = self.typed.adjusts.get(&expr) else { bail!("method call has no receiver adjustment") };
        if self.tupled_closure_call(item, &substs) {
            bail!("a closure method call with a tupled argument is not lowered yet");
        }
        let first = self.receiver((expr, receiver), adjust)?;
        let values = self.args(params, first.into_iter().collect())?;
        self.invoke(item, &substs, values)
    }

    pub(super) fn args(&mut self, args: List, mut values: Vec<Value>) -> Result<Vec<Value>> {
        for arg in self.body.list(args).to_vec() {
            values.extend(self.expr(ExprId(arg))?);
        }
        Ok(values)
    }

    pub(super) fn invoke(&mut self, item: ItemRef, substs: &[Ty], values: Vec<Value>) -> Result<Option<Value>> {
        if !self.open {
            return Ok(None);
        }
        if let Some(which) = self.cx.intrinsic(item) {
            return self.intrinsic(which, substs, &values);
        }
        let index = self.reached_index(item, substs)?;
        let Ok(callee) = &self.reached.list[index].1 else { bail!("{} did not type-check", self.cx.name(item)) };
        let code = self.code_of(index);
        if self.cx.foreign(item) {
            let params = callee.params.iter().map(|&local| callee.locals[local as usize].ty).collect();
            return self.native_call(code, (params, callee.ret), values);
        }
        let signature = super::signature(self.cx, callee)?;
        self.call_code(code, (signature, callee.ret), values)
    }

    fn pointer_call(&mut self, expr: ExprId, callee: ExprId, args: List) -> Result<Option<Value>> {
        let (code, ty) = match self.typed.autoderefs.contains_key(&expr) {
            true => match self.autoderef_place(expr, callee)? {
                Some((place, ty)) => (self.read(place), ty),
                None => return Ok(None),
            },
            false => (self.expr(callee)?, self.ty_of(callee)),
        };
        if let Kind::FnDef(item, substs) = self.cx.types.kind(ty).clone() {
            let values = self.args(args, Vec::new())?;
            return self.invoke(item, &substs, values);
        }
        let Kind::FnPtr(params, ret) = self.cx.types.kind(ty).clone() else { bail!("cannot lower calls through values yet") };
        let Some(code) = code else { return Ok(None) };
        let values = self.args(args, Vec::new())?;
        if !self.open {
            return Ok(None);
        }
        let signature = super::signature_of(self.cx, Some(ret), params)?;
        self.call_code(code, (signature, Some(ret)), values)
    }

    /// The index of the reached instance of `item` at `substs`.
    pub(super) fn reached_index(&self, item: ItemRef, substs: &[Ty]) -> Result<usize> {
        self.reached.index.get(&(item, substs.to_vec())).copied().ok_or_else(|| anyhow!("a use of `{}` that was not reached", self.cx.name(item)))
    }

    /// The code address of reached function `index`, read from its call table entry.
    pub(super) fn code_of(&mut self, index: usize) -> Value {
        let entry = self.slots.entry(index);
        let table = self.ins().iconst(Type::I64, entry);
        self.ins().load(Type::I64, table, 0)
    }

    pub(super) fn call_code(&mut self, code: Value, (signature, ret): (Signature, Option<Ty>), mut values: Vec<Value>) -> Result<Option<Value>> {
        let result = ret.map(|ret| repr(self.cx, ret)).transpose()?.unwrap_or(Repr::Zst);
        let out = match result {
            Repr::Mem { .. } => self.slot(result),
            _ => None,
        };
        if let Some(out) = out {
            let address = self.ins().stack_addr(Type::I64, out, 0);
            values.insert(0, address);
        }
        let returns = signature.returns.first().map(|param| param.value_type);
        let sig = self.b.import_signature(signature);
        let call = self.ins().call_indirect(sig, code, &values);
        if ret.is_some_and(|ret| matches!(self.cx.types.kind(ret), Kind::Never)) {
            self.ins().trap();
            self.close();
            return Ok(None);
        }
        if let Some(out) = out {
            return Ok(Some(self.ins().stack_addr(Type::I64, out, 0)));
        }
        Ok(returns.map(|_| self.b.inst_results(call)[0]))
    }

    pub(super) fn branch_value(&mut self, value: Option<Value>, join: Block) {
        if self.open {
            let args: Vec<Value> = value.into_iter().collect();
            let args = if self.b.block_params(join).is_empty() { Vec::new() } else { args };
            self.jump(join, &args);
            self.close();
        }
    }

    fn if_expr(&mut self, expr: ExprId, cond: ExprId, then: ExprId, otherwise: Option<ExprId>) -> Result<Option<Value>> {
        let mark = self.owned.len();
        let Some(test) = self.expr(cond)? else { return Ok(None) };
        let (then_block, else_block, join) = (self.b.create_block(), self.b.create_block(), self.b.create_block());
        let result = match otherwise {
            Some(_) => Self::carrier(self.repr_of(expr)?),
            None => None,
        };
        let joined = result.map(|ty| self.b.append_block_param(join, ty));
        self.brif(test, (then_block, &[]), (else_block, &[]));
        self.enter(then_block);
        let value = self.expr(then)?;
        if self.open {
            self.drop_to(mark)?;
        }
        self.branch_value(value, join);
        self.enter(else_block);
        self.drop_to(mark)?;
        let value = match otherwise {
            Some(otherwise) => self.expr(otherwise)?,
            None => None,
        };
        self.branch_value(value, join);
        self.owned.truncate(mark);
        self.enter(join);
        Ok(joined)
    }

    fn while_expr(&mut self, cond: ExprId, body: ExprId, label: Option<u32>) -> Result<Option<Value>> {
        let (head, inner, exit) = (self.b.create_block(), self.b.create_block(), self.b.create_block());
        self.jump(head, &[]);
        self.enter(head);
        let depth = self.owned.len();
        let test = self.expr(cond)?;
        if let Some(test) = test {
            let fail = self.b.create_block();
            self.brif(test, (inner, &[]), (fail, &[]));
            self.enter(fail);
            self.drop_to(depth)?;
            self.jump(exit, &[]);
        }
        self.enter(inner);
        self.loops.push(Loop { head, exit, value: None, depth, label, block: false });
        let result = self.expr(body);
        self.loops.pop();
        result?;
        self.end_scope(depth)?;
        self.branch_value(None, head);
        self.enter(exit);
        Ok(None)
    }

    fn labeled_block(&mut self, expr: ExprId, (stmts, tail): (List, Option<ExprId>), label: u32) -> Result<Option<Value>> {
        let exit = self.b.create_block();
        let value = self.value_slot(expr)?;
        self.loops.push(Loop { head: exit, exit, value, depth: self.owned.len(), label: Some(label), block: true });
        let output = self.block(stmts, tail);
        self.loops.pop();
        if let (Some(output), Some((slot, _))) = (output?, value)
            && self.open
        {
            self.ins().stack_store(output, slot, 0);
        }
        self.branch_value(None, exit);
        self.enter(exit);
        Ok(value.map(|(slot, ty)| self.ins().stack_load(ty, slot, 0)))
    }

    /// A stack slot for the value of the loop or block `expr` that its `break`s leave with, when it has a scalar value.
    pub(super) fn value_slot(&mut self, expr: ExprId) -> Result<Option<(StackSlot, Type)>> {
        Ok(Self::carrier(self.repr_of(expr)?).map(|ty| (self.b.create_stack_slot(ty.bytes(), ty.bytes().trailing_zeros() as u8), ty)))
    }

    fn loop_expr(&mut self, expr: ExprId, body: ExprId, label: Option<u32>) -> Result<Option<Value>> {
        let (head, exit) = (self.b.create_block(), self.b.create_block());
        let value = self.value_slot(expr)?;
        self.jump(head, &[]);
        self.enter(head);
        let depth = self.owned.len();
        self.loops.push(Loop { head, exit, value, depth, label, block: false });
        let result = self.expr(body);
        self.loops.pop();
        result?;
        self.branch_value(None, head);
        self.enter(exit);
        Ok(value.map(|(slot, ty)| self.ins().stack_load(ty, slot, 0)))
    }

    fn jump_target(&self, label: Option<u32>) -> Option<Loop> {
        self.loops.iter().rev().find(|frame| crate::body::jumps_to(self.src, (frame.label, frame.block), label)).copied()
    }

    fn break_expr(&mut self, label: Option<u32>, value: Option<ExprId>) -> Result<Option<Value>> {
        let value = match value {
            Some(value) => self.expr(value)?,
            None => None,
        };
        let Some(Loop { exit, value: slot, depth, .. }) = self.jump_target(label) else { bail!("`break` outside a loop") };
        if let (Some(value), Some((slot, _))) = (value, slot) {
            self.ins().stack_store(value, slot, 0);
        }
        self.drop_to(depth)?;
        self.branch_value(None, exit);
        Ok(None)
    }

    fn continue_expr(&mut self, label: Option<u32>) -> Result<Option<Value>> {
        let Some(Loop { head, depth, .. }) = self.jump_target(label) else { bail!("`continue` outside a loop") };
        self.drop_to(depth)?;
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

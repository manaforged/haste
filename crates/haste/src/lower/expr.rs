use std::sync::Arc;

use anyhow::{Result, bail};
use haste_codegen::{Block, Signature, Type, Value};

use crate::body::{Expr, ExprId, List, UnOp};
use crate::names::ItemRef;
use crate::outline::item::ItemKind;
use crate::sema::Context;
use crate::sema::infer::{Coercion, Res, Shape};
use crate::sema::reach::Reached;
use crate::sema::substs::Substs;
use crate::sema::ty::{Kind, Ty};

use super::place::Place;
use super::stencil::{InlinedCallee, LowerMode, SymbolSource, item_bits, resolve_inline};
use super::{Loop, Lower, Parts, Repr, repr};

/// The most instructions a sysroot body may have, after slot promotion, for a call to copy it inline.
const INLINE_LIMIT: usize = 30;

/// The most instructions an `#[inline(always)]` sysroot body may have for a call to copy it inline.
const ALWAYS_LIMIT: usize = 100;

/// The most source expressions a sysroot body may have for lowering it to try `INLINE_LIMIT`.
const INLINE_EXPRS: usize = 20;

/// What a call runs: code at an address, or reached function `index`, which may be copied inline.
enum Callee {
    Code(Value),
    Reached(usize),
}

/// Whether a call may try to copy reached function `index` inline.
pub fn may_inline(cx: &mut Context<'_>, reached: &Reached, index: usize) -> bool {
    inline_limit(cx, reached, index).is_some()
}

/// Whether reached function `index` lowers to a body small and copyable enough to inline.
pub fn can_inline_body(cx: &mut Context<'_>, reached: &Reached, slots: &dyn super::Slots, index: usize) -> bool {
    let Some(limit) = inline_limit(cx, reached, index) else { return false };
    resolve_inline(cx, reached, slots, index, limit).is_ok_and(|(callee, _)| callee.is_some())
}

/// The most instructions a call may copy inline of reached function `index`, or none when calls never inline it.
fn inline_limit(cx: &mut Context<'_>, reached: &Reached, index: usize) -> Option<usize> {
    let ((item, _), Ok(typed)) = &reached.list[index] else { return None };
    if !cx.outline.crates[item.krate as usize].env.release {
        return None;
    }
    let source = match &typed.shape {
        Shape::Body if cx.inline_always(*item) => return Some(ALWAYS_LIMIT),
        Shape::Body => *item,
        Shape::Closure(closure) => closure.parent,
        _ => return Some(INLINE_LIMIT),
    };
    cx.body(source).is_ok_and(|body| body.exprs.len() <= INLINE_EXPRS).then_some(INLINE_LIMIT)
}

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
                Ok(Some(self.wide(data, table)))
            }
            (Some(Coercion::Deref(steps, _)), Some(value)) => self.deref_coerced(value, steps.clone()),
            (Some(&Coercion::Upcast(slot)), Some(value)) => Ok(Some(self.upcast(value, slot))),
            (Some(&Coercion::Reify(Res::Fn(item, substs))), _) => {
                let index = self.reached_index(item, substs)?;
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
            Expr::Unary(op, operand) => self.unary(expr, op, operand),
            Expr::Binary(op, lhs, rhs) => self.binary(expr, op, lhs, rhs),
            Expr::Assign(lhs, rhs) => self.assign(expr, (lhs, rhs), None),
            Expr::AssignOp(op, lhs, rhs) => self.assign(expr, (lhs, rhs), Some(op)),
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
            return self.move_out_of_box(expr, operand);
        }
        let place = self.place_of(expr)?;
        Ok(place.and_then(|place| self.read(place)))
    }

    fn move_out_of_box(&mut self, expr: ExprId, operand: ExprId) -> Result<Option<Value>> {
        let Some((root, path)) = self.move_path(operand) else { bail!("moving out of a `Box` that no local owns is not supported yet") };
        let boxed = self.ty_of(operand);
        let Some(Res::Fn(free, substs)) = crate::sema::infer::drop_fn(self.cx, boxed)? else { bail!("`Box` has no `Drop` impl") };
        let contents = self.place_of(expr)?;
        let value = match contents {
            Some(place) if matches!(place.repr, Repr::Mem { .. }) => {
                let read = self.read(place);
                self.spill(read, self.ty_of(expr))?.and_then(|copy| self.read(copy))
            }
            place => place.and_then(|place| self.read(place)),
        };
        let owner = self.place_of(operand)?;
        let target = self.address_or_dangling(owner, boxed)?;
        self.invoke(free, substs, vec![target])?;
        self.moved_out(root, &path)?;
        Ok(value)
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
            Expr::Block { kind: crate::body::BlockKind::Async | crate::body::BlockKind::AsyncMove, .. } => self.closure(expr),
            Expr::Await(inner) => self.await_expr(expr, inner),
            Expr::Block { stmts, tail, label: None, kind: crate::body::BlockKind::Expansion } => self.expansion(stmts, tail),
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
            Some(&Res::Fn(item, substs)) => self.item_path(expr, item, substs),
            Some(&Res::Const(value)) => {
                let (Kind::Value(bits), Repr::Scalar(machine)) = (self.cx.types.kind(value).clone(), self.repr_of(expr)?) else { bail!("a const parameter whose value is not known") };
                Ok(Some(self.int_const(machine, bits)))
            }
            Some(Res::Ctor(..)) if self.typed.coercions.contains_key(&expr) => Ok(None),
            Some(&Res::Ctor(_, variant, _)) => self.construct(self.ty_of(expr), variant, Vec::new(), None),
            None => bail!("path has no resolution"),
        }
    }

    fn item_path(&mut self, expr: ExprId, item: ItemRef, substs: Substs) -> Result<Option<Value>> {
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

    /// The frame slot that branches write a value of `repr` into for the block they join, when it is a small value made of scalars; other values cross the edge in their carrier.
    pub(super) fn join_slot(&mut self, repr: Repr) -> Option<Place> {
        match repr {
            Repr::Mem { parts, .. } if parts != Parts::Bytes => self.slot(repr).map(|slot| Place::slot(slot, repr)),
            _ => None,
        }
    }

    fn call(&mut self, expr: ExprId, callee: ExprId, args: List) -> Result<Option<Value>> {
        if let Some(&Res::Fn(item, substs)) = self.typed.res.get(&expr) {
            return self.call_value((expr, callee), (item, substs), args);
        }
        let named = self.typed.res.get(&callee).filter(|_| self.body.names_callee(callee));
        let (item, substs) = match named.copied() {
            Some(Res::Fn(item, _)) if matches!(self.cx.item(item).1.kind, ItemKind::Const | ItemKind::Static) => return self.pointer_call(expr, callee, args),
            Some(Res::Fn(item, substs)) => (item, substs),
            Some(Res::Ctor(_, variant, _)) => return self.ctor_call(expr, variant, args),
            _ => return self.pointer_call(expr, callee, args),
        };
        if self.tupled_closure_call(item, substs) {
            bail!("a `call_once` of a closure with a tupled argument is not lowered yet");
        }
        let values = self.args(args, Vec::new())?;
        self.invoke(item, substs, values)
    }

    fn method(&mut self, expr: ExprId, receiver: ExprId, params: List) -> Result<Option<Value>> {
        let Some(&Res::Fn(item, substs)) = self.typed.res.get(&expr) else { bail!("method call has no resolution") };
        let Some(&adjust) = self.typed.adjusts.get(&expr) else { bail!("method call has no receiver adjustment") };
        if self.tupled_closure_call(item, substs) {
            bail!("a closure method call with a tupled argument is not lowered yet");
        }
        let first = self.receiver((expr, receiver), adjust)?;
        let values = self.args(params, first.into_iter().collect())?;
        self.invoke(item, substs, values)
    }

    pub(super) fn args(&mut self, args: List, mut values: Vec<Value>) -> Result<Vec<Value>> {
        for arg in self.body.list(args).to_vec() {
            if !self.typed.const_args.contains(&ExprId(arg)) {
                values.extend(self.expr(ExprId(arg))?);
            }
        }
        Ok(values)
    }

    pub(super) fn invoke(&mut self, item: ItemRef, substs: Substs, values: Vec<Value>) -> Result<Option<Value>> {
        if !self.open {
            return Ok(None);
        }
        if let Some(which) = self.cx.intrinsic(item) {
            let slice = self.cx.types.substs(substs).to_vec();
            return self.intrinsic(which, &slice, &values);
        }
        let index = self.reached_index(item, substs)?;
        let Ok(callee) = &self.reached.list[index].1 else { bail!("{} did not type-check", self.cx.name(item)) };
        let params = callee.params.iter().map(|&local| callee.locals[local as usize].ty).collect();
        if self.cx.foreign(item) {
            let code = self.code_of(index);
            return self.native_call(code, (params, callee.ret), values);
        }
        self.call_with(Callee::Reached(index), (params, callee.ret), values)
    }

    fn inline_callee(&mut self, index: usize) -> Result<Option<Arc<InlinedCallee>>> {
        let Some(limit) = inline_limit(self.cx, self.reached, index) else {
            return Ok(None);
        };
        let (callee, cycle) = resolve_inline(self.cx, self.reached, self.slots, index, limit)?;
        self.cycle_root = self.cycle_root.min(cycle);
        Ok(callee)
    }

    fn emit_inlined(&mut self, inlined: &InlinedCallee, values: &[Value]) -> Vec<Value> {
        if !self.open {
            let dead = *self.dead.get_or_insert_with(|| self.b.create_block());
            self.b.switch_to_block(dead);
        }
        let sym_base = self.relocs.len() as u32;
        self.relocs.extend(inlined.relocs.iter().cloned());
        match self.mode {
            LowerMode::Probe => self.b.inline_probe(&inlined.body, values, item_bits(inlined.item), inlined.footprint.0, inlined.returns),
            LowerMode::Stencil => self.b.inline_shifted(&inlined.body, values, sym_base),
            LowerMode::Dict => {
                let dict = self.dict_param.expect("dictionary parameter is set");
                self.b.inline_dict(&inlined.body, values, dict, sym_base)
            }
        }
    }

    /// Calls `callee` with `values` by `signature`, or copies its body inline, and returns its results.
    fn emit_call(&mut self, callee: Callee, signature: Signature, values: &[Value]) -> Result<Vec<Value>> {
        let code = match callee {
            Callee::Code(code) => code,
            Callee::Reached(index) => match self.inline_callee(index)? {
                Some(inlined) => return Ok(self.emit_inlined(&inlined, values)),
                None => self.code_of(index),
            },
        };
        let sig = self.b.import_signature(signature);
        let call = self.ins().call_indirect(sig, code, values);
        Ok(self.b.inst_results(call).to_vec())
    }

    fn pointer_call(&mut self, expr: ExprId, callee: ExprId, args: List) -> Result<Option<Value>> {
        let (code, ty) = match self.typed.autoderefs.contains_key(&expr) {
            true => match self.autoderef_place(expr, callee)? {
                Some((place, ty)) => (self.read(place), ty),
                None => return Ok(None),
            },
            false => (self.expr(callee)?, self.ty_of(callee)),
        };
        if let Kind::FnDef(item, substs) = *self.cx.types.kind(ty) {
            let values = self.args(args, Vec::new())?;
            return self.invoke(item, substs, values);
        }
        let Kind::FnPtr(params, ret) = *self.cx.types.kind(ty) else { bail!("cannot lower calls through values yet") };
        let Some(code) = code else { return Ok(None) };
        let params = self.cx.types.substs(params).to_vec();
        let values = self.args(args, Vec::new())?;
        if !self.open {
            return Ok(None);
        }
        self.call_code(code, (params, Some(ret)), values)
    }

    /// The index of the reached instance of `item` at `substs`.
    pub(super) fn reached_index(&self, item: ItemRef, substs: Substs) -> Result<usize> {
        let instance = (item, substs);
        if let Some(&index) = self.reached.index.get(&instance)
            && !self.reached.is_untyped(index)
        {
            return Ok(index);
        }
        self.slots.want(instance);
        bail!("a use of `{}` that was not reached", self.cx.name(item))
    }

    /// The code address of reached function `index`, read from its call table entry.
    pub(super) fn code_of(&mut self, index: usize) -> Value {
        let table = self.emit_slot(SymbolSource::Slot(index));
        self.ins().load(Type::I64, table, 0)
    }

    /// Calls the generated code at `code`, which takes `params` and returns `ret`, with the values of its arguments that are not zero-sized.
    pub(super) fn call_code(&mut self, code: Value, types: (Vec<Ty>, Option<Ty>), values: Vec<Value>) -> Result<Option<Value>> {
        self.call_with(Callee::Code(code), types, values)
    }

    fn call_with(&mut self, callee: Callee, (params, ret): (Vec<Ty>, Option<Ty>), values: Vec<Value>) -> Result<Option<Value>> {
        let signature = super::signature_of(self.cx, ret, params.iter().copied())?;
        let mut values = self.pass_all(&params, values)?;
        let result = ret.map(|ret| repr(self.cx, ret)).transpose()?.unwrap_or(Repr::Zst);
        let out = if super::indirect(&signature) { self.slot(result) } else { None };
        if let Some(out) = out {
            let address = self.ins().stack_addr(Type::I64, out, 0);
            values.insert(0, address);
        }
        let results = self.emit_call(callee, signature, &values)?;
        let Some(ret) = ret else { return Ok(None) };
        if matches!(self.cx.types.kind(ret), Kind::Never) {
            self.ins().trap();
            self.close();
            return Ok(None);
        }
        if let Some(out) = out {
            return Ok(Some(self.ins().stack_addr(Type::I64, out, 0)));
        }
        let parts: Vec<(Type, i32)> = super::returned(self.cx, ret)?.into_iter().flat_map(Parts::scalars).collect();
        self.received(ret, &parts, &results)
    }

    /// The argument `values` of a call that takes `params`, as they cross it: zero-sized parameters take no value, and a value whose type travels unwrapped crosses as its scalar.
    pub(super) fn pass_all(&mut self, params: &[Ty], values: Vec<Value>) -> Result<Vec<Value>> {
        let mut values = values.into_iter();
        let mut passed = Vec::new();
        for &param in params {
            if repr(self.cx, param)? == Repr::Zst {
                continue;
            }
            let Some(value) = values.next() else { break };
            passed.push(self.passed(param, value)?);
        }
        passed.extend(values);
        Ok(passed)
    }

    pub(super) fn branch_value(&mut self, value: Option<Value>, join: Block) {
        if self.open {
            let args: Vec<Value> = value.into_iter().collect();
            let args = if self.b.block_params(join).is_empty() { Vec::new() } else { args };
            self.jump(join, &args);
            self.close();
        }
    }

    /// The block that the branches of `expr` join, and where each leaves its value: the join's parameter, or the slot of `join_slot`.
    pub(super) fn join_block(&mut self, expr: ExprId) -> Result<(Block, Option<Value>, Option<Place>)> {
        let join = self.b.create_block();
        let repr = self.repr_of(expr)?;
        let slot = self.join_slot(repr);
        let joined = Self::carrier(repr).filter(|_| slot.is_none()).map(|ty| self.b.append_block_param(join, ty));
        Ok((join, joined, slot))
    }

    /// Goes to `join` with `value`, written into `slot` first when the join takes it there.
    pub(super) fn branch_into(&mut self, value: Option<Value>, (join, slot): (Block, Option<Place>)) {
        match (slot, value) {
            (Some(place), Some(value)) if self.open => {
                self.write(place, value);
                self.branch_value(None, join);
            }
            _ => self.branch_value(value, join),
        }
    }

    /// The value that the branches left at `join`, once the cursor is there.
    pub(super) fn joined(&mut self, joined: Option<Value>, slot: Option<Place>) -> Option<Value> {
        match slot {
            Some(place) => self.read(place),
            None => joined,
        }
    }

    fn if_expr(&mut self, expr: ExprId, cond: ExprId, then: ExprId, otherwise: Option<ExprId>) -> Result<Option<Value>> {
        let mark = self.owned.len();
        let Some(test) = self.expr(cond)? else { return Ok(None) };
        let (then_block, else_block) = (self.b.create_block(), self.b.create_block());
        let (join, joined, slot) = match otherwise {
            Some(_) => self.join_block(expr)?,
            None => (self.b.create_block(), None, None),
        };
        self.brif(test, (then_block, &[]), (else_block, &[]));
        self.enter(then_block);
        let value = self.expr(then)?;
        if self.open {
            self.drop_to(mark)?;
        }
        self.branch_into(value, (join, slot));
        self.enter(else_block);
        self.drop_to(mark)?;
        let value = match otherwise {
            Some(otherwise) => self.expr(otherwise)?,
            None => None,
        };
        self.branch_into(value, (join, slot));
        self.owned.truncate(mark);
        self.enter(join);
        Ok(self.joined(joined, slot))
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
        self.loop_body((head, exit, depth), label, body)
    }

    fn labeled_block(&mut self, expr: ExprId, (stmts, tail): (List, Option<ExprId>), label: u32) -> Result<Option<Value>> {
        let exit = self.b.create_block();
        let value = self.value_slot(expr)?;
        self.loops.push(Loop { head: exit, exit, value, depth: self.owned.len(), label: Some(label), block: true });
        let output = self.block(stmts, tail);
        self.loops.pop();
        if let (Some(output), Some(place)) = (output?, value)
            && self.open
        {
            self.write(place, output);
        }
        self.branch_value(None, exit);
        self.enter(exit);
        Ok(value.and_then(|place| self.read(place)))
    }

    /// The frame slot that the `break`s of the loop or block `expr` leave its value in: the value itself when `join_slot` holds it, else its carrier.
    pub(super) fn value_slot(&mut self, expr: ExprId) -> Result<Option<Place>> {
        let repr = self.repr_of(expr)?;
        if let Some(place) = self.join_slot(repr) {
            return Ok(Some(place));
        }
        let held = Self::carrier(repr).map_or(Repr::Zst, Repr::Scalar);
        Ok(self.slot(held).map(|slot| Place::slot(slot, held)))
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
        Ok(value.and_then(|place| self.read(place)))
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
        if let (Some(value), Some(place)) = (value, slot) {
            self.write(place, value);
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

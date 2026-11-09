use anyhow::{Result, bail};
use haste_codegen::{Type, Value};

use crate::body::{ExprId, List};
use crate::names::ItemRef;
use crate::sema::infer::{By, ClosureBody, Res, Shape, needs_drop};
use crate::sema::ty::{Kind, Mut, Ty};

use super::moves::Root;
use super::place::{Base, Place};
use super::{Lower, repr};

impl Lower<'_, '_> {
    /// Lowers the closure expression `expr` to its environment: each capture's value or address, in a new frame slot.
    pub(super) fn closure(&mut self, expr: ExprId) -> Result<Option<Value>> {
        let ty = self.ty_of(expr);
        let Some(env) = self.cx.closure_env(ty) else { bail!("a closure whose captures are not known") };
        let Some(closure) = self.typed.closures.get(&expr).cloned() else { bail!("a closure expression with no inferred closure") };
        let whole = repr(self.cx, env)?;
        let Some(slot) = self.slot(whole) else { return Ok(None) };
        let whole = Place::slot(slot, whole);
        for (index, capture) in closure.captures.iter().enumerate() {
            let held = self.locals[capture.local as usize];
            let local_ty = self.typed.locals[capture.local as usize].ty;
            let value = match capture.by {
                By::Value => {
                    let value = held.and_then(|place| self.read(place));
                    if needs_drop(self.cx, local_ty)? {
                        self.moved_out(Root::Local(capture.local), &[])?;
                    }
                    value
                }
                By::Ref | By::Mut => Some(self.address_or_dangling(held, local_ty)?),
            };
            let part = self.part_place(whole, env, (0, index as u32))?;
            if let Some(value) = value {
                self.write(part, value);
            }
        }
        Ok(self.read(whole))
    }

    /// Points each local that the closure being lowered captures at its place in the environment.
    pub(super) fn bind_captures(&mut self) -> Result<()> {
        let Shape::Closure(ClosureBody { env: Some(local), closure, .. }) = self.typed.shape.clone() else { return Ok(()) };
        let Kind::Ref(_, env) = *self.cx.types.kind(self.typed.locals[local as usize].ty) else { bail!("a closure environment that is not a reference") };
        let Some(pointer) = self.locals[local as usize].and_then(|place| self.read(place)) else { bail!("a closure environment with no address") };
        let whole = Place { base: Base::Addr(pointer), offset: 0, repr: repr(self.cx, env)? };
        for (index, capture) in closure.captures.iter().enumerate() {
            let part = self.part_place(whole, env, (0, index as u32))?;
            let local_ty = self.typed.locals[capture.local as usize].ty;
            if closure.consumes && capture.by == By::Value {
                self.own(Root::Local(capture.local), local_ty, Some(part), true)?;
            }
            self.locals[capture.local as usize] = Some(match capture.by {
                By::Value => part,
                By::Ref | By::Mut => {
                    let Some(address) = self.read(part) else { bail!("a captured reference with no value") };
                    Place { base: Base::Addr(address), offset: 0, repr: repr(self.cx, local_ty)? }
                }
            });
        }
        Ok(())
    }

    /// Lowers a call of the callee value `callee`, a closure, a function item, a trait object or a value with its own `Fn` impl, which inference resolved to `(item, substs)`.
    pub(super) fn call_value(&mut self, (expr, callee): (ExprId, ExprId), (item, substs): (ItemRef, Vec<Ty>), args: List) -> Result<Option<Value>> {
        let item_value = matches!(self.cx.types.kind(self.ty_of(callee)), Kind::FnDef(..));
        if self.typed.overloaded.contains(&expr) || !item_value && substs.first().is_some_and(|ty| matches!(self.cx.types.kind(*ty), Kind::Dyn(..))) {
            return self.object_call((expr, callee), (item, substs), args);
        }
        if self.typed.autoderefs.contains_key(&expr) {
            let values = self.pointed_env(expr, callee)?;
            let values = self.args(args, values)?;
            return self.invoke(item, &substs, values);
        }
        let values = self.callee_env(callee)?;
        let values = self.args(args, values)?;
        self.invoke(item, &substs, values)
    }

    fn callee_env(&mut self, callee: ExprId) -> Result<Vec<Value>> {
        let callee_ty = self.ty_of(callee);
        if !self.cx.closure_takes_env(callee_ty) {
            self.expr(callee)?;
            return Ok(Vec::new());
        }
        let place = self.place_or_temp(callee)?;
        let env = self.address_or_dangling(place, callee_ty)?;
        if self.cx.closure_consumes(callee_ty)
            && let Some((root, path)) = self.scrutinee_path(callee)
        {
            self.moved_out(root, &path)?;
        }
        Ok(vec![env])
    }

    /// Lowers the body of `Fn::call` or `FnMut::call_mut` at a closure: the call `res` of the closure's body with the fields of the argument tuple.
    pub(super) fn untuple_call(&mut self, res: Res) -> Result<Option<Value>> {
        let Res::Fn(item, substs) = res else { bail!("a closure trait call that calls no function") };
        let tuple = self.typed.locals[self.typed.params[1] as usize].ty;
        let mut values = Vec::new();
        if self.cx.call_once() == Some(item) && self.cx.closure_takes_env(substs[0]) {
            let Some(receiver) = self.locals[self.typed.params[0] as usize] else { bail!("a closure trait call with no receiver") };
            values.extend(self.read(receiver));
        }
        let Kind::Tuple(items) = self.cx.types.kind(tuple).clone() else { bail!("closure arguments that are not a tuple") };
        if let Some(args) = self.locals[self.typed.params[1] as usize] {
            for index in 0..items.len() as u32 {
                let part = self.part_place(args, tuple, (0, index))?;
                values.extend(self.read(part));
            }
        }
        self.invoke(item, &substs, values)
    }

    fn object_call(&mut self, (expr, callee): (ExprId, ExprId), (item, substs): (ItemRef, Vec<Ty>), args: List) -> Result<Option<Value>> {
        if self.cx.call_once() == Some(item) {
            return self.boxed_once_call((expr, callee), (item, substs), args);
        }
        let Some((place, _)) = self.autoderef_place(expr, callee)? else { return Ok(None) };
        let receiver = self.address(place);
        let tuple = match self.typed.overloaded.contains(&expr) {
            true => {
                let items = self.body.list(args).iter().map(|&arg| self.ty_of(ExprId(arg))).collect();
                self.cx.types.intern(Kind::Tuple(items))
            }
            false => *substs.get(1).ok_or_else(|| anyhow::anyhow!("a trait object call with no argument tuple"))?,
        };
        let packed = self.tuple_of(tuple, args)?;
        self.invoke(item, &substs, std::iter::once(receiver).chain(packed).collect())
    }

    fn boxed_once_call(&mut self, (expr, callee): (ExprId, ExprId), (item, substs): (ItemRef, Vec<Ty>), args: List) -> Result<Option<Value>> {
        let Some((root, path)) = self.scrutinee_path(callee) else { bail!("calling a temporary `Box<dyn FnOnce>` is not supported yet") };
        let Some(Res::Fn(free, free_substs)) = self.typed.box_frees.get(&expr).cloned() else { bail!("a `Box<dyn FnOnce>` call with no box to free") };
        let Some(boxed) = self.place_or_temp(callee)? else { return Ok(None) };
        let Some((place, _)) = self.deref_step((expr, 0), Some(boxed), self.ty_of(callee))? else { return Ok(None) };
        let Some(slot) = crate::sema::infer::virtual_slot(self.cx, item, &substs)? else { bail!("`call_once` has no vtable slot") };
        let pair = self.address(place);
        let data = self.ins().load(Type::I64, pair, 0);
        let vtable = self.ins().load(Type::I64, pair, 8);
        let code = self.ins().load(Type::I64, vtable, slot as i32 * 8);
        let Some(&tuple) = substs.get(1) else { bail!("a trait object call with no argument tuple") };
        let packed = self.tuple_of(tuple, args)?;
        let ret = self.ty_of(expr);
        let unit = self.cx.types.unit();
        let pointer = self.cx.types.intern(Kind::Ptr(Mut::Mut, unit));
        let signature = super::signature_of(self.cx, Some(ret), [pointer, tuple])?;
        let result = self.call_code(code, (signature, Some(ret)), std::iter::once(data).chain(packed).collect())?;
        self.moved_out(root, &path)?;
        let target = self.address(boxed);
        self.invoke(free, &free_substs, vec![target])?;
        Ok(result)
    }

    fn pointed_env(&mut self, expr: ExprId, callee: ExprId) -> Result<Vec<Value>> {
        let Some((place, ty)) = self.autoderef_place(expr, callee)? else { return Ok(Vec::new()) };
        if self.cx.closure_consumes(ty) {
            bail!("calling a closure that consumes itself through a pointer is not supported yet");
        }
        if !self.cx.closure_takes_env(ty) {
            return Ok(Vec::new());
        }
        Ok(vec![self.address(place)])
    }

    /// Whether a call of `item` at `substs` is a trait call of a closure, whose tupled arguments are not lowered yet.
    pub(super) fn tupled_closure_call(&mut self, item: ItemRef, substs: &[Ty]) -> bool {
        substs.first().is_some_and(|ty| matches!(self.cx.types.kind(*ty), Kind::Closure(..))) && self.cx.call_once() == Some(item)
    }
}

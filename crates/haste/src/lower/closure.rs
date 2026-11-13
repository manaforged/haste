use anyhow::{Result, bail};
use haste_codegen::{Type, Value};

use crate::body::{ExprId, List};
use crate::names::ItemRef;
use crate::sema::infer::{By, Capture, ClosureBody, Res, Shape, needs_drop};
use crate::sema::substs::Substs;
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
        let shape = repr(self.cx, env)?;
        let Some(slot) = self.slot(shape) else {
            return Ok(None);
        };
        let object = Place::slot(slot, shape);
        let (whole, env) = match closure.coroutine {
            true => self.coroutine_header(ty, object, env)?,
            false => (object, env),
        };
        for (index, capture) in closure.captures.iter().enumerate() {
            let value = self.capture_value(capture)?;
            let part = self.part_place(whole, env, (0, index as u32))?;
            if let Some(value) = value {
                self.write(part, value);
            }
        }
        Ok(self.read(object))
    }

    /// Zeroes the state word and frame pointer of the coroutine object at `object`, and gives the place and type of its captures.
    fn coroutine_header(&mut self, ty: Ty, object: Place, env: Ty) -> Result<(Place, Ty)> {
        let zero = self.ins().iconst(Type::I64, 0);
        for word in 0..2 {
            let part = self.part_place(object, env, (0, word))?;
            self.write(part, zero);
        }
        let (inner, offset) = self.captures_of(ty)?;
        Ok((object.at(offset, repr(self.cx, inner)?), inner))
    }

    fn capture_value(&mut self, capture: &Capture) -> Result<Option<Value>> {
        let held = self.locals[capture.local as usize];
        let local_ty = self.typed.locals[capture.local as usize].ty;
        Ok(match capture.by {
            By::Value => {
                let value = held.and_then(|place| self.read(place));
                if needs_drop(self.cx, local_ty)? {
                    self.moved_out(Root::Local(capture.local), &[])?;
                }
                value
            }
            By::Ref | By::Mut => Some(self.address_or_dangling(held, local_ty)?),
        })
    }

    /// Points each local that the closure being lowered captures at its place in the environment.
    pub(super) fn bind_captures(&mut self) -> Result<()> {
        let Shape::Closure(ClosureBody { env: Some(local), closure, .. }) = self.typed.shape.clone() else {
            return Ok(());
        };
        let (whole, env) = self.env_place(local, closure.coroutine)?;
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

    /// The place of the captures tuple that the closure or coroutine parameter `local` points at, and its type.
    fn env_place(&mut self, local: u32, coroutine: bool) -> Result<(Place, Ty)> {
        let local_ty = self.typed.locals[local as usize].ty;
        let Some(held) = self.locals[local as usize] else { bail!("a closure environment with no place") };
        let (pointer, pointee) = match (coroutine, self.cx.types.kind(local_ty).clone()) {
            (true, Kind::Adt(_, args)) => {
                let first = self.cx.types.substs(args).first().copied();
                let field = self.part_place(held, local_ty, (0, 0))?;
                (self.read(field), first)
            }
            (false, Kind::Ref(_, env)) => (self.read(held), Some(env)),
            _ => bail!("a closure environment that is not a reference"),
        };
        let (Some(pointer), Some(pointee)) = (pointer, pointee) else { bail!("a closure environment with no address") };
        let pointee = match *self.cx.types.kind(pointee) {
            Kind::Ref(_, inner) => inner,
            _ => pointee,
        };
        if !coroutine {
            return Ok((Place { base: Base::Addr(pointer), offset: 0, repr: repr(self.cx, pointee)? }, pointee));
        }
        let (inner, offset) = self.captures_of(pointee)?;
        Ok((Place { base: Base::Addr(pointer), offset, repr: repr(self.cx, inner)? }, inner))
    }

    /// Lowers a call of the callee value `callee`, a closure, a function item, a trait object or a value with its own `Fn` impl, which inference resolved to `(item, substs)`.
    pub(super) fn call_value(&mut self, (expr, callee): (ExprId, ExprId), (item, substs): (ItemRef, Substs), args: List) -> Result<Option<Value>> {
        let item_value = matches!(self.cx.types.kind(self.ty_of(callee)), Kind::FnDef(..));
        let first = self.cx.types.substs(substs).first().copied();
        if self.by_impl(expr, (item, substs)) || !item_value && first.is_some_and(|ty| matches!(self.cx.types.kind(ty), Kind::Dyn(..))) {
            return self.object_call((expr, callee), (item, substs), args);
        }
        if self.typed.autoderefs.get(&expr) == Some(&1) && self.cx.boxed(self.ty_of(callee)).is_some_and(|inner| self.cx.closure_consumes(inner)) {
            return self.boxed_closure_call((expr, callee), (item, substs), args);
        }
        if self.typed.autoderefs.contains_key(&expr) {
            let values = self.pointed_env(expr, callee)?;
            let values = self.args(args, values)?;
            return self.invoke(super::reloc::SiteOrigin::Table(crate::sema::infer::SiteKey::ExprRes(expr)), item, substs, values);
        }
        let values = self.callee_env(callee)?;
        let values = self.args(args, values)?;
        self.invoke(super::reloc::SiteOrigin::Table(crate::sema::infer::SiteKey::ExprRes(expr)), item, substs, values)
    }

    pub(super) fn callee_env(&mut self, callee: ExprId) -> Result<Vec<Value>> {
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
        let env = if self.cx.closure_consumes(callee_ty)
            && self.may_alias_place(callee)
            && let Some(place) = place
            && matches!(place.repr, super::Repr::Mem { .. })
        {
            self.snapshot_repr(place.repr, env)
        } else {
            env
        };
        Ok(vec![env])
    }

    /// Lowers the body of `Fn::call` or `FnMut::call_mut` at a closure: the call `res` of the closure's body with the fields of the argument tuple.
    pub(super) fn untuple_call(&mut self, res: Res) -> Result<Option<Value>> {
        let (item, substs) = match res {
            Res::Fn(item, substs) => (item, substs),
            Res::Local(_) => return self.pointer_untuple_call(),
            _ => bail!("a closure trait call that calls no function"),
        };
        let mut values = Vec::new();
        let first = self.cx.types.substs(substs).first().copied();
        if self.cx.call_once() == Some(item) && first.is_some_and(|ty| self.cx.closure_takes_env(ty)) {
            let Some(receiver) = self.locals[self.typed.params[0] as usize] else { bail!("a closure trait call with no receiver") };
            values.extend(self.read(receiver));
            self.moved_out(Root::Local(self.typed.params[0]), &[])?;
        }
        values.extend(self.untupled()?.1);
        self.invoke(super::reloc::SiteOrigin::ShapeUntuple, item, substs, values)
    }

    /// Lowers the body of a call trait method at a function pointer: a call of the pointer that the receiver points at, with the fields of the argument tuple.
    fn pointer_untuple_call(&mut self) -> Result<Option<Value>> {
        let Some(address) = self.locals[self.typed.params[0] as usize].and_then(|place| self.read(place)) else { bail!("a function pointer call with no pointer") };
        let code = self.ins().load(Type::I64, address, 0);
        let (params, values) = self.untupled()?;
        self.call_code(code, (params, self.typed.ret), values)
    }

    /// The types of the fields of the argument tuple parameter of a call trait method, and their values, which then count as moved.
    fn untupled(&mut self) -> Result<(Vec<Ty>, Vec<Value>)> {
        let tuple = self.typed.locals[self.typed.params[1] as usize].ty;
        let Kind::Tuple(items) = *self.cx.types.kind(tuple) else { bail!("closure arguments that are not a tuple") };
        let types = self.cx.types.substs(items).to_vec();
        let mut values = Vec::new();
        if let Some(args) = self.locals[self.typed.params[1] as usize] {
            for index in 0..types.len() as u32 {
                let part = self.part_place(args, tuple, (0, index))?;
                values.extend(self.read(part));
            }
            self.moved_out(Root::Local(self.typed.params[1]), &[])?;
        }
        Ok((types, values))
    }

    fn object_call(&mut self, (expr, callee): (ExprId, ExprId), (item, substs): (ItemRef, Substs), args: List) -> Result<Option<Value>> {
        if self.cx.takes_by_value(item) && self.by_impl(expr, (item, substs)) {
            return self.owned_once_call((expr, callee), (item, substs), args);
        }
        if self.cx.call_once() == Some(item) {
            return self.boxed_once_call((expr, callee), (item, substs), args);
        }
        let Some((place, _)) = self.autoderef_place(expr, callee)? else {
            return Ok(None);
        };
        let receiver = self.address(place);
        let tuple = match self.typed.overloaded.contains(&expr) {
            true => self.args_tuple(args),
            false => self.cx.types.substs(substs).get(1).copied().ok_or_else(|| anyhow::anyhow!("a trait object call with no argument tuple"))?,
        };
        let packed = self.tuple_of(tuple, args)?;
        self.invoke(super::reloc::SiteOrigin::Table(crate::sema::infer::SiteKey::ExprRes(expr)), item, substs, std::iter::once(receiver).chain(packed).collect())
    }

    /// Lowers the call of a value whose own `FnOnce` impl takes it by value, as `AssertUnwindSafe` has: the value moves into `call_once` with the argument tuple.
    fn owned_once_call(&mut self, (expr, callee): (ExprId, ExprId), (item, substs): (ItemRef, Substs), args: List) -> Result<Option<Value>> {
        if self.typed.autoderefs.contains_key(&expr) {
            bail!("calling a value through a pointer by `FnOnce` is not supported yet");
        }
        let place = self.place_or_temp(callee)?;
        let mut values: Vec<Value> = place.and_then(|place| self.read(place)).into_iter().collect();
        if let Some((root, path)) = self.scrutinee_path(callee) {
            self.moved_out(root, &path)?;
        }
        let tuple = self.args_tuple(args);
        values.extend(self.tuple_of(tuple, args)?);
        self.invoke(super::reloc::SiteOrigin::Table(crate::sema::infer::SiteKey::ExprRes(expr)), item, substs, values)
    }

    /// Whether the call `expr` resolved to `(item, substs)` goes through the callee type's own `Fn` impl: inference saw that impl, or a generic body calls a call trait method at a struct such as `AssertUnwindSafe`.
    fn by_impl(&mut self, expr: ExprId, (item, substs): (ItemRef, Substs)) -> bool {
        let trait_call = self.cx.call_once() == Some(item) || crate::sema::infer::by_reference(self.cx, item).is_some();
        let first = self.cx.types.substs(substs).first().copied();
        self.typed.overloaded.contains(&expr) || trait_call && first.is_some_and(|ty| matches!(self.cx.types.kind(ty), Kind::Adt(..)))
    }

    /// The tuple type of the call arguments `args`.
    fn args_tuple(&mut self, args: List) -> Ty {
        let items: Vec<Ty> = self.body.list(args).iter().map(|&arg| self.ty_of(ExprId(arg))).collect();
        self.cx.types.tuple(&items)
    }

    fn boxed_once_call(&mut self, (expr, callee): (ExprId, ExprId), (item, substs): (ItemRef, Substs), args: List) -> Result<Option<Value>> {
        let Some((root, path)) = self.scrutinee_path(callee) else { bail!("calling a temporary `Box<dyn FnOnce>` is not supported yet") };
        let Some(&Res::Fn(free, free_substs)) = self.typed.box_frees.get(&expr) else { bail!("a `Box<dyn FnOnce>` call with no box to free") };
        let Some(boxed) = self.place_or_temp(callee)? else { return Ok(None) };
        let Some((place, _)) = self.deref_step((expr, 0), Some(boxed), self.ty_of(callee))? else { return Ok(None) };
        let Some(slot) = crate::sema::infer::virtual_slot(self.cx, item, substs)? else { bail!("`call_once` has no vtable slot") };
        let pair = self.address(place);
        let data = self.ins().load(Type::I64, pair, 0);
        let vtable = self.ins().load(Type::I64, pair, 8);
        let code = self.ins().load(Type::I64, vtable, slot as i32 * 8);
        let Some(&tuple) = self.cx.types.substs(substs).get(1) else { bail!("a trait object call with no argument tuple") };
        let packed = self.tuple_of(tuple, args)?;
        let ret = self.ty_of(expr);
        let unit = self.cx.types.unit();
        let pointer = self.cx.types.intern(Kind::Ptr(Mut::Mut, unit));
        let result = self.call_code(code, (vec![pointer, tuple], Some(ret)), std::iter::once(data).chain(packed).collect())?;
        self.moved_out(root, &path)?;
        let target = self.address(boxed);
        self.invoke(super::reloc::SiteOrigin::Table(crate::sema::infer::SiteKey::BoxFree(expr)), free, free_substs, vec![target])?;
        Ok(result)
    }

    /// Lowers the call of a `Box` of a closure that consumes itself: the closure runs on its captures inside the box, and then the box frees its memory and counts as moved.
    fn boxed_closure_call(&mut self, (expr, callee): (ExprId, ExprId), (item, substs): (ItemRef, Substs), args: List) -> Result<Option<Value>> {
        let Some((root, path)) = self.scrutinee_path(callee) else { bail!("calling a `Box` of a closure that no local owns is not supported yet") };
        let boxed = self.ty_of(callee);
        let Some(Res::Fn(free, free_substs)) = crate::sema::infer::drop_fn(self.cx, boxed)? else { bail!("`Box` has no `Drop` impl") };
        let Some(owner) = self.place_or_temp(callee)? else { return Ok(None) };
        let Some((place, closure)) = self.deref_step((expr, 0), Some(owner), boxed)? else { return Ok(None) };
        let env = if self.cx.closure_takes_env(closure) { vec![self.address(place)] } else { Vec::new() };
        let values = self.args(args, env)?;
        let result = self.invoke(super::reloc::SiteOrigin::Table(crate::sema::infer::SiteKey::ExprRes(expr)), item, substs, values)?;
        let target = self.address(owner);
        self.invoke(super::reloc::SiteOrigin::DropFn(super::reloc::TyOrigin::Expr(callee)), free, free_substs, vec![target])?;
        self.moved_out(root, &path)?;
        Ok(result)
    }

    fn pointed_env(&mut self, expr: ExprId, callee: ExprId) -> Result<Vec<Value>> {
        let Some((place, ty)) = self.autoderef_place(expr, callee)? else {
            return Ok(Vec::new());
        };
        if self.cx.closure_consumes(ty) {
            bail!("calling a closure that consumes itself through a pointer is not supported yet");
        }
        if !self.cx.closure_takes_env(ty) {
            return Ok(Vec::new());
        }
        Ok(vec![self.address(place)])
    }

    pub(super) fn tupled_closure_call(&mut self, item: ItemRef, substs: Substs) -> bool {
        let first = self.cx.types.substs(substs).first().copied();
        first.is_some_and(|ty| matches!(self.cx.types.kind(ty), Kind::Closure(..))) && self.cx.call_once() == Some(item)
    }

    pub(super) fn untuple_arg(&mut self, tuple_expr: ExprId, values: &mut Vec<Value>) -> Result<()> {
        let tuple_ty = self.ty_of(tuple_expr);
        let Kind::Tuple(items) = *self.cx.types.kind(tuple_ty) else { bail!("closure arguments that are not a tuple") };
        let count = self.cx.types.substs(items).len() as u32;
        if let Some(place) = self.place_or_temp(tuple_expr)? {
            for index in 0..count {
                let part = self.part_place(place, tuple_ty, (0, index))?;
                values.extend(self.read(part));
            }
        }
        if let Some((root, path)) = self.scrutinee_path(tuple_expr) {
            self.moved_out(root, &path)?;
        }
        Ok(())
    }
}

use anyhow::{Result, bail};
use haste_codegen::{Coroutine, IntCC, Type, Value};

use crate::body::{Body, Expr, ExprId, List, Stmt};
use crate::names::ItemRef;
use crate::sema::infer::{By, Capture, ClosureBody, Res, Shape, Typed, needs_drop};
use crate::sema::substs::Substs;
use crate::sema::ty::{Kind, Mut, Ty};

use super::moves::Root;
use super::place::{Base, Place, transparent_in};
use super::{Lower, Repr, repr};

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
            if let Some(value) = self.capture_value(capture)? {
                let part = self.part_place(whole, env, (0, index as u32))?;
                self.write(part, value);
            }
        }
        Ok(self.read(object))
    }

    /// Initializes the state word to `Coroutine::UNRESUMED` and zeroes the frame pointer of the coroutine object at `object`, and gives the place and type of its captures.
    fn coroutine_header(&mut self, ty: Ty, object: Place, env: Ty) -> Result<(Place, Ty)> {
        let init_state = self.ins().iconst(Type::I64, Coroutine::UNRESUMED);
        let state_part = self.part_place(object, env, (0, 0))?;
        self.write(state_part, init_state);
        let null_frame = self.ins().iconst(Type::I64, 0);
        let frame_part = self.part_place(object, env, (0, 1))?;
        self.write(frame_part, null_frame);
        let (inner, offset) = self.captures_of(ty)?;
        Ok((object.at(offset, repr(self.cx, inner)?), inner))
    }

    fn capture_source_place(&mut self, capture: &Capture) -> Result<(u32, Vec<(u32, u32)>, Option<Place>, Ty)> {
        let (root_local, rel_path, mut place, mut ty) = if self.locals[capture.local as usize].is_none()
            && let Some((slot_local, place, k)) = (1..=capture.path.len()).rev().find_map(|k| {
                self.capture_by_path(capture.local, &capture.path[..k]).map(|(s, p)| (s, p, k))
            })
        {
            (slot_local, capture.path[k..].to_vec(), place, self.typed.locals[slot_local as usize].ty)
        } else {
            (capture.local, capture.path.clone(), self.locals[capture.local as usize], self.typed.locals[capture.local as usize].ty)
        };
        for &step in &rel_path {
            place = place.map(|p| self.part_place(p, ty, step)).transpose()?;
            ty = Capture::step_ty(self.cx, ty, step)?;
        }
        Ok((root_local, rel_path, place, ty))
    }

    fn capture_value(&mut self, capture: &Capture) -> Result<Option<Value>> {
        let (root_local, rel_path, held, cap_ty) = self.capture_source_place(capture)?;
        Ok(match capture.by {
            By::Value => {
                let value = held.and_then(|place| self.read(place));
                if needs_drop(self.cx, cap_ty)? {
                    self.moved_out(Root::Local(root_local), &rel_path)?;
                }
                value
            }
            By::Ref | By::Mut => Some(self.address_or_dangling(held, cap_ty)?),
        })
    }

    fn projected_capture_local(closure: &crate::sema::infer::Closure, env_local: u32, index: usize) -> u32 {
        let ordinal = closure.captures[..index].iter().filter(|c| !c.path.is_empty()).count() as u32;
        env_local + 1 + u32::from(closure.coroutine) + ordinal
    }

    pub(super) fn capture_local_by_path(&self, local: u32, path: &[(u32, u32)]) -> Option<u32> {
        if path.is_empty() {
            return None;
        }
        let Shape::Closure(ClosureBody { env: Some(env_local), closure, .. }) = &self.typed.shape else {
            return None;
        };
        let index = closure.captures.iter().position(|c| c.local == local && c.path == path)?;
        Some(Self::projected_capture_local(closure, *env_local, index))
    }

    pub(super) fn capture_by_path(&self, local: u32, path: &[(u32, u32)]) -> Option<(u32, Option<Place>)> {
        let cap_local = self.capture_local_by_path(local, path)?;
        Some((cap_local, self.locals[cap_local as usize]))
    }

    pub(super) fn expr_capture_path(&self, expr: ExprId) -> Option<(u32, Vec<(u32, u32)>)> {
        expr_capture_path_in(self.body, self.typed, expr)
    }

    /// Points each local that the closure being lowered captures at its place in the environment.
    pub(super) fn bind_captures(&mut self) -> Result<()> {
        let Shape::Closure(ClosureBody { env: Some(local), closure, .. }) = self.typed.shape.clone() else {
            return Ok(());
        };
        let (whole, env) = self.env_place(local, closure.coroutine)?;
        let mut to_own = Vec::new();
        for (index, capture) in closure.captures.iter().enumerate() {
            let part = self.part_place(whole, env, (0, index as u32))?;
            let cap_ty = capture.ty(self.cx, self.typed.locals[capture.local as usize].ty)?;
            let slot_local = if capture.path.is_empty() { capture.local } else { Self::projected_capture_local(&closure, local, index) };
            if closure.consumes && capture.by == By::Value {
                to_own.push((index, slot_local, cap_ty, part));
            }
            self.locals[slot_local as usize] = Some(match capture.by {
                By::Value => part,
                By::Ref | By::Mut => {
                    let Some(address) = self.read(part) else { bail!("a captured reference with no value") };
                    Place { base: Base::Addr(address), offset: 0, repr: repr(self.cx, cap_ty)? }
                }
            });
        }
        if !to_own.is_empty() {
            let order = capture_drop_order(self.body, self.typed, &closure);
            for idx in order.into_iter().rev() {
                if let Some(&(_, slot_local, cap_ty, part)) = to_own.iter().find(|entry| entry.0 == idx) {
                    self.own(Root::Local(slot_local), cap_ty, Some(part), true)?;
                }
            }
        }
        Ok(())
    }

    pub(super) fn drop_closure_parts(&mut self, ty: Ty, target: Value, origin: super::reloc::TyOrigin) -> Result<()> {
        let Some(info) = self.cx.closure(ty) else { bail!("drop of a closure whose defining body was not inferred") };
        let env_origin = origin.step(super::reloc::CLOSURE_ENV_STEP, 0);
        if info.closure.coroutine {
            return self.drop_parts(info.env, target, env_origin);
        }
        let Kind::Tuple(items) = *self.cx.types.kind(info.env) else {
            return self.drop_parts(info.env, target, env_origin);
        };
        let fields = self.cx.types.substs(items).to_vec();
        let body = self.cx.body(info.parent)?;
        let order = capture_drop_order(&body, &info.typed, &info.closure);
        let place = self.glue_place(info.env, target)?;
        for idx in order {
            let field_ty = fields[idx];
            let part = self.part_place(place, info.env, (0, idx as u32))?;
            self.glue(field_ty, Some(part), env_origin.clone().step(0, idx as u32))?;
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
        let callee_ty = self.ty_of(callee);
        let by_value = self.cx.closure_consumes(callee_ty);
        let values = self.callee_env(callee, by_value)?;
        let guards = if !self.body.list(args).is_empty() {
            self.guard_consumed_callee(callee, &values, by_value)?
        } else {
            Vec::new()
        };
        let values = self.eval_guarded_args(args, values, guards, true)?;
        self.invoke(super::reloc::SiteOrigin::Table(crate::sema::infer::SiteKey::ExprRes(expr)), item, substs, values)
    }

    pub(super) fn guard_consumed_callee(&mut self, callee: ExprId, values: &[Value], by_value: bool) -> Result<Vec<usize>> {
        let mut guards = Vec::new();
        let callee_ty = self.ty_of(callee);
        if by_value
            && self.cx.closure_takes_env(callee_ty)
            && self.wants_field_unwind()?
            && let Some(&env) = values.first()
        {
            let place = Place { base: Base::Addr(env), offset: 0, repr: repr(self.cx, callee_ty)? };
            if let Some(guard) = self.push_owned_guard(callee_ty, Some(place), callee)? {
                guards.push(guard);
            }
        }
        Ok(guards)
    }

    pub(super) fn callee_env(&mut self, callee: ExprId, by_value: bool) -> Result<Vec<Value>> {
        let callee_ty = self.ty_of(callee);
        if !self.cx.closure_takes_env(callee_ty) {
            self.expr(callee)?;
            return Ok(Vec::new());
        }
        let place = self.place_or_temp(callee)?;
        if !self.open {
            return Ok(Vec::new());
        }
        let env = self.address_or_dangling(place, callee_ty)?;
        if self.cx.closure_consumes(callee_ty) && let Some((root, path)) = self.scrutinee_path(callee) {
            self.moved_out(root, &path)?;
        }
        let env = match (place, self.needs_callee_snapshot(callee, by_value, place)) {
            (Some(place), true) => self.snapshot_repr(place.repr, env),
            _ => env,
        };
        Ok(vec![env])
    }

    fn needs_callee_snapshot(&self, callee: ExprId, by_value: bool, place: Option<Place>) -> bool {
        let Some(place) = place else {
            return false;
        };
        if !by_value || !matches!(place.repr, Repr::Mem { .. }) || !self.may_alias_place(callee) {
            return false;
        }
        !self.unaliased_once_local(callee)
    }

    fn unaliased_once_local(&self, mut callee: ExprId) -> bool {
        while let Some(inner) = self.transparent(callee) {
            callee = inner;
        }
        let Some(&Res::Local(local)) = self.typed.res.get(&callee).filter(|_| matches!(*self.body.expr(callee), Expr::Path(_))) else {
            return false;
        };
        if !self.loops.is_empty() {
            return false;
        }
        if self.typed.closures.values().any(|closure| closure.captures.iter().any(|cap| cap.local == local)) {
            return false;
        }
        let uses = self.typed.res.iter().filter(|(_, res)| matches!(res, Res::Local(id) if *id == local)).count();
        uses == 1
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

    fn mark_boxed_moved(&mut self, moved: Option<(Root, super::moves::MovePath)>, callee: ExprId) -> Result<bool> {
        if !self.open {
            return Ok(false);
        }
        match moved {
            Some((root, path)) => self.moved_out(root, &path)?,
            None => self.moved_out(Root::Temp(callee), &[])?,
        }
        Ok(true)
    }

    fn boxed_once_call(&mut self, (expr, callee): (ExprId, ExprId), (item, substs): (ItemRef, Substs), args: List) -> Result<Option<Value>> {
        let moved = self.scrutinee_path(callee);
        let box_ty = self.ty_of(callee);
        let Some(&Res::Fn(free, free_substs)) = self.typed.box_frees.get(&expr) else { bail!("a `Box<dyn FnOnce>` call with no box to free") };
        let Some(boxed) = self.place_or_temp(callee)? else { return Ok(None) };
        let Some((place, inner_ty)) = self.deref_step((expr, 0), Some(boxed), box_ty)? else { return Ok(None) };
        let Some(slot) = crate::sema::infer::virtual_slot(self.cx, item, substs)? else { bail!("`call_once` has no vtable slot") };
        let pair = self.address(place);
        let data = self.ins().load(Type::I64, pair, 0);
        let vtable = self.ins().load(Type::I64, pair, 8);
        let code = self.ins().load(Type::I64, vtable, slot as i32 * 8);
        let Some(&tuple) = self.cx.types.substs(substs).get(1) else { bail!("a trait object call with no argument tuple") };
        let packed = self.tuple_of(tuple, args)?;
        if !self.mark_boxed_moved(moved, callee)? {
            return Ok(None);
        }
        let need_drop = if matches!(*self.cx.types.kind(inner_ty), Kind::Dyn(of, ..) if self.cx.lang(of) == Some("fn_once"))
            && crate::prep::vtable(self.cx, inner_ty).is_none()
        {
            let already = self.ins().load(Type::I64, vtable, 32);
            self.ins().icmp_imm(IntCC::Equal, already, 0)
        } else {
            let flag = i64::from(crate::prep::vtable(self.cx, inner_ty).is_none());
            self.ins().iconst(Type::I8, flag)
        };
        let mut guards = Vec::new();
        if self.wants_field_unwind()?
            && let Some(guard) = self.push_owned_guard(box_ty, Some(boxed), callee)?
        {
            let flag = self.owned[guard].flag;
            let unwind = self.owned[guard].unwind;
            self.set_dynamic_active(flag, unwind, need_drop);
            guards.push(guard);
        }
        let ret = self.ty_of(expr);
        let unit = self.cx.types.unit();
        let pointer = self.cx.types.intern(Kind::Ptr(Mut::Mut, unit));
        let result = self.call_code(code, (vec![pointer, tuple], Some(ret)), std::iter::once(data).chain(packed).collect())?;
        self.disarm_owned_guards(guards);
        if self.open {
            self.when(need_drop, |lower| lower.drop_object(pair))?;
            let target = self.address(boxed);
            self.invoke(super::reloc::SiteOrigin::Table(crate::sema::infer::SiteKey::BoxFree(expr)), free, free_substs, vec![target])?;
        }
        Ok(result)
    }

    /// Lowers the call of a `Box` of a closure that consumes itself: the closure runs on its captures inside the box, and then the box frees its memory and counts as moved.
    fn boxed_closure_call(&mut self, (expr, callee): (ExprId, ExprId), (item, substs): (ItemRef, Substs), args: List) -> Result<Option<Value>> {
        let moved = self.scrutinee_path(callee);
        let boxed = self.ty_of(callee);
        let Some(Res::Fn(free, free_substs)) = crate::sema::infer::drop_fn(self.cx, boxed)? else { bail!("`Box` has no `Drop` impl") };
        let Some(owner) = self.place_or_temp(callee)? else { return Ok(None) };
        let Some((place, closure)) = self.deref_step((expr, 0), Some(owner), boxed)? else { return Ok(None) };
        let env = if self.cx.closure_takes_env(closure) { vec![self.address(place)] } else { Vec::new() };
        let values = self.args(args, env)?;
        if !self.mark_boxed_moved(moved, callee)? {
            return Ok(None);
        }
        let result = self.invoke(super::reloc::SiteOrigin::Table(crate::sema::infer::SiteKey::ExprRes(expr)), item, substs, values)?;
        let target = self.address(owner);
        self.invoke(super::reloc::SiteOrigin::DropFn(super::reloc::TyOrigin::Expr(callee)), free, free_substs, vec![target])?;
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

    pub(super) fn untuple_arg_guarded(&mut self, tuple_expr: ExprId, values: &mut Vec<Value>, guards: Vec<usize>) -> Result<()> {
        let mut cur = tuple_expr;
        while !self.typed.coercions.contains_key(&cur)
            && let Some(inner) = self.transparent(cur)
        {
            cur = inner;
        }
        if !self.typed.coercions.contains_key(&cur)
            && let Expr::Tuple(list) = *self.body.expr(cur)
        {
            *values = self.eval_guarded_args(list, std::mem::take(values), guards, false)?;
            return Ok(());
        }
        let val = self.eval_operand(tuple_expr)?;
        self.disarm_owned_guards(guards);
        if !self.open {
            return Ok(());
        }
        let tuple_ty = self.ty_of(tuple_expr);
        let Kind::Tuple(items) = *self.cx.types.kind(tuple_ty) else { bail!("closure arguments that are not a tuple") };
        let count = self.cx.types.substs(items).len() as u32;
        if let Some(place) = self.spill(val, tuple_ty)? {
            for index in 0..count {
                let part = self.part_place(place, tuple_ty, (0, index))?;
                values.extend(self.read(part));
            }
        }
        Ok(())
    }
}

fn expr_capture_path_in(body: &Body, typed: &Typed, mut expr: ExprId) -> Option<(u32, Vec<(u32, u32)>)> {
    while let Some(inner) = transparent_in(body, expr) {
        expr = inner;
    }
    match *body.expr(expr) {
        Expr::Path(_) => match typed.res.get(&expr) {
            Some(&Res::Local(local)) => Some((local, Vec::new())),
            _ => None,
        },
        Expr::TupleField(base, index) if !typed.autoderefs.contains_key(&expr) => {
            let (local, mut path) = expr_capture_path_in(body, typed, base)?;
            path.push((0, index));
            Some((local, path))
        }
        Expr::Field(base, _) if !typed.autoderefs.contains_key(&expr) => {
            let index = *typed.fields.get(&expr)?;
            let (local, mut path) = expr_capture_path_in(body, typed, base)?;
            path.push((0, index));
            Some((local, path))
        }
        _ => None,
    }
}

fn capture_drop_order(body: &Body, typed: &Typed, closure: &crate::sema::infer::Closure) -> Vec<usize> {
    let mut order = Vec::with_capacity(closure.captures.len());
    let mut seen = vec![false; closure.captures.len()];
    visit_capture_order(body, typed, closure, closure.body, &mut seen, &mut order);
    for (idx, was_seen) in seen.into_iter().enumerate() {
        if !was_seen {
            order.push(idx);
        }
    }
    order
}

fn visit_capture_order(
    body: &Body,
    typed: &Typed,
    closure: &crate::sema::infer::Closure,
    expr: ExprId,
    seen: &mut [bool],
    order: &mut Vec<usize>,
) {
    if let Some((local, path)) = expr_capture_path_in(body, typed, expr)
        && let Some(idx) = closure.captures.iter().position(|cap| cap.local == local && path.starts_with(&cap.path))
    {
        if !seen[idx] {
            seen[idx] = true;
            order.push(idx);
        }
        return;
    }
    let visit_list = |list: List, seen: &mut [bool], order: &mut Vec<usize>| {
        for &at in body.list(list) {
            visit_capture_order(body, typed, closure, ExprId(at), seen, order);
        }
    };
    match *body.expr(expr) {
        Expr::Unary(_, e)
        | Expr::Field(e, _)
        | Expr::TupleField(e, _)
        | Expr::Cast(e, _)
        | Expr::Ref { expr: e, .. }
        | Expr::Try(e)
        | Expr::Await(e)
        | Expr::Become(e)
        | Expr::Paren(e) => visit_capture_order(body, typed, closure, e, seen, order),
        Expr::Binary(_, a, b) | Expr::Assign(a, b) | Expr::AssignOp(_, a, b) | Expr::Index(a, b) | Expr::Repeat(a, b) => {
            visit_capture_order(body, typed, closure, a, seen, order);
            visit_capture_order(body, typed, closure, b, seen, order);
        }
        Expr::Call(callee, list) => {
            visit_capture_order(body, typed, closure, callee, seen, order);
            visit_list(list, seen, order);
        }
        Expr::Method { receiver, params, .. } => {
            visit_capture_order(body, typed, closure, receiver, seen, order);
            visit_list(params, seen, order);
        }
        Expr::Block { stmts, tail, .. } => {
            for &at in body.list(stmts) {
                match body.stmts[at as usize] {
                    Stmt::Let { init, otherwise, .. } => {
                        if let Some(init) = init {
                            visit_capture_order(body, typed, closure, init, seen, order);
                        }
                        if let Some(otherwise) = otherwise {
                            visit_capture_order(body, typed, closure, otherwise, seen, order);
                        }
                    }
                    Stmt::Expr(e, _) => visit_capture_order(body, typed, closure, e, seen, order),
                    Stmt::Item(..) | Stmt::Macro { .. } | Stmt::Empty => {}
                }
            }
            if let Some(tail) = tail {
                visit_capture_order(body, typed, closure, tail, seen, order);
            }
        }
        Expr::If(cond, then, otherwise) => {
            visit_capture_order(body, typed, closure, cond, seen, order);
            visit_capture_order(body, typed, closure, then, seen, order);
            if let Some(otherwise) = otherwise {
                visit_capture_order(body, typed, closure, otherwise, seen, order);
            }
        }
        Expr::Let(_, init) => visit_capture_order(body, typed, closure, init, seen, order),
        Expr::While { cond, body: inner, .. } => {
            visit_capture_order(body, typed, closure, cond, seen, order);
            visit_capture_order(body, typed, closure, inner, seen, order);
        }
        Expr::Loop { body: inner, .. } | Expr::Closure { body: inner, .. } => {
            visit_capture_order(body, typed, closure, inner, seen, order);
        }
        Expr::For { iter, body: inner, .. } => {
            visit_capture_order(body, typed, closure, iter, seen, order);
            visit_capture_order(body, typed, closure, inner, seen, order);
        }
        Expr::Match(scrut, arms) => {
            visit_capture_order(body, typed, closure, scrut, seen, order);
            for &at in body.list(arms) {
                let arm = body.arms[at as usize];
                if let Some(guard) = arm.guard {
                    visit_capture_order(body, typed, closure, guard, seen, order);
                }
                visit_capture_order(body, typed, closure, arm.body, seen, order);
            }
        }
        Expr::Tuple(list) | Expr::Array(list) => visit_list(list, seen, order),
        Expr::Struct { fields, base, .. } => {
            for &at in body.list(fields) {
                visit_capture_order(body, typed, closure, body.fields[at as usize].value, seen, order);
            }
            if let Some(base) = base {
                visit_capture_order(body, typed, closure, base, seen, order);
            }
        }
        Expr::Range { from, to, .. } => {
            if let Some(from) = from {
                visit_capture_order(body, typed, closure, from, seen, order);
            }
            if let Some(to) = to {
                visit_capture_order(body, typed, closure, to, seen, order);
            }
        }
        Expr::Return(e) | Expr::Break(_, e) | Expr::Yield(e) => {
            if let Some(e) = e {
                visit_capture_order(body, typed, closure, e, seen, order);
            }
        }
        Expr::Lit(..)
        | Expr::Path(_)
        | Expr::Continue(_)
        | Expr::Macro { .. }
        | Expr::Underscore
        | Expr::FormatArgs(_)
        | Expr::OffsetOf { .. } => {}
    }
}

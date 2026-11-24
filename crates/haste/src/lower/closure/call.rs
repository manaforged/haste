use anyhow::{Result, bail};
use haste_codegen::{IntCC, Type, Value};

use crate::body::{Expr, ExprId, List};
use crate::names::ItemRef;
use crate::sema::infer::{Res, SiteKey};
use crate::sema::substs::Substs;
use crate::sema::ty::{Kind, Mut, Ty};

use super::super::moves::{MovePath, Root};
use super::super::place::{Base, Place};
use super::super::reloc::{SiteOrigin, TyOrigin};
use super::super::{Lower, Repr, repr};

impl Lower<'_, '_> {
    /// Lowers a call of the callee value `callee`, a closure, a function item, a trait object or a value with its own `Fn` impl, which inference resolved to `(item, substs)`.
    pub(in super::super) fn call_value(
        &mut self,
        (expr, callee): (ExprId, ExprId),
        (item, substs): (ItemRef, Substs),
        args: List,
    ) -> Result<Option<Value>> {
        let item_value = matches!(self.cx.types.kind(self.ty_of(callee)), Kind::FnDef(..));
        let first = self.cx.types.substs(substs).first().copied();
        if self.by_impl(expr, (item, substs))
            || !item_value
                && first.is_some_and(|ty| matches!(self.cx.types.kind(ty), Kind::Dyn(..)))
        {
            return self.object_call((expr, callee), (item, substs), args);
        }
        if self.typed.autoderefs.get(&expr) == Some(&1)
            && self
                .cx
                .boxed(self.ty_of(callee))
                .is_some_and(|inner| self.cx.closure_consumes(inner))
        {
            return self.boxed_closure_call((expr, callee), (item, substs), args);
        }
        if self.typed.autoderefs.contains_key(&expr) {
            let values = self.pointed_env(expr, callee)?;
            let values = self.args(args, values)?;
            return self.invoke(
                SiteOrigin::Table(SiteKey::ExprRes(expr)),
                item,
                substs,
                values,
            );
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
        self.invoke(
            SiteOrigin::Table(SiteKey::ExprRes(expr)),
            item,
            substs,
            values,
        )
    }

    pub(in super::super) fn guard_consumed_callee(
        &mut self,
        callee: ExprId,
        values: &[Value],
        by_value: bool,
    ) -> Result<Vec<usize>> {
        let mut guards = Vec::new();
        let callee_ty = self.ty_of(callee);
        if by_value
            && self.cx.closure_takes_env(callee_ty)
            && self.wants_field_unwind()?
            && let Some(&env) = values.first()
        {
            let place = Place {
                base: Base::Addr(env),
                offset: 0,
                repr: repr(self.cx, callee_ty)?,
            };
            if let Some(guard) = self.push_owned_guard(callee_ty, Some(place), callee)? {
                guards.push(guard);
            }
        }
        Ok(guards)
    }

    pub(in super::super) fn callee_env(
        &mut self,
        callee: ExprId,
        by_value: bool,
    ) -> Result<Vec<Value>> {
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
        if self.cx.closure_consumes(callee_ty)
            && let Some((root, path)) = self.scrutinee_path(callee)
        {
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
        let Some(&Res::Local(local)) = self
            .typed
            .res
            .get(&callee)
            .filter(|_| matches!(*self.body.expr(callee), Expr::Path(_)))
        else {
            return false;
        };
        if !self.loops.is_empty() {
            return false;
        }
        if self
            .typed
            .closures
            .values()
            .any(|closure| closure.captures.iter().any(|cap| cap.local == local))
        {
            return false;
        }
        let uses = self
            .typed
            .res
            .iter()
            .filter(|(_, res)| matches!(res, Res::Local(id) if *id == local))
            .count();
        uses == 1
    }

    /// Lowers the body of `Fn::call` or `FnMut::call_mut` at a closure: the call `res` of the closure's body with the fields of the argument tuple.
    pub(in super::super) fn untuple_call(&mut self, res: Res) -> Result<Option<Value>> {
        let (item, substs) = match res {
            Res::Fn(item, substs) => (item, substs),
            Res::Local(_) => return self.pointer_untuple_call(),
            _ => bail!("a closure trait call that calls no function"),
        };
        let mut values = Vec::new();
        let first = self.cx.types.substs(substs).first().copied();
        if self.cx.call_once() == Some(item)
            && first.is_some_and(|ty| self.cx.closure_takes_env(ty))
        {
            let Some(receiver) = self.locals[self.typed.params[0] as usize] else {
                bail!("a closure trait call with no receiver");
            };
            values.extend(self.read(receiver));
            self.moved_out(Root::Local(self.typed.params[0]), &[])?;
        }
        values.extend(self.untupled()?.1);
        self.invoke(SiteOrigin::ShapeUntuple, item, substs, values)
    }

    fn pointer_untuple_call(&mut self) -> Result<Option<Value>> {
        let Some(address) =
            self.locals[self.typed.params[0] as usize].and_then(|place| self.read(place))
        else {
            bail!("a function pointer call with no pointer");
        };
        let code = self.ins().load(Type::I64, address, 0);
        let (params, values) = self.untupled()?;
        self.call_code(code, (params, self.typed.ret), values)
    }

    fn untupled(&mut self) -> Result<(Vec<Ty>, Vec<Value>)> {
        let tuple = self.typed.locals[self.typed.params[1] as usize].ty;
        let Kind::Tuple(items) = *self.cx.types.kind(tuple) else {
            bail!("closure arguments that are not a tuple");
        };
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

    fn object_call(
        &mut self,
        (expr, callee): (ExprId, ExprId),
        (item, substs): (ItemRef, Substs),
        args: List,
    ) -> Result<Option<Value>> {
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
            false => self
                .cx
                .types
                .substs(substs)
                .get(1)
                .copied()
                .ok_or_else(|| anyhow::anyhow!("a trait object call with no argument tuple"))?,
        };
        let packed = self.tuple_of(tuple, args)?;
        self.invoke(
            SiteOrigin::Table(SiteKey::ExprRes(expr)),
            item,
            substs,
            std::iter::once(receiver).chain(packed).collect(),
        )
    }

    fn owned_once_call(
        &mut self,
        (expr, callee): (ExprId, ExprId),
        (item, substs): (ItemRef, Substs),
        args: List,
    ) -> Result<Option<Value>> {
        if self.typed.autoderefs.contains_key(&expr) {
            bail!("calling a value through a pointer by `FnOnce` is not supported yet");
        }
        let place = self.place_or_temp(callee)?;
        let mut values: Vec<Value> = place
            .and_then(|place| self.read(place))
            .into_iter()
            .collect();
        if let Some((root, path)) = self.scrutinee_path(callee) {
            self.moved_out(root, &path)?;
        }
        let tuple = self.args_tuple(args);
        values.extend(self.tuple_of(tuple, args)?);
        self.invoke(
            SiteOrigin::Table(SiteKey::ExprRes(expr)),
            item,
            substs,
            values,
        )
    }

    fn by_impl(&mut self, expr: ExprId, (item, substs): (ItemRef, Substs)) -> bool {
        let trait_call = self.cx.call_once() == Some(item)
            || crate::sema::infer::by_reference(self.cx, item).is_some();
        let first = self.cx.types.substs(substs).first().copied();
        self.typed.overloaded.contains(&expr)
            || trait_call && first.is_some_and(|ty| matches!(self.cx.types.kind(ty), Kind::Adt(..)))
    }

    fn args_tuple(&mut self, args: List) -> Ty {
        let items: Vec<Ty> = self
            .body
            .list(args)
            .iter()
            .map(|&arg| self.ty_of(ExprId(arg)))
            .collect();
        self.cx.types.tuple(&items)
    }

    fn mark_boxed_moved(
        &mut self,
        moved: Option<(Root, MovePath)>,
        callee: ExprId,
    ) -> Result<bool> {
        if !self.open {
            return Ok(false);
        }
        match moved {
            Some((root, path)) => self.moved_out(root, &path)?,
            None => self.moved_out(Root::Temp(callee), &[])?,
        }
        Ok(true)
    }

    fn boxed_once_call(
        &mut self,
        (expr, callee): (ExprId, ExprId),
        (item, substs): (ItemRef, Substs),
        args: List,
    ) -> Result<Option<Value>> {
        let moved = self.scrutinee_path(callee);
        let box_ty = self.ty_of(callee);
        let Some(&Res::Fn(free, free_substs)) = self.typed.box_frees.get(&expr) else {
            bail!("a `Box<dyn FnOnce>` call with no box to free");
        };
        let Some(boxed) = self.place_or_temp(callee)? else {
            return Ok(None);
        };
        let Some((place, inner_ty)) = self.deref_step((expr, 0), Some(boxed), box_ty)? else {
            return Ok(None);
        };
        let Some(slot) = crate::sema::infer::virtual_slot(self.cx, item, substs)? else {
            bail!("`call_once` has no vtable slot");
        };
        let pair = self.address(place);
        let data = self.ins().load(Type::I64, pair, 0);
        let vtable = self.ins().load(Type::I64, pair, 8);
        let code = self.ins().load(Type::I64, vtable, slot as i32 * 8);
        let Some(&tuple) = self.cx.types.substs(substs).get(1) else {
            bail!("a trait object call with no argument tuple");
        };
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
        let result = self.call_code(
            code,
            (vec![pointer, tuple], Some(ret)),
            std::iter::once(data).chain(packed).collect(),
        )?;
        self.disarm_owned_guards(guards);
        if self.open {
            self.when(need_drop, |lower| lower.drop_object(pair))?;
            let target = self.address(boxed);
            self.invoke(
                SiteOrigin::Table(SiteKey::BoxFree(expr)),
                free,
                free_substs,
                vec![target],
            )?;
        }
        Ok(result)
    }

    fn boxed_closure_call(
        &mut self,
        (expr, callee): (ExprId, ExprId),
        (item, substs): (ItemRef, Substs),
        args: List,
    ) -> Result<Option<Value>> {
        let moved = self.scrutinee_path(callee);
        let boxed = self.ty_of(callee);
        let Some(Res::Fn(free, free_substs)) = crate::sema::infer::drop_fn(self.cx, boxed)? else {
            bail!("`Box` has no `Drop` impl");
        };
        let Some(owner) = self.place_or_temp(callee)? else {
            return Ok(None);
        };
        let Some((place, closure)) = self.deref_step((expr, 0), Some(owner), boxed)? else {
            return Ok(None);
        };
        let env = if self.cx.closure_takes_env(closure) {
            vec![self.address(place)]
        } else {
            Vec::new()
        };
        let values = self.args(args, env)?;
        if !self.mark_boxed_moved(moved, callee)? {
            return Ok(None);
        }
        let result = self.invoke(
            SiteOrigin::Table(SiteKey::ExprRes(expr)),
            item,
            substs,
            values,
        )?;
        let target = self.address(owner);
        self.invoke(
            SiteOrigin::DropFn(TyOrigin::Expr(callee)),
            free,
            free_substs,
            vec![target],
        )?;
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

    pub(in super::super) fn tupled_closure_call(&mut self, item: ItemRef, substs: Substs) -> bool {
        let first = self.cx.types.substs(substs).first().copied();
        first.is_some_and(|ty| matches!(self.cx.types.kind(ty), Kind::Closure(..)))
            && self.cx.call_once() == Some(item)
    }

    pub(in super::super) fn untuple_arg_guarded(
        &mut self,
        tuple_expr: ExprId,
        values: &mut Vec<Value>,
        guards: Vec<usize>,
    ) -> Result<()> {
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
        let Kind::Tuple(items) = *self.cx.types.kind(tuple_ty) else {
            bail!("closure arguments that are not a tuple");
        };
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

use anyhow::{Result, bail};
use haste_codegen::Value;

use crate::body::{ExprId, List};
use crate::names::ItemRef;
use crate::sema::infer::{By, ClosureBody, Shape, needs_drop};
use crate::sema::ty::{Kind, Ty};

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

    /// Lowers a call of the callee value `callee`, a closure or a function item, which inference resolved to `(item, substs)`.
    pub(super) fn call_value(&mut self, (expr, callee): (ExprId, ExprId), (item, substs): (ItemRef, Vec<Ty>), args: List) -> Result<Option<Value>> {
        if self.typed.autoderefs.contains_key(&expr) {
            let values = self.pointed_env(expr, callee)?;
            let values = self.args(args, values)?;
            return self.invoke(item, &substs, values);
        }
        let callee_ty = self.ty_of(callee);
        let mut values = Vec::new();
        if self.cx.closure_takes_env(callee_ty) {
            let place = self.place_or_temp(callee)?;
            values.push(self.address_or_dangling(place, callee_ty)?);
            if self.cx.closure_consumes(callee_ty)
                && let Some((root, path)) = self.scrutinee_path(callee)
            {
                self.moved_out(root, &path)?;
            }
        } else {
            self.expr(callee)?;
        }
        let values = self.args(args, values)?;
        self.invoke(item, &substs, values)
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

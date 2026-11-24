mod call;

use anyhow::{Result, bail};
use haste_codegen::{Coroutine, Type, Value};

use crate::body::{Body, Expr, ExprId, List, Stmt};
use crate::sema::infer::{By, Capture, ClosureBody, Res, Shape, Typed, needs_drop};
use crate::sema::ty::{Kind, Ty};

use super::moves::Root;
use super::place::{Base, Place, transparent_in};
use super::{Lower, repr};

impl Lower<'_, '_> {
    /// Lowers the closure expression `expr` to its environment: each capture's value or address, in a new frame slot.
    pub(super) fn closure(&mut self, expr: ExprId) -> Result<Option<Value>> {
        let ty = self.ty_of(expr);
        let Some(env) = self.cx.closure_env(ty) else {
            bail!("a closure whose captures are not known");
        };
        let Some(closure) = self.typed.closures.get(&expr).cloned() else {
            bail!("a closure expression with no inferred closure");
        };
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

    fn capture_source_place(
        &mut self,
        capture: &Capture,
    ) -> Result<(u32, Vec<(u32, u32)>, Option<Place>, Ty)> {
        let (root_local, rel_path, mut place, mut ty) = if self.locals[capture.local as usize]
            .is_none()
            && let Some((slot_local, place, k)) = (1..=capture.path.len()).rev().find_map(|k| {
                self.capture_by_path(capture.local, &capture.path[..k])
                    .map(|(s, p)| (s, p, k))
            }) {
            (
                slot_local,
                capture.path[k..].to_vec(),
                place,
                self.typed.locals[slot_local as usize].ty,
            )
        } else {
            (
                capture.local,
                capture.path.clone(),
                self.locals[capture.local as usize],
                self.typed.locals[capture.local as usize].ty,
            )
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

    fn projected_capture_local(
        closure: &crate::sema::infer::Closure,
        env_local: u32,
        index: usize,
    ) -> u32 {
        let ordinal = closure.captures[..index]
            .iter()
            .filter(|c| !c.path.is_empty())
            .count() as u32;
        env_local + 1 + u32::from(closure.coroutine) + ordinal
    }

    pub(super) fn capture_local_by_path(&self, local: u32, path: &[(u32, u32)]) -> Option<u32> {
        if path.is_empty() {
            return None;
        }
        let Shape::Closure(ClosureBody {
            env: Some(env_local),
            closure,
            ..
        }) = &self.typed.shape
        else {
            return None;
        };
        let index = closure
            .captures
            .iter()
            .position(|c| c.local == local && c.path == path)?;
        Some(Self::projected_capture_local(closure, *env_local, index))
    }

    pub(super) fn capture_by_path(
        &self,
        local: u32,
        path: &[(u32, u32)],
    ) -> Option<(u32, Option<Place>)> {
        let cap_local = self.capture_local_by_path(local, path)?;
        Some((cap_local, self.locals[cap_local as usize]))
    }

    pub(super) fn expr_capture_path(&self, expr: ExprId) -> Option<(u32, Vec<(u32, u32)>)> {
        expr_capture_path_in(self.body, self.typed, expr)
    }

    /// Points each local that the closure being lowered captures at its place in the environment.
    pub(super) fn bind_captures(&mut self) -> Result<()> {
        let Shape::Closure(ClosureBody {
            env: Some(local),
            closure,
            ..
        }) = self.typed.shape.clone()
        else {
            return Ok(());
        };
        let (whole, env) = self.env_place(local, closure.coroutine)?;
        let mut to_own = Vec::new();
        for (index, capture) in closure.captures.iter().enumerate() {
            let part = self.part_place(whole, env, (0, index as u32))?;
            let cap_ty = capture.ty(self.cx, self.typed.locals[capture.local as usize].ty)?;
            let slot_local = if capture.path.is_empty() {
                capture.local
            } else {
                Self::projected_capture_local(&closure, local, index)
            };
            if closure.consumes && capture.by == By::Value {
                to_own.push((index, slot_local, cap_ty, part));
            }
            self.locals[slot_local as usize] = Some(match capture.by {
                By::Value => part,
                By::Ref | By::Mut => {
                    let Some(address) = self.read(part) else {
                        bail!("a captured reference with no value");
                    };
                    Place {
                        base: Base::Addr(address),
                        offset: 0,
                        repr: repr(self.cx, cap_ty)?,
                    }
                }
            });
        }
        if !to_own.is_empty() {
            let order = capture_drop_order(self.body, self.typed, &closure);
            for idx in order.into_iter().rev() {
                if let Some(&(_, slot_local, cap_ty, part)) =
                    to_own.iter().find(|entry| entry.0 == idx)
                {
                    self.own(Root::Local(slot_local), cap_ty, Some(part), true)?;
                }
            }
        }
        Ok(())
    }

    pub(super) fn drop_closure_parts(
        &mut self,
        ty: Ty,
        target: Value,
        origin: super::reloc::TyOrigin,
    ) -> Result<()> {
        let Some(info) = self.cx.closure(ty) else {
            bail!("drop of a closure whose defining body was not inferred");
        };
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

    fn env_place(&mut self, local: u32, coroutine: bool) -> Result<(Place, Ty)> {
        let local_ty = self.typed.locals[local as usize].ty;
        let Some(held) = self.locals[local as usize] else {
            bail!("a closure environment with no place");
        };
        let (pointer, pointee) = match (coroutine, self.cx.types.kind(local_ty).clone()) {
            (true, Kind::Adt(_, args)) => {
                let first = self.cx.types.substs(args).first().copied();
                let field = self.part_place(held, local_ty, (0, 0))?;
                (self.read(field), first)
            }
            (false, Kind::Ref(_, env)) => (self.read(held), Some(env)),
            _ => bail!("a closure environment that is not a reference"),
        };
        let (Some(pointer), Some(pointee)) = (pointer, pointee) else {
            bail!("a closure environment with no address");
        };
        let pointee = match *self.cx.types.kind(pointee) {
            Kind::Ref(_, inner) => inner,
            _ => pointee,
        };
        if !coroutine {
            return Ok((
                Place {
                    base: Base::Addr(pointer),
                    offset: 0,
                    repr: repr(self.cx, pointee)?,
                },
                pointee,
            ));
        }
        let (inner, offset) = self.captures_of(pointee)?;
        Ok((
            Place {
                base: Base::Addr(pointer),
                offset,
                repr: repr(self.cx, inner)?,
            },
            inner,
        ))
    }
}

fn expr_capture_path_in(
    body: &Body,
    typed: &Typed,
    mut expr: ExprId,
) -> Option<(u32, Vec<(u32, u32)>)> {
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

fn capture_drop_order(
    body: &Body,
    typed: &Typed,
    closure: &crate::sema::infer::Closure,
) -> Vec<usize> {
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
        && let Some(idx) = closure
            .captures
            .iter()
            .position(|cap| cap.local == local && path.starts_with(&cap.path))
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
        Expr::Binary(_, a, b)
        | Expr::Assign(a, b)
        | Expr::AssignOp(_, a, b)
        | Expr::Index(a, b)
        | Expr::Repeat(a, b) => {
            visit_capture_order(body, typed, closure, a, seen, order);
            visit_capture_order(body, typed, closure, b, seen, order);
        }
        Expr::Call(callee, list) => {
            visit_capture_order(body, typed, closure, callee, seen, order);
            visit_list(list, seen, order);
        }
        Expr::Method {
            receiver, params, ..
        } => {
            visit_capture_order(body, typed, closure, receiver, seen, order);
            visit_list(params, seen, order);
        }
        Expr::Block { stmts, tail, .. } => {
            for &at in body.list(stmts) {
                match body.stmts[at as usize] {
                    Stmt::Let {
                        init, otherwise, ..
                    } => {
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
        Expr::While {
            cond, body: inner, ..
        } => {
            visit_capture_order(body, typed, closure, cond, seen, order);
            visit_capture_order(body, typed, closure, inner, seen, order);
        }
        Expr::Loop { body: inner, .. } | Expr::Closure { body: inner, .. } => {
            visit_capture_order(body, typed, closure, inner, seen, order);
        }
        Expr::For {
            iter, body: inner, ..
        } => {
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
                visit_capture_order(
                    body,
                    typed,
                    closure,
                    body.fields[at as usize].value,
                    seen,
                    order,
                );
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

mod call;
mod capture;
mod store;

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, OnceLock};

use anyhow::{Result, bail};

use crate::body::{Expr, ExprId, List, Param, TyId};
use crate::names::ItemRef;

use super::super::Context;
use super::super::ty::{Kind, Ty};
use super::{Infer, Res, Typed};

pub(super) use capture::Captures;
pub use store::{by_reference, closure_typed, coroutine_typed, register, untuple_typed};

/// How a closure holds one variable of the body around it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum By {
    Ref,
    Mut,
    Value,
}

/// One variable or projected field path that a closure captures, as a local of the body that defines the closure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Capture {
    pub local: u32,
    pub by: By,
    pub path: Vec<(u32, u32)>,
}

impl Capture {
    /// The type of this capture given the defining body's local type `ty`.
    pub fn ty(&self, cx: &mut Context<'_>, mut ty: Ty) -> Result<Ty> {
        for &step in &self.path {
            ty = step_field_ty(cx, ty, step)?;
        }
        Ok(ty)
    }

    /// The type of field `(variant, index)` inside `ty`.
    pub fn step_ty(cx: &mut Context<'_>, ty: Ty, step: (u32, u32)) -> Result<Ty> {
        step_field_ty(cx, ty, step)
    }
}

fn step_field_ty(cx: &mut Context<'_>, ty: Ty, (variant, index): (u32, u32)) -> Result<Ty> {
    match *cx.types.kind(ty) {
        Kind::Tuple(items) => cx
            .types
            .substs(items)
            .get(index as usize)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("capture tuple field out of bounds")),
        Kind::Adt(item, args) => {
            let slice = cx.types.substs(args).to_vec();
            super::field_types(cx, item, &slice)?
                .get(variant as usize)
                .and_then(|f| f.get(index as usize))
                .copied()
                .ok_or_else(|| anyhow::anyhow!("capture adt field out of bounds"))
        }
        _ => bail!("capture projection on non-tuple/adt type"),
    }
}

/// A closure expression of a body: its parameter locals, its result, its body expression and what it captures.
#[derive(Clone, Debug)]
pub struct Closure {
    pub params: Vec<u32>,
    pub ret: Ty,
    pub body: ExprId,
    pub captures: Vec<Capture>,
    /// Whether the body moves a captured value out, so a call consumes the closure and its body owns the captures it takes by value.
    pub consumes: bool,
    /// Whether this is an `async` block or body: a coroutine that implements `Future` and holds a state word and a frame before its captures.
    pub coroutine: bool,
}

/// What a closure's own function lowers: the closure expression of `parent`, and the local that holds its environment when it captures anything.
#[derive(Clone, Debug)]
pub struct ClosureBody {
    pub parent: ItemRef,
    pub closure: Closure,
    pub env: Option<u32>,
    pub(in crate::sema) parent_laid: bool,
    pub(in crate::sema) parent_locals: u32,
}

/// A closure type once its defining body is inferred: the body's typed form, the closure, and the tuple its captures form.
#[derive(Debug)]
pub struct ClosureInfo {
    pub parent: ItemRef,
    pub typed: std::sync::Arc<Typed>,
    pub closure: Closure,
    pub env: Ty,
    pub(in crate::sema) parent_drops: Arc<OnceLock<Box<[Res]>>>,
    pub(in crate::sema) parent_laid: Arc<AtomicBool>,
}

impl<'a> Infer<'_, 'a> {
    pub(super) fn closure_expr(
        &mut self,
        expr: ExprId,
        (params, ret, body, by_move): (List, Option<TyId>, ExprId, bool),
        expected: Option<Ty>,
    ) -> Result<Ty> {
        let Some((parent, substs)) = self.instance else {
            bail!("a closure outside a function body")
        };
        let hint = expected
            .and_then(|ty| self.expected_sig(ty))
            .filter(|(inputs, _)| inputs.len() == self.body.list(params).len());
        let depth = self.scopes.len();
        self.uses.push(Captures {
            first: self.out.locals.len() as u32,
            ..Captures::default()
        });
        let ret = self.closure_part(ret, hint.as_ref().map(|(_, output)| *output))?;
        let param_first = self.out.locals.len() as u32;
        let mut locals = Vec::new();
        for (index, &at) in self.body.list(params).iter().enumerate() {
            let Param { pat, ty } = self.body.params[at as usize];
            let ty = self.closure_part(ty, hint.as_ref().map(|(inputs, _)| inputs[index]))?;
            let arena = self.body;
            locals.push(self.param_pat(arena, pat, ty, Some(pat))?);
        }
        let param_locals: Vec<u32> = (param_first..self.out.locals.len() as u32).collect();
        let async_body = matches!(
            *self.body.expr(body),
            Expr::Block {
                kind: crate::body::BlockKind::Async,
                stmts,
                ..
            } if self.body.list(stmts).is_empty()
        );
        let found = self.closure_body(ret, |infer| {
            let found = if async_body {
                let ty = infer.coroutine_expr(body, (None, false), &param_locals)?;
                infer.out.exprs[body.0 as usize] = ty;
                ty
            } else {
                infer.expr(body, Some(ret))?
            };
            infer.coerce_expr(body, found, ret)
        });
        self.scopes.truncate(depth);
        let uses = self.uses.pop().expect("the closure pushed its uses");
        found?;
        self.record_closure(expr, uses, (locals, ret, body, false), (by_move, &[]))?;
        Ok(self.cx.types.intern(Kind::Closure(parent, substs, expr.0)))
    }

    pub(super) fn coroutine_expr(
        &mut self,
        expr: ExprId,
        (output, by_move): (Option<Ty>, bool),
        forced: &[u32],
    ) -> Result<Ty> {
        self.poly_fallback();
        let Some((parent, substs)) = self.instance else {
            bail!("an `async` block outside a function body")
        };
        let Expr::Block { stmts, tail, .. } = *self.body.expr(expr) else {
            bail!("an `async` body that is not a block")
        };
        let depth = self.scopes.len();
        self.uses.push(Captures {
            first: self.out.locals.len() as u32,
            ..Captures::default()
        });
        let ret = output.unwrap_or_else(|| self.fresh(Kind::Var));
        let found = self.closure_body(ret, |infer| {
            infer
                .block(stmts, tail, Some(ret))
                .and_then(|found| infer.coerce(found, ret))
        });
        self.scopes.truncate(depth);
        let uses = self.uses.pop().expect("the coroutine pushed its uses");
        found?;
        self.record_closure(expr, uses, (Vec::new(), ret, expr, true), (by_move, forced))?;
        Ok(self.cx.types.intern(Kind::Closure(parent, substs, expr.0)))
    }

    fn closure_part(&mut self, annotated: Option<TyId>, hinted: Option<Ty>) -> Result<Ty> {
        let Some(annotated) = annotated else {
            return Ok(hinted.unwrap_or_else(|| self.fresh(Kind::Var)));
        };
        let ty = self.lower_ty(self.body, annotated)?;
        if let Some(hinted) = hinted {
            self.unifies(ty, hinted);
        }
        Ok(ty)
    }

    fn record_closure(
        &mut self,
        expr: ExprId,
        uses: Captures,
        (params, ret, body, coroutine): (Vec<u32>, Ty, ExprId, bool),
        (by_move, forced): (bool, &[u32]),
    ) -> Result<()> {
        if let Some(around) = self.uses.last_mut() {
            around.inner_closures.push(expr);
        }
        let (captures, consumes, known) = self.decide_captures(&uses, (by_move, forced))?;
        if !known {
            self.unsure.push((expr, uses, by_move, forced.to_vec()));
        }
        self.out.closures.insert(
            expr,
            Closure {
                params,
                ret,
                body,
                captures,
                consumes: coroutine || consumes,
                coroutine,
            },
        );
        Ok(())
    }

    fn closure_body(&mut self, ret: Ty, run: impl FnOnce(&mut Self) -> Result<()>) -> Result<()> {
        let outer = (
            self.out.ret.replace(ret),
            std::mem::take(&mut self.loops),
            std::mem::take(&mut self.try_blocks),
            self.diverges,
        );
        let found = run(self);
        (self.out.ret, self.loops, self.try_blocks, self.diverges) = outer;
        found
    }

    pub(super) fn coroutine_output(&self, ty: Ty) -> Option<Ty> {
        let Kind::Closure(parent, substs, at) = self.cx.types.kind(self.resolve(ty)) else {
            return None;
        };
        let here = self
            .instance
            .as_ref()
            .is_some_and(|(item, args)| item == parent && args == substs);
        let local = here
            .then(|| self.out.closures.get(&ExprId(*at)))
            .flatten()
            .filter(|c| c.coroutine)
            .map(|c| c.ret);
        local
            .or_else(|| {
                self.cx
                    .closure(self.resolve(ty))
                    .filter(|i| i.closure.coroutine)
                    .map(|i| i.closure.ret)
            })
            .or_else(|| self.cx.future_output(self.resolve(ty)))
    }

    pub(super) fn await_expr(&mut self, expr: ExprId, inner: ExprId) -> Result<Ty> {
        self.poly_fallback();
        let found = self.expr(inner, None)?;
        if matches!(self.cx.types.kind(self.resolve(found)), Kind::Var(_)) {
            self.settle_opaques(false)?;
        }
        if let Some(output) = self.opaque_output(self.resolve(found)) {
            self.awaits.push((expr, found));
            return Ok(output);
        }
        self.awaited(expr, found)
    }

    pub(super) fn settle_awaits(&mut self) -> Result<()> {
        for (expr, found) in std::mem::take(&mut self.awaits) {
            let output = self.awaited(expr, found)?;
            self.unify(output, self.out.exprs[expr.0 as usize])?;
        }
        Ok(())
    }

    fn awaited(&mut self, expr: ExprId, found: Ty) -> Result<Ty> {
        let (into_future, future) = self.lang_call("into_future", found)?;
        let Some(future_trait) = self.cx.lang_item("future_trait") else {
            bail!("core has no `Future`")
        };
        let output = self.project(future_trait, future, Vec::new(), "Output")?;
        let Some(poll) = self.cx.lang_item("poll") else {
            bail!("core has no `Future::poll`")
        };
        let (item, substs) = self.future_poll(future_trait, poll, future)?;
        let polled = self.fn_types(item, &substs)?.1;
        self.out.awaits.insert(
            expr,
            super::Awaited {
                into_future,
                future,
                poll: Res::fn_of(&self.cx.types, item, &substs),
                polled,
            },
        );
        Ok(output)
    }

    fn future_poll(
        &mut self,
        of: ItemRef,
        poll: ItemRef,
        future: Ty,
    ) -> Result<(ItemRef, Vec<Ty>)> {
        let future = self.resolve(future);
        if self.compiler_impl(of, future) || self.object_holds(of, future, &[]) == Some(true) {
            return Ok((poll, vec![future]));
        }
        let Some((block, args)) = self.select(of, future, &[])? else {
            bail!("{} is not a future", self.show(future))
        };
        let Some(member) = self.cx.member(block, "poll") else {
            bail!("the `Future` impl of {} has no `poll`", self.show(future))
        };
        Ok((member, args.into_iter().map(|arg| self.zonk(arg)).collect()))
    }
}

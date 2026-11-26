use std::sync::Arc;

use anyhow::{Result, bail};

use crate::body::{BlockKind, Expr, ExprId, List};
use crate::names::ItemRef;
use crate::outline::item::ItemKind;

use super::super::super::ty::{Kind, Ty};
use super::super::{Infer, Res};
use super::{By, ClosureInfo};

impl<'a> Infer<'_, 'a> {
    fn ensure_closure(&mut self, ty: Ty) -> Result<Option<(u32, Option<Arc<ClosureInfo>>)>> {
        let Kind::Closure(parent, substs, at) = *self.cx.types.kind(ty) else {
            return Ok(None);
        };
        let local = self.instance == Some((parent, substs));
        if !local && !self.cx.has_closure(ty) {
            self.cx.ensure_closure(ty);
        }
        Ok(Some((at, (!local).then(|| self.cx.closure(ty)).flatten())))
    }

    pub(in super::super) fn closure_sig(&mut self, ty: Ty) -> Result<(Vec<Ty>, Ty, bool)> {
        let Some((at, info)) = self.ensure_closure(ty)? else {
            bail!("not a closure")
        };
        let (closure, locals) = match &info {
            Some(i) => (Some(&i.closure), &i.typed.locals),
            None => (self.out.closures.get(&ExprId(at)), &self.out.locals),
        };
        let Some(closure) = closure else {
            bail!("a closure whose body is not inferred yet")
        };
        Ok((
            closure
                .params
                .iter()
                .map(|&local| locals[local as usize].ty)
                .collect(),
            closure.ret,
            !closure.captures.is_empty(),
        ))
    }

    pub(in super::super) fn closure_mutates(&mut self, ty: Ty) -> Result<bool> {
        let Some((at, info)) = self.ensure_closure(ty)? else {
            return Ok(false);
        };
        let closure = info
            .as_ref()
            .map(|info| &info.closure)
            .or_else(|| self.out.closures.get(&ExprId(at)));
        Ok(closure
            .is_some_and(|closure| closure.captures.iter().any(|capture| capture.by == By::Mut)))
    }

    pub(in super::super) fn closure_call(
        &mut self,
        expr: ExprId,
        ty: Ty,
        args: List,
    ) -> Result<Ty> {
        let (params, ret, _) = self.closure_sig(ty)?;
        self.call_once_of(expr, ty, (params, ret), args)
    }

    pub(in super::super) fn call_once_of(
        &mut self,
        expr: ExprId,
        ty: Ty,
        (params, ret): (Vec<Ty>, Ty),
        args: List,
    ) -> Result<Ty> {
        self.infer_call_args(args, &params, None)?;
        let Some(call_once) = self.cx.call_once() else {
            bail!("core has no `FnOnce::call_once`")
        };
        let tuple = self.cx.types.tuple(&params);
        self.out
            .res
            .insert(expr, Res::fn_of(&self.cx.types, call_once, &[ty, tuple]));
        Ok(ret)
    }

    pub(super) fn expected_sig(&mut self, expected: Ty) -> Option<(Vec<Ty>, Ty)> {
        let expected = self.resolve(expected);
        let own_ret = self
            .out
            .ret
            .is_some_and(|ret| self.resolve(ret) == expected);
        let expected = self.ret_sig.filter(|_| own_ret).unwrap_or(expected);
        match *self.cx.types.kind(expected) {
            Kind::FnPtr(inputs, output) => Some((self.cx.types.substs(inputs).to_vec(), output)),
            Kind::Dyn(of, parts, members, _) if self.is_callable_trait(of) => {
                let (first, output) = {
                    let (given, mut bound) = super::super::super::ty::object_parts(
                        self.cx.types.substs(parts),
                        self.cx.types.items(members),
                    );
                    (*given.first()?, bound.next().map(|(_, ty)| ty))
                };
                let Kind::Tuple(inputs) = *self.cx.types.kind(self.resolve(first)) else {
                    return None;
                };
                let output = if self.is_async_callable_trait(of) {
                    self.fresh(Kind::Var)
                } else {
                    output.unwrap_or_else(|| self.cx.types.unit())
                };
                Some((self.cx.types.substs(inputs).to_vec(), output))
            }
            _ => None,
        }
    }

    pub(in super::super) fn call_arg(
        &mut self,
        callee: Option<(ItemRef, &[Ty])>,
        arg: ExprId,
        (param, expectation): (Ty, Option<Ty>),
        held: &mut Option<(ItemRef, Vec<Ty>)>,
    ) -> Result<()> {
        let callable = match *self.cx.types.kind(self.resolve(param)) {
            Kind::Ref(_, pointee) => pointee,
            _ => param,
        };
        let (hint, bound) = match callee {
            Some(c) => (
                self.closure_hint(c, arg, param)?,
                self.callable_hint(c, callable)?,
            ),
            None => (None, None),
        };
        let closure = self.is_closure(arg);
        let propagate = closure
            || self.holds_closure(arg)
            || matches!(self.body.expr(self.wrapped(arg)), Expr::Array(_));
        let usable = expectation.filter(|&wanted| propagate || self.sized_expectation(wanted));
        let found = self.expr(arg, Some(hint.or(usable).unwrap_or(param)))?;
        self.release_if_fits(held, found, param)?;
        self.coerce_arg((arg, found, closure), param, expectation)?;
        match bound {
            Some((of, bound)) => self.tie_callable_bound(of, callable, bound),
            None => Ok(()),
        }
    }

    fn sized_expectation(&mut self, ty: Ty) -> bool {
        let before = self.cx.quiet_mark();
        let filled = self.zonk(ty);
        matches!(self.cx.is_unsized(filled), Ok(false)) && self.cx.quiet_mark() == before
    }

    fn holds_closure(&self, expr: ExprId) -> bool {
        let any = |list: List| {
            self.body
                .list(list)
                .iter()
                .any(|&at| self.holds_closure(ExprId(at)))
        };
        match *self.body.expr(self.wrapped(expr)) {
            Expr::Closure { .. } => true,
            Expr::Tuple(items) | Expr::Array(items) => any(items),
            Expr::Ref { expr: inner, .. } => self.holds_closure(inner),
            Expr::If(_, then_branch, else_branch) => {
                self.holds_closure(then_branch)
                    || else_branch.is_some_and(|at| self.holds_closure(at))
            }
            Expr::Match(_, arms) => self
                .body
                .list(arms)
                .iter()
                .any(|&at| self.holds_closure(self.body.arms[at as usize].body)),
            Expr::Struct { fields, base, .. } => {
                self.body
                    .list(fields)
                    .iter()
                    .any(|&at| self.holds_closure(self.body.fields[at as usize].value))
                    || base.is_some_and(|at| self.holds_closure(at))
            }
            _ => false,
        }
    }

    fn coerce_arg(
        &mut self,
        (arg, found, closure): (ExprId, Ty, bool),
        param: Ty,
        expectation: Option<Ty>,
    ) -> Result<()> {
        let reified = expectation.is_some_and(|wanted| {
            matches!(self.cx.types.kind(self.resolve(wanted)), Kind::FnPtr(..))
        });
        match expectation.filter(|_| !closure || reified) {
            Some(wanted) if self.coerces_to(found, wanted) => {
                self.coerce_expr(arg, found, wanted)?;
                self.unify(param, wanted)
            }
            _ => self.coerce_expr(arg, found, param),
        }
    }

    pub(in super::super) fn coerces_to(&mut self, found: Ty, wanted: Ty) -> bool {
        let shared = self.weakened(found, wanted);
        self.probe(|infer| {
            matches!(infer.coercion(found, wanted), Ok(Some(_)))
                || shared.is_some_and(|(a, b)| infer.unifies(a, b))
        })
    }

    pub(in super::super) fn callable_sig(&mut self, ty: Ty) -> Result<Option<(Vec<Ty>, Ty)>> {
        let ty = self.resolve(ty);
        Ok(Some(match *self.cx.types.kind(ty) {
            Kind::Ref(_, inner) => return self.callable_sig(inner),
            Kind::Closure(..) => {
                let (params, ret, _) = self.closure_sig(ty)?;
                (params, ret)
            }
            Kind::FnDef(item, substs) => {
                let args = self.cx.types.substs(substs).to_vec();
                self.fn_types(item, &args)?
            }
            Kind::FnPtr(params, ret) => (self.cx.types.substs(params).to_vec(), ret),
            Kind::Param(_) | Kind::Proj(_) => {
                let Some(sig) = self.poly_fn_sig(ty) else {
                    return Ok(None);
                };
                sig
            }
            Kind::Var(_) => match self
                .opaque_sig(ty)
                .map(|sig| self.cx.types.kind(sig).clone())
            {
                Some(Kind::FnPtr(params, ret)) => (self.cx.types.substs(params).to_vec(), ret),
                _ => return Ok(None),
            },
            _ => return Ok(None),
        }))
    }

    pub(in super::super) fn is_closure(&self, arg: ExprId) -> bool {
        matches!(*self.body.expr(self.wrapped(arg)), Expr::Closure { .. })
    }

    fn wrapped(&self, expr: ExprId) -> ExprId {
        match *self.body.expr(expr) {
            Expr::Paren(inner) => self.wrapped(inner),
            Expr::Block {
                tail: Some(tail),
                label: None,
                kind: BlockKind::Plain | BlockKind::Expansion | BlockKind::Unsafe,
                ..
            } => self.wrapped(tail),
            _ => expr,
        }
    }

    fn closure_hint(
        &mut self,
        callee: (ItemRef, &[Ty]),
        arg: ExprId,
        param: Ty,
    ) -> Result<Option<Ty>> {
        match *self.body.expr(self.wrapped(arg)) {
            Expr::Closure { .. } => Ok(self
                .callable_hint(callee, param)?
                .map(|(of, sig)| self.closure_hint_sig(of, sig))),
            Expr::Ref {
                raw: false,
                expr: inner,
                ..
            } => {
                let Kind::Ref(mutability, pointee) = *self.cx.types.kind(self.resolve(param))
                else {
                    return Ok(None);
                };
                Ok(self
                    .closure_hint(callee, inner, pointee)?
                    .map(|hint| self.cx.types.intern(Kind::Ref(mutability, hint))))
            }
            _ => Ok(None),
        }
    }

    fn callable_hint(
        &mut self,
        (item, substs): (ItemRef, &[Ty]),
        param: Ty,
    ) -> Result<Option<(ItemRef, Ty)>> {
        let param = self.resolve(param);
        if !matches!(self.cx.types.kind(param), Kind::Var(_)) {
            return Ok(None);
        }
        if matches!(
            self.cx.item(item).1.kind,
            ItemKind::Struct | ItemKind::Enum | ItemKind::Union
        ) {
            return self.field_hint((item, substs), param);
        }
        let frame = self.frame(item, substs)?;
        let Some(at) = frame
            .generics
            .iter()
            .position(|(_, ty)| self.resolve(*ty) == param)
        else {
            return Ok(None);
        };
        let sig = self.cx.signature(item)?;
        let own: Vec<_> = super::super::scope::type_params(&sig.generics).collect();
        let (home, name) = (frame.home, frame.generics[at].0);
        let Some(index) = at.checked_sub(frame.generics.len() - own.len()) else {
            return self.with_frame(frame, |infer| infer.header_callable_bound(name, home));
        };
        let bounds = super::super::scope::generic_bounds(
            &sig.body,
            &self.cx.file(item).source,
            &sig.predicates,
            own[index],
        );
        self.first_callable_bound(frame, &sig.body, bounds)
    }

    fn field_hint(
        &mut self,
        (item, args): (ItemRef, &[Ty]),
        param: Ty,
    ) -> Result<Option<(ItemRef, Ty)>> {
        let def = self.cx.adt_def(item)?;
        let Some(at) = args.iter().position(|&arg| self.resolve(arg) == param) else {
            return Ok(None);
        };
        let Some(generic) = super::super::scope::type_params(&def.generics.params).nth(at) else {
            return Ok(None);
        };
        let bounds = super::super::scope::generic_bounds(
            &def.body,
            &self.cx.file(item).source,
            &def.generics.predicates,
            generic,
        );
        if bounds.is_empty() {
            return Ok(None);
        }
        let frame = self.adt_frame(item, &def.generics.params, args)?;
        self.first_callable_bound(frame, &def.body, bounds)
    }
}

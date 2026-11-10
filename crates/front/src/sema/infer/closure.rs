use std::rc::Rc;

use anyhow::{Result, bail};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::body::sig::Signature;
use crate::body::{Args, Body, BlockKind, Expr, ExprId, List, Param, PathId, TyId, UnOp};
use crate::names::{ItemRef, ModRef};

use super::super::Context;
use super::super::ty::{Kind, Mut, Ty};
use super::{Infer, Local, Res, Shape, Typed};

/// How a closure holds one variable of the body around it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum By {
    Ref,
    Mut,
    Value,
}

/// One variable that a closure captures, as a local of the body that defines the closure.
#[derive(Clone, Copy, Debug)]
pub struct Capture {
    pub local: u32,
    pub by: By,
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
}

/// A closure type once its defining body is inferred: the body's typed form, the closure, and the tuple its captures form.
#[derive(Debug)]
pub struct ClosureInfo {
    pub parent: ItemRef,
    pub typed: Rc<Typed>,
    pub closure: Closure,
    pub env: Ty,
}

/// The variables that one closure being inferred uses from outside it: locals below `first` belong to the bodies around it.
#[derive(Default)]
pub(super) struct Uses {
    first: u32,
    used: Vec<u32>,
    mutated: Vec<u32>,
    reads: Vec<(ExprId, u32)>,
}

impl Infer<'_, '_> {
    pub(super) fn closure_expr(&mut self, expr: ExprId, (params, ret, body, by_move): (List, Option<TyId>, ExprId, bool), expected: Option<Ty>) -> Result<Ty> {
        let Some((parent, substs)) = self.instance.clone() else { bail!("a closure outside a function body") };
        let hint = expected.and_then(|ty| self.expected_sig(ty)).filter(|(inputs, _)| inputs.len() == self.body.list(params).len());
        let depth = self.scopes.len();
        self.uses.push(Uses { first: self.out.locals.len() as u32, ..Uses::default() });
        let mut locals = Vec::new();
        for (index, &at) in self.body.list(params).iter().enumerate() {
            let Param { pat, ty } = self.body.params[at as usize];
            let ty = match (ty, &hint) {
                (Some(ty), _) => self.lower_ty(self.body, ty)?,
                (None, Some((inputs, _))) => inputs[index],
                (None, None) => self.fresh(Kind::Var),
            };
            let arena = self.body;
            locals.push(self.param_pat(arena, pat, ty, Some(pat))?);
        }
        let ret = match (ret, hint) {
            (Some(ty), _) => self.lower_ty(self.body, ty)?,
            (None, Some((_, output))) => output,
            (None, None) => self.fresh(Kind::Var),
        };
        let found = self.closure_body(ret, |infer| infer.expr(body, Some(ret)).and_then(|found| infer.coerce_expr(body, found, ret)));
        self.scopes.truncate(depth);
        let uses = self.uses.pop().expect("the closure pushed its uses");
        found?;
        let moved = self.moved_reads(&uses.reads)?;
        let captures = self.captures(uses, (by_move, &moved), &[]);
        self.out.closures.insert(expr, Closure { params: locals, ret, body, captures, consumes: !moved.is_empty(), coroutine: false });
        Ok(self.cx.types.intern(Kind::Closure(parent, substs, expr.0)))
    }

    /// Infers the `async` block or body `expr` as a coroutine whose output is `output` when known, capturing by value when `by_move` and always capturing `forced`.
    pub(super) fn coroutine_expr(&mut self, expr: ExprId, (output, by_move): (Option<Ty>, bool), forced: &[u32]) -> Result<Ty> {
        let Some((parent, substs)) = self.instance.clone() else { bail!("an `async` block outside a function body") };
        let Expr::Block { stmts, tail, .. } = *self.body.expr(expr) else { bail!("an `async` body that is not a block") };
        let depth = self.scopes.len();
        self.uses.push(Uses { first: self.out.locals.len() as u32, ..Uses::default() });
        let ret = output.unwrap_or_else(|| self.fresh(Kind::Var));
        let found = self.closure_body(ret, |infer| {
            let found = infer.block(stmts, tail, Some(ret))?;
            infer.coerce(found, ret)
        });
        self.scopes.truncate(depth);
        let uses = self.uses.pop().expect("the coroutine pushed its uses");
        found?;
        let moved = self.moved_reads(&uses.reads)?;
        let captures = self.captures(uses, (by_move, &moved), forced);
        self.out.closures.insert(expr, Closure { params: Vec::new(), ret, body: expr, captures, consumes: true, coroutine: true });
        Ok(self.cx.types.intern(Kind::Closure(parent, substs, expr.0)))
    }

    fn closure_body(&mut self, ret: Ty, run: impl FnOnce(&mut Self) -> Result<()>) -> Result<()> {
        let outer = (self.out.ret.replace(ret), std::mem::take(&mut self.loops), std::mem::take(&mut self.try_blocks), self.diverges);
        let found = run(self);
        (self.out.ret, self.loops, self.try_blocks, self.diverges) = outer;
        found
    }

    /// The output of `ty` when it is a coroutine: one this body defines, one whose defining body is inferred, or an async fn future named while its body is inferred.
    pub(super) fn coroutine_output(&self, ty: Ty) -> Option<Ty> {
        let Kind::Closure(parent, substs, at) = self.cx.types.kind(self.resolve(ty)) else { return None };
        let here = self.instance.as_ref().is_some_and(|(item, args)| item == parent && args == substs);
        let local = here.then(|| self.out.closures.get(&ExprId(*at))).flatten().filter(|closure| closure.coroutine).map(|closure| closure.ret);
        local.or_else(|| self.cx.closure(self.resolve(ty)).filter(|info| info.closure.coroutine).map(|info| info.closure.ret)).or_else(|| self.cx.future_output(self.resolve(ty)))
    }

    /// Infers `inner.await`: the future that `IntoFuture::into_future` makes of `inner`, polled until it gives its output.
    pub(super) fn await_expr(&mut self, expr: ExprId, inner: ExprId) -> Result<Ty> {
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

    /// Finishes each `.await` on an async fn future that was pending, once the futures are revealed.
    pub(super) fn settle_awaits(&mut self) -> Result<()> {
        for (expr, found) in std::mem::take(&mut self.awaits) {
            let output = self.awaited(expr, found)?;
            let declared = self.out.exprs[expr.0 as usize];
            self.unify(output, declared)?;
        }
        Ok(())
    }

    /// Records how `.await` polls the value of type `found`, and returns its output.
    fn awaited(&mut self, expr: ExprId, found: Ty) -> Result<Ty> {
        let (into_future, future) = self.lang_call("into_future", found)?;
        let Some(future_trait) = self.cx.lang_item("future_trait") else { bail!("core has no `Future`") };
        let output = self.project(future_trait, future, Vec::new(), "Output")?;
        let Some(poll) = self.cx.lang_item("poll") else { bail!("core has no `Future::poll`") };
        let (item, substs) = self.future_poll(future_trait, poll, future)?;
        let polled = self.fn_types(item, &substs)?.1;
        self.out.awaits.insert(expr, super::Awaited { into_future, future, poll: Res::Fn(item, substs), polled });
        Ok(output)
    }

    /// The `Future::poll` that polls a future of type `future`: the compiler's for a coroutine or a trait object, else the one of the impl that applies.
    fn future_poll(&mut self, of: ItemRef, poll: ItemRef, future: Ty) -> Result<(ItemRef, Vec<Ty>)> {
        let future = self.resolve(future);
        if self.compiler_impl(of, future) || self.object_holds(of, future, &[]) == Some(true) {
            return Ok((poll, vec![future]));
        }
        let Some((block, args)) = self.select(of, future, &[])? else { bail!("{} is not a future", self.show(future)) };
        let Some(member) = self.cx.member(block, "poll") else { bail!("the `Future` impl of {} has no `poll`", self.show(future)) };
        Ok((member, args.into_iter().map(|arg| self.zonk(arg)).collect()))
    }

    fn captures(&mut self, uses: Uses, (by_move, moved): (bool, &[u32]), forced: &[u32]) -> Vec<Capture> {
        let mut captures: Vec<Capture> = forced.iter().map(|&local| Capture { local, by: By::Value }).collect();
        for local in uses.used {
            if captures.iter().any(|capture| capture.local == local) {
                continue;
            }
            let by = match (by_move || moved.contains(&local), uses.mutated.contains(&local)) {
                (true, _) => By::Value,
                (false, true) => By::Mut,
                (false, false) => By::Ref,
            };
            captures.push(Capture { local, by });
        }
        if let Some(around) = self.uses.last_mut() {
            for capture in captures.iter().filter(|capture| capture.local < around.first) {
                around.used.push(capture.local);
                if capture.by != By::Ref {
                    around.mutated.push(capture.local);
                }
            }
        }
        captures
    }

    /// Notes that the path expression `expr` names local `local`, which a closure being inferred may move out of.
    pub(super) fn note_read(&mut self, expr: ExprId, local: u32) {
        for uses in self.uses.iter_mut().filter(|uses| local < uses.first) {
            uses.reads.push((expr, local));
        }
    }

    /// The captured locals that the reads `reads` move out of: a read, or a field path from it, used as a value whose type needs dropping.
    fn moved_reads(&mut self, reads: &[(ExprId, u32)]) -> Result<Vec<u32>> {
        if reads.is_empty() {
            return Ok(Vec::new());
        }
        let (places, parents) = self.place_uses();
        let mut moved = Vec::new();
        for &(read, local) in reads {
            let mut at = read;
            while let Some(&parent) = parents.get(&at) {
                at = parent;
            }
            let ty = self.resolve(self.out.exprs[at.0 as usize]);
            if !places.contains(&at) && !moved.contains(&local) && self.needs_drop(ty)? {
                moved.push(local);
            }
        }
        Ok(moved)
    }

    /// The expressions of the body used as places rather than values, and the field or parenthesis expression that projects each projected expression.
    fn place_uses(&self) -> (HashSet<ExprId>, HashMap<ExprId, ExprId>) {
        let (mut places, mut parents) = (HashSet::default(), HashMap::default());
        for (at, expr) in self.body.exprs.iter().enumerate() {
            let at = ExprId(at as u32);
            match *expr {
                Expr::Field(base, _) | Expr::TupleField(base, _) | Expr::Paren(base) => {
                    parents.insert(base, at);
                }
                Expr::Ref { expr, .. } | Expr::Unary(UnOp::Deref, expr) | Expr::Index(expr, _) | Expr::Assign(expr, _) | Expr::AssignOp(_, expr, _) | Expr::Match(expr, _) | Expr::Let(_, expr) | Expr::Call(expr, _) => {
                    places.insert(expr);
                }
                Expr::Method { receiver, .. } if self.out.adjusts.get(&at).is_some_and(|adjust| adjust.autoref.is_some() || adjust.derefs > 0) => {
                    places.insert(receiver);
                }
                _ => {}
            }
        }
        (places, parents)
    }

    /// Notes that the body reads local `local`, which a closure being inferred captures when it lies outside it.
    pub(super) fn note_use(&mut self, local: u32) {
        for uses in self.uses.iter_mut().filter(|uses| local < uses.first) {
            uses.used.push(local);
        }
    }

    /// Notes that the place expression `expr` is written or borrowed mutably: its overloaded dereferences go through `DerefMut`, and a closure captures its local by unique borrow.
    pub(super) fn note_mut(&mut self, expr: ExprId) -> Result<()> {
        self.deref_mutably(expr)?;
        let Some(local) = self.place_local(expr) else { return Ok(()) };
        for uses in self.uses.iter_mut().filter(|uses| local < uses.first) {
            uses.mutated.push(local);
        }
        Ok(())
    }

    fn place_local(&self, expr: ExprId) -> Option<u32> {
        match *self.body.expr(expr) {
            Expr::Paren(inner) | Expr::Field(inner, _) | Expr::TupleField(inner, _) | Expr::Index(inner, _) | Expr::Unary(UnOp::Deref, inner) => self.place_local(inner),
            Expr::Path(_) => match self.out.res.get(&expr) {
                Some(Res::Local(local)) => Some(*local),
                _ => None,
            },
            _ => None,
        }
    }

    /// The parameter and result types of the closure type `ty`, from this body when it defines the closure, and whether it captures anything.
    pub(super) fn closure_sig(&mut self, ty: Ty) -> Result<(Vec<Ty>, Ty, bool)> {
        let Kind::Closure(parent, substs, at) = self.cx.types.kind(ty).clone() else { bail!("not a closure") };
        let local = self.instance.as_ref().is_some_and(|(item, args)| *item == parent && *args == substs);
        if !local && self.cx.closure(ty).is_none() {
            super::opaque::revealed(self.cx, parent, &substs)?;
        }
        let info = if local { None } else { self.cx.closure(ty) };
        let (closure, locals) = match &info {
            Some(info) => (Some(&info.closure), &info.typed.locals),
            None => (self.out.closures.get(&ExprId(at)), &self.out.locals),
        };
        let Some(closure) = closure else { bail!("a closure whose body is not inferred yet") };
        Ok((closure.params.iter().map(|&local| locals[local as usize].ty).collect(), closure.ret, !closure.captures.is_empty()))
    }

    /// Infers the call `expr` of the closure value of type `ty`, recording it as a call of `FnOnce::call_once` at the closure and its argument tuple.
    pub(super) fn closure_call(&mut self, expr: ExprId, ty: Ty, args: List) -> Result<Ty> {
        let (params, ret, _) = self.closure_sig(ty)?;
        let args = self.body.list(args).to_vec();
        if args.len() != params.len() {
            bail!("closure takes {} arguments, {} given", params.len(), args.len());
        }
        for (&arg, &param) in args.iter().zip(&params) {
            self.call_arg(None, ExprId(arg), (param, None))?;
        }
        let Some(call_once) = self.cx.call_once() else { bail!("core has no `FnOnce::call_once`") };
        let tuple = self.cx.types.intern(Kind::Tuple(params));
        self.out.res.insert(expr, Res::Fn(call_once, vec![ty, tuple]));
        Ok(ret)
    }

    fn expected_sig(&mut self, expected: Ty) -> Option<(Vec<Ty>, Ty)> {
        let expected = self.resolve(expected);
        let own_ret = self.out.ret.is_some_and(|ret| self.resolve(ret) == expected);
        let expected = self.ret_sig.filter(|_| own_ret).unwrap_or(expected);
        match self.cx.types.kind(expected).clone() {
            Kind::FnPtr(inputs, output) => Some((inputs, output)),
            Kind::Dyn(of, parts, members, _) if self.is_callable_trait(of) => {
                let (given, mut bound) = super::super::ty::object_parts(&parts, &members);
                let Kind::Tuple(inputs) = self.cx.types.kind(self.resolve(*given.first()?)).clone() else { return None };
                let output = bound.next().map_or_else(|| self.cx.types.unit(), |(_, ty)| ty);
                Some((inputs, output))
            }
            _ => None,
        }
    }

    /// Infers the call `expr` of the `dyn Fn` or `dyn FnMut` value of type `ty`, recording it as a call of `call` or `call_mut` on the trait object.
    pub(super) fn object_call(&mut self, (expr, found): (ExprId, Ty), ty: Ty, args: List) -> Result<Ty> {
        let Kind::Dyn(of, parts, members, _) = self.cx.types.kind(ty).clone() else { bail!("an object call of a type that is not a trait object") };
        let (given, mut bound) = super::super::ty::object_parts(&parts, &members);
        let Some(&tuple) = given.first() else { bail!("a callable trait object with no argument tuple") };
        let ret = bound.next().map_or_else(|| self.cx.types.unit(), |(_, ret)| ret);
        let Kind::Tuple(params) = self.cx.types.kind(self.resolve(tuple)).clone() else { bail!("a callable trait object whose arguments are not a tuple") };
        let args = self.body.list(args).to_vec();
        if args.len() != params.len() {
            bail!("call takes {} arguments, {} given", params.len(), args.len());
        }
        for (&arg, &param) in args.iter().zip(&params) {
            self.call_arg(None, ExprId(arg), (param, None))?;
        }
        let name = match self.cx.lang(of) {
            Some("fn") => "call",
            Some("fn_mut") => "call_mut",
            _ => {
                self.free_box(expr, found)?;
                "call_once"
            }
        };
        let Some(method) = self.cx.trait_member(of, name) else { bail!("`{}` has no `{name}`", self.cx.name(of)) };
        self.out.res.insert(expr, Res::Fn(method, vec![ty, tuple]));
        Ok(ret)
    }

    /// Infers the call `expr` of a value of type `found`, which implements `Fn` or `FnMut` itself at some step of its autoderef, recording it as a call of `call` or `call_mut` at that step's type and the argument tuple.
    pub(super) fn overloaded_call(&mut self, expr: ExprId, found: Ty, args: List) -> Result<Ty> {
        let args = self.body.list(args).to_vec();
        let params: Vec<Ty> = args.iter().map(|_| self.fresh(Kind::Var)).collect();
        let tuple = self.cx.types.intern(Kind::Tuple(params.clone()));
        let (mut ty, mut steps) = (self.resolve(found), 0);
        loop {
            if let Some(method) = self.callable_impl(ty, tuple)? {
                self.out.autoderefs.remove(&expr);
                if steps > 0 {
                    self.out.autoderefs.insert(expr, steps);
                }
                for (&arg, &param) in args.iter().zip(&params) {
                    self.call_arg(None, ExprId(arg), (param, None))?;
                }
                self.out.res.insert(expr, Res::Fn(method, vec![ty, tuple]));
                self.out.overloaded.insert(expr);
                let Some(once) = self.cx.lang_item("fn_once") else { bail!("core has no `FnOnce`") };
                return self.project(once, ty, vec![tuple], "Output");
            }
            let Some(inner) = self.builtin_deref(ty) else { bail!("calling {} is not supported yet", self.show(found)) };
            (ty, steps) = (self.resolve(inner), steps + 1);
        }
    }

    /// `Fn::call`, `FnMut::call_mut` or `FnOnce::call_once`, the first that an impl gives `ty` at the argument tuple `tuple`.
    fn callable_impl(&mut self, ty: Ty, tuple: Ty) -> Result<Option<ItemRef>> {
        for (lang, name) in [("fn", "call"), ("fn_mut", "call_mut"), ("fn_once", "call_once")] {
            let Some(of) = self.cx.lang_item(lang) else { continue };
            if self.trial(|infer| Ok(infer.select(of, ty, &[tuple])?.map(|_| ())))?.is_some() {
                return Ok(self.cx.trait_member(of, name));
            }
        }
        Ok(None)
    }

    fn free_box(&mut self, expr: ExprId, found: Ty) -> Result<()> {
        let found = self.resolve(found);
        if self.cx.boxed(found).is_none() || self.out.autoderefs.get(&expr) != Some(&1) {
            bail!("a `dyn FnOnce` is called by value only through a `Box`");
        }
        let Some((block, args)) = self.drop_impl(found)? else { bail!("`Box` has no `Drop` impl") };
        let Some(member) = self.cx.member(block, "drop") else { bail!("the `Drop` impl of `Box` has no `drop`") };
        let args = args.into_iter().map(|arg| self.zonk(arg)).collect();
        self.out.box_frees.insert(expr, Res::Fn(member, args));
        Ok(())
    }

    /// Infers and coerces the argument `arg` of a call of `callee` to its parameter type `param`, giving a closure argument the signature of the parameter's `Fn` bound.
    pub(super) fn call_arg(&mut self, callee: Option<(ItemRef, &[Ty])>, arg: ExprId, (param, expectation): (Ty, Option<Ty>)) -> Result<()> {
        let callable = match *self.cx.types.kind(self.resolve(param)) {
            Kind::Ref(_, pointee) => pointee,
            _ => param,
        };
        let (hint, bound) = match callee {
            Some(callee) => (self.closure_hint(callee, arg, param)?, self.callable_hint(callee, callable)?),
            None => (None, None),
        };
        let closure = self.is_closure(arg);
        let soft = closure || matches!(self.body.expr(arg), Expr::Tuple(_));
        let hint = hint.or(expectation.filter(|_| soft));
        let found = self.expr(arg, Some(hint.unwrap_or(param)))?;
        self.coerce_arg((arg, found, closure), param, expectation)?;
        match bound {
            Some(bound) => self.tie_callable(callable, bound),
            None => Ok(()),
        }
    }

    fn coerce_arg(&mut self, (arg, found, closure): (ExprId, Ty, bool), param: Ty, expectation: Option<Ty>) -> Result<()> {
        let reified = expectation.is_some_and(|wanted| matches!(self.cx.types.kind(self.resolve(wanted)), Kind::FnPtr(..)));
        match expectation.filter(|_| !closure || reified) {
            Some(wanted) if self.coerces_to(found, wanted) => {
                self.coerce_expr(arg, found, wanted)?;
                self.unify(param, wanted)
            }
            _ => self.coerce_expr(arg, found, param),
        }
    }

    /// Whether `found` coerces to `wanted` by a coercion or by giving up the `mut` of a reference.
    pub(super) fn coerces_to(&mut self, found: Ty, wanted: Ty) -> bool {
        let shared = self.weakened(found, wanted);
        self.probe(|infer| matches!(infer.coercion(found, wanted), Ok(Some(_))) || shared.is_some_and(|(a, b)| infer.unify(a, b).is_ok()))
    }

    /// Unifies the signature of the callable `ty`, a closure, function item or function pointer, with `bound`, the function pointer type that an `Fn(A) -> B` bound on it spells.
    pub(super) fn tie_callable(&mut self, ty: Ty, bound: Ty) -> Result<()> {
        let Kind::FnPtr(inputs, output) = self.cx.types.kind(bound).clone() else { return Ok(()) };
        let Some((params, ret)) = self.callable_sig(ty)? else { return self.tie_output(ty, inputs, output) };
        if params.len() != inputs.len() {
            bail!("a callable of {} arguments where its bound takes {}", params.len(), inputs.len());
        }
        params.into_iter().zip(inputs).try_for_each(|(param, input)| self.unify(param, input))?;
        self.unify(ret, output)
    }

    /// Unifies `output` with the `FnOnce::Output` at `inputs` of `ty`, a known type that is callable through its own `Fn` impls.
    fn tie_output(&mut self, ty: Ty, inputs: Vec<Ty>, output: Ty) -> Result<()> {
        let once = self.cx.lang_item("fn_once");
        let (Some(once), false) = (once, matches!(self.cx.types.kind(self.resolve(ty)), Kind::Var(_))) else { return Ok(()) };
        let tuple = self.cx.types.intern(Kind::Tuple(inputs));
        let found = self.project(once, ty, vec![tuple], "Output")?;
        self.unify(found, output)
    }

    /// The parameters and return type of the closure, function item or function pointer `ty`, or the one a reference to it calls.
    pub(super) fn callable_sig(&mut self, ty: Ty) -> Result<Option<(Vec<Ty>, Ty)>> {
        let ty = self.resolve(ty);
        Ok(Some(match self.cx.types.kind(ty).clone() {
            Kind::Ref(_, inner) => return self.callable_sig(inner),
            Kind::Closure(..) => {
                let (params, ret, _) = self.closure_sig(ty)?;
                (params, ret)
            }
            Kind::FnDef(item, substs) => self.fn_types(item, &substs)?,
            Kind::FnPtr(params, ret) => (params, ret),
            Kind::Var(_) => match self.opaque_sig(ty).map(|sig| self.cx.types.kind(sig).clone()) {
                Some(Kind::FnPtr(params, ret)) => (params, ret),
                _ => return Ok(None),
            },
            _ => return Ok(None),
        }))
    }

    fn is_closure(&self, arg: ExprId) -> bool {
        matches!(*self.body.expr(self.wrapped(arg)), Expr::Closure { .. })
    }

    fn wrapped(&self, expr: ExprId) -> ExprId {
        match *self.body.expr(expr) {
            Expr::Paren(inner) => self.wrapped(inner),
            Expr::Block { stmts, tail: Some(tail), label: None, kind: BlockKind::Plain | BlockKind::Expansion | BlockKind::Unsafe } if self.body.list(stmts).is_empty() => self.wrapped(tail),
            _ => expr,
        }
    }

    fn closure_hint(&mut self, callee: (ItemRef, &[Ty]), arg: ExprId, param: Ty) -> Result<Option<Ty>> {
        let arg = self.wrapped(arg);
        match *self.body.expr(arg) {
            Expr::Closure { .. } => self.callable_hint(callee, param),
            Expr::Ref { raw: false, expr: inner, .. } => {
                let Kind::Ref(mutability, pointee) = *self.cx.types.kind(self.resolve(param)) else { return Ok(None) };
                let hint = self.closure_hint(callee, inner, pointee)?;
                Ok(hint.map(|hint| self.cx.types.intern(Kind::Ref(mutability, hint))))
            }
            _ => Ok(None),
        }
    }

    fn callable_hint(&mut self, (item, substs): (ItemRef, &[Ty]), param: Ty) -> Result<Option<Ty>> {
        let param = self.resolve(param);
        if !matches!(self.cx.types.kind(param), Kind::Var(_)) {
            return Ok(None);
        }
        let frame = self.frame(item, substs)?;
        let Some(at) = frame.generics.iter().position(|(_, ty)| self.resolve(*ty) == param) else { return Ok(None) };
        let sig = self.cx.signature(item)?;
        let src = &self.cx.file(item).source;
        let own: Vec<_> = super::scope::type_params(&sig.generics).collect();
        let (home, name) = (frame.home, frame.generics[at].0);
        let Some(index) = at.checked_sub(frame.generics.len() - own.len()) else { return self.with_frame(frame, |infer| infer.header_fn_bound(name, home)) };
        let bounds = super::scope::generic_bounds(&sig.body, src, &sig.predicates, own[index]);
        self.with_frame(frame, |infer| {
            for bound in bounds {
                if let Some(hint) = infer.fn_bound(&sig.body, home, bound)? {
                    return Ok(Some(hint));
                }
            }
            Ok(None)
        })
    }

    /// The function pointer type that an `Fn` bound on the outer parameter `name` spells in a header of the frame: the function's own, or its impl's or trait's.
    fn header_fn_bound(&mut self, name: &str, home: ModRef) -> Result<Option<Ty>> {
        for (header, src) in self.headers()? {
            let (arena, generics, predicates) = header.parts();
            for bound in super::scope::named_bounds(arena, src, (generics, predicates), name) {
                if let Some(hint) = self.in_source(src, |infer| infer.fn_bound(arena, home, bound))? {
                    return Ok(Some(hint));
                }
            }
        }
        Ok(None)
    }

    /// The function pointer type that an `impl Fn(A) -> B` return type of the body spells, which a closure it returns takes its signature from.
    pub(super) fn opaque_fn_sig(&mut self) -> Result<Option<Ty>> {
        self.ret_fn_sig(self.sig)
    }

    /// The function pointer type that an `impl Fn(A) -> B` return type of `sig` spells in the current frame.
    pub(super) fn ret_fn_sig(&mut self, sig: &Signature) -> Result<Option<Ty>> {
        let Some(crate::body::Ty::ImplTrait(_, bounds)) = sig.ret.map(|ret| *sig.body.ty(ret)) else { return Ok(None) };
        for &bound in sig.body.list(bounds) {
            if let Some(found) = self.fn_bound(&sig.body, self.home, bound)? {
                return Ok(Some(found));
            }
        }
        Ok(None)
    }

    /// The trait bounds other than `Fn` that an `impl Trait` return type of `sig` spells in the current frame, each with `ret` as `Self`.
    pub(super) fn ret_bounds(&mut self, sig: &Signature, ret: Ty) -> Result<Vec<(ItemRef, Vec<Ty>)>> {
        let Some(crate::body::Ty::ImplTrait(_, bounds)) = sig.ret.map(|ret| *sig.body.ty(ret)) else { return Ok(Vec::new()) };
        let mut found = Vec::new();
        for &bound in sig.body.list(bounds) {
            let Some((of, path)) = self.bound_trait_at(&sig.body, bound, self.home) else { continue };
            if !self.is_callable_trait(of) && let Ok(args) = self.bound_args(of, &sig.body, path, ret) {
                found.push((of, std::iter::once(ret).chain(args).collect()));
            }
        }
        Ok(found)
    }

    pub(super) fn fn_bound(&mut self, arena: &Body, home: ModRef, bound: u32) -> Result<Option<Ty>> {
        let Some((of, path)) = self.bound_trait_at(arena, bound, home) else { return Ok(None) };
        if !self.is_callable_trait(of) {
            return Ok(None);
        }
        let Some(&last) = arena.list(arena.path(path).segments).last() else { return Ok(None) };
        let (inputs, output) = match arena.segments[last as usize].args {
            Some(Args::Paren { inputs, output }) => self.paren_sig(arena, inputs, output)?,
            Some(Args::Angle(_)) => match self.angle_sig(arena, (of, path))? {
                Some(sig) => sig,
                None => return Ok(None),
            },
            _ => return Ok(None),
        };
        Ok(Some(self.cx.types.intern(Kind::FnPtr(inputs, output))))
    }

    /// The inputs and output that the angle form `FnOnce<(A, B), Output = R>` of callable trait `of` names, once its argument tuple is known.
    fn angle_sig(&mut self, arena: &Body, (of, path): (ItemRef, PathId)) -> Result<Option<(Vec<Ty>, Ty)>> {
        let self_ty = self.fresh(Kind::Var);
        let Some(&tuple) = self.bound_args(of, arena, path, self_ty)?.first() else { return Ok(None) };
        let Kind::Tuple(inputs) = self.cx.types.kind(self.resolve(tuple)).clone() else { return Ok(None) };
        let output = self.path_bindings(arena, path)?.into_iter().find(|(name, _)| name == "Output").map(|(_, ty)| ty);
        Ok(Some((inputs, output.unwrap_or_else(|| self.fresh(Kind::Var)))))
    }
}

/// Whether `item` is `Fn::call` or `FnMut::call_mut`, and how each takes the closure.
pub fn by_reference(cx: &Context<'_>, item: ItemRef) -> Option<Mut> {
    match cx.owner_trait(item).and_then(|of| cx.lang(of)) {
        Some("fn") => Some(Mut::Not),
        Some("fn_mut") => Some(Mut::Mut),
        _ => None,
    }
}

/// The typed form of a call trait method at a closure or function item and its argument tuple, which calls the closure's body or the function with the tuple's fields.
pub fn untuple_typed(cx: &mut Context<'_>, item: ItemRef, substs: &[Ty]) -> Result<Typed> {
    let (Some(&callee), Some(&tuple)) = (substs.first(), substs.get(1)) else { bail!("a call trait method with no callee and arguments") };
    let (target, ret) = match cx.types.kind(callee).clone() {
        Kind::FnDef(target, args) => {
            let ret = super::detached(cx, target, |infer| infer.fn_types(target, &args))?.1;
            (Res::Fn(target, args), ret)
        }
        _ => {
            let Some(info) = cx.closure(callee) else { bail!("a closure whose defining body was not inferred") };
            let Some(call_once) = cx.call_once() else { bail!("core has no `FnOnce::call_once`") };
            (Res::Fn(call_once, vec![callee, tuple]), info.closure.ret)
        }
    };
    let receiver = by_reference(cx, item).map_or(callee, |by| cx.types.intern(Kind::Ref(by, callee)));
    let locals = vec![Local { ty: receiver, mutable: false }, Local { ty: tuple, mutable: false }];
    Ok(Typed { locals, params: vec![0, 1], ret: Some(ret), shape: Shape::Untuple(target), ..Typed::default() })
}

/// Records the closures of the finished body `typed` of `parent` at `substs` in `cx`, so other bodies can call them.
pub fn register(cx: &mut Context<'_>, (parent, substs): (ItemRef, &[Ty]), typed: &Typed) {
    if typed.closures.is_empty() {
        return;
    }
    let shared = Rc::new(Typed { drops: Vec::new(), ..typed.clone() });
    for (&at, closure) in &typed.closures {
        let ty = cx.types.intern(Kind::Closure(parent, substs.to_vec(), at.0));
        let parts = closure.captures.iter().map(|capture| {
            let held = typed.locals[capture.local as usize].ty;
            match capture.by {
                By::Ref => cx.types.intern(Kind::Ref(Mut::Not, held)),
                By::Mut => cx.types.intern(Kind::Ref(Mut::Mut, held)),
                By::Value => held,
            }
        });
        let env = Kind::Tuple(parts.collect());
        let mut env = cx.types.intern(env);
        if closure.coroutine {
            let word = cx.types.prim(crate::names::Primitive::Usize);
            env = cx.types.intern(Kind::Tuple(vec![word, word, env]));
        }
        cx.set_closure(ty, ClosureInfo { parent, typed: shared.clone(), closure: closure.clone(), env });
    }
}

/// The typed form of `Future::poll` at the coroutine type `ty`: its defining body, resumed with the pinned coroutine and the task context, returning `Poll` of its output.
pub fn coroutine_typed(cx: &mut Context<'_>, poll: ItemRef, ty: Ty) -> Result<Typed> {
    let Some(info) = cx.closure(ty) else { bail!("a coroutine whose defining body was not inferred") };
    let (params, ret) = super::detached(cx, poll, |infer| infer.fn_types(poll, &[ty]))?;
    let mut typed = (*info.typed).clone();
    let first = typed.locals.len() as u32;
    typed.locals.extend(params.iter().map(|&ty| Local { ty, mutable: false }));
    typed.params = vec![first, first + 1];
    typed.ret = Some(ret);
    typed.shape = Shape::Closure(ClosureBody { parent: info.parent, closure: info.closure.clone(), env: Some(first) });
    typed.drops = super::drop::drop_calls(cx, &typed)?;
    Ok(typed)
}

/// The typed form of the function that the closure type `ty` runs: its defining body, with an environment parameter and the closure's own parameters.
pub fn closure_typed(cx: &mut Context<'_>, ty: Ty) -> Result<Typed> {
    let Some(info) = cx.closure(ty) else { bail!("a closure whose defining body was not inferred") };
    let mut typed = (*info.typed).clone();
    let env = cx.closure_takes_env(ty).then(|| {
        let env_ref = cx.types.intern(Kind::Ref(Mut::Mut, info.env));
        typed.locals.push(Local { ty: env_ref, mutable: false });
        typed.locals.len() as u32 - 1
    });
    typed.params = env.into_iter().chain(info.closure.params.iter().copied()).collect();
    typed.ret = Some(info.closure.ret);
    typed.shape = Shape::Closure(ClosureBody { parent: info.parent, closure: info.closure.clone(), env });
    typed.drops = super::drop::drop_calls(cx, &typed)?;
    Ok(typed)
}

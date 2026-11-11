use std::sync::Arc;

use anyhow::{Result, bail};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::body::{BlockKind, Expr, ExprId, List, Param, TyId, UnOp};
use crate::names::ItemRef;
use crate::outline::item::ItemKind;

use super::super::Context;
use super::super::substs::Substs;
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
    pub typed: std::sync::Arc<Typed>,
    pub closure: Closure,
    pub env: Ty,
}

/// The variables that one closure being inferred uses from outside it: locals below `first` belong to the bodies around it.
#[derive(Clone, Default)]
pub(super) struct Captures {
    first: u32,
    used: Vec<u32>,
    mutated: Vec<u32>,
    reads: Vec<(ExprId, u32)>,
}

impl<'a> Infer<'_, 'a> {
    pub(super) fn closure_expr(&mut self, expr: ExprId, (params, ret, body, by_move): (List, Option<TyId>, ExprId, bool), expected: Option<Ty>) -> Result<Ty> {
        let Some((parent, substs)) = self.instance else { bail!("a closure outside a function body") };
        let hint = expected.and_then(|ty| self.expected_sig(ty)).filter(|(inputs, _)| inputs.len() == self.body.list(params).len());
        let depth = self.scopes.len();
        self.uses.push(Captures { first: self.out.locals.len() as u32, ..Captures::default() });
        let ret = match (ret, &hint) {
            (Some(ty), _) => self.lower_ty(self.body, ty)?,
            (None, Some((_, output))) => *output,
            (None, None) => self.fresh(Kind::Var),
        };
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
        let found = self.closure_body(ret, |infer| infer.expr(body, Some(ret)).and_then(|found| infer.coerce_expr(body, found, ret)));
        self.scopes.truncate(depth);
        let uses = self.uses.pop().expect("the closure pushed its uses");
        found?;
        self.record_closure(expr, uses, (locals, ret, body, false), (by_move, &[]))?;
        Ok(self.cx.types.intern(Kind::Closure(parent, substs, expr.0)))
    }

    /// Infers the `async` block or body `expr` as a coroutine whose output is `output` when known, capturing by value when `by_move` and always capturing `forced`.
    pub(super) fn coroutine_expr(&mut self, expr: ExprId, (output, by_move): (Option<Ty>, bool), forced: &[u32]) -> Result<Ty> {
        self.poly_fallback();
        let Some((parent, substs)) = self.instance else { bail!("an `async` block outside a function body") };
        let Expr::Block { stmts, tail, .. } = *self.body.expr(expr) else { bail!("an `async` body that is not a block") };
        let depth = self.scopes.len();
        self.uses.push(Captures { first: self.out.locals.len() as u32, ..Captures::default() });
        let ret = output.unwrap_or_else(|| self.fresh(Kind::Var));
        let found = self.closure_body(ret, |infer| infer.block(stmts, tail, Some(ret)).and_then(|found| infer.coerce(found, ret)));
        self.scopes.truncate(depth);
        let uses = self.uses.pop().expect("the coroutine pushed its uses");
        found?;
        self.record_closure(expr, uses, (Vec::new(), ret, expr, true), (by_move, forced))?;
        Ok(self.cx.types.intern(Kind::Closure(parent, substs, expr.0)))
    }

    fn record_closure(&mut self, expr: ExprId, uses: Captures, (params, ret, body, coroutine): (Vec<u32>, Ty, ExprId, bool), (by_move, forced): (bool, &[u32])) -> Result<()> {
        let moved = self.decide_later(expr, &uses, (by_move, forced))?;
        let captures = self.captures(uses, (by_move, &moved), forced);
        self.out.closures.insert(expr, Closure { params, ret, body, captures, consumes: coroutine || !moved.is_empty(), coroutine });
        Ok(())
    }

    fn closure_body(&mut self, ret: Ty, run: impl FnOnce(&mut Self) -> Result<()>) -> Result<()> {
        let outer = (self.out.ret.replace(ret), std::mem::take(&mut self.loops), std::mem::take(&mut self.try_blocks), self.diverges);
        let found = run(self);
        (self.out.ret, self.loops, self.try_blocks, self.diverges) = outer;
        found
    }

    /// The output of `ty` when it is a coroutine: one this body defines, one whose defining body is inferred, or an async fn future named while its body is inferred.
    pub(super) fn coroutine_output(&self, ty: Ty) -> Option<Ty> {
        let Kind::Closure(parent, substs, at) = self.cx.types.kind(self.resolve(ty)) else {
            return None;
        };
        let here = self.instance.as_ref().is_some_and(|(item, args)| item == parent && args == substs);
        let local = here.then(|| self.out.closures.get(&ExprId(*at))).flatten().filter(|closure| closure.coroutine).map(|closure| closure.ret);
        local.or_else(|| self.cx.closure(self.resolve(ty)).filter(|info| info.closure.coroutine).map(|info| info.closure.ret)).or_else(|| self.cx.future_output(self.resolve(ty)))
    }

    /// Infers `inner.await`: the future that `IntoFuture::into_future` makes of `inner`, polled until it gives its output.
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
        let poll = Res::fn_of(&self.cx.types, item, &substs);
        self.out.awaits.insert(expr, super::Awaited { into_future, future, poll, polled });
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

    fn captures(&mut self, uses: Captures, (by_move, moved): (bool, &[u32]), forced: &[u32]) -> Vec<Capture> {
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

    /// The captured locals that the closure `expr` moves out of, keeping its uses to decide again once the body settles when a read's type is not known yet.
    fn decide_later(&mut self, expr: ExprId, uses: &Captures, (by_move, forced): (bool, &[u32])) -> Result<Vec<u32>> {
        let (moved, known) = self.moved_reads(&uses.reads)?;
        if !known {
            self.unsure.push((expr, uses.clone(), by_move, forced.to_vec()));
        }
        Ok(moved)
    }

    /// Decides again what each closure whose reads had unknown types captures, now that the body's types are settled.
    pub(super) fn settle_captures(&mut self) -> Result<()> {
        for (expr, uses, by_move, forced) in std::mem::take(&mut self.unsure) {
            let (moved, _) = self.moved_reads(&uses.reads)?;
            let captures = self.captures(uses, (by_move, &moved), &forced);
            let Some(closure) = self.out.closures.get_mut(&expr) else {
                continue;
            };
            closure.consumes = closure.coroutine || !moved.is_empty();
            closure.captures = captures;
        }
        Ok(())
    }

    /// The captured locals that the reads `reads` move out of: a read, or a field path from it, used as a value whose type needs dropping or is not `Copy`, where a `&mut` read reborrows.
    fn moved_reads(&mut self, reads: &[(ExprId, u32)]) -> Result<(Vec<u32>, bool)> {
        if reads.is_empty() {
            return Ok((Vec::new(), true));
        }
        let (places, parents) = self.place_uses();
        let (mut moved, mut known) = (Vec::new(), true);
        for &(read, local) in reads {
            let mut at = read;
            while let Some(&parent) = parents.get(&at) {
                at = parent;
            }
            let ty = self.resolve(self.out.exprs[at.0 as usize]);
            if places.contains(&at) || moved.contains(&local) {
                continue;
            }
            known &= !matches!(self.cx.types.kind(ty), Kind::Var(_));
            if self.moves(ty)? {
                moved.push(local);
            }
        }
        Ok((moved, known))
    }

    fn moves(&mut self, ty: Ty) -> Result<bool> {
        if let Some(captures) = self.own_captures(ty) {
            return captures.into_iter().try_fold(false, |moves, (local, by)| Ok(moves || by == By::Mut || (by == By::Value && self.moves(self.out.locals[local as usize].ty)?)));
        }
        let reborrows = matches!(self.cx.types.kind(self.resolve(ty)), Kind::Ref(Mut::Mut, _));
        Ok(self.needs_drop(ty)? || (!reborrows && !self.is_copy(ty)?))
    }

    /// What the closure type `ty` captures when this body defines it and its store entry is not written yet.
    fn own_captures(&self, ty: Ty) -> Option<Vec<(u32, By)>> {
        let closure = self.own_closure(ty)?;
        Some(closure.captures.iter().map(|capture| (capture.local, capture.by)).collect())
    }

    /// The closure of type `ty` when this body defines it and its store entry is not written yet.
    fn own_closure(&self, ty: Ty) -> Option<&Closure> {
        let ty = self.resolve(ty);
        let Kind::Closure(parent, substs, at) = self.cx.types.kind(ty) else {
            return None;
        };
        let here = self.instance.as_ref().is_some_and(|(item, args)| item == parent && args == substs);
        here.then(|| self.out.closures.get(&ExprId(*at))).flatten().filter(|_| !self.cx.has_closure(ty))
    }

    /// The expressions of the body used as places rather than values, and the field or parenthesis expression that projects each projected expression.
    fn place_uses(&mut self) -> (HashSet<ExprId>, HashMap<ExprId, ExprId>) {
        let (mut places, mut parents) = (HashSet::default(), HashMap::default());
        let call_once = self.cx.call_once();
        for (at, expr) in self.body.exprs.iter().enumerate() {
            let at = ExprId(at as u32);
            match *expr {
                Expr::Field(base, _) | Expr::TupleField(base, _) | Expr::Paren(base) => {
                    parents.insert(base, at);
                }
                Expr::Call(callee, _) if self.consumes_callee(at, callee, call_once) => {}
                Expr::Ref { expr, .. } | Expr::Unary(UnOp::Deref, expr) | Expr::Index(expr, _) | Expr::Assign(expr, _) | Expr::AssignOp(_, expr, _) | Expr::Match(expr, _) | Expr::Let(_, expr) | Expr::Call(expr, _) => {
                    places.insert(expr);
                }
                Expr::Binary(op, lhs, rhs) if super::expr::is_compare(op) => {
                    places.insert(lhs);
                    places.insert(rhs);
                }
                Expr::Method { receiver, .. } if self.out.adjusts.get(&at).is_some_and(|adjust| adjust.autoref.is_some() || adjust.derefs > 0) => {
                    places.insert(receiver);
                }
                _ => {}
            }
        }
        (places, parents)
    }

    /// Whether the call `call` of `callee` goes through `FnOnce::call_once` and so moves the callee: a closure that consumes itself, or a value such as a `Box<dyn FnOnce>`.
    fn consumes_callee(&self, call: ExprId, callee: ExprId, call_once: Option<ItemRef>) -> bool {
        let Some(Res::Fn(method, substs)) = self.out.res.get(&call) else {
            return false;
        };
        if Some(*method) != call_once {
            return false;
        }
        let ty = self.resolve(self.cx.types.substs(*substs).first().copied().unwrap_or(self.out.exprs[callee.0 as usize]));
        match self.cx.types.kind(ty) {
            Kind::Closure(..) => self.own_closure(ty).map_or_else(|| self.cx.closure_consumes(ty), |closure| closure.consumes),
            _ => true,
        }
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
        let Some(local) = self.place_local(expr) else {
            return Ok(());
        };
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

    fn ensure_closure(&mut self, ty: Ty) -> Result<Option<(u32, Option<Arc<ClosureInfo>>)>> {
        let Kind::Closure(parent, substs, at) = *self.cx.types.kind(ty) else { return Ok(None) };
        let local = self.instance == Some((parent, substs));
        if !local && !self.cx.has_closure(ty) {
            super::opaque::revealed(self.cx, parent, substs)?;
        }
        Ok(Some((at, (!local).then(|| self.cx.closure(ty)).flatten())))
    }

    /// The parameter and result types of the closure type `ty`, from this body when it defines the closure, and whether it captures anything.
    pub(super) fn closure_sig(&mut self, ty: Ty) -> Result<(Vec<Ty>, Ty, bool)> {
        let Some((at, info)) = self.ensure_closure(ty)? else { bail!("not a closure") };
        let (closure, locals) = match &info {
            Some(info) => (Some(&info.closure), &info.typed.locals),
            None => (self.out.closures.get(&ExprId(at)), &self.out.locals),
        };
        let Some(closure) = closure else { bail!("a closure whose body is not inferred yet") };
        Ok((closure.params.iter().map(|&local| locals[local as usize].ty).collect(), closure.ret, !closure.captures.is_empty()))
    }

    pub(super) fn closure_mutates(&mut self, ty: Ty) -> Result<bool> {
        let Some((at, info)) = self.ensure_closure(ty)? else {
            return Ok(false);
        };
        let closure = info.as_ref().map(|info| &info.closure).or_else(|| self.out.closures.get(&ExprId(at)));
        Ok(closure.is_some_and(|closure| closure.captures.iter().any(|capture| capture.by == By::Mut)))
    }

    /// Infers the call `expr` of the closure value of type `ty`, recording it as a call of `FnOnce::call_once` at the closure and its argument tuple.
    pub(super) fn closure_call(&mut self, expr: ExprId, ty: Ty, args: List) -> Result<Ty> {
        let (params, ret, _) = self.closure_sig(ty)?;
        self.infer_call_args(args, &params, None)?;
        let Some(call_once) = self.cx.call_once() else { bail!("core has no `FnOnce::call_once`") };
        let tuple = self.cx.types.tuple(&params);
        self.out.res.insert(expr, Res::fn_of(&self.cx.types, call_once, &[ty, tuple]));
        Ok(ret)
    }

    fn expected_sig(&mut self, expected: Ty) -> Option<(Vec<Ty>, Ty)> {
        let expected = self.resolve(expected);
        let own_ret = self.out.ret.is_some_and(|ret| self.resolve(ret) == expected);
        let expected = self.ret_sig.filter(|_| own_ret).unwrap_or(expected);
        match self.cx.types.kind(expected).clone() {
            Kind::FnPtr(inputs, output) => Some((self.cx.types.substs(inputs).to_vec(), output)),
            Kind::Dyn(of, parts, members, _) if self.is_callable_trait(of) => {
                let (first, output) = {
                    let (given, mut bound) = super::super::ty::object_parts(self.cx.types.substs(parts), &members);
                    (*given.first()?, bound.next().map(|(_, ty)| ty))
                };
                let Kind::Tuple(inputs) = *self.cx.types.kind(self.resolve(first)) else { return None };
                let inputs = self.cx.types.substs(inputs).to_vec();
                Some((inputs, output.unwrap_or_else(|| self.cx.types.unit())))
            }
            _ => None,
        }
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
        let propagate = closure || self.holds_closure(arg) || matches!(self.body.expr(self.wrapped(arg)), Expr::Array(_));
        let usable = expectation.filter(|&wanted| propagate || self.sized_expectation(wanted));
        let found = self.expr(arg, Some(hint.or(usable).unwrap_or(param)))?;
        self.coerce_arg((arg, found, closure), param, expectation)?;
        match bound {
            Some(bound) => self.tie_callable(callable, bound),
            None => Ok(()),
        }
    }

    fn sized_expectation(&mut self, ty: Ty) -> bool {
        let before = self.cx.quiet_mark();
        let filled = self.zonk(ty);
        matches!(self.cx.is_unsized(filled), Ok(false)) && self.cx.quiet_mark() == before
    }

    fn holds_closure(&self, expr: ExprId) -> bool {
        let any = |list: List| self.body.list(list).iter().any(|&at| self.holds_closure(ExprId(at)));
        match *self.body.expr(self.wrapped(expr)) {
            Expr::Closure { .. } => true,
            Expr::Tuple(items) | Expr::Array(items) => any(items),
            Expr::Ref { expr: inner, .. } => self.holds_closure(inner),
            Expr::If(_, then_branch, else_branch) => self.holds_closure(then_branch) || else_branch.is_some_and(|at| self.holds_closure(at)),
            Expr::Match(_, arms) => self.body.list(arms).iter().any(|&at| self.holds_closure(self.body.arms[at as usize].body)),
            Expr::Struct { fields, base, .. } => self.body.list(fields).iter().any(|&at| self.holds_closure(self.body.fields[at as usize].value)) || base.is_some_and(|at| self.holds_closure(at)),
            _ => false,
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
        self.probe(|infer| matches!(infer.coercion(found, wanted), Ok(Some(_))) || shared.is_some_and(|(a, b)| infer.unifies(a, b)))
    }

    /// Unifies the signature of the callable `ty`, a closure, function item or function pointer, with `bound`, the function pointer type that an `Fn(A) -> B` bound on it spells.
    pub(super) fn tie_callable(&mut self, ty: Ty, bound: Ty) -> Result<()> {
        let Kind::FnPtr(inputs, output) = *self.cx.types.kind(bound) else { return Ok(()) };
        let inputs = self.cx.types.substs(inputs).to_vec();
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
        let tuple = self.cx.types.tuple(&inputs);
        let found = self.project(once, ty, vec![tuple], "Output")?;
        self.unify(found, output)
    }

    /// The parameters and return type of the closure, function item or function pointer `ty`, or the one a reference to it calls.
    pub(super) fn callable_sig(&mut self, ty: Ty) -> Result<Option<(Vec<Ty>, Ty)>> {
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
            Kind::Param(_) | Kind::Proj(_) => match self.poly_fn_sig(ty) {
                Some(sig) => sig,
                None => return Ok(None),
            },
            Kind::Var(_) => match self.opaque_sig(ty).map(|sig| self.cx.types.kind(sig).clone()) {
                Some(Kind::FnPtr(params, ret)) => (self.cx.types.substs(params).to_vec(), ret),
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
            Expr::Block { tail: Some(tail), label: None, kind: BlockKind::Plain | BlockKind::Expansion | BlockKind::Unsafe, .. } => self.wrapped(tail),
            _ => expr,
        }
    }

    fn closure_hint(&mut self, callee: (ItemRef, &[Ty]), arg: ExprId, param: Ty) -> Result<Option<Ty>> {
        let arg = self.wrapped(arg);
        match *self.body.expr(arg) {
            Expr::Closure { .. } => self.callable_hint(callee, param),
            Expr::Ref { raw: false, expr: inner, .. } => {
                let Kind::Ref(mutability, pointee) = *self.cx.types.kind(self.resolve(param)) else {
                    return Ok(None);
                };
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
        if matches!(self.cx.item(item).1.kind, ItemKind::Struct | ItemKind::Enum | ItemKind::Union) {
            return self.field_hint((item, substs), param);
        }
        let frame = self.frame(item, substs)?;
        let Some(at) = frame.generics.iter().position(|(_, ty)| self.resolve(*ty) == param) else {
            return Ok(None);
        };
        let sig = self.cx.signature(item)?;
        let src = &self.cx.file(item).source;
        let own: Vec<_> = super::scope::type_params(&sig.generics).collect();
        let (home, name) = (frame.home, frame.generics[at].0);
        let Some(index) = at.checked_sub(frame.generics.len() - own.len()) else {
            return self.with_frame(frame, |infer| infer.header_fn_bound(name, home));
        };
        let bounds = super::scope::generic_bounds(&sig.body, src, &sig.predicates, own[index]);
        self.first_fn_bound(frame, &sig.body, bounds)
    }

    /// The function pointer type that an `Fn` bound declared on the struct, enum or union `item` spells for its type parameter that `param` fills in `args`.
    fn field_hint(&mut self, (item, args): (ItemRef, &[Ty]), param: Ty) -> Result<Option<Ty>> {
        let def = self.cx.adt_def(item)?;
        let Some(at) = args.iter().position(|&arg| self.resolve(arg) == param) else {
            return Ok(None);
        };
        let Some(generic) = super::scope::type_params(&def.generics.params).nth(at) else {
            return Ok(None);
        };
        let bounds = super::scope::generic_bounds(&def.body, &self.cx.file(item).source, &def.generics.predicates, generic);
        if bounds.is_empty() {
            return Ok(None);
        }
        let frame = self.adt_frame(item, &def.generics.params, args)?;
        self.first_fn_bound(frame, &def.body, bounds)
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
pub fn untuple_typed(cx: &mut Context<'_>, item: ItemRef, substs: Substs) -> Result<Typed> {
    let slice = cx.types.substs(substs);
    let (Some(&callee), Some(&tuple)) = (slice.first(), slice.get(1)) else { bail!("a call trait method with no callee and arguments") };
    let (target, ret) = match *cx.types.kind(callee) {
        Kind::FnDef(target, args) => {
            let arg_slice = cx.types.substs(args).to_vec();
            let (res, (_, ret)) = super::detached(cx, target, |infer| Ok((infer.trait_call(target, args)?, infer.fn_types(target, &arg_slice)?)))?;
            (res, ret)
        }
        _ => {
            let Some(info) = cx.closure(callee) else { bail!("a closure whose defining body was not inferred") };
            let Some(call_once) = cx.call_once() else { bail!("core has no `FnOnce::call_once`") };
            (Res::fn_of(&cx.types, call_once, &[callee, tuple]), info.closure.ret)
        }
    };
    let receiver = by_reference(cx, item).map_or(callee, |by| cx.types.intern(Kind::Ref(by, callee)));
    let locals = vec![Local { ty: receiver }, Local { ty: tuple }];
    Ok(Typed { locals, params: vec![0, 1], ret: Some(ret), shape: Shape::Untuple(target), ..Typed::default() })
}

/// Records the closures of the finished body `typed` of `parent` at `substs` in `cx`, so other bodies can call them.
pub fn register(cx: &mut Context<'_>, (parent, substs): (ItemRef, Substs), typed: &Typed) {
    if typed.closures.is_empty() {
        return;
    }
    let shared = Arc::new(Typed { drops: Vec::new(), ..typed.clone() });
    for (at, closure) in &typed.closures {
        let ty = cx.types.intern(Kind::Closure(parent, substs, at.0));
        let parts: Vec<Ty> = closure
            .captures
            .iter()
            .map(|capture| {
                let held = typed.locals[capture.local as usize].ty;
                match capture.by {
                    By::Ref => cx.types.intern(Kind::Ref(Mut::Not, held)),
                    By::Mut => cx.types.intern(Kind::Ref(Mut::Mut, held)),
                    By::Value => held,
                }
            })
            .collect();
        let mut env = cx.types.tuple(&parts);
        if closure.coroutine {
            let word = cx.types.prim(crate::names::Primitive::Usize);
            env = cx.types.tuple(&[word, word, env]);
        }
        cx.set_closure(ty, ClosureInfo { parent, typed: shared.clone(), closure: closure.clone(), env });
    }
}

fn finish_closure(cx: &mut Context<'_>, mut typed: Typed, info: &ClosureInfo, env: Option<u32>, ret: Ty) -> Result<Typed> {
    typed.ret = Some(ret);
    typed.shape = Shape::Closure(ClosureBody { parent: info.parent, closure: info.closure.clone(), env });
    typed.drops = super::drop::drop_calls(cx, &typed)?;
    Ok(typed)
}

/// The typed form of `Future::poll` at the coroutine type `ty`: its defining body, resumed with the pinned coroutine and the task context, returning `Poll` of its output.
pub fn coroutine_typed(cx: &mut Context<'_>, poll: ItemRef, ty: Ty) -> Result<Typed> {
    let Some(info) = cx.closure(ty) else { bail!("a coroutine whose defining body was not inferred") };
    let (params, ret) = super::detached(cx, poll, |infer| infer.fn_types(poll, &[ty]))?;
    let mut typed = (*info.typed).clone();
    let first = typed.locals.len() as u32;
    typed.locals.extend(params.iter().map(|&ty| Local { ty }));
    typed.params = vec![first, first + 1];
    finish_closure(cx, typed, &info, Some(first), ret)
}

/// The typed form of the function that the closure type `ty` runs: its defining body, with an environment parameter and the closure's own parameters.
pub fn closure_typed(cx: &mut Context<'_>, ty: Ty) -> Result<Typed> {
    let Some(info) = cx.closure(ty) else { bail!("a closure whose defining body was not inferred") };
    let mut typed = (*info.typed).clone();
    let env = cx.closure_takes_env(ty).then(|| {
        typed.locals.push(Local { ty: cx.types.intern(Kind::Ref(Mut::Mut, info.env)) });
        typed.locals.len() as u32 - 1
    });
    typed.params = env.into_iter().chain(info.closure.params.iter().copied()).collect();
    finish_closure(cx, typed, &info, env, info.closure.ret)
}

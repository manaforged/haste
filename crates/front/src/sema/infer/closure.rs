use std::rc::Rc;

use anyhow::{Result, bail};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::body::{Args, Body, Bound, Expr, ExprId, List, Param, TyId, UnOp};
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
        let hint = match expected.map(|ty| self.cx.types.kind(self.resolve(ty)).clone()) {
            Some(Kind::FnPtr(inputs, output)) if inputs.len() == self.body.list(params).len() => Some((inputs, output)),
            _ => None,
        };
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
            locals.push(self.param_pat(arena, pat, ty)?);
        }
        let ret = match (ret, hint) {
            (Some(ty), _) => self.lower_ty(self.body, ty)?,
            (None, Some((_, output))) => output,
            (None, None) => self.fresh(Kind::Var),
        };
        let outer = (self.out.ret.replace(ret), std::mem::take(&mut self.loops), self.diverges);
        let found = self.expr(body, Some(ret)).and_then(|found| self.coerce_expr(body, found, ret));
        (self.out.ret, self.loops, self.diverges) = outer;
        self.scopes.truncate(depth);
        let uses = self.uses.pop().expect("the closure pushed its uses");
        found?;
        let moved = self.moved_reads(&uses.reads)?;
        let consumes = !moved.is_empty();
        let captures = self.captures(uses, (by_move, &moved));
        self.out.closures.insert(expr, Closure { params: locals, ret, body, captures, consumes });
        Ok(self.cx.types.intern(Kind::Closure(parent, substs, expr.0)))
    }

    fn captures(&mut self, uses: Uses, (by_move, moved): (bool, &[u32])) -> Vec<Capture> {
        let mut captures: Vec<Capture> = Vec::new();
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
    pub(super) fn closure_sig(&self, ty: Ty) -> Result<(Vec<Ty>, Ty, bool)> {
        let Kind::Closure(parent, ref substs, at) = *self.cx.types.kind(ty) else { bail!("not a closure") };
        let local = self.instance.as_ref().is_some_and(|(item, args)| *item == parent && args == substs);
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
            self.call_arg(None, ExprId(arg), param)?;
        }
        let Some(call_once) = self.cx.call_once() else { bail!("core has no `FnOnce::call_once`") };
        let tuple = self.cx.types.intern(Kind::Tuple(params));
        self.out.res.insert(expr, Res::Fn(call_once, vec![ty, tuple]));
        Ok(ret)
    }

    /// Infers and coerces the argument `arg` of a call of `callee` to its parameter type `param`, giving a closure argument the signature of the parameter's `Fn` bound.
    pub(super) fn call_arg(&mut self, callee: Option<(ItemRef, &[Ty])>, arg: ExprId, param: Ty) -> Result<()> {
        let hint = match callee {
            Some(callee) => self.closure_hint(callee, arg, param)?,
            None => None,
        };
        let found = self.expr(arg, Some(hint.unwrap_or(param)))?;
        self.coerce_expr(arg, found, param)
    }

    fn closure_hint(&mut self, callee: (ItemRef, &[Ty]), arg: ExprId, param: Ty) -> Result<Option<Ty>> {
        match *self.body.expr(arg) {
            Expr::Paren(inner) => self.closure_hint(callee, inner, param),
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
        let Some(&(name, _)) = frame.generics.iter().find(|(_, ty)| self.resolve(*ty) == param) else { return Ok(None) };
        let sig = self.cx.signature(item)?;
        let src = &self.cx.file(item).source;
        let bounds = super::scope::named_bounds(&sig.body, src, (&sig.generics, &sig.predicates), name);
        let home = frame.home;
        self.with_frame(frame, |infer| {
            for bound in bounds {
                if let Some(hint) = infer.fn_bound(&sig.body, home, bound)? {
                    return Ok(Some(hint));
                }
            }
            Ok(None)
        })
    }

    fn fn_bound(&mut self, arena: &Body, home: ModRef, bound: u32) -> Result<Option<Ty>> {
        let Bound::Trait { path, maybe: false, .. } = arena.bounds[bound as usize] else { return Ok(None) };
        let Some(of) = self.trait_named(arena, path, home) else { return Ok(None) };
        if !self.cx.lang(of).is_some_and(|lang| super::select::CALLABLE.contains(&lang)) {
            return Ok(None);
        }
        let Some(&last) = arena.list(arena.path(path).segments).last() else { return Ok(None) };
        let Some(Args::Paren { inputs, output }) = arena.segments[last as usize].args else { return Ok(None) };
        let inputs = arena.list(inputs).iter().map(|&at| self.lower_ty(arena, TyId(at))).collect::<Result<Vec<_>>>()?;
        let output = match output {
            Some(ty) => self.lower_ty(arena, ty)?,
            None => self.cx.types.unit(),
        };
        Ok(Some(self.cx.types.intern(Kind::FnPtr(inputs, output))))
    }
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
        let env = cx.types.intern(env);
        cx.set_closure(ty, ClosureInfo { parent, typed: shared.clone(), closure: closure.clone(), env });
    }
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

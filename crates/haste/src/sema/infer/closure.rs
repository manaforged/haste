use std::sync::Arc;

use anyhow::{Result, bail};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::body::{BlockKind, Expr, ExprId, List, Param, Pat, PatId, Stmt, TyId, UnOp};
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
        Kind::Tuple(items) => cx.types.substs(items).get(index as usize).copied().ok_or_else(|| anyhow::anyhow!("capture tuple field out of bounds")),
        Kind::Adt(item, args) => {
            let slice = cx.types.substs(args).to_vec();
            super::field_types(cx, item, &slice)?.get(variant as usize).and_then(|f| f.get(index as usize)).copied().ok_or_else(|| anyhow::anyhow!("capture adt field out of bounds"))
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
}

/// A closure type once its defining body is inferred: the body's typed form, the closure, and the tuple its captures form.
#[derive(Debug)]
pub struct ClosureInfo {
    pub parent: ItemRef,
    pub typed: std::sync::Arc<Typed>,
    pub closure: Closure,
    pub env: Ty,
}

#[derive(Clone, Default)]
pub(super) struct Captures {
    first: u32,
    used: Vec<u32>,
    mutated: Vec<(u32, ExprId)>,
    reads: Vec<(ExprId, u32)>,
    inner_closures: Vec<ExprId>,
}

impl<'a> Infer<'_, 'a> {
    pub(super) fn closure_expr(&mut self, expr: ExprId, (params, ret, body, by_move): (List, Option<TyId>, ExprId, bool), expected: Option<Ty>) -> Result<Ty> {
        let Some((parent, substs)) = self.instance else { bail!("a closure outside a function body") };
        let hint = expected.and_then(|ty| self.expected_sig(ty)).filter(|(inputs, _)| inputs.len() == self.body.list(params).len());
        let depth = self.scopes.len();
        self.uses.push(Captures { first: self.out.locals.len() as u32, ..Captures::default() });
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
        let async_body = matches!(*self.body.expr(body), Expr::Block { kind: crate::body::BlockKind::Async, stmts, .. } if self.body.list(stmts).is_empty());
        let found = self.closure_body(ret, |infer| {
            let found = if async_body { let ty = infer.coroutine_expr(body, (None, false), &param_locals)?; infer.out.exprs[body.0 as usize] = ty; ty } else { infer.expr(body, Some(ret))? };
            infer.coerce_expr(body, found, ret)
        });
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

    /// The type of a closure parameter or return: its annotation, made the same as the expected `hinted` where the two agree, else `hinted`, else a fresh variable.
    fn closure_part(&mut self, annotated: Option<TyId>, hinted: Option<Ty>) -> Result<Ty> {
        let Some(annotated) = annotated else { return Ok(hinted.unwrap_or_else(|| self.fresh(Kind::Var))) };
        let ty = self.lower_ty(self.body, annotated)?;
        if let Some(hinted) = hinted { self.unifies(ty, hinted); }
        Ok(ty)
    }

    fn record_closure(&mut self, expr: ExprId, uses: Captures, (params, ret, body, coroutine): (Vec<u32>, Ty, ExprId, bool), (by_move, forced): (bool, &[u32])) -> Result<()> {
        if let Some(around) = self.uses.last_mut() { around.inner_closures.push(expr); }
        let (captures, consumes, known) = self.decide_captures(&uses, (by_move, forced))?;
        if !known { self.unsure.push((expr, uses, by_move, forced.to_vec())); }
        self.out.closures.insert(expr, Closure { params, ret, body, captures, consumes: coroutine || consumes, coroutine });
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
        let Kind::Closure(parent, substs, at) = self.cx.types.kind(self.resolve(ty)) else { return None };
        let here = self.instance.as_ref().is_some_and(|(item, args)| item == parent && args == substs);
        let local = here.then(|| self.out.closures.get(&ExprId(*at))).flatten().filter(|c| c.coroutine).map(|c| c.ret);
        local.or_else(|| self.cx.closure(self.resolve(ty)).filter(|i| i.closure.coroutine).map(|i| i.closure.ret)).or_else(|| self.cx.future_output(self.resolve(ty)))
    }

    /// Infers `inner.await`: the future that `IntoFuture::into_future` makes of `inner`, polled until it gives its output.
    pub(super) fn await_expr(&mut self, expr: ExprId, inner: ExprId) -> Result<Ty> {
        self.poly_fallback();
        let found = self.expr(inner, None)?;
        if matches!(self.cx.types.kind(self.resolve(found)), Kind::Var(_)) { self.settle_opaques(false)?; }
        if let Some(output) = self.opaque_output(self.resolve(found)) { self.awaits.push((expr, found)); return Ok(output); }
        self.awaited(expr, found)
    }

    /// Finishes each `.await` on an async fn future that was pending, once the futures are revealed.
    pub(super) fn settle_awaits(&mut self) -> Result<()> {
        for (expr, found) in std::mem::take(&mut self.awaits) {
            let output = self.awaited(expr, found)?;
            self.unify(output, self.out.exprs[expr.0 as usize])?;
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
        self.out.awaits.insert(expr, super::Awaited { into_future, future, poll: Res::fn_of(&self.cx.types, item, &substs), polled });
        Ok(output)
    }

    /// The `Future::poll` that polls a future of type `future`: the compiler's for a coroutine or a trait object, else the one of the impl that applies.
    fn future_poll(&mut self, of: ItemRef, poll: ItemRef, future: Ty) -> Result<(ItemRef, Vec<Ty>)> {
        let future = self.resolve(future);
        if self.compiler_impl(of, future) || self.object_holds(of, future, &[]) == Some(true) { return Ok((poll, vec![future])); }
        let Some((block, args)) = self.select(of, future, &[])? else { bail!("{} is not a future", self.show(future)) };
        let Some(member) = self.cx.member(block, "poll") else { bail!("the `Future` impl of {} has no `poll`", self.show(future)) };
        Ok((member, args.into_iter().map(|arg| self.zonk(arg)).collect()))
    }

    fn max_by(a: By, b: By) -> By { match (a, b) { (By::Value, _) | (_, By::Value) => By::Value, (By::Mut, _) | (_, By::Mut) => By::Mut, _ => By::Ref } }
    fn resolve_opt(&self, ty: Ty) -> Ty { if ty == Ty::NONE { Ty::NONE } else { self.resolve(ty) } }
    fn append_step(path: &[(u32, u32)], idx: u32) -> Vec<(u32, u32)> { let mut p = path.to_vec(); p.push((0, idx)); p }

    fn projectable_fields(&mut self, ty: Ty) -> Result<Option<usize>> {
        let ty = self.resolve(ty);
        Ok(match *self.cx.types.kind(ty) {
            Kind::Tuple(items) => Some(self.cx.types.substs(items).len()),
            Kind::Adt(item, _) if self.cx.boxed(ty).is_none() && self.cx.lang(item) != Some("manually_drop") && matches!(self.cx.item(item).1.kind, ItemKind::Struct | ItemKind::Enum) => {
                let def = self.cx.adt_def(item)?;
                (def.variants.len() == 1).then(|| def.variants[0].fields.len())
            }
            _ => None,
        })
    }

    fn resolve_path(&mut self, mut ty: Ty, path: &[(u32, u32)], stop_drop: bool, known: &mut bool) -> Result<(Vec<(u32, u32)>, Ty)> {
        let mut out = Vec::new();
        let mut drop_at = None;
        for &step in path {
            ty = self.resolve_opt(ty);
            if ty == Ty::NONE { return Ok((out, Ty::NONE)); }
            if self.has_vars(ty) { *known = false; return Ok((out, if stop_drop { ty } else { Ty::NONE })); }
            if stop_drop && drop_at.is_none() && matches!(self.cx.types.kind(ty), Kind::Adt(..)) && self.drop_impl(ty)?.is_some() {
                drop_at = Some((out.len(), ty));
            }
            out.push(step);
            ty = step_field_ty(self.cx, ty, step)?;
        }
        let ty = self.resolve_opt(ty);
        if ty != Ty::NONE && self.has_vars(ty) { *known = false; }
        if let Some((len, drop_ty)) = drop_at && (ty == Ty::NONE || self.has_vars(ty) || !self.is_copy(ty)? || self.needs_drop(ty)?) {
            out.truncate(len);
            return Ok((out, drop_ty));
        }
        Ok((out, ty))
    }

    fn truncate_value_paths(&mut self, caps: &mut [Capture], known: &mut bool) -> Result<()> {
        for cap in caps {
            if cap.by == By::Value && !cap.path.is_empty() {
                (cap.path, _) = self.resolve_path(self.out.locals[cap.local as usize].ty, &cap.path, true, known)?;
            }
        }
        Ok(())
    }

    fn expr_proj(&mut self, expr: ExprId) -> Result<(Vec<(u32, u32)>, bool)> {
        let (mut rev, mut cur, mut known) = (Vec::new(), expr, true);
        loop {
            match *self.body.expr(cur) {
                Expr::Paren(inner) => cur = inner,
                Expr::TupleField(base, idx) if !self.out.autoderefs.contains_key(&cur) => { rev.push((base, (0, idx))); cur = base; }
                Expr::Field(base, _) if !self.out.autoderefs.contains_key(&cur) => { let Some(&idx) = self.out.fields.get(&cur) else { rev.clear(); break; }; rev.push((base, (0, idx))); cur = base; }
                Expr::Index(base, _) | Expr::Unary(UnOp::Deref, base) => { rev.clear(); cur = base; }
                _ => break,
            }
        }
        let mut out = Vec::with_capacity(rev.len());
        for (base, step) in rev.into_iter().rev() {
            let ty = self.resolve_opt(self.out.exprs[base.0 as usize]);
            if ty != Ty::NONE && matches!(self.cx.types.kind(ty), Kind::Var(_)) { known = false; }
            if ty == Ty::NONE || self.projectable_fields(ty)?.is_none() { break; }
            out.push(step);
        }
        Ok((out, known))
    }

    /// Notes that the path expression `expr` names local `local`, which a closure being inferred may move out of.
    pub(super) fn note_read(&mut self, expr: ExprId, local: u32) { self.uses.iter_mut().filter(|uses| local < uses.first).for_each(|uses| uses.reads.push((expr, local))); }

    /// Decides again what each closure whose reads had unknown types captures, now that the body's types are settled.
    pub(super) fn settle_captures(&mut self) -> Result<()> {
        for (expr, uses, by_move, forced) in std::mem::take(&mut self.unsure) {
            let (captures, consumes, _) = self.decide_captures(&uses, (by_move, &forced))?;
            if let Some(closure) = self.out.closures.get_mut(&expr) { (closure.consumes, closure.captures) = (closure.coroutine || consumes, captures); }
        }
        Ok(())
    }

    fn walk_read_proj(&mut self, read: ExprId, parents: &HashMap<ExprId, ExprId>, known: &mut bool) -> Result<(ExprId, Vec<(u32, u32)>)> {
        let (mut at, mut proj, mut ok) = (read, Vec::new(), true);
        while let Some(&parent) = parents.get(&at) {
            if ok {
                let cur_ty = self.resolve_opt(self.out.exprs[at.0 as usize]);
                if cur_ty != Ty::NONE && matches!(self.cx.types.kind(cur_ty), Kind::Var(_)) { *known = false; }
                let step = match *self.body.expr(parent) {
                    Expr::Paren(_) => { at = parent; continue; }
                    Expr::TupleField(_, idx) if !self.out.autoderefs.contains_key(&parent) => Some((0, idx)),
                    Expr::Field(_, _) if !self.out.autoderefs.contains_key(&parent) => self.out.fields.get(&parent).map(|&idx| (0, idx)),
                    _ => None,
                };
                match step { Some(s) if cur_ty != Ty::NONE && self.projectable_fields(cur_ty)?.is_some() => proj.push(s), _ => ok = false }
            }
            at = parent;
        }
        Ok((at, proj))
    }

    fn decide_captures(&mut self, uses: &Captures, (by_move, forced): (bool, &[u32])) -> Result<(Vec<Capture>, bool, bool)> {
        let mut known = true;
        let mut raw: Vec<Capture> = forced.iter().map(|&local| Capture { local, by: By::Value, path: Vec::new() }).collect();
        let mut read_locals = HashSet::default();
        if !uses.reads.is_empty() || !uses.inner_closures.is_empty() {
            let (places, parents, deref_parents, scrutinees) = self.place_uses();
            for &(read, local) in &uses.reads {
                read_locals.insert(local);
                let (at, proj) = self.walk_read_proj(read, &parents, &mut known)?;
                if let Some(pats) = scrutinees.get(&at) {
                    let at_ty = self.resolve_opt(self.out.exprs[at.0 as usize]);
                    known &= at_ty == Ty::NONE || !matches!(self.cx.types.kind(at_ty), Kind::Var(_));
                    for &pat in pats { self.pat_captures(pat, &proj, (false, true), &mut known, local, &mut raw)?; }
                } else if places.contains(&at) {
                    let mut cur = at;
                    while let Some(&p) = parents.get(&cur).or_else(|| deref_parents.get(&cur)) { cur = p; }
                    let mut sub = Vec::new();
                    for &pat in scrutinees.get(&cur).into_iter().flatten() { self.pat_captures(pat, &[], (true, false), &mut known, local, &mut sub)?; }
                    raw.push(Capture { local, by: if sub.iter().any(|c| c.by == By::Mut) { By::Mut } else { By::Ref }, path: proj });
                } else {
                    let ty = self.resolve_opt(self.out.exprs[at.0 as usize]);
                    if ty == Ty::NONE { continue; }
                    known &= !matches!(self.cx.types.kind(ty), Kind::Var(_));
                    raw.push(Capture { local, by: if self.moves(ty)? { By::Value } else { By::Ref }, path: proj });
                }
            }
            for &child in &uses.inner_closures {
                let Some(caps) = self.out.closures.get(&child).map(|c| c.captures.clone()) else { continue };
                for cap in caps.into_iter().filter(|c| c.local < uses.first) {
                    let (_, cap_ty) = self.resolve_path(self.out.locals[cap.local as usize].ty, &cap.path, false, &mut known)?;
                    let by = if cap.by == By::Value && cap_ty != Ty::NONE && (!self.is_copy(cap_ty)? || self.needs_drop(cap_ty)?) { By::Value }
                        else if cap.by == By::Mut { By::Mut } else { By::Ref };
                    raw.push(Capture { local: cap.local, by, path: cap.path });
                }
            }
        }
        for &(local, mut_expr) in &uses.mutated {
            let (mut_proj, mut_known) = self.expr_proj(mut_expr)?;
            known &= mut_known;
            raw.push(Capture { local, by: By::Mut, path: mut_proj });
        }
        if uses.inner_closures.is_empty() { uses.used.iter().filter(|l| !read_locals.contains(l)).for_each(|&local| raw.push(Capture { local, by: By::Ref, path: Vec::new() })); }
        let consumes = raw.iter().any(|c| c.by == By::Value);
        if by_move { raw.iter_mut().for_each(|cap| cap.by = By::Value); }
        self.truncate_value_paths(&mut raw, &mut known)?;
        raw.sort_by(|a, b| (a.local, a.path.len(), &a.path).cmp(&(b.local, b.path.len(), &b.path)));
        let mut captures: Vec<Capture> = Vec::new();
        for cap in raw {
            if let Some(prev) = captures.iter_mut().find(|p| p.local == cap.local && cap.path.starts_with(&p.path)) { prev.by = Self::max_by(prev.by, cap.by); }
            else { captures.push(cap); }
        }
        self.truncate_value_paths(&mut captures, &mut known)?;
        Ok((captures, consumes, known))
    }

    fn pat_res_inspects(&mut self, pat: PatId) -> bool {
        match self.out.pat_res.get(&pat).copied() {
            Some(Res::Ctor(item, _, _)) => self.cx.adt_def(item).is_ok_and(|def| def.variants.len() > 1),
            Some(Res::Fn(..)) => true,
            _ => false,
        }
    }

    fn pat_captures(&mut self, pat: PatId, path: &[(u32, u32)], (through_ref, allow_proj): (bool, bool), known: &mut bool, local: u32, out: &mut Vec<Capture>) -> Result<()> {
        let ty = self.resolve_opt(self.out.pats.get(&pat).copied().unwrap_or(Ty::NONE));
        if ty != Ty::NONE && matches!(self.cx.types.kind(ty), Kind::Var(_)) { *known = false; }
        let derefs = self.out.pat_derefs.get(&pat).copied().unwrap_or(0);
        let through_ref = through_ref || derefs > 0;
        let allow_proj = allow_proj && derefs == 0;
        let push = |out: &mut Vec<Capture>, by: By| out.push(Capture { local, by, path: path.to_vec() });
        match *self.body.pat(pat) {
            Pat::Wild | Pat::Rest => {}
            Pat::Paren(inner) => self.pat_captures(inner, path, (through_ref, allow_proj), known, local, out)?,
            Pat::Ref { pat: inner, .. } => self.pat_captures(inner, path, (true, false), known, local, out)?,
            Pat::Box(inner) | Pat::Deref(inner) => { push(out, By::Ref); self.pat_captures(inner, path, (through_ref, false), known, local, out)?; }
            Pat::Bind { sub, .. } => {
                if self.out.pat_res.contains_key(&pat) { if self.pat_res_inspects(pat) { push(out, By::Ref); } }
                else if let Some(&m) = self.out.ref_binds.get(&pat) { push(out, if m == Mut::Mut { By::Mut } else { By::Ref }); }
                else { push(out, if !through_ref && ty != Ty::NONE && self.moves(ty)? { By::Value } else { By::Ref }); }
                if let Some(sub) = sub { self.pat_captures(sub, path, (through_ref, allow_proj), known, local, out)?; }
            }
            Pat::Lit(_) | Pat::Range { .. } | Pat::Path(_) | Pat::Never | Pat::Macro { .. } => push(out, By::Ref),
            Pat::Tuple(list) | Pat::TupleStruct(_, list) => {
                if allow_proj && !self.pat_res_inspects(pat) && ty != Ty::NONE && let Some(count) = self.projectable_fields(ty)? {
                    let (pats, rest) = super::rest_split(self.body, list)?;
                    for (at, &sub) in pats.to_vec().iter().enumerate() {
                        if let Some(idx) = super::position(at, pats.len(), rest, count) { self.pat_captures(sub, &Self::append_step(path, idx as u32), (through_ref, true), known, local, out)?; }
                    }
                    return Ok(());
                }
                if matches!(*self.body.pat(pat), Pat::TupleStruct(..)) && self.pat_res_inspects(pat) { push(out, By::Ref); }
                for &at in &self.body.list(list).to_vec() { self.pat_captures(PatId(at), path, (through_ref, false), known, local, out)?; }
            }
            Pat::Struct { fields, .. } => {
                let can_proj = allow_proj && !self.pat_res_inspects(pat) && ty != Ty::NONE && self.projectable_fields(ty)?.is_some();
                if !can_proj && self.pat_res_inspects(pat) { push(out, By::Ref); }
                for &at in &self.body.list(fields).to_vec() {
                    let sub = self.body.field_pats[at as usize].pat;
                    if can_proj && let Some(&idx) = self.out.field_pats.get(&sub) { self.pat_captures(sub, &Self::append_step(path, idx), (through_ref, true), known, local, out)?; }
                    else { if can_proj { *known = false; } self.pat_captures(sub, path, (through_ref, false), known, local, out)?; }
                }
            }
            Pat::Or(list) | Pat::Slice(list) => {
                if matches!(*self.body.pat(pat), Pat::Slice(_)) && ty != Ty::NONE && matches!(self.cx.types.kind(ty), Kind::Slice(_)) { push(out, By::Ref); }
                for &at in &self.body.list(list).to_vec() { self.pat_captures(PatId(at), path, (through_ref, false), known, local, out)?; }
            }
        }
        Ok(())
    }

    fn moves(&mut self, ty: Ty) -> Result<bool> {
        if let Some(captures) = self.own_closure(ty).map(|c| c.captures.clone()) {
            for cap in captures {
                let mut known = true;
                let (_, cap_ty) = self.resolve_path(self.out.locals[cap.local as usize].ty, &cap.path, false, &mut known)?;
                if cap.by == By::Mut || (cap.by == By::Value && cap_ty != Ty::NONE && self.moves(cap_ty)?) { return Ok(true); }
            }
            return Ok(false);
        }
        let reborrows = matches!(self.cx.types.kind(self.resolve(ty)), Kind::Ref(Mut::Mut, _));
        Ok(self.needs_drop(ty)? || (!reborrows && !self.is_copy(ty)?))
    }

    /// The closure of type `ty` when this body defines it and its store entry is not written yet.
    fn own_closure(&self, ty: Ty) -> Option<&Closure> {
        let ty = self.resolve(ty);
        let Kind::Closure(parent, substs, at) = self.cx.types.kind(ty) else { return None };
        let here = self.instance.as_ref().is_some_and(|(item, args)| item == parent && args == substs);
        here.then(|| self.out.closures.get(&ExprId(*at))).flatten().filter(|_| !self.cx.has_closure(ty))
    }

    /// The place expressions, projection parents, non-raw deref parents, and pattern scrutinees of the body.
    fn place_uses(&mut self) -> (HashSet<ExprId>, HashMap<ExprId, ExprId>, HashMap<ExprId, ExprId>, HashMap<ExprId, Vec<PatId>>) {
        let (mut places, mut parents, mut deref_parents, mut scrutinees): (HashSet<ExprId>, HashMap<ExprId, ExprId>, HashMap<ExprId, ExprId>, HashMap<ExprId, Vec<PatId>>) = Default::default();
        let call_once = self.cx.call_once();
        for (at, expr) in self.body.exprs.iter().enumerate() {
            let at = ExprId(at as u32);
            match *expr {
                Expr::Field(base, _) | Expr::TupleField(base, _) | Expr::Paren(base) => { parents.insert(base, at); }
                Expr::Unary(UnOp::Deref, base) => {
                    let base_ty = self.resolve_opt(self.out.exprs[base.0 as usize]);
                    if base_ty != Ty::NONE && !self.out.derefs.contains_key(&(at, 0)) && self.cx.boxed(base_ty).is_some() { parents.insert(base, at); }
                    else { places.insert(base); if base_ty != Ty::NONE && !matches!(self.cx.types.kind(base_ty), Kind::Ptr(..)) { deref_parents.insert(base, at); } }
                }
                Expr::Call(callee, _) if self.consumes_callee(at, callee, call_once) => {}
                Expr::Match(scrut, arms) => { places.insert(scrut); scrutinees.entry(scrut).or_default().extend(self.body.list(arms).iter().map(|&id| self.body.arms[id as usize].pat)); }
                Expr::Let(pat, scrut) => { places.insert(scrut); scrutinees.entry(scrut).or_default().push(pat); }
                Expr::Ref { expr, .. } | Expr::Index(expr, _) | Expr::Assign(expr, _) | Expr::AssignOp(_, expr, _) | Expr::Call(expr, _) => { places.insert(expr); }
                Expr::Binary(op, lhs, rhs) if super::expr::is_compare(op) => { places.insert(lhs); places.insert(rhs); }
                Expr::Method { receiver, .. } if self.out.adjusts.get(&at).is_some_and(|a| a.autoref.is_some() || a.derefs > 0) => { places.insert(receiver); }
                _ => {}
            }
        }
        for stmt in &self.body.stmts { if let Stmt::Let { pat, init: Some(init), .. } = *stmt { scrutinees.entry(init).or_default().push(pat); } }
        (places, parents, deref_parents, scrutinees)
    }

    /// Whether the call `call` of `callee` goes through `FnOnce::call_once` and so moves the callee: a closure that consumes itself, or a value such as a `Box<dyn FnOnce>`.
    fn consumes_callee(&self, call: ExprId, callee: ExprId, call_once: Option<ItemRef>) -> bool {
        let Some(Res::Fn(_method, substs)) = self.out.res.get(&call).filter(|r| matches!(r, Res::Fn(m, _) if Some(*m) == call_once)) else { return false };
        let ty = self.resolve(self.cx.types.substs(*substs).first().copied().unwrap_or(self.out.exprs[callee.0 as usize]));
        match self.cx.types.kind(ty) { Kind::Closure(..) => self.own_closure(ty).map_or_else(|| self.cx.closure_consumes(ty), |c| c.consumes), _ => true }
    }

    /// Notes that the body reads local `local`, which a closure being inferred captures when it lies outside it.
    pub(super) fn note_use(&mut self, local: u32) {
        self.uses.iter_mut().filter(|uses| local < uses.first).for_each(|uses| uses.used.push(local));
    }

    /// Notes that the place expression `expr` is written or borrowed mutably: its overloaded dereferences go through `DerefMut`, and a closure captures its local by unique borrow.
    pub(super) fn note_mut(&mut self, expr: ExprId) -> Result<()> {
        self.deref_mutably(expr)?;
        let Some(local) = self.place_local(expr) else { return Ok(()) };
        self.uses.iter_mut().filter(|uses| local < uses.first).for_each(|uses| uses.mutated.push((local, expr)));
        Ok(())
    }

    fn place_local(&self, expr: ExprId) -> Option<u32> {
        match *self.body.expr(expr) {
            Expr::Paren(inner) | Expr::Field(inner, _) | Expr::TupleField(inner, _) | Expr::Index(inner, _) | Expr::Unary(UnOp::Deref, inner) => {
                let inner_ty = self.resolve_opt(self.out.exprs[inner.0 as usize]);
                (inner_ty == Ty::NONE || !matches!(self.cx.types.kind(inner_ty), Kind::Ptr(..))).then(|| self.place_local(inner)).flatten()
            }
            Expr::Path(_) => match self.out.res.get(&expr) { Some(Res::Local(local)) => Some(*local), _ => None },
            _ => None,
        }
    }

    fn ensure_closure(&mut self, ty: Ty) -> Result<Option<(u32, Option<Arc<ClosureInfo>>)>> {
        let Kind::Closure(parent, substs, at) = *self.cx.types.kind(ty) else { return Ok(None) };
        let local = self.instance == Some((parent, substs));
        if !local && !self.cx.has_closure(ty) { self.cx.ensure_closure(ty); }
        Ok(Some((at, (!local).then(|| self.cx.closure(ty)).flatten())))
    }

    /// The parameter and result types of the closure type `ty`, from this body when it defines the closure, and whether it captures anything.
    pub(super) fn closure_sig(&mut self, ty: Ty) -> Result<(Vec<Ty>, Ty, bool)> {
        let Some((at, info)) = self.ensure_closure(ty)? else { bail!("not a closure") };
        let (closure, locals) = match &info { Some(i) => (Some(&i.closure), &i.typed.locals), None => (self.out.closures.get(&ExprId(at)), &self.out.locals) };
        let Some(closure) = closure else { bail!("a closure whose body is not inferred yet") };
        Ok((closure.params.iter().map(|&local| locals[local as usize].ty).collect(), closure.ret, !closure.captures.is_empty()))
    }

    pub(super) fn closure_mutates(&mut self, ty: Ty) -> Result<bool> {
        let Some((at, info)) = self.ensure_closure(ty)? else { return Ok(false) };
        let closure = info.as_ref().map(|info| &info.closure).or_else(|| self.out.closures.get(&ExprId(at)));
        Ok(closure.is_some_and(|closure| closure.captures.iter().any(|capture| capture.by == By::Mut)))
    }

    /// Infers the call `expr` of the closure value of type `ty`, recording it as a call of `FnOnce::call_once` at the closure and its argument tuple.
    pub(super) fn closure_call(&mut self, expr: ExprId, ty: Ty, args: List) -> Result<Ty> {
        let (params, ret, _) = self.closure_sig(ty)?;
        self.call_once_of(expr, ty, (params, ret), args)
    }

    /// Infers the call `expr` of the closure or pending `impl Fn` value `ty` with signature `(params, ret)` as `FnOnce::call_once` of it.
    pub(super) fn call_once_of(&mut self, expr: ExprId, ty: Ty, (params, ret): (Vec<Ty>, Ty), args: List) -> Result<Ty> {
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
                let output = if self.is_async_callable_trait(of) { self.fresh(Kind::Var) } else { output.unwrap_or_else(|| self.cx.types.unit()) };
                Some((self.cx.types.substs(inputs).to_vec(), output))
            }
            _ => None,
        }
    }

    /// Infers and coerces the argument `arg` of a call of `callee` to its parameter type `param`, giving a closure argument the signature of the parameter's `Fn` bound.
    pub(super) fn call_arg(&mut self, callee: Option<(ItemRef, &[Ty])>, arg: ExprId, (param, expectation): (Ty, Option<Ty>), held: &mut Option<(ItemRef, Vec<Ty>)>) -> Result<()> {
        let callable = match *self.cx.types.kind(self.resolve(param)) { Kind::Ref(_, pointee) => pointee, _ => param };
        let (hint, bound) = match callee { Some(c) => (self.closure_hint(c, arg, param)?, self.callable_hint(c, callable)?), None => (None, None) };
        let closure = self.is_closure(arg);
        let propagate = closure || self.holds_closure(arg) || matches!(self.body.expr(self.wrapped(arg)), Expr::Array(_));
        let usable = expectation.filter(|&wanted| propagate || self.sized_expectation(wanted));
        let found = self.expr(arg, Some(hint.or(usable).unwrap_or(param)))?;
        self.release_if_fits(held, found, param)?;
        self.coerce_arg((arg, found, closure), param, expectation)?;
        match bound { Some((of, bound)) => self.tie_callable_bound(of, callable, bound), None => Ok(()) }
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
            Some(wanted) if self.coerces_to(found, wanted) => { self.coerce_expr(arg, found, wanted)?; self.unify(param, wanted) }
            _ => self.coerce_expr(arg, found, param),
        }
    }

    /// Whether `found` coerces to `wanted` by a coercion or by giving up the `mut` of a reference.
    pub(super) fn coerces_to(&mut self, found: Ty, wanted: Ty) -> bool {
        let shared = self.weakened(found, wanted);
        self.probe(|infer| matches!(infer.coercion(found, wanted), Ok(Some(_))) || shared.is_some_and(|(a, b)| infer.unifies(a, b)))
    }

    /// The parameters and return type of the closure, function item or function pointer `ty`, or the one a reference to it calls.
    pub(super) fn callable_sig(&mut self, ty: Ty) -> Result<Option<(Vec<Ty>, Ty)>> {
        let ty = self.resolve(ty);
        Ok(Some(match *self.cx.types.kind(ty) {
            Kind::Ref(_, inner) => return self.callable_sig(inner),
            Kind::Closure(..) => { let (params, ret, _) = self.closure_sig(ty)?; (params, ret) }
            Kind::FnDef(item, substs) => { let args = self.cx.types.substs(substs).to_vec(); self.fn_types(item, &args)? }
            Kind::FnPtr(params, ret) => (self.cx.types.substs(params).to_vec(), ret),
            Kind::Param(_) | Kind::Proj(_) => { let Some(sig) = self.poly_fn_sig(ty) else { return Ok(None) }; sig }
            Kind::Var(_) => match self.opaque_sig(ty).map(|sig| self.cx.types.kind(sig).clone()) {
                Some(Kind::FnPtr(params, ret)) => (self.cx.types.substs(params).to_vec(), ret),
                _ => return Ok(None),
            },
            _ => return Ok(None),
        }))
    }

    pub(super) fn is_closure(&self, arg: ExprId) -> bool { matches!(*self.body.expr(self.wrapped(arg)), Expr::Closure { .. }) }

    fn wrapped(&self, expr: ExprId) -> ExprId {
        match *self.body.expr(expr) {
            Expr::Paren(inner) => self.wrapped(inner),
            Expr::Block { tail: Some(tail), label: None, kind: BlockKind::Plain | BlockKind::Expansion | BlockKind::Unsafe, .. } => self.wrapped(tail),
            _ => expr,
        }
    }

    fn closure_hint(&mut self, callee: (ItemRef, &[Ty]), arg: ExprId, param: Ty) -> Result<Option<Ty>> {
        match *self.body.expr(self.wrapped(arg)) {
            Expr::Closure { .. } => Ok(self.callable_hint(callee, param)?.map(|(of, sig)| self.closure_hint_sig(of, sig))),
            Expr::Ref { raw: false, expr: inner, .. } => {
                let Kind::Ref(mutability, pointee) = *self.cx.types.kind(self.resolve(param)) else { return Ok(None) };
                Ok(self.closure_hint(callee, inner, pointee)?.map(|hint| self.cx.types.intern(Kind::Ref(mutability, hint))))
            }
            _ => Ok(None),
        }
    }

    fn callable_hint(&mut self, (item, substs): (ItemRef, &[Ty]), param: Ty) -> Result<Option<(ItemRef, Ty)>> {
        let param = self.resolve(param);
        if !matches!(self.cx.types.kind(param), Kind::Var(_)) { return Ok(None); }
        if matches!(self.cx.item(item).1.kind, ItemKind::Struct | ItemKind::Enum | ItemKind::Union) { return self.field_hint((item, substs), param); }
        let frame = self.frame(item, substs)?;
        let Some(at) = frame.generics.iter().position(|(_, ty)| self.resolve(*ty) == param) else { return Ok(None) };
        let sig = self.cx.signature(item)?;
        let own: Vec<_> = super::scope::type_params(&sig.generics).collect();
        let (home, name) = (frame.home, frame.generics[at].0);
        let Some(index) = at.checked_sub(frame.generics.len() - own.len()) else { return self.with_frame(frame, |infer| infer.header_callable_bound(name, home)); };
        let bounds = super::scope::generic_bounds(&sig.body, &self.cx.file(item).source, &sig.predicates, own[index]);
        self.first_callable_bound(frame, &sig.body, bounds)
    }

    /// The callable trait and function pointer type that a callable bound declared on the struct, enum or union `item` spells for its type parameter that `param` fills in `args`.
    fn field_hint(&mut self, (item, args): (ItemRef, &[Ty]), param: Ty) -> Result<Option<(ItemRef, Ty)>> {
        let def = self.cx.adt_def(item)?;
        let Some(at) = args.iter().position(|&arg| self.resolve(arg) == param) else { return Ok(None) };
        let Some(generic) = super::scope::type_params(&def.generics.params).nth(at) else { return Ok(None) };
        let bounds = super::scope::generic_bounds(&def.body, &self.cx.file(item).source, &def.generics.predicates, generic);
        if bounds.is_empty() { return Ok(None); }
        let frame = self.adt_frame(item, &def.generics.params, args)?;
        self.first_callable_bound(frame, &def.body, bounds)
    }
}

/// Whether `item` is `Fn::call`, `AsyncFn::async_call`, `FnMut::call_mut` or `AsyncFnMut::async_call_mut`, and how each takes the closure.
pub fn by_reference(cx: &Context<'_>, item: ItemRef) -> Option<Mut> {
    match cx.owner_trait(item).and_then(|of| cx.lang(of)) {
        Some("fn" | "async_fn") => Some(Mut::Not),
        Some("fn_mut" | "async_fn_mut") => Some(Mut::Mut),
        _ => None,
    }
}

/// The typed form of a call trait method at a closure, function item or function pointer and its argument tuple, which calls the closure's body, the function or the pointer that the receiver `Res::Local(0)` points at, as a vtable passes it, with the tuple's fields.
pub fn untuple_typed(cx: &mut Context<'_>, item: ItemRef, substs: Substs) -> Result<Typed> {
    let slice = cx.types.substs(substs);
    let (Some(&callee), Some(&tuple)) = (slice.first(), slice.get(1)) else { bail!("a call trait method with no callee and arguments") };
    let (target, ret) = match *cx.types.kind(callee) {
        Kind::FnPtr(_, ret) => (Res::Local(0), ret),
        Kind::FnDef(target, args) => {
            let arg_slice = cx.types.substs(args).to_vec();
            let (res, (_, ret)) = super::detached(cx, target, |infer| Ok((infer.trait_call(target, args)?, infer.fn_types(target, &arg_slice)?)))?;
            (res, ret)
        }
        _ => {
            let info = cx.ensure_closure_res(callee, false)?;
            let Some(call_once) = cx.call_once() else { bail!("core has no `FnOnce::call_once`") };
            (Res::fn_of(&cx.types, call_once, &[callee, tuple]), info.closure.ret)
        }
    };
    let pointer = matches!(cx.types.kind(callee), Kind::FnPtr(..)).then_some(Mut::Not);
    let receiver = by_reference(cx, item).or(pointer).map_or(callee, |by| cx.types.intern(Kind::Ref(by, callee)));
    Ok(Typed { locals: vec![Local { ty: receiver }, Local { ty: tuple }], params: vec![0, 1], ret: Some(ret), shape: Shape::Untuple(target), ..Typed::default() })
}

/// Records the closures of the finished body `typed` of `parent` at `substs` in `cx`, so other bodies can call them.
pub fn register(cx: &mut Context<'_>, (parent, substs): (ItemRef, Substs), typed: &Typed) {
    if !typed.has_closures() { return; }
    let typed = typed.scoped(&cx.types);
    typed.ensure_tables(&cx.types);
    let shared = Arc::new(Typed { drops: Vec::new(), ..(*typed).clone() });
    for (at, closure) in &typed.closures {
        let ty = cx.types.intern(Kind::Closure(parent, substs, at.0));
        let parts: Vec<Ty> = closure.captures.iter().map(|c| {
            let held = typed.locals[c.local as usize].ty;
            let held = c.ty(cx, held).unwrap_or(held);
            match c.by { By::Ref => cx.types.intern(Kind::Ref(Mut::Not, held)), By::Mut => cx.types.intern(Kind::Ref(Mut::Mut, held)), By::Value => held }
        }).collect();
        let mut env = cx.types.tuple(&parts);
        if closure.coroutine {
            let word = cx.types.prim(crate::names::Primitive::Usize);
            env = cx.types.tuple(&[word, word, env]);
        }
        cx.set_closure(ty, ClosureInfo { parent, typed: shared.clone(), closure: closure.clone(), env });
    }
}

fn finish_closure(cx: &mut Context<'_>, mut typed: Typed, info: &ClosureInfo, env: Option<u32>, ret: Ty) -> Result<Typed> {
    if env.is_some() {
        for cap in &info.closure.captures {
            if !cap.path.is_empty() { typed.locals.push(Local { ty: cap.ty(cx, typed.locals[cap.local as usize].ty)? }); }
        }
    }
    typed.ret = Some(ret);
    typed.shape = Shape::Closure(ClosureBody { parent: info.parent, closure: info.closure.clone(), env });
    typed.drops = super::drop::drop_calls(cx, &typed)?;
    Ok(typed)
}

/// The typed form of `Future::poll` at the coroutine type `ty`: its defining body, resumed with the pinned coroutine and the task context, returning `Poll` of its output.
pub fn coroutine_typed(cx: &mut Context<'_>, poll: ItemRef, ty: Ty) -> Result<Typed> {
    let info = cx.ensure_closure_res(ty, true)?;
    let (params, ret) = super::detached(cx, poll, |infer| infer.fn_types(poll, &[ty]))?;
    let mut typed = (*info.typed).clone();
    let first = typed.locals.len() as u32;
    typed.locals.extend(params.iter().map(|&ty| Local { ty }));
    typed.params = vec![first, first + 1];
    finish_closure(cx, typed, &info, Some(first), ret)
}

/// The typed form of the function that the closure type `ty` runs: its defining body, with an environment parameter and the closure's own parameters.
pub fn closure_typed(cx: &mut Context<'_>, ty: Ty) -> Result<Typed> {
    let info = cx.ensure_closure_res(ty, true)?;
    let mut typed = (*info.typed).clone();
    let env = cx.closure_takes_env(ty).then(|| { typed.locals.push(Local { ty: cx.types.intern(Kind::Ref(Mut::Mut, info.env)) }); typed.locals.len() as u32 - 1 });
    typed.params = env.into_iter().chain(info.closure.params.iter().copied()).collect();
    finish_closure(cx, typed, &info, env, info.closure.ret)
}

impl<'a> Context<'a> {
    /// Returns the recorded closure info for `ty`, lazily inferring its defining body once when needed.
    pub fn ensure_closure(&mut self, ty: Ty) -> Option<Arc<ClosureInfo>> { self.ensure_closure_res(ty, false).ok() }

    pub(in crate::sema) fn ensure_closure_env(&mut self, ty: Ty) -> Result<Ty> {
        if let Some(info) = self.closure_ready(ty, false) { return Ok(info.env); }
        if let Some(info) = self.load_closure_owner(ty, false)? { return Ok(info.env); }
        self.async_coroutine_env(ty)?.ok_or_else(|| anyhow::anyhow!("layout of a closure whose defining body was not inferred"))
    }

    fn closure_ready(&self, ty: Ty, full: bool) -> Option<Arc<ClosureInfo>> { self.store.closures.get(&ty).filter(|info| !full || !info.typed.exprs.is_empty()) }
    fn ensure_closure_res(&mut self, ty: Ty, full: bool) -> Result<Arc<ClosureInfo>> {
        self.closure_ready(ty, full).map_or_else(|| self.load_closure_owner(ty, full)?.ok_or_else(|| anyhow::anyhow!("a closure whose defining body was not inferred")), Ok)
    }

    fn load_closure_owner(&mut self, ty: Ty, full: bool) -> Result<Option<Arc<ClosureInfo>>> {
        let Kind::Closure(parent, substs, _) = *self.types.kind(ty) else { bail!("not a closure") };
        if let Some(info) = self.register_final_closure((parent, substs), ty, full) { return Ok(Some(info)); }
        if crate::sema::reach::is_inferring((parent, substs)) { return Ok(None); }
        let mut cycled = false;
        let reveals = self.store.lookups.reveals.clone();
        reveals.run(self, (parent, substs), |cx| Ok(crate::sema::reach::typed_once(cx, &(parent, substs), &mut Vec::new()).map_err(|m| anyhow::anyhow!("{m}"))?.ret.unwrap_or(Ty::NONE)), |_| { cycled = true; Ok(Ty::NONE) })?;
        if let Some(info) = self.closure_ready(ty, full) { return Ok(Some(info)); }
        if cycled { return Ok(None); }
        if let Some(info) = self.register_final_closure((parent, substs), ty, full) { return Ok(Some(info)); }
        let typed = crate::sema::reach::typed_once(self, &(parent, substs), &mut Vec::new()).map_err(|m| anyhow::anyhow!("{m}"))?;
        register(self, (parent, substs), &typed);
        Ok(Some(self.closure_ready(ty, full).ok_or_else(|| anyhow::anyhow!("closure owner `{}` did not record closure", self.name(parent)))?))
    }

    fn register_final_closure(&mut self, instance: (ItemRef, Substs), ty: Ty, full: bool) -> Option<Arc<ClosureInfo>> {
        let typed = self.store.finals.get(&instance)?;
        register(self, instance, &typed);
        self.closure_ready(ty, full)
    }

    fn async_coroutine_env(&mut self, ty: Ty) -> Result<Option<Ty>> {
        let Kind::Closure(parent, substs, at) = *self.types.kind(ty) else { return Ok(None) };
        let sig = self.signature(parent)?;
        if !sig.is_async || self.body(parent)?.root.is_none_or(|root| root.0 != at) { return Ok(None); }
        if sig.params.iter().any(|&(pat, _)| !matches!(*sig.body.pat(pat), crate::body::Pat::Bind { sub: None, by_ref: false, .. } | crate::body::Pat::Wild)) { return Ok(None); }
        let slice = self.types.substs(substs).to_vec();
        let params = super::detached(self, parent, |infer| {
            let frame = infer.frame(parent, &slice)?;
            let (params, _) = infer.with_frame(frame, |infer| infer.sig_types(&sig))?;
            infer.settle_projections()?;
            Ok(params.into_iter().map(|p| infer.zonk(p)).collect::<Vec<_>>())
        })?;
        let captures = self.types.tuple(&params);
        let word = self.types.prim(crate::names::Primitive::Usize);
        Ok(Some(self.types.tuple(&[word, word, captures])))
    }
}

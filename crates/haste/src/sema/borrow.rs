//! Ownership, mutability, and NLL-style borrow checking for local workspace function bodies.

use anyhow::{Result, bail};
use rustc_hash::FxHashSet as HashSet;

use crate::body::{Body, Expr, ExprId, List, Pat, PatId, Stmt, UnOp};
use crate::names::ItemRef;
use crate::outline::source::Source;

use super::Context;
use super::infer::{Res, Typed};
use super::ty::{Kind, Mut};

#[derive(Clone, Debug)]
struct Loan {
    target: u32,
    holder: Option<u32>,
    mutable: bool,
}

#[derive(Clone, Debug)]
struct State {
    init: Vec<bool>,
    moved: Vec<bool>,
    loans: Vec<Loan>,
    borrows_local: Vec<bool>,
    diverged: bool,
}

struct Checker<'c, 'a> {
    types: &'a super::ty::Types,
    body: &'c Body,
    typed: &'c Typed,
    mutable: Vec<bool>,
    is_copy: Vec<bool>,
    is_param: Vec<bool>,
    names: Vec<Option<String>>,
    deref_non_copy: HashSet<ExprId>,
    returns_ref: bool,
}

/// Checks ownership, mutability, borrow aliasing, and dangling local reference returns on `body`.
pub(super) fn check(cx: &mut Context<'_>, item: ItemRef, (body, sig): (&Body, &crate::body::sig::Signature), src: &Source, typed: &Typed) -> Result<()> {
    let Some(root) = body.root else { return Ok(()) };
    if typed.locals.is_empty() {
        return Ok(());
    }
    let n = typed.locals.len();
    let mut is_copy = vec![true; n];
    for (idx, local) in typed.locals.iter().enumerate() {
        if local.ty != super::ty::Ty::NONE {
            is_copy[idx] = matches!(cx.types.kind(local.ty), Kind::Ref(..)) || super::infer::is_copy(cx, item, local.ty).unwrap_or(true);
        }
    }
    let mut mutable = vec![false; n];
    let mut is_param = vec![false; n];
    let mut names = vec![None; n];
    let mut init = vec![false; n];
    for &p in &typed.params {
        if let Some(slot) = is_param.get_mut(p as usize) {
            *slot = true;
            init[p as usize] = true;
        }
    }
    let mut param_cursor = 0usize;
    if let Some((_, recv_mut)) = sig.receiver {
        if let Some(&local) = typed.params.get(param_cursor)
            && (local as usize) < mutable.len()
        {
            mutable[local as usize] = recv_mut;
            names[local as usize] = Some("self".to_owned());
        }
        param_cursor += 1;
    }
    for &(pat, _) in &sig.params {
        if let Pat::Bind { name, mutable: is_mut, .. } = *sig.body.pat(pat)
            && let Some(&local) = typed.params.get(param_cursor)
            && (local as usize) < mutable.len()
        {
            if is_mut {
                mutable[local as usize] = true;
            }
            if names[local as usize].is_none() {
                names[local as usize] = Some(src.text(name).to_owned());
            }
        }
        param_cursor += 1;
    }
    for &pat in &body.fn_params {
        collect_pat_meta(body, typed, src, pat, &mut mutable, &mut names);
    }
    for at in 0..body.pats.len() as u32 {
        collect_pat_meta(body, typed, src, PatId(at), &mut mutable, &mut names);
    }
    let mut deref_non_copy = HashSet::default();
    for at in 0..body.exprs.len() as u32 {
        let id = ExprId(at);
        if let Expr::Unary(UnOp::Deref, inner) = *body.expr(id)
            && let Some(&inner_ty) = typed.exprs.get(inner.0 as usize)
            && inner_ty != super::ty::Ty::NONE
            && let Kind::Ref(_, pointee) = cx.types.kind(inner_ty)
            && !super::infer::is_copy(cx, item, *pointee).unwrap_or(true)
        {
            deref_non_copy.insert(id);
        }
    }
    let returns_ref = typed.ret.is_some_and(|ty| matches!(cx.types.kind(ty), Kind::Ref(..)));
    let checker = Checker { types: &cx.types, body, typed, mutable, is_copy, is_param, names, deref_non_copy, returns_ref };
    let mut state = State { init, moved: vec![false; n], loans: Vec::new(), borrows_local: vec![false; n], diverged: false };
    checker.check_expr(root, &mut state, false)?;
    if checker.returns_ref && !state.diverged {
        checker.check_return_escape(root, &state)?;
    }
    Ok(())
}

fn collect_pat_meta(body: &Body, typed: &Typed, src: &Source, pat: PatId, mutable: &mut [bool], names: &mut [Option<String>]) {
    if let Pat::Bind { name, mutable: is_mut, .. } = *body.pat(pat)
        && let Some(&local) = typed.binds.get(&pat)
        && (local as usize) < mutable.len()
    {
        if is_mut {
            mutable[local as usize] = true;
        }
        if names[local as usize].is_none() {
            names[local as usize] = Some(src.text(name).to_owned());
        }
    }
}

impl Checker<'_, '_> {
    fn local_name(&self, local: u32) -> String {
        self.names.get(local as usize).and_then(|n| n.clone()).unwrap_or_else(|| format!("_{local}"))
    }

    fn root_local(&self, expr: ExprId) -> Option<(u32, bool)> {
        match *self.body.expr(expr) {
            Expr::Paren(inner) => self.root_local(inner),
            Expr::Path(_) => match self.typed.res.get(&expr) {
                Some(Res::Local(local)) => Some((*local, false)),
                _ => None,
            },
            Expr::Field(base, _) | Expr::TupleField(base, _) => {
                let (local, through_ref) = self.root_local(base)?;
                let deref = matches!(self.typed.exprs.get(base.0 as usize).copied(), Some(ty) if ty != super::ty::Ty::NONE && self.is_ref_or_ptr_expr(base));
                Some((local, through_ref || deref))
            }
            Expr::Index(base, _) => {
                let (local, _) = self.root_local(base)?;
                Some((local, true))
            }
            Expr::Unary(UnOp::Deref, inner) => self.root_local(inner).map(|(l, _)| (l, true)),
            _ => None,
        }
    }

    fn place_behind_shared_ref(&self, expr: ExprId) -> bool {
        match *self.body.expr(expr) {
            Expr::Paren(inner) => self.place_behind_shared_ref(inner),
            Expr::Unary(UnOp::Deref, inner) => {
                self.typed
                    .exprs
                    .get(inner.0 as usize)
                    .copied()
                    .is_some_and(|ty| ty != super::ty::Ty::NONE && matches!(self.types.kind(ty), Kind::Ref(Mut::Not, _)))
            }
            Expr::Field(base, _) | Expr::TupleField(base, _) | Expr::Index(base, _) => {
                let base_shared = self
                    .typed
                    .exprs
                    .get(base.0 as usize)
                    .copied()
                    .is_some_and(|ty| ty != super::ty::Ty::NONE && matches!(self.types.kind(ty), Kind::Ref(Mut::Not, _)));
                base_shared || self.place_behind_shared_ref(base)
            }
            _ => false,
        }
    }

    fn is_ref_or_ptr_expr(&self, expr: ExprId) -> bool {
        if self.typed.autoderefs.get(&expr).is_some_and(|&steps| steps > 0) {
            return true;
        }
        self.typed
            .exprs
            .get(expr.0 as usize)
            .copied()
            .is_some_and(|ty| ty != super::ty::Ty::NONE && matches!(self.types.kind(ty), Kind::Ref(..) | Kind::Ptr(..)))
    }

    fn check_not_moved(&self, local: u32, state: &State) -> Result<()> {
        if state.moved.get(local as usize).copied().unwrap_or(false) {
            bail!("use of moved value `{}`", self.local_name(local));
        }
        Ok(())
    }

    fn check_can_read(&self, local: u32, state: &State) -> Result<()> {
        self.check_not_moved(local, state)?;
        if state.loans.iter().any(|l| l.target == local && l.mutable) {
            bail!("cannot use `{}` because it was mutably borrowed", self.local_name(local));
        }
        Ok(())
    }

    fn check_can_move(&self, local: u32, state: &State) -> Result<()> {
        self.check_not_moved(local, state)?;
        if state.loans.iter().any(|l| l.target == local) {
            bail!("cannot move out of `{}` because it is borrowed", self.local_name(local));
        }
        Ok(())
    }

    fn check_can_mutate(&self, local: u32, through_ref: bool, state: &State, init_ok: bool) -> Result<()> {
        if !through_ref && !self.mutable.get(local as usize).copied().unwrap_or(true) && !(init_ok && !state.init.get(local as usize).copied().unwrap_or(true)) {
            bail!("cannot assign to immutable variable `{}`", self.local_name(local));
        }
        if !through_ref && state.loans.iter().any(|l| l.target == local) {
            bail!("cannot assign to `{}` because it is borrowed", self.local_name(local));
        }
        Ok(())
    }

    fn check_borrow(&self, local: u32, through_ref: bool, mutable: bool, state: &State) -> Result<()> {
        self.check_not_moved(local, state)?;
        if mutable && !through_ref && !self.mutable.get(local as usize).copied().unwrap_or(true) {
            bail!("cannot borrow `{}` as mutable, as it is not declared as mutable", self.local_name(local));
        }
        if !through_ref {
            if mutable && state.loans.iter().any(|l| l.target == local && l.mutable) {
                bail!("cannot borrow `{}` as mutable more than once at a time", self.local_name(local));
            }
            if mutable && state.loans.iter().any(|l| l.target == local) {
                bail!("cannot borrow `{}` as mutable because it is also borrowed as immutable", self.local_name(local));
            }
            if !mutable && state.loans.iter().any(|l| l.target == local && l.mutable) {
                bail!("cannot borrow `{}` as immutable because it is also borrowed as mutable", self.local_name(local));
            }
        }
        Ok(())
    }

    fn expr_borrow_source(&self, expr: ExprId, state: &State) -> Option<(u32, bool)> {
        match *self.body.expr(expr) {
            Expr::Paren(inner) => self.expr_borrow_source(inner, state),
            Expr::Ref { mutable, raw: false, expr: inner } => {
                let (local, through_ref) = self.root_local(inner)?;
                (!through_ref).then_some((local, mutable))
            }
            Expr::Path(_) => {
                let Some(Res::Local(local)) = self.typed.res.get(&expr) else { return None };
                state.loans.iter().find(|l| l.holder == Some(*local)).map(|l| (l.target, l.mutable))
            }
            _ => None,
        }
    }

    fn check_return_escape(&self, expr: ExprId, state: &State) -> Result<()> {
        match *self.body.expr(expr) {
            Expr::Paren(inner) | Expr::Block { tail: Some(inner), .. } => self.check_return_escape(inner, state),
            Expr::Ref { raw: false, expr: inner, .. } => {
                if let Some((local, false)) = self.root_local(inner)
                    && !self.is_param.get(local as usize).copied().unwrap_or(true)
                {
                    bail!("cannot return reference to local variable `{}`", self.local_name(local));
                }
                Ok(())
            }
            Expr::Path(_) => {
                if let Some(Res::Local(local)) = self.typed.res.get(&expr)
                    && state.borrows_local.get(*local as usize).copied().unwrap_or(false)
                {
                    bail!("cannot return value referencing local variable via `{}`", self.local_name(*local));
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn bind_pat(&self, pat: PatId, state: &mut State, borrow_src: Option<(u32, bool)>) {
        match *self.body.pat(pat) {
            Pat::Bind { sub, .. } => {
                if let Some(&local) = self.typed.binds.get(&pat)
                    && (local as usize) < state.init.len()
                {
                    state.init[local as usize] = true;
                    state.moved[local as usize] = false;
                    state.loans.retain(|l| l.holder != Some(local));
                    if let Some((target, mutable)) = borrow_src {
                        state.loans.push(Loan { target, holder: Some(local), mutable });
                        state.borrows_local[local as usize] = !self.is_param.get(target as usize).copied().unwrap_or(true);
                    } else {
                        state.borrows_local[local as usize] = false;
                    }
                }
                if let Some(sub) = sub {
                    self.bind_pat(sub, state, None);
                }
            }
            Pat::Paren(inner) | Pat::Ref { pat: inner, .. } | Pat::Box(inner) | Pat::Deref(inner) => self.bind_pat(inner, state, None),
            Pat::Tuple(list) | Pat::TupleStruct(_, list) | Pat::Slice(list) | Pat::Or(list) => {
                for &at in self.body.list(list) {
                    self.bind_pat(PatId(at), state, None);
                }
            }
            Pat::Struct { fields, .. } => {
                for &at in self.body.list(fields) {
                    self.bind_pat(self.body.field_pats[at as usize].pat, state, None);
                }
            }
            _ => {}
        }
    }

    fn collect_reads_stmt(&self, stmt: Stmt, out: &mut HashSet<u32>) {
        match stmt {
            Stmt::Let { init, otherwise, .. } => {
                if let Some(e) = init {
                    self.collect_reads_expr(e, out);
                }
                if let Some(e) = otherwise {
                    self.collect_reads_expr(e, out);
                }
            }
            Stmt::Expr(e, _) => self.collect_reads_expr(e, out),
            Stmt::Item(..) | Stmt::Macro { .. } | Stmt::Empty => {}
        }
    }

    fn collect_reads_expr(&self, expr: ExprId, out: &mut HashSet<u32>) {
        match *self.body.expr(expr) {
            Expr::Path(_) => {
                if let Some(Res::Local(local)) = self.typed.res.get(&expr) {
                    out.insert(*local);
                }
            }
            Expr::Unary(_, e) | Expr::Field(e, _) | Expr::TupleField(e, _) | Expr::Cast(e, _) | Expr::Ref { expr: e, .. } | Expr::Try(e) | Expr::Await(e) | Expr::Paren(e) | Expr::Become(e) => {
                self.collect_reads_expr(e, out);
            }
            Expr::Binary(_, a, b) | Expr::Assign(a, b) | Expr::AssignOp(_, a, b) | Expr::Index(a, b) | Expr::Repeat(a, b) => {
                self.collect_reads_expr(a, out);
                self.collect_reads_expr(b, out);
            }
            Expr::Call(callee, args) => {
                self.collect_reads_expr(callee, out);
                self.body.list(args).iter().for_each(|&a| self.collect_reads_expr(ExprId(a), out));
            }
            Expr::Method { receiver, params, .. } => {
                self.collect_reads_expr(receiver, out);
                self.body.list(params).iter().for_each(|&a| self.collect_reads_expr(ExprId(a), out));
            }
            Expr::Block { stmts, tail, .. } => {
                for &s in self.body.list(stmts) {
                    self.collect_reads_stmt(self.body.stmts[s as usize], out);
                }
                if let Some(t) = tail {
                    self.collect_reads_expr(t, out);
                }
            }
            Expr::If(c, t, e) => {
                self.collect_reads_expr(c, out);
                self.collect_reads_expr(t, out);
                if let Some(e) = e {
                    self.collect_reads_expr(e, out);
                }
            }
            Expr::Let(_, init) => self.collect_reads_expr(init, out),
            Expr::While { cond, body, .. } => {
                self.collect_reads_expr(cond, out);
                self.collect_reads_expr(body, out);
            }
            Expr::Loop { body, .. } | Expr::Closure { body, .. } => self.collect_reads_expr(body, out),
            Expr::For { iter, body, .. } => {
                self.collect_reads_expr(iter, out);
                self.collect_reads_expr(body, out);
            }
            Expr::Match(scrut, arms) => {
                self.collect_reads_expr(scrut, out);
                for &a in self.body.list(arms) {
                    let arm = self.body.arms[a as usize];
                    if let Some(g) = arm.guard {
                        self.collect_reads_expr(g, out);
                    }
                    self.collect_reads_expr(arm.body, out);
                }
            }
            Expr::Tuple(list) | Expr::Array(list) => self.body.list(list).iter().for_each(|&a| self.collect_reads_expr(ExprId(a), out)),
            Expr::Struct { fields, base, .. } => {
                self.body.list(fields).iter().for_each(|&f| self.collect_reads_expr(self.body.fields[f as usize].value, out));
                if let Some(b) = base {
                    self.collect_reads_expr(b, out);
                }
            }
            Expr::Range { from, to, .. } => {
                if let Some(f) = from {
                    self.collect_reads_expr(f, out);
                }
                if let Some(t) = to {
                    self.collect_reads_expr(t, out);
                }
            }
            Expr::Return(e) | Expr::Break(_, e) | Expr::Yield(e) => {
                if let Some(e) = e {
                    self.collect_reads_expr(e, out);
                }
            }
            _ => {}
        }
    }

    fn check_expr(&self, expr: ExprId, state: &mut State, place_ctx: bool) -> Result<()> {
        if state.diverged {
            return Ok(());
        }
        match *self.body.expr(expr) {
            Expr::Path(_) => {
                if let Some(Res::Local(local)) = self.typed.res.get(&expr)
                    && (*local as usize) < self.is_copy.len()
                {
                    if place_ctx || self.is_copy[*local as usize] {
                        self.check_can_read(*local, state)?;
                    } else {
                        self.check_can_move(*local, state)?;
                        state.moved[*local as usize] = true;
                    }
                }
            }
            Expr::Ref { mutable, raw, expr: inner } => {
                if !raw && mutable && self.place_behind_shared_ref(inner) {
                    bail!("cannot borrow data in a `&` reference as mutable");
                }
                if !raw && let Some((local, through_ref)) = self.root_local(inner) {
                    self.check_borrow(local, through_ref, mutable, state)?;
                }
                self.check_expr(inner, state, true)?;
            }
            Expr::Assign(lhs, rhs) => {
                self.check_expr(rhs, state, false)?;
                if self.place_behind_shared_ref(lhs) {
                    bail!("cannot assign to data in a `&` reference");
                }
                let borrow_src = self.expr_borrow_source(rhs, state);
                if let Some((local, through_ref)) = self.root_local(lhs) {
                    self.check_can_mutate(local, through_ref, state, true)?;
                    if !through_ref && (local as usize) < state.init.len() {
                        state.init[local as usize] = true;
                        state.moved[local as usize] = false;
                        state.loans.retain(|l| l.holder != Some(local));
                        if let Some((target, mutable)) = borrow_src {
                            state.loans.push(Loan { target, holder: Some(local), mutable });
                            state.borrows_local[local as usize] = !self.is_param.get(target as usize).copied().unwrap_or(true);
                        }
                    }
                } else {
                    self.check_expr(lhs, state, true)?;
                }
            }
            Expr::AssignOp(_, lhs, rhs) => {
                self.check_expr(rhs, state, false)?;
                if self.place_behind_shared_ref(lhs) {
                    bail!("cannot assign to data in a `&` reference");
                }
                if let Some((local, through_ref)) = self.root_local(lhs) {
                    self.check_not_moved(local, state)?;
                    self.check_can_mutate(local, through_ref, state, false)?;
                } else {
                    self.check_expr(lhs, state, true)?;
                }
            }
            Expr::Block { stmts, tail, .. } => {
                let ids: Vec<u32> = self.body.list(stmts).to_vec();
                for (pos, &stmt_id) in ids.iter().enumerate() {
                    if !state.loans.is_empty() {
                        let mut live = HashSet::default();
                        for &later in &ids[pos..] {
                            self.collect_reads_stmt(self.body.stmts[later as usize], &mut live);
                        }
                        if let Some(t) = tail {
                            self.collect_reads_expr(t, &mut live);
                        }
                        state.loans.retain(|l| l.holder.is_none_or(|h| live.contains(&h)));
                    }
                    match self.body.stmts[stmt_id as usize] {
                        Stmt::Let { pat, init, otherwise, .. } => {
                            let destructures = !matches!(*self.body.pat(pat), Pat::Bind { by_ref: false, sub: None, .. } | Pat::Wild);
                            let borrow_src = if let Some(e) = init {
                                self.check_expr(e, state, destructures)?;
                                self.expr_borrow_source(e, state)
                            } else {
                                None
                            };
                            if let Some(e) = otherwise {
                                let mut alt = state.clone();
                                self.check_expr(e, &mut alt, false)?;
                            }
                            if init.is_some() {
                                self.bind_pat(pat, state, borrow_src);
                            }
                        }
                        Stmt::Expr(e, _) => self.check_expr(e, state, false)?,
                        Stmt::Item(..) | Stmt::Macro { .. } | Stmt::Empty => {}
                    }
                }
                if let Some(t) = tail {
                    if !state.loans.is_empty() {
                        let mut live = HashSet::default();
                        self.collect_reads_expr(t, &mut live);
                        state.loans.retain(|l| l.holder.is_none_or(|h| live.contains(&h)));
                    }
                    self.check_expr(t, state, place_ctx)?;
                }
            }
            Expr::If(cond, then, otherwise) => {
                self.check_expr(cond, state, false)?;
                let mut then_state = state.clone();
                self.check_expr(then, &mut then_state, place_ctx)?;
                let mut else_state = state.clone();
                if let Some(alt) = otherwise {
                    self.check_expr(alt, &mut else_state, place_ctx)?;
                }
                merge_branch(state, &then_state, &else_state);
            }
            Expr::Let(pat, init) => {
                self.check_expr(init, state, false)?;
                let src = self.expr_borrow_source(init, state);
                self.bind_pat(pat, state, src);
            }
            Expr::Match(scrut, arms) => {
                self.check_expr(scrut, state, true)?;
                let arm_ids: Vec<u32> = self.body.list(arms).to_vec();
                let mut merged: Option<State> = None;
                for arm_id in arm_ids {
                    let arm = self.body.arms[arm_id as usize];
                    let mut arm_state = state.clone();
                    self.bind_pat(arm.pat, &mut arm_state, None);
                    if let Some(g) = arm.guard {
                        self.check_expr(g, &mut arm_state, false)?;
                    }
                    self.check_expr(arm.body, &mut arm_state, place_ctx)?;
                    if !arm_state.diverged {
                        match &mut merged {
                            Some(acc) => join_moved(acc, &arm_state),
                            None => merged = Some(arm_state),
                        }
                    }
                }
                if let Some(acc) = merged {
                    *state = acc;
                } else {
                    state.diverged = true;
                }
            }
            Expr::Loop { body, .. } => {
                self.check_loop(None, None, body, state)?;
            }
            Expr::While { cond, body, .. } => {
                self.check_loop(Some(cond), None, body, state)?;
            }
            Expr::For { pat, iter, body, .. } => {
                self.check_expr(iter, state, false)?;
                self.check_loop(None, Some(pat), body, state)?;
            }
            Expr::Call(callee, args) => {
                self.check_expr(callee, state, true)?;
                self.check_list(args, state)?;
            }
            Expr::Method { receiver, params, .. } => {
                let recv_borrow = self.typed.adjusts.get(&expr).and_then(|m| m.autoref);
                if recv_borrow == Some(Mut::Mut)
                    && (self.place_behind_shared_ref(receiver)
                        || self
                            .typed
                            .exprs
                            .get(receiver.0 as usize)
                            .copied()
                            .is_some_and(|ty| ty != super::ty::Ty::NONE && matches!(self.types.kind(ty), Kind::Ref(Mut::Not, _))))
                {
                    bail!("cannot borrow data in a `&` reference as mutable");
                }
                if let Some(mutability) = recv_borrow
                    && let Some((local, through_ref)) = self.root_local(receiver)
                {
                    self.check_borrow(local, through_ref, mutability == Mut::Mut, state)?;
                    self.check_expr(receiver, state, true)?;
                } else {
                    self.check_expr(receiver, state, recv_borrow.is_some())?;
                }
                self.check_list(params, state)?;
            }
            Expr::Unary(op, inner) => {
                if op == UnOp::Deref && !place_ctx && self.deref_non_copy.contains(&expr) {
                    bail!("cannot move out of a shared or mutable reference");
                }
                self.check_expr(inner, state, op == UnOp::Deref)?;
            }
            Expr::Index(a, b) => {
                self.check_expr(a, state, true)?;
                self.check_expr(b, state, false)?;
            }
            Expr::Binary(op, a, b) => {
                let cmp = matches!(op, crate::body::BinOp::Eq | crate::body::BinOp::Ne | crate::body::BinOp::Lt | crate::body::BinOp::Le | crate::body::BinOp::Gt | crate::body::BinOp::Ge);
                self.check_expr(a, state, cmp)?;
                self.check_expr(b, state, cmp)?;
            }
            Expr::Repeat(a, b) => {
                self.check_expr(a, state, false)?;
                self.check_expr(b, state, false)?;
            }
            Expr::Field(base, _) | Expr::TupleField(base, _) => self.check_expr(base, state, true)?,
            Expr::Cast(inner, _) | Expr::Try(inner) | Expr::Await(inner) | Expr::Paren(inner) | Expr::Become(inner) => self.check_expr(inner, state, place_ctx)?,
            Expr::Tuple(list) | Expr::Array(list) => self.check_list(list, state)?,
            Expr::Struct { fields, base, .. } => {
                for &f in self.body.list(fields) {
                    self.check_expr(self.body.fields[f as usize].value, state, false)?;
                }
                if let Some(b) = base {
                    self.check_expr(b, state, false)?;
                }
            }
            Expr::Range { from, to, .. } => {
                if let Some(f) = from {
                    self.check_expr(f, state, false)?;
                }
                if let Some(t) = to {
                    self.check_expr(t, state, false)?;
                }
            }
            Expr::Return(val) => {
                if let Some(v) = val {
                    self.check_expr(v, state, false)?;
                    if self.returns_ref {
                        self.check_return_escape(v, state)?;
                    }
                }
                state.diverged = true;
            }
            Expr::Break(_, val) => {
                if let Some(v) = val {
                    self.check_expr(v, state, false)?;
                }
                state.diverged = true;
            }
            Expr::Continue(_) => state.diverged = true,
            Expr::Yield(val) => {
                if let Some(v) = val {
                    self.check_expr(v, state, false)?;
                }
            }
            Expr::Closure { .. } | Expr::Lit(..) | Expr::Macro { .. } | Expr::Underscore | Expr::FormatArgs(_) | Expr::OffsetOf { .. } => {}
        }
        Ok(())
    }

    fn check_list(&self, list: List, state: &mut State) -> Result<()> {
        for &arg in self.body.list(list) {
            self.check_expr(ExprId(arg), state, false)?;
        }
        Ok(())
    }

    fn check_loop(&self, cond: Option<ExprId>, pat: Option<PatId>, body: ExprId, state: &mut State) -> Result<()> {
        let mut iter_state = state.clone();
        for _ in 0..2 {
            iter_state.diverged = false;
            if let Some(c) = cond {
                self.check_expr(c, &mut iter_state, false)?;
            }
            if let Some(p) = pat {
                self.bind_pat(p, &mut iter_state, None);
            }
            self.check_expr(body, &mut iter_state, false)?;
        }
        join_moved(state, &iter_state);
        Ok(())
    }
}

fn join_moved(dst: &mut State, other: &State) {
    for (d, &o) in dst.moved.iter_mut().zip(&other.moved) {
        *d |= o;
    }
}

fn merge_branch(out: &mut State, a: &State, b: &State) {
    match (a.diverged, b.diverged) {
        (true, true) => out.diverged = true,
        (true, false) => *out = b.clone(),
        (false, true) => *out = a.clone(),
        (false, false) => {
            *out = a.clone();
            join_moved(out, b);
        }
    }
}

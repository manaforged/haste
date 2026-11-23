//! Ownership, mutability, and NLL-style borrow checking for local workspace function bodies.

mod bind;
mod calls;
mod checks;
mod flow;
mod liveness;
mod places;
mod propagate;

use anyhow::{Result, bail};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::body::{BinOp, BlockKind, Body, Expr, ExprId, List, Pat, PatId, UnOp, jumps_to};
use crate::names::ItemRef;
use crate::outline::item::ItemKind;
use crate::outline::source::Source;

use super::Context;
use super::infer::{Res, Typed};
use super::ty::{Kind, Mut, Ty};

use places::{PlaceStep, steps_conflict};
use propagate::BorrowSource;

const TEMP_HOLDER_BASE: u32 = u32::MAX - 65_536;
const ITER_HOLDER_BASE: u32 = TEMP_HOLDER_BASE - 65_536;
pub(super) const STMT_TEMP_OFFSET: u32 = 10_000;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Loan {
    target: u32,
    steps: Vec<PlaceStep>,
    holder: Option<u32>,
    mutable: bool,
    through_ref: bool,
    two_phase: bool,
    env_only: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PartialMove {
    local: u32,
    steps: Vec<PlaceStep>,
}

#[derive(Clone, Debug)]
struct State {
    init: Vec<bool>,
    maybe_init: Vec<bool>,
    moved: Vec<bool>,
    partial_moves: Vec<PartialMove>,
    loans: Vec<Loan>,
    diverged: bool,
}

struct BreakFrame {
    label: Option<u32>,
    is_block: bool,
    depth: u32,
    states: Vec<State>,
}

struct Checker<'c, 'a> {
    cx: &'c mut Context<'a>,
    item: ItemRef,
    body: &'c Body,
    src: &'c Source,
    typed: &'c Typed,
    mutable: Vec<bool>,
    is_copy: Vec<bool>,
    is_param: Vec<bool>,
    scope_depth: Vec<u32>,
    names: Vec<Option<String>>,
    no_move_captures: Vec<(u32, Vec<PlaceStep>)>,
    returns_borrow: bool,
    return_min_depth: u32,
    closure_depth: u32,
    const_depth: u32,
    unsafe_depth: u32,
    closure_by_val_caps: Vec<u32>,
    stmt_depth: u32,
    temp_scope_depth: u32,
    breaks: Vec<BreakFrame>,
    temp_depth: u32,
    let_extend_depth: Option<u32>,
    temp_sources: HashMap<ExprId, Vec<BorrowSource>>,
    closure_effects: HashMap<ExprId, checks::ClosureEffects>,
    drop_cache: HashMap<Ty, bool>,
    drop_locals: HashSet<u32>,
}

/// Whether `ty` is `Copy` in `item`; defaults conservatively to `false` when classification fails on a real type.
pub(super) fn is_copy_ty(cx: &mut Context<'_>, item: ItemRef, ty: Ty) -> bool {
    ty == Ty::NONE
        || matches!(cx.types.kind(ty), Kind::Ref(Mut::Not, _))
        || super::infer::is_copy(cx, item, ty).unwrap_or(false)
}

/// Checks ownership, mutability, borrow aliasing, and dangling local reference escapes on `body`.
pub(super) fn check(
    cx: &mut Context<'_>,
    item: ItemRef,
    (body, sig): (&Body, &crate::body::sig::Signature),
    src: &Source,
    typed: &Typed,
) -> Result<()> {
    let Some(root) = body.root else { return Ok(()) };
    let n = typed.locals.len();
    let mut is_copy = vec![false; n];
    for (idx, local) in typed.locals.iter().enumerate() {
        is_copy[idx] = is_copy_ty(cx, item, local.ty);
    }
    let mut mutable = vec![false; n];
    let mut is_param = vec![false; n];
    let mut scope_depth = vec![1u32; n];
    let mut names = vec![None; n];
    let mut init = vec![false; n];
    for &p in &typed.params {
        if let Some(slot) = is_param.get_mut(p as usize) {
            *slot = true;
            init[p as usize] = true;
            scope_depth[p as usize] = 0;
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
    for at in 0..body.exprs.len() as u32 {
        let id = ExprId(at);
        if let Expr::Closure { params, .. } = *body.expr(id)
            && let Some(closure) = typed.closures.get(&id)
        {
            for (&p_id, &local) in body.list(params).iter().zip(&closure.params) {
                collect_pat_meta_for_local(body, src, body.params[p_id as usize].pat, local, &mut mutable, &mut names);
            }
        }
    }
    let mut checker = Checker {
        cx, item, body, src, typed, mutable, is_copy, is_param, scope_depth, names,
        no_move_captures: Vec::new(), returns_borrow: false, return_min_depth: 1, closure_depth: 0, const_depth: 0,
        unsafe_depth: u32::from(sig.is_unsafe),
        closure_by_val_caps: Vec::new(), stmt_depth: 1, temp_scope_depth: 1,
        breaks: Vec::new(), temp_depth: 0, let_extend_depth: None, temp_sources: HashMap::default(),
        closure_effects: HashMap::default(), drop_cache: HashMap::default(), drop_locals: HashSet::default(),
    };
    for (idx, local) in typed.locals.iter().enumerate() {
        if local.ty != Ty::NONE && checker.ty_holds_borrow(local.ty) && checker.ty_has_borrow_drop(local.ty) {
            checker.drop_locals.insert(idx as u32);
        }
    }
    checker.returns_borrow = typed.ret.is_some_and(|ty| checker.ty_holds_borrow(ty));
    let mut state = State {
        maybe_init: init.clone(), init, moved: vec![false; n],
        partial_moves: Vec::new(), loans: Vec::new(), diverged: false,
    };
    let param_offset = usize::from(sig.receiver.is_some());
    for (idx, &pat) in body.fn_params.iter().enumerate() {
        let fallback = typed.params.get(param_offset + idx).map(|&pl| vec![checker.param_fallback_source(pl)]);
        checker.check_param_pat(pat, fallback.as_deref().unwrap_or(&[]), &mut state)?;
    }
    let empty_live = HashSet::default();
    let root_place_ctx = checker.is_mut_ref_ty(checker.expr_ty(root)) && typed.ret.is_some_and(|r| checker.is_mut_ref_ty(r));
    checker.check_expr(root, &mut state, root_place_ctx, &empty_live, true, 0)?;
    if checker.returns_borrow && !state.diverged {
        checker.check_return_escape(root, &state)?;
    }
    checker.check_coercion_vtables()?;
    Ok(())
}

fn collect_pat_meta(body: &Body, typed: &Typed, src: &Source, pat: PatId, mutable: &mut [bool], names: &mut [Option<String>]) {
    if let Some(&local) = typed.binds.get(&pat) {
        collect_pat_meta_for_local(body, src, pat, local, mutable, names);
    }
}

fn collect_pat_meta_for_local(body: &Body, src: &Source, pat: PatId, local: u32, mutable: &mut [bool], names: &mut [Option<String>]) {
    if let Pat::Bind { name, mutable: is_mut, .. } = *body.pat(pat)
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

fn loan_conflicts_borrow(l: &Loan, local: u32, steps: &[PlaceStep]) -> bool {
    l.target == local && steps_conflict(&l.steps, steps)
}

fn loan_conflicts_assign(l: &Loan, local: u32, through_ref: bool, steps: &[PlaceStep], is_ref_local: bool) -> bool {
    if l.target != local {
        return false;
    }
    if !through_ref && is_ref_local && steps.is_empty() && l.steps.first() == Some(&PlaceStep::Deref) {
        return false;
    }
    steps_conflict(&l.steps, steps)
}

pub(super) fn is_temp_holder(holder: Option<u32>) -> bool {
    holder.is_some_and(|h| h >= TEMP_HOLDER_BASE)
}

fn prune_moved_loans(state: &mut State) {
    state.loans.retain(|l| {
        l.holder
            .is_none_or(|h| is_temp_holder(Some(h)) || !state.moved.get(h as usize).copied().unwrap_or(false))
    });
}

fn prune_dead_loans(state: &mut State, live: &HashSet<u32>) {
    if !state.loans.is_empty() {
        state
            .loans
            .retain(|l| l.holder.is_none_or(|h| is_temp_holder(Some(h)) || live.contains(&h)));
    }
}

impl Checker<'_, '_> {
    fn next_temp_holder(&mut self) -> u32 {
        let id = TEMP_HOLDER_BASE + self.temp_depth;
        self.temp_depth += 1;
        id
    }

    fn next_iter_holder(&mut self) -> u32 {
        let id = ITER_HOLDER_BASE + self.temp_depth;
        self.temp_depth += 1;
        id
    }

    pub(super) fn is_const_or_static_item(&self) -> bool {
        self.const_depth > 0
            || (self.closure_depth == 0
                && matches!(self.cx.item(self.item).1.kind, ItemKind::Const | ItemKind::Static))
    }

    pub(super) fn with_let_extend<T>(&mut self, depth: Option<u32>, f: impl FnOnce(&mut Self) -> T) -> T {
        let saved = std::mem::replace(&mut self.let_extend_depth, depth);
        let res = f(self);
        self.let_extend_depth = saved;
        res
    }

    pub(super) fn param_fallback_source(&mut self, local: u32) -> BorrowSource {
        let ty = self.local_ty(local);
        BorrowSource {
            target: local,
            steps: Vec::new(),
            mutable: self.is_mut_ref_ty(ty),
            through_ref: self.ty_holds_borrow(ty),
            env_only: false,
        }
    }

    fn is_escaping_return(&self, src: &BorrowSource) -> bool {
        if checks::is_temp_target(src.target).is_some() {
            return !self.is_const_or_static_item();
        }
        if self.closure_depth > 0
            && !src.through_ref
            && !self.is_ref_ty(self.local_ty(src.target))
            && self.closure_by_val_caps.contains(&src.target)
        {
            return true;
        }
        if src.through_ref && (self.is_ref_ty(self.local_ty(src.target)) || self.is_closure_param(src.target)) {
            return false;
        }
        if self.return_min_depth <= 1 {
            return !self.from_caller(src.target, src.through_ref);
        }
        self.scope_depth.get(src.target as usize).copied().unwrap_or(0) >= self.return_min_depth
    }

    pub(super) fn escapes_depth(&self, target: u32, through_ref: bool, depth: u32) -> bool {
        if let Some(t_depth) = checks::is_temp_target(target) {
            if self.is_const_or_static_item() {
                return false;
            }
            let effective = if t_depth >= STMT_TEMP_OFFSET { t_depth - STMT_TEMP_OFFSET } else { t_depth };
            return effective >= depth;
        }
        !self.from_caller(target, through_ref)
            && !(through_ref && self.is_ref_ty(self.local_ty(target)))
            && self.scope_depth.get(target as usize).copied().unwrap_or(0) >= depth
    }

    pub(super) fn push_partial_move(&self, state: &mut State, local: u32, steps: Vec<PlaceStep>) {
        let pm = PartialMove { local, steps };
        if !state.partial_moves.contains(&pm) {
            state.partial_moves.push(pm);
        }
    }

    pub(super) fn check_static_mut(&self, expr: ExprId, extra_deref: bool) -> Result<()> {
        if let Some((item, mut mr, bs)) = self.static_root(expr) {
            if extra_deref && self.is_mut_ref_ty(self.expr_ty(expr)) && !bs {
                mr = true;
            }
            if bs || (!mr && !self.is_static_mut(item)) {
                bail!("cannot borrow or assign to immutable static item `{}`", self.cx.name(item));
            }
        }
        Ok(())
    }

    fn check_param_pat(&mut self, pat: PatId, fallback: &[BorrowSource], state: &mut State) -> Result<()> {
        self.bind_pat_param(pat, fallback, state)
    }

    fn check_not_moved(&self, local: u32, steps: &[PlaceStep], state: &State) -> Result<()> {
        if !state.init.get(local as usize).copied().unwrap_or(true)
            || state.moved.get(local as usize).copied().unwrap_or(false)
            || state
                .partial_moves
                .iter()
                .any(|m| m.local == local && steps_conflict(&m.steps, steps))
        {
            bail!("use of moved or uninitialized value `{}`", self.local_name(local));
        }
        Ok(())
    }

    fn check_can_read(&self, local: u32, steps: &[PlaceStep], state: &State) -> Result<()> {
        self.check_can_read_inner(local, steps, state, false)
    }

    fn check_can_read_inner(&self, local: u32, steps: &[PlaceStep], state: &State, allow_shared: bool) -> Result<()> {
        self.check_not_moved(local, steps, state)?;
        if state.loans.iter().any(|l| {
            loan_conflicts_borrow(l, local, steps) && l.mutable && (allow_shared || !l.two_phase)
        }) {
            bail!("cannot use `{}` because it was mutably borrowed", self.local_name(local));
        }
        Ok(())
    }

    fn check_can_move(&self, local: u32, steps: &[PlaceStep], state: &State) -> Result<()> {
        self.check_not_moved(local, steps, state)?;
        if self.no_move_captures.iter().any(|(l, s)| *l == local && steps_conflict(s, steps)) {
            bail!("cannot move out of `{}` in a captured closure", self.local_name(local));
        }
        if state.loans.iter().any(|l| loan_conflicts_borrow(l, local, steps)) {
            bail!("cannot move out of `{}` because it is borrowed", self.local_name(local));
        }
        Ok(())
    }

    fn check_can_mutate(
        &self,
        local: u32,
        through_ref: bool,
        mut_ref_root: bool,
        behind_shared: bool,
        steps: &[PlaceStep],
        state: &State,
        init_ok: bool,
    ) -> Result<()> {
        if behind_shared {
            bail!("cannot assign to data in a `&` reference or `*const` pointer");
        }
        if !mut_ref_root
            && !self.mutable.get(local as usize).copied().unwrap_or(true)
            && !(init_ok && !state.maybe_init.get(local as usize).copied().unwrap_or(true))
        {
            bail!("cannot assign to immutable variable `{}`", self.local_name(local));
        }
        let is_ref_local = self.is_ref_ty(self.local_ty(local));
        if state
            .loans
            .iter()
            .any(|l| loan_conflicts_assign(l, local, through_ref, steps, is_ref_local))
        {
            bail!("cannot assign to `{}` because it is borrowed", self.local_name(local));
        }
        Ok(())
    }

    fn check_borrow(
        &self,
        local: u32,
        _through_ref: bool,
        mut_ref_root: bool,
        steps: &[PlaceStep],
        mutable: bool,
        state: &State,
    ) -> Result<()> {
        self.check_not_moved(local, steps, state)?;
        if mutable && !mut_ref_root && !self.mutable.get(local as usize).copied().unwrap_or(true) {
            bail!("cannot borrow `{}` as mutable, as it is not declared as mutable", self.local_name(local));
        }
        if mutable && state.loans.iter().any(|l| loan_conflicts_borrow(l, local, steps) && l.mutable) {
            bail!("cannot borrow `{}` as mutable more than once at a time", self.local_name(local));
        }
        if mutable && state.loans.iter().any(|l| loan_conflicts_borrow(l, local, steps)) {
            bail!("cannot borrow `{}` as mutable because it is also borrowed as immutable", self.local_name(local));
        }
        if !mutable && state.loans.iter().any(|l| loan_conflicts_borrow(l, local, steps) && l.mutable && !l.two_phase) {
            bail!("cannot borrow `{}` as immutable because it is also borrowed as mutable", self.local_name(local));
        }
        Ok(())
    }

    fn check_return_escape(&mut self, expr: ExprId, state: &State) -> Result<()> {
        match *self.body.expr(expr) {
            Expr::Block { kind: BlockKind::Const, .. } => Ok(()),
            Expr::Paren(inner) | Expr::Block { tail: Some(inner), .. } => self.check_return_escape(inner, state),
            Expr::If(_, then, otherwise) => {
                self.check_return_escape(then, state)?;
                if let Some(alt) = otherwise { self.check_return_escape(alt, state)?; }
                Ok(())
            }
            Expr::Match(_, arms) => {
                for id in self.body.list(arms).to_vec() {
                    self.check_return_escape(self.body.arms[id as usize].body, state)?;
                }
                Ok(())
            }
            _ => {
                for src in self.expr_borrow_sources(expr, state) {
                    if self.is_escaping_return(&src) {
                        bail!("cannot return value referencing local variable `{}`", self.local_name(src.target));
                    }
                }
                Ok(())
            }
        }
    }

    fn check_block_escape(&self, block_depth: u32, state: &State, outer_live: &HashSet<u32>) -> Result<()> {
        for l in &state.loans {
            let Some(holder) = l.holder else { continue };
            if !is_temp_holder(Some(holder)) && outer_live.contains(&holder) && self.escapes_depth(l.target, l.through_ref, block_depth) {
                bail!("`{}` does not live long enough", self.local_name(l.target));
            }
        }
        Ok(())
    }

    fn check_expr(&mut self, expr: ExprId, state: &mut State, place_ctx: bool, outer_live: &HashSet<u32>, value_used: bool, depth: u32) -> Result<()> {
        if state.diverged {
            return Ok(());
        }
        match *self.body.expr(expr) {
            Expr::Path(_) => {
                if let Some(Res::Local(local)) = self.typed.res.get(&expr)
                    && (*local as usize) < self.is_copy.len()
                {
                    if place_ctx || self.is_copy[*local as usize] {
                        self.check_can_read(*local, &[], state)?;
                    } else {
                        self.check_can_move(*local, &[], state)?;
                        state.moved[*local as usize] = true;
                        state.partial_moves.retain(|m| m.local != *local);
                    }
                } else if let Some(Res::Fn(item, _)) = self.typed.res.get(&expr)
                    && self.cx.item(*item).1.kind == ItemKind::Static
                {
                    if self.unsafe_depth == 0 && self.is_static_mut(*item) {
                        bail!("use of mutable static `{}` is unsafe and requires unsafe function or block", self.cx.name(*item));
                    }
                    if !place_ctx && !self.ty_is_copy(self.expr_ty(expr)) {
                        bail!("cannot move out of static item `{}`", self.cx.name(*item));
                    }
                }
            }
            Expr::Ref { mutable, raw, expr: inner } => {
                if !raw {
                    self.check_union_place(inner)?;
                }
                if mutable && self.place_behind_shared_ref(inner) {
                    bail!("cannot borrow data in a `&` reference as mutable");
                }
                if mutable {
                    self.check_static_mut(inner, false)?;
                }
                if let Some(p) = self.root_place(inner) {
                    if mutable && p.behind_shared {
                        bail!("cannot borrow data in a `&` reference or `*const` pointer as mutable");
                    }
                    if let Some(prefix_len) = p.raw_ptr_prefix {
                        self.check_can_read(p.local, &p.steps[..prefix_len], state)?;
                    } else {
                        self.check_borrow(p.local, p.through_ref, p.mut_ref_root, &p.steps, mutable, state)?;
                    }
                }
                self.check_expr(inner, state, true, outer_live, false, depth)?;
            }
            Expr::Assign(lhs, rhs) => {
                let mut after_rhs = outer_live.clone();
                self.collect_live_expr(lhs, &mut after_rhs);
                let reborrow_mut = self.is_mut_ref_ty(self.expr_ty(lhs)) && self.is_mut_ref_ty(self.expr_ty(rhs));
                self.check_expr(rhs, state, reborrow_mut, &after_rhs, true, depth)?;
                if !state.diverged {
                    self.check_mut_ref_reborrow(lhs, rhs, state)?;
                    let sources = self.expr_borrow_sources(rhs, state);
                    prune_dead_loans(state, &after_rhs);
                    self.assign_place(lhs, &sources, state, outer_live, depth)?;
                    prune_moved_loans(state);
                }
            }
            Expr::AssignOp(_, lhs, rhs) => {
                self.check_union_place(lhs)?;
                let mut after_rhs = outer_live.clone();
                self.collect_live_expr(lhs, &mut after_rhs);
                self.check_expr(rhs, state, false, &after_rhs, true, depth)?;
                if !state.diverged {
                    let sources = if self.typed.operators.contains_key(&expr) && self.ty_holds_borrow(self.expr_ty(lhs)) {
                        self.expr_borrow_sources(rhs, state)
                    } else {
                        Vec::new()
                    };
                    prune_dead_loans(state, &after_rhs);
                    if self.place_behind_shared_ref(lhs) {
                        bail!("cannot assign to data in a `&` reference");
                    }
                    self.check_static_mut(lhs, false)?;
                    if let Some(p) = self.root_place(lhs) {
                        self.check_not_moved(p.local, &p.steps, state)?;
                        self.check_can_mutate(p.local, p.through_ref, p.mut_ref_root, p.behind_shared, &p.steps, state, false)?;
                        if !sources.is_empty() {
                            self.propagate_assign_sources(&p, &sources, state)?;
                        }
                    } else {
                        self.check_expr(lhs, state, true, outer_live, false, depth)?;
                    }
                }
            }
            Expr::Block { kind, stmts, tail, label } => {
                if matches!(kind, BlockKind::Async | BlockKind::AsyncMove) {
                    self.check_closure(expr, None, expr, state, depth)?;
                } else {
                    self.check_block(expr, kind, stmts, tail, label, state, (place_ctx, value_used), outer_live, depth)?;
                }
            }
            Expr::If(cond, then, otherwise) => {
                self.check_if(expr, cond, then, otherwise, state, (place_ctx, value_used), outer_live, depth)?;
            }
            Expr::Let(pat, init) => {
                self.check_expr(init, state, true, outer_live, true, depth)?;
                let scrut_place = self.root_place(init);
                self.bind_pat(pat, Some(init), scrut_place.as_ref(), &[], &[], state, (true, true, depth))?;
            }
            Expr::Match(scrut, arms) => {
                self.check_match(expr, scrut, arms, state, (place_ctx, value_used), outer_live, depth)?;
            }
            Expr::Loop { label, body } => self.check_loop(label, None, None, None, body, state, outer_live, depth)?,
            Expr::While { label, cond, body } => self.check_loop(label, Some(cond), None, None, body, state, outer_live, depth)?,
            Expr::For { label, pat, iter, body } => self.check_for(label, pat, iter, body, state, outer_live, depth)?,
            Expr::Call(callee, args) => self.check_call(expr, callee, args, state, outer_live, depth)?,
            Expr::Method { receiver, params, .. } => self.check_method(expr, receiver, params, state, outer_live, depth)?,
            Expr::Unary(op, inner) => {
                if op == UnOp::Deref
                    && self.unsafe_depth == 0
                    && self.expr_ty(inner) != Ty::NONE
                    && matches!(self.cx.types.kind(self.expr_ty(inner)), Kind::Ptr(..))
                {
                    bail!("dereference of raw pointer is unsafe and requires unsafe function or block");
                }
                if op == UnOp::Deref && !place_ctx && !self.ty_is_copy(self.expr_ty(expr)) {
                    if self.typed.derefs.contains_key(&(expr, 0)) || self.place_behind_ref_or_ptr(expr) {
                        bail!("cannot move out of a shared or mutable reference");
                    }
                    self.check_expr(inner, state, true, outer_live, false, depth)?;
                    if let Some(p) = self.root_place(expr) {
                        self.check_can_move(p.local, &p.steps, state)?;
                        state.partial_moves.push(PartialMove { local: p.local, steps: p.steps });
                    }
                } else {
                    self.check_expr(inner, state, op == UnOp::Deref, outer_live, false, depth)?;
                }
            }
            Expr::Index(a, b) => {
                if !place_ctx && !self.ty_is_copy(self.expr_ty(expr)) {
                    bail!("cannot move out of index");
                }
                self.check_expr(a, state, true, outer_live, false, depth)?;
                self.check_expr(b, state, false, outer_live, true, depth)?;
            }
            Expr::Binary(op, a, b) => {
                if matches!(op, BinOp::And | BinOp::Or) {
                    let mut after_a = outer_live.clone();
                    self.collect_live_expr(b, &mut after_a);
                    self.check_expr(a, state, false, &after_a, true, depth)?;
                    if !state.diverged {
                        let before_rhs = state.clone();
                        let mut rhs_state = state.clone();
                        self.check_expr(b, &mut rhs_state, false, outer_live, true, depth)?;
                        merge_branch(state, &before_rhs, &rhs_state);
                    }
                } else {
                    let cmp = matches!(op, BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge);
                    self.check_expr(a, state, cmp, outer_live, !cmp, depth)?;
                    self.check_expr(b, state, cmp, outer_live, !cmp, depth)?;
                }
            }
            Expr::Repeat(a, b) => {
                self.check_expr(a, state, false, outer_live, true, depth)?;
                self.check_expr(b, state, false, outer_live, true, depth)?;
            }
            Expr::Field(base, _) | Expr::TupleField(base, _) => {
                if !place_ctx && matches!(*self.body.expr(expr), Expr::Field(..)) {
                    self.check_union_access(expr, base)?;
                }
                let moving = !place_ctx && !self.ty_is_copy(self.expr_ty(expr));
                if let Some(p) = self.root_place(expr)
                    && !p.steps.is_empty()
                    && !p.steps.iter().any(|s| matches!(s, PlaceStep::Opaque(_)))
                {
                    if p.steps.last() == Some(&PlaceStep::Union) {
                        let prefix = &p.steps[..p.steps.len() - 1];
                        if !moving {
                            self.check_can_read(p.local, prefix, state)?;
                        } else if p.through_ref {
                            bail!("cannot move out of `{}` which is behind a reference", self.local_name(p.local));
                        } else {
                            self.check_can_move(p.local, prefix, state)?;
                            if prefix.is_empty() {
                                state.moved[p.local as usize] = true;
                                state.partial_moves.retain(|m| m.local != p.local);
                            } else {
                                self.push_partial_move(state, p.local, prefix.to_vec());
                            }
                        }
                    } else if !moving {
                        self.check_can_read(p.local, &p.steps, state)?;
                    } else if p.through_ref {
                        bail!("cannot move out of `{}` which is behind a reference", self.local_name(p.local));
                    } else {
                        self.check_adt_drop_move(self.field_container_ty(expr, base))?;
                        self.check_can_move(p.local, &p.steps, state)?;
                        self.push_partial_move(state, p.local, p.steps);
                    }
                } else {
                    if moving {
                        if self.place_behind_ref_or_ptr(expr) {
                            bail!("cannot move out of a reference");
                        }
                        if self.static_root(expr).is_some() {
                            bail!("cannot move out of static item");
                        }
                        self.check_adt_drop_move(self.field_container_ty(expr, base))?;
                    }
                    self.check_expr(base, state, true, outer_live, false, depth)?;
                }
            }
            Expr::Cast(inner, _) => {
                let reborrow_mut = self.is_mut_ref_ty(self.expr_ty(inner));
                self.check_expr(inner, state, reborrow_mut, outer_live, true, depth)?;
            }
            Expr::Try(inner) | Expr::Await(inner) | Expr::Become(inner) => {
                self.check_expr(inner, state, false, outer_live, true, depth)?;
            }
            Expr::Paren(inner) => {
                self.check_expr(inner, state, place_ctx, outer_live, value_used, depth)?;
            }
            Expr::Tuple(list) | Expr::Array(list) => {
                let ids: Vec<u32> = self.body.list(list).to_vec();
                self.check_arg_list(&ids, None, None, false, state, outer_live, depth)?;
            }
            Expr::Struct { fields, base, .. } => {
                self.check_struct_expr(expr, fields, base, state, outer_live, depth)?;
            }
            Expr::Range { from, to, .. } => {
                if let Some(f) = from {
                    self.check_expr(f, state, false, outer_live, true, depth)?;
                }
                if let Some(t) = to {
                    self.check_expr(t, state, false, outer_live, true, depth)?;
                }
            }
            Expr::Return(val) => {
                if let Some(v) = val {
                    let reborrow_mut = self.is_mut_ref_ty(self.expr_ty(v)) && self.typed.ret.is_some_and(|r| self.is_mut_ref_ty(r));
                    self.check_expr(v, state, reborrow_mut, outer_live, true, depth)?;
                    if self.returns_borrow {
                        self.check_return_escape(v, state)?;
                    }
                }
                state.diverged = true;
            }
            Expr::Break(label, val) => {
                if let Some(v) = val {
                    self.check_expr(v, state, false, outer_live, true, depth)?;
                }
                if let Some(idx) = self.breaks.iter().rposition(|f| jumps_to(self.src, (f.label, f.is_block), label)) {
                    let target_depth = self.breaks[idx].depth;
                    if let Some(v) = val {
                        for src in self.expr_borrow_sources(v, state) {
                            if self.escapes_depth(src.target, src.through_ref, target_depth) {
                                bail!("`{}` does not live long enough", self.local_name(src.target));
                            }
                        }
                    }
                    self.breaks[idx].states.push(state.clone());
                }
                state.diverged = true;
            }
            Expr::Continue(_) => state.diverged = true,
            Expr::Yield(val) => {
                if let Some(v) = val {
                    self.check_expr(v, state, false, outer_live, true, depth)?;
                }
            }
            Expr::Closure { params, body, .. } => self.check_closure(expr, Some(params), body, state, depth)?,
            Expr::Lit(..) | Expr::Macro { .. } | Expr::Underscore | Expr::FormatArgs(_) | Expr::OffsetOf { .. } => {}
        }
        let expr_ty = self.expr_ty(expr);
        if expr_ty != Ty::NONE && matches!(*self.cx.types.kind(expr_ty), Kind::Never) {
            state.diverged = true;
        }
        Ok(())
    }

    fn check_struct_expr(&mut self, expr: ExprId, fields: List, base: Option<ExprId>, state: &mut State, outer_live: &HashSet<u32>, depth: u32) -> Result<()> {
        let val_ids: Vec<u32> = self.body.list(fields).iter().map(|&f| self.body.fields[f as usize].value.0).collect();
        self.check_arg_list(&val_ids, None, None, false, state, outer_live, depth)?;
        if let Some(b) = base {
            self.check_fru_base(expr, fields, b, state, outer_live, depth)?;
        }
        Ok(())
    }
}

fn join_moved(dst: &mut State, other: &State) {
    for (d, &o) in dst.init.iter_mut().zip(&other.init) { *d &= o; }
    for (d, &o) in dst.maybe_init.iter_mut().zip(&other.maybe_init) { *d |= o; }
    for (d, &o) in dst.moved.iter_mut().zip(&other.moved) { *d |= o; }
    for pm in &other.partial_moves {
        if !dst.partial_moves.contains(pm) { dst.partial_moves.push(pm.clone()); }
    }
}

fn merge_states(dst: &mut State, other: &State) {
    join_moved(dst, other);
    for l in &other.loans {
        if !dst.loans.contains(l) { dst.loans.push(l.clone()); }
    }
}

fn merge_branch(out: &mut State, a: &State, b: &State) {
    match (a.diverged, b.diverged) {
        (true, true) => out.diverged = true,
        (true, false) => *out = b.clone(),
        (false, true) => *out = a.clone(),
        (false, false) => { *out = a.clone(); merge_states(out, b); }
    }
}

mod bind;
mod calls;
mod checks;
mod diag;
mod expr;
mod flow;
mod liveness;
mod places;
mod propagate;

use anyhow::Result;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::body::{BlockKind, Body, Expr, ExprId, Pat, PatId};
use crate::names::ItemRef;
use crate::outline::item::ItemKind;
use crate::outline::source::Source;

use super::Context;
use super::infer::Typed;
use super::ty::{Kind, Mut, Ty};

use places::{PlaceStep, steps_conflict};
use propagate::{BorrowSource, TupleSources};

const TEMP_HOLDER_BASE: u32 = u32::MAX - 65_536;
const ITER_HOLDER_BASE: u32 = TEMP_HOLDER_BASE - 65_536;
pub(super) const STMT_TEMP_OFFSET: u32 = 10_000;

#[derive(Clone, Debug)]
pub(super) struct Loan {
    target: u32,
    steps: Vec<PlaceStep>,
    holder: Option<u32>,
    mutable: bool,
    through_ref: bool,
    two_phase: bool,
    env_only: bool,
    origin_only: bool,
    site: Option<ExprId>,
}

impl PartialEq for Loan {
    fn eq(&self, other: &Self) -> bool {
        let Self {
            target,
            steps,
            holder,
            mutable,
            through_ref,
            two_phase,
            env_only,
            origin_only,
            site: _,
        } = self;
        *target == other.target
            && *steps == other.steps
            && *holder == other.holder
            && *mutable == other.mutable
            && *through_ref == other.through_ref
            && *two_phase == other.two_phase
            && *env_only == other.env_only
            && *origin_only == other.origin_only
    }
}

impl Eq for Loan {}

#[derive(Clone, Debug)]
struct PartialMove {
    local: u32,
    steps: Vec<PlaceStep>,
    site: Option<ExprId>,
}

impl PartialEq for PartialMove {
    fn eq(&self, other: &Self) -> bool {
        let Self {
            local,
            steps,
            site: _,
        } = self;
        *local == other.local && *steps == other.steps
    }
}

impl Eq for PartialMove {}

#[derive(Clone, Debug)]
struct State {
    init: Vec<bool>,
    maybe_init: Vec<bool>,
    moved: Vec<bool>,
    moved_at: Vec<Option<ExprId>>,
    partial_moves: Vec<PartialMove>,
    loans: Vec<Loan>,
    tuple_elem_sources: HashMap<u32, TupleSources>,
    diverged: bool,
}

impl State {
    fn new(init: Vec<bool>) -> Self {
        let n = init.len();
        Self {
            maybe_init: init.clone(),
            init,
            moved: vec![false; n],
            moved_at: vec![None; n],
            partial_moves: Vec::new(),
            loans: Vec::new(),
            tuple_elem_sources: HashMap::default(),
            diverged: false,
        }
    }

    pub(super) fn for_closure(outer: &Self) -> Self {
        Self {
            init: outer.init.clone(),
            maybe_init: outer.maybe_init.clone(),
            moved: outer.moved.clone(),
            moved_at: outer.moved_at.clone(),
            partial_moves: outer.partial_moves.clone(),
            loans: Vec::new(),
            tuple_elem_sources: HashMap::default(),
            diverged: false,
        }
    }

    pub(super) fn set_tuple_sources(&mut self, local: u32, srcs: Option<TupleSources>) {
        if let Some(ts) = srcs {
            self.tuple_elem_sources.insert(local, ts);
        } else {
            self.tuple_elem_sources.remove(&local);
        }
    }
}

struct BreakFrame {
    label: Option<u32>,
    is_block: bool,
    depth: u32,
    states: Vec<State>,
    sources: Vec<BorrowSource>,
}

struct ActiveClosure {
    params: Vec<u32>,
    ret_params: Vec<usize>,
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
    returns_only_static: bool,
    param_lts: calls::ParamLifetimes<'c>,
    return_min_depth: u32,
    closure_depth: u32,
    const_depth: u32,
    unsafe_depth: u32,
    closure_by_val_caps: Vec<u32>,
    active_closures: Vec<ActiveClosure>,
    stmt_depth: u32,
    temp_scope_depth: u32,
    breaks: Vec<BreakFrame>,
    temp_depth: u32,
    let_extend_depth: Option<u32>,
    temp_sources: HashMap<ExprId, Vec<BorrowSource>>,
    closure_effects: HashMap<ExprId, checks::ClosureEffects>,
    closure_param_origins: HashMap<u32, Vec<BorrowSource>>,
    drop_cache: HashMap<Ty, bool>,
    drop_locals: HashSet<u32>,
    current_expr: Option<ExprId>,
    current_pat: Option<PatId>,
}

/// Whether `ty` is `Copy` in `item`, resolving closures declared inside `typed` directly and defaulting conservatively to `false`.
pub(super) fn is_copy_ty(cx: &mut Context<'_>, item: ItemRef, typed: &Typed, ty: Ty) -> bool {
    if ty == Ty::NONE {
        return true;
    }
    match cx.types.kind(ty).clone() {
        Kind::Ref(Mut::Not, _) => true,
        Kind::Ref(Mut::Mut, _)
        | Kind::Slice(_)
        | Kind::Dyn(..)
        | Kind::Prim(crate::names::Primitive::Str) => false,
        Kind::Tuple(items) => cx
            .types
            .substs(items)
            .to_vec()
            .into_iter()
            .all(|t| is_copy_ty(cx, item, typed, t)),
        Kind::Array(inner, _) => is_copy_ty(cx, item, typed, inner),
        Kind::Closure(parent, _, at) if parent == item => {
            typed.closures.get(&ExprId(at)).is_some_and(|c| {
                !c.coroutine
                    && c.captures.iter().all(|cap| match cap.by {
                        super::infer::By::Ref => true,
                        super::infer::By::Mut => false,
                        super::infer::By::Value => {
                            let local_ty = typed.locals[cap.local as usize].ty;
                            let cap_ty = cap.ty(cx, local_ty).unwrap_or(local_ty);
                            is_copy_ty(cx, item, typed, cap_ty)
                        }
                    })
            })
        }
        _ => super::infer::is_copy(cx, item, ty).unwrap_or(false),
    }
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
        is_copy[idx] = is_copy_ty(cx, item, typed, local.ty);
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
        if let Pat::Bind {
            name,
            mutable: is_mut,
            ..
        } = *sig.body.pat(pat)
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
                collect_pat_meta_for_local(
                    body,
                    src,
                    body.params[p_id as usize].pat,
                    local,
                    &mut mutable,
                    &mut names,
                );
            }
        }
    }
    let returns_only_static = !sig.is_async
        && sig.ret.is_some_and(|ret_ast| {
            let mut ret_lts = calls::Lifetimes::default();
            calls::collect_ty_lifetimes(&sig.body, src, ret_ast, &mut ret_lts);
            calls::sig_returns_only_static(sig, src, ret_ast, &ret_lts)
        });
    let param_lts = calls::compute_param_lifetimes(cx, item, sig, src, typed, n);
    let mut checker = Checker {
        cx,
        item,
        body,
        src,
        typed,
        mutable,
        is_copy,
        is_param,
        scope_depth,
        names,
        no_move_captures: Vec::new(),
        returns_borrow: false,
        returns_only_static,
        param_lts,
        return_min_depth: 1,
        closure_depth: 0,
        const_depth: 0,
        unsafe_depth: u32::from(sig.is_unsafe),
        closure_by_val_caps: Vec::new(),
        active_closures: Vec::new(),
        stmt_depth: 1,
        temp_scope_depth: 1,
        breaks: Vec::new(),
        temp_depth: 0,
        let_extend_depth: None,
        temp_sources: HashMap::default(),
        closure_effects: HashMap::default(),
        closure_param_origins: HashMap::default(),
        drop_cache: HashMap::default(),
        drop_locals: HashSet::default(),
        current_expr: None,
        current_pat: None,
    };
    for (idx, local) in typed.locals.iter().enumerate() {
        if local.ty != Ty::NONE {
            if checker.ty_holds_borrow(local.ty) && checker.ty_has_borrow_drop(local.ty) {
                checker.drop_locals.insert(idx as u32);
            }
            if checker.is_param.get(idx).copied().unwrap_or(false)
                && let Kind::Ref(_, inner) = *checker.cx.types.kind(local.ty)
                && !checker.ty_holds_borrow(inner)
                && !checker.param_lts.inner_has_elided_lt[idx]
                && checker.param_lts.inner_named_lts[idx].is_empty()
            {
                checker.param_lts.inner_is_static[idx] = true;
            }
        }
    }
    checker.returns_borrow = typed.ret.is_some_and(|ty| checker.ty_holds_borrow(ty));
    let mut state = State::new(init);
    let param_offset = usize::from(sig.receiver.is_some());
    for (idx, &pat) in body.fn_params.iter().enumerate() {
        let fallback = typed
            .params
            .get(param_offset + idx)
            .map(|&pl| vec![checker.param_fallback_source(pl)]);
        checker.check_param_pat(pat, fallback.as_deref().unwrap_or(&[]), &mut state)?;
    }
    let empty_live = HashSet::default();
    let root_place_ctx = checker.is_mut_ref_ty(checker.expr_ty(root))
        && typed.ret.is_some_and(|r| checker.is_mut_ref_ty(r));
    checker.check_expr(root, &mut state, root_place_ctx, &empty_live, true, 0)?;
    if checker.returns_borrow && !state.diverged {
        checker.check_return_escape(root, &state)?;
    }
    checker.check_coercion_vtables()?;
    Ok(())
}

fn collect_pat_meta(
    body: &Body,
    typed: &Typed,
    src: &Source,
    pat: PatId,
    mutable: &mut [bool],
    names: &mut [Option<String>],
) {
    if let Some(&local) = typed.binds.get(&pat) {
        collect_pat_meta_for_local(body, src, pat, local, mutable, names);
    }
}

fn collect_pat_meta_for_local(
    body: &Body,
    src: &Source,
    pat: PatId,
    local: u32,
    mutable: &mut [bool],
    names: &mut [Option<String>],
) {
    if let Pat::Bind {
        name,
        mutable: is_mut,
        ..
    } = *body.pat(pat)
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
    !l.origin_only && l.target == local && steps_conflict(&l.steps, steps)
}

fn loan_conflicts_assign(
    l: &Loan,
    local: u32,
    through_ref: bool,
    steps: &[PlaceStep],
    is_ref_local: bool,
) -> bool {
    if l.origin_only || l.target != local {
        return false;
    }
    if !through_ref
        && is_ref_local
        && steps.is_empty()
        && l.steps.first() == Some(&PlaceStep::Deref)
    {
        return false;
    }
    steps_conflict(&l.steps, steps)
}

pub(super) fn is_temp_holder(holder: Option<u32>) -> bool {
    holder.is_some_and(|h| h >= TEMP_HOLDER_BASE)
}

fn prune_moved_loans(state: &mut State) {
    state.loans.retain(|l| {
        l.holder.is_none_or(|h| {
            is_temp_holder(Some(h)) || !state.moved.get(h as usize).copied().unwrap_or(false)
        })
    });
}

fn prune_dead_loans(state: &mut State, live: &HashSet<u32>) {
    if state.loans.is_empty() {
        return;
    }
    let mut keep = live.clone();
    loop {
        let mut grew = false;
        for l in &state.loans {
            if l.holder
                .is_none_or(|h| is_temp_holder(Some(h)) || keep.contains(&h))
                && keep.insert(l.target)
            {
                grew = true;
            }
        }
        if !grew {
            break;
        }
    }
    state.loans.retain(|l| {
        l.holder
            .is_none_or(|h| is_temp_holder(Some(h)) || keep.contains(&h))
    });
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
                && matches!(
                    self.cx.item(self.item).1.kind,
                    ItemKind::Const | ItemKind::Static
                ))
    }

    pub(super) fn with_let_extend<T>(
        &mut self,
        depth: Option<u32>,
        f: impl FnOnce(&mut Self) -> T,
    ) -> T {
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
            origin_only: true,
        }
    }

    pub(super) fn is_active_closure_param(&self, local: u32) -> bool {
        self.active_closures
            .iter()
            .any(|c| c.params.contains(&local))
    }

    pub(super) fn is_non_static_param(&self, local: u32) -> bool {
        self.is_param.get(local as usize).copied().unwrap_or(false)
            && !self
                .param_lts
                .is_static
                .get(local as usize)
                .copied()
                .unwrap_or(false)
    }

    pub(super) fn param_inner_is_static(&self, local: u32) -> bool {
        self.param_lts
            .inner_is_static
            .get(local as usize)
            .copied()
            .unwrap_or(false)
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
        if src.through_ref
            && (self.is_ref_ty(self.local_ty(src.target)) || self.is_closure_param(src.target))
        {
            return false;
        }
        if self.return_min_depth <= 1 {
            return !self.from_caller(src.target, src.through_ref);
        }
        self.scope_depth
            .get(src.target as usize)
            .copied()
            .unwrap_or(0)
            >= self.return_min_depth
    }

    pub(super) fn escapes_depth(&self, target: u32, through_ref: bool, depth: u32) -> bool {
        if let Some(t_depth) = checks::is_temp_target(target) {
            if self.is_const_or_static_item() {
                return false;
            }
            let effective = if t_depth >= STMT_TEMP_OFFSET {
                t_depth - STMT_TEMP_OFFSET
            } else {
                t_depth
            };
            return effective >= depth;
        }
        !self.from_caller(target, through_ref)
            && !(through_ref
                && (self.is_ref_ty(self.local_ty(target)) || self.is_closure_param(target)))
            && self.scope_depth.get(target as usize).copied().unwrap_or(0) >= depth
    }

    pub(super) fn mark_local_moved(&self, state: &mut State, local: u32, site: Option<ExprId>) {
        if let Some(slot) = state.moved.get_mut(local as usize) {
            *slot = true;
        }
        if let Some(slot) = state.moved_at.get_mut(local as usize) {
            *slot = site.or(self.current_expr);
        }
        state.partial_moves.retain(|m| m.local != local);
    }

    pub(super) fn push_partial_move(
        &self,
        state: &mut State,
        local: u32,
        steps: Vec<PlaceStep>,
        site: Option<ExprId>,
    ) {
        let pm = PartialMove {
            local,
            steps,
            site: site.or(self.current_expr),
        };
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
                return Err(self.borrow_err(
                    Some(expr),
                    format!(
                        "cannot borrow or assign to immutable static item `{}`",
                        self.cx.name(item)
                    ),
                ));
            }
        }
        Ok(())
    }

    fn check_param_pat(
        &mut self,
        pat: PatId,
        fallback: &[BorrowSource],
        state: &mut State,
    ) -> Result<()> {
        self.bind_pat_param(pat, fallback, state)
    }

    fn check_return_escape(&mut self, expr: ExprId, state: &State) -> Result<()> {
        let had_cached = if let Some(cached) = self.temp_sources.get(&expr).cloned() {
            let site = self.return_diag_site(expr);
            self.check_return_sources(site, &cached)?;
            true
        } else {
            false
        };
        match *self.body.expr(expr) {
            Expr::Block {
                kind: BlockKind::Const,
                ..
            } => Ok(()),
            Expr::Loop { .. } | Expr::Block { tail: None, .. } if had_cached => Ok(()),
            Expr::Paren(inner)
            | Expr::Block {
                tail: Some(inner), ..
            } => self.check_return_escape(inner, state),
            Expr::If(_, then, otherwise) => {
                self.check_return_escape(then, state)?;
                if let Some(alt) = otherwise {
                    self.check_return_escape(alt, state)?;
                }
                Ok(())
            }
            Expr::Match(_, arms) => {
                for id in self.body.list(arms).to_vec() {
                    self.check_return_escape(self.body.arms[id as usize].body, state)?;
                }
                Ok(())
            }
            _ => {
                let sources = self.expr_borrow_sources(expr, state);
                self.check_return_sources(expr, &sources)
            }
        }
    }

    fn return_diag_site(&self, mut expr: ExprId) -> ExprId {
        loop {
            match *self.body.expr(expr) {
                Expr::Paren(inner)
                | Expr::Block {
                    tail: Some(inner), ..
                } => expr = inner,
                _ => return expr,
            }
        }
    }

    fn check_return_sources(&mut self, expr: ExprId, sources: &[BorrowSource]) -> Result<()> {
        if let Some(active) = self.active_closures.last_mut() {
            for src in sources {
                if let Some(idx) = active.params.iter().position(|&p| p == src.target)
                    && !active.ret_params.contains(&idx)
                {
                    active.ret_params.push(idx);
                }
            }
        }
        for src in sources {
            if self.is_escaping_return(src) {
                return Err(self.borrow_err(
                    Some(expr),
                    format!(
                        "cannot return value referencing local variable `{}`",
                        self.local_name(src.target)
                    ),
                ));
            }
            if self.closure_depth == 0
                && self.returns_only_static
                && self.is_non_static_param(src.target)
            {
                return Err(self.borrow_err(
                    Some(expr),
                    format!(
                        "borrowed data `{}` escapes outside of function: return type requires `'static` lifetime",
                        self.local_name(src.target)
                    ),
                ));
            }
        }
        Ok(())
    }

    fn check_block_escape(
        &self,
        block_depth: u32,
        state: &State,
        outer_live: &HashSet<u32>,
    ) -> Result<()> {
        for l in &state.loans {
            let Some(holder) = l.holder else { continue };
            if !is_temp_holder(Some(holder))
                && outer_live.contains(&holder)
                && self.escapes_depth(l.target, l.through_ref, block_depth)
            {
                return Err(self.does_not_live_long_enough(l.site, l.target));
            }
        }
        Ok(())
    }
}

fn join_moved(dst: &mut State, other: &State) {
    for (d, &o) in dst.init.iter_mut().zip(&other.init) {
        *d &= o;
    }
    for (d, &o) in dst.maybe_init.iter_mut().zip(&other.maybe_init) {
        *d |= o;
    }
    for (d, &o) in dst.moved.iter_mut().zip(&other.moved) {
        *d |= o;
    }
    for (d, &o) in dst.moved_at.iter_mut().zip(&other.moved_at) {
        if d.is_none() {
            *d = o;
        }
    }
    for pm in &other.partial_moves {
        if !dst.partial_moves.contains(pm) {
            dst.partial_moves.push(pm.clone());
        }
    }
}

fn merge_states(dst: &mut State, other: &State) {
    join_moved(dst, other);
    for l in &other.loans {
        if !dst.loans.contains(l) {
            dst.loans.push(l.clone());
        }
    }
    dst.tuple_elem_sources
        .retain(|k, ts| other.tuple_elem_sources.get(k).is_some_and(|o| ts.merge(o)));
}

fn merge_branch(out: &mut State, a: &State, b: &State) {
    match (a.diverged, b.diverged) {
        (true, true) => out.diverged = true,
        (true, false) => *out = b.clone(),
        (false, true) => *out = a.clone(),
        (false, false) => {
            *out = a.clone();
            merge_states(out, b);
        }
    }
}

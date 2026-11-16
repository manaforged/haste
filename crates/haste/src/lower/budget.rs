use std::sync::Arc;

use anyhow::Result;
use haste_codegen::{Function, FunctionBuilder, Type, Value};

use crate::body::{BinOp, BlockKind, Body, Expr, ExprId, Pat, PatId, Stmt, UnOp};
use crate::names::ItemRef;
use crate::sema::Context;
use crate::sema::infer::{Res, Shape, Typed, needs_drop};
use crate::sema::reach::Reached;
use crate::sema::substs::Substs;
use crate::sema::ty::Kind;

use super::expr::const_bool;
use super::plan::eval_footprint_with_cycle;
use super::reloc::{RelocPlan, RelocStep};
use super::stencil::{FootprintId, Form, InlinedCallee, LowerMode, Quiet, SymbolSource, Tally, simplify_tallied};
use super::{Inline, Lower, Slots, is_static, signature};

/// How large a sysroot body a call may copy inline under one build profile.
struct Budget {
    /// The most instructions a body may have, after slot promotion, for a call to copy it inline.
    insts: usize,
    /// The most instructions an `#[inline]` body may have, after slot promotion, for a call to copy it inline.
    inline_insts: usize,
    /// The most instructions an `#[inline(always)]` body may have for a call to copy it inline.
    always: usize,
    /// The most source expressions a function body may have for lowering it to try `insts`.
    exprs: usize,
    /// The most instructions an unhinted body that carries a loop may have for a compact inline body to copy it rather than call it.
    outline: usize,
    /// The fewest copies of one bulky callee in an inline body for lowering it again in compact form.
    copies: usize,
}

/// The budget of the dev profile.
const DEBUG_BUDGET: Budget = Budget { insts: 30, inline_insts: 30, always: 100, exprs: 20, outline: 30, copies: 2 };

/// The budget of the release profile.
const RELEASE_BUDGET: Budget = Budget { insts: 500, inline_insts: 900, always: 1200, exprs: 500, outline: 96, copies: 2 };

const CYCLE_BUDGET: usize = 12;

const SOLE_INLINE_INSTS: usize = 1600;

const SOLE_OUTLINE: usize = 64;

const FAST_PATH_MAX_EXPRS: usize = 32;

const FAST_PATH_MAX_INSTS: usize = 40;

/// The budget of the build profile.
fn budget() -> &'static Budget {
    match crate::profile() {
        crate::outline::tree::Profile::Debug => &DEBUG_BUDGET,
        crate::outline::tree::Profile::Release => &RELEASE_BUDGET,
    }
}

/// Whether the build profile is release.
fn release() -> bool {
    crate::profile() == crate::outline::tree::Profile::Release
}

/// Whether calls may copy bodies of the crate with `env` inline: the standard library's, and every crate under the release profile.
fn inlinable(env: &crate::outline::tree::Env) -> bool {
    env.release || release()
}

/// What an item's attributes ask of inlining.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Hint {
    Default,
    Inline,
    Always,
    Never,
}

/// The inlining hint of `item`: `#[inline(always)]`, `#[inline(never)]`/`#[cold]`, or `#[inline]`.
fn hint(cx: &Context<'_>, item: ItemRef) -> Hint {
    if cx.inline_always(item) {
        Hint::Always
    } else if cx.attr_value(item, "cold").is_some() || cx.attr_value(item, "inline") == Some("never") {
        Hint::Never
    } else if cx.attr_value(item, "inline").is_some() {
        Hint::Inline
    } else {
        Hint::Default
    }
}

fn is_leaf_body(body: &crate::body::Body) -> bool {
    !body.exprs.iter().any(|expr| matches!(expr, Expr::Loop { .. } | Expr::While { .. } | Expr::For { .. } | Expr::Await(..) | Expr::Yield(..)))
}

fn is_cold_or_never(cx: &Context<'_>, reached: &Reached, item: ItemRef, substs: Substs) -> bool {
    if hint(cx, item) == Hint::Never {
        return true;
    }
    reached
        .index
        .get(&(item, substs))
        .and_then(|&idx| reached.list[idx].1.as_ref().ok())
        .and_then(|callee| callee.ret)
        .is_some_and(|ret| matches!(cx.types.kind(ret), Kind::Never))
}

fn cold_calls(cx: &Context<'_>, reached: &Reached, typed: &Typed, body: &Body, cap: usize) -> usize {
    let cold = |&(item, substs): &(ItemRef, Substs)| is_cold_or_never(cx, reached, item, substs);
    let found = typed.calls().filter(cold).take(cap).count();
    if found < cap {
        return found;
    }
    let untaken = untaken_cold_calls(cx, reached, typed, body);
    typed.calls().filter(cold).take(cap + untaken).count().saturating_sub(untaken)
}

fn untaken_cold_calls(cx: &Context<'_>, reached: &Reached, typed: &Typed, body: &Body) -> usize {
    let untaken = |expr: &Expr| match *expr {
        Expr::If(cond, then, otherwise) => match (const_bool(body, cond)?, otherwise) {
            (false, _) => Some((cond.0, then.0)),
            (true, Some(otherwise)) => Some((then.0, otherwise.0)),
            (true, None) => None,
        },
        _ => None,
    };
    let spans: Vec<(u32, u32)> = body.exprs.iter().filter_map(untaken).collect();
    if spans.is_empty() {
        return 0;
    }
    let scoped = typed.scoped(&cx.types);
    let cold = |res: &Res| matches!(*res, Res::Fn(item, substs) if is_cold_or_never(cx, reached, item, substs));
    scoped.res.iter().filter(|&(at, res)| spans.iter().any(|&(after, last)| after < at.0 && at.0 <= last) && cold(res)).count()
}

fn allowed_cold(cx: &Context<'_>, reached: &Reached, typed: &Typed, body: &Body, hint: Hint) -> bool {
    if hint == Hint::Always {
        return true;
    }
    let lenient = release() && hint == Hint::Inline;
    let cap = if lenient { 7 } else { 3 };
    let cold = cold_calls(cx, reached, typed, body, cap);
    cold < cap && (lenient || cold < 2 || body.exprs.len() <= 80)
}

fn body_budget(cx: &Context<'_>, item: ItemRef, body: &Body, hint: Hint) -> Option<usize> {
    let (budget, release) = (budget(), release());
    let env_release = cx.outline.crates[item.krate as usize].env.release;
    match hint {
        Hint::Never => None,
        Hint::Always => Some(budget.always),
        Hint::Inline => {
            let max_exprs = if release { 1200 } else if env_release { 400 } else { 200 };
            let inst_limit = if release || env_release { budget.inline_insts } else { budget.insts };
            (body.exprs.len() <= max_exprs).then_some(inst_limit)
        }
        Hint::Default => {
            let max_exprs = if release && is_leaf_body(body) { budget.exprs * 2 } else { budget.exprs };
            (body.exprs.len() <= max_exprs).then_some(budget.insts)
        }
    }
}

/// The most instructions a call may copy inline of reached function `index`, or none when calls never inline it.
pub(super) fn inline_limit(cx: &mut Context<'_>, reached: &Reached, index: usize) -> Option<usize> {
    let ((item, _), Ok(typed)) = &reached.list[index] else {
        return None;
    };
    if cx.foreign(*item) {
        return None;
    }
    let hint = hint(cx, *item);
    if hint == Hint::Never || typed.ret.is_some_and(|ret| matches!(cx.types.kind(ret), Kind::Never)) {
        return None;
    }
    let env = &cx.outline.crates[item.krate as usize].env;
    if !inlinable(env) || crate::prep::bind(cx, reached, index).is_some() {
        return None;
    }
    if !matches!(typed.shape, Shape::Body) {
        return Some(budget().insts);
    }
    let body = cx.body(*item).ok()?;
    allowed_cold(cx, reached, typed, &body, hint).then(|| body_budget(cx, *item, &body, hint)).flatten()
}

pub(super) fn cycle_limit(cx: &mut Context<'_>, reached: &Reached, index: usize) -> Option<usize> {
    let ((item, _), Ok(typed)) = &reached.list[index] else {
        return None;
    };
    let short = !matches!(typed.shape, Shape::Body) || cx.body(*item).is_ok_and(|body| body.exprs.len() <= CYCLE_BUDGET);
    short.then_some(CYCLE_BUDGET)
}

/// The limit a leaf body of `limit` is measured against under the release profile: half as much again, up to the `#[inline]` budget.
pub(super) fn leaf_limit(limit: usize) -> usize {
    if release() { limit.saturating_add(limit / 2).min(budget().inline_insts).max(limit) } else { limit }
}

fn body_item(reached: &Reached, index: usize) -> Option<ItemRef> {
    match &reached.list[index] {
        ((item, _), Ok(typed)) if matches!(typed.shape, Shape::Body) => Some(*item),
        _ => None,
    }
}

fn callee_hint(cx: &Context<'_>, reached: &Reached, index: usize) -> Option<Hint> {
    body_item(reached, index).map(|item| hint(cx, item))
}

/// The limit the compact form of reached function `index` is measured against, when `limit` is the release budget of an `#[inline]` body.
pub(super) fn compact_limit(cx: &Context<'_>, reached: &Reached, index: usize, limit: usize) -> Option<usize> {
    let budget = budget();
    let hinted = release() && limit == budget.inline_insts && body_item(reached, index).is_some_and(|item| hint(cx, item) == Hint::Inline);
    hinted.then(|| if reached.is_sole_called(index) { SOLE_INLINE_INSTS } else { budget.always })
}

/// The form of reached function `index` before its callees count: a loop when its source body holds one.
pub(super) fn own_form(cx: &mut Context<'_>, reached: &Reached, index: usize) -> Form {
    let loops = body_item(reached, index).is_some_and(|item| cx.body(item).is_ok_and(|body| !is_leaf_body(&body)));
    if loops { Form::Loop } else { Form::Lean }
}

fn callee_bulky(cx: &Context<'_>, reached: &Reached, index: usize, callee: &InlinedCallee, sole: bool) -> (bool, Option<Hint>) {
    let hint = callee_hint(cx, reached, index);
    if !sole {
        let bulky = callee.form != Form::Lean && usize::from(callee.size) > budget().outline && hint == Some(Hint::Default);
        return (bulky, hint);
    }
    let outline = budget().outline.min(SOLE_OUTLINE);
    let bulky = callee.form != Form::Lean && usize::from(callee.size) > outline && matches!(hint, Some(Hint::Default | Hint::Inline));
    (bulky, hint)
}

/// What the inline body being lowered copies: whether it is the compact form, whether the host has one caller, the densest form among its callees, whether a compact form is worth lowering, and its bulky callees.
#[derive(Default)]
pub struct Copies {
    pub compact: bool,
    pub sole: bool,
    pub seen: Form,
    pub retry: bool,
    bulky: Vec<usize>,
}

impl Copies {
    fn repeated(&self, index: usize) -> bool {
        self.bulky.iter().filter(|&&at| at == index).count() >= budget().copies
    }

    fn compact_drop(&self, index: usize, hint: Option<Hint>) -> bool {
        if !self.sole {
            return true;
        }
        hint == Some(Hint::Default) || (!self.bulky.is_empty() && self.repeated(index))
    }

    /// Whether this host would drop bulky `callee` at `index` in compact mode.
    pub(super) fn would_drop(&self, cx: &Context<'_>, reached: &Reached, index: usize, callee: &InlinedCallee) -> bool {
        let (bulky, hint) = callee_bulky(cx, reached, index, callee, self.sole);
        bulky && self.compact && self.compact_drop(index, hint)
    }

    /// Records the copy `callee` of reached function `index` and returns whether the body takes it: a compact body leaves bulky callees out of line.
    pub(super) fn takes(&mut self, cx: &Context<'_>, reached: &Reached, index: usize, callee: &InlinedCallee) -> bool {
        let (bulky, hint) = callee_bulky(cx, reached, index, callee, self.sole);
        if bulky && self.compact && self.compact_drop(index, hint) {
            return false;
        }
        if bulky && !self.compact {
            self.bulky.push(index);
            self.retry |= self.repeated(index);
        }
        self.retry |= callee.form == Form::Compact || (self.sole && callee.form == Form::Dense);
        self.seen = self.seen.max(if bulky { Form::Dense } else { callee.form });
        true
    }
}

struct FastSplit {
    prelude: Vec<u32>,
    cond: ExprId,
    fast_on_true: bool,
    fast: FastBody,
}

enum FastBody {
    Expr(ExprId),
    Tail { stmts: Vec<u32>, tail: Option<ExprId> },
}

fn peel_block(body: &Body, mut expr: ExprId) -> ExprId {
    loop {
        match *body.expr(expr) {
            Expr::Paren(inner) => expr = inner,
            Expr::Block { stmts, tail: Some(tail), label: None, kind }
                if body.list(stmts).is_empty() && matches!(kind, BlockKind::Plain | BlockKind::Expansion | BlockKind::Unsafe) =>
            {
                expr = tail;
            }
            _ => return expr,
        }
    }
}

fn simple_pat(body: &Body, pat: PatId) -> bool {
    match *body.pat(pat) {
        Pat::Wild | Pat::Bind { sub: None, by_ref: false, .. } => true,
        Pat::Paren(inner) => simple_pat(body, inner),
        Pat::Tuple(items) => body.list(items).iter().all(|&item| simple_pat(body, PatId(item))),
        _ => false,
    }
}

fn no_drop(cx: &mut Context<'_>, typed: &Typed, expr: ExprId) -> bool {
    matches!(needs_drop(cx, typed.exprs[expr.0 as usize]), Ok(false))
}

fn pure_path(cx: &mut Context<'_>, typed: &Typed, expr: ExprId) -> bool {
    if !no_drop(cx, typed, expr) {
        return false;
    }
    matches!(typed.res.get(&expr), Some(Res::Local(_) | Res::Const(_) | Res::Ctor(..)))
}

fn pure_field(cx: &mut Context<'_>, body: &Body, typed: &Typed, (expr, base): (ExprId, ExprId), rem: &mut usize) -> bool {
    if !no_drop(cx, typed, expr) {
        return false;
    }
    let steps = typed.autoderefs.get(&expr).copied().unwrap_or(0);
    if steps > 1 {
        return false;
    }
    let base_ty = typed.exprs[base.0 as usize];
    if steps == 1 && !matches!(cx.types.kind(base_ty).clone(), Kind::Ref(_, inner) if matches!(cx.is_unsized(inner), Ok(false))) {
        return false;
    }
    pure_expr(cx, body, typed, base, rem)
}

fn pure_ctor_call(cx: &mut Context<'_>, body: &Body, typed: &Typed, (expr, callee, args): (ExprId, ExprId, crate::body::List), rem: &mut usize) -> bool {
    no_drop(cx, typed, expr)
        && body.names_callee(callee)
        && matches!(typed.res.get(&callee), Some(Res::Ctor(..)))
        && body.list(args).iter().all(|&arg| pure_expr(cx, body, typed, ExprId(arg), rem))
}

fn pure_data(cx: &mut Context<'_>, body: &Body, typed: &Typed, expr: ExprId, rem: &mut usize) -> bool {
    match *body.expr(expr) {
        Expr::Tuple(items) => no_drop(cx, typed, expr) && body.list(items).iter().all(|&item| pure_expr(cx, body, typed, ExprId(item), rem)),
        Expr::Struct { fields, base: None, .. } => {
            no_drop(cx, typed, expr) && body.list(fields).iter().all(|&at| pure_expr(cx, body, typed, body.fields[at as usize].value, rem))
        }
        Expr::Call(callee, args) => pure_ctor_call(cx, body, typed, (expr, callee, args), rem),
        Expr::Block { stmts, tail, label: None, kind } if matches!(kind, BlockKind::Plain | BlockKind::Expansion | BlockKind::Unsafe) => {
            pure_tail(cx, body, typed, (body.list(stmts), tail), rem)
        }
        Expr::If(cond, then_e, Some(else_e)) => {
            pure_expr(cx, body, typed, cond, rem) && pure_expr(cx, body, typed, then_e, rem) && pure_expr(cx, body, typed, else_e, rem)
        }
        _ => false,
    }
}

fn pure_unary(cx: &mut Context<'_>, body: &Body, typed: &Typed, (expr, op, inner): (ExprId, UnOp, ExprId), rem: &mut usize) -> bool {
    match op {
        UnOp::Neg | UnOp::Not => pure_expr(cx, body, typed, inner, rem),
        UnOp::Deref => {
            matches!(cx.types.kind(typed.exprs[inner.0 as usize]), Kind::Ref(..))
                && no_drop(cx, typed, expr)
                && pure_expr(cx, body, typed, inner, rem)
        }
    }
}

fn pure_expr(cx: &mut Context<'_>, body: &Body, typed: &Typed, expr: ExprId, rem: &mut usize) -> bool {
    if *rem == 0 || typed.coercions.contains_key(&expr) {
        return false;
    }
    *rem -= 1;
    let autoderef = typed.autoderefs.get(&expr).copied().unwrap_or(0);
    if matches!(*body.expr(expr), Expr::Field(..) | Expr::TupleField(..)) {
        return match *body.expr(expr) {
            Expr::Field(base, _) | Expr::TupleField(base, _) => pure_field(cx, body, typed, (expr, base), rem),
            _ => false,
        };
    }
    if autoderef != 0 {
        return false;
    }
    match *body.expr(expr) {
        Expr::Lit(crate::body::Lit::Bool(_) | crate::body::Lit::Char | crate::body::Lit::Int | crate::body::Lit::Byte, _) => true,
        Expr::Path(_) => pure_path(cx, typed, expr),
        Expr::Paren(inner) | Expr::Cast(inner, _) => pure_expr(cx, body, typed, inner, rem),
        Expr::Unary(op, inner) => pure_unary(cx, body, typed, (expr, op, inner), rem),
        Expr::Binary(op, lhs, rhs) => !matches!(op, BinOp::Div | BinOp::Rem) && pure_expr(cx, body, typed, lhs, rem) && pure_expr(cx, body, typed, rhs, rem),
        _ => pure_data(cx, body, typed, expr, rem),
    }
}

fn pure_stmt(cx: &mut Context<'_>, body: &Body, typed: &Typed, stmt: u32, rem: &mut usize) -> bool {
    match body.stmts[stmt as usize] {
        Stmt::Empty | Stmt::Item(..) => true,
        Stmt::Let { pat, init: Some(init), otherwise: None, .. } => simple_pat(body, pat) && pure_expr(cx, body, typed, init, rem),
        Stmt::Expr(expr, _) => pure_expr(cx, body, typed, expr, rem),
        _ => false,
    }
}

fn pure_tail(cx: &mut Context<'_>, body: &Body, typed: &Typed, (stmts, tail): (&[u32], Option<ExprId>), rem: &mut usize) -> bool {
    stmts.iter().all(|&stmt| pure_stmt(cx, body, typed, stmt, rem))
        && tail.is_none_or(|expr| pure_expr(cx, body, typed, expr, rem))
}

fn always_returns(body: &Body, expr: ExprId) -> bool {
    match *body.expr(peel_block(body, expr)) {
        Expr::Return(_) => true,
        Expr::Block { stmts, tail, label: None, kind } if matches!(kind, BlockKind::Plain | BlockKind::Expansion | BlockKind::Unsafe) => {
            body.list(stmts).iter().any(|&s| matches!(body.stmts[s as usize], Stmt::Expr(e, _) if always_returns(body, e)))
                || tail.is_some_and(|t| always_returns(body, t))
        }
        Expr::If(_, t, Some(e)) => always_returns(body, t) && always_returns(body, e),
        _ => false,
    }
}

fn pure_return(cx: &mut Context<'_>, body: &Body, typed: &Typed, expr: ExprId, rem: &mut usize) -> bool {
    if !always_returns(body, expr) {
        return false;
    }
    match *body.expr(peel_block(body, expr)) {
        Expr::Return(Some(val)) => pure_expr(cx, body, typed, val, rem),
        Expr::Return(None) => true,
        Expr::Block { stmts, tail, label: None, .. } => {
            let stmts = body.list(stmts);
            let Some((&last, init)) = stmts.split_last() else {
                return tail.is_some_and(|t| pure_return(cx, body, typed, t, rem));
            };
            if init.iter().any(|&s| !pure_stmt(cx, body, typed, s, rem)) {
                return false;
            }
            match (body.stmts[last as usize], tail) {
                (Stmt::Expr(e, _), None) => pure_return(cx, body, typed, e, rem),
                (_, Some(t)) => pure_stmt(cx, body, typed, last, rem) && pure_return(cx, body, typed, t, rem),
                _ => false,
            }
        }
        _ => false,
    }
}

fn root_stmts_tail(body: &Body, root: ExprId) -> (Vec<u32>, Option<ExprId>) {
    let mut cur = root;
    let mut prelude = Vec::new();
    loop {
        match *body.expr(cur) {
            Expr::Paren(inner) => cur = inner,
            Expr::Block { stmts, tail, label: None, kind } if matches!(kind, BlockKind::Plain | BlockKind::Expansion | BlockKind::Unsafe) => {
                prelude.extend_from_slice(body.list(stmts));
                let Some(next) = tail else { return (prelude, None); };
                if prelude.is_empty() || matches!(*body.expr(next), Expr::Block { .. } | Expr::Paren(_)) {
                    cur = next;
                    continue;
                }
                return (prelude, Some(next));
            }
            _ => return (prelude, Some(cur)),
        }
    }
}

fn fast_split(cx: &mut Context<'_>, body: &Body, typed: &Typed, root: ExprId) -> Option<FastSplit> {
    let (stmts, tail) = root_stmts_tail(body, root);
    let mut rem = FAST_PATH_MAX_EXPRS;
    if let Some(tail_id) = tail
        && let Expr::If(cond, then_e, Some(else_e)) = *body.expr(peel_block(body, tail_id))
        && stmts.iter().all(|&s| pure_stmt(cx, body, typed, s, &mut rem))
        && pure_expr(cx, body, typed, cond, &mut rem)
    {
        let mut t_rem = rem;
        let then_pure = pure_expr(cx, body, typed, then_e, &mut t_rem);
        let mut e_rem = rem;
        let else_pure = pure_expr(cx, body, typed, else_e, &mut e_rem);
        if then_pure != else_pure {
            return Some(FastSplit { prelude: stmts, cond, fast_on_true: then_pure, fast: FastBody::Expr(if then_pure { then_e } else { else_e }) });
        }
    }
    let mut pre_rem = FAST_PATH_MAX_EXPRS;
    for (idx, &stmt_id) in stmts.iter().enumerate() {
        if let Stmt::Expr(e, _) = body.stmts[stmt_id as usize]
            && let Expr::If(cond, then_e, None) = *body.expr(peel_block(body, e))
        {
            if !pure_expr(cx, body, typed, cond, &mut pre_rem) {
                return None;
            }
            let prelude = stmts[..idx].to_vec();
            let mut f_rem = pre_rem;
            if pure_return(cx, body, typed, then_e, &mut f_rem) {
                return Some(FastSplit { prelude, cond, fast_on_true: true, fast: FastBody::Expr(then_e) });
            }
            let rest = &stmts[idx + 1..];
            let mut r_rem = pre_rem;
            if pure_tail(cx, body, typed, (rest, tail), &mut r_rem) {
                return Some(FastSplit { prelude, cond, fast_on_true: false, fast: FastBody::Tail { stmts: rest.to_vec(), tail } });
            }
            return None;
        }
        if !pure_stmt(cx, body, typed, stmt_id, &mut pre_rem) {
            return None;
        }
    }
    None
}

fn eval_fast(lower: &mut Lower<'_, '_>, fast: &FastBody) -> Result<Option<Value>> {
    match fast {
        FastBody::Expr(expr) => lower.expr(*expr),
        FastBody::Tail { stmts, tail } => {
            for &stmt in stmts {
                lower.stmt(lower.body.stmts[stmt as usize])?;
            }
            match tail {
                Some(expr) => lower.expr(*expr),
                None => Ok(None),
            }
        }
    }
}

fn emit_slow_call(lower: &mut Lower<'_, '_>, index: usize, entry_params: &[Value]) {
    let table = lower.emit_reloc(SymbolSource::Slot(index), RelocStep::Const(SymbolSource::Slot(index)), false, lower.slots.low_entries());
    let code = lower.ins().load(Type::I64, table, 0);
    let sig = lower.b.import_signature(lower.b.func.signature.clone());
    let call = lower.ins().call_indirect(sig, code, entry_params);
    let results = lower.b.inst_results(call).to_vec();
    lower.ins().return_(&results);
    lower.close();
}

fn lower_pure_ok(lower: &Lower<'_, '_>) -> bool {
    lower.open && lower.owned.is_empty() && lower.flags.is_empty()
}

fn emit_partial(lower: &mut Lower<'_, '_>, index: usize, split: &FastSplit) -> Result<bool> {
    let entry = lower.b.create_block();
    lower.b.append_block_params_for_function_params(entry);
    let entry_params = lower.b.block_params(entry).to_vec();
    lower.b.switch_to_block(entry);
    let (prologue, start) = (lower.b.create_block(), lower.b.create_block());
    lower.jump(prologue, &[]);
    lower.b.switch_to_block(start);
    lower.moved = lower.moved_paths()?;
    lower.declare_params(entry)?;
    for &stmt in &split.prelude {
        lower.stmt(lower.body.stmts[stmt as usize])?;
    }
    let test = if lower_pure_ok(lower) { lower.expr(split.cond)? } else { None };
    let Some(test) = test.filter(|_| lower_pure_ok(lower)) else {
        return Ok(false);
    };
    let (fast_b, slow_b) = (lower.b.create_block(), lower.b.create_block());
    let (then_b, else_b) = if split.fast_on_true { (fast_b, slow_b) } else { (slow_b, fast_b) };
    lower.brif(test, (then_b, &[]), (else_b, &[]));
    lower.enter(fast_b);
    let value = eval_fast(lower, &split.fast)?;
    if !lower.owned.is_empty() || !lower.flags.is_empty() {
        return Ok(false);
    }
    if lower.open {
        lower.finish(value)?;
    }
    lower.enter(slow_b);
    if !lower.open {
        return Ok(false);
    }
    emit_slow_call(lower, index, &entry_params);
    lower.enter(prologue);
    lower.jump(start, &[]);
    lower.close();
    lower.seal();
    Ok(true)
}

fn lower_partial_func(cx: &mut Context<'_>, (reached, slots): (&Reached, &dyn Slots), inline: &mut Inline, (index, item, typed, body, split): (usize, ItemRef, &Typed, &Body, &FastSplit)) -> Result<Option<(Function, Vec<SymbolSource>, Vec<RelocStep>)>> {
    let mut func = Function::new(signature(cx, typed)?);
    let (mut relocs, mut reloc_plan) = (Vec::new(), Vec::new());
    let src = cx.body_src(item);
    let overflow_checks = cx.outline.crates[item.krate as usize].env.overflow_checks();
    let built = {
        let mut lower = Lower {
            cx,
            reached,
            slots,
            inline,
            body,
            src,
            typed,
            b: FunctionBuilder::new(&mut func),
            locals: vec![None; typed.locals.len()],
            loops: Vec::new(),
            try_exits: Vec::new(),
            open: true,
            dead: None,
            sret: None,
            dict_param: None,
            mode: LowerMode::Stencil,
            relocs: &mut relocs,
            reloc_plan: &mut reloc_plan,
            promoted_seq: 0,
            cycle_root: usize::MAX,
            owned: Vec::new(),
            flags: Vec::new(),
            moved: Default::default(),
            extended: Default::default(),
            pending: Vec::new(),
            index,
            resume: None,
            overflow_checks,
        };
        emit_partial(&mut lower, index, split)?
    };
    let ok = built && reloc_plan.iter().all(|step| matches!(step, RelocStep::Const(_)));
    Ok(ok.then_some((func, relocs, reloc_plan)))
}

fn partial_body(cx: &mut Context<'_>, (reached, slots): (&Reached, &dyn Slots), inline: &mut Inline, index: usize, fp: Option<FootprintId>) -> Result<Option<Arc<InlinedCallee>>> {
    let ((item, substs), typed_res) = &reached.list[index];
    if !substs.is_empty() || is_static(cx, *item) || inline_limit(cx, reached, index).is_none() {
        return Ok(None);
    }
    let Ok(typed) = typed_res else { return Ok(None) };
    let typed = typed.scoped(&cx.types);
    let typed: &Typed = &typed;
    let Shape::Body = typed.shape else { return Ok(None) };
    let body = cx.body(*item)?;
    let Some(split) = body.root.and_then(|root| fast_split(cx, &body, typed, root)) else {
        return Ok(None);
    };
    let Some(footprint) = resolve_partial_fp(cx, (reached, slots), inline, index, fp)? else {
        return Ok(None);
    };
    let Some((mut func, relocs, reloc_plan)) = lower_partial_func(cx, (reached, slots), inline, (index, *item, typed, &body, &split))? else {
        return Ok(None);
    };
    simplify_tallied(&mut func, Tally::Built);
    if !func.can_inline() || !func.has_return() || func.inline_size() > FAST_PATH_MAX_INSTS {
        return Ok(None);
    }
    let size = func.size().min(u16::MAX as usize) as u16;
    Ok(Some(Arc::new(InlinedCallee {
        item: *item,
        footprint,
        relocs,
        reloc_plan: Arc::new(RelocPlan::from_vec(reloc_plan)),
        body: Arc::new(func),
        returns: true,
        height: 0,
        size,
        form: Form::Lean,
    })))
}

fn resolve_partial_fp(cx: &mut Context<'_>, (reached, slots): (&Reached, &dyn Slots), inline: &mut Inline, index: usize, fp: Option<FootprintId>) -> Result<Option<FootprintId>> {
    if let Some(fp) = fp {
        return Ok(Some(fp));
    }
    let depth = inline.stack.len();
    let saved = (inline.max_callee_height, inline.max_callee_size);
    let (fp, cycle) = eval_footprint_with_cycle(cx, (reached, slots), inline, index)?;
    (inline.max_callee_height, inline.max_callee_size) = saved;
    Ok((cycle > depth).then_some(fp))
}

/// Returns a cached or freshly split `Form::Lean` partial inline body when `full` is missing or dropped.
pub(super) fn maybe_partial(cx: &mut Context<'_>, (reached, slots): (&Reached, &dyn Slots), inline: &mut Inline, (index, full): (usize, Option<Arc<InlinedCallee>>)) -> Result<Option<Arc<InlinedCallee>>> {
    let drop_full = full.as_ref().is_some_and(|c| inline.host.would_drop(cx, reached, index, c));
    if (full.is_some() && !drop_full) || inline.stack.contains(&index) {
        return Ok(full);
    }
    let key = (index, slots.low_entries());
    if let Some(cached) = slots.bodies().partial.get(&key) {
        return Ok(cached.or(full));
    }
    let quiet = Quiet::mark(cx, slots);
    let fp = full.as_ref().map(|c| c.footprint);
    let Some(built) = quiet.lowered(slots, partial_body(cx, (reached, slots), inline, index, fp))? else {
        return Ok(full);
    };
    let built = if quiet.holds(cx, slots) {
        slots.bodies().bump_transient();
        slots.bodies().partial.get_or_insert_with(key, || built)
    } else {
        built
    };
    Ok(built.or(full))
}

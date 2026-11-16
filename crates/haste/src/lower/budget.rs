use crate::body::{Body, Expr};
use crate::names::ItemRef;
use crate::sema::Context;
use crate::sema::infer::{Res, Shape, Typed};
use crate::sema::reach::Reached;
use crate::sema::substs::Substs;
use crate::sema::ty::Kind;

use super::expr::const_bool;
use super::stencil::{Form, InlinedCallee};

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
    let budget = budget();
    let release = release();
    if !matches!(typed.shape, Shape::Body) {
        return Some(budget.insts);
    }
    let Ok(body) = cx.body(*item) else {
        return None;
    };
    if hint != Hint::Always {
        let lenient = release && hint == Hint::Inline;
        let cap = if lenient { 7 } else { 3 };
        let cold = cold_calls(cx, reached, typed, &body, cap);
        if cold >= cap || (!lenient && cold >= 2 && body.exprs.len() > 80) {
            return None;
        }
    }
    match hint {
        Hint::Never => None,
        Hint::Always => Some(budget.always),
        Hint::Inline => {
            let max_exprs = if release { 1200 } else if env.release { 400 } else { 200 };
            let inst_limit = if release || env.release { budget.inline_insts } else { budget.insts };
            (body.exprs.len() <= max_exprs).then_some(inst_limit)
        }
        Hint::Default => {
            let max_exprs = if release && is_leaf_body(&body) { budget.exprs * 2 } else { budget.exprs };
            (body.exprs.len() <= max_exprs).then_some(budget.insts)
        }
    }
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
        self.retry |= callee.form == Form::Compact;
        self.seen = self.seen.max(if bulky { Form::Dense } else { callee.form });
        true
    }
}

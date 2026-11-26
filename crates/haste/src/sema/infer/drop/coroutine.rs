use rustc_hash::FxHashMap;

use crate::body::{BinOp, BlockKind, Body, Expr, ExprId, Pat, PatId, Stmt};

use super::super::super::ty::Ty;
use super::super::{BodyTables, Local, zonk::InferOut};

pub(in super::super) enum CoroutineSource<'a> {
    Infer(&'a InferOut),
    Typed(&'a BodyTables, &'a [Local]),
}

impl CoroutineSource<'_> {
    fn bind(&self, pat: PatId) -> Option<u32> {
        match self {
            Self::Infer(out) => out.binds.get(&pat).copied(),
            Self::Typed(tables, _) => tables.syntax.binds.get(&pat).copied(),
        }
    }

    fn local_ty(&self, local: u32) -> Option<Ty> {
        let locals = match self {
            Self::Infer(out) => &out.locals[..],
            Self::Typed(_, locals) => locals,
        };
        locals.get(local as usize).map(|l| l.ty)
    }

    fn await_ty(&self, id: ExprId) -> Option<Ty> {
        match self {
            Self::Infer(out) => out.awaits.get(&id).map(|a| a.future),
            Self::Typed(tables, _) => tables.awaits.get(&id).map(|a| a.future),
        }
    }

    fn for_ty(&self, id: ExprId) -> Option<Ty> {
        match self {
            Self::Infer(out) => out.fors.get(&id).map(|f| f.iter),
            Self::Typed(tables, _) => tables.fors.get(&id).map(|f| f.iter),
        }
    }
}

pub(in super::super) fn coroutine_body_tys(
    body: &Body,
    src: CoroutineSource<'_>,
    root: ExprId,
) -> Vec<Ty> {
    let mut awaits = FxHashMap::default();
    if !expr_contains_await(body, root, root, &mut awaits) {
        return Vec::new();
    }
    let (mut tys, mut exprs) = (Vec::new(), vec![root]);
    while let Some(id) = exprs.pop() {
        visit_coroutine_expr((body, &src, root), id, &mut awaits, &mut tys, &mut exprs);
    }
    tys
}

fn expr_contains_await(
    body: &Body,
    root: ExprId,
    id: ExprId,
    cache: &mut FxHashMap<ExprId, bool>,
) -> bool {
    if let Some(&known) = cache.get(&id) {
        return known;
    }
    let has = match *body.expr(id) {
        Expr::Await(_) => true,
        Expr::Closure { body: inner, .. } if id == root => {
            expr_contains_await(body, root, inner, cache)
        }
        Expr::Closure { .. } => false,
        Expr::Block { kind, .. }
            if id != root && matches!(kind, BlockKind::Async | BlockKind::AsyncMove) =>
        {
            false
        }
        Expr::Block { stmts, tail, .. } => {
            body.list(stmts)
                .iter()
                .any(|&s| match body.stmts[s as usize] {
                    Stmt::Let {
                        init, otherwise, ..
                    } => init
                        .into_iter()
                        .chain(otherwise)
                        .any(|e| expr_contains_await(body, root, e, cache)),
                    Stmt::Expr(e, _) => expr_contains_await(body, root, e, cache),
                    Stmt::Item(..) | Stmt::Macro { .. } | Stmt::Empty => false,
                })
                || tail.is_some_and(|t| expr_contains_await(body, root, t, cache))
        }
        Expr::For {
            iter, body: blk, ..
        } => {
            expr_contains_await(body, root, iter, cache)
                || expr_contains_await(body, root, blk, cache)
        }
        Expr::Let(_, init) => expr_contains_await(body, root, init, cache),
        Expr::Match(scrut, arms) => {
            expr_contains_await(body, root, scrut, cache)
                || body.list(arms).iter().any(|&a| {
                    let arm = body.arms[a as usize];
                    arm.guard
                        .is_some_and(|g| expr_contains_await(body, root, g, cache))
                        || expr_contains_await(body, root, arm.body, cache)
                })
        }
        _ => {
            let mut kids = Vec::new();
            push_sub_exprs(body, id, &mut kids);
            kids.into_iter()
                .any(|e| expr_contains_await(body, root, e, cache))
        }
    };
    cache.insert(id, has);
    has
}

fn push_sub_exprs(body: &Body, id: ExprId, out: &mut Vec<ExprId>) {
    match *body.expr(id) {
        Expr::Paren(a)
        | Expr::Unary(_, a)
        | Expr::Cast(a, _)
        | Expr::Field(a, _)
        | Expr::TupleField(a, _)
        | Expr::Try(a)
        | Expr::Become(a)
        | Expr::Ref { expr: a, .. }
        | Expr::Loop { body: a, .. } => out.push(a),
        Expr::Yield(opt) | Expr::Return(opt) | Expr::Break(_, opt) => out.extend(opt),
        Expr::Binary(_, a, b)
        | Expr::Assign(a, b)
        | Expr::AssignOp(_, a, b)
        | Expr::Index(a, b)
        | Expr::Repeat(a, b)
        | Expr::While {
            cond: a, body: b, ..
        } => out.extend([a, b]),
        Expr::Range { from, to, .. } => out.extend(from.into_iter().chain(to)),
        Expr::If(c, t, o) => out.extend([Some(c), Some(t), o].into_iter().flatten()),
        Expr::Call(callee, list)
        | Expr::Method {
            receiver: callee,
            params: list,
            ..
        } => {
            out.push(callee);
            out.extend(body.list(list).iter().copied().map(ExprId));
        }
        Expr::Tuple(list) | Expr::Array(list) => {
            out.extend(body.list(list).iter().copied().map(ExprId));
        }
        Expr::Struct { fields, base, .. } => {
            out.extend(
                body.list(fields)
                    .iter()
                    .map(|&f| body.fields[f as usize].value),
            );
            out.extend(base);
        }
        _ => {}
    }
}

fn visit_cond_expr(
    ctx @ (body, src, root): (&Body, &CoroutineSource<'_>, ExprId),
    cond: ExprId,
    body_has_await: bool,
    awaits: &mut FxHashMap<ExprId, bool>,
    tys: &mut Vec<Ty>,
    exprs: &mut Vec<ExprId>,
) {
    match *body.expr(cond) {
        Expr::Let(pat, init) => {
            if body_has_await {
                push_pat_bind_tys(body, src, pat, tys);
            }
            if expr_contains_await(body, root, init, awaits) {
                exprs.push(init);
            }
        }
        Expr::Binary(BinOp::And, a, b) => {
            let b_has = expr_contains_await(body, root, b, awaits);
            visit_cond_expr(ctx, a, body_has_await || b_has, awaits, tys, exprs);
            visit_cond_expr(ctx, b, body_has_await, awaits, tys, exprs);
        }
        Expr::Paren(inner) => visit_cond_expr(ctx, inner, body_has_await, awaits, tys, exprs),
        _ => {
            if expr_contains_await(body, root, cond, awaits) {
                exprs.push(cond);
            }
        }
    }
}

fn visit_coroutine_expr(
    ctx @ (body, src, root): (&Body, &CoroutineSource<'_>, ExprId),
    id: ExprId,
    awaits: &mut FxHashMap<ExprId, bool>,
    tys: &mut Vec<Ty>,
    exprs: &mut Vec<ExprId>,
) {
    match *body.expr(id) {
        Expr::Closure { body: inner, .. } if id == root => exprs.push(inner),
        Expr::Closure { .. } => {}
        Expr::Block { kind, .. }
            if id != root && matches!(kind, BlockKind::Async | BlockKind::AsyncMove) => {}
        Expr::Block { stmts, tail, .. } => {
            let mut await_after = tail.is_some_and(|t| expr_contains_await(body, root, t, awaits));
            if await_after {
                exprs.extend(tail);
            }
            for &s in body.list(stmts).iter().rev() {
                match body.stmts[s as usize] {
                    Stmt::Let {
                        pat,
                        init,
                        otherwise,
                        ..
                    } => {
                        if await_after {
                            push_pat_bind_tys(body, src, pat, tys);
                        }
                        for e in init.into_iter().chain(otherwise) {
                            if expr_contains_await(body, root, e, awaits) {
                                await_after = true;
                                exprs.push(e);
                            }
                        }
                    }
                    Stmt::Expr(e, _) => {
                        if expr_contains_await(body, root, e, awaits) {
                            await_after = true;
                            exprs.push(e);
                        }
                    }
                    Stmt::Item(..) | Stmt::Macro { .. } | Stmt::Empty => {}
                }
            }
        }
        Expr::Await(e) => {
            tys.extend(src.await_ty(id));
            if expr_contains_await(body, root, e, awaits) {
                exprs.push(e);
            }
        }
        Expr::For {
            pat,
            iter,
            body: blk,
            ..
        } => {
            if expr_contains_await(body, root, iter, awaits) {
                exprs.push(iter);
            }
            if expr_contains_await(body, root, blk, awaits) {
                tys.extend(src.for_ty(id));
                push_pat_bind_tys(body, src, pat, tys);
                exprs.push(blk);
            }
        }
        Expr::If(c, t, o) => {
            let then_has = expr_contains_await(body, root, t, awaits);
            visit_cond_expr(ctx, c, then_has, awaits, tys, exprs);
            if then_has {
                exprs.push(t);
            }
            if let Some(o) = o
                && expr_contains_await(body, root, o, awaits)
            {
                exprs.push(o);
            }
        }
        Expr::While {
            cond, body: blk, ..
        } => {
            let blk_has = expr_contains_await(body, root, blk, awaits);
            visit_cond_expr(ctx, cond, blk_has, awaits, tys, exprs);
            if blk_has {
                exprs.push(blk);
            }
        }
        Expr::Let(_, init) => {
            if expr_contains_await(body, root, init, awaits) {
                exprs.push(init);
            }
        }
        Expr::Match(scrut, arms) => {
            if expr_contains_await(body, root, scrut, awaits) {
                exprs.push(scrut);
            }
            for &arm_id in body.list(arms) {
                let arm = body.arms[arm_id as usize];
                let g_has = arm
                    .guard
                    .is_some_and(|g| expr_contains_await(body, root, g, awaits));
                let b_has = expr_contains_await(body, root, arm.body, awaits);
                if g_has || b_has {
                    push_pat_bind_tys(body, src, arm.pat, tys);
                    if g_has {
                        exprs.extend(arm.guard);
                    }
                    if b_has {
                        exprs.push(arm.body);
                    }
                }
            }
        }
        _ => {
            let mut kids = Vec::new();
            push_sub_exprs(body, id, &mut kids);
            for kid in kids {
                if expr_contains_await(body, root, kid, awaits) {
                    exprs.push(kid);
                }
            }
        }
    }
}

fn push_pat_bind_tys(body: &Body, src: &CoroutineSource<'_>, root: PatId, tys: &mut Vec<Ty>) {
    let mut stack = vec![root];
    while let Some(pat) = stack.pop() {
        if let Some(local) = src.bind(pat)
            && let Some(ty) = src.local_ty(local)
        {
            tys.push(ty);
        }
        match *body.pat(pat) {
            Pat::Bind { sub: Some(sub), .. }
            | Pat::Ref { pat: sub, .. }
            | Pat::Box(sub)
            | Pat::Paren(sub)
            | Pat::Deref(sub) => stack.push(sub),
            Pat::Tuple(l) | Pat::Slice(l) | Pat::Or(l) | Pat::TupleStruct(_, l) => {
                stack.extend(body.list(l).iter().copied().map(PatId));
            }
            Pat::Struct { fields, .. } => {
                stack.extend(
                    body.list(fields)
                        .iter()
                        .map(|&f| body.field_pats[f as usize].pat),
                );
            }
            _ => {}
        }
    }
}

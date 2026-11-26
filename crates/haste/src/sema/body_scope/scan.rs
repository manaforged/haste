use anyhow::{Result, anyhow};
use rustc_hash::FxHashMap as HashMap;

use crate::body::{self as syntax, Body, Stmt, sig};
use crate::outline::item::{Item, ItemKind, Items};
use crate::outline::scan;
use crate::outline::source::Source;

pub(in super::super) fn parse_in(
    src: &Source,
    config: syntax::Config<'_>,
    entry: &Item,
) -> Result<Body> {
    match (entry.kind, entry.name) {
        (ItemKind::Const | ItemKind::Static, Some(name)) => {
            let from = sig::initializer(src, name, entry.end)
                .ok_or_else(|| anyhow!("constant {} has no value", src.text(name)))?;
            syntax::parse_expr(src, from, entry.end - 1, config)
        }
        _ => syntax::parse(
            src,
            entry.body.ok_or_else(|| {
                anyhow!(
                    "function {} has no body",
                    src.text(entry.name.unwrap_or(entry.start))
                )
            })?,
            config,
        ),
    }
}

/// The items that `body` declares, and those that the bodies of its functions and constants declare in turn, with their block scopes.
pub(in super::super) fn declared(
    source: &Source,
    config: syntax::Config<'_>,
    body: &Body,
) -> Result<(Items, Vec<Option<u32>>, Vec<u32>)> {
    let mut items = Items::default();
    let mut scopes = vec![None];
    let mut item_scopes = Vec::new();
    let starts = body_starts(body, 0, &mut scopes)
        .into_iter()
        .map(|(at, scope)| (at, None, scope))
        .collect();
    scan_nested(
        source,
        config,
        starts,
        &mut items,
        &mut scopes,
        &mut item_scopes,
    )?;
    Ok((items, scopes, item_scopes))
}

pub(in super::super) fn scan_nested(
    source: &Source,
    config: syntax::Config<'_>,
    mut starts: Vec<(u32, Option<u32>, u32)>,
    items: &mut Items,
    scopes: &mut Vec<Option<u32>>,
    item_scopes: &mut Vec<u32>,
) -> Result<()> {
    while !starts.is_empty() {
        let first = items.items.len();
        let by_lead: HashMap<u32, u32> = starts.iter().map(|&(at, _, scope)| (at, scope)).collect();
        let pairs = std::mem::take(&mut starts)
            .into_iter()
            .map(|(at, owner, _)| (at, owner));
        *items = scan::body_items(source, pairs, (std::mem::take(items), config.calls))?;
        for entry in &items.items[first..] {
            let scope = by_lead
                .get(&entry.lead)
                .copied()
                .or_else(|| entry.parent.map(|parent| item_scopes[parent as usize]))
                .unwrap_or(0);
            item_scopes.push(scope);
        }
        for index in first..items.items.len() {
            let entry = &items.items[index];
            let has_body = matches!(entry.kind, ItemKind::Const | ItemKind::Static)
                || (entry.kind == ItemKind::Fn && entry.body.is_some());
            if has_body && let Ok(inner) = parse_in(source, config, entry) {
                starts.extend(
                    body_starts(&inner, item_scopes[index], scopes)
                        .into_iter()
                        .map(|(at, scope)| (at, Some(index as u32), scope)),
                );
            }
        }
    }
    Ok(())
}

fn body_starts(body: &Body, base: u32, scopes: &mut Vec<Option<u32>>) -> Vec<(u32, u32)> {
    let mut starts = Vec::new();
    if let Some(root) = body.root {
        ScopeWalk {
            body,
            scopes,
            starts: &mut starts,
        }
        .expr(root, base);
    }
    starts
}

struct ScopeWalk<'b> {
    body: &'b Body,
    scopes: &'b mut Vec<Option<u32>>,
    starts: &'b mut Vec<(u32, u32)>,
}

impl ScopeWalk<'_> {
    fn expr_list(&mut self, list: syntax::List, scope: u32) {
        for &id in self.body.list(list) {
            self.expr(syntax::ExprId(id), scope);
        }
    }

    fn expr(&mut self, id: syntax::ExprId, scope: u32) {
        use syntax::{BlockKind, Expr};
        match *self.body.expr(id) {
            Expr::Block {
                stmts, tail, kind, ..
            } => {
                let inner = if kind == BlockKind::Expansion {
                    scope
                } else {
                    let next = self.scopes.len() as u32;
                    self.scopes.push(Some(scope));
                    next
                };
                for &stmt in self.body.list(stmts) {
                    self.stmt(stmt, inner);
                }
                if let Some(tail) = tail {
                    self.expr(tail, inner);
                }
            }
            Expr::If(cond, then, otherwise) => {
                self.expr(cond, scope);
                self.expr(then, scope);
                if let Some(otherwise) = otherwise {
                    self.expr(otherwise, scope);
                }
            }
            Expr::Match(scrutinee, arms) => {
                self.expr(scrutinee, scope);
                for &arm in self.body.list(arms) {
                    let arm = self.body.arms[arm as usize];
                    self.pat(arm.pat, scope);
                    if let Some(guard) = arm.guard {
                        self.expr(guard, scope);
                    }
                    self.expr(arm.body, scope);
                }
            }
            other => self.expr_compound(other, scope),
        }
    }

    fn expr_compound(&mut self, expr: syntax::Expr, scope: u32) {
        use syntax::Expr;
        match expr {
            Expr::Unary(_, a)
            | Expr::Field(a, _)
            | Expr::TupleField(a, _)
            | Expr::Ref { expr: a, .. }
            | Expr::Loop { body: a, .. }
            | Expr::Try(a)
            | Expr::Await(a)
            | Expr::Become(a)
            | Expr::Paren(a) => {
                self.expr(a, scope);
            }
            Expr::Binary(_, a, b)
            | Expr::Assign(a, b)
            | Expr::AssignOp(_, a, b)
            | Expr::Index(a, b)
            | Expr::While {
                cond: a, body: b, ..
            }
            | Expr::Repeat(a, b) => {
                self.expr(a, scope);
                self.expr(b, scope);
            }
            Expr::Let(pat, init) => {
                self.pat(pat, scope);
                self.expr(init, scope);
            }
            Expr::For {
                pat, iter, body, ..
            } => {
                self.pat(pat, scope);
                self.expr(iter, scope);
                self.expr(body, scope);
            }
            Expr::Closure {
                params, ret, body, ..
            } => {
                for &param in self.body.list(params) {
                    let param = self.body.params[param as usize];
                    self.pat(param.pat, scope);
                    if let Some(ty) = param.ty {
                        self.ty(ty, scope);
                    }
                }
                if let Some(ret) = ret {
                    self.ty(ret, scope);
                }
                self.expr(body, scope);
            }
            other => self.expr_leaf(other, scope),
        }
    }

    fn expr_leaf(&mut self, expr: syntax::Expr, scope: u32) {
        use syntax::Expr;
        match expr {
            Expr::Call(callee, args) => {
                self.expr(callee, scope);
                self.expr_list(args, scope);
            }
            Expr::Method {
                receiver, params, ..
            } => {
                self.expr(receiver, scope);
                self.expr_list(params, scope);
            }
            Expr::Tuple(list) | Expr::Array(list) => self.expr_list(list, scope),
            Expr::Struct { fields, base, .. } => {
                for &field in self.body.list(fields) {
                    self.expr(self.body.fields[field as usize].value, scope);
                }
                if let Some(base) = base {
                    self.expr(base, scope);
                }
            }
            Expr::Range { from, to, .. } => {
                if let Some(from) = from {
                    self.expr(from, scope);
                }
                if let Some(to) = to {
                    self.expr(to, scope);
                }
            }
            Expr::Cast(inner, ty) => {
                self.expr(inner, scope);
                self.ty(ty, scope);
            }
            Expr::Return(Some(a)) | Expr::Break(_, Some(a)) | Expr::Yield(Some(a)) => {
                self.expr(a, scope)
            }
            _ => {}
        }
    }

    fn stmt(&mut self, index: u32, scope: u32) {
        match self.body.stmts[index as usize] {
            Stmt::Item(start, _) => self.starts.push((start, scope)),
            Stmt::Expr(expr, _) => self.expr(expr, scope),
            Stmt::Let {
                pat,
                ty,
                init,
                otherwise,
                ..
            } => {
                self.pat(pat, scope);
                if let Some(ty) = ty {
                    self.ty(ty, scope);
                }
                if let Some(init) = init {
                    self.expr(init, scope);
                }
                if let Some(otherwise) = otherwise {
                    self.expr(otherwise, scope);
                }
            }
            Stmt::Macro { .. } | Stmt::Empty => {}
        }
    }

    fn pat(&mut self, id: syntax::PatId, scope: u32) {
        use syntax::Pat;
        match *self.body.pat(id) {
            Pat::Lit(expr) => self.expr(expr, scope),
            Pat::Range { from, to, .. } => {
                if let Some(from) = from {
                    self.expr(from, scope);
                }
                if let Some(to) = to {
                    self.expr(to, scope);
                }
            }
            Pat::Bind {
                sub: Some(inner), ..
            }
            | Pat::Ref { pat: inner, .. }
            | Pat::Box(inner)
            | Pat::Paren(inner)
            | Pat::Deref(inner) => self.pat(inner, scope),
            Pat::TupleStruct(_, list) | Pat::Tuple(list) | Pat::Slice(list) | Pat::Or(list) => {
                for &pat in self.body.list(list) {
                    self.pat(syntax::PatId(pat), scope);
                }
            }
            Pat::Struct { fields, .. } => {
                for &field in self.body.list(fields) {
                    self.pat(self.body.field_pats[field as usize].pat, scope);
                }
            }
            _ => {}
        }
    }

    fn ty(&mut self, id: syntax::TyId, scope: u32) {
        use syntax::Ty;
        match *self.body.ty(id) {
            Ty::Array(inner, len) => {
                self.ty(inner, scope);
                self.expr(len, scope);
            }
            Ty::Const(expr) => self.expr(expr, scope),
            Ty::Ref { ty: inner, .. }
            | Ty::Ptr { ty: inner, .. }
            | Ty::Slice(inner)
            | Ty::Paren(inner)
            | Ty::Pat(inner, _) => self.ty(inner, scope),
            Ty::Tuple(list) => {
                for &ty in self.body.list(list) {
                    self.ty(syntax::TyId(ty), scope);
                }
            }
            _ => {}
        }
    }
}

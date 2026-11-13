use crate::body::{Body, Expr, ExprId, Pat, PatId, Stmt};

use super::{BodyFootprintPlan, FieldSite, PatSite, PatSiteKind};

pub(super) fn build_plan(body: &Body, closure_root: Option<ExprId>) -> BodyFootprintPlan {
    let (expr_ids, pat_ids) = collect_subtree(body, closure_root);
    let mut field_sites = Vec::new();
    let mut num_sites = Vec::new();
    let mut repeat_sites = Vec::new();
    let mut closure_exprs = Vec::new();
    let mut has_partial_moves = false;
    for &id in &expr_ids {
        match *body.expr(id) {
            Expr::TupleField(base, index) => {
                has_partial_moves = true;
                field_sites.push(FieldSite::TupleAccess { expr: id, base, index });
            }
            Expr::Field(base, _) => {
                has_partial_moves = true;
                field_sites.push(FieldSite::NamedAccess { expr: id, base });
            }
            Expr::Struct { fields, .. } => {
                let values = body.list(fields).iter().map(|&f| body.fields[f as usize].value).collect();
                field_sites.push(FieldSite::StructInit { expr: id, values });
            }
            Expr::Range { .. } => field_sites.push(FieldSite::RangeInit { expr: id }),
            Expr::Call(callee, args) => {
                let arg_ids = body.list(args).iter().map(|&a| ExprId(a)).collect();
                field_sites.push(FieldSite::CallSite { expr: id, callee, args: arg_ids });
            }
            Expr::Method { receiver, params, .. } => {
                let args = body.list(params).iter().map(|&a| ExprId(a)).collect();
                field_sites.push(FieldSite::MethodSite { expr: id, receiver, args });
            }
            Expr::Index(base, _) => field_sites.push(FieldSite::IndexSite { expr: id, base }),
            Expr::Unary(_, opnd) => {
                num_sites.push(id);
                num_sites.push(opnd);
            }
            Expr::Binary(_, lhs, rhs) | Expr::AssignOp(_, lhs, rhs) => {
                num_sites.extend([id, lhs, rhs]);
            }
            Expr::Cast(opnd, _) => {
                num_sites.extend([id, opnd]);
            }
            Expr::Repeat(val, _) => {
                if let Some(site) = super::super::array::uninit_call_site(body, val) {
                    repeat_sites.push(site);
                }
            }
            Expr::Ref { mutable, raw, expr: inner } => {
                field_sites.push(FieldSite::RefSite { inner, shared: !mutable && !raw });
            }
            Expr::Closure { .. } => closure_exprs.push(id),
            Expr::Match(..) | Expr::Let(..) => has_partial_moves = true,
            _ => {}
        }
    }
    let mut pat_sites = Vec::with_capacity(pat_ids.len());
    for pat in pat_ids {
        let kind = match *body.pat(pat) {
            Pat::Range { .. } => PatSiteKind::Range,
            Pat::Tuple(list) | Pat::TupleStruct(_, list) => {
                has_partial_moves = true;
                PatSiteKind::Tuple(list.len)
            }
            Pat::Slice(_) => {
                has_partial_moves = true;
                PatSiteKind::Slice
            }
            Pat::Struct { fields, .. } => {
                has_partial_moves = true;
                let sub_pats = body.list(fields).iter().map(|&f| body.field_pats[f as usize].pat).collect();
                PatSiteKind::Struct(sub_pats)
            }
            Pat::Box(_) => {
                has_partial_moves = true;
                PatSiteKind::Plain
            }
            _ => PatSiteKind::Plain,
        };
        pat_sites.push(PatSite { pat, kind });
    }
    BodyFootprintPlan {
        expr_ids: expr_ids.into_boxed_slice(),
        pat_sites: pat_sites.into_boxed_slice(),
        field_sites: field_sites.into_boxed_slice(),
        num_sites: num_sites.into_boxed_slice(),
        repeat_sites: repeat_sites.into_boxed_slice(),
        closure_exprs: closure_exprs.into_boxed_slice(),
        has_partial_moves,
    }
}

fn collect_subtree(body: &Body, closure_root: Option<ExprId>) -> (Vec<ExprId>, Vec<PatId>) {
    let Some(root) = closure_root else {
        let exprs = (0..body.exprs.len() as u32).map(ExprId).collect();
        let pats = (0..body.pats.len() as u32).map(PatId).collect();
        return (exprs, pats);
    };
    let mut exprs = Vec::new();
    let mut pats = Vec::new();
    for param in &body.params {
        walk_pat(body, param.pat, &mut exprs, &mut pats);
    }
    walk_expr(body, root, &mut exprs, &mut pats);
    (exprs, pats)
}

fn walk_expr(body: &Body, id: ExprId, exprs: &mut Vec<ExprId>, pats: &mut Vec<PatId>) {
    exprs.push(id);
    match *body.expr(id) {
        Expr::Unary(_, a) | Expr::Field(a, _) | Expr::TupleField(a, _) | Expr::Cast(a, _) | Expr::Ref { expr: a, .. } | Expr::Try(a) | Expr::Await(a) | Expr::Paren(a) | Expr::Become(a) => walk_expr(body, a, exprs, pats),
        Expr::Binary(_, a, b) | Expr::Assign(a, b) | Expr::AssignOp(_, a, b) | Expr::Index(a, b) | Expr::Repeat(a, b) | Expr::While { cond: a, body: b, .. } => {
            walk_expr(body, a, exprs, pats);
            walk_expr(body, b, exprs, pats);
        }
        Expr::Call(c, args) => {
            walk_expr(body, c, exprs, pats);
            body.list(args).iter().for_each(|&x| walk_expr(body, ExprId(x), exprs, pats));
        }
        Expr::Method { receiver, params, .. } => {
            walk_expr(body, receiver, exprs, pats);
            body.list(params).iter().for_each(|&x| walk_expr(body, ExprId(x), exprs, pats));
        }
        Expr::Block { stmts, tail, .. } => {
            for &s in body.list(stmts) {
                match body.stmts[s as usize] {
                    Stmt::Let { pat, init, otherwise, .. } => {
                        walk_pat(body, pat, exprs, pats);
                        init.into_iter().chain(otherwise).for_each(|e| walk_expr(body, e, exprs, pats));
                    }
                    Stmt::Expr(e, _) => walk_expr(body, e, exprs, pats),
                    _ => {}
                }
            }
            if let Some(t) = tail {
                walk_expr(body, t, exprs, pats);
            }
        }
        Expr::If(c, t, e) => {
            walk_expr(body, c, exprs, pats);
            walk_expr(body, t, exprs, pats);
            if let Some(e) = e {
                walk_expr(body, e, exprs, pats);
            }
        }
        Expr::Let(pat, e) => {
            walk_pat(body, pat, exprs, pats);
            walk_expr(body, e, exprs, pats);
        }
        Expr::Loop { body: b, .. } => walk_expr(body, b, exprs, pats),
        Expr::For { pat, iter, body: b, .. } => {
            walk_pat(body, pat, exprs, pats);
            walk_expr(body, iter, exprs, pats);
            walk_expr(body, b, exprs, pats);
        }
        Expr::Match(scrut, arms) => {
            walk_expr(body, scrut, exprs, pats);
            for &a in body.list(arms) {
                let arm = body.arms[a as usize];
                walk_pat(body, arm.pat, exprs, pats);
                if let Some(g) = arm.guard {
                    walk_expr(body, g, exprs, pats);
                }
                walk_expr(body, arm.body, exprs, pats);
            }
        }
        Expr::Closure { params, .. } => {
            for &p in body.list(params) {
                walk_pat(body, body.params[p as usize].pat, exprs, pats);
            }
        }
        Expr::Tuple(list) | Expr::Array(list) => body.list(list).iter().for_each(|&x| walk_expr(body, ExprId(x), exprs, pats)),
        Expr::Struct { fields, base, .. } => {
            body.list(fields).iter().for_each(|&f| walk_expr(body, body.fields[f as usize].value, exprs, pats));
            if let Some(b) = base {
                walk_expr(body, b, exprs, pats);
            }
        }
        Expr::Range { from, to, .. } => from.into_iter().chain(to).for_each(|e| walk_expr(body, e, exprs, pats)),
        Expr::Return(e) | Expr::Break(_, e) | Expr::Yield(e) => {
            if let Some(e) = e {
                walk_expr(body, e, exprs, pats);
            }
        }
        _ => {}
    }
}

fn walk_pat(body: &Body, id: PatId, exprs: &mut Vec<ExprId>, pats: &mut Vec<PatId>) {
    pats.push(id);
    match *body.pat(id) {
        Pat::Bind { sub: Some(s), .. } | Pat::Ref { pat: s, .. } | Pat::Box(s) | Pat::Paren(s) | Pat::Deref(s) => walk_pat(body, s, exprs, pats),
        Pat::Lit(e) => walk_expr(body, e, exprs, pats),
        Pat::Range { from, to, .. } => from.into_iter().chain(to).for_each(|e| walk_expr(body, e, exprs, pats)),
        Pat::TupleStruct(_, list) | Pat::Tuple(list) | Pat::Slice(list) | Pat::Or(list) => {
            body.list(list).iter().for_each(|&p| walk_pat(body, PatId(p), exprs, pats));
        }
        Pat::Struct { fields, .. } => {
            body.list(fields).iter().for_each(|&f| walk_pat(body, body.field_pats[f as usize].pat, exprs, pats));
        }
        _ => {}
    }
}

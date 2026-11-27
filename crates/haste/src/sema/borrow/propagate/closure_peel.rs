use rustc_hash::FxHashSet as HashSet;

use crate::body::{BlockKind, Expr, ExprId, Pat, PatId, Stmt, UnOp};
use crate::sema::infer::Res;
use crate::sema::ty::{Kind, Mut, Ty};

use super::super::Checker;

impl Checker<'_, '_> {
    pub(super) fn closure_peels_outer_ref(&self, arg: ExprId) -> bool {
        let cur = self.unwrap_parens(arg);
        let Expr::Closure { params, .. } = *self.body.expr(cur) else {
            return false;
        };
        let Some(param_local) = self
            .typed
            .closures
            .get(&cur)
            .filter(|c| c.params.len() == 1)
            .map(|c| c.params[0])
        else {
            return false;
        };
        let Kind::Ref(_, inner_ty) = *self.cx.types.kind(self.local_ty(param_local)) else {
            return false;
        };
        if !matches!(*self.cx.types.kind(inner_ty), Kind::Ref(Mut::Not, _)) {
            return false;
        }
        let param_ids = self.body.list(params);
        param_ids.len() == 1
            && matches!(
                *self.body.pat(self.body.params[param_ids[0] as usize].pat),
                Pat::Ref { .. }
            )
    }

    pub(super) fn expr_projects_local(&self, mut expr: ExprId) -> Option<u32> {
        loop {
            expr = self.unwrap_parens(expr);
            match *self.body.expr(expr) {
                Expr::Field(base, _)
                | Expr::TupleField(base, _)
                | Expr::Index(base, _)
                | Expr::Unary(UnOp::Deref, base)
                | Expr::Ref {
                    raw: false,
                    expr: base,
                    ..
                } => expr = base,
                _ => return self.expr_plain_local(expr),
            }
        }
    }

    pub(super) fn closure_copies_inner_ref(&mut self, arg: ExprId) -> bool {
        let cur = self.unwrap_parens(arg);
        let Expr::Closure { params, body, .. } = *self.body.expr(cur) else {
            return false;
        };
        let Some((param_local, ret_ty, no_captures)) = self
            .typed
            .closures
            .get(&cur)
            .filter(|c| c.params.len() == 1)
            .map(|c| (c.params[0], c.ret, c.captures.is_empty()))
        else {
            return false;
        };
        let param_ty = self.local_ty(param_local);
        let Kind::Ref(_, inner_ty) = *self.cx.types.kind(param_ty) else {
            return false;
        };
        let unwrapped_body = self.unwrap_parens_and_empty_blocks(body);
        if no_captures {
            if matches!(*self.cx.types.kind(inner_ty), Kind::Ref(Mut::Not, _)) {
                if let Expr::Unary(UnOp::Deref, inner) = *self.body.expr(unwrapped_body)
                    && self.expr_plain_local(inner) == Some(param_local)
                {
                    return true;
                }
                let param_ids = self.body.list(params);
                if param_ids.len() == 1
                    && let Pat::Ref { pat: inner_pat, .. } =
                        *self.body.pat(self.body.params[param_ids[0] as usize].pat)
                    && let Some(&bound) = self.typed.binds.get(&inner_pat)
                    && self.expr_plain_local(unwrapped_body) == Some(bound)
                {
                    return true;
                }
            }
            if matches!(
                *self.body.expr(unwrapped_body),
                Expr::Field(..) | Expr::TupleField(..) | Expr::Index(..)
            ) && self.expr_projects_local(unwrapped_body) == Some(param_local)
                && self.ty_is_copy(ret_ty)
            {
                return true;
            }
            if self.closure_body_clones_or_copies_local(
                unwrapped_body,
                &HashSet::from_iter([param_local]),
            ) {
                return true;
            }
        }
        if self.expr_has_return(body) {
            return false;
        }
        let mut outer_unpeeled = HashSet::default();
        outer_unpeeled.insert(param_local);
        let param_ids = self.body.list(params).to_vec();
        if param_ids.len() == 1 {
            self.collect_unpeeled_pat(
                self.body.params[param_ids[0] as usize].pat,
                false,
                &mut outer_unpeeled,
            );
        }
        self.expr_peels_outer_ref(body, &outer_unpeeled)
    }

    fn collect_unpeeled_pat(&mut self, pat: PatId, outer_peeled: bool, out: &mut HashSet<u32>) {
        match *self.body.pat(pat) {
            Pat::Bind { sub, .. } => {
                if !outer_peeled
                    && let Some(&local) = self.typed.binds.get(&pat)
                    && !self.ty_is_borrow_free(self.local_ty(local))
                {
                    out.insert(local);
                }
                if let Some(sub) = sub {
                    self.collect_unpeeled_pat(sub, outer_peeled, out);
                }
            }
            Pat::Paren(inner) | Pat::Box(inner) => {
                self.collect_unpeeled_pat(inner, outer_peeled, out);
            }
            Pat::Ref { pat: inner, .. } => {
                self.collect_unpeeled_pat(inner, true, out);
            }
            Pat::Tuple(list) | Pat::TupleStruct(_, list) | Pat::Slice(list) | Pat::Or(list) => {
                for &id in self.body.list(list) {
                    self.collect_unpeeled_pat(PatId(id), outer_peeled, out);
                }
            }
            Pat::Struct { fields, .. } => {
                for &id in self.body.list(fields) {
                    let p = self.body.field_pats[id as usize].pat;
                    self.collect_unpeeled_pat(p, outer_peeled, out);
                }
            }
            _ => {}
        }
    }

    fn expr_peels_outer_ref(&mut self, expr: ExprId, outer_unpeeled: &HashSet<u32>) -> bool {
        let cur = self.unwrap_parens(expr);
        let ty = self.expr_ty(cur);
        if ty != Ty::NONE && self.ty_is_borrow_free(ty) {
            return true;
        }
        match *self.body.expr(cur) {
            Expr::Cast(inner, _) => self.expr_peels_outer_ref(inner, outer_unpeeled),
            Expr::Block {
                kind: BlockKind::Plain | BlockKind::Unsafe,
                stmts,
                tail,
                ..
            } => {
                let mut block_outer = outer_unpeeled.clone();
                for &id in &self.body.list(stmts).to_vec() {
                    if let Stmt::Let { pat, init, .. } = self.body.stmts[id as usize] {
                        let peels_init =
                            init.is_some_and(|e| self.expr_peels_outer_ref(e, &block_outer));
                        if !peels_init {
                            self.collect_unpeeled_pat(pat, false, &mut block_outer);
                        }
                    }
                }
                tail.is_some_and(|t| self.expr_peels_outer_ref(t, &block_outer))
            }
            Expr::Tuple(list) | Expr::Array(list) => self
                .body
                .list(list)
                .to_vec()
                .into_iter()
                .all(|id| self.expr_peels_outer_ref(ExprId(id), outer_unpeeled)),
            Expr::Struct { fields, base, .. } => {
                let field_ok = self.body.list(fields).to_vec().into_iter().all(|id| {
                    let val = self.body.fields[id as usize].value;
                    self.expr_peels_outer_ref(val, outer_unpeeled)
                });
                field_ok && base.is_none_or(|b| self.expr_peels_outer_ref(b, outer_unpeeled))
            }
            Expr::Call(callee, args)
                if matches!(self.typed.res.get(&callee), Some(Res::Ctor(..)))
                    || matches!(self.typed.res.get(&cur), Some(Res::Ctor(..))) =>
            {
                self.body
                    .list(args)
                    .to_vec()
                    .into_iter()
                    .all(|id| self.expr_peels_outer_ref(ExprId(id), outer_unpeeled))
            }
            Expr::If(_, then_br, Some(else_br)) => {
                self.expr_peels_outer_ref(then_br, outer_unpeeled)
                    && self.expr_peels_outer_ref(else_br, outer_unpeeled)
            }
            Expr::Match(_, arms) => self.body.list(arms).to_vec().into_iter().all(|id| {
                let arm = self.body.arms[id as usize];
                let mut arm_outer = outer_unpeeled.clone();
                self.collect_unpeeled_pat(arm.pat, false, &mut arm_outer);
                self.expr_peels_outer_ref(arm.body, &arm_outer)
            }),
            Expr::Unary(UnOp::Deref, inner) => {
                let inner_ty = self.expr_ty(inner);
                let pointee = match *self.cx.types.kind(inner_ty) {
                    Kind::Ref(_, p) => Some(p),
                    _ => None,
                };
                ty != Ty::NONE
                    && self.ty_is_copy(ty)
                    && inner_ty != Ty::NONE
                    && pointee.is_some_and(|p| {
                        matches!(*self.cx.types.kind(p), Kind::Ref(Mut::Not, _))
                            || self.ty_is_borrow_free(p)
                    })
                    && self.expr_projects_any(inner, outer_unpeeled)
            }
            Expr::Field(..) | Expr::TupleField(..) | Expr::Index(..) => {
                ty != Ty::NONE
                    && self.ty_is_copy(ty)
                    && matches!(*self.cx.types.kind(ty), Kind::Ref(Mut::Not, _))
                    && self.expr_projects_any(cur, outer_unpeeled)
            }
            Expr::Path(_) => self
                .expr_plain_local(cur)
                .is_none_or(|local| !outer_unpeeled.contains(&local)),
            _ => self.closure_body_clones_or_copies_local(cur, outer_unpeeled),
        }
    }

    fn expr_projects_any(&self, expr: ExprId, locals: &HashSet<u32>) -> bool {
        self.expr_projects_local(expr)
            .is_some_and(|local| locals.contains(&local))
    }

    fn closure_body_clones_or_copies_local(&mut self, expr: ExprId, locals: &HashSet<u32>) -> bool {
        match *self.body.expr(expr) {
            Expr::Method { receiver, .. } => {
                let Some(Res::Fn(fn_item, _)) = self.typed.res.get(&expr).copied() else {
                    return false;
                };
                let (is_clone, is_copy_method) = {
                    let name = self.cx.name(fn_item);
                    (name == "clone", matches!(name, "copied" | "cloned"))
                };
                if is_clone {
                    return self.expr_projects_any(receiver, locals);
                }
                if is_copy_method
                    && (self.is_iter_mod(fn_item) || self.item_in_std_mod(fn_item, "option"))
                {
                    let mut cur = receiver;
                    for _ in 0..8 {
                        cur = self.unwrap_parens(cur);
                        match *self.body.expr(cur) {
                            Expr::Method { receiver: r, .. } | Expr::Try(r) => cur = r,
                            _ => break,
                        }
                    }
                    return self.expr_projects_any(cur, locals);
                }
                false
            }
            Expr::Call(callee, args) if self.body.list(args).len() == 1 => {
                let arg = ExprId(self.body.list(args)[0]);
                matches!(
                    self.typed
                        .res
                        .get(&callee)
                        .or_else(|| self.typed.res.get(&expr))
                        .copied(),
                    Some(Res::Fn(fn_item, _)) if self.cx.name(fn_item) == "clone"
                ) && self.expr_projects_any(arg, locals)
            }
            _ => false,
        }
    }

    pub(super) fn recv_borrow_sources_after_deref(
        &mut self,
        call_expr: ExprId,
        recv: ExprId,
        sig: &crate::body::sig::Signature,
        state: &super::super::State,
    ) -> Vec<super::BorrowSource> {
        let srcs = self.expr_borrow_sources(recv, state);
        let adj = self.typed.adjusts.get(&call_expr).copied();
        let derefs = adj.map_or(0, |a| a.derefs);
        if derefs == 0
            || adj.and_then(|a| a.autoref).is_some()
            || !matches!(sig.receiver, Some((crate::body::sig::SelfKind::Ref, _)))
        {
            return srcs;
        }
        let mut cur = self.expr_ty(recv);
        for step in 0..derefs {
            if cur == Ty::NONE {
                return srcs;
            }
            cur = if self.typed.derefs.contains_key(&(call_expr, step)) {
                self.deref_target_ty(call_expr, step).unwrap_or(Ty::NONE)
            } else {
                self.builtin_deref_ty(cur).unwrap_or(Ty::NONE)
            };
        }
        if cur != Ty::NONE && matches!(*self.cx.types.kind(cur), Kind::Ref(Mut::Not, _)) {
            self.unwrap_outer_container_loans_with_res(srcs, state, true, self.expr_ty(call_expr))
        } else {
            srcs
        }
    }

    pub(in super::super) fn unwrap_outer_container_loans_with_res(
        &mut self,
        sources: Vec<super::BorrowSource>,
        state: &super::super::State,
        unwrap_temp: bool,
        res_ty: Ty,
    ) -> Vec<super::BorrowSource> {
        let res_depth = self.ty_max_ref_depth(res_ty);
        let targets: Vec<u32> = sources.iter().map(|s| s.target).collect();
        let mut out = Vec::new();
        let mut rest = Vec::new();
        for s in sources {
            if !self.try_promote_by_val_param(&s, &targets, res_depth, state, &mut out) {
                rest.push(s);
            }
        }
        out.extend(self.unwrap_outer_container_loans_inner(
            rest,
            state,
            unwrap_temp,
            Some((res_ty, true)),
        ));
        super::dedup_sources(out)
    }

    pub(super) fn unwrap_ref_place_container_loan(
        &mut self,
        local: u32,
        steps: &[super::super::places::PlaceStep],
        loan: &super::super::Loan,
        mutable: bool,
        state: &super::super::State,
        out: &mut Vec<super::BorrowSource>,
    ) -> bool {
        let Kind::Ref(_, inner) = *self.cx.types.kind(self.local_ty(local)) else {
            return false;
        };
        if !self.is_ref_ty(inner)
            || steps.first() != Some(&super::super::places::PlaceStep::Deref)
            || self.from_caller(loan.target, loan.through_ref)
            || self.is_param_local(loan.target)
            || self.is_closure_param(loan.target)
            || !self.ty_holds_borrow(self.local_ty(loan.target))
        {
            return false;
        }
        for l in state.loans.iter().filter(|l| l.holder == Some(loan.target)) {
            out.push(super::BorrowSource {
                mutable: mutable && l.mutable,
                ..super::BorrowSource::from_loan(l)
            });
        }
        true
    }
}

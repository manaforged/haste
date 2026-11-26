use crate::body::{Expr, ExprId, UnOp};
use crate::sema::ty::{Kind, Ty};

use super::super::checks;
use super::super::places::{PlaceStep, push_step};
use super::super::{Checker, State, is_temp_holder};
use super::{BorrowSource, dedup_sources, state_holds_other_target};

impl Checker<'_, '_> {
    fn try_promote_by_val_param(
        &mut self,
        s: &BorrowSource,
        targets: &[u32],
        res_depth: u32,
        state: &State,
        out: &mut Vec<BorrowSource>,
    ) -> bool {
        let target_ty = self.local_ty(s.target);
        if s.through_ref
            || !self.is_param_local(s.target)
            || self.is_ref_ty(target_ty)
            || !self.ty_holds_borrow(target_ty)
            || state_holds_other_target(targets, s.target, state)
            || res_depth > self.ty_max_ref_depth(target_ty)
        {
            return false;
        }
        for l in state.loans.iter().filter(|l| l.holder == Some(s.target)) {
            out.push(BorrowSource::from_loan(l));
        }
        out.push(self.promote_param_behind_deref(s, false));
        true
    }

    pub(super) fn guard_take_recv_sources(
        &mut self,
        call_expr: ExprId,
        recv: ExprId,
        recv_sources: Vec<BorrowSource>,
        state: &State,
    ) -> Vec<BorrowSource> {
        let call_depth = self.ty_max_ref_depth(self.expr_ty(call_expr));
        let recv_local = self.expr_plain_local(self.unwrap_parens(recv));
        let targets: Vec<u32> = recv_sources.iter().map(|s| s.target).collect();
        let mut promoted = Vec::new();
        let mut rest = Vec::new();
        for s in recv_sources {
            let held_elsewhere = state.loans.iter().any(|l| {
                l.target == s.target
                    && l.holder
                        .is_some_and(|h| !is_temp_holder(Some(h)) && Some(h) != recv_local)
            });
            if recv_local.is_some()
                && !held_elsewhere
                && self.try_promote_by_val_param(&s, &targets, call_depth, state, &mut promoted)
            {
                continue;
            }
            rest.push(s);
        }
        let mut out = self.copied_recv_sources(call_expr, recv, rest, state, true);
        out.extend(promoted);
        dedup_sources(out)
    }

    pub(super) fn finish_projected_borrow_sources(
        &mut self,
        expr: ExprId,
        base_ty: Ty,
        unwrap_temp: bool,
        sources: Vec<BorrowSource>,
        state: &State,
    ) -> Vec<BorrowSource> {
        let is_deref = matches!(*self.body.expr(expr), Expr::Unary(UnOp::Deref, _));
        let expr_ty = self.expr_ty(expr);
        let base_is_ref = self.is_ref_ty(base_ty);
        let inner_static_res = base_is_ref.then_some((expr_ty, is_deref));
        let no_temps = sources
            .iter()
            .all(|s| checks::is_temp_target(s.target).is_none());
        let project_value_from_ref = base_is_ref
            && no_temps
            && !self.is_mut_ref_ty(expr_ty)
            && (!is_deref || !self.is_ref_ty(expr_ty));
        let targets: Vec<u32> = sources.iter().map(|s| s.target).collect();
        let direct_targets: Vec<u32> = sources
            .iter()
            .filter(|s| !s.through_ref)
            .map(|s| s.target)
            .collect();
        let expr_depth = self.ty_max_ref_depth(expr_ty);
        let mut extra = Vec::new();
        let mut to_unwrap = Vec::new();
        for mut s in sources {
            if is_deref
                && direct_targets.iter().all(|&t| t == s.target)
                && self.try_promote_by_val_param(&s, &targets, expr_depth, state, &mut extra)
            {
                continue;
            }
            let target_ty = self.local_ty(s.target);
            if project_value_from_ref
                && s.through_ref
                && self.from_caller(s.target, true)
                && let Kind::Ref(_, pointee) = *self.cx.types.kind(target_ty)
                && (self.ty_holds_borrow(pointee)
                    || matches!(self.cx.types.kind(pointee), Kind::Param(..)))
                && state.loans.iter().all(|l| l.holder != Some(s.target))
            {
                s.origin_only = true;
                s.mutable = false;
            }
            to_unwrap.push(s);
        }
        let mut out = self.unwrap_outer_container_loans_inner(
            to_unwrap,
            state,
            unwrap_temp,
            inner_static_res,
        );
        out.extend(extra);
        dedup_sources(out)
    }

    fn leading_ref_depth(&self, mut ty: Ty) -> u32 {
        let mut depth = 0;
        while ty != Ty::NONE
            && let Kind::Ref(_, inner) = *self.cx.types.kind(ty)
        {
            depth += 1;
            ty = inner;
        }
        depth
    }

    pub(in super::super) fn ty_max_ref_depth(&self, ty: Ty) -> u32 {
        if ty == Ty::NONE {
            return 0;
        }
        match *self.cx.types.kind(ty) {
            Kind::Ref(_, inner) => 1 + self.ty_max_ref_depth(inner),
            Kind::Slice(inner) | Kind::Array(inner, _) | Kind::Pat(inner, ..) => {
                self.ty_max_ref_depth(inner)
            }
            Kind::Tuple(items) | Kind::Adt(_, items) => self
                .cx
                .types
                .substs(items)
                .to_vec()
                .into_iter()
                .map(|t| self.ty_max_ref_depth(t))
                .max()
                .unwrap_or(0),
            _ => 0,
        }
    }

    pub(in super::super) fn can_unwrap_inner_static_param(
        &mut self,
        s: &BorrowSource,
        res_ty: Ty,
        is_deref: bool,
    ) -> bool {
        if !s.through_ref
            || !self.is_non_static_param(s.target)
            || !self.param_inner_is_static(s.target)
            || !s.steps.iter().all(|st| matches!(st, PlaceStep::Deref))
        {
            return false;
        }
        let local_ty = self.local_ty(s.target);
        if res_ty == local_ty {
            return false;
        }
        let Kind::Ref(_, pointee) = *self.cx.types.kind(local_ty) else {
            return false;
        };
        if !self.ty_holds_borrow(pointee) {
            return false;
        }
        if is_deref && self.leading_ref_depth(res_ty) >= self.leading_ref_depth(local_ty) {
            return false;
        }
        self.ty_max_ref_depth(res_ty) <= self.ty_max_ref_depth(pointee)
    }

    pub(in super::super) fn promote_param_behind_deref(
        &self,
        s: &BorrowSource,
        is_cparam: bool,
    ) -> BorrowSource {
        let mut steps = s.steps.clone();
        if self.is_ref_ty(self.local_ty(s.target)) {
            push_step(&mut steps, PlaceStep::Deref);
        }
        BorrowSource {
            steps,
            through_ref: true,
            origin_only: s.origin_only || !is_cparam,
            ..s.clone()
        }
    }

    pub(super) fn is_mut_ref_to_shared_ref(&self, ty: Ty) -> bool {
        ty != Ty::NONE
            && matches!(
                *self.cx.types.kind(ty),
                Kind::Ref(crate::sema::ty::Mut::Mut, inner)
                    if matches!(*self.cx.types.kind(inner), Kind::Ref(crate::sema::ty::Mut::Not, _))
            )
    }

    pub(super) fn filter_owned_closure_param_sources(
        &mut self,
        ret_ty: Ty,
        sources: &mut Vec<BorrowSource>,
    ) {
        if self.ty_holds_non_dyn_borrow(ret_ty) {
            return;
        }
        let mut kept = Vec::with_capacity(sources.len());
        for s in sources.drain(..) {
            let target_ty = self.local_ty(s.target);
            let drop_param = self.is_closure_param(s.target)
                && target_ty != Ty::NONE
                && self.closure_param_inner_has_no_ref(target_ty);
            if !drop_param {
                kept.push(s);
            }
        }
        *sources = kept;
    }

    fn closure_param_inner_has_no_ref(&mut self, ty: Ty) -> bool {
        match self.cx.types.kind(ty).clone() {
            Kind::Ref(_, pointee) => !self.ty_holds_non_dyn_borrow(pointee),
            Kind::Tuple(items) => {
                let elems = self.cx.types.substs(items).to_vec();
                elems.into_iter().all(|elem| {
                    let inner = match *self.cx.types.kind(elem) {
                        Kind::Ref(_, inner) => inner,
                        _ => elem,
                    };
                    !self.ty_holds_non_dyn_borrow(inner)
                })
            }
            _ => false,
        }
    }

    pub(super) fn is_static_field_read(&mut self, expr: ExprId, base: ExprId) -> bool {
        if self.typed.derefs.contains_key(&(expr, 0)) {
            return false;
        }
        let field_idx = match *self.body.expr(expr) {
            Expr::Field(..) => match self.typed.fields.get(&expr).copied() {
                Some(idx) => idx as usize,
                None => return false,
            },
            Expr::TupleField(_, idx) => idx as usize,
            _ => return false,
        };
        if self.is_mut_ref_ty(self.expr_ty(expr)) {
            return false;
        }
        let mut base_ty = self.expr_ty(base);
        while let Kind::Ref(_, inner) = *self.cx.types.kind(base_ty) {
            base_ty = inner;
        }
        let Kind::Adt(adt_item, _) = *self.cx.types.kind(base_ty) else {
            return false;
        };
        let Ok(def) = self.cx.adt_def(adt_item) else {
            return false;
        };
        if !def.generics.params.iter().any(|g| {
            matches!(
                g.kind,
                crate::body::sig::GenericKind::Lifetime | crate::body::sig::GenericKind::Type
            )
        }) {
            return true;
        }
        let Some(field) = def.variants.first().and_then(|v| v.fields.get(field_idx)) else {
            return false;
        };
        let adt_src = &self.cx.file(adt_item).source;
        matches!(
            *def.body.ty(field.ty),
            crate::body::Ty::Ref {
                mutable: false,
                lifetime: Some(lt),
                ..
            } if adt_src.text(lt) == "'static"
        )
    }

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
        if !matches!(
            *self.cx.types.kind(inner_ty),
            Kind::Ref(crate::sema::ty::Mut::Not, _)
        ) {
            return false;
        }
        let param_ids = self.body.list(params);
        param_ids.len() == 1
            && matches!(
                *self.body.pat(self.body.params[param_ids[0] as usize].pat),
                crate::body::Pat::Ref { .. }
            )
    }

    pub(super) fn closure_copies_inner_ref(&mut self, arg: ExprId) -> bool {
        let cur = self.unwrap_parens(arg);
        let Expr::Closure { params, body, .. } = *self.body.expr(cur) else {
            return false;
        };
        let Some((param_local, ret_ty)) = self
            .typed
            .closures
            .get(&cur)
            .filter(|c| c.params.len() == 1 && c.captures.is_empty())
            .map(|c| (c.params[0], c.ret))
        else {
            return false;
        };
        let param_ty = self.local_ty(param_local);
        let Kind::Ref(_, inner_ty) = *self.cx.types.kind(param_ty) else {
            return false;
        };
        let body = self.unwrap_parens_and_empty_blocks(body);
        if matches!(
            *self.cx.types.kind(inner_ty),
            Kind::Ref(crate::sema::ty::Mut::Not, _)
        ) {
            if let Expr::Unary(crate::body::UnOp::Deref, inner) = *self.body.expr(body)
                && self.expr_plain_local(inner) == Some(param_local)
            {
                return true;
            }
            let param_ids = self.body.list(params);
            if param_ids.len() == 1
                && let crate::body::Pat::Ref { pat: inner_pat, .. } =
                    *self.body.pat(self.body.params[param_ids[0] as usize].pat)
                && let Some(&bound) = self.typed.binds.get(&inner_pat)
                && self.expr_plain_local(body) == Some(bound)
            {
                return true;
            }
        }
        matches!(
            *self.body.expr(body),
            Expr::Field(base, _) | Expr::TupleField(base, _) | Expr::Index(base, _)
                if self.expr_plain_local(base) == Some(param_local)
        ) && self.ty_is_copy(ret_ty)
    }
}

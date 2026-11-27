use crate::body::sig::{GenericKind, Signature};
use crate::body::{Expr, ExprId, Ty as AstTy, TyId, UnOp};
use crate::names::ItemRef;
use crate::sema::infer::Res;
use crate::sema::ty::{Kind, Ty};

use super::super::calls::{Lifetimes, ast_ty_has_type_param, collect_ty_lifetimes};
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
        if !self.ty_holds_borrow(pointee) || self.is_mut_ref_ty(pointee) {
            return false;
        }
        if is_deref && self.leading_ref_depth(res_ty) >= self.leading_ref_depth(local_ty) {
            return false;
        }
        self.ty_max_ref_depth(res_ty) <= self.ty_max_ref_depth(pointee)
    }

    pub(in super::super) fn expr_derefs_ref(&self, expr: ExprId) -> bool {
        self.typed.adjusts.get(&expr).is_some_and(|a| a.derefs > 0)
            || matches!(
                self.typed
                    .coercions
                    .get(&expr)
                    .map(crate::sema::infer::Coercion::leaf),
                Some(crate::sema::infer::Coercion::Deref(..))
            )
    }

    pub(in super::super) fn is_peeled_static_shared_ref_param(
        &self,
        expr: ExprId,
        local: u32,
    ) -> bool {
        let Kind::Ref(_, inner) = *self.cx.types.kind(self.local_ty(local)) else {
            return false;
        };
        self.is_non_static_param(local)
            && self.param_inner_is_static(local)
            && self.expr_derefs_ref(expr)
            && matches!(
                *self.cx.types.kind(inner),
                Kind::Ref(crate::sema::ty::Mut::Not, _)
            )
    }

    pub(in super::super) fn promote_param_behind_deref(
        &self,
        s: &BorrowSource,
        is_cparam: bool,
    ) -> BorrowSource {
        let mut steps = s.steps.clone();
        if self.is_ref_ty(self.local_ty(s.target)) && s.steps.first() != Some(&PlaceStep::Deref) {
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

    fn peel_field_base_ty(&mut self, expr: ExprId, base: ExprId) -> Ty {
        let mut base_ty = self.expr_ty(base);
        let autoderefs = self.typed.autoderefs.get(&expr).copied().unwrap_or(0);
        for step in 0..autoderefs {
            base_ty = if self.typed.derefs.contains_key(&(expr, step)) {
                self.deref_target_ty(expr, step).unwrap_or(Ty::NONE)
            } else {
                self.builtin_deref_ty(base_ty).unwrap_or(Ty::NONE)
            };
            if base_ty == Ty::NONE {
                return Ty::NONE;
            }
        }
        while base_ty != Ty::NONE
            && let Kind::Ref(_, inner) = *self.cx.types.kind(base_ty)
        {
            base_ty = inner;
        }
        base_ty
    }

    pub(super) fn is_static_field_read(&mut self, expr: ExprId, base: ExprId) -> bool {
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
        let base_ty = self.peel_field_base_ty(expr, base);
        if base_ty == Ty::NONE {
            return false;
        }
        let Kind::Adt(adt_item, args) = *self.cx.types.kind(base_ty) else {
            return false;
        };
        let args_vec = self.cx.types.substs(args).to_vec();
        let arg_is_static: Vec<bool> = args_vec
            .iter()
            .map(|&arg| !self.ty_has_non_static_param(arg) && !self.ty_holds_non_dyn_borrow(arg))
            .collect();
        let Ok(def) = self.cx.adt_def(adt_item) else {
            return false;
        };
        if !def
            .generics
            .params
            .iter()
            .any(|g| matches!(g.kind, GenericKind::Lifetime | GenericKind::Type))
        {
            return true;
        }
        if def.variants.len() != 1 {
            return false;
        }
        let Some(field) = def.variants[0].fields.get(field_idx) else {
            return false;
        };
        let adt_src = &self.cx.file(adt_item).source;
        let mut f_lts = Lifetimes::default();
        collect_ty_lifetimes(&def.body, adt_src, field.ty, &mut f_lts);
        let mut ty_params = Vec::new();
        let mut arg_idx = 0usize;
        for g in &def.generics.params {
            if g.kind == GenericKind::Lifetime {
                continue;
            }
            if g.kind == GenericKind::Type && !arg_is_static.get(arg_idx).copied().unwrap_or(false)
            {
                ty_params.push(adt_src.text(g.name));
            }
            arg_idx += 1;
        }
        !f_lts.has_elided
            && f_lts.named.is_empty()
            && !ast_ty_has_type_param(&def.body, adt_src, field.ty, &ty_params)
    }

    pub(super) fn place_steps_ty(&mut self, mut cur: Ty, steps: &[PlaceStep]) -> Ty {
        for step in steps {
            if cur == Ty::NONE {
                break;
            }
            cur = if matches!(step, PlaceStep::Deref) {
                match *self.cx.types.kind(cur) {
                    Kind::Ref(_, inner) | Kind::Ptr(_, inner) => inner,
                    _ => self.cx.boxed(cur).unwrap_or(Ty::NONE),
                }
            } else {
                self.step_place_ty(cur, step)
            };
        }
        cur
    }

    pub(super) fn combine_closure_loan_source(
        &mut self,
        local: u32,
        l: &super::super::Loan,
        steps: &[PlaceStep],
        tail_steps: &[PlaceStep],
        mutable: bool,
    ) -> BorrowSource {
        let mut combined = l.steps.clone();
        if self.is_closure_param(l.target) {
            let target_ty = self.local_ty(l.target);
            let local_ty = self.local_ty(local);
            let extra = if target_ty != Ty::NONE
                && local_ty != Ty::NONE
                && steps.first() == Some(&PlaceStep::Deref)
                && self.place_steps_ty(target_ty, &l.steps) == local_ty
            {
                steps
            } else {
                tail_steps
            };
            for s in extra {
                push_step(&mut combined, s.clone());
            }
        }
        BorrowSource {
            steps: combined,
            mutable: mutable && l.mutable,
            ..BorrowSource::from_loan(l)
        }
    }

    pub(super) fn recv_is_static_field_read(&mut self, mut cur: ExprId) -> bool {
        for _ in 0..8 {
            cur = self.unwrap_parens(cur);
            match *self.body.expr(cur) {
                Expr::Field(base, _) | Expr::TupleField(base, _) => {
                    return self.is_static_field_read(cur, base);
                }
                Expr::Method { receiver, .. }
                | Expr::Ref {
                    raw: false,
                    expr: receiver,
                    ..
                }
                | Expr::Unary(UnOp::Deref, receiver)
                | Expr::Try(receiver) => cur = receiver,
                _ => return false,
            }
        }
        false
    }

    pub(super) fn unwrap_container_loans_deep(
        &mut self,
        sources: Vec<BorrowSource>,
        state: &State,
        initial_targets: &[u32],
    ) -> Vec<BorrowSource> {
        let mut seen = rustc_hash::FxHashSet::default();
        let mut cur = super::unwrap_container_loans(sources, state, |s| {
            initial_targets.contains(&s.target) && seen.insert(s.target)
        });
        for _ in 0..8 {
            let mut grew = false;
            cur = super::unwrap_container_loans(cur, state, |s| {
                let should = !self.from_caller(s.target, s.through_ref)
                    && !self.is_param_local(s.target)
                    && !self.is_closure_param(s.target)
                    && checks::is_temp_target(s.target).is_none()
                    && self.ty_holds_borrow(self.local_ty(s.target))
                    && state.loans.iter().any(|l| l.holder == Some(s.target))
                    && seen.insert(s.target);
                grew |= should;
                should
            });
            if !grew {
                break;
            }
        }
        cur
    }

    pub(super) fn path_deref_coercion_sources(
        &mut self,
        expr: ExprId,
        local: u32,
        sources: Vec<BorrowSource>,
        state: &State,
    ) -> Vec<BorrowSource> {
        let local_ty = self.local_ty(local);
        let has_builtin_deref = self.is_ref_ty(local_ty)
            && matches!(
                self.typed
                    .coercions
                    .get(&expr)
                    .map(crate::sema::infer::Coercion::leaf),
                Some(crate::sema::infer::Coercion::Deref(steps, _))
                    if steps.iter().any(|(_, res)| res.is_none())
            );
        if has_builtin_deref {
            self.finish_projected_borrow_sources(expr, local_ty, true, sources, state)
        } else {
            sources
        }
    }

    pub(super) fn recv_independent_of_named_ret(
        &mut self,
        fn_item: ItemRef,
        sig: &Signature,
        sig_src: &crate::outline::source::Source,
        ret_lts: &Lifetimes<'_>,
    ) -> bool {
        if ret_lts.has_elided || ret_lts.named.is_empty() || ret_lts.names.contains(&"Self") {
            return false;
        }
        let owner_hdr = if let Some(blk) = self.cx.owner_impl(fn_item)
            && let Ok(hdr) = self.cx.impl_header(blk)
        {
            Some((&self.cx.impl_file(blk).source, hdr.generics.clone()))
        } else if let Some(tr) = self.cx.owner_trait(fn_item)
            && let Ok(tg) = self.cx.trait_generics(tr)
        {
            Some((&self.cx.file(tr).source, tg.1.params.clone()))
        } else {
            None
        };
        let Some((h_src, h_generics)) = owner_hdr else {
            return true;
        };
        let owner_has_tied_lt = h_generics.iter().any(|g| match g.kind {
            GenericKind::Lifetime => {
                let name = h_src.text(g.name);
                ret_lts.named.contains(&name)
                    || ret_lts
                        .named
                        .iter()
                        .any(|&r| super::super::calls::lifetime_outlives(sig, sig_src, name, r))
            }
            GenericKind::Type => ret_lts.names.contains(&h_src.text(g.name)),
            _ => false,
        });
        !owner_has_tied_lt
    }

    pub(super) fn expr_is_any_downcast(&mut self, mut expr: ExprId, state: &State) -> bool {
        for _ in 0..6 {
            expr = self.unwrap_parens(expr);
            match *self.body.expr(expr) {
                Expr::Try(inner) => expr = inner,
                Expr::Method { receiver, .. } => {
                    let Some(Res::Fn(fn_item, _)) = self.typed.res.get(&expr).copied() else {
                        return false;
                    };
                    let name = self.cx.name(fn_item);
                    if self.item_in_std_mod(fn_item, "any")
                        && matches!(name, "downcast_mut" | "downcast_ref")
                    {
                        return true;
                    }
                    if (self.item_in_std_mod(fn_item, "option")
                        || self.item_in_std_mod(fn_item, "result"))
                        && matches!(
                            name,
                            "unwrap" | "expect" | "unwrap_unchecked" | "as_deref" | "as_deref_mut"
                        )
                    {
                        expr = receiver;
                        continue;
                    }
                    return false;
                }
                Expr::Path(_) => {
                    let Some(local) = self.expr_plain_local(expr) else {
                        return false;
                    };
                    let Some(site) = state
                        .loans
                        .iter()
                        .find_map(|l| (l.holder == Some(local)).then_some(l.site).flatten())
                    else {
                        return false;
                    };
                    if site == expr {
                        return false;
                    }
                    expr = site;
                }
                _ => return false,
            }
        }
        false
    }

    fn generics_have_type_param(
        generics: &[crate::body::sig::Generic],
        src: &crate::outline::source::Source,
        name: &str,
    ) -> bool {
        generics
            .iter()
            .any(|g| g.kind == GenericKind::Type && src.text(g.name) == name)
    }

    pub(super) fn sig_returns_type_param(
        &mut self,
        fn_item: ItemRef,
        sig: &Signature,
        ret_ast: TyId,
    ) -> bool {
        let mut cur = ret_ast;
        while let AstTy::Paren(inner) = *sig.body.ty(cur) {
            cur = inner;
        }
        let AstTy::Path(path_id) = *sig.body.ty(cur) else {
            return false;
        };
        let path = sig.body.path(path_id);
        let segs = sig.body.list(path.segments);
        if path.qself.is_some() || segs.len() != 1 {
            return false;
        }
        let seg = sig.body.segments[segs[0] as usize];
        if seg.args.is_some() {
            return false;
        }
        let name = self.cx.item(fn_item).0.text(seg.name);
        if name == "Self"
            || Self::generics_have_type_param(&sig.generics, self.cx.item(fn_item).0, name)
        {
            return true;
        }
        if let Some(blk) = self.cx.owner_impl(fn_item)
            && let Ok(hdr) = self.cx.impl_header(blk)
            && Self::generics_have_type_param(&hdr.generics, &self.cx.impl_file(blk).source, name)
        {
            return true;
        }
        self.cx.owner_trait(fn_item).is_some_and(|tr| {
            self.cx.trait_generics(tr).is_ok_and(|tg| {
                Self::generics_have_type_param(&tg.1.params, &self.cx.file(tr).source, name)
            })
        })
    }

    pub(super) fn is_copied_container_target(
        &mut self,
        s: &BorrowSource,
        (call_ty, call_owned, unwrap): (Ty, bool, bool),
        (root, indirect, targets): (Option<u32>, bool, &[u32]),
        state: &State,
    ) -> bool {
        let held_by_other = state_holds_other_target(targets, s.target, state);
        if self.is_non_static_param(s.target) || self.is_closure_param(s.target) {
            if held_by_other {
                return false;
            }
            let target_ty = self.local_ty(s.target);
            let pointee = (target_ty != Ty::NONE)
                .then(|| match *self.cx.types.kind(target_ty) {
                    Kind::Ref(_, pointee) => Some(pointee),
                    _ => None,
                })
                .flatten();
            if pointee.is_none()
                && !s.through_ref
                && unwrap
                && call_owned
                && target_ty != Ty::NONE
                && !self.ty_holds_non_dyn_borrow(target_ty)
                && !self.ty_has_non_static_param(call_ty)
            {
                return true;
            }
            return pointee.is_some_and(|p| {
                (!unwrap && self.ty_is_borrow_free(p) && (call_owned || Some(s.target) == root))
                    || (((unwrap && !self.is_ref_ty(call_ty)) || call_owned)
                        && !self.ty_holds_non_dyn_borrow(p))
            }) || self.can_unwrap_inner_static_param(s, call_ty, false);
        }
        if checks::is_temp_target(s.target).is_some()
            || (Some(s.target) == root
                && (!self.is_ref_ty(call_ty) || self.ty_holds_borrow(self.local_ty(s.target))))
        {
            return true;
        }
        root.filter(|_| indirect).is_some_and(|r| {
            state
                .loans
                .iter()
                .any(|l| l.holder == Some(r) && l.target == s.target)
                && self.ty_holds_borrow(self.local_ty(s.target))
                && !held_by_other
        })
    }

    pub(super) fn copied_recv_root(
        &mut self,
        call_expr: ExprId,
        recv: ExprId,
    ) -> (Option<u32>, bool) {
        let peel_ref = !self.is_ref_ty(self.expr_ty(call_expr));
        let mut cur = recv;
        loop {
            match *self.body.expr(cur) {
                Expr::Paren(inner)
                | Expr::Method {
                    receiver: inner, ..
                }
                | Expr::Field(inner, _)
                | Expr::TupleField(inner, _)
                | Expr::Index(inner, _)
                | Expr::Unary(UnOp::Deref, inner)
                | Expr::Try(inner) => cur = inner,
                Expr::Ref {
                    raw: false,
                    expr: inner,
                    ..
                } if peel_ref => cur = inner,
                _ => break,
            }
        }
        let root = self.root_place(cur).map(|p| p.local);
        let indirect = root.is_some_and(|r| {
            let r_ty = self.local_ty(r);
            cur == recv || self.is_ref_ty(r_ty) || self.ty_holds_closure(r_ty)
        });
        (root, indirect)
    }

    pub(super) fn finish_unwrapped_copied_sources(
        &mut self,
        call_ty: Ty,
        recv_is_mut_ref: bool,
        root: Option<u32>,
        out: Vec<BorrowSource>,
        state: &State,
    ) -> Vec<BorrowSource> {
        let mut expanded = Vec::with_capacity(out.len());
        let is_ref_call = self.is_ref_ty(call_ty);
        for s in out {
            if !is_ref_call
                && self.is_closure_param(s.target)
                && let Some(origins) = self.closure_param_origins.get(&s.target).cloned()
            {
                expanded.extend(self.unwrap_outer_container_loans(origins, state, true));
                continue;
            }
            expanded.push(s);
        }
        let mut expanded = dedup_sources(expanded);
        let call_depth = self.ty_max_ref_depth(call_ty);
        let clears_mut = !self.ty_holds_mut_ref(call_ty);
        for s in &mut expanded {
            if self.is_param_local(s.target) || self.is_closure_param(s.target) {
                let target_ty = self.local_ty(s.target);
                if target_ty != Ty::NONE {
                    let inner = match *self.cx.types.kind(target_ty) {
                        Kind::Ref(_, pointee) => pointee,
                        _ => target_ty,
                    };
                    if self.ty_holds_borrow(inner)
                        && (call_depth <= self.ty_max_ref_depth(inner)
                            || (Some(s.target) != root && recv_is_mut_ref && clears_mut))
                    {
                        s.origin_only = true;
                        s.mutable = false;
                    }
                }
            }
            if clears_mut {
                s.mutable = false;
            }
        }
        expanded
    }

    pub(super) fn unwrap_ref_place_container_loan(
        &mut self,
        local: u32,
        steps: &[PlaceStep],
        loan: &super::super::Loan,
        mutable: bool,
        state: &State,
        out: &mut Vec<BorrowSource>,
    ) -> bool {
        let Kind::Ref(_, inner) = *self.cx.types.kind(self.local_ty(local)) else {
            return false;
        };
        if !self.is_ref_ty(inner)
            || steps.first() != Some(&PlaceStep::Deref)
            || self.from_caller(loan.target, loan.through_ref)
            || self.is_param_local(loan.target)
            || self.is_closure_param(loan.target)
            || !self.ty_holds_borrow(self.local_ty(loan.target))
        {
            return false;
        }
        for l in state.loans.iter().filter(|l| l.holder == Some(loan.target)) {
            out.push(BorrowSource {
                mutable: mutable && l.mutable,
                ..BorrowSource::from_loan(l)
            });
        }
        true
    }
}

use rustc_hash::FxHashSet as HashSet;

use crate::body::sig::SelfKind;
use crate::body::{BlockKind, Expr, ExprId, List, Ty as AstTy, TyId, UnOp};
use crate::names::ItemRef;
use crate::sema::infer::Res;
use crate::sema::ty::{Kind, Mut, Ty};

use super::calls::{Lifetimes, collect_ty_lifetimes, receiver_has_any_named, recv_is_by_ref, recv_ref_mut, ty_shares_name};
use super::places::{PlaceStep, steps_conflict};
use super::{Checker, Loan, State};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct BorrowSource {
    pub(super) target: u32,
    pub(super) steps: Vec<PlaceStep>,
    pub(super) mutable: bool,
    pub(super) through_ref: bool,
    pub(super) env_only: bool,
}

impl BorrowSource {
    pub(super) fn from_loan(l: &Loan) -> Self {
        Self {
            target: l.target,
            steps: l.steps.clone(),
            mutable: l.mutable,
            through_ref: l.through_ref,
            env_only: l.env_only,
        }
    }

    pub(super) fn temp(depth: u32, mutable: bool, env_only: bool) -> Self {
        Self {
            target: super::checks::temp_target_for(depth),
            steps: Vec::new(),
            mutable,
            through_ref: false,
            env_only,
        }
    }
}

impl Checker<'_, '_> {
    pub(super) fn from_caller(&self, target: u32, through_ref: bool) -> bool {
        self.is_param.get(target as usize).copied().unwrap_or(false)
            && (through_ref || self.is_ref_ty(self.local_ty(target)))
    }

    pub(super) fn expr_borrow_sources(&mut self, expr: ExprId, state: &State) -> Vec<BorrowSource> {
        self.expr_borrow_sources_inner(expr, state, true, None)
    }

    pub(super) fn expr_borrow_sources_let(&mut self, expr: ExprId, state: &State) -> Vec<BorrowSource> {
        let extend_depth = self.let_extend_depth;
        self.expr_borrow_sources_inner(expr, state, true, extend_depth)
    }

    fn expr_borrow_sources_inner(&mut self, expr: ExprId, state: &State, allow_mut_ref_reborrow: bool, extend_depth: Option<u32>) -> Vec<BorrowSource> {
        let ty = self.expr_ty(expr);
        if ty != Ty::NONE && !self.ty_holds_borrow(ty) {
            return Vec::new();
        }
        match *self.body.expr(expr) {
            Expr::Paren(inner) | Expr::Cast(inner, _) => {
                self.expr_borrow_sources_inner(inner, state, allow_mut_ref_reborrow, extend_depth)
            }
            Expr::Try(inner) => {
                self.expr_borrow_sources_inner(inner, state, allow_mut_ref_reborrow, None)
            }
            Expr::Ref { mutable, raw: false, expr: inner } => {
                if self.is_raw_ptr_deref(inner) {
                    return Vec::new();
                }
                if allow_mut_ref_reborrow {
                    self.place_borrow_sources(inner, mutable, state, extend_depth)
                } else {
                    self.expr_borrow_sources_inner(inner, state, false, None)
                }
            }
            Expr::Field(..) | Expr::TupleField(..) | Expr::Unary(UnOp::Deref, _)
                if allow_mut_ref_reborrow && self.is_mut_ref_ty(ty) =>
            {
                self.place_borrow_sources(expr, true, state, None)
            }
            Expr::Field(base, _)
            | Expr::TupleField(base, _)
            | Expr::Index(base, _)
            | Expr::Unary(UnOp::Deref, base) => {
                self.copy_projected_borrow_sources(expr, base, state)
            }
            Expr::Path(_) => self.path_borrow_sources(expr, state, allow_mut_ref_reborrow),
            Expr::Tuple(list) | Expr::Array(list) => {
                let items: Vec<u32> = self.body.list(list).to_vec();
                let mut out = Vec::new();
                for id in items {
                    out.extend(self.expr_borrow_sources_inner(ExprId(id), state, allow_mut_ref_reborrow, extend_depth));
                }
                out
            }
            Expr::Repeat(val, _) => self.expr_borrow_sources_inner(val, state, allow_mut_ref_reborrow, None),
            Expr::Struct { fields, base, .. } => {
                let shared_fields = self.struct_shared_ref_fields(expr);
                let field_ids: Vec<u32> = self.body.list(fields).to_vec();
                let field_extend = if base.is_none() { extend_depth } else { None };
                let mut out = Vec::new();
                for id in field_ids {
                    let field = self.body.fields[id as usize];
                    let is_shared = shared_fields.contains(self.src.text(field.name));
                    let mut sources = self.expr_borrow_sources_inner(
                        field.value,
                        state,
                        allow_mut_ref_reborrow,
                        field_extend,
                    );
                    if is_shared {
                        for s in &mut sources {
                            s.mutable = false;
                        }
                    }
                    out.extend(sources);
                }
                if let Some(b) = base {
                    out.extend(self.expr_borrow_sources_inner(b, state, allow_mut_ref_reborrow, None));
                }
                out
            }
            Expr::Block { kind, tail, .. } => {
                if kind == BlockKind::Const {
                    return Vec::new();
                }
                if let Some(cached) = self.temp_sources.get(&expr) {
                    return cached.clone();
                }
                if matches!(kind, BlockKind::Async | BlockKind::AsyncMove) {
                    self.closure_borrow_sources(expr, state)
                } else if let Some(t) = tail {
                    self.expr_borrow_sources_inner(t, state, allow_mut_ref_reborrow, extend_depth)
                } else {
                    Vec::new()
                }
            }
            Expr::If(_, then, otherwise) => {
                if let Some(cached) = self.temp_sources.get(&expr) {
                    return cached.clone();
                }
                let mut out = self.expr_borrow_sources_inner(then, state, allow_mut_ref_reborrow, extend_depth);
                if let Some(alt) = otherwise {
                    out.extend(self.expr_borrow_sources_inner(alt, state, allow_mut_ref_reborrow, extend_depth));
                }
                dedup_sources(out)
            }
            Expr::Match(_, arms) => {
                if let Some(cached) = self.temp_sources.get(&expr) {
                    return cached.clone();
                }
                let arm_ids: Vec<u32> = self.body.list(arms).to_vec();
                let mut out = Vec::new();
                for id in arm_ids {
                    out.extend(self.expr_borrow_sources_inner(
                        self.body.arms[id as usize].body,
                        state,
                        allow_mut_ref_reborrow,
                        extend_depth,
                    ));
                }
                dedup_sources(out)
            }
            Expr::Closure { .. } => self.closure_borrow_sources(expr, state),
            Expr::Call(callee, args) => self.call_borrow_sources(expr, callee, args, state, extend_depth),
            Expr::Method { receiver, params, .. } => self.method_borrow_sources(expr, receiver, params, state),
            _ => Vec::new(),
        }
    }

    fn copy_projected_borrow_sources(
        &mut self,
        expr: ExprId,
        base: ExprId,
        state: &State,
    ) -> Vec<BorrowSource> {
        let mut unwrapped_base = base;
        while let Expr::Paren(inner) = *self.body.expr(unwrapped_base) {
            unwrapped_base = inner;
        }
        if let Expr::Ref { expr: inner, .. } = *self.body.expr(unwrapped_base) {
            return self.expr_borrow_sources_inner(inner, state, false, None);
        }
        let base_ty = self.expr_ty(base);
        let sources = self.expr_borrow_sources_inner(base, state, false, None);
        if matches!(*self.body.expr(expr), Expr::Unary(UnOp::Deref, _))
            && sources.is_empty()
            && let Some(local) = self.expr_plain_local(unwrapped_base)
            && self.is_closure_param(local)
            && self.is_ref_ty(self.local_ty(local))
        {
            return vec![BorrowSource {
                target: local,
                steps: vec![PlaceStep::Deref],
                mutable: self.is_mut_ref_ty(self.local_ty(local)),
                through_ref: true,
                env_only: false,
            }];
        }
        let derefs_base = self.is_ref_ty(base_ty)
            || self.cx.boxed(base_ty).is_some()
            || self.typed.autoderefs.get(&expr).is_some_and(|&d| d > 0);
        if !derefs_base {
            return sources;
        }
        let unwrap_temp = matches!(*self.body.expr(expr), Expr::Index(..))
            || matches!(*self.body.expr(unwrapped_base), Expr::Method { .. });
        self.unwrap_outer_container_loans(sources, state, unwrap_temp)
    }

    pub(super) fn unwrap_outer_container_loans(
        &mut self,
        sources: Vec<BorrowSource>,
        state: &State,
        unwrap_temp: bool,
    ) -> Vec<BorrowSource> {
        let targets: Vec<u32> = sources.iter().map(|s| s.target).collect();
        unwrap_container_loans(sources, state, |s| {
            if super::checks::is_temp_target(s.target).is_some() {
                return unwrap_temp;
            }
            !self.from_caller(s.target, s.through_ref)
                && self.ty_holds_borrow(self.local_ty(s.target))
                && !targets.iter().any(|&other| {
                    other != s.target && state.loans.iter().any(|l| l.holder == Some(other) && l.target == s.target)
                })
        })
    }

    fn struct_shared_ref_fields(&mut self, expr: ExprId) -> HashSet<String> {
        let (adt_item, variant) = match self.typed.res.get(&expr).copied() {
            Some(Res::Ctor(item, v, _)) => (item, v as usize),
            _ => match *self.cx.types.kind(self.expr_ty(expr)) {
                Kind::Adt(item, _) => (item, 0),
                _ => return HashSet::default(),
            },
        };
        let Ok(def) = self.cx.adt_def(adt_item) else {
            return HashSet::default();
        };
        let Some(var) = def.variants.get(variant) else {
            return HashSet::default();
        };
        let adt_src = &self.cx.file(adt_item).source;
        var.fields
            .iter()
            .filter(|f| matches!(def.body.ty(f.ty), AstTy::Ref { mutable: false, .. }))
            .filter_map(|f| f.name.map(|n| adt_src.text(n).to_owned()))
            .collect()
    }

    fn path_borrow_sources(&self, expr: ExprId, state: &State, allow_mut_ref_reborrow: bool) -> Vec<BorrowSource> {
        let Some(Res::Local(local)) = self.typed.res.get(&expr) else {
            return Vec::new();
        };
        if let Kind::Ref(_, inner) = *self.cx.types.kind(self.local_ty(*local))
            && self.is_ref_ty(inner)
            && (self.typed.adjusts.get(&expr).is_some_and(|a| a.derefs > 0)
                || matches!(self.typed.coercions.get(&expr), Some(crate::sema::infer::Coercion::Deref(..))))
        {
            return state
                .loans
                .iter()
                .filter(|l| l.holder == Some(*local))
                .flat_map(|outer| state.loans.iter().filter(move |inner| inner.holder == Some(outer.target)).map(BorrowSource::from_loan))
                .collect();
        }
        let mut out: Vec<BorrowSource> = state
            .loans
            .iter()
            .filter(|l| l.holder == Some(*local))
            .map(BorrowSource::from_loan)
            .collect();
        if allow_mut_ref_reborrow
            && self.is_mut_ref_ty(self.local_ty(*local))
            && !out.iter().any(|s| s.target == *local && s.steps.first() == Some(&PlaceStep::Deref))
        {
            out.push(BorrowSource {
                target: *local,
                steps: vec![PlaceStep::Deref],
                mutable: true,
                through_ref: true,
                env_only: false,
            });
        }
        out
    }

    pub(super) fn place_borrow_sources(&mut self, expr: ExprId, mutable: bool, state: &State, extend_depth: Option<u32>) -> Vec<BorrowSource> {
        let Some(p) = self.root_place(expr) else {
            return self.non_place_borrow_sources(expr, mutable, state, extend_depth);
        };
        self.place_to_sources(&p, mutable, state)
    }

    fn method_recv_borrow_sources(&mut self, call: ExprId, recv: ExprId, mutable: bool, state: &State) -> Vec<BorrowSource> {
        let Some(mut p) = self.root_place(recv) else {
            return self.non_place_borrow_sources(recv, mutable, state, None);
        };
        let derefs = self.typed.adjusts.get(&call).map_or(0, |a| a.derefs);
        if derefs > 0 {
            self.apply_deref_steps(call, self.expr_ty(recv), derefs, &mut p);
        }
        self.place_to_sources(&p, mutable, state)
    }

    fn non_place_borrow_sources(&mut self, expr: ExprId, mutable: bool, state: &State, extend_depth: Option<u32>) -> Vec<BorrowSource> {
        match *self.body.expr(expr) {
            Expr::Paren(inner) | Expr::Cast(inner, _) => {
                self.non_place_borrow_sources(inner, mutable, state, extend_depth)
            }
            Expr::Field(base, _) | Expr::TupleField(base, _) => {
                if self.is_ref_ty(self.expr_ty(base)) {
                    clamp_mut(self.expr_borrow_sources_inner(base, state, true, None), mutable)
                } else {
                    self.non_place_borrow_sources(base, mutable, state, extend_depth)
                }
            }
            Expr::Index(base, _) => {
                if self.is_ref_ty(self.expr_ty(base)) {
                    clamp_mut(self.expr_borrow_sources_inner(base, state, true, None), mutable)
                } else {
                    self.non_place_borrow_sources(base, mutable, state, extend_depth)
                }
            }
            Expr::Unary(UnOp::Deref, inner) => {
                let inner_ty = self.expr_ty(inner);
                if inner_ty != Ty::NONE && matches!(self.cx.types.kind(inner_ty), Kind::Ptr(..)) {
                    Vec::new()
                } else if self.is_ref_ty(inner_ty) {
                    clamp_mut(self.expr_borrow_sources_inner(inner, state, true, None), mutable)
                } else {
                    self.non_place_borrow_sources(inner, mutable, state, extend_depth)
                }
            }
            _ => {
                if self.is_ref_ty(self.expr_ty(expr)) {
                    return clamp_mut(self.expr_borrow_sources_inner(expr, state, true, extend_depth), mutable);
                }
                let mut out = Vec::new();
                if self.is_non_promotable_temp(expr) {
                    let depth = extend_depth.unwrap_or(self.temp_scope_depth);
                    out.push(BorrowSource::temp(depth, mutable, false));
                }
                if self.ty_holds_borrow(self.expr_ty(expr)) {
                    for s in clamp_mut(self.expr_borrow_sources_inner(expr, state, true, extend_depth), mutable) {
                        if !out.contains(&s) {
                            out.push(s);
                        }
                    }
                }
                out
            }
        }
    }

    pub(super) fn place_to_sources(&self, p: &super::places::Place, mutable: bool, state: &State) -> Vec<BorrowSource> {
        if p.raw_ptr_prefix.is_some() {
            return Vec::new();
        }
        let local = p.local;
        let existing: Vec<_> = state.loans.iter().filter(|l| l.holder == Some(local)).collect();
        if p.reborrows_shared_field {
            return dedup_sources(
                existing
                    .into_iter()
                    .map(|l| BorrowSource {
                        mutable: mutable && l.mutable,
                        ..BorrowSource::from_loan(l)
                    })
                    .collect(),
            );
        }
        let steps = &p.steps;
        if !p.through_ref {
            let mut out = vec![BorrowSource {
                target: local,
                steps: steps.to_vec(),
                mutable,
                through_ref: false,
                env_only: false,
            }];
            for l in existing {
                out.push(BorrowSource {
                    mutable: mutable && l.mutable,
                    ..BorrowSource::from_loan(l)
                });
            }
            return dedup_sources(out);
        }
        if !existing.is_empty() {
            let tail_steps = match steps.first() {
                Some(PlaceStep::Deref) => &steps[1..],
                _ => steps,
            };
            let mut out = Vec::new();
            for l in existing {
                if steps_conflict(&l.steps, tail_steps) || tail_steps.is_empty() || l.steps.is_empty() {
                    let mut combined = l.steps.clone();
                    for s in tail_steps {
                        super::places::push_step(&mut combined, s.clone());
                    }
                    out.push(BorrowSource {
                        steps: combined,
                        mutable: mutable && l.mutable,
                        ..BorrowSource::from_loan(l)
                    });
                }
            }
            if self.is_mut_ref_ty(self.local_ty(local)) {
                out.push(BorrowSource {
                    target: local,
                    steps: steps.to_vec(),
                    mutable,
                    through_ref: true,
                    env_only: false,
                });
            }
            return dedup_sources(out);
        }
        vec![BorrowSource {
            target: local,
            steps: steps.to_vec(),
            mutable,
            through_ref: true,
            env_only: false,
        }]
    }

    fn call_borrow_sources(&mut self, expr: ExprId, callee: ExprId, args: List, state: &State, extend_depth: Option<u32>) -> Vec<BorrowSource> {
        let res = if self.body.names_callee(callee) {
            self.typed.res.get(&callee).or_else(|| self.typed.res.get(&expr)).copied()
        } else {
            self.typed.res.get(&expr).copied()
        };
        match res {
            Some(Res::Ctor(..)) => {
                let arg_ids: Vec<u32> = self.body.list(args).to_vec();
                let mut out = Vec::new();
                for id in arg_ids {
                    out.extend(self.expr_borrow_sources_inner(ExprId(id), state, true, extend_depth));
                }
                dedup_sources(out)
            }
            Some(Res::Fn(fn_item, _))
                if !matches!(self.cx.types.kind(self.expr_ty(callee)), Kind::Closure(..))
                    && !self.cx.owner_trait(fn_item).and_then(|tr| self.cx.lang(tr)).is_some_and(crate::sema::infer::is_callable_lang) =>
            {
                self.sig_borrow_sources(expr, fn_item, None, args, state)
            }
            _ => {
                let mut out = self.expr_borrow_sources_inner(callee, state, true, None);
                let ret_has_borrow = self.ty_holds_borrow(self.expr_ty(expr));
                let hrtb_non_ret = self.callee_hrtb_non_ret_params(callee, ret_has_borrow);
                let arg_ids: Vec<u32> = self.body.list(args).to_vec();
                for (idx, id) in arg_ids.into_iter().enumerate() {
                    if hrtb_non_ret.contains(&idx) {
                        continue;
                    }
                    out.extend(self.expr_borrow_sources_inner(ExprId(id), state, true, None));
                }
                self.finish_sig_sources(expr, out, false)
            }
        }
    }

    fn is_raw_ptr_deref(&self, expr: ExprId) -> bool {
        match *self.body.expr(expr) {
            Expr::Paren(inner) => self.is_raw_ptr_deref(inner),
            Expr::Block { tail: Some(t), .. } => self.is_raw_ptr_deref(t),
            Expr::Ref { raw: false, expr: inner, .. } => self.is_raw_ptr_deref(inner),
            Expr::Unary(crate::body::UnOp::Deref, ptr) => matches!(self.cx.types.kind(self.expr_ty(ptr)), Kind::Ptr(..)),
            _ => false,
        }
    }

    fn method_borrow_sources(&mut self, expr: ExprId, receiver: ExprId, params: List, state: &State) -> Vec<BorrowSource> {
        let Some(Res::Fn(fn_item, _)) = self.typed.res.get(&expr).copied() else {
            return Vec::new();
        };
        if self.body.list(params).iter().any(|&id| {
            matches!(*self.body.expr(ExprId(id)), Expr::Closure { body, .. } if self.is_raw_ptr_deref(body))
        }) {
            return Vec::new();
        }
        self.sig_borrow_sources(expr, fn_item, Some(receiver), params, state)
    }

    fn sig_borrow_sources(&mut self, expr: ExprId, fn_item: ItemRef, receiver: Option<ExprId>, args: List, state: &State) -> Vec<BorrowSource> {
        let Ok(sig) = self.cx.signature(fn_item) else {
            return Vec::new();
        };
        let Some(ret_ast) = sig.ret else {
            return Vec::new();
        };
        let (sig_src, sig_item) = self.cx.item(fn_item);
        let mut ret_lts = Lifetimes::default();
        collect_ty_lifetimes(&sig.body, sig_src, ret_ast, &mut ret_lts);
        if ret_lts.names.contains(&"Self")
            && let Some(block) = self.cx.owner_impl(fn_item)
            && let Ok(hdr) = self.cx.impl_header(block)
        {
            let impl_src = &self.cx.impl_file(block).source;
            collect_ty_lifetimes(&hdr.body, impl_src, hdr.self_ty, &mut ret_lts);
        }
        let raw_args = self.body.list(args);
        let (receiver, arg_ids): (Option<ExprId>, Vec<u32>) = if receiver.is_none() && sig.receiver.is_some() && !raw_args.is_empty() {
            (Some(ExprId(raw_args[0])), raw_args[1..].to_vec())
        } else {
            (receiver, raw_args.to_vec())
        };
        let fn_name = self.cx.name(fn_item);
        let is_iter_fn = self.is_iter_mod(fn_item);
        let is_opt_fn = self.item_in_std_mod(fn_item, "option");
        let is_res_fn = self.item_in_std_mod(fn_item, "result");
        let is_by_ref = is_iter_fn && fn_name == "by_ref";
        let mut out = Vec::new();
        if !ret_lts.has_elided && ret_lts.named.is_empty() {
            if super::checks::closure::sig_returns_only_static(&sig, sig_src, ret_ast, &ret_lts) {
                return Vec::new();
            }
            let mut param_matched_own_generic = false;
            for (idx, &arg) in arg_ids.iter().enumerate() {
                if let Some(&(_, param_ty)) = sig.params.get(idx)
                    && !matches!(sig.body.ty(param_ty), AstTy::Ref { .. })
                    && (ty_shares_name(&sig.body, sig_src, param_ty, &ret_lts.names)
                        || ret_lts.names.contains(&"Self"))
                {
                    if sig
                        .generics
                        .iter()
                        .any(|g| ret_lts.names.contains(&sig_src.text(g.name)))
                        && ty_shares_name(&sig.body, sig_src, param_ty, &ret_lts.names)
                    {
                        param_matched_own_generic = true;
                    }
                    out.extend(self.expr_borrow_sources(ExprId(arg), state));
                }
            }
            if let Some(recv) = receiver
                && (!param_matched_own_generic || ret_lts.names.contains(&"Self"))
            {
                let mut recv_sources = self.expr_borrow_sources_inner(recv, state, !recv_is_by_ref(&sig), None);
                if (is_iter_fn || is_opt_fn) && matches!(fn_name, "copied" | "cloned") {
                    recv_sources = self.copied_recv_sources(recv, recv_sources, state, is_opt_fn);
                } else if (is_iter_fn
                    && matches!(fn_name, "map" | "filter_map" | "flat_map")
                    && arg_ids.first().is_some_and(|&a| self.iter_closure_ignores_param(ExprId(a), state)))
                    || ((is_opt_fn || is_res_fn)
                        && matches!(fn_name, "map" | "and_then")
                        && arg_ids.first().is_some_and(|&a| self.fn_arg_returns_static(ExprId(a))))
                {
                    for s in &mut recv_sources {
                        s.env_only = true;
                    }
                }
                out.extend(recv_sources);
            }
            return self.finish_sig_sources(expr, out, is_by_ref);
        }
        let allow_elided_param = !recv_is_by_ref(&sig);
        for (idx, &arg) in arg_ids.iter().enumerate() {
            let Some(&(_, param_ty)) = sig.params.get(idx) else { continue };
            let pairs: Vec<(TyId, ExprId)> = match (sig.body.ty(param_ty), *self.body.expr(ExprId(arg))) {
                (AstTy::Tuple(ptys), Expr::Tuple(aexprs)) if ptys.len == aexprs.len => sig
                    .body
                    .list(*ptys)
                    .iter()
                    .zip(self.body.list(aexprs))
                    .map(|(&pt, &ae)| (TyId(pt), ExprId(ae)))
                    .collect(),
                _ => vec![(param_ty, ExprId(arg))],
            };
            for (pt, ae) in pairs {
                let mut param_lts = Lifetimes::default();
                collect_ty_lifetimes(&sig.body, sig_src, pt, &mut param_lts);
                let matched = (ret_lts.has_elided && allow_elided_param && (param_lts.has_elided || !param_lts.named.is_empty()))
                    || ret_lts.named.iter().any(|n| param_lts.named.contains(n))
                    || (!matches!(sig.body.ty(pt), AstTy::Ref { .. })
                        && ty_shares_name(&sig.body, sig_src, pt, &ret_lts.names));
                if matched {
                    let inner_only = !ret_lts.has_elided && matches!(sig.body.ty(pt), AstTy::Ref { lifetime: None, .. });
                    let mut arg_sources = self.expr_borrow_sources_inner(ae, state, !inner_only, None);
                    if matches!(sig.body.ty(pt), AstTy::Ref { mutable: false, .. }) {
                        for s in &mut arg_sources {
                            s.mutable = false;
                        }
                    }
                    out.extend(arg_sources);
                }
            }
        }
        if let Some(recv) = receiver {
            let recv_creates_borrow = match sig.receiver {
                Some((SelfKind::Ref | SelfKind::RefMut, _)) => {
                    ret_lts.has_elided
                        || (!ret_lts.named.is_empty() && receiver_has_any_named(sig_src, sig_item, &ret_lts.named))
                }
                Some((SelfKind::Typed(self_ty), _)) => {
                    let mut self_lts = Lifetimes::default();
                    collect_ty_lifetimes(&sig.body, sig_src, self_ty, &mut self_lts);
                    (ret_lts.has_elided && (self_lts.has_elided || !self_lts.named.is_empty()))
                        || ret_lts.named.iter().any(|n| self_lts.named.contains(n))
                }
                Some((SelfKind::Value, _)) => true,
                None => false,
            };
            let recv_is_mut = recv_ref_mut(&sig) == Some(Mut::Mut);
            if is_by_ref {
                out.extend(self.by_ref_recv_sources(expr, recv, state));
            } else if recv_creates_borrow && !recv_is_mut && self.is_mut_ref_to_shared_ref(self.expr_ty(recv)) {
                let mut inner_srcs = if let Some(mut p) = self.root_place(recv) {
                    p.reborrows_shared_field = true;
                    self.place_to_sources(&p, false, state)
                } else {
                    self.expr_borrow_sources_inner(recv, state, false, None)
                };
                inner_srcs = unwrap_container_loans(inner_srcs, state, |s| self.is_ref_ty(self.local_ty(s.target)));
                out.extend(inner_srcs);
            } else if recv_creates_borrow {
                if let Some(mutability) = self.typed.adjusts.get(&expr).and_then(|a| a.autoref) {
                    out.extend(self.method_recv_borrow_sources(expr, recv, mutability == Mut::Mut, state));
                } else {
                    out.extend(self.expr_borrow_sources(recv, state));
                }
            } else {
                out.extend(self.expr_borrow_sources_inner(recv, state, !recv_is_by_ref(&sig), None));
            }
        }
        self.finish_sig_sources(expr, out, is_by_ref)
    }

    fn is_mut_ref_to_shared_ref(&self, ty: Ty) -> bool {
        ty != Ty::NONE
            && matches!(*self.cx.types.kind(ty), Kind::Ref(Mut::Mut, inner) if matches!(*self.cx.types.kind(inner), Kind::Ref(Mut::Not, _)))
    }

    fn copied_recv_sources(&mut self, recv: ExprId, mut recv_sources: Vec<BorrowSource>, state: &State, unwrap: bool) -> Vec<BorrowSource> {
        let mut cur = recv;
        loop {
            match *self.body.expr(cur) {
                Expr::Paren(inner) | Expr::Method { receiver: inner, .. } => cur = inner,
                _ => break,
            }
        }
        let root = self.root_place(cur).map(|p| p.local);
        let indirect = root.is_some_and(|r| {
            let r_ty = self.local_ty(r);
            cur == recv || self.is_ref_ty(r_ty) || self.ty_holds_closure(r_ty)
        });
        let targets: Vec<u32> = recv_sources.iter().map(|s| s.target).collect();
        let container_targets: Vec<u32> = recv_sources
            .iter()
            .filter(|s| {
                if super::checks::is_temp_target(s.target).is_some() || Some(s.target) == root {
                    return true;
                }
                if !indirect {
                    return false;
                }
                let r = root.expect("indirect requires root");
                if !state.loans.iter().any(|l| l.holder == Some(r) && l.target == s.target) {
                    return false;
                }
                if !self.ty_holds_borrow(self.local_ty(s.target)) {
                    return false;
                }
                !targets.iter().any(|&other| {
                    other != s.target && state.loans.iter().any(|l| l.holder == Some(other) && l.target == s.target)
                })
            })
            .map(|s| s.target)
            .collect();
        if unwrap {
            unwrap_container_loans(recv_sources, state, |s| container_targets.contains(&s.target))
        } else {
            for s in &mut recv_sources {
                if container_targets.contains(&s.target) {
                    s.env_only = true;
                }
            }
            recv_sources
        }
    }

    fn by_ref_recv_sources(&mut self, expr: ExprId, recv: ExprId, state: &State) -> Vec<BorrowSource> {
        if let Some(p) = self.root_place(recv) {
            let mut srcs = if let Some(mutability) = self.typed.adjusts.get(&expr).and_then(|a| a.autoref) {
                self.method_recv_borrow_sources(expr, recv, mutability == Mut::Mut, state)
            } else {
                self.expr_borrow_sources(recv, state)
            };
            for s in &mut srcs {
                if s.target == p.local
                    || (p.through_ref
                        && self.ty_holds_borrow(self.local_ty(s.target))
                        && state.loans.iter().any(|l| l.holder == Some(p.local) && l.target == s.target))
                {
                    s.env_only = true;
                }
            }
            srcs
        } else {
            let mut srcs = self.expr_borrow_sources_inner(recv, state, true, None);
            if self.is_non_promotable_temp(recv) {
                srcs.push(BorrowSource::temp(self.temp_scope_depth, true, true));
            }
            srcs
        }
    }

    fn iter_closure_ignores_param(&mut self, arg: ExprId, state: &State) -> bool {
        if self.fn_arg_returns_static(arg) { return true; }
        let cur = self.unwrap_parens(arg);
        let Some(closure) = self.typed.closures.get(&cur).cloned() else {
            return false;
        };
        let Some(targets) = self.closure_ret_targets(cur, closure.ret, state) else {
            return false;
        };
        targets.iter().all(|t| {
            closure.captures.iter().any(|c| c.local == *t) || state.loans.iter().any(|l| l.target == *t)
        })
    }

    fn finish_sig_sources(&mut self, expr: ExprId, mut out: Vec<BorrowSource>, keep_env_only: bool) -> Vec<BorrowSource> {
        let ret_ty = self.expr_ty(expr);
        if !keep_env_only && ret_ty != Ty::NONE && !self.ty_holds_closure(ret_ty) {
            out.retain(|s| !s.env_only);
            for s in &mut out {
                if self.is_ref_ty(self.local_ty(s.target)) {
                    s.through_ref = true;
                }
            }
        }
        dedup_sources(out)
    }
}

pub(super) fn unwrap_container_loans(
    sources: Vec<BorrowSource>,
    state: &State,
    mut should_unwrap: impl FnMut(&BorrowSource) -> bool,
) -> Vec<BorrowSource> {
    let mut out = Vec::new();
    for s in sources {
        if super::checks::is_temp_target(s.target).is_some() {
            if !should_unwrap(&s) {
                out.push(s);
            }
            continue;
        }
        if should_unwrap(&s) {
            for l in state.loans.iter().filter(|l| l.holder == Some(s.target)) {
                out.push(BorrowSource::from_loan(l));
            }
        } else {
            out.push(s);
        }
    }
    dedup_sources(out)
}

fn clamp_mut(mut out: Vec<BorrowSource>, mutable: bool) -> Vec<BorrowSource> {
    if !mutable {
        for s in &mut out {
            s.mutable = false;
        }
    }
    out
}

pub(super) fn dedup_sources(sources: Vec<BorrowSource>) -> Vec<BorrowSource> {
    let mut out = Vec::with_capacity(sources.len());
    for s in sources {
        if !out.contains(&s) {
            out.push(s);
        }
    }
    out
}

use rustc_hash::FxHashSet as HashSet;

use crate::body::sig::{GenericKind, SelfKind};
use crate::body::{Args, BlockKind, Body, Bound, Expr, ExprId, GenericArg, List, Stmt, Ty as AstTy, TyId};
use crate::names::ItemRef;
use crate::outline::item::Item;
use crate::outline::source::Source;
use crate::sema::infer::{By, Res};
use crate::sema::ty::{Kind, Mut, Ty};

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
    fn from_loan(l: &Loan) -> Self {
        Self {
            target: l.target,
            steps: l.steps.clone(),
            mutable: l.mutable,
            through_ref: l.through_ref,
            env_only: l.env_only,
        }
    }
}

#[derive(Default)]
struct Lifetimes<'s> {
    has_elided: bool,
    saw_static: bool,
    named: Vec<&'s str>,
    names: Vec<&'s str>,
}

impl Checker<'_, '_> {
    pub(super) fn from_caller(&self, target: u32, through_ref: bool) -> bool {
        self.is_param.get(target as usize).copied().unwrap_or(false)
            && (through_ref || self.is_ref_ty(self.local_ty(target)))
    }

    pub(super) fn ty_holds_borrow(&mut self, ty: Ty) -> bool {
        let mut seen = HashSet::default();
        self.ty_holds_borrow_inner(ty, &mut seen)
    }

    fn ty_holds_borrow_inner(&mut self, ty: Ty, seen: &mut HashSet<Ty>) -> bool {
        if ty == Ty::NONE || !seen.insert(ty) {
            return false;
        }
        match self.cx.types.kind(ty).clone() {
            Kind::Ref(..) | Kind::Dyn(..) => true,
            Kind::Slice(elem) | Kind::Array(elem, _) | Kind::Pat(elem, ..) => {
                self.ty_holds_borrow_inner(elem, seen)
            }
            Kind::Tuple(items) => {
                let items = self.cx.types.substs(items).to_vec();
                items.into_iter().any(|item| self.ty_holds_borrow_inner(item, seen))
            }
            Kind::Closure(..) => self.closure_ty_holds_borrow(ty, seen),
            Kind::Adt(item, args) => {
                let args = self.cx.types.substs(args).to_vec();
                if args.into_iter().any(|arg| self.ty_holds_borrow_inner(arg, seen)) {
                    return true;
                }
                self.cx.adt_def(item).is_ok_and(|def| {
                    def.generics.params.iter().any(|g| g.kind == GenericKind::Lifetime)
                })
            }
            _ => false,
        }
    }

    fn closure_ty_holds_borrow(&mut self, ty: Ty, seen: &mut HashSet<Ty>) -> bool {
        let Kind::Closure(parent, _, at) = *self.cx.types.kind(ty) else {
            return false;
        };
        if let Some(info) = self.cx.closure(ty) {
            return info.closure.captures.iter().any(|c| {
                c.by != By::Value
                    || self.ty_holds_borrow_inner(info.typed.locals.get(c.local as usize).map_or(Ty::NONE, |l| l.ty), seen)
            });
        }
        if parent == self.item
            && let Some(closure) = self.typed.closures.get(&ExprId(at))
        {
            return closure.captures.iter().any(|c| {
                c.by != By::Value || self.ty_holds_borrow_inner(self.local_ty(c.local), seen)
            });
        }
        true
    }

    fn ty_holds_closure(&mut self, ty: Ty) -> bool {
        let mut seen = HashSet::default();
        self.ty_holds_closure_inner(ty, &mut seen)
    }

    fn ty_holds_closure_inner(&mut self, ty: Ty, seen: &mut HashSet<Ty>) -> bool {
        if ty == Ty::NONE || !seen.insert(ty) {
            return false;
        }
        match self.cx.types.kind(ty).clone() {
            Kind::Closure(..) | Kind::Dyn(..) | Kind::FnDef(..) | Kind::FnPtr(..) => true,
            Kind::Ref(_, inner) | Kind::Slice(inner) | Kind::Array(inner, _) | Kind::Pat(inner, ..) => {
                self.ty_holds_closure_inner(inner, seen)
            }
            Kind::Tuple(items) => {
                let items = self.cx.types.substs(items).to_vec();
                items.into_iter().any(|item| self.ty_holds_closure_inner(item, seen))
            }
            Kind::Adt(item, items) => {
                let mod_path = self.cx.module_text(item);
                if mod_path.starts_with("core::iter") || mod_path.starts_with("std::iter") {
                    return true;
                }
                let items = self.cx.types.substs(items).to_vec();
                items.into_iter().any(|item| self.ty_holds_closure_inner(item, seen))
            }
            _ => false,
        }
    }

    pub(super) fn expr_borrow_sources(&mut self, expr: ExprId, state: &State) -> Vec<BorrowSource> {
        self.expr_borrow_sources_inner(expr, state, true)
    }

    fn expr_borrow_sources_inner(&mut self, expr: ExprId, state: &State, allow_mut_ref_reborrow: bool) -> Vec<BorrowSource> {
        let ty = self.expr_ty(expr);
        if ty != Ty::NONE && !self.ty_holds_borrow(ty) {
            return Vec::new();
        }
        match *self.body.expr(expr) {
            Expr::Paren(inner) | Expr::Cast(inner, _) | Expr::Try(inner) => {
                self.expr_borrow_sources_inner(inner, state, allow_mut_ref_reborrow)
            }
            Expr::Ref { mutable, raw: false, expr: inner } => {
                if self.is_raw_ptr_deref(inner) {
                    return Vec::new();
                }
                if allow_mut_ref_reborrow {
                    self.place_borrow_sources(inner, mutable, state)
                } else {
                    self.expr_borrow_sources_inner(inner, state, false)
                }
            }
            Expr::Path(_) => self.path_borrow_sources(expr, state, allow_mut_ref_reborrow),
            Expr::Tuple(list) | Expr::Array(list) => {
                let items: Vec<u32> = self.body.list(list).to_vec();
                let mut out = Vec::new();
                for id in items {
                    out.extend(self.expr_borrow_sources_inner(ExprId(id), state, allow_mut_ref_reborrow));
                }
                out
            }
            Expr::Repeat(val, _) => self.expr_borrow_sources_inner(val, state, allow_mut_ref_reborrow),
            Expr::Struct { fields, base, .. } => {
                let shared_fields = self.struct_shared_ref_fields(expr);
                let field_ids: Vec<u32> = self.body.list(fields).to_vec();
                let mut out = Vec::new();
                for id in field_ids {
                    let field = self.body.fields[id as usize];
                    let is_shared = shared_fields.contains(self.src.text(field.name));
                    let mut sources = self.expr_borrow_sources_inner(
                        field.value,
                        state,
                        allow_mut_ref_reborrow,
                    );
                    if is_shared {
                        for s in &mut sources {
                            s.mutable = false;
                        }
                    }
                    out.extend(sources);
                }
                if let Some(b) = base {
                    out.extend(self.expr_borrow_sources_inner(b, state, allow_mut_ref_reborrow));
                }
                out
            }
            Expr::Block { kind, tail, .. } => {
                if matches!(kind, BlockKind::Async | BlockKind::AsyncMove) {
                    self.closure_borrow_sources(expr, state)
                } else if let Some(t) = tail {
                    self.expr_borrow_sources_inner(t, state, allow_mut_ref_reborrow)
                } else {
                    Vec::new()
                }
            }
            Expr::If(_, then, otherwise) => {
                let mut out = self.expr_borrow_sources_inner(then, state, allow_mut_ref_reborrow);
                if let Some(alt) = otherwise {
                    out.extend(self.expr_borrow_sources_inner(alt, state, allow_mut_ref_reborrow));
                }
                dedup_sources(out)
            }
            Expr::Match(_, arms) => {
                let arm_ids: Vec<u32> = self.body.list(arms).to_vec();
                let mut out = Vec::new();
                for id in arm_ids {
                    out.extend(self.expr_borrow_sources_inner(
                        self.body.arms[id as usize].body,
                        state,
                        allow_mut_ref_reborrow,
                    ));
                }
                dedup_sources(out)
            }
            Expr::Closure { .. } => self.closure_borrow_sources(expr, state),
            Expr::Call(callee, args) => self.call_borrow_sources(expr, callee, args, state),
            Expr::Method { receiver, params, .. } => self.method_borrow_sources(expr, receiver, params, state),
            _ => Vec::new(),
        }
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

    pub(super) fn closure_borrow_sources(&mut self, expr: ExprId, state: &State) -> Vec<BorrowSource> {
        let Some(closure) = self.typed.closures.get(&expr).cloned() else {
            return Vec::new();
        };
        let ret_targets = self.closure_ret_targets(expr, closure.ret, state);
        let capture_env_only = |target: u32| ret_targets.as_ref().is_some_and(|set| !set.contains(&target));
        let mut out = Vec::new();
        for cap in &closure.captures {
            let cap_steps: Vec<PlaceStep> = cap.path.iter().map(|&(_, f)| PlaceStep::Field(super::places::FieldKey::Tuple(f))).collect();
            if cap.by == By::Value {
                let cap_ty = cap.ty(self.cx, self.local_ty(cap.local)).unwrap_or_else(|_| self.local_ty(cap.local));
                if self.is_mut_ref_ty(cap_ty) {
                    let mut steps = cap_steps;
                    steps.push(PlaceStep::Deref);
                    out.push(BorrowSource { target: cap.local, steps, mutable: true, through_ref: true, env_only: capture_env_only(cap.local) });
                }
            } else {
                for (steps, is_mut, through_ref, _) in self.closure_capture_info(expr, cap.by, cap.local) {
                    if steps.starts_with(&cap_steps) {
                        out.push(BorrowSource { target: cap.local, steps, mutable: is_mut, through_ref, env_only: capture_env_only(cap.local) });
                    }
                }
            }
            for l in state.loans.iter().filter(|l| l.holder == Some(cap.local)) {
                out.push(BorrowSource { env_only: l.env_only || capture_env_only(l.target), ..BorrowSource::from_loan(l) });
            }
        }
        dedup_sources(out)
    }

    fn closure_ret_targets(&mut self, expr: ExprId, ret_ty: Ty, state: &State) -> Option<HashSet<u32>> {
        if !self.ty_holds_borrow(ret_ty) {
            return Some(HashSet::default());
        }
        let Expr::Closure { body, .. } = *self.body.expr(expr) else {
            return None;
        };
        let mut inner_locals = HashSet::default();
        if !self.can_analyze_closure_ret(body, &mut inner_locals) {
            return None;
        }
        let targets = self.expr_borrow_sources(body, state).into_iter().map(|s| s.target).collect();
        Some(targets)
    }

    fn can_analyze_closure_ret(&mut self, expr: ExprId, inner_locals: &mut HashSet<u32>) -> bool {
        match *self.body.expr(expr) {
            Expr::Paren(inner) | Expr::Cast(inner, _) | Expr::Try(inner) => {
                self.can_analyze_closure_ret(inner, inner_locals)
            }
            Expr::Block { kind, stmts, tail, .. } => {
                if matches!(kind, BlockKind::Async | BlockKind::AsyncMove) {
                    return false;
                }
                let stm_ids: Vec<u32> = self.body.list(stmts).to_vec();
                for id in stm_ids {
                    match self.body.stmts[id as usize] {
                        Stmt::Let { pat, init, otherwise, .. } => {
                            self.collect_pat_locals(pat, inner_locals);
                            if init.is_some_and(|e| self.expr_has_return(e))
                                || otherwise.is_some_and(|e| self.expr_has_return(e))
                            {
                                return false;
                            }
                        }
                        Stmt::Expr(e, _) => {
                            if self.expr_has_return(e) {
                                return false;
                            }
                        }
                        Stmt::Item(..) | Stmt::Macro { .. } | Stmt::Empty => {}
                    }
                }
                tail.is_none_or(|t| self.can_analyze_closure_ret(t, inner_locals))
            }
            Expr::If(c, t, e) => {
                !self.expr_has_return(c)
                    && self.can_analyze_closure_ret(t, inner_locals)
                    && e.is_none_or(|alt| self.can_analyze_closure_ret(alt, inner_locals))
            }
            Expr::Match(scrut, arms) => {
                if self.expr_has_return(scrut) {
                    return false;
                }
                let arm_ids: Vec<u32> = self.body.list(arms).to_vec();
                for id in arm_ids {
                    let arm = self.body.arms[id as usize];
                    let mut arm_locals = inner_locals.clone();
                    self.collect_pat_locals(arm.pat, &mut arm_locals);
                    if arm.guard.is_some_and(|g| self.expr_has_return(g))
                        || !self.can_analyze_closure_ret(arm.body, &mut arm_locals)
                    {
                        return false;
                    }
                }
                true
            }
            _ => !self.expr_has_return(expr) && !self.expr_uses_inner_borrow_local(expr, inner_locals),
        }
    }

    pub(super) fn place_borrow_sources(&self, expr: ExprId, mutable: bool, state: &State) -> Vec<BorrowSource> {
        let Some(p) = self.root_place(expr) else {
            return Vec::new();
        };
        self.place_to_sources(&p, mutable, state)
    }

    fn method_recv_borrow_sources(&self, call: ExprId, recv: ExprId, mutable: bool, state: &State) -> Vec<BorrowSource> {
        let Some(mut p) = self.root_place(recv) else {
            return Vec::new();
        };
        let derefs = self.typed.adjusts.get(&call).map_or(0, |a| a.derefs);
        if derefs > 0 {
            self.apply_deref_steps(call, self.expr_ty(recv), derefs, &mut p);
        }
        self.place_to_sources(&p, mutable, state)
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

    fn call_borrow_sources(&mut self, expr: ExprId, callee: ExprId, args: List, state: &State) -> Vec<BorrowSource> {
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
                    out.extend(self.expr_borrow_sources(ExprId(id), state));
                }
                dedup_sources(out)
            }
            Some(Res::Fn(fn_item, _)) => self.sig_borrow_sources(expr, fn_item, None, args, state),
            _ => {
                let mut out = self.expr_borrow_sources(callee, state);
                let arg_ids: Vec<u32> = self.body.list(args).to_vec();
                for id in arg_ids {
                    out.extend(self.expr_borrow_sources(ExprId(id), state));
                }
                self.finish_sig_sources(expr, out)
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
        let arg_ids: Vec<u32> = self.body.list(args).to_vec();
        let mut out = Vec::new();
        if !ret_lts.has_elided && ret_lts.named.is_empty() {
            if ret_lts.saw_static {
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
                let mut recv_sources = self.expr_borrow_sources_inner(recv, state, !recv_is_by_ref(&sig));
                if matches!(self.cx.name(fn_item), "copied" | "cloned")
                    && self.cx.module_text(fn_item).starts_with("core::iter")
                {
                    let mut cur = recv;
                    loop {
                        match *self.body.expr(cur) {
                            Expr::Paren(inner) | Expr::Method { receiver: inner, .. } => cur = inner,
                            _ => break,
                        }
                    }
                    if let Some(p) = self.root_place(cur) {
                        for s in &mut recv_sources {
                            if s.target == p.local && s.steps.starts_with(&p.steps) {
                                s.env_only = true;
                            }
                        }
                    }
                }
                out.extend(recv_sources);
            }
            return self.finish_sig_sources(expr, out);
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
                    let mut arg_sources = self.expr_borrow_sources_inner(ae, state, !inner_only);
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
            if recv_creates_borrow {
                if let Some(mutability) = self.typed.adjusts.get(&expr).and_then(|a| a.autoref) {
                    out.extend(self.method_recv_borrow_sources(expr, recv, mutability == Mut::Mut, state));
                } else {
                    out.extend(self.expr_borrow_sources(recv, state));
                }
            } else {
                out.extend(self.expr_borrow_sources_inner(recv, state, !recv_is_by_ref(&sig)));
            }
        }
        self.finish_sig_sources(expr, out)
    }

    fn finish_sig_sources(&mut self, expr: ExprId, mut out: Vec<BorrowSource>) -> Vec<BorrowSource> {
        let ret_ty = self.expr_ty(expr);
        if ret_ty != Ty::NONE && !self.ty_holds_closure(ret_ty) {
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

fn dedup_sources(sources: Vec<BorrowSource>) -> Vec<BorrowSource> {
    let mut out = Vec::with_capacity(sources.len());
    for s in sources {
        if !out.contains(&s) {
            out.push(s);
        }
    }
    out
}

fn collect_ty_lifetimes<'s>(body: &Body, src: &'s Source, ty: TyId, out: &mut Lifetimes<'s>) {
    match *body.ty(ty) {
        AstTy::Ref { lifetime, ty: inner, .. } => {
            match lifetime {
                None => out.has_elided = true,
                Some(lt) => push_lt(src.text(lt), out),
            }
            collect_ty_lifetimes(body, src, inner, out);
        }
        AstTy::Path(path) => {
            let p = body.path(path);
            let segs = body.list(p.segments);
            if let Some(&first) = segs.first() {
                let name = src.text(body.segments[first as usize].name);
                if !out.names.contains(&name) {
                    out.names.push(name);
                }
            }
            for &seg_id in segs {
                match body.segments[seg_id as usize].args {
                    Some(Args::Angle(list)) => {
                        for &arg_id in body.list(list) {
                            match body.args[arg_id as usize] {
                                GenericArg::Lifetime(lt) => push_lt(src.text(lt), out),
                                GenericArg::Ty(t) | GenericArg::Binding { ty: t, .. } => {
                                    collect_ty_lifetimes(body, src, t, out);
                                }
                                GenericArg::Constraint { .. } | GenericArg::Const(_) => {}
                            }
                        }
                    }
                    Some(Args::Paren { inputs, output }) => {
                        for &t in body.list(inputs) {
                            collect_ty_lifetimes(body, src, TyId(t), out);
                        }
                        if let Some(o) = output {
                            collect_ty_lifetimes(body, src, o, out);
                        }
                    }
                    Some(Args::ReturnType) | None => {}
                }
            }
        }
        AstTy::Tuple(list) => {
            for &t in body.list(list) {
                collect_ty_lifetimes(body, src, TyId(t), out);
            }
        }
        AstTy::ImplTrait(_, list) | AstTy::DynTrait(list) => {
            for &b_id in body.list(list) {
                match body.bounds[b_id as usize] {
                    Bound::Lifetime(lt) => push_lt(src.text(lt), out),
                    Bound::Trait { path, .. } => {
                        for &seg_id in body.list(body.path(path).segments) {
                            if let Some(Args::Angle(args)) = body.segments[seg_id as usize].args {
                                for &arg_id in body.list(args) {
                                    match body.args[arg_id as usize] {
                                        GenericArg::Lifetime(lt) => push_lt(src.text(lt), out),
                                        GenericArg::Ty(t) | GenericArg::Binding { ty: t, .. } => {
                                            collect_ty_lifetimes(body, src, t, out);
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }
                    }
                    Bound::Use(_) => {}
                }
            }
        }
        AstTy::Slice(t) | AstTy::Array(t, _) | AstTy::Ptr { ty: t, .. } | AstTy::Paren(t) | AstTy::Pat(t, _) => {
            collect_ty_lifetimes(body, src, t, out);
        }
        _ => {}
    }
}

fn push_lt<'s>(text: &'s str, out: &mut Lifetimes<'s>) {
    if text == "'_" {
        out.has_elided = true;
    } else if text == "'static" {
        out.saw_static = true;
    } else if !out.named.contains(&text) {
        out.named.push(text);
    }
}

fn ty_shares_name(body: &Body, src: &Source, ty: TyId, ret_names: &[&str]) -> bool {
    if ret_names.is_empty() {
        return false;
    }
    let mut param_lts = Lifetimes::default();
    collect_ty_lifetimes(body, src, ty, &mut param_lts);
    param_lts.names.iter().any(|n| ret_names.contains(n))
}

pub(super) fn param_is_shared_ref(sig: &crate::body::sig::Signature, idx: usize) -> bool {
    let Some(&(_, mut ty)) = sig.params.get(idx) else {
        return false;
    };
    while let AstTy::Paren(inner) = *sig.body.ty(ty) {
        ty = inner;
    }
    matches!(*sig.body.ty(ty), AstTy::Ref { mutable: false, .. })
}

fn recv_is_by_ref(sig: &crate::body::sig::Signature) -> bool {
    matches!(sig.receiver, Some((SelfKind::Ref | SelfKind::RefMut, _)))
        || matches!(sig.receiver, Some((SelfKind::Typed(self_ty), _)) if matches!(sig.body.ty(self_ty), AstTy::Ref { .. }))
}

fn receiver_has_any_named(src: &Source, item: &Item, named: &[&str]) -> bool {
    let mut at = item.start;
    while at < item.end && src.kind(at) != rustc_lexer::TokenKind::OpenParen {
        at += 1;
    }
    let end = (at + 12).min(item.end);
    for tok in at + 1..end {
        if src.text(tok) == "self" {
            return (at + 1..tok).any(|t| named.contains(&src.text(t)));
        }
    }
    false
}

use anyhow::Result;

use crate::body::sig::{GenericKind, Signature};
use crate::body::{Args, Body, Bound, Expr, ExprId, GenericArg, List, Ty as AstTy, TyId};
use crate::outline::source::Source;
use crate::sema::ty::Kind;

use super::super::bind::push_loan;
use super::super::calls::{Lifetimes, collect_ty_lifetimes};
use super::super::places::{Place, PlaceStep};
use super::super::propagate::{BorrowSource, dedup_sources};
use super::super::{Checker, State, is_temp_holder};
use super::is_temp_target;

#[derive(Default)]
struct ParamBounds {
    req_fn: bool,
    req_fn_mut: bool,
    req_static: bool,
    hrtb_params: Vec<usize>,
    hrtb_non_ret: Vec<usize>,
    caller_lt_params: Vec<(usize, bool)>,
}

#[derive(Clone, Debug)]
pub(in super::super) struct ClosureParamLoan {
    pub(in super::super) param_idx: usize,
    pub(in super::super) holder: u32,
    pub(in super::super) copies_inner_ref: bool,
    pub(in super::super) derefs_past_outer_ref: bool,
}

#[derive(Clone, Debug, Default)]
pub(in super::super) struct ClosureEffects {
    pub(in super::super) moves_capture: bool,
    pub(in super::super) mutates_capture: bool,
    pub(in super::super) borrows_local: bool,
    pub(in super::super) outer_loans: Vec<(u32, BorrowSource)>,
    pub(in super::super) param_loans: Vec<ClosureParamLoan>,
}

impl Checker<'_, '_> {
    pub(in super::super) fn check_call_bounds(
        &mut self,
        arg_ids: &[u32],
        sig: Option<&Signature>,
        fn_item: Option<crate::names::ItemRef>,
        has_self_arg: bool,
        state: &State,
    ) -> Result<()> {
        let Some(sig) = sig else { return Ok(()) };
        let sig_src = fn_item.map_or(self.src, |item| self.cx.item(item).0);
        for (idx, &id) in arg_ids.iter().enumerate() {
            let param_idx = if has_self_arg {
                let Some(p) = idx.checked_sub(1) else {
                    continue;
                };
                p
            } else {
                idx
            };
            let Some(&(_, param_ty)) = sig.params.get(param_idx) else {
                continue;
            };
            let arg = ExprId(id);
            let ref_static = matches!(*sig.body.ty(param_ty), AstTy::Ref { lifetime: Some(lt), .. } if sig_src.text(lt) == "'static");
            let bounds = self.param_bound_reqs(sig, sig_src, param_ty);
            if bounds.req_static || ref_static {
                self.check_static_arg(arg, state)?;
            }
            if bounds.req_fn
                || bounds.req_fn_mut
                || bounds.req_static
                || !bounds.hrtb_params.is_empty()
            {
                self.check_closure_arg_reqs(
                    arg,
                    bounds.req_fn,
                    bounds.req_fn_mut,
                    bounds.req_static,
                    &bounds.hrtb_params,
                )?;
            }
        }
        Ok(())
    }

    fn callee_param_bounds(&mut self, callee: ExprId) -> Option<ParamBounds> {
        let cur = self.unwrap_parens(callee);
        let crate::sema::infer::Res::Local(local) = self.typed.res.get(&cur).copied()? else {
            return None;
        };
        let pos = self.typed.params.iter().position(|&p| p == local)?;
        let sig = self.cx.signature(self.item).ok()?;
        let param_idx = if sig.receiver.is_some() {
            pos.checked_sub(1)?
        } else {
            pos
        };
        let &(_, param_ty) = sig.params.get(param_idx)?;
        Some(self.param_bound_reqs(&sig, self.src, param_ty))
    }

    pub(in super::super) fn callee_hrtb_non_ret_params(
        &mut self,
        callee: ExprId,
        ret_has_borrow: bool,
    ) -> Vec<usize> {
        let Some(bounds) = self.callee_param_bounds(callee) else {
            return Vec::new();
        };
        if ret_has_borrow {
            bounds.hrtb_non_ret
        } else {
            bounds.hrtb_params
        }
    }

    pub(in super::super) fn check_callee_bound_args(
        &mut self,
        callee: ExprId,
        arg_ids: &[u32],
        state: &State,
    ) -> Result<()> {
        let Some(bounds) = self.callee_param_bounds(callee) else {
            return Ok(());
        };
        for (p_idx, outer_elided) in bounds.caller_lt_params {
            let Some(&id) = arg_ids.get(p_idx) else {
                continue;
            };
            let arg = self.unwrap_parens(ExprId(id));
            let check_expr = if outer_elided
                && let Expr::Ref {
                    raw: false,
                    expr: inner,
                    ..
                } = *self.body.expr(arg)
            {
                inner
            } else {
                arg
            };
            self.check_static_arg(check_expr, state)?;
        }
        Ok(())
    }

    fn param_bound_reqs(&self, sig: &Signature, src: &Source, param_ty: TyId) -> ParamBounds {
        let mut out = ParamBounds::default();
        let mut check_bounds = |list: List| {
            for &b_id in sig.body.list(list) {
                match sig.body.bounds[b_id as usize] {
                    Bound::Lifetime(lt) if src.text(lt) == "'static" => out.req_static = true,
                    Bound::Trait { path, .. } => {
                        if let Some(&seg_id) = sig.body.list(sig.body.path(path).segments).last() {
                            let seg = sig.body.segments[seg_id as usize];
                            let is_callable = match src.text(seg.name) {
                                "Fn" => {
                                    out.req_fn = true;
                                    true
                                }
                                "FnMut" => {
                                    out.req_fn_mut = true;
                                    true
                                }
                                "FnOnce" => true,
                                _ => false,
                            };
                            if is_callable && let Some(Args::Paren { inputs, output }) = seg.args {
                                let out_elided = output.is_some_and(|out_ty| {
                                    ast_ty_may_carry_elided_lt(&sig.body, src, out_ty)
                                });
                                let mut out_lts = Lifetimes::default();
                                if let Some(out_ty) = output {
                                    collect_ty_lifetimes(&sig.body, src, out_ty, &mut out_lts);
                                }
                                for (p_idx, &in_ty) in sig.body.list(inputs).iter().enumerate() {
                                    let mut in_lts = Lifetimes::default();
                                    collect_ty_lifetimes(&sig.body, src, TyId(in_ty), &mut in_lts);
                                    let outer_elided = matches!(
                                        *sig.body.ty(TyId(in_ty)),
                                        AstTy::Ref { lifetime: None, .. }
                                    );
                                    let outer_named_hrtb = match *sig.body.ty(TyId(in_ty)) {
                                        AstTy::Ref {
                                            lifetime: Some(lt), ..
                                        } => {
                                            let name = src.text(lt);
                                            (name != "'static"
                                                && !sig.generics.iter().any(|g| {
                                                    g.kind == GenericKind::Lifetime
                                                        && src.text(g.name) == name
                                                }))
                                            .then_some(name)
                                        }
                                        _ => None,
                                    };
                                    let has_caller_lt = in_lts.saw_static
                                        || in_lts.named.iter().any(|&lt| {
                                            sig.generics.iter().any(|g| {
                                                g.kind == GenericKind::Lifetime
                                                    && src.text(g.name) == lt
                                            })
                                        });
                                    let outer_hrtb = outer_elided || outer_named_hrtb.is_some();
                                    if has_caller_lt
                                        && !out
                                            .caller_lt_params
                                            .iter()
                                            .any(|&(idx, _)| idx == p_idx)
                                    {
                                        out.caller_lt_params.push((p_idx, outer_hrtb));
                                    }
                                    let is_hrtb = (outer_elided && in_lts.named.is_empty())
                                        || outer_named_hrtb.is_some();
                                    if is_hrtb {
                                        if !out.hrtb_params.contains(&p_idx) {
                                            out.hrtb_params.push(p_idx);
                                        }
                                        let ret_uses_hrtb = outer_named_hrtb
                                            .map_or(out_elided, |name| {
                                                out_lts.named.contains(&name)
                                            });
                                        if !ret_uses_hrtb && !out.hrtb_non_ret.contains(&p_idx) {
                                            out.hrtb_non_ret.push(p_idx);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        };
        match *sig.body.ty(param_ty) {
            AstTy::ImplTrait(_, bounds) | AstTy::DynTrait(bounds) => check_bounds(bounds),
            AstTy::Ref { ty: inner, .. } => {
                if let AstTy::ImplTrait(_, bounds) | AstTy::DynTrait(bounds) = *sig.body.ty(inner) {
                    check_bounds(bounds);
                }
            }
            AstTy::Path(path) => {
                let p = sig.body.path(path);
                let segs = sig.body.list(p.segments);
                if p.qself.is_none() && segs.len() == 1 {
                    let name = src.text(sig.body.segments[segs[0] as usize].name);
                    for g in &sig.generics {
                        if g.kind == GenericKind::Type && src.text(g.name) == name {
                            check_bounds(g.bounds);
                        }
                    }
                    for pred in &sig.predicates {
                        if let Some(pty) = pred.ty
                            && let AstTy::Path(pp) = *sig.body.ty(pty)
                        {
                            let psegs = sig.body.list(sig.body.path(pp).segments);
                            if psegs.len() == 1
                                && src.text(sig.body.segments[psegs[0] as usize].name) == name
                            {
                                check_bounds(pred.bounds);
                            }
                        }
                    }
                }
            }
            _ => {}
        }
        out
    }

    fn check_static_arg(&mut self, expr: ExprId, state: &State) -> Result<()> {
        let in_const = self.is_const_or_static_item();
        for s in self.expr_borrow_sources(expr, state) {
            if is_temp_target(s.target).is_some() {
                if in_const && !self.is_non_promotable_temp(expr) {
                    continue;
                }
                return Err(self.does_not_live_long_enough(Some(expr), s.target));
            }
            if !self.from_caller(s.target, s.through_ref) {
                return Err(self.does_not_live_long_enough(Some(expr), s.target));
            }
        }
        Ok(())
    }

    pub(in super::super) fn fn_arg_returns_static(&mut self, arg: ExprId) -> bool {
        let arg = self.unwrap_parens(arg);
        let Some(crate::sema::infer::Res::Fn(f_item, _)) = self.typed.res.get(&arg).copied() else {
            return false;
        };
        let Ok(sig) = self.cx.signature(f_item) else {
            return false;
        };
        let Some(ret_ast) = sig.ret else { return false };
        let sig_src = self.cx.item(f_item).0;
        let mut lts = Lifetimes::default();
        collect_ty_lifetimes(&sig.body, sig_src, ret_ast, &mut lts);
        sig_returns_only_static(&sig, sig_src, ret_ast, &lts)
    }

    fn check_closure_effects(
        &self,
        expr: Option<ExprId>,
        eff: &ClosureEffects,
        req_fn: bool,
        req_fn_mut: bool,
        req_static: bool,
    ) -> Result<()> {
        if (req_fn || req_fn_mut) && eff.moves_capture {
            return Err(self.borrow_err(
                expr,
                "cannot move out of a captured variable in an `Fn` or `FnMut` closure",
            ));
        }
        if req_fn && eff.mutates_capture {
            return Err(self.borrow_err(
                expr,
                "cannot borrow captured variable as mutable in an `Fn` closure",
            ));
        }
        if req_static && eff.borrows_local {
            return Err(self.borrow_err(
                expr,
                "closure may outlive the current function, but it borrows a local variable",
            ));
        }
        Ok(())
    }

    fn check_closure_arg_reqs(
        &self,
        expr: ExprId,
        req_fn: bool,
        req_fn_mut: bool,
        req_static: bool,
        hrtb_params: &[usize],
    ) -> Result<()> {
        let Some(c_expr) = self.closure_expr_of(expr) else {
            return Ok(());
        };
        let Some(eff) = self.closure_effects.get(&c_expr) else {
            return Ok(());
        };
        if eff
            .param_loans
            .iter()
            .any(|pl| hrtb_params.contains(&pl.param_idx) && !pl.derefs_past_outer_ref)
        {
            return Err(self.borrow_err(Some(expr), "borrowed data escapes outside of closure"));
        }
        self.check_closure_effects(Some(expr), eff, req_fn, req_fn_mut, req_static)
    }

    fn unwrap_expr(&self, mut expr: ExprId) -> ExprId {
        loop {
            match *self.body.expr(expr) {
                Expr::Paren(inner) | Expr::Ref { expr: inner, .. } => expr = inner,
                _ => return expr,
            }
        }
    }

    pub(in super::super) fn closure_expr_of(&self, expr: ExprId) -> Option<ExprId> {
        let unwrapped = self.unwrap_expr(expr);
        if matches!(*self.body.expr(unwrapped), Expr::Closure { .. }) {
            return Some(unwrapped);
        }
        let ty = self.expr_ty(expr);
        if ty == crate::sema::ty::Ty::NONE {
            return None;
        }
        match *self.cx.types.kind(ty) {
            Kind::Closure(parent, _, at) if parent == self.item => Some(ExprId(at)),
            _ => None,
        }
    }

    pub(in super::super) fn attach_closure_arg_param_loans(
        &mut self,
        recv_sources: Option<&[BorrowSource]>,
        arg_ids: &[u32],
        arg_sources: &[Vec<BorrowSource>],
        state: &mut State,
    ) -> Result<()> {
        for (idx, &id) in arg_ids.iter().enumerate() {
            let Some(c_expr) = self.closure_expr_of(ExprId(id)) else {
                continue;
            };
            let Some(eff) = self.closure_effects.get(&c_expr).cloned() else {
                continue;
            };
            if eff.param_loans.is_empty() {
                continue;
            }
            let mut outer_sources: Vec<BorrowSource> = recv_sources
                .into_iter()
                .flatten()
                .filter(|s| !s.env_only)
                .cloned()
                .collect();
            for (other_idx, &other_id) in arg_ids.iter().enumerate() {
                if other_idx != idx
                    && self.closure_expr_of(ExprId(other_id)).is_none()
                    && let Some(srcs) = arg_sources.get(other_idx)
                {
                    outer_sources.extend(srcs.iter().filter(|s| !s.env_only).cloned());
                }
            }
            let outer_sources = dedup_sources(outer_sources);
            if outer_sources.is_empty() {
                continue;
            }
            for pl in &eff.param_loans {
                let sources = if pl.copies_inner_ref {
                    self.unwrap_outer_container_loans(outer_sources.clone(), state, true)
                } else {
                    outer_sources.clone()
                };
                self.propagate_closure_param_loan(pl.holder, &sources, state)?;
            }
        }
        Ok(())
    }

    pub(in super::super) fn propagate_closure_param_loan(
        &mut self,
        holder: u32,
        sources: &[BorrowSource],
        state: &mut State,
    ) -> Result<()> {
        let through_ref = self.is_param_local(holder) && self.is_mut_ref_ty(self.local_ty(holder));
        let dst = Place {
            local: holder,
            through_ref,
            through_deref: false,
            mut_ref_root: false,
            behind_shared: false,
            reborrows_shared_field: false,
            raw_ptr_prefix: None,
            steps: Vec::new(),
        };
        self.propagate_assign_sources(&dst, sources, state)
    }

    pub(in super::super) fn record_closure_body_loans(
        &mut self,
        expr: ExprId,
        closure_params: &[u32],
        closure_depth: u32,
        body_state: &State,
        state: &mut State,
        (moves_capture, mutates_capture, borrows_local): (bool, bool, bool),
    ) {
        let mut outer_loans = Vec::new();
        let mut param_loans = Vec::new();
        for l in &body_state.loans {
            let Some(holder) = l.holder else { continue };
            if is_temp_holder(Some(holder))
                || self.scope_depth.get(holder as usize).copied().unwrap_or(0) >= closure_depth
            {
                continue;
            }
            if let Some(param_idx) = closure_params.iter().position(|&p| p == l.target) {
                let target_ty = self.local_ty(l.target);
                let copies_inner_ref = l.steps.as_slice() == [PlaceStep::Deref]
                    && target_ty != crate::sema::ty::Ty::NONE
                    && matches!(*self.cx.types.kind(target_ty), Kind::Ref(_, inner) if self.is_ref_ty(inner));
                let derefs_past_outer_ref = l
                    .steps
                    .iter()
                    .filter(|s| matches!(s, PlaceStep::Deref))
                    .count()
                    >= 2;
                param_loans.push(ClosureParamLoan {
                    param_idx,
                    holder,
                    copies_inner_ref,
                    derefs_past_outer_ref,
                });
            } else if is_temp_target(l.target).is_none()
                && self
                    .scope_depth
                    .get(l.target as usize)
                    .copied()
                    .unwrap_or(0)
                    < closure_depth
            {
                outer_loans.push((holder, BorrowSource::from_loan(l)));
            }
        }
        for (holder, src) in &outer_loans {
            push_loan(
                &mut state.loans,
                src.target,
                src.steps.clone(),
                Some(*holder),
                src.mutable,
                src.through_ref,
                false,
                src.env_only,
                Some(expr),
            );
        }
        self.closure_effects.insert(
            expr,
            ClosureEffects {
                moves_capture,
                mutates_capture,
                borrows_local,
                outer_loans,
                param_loans,
            },
        );
    }

    pub(in super::super) fn apply_closure_call_loans(
        &mut self,
        callee: ExprId,
        arg_ids: &[u32],
        arg_sources: &[Vec<BorrowSource>],
        state: &mut State,
    ) -> Result<()> {
        if let Some(c_expr) = self.closure_expr_of(callee)
            && let Some(eff) = self.closure_effects.get(&c_expr).cloned()
        {
            for (holder, src) in &eff.outer_loans {
                self.propagate_closure_param_loan(*holder, std::slice::from_ref(src), state)?;
            }
            for pl in &eff.param_loans {
                if let Some(srcs) = arg_sources.get(pl.param_idx) {
                    let sources = if pl.copies_inner_ref {
                        self.unwrap_outer_container_loans(srcs.clone(), state, false)
                    } else {
                        srcs.clone()
                    };
                    self.propagate_closure_param_loan(pl.holder, &sources, state)?;
                }
            }
            Ok(())
        } else {
            self.attach_closure_arg_param_loans(None, arg_ids, arg_sources, state)
        }
    }

    pub(in super::super) fn check_coercion_vtables(&self) -> Result<()> {
        for (&coercion_expr, coercion) in &self.typed.coercions {
            let Some(vtable) = coercion.vtable() else {
                continue;
            };
            let Kind::Closure(parent, _, at) = *self.cx.types.kind(vtable.ty) else {
                continue;
            };
            if parent != self.item {
                continue;
            }
            let Kind::Dyn(of, ..) = *self.cx.types.kind(vtable.object) else {
                continue;
            };
            let Some(eff) = self.closure_effects.get(&ExprId(at)) else {
                continue;
            };
            let lang = self.cx.lang(of);
            self.check_closure_effects(
                Some(coercion_expr),
                eff,
                lang == Some("fn"),
                lang == Some("fn_mut"),
                false,
            )?;
        }
        Ok(())
    }
}

fn is_prim_name(name: &str) -> bool {
    matches!(
        name,
        "bool"
            | "char"
            | "str"
            | "u8"
            | "u16"
            | "u32"
            | "u64"
            | "u128"
            | "usize"
            | "i8"
            | "i16"
            | "i32"
            | "i64"
            | "i128"
            | "isize"
            | "f32"
            | "f64"
            | "String"
    )
}

fn ast_ty_may_carry_elided_lt(body: &Body, src: &Source, ty_id: TyId) -> bool {
    match *body.ty(ty_id) {
        AstTy::Never | AstTy::Ptr { .. } => false,
        AstTy::Paren(t) | AstTy::Slice(t) | AstTy::Array(t, _) | AstTy::Pat(t, _) => {
            ast_ty_may_carry_elided_lt(body, src, t)
        }
        AstTy::Tuple(list) => body
            .list(list)
            .iter()
            .any(|&t| ast_ty_may_carry_elided_lt(body, src, TyId(t))),
        AstTy::Ref {
            lifetime: Some(lt),
            ty,
            ..
        } if src.text(lt) == "'static" => ast_ty_may_carry_elided_lt(body, src, ty),
        AstTy::Path(path_id) => {
            let p = body.path(path_id);
            let segs = body.list(p.segments);
            if p.qself.is_some() || segs.len() != 1 {
                return true;
            }
            let seg = body.segments[segs[0] as usize];
            let name = src.text(seg.name);
            if seg.args.is_none() && is_prim_name(name) {
                return false;
            }
            if matches!(name, "Option" | "Result" | "Vec" | "Box")
                && let Some(Args::Angle(args)) = seg.args
            {
                return body
                    .list(args)
                    .iter()
                    .any(|&a| match body.args[a as usize] {
                        GenericArg::Ty(t) | GenericArg::Binding { ty: t, .. } => {
                            ast_ty_may_carry_elided_lt(body, src, t)
                        }
                        GenericArg::Lifetime(lt) => src.text(lt) != "'static",
                        GenericArg::Const(_) => false,
                        GenericArg::Constraint { .. } => true,
                    });
            }
            true
        }
        _ => true,
    }
}

pub(in super::super) fn sig_returns_only_static(
    sig: &Signature,
    src: &Source,
    ret_ast: TyId,
    lts: &Lifetimes,
) -> bool {
    if lts.has_elided || !lts.named.is_empty() || !lts.saw_static {
        return false;
    }
    let type_params: Vec<&str> = sig
        .generics
        .iter()
        .filter(|g| g.kind == GenericKind::Type)
        .map(|g| src.text(g.name))
        .collect();
    !ast_ty_has_type_param(&sig.body, src, ret_ast, &type_params)
}

fn ast_ty_has_type_param(body: &Body, src: &Source, ty_id: TyId, params: &[&str]) -> bool {
    match *body.ty(ty_id) {
        AstTy::Ref { ty, .. }
        | AstTy::Ptr { ty, .. }
        | AstTy::Slice(ty)
        | AstTy::Array(ty, _)
        | AstTy::Paren(ty)
        | AstTy::Pat(ty, _) => ast_ty_has_type_param(body, src, ty, params),
        AstTy::Tuple(list) => body
            .list(list)
            .iter()
            .any(|&t| ast_ty_has_type_param(body, src, TyId(t), params)),
        AstTy::Path(path_id) => {
            let p = body.path(path_id);
            if p.qself
                .is_some_and(|(q, _)| ast_ty_has_type_param(body, src, q, params))
            {
                return true;
            }
            let segs = body.list(p.segments);
            if p.qself.is_none() && segs.len() == 1 {
                let name = src.text(body.segments[segs[0] as usize].name);
                if name == "Self" || params.contains(&name) {
                    return true;
                }
            }
            segs.iter().any(|&s| match body.segments[s as usize].args {
                Some(Args::Angle(args)) => {
                    body.list(args)
                        .iter()
                        .any(|&a| match body.args[a as usize] {
                            GenericArg::Ty(t) | GenericArg::Binding { ty: t, .. } => {
                                ast_ty_has_type_param(body, src, t, params)
                            }
                            _ => false,
                        })
                }
                Some(Args::Paren { inputs, output }) => {
                    body.list(inputs)
                        .iter()
                        .any(|&t| ast_ty_has_type_param(body, src, TyId(t), params))
                        || output.is_some_and(|o| ast_ty_has_type_param(body, src, o, params))
                }
                _ => false,
            })
        }
        AstTy::Infer | AstTy::ImplTrait(..) | AstTy::DynTrait(..) => true,
        _ => false,
    }
}

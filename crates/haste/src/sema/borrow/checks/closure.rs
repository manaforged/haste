use anyhow::Result;

use crate::body::sig::{GenericKind, Signature};
use crate::body::{Args, Bound, Expr, ExprId, List, Ty as AstTy, TyId};
use crate::outline::source::Source;
use crate::sema::ty::Kind;

use super::super::bind::push_loan;
use super::super::calls::{
    Lifetimes, ast_ty_may_carry_elided_lt, collect_ty_lifetimes, for_each_param_bound_list,
    sig_param_idx, sig_returns_only_static, sig_type_params, ty_shares_name,
};
use super::super::places::{Place, PlaceStep};
use super::super::propagate::{BorrowSource, TupleSources, dedup_sources};
use super::super::{Checker, Loan, State, is_temp_holder};
use super::is_temp_target;

#[derive(Clone)]
struct CallerLtReq {
    outer_hrtb: bool,
    requires_static: bool,
    named_lts: Vec<String>,
    elems: Vec<Option<CallerLtReq>>,
}

impl CallerLtReq {
    fn merge(&mut self, other: Self) {
        self.requires_static |= other.requires_static;
        self.outer_hrtb &= other.outer_hrtb;
        for lt in other.named_lts {
            if !self.named_lts.contains(&lt) {
                self.named_lts.push(lt);
            }
        }
        if self.elems.len() == other.elems.len() {
            for (dst, src_req) in self.elems.iter_mut().zip(other.elems) {
                if let Some(sr) = src_req {
                    match dst {
                        Some(d) => d.merge(sr),
                        None => *dst = Some(sr),
                    }
                }
            }
        } else {
            self.elems.clear();
        }
    }
}

struct CallerLtParam {
    param_idx: usize,
    req: CallerLtReq,
}

#[derive(Default)]
struct ParamBounds {
    req_fn: bool,
    req_fn_mut: bool,
    req_static: bool,
    hrtb_params: Vec<usize>,
    hrtb_non_ret: Vec<usize>,
    caller_lt_params: Vec<CallerLtParam>,
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
    pub(in super::super) ret_params: Vec<usize>,
}

impl Checker<'_, '_> {
    pub(in super::super) fn check_call_bounds(
        &mut self,
        arg_ids: &[u32],
        arg_sources: &[Vec<BorrowSource>],
        sig: Option<&Signature>,
        fn_item: Option<crate::names::ItemRef>,
        has_self_arg: bool,
    ) -> Result<()> {
        let Some(sig) = sig else { return Ok(()) };
        let sig_src = fn_item.map_or(self.src, |item| self.cx.item(item).0);
        for (idx, &id) in arg_ids.iter().enumerate() {
            let Some(param_idx) = sig_param_idx(idx, has_self_arg) else {
                continue;
            };
            let Some(&(_, param_ty)) = sig.params.get(param_idx) else {
                continue;
            };
            let arg = ExprId(id);
            let ref_static = matches!(*sig.body.ty(param_ty), AstTy::Ref { lifetime: Some(lt), .. } if sig_src.text(lt) == "'static");
            let bounds = self.param_bound_reqs(sig, sig_src, param_ty);
            if bounds.req_static || ref_static {
                let sources = arg_sources.get(idx).map_or(&[][..], Vec::as_slice);
                self.check_caller_lt_sources(arg, sources, true, &[], false)?;
            }
            if bounds.req_fn
                || bounds.req_fn_mut
                || bounds.req_static
                || !bounds.hrtb_params.is_empty()
                || !bounds.hrtb_non_ret.is_empty()
            {
                self.check_closure_arg_reqs(
                    arg,
                    bounds.req_fn,
                    bounds.req_fn_mut,
                    bounds.req_static,
                    &bounds.hrtb_params,
                    &bounds.hrtb_non_ret,
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
        arg_sources: &[Vec<BorrowSource>],
        state: &State,
    ) -> Result<()> {
        let Some(bounds) = self.callee_param_bounds(callee) else {
            return Ok(());
        };
        for param in bounds.caller_lt_params {
            let Some(&id) = arg_ids.get(param.param_idx) else {
                continue;
            };
            let pre = arg_sources.get(param.param_idx).map(Vec::as_slice);
            self.check_caller_lt_req(ExprId(id), &param.req, pre, state)?;
        }
        Ok(())
    }

    fn check_caller_lt_req(
        &mut self,
        expr: ExprId,
        req: &CallerLtReq,
        pre_sources: Option<&[BorrowSource]>,
        state: &State,
    ) -> Result<()> {
        let arg = self.unwrap_parens_and_empty_blocks(expr);
        if !req.elems.is_empty() {
            match *self.body.expr(arg) {
                Expr::Tuple(aexprs) if aexprs.len as usize == req.elems.len() => {
                    for (elem_req, elem_id) in req.elems.iter().zip(self.body.list(aexprs).to_vec())
                    {
                        if let Some(er) = elem_req {
                            self.check_caller_lt_req(ExprId(elem_id), er, None, state)?;
                        }
                    }
                    return Ok(());
                }
                Expr::If(_, then, Some(alt)) => {
                    self.check_caller_lt_req(then, req, None, state)?;
                    return self.check_caller_lt_req(alt, req, None, state);
                }
                Expr::Path(_) => {
                    if let Some(local) = self.expr_plain_local(arg)
                        && let Some(ts) = state
                            .tuple_elem_sources
                            .get(&local)
                            .filter(|ts| {
                                matches!(ts, TupleSources::Tuple(e) if e.len() == req.elems.len())
                            })
                            .cloned()
                    {
                        return self.check_caller_lt_tuple_sources(arg, req, &ts);
                    }
                }
                _ => {}
            }
        }
        let named: Vec<&str> = req.named_lts.iter().map(String::as_str).collect();
        if req.outer_hrtb
            && let Expr::Ref {
                raw: false,
                expr: inner,
                ..
            } = *self.body.expr(arg)
        {
            return self.check_caller_lt_arg(inner, state, req.requires_static, &named, false);
        }
        if let Some(sources) = pre_sources {
            return self.check_caller_lt_sources(
                arg,
                sources,
                req.requires_static,
                &named,
                req.outer_hrtb,
            );
        }
        self.check_caller_lt_arg(arg, state, req.requires_static, &named, req.outer_hrtb)
    }

    fn check_caller_lt_tuple_sources(
        &mut self,
        expr: ExprId,
        req: &CallerLtReq,
        ts: &TupleSources,
    ) -> Result<()> {
        if !req.elems.is_empty()
            && let TupleSources::Tuple(elems) = ts
            && elems.len() == req.elems.len()
        {
            for (elem_req, elem_ts) in req.elems.iter().zip(elems) {
                if let Some(er) = elem_req {
                    self.check_caller_lt_tuple_sources(expr, er, elem_ts)?;
                }
            }
            return Ok(());
        }
        let named: Vec<&str> = req.named_lts.iter().map(String::as_str).collect();
        let (srcs, next_outer_hrtb) = ts.sources_for(req.outer_hrtb);
        self.check_caller_lt_sources(expr, &srcs, req.requires_static, &named, next_outer_hrtb)
    }

    fn param_bound_reqs(&self, sig: &Signature, src: &Source, param_ty: TyId) -> ParamBounds {
        let mut out = ParamBounds::default();
        let outer_params = sig_type_params(sig, src);
        let check_bounds = |list: List| {
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
                                    ast_ty_may_carry_elided_lt(
                                        &sig.body,
                                        src,
                                        out_ty,
                                        &outer_params,
                                    )
                                });
                                let mut out_lts = Lifetimes::default();
                                if let Some(out_ty) = output {
                                    collect_ty_lifetimes(&sig.body, src, out_ty, &mut out_lts);
                                }
                                for (p_idx, &in_ty) in sig.body.list(inputs).iter().enumerate() {
                                    let (in_lts, outer_named_hrtb, req) =
                                        caller_lt_req_for_ty(sig, src, TyId(in_ty));
                                    if let Some(req) = req {
                                        if let Some(existing) = out
                                            .caller_lt_params
                                            .iter_mut()
                                            .find(|c| c.param_idx == p_idx)
                                        {
                                            existing.req.merge(req);
                                        } else {
                                            out.caller_lt_params.push(CallerLtParam {
                                                param_idx: p_idx,
                                                req,
                                            });
                                        }
                                    }
                                    let is_hrtb = (in_lts.has_elided && in_lts.named.is_empty())
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
        for_each_param_bound_list(sig, src, param_ty, check_bounds);
        out
    }

    pub(in super::super) fn check_static_arg(&mut self, expr: ExprId, state: &State) -> Result<()> {
        self.check_caller_lt_arg(expr, state, true, &[], false)
    }

    fn check_caller_lt_arg(
        &mut self,
        expr: ExprId,
        state: &State,
        requires_static: bool,
        req_named_lts: &[&str],
        outer_hrtb: bool,
    ) -> Result<()> {
        let saved_static = self.returns_only_static;
        self.returns_only_static = true;
        let sources = self.expr_borrow_sources(expr, state);
        self.returns_only_static = saved_static;
        self.check_caller_lt_sources(expr, &sources, requires_static, req_named_lts, outer_hrtb)
    }

    fn check_caller_lt_sources(
        &mut self,
        expr: ExprId,
        sources: &[BorrowSource],
        requires_static: bool,
        req_named_lts: &[&str],
        outer_hrtb: bool,
    ) -> Result<()> {
        let in_const = self.is_const_or_static_item();
        let plain = self.expr_plain_local(expr);
        for s in sources {
            if is_temp_target(s.target).is_some() {
                if in_const && !self.is_non_promotable_temp(expr) {
                    continue;
                }
                return Err(self.does_not_live_long_enough(Some(expr), s.target));
            }
            if let Some(origins) = self.closure_param_origins.get(&s.target).cloned() {
                self.check_caller_lt_sources(
                    expr,
                    &origins,
                    requires_static,
                    req_named_lts,
                    outer_hrtb,
                )?;
                continue;
            }
            let peel_outer = outer_hrtb
                && self.is_ref_ty(self.local_ty(s.target))
                && (plain == Some(s.target) || s.through_ref);
            let lt_mismatch = if requires_static {
                self.is_non_static_param(s.target)
                    && !(peel_outer && self.param_inner_is_static(s.target))
            } else {
                !req_named_lts.is_empty()
                    && !self.param_satisfies_caller_lt(s.target, req_named_lts, peel_outer)
            };
            if !self.from_caller(s.target, s.through_ref) || lt_mismatch {
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

    pub(super) fn check_closure_effects(
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
        &mut self,
        expr: ExprId,
        req_fn: bool,
        req_fn_mut: bool,
        req_static: bool,
        hrtb_params: &[usize],
        hrtb_non_ret: &[usize],
    ) -> Result<()> {
        let Some(c_expr) = self.closure_expr_of(expr) else {
            if !hrtb_non_ret.is_empty() && self.fn_item_returns_hrtb_param(expr, hrtb_non_ret) {
                return Err(self.borrow_err(Some(expr), "borrowed data escapes outside of closure"));
            }
            return Ok(());
        };
        let Some(eff) = self.closure_effects.get(&c_expr) else {
            return Ok(());
        };
        if eff
            .param_loans
            .iter()
            .any(|pl| hrtb_params.contains(&pl.param_idx) && !pl.derefs_past_outer_ref)
            || eff.ret_params.iter().any(|p| hrtb_non_ret.contains(p))
        {
            return Err(self.borrow_err(Some(expr), "borrowed data escapes outside of closure"));
        }
        self.check_closure_effects(Some(expr), eff, req_fn, req_fn_mut, req_static)
    }

    fn fn_item_returns_hrtb_param(&mut self, arg: ExprId, hrtb_non_ret: &[usize]) -> bool {
        let cur = self.unwrap_parens(arg);
        let Some(crate::sema::infer::Res::Fn(f_item, _)) = self.typed.res.get(&cur).copied() else {
            return false;
        };
        let Ok(sig) = self.cx.signature(f_item) else {
            return false;
        };
        let Some(ret_ast) = sig.ret else {
            return false;
        };
        let sig_src = self.cx.item(f_item).0;
        let mut ret_lts = Lifetimes::default();
        collect_ty_lifetimes(&sig.body, sig_src, ret_ast, &mut ret_lts);
        if sig_returns_only_static(&sig, sig_src, ret_ast, &ret_lts) {
            return false;
        }
        let has_ret_lt = ret_lts.has_elided || !ret_lts.named.is_empty();
        hrtb_non_ret.iter().any(|&idx| {
            sig.params.get(idx).is_some_and(|&(_, pt)| {
                if has_ret_lt {
                    let mut plts = Lifetimes::default();
                    collect_ty_lifetimes(&sig.body, sig_src, pt, &mut plts);
                    if (ret_lts.has_elided && (plts.has_elided || !plts.named.is_empty()))
                        || ret_lts.named.iter().any(|n| plts.named.contains(n))
                    {
                        return true;
                    }
                }
                !matches!(sig.body.ty(pt), AstTy::Ref { .. })
                    && ty_shares_name(&sig.body, sig_src, pt, &ret_lts.names)
            })
        })
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
        ret_params: Vec<usize>,
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
                let copies_inner_ref = !l.steps.is_empty()
                    && l.steps.iter().all(|s| matches!(s, PlaceStep::Deref))
                    && target_ty != crate::sema::ty::Ty::NONE
                    && matches!(
                        *self.cx.types.kind(target_ty),
                        Kind::Ref(_, inner)
                            if matches!(*self.cx.types.kind(inner), Kind::Ref(crate::sema::ty::Mut::Not, _))
                    );
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
                Loan {
                    target: src.target,
                    steps: src.steps.clone(),
                    holder: Some(*holder),
                    mutable: src.mutable,
                    through_ref: src.through_ref,
                    two_phase: false,
                    env_only: src.env_only,
                    origin_only: src.origin_only,
                    site: Some(expr),
                },
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
                ret_params,
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
}

fn sig_has_lifetime(sig: &Signature, src: &Source, name: &str) -> bool {
    sig.generics
        .iter()
        .any(|g| g.kind == GenericKind::Lifetime && src.text(g.name) == name)
}

fn caller_lt_req_for_ty<'s>(
    sig: &Signature,
    src: &'s Source,
    mut ty_id: TyId,
) -> (Lifetimes<'s>, Option<&'s str>, Option<CallerLtReq>) {
    while let AstTy::Paren(inner) = *sig.body.ty(ty_id) {
        ty_id = inner;
    }
    let mut in_lts = Lifetimes::default();
    collect_ty_lifetimes(&sig.body, src, ty_id, &mut in_lts);
    let outer_elided = matches!(*sig.body.ty(ty_id), AstTy::Ref { lifetime: None, .. });
    let outer_named_hrtb = match *sig.body.ty(ty_id) {
        AstTy::Ref {
            lifetime: Some(lt), ..
        } => {
            let name = src.text(lt);
            (name != "'static" && !sig_has_lifetime(sig, src, name)).then_some(name)
        }
        _ => None,
    };
    let caller_named: Vec<String> = in_lts
        .named
        .iter()
        .filter(|&&lt| sig_has_lifetime(sig, src, lt))
        .map(|&lt| lt.to_owned())
        .collect();
    let req = (in_lts.saw_static || !caller_named.is_empty()).then(|| {
        let elems = match *sig.body.ty(ty_id) {
            AstTy::Tuple(list) => sig
                .body
                .list(list)
                .iter()
                .map(|&t| caller_lt_req_for_ty(sig, src, TyId(t)).2)
                .collect(),
            _ => Vec::new(),
        };
        CallerLtReq {
            outer_hrtb: outer_elided || outer_named_hrtb.is_some(),
            requires_static: in_lts.saw_static,
            named_lts: caller_named,
            elems,
        }
    });
    (in_lts, outer_named_hrtb, req)
}

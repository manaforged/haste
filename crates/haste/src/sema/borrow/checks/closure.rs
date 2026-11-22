use anyhow::{Result, bail};

use crate::body::sig::{GenericKind, Signature};
use crate::body::{Args, Bound, Expr, ExprId, List, Ty as AstTy, TyId};
use crate::outline::source::Source;
use crate::sema::infer::Coercion;
use crate::sema::ty::Kind;

use super::super::bind::push_loan;
use super::super::places::{Place, PlaceStep};
use super::super::propagate::{BorrowSource, dedup_sources};
use super::super::{Checker, State, is_temp_holder};
use super::is_temp_target;

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
                let Some(p) = idx.checked_sub(1) else { continue };
                p
            } else {
                idx
            };
            let Some(&(_, param_ty)) = sig.params.get(param_idx) else { continue };
            let arg = ExprId(id);
            let ref_static = matches!(*sig.body.ty(param_ty), AstTy::Ref { lifetime: Some(lt), .. } if sig_src.text(lt) == "'static");
            let (req_fn, req_fn_mut, req_static, hrtb_params) = self.param_bound_reqs(sig, sig_src, param_ty);
            if req_static || ref_static {
                self.check_static_arg(arg, state)?;
            }
            if req_fn || req_fn_mut || req_static || !hrtb_params.is_empty() {
                self.check_closure_arg_reqs(arg, req_fn, req_fn_mut, req_static, &hrtb_params)?;
            }
        }
        Ok(())
    }

    fn param_bound_reqs(&self, sig: &Signature, src: &Source, param_ty: TyId) -> (bool, bool, bool, Vec<usize>) {
        let (mut req_fn, mut req_fn_mut, mut req_static) = (false, false, false);
        let mut hrtb_params = Vec::new();
        let mut check_bounds = |list: List| {
            for &b_id in sig.body.list(list) {
                match sig.body.bounds[b_id as usize] {
                    Bound::Lifetime(lt) if src.text(lt) == "'static" => req_static = true,
                    Bound::Trait { path, .. } => {
                        if let Some(&seg_id) = sig.body.list(sig.body.path(path).segments).last() {
                            let seg = sig.body.segments[seg_id as usize];
                            let is_callable = match src.text(seg.name) {
                                "Fn" => {
                                    req_fn = true;
                                    true
                                }
                                "FnMut" => {
                                    req_fn_mut = true;
                                    true
                                }
                                "FnOnce" => true,
                                _ => false,
                            };
                            if is_callable && let Some(Args::Paren { inputs, .. }) = seg.args {
                                for (p_idx, &in_ty) in sig.body.list(inputs).iter().enumerate() {
                                    if matches!(*sig.body.ty(TyId(in_ty)), AstTy::Ref { lifetime: None, .. })
                                        && !hrtb_params.contains(&p_idx)
                                    {
                                        hrtb_params.push(p_idx);
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
                            if psegs.len() == 1 && src.text(sig.body.segments[psegs[0] as usize].name) == name {
                                check_bounds(pred.bounds);
                            }
                        }
                    }
                }
            }
            _ => {}
        }
        (req_fn, req_fn_mut, req_static, hrtb_params)
    }

    fn check_static_arg(&mut self, expr: ExprId, state: &State) -> Result<()> {
        for s in self.expr_borrow_sources(expr, state) {
            if is_temp_target(s.target).is_some() || !self.from_caller(s.target, s.through_ref) {
                bail!("`{}` does not live long enough", self.local_name(s.target));
            }
        }
        Ok(())
    }

    fn check_closure_effects(&self, eff: &ClosureEffects, req_fn: bool, req_fn_mut: bool, req_static: bool) -> Result<()> {
        if (req_fn || req_fn_mut) && eff.moves_capture {
            bail!("cannot move out of a captured variable in an `Fn` or `FnMut` closure");
        }
        if req_fn && eff.mutates_capture {
            bail!("cannot borrow captured variable as mutable in an `Fn` closure");
        }
        if req_static && eff.borrows_local {
            bail!("closure may outlive the current function, but it borrows a local variable");
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
        let Some(c_expr) = self.closure_expr_of(expr) else { return Ok(()) };
        let Some(eff) = self.closure_effects.get(&c_expr) else { return Ok(()) };
        if eff.param_loans.iter().any(|pl| hrtb_params.contains(&pl.param_idx) && !pl.derefs_past_outer_ref) {
            bail!("borrowed data escapes outside of closure");
        }
        self.check_closure_effects(eff, req_fn, req_fn_mut, req_static)
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
            let Some(c_expr) = self.closure_expr_of(ExprId(id)) else { continue };
            let Some(eff) = self.closure_effects.get(&c_expr).cloned() else { continue };
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
                if other_idx != idx && self.closure_expr_of(ExprId(other_id)).is_none()
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
                let derefs_past_outer_ref = l.steps.iter().filter(|s| matches!(s, PlaceStep::Deref)).count() >= 2;
                param_loans.push(ClosureParamLoan {
                    param_idx,
                    holder,
                    copies_inner_ref,
                    derefs_past_outer_ref,
                });
            } else if is_temp_target(l.target).is_none()
                && self.scope_depth.get(l.target as usize).copied().unwrap_or(0) < closure_depth
            {
                outer_loans.push((holder, BorrowSource::from_loan(l)));
            }
        }
        for (holder, src) in &outer_loans {
            push_loan(&mut state.loans, src.target, src.steps.clone(), Some(*holder), src.mutable, src.through_ref, false, src.env_only);
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
        for coercion in self.typed.coercions.values() {
            let Coercion::Vtable(vtable) = coercion else { continue };
            let Kind::Closure(parent, _, at) = *self.cx.types.kind(vtable.ty) else { continue };
            if parent != self.item {
                continue;
            }
            let Kind::Dyn(of, ..) = *self.cx.types.kind(vtable.object) else { continue };
            let Some(eff) = self.closure_effects.get(&ExprId(at)) else { continue };
            let lang = self.cx.lang(of);
            self.check_closure_effects(eff, lang == Some("fn"), lang == Some("fn_mut"), false)?;
        }
        Ok(())
    }
}

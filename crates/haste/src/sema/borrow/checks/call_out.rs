use anyhow::Result;

use crate::body::sig::{GenericKind, Signature};
use crate::body::{Args, Body, Bound, Expr, ExprId, GenericArg, Ty as AstTy, TyId};
use crate::outline::source::Source;
use crate::sema::ty::{Kind, Mut, Ty};

use super::super::calls::lifetimes::collect_ty_lifetimes;
use super::super::calls::{Lifetimes, recv_ref_mut};
use super::super::checks::{collect_ast_invariant_lifetimes, collect_ast_variance};
use super::super::places::{Place, PlaceStep};
use super::super::propagate::BorrowSource;
use super::super::{Checker, State};

enum CallOutDst {
    Place {
        place: Place,
        match_ty: TyId,
        extra_ptr_targets: Vec<u32>,
    },
    Value {
        arg: ExprId,
        match_ty: TyId,
    },
}

impl Checker<'_, '_> {
    pub(in super::super) fn attach_call_out_sources(
        &mut self,
        expr: ExprId,
        arg_ids: &[u32],
        arg_sources: &[Vec<BorrowSource>],
        sig: Option<&Signature>,
        fn_item: Option<crate::names::ItemRef>,
        has_self_arg: bool,
        state: &mut State,
    ) -> Result<()> {
        let Some(sig) = sig else { return Ok(()) };
        if has_self_arg && !arg_ids.is_empty() {
            let recv_mut = recv_ref_mut(sig) == Some(Mut::Mut);
            let recv_expr = match *self.body.expr(self.unwrap_parens(ExprId(arg_ids[0]))) {
                Expr::Ref {
                    raw: false,
                    expr: inner,
                    ..
                } if recv_ref_mut(sig).is_some() => inner,
                _ => ExprId(arg_ids[0]),
            };
            self.attach_method_arg_sources(
                expr,
                recv_expr,
                &arg_ids[1..],
                arg_sources.get(1..).unwrap_or(&[]),
                recv_mut,
                (Some(sig), fn_item),
                state,
            )?;
        }
        let param_args = if has_self_arg {
            arg_ids.get(1..).unwrap_or(&[])
        } else {
            arg_ids
        };
        let param_arg_sources = if has_self_arg {
            arg_sources.get(1..).unwrap_or(&[])
        } else {
            arg_sources
        };
        if param_args.len() < 2 || sig.params.len() < 2 {
            return Ok(());
        }
        let fn_src = fn_item.map_or(self.src, |item| self.cx.item(item).0);
        for (dst_idx, &dst_id) in param_args.iter().enumerate() {
            let Some(&(_, dst_ast)) = sig.params.get(dst_idx) else {
                continue;
            };
            let Some(dst) = self.call_out_dst(sig, fn_item, fn_src, dst_ast, ExprId(dst_id), state)
            else {
                continue;
            };
            for (src_idx, &src_id) in param_args.iter().enumerate() {
                if src_idx == dst_idx {
                    continue;
                }
                let Some(&(_, src_ast)) = sig.params.get(src_idx) else {
                    continue;
                };
                self.check_call_out_pair(
                    sig,
                    fn_src,
                    &dst,
                    (ExprId(src_id), src_ast, param_arg_sources.get(src_idx)),
                    state,
                )?;
            }
        }
        Ok(())
    }

    fn call_out_dst(
        &mut self,
        sig: &Signature,
        fn_item: Option<crate::names::ItemRef>,
        fn_src: &Source,
        dst_ast: TyId,
        dst_arg: ExprId,
        state: &State,
    ) -> Option<CallOutDst> {
        let arg_unwrapped = self.unwrap_parens(dst_arg);
        match *sig.body.ty(dst_ast) {
            AstTy::Ref {
                mutable: true,
                ty: dst_inner,
                ..
            } => self.call_out_mut_ref_dst(
                sig,
                fn_item,
                fn_src,
                dst_ast,
                dst_inner,
                dst_arg,
                arg_unwrapped,
            ),
            AstTy::Ref {
                mutable: false,
                ty: dst_inner,
                ..
            } => self.call_out_shared_ref_dst(dst_inner, dst_arg, arg_unwrapped),
            AstTy::Ptr {
                mutable: true,
                ty: dst_inner,
            } => self.call_out_mut_ptr_dst(dst_inner, dst_arg, arg_unwrapped, state),
            _ => {
                let dst_var = collect_ast_variance(
                    self.cx,
                    fn_item,
                    &sig.body,
                    fn_src,
                    dst_ast,
                    Some(self.expr_ty(dst_arg)),
                );
                (!dst_var.inv.is_empty() || !dst_var.contra.is_empty()).then_some(
                    CallOutDst::Value {
                        arg: arg_unwrapped,
                        match_ty: dst_ast,
                    },
                )
            }
        }
    }

    fn call_out_mut_ref_dst(
        &mut self,
        sig: &Signature,
        fn_item: Option<crate::names::ItemRef>,
        fn_src: &Source,
        dst_ast: TyId,
        dst_inner: TyId,
        dst_arg: ExprId,
        arg_unwrapped: ExprId,
    ) -> Option<CallOutDst> {
        let (mut place, pointee_ty) = match *self.body.expr(arg_unwrapped) {
            Expr::Ref {
                mutable: true,
                raw: false,
                expr: inner,
            } => {
                let p = self.root_place(inner)?;
                let ety = self.expr_ty(inner);
                let pty = if ety != Ty::NONE {
                    ety
                } else {
                    self.local_ty(p.local)
                };
                (p, pty)
            }
            _ if self.is_mut_ref_ty(self.expr_ty(dst_arg)) => {
                let ety = self.expr_ty(dst_arg);
                let mut p = self.root_place(arg_unwrapped)?;
                let pty = match *self.cx.types.kind(ety) {
                    Kind::Ref(_, pointee) => pointee,
                    _ => self.local_ty(p.local),
                };
                self.apply_deref_steps(dst_arg, ety, 1, &mut p);
                (p, pty)
            }
            _ => return None,
        };
        let inner_inv = collect_ast_invariant_lifetimes(
            self.cx,
            fn_item,
            &sig.body,
            fn_src,
            dst_ast,
            Some(self.expr_ty(dst_arg)),
        );
        if self.ty_holds_borrow(pointee_ty)
            || self.ty_has_borrow_interior_mut(pointee_ty)
            || !inner_inv.is_empty()
        {
            if self.is_mut_ptr_ty(pointee_ty) {
                place.through_ref = true;
            }
            Some(CallOutDst::Place {
                place,
                match_ty: dst_inner,
                extra_ptr_targets: Vec::new(),
            })
        } else {
            None
        }
    }

    fn call_out_shared_ref_dst(
        &mut self,
        dst_inner: TyId,
        dst_arg: ExprId,
        arg_unwrapped: ExprId,
    ) -> Option<CallOutDst> {
        let place = match *self.body.expr(arg_unwrapped) {
            Expr::Ref {
                raw: false,
                expr: inner,
                ..
            } => {
                let p = self.root_place(inner)?;
                let ety = self.expr_ty(inner);
                let pty = if ety != Ty::NONE {
                    ety
                } else {
                    self.local_ty(p.local)
                };
                if !self.ty_has_borrow_interior_mut(pty) {
                    return None;
                }
                p
            }
            _ => {
                let ety = self.expr_ty(dst_arg);
                if ety == Ty::NONE {
                    return None;
                }
                let Kind::Ref(_, pointee) = *self.cx.types.kind(ety) else {
                    return None;
                };
                if !self.ty_has_borrow_interior_mut(pointee) {
                    return None;
                }
                let mut p = self.root_place(arg_unwrapped)?;
                self.apply_deref_steps(dst_arg, ety, 1, &mut p);
                p
            }
        };
        Some(CallOutDst::Place {
            place,
            match_ty: dst_inner,
            extra_ptr_targets: Vec::new(),
        })
    }

    fn call_out_mut_ptr_dst(
        &mut self,
        dst_inner: TyId,
        dst_arg: ExprId,
        arg_unwrapped: ExprId,
        state: &State,
    ) -> Option<CallOutDst> {
        if let Expr::Ref { expr: inner, .. } = *self.body.expr(arg_unwrapped) {
            let place = self.root_place(inner)?;
            return Some(CallOutDst::Place {
                place,
                match_ty: dst_inner,
                extra_ptr_targets: Vec::new(),
            });
        }
        if self.is_mut_ptr_ty(self.expr_ty(dst_arg))
            && let Some(mut place) = self.root_place(arg_unwrapped)
        {
            self.apply_deref_steps(dst_arg, self.expr_ty(dst_arg), 1, &mut place);
            return Some(CallOutDst::Place {
                place,
                match_ty: dst_inner,
                extra_ptr_targets: Vec::new(),
            });
        }
        let mut ptr_targets = Vec::new();
        self.collect_ptr_targets_expr(dst_arg, state, &mut ptr_targets);
        let &dummy = ptr_targets.first()?;
        Some(CallOutDst::Place {
            place: Place {
                local: dummy,
                through_ref: true,
                through_deref: false,
                mut_ref_root: true,
                behind_shared: false,
                reborrows_shared_field: false,
                raw_ptr_prefix: None,
                steps: Vec::new(),
            },
            match_ty: dst_inner,
            extra_ptr_targets: ptr_targets,
        })
    }

    fn check_call_out_pair(
        &mut self,
        sig: &Signature,
        fn_src: &Source,
        dst: &CallOutDst,
        (src_id, src_ast, pre_sources): (ExprId, TyId, Option<&Vec<BorrowSource>>),
        state: &mut State,
    ) -> Result<()> {
        let match_ty = match dst {
            CallOutDst::Place { match_ty, .. } | CallOutDst::Value { match_ty, .. } => *match_ty,
        };
        if let (AstTy::Tuple(ptys), Expr::Tuple(aexprs)) = (
            *sig.body.ty(src_ast),
            *self.body.expr(self.unwrap_parens(src_id)),
        ) && ptys.len == aexprs.len
        {
            let pairs: Vec<(u32, u32)> = sig
                .body
                .list(ptys)
                .iter()
                .copied()
                .zip(self.body.list(aexprs).iter().copied())
                .collect();
            for (pt, ae) in pairs {
                self.check_call_out_pair(sig, fn_src, dst, (ExprId(ae), TyId(pt), None), state)?;
            }
            return Ok(());
        }
        if !self.sig_params_share_lt_or_ty(sig, fn_src, match_ty, src_ast) {
            return Ok(());
        }
        let src_expr = match (
            *sig.body.ty(src_ast),
            *self.body.expr(self.unwrap_parens(src_id)),
        ) {
            (
                AstTy::Ref {
                    lifetime: None,
                    mutable: true,
                    ty: src_inner,
                },
                Expr::Ref {
                    mutable: true,
                    raw: false,
                    expr,
                },
            ) if !matches!(*sig.body.ty(match_ty), AstTy::Ref { .. })
                && self.sig_params_share_lt_or_ty(sig, fn_src, match_ty, src_inner) =>
            {
                expr
            }
            _ => src_id,
        };
        let sources = if src_expr == src_id
            && let Some(pre) = pre_sources
        {
            pre.clone()
        } else {
            self.expr_borrow_sources(src_expr, state)
        };
        match dst {
            CallOutDst::Place {
                place,
                extra_ptr_targets,
                ..
            } => {
                if extra_ptr_targets.is_empty() {
                    self.propagate_assign_sources(place, &sources, state)?;
                    if let Some(src_local) = self.expr_plain_local(src_expr) {
                        self.check_local_place_fits_target(src_expr, place, src_local, state)?;
                    }
                } else {
                    if !sources.is_empty() {
                        self.check_ptr_deref_write(extra_ptr_targets, &sources, state)?;
                    }
                    if let Some(src_local) = self.expr_plain_local(src_expr) {
                        for &t in extra_ptr_targets {
                            self.check_src_local_fits_target_local(
                                src_expr,
                                src_local,
                                t,
                                (false, false),
                            )?;
                        }
                    }
                }
            }
            CallOutDst::Value { arg: dst_arg, .. } => {
                if let Some(dst_local) = self.expr_plain_local(*dst_arg) {
                    let (mut req_static, mut req_named) =
                        self.target_invariant_req(dst_local, false, false);
                    let dst_contra = self.target_contra_req(dst_local, false);
                    if dst_contra.saw_static {
                        req_static = true;
                    }
                    for lt in dst_contra.named {
                        if !req_named.contains(&lt) {
                            req_named.push(lt);
                        }
                    }
                    if req_static || !req_named.is_empty() {
                        self.check_sources_fit_invariant_lts(
                            Some(src_expr),
                            &sources,
                            req_static,
                            &req_named,
                        )?;
                        if let Some(src_local) = self.expr_plain_local(src_expr) {
                            self.check_src_local_fits_target_local(
                                src_expr,
                                src_local,
                                dst_local,
                                (false, false),
                            )?;
                        }
                    }
                } else {
                    let dst_var = self.expr_variance_lts(*dst_arg);
                    if dst_var.contra.saw_static || dst_var.inv.is_only_static() {
                        self.check_sources_fit_invariant_lts(Some(src_expr), &sources, true, &[])?;
                    }
                }
            }
        }
        Ok(())
    }

    fn check_src_local_fits_target_local(
        &mut self,
        site: ExprId,
        src_local: u32,
        target_local: u32,
        (peel_target, has_field): (bool, bool),
    ) -> Result<()> {
        let (req_static, req_named) =
            self.target_invariant_req(target_local, peel_target, has_field);
        let peel_src = self.is_ref_ty(self.local_ty(src_local));
        self.check_local_place_invariant_fit(
            Some(site),
            src_local,
            peel_src,
            req_static,
            &req_named,
        )
    }

    pub(super) fn check_local_place_fits_target(
        &mut self,
        site: ExprId,
        target: &Place,
        src_local: u32,
        state: &State,
    ) -> Result<()> {
        let has_field = target
            .steps
            .iter()
            .any(|st| matches!(st, PlaceStep::Field(..)));
        if !target.through_ref {
            return self.check_src_local_fits_target_local(
                site,
                src_local,
                target.local,
                (false, has_field),
            );
        }
        let is_ptr = self.is_mut_ptr_ty(self.local_ty(target.local));
        let loan_targets: Vec<u32> = state
            .loans
            .iter()
            .filter(|l| {
                l.holder == Some(target.local)
                    && (l.mutable
                        || is_ptr
                        || self.ty_has_borrow_interior_mut(self.local_ty(l.target)))
            })
            .map(|l| l.target)
            .collect();
        for t in loan_targets {
            self.check_src_local_fits_target_local(site, src_local, t, (false, false))?;
        }
        if is_ptr && let Some(ptr_tgts) = state.ptr_targets.get(&target.local).cloned() {
            for t in ptr_tgts {
                self.check_src_local_fits_target_local(site, src_local, t, (false, false))?;
            }
        }
        self.check_src_local_fits_target_local(site, src_local, target.local, (true, has_field))
    }

    fn sig_params_share_lt_or_ty(&self, sig: &Signature, src: &Source, a: TyId, b: TyId) -> bool {
        let mut lts_a = Lifetimes::default();
        let mut lts_b = Lifetimes::default();
        collect_ty_lifetimes(&sig.body, src, a, &mut lts_a);
        collect_ty_lifetimes(&sig.body, src, b, &mut lts_b);
        if lts_a.named.iter().any(|n| lts_b.named.contains(n)) {
            return true;
        }
        let mut b_value_names = Vec::new();
        collect_non_callable_input_names(&sig.body, src, b, &mut b_value_names);
        sig.generics.iter().any(|g| {
            g.kind == GenericKind::Type && {
                let name = src.text(g.name);
                lts_a.names.contains(&name) && b_value_names.contains(&name)
            }
        })
    }
}

fn collect_non_callable_input_names<'s>(
    body: &Body,
    src: &'s Source,
    ty: TyId,
    out: &mut Vec<&'s str>,
) {
    match *body.ty(ty) {
        AstTy::Ref { ty: inner, .. }
        | AstTy::Ptr { ty: inner, .. }
        | AstTy::Slice(inner)
        | AstTy::Array(inner, _)
        | AstTy::Paren(inner)
        | AstTy::Pat(inner, _) => collect_non_callable_input_names(body, src, inner, out),
        AstTy::Tuple(list) => {
            for &t in body.list(list) {
                collect_non_callable_input_names(body, src, TyId(t), out);
            }
        }
        AstTy::Path(path) => {
            let p = body.path(path);
            let segs = body.list(p.segments);
            if let Some(&first) = segs.first() {
                let name = src.text(body.segments[first as usize].name);
                if !out.contains(&name) {
                    out.push(name);
                }
            }
            for &seg_id in segs {
                collect_seg_non_callable_names(body, src, body.segments[seg_id as usize].args, out);
            }
        }
        AstTy::ImplTrait(_, list) | AstTy::DynTrait(list) => {
            for &b_id in body.list(list) {
                if let Bound::Trait { path, .. } = body.bounds[b_id as usize] {
                    for &seg_id in body.list(body.path(path).segments) {
                        collect_seg_non_callable_names(
                            body,
                            src,
                            body.segments[seg_id as usize].args,
                            out,
                        );
                    }
                }
            }
        }
        AstTy::Fn { ret: Some(r), .. } => collect_non_callable_input_names(body, src, r, out),
        _ => {}
    }
}

fn collect_seg_non_callable_names<'s>(
    body: &Body,
    src: &'s Source,
    args: Option<Args>,
    out: &mut Vec<&'s str>,
) {
    match args {
        Some(Args::Angle(list)) => {
            for &arg_id in body.list(list) {
                if let GenericArg::Ty(t) | GenericArg::Binding { ty: t, .. } =
                    body.args[arg_id as usize]
                {
                    collect_non_callable_input_names(body, src, t, out);
                }
            }
        }
        Some(Args::Paren {
            output: Some(o), ..
        }) => collect_non_callable_input_names(body, src, o, out),
        _ => {}
    }
}

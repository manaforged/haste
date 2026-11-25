use anyhow::Result;
use rustc_hash::FxHashSet as HashSet;

use crate::body::sig::{GenericKind, Signature};
use crate::body::{Args, Bound, Expr, ExprId, GenericArg, Ty as AstTy, TyId};
use crate::lower::is_freeze_ty;
use crate::names::ItemRef;
use crate::outline::item::ItemKind;
use crate::outline::source::Source;
use crate::sema::infer::{Coercion, Res};
use crate::sema::ty::{Kind, Mut, Ty};

use super::super::calls::{
    Lifetimes, collect_ty_lifetimes, lifetime_outlives, pred_matches_name, recv_is_by_ref,
    recv_ref_mut, ty_shares_name,
};
use super::super::propagate::{BorrowSource, unwrap_container_loans};
use super::super::{Checker, State};
use super::is_temp_target;

impl Checker<'_, '_> {
    pub(in super::super) fn attach_method_arg_sources(
        &mut self,
        expr: ExprId,
        receiver: ExprId,
        param_ids: &[u32],
        recv_mut: bool,
        sig: Option<&Signature>,
        state: &mut State,
    ) -> Result<()> {
        let Some(mut p) = self.root_place(receiver) else {
            return Ok(());
        };
        let adjust = self.typed.adjusts.get(&expr).copied();
        let derefs = adjust.map_or(0, |a| a.derefs);
        if derefs > 0 {
            self.apply_deref_steps(expr, self.expr_ty(receiver), derefs, &mut p);
        } else if adjust.and_then(|a| a.autoref).is_none()
            && self.is_mut_ref_ty(self.expr_ty(receiver))
            && sig.is_some_and(recv_is_by_ref)
        {
            self.apply_deref_steps(expr, self.expr_ty(receiver), 1, &mut p);
        }
        let self_ty = self.method_self_ty(expr, receiver, sig);
        let recv_ty = if self_ty != Ty::NONE {
            self_ty
        } else {
            self.local_ty(p.local)
        };
        if (!recv_mut && !self.ty_holds_closure(recv_ty)) || !self.ty_holds_borrow(recv_ty) {
            return Ok(());
        }
        let fn_item = match self.typed.res.get(&expr).copied() {
            Some(Res::Fn(item, _)) => Some(item),
            _ => None,
        };
        let keep_env = self.ty_holds_closure(recv_ty);
        let mut self_lts = Lifetimes::default();
        if let Some(item) = fn_item
            && let Some(block) = self.cx.owner_impl(item)
            && let Ok(hdr) = self.cx.impl_header(block)
        {
            let impl_src = &self.cx.impl_file(block).source;
            collect_ty_lifetimes(&hdr.body, impl_src, hdr.self_ty, &mut self_lts);
        }
        let mut all_sources = Vec::new();
        for (idx, &id) in param_ids.iter().enumerate() {
            let arg = ExprId(id);
            let arg_ty = self.expr_ty(arg);
            if arg_ty != Ty::NONE && !self.ty_holds_borrow(arg_ty) {
                continue;
            }
            let mut is_fresh_ref = false;
            let mut param_ty_info: Option<(ItemRef, TyId)> = None;
            if let Some(s) = sig
                && let Some(item) = fn_item
                && let Some(&(_, param_ty)) = s.params.get(idx)
            {
                let (sig_src, _) = self.cx.item(item);
                let mut p_lts = Lifetimes::default();
                collect_ty_lifetimes(&s.body, sig_src, param_ty, &mut p_lts);
                is_fresh_ref = matches!(s.body.ty(param_ty), AstTy::Ref { lifetime: None, .. });
                let shares_self = p_lts.named.iter().any(|n| self_lts.named.contains(n))
                    || (!is_fresh_ref
                        && ty_shares_name(&s.body, sig_src, param_ty, &self_lts.names))
                    || self.generic_param_ties_to_self(s, sig_src, &p_lts, &self_lts);
                if !shares_self {
                    continue;
                }
                param_ty_info = Some((item, param_ty));
            }
            let arg_sources = if let Some((item, param_ty)) = param_ty_info
                && let Some(s) = sig
                && let Some(filtered) =
                    self.matching_adt_arg_sources(expr, arg, s, item, param_ty, &self_lts, state)
            {
                filtered
            } else if is_fresh_ref {
                self.copied_arg_borrow_sources(expr, arg, state)
            } else {
                self.expr_borrow_sources(arg, state)
            };
            for s in self.unwrap_collection_arg_sources(arg, arg_ty, recv_ty, arg_sources, state) {
                if !s.env_only || keep_env {
                    all_sources.push(s);
                }
            }
        }
        if !all_sources.is_empty() {
            state.set_tuple_sources(p.local, None);
        }
        self.propagate_assign_sources(&p, &all_sources, state)
    }

    fn matching_adt_arg_sources(
        &mut self,
        call_expr: ExprId,
        arg: ExprId,
        sig: &Signature,
        fn_item: ItemRef,
        param_ty: TyId,
        self_lts: &Lifetimes<'_>,
        state: &State,
    ) -> Option<Vec<BorrowSource>> {
        if self_lts.named.is_empty() {
            return None;
        }
        let inner_param = match *sig.body.ty(param_ty) {
            AstTy::Ref {
                lifetime: None,
                ty: inner,
                ..
            } => inner,
            _ => param_ty,
        };
        let AstTy::Path(path_id) = *sig.body.ty(inner_param) else {
            return None;
        };
        let sig_src = self.cx.item(fn_item).0;
        let last_seg = *sig.body.list(sig.body.path(path_id).segments).last()?;
        let seg_name = sig_src.text(sig.body.segments[last_seg as usize].name);
        let Some(Args::Angle(angle)) = sig.body.segments[last_seg as usize].args else {
            return None;
        };
        let mut arg_lts = Vec::new();
        for &a in sig.body.list(angle) {
            match sig.body.args[a as usize] {
                GenericArg::Lifetime(lt) => arg_lts.push(sig_src.text(lt)),
                GenericArg::Ty(t) | GenericArg::Binding { ty: t, .. } => {
                    let mut t_lts = Lifetimes::default();
                    collect_ty_lifetimes(&sig.body, sig_src, t, &mut t_lts);
                    if t_lts.has_elided || !t_lts.named.is_empty() {
                        return None;
                    }
                }
                _ => {}
            }
        }
        if arg_lts.len() < 2
            || !arg_lts.iter().any(|&n| {
                !self_lts
                    .named
                    .iter()
                    .any(|&s_lt| n == s_lt || lifetime_outlives(sig, sig_src, n, s_lt))
            })
        {
            return None;
        }
        let val_expr = match *self.body.expr(self.unwrap_parens(arg)) {
            Expr::Ref {
                raw: false,
                expr: inner,
                ..
            } => self.unwrap_parens(inner),
            _ => self.unwrap_parens(arg),
        };
        let (struct_expr, is_inline) = match *self.body.expr(val_expr) {
            Expr::Struct { base: None, .. } => (val_expr, true),
            _ => {
                let local = self.expr_plain_local(val_expr)?;
                let mut site = None;
                for l in state.loans.iter().filter(|l| l.holder == Some(local)) {
                    let s = l.site?;
                    if site.is_some_and(|prev| prev != s) {
                        return None;
                    }
                    site = Some(s);
                }
                let s = site?;
                matches!(*self.body.expr(s), Expr::Struct { base: None, .. })
                    .then_some((s, false))?
            }
        };
        let Kind::Adt(adt_item, _) = *self.cx.types.kind(self.expr_ty(struct_expr)) else {
            return None;
        };
        if self.cx.name(adt_item) != seg_name
            || self.cx.file(fn_item).items.items.iter().any(|it| {
                it.kind == ItemKind::Type
                    && it
                        .name
                        .is_some_and(|n| self.cx.file(fn_item).source.text(n) == seg_name)
            })
        {
            return None;
        }
        let def = self.cx.adt_def(adt_item).ok()?;
        if def.variants.len() != 1
            || !def.generics.predicates.is_empty()
            || def
                .generics
                .params
                .iter()
                .any(|g| g.kind == GenericKind::Lifetime && g.bounds.len > 0)
        {
            return None;
        }
        let var = &def.variants[0];
        let adt_src = &self.cx.file(adt_item).source;
        let def_lts: Vec<&str> = def
            .generics
            .params
            .iter()
            .filter(|g| g.kind == GenericKind::Lifetime)
            .map(|g| adt_src.text(g.name))
            .collect();
        let def_ty_params: Vec<&str> = def
            .generics
            .params
            .iter()
            .filter(|g| g.kind == GenericKind::Type)
            .map(|g| adt_src.text(g.name))
            .collect();
        if def_lts.len() != arg_lts.len() {
            return None;
        }
        let shared_def_lts: Vec<&str> = def_lts
            .iter()
            .zip(&arg_lts)
            .filter(|(_, arg_lt)| {
                self_lts
                    .named
                    .iter()
                    .any(|&s_lt| **arg_lt == s_lt || lifetime_outlives(sig, sig_src, arg_lt, s_lt))
            })
            .map(|(&def_lt, _)| def_lt)
            .collect();
        let Expr::Struct { fields, .. } = *self.body.expr(struct_expr) else {
            return None;
        };
        let field_ids: Vec<u32> = self.body.list(fields).to_vec();
        let mut out = Vec::new();
        for id in field_ids {
            let field_expr = self.body.fields[id as usize];
            let fname = self.src.text(field_expr.name);
            let fdef = var
                .fields
                .iter()
                .find(|f| f.name.is_some_and(|n| adt_src.text(n) == fname))?;
            let mut f_lts = Lifetimes::default();
            collect_ty_lifetimes(&def.body, adt_src, fdef.ty, &mut f_lts);
            if f_lts.names.iter().any(|n| def_ty_params.contains(n)) {
                return None;
            }
            if !f_lts.named.iter().any(|n| shared_def_lts.contains(n)) {
                continue;
            }
            if !is_inline {
                let mut used = HashSet::default();
                self.collect_live_expr(field_expr.value, &mut used);
                for local in used {
                    if self.mutable.get(local as usize).copied().unwrap_or(true)
                        || !is_freeze_ty(self.cx, self.local_ty(local), 0)
                    {
                        return None;
                    }
                }
            }
            let outer_non_shared_ref = matches!(
                *def.body.ty(fdef.ty),
                AstTy::Ref {
                    lifetime: Some(lt),
                    ..
                } if !shared_def_lts.contains(&adt_src.text(lt))
            );
            if outer_non_shared_ref {
                out.extend(self.copied_arg_borrow_sources(call_expr, field_expr.value, state));
            } else {
                out.extend(self.expr_borrow_sources(field_expr.value, state));
            }
        }
        Some(out)
    }

    fn unwrap_collection_arg_sources(
        &mut self,
        arg: ExprId,
        arg_ty: Ty,
        recv_ty: Ty,
        sources: Vec<BorrowSource>,
        state: &State,
    ) -> Vec<BorrowSource> {
        if arg_ty == Ty::NONE || recv_ty == Ty::NONE || sources.is_empty() {
            return sources;
        }
        let Kind::Ref(_, inner) = *self.cx.types.kind(arg_ty) else {
            return sources;
        };
        let inner_elems: Vec<Ty> = match *self.cx.types.kind(inner) {
            Kind::Slice(e) | Kind::Array(e, _) => vec![e],
            Kind::Adt(_, args) => self.cx.types.substs(args).to_vec(),
            _ => return sources,
        };
        let Kind::Adt(recv_item, recv_args) = *self.cx.types.kind(recv_ty) else {
            return sources;
        };
        let recv_substs = self.cx.types.substs(recv_args).to_vec();
        let coerced_target = match self.typed.coercions.get(&arg).map(Coercion::leaf) {
            Some(Coercion::Deref(_, to)) => Some(*to),
            Some(Coercion::Vtable(v)) => Some(v.object),
            Some(Coercion::Len(_)) => match *self.cx.types.kind(inner) {
                Kind::Array(elem, _) => Some(self.cx.types.intern(Kind::Slice(elem))),
                _ => None,
            },
            _ => None,
        };
        if inner_elems.is_empty()
            || recv_substs.is_empty()
            || self.cx.adt_def(recv_item).is_ok_and(|def| {
                def.generics
                    .params
                    .iter()
                    .any(|g| g.kind == GenericKind::Lifetime)
            })
            || recv_substs
                .iter()
                .any(|&t| self.ty_contains_ref_to(t, inner, coerced_target, &inner_elems))
        {
            return sources;
        }
        let mut unwrapped = arg;
        while let Expr::Paren(e) = *self.body.expr(unwrapped) {
            unwrapped = e;
        }
        let container_local = match *self.body.expr(unwrapped) {
            Expr::Ref { expr: inner_e, .. } => self.root_place(inner_e).map(|p| p.local),
            _ => self.root_place(unwrapped).and_then(|p| {
                state
                    .loans
                    .iter()
                    .find(|l| l.holder == Some(p.local) && !self.is_ref_ty(self.local_ty(l.target)))
                    .map(|l| l.target)
            }),
        };
        unwrap_container_loans(sources, state, |s| {
            is_temp_target(s.target).is_some() || container_local == Some(s.target)
        })
    }

    fn ty_contains_ref_to(&self, ty: Ty, container: Ty, coerced: Option<Ty>, elems: &[Ty]) -> bool {
        match *self.cx.types.kind(ty) {
            Kind::Ref(_, target) => {
                target == container || coerced == Some(target) || elems.contains(&target)
            }
            Kind::Tuple(items) | Kind::Adt(_, items) => self
                .cx
                .types
                .substs(items)
                .iter()
                .any(|&t| self.ty_contains_ref_to(t, container, coerced, elems)),
            Kind::Slice(t) | Kind::Array(t, _) | Kind::Pat(t, ..) => {
                self.ty_contains_ref_to(t, container, coerced, elems)
            }
            _ => false,
        }
    }

    fn method_self_ty(&self, expr: ExprId, receiver: ExprId, sig: Option<&Signature>) -> Ty {
        let mut cur = self.expr_ty(receiver);
        if cur == Ty::NONE {
            return Ty::NONE;
        }
        let adjust = self.typed.adjusts.get(&expr).copied();
        let derefs = adjust.map_or(0, |a| a.derefs);
        for step in 0..derefs {
            if self.typed.derefs.contains_key(&(expr, step)) {
                return Ty::NONE;
            }
            cur = self.builtin_deref_ty(cur).unwrap_or(Ty::NONE);
            if cur == Ty::NONE {
                return Ty::NONE;
            }
        }
        if adjust.and_then(|a| a.autoref).is_none()
            && sig.is_some_and(recv_is_by_ref)
            && self.is_ref_ty(cur)
        {
            cur = self.builtin_deref_ty(cur).unwrap_or(Ty::NONE);
        }
        cur
    }

    pub(in super::super) fn generic_param_ties_to_self(
        &self,
        sig: &Signature,
        src: &Source,
        p_lts: &Lifetimes<'_>,
        self_lts: &Lifetimes<'_>,
    ) -> bool {
        let bounds_tie =
            |bounds: crate::body::List| {
                let mut b_lts = Lifetimes::default();
                for &b_id in sig.body.list(bounds) {
                    if let Bound::Lifetime(lt) = sig.body.bounds[b_id as usize] {
                        let lt_name = src.text(lt);
                        if self_lts.named.iter().any(|&s_lt| {
                            lt_name == s_lt || lifetime_outlives(sig, src, lt_name, s_lt)
                        }) {
                            return true;
                        }
                    } else if let Bound::Trait { path, .. } = sig.body.bounds[b_id as usize] {
                        for &seg in sig.body.list(sig.body.path(path).segments) {
                            if let Some(Args::Angle(args)) = sig.body.segments[seg as usize].args {
                                for &a in sig.body.list(args) {
                                    match sig.body.args[a as usize] {
                                        GenericArg::Lifetime(lt) => {
                                            let lt_name = src.text(lt);
                                            if self_lts.named.iter().any(|&s_lt| {
                                                lt_name == s_lt
                                                    || lifetime_outlives(sig, src, lt_name, s_lt)
                                            }) {
                                                return true;
                                            }
                                        }
                                        GenericArg::Ty(ty) | GenericArg::Binding { ty, .. } => {
                                            collect_ty_lifetimes(&sig.body, src, ty, &mut b_lts);
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }
                    }
                }
                b_lts.names.iter().any(|n| self_lts.names.contains(n))
                    || b_lts.named.iter().any(|n| {
                        self_lts
                            .named
                            .iter()
                            .any(|&s_lt| *n == s_lt || lifetime_outlives(sig, src, n, s_lt))
                    })
            };
        for g in &sig.generics {
            let gname = src.text(g.name);
            if !p_lts.names.contains(&gname) && !p_lts.named.contains(&gname) {
                continue;
            }
            if bounds_tie(g.bounds) {
                return true;
            }
            for pred in &sig.predicates {
                if pred_matches_name(sig, src, pred, gname) && bounds_tie(pred.bounds) {
                    return true;
                }
            }
        }
        false
    }

    pub(in super::super) fn attach_call_out_sources(
        &mut self,
        expr: ExprId,
        arg_ids: &[u32],
        sig: Option<&Signature>,
        fn_item: Option<ItemRef>,
        has_self_arg: bool,
        state: &mut State,
    ) -> Result<()> {
        let Some(sig) = sig else { return Ok(()) };
        if has_self_arg && !arg_ids.is_empty() {
            let recv_mut = recv_ref_mut(sig) == Some(Mut::Mut);
            let recv_expr = match *self.body.expr(ExprId(arg_ids[0])) {
                Expr::Ref {
                    mutable: true,
                    raw: false,
                    expr: inner,
                } => inner,
                _ => ExprId(arg_ids[0]),
            };
            self.attach_method_arg_sources(
                expr,
                recv_expr,
                &arg_ids[1..],
                recv_mut,
                Some(sig),
                state,
            )?;
        }
        let param_args = if has_self_arg {
            arg_ids.get(1..).unwrap_or(&[])
        } else {
            arg_ids
        };
        if param_args.len() < 2 || sig.params.len() < 2 {
            return Ok(());
        }
        let fn_src = fn_item.map_or(self.src, |item| self.cx.item(item).0);
        for (dst_idx, &dst_id) in param_args.iter().enumerate() {
            let Some(&(_, dst_ast)) = sig.params.get(dst_idx) else {
                continue;
            };
            let AstTy::Ref {
                mutable: true,
                ty: dst_inner,
                ..
            } = *sig.body.ty(dst_ast)
            else {
                continue;
            };
            let dst_expr = match *self.body.expr(ExprId(dst_id)) {
                Expr::Ref {
                    mutable: true,
                    raw: false,
                    expr,
                } => expr,
                _ => continue,
            };
            let Some(p) = self.root_place(dst_expr) else {
                continue;
            };
            let dst_ty = self.expr_ty(dst_expr);
            if !self.ty_holds_borrow(if dst_ty != Ty::NONE {
                dst_ty
            } else {
                self.local_ty(p.local)
            }) {
                continue;
            }
            for (src_idx, &src_id) in param_args.iter().enumerate() {
                if src_idx == dst_idx {
                    continue;
                }
                let Some(&(_, src_ast)) = sig.params.get(src_idx) else {
                    continue;
                };
                if self.sig_params_share_lt_or_ty(sig, fn_src, dst_inner, src_ast) {
                    let src_expr = match (*sig.body.ty(src_ast), *self.body.expr(ExprId(src_id))) {
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
                        ) if !matches!(*sig.body.ty(dst_inner), AstTy::Ref { .. })
                            && self
                                .sig_params_share_lt_or_ty(sig, fn_src, dst_inner, src_inner) =>
                        {
                            expr
                        }
                        _ => ExprId(src_id),
                    };
                    let sources = self.expr_borrow_sources(src_expr, state);
                    self.propagate_assign_sources(&p, &sources, state)?;
                }
            }
        }
        Ok(())
    }

    fn sig_params_share_lt_or_ty(&self, sig: &Signature, src: &Source, a: TyId, b: TyId) -> bool {
        let mut lts_a = Lifetimes::default();
        let mut lts_b = Lifetimes::default();
        collect_ty_lifetimes(&sig.body, src, a, &mut lts_a);
        collect_ty_lifetimes(&sig.body, src, b, &mut lts_b);
        if lts_a.named.iter().any(|n| lts_b.named.contains(n)) {
            return true;
        }
        sig.generics.iter().any(|g| {
            g.kind == GenericKind::Type && {
                let name = src.text(g.name);
                lts_a.names.contains(&name) && lts_b.names.contains(&name)
            }
        })
    }
}

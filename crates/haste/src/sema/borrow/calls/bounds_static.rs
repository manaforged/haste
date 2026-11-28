use anyhow::Result;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::body::sig::{GenericKind, Signature};
use crate::body::{Args, ExprId, Ty as AstTy, TyId};
use crate::names::ItemRef;
use crate::sema::infer::Res;
use crate::sema::ty::{Kind, Ty};

use super::bounds::callable_param_inputs;
use super::lifetimes::{Lifetimes, collect_ty_lifetimes};

impl super::super::Checker<'_, '_> {
    pub(in super::super) fn check_call_static_type_args(
        &mut self,
        call_expr: ExprId,
        call_res: Option<Res>,
        sig: Option<&Signature>,
        fn_item: Option<ItemRef>,
    ) -> Result<()> {
        let (Some(sig), Some(fn_item), Some(Res::Fn(res_item, substs))) = (sig, fn_item, call_res)
        else {
            return Ok(());
        };
        if res_item != fn_item {
            return Ok(());
        }
        let subst_tys: Vec<Ty> = self.cx.types.substs(substs).to_vec();
        if !subst_tys.iter().any(|&t| self.ty_has_non_static_param(t)) {
            return Ok(());
        }
        let sig_src = self.cx.item(fn_item).0;
        let (static_indices, _) =
            super::super::checks::compute_static_type_params(self.cx, fn_item, sig, sig_src);
        for idx in static_indices {
            if subst_tys
                .get(idx as usize)
                .is_some_and(|&t| self.ty_has_non_static_param(t))
            {
                return Err(self.borrow_err(
                    Some(call_expr),
                    "the parameter type may not live long enough: `'static` required",
                ));
            }
        }
        Ok(())
    }

    pub(in super::super) fn check_coercion_vtables(&mut self) -> Result<()> {
        let vtables: Vec<_> = self
            .typed
            .coercions
            .iter()
            .filter_map(|(&e, c)| c.vtable().map(|v| (e, v)))
            .collect();
        let mut cache = HashMap::default();
        for (coercion_expr, vtable) in vtables {
            let Kind::Dyn(of, ..) = *self.cx.types.kind(vtable.object) else {
                continue;
            };
            let static_trait = super::super::checks::trait_has_static_supertrait(
                self.cx,
                of,
                &mut HashSet::default(),
                &mut cache,
            );
            if static_trait && self.ty_has_non_static_param(vtable.ty) {
                return Err(self.borrow_err(
                    Some(coercion_expr),
                    "the parameter type may not live long enough: `'static` required for trait object",
                ));
            }
            if let Kind::Closure(parent, _, at) = *self.cx.types.kind(vtable.ty)
                && parent == self.item
                && let Some(eff) = self.closure_effects.get(&ExprId(at))
            {
                let lang = self.cx.lang(of);
                self.check_closure_effects(
                    Some(coercion_expr),
                    eff,
                    lang == Some("fn"),
                    lang == Some("fn_mut"),
                    static_trait,
                )?;
            }
        }
        Ok(())
    }

    pub(in super::super) fn check_callee_bound_contra(
        &mut self,
        callee: ExprId,
        arg_ids: &[u32],
    ) -> Result<()> {
        let Some((sig, param_ty)) = self.local_param_sig_ty(callee) else {
            return Ok(());
        };
        let inputs = callable_param_inputs(&sig, self.src, param_ty);
        for (in_ty, &arg_id) in inputs.into_iter().zip(arg_ids) {
            let arg = ExprId(arg_id);
            let var = super::super::checks::collect_ast_variance(
                self.cx,
                Some(self.item),
                &sig.body,
                self.src,
                in_ty,
                Some(self.expr_ty(arg)),
            );
            if !var.contra.is_empty() {
                self.check_expr_contra_fit(arg, &var.contra)?;
            }
        }
        Ok(())
    }
    pub(in super::super) fn fn_item_returns_hrtb_param(
        &mut self,
        arg: ExprId,
        hrtb_non_ret: &[usize],
    ) -> bool {
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
        if super::lifetimes::sig_returns_only_static(&sig, sig_src, ret_ast, &ret_lts) {
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
                    && super::lifetimes::ty_shares_name(&sig.body, sig_src, pt, &ret_lts.names)
            })
        })
    }

    pub(in super::super) fn check_impl_lifetime_arg(
        &mut self,
        arg: ExprId,
        sources: &[super::super::propagate::BorrowSource],
        sig: &Signature,
        fn_item: Option<ItemRef>,
        param_ty: TyId,
    ) -> Result<()> {
        if self.closure_depth == 0 || sig.receiver.is_none() {
            return Ok(());
        }
        let Some(fn_item) = fn_item else {
            return Ok(());
        };
        let Some(impl_ref) = self.cx.owner_impl(fn_item) else {
            return Ok(());
        };
        let sig_src = self.cx.item(fn_item).0;
        let AstTy::Ref {
            lifetime: Some(lt), ..
        } = *sig.body.ty(param_ty)
        else {
            return Ok(());
        };
        let lt_name = sig_src.text(lt);
        if lt_name == "'static"
            || sig
                .generics
                .iter()
                .any(|g| g.kind == GenericKind::Lifetime && sig_src.text(g.name) == lt_name)
        {
            return Ok(());
        }
        let Ok(hdr) = self.cx.impl_header(impl_ref) else {
            return Ok(());
        };
        let impl_src = &self.cx.impl_file(impl_ref).source;
        let in_impl = hdr
            .generics
            .iter()
            .any(|g| g.kind == GenericKind::Lifetime && impl_src.text(g.name) == lt_name);
        if !in_impl {
            return Ok(());
        }
        for s in sources {
            if self.is_closure_param(s.target) && (s.through_ref || s.origin_only) {
                continue;
            }
            if super::super::checks::is_temp_target(s.target).is_some()
                || (!self.from_caller(s.target, s.through_ref)
                    && self
                        .scope_depth
                        .get(s.target as usize)
                        .copied()
                        .unwrap_or(0)
                        >= self.closure_depth)
            {
                return Err(self.does_not_live_long_enough(Some(arg), s.target));
            }
        }
        if let Some(local) = self.expr_plain_local(arg)
            && self.scope_depth.get(local as usize).copied().unwrap_or(0) >= self.closure_depth
            && self.is_ref_ty(self.local_ty(local))
            && sources.is_empty()
            && self
                .typed
                .ref_binds
                .iter()
                .any(|(pat, _)| self.typed.binds.get(pat) == Some(&local))
        {
            return Err(self.does_not_live_long_enough(Some(arg), local));
        }
        Ok(())
    }

    pub(in super::super) fn fn_ptr_field_named_mut_lt(
        &mut self,
        callee_expr: Option<ExprId>,
        idx: usize,
        local: Option<u32>,
    ) -> bool {
        let (Some(callee), Some(local)) = (callee_expr, local) else {
            return false;
        };
        if self
            .param_lts
            .named_lts
            .get(local as usize)
            .is_none_or(Vec::is_empty)
        {
            return false;
        }
        let crate::body::Expr::Field(base, field_name) =
            *self.body.expr(self.unwrap_parens(callee))
        else {
            return false;
        };
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
        let Some(var) = def.variants.first() else {
            return false;
        };
        let adt_src = &self.cx.file(adt_item).source;
        let fname = crate::names::plain(self.src.text(field_name));
        let Some(field) = var.fields.iter().find(|f| {
            f.name
                .is_some_and(|n| crate::names::plain(adt_src.text(n)) == fname)
        }) else {
            return false;
        };
        let AstTy::Fn { params, .. } = *def.body.ty(field.ty) else {
            return false;
        };
        let Some(&p_id) = def.body.list(params).get(idx) else {
            return false;
        };
        matches!(
            *def.body.ty(TyId(p_id)),
            AstTy::Ref {
                mutable: true,
                lifetime: Some(_),
                ..
            }
        )
    }

    pub(in super::super) fn check_method_impl_static_self(
        &mut self,
        receiver: ExprId,
        call_res: Option<Res>,
        fn_item: Option<ItemRef>,
        state: &super::super::State,
    ) -> Result<()> {
        let (Some(fn_item), Some(Res::Fn(_, substs))) = (fn_item, call_res) else {
            return Ok(());
        };
        let Some(tr) = self.cx.owner_trait(fn_item) else {
            return Ok(());
        };
        let recv_ty = self.expr_ty(receiver);
        if recv_ty == Ty::NONE
            || self.cx.types.has_params(recv_ty)
            || !self.ty_holds_borrow(recv_ty)
        {
            return Ok(());
        }
        let Ok(tg) = self.cx.trait_generics(tr) else {
            return Ok(());
        };
        let extra_params =
            tg.1.params
                .iter()
                .filter(|g| g.kind == GenericKind::Type)
                .count();
        let all_substs = self.cx.types.substs(substs).to_vec();
        if extra_params == 0 || all_substs.len() < 1 + extra_params {
            return Ok(());
        }
        let trait_args = &all_substs[1..1 + extra_params];
        let Some(impl_ref) = crate::sema::infer::detached(self.cx, self.item, |infer| {
            Ok(infer.select(tr, recv_ty, trait_args)?.map(|(blk, _)| blk))
        })
        .ok()
        .flatten() else {
            return Ok(());
        };
        let Ok(hdr) = self.cx.impl_header(impl_ref) else {
            return Ok(());
        };
        let Some(AstTy::Path(tr_path)) = hdr.trait_ref.map(|t| *hdr.body.ty(t)) else {
            return Ok(());
        };
        let impl_src = &self.cx.impl_file(impl_ref).source;
        let AstTy::Path(self_path) = *hdr.body.ty(hdr.self_ty) else {
            return Ok(());
        };
        let self_segs = hdr.body.list(hdr.body.path(self_path).segments);
        if self_segs.len() != 1 {
            return Ok(());
        }
        let self_name = impl_src.text(hdr.body.segments[self_segs[0] as usize].name);
        if !hdr
            .generics
            .iter()
            .any(|g| g.kind == GenericKind::Type && impl_src.text(g.name) == self_name)
        {
            return Ok(());
        }
        let tr_segs = hdr.body.list(hdr.body.path(tr_path).segments);
        let Some(&last_seg) = tr_segs.last() else {
            return Ok(());
        };
        let Some(Args::Angle(tr_args)) = hdr.body.segments[last_seg as usize].args else {
            return Ok(());
        };
        let implies_static = hdr.body.list(tr_args).iter().any(|&a| {
            let crate::body::GenericArg::Ty(arg_ty) = hdr.body.args[a as usize] else {
                return false;
            };
            let AstTy::Ref {
                lifetime: Some(lt),
                ty: inner,
                ..
            } = *hdr.body.ty(arg_ty)
            else {
                return false;
            };
            impl_src.text(lt) == "'static"
                && matches!(
                    *hdr.body.ty(inner),
                    AstTy::Path(p) if {
                        let ss = hdr.body.list(hdr.body.path(p).segments);
                        ss.len() == 1 && impl_src.text(hdr.body.segments[ss[0] as usize].name) == self_name
                    }
                )
        });
        if implies_static {
            self.check_static_arg(receiver, state)?;
        }
        Ok(())
    }
}

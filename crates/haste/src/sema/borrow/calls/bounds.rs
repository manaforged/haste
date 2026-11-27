use crate::body::sig::{GenericKind, SelfKind, Signature};
use crate::body::{Args, Body, Bound, Ty as AstTy, TyId};
use crate::outline::source::Source;
use crate::sema::ty::{Kind, Ty};

use super::lifetimes::{
    Lifetimes, collect_ty_lifetimes, lifetime_outlives_in_headers, pred_matches_name,
};

#[derive(Clone, Copy)]
pub(in super::super) enum ShareTarget {
    Local(u32, Ty),
    Ret(Ty),
}

#[derive(Clone, Copy)]
enum ParamAst {
    Sig(TyId),
    SelfTy,
}

pub(in super::super) fn lifetime_outlives_in_item(
    cx: &mut crate::sema::Context<'_>,
    item: crate::names::ItemRef,
    sig: &Signature,
    src: &Source,
    a: &str,
    b: &str,
) -> bool {
    let impl_hdr = cx
        .owner_impl(item)
        .and_then(|block| cx.impl_header(block).ok().map(|hdr| (block, hdr)));
    match impl_hdr {
        Some((block, ref hdr)) => {
            let impl_src = &cx.impl_file(block).source;
            lifetime_outlives_in_headers(
                &[
                    (&sig.body, src, &sig.generics, &sig.predicates),
                    (&hdr.body, impl_src, &hdr.generics, &hdr.predicates),
                ],
                a,
                b,
            )
        }
        None => {
            lifetime_outlives_in_headers(&[(&sig.body, src, &sig.generics, &sig.predicates)], a, b)
        }
    }
}

impl super::super::Checker<'_, '_> {
    fn collect_non_static_params(&self, ty: Ty, out: &mut Vec<u16>, allow_proj: bool) -> bool {
        if ty == Ty::NONE || (!allow_proj && self.cx.types.has_projs(ty)) {
            return false;
        }
        if !self.cx.types.has_params(ty) {
            return true;
        }
        match *self.cx.types.kind(ty) {
            Kind::Param(idx) => {
                if !self.static_type_params.contains(&idx) && !out.contains(&idx) {
                    out.push(idx);
                }
                true
            }
            Kind::Ref(_, inner)
            | Kind::Ptr(_, inner)
            | Kind::Slice(inner)
            | Kind::Array(inner, _)
            | Kind::Pat(inner, ..) => self.collect_non_static_params(inner, out, allow_proj),
            Kind::Adt(_, args) | Kind::Tuple(args) | Kind::FnDef(_, args) => self
                .cx
                .types
                .substs(args)
                .to_vec()
                .into_iter()
                .all(|a| self.collect_non_static_params(a, out, allow_proj)),
            Kind::FnPtr(params, ret) => {
                let ps = self.cx.types.substs(params).to_vec();
                ps.into_iter()
                    .all(|p| self.collect_non_static_params(p, out, allow_proj))
                    && self.collect_non_static_params(ret, out, allow_proj)
            }
            _ => allow_proj,
        }
    }

    pub(in super::super) fn src_non_static_params_shared_with(
        &mut self,
        local: u32,
        peel: bool,
        target: ShareTarget,
    ) -> bool {
        let target_ty = match target {
            ShareTarget::Local(_, ty) | ShareTarget::Ret(ty) => ty,
        };
        let mut src_ty = self.local_ty(local);
        if peel {
            let Kind::Ref(_, inner) = *self.cx.types.kind(src_ty) else {
                return false;
            };
            src_ty = inner;
        }
        let mut src_params = Vec::new();
        let mut dst_params = Vec::new();
        if !self.collect_non_static_params(src_ty, &mut src_params, false)
            || !self.collect_non_static_params(target_ty, &mut dst_params, true)
        {
            return false;
        }
        if !src_params.is_empty() {
            return src_params.iter().all(|p| dst_params.contains(p));
        }
        if self.cx.types.has_params(src_ty) {
            return false;
        }
        self.ast_unbounded_ty_params_shared_with(local, peel, target)
    }

    fn collect_non_static_ty_param_names(&mut self, sig: &Signature) -> Vec<String> {
        let src = self.src;
        let mut out = Vec::new();
        let mut idx = 0u16;
        if let Some(impl_ref) = self.cx.owner_impl(self.item) {
            if let Ok(header) = self.cx.impl_header(impl_ref) {
                let impl_src = &self.cx.impl_file(impl_ref).source;
                for g in crate::sema::infer::scope::type_params(&header.generics) {
                    if !self.static_type_params.contains(&idx) {
                        out.push(impl_src.text(g.name).to_owned());
                    }
                    idx += 1;
                }
            }
        } else if let Some(trait_ref) = self.cx.owner_trait(self.item) {
            if !self.static_type_params.contains(&idx) {
                out.push("Self".to_owned());
            }
            idx += 1;
            if let Ok(tg) = self.cx.trait_generics(trait_ref) {
                let tsrc = &self.cx.file(trait_ref).source;
                for g in crate::sema::infer::scope::type_params(&tg.1.params) {
                    if !self.static_type_params.contains(&idx) {
                        out.push(tsrc.text(g.name).to_owned());
                    }
                    idx += 1;
                }
            }
        }
        for g in crate::sema::infer::scope::type_params(&sig.generics) {
            if !self.static_type_params.contains(&idx) {
                out.push(src.text(g.name).to_owned());
            }
            idx += 1;
        }
        out
    }

    fn param_ast_ty(&self, sig: &Signature, local: u32, peel: bool) -> Option<ParamAst> {
        let mut cursor = 0usize;
        if let Some((kind, _)) = sig.receiver {
            if self.typed.params.first() == Some(&local) {
                return match kind {
                    SelfKind::Typed(ty) => Self::peel_sig_ast_ty(sig, ty, peel),
                    SelfKind::Ref | SelfKind::RefMut if peel => Some(ParamAst::SelfTy),
                    SelfKind::Value if !peel => Some(ParamAst::SelfTy),
                    _ => None,
                };
            }
            cursor = 1;
        }
        for (idx, &(_, param_ty)) in sig.params.iter().enumerate() {
            if self.typed.params.get(cursor + idx) == Some(&local) {
                return Self::peel_sig_ast_ty(sig, param_ty, peel);
            }
        }
        None
    }

    fn peel_sig_ast_ty(sig: &Signature, ty: TyId, peel: bool) -> Option<ParamAst> {
        if !peel {
            return Some(ParamAst::Sig(ty));
        }
        match *sig.body.ty(ty) {
            AstTy::Ref { ty: inner, .. } => Some(ParamAst::Sig(inner)),
            _ => None,
        }
    }

    fn param_ast_unbounded(
        &mut self,
        sig: &Signature,
        ast: ParamAst,
        non_static: &[String],
    ) -> Vec<String> {
        let mut out = Vec::new();
        match ast {
            ParamAst::Sig(ty_id) => {
                let src = self.src;
                collect_ast_unbounded_params(
                    self.cx, self.item, &sig.body, src, ty_id, non_static, &mut out,
                );
            }
            ParamAst::SelfTy => {
                if let Some(impl_ref) = self.cx.owner_impl(self.item)
                    && let Ok(hdr) = self.cx.impl_header(impl_ref)
                {
                    let impl_src = &self.cx.impl_file(impl_ref).source;
                    push_matching_params(&hdr.body, impl_src, hdr.self_ty, non_static, &mut out);
                } else if non_static.iter().any(|s| s == "Self") {
                    out.push("Self".to_owned());
                }
            }
        }
        out
    }

    fn ast_unbounded_ty_params_shared_with(
        &mut self,
        local: u32,
        peel: bool,
        target: ShareTarget,
    ) -> bool {
        let Ok(sig) = self.cx.signature(self.item) else {
            return false;
        };
        let non_static = self.collect_non_static_ty_param_names(&sig);
        let Some(src_ast) = self.param_ast_ty(&sig, local, peel) else {
            return false;
        };
        let src_names = self.param_ast_unbounded(&sig, src_ast, &non_static);
        if src_names.is_empty() {
            return false;
        }
        let dst_names = match target {
            ShareTarget::Ret(_) => {
                let Some(ret_ast) = sig.ret else {
                    return false;
                };
                self.param_ast_unbounded(&sig, ParamAst::Sig(ret_ast), &non_static)
            }
            ShareTarget::Local(dst_local, _) => {
                if let Some(dst_ast) = self.param_ast_ty(&sig, dst_local, false) {
                    self.param_ast_unbounded(&sig, dst_ast, &non_static)
                } else if let Some(&ann_ty) = self.variance.local_ann_ast_ty.get(&dst_local) {
                    let (body, src) = (self.body, self.src);
                    let mut out = Vec::new();
                    collect_ast_unbounded_params(
                        self.cx,
                        self.item,
                        body,
                        src,
                        ann_ty,
                        &non_static,
                        &mut out,
                    );
                    out
                } else {
                    return false;
                }
            }
        };
        src_names.iter().all(|n| dst_names.contains(n))
    }
}

fn push_matching_params(
    body: &Body,
    src: &Source,
    ty_id: TyId,
    non_static: &[String],
    out: &mut Vec<String>,
) {
    let mut lts = Lifetimes::default();
    collect_ty_lifetimes(body, src, ty_id, &mut lts);
    for &name in &lts.names {
        if non_static.iter().any(|s| s == name) && !out.iter().any(|s| s == name) {
            out.push(name.to_owned());
        }
    }
}

fn collect_ast_unbounded_params(
    cx: &mut crate::sema::Context<'_>,
    item: crate::names::ItemRef,
    body: &Body,
    src: &Source,
    ty_id: TyId,
    non_static: &[String],
    out: &mut Vec<String>,
) {
    let mut lts = Lifetimes::default();
    collect_ty_lifetimes(body, src, ty_id, &mut lts);
    for &name in &lts.names {
        if name == "Self"
            && let Some(impl_ref) = cx.owner_impl(item)
            && let Ok(hdr) = cx.impl_header(impl_ref)
        {
            let impl_src = &cx.impl_file(impl_ref).source;
            push_matching_params(&hdr.body, impl_src, hdr.self_ty, non_static, out);
        }
        if non_static.iter().any(|s| s == name) && !out.iter().any(|s| s == name) {
            out.push(name.to_owned());
        }
    }
}

pub(in super::super) fn for_each_param_bound_list(
    sig: &Signature,
    src: &Source,
    param_ty: TyId,
    mut visit: impl FnMut(crate::body::List),
) {
    match *sig.body.ty(param_ty) {
        AstTy::ImplTrait(_, bounds) | AstTy::DynTrait(bounds) => visit(bounds),
        AstTy::Ref { ty: inner, .. } => {
            if let AstTy::ImplTrait(_, bounds) | AstTy::DynTrait(bounds) = *sig.body.ty(inner) {
                visit(bounds);
            }
        }
        AstTy::Path(path) => {
            let p = sig.body.path(path);
            let segs = sig.body.list(p.segments);
            if p.qself.is_none() && segs.len() == 1 {
                let name = src.text(sig.body.segments[segs[0] as usize].name);
                for g in sig
                    .generics
                    .iter()
                    .filter(|g| g.kind == GenericKind::Type && src.text(g.name) == name)
                {
                    visit(g.bounds);
                }
                for pred in sig
                    .predicates
                    .iter()
                    .filter(|pred| pred_matches_name(sig, src, pred, name))
                {
                    visit(pred.bounds);
                }
            }
        }
        _ => {}
    }
}

pub(in super::super) fn callable_param_inputs(
    sig: &Signature,
    src: &Source,
    param_ty: TyId,
) -> Vec<TyId> {
    let mut inputs_out = Vec::new();
    for_each_param_bound_list(sig, src, param_ty, |list| {
        for &b_id in sig.body.list(list) {
            if let Bound::Trait { path, .. } = sig.body.bounds[b_id as usize]
                && let Some(&seg_id) = sig.body.list(sig.body.path(path).segments).last()
                && let seg = sig.body.segments[seg_id as usize]
                && matches!(src.text(seg.name), "Fn" | "FnMut" | "FnOnce")
                && let Some(Args::Paren { inputs, .. }) = seg.args
            {
                inputs_out.extend(sig.body.list(inputs).iter().copied().map(TyId));
            }
        }
    });
    inputs_out
}

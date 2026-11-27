use std::sync::Arc;

use crate::body::sig::{GenericKind, Signature};
use crate::body::{Args, Body, Bound, Expr, ExprId, GenericArg, PatId, Ty as AstTy, TyId};
use crate::names::{ItemRef, plain};
use crate::outline::item::ItemKind;
use crate::outline::source::Source;
use crate::sema::Alias;
use crate::sema::infer::Res;

use super::super::Checker;
use super::super::checks::resolve_ast_type_path;
use super::arg_refs::sig_param_idx;
use super::lifetimes::{
    LT_STATIC, for_each_param_bound_list, is_owned_container, is_prim_name, lifetime_is_static,
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum DynDefault {
    Inferred,
    InSig,
    InSigOwned,
}

impl DynDefault {
    fn enter_owned(self) -> Self {
        match self {
            Self::Inferred => Self::Inferred,
            Self::InSig | Self::InSigOwned => Self::InSigOwned,
        }
    }

    fn reset_owned(self) -> Self {
        match self {
            Self::Inferred => Self::Inferred,
            Self::InSig | Self::InSigOwned => Self::InSig,
        }
    }
}

pub(super) fn ast_ty_is_provably_static(
    sig: &crate::body::sig::Signature,
    src: &Source,
    ty: TyId,
) -> bool {
    ast_ty_is_provably_static_in_mode(&sig.body, src, Some((sig, src)), ty, DynDefault::InSig)
}

pub(super) fn ast_ty_contains_dyn(body: &Body, ty: TyId) -> bool {
    match *body.ty(ty) {
        AstTy::DynTrait(_) => true,
        AstTy::Paren(t)
        | AstTy::Slice(t)
        | AstTy::Array(t, _)
        | AstTy::Pat(t, _)
        | AstTy::Ptr { ty: t, .. }
        | AstTy::Ref { ty: t, .. } => ast_ty_contains_dyn(body, t),
        AstTy::Tuple(list) => body
            .list(list)
            .iter()
            .any(|&t| ast_ty_contains_dyn(body, TyId(t))),
        AstTy::Path(path_id) => {
            body.list(body.path(path_id).segments).iter().any(|&s_id| {
                match body.segments[s_id as usize].args {
                    Some(Args::Angle(args)) => body.list(args).iter().any(|&a| {
                        matches!(
                            body.args[a as usize],
                            GenericArg::Ty(t) | GenericArg::Binding { ty: t, .. }
                                if ast_ty_contains_dyn(body, t)
                        )
                    }),
                    _ => false,
                }
            })
        }
        _ => false,
    }
}

pub(in super::super) fn ast_ty_is_provably_static_in(
    body: &Body,
    src: &Source,
    sig: Option<(&crate::body::sig::Signature, &Source)>,
    ty: TyId,
) -> bool {
    ast_ty_is_provably_static_in_mode(body, src, sig, ty, DynDefault::Inferred)
}

pub(in super::super) fn ast_ref_dyn_requires_static(
    body: &Body,
    src: &Source,
    sig: Option<(&crate::body::sig::Signature, &Source)>,
    mut ty: TyId,
) -> bool {
    while let AstTy::Paren(inner) = *body.ty(ty) {
        ty = inner;
    }
    let AstTy::Ref {
        ty: mut pointee, ..
    } = *body.ty(ty)
    else {
        return false;
    };
    while let AstTy::Paren(inner) = *body.ty(pointee) {
        pointee = inner;
    }
    let AstTy::DynTrait(list) = *body.ty(pointee) else {
        return false;
    };
    body.list(list).iter().any(|&b_id| {
        let Bound::Lifetime(lt) = body.bounds[b_id as usize] else {
            return false;
        };
        let text = src.text(lt);
        match sig {
            Some((s, s_src)) => lifetime_is_static(s, s_src, text),
            None => text == LT_STATIC,
        }
    })
}

fn path_args_are_static(
    body: &Body,
    src: &Source,
    sig: Option<(&crate::body::sig::Signature, &Source)>,
    path: crate::body::PathId,
    lt_static: &impl Fn(&str) -> bool,
    mode: DynDefault,
) -> bool {
    let base = mode.reset_owned();
    body.list(body.path(path).segments).iter().all(|&s_id| {
        match body.segments[s_id as usize].args {
            None => true,
            Some(Args::Angle(args)) => {
                body.list(args)
                    .iter()
                    .all(|&a| match body.args[a as usize] {
                        GenericArg::Ty(t) | GenericArg::Binding { ty: t, .. } => {
                            ast_ty_is_provably_static_in_mode(body, src, sig, t, base)
                        }
                        GenericArg::Lifetime(lt) => lt_static(src.text(lt)),
                        GenericArg::Const(_) => true,
                        GenericArg::Constraint { .. } => false,
                    })
            }
            Some(Args::Paren { inputs, output }) => {
                body.list(inputs)
                    .iter()
                    .all(|&t| fn_param_is_provably_static(body, src, sig, TyId(t), lt_static, base))
                    && output
                        .is_none_or(|o| ast_ty_is_provably_static_in_mode(body, src, sig, o, base))
            }
            Some(Args::ReturnType) => false,
        }
    })
}

fn fn_param_is_provably_static(
    body: &Body,
    src: &Source,
    sig: Option<(&crate::body::sig::Signature, &Source)>,
    ty: TyId,
    lt_static: &impl Fn(&str) -> bool,
    mode: DynDefault,
) -> bool {
    match *body.ty(ty) {
        AstTy::Ref {
            lifetime: None,
            ty: inner,
            ..
        } => ast_ty_is_provably_static_in_mode(body, src, sig, inner, mode.reset_owned()),
        AstTy::Ref {
            lifetime: Some(lt),
            ty: inner,
            ..
        } => {
            lt_static(src.text(lt))
                && ast_ty_is_provably_static_in_mode(body, src, sig, inner, mode.enter_owned())
        }
        _ => ast_ty_is_provably_static_in_mode(body, src, sig, ty, mode.reset_owned()),
    }
}

fn ast_ty_is_provably_static_in_mode(
    body: &Body,
    src: &Source,
    sig: Option<(&crate::body::sig::Signature, &Source)>,
    ty: TyId,
    mode: DynDefault,
) -> bool {
    let lt_static = |lt: &str| match sig {
        Some((s, s_src)) => lifetime_is_static(s, s_src, lt),
        None => lt == LT_STATIC,
    };
    match *body.ty(ty) {
        AstTy::Never => true,
        AstTy::Ptr { ty: t, .. } => {
            ast_ty_is_provably_static_in_mode(body, src, sig, t, mode.enter_owned())
        }
        AstTy::Paren(t) | AstTy::Slice(t) | AstTy::Array(t, _) | AstTy::Pat(t, _) => {
            ast_ty_is_provably_static_in_mode(body, src, sig, t, mode)
        }
        AstTy::Tuple(list) => body
            .list(list)
            .iter()
            .all(|&t| ast_ty_is_provably_static_in_mode(body, src, sig, TyId(t), mode)),
        AstTy::Ref {
            lifetime: Some(lt),
            ty: inner,
            ..
        } => {
            lt_static(src.text(lt))
                && ast_ty_is_provably_static_in_mode(body, src, sig, inner, mode.enter_owned())
        }
        AstTy::ImplTrait(_, list) | AstTy::DynTrait(list) => {
            let is_dyn = matches!(*body.ty(ty), AstTy::DynTrait(_));
            let bounds = body.list(list);
            let has_lt = bounds
                .iter()
                .any(|&b_id| matches!(body.bounds[b_id as usize], Bound::Lifetime(_)));
            let lifetime_ok = if has_lt {
                bounds.iter().any(|&b_id| {
                    matches!(body.bounds[b_id as usize], Bound::Lifetime(lt) if lt_static(src.text(lt)))
                })
            } else {
                is_dyn && mode == DynDefault::InSigOwned
            };
            lifetime_ok
                && bounds.iter().all(|&b_id| match body.bounds[b_id as usize] {
                    Bound::Lifetime(lt) => lt_static(src.text(lt)),
                    Bound::Trait { path, .. } => {
                        path_args_are_static(body, src, sig, path, &lt_static, mode)
                    }
                    Bound::Use(_) => true,
                })
        }
        AstTy::Path(path_id) => {
            let p = body.path(path_id);
            let segs = body.list(p.segments);
            if p.qself.is_some() || segs.is_empty() {
                return false;
            }
            if segs[..segs.len() - 1]
                .iter()
                .any(|&s| body.segments[s as usize].args.is_some())
            {
                return false;
            }
            let seg = body.segments[segs[segs.len() - 1] as usize];
            let name = crate::names::plain(src.text(seg.name));
            if seg.args.is_none() && is_prim_name(name) {
                return true;
            }
            if is_owned_container(name)
                && let Some(Args::Angle(args)) = seg.args
            {
                return body
                    .list(args)
                    .iter()
                    .all(|&a| match body.args[a as usize] {
                        GenericArg::Ty(t) | GenericArg::Binding { ty: t, .. } => {
                            ast_ty_is_provably_static_in_mode(body, src, sig, t, mode.enter_owned())
                        }
                        GenericArg::Lifetime(lt) => lt_static(src.text(lt)),
                        GenericArg::Const(_) => true,
                        GenericArg::Constraint { .. } => false,
                    });
            }
            false
        }
        _ => false,
    }
}

enum StaticAstOwner {
    Sig(Arc<Signature>),
    Alias(Arc<Alias>),
}

impl Checker<'_, '_> {
    pub(in super::super) fn scrut_variant_field_is_static(
        &mut self,
        pat: PatId,
        scrut: ExprId,
        field_idx: usize,
    ) -> bool {
        let Some(Res::Ctor(adt_item, variant_idx, _)) = self.typed.pat_res.get(&pat).copied()
        else {
            return false;
        };
        let scrut = self.unwrap_parens(scrut);
        let fn_item = match *self.body.expr(scrut) {
            Expr::Method { .. } => match self.typed.res.get(&scrut).copied() {
                Some(Res::Fn(it, _)) => Some(it),
                _ => None,
            },
            Expr::Call(callee, _) => self.call_sig_fn_item(scrut, callee),
            _ => None,
        };
        let Some(fn_item) = fn_item else {
            return false;
        };
        let Ok(sig) = self.cx.signature(fn_item) else {
            return false;
        };
        let Some(mut cur_ty) = sig.ret else {
            return false;
        };
        let mut cur_scope: ItemRef = fn_item;
        let mut owner = StaticAstOwner::Sig(sig);
        for _ in 0..8 {
            let cur_body = match &owner {
                StaticAstOwner::Sig(s) => &s.body,
                StaticAstOwner::Alias(a) => &a.arena,
            };
            while let AstTy::Paren(inner) = *cur_body.ty(cur_ty) {
                cur_ty = inner;
            }
            let AstTy::Path(path_id) = *cur_body.ty(cur_ty) else {
                return false;
            };
            let cur_src = self.cx.item(cur_scope).0;
            let Some(resolved) =
                resolve_ast_type_path(self.cx, Some(cur_scope), cur_body, cur_src, path_id)
            else {
                return false;
            };
            if self.cx.item(resolved).1.kind == ItemKind::Type {
                let Ok(alias) = self.cx.alias_def(resolved) else {
                    return false;
                };
                cur_ty = alias.ty;
                cur_scope = resolved;
                owner = StaticAstOwner::Alias(alias);
                continue;
            }
            if resolved != adt_item {
                return false;
            }
            let Ok(def) = self.cx.adt_def(adt_item) else {
                return false;
            };
            let Some(field) = def
                .variants
                .get(variant_idx as usize)
                .and_then(|v| v.fields.get(field_idx))
            else {
                return false;
            };
            let adt_src = &self.cx.file(adt_item).source;
            let mut f_ty = field.ty;
            while let AstTy::Paren(inner) = *def.body.ty(f_ty) {
                f_ty = inner;
            }
            if let AstTy::Path(fp) = *def.body.ty(f_ty)
                && def.body.path(fp).qself.is_none()
                && def.body.list(def.body.path(fp).segments).len() == 1
            {
                let seg = def.body.segments[def.body.list(def.body.path(fp).segments)[0] as usize];
                let fname = plain(adt_src.text(seg.name));
                if seg.args.is_none()
                    && let Some(param_pos) = def
                        .generics
                        .params
                        .iter()
                        .filter(|g| g.kind == GenericKind::Type)
                        .position(|g| plain(adt_src.text(g.name)) == fname)
                {
                    let cur_body = match &owner {
                        StaticAstOwner::Sig(s) => &s.body,
                        StaticAstOwner::Alias(a) => &a.arena,
                    };
                    let cur_src = self.cx.item(cur_scope).0;
                    let cur_sig = match &owner {
                        StaticAstOwner::Sig(s) => Some((s.as_ref(), cur_src)),
                        StaticAstOwner::Alias(_) => None,
                    };
                    let Some(&last_seg) = cur_body.list(cur_body.path(path_id).segments).last()
                    else {
                        return false;
                    };
                    let Some(Args::Angle(angle)) = cur_body.segments[last_seg as usize].args else {
                        return false;
                    };
                    let Some(arg_ty) = cur_body
                        .list(angle)
                        .iter()
                        .filter_map(|&a| match cur_body.args[a as usize] {
                            GenericArg::Ty(t) => Some(t),
                            _ => None,
                        })
                        .nth(param_pos)
                    else {
                        return false;
                    };
                    return ast_ty_is_provably_static_in_mode(
                        cur_body,
                        cur_src,
                        cur_sig,
                        arg_ty,
                        DynDefault::InSig,
                    );
                }
            }
            return ast_ty_is_provably_static_in_mode(
                &def.body,
                adt_src,
                None,
                f_ty,
                DynDefault::InSig,
            );
        }
        false
    }

    pub(in super::super) fn sig_ret_ast_is_provably_static(
        &mut self,
        fn_item: ItemRef,
        sig: &Signature,
        sig_src: &Source,
        ret_ast: TyId,
    ) -> bool {
        !sig.is_async
            && ast_ty_is_provably_static_in_item(self.cx, fn_item, sig, sig_src, ret_ast, &[])
    }

    pub(in super::super) fn for_iter_item_is_static(&mut self, iter: ExprId) -> bool {
        let Some(local) = self.expr_plain_local(self.unwrap_parens(iter)) else {
            return false;
        };
        if !self.is_param_local(local) {
            return false;
        }
        let Some(pos) = self.typed.params.iter().position(|&p| p == local) else {
            return false;
        };
        let Ok(sig) = self.cx.signature(self.item) else {
            return false;
        };
        let Some(param_idx) = sig_param_idx(pos, sig.receiver.is_some()) else {
            return false;
        };
        let Some(&(_, param_ty)) = sig.params.get(param_idx) else {
            return false;
        };
        let mut item_tys = Vec::new();
        for_each_param_bound_list(&sig, self.src, param_ty, |list| {
            for &b_id in sig.body.list(list) {
                let Bound::Trait { path, .. } = sig.body.bounds[b_id as usize] else {
                    continue;
                };
                let Some(&last_seg) = sig.body.list(sig.body.path(path).segments).last() else {
                    continue;
                };
                let seg = sig.body.segments[last_seg as usize];
                if !matches!(plain(self.src.text(seg.name)), "IntoIterator" | "Iterator") {
                    continue;
                }
                let Some(Args::Angle(args)) = seg.args else {
                    continue;
                };
                for &a_id in sig.body.list(args) {
                    if let GenericArg::Binding { name, ty, .. } = sig.body.args[a_id as usize]
                        && plain(self.src.text(name)) == "Item"
                    {
                        item_tys.push(ty);
                    }
                }
            }
        });
        if item_tys.is_empty() {
            return false;
        }
        let static_names = self.static_param_names(&sig);
        item_tys.into_iter().any(|item_ty| {
            ast_ty_is_provably_static_in_item(
                self.cx,
                self.item,
                &sig,
                self.src,
                item_ty,
                &static_names,
            )
        })
    }

    fn static_param_names(&mut self, sig: &Signature) -> Vec<String> {
        let mut names = Vec::new();
        let mut idx = 0u16;
        if let Some(impl_ref) = self.cx.owner_impl(self.item) {
            if let Ok(hdr) = self.cx.impl_header(impl_ref) {
                let impl_src = &self.cx.impl_file(impl_ref).source;
                for g in crate::sema::infer::scope::type_params(&hdr.generics) {
                    if self.static_type_params.contains(&idx) {
                        names.push(impl_src.text(g.name).to_owned());
                    }
                    idx += 1;
                }
            }
        } else if let Some(trait_ref) = self.cx.owner_trait(self.item) {
            idx += 1;
            if let Ok(tg) = self.cx.trait_generics(trait_ref) {
                let tsrc = &self.cx.file(trait_ref).source;
                for g in crate::sema::infer::scope::type_params(&tg.1.params) {
                    if self.static_type_params.contains(&idx) {
                        names.push(tsrc.text(g.name).to_owned());
                    }
                    idx += 1;
                }
            }
        }
        for g in crate::sema::infer::scope::type_params(&sig.generics) {
            if self.static_type_params.contains(&idx) {
                names.push(self.src.text(g.name).to_owned());
            }
            idx += 1;
        }
        names
    }
}

struct StaticAstCtx<'a> {
    scope: ItemRef,
    body: &'a Body,
    src: &'a Source,
    sig: Option<(&'a Signature, &'a Source)>,
    static_names: &'a [String],
}

pub(super) fn ast_ty_is_provably_static_in_item(
    cx: &mut crate::sema::Context<'_>,
    scope: ItemRef,
    sig: &Signature,
    src: &Source,
    ty: TyId,
    static_names: &[String],
) -> bool {
    let ctx = StaticAstCtx {
        scope,
        body: &sig.body,
        src,
        sig: Some((sig, src)),
        static_names,
    };
    ast_ty_is_provably_static_resolved(cx, &ctx, ty, DynDefault::InSig, 0)
}

fn ast_ty_is_provably_static_resolved(
    cx: &mut crate::sema::Context<'_>,
    ctx: &StaticAstCtx<'_>,
    ty: TyId,
    mode: DynDefault,
    depth: u8,
) -> bool {
    if depth >= 8 {
        return false;
    }
    if ast_ty_is_provably_static_in_mode(ctx.body, ctx.src, ctx.sig, ty, mode) {
        return true;
    }
    let lt_static = |lt: &str| match ctx.sig {
        Some((s, s_src)) => lifetime_is_static(s, s_src, lt),
        None => lt == LT_STATIC,
    };
    match *ctx.body.ty(ty) {
        AstTy::Ptr { ty: t, .. } => {
            ast_ty_is_provably_static_resolved(cx, ctx, t, mode.enter_owned(), depth + 1)
        }
        AstTy::Paren(t) | AstTy::Slice(t) | AstTy::Array(t, _) | AstTy::Pat(t, _) => {
            ast_ty_is_provably_static_resolved(cx, ctx, t, mode, depth + 1)
        }
        AstTy::Tuple(list) => ctx
            .body
            .list(list)
            .iter()
            .all(|&t| ast_ty_is_provably_static_resolved(cx, ctx, TyId(t), mode, depth + 1)),
        AstTy::Ref {
            lifetime: Some(lt),
            ty: inner,
            ..
        } => {
            lt_static(ctx.src.text(lt))
                && ast_ty_is_provably_static_resolved(cx, ctx, inner, mode.enter_owned(), depth + 1)
        }
        AstTy::Path(path_id) => path_is_provably_static_resolved(cx, ctx, path_id, mode, depth),
        _ => false,
    }
}

fn path_is_provably_static_resolved(
    cx: &mut crate::sema::Context<'_>,
    ctx: &StaticAstCtx<'_>,
    path_id: crate::body::PathId,
    mode: DynDefault,
    depth: u8,
) -> bool {
    let p = ctx.body.path(path_id);
    let segs = ctx.body.list(p.segments);
    if p.qself.is_some()
        || segs.is_empty()
        || segs[..segs.len() - 1]
            .iter()
            .any(|&s| ctx.body.segments[s as usize].args.is_some())
    {
        return false;
    }
    let seg = ctx.body.segments[segs[segs.len() - 1] as usize];
    let name = plain(ctx.src.text(seg.name));
    if segs.len() == 1 && seg.args.is_none() && ctx.static_names.iter().any(|s| s == name) {
        return true;
    }
    let args_static =
        |cx: &mut crate::sema::Context<'_>, args: Option<Args>, arg_mode: DynDefault| -> bool {
            let lt_static = |lt: &str| match ctx.sig {
                Some((s, s_src)) => lifetime_is_static(s, s_src, lt),
                None => lt == LT_STATIC,
            };
            match args {
                None => true,
                Some(Args::Angle(list)) => {
                    ctx.body
                        .list(list)
                        .iter()
                        .all(|&a| match ctx.body.args[a as usize] {
                            GenericArg::Ty(t) | GenericArg::Binding { ty: t, .. } => {
                                ast_ty_is_provably_static_resolved(cx, ctx, t, arg_mode, depth + 1)
                            }
                            GenericArg::Lifetime(lt) => lt_static(ctx.src.text(lt)),
                            GenericArg::Const(_) => true,
                            GenericArg::Constraint { .. } => false,
                        })
                }
                _ => false,
            }
        };
    if is_owned_container(name) && seg.args.is_some() {
        return args_static(cx, seg.args, mode.enter_owned());
    }
    let Some(resolved) = resolve_ast_type_path(cx, Some(ctx.scope), ctx.body, ctx.src, path_id)
    else {
        return false;
    };
    match cx.item(resolved).1.kind {
        ItemKind::Type => {
            let Ok(alias) = cx.alias_def(resolved) else {
                return false;
            };
            if !alias.generics.1.params.is_empty() {
                return false;
            }
            let alias_src = cx.item(resolved).0;
            let alias_ctx = StaticAstCtx {
                scope: resolved,
                body: &alias.arena,
                src: alias_src,
                sig: None,
                static_names: &[],
            };
            ast_ty_is_provably_static_resolved(cx, &alias_ctx, alias.ty, mode, depth + 1)
        }
        ItemKind::Struct | ItemKind::Enum | ItemKind::Union => {
            let Ok(def) = cx.adt_def(resolved) else {
                return false;
            };
            !def.generics
                .params
                .iter()
                .any(|g| g.kind == GenericKind::Lifetime)
                && args_static(cx, seg.args, mode.enter_owned())
        }
        _ => false,
    }
}

pub(super) fn non_env_sources(
    srcs: &[super::super::propagate::BorrowSource],
) -> Vec<super::super::propagate::BorrowSource> {
    srcs.iter().filter(|s| !s.env_only).cloned().collect()
}

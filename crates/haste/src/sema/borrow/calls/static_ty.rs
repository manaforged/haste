use crate::body::{Args, Body, Bound, GenericArg, Ty as AstTy, TyId};
use crate::outline::source::Source;

use super::lifetimes::{LT_STATIC, is_owned_container, is_prim_name, lifetime_is_static};

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

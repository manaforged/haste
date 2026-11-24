use crate::body::{Args, Body, Bound, GenericArg, Ty as AstTy, TyId};
use crate::outline::item::Item;
use crate::outline::source::Source;

#[derive(Default)]
pub(in super::super) struct Lifetimes<'s> {
    pub(in super::super) has_elided: bool,
    pub(in super::super) saw_static: bool,
    pub(in super::super) named: Vec<&'s str>,
    pub(in super::super) names: Vec<&'s str>,
}

pub(in super::super) fn collect_ty_lifetimes<'s>(
    body: &Body,
    src: &'s Source,
    ty: TyId,
    out: &mut Lifetimes<'s>,
) {
    match *body.ty(ty) {
        AstTy::Ref {
            lifetime,
            ty: inner,
            ..
        } => {
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
        AstTy::Slice(t)
        | AstTy::Array(t, _)
        | AstTy::Ptr { ty: t, .. }
        | AstTy::Paren(t)
        | AstTy::Pat(t, _) => {
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

pub(in super::super) fn ty_shares_name(
    body: &Body,
    src: &Source,
    ty: TyId,
    ret_names: &[&str],
) -> bool {
    if ret_names.is_empty() {
        return false;
    }
    let mut param_lts = Lifetimes::default();
    collect_ty_lifetimes(body, src, ty, &mut param_lts);
    param_lts.names.iter().any(|n| ret_names.contains(n))
}

pub(in super::super) fn receiver_has_any_named(src: &Source, item: &Item, named: &[&str]) -> bool {
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

pub(in super::super) fn compute_param_is_static(
    cx: &mut crate::sema::Context<'_>,
    item: crate::names::ItemRef,
    sig: &crate::body::sig::Signature,
    src: &Source,
    typed: &crate::sema::infer::Typed,
    n: usize,
) -> Vec<bool> {
    let mut out = vec![true; n];
    let impl_has_lt = cx.owner_impl(item).is_some_and(|block| {
        cx.impl_header(block).is_ok_and(|hdr| {
            hdr.generics
                .iter()
                .any(|g| g.kind == crate::body::sig::GenericKind::Lifetime)
        })
    });
    let mut cursor = 0usize;
    if let Some((kind, _)) = sig.receiver {
        if let Some(&local) = typed.params.get(cursor)
            && (local as usize) < n
        {
            out[local as usize] = match kind {
                crate::body::sig::SelfKind::Ref | crate::body::sig::SelfKind::RefMut => sig
                    .receiver_lifetime
                    .is_some_and(|lt| lifetime_is_static(sig, src, src.text(lt))),
                crate::body::sig::SelfKind::Typed(ty) => {
                    param_ty_is_static(sig, src, ty, impl_has_lt)
                }
                crate::body::sig::SelfKind::Value => !impl_has_lt,
            };
        }
        cursor += 1;
    }
    for &(_, param_ty) in &sig.params {
        if let Some(&local) = typed.params.get(cursor)
            && (local as usize) < n
        {
            out[local as usize] = param_ty_is_static(sig, src, param_ty, impl_has_lt);
        }
        cursor += 1;
    }
    out
}

fn has_static_lifetime_bound(
    sig: &crate::body::sig::Signature,
    src: &Source,
    bounds: crate::body::List,
) -> bool {
    sig.body.list(bounds).iter().any(|&b| {
        matches!(
            sig.body.bounds[b as usize],
            Bound::Lifetime(lt) if src.text(lt) == "'static"
        )
    })
}

fn lifetime_is_static(sig: &crate::body::sig::Signature, src: &Source, name: &str) -> bool {
    name == "'static"
        || sig.generics.iter().any(|g| {
            g.kind == crate::body::sig::GenericKind::Lifetime
                && src.text(g.name) == name
                && has_static_lifetime_bound(sig, src, g.bounds)
        })
}

fn param_ty_is_static(
    sig: &crate::body::sig::Signature,
    src: &Source,
    ty: TyId,
    impl_has_lt: bool,
) -> bool {
    let mut lts = Lifetimes::default();
    collect_ty_lifetimes(&sig.body, src, ty, &mut lts);
    if lts.has_elided {
        return false;
    }
    if impl_has_lt && lts.names.contains(&"Self") {
        return false;
    }
    if lts
        .named
        .iter()
        .any(|&lt| !lifetime_is_static(sig, src, lt))
    {
        return false;
    }
    let bounded_static = |name: &str| {
        sig.generics.iter().any(|g| {
            g.kind == crate::body::sig::GenericKind::Type
                && src.text(g.name) == name
                && has_static_lifetime_bound(sig, src, g.bounds)
        }) || sig.predicates.iter().any(|p| {
            p.ty.is_some_and(|pt| match *sig.body.ty(pt) {
                AstTy::Path(path) => {
                    let segs = sig.body.list(sig.body.path(path).segments);
                    segs.len() == 1 && src.text(sig.body.segments[segs[0] as usize].name) == name
                }
                _ => false,
            }) && has_static_lifetime_bound(sig, src, p.bounds)
        })
    };
    !lts.names.iter().any(|&name| {
        sig.generics
            .iter()
            .any(|g| g.kind == crate::body::sig::GenericKind::Type && src.text(g.name) == name)
            && !bounded_static(name)
    })
}

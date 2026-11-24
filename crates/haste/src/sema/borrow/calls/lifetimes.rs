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

pub(in super::super) fn impl_has_lifetime(
    cx: &mut crate::sema::Context<'_>,
    item: crate::names::ItemRef,
) -> bool {
    cx.owner_impl(item).is_some_and(|block| {
        cx.impl_header(block).is_ok_and(|hdr| {
            hdr.generics
                .iter()
                .any(|g| g.kind == crate::body::sig::GenericKind::Lifetime)
        })
    })
}

pub(in super::super) struct ParamLifetimes<'s> {
    pub(in super::super) is_static: Vec<bool>,
    pub(in super::super) inner_is_static: Vec<bool>,
    pub(in super::super) named_lts: Vec<Vec<&'s str>>,
    pub(in super::super) has_elided_lt: Vec<bool>,
}

pub(in super::super) fn is_prim_name(name: &str) -> bool {
    crate::names::Primitive::named(name).is_some() || name == "String"
}

pub(in super::super) fn is_owned_container(name: &str) -> bool {
    matches!(name, "Option" | "Result" | "Vec" | "Box")
}

pub(in super::super) fn pred_matches_name(
    sig: &crate::body::sig::Signature,
    src: &Source,
    pred: &crate::body::sig::Predicate,
    name: &str,
) -> bool {
    pred.ty.is_some_and(|pt| match *sig.body.ty(pt) {
        AstTy::Path(path) => {
            let segs = sig.body.list(sig.body.path(path).segments);
            segs.len() == 1 && src.text(sig.body.segments[segs[0] as usize].name) == name
        }
        _ => false,
    })
}

pub(in super::super) fn compute_param_lifetimes<'s>(
    cx: &mut crate::sema::Context<'_>,
    item: crate::names::ItemRef,
    sig: &crate::body::sig::Signature,
    src: &'s Source,
    typed: &crate::sema::infer::Typed,
    n: usize,
) -> ParamLifetimes<'s> {
    let mut info = ParamLifetimes {
        is_static: vec![true; n],
        inner_is_static: vec![false; n],
        named_lts: vec![Vec::new(); n],
        has_elided_lt: vec![false; n],
    };
    let impl_has_lt = impl_has_lifetime(cx, item);
    let mut cursor = 0usize;
    if let Some((kind, _)) = sig.receiver {
        if let Some(&local) = typed.params.get(cursor)
            && (local as usize) < n
        {
            let slot = local as usize;
            match kind {
                crate::body::sig::SelfKind::Ref | crate::body::sig::SelfKind::RefMut => {
                    if let Some(lt) = sig.receiver_lifetime {
                        let name = src.text(lt);
                        let lt_static = lifetime_is_static(sig, src, name);
                        info.is_static[slot] = lt_static;
                        info.inner_is_static[slot] = !impl_has_lt || lt_static;
                        if name == "'_" {
                            info.has_elided_lt[slot] = true;
                        } else if name != "'static" {
                            info.named_lts[slot].push(name);
                        }
                    } else {
                        info.is_static[slot] = false;
                        info.has_elided_lt[slot] = true;
                        info.inner_is_static[slot] = !impl_has_lt;
                    }
                }
                crate::body::sig::SelfKind::Typed(ty) => {
                    record_param_ty(sig, src, ty, impl_has_lt, slot, &mut info);
                }
                crate::body::sig::SelfKind::Value => {
                    info.is_static[slot] = !impl_has_lt;
                    info.has_elided_lt[slot] = impl_has_lt;
                }
            }
        }
        cursor += 1;
    }
    for &(_, param_ty) in &sig.params {
        if let Some(&local) = typed.params.get(cursor)
            && (local as usize) < n
        {
            record_param_ty(sig, src, param_ty, impl_has_lt, local as usize, &mut info);
        }
        cursor += 1;
    }
    info
}

fn record_param_ty<'s>(
    sig: &crate::body::sig::Signature,
    src: &'s Source,
    ty: TyId,
    impl_has_lt: bool,
    slot: usize,
    info: &mut ParamLifetimes<'s>,
) {
    let mut lts = Lifetimes::default();
    collect_ty_lifetimes(&sig.body, src, ty, &mut lts);
    info.is_static[slot] = param_ty_is_static(sig, src, &lts, impl_has_lt);
    info.has_elided_lt[slot] = lts.has_elided
        || (impl_has_lt && lts.names.contains(&"Self"))
        || has_unbounded_type_param(sig, src, &lts);
    info.named_lts[slot] = lts.named;
    if let AstTy::Ref { ty: inner, .. } = *sig.body.ty(ty) {
        info.inner_is_static[slot] = ast_ty_is_provably_static(sig, src, inner, impl_has_lt);
    }
}

fn has_lifetime_bound(
    sig: &crate::body::sig::Signature,
    src: &Source,
    bounds: crate::body::List,
    target: &str,
) -> bool {
    sig.body.list(bounds).iter().any(|&b| {
        matches!(
            sig.body.bounds[b as usize],
            Bound::Lifetime(lt) if src.text(lt) == target
        )
    })
}

fn lifetime_is_static(sig: &crate::body::sig::Signature, src: &Source, name: &str) -> bool {
    lifetime_outlives(sig, src, name, "'static")
}

pub(in super::super) fn lifetime_outlives(
    sig: &crate::body::sig::Signature,
    src: &Source,
    a: &str,
    b: &str,
) -> bool {
    if a == b || a == "'static" {
        return true;
    }
    let mut visited = vec![a];
    let mut cursor = 0;
    while cursor < visited.len() {
        let cur = visited[cursor];
        cursor += 1;
        let bound_lists = sig
            .generics
            .iter()
            .filter(|g| {
                g.kind == crate::body::sig::GenericKind::Lifetime && src.text(g.name) == cur
            })
            .map(|g| g.bounds)
            .chain(
                sig.predicates
                    .iter()
                    .filter(|p| p.lifetime.is_some_and(|lt| src.text(lt) == cur))
                    .map(|p| p.bounds),
            );
        for bounds in bound_lists {
            for &bound_id in sig.body.list(bounds) {
                if let Bound::Lifetime(lt) = sig.body.bounds[bound_id as usize] {
                    let next = src.text(lt);
                    if next == b || next == "'static" {
                        return true;
                    }
                    if !visited.contains(&next) {
                        visited.push(next);
                    }
                }
            }
        }
    }
    false
}

fn has_unbounded_type_param(
    sig: &crate::body::sig::Signature,
    src: &Source,
    lts: &Lifetimes<'_>,
) -> bool {
    let bounded_static = |name: &str| {
        sig.generics.iter().any(|g| {
            g.kind == crate::body::sig::GenericKind::Type
                && src.text(g.name) == name
                && has_lifetime_bound(sig, src, g.bounds, "'static")
        }) || sig.predicates.iter().any(|p| {
            pred_matches_name(sig, src, p, name)
                && has_lifetime_bound(sig, src, p.bounds, "'static")
        })
    };
    lts.names.iter().any(|&name| {
        sig.generics
            .iter()
            .any(|g| g.kind == crate::body::sig::GenericKind::Type && src.text(g.name) == name)
            && !bounded_static(name)
    })
}

fn param_ty_is_static(
    sig: &crate::body::sig::Signature,
    src: &Source,
    lts: &Lifetimes<'_>,
    impl_has_lt: bool,
) -> bool {
    if lts.has_elided || (impl_has_lt && lts.names.contains(&"Self")) {
        return false;
    }
    if lts
        .named
        .iter()
        .any(|&lt| !lifetime_is_static(sig, src, lt))
    {
        return false;
    }
    !has_unbounded_type_param(sig, src, lts)
}

fn ast_ty_is_provably_static(
    sig: &crate::body::sig::Signature,
    src: &Source,
    ty: TyId,
    impl_has_lt: bool,
) -> bool {
    match *sig.body.ty(ty) {
        AstTy::Never | AstTy::Ptr { .. } => true,
        AstTy::Paren(t) | AstTy::Slice(t) | AstTy::Array(t, _) | AstTy::Pat(t, _) => {
            ast_ty_is_provably_static(sig, src, t, impl_has_lt)
        }
        AstTy::Tuple(list) => sig
            .body
            .list(list)
            .iter()
            .all(|&t| ast_ty_is_provably_static(sig, src, TyId(t), impl_has_lt)),
        AstTy::Ref {
            lifetime: Some(lt),
            ty: inner,
            ..
        } => {
            lifetime_is_static(sig, src, src.text(lt))
                && ast_ty_is_provably_static(sig, src, inner, impl_has_lt)
        }
        AstTy::Path(path_id) => {
            let p = sig.body.path(path_id);
            let segs = sig.body.list(p.segments);
            if p.qself.is_some() || segs.len() != 1 {
                return false;
            }
            let seg = sig.body.segments[segs[0] as usize];
            let name = src.text(seg.name);
            if seg.args.is_none() && is_prim_name(name) {
                return true;
            }
            if is_owned_container(name)
                && let Some(Args::Angle(args)) = seg.args
            {
                return sig
                    .body
                    .list(args)
                    .iter()
                    .all(|&a| match sig.body.args[a as usize] {
                        GenericArg::Ty(t) | GenericArg::Binding { ty: t, .. } => {
                            ast_ty_is_provably_static(sig, src, t, impl_has_lt)
                        }
                        GenericArg::Lifetime(lt) => lifetime_is_static(sig, src, src.text(lt)),
                        GenericArg::Const(_) => true,
                        GenericArg::Constraint { .. } => false,
                    });
            }
            false
        }
        _ => false,
    }
}

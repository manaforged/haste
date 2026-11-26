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
                collect_segment_args_lifetimes(body, src, body.segments[seg_id as usize].args, out);
            }
        }
        AstTy::Tuple(list) => {
            for &t in body.list(list) {
                collect_ty_lifetimes(body, src, TyId(t), out);
            }
        }
        AstTy::ImplTrait(_, list) | AstTy::DynTrait(list) => {
            let has_static = body.list(list).iter().any(|&b_id| {
                matches!(body.bounds[b_id as usize], Bound::Lifetime(lt) if src.text(lt) == LT_STATIC)
            });
            if has_static {
                out.saw_static = true;
                return;
            }
            for &b_id in body.list(list) {
                match body.bounds[b_id as usize] {
                    Bound::Lifetime(lt) => push_lt(src.text(lt), out),
                    Bound::Trait { path, .. } => {
                        for &seg_id in body.list(body.path(path).segments) {
                            collect_segment_args_lifetimes(
                                body,
                                src,
                                body.segments[seg_id as usize].args,
                                out,
                            );
                        }
                    }
                    Bound::Use(_) => {}
                }
            }
        }
        AstTy::Fn { params, ret, .. } => {
            let mut fn_lts = Lifetimes::default();
            for &t in body.list(params) {
                collect_ty_lifetimes(body, src, TyId(t), &mut fn_lts);
            }
            if let Some(r) = ret {
                collect_ty_lifetimes(body, src, r, &mut fn_lts);
            }
            merge_fn_lifetimes(fn_lts, out);
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

fn collect_segment_args_lifetimes<'s>(
    body: &Body,
    src: &'s Source,
    args: Option<Args>,
    out: &mut Lifetimes<'s>,
) {
    match args {
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
            let mut fn_lts = Lifetimes::default();
            for &t in body.list(inputs) {
                collect_ty_lifetimes(body, src, TyId(t), &mut fn_lts);
            }
            if let Some(o) = output {
                collect_ty_lifetimes(body, src, o, &mut fn_lts);
            }
            merge_fn_lifetimes(fn_lts, out);
        }
        Some(Args::ReturnType) | None => {}
    }
}

fn merge_fn_lifetimes<'s>(fn_lts: Lifetimes<'s>, out: &mut Lifetimes<'s>) {
    out.saw_static |= fn_lts.saw_static;
    for lt in fn_lts.named {
        if !out.named.contains(&lt) {
            out.named.push(lt);
        }
    }
    for name in fn_lts.names {
        if !out.names.contains(&name) {
            out.names.push(name);
        }
    }
}

pub(in super::super) const LT_STATIC: &str = "'static";
pub(in super::super) const LT_ELIDED: &str = "'_";

pub(in super::super) fn is_named_lt(name: &str) -> bool {
    name != LT_ELIDED && name != LT_STATIC
}

pub(super) fn owned_lts(lts: Vec<&str>) -> Vec<String> {
    lts.into_iter().map(str::to_owned).collect()
}

fn push_lt<'s>(text: &'s str, out: &mut Lifetimes<'s>) {
    if text == LT_ELIDED {
        out.has_elided = true;
    } else if text == LT_STATIC {
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

fn impl_self_lifetimes(
    cx: &mut crate::sema::Context<'_>,
    item: crate::names::ItemRef,
    sig: &crate::body::sig::Signature,
    src: &Source,
) -> (bool, bool, Vec<String>) {
    let Some(block) = cx.owner_impl(item) else {
        return (false, false, Vec::new());
    };
    let Ok(hdr) = cx.impl_header(block) else {
        return (false, false, Vec::new());
    };
    let has_lt_generic = hdr
        .generics
        .iter()
        .any(|g| g.kind == crate::body::sig::GenericKind::Lifetime);
    let impl_src = &cx.impl_file(block).source;
    let bounded_static = |body: &Body, s: &Source, bounds: crate::body::List| {
        body.list(bounds).iter().any(
            |&b| matches!(body.bounds[b as usize], Bound::Lifetime(lt) if s.text(lt) == LT_STATIC),
        )
    };
    let has_non_static_ty = hdr.generics.iter().any(|g| {
        if g.kind != crate::body::sig::GenericKind::Type {
            return false;
        }
        let name = impl_src.text(g.name);
        !(bounded_static(&hdr.body, impl_src, g.bounds)
            || hdr.predicates.iter().any(|p| {
                pred_matches_name_in(&hdr.body, impl_src, p, name)
                    && bounded_static(&hdr.body, impl_src, p.bounds)
            })
            || sig.predicates.iter().any(|p| {
                pred_matches_name(sig, src, p, name) && bounded_static(&sig.body, src, p.bounds)
            }))
    });
    let mut lts = Lifetimes::default();
    collect_ty_lifetimes(&hdr.body, impl_src, hdr.self_ty, &mut lts);
    let has_lt = has_lt_generic || has_non_static_ty || lts.has_elided || !lts.named.is_empty();
    let has_elided = has_lt && lts.named.is_empty();
    (has_lt, has_elided, owned_lts(lts.named))
}

pub(in super::super) struct ParamLifetimes {
    pub(in super::super) is_static: Vec<bool>,
    pub(in super::super) inner_is_static: Vec<bool>,
    pub(in super::super) named_lts: Vec<Vec<String>>,
    pub(in super::super) has_elided_lt: Vec<bool>,
    pub(in super::super) inner_named_lts: Vec<Vec<String>>,
    pub(in super::super) inner_has_elided_lt: Vec<bool>,
}

impl ParamLifetimes {
    pub(super) fn slot(&self, slot: usize, peel: bool) -> (bool, Option<&[String]>) {
        if peel {
            (
                self.inner_has_elided_lt.get(slot).copied().unwrap_or(true),
                self.inner_named_lts.get(slot).map(Vec::as_slice),
            )
        } else {
            (
                self.has_elided_lt.get(slot).copied().unwrap_or(true),
                self.named_lts.get(slot).map(Vec::as_slice),
            )
        }
    }
}

pub(in super::super) fn is_prim_name(name: &str) -> bool {
    crate::names::Primitive::named(name).is_some() || name == "String"
}

pub(in super::super) fn is_owned_container(name: &str) -> bool {
    matches!(
        name,
        "Option"
            | "Result"
            | "Vec"
            | "Box"
            | "Cell"
            | "UnsafeCell"
            | "RefCell"
            | "OnceCell"
            | "OnceLock"
            | "Mutex"
            | "RwLock"
            | "SyncUnsafeCell"
            | "UnsafePinned"
            | "PhantomData"
            | "ManuallyDrop"
            | "Pin"
            | "Rc"
            | "Arc"
            | "NonNull"
            | "Unique"
    )
}

pub(in super::super) fn pred_matches_name_in(
    body: &Body,
    src: &Source,
    pred: &crate::body::sig::Predicate,
    name: &str,
) -> bool {
    pred.ty.is_some_and(|pt| match *body.ty(pt) {
        AstTy::Path(path) => {
            let p = body.path(path);
            let segs = body.list(p.segments);
            p.qself.is_none()
                && segs.len() == 1
                && src.text(body.segments[segs[0] as usize].name) == name
        }
        _ => false,
    })
}

pub(in super::super) fn pred_matches_name(
    sig: &crate::body::sig::Signature,
    src: &Source,
    pred: &crate::body::sig::Predicate,
    name: &str,
) -> bool {
    pred_matches_name_in(&sig.body, src, pred, name)
}

pub(in super::super) fn compute_param_lifetimes(
    cx: &mut crate::sema::Context<'_>,
    item: crate::names::ItemRef,
    sig: &crate::body::sig::Signature,
    src: &Source,
    typed: &crate::sema::infer::Typed,
    (n, static_names): (usize, &[String]),
) -> ParamLifetimes {
    let mut info = ParamLifetimes {
        is_static: vec![true; n],
        inner_is_static: vec![false; n],
        named_lts: vec![Vec::new(); n],
        has_elided_lt: vec![false; n],
        inner_named_lts: vec![Vec::new(); n],
        inner_has_elided_lt: vec![false; n],
    };
    for &p in &typed.params {
        if let Some(slot) = info.is_static.get_mut(p as usize) {
            *slot = false;
        }
    }
    let (impl_has_lt, impl_has_elided, impl_named) = impl_self_lifetimes(cx, item, sig, src);
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
                        if name == LT_ELIDED {
                            info.has_elided_lt[slot] = true;
                        } else if is_named_lt(name) {
                            info.named_lts[slot].push(name.to_owned());
                        }
                    } else {
                        info.is_static[slot] = false;
                        info.has_elided_lt[slot] = true;
                        info.inner_is_static[slot] = !impl_has_lt;
                    }
                    info.inner_named_lts[slot] = impl_named.clone();
                    info.inner_has_elided_lt[slot] = !info.inner_is_static[slot] && impl_has_elided;
                }
                crate::body::sig::SelfKind::Typed(ty) => {
                    record_param_ty(sig, src, ty, (impl_has_lt, static_names), slot, &mut info);
                }
                crate::body::sig::SelfKind::Value => {
                    info.is_static[slot] = !impl_has_lt;
                    info.named_lts[slot] = impl_named.clone();
                    info.has_elided_lt[slot] = impl_has_elided;
                }
            }
        }
        cursor += 1;
    }
    for &(_, param_ty) in &sig.params {
        if let Some(&local) = typed.params.get(cursor)
            && (local as usize) < n
        {
            record_param_ty(
                sig,
                src,
                param_ty,
                (impl_has_lt, static_names),
                local as usize,
                &mut info,
            );
        }
        cursor += 1;
    }
    info
}

fn has_elided_or_unbounded_lt(
    sig: &crate::body::sig::Signature,
    src: &Source,
    lts: &Lifetimes<'_>,
    impl_has_lt: bool,
    static_names: &[String],
) -> bool {
    lts.has_elided
        || (impl_has_lt && lts.names.contains(&"Self"))
        || has_unbounded_type_param(sig, src, lts, static_names)
}

fn record_param_ty(
    sig: &crate::body::sig::Signature,
    src: &Source,
    ty: TyId,
    (impl_has_lt, static_names): (bool, &[String]),
    slot: usize,
    info: &mut ParamLifetimes,
) {
    let mut lts = Lifetimes::default();
    collect_ty_lifetimes(&sig.body, src, ty, &mut lts);
    info.is_static[slot] = param_ty_is_static(sig, src, &lts, impl_has_lt, static_names);
    info.has_elided_lt[slot] =
        has_elided_or_unbounded_lt(sig, src, &lts, impl_has_lt, static_names);
    info.named_lts[slot] = owned_lts(lts.named);
    if let AstTy::Ref {
        lifetime,
        ty: inner,
        ..
    } = *sig.body.ty(ty)
    {
        if let Some(lt) = lifetime
            && is_named_lt(src.text(lt))
        {
            info.has_elided_lt[slot] = false;
            info.named_lts[slot] = vec![src.text(lt).to_owned()];
        }
        info.inner_is_static[slot] = super::return_lt::ast_ty_is_provably_static(sig, src, inner);
        let mut inner_lts = Lifetimes::default();
        collect_ty_lifetimes(&sig.body, src, inner, &mut inner_lts);
        info.inner_has_elided_lt[slot] =
            has_elided_or_unbounded_lt(sig, src, &inner_lts, impl_has_lt, static_names);
        info.inner_named_lts[slot] = owned_lts(inner_lts.named);
    }
}

pub(in super::super) fn lifetime_is_static(
    sig: &crate::body::sig::Signature,
    src: &Source,
    name: &str,
) -> bool {
    lifetime_outlives(sig, src, name, "'static")
}

pub(super) fn push_outlives_in_header<'s>(
    body: &Body,
    src: &'s Source,
    generics: &[crate::body::sig::Generic],
    predicates: &[crate::body::sig::Predicate],
    cur: &str,
    visited: &mut Vec<&'s str>,
    target: &str,
) -> bool {
    let bound_lists = generics
        .iter()
        .filter(|g| g.kind == crate::body::sig::GenericKind::Lifetime && src.text(g.name) == cur)
        .map(|g| g.bounds)
        .chain(
            predicates
                .iter()
                .filter(|p| p.lifetime.is_some_and(|lt| src.text(lt) == cur))
                .map(|p| p.bounds),
        );
    for bounds in bound_lists {
        for &bound_id in body.list(bounds) {
            if let Bound::Lifetime(lt) = body.bounds[bound_id as usize] {
                let next = src.text(lt);
                if next == target || next == LT_STATIC {
                    return true;
                }
                if !visited.contains(&next) {
                    visited.push(next);
                }
            }
        }
    }
    false
}

pub(super) fn lifetime_outlives_in_headers<'s>(
    headers: &[(
        &Body,
        &'s Source,
        &[crate::body::sig::Generic],
        &[crate::body::sig::Predicate],
    )],
    a: &'s str,
    b: &str,
) -> bool {
    if a == b || a == LT_STATIC {
        return true;
    }
    let mut visited = vec![a];
    let mut cursor = 0;
    while cursor < visited.len() {
        let cur = visited[cursor];
        cursor += 1;
        for &(body, src, generics, predicates) in headers {
            if push_outlives_in_header(body, src, generics, predicates, cur, &mut visited, b) {
                return true;
            }
        }
    }
    false
}

pub(in super::super) fn lifetime_outlives(
    sig: &crate::body::sig::Signature,
    src: &Source,
    a: &str,
    b: &str,
) -> bool {
    lifetime_outlives_in_headers(&[(&sig.body, src, &sig.generics, &sig.predicates)], a, b)
}

fn has_unbounded_type_param(
    sig: &crate::body::sig::Signature,
    src: &Source,
    lts: &Lifetimes<'_>,
    static_names: &[String],
) -> bool {
    lts.names.iter().any(|&name| {
        sig.generics
            .iter()
            .any(|g| g.kind == crate::body::sig::GenericKind::Type && src.text(g.name) == name)
            && !static_names.iter().any(|s| s == name)
    })
}

fn param_ty_is_static(
    sig: &crate::body::sig::Signature,
    src: &Source,
    lts: &Lifetimes<'_>,
    impl_has_lt: bool,
    static_names: &[String],
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
    !has_unbounded_type_param(sig, src, lts, static_names)
}

pub(in super::super) fn ast_ty_may_carry_elided_lt(
    body: &Body,
    src: &Source,
    ty_id: TyId,
    outer_params: &[&str],
) -> bool {
    match *body.ty(ty_id) {
        AstTy::Never | AstTy::Ptr { .. } => false,
        AstTy::Paren(t) | AstTy::Slice(t) | AstTy::Array(t, _) | AstTy::Pat(t, _) => {
            ast_ty_may_carry_elided_lt(body, src, t, outer_params)
        }
        AstTy::Tuple(list) => body
            .list(list)
            .iter()
            .any(|&t| ast_ty_may_carry_elided_lt(body, src, TyId(t), outer_params)),
        AstTy::Ref {
            lifetime: Some(lt),
            ty,
            ..
        } if src.text(lt) == "'static" => ast_ty_may_carry_elided_lt(body, src, ty, outer_params),
        AstTy::Path(path_id) => {
            let p = body.path(path_id);
            let segs = body.list(p.segments);
            if p.qself.is_some() || segs.len() != 1 {
                return true;
            }
            let seg = body.segments[segs[0] as usize];
            let name = src.text(seg.name);
            if seg.args.is_none() && (is_prim_name(name) || outer_params.contains(&name)) {
                return false;
            }
            if is_owned_container(name)
                && let Some(Args::Angle(args)) = seg.args
            {
                return body
                    .list(args)
                    .iter()
                    .any(|&a| match body.args[a as usize] {
                        GenericArg::Ty(t) | GenericArg::Binding { ty: t, .. } => {
                            ast_ty_may_carry_elided_lt(body, src, t, outer_params)
                        }
                        GenericArg::Lifetime(lt) => src.text(lt) != "'static",
                        GenericArg::Const(_) => false,
                        GenericArg::Constraint { .. } => true,
                    });
            }
            true
        }
        _ => true,
    }
}

pub(in super::super) fn sig_type_params<'s>(
    sig: &crate::body::sig::Signature,
    src: &'s Source,
) -> Vec<&'s str> {
    sig.generics
        .iter()
        .filter(|g| g.kind == crate::body::sig::GenericKind::Type)
        .map(|g| src.text(g.name))
        .collect()
}

pub(in super::super) fn sig_returns_only_static(
    sig: &crate::body::sig::Signature,
    src: &Source,
    ret_ast: TyId,
    lts: &Lifetimes,
) -> bool {
    if sig.is_async || lts.has_elided || !lts.named.is_empty() || !lts.saw_static {
        return false;
    }
    let mut peeled = ret_ast;
    while let AstTy::Paren(inner) = *sig.body.ty(peeled) {
        peeled = inner;
    }
    if let AstTy::Fn { params, ret, .. } = *sig.body.ty(peeled) {
        let mut ret_only = Lifetimes::default();
        if let Some(r) = ret {
            collect_ty_lifetimes(&sig.body, src, r, &mut ret_only);
        }
        if !ret_only.saw_static || !sig.body.list(params).is_empty() {
            return false;
        }
    }
    let type_params = sig_type_params(sig, src);
    !ast_ty_has_type_param(&sig.body, src, ret_ast, &type_params)
}

pub(in super::super) fn sig_return_named_lifetimes<'s>(
    sig: &crate::body::sig::Signature,
    _src: &'s Source,
    _ret_ast: TyId,
    lts: &Lifetimes<'s>,
) -> Vec<&'s str> {
    if sig.is_async || lts.has_elided || lts.named.is_empty() {
        return Vec::new();
    }
    lts.named.clone()
}

fn path_args_have_type_param(
    body: &Body,
    src: &Source,
    path_id: crate::body::PathId,
    type_params: &[&str],
) -> bool {
    let p = body.path(path_id);
    if p.qself
        .is_some_and(|(q, _)| ast_ty_has_type_param(body, src, q, type_params))
    {
        return true;
    }
    body.list(p.segments)
        .iter()
        .any(|&s| match body.segments[s as usize].args {
            Some(Args::Angle(args)) => {
                body.list(args)
                    .iter()
                    .any(|&a| match body.args[a as usize] {
                        GenericArg::Ty(t) | GenericArg::Binding { ty: t, .. } => {
                            ast_ty_has_type_param(body, src, t, type_params)
                        }
                        _ => false,
                    })
            }
            Some(Args::Paren { inputs, output }) => {
                body.list(inputs)
                    .iter()
                    .any(|&t| ast_ty_has_type_param(body, src, TyId(t), type_params))
                    || output.is_some_and(|o| ast_ty_has_type_param(body, src, o, type_params))
            }
            _ => false,
        })
}

pub(in super::super) fn ast_ty_has_type_param(
    body: &Body,
    src: &Source,
    ty_id: TyId,
    type_params: &[&str],
) -> bool {
    match *body.ty(ty_id) {
        AstTy::Ref { ty, .. }
        | AstTy::Ptr { ty, .. }
        | AstTy::Slice(ty)
        | AstTy::Array(ty, _)
        | AstTy::Paren(ty)
        | AstTy::Pat(ty, _) => ast_ty_has_type_param(body, src, ty, type_params),
        AstTy::Tuple(list) => body
            .list(list)
            .iter()
            .any(|&t| ast_ty_has_type_param(body, src, TyId(t), type_params)),
        AstTy::Path(path_id) => {
            let p = body.path(path_id);
            let segs = body.list(p.segments);
            if p.qself.is_none() && segs.len() == 1 {
                let name = src.text(body.segments[segs[0] as usize].name);
                if name == "Self" || type_params.contains(&name) {
                    return true;
                }
            }
            path_args_have_type_param(body, src, path_id, type_params)
        }
        AstTy::Fn { params, ret, .. } => {
            body.list(params)
                .iter()
                .any(|&t| ast_ty_has_type_param(body, src, TyId(t), type_params))
                || ret.is_some_and(|r| ast_ty_has_type_param(body, src, r, type_params))
        }
        AstTy::ImplTrait(_, list) | AstTy::DynTrait(list) => {
            let has_static = body.list(list).iter().any(|&b_id| {
                matches!(body.bounds[b_id as usize], Bound::Lifetime(lt) if src.text(lt) == LT_STATIC)
            });
            !has_static
                || body.list(list).iter().any(|&b_id| {
                    matches!(body.bounds[b_id as usize], Bound::Trait { path, .. }
                        if path_args_have_type_param(body, src, path, type_params))
                })
        }
        AstTy::Infer => true,
        _ => false,
    }
}

pub(in super::super) fn for_each_param_bound_list(
    sig: &crate::body::sig::Signature,
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
                for g in &sig.generics {
                    if g.kind == crate::body::sig::GenericKind::Type && src.text(g.name) == name {
                        visit(g.bounds);
                    }
                }
                for pred in &sig.predicates {
                    if pred_matches_name(sig, src, pred, name) {
                        visit(pred.bounds);
                    }
                }
            }
        }
        _ => {}
    }
}

pub(in super::super) fn callable_param_inputs(
    sig: &crate::body::sig::Signature,
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

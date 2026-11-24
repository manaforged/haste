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

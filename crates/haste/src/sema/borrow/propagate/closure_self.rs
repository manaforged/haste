use crate::body::{Args, Body, Bound, ExprId, GenericArg, Ty as AstTy, TyId};
use crate::outline::source::Source;

use super::super::Checker;

pub(super) fn extend_ret_lts_from_self_ty<'a>(
    body: &Body,
    src: &'a Source,
    self_ty: TyId,
    ret_lts: &mut super::super::calls::Lifetimes<'a>,
) {
    let mut self_lts = super::super::calls::Lifetimes::default();
    super::super::calls::collect_ty_lifetimes(body, src, self_ty, &mut self_lts);
    for n in self_lts.named {
        if !ret_lts.named.contains(&n) {
            ret_lts.named.push(n);
        }
    }
    for n in self_lts.names {
        if !ret_lts.names.contains(&n) {
            ret_lts.names.push(n);
        }
    }
    ret_lts.saw_static |= self_lts.saw_static;
}

fn self_ref_status(
    body: &Body,
    src: &Source,
    ty: TyId,
    behind_ref: bool,
    behind_named: bool,
) -> (bool, bool) {
    let merge = |(a0, a1): (bool, bool), (b0, b1): (bool, bool)| (a0 || b0, a1 || b1);
    let check_args = |args: Option<Args>, in_ref: bool, in_named: bool| -> (bool, bool) {
        match args {
            Some(Args::Angle(list)) => body.list(list).iter().fold((false, false), |acc, &arg| {
                let sub = match body.args[arg as usize] {
                    GenericArg::Ty(t) | GenericArg::Binding { ty: t, .. } => {
                        self_ref_status(body, src, t, in_ref, in_named)
                    }
                    _ => (false, false),
                };
                (acc.0 || sub.0, acc.1 || sub.1)
            }),
            Some(Args::Paren { inputs, output }) => {
                let mut acc = body.list(inputs).iter().fold((false, false), |acc, &t| {
                    let sub = self_ref_status(body, src, TyId(t), false, false);
                    (acc.0 || sub.0, acc.1 || sub.1)
                });
                if let Some(o) = output {
                    let sub = self_ref_status(body, src, o, false, false);
                    acc = (acc.0 || sub.0, acc.1 || sub.1);
                }
                acc
            }
            _ => (false, false),
        }
    };
    match *body.ty(ty) {
        AstTy::Ref {
            lifetime,
            ty: inner,
            ..
        } => {
            let next_named = behind_named
                || lifetime
                    .is_some_and(|lt| super::super::calls::lifetimes::is_named_lt(src.text(lt)));
            self_ref_status(body, src, inner, true, next_named)
        }
        AstTy::Path(path) => {
            let segs = body.list(body.path(path).segments);
            let mut acc = if segs
                .first()
                .is_some_and(|&s| src.text(body.segments[s as usize].name) == "Self")
            {
                (behind_ref, !behind_named)
            } else {
                (false, false)
            };
            for &s in segs {
                acc = merge(
                    acc,
                    check_args(body.segments[s as usize].args, behind_ref, behind_named),
                );
            }
            acc
        }
        AstTy::Tuple(list) => body.list(list).iter().fold((false, false), |acc, &t| {
            let sub = self_ref_status(body, src, TyId(t), behind_ref, behind_named);
            (acc.0 || sub.0, acc.1 || sub.1)
        }),
        AstTy::Slice(t) | AstTy::Array(t, _) | AstTy::Paren(t) | AstTy::Pat(t, _) => {
            self_ref_status(body, src, t, behind_ref, behind_named)
        }
        AstTy::Ptr { ty: t, .. } => self_ref_status(body, src, t, false, false),
        AstTy::Fn { params, ret, .. } => {
            let mut acc = body.list(params).iter().fold((false, false), |acc, &t| {
                let sub = self_ref_status(body, src, TyId(t), false, false);
                (acc.0 || sub.0, acc.1 || sub.1)
            });
            if let Some(r) = ret {
                acc = merge(acc, self_ref_status(body, src, r, false, false));
            }
            acc
        }
        AstTy::ImplTrait(_, list) | AstTy::DynTrait(list) => {
            body.list(list).iter().fold((false, false), |mut acc, &b| {
                if let Bound::Trait { path, .. } = body.bounds[b as usize] {
                    for &s in body.list(body.path(path).segments) {
                        acc = merge(
                            acc,
                            check_args(body.segments[s as usize].args, false, false),
                        );
                    }
                }
                acc
            })
        }
        _ => (false, false),
    }
}

pub(in super::super) fn all_self_behind_named_ref(body: &Body, src: &Source, ty: TyId) -> bool {
    !self_ref_status(body, src, ty, false, false).1
}

pub(in super::super) fn ast_ty_refs_self(body: &Body, src: &Source, ty: TyId) -> bool {
    self_ref_status(body, src, ty, false, false).0
}

pub(in super::super) fn ast_ty_contains_impl_trait(body: &Body, ty: TyId) -> bool {
    let args_contain = |args: Option<Args>| match args {
        Some(Args::Angle(list)) => {
            body.list(list)
                .iter()
                .any(|&arg| match body.args[arg as usize] {
                    GenericArg::Ty(t) | GenericArg::Binding { ty: t, .. } => {
                        ast_ty_contains_impl_trait(body, t)
                    }
                    _ => false,
                })
        }
        Some(Args::Paren { inputs, output }) => {
            body.list(inputs)
                .iter()
                .any(|&t| ast_ty_contains_impl_trait(body, TyId(t)))
                || output.is_some_and(|o| ast_ty_contains_impl_trait(body, o))
        }
        _ => false,
    };
    match *body.ty(ty) {
        AstTy::ImplTrait(..) => true,
        AstTy::Ref { ty: inner, .. }
        | AstTy::Ptr { ty: inner, .. }
        | AstTy::Slice(inner)
        | AstTy::Array(inner, _)
        | AstTy::Paren(inner)
        | AstTy::Pat(inner, _) => ast_ty_contains_impl_trait(body, inner),
        AstTy::Tuple(list) => body
            .list(list)
            .iter()
            .any(|&t| ast_ty_contains_impl_trait(body, TyId(t))),
        AstTy::Path(path) => body
            .list(body.path(path).segments)
            .iter()
            .any(|&s| args_contain(body.segments[s as usize].args)),
        AstTy::Fn { params, ret, .. } => {
            body.list(params)
                .iter()
                .any(|&t| ast_ty_contains_impl_trait(body, TyId(t)))
                || ret.is_some_and(|r| ast_ty_contains_impl_trait(body, r))
        }
        AstTy::DynTrait(list) => body.list(list).iter().any(|&b| {
            matches!(
                body.bounds[b as usize],
                Bound::Trait { path, .. }
                    if body
                        .list(body.path(path).segments)
                        .iter()
                        .any(|&s| args_contain(body.segments[s as usize].args))
            )
        }),
        _ => false,
    }
}

impl Checker<'_, '_> {
    pub(in super::super) fn sig_captures_edition_2024_rpit(
        &mut self,
        fn_item: crate::names::ItemRef,
        sig: &crate::body::sig::Signature,
        sig_src: &Source,
        ret_ast: TyId,
    ) -> bool {
        if sig.is_async
            || !self
                .cx
                .crate_edition(fn_item.krate)
                .is_some_and(|e| e >= "2024")
            || !ast_ty_contains_impl_trait(&sig.body, ret_ast)
        {
            return false;
        }
        let mut lts = super::super::calls::Lifetimes::default();
        super::super::calls::collect_ty_lifetimes(&sig.body, sig_src, ret_ast, &mut lts);
        !super::super::calls::sig_returns_only_static(sig, sig_src, ret_ast, &lts)
            && !self.sig_ret_ast_is_provably_static(fn_item, sig, sig_src, ret_ast)
    }

    pub(in super::super) fn expr_is_capturing_rpit(&mut self, expr: ExprId) -> bool {
        let cur = self.unwrap_parens_and_empty_blocks(expr);
        let fn_item = match *self.body.expr(cur) {
            crate::body::Expr::Method { .. } => match self.typed.res.get(&cur).copied() {
                Some(crate::sema::infer::Res::Fn(it, _)) => Some(it),
                _ => None,
            },
            crate::body::Expr::Call(callee, _) => self.call_sig_fn_item(cur, callee),
            _ => None,
        };
        let Some(fn_item) = fn_item else {
            return false;
        };
        let Ok(sig) = self.cx.signature(fn_item) else {
            return false;
        };
        let Some(ret_ast) = sig.ret else {
            return false;
        };
        let sig_src = self.cx.item(fn_item).0;
        self.sig_captures_edition_2024_rpit(fn_item, &sig, sig_src, ret_ast)
    }
}

fn path_lifetime_args(body: &Body, src: &Source, ty: TyId) -> Option<(Vec<u32>, Vec<TyId>)> {
    let AstTy::Path(path) = *body.ty(ty) else {
        return None;
    };
    let &last = body.list(body.path(path).segments).last()?;
    let crate::body::Args::Angle(list) = body.segments[last as usize].args? else {
        return None;
    };
    let mut lts = Vec::new();
    let mut tys = Vec::new();
    for &arg in body.list(list) {
        match body.args[arg as usize] {
            GenericArg::Lifetime(tok) if src.text(tok) != "'static" => lts.push(tok),
            GenericArg::Ty(inner) => tys.push(inner),
            _ => {}
        }
    }
    Some((lts, tys))
}

fn recv_lifetime_indices(
    sig: &crate::body::sig::Signature,
    sig_src: &Source,
    ret_ast: TyId,
) -> Option<(usize, Vec<usize>)> {
    let recv_lt = sig.receiver_lifetime.map_or("'_", |tok| sig_src.text(tok));
    let (lts, tys) = path_lifetime_args(&sig.body, sig_src, ret_ast)?;
    let indices: Vec<usize> = lts
        .iter()
        .enumerate()
        .filter(|&(_, &tok)| sig_src.text(tok) == recv_lt)
        .map(|(i, _)| i)
        .collect();
    if indices.is_empty() {
        return None;
    }
    for ty in tys {
        let mut inner_lts = super::super::calls::Lifetimes::default();
        super::super::calls::collect_ty_lifetimes(&sig.body, sig_src, ty, &mut inner_lts);
        if (recv_lt == "'_" && inner_lts.has_elided) || inner_lts.named.contains(&recv_lt) {
            return None;
        }
    }
    Some((lts.len(), indices))
}

impl Checker<'_, '_> {
    fn iter_item_independent_of_lts(
        &mut self,
        adt: crate::names::ItemRef,
        total_lts: usize,
        recv_indices: &[usize],
    ) -> bool {
        let impl_count = self.cx.outline.crates[adt.krate as usize].impls.len();
        for index in 0..impl_count {
            let block = crate::sema::impls::ImplRef::Module {
                krate: adt.krate,
                index: index as u32,
            };
            let Some(tr) = self.cx.impl_trait(block) else {
                continue;
            };
            if self.cx.plain_name(tr) != "Iterator" || !self.cx.outline.is_sysroot(tr.krate) {
                continue;
            }
            if self.cx.impl_self_adt(block) != Some(adt) {
                continue;
            }
            let Ok(header) = self.cx.impl_header(block) else {
                continue;
            };
            let impl_src = &self.cx.impl_file(block).source;
            let Some((impl_lts, _)) = path_lifetime_args(&header.body, impl_src, header.self_ty)
                .filter(|(lts, _)| lts.len() == total_lts)
            else {
                continue;
            };
            let Some(item_member) = self.cx.member(block, "Item") else {
                continue;
            };
            let Ok(alias) = self.cx.alias_def(item_member) else {
                continue;
            };
            let mut item_lts = super::super::calls::Lifetimes::default();
            super::super::calls::collect_ty_lifetimes(
                &alias.arena,
                self.cx.item(item_member).0,
                alias.ty,
                &mut item_lts,
            );
            return !item_lts.has_elided
                && !item_lts.names.contains(&"Self")
                && !recv_indices.iter().any(|&i| {
                    let name = impl_src.text(impl_lts[i]);
                    name == "'_" || item_lts.named.contains(&name)
                });
        }
        false
    }

    pub(super) fn mark_iter_env_only_recv(
        &mut self,
        call_expr: ExprId,
        recv: ExprId,
        (sig, sig_src): (&crate::body::sig::Signature, &Source),
        mut srcs: Vec<super::super::BorrowSource>,
    ) -> (Vec<super::super::BorrowSource>, bool) {
        let Some(ret_ast) = sig.ret else {
            return (srcs, false);
        };
        if !matches!(
            sig.receiver,
            Some((
                crate::body::sig::SelfKind::Ref | crate::body::sig::SelfKind::RefMut,
                _
            ))
        ) {
            return (srcs, false);
        }
        let crate::sema::ty::Kind::Adt(adt, _) = *self.cx.types.kind(self.expr_ty(call_expr))
        else {
            return (srcs, false);
        };
        let Some((total_lts, indices)) = recv_lifetime_indices(sig, sig_src, ret_ast) else {
            return (srcs, false);
        };
        if !self.iter_item_independent_of_lts(adt, total_lts, &indices) {
            return (srcs, false);
        }
        let direct_local = self
            .root_place(recv)
            .filter(|p| !p.through_ref)
            .map(|p| p.local);
        let mut marked = false;
        for s in &mut srcs {
            if super::super::checks::is_temp_target(s.target).is_some()
                || (direct_local == Some(s.target) && !self.is_non_static_param(s.target))
            {
                s.env_only = true;
                marked = true;
            }
        }
        (srcs, marked)
    }
}

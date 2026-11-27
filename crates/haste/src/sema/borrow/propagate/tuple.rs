use crate::body::{Args, Body, Expr, ExprId, GenericArg, Ty as AstTy, TyId};
use crate::lower::is_freeze_ty;
use crate::names::plain;
use crate::outline::source::Source;
use crate::sema::infer::Res;
use crate::sema::ty::{Kind, Ty};

use super::super::{Checker, State};
use super::sig::SigCall;
use super::{BorrowSource, dedup_sources};

#[derive(Clone, Debug)]
pub(in super::super) enum TupleSources {
    Leaf {
        sources: Vec<BorrowSource>,
        inner_ref: Option<Vec<BorrowSource>>,
    },
    Tuple(Vec<TupleSources>),
}

impl TupleSources {
    pub(in super::super) fn merge(&mut self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Leaf {
                    sources: a,
                    inner_ref: ar,
                },
                Self::Leaf {
                    sources: b,
                    inner_ref: br,
                },
            ) => {
                a.extend(b.iter().cloned());
                *a = dedup_sources(std::mem::take(a));
                *ar = match (ar.take(), br) {
                    (Some(mut x), Some(y)) => {
                        x.extend(y.iter().cloned());
                        Some(dedup_sources(x))
                    }
                    _ => None,
                };
                true
            }
            (Self::Tuple(a), Self::Tuple(b)) if a.len() == b.len() => {
                a.iter_mut().zip(b).all(|(x, y)| x.merge(y))
            }
            _ => false,
        }
    }

    pub(in super::super) fn flatten_into(&self, out: &mut Vec<BorrowSource>) {
        match self {
            Self::Leaf { sources, .. } => out.extend(sources.iter().cloned()),
            Self::Tuple(elems) => {
                for elem in elems {
                    elem.flatten_into(out);
                }
            }
        }
    }

    pub(in super::super) fn sources_for(&self, outer_hrtb: bool) -> (Vec<BorrowSource>, bool) {
        if outer_hrtb
            && let Self::Leaf {
                inner_ref: Some(inner),
                ..
            } = self
        {
            return (inner.clone(), false);
        }
        let mut flat = Vec::new();
        self.flatten_into(&mut flat);
        (flat, outer_hrtb)
    }
}

impl<'a> Checker<'_, 'a> {
    pub(in super::super) fn expr_tuple_sources(
        &mut self,
        expr: ExprId,
        state: &State,
        use_let_extend: bool,
    ) -> Option<TupleSources> {
        let cur = self.unwrap_parens(expr);
        match *self.body.expr(cur) {
            Expr::Path(_) => {
                return self
                    .expr_plain_local(cur)
                    .and_then(|l| state.tuple_elem_sources.get(&l).cloned());
            }
            Expr::Call(..) | Expr::Method { .. } | Expr::Try(_) => {
                return self.sig_tuple_sources(cur, state);
            }
            Expr::TupleField(base, idx) => {
                return match self.expr_tuple_sources(base, state, use_let_extend) {
                    Some(TupleSources::Tuple(elems)) => elems.into_iter().nth(idx as usize),
                    _ => None,
                };
            }
            _ => {}
        }
        let ty = self.expr_ty(cur);
        if ty == Ty::NONE || !is_freeze_ty(self.cx, ty, 0) {
            return None;
        }
        match *self.body.expr(cur) {
            Expr::Tuple(list) => {
                let items: Vec<u32> = self.body.list(list).to_vec();
                let elems = items
                    .into_iter()
                    .map(|id| {
                        let elem = ExprId(id);
                        self.expr_tuple_sources(elem, state, use_let_extend)
                            .unwrap_or_else(|| {
                                let sources = if use_let_extend {
                                    self.expr_borrow_sources_let(elem, state)
                                } else {
                                    self.expr_borrow_sources(elem, state)
                                };
                                let inner_ref = match *self.body.expr(self.unwrap_parens(elem)) {
                                    Expr::Ref {
                                        raw: false,
                                        expr: inner,
                                        ..
                                    } if is_freeze_ty(self.cx, self.expr_ty(inner), 0) => {
                                        Some(self.expr_borrow_sources(inner, state))
                                    }
                                    _ => None,
                                };
                                TupleSources::Leaf { sources, inner_ref }
                            })
                    })
                    .collect();
                Some(TupleSources::Tuple(elems))
            }
            _ => None,
        }
    }

    pub(super) fn tuple_field_sources(
        &mut self,
        base: ExprId,
        index: u32,
        state: &State,
    ) -> Option<Vec<BorrowSource>> {
        let TupleSources::Tuple(elems) = self.expr_tuple_sources(base, state, false)? else {
            return None;
        };
        let elem = elems.get(index as usize)?;
        let mut out = Vec::new();
        elem.flatten_into(&mut out);
        Some(dedup_sources(out))
    }

    fn sig_tuple_sources(&mut self, expr: ExprId, state: &State) -> Option<TupleSources> {
        let mut cur = self.unwrap_parens(expr);
        let peeled_try = if let Expr::Try(inner) = *self.body.expr(cur) {
            cur = self.unwrap_parens(inner);
            true
        } else {
            false
        };
        let (fn_item, receiver, args) = match *self.body.expr(cur) {
            Expr::Call(callee, args) => (self.call_sig_fn_item(cur, callee)?, None, args),
            Expr::Method {
                receiver, params, ..
            } => {
                let Some(Res::Fn(fn_item, _)) = self.typed.res.get(&cur).copied() else {
                    return None;
                };
                if self.body.list(params).iter().any(|&id| {
                    matches!(*self.body.expr(ExprId(id)), Expr::Closure { body, .. } if self.is_raw_ptr_deref(body))
                }) {
                    return None;
                }
                (fn_item, Some(receiver), params)
            }
            _ => return None,
        };
        let sig = self.cx.signature(fn_item).ok()?;
        if sig.is_async {
            return None;
        }
        let (sig_src, sig_item) = self.cx.item(fn_item);
        let mut ret_ast = sig.ret?;
        if peeled_try {
            ret_ast = peel_try_ret_ast(self.cx, fn_item, &sig.body, sig_src, ret_ast)?;
        }
        ret_ast = unwrap_ast_parens(&sig.body, ret_ast);
        if !matches!(*sig.body.ty(ret_ast), AstTy::Tuple(_)) {
            return None;
        }
        let (receiver, arg_ids) = self.normalize_sig_args(receiver, args, &sig);
        let call = SigCall {
            expr: cur,
            fn_item,
            sig: &sig,
            sig_src,
            sig_item,
            receiver,
            arg_ids: &arg_ids,
        };
        Some(self.sig_ast_tuple_sources(&call, ret_ast, self.expr_ty(expr), state))
    }

    fn sig_ast_tuple_sources(
        &mut self,
        call: &SigCall<'a, '_>,
        ret_ast: TyId,
        mut ret_ty: Ty,
        state: &State,
    ) -> TupleSources {
        let ret_ast = unwrap_ast_parens(&call.sig.body, ret_ast);
        while ret_ty != Ty::NONE
            && let Kind::Pat(inner, ..) = *self.cx.types.kind(ret_ty)
        {
            ret_ty = inner;
        }
        if let AstTy::Tuple(list) = *call.sig.body.ty(ret_ast) {
            let ast_elems: Vec<u32> = call.sig.body.list(list).to_vec();
            let elem_tys: Vec<Ty> = if ret_ty != Ty::NONE
                && let Kind::Tuple(substs) = *self.cx.types.kind(ret_ty)
                && self.cx.types.substs(substs).len() == ast_elems.len()
            {
                self.cx.types.substs(substs).to_vec()
            } else {
                Vec::new()
            };
            let elems = ast_elems
                .into_iter()
                .enumerate()
                .map(|(i, id)| {
                    self.sig_ast_tuple_sources(
                        call,
                        TyId(id),
                        elem_tys.get(i).copied().unwrap_or(Ty::NONE),
                        state,
                    )
                })
                .collect();
            return TupleSources::Tuple(elems);
        }
        let sources = self.sig_sources_for_ret(call, ret_ast, ret_ty, state);
        TupleSources::Leaf {
            sources,
            inner_ref: None,
        }
    }

    pub(in super::super) fn for_iter_item_sources(
        &mut self,
        iter: ExprId,
        iter_sources: &[BorrowSource],
        state: &State,
    ) -> (Vec<BorrowSource>, Option<TupleSources>) {
        let item_sources = if self.for_iter_item_is_static(iter) {
            Vec::new()
        } else {
            let srcs: Vec<BorrowSource> = iter_sources
                .iter()
                .filter(|s| !s.env_only)
                .cloned()
                .collect();
            if self.is_iter_copied_call(iter) {
                let iter_ty = self.expr_ty(iter);
                self.unwrap_outer_container_loans_with_res(srcs, state, true, iter_ty)
            } else {
                srcs
            }
        };
        let item_tuple = self.iter_tuple_sources(iter, state);
        (item_sources, item_tuple)
    }

    fn is_iter_copied_call(&mut self, expr: ExprId) -> bool {
        let cur = self.unwrap_parens(expr);
        let Expr::Method { params, .. } = *self.body.expr(cur) else {
            return false;
        };
        if !self.body.list(params).is_empty() {
            return false;
        }
        let Some(Res::Fn(fn_item, _)) = self.typed.res.get(&cur).copied() else {
            return false;
        };
        self.is_iter_mod(fn_item) && matches!(self.cx.name(fn_item), "copied" | "cloned")
    }

    fn unwrap_tuple_sources_containers(&mut self, ts: TupleSources, state: &State) -> TupleSources {
        match ts {
            TupleSources::Leaf { sources, inner_ref } => TupleSources::Leaf {
                sources: self.unwrap_outer_container_loans(sources, state, true),
                inner_ref,
            },
            TupleSources::Tuple(elems) => TupleSources::Tuple(
                elems
                    .into_iter()
                    .map(|e| self.unwrap_tuple_sources_containers(e, state))
                    .collect(),
            ),
        }
    }

    fn leaf_iter_sources(&mut self, expr: ExprId, state: &State) -> TupleSources {
        let sources = if self.for_iter_item_is_static(expr) {
            Vec::new()
        } else {
            self.expr_borrow_sources(expr, state)
                .into_iter()
                .filter(|s| !s.env_only)
                .collect()
        };
        TupleSources::Leaf {
            sources,
            inner_ref: None,
        }
    }

    fn iter_tuple_sources(&mut self, expr: ExprId, state: &State) -> Option<TupleSources> {
        let cur = self.unwrap_parens(expr);
        let Expr::Method {
            receiver, params, ..
        } = *self.body.expr(cur)
        else {
            return None;
        };
        let Some(Res::Fn(fn_item, _)) = self.typed.res.get(&cur).copied() else {
            return None;
        };
        let name = self.cx.name(fn_item);
        let in_iter = self.is_iter_mod(fn_item);
        let param_ids: Vec<u32> = self.body.list(params).to_vec();
        if in_iter && name == "enumerate" && param_ids.is_empty() {
            let inner = self
                .iter_tuple_sources(receiver, state)
                .unwrap_or_else(|| self.leaf_iter_sources(receiver, state));
            return Some(TupleSources::Tuple(vec![
                TupleSources::Leaf {
                    sources: Vec::new(),
                    inner_ref: None,
                },
                inner,
            ]));
        }
        if ((in_iter && name == "zip") || name == "zip_eq") && param_ids.len() == 1 {
            let left = self
                .iter_tuple_sources(receiver, state)
                .unwrap_or_else(|| self.leaf_iter_sources(receiver, state));
            let arg = ExprId(param_ids[0]);
            let right = self
                .iter_tuple_sources(arg, state)
                .unwrap_or_else(|| self.leaf_iter_sources(arg, state));
            return Some(TupleSources::Tuple(vec![left, right]));
        }
        if in_iter && matches!(name, "copied" | "cloned") && param_ids.is_empty() {
            let inner = self
                .iter_tuple_sources(receiver, state)
                .unwrap_or_else(|| self.leaf_iter_sources(receiver, state));
            return Some(self.unwrap_tuple_sources_containers(inner, state));
        }
        if in_iter
            && matches!(
                name,
                "skip"
                    | "take"
                    | "step_by"
                    | "rev"
                    | "peekable"
                    | "fuse"
                    | "filter"
                    | "inspect"
                    | "take_while"
                    | "skip_while"
            )
        {
            return self.iter_tuple_sources(receiver, state);
        }
        None
    }
}

fn unwrap_ast_parens(body: &Body, mut ty: TyId) -> TyId {
    while let AstTy::Paren(inner) = *body.ty(ty) {
        ty = inner;
    }
    ty
}

fn peel_try_ret_ast(
    cx: &crate::sema::Context<'_>,
    fn_item: crate::names::ItemRef,
    body: &Body,
    src: &Source,
    ty: TyId,
) -> Option<TyId> {
    let ty = unwrap_ast_parens(body, ty);
    let AstTy::Path(path_id) = *body.ty(ty) else {
        return None;
    };
    let path = body.path(path_id);
    if path.qself.is_some() {
        return None;
    }
    let segs = body.list(path.segments);
    let &seg_id = segs.last()?;
    let seg = body.segments[seg_id as usize];
    let name = plain(src.text(seg.name));
    if !matches!(name, "Result" | "Option") {
        return None;
    }
    let Some(Args::Angle(arg_list)) = seg.args else {
        return None;
    };
    let mut ty_args = body
        .list(arg_list)
        .iter()
        .filter_map(|&id| match body.args[id as usize] {
            GenericArg::Ty(t) => Some(t),
            _ => None,
        });
    let first = ty_args.next()?;
    let has_second = ty_args.next().is_some();
    if has_second {
        let std_path = match segs.len() {
            1 => {
                let file = cx.file(fn_item);
                let shadowed_type = file.items.items.iter().any(|it| {
                    it.kind == crate::outline::item::ItemKind::Type
                        && it.name.is_some_and(|n| plain(file.source.text(n)) == name)
                });
                let shadowed_import = file.items.imports.iter().any(|imp| {
                    let imported = match imp.leaf {
                        crate::outline::item::Leaf::Name(Some(a)) => Some(a),
                        crate::outline::item::Leaf::Name(None) => imp.path.last().copied(),
                        _ => None,
                    };
                    if !imported.is_some_and(|t| plain(file.source.text(t)) == name) {
                        return false;
                    }
                    let root = imp.path.first().map(|&t| plain(file.source.text(t)));
                    !matches!(root, Some("core" | "std"))
                });
                !shadowed_type && !shadowed_import
            }
            3 => {
                let s0 = plain(src.text(body.segments[segs[0] as usize].name));
                let s1 = plain(src.text(body.segments[segs[1] as usize].name));
                matches!(s0, "core" | "std") && matches!(s1, "result" | "option")
            }
            _ => false,
        };
        if !std_path {
            return None;
        }
    }
    Some(first)
}

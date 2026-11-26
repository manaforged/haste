use rustc_hash::FxHashSet as HashSet;

use crate::body::sig::GenericKind;
use crate::body::{Args, Body, Bound, GenericArg, PathId, Ty as AstTy, TyId};
use crate::names::{ItemRef, plain};
use crate::outline::source::Source;
use crate::sema::Context;
use crate::sema::ty::{Kind, Ty};

use super::super::calls::lifetimes::{
    LT_ELIDED, LT_STATIC, Lifetimes, collect_ty_lifetimes, is_owned_container,
};

#[derive(Clone, Default)]
pub(in super::super) struct InvariantLifetimes {
    pub(in super::super) saw_static: bool,
    pub(in super::super) has_elided: bool,
    pub(in super::super) named: Vec<String>,
}

impl InvariantLifetimes {
    pub(in super::super) fn is_empty(&self) -> bool {
        !self.saw_static && !self.has_elided && self.named.is_empty()
    }

    pub(in super::super) fn is_only_static(&self) -> bool {
        self.saw_static && !self.has_elided && self.named.is_empty()
    }

    pub(in super::super) fn push_lt(&mut self, lt: &str) {
        if lt == LT_STATIC {
            self.saw_static = true;
        } else if lt == LT_ELIDED {
            self.has_elided = true;
        } else if !self.named.iter().any(|s| s == lt) {
            self.named.push(lt.to_owned());
        }
    }

    fn merge_all(&mut self, lts: &Lifetimes<'_>) {
        self.saw_static |= lts.saw_static;
        self.has_elided |= lts.has_elided;
        for &lt in &lts.named {
            self.push_lt(lt);
        }
    }
}

pub(in super::super) fn is_interior_mut_name(name: &str) -> bool {
    matches!(
        name,
        "Cell"
            | "UnsafeCell"
            | "RefCell"
            | "OnceCell"
            | "OnceLock"
            | "Mutex"
            | "RwLock"
            | "SyncUnsafeCell"
            | "UnsafePinned"
    )
}

struct InvWalk<'a, 'b> {
    cx: &'b Context<'a>,
    body: &'b Body,
    src: &'b Source,
    seen_adts: &'b mut HashSet<ItemRef>,
    out: &'b mut InvariantLifetimes,
    inv_names: Option<&'b mut Vec<String>>,
}

impl InvWalk<'_, '_> {
    fn push_name(&mut self, name: &str) {
        if let Some(names) = self.inv_names.as_deref_mut()
            && !names.iter().any(|n| n == name)
        {
            names.push(name.to_owned());
        }
    }
}

pub(in super::super) fn collect_ast_invariant_lifetimes(
    cx: &Context<'_>,
    body: &Body,
    src: &Source,
    ty_id: TyId,
    hint_ty: Option<Ty>,
) -> InvariantLifetimes {
    collect_ast_invariant_lifetimes_mode(cx, body, src, ty_id, hint_ty, false)
}

pub(in super::super) fn collect_ast_invariant_lifetimes_mode(
    cx: &Context<'_>,
    body: &Body,
    src: &Source,
    ty_id: TyId,
    hint_ty: Option<Ty>,
    initial_in_inv: bool,
) -> InvariantLifetimes {
    let mut out = InvariantLifetimes::default();
    let mut seen_adts = HashSet::default();
    let mut walk = InvWalk {
        cx,
        body,
        src,
        seen_adts: &mut seen_adts,
        out: &mut out,
        inv_names: None,
    };
    collect_ast_invariant_rec(&mut walk, ty_id, hint_ty, initial_in_inv);
    out
}

fn adt_invariant_param_mask(
    cx: &Context<'_>,
    adt_item: ItemRef,
    seen_adts: &mut HashSet<ItemRef>,
) -> Vec<(GenericKind, bool)> {
    let Some(def) = cx.store().adts.get(&adt_item) else {
        return Vec::new();
    };
    let adt_src = &cx.file(adt_item).source;
    let params: Vec<(GenericKind, &str)> = def
        .generics
        .params
        .iter()
        .filter(|g| matches!(g.kind, GenericKind::Lifetime | GenericKind::Type))
        .map(|g| (g.kind, adt_src.text(g.name)))
        .collect();
    if params.is_empty() || !seen_adts.insert(adt_item) {
        return params.into_iter().map(|(k, _)| (k, false)).collect();
    }
    let mut field_inv = InvariantLifetimes::default();
    let mut field_names = Vec::new();
    let mut walk = InvWalk {
        cx,
        body: &def.body,
        src: adt_src,
        seen_adts,
        out: &mut field_inv,
        inv_names: Some(&mut field_names),
    };
    for var in &def.variants {
        for field in &var.fields {
            collect_ast_invariant_rec(&mut walk, field.ty, None, false);
        }
    }
    seen_adts.remove(&adt_item);
    params
        .into_iter()
        .map(|(kind, name)| {
            let inv = match kind {
                GenericKind::Lifetime => field_inv.named.iter().any(|n| n == name),
                GenericKind::Type => field_names.iter().any(|n| n == name),
                _ => false,
            };
            (kind, inv)
        })
        .collect()
}

fn peel_hint_ty(cx: &Context<'_>, hint_ty: Option<Ty>) -> Option<Ty> {
    let mut ty = hint_ty.filter(|&t| t != Ty::NONE)?;
    while let Kind::Pat(inner, ..) = *cx.types.kind(ty) {
        ty = inner;
    }
    Some(ty)
}

fn collect_ast_invariant_rec(
    walk: &mut InvWalk<'_, '_>,
    ty_id: TyId,
    hint_ty: Option<Ty>,
    in_inv: bool,
) {
    let hint_ty = peel_hint_ty(walk.cx, hint_ty);
    match *walk.body.ty(ty_id) {
        AstTy::Paren(inner) | AstTy::Pat(inner, _) => {
            collect_ast_invariant_rec(walk, inner, hint_ty, in_inv);
        }
        AstTy::Slice(inner) | AstTy::Array(inner, _) => {
            let inner_hint = hint_ty.and_then(|t| match *walk.cx.types.kind(t) {
                Kind::Slice(e) | Kind::Array(e, _) => Some(e),
                _ => None,
            });
            collect_ast_invariant_rec(walk, inner, inner_hint, in_inv);
        }
        AstTy::Tuple(list) => {
            let elem_hints = hint_ty.and_then(|t| match *walk.cx.types.kind(t) {
                Kind::Tuple(substs) => Some(walk.cx.types.substs(substs)),
                _ => None,
            });
            for (idx, &t) in walk.body.list(list).iter().enumerate() {
                let eh = elem_hints.and_then(|s| s.get(idx).copied());
                collect_ast_invariant_rec(walk, TyId(t), eh, in_inv);
            }
        }
        AstTy::Ref {
            mutable,
            lifetime,
            ty: inner,
        } => {
            if in_inv {
                match lifetime {
                    None => walk.out.has_elided = true,
                    Some(lt) => walk.out.push_lt(walk.src.text(lt)),
                }
            }
            let inner_hint = hint_ty.and_then(|t| match *walk.cx.types.kind(t) {
                Kind::Ref(_, pointee) => Some(pointee),
                _ => None,
            });
            collect_ast_invariant_rec(walk, inner, inner_hint, in_inv || mutable);
        }
        AstTy::Ptr { mutable, ty: inner } => {
            let inner_hint = hint_ty.and_then(|t| match *walk.cx.types.kind(t) {
                Kind::Ptr(_, pointee) => Some(pointee),
                _ => None,
            });
            collect_ast_invariant_rec(walk, inner, inner_hint, in_inv || mutable);
        }
        AstTy::ImplTrait(_, list) | AstTy::DynTrait(list) if in_inv => {
            collect_bounds_invariant(walk, list);
        }
        AstTy::Fn { params, ret, .. } if in_inv => {
            let mut lts = Lifetimes::default();
            for &t in walk.body.list(params) {
                collect_ty_lifetimes(walk.body, walk.src, TyId(t), &mut lts);
            }
            if let Some(r) = ret {
                collect_ty_lifetimes(walk.body, walk.src, r, &mut lts);
            }
            walk.out.merge_all(&lts);
            for &n in &lts.names {
                walk.push_name(n);
            }
        }
        AstTy::Path(path_id) => {
            collect_path_invariant(walk, path_id, hint_ty, in_inv);
        }
        _ => {}
    }
}

fn collect_bounds_invariant(walk: &mut InvWalk<'_, '_>, list: crate::body::List) {
    let mut lts = Lifetimes::default();
    for &b_id in walk.body.list(list) {
        match walk.body.bounds[b_id as usize] {
            Bound::Lifetime(lt) => walk.out.push_lt(walk.src.text(lt)),
            Bound::Trait { path, .. } => {
                for &seg_id in walk.body.list(walk.body.path(path).segments) {
                    if let Some(Args::Angle(args)) = walk.body.segments[seg_id as usize].args {
                        for &a in walk.body.list(args) {
                            match walk.body.args[a as usize] {
                                GenericArg::Lifetime(lt) => walk.out.push_lt(walk.src.text(lt)),
                                GenericArg::Ty(t) | GenericArg::Binding { ty: t, .. } => {
                                    collect_ty_lifetimes(walk.body, walk.src, t, &mut lts);
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
    walk.out.merge_all(&lts);
    for &n in &lts.names {
        walk.push_name(n);
    }
}

fn mask_nth_is_inv(mask: &Option<Vec<(GenericKind, bool)>>, kind: GenericKind, idx: usize) -> bool {
    mask.as_ref().is_some_and(|m| {
        m.iter()
            .filter(|&&(k, _)| k == kind)
            .nth(idx)
            .is_some_and(|&(_, inv)| inv)
    })
}

fn collect_path_invariant(
    walk: &mut InvWalk<'_, '_>,
    path_id: PathId,
    hint_ty: Option<Ty>,
    in_inv: bool,
) {
    let p = walk.body.path(path_id);
    let segs = walk.body.list(p.segments);
    let Some(&last_id) = segs.last() else {
        return;
    };
    let last_seg = walk.body.segments[last_id as usize];
    let last_name = plain(walk.src.text(last_seg.name));
    if in_inv && p.qself.is_none() && segs.len() == 1 && last_seg.args.is_none() {
        walk.push_name(last_name);
    }
    let adt_info = hint_ty.and_then(|t| match *walk.cx.types.kind(t) {
        Kind::Adt(item, substs) => Some((item, walk.cx.types.substs(substs))),
        _ => None,
    });
    let mut arg_mode: Option<Vec<(GenericKind, bool)>> = None;
    let next_inv = if in_inv
        || is_interior_mut_name(last_name)
        || adt_info.is_some_and(|(it, _)| {
            matches!(walk.cx.lang(it), Some("unsafe_cell" | "unsafe_pinned"))
        }) {
        true
    } else if is_owned_container(last_name) {
        false
    } else if let Some((adt_item, _)) = adt_info {
        let mask = adt_invariant_param_mask(walk.cx, adt_item, walk.seen_adts);
        if mask.iter().any(|&(_, inv)| inv) {
            arg_mode = Some(mask);
        }
        false
    } else {
        false
    };
    let Some(Args::Angle(arg_list)) = last_seg.args else {
        if arg_mode
            .as_ref()
            .is_some_and(|m| m.iter().any(|&(k, inv)| inv && k == GenericKind::Lifetime))
        {
            walk.out.has_elided = true;
        }
        return;
    };
    let mut lt_idx = 0usize;
    let mut ty_idx = 0usize;
    for &arg_id in walk.body.list(arg_list) {
        match walk.body.args[arg_id as usize] {
            GenericArg::Lifetime(lt) => {
                let is_inv = next_inv || mask_nth_is_inv(&arg_mode, GenericKind::Lifetime, lt_idx);
                lt_idx += 1;
                if is_inv {
                    walk.out.push_lt(walk.src.text(lt));
                }
            }
            GenericArg::Ty(t) | GenericArg::Binding { ty: t, .. } => {
                let is_inv = next_inv || mask_nth_is_inv(&arg_mode, GenericKind::Type, ty_idx);
                let sub_hint = adt_info.and_then(|(_, substs)| substs.get(ty_idx).copied());
                ty_idx += 1;
                collect_ast_invariant_rec(walk, t, sub_hint, is_inv);
            }
            _ => {}
        }
    }
    if let Some(mask) = arg_mode
        && mask
            .iter()
            .filter(|&&(k, _)| k == GenericKind::Lifetime)
            .skip(lt_idx)
            .any(|&(_, inv)| inv)
    {
        walk.out.has_elided = true;
    }
}

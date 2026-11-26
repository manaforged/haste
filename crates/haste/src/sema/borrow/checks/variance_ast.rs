use rustc_hash::FxHashSet as HashSet;

use crate::body::sig::GenericKind;
use crate::body::{Args, Body, Bound, GenericArg, PathId, Ty as AstTy, TyId};
use crate::names::path::Path as NamePath;
use crate::names::{Def, ItemRef, Ns, plain};
use crate::outline::item::ItemKind;
use crate::outline::source::Source;
use crate::sema::Context;
use crate::sema::body_scope::{BodyPath, outside};
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

fn is_interior_mut_adt(cx: &Context<'_>, item: ItemRef) -> bool {
    matches!(cx.lang(item), Some("unsafe_cell" | "unsafe_pinned"))
        || is_interior_mut_name(cx.plain_name(item))
}

fn resolve_ast_type_path(
    cx: &mut Context<'_>,
    scope: Option<ItemRef>,
    body: &Body,
    src: &Source,
    path_id: PathId,
) -> Option<ItemRef> {
    if body.path(path_id).qself.is_some() {
        return None;
    }
    let scope = scope?;
    let home = cx.module_of(scope).ok()?;
    let (segs, absolute, dollar) =
        crate::sema::infer::segments(&cx.names.symbols, src, body, path_id);
    let name_path = NamePath {
        segments: &segs,
        absolute,
        dollar,
    };
    let root = scope.body_owner();
    let from = scope.outer().map(|(_, at)| at);
    let def = match cx.local_path(home, Some((root, from)), &name_path, Ns::Types) {
        Some(BodyPath::Def(def)) => Some(def),
        Some(BodyPath::Variant(_)) => None,
        Some(BodyPath::Outside(skip)) => cx
            .paths(home.krate)
            .resolve(&outside(&name_path, skip), home),
        None => cx
            .body_use_path((root, home), &name_path, from)
            .or_else(|| cx.paths(home.krate).resolve(&name_path, home)),
    };
    match def {
        Some(Def::Item(item)) => Some(item),
        _ => None,
    }
}

struct InvWalk<'a, 'b> {
    cx: &'b mut Context<'a>,
    scope: Option<ItemRef>,
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
    cx: &mut Context<'_>,
    scope: Option<ItemRef>,
    body: &Body,
    src: &Source,
    ty_id: TyId,
    hint_ty: Option<Ty>,
) -> InvariantLifetimes {
    collect_ast_invariant_lifetimes_mode(cx, scope, body, src, ty_id, hint_ty, false)
}

pub(in super::super) fn collect_ast_invariant_lifetimes_mode(
    cx: &mut Context<'_>,
    scope: Option<ItemRef>,
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
        scope,
        body,
        src,
        seen_adts: &mut seen_adts,
        out: &mut out,
        inv_names: None,
    };
    collect_ast_invariant_rec(&mut walk, ty_id, hint_ty, initial_in_inv, false);
    out
}

fn adt_invariant_param_mask(
    cx: &mut Context<'_>,
    adt_item: ItemRef,
    seen_adts: &mut HashSet<ItemRef>,
) -> Vec<(GenericKind, bool)> {
    let Ok(def) = cx.adt_def(adt_item) else {
        return Vec::new();
    };
    let adt_src = cx.item(adt_item).0;
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
        scope: Some(adt_item),
        body: &def.body,
        src: adt_src,
        seen_adts,
        out: &mut field_inv,
        inv_names: Some(&mut field_names),
    };
    for var in &def.variants {
        for field in &var.fields {
            collect_ast_invariant_rec(&mut walk, field.ty, None, false, false);
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

fn alias_invariant_info(
    cx: &mut Context<'_>,
    alias_item: ItemRef,
    seen_adts: &mut HashSet<ItemRef>,
) -> (InvariantLifetimes, Vec<(GenericKind, bool)>) {
    let Ok(def) = cx.alias_def(alias_item) else {
        return (InvariantLifetimes::default(), Vec::new());
    };
    let alias_src = cx.item(alias_item).0;
    let params: Vec<(GenericKind, &str)> = def
        .generics
        .1
        .params
        .iter()
        .filter(|g| matches!(g.kind, GenericKind::Lifetime | GenericKind::Type))
        .map(|g| (g.kind, alias_src.text(g.name)))
        .collect();
    if !seen_adts.insert(alias_item) {
        return (
            InvariantLifetimes::default(),
            params.into_iter().map(|(k, _)| (k, false)).collect(),
        );
    }
    let mut body_inv = InvariantLifetimes::default();
    let mut body_names = Vec::new();
    let mut walk = InvWalk {
        cx,
        scope: Some(alias_item),
        body: &def.arena,
        src: alias_src,
        seen_adts,
        out: &mut body_inv,
        inv_names: Some(&mut body_names),
    };
    collect_ast_invariant_rec(&mut walk, def.ty, None, false, false);
    seen_adts.remove(&alias_item);
    let mask = params
        .iter()
        .map(|&(kind, name)| {
            let inv = match kind {
                GenericKind::Lifetime => body_inv.named.iter().any(|n| n == name),
                GenericKind::Type => body_names.iter().any(|n| n == name),
                _ => false,
            };
            (kind, inv)
        })
        .collect();
    let fixed_inv = InvariantLifetimes {
        saw_static: body_inv.saw_static,
        has_elided: body_inv.has_elided,
        named: Vec::new(),
    };
    (fixed_inv, mask)
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
    in_owned: bool,
) {
    let hint_ty = peel_hint_ty(walk.cx, hint_ty);
    match *walk.body.ty(ty_id) {
        AstTy::Paren(inner) | AstTy::Pat(inner, _) => {
            collect_ast_invariant_rec(walk, inner, hint_ty, in_inv, in_owned);
        }
        AstTy::Slice(inner) | AstTy::Array(inner, _) => {
            let inner_hint = hint_ty.and_then(|t| match *walk.cx.types.kind(t) {
                Kind::Slice(e) | Kind::Array(e, _) => Some(e),
                _ => None,
            });
            collect_ast_invariant_rec(walk, inner, inner_hint, in_inv, in_owned);
        }
        AstTy::Tuple(list) => {
            let elem_hints: Option<Vec<Ty>> = hint_ty.and_then(|t| match *walk.cx.types.kind(t) {
                Kind::Tuple(substs) => Some(walk.cx.types.substs(substs).to_vec()),
                _ => None,
            });
            for (idx, &t) in walk.body.list(list).iter().enumerate() {
                let eh = elem_hints.as_ref().and_then(|s| s.get(idx).copied());
                collect_ast_invariant_rec(walk, TyId(t), eh, in_inv, in_owned);
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
            collect_ast_invariant_rec(walk, inner, inner_hint, in_inv || mutable, false);
        }
        AstTy::Ptr { mutable, ty: inner } => {
            let inner_hint = hint_ty.and_then(|t| match *walk.cx.types.kind(t) {
                Kind::Ptr(_, pointee) => Some(pointee),
                _ => None,
            });
            collect_ast_invariant_rec(walk, inner, inner_hint, in_inv || mutable, true);
        }
        AstTy::ImplTrait(_, list) if in_inv => {
            collect_bounds_invariant(walk, list, false);
        }
        AstTy::DynTrait(list) if in_inv => {
            collect_bounds_invariant(walk, list, in_owned);
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

fn collect_bounds_invariant(
    walk: &mut InvWalk<'_, '_>,
    list: crate::body::List,
    implicit_static: bool,
) {
    let mut lts = Lifetimes::default();
    let mut has_lt = false;
    for &b_id in walk.body.list(list) {
        match walk.body.bounds[b_id as usize] {
            Bound::Lifetime(lt) => {
                has_lt = true;
                walk.out.push_lt(walk.src.text(lt));
            }
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
    if implicit_static && !has_lt {
        walk.out.saw_static = true;
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
    let adt_info: Option<(ItemRef, Vec<Ty>)> = hint_ty.and_then(|t| match *walk.cx.types.kind(t) {
        Kind::Adt(item, substs) => Some((item, walk.cx.types.substs(substs).to_vec())),
        _ => None,
    });
    let resolved_item = resolve_ast_type_path(walk.cx, walk.scope, walk.body, walk.src, path_id);
    let is_alias = resolved_item.is_some_and(|it| walk.cx.item(it).1.kind == ItemKind::Type);
    let mut arg_mode: Option<Vec<(GenericKind, bool)>> = None;
    let (next_inv, next_owned) = if is_alias && let Some(alias_item) = resolved_item {
        let (alias_inv, mask) = alias_invariant_info(walk.cx, alias_item, walk.seen_adts);
        walk.out.saw_static |= alias_inv.saw_static;
        walk.out.has_elided |= alias_inv.has_elided;
        if mask.iter().any(|&(_, inv)| inv) {
            arg_mode = Some(mask);
        }
        (in_inv, false)
    } else {
        let adt_item = adt_info.as_ref().map(|(it, _)| *it).or(resolved_item);
        let owned = is_owned_container(last_name)
            || adt_item.is_some_and(|it| is_owned_container(walk.cx.plain_name(it)));
        let inv = if in_inv
            || is_interior_mut_name(last_name)
            || adt_item.is_some_and(|it| is_interior_mut_adt(walk.cx, it))
        {
            true
        } else if owned {
            false
        } else if let Some(it) = adt_item {
            let mask = adt_invariant_param_mask(walk.cx, it, walk.seen_adts);
            if mask.iter().any(|&(_, inv)| inv) {
                arg_mode = Some(mask);
            }
            false
        } else {
            false
        };
        (inv, owned)
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
                let sub_hint = (!is_alias)
                    .then(|| {
                        adt_info
                            .as_ref()
                            .and_then(|(_, substs)| substs.get(ty_idx).copied())
                    })
                    .flatten();
                ty_idx += 1;
                collect_ast_invariant_rec(walk, t, sub_hint, is_inv, next_owned);
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

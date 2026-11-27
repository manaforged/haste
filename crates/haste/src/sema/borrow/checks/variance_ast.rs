use rustc_hash::FxHashSet as HashSet;

use crate::body::sig::{GenericKind, SelfKind};
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

#[derive(Clone, Copy, PartialEq, Eq)]
pub(in super::super) enum Variance {
    Covariant,
    Contravariant,
    Invariant,
}

impl Variance {
    fn flip(self) -> Self {
        match self {
            Self::Covariant => Self::Contravariant,
            Self::Contravariant => Self::Covariant,
            Self::Invariant => Self::Invariant,
        }
    }

    fn xform(self, inner: Self) -> Self {
        match self {
            Self::Covariant => inner,
            Self::Contravariant => inner.flip(),
            Self::Invariant => Self::Invariant,
        }
    }

    fn join(self, other: Self) -> Self {
        if self == other { self } else { Self::Invariant }
    }
}

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

#[derive(Clone, Default)]
pub(in super::super) struct ContraLifetimes {
    pub(in super::super) saw_static: bool,
    pub(in super::super) has_hrtb: bool,
    pub(in super::super) named: Vec<String>,
}

impl ContraLifetimes {
    pub(in super::super) fn is_empty(&self) -> bool {
        !self.saw_static && !self.has_hrtb && self.named.is_empty()
    }

    pub(in super::super) fn is_only_static(&self) -> bool {
        self.saw_static && !self.has_hrtb && self.named.is_empty()
    }

    pub(in super::super) fn is_only_hrtb(&self) -> bool {
        self.has_hrtb && !self.saw_static && self.named.is_empty()
    }

    pub(in super::super) fn push_lt(&mut self, lt: &str, is_free_lt: bool) {
        if lt == LT_STATIC {
            self.saw_static = true;
        } else if lt == LT_ELIDED || !is_free_lt {
            self.has_hrtb = true;
        } else if !self.named.iter().any(|s| s == lt) {
            self.named.push(lt.to_owned());
        }
    }
}

#[derive(Clone, Default)]
pub(in super::super) struct VarianceLifetimes {
    pub(in super::super) inv: InvariantLifetimes,
    pub(in super::super) contra: ContraLifetimes,
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

pub(in super::super) fn resolve_ast_type_path(
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

fn collect_free_lifetimes(cx: &mut Context<'_>, scope: Option<ItemRef>) -> Vec<String> {
    let Some(item) = scope else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let mut push = |name: &str| {
        if !out.iter().any(|s| s == name) {
            out.push(name.to_owned());
        }
    };
    if let Ok(sig) = cx.signature(item) {
        let src = cx.item(item).0;
        for g in &sig.generics {
            if g.kind == GenericKind::Lifetime {
                push(src.text(g.name));
            }
        }
        if let Some(block) = cx.owner_impl(item)
            && let Ok(hdr) = cx.impl_header(block)
        {
            let impl_src = &cx.impl_file(block).source;
            for g in &hdr.generics {
                if g.kind == GenericKind::Lifetime {
                    push(impl_src.text(g.name));
                }
            }
        } else if let Some(tr) = cx.owner_trait(item)
            && let Ok(tg) = cx.trait_generics(tr)
        {
            let tsrc = &cx.file(tr).source;
            for g in &tg.1.params {
                if g.kind == GenericKind::Lifetime {
                    push(tsrc.text(g.name));
                }
            }
        }
    } else if let Ok(def) = cx.adt_def(item) {
        let src = cx.item(item).0;
        for g in &def.generics.params {
            if g.kind == GenericKind::Lifetime {
                push(src.text(g.name));
            }
        }
    } else if let Ok(def) = cx.alias_def(item) {
        let src = cx.item(item).0;
        for g in &def.generics.1.params {
            if g.kind == GenericKind::Lifetime {
                push(src.text(g.name));
            }
        }
    }
    out
}

struct InvWalk<'a, 'b> {
    cx: &'b mut Context<'a>,
    scope: Option<ItemRef>,
    body: &'b Body,
    src: &'b Source,
    free_lts: &'b [String],
    seen_adts: &'b mut HashSet<ItemRef>,
    out: &'b mut VarianceLifetimes,
    param_vars: Option<&'b mut Vec<(GenericKind, String, Variance)>>,
}

impl InvWalk<'_, '_> {
    fn is_free_lt(&self, name: &str) -> bool {
        self.free_lts.iter().any(|s| s == name)
    }

    fn record_param(&mut self, kind: GenericKind, name: &str, var: Variance) {
        if let Some(vars) = self.param_vars.as_deref_mut() {
            if let Some(entry) = vars.iter_mut().find(|(k, n, _)| *k == kind && n == name) {
                entry.2 = entry.2.join(var);
            } else {
                vars.push((kind, name.to_owned(), var));
            }
        }
    }

    fn push_lt(&mut self, text: &str, var: Variance) {
        self.record_param(GenericKind::Lifetime, text, var);
        match var {
            Variance::Invariant => self.out.inv.push_lt(text),
            Variance::Contravariant => {
                let is_free = self.is_free_lt(text);
                self.out.contra.push_lt(text, is_free);
            }
            Variance::Covariant => {}
        }
    }

    fn mark_omitted_lt(&mut self, base_var: Variance, pv: Variance) {
        match base_var.xform(pv) {
            Variance::Invariant => self.out.inv.has_elided = true,
            Variance::Contravariant => self.out.contra.has_hrtb = true,
            Variance::Covariant => {}
        }
    }
}

pub(in super::super) fn collect_ast_variance(
    cx: &mut Context<'_>,
    scope: Option<ItemRef>,
    body: &Body,
    src: &Source,
    ty_id: TyId,
    hint_ty: Option<Ty>,
) -> VarianceLifetimes {
    collect_ast_variance_mode(cx, scope, body, src, ty_id, hint_ty, Variance::Covariant)
}

pub(in super::super) fn collect_ast_variance_mode(
    cx: &mut Context<'_>,
    scope: Option<ItemRef>,
    body: &Body,
    src: &Source,
    ty_id: TyId,
    hint_ty: Option<Ty>,
    initial_var: Variance,
) -> VarianceLifetimes {
    let free_lts = collect_free_lifetimes(cx, scope);
    let mut out = VarianceLifetimes::default();
    let mut seen_adts = HashSet::default();
    let mut walk = InvWalk {
        cx,
        scope,
        body,
        src,
        free_lts: &free_lts,
        seen_adts: &mut seen_adts,
        out: &mut out,
        param_vars: None,
    };
    collect_ast_variance_rec(&mut walk, ty_id, hint_ty, initial_var, false);
    out
}

pub(in super::super) fn collect_ast_invariant_lifetimes(
    cx: &mut Context<'_>,
    scope: Option<ItemRef>,
    body: &Body,
    src: &Source,
    ty_id: TyId,
    hint_ty: Option<Ty>,
) -> InvariantLifetimes {
    collect_ast_variance(cx, scope, body, src, ty_id, hint_ty).inv
}

pub(in super::super) fn collect_fn_item_variance(
    cx: &mut Context<'_>,
    fn_item: ItemRef,
) -> VarianceLifetimes {
    let Ok(sig) = cx.signature(fn_item) else {
        return VarianceLifetimes::default();
    };
    let fn_src = cx.item(fn_item).0;
    let mut out = VarianceLifetimes::default();
    let mut seen_adts = HashSet::default();
    let mut walk = InvWalk {
        cx,
        scope: Some(fn_item),
        body: &sig.body,
        src: fn_src,
        free_lts: &[],
        seen_adts: &mut seen_adts,
        out: &mut out,
        param_vars: None,
    };
    if let Some((kind, _)) = sig.receiver {
        match kind {
            SelfKind::Ref | SelfKind::RefMut => match sig.receiver_lifetime {
                Some(lt) => walk.out.contra.push_lt(fn_src.text(lt), false),
                None => walk.out.contra.has_hrtb = true,
            },
            SelfKind::Typed(ty) => {
                collect_ast_variance_rec(&mut walk, ty, None, Variance::Contravariant, false);
            }
            SelfKind::Value => {}
        }
    }
    for &(_, param_ty) in &sig.params {
        collect_ast_variance_rec(&mut walk, param_ty, None, Variance::Contravariant, false);
    }
    if let Some(ret_ty) = sig.ret {
        collect_ast_variance_rec(&mut walk, ret_ty, None, Variance::Covariant, false);
    }
    out
}

fn adt_param_variances(
    cx: &mut Context<'_>,
    adt_item: ItemRef,
    seen_adts: &mut HashSet<ItemRef>,
) -> Vec<(GenericKind, Option<Variance>)> {
    let Ok(def) = cx.adt_def(adt_item) else {
        return Vec::new();
    };
    let adt_src = cx.item(adt_item).0;
    let params: Vec<(GenericKind, String)> = def
        .generics
        .params
        .iter()
        .filter(|g| matches!(g.kind, GenericKind::Lifetime | GenericKind::Type))
        .map(|g| (g.kind, adt_src.text(g.name).to_owned()))
        .collect();
    if params.is_empty() || !seen_adts.insert(adt_item) {
        return params.into_iter().map(|(k, _)| (k, None)).collect();
    }
    let free_lts: Vec<String> = params
        .iter()
        .filter(|(k, _)| *k == GenericKind::Lifetime)
        .map(|(_, n)| n.clone())
        .collect();
    let mut field_out = VarianceLifetimes::default();
    let mut field_vars = Vec::new();
    let mut walk = InvWalk {
        cx,
        scope: Some(adt_item),
        body: &def.body,
        src: adt_src,
        free_lts: &free_lts,
        seen_adts,
        out: &mut field_out,
        param_vars: Some(&mut field_vars),
    };
    for var in &def.variants {
        for field in &var.fields {
            collect_ast_variance_rec(&mut walk, field.ty, None, Variance::Covariant, false);
        }
    }
    seen_adts.remove(&adt_item);
    params
        .into_iter()
        .map(|(kind, name)| {
            let v = field_vars
                .iter()
                .find(|(k, n, _)| *k == kind && *n == name)
                .map(|(_, _, v)| *v);
            (kind, v)
        })
        .collect()
}

fn alias_variance_info(
    cx: &mut Context<'_>,
    alias_item: ItemRef,
    seen_adts: &mut HashSet<ItemRef>,
) -> (VarianceLifetimes, Vec<(GenericKind, Option<Variance>)>) {
    let Ok(def) = cx.alias_def(alias_item) else {
        return (VarianceLifetimes::default(), Vec::new());
    };
    let alias_src = cx.item(alias_item).0;
    let params: Vec<(GenericKind, String)> = def
        .generics
        .1
        .params
        .iter()
        .filter(|g| matches!(g.kind, GenericKind::Lifetime | GenericKind::Type))
        .map(|g| (g.kind, alias_src.text(g.name).to_owned()))
        .collect();
    if !seen_adts.insert(alias_item) {
        return (
            VarianceLifetimes::default(),
            params.into_iter().map(|(k, _)| (k, None)).collect(),
        );
    }
    let free_lts: Vec<String> = params
        .iter()
        .filter(|(k, _)| *k == GenericKind::Lifetime)
        .map(|(_, n)| n.clone())
        .collect();
    let mut body_out = VarianceLifetimes::default();
    let mut body_vars = Vec::new();
    let mut walk = InvWalk {
        cx,
        scope: Some(alias_item),
        body: &def.arena,
        src: alias_src,
        free_lts: &free_lts,
        seen_adts,
        out: &mut body_out,
        param_vars: Some(&mut body_vars),
    };
    collect_ast_variance_rec(&mut walk, def.ty, None, Variance::Covariant, false);
    seen_adts.remove(&alias_item);
    let mask = params
        .into_iter()
        .map(|(kind, name)| {
            let v = body_vars
                .iter()
                .find(|(k, n, _)| *k == kind && *n == name)
                .map(|(_, _, v)| *v);
            (kind, v)
        })
        .collect();
    let fixed = VarianceLifetimes {
        inv: InvariantLifetimes {
            saw_static: body_out.inv.saw_static,
            has_elided: body_out.inv.has_elided,
            named: Vec::new(),
        },
        contra: ContraLifetimes {
            saw_static: body_out.contra.saw_static,
            has_hrtb: body_out.contra.has_hrtb,
            named: Vec::new(),
        },
    };
    (fixed, mask)
}

fn peel_hint_ty(cx: &Context<'_>, hint_ty: Option<Ty>) -> Option<Ty> {
    let mut ty = hint_ty.filter(|&t| t != Ty::NONE)?;
    while let Kind::Pat(inner, ..) = *cx.types.kind(ty) {
        ty = inner;
    }
    Some(ty)
}

fn collect_ast_variance_rec(
    walk: &mut InvWalk<'_, '_>,
    ty_id: TyId,
    hint_ty: Option<Ty>,
    variance: Variance,
    in_owned: bool,
) {
    let hint_ty = peel_hint_ty(walk.cx, hint_ty);
    match *walk.body.ty(ty_id) {
        AstTy::Paren(inner) | AstTy::Pat(inner, _) => {
            collect_ast_variance_rec(walk, inner, hint_ty, variance, in_owned);
        }
        AstTy::Slice(inner) | AstTy::Array(inner, _) => {
            let inner_hint = hint_ty.and_then(|t| match *walk.cx.types.kind(t) {
                Kind::Slice(e) | Kind::Array(e, _) => Some(e),
                _ => None,
            });
            collect_ast_variance_rec(walk, inner, inner_hint, variance, in_owned);
        }
        AstTy::Tuple(list) => {
            let elem_hints: Option<Vec<Ty>> = hint_ty.and_then(|t| match *walk.cx.types.kind(t) {
                Kind::Tuple(substs) => Some(walk.cx.types.substs(substs).to_vec()),
                _ => None,
            });
            for (idx, &t) in walk.body.list(list).iter().enumerate() {
                let eh = elem_hints.as_ref().and_then(|s| s.get(idx).copied());
                collect_ast_variance_rec(walk, TyId(t), eh, variance, in_owned);
            }
        }
        AstTy::Ref {
            mutable,
            lifetime,
            ty: inner,
        } => {
            match lifetime {
                Some(lt) => walk.push_lt(walk.src.text(lt), variance),
                None if variance == Variance::Invariant => walk.out.inv.has_elided = true,
                None if variance == Variance::Contravariant => walk.out.contra.has_hrtb = true,
                None => {}
            }
            let inner_hint = hint_ty.and_then(|t| match *walk.cx.types.kind(t) {
                Kind::Ref(_, pointee) => Some(pointee),
                _ => None,
            });
            let next_var = if mutable {
                Variance::Invariant
            } else {
                variance
            };
            collect_ast_variance_rec(walk, inner, inner_hint, next_var, false);
        }
        AstTy::Ptr { mutable, ty: inner } => {
            let inner_hint = hint_ty.and_then(|t| match *walk.cx.types.kind(t) {
                Kind::Ptr(_, pointee) => Some(pointee),
                _ => None,
            });
            let next_var = if mutable {
                Variance::Invariant
            } else {
                variance
            };
            collect_ast_variance_rec(walk, inner, inner_hint, next_var, true);
        }
        AstTy::ImplTrait(_, list) if variance != Variance::Covariant => {
            collect_bounds_variance(walk, list, variance, false);
        }
        AstTy::DynTrait(list) if variance != Variance::Covariant => {
            collect_bounds_variance(walk, list, variance, in_owned);
        }
        AstTy::Fn { params, ret, .. } => {
            let param_var = variance.flip();
            let param_hints: Option<Vec<Ty>> = hint_ty.and_then(|t| match *walk.cx.types.kind(t) {
                Kind::FnPtr(ps, _) => Some(walk.cx.types.substs(ps).to_vec()),
                _ => None,
            });
            for (idx, &t) in walk.body.list(params).iter().enumerate() {
                let ph = param_hints.as_ref().and_then(|s| s.get(idx).copied());
                collect_ast_variance_rec(walk, TyId(t), ph, param_var, false);
            }
            if let Some(r) = ret {
                let ret_hint = hint_ty.and_then(|t| match *walk.cx.types.kind(t) {
                    Kind::FnPtr(_, ret_ty) => Some(ret_ty),
                    _ => None,
                });
                collect_ast_variance_rec(walk, r, ret_hint, variance, false);
            }
        }
        AstTy::Path(path_id) => {
            collect_path_variance(walk, path_id, hint_ty, variance);
        }
        _ => {}
    }
}

fn collect_bounds_variance(
    walk: &mut InvWalk<'_, '_>,
    list: crate::body::List,
    variance: Variance,
    implicit_static: bool,
) {
    let mut lts = Lifetimes::default();
    let mut has_lt = false;
    for &b_id in walk.body.list(list) {
        match walk.body.bounds[b_id as usize] {
            Bound::Lifetime(lt) => {
                has_lt = true;
                walk.push_lt(walk.src.text(lt), variance);
            }
            Bound::Trait { path, .. } => {
                for &seg_id in walk.body.list(walk.body.path(path).segments) {
                    if let Some(Args::Angle(args)) = walk.body.segments[seg_id as usize].args {
                        for &a in walk.body.list(args) {
                            match walk.body.args[a as usize] {
                                GenericArg::Lifetime(lt) => {
                                    walk.push_lt(walk.src.text(lt), variance);
                                }
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
        if variance == Variance::Invariant {
            walk.out.inv.saw_static = true;
        } else {
            walk.out.contra.saw_static = true;
        }
    }
    if variance == Variance::Invariant {
        walk.out.inv.merge_all(&lts);
        for &n in &lts.names {
            walk.record_param(GenericKind::Type, n, Variance::Invariant);
        }
    }
}

fn mask_nth_var(
    mask: &Option<Vec<(GenericKind, Option<Variance>)>>,
    kind: GenericKind,
    idx: usize,
) -> Option<Variance> {
    mask.as_ref().and_then(|m| {
        m.iter()
            .filter(|&&(k, _)| k == kind)
            .nth(idx)
            .and_then(|&(_, v)| v)
    })
}

fn collect_path_variance(
    walk: &mut InvWalk<'_, '_>,
    path_id: PathId,
    hint_ty: Option<Ty>,
    variance: Variance,
) {
    let p = walk.body.path(path_id);
    let segs = walk.body.list(p.segments);
    let Some(&last_id) = segs.last() else {
        return;
    };
    let last_seg = walk.body.segments[last_id as usize];
    let last_name = plain(walk.src.text(last_seg.name));
    if p.qself.is_none() && segs.len() == 1 && last_seg.args.is_none() {
        walk.record_param(GenericKind::Type, last_name, variance);
    }
    let adt_info: Option<(ItemRef, Vec<Ty>)> = hint_ty.and_then(|t| match *walk.cx.types.kind(t) {
        Kind::Adt(item, substs) => Some((item, walk.cx.types.substs(substs).to_vec())),
        _ => None,
    });
    let resolved_item = resolve_ast_type_path(walk.cx, walk.scope, walk.body, walk.src, path_id);
    let is_alias = resolved_item.is_some_and(|it| walk.cx.item(it).1.kind == ItemKind::Type);
    let mut arg_mode: Option<Vec<(GenericKind, Option<Variance>)>> = None;
    let (base_var, next_owned) = if is_alias && let Some(alias_item) = resolved_item {
        let (alias_fixed, mask) = alias_variance_info(walk.cx, alias_item, walk.seen_adts);
        walk.out.inv.saw_static |= alias_fixed.inv.saw_static;
        walk.out.inv.has_elided |= alias_fixed.inv.has_elided;
        match variance {
            Variance::Covariant => {
                walk.out.contra.saw_static |= alias_fixed.contra.saw_static;
                walk.out.contra.has_hrtb |= alias_fixed.contra.has_hrtb;
            }
            Variance::Invariant => {
                walk.out.inv.saw_static |= alias_fixed.contra.saw_static;
                walk.out.inv.has_elided |= alias_fixed.contra.has_hrtb;
            }
            Variance::Contravariant => {}
        }
        if mask.iter().any(|&(_, v)| v.is_some()) {
            arg_mode = Some(mask);
        }
        (variance, false)
    } else {
        let adt_item = adt_info.as_ref().map(|(it, _)| *it).or(resolved_item);
        let owned = match adt_item {
            Some(it) => {
                walk.cx.outline.is_sysroot(it.krate) && is_owned_container(walk.cx.plain_name(it))
            }
            None => segs.len() == 1 && is_owned_container(last_name),
        };
        let next_var = if variance == Variance::Invariant
            || is_interior_mut_name(last_name)
            || adt_item.is_some_and(|it| is_interior_mut_adt(walk.cx, it))
        {
            Variance::Invariant
        } else if owned {
            variance
        } else if let Some(it) = adt_item {
            let mask = adt_param_variances(walk.cx, it, walk.seen_adts);
            if mask.iter().any(|&(_, v)| v.is_some()) {
                arg_mode = Some(mask);
            }
            variance
        } else {
            variance
        };
        (next_var, owned)
    };
    let Some(Args::Angle(arg_list)) = last_seg.args else {
        if let Some(mask) = arg_mode {
            for &(k, v) in &mask {
                if k == GenericKind::Lifetime
                    && let Some(pv) = v
                {
                    walk.mark_omitted_lt(base_var, pv);
                }
            }
        }
        return;
    };
    let mut lt_idx = 0usize;
    let mut ty_idx = 0usize;
    for &arg_id in walk.body.list(arg_list) {
        match walk.body.args[arg_id as usize] {
            GenericArg::Lifetime(lt) => {
                let param_var = mask_nth_var(&arg_mode, GenericKind::Lifetime, lt_idx);
                lt_idx += 1;
                let arg_var = if base_var == Variance::Invariant {
                    Variance::Invariant
                } else {
                    param_var.map_or(base_var, |pv| base_var.xform(pv))
                };
                walk.push_lt(walk.src.text(lt), arg_var);
            }
            GenericArg::Ty(t) | GenericArg::Binding { ty: t, .. } => {
                let param_var = mask_nth_var(&arg_mode, GenericKind::Type, ty_idx);
                let arg_var = if base_var == Variance::Invariant {
                    Variance::Invariant
                } else {
                    param_var.map_or(base_var, |pv| base_var.xform(pv))
                };
                let sub_hint = (!is_alias)
                    .then(|| {
                        adt_info
                            .as_ref()
                            .and_then(|(_, substs)| substs.get(ty_idx).copied())
                    })
                    .flatten();
                ty_idx += 1;
                collect_ast_variance_rec(walk, t, sub_hint, arg_var, next_owned);
            }
            _ => {}
        }
    }
    if let Some(mask) = arg_mode {
        for &(_, v) in mask
            .iter()
            .filter(|&&(k, _)| k == GenericKind::Lifetime)
            .skip(lt_idx)
        {
            if let Some(pv) = v {
                walk.mark_omitted_lt(base_var, pv);
            }
        }
    }
}

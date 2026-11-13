use std::sync::Arc;

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::body::sig::GenericKind;
use crate::names::{ItemRef, Primitive};

use super::super::Context;
use super::super::infer::{LazyTables, Res, Shape, Typed, body_relaxes_sized, maybe_unsized, param_relaxes_sized};
use super::super::substs::Substs;
use super::super::summary::{PolyObligation, SummaryKind};
use super::super::ty::{Kind, Ty};
use super::Instance;

#[derive(Default)]
pub(super) struct ReachFilter {
    pub(super) seen_fixed_poly: HashSet<ItemRef>,
    pub(super) seen_sealed_sysroot: HashSet<ItemRef>,
    pub(super) sealed_cache: HashMap<ItemRef, bool>,
}

impl ReachFilter {
    pub(super) fn check_sealed(&mut self, cx: &mut Context<'_>, item: ItemRef, visiting: &mut HashSet<ItemRef>) -> Option<bool> {
        if let Some(&cached) = self.sealed_cache.get(&item) {
            return Some(cached);
        }
        if !cx.outline.is_fixed(item.krate) {
            self.sealed_cache.insert(item, false);
            return Some(false);
        }
        if !visiting.insert(item) {
            return Some(false);
        }
        let summary = match cx.store.lookups.summaries.get_ready(&item) {
            Some(SummaryKind::Poly(summary)) => summary,
            Some(SummaryKind::MonomorphicFallback) => {
                visiting.remove(&item);
                self.sealed_cache.insert(item, false);
                return Some(false);
            }
            None => match super::super::summary::frozen::frozen_summary(cx, item) {
                Some(summary) => summary,
                None => {
                    visiting.remove(&item);
                    return None;
                }
            },
        };
        let local_ok = summary.projections.is_empty()
            && summary.obligations.iter().all(|o| matches!(o, PolyObligation::Prove { .. }))
            && summary.call_holes.is_empty()
            && summary.op_holes.is_empty()
            && summary.const_holes.is_empty()
            && summary.template.closures.is_empty()
            && summary.layout_tys.iter().all(|&ty| layout_ty_infallible(cx, item, ty))
            && summary.drop_tys.iter().all(|&ty| layout_ty_infallible(cx, item, ty));
        if !local_ok {
            visiting.remove(&item);
            self.sealed_cache.insert(item, false);
            return Some(false);
        }
        let mut all_ready = true;
        for &(_, callee, _) in &summary.direct_calls {
            match self.check_sealed(cx, callee, visiting) {
                Some(true) => {}
                Some(false) => {
                    visiting.remove(&item);
                    self.sealed_cache.insert(item, false);
                    return Some(false);
                }
                None => all_ready = false,
            }
        }
        visiting.remove(&item);
        if all_ready {
            self.sealed_cache.insert(item, true);
            Some(true)
        } else {
            None
        }
    }
}

fn param_is_sized(cx: &mut Context<'_>, item: ItemRef, idx: u16) -> bool {
    if cx.owner_trait(item).is_some() && idx == 0 {
        return false;
    }
    let Ok(sig) = cx.signature(item) else { return false };
    if body_relaxes_sized(&sig.body, &cx.file(item).source) {
        return false;
    }
    if let Some(owner) = cx.owner_impl(item) {
        let Ok(h) = cx.impl_header(owner) else { return false };
        if body_relaxes_sized(&h.body, &cx.impl_file(owner).source) {
            return false;
        }
    }
    if let Some(tr) = cx.owner_trait(item) {
        let Ok(tg) = cx.trait_generics(tr) else { return false };
        if body_relaxes_sized(&tg.0, &cx.file(tr).source) {
            return false;
        }
    }
    true
}

fn ty_is_sized(cx: &mut Context<'_>, item: ItemRef, ty: Ty) -> bool {
    match *cx.types.kind(ty) {
        Kind::Prim(p) => !matches!(p, Primitive::Str),
        Kind::Ref(..) | Kind::Ptr(..) | Kind::FnDef(..) | Kind::FnPtr(..) | Kind::Never | Kind::Array(..) => true,
        Kind::Slice(..) | Kind::Dyn(..) => false,
        Kind::Param(idx) => param_is_sized(cx, item, idx),
        Kind::Tuple(elems) => cx.types.substs(elems).last().copied().is_none_or(|last| ty_is_sized(cx, item, last)),
        Kind::Adt(adt, args) => {
            if !cx.outline.is_fixed(adt.krate) {
                return false;
            }
            if !maybe_unsized(cx, adt).unwrap_or(true) {
                return true;
            }
            cx.types.substs(args).to_vec().into_iter().all(|arg| ty_is_sized(cx, item, arg))
        }
        _ => false,
    }
}

fn layout_ty_infallible(cx: &mut Context<'_>, item: ItemRef, ty: Ty) -> bool {
    match *cx.types.kind(ty) {
        Kind::Prim(_) | Kind::Ref(..) | Kind::Ptr(..) | Kind::FnDef(..) | Kind::FnPtr(..) | Kind::Never => true,
        Kind::Param(idx) => param_is_sized(cx, item, idx),
        Kind::Slice(elem) => ty_is_sized(cx, item, elem) && layout_ty_infallible(cx, item, elem),
        Kind::Tuple(elems) => {
            let elems = cx.types.substs(elems).to_vec();
            let non_last = elems.len().saturating_sub(1);
            elems[..non_last].iter().all(|&e| ty_is_sized(cx, item, e)) && elems.iter().all(|&e| layout_ty_infallible(cx, item, e))
        }
        Kind::Adt(adt, args) => {
            if !cx.outline.is_fixed(adt.krate) {
                return false;
            }
            let repr = cx.repr(adt);
            if repr.flags.contains(rustc_abi::ReprFlags::IS_SIMD) || repr.pack.is_some() {
                return false;
            }
            let Ok(def) = cx.adt_def(adt) else { return false };
            if def.generics.params.iter().any(|g| matches!(g.kind, GenericKind::Const(_))) {
                return false;
            }
            let any_relaxed = body_relaxes_sized(&def.body, &cx.file(adt).source);
            let adt_unsized = any_relaxed && maybe_unsized(cx, adt).unwrap_or(true);
            let src = &cx.file(adt).source;
            let args = cx.types.substs(args).to_vec();
            let type_params: Vec<_> = def.generics.params.iter().filter(|g| g.kind != GenericKind::Lifetime).collect();
            for (i, &arg) in args.iter().enumerate() {
                let param_relaxed = type_params.get(i).is_some_and(|g| param_relaxes_sized(&def.body, src, &def.generics.predicates, g));
                if !adt_unsized && param_relaxed {
                    continue;
                }
                if !layout_ty_infallible(cx, item, arg) || !ty_is_sized(cx, item, arg) {
                    return false;
                }
            }
            true
        }
        _ => false,
    }
}

fn ty_has_non_fixed(cx: &Context<'_>, ty: Ty) -> bool {
    if ty == Ty::NONE {
        return true;
    }
    match cx.types.kind(ty) {
        Kind::Prim(_) | Kind::Never | Kind::Value(_) => false,
        Kind::Ref(_, inner) | Kind::Ptr(_, inner) | Kind::Slice(inner) | Kind::Pat(inner, ..) => ty_has_non_fixed(cx, *inner),
        Kind::Array(elem, count) => ty_has_non_fixed(cx, *elem) || ty_has_non_fixed(cx, *count),
        Kind::Tuple(args) => substs_has_non_fixed(cx, *args),
        Kind::FnPtr(args, ret) => substs_has_non_fixed(cx, *args) || ty_has_non_fixed(cx, *ret),
        Kind::Adt(def, args) | Kind::FnDef(def, args) | Kind::Closure(def, args, _) => !cx.outline.is_fixed(def.krate) || substs_has_non_fixed(cx, *args),
        Kind::Dyn(tr, args, assoc, auto) => !cx.outline.is_fixed(tr.krate) || assoc.iter().chain(auto).any(|d| !cx.outline.is_fixed(d.krate)) || substs_has_non_fixed(cx, *args),
        Kind::Proj(_) | Kind::Param(_) | Kind::Var(_) | Kind::IntVar(_) | Kind::FloatVar(_) => true,
    }
}

fn substs_has_non_fixed(cx: &Context<'_>, substs: Substs) -> bool {
    !substs.is_empty() && cx.types.substs(substs).iter().any(|&ty| ty_has_non_fixed(cx, ty))
}

pub(super) fn pre_collect_callees(
    cx: &mut Context<'_>,
    filter: &mut ReachFilter,
    typed: &Result<Arc<Typed>, String>,
    include_fixed: bool,
    out: &mut Vec<(Instance, bool)>,
) {
    let Ok(typed) = typed else { return };
    let check_only = cx.store.lookups.skip_warm_sysroot_bodies;
    let mut visiting = HashSet::default();
    let mut push_callee = |cx: &mut Context<'_>, filter: &mut ReachFilter, item: ItemRef, substs: Substs| {
        if cx.intrinsic(item).is_some() {
            return;
        }
        let mut sealed = false;
        if check_only && cx.outline.is_fixed(item.krate) {
            if !substs_has_non_fixed(cx, substs) {
                return;
            }
            sealed = filter.check_sealed(cx, item, &mut visiting) == Some(true);
        }
        out.push(((item, substs), sealed));
    };
    match &typed.tables {
        LazyTables::Poly(poly) => {
            if include_fixed {
                for (item, substs) in poly.calls() { push_callee(cx, filter, item, substs); }
            } else {
                for (item, substs) in poly.dynamic_calls() { push_callee(cx, filter, item, substs); }
            }
            for r in &typed.drops {
                if let Res::Fn(item, substs) = *r { push_callee(cx, filter, item, substs); }
            }
            match &typed.shape {
                Shape::Glue(g) => for r in g.drop.iter().chain(&g.parts).chain(&g.resume) {
                    if let Res::Fn(item, substs) = *r { push_callee(cx, filter, item, substs); }
                },
                Shape::Untuple(r) => if let Res::Fn(item, substs) = *r { push_callee(cx, filter, item, substs); },
                Shape::Clone(calls) => for r in calls {
                    if let Res::Fn(item, substs) = *r { push_callee(cx, filter, item, substs); }
                },
                _ => {}
            }
        }
        LazyTables::Eager(_) => {
            for (item, substs) in typed.calls() { push_callee(cx, filter, item, substs); }
        }
    }
}

pub(super) fn commit_filter_caller(cx: &Context<'_>, filter: &mut ReachFilter, caller: Instance, typed: &Result<Arc<Typed>, String>) {
    if let Ok(typed) = typed && matches!(typed.tables, LazyTables::Poly(_)) {
        filter.seen_fixed_poly.insert(caller.0);
        if cx.outline.is_fixed(caller.0.krate) {
            filter.seen_sealed_sysroot.insert(caller.0);
        }
    }
}

pub(super) fn collect_filtered_callees(
    cx: &mut Context<'_>,
    filter: &mut ReachFilter,
    caller: Instance,
    typed: &Result<Arc<Typed>, String>,
    out: &mut Vec<Instance>,
) {
    let include_fixed = !filter.seen_fixed_poly.contains(&caller.0);
    let mut pre = Vec::new();
    pre_collect_callees(cx, filter, typed, include_fixed, &mut pre);
    commit_filter_caller(cx, filter, caller, typed);
    for (callee, sealed) in pre {
        if sealed && !filter.seen_sealed_sysroot.insert(callee.0) {
            continue;
        }
        out.push(callee);
    }
}

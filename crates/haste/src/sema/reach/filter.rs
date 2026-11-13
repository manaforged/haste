use std::sync::Arc;

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::body::sig::GenericKind;
use crate::names::{ItemRef, Primitive};

use super::super::Context;
use super::super::infer::{LazyTables, Res, Shape, Typed, body_relaxes_sized, maybe_unsized, param_relaxes_sized};
use super::super::impls::head;
use super::super::intrinsic::Intrinsic;
use super::super::substs::Substs;
use super::super::summary::{ConstHole, PolyObligation, poly_summary};
use super::super::ty::{Kind, Ty};
use super::Instance;

#[derive(Default)]
pub(super) struct ReachFilter {
    pub(super) seen_fixed_poly: HashSet<ItemRef>,
    pub(super) seen_sealed_sysroot: HashSet<ItemRef>,
    pub(super) sealed_cache: HashMap<ItemRef, bool>,
    pub(super) sealed_log: Vec<(ItemRef, bool)>,
    pub(super) dirty: Vec<(ItemRef, bool)>,
}

impl ReachFilter {
    fn record_sealed(&mut self, item: ItemRef, sealed: bool) {
        if self.sealed_cache.insert(item, sealed).is_none() {
            self.dirty.push((item, sealed));
        }
    }

    pub(super) fn sync_from(&mut self, global: &Self, synced_len: &mut usize) {
        if global.sealed_log.len() > *synced_len {
            for &(k, v) in &global.sealed_log[*synced_len..] {
                self.sealed_cache.entry(k).or_insert(v);
            }
            *synced_len = global.sealed_log.len();
        }
    }

    pub(super) fn drain_to(&mut self, global: &mut Self) {
        for (k, v) in self.dirty.drain(..) {
            if global.sealed_cache.insert(k, v).is_none() {
                global.sealed_log.push((k, v));
            }
        }
    }

    pub(super) fn check_sealed(&mut self, cx: &mut Context<'_>, item: ItemRef, visiting: &mut HashSet<ItemRef>) -> Option<bool> {
        if let Some(&cached) = self.sealed_cache.get(&item) {
            return Some(cached);
        }
        if let Some(intrinsic) = cx.intrinsic(item) {
            let safe = !matches!(intrinsic, Intrinsic::Transmute);
            self.record_sealed(item, safe);
            return Some(safe);
        }
        if !cx.outline.is_fixed(item.krate) {
            self.record_sealed(item, false);
            return Some(false);
        }
        if !visiting.insert(item) {
            return Some(false);
        }
        let Some(summary) = poly_summary(cx, item) else {
            visiting.remove(&item);
            self.record_sealed(item, false);
            return Some(false);
        };
        let local_ok = summary.projections.is_empty()
            && summary.obligations.iter().all(|o| matches!(o, PolyObligation::Prove { .. }))
            && summary.op_holes.is_empty()
            && summary.const_holes.iter().all(|h| matches!(h, ConstHole::Intrinsic { .. }))
            && summary.template.closures.is_empty()
            && summary.layout_tys.iter().all(|&ty| layout_ty_infallible(cx, item, ty))
            && summary.drop_tys.iter().all(|&ty| layout_ty_infallible(cx, item, ty));
        if !local_ok {
            visiting.remove(&item);
            self.record_sealed(item, false);
            return Some(false);
        }
        let mut all_ready = true;
        for &(_, callee, _) in &summary.direct_calls {
            match self.check_sealed(cx, callee, visiting) {
                Some(true) => {}
                Some(false) => {
                    visiting.remove(&item);
                    self.record_sealed(item, false);
                    return Some(false);
                }
                None => all_ready = false,
            }
        }
        for &(_, hole_item, hole_substs) in &summary.call_holes {
            match self.call_hole_sealed(cx, hole_item, hole_substs, visiting) {
                Some(true) => {}
                Some(false) => {
                    visiting.remove(&item);
                    self.record_sealed(item, false);
                    return Some(false);
                }
                None => all_ready = false,
            }
        }
        visiting.remove(&item);
        if all_ready {
            self.record_sealed(item, true);
            Some(true)
        } else {
            None
        }
    }

    fn call_hole_sealed(&mut self, cx: &mut Context<'_>, item: ItemRef, substs: Substs, visiting: &mut HashSet<ItemRef>) -> Option<bool> {
        let Some(tr) = cx.owner_trait(item) else { return Some(false) };
        if !cx.outline.is_fixed(tr.krate) {
            return Some(false);
        }
        let Ok(tg) = cx.trait_generics(tr) else { return Some(false) };
        let count = 1 + tg.1.params.iter().filter(|g| g.kind != GenericKind::Lifetime).count();
        let slice = cx.types.substs(substs).to_vec();
        if slice.len() < count || slice[1..count].iter().any(|&ty| ty_has_non_fixed(cx, ty)) {
            return Some(false);
        }
        let ok_head = match *cx.types.kind(slice[0]) {
            Kind::Prim(_) | Kind::Never | Kind::Ptr(..) | Kind::Slice(..) | Kind::Array(..) | Kind::FnPtr(..) => true,
            Kind::Adt(adt, _) => cx.outline.is_fixed(adt.krate) && cx.lang(adt) != Some("owned_box") && cx.attr_at(adt, "fundamental").is_none(),
            _ => false,
        };
        if !ok_head {
            return Some(false);
        }
        let mut holders = vec![tr.krate];
        let mut stack: Vec<Ty> = slice[..count].to_vec();
        while let Some(ty) = stack.pop() {
            let kind = cx.types.kind(ty).clone();
            if let Kind::Adt(def, _) | Kind::Dyn(def, ..) = kind
                && cx.outline.is_fixed(def.krate)
                && !holders.contains(&def.krate)
            {
                holders.push(def.krate);
            }
            stack.extend(kind.children(&cx.types));
        }
        let Some(head_key) = head(cx.types.kind(slice[0])) else { return Some(false) };
        let blocks = cx.trait_impls(tr, Some(head_key), &holders);
        if blocks.is_empty() {
            return Some(false);
        }
        let name = cx.name(item);
        let mut all_ready = true;
        for block in blocks {
            if cx.reserved(block) {
                continue;
            }
            let member = cx.member(block, name).unwrap_or(item);
            match self.check_sealed(cx, member, visiting) {
                Some(true) => {}
                Some(false) => return Some(false),
                None => all_ready = false,
            }
        }
        all_ready.then_some(true)
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
    (caller, typed, include_fixed): (Instance, &Result<Arc<Typed>, String>, bool),
    out: &mut Vec<(Instance, bool)>,
) {
    let Ok(typed) = typed else { return };
    let check_only = cx.store.lookups.skip_warm_sysroot_bodies;
    let caller_is_fixed = cx.outline.is_fixed(caller.0.krate);
    let mut visiting = HashSet::default();
    let mut push = |item: ItemRef, substs: Substs| push_filtered_callee(cx, filter, (&mut visiting, check_only && caller_is_fixed), (item, substs), out);
    match &typed.tables {
        LazyTables::Poly(poly) => push_poly_callees(typed, poly, include_fixed, &mut push),
        LazyTables::Eager(_) => {
            for (item, substs) in typed.calls() {
                push(item, substs);
            }
        }
    }
}

fn push_filtered_callee(
    cx: &mut Context<'_>,
    filter: &mut ReachFilter,
    (visiting, prune_fixed): (&mut HashSet<ItemRef>, bool),
    (item, substs): Instance,
    out: &mut Vec<(Instance, bool)>,
) {
    if cx.intrinsic(item).is_some() {
        return;
    }
    let mut sealed = false;
    if prune_fixed && cx.outline.is_fixed(item.krate) {
        if !substs_has_non_fixed(cx, substs) {
            return;
        }
        sealed = filter.check_sealed(cx, item, visiting) == Some(true);
    }
    out.push(((item, substs), sealed));
}

fn push_poly_callees(typed: &Typed, poly: &crate::sema::summary::PolyInst, include_fixed: bool, push: &mut impl FnMut(ItemRef, Substs)) {
    if include_fixed {
        for (item, substs) in poly.calls() { push(item, substs); }
    } else {
        for (item, substs) in poly.dynamic_calls() { push(item, substs); }
    }
    for r in &typed.drops {
        if let Res::Fn(item, substs) = *r { push(item, substs); }
    }
    match &typed.shape {
        Shape::Glue(g) => for r in g.drop.iter().chain(&g.parts).chain(&g.resume) {
            if let Res::Fn(item, substs) = *r { push(item, substs); }
        },
        Shape::Untuple(r) => if let Res::Fn(item, substs) = *r { push(item, substs); },
        Shape::Clone(calls) => for r in calls {
            if let Res::Fn(item, substs) = *r { push(item, substs); }
        },
        _ => {}
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
    pre_collect_callees(cx, filter, (caller, typed, include_fixed), &mut pre);
    commit_filter_caller(cx, filter, caller, typed);
    for (callee, sealed) in pre {
        if sealed && !filter.seen_sealed_sysroot.insert(callee.0) {
            continue;
        }
        out.push(callee);
    }
}

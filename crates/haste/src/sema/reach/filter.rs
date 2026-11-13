use std::sync::Arc;

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::body::sig::GenericKind;
use crate::names::{ItemRef, Primitive};

use super::super::Context;
use super::super::impls::head;
use super::super::infer::{LazyTables, Res, Shape, Typed, body_relaxes_sized, maybe_unsized, param_relaxes_sized, type_params};
use super::super::intrinsic::Intrinsic;
use super::super::substs::Substs;
use super::super::summary::{ConstHole, PolyObligation, PolySummary, poly_summary};
use super::super::ty::{Kind, Ty};
use super::Instance;

#[derive(Clone, Copy, PartialEq, Eq)]
struct SealInfo {
    sealed: bool,
    params: u16,
}

impl SealInfo {
    const UNSEALED: Self = Self { sealed: false, params: 0 };
}

#[derive(Default)]
pub(super) struct ReachFilter {
    pub(super) seen_fixed_poly: HashSet<ItemRef>,
    pub(super) seen_sealed_sysroot: HashSet<ItemRef>,
    sealed_cache: HashMap<ItemRef, SealInfo>,
    sealed_log: Vec<(ItemRef, SealInfo)>,
    seen_sealed_log: Vec<ItemRef>,
    dirty: Vec<(ItemRef, SealInfo)>,
}

impl ReachFilter {
    fn record_sealed(&mut self, item: ItemRef, info: SealInfo) {
        if self.sealed_cache.insert(item, info).is_none() {
            self.dirty.push((item, info));
        }
    }

    pub(super) fn mark_seen_sealed(&mut self, item: ItemRef) -> bool {
        let fresh = self.seen_sealed_sysroot.insert(item);
        if fresh {
            self.seen_sealed_log.push(item);
        }
        fresh
    }

    pub(super) fn sync_from(&mut self, global: &Self, synced: &mut (usize, usize)) {
        if global.sealed_log.len() > synced.0 {
            for &(k, v) in &global.sealed_log[synced.0..] {
                self.sealed_cache.entry(k).or_insert(v);
            }
            synced.0 = global.sealed_log.len();
        }
        if global.seen_sealed_log.len() > synced.1 {
            self.seen_sealed_sysroot.extend(global.seen_sealed_log[synced.1..].iter().copied());
            synced.1 = global.seen_sealed_log.len();
        }
    }

    pub(super) fn drain_to(&mut self, global: &mut Self) {
        for (k, v) in self.dirty.drain(..) {
            if global.sealed_cache.insert(k, v).is_none() {
                global.sealed_log.push((k, v));
            }
        }
    }

    fn seal_info(&mut self, cx: &mut Context<'_>, item: ItemRef) -> SealInfo {
        if let Some(&info) = self.sealed_cache.get(&item) {
            return info;
        }
        let mut stack = Vec::new();
        let Some(_) = self.compute_seal(cx, item, &mut stack) else {
            return SealInfo::UNSEALED;
        };
        self.sealed_cache.get(&item).copied().unwrap_or(SealInfo::UNSEALED)
    }

    fn compute_seal(&mut self, cx: &mut Context<'_>, item: ItemRef, stack: &mut Vec<ItemRef>) -> Option<usize> {
        if let Some(&info) = self.sealed_cache.get(&item) {
            return info.sealed.then_some(usize::MAX);
        }
        if let Some(intrinsic) = cx.intrinsic(item) {
            let sealed = !matches!(intrinsic, Intrinsic::Transmute);
            self.record_sealed(item, SealInfo { sealed, params: 0 });
            return sealed.then_some(usize::MAX);
        }
        if !cx.outline.is_fixed(item.krate) || matches!(cx.lang(item), Some("drop_glue" | "clone_fn" | "poll")) {
            self.record_sealed(item, SealInfo::UNSEALED);
            return None;
        }
        if let Some(pos) = stack.iter().position(|&x| x == item) {
            return Some(pos);
        }
        let Some(summary) = poly_summary(cx, item) else {
            self.record_sealed(item, SealInfo::UNSEALED);
            return None;
        };
        if !local_sealed(cx, item, &summary) {
            self.record_sealed(item, SealInfo::UNSEALED);
            return None;
        }
        let my_depth = stack.len();
        stack.push(item);
        let dep = self.seal_callees(cx, &summary, stack);
        stack.pop();
        let Some(dep) = dep else {
            self.record_sealed(item, SealInfo::UNSEALED);
            return None;
        };
        if dep >= my_depth {
            self.record_sealed(item, SealInfo { sealed: true, params: summary.param_count });
            return Some(usize::MAX);
        }
        Some(dep)
    }

    fn seal_callees(&mut self, cx: &mut Context<'_>, summary: &PolySummary, stack: &mut Vec<ItemRef>) -> Option<usize> {
        let mut dep = usize::MAX;
        for &(_, callee, _) in &summary.direct_calls {
            if cx.intrinsic(callee).is_some() {
                continue;
            }
            dep = dep.min(self.compute_seal(cx, callee, stack)?);
        }
        for &(_, hole_item, hole_substs) in &summary.call_holes {
            dep = dep.min(self.call_hole_seal(cx, hole_item, hole_substs, stack)?);
        }
        Some(dep)
    }

    fn call_hole_seal(&mut self, cx: &mut Context<'_>, item: ItemRef, substs: Substs, stack: &mut Vec<ItemRef>) -> Option<usize> {
        let tr = cx.owner_trait(item)?;
        if !cx.outline.is_fixed(tr.krate) {
            return None;
        }
        let tg = cx.trait_generics(tr).ok()?;
        let count = 1 + tg.1.params.iter().filter(|g| g.kind != GenericKind::Lifetime).count();
        let slice = cx.types.substs(substs).to_vec();
        if slice.len() < count || slice[1..count].iter().any(|&ty| ty_has_non_fixed(cx, ty)) {
            return None;
        }
        if !ok_call_hole_head(cx, slice[0]) {
            return None;
        }
        let head_key = head(cx.types.kind(slice[0]))?;
        let holders = fixed_holders(cx, &slice[..count], tr.krate);
        let blocks = cx.trait_impls(tr, Some(head_key), &holders);
        if blocks.is_empty() {
            return None;
        }
        let name = cx.name(item);
        let mut dep = usize::MAX;
        for block in blocks {
            if cx.reserved(block) {
                continue;
            }
            let member = cx.member(block, name).unwrap_or(item);
            dep = dep.min(self.compute_seal(cx, member, stack)?);
        }
        Some(dep)
    }
}

fn ok_call_hole_head(cx: &Context<'_>, ty: Ty) -> bool {
    match *cx.types.kind(ty) {
        Kind::Prim(_) | Kind::Never | Kind::Ptr(..) | Kind::Slice(..) | Kind::Array(..) | Kind::FnPtr(..) => true,
        Kind::Ref(..) | Kind::Tuple(..) => !ty_has_non_fixed(cx, ty),
        Kind::Adt(adt, _) => {
            cx.outline.is_fixed(adt.krate)
                && ((cx.lang(adt) != Some("owned_box") && cx.attr_at(adt, "fundamental").is_none()) || !ty_has_non_fixed(cx, ty))
        }
        _ => false,
    }
}

fn fixed_holders(cx: &Context<'_>, root_tys: &[Ty], tr_krate: u32) -> Vec<u32> {
    let mut holders = vec![tr_krate];
    let mut work: Vec<Ty> = root_tys.to_vec();
    while let Some(ty) = work.pop() {
        let kind = cx.types.kind(ty).clone();
        if let Kind::Adt(def, _) | Kind::Dyn(def, ..) = kind
            && cx.outline.is_fixed(def.krate)
            && !holders.contains(&def.krate)
        {
            holders.push(def.krate);
        }
        work.extend(kind.children(&cx.types));
    }
    holders
}

fn local_sealed(cx: &mut Context<'_>, item: ItemRef, summary: &PolySummary) -> bool {
    summary.projections.is_empty()
        && summary.obligations.iter().all(|o| matches!(o, PolyObligation::Prove { .. }))
        && summary.op_holes.is_empty()
        && summary.const_holes.iter().all(|h| matches!(h, ConstHole::Intrinsic { .. }))
        && summary.template.closures.is_empty()
        && summary.layout_tys.iter().all(|&ty| layout_ty_infallible(cx, item, ty))
        && summary.drop_tys.iter().all(|&ty| layout_ty_infallible(cx, item, ty))
}

fn param_is_sized(cx: &mut Context<'_>, item: ItemRef, idx: u16) -> bool {
    let Ok(sig) = cx.signature(item) else { return false };
    let idx = usize::from(idx);
    if let Some(owner) = cx.owner_impl(item) {
        let Ok(h) = cx.impl_header(owner) else { return false };
        let outer = type_params(&h.generics).count();
        if idx < outer {
            return !nth_param_relaxes(&h.body, &cx.impl_file(owner).source, &h.predicates, &h.generics, idx);
        }
        return !nth_param_relaxes(&sig.body, &cx.file(item).source, &sig.predicates, &sig.generics, idx - outer);
    }
    if let Some(tr) = cx.owner_trait(item) {
        if idx == 0 {
            return false;
        }
        let Ok(tg) = cx.trait_generics(tr) else { return false };
        let outer = type_params(&tg.1.params).count();
        if idx - 1 < outer {
            return !nth_param_relaxes(&tg.0, &cx.file(tr).source, &tg.1.predicates, &tg.1.params, idx - 1);
        }
        return !nth_param_relaxes(&sig.body, &cx.file(item).source, &sig.predicates, &sig.generics, idx - 1 - outer);
    }
    !nth_param_relaxes(&sig.body, &cx.file(item).source, &sig.predicates, &sig.generics, idx)
}

fn nth_param_relaxes(
    arena: &crate::body::Body,
    src: &crate::outline::source::Source,
    predicates: &[crate::body::sig::Predicate],
    generics: &[crate::body::sig::Generic],
    idx: usize,
) -> bool {
    type_params(generics).nth(idx).is_none_or(|g| param_relaxes_sized(arena, src, predicates, g))
}

fn ty_is_sized(cx: &mut Context<'_>, item: ItemRef, ty: Ty) -> bool {
    match *cx.types.kind(ty) {
        Kind::Prim(p) => !matches!(p, Primitive::Str),
        Kind::Ref(..) | Kind::Ptr(..) | Kind::FnDef(..) | Kind::FnPtr(..) | Kind::Never | Kind::Array(..) => true,
        Kind::Pat(inner, ..) => ty_is_sized(cx, item, inner),
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
        Kind::Slice(elem) => sized_layout_infallible(cx, item, elem),
        Kind::Array(elem, len) => matches!(cx.types.kind(len), Kind::Value(n) if *n <= 1 || !cx.types.has_params(elem)) && sized_layout_infallible(cx, item, elem),
        Kind::Pat(inner, ..) => !cx.types.has_params(inner) && sized_layout_infallible(cx, item, inner),
        Kind::Tuple(elems) => tuple_layout_infallible(cx, item, elems),
        Kind::Adt(adt, args) => adt_layout_infallible(cx, item, adt, args),
        _ => false,
    }
}

fn sized_layout_infallible(cx: &mut Context<'_>, item: ItemRef, ty: Ty) -> bool {
    ty_is_sized(cx, item, ty) && layout_ty_infallible(cx, item, ty)
}

fn tuple_layout_infallible(cx: &mut Context<'_>, item: ItemRef, elems: Substs) -> bool {
    let elems = cx.types.substs(elems).to_vec();
    let non_last = elems.len().saturating_sub(1);
    elems[..non_last].iter().all(|&e| ty_is_sized(cx, item, e)) && elems.iter().all(|&e| layout_ty_infallible(cx, item, e))
}

fn adt_layout_infallible(cx: &mut Context<'_>, item: ItemRef, adt: ItemRef, args: Substs) -> bool {
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
    let mut push = |item: ItemRef, substs: Substs| push_filtered_callee(cx, filter, check_only && caller_is_fixed, (item, substs), out);
    match &typed.tables {
        LazyTables::Poly(poly) => push_poly_callees(typed, poly.calls(), poly.dynamic_calls(), include_fixed, &mut push),
        LazyTables::CheckOnly(check) => push_poly_callees(typed, check.calls(), check.dynamic_calls(), include_fixed, &mut push),
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
    prune_fixed: bool,
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
        sealed = can_prune_fixed(cx, filter, (item, substs));
        if sealed && filter.seen_sealed_sysroot.contains(&item) {
            return;
        }
    }
    out.push(((item, substs), sealed));
}

fn can_prune_fixed(cx: &mut Context<'_>, filter: &mut ReachFilter, (item, substs): Instance) -> bool {
    let info = filter.seal_info(cx, item);
    if !info.sealed {
        return false;
    }
    let slice = cx.types.substs(substs);
    slice.len() == usize::from(info.params) && !slice.iter().any(|&ty| matches!(cx.types.kind(ty), Kind::Never | Kind::Dyn(..)))
}

fn push_poly_callees(
    typed: &Typed,
    all: impl Iterator<Item = (ItemRef, Substs)>,
    dynamic: impl Iterator<Item = (ItemRef, Substs)>,
    include_fixed: bool,
    push: &mut impl FnMut(ItemRef, Substs),
) {
    if include_fixed {
        for (item, substs) in all { push(item, substs); }
    } else {
        for (item, substs) in dynamic { push(item, substs); }
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
    if let Ok(typed) = typed && matches!(typed.tables, LazyTables::Poly(_) | LazyTables::CheckOnly(_)) {
        filter.seen_fixed_poly.insert(caller.0);
        if cx.outline.is_fixed(caller.0.krate) {
            filter.mark_seen_sealed(caller.0);
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
        if sealed && !filter.mark_seen_sealed(callee.0) {
            continue;
        }
        out.push(callee);
    }
}

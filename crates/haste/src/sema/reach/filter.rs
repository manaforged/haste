use std::sync::Arc;

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::body::Bound;
use crate::body::sig::GenericKind;
use crate::names::{ItemRef, Primitive};

use super::super::Context;
use super::super::infer::{LazyTables, Res, Shape, Typed, maybe_unsized};
use super::super::substs::Substs;
use super::super::summary::SummaryKind;
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
        let Some(kind) = cx.store.lookups.summaries.get_ready(&item) else {
            visiting.remove(&item);
            return None;
        };
        let SummaryKind::Poly(summary) = kind else {
            visiting.remove(&item);
            self.sealed_cache.insert(item, false);
            return Some(false);
        };
        let local_ok = summary.projections.is_empty()
            && summary.obligations.is_empty()
            && summary.call_holes.is_empty()
            && summary.op_holes.is_empty()
            && summary.const_holes.is_empty()
            && summary.drop_tys.is_empty()
            && summary.template.closures.is_empty()
            && summary.layout_tys.iter().all(|&ty| layout_ty_infallible(cx, item, ty));
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

fn has_relaxed_bound(bounds: &[Bound]) -> bool {
    bounds.iter().any(|b| matches!(b, Bound::Trait { maybe: true, .. }))
}

fn param_is_sized(cx: &mut Context<'_>, item: ItemRef, idx: u16) -> bool {
    if cx.owner_trait(item).is_some() && idx == 0 {
        return false;
    }
    let Ok(sig) = cx.signature(item) else { return false };
    if has_relaxed_bound(&sig.body.bounds) {
        return false;
    }
    if let Some(owner) = cx.owner_impl(item)
        && !cx.impl_header(owner).is_ok_and(|h| !has_relaxed_bound(&h.body.bounds))
    {
        return false;
    }
    if let Some(tr) = cx.owner_trait(item)
        && !cx.trait_generics(tr).is_ok_and(|tg| !has_relaxed_bound(&tg.0.bounds))
    {
        return false;
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
            let args = cx.types.substs(args).to_vec();
            if !args.iter().all(|&arg| layout_ty_infallible(cx, item, arg)) {
                return false;
            }
            if !has_relaxed_bound(&def.body.bounds) || maybe_unsized(cx, adt).unwrap_or(true) {
                args.iter().all(|&arg| ty_is_sized(cx, item, arg))
            } else {
                true
            }
        }
        _ => false,
    }
}

pub(super) fn collect_filtered_callees(
    cx: &mut Context<'_>,
    filter: &mut ReachFilter,
    caller: Instance,
    typed: &Result<Arc<Typed>, String>,
    out: &mut Vec<Instance>,
) {
    let Ok(typed) = typed else { return };
    let check_only = cx.store.lookups.skip_warm_sysroot_bodies;
    let mut visiting = HashSet::default();
    let mut push_callee = |cx: &mut Context<'_>, filter: &mut ReachFilter, item: ItemRef, substs: Substs| {
        if cx.intrinsic(item).is_some() {
            return;
        }
        if check_only && cx.outline.is_fixed(item.krate) && filter.seen_sealed_sysroot.contains(&item) && filter.check_sealed(cx, item, &mut visiting) == Some(true) {
            return;
        }
        out.push((item, substs));
    };
    match &typed.tables {
        LazyTables::Poly(poly) => {
            if filter.seen_fixed_poly.insert(caller.0) {
                for (item, substs) in poly.calls() { push_callee(cx, filter, item, substs); }
            } else {
                for (item, substs) in poly.dynamic_calls() { push_callee(cx, filter, item, substs); }
            }
            if cx.outline.is_fixed(caller.0.krate) {
                filter.seen_sealed_sysroot.insert(caller.0);
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

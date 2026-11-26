use crate::body::Body;
use crate::body::sig::{Generic, GenericKind, Predicate};
use crate::names::{ItemRef, Primitive};
use crate::outline::source::Source;
use crate::sema::Context;
use crate::sema::infer::{body_relaxes_sized, maybe_unsized, param_relaxes_sized, type_params};
use crate::sema::substs::Substs;
use crate::sema::summary::{ConstHole, PolyObligation, PolySummary};
use crate::sema::ty::{Kind, Ty};

pub(super) fn propagate_callee_sized(
    cx: &mut Context<'_>,
    caller: ItemRef,
    substs: Substs,
    mut mask: u64,
    req_sized: &mut u64,
) -> bool {
    if mask == 0 {
        return true;
    }
    let slice = cx.types.substs(substs).to_vec();
    while mask != 0 {
        let idx = mask.trailing_zeros() as usize;
        mask &= mask - 1;
        let Some(&arg) = slice.get(idx) else {
            return false;
        };
        if !ty_is_sized(cx, caller, arg, req_sized) {
            return false;
        }
    }
    true
}

pub(super) fn local_sealed(
    cx: &mut Context<'_>,
    item: ItemRef,
    summary: &PolySummary,
) -> Option<u64> {
    if !summary.projections.is_empty()
        || !summary
            .obligations
            .iter()
            .all(|o| matches!(o, PolyObligation::Prove { .. }))
        || !summary.op_holes.is_empty()
        || !summary
            .const_holes
            .iter()
            .all(|h| matches!(h, ConstHole::Intrinsic { .. }))
        || !summary.template.closures.is_empty()
    {
        return None;
    }
    let mut req_sized = 0u64;
    for &ty in summary.layout_tys.iter().chain(&summary.drop_tys) {
        if !layout_ty_infallible(cx, item, ty, &mut req_sized) {
            return None;
        }
    }
    Some(req_sized)
}

fn param_is_sized(cx: &mut Context<'_>, item: ItemRef, idx: u16) -> bool {
    let Ok(sig) = cx.signature(item) else {
        return false;
    };
    let idx = usize::from(idx);
    if let Some(owner) = cx.owner_impl(item) {
        let Ok(h) = cx.impl_header(owner) else {
            return false;
        };
        let outer = type_params(&h.generics).count();
        if idx < outer {
            return !nth_param_relaxes(
                &h.body,
                &cx.impl_file(owner).source,
                &h.predicates,
                &h.generics,
                idx,
            );
        }
        return !nth_param_relaxes(
            &sig.body,
            &cx.file(item).source,
            &sig.predicates,
            &sig.generics,
            idx - outer,
        );
    }
    if let Some(tr) = cx.owner_trait(item) {
        if idx == 0 {
            return false;
        }
        let Ok(tg) = cx.trait_generics(tr) else {
            return false;
        };
        let outer = type_params(&tg.1.params).count();
        if idx - 1 < outer {
            return !nth_param_relaxes(
                &tg.0,
                &cx.file(tr).source,
                &tg.1.predicates,
                &tg.1.params,
                idx - 1,
            );
        }
        return !nth_param_relaxes(
            &sig.body,
            &cx.file(item).source,
            &sig.predicates,
            &sig.generics,
            idx - 1 - outer,
        );
    }
    !nth_param_relaxes(
        &sig.body,
        &cx.file(item).source,
        &sig.predicates,
        &sig.generics,
        idx,
    )
}

fn nth_param_relaxes(
    arena: &Body,
    src: &Source,
    predicates: &[Predicate],
    generics: &[Generic],
    idx: usize,
) -> bool {
    type_params(generics)
        .nth(idx)
        .is_none_or(|g| param_relaxes_sized(arena, src, predicates, g))
}

fn require_param_sized(cx: &mut Context<'_>, item: ItemRef, idx: u16, req_sized: &mut u64) -> bool {
    if param_is_sized(cx, item, idx) {
        return true;
    }
    if idx < 64 {
        *req_sized |= 1u64 << idx;
        return true;
    }
    false
}

fn ty_is_sized(cx: &mut Context<'_>, item: ItemRef, ty: Ty, req_sized: &mut u64) -> bool {
    match *cx.types.kind(ty) {
        Kind::Prim(p) => !matches!(p, Primitive::Str),
        Kind::Ref(..)
        | Kind::Ptr(..)
        | Kind::FnDef(..)
        | Kind::FnPtr(..)
        | Kind::UnsafeFnPtr(..)
        | Kind::Never
        | Kind::Array(..) => true,
        Kind::Pat(inner, ..) => ty_is_sized(cx, item, inner, req_sized),
        Kind::Slice(..) | Kind::Dyn(..) => false,
        Kind::Param(idx) => require_param_sized(cx, item, idx, req_sized),
        Kind::Tuple(elems) => cx
            .types
            .substs(elems)
            .last()
            .copied()
            .is_none_or(|last| ty_is_sized(cx, item, last, req_sized)),
        Kind::Adt(adt, args) => {
            if !cx.outline.is_fixed(adt.krate) {
                return false;
            }
            if !maybe_unsized(cx, adt).unwrap_or(true) {
                return true;
            }
            cx.types
                .substs(args)
                .to_vec()
                .into_iter()
                .all(|arg| ty_is_sized(cx, item, arg, req_sized))
        }
        _ => false,
    }
}

pub(super) fn concrete_ty_is_sized(cx: &mut Context<'_>, ty: Ty) -> bool {
    match *cx.types.kind(ty) {
        Kind::Prim(p) => !matches!(p, Primitive::Str),
        Kind::Ref(..)
        | Kind::Ptr(..)
        | Kind::FnDef(..)
        | Kind::FnPtr(..)
        | Kind::UnsafeFnPtr(..)
        | Kind::Array(..)
        | Kind::Closure(..) => true,
        Kind::Pat(inner, ..) => concrete_ty_is_sized(cx, inner),
        Kind::Tuple(elems) => cx
            .types
            .substs(elems)
            .last()
            .copied()
            .is_none_or(|last| concrete_ty_is_sized(cx, last)),
        Kind::Adt(adt, _) => !maybe_unsized(cx, adt).unwrap_or(true),
        _ => false,
    }
}

fn layout_ty_infallible(cx: &mut Context<'_>, item: ItemRef, ty: Ty, req_sized: &mut u64) -> bool {
    match *cx.types.kind(ty) {
        Kind::Prim(_)
        | Kind::Ref(..)
        | Kind::Ptr(..)
        | Kind::FnDef(..)
        | Kind::FnPtr(..)
        | Kind::UnsafeFnPtr(..)
        | Kind::Never => true,
        Kind::Param(idx) => require_param_sized(cx, item, idx, req_sized),
        Kind::Slice(elem) => sized_layout_infallible(cx, item, elem, req_sized),
        Kind::Array(elem, len) => {
            matches!(cx.types.kind(len), Kind::Value(n) if *n <= 1 || !cx.types.has_params(elem))
                && sized_layout_infallible(cx, item, elem, req_sized)
        }
        Kind::Pat(inner, ..) => {
            !cx.types.has_params(inner) && sized_layout_infallible(cx, item, inner, req_sized)
        }
        Kind::Tuple(elems) => tuple_layout_infallible(cx, item, elems, req_sized),
        Kind::Adt(adt, args) => adt_layout_infallible(cx, item, adt, args, req_sized),
        _ => false,
    }
}

fn sized_layout_infallible(
    cx: &mut Context<'_>,
    item: ItemRef,
    ty: Ty,
    req_sized: &mut u64,
) -> bool {
    ty_is_sized(cx, item, ty, req_sized) && layout_ty_infallible(cx, item, ty, req_sized)
}

fn tuple_layout_infallible(
    cx: &mut Context<'_>,
    item: ItemRef,
    elems: Substs,
    req_sized: &mut u64,
) -> bool {
    let elems = cx.types.substs(elems).to_vec();
    let non_last = elems.len().saturating_sub(1);
    elems[..non_last]
        .iter()
        .all(|&e| ty_is_sized(cx, item, e, req_sized))
        && elems
            .iter()
            .all(|&e| layout_ty_infallible(cx, item, e, req_sized))
}

fn adt_layout_infallible(
    cx: &mut Context<'_>,
    item: ItemRef,
    adt: ItemRef,
    args: Substs,
    req_sized: &mut u64,
) -> bool {
    if !cx.outline.is_fixed(adt.krate) {
        return false;
    }
    let repr = cx.repr(adt);
    if repr.flags.contains(rustc_abi::ReprFlags::IS_SIMD) || repr.pack.is_some() {
        return false;
    }
    let Ok(def) = cx.adt_def(adt) else {
        return false;
    };
    if def
        .generics
        .params
        .iter()
        .any(|g| matches!(g.kind, GenericKind::Const(_)))
    {
        return false;
    }
    let any_relaxed = body_relaxes_sized(&def.body, &cx.file(adt).source);
    let adt_unsized = any_relaxed && maybe_unsized(cx, adt).unwrap_or(true);
    let src = &cx.file(adt).source;
    let args = cx.types.substs(args).to_vec();
    let type_params: Vec<_> = def
        .generics
        .params
        .iter()
        .filter(|g| g.kind != GenericKind::Lifetime)
        .collect();
    for (i, &arg) in args.iter().enumerate() {
        let param_relaxed = type_params
            .get(i)
            .is_some_and(|g| param_relaxes_sized(&def.body, src, &def.generics.predicates, g));
        if !adt_unsized && param_relaxed {
            continue;
        }
        if !layout_ty_infallible(cx, item, arg, req_sized) || !ty_is_sized(cx, item, arg, req_sized)
        {
            return false;
        }
    }
    true
}

mod compat;
mod env;
mod solve;

pub(super) use compat::check_trait_impl_block;

use std::sync::Arc;

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::body::ExprId;
use crate::names::ItemRef;

use super::super::Context;
use super::super::impls::ImplRef;
use super::super::intrinsic::Intrinsic;
use super::super::substs::{ProjId, Substs};
use super::super::summary::{ConstHole, PolyObligation, PolySummary, SummaryKind};
use super::super::ty::{Kind, Ty, Types};
use super::{Infer, Res, Typed};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::sema) enum PolyPhase {
    #[default]
    HeaderInit,
    Active,
    Fallback,
}

#[derive(Default)]
pub(in crate::sema) struct PolyState {
    pub phase: PolyPhase,
    pub unsized_params: HashSet<u16>,
    pub assumptions: Vec<(ItemRef, Vec<Ty>)>,
    pub assoc_eqs: Vec<(ItemRef, Ty, Substs, String, Ty)>,
    pub fn_bounds: Vec<(Ty, ItemRef, Ty)>,
    pub obligations: Vec<PolyObligation>,
    pub projections: Vec<ProjId>,
    pub proj_set: HashSet<ProjId>,
    pub offset_holes: Vec<ConstHole>,
    pub method_traits: HashMap<ExprId, ItemRef>,
    pub checked_impls: HashSet<ImplRef>,
    pub skip_compat: bool,
}

fn collect_poly_traits(cx: &Context<'_>, poly: Option<&PolyState>, out: &mut Vec<ItemRef>) {
    let Some(poly) = poly else { return };
    for &(of, _) in &poly.assumptions {
        if !out.contains(&of) {
            out.push(of);
        }
    }
    for ob in &poly.obligations {
        if let PolyObligation::Prove { of, .. } = *ob
            && !out.contains(&of)
        {
            out.push(of);
        }
    }
    for &method in poly.method_traits.values() {
        if let Some(of) = cx.owner_trait(method)
            && !out.contains(&of)
        {
            out.push(of);
        }
    }
}

pub(in crate::sema) fn build_poly_summary(
    cx: &mut Context<'_>,
    item: ItemRef,
    param_count: u16,
) -> SummaryKind {
    let mut poly_traits = Vec::new();
    let summary = build_poly_summary_inner(cx, item, param_count, &mut poly_traits);
    if matches!(summary, SummaryKind::Error(_)) || cx.outline.is_fixed(item.krate) {
        return summary;
    }
    if let Err(err) = compat::check_item_trait_impls(cx, item, &poly_traits) {
        return SummaryKind::Error(crate::diag::format_error(cx.name(item), &err));
    }
    summary
}

fn build_poly_summary_inner(
    cx: &mut Context<'_>,
    item: ItemRef,
    param_count: u16,
    poly_traits: &mut Vec<ItemRef>,
) -> SummaryKind {
    let before = cx.quiet_mark();
    if cx.has_attr(item, "define_opaque") {
        return SummaryKind::MonomorphicFallback;
    }
    let (Ok(sig), Ok(body), Ok(home)) = (cx.signature(item), cx.body(item), cx.module_of(item))
    else {
        return SummaryKind::MonomorphicFallback;
    };
    if sig.is_async
        || (param_count == 0
            && sig
                .body
                .tys
                .iter()
                .any(|ty| matches!(ty, crate::body::Ty::ImplTrait(..))))
    {
        return SummaryKind::MonomorphicFallback;
    }
    let is_user = !cx.outline.is_fixed(item.krate);
    let src = cx.body_src(item);
    let mut infer = Infer::start(cx, (&body, &sig, src), home);
    if infer.init_poly_for_item(item, param_count).is_err() {
        return SummaryKind::MonomorphicFallback;
    }
    infer.out.exprs.resize(body.exprs.len(), Ty::NONE);
    if let Err(err) = infer.run(false) {
        let fell_back = infer.poly_fell_back();
        let took = infer.poly.take();
        collect_poly_traits(infer.cx, took.as_deref(), poly_traits);
        let hints = took.map(|poly| poly.method_traits).unwrap_or_default();
        drop(infer);
        if is_user && !fell_back {
            return SummaryKind::Error(crate::diag::format_error(cx.name(item), &err));
        }
        return fallback_kind(hints);
    }
    if is_user
        && !infer.poly_fell_back()
        && let Err(err) = infer.check_const_body_overflow(item)
    {
        drop(infer);
        return SummaryKind::Error(crate::diag::format_error(cx.name(item), &err));
    }
    collect_poly_traits(infer.cx, infer.poly.as_deref(), poly_traits);
    let pre_hints = infer
        .poly
        .as_ref()
        .map(|poly| poly.method_traits.clone())
        .unwrap_or_default();
    let (mut typed, poly, clean) = match infer.finish_poly() {
        Ok(ok) => ok,
        Err((err, fell_back)) => {
            if is_user && !fell_back {
                return SummaryKind::Error(crate::diag::format_error(cx.name(item), &err));
            }
            return fallback_kind(pre_hints);
        }
    };
    collect_poly_traits(cx, poly.as_deref(), poly_traits);
    if cx.quiet_mark() != before
        || (!typed.coercions.is_empty() && super::object::fill_drops(cx, &mut typed).is_err())
    {
        return fallback_kind(poly.map(|poly| poly.method_traits).unwrap_or(pre_hints));
    }
    if is_user
        && clean
        && poly
            .as_ref()
            .is_some_and(|p| p.phase != PolyPhase::Fallback)
        && matches!(typed.shape, super::Shape::Body)
        && let Err(err) = super::super::borrow::check(cx, item, (&body, &sig), src, &typed)
    {
        return SummaryKind::Error(crate::diag::format_error(cx.name(item), &err));
    }
    finalize_summary(cx, typed, poly, clean, param_count)
}

fn fallback_kind(hints: HashMap<ExprId, ItemRef>) -> SummaryKind {
    if hints.is_empty() {
        SummaryKind::MonomorphicFallback
    } else {
        SummaryKind::MonomorphicHints(Arc::new(hints))
    }
}

pub(super) fn zonk_poly_state(infer: &mut Infer<'_, '_>, poly: &mut PolyState) {
    poly.obligations = zonk_obligations(infer, &mut poly.obligations);
    let raw_offsets = std::mem::take(&mut poly.offset_holes);
    poly.offset_holes = zonk_offset_holes(infer, raw_offsets);
}

fn finalize_summary(
    cx: &Context<'_>,
    typed: Typed,
    poly: Option<Box<PolyState>>,
    clean: bool,
    param_count: u16,
) -> SummaryKind {
    let Some(mut poly) = poly else {
        return SummaryKind::MonomorphicFallback;
    };
    if poly.phase == PolyPhase::Fallback || !clean || has_param_closure(cx, &typed) {
        return fallback_kind(poly.method_traits);
    }
    collect_remaining_projs(&cx.types, &mut poly, &typed);
    let mut const_holes = poly.offset_holes;
    let mut layout_tys = Vec::new();
    let mut layout_seen = HashSet::default();
    collect_layouts_and_consts(
        cx,
        &typed,
        &mut layout_tys,
        &mut layout_seen,
        &mut const_holes,
    );
    let super::LazyTables::Eager(template) = typed.tables else {
        return fallback_kind(poly.method_traits);
    };
    SummaryKind::Poly(Arc::new(PolySummary {
        param_count,
        template,
        locals: typed.locals.into_boxed_slice(),
        params: typed.params.into_boxed_slice(),
        ret: typed.ret,
        obligations: poly.obligations.into_boxed_slice(),
        projections: poly.projections.into_boxed_slice(),
        parametric_tys: Box::new([]),
        layout_tys: layout_tys.into_boxed_slice(),
        const_holes: const_holes.into_boxed_slice(),
        fixed_calls: Box::default(),
        direct_calls: Box::default(),
        call_holes: Box::default(),
        op_holes: Box::default(),
        fixed_drops: Box::default(),
        drop_tys: Box::default(),
    }))
}

fn has_param_closure(cx: &Context<'_>, typed: &Typed) -> bool {
    typed.closures.as_slice().iter().any(|(_, c)| {
        c.captures
            .iter()
            .any(|cap| cx.types.has_params(typed.locals[cap.local as usize].ty))
    })
}

fn zonk_obligations(
    infer: &mut Infer<'_, '_>,
    raw: &mut Vec<PolyObligation>,
) -> Vec<PolyObligation> {
    let mut out = Vec::with_capacity(raw.len());
    for entry in raw.drain(..) {
        let mapped = match entry {
            PolyObligation::Prove { of, self_ty, args } => {
                let self_ty = infer.zonk(self_ty);
                let mapped_args = infer.map_substs(args, Infer::zonk);
                PolyObligation::Prove {
                    of,
                    self_ty,
                    args: mapped_args,
                }
            }
            PolyObligation::Call { expr, found, tuple } => PolyObligation::Call {
                expr,
                found: infer.zonk(found),
                tuple: infer.zonk(tuple),
            },
        };
        if !out.contains(&mapped) {
            out.push(mapped);
        }
    }
    out
}

fn zonk_offset_holes(infer: &mut Infer<'_, '_>, raw: Vec<ConstHole>) -> Vec<ConstHole> {
    raw.into_iter()
        .map(|hole| match hole {
            ConstHole::OffsetOf { expr, steps } => {
                let mapped: Vec<(Ty, u32)> = steps
                    .iter()
                    .map(|&(ty, idx)| (infer.zonk(ty), idx))
                    .collect();
                ConstHole::OffsetOf {
                    expr,
                    steps: mapped.into_boxed_slice(),
                }
            }
            other => other,
        })
        .collect()
}

fn peel_pointer_aggregate(types: &Types, mut ty: Ty) -> Option<Ty> {
    while let Kind::Ref(_, next) | Kind::Ptr(_, next) = *types.kind(ty) {
        ty = next;
    }
    matches!(
        types.kind(ty),
        Kind::Adt(..) | Kind::Tuple(_) | Kind::Array(..)
    )
    .then_some(ty)
}

fn collect_layouts_and_consts(
    cx: &Context<'_>,
    typed: &Typed,
    layout_tys: &mut Vec<Ty>,
    layout_seen: &mut HashSet<Ty>,
    const_holes: &mut Vec<ConstHole>,
) {
    let mut push_layout = |ty: Ty| {
        if ty != Ty::NONE && cx.types.has_params(ty) && layout_seen.insert(ty) {
            layout_tys.push(ty);
        }
        if ty != Ty::NONE
            && let Some(inner) = peel_pointer_aggregate(&cx.types, ty)
            && cx.types.has_params(inner)
            && layout_seen.insert(inner)
        {
            layout_tys.push(inner);
        }
    };
    for hole in const_holes.iter() {
        if let ConstHole::OffsetOf { steps, .. } = hole {
            steps.iter().for_each(|&(ty, _)| push_layout(ty));
        }
    }
    typed.locals.iter().for_each(|local| push_layout(local.ty));
    typed.exprs.iter().copied().for_each(&mut push_layout);
    if let Some(ret) = typed.ret {
        push_layout(ret);
    }
    collect_res_holes(cx, typed, &mut push_layout, const_holes);
}

fn collect_res_holes(
    cx: &Context<'_>,
    typed: &Typed,
    push_layout: &mut impl FnMut(Ty),
    const_holes: &mut Vec<ConstHole>,
) {
    for res in typed.res.values().chain(typed.pat_res.values()) {
        if let &Res::Const(ty) = res
            && cx.types.has_params(ty)
        {
            let hole = ConstHole::Value(ty);
            if !const_holes.contains(&hole) {
                const_holes.push(hole);
            }
        }
    }
    for (item, substs) in typed.calls() {
        let Some(intrinsic) = cx.intrinsic(item) else {
            continue;
        };
        for &arg in cx.types.substs(substs) {
            push_layout(arg);
        }
        if cx.types.substs_have_params(substs)
            && matches!(
                intrinsic,
                Intrinsic::SizeOf
                    | Intrinsic::AlignOf
                    | Intrinsic::NeedsDrop
                    | Intrinsic::TypeId
                    | Intrinsic::TypeName
            )
        {
            let hole = ConstHole::Intrinsic { item, substs };
            if !const_holes.contains(&hole) {
                const_holes.push(hole);
            }
        }
    }
}

fn push_proj_topo(types: &Types, state: &mut PolyState, seen: &mut HashSet<Ty>, id: ProjId) {
    if state.proj_set.contains(&id) {
        return;
    }
    let def = types.proj(id);
    visit_ty_projs(types, state, seen, def.self_ty);
    for &arg in types.substs(def.args) {
        visit_ty_projs(types, state, seen, arg);
    }
    for &arg in types.substs(def.own) {
        visit_ty_projs(types, state, seen, arg);
    }
    if state.proj_set.insert(id) {
        state.projections.push(id);
    }
}

fn visit_ty_projs(types: &Types, state: &mut PolyState, seen: &mut HashSet<Ty>, ty: Ty) {
    if !types.has_projs(ty) || !seen.insert(ty) {
        return;
    }
    let kind = types.kind(ty).clone();
    match kind {
        Kind::Proj(inner) => push_proj_topo(types, state, seen, inner),
        Kind::Closure(_, args, _) => types
            .substs(args)
            .iter()
            .for_each(|&arg| visit_ty_projs(types, state, seen, arg)),
        _ => kind
            .children(types)
            .for_each(|child| visit_ty_projs(types, state, seen, child)),
    }
}

fn collect_remaining_projs(types: &Types, state: &mut PolyState, typed: &Typed) {
    let mut seen = HashSet::default();
    let res_ty = |r: &Res, seen: &mut HashSet<Ty>, st: &mut PolyState| match *r {
        Res::Const(t) => visit_ty_projs(types, st, seen, t),
        Res::Fn(_, s) | Res::Ctor(_, _, s) => types
            .substs(s)
            .iter()
            .for_each(|&t| visit_ty_projs(types, st, seen, t)),
        Res::Local(_) => {}
    };
    let tys = typed
        .locals
        .iter()
        .map(|l| l.ty)
        .chain(typed.ret)
        .chain(typed.exprs.iter().copied())
        .chain(typed.pats.values().copied())
        .chain(typed.closures.values().map(|c| c.ret));
    let flow_tys = typed
        .fors
        .values()
        .flat_map(|f| [f.iter, f.option])
        .chain(typed.tries.values().map(|t| t.flow))
        .chain(typed.awaits.values().flat_map(|a| [a.future, a.polled]));
    let hole_tys: Vec<Ty> = state
        .obligations
        .iter()
        .flat_map(|o| match *o {
            PolyObligation::Prove { self_ty, args, .. } => {
                let mut v = types.substs(args).to_vec();
                v.push(self_ty);
                v
            }
            PolyObligation::Call { found, tuple, .. } => vec![found, tuple],
        })
        .chain(state.offset_holes.iter().flat_map(|h| match h {
            ConstHole::OffsetOf { steps, .. } => steps.iter().map(|&(t, _)| t).collect::<Vec<_>>(),
            _ => Vec::new(),
        }))
        .collect();
    tys.chain(flow_tys)
        .chain(hole_tys)
        .for_each(|ty| visit_ty_projs(types, state, &mut seen, ty));
    let direct = typed
        .res
        .values()
        .chain(typed.pat_res.values())
        .chain(typed.derefs.values())
        .chain(typed.operators.values())
        .chain(typed.try_blocks.values())
        .chain(typed.box_frees.values());
    let flow_res = typed
        .fors
        .values()
        .flat_map(|f| [&f.into_iter, &f.next])
        .chain(
            typed
                .tries
                .values()
                .flat_map(|t| [&t.branch, &t.from_residual]),
        )
        .chain(
            typed
                .awaits
                .values()
                .flat_map(|a| [&a.into_future, &a.poll]),
        );
    direct
        .chain(flow_res)
        .for_each(|r| res_ty(r, &mut seen, state));
    for c in typed.coercions.values() {
        match c {
            super::Coercion::Len(t) => visit_ty_projs(types, state, &mut seen, *t),
            super::Coercion::Reify(r) => res_ty(r, &mut seen, state),
            super::Coercion::Deref(steps, to) => {
                steps.iter().for_each(|(t, r)| {
                    visit_ty_projs(types, state, &mut seen, *t);
                    r.iter().for_each(|r| res_ty(r, &mut seen, state));
                });
                visit_ty_projs(types, state, &mut seen, *to);
            }
            super::Coercion::Field { target, inner } => {
                visit_ty_projs(types, state, &mut seen, *target);
                if let super::Coercion::Len(t) = inner.leaf() {
                    visit_ty_projs(types, state, &mut seen, *t);
                }
            }
            _ => {}
        }
    }
}

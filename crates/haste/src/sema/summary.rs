pub mod codec;
mod eval;
pub(super) mod frozen;
mod inst;
mod prepare;
pub(super) mod thaw;

pub use inst::{PolyCheck, PolyInst};

use std::sync::{Arc, OnceLock};

use anyhow::Result;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::body::{BinOp, ExprId};
use crate::names::ItemRef;
use crate::outline::item::ItemKind;

use super::Context;
use super::infer::{BodyTables, LazyTables, Local, Res, Shape, SiteKey, Typed};
use super::reach::Instance;
use super::substs::{ProjId, Substs};
use super::ty::{Kind, Ty};
use eval::{
    SubState, check_const_holes, check_obligations, eval_offset_holes, eval_projections,
    resolve_instance_calls, sub_ty,
};
use inst::{InstOutcome, instantiate_pure};
use prepare::{is_aggregate, lay_out_fixed, prepare_summary};

/// How generic sema summarizes one function item: a reusable polymorphic summary, monomorphic inference with bounded-trait method hints, or plain monomorphic fallback.
#[derive(Clone, Debug)]
pub enum SummaryKind {
    Poly(Arc<PolySummary>),
    MonomorphicHints(Arc<HashMap<ExprId, ItemRef>>),
    MonomorphicFallback,
    Error(String),
}

/// Pre-extracted operand types for deciding whether an overloaded operator simplifies to a builtin.
#[derive(Clone, Copy, Debug)]
pub enum OpCheck {
    Unary(Ty),
    Binary(BinOp, Ty, Ty),
    AssignOp(BinOp, Ty, Ty),
    Index(Option<Ty>, Ty),
    Never,
}

/// Per-call-site resolution override produced by a concrete callable obligation.
#[derive(Clone, Debug)]
pub struct CallOverride {
    pub expr: ExprId,
    pub res: Option<Res>,
    pub autoderefs: Option<u32>,
    pub overloaded: bool,
    pub box_free: Option<Res>,
}

/// The check-once polymorphic summary `PolySummary(O, P, L, C)` of a generic function body.
#[derive(Clone, Debug)]
pub struct PolySummary {
    pub param_count: u16,
    pub template: Arc<BodyTables>,
    pub locals: Box<[Local]>,
    pub params: Box<[u32]>,
    pub ret: Option<Ty>,
    pub obligations: Box<[PolyObligation]>,
    pub projections: Box<[ProjId]>,
    pub layout_tys: Box<[Ty]>,
    pub const_holes: Box<[ConstHole]>,
    pub fixed_calls: Box<[(ItemRef, Substs)]>,
    pub direct_calls: Box<[(SiteKey, ItemRef, Substs)]>,
    pub call_holes: Box<[(SiteKey, ItemRef, Substs)]>,
    pub op_holes: Box<[(ExprId, OpCheck, Res)]>,
    pub fixed_drops: Box<[Res]>,
    pub drop_tys: Box<[Ty]>,
    pub parametric_tys: Box<[Ty]>,
}

/// A deferred trait or callable obligation recorded in `O` of a `PolySummary`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PolyObligation {
    Prove {
        of: ItemRef,
        self_ty: Ty,
        args: Substs,
    },
    Call {
        expr: ExprId,
        found: Ty,
        tuple: Ty,
    },
}

/// A parameter-dependent constant, `offset_of!`, or type-metadata intrinsic hole in `C` of a `PolySummary`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConstHole {
    Value(Ty),
    OffsetOf {
        expr: ExprId,
        steps: Box<[(Ty, u32)]>,
    },
    Intrinsic {
        item: ItemRef,
        substs: Substs,
    },
}

pub(super) fn typed_once_poly(
    cx: &mut Context<'_>,
    instance: &Instance,
    unlaid: &mut Vec<String>,
) -> Option<Result<Arc<Typed>, String>> {
    let before = cx.quiet_mark();
    let (item, substs) = *instance;
    if !eligible(cx, item, substs) {
        return None;
    }
    let expected = expected_param_count(cx, item)?;
    if cx.types.substs(substs).len() != expected || expected > u16::MAX as usize {
        return None;
    }
    let summaries = cx.store.lookups.summaries.clone();
    let built = summaries.run(
        cx,
        item,
        |cx| Ok(load_summary(cx, item, expected as u16)),
        |_| Ok(SummaryKind::MonomorphicFallback),
    );
    let summary = match built {
        Ok(SummaryKind::Poly(summary)) => summary,
        Ok(SummaryKind::Error(err)) => return Some(Err(err)),
        Ok(SummaryKind::MonomorphicHints(hints)) => {
            let m_before = cx.quiet_mark();
            let typed =
                match super::infer::infer_with_hints(cx, instance.0, instance.1, Some(hints)) {
                    Ok(typed) => Arc::new(typed),
                    Err(error) => {
                        return Some(Err(crate::diag::format_error(cx.name(instance.0), &error)));
                    }
                };
            thaw::keep_laid(cx, *instance, &typed, unlaid, m_before);
            return Some(Ok(typed));
        }
        _ => return None,
    };
    if let Err(err) = thaw::warm_body(cx, item, before) {
        return Some(Err(err));
    }
    instantiate_summary(cx, *instance, &summary, unlaid, before)
}

fn cached_poly_summary(cx: &mut Context<'_>, item: ItemRef) -> Option<Arc<PolySummary>> {
    if let Some(kind) = cx.store.lookups.summaries.get_ready(&item) {
        return match kind {
            SummaryKind::Poly(summary) => Some(summary),
            SummaryKind::MonomorphicHints(_)
            | SummaryKind::MonomorphicFallback
            | SummaryKind::Error(_) => None,
        };
    }
    frozen::frozen_summary(cx, item)
}

fn has_summary_body(entry: &crate::outline::item::Item) -> bool {
    (entry.kind == ItemKind::Fn && entry.body.is_some())
        || (entry.kind == ItemKind::Const && entry.name.is_some())
}

pub(super) fn poly_summary(cx: &mut Context<'_>, item: ItemRef) -> Option<Arc<PolySummary>> {
    if let Some(summary) = cached_poly_summary(cx, item) {
        return Some(summary);
    }
    let (_, entry) = cx.item(item);
    if !has_summary_body(entry) || cx.foreign(item) {
        return None;
    }
    let expected = expected_param_count(cx, item)?;
    if (expected == 0 && !cx.outline.is_fixed(item.krate)) || expected > u16::MAX as usize {
        return None;
    }
    let summaries = cx.store.lookups.summaries.clone();
    let built = summaries.run(
        cx,
        item,
        |cx| Ok(load_summary(cx, item, expected as u16)),
        |_| Ok(SummaryKind::MonomorphicFallback),
    );
    match built {
        Ok(SummaryKind::Poly(summary)) => Some(summary),
        _ => None,
    }
}

fn eligible(cx: &mut Context<'_>, item: ItemRef, substs: Substs) -> bool {
    let slice = cx.types.substs(substs);
    let is_fixed = cx.outline.is_fixed(item.krate);
    if (slice.is_empty() && !is_fixed)
        || cx.types.substs_have_params(substs)
        || slice
            .iter()
            .any(|&ty| matches!(cx.types.kind(ty), Kind::Never))
    {
        return false;
    }
    let (_, entry) = cx.item(item);
    if !has_summary_body(entry) || cx.foreign(item) {
        return false;
    }
    if !matches!(super::infer::virtual_slot(cx, item, substs), Ok(None)) {
        return false;
    }
    true
}

/// Checks a generic local function body polymorphically when possible, returning any borrow or unsafe error.
pub fn check_poly_item(cx: &mut Context<'_>, item: ItemRef) -> Result<(), String> {
    let (_, entry) = cx.item(item);
    if !has_summary_body(entry) || cx.foreign(item) || cx.outline.is_fixed(item.krate) {
        return Ok(());
    }
    let Some(expected) = expected_param_count(cx, item) else {
        return Ok(());
    };
    if expected == 0 || expected > u16::MAX as usize {
        return Ok(());
    }
    let summaries = cx.store.lookups.summaries.clone();
    let built = summaries.run(
        cx,
        item,
        |cx| Ok(load_summary(cx, item, expected as u16)),
        |_| Ok(SummaryKind::MonomorphicFallback),
    );
    match built {
        Ok(SummaryKind::Error(err)) => Err(err),
        _ => Ok(()),
    }
}

pub(super) fn expected_param_count(cx: &mut Context<'_>, item: ItemRef) -> Option<usize> {
    let sig = cx.signature(item).ok()?;
    let own = super::infer::type_params(&sig.generics).count();
    if let Some(block) = cx.owner_impl(item) {
        let outer = super::infer::type_params(&cx.impl_header_ref(block).ok()?.generics).count();
        return Some(outer + own);
    }
    if let Some(of) = cx.owner_trait(item) {
        let outer = 1 + super::infer::type_params(&cx.trait_generics(of).ok()?.1.params).count();
        return Some(outer + own);
    }
    Some(own)
}

fn load_summary(cx: &mut Context<'_>, item: ItemRef, param_count: u16) -> SummaryKind {
    let Some(summary) = frozen::frozen_summary(cx, item) else {
        return thaw::frozen_verdict(cx, item)
            .unwrap_or_else(|| build_summary(cx, item, param_count));
    };
    lay_out_fixed(cx, &summary);
    SummaryKind::Poly(summary)
}

fn build_summary(cx: &mut Context<'_>, item: ItemRef, param_count: u16) -> SummaryKind {
    let before = cx.quiet_mark();
    let built = super::infer::poly::build_poly_summary(cx, item, param_count);
    let SummaryKind::Poly(mut arc) = built else {
        thaw::settle_verdict(cx, item, before, &built);
        return built;
    };
    let Some(summary) = Arc::get_mut(&mut arc) else {
        return SummaryKind::MonomorphicFallback;
    };
    if prepare_summary(cx, item, summary).is_err() {
        let fallback = SummaryKind::MonomorphicFallback;
        thaw::settle_verdict(cx, item, before, &fallback);
        return fallback;
    }
    let kind = SummaryKind::Poly(arc);
    thaw::note_fresh_summary(cx, item, &kind);
    kind
}

fn record_instantiated(
    cx: &mut Context<'_>,
    instance: Instance,
    layouts: &[Ty],
    typed: &Arc<Typed>,
    unlaid: &mut Vec<String>,
    before: (usize, usize),
) -> bool {
    let laid = lay_out_instantiated(cx, layouts, typed)
        .map_err(|err| unlaid.push(format!("{}: {err:#}", cx.name(instance.0))));
    let clean = laid.is_ok() && cx.quiet_mark() == before;
    if clean {
        cx.store.finals.insert(instance, Arc::clone(typed));
    }
    clean
}

fn is_pure_summary(summary: &PolySummary) -> bool {
    summary.projections.is_empty()
        && summary.obligations.is_empty()
        && summary.call_holes.is_empty()
        && summary.op_holes.is_empty()
        && summary.const_holes.is_empty()
        && frozen::closure_free(summary)
}

fn is_zero_hole_mono(summary: &PolySummary) -> bool {
    summary.param_count == 0
        && summary.direct_calls.is_empty()
        && summary.drop_tys.is_empty()
        && summary.parametric_tys.is_empty()
        && is_pure_summary(summary)
}

fn detached_outcome(
    cx: &mut Context<'_>,
    instance: Instance,
    summary: &Arc<PolySummary>,
    concrete_args: &[Ty],
    check_only: bool,
) -> Option<InstOutcome> {
    super::infer::detached_summary(cx, instance.0, |infer| {
        let mut sub = SubState::new(concrete_args);
        eval_projections(infer, &mut sub, &summary.projections)?;
        let call_overrides = check_obligations(infer, &mut sub, &summary.obligations)?;
        let offset_overrides = if check_only {
            Vec::new()
        } else {
            eval_offset_holes(infer, &mut sub, &summary.const_holes)?
        };
        check_const_holes(infer, &mut sub, &summary.const_holes)?;
        let (resolved_calls, builtin_ops) =
            resolve_instance_calls(infer, &mut sub, summary, &call_overrides, check_only)?;
        let ret = summary
            .ret
            .map(|ty| sub_ty(infer, &mut sub, ty))
            .transpose()?;
        let layouts = summary
            .layout_tys
            .iter()
            .map(|&ty| sub_ty(infer, &mut sub, ty))
            .collect::<Result<Vec<_>>>()?;
        let drop_tys = summary
            .drop_tys
            .iter()
            .map(|&ty| sub_ty(infer, &mut sub, ty))
            .collect::<Result<Vec<_>>>()?;
        if check_only {
            return Ok(InstOutcome::CheckOnly(
                PolyCheck {
                    summary: Arc::clone(summary),
                    resolved_calls,
                },
                ret,
                layouts,
                drop_tys,
            ));
        }
        let locals = summary
            .locals
            .iter()
            .map(|local| {
                Ok(Local {
                    ty: sub_ty(infer, &mut sub, local.ty)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut projs: Vec<_> = sub.projs.into_iter().collect();
        projs.sort_unstable_by_key(|&(id, _)| id.0);
        let inst = PolyInst {
            args: instance.1,
            summary: Arc::clone(summary),
            projs: projs.into_boxed_slice(),
            resolved_calls,
            builtin_ops,
            call_overrides: call_overrides.into_boxed_slice(),
            offset_overrides: offset_overrides.into_boxed_slice(),
            materialized: OnceLock::new(),
        };
        Ok(InstOutcome::Full(inst, locals, ret, layouts, drop_tys))
    })
    .ok()
}

fn settle_outcome(
    cx: &mut Context<'_>,
    instance: Instance,
    summary: &Arc<PolySummary>,
    outcome: InstOutcome,
) -> Option<(Arc<Typed>, Vec<Ty>)> {
    match outcome {
        InstOutcome::CheckOnly(check, ret, layouts, drop_tys) => {
            let drops = instantiate_drops(cx, &summary.fixed_drops, &drop_tys).ok()?;
            Some((
                Arc::new(Typed {
                    tables: LazyTables::CheckOnly(Box::new(check)),
                    locals: Vec::new(),
                    params: Vec::new(),
                    ret,
                    drops,
                    shape: Shape::Body,
                }),
                layouts,
            ))
        }
        InstOutcome::Full(poly_inst, locals, ret, layouts, _) if !frozen::closure_free(summary) => {
            let raw = Typed {
                tables: LazyTables::Eager(Arc::new(poly_inst.materialize(&cx.types))),
                locals,
                params: summary.params.to_vec(),
                ret,
                drops: Vec::new(),
                shape: Shape::Body,
            };
            Some((
                Arc::new(super::infer::settle(cx, instance, raw).ok()?),
                layouts,
            ))
        }
        InstOutcome::Full(poly_inst, locals, ret, layouts, drop_tys) => {
            let drops = instantiate_drops(cx, &summary.fixed_drops, &drop_tys).ok()?;
            Some((
                Arc::new(Typed {
                    tables: LazyTables::Poly(Box::new(poly_inst)),
                    locals,
                    params: summary.params.to_vec(),
                    ret,
                    drops,
                    shape: Shape::Body,
                }),
                layouts,
            ))
        }
    }
}

fn instantiate_summary(
    cx: &mut Context<'_>,
    instance: Instance,
    summary: &Arc<PolySummary>,
    unlaid: &mut Vec<String>,
    before: (usize, usize),
) -> Option<Result<Arc<Typed>, String>> {
    if is_zero_hole_mono(summary) {
        let typed = Arc::new(Typed {
            tables: LazyTables::Eager(Arc::clone(&summary.template)),
            locals: summary.locals.to_vec(),
            params: summary.params.to_vec(),
            ret: summary.ret,
            drops: summary.fixed_drops.to_vec(),
            shape: Shape::Body,
        });
        record_instantiated(cx, instance, &summary.layout_tys, &typed, unlaid, before);
        return Some(Ok(typed));
    }
    let concrete_args = cx.types.substs(instance.1).to_vec();
    if concrete_args.len() != usize::from(summary.param_count) {
        return None;
    }
    if let Some((poly_inst, locals, ret, concrete_layouts, drops)) =
        frozen::frozen_instance(cx, instance, summary)
    {
        let typed = Arc::new(Typed {
            tables: LazyTables::Poly(Box::new(poly_inst)),
            locals,
            params: summary.params.to_vec(),
            ret,
            drops,
            shape: Shape::Body,
        });
        record_instantiated(cx, instance, &concrete_layouts, &typed, unlaid, before);
        return Some(Ok(typed));
    }
    let check_only = cx.store.lookups.skip_warm_sysroot_bodies
        && cx.outline.is_fixed(instance.0.krate)
        && frozen::closure_free(summary);
    let outcome = if is_pure_summary(summary) {
        instantiate_pure(&cx.types, instance.1, summary, check_only)
    } else {
        frozen::try_cached_instantiate(cx, instance, summary, check_only)
            .or_else(|| detached_outcome(cx, instance, summary, &concrete_args, check_only))?
    };
    let (typed, concrete_layouts) = settle_outcome(cx, instance, summary, outcome)?;
    if !cx.outline.is_fixed(instance.0.krate) && matches!(typed.shape, Shape::Body) {
        typed.ensure_tables(&cx.types);
        let body = match cx.body(instance.0) {
            Ok(b) => b,
            Err(e) => return Some(Err(crate::diag::format_error(cx.name(instance.0), &e))),
        };
        let sig = match cx.signature(instance.0) {
            Ok(s) => s,
            Err(e) => return Some(Err(crate::diag::format_error(cx.name(instance.0), &e))),
        };
        let src = cx.body_src(instance.0);
        if let Err(err) = super::borrow::check(cx, instance.0, (&body, &sig), src, &typed) {
            return Some(Err(crate::diag::format_error(cx.name(instance.0), &err)));
        }
    }
    record_instantiated(cx, instance, &concrete_layouts, &typed, unlaid, before);
    Some(Ok(typed))
}

fn instantiate_drops(cx: &mut Context<'_>, fixed: &[Res], concrete_tys: &[Ty]) -> Result<Vec<Res>> {
    let mut drops = fixed.to_vec();
    let mut seen = HashSet::default();
    for &ty in concrete_tys {
        if ty != Ty::NONE
            && seen.insert(ty)
            && let Some(call) = super::infer::glue_call(cx, ty)?
            && !drops.contains(&call)
        {
            drops.push(call);
        }
    }
    Ok(drops)
}

fn lay_out_instantiated(cx: &mut Context<'_>, layouts: &[Ty], typed: &Typed) -> Result<()> {
    for &ty in layouts {
        if is_aggregate(cx, ty) {
            cx.layout(ty)?;
        }
    }
    if typed.has_closures() {
        super::reach::lay_out(cx, typed)?;
    }
    Ok(())
}

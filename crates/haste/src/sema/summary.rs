pub mod codec;
pub(super) mod frozen;
mod inst;

pub use inst::PolyInst;

use std::sync::{Arc, OnceLock};

use anyhow::{Result, bail};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::body::{BinOp, Body, Expr, ExprId, UnOp};
use crate::names::ItemRef;
use crate::outline::item::ItemKind;

use super::Context;
use super::infer::{BodyTables, Coercion, Infer, LazyTables, Local, Res, Shape, SiteKey, Typed};
use super::reach::Instance;
use super::substs::{ProjId, Substs};
use super::ty::{Kind, Ty, Types, object_parts};

/// How generic sema summarizes one function item: a reusable polymorphic summary or monomorphic fallback.
#[derive(Clone, Debug)]
pub enum SummaryKind {
    Poly(Arc<PolySummary>),
    MonomorphicFallback,
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
    Prove { of: ItemRef, self_ty: Ty, args: Substs },
    Call { expr: ExprId, found: Ty, tuple: Ty },
}

/// A parameter-dependent constant, `offset_of!`, or type-metadata intrinsic hole in `C` of a `PolySummary`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConstHole {
    Value(Ty),
    OffsetOf { expr: ExprId, steps: Box<[(Ty, u32)]> },
    Intrinsic { item: ItemRef, substs: Substs },
}

struct SubState<'a> {
    args: &'a [Ty],
    projs: HashMap<ProjId, Ty>,
    tys: HashMap<Ty, Ty>,
}

impl<'a> SubState<'a> {
    fn new(args: &'a [Ty]) -> Self {
        Self { args, projs: HashMap::default(), tys: HashMap::default() }
    }
}

pub(super) fn typed_once_poly(cx: &mut Context<'_>, instance: &Instance, unlaid: &mut Vec<String>) -> Option<Result<Arc<Typed>, String>> {
    let before = cx.quiet_mark();
    let (item, substs) = *instance;
    let is_sysroot = cx.outline.is_sysroot(item.krate);
    let summaries = cx.store.lookups.summaries.clone();
    let built = if substs.is_empty() && is_sysroot {
        let (_, entry) = cx.item(item);
        if entry.kind != ItemKind::Fn || entry.body.is_none() || cx.foreign(item) {
            return None;
        }
        summaries.run(
            cx,
            item,
            |cx| {
                if let Some(poly) = frozen::frozen_summary(cx, item) {
                    return Ok(SummaryKind::Poly(poly));
                }
                if !eligible(cx, item, substs) || expected_param_count(cx, item) != Some(0) {
                    return Ok(SummaryKind::MonomorphicFallback);
                }
                Ok(build_summary(cx, item, 0))
            },
            |_| Ok(SummaryKind::MonomorphicFallback),
        )
    } else {
        if !eligible(cx, item, substs) {
            return None;
        }
        let expected = expected_param_count(cx, item)?;
        if cx.types.substs(substs).len() != expected || expected > u16::MAX as usize {
            return None;
        }
        summaries.run(
            cx,
            item,
            |cx| {
                if let Some(poly) = frozen::frozen_summary(cx, item) {
                    return Ok(SummaryKind::Poly(poly));
                }
                Ok(build_summary(cx, item, expected as u16))
            },
            |_| Ok(SummaryKind::MonomorphicFallback),
        )
    };
    let Ok(SummaryKind::Poly(summary)) = built else { return None };
    if is_sysroot
        && !cx.store.lookups.skip_warm_sysroot_bodies
        && cx.store.bodies.get(&item).is_none()
        && crate::profile() == crate::outline::tree::Profile::Release
        && cx.store.lookups.substrate.is_some()
        && cx.attr_value(item, "cold").is_none()
        && cx.attr_value(item, "inline") != Some("never")
        && let Err(err) = cx.body(item)
        && cx.quiet_mark() != before
    {
        return Some(Err(format!("{}: {err:#}", cx.name(item))));
    }
    instantiate_summary(cx, *instance, &summary, unlaid, before)
}

fn eligible(cx: &mut Context<'_>, item: ItemRef, substs: Substs) -> bool {
    let slice = cx.types.substs(substs);
    let is_sysroot = cx.outline.is_sysroot(item.krate);
    if (slice.is_empty() && !is_sysroot) || cx.types.substs_have_params(substs) || slice.iter().any(|&ty| matches!(cx.types.kind(ty), Kind::Never)) {
        return false;
    }
    let (_, entry) = cx.item(item);
    if entry.kind != ItemKind::Fn || entry.body.is_none() || cx.foreign(item) {
        return false;
    }
    if !matches!(super::infer::compiler_made(cx, item, substs), Ok(None)) {
        return false;
    }
    matches!(super::infer::virtual_slot(cx, item, substs), Ok(None))
}

fn expected_param_count(cx: &mut Context<'_>, item: ItemRef) -> Option<usize> {
    let sig = cx.signature(item).ok()?;
    let own = super::infer::type_params(&sig.generics).count();
    if let Some(block) = cx.owner_impl(item) {
        let outer = super::infer::type_params(&cx.impl_header(block).ok()?.generics).count();
        return Some(outer + own);
    }
    if let Some(of) = cx.owner_trait(item) {
        let outer = 1 + super::infer::type_params(&cx.trait_generics(of).ok()?.1.params).count();
        return Some(outer + own);
    }
    Some(own)
}

fn build_summary(cx: &mut Context<'_>, item: ItemRef, param_count: u16) -> SummaryKind {
    let SummaryKind::Poly(mut arc) = super::infer::poly::build_poly_summary(cx, item, param_count) else {
        return SummaryKind::MonomorphicFallback;
    };
    if cx.outline.is_sysroot(item.krate) {
        cx.store.lookups.has_new_sysroot_summaries.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    let Some(summary) = Arc::get_mut(&mut arc) else { return SummaryKind::MonomorphicFallback };
    if prepare_summary(cx, item, summary).is_err() {
        return SummaryKind::MonomorphicFallback;
    }
    SummaryKind::Poly(arc)
}

fn prepare_summary(cx: &mut Context<'_>, item: ItemRef, summary: &mut PolySummary) -> Result<()> {
    for c in summary.template.coercions.values() {
        if let Coercion::Vtable(vt) = c
            && (cx.types.has_params(vt.ty) || cx.types.has_params(vt.object))
        {
            bail!("parametric vtable coercion");
        }
    }
    extract_call_holes(cx, summary);
    if !summary.template.operators.is_empty() {
        let body = cx.body(item)?;
        summary.op_holes = extract_op_holes(&cx.types, &body, &summary.template);
    }
    extract_drops_and_tys(cx, summary)
}

fn extract_call_holes(cx: &Context<'_>, summary: &mut PolySummary) {
    let types = &cx.types;
    let tpl = &summary.template;
    let call_obs: HashSet<ExprId> = summary.obligations.iter().filter_map(|o| match *o {
        PolyObligation::Call { expr, .. } => Some(expr),
        _ => None,
    }).collect();
    let mut fixed = Vec::new();
    let mut direct = Vec::new();
    let mut holes = Vec::new();
    let mut push = |key: SiteKey, r: Res| {
        if let Res::Fn(i, s) = r {
            if !types.substs_have_params(s) {
                fixed.push((i, s));
            } else if cx.owner_trait(i).is_none() {
                direct.push((key, i, s));
            } else {
                holes.push((key, i, s));
            }
        }
    };
    for &(id, r) in tpl.res.as_slice() {
        if !call_obs.contains(&id) {
            push(SiteKey::ExprRes(id), r);
        }
    }
    tpl.pat_res.as_slice().iter().for_each(|&(id, r)| push(SiteKey::PatRes(id), r));
    tpl.derefs.as_slice().iter().for_each(|&((id, step), r)| push(SiteKey::Deref(id, step), r));
    tpl.try_blocks.as_slice().iter().for_each(|&(id, r)| push(SiteKey::TryBlock(id), r));
    for &(id, r) in tpl.box_frees.as_slice() {
        if !call_obs.contains(&id) {
            push(SiteKey::BoxFree(id), r);
        }
    }
    for &(id, ref f) in tpl.fors.as_slice() {
        push(SiteKey::ForInto(id), f.into_iter);
        push(SiteKey::ForNext(id), f.next);
    }
    for &(id, ref t) in tpl.tries.as_slice() {
        push(SiteKey::TryBranch(id), t.branch);
        push(SiteKey::TryFrom(id), t.from_residual);
    }
    for &(id, ref a) in tpl.awaits.as_slice() {
        push(SiteKey::AwaitInto(id), a.into_future);
        push(SiteKey::AwaitPoll(id), a.poll);
    }
    for &(id, ref c) in tpl.coercions.as_slice() {
        match c {
            Coercion::Reify(r) => push(SiteKey::CoercionReify(id), *r),
            Coercion::Deref(steps, _) => steps.iter().enumerate().for_each(|(idx, &(_, r))| {
                if let Some(r) = r {
                    push(SiteKey::CoercionDeref(id, idx as u16), r);
                }
            }),
            _ => {}
        }
    }
    for &(_, ref c) in tpl.coercions.as_slice() {
        if let Coercion::Vtable(vt) = c {
            for r in vt.calls() {
                if let Res::Fn(i, s) = r {
                    fixed.push((*i, *s));
                }
            }
        }
    }
    summary.fixed_calls = fixed.into_boxed_slice();
    summary.direct_calls = direct.into_boxed_slice();
    summary.call_holes = holes.into_boxed_slice();
}

fn extract_op_holes(types: &Types, body: &Body, tpl: &BodyTables) -> Box<[(ExprId, OpCheck, Res)]> {
    tpl.operators
        .as_slice()
        .iter()
        .map(|&(expr, r)| {
            let check = match *body.expr(expr) {
                Expr::Unary(op, operand) if op != UnOp::Deref => OpCheck::Unary(tpl.exprs[operand.0 as usize]),
                Expr::Binary(op, lhs, rhs) => OpCheck::Binary(op, tpl.exprs[lhs.0 as usize], tpl.exprs[rhs.0 as usize]),
                Expr::AssignOp(op, lhs, rhs) => OpCheck::AssignOp(op, tpl.exprs[lhs.0 as usize], tpl.exprs[rhs.0 as usize]),
                Expr::Index(_, idx) => {
                    let step = match r {
                        Res::Fn(_, s) => types.substs(s).first().copied(),
                        _ => None,
                    };
                    OpCheck::Index(step, tpl.exprs[idx.0 as usize])
                }
                _ => OpCheck::Never,
            };
            (expr, check, r)
        })
        .collect()
}

fn extract_drops_and_tys(cx: &mut Context<'_>, summary: &mut PolySummary) -> Result<()> {
    let mut seen_drop = HashSet::default();
    let mut fixed_drops = Vec::new();
    let mut drop_tys = Vec::new();
    for ty in summary.locals.iter().map(|l| l.ty).chain(summary.template.exprs.iter().copied()) {
        if ty == Ty::NONE || !seen_drop.insert(ty) {
            continue;
        }
        if !cx.types.has_params(ty) {
            if matches!(cx.types.kind(ty), Kind::Adt(..) | Kind::Tuple(_) | Kind::Array(..)) {
                drop(cx.layout(ty));
            }
            if let Some(call) = super::infer::glue_call(cx, ty)? {
                fixed_drops.push(call);
            }
        } else if !matches!(cx.types.kind(ty), Kind::Prim(_) | Kind::Ref(..) | Kind::Ptr(..) | Kind::FnDef(..) | Kind::FnPtr(..) | Kind::Never | Kind::Value(_)) {
            drop_tys.push(ty);
        }
    }
    let mut seen_param = HashSet::default();
    let mut parametric_tys = Vec::new();
    let all_tys = summary.locals.iter().map(|l| l.ty).chain(summary.ret).chain(summary.template.pats.values().copied()).chain(summary.template.exprs.iter().copied());
    for ty in all_tys {
        if ty != Ty::NONE && cx.types.has_params(ty) && seen_param.insert(ty) {
            parametric_tys.push(ty);
        }
    }
    summary.fixed_drops = fixed_drops.into_boxed_slice();
    summary.drop_tys = drop_tys.into_boxed_slice();
    summary.parametric_tys = parametric_tys.into_boxed_slice();
    Ok(())
}

fn instantiate_summary(cx: &mut Context<'_>, instance: Instance, summary: &Arc<PolySummary>, unlaid: &mut Vec<String>, before: (usize, usize)) -> Option<Result<Arc<Typed>, String>> {
    if summary.param_count == 0
        && summary.projections.is_empty()
        && summary.obligations.is_empty()
        && summary.direct_calls.is_empty()
        && summary.call_holes.is_empty()
        && summary.op_holes.is_empty()
        && summary.const_holes.is_empty()
        && summary.drop_tys.is_empty()
        && summary.parametric_tys.is_empty()
        && summary.template.closures.is_empty()
    {
        let typed = Arc::new(Typed {
            tables: LazyTables::Eager(Arc::clone(&summary.template)),
            locals: summary.locals.to_vec(),
            params: summary.params.to_vec(),
            ret: summary.ret,
            drops: summary.fixed_drops.to_vec(),
            shape: Shape::Body,
        });
        let laid = lay_out_instantiated(cx, &summary.layout_tys, &typed).map_err(|err| unlaid.push(format!("{}: {err:#}", cx.name(instance.0))));
        if laid.is_ok() && cx.quiet_mark() == before {
            cx.store.finals.insert(instance, Arc::clone(&typed));
        }
        return Some(Ok(typed));
    }
    let concrete_args = cx.types.substs(instance.1).to_vec();
    if concrete_args.len() != usize::from(summary.param_count) {
        return None;
    }
    if let Some((poly_inst, locals, ret, concrete_layouts, drops)) = frozen::frozen_instance(cx, instance, summary) {
        let typed = Arc::new(Typed { tables: LazyTables::Poly(Box::new(poly_inst)), locals, params: summary.params.to_vec(), ret, drops, shape: Shape::Body });
        let laid = lay_out_instantiated(cx, &concrete_layouts, &typed).map_err(|err| unlaid.push(format!("{}: {err:#}", cx.name(instance.0))));
        if laid.is_ok() && cx.quiet_mark() == before {
            cx.store.finals.insert(instance, Arc::clone(&typed));
        }
        return Some(Ok(typed));
    }
    let pure = summary.projections.is_empty()
        && summary.obligations.is_empty()
        && summary.call_holes.is_empty()
        && summary.op_holes.is_empty()
        && summary.const_holes.is_empty()
        && summary.template.closures.is_empty();
    let (poly_inst, locals, ret, concrete_layouts, concrete_drop_tys) = if pure {
        let mut inst = PolyInst {
            args: instance.1,
            summary: Arc::clone(summary),
            projs: Box::new([]),
            resolved_calls: Box::new([]),
            builtin_ops: Box::new([]),
            call_overrides: Box::new([]),
            offset_overrides: Box::new([]),
            materialized: OnceLock::new(),
        };
        if !summary.direct_calls.is_empty() {
            let mut resolved: Vec<(SiteKey, ItemRef, Substs)> = summary
                .direct_calls
                .iter()
                .map(|&(key, item, substs)| (key, item, inst.sub_substs(&cx.types, substs)))
                .collect();
            resolved.sort_unstable_by_key(|&(key, ..)| key);
            inst.resolved_calls = resolved.into_boxed_slice();
        }
        let locals = summary.locals.iter().map(|local| Local { ty: inst.sub_ty(&cx.types, local.ty) }).collect();
        let ret = summary.ret.map(|ty| inst.sub_ty(&cx.types, ty));
        let layouts = summary.layout_tys.iter().map(|&ty| inst.sub_ty(&cx.types, ty)).collect();
        let drop_tys = summary.drop_tys.iter().map(|&ty| inst.sub_ty(&cx.types, ty)).collect();
        (inst, locals, ret, layouts, drop_tys)
    } else if let Some(cached) = frozen::try_cached_instantiate(cx, instance, summary) {
        cached
    } else {
        let Ok(built) = super::infer::detached_summary(cx, instance.0, |infer| {
            let mut sub = SubState::new(&concrete_args);
            eval_projections(infer, &mut sub, &summary.projections)?;
            let call_overrides = check_obligations(infer, &mut sub, &summary.obligations)?;
            let offset_overrides = eval_offset_holes(infer, &mut sub, &summary.const_holes)?;
            check_const_holes(infer, &mut sub, &summary.const_holes)?;
            let (resolved_calls, builtin_ops) = resolve_instance_calls(infer, &mut sub, summary, &call_overrides)?;
            let locals = summary.locals.iter().map(|local| Ok(Local { ty: sub_ty(infer, &mut sub, local.ty)? })).collect::<Result<Vec<_>>>()?;
            let ret = summary.ret.map(|ty| sub_ty(infer, &mut sub, ty)).transpose()?;
            let layouts = summary.layout_tys.iter().map(|&ty| sub_ty(infer, &mut sub, ty)).collect::<Result<Vec<_>>>()?;
            let drop_tys = summary.drop_tys.iter().map(|&ty| sub_ty(infer, &mut sub, ty)).collect::<Result<Vec<_>>>()?;
            let mut projs: Vec<(ProjId, Ty)> = sub.projs.into_iter().collect();
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
            Ok((inst, locals, ret, layouts, drop_tys))
        }) else {
            return None;
        };
        built
    };
    let typed = if !summary.template.closures.is_empty() {
        let tables = Arc::new(poly_inst.materialize(&cx.types));
        let raw = Typed { tables: LazyTables::Eager(tables), locals, params: summary.params.to_vec(), ret, drops: Vec::new(), shape: Shape::Body };
        let Ok(settled) = super::infer::settle(cx, instance, raw) else { return None };
        Arc::new(settled)
    } else {
        let Ok(drops) = instantiate_drops(cx, &summary.fixed_drops, &concrete_drop_tys) else { return None };
        Arc::new(Typed { tables: LazyTables::Poly(Box::new(poly_inst)), locals, params: summary.params.to_vec(), ret, drops, shape: Shape::Body })
    };
    let laid = lay_out_instantiated(cx, &concrete_layouts, &typed).map_err(|err| unlaid.push(format!("{}: {err:#}", cx.name(instance.0))));
    if laid.is_ok() && cx.quiet_mark() == before {
        if summary.template.closures.is_empty() && cx.outline.is_sysroot(instance.0.krate) {
            cx.store.lookups.has_new_sysroot_instances.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        cx.store.finals.insert(instance, Arc::clone(&typed));
    }
    Some(Ok(typed))
}

fn instantiate_drops(cx: &mut Context<'_>, fixed: &[Res], concrete_tys: &[Ty]) -> Result<Vec<Res>> {
    let mut drops = fixed.to_vec();
    let mut seen = HashSet::default();
    for &ty in concrete_tys {
        if ty != Ty::NONE && seen.insert(ty) && let Some(call) = super::infer::glue_call(cx, ty)? && !drops.contains(&call) {
            drops.push(call);
        }
    }
    Ok(drops)
}

type ResolvedSites = (Box<[(SiteKey, ItemRef, Substs)]>, Box<[ExprId]>);

fn resolve_instance_calls(infer: &mut Infer<'_, '_>, sub: &mut SubState<'_>, summary: &PolySummary, calls: &[CallOverride]) -> Result<ResolvedSites> {
    let mut resolved = Vec::with_capacity(summary.direct_calls.len() + summary.call_holes.len() + summary.op_holes.len() + calls.len());
    for &(key, item, substs) in &summary.direct_calls {
        let mapped = sub_substs(infer, sub, substs)?;
        resolved.push((key, item, mapped));
    }
    for &(key, item, substs) in &summary.call_holes {
        let mapped = sub_substs(infer, sub, substs)?;
        if let Res::Fn(ri, rs) = infer.trait_call(item, mapped)? {
            resolved.push((key, ri, rs));
        }
    }
    let mut builtin_ops = Vec::new();
    for &(expr, check, ref r) in &summary.op_holes {
        let concrete_check = sub_op_check(infer, sub, check)?;
        if infer.is_builtin_op(concrete_check) {
            builtin_ops.push(expr);
        } else if let Res::Fn(ri, rs) = sub_res(infer, sub, r)? {
            resolved.push((SiteKey::Operator(expr), ri, rs));
        }
    }
    for ov in calls {
        if let Some(Res::Fn(ri, rs)) = ov.res {
            resolved.push((SiteKey::ExprRes(ov.expr), ri, rs));
        }
        if let Some(Res::Fn(ri, rs)) = ov.box_free {
            resolved.push((SiteKey::BoxFree(ov.expr), ri, rs));
        }
    }
    resolved.sort_unstable_by_key(|&(key, ..)| key);
    builtin_ops.sort_unstable();
    Ok((resolved.into_boxed_slice(), builtin_ops.into_boxed_slice()))
}

fn sub_op_check(infer: &mut Infer<'_, '_>, sub: &mut SubState<'_>, check: OpCheck) -> Result<OpCheck> {
    Ok(match check {
        OpCheck::Unary(ty) => OpCheck::Unary(sub_ty(infer, sub, ty)?),
        OpCheck::Binary(op, l, r) => OpCheck::Binary(op, sub_ty(infer, sub, l)?, sub_ty(infer, sub, r)?),
        OpCheck::AssignOp(op, l, r) => OpCheck::AssignOp(op, sub_ty(infer, sub, l)?, sub_ty(infer, sub, r)?),
        OpCheck::Index(step, idx) => OpCheck::Index(step.map(|s| sub_ty(infer, sub, s)).transpose()?, sub_ty(infer, sub, idx)?),
        OpCheck::Never => OpCheck::Never,
    })
}

fn lay_out_instantiated(cx: &mut Context<'_>, layouts: &[Ty], typed: &Typed) -> Result<()> {
    for &ty in layouts {
        if ty != Ty::NONE && matches!(cx.types.kind(ty), Kind::Adt(..) | Kind::Tuple(_) | Kind::Array(..)) {
            cx.layout(ty)?;
        }
    }
    if typed.has_closures() {
        super::reach::lay_out(cx, typed)?;
    }
    Ok(())
}

fn eval_projections(infer: &mut Infer<'_, '_>, sub: &mut SubState<'_>, projections: &[ProjId]) -> Result<()> {
    for &proj_id in projections {
        eval_one_proj(infer, sub, proj_id)?;
    }
    Ok(())
}

fn eval_one_proj(infer: &mut Infer<'_, '_>, sub: &mut SubState<'_>, proj_id: ProjId) -> Result<Ty> {
    if let Some(&cached) = sub.projs.get(&proj_id) {
        return Ok(cached);
    }
    let def = infer.cx.types.proj(proj_id);
    let self_ty = sub_ty(infer, sub, def.self_ty)?;
    let trait_args = sub_substs_vec(infer, sub, def.args)?;
    let own_args = sub_substs_vec(infer, sub, def.own)?;
    let trait_substs = infer.cx.types.intern_substs(&trait_args);
    let own_substs = infer.cx.types.intern_substs(&own_args);
    let cacheable = !infer.has_vars(self_ty)
        && !infer.cx.types.has_params(self_ty)
        && !infer.cx.types.substs_have_params(trait_substs)
        && !infer.cx.types.substs(trait_substs).iter().any(|&ty| infer.has_vars(ty))
        && !infer.cx.types.substs_have_params(own_substs)
        && !infer.cx.types.substs(own_substs).iter().any(|&ty| infer.has_vars(ty));
    let key = (def.of, self_ty, trait_substs, def.assoc, own_substs);
    if cacheable && let Some(cached) = infer.cx.store.lookups.assoc_types.get(&key) {
        sub.projs.insert(proj_id, cached);
        return Ok(cached);
    }
    let before = infer.cx.quiet_mark();
    let name = infer.cx.name(def.assoc).to_string();
    let Some(value) = infer.picked_alias(def.of, self_ty, &trait_args, (&name, &own_args))? else { bail!("unnormalized projection `{name}` on {}", infer.show(self_ty)) };
    infer.settle_projections()?;
    let zonked = infer.zonk(value);
    if infer.cx.types.has_params(zonked) || infer.has_vars(zonked) || matches!(infer.cx.types.kind(zonked), Kind::Never) {
        bail!("projection `{name}` did not resolve to a concrete type");
    }
    if cacheable && infer.cx.quiet_mark() == before {
        infer.cx.store.lookups.assoc_types.insert(key, zonked);
    }
    sub.projs.insert(proj_id, zonked);
    Ok(zonked)
}

fn sub_ty(infer: &mut Infer<'_, '_>, sub: &mut SubState<'_>, ty: Ty) -> Result<Ty> {
    if ty == Ty::NONE || !infer.cx.types.has_params(ty) {
        return Ok(ty);
    }
    if let Some(&cached) = sub.tys.get(&ty) {
        return Ok(cached);
    }
    let kind = infer.cx.types.kind(ty).clone();
    let mapped = match kind {
        Kind::Param(idx) => *sub.args.get(idx as usize).ok_or_else(|| anyhow::anyhow!("param {idx} out of bounds"))?,
        Kind::Proj(proj_id) => eval_one_proj(infer, sub, proj_id)?,
        _ => sub_compound_kind(infer, sub, &kind)?,
    };
    sub.tys.insert(ty, mapped);
    Ok(mapped)
}

fn sub_compound_kind(infer: &mut Infer<'_, '_>, sub: &mut SubState<'_>, kind: &Kind) -> Result<Ty> {
    let pools = infer.cx.types.pools();
    let mut err: Option<anyhow::Error> = None;
    let mapped = kind.map_pool(&pools, &mut |child| match sub_ty(infer, sub, child) {
        Ok(found) => found,
        Err(e) => {
            if err.is_none() {
                err = Some(e);
            }
            child
        }
    });
    if let Some(e) = err {
        return Err(e);
    }
    Ok(infer.cx.types.intern(mapped))
}

fn sub_substs(infer: &mut Infer<'_, '_>, sub: &mut SubState<'_>, substs: Substs) -> Result<Substs> {
    if !infer.cx.types.substs_have_params(substs) {
        return Ok(substs);
    }
    let mapped = sub_substs_vec(infer, sub, substs)?;
    Ok(infer.cx.types.intern_substs(&mapped))
}

fn sub_substs_vec(infer: &mut Infer<'_, '_>, sub: &mut SubState<'_>, substs: Substs) -> Result<Vec<Ty>> {
    let raw = infer.cx.types.substs(substs).to_vec();
    raw.into_iter().map(|ty| sub_ty(infer, sub, ty)).collect()
}

fn sub_res(infer: &mut Infer<'_, '_>, sub: &mut SubState<'_>, res: &Res) -> Result<Res> {
    match *res {
        Res::Local(local) => Ok(Res::Local(local)),
        Res::Const(ty) => Ok(Res::Const(sub_ty(infer, sub, ty)?)),
        Res::Ctor(item, variant, args) => Ok(Res::Ctor(item, variant, sub_substs(infer, sub, args)?)),
        Res::Fn(item, args) => {
            let mapped = sub_substs(infer, sub, args)?;
            infer.trait_call(item, mapped)
        }
    }
}

fn check_obligations(infer: &mut Infer<'_, '_>, sub: &mut SubState<'_>, obligations: &[PolyObligation]) -> Result<Vec<CallOverride>> {
    let mut calls = Vec::new();
    for obligation in obligations {
        match *obligation {
            PolyObligation::Prove { of, self_ty, args } => {
                let concrete_self = sub_ty(infer, sub, self_ty)?;
                let concrete_args = sub_substs_vec(infer, sub, args)?;
                if !concrete_holds(infer, of, concrete_self, &concrete_args)? {
                    bail!("concrete obligation `{}` failed for {}", infer.cx.name(of), infer.show(concrete_self));
                }
            }
            PolyObligation::Call { expr, found, tuple } => {
                let concrete_found = sub_ty(infer, sub, found)?;
                let concrete_tuple = sub_ty(infer, sub, tuple)?;
                calls.push(resolve_call_override(infer, expr, concrete_found, concrete_tuple)?);
            }
        }
    }
    Ok(calls)
}

fn concrete_holds(infer: &mut Infer<'_, '_>, of: ItemRef, self_ty: Ty, args: &[Ty]) -> Result<bool> {
    if infer.compiler_decides(of, self_ty) {
        return Ok(infer.compiler_holds(of, self_ty));
    }
    if infer.compiler_impl(of, self_ty) || infer.object_holds(of, self_ty, args) == Some(true) {
        return Ok(true);
    }
    if infer.is_callable_trait(of) && matches!(infer.cx.types.kind(self_ty), Kind::Closure(..) | Kind::FnDef(..) | Kind::FnPtr(..)) {
        return Ok(true);
    }
    Ok(infer.select(of, self_ty, args)?.is_some())
}

fn resolve_call_override(infer: &mut Infer<'_, '_>, expr: ExprId, found: Ty, tuple: Ty) -> Result<CallOverride> {
    let (mut ty, mut steps) = (found, 0u32);
    while !matches!(infer.cx.types.kind(ty), Kind::Closure(..) | Kind::FnDef(..) | Kind::FnPtr(..) | Kind::Ptr(..))
        && let Some(inner) = infer.builtin_deref(ty)
    {
        (ty, steps) = (inner, steps + 1);
    }
    let autoderefs = (steps > 0).then_some(steps);
    match infer.cx.types.kind(ty).clone() {
        Kind::Closure(..) => {
            let Some(call_once) = infer.cx.call_once() else { bail!("core has no `FnOnce::call_once`") };
            let res = Some(Res::fn_of(&infer.cx.types, call_once, &[ty, tuple]));
            Ok(CallOverride { expr, res, autoderefs, overloaded: false, box_free: None })
        }
        Kind::FnDef(item, substs) => {
            let res = Some(infer.trait_call(item, substs)?);
            Ok(CallOverride { expr, res, autoderefs, overloaded: false, box_free: None })
        }
        Kind::FnPtr(..) => Ok(CallOverride { expr, res: None, autoderefs, overloaded: false, box_free: None }),
        Kind::Dyn(of, parts, members, _) if infer.is_callable_trait(of) => resolve_dyn_call(infer, (expr, found, ty), (of, parts, &members), autoderefs),
        Kind::Adt(..) => resolve_overloaded_call(infer, expr, found, tuple),
        _ => bail!("unsupported concrete callable {}", infer.show(ty)),
    }
}

fn resolve_dyn_call(infer: &mut Infer<'_, '_>, (expr, found, ty): (ExprId, Ty, Ty), (of, parts, members): (ItemRef, Substs, &[ItemRef]), autoderefs: Option<u32>) -> Result<CallOverride> {
    let dyn_tuple = *object_parts(infer.cx.types.substs(parts), members).0.first().ok_or_else(|| anyhow::anyhow!("callable trait object without tuple"))?;
    let (name, box_free) = match infer.cx.lang(of) {
        Some("fn") => ("call", None),
        Some("fn_mut") => ("call_mut", None),
        _ => ("call_once", Some(resolve_box_free(infer, found, autoderefs)?)),
    };
    let Some(method) = infer.cx.trait_member(of, name) else { bail!("missing `{name}`") };
    let res = Some(Res::fn_of(&infer.cx.types, method, &[ty, dyn_tuple]));
    Ok(CallOverride { expr, res, autoderefs, overloaded: false, box_free })
}

fn resolve_box_free(infer: &mut Infer<'_, '_>, found: Ty, autoderefs: Option<u32>) -> Result<Res> {
    if infer.cx.boxed(found).is_none() || autoderefs != Some(1) {
        bail!("`dyn FnOnce` called by value without `Box`");
    }
    infer.own_drop(found)?.ok_or_else(|| anyhow::anyhow!("`Box` has no `Drop` impl"))
}

fn resolve_overloaded_call(infer: &mut Infer<'_, '_>, expr: ExprId, found: Ty, tuple: Ty) -> Result<CallOverride> {
    let (mut ty, mut steps) = (found, 0u32);
    loop {
        if let Some(method) = infer.callable_impl(ty, tuple)? {
            let substs = infer.cx.types.intern_substs(&[ty, tuple]);
            let res = Some(infer.trait_call(method, substs)?);
            return Ok(CallOverride { expr, res, autoderefs: (steps > 0).then_some(steps), overloaded: true, box_free: None });
        }
        let Some(inner) = infer.builtin_deref(ty) else { bail!("{} is not callable", infer.show(found)) };
        (ty, steps) = (inner, steps + 1);
    }
}

fn eval_offset_holes(infer: &mut Infer<'_, '_>, sub: &mut SubState<'_>, holes: &[ConstHole]) -> Result<Vec<(ExprId, u64)>> {
    let mut out = Vec::new();
    for hole in holes {
        let ConstHole::OffsetOf { expr, steps } = hole else { continue };
        let mut offset = 0u64;
        for &(step_ty, index) in &**steps {
            let concrete = sub_ty(infer, sub, step_ty)?;
            let layout = infer.cx.layout(concrete)?;
            offset += super::layout::field_offset(&layout, 0, index) as u64;
        }
        out.push((*expr, offset));
    }
    Ok(out)
}

fn check_const_holes(infer: &mut Infer<'_, '_>, sub: &mut SubState<'_>, holes: &[ConstHole]) -> Result<()> {
    for hole in holes {
        match *hole {
            ConstHole::Value(ty) => {
                let concrete = sub_ty(infer, sub, ty)?;
                if !matches!(infer.cx.types.kind(concrete), Kind::Value(_)) {
                    bail!("const hole did not resolve to value");
                }
            }
            ConstHole::Intrinsic { substs, .. } => {
                let concrete = sub_substs(infer, sub, substs)?;
                if infer.cx.types.substs_have_params(concrete) {
                    bail!("intrinsic hole retained parameters");
                }
            }
            ConstHole::OffsetOf { .. } => {}
        }
    }
    Ok(())
}

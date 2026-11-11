use std::sync::Arc;

use anyhow::{Result, bail};
use rustc_hash::FxHashMap as HashMap;

use crate::body::{ExprId, PatId};
use crate::names::ItemRef;
use crate::outline::item::ItemKind;

use super::Context;
use super::infer::{Awaited, BodyTables, Closure, Coercion, For, Infer, Local, Res, Shape, Tried, Typed};
use super::reach::Instance;
use super::substs::{PackedMap, PackedSet, ProjId, Substs};
use super::ty::{Kind, Ty, object_parts};

/// How generic sema summarizes one function item: a reusable polymorphic summary or monomorphic fallback.
#[derive(Clone, Debug)]
pub enum SummaryKind {
    Poly(Arc<PolySummary>),
    MonomorphicFallback,
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

struct CallOverride {
    expr: ExprId,
    res: Option<Res>,
    autoderefs: Option<u32>,
    overloaded: bool,
    box_free: Option<Res>,
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
    if !eligible(cx, item, substs) {
        return None;
    }
    let expected = expected_param_count(cx, item)?;
    if cx.types.substs(substs).len() != expected || expected > u16::MAX as usize {
        return None;
    }
    let summaries = cx.store.lookups.summaries.clone();
    let built = summaries.run(cx, item, |cx| Ok(build_summary(cx, item, expected as u16)), |_| Ok(SummaryKind::MonomorphicFallback));
    let Ok(SummaryKind::Poly(summary)) = built else { return None };
    instantiate_summary(cx, *instance, &summary, unlaid, before)
}

fn eligible(cx: &mut Context<'_>, item: ItemRef, substs: Substs) -> bool {
    let slice = cx.types.substs(substs);
    if slice.is_empty() || cx.types.substs_have_params(substs) || slice.iter().any(|&ty| matches!(cx.types.kind(ty), Kind::Never)) {
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
    let built = super::infer::poly::build_poly_summary(cx, item, param_count);
    if let SummaryKind::Poly(summary) = &built {
        drop(prelayout_concrete(cx, summary));
    }
    built
}

fn prelayout_concrete(cx: &mut Context<'_>, summary: &PolySummary) -> Result<()> {
    let locals = summary.locals.iter().map(|l| l.ty);
    for ty in summary.template.exprs.iter().copied().chain(locals) {
        if ty != Ty::NONE && !cx.types.has_params(ty) && matches!(cx.types.kind(ty), Kind::Adt(..) | Kind::Tuple(_) | Kind::Array(..)) {
            cx.layout(ty)?;
        }
    }
    Ok(())
}

fn instantiate_summary(cx: &mut Context<'_>, instance: Instance, summary: &PolySummary, unlaid: &mut Vec<String>, before: (usize, usize)) -> Option<Result<Arc<Typed>, String>> {
    let concrete_args = cx.types.substs(instance.1).to_vec();
    if concrete_args.len() != usize::from(summary.param_count) {
        return None;
    }
    let Ok((raw_typed, concrete_layouts)) = super::infer::detached(cx, instance.0, |infer| {
        let mut sub = SubState::new(&concrete_args);
        eval_projections(infer, &mut sub, &summary.projections)?;
        let calls = check_obligations(infer, &mut sub, &summary.obligations)?;
        let offsets = eval_offset_holes(infer, &mut sub, &summary.const_holes)?;
        check_const_holes(infer, &mut sub, &summary.const_holes)?;
        let tables = instantiate_tables(infer, &mut sub, instance.0, &summary.template, (&calls, &offsets))?;
        let locals = summary.locals.iter().map(|local| Ok(Local { ty: sub_ty(infer, &mut sub, local.ty)? })).collect::<Result<Vec<_>>>()?;
        let ret = summary.ret.map(|ty| sub_ty(infer, &mut sub, ty)).transpose()?;
        let layouts = summary.layout_tys.iter().map(|&ty| sub_ty(infer, &mut sub, ty)).collect::<Result<Vec<_>>>()?;
        let typed = Typed { tables: Arc::new(tables), locals, params: summary.params.to_vec(), ret, drops: Vec::new(), shape: Shape::Body };
        Ok((typed, layouts))
    }) else {
        return None;
    };
    let Ok(settled) = super::infer::settle(cx, instance, raw_typed) else { return None };
    let typed = Arc::new(settled);
    let laid = lay_out_instantiated(cx, &concrete_layouts, &typed).map_err(|err| unlaid.push(format!("{}: {err:#}", cx.name(instance.0))));
    if laid.is_ok() && cx.quiet_mark() == before {
        cx.store.finals.insert(instance, Arc::clone(&typed));
    }
    Some(Ok(typed))
}

fn lay_out_instantiated(cx: &mut Context<'_>, layouts: &[Ty], typed: &Typed) -> Result<()> {
    for &ty in layouts {
        if ty != Ty::NONE && matches!(cx.types.kind(ty), Kind::Adt(..) | Kind::Tuple(_) | Kind::Array(..)) {
            cx.layout(ty)?;
        }
    }
    if !typed.closures.is_empty() {
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
    let name = infer.cx.name(def.assoc).to_string();
    let Some(value) = infer.picked_alias(def.of, self_ty, &trait_args, (&name, &own_args))? else { bail!("unnormalized projection `{name}` on {}", infer.show(self_ty)) };
    infer.settle_projections()?;
    let zonked = infer.zonk(value);
    if infer.cx.types.has_params(zonked) || infer.has_vars(zonked) || matches!(infer.cx.types.kind(zonked), Kind::Never) {
        bail!("projection `{name}` did not resolve to a concrete type");
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

fn instantiate_tables(infer: &mut Infer<'_, '_>, sub: &mut SubState<'_>, item: ItemRef, tpl: &BodyTables, (calls, offset_overrides): (&[CallOverride], &[(ExprId, u64)])) -> Result<BodyTables> {
    let exprs = tpl.exprs.iter().map(|&ty| sub_ty(infer, sub, ty)).collect::<Result<Vec<_>>>()?.into_boxed_slice();
    let pats: PackedMap<PatId, Ty> = tpl.pats.try_map_values(|&ty| sub_ty(infer, sub, ty))?;
    let res = instantiate_res(infer, sub, &tpl.res, calls)?;
    let pat_res = tpl.pat_res.try_map_values(|r| sub_res(infer, sub, r))?;
    let derefs = tpl.derefs.try_map_values(|r| sub_res(infer, sub, r))?;
    let operators = instantiate_operators(infer, sub, item, &tpl.operators, &exprs)?;
    let (fors, tries, awaits, coercions, closures, try_blocks) = instantiate_flow_tables(infer, sub, tpl)?;
    let (autoderefs, overloaded, box_frees) = apply_call_tables(infer, sub, tpl, calls)?;
    let offsets = apply_offsets(&tpl.offsets, offset_overrides);
    Ok(BodyTables { syntax: Arc::clone(&tpl.syntax), exprs, res, autoderefs, derefs, operators, coercions, offsets, pat_res, overloaded, pats, fors, tries, awaits, closures, try_blocks, box_frees })
}

fn instantiate_operators(infer: &mut Infer<'_, '_>, sub: &mut SubState<'_>, item: ItemRef, tpl_ops: &PackedMap<ExprId, Res>, exprs: &[Ty]) -> Result<PackedMap<ExprId, Res>> {
    if tpl_ops.is_empty() {
        return Ok(PackedMap::default());
    }
    let body = infer.cx.body(item)?;
    let mut pairs = Vec::with_capacity(tpl_ops.len());
    for &(expr, ref r) in tpl_ops.as_slice() {
        let step_ty = match *r {
            Res::Fn(_, s) => infer.cx.types.substs(s).first().copied(),
            _ => None,
        };
        let concrete_step = step_ty.map(|t| sub_ty(infer, sub, t)).transpose()?;
        if infer.is_builtin_operator(&body, expr, exprs, concrete_step) {
            continue;
        }
        pairs.push((expr, sub_res(infer, sub, r)?));
    }
    Ok(PackedMap::pack(pairs))
}

type FlowTables = (PackedMap<ExprId, For>, PackedMap<ExprId, Tried>, PackedMap<ExprId, Awaited>, PackedMap<ExprId, Coercion>, PackedMap<ExprId, Closure>, PackedMap<ExprId, Res>);

fn instantiate_flow_tables(infer: &mut Infer<'_, '_>, sub: &mut SubState<'_>, tpl: &BodyTables) -> Result<FlowTables> {
    let fors = tpl.fors.try_map_values(|f| sub_for(infer, sub, f))?;
    let tries = tpl.tries.try_map_values(|t| sub_tried(infer, sub, t))?;
    let awaits = tpl.awaits.try_map_values(|a| sub_awaited(infer, sub, a))?;
    let coercions = tpl.coercions.try_map_values(|c| sub_coercion(infer, sub, c))?;
    let closures = tpl.closures.try_map_values(|c| Ok::<_, anyhow::Error>(Closure { ret: sub_ty(infer, sub, c.ret)?, ..c.clone() }))?;
    let try_blocks = tpl.try_blocks.try_map_values(|r| sub_res(infer, sub, r))?;
    Ok((fors, tries, awaits, coercions, closures, try_blocks))
}

fn instantiate_res(infer: &mut Infer<'_, '_>, sub: &mut SubState<'_>, tpl_res: &PackedMap<ExprId, Res>, calls: &[CallOverride]) -> Result<PackedMap<ExprId, Res>> {
    if calls.is_empty() {
        return tpl_res.try_map_values(|r| sub_res(infer, sub, r));
    }
    let mut pairs = Vec::with_capacity(tpl_res.len() + calls.len());
    for &(expr, ref r) in tpl_res.as_slice() {
        if let Some(ov) = calls.iter().find(|c| c.expr == expr) {
            if let Some(new_res) = &ov.res {
                pairs.push((expr, *new_res));
            }
        } else {
            pairs.push((expr, sub_res(infer, sub, r)?));
        }
    }
    for ov in calls {
        if !tpl_res.contains_key(&ov.expr)
            && let Some(new_res) = &ov.res
        {
            pairs.push((ov.expr, *new_res));
        }
    }
    Ok(PackedMap::pack(pairs))
}

type CallTables = (PackedMap<ExprId, u32>, PackedSet<ExprId>, PackedMap<ExprId, Res>);

fn apply_call_tables(infer: &mut Infer<'_, '_>, sub: &mut SubState<'_>, tpl: &BodyTables, calls: &[CallOverride]) -> Result<CallTables> {
    let mut box_frees = tpl.box_frees.try_map_values(|r| sub_res(infer, sub, r))?;
    if calls.is_empty() {
        return Ok((tpl.autoderefs.clone(), tpl.overloaded.clone(), box_frees));
    }
    let mut deref_pairs: Vec<(ExprId, u32)> = tpl.autoderefs.as_slice().iter().copied().filter(|(e, _)| !calls.iter().any(|c| c.expr == *e)).collect();
    let mut over_set: Vec<ExprId> = tpl.overloaded.as_slice().to_vec();
    let mut box_pairs: Vec<(ExprId, Res)> = box_frees.as_slice().to_vec();
    for ov in calls {
        if let Some(steps) = ov.autoderefs {
            deref_pairs.push((ov.expr, steps));
        }
        if ov.overloaded {
            over_set.push(ov.expr);
        }
        if let Some(free) = &ov.box_free {
            box_pairs.push((ov.expr, *free));
        }
    }
    if calls.iter().any(|c| c.box_free.is_some()) {
        box_frees = PackedMap::pack(box_pairs);
    }
    Ok((PackedMap::pack(deref_pairs), PackedSet::pack(over_set), box_frees))
}

fn apply_offsets(tpl_offsets: &PackedMap<ExprId, u64>, overrides: &[(ExprId, u64)]) -> PackedMap<ExprId, u64> {
    if overrides.is_empty() {
        return tpl_offsets.clone();
    }
    let mut pairs = tpl_offsets.as_slice().to_vec();
    pairs.extend_from_slice(overrides);
    PackedMap::pack(pairs)
}

fn sub_for(infer: &mut Infer<'_, '_>, sub: &mut SubState<'_>, f: &For) -> Result<For> {
    Ok(For { into_iter: sub_res(infer, sub, &f.into_iter)?, next: sub_res(infer, sub, &f.next)?, iter: sub_ty(infer, sub, f.iter)?, option: sub_ty(infer, sub, f.option)?, some: f.some })
}

fn sub_tried(infer: &mut Infer<'_, '_>, sub: &mut SubState<'_>, t: &Tried) -> Result<Tried> {
    Ok(Tried { branch: sub_res(infer, sub, &t.branch)?, from_residual: sub_res(infer, sub, &t.from_residual)?, flow: sub_ty(infer, sub, t.flow)?, ..*t })
}

fn sub_awaited(infer: &mut Infer<'_, '_>, sub: &mut SubState<'_>, a: &Awaited) -> Result<Awaited> {
    Ok(Awaited { into_future: sub_res(infer, sub, &a.into_future)?, future: sub_ty(infer, sub, a.future)?, poll: sub_res(infer, sub, &a.poll)?, polled: sub_ty(infer, sub, a.polled)? })
}

fn sub_coercion(infer: &mut Infer<'_, '_>, sub: &mut SubState<'_>, coercion: &Coercion) -> Result<Coercion> {
    match coercion {
        Coercion::Len(len) => Ok(Coercion::Len(sub_ty(infer, sub, *len)?)),
        Coercion::Autos | Coercion::Upcast(_) => Ok(coercion.clone()),
        Coercion::Reify(res) => Ok(Coercion::Reify(sub_res(infer, sub, res)?)),
        Coercion::Deref(steps, to) => sub_deref_coercion(infer, sub, steps, *to),
        Coercion::Vtable(vtable) if !infer.cx.types.has_params(vtable.ty) && !infer.cx.types.has_params(vtable.object) => Ok(Coercion::Vtable(vtable.clone())),
        Coercion::Vtable(_) => bail!("vtable coercion requires monomorphic fallback"),
    }
}

fn sub_deref_coercion(infer: &mut Infer<'_, '_>, sub: &mut SubState<'_>, steps: &[(Ty, Option<Res>)], to: Ty) -> Result<Coercion> {
    let mut mapped = Vec::with_capacity(steps.len());
    for (ty, res) in steps {
        let next_res = match res {
            Some(r) => Some(sub_res(infer, sub, r)?),
            None => None,
        };
        mapped.push((sub_ty(infer, sub, *ty)?, next_res));
    }
    Ok(Coercion::Deref(mapped, sub_ty(infer, sub, to)?))
}

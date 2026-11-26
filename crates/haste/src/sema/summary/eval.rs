use anyhow::{Result, bail};
use rustc_hash::FxHashMap as HashMap;

use crate::body::ExprId;
use crate::names::ItemRef;

use super::super::infer::{Infer, Res, SiteKey};
use super::super::substs::{ItemList, ProjId, Substs};
use super::super::ty::{Kind, Ty, Types, object_parts};
use super::inst::{TyFolder, fold_substs, fold_ty};
use super::{CallOverride, ConstHole, OpCheck, PolyObligation, PolySummary};

pub(super) struct SubState<'a> {
    pub(super) args: &'a [Ty],
    pub(super) projs: HashMap<ProjId, Ty>,
    pub(super) tys: HashMap<Ty, Ty>,
}

impl<'a> SubState<'a> {
    pub(super) fn new(args: &'a [Ty]) -> Self {
        Self {
            args,
            projs: HashMap::default(),
            tys: HashMap::default(),
        }
    }
}

pub(super) type ResolvedSites = (Box<[(SiteKey, ItemRef, Substs)]>, Box<[ExprId]>);

pub(super) fn resolve_instance_calls(
    infer: &mut Infer<'_, '_>,
    sub: &mut SubState<'_>,
    summary: &PolySummary,
    calls: &[CallOverride],
    check_only: bool,
) -> Result<ResolvedSites> {
    let mut resolved = Vec::with_capacity(
        summary.direct_calls.len()
            + summary.call_holes.len()
            + summary.op_holes.len()
            + calls.len(),
    );
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
            if !check_only {
                builtin_ops.push(expr);
            }
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
    if !check_only {
        resolved.sort_unstable_by_key(|&(key, ..)| key);
        builtin_ops.sort_unstable();
    }
    Ok((resolved.into_boxed_slice(), builtin_ops.into_boxed_slice()))
}

fn sub_op_check(
    infer: &mut Infer<'_, '_>,
    sub: &mut SubState<'_>,
    check: OpCheck,
) -> Result<OpCheck> {
    Ok(match check {
        OpCheck::Unary(ty) => OpCheck::Unary(sub_ty(infer, sub, ty)?),
        OpCheck::Binary(op, l, r) => {
            OpCheck::Binary(op, sub_ty(infer, sub, l)?, sub_ty(infer, sub, r)?)
        }
        OpCheck::AssignOp(op, l, r) => {
            OpCheck::AssignOp(op, sub_ty(infer, sub, l)?, sub_ty(infer, sub, r)?)
        }
        OpCheck::Index(step, idx) => OpCheck::Index(
            step.map(|s| sub_ty(infer, sub, s)).transpose()?,
            sub_ty(infer, sub, idx)?,
        ),
        OpCheck::Never => OpCheck::Never,
    })
}

pub(super) fn eval_projections(
    infer: &mut Infer<'_, '_>,
    sub: &mut SubState<'_>,
    projections: &[ProjId],
) -> Result<()> {
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
        && !infer
            .cx
            .types
            .substs(trait_substs)
            .iter()
            .any(|&ty| infer.has_vars(ty))
        && !infer.cx.types.substs_have_params(own_substs)
        && !infer
            .cx
            .types
            .substs(own_substs)
            .iter()
            .any(|&ty| infer.has_vars(ty));
    let key = (def.of, self_ty, trait_substs, def.assoc, own_substs);
    if cacheable && let Some(cached) = infer.cx.store.lookups.assoc_types.get(&key) {
        sub.projs.insert(proj_id, cached);
        return Ok(cached);
    }
    let before = infer.cx.quiet_mark();
    let name = infer.cx.name(def.assoc).to_string();
    let Some(value) = infer.picked_alias(def.of, self_ty, &trait_args, (&name, &own_args))? else {
        bail!(
            "unnormalized projection `{name}` on {}",
            infer.show(self_ty)
        )
    };
    infer.settle_projections()?;
    let zonked = infer.zonk(value);
    if infer.cx.types.has_params(zonked)
        || infer.has_vars(zonked)
        || matches!(infer.cx.types.kind(zonked), Kind::Never)
    {
        bail!("projection `{name}` did not resolve to a concrete type");
    }
    if cacheable && infer.cx.quiet_mark() == before {
        infer.cx.store.lookups.assoc_types.insert(key, zonked);
    }
    sub.projs.insert(proj_id, zonked);
    Ok(zonked)
}

struct SummaryFolder<'a, 'b, 'c, 'd> {
    infer: &'a mut Infer<'b, 'c>,
    sub: &'a mut SubState<'d>,
}

impl TyFolder for SummaryFolder<'_, '_, '_, '_> {
    type Error = anyhow::Error;
    fn types(&self) -> &Types {
        &self.infer.cx.types
    }
    fn args(&self) -> &[Ty] {
        self.sub.args
    }
    fn get_cached(&self, ty: Ty) -> Option<Ty> {
        self.sub.tys.get(&ty).copied()
    }
    fn set_cached(&mut self, ty: Ty, mapped: Ty) {
        self.sub.tys.insert(ty, mapped);
    }
    fn param_oob(&self, idx: u16) -> Self::Error {
        anyhow::anyhow!("param {idx} out of bounds")
    }
    fn fold_proj(&mut self, proj_id: ProjId) -> Result<Ty> {
        eval_one_proj(self.infer, self.sub, proj_id)
    }
}

pub(super) fn sub_ty(infer: &mut Infer<'_, '_>, sub: &mut SubState<'_>, ty: Ty) -> Result<Ty> {
    fold_ty(&mut SummaryFolder { infer, sub }, ty)
}

fn sub_substs(infer: &mut Infer<'_, '_>, sub: &mut SubState<'_>, substs: Substs) -> Result<Substs> {
    fold_substs(&mut SummaryFolder { infer, sub }, substs)
}

fn sub_substs_vec(
    infer: &mut Infer<'_, '_>,
    sub: &mut SubState<'_>,
    substs: Substs,
) -> Result<Vec<Ty>> {
    let raw = infer.cx.types.substs(substs).to_vec();
    raw.into_iter().map(|ty| sub_ty(infer, sub, ty)).collect()
}

fn sub_res(infer: &mut Infer<'_, '_>, sub: &mut SubState<'_>, res: &Res) -> Result<Res> {
    match *res {
        Res::Local(local) => Ok(Res::Local(local)),
        Res::Const(ty) => Ok(Res::Const(sub_ty(infer, sub, ty)?)),
        Res::Ctor(item, variant, args) => {
            Ok(Res::Ctor(item, variant, sub_substs(infer, sub, args)?))
        }
        Res::Fn(item, args) => {
            let mapped = sub_substs(infer, sub, args)?;
            infer.trait_call(item, mapped)
        }
    }
}

pub(super) fn check_obligations(
    infer: &mut Infer<'_, '_>,
    sub: &mut SubState<'_>,
    obligations: &[PolyObligation],
) -> Result<Vec<CallOverride>> {
    let mut calls = Vec::new();
    for obligation in obligations {
        match *obligation {
            PolyObligation::Prove { of, self_ty, args } => {
                let concrete_self = sub_ty(infer, sub, self_ty)?;
                let concrete_args = sub_substs_vec(infer, sub, args)?;
                if !concrete_holds(infer, of, concrete_self, &concrete_args)? {
                    bail!(
                        "concrete obligation `{}` failed for {}",
                        infer.cx.name(of),
                        infer.show(concrete_self)
                    );
                }
            }
            PolyObligation::Call { expr, found, tuple } => {
                let concrete_found = sub_ty(infer, sub, found)?;
                let concrete_tuple = sub_ty(infer, sub, tuple)?;
                calls.push(resolve_call_override(
                    infer,
                    expr,
                    concrete_found,
                    concrete_tuple,
                )?);
            }
        }
    }
    Ok(calls)
}

fn concrete_holds(
    infer: &mut Infer<'_, '_>,
    of: ItemRef,
    self_ty: Ty,
    args: &[Ty],
) -> Result<bool> {
    if infer.compiler_decides(of, self_ty) {
        return Ok(infer.compiler_holds(of, self_ty));
    }
    if infer.compiler_impl(of, self_ty) || infer.object_holds(of, self_ty, args) == Some(true) {
        return Ok(true);
    }
    if infer.is_callable_trait(of)
        && matches!(
            infer.cx.types.kind(self_ty),
            Kind::Closure(..) | Kind::FnDef(..) | Kind::FnPtr(..)
        )
    {
        return infer.check_callable_obligation(of, self_ty, args);
    }
    Ok(infer.select(of, self_ty, args)?.is_some())
}

fn resolve_call_override(
    infer: &mut Infer<'_, '_>,
    expr: ExprId,
    found: Ty,
    tuple: Ty,
) -> Result<CallOverride> {
    let (mut ty, mut steps) = (found, 0u32);
    while !matches!(
        infer.cx.types.kind(ty),
        Kind::Closure(..) | Kind::FnDef(..) | Kind::FnPtr(..) | Kind::Ptr(..)
    ) && let Some(inner) = infer.builtin_deref(ty)
    {
        (ty, steps) = (inner, steps + 1);
    }
    let autoderefs = (steps > 0).then_some(steps);
    match infer.cx.types.kind(ty).clone() {
        Kind::Closure(..) => {
            let Some(call_once) = infer.cx.call_once() else {
                bail!("core has no `FnOnce::call_once`")
            };
            let res = Some(Res::fn_of(&infer.cx.types, call_once, &[ty, tuple]));
            Ok(CallOverride {
                expr,
                res,
                autoderefs,
                overloaded: false,
                box_free: None,
            })
        }
        Kind::FnDef(item, substs) => {
            let res = Some(infer.trait_call(item, substs)?);
            Ok(CallOverride {
                expr,
                res,
                autoderefs,
                overloaded: false,
                box_free: None,
            })
        }
        Kind::FnPtr(..) => Ok(CallOverride {
            expr,
            res: None,
            autoderefs,
            overloaded: false,
            box_free: None,
        }),
        Kind::Dyn(of, parts, members, _) if infer.is_callable_trait(of) => {
            resolve_dyn_call(infer, (expr, found, ty), (of, parts, members), autoderefs)
        }
        Kind::Adt(..) => resolve_overloaded_call(infer, expr, found, tuple),
        _ => bail!("unsupported concrete callable {}", infer.show(ty)),
    }
}

fn resolve_dyn_call(
    infer: &mut Infer<'_, '_>,
    (expr, found, ty): (ExprId, Ty, Ty),
    (of, parts, members): (ItemRef, Substs, ItemList),
    autoderefs: Option<u32>,
) -> Result<CallOverride> {
    let dyn_tuple = *object_parts(infer.cx.types.substs(parts), infer.cx.types.items(members))
        .0
        .first()
        .ok_or_else(|| anyhow::anyhow!("callable trait object without tuple"))?;
    let name = super::super::infer::callable_method_name(infer.cx.lang(of));
    let box_free = matches!(name, "call_once" | "async_call_once")
        .then(|| resolve_box_free(infer, found, autoderefs))
        .transpose()?;
    let Some(method) = infer.cx.trait_member(of, name) else {
        bail!("missing `{name}`")
    };
    let res = Some(Res::fn_of(&infer.cx.types, method, &[ty, dyn_tuple]));
    Ok(CallOverride {
        expr,
        res,
        autoderefs,
        overloaded: false,
        box_free,
    })
}

fn resolve_box_free(infer: &mut Infer<'_, '_>, found: Ty, autoderefs: Option<u32>) -> Result<Res> {
    if infer.cx.boxed(found).is_none() || autoderefs != Some(1) {
        bail!("`dyn FnOnce` called by value without `Box`");
    }
    infer
        .own_drop(found)?
        .ok_or_else(|| anyhow::anyhow!("`Box` has no `Drop` impl"))
}

fn resolve_overloaded_call(
    infer: &mut Infer<'_, '_>,
    expr: ExprId,
    found: Ty,
    tuple: Ty,
) -> Result<CallOverride> {
    let (mut ty, mut steps) = (found, 0u32);
    loop {
        if let Some(method) = infer.callable_impl(ty, tuple)? {
            let substs = infer.cx.types.intern_substs(&[ty, tuple]);
            let res = Some(infer.trait_call(method, substs)?);
            return Ok(CallOverride {
                expr,
                res,
                autoderefs: (steps > 0).then_some(steps),
                overloaded: true,
                box_free: None,
            });
        }
        let Some(inner) = infer.builtin_deref(ty) else {
            bail!("{} is not callable", infer.show(found))
        };
        (ty, steps) = (inner, steps + 1);
    }
}

pub(super) fn eval_offset_holes(
    infer: &mut Infer<'_, '_>,
    sub: &mut SubState<'_>,
    holes: &[ConstHole],
) -> Result<Vec<(ExprId, u64)>> {
    let mut out = Vec::new();
    for hole in holes {
        let ConstHole::OffsetOf { expr, steps } = hole else {
            continue;
        };
        let mut offset = 0u64;
        for &(step_ty, index) in &**steps {
            let concrete = sub_ty(infer, sub, step_ty)?;
            let layout = infer.cx.layout(concrete)?;
            offset += super::super::layout::field_offset(&layout, 0, index) as u64;
        }
        out.push((*expr, offset));
    }
    Ok(out)
}

pub(super) fn check_const_holes(
    infer: &mut Infer<'_, '_>,
    sub: &mut SubState<'_>,
    holes: &[ConstHole],
) -> Result<()> {
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

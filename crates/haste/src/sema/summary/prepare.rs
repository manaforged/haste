use anyhow::{Result, bail};
use rustc_hash::FxHashSet as HashSet;

use crate::body::{Body, Expr, ExprId, UnOp};
use crate::names::ItemRef;

use super::super::Context;
use super::super::infer::{BodyTables, Coercion, Res, SiteKey};
use super::super::ty::{Kind, Ty};
use super::{OpCheck, PolyObligation, PolySummary, frozen};

pub(super) fn prepare_summary(
    cx: &mut Context<'_>,
    item: ItemRef,
    summary: &mut PolySummary,
) -> Result<()> {
    for c in summary.template.coercions.values() {
        if let Some(vt) = c.vtable()
            && (cx.types.has_params(vt.ty) || cx.types.has_params(vt.object))
        {
            bail!("parametric vtable coercion");
        }
    }
    extract_call_holes(cx, summary);
    if !summary.template.operators.is_empty() {
        let body = cx.body(item)?;
        summary.op_holes = extract_op_holes(cx, &body, &summary.template);
    }
    extract_drops_and_tys(cx, summary)
}

fn extract_call_holes(cx: &Context<'_>, summary: &mut PolySummary) {
    let types = &cx.types;
    let tpl = &summary.template;
    let call_obs: HashSet<ExprId> = summary
        .obligations
        .iter()
        .filter_map(|o| match *o {
            PolyObligation::Call { expr, .. } => Some(expr),
            _ => None,
        })
        .collect();
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
    tpl.pat_res
        .as_slice()
        .iter()
        .for_each(|&(id, r)| push(SiteKey::PatRes(id), r));
    tpl.derefs
        .as_slice()
        .iter()
        .for_each(|&((id, step), r)| push(SiteKey::Deref(id, step), r));
    for &(id, r) in tpl.operators.as_slice() {
        if !matches!(r, Res::Fn(i, s) if types.substs_have_params(s) && cx.owner_trait(i).is_some())
        {
            push(SiteKey::Operator(id), r);
        }
    }
    tpl.try_blocks
        .as_slice()
        .iter()
        .for_each(|&(id, r)| push(SiteKey::TryBlock(id), r));
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
        if let Some(vt) = c.vtable() {
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

fn extract_op_holes(
    cx: &Context<'_>,
    body: &Body,
    tpl: &BodyTables,
) -> Box<[(ExprId, OpCheck, Res)]> {
    let types = &cx.types;
    tpl.operators
        .as_slice()
        .iter()
        .filter(|&&(_, r)| {
            matches!(r, Res::Fn(i, s) if types.substs_have_params(s) && cx.owner_trait(i).is_some())
        })
        .map(|&(expr, r)| {
            let check = match *body.expr(expr) {
                Expr::Unary(op, operand) if op != UnOp::Deref => {
                    OpCheck::Unary(tpl.exprs[operand.0 as usize])
                }
                Expr::Binary(op, lhs, rhs) => {
                    OpCheck::Binary(op, tpl.exprs[lhs.0 as usize], tpl.exprs[rhs.0 as usize])
                }
                Expr::AssignOp(op, lhs, rhs) => {
                    OpCheck::AssignOp(op, tpl.exprs[lhs.0 as usize], tpl.exprs[rhs.0 as usize])
                }
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

pub(super) fn is_aggregate(cx: &Context<'_>, ty: Ty) -> bool {
    ty != Ty::NONE
        && matches!(
            cx.types.kind(ty),
            Kind::Adt(..) | Kind::Tuple(_) | Kind::Array(..)
        )
}

fn owned_tys(summary: &PolySummary) -> impl Iterator<Item = Ty> + '_ {
    summary
        .locals
        .iter()
        .map(|l| l.ty)
        .chain(summary.ret)
        .chain(summary.template.owned_tys())
}

pub(super) fn lay_out_fixed(cx: &mut Context<'_>, summary: &PolySummary) {
    let fixed = frozen::closure_free(summary).then(|| owned_tys(summary));
    for ty in fixed.into_iter().flatten() {
        if is_aggregate(cx, ty) && !cx.types.has_params(ty) && !cx.types.has_var_flag(ty) {
            cx.layout(ty).ok();
        }
    }
}

fn record_drop_ty(
    cx: &mut Context<'_>,
    ty: Ty,
    fixed_drops: &mut Vec<Res>,
    drop_tys: &mut Vec<Ty>,
) -> Result<()> {
    if cx.types.has_var_flag(ty) || cx.types.has_closure(ty) {
        return Ok(());
    }
    if !cx.types.has_params(ty) {
        if is_aggregate(cx, ty) {
            cx.layout(ty).ok();
        }
        fixed_drops.extend(super::super::infer::glue_call(cx, ty)?);
    } else if !matches!(
        cx.types.kind(ty),
        Kind::Prim(_)
            | Kind::Ref(..)
            | Kind::Ptr(..)
            | Kind::FnDef(..)
            | Kind::FnPtr(..)
            | Kind::UnsafeFnPtr(..)
            | Kind::Never
            | Kind::Value(_)
    ) {
        drop_tys.push(ty);
    }
    Ok(())
}

fn extract_drops_and_tys(cx: &mut Context<'_>, summary: &mut PolySummary) -> Result<()> {
    let mut seen = HashSet::default();
    let (mut fixed_drops, mut drop_tys, mut parametric_tys) = (Vec::new(), Vec::new(), Vec::new());
    for ty in owned_tys(summary) {
        if ty == Ty::NONE || !seen.insert(ty) {
            continue;
        }
        record_drop_ty(cx, ty, &mut fixed_drops, &mut drop_tys)?;
        if cx.types.has_params(ty) {
            parametric_tys.push(ty);
        }
    }
    summary.fixed_drops = fixed_drops.into_boxed_slice();
    summary.drop_tys = drop_tys.into_boxed_slice();
    summary.parametric_tys = parametric_tys.into_boxed_slice();
    Ok(())
}

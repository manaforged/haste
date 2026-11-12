use std::sync::Arc;

use anyhow::{Result, bail};

use crate::body::ExprId;
use crate::names::ItemRef;
use crate::sema::Context;
use crate::sema::infer::{Coercion, LazyTables, Res, Shape, SiteKey, Typed, drop_fn, field_types, type_name};
use crate::sema::reach::Reached;
use crate::sema::substs::Substs;
use crate::sema::ty::{Kind, Ty};

use super::moves::{BOX_STEP, Root};
use super::object::build_vtable_id;
use super::stencil::{RelocVec, SymbolSource};
use super::Slots;

pub(super) const CLOSURE_ENV_STEP: u32 = u32::MAX - 1;

#[derive(Clone, Debug)]
pub(super) enum TyOrigin {
    Subst(u16),
    Local(u32),
    Expr(ExprId),
    ForIter(ExprId),
    AwaitFuture(ExprId),
    SiteSubst(Box<SiteOrigin>, u8),
    Step { base: Box<TyOrigin>, variant: u32, field: u32 },
}

impl TyOrigin {
    pub(super) fn from_root(root: Root, typed: &Typed) -> Self {
        match root {
            Root::Local(local) => Self::Local(local),
            Root::Temp(expr) => Self::Expr(expr),
            Root::Held(expr) => match &typed.tables {
                LazyTables::Eager(t) if t.awaits.contains_key(&expr) => Self::AwaitFuture(expr),
                LazyTables::Poly(p) if p.summary.template.awaits.contains_key(&expr) => Self::AwaitFuture(expr),
                _ => Self::ForIter(expr),
            },
        }
    }

    pub(super) fn stepped(mut self, path: &[(u32, u32)]) -> Self {
        for &(variant, field) in path {
            self = Self::Step { base: Box::new(self), variant, field };
        }
        self
    }

    pub(super) fn step(self, variant: u32, field: u32) -> Self {
        Self::Step { base: Box::new(self), variant, field }
    }
}

#[derive(Clone, Debug)]
pub(super) enum SiteOrigin {
    Table(SiteKey),
    PointerFnDef { expr: ExprId, callee: ExprId },
    DropGlue(TyOrigin),
    DropFn(TyOrigin),
    ConstEvalSelect(Box<SiteOrigin>),
    ShapeGlueDrop,
    ShapeGlueResume,
    ShapeUntuple,
    ShapeClone(u16),
}

#[derive(Clone, Debug)]
pub(super) enum RelocStep {
    Const(SymbolSource),
    Promoted { seq: u32, size: u32 },
    VtableAt(ExprId),
    SiteSlot(SiteOrigin),
    SiteInline { origin: SiteOrigin, plan: Arc<RelocPlan> },
    TypeId(TyOrigin),
    TypeName(TyOrigin),
}

/// Parametric relocation plan recorded on a stencil leader and evaluated in `O(|relocs|)` per follower.
#[derive(Clone, Debug, Default)]
pub struct RelocPlan {
    pub(super) steps: Box<[RelocStep]>,
}

impl RelocPlan {
    pub(super) fn from_vec(steps: Vec<RelocStep>) -> Self {
        Self { steps: steps.into_boxed_slice() }
    }

    pub(super) fn has_promoted(&self) -> bool {
        self.steps.iter().any(|step| match step {
            RelocStep::Promoted { .. } => true,
            RelocStep::SiteInline { plan, .. } => plan.has_promoted(),
            _ => false,
        })
    }
}

/// Evaluates `plan` against reached concrete instance `index` to produce its exact relocation vector.
pub fn eval_relocs(cx: &mut Context<'_>, reached: &Reached, slots: &dyn Slots, plan: &RelocPlan, index: usize) -> Result<RelocVec> {
    let mut out = RelocVec::new();
    eval_into(cx, reached, slots, plan, index, &mut out)?;
    Ok(out)
}

fn eval_into(cx: &mut Context<'_>, reached: &Reached, slots: &dyn Slots, plan: &RelocPlan, index: usize, out: &mut RelocVec) -> Result<()> {
    let ((item, substs), result) = &reached.list[index];
    let typed = match result {
        Ok(typed) => typed.as_ref(),
        Err(err) => bail!("{}: {err}", cx.name(*item)),
    };
    let substs = *substs;
    for step in &plan.steps {
        match step {
            RelocStep::Const(src) => out.push(src.clone()),
            RelocStep::Promoted { seq, size } => out.push(SymbolSource::PromotedCell { owner: index, seq: *seq, size: *size }),
            RelocStep::VtableAt(expr) => {
                let id = eval_vtable_at(cx, reached, slots, typed, *expr)?;
                out.push(SymbolSource::Vtable(id));
            }
            RelocStep::SiteSlot(origin) => {
                let (callee_item, callee_substs) = eval_site(cx, reached, typed, substs, origin)?;
                let callee_idx = reached_index_of(cx, reached, slots, (callee_item, callee_substs))?;
                out.push(SymbolSource::Slot(callee_idx));
            }
            RelocStep::SiteInline { origin, plan: sub_plan } => {
                let (callee_item, callee_substs) = eval_site(cx, reached, typed, substs, origin)?;
                let callee_idx = reached_index_of(cx, reached, slots, (callee_item, callee_substs))?;
                eval_into(cx, reached, slots, sub_plan, callee_idx, out)?;
            }
            RelocStep::TypeId(origin) => {
                let ty = eval_ty(cx, reached, typed, substs, origin)?;
                out.push(SymbolSource::Word(u64::from(ty.0)));
            }
            RelocStep::TypeName(origin) => {
                let ty = eval_ty(cx, reached, typed, substs, origin)?;
                let name = type_name(cx, ty);
                let id = slots.rodata(name.as_bytes(), &[]);
                out.push(SymbolSource::Rodata(id));
                out.push(SymbolSource::Word(name.len() as u64));
            }
        }
    }
    Ok(())
}

pub(super) fn reached_index_of(cx: &Context<'_>, reached: &Reached, slots: &dyn Slots, instance: (ItemRef, Substs)) -> Result<usize> {
    if let Some(&idx) = reached.index.get(&instance)
        && !reached.is_untyped(idx)
    {
        return Ok(idx);
    }
    slots.want(instance);
    bail!("a use of `{}` that was not reached", cx.name(instance.0))
}

fn eval_vtable_at(cx: &mut Context<'_>, reached: &Reached, slots: &dyn Slots, typed: &Typed, expr: ExprId) -> Result<super::stencil::VtableId> {
    let vtable = match &typed.tables {
        LazyTables::Eager(t) => match t.coercions.get(&expr) {
            Some(Coercion::Vtable(vt)) => vt,
            _ => bail!("missing vtable coercion at {:?}", expr),
        },
        LazyTables::Poly(p) => match p.summary.template.coercions.get(&expr) {
            Some(Coercion::Vtable(vt)) => vt,
            _ => bail!("missing poly vtable coercion at {:?}", expr),
        },
    };
    build_vtable_id(cx, reached, slots, vtable)
}

fn eval_site(cx: &mut Context<'_>, reached: &Reached, typed: &Typed, substs: Substs, origin: &SiteOrigin) -> Result<(ItemRef, Substs)> {
    match origin {
        SiteOrigin::Table(key) => eval_table_site(typed, *key).ok_or_else(|| anyhow::anyhow!("missing site {:?}", key)),
        SiteOrigin::PointerFnDef { expr, callee } => eval_pointer_fndef(cx, reached, typed, *expr, *callee),
        SiteOrigin::DropGlue(ty_origin) => {
            let ty = eval_ty(cx, reached, typed, substs, ty_origin)?;
            let Some(glue) = cx.lang_item("drop_glue") else { bail!("core has no drop glue") };
            Ok((glue, cx.types.intern_substs(&[ty])))
        }
        SiteOrigin::DropFn(ty_origin) => {
            let ty = eval_ty(cx, reached, typed, substs, ty_origin)?;
            let Some(Res::Fn(item, fn_substs)) = drop_fn(cx, ty)? else { bail!("type has no `Drop::drop`") };
            Ok((item, fn_substs))
        }
        SiteOrigin::ConstEvalSelect(inner) => {
            let (_, inner_substs) = eval_site(cx, reached, typed, substs, inner)?;
            let Some(&fn_ty) = cx.types.substs(inner_substs).get(2) else { bail!("const_eval_select missing runtime fn") };
            let Kind::FnDef(item, fn_substs) = *cx.types.kind(fn_ty) else { bail!("const_eval_select runtime arg is not FnDef") };
            Ok((item, fn_substs))
        }
        SiteOrigin::ShapeGlueDrop => match &typed.shape {
            Shape::Glue(glue) => match glue.drop {
                Some(Res::Fn(item, fn_substs)) => Ok((item, fn_substs)),
                _ => bail!("glue shape has no own drop"),
            },
            _ => bail!("expected glue shape"),
        },
        SiteOrigin::ShapeGlueResume => match &typed.shape {
            Shape::Glue(glue) => match glue.resume {
                Some(Res::Fn(item, fn_substs)) => Ok((item, fn_substs)),
                _ => bail!("glue shape has no coroutine resume"),
            },
            _ => bail!("expected glue shape"),
        },
        SiteOrigin::ShapeUntuple => match &typed.shape {
            Shape::Untuple(Res::Fn(item, fn_substs)) => Ok((*item, *fn_substs)),
            _ => bail!("expected untuple fn shape"),
        },
        SiteOrigin::ShapeClone(idx) => match &typed.shape {
            Shape::Clone(calls) => match calls.get(*idx as usize) {
                Some(Res::Fn(item, fn_substs)) => Ok((*item, *fn_substs)),
                _ => bail!("missing clone call {}", idx),
            },
            _ => bail!("expected clone shape"),
        },
    }
}

pub(super) fn eval_table_site(typed: &Typed, key: SiteKey) -> Option<(ItemRef, Substs)> {
    match &typed.tables {
        LazyTables::Poly(poly) => poly.site_fn(key),
        LazyTables::Eager(tables) => match tables.site_res(key)? {
            Res::Fn(item, substs) => Some((item, substs)),
            _ => None,
        },
    }
}

pub(super) fn eval_expr_ty(cx: &Context<'_>, typed: &Typed, expr: ExprId) -> Ty {
    match &typed.tables {
        LazyTables::Poly(poly) => poly.expr_ty(&cx.types, expr),
        LazyTables::Eager(tables) => tables.exprs[expr.0 as usize],
    }
}

pub(super) fn eval_autoderef_ty(cx: &Context<'_>, reached: &Reached, typed: &Typed, expr: ExprId, base: ExprId) -> Result<Ty> {
    let mut ty = eval_expr_ty(cx, typed, base);
    let steps = match &typed.tables {
        LazyTables::Poly(poly) => poly.autoderefs_at(expr),
        LazyTables::Eager(tables) => tables.autoderefs.get(&expr).copied().unwrap_or(0),
    };
    for step in 0..steps {
        if let Some((item, substs)) = eval_table_site(typed, SiteKey::Deref(expr, step)) {
            let Some(&idx) = reached.index.get(&(item, substs)) else { bail!("unreached deref callee") };
            let Ok(callee) = &reached.list[idx].1 else { bail!("broken deref callee") };
            let Some(ret) = callee.ret else { bail!("deref callee without return") };
            let Kind::Ref(_, inner) = *cx.types.kind(ret) else { bail!("deref callee did not return ref") };
            ty = inner;
        } else if let Some(inner) = cx.boxed(ty) {
            ty = inner;
        } else if let Kind::Ref(_, inner) | Kind::Ptr(_, inner) = *cx.types.kind(ty) {
            ty = inner;
        } else {
            bail!("cannot autoderef {:?}", cx.types.kind(ty));
        }
    }
    Ok(ty)
}

fn eval_pointer_fndef(cx: &Context<'_>, reached: &Reached, typed: &Typed, expr: ExprId, callee: ExprId) -> Result<(ItemRef, Substs)> {
    let ty = eval_autoderef_ty(cx, reached, typed, expr, callee)?;
    let Kind::FnDef(item, substs) = *cx.types.kind(ty) else { bail!("pointer call callee is not FnDef") };
    Ok((item, substs))
}

fn eval_ty(cx: &mut Context<'_>, reached: &Reached, typed: &Typed, substs: Substs, origin: &TyOrigin) -> Result<Ty> {
    match origin {
        TyOrigin::Subst(idx) => cx.types.substs(substs).get(*idx as usize).copied().ok_or_else(|| anyhow::anyhow!("subst {} out of bounds", idx)),
        TyOrigin::Local(idx) => typed.locals.get(*idx as usize).map(|l| l.ty).ok_or_else(|| anyhow::anyhow!("local {} out of bounds", idx)),
        TyOrigin::Expr(id) => Ok(eval_expr_ty(cx, typed, *id)),
        TyOrigin::ForIter(id) => match &typed.tables {
            LazyTables::Poly(p) => p.summary.template.fors.get(id).map(|f| p.sub_ty(&cx.types, f.iter)).ok_or_else(|| anyhow::anyhow!("missing for {}", id.0)),
            LazyTables::Eager(t) => t.fors.get(id).map(|f| f.iter).ok_or_else(|| anyhow::anyhow!("missing for {}", id.0)),
        },
        TyOrigin::AwaitFuture(id) => match &typed.tables {
            LazyTables::Poly(p) => p.summary.template.awaits.get(id).map(|a| p.sub_ty(&cx.types, a.future)).ok_or_else(|| anyhow::anyhow!("missing await {}", id.0)),
            LazyTables::Eager(t) => t.awaits.get(id).map(|a| a.future).ok_or_else(|| anyhow::anyhow!("missing await {}", id.0)),
        },
        TyOrigin::SiteSubst(site, idx) => {
            let (_, site_substs) = eval_site(cx, reached, typed, substs, site)?;
            cx.types.substs(site_substs).get(*idx as usize).copied().ok_or_else(|| anyhow::anyhow!("site subst {} out of bounds", idx))
        }
        TyOrigin::Step { base, variant, field } => {
            let parent = eval_ty(cx, reached, typed, substs, base)?;
            step_ty(cx, parent, *variant, *field)
        }
    }
}

fn step_ty(cx: &mut Context<'_>, parent: Ty, variant: u32, field: u32) -> Result<Ty> {
    if variant == BOX_STEP.0 {
        return cx.boxed(parent).ok_or_else(|| anyhow::anyhow!("box step on non-Box"));
    }
    if variant == CLOSURE_ENV_STEP {
        return cx.closure_env(parent).ok_or_else(|| anyhow::anyhow!("closure env step on non-closure"));
    }
    match cx.types.kind(parent).clone() {
        Kind::Tuple(items) => cx.types.substs(items).get(field as usize).copied().ok_or_else(|| anyhow::anyhow!("tuple step {} out of bounds", field)),
        Kind::Array(elem, _) | Kind::Slice(elem) => Ok(elem),
        Kind::Adt(item, args) => {
            let slice = cx.types.substs(args).to_vec();
            let variants = field_types(cx, item, &slice)?;
            variants.get(variant as usize).and_then(|v| v.get(field as usize)).copied().ok_or_else(|| anyhow::anyhow!("adt step ({}, {}) out of bounds", variant, field))
        }
        other => bail!("cannot step into {:?}", other),
    }
}

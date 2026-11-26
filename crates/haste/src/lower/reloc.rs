use std::sync::Arc;

use anyhow::{Result, bail};

use crate::body::ExprId;
use crate::names::ItemRef;
use crate::sema::Context;
use crate::sema::infer::{Coercion, Res, Shape, SiteKey, Typed, drop_fn, field_types, type_name};
use crate::sema::reach::Reached;
use crate::sema::substs::Substs;
use crate::sema::ty::{Kind, Ty};

use super::Slots;
use super::moves::{BOX_STEP, Root};
use super::object::build_vtable_id;
use super::stencil::{RelocVec, SymbolSource};

pub(super) const CLOSURE_ENV_STEP: u32 = u32::MAX - 1;

#[derive(Clone, Debug)]
pub(crate) enum TyOrigin {
    Subst(u16),
    Local(u32),
    Expr(ExprId),
    ForIter(ExprId),
    AwaitFuture(ExprId),
    SiteSubst(Box<SiteOrigin>, u8),
    Step {
        base: Box<TyOrigin>,
        variant: u32,
        field: u32,
    },
}

impl TyOrigin {
    pub(super) fn from_root(root: Root, typed: &Typed) -> Self {
        match root {
            Root::Local(local) => Self::Local(local),
            Root::Temp(expr) => Self::Expr(expr),
            Root::Held(expr) => {
                if typed.has_await_at(expr) {
                    Self::AwaitFuture(expr)
                } else {
                    Self::ForIter(expr)
                }
            }
        }
    }

    pub(super) fn stepped(mut self, path: &[(u32, u32)]) -> Self {
        for &(variant, field) in path {
            self = Self::Step {
                base: Box::new(self),
                variant,
                field,
            };
        }
        self
    }

    pub(super) fn step(self, variant: u32, field: u32) -> Self {
        Self::Step {
            base: Box::new(self),
            variant,
            field,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) enum SiteOrigin {
    Table(SiteKey),
    PointerFnDef {
        expr: ExprId,
        callee: ExprId,
    },
    DropGlue(TyOrigin),
    DropFn(TyOrigin),
    ConstEvalSelect(Box<SiteOrigin>),
    ShapeGlueDrop,
    ShapeGlueResume,
    ShapeUntuple,
    ShapeClone(u16),
    VtableMethod {
        expr: ExprId,
        slot: u16,
    },
    Known(ItemRef, Substs),
    /// The function that symbol `n` of the same plan names, counted from the plan's first symbol.
    Symbol(u32),
}

#[derive(Clone, Debug)]
pub(crate) enum RelocStep {
    Const(SymbolSource),
    Promoted {
        seq: u32,
        size: u32,
    },
    VtableAt(ExprId),
    VtableSlot {
        expr: ExprId,
        slot: u16,
    },
    SiteSlot(SiteOrigin),
    SiteInline {
        origin: SiteOrigin,
        plan: Arc<RelocPlan>,
    },
    TypeId(TyOrigin),
    TypeName(TyOrigin),
}

impl RelocStep {
    /// Whether this step names a function reified to a pointer, whose calls are calls of that function.
    pub(crate) fn is_reify(&self) -> bool {
        matches!(
            self,
            RelocStep::SiteSlot(SiteOrigin::Table(SiteKey::CoercionReify(_)))
        )
    }

    /// How many symbols this step produces.
    pub(crate) fn symbols(&self) -> usize {
        match self {
            RelocStep::TypeName(_) | RelocStep::TypeId(_) => 2,
            RelocStep::SiteInline { plan, .. } => plan.reloc_len(),
            _ => 1,
        }
    }

    fn step_has_calls(&self) -> bool {
        match self {
            RelocStep::SiteSlot(_)
            | RelocStep::VtableSlot { .. }
            | RelocStep::Const(SymbolSource::Slot(_) | SymbolSource::Host(_)) => true,
            RelocStep::SiteInline { plan, .. } => plan.has_calls(),
            _ => false,
        }
    }

    fn step_has_fn_ptrs(&self) -> bool {
        match self {
            RelocStep::SiteInline { plan, .. } => plan.has_fn_ptrs(),
            other => other.is_reify(),
        }
    }
}

/// Parametric relocation plan recorded on a stencil leader and evaluated in `O(|relocs|)` per follower.
#[derive(Clone, Debug, Default)]
pub struct RelocPlan {
    pub(crate) steps: Box<[RelocStep]>,
    symbol_count: u32,
    has_calls: bool,
    has_fn_ptrs: bool,
}

impl RelocPlan {
    pub(crate) fn from_vec(steps: Vec<RelocStep>) -> Self {
        let symbol_count = steps.iter().map(RelocStep::symbols).sum::<usize>() as u32;
        let has_calls = steps.iter().any(RelocStep::step_has_calls);
        let has_fn_ptrs = steps.iter().any(RelocStep::step_has_fn_ptrs);
        Self {
            steps: steps.into_boxed_slice(),
            symbol_count,
            has_calls,
            has_fn_ptrs,
        }
    }

    #[inline]
    pub(crate) fn reloc_len(&self) -> usize {
        self.symbol_count as usize
    }

    #[inline]
    pub(crate) fn has_calls(&self) -> bool {
        self.has_calls
    }

    #[inline]
    pub(crate) fn has_fn_ptrs(&self) -> bool {
        self.has_fn_ptrs
    }

    /// The step that produces symbol `target_sym`, looking inside inlined plans.
    pub(super) fn step_at_slice(steps: &[RelocStep], mut target_sym: usize) -> Option<&RelocStep> {
        for step in steps {
            let len = step.symbols();
            if target_sym < len {
                return match step {
                    RelocStep::SiteInline { plan, .. } => {
                        Self::step_at_slice(&plan.steps, target_sym)
                    }
                    other => Some(other),
                };
            }
            target_sym -= len;
        }
        None
    }

    /// Whether any of `steps`, inside inlined plans too, names a function reified to a pointer.
    pub(super) fn steps_have_fn_ptrs(steps: &[RelocStep]) -> bool {
        steps.iter().any(RelocStep::step_has_fn_ptrs)
    }

    pub(super) fn vtable_slot_at_slice(
        steps: &[RelocStep],
        target_sym: usize,
    ) -> Option<(ExprId, u16)> {
        let mut offset = 0usize;
        for step in steps {
            let len = match step {
                RelocStep::VtableSlot { expr, slot } if offset == target_sym => {
                    return Some((*expr, *slot));
                }
                other => other.symbols(),
            };
            if target_sym < offset + len {
                return None;
            }
            offset += len;
        }
        None
    }
}

/// Evaluates `plan` against reached concrete instance `index` to produce its exact relocation vector.
pub fn eval_relocs(
    cx: &mut Context<'_>,
    reached: &Reached,
    slots: &dyn Slots,
    plan: &RelocPlan,
    index: usize,
) -> Result<RelocVec> {
    let mut out = RelocVec::new();
    eval_into(cx, reached, slots, plan, index, &mut out)?;
    Ok(out)
}

fn eval_into(
    cx: &mut Context<'_>,
    reached: &Reached,
    slots: &dyn Slots,
    plan: &RelocPlan,
    index: usize,
    out: &mut RelocVec,
) -> Result<()> {
    let ((item, substs), result) = &reached.list[index];
    let typed = match result {
        Ok(typed) => typed.as_ref(),
        Err(err) => bail!("{}: {err}", cx.name(*item)),
    };
    let (substs, base) = (*substs, out.len());
    for step in &plan.steps {
        match step {
            RelocStep::Const(src) => out.push(src.clone()),
            RelocStep::Promoted { seq, size } => out.push(SymbolSource::PromotedCell {
                owner: index,
                seq: *seq,
                size: *size,
            }),
            RelocStep::VtableAt(expr) => {
                let id = eval_vtable_at(cx, reached, slots, typed, *expr)?;
                out.push(SymbolSource::Vtable(id));
            }
            RelocStep::VtableSlot { expr, slot } => {
                let id = eval_vtable_at(cx, reached, slots, typed, *expr)?;
                let entries = slots
                    .vtable_entries(id)
                    .ok_or_else(|| anyhow::anyhow!("missing vtable entries for {:?}", id))?;
                match entries.get(*slot as usize) {
                    Some(super::stencil::VtableEntry::Word(super::Word::Code(callee_idx))) => {
                        out.push(SymbolSource::Slot(*callee_idx))
                    }
                    other => bail!("vtable slot {} is not code: {:?}", slot, other),
                }
            }
            RelocStep::SiteSlot(origin) => {
                let (callee_item, callee_substs) = eval_site(cx, reached, typed, substs, origin)?;
                let callee_idx =
                    reached_index_of(cx, reached, slots, (callee_item, callee_substs))?;
                out.push(SymbolSource::Slot(callee_idx));
            }
            RelocStep::SiteInline {
                origin: SiteOrigin::VtableMethod { expr, slot },
                plan: sub_plan,
            } => {
                let id = eval_vtable_at(cx, reached, slots, typed, *expr)?;
                let entries = slots
                    .vtable_entries(id)
                    .ok_or_else(|| anyhow::anyhow!("missing vtable entries for {:?}", id))?;
                let callee_idx = match entries.get(*slot as usize) {
                    Some(super::stencil::VtableEntry::Word(super::Word::Code(idx))) => *idx,
                    other => bail!("vtable slot {} is not code: {:?}", slot, other),
                };
                eval_into(cx, reached, slots, sub_plan, callee_idx, out)?;
            }
            RelocStep::SiteInline {
                origin: SiteOrigin::Symbol(at),
                plan: sub_plan,
            } => {
                let Some(&SymbolSource::Slot(callee_idx)) = out.get(base + *at as usize) else {
                    bail!("symbol {at} names no function")
                };
                eval_into(cx, reached, slots, sub_plan, callee_idx, out)?;
            }
            RelocStep::SiteInline {
                origin,
                plan: sub_plan,
            } => {
                let (callee_item, callee_substs) = eval_site(cx, reached, typed, substs, origin)?;
                let callee_idx =
                    reached_index_of(cx, reached, slots, (callee_item, callee_substs))?;
                eval_into(cx, reached, slots, sub_plan, callee_idx, out)?;
            }
            RelocStep::TypeId(origin) => {
                let ty = eval_ty(cx, reached, typed, substs, origin)?;
                out.extend(type_id_words(cx, ty));
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

fn fmt_ty_brief(cx: &Context<'_>, ty: Ty) -> String {
    match cx.types.kind(ty) {
        Kind::Adt(def, args) => {
            let inner: Vec<String> = cx
                .types
                .substs(*args)
                .iter()
                .map(|&t| fmt_ty_brief(cx, t))
                .collect();
            if inner.is_empty() {
                cx.name(*def).to_owned()
            } else {
                format!("{}<{}>", cx.name(*def), inner.join(", "))
            }
        }
        Kind::Closure(def, ..) => format!("closure({})", cx.name(*def)),
        Kind::Slice(inner) => format!("[{}]", fmt_ty_brief(cx, *inner)),
        Kind::Array(inner, _) => format!("[{}; _]", fmt_ty_brief(cx, *inner)),
        Kind::Ref(_, inner) => format!("&{}", fmt_ty_brief(cx, *inner)),
        Kind::Ptr(_, inner) => format!("*{}", fmt_ty_brief(cx, *inner)),
        Kind::Tuple(args) => {
            let inner: Vec<String> = cx
                .types
                .substs(*args)
                .iter()
                .map(|&t| fmt_ty_brief(cx, t))
                .collect();
            format!("({})", inner.join(", "))
        }
        other => format!("{other:?}"),
    }
}

pub(crate) fn reached_index_of(
    cx: &Context<'_>,
    reached: &Reached,
    slots: &dyn Slots,
    instance: (ItemRef, Substs),
) -> Result<usize> {
    if let Some(&idx) = reached.index.get(&instance)
        && !reached.is_untyped(idx)
    {
        return Ok(idx);
    }
    slots.want(instance);
    let args: Vec<String> = cx
        .types
        .substs(instance.1)
        .iter()
        .map(|&ty| fmt_ty_brief(cx, ty))
        .collect();
    let why = if reached.index.contains_key(&instance) {
        "untyped"
    } else {
        "not reached"
    };
    if args.is_empty() {
        bail!(
            "a use of `{}` ({:?}) that was {why}",
            cx.name(instance.0),
            instance.0
        )
    }
    bail!(
        "a use of `{}<{}>` ({:?}) that was {why}",
        cx.name(instance.0),
        args.join(", "),
        instance.0
    )
}

fn eval_vtable_at(
    cx: &mut Context<'_>,
    reached: &Reached,
    slots: &dyn Slots,
    typed: &Typed,
    expr: ExprId,
) -> Result<super::stencil::VtableId> {
    let scoped = typed.scoped(&cx.types);
    let Some(vt) = scoped.coercions.get(&expr).and_then(Coercion::vtable) else {
        bail!("missing vtable coercion at {:?}", expr);
    };
    build_vtable_id(cx, reached, slots, vt)
}

fn eval_site(
    cx: &mut Context<'_>,
    reached: &Reached,
    typed: &Typed,
    substs: Substs,
    origin: &SiteOrigin,
) -> Result<(ItemRef, Substs)> {
    match origin {
        SiteOrigin::Table(key) => typed
            .site_fn(*key)
            .ok_or_else(|| anyhow::anyhow!("missing site {:?}", key)),
        SiteOrigin::PointerFnDef { expr, callee } => {
            eval_pointer_fndef(cx, reached, typed, *expr, *callee)
        }
        SiteOrigin::DropGlue(ty_origin) => {
            let ty = eval_ty(cx, reached, typed, substs, ty_origin)?;
            let Some(glue) = cx.lang_item("drop_glue") else {
                bail!("core has no drop glue")
            };
            Ok((glue, cx.types.intern_substs(&[ty])))
        }
        SiteOrigin::DropFn(ty_origin) => {
            let ty = eval_ty(cx, reached, typed, substs, ty_origin)?;
            let Some(Res::Fn(item, fn_substs)) = drop_fn(cx, ty)? else {
                bail!("type has no `Drop::drop`")
            };
            Ok((item, fn_substs))
        }
        SiteOrigin::ConstEvalSelect(inner) => {
            let (_, inner_substs) = eval_site(cx, reached, typed, substs, inner)?;
            let Some(&fn_ty) = cx.types.substs(inner_substs).get(2) else {
                bail!("const_eval_select missing runtime fn")
            };
            let Kind::FnDef(item, fn_substs) = *cx.types.kind(fn_ty) else {
                bail!("const_eval_select runtime arg is not FnDef")
            };
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
        SiteOrigin::VtableMethod { slot, .. } => {
            bail!("unexpected eval_site for vtable method slot {slot}")
        }
        SiteOrigin::Known(item, call_substs) => Ok((*item, *call_substs)),
        SiteOrigin::Symbol(at) => bail!("unexpected eval_site for symbol {at}"),
    }
}

pub(super) fn eval_deref_step(
    cx: &Context<'_>,
    reached: &Reached,
    typed: &Typed,
    expr: ExprId,
    step: u32,
    ty: Ty,
) -> Result<Ty> {
    if let Some((item, substs)) = typed.site_fn(SiteKey::Deref(expr, step)) {
        let Some(&idx) = reached.index.get(&(item, substs)) else {
            bail!("unreached deref callee")
        };
        let Ok(callee) = &reached.list[idx].1 else {
            bail!("broken deref callee")
        };
        let Some(ret) = callee.ret else {
            bail!("deref callee without return")
        };
        let Kind::Ref(_, inner) = *cx.types.kind(ret) else {
            bail!("deref callee did not return ref")
        };
        Ok(inner)
    } else if let Some(inner) = cx.boxed(ty) {
        Ok(inner)
    } else if let Kind::Ref(_, inner) | Kind::Ptr(_, inner) = *cx.types.kind(ty) {
        Ok(inner)
    } else {
        bail!("cannot autoderef {:?}", cx.types.kind(ty));
    }
}

pub(super) fn eval_autoderef_ty(
    cx: &Context<'_>,
    reached: &Reached,
    typed: &Typed,
    expr: ExprId,
    base: ExprId,
) -> Result<Ty> {
    let mut ty = typed.expr_ty(&cx.types, base);
    if ty == Ty::NONE {
        return Ok(Ty::NONE);
    }
    let steps = typed.autoderefs_at(expr);
    for step in 0..steps {
        ty = eval_deref_step(cx, reached, typed, expr, step, ty)?;
    }
    Ok(ty)
}

fn eval_pointer_fndef(
    cx: &Context<'_>,
    reached: &Reached,
    typed: &Typed,
    expr: ExprId,
    callee: ExprId,
) -> Result<(ItemRef, Substs)> {
    let ty = eval_autoderef_ty(cx, reached, typed, expr, callee)?;
    let Kind::FnDef(item, substs) = *cx.types.kind(ty) else {
        bail!("pointer call callee is not FnDef")
    };
    Ok((item, substs))
}

fn eval_ty(
    cx: &mut Context<'_>,
    reached: &Reached,
    typed: &Typed,
    substs: Substs,
    origin: &TyOrigin,
) -> Result<Ty> {
    match origin {
        TyOrigin::Subst(idx) => cx
            .types
            .substs(substs)
            .get(*idx as usize)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("subst {} out of bounds", idx)),
        TyOrigin::Local(idx) => typed
            .locals
            .get(*idx as usize)
            .map(|l| l.ty)
            .ok_or_else(|| anyhow::anyhow!("local {} out of bounds", idx)),
        TyOrigin::Expr(id) => Ok(typed.expr_ty(&cx.types, *id)),
        TyOrigin::ForIter(id) => typed
            .for_iter(&cx.types, *id)
            .ok_or_else(|| anyhow::anyhow!("missing for {}", id.0)),
        TyOrigin::AwaitFuture(id) => typed
            .await_future(&cx.types, *id)
            .ok_or_else(|| anyhow::anyhow!("missing await {}", id.0)),
        TyOrigin::SiteSubst(site, idx) => {
            let (_, site_substs) = eval_site(cx, reached, typed, substs, site)?;
            cx.types
                .substs(site_substs)
                .get(*idx as usize)
                .copied()
                .ok_or_else(|| anyhow::anyhow!("site subst {} out of bounds", idx))
        }
        TyOrigin::Step {
            base,
            variant,
            field,
        } => {
            let parent = eval_ty(cx, reached, typed, substs, base)?;
            step_ty(cx, parent, *variant, *field)
        }
    }
}

fn step_ty(cx: &mut Context<'_>, parent: Ty, variant: u32, field: u32) -> Result<Ty> {
    if variant == BOX_STEP.0 {
        return cx
            .boxed(parent)
            .ok_or_else(|| anyhow::anyhow!("box step on non-Box"));
    }
    if variant == CLOSURE_ENV_STEP {
        return cx
            .closure_env(parent)
            .ok_or_else(|| anyhow::anyhow!("closure env step on non-closure"));
    }
    match cx.types.kind(parent).clone() {
        Kind::Tuple(items) => cx
            .types
            .substs(items)
            .get(field as usize)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("tuple step {} out of bounds", field)),
        Kind::Array(elem, _) | Kind::Slice(elem) => Ok(elem),
        Kind::Adt(item, args) => {
            let slice = cx.types.substs(args).to_vec();
            let variants = field_types(cx, item, &slice)?;
            variants
                .get(variant as usize)
                .and_then(|v| v.get(field as usize))
                .copied()
                .ok_or_else(|| anyhow::anyhow!("adt step ({}, {}) out of bounds", variant, field))
        }
        Kind::Closure(..) => match cx.closure_env(parent) {
            Some(env) => step_ty(cx, env, variant, field),
            None => bail!("cannot step into a closure whose defining body was not inferred"),
        },
        other => bail!("cannot step into {:?}", other),
    }
}

/// The two words a `TypeId` of `ty` holds: the low and high halves of its key, each small enough for any target's word reloc.
pub(super) fn type_id_words(cx: &Context<'_>, ty: Ty) -> [SymbolSource; 2] {
    let key = crate::sema::summary::codec::type_id_key(cx.outline, &cx.types, ty);
    [
        SymbolSource::Word(key & u64::from(u32::MAX)),
        SymbolSource::Word(key >> 32),
    ]
}

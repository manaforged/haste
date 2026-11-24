mod body;
mod build;
mod shapes;
mod sites;
mod ty;

use anyhow::{Result, bail};

use crate::body::{ExprId, PatId};
use crate::names::ItemRef;
use crate::sema::Context;
use crate::sema::infer::{ClosureBody, Shape, Typed};
use crate::sema::reach::Reached;
use crate::sema::substs::Substs;
use crate::sema::ty::Kind;

use super::stencil::FootprintId;
use super::{Inline, Slots, passed_parts, signature};

use body::eval_body_footprint;
use shapes::{
    hash_clone_shape, hash_closure_header, hash_ctor_shape, hash_glue_shape, hash_untuple_shape,
    hash_virtual_shape,
};
pub(super) use sites::closure_calls_uncached;
use ty::{HashMix, hash_parts, hash_repr, hash_sig, hash_ty_facts};

#[derive(Clone, Debug)]
enum FieldSite {
    TupleAccess {
        expr: ExprId,
        base: ExprId,
        index: u32,
    },
    NamedAccess {
        expr: ExprId,
        base: ExprId,
    },
    StructInit {
        expr: ExprId,
        values: Box<[ExprId]>,
    },
    RangeInit {
        expr: ExprId,
    },
    CallSite {
        expr: ExprId,
        callee: ExprId,
        args: Box<[ExprId]>,
    },
    MethodSite {
        expr: ExprId,
        receiver: ExprId,
        args: Box<[ExprId]>,
    },
    IndexSite {
        expr: ExprId,
        base: ExprId,
    },
    RefSite {
        inner: ExprId,
        shared: bool,
    },
}

#[derive(Clone, Debug)]
enum PatSiteKind {
    Plain,
    Range,
    Tuple(u32),
    Slice,
    Struct(Box<[PatId]>),
}

#[derive(Clone, Debug)]
struct PatSite {
    pat: PatId,
    kind: PatSiteKind,
}

/// Reusable per-body structural hole plan cached once per `(source_item, closure_root)`.
#[derive(Debug)]
pub(super) struct BodyFootprintPlan {
    expr_ids: Box<[ExprId]>,
    pat_sites: Box<[PatSite]>,
    field_sites: Box<[FieldSite]>,
    num_sites: Box<[ExprId]>,
    repeat_sites: Box<[(ExprId, ExprId)]>,
    closure_exprs: Box<[ExprId]>,
    has_partial_moves: bool,
}

impl BodyFootprintPlan {
    pub(super) fn contains_expr(&self, expr: ExprId) -> bool {
        self.expr_ids.binary_search(&expr).is_ok()
    }

    pub(super) fn contains_pat(&self, pat: PatId) -> bool {
        self.pat_sites
            .binary_search_by_key(&pat, |ps| ps.pat)
            .is_ok()
    }
}

/// Evaluates the exact stencil `FootprintId` of reached instance `index` without running AST lowering.
pub fn eval_footprint(
    cx: &mut Context<'_>,
    reached: &Reached,
    slots: &dyn Slots,
    index: usize,
) -> Result<FootprintId> {
    let mut inline = Inline::default();
    let (fp, _) = eval_footprint_with_cycle(cx, (reached, slots), &mut inline, index)?;
    Ok(fp)
}

pub(super) fn eval_footprint_with_cycle(
    cx: &mut Context<'_>,
    (reached, slots): (&Reached, &dyn Slots),
    inline: &mut Inline,
    index: usize,
) -> Result<(FootprintId, usize)> {
    let ((item, substs), result) = &reached.list[index];
    let typed = match result {
        Ok(t) => t.as_ref(),
        Err(err) => bail!("{}: {err}", cx.name(*item)),
    };
    let (item, substs) = (*item, *substs);
    if let Some(unique) = try_unique_footprint(cx, reached, slots, item, substs, typed, index) {
        return Ok((unique, usize::MAX));
    }
    let mut mix = HashMix::new(0x51a7_0000);
    mix.item(cx.outline, item);
    let mut cycle_root = usize::MAX;
    hash_instance_header(
        cx,
        &mut mix,
        (slots.low_entries(), slots.allow_inline()),
        item,
        typed,
    )?;
    mix.bool(crate::sema::reach::unwind_enabled(cx, reached));
    mix.bool(reached.may_unwind(cx, index));
    match &typed.shape {
        Shape::Glue(glue) => hash_glue_shape(
            cx,
            (reached, slots),
            inline,
            &mut mix,
            index,
            substs,
            glue,
            &mut cycle_root,
        )?,
        Shape::Untuple(target) => hash_untuple_shape(
            cx,
            (reached, slots),
            inline,
            &mut mix,
            index,
            typed,
            *target,
            &mut cycle_root,
        )?,
        Shape::Clone(calls) => hash_clone_shape(
            cx,
            (reached, slots),
            inline,
            &mut mix,
            index,
            typed,
            calls,
            &mut cycle_root,
        )?,
        Shape::Ctor(variant) => hash_ctor_shape(cx, &mut mix, typed, *variant)?,
        Shape::Virtual(slot) => hash_virtual_shape(cx, &mut mix, typed, *slot)?,
        Shape::Body => eval_body_footprint(
            cx,
            (reached, slots),
            inline,
            &mut mix,
            index,
            (item, None),
            typed,
            &mut cycle_root,
        )?,
        Shape::Closure(cb) => {
            hash_closure_header(cx, &mut mix, typed, cb)?;
            eval_body_footprint(
                cx,
                (reached, slots),
                inline,
                &mut mix,
                index,
                (cb.parent, Some(cb.closure.body)),
                typed,
                &mut cycle_root,
            )?;
        }
        Shape::Anon(_) | Shape::Asm(_) | Shape::GlobalAlloc(_) => mix.local_index(index),
    }
    Ok((mix.finish(), cycle_root))
}

fn mix_instance_flags(
    cx: &mut Context<'_>,
    reached: &Reached,
    slots: &dyn Slots,
    mix: &mut HashMix,
    krate: u32,
    index: usize,
) {
    mix.bool(slots.low_entries());
    mix.bool(slots.allow_inline());
    mix.bool(cx.overflow_checks(krate));
    mix.bool(crate::sema::reach::unwind_enabled(cx, reached));
    mix.bool(reached.may_unwind(cx, index));
    mix.bool(cx.global_allocator().is_some());
}

fn try_unique_closure_footprint(
    cx: &mut Context<'_>,
    reached: &Reached,
    slots: &dyn Slots,
    item: ItemRef,
    substs: Substs,
    cb: &ClosureBody,
    index: usize,
) -> Option<FootprintId> {
    if cb.closure.coroutine {
        let mut mix = HashMix::new(0x51a7_1c03);
        mix.local_index(index);
        return Some(mix.finish());
    }
    let first = *cx.types.substs(substs).first()?;
    let Kind::Closure(c_def, p_substs, closure_idx) = *cx.types.kind(first) else {
        return None;
    };
    if !p_substs.is_empty() {
        return None;
    }
    let mut mix = HashMix::new(0x51a7_1c04);
    if cx.outline.is_fixed(c_def.krate) {
        mix.item(cx.outline, item);
        mix.item(cx.outline, c_def);
        mix.u32(closure_idx);
        mix_instance_flags(cx, reached, slots, &mut mix, c_def.krate, index);
    } else {
        mix.local_index(index);
    }
    Some(mix.finish())
}

fn try_unique_footprint(
    cx: &mut Context<'_>,
    reached: &Reached,
    slots: &dyn Slots,
    item: ItemRef,
    substs: Substs,
    typed: &Typed,
    index: usize,
) -> Option<FootprintId> {
    if cx.foreign(item) {
        let mut mix = HashMix::new(0xff);
        mix.item(cx.outline, item);
        return Some(mix.finish());
    }
    if matches!(typed.shape, Shape::Anon(_) | Shape::GlobalAlloc(_)) || typed.has_awaits() {
        let mut mix = HashMix::new(0x51a7_1c02);
        mix.item(cx.outline, item);
        mix.local_index(index);
        return Some(mix.finish());
    }
    if substs.is_empty() && !matches!(typed.shape, Shape::Closure(_)) {
        let mut mix = HashMix::new(0x51a7_1c01);
        mix.item(cx.outline, item);
        if matches!(typed.shape, Shape::Body) || cx.outline.is_fixed(item.krate) {
            mix_instance_flags(cx, reached, slots, &mut mix, item.krate, index);
        } else {
            mix.local_index(index);
        }
        return Some(mix.finish());
    }
    if let Shape::Closure(cb) = &typed.shape {
        return try_unique_closure_footprint(cx, reached, slots, item, substs, cb, index);
    }
    None
}

fn hash_instance_header(
    cx: &mut Context<'_>,
    mix: &mut HashMix,
    (low, allow_inline): (bool, bool),
    item: ItemRef,
    typed: &Typed,
) -> Result<()> {
    mix.bool(low);
    mix.bool(allow_inline);
    mix.bool(cx.overflow_checks(item.krate));
    let shape_tag = match &typed.shape {
        Shape::Body => 0,
        Shape::Glue(_) => 1,
        Shape::Closure(_) => 2,
        Shape::Virtual(_) => 3,
        Shape::Untuple(_) => 4,
        Shape::Ctor(_) => 5,
        Shape::Clone(_) => 6,
        Shape::Anon(_) => 7,
        Shape::Asm(_) => 8,
        Shape::GlobalAlloc(_) => 9,
    };
    mix.byte(shape_tag);
    let sig = signature(cx, typed)?;
    hash_sig(mix, &sig);
    mix.usize(typed.params.len());
    for &p in &typed.params {
        let pty = typed.locals[p as usize].ty;
        hash_repr(cx, mix, pty)?;
        if let Some(parts) = passed_parts(cx, pty)? {
            mix.byte(1);
            hash_parts(mix, parts);
        } else {
            mix.byte(0);
        }
    }
    match typed.ret {
        Some(ret) => {
            mix.byte(1);
            hash_ty_facts(cx, mix, ret)?;
        }
        None => mix.byte(0),
    }
    Ok(())
}

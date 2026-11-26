use anyhow::Result;
use rustc_hash::FxHashSet;

use crate::lower::abi::parts;
use crate::lower::budget;
use crate::lower::reloc::reached_index_of;
use crate::lower::stencil::{Form, resolve_inline_meta};
use crate::lower::{Inline, Repr, Slots, is_static, repr, signature};
use crate::names::ItemRef;
use crate::outline::item::ItemKind;
use crate::sema::Context;
use crate::sema::infer::{
    Fields, Res, Typed, by_reference, drop_fn, field_types, has_drop_impl, needs_drop, virtual_slot,
};
use crate::sema::intrinsic::Intrinsic;
use crate::sema::layout::field_offset;
use crate::sema::reach::Reached;
use crate::sema::substs::Substs;
use crate::sema::ty::{Kind, Ty};

use super::{
    HashMix, hash_num_tag, hash_parts, hash_repr, hash_sig, hash_tuple_offsets, hash_ty_facts,
    skips_drop_fields,
};

fn hash_fn_trait_edge(
    cx: &mut Context<'_>,
    (reached, slots): (&Reached, &dyn Slots),
    inline: &mut Inline,
    mix: &mut HashMix,
    (caller, item, substs): (usize, ItemRef, Substs),
    cycle_root: &mut usize,
) -> Result<()> {
    let is_once = cx.call_once() == Some(item);
    let by_ref = by_reference(cx, item);
    mix.bool(is_once);
    mix.byte(by_ref.map_or(0, |m| 1 + m as u8));
    let vslot = virtual_slot(cx, item, substs)?;
    mix.u32(vslot.unwrap_or(u32::MAX));
    if !is_once && by_ref.is_none() {
        return Ok(());
    }
    if let Some(&first) = cx.types.substs(substs).first() {
        hash_ty_facts(cx, mix, first)?;
        if let Some(boxed_inner) = cx.boxed(first)
            && cx.closure_consumes(boxed_inner)
            && let Some(Res::Fn(fi, fs)) = drop_fn(cx, first)?
        {
            hash_call_edge(
                cx,
                (reached, slots),
                inline,
                mix,
                caller,
                fi,
                fs,
                cycle_root,
            )?;
        }
    }
    if let Some(&tuple) = cx.types.substs(substs).get(1) {
        hash_tuple_offsets(cx, mix, tuple)?;
    }
    Ok(())
}

fn hash_static_edge(
    cx: &mut Context<'_>,
    reached: &Reached,
    mix: &mut HashMix,
    item: ItemRef,
    (caller, callee_idx): (usize, usize),
) -> Result<()> {
    mix.bool(caller == callee_idx);
    mix.bool(cx.thread_local(item));
    mix.bool(crate::prep::static_address(cx, item).is_some());
    if let Ok(t) = &reached.list[callee_idx].1
        && let Some(r) = t.ret
    {
        hash_ty_facts(cx, mix, r)?;
    }
    Ok(())
}

fn hash_foreign_callee(cx: &mut Context<'_>, mix: &mut HashMix, callee: &Typed) -> Result<()> {
    for &p in &callee.params {
        let pty = callee.locals[p as usize].ty;
        hash_repr(cx, mix, pty)?;
        if matches!(repr(cx, pty)?, Repr::Mem { .. }) {
            hash_parts(mix, parts(&*cx.layout(pty)?)?);
        }
    }
    if let Some(r) = callee.ret {
        hash_repr(cx, mix, r)?;
        if matches!(repr(cx, r)?, Repr::Mem { .. }) {
            hash_parts(mix, parts(&*cx.layout(r)?)?);
        }
    }
    Ok(())
}

fn hash_callee_inline(
    cx: &mut Context<'_>,
    (reached, slots): (&Reached, &dyn Slots),
    inline: &mut Inline,
    mix: &mut HashMix,
    (caller, callee_idx): (usize, usize),
    cycle_root: &mut usize,
) -> Result<()> {
    if !slots.allow_inline() {
        mix.byte(0);
        return Ok(());
    }
    let Some(limit) = budget::caller_inline_limit(cx, reached, (caller, callee_idx), false) else {
        mix.byte(0);
        return Ok(());
    };
    hash_inline_meta(
        cx,
        (reached, slots),
        inline,
        mix,
        (callee_idx, limit),
        cycle_root,
    )?;
    if budget::own_form(cx, reached, caller) == Form::Loop
        && let Some(looped) = budget::caller_inline_limit(cx, reached, (caller, callee_idx), true)
        && looped != limit
    {
        hash_inline_meta(
            cx,
            (reached, slots),
            inline,
            mix,
            (callee_idx, looped),
            cycle_root,
        )?;
    }
    Ok(())
}

pub(in super::super) fn hash_call_edge(
    cx: &mut Context<'_>,
    (reached, slots): (&Reached, &dyn Slots),
    inline: &mut Inline,
    mix: &mut HashMix,
    caller: usize,
    item: ItemRef,
    substs: Substs,
    cycle_root: &mut usize,
) -> Result<()> {
    if let Some(which) = cx.intrinsic(item) {
        return hash_intrinsic_edge(
            cx,
            (reached, slots),
            inline,
            mix,
            caller,
            item,
            which,
            substs,
            cycle_root,
        );
    }
    let callee_idx = reached_index_of(cx, reached, slots, (item, substs))?;
    hash_fn_trait_edge(
        cx,
        (reached, slots),
        inline,
        mix,
        (caller, item, substs),
        cycle_root,
    )?;
    let (foreign, stat, is_const) = (
        cx.foreign(item),
        is_static(cx, item),
        matches!(cx.item(item).1.kind, ItemKind::Const),
    );
    mix.bool(foreign);
    mix.bool(stat);
    mix.bool(is_const);
    if stat {
        return hash_static_edge(cx, reached, mix, item, (caller, callee_idx));
    }
    if is_const && substs.is_empty() {
        mix.item(cx.outline, item);
    }
    let callee = reached.typed(callee_idx, cx.name(item))?;
    let sig = signature(cx, callee)?;
    hash_sig(mix, &sig);
    mix.bool(
        callee
            .ret
            .is_some_and(|r| matches!(cx.types.kind(r), Kind::Never)),
    );
    mix.bool(callee_idx == caller);
    mix.bool(reached.may_unwind(cx, callee_idx));
    mix.bool(cx.track_caller(item));
    if foreign {
        return hash_foreign_callee(cx, mix, callee);
    }
    hash_callee_inline(
        cx,
        (reached, slots),
        inline,
        mix,
        (caller, callee_idx),
        cycle_root,
    )
}

fn hash_inline_meta(
    cx: &mut Context<'_>,
    (reached, slots): (&Reached, &dyn Slots),
    inline: &mut Inline,
    mix: &mut HashMix,
    (callee_idx, limit): (usize, usize),
    cycle_root: &mut usize,
) -> Result<()> {
    let (meta, cyc) = resolve_inline_meta(cx, (reached, slots), inline, callee_idx, limit)?;
    *cycle_root = (*cycle_root).min(cyc);
    match meta {
        Some((inl_item, inl_fp, returns)) => {
            mix.byte(1);
            mix.item(cx.outline, inl_item);
            mix.footprint(inl_fp);
            mix.bool(returns);
        }
        None => mix.byte(0),
    }
    Ok(())
}

fn hash_intrinsic_edge(
    cx: &mut Context<'_>,
    (reached, slots): (&Reached, &dyn Slots),
    inline: &mut Inline,
    mix: &mut HashMix,
    caller: usize,
    item: ItemRef,
    which: Intrinsic,
    substs: Substs,
    cycle_root: &mut usize,
) -> Result<()> {
    mix.byte(0xe0);
    mix.item(cx.outline, item);
    let args = cx.types.substs(substs).to_vec();
    mix.usize(args.len());
    for &arg in &args {
        hash_ty_facts(cx, mix, arg)?;
        hash_num_tag(cx, mix, arg);
    }
    if matches!(which, Intrinsic::Simd(_)) {
        for &arg in &args {
            if let Kind::Adt(adt_item, adt_args) = *cx.types.kind(arg) {
                let s = cx.types.substs(adt_args).to_vec();
                if let Ok(fields) = field_types(cx, adt_item, &s)
                    && let Some(v0) = fields.first()
                {
                    mix.usize(v0.len());
                    for &fty in v0 {
                        hash_ty_facts(cx, mix, fty)?;
                        hash_num_tag(cx, mix, fty);
                    }
                }
            }
        }
    }
    if which == Intrinsic::ConstEvalSelect
        && let Some(&fn_ty) = args.get(2)
        && let Kind::FnDef(fi, fs) = *cx.types.kind(fn_ty)
    {
        if let Some(&tup) = args.first() {
            hash_tuple_offsets(cx, mix, tup)?;
        }
        hash_call_edge(
            cx,
            (reached, slots),
            inline,
            mix,
            caller,
            fi,
            fs,
            cycle_root,
        )?;
    }
    Ok(())
}

pub(in super::super) fn hash_drop_tree(
    cx: &mut Context<'_>,
    (reached, slots): (&Reached, &dyn Slots),
    inline: &mut Inline,
    mix: &mut HashMix,
    caller: usize,
    ty: Ty,
    depth: u8,
    cycle_root: &mut usize,
) -> Result<()> {
    let mut seen = FxHashSet::default();
    hash_drop_tree_seen(
        cx,
        (reached, slots),
        inline,
        (mix, caller, cycle_root),
        (ty, depth, &mut seen),
    )
}

fn hash_drop_tree_seen(
    cx: &mut Context<'_>,
    (reached, slots): (&Reached, &dyn Slots),
    inline: &mut Inline,
    (mix, caller, cycle_root): (&mut HashMix, usize, &mut usize),
    (ty, depth, seen): (Ty, u8, &mut FxHashSet<Ty>),
) -> Result<()> {
    if depth > 32 || ty == Ty::NONE || !seen.insert(ty) || !matches!(needs_drop(cx, ty), Ok(true)) {
        return Ok(());
    }
    if let Some(glue) = cx.lang_item("drop_glue") {
        let gs = cx.types.intern_substs(&[ty]);
        hash_call_edge(
            cx,
            (reached, slots),
            inline,
            mix,
            caller,
            glue,
            gs,
            cycle_root,
        )?;
    }
    if let Some(inner) = cx.boxed(ty) {
        if let Ok(Some(Res::Fn(fi, fs))) = drop_fn(cx, ty) {
            hash_call_edge(
                cx,
                (reached, slots),
                inline,
                mix,
                caller,
                fi,
                fs,
                cycle_root,
            )?;
        }
        return hash_drop_tree_seen(
            cx,
            (reached, slots),
            inline,
            (mix, caller, cycle_root),
            (inner, depth + 1, seen),
        );
    }
    if !matches!(has_drop_impl(cx, ty), Ok(false)) {
        return Ok(());
    }
    hash_drop_variants(
        cx,
        (reached, slots),
        inline,
        (mix, caller, cycle_root),
        (ty, depth, seen),
    )
}

fn hash_drop_variants(
    cx: &mut Context<'_>,
    (reached, slots): (&Reached, &dyn Slots),
    inline: &mut Inline,
    (mix, caller, cycle_root): (&mut HashMix, usize, &mut usize),
    (ty, depth, seen): (Ty, u8, &mut FxHashSet<Ty>),
) -> Result<()> {
    let variants: Fields = match cx.types.kind(ty).clone() {
        Kind::Tuple(items) => vec![cx.types.substs(items).to_vec()].into(),
        Kind::Array(elem, _) | Kind::Slice(elem) => {
            let Ok(layout) = cx.layout(elem) else {
                mix.byte(0xfc);
                return Ok(());
            };
            mix.u64(layout.size.bytes());
            return hash_drop_tree_seen(
                cx,
                (reached, slots),
                inline,
                (mix, caller, cycle_root),
                (elem, depth + 1, seen),
            );
        }
        Kind::Adt(adt_item, _) if skips_drop_fields(cx, adt_item) => return Ok(()),
        Kind::Adt(adt_item, adt_args) => {
            let slice = cx.types.substs(adt_args).to_vec();
            let Ok(fields) = field_types(cx, adt_item, &slice) else {
                mix.byte(0xfd);
                return Ok(());
            };
            fields
        }
        _ => return Ok(()),
    };
    hash_drop_field_list(
        cx,
        (reached, slots),
        inline,
        (mix, caller, cycle_root),
        (ty, &variants, depth, seen),
    )
}

fn hash_drop_field_list(
    cx: &mut Context<'_>,
    (reached, slots): (&Reached, &dyn Slots),
    inline: &mut Inline,
    (mix, caller, cycle_root): (&mut HashMix, usize, &mut usize),
    (ty, variants, depth, seen): (Ty, &Fields, u8, &mut FxHashSet<Ty>),
) -> Result<()> {
    let Ok(layout) = cx.layout(ty) else {
        mix.byte(0xfc);
        return Ok(());
    };
    for (v_idx, fields) in variants.iter().enumerate() {
        for (f_idx, &fty) in fields.iter().enumerate() {
            mix.i32(field_offset(&layout, v_idx as u32, f_idx as u32));
            if hash_repr(cx, mix, fty).is_err() {
                mix.byte(0xfc);
                continue;
            }
            hash_drop_tree_seen(
                cx,
                (reached, slots),
                inline,
                (mix, caller, cycle_root),
                (fty, depth + 1, seen),
            )?;
        }
    }
    Ok(())
}

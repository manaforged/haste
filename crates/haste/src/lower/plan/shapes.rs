use anyhow::{Result, bail};

use crate::outline::item::ItemKind;
use crate::sema::Context;
use crate::sema::infer::{
    ClosureBody, Fields, Glue, Res, Typed, cloned_tuple, field_types, needs_drop,
};
use crate::sema::layout::field_offset;
use crate::sema::reach::Reached;
use crate::sema::substs::Substs;
use crate::sema::ty::{Kind, Mut, Ty};

use super::super::{Inline, Slots, signature_of};
use super::ty::{
    HashMix, hash_call_edge, hash_dyn_facts, hash_repr, hash_sig, hash_tuple_offsets, hash_ty_facts,
};

pub(super) fn hash_glue_shape(
    cx: &mut Context<'_>,
    (reached, slots): (&Reached, &dyn Slots),
    inline: &mut Inline,
    mix: &mut HashMix,
    index: usize,
    substs: Substs,
    glue: &Glue,
    cycle_root: &mut usize,
) -> Result<()> {
    if glue.resume.is_some() {
        mix.local_index(index);
        return Ok(());
    }
    let Some(&target_ty) = cx.types.substs(substs).first() else {
        bail!("glue missing target type")
    };
    hash_ty_facts(cx, mix, target_ty)?;
    if let Some(Res::Fn(di, ds)) = glue.drop {
        mix.byte(1);
        hash_call_edge(cx, (reached, slots), inline, mix, index, di, ds, cycle_root)?;
    } else {
        mix.byte(0);
    }
    if let Some(inner) = cx.boxed(target_ty) {
        mix.byte(0xb1);
        let rm = cx.types.intern(Kind::Ref(Mut::Mut, inner));
        hash_repr(cx, mix, rm)?;
        hash_ty_facts(cx, mix, inner)?;
        if needs_drop(cx, inner)?
            && let Some(g) = cx.lang_item("drop_glue")
        {
            hash_call_edge(
                cx,
                (reached, slots),
                inline,
                mix,
                index,
                g,
                cx.types.intern_substs(&[inner]),
                cycle_root,
            )?;
        }
        return Ok(());
    }
    let mut cur = target_ty;
    while let Kind::Closure(..) = *cx.types.kind(cur) {
        mix.byte(0xc1);
        let Some(env) = cx.closure_env(cur) else {
            bail!("closure without env")
        };
        cur = env;
    }
    match cx.types.kind(cur).clone() {
        Kind::Array(elem, len_ty) => {
            mix.byte(1);
            mix.u64(cx.types.len(len_ty).unwrap_or(u64::MAX));
            hash_elem_glue(cx, (reached, slots), inline, mix, index, elem, cycle_root)?;
        }
        Kind::Slice(elem) => {
            mix.byte(2);
            hash_elem_glue(cx, (reached, slots), inline, mix, index, elem, cycle_root)?;
        }
        Kind::Dyn(..) => mix.byte(3),
        Kind::Tuple(items) => {
            mix.byte(4);
            let variants: Fields = vec![cx.types.substs(items).to_vec()].into();
            hash_glue_variants(
                cx,
                (reached, slots),
                inline,
                mix,
                index,
                cur,
                &variants,
                cycle_root,
            )?;
        }
        Kind::Adt(adt_item, adt_args) => {
            if cx.lang(adt_item) == Some("manually_drop")
                || cx.item(adt_item).1.kind == ItemKind::Union
            {
                mix.byte(5);
            } else {
                mix.byte(6);
                let slice = cx.types.substs(adt_args).to_vec();
                let variants = field_types(cx, adt_item, &slice)?;
                hash_glue_variants(
                    cx,
                    (reached, slots),
                    inline,
                    mix,
                    index,
                    cur,
                    &variants,
                    cycle_root,
                )?;
            }
        }
        _ => mix.byte(7),
    }
    Ok(())
}

fn hash_elem_glue(
    cx: &mut Context<'_>,
    (reached, slots): (&Reached, &dyn Slots),
    inline: &mut Inline,
    mix: &mut HashMix,
    index: usize,
    elem: Ty,
    cycle_root: &mut usize,
) -> Result<()> {
    let nd = needs_drop(cx, elem)?;
    mix.bool(nd);
    if nd {
        mix.u64(cx.layout(elem)?.size.bytes());
        hash_repr(cx, mix, elem)?;
        if let Some(g) = cx.lang_item("drop_glue") {
            hash_call_edge(
                cx,
                (reached, slots),
                inline,
                mix,
                index,
                g,
                cx.types.intern_substs(&[elem]),
                cycle_root,
            )?;
        }
    }
    Ok(())
}

fn hash_glue_variants(
    cx: &mut Context<'_>,
    (reached, slots): (&Reached, &dyn Slots),
    inline: &mut Inline,
    mix: &mut HashMix,
    index: usize,
    cur: Ty,
    variants: &[Vec<Ty>],
    cycle_root: &mut usize,
) -> Result<()> {
    let unsized_cur = cx.is_unsized(cur)?;
    mix.bool(unsized_cur);
    hash_ty_facts(cx, mix, cur)?;
    let layout = (!unsized_cur).then(|| cx.layout(cur)).transpose()?;
    mix.usize(variants.len());
    for (v_idx, fields) in variants.iter().enumerate() {
        mix.usize(fields.len());
        for (f_idx, &fty) in fields.iter().enumerate() {
            if let Some(ref l) = layout {
                mix.i32(field_offset(l, v_idx as u32, f_idx as u32));
            }
            let f_unsized = cx.is_unsized(fty)?;
            mix.bool(f_unsized);
            if !f_unsized {
                hash_repr(cx, mix, fty)?;
            }
            let nd = needs_drop(cx, fty)?;
            mix.bool(nd);
            if nd && let Some(g) = cx.lang_item("drop_glue") {
                hash_call_edge(
                    cx,
                    (reached, slots),
                    inline,
                    mix,
                    index,
                    g,
                    cx.types.intern_substs(&[fty]),
                    cycle_root,
                )?;
            }
        }
    }
    Ok(())
}

pub(super) fn hash_untuple_shape(
    cx: &mut Context<'_>,
    (reached, slots): (&Reached, &dyn Slots),
    inline: &mut Inline,
    mix: &mut HashMix,
    index: usize,
    typed: &Typed,
    target: Res,
    cycle_root: &mut usize,
) -> Result<()> {
    let (recv_ty, tuple_ty) = (typed.locals[0].ty, typed.locals[1].ty);
    hash_ty_facts(cx, mix, recv_ty)?;
    hash_ty_facts(cx, mix, tuple_ty)?;
    hash_tuple_offsets(cx, mix, tuple_ty)?;
    match target {
        Res::Fn(ti, ts) => {
            mix.byte(1);
            hash_call_edge(cx, (reached, slots), inline, mix, index, ti, ts, cycle_root)?;
        }
        Res::Local(l) => {
            mix.byte(2);
            mix.u32(l);
            let mut inner = recv_ty;
            while let Kind::Ref(_, sub) | Kind::Ptr(_, sub) = *cx.types.kind(inner) {
                inner = sub;
            }
            if let Some(boxed) = cx.boxed(inner) {
                inner = boxed;
            }
            if let Kind::FnPtr(params, ret) = *cx.types.kind(inner) {
                let params = cx.types.substs(params).to_vec();
                let sig = signature_of(cx, Some(ret), params)?;
                hash_sig(mix, &sig);
            }
        }
        _ => mix.byte(3),
    }
    Ok(())
}

pub(super) fn hash_clone_shape(
    cx: &mut Context<'_>,
    (reached, slots): (&Reached, &dyn Slots),
    inline: &mut Inline,
    mix: &mut HashMix,
    index: usize,
    typed: &Typed,
    calls: &[Res],
    cycle_root: &mut usize,
) -> Result<()> {
    let ret = typed.ret_ty();
    mix.bool(cloned_tuple(cx, ret).is_some());
    hash_ty_facts(cx, mix, ret)?;
    hash_tuple_offsets(cx, mix, ret)?;
    mix.usize(calls.len());
    for call in calls {
        if let Res::Fn(ci, cs) = *call {
            hash_call_edge(cx, (reached, slots), inline, mix, index, ci, cs, cycle_root)?;
        }
    }
    Ok(())
}

pub(super) fn hash_ctor_shape(
    cx: &mut Context<'_>,
    mix: &mut HashMix,
    typed: &Typed,
    variant: u32,
) -> Result<()> {
    let ret = typed.ret_ty();
    mix.u32(variant);
    hash_ty_facts(cx, mix, ret)?;
    let layout = cx.layout(ret)?;
    for (idx, &param) in typed.params.iter().enumerate() {
        mix.i32(field_offset(&layout, variant, idx as u32));
        hash_repr(cx, mix, typed.locals[param as usize].ty)?;
    }
    Ok(())
}

pub(super) fn hash_virtual_shape(
    cx: &mut Context<'_>,
    mix: &mut HashMix,
    typed: &Typed,
    slot: u32,
) -> Result<()> {
    mix.u32(slot);
    let recv_ty = typed.locals[0].ty;
    hash_ty_facts(cx, mix, recv_ty)?;
    let mut obj = recv_ty;
    if let Some(inner) = cx.boxed(obj) {
        obj = inner;
    } else if let Kind::Ref(_, inner) | Kind::Ptr(_, inner) = *cx.types.kind(obj) {
        obj = inner;
    }
    if let Kind::Dyn(of, ..) = *cx.types.kind(obj) {
        hash_dyn_facts(cx, mix, obj, of)?;
    }
    Ok(())
}

pub(super) fn hash_closure_header(
    cx: &mut Context<'_>,
    mix: &mut HashMix,
    typed: &Typed,
    cb: &ClosureBody,
) -> Result<()> {
    mix.item(cx.outline, cb.parent);
    mix.u32(cb.closure.body.0);
    mix.bool(cb.closure.consumes);
    mix.bool(cb.env.is_some());
    if let Some(env_local) = cb.env {
        let env_ref = typed.locals[env_local as usize].ty;
        hash_ty_facts(cx, mix, env_ref)?;
        if let Kind::Ref(_, pointee) = *cx.types.kind(env_ref) {
            let env_ty = match *cx.types.kind(pointee) {
                Kind::Ref(_, inner) => inner,
                _ => pointee,
            };
            hash_tuple_offsets(cx, mix, env_ty)?;
        }
    }
    mix.usize(cb.closure.captures.len());
    for cap in &cb.closure.captures {
        mix.byte(cap.by as u8);
        mix.usize(cap.path.len());
        for &(variant, index) in &cap.path {
            mix.u32(variant);
            mix.u32(index);
        }
        let cap_ty = cap.ty(cx, typed.locals[cap.local as usize].ty)?;
        hash_ty_facts(cx, mix, cap_ty)?;
    }
    Ok(())
}

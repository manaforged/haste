use anyhow::Result;
use rustc_abi::{TagEncoding, Variants};
use rustc_hash::FxHashSet;
use rustc_index::Idx;

use crate::lower::abi::copied_parts;
use crate::lower::adt::tag_bits;
use crate::lower::stencil::FootprintId;
use crate::lower::{passed_parts, signature_of};
use crate::names::{ItemRef, Primitive};
use crate::outline::item::ItemKind;
use crate::sema::Context;
use crate::sema::infer::{field_types, has_drop_impl, needs_drop};
use crate::sema::layout::Layout;
use crate::sema::ty::{Kind, Mut, Ty};

use super::{
    HashMix, hash_parts, hash_repr, hash_sig, hash_tuple_offsets, hash_type, skips_drop_fields,
};

pub(in super::super) fn hash_ty_facts(
    cx: &mut Context<'_>,
    mix: &mut HashMix,
    ty: Ty,
) -> Result<()> {
    if ty == Ty::NONE {
        mix.byte(0);
        return Ok(());
    }
    let cacheable = !cx.types.has_var_flag(ty) && !cx.types.has_params(ty);
    if cacheable && let Some(&bits) = cx.ty_facts.get(&ty) {
        mix.footprint(FootprintId(bits));
        return Ok(());
    }
    let before = cx.quiet_mark();
    let mut sub = HashMix::new(0x51a7_7f01);
    hash_ty_facts_depth(cx, &mut sub, ty, 0)?;
    if !cx.types.has_params(ty) && matches!(needs_drop(cx, ty), Ok(true)) {
        hash_nested_drop_facts(cx, &mut sub, ty, 0, &mut FxHashSet::default());
    }
    let fp = sub.finish();
    if cacheable && cx.quiet_mark() == before {
        cx.ty_facts.insert(ty, fp.0);
    }
    mix.footprint(fp);
    Ok(())
}

fn hash_nested_drop_facts(
    cx: &mut Context<'_>,
    mix: &mut HashMix,
    ty: Ty,
    depth: u8,
    seen: &mut FxHashSet<Ty>,
) {
    if depth >= 32 || ty == Ty::NONE || cx.types.has_params(ty) || !seen.insert(ty) {
        return;
    }
    let Ok(nd) = needs_drop(cx, ty) else {
        mix.byte(0xff);
        return;
    };
    mix.bool(nd);
    if !nd {
        return;
    }
    let Ok(own_drop) = has_drop_impl(cx, ty) else {
        mix.byte(0xfe);
        return;
    };
    mix.bool(own_drop);
    if let Some(inner) = cx.boxed(ty) {
        mix.byte(0xb1);
        hash_nested_drop_facts(cx, mix, inner, depth + 1, seen);
        return;
    }
    hash_nested_drop_parts(cx, mix, ty, depth, seen);
}

fn hash_nested_drop_parts(
    cx: &mut Context<'_>,
    mix: &mut HashMix,
    ty: Ty,
    depth: u8,
    seen: &mut FxHashSet<Ty>,
) {
    match cx.types.kind(ty).clone() {
        Kind::Tuple(items) => {
            for item in cx.types.substs(items).to_vec() {
                hash_nested_drop_facts(cx, mix, item, depth + 1, seen);
            }
        }
        Kind::Array(elem, _) | Kind::Slice(elem) => {
            hash_nested_drop_facts(cx, mix, elem, depth + 1, seen);
        }
        Kind::Closure(..) => {
            if let Some(env) = cx.closure_env(ty) {
                hash_nested_drop_facts(cx, mix, env, depth + 1, seen);
            }
        }
        Kind::Adt(item, args) => {
            let slice = cx.types.substs(args).to_vec();
            hash_nested_adt_drop_facts(cx, mix, (item, &slice), depth, seen);
        }
        _ => {}
    }
}

fn hash_nested_adt_drop_facts(
    cx: &mut Context<'_>,
    mix: &mut HashMix,
    (item, args): (ItemRef, &[Ty]),
    depth: u8,
    seen: &mut FxHashSet<Ty>,
) {
    if skips_drop_fields(cx, item) {
        return;
    }
    let Ok(variants) = field_types(cx, item, args) else {
        mix.byte(0xfd);
        return;
    };
    for fields in variants.iter() {
        for &fty in fields {
            hash_nested_drop_facts(cx, mix, fty, depth + 1, seen);
        }
    }
}

fn hash_ty_facts_depth(cx: &mut Context<'_>, mix: &mut HashMix, ty: Ty, depth: u8) -> Result<()> {
    if ty == Ty::NONE {
        mix.byte(0);
        return Ok(());
    }
    if let Kind::Value(bits) = *cx.types.kind(ty) {
        mix.byte(0x12);
        mix.u128(bits);
        return Ok(());
    }
    if cx.is_unsized(ty)? {
        mix.byte(0xf0);
        return hash_unsized_facts(cx, mix, ty, depth);
    }
    hash_repr(cx, mix, ty)?;
    let layout = cx.layout(ty)?;
    hash_parts(mix, copied_parts(&layout));
    if let Some(p) = passed_parts(cx, ty)? {
        mix.byte(1);
        hash_parts(mix, p);
    } else {
        mix.byte(0);
    }
    mix.bool(needs_drop(cx, ty)?);
    mix.bool(has_drop_impl(cx, ty)?);
    if let Some(inner) = cx.boxed(ty) {
        mix.byte(0xb0);
        let rm = cx.types.intern(Kind::Ref(Mut::Mut, inner));
        hash_repr(cx, mix, rm)?;
        if depth < 4 {
            hash_ty_facts_depth(cx, mix, inner, depth + 1)?;
        }
    }
    hash_ty_kind_facts(cx, mix, ty, &layout, depth)
}

fn hash_ptr_kind_facts(
    cx: &mut Context<'_>,
    mix: &mut HashMix,
    is_ref: bool,
    mutability: Mut,
    inner: Ty,
    depth: u8,
) -> Result<()> {
    mix.byte(if is_ref { 0x13 } else { 0x14 });
    mix.byte(mutability as u8);
    let unsized_inner = cx.is_unsized(inner)?;
    mix.bool(unsized_inner);
    if unsized_inner {
        hash_unsized_facts(cx, mix, inner, depth + 1)?;
    } else if !is_ref && depth < 3 {
        mix.u64(cx.layout(inner)?.size.bytes());
    }
    Ok(())
}

fn hash_ty_kind_facts(
    cx: &mut Context<'_>,
    mix: &mut HashMix,
    ty: Ty,
    layout: &Layout,
    depth: u8,
) -> Result<()> {
    match cx.types.kind(ty).clone() {
        Kind::Prim(p) => {
            mix.byte(0x10);
            mix.byte(p as u8);
        }
        Kind::Never => mix.byte(0x11),
        Kind::Value(bits) => {
            mix.byte(0x12);
            mix.u128(bits);
        }
        Kind::Ref(m, inner) => hash_ptr_kind_facts(cx, mix, true, m, inner, depth)?,
        Kind::Ptr(m, inner) => hash_ptr_kind_facts(cx, mix, false, m, inner, depth)?,
        Kind::Array(elem, len_ty) => {
            mix.byte(0x15);
            mix.u64(cx.types.len(len_ty).unwrap_or(u64::MAX));
            mix.u64(cx.layout(elem)?.size.bytes());
            hash_repr(cx, mix, elem)?;
        }
        Kind::Slice(elem) => {
            mix.byte(0x16);
            mix.u64(cx.layout(elem)?.size.bytes());
            hash_repr(cx, mix, elem)?;
        }
        Kind::Tuple(items) => {
            mix.byte(0x17);
            mix.usize(cx.types.substs(items).len());
        }
        Kind::FnDef(fi, _) => {
            mix.byte(0x18);
            mix.bool(cx.foreign(fi));
        }
        Kind::FnPtr(params, ret) | Kind::UnsafeFnPtr(params, ret) => {
            mix.byte(0x19);
            let params = cx.types.substs(params).to_vec();
            let sig = signature_of(cx, Some(ret), params)?;
            hash_sig(mix, &sig);
            mix.bool(matches!(cx.types.kind(ret), Kind::Never));
        }
        Kind::Closure(parent, _, at) => {
            mix.byte(0x1a);
            mix.item(cx.outline, parent);
            mix.u32(at);
            mix.bool(cx.closure_takes_env(ty));
            mix.bool(cx.closure_consumes(ty));
            if depth < 3
                && let Some(env_ty) = cx.closure_env(ty)
            {
                hash_ty_facts_depth(cx, mix, env_ty, depth + 1)?;
                hash_tuple_offsets(cx, mix, env_ty)?;
            }
        }
        Kind::Dyn(of, ..) => {
            mix.byte(0x1b);
            hash_dyn_facts(cx, mix, ty, of)?;
        }
        Kind::Pat(inner, range) => {
            let (start, end) = cx.types.pat(range);
            mix.byte(0x1c);
            mix.u128(start);
            mix.u128(end);
            if depth < 4 {
                hash_ty_facts_depth(cx, mix, inner, depth + 1)?;
            }
        }
        Kind::Adt(adt_item, _) => {
            mix.byte(0x1d);
            let kind = cx.item(adt_item).1.kind;
            mix.byte(kind as u8);
            if kind == ItemKind::Enum {
                hash_enum_encoding(cx, mix, adt_item, layout)?;
            }
        }
        _ => mix.byte(0x1f),
    }
    Ok(())
}

fn hash_unsized_facts(cx: &mut Context<'_>, mix: &mut HashMix, ty: Ty, depth: u8) -> Result<()> {
    mix.bool(needs_drop(cx, ty)?);
    match cx.types.kind(ty).clone() {
        Kind::Slice(elem) => {
            mix.byte(1);
            mix.u64(cx.layout(elem)?.size.bytes());
            hash_repr(cx, mix, elem)?;
        }
        Kind::Prim(Primitive::Str) => mix.byte(2),
        Kind::Dyn(of, ..) => {
            mix.byte(3);
            hash_dyn_facts(cx, mix, ty, of)?;
        }
        Kind::Adt(adt_item, adt_args) if depth < 4 => {
            mix.byte(4);
            let slice = cx.types.substs(adt_args).to_vec();
            let fields = field_types(cx, adt_item, &slice)?;
            if let Some(v0) = fields.first() {
                mix.usize(v0.len());
                for &fty in v0 {
                    if cx.is_unsized(fty)? {
                        hash_unsized_facts(cx, mix, fty, depth + 1)?;
                    } else {
                        mix.u64(cx.layout(fty)?.size.bytes());
                        mix.u64(cx.layout(fty)?.align.bytes());
                    }
                }
            }
        }
        _ => mix.byte(5),
    }
    Ok(())
}

pub(in super::super) fn hash_dyn_facts(
    cx: &mut Context<'_>,
    mix: &mut HashMix,
    ty: Ty,
    of: ItemRef,
) -> Result<()> {
    mix.item(cx.outline, of);
    match crate::prep::vtable(cx, ty) {
        Some(laid) => {
            mix.byte(1);
            mix.usize(laid.slots.len());
            for s in &laid.slots {
                mix.item(cx.outline, s.method);
                mix.usize(s.at);
                mix.usize(s.carries.len());
            }
            for &(sup, at) in &laid.supers {
                mix.item(cx.outline, sup);
                mix.usize(at.unwrap_or(usize::MAX));
            }
        }
        None => mix.byte(0),
    }
    Ok(())
}

fn hash_enum_encoding(
    cx: &mut Context<'_>,
    mix: &mut HashMix,
    item: ItemRef,
    layout: &Layout,
) -> Result<()> {
    let discs = cx.discriminants(item)?;
    mix.usize(discs.len());
    for &d in discs.iter() {
        mix.u128(d as u128);
    }
    match &layout.variants {
        Variants::Empty => mix.byte(0),
        Variants::Single { index } => {
            mix.byte(1);
            mix.usize(index.index());
        }
        Variants::Multiple {
            tag,
            tag_encoding,
            tag_field,
            variants,
        } => {
            mix.byte(2);
            mix.usize(tag_field.index());
            mix.i32(layout.fields.offset(tag_field.index()).bytes() as i32);
            mix.u64(tag.size(crate::sema::layout::target()?).bytes());
            match tag_encoding {
                TagEncoding::Direct => mix.byte(0),
                TagEncoding::Niche {
                    untagged_variant,
                    niche_variants,
                    niche_start,
                } => {
                    mix.byte(1);
                    mix.usize(untagged_variant.index());
                    mix.usize(niche_variants.start.index());
                    mix.usize(niche_variants.last.index());
                    mix.u128(*niche_start);
                }
            }
            for v in 0..variants.len() as u32 {
                if let Some((offset, ty, bits)) = tag_bits(cx, item, layout, v)? {
                    mix.byte(1);
                    mix.i32(offset);
                    hash_type(mix, ty);
                    mix.u128(bits);
                } else {
                    mix.byte(0);
                }
            }
        }
    }
    Ok(())
}

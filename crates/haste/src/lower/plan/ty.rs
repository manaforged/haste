use anyhow::{Result, bail};
use haste_codegen::{ArgumentPurpose, Signature, Type};
use rustc_abi::{TagEncoding, Variants};
use rustc_index::Idx;

use crate::names::{ItemRef, Primitive};
use crate::outline::item::ItemKind;
use crate::sema::Context;
use crate::sema::infer::{Fields, Res, by_reference, drop_fn, field_types, has_drop_impl, needs_drop, virtual_slot};
use crate::sema::intrinsic::Intrinsic;
use crate::sema::layout::field_offset;
use crate::sema::reach::Reached;
use crate::sema::substs::Substs;
use crate::sema::ty::{Kind, Mut, Ty, float_bits, int_info};

use super::super::abi::{copied_parts, parts};
use super::super::adt::tag_bits;
use super::super::expr::inline_limit;
use super::super::reloc::reached_index_of;
use super::super::stencil::{FootprintId, item_bits, resolve_inline_meta};
use super::super::{Inline, Parts, Repr, Slots, is_static, repr, returned, signature, signature_of, unwrapped};

#[derive(Clone, Copy)]
pub(super) struct HashMix {
    a: u64,
    b: u64,
}

impl HashMix {
    pub(super) fn new(seed: u128) -> Self {
        Self { a: 0x9e37_79b9_7f4a_7c15 ^ (seed as u64), b: 0xbf58_476d_1ce4_e5b9 ^ ((seed >> 64) as u64) }
    }

    pub(super) fn u64(&mut self, v: u64) {
        self.a = (self.a ^ v).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        self.a ^= self.a >> 30;
        self.b = (self.b.rotate_left(23) ^ v).wrapping_mul(0x94d0_49bb_1331_11eb);
    }

    pub(super) fn byte(&mut self, v: u8) {
        self.u64(u64::from(v) | 0x0100_0000_0000_0000);
    }

    pub(super) fn bool(&mut self, v: bool) {
        self.byte(u8::from(v));
    }

    pub(super) fn u32(&mut self, v: u32) {
        self.u64(u64::from(v) | 0x0200_0000_0000_0000);
    }

    pub(super) fn i32(&mut self, v: i32) {
        self.u32(v as u32);
    }

    pub(super) fn usize(&mut self, v: usize) {
        self.u64(v as u64);
    }

    pub(super) fn u128(&mut self, v: u128) {
        self.u64(v as u64);
        self.u64((v >> 64) as u64);
    }

    pub(super) fn item(&mut self, item: ItemRef) {
        self.u128(item_bits(item));
    }

    pub(super) fn finish(self) -> FootprintId {
        let mut lo = self.a ^ self.b.rotate_left(29);
        lo = (lo ^ (lo >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        lo = (lo ^ (lo >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        lo ^= lo >> 31;
        let mut hi = self.b ^ self.a.rotate_left(17);
        hi = (hi ^ (hi >> 30)).wrapping_mul(0x94d0_49bb_1331_11eb);
        hi = (hi ^ (hi >> 27)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        hi ^= hi >> 31;
        FootprintId(((hi as u128) << 64) | (lo as u128))
    }
}

pub(super) fn type_code(ty: Type) -> u8 {
    match ty {
        Type::I8 => 1,
        Type::I16 => 2,
        Type::I32 => 3,
        Type::I64 => 4,
        Type::I128 => 5,
        Type::F32 => 6,
        Type::F64 => 7,
        Type::V128 => 8,
    }
}

pub(super) fn hash_parts(mix: &mut HashMix, p: Parts) {
    match p {
        Parts::One(t) => {
            mix.byte(1);
            mix.byte(type_code(t));
        }
        Parts::Pair(a, b, off) => {
            mix.byte(2);
            mix.byte(type_code(a));
            mix.byte(type_code(b));
            mix.i32(off);
        }
        Parts::Bytes => mix.byte(3),
    }
}

pub(super) fn hash_repr(cx: &mut Context<'_>, mix: &mut HashMix, ty: Ty) -> Result<()> {
    if ty == Ty::NONE || matches!(cx.types.kind(ty), Kind::Value(_)) {
        mix.byte(0);
        return Ok(());
    }
    match repr(cx, ty)? {
        Repr::Zst => mix.byte(1),
        Repr::Scalar(t) => {
            mix.byte(2);
            mix.byte(type_code(t));
        }
        Repr::Mem { size, align, parts: p } => {
            mix.byte(3);
            mix.u32(size);
            mix.u32(align as u32);
            hash_parts(mix, p);
        }
    }
    Ok(())
}

fn hash_purpose(mix: &mut HashMix, purpose: ArgumentPurpose) {
    match purpose {
        ArgumentPurpose::Normal => mix.byte(0),
        ArgumentPurpose::StructReturn => mix.byte(1),
        ArgumentPurpose::StructArgument(n) => {
            mix.byte(2);
            mix.u32(n);
        }
        ArgumentPurpose::Dict => mix.byte(3),
    }
}

pub(super) fn hash_sig(mix: &mut HashMix, sig: &Signature) {
    mix.byte(sig.call_conv as u8);
    mix.usize(sig.params.len());
    for p in &sig.params {
        mix.byte(type_code(p.value_type));
        hash_purpose(mix, p.purpose);
    }
    mix.usize(sig.returns.len());
    for r in &sig.returns {
        mix.byte(type_code(r.value_type));
        hash_purpose(mix, r.purpose);
    }
}

pub(super) fn hash_num_tag(cx: &Context<'_>, mix: &mut HashMix, ty: Ty) {
    if ty == Ty::NONE {
        mix.byte(0);
        return;
    }
    let inner = match *cx.types.kind(ty) {
        Kind::Pat(base, ..) => base,
        _ => ty,
    };
    match *cx.types.kind(inner) {
        Kind::Prim(Primitive::Bool) => mix.byte(1),
        Kind::Prim(Primitive::Char) => mix.byte(2),
        Kind::Prim(p) => {
            if let Some((bits, signed)) = int_info(p) {
                mix.byte(3);
                mix.u32(bits);
                mix.bool(signed);
            } else if let Some(bits) = float_bits(p) {
                mix.byte(4);
                mix.u32(bits);
            } else {
                mix.byte(5);
            }
        }
        Kind::Ptr(..) | Kind::FnPtr(..) => mix.byte(6),
        _ => mix.byte(7),
    }
}

pub(super) fn hash_ty_facts(cx: &mut Context<'_>, mix: &mut HashMix, ty: Ty) -> Result<()> {
    hash_ty_facts_depth(cx, mix, ty, 0)
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
    if let Some(r) = returned(cx, ty)? {
        mix.byte(1);
        hash_parts(mix, r);
    } else {
        mix.byte(0);
    }
    mix.byte(unwrapped(cx, ty)?.map_or(0, type_code));
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
        Kind::Ref(m, inner) | Kind::Ptr(m, inner) => {
            let is_ref = matches!(cx.types.kind(ty), Kind::Ref(..));
            mix.byte(if is_ref { 0x13 } else { 0x14 });
            mix.byte(m as u8);
            let unsized_inner = cx.is_unsized(inner)?;
            mix.bool(unsized_inner);
            if unsized_inner {
                hash_unsized_facts(cx, mix, inner, depth + 1)?;
            } else if !is_ref && depth < 3 {
                mix.u64(cx.layout(inner)?.size.bytes());
            }
        }
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
        Kind::FnPtr(params, ret) => {
            mix.byte(0x19);
            let params = cx.types.substs(params).to_vec();
            let sig = signature_of(cx, Some(ret), params)?;
            hash_sig(mix, &sig);
            mix.bool(matches!(cx.types.kind(ret), Kind::Never));
        }
        Kind::Closure(parent, _, at) => {
            mix.byte(0x1a);
            mix.item(parent);
            mix.u32(at);
            mix.bool(cx.closure_takes_env(ty));
            mix.bool(cx.closure_consumes(ty));
        }
        Kind::Dyn(of, ..) => {
            mix.byte(0x1b);
            hash_dyn_facts(cx, mix, ty, of)?;
        }
        Kind::Pat(inner, start, end) => {
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
                hash_enum_encoding(cx, mix, adt_item, &layout)?;
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

pub(super) fn hash_dyn_facts(cx: &mut Context<'_>, mix: &mut HashMix, ty: Ty, of: ItemRef) -> Result<()> {
    mix.item(of);
    match crate::prep::vtable(cx, ty) {
        Some(laid) => {
            mix.byte(1);
            mix.usize(laid.slots.len());
            for s in &laid.slots {
                mix.item(s.method);
                mix.usize(s.at);
                mix.usize(s.carries.len());
            }
            for &(sup, at) in &laid.supers {
                mix.item(sup);
                mix.usize(at.unwrap_or(usize::MAX));
            }
        }
        None => mix.byte(0),
    }
    Ok(())
}

fn hash_enum_encoding(cx: &mut Context<'_>, mix: &mut HashMix, item: ItemRef, layout: &crate::sema::layout::Layout) -> Result<()> {
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
        Variants::Multiple { tag, tag_encoding, tag_field, variants } => {
            mix.byte(2);
            mix.usize(tag_field.index());
            mix.i32(layout.fields.offset(tag_field.index()).bytes() as i32);
            mix.u64(tag.size(crate::sema::layout::target()?).bytes());
            match tag_encoding {
                TagEncoding::Direct => mix.byte(0),
                TagEncoding::Niche { untagged_variant, niche_variants, niche_start } => {
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
                    mix.byte(type_code(ty));
                    mix.u128(bits);
                } else {
                    mix.byte(0);
                }
            }
        }
    }
    Ok(())
}

pub(super) fn hash_call_edge(cx: &mut Context<'_>, (reached, slots): (&Reached, &dyn Slots), inline: &mut Inline, mix: &mut HashMix, caller: usize, item: ItemRef, substs: Substs, cycle_root: &mut usize) -> Result<()> {
    if let Some(which) = cx.intrinsic(item) {
        return hash_intrinsic_edge(cx, (reached, slots), inline, mix, caller, item, which, substs, cycle_root);
    }
    let callee_idx = reached_index_of(cx, reached, slots, (item, substs))?;
    let is_once = cx.call_once() == Some(item);
    let by_ref = by_reference(cx, item);
    mix.bool(is_once);
    mix.byte(by_ref.map_or(0, |m| 1 + m as u8));
    let vslot = virtual_slot(cx, item, substs)?;
    mix.u32(vslot.unwrap_or(u32::MAX));
    if is_once || by_ref.is_some() {
        if let Some(&first) = cx.types.substs(substs).first() {
            hash_ty_facts(cx, mix, first)?;
            if let Some(boxed_inner) = cx.boxed(first)
                && cx.closure_consumes(boxed_inner)
                && let Some(Res::Fn(fi, fs)) = drop_fn(cx, first)?
            {
                hash_call_edge(cx, (reached, slots), inline, mix, caller, fi, fs, cycle_root)?;
            }
        }
        if let Some(&tuple) = cx.types.substs(substs).get(1) {
            hash_tuple_offsets(cx, mix, tuple)?;
        }
    }
    let foreign = cx.foreign(item);
    let stat = is_static(cx, item);
    let is_const = matches!(cx.item(item).1.kind, ItemKind::Const);
    mix.bool(foreign);
    mix.bool(stat);
    mix.bool(is_const);
    if stat {
        mix.bool(caller == callee_idx);
        mix.bool(cx.thread_local(item));
        mix.bool(crate::prep::static_address(cx, item).is_some());
        if let Ok(t) = &reached.list[callee_idx].1
            && let Some(r) = t.ret
        {
            hash_ty_facts(cx, mix, r)?;
        }
        return Ok(());
    }
    if is_const && substs.is_empty() {
        mix.item(item);
    }
    let Ok(callee) = &reached.list[callee_idx].1 else { bail!("{} did not type-check", cx.name(item)) };
    let sig = signature(cx, callee)?;
    hash_sig(mix, &sig);
    mix.bool(callee.ret.is_some_and(|r| matches!(cx.types.kind(r), Kind::Never)));
    if foreign {
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
        return Ok(());
    }
    if let Some(limit) = inline_limit(cx, reached, callee_idx) {
        let (meta, cyc) = resolve_inline_meta(cx, (reached, slots), inline, callee_idx, limit)?;
        *cycle_root = (*cycle_root).min(cyc);
        match meta {
            Some((inl_item, inl_fp, returns)) => {
                mix.byte(1);
                mix.item(inl_item);
                mix.u128(inl_fp.0);
                mix.bool(returns);
            }
            None => mix.byte(0),
        }
    } else {
        mix.byte(0);
    }
    Ok(())
}

fn hash_intrinsic_edge(cx: &mut Context<'_>, (reached, slots): (&Reached, &dyn Slots), inline: &mut Inline, mix: &mut HashMix, caller: usize, item: ItemRef, which: Intrinsic, substs: Substs, cycle_root: &mut usize) -> Result<()> {
    mix.byte(0xe0);
    mix.item(item);
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
        hash_call_edge(cx, (reached, slots), inline, mix, caller, fi, fs, cycle_root)?;
    }
    Ok(())
}

pub(super) fn hash_tuple_offsets(cx: &mut Context<'_>, mix: &mut HashMix, tuple: Ty) -> Result<()> {
    if let Kind::Tuple(items) = *cx.types.kind(tuple) {
        let slice = cx.types.substs(items).to_vec();
        let layout = cx.layout(tuple)?;
        mix.usize(slice.len());
        for (idx, &fty) in slice.iter().enumerate() {
            mix.i32(field_offset(&layout, 0, idx as u32));
            hash_repr(cx, mix, fty)?;
        }
    }
    Ok(())
}

pub(super) fn hash_drop_tree(cx: &mut Context<'_>, (reached, slots): (&Reached, &dyn Slots), inline: &mut Inline, mix: &mut HashMix, caller: usize, ty: Ty, depth: u8, cycle_root: &mut usize) -> Result<()> {
    if depth > 5 || ty == Ty::NONE || !needs_drop(cx, ty)? {
        return Ok(());
    }
    if let Some(glue) = cx.lang_item("drop_glue") {
        let gs = cx.types.intern_substs(&[ty]);
        hash_call_edge(cx, (reached, slots), inline, mix, caller, glue, gs, cycle_root)?;
    }
    if let Some(inner) = cx.boxed(ty) {
        if let Some(Res::Fn(fi, fs)) = drop_fn(cx, ty)? {
            hash_call_edge(cx, (reached, slots), inline, mix, caller, fi, fs, cycle_root)?;
        }
        return hash_drop_tree(cx, (reached, slots), inline, mix, caller, inner, depth + 1, cycle_root);
    }
    if has_drop_impl(cx, ty)? {
        return Ok(());
    }
    let variants: Fields = match cx.types.kind(ty).clone() {
        Kind::Tuple(items) => vec![cx.types.substs(items).to_vec()].into(),
        Kind::Array(elem, _) => {
            let layout = cx.layout(elem)?;
            mix.u64(layout.size.bytes());
            return hash_drop_tree(cx, (reached, slots), inline, mix, caller, elem, depth + 1, cycle_root);
        }
        Kind::Adt(adt_item, adt_args) => {
            let slice = cx.types.substs(adt_args).to_vec();
            field_types(cx, adt_item, &slice)?
        }
        _ => return Ok(()),
    };
    let layout = cx.layout(ty)?;
    for (v_idx, fields) in variants.iter().enumerate() {
        for (f_idx, &fty) in fields.iter().enumerate() {
            mix.i32(field_offset(&layout, v_idx as u32, f_idx as u32));
            hash_repr(cx, mix, fty)?;
            hash_drop_tree(cx, (reached, slots), inline, mix, caller, fty, depth + 1, cycle_root)?;
        }
    }
    Ok(())
}

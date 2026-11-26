mod edges;
mod facts;

use anyhow::Result;
use haste_codegen::{ArgumentPurpose, Signature, Type};

use crate::names::{ItemRef, Primitive};
use crate::outline::item::ItemKind;
use crate::sema::Context;
use crate::sema::layout::field_offset;
use crate::sema::ty::{Kind, Ty, float_bits, int_info};

use super::super::stencil::{FootprintId, item_bits};
use super::super::{Parts, Repr, repr};

pub(super) use edges::{hash_call_edge, hash_drop_tree};
pub(super) use facts::{hash_dyn_facts, hash_ty_facts};

#[derive(Clone, Copy)]
pub(super) struct HashMix {
    a: u64,
    b: u64,
    local: bool,
}

impl HashMix {
    pub(super) fn new(seed: u128) -> Self {
        Self {
            a: 0x9e37_79b9_7f4a_7c15 ^ (seed as u64),
            b: 0xbf58_476d_1ce4_e5b9 ^ ((seed >> 64) as u64),
            local: false,
        }
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

    pub(super) fn local_index(&mut self, v: usize) {
        self.local = true;
        self.usize(v);
    }

    pub(super) fn u128(&mut self, v: u128) {
        self.u64(v as u64);
        self.u64((v >> 64) as u64);
    }

    pub(super) fn footprint(&mut self, fp: FootprintId) {
        if !fp.is_portable() {
            self.local = true;
        }
        self.u128(fp.0);
    }

    pub(super) fn item(&mut self, outline: &crate::outline::Outline, item: ItemRef) {
        if let Some(key) = crate::substrate::sysroot_item_key(outline, item) {
            self.u128((1u128 << 96) | u128::from(key));
        } else {
            self.local = true;
            self.u128(item_bits(item));
        }
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
        let raw = ((hi as u128) << 64) | (lo as u128);
        FootprintId(if self.local { (raw & !1) | 1 } else { raw & !1 })
    }
}

pub(super) fn hash_type(mix: &mut HashMix, ty: Type) {
    match ty {
        Type::I8 => mix.byte(1),
        Type::I16 => mix.byte(2),
        Type::I32 => mix.byte(3),
        Type::I64 => mix.byte(4),
        Type::I128 => mix.byte(5),
        Type::F32 => mix.byte(6),
        Type::F64 => mix.byte(7),
        Type::Vec { lane, count } => {
            mix.byte(8);
            mix.byte(lane as u8);
            mix.byte(count);
        }
    }
}

pub(super) fn hash_parts(mix: &mut HashMix, p: Parts) {
    match p {
        Parts::One(t) => {
            mix.byte(1);
            hash_type(mix, t);
        }
        Parts::Pair(a, b, off) => {
            mix.byte(2);
            hash_type(mix, a);
            hash_type(mix, b);
            mix.i32(off);
        }
        Parts::Words(words, len) => {
            mix.byte(4);
            mix.byte(len);
            for &(ty, off) in &words[..usize::from(len)] {
                hash_type(mix, ty);
                mix.i32(off);
            }
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
            hash_type(mix, t);
        }
        Repr::Mem {
            size,
            align,
            parts: p,
        } => {
            mix.byte(3);
            mix.u32(size);
            mix.u32(align as u32);
            hash_parts(mix, p);
        }
    }
    Ok(())
}

fn hash_purpose(mix: &mut HashMix, purpose: ArgumentPurpose) {
    let (tag, extra) = crate::lower::purpose_tag(purpose);
    mix.byte(tag);
    if let Some(n) = extra {
        mix.u32(n);
    }
}

pub(super) fn hash_sig(mix: &mut HashMix, sig: &Signature) {
    mix.byte(sig.call_conv as u8);
    mix.usize(sig.params.len());
    for p in &sig.params {
        hash_type(mix, p.value_type);
        hash_purpose(mix, p.purpose);
        mix.bool(p.readonly);
        mix.bool(p.noalias);
        mix.u32(p.deref_bytes);
    }
    mix.usize(sig.returns.len());
    for r in &sig.returns {
        hash_type(mix, r.value_type);
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
        Kind::Ptr(..) | Kind::FnPtr(..) | Kind::UnsafeFnPtr(..) => mix.byte(6),
        _ => mix.byte(7),
    }
}

pub(super) fn hash_tuple_offsets(cx: &mut Context<'_>, mix: &mut HashMix, tuple: Ty) -> Result<()> {
    if tuple == Ty::NONE {
        mix.byte(0);
        return Ok(());
    }
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

fn skips_drop_fields(cx: &Context<'_>, item: ItemRef) -> bool {
    cx.lang(item) == Some("manually_drop") || cx.item(item).1.kind == ItemKind::Union
}

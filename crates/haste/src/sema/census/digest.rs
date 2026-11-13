use std::sync::atomic::{AtomicI64, Ordering};

use haste_codegen::{ArgumentPurpose, Signature};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::lower::{self, Parts, Repr, Slots, Word};
use crate::names::{ItemRef, Primitive};
use crate::sema::Context;
use crate::sema::impls::ImplRef;
use crate::sema::infer::header_bound_traits;
use crate::sema::reach::Reached;
use crate::sema::substs::Substs;
use crate::sema::ty::{Kind, Ty, int_info};

#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct Digest(pub u64, pub u64);

#[derive(Clone, Copy)]
pub(super) struct Mix(u64, u64);

impl Mix {
    pub(super) fn new(seed: u64) -> Self {
        Self(0xcbf2_9ce4_8422_2325 ^ seed, 0x9e37_79b9_7f4a_7c15 ^ seed.rotate_left(17))
    }

    pub(super) fn byte(&mut self, value: u8) {
        self.0 = (self.0 ^ u64::from(value)).wrapping_mul(0x0000_0100_0000_01b3);
        self.1 = (self.1.wrapping_add(u64::from(value) + 1)).wrapping_mul(0xbf58_476d_1ce4_e5b9).rotate_left(13);
    }

    pub(super) fn flag(&mut self, value: bool) {
        self.byte(u8::from(value));
    }

    fn raw(&mut self, data: &[u8]) {
        for &byte in data {
            self.byte(byte);
        }
    }

    pub(super) fn u32(&mut self, value: u32) {
        self.raw(&value.to_le_bytes());
    }

    pub(super) fn i32(&mut self, value: i32) {
        self.raw(&value.to_le_bytes());
    }

    pub(super) fn u64(&mut self, value: u64) {
        self.raw(&value.to_le_bytes());
    }

    pub(super) fn u128(&mut self, value: u128) {
        self.raw(&value.to_le_bytes());
    }

    pub(super) fn usize(&mut self, value: usize) {
        self.u64(value as u64);
    }

    pub(super) fn bytes(&mut self, data: &[u8]) {
        self.u64(data.len() as u64);
        self.raw(data);
    }

    pub(super) fn item(&mut self, item: ItemRef) {
        self.u32(item.krate);
        self.u32(item.file);
        self.u32(item.item);
    }

    pub(super) fn digest(&mut self, digest: Digest) {
        self.u64(digest.0);
        self.u64(digest.1);
    }

    pub(super) fn finish(self) -> Digest {
        Digest(avalanche(self.0), avalanche(self.1))
    }
}

fn avalanche(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

const PROBE_DATA: i64 = 0x3000_0000;

#[derive(Default)]
struct ProbeSlots {
    next: AtomicI64,
    bodies: lower::Bodies,
}

impl ProbeSlots {
    fn alloc(&self, len: usize) -> i64 {
        let step = ((len.max(1) as i64) + 15) & !15;
        let prev = self.next.try_update(Ordering::Relaxed, Ordering::Relaxed, |cur| Some(cur.max(PROBE_DATA) + step)).unwrap_or_else(|cur| cur);
        prev.max(PROBE_DATA)
    }
}

impl Slots for ProbeSlots {
    fn entry(&self, index: usize) -> i64 {
        0x1000_0000 + (index as i64) * 8
    }

    fn data(&self, bytes: &[u8]) -> i64 {
        self.alloc(bytes.len())
    }

    fn zeroed(&self, len: usize) -> i64 {
        self.alloc(len)
    }

    fn table(&self, words: &[Word]) -> i64 {
        self.alloc(words.len() * 8)
    }

    fn cell(&self, index: usize, _size: u32) -> i64 {
        0x2000_0000 + (index as i64) * 16
    }

    fn bodies(&self) -> &lower::Bodies {
        &self.bodies
    }
}

pub(super) fn inline_flags(cx: &mut Context<'_>, reached: &Reached) -> Vec<bool> {
    let slots = ProbeSlots::default();
    let mut flags = vec![false; reached.list.len()];
    for (index, flag) in flags.iter_mut().enumerate() {
        let ((item, _), typed) = &reached.list[index];
        if typed.is_ok() && !cx.foreign(*item) {
            *flag = lower::can_inline_body(cx, reached, &slots, index);
        }
    }
    flags
}

pub(super) struct TraitFacts {
    spec_trait_set: FxHashSet<ItemRef>,
    item_trait_map: FxHashMap<ItemRef, Option<ItemRef>>,
    owner_bounds_map: FxHashMap<ItemRef, FxHashSet<ItemRef>>,
}

impl TraitFacts {
    pub(super) fn build(cx: &mut Context<'_>) -> Self {
        let mut spec_trait_set = FxHashSet::default();
        let krate_count = cx.outline.crates.len();
        for krate in 0..krate_count {
            let impl_len = cx.outline.crates[krate].impls.len();
            for index in 0..impl_len {
                let block = ImplRef::Module { krate: krate as u32, index: index as u32 };
                if cx.specializable(block)
                    && let Some(of) = cx.impl_trait(block)
                {
                    spec_trait_set.insert(of);
                }
            }
        }
        Self { spec_trait_set, item_trait_map: FxHashMap::default(), owner_bounds_map: FxHashMap::default() }
    }

    pub(super) fn trait_of(&mut self, cx: &mut Context<'_>, item: ItemRef) -> Option<ItemRef> {
        if let Some(&cached) = self.item_trait_map.get(&item) {
            return cached;
        }
        let found = cx.owner_trait(item).or_else(|| cx.owner_impl(item).and_then(|block| cx.impl_trait(block)));
        self.item_trait_map.insert(item, found);
        found
    }

    pub(super) fn is_unbounded_spec_callee(&mut self, cx: &mut Context<'_>, owner: ItemRef, item: ItemRef) -> bool {
        let Some(of) = self.trait_of(cx, item) else { return false };
        if !self.spec_trait_set.contains(&of) {
            return false;
        }
        if !self.owner_bounds_map.contains_key(&owner) {
            let set: FxHashSet<ItemRef> = header_bound_traits(cx, owner).into_iter().collect();
            self.owner_bounds_map.insert(owner, set);
        }
        !self.owner_bounds_map[&owner].contains(&of)
    }
}

pub(super) fn outer_shape_digest(cx: &mut Context<'_>, substs: Substs) -> Digest {
    let slice = cx.types.substs(substs).to_vec();
    let mut mix = Mix::new(99);
    mix.usize(slice.len());
    for &ty in &slice {
        if let Kind::Value(bits) = *cx.types.kind(ty) {
            mix.byte(1);
            mix.u128(bits);
            continue;
        }
        if cx.types.kind(ty).is_unsized() || cx.is_unsized(ty).unwrap_or(false) {
            mix.byte(2);
            hash_unsized_kind(cx, ty, &mut mix);
            continue;
        }
        mix.byte(3);
        if let Ok(r) = lower::repr(cx, ty) {
            hash_repr(&r, &mut mix);
        }
        if let Ok(layout) = cx.layout(ty) {
            mix.u64(layout.size.bytes());
            mix.u64(layout.align.abi.bytes());
        }
    }
    mix.finish()
}

pub(super) fn hash_unsized_kind(cx: &mut Context<'_>, ty: Ty, mix: &mut Mix) {
    let tail = cx.tail(ty).unwrap_or(ty);
    match cx.types.kind(tail).clone() {
        Kind::Slice(elem) => {
            mix.byte(1);
            let size = cx.layout(elem).map_or(0, |l| l.size.bytes());
            let align = cx.layout(elem).map_or(1, |l| l.align.abi.bytes());
            mix.u64(size);
            mix.u64(align);
        }
        Kind::Prim(Primitive::Str) => mix.byte(2),
        Kind::Dyn(..) => mix.byte(3),
        _ => mix.byte(4),
    }
}

pub(super) fn hash_repr(repr: &Repr, mix: &mut Mix) {
    match *repr {
        Repr::Zst => mix.byte(0),
        Repr::Scalar(ty) => {
            mix.byte(1);
            mix.byte(ty as u8);
        }
        Repr::Mem { size, align, parts } => {
            mix.byte(2);
            mix.u32(size);
            mix.byte(align);
            match parts {
                Parts::Bytes => mix.byte(0),
                Parts::One(ty) => {
                    mix.byte(1);
                    mix.byte(ty as u8);
                }
                Parts::Pair(a, b, off) => {
                    mix.byte(2);
                    mix.byte(a as u8);
                    mix.byte(b as u8);
                    mix.i32(off);
                }
            }
        }
    }
}

pub(super) fn hash_signature(sig: &Signature, mix: &mut Mix) {
    mix.usize(sig.params.len());
    for param in &sig.params {
        mix.byte(param.value_type as u8);
        mix.byte(u8::from(param.purpose == ArgumentPurpose::StructReturn));
    }
    mix.usize(sig.returns.len());
    for ret in &sig.returns {
        mix.byte(ret.value_type as u8);
        mix.byte(u8::from(ret.purpose == ArgumentPurpose::StructReturn));
    }
}

pub(super) fn hash_num_tag(cx: &mut Context<'_>, ty: Ty, mix: &mut Mix) {
    if ty == Ty::NONE {
        mix.byte(0);
        return;
    }
    let machine = lower::machine(cx, ty).ok().flatten();
    match (cx.types.kind(ty), machine) {
        (Kind::Prim(Primitive::Bool), _) => mix.byte(1),
        (Kind::Prim(Primitive::F32 | Primitive::F64), Some(m)) => {
            mix.byte(2);
            mix.byte(m as u8);
        }
        (Kind::Prim(prim), Some(m)) if int_info(*prim).is_some() => {
            mix.byte(3);
            mix.byte(m as u8);
            mix.flag(int_info(*prim).is_some_and(|(_, signed)| signed));
        }
        (_, Some(m)) => {
            mix.byte(4);
            mix.byte(m as u8);
        }
        _ => mix.byte(5),
    }
}

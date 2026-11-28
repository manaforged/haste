use std::num::NonZero;

use anyhow::{Context as _, Result, bail};
use rustc_abi::{
    AddressSpace, Align, BackendLaneCount, BackendRepr, FieldsShape, Float, Integer, IntegerType,
    Niche, NumScalableVectors, Primitive as AbiPrim, ReprFlags, ReprOptions, ScalableElt, Scalar,
    Size, Variants, WrappingRange,
};
use rustc_hash::FxHashMap as HashMap;
use rustc_index::Idx;

use crate::names::ItemRef;
use crate::sema::impls::{Head, ImplRef, Index as ImplIndex, TraitBucket};
use crate::sema::layout::{Field, Layout, Variant};
use crate::sema::ty::Mut;

use super::codec::{
    Cursor, decode_prim, encode_prim, read_opt, read_vec, write_opt, write_u16, write_u32,
    write_u64, write_u128,
};
use super::warm_expr::{read_opt_u32, write_opt_u32};

const INTS: [Integer; 5] = [
    Integer::I8,
    Integer::I16,
    Integer::I32,
    Integer::I64,
    Integer::I128,
];

const FLOATS: [Float; 6] = [
    Float::F16,
    Float::F16B,
    Float::F32,
    Float::F64,
    Float::F128,
    Float::PpcF128,
];

fn write_abi_prim(out: &mut Vec<u8>, p: AbiPrim) {
    match p {
        AbiPrim::Int(int, signed) => {
            out.push(0);
            out.push(int as u8);
            out.push(u8::from(signed));
        }
        AbiPrim::Float(fl) => {
            out.push(1);
            out.push(fl as u8);
        }
        AbiPrim::Pointer(addr) => {
            out.push(2);
            write_u32(out, addr.0);
        }
    }
}

fn read_abi_prim(cur: &mut Cursor<'_>) -> Result<AbiPrim> {
    Ok(match cur.u8()? {
        0 => AbiPrim::Int(
            INTS.get(cur.u8()? as usize)
                .copied()
                .context("invalid abi int")?,
            cur.u8()? != 0,
        ),
        1 => AbiPrim::Float(
            FLOATS
                .get(cur.u8()? as usize)
                .copied()
                .context("invalid abi float")?,
        ),
        2 => AbiPrim::Pointer(AddressSpace(cur.u32()?)),
        _ => bail!("invalid abi prim tag"),
    })
}

fn write_scalar(out: &mut Vec<u8>, s: Scalar) {
    match s {
        Scalar::Initialized { value, valid_range } => {
            out.push(0);
            write_abi_prim(out, value);
            write_u128(out, valid_range.start);
            write_u128(out, valid_range.end);
        }
        Scalar::Union { value } => {
            out.push(1);
            write_abi_prim(out, value);
        }
    }
}

fn read_scalar(cur: &mut Cursor<'_>) -> Result<Scalar> {
    Ok(match cur.u8()? {
        0 => Scalar::Initialized {
            value: read_abi_prim(cur)?,
            valid_range: WrappingRange {
                start: cur.u128()?,
                end: cur.u128()?,
            },
        },
        1 => Scalar::Union {
            value: read_abi_prim(cur)?,
        },
        _ => bail!("invalid scalar tag"),
    })
}

pub(super) fn write_repr_options(out: &mut Vec<u8>, repr: &ReprOptions) {
    let _ = write_opt(out, repr.int.as_ref(), |o, it| {
        match *it {
            IntegerType::Pointer(s) => {
                o.push(0);
                o.push(u8::from(s));
            }
            IntegerType::Fixed(int, s) => {
                o.push(1);
                o.push(int as u8);
                o.push(u8::from(s));
            }
        }
        Some(())
    });
    let _ = write_opt(out, repr.align.as_ref(), |o, a| {
        write_u64(o, a.bytes());
        Some(())
    });
    let _ = write_opt(out, repr.pack.as_ref(), |o, a| {
        write_u64(o, a.bytes());
        Some(())
    });
    out.push(repr.flags.bits());
    let _ = write_opt(out, repr.scalable.as_ref(), |o, s| {
        match *s {
            ScalableElt::ElementCount(n) => {
                o.push(0);
                write_u16(o, n);
            }
            ScalableElt::Container => o.push(1),
        }
        Some(())
    });
}

pub(super) fn read_repr_options(cur: &mut Cursor<'_>) -> Result<ReprOptions> {
    let int = read_opt(cur, |c| match c.u8()? {
        0 => Ok(IntegerType::Pointer(c.u8()? != 0)),
        1 => Ok(IntegerType::Fixed(
            INTS.get(c.u8()? as usize)
                .copied()
                .context("invalid repr int")?,
            c.u8()? != 0,
        )),
        _ => bail!("invalid repr int tag"),
    })?;
    let align = read_opt(cur, |c| {
        Align::from_bytes(c.u64()?).map_err(|_| anyhow::anyhow!("invalid align"))
    })?;
    let pack = read_opt(cur, |c| {
        Align::from_bytes(c.u64()?).map_err(|_| anyhow::anyhow!("invalid pack align"))
    })?;
    let flags = ReprFlags::from_bits(cur.u8()?).context("invalid repr flags")?;
    let scalable = read_opt(cur, |c| match c.u8()? {
        0 => Ok(ScalableElt::ElementCount(c.u16()?)),
        1 => Ok(ScalableElt::Container),
        _ => bail!("invalid scalable tag"),
    })?;
    Ok(ReprOptions {
        int,
        align,
        pack,
        flags,
        scalable,
        field_shuffle_seed: Default::default(),
    })
}

pub(super) fn write_layout(out: &mut Vec<u8>, layout: &Layout) -> Option<()> {
    let variant = match layout.variants {
        Variants::Empty => None,
        Variants::Single { index } => Some(index.index() as u32),
        Variants::Multiple { .. } => return None,
    };
    match &layout.fields {
        FieldsShape::Primitive => out.push(0),
        FieldsShape::Union(n) => {
            out.push(1);
            write_u64(out, n.get() as u64);
        }
        FieldsShape::Array { stride, count } => {
            out.push(2);
            write_u64(out, stride.bytes());
            write_u64(out, *count);
        }
        FieldsShape::Arbitrary {
            offsets,
            in_memory_order,
        } => {
            out.push(3);
            write_u32(out, offsets.len() as u32);
            offsets.iter().for_each(|off| write_u64(out, off.bytes()));
            write_u32(out, in_memory_order.len() as u32);
            in_memory_order
                .iter()
                .for_each(|f| write_u32(out, f.index() as u32));
        }
    }
    write_opt_u32(out, variant);
    match layout.backend_repr {
        BackendRepr::Scalar(s) => {
            out.push(0);
            write_scalar(out, s);
        }
        BackendRepr::ScalarPair { a, b, b_offset } => {
            out.push(1);
            write_scalar(out, a);
            write_scalar(out, b);
            write_u64(out, b_offset.bytes());
        }
        BackendRepr::SimdScalableVector {
            element,
            count,
            number_of_vectors,
        } => {
            out.push(2);
            write_scalar(out, element);
            write_u32(out, count.as_u32());
            out.push(number_of_vectors.0);
        }
        BackendRepr::SimdVector { element, count } => {
            out.push(3);
            write_scalar(out, element);
            write_u32(out, count.as_u32());
        }
        BackendRepr::Memory { sized } => {
            out.push(4);
            out.push(u8::from(sized));
        }
    }
    let _ = write_opt(out, layout.largest_niche.as_ref(), |o, n| {
        write_u64(o, n.offset.bytes());
        write_abi_prim(o, n.value);
        write_u128(o, n.valid_range.start);
        write_u128(o, n.valid_range.end);
        Some(())
    });
    out.push(u8::from(layout.uninhabited) | (u8::from(layout.repr_c) << 1));
    write_u64(out, layout.align.bytes());
    write_u64(out, layout.size.bytes());
    let _ = write_opt(out, layout.max_repr_align.as_ref(), |o, a| {
        write_u64(o, a.bytes());
        Some(())
    });
    write_u64(out, layout.unadjusted_abi_align.bytes());
    Some(())
}

pub(super) fn read_layout(cur: &mut Cursor<'_>) -> Result<Layout> {
    let fields = match cur.u8()? {
        0 => FieldsShape::Primitive,
        1 => FieldsShape::Union(
            NonZero::new(usize::try_from(cur.u64()?).context("union size overflow")?)
                .context("zero union fields")?,
        ),
        2 => FieldsShape::Array {
            stride: Size::from_bytes(cur.u64()?),
            count: cur.u64()?,
        },
        3 => FieldsShape::Arbitrary {
            offsets: read_vec(cur, |c| Ok(Size::from_bytes(c.u64()?)))?
                .into_iter()
                .collect(),
            in_memory_order: read_vec(cur, |c| Ok(Field::new(c.u32()? as usize)))?
                .into_iter()
                .collect(),
        },
        _ => bail!("invalid fields shape"),
    };
    let variants = match read_opt_u32(cur)? {
        None => Variants::Empty,
        Some(idx) => Variants::Single {
            index: Variant::new(idx as usize),
        },
    };
    let backend_repr = match cur.u8()? {
        0 => BackendRepr::Scalar(read_scalar(cur)?),
        1 => BackendRepr::ScalarPair {
            a: read_scalar(cur)?,
            b: read_scalar(cur)?,
            b_offset: Size::from_bytes(cur.u64()?),
        },
        2 => BackendRepr::SimdScalableVector {
            element: read_scalar(cur)?,
            count: BackendLaneCount::new::<()>(u64::from(cur.u32()?))
                .map_err(|_| anyhow::anyhow!("invalid lane count"))?,
            number_of_vectors: NumScalableVectors(cur.u8()?),
        },
        3 => BackendRepr::SimdVector {
            element: read_scalar(cur)?,
            count: BackendLaneCount::new::<()>(u64::from(cur.u32()?))
                .map_err(|_| anyhow::anyhow!("invalid lane count"))?,
        },
        4 => BackendRepr::Memory {
            sized: cur.u8()? != 0,
        },
        _ => bail!("invalid backend repr"),
    };
    let largest_niche = read_opt(cur, |c| {
        Ok(Niche {
            offset: Size::from_bytes(c.u64()?),
            value: read_abi_prim(c)?,
            valid_range: WrappingRange {
                start: c.u128()?,
                end: c.u128()?,
            },
        })
    })?;
    let flags = cur.u8()?;
    Ok(Layout {
        fields,
        variants,
        backend_repr,
        largest_niche,
        uninhabited: flags & 1 != 0,
        align: Align::from_bytes(cur.u64()?)
            .map_err(|_| anyhow::anyhow!("invalid layout align"))?,
        size: Size::from_bytes(cur.u64()?),
        max_repr_align: read_opt(cur, |c| {
            Align::from_bytes(c.u64()?).map_err(|_| anyhow::anyhow!("invalid max align"))
        })?,
        unadjusted_abi_align: Align::from_bytes(cur.u64()?)
            .map_err(|_| anyhow::anyhow!("invalid abi align"))?,
        repr_c: flags & 2 != 0,
        randomization_seed: Default::default(),
    })
}

fn write_head(
    out: &mut Vec<u8>,
    head: Head,
    item_id: &mut impl FnMut(ItemRef) -> Option<u32>,
) -> Option<()> {
    match head {
        Head::Prim(prim) => {
            out.push(0);
            out.push(encode_prim(prim)?);
        }
        Head::Adt(item) => {
            out.push(1);
            write_u32(out, item_id(item)?);
        }
        Head::Ref(Mut::Not) => out.push(2),
        Head::Ref(Mut::Mut) => out.push(3),
        Head::Ptr(Mut::Not) => out.push(4),
        Head::Ptr(Mut::Mut) => out.push(5),
        Head::Slice => out.push(6),
        Head::Array => out.push(7),
        Head::Tuple => out.push(8),
        Head::Never => out.push(9),
        Head::Dyn(item) => {
            out.push(10);
            write_u32(out, item_id(item)?);
        }
    }
    Some(())
}

fn read_head(cur: &mut Cursor<'_>, item_at: &impl Fn(u32) -> Result<ItemRef>) -> Result<Head> {
    Ok(match cur.u8()? {
        0 => Head::Prim(decode_prim(cur.u8()?)?),
        1 => Head::Adt(item_at(cur.u32()?)?),
        2 => Head::Ref(Mut::Not),
        3 => Head::Ref(Mut::Mut),
        4 => Head::Ptr(Mut::Not),
        5 => Head::Ptr(Mut::Mut),
        6 => Head::Slice,
        7 => Head::Array,
        8 => Head::Tuple,
        9 => Head::Never,
        10 => Head::Dyn(item_at(cur.u32()?)?),
        _ => bail!("invalid impl head tag"),
    })
}

pub(super) fn write_impl_ref(
    out: &mut Vec<u8>,
    block: ImplRef,
    item_id: &mut impl FnMut(ItemRef) -> Option<u32>,
) -> Option<()> {
    match block {
        ImplRef::Module { krate, index } => {
            out.push(0);
            write_u32(out, krate);
            write_u32(out, index);
        }
        ImplRef::Body(item) => {
            let root = item.declaring_body()?;
            item_id(root)?;
            out.push(1);
            write_u32(out, item.krate);
            write_u32(out, item.file);
            write_u32(out, item.item);
        }
    }
    Some(())
}

pub(super) fn read_impl_ref(cur: &mut Cursor<'_>) -> Result<ImplRef> {
    Ok(match cur.u8()? {
        0 => ImplRef::Module {
            krate: cur.u32()?,
            index: cur.u32()?,
        },
        1 => ImplRef::Body(ItemRef {
            krate: cur.u32()?,
            file: cur.u32()?,
            item: cur.u32()?,
        }),
        _ => bail!("invalid impl ref tag"),
    })
}

fn write_impl_refs(
    out: &mut Vec<u8>,
    refs: &[ImplRef],
    item_id: &mut impl FnMut(ItemRef) -> Option<u32>,
    block_ok: &impl Fn(ImplRef) -> bool,
) -> Option<()> {
    write_u32(out, refs.len() as u32);
    for &r in refs {
        if !block_ok(r) {
            return None;
        }
        write_impl_ref(out, r, item_id)?;
    }
    Some(())
}

pub(super) fn write_impl_index(
    out: &mut Vec<u8>,
    index: &ImplIndex,
    item_id: &mut impl FnMut(ItemRef) -> Option<u32>,
    block_ok: &impl Fn(ImplRef) -> bool,
) -> Option<()> {
    write_u32(out, index.inherent.len() as u32);
    for (&head, refs) in &index.inherent {
        write_head(out, head, item_id)?;
        write_impl_refs(out, refs, item_id, block_ok)?;
    }
    write_u32(out, index.traits.len() as u32);
    for (&of, bucket) in &index.traits {
        write_u32(out, item_id(of)?);
        write_impl_refs(out, &bucket.all, item_id, block_ok)?;
        write_impl_refs(out, &bucket.blanket, item_id, block_ok)?;
        write_u32(out, bucket.by_head.len() as u32);
        for (&head, refs) in &bucket.by_head {
            write_head(out, head, item_id)?;
            write_impl_refs(out, refs, item_id, block_ok)?;
        }
    }
    write_u32(out, index.negative.len() as u32);
    for (&of, heads) in &index.negative {
        write_u32(out, item_id(of)?);
        write_u32(out, heads.len() as u32);
        for &head in heads {
            write_head(out, head, item_id)?;
        }
    }
    Some(())
}

pub(super) fn read_impl_index(
    cur: &mut Cursor<'_>,
    item_at: &impl Fn(u32) -> Result<ItemRef>,
) -> Result<ImplIndex> {
    let read_refs = |c: &mut Cursor<'_>| read_vec(c, read_impl_ref);
    let inherent_len = cur.count()?;
    let mut inherent = HashMap::with_capacity_and_hasher(inherent_len, Default::default());
    for _ in 0..inherent_len {
        inherent.insert(read_head(cur, item_at)?, read_refs(cur)?);
    }
    let traits_len = cur.count()?;
    let mut traits = HashMap::with_capacity_and_hasher(traits_len, Default::default());
    for _ in 0..traits_len {
        let of = item_at(cur.u32()?)?;
        let all = read_refs(cur)?;
        let blanket = read_refs(cur)?;
        let by_head_len = cur.count()?;
        let mut by_head = HashMap::with_capacity_and_hasher(by_head_len, Default::default());
        for _ in 0..by_head_len {
            by_head.insert(read_head(cur, item_at)?, read_refs(cur)?);
        }
        traits.insert(
            of,
            TraitBucket {
                all,
                blanket,
                by_head,
            },
        );
    }
    let negative_len = cur.count()?;
    let mut negative = HashMap::with_capacity_and_hasher(negative_len, Default::default());
    for _ in 0..negative_len {
        let of = item_at(cur.u32()?)?;
        let heads = read_vec(cur, |c| read_head(c, item_at))?;
        negative.insert(of, heads);
    }
    Ok(ImplIndex::from_parts(inherent, traits, negative))
}

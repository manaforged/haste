mod native;
mod provenance;

pub(crate) use provenance::is_freeze_ty;

use anyhow::{Result, bail};
use haste_codegen::{AbiParam, ArgumentPurpose, CallConv, Lane, Signature, Type};
use rustc_abi::{
    BackendLaneCount, BackendRepr, FieldsShape, Float, Primitive, Scalar, TargetDataLayout,
    Variants,
};
use rustc_index::Idx;

use crate::sema::Context;
use crate::sema::infer::{Shape, Typed};
use crate::sema::layout::{Layout, target};
use crate::sema::ty::Ty;

use super::adt::tag_type;
use super::{Parts, Repr, host_is_x64_or_arm64, indirect, passed_parts, repr};
use provenance::apply_param_provenance;

/// The canonical discriminant and optional payload of `purpose` for signature hashing.
pub(crate) fn purpose_tag(purpose: ArgumentPurpose) -> (u8, Option<u32>) {
    match purpose {
        ArgumentPurpose::Normal => (0, None),
        ArgumentPurpose::StructReturn => (1, None),
        ArgumentPurpose::StructArgument(n) => (2, Some(n)),
        ArgumentPurpose::Dict => (3, None),
        ArgumentPurpose::Variadic => (4, None),
    }
}

/// The native signature of a reached function: memory values travel by address unless `passed_parts` splits them into up to four scalar registers, and a memory result returns in those registers or through a first `StructReturn` pointer.
pub fn signature(cx: &mut Context<'_>, typed: &Typed) -> Result<Signature> {
    let mut sig = signature_of(
        cx,
        typed.ret,
        typed
            .params
            .iter()
            .map(|&local| typed.locals[local as usize].ty),
    )?;
    if matches!(&typed.shape, Shape::Closure(closure) if closure.closure.coroutine) {
        let object = usize::from(indirect(&sig));
        if let Some(cx_param) = sig.params.get_mut(object + 1) {
            cx_param.deref_bytes = 0;
            cx_param.noalias = false;
            cx_param.readonly = false;
        }
    }
    Ok(sig)
}

fn sret_param(size: u32) -> AbiParam {
    AbiParam::special(Type::I64, ArgumentPurpose::StructReturn)
        .with_noalias()
        .with_deref_size(size)
}

/// The native signature of a function that returns `ret` and takes `params`.
pub fn signature_of(
    cx: &mut Context<'_>,
    ret: Option<Ty>,
    params: impl IntoIterator<Item = Ty>,
) -> Result<Signature> {
    let mut signature = Signature::new(CallConv::Internal);
    if let Some(ret) = ret {
        match repr(cx, ret)? {
            Repr::Scalar(ty) => signature.returns.push(AbiParam::new(ty)),
            Repr::Mem { size, .. } => match passed_parts(cx, ret)? {
                Some(parts) => signature
                    .returns
                    .extend(parts.scalars().map(|(machine, _)| AbiParam::new(machine))),
                None => signature.params.push(sret_param(size)),
            },
            Repr::Zst => {}
        }
    }
    for param in params {
        match repr(cx, param)? {
            Repr::Scalar(ty) => {
                signature
                    .params
                    .push(apply_param_provenance(cx, AbiParam::new(ty), param, 0));
            }
            Repr::Mem { size, .. } => match passed_parts(cx, param)? {
                Some(parts) => signature
                    .params
                    .extend(parts.scalars().map(|(machine, off)| {
                        apply_param_provenance(cx, AbiParam::new(machine), param, off)
                    })),
                None => signature.params.push(
                    AbiParam::new(Type::I64)
                        .with_noalias()
                        .with_deref_bytes(size),
                ),
            },
            Repr::Zst => {}
        }
    }
    Ok(signature)
}

pub(super) fn scalar_type(dl: &TargetDataLayout, scalar: &Scalar) -> Result<Type> {
    match scalar.primitive() {
        Primitive::Float(Float::F32) => Ok(Type::F32),
        Primitive::Float(Float::F64) => Ok(Type::F64),
        Primitive::Float(other) => bail!("a native argument of float type {other:?}"),
        other => tag_type(other.size(dl).bytes()),
    }
}

/// The one or two scalars that `layout` names, each with its byte offset, or `Bytes` when it names none.
pub(super) fn parts(layout: &Layout) -> Result<Parts> {
    let dl = target()?;
    Ok(match &layout.backend_repr {
        BackendRepr::Scalar(scalar) => Parts::One(scalar_type(dl, scalar)?),
        BackendRepr::ScalarPair { a, b, b_offset } => Parts::Pair(
            scalar_type(dl, a)?,
            scalar_type(dl, b)?,
            b_offset.bytes() as i32,
        ),
        _ => Parts::Bytes,
    })
}

const WIDTHS: [(Type, u64); 4] = [
    (Type::I64, 8),
    (Type::I32, 4),
    (Type::I16, 2),
    (Type::I8, 1),
];

fn table(chunks: impl IntoIterator<Item = (Type, i32)>) -> Parts {
    let mut words = [(Type::I8, 0); 4];
    let mut len = 0usize;
    for chunk in chunks {
        if len == 4 {
            return Parts::Bytes;
        }
        words[len] = chunk;
        len += 1;
    }
    match len {
        0 => Parts::Bytes,
        1 if words[0].1 == 0 => Parts::One(words[0].0),
        2 if words[0].1 == 0 => Parts::Pair(words[0].0, words[1].0, words[1].1),
        _ => Parts::Words(words, len as u8),
    }
}

fn span_chunks(start: u64, end: u64) -> impl Iterator<Item = (Type, i32)> {
    let mut at = start;
    std::iter::from_fn(move || {
        let (ty, width) = WIDTHS
            .into_iter()
            .find(|&(_, width)| at % width == 0 && at + width <= end)?;
        let chunk = (ty, at as i32);
        at += width;
        Some(chunk)
    })
}

fn word_parts(size: u64) -> Parts {
    if !(1..=32).contains(&size) {
        return Parts::Bytes;
    }
    table(span_chunks(0, size))
}

fn eightbytes(size: u32) -> Vec<(Type, i32)> {
    (0..size.div_ceil(8))
        .map(|eb| (Type::I64, (eb * 8) as i32))
        .collect()
}

/// Breaks `size` (`1..=32`) into up to four `I64` words when `word_parts` overflows four exact power-of-two chunks.
pub(super) fn packed_word_parts(size: u32) -> Parts {
    match word_parts(u64::from(size)) {
        Parts::Bytes if (1..=32).contains(&size) => table(eightbytes(size)),
        found => found,
    }
}

fn merge_ranges(mut ranges: Vec<(u64, u64)>) -> Vec<(u64, u64)> {
    ranges.sort_unstable();
    let mut merged: Vec<(u64, u64)> = Vec::with_capacity(ranges.len());
    for (lo, hi) in ranges {
        match merged.last_mut() {
            Some(prev) if lo <= prev.1 => prev.1 = prev.1.max(hi),
            _ => merged.push((lo, hi)),
        }
    }
    merged
}

fn enum_occupied(
    layout: &Layout,
    size: u64,
    dl: &TargetDataLayout,
    cuts: &mut Vec<u64>,
) -> Vec<(u64, u64)> {
    let Variants::Multiple {
        tag,
        tag_field,
        variants,
        ..
    } = &layout.variants
    else {
        cuts.extend([0, size]);
        return vec![(0, size)];
    };
    let at = layout.fields.offset(tag_field.index()).bytes();
    let tag_end = at
        .saturating_add(tag.primitive().size(dl).bytes())
        .min(size);
    cuts.extend([at, tag_end]);
    let mut ranges = Vec::with_capacity(variants.len() + 1);
    if at < tag_end {
        ranges.push((at, tag_end));
    }
    for variant in variants.iter().filter(|variant| !variant.is_uninhabited()) {
        cuts.extend(variant.field_offsets.iter().map(|offset| offset.bytes()));
        let Some(start) = variant
            .field_offsets
            .iter()
            .map(|offset| offset.bytes())
            .min()
        else {
            continue;
        };
        let end = variant.size.bytes().min(size);
        if start < end {
            ranges.push((start, end));
        }
    }
    if layout.fields.count() > 1 {
        cuts.extend([0, size]);
        return vec![(0, size)];
    }
    let occupied = merge_ranges(ranges);
    for &(lo, hi) in &occupied {
        cuts.extend([lo, hi]);
    }
    occupied
}

fn field_parts(layout: &Layout) -> Parts {
    let size = layout.size.bytes();
    let Ok(dl) = target() else {
        return word_parts(size);
    };
    if !(1..=16).contains(&size) || matches!(layout.fields, FieldsShape::Union(_)) {
        return word_parts(size);
    }
    let mut cuts = Vec::new();
    let occupied = enum_occupied(layout, size, dl, &mut cuts);
    if let FieldsShape::Arbitrary { offsets, .. } = &layout.fields {
        cuts.extend(offsets.iter().map(|offset| offset.bytes()));
    }
    cuts.retain(|&cut| cut <= size);
    cuts.sort_unstable();
    cuts.dedup();
    let spans = cuts.windows(2).filter(|span| {
        occupied
            .iter()
            .any(|&(lo, hi)| span[0] >= lo && span[1] <= hi)
    });
    match table(spans.flat_map(|span| span_chunks(span[0], span[1]))) {
        Parts::Bytes => word_parts(size),
        found => found,
    }
}

fn simd_vec_type(element: &Scalar, count: impl FnOnce(Lane) -> u64) -> Option<Type> {
    let dl = target().ok()?;
    let scalar = scalar_type(dl, element).ok()?;
    let lane = Lane::from_type(scalar)?;
    let count = u8::try_from(count(lane)).ok()?;
    let ty = Type::Vec { lane, count };
    ty.is_valid_vec().then_some(ty)
}

fn simd_parts(element: &Scalar, count: BackendLaneCount, size: u64) -> Parts {
    let native = host_is_x64_or_arm64();
    let half = || simd_vec_type(element, |lane| u64::from(16 / lane.bytes())).filter(|_| native);
    match (
        simd_vec_type(element, |_| count.as_u64()).filter(|_| native),
        size,
    ) {
        (Some(ty), _) => Parts::One(ty),
        (None, 8) => Parts::One(Type::I64),
        (None, 16) => Parts::Pair(Type::I64, Type::I64, 8),
        (None, 32) => half().map_or(Parts::Bytes, |half| {
            match half
                .double_vec()
                .filter(|wide| wide.is_register_vec(crate::compile_target()))
            {
                Some(wide) => Parts::One(wide),
                None => Parts::Pair(half, half, 16),
            }
        }),
        (None, 64) => half().map_or(Parts::Bytes, |half| {
            Parts::Words([(half, 0), (half, 16), (half, 32), (half, 48)], 4)
        }),
        _ => Parts::Bytes,
    }
}

/// The parts that a typed copy of a value of `layout` moves: its scalars, the whole of a vector that fills one register, the 16-byte halves of a wider one, or contiguous word chunks up to 32 bytes when both scalars are a union's, or chunks cut at its field and tag boundaries when the layout names none.
pub(super) fn copied_parts(layout: &Layout) -> Parts {
    match &layout.backend_repr {
        BackendRepr::ScalarPair {
            a: Scalar::Union { .. },
            b: Scalar::Union { .. },
            ..
        } => word_parts(layout.size.bytes()),
        BackendRepr::SimdVector { element, count } => {
            simd_parts(element, *count, layout.size.bytes())
        }
        _ => match parts(layout).unwrap_or(Parts::Bytes) {
            Parts::Bytes => field_parts(layout),
            found => found,
        },
    }
}

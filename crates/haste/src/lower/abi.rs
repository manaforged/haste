use anyhow::{Result, bail};
use haste_codegen::{AbiParam, Arch, ArgumentPurpose, CallConv, Lane, Signature, Target, Type, Value};
use rustc_abi::{BackendLaneCount, BackendRepr, FieldsShape, Float, Primitive, Scalar, TargetDataLayout, Variants};
use rustc_index::Idx;

use crate::sema::Context;
use crate::sema::infer::field_types;
use crate::sema::layout::{Layout, field_offset, target};
use crate::sema::ty::{Kind, Ty};

use super::adt::tag_type;
use super::place::Place;
use super::{Lower, Parts, Repr, repr};

/// The machine values that a value of one type travels as in the platform convention, each with its byte offset inside the value.
type Passed = Vec<(Type, i32)>;

enum NativePass {
    Direct(Passed),
    Indirect,
    ByVal(u32),
}

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
        BackendRepr::ScalarPair { a, b, b_offset } => Parts::Pair(scalar_type(dl, a)?, scalar_type(dl, b)?, b_offset.bytes() as i32),
        _ => Parts::Bytes,
    })
}

fn word_parts(size: u64) -> Parts {
    if !(1..=32).contains(&size) {
        return Parts::Bytes;
    }
    let mut words = [(Type::I8, 0); 4];
    let mut len = 0usize;
    let mut offset = 0u64;
    for (ty, width) in [(Type::I64, 8), (Type::I32, 4), (Type::I16, 2), (Type::I8, 1)] {
        while offset + width <= size {
            if len == 4 {
                return Parts::Bytes;
            }
            words[len] = (ty, offset as i32);
            len += 1;
            offset += width;
        }
    }
    match len {
        1 => Parts::One(words[0].0),
        2 => Parts::Pair(words[0].0, words[1].0, words[1].1),
        _ => Parts::Words(words, len as u8),
    }
}

/// Breaks `size` (`1..=32`) into up to four `I64` words when `word_parts` overflows four exact power-of-two chunks.
pub(super) fn packed_word_parts(size: u32) -> Parts {
    match word_parts(u64::from(size)) {
        Parts::Bytes if (1..=32).contains(&size) => {
            let mut words = [(Type::I8, 0); 4];
            let len = size.div_ceil(8) as usize;
            for (i, slot) in words[..len].iter_mut().enumerate() {
                *slot = (Type::I64, (i * 8) as i32);
            }
            Parts::Words(words, len as u8)
        }
        found => found,
    }
}

fn simd_vec_type(element: &Scalar, count: BackendLaneCount) -> Option<Type> {
    let dl = target().ok()?;
    let scalar = scalar_type(dl, element).ok()?;
    let lane = Lane::from_type(scalar)?;
    let count = u8::try_from(count.as_u64()).ok()?;
    let ty = Type::Vec { lane, count };
    ty.is_valid_vec().then_some(ty)
}

/// The parts that a typed copy of a value of `layout` moves: its scalars, or contiguous word chunks up to 32 bytes when both scalars are a union's or the layout names none.
pub(super) fn copied_parts(layout: &Layout) -> Parts {
    match &layout.backend_repr {
        BackendRepr::ScalarPair { a: Scalar::Union { .. }, b: Scalar::Union { .. }, .. } => word_parts(layout.size.bytes()),
        BackendRepr::SimdVector { element, count } => match simd_vec_type(element, *count) {
            Some(ty) if matches!(Target::host().arch, Arch::X64 | Arch::Arm64) => Parts::One(ty),
            _ => match layout.size.bytes() {
                8 => Parts::One(Type::I64),
                16 => Parts::Pair(Type::I64, Type::I64, 8),
                _ => Parts::Bytes,
            },
        },
        _ => match parts(layout).unwrap_or(Parts::Bytes) {
            Parts::Bytes => word_parts(layout.size.bytes()),
            found => found,
        },
    }
}

fn collect_leaves(cx: &mut Context<'_>, ty: Ty, base: i32, out: &mut Passed) -> Result<bool> {
    match repr(cx, ty)? {
        Repr::Zst => return Ok(true),
        Repr::Scalar(m) => {
            out.push((m, base));
            return Ok(true);
        }
        Repr::Mem { .. } => {}
    }
    let layout = cx.layout(ty)?;
    if matches!(layout.fields, FieldsShape::Union(_)) {
        return Ok(false);
    }
    match cx.types.kind(ty).clone() {
        Kind::Pat(inner, ..) => collect_leaves(cx, inner, base, out),
        Kind::Array(elem, len) => collect_array_leaves(cx, elem, len, base, out),
        Kind::Tuple(items) => collect_tuple_leaves(cx, items, &layout, base, out),
        Kind::Adt(item, args) => collect_adt_leaves(cx, (item, args), &layout, base, out),
        _ => Ok(false),
    }
}

fn collect_tuple_leaves(cx: &mut Context<'_>, items: crate::sema::substs::Substs, layout: &Layout, base: i32, out: &mut Passed) -> Result<bool> {
    for (i, field) in cx.types.substs(items).to_vec().into_iter().enumerate() {
        if !collect_leaves(cx, field, base + layout.fields.offset(i).bytes() as i32, out)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn collect_adt_leaves(cx: &mut Context<'_>, (item, args): (crate::names::ItemRef, crate::sema::substs::Substs), layout: &Layout, base: i32, out: &mut Passed) -> Result<bool> {
    let Variants::Single { index } = layout.variants else { return Ok(false) };
    let slice = cx.types.substs(args).to_vec();
    let fields = field_types(cx, item, &slice)?[index.index()].clone();
    for (i, field) in fields.into_iter().enumerate() {
        if !collect_leaves(cx, field, base + field_offset(layout, index.index() as u32, i as u32), out)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn collect_array_leaves(cx: &mut Context<'_>, elem: Ty, len: Ty, base: i32, out: &mut Passed) -> Result<bool> {
    let Some(count) = cx.types.len(len).filter(|&n| n <= 32) else { return Ok(false) };
    let stride = cx.layout(elem)?.size.bytes() as i32;
    for i in 0..count {
        if !collect_leaves(cx, elem, base + i as i32 * stride, out)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn hfa_parts(leaves: &[(Type, i32)], size: u32) -> Option<Passed> {
    let first = leaves.first()?.0;
    if !(1..=4).contains(&leaves.len()) || !matches!(first, Type::F32 | Type::F64) {
        return None;
    }
    let stride = first.bytes() as i32;
    let exact_size = size as usize == leaves.len() * stride as usize;
    let contiguous = leaves.iter().enumerate().all(|(i, &(ty, off))| ty == first && off == i as i32 * stride);
    (exact_size && contiguous).then(|| leaves.to_vec())
}

fn eightbytes(size: u32) -> Passed {
    (0..size.div_ceil(8)).map(|eb| (Type::I64, (eb * 8) as i32)).collect()
}

fn x64_eightbyte(leaves: &[(Type, i32)], start: i32) -> (Type, i32) {
    let end = start + 8;
    let (mut has_float, mut has_other, mut max_end) = (false, false, start);
    for &(ty, off) in leaves {
        let leaf_end = off + ty.bytes() as i32;
        if off < end && leaf_end > start {
            if off < start || leaf_end > end || !matches!(ty, Type::F32 | Type::F64) {
                has_other = true;
            } else {
                has_float = true;
                max_end = max_end.max(leaf_end);
            }
        }
    }
    let ty = match (has_float && !has_other, max_end - start <= 4) {
        (true, true) => Type::F32,
        (true, false) => Type::F64,
        (false, _) => Type::I64,
    };
    (ty, start)
}

fn x64_eightbytes(leaves: Option<&[(Type, i32)]>, size: u32) -> Passed {
    match leaves {
        Some(leaves) => (0..size.div_ceil(8)).map(|eb| x64_eightbyte(leaves, (eb * 8) as i32)).collect(),
        None => eightbytes(size),
    }
}

fn simd_native_type(element: &Scalar, count: BackendLaneCount, size: u32, arch: Arch) -> Result<Type> {
    if matches!(arch, Arch::X64 | Arch::Arm64)
        && let Some(vec_ty) = simd_vec_type(element, count)
    {
        return Ok(vec_ty);
    }
    match size {
        16 => Ok(Type::Vec { lane: Lane::I64, count: 2 }),
        8 if arch == Arch::Arm64 => Ok(Type::F64),
        8 => Ok(Type::I64),
        _ => bail!("a {size}-byte SIMD vector across a native call is not supported yet"),
    }
}

fn arm64_native_pass(leaves: Option<&[(Type, i32)]>, size: u32) -> NativePass {
    if let Some(hfa) = leaves.and_then(|list| hfa_parts(list, size)) {
        NativePass::Direct(hfa)
    } else if size <= 16 {
        NativePass::Direct(eightbytes(size))
    } else {
        NativePass::Indirect
    }
}

impl Lower<'_, '_> {
    fn native_pass(&mut self, ty: Ty) -> Result<NativePass> {
        let size = match repr(self.cx, ty)? {
            Repr::Scalar(machine) => return Ok(NativePass::Direct(vec![(machine, 0)])),
            Repr::Zst => return Ok(NativePass::Direct(Vec::new())),
            Repr::Mem { size, .. } => size,
        };
        let layout = self.cx.layout(ty)?;
        let arch = Target::host().arch;
        if let BackendRepr::SimdVector { element, count } = &layout.backend_repr {
            return Ok(NativePass::Direct(vec![(simd_native_type(element, *count, size, arch)?, 0)]));
        }
        let mut leaves = Vec::new();
        let flat = collect_leaves(self.cx, ty, 0, &mut leaves)?;
        let flat_leaves = flat.then_some(leaves.as_slice());
        match arch {
            Arch::Arm64 => Ok(arm64_native_pass(flat_leaves, size)),
            Arch::X64 => Ok(if size <= 16 { NativePass::Direct(x64_eightbytes(flat_leaves, size)) } else { NativePass::ByVal(size) }),
            _ => match parts(&layout)? {
                Parts::Bytes => bail!("a {size}-byte aggregate across a native call is not supported yet"),
                found => Ok(NativePass::Direct(found.scalars().collect())),
            },
        }
    }

    fn load_native_part(&mut self, base: Value, offset: i32, ty: Type, size: u32) -> Value {
        if offset as u32 + ty.bytes() <= size {
            return self.ins().load(ty, base, offset);
        }
        let rem = size - offset as u32;
        let tmp = self.b.create_stack_slot(8, 3);
        let zero = self.ins().iconst(Type::I64, 0);
        self.ins().stack_store(zero, tmp, 0);
        for (chunk_ty, chunk_off) in word_parts(u64::from(rem)).scalars() {
            let byte_val = self.ins().load(chunk_ty, base, offset + chunk_off);
            self.ins().stack_store(byte_val, tmp, chunk_off);
        }
        self.ins().stack_load(ty, tmp, 0)
    }

    fn push_native_arg(&mut self, param: Ty, value: Value, signature: &mut Signature, args: &mut Vec<Value>) -> Result<()> {
        match self.native_pass(param)? {
            NativePass::Direct(parts) => {
                if let Repr::Mem { size, .. } = repr(self.cx, param)? {
                    args.extend(parts.iter().map(|&(ty, off)| self.load_native_part(value, off, ty, size)));
                } else {
                    args.push(value);
                }
                signature.params.extend(parts.iter().map(|&(ty, _)| AbiParam::new(ty)));
            }
            NativePass::Indirect => {
                let whole = repr(self.cx, param)?;
                args.push(self.snapshot_repr(whole, value));
                signature.params.push(AbiParam::new(Type::I64));
            }
            NativePass::ByVal(size) => {
                args.push(value);
                signature.params.push(AbiParam::special(Type::I64, ArgumentPurpose::StructArgument(size)));
            }
        }
        Ok(())
    }

    fn push_sret_arg(&mut self, ret_pass: Option<&NativePass>, ret: Option<Ty>, signature: &mut Signature, args: &mut Vec<Value>) -> Result<Option<haste_codegen::StackSlot>> {
        let (Some(NativePass::Indirect | NativePass::ByVal(_)), Some(ret_ty)) = (ret_pass, ret) else {
            return Ok(None);
        };
        let ret_repr = repr(self.cx, ret_ty)?;
        let slot = self.slot(ret_repr);
        if let Some(slot) = slot {
            signature.params.push(AbiParam::special(Type::I64, ArgumentPurpose::StructReturn));
            args.push(self.ins().stack_addr(Type::I64, slot, 0));
        }
        Ok(slot)
    }

    fn push_variadic_args(&mut self, values: impl Iterator<Item = Value>, signature: &mut Signature, args: &mut Vec<Value>) {
        for value in values {
            let (ty, value) = match self.b.func.ty(value) {
                Type::F32 => (Type::F64, self.ins().fpromote(Type::F64, value)),
                ty => (ty, value),
            };
            signature.params.push(AbiParam::variadic(ty));
            args.push(value);
        }
    }

    /// Calls the native function at `code`, which takes `params` and returns `ret` in the platform convention, with the generated-code `values` of its arguments.
    pub(super) fn native_call(&mut self, code: Value, (params, ret): (Vec<Ty>, Option<Ty>), values: Vec<Value>) -> Result<Option<Value>> {
        let mut signature = Signature::new(CallConv::Native);
        let (mut args, mut values) = (Vec::new(), values.into_iter());
        let ret_pass = ret.map(|ret| self.native_pass(ret)).transpose()?;
        let sret_slot = self.push_sret_arg(ret_pass.as_ref(), ret, &mut signature, &mut args)?;
        for param in params {
            if repr(self.cx, param)? == Repr::Zst {
                continue;
            }
            let Some(value) = values.next() else { bail!("a native call with too few arguments") };
            self.push_native_arg(param, value, &mut signature, &mut args)?;
        }
        self.push_variadic_args(values, &mut signature, &mut args);
        let returned = match ret_pass {
            Some(NativePass::Direct(parts)) => parts,
            _ => Vec::new(),
        };
        signature.returns.extend(returned.iter().map(|&(ty, _)| AbiParam::new(ty)));
        let sig = self.b.import_signature(signature);
        let call = self.ins().call_indirect(sig, code, &args);
        let results = self.b.inst_results(call).to_vec();
        match (ret, sret_slot) {
            (Some(ret), _) if matches!(self.cx.types.kind(ret), Kind::Never) => {
                self.ins().trap();
                self.close();
                Ok(None)
            }
            (_, Some(slot)) => Ok(Some(self.ins().stack_addr(Type::I64, slot, 0))),
            (Some(ret), None) => self.received(ret, &returned, &results),
            (None, None) => Ok(None),
        }
    }

    pub(super) fn load_packed_part(&mut self, base: Value, offset: i32, ty: Type, size: u32) -> Value {
        if offset as u32 + ty.bytes() <= size {
            return self.ins().load(ty, base, offset);
        }
        let rem = size - offset as u32;
        let mut acc: Option<Value> = None;
        for (chunk_ty, chunk_off) in word_parts(u64::from(rem)).scalars() {
            let raw = self.ins().load(chunk_ty, base, offset + chunk_off);
            let wide = if chunk_ty == Type::I64 { raw } else { self.ins().uextend(Type::I64, raw) };
            let shifted = if chunk_off == 0 {
                wide
            } else {
                let shift = self.ins().iconst(Type::I64, i64::from(chunk_off) * 8);
                self.ins().ishl(wide, shift)
            };
            acc = Some(match acc {
                Some(prev) => self.ins().bor(prev, shifted),
                None => shifted,
            });
        }
        let packed = acc.unwrap_or_else(|| self.ins().iconst(Type::I64, 0));
        if ty == Type::I64 { packed } else { self.ins().ireduce(ty, packed) }
    }

    pub(super) fn store_packed_part(&mut self, place: Place, offset: i32, ty: Type, value: Value, size: u32) {
        if offset as u32 + ty.bytes() <= size {
            self.write(place.at(offset, Repr::Scalar(ty)), value);
            return;
        }
        let rem = size - offset as u32;
        let wide = if self.b.func.ty(value) == Type::I64 { value } else { self.ins().uextend(Type::I64, value) };
        for (chunk_ty, chunk_off) in word_parts(u64::from(rem)).scalars() {
            let shifted = if chunk_off == 0 {
                wide
            } else {
                let shift = self.ins().iconst(Type::I64, i64::from(chunk_off) * 8);
                self.ins().ushr(wide, shift)
            };
            let piece = if chunk_ty == Type::I64 { shifted } else { self.ins().ireduce(chunk_ty, shifted) };
            self.write(place.at(offset + chunk_off, Repr::Scalar(chunk_ty)), piece);
        }
    }

    /// The value of a result of type `ret` that a call left in the registers `results`, one per part in `parts`: a frame slot holding them when the result lives in memory.
    pub(super) fn received(&mut self, ret: Ty, parts: &[(Type, i32)], results: &[Value]) -> Result<Option<Value>> {
        let whole = repr(self.cx, ret)?;
        let Repr::Mem { size, .. } = whole else {
            return Ok(results.first().copied());
        };
        let Some(slot) = self.slot(whole) else {
            return Ok(None);
        };
        let place = Place::slot(slot, whole);
        for (&(ty, offset), &value) in parts.iter().zip(results) {
            self.store_packed_part(place, offset, ty, value, size);
        }
        Ok(Some(self.ins().stack_addr(Type::I64, slot, 0)))
    }
}

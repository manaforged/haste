use rustc_hash::FxHashMap;

use crate::body::BinOp;
use crate::names::Primitive;
use crate::sema::intrinsic::Intrinsic;
use crate::sema::layout::field_offset;
use crate::sema::substs::Substs;
use crate::sema::ty::{Kind, Ty, int_info, mask};

use super::super::Lower;
use super::super::stencil::{RodataId, RodataReloc, RodataTarget};
use super::literal::{LITERAL_BYTES, apply_int_binary, prim_info, sign_extend};

pub(super) const MAX_CONST_STEPS: u32 = 8_192;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum PtrTarget {
    Obj { id: u32, addend: u64 },
    Rodata { id: RodataId, addend: u64 },
    Dangling(u64),
}

#[derive(Clone)]
pub(super) struct Val {
    pub(super) bytes: Vec<u8>,
    pub(super) ptrs: Vec<(usize, PtrTarget)>,
}

#[derive(Clone)]
pub(super) struct PlaceRef {
    pub(super) target: PtrTarget,
    pub(super) ty: Ty,
    pub(super) slice_len: Option<u64>,
}

pub(super) struct MemObj {
    pub(super) bytes: Vec<u8>,
    pub(super) ptrs: Vec<(usize, PtrTarget)>,
}

pub(super) struct Frame {
    pub(super) locals: Vec<Option<u32>>,
}

pub(super) enum Flow {
    Val(Val),
    Return(Val),
    Break(Option<u32>, Val),
    Continue(Option<u32>),
}

pub(super) struct ConstEnv<'l, 'a, 'r> {
    pub(super) lower: &'l mut Lower<'a, 'r>,
    pub(super) objs: Vec<MemObj>,
    pub(super) statics: FxHashMap<(crate::names::ItemRef, Substs), u32>,
    pub(super) steps: u32,
    pub(super) depth: u32,
}

impl Val {
    /// Zeroed value buffer of `size` bytes with no pointers.
    pub(super) fn zeroed(size: usize) -> Self {
        Self { bytes: vec![0; size], ptrs: Vec::new() }
    }

    /// Empty ZST value.
    pub(super) fn zst() -> Self {
        Self { bytes: Vec::new(), ptrs: Vec::new() }
    }

    /// Scalar integer/bool/char value encoded as little-endian bytes.
    pub(super) fn scalar(size: usize, bits: u128) -> Self {
        let mut bytes = vec![0; size];
        let raw = bits.to_le_bytes();
        let copy = size.min(raw.len());
        bytes[..copy].copy_from_slice(&raw[..copy]);
        Self { bytes, ptrs: Vec::new() }
    }

    /// Interprets the value as a little-endian integer up to 16 bytes.
    pub(super) fn as_u128(&self) -> Option<u128> {
        if self.bytes.len() > 16 || !self.ptrs.is_empty() {
            return None;
        }
        let mut buf = [0u8; 16];
        buf[..self.bytes.len()].copy_from_slice(&self.bytes);
        Some(u128::from_le_bytes(buf))
    }

    /// Encodes a thin pointer at `at` with synthetic non-null tag bytes.
    pub(super) fn write_ptr(&mut self, at: usize, target: PtrTarget) -> Option<()> {
        let slice = self.bytes.get_mut(at..at + 8)?;
        slice.copy_from_slice(&target.synth_addr().to_le_bytes());
        self.ptrs.retain(|&(off, _)| off + 8 <= at || off >= at + 8);
        self.ptrs.push((at, target));
        Some(())
    }

    /// Builds a thin pointer value to `target`.
    pub(super) fn ptr(target: PtrTarget) -> Self {
        let mut out = Self::zeroed(8);
        if out.write_ptr(0, target).is_some() { out } else { Self::zst() }
    }

    /// Builds a 16-byte wide slice/str pointer value `(target, len)`.
    pub(super) fn wide(target: PtrTarget, len: u64) -> Self {
        let mut out = Self::zeroed(16);
        out.bytes[8..16].copy_from_slice(&len.to_le_bytes());
        if out.write_ptr(0, target).is_some() { out } else { Self::zst() }
    }

    /// Reads a pointer target stored at byte offset `at`, falling back to `Dangling`.
    pub(super) fn read_ptr(&self, at: usize) -> Option<PtrTarget> {
        if let Some(&(_, target)) = self.ptrs.iter().find(|&&(off, _)| off == at) {
            return Some(target);
        }
        let slice = self.bytes.get(at..at + 8)?;
        let addr = u64::from_le_bytes(slice.try_into().ok()?);
        (addr != 0).then_some(PtrTarget::Dangling(addr))
    }
}

impl PtrTarget {
    /// Synthetic non-zero address used to satisfy non-null and niche tag checks.
    pub(super) fn synth_addr(self) -> u64 {
        match self {
            Self::Obj { id, addend } => ((u64::from(id) + 1) << 32).wrapping_add(addend),
            Self::Rodata { id, addend } => ((u64::from(id.0) + 1) << 48).wrapping_add(addend),
            Self::Dangling(align) => align.max(1),
        }
    }

    /// Returns a target shifted forward by `delta` bytes.
    pub(super) fn offset(self, delta: u64) -> Option<Self> {
        match self {
            Self::Obj { id, addend } => addend.checked_add(delta).map(|addend| Self::Obj { id, addend }),
            Self::Rodata { id, addend } => addend.checked_add(delta).map(|addend| Self::Rodata { id, addend }),
            Self::Dangling(align) if delta == 0 => Some(Self::Dangling(align)),
            Self::Dangling(_) => None,
        }
    }
}

impl<'l, 'a, 'r> ConstEnv<'l, 'a, 'r> {
    pub(super) fn new(lower: &'l mut Lower<'a, 'r>, depth: u32) -> Self {
        Self { lower, objs: Vec::new(), statics: FxHashMap::default(), steps: 0, depth }
    }

    /// Consumes one evaluation step, returning `None` once the budget is spent.
    pub(super) fn tick(&mut self) -> Option<()> {
        if self.steps >= MAX_CONST_STEPS {
            return None;
        }
        self.steps += 1;
        Some(())
    }

    /// Allocates a mutable interpreter object initialized to `val`.
    pub(super) fn alloc_val(&mut self, val: Val) -> Option<u32> {
        let id = u32::try_from(self.objs.len()).ok()?;
        self.objs.push(MemObj { bytes: val.bytes, ptrs: val.ptrs });
        Some(id)
    }

    /// Allocates a zeroed object of type `ty`.
    pub(super) fn alloc_ty(&mut self, ty: Ty) -> Option<u32> {
        let size = self.ty_size(ty)?;
        self.alloc_val(Val::zeroed(size))
    }

    /// Returns the byte size of sized type `ty` when bounded within `LITERAL_BYTES`.
    pub(super) fn ty_size(&mut self, ty: Ty) -> Option<usize> {
        if ty == Ty::NONE || self.lower.cx.types.has_params(ty) {
            return None;
        }
        let layout = self.lower.cx.layout(ty).ok()?;
        if layout.align.bytes() > 16 {
            return None;
        }
        usize::try_from(layout.size.bytes()).ok().filter(|&s| s <= LITERAL_BYTES)
    }

    /// Reads `size` bytes with embedded pointer relocations starting at `target`.
    pub(super) fn read_bytes(&self, target: PtrTarget, size: usize) -> Option<Val> {
        if size == 0 {
            return Some(Val::zst());
        }
        let PtrTarget::Obj { id, addend } = target else { return None };
        let obj = self.objs.get(id as usize)?;
        let start = usize::try_from(addend).ok()?;
        let end = start.checked_add(size)?;
        let bytes = obj.bytes.get(start..end)?.to_vec();
        let ptrs = obj.ptrs.iter().filter(|&&(off, _)| off >= start && off + 8 <= end).map(|&(off, rel)| (off - start, rel)).collect();
        Some(Val { bytes, ptrs })
    }

    /// Reads the value represented by `place` (`16`-byte wide pointer if unsized).
    pub(super) fn read_place(&mut self, place: &PlaceRef) -> Option<Val> {
        if self.lower.cx.is_unsized(place.ty).ok()? {
            return Some(Val::wide(place.target, place.slice_len?));
        }
        let size = self.ty_size(place.ty)?;
        self.read_bytes(place.target, size)
    }

    /// Writes `val` at `target`, overwriting bytes and overlapping pointer tags.
    pub(super) fn write_bytes(&mut self, target: PtrTarget, val: &Val) -> Option<()> {
        if val.bytes.is_empty() {
            return Some(());
        }
        let PtrTarget::Obj { id, addend } = target else { return None };
        let obj = self.objs.get_mut(id as usize)?;
        let start = usize::try_from(addend).ok()?;
        let end = start.checked_add(val.bytes.len())?;
        obj.bytes.get_mut(start..end)?.copy_from_slice(&val.bytes);
        obj.ptrs.retain(|&(off, _)| off + 8 <= start || off >= end);
        for &(off, rel) in &val.ptrs {
            obj.ptrs.push((start + off, rel));
        }
        Some(())
    }

    /// Exports an evaluated `Val` into rodata bytes and `RodataReloc` entries.
    pub(super) fn export_val(&mut self, val: Val, exported: &mut FxHashMap<u32, RodataId>) -> Option<(Vec<u8>, Vec<RodataReloc>)> {
        let mut bytes = val.bytes;
        let mut relocs = Vec::new();
        for (off, target) in val.ptrs {
            let rod_id = match target {
                PtrTarget::Obj { id, addend: 0 } => self.export_obj(id, exported)?,
                PtrTarget::Obj { .. } => return None,
                PtrTarget::Rodata { id, addend: 0 } => id,
                PtrTarget::Rodata { .. } => return None,
                PtrTarget::Dangling(_) => self.lower.slots.rodata(&[], &[]),
            };
            bytes.get_mut(off..off + 8)?.fill(0);
            relocs.push(RodataReloc { offset: u32::try_from(off).ok()?, target: RodataTarget::Rodata(rod_id) });
        }
        Some((bytes, relocs))
    }

    fn export_obj(&mut self, id: u32, exported: &mut FxHashMap<u32, RodataId>) -> Option<RodataId> {
        if let Some(&rod_id) = exported.get(&id) {
            return Some(rod_id);
        }
        let obj = self.objs.get(id as usize)?;
        let val = Val { bytes: obj.bytes.clone(), ptrs: obj.ptrs.clone() };
        let (bytes, relocs) = self.export_val(val, exported)?;
        let rod_id = self.lower.slots.rodata(&bytes, &relocs);
        exported.insert(id, rod_id);
        Some(rod_id)
    }

    /// Evaluates pure compiler intrinsics (`ptr_metadata`, `size_of`, integer ops, `const_eval_select`).
    pub(super) fn eval_intrinsic(&mut self, which: Intrinsic, substs: &[Ty], args: &[Val]) -> Option<Val> {
        match which {
            Intrinsic::PtrMetadata => self.eval_ptr_metadata(substs, args),
            Intrinsic::SizeOf | Intrinsic::AlignOf | Intrinsic::NeedsDrop => self.eval_type_intrinsic(which, substs),
            Intrinsic::Assume => (args.first()?.as_u128()? != 0).then(Val::zst),
            Intrinsic::NoOp => Some(Val::zst()),
            Intrinsic::BlackBox => args.first().cloned(),
            Intrinsic::Transmute => {
                let dst = *substs.get(1)?;
                let val = args.first()?.clone();
                (val.bytes.len() == self.ty_size(dst)?).then_some(val)
            }
            Intrinsic::ConstEvalSelect => self.eval_const_eval_select(substs, args),
            Intrinsic::ThreeWayCompare => self.eval_three_way_cmp(*substs.first()?, args),
            _ => self.eval_int_intrinsic(which, substs, args),
        }
    }

    fn eval_ptr_metadata(&mut self, substs: &[Ty], args: &[Val]) -> Option<Val> {
        let tail = self.lower.cx.tail(*substs.first()?).ok()?;
        if !matches!(self.lower.cx.types.kind(tail), Kind::Slice(_) | Kind::Prim(Primitive::Str)) {
            return None;
        }
        let slice = args.first()?.bytes.get(8..16)?;
        Some(Val::scalar(8, u128::from(u64::from_le_bytes(slice.try_into().ok()?))))
    }

    fn eval_type_intrinsic(&mut self, which: Intrinsic, substs: &[Ty]) -> Option<Val> {
        let ty = *substs.first()?;
        match which {
            Intrinsic::SizeOf => Some(Val::scalar(8, u128::from(self.lower.cx.layout(ty).ok()?.size.bytes()))),
            Intrinsic::AlignOf => Some(Val::scalar(8, u128::from(self.lower.cx.layout(ty).ok()?.align.bytes()))),
            Intrinsic::NeedsDrop => Some(Val::scalar(1, u128::from(crate::sema::infer::needs_drop(self.lower.cx, ty).ok()?))),
            _ => None,
        }
    }

    fn eval_const_eval_select(&mut self, substs: &[Ty], args: &[Val]) -> Option<Val> {
        let &tuple = substs.first()?;
        let Kind::Tuple(fields) = *self.lower.cx.types.kind(tuple) else { return None };
        let field_tys = self.lower.cx.types.substs(fields).to_vec();
        let layout = self.lower.cx.layout(tuple).ok()?;
        let mut unpacked = Vec::new();
        if let Some(arg) = args.first() {
            let id = self.alloc_val(arg.clone())?;
            for (idx, fty) in field_tys.into_iter().enumerate() {
                let off = u64::try_from(field_offset(&layout, 0, idx as u32)).ok()?;
                unpacked.push(self.read_place(&PlaceRef { target: PtrTarget::Obj { id, addend: off }, ty: fty, slice_len: None })?);
            }
        }
        for candidate in [substs.get(1).copied(), substs.get(2).copied()].into_iter().flatten() {
            if let Kind::FnDef(item, fn_substs) = *self.lower.cx.types.kind(candidate)
                && let Some(val) = self.call_fn(item, fn_substs, unpacked.clone())
            {
                return Some(val);
            }
        }
        None
    }

    fn eval_three_way_cmp(&self, ty: Ty, args: &[Val]) -> Option<Val> {
        let (&[ref a, ref b], &Kind::Prim(prim)) = (args, self.lower.cx.types.kind(ty)) else { return None };
        let (l, r) = (a.as_u128()?, b.as_u128()?);
        let (bits, signed) = prim_info(prim)?;
        let res: i8 = if signed { sign_extend(l, bits).cmp(&sign_extend(r, bits)) as i8 } else { l.cmp(&r) as i8 };
        Some(Val::scalar(1, (res as i128 as u128) & 0xff))
    }

    fn eval_int_intrinsic(&mut self, which: Intrinsic, substs: &[Ty], args: &[Val]) -> Option<Val> {
        let Kind::Prim(prim) = *self.lower.cx.types.kind(*substs.first()?) else { return None };
        let (bits, _) = int_info(prim)?;
        let size = (bits / 8) as usize;
        match (which, args) {
            (Intrinsic::WrappingAdd | Intrinsic::WrappingSub | Intrinsic::WrappingMul, [x, y]) => self.eval_wrapping_intrinsic(which, (bits, size), x, y),
            (Intrinsic::RotateLeft | Intrinsic::RotateRight, [x, by]) => self.eval_rotate_intrinsic(which == Intrinsic::RotateLeft, (bits, size), x, by),
            (Intrinsic::Ctpop | Intrinsic::Ctlz | Intrinsic::Cttz | Intrinsic::CttzNonzero, [x]) => self.eval_bitcount_intrinsic(which, bits, x),
            (Intrinsic::Unchecked(op), [x, y]) => apply_int_binary(op, prim, x.as_u128()?, y.as_u128()?).map(|v| Val::scalar(size, v)),
            (Intrinsic::WithOverflow(op), [x, y]) => self.eval_overflow_intrinsic((op, prim, bits), (x.as_u128()?, y.as_u128()?), size),
            _ => None,
        }
    }

    fn eval_wrapping_intrinsic(&self, which: Intrinsic, (bits, size): (u32, usize), x: &Val, y: &Val) -> Option<Val> {
        let (a, b) = (x.as_u128()?, y.as_u128()?);
        let out = match which {
            Intrinsic::WrappingAdd => a.wrapping_add(b),
            Intrinsic::WrappingSub => a.wrapping_sub(b),
            Intrinsic::WrappingMul => a.wrapping_mul(b),
            _ => return None,
        };
        Some(Val::scalar(size, out & mask(bits)))
    }

    fn eval_rotate_intrinsic(&self, left: bool, (bits, size): (u32, usize), x: &Val, by: &Val) -> Option<Val> {
        let shift = (by.as_u128()? as u32) % bits;
        let v = x.as_u128()? & mask(bits);
        let out = if shift == 0 {
            v
        } else if left {
            ((v << shift) | (v >> (bits - shift))) & mask(bits)
        } else {
            ((v >> shift) | (v << (bits - shift))) & mask(bits)
        };
        Some(Val::scalar(size, out))
    }

    fn eval_bitcount_intrinsic(&self, which: Intrinsic, bits: u32, x: &Val) -> Option<Val> {
        let v = x.as_u128()? & mask(bits);
        let count = match which {
            Intrinsic::Ctpop => v.count_ones(),
            Intrinsic::Ctlz => v.leading_zeros() - (128 - bits),
            Intrinsic::Cttz | Intrinsic::CttzNonzero => if v == 0 { bits } else { v.trailing_zeros() },
            _ => return None,
        };
        Some(Val::scalar(4, u128::from(count)))
    }

    fn eval_overflow_intrinsic(&mut self, (op, prim, bits): (BinOp, Primitive, u32), (x, y): (u128, u128), size: usize) -> Option<Val> {
        let wrapping = match op {
            BinOp::Add => x.wrapping_add(y) & mask(bits),
            BinOp::Sub => x.wrapping_sub(y) & mask(bits),
            BinOp::Mul => x.wrapping_mul(y) & mask(bits),
            _ => return None,
        };
        let overflowed = apply_int_binary(op, prim, x, y).is_none();
        let tuple_ty = self.lower.cx.types.tuple(&[self.lower.cx.types.prim(prim), self.lower.cx.types.prim(Primitive::Bool)]);
        let layout = self.lower.cx.layout(tuple_ty).ok()?;
        let mut out = Val::zeroed(usize::try_from(layout.size.bytes()).ok()?);
        let (off0, off1) = (usize::try_from(field_offset(&layout, 0, 0)).ok()?, usize::try_from(field_offset(&layout, 0, 1)).ok()?);
        out.bytes[off0..off0 + size].copy_from_slice(&wrapping.to_le_bytes()[..size]);
        out.bytes[off1] = u8::from(overflowed);
        Some(out)
    }
}

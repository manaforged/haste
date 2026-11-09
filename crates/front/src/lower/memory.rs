use anyhow::{Result, bail};
use haste_codegen::{IntCC, Type, Value};

use crate::body::BinOp;
use crate::sema::infer::{ALIGN_SLOT, SIZE_SLOT};
use crate::sema::intrinsic::Intrinsic;
use crate::sema::ty::{Kind, Ty};

use super::ops::Num;
use super::place::{Base, Place};
use super::{Lower, Repr, repr};

impl Lower<'_, '_> {
    /// Emits the memory and layout intrinsic `which` inline, and hands every other intrinsic to the value intrinsics.
    pub(super) fn memory_intrinsic(&mut self, which: Intrinsic, substs: &[Ty], args: &[Value]) -> Result<Option<Value>> {
        use Intrinsic::*;
        let ty = substs.first().copied();
        match (which, ty, args) {
            (SizeOf | AlignOf, Some(ty), _) => self.layout_const(which, ty).map(Some),
            (SizeOfVal | AlignOfVal, Some(ty), &[pointer]) => self.value_layout(which == SizeOfVal, ty, pointer).map(Some),
            (NeedsDrop | TypeId, Some(ty), _) => self.type_fact(which, ty).map(Some),
            (TypeName, Some(ty), _) => {
                let name = self.cx.type_text(ty, &|ty| ty);
                let data = self.slots.data(name.as_bytes());
                let data = self.ins().iconst(Type::I64, data);
                Ok(Some(self.wide_pointer(data, name.len() as i64)))
            }
            (NoOp, _, _) => Ok(None),
            (Fence, _, _) => {
                self.ins().fence();
                Ok(None)
            }
            (BlackBox, _, _) => Ok(args.first().copied()),
            (CompareBytes | RawEq, _, _) => self.byte_compare(which, ty, args).map(Some),
            (DiscriminantValue, Some(ty), &[pointer]) => {
                let place = self.pointee_place(pointer, ty)?;
                self.discriminant_value(place, ty).map(Some)
            }
            (Atomic(atomic), Some(ty), _) => self.atomic(atomic, ty, args),
            (Unchecked(op), Some(ty), &[x, y]) => self.unchecked(op, (ty, substs.get(1).copied()), x, y).map(Some),
            (ThreeWayCompare, Some(ty), &[x, y]) => self.compare(ty, x, y).map(Some),
            (RawPointer | Offset(_) | OffsetFrom(_) | Read | Write | WriteBox | MemCopy(_) | WriteBytes | SliceElement, Some(_), _) => self.access_intrinsic(which, substs, args),
            _ => self.value_intrinsic(which, substs, args).map(Some),
        }
    }

    fn access_intrinsic(&mut self, which: Intrinsic, substs: &[Ty], args: &[Value]) -> Result<Option<Value>> {
        use Intrinsic::*;
        let ty = substs[0];
        match (which, args) {
            (RawPointer, &[data, ..]) => match (repr(self.cx, ty)?, args.get(1)) {
                (Repr::Mem { .. }, Some(&meta)) => Ok(Some(self.wide(data, meta))),
                (Repr::Scalar(_), _) => Ok(Some(data)),
                _ => bail!("aggregate_raw_ptr into a pointer with metadata that is not a slice length"),
            },
            (Offset(pointer), &[base, count]) => self.offset(ty, pointer, base, count).map(Some),
            (OffsetFrom(signed), &[to, from]) => self.offset_from(ty, signed, to, from).map(Some),
            (Read, &[from]) => self.read_at(ty, from),
            (Write, &[to, value]) => {
                let at = Place { base: Base::Addr(to), offset: 0, repr: repr(self.cx, ty)? };
                self.write(at, value);
                Ok(None)
            }
            (Write, &[_]) => Ok(None),
            (WriteBox, &[boxed, ..]) => self.write_box(ty, boxed, args.get(1).copied()).map(Some),
            (MemCopy(overlapping), &[from, to, count]) => self.copy_items(ty, overlapping, (from, to), count).map(|()| None),
            (WriteBytes, &[to, byte, count]) => self.write_bytes(ty, to, byte, count).map(|()| None),
            (SliceElement, &[slice, index]) => self.slice_element(substs, slice, index).map(Some),
            _ => bail!("intrinsic {which:?} with {} arguments", args.len()),
        }
    }

    fn write_box(&mut self, ty: Ty, boxed: Value, value: Option<Value>) -> Result<Value> {
        if let Some(value) = value {
            let data = self.box_data(boxed);
            let at = Place { base: Base::Addr(data), offset: 0, repr: repr(self.cx, ty)? };
            self.write(at, value);
        }
        Ok(boxed)
    }

    fn type_fact(&mut self, which: Intrinsic, ty: Ty) -> Result<Value> {
        if which == Intrinsic::TypeId {
            return Ok(self.type_id(ty));
        }
        let needs = crate::sema::infer::needs_drop(self.cx, ty)?;
        Ok(self.ins().iconst(Type::I8, i64::from(needs)))
    }

    fn type_id(&mut self, ty: Ty) -> Value {
        let slot = self.b.create_stack_slot(16, 3);
        let id = self.ins().iconst(Type::I64, i64::from(ty.0));
        let zero = self.ins().iconst(Type::I64, 0);
        self.ins().stack_store(id, slot, 0);
        self.ins().stack_store(zero, slot, 8);
        self.ins().stack_addr(Type::I64, slot, 0)
    }

    fn layout_const(&mut self, which: Intrinsic, ty: Ty) -> Result<Value> {
        let layout = self.cx.layout(ty)?;
        let value = if which == Intrinsic::SizeOf { layout.size.bytes() } else { layout.align.abi.bytes() };
        Ok(self.ins().iconst(Type::I64, value as i64))
    }

    fn value_layout(&mut self, size: bool, ty: Ty, pointer: Value) -> Result<Value> {
        let which = if size { Intrinsic::SizeOf } else { Intrinsic::AlignOf };
        match self.cx.types.kind(ty).clone() {
            Kind::Slice(element) if size => {
                let len = self.ins().load(Type::I64, pointer, 8);
                let bytes = self.size(element)?;
                Ok(self.ins().imul_imm(len, bytes))
            }
            Kind::Slice(element) => self.layout_const(which, element),
            Kind::Prim(crate::names::Primitive::Str) if size => Ok(self.ins().load(Type::I64, pointer, 8)),
            Kind::Prim(crate::names::Primitive::Str) => Ok(self.ins().iconst(Type::I64, 1)),
            Kind::Dyn(..) => {
                let vtable = self.ins().load(Type::I64, pointer, 8);
                let slot = if size { SIZE_SLOT } else { ALIGN_SLOT };
                Ok(self.ins().load(Type::I64, vtable, slot as i32 * 8))
            }
            Kind::Adt(item, args) if self.cx.is_unsized(ty)? => self.tail_layout(size, (ty, item, &args), pointer),
            _ => self.layout_const(which, ty),
        }
    }

    fn tail_layout(&mut self, size: bool, (ty, item, args): (Ty, crate::names::ItemRef, &[Ty]), pointer: Value) -> Result<Value> {
        let layout = self.cx.layout(ty)?;
        let fields = crate::sema::infer::field_types(self.cx, item, args)?;
        let Some((index, &field)) = fields.first().and_then(|fields| fields.iter().enumerate().last()) else { bail!("an unsized struct with no fields") };
        let offset = crate::sema::layout::field_offset(&layout, 0, index as u32);
        let field_align = self.value_layout(false, field, pointer)?;
        let own_align = self.ins().iconst(Type::I64, layout.align.abi.bytes() as i64);
        let wider = self.ins().icmp(IntCC::UnsignedGreaterThan, own_align, field_align);
        let align = self.ins().select(wider, own_align, field_align);
        if !size {
            return Ok(align);
        }
        let field_size = self.value_layout(true, field, pointer)?;
        let end = self.ins().iadd_imm(field_size, i64::from(offset));
        Ok(self.align_up(end, align))
    }

    /// The least multiple of `align`, a power of two, that is at least `value`.
    pub(super) fn align_up(&mut self, value: Value, align: Value) -> Value {
        let up = self.ins().iadd(value, align);
        let up = self.ins().iadd_imm(up, -1);
        let mask = self.ins().ineg(align);
        self.ins().band(up, mask)
    }

    fn slice_element(&mut self, substs: &[Ty], slice: Value, index: Value) -> Result<Value> {
        let Some(&element) = substs.get(2) else { bail!("slice_get_unchecked takes three type arguments") };
        let size = self.size(element)?;
        let data = self.ins().load(Type::I64, slice, 0);
        let bytes = self.ins().imul_imm(index, size);
        Ok(self.ins().iadd(data, bytes))
    }

    fn size(&mut self, ty: Ty) -> Result<i64> {
        Ok(self.cx.layout(ty)?.size.bytes() as i64)
    }

    fn offset(&mut self, ty: Ty, pointer: bool, base: Value, count: Value) -> Result<Value> {
        let pointee = match (pointer, self.cx.types.kind(ty)) {
            (true, Kind::Ptr(_, inner) | Kind::Ref(_, inner)) => *inner,
            (true, other) => bail!("offset of a pointer of type {other:?}"),
            (false, _) => ty,
        };
        let size = self.size(pointee)?;
        let bytes = self.ins().imul_imm(count, size);
        Ok(self.ins().iadd(base, bytes))
    }

    fn offset_from(&mut self, ty: Ty, signed: bool, to: Value, from: Value) -> Result<Value> {
        let size = self.size(ty)?;
        let bytes = self.ins().isub(to, from);
        let size = self.ins().iconst(Type::I64, size.max(1));
        Ok(if signed { self.ins().sdiv(bytes, size) } else { self.ins().udiv(bytes, size) })
    }

    fn read_at(&mut self, ty: Ty, from: Value) -> Result<Option<Value>> {
        let whole = repr(self.cx, ty)?;
        let at = Place { base: Base::Addr(from), offset: 0, repr: whole };
        let Repr::Mem { size, .. } = whole else { return Ok(self.read(at)) };
        let Some(slot) = self.slot(whole) else { return Ok(None) };
        let to = self.ins().stack_addr(Type::I64, slot, 0);
        self.copy(to, from, size);
        Ok(Some(to))
    }

    fn copy_items(&mut self, ty: Ty, overlapping: bool, (from, to): (Value, Value), count: Value) -> Result<()> {
        let size = self.size(ty)?;
        if size == 0 {
            return Ok(());
        }
        let backward = match overlapping {
            true => Some(self.ins().icmp(IntCC::UnsignedGreaterThan, to, from)),
            false => None,
        };
        let last = self.ins().iadd_imm(count, -1);
        self.each(count, |lower, index| {
            let item = match backward {
                Some(backward) => {
                    let down = lower.ins().isub(last, index);
                    lower.ins().select(backward, down, index)
                }
                None => index,
            };
            let offset = lower.ins().imul_imm(item, size);
            let (source, target) = (lower.ins().iadd(from, offset), lower.ins().iadd(to, offset));
            lower.copy(target, source, size as u32);
        });
        Ok(())
    }

    fn write_bytes(&mut self, ty: Ty, to: Value, byte: Value, count: Value) -> Result<()> {
        let size = self.size(ty)?;
        let bytes = self.ins().imul_imm(count, size);
        self.each(bytes, |lower, index| {
            let at = lower.ins().iadd(to, index);
            lower.ins().store(byte, at, 0);
        });
        Ok(())
    }

    fn byte_compare(&mut self, which: Intrinsic, ty: Option<Ty>, args: &[Value]) -> Result<Value> {
        match (which, ty, args) {
            (Intrinsic::CompareBytes, _, &[x, y, count]) => Ok(self.compare_bytes((x, y), count)),
            (Intrinsic::RawEq, Some(ty), &[x, y]) => {
                let size = self.layout_const(Intrinsic::SizeOf, ty)?;
                let order = self.compare_bytes((x, y), size);
                Ok(self.ins().icmp_imm(IntCC::Equal, order, 0))
            }
            _ => bail!("intrinsic {which:?} with {} arguments", args.len()),
        }
    }

    fn compare_bytes(&mut self, (x, y): (Value, Value), count: Value) -> Value {
        let order = self.b.create_stack_slot(4, 2);
        let zero = self.ins().iconst(Type::I32, 0);
        self.ins().stack_store(zero, order, 0);
        self.each(count, |lower, index| {
            let (left, right) = (lower.ins().iadd(x, index), lower.ins().iadd(y, index));
            let (left, right) = (lower.ins().load(Type::I8, left, 0), lower.ins().load(Type::I8, right, 0));
            let (left, right) = (lower.ins().uextend(Type::I32, left), lower.ins().uextend(Type::I32, right));
            let difference = lower.ins().isub(left, right);
            let so_far = lower.ins().stack_load(Type::I32, order, 0);
            let zero = lower.ins().iconst(Type::I32, 0);
            let undecided = lower.ins().icmp(IntCC::Equal, so_far, zero);
            let next = lower.ins().select(undecided, difference, so_far);
            lower.ins().stack_store(next, order, 0);
        });
        self.ins().stack_load(Type::I32, order, 0)
    }

    pub(super) fn each(&mut self, count: Value, mut body: impl FnMut(&mut Self, Value)) {
        let (head, inner, exit) = (self.b.create_block(), self.b.create_block(), self.b.create_block());
        let index = self.b.append_block_param(head, Type::I64);
        let zero = self.ins().iconst(Type::I64, 0);
        self.jump(head, &[zero]);
        self.enter(head);
        let more = self.ins().icmp(IntCC::UnsignedLessThan, index, count);
        self.brif(more, (inner, &[]), (exit, &[]));
        self.enter(inner);
        body(self, index);
        let next = self.ins().iadd_imm(index, 1);
        self.jump(head, &[next]);
        self.enter(exit);
    }

    fn unchecked(&mut self, op: BinOp, (ty, by): (Ty, Option<Ty>), x: Value, y: Value) -> Result<Value> {
        let Num::Int { ty: machine, signed } = self.num(ty)? else { bail!("unchecked {op:?} on a non-integer") };
        let y = match by.map(|by| self.num(by)).transpose()? {
            Some(Num::Int { ty: from, signed: by_signed }) if from != machine => self.resize(y, (from, by_signed), machine),
            _ => y,
        };
        Ok(match (op, signed) {
            (BinOp::Div, true) => self.ins().sdiv(x, y),
            (BinOp::Div, false) => self.ins().udiv(x, y),
            (BinOp::Rem, true) => self.ins().srem(x, y),
            (BinOp::Rem, false) => self.ins().urem(x, y),
            (BinOp::Shl, _) => self.ins().ishl(x, y),
            (_, true) => self.ins().sshr(x, y),
            (_, false) => self.ins().ushr(x, y),
        })
    }

    fn compare(&mut self, ty: Ty, x: Value, y: Value) -> Result<Value> {
        let Num::Int { signed, .. } = self.num(ty)? else { bail!("three_way_compare on a non-integer") };
        let (less, greater) = match signed {
            true => (IntCC::SignedLessThan, IntCC::SignedGreaterThan),
            false => (IntCC::UnsignedLessThan, IntCC::UnsignedGreaterThan),
        };
        let (below, above) = (self.ins().icmp(less, x, y), self.ins().icmp(greater, x, y));
        let (minus, zero, one) = (self.ins().iconst(Type::I8, -1), self.ins().iconst(Type::I8, 0), self.ins().iconst(Type::I8, 1));
        let other = self.ins().select(above, one, zero);
        let order = self.ins().select(below, minus, other);
        let slot = self.b.create_stack_slot(1, 0);
        self.ins().stack_store(order, slot, 0);
        Ok(self.ins().stack_addr(Type::I64, slot, 0))
    }
}

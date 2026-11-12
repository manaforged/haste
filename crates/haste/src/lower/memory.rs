use anyhow::{Result, bail};
use haste_codegen::{IntCC, RuntimeSymbol, Type, Value};

use crate::body::BinOp;
use crate::names::Primitive;
use crate::sema::infer::{ALIGN_SLOT, SIZE_SLOT};
use crate::sema::intrinsic::Intrinsic;
use crate::sema::ty::{Kind, Ty};

use super::ops::Num;
use super::place::{Base, Place};
use super::stencil::SymbolSource;
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
                let id = self.slots.rodata(name.as_bytes(), &[]);
                let data = self.emit_symbol(SymbolSource::Rodata(id));
                let len = self.emit_symbol(SymbolSource::Word(name.len() as u64));
                Ok(Some(self.wide(data, len)))
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
            (RawPointer | Offset(_) | PtrMask | OffsetFrom(_) | Read | Write | WriteBox | MemCopy(_) | WriteBytes | SliceElement, Some(_), _) => self.access_intrinsic(which, substs, args),
            _ => self.value_intrinsic(which, substs, args).map(Some),
        }
    }

    fn access_intrinsic(&mut self, which: Intrinsic, substs: &[Ty], args: &[Value]) -> Result<Option<Value>> {
        use Intrinsic::*;
        let ty = substs[0];
        match (which, args) {
            (RawPointer, &[data, ..]) => self.raw_pointer(substs, data, args.get(1).copied()),
            (Offset(pointer), &[base, count]) => self.offset(ty, pointer, base, count).map(Some),
            (PtrMask, &[pointer, mask]) => Ok(Some(self.ins().band(pointer, mask))),
            (OffsetFrom(signed), &[to, from]) => self.offset_from(ty, signed, to, from).map(Some),
            (Read, &[from]) => self.read_at(ty, from),
            (Write, &[to, value]) => {
                let at = Place { base: Base::Addr(to), offset: 0, repr: repr(self.cx, ty)? };
                self.write(at, value);
                Ok(None)
            }
            (Write, &[_]) => Ok(None),
            (WriteBox, &[boxed, ..]) => self.write_box(ty, boxed, args.get(1).copied()).map(Some),
            (MemCopy(_), &[from, to, count]) => self.copy_items(ty, (from, to), count).map(|()| None),
            (WriteBytes, &[to, byte, count]) => self.write_bytes(ty, to, byte, count).map(|()| None),
            (SliceElement, &[slice, index]) => self.slice_element(substs, slice, index).map(Some),
            _ => bail!("intrinsic {which:?} with {} arguments", args.len()),
        }
    }

    fn raw_pointer(&mut self, substs: &[Ty], data: Value, meta: Option<Value>) -> Result<Option<Value>> {
        let ty = substs[0];
        let data = match substs.get(1).copied().map(|from| repr(self.cx, from)).transpose()? {
            Some(Repr::Mem { .. }) => self.ins().load(Type::I64, data, 0),
            _ => data,
        };
        if matches!(repr(self.cx, ty)?, Repr::Scalar(_)) {
            return Ok(Some(data));
        }
        let (Kind::Ptr(_, pointee) | Kind::Ref(_, pointee)) = *self.cx.types.kind(ty) else {
            bail!("aggregate_raw_ptr into a type that is not a pointer");
        };
        let Some(meta) = meta else { bail!("aggregate_raw_ptr into a wide pointer with no metadata") };
        let tail = self.cx.tail(pointee)?;
        let meta = match self.cx.types.kind(tail) {
            Kind::Dyn(..) => self.ins().load(Type::I64, meta, 0),
            _ => meta,
        };
        Ok(Some(self.wide(data, meta)))
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
        let id = self.emit_symbol(SymbolSource::Word(u64::from(ty.0)));
        let zero = self.emit_symbol(SymbolSource::Word(0));
        self.ins().stack_store(id, slot, 0);
        self.ins().stack_store(zero, slot, 8);
        self.ins().stack_addr(Type::I64, slot, 0)
    }

    fn layout_const(&mut self, which: Intrinsic, ty: Ty) -> Result<Value> {
        let layout = self.cx.layout(ty)?;
        let value = if which == Intrinsic::SizeOf { layout.size.bytes() } else { layout.align.bytes() };
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
            Kind::Prim(Primitive::Str) if size => Ok(self.ins().load(Type::I64, pointer, 8)),
            Kind::Prim(Primitive::Str) => Ok(self.ins().iconst(Type::I64, 1)),
            Kind::Dyn(..) => {
                let vtable = self.ins().load(Type::I64, pointer, 8);
                let slot = if size { SIZE_SLOT } else { ALIGN_SLOT };
                Ok(self.ins().load(Type::I64, vtable, slot as i32 * 8))
            }
            Kind::Adt(item, args) if self.cx.is_unsized(ty)? => {
                let slice = self.cx.types.substs(args).to_vec();
                self.tail_layout(size, (ty, item, &slice), pointer)
            }
            _ => self.layout_const(which, ty),
        }
    }

    fn tail_layout(&mut self, size: bool, (ty, item, args): (Ty, crate::names::ItemRef, &[Ty]), pointer: Value) -> Result<Value> {
        let layout = self.cx.layout(ty)?;
        let fields = crate::sema::infer::field_types(self.cx, item, args)?;
        let Some((index, &field)) = fields.first().and_then(|fields| fields.iter().enumerate().last()) else { bail!("an unsized struct with no fields") };
        let offset = crate::sema::layout::field_offset(&layout, 0, index as u32);
        let field_align = self.value_layout(false, field, pointer)?;
        let own_align = self.ins().iconst(Type::I64, layout.align.bytes() as i64);
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
        let Repr::Mem { .. } = whole else {
            return Ok(self.read(at));
        };
        let Some(slot) = self.slot(whole) else {
            return Ok(None);
        };
        let to = self.ins().stack_addr(Type::I64, slot, 0);
        self.copy(to, from, whole);
        Ok(Some(to))
    }

    fn copy_items(&mut self, ty: Ty, (from, to): (Value, Value), count: Value) -> Result<()> {
        let size = self.size(ty)?;
        if size == 0 {
            return Ok(());
        }
        let bytes = self.ins().imul_imm(count, size);
        self.ins().copy(to, from, bytes);
        Ok(())
    }

    fn write_bytes(&mut self, ty: Ty, to: Value, byte: Value, count: Value) -> Result<()> {
        let size = self.size(ty)?;
        let bytes = self.ins().imul_imm(count, size);
        let byte = self.ins().uextend(Type::I64, byte);
        self.call_libc(RuntimeSymbol::Memset, vec![to, byte, bytes], None)?;
        Ok(())
    }

    fn call_libc(&mut self, symbol: RuntimeSymbol, args: Vec<Value>, ret: Option<Ty>) -> Result<Option<Value>> {
        let usize_ty = self.cx.types.prim(Primitive::Usize);
        let code = self.ins().runtime(symbol);
        self.call_code(code, (vec![usize_ty; args.len()], ret), args)
    }

    fn byte_compare(&mut self, which: Intrinsic, ty: Option<Ty>, args: &[Value]) -> Result<Value> {
        match (which, ty, args) {
            (Intrinsic::CompareBytes, _, &[x, y, count]) => self.compare_bytes((x, y), count),
            (Intrinsic::RawEq, Some(ty), &[x, y]) => {
                let size = self.layout_const(Intrinsic::SizeOf, ty)?;
                let order = self.compare_bytes((x, y), size)?;
                Ok(self.ins().icmp_imm(IntCC::Equal, order, 0))
            }
            _ => bail!("intrinsic {which:?} with {} arguments", args.len()),
        }
    }

    fn compare_bytes(&mut self, (x, y): (Value, Value), count: Value) -> Result<Value> {
        let int = self.cx.types.prim(Primitive::I32);
        let Some(order) = self.call_libc(RuntimeSymbol::Memcmp, vec![x, y, count], Some(int))? else { bail!("memcmp returned no value") };
        Ok(order)
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

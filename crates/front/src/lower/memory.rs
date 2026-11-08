use anyhow::{Result, bail};
use haste_codegen::{IntCC, Type, Value};

use crate::body::BinOp;
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
            (NeedsDrop, Some(ty), _) => {
                let needs = crate::sema::infer::needs_drop(self.cx, ty)?;
                Ok(Some(self.ins().iconst(Type::I8, i64::from(needs))))
            }
            (NoOp, _, _) => Ok(None),
            (Fence, _, _) => {
                self.ins().fence();
                Ok(None)
            }
            (BlackBox, _, _) => Ok(args.first().copied()),
            (Unchecked(op), Some(ty), &[x, y]) => self.unchecked(op, (ty, substs.get(1).copied()), x, y).map(Some),
            (ThreeWayCompare, Some(ty), &[x, y]) => self.compare(ty, x, y).map(Some),
            (RawPointer | Offset(_) | OffsetFrom(_) | Read | Write | MemCopy(_) | WriteBytes, Some(ty), _) => self.access_intrinsic(which, ty, args),
            _ => self.value_intrinsic(which, substs, args).map(Some),
        }
    }

    fn access_intrinsic(&mut self, which: Intrinsic, ty: Ty, args: &[Value]) -> Result<Option<Value>> {
        use Intrinsic::*;
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
            (MemCopy(overlapping), &[from, to, count]) => self.copy_items(ty, overlapping, (from, to), count).map(|()| None),
            (WriteBytes, &[to, byte, count]) => self.write_bytes(ty, to, byte, count).map(|()| None),
            _ => bail!("intrinsic {which:?} with {} arguments", args.len()),
        }
    }

    fn layout_const(&mut self, which: Intrinsic, ty: Ty) -> Result<Value> {
        let layout = self.cx.layout(ty)?;
        let value = if which == Intrinsic::SizeOf { layout.size.bytes() } else { layout.align.abi.bytes() };
        Ok(self.ins().iconst(Type::I64, value as i64))
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

    fn each(&mut self, count: Value, mut body: impl FnMut(&mut Self, Value)) {
        let (head, inner, exit) = (self.b.create_block(), self.b.create_block(), self.b.create_block());
        let index = self.b.append_block_param(head, Type::I64);
        let zero = self.ins().iconst(Type::I64, 0);
        self.ins().jump(head, &[zero]);
        self.b.switch_to_block(head);
        let more = self.ins().icmp(IntCC::UnsignedLessThan, index, count);
        self.ins().brif(more, inner, &[], exit, &[]);
        self.b.switch_to_block(inner);
        body(self, index);
        let next = self.ins().iadd_imm(index, 1);
        self.ins().jump(head, &[next]);
        self.b.switch_to_block(exit);
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

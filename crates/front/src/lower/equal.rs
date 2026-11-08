use anyhow::{Result, bail};
use haste_codegen::{FloatCC, IntCC, Type, Value};

use crate::names::Primitive;
use crate::sema::infer::field_types;
use crate::sema::ty::{Kind, Ty};

use super::place::{Base, Place};
use super::{Lower, Repr, repr};

impl Lower<'_, '_> {
    /// Whether the values `x` and `y` of `ty` are equal part by part, the way a literal or constant pattern compares them, as an 8-bit flag.
    pub(super) fn equal(&mut self, ty: Ty, x: Value, y: Value) -> Result<Value> {
        let kind = self.cx.types.kind(ty).clone();
        if let Kind::Ref(_, inner) = kind {
            return self.equal_pointee(inner, x, y);
        }
        match repr(self.cx, ty)? {
            Repr::Zst => Ok(self.ins().iconst(Type::I8, 1)),
            Repr::Scalar(Type::F32 | Type::F64) => Ok(self.ins().fcmp(FloatCC::Equal, x, y)),
            Repr::Scalar(_) => Ok(self.ins().icmp(IntCC::Equal, x, y)),
            Repr::Mem { .. } => self.equal_parts(ty, kind, x, y),
        }
    }

    fn equal_parts(&mut self, ty: Ty, kind: Kind, x: Value, y: Value) -> Result<Value> {
        let fields = match kind {
            Kind::Array(element, count) => {
                let count = self.len(count)?;
                let count = self.ins().iconst(Type::I64, count);
                return self.equal_items(element, (x, y), count);
            }
            Kind::Tuple(items) => items,
            Kind::Adt(item, args) => match &field_types(self.cx, item, &args)?[..] {
                [fields] => fields.clone(),
                _ => bail!("a pattern constant of an enum is not compared yet"),
            },
            other => bail!("a pattern value of {other:?} is not compared yet"),
        };
        let whole = repr(self.cx, ty)?;
        let mut all = self.ins().iconst(Type::I8, 1);
        for (index, field) in fields.into_iter().enumerate() {
            let at = |base| Place { base: Base::Addr(base), offset: 0, repr: whole };
            let (left, right) = (self.part_place(at(x), ty, (0, index as u32))?, self.part_place(at(y), ty, (0, index as u32))?);
            let (Some(left), Some(right)) = (self.read(left), self.read(right)) else { continue };
            let same = self.equal(field, left, right)?;
            all = self.ins().band(all, same);
        }
        Ok(all)
    }

    fn equal_pointee(&mut self, inner: Ty, x: Value, y: Value) -> Result<Value> {
        let element = match self.cx.types.kind(inner).clone() {
            Kind::Prim(Primitive::Str) => self.cx.types.intern(Kind::Prim(Primitive::U8)),
            Kind::Slice(element) => element,
            _ => {
                let pointee = repr(self.cx, inner)?;
                let (left, right) = (Place { base: Base::Addr(x), offset: 0, repr: pointee }, Place { base: Base::Addr(y), offset: 0, repr: pointee });
                let (Some(left), Some(right)) = (self.read(left), self.read(right)) else { return Ok(self.ins().iconst(Type::I8, 1)) };
                return self.equal(inner, left, right);
            }
        };
        let (data_x, len_x) = (self.ins().load(Type::I64, x, 0), self.ins().load(Type::I64, x, 8));
        let (data_y, len_y) = (self.ins().load(Type::I64, y, 0), self.ins().load(Type::I64, y, 8));
        let same_len = self.ins().icmp(IntCC::Equal, len_x, len_y);
        let zero = self.ins().iconst(Type::I64, 0);
        let count = self.ins().select(same_len, len_x, zero);
        let items = self.equal_items(element, (data_x, data_y), count)?;
        Ok(self.ins().band(same_len, items))
    }

    fn equal_items(&mut self, element: Ty, (x, y): (Value, Value), count: Value) -> Result<Value> {
        let size = self.cx.layout(element)?.size.bytes() as i64;
        let part = repr(self.cx, element)?;
        let all = self.b.create_stack_slot(1, 0);
        let one = self.ins().iconst(Type::I8, 1);
        self.ins().stack_store(one, all, 0);
        let mut failed = None;
        self.each(count, |lower, index| {
            let offset = lower.ins().imul_imm(index, size);
            let (left, right) = (lower.ins().iadd(x, offset), lower.ins().iadd(y, offset));
            let at = |base| Place { base: Base::Addr(base), offset: 0, repr: part };
            let (Some(left), Some(right)) = (lower.read(at(left)), lower.read(at(right))) else { return };
            match lower.equal(element, left, right) {
                Ok(same) => {
                    let so_far = lower.ins().stack_load(Type::I8, all, 0);
                    let both = lower.ins().band(so_far, same);
                    lower.ins().stack_store(both, all, 0);
                }
                Err(error) => failed = Some(error),
            }
        });
        if let Some(error) = failed {
            return Err(error);
        }
        Ok(self.ins().stack_load(Type::I8, all, 0))
    }
}

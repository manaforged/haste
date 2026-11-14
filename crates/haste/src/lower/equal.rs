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
        match kind {
            Kind::Ref(_, inner) => return self.equal_pointee(inner, x, y),
            Kind::Pat(base, ..) => return self.equal(base, x, y),
            _ => {}
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
                if self.is_bitwise_int(element) {
                    let total = (count as u64).saturating_mul(self.cx.layout(element)?.size.bytes());
                    if total <= 16 {
                        return Ok(self.equal_bytes_const((x, y), total));
                    }
                }
                let count = self.ins().iconst(Type::I64, count);
                return self.equal_items(element, (x, y), count);
            }
            Kind::Tuple(items) => self.cx.types.substs(items).to_vec(),
            Kind::Adt(item, args) => {
                let slice = self.cx.types.substs(args).to_vec();
                match &field_types(self.cx, item, &slice)?[..] {
                    [fields] => fields.clone(),
                    variants => return self.equal_enum(ty, variants, (x, y)),
                }
            }
            Kind::Ptr(..) => return self.equal_words(ty, x, y),
            other => bail!("a pattern value of {other:?} is not compared yet"),
        };
        let whole = repr(self.cx, ty)?;
        let mut all = self.ins().iconst(Type::I8, 1);
        for (index, field) in fields.into_iter().enumerate() {
            let at = |base| Place { base: Base::Addr(base), offset: 0, repr: whole };
            let (left, right) = (self.part_place(at(x), ty, (0, index as u32))?, self.part_place(at(y), ty, (0, index as u32))?);
            let (Some(left), Some(right)) = (self.read(left), self.read(right)) else {
                continue;
            };
            let same = self.equal(field, left, right)?;
            all = self.ins().band(all, same);
        }
        Ok(all)
    }

    /// Whether the enum values at `x` and `y` of `ty`, whose variants hold the fields `variants`, hold the same variant with equal fields.
    fn equal_enum(&mut self, ty: Ty, variants: &[Vec<Ty>], (x, y): (Value, Value)) -> Result<Value> {
        let whole = repr(self.cx, ty)?;
        let at = |base| Place { base: Base::Addr(base), offset: 0, repr: whole };
        let (left, right) = (self.discriminant_value(at(x), ty)?, self.discriminant_value(at(y), ty)?);
        let same = self.ins().icmp(IntCC::Equal, left, right);
        let all = self.b.create_stack_slot(1, 0);
        self.ins().stack_store(same, all, 0);
        for (variant, fields) in variants.iter().enumerate().filter(|(_, fields)| !fields.is_empty()) {
            let Some(holds) = self.is_variant(Some(at(x)), ty, variant as u32)? else { continue };
            let (compare, next) = (self.b.create_block(), self.b.create_block());
            let both = self.ins().band(holds, same);
            self.brif(both, (compare, &[]), (next, &[]));
            self.enter(compare);
            let mut equal = self.ins().iconst(Type::I8, 1);
            for (index, &field) in fields.iter().enumerate() {
                let (left, right) = (self.variant_field(at(x), ty, (variant as u32, index as u32))?, self.variant_field(at(y), ty, (variant as u32, index as u32))?);
                let (Some(left), Some(right)) = (self.read(left), self.read(right)) else { continue };
                let same_field = self.equal(field, left, right)?;
                equal = self.ins().band(equal, same_field);
            }
            self.ins().stack_store(equal, all, 0);
            self.jump(next, &[]);
            self.enter(next);
        }
        Ok(self.ins().stack_load(Type::I8, all, 0))
    }

    /// Whether the values at `x` and `y` of `ty`, a wide raw pointer, hold the same address and metadata.
    fn equal_words(&mut self, ty: Ty, x: Value, y: Value) -> Result<Value> {
        let size = self.cx.layout(ty)?.size.bytes() as i32;
        let mut all = self.ins().iconst(Type::I8, 1);
        for offset in (0..size).step_by(8) {
            let (left, right) = (self.ins().load(Type::I64, x, offset), self.ins().load(Type::I64, y, offset));
            let same = self.ins().icmp(IntCC::Equal, left, right);
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
                let (Some(left), Some(right)) = (self.read(left), self.read(right)) else {
                    return Ok(self.ins().iconst(Type::I8, 1));
                };
                return self.equal(inner, left, right);
            }
        };
        let (data_x, len_x) = (self.ins().load(Type::I64, x, 0), self.ins().load(Type::I64, x, 8));
        let (data_y, len_y) = (self.ins().load(Type::I64, y, 0), self.ins().load(Type::I64, y, 8));
        let same_len = self.ins().icmp(IntCC::Equal, len_x, len_y);
        let (check, done) = (self.b.create_block(), self.b.create_block());
        let res = self.b.append_block_param(done, Type::I8);
        let zero = self.ins().iconst(Type::I8, 0);
        self.brif(same_len, (check, &[]), (done, &[zero]));
        self.enter(check);
        let items = self.equal_items(element, (data_x, data_y), len_y)?;
        self.jump(done, &[items]);
        self.enter(done);
        Ok(res)
    }

    fn is_byte(&self, ty: Ty) -> bool {
        matches!(self.cx.types.kind(ty), Kind::Prim(Primitive::U8 | Primitive::I8 | Primitive::Bool))
    }

    fn is_bitwise_int(&self, ty: Ty) -> bool {
        self.is_byte(ty)
            || matches!(
                self.cx.types.kind(ty),
                Kind::Prim(Primitive::U16 | Primitive::I16 | Primitive::U32 | Primitive::I32 | Primitive::U64 | Primitive::I64 | Primitive::Char)
            )
    }

    fn equal_load(&mut self, ty: Type, (x, y): (Value, Value), off: i32) -> Value {
        let (a, b) = (self.ins().load(ty, x, off), self.ins().load(ty, y, off));
        self.ins().icmp(IntCC::Equal, a, b)
    }

    fn xor_load(&mut self, ty: Type, (x, y): (Value, Value), off: i32) -> Value {
        let (a, b) = (self.ins().load(ty, x, off), self.ins().load(ty, y, off));
        self.ins().bxor(a, b)
    }

    fn equal_pair_xor(&mut self, ty: Type, (x0, y0): (Value, Value), off0: i32, (x1, y1): (Value, Value), off1: i32) -> Value {
        let d0 = self.xor_load(ty, (x0, y0), off0);
        let d1 = self.xor_load(ty, (x1, y1), off1);
        let any = self.ins().bor(d0, d1);
        let zero = self.ins().iconst(ty, 0);
        self.ins().icmp(IntCC::Equal, any, zero)
    }

    fn equal_tail_pair(&mut self, ty: Type, bytes: i64, (x, y): (Value, Value), (off, len): (Value, Value)) -> Value {
        let (px, py) = (self.ins().iadd(x, off), self.ins().iadd(y, off));
        let end_off = self.ins().iadd_imm(len, -bytes);
        let (ex, ey) = (self.ins().iadd(x, end_off), self.ins().iadd(y, end_off));
        self.equal_pair_xor(ty, (px, py), 0, (ex, ey), 0)
    }

    fn equal_bytes_const(&mut self, (x, y): (Value, Value), len: u64) -> Value {
        match len {
            0 => self.ins().iconst(Type::I8, 1),
            1 => self.equal_load(Type::I8, (x, y), 0),
            2 => self.equal_load(Type::I16, (x, y), 0),
            3 => self.equal_pair_xor(Type::I16, (x, y), 0, (x, y), 1),
            4 => self.equal_load(Type::I32, (x, y), 0),
            5..=7 => self.equal_pair_xor(Type::I32, (x, y), 0, (x, y), (len - 4) as i32),
            8 => self.equal_load(Type::I64, (x, y), 0),
            9..=16 => self.equal_pair_xor(Type::I64, (x, y), 0, (x, y), (len - 8) as i32),
            _ => {
                let count = self.ins().iconst(Type::I64, len as i64);
                self.equal_bytes_runtime((x, y), count)
            }
        }
    }

    fn equal_bytes_runtime(&mut self, (x, y): (Value, Value), len: Value) -> Value {
        let (head, body, tail) = (self.b.create_block(), self.b.create_block(), self.b.create_block());
        let (ge4, lt4, ge2, lt2, one_byte, done) = (
            self.b.create_block(),
            self.b.create_block(),
            self.b.create_block(),
            self.b.create_block(),
            self.b.create_block(),
            self.b.create_block(),
        );
        let off = self.b.append_block_param(head, Type::I64);
        let res = self.b.append_block_param(done, Type::I8);
        let zero64 = self.ins().iconst(Type::I64, 0);
        let zero8 = self.ins().iconst(Type::I8, 0);
        let one8 = self.ins().iconst(Type::I8, 1);
        self.jump(head, &[zero64]);

        self.enter(head);
        let rem = self.ins().isub(len, off);
        let has8 = self.ins().icmp_imm(IntCC::UnsignedGreaterThanOrEqual, rem, 8);
        self.brif(has8, (body, &[]), (tail, &[]));

        self.enter(body);
        let (px, py) = (self.ins().iadd(x, off), self.ins().iadd(y, off));
        let eq8 = self.equal_load(Type::I64, (px, py), 0);
        let next = self.ins().iadd_imm(off, 8);
        self.brif(eq8, (head, &[next]), (done, &[zero8]));

        self.enter(tail);
        let has4 = self.ins().icmp_imm(IntCC::UnsignedGreaterThanOrEqual, rem, 4);
        self.brif(has4, (ge4, &[]), (lt4, &[]));

        self.enter(ge4);
        let eq4 = self.equal_tail_pair(Type::I32, 4, (x, y), (off, len));
        self.jump(done, &[eq4]);

        self.enter(lt4);
        let has2 = self.ins().icmp_imm(IntCC::UnsignedGreaterThanOrEqual, rem, 2);
        self.brif(has2, (ge2, &[]), (lt2, &[]));

        self.enter(ge2);
        let eq2 = self.equal_tail_pair(Type::I16, 2, (x, y), (off, len));
        self.jump(done, &[eq2]);

        self.enter(lt2);
        let is_zero = self.ins().icmp_imm(IntCC::Equal, rem, 0);
        self.brif(is_zero, (done, &[one8]), (one_byte, &[]));

        self.enter(one_byte);
        let (px, py) = (self.ins().iadd(x, off), self.ins().iadd(y, off));
        let eq1 = self.equal_load(Type::I8, (px, py), 0);
        self.jump(done, &[eq1]);

        self.enter(done);
        res
    }

    fn equal_items(&mut self, element: Ty, (x, y): (Value, Value), count: Value) -> Result<Value> {
        if self.is_byte(element) {
            return Ok(self.equal_bytes_runtime((x, y), count));
        }
        let size = self.cx.layout(element)?.size.bytes() as i64;
        let part = repr(self.cx, element)?;
        let (head, body, done) = (self.b.create_block(), self.b.create_block(), self.b.create_block());
        let index = self.b.append_block_param(head, Type::I64);
        let res = self.b.append_block_param(done, Type::I8);
        let zero64 = self.ins().iconst(Type::I64, 0);
        let zero8 = self.ins().iconst(Type::I8, 0);
        let one8 = self.ins().iconst(Type::I8, 1);
        self.jump(head, &[zero64]);

        self.enter(head);
        let more = self.ins().icmp(IntCC::UnsignedLessThan, index, count);
        self.brif(more, (body, &[]), (done, &[one8]));

        self.enter(body);
        let offset = self.ins().imul_imm(index, size);
        let (left, right) = (self.ins().iadd(x, offset), self.ins().iadd(y, offset));
        let at = |base| Place { base: Base::Addr(base), offset: 0, repr: part };
        let next = self.ins().iadd_imm(index, 1);
        if let (Some(left), Some(right)) = (self.read(at(left)), self.read(at(right))) {
            let same = self.equal(element, left, right)?;
            self.brif(same, (head, &[next]), (done, &[zero8]));
        } else {
            self.jump(head, &[next]);
        }

        self.enter(done);
        Ok(res)
    }
}

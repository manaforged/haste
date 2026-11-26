use anyhow::{Result, bail};
use haste_codegen::{FloatCC, IntCC, Type, Value};

use crate::names::Primitive;
use crate::sema::infer::field_types;
use crate::sema::ty::{Kind, Ty};

use super::place::{Base, Place};
use super::{Lower, Repr, repr};

const INLINE_EQ_BYTES: u64 = 32;

pub(super) fn bitwise_prim_bytes(prim: Primitive) -> Option<usize> {
    match prim {
        Primitive::U8 | Primitive::I8 | Primitive::Bool => Some(1),
        Primitive::U16 | Primitive::I16 => Some(2),
        Primitive::U32 | Primitive::I32 | Primitive::Char => Some(4),
        Primitive::U64 | Primitive::I64 => Some(8),
        _ => None,
    }
}

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
            Kind::Array(element, count) => return self.equal_array(element, count, (x, y)),
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
            let at = |base| Place {
                base: Base::Addr(base),
                offset: 0,
                repr: whole,
            };
            let (left, right) = (
                self.part_place(at(x), ty, (0, index as u32))?,
                self.part_place(at(y), ty, (0, index as u32))?,
            );
            let (Some(left), Some(right)) = (self.read(left), self.read(right)) else {
                continue;
            };
            let same = self.equal(field, left, right)?;
            all = self.ins().band(all, same);
        }
        Ok(all)
    }

    fn equal_array(&mut self, element: Ty, count: Ty, (x, y): (Value, Value)) -> Result<Value> {
        let count = self.len(count)?;
        if self.is_bitwise_int(element) {
            let total = (count as u64).saturating_mul(self.cx.layout(element)?.size.bytes());
            if total <= INLINE_EQ_BYTES {
                return self.equal_bytes_const((x, y), total);
            }
        }
        let count = self.ins().iconst(Type::I64, count);
        self.equal_items(element, (x, y), count)
    }

    /// Whether the enum values at `x` and `y` of `ty`, whose variants hold the fields `variants`, hold the same variant with equal fields.
    fn equal_enum(
        &mut self,
        ty: Ty,
        variants: &[Vec<Ty>],
        (x, y): (Value, Value),
    ) -> Result<Value> {
        let whole = repr(self.cx, ty)?;
        let at = |base| Place {
            base: Base::Addr(base),
            offset: 0,
            repr: whole,
        };
        let (left, right) = (
            self.discriminant_value(at(x), ty)?,
            self.discriminant_value(at(y), ty)?,
        );
        let mut same = self.ins().icmp(IntCC::Equal, left, right);
        for (variant, fields) in variants
            .iter()
            .enumerate()
            .filter(|(_, fields)| !fields.is_empty())
        {
            let Some(holds) = self.is_variant(Some(at(x)), ty, variant as u32)? else {
                continue;
            };
            let (compare, next) = (self.b.create_block(), self.b.create_block());
            let out = self.b.append_block_param(next, Type::I8);
            let both = self.ins().band(holds, same);
            self.brif(both, (compare, &[]), (next, &[same]));
            self.enter(compare);
            let mut equal = self.ins().iconst(Type::I8, 1);
            for (index, &field) in fields.iter().enumerate() {
                let (left, right) = (
                    self.variant_field(at(x), ty, (variant as u32, index as u32))?,
                    self.variant_field(at(y), ty, (variant as u32, index as u32))?,
                );
                let (Some(left), Some(right)) = (self.read(left), self.read(right)) else {
                    continue;
                };
                let same_field = self.equal(field, left, right)?;
                equal = self.ins().band(equal, same_field);
            }
            self.jump(next, &[equal]);
            self.enter(next);
            same = out;
        }
        Ok(same)
    }

    /// Whether the values at `x` and `y` of `ty`, a wide raw pointer, hold the same address and metadata.
    fn equal_words(&mut self, ty: Ty, x: Value, y: Value) -> Result<Value> {
        let size = self.cx.layout(ty)?.size.bytes() as i32;
        let mut all = self.ins().iconst(Type::I8, 1);
        for offset in (0..size).step_by(8) {
            let (left, right) = (
                self.ins().load(Type::I64, x, offset),
                self.ins().load(Type::I64, y, offset),
            );
            let same = self.ins().icmp(IntCC::Equal, left, right);
            all = self.ins().band(all, same);
        }
        Ok(all)
    }

    fn load_wide_pair(&mut self, ptr: Value) -> (Value, Value, Option<u64>) {
        let static_len = self.b.wide_const_len(ptr);
        let data = self.ins().load(Type::I64, ptr, 0);
        let len = match static_len.and_then(|n| i64::try_from(n).ok()) {
            Some(n) => self.ins().iconst(Type::I64, n),
            None => self.ins().load(Type::I64, ptr, 8),
        };
        (data, len, static_len)
    }

    fn equal_pointee(&mut self, inner: Ty, x: Value, y: Value) -> Result<Value> {
        let element = match self.cx.types.kind(inner).clone() {
            Kind::Prim(Primitive::Str) => self.cx.types.intern(Kind::Prim(Primitive::U8)),
            Kind::Slice(element) => element,
            _ => {
                let pointee = repr(self.cx, inner)?;
                let (left, right) = (
                    Place {
                        base: Base::Addr(x),
                        offset: 0,
                        repr: pointee,
                    },
                    Place {
                        base: Base::Addr(y),
                        offset: 0,
                        repr: pointee,
                    },
                );
                let (Some(left), Some(right)) = (self.read(left), self.read(right)) else {
                    return Ok(self.ins().iconst(Type::I8, 1));
                };
                return self.equal(inner, left, right);
            }
        };
        let (data_x, len_x, static_x) = self.load_wide_pair(x);
        let (data_y, len_y, static_y) = self.load_wide_pair(y);
        let elem_bytes = self
            .is_bitwise_int(element)
            .then(|| self.cx.layout(element).map(|l| l.size.bytes()))
            .transpose()?;
        let static_bytes = static_y
            .or(static_x)
            .zip(elem_bytes)
            .and_then(|(c, b)| c.checked_mul(b))
            .filter(|&bytes| bytes <= INLINE_EQ_BYTES);
        let same_len = self.ins().icmp(IntCC::Equal, len_x, len_y);
        let (check, done) = (self.b.create_block(), self.b.create_block());
        let res = self.b.append_block_param(done, Type::I8);
        let zero = self.ins().iconst(Type::I8, 0);
        self.brif(same_len, (check, &[]), (done, &[zero]));
        self.enter(check);
        let items = match static_bytes {
            Some(bytes) => self.equal_bytes_const((data_x, data_y), bytes)?,
            None => self.equal_items(element, (data_x, data_y), len_y)?,
        };
        self.jump(done, &[items]);
        self.enter(done);
        Ok(res)
    }

    fn is_bitwise_int(&self, ty: Ty) -> bool {
        matches!(*self.cx.types.kind(ty), Kind::Prim(prim) if bitwise_prim_bytes(prim).is_some())
    }

    fn equal_load(&mut self, ty: Type, (x, y): (Value, Value), off: i32) -> Value {
        let (a, b) = (self.ins().load(ty, x, off), self.ins().load(ty, y, off));
        self.ins().icmp(IntCC::Equal, a, b)
    }

    fn xor_load(&mut self, ty: Type, (x, y): (Value, Value), off: i32) -> Value {
        let (a, b) = (self.ins().load(ty, x, off), self.ins().load(ty, y, off));
        self.ins().bxor(a, b)
    }

    fn equal_xors(&mut self, ty: Type, pair: (Value, Value), offsets: &[i32]) -> Value {
        let mut any = self.xor_load(ty, pair, offsets[0]);
        for &off in &offsets[1..] {
            let next = self.xor_load(ty, pair, off);
            any = self.ins().bor(any, next);
        }
        let zero = self.ins().iconst(ty, 0);
        self.ins().icmp(IntCC::Equal, any, zero)
    }

    pub(super) fn equal_bytes_const(&mut self, (x, y): (Value, Value), len: u64) -> Result<Value> {
        let eq = match len {
            0 => self.ins().iconst(Type::I8, 1),
            1 => self.equal_load(Type::I8, (x, y), 0),
            2 => self.equal_load(Type::I16, (x, y), 0),
            3 => self.equal_xors(Type::I16, (x, y), &[0, 1]),
            4 => self.equal_load(Type::I32, (x, y), 0),
            5..=7 => self.equal_xors(Type::I32, (x, y), &[0, (len - 4) as i32]),
            8 => self.equal_load(Type::I64, (x, y), 0),
            9..=16 => self.equal_xors(Type::I64, (x, y), &[0, (len - 8) as i32]),
            17..=24 => self.equal_xors(Type::I64, (x, y), &[0, 8, (len - 8) as i32]),
            25..=32 => self.equal_xors(Type::I64, (x, y), &[0, 8, 16, (len - 8) as i32]),
            _ => {
                let count = self.ins().iconst(Type::I64, len as i64);
                let order = self.compare_bytes((x, y), count)?;
                self.ins().icmp_imm(IntCC::Equal, order, 0)
            }
        };
        Ok(eq)
    }

    fn equal_bytes_runtime(&mut self, (x, y): (Value, Value), len: Value) -> Result<Value> {
        if let Some(n) = self.b.const_int(len)
            && n >= 0
            && (n as u64) <= INLINE_EQ_BYTES
        {
            return self.equal_bytes_const((x, y), n as u64);
        }
        let order = self.compare_bytes((x, y), len)?;
        Ok(self.ins().icmp_imm(IntCC::Equal, order, 0))
    }

    fn equal_items(&mut self, element: Ty, (x, y): (Value, Value), count: Value) -> Result<Value> {
        if self.is_bitwise_int(element) {
            let elem_size = self.cx.layout(element)?.size.bytes() as i64;
            let bytes = if elem_size == 1 {
                count
            } else {
                self.ins().imul_imm(count, elem_size)
            };
            return self.equal_bytes_runtime((x, y), bytes);
        }
        let size = self.cx.layout(element)?.size.bytes() as i64;
        let part = repr(self.cx, element)?;
        let (head, body, done) = (
            self.b.create_block(),
            self.b.create_block(),
            self.b.create_block(),
        );
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
        let at = |base| Place {
            base: Base::Addr(base),
            offset: 0,
            repr: part,
        };
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

use anyhow::{Result, bail};
use haste_codegen::{IntCC, Type, Value};
use rustc_abi::{Primitive, TagEncoding, Variants};
use rustc_index::Idx;

use crate::body::{ExprId, List};
use crate::names::ItemRef;
use crate::sema::Context;
use crate::sema::infer::{ALIGN_SLOT, field_types};
use crate::sema::layout::{Layout, Variant, field_offset, target};
use crate::sema::ty::{Kind, Ty};

use super::ops::Num;
use super::place::{Base, Place};
use super::{Lower, Repr, repr};

pub(super) fn tag_type(bytes: u64) -> Result<Type> {
    Ok(match bytes {
        1 => Type::I8,
        2 => Type::I16,
        4 => Type::I32,
        8 => Type::I64,
        16 => Type::I128,
        other => bail!("no tag of {other} bytes"),
    })
}

/// Where an enum keeps its tag and how it encodes its variants in it.
struct Tag {
    offset: i32,
    ty: Type,
    mask: u128,
    encoding: TagEncoding<Variant>,
}

fn tag_of(layout: &Layout) -> Result<Option<Tag>> {
    let Variants::Multiple { tag, tag_encoding, tag_field, .. } = &layout.variants else {
        return Ok(None);
    };
    let size = tag.size(target()?);
    let mask = crate::sema::ty::mask(size.bits() as u32);
    Ok(Some(Tag { offset: layout.fields.offset(tag_field.index()).bytes() as i32, ty: tag_type(size.bytes())?, mask, encoding: tag_encoding.clone() }))
}

/// The offset, type and bits of the tag that a value of variant `variant` of the enum `item` with `layout` holds, or none when the layout keeps no tag for it.
pub(super) fn tag_bits(cx: &mut Context<'_>, item: ItemRef, layout: &Layout, variant: u32) -> Result<Option<(i32, Type, u128)>> {
    let Some(tag) = tag_of(layout)? else {
        return Ok(None);
    };
    let value = match &tag.encoding {
        TagEncoding::Direct => cx.discriminants(item)?[variant as usize] as u128,
        TagEncoding::Niche { untagged_variant, niche_variants, niche_start } => {
            if untagged_variant.index() == variant as usize {
                return Ok(None);
            }
            ((variant as u128).wrapping_sub(niche_variants.start.index() as u128)).wrapping_add(*niche_start)
        }
    };
    Ok(Some((tag.offset, tag.ty, value & tag.mask)))
}

impl Lower<'_, '_> {
    /// Writes variant `variant` of `ty`, an ADT `item` at `args`, with `fields` given by index, over a copy of `base`; returns the value.
    pub fn construct(&mut self, ty: Ty, variant: u32, fields: Vec<(u32, Option<Value>)>, base: Option<Value>) -> Result<Option<Value>> {
        let Kind::Adt(item, args) = *self.cx.types.kind(ty) else { bail!("constructing {:?}, a type that is not an ADT", self.cx.types.kind(ty)) };
        let whole = repr(self.cx, ty)?;
        let Some(slot) = self.slot(whole) else {
            return Ok(None);
        };
        let place = Place::slot(slot, whole);
        if let Some(base) = base {
            self.write(place, base);
        }
        let layout = self.cx.layout(ty)?;
        let slice = self.cx.types.substs(args).to_vec();
        let all = field_types(self.cx, item, &slice)?;
        let types = &all[variant as usize];
        for (index, value) in fields {
            let Some(&field_ty) = types.get(index as usize) else { bail!("variant {variant} of {} has no field {index}", self.cx.name(item)) };
            let field = repr(self.cx, field_ty)?;
            if let Some(value) = value {
                self.write(place.at(field_offset(&layout, variant, index), field), value);
            }
        }
        self.set_tag(place, item, &layout, variant)?;
        Ok(self.read(place))
    }

    fn set_tag(&mut self, place: Place, item: ItemRef, layout: &Layout, variant: u32) -> Result<()> {
        let Some((offset, ty, bits)) = tag_bits(self.cx, item, layout, variant)? else {
            return Ok(());
        };
        let value = self.int_const(ty, bits);
        self.write(place.at(offset, Repr::Scalar(ty)), value);
        Ok(())
    }

    /// Whether the value of ADT `ty` at `place`, which a zero-sized value lacks, holds variant `variant`, or none when it is the only variant the value can hold.
    pub fn is_variant(&mut self, place: Option<Place>, ty: Ty, variant: u32) -> Result<Option<Value>> {
        let Kind::Adt(item, _) = self.cx.types.kind(ty).clone() else { bail!("variant test on a type that is not an ADT") };
        let layout = self.cx.layout(ty)?;
        let Some(tag) = tag_of(&layout)? else {
            return Ok(match layout.variants {
                Variants::Single { index } if index.index() != variant as usize => Some(self.ins().iconst(Type::I8, 0)),
                _ => None,
            });
        };
        let Some(value) = place.and_then(|place| self.read(place.at(tag.offset, Repr::Scalar(tag.ty)))) else { bail!("tag with no value") };
        Ok(Some(match &tag.encoding {
            TagEncoding::Direct => {
                let expected = self.cx.discriminants(item)?[variant as usize] as u128 & tag.mask;
                let expected = self.int_const(tag.ty, expected);
                self.ins().icmp(IntCC::Equal, value, expected)
            }
            TagEncoding::Niche { untagged_variant, niche_variants, niche_start } => {
                let start = self.int_const(tag.ty, *niche_start & tag.mask);
                let relative = self.ins().isub(value, start);
                let first = niche_variants.start.index() as u128;
                let span = self.int_const(tag.ty, niche_variants.last.index() as u128 - first);
                if untagged_variant.index() == variant as usize {
                    self.ins().icmp(IntCC::UnsignedGreaterThan, relative, span)
                } else {
                    let expected = self.int_const(tag.ty, (variant as u128).wrapping_sub(first));
                    self.ins().icmp(IntCC::Equal, relative, expected)
                }
            }
        }))
    }

    /// The discriminant of the fieldless enum `ty` at `place`, in the integer type of its tag.
    pub(super) fn discriminant(&mut self, place: Option<Place>, ty: Ty) -> Result<(Value, Num)> {
        let Kind::Adt(item, _) = self.cx.types.kind(ty).clone() else { bail!("discriminant of a type that is not an enum") };
        let layout = self.cx.layout(ty)?;
        let Variants::Multiple { tag: scalar, .. } = &layout.variants else {
            let value = self.cx.discriminants(item)?.first().copied().unwrap_or(0);
            return Ok((self.int_const(Type::I64, value as u128), Num::Int { ty: Type::I64, signed: true }));
        };
        let signed = matches!(scalar.primitive(), Primitive::Int(_, true));
        let Some(tag) = tag_of(&layout)? else { bail!("enum with variants has no tag") };
        if !matches!(tag.encoding, TagEncoding::Direct) {
            bail!("casting an enum with fields");
        }
        let Some(value) = place.and_then(|place| self.read(place.at(tag.offset, Repr::Scalar(tag.ty)))) else { bail!("enum with no value") };
        Ok((value, Num::Int { ty: tag.ty, signed }))
    }

    /// The declared discriminant of the enum `ty` at `place`, in its discriminant integer type, whatever its tag encoding.
    pub(super) fn discriminant_value(&mut self, place: Place, ty: Ty) -> Result<Value> {
        let Kind::Adt(item, _) = self.cx.types.kind(ty).clone() else { bail!("discriminant_value of a type that is not an enum") };
        let int = self.cx.types.prim(self.cx.discriminant_primitive(item));
        let Repr::Scalar(to) = repr(self.cx, int)? else { bail!("a discriminant that is not a scalar") };
        let values = self.cx.discriminants(item)?;
        let Some((&last, rest)) = values.split_last() else { bail!("discriminant_value of an enum with no variants") };
        let mut found = self.int_const(to, last as u128);
        for (variant, &value) in rest.iter().enumerate() {
            let Some(test) = self.is_variant(Some(place), ty, variant as u32)? else {
                continue;
            };
            let value = self.int_const(to, value as u128);
            found = self.ins().select(test, value, found);
        }
        Ok(found)
    }

    /// The place of field `index` of variant `variant` inside the ADT `ty` at `place`.
    pub fn variant_field(&mut self, place: Place, ty: Ty, (variant, index): (u32, u32)) -> Result<Place> {
        let Kind::Adt(item, args) = *self.cx.types.kind(ty) else { bail!("field of a type that is not an ADT") };
        let layout = self.cx.layout(ty)?;
        let slice = self.cx.types.substs(args).to_vec();
        let fields = field_types(self.cx, item, &slice)?;
        let Some(&field) = fields.get(variant as usize).and_then(|fields| fields.get(index as usize)) else { bail!("{} has no field {index} in variant {variant}", self.cx.name(item)) };
        if layout.is_unsized() {
            return self.unsized_field(place, field_offset(&layout, variant, index), field);
        }
        Ok(place.at(field_offset(&layout, variant, index), repr(self.cx, field)?))
    }

    fn unsized_field(&mut self, wide: Place, offset: i32, field: Ty) -> Result<Place> {
        let Some(pair) = self.read(wide) else { bail!("an unsized value with no pointer") };
        let data = self.ins().load(Type::I64, pair, 0);
        if !self.cx.is_unsized(field)? {
            let at = self.ins().iadd_imm(data, i64::from(offset));
            return Ok(Place { base: Base::Addr(at), offset: 0, repr: repr(self.cx, field)? });
        }
        let metadata = self.ins().load(Type::I64, pair, 8);
        let tail = self.cx.tail(field)?;
        let offset = if matches!(self.cx.types.kind(tail), Kind::Dyn(..)) {
            let align = self.ins().load(Type::I64, metadata, ALIGN_SLOT as i32 * 8);
            let start = self.ins().iconst(Type::I64, i64::from(offset));
            self.align_up(start, align)
        } else {
            self.ins().iconst(Type::I64, i64::from(offset))
        };
        let at = self.ins().iadd(data, offset);
        Ok(Place { base: Base::Addr(self.wide(at, metadata)), offset: 0, repr: super::ops::WIDE })
    }

    /// Lowers the struct literal `expr` with field initializers `fields`.
    pub fn struct_literal(&mut self, expr: ExprId, fields: List, base: Option<ExprId>) -> Result<Option<Value>> {
        let Some(crate::sema::infer::Res::Ctor(_, variant, _)) = self.typed.res.get(&expr).cloned() else { bail!("struct literal has no constructor") };
        let mut values = Vec::new();
        for at in self.body.list(fields).to_vec() {
            let init = self.body.fields[at as usize];
            let Some(&index) = self.typed.inits.get(&init.value) else { bail!("field initializer has no field") };
            values.push((index, self.expr(init.value)?));
        }
        let base = match base {
            Some(base) => self.expr(base)?,
            None => None,
        };
        self.construct(self.ty_of(expr), variant, values, base)
    }

    /// Lowers the body of a constructor function: variant `variant` of the return type, built from the parameters.
    pub fn ctor_body(&mut self, variant: u32) -> Result<Option<Value>> {
        let Some(ret) = self.typed.ret else { bail!("a constructor function with no return type") };
        let mut values = Vec::new();
        for (index, local) in self.typed.params.clone().into_iter().enumerate() {
            values.push((index as u32, self.locals[local as usize].and_then(|place| self.read(place))));
            self.moved_out(super::moves::Root::Local(local), &[])?;
        }
        self.construct(ret, variant, values, None)
    }

    /// Lowers a call of the tuple constructor of variant `variant` with `args`.
    pub fn ctor_call(&mut self, expr: ExprId, variant: u32, args: List) -> Result<Option<Value>> {
        let mut values = Vec::new();
        for (index, at) in self.body.list(args).to_vec().into_iter().enumerate() {
            values.push((index as u32, self.expr(ExprId(at))?));
        }
        self.construct(self.ty_of(expr), variant, values, None)
    }
}

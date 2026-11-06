use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{InstBuilder, Type, Value, types};
use hir_def::{AdtId, VariantId};
use hir_ty::layout::{Layout, RustcEnumVariantIdx};
use hir_ty::next_solver::TyKind;
use rustc_abi::{Primitive, Scalar as AbiScalar, TagEncoding, Variants};
use rustc_type_ir::inherent::IntoKind;

use super::{Lower, Place};

fn int_type(bytes: u64) -> Result<Type, String> {
    match bytes {
        1 => Ok(types::I8),
        2 => Ok(types::I16),
        4 => Ok(types::I32),
        8 => Ok(types::I64),
        16 => Ok(types::I128),
        other => Err(format!("{other}-byte tag")),
    }
}

pub(super) fn tag_bytes(tag: &AbiScalar) -> Result<u64, String> {
    match tag.primitive() {
        Primitive::Int(integer, _) => Ok(integer.size().bytes()),
        Primitive::Pointer(_) => Ok(8),
        other => Err(format!("tag {other:?}")),
    }
}

impl<'db> Lower<'_, '_, 'db> {
    pub(super) fn variant_fields<'l>(&self, layout: &'l Layout, variant: Option<VariantId>) -> Result<&'l Layout, String> {
        match &layout.variants {
            Variants::Single { .. } | Variants::Empty => Ok(layout),
            Variants::Multiple { variants, .. } => match variant {
                Some(VariantId::EnumVariantId(id)) => Ok(&variants[RustcEnumVariantIdx(id.index(self.db))]),
                _ => Err("field of an enum without a variant".to_owned()),
            },
        }
    }

    pub(super) fn write_variant(
        &mut self,
        dest: Place<'db>,
        variant: VariantId,
        fields: &[Place<'db>],
    ) -> Result<(), String> {
        let layout = self.layout(dest.ty.ty)?;
        let shape = self.variant_fields(&layout, Some(variant))?;
        for (index, field) in fields.iter().enumerate() {
            let size = self.layout(field.ty.ty)?.size.bytes();
            let to = self.builder.ins().iadd_imm(dest.addr, shape.fields.offset(index).bytes() as i64);
            self.copy(to, field.addr, size);
        }
        let VariantId::EnumVariantId(id) = variant else {
            return Ok(());
        };
        let Variants::Multiple { tag, tag_encoding, variants, .. } = &layout.variants else {
            return Ok(());
        };
        let bytes = tag_bytes(tag)?;
        let offset = layout.fields.offset(0).bytes() as i64;
        let index = RustcEnumVariantIdx(id.index(self.db));
        let value = match tag_encoding {
            TagEncoding::Direct => self.db.const_eval_discriminant(id).map_err(|e| format!("discriminant: {e:?}"))?,
            TagEncoding::Niche { untagged_variant, niche_start, .. } => {
                if index == *untagged_variant {
                    return Ok(());
                }
                let position = variants
                    .iter_enumerated()
                    .map(|(it, _)| it)
                    .filter(|it| it != untagged_variant)
                    .position(|it| it == index)
                    .ok_or("niche variant")?;
                position as i128 + *niche_start as i128
            }
        };
        let tag_type = int_type(bytes)?;
        let tag_value = self.int_const(tag_type, value);
        let to = self.builder.ins().iadd_imm(dest.addr, offset);
        self.store(tag_value, to, 0);
        Ok(())
    }

    pub(super) fn discriminant(&mut self, dest: Place<'db>, place: Place<'db>) -> Result<(), String> {
        let scalar = self.scalar(dest.ty.ty)?;
        let constant = |lower: &mut Self, value: i128| lower.int_const(scalar.ty, value);
        let TyKind::Adt(adt, _) = place.ty.ty.kind() else {
            let zero = constant(self, 0);
            self.store(zero, dest.addr, 0);
            return Ok(());
        };
        let AdtId::EnumId(enum_id) = adt.def_id() else {
            let zero = constant(self, 0);
            self.store(zero, dest.addr, 0);
            return Ok(());
        };
        let discriminant_of = |lower: &Self, index: usize| {
            let id = enum_id.enum_variants(lower.db).variants[index].0;
            lower.db.const_eval_discriminant(id).map_err(|e| format!("discriminant: {e:?}"))
        };
        let layout = self.layout(place.ty.ty)?;
        let value = match &layout.variants {
            Variants::Empty => {
                self.trap();
                let next = self.builder.create_block();
                self.builder.switch_to_block(next);
                return Ok(());
            }
            Variants::Single { index } => {
                let value = discriminant_of(self, index.0)?;
                constant(self, value)
            }
            Variants::Multiple { tag, tag_encoding, variants, .. } => {
                let bytes = tag_bytes(tag)?;
                let tag_type = int_type(bytes)?;
                let offset = layout.fields.offset(0).bytes() as i32;
                let raw = self.load(tag_type, place.addr, offset);
                match tag_encoding {
                    TagEncoding::Direct => self.fit(raw, tag_type, scalar.ty, tag.is_signed()),
                    TagEncoding::Niche { untagged_variant, niche_start, .. } => {
                        let start = self.int_const(tag_type, *niche_start as i128);
                        let relative = self.builder.ins().isub(raw, start);
                        let fallback = discriminant_of(self, untagged_variant.0)?;
                        let mut result = constant(self, fallback);
                        let others: Vec<usize> = variants
                            .iter_enumerated()
                            .map(|(it, _)| it)
                            .filter(|it| it != untagged_variant)
                            .map(|it| it.0)
                            .collect();
                        for (position, index) in others.into_iter().enumerate() {
                            let position = self.int_const(tag_type, position as i128);
                            let hit = self.builder.ins().icmp(IntCC::Equal, relative, position);
                            let value = discriminant_of(self, index)?;
                            let value = constant(self, value);
                            result = self.builder.ins().select(hit, value, result);
                        }
                        result
                    }
                }
            }
        };
        self.store(value, dest.addr, 0);
        Ok(())
    }

    pub(super) fn int_const(&mut self, ty: Type, value: i128) -> Value {
        if ty == types::I128 {
            let low = self.builder.ins().iconst(types::I64, value as i64);
            let high = self.builder.ins().iconst(types::I64, (value >> 64) as i64);
            return self.builder.ins().iconcat(low, high);
        }
        let mask = if ty.bytes() == 8 { -1_i128 } else { (1_i128 << ty.bits()) - 1 };
        self.builder.ins().iconst(ty, (value & mask) as u64 as i64)
    }

    fn fit(&mut self, value: Value, from: Type, to: Type, signed: bool) -> Value {
        let ins = self.builder.ins();
        if to.bits() > from.bits() {
            if signed { ins.sextend(to, value) } else { ins.uextend(to, value) }
        } else if to.bits() < from.bits() {
            ins.ireduce(to, value)
        } else {
            value
        }
    }
}


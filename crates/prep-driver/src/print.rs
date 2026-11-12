use rustc_abi::{BackendRepr, FieldsShape, Primitive, Scalar, TagEncoding, Variants};
use rustc_index::Idx;

/// What a layout walk prints in place of a trait object or function pointer type, whose code no layout describes.
pub const INDIRECT: &str = "indirect";

/// One layout as text: size, alignment, passing form, the field placement `fields` already printed, and variants with their tag encoding.
pub fn layout<F: Idx, V: Idx>((size, align): (u64, u64), repr: &BackendRepr, fields: &str, variants: &Variants<F, V>) -> String {
    format!("{size}:{align}:{}:{fields}:{}", backend(repr), self::variants(variants))
}

/// A field placement as text: primitive, union, array or offsets.
pub fn fields<F: Idx>(shape: &FieldsShape<F>) -> String {
    match shape {
        FieldsShape::Primitive => "P".to_owned(),
        FieldsShape::Union(count) => format!("U{count}"),
        FieldsShape::Array { stride, count } => format!("A{}x{count}", stride.bytes()),
        FieldsShape::Arbitrary { offsets, .. } => format!("O{}", joined(offsets.iter())),
    }
}

fn backend(repr: &BackendRepr) -> String {
    match repr {
        BackendRepr::Scalar(one) => format!("s{}", scalar(one)),
        BackendRepr::ScalarPair { a, b, .. } => format!("p{},{}", scalar(a), scalar(b)),
        BackendRepr::SimdVector { .. } | BackendRepr::SimdScalableVector { .. } => "v".to_owned(),
        BackendRepr::Memory { sized } => format!("m{}", u8::from(*sized)),
    }
}

fn scalar(scalar: &Scalar) -> String {
    match scalar {
        Scalar::Initialized { value, valid_range } => format!("{}[{}..{}]", primitive(value), valid_range.start, valid_range.end),
        Scalar::Union { value } => format!("{}u", primitive(value)),
    }
}

fn primitive(value: &Primitive) -> String {
    match value {
        Primitive::Int(int, signed) => format!("{}{}", if *signed { "i" } else { "u" }, int.size().bits()),
        Primitive::Float(float) => format!("f{}", float.size().bits()),
        Primitive::Pointer(_) => "p".to_owned(),
    }
}

fn joined<'a>(offsets: impl Iterator<Item = &'a rustc_abi::Size>) -> String {
    offsets.map(|offset| offset.bytes().to_string()).collect::<Vec<_>>().join(",")
}

fn variants<F: Idx, V: Idx>(variants: &Variants<F, V>) -> String {
    match variants {
        Variants::Empty => "E".to_owned(),
        Variants::Single { index } => format!("S{}", index.index()),
        Variants::Multiple { tag, tag_encoding, tag_field, variants } => {
            let each: Vec<String> = variants.iter().map(|variant| joined(variant.field_offsets.iter())).collect();
            format!("M{}@{}{}|{}", scalar(tag), tag_field.index(), encoding(tag_encoding), each.join("|"))
        }
    }
}

fn encoding<V: Idx>(encoding: &TagEncoding<V>) -> String {
    match encoding {
        TagEncoding::Direct => "d".to_owned(),
        TagEncoding::Niche { untagged_variant, niche_variants, niche_start } => {
            format!("n{}/{}-{}/{niche_start}", untagged_variant.index(), niche_variants.start.index(), niche_variants.last.index())
        }
    }
}

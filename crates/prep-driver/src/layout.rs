use rustc_abi::{BackendRepr, FieldIdx, FieldsShape, LayoutData, Primitive, Scalar, TagEncoding, VariantIdx, Variants};
use rustc_data_structures::fx::FxHashSet;
use rustc_middle::ty::{self, Ty, TyCtxt, TypingEnv};
use std::collections::BTreeSet;

/// The most types one instance's arguments and return may reach before the driver stops listing their layouts.
const REACHED_TYPES: usize = 256;

/// The layout of every type that the values of `roots` hold or point to, each printed in the form Haste prints its own layouts in, or none when one cannot be printed or there are too many.
pub fn reached<'tcx>(tcx: TyCtxt<'tcx>, roots: &[Ty<'tcx>]) -> Option<Vec<String>> {
    let env = TypingEnv::fully_monomorphized();
    let (mut seen, mut stack, mut printed) = (FxHashSet::default(), roots.to_vec(), BTreeSet::new());
    while let Some(ty) = stack.pop() {
        if !seen.insert(ty) {
            continue;
        }
        if seen.len() > REACHED_TYPES {
            return None;
        }
        let layout = tcx.layout_of(env.as_query_input(ty)).ok()?;
        printed.insert(print(&layout.layout.0.0));
        stack.extend(inner(tcx, env, ty)?);
    }
    Some(printed.into_iter().collect())
}

fn inner<'tcx>(tcx: TyCtxt<'tcx>, env: TypingEnv<'tcx>, ty: Ty<'tcx>) -> Option<Vec<Ty<'tcx>>> {
    Some(match *ty.kind() {
        ty::Bool | ty::Char | ty::Int(_) | ty::Uint(_) | ty::Float(_) | ty::Never => Vec::new(),
        ty::Ref(_, inner, _) | ty::RawPtr(inner, _) | ty::Slice(inner) | ty::Array(inner, _) | ty::Pat(inner, _) => vec![inner],
        ty::Tuple(items) => items.to_vec(),
        ty::Adt(def, args) => def.all_fields().map(|field| tcx.normalize_erasing_regions(env, field.ty(tcx, args))).collect(),
        _ => return None,
    })
}

/// One layout as text: size, alignment, passing form, field placement and variants with their tag encoding.
pub fn print(layout: &LayoutData<FieldIdx, VariantIdx>) -> String {
    format!("{}:{}:{}:{}:{}", layout.size.bytes(), layout.align.abi.bytes(), repr(&layout.backend_repr), fields(&layout.fields), variants(&layout.variants))
}

fn repr(repr: &BackendRepr) -> String {
    match repr {
        BackendRepr::Scalar(one) => format!("s{}", scalar(one)),
        BackendRepr::ScalarPair { a, b, .. } => format!("p{},{}", scalar(a), scalar(b)),
        BackendRepr::SimdVector { .. } | BackendRepr::SimdScalableVector { .. } => "v".to_string(),
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
        Primitive::Pointer(_) => "p".to_string(),
    }
}

fn fields(shape: &FieldsShape<FieldIdx>) -> String {
    match shape {
        FieldsShape::Primitive => "P".to_string(),
        FieldsShape::Union(count) => format!("U{count}"),
        FieldsShape::Array { stride, count } => format!("A{}x{count}", stride.bytes()),
        FieldsShape::Arbitrary { offsets, .. } => format!("O{}", offsets.iter().map(|offset| offset.bytes().to_string()).collect::<Vec<_>>().join(",")),
    }
}

fn variants(variants: &Variants<FieldIdx, VariantIdx>) -> String {
    match variants {
        Variants::Empty => "E".to_string(),
        Variants::Single { index } => format!("S{}", index.as_usize()),
        Variants::Multiple { tag, tag_encoding, tag_field, variants } => {
            let each: Vec<String> = variants.iter().map(|variant| variant.field_offsets.iter().map(|offset| offset.bytes().to_string()).collect::<Vec<_>>().join(",")).collect();
            format!("M{}@{}{}|{}", scalar(tag), tag_field.as_usize(), encoding(tag_encoding), each.join("|"))
        }
    }
}

fn encoding(encoding: &TagEncoding<VariantIdx>) -> String {
    match encoding {
        TagEncoding::Direct => "d".to_string(),
        TagEncoding::Niche { untagged_variant, niche_variants, niche_start } => {
            format!("n{}/{}-{}/{niche_start}", untagged_variant.as_usize(), niche_variants.start.as_usize(), niche_variants.last.as_usize())
        }
    }
}

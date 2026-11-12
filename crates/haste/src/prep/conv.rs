use haste_codegen::Type;
use rustc_abi::{BackendRepr, FieldsShape, Primitive, Scalar, TagEncoding, Variants};
use rustc_hash::FxHashSet as HashSet;

use crate::lower::{Parts, Repr, repr, returned, unwrapped};
use crate::sema::Context;
use crate::sema::infer::field_types;
use crate::sema::layout::Layout;
use crate::sema::ty::{Kind, Ty};

use super::key::Miss;
use super::record::{Abi, Arg, Class, Conv, Pass, Reg};

/// How Haste passes one argument or return: in no register, in one, in two at an offset, or by address.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Haste {
    Nothing,
    One(Type),
    Two(Type, Type, i32),
    Address,
}

/// Whether rustc passes each argument and the return of `abi` as Haste passes `params` and `ret`, each at the size and alignment of Haste's layout.
pub(super) fn agrees(cx: &mut Context<'_>, (params, ret): (&[Ty], Option<Ty>), abi: &Abi) -> bool {
    if abi.conv != Conv::Rust || abi.variadic || params.len() != abi.args.len() {
        return false;
    }
    let returns = match ret {
        Some(ty) if matches!(cx.types.kind(ty), Kind::Never) => return false,
        Some(ty) => sized(cx, ty, &abi.ret) && result(cx, ty).is_some_and(|haste| same(haste, &abi.ret.pass)),
        None => abi.ret.pass == Pass::Ignore,
    };
    returns && params.iter().zip(&abi.args).all(|(&ty, arg)| sized(cx, ty, arg) && argument(cx, ty).is_some_and(|haste| same(haste, &arg.pass)))
}

fn sized(cx: &mut Context<'_>, ty: Ty, arg: &Arg) -> bool {
    cx.layout(ty).is_ok_and(|layout| layout.size.bytes() == arg.size && layout.align.bytes() == arg.align)
}

fn argument(cx: &mut Context<'_>, ty: Ty) -> Option<Haste> {
    Some(match repr(cx, ty).ok()? {
        Repr::Zst => Haste::Nothing,
        Repr::Scalar(machine) => Haste::One(machine),
        Repr::Mem { .. } => unwrapped(cx, ty).ok()?.map_or(Haste::Address, Haste::One),
    })
}

fn result(cx: &mut Context<'_>, ty: Ty) -> Option<Haste> {
    Some(match repr(cx, ty).ok()? {
        Repr::Zst => Haste::Nothing,
        Repr::Scalar(machine) => Haste::One(machine),
        Repr::Mem { .. } => match returned(cx, ty).ok()? {
            Some(Parts::One(machine)) => Haste::One(machine),
            Some(Parts::Pair(first, second, offset)) => Haste::Two(first, second, offset),
            Some(Parts::Bytes) | None => Haste::Address,
        },
    })
}

fn same(haste: Haste, pass: &Pass) -> bool {
    match (haste, pass) {
        (Haste::Nothing, Pass::Ignore) => true,
        (Haste::One(machine), Pass::Direct { regs }) => matches!(regs.as_slice(), [Reg { class, offset: 0 }] if class.machine() == Some(machine)),
        (Haste::Two(first, second, at), Pass::Pair { regs }) => match regs.as_slice() {
            [one, two] => one.offset == 0 && one.class.machine() == Some(first) && two.class.machine() == Some(second) && i32::try_from(two.offset) == Ok(at),
            _ => false,
        },
        (Haste::Address, Pass::Indirect { on_stack, dynamic }) => !on_stack && !dynamic,
        _ => false,
    }
}

impl Class {
    /// The machine type a value of this register class travels in, when Haste has one.
    fn machine(self) -> Option<Type> {
        Some(match self {
            Class::I8 | Class::U8 => Type::I8,
            Class::I16 | Class::U16 => Type::I16,
            Class::I32 | Class::U32 => Type::I32,
            Class::I64 | Class::U64 | Class::Ptr => Type::I64,
            Class::I128 | Class::U128 => Type::I128,
            Class::F32 => Type::F32,
            Class::F64 => Type::F64,
            Class::Vector | Class::Other => return None,
        })
    }
}

/// The layouts of `types` and of every type their values hold or point to, in the order a depth-first walk first meets them, printed as the driver prints rustc's.
pub(super) fn layouts(cx: &mut Context<'_>, types: &[Ty]) -> Result<Vec<String>, Miss> {
    let (mut seen, mut stack, mut printed) = (HashSet::default(), types.to_vec(), Vec::new());
    while let Some(ty) = stack.pop() {
        if !seen.insert(ty) {
            continue;
        }
        printed.push(print(&*cx.layout(ty).map_err(|_| Miss::Unnamed)?));
        stack.extend(inner(cx, ty)?);
    }
    Ok(printed)
}

/// The types directly inside a value of `ty` in the order the driver walks rustc's: a pointee or element, the items of a tuple, every field of every variant, or a closure's captures.
fn inner(cx: &mut Context<'_>, ty: Ty) -> Result<Vec<Ty>, Miss> {
    Ok(match cx.types.kind(ty).clone() {
        Kind::Prim(_) | Kind::Never | Kind::FnDef(..) => Vec::new(),
        Kind::Ref(_, inner) | Kind::Ptr(_, inner) | Kind::Slice(inner) | Kind::Pat(inner, ..) | Kind::Array(inner, _) => vec![inner],
        Kind::Tuple(items) => cx.types.substs(items).to_vec(),
        Kind::Adt(item, args) => {
            let args = cx.types.substs(args).to_vec();
            field_types(cx, item, &args).map_err(|_| Miss::Unnamed)?.iter().flatten().copied().collect()
        }
        Kind::Closure(..) => match cx.closure_env(ty).map(|env| cx.types.kind(env).clone()) {
            Some(Kind::Tuple(items)) => cx.types.substs(items).to_vec(),
            _ => return Err(Miss::Unnamed),
        },
        Kind::FnPtr(..) | Kind::Dyn(..) => return Err(Miss::Indirect),
        _ => return Err(Miss::Unnamed),
    })
}

fn print(layout: &Layout) -> String {
    format!("{}:{}:{}:{}:{}", layout.size.bytes(), layout.align.bytes(), backend(&layout.backend_repr), fields(&layout.fields), variants(&layout.variants))
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

fn fields<F: rustc_index::Idx>(shape: &FieldsShape<F>) -> String {
    match shape {
        FieldsShape::Primitive => "P".to_owned(),
        FieldsShape::Union(count) => format!("U{count}"),
        FieldsShape::Array { stride, count } => format!("A{}x{count}", stride.bytes()),
        FieldsShape::Arbitrary { offsets, .. } => format!("O{}", joined(offsets.iter())),
    }
}

fn variants<F: rustc_index::Idx, V: rustc_index::Idx>(variants: &Variants<F, V>) -> String {
    match variants {
        Variants::Empty => "E".to_owned(),
        Variants::Single { index } => format!("S{}", index.index()),
        Variants::Multiple { tag, tag_encoding, tag_field, variants } => {
            let each: Vec<String> = variants.iter().map(|variant| joined(variant.field_offsets.iter())).collect();
            format!("M{}@{}{}|{}", scalar(tag), tag_field.index(), encoding(tag_encoding), each.join("|"))
        }
    }
}

fn encoding<V: rustc_index::Idx>(encoding: &TagEncoding<V>) -> String {
    match encoding {
        TagEncoding::Direct => "d".to_owned(),
        TagEncoding::Niche { untagged_variant, niche_variants, niche_start } => {
            format!("n{}/{}-{}/{niche_start}", untagged_variant.index(), niche_variants.start.index(), niche_variants.last.index())
        }
    }
}

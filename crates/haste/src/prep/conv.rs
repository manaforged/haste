use haste_codegen::Type;
use rustc_abi::{FieldsShape, Variants};
use rustc_hash::FxHashSet as HashSet;

use crate::lower::{Parts, Repr, repr, returned, unwrapped};
use crate::sema::Context;
use crate::sema::infer::field_types;
use crate::sema::layout::Layout;
use crate::sema::ty::{Kind, Ty};

use super::bridge::Carry;
use super::key::Miss;
use super::print;
use super::record::{Abi, Arg, Class, Conv, Pass, Reg};

/// How the driver prints the fields of a struct or pattern type whose one value lies at offset zero.
const NEWTYPE_FIELDS: &str = "O0";

/// How Haste passes one argument or return: in no register, in one, in two at an offset, or by address.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Haste {
    Nothing,
    One(Type),
    Two(Type, Type, i32),
    Address,
}

/// How each argument reaches rustc's code for `abi` when Haste passes `params` and `ret`, each at the size and alignment of Haste's layout: none when rustc passes an argument or the return otherwise than Haste can carry it.
pub(super) fn carries(cx: &mut Context<'_>, (params, ret): (&[Ty], Option<Ty>), abi: &Abi) -> Option<Vec<Carry>> {
    carries_after(cx, (params, ret), abi, 0)
}

/// `carries` for a call whose first `thin` arguments are receivers that both sides pass as one thin pointer.
fn carries_after(cx: &mut Context<'_>, (params, ret): (&[Ty], Option<Ty>), abi: &Abi, thin: usize) -> Option<Vec<Carry>> {
    if abi.conv != Conv::Rust || abi.variadic || params.len() != abi.args.len() || abi.args.iter().take(thin).any(|arg| !thin_pointer(arg)) {
        return None;
    }
    let returns = match ret {
        Some(ty) if matches!(cx.types.kind(ty), Kind::Never) => return None,
        Some(ty) => sized(cx, ty, &abi.ret) && result(cx, ty).is_some_and(|haste| same(haste, &abi.ret.pass)),
        None => abi.ret.pass == Pass::Ignore,
    };
    returns.then_some(())?;
    let rest = params.iter().zip(&abi.args).skip(thin).map(|(&ty, arg)| carry(cx, ty, arg));
    std::iter::repeat_n(Some(Carry::Same), thin.min(params.len())).chain(rest).collect()
}

fn thin_pointer(arg: &Arg) -> bool {
    arg.size == 8 && same(Haste::One(Type::I64), &arg.pass)
}

/// How each argument of a call through a vtable slot (`thin` is one, for its receiver) or a function pointer (`thin` is zero) that Haste's code makes with `params` and `ret` reaches rustc's convention `abi`, with the parameter types as rustc counts them: a last tuple of no items, which rustc's call methods take apart, dropped.
pub(super) fn spread_carries(cx: &mut Context<'_>, (params, ret): (&[Ty], Option<Ty>), abi: &Abi, thin: usize) -> Option<(Vec<Ty>, Vec<Carry>)> {
    let mut spread = params.to_vec();
    if spread.len() != abi.args.len()
        && let Some(&last) = spread.last()
        && let Kind::Tuple(items) = cx.types.kind(last).clone()
        && cx.types.substs(items).is_empty()
    {
        spread.pop();
    }
    let carries = carries_after(cx, (&spread, ret), abi, thin)?;
    Some((spread, carries))
}

/// How an argument of `ty` reaches rustc's code that takes it as `arg`: as Haste passes it, or split from its address into the two registers of a scalar pair whose parts no signed narrow integer is, since a load widens with zeros.
fn carry(cx: &mut Context<'_>, ty: Ty, arg: &Arg) -> Option<Carry> {
    if !sized(cx, ty, arg) {
        return None;
    }
    let haste = argument(cx, ty)?;
    if same(haste, &arg.pass) {
        return Some(Carry::Same);
    }
    let Pass::Pair { regs } = &arg.pass else { return None };
    match (haste, repr(cx, ty).ok()?) {
        (Haste::Address, Repr::Mem { parts: Parts::Pair(first, second, at), .. }) if same(Haste::Two(first, second, at), &arg.pass) && !regs.iter().any(|reg| matches!(reg.class, Class::I8 | Class::I16)) => Some(Carry::Split(first, second, at)),
        _ => None,
    }
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

/// The layouts of `types` and of every type their values hold or point to, in the order a depth-first walk first meets them, printed as the driver prints rustc's, with the trait object and function pointer types among them.
pub(super) fn layouts(cx: &mut Context<'_>, types: &[Ty]) -> Result<(Vec<String>, Vec<Ty>), Miss> {
    let (mut seen, mut stack, mut printed, mut indirect) = (HashSet::default(), types.to_vec(), Vec::new(), Vec::new());
    while let Some(ty) = stack.pop() {
        if !seen.insert(ty) {
            continue;
        }
        if matches!(cx.types.kind(ty), Kind::FnPtr(..) | Kind::Dyn(..)) {
            printed.push(print::INDIRECT.to_owned());
            indirect.push(ty);
            continue;
        }
        let layout = cx.layout(ty).map_err(|_| Miss::Unnamed)?;
        let fields = inner(cx, ty)?;
        let newtype = match cx.types.kind(ty) {
            Kind::Adt(..) => fields.len() == 1,
            Kind::Pat(..) => true,
            _ => false,
        };
        printed.push(layout_text(&layout, newtype));
        stack.extend(fields);
    }
    Ok((printed, indirect))
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
        _ => return Err(Miss::Unnamed),
    })
}

/// One layout as the driver prints rustc's; a struct of one field or a pattern type, whose one value lies at offset zero and which Haste may lay out as the scalar it holds, prints its fields as rustc places them.
fn layout_text(layout: &Layout, newtype: bool) -> String {
    let at_zero = match &layout.fields {
        FieldsShape::Primitive => true,
        FieldsShape::Arbitrary { offsets, .. } => offsets.iter().next().is_some_and(|offset| offset.bytes() == 0),
        _ => false,
    };
    let fields = if newtype && at_zero && matches!(layout.variants, Variants::Single { .. }) { NEWTYPE_FIELDS.to_owned() } else { print::fields(&layout.fields) };
    print::layout((layout.size.bytes(), layout.align.bytes()), &layout.backend_repr, &fields, &layout.variants)
}

use haste_codegen::Type;
use rustc_abi::{FieldsShape, Variants};
use rustc_hash::FxHashSet as HashSet;

use crate::lower::{Parts, Repr, passed_parts, repr};
use crate::sema::Context;
use crate::sema::infer::field_types;
use crate::sema::layout::Layout;
use crate::sema::ty::{Kind, Ty};

use super::bridge::Carry;
use super::key::{Keys, Miss};
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
        Some(ty) => sized(cx, ty, &abi.ret) && haste_pass(cx, ty).is_some_and(|haste| same(haste, &abi.ret.pass)),
        None => abi.ret.pass == Pass::Ignore,
    };
    returns.then_some(())?;
    let rest = params.iter().zip(&abi.args).skip(thin).map(|(&ty, arg)| carry(cx, ty, arg));
    std::iter::repeat_n(Some(Carry::Same), thin.min(params.len())).chain(rest).collect()
}

fn thin_pointer(arg: &Arg) -> bool {
    arg.size == 8 && same(Haste::One(Type::I64), &arg.pass)
}

/// How each argument of a call through a vtable slot (`thin` is one, for its receiver) or a function pointer (`thin` is zero) that Haste's code makes with `params` and `ret` reaches rustc's convention `abi`, with the parameter types as rustc counts them: a last tuple taken apart as rustc's call methods take it when it has no items and `abi` counts one argument fewer, or when its keys are the ones `abi` passes.
pub(super) fn spread_carries(cx: &mut Context<'_>, keys: &Keys, (params, ret): (&[Ty], Option<Ty>), abi: &Abi, thin: usize) -> Option<(Vec<Ty>, Vec<Carry>)> {
    let spread = match spread(cx, params) {
        Some(spread) if (spread.len() < params.len() && params.len() != abi.args.len()) || keyed(cx, keys, (&spread, ret), abi, thin) => spread,
        _ => params.to_vec(),
    };
    let carries = carries_after(cx, (&spread, ret), abi, thin)?;
    Some((spread, carries))
}

/// The parameters `params` of a function returning `ret` as convention `abi` counts them, as they are or with a last tuple taken apart as rustc's call methods take it, when the keys of the parameters after the first `thin` and of the return are those of the types `abi` passes there.
pub(super) fn counted(cx: &mut Context<'_>, keys: &Keys, (params, ret): (&[Ty], Option<Ty>), abi: &Abi, thin: usize) -> Option<Vec<Ty>> {
    if keyed(cx, keys, (params, ret), abi, thin) {
        return Some(params.to_vec());
    }
    let spread = spread(cx, params)?;
    keyed(cx, keys, (&spread, ret), abi, thin).then_some(spread)
}

/// Whether the parameters `params` after the first `thin` and the return `ret`, `()` when none, have the keys of the types convention `abi` passes there.
fn keyed(cx: &mut Context<'_>, keys: &Keys, (params, ret): (&[Ty], Option<Ty>), abi: &Abi, thin: usize) -> bool {
    let ret = ret.unwrap_or_else(|| cx.types.tuple(&[]));
    let theirs: Vec<Option<&str>> = abi.args.iter().chain([&abi.ret]).skip(thin).map(|arg| arg.key.as_deref()).collect();
    let ours: Vec<Ty> = params.iter().skip(thin).copied().chain([ret]).collect();
    theirs.len() == ours.len() && ours.into_iter().zip(theirs).all(|(ty, theirs)| theirs.is_some_and(|theirs| keys.ty(cx, ty).is_ok_and(|key| key == theirs)))
}

/// `params` with a last tuple taken apart as rustc's call methods take it: a tuple of no items dropped, and a tuple of one item, which Haste lays out and passes as that item, replaced by it; none when the last parameter is no such tuple.
fn spread(cx: &mut Context<'_>, params: &[Ty]) -> Option<Vec<Ty>> {
    let (&last, rest) = params.split_last()?;
    let Kind::Tuple(items) = cx.types.kind(last).clone() else { return None };
    let mut spread = rest.to_vec();
    match *cx.types.substs(items).to_vec() {
        [] => {}
        [item] if same_value(cx, last, item) => spread.push(item),
        _ => return None,
    }
    Some(spread)
}

/// Whether Haste lays out and passes a tuple `tuple` of the one item `item` exactly as `item`: the same size and alignment, the item at offset zero, and one way of passing.
fn same_value(cx: &mut Context<'_>, tuple: Ty, item: Ty) -> bool {
    let (Ok(outer), Ok(inner)) = (cx.layout(tuple), cx.layout(item)) else { return false };
    let at_zero = crate::sema::layout::field_offset(&outer, 0, 0) == 0;
    at_zero && outer.size == inner.size && outer.align.bytes() == inner.align.bytes() && haste_pass(cx, tuple).is_some_and(|haste| haste_pass(cx, item) == Some(haste))
}

/// How an argument of `ty` reaches rustc's code that takes it as `arg`: as Haste passes it, or split from its address into the two registers of a scalar pair whose parts no signed narrow integer is, since a load widens with zeros.
fn carry(cx: &mut Context<'_>, ty: Ty, arg: &Arg) -> Option<Carry> {
    if !sized(cx, ty, arg) {
        return None;
    }
    let haste = haste_pass(cx, ty)?;
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

fn haste_pass(cx: &mut Context<'_>, ty: Ty) -> Option<Haste> {
    Some(match repr(cx, ty).ok()? {
        Repr::Zst => Haste::Nothing,
        Repr::Scalar(machine) => Haste::One(machine),
        Repr::Mem { .. } => match passed_parts(cx, ty).ok()? {
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
        (Haste::Two(first, second, at), Pass::Pair { regs }) if !regs.iter().any(|reg| matches!(reg.class, Class::I8 | Class::I16)) => match regs.as_slice() {
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

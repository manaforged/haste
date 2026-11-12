use crate::body::sig::GenericKind;
use crate::names::ItemRef;

use super::super::Context;
use super::super::substs::Substs;
use super::super::ty::{Kind, Mut, Ty, object_parts};
use super::detached;
use super::select::CALLABLE;

/// The name that `std::any::type_name` gives `ty`, spelled as rustc spells it: each item by its full path, lifetimes as `'_`, and trailing type arguments that equal their defaults left out.
pub fn type_name(cx: &mut Context<'_>, ty: Ty) -> String {
    match cx.types.kind(ty).clone() {
        Kind::Tuple(items) => tuple_name(cx, items),
        Kind::Adt(item, args) => adt_name(cx, item, args),
        Kind::FnDef(item, args) => {
            let args = names(cx, cx.types.substs(args).to_vec());
            format!("{}{}", def_path(cx, item), angled(&args))
        }
        Kind::FnPtr(params, ret) => format!("fn({}){}", names(cx, cx.types.substs(params).to_vec()).join(", "), returns(cx, ret)),
        Kind::Closure(item, ..) => format!("{}::{{{{closure}}}}", def_path(cx, item)),
        Kind::Dyn(of, args, members, autos) => dyn_name(cx, (of, args), &members, &autos),
        _ => shape_name(cx, ty),
    }
}

/// The name of a primitive, reference, pointer, array, slice or const value `ty`.
fn shape_name(cx: &mut Context<'_>, ty: Ty) -> String {
    match cx.types.kind(ty).clone() {
        Kind::Prim(primitive) => format!("{primitive:?}").to_lowercase(),
        Kind::Never => "!".to_owned(),
        Kind::Ref(mutable, inner) => format!("&{}{}", if mutable == Mut::Mut { "mut " } else { "" }, type_name(cx, inner)),
        Kind::Ptr(mutable, inner) => format!("*{} {}", if mutable == Mut::Mut { "mut" } else { "const" }, type_name(cx, inner)),
        Kind::Array(element, len) => format!("[{}; {}]", type_name(cx, element), type_name(cx, len)),
        Kind::Slice(element) => format!("[{}]", type_name(cx, element)),
        Kind::Value(bits) => bits.to_string(),
        _ => cx.type_text(ty, &|ty| ty),
    }
}

/// The full path of `item`: its crate, its modules, and the function around it when a body declares it.
fn def_path(cx: &mut Context<'_>, item: ItemRef) -> String {
    let name = cx.name(item);
    if let Some((outer, _)) = item.outer() {
        return format!("{}::{name}", def_path(cx, outer));
    }
    let module = cx.module_of(item).map_or_else(|_| cx.outline.roots[item.krate as usize].name.clone(), |at| cx.outline.module_path(at));
    format!("{module}::{name}")
}

fn names(cx: &mut Context<'_>, tys: Vec<Ty>) -> Vec<String> {
    tys.into_iter().map(|ty| type_name(cx, ty)).collect()
}

fn angled(parts: &[String]) -> String {
    if parts.is_empty() { String::new() } else { format!("<{}>", parts.join(", ")) }
}

fn tuple_name(cx: &mut Context<'_>, items: Substs) -> String {
    let parts = names(cx, cx.types.substs(items).to_vec());
    match &parts[..] {
        [one] => format!("({one},)"),
        _ => format!("({})", parts.join(", ")),
    }
}

/// ` -> T` for a function returning `ret`, or nothing when it returns `()`.
fn returns(cx: &mut Context<'_>, ret: Ty) -> String {
    match cx.types.kind(ret) {
        Kind::Tuple(items) if items.is_empty() => String::new(),
        _ => format!(" -> {}", type_name(cx, ret)),
    }
}

fn adt_name(cx: &mut Context<'_>, item: ItemRef, args: Substs) -> String {
    let args = cx.types.substs(args).to_vec();
    let lifetimes = cx.adt_def(item).map_or(0, |def| def.generics.params.iter().filter(|param| param.kind == GenericKind::Lifetime).count());
    let shown = shown_args(cx, item, &args);
    let mut parts: Vec<String> = std::iter::repeat_n("'_".to_owned(), lifetimes).collect();
    parts.extend(names(cx, args[..shown].to_vec()));
    format!("{}{}", def_path(cx, item), angled(&parts))
}

/// How many of the type arguments `args` of `item` rustc prints: all but the trailing ones that equal the default their parameter gives at the arguments before them.
fn shown_args(cx: &mut Context<'_>, item: ItemRef, args: &[Ty]) -> usize {
    let mut count = args.len();
    while count > 0 && is_default(cx, item, args, count - 1) {
        count -= 1;
    }
    count
}

fn is_default(cx: &mut Context<'_>, item: ItemRef, args: &[Ty], at: usize) -> bool {
    let has_default = cx.adt_def(item).is_ok_and(|def| def.generics.params.iter().filter(|param| param.kind != GenericKind::Lifetime).nth(at).is_some_and(|param| param.default.is_some()));
    has_default && detached(cx, item, |infer| infer.adt_args(item, args[..at].to_vec(), false)).is_ok_and(|filled| filled.get(at) == Some(&args[at]))
}

fn dyn_name(cx: &mut Context<'_>, (of, args): (ItemRef, Substs), members: &[ItemRef], autos: &[ItemRef]) -> String {
    let all = cx.types.substs(args).to_vec();
    let (trait_args, bound) = object_parts(&all, members);
    let bound: Vec<(ItemRef, Ty)> = bound.collect();
    let mut text = format!("dyn {}", def_path(cx, of));
    if cx.lang(of).is_some_and(|lang| CALLABLE.contains(&lang)) {
        let inputs = match trait_args.first().map(|&tuple| cx.types.kind(tuple).clone()) {
            Some(Kind::Tuple(items)) => names(cx, cx.types.substs(items).to_vec()),
            _ => Vec::new(),
        };
        let output = bound.first().map_or_else(String::new, |&(_, ty)| returns(cx, ty));
        text.push_str(&format!("({}){output}", inputs.join(", ")));
    } else {
        let mut parts = names(cx, trait_args.to_vec());
        for (member, ty) in bound {
            parts.push(format!("{} = {}", cx.name(member), type_name(cx, ty)));
        }
        text.push_str(&angled(&parts));
    }
    for &auto in autos {
        text.push_str(&format!(" + {}", def_path(cx, auto)));
    }
    text
}

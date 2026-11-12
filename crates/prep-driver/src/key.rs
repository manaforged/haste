use std::path::{Path, PathBuf};

use rustc_hir::def_id::DefId;
use rustc_middle::ty::{self, GenericArgKind, GenericArgsRef, Instance, InstanceKind, Ty, TyCtxt};
use rustc_span::Span;
use serde_json::{Value, json};

use crate::form;

/// The kinds `record::kind` gives a closure's by-value call shim and a drop glue.
const CLOSURE_ONCE: &str = "Shim::ClosureOnce";
const DROP_GLUE: &str = "Shim::DropGlue";

/// The key that names `instance` to Haste: a function's definition and type arguments, a closure's place in its root with the root's arguments, or the type a drop glue drops; none when a part of it has no key.
pub fn instance<'tcx>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>) -> Option<String> {
    if let InstanceKind::Item(def_id) = instance.def {
        if tcx.is_closure_like(def_id) {
            return closure(tcx, def_id, instance.args);
        }
        return Some(form::named(&def(tcx, def_id)?, &args(tcx, instance.args)?));
    }
    let first = instance.args.types().next()?;
    match crate::record::kind(instance.def).as_str() {
        CLOSURE_ONCE => match *first.kind() {
            ty::Closure(def_id, args) => closure(tcx, def_id, args),
            _ => None,
        },
        DROP_GLUE => Some(format!("drop {}", ty(tcx, first)?)),
        _ => None,
    }
}

/// Where definition `def_id` is: its crate, and the file and byte offset of its name, or of its whole span when it has no name, unless a macro expansion wrote it.
pub fn at(tcx: TyCtxt<'_>, def_id: DefId) -> Value {
    let place = place(tcx, tcx.def_ident_span(def_id).unwrap_or_else(|| tcx.def_span(def_id)));
    json!({ "crate": tcx.crate_name(def_id.krate).as_str(), "file": place.as_ref().map(|(file, _)| file), "lo": place.map(|(_, lo)| lo) })
}

/// `ty` as a key, or none when it holds a type with no key.
pub fn ty<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Option<String> {
    match *ty.kind() {
        ty::Bool | ty::Char | ty::Str | ty::Never | ty::Int(_) | ty::Uint(_) | ty::Float(_) => scalar(ty),
        ty::Ref(..) | ty::RawPtr(..) | ty::Array(..) | ty::Slice(_) | ty::Tuple(_) => built(tcx, ty),
        _ => named(tcx, ty),
    }
}

/// The key of a primitive type.
fn scalar(ty: Ty<'_>) -> Option<String> {
    Some(match *ty.kind() {
        ty::Bool => "bool".to_owned(),
        ty::Char => "char".to_owned(),
        ty::Str => "str".to_owned(),
        ty::Never => form::NEVER.to_owned(),
        ty::Int(int) => int.name_str().to_owned(),
        ty::Uint(uint) => uint.name_str().to_owned(),
        ty::Float(float) => float.name_str().to_owned(),
        _ => return None,
    })
}

/// The key of a reference, raw pointer, array, slice or tuple type, built from the keys of its parts.
fn built<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Option<String> {
    Some(match *ty.kind() {
        ty::Ref(_, inner, mutability) => form::reference(mutability.is_mut(), &self::ty(tcx, inner)?),
        ty::RawPtr(inner, mutability) => form::raw(mutability.is_mut(), &self::ty(tcx, inner)?),
        ty::Array(inner, len) => form::array(&self::ty(tcx, inner)?, &value(len)?),
        ty::Slice(inner) => form::slice(&self::ty(tcx, inner)?),
        ty::Tuple(items) => form::tuple(&types(tcx, items.iter())?),
        _ => return None,
    })
}

/// The key of a type a definition names: a struct, enum or union, a function item, a closure, a function pointer or a trait object.
fn named<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Option<String> {
    Some(match *ty.kind() {
        ty::Adt(adt, args) => form::named(&def(tcx, adt.did())?, &self::args(tcx, args)?),
        ty::FnDef(def_id, args) => form::fn_def(&def(tcx, def_id)?, &self::args(tcx, args.skip_binder())?),
        ty::Closure(def_id, args) => closure(tcx, def_id, args)?,
        ty::FnPtr(..) => pointer(tcx, ty)?,
        ty::Dynamic(predicates, _) => object(tcx, predicates)?,
        _ => return None,
    })
}

fn types<'tcx>(tcx: TyCtxt<'tcx>, items: impl Iterator<Item = Ty<'tcx>>) -> Option<Vec<String>> {
    items.map(|item| self::ty(tcx, item)).collect()
}

/// The key of a function pointer type: its parameter and return types.
fn pointer<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Option<String> {
    let sig = tcx.instantiate_bound_regions_with_erased(ty.fn_sig(tcx));
    Some(form::fn_ptr(&types(tcx, sig.inputs().iter().copied())?, &self::ty(tcx, sig.output())?))
}

/// The key of a trait object type: its principal trait with the trait's arguments, and each associated type it fixes by name.
fn object<'tcx>(tcx: TyCtxt<'tcx>, predicates: &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>) -> Option<String> {
    let principal = predicates.principal()?.skip_binder();
    let mut fixed = Vec::new();
    for projection in predicates.projection_bounds() {
        let projection = projection.skip_binder();
        fixed.push((tcx.item_name(projection.def_id).to_string(), self::ty(tcx, projection.term.as_type()?)?));
    }
    Some(form::object(&def(tcx, principal.def_id)?, &args(tcx, principal.args)?, &fixed))
}

fn args<'tcx>(tcx: TyCtxt<'tcx>, args: &[ty::GenericArg<'tcx>]) -> Option<Vec<String>> {
    let mut keys = Vec::new();
    for arg in args {
        match arg.kind() {
            GenericArgKind::Lifetime(_) => {}
            GenericArgKind::Type(one) => keys.push(ty(tcx, one)?),
            GenericArgKind::Const(one) => keys.push(value(one)?),
        }
    }
    Some(keys)
}

fn value(constant: ty::Const<'_>) -> Option<String> {
    Some(constant.try_to_value()?.valtree.try_to_leaf()?.to_bits_unchecked().to_string())
}

fn closure<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId, closure_args: GenericArgsRef<'tcx>) -> Option<String> {
    let root = tcx.typeck_root_def_id(def_id);
    let order = crate::record::order(tcx, def_id)?;
    let parent = closure_args.iter().take(tcx.generics_of(def_id).parent_count).collect::<Vec<_>>();
    Some(form::closure(&def(tcx, root)?, order, &args(tcx, &parent)?))
}

/// The key of definition `def_id`: its crate, file, the byte offset of its name and the name, for a named definition outside any macro expansion.
pub fn def(tcx: TyCtxt<'_>, def_id: DefId) -> Option<String> {
    let (file, lo) = place(tcx, tcx.def_ident_span(def_id)?)?;
    let krate = crate::source::crate_key(tcx.crate_name(def_id.krate).as_str(), crate::link::sysroot(tcx, def_id.krate));
    Some(format!("{krate}:{file}:{lo}:{}", tcx.item_name(def_id)))
}

fn place(tcx: TyCtxt<'_>, span: Span) -> Option<(String, u32)> {
    if span.from_expansion() {
        return None;
    }
    let map = tcx.sess.source_map();
    let found = map.lookup_byte_offset(span.lo());
    let shown = map.span_to_diagnostic_string(span);
    let name = shown.rsplitn(2, ": ").last()?.rsplitn(3, ':').last()?;
    Some((crate::source::normalize(&absolute(Path::new(name)))?, found.pos.0))
}

fn absolute(path: &Path) -> PathBuf {
    if path.is_absolute() {
        return path.to_path_buf();
    }
    std::env::current_dir().map_or_else(|_| path.to_path_buf(), |dir| dir.join(path))
}

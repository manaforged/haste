use std::path::{Component, Path, PathBuf};

use rustc_hir::def_id::DefId;
use rustc_middle::ty::{self, GenericArgKind, GenericArgsRef, Instance, InstanceKind, Ty, TyCtxt};
use rustc_span::Span;
use serde_json::{Value, json};

/// The directory whose parent starts a source path in a key, so one crate's file prints alike wherever its package lies.
const SOURCE_DIR: &str = "src";

/// The kinds `record::kind` gives a closure's by-value call shim and a drop glue.
const CLOSURE_ONCE: &str = "Shim::ClosureOnce";
const DROP_GLUE: &str = "Shim::DropGlue";

/// The key that names `instance` to Haste: a function's definition and type arguments, a closure's place in its root with the root's arguments, or the type a drop glue drops; none when a part of it has no key.
pub fn instance<'tcx>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>) -> Option<String> {
    if let InstanceKind::Item(def_id) = instance.def {
        if tcx.is_closure_like(def_id) {
            return closure(tcx, def_id, instance.args);
        }
        return Some(format!("{}<{}>", def(tcx, def_id)?, args(tcx, instance.args)?));
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
    Some(match *ty.kind() {
        ty::Bool => "bool".to_string(),
        ty::Char => "char".to_string(),
        ty::Str => "str".to_string(),
        ty::Never => "!".to_string(),
        ty::Int(int) => int.name_str().to_string(),
        ty::Uint(uint) => uint.name_str().to_string(),
        ty::Float(float) => float.name_str().to_string(),
        ty::Ref(_, inner, mutability) => format!("&{}{}", mutability.prefix_str(), self::ty(tcx, inner)?),
        ty::RawPtr(inner, mutability) => format!("*{} {}", mutability.ptr_str(), self::ty(tcx, inner)?),
        ty::Array(inner, len) => format!("[{};{}]", self::ty(tcx, inner)?, value(len)?),
        ty::Slice(inner) => format!("[{}]", self::ty(tcx, inner)?),
        ty::Tuple(items) => format!("({})", items.iter().map(|item| self::ty(tcx, item).map(|key| key + ",")).collect::<Option<String>>()?),
        ty::Adt(adt, args) => format!("{}<{}>", def(tcx, adt.did())?, self::args(tcx, args)?),
        ty::FnDef(def_id, args) => format!("fn {}<{}>", def(tcx, def_id)?, self::args(tcx, args.skip_binder())?),
        ty::Closure(def_id, args) => closure(tcx, def_id, args)?,
        _ => return None,
    })
}

fn args<'tcx>(tcx: TyCtxt<'tcx>, args: &[ty::GenericArg<'tcx>]) -> Option<String> {
    let mut keys = Vec::new();
    for arg in args {
        match arg.kind() {
            GenericArgKind::Lifetime(_) => {}
            GenericArgKind::Type(one) => keys.push(ty(tcx, one)?),
            GenericArgKind::Const(one) => keys.push(value(one)?),
        }
    }
    Some(keys.join(","))
}

fn value(constant: ty::Const<'_>) -> Option<String> {
    Some(constant.try_to_value()?.valtree.try_to_leaf()?.to_bits_unchecked().to_string())
}

fn closure<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId, closure_args: GenericArgsRef<'tcx>) -> Option<String> {
    let root = tcx.typeck_root_def_id(def_id);
    let (order, count) = crate::record::order(tcx, def_id)?;
    let parent = closure_args.iter().take(tcx.generics_of(def_id).parent_count).collect::<Vec<_>>();
    Some(format!("closure {}#{order}/{count}<{}>", def(tcx, root)?, args(tcx, &parent)?))
}

/// The key of definition `def_id`: its crate, file, the byte offset of its name and the name, for a named definition outside any macro expansion.
fn def(tcx: TyCtxt<'_>, def_id: DefId) -> Option<String> {
    let (file, lo) = place(tcx, tcx.def_ident_span(def_id)?)?;
    Some(format!("{}:{file}:{lo}:{}", tcx.crate_name(def_id.krate), tcx.item_name(def_id)))
}

fn place(tcx: TyCtxt<'_>, span: Span) -> Option<(String, u32)> {
    if span.from_expansion() {
        return None;
    }
    let map = tcx.sess.source_map();
    let found = map.lookup_byte_offset(span.lo());
    let shown = map.span_to_diagnostic_string(span);
    let name = shown.rsplitn(2, ": ").last()?.rsplitn(3, ':').last()?;
    Some((normalize(&absolute(Path::new(name)))?, found.pos.0))
}

fn absolute(path: &Path) -> PathBuf {
    if path.is_absolute() {
        return path.to_path_buf();
    }
    std::env::current_dir().map_or_else(|_| path.to_path_buf(), |dir| dir.join(path))
}

/// The components of `path` from the parent of its last `src` directory on, joined by slashes.
fn normalize(path: &Path) -> Option<String> {
    let parts: Vec<Component<'_>> = path.components().collect();
    let at = parts.iter().rposition(|part| part.as_os_str() == SOURCE_DIR)?;
    let kept = parts.get(at.checked_sub(1)?..)?.iter().map(|part| part.as_os_str().to_string_lossy());
    Some(kept.collect::<Vec<_>>().join("/"))
}

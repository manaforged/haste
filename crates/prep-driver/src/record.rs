use std::fs::File;
use std::io::{BufWriter, Write};

use rustc_data_structures::fx::FxHashSet;
use rustc_hir::def_id::DefId;
use rustc_middle::mono::{MonoItem, MonoItemData};
use rustc_middle::ty::print::{with_resolve_crate_name, with_no_trimmed_paths};
use rustc_middle::middle::exported_symbols::ExportedSymbol;
use rustc_middle::ty::{self, GenericArgKind, Instance, InstanceKind, TyCtxt};
use rustc_session::config::CrateType;
use rustc_span::Symbol;
use serde_json::{Value, json};

use crate::error::Result;
use crate::wrap::Unit;
use crate::{abi, key, uses};

/// The file in each unit directory that holds the unit's manifest, one JSON record per line.
pub const MANIFEST: &str = "manifest.jsonl";

/// The key of the table that describes the whole library: the symbols of the shared standard library's argument initializer, of its `stdout` and of `flush` on what that returns.
pub const LIBRARY: &str = "library";

/// The key of the array of instance records in the merged manifest.
pub const RECORD: &str = "record";

/// The diagnostic items of `std::io::stdout`, of the `Write` trait, and the name of its flush method.
const STDOUT_ITEM: &str = "io_stdout";
const WRITE_ITEM: &str = "IoWrite";
const FLUSH: &str = "flush";

/// The definition path of the function that the shared standard library's loader calls with the process arguments.
const ARGS_INIT: &str = "std::sys::args::unix::imp::ARGV_INIT_ARRAY::init_wrapper";

/// The file in a linking unit's directory that names the sysroot libraries and native libraries its prep library links.
pub const LINK: &str = "link.txt";

/// Writes the manifest of every function instance the crate generates code for and of every trait object and function pointer type their code touches, and, for a crate that links, the libraries it links.
pub fn write(tcx: TyCtxt<'_>, unit: &Unit) -> Result<()> {
    let mut out = BufWriter::new(File::create(unit.dir.join(MANIFEST))?);
    writeln!(out, "{}", header(tcx, unit))?;
    let (mut seen, mut indirect) = (FxHashSet::default(), FxHashSet::default());
    for cgu in tcx.collect_and_partition_mono_items(()).codegen_units {
        for (item, data) in cgu.items() {
            let MonoItem::Fn(instance) = *item else {
                continue;
            };
            if seen.insert(instance) {
                writeln!(out, "{}", record(tcx, instance, data, (cgu.name(), &mut indirect))?)?;
            }
        }
    }
    for ty in indirect {
        writeln!(out, "{}", crate::object::facts(tcx, ty)?)?;
    }
    out.flush()?;
    if tcx.crate_types().iter().any(|kind| *kind != CrateType::Rlib) {
        crate::link::write_libraries(tcx, &unit.dir.join(LINK))?;
    }
    Ok(())
}

fn header(tcx: TyCtxt<'_>, unit: &Unit) -> Value {
    let sess = tcx.sess;
    json!({
        "unit": tcx.crate_name(rustc_hir::def_id::LOCAL_CRATE).as_str(),
        "version": unit.version,
        "crate_id": crate_id(tcx, rustc_hir::def_id::LOCAL_CRATE.as_def_id()),
        "crate_types": tcx.crate_types().iter().map(|kind| kind.to_string()).collect::<Vec<_>>(),
        "rustc": rustc_interface::util::rustc_version_str().unwrap_or_default(),
        "target": sess.opts.target_triple.to_string(),
        "opt_level": format!("{:?}", sess.opts.optimize),
        "codegen_units": sess.codegen_units().as_usize(),
        "panic": sess.panic_strategy().desc(),
        LIBRARY: { "args_init": args_init(tcx), "stdout": stdout(tcx).map(|(stdout, _)| stdout), "stdout_flush": stdout(tcx).map(|(_, flush)| flush) },
    })
}

/// The symbols of `std::io::stdout` and of `Write::flush` on the `Stdout` it returns.
fn stdout(tcx: TyCtxt<'_>) -> Option<(String, String)> {
    let stdout = tcx.get_diagnostic_item(Symbol::intern(STDOUT_ITEM))?;
    let write = tcx.get_diagnostic_item(Symbol::intern(WRITE_ITEM))?;
    let flush = tcx.associated_items(write).filter_by_name_unhygienic(Symbol::intern(FLUSH)).next()?.def_id;
    let ty = tcx.fn_sig(stdout).instantiate_identity().skip_normalization().output().skip_binder();
    let instance = Instance::try_resolve(tcx, ty::TypingEnv::fully_monomorphized(), flush, tcx.mk_args(&[ty.into()])).ok()??;
    Some((tcx.symbol_name(Instance::mono(tcx, stdout)).name.to_string(), tcx.symbol_name(instance).name.to_string()))
}

fn args_init(tcx: TyCtxt<'_>) -> Option<String> {
    let std = tcx.crates(()).iter().copied().find(|&cnum| tcx.crate_name(cnum).as_str() == "std" && crate::link::sysroot(tcx, cnum))?;
    tcx.exported_non_generic_symbols(std).iter().find_map(|(symbol, _)| match symbol {
        ExportedSymbol::NonGeneric(def_id) if path(tcx, *def_id) == ARGS_INIT => Some(tcx.symbol_name(Instance::mono(tcx, *def_id)).name.to_string()),
        _ => None,
    })
}

fn record<'tcx>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>, data: &MonoItemData, (cgu, indirect): (Symbol, &mut FxHashSet<ty::Ty<'tcx>>)) -> Result<Value> {
    let def_id = instance.def_id();
    let used = uses::used(tcx, instance);
    indirect.extend(used.indirect);
    let (abi, crossing) = abi::of(tcx, instance)?;
    Ok(json!({
        "symbol": tcx.symbol_name(instance).name,
        "key": key::instance(tcx, instance),
        "at": key::at(tcx, def_id),
        "crate": tcx.crate_name(def_id.krate).as_str(),
        "crate_id": crate_id(tcx, def_id),
        "kind": kind(instance.def),
        "path": path(tcx, def_id),
        "closure": closure(tcx, instance),
        "span": span(tcx, def_id),
        "args": args(tcx, instance),
        "local_copy": data.inlined,
        "cgu": cgu.as_str(),
        "abi": abi,
        "layouts": crate::layout::reached(tcx, &crossing),
        "inlined": uses::inlined(tcx, instance),
        "uses": used.symbols,
        "state": used.state,
    }))
}

fn crate_id(tcx: TyCtxt<'_>, def_id: DefId) -> String {
    format!("{:016x}", tcx.stable_crate_id(def_id.krate).as_u64())
}

fn path(tcx: TyCtxt<'_>, def_id: DefId) -> String {
    with_no_trimmed_paths!(with_resolve_crate_name!(tcx.def_path_str(def_id)))
}

pub fn kind(def: InstanceKind<'_>) -> String {
    let text = format!("{def:?}");
    let mut words = text.split(['(', ' ', ',', ')']).filter(|word| !word.is_empty());
    match (words.next(), words.next()) {
        (Some("Shim"), Some(shim)) => format!("Shim::{shim}"),
        (Some(kind), _) => kind.to_string(),
        _ => text.clone(),
    }
}

fn closure<'tcx>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>) -> Value {
    let def_id = instance.def_id();
    if !tcx.is_closure_like(def_id) {
        return Value::Null;
    }
    let parent = tcx.parent(def_id);
    let root = tcx.typeck_root_def_id(def_id);
    let index = tcx.def_key(def_id).disambiguated_data.disambiguator;
    let kind = match tcx.type_of(def_id).instantiate_identity().skip_normalization().kind() {
        ty::Closure(..) => Some(format!("{:?}", instance.args.as_closure().kind())),
        _ => None,
    };
    let (order, count) = order(tcx, def_id).unzip();
    json!({ "parent": path(tcx, parent), "root": path(tcx, root), "index": index, "kind": kind, "order": order, "count": count })
}

/// The place of local closure `def_id` among the closures of its typeck root in the order their expressions end, inner before outer, and how many there are; none when the root holds a coroutine, an inline constant or a closure a macro wrote.
pub fn order(tcx: TyCtxt<'_>, def_id: DefId) -> Option<(usize, usize)> {
    let local = def_id.as_local()?;
    let root = tcx.typeck_root_def_id(def_id).as_local()?;
    let mut ends = Vec::new();
    for nested in tcx.nested_bodies_within(root) {
        let span = tcx.hir_span_with_body(tcx.local_def_id_to_hir_id(nested));
        let plain = matches!(tcx.type_of(nested).instantiate_identity().skip_normalization().kind(), ty::Closure(..));
        if !plain || span.from_expansion() {
            return None;
        }
        ends.push((span.hi(), std::cmp::Reverse(span.lo()), nested));
    }
    ends.sort_by_key(|(hi, lo, _)| (*hi, *lo));
    Some((ends.iter().position(|(_, _, nested)| *nested == local)?, ends.len()))
}

fn span(tcx: TyCtxt<'_>, def_id: DefId) -> Value {
    let span = tcx.def_span(def_id);
    let map = tcx.sess.source_map();
    let lo = map.lookup_byte_offset(span.lo());
    let hi = map.lookup_byte_offset(span.hi());
    json!({ "at": map.span_to_diagnostic_string(span), "lo": lo.pos.0, "hi": hi.pos.0 })
}

fn args<'tcx>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>) -> Vec<String> {
    let def_id = instance.def_id();
    let count = if tcx.is_closure_like(def_id) { tcx.generics_of(def_id).parent_count } else { instance.args.len() };
    let kept = instance.args.iter().take(count).filter(|arg| !matches!(arg.kind(), GenericArgKind::Lifetime(_)));
    kept.map(|arg| with_no_trimmed_paths!(arg.to_string())).collect()
}

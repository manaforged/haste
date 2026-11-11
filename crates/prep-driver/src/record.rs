use std::fs::File;
use std::io::{BufWriter, Write};

use rustc_data_structures::fx::FxHashSet;
use rustc_hir::def_id::DefId;
use rustc_middle::mono::{MonoItem, MonoItemData};
use rustc_middle::ty::print::{with_resolve_crate_name, with_no_trimmed_paths};
use rustc_middle::ty::{GenericArgKind, Instance, InstanceKind, TyCtxt};
use rustc_session::config::CrateType;
use rustc_span::Symbol;
use serde_json::{Value, json};

use crate::error::Result;
use crate::wrap::Unit;
use crate::{abi, uses};

/// The file in each unit directory that holds the unit's manifest, one JSON record per line.
pub const MANIFEST: &str = "manifest.jsonl";

/// The file in a linking unit's directory that names the sysroot libraries and native libraries its prep library links.
pub const LINK: &str = "link.txt";

/// Writes the manifest of every function instance the crate generates code for, and, for a crate that links, the libraries it links.
pub fn write(tcx: TyCtxt<'_>, unit: &Unit) -> Result<()> {
    let mut out = BufWriter::new(File::create(unit.dir.join(MANIFEST))?);
    writeln!(out, "{}", header(tcx, unit))?;
    let mut seen = FxHashSet::default();
    for cgu in tcx.collect_and_partition_mono_items(()).codegen_units {
        for (item, data) in cgu.items() {
            let MonoItem::Fn(instance) = *item else {
                continue;
            };
            if seen.insert(instance) {
                writeln!(out, "{}", record(tcx, instance, data, cgu.name())?)?;
            }
        }
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
    })
}

fn record<'tcx>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>, data: &MonoItemData, cgu: Symbol) -> Result<Value> {
    let def_id = instance.def_id();
    Ok(json!({
        "symbol": tcx.symbol_name(instance).name,
        "crate": tcx.crate_name(def_id.krate).as_str(),
        "crate_id": crate_id(tcx, def_id),
        "kind": kind(instance.def),
        "path": path(tcx, def_id),
        "closure": closure(tcx, def_id),
        "span": span(tcx, def_id),
        "args": args(tcx, instance),
        "local_copy": data.inlined,
        "cgu": cgu.as_str(),
        "abi": abi::of(tcx, instance)?,
        "inlined": uses::inlined(tcx, instance),
        "uses": uses::used(tcx, instance),
    }))
}

fn crate_id(tcx: TyCtxt<'_>, def_id: DefId) -> String {
    format!("{:016x}", tcx.stable_crate_id(def_id.krate).as_u64())
}

fn path(tcx: TyCtxt<'_>, def_id: DefId) -> String {
    with_no_trimmed_paths!(with_resolve_crate_name!(tcx.def_path_str(def_id)))
}

fn kind(def: InstanceKind<'_>) -> String {
    let text = format!("{def:?}");
    let mut words = text.split(['(', ' ', ',', ')']).filter(|word| !word.is_empty());
    match (words.next(), words.next()) {
        (Some("Shim"), Some(shim)) => format!("Shim::{shim}"),
        (Some(kind), _) => kind.to_string(),
        _ => text.clone(),
    }
}

fn closure(tcx: TyCtxt<'_>, def_id: DefId) -> Value {
    if !tcx.is_closure_like(def_id) {
        return Value::Null;
    }
    let parent = tcx.parent(def_id);
    let root = tcx.typeck_root_def_id(def_id);
    let index = tcx.def_key(def_id).disambiguated_data.disambiguator;
    json!({ "parent": path(tcx, parent), "root": path(tcx, root), "index": index })
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

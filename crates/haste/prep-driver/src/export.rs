use std::sync::OnceLock;

use rustc_data_structures::fx::FxHashSet;
use rustc_hir::attrs::Linkage;
use rustc_middle::bug;
use rustc_middle::middle::exported_symbols::{ExportedSymbol, SymbolExportInfo, SymbolExportKind, SymbolExportLevel};
use rustc_middle::mono::{CodegenUnit, MonoItem, MonoItemData, MonoItemPartitions, Visibility};
use rustc_middle::query::LocalCrate;
use rustc_middle::ty::TyCtxt;
use rustc_middle::util::Providers;
use rustc_session::Session;

static BASE: OnceLock<Providers> = OnceLock::new();

/// Replaces the crate's partitioning and its exported generic symbols so that every function instance and static keeps an exported definition through LLVM and local ThinLTO.
pub fn override_queries(_: &Session, providers: &mut Providers) {
    BASE.get_or_init(|| *providers);
    providers.queries.collect_and_partition_mono_items = partition;
    providers.queries.exported_generic_symbols = exported;
}

/// Replaces the crate's partitioning so that every static keeps an exported definition, which the library's two builds of the crate then share, and functions keep rustc's linkage.
pub fn override_statics(_: &Session, providers: &mut Providers) {
    BASE.get_or_init(|| *providers);
    providers.queries.collect_and_partition_mono_items = partition_statics;
}

fn base() -> &'static Providers {
    match BASE.get() {
        Some(providers) => providers,
        None => bug!("prep queries ran before they were overridden"),
    }
}

/// Which items a partitioning exports: statics only, or functions too.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Exported {
    Statics,
    All,
}

fn partition(tcx: TyCtxt<'_>, key: ()) -> MonoItemPartitions<'_> {
    repartition(tcx, key, Exported::All)
}

fn partition_statics(tcx: TyCtxt<'_>, key: ()) -> MonoItemPartitions<'_> {
    repartition(tcx, key, Exported::Statics)
}

fn repartition(tcx: TyCtxt<'_>, key: (), exported: Exported) -> MonoItemPartitions<'_> {
    let base = (base().queries.collect_and_partition_mono_items)(tcx, key);
    let units: Vec<CodegenUnit<'_>> = base.codegen_units.iter().map(|unit| exported_unit(unit, exported)).collect();
    MonoItemPartitions { codegen_units: tcx.arena.alloc_from_iter(units), all_mono_items: base.all_mono_items }
}

fn exported_unit<'tcx>(unit: &CodegenUnit<'tcx>, exported: Exported) -> CodegenUnit<'tcx> {
    let mut copy = CodegenUnit::new(unit.name());
    copy.set_symbol_name(unit.symbol_name());
    if unit.is_primary() {
        copy.make_primary();
    }
    for (item, data) in unit.items() {
        copy.items_mut().insert(*item, exported_item(item, *data, exported));
    }
    copy.compute_size_estimate();
    copy
}

/// How the library defines `item`: a function, when all are exported, as a weak definition every copy may hold; a static as one exported definition, which the library's two builds share once linking keeps the first.
fn exported_item(item: &MonoItem<'_>, data: MonoItemData, exported: Exported) -> MonoItemData {
    let linkage = match (item, data.linkage) {
        (MonoItem::Fn(_), Linkage::Internal) if exported == Exported::All => Linkage::WeakODR,
        (MonoItem::Fn(_), _) if exported == Exported::All => data.linkage,
        (MonoItem::Static(_), Linkage::Internal) => Linkage::External,
        (MonoItem::Static(_), _) => data.linkage,
        _ => return data,
    };
    MonoItemData { linkage, visibility: Visibility::Default, ..data }
}

fn exported<'tcx>(tcx: TyCtxt<'tcx>, krate: LocalCrate) -> &'tcx [(ExportedSymbol<'tcx>, SymbolExportInfo)] {
    let base = (base().queries.exported_generic_symbols)(tcx, krate);
    let info = SymbolExportInfo { level: SymbolExportLevel::Rust, kind: SymbolExportKind::Text, used: false, rustc_std_internal_symbol: false };
    let mut seen = FxHashSet::default();
    let units = tcx.collect_and_partition_mono_items(()).codegen_units;
    let instances = units.iter().flat_map(|unit| unit.items().keys()).filter_map(|item| match item {
        MonoItem::Fn(instance) => Some(*instance),
        _ => None,
    });
    let added: Vec<_> = instances.filter(|instance| seen.insert(*instance)).map(|instance| (ExportedSymbol::NoDefId(tcx.symbol_name(instance)), info)).collect();
    tcx.arena.alloc_from_iter(base.iter().copied().chain(added))
}

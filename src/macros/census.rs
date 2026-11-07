use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Result;
use base_db::all_crates;
use hir_def::expr_store::Body;
use hir_def::nameres::crate_def_map;
use hir_def::{AssocItemId, DefWithBodyId, ModuleDefId};
use hir_expand::{MacroCallId, MacroCallKind, MacroDefKind};
use ide_db::RootDatabase;
use load_cargo::{LoadCargoConfig, ProcMacroServerChoice, load_units_at};
use project_model::CargoConfig;
use vfs::Vfs;

type Key = (String, u32, u32, u32);
type Expansions = BTreeMap<Key, Vec<String>>;

/// Compares every proc-macro expansion of the engine load with the expansion that rustc-built proc macros give.
pub fn run(engine: &RootDatabase, engine_vfs: &Vfs, path: &Path, config: &CargoConfig) -> Result<()> {
    let ours = expansions(engine, engine_vfs);
    let load = LoadCargoConfig {
        load_out_dirs_from_check: true,
        with_proc_macro_server: ProcMacroServerChoice::Sysroot,
        prefill_caches: false,
        num_worker_threads: 1,
        proc_macro_processes: 1,
    };
    let (db, vfs, _server, _roots) = load_units_at(path, config, &load, &|_| {})?;
    let theirs = expansions(&db, &vfs);
    report(&ours, &theirs);
    Ok(())
}

fn report(ours: &Expansions, theirs: &Expansions) {
    let (mut agree, mut differ, mut missing) = (0, 0, 0);
    for (key, expected) in theirs {
        match ours.get(key) {
            Some(found) if found == expected => agree += 1,
            Some(found) => {
                differ += 1;
                if differ <= 5 {
                    println!("differ {key:?}\n  engine: {}\n  rustc:  {}", found.join(" | "), expected.join(" | "));
                }
            }
            None => {
                missing += 1;
                println!("missing {key:?}");
            }
        }
    }
    let extra = ours.keys().filter(|key| !theirs.contains_key(*key)).count();
    println!("macros total={} agree={agree} differ={differ} missing={missing} extra={extra}", theirs.len());
}

fn expansions(db: &RootDatabase, vfs: &Vfs) -> Expansions {
    let mut found = Expansions::new();
    for call in calls(db) {
        let loc = call.loc(db);
        if !matches!(loc.def.kind, MacroDefKind::ProcMacro(..)) {
            continue;
        }
        let range = loc.kind.original_call_range_with_input(db);
        let index = if let MacroCallKind::Derive { derive_index, .. } = loc.kind { derive_index } else { 0 };
        let file = vfs.file_path(range.file_id.file_id(db)).to_string();
        let key = (file, range.range.start().into(), range.range.end().into(), index);
        found.entry(key).or_default().push(text(db, call));
    }
    for texts in found.values_mut() {
        texts.sort();
    }
    found
}

fn text(db: &RootDatabase, call: MacroCallId) -> String {
    let expanded = call.parse_macro_expansion(db);
    let body = expanded.value.0.syntax_node().to_string();
    match &expanded.err {
        Some(error) => format!("{body} !error {}", error.render_to_string(db).message),
        None => body,
    }
}

fn calls(db: &RootDatabase) -> Vec<MacroCallId> {
    let mut calls = Vec::new();
    for krate in all_crates(db).iter().copied() {
        let map = crate_def_map(db, krate);
        for (_, module) in map.modules() {
            calls.extend(module.scope.all_macro_calls());
            for owner in bodies(db, &module.scope) {
                let (_, source) = Body::with_source_map(db, owner);
                calls.extend(source.expansions().map(|(_, call)| *call));
            }
        }
    }
    calls
}

fn bodies(db: &RootDatabase, scope: &hir_def::item_scope::ItemScope) -> Vec<DefWithBodyId> {
    let mut owners: Vec<DefWithBodyId> = scope.declarations().filter_map(|def| match def {
        ModuleDefId::FunctionId(id) => Some(id.into()),
        ModuleDefId::ConstId(id) => Some(id.into()),
        ModuleDefId::StaticId(id) => Some(id.into()),
        _ => None,
    }).collect();
    for imp in scope.impls() {
        owners.extend(imp.impl_items(db).items.iter().filter_map(|(_, item)| match item {
            AssocItemId::FunctionId(id) => Some(DefWithBodyId::from(*id)),
            AssocItemId::ConstId(id) => Some(DefWithBodyId::from(*id)),
            AssocItemId::TypeAliasId(_) => None,
        }));
    }
    owners
}

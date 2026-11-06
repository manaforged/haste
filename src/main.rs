mod backend;
mod fault;
mod collect;
mod lower;
mod patch;

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use base_db::all_crates;
use hir_def::FunctionId;
use hir_def::nameres::crate_def_map;
use hir_def::signatures::FunctionSignature;
use hir_def::ModuleDefId;
use hir_ty::db::HirDatabase;
use load_cargo::{LoadCargoConfig, ProcMacroServerChoice, load_workspace_at};
use project_model::{CargoConfig, RustLibSource};

const USAGE: &str = "usage: engine-tracer collect PATH | patch PATH ENTRY EDITS [dump] | coverage PATH CRATE | check PATH";

fn main() -> Result<()> {
    fault::install();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = std::path::absolute(PathBuf::from(args.get(1).context(USAGE)?))?;
    let config = CargoConfig {
        sysroot: Some(RustLibSource::Discover),
        ..Default::default()
    };
    let load = LoadCargoConfig {
        load_out_dirs_from_check: true,
        with_proc_macro_server: ProcMacroServerChoice::Sysroot,
        prefill_caches: false,
        num_worker_threads: 1,
        proc_macro_processes: 1,
    };
    let started = Instant::now();
    let (db, vfs, _proc) = load_workspace_at(&path, &config, &load, &|_| {})?;
    println!("load {:.2}s", started.elapsed().as_secs_f64());
    match args[0].as_str() {
        "collect" => {
            let main = find(&db, "main")?;
            collect::run(&db, main);
            Ok(())
        }
        "patch" => {
            let entry = args.get(2).context(USAGE)?;
            let edits = args.get(3).context(USAGE)?;
            let dump = args.get(4).is_some_and(|arg| arg == "dump");
            patch::run(db, &vfs, &path, entry, edits.as_ref(), dump)
        }
        "check" => patch::check(db, &args[2..]),
        "coverage" => {
            let krate = args.get(2).context(USAGE)?;
            patch::coverage(db, krate)
        }
        _ => bail!(USAGE),
    }
}

pub fn local_crates(db: &dyn HirDatabase) -> Vec<base_db::Crate> {
    all_crates(db).iter().copied().filter(|krate| krate.data(db).origin.is_local()).collect()
}

pub fn free_functions_of(db: &dyn HirDatabase, krate: base_db::Crate) -> Vec<FunctionId> {
    let map = crate_def_map(db, krate);
    let mut found = Vec::new();
    for (_, module) in map.modules() {
        for def in module.scope.declarations() {
            if let ModuleDefId::FunctionId(id) = def {
                found.push(id);
            }
        }
    }
    found
}

pub fn free_functions(db: &dyn HirDatabase, crate_name: &str) -> Vec<FunctionId> {
    let named = |krate: &base_db::Crate| {
        krate.extra_data(db).display_name.as_ref().is_some_and(|name| name.to_string() == crate_name)
    };
    local_crates(db).into_iter().filter(named).flat_map(|krate| free_functions_of(db, krate)).collect()
}

pub fn find(db: &dyn HirDatabase, name: &str) -> Result<FunctionId> {
    for krate in all_crates(db).iter().copied() {
        if !krate.data(db).origin.is_local() {
            continue;
        }
        let map = crate_def_map(db, krate);
        for (_, module) in map.modules() {
            for def in module.scope.declarations() {
                if let ModuleDefId::FunctionId(id) = def
                    && FunctionSignature::of(db, id).name.as_str() == name
                {
                    return Ok(id);
                }
            }
        }
    }
    bail!("no function {name} in a local crate")
}

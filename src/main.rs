mod backend;
mod fault;
mod unwind;
mod collect;
mod lower;
mod macros;
mod image;
mod demand;
mod index;
#[cfg(target_os = "macos")]
mod park;
mod patch;
mod script;

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use base_db::all_crates;
use hir_def::FunctionId;
use hir_def::nameres::crate_def_map;
use hir_def::signatures::FunctionSignature;
use hir_def::ModuleDefId;
use hir_ty::db::HirDatabase;
use load_cargo::{LoadCargoConfig, ProcMacroServerChoice, load_units_at};
use project_model::{CargoConfig, RustLibSource};

const USAGE: &str = "usage: haste collect PATH | patch PATH ENTRY EDITS [dump] | index PATH | demand PATH ENTRY | scripts PATH | image PATH ENTRY EDITS | coverage PATH CRATE | check PATH | macros PATH";

fn main() -> Result<()> {
    if std::env::var("RUSTC_BOOTSTRAP").ok().as_deref() == option_env!("RUSTC_BOOTSTRAP") {
        // SAFETY: no other thread runs yet, so nothing reads the environment while it changes.
        unsafe { std::env::remove_var("RUSTC_BOOTSTRAP") };
    }
    let program = std::thread::Builder::new().name("program".to_owned()).stack_size(patch::PROGRAM_STACK).spawn(|| {
        fault::install();
        program()
    })?;
    program.join().map_err(|_| anyhow::anyhow!("the program thread panicked"))?
}

const PROC_MACRO_PROCESSES: usize = 4;

fn program() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = std::path::absolute(PathBuf::from(args.get(1).context(USAGE)?))?;
    let config = CargoConfig {
        sysroot: Some(RustLibSource::Discover),
        ..Default::default()
    };
    let load = LoadCargoConfig {
        load_out_dirs_from_check: false,
        with_proc_macro_server: ProcMacroServerChoice::Engine,
        prefill_caches: false,
        num_worker_threads: 1,
        proc_macro_processes: PROC_MACRO_PROCESSES,
    };
    if args[0] == "demand" {
        base_db::demand::enable();
    }
    let started = Instant::now();
    let (mut db, mut vfs, _proc, roots) = load_units_at(&path, &config, &load, &|_| {})?;
    macros::declare(&mut db, &vfs);
    println!("load {:.2}s", started.elapsed().as_secs_f64());
    let ran = Instant::now();
    let (db, outcomes) = script::run_all(db, &mut vfs, &roots)?;
    println!("build_scripts {:.2}s", ran.elapsed().as_secs_f64());
    if args[0] == "index" {
        index::oracle::run(&db, &vfs);
        index::headers::run(&db, &vfs);
        return Ok(());
    }
    if args[0] == "macros" {
        return macros::census::run(&db, &vfs, &path, &config);
    }
    if args[0] == "scripts" {
        return script::census::run(&outcomes, &path);
    }
    if args[0] == "demand" {
        return demand::run(db, args.get(2).context(USAGE)?);
    }
    let cores = std::thread::available_parallelism().map_or(1, |count| count.get());
    let primed = Instant::now();
    ide_db::prime_caches::prime_semantics(&db, &all_crates(&db), cores);
    println!("prime {:.2}s", primed.elapsed().as_secs_f64());
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
        "image" => {
            let entry = args.get(2).context(USAGE)?;
            let edits = args.get(3).context(USAGE)?;
            image::run(db, &vfs, &path, entry, edits.as_ref())
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

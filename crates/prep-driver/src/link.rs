use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use rustc_attr_parsing::eval_config_entry;
use rustc_hir::def_id::{CrateNum, LOCAL_CRATE};
use rustc_middle::ty::TyCtxt;
use rustc_attr_ir::NativeLibKind;
use rustc_crate_store::LinkagePreference;
use serde_json::Value;

use crate::elf::{self, Scope, Table};
use crate::object;
use crate::statics;
use crate::error::{Error, Result};
use crate::record::{self, LINK, MANIFEST};
use crate::wrap::RELEASE;

/// Links every unit under a prep directory into one shared library and writes its manifest beside it.
pub fn main(args: &[String]) -> ExitCode {
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("prep link: {error}");
            ExitCode::FAILURE
        }
    }
}

/// Whether crate `cnum` comes from the toolchain's sysroot, as the standard crates do.
pub fn sysroot(tcx: TyCtxt<'_>, cnum: CrateNum) -> bool {
    if cnum == LOCAL_CRATE {
        return false;
    }
    let source = tcx.used_crate_source(cnum);
    [&source.rlib, &source.rmeta, &source.dylib].into_iter().flatten().any(|path| path.starts_with(tcx.sess.opts.sysroot.path()))
}

/// Writes the units, sysroot dylibs and native libraries that the crate being compiled links, one per line, marking each sysroot rlib whose crate no sysroot dylib holds.
pub fn write_libraries(tcx: TyCtxt<'_>, path: &Path) -> Result<()> {
    let mut out = BufWriter::new(File::create(path)?);
    let held = in_dylibs(tcx);
    for &cnum in tcx.crates(()) {
        let source = tcx.used_crate_source(cnum);
        let unit = source.rlib.as_ref().filter(|_| !sysroot(tcx, cnum)).and_then(|rlib| rlib.file_stem()?.to_str()?.strip_prefix(RLIB_PREFIX));
        if let Some(unit) = unit {
            writeln!(out, "{UNIT} {unit}")?;
        }
        if let Some(path) = source.dylib.as_ref().filter(|_| sysroot(tcx, cnum)) {
            writeln!(out, "{DYLIB} {}", path.display())?;
        }
        if let Some(path) = source.rlib.as_ref().filter(|_| sysroot(tcx, cnum) && !held.contains(&cnum)) {
            writeln!(out, "{ALONE} {}", path.display())?;
        }
        for lib in tcx.native_libraries(cnum) {
            let relevant = lib.cfg.as_ref().is_none_or(|cfg| eval_config_entry(tcx.sess, cfg).as_bool());
            if relevant && matches!(lib.kind, NativeLibKind::Dylib { .. } | NativeLibKind::Unspecified) {
                writeln!(out, "{NATIVE} {}", lib.name)?;
            }
        }
    }
    out.flush()?;
    Ok(())
}

/// The sysroot crates whose code a sysroot dylib holds: each such dylib's crate and the crates it links statically.
fn in_dylibs(tcx: TyCtxt<'_>) -> HashSet<CrateNum> {
    let mut held = HashSet::new();
    for &cnum in tcx.crates(()).iter().filter(|&&cnum| sysroot(tcx, cnum) && tcx.used_crate_source(cnum).dylib.is_some()) {
        held.insert(cnum);
        let linked = tcx.dylib_dependency_formats(cnum).iter().filter(|(_, preference)| *preference == LinkagePreference::RequireStatic);
        held.extend(linked.map(|&(dependency, _)| dependency));
    }
    held
}

const DYLIB: &str = "dylib";
const ALONE: &str = "alone";
const NATIVE: &str = "lib";
const UNIT: &str = "unit";
const RLIB_PREFIX: &str = "lib";

const ALLOCATOR_SUFFIX: &str = "___rust_alloc";

/// The kinds of lines that describe a trait object or function pointer type or a static, in the order the merged manifest lists them.
const INDIRECT: [&str; 3] = [object::OBJECT, object::POINTER, statics::STATIC];

/// The extension of the merged manifest that `link` writes beside the library, in TOML.
pub const MANIFEST_EXTENSION: &str = "toml";

#[derive(Default)]
struct Inputs {
    objects: Vec<PathBuf>,
    released: Vec<PathBuf>,
    dylibs: BTreeSet<String>,
    alone: BTreeSet<String>,
    libs: Vec<String>,
    manifests: Vec<PathBuf>,
}

fn run(args: &[String]) -> Result<()> {
    let [root, out] = args else {
        return Err(Error("usage: prep-driver link <prep directory> <library>".to_string()));
    };
    let mut inputs = gather(Path::new(root))?;
    inputs.objects = without_allocator(&inputs.objects)?;
    inputs.released = without_allocator(&inputs.released)?;
    let symbols = symbols(&inputs.manifests)?;
    let released = released(&inputs.released, &symbols)?;
    let mut rebound = 0;
    for object in inputs.released.iter().chain(&inputs.objects) {
        rebound += elf::rebind(object, &symbols)?;
    }
    link(&inputs, Path::new(out))?;
    let defined = elf::defined(Path::new(out), Table::Dynamic, Scope::All)?;
    let mut shared = HashSet::new();
    for dylib in &inputs.dylibs {
        shared.extend(elf::defined(Path::new(dylib), Table::Dynamic, Scope::All)?);
    }
    let (kept, missing) = merge(&inputs.manifests, (&defined, &shared, &released), &PathBuf::from(format!("{out}.{MANIFEST_EXTENSION}")))?;
    let size = fs::metadata(out)?.len();
    let counts = format!("units {} objects {} released {}", inputs.manifests.len(), inputs.objects.len() + inputs.released.len(), released.len());
    println!("{counts} rebound {rebound} records {kept} missing {missing} bytes {size}");
    Ok(())
}

/// The inputs of the units under `root` that a linked crate of the package links, or of every unit when no crate of it links.
fn gather(root: &Path) -> Result<Inputs> {
    let mut inputs = Inputs::default();
    let mut dirs: Vec<PathBuf> = fs::read_dir(root)?.filter_map(|entry| entry.ok().map(|entry| entry.path())).collect();
    dirs.sort();
    let mut linked = HashSet::new();
    for dir in dirs.iter().filter(|dir| dir.join(LINK).exists()) {
        let text = fs::read_to_string(dir.join(LINK))?;
        linked.extend(text.lines().filter_map(|line| line.strip_prefix(UNIT)?.strip_prefix(' ')).map(str::to_string));
    }
    let wanted = |dir: &PathBuf| linked.is_empty() || dir.join(LINK).exists() || dir.file_name().and_then(|name| name.to_str()).is_some_and(|name| linked.contains(name));
    for dir in dirs.iter().filter(|dir| dir.join(MANIFEST).exists() && wanted(dir)) {
        inputs.manifests.push(dir.join(MANIFEST));
        inputs.objects.extend(objects(dir)?);
        inputs.released.extend(objects(&dir.join(RELEASE))?);
        if dir.join(LINK).exists() {
            libraries(&dir.join(LINK), &mut inputs)?;
        }
    }
    Ok(inputs)
}

fn objects(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut found = Vec::new();
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().is_some_and(|ext| ext == "o") {
            found.push(path);
        }
    }
    found.sort();
    Ok(found)
}

fn libraries(path: &Path, inputs: &mut Inputs) -> Result<()> {
    for line in fs::read_to_string(path)?.lines() {
        match line.split_once(' ') {
            Some((DYLIB, dylib)) => _ = inputs.dylibs.insert(dylib.to_string()),
            Some((ALONE, rlib)) => _ = inputs.alone.insert(rlib.to_string()),
            Some((NATIVE, lib)) if !inputs.libs.iter().any(|known| known == lib) => inputs.libs.push(lib.to_string()),
            _ => {}
        }
    }
    Ok(())
}

/// Links the inputs into the shared library `out` against the toolchain's shared libstd, so the process holds one std.
fn link(inputs: &Inputs, out: &Path) -> Result<()> {
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let mut command = Command::new(cc);
    command.args(["-shared", "-Wl,-Bsymbolic", "-o"]).arg(out);
    command.args(&inputs.released).args(&inputs.objects);
    command.args(&inputs.dylibs);
    let dirs: BTreeSet<_> = inputs.dylibs.iter().filter_map(|dylib| Path::new(dylib).parent()).collect();
    command.args(dirs.iter().map(|dir| format!("-Wl,-rpath,{}", dir.display())));
    command.arg("-Wl,--start-group").args(&inputs.alone).arg("-Wl,--end-group");
    command.args(inputs.libs.iter().map(|lib| format!("-l{lib}")));
    let status = command.status()?;
    if !status.success() {
        return Err(Error(format!("the linker failed: {status}")));
    }
    Ok(())
}

fn symbols(manifests: &[PathBuf]) -> Result<HashSet<String>> {
    let mut symbols = HashSet::new();
    for manifest in manifests {
        for line in BufReader::new(File::open(manifest)?).lines() {
            let record: Value = serde_json::from_str(&line?)?;
            symbols.extend(record.get("symbol").and_then(Value::as_str).map(str::to_string));
        }
    }
    Ok(symbols)
}

fn without_allocator(objects: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut kept = Vec::new();
    for object in objects {
        if !elf::defined(object, Table::Static, Scope::All)?.iter().any(|name| name.ends_with(ALLOCATOR_SUFFIX)) {
            kept.push(object.clone());
        }
    }
    Ok(kept)
}

/// The instances whose release copy keeps the convention its record states: those a release object defines as global, since LLVM may rewrite the arguments of a local function whose every caller it sees.
fn released(objects: &[PathBuf], symbols: &HashSet<String>) -> Result<HashSet<String>> {
    let mut found = HashSet::new();
    for object in objects {
        found.extend(elf::defined(object, Table::Static, Scope::Extern)?.into_iter().filter(|name| symbols.contains(name)));
    }
    Ok(found)
}

/// Writes the merged manifest of `manifests` to `out`: each record whose symbol the library defines, or the shared standard library for a record of its code, and each distinct line describing a type or a static.
fn merge(manifests: &[PathBuf], (defined, shared, released): (&HashSet<String>, &HashSet<String>, &HashSet<String>), out: &Path) -> Result<(usize, usize)> {
    let (mut records, mut library) = (Vec::new(), serde_json::Map::new());
    let mut indirect: [BTreeMap<String, Value>; 3] = Default::default();
    let mut missing = 0;
    for manifest in manifests {
        for line in BufReader::new(File::open(manifest)?).lines() {
            let mut record: Value = serde_json::from_str(&line?)?;
            if let Some((kind, name)) = INDIRECT.iter().enumerate().find_map(|(at, kind)| Some((at, record.get(*kind)?.as_str()?.to_string()))) {
                indirect[kind].entry(name).or_insert(record);
                continue;
            }
            let Some(symbol) = record.get("symbol").and_then(Value::as_str).map(str::to_string) else {
                let facts = record.get(record::LIBRARY).and_then(Value::as_object).cloned().unwrap_or_default();
                library.extend(facts.into_iter().filter(|(_, value)| !value.is_null()));
                continue;
            };
            let upstream = record.get(record::UPSTREAM).and_then(Value::as_bool) == Some(true);
            if !defined.contains(&symbol) && !(upstream && shared.contains(&symbol)) {
                missing += 1;
                continue;
            }
            record["release_copy"] = Value::Bool(released.contains(&symbol));
            records.push(record);
        }
    }
    let kept = records.len();
    let mut document = serde_json::Map::new();
    document.insert(record::LIBRARY.to_string(), Value::Object(library));
    document.insert(record::RECORD.to_string(), Value::Array(records));
    for (kind, lines) in INDIRECT.iter().zip(indirect) {
        document.insert(kind.to_string(), Value::Array(lines.into_values().collect()));
    }
    let document = Value::Object(document);
    let text = toml::to_string(&without_nulls(document)).map_err(|error| Error(error.to_string()))?;
    fs::write(out, text)?;
    Ok((kept, missing))
}

fn without_nulls(value: Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(map.into_iter().filter(|(_, value)| !value.is_null()).map(|(key, value)| (key, without_nulls(value))).collect()),
        Value::Array(items) => Value::Array(items.into_iter().map(without_nulls).collect()),
        other => other,
    }
}

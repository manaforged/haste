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
use crate::files;
use crate::object;
use crate::statics;
use crate::error::{Error, Result};
use crate::record::{self, LINK, MANIFEST};
use crate::wrap::RELEASE;

/// Links every unit under a prep directory into one shared library and writes its manifest beside it, or, given an earlier library, only the units rebuilt since into a second library that resolves against it.
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
            writeln!(out, "{} {unit}", record::UNIT)?;
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
const RLIB_PREFIX: &str = "lib";

const ALLOCATOR_SUFFIX: &str = "___rust_alloc";

const USAGE: &str = "usage: prep-driver link <prep directory> <library directory> [<directory of an earlier library>]";

/// The name of the lld driver in the toolchain's linker directory.
const LLD: &str = "ld.lld";

/// What a second library adds to its link so it finds the first one beside it at run time and leaves no symbol unresolved.
const AGAINST_BASE: [&str; 2] = ["-Wl,-rpath,$ORIGIN", "-Wl,--no-undefined"];

/// The kinds of lines that describe a trait object or function pointer type or a static, in the order the merged manifest lists them.
const INDIRECT: [&str; 3] = [object::OBJECT, object::POINTER, statics::STATIC];

/// What a link of some units takes: their objects without allocator shims, their release objects, what the linking units link, their manifests, the instances whose release copy keeps its convention, and how many symbols rebinding changed.
#[derive(Default)]
pub struct Inputs {
    objects: Vec<PathBuf>,
    released: Vec<PathBuf>,
    dylibs: BTreeSet<String>,
    alone: BTreeSet<String>,
    libs: Vec<String>,
    manifests: Vec<PathBuf>,
    copies: HashSet<String>,
    rebound: usize,
}

/// Where a link finds what its units do not define: the shared standard library and the sysroot rlibs no dylib holds, or also the library in a directory, which already links those rlibs.
#[derive(Clone, Copy)]
pub enum Against<'a> {
    Sysroot,
    Base(&'a Path),
}

fn run(args: &[String]) -> Result<()> {
    match args {
        [root, dir] => whole(Path::new(root), Path::new(dir)),
        [root, dir, base] => crate::delta::link(Path::new(root), Path::new(dir), Path::new(base)).or_else(|error| {
            eprintln!("prep link: linking every unit again: {error}");
            crate::delta::clear(Path::new(dir))?;
            whole(Path::new(root), Path::new(dir))
        }),
        _ => Err(Error(USAGE.to_string())),
    }
}

/// Links every unit under `root` into the library in `dir`, and writes its manifest and the list of its units beside it.
fn whole(root: &Path, dir: &Path) -> Result<()> {
    let units = units(root)?;
    let inputs = inputs(&units, &units)?;
    let out = dir.join(files::LIBRARY);
    link(&inputs, &out, Against::Sysroot)?;
    let counts = merge(&inputs, &out, (&dir.join(files::manifest(files::LIBRARY)), &[]))?;
    crate::delta::write_units(&units, &dir.join(files::UNITS))?;
    report(&inputs, &out, counts)
}

/// The directories of the units under `root` that a linked crate of the package links, or of every unit when no crate of it links.
pub fn units(root: &Path) -> Result<Vec<PathBuf>> {
    let mut dirs: Vec<PathBuf> = fs::read_dir(root)?.filter_map(|entry| entry.ok().map(|entry| entry.path())).collect();
    dirs.sort();
    let mut linked = HashSet::new();
    for dir in dirs.iter().filter(|dir| dir.join(LINK).exists()) {
        let text = fs::read_to_string(dir.join(LINK))?;
        linked.extend(text.lines().filter_map(|line| line.strip_prefix(record::UNIT)?.strip_prefix(' ')).map(str::to_string));
    }
    let wanted = |dir: &PathBuf| linked.is_empty() || dir.join(LINK).exists() || dir.file_name().and_then(|name| name.to_str()).is_some_and(|name| linked.contains(name));
    Ok(dirs.iter().filter(|dir| dir.join(MANIFEST).exists() && wanted(dir)).cloned().collect())
}

/// The inputs of a link of the units in `dirs`, with what the linking units of `all` link, each object rebound so the library exports every recorded instance and keeps its release copy.
pub fn inputs(dirs: &[PathBuf], all: &[PathBuf]) -> Result<Inputs> {
    let mut inputs = Inputs::default();
    for dir in dirs {
        inputs.manifests.push(dir.join(MANIFEST));
        inputs.objects.extend(objects(dir)?);
        inputs.released.extend(objects(&dir.join(RELEASE))?);
    }
    for dir in all.iter().filter(|dir| dir.join(LINK).exists()) {
        libraries(&dir.join(LINK), &mut inputs)?;
    }
    inputs.objects = without_allocator(&inputs.objects)?;
    inputs.released = without_allocator(&inputs.released)?;
    let symbols = symbols(&inputs.manifests)?;
    inputs.copies = released(&inputs.released, &symbols)?;
    for object in inputs.released.iter().chain(&inputs.objects) {
        inputs.rebound += elf::rebind(object, &symbols)?;
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

/// Links `inputs` into the shared library `out` against the toolchain's shared libstd, so the process holds one std, and against what `against` names.
pub fn link(inputs: &Inputs, out: &Path, against: Against<'_>) -> Result<()> {
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let mut command = Command::new(cc);
    command.args(["-shared", "-Wl,-Bsymbolic", "-o"]).arg(out);
    command.args(&inputs.released).args(&inputs.objects);
    command.args(&inputs.dylibs);
    let dirs: BTreeSet<_> = inputs.dylibs.iter().filter_map(|dylib| Path::new(dylib).parent()).collect();
    command.args(dirs.iter().map(|dir| format!("-Wl,-rpath,{}", dir.display())));
    match against {
        Against::Sysroot => {
            command.arg("-Wl,--start-group").args(&inputs.alone).arg("-Wl,--end-group");
        }
        Against::Base(dir) => {
            command.arg(format!("-L{}", dir.display())).arg(format!("-l:{}", files::LIBRARY)).args(AGAINST_BASE);
            command.args(lld());
        }
    }
    command.args(inputs.libs.iter().map(|lib| format!("-l{lib}")));
    let status = command.status()?;
    if !status.success() {
        return Err(Error(format!("the linker failed: {status}")));
    }
    Ok(())
}

/// The options that make the C compiler link with the toolchain's own lld, which links a small library against a large one several times faster than the system linker; none when the toolchain has none.
fn lld() -> Vec<String> {
    let (Some(sysroot), Some(target)) = (option_env!("PREP_SYSROOT"), option_env!("PREP_TARGET")) else { return Vec::new() };
    let dir = Path::new(sysroot).join(format!("lib/rustlib/{target}/bin/gcc-ld"));
    if !dir.join(LLD).is_file() {
        return Vec::new();
    }
    vec![format!("-B{}", dir.display()), "-fuse-ld=lld".to_string()]
}

/// Prints what the link of `inputs` into `out` took and kept.
pub fn report(inputs: &Inputs, out: &Path, (kept, missing): (usize, usize)) -> Result<()> {
    let size = fs::metadata(out)?.len();
    let counts = format!("units {} objects {} released {}", inputs.manifests.len(), inputs.objects.len() + inputs.released.len(), inputs.copies.len());
    println!("{counts} rebound {} records {kept} missing {missing} bytes {size}", inputs.rebound);
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

/// Writes to `out` the manifest of `library`, linked from `inputs`, with the units whose records of an earlier library's manifest it replaces, and returns how many records it kept and dropped.
pub fn merge(inputs: &Inputs, library: &Path, (out, replaces): (&Path, &[String])) -> Result<(usize, usize)> {
    let defined = elf::defined(library, Table::Dynamic, Scope::All)?;
    let mut shared = HashSet::new();
    for dylib in &inputs.dylibs {
        shared.extend(elf::defined(Path::new(dylib), Table::Dynamic, Scope::All)?);
    }
    let mut merged = Merged::default();
    for manifest in &inputs.manifests {
        let unit = manifest.parent().and_then(|dir| dir.file_name()).map(|name| name.to_string_lossy().into_owned()).unwrap_or_default();
        for line in BufReader::new(File::open(manifest)?).lines() {
            merged.add(serde_json::from_str(&line?)?, &unit, (&defined, &shared, &inputs.copies));
        }
    }
    merged.write(out, replaces)
}

/// A merged manifest being built: the records kept, the library facts, each distinct line describing a type or a static, and how many records name a symbol the library lacks.
#[derive(Default)]
struct Merged {
    records: Vec<Value>,
    library: serde_json::Map<String, Value>,
    indirect: [BTreeMap<String, Value>; 3],
    missing: usize,
}

impl Merged {
    /// Adds a line of the manifest of `unit`: a record when the library defines its symbol, or the shared standard library for a record of its code, a type or static line once with every unit that has it, library facts otherwise.
    fn add(&mut self, mut record: Value, unit: &str, (defined, shared, copies): (&HashSet<String>, &HashSet<String>, &HashSet<String>)) {
        if let Some((kind, name)) = INDIRECT.iter().enumerate().find_map(|(at, kind)| Some((at, record.get(*kind)?.as_str()?.to_string()))) {
            let line = self.indirect[kind].entry(name).or_insert_with(|| {
                record[record::USED_BY] = Value::Array(Vec::new());
                record
            });
            if let Some(units) = line[record::USED_BY].as_array_mut() {
                units.push(Value::String(unit.to_string()));
            }
            return;
        }
        let Some(symbol) = record.get("symbol").and_then(Value::as_str).map(str::to_string) else {
            let facts = record.get(record::LIBRARY).and_then(Value::as_object).cloned().unwrap_or_default();
            self.library.extend(facts.into_iter().filter(|(_, value)| !value.is_null()));
            return;
        };
        let upstream = record.get(record::UPSTREAM).and_then(Value::as_bool) == Some(true);
        if !defined.contains(&symbol) && !(upstream && shared.contains(&symbol)) {
            self.missing += 1;
            return;
        }
        record["release_copy"] = Value::Bool(copies.contains(&symbol));
        record[record::UNIT] = Value::String(unit.to_string());
        self.records.push(record);
    }

    /// Writes the merged manifest to `out` in TOML, naming the units in `replaces` when there are any, and returns how many records it kept and dropped.
    fn write(self, out: &Path, replaces: &[String]) -> Result<(usize, usize)> {
        let kept = self.records.len();
        let mut document = serde_json::Map::new();
        document.insert(record::LIBRARY.to_string(), Value::Object(self.library));
        document.insert(record::RECORD.to_string(), Value::Array(self.records));
        for (kind, lines) in INDIRECT.iter().zip(self.indirect) {
            document.insert(kind.to_string(), Value::Array(lines.into_values().collect()));
        }
        if !replaces.is_empty() {
            document.insert(record::REPLACES.to_string(), Value::Array(replaces.iter().cloned().map(Value::String).collect()));
        }
        let text = toml::to_string(&without_nulls(Value::Object(document))).map_err(|error| Error(error.to_string()))?;
        fs::write(out, text)?;
        Ok((kept, self.missing))
    }
}

fn without_nulls(value: Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(map.into_iter().filter(|(_, value)| !value.is_null()).map(|(key, value)| (key, without_nulls(value))).collect()),
        Value::Array(items) => Value::Array(items.into_iter().map(without_nulls).collect()),
        other => other,
    }
}

use std::collections::{BTreeSet, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use rustc_attr_parsing::eval_config_entry;
use rustc_middle::ty::TyCtxt;
use rustc_attr_ir::NativeLibKind;
use serde_json::Value;

use crate::elf;
use crate::error::{Error, Result};
use crate::record::{LINK, MANIFEST};
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

/// Writes the sysroot libraries and native libraries that the crate being compiled links, one per line.
pub fn write_libraries(tcx: TyCtxt<'_>, path: &Path) -> Result<()> {
    let sysroot = tcx.sess.opts.sysroot.path();
    let mut out = BufWriter::new(File::create(path)?);
    for &cnum in tcx.crates(()) {
        let source = tcx.used_crate_source(cnum);
        for (kind, path) in [("rlib", &source.rlib), ("dylib", &source.dylib)] {
            if let Some(path) = path
                && path.starts_with(sysroot)
            {
                writeln!(out, "{kind} {}", path.display())?;
            }
        }
        for lib in tcx.native_libraries(cnum) {
            let relevant = lib.cfg.as_ref().is_none_or(|cfg| eval_config_entry(tcx.sess, cfg).as_bool());
            if relevant && matches!(lib.kind, NativeLibKind::Dylib { .. } | NativeLibKind::Unspecified) {
                writeln!(out, "lib {}", lib.name)?;
            }
        }
    }
    out.flush()?;
    Ok(())
}

const ALLOCATOR_SUFFIX: &str = "___rust_alloc";

#[derive(Default)]
struct Inputs {
    objects: Vec<PathBuf>,
    released: Vec<PathBuf>,
    rlibs: BTreeSet<String>,
    dylibs: BTreeSet<String>,
    libs: Vec<String>,
    manifests: Vec<PathBuf>,
}

fn run(args: &[String]) -> Result<()> {
    let (root, out, std) = match args {
        [root, out] => (root, out, Std::Shared),
        [root, out, flag, kind] if flag == "--std" && kind == "static" => (root, out, Std::Static),
        [root, out, flag, kind] if flag == "--std" && kind == "shared" => (root, out, Std::Shared),
        _ => return Err(Error("usage: prep-driver link <prep directory> <library> [--std shared|static]".to_string())),
    };
    let mut inputs = gather(Path::new(root))?;
    if std == Std::Shared {
        inputs.objects = without_allocator(&inputs.objects)?;
        inputs.released = without_allocator(&inputs.released)?;
    }
    let symbols = symbols(&inputs.manifests)?;
    let released = released(&inputs.released, &symbols)?;
    let list = Path::new(root).join("released.txt");
    fs::write(&list, released.iter().map(|name| format!("{name}\n")).collect::<String>())?;
    let globalize = format!("--globalize-symbols={}", list.display());
    let mut rebound = 0;
    for object in &inputs.released {
        objcopy(object, &[&globalize])?;
        rebound += elf::rebind(object, &symbols)?;
    }
    for object in &inputs.objects {
        rebound += elf::rebind(object, &symbols)?;
    }
    link(&inputs, Path::new(out), std)?;
    let defined = dynamic(Path::new(out))?;
    let (kept, missing) = merge(&inputs.manifests, (&defined, &released), &PathBuf::from(format!("{out}.manifest.jsonl")))?;
    let size = fs::metadata(out)?.len();
    let counts = format!("units {} objects {} released {}", inputs.manifests.len(), inputs.objects.len() + inputs.released.len(), released.len());
    println!("{counts} rebound {rebound} records {kept} missing {missing} bytes {size}");
    Ok(())
}

fn gather(root: &Path) -> Result<Inputs> {
    let mut inputs = Inputs::default();
    let mut dirs: Vec<PathBuf> = fs::read_dir(root)?.filter_map(|entry| entry.ok().map(|entry| entry.path())).collect();
    dirs.sort();
    for dir in dirs.iter().filter(|dir| dir.join(MANIFEST).exists()) {
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
            Some(("rlib", rlib)) => {
                inputs.rlibs.insert(rlib.to_string());
            }
            Some(("dylib", dylib)) => {
                inputs.dylibs.insert(dylib.to_string());
            }
            Some(("lib", lib)) if !inputs.libs.iter().any(|known| known == lib) => inputs.libs.push(lib.to_string()),
            _ => {}
        }
    }
    Ok(())
}

/// Which copy of std the prep library links: the toolchain's shared libstd, so the process holds one std, or the static rlibs.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Std {
    Shared,
    Static,
}

fn link(inputs: &Inputs, out: &Path, std: Std) -> Result<()> {
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let mut command = Command::new(cc);
    command.args(["-shared", "-Wl,-Bsymbolic", "-o"]).arg(out);
    command.args(&inputs.released).args(&inputs.objects);
    if std == Std::Shared {
        command.args(&inputs.dylibs);
        let dirs: BTreeSet<_> = inputs.dylibs.iter().filter_map(|dylib| Path::new(dylib).parent()).collect();
        command.args(dirs.iter().map(|dir| format!("-Wl,-rpath,{}", dir.display())));
    } else {
        command.arg("-Wl,--start-group").args(&inputs.rlibs).arg("-Wl,--end-group");
    }
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
        if !nm(object, &["--defined-only"])?.iter().any(|name| name.ends_with(ALLOCATOR_SUFFIX)) {
            kept.push(object.clone());
        }
    }
    Ok(kept)
}

fn objcopy(object: &Path, flags: &[&str]) -> Result<()> {
    let status = Command::new("objcopy").args(flags).arg(object).status()?;
    if !status.success() {
        return Err(Error(format!("objcopy {} failed: {status}", object.display())));
    }
    Ok(())
}

fn released(objects: &[PathBuf], symbols: &HashSet<String>) -> Result<HashSet<String>> {
    let mut found = HashSet::new();
    for object in objects {
        found.extend(nm(object, &["--defined-only"])?.into_iter().filter(|name| symbols.contains(name)));
    }
    Ok(found)
}

fn dynamic(library: &Path) -> Result<HashSet<String>> {
    nm(library, &["-D", "--defined-only"])
}

fn nm(path: &Path, flags: &[&str]) -> Result<HashSet<String>> {
    let output = Command::new("nm").args(flags).arg(path).output()?;
    if !output.status.success() {
        return Err(Error(format!("nm {} failed: {}", path.display(), output.status)));
    }
    let text = String::from_utf8_lossy(&output.stdout);
    Ok(text.lines().filter_map(|line| line.split_whitespace().nth(2)).map(str::to_string).collect())
}

fn merge(manifests: &[PathBuf], (defined, released): (&HashSet<String>, &HashSet<String>), out: &Path) -> Result<(usize, usize)> {
    let mut writer = BufWriter::new(File::create(out)?);
    let (mut kept, mut missing) = (0, 0);
    for manifest in manifests {
        for line in BufReader::new(File::open(manifest)?).lines() {
            let mut record: Value = serde_json::from_str(&line?)?;
            let symbol = record.get("symbol").and_then(Value::as_str).map(str::to_string);
            if let Some(symbol) = symbol {
                if !defined.contains(&symbol) {
                    missing += 1;
                    continue;
                }
                kept += 1;
                record["release_copy"] = Value::Bool(released.contains(&symbol));
            }
            writeln!(writer, "{record}")?;
        }
    }
    writer.flush()?;
    Ok((kept, missing))
}

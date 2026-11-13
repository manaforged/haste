use std::os::unix::fs::MetadataExt as _;
use std::path::Path;

use anyhow::{Context as _, Result};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use serde::Deserialize;

use super::files;

/// The manifest that the prep driver writes beside its library: the library's own facts, one record per function instance it exports, each trait object and function pointer type that rustc's code makes, calls or hands to the shared standard library, and each static that holds state.
#[derive(Deserialize)]
pub(super) struct Library {
    #[serde(default)]
    pub library: Facts,
    #[serde(default)]
    pub record: Vec<Record>,
    #[serde(default)]
    pub object: Vec<Object>,
    #[serde(default)]
    pub pointer: Vec<Pointer>,
    #[serde(default, rename = "static")]
    pub statics: Vec<Static>,
    #[serde(default)]
    pub replaces: Vec<String>,
}

/// A static that holds state, which the library defines once: its key, symbol, the layouts its value holds or points to, whether each thread has its own, and the units whose code uses it.
#[derive(Deserialize)]
pub(super) struct Static {
    #[serde(rename = "static")]
    pub key: String,
    pub symbol: String,
    pub layouts: Option<Vec<String>>,
    pub thread_local: bool,
    #[serde(default)]
    pub used_by: Vec<String>,
}

/// What the driver found about the library as a whole: the symbols of the shared standard library's argument initializer, of its `stdout`, and of `flush` on the `Stdout` that returns, and whether rustc's code touches a trait object or function pointer type that has no key.
#[derive(Deserialize, Default)]
pub(super) struct Facts {
    pub args_init: Option<String>,
    pub stdout: Option<String>,
    pub stdout_flush: Option<String>,
    #[serde(default)]
    pub unkeyed: bool,
}

/// A trait object type that rustc's code touches: its key, each method slot of its vtable as rustc lays it out, each trait whose methods its vtable holds, and the units whose code touches it.
#[derive(Deserialize)]
pub(super) struct Object {
    pub object: String,
    #[serde(default)]
    pub slots: Vec<Slot>,
    #[serde(default)]
    pub supers: Vec<Super>,
    #[serde(default)]
    pub used_by: Vec<String>,
}

/// One method slot of a vtable: its index, the key of the trait method it holds, the convention of a call through it, and the layouts that call crosses.
#[derive(Deserialize)]
pub(super) struct Slot {
    pub at: usize,
    pub method: Option<String>,
    pub abi: Abi,
    pub layouts: Option<Vec<String>>,
}

/// A trait whose methods a vtable holds: its key, and the slot that holds a pointer to its own vtable, or none when its vtable starts this one.
#[derive(Deserialize)]
pub(super) struct Super {
    #[serde(rename = "trait")]
    pub of: Option<String>,
    pub vptr: Option<usize>,
}

/// A function pointer type that rustc's code touches: its key, the convention of a call through it, the layouts that call crosses, and the units whose code touches it.
#[derive(Deserialize)]
pub(super) struct Pointer {
    pub pointer: String,
    pub abi: Abi,
    pub layouts: Option<Vec<String>>,
    #[serde(default)]
    pub used_by: Vec<String>,
}

/// One function instance in the library, from one unit: its symbol, the key that names it, the stable id of the crate that defines it, where it and the functions rustc inlined into it are defined, its definition path, closure identity, type arguments, calling convention, the layouts it crosses with, what it calls, the statics holding state it touches, the keys of the function pointer types it calls through (none when one has no key), whether it reaches standard library state each std keeps apart, and whether rustc's release build kept this copy.
#[derive(Deserialize)]
pub(super) struct Record {
    pub symbol: String,
    pub key: Option<String>,
    pub crate_id: String,
    pub at: At,
    pub kind: InstanceKind,
    pub path: String,
    pub closure: Option<ClosureId>,
    pub args: Vec<String>,
    pub abi: Abi,
    pub layouts: Option<Vec<String>>,
    pub inlined: Vec<At>,
    pub uses: Vec<String>,
    #[serde(default)]
    pub statics: Vec<String>,
    pub calls: Option<Vec<String>>,
    pub held: bool,
    pub release_copy: bool,
    #[serde(default)]
    pub leaf: bool,
    #[serde(default)]
    pub unit: String,
}

/// Where a definition is: its crate, and its file and the byte offset of its name unless a macro expansion wrote it.
#[derive(Deserialize)]
pub(super) struct At {
    #[serde(rename = "crate")]
    pub krate: String,
    pub file: Option<String>,
    pub lo: Option<u32>,
}

/// What rustc generated an instance's code from: a function's own body, a closure's by-value call shim, a drop glue, or another shim.
#[derive(Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum InstanceKind {
    Item,
    #[serde(rename = "Shim::ClosureOnce")]
    ClosureOnce,
    #[serde(rename = "Shim::DropGlue")]
    DropGlue,
    #[serde(other)]
    Other,
}

/// What a closure record says of its closure: the call trait it implements at most.
#[derive(Deserialize)]
pub(super) struct ClosureId {
    pub kind: Option<ClosureKind>,
}

/// The call trait a closure implements at most.
#[derive(Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum ClosureKind {
    Fn,
    FnMut,
    FnOnce,
}

/// The calling convention rustc lowered an instance to.
#[derive(Deserialize, PartialEq, Eq)]
pub(super) struct Abi {
    pub conv: Conv,
    pub variadic: bool,
    pub ret: Arg,
    pub args: Vec<Arg>,
}

/// The convention of an instance: the Rust one, or another that Haste never binds.
#[derive(Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Conv {
    Rust,
    #[serde(other)]
    Other,
}

/// One argument or the return: the key of its type, its size, alignment and passing mode.
#[derive(Deserialize, PartialEq, Eq)]
pub(super) struct Arg {
    pub key: Option<String>,
    pub size: u64,
    pub align: u64,
    pub pass: Pass,
}

/// How rustc passes one value.
#[derive(Deserialize, PartialEq, Eq, Debug)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub(super) enum Pass {
    Ignore,
    Direct { regs: Vec<Reg> },
    Pair { regs: Vec<Reg> },
    Cast {},
    Indirect {
        on_stack: bool,
        #[serde(rename = "unsized")]
        dynamic: bool,
    },
}

/// One register a value or a part of it travels in, by class and byte offset in the value.
#[derive(Deserialize, PartialEq, Eq, Debug)]
pub(super) struct Reg {
    pub class: Class,
    pub offset: u64,
}

/// The class of one register part: an integer of a width and signedness, a float, a pointer, a vector, or a class Haste has no machine type for.
#[derive(Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(rename_all = "lowercase")]
pub(super) enum Class {
    I8,
    I16,
    I32,
    I64,
    I128,
    U8,
    U16,
    U32,
    U64,
    U128,
    F32,
    F64,
    Ptr,
    Vector,
    #[serde(other)]
    Other,
}

/// A manifest read before it was needed: the device and inode of its file, and what it holds.
pub(super) struct Parsed {
    file: (u64, u64),
    library: Library,
}

impl Parsed {
    /// The manifest at `path`, with the identity of the file read.
    pub(super) fn read(path: &Path) -> Result<Parsed> {
        let mut opened = std::fs::File::open(path).with_context(|| format!("reading {}", path.display()))?;
        let meta = opened.metadata()?;
        let mut text = String::new();
        std::io::Read::read_to_string(&mut opened, &mut text).with_context(|| format!("reading {}", path.display()))?;
        let library = toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        Ok(Parsed { file: (meta.dev(), meta.ino()), library })
    }
}

impl Library {
    /// The manifest at `path`.
    pub(super) fn read(path: &Path) -> Result<Library> {
        Parsed::read(path).map(|parsed| parsed.library)
    }

    /// The manifest of what the libraries in `dir` hold: the first library's, taken from `ahead` when that read the same file, with the records and statics of the units a second library beside it replaces taken from that one.
    pub(super) fn load(dir: &Path, ahead: Option<Parsed>) -> Result<Library> {
        let path = dir.join(files::manifest(files::LIBRARY));
        let file = std::fs::metadata(&path).with_context(|| format!("reading {}", path.display()))?;
        let mut library = match ahead {
            Some(parsed) if parsed.file == (file.dev(), file.ino()) => parsed.library,
            _ => Library::read(&path)?,
        };
        let delta = dir.join(files::manifest(files::DELTA));
        if delta.is_file() {
            library.layer(Library::read(&delta)?);
        }
        Ok(library)
    }

    /// This manifest with `delta` over it: its records of the units `delta` replaces dropped and `delta`'s added after the rest, each type and static line kept while a unit `delta` does not replace uses it and taken from `delta` when it states one, and its facts kept where `delta` states none.
    fn layer(&mut self, delta: Library) {
        let replaced: HashSet<&str> = delta.replaces.iter().map(String::as_str).collect();
        self.record.retain(|record| !replaced.contains(record.unit.as_str()));
        self.record.extend(delta.record);
        overlay(&mut self.statics, delta.statics, &replaced);
        overlay(&mut self.object, delta.object, &replaced);
        overlay(&mut self.pointer, delta.pointer, &replaced);
        let (facts, more) = (&mut self.library, delta.library);
        facts.args_init = more.args_init.or(facts.args_init.take());
        facts.stdout = more.stdout.or(facts.stdout.take());
        facts.stdout_flush = more.stdout_flush.or(facts.stdout_flush.take());
        facts.unkeyed |= more.unkeyed;
    }
}

/// A type or static line of a manifest, which several units may use.
trait Shared {
    /// The key that names it.
    fn name(&self) -> &str;
    /// The units whose code uses it.
    fn used_by(&mut self) -> &mut Vec<String>;
}

impl Shared for Static {
    fn name(&self) -> &str {
        &self.key
    }
    fn used_by(&mut self) -> &mut Vec<String> {
        &mut self.used_by
    }
}

impl Shared for Object {
    fn name(&self) -> &str {
        &self.object
    }
    fn used_by(&mut self) -> &mut Vec<String> {
        &mut self.used_by
    }
}

impl Shared for Pointer {
    fn name(&self) -> &str {
        &self.pointer
    }
    fn used_by(&mut self) -> &mut Vec<String> {
        &mut self.used_by
    }
}

/// The lines of `base` that a unit not in `replaced` still uses, each replaced by the line of `delta` with its name, with every other line of `delta` after them.
fn overlay<T: Shared>(base: &mut Vec<T>, delta: Vec<T>, replaced: &HashSet<&str>) {
    for line in base.iter_mut() {
        line.used_by().retain(|unit| !replaced.contains(unit.as_str()));
    }
    base.retain_mut(|line| !line.used_by().is_empty());
    let mut at: HashMap<String, usize> = base.iter().enumerate().map(|(index, line)| (line.name().to_owned(), index)).collect();
    for mut line in delta {
        match at.get(line.name()) {
            Some(&index) => {
                let kept = std::mem::take(base[index].used_by());
                line.used_by().extend(kept);
                base[index] = line;
            }
            None => {
                at.insert(line.name().to_owned(), base.len());
                base.push(line);
            }
        }
    }
}

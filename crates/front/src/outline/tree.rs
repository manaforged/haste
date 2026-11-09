use std::collections::BTreeSet;

use rustc_hash::FxHashMap as HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{Result, anyhow};

use super::item::{Attr, Items, Vis};
use super::rules::Rule;
use super::scan;
use super::source::Source;
use crate::graph::cfg::{Cfg, Target};

/// A lexed and scanned file.
#[derive(Debug)]
pub struct File {
    pub source: Source,
    pub items: Items,
    pub expanded: bool,
    bodies: OnceLock<Box<[OnceLock<Option<Box<Expansion>>>]>>,
}

/// A body read in full: its file with the expansions of its macro calls laid after it and the items it declares, and the braced block that each call's opening group became.
#[derive(Debug)]
pub struct Expansion {
    pub file: File,
    pub calls: HashMap<u32, u32>,
}

impl File {
    /// A file of `source` scanned to `items`, with no body expanded yet.
    pub fn new(source: Source, items: Items, expanded: bool) -> File {
        File { source, items, expanded, bodies: OnceLock::new() }
    }

    /// The expansion of the body of item `item`, once a pass read it and found macro calls or items in it.
    pub fn expansion(&self, item: u32) -> Option<&Expansion> {
        self.bodies.get()?.get(item as usize)?.get()?.as_deref()
    }

    /// Whether a pass already read the body of item `item`.
    pub fn body_read(&self, item: u32) -> bool {
        self.bodies.get().and_then(|slots| slots.get(item as usize)).is_some_and(|slot| slot.get().is_some())
    }

    /// Records what reading the body of item `item` found, unless a pass already did.
    pub fn set_expansion(&self, item: u32, found: Option<Box<Expansion>>) {
        let slots = self.bodies.get_or_init(|| (0..self.items.items.len()).map(|_| OnceLock::new()).collect());
        if let Some(slot) = slots.get(item as usize) {
            let _ = slot.set(found);
        }
    }
}

/// A `macro_rules!` definition and the crate that defines it.
#[derive(Debug)]
pub struct Macro {
    pub file: Arc<File>,
    pub rules: Vec<Rule>,
    pub krate: usize,
}

/// The macros in textual scope at one point: a definition and the scope before it.
#[derive(Debug)]
pub struct Textual {
    pub name: String,
    pub def: Arc<Macro>,
    pub next: Option<Arc<Textual>>,
}

/// An item macro call that waits for its definition, with the textual scope it was written in.
#[derive(Clone, Debug)]
pub struct Waiting {
    pub file: Arc<File>,
    pub owner: usize,
    pub at: u32,
    pub scope: Option<Arc<Textual>>,
}

/// A derive or attribute that may name a procedural macro, on item `at` of `file` in module `owner`, written by the path from token `path.0` to token `path.1`.
#[derive(Clone, Debug)]
pub struct Held {
    pub file: Arc<File>,
    pub owner: usize,
    pub at: u32,
    pub path: (u32, u32),
}

/// The macro named `name` in textual scope `scope`.
pub fn textual(mut scope: Option<&Arc<Textual>>, name: &str) -> Option<Arc<Macro>> {
    while let Some(entry) = scope {
        if entry.name == name {
            return Some(entry.def.clone());
        }
        scope = entry.next.as_ref();
    }
    None
}

/// Where the items of a declared module are.
#[derive(Clone, Debug)]
pub enum Decl {
    Inline { file: usize, at: u32 },
    File { candidates: Vec<PathBuf>, by_attr: bool },
}

/// Whether a module's items are read.
#[derive(Clone, Debug)]
pub enum State {
    Declared(Decl),
    Loaded,
    Missing,
    Gated,
}

/// One module: a whole file, or an inline `mod` item inside a file.
#[derive(Clone, Debug)]
pub struct Module {
    pub file: usize,
    pub item: Option<u32>,
    pub parent: Option<usize>,
    pub name: String,
    pub dir: PathBuf,
    pub parts: Vec<usize>,
    pub vis: Vis,
    pub macro_use: bool,
    pub state: State,
    pub textual: Option<Arc<Textual>>,
}

impl Module {
    /// Whether the module's items are read.
    pub fn loaded(&self) -> bool {
        matches!(self.state, State::Loaded)
    }
}

/// One impl block: its file and item, the module around it, and the expansions of the macro calls among its items.
#[derive(Clone, Debug)]
pub struct Impl {
    pub file: usize,
    pub at: u32,
    pub module: usize,
    pub parts: Vec<usize>,
}

/// The module tree of one crate, read as far as lookups needed it.
#[derive(Debug, Default)]
pub struct Crate {
    pub env: Env,
    pub files: Vec<Arc<File>>,
    pub modules: Vec<Module>,
    pub missing: Vec<PathBuf>,
    pub gated: Vec<PathBuf>,
    pub errors: Vec<String>,
    pub exports: HashMap<String, Arc<Macro>>,
    pub defined: HashMap<(u32, u32), Arc<Macro>>,
    pub deferred: Vec<Waiting>,
    pub impls: Vec<Impl>,
    pub members: Vec<Waiting>,
    pub fallback: Vec<usize>,
    pub foreign: Vec<Waiting>,
    pub derived: Vec<Held>,
    pub attributed: Vec<Held>,
    pub consumed: rustc_hash::FxHashSet<(u32, u32)>,
    pub macro_use: Vec<usize>,
    pub expansions: usize,
}

impl Crate {
    /// Whether some declared module is not read yet.
    pub fn partial(&self) -> bool {
        self.modules.iter().any(|module| matches!(module.state, State::Declared(_)))
    }

    /// The names of the item macro calls that no definition matched.
    pub fn unresolved(&self) -> Vec<String> {
        let names = self.deferred.iter().filter_map(|call| call.file.items.items[call.at as usize].name.map(|name| call.file.source.text(name).to_owned()));
        names.collect()
    }
}

/// Whether a cfg holds, does not hold, or depends on values the front end does not know yet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Truth {
    Yes,
    No,
    Unknown,
}

/// The cfg values of a crate: the target, the unit's features, the profile and what the unit's build script set; `known` is false while a build script that may set more has not run.
#[derive(Debug, Default)]
pub struct Env {
    pub target: Target,
    pub features: BTreeSet<String>,
    pub script: Option<std::sync::Arc<crate::graph::script::Output>>,
    pub known: bool,
}

const PROFILE: [&str; 3] = ["debug_assertions", "overflow_checks", "ub_checks"];
const OFF: [&str; 10] = ["test", "doc", "doctest", "miri", "rustfmt", "clippy", "windows", "wasm", "bootstrap", "docsrs"];

impl Env {
    /// Whether `cfg` may hold here: true unless this environment rules it out.
    pub fn allows(&self, cfg: &Cfg) -> bool {
        self.eval(cfg) != Truth::No
    }

    pub fn eval(&self, cfg: &Cfg) -> Truth {
        match cfg {
            Cfg::Name(name) => self.name(name),
            Cfg::KeyValue(key, value) => self.pair(key, value),
            Cfg::All(all) => all.iter().map(|cfg| self.eval(cfg)).fold(Truth::Yes, and),
            Cfg::Any(any) => any.iter().map(|cfg| self.eval(cfg)).fold(Truth::No, or),
            Cfg::Not(cfg) => not(self.eval(cfg)),
        }
    }

    fn name(&self, name: &str) -> Truth {
        if self.target.has_name(name) || PROFILE.contains(&name) || self.script.as_ref().is_some_and(|set| set.names.contains(name)) {
            Truth::Yes
        } else if self.known || OFF.contains(&name) {
            Truth::No
        } else {
            Truth::Unknown
        }
    }

    fn pair(&self, key: &str, value: &str) -> Truth {
        if key == "feature" {
            truth(self.features.contains(value))
        } else if key.starts_with("target_") || matches!(key, "panic" | "fmt_debug" | "relocation_model") {
            truth(self.target.has_pair(key, value))
        } else if self.script.as_ref().is_some_and(|set| set.pairs.contains(&(key.to_owned(), value.to_owned()))) {
            Truth::Yes
        } else if self.known {
            Truth::No
        } else {
            Truth::Unknown
        }
    }

    /// Whether every `cfg` attribute in `attrs`, including those `cfg_attr` enables, may hold.
    pub fn enabled(&self, attrs: &[Attr]) -> bool {
        attrs.iter().all(|attr| match attr {
            Attr::Cfg(cfg) => self.allows(cfg),
            Attr::CfgAttr(cfg, inner) => self.eval(cfg) == Truth::No || self.enabled(inner),
            _ => true,
        })
    }

    /// The `#[path]` value in `attrs`, including one that `cfg_attr` may enable.
    pub fn path<'a>(&self, attrs: &'a [Attr]) -> Option<&'a str> {
        attrs.iter().find_map(|attr| match attr {
            Attr::Path(path) => Some(path.as_str()),
            Attr::CfgAttr(cfg, inner) if self.allows(cfg) => self.path(inner),
            _ => None,
        })
    }
}

fn truth(value: bool) -> Truth {
    if value { Truth::Yes } else { Truth::No }
}

fn and(left: Truth, right: Truth) -> Truth {
    match (left, right) {
        (Truth::No, _) | (_, Truth::No) => Truth::No,
        (Truth::Yes, Truth::Yes) => Truth::Yes,
        _ => Truth::Unknown,
    }
}

fn or(left: Truth, right: Truth) -> Truth {
    match (left, right) {
        (Truth::Yes, _) | (_, Truth::Yes) => Truth::Yes,
        (Truth::No, Truth::No) => Truth::No,
        _ => Truth::Unknown,
    }
}

fn not(value: Truth) -> Truth {
    match value {
        Truth::Yes => Truth::No,
        Truth::No => Truth::Yes,
        Truth::Unknown => Truth::Unknown,
    }
}

type Slot = Arc<OnceLock<Result<Arc<File>, String>>>;

/// Files read and scanned once, shared by every crate that includes them.
#[derive(Default)]
pub struct Cache {
    slots: Mutex<HashMap<PathBuf, Slot>>,
}

const PER_WORKER: usize = 4;

fn slot(slots: &Mutex<HashMap<PathBuf, Slot>>, path: &Path) -> Result<Slot> {
    Ok(slots.lock().map_err(|_| anyhow!("file cache poisoned"))?.entry(path.to_owned()).or_default().clone())
}

fn fill(slot: &Slot, path: &Path) -> Result<Arc<File>, String> {
    slot.get_or_init(|| read(path).map(Arc::new).map_err(|error| format!("{error:#}"))).clone()
}

impl Cache {
    /// The scanned file at `path`, reading it on first use.
    pub fn file(&self, path: &Path) -> Result<Arc<File>> {
        fill(&slot(&self.slots, path)?, path).map_err(|error| anyhow!(error))
    }

    /// Reads and scans the existing files among `paths` on scoped threads that finish before it returns.
    pub fn read_all(&self, paths: &[PathBuf]) {
        let cores = std::thread::available_parallelism().map_or(1, |count| count.get());
        let workers = cores.min(paths.len() / PER_WORKER);
        if workers < 2 {
            return;
        }
        let next = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..workers {
                scope.spawn(|| {
                    while let Some(path) = paths.get(next.fetch_add(1, Ordering::Relaxed)) {
                        if exists(path) && let Ok(slot) = slot(&self.slots, path) {
                            let _ = fill(&slot, path);
                        }
                    }
                });
            }
        });
    }

    /// Forgets the files at `paths`, so that the next use reads them again.
    pub fn forget(&self, paths: &[PathBuf]) {
        if let Ok(mut slots) = self.slots.lock() {
            paths.iter().for_each(|path| _ = slots.remove(path));
        }
    }

    /// The number of files read.
    pub fn len(&self) -> usize {
        self.slots.lock().map_or(0, |slots| slots.len())
    }

    /// Whether no file was read.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The files that haste supplies itself, read from these paths: its `proc_macro` crate, which stands in for the toolchain's.
const EMBEDDED: [(&str, &str); 3] = [
    ("/haste/proc_macro/lib.rs", include_str!("../../proc_macro/lib.rs")),
    ("/haste/proc_macro/lex.rs", include_str!("../../proc_macro/lex.rs")),
    ("/haste/proc_macro/literal.rs", include_str!("../../proc_macro/literal.rs")),
];

/// The root file of the `proc_macro` crate that haste supplies.
pub const PROC_MACRO_ROOT: &str = EMBEDDED[0].0;

fn embedded(path: &Path) -> Option<&'static str> {
    EMBEDDED.iter().find(|(at, _)| Path::new(at) == path).map(|(_, text)| *text)
}

/// Whether there is a file at `path`, on disk or among the files haste supplies.
pub fn exists(path: &Path) -> bool {
    embedded(path).is_some() || path.is_file()
}

/// Reads and scans the file at `path`.
pub fn read(path: &Path) -> Result<File> {
    let source = match embedded(path) {
        Some(text) => Source::parse(path.to_owned(), text.to_owned())?,
        None => Source::read(path)?,
    };
    let items = scan::scan(&source)?;
    Ok(File::new(source, items, false))
}

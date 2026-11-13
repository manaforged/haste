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
    pub expand_origin: u64,
    bodies: OnceLock<Box<Bodies>>,
    index: OnceLock<Index>,
}

/// The body expansions that one crate read from a file, then those of the next crate that includes the same file, since a body expands to different paths in a host and a target build of one package.
#[derive(Debug)]
struct Bodies {
    krate: u32,
    slots: Box<[OnceLock<Option<Box<Expansion>>>]>,
    next: OnceLock<Box<Bodies>>,
}

impl Bodies {
    fn new(krate: u32, items: usize) -> Box<Bodies> {
        Box::new(Bodies { krate, slots: (0..items).map(|_| OnceLock::new()).collect(), next: OnceLock::new() })
    }

    fn of(&self, krate: u32) -> Option<&Bodies> {
        let mut at = self;
        while at.krate != krate {
            at = at.next.get()?;
        }
        Some(at)
    }

    fn claim(&self, krate: u32, items: usize) -> &Bodies {
        let mut at = self;
        while at.krate != krate {
            at = at.next.get_or_init(|| Bodies::new(krate, items));
        }
        at
    }
}

/// The items and imports of a file by name, each list in declaration order.
#[derive(Debug, Default)]
pub struct Index {
    /// The items that each item, or the top level for none, holds directly or in an extern block, by plain name.
    pub items: HashMap<Option<u32>, HashMap<Box<str>, Vec<u32>>>,
    /// The named imports by the name they bind.
    pub imports: HashMap<Box<str>, Vec<u32>>,
    /// The glob imports.
    pub globs: Vec<u32>,
    /// The functions and modules whose tokens hold each import.
    pub holders: Vec<Box<[u32]>>,
}

/// A body read in full: its file with the expansions of its macro calls laid after it and the items it declares, the braced block that each call's opening group became, and the block-scope tree of its items.
#[derive(Debug)]
pub struct Expansion {
    pub file: File,
    pub calls: HashMap<u32, u32>,
    pub scopes: Box<[Option<u32>]>,
    pub item_scopes: Box<[u32]>,
}

impl File {
    /// A file of `source` scanned to `items`, with no body expanded yet.
    pub fn new(source: Source, items: Items, expanded: bool) -> File {
        File { source, items, expanded, expand_origin: 0, bodies: OnceLock::new(), index: OnceLock::new() }
    }

    /// An expanded file carrying deterministic callsite provenance `expand_origin`.
    pub fn new_expanded(source: Source, items: Items, expand_origin: u64) -> File {
        File { source, items, expanded: true, expand_origin, bodies: OnceLock::new(), index: OnceLock::new() }
    }

    /// The name index of this file, made by `build` on first use and kept.
    pub fn index(&self, build: impl FnOnce(&File) -> Index) -> &Index {
        self.index.get_or_init(|| build(self))
    }

    fn body_slot(&self, krate: u32, item: u32) -> Option<&OnceLock<Option<Box<Expansion>>>> {
        self.bodies.get()?.of(krate)?.slots.get(item as usize)
    }

    /// The expansion of the body of item `item` in crate `krate`, once a pass read it and found macro calls or items in it.
    pub fn expansion(&self, krate: u32, item: u32) -> Option<&Expansion> {
        self.body_slot(krate, item)?.get()?.as_deref()
    }

    /// Whether a pass already read the body of item `item` in crate `krate`.
    pub fn body_read(&self, krate: u32, item: u32) -> bool {
        self.body_slot(krate, item).is_some_and(|slot| slot.get().is_some())
    }

    /// Records what reading the body of item `item` in crate `krate` found, unless a pass already did.
    pub fn set_expansion(&self, krate: u32, item: u32, found: Option<Box<Expansion>>) {
        let items = self.items.items.len();
        let bodies = self.bodies.get_or_init(|| Bodies::new(krate, items)).claim(krate, items);
        if let Some(slot) = bodies.slots.get(item as usize) {
            drop(slot.set(found));
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

/// Which associated-item block of a crate a macro call expands into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemberBlock {
    Impl(usize),
    Trait(usize),
}

/// One impl or trait block: its file and item, the module around it, and the expansions of the macro calls among its items.
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
    pub traits: Vec<Impl>,
    /// The file and item of each module-level `const _` and of each module-level function whose body holds an `impl`, whose impls apply crate-wide.
    pub impl_bodies: Vec<(usize, u32)>,
    pub members: Vec<(MemberBlock, Waiting)>,
    pub fallback: Vec<usize>,
    pub foreign: Vec<Waiting>,
    pub derived: Vec<Held>,
    pub attributed: Vec<Held>,
    pub consumed: rustc_hash::FxHashSet<(u32, u32)>,
    pub macro_use: Vec<usize>,
    pub expansions: usize,
}

/// What a loader of another crate sees of one crate: its exported macros and whether it may still export more.
#[derive(Debug, Default)]
pub struct View {
    pub exports: HashMap<String, Arc<Macro>>,
    pub partial: bool,
}

impl Crate {
    /// Whether this crate is an immutable release sysroot crate.
    pub const fn is_sysroot(&self) -> bool {
        self.env.release && self.env.fixed
    }

    /// The modules declared but not read yet.
    pub fn declared(&self) -> impl Iterator<Item = usize> + '_ {
        self.modules.iter().enumerate().filter(|(_, module)| matches!(module.state, State::Declared(_))).map(|(at, _)| at)
    }

    /// Whether some declared module is not read yet.
    pub fn partial(&self) -> bool {
        self.declared().next().is_some()
    }

    /// Whether the crate may still export more: a declared module is unread, or a loader holds the crate.
    pub fn unsettled(&self) -> bool {
        self.partial() || self.modules.is_empty()
    }

    /// What a loader of another crate sees of this crate now.
    pub fn view(&self) -> View {
        View { exports: self.exports.clone(), partial: self.unsettled() }
    }

    /// The files that may hold module `module` while a `mod` item without a body only declares it; empty otherwise.
    pub fn unread_files(&self, module: usize) -> &[PathBuf] {
        match &self.modules[module].state {
            State::Declared(Decl::File { candidates, .. }) => candidates,
            _ => &[],
        }
    }

    /// How much of the crate is read: its files, impls, item-holding bodies and consumed items, and whether a declared module is still unread; loading, expanding or consuming changes it.
    pub fn extent(&self) -> [usize; 5] {
        [self.files.len(), self.impls.len(), self.impl_bodies.len(), self.consumed.len(), usize::from(self.partial())]
    }

    /// The names of the item macro calls that no definition matched.
    pub fn unresolved(&self) -> Vec<String> {
        let names = self.deferred.iter().filter_map(|call| call.file.items.items[call.at as usize].name.map(|name| call.file.source.text(name).to_owned()));
        names.collect()
    }
}

/// Whether a cfg holds, does not hold, or depends on values not known yet, such as build script output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Truth {
    Yes,
    No,
    Unknown,
}

/// The cfg values and compile-time variables of a crate: the target, the unit's features, the profile, Cargo's variables and what the unit's build script set; `known` is false while a build script that may set more has not run.
#[derive(Debug, Default)]
pub struct Env {
    pub target: Target,
    pub features: BTreeSet<String>,
    pub script: Option<std::sync::Arc<crate::graph::script::Output>>,
    pub known: bool,
    pub vars: Vec<(String, String)>,
    pub release: bool,
    /// Whether the crate's sources cannot change while haste runs: the standard library, or a registry or git package.
    pub fixed: bool,
}

/// The build profile every crate compiles under: Cargo's dev profile, or its release profile.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Profile {
    #[default]
    Debug,
    Release,
}

pub(crate) const PROFILE: [&str; 3] = ["debug_assertions", "overflow_checks", "ub_checks"];
const RELEASE: [&str; 1] = ["ub_checks"];
const OFF: [&str; 10] = ["test", "doc", "doctest", "miri", "rustfmt", "clippy", "windows", "wasm", "bootstrap", "docsrs"];

impl Env {
    /// The value `env!` gives variable `name`: what the build script or Cargo set for this crate, else the process environment's.
    pub fn var(&self, name: &str) -> Option<String> {
        let set = self.script.iter().flat_map(|output| &output.vars).chain(&self.vars);
        set.filter(|(key, _)| key == name).map(|(_, value)| value.clone()).next().or_else(|| std::env::var(name).ok())
    }

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

    /// Whether rustc checks arithmetic for overflow in this crate's bodies.
    pub fn overflow_checks(&self) -> bool {
        self.profile().contains(&"overflow_checks")
    }

    /// The profile cfgs that hold: none under the release profile, else the debug profile, or for a sysroot crate the release build rustc ships, whose UB checks follow the user crate.
    fn profile(&self) -> &'static [&'static str] {
        match crate::profile() {
            Profile::Release => &[],
            Profile::Debug if self.release => &RELEASE,
            Profile::Debug => &PROFILE,
        }
    }

    fn name(&self, name: &str) -> Truth {
        if self.target.has_name(name) || self.profile().contains(&name) || self.script.as_ref().is_some_and(|set| set.names.contains(name)) {
            Truth::Yes
        } else if self.known || OFF.contains(&name) || (crate::profile() == Profile::Release && PROFILE.contains(&name)) {
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
        self.find_attr(attrs, Truth::Unknown, &|attr| {
            if let Attr::Path(path) = attr { Some(path.as_str()) } else { None }
        })
    }

    /// The first value that `pick` finds in `attrs`, including in those of each `cfg_attr` whose condition is `least` or `Yes`.
    pub fn find_attr<'a, T>(&self, attrs: &'a [Attr], least: Truth, pick: &impl Fn(&'a Attr) -> Option<T>) -> Option<T> {
        attrs.iter().find_map(|attr| match attr {
            Attr::CfgAttr(cfg, inner) if matches!((self.eval(cfg), least), (Truth::Yes, _) | (Truth::Unknown, Truth::Unknown)) => self.find_attr(inner, least, pick),
            attr => pick(attr),
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
pub struct Files {
    slots: Mutex<HashMap<PathBuf, Slot>>,
}

const PER_WORKER: usize = 4;

fn slot(slots: &Mutex<HashMap<PathBuf, Slot>>, path: &Path) -> Result<Slot> {
    Ok(slots.lock().map_err(|_| anyhow!("file cache poisoned"))?.entry(path.to_owned()).or_default().clone())
}

fn fill(slot: &Slot, path: &Path) -> Result<Arc<File>, String> {
    slot.get_or_init(|| read(path).map(Arc::new).map_err(|error| format!("{error:#}"))).clone()
}

impl Files {
    /// The scanned file at `path`, reading it on first use.
    pub fn file(&self, path: &Path) -> Result<Arc<File>> {
        fill(&slot(&self.slots, path)?, path).map_err(|error| anyhow!(error))
    }

    /// Reads and scans the existing files among `paths` on scoped threads that finish before it returns.
    pub fn read_all(&self, paths: &[PathBuf]) {
        let jobs = crate::jobs();
        let workers = jobs.min(paths.len() / PER_WORKER);
        if workers < 2 {
            return;
        }
        let next = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..workers {
                scope.spawn(|| {
                    while let Some(path) = paths.get(next.fetch_add(1, Ordering::Relaxed)) {
                        if exists(path)
                            && let Ok(slot) = slot(&self.slots, path)
                        {
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
}

/// The files that haste supplies itself, read from these paths: its `proc_macro` crate, which stands in for the toolchain's.
const EMBEDDED: [(&str, &str); 3] =
    [("/haste/proc_macro/lib.rs", include_str!("../../proc_macro/lib.rs")), ("/haste/proc_macro/lex.rs", include_str!("../../proc_macro/lex.rs")), ("/haste/proc_macro/literal.rs", include_str!("../../proc_macro/literal.rs"))];

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

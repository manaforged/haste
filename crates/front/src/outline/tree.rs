use std::collections::BTreeSet;

use rustc_hash::FxHashMap as HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Sender, channel};
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
    pub deferred: Vec<(Arc<File>, usize, u32)>,
    pub fallback: Vec<usize>,
    pub foreign: Vec<(Arc<File>, usize, u32)>,
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
        let names = self.deferred.iter().filter_map(|(file, _, at)| file.items.items[*at as usize].name.map(|name| file.source.text(name).to_owned()));
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

/// The cfg values known before build scripts run: the target, the unit's features and the profile.
#[derive(Debug, Default)]
pub struct Env {
    pub target: Target,
    pub features: BTreeSet<String>,
}

const PROFILE: [&str; 3] = ["debug_assertions", "overflow_checks", "ub_checks"];
const OFF: [&str; 10] = ["test", "doc", "doctest", "miri", "rustfmt", "clippy", "windows", "wasm", "bootstrap", "docsrs"];

impl Env {
    /// Evaluates `cfg`.
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
        if self.target.has_name(name) || PROFILE.contains(&name) {
            Truth::Yes
        } else if OFF.contains(&name) {
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
        } else {
            Truth::Unknown
        }
    }

    /// Whether every `cfg` attribute in `attrs`, including those `cfg_attr` enables, may hold.
    pub fn enabled(&self, attrs: &[Attr]) -> bool {
        attrs.iter().all(|attr| match attr {
            Attr::Cfg(cfg) => self.eval(cfg) != Truth::No,
            Attr::CfgAttr(cfg, inner) => self.eval(cfg) == Truth::No || self.enabled(inner),
            _ => true,
        })
    }

    /// The `#[path]` value in `attrs`, including one that `cfg_attr` may enable.
    pub fn path<'a>(&self, attrs: &'a [Attr]) -> Option<&'a str> {
        attrs.iter().find_map(|attr| match attr {
            Attr::Path(path) => Some(path.as_str()),
            Attr::CfgAttr(cfg, inner) if self.eval(cfg) != Truth::No => self.path(inner),
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

/// Files read and scanned once, shared by every crate that includes them. Worker threads read files ahead of use.
#[derive(Default)]
pub struct Cache {
    slots: Arc<Mutex<HashMap<PathBuf, Slot>>>,
    queue: OnceLock<Sender<PathBuf>>,
}

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

    /// Starts reading `path` on a worker thread if it exists, so a later `file` finds it read.
    pub fn prefetch(&self, path: PathBuf) {
        let queue = self.queue.get_or_init(|| {
            let (sender, receiver) = channel::<PathBuf>();
            let receiver = Arc::new(Mutex::new(receiver));
            let workers = std::thread::available_parallelism().map_or(1, |count| count.get().saturating_sub(1).max(1));
            for _ in 0..workers {
                let (receiver, slots) = (receiver.clone(), self.slots.clone());
                std::thread::spawn(move || {
                    loop {
                        let next = receiver.lock().ok().and_then(|queue| queue.recv().ok());
                        let Some(path) = next else { return };
                        if path.is_file() && let Ok(slot) = slot(&slots, &path) {
                            let _ = fill(&slot, &path);
                        }
                    }
                });
            }
            sender
        });
        let _ = queue.send(path);
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

fn read(path: &Path) -> Result<File> {
    let source = Source::read(path)?;
    let items = scan::scan(&source)?;
    Ok(File { source, items, expanded: false })
}

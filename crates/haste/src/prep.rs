//! Prepped code: standard library and dependency functions that rustc optimizes once into a cached library, which Haste calls through wrappers instead of compiling.

mod cargo;
mod choose;
mod name;
mod select;
mod shim;

use std::collections::BTreeSet;
use std::ffi::{CStr, CString};
use std::hash::BuildHasher as _;
use std::os::unix::ffi::OsStrExt as _;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{Context as _, Result, bail};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::graph::Workspace;
use crate::host::HostSymbol;
use crate::names::Names;
use crate::outline::Outline;
use crate::sema::Context;
use crate::sema::infer::Typed;
use crate::sema::reach::{Instance, Reached};
use crate::sema::ty::Ty;

use cargo::{Dirs, Toolchain};
use name::{Paths, Printer};
use select::{Crates, Purpose};
use shim::{Shim, Wrapper};

const ABI_VERSION: u32 = 1;
const SUPPORTED: bool = cfg!(all(target_arch = "x86_64", target_os = "linux"));
const STORE: &str = "prep";
const TARGET: &str = "target";
const MANIFEST: &str = "manifest";
const HEADER: &str = "haste-prep";
const LIBRARY: &str = "library";
const ENTRY: &str = "entry";
const DROPPED: &str = "dropped";
const FIELD: char = '\t';

/// The prepped library of one build, the wrapper of each function it holds, and what binding a reached function to one needs.
pub struct Manifest {
    stored: Stored,
    paths: Paths,
    crates: Crates,
}

/// What `haste prep` keeps on disk for one key: the library, each wrapper by its function's text, the texts rustc rejected, and the names of the functions it wraps, which a lookup checks before it prints a callee.
struct Stored {
    library: PathBuf,
    entries: HashMap<String, Entry>,
    dropped: BTreeSet<String>,
    names: HashSet<String>,
}

struct Entry {
    symbol: String,
    layout: u64,
}

/// What a build runs: the reached functions bound to the address of a prepped wrapper, those Haste still compiles, and what preparing them did.
pub struct Plan {
    pub manifest: Manifest,
    pub prepped: Vec<(usize, usize)>,
    pub compiled: Vec<usize>,
    pub counts: Counts,
}

/// What `prep` did: the functions it selected, the wrappers it built and those it found stored, and the selected functions rustc rejected.
#[derive(Clone, Copy, Debug, Default)]
pub struct Counts {
    pub selected: usize,
    pub built: usize,
    pub reused: usize,
    pub dropped: usize,
}

static BOUND: OnceLock<Manifest> = OnceLock::new();

impl std::fmt::Display for Counts {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(out, "prep {} selected, {} built, {} dropped, {} reused", self.selected, self.built, self.dropped, self.reused)
    }
}

/// Selects the reached functions whose whole call tree rustc may compile, builds their wrappers unless the store holds them for the same key and layouts, opens the library, and plans which functions Haste still compiles.
pub fn prep(cx: &mut Context<'_>, workspace: &Workspace, reached: &Reached) -> Result<Plan> {
    if !SUPPORTED {
        bail!("prep supports x86-64 Linux only");
    }
    let toolchain = Toolchain::find()?;
    let key = key(cx.outline, workspace, &toolchain)?;
    let (crates, paths) = (Crates::new(cx.outline, workspace), Paths::new(cx.outline, cx.names, workspace));
    let entries = entries(reached);
    let (wrappers, krates) = choose::choose(cx, (&crates, &paths), reached, &entries)?;
    let chosen: Vec<(usize, String, u64)> = wrappers.iter().map(|wrapper| (wrapper.index, wrapper.key.clone(), wrapper.layout)).collect();
    let dir = dir(key);
    let (stored, made) = match Stored::read(&dir)? {
        Some(kept) if kept.covers(&chosen) => (kept, false),
        _ => (make(cx, workspace, (&toolchain, key, &dir), (wrappers, &krates))?, true),
    };
    let counts = stored.counts(&chosen, made);
    let manifest = Manifest { stored, paths, crates };
    manifest.open()?;
    let bound: HashMap<usize, usize> = chosen.iter().filter_map(|(at, key, _)| Some((*at, manifest.symbol(key)?))).collect();
    let (compiled, stopped) = select::visit(reached, &entries, |at| bound.contains_key(&at));
    let prepped = stopped.into_iter().filter_map(|at| Some((at, *bound.get(&at)?))).collect();
    Ok(Plan { manifest, prepped, compiled, counts })
}

/// Opens `manifest` and makes it the one every later binding of this process reads.
pub fn install(manifest: Manifest) -> Result<()> {
    manifest.open()?;
    if BOUND.set(manifest).is_err() {
        bail!("a prepped library is already bound");
    }
    Ok(())
}

/// The address of the prepped wrapper that runs reached function `at`, when a manifest is installed and holds it.
pub fn bind(cx: &mut Context<'_>, reached: &Reached, at: usize) -> Option<usize> {
    let manifest = BOUND.get()?;
    let (instance, Ok(typed)) = &reached.list[at] else { return None };
    manifest.address(cx, *instance, typed)
}

/// The manifest that `prep` stored for the build of `workspace` with the resolved `names`, when one matches its key.
pub fn load(outline: &Outline, names: &Names, workspace: &Workspace) -> Result<Option<Manifest>> {
    if !SUPPORTED {
        return Ok(None);
    }
    let toolchain = Toolchain::find()?;
    let Some(stored) = Stored::read(&dir(key(outline, workspace, &toolchain)?))? else { return Ok(None) };
    Ok(Some(Manifest { stored, paths: Paths::new(outline, names, workspace), crates: Crates::new(outline, workspace) }))
}

/// The reached functions that no reached function calls, where a build's calls start.
fn entries(reached: &Reached) -> Vec<usize> {
    let callers = select::callers(reached);
    (0..reached.list.len()).filter(|&at| callers[at].is_empty()).collect()
}

fn candidate(cx: &mut Context<'_>, (printer, crates): (&mut Printer<'_>, &Crates), reached: &Reached, at: usize) -> Result<Option<Wrapper>> {
    let Ok(typed) = &reached.list[at].1 else { return Ok(None) };
    let (params, ret) = signature(typed);
    let Some(callee) = printer.callee(cx, reached.list[at].0) else { return Ok(None) };
    let Some(layout) = crossing(cx, crates, &params, ret) else { return Ok(None) };
    Wrapper::new(cx, printer, (at, callee, layout), (&params, ret))
}

fn signature(typed: &Typed) -> (Vec<Ty>, Option<Ty>) {
    (typed.params.iter().map(|&local| typed.locals[local as usize].ty).collect(), typed.ret)
}

fn crossing(cx: &mut Context<'_>, crates: &Crates, params: &[Ty], ret: Option<Ty>) -> Option<u64> {
    let mut types = params.to_vec();
    types.extend(ret);
    select::walk(cx, crates, &types, Purpose::Crossing)
}

fn make(cx: &Context<'_>, workspace: &Workspace, (toolchain, key, dir): (&Toolchain, u64, &Path), (wrappers, krates): (Vec<Wrapper>, &BTreeSet<u32>)) -> Result<Stored> {
    let mut shim = Shim::new(cx, workspace, key, wrappers, krates);
    let top = crate::graph::workspace_root(&root(workspace)?)?;
    let target = store().join(TARGET);
    let library = cargo::build(toolchain, &Dirs { shim: dir, target: &target, workspace: &top }, &mut shim)?;
    let stored = Stored::built(library, &shim);
    stored.write(dir)?;
    Ok(stored)
}

/// The key of everything prepped code depends on besides the functions it holds: the wrapper format, the toolchain, the target, the dependency graph and the workspace.
fn key(outline: &Outline, workspace: &Workspace, toolchain: &Toolchain) -> Result<u64> {
    let root = root(workspace)?;
    let triple = crate::graph::cfg::Target::host().triple;
    let format = (ABI_VERSION, shim::PRELUDE, cargo::SETTINGS, crate::profile());
    Ok(rustc_hash::FxBuildHasher.hash_one((format, &toolchain.version, &toolchain.library, triple, &outline.pins, root)))
}

fn root(workspace: &Workspace) -> Result<PathBuf> {
    let member = workspace.graph.members.first().context("the workspace has no member package")?;
    Ok(workspace.packages[*member].manifest.dir.clone())
}

fn store() -> PathBuf {
    crate::graph::store::home().join(STORE)
}

fn dir(key: u64) -> PathBuf {
    store().join(format!("{key:016x}"))
}

impl Stored {
    fn built(library: PathBuf, shim: &Shim) -> Stored {
        let mut stored = Stored { library, entries: HashMap::default(), dropped: BTreeSet::new(), names: HashSet::default() };
        for (at, wrapper) in shim.wrappers.iter().enumerate() {
            if shim.dropped[at] {
                stored.dropped.insert(wrapper.key.clone());
            } else {
                stored.hold(wrapper.key.clone(), Entry { symbol: shim.symbol(at), layout: wrapper.layout });
            }
        }
        stored
    }

    fn hold(&mut self, key: String, entry: Entry) {
        self.names.insert(name::function(&key).to_owned());
        self.entries.insert(key, entry);
    }

    fn counts(&self, chosen: &[(usize, String, u64)], made: bool) -> Counts {
        let dropped = chosen.iter().filter(|(_, key, _)| self.dropped.contains(key)).count();
        let held = chosen.len() - dropped;
        Counts { selected: chosen.len(), built: if made { held } else { 0 }, reused: if made { 0 } else { held }, dropped }
    }

    fn covers(&self, chosen: &[(usize, String, u64)]) -> bool {
        chosen.iter().all(|(_, key, layout)| self.dropped.contains(key) || self.entries.get(key).is_some_and(|entry| entry.layout == *layout))
    }

    fn write(&self, dir: &Path) -> Result<()> {
        let mut text = format!("{HEADER} {ABI_VERSION}\n{LIBRARY} {}\n", self.library.display());
        let mut entries: Vec<(&String, &Entry)> = self.entries.iter().collect();
        entries.sort_by(|left, right| left.0.cmp(right.0));
        for (key, entry) in entries {
            text.push_str(&format!("{ENTRY} {}{FIELD}{:016x}{FIELD}{key}\n", entry.symbol, entry.layout));
        }
        for key in &self.dropped {
            text.push_str(&format!("{DROPPED} {key}\n"));
        }
        crate::graph::store::write_new(&dir.join(MANIFEST), &text).with_context(|| format!("writing the prep manifest in {}", dir.display()))
    }

    fn read(dir: &Path) -> Result<Option<Stored>> {
        let text = match std::fs::read_to_string(dir.join(MANIFEST)) {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error).with_context(|| format!("reading the prep manifest in {}", dir.display())),
        };
        Ok(Stored::parse(&text))
    }

    fn parse(text: &str) -> Option<Stored> {
        let mut lines = text.lines();
        (lines.next()? == format!("{HEADER} {ABI_VERSION}")).then_some(())?;
        let library = PathBuf::from(lines.next()?.strip_prefix(LIBRARY)?.trim_start());
        let mut stored = Stored { library, entries: HashMap::default(), dropped: BTreeSet::new(), names: HashSet::default() };
        for line in lines {
            let (word, rest) = line.split_once(' ')?;
            match word {
                ENTRY => {
                    let mut fields = rest.splitn(3, FIELD);
                    let (symbol, layout, key) = (fields.next()?, fields.next()?, fields.next()?);
                    stored.hold(key.to_owned(), Entry { symbol: symbol.to_owned(), layout: u64::from_str_radix(layout, 16).ok()? });
                }
                DROPPED => _ = stored.dropped.insert(rest.to_owned()),
                _ => return None,
            }
        }
        Some(stored)
    }
}

impl Manifest {
    /// The prepped library this manifest names.
    pub fn library(&self) -> &Path {
        &self.stored.library
    }

    /// Loads the library into this process with its symbols global, so the dynamic linker finds every wrapper.
    pub fn open(&self) -> Result<()> {
        let path = CString::new(self.stored.library.as_os_str().as_bytes())?;
        // SAFETY: path is a NUL-terminated string that lives across the call; the library holds only rustc output whose initializers are the standard library's.
        let handle = unsafe { libc::dlopen(path.as_ptr(), libc::RTLD_NOW | libc::RTLD_GLOBAL) };
        if !handle.is_null() {
            return Ok(());
        }
        // SAFETY: dlerror takes no arguments and returns null or a message that stays valid until the next dynamic linker call on this thread.
        let reason = unsafe { libc::dlerror() };
        // SAFETY: reason is a live NUL-terminated string when it is not null, and it is copied before any other dynamic linker call.
        let reason = if reason.is_null() { String::new() } else { unsafe { CStr::from_ptr(reason) }.to_string_lossy().into_owned() };
        bail!("loading {}: {reason}", self.stored.library.display())
    }

    fn symbol(&self, key: &str) -> Option<usize> {
        HostSymbol::Dynamic(self.stored.entries.get(key)?.symbol.as_str().into()).host_addr()
    }

    /// The address of the wrapper of `instance` with body `typed`, when this manifest holds it at the layouts Haste computes now and the open library exports it.
    pub fn address(&self, cx: &mut Context<'_>, instance: Instance, typed: &Typed) -> Option<usize> {
        if !self.stored.names.contains(cx.name(instance.0)) {
            return None;
        }
        let callee = Printer::new(&self.paths).callee(cx, instance)?;
        let entry = self.stored.entries.get(&callee.key)?;
        let (params, ret) = signature(typed);
        (crossing(cx, &self.crates, &params, ret)? == entry.layout).then_some(())?;
        self.symbol(&callee.key)
    }
}

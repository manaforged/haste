//! Prepped code: rustc optimizes every function instance of a build once into a cached library, and Haste binds each function it reaches to that code when the instance, its convention and its layouts agree.

mod bind;
mod bridge;
mod cargo;
mod conv;
mod driver;
mod edit;
mod ident;
mod indirect;
mod key;
mod record;
mod statics;
#[path = "../../prep-driver/src/form.rs"]
mod form;
#[path = "../../prep-driver/src/print.rs"]
mod print;
#[path = "../../prep-driver/src/source.rs"]
mod source;

use std::ffi::{CStr, CString};
use std::hash::BuildHasher as _;
use std::os::unix::ffi::OsStrExt as _;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{Context as _, Result, bail};
use rustc_hash::FxHashMap as HashMap;

use crate::graph::Workspace;
use crate::host::HostSymbol;
use crate::lower::Slots;
use crate::outline::Outline;
use crate::sema::Context;
use crate::sema::infer::Typed;
use crate::sema::reach::Reached;
use crate::sema::ty::Ty;

use bind::{Bindings, Decision};
use cargo::Toolchain;
use edit::Snapshot;

pub use bind::Tally;
pub use bridge::Carry;
pub use indirect::{Method, Vtable};

/// What a prep made: the records of the library it built.
#[derive(Clone, Copy, Debug)]
pub struct Counts {
    pub records: usize,
}

impl std::fmt::Display for Counts {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(out, "prep rustc {} records", self.records)
    }
}

const ABI_VERSION: u32 = 2;
const SUPPORTED: bool = cfg!(all(target_arch = "x86_64", target_os = "linux"));
const STORE: &str = "prep";
const RUSTC: &str = "rustc";
const LOCK: &str = "rustc.lock";
const PREP_ARGS: [&str; 2] = ["--release", "prep"];

/// The driver's library of one build and what binding a reached function to it needs.
pub struct Manifest {
    rustc: Bindings,
    entries: Mutex<Entries>,
}

/// The vtable entries made so far, by reached function and argument carries, and the code memory they live in, mapped before the program's process forks so that it runs them.
struct Entries {
    memory: haste_codegen::Memory,
    placed: HashMap<(usize, Vec<Carry>), u64>,
}

/// The bytes of code memory that vtable entries may fill.
const ENTRY_BYTES: usize = 1 << 20;

static BOUND: OnceLock<Manifest> = OnceLock::new();

/// Builds the workspace's member package with the prep driver into the store, beside a snapshot of the sources of `outline`, and says what the library holds.
pub fn prep(outline: &Outline, workspace: &Workspace) -> Result<Counts> {
    if !SUPPORTED {
        bail!("prep supports x86-64 Linux only");
    }
    let toolchain = Toolchain::find()?;
    let key = key(outline, workspace, &toolchain)?;
    let rustc = rustc(outline, workspace, (&toolchain, key))?.context("the prep driver's library does not match the sources")?;
    Ok(Counts { records: rustc.len() })
}

/// Opens `manifest`, gives the shared standard library of its library the program arguments `args`, and makes it the one every later binding of this process reads.
pub fn install(mut manifest: Manifest, args: &[String]) -> Result<()> {
    manifest.rustc.open()?;
    manifest.rustc.start(args)?;
    let entries = manifest.entries.get_mut().map_err(|_| anyhow::anyhow!("vtable entries poisoned"))?;
    entries.memory.reserve(ENTRY_BYTES).map_err(|error| anyhow::anyhow!("reserving vtable entry memory: {error:?}"))?;
    entries.memory.split(1, 0);
    if BOUND.set(manifest).is_err() {
        bail!("a prepped library is already bound");
    }
    Ok(())
}

/// The address at which a library loaded into this process exports symbol `name`.
fn exported(name: &str) -> Option<usize> {
    HostSymbol::Dynamic(name.into()).host_addr()
}

/// Flushes what rustc's code of the bound library wrote to standard output, once the program's `main` has returned.
pub fn finish() {
    if let Some(manifest) = BOUND.get() {
        manifest.rustc.finish();
    }
}

/// What binding decided for each function this process compiled or bound, when a manifest is installed.
pub fn decided() -> Option<Tally> {
    BOUND.get().map(|manifest| manifest.rustc.decided())
}

/// What binding decided for reached function `at`, when a manifest is installed and binding asked about it.
pub fn why(reached: &Reached, at: usize) -> Option<String> {
    BOUND.get()?.rustc.decision(reached.list[at].0).map(|decision| format!("{decision:?}"))
}

/// rustc's code for a reached function: its address, or a function in Haste's convention that calls it there.
pub enum Prepped {
    Direct(usize),
    Adapter(haste_codegen::Function),
}

/// The address of rustc's code for reached function `at`, when a manifest is installed and may bind it.
pub fn bind(cx: &mut Context<'_>, reached: &Reached, at: usize) -> Option<usize> {
    bound(cx, reached, at).map(|(address, _)| address)
}

/// The code that runs reached function `at` in rustc's code, when a manifest is installed and may bind it: rustc's own when its convention is Haste's, else an adapter to it.
pub fn prepped(cx: &mut Context<'_>, reached: &Reached, at: usize) -> Option<Prepped> {
    let (address, carries) = bound(cx, reached, at)?;
    let Some(carries) = carries else { return Some(Prepped::Direct(address)) };
    let Ok(typed) = &reached.list[at].1 else { return None };
    let (params, ret) = signature(typed);
    bridge::adapter(cx, (&params, ret), &carries, address).ok().map(Prepped::Adapter)
}

/// The address of rustc's code for reached function `at` and how its arguments reach it when any does not travel as Haste passes it, when a manifest is installed and may bind it.
fn bound(cx: &mut Context<'_>, reached: &Reached, at: usize) -> Option<(usize, Option<Vec<bridge::Carry>>)> {
    let manifest = BOUND.get()?;
    let (instance, Ok(typed)) = &reached.list[at] else { return None };
    match manifest.rustc.decide(cx, *instance, typed) {
        (Decision::Bound(address), carries) => Some((address, carries)),
        _ => None,
    }
}

/// The address of the prepped library's copy of static `item`, which Haste's code then reads and writes in place of its own, when a manifest is installed and Haste shares that static.
pub fn static_address(cx: &mut Context<'_>, item: crate::names::ItemRef) -> Option<usize> {
    BOUND.get()?.rustc.static_address(cx, item)
}

/// rustc's layout of every vtable of trait object type `ty`, which Haste's code then builds and calls in rustc's layout and convention, when a bound library touches the type and Haste can follow it.
pub fn vtable(cx: &mut Context<'_>, ty: Ty) -> Option<Arc<Vtable>> {
    match BOUND.get()?.rustc.mode(cx, ty) {
        indirect::Mode::Rustc(vtable) => Some(vtable),
        indirect::Mode::Free | indirect::Mode::Opaque => None,
    }
}

/// The address of a function in rustc's convention for vtable slot `method` that runs reached function `index` in Haste's convention through its call table entry, which `slots` places; made once per function and slot shape.
pub fn entry(cx: &mut Context<'_>, slots: &dyn Slots, index: usize, method: &Method) -> Result<u64> {
    let Some(manifest) = BOUND.get() else { bail!("a vtable entry with no bound library") };
    let key = (index, method.carries.clone());
    if let Some(address) = manifest.entries.lock().ok().and_then(|entries| entries.placed.get(&key).copied()) {
        return Ok(address);
    }
    let func = bridge::entry(cx, (&method.params, method.ret), &method.carries, slots.entry(index))?;
    let compiled = haste_codegen::compile(func, haste_codegen::Target::host()).map_err(|error| anyhow::anyhow!("compiling a vtable entry: {error:?}"))?;
    if !compiled.relocs().is_empty() || !compiled.sites().is_empty() {
        bail!("a vtable entry whose code names addresses it is placed at");
    }
    let mut entries = manifest.entries.lock().map_err(|_| anyhow::anyhow!("vtable entries poisoned"))?;
    let address = entries.memory.write(compiled.bytes()).map_err(|error| anyhow::anyhow!("placing a vtable entry: {error:?}"))? as u64;
    entries.placed.insert(key, address);
    Ok(address)
}

/// Calls rustc's code at `code` through a vtable slot or function pointer from Haste's code, with the Haste-convention argument values `values` of a call taking `params` and returning `ret`, each reaching rustc's convention as `carries` says, and returns the call's results.
pub fn call_native(cx: &mut Context<'_>, b: &mut haste_codegen::FunctionBuilder<'_>, ((params, ret), carries): ((&[Ty], Option<Ty>), &[Carry]), (code, values): (haste_codegen::Value, &[haste_codegen::Value])) -> Result<Vec<haste_codegen::Value>> {
    let shapes = bridge::shapes(cx, (params, ret), carries)?;
    Ok(bridge::call_native(b, &shapes, code, values))
}

/// The manifest that `prep` stored for the build of `workspace` when one matches its key and the sources outside functions, or why it cannot be read, and whether it was built from the sources as they are now.
pub fn load(outline: &Outline, workspace: &Workspace) -> Result<(Result<Option<Manifest>>, bool)> {
    if !SUPPORTED {
        return Ok((Ok(None), true));
    }
    let toolchain = Toolchain::find()?;
    let key = key(outline, workspace, &toolchain)?;
    let rustc = Bindings::read(&dir(key).join(RUSTC), &Snapshot::take(outline, workspace), (outline, workspace));
    let current = rustc.as_ref().is_ok_and(|rustc| rustc.as_ref().is_some_and(Bindings::current));
    Ok((rustc.map(|rustc| rustc.map(|rustc| Manifest { rustc, entries: Mutex::new(Entries { memory: haste_codegen::Memory::default(), placed: HashMap::default() }) })), current))
}

/// Builds the workspace's member package with the prep driver into the store under `key`, beside a snapshot of its sources, while no other prep of the key runs, and reads what binding to it allows now; a library built while a local source changed is dropped.
fn rustc(outline: &Outline, workspace: &Workspace, (toolchain, key): (&Toolchain, u64)) -> Result<Option<Bindings>> {
    std::fs::create_dir_all(dir(key))?;
    let _lock = lock(&dir(key), true)?;
    let out = dir(key).join(RUSTC);
    let before = Snapshot::take(outline, workspace);
    let fresh = driver::run(toolchain, &store(), &root(workspace)?, &out)?;
    if !edit::settled(outline, workspace)? {
        std::fs::remove_dir_all(&fresh)?;
        bail!("a local source changed while the prep driver ran");
    }
    before.write(&fresh.join(bind::SNAPSHOT))?;
    driver::publish(&fresh, &out)?;
    Bindings::read(&out, &before, (outline, workspace))
}

/// Starts `haste --release prep` on the build's member package in the background, unless a prep of the build runs now; it never waits for it.
pub fn refresh(outline: &Outline, workspace: &Workspace) -> Result<()> {
    let dir = dir(key(outline, workspace, &Toolchain::find()?)?);
    std::fs::create_dir_all(&dir)?;
    if lock(&dir, false)?.is_none() {
        return Ok(());
    }
    let mut command = std::process::Command::new(std::env::current_exe()?);
    command.args(PREP_ARGS).arg(root(workspace)?).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());
    std::os::unix::process::CommandExt::process_group(&mut command, 0);
    command.spawn().context("starting a background prep")?;
    Ok(())
}

/// The lock that one prep of the build in `dir` holds while it runs, waiting for it when `wait` is set, or none when another process holds it and `wait` is not set.
fn lock(dir: &Path, wait: bool) -> Result<Option<std::fs::File>> {
    let file = std::fs::OpenOptions::new().create(true).truncate(false).write(true).open(dir.join(LOCK))?;
    let mode = if wait { libc::LOCK_EX } else { libc::LOCK_EX | libc::LOCK_NB };
    // SAFETY: the descriptor belongs to `file`, which lives across the call; flock only changes the lock the open file holds.
    let held = unsafe { libc::flock(std::os::fd::AsRawFd::as_raw_fd(&file), mode) } == 0;
    Ok(held.then_some(file))
}

/// Loads the library at `path` into this process with its symbols global, so the dynamic linker finds every symbol it exports.
fn open(path: &Path) -> Result<()> {
    let text = CString::new(path.as_os_str().as_bytes())?;
    // SAFETY: text is a NUL-terminated string that lives across the call; the library holds only rustc output whose initializers are the standard library's.
    let handle = unsafe { libc::dlopen(text.as_ptr(), libc::RTLD_NOW | libc::RTLD_GLOBAL) };
    if !handle.is_null() {
        return Ok(());
    }
    // SAFETY: dlerror takes no arguments and returns null or a message that stays valid until the next dynamic linker call on this thread.
    let reason = unsafe { libc::dlerror() };
    // SAFETY: reason is a live NUL-terminated string when it is not null, and it is copied before any other dynamic linker call.
    let reason = if reason.is_null() { String::new() } else { unsafe { CStr::from_ptr(reason) }.to_string_lossy().into_owned() };
    bail!("loading {}: {reason}", path.display())
}

fn signature(typed: &Typed) -> (Vec<Ty>, Option<Ty>) {
    (typed.params.iter().map(|&local| typed.locals[local as usize].ty).collect(), typed.ret)
}

/// The key of everything prepped code depends on besides the local sources: the manifest format, the driver, the toolchain, the target, the dependency graph and the workspace.
fn key(outline: &Outline, workspace: &Workspace, toolchain: &Toolchain) -> Result<u64> {
    let root = root(workspace)?;
    let triple = crate::graph::cfg::Target::host().triple;
    let format = (ABI_VERSION, driver::sources(), cargo::SETTINGS, crate::profile());
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

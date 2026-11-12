use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Result, bail};
use rustc_hash::FxHashMap as HashMap;

use crate::sema::Context;
use crate::sema::infer::Typed;
use crate::sema::reach::{Instance, Reached};

use super::edit::Snapshot;
use super::ident::{self, Ident};
use super::record::{Facts, InstanceKind, Library, Record};
use super::select::Crates;
use super::{conv, driver};

const MAIN: &str = "main";
const START: &str = "std::rt::lang_start";

/// The file beside a driver library that holds the snapshot of the sources it was built from.
pub(super) const SNAPSHOT: &str = "snapshot";

/// What binding one reached function to rustc's code decided.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Decision {
    /// Bound to the code at this address.
    Bound(usize),
    /// No record names the function.
    Unmatched,
    /// An edit since prep changed the function, code rustc inlined into it, or code it calls.
    Stale,
    /// It or code it calls touches a mutable static outside the standard crates, which Haste's code holds apart.
    State,
    /// rustc passes an argument or the return otherwise than Haste does.
    Convention,
    /// rustc lays out a type it crosses with otherwise than Haste does, or the type cannot cross.
    Layout,
    /// The open library does not export the record's symbol.
    Missing,
}

/// The driver's library of one build and what binding to it needs: its records by the path that names them, which records an edit or shared state rules out, and the facts of the build's crates.
pub(super) struct Bindings {
    library: PathBuf,
    facts: Facts,
    records: Vec<Record>,
    index: HashMap<String, Vec<usize>>,
    stale: Vec<bool>,
    state: Vec<bool>,
    sysroot: Vec<bool>,
    crates: Crates,
    decided: Mutex<HashMap<Instance, Decision>>,
}

impl Bindings {
    /// The driver library in `dir` with its records ruled out by the edits from its snapshot to `now`; none when `dir` holds no library or an edit outside functions makes every record stale.
    pub(super) fn read(dir: &Path, now: &Snapshot, (sysroot, crates): (Vec<bool>, Crates)) -> Result<Option<Bindings>> {
        let (manifest, snapshot) = (dir.join(driver::MANIFEST), dir.join(SNAPSHOT));
        if !manifest.is_file() || !snapshot.is_file() {
            return Ok(None);
        }
        let Some(changed) = now.changed(&Snapshot::read(&snapshot)?) else { return Ok(None) };
        let Library { library, record: records } = Library::read(&manifest)?;
        let mut index: HashMap<String, Vec<usize>> = HashMap::default();
        for (at, record) in records.iter().enumerate() {
            let key = record.closure.as_ref().map_or(&record.path, |closure| &closure.root);
            index.entry(key.clone()).or_default().push(at);
        }
        let callers = callers(&records);
        let stale = spread(&callers, records.iter().map(|record| touched(record, &changed)).collect());
        let mut state = spread(&callers, records.iter().map(|record| record.state).collect());
        for at in (0..records.len()).filter(|&at| entry(&records, &callers, at)) {
            state[at] = false;
        }
        Ok(Some(Bindings { library: dir.join(driver::LIBRARY), facts: library, records, index, stale, state, sysroot, crates, decided: Mutex::default() }))
    }

    /// Loads the library into this process.
    pub(super) fn open(&self) -> Result<()> {
        super::open(&self.library)
    }

    /// Gives the shared standard library that the open library links the arguments `args` that the program's own code receives.
    pub(super) fn start(&self, args: &[String]) -> Result<()> {
        let Some(name) = &self.facts.args_init else { bail!("{} names no argument initializer", self.library.display()) };
        let Some(address) = super::exported(name) else { bail!("the shared standard library exports no {name}") };
        let owned = args.iter().map(|arg| CString::new(arg.as_str())).collect::<Result<Vec<_>, _>>()?;
        let mut argv: Vec<*const libc::c_char> = owned.iter().map(|arg| arg.as_ptr()).collect();
        argv.push(std::ptr::null());
        // SAFETY: address is the shared standard library's `init_wrapper(argc, argv, envp)`, which stores argv for `std::env::args`; argv and its strings are leaked so they outlive every reader, and the function reads no environment pointer.
        let init: extern "C" fn(libc::c_int, *const *const libc::c_char, *const *const libc::c_char) = unsafe { std::mem::transmute::<usize, _>(address) };
        init(owned.len() as libc::c_int, argv.as_ptr(), std::ptr::null());
        std::mem::forget(owned);
        std::mem::forget(argv);
        Ok(())
    }

    /// Flushes the standard output of the shared standard library, which its own runtime would flush when its `main` returned.
    pub(super) fn finish(&self) {
        let symbol = |name: &Option<String>| name.as_deref().and_then(super::exported);
        let (Some(stdout), Some(flush)) = (symbol(&self.facts.stdout), symbol(&self.facts.stdout_flush)) else { return };
        // SAFETY: stdout is the shared standard library's `std::io::stdout() -> Stdout`, and flush its `<Stdout as Write>::flush(&mut Stdout) -> io::Result<()>`; `Stdout` is one pointer and the result one word, which rustc's convention passes in the registers the C convention uses on this target.
        let (stdout, flush): (extern "C" fn() -> usize, extern "C" fn(*mut usize) -> usize) = unsafe { (std::mem::transmute::<usize, _>(stdout), std::mem::transmute::<usize, _>(flush)) };
        let mut handle = stdout();
        flush(&mut handle);
    }

    /// The record that names `instance` with body `typed`, when one does.
    pub(super) fn record(&self, cx: &mut Context<'_>, instance: Instance, typed: &Typed) -> Option<usize> {
        let ident = ident::ident(cx, instance, typed, &self.sysroot)?;
        self.index.get(ident.key())?.iter().copied().find(|&at| Ident::names(&ident, &self.records[at]))
    }

    /// What binding `instance` with body `typed` to rustc's code decides, worked out once per instance.
    pub(super) fn decide(&self, cx: &mut Context<'_>, instance: Instance, typed: &Typed) -> Decision {
        if let Some(found) = self.decided.lock().ok().and_then(|decided| decided.get(&instance).copied()) {
            return found;
        }
        let found = self.work_out(cx, instance, typed);
        if let Ok(mut decided) = self.decided.lock() {
            decided.insert(instance, found);
        }
        found
    }

    fn work_out(&self, cx: &mut Context<'_>, instance: Instance, typed: &Typed) -> Decision {
        let Some(at) = self.record(cx, instance, typed) else { return Decision::Unmatched };
        let record = &self.records[at];
        if self.stale[at] {
            return Decision::Stale;
        }
        if self.state[at] {
            return Decision::State;
        }
        let (params, ret) = super::signature(typed);
        if !conv::agrees(cx, (&params, ret), &record.abi) {
            return Decision::Convention;
        }
        let mut types = params;
        types.push(ret.unwrap_or_else(|| cx.types.tuple(&[])));
        if conv::layouts(cx, &self.crates, &types).is_none_or(|found| record.layouts.as_ref() != Some(&found)) {
            return Decision::Layout;
        }
        super::exported(&record.symbol).map_or(Decision::Missing, Decision::Bound)
    }

    /// What binding each function of `reached` decides, and which records no reached function names.
    pub(super) fn tally(&self, cx: &mut Context<'_>, reached: &Reached) -> Tally {
        let mut tally = Tally { records: self.records.len(), ..Tally::default() };
        let mut matched = vec![false; self.records.len()];
        for (instance, typed) in &reached.list {
            let Ok(typed) = typed else { continue };
            let found = self.record(cx, *instance, typed);
            let decision = self.decide(cx, *instance, typed);
            if let Some(at) = found {
                matched[at] = true;
                tally.release += usize::from(matches!(decision, Decision::Bound(_)) && self.records[at].release_copy);
            }
            tally.count(decision);
        }
        tally.matched = matched.iter().filter(|&&found| found).count();
        for (record, _) in self.records.iter().zip(&matched).filter(|(_, found)| !**found) {
            tally.unmatched(record);
        }
        tally
    }
}

/// What binding the reached functions of one build to the driver's library decided, and the kinds of the records no reached function names.
#[derive(Clone, Copy, Debug, Default)]
pub struct Tally {
    pub records: usize,
    pub matched: usize,
    pub bound: usize,
    pub release: usize,
    pub stale: usize,
    pub state: usize,
    pub convention: usize,
    pub layout: usize,
    pub missing: usize,
    pub generic: usize,
    pub shims: usize,
    pub closures: usize,
    pub items: usize,
}

impl Tally {
    fn count(&mut self, decision: Decision) {
        match decision {
            Decision::Bound(_) => self.bound += 1,
            Decision::Stale => self.stale += 1,
            Decision::State => self.state += 1,
            Decision::Convention => self.convention += 1,
            Decision::Layout => self.layout += 1,
            Decision::Missing => self.missing += 1,
            Decision::Unmatched => {}
        }
    }

    fn unmatched(&mut self, record: &Record) {
        match (record.kind, &record.closure) {
            (InstanceKind::Other, _) => self.shims += 1,
            (_, Some(_)) => self.closures += 1,
            _ if !record.args.is_empty() => self.generic += 1,
            _ => self.items += 1,
        }
    }
}

impl std::fmt::Display for Tally {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(out, "rustc {} records, {} matched, {} bound ({} release copies), refused {} stale {} state {} convention {} layout {} missing; unmatched {} generic {} shims {} closures {} items", self.records, self.matched, self.bound, self.release, self.stale, self.state, self.convention, self.layout, self.missing, self.generic, self.shims, self.closures, self.items)
    }
}

/// Whether an edit to the functions at `changed` makes `record` stale: it is one of them or inside one, or rustc inlined one into it.
fn touched(record: &Record, changed: &[String]) -> bool {
    let root = record.closure.as_ref().map(|closure| closure.root.as_str());
    let mut paths = std::iter::once(record.path.as_str()).chain(root).chain(record.inlined.iter().map(String::as_str));
    paths.any(|path| changed.iter().any(|changed| ident::within(path, changed)))
}

/// Whether record `at` is a program's `main`, which only the runtime's start calls: when it runs rustc's code, every function the program runs does, so state that each std keeps apart never crosses.
fn entry(records: &[Record], callers: &[Vec<usize>], at: usize) -> bool {
    let record = &records[at];
    let main = record.closure.is_none() && record.args.is_empty() && record.path.rsplit(ident::SEPARATOR).next() == Some(MAIN);
    main && callers[at].iter().all(|&caller| records[caller].path.starts_with(START))
}

/// For each record, the records that call it by symbol.
fn callers(records: &[Record]) -> Vec<Vec<usize>> {
    let by_symbol: HashMap<&str, usize> = records.iter().enumerate().map(|(at, record)| (record.symbol.as_str(), at)).collect();
    let mut callers = vec![Vec::new(); records.len()];
    for (at, record) in records.iter().enumerate() {
        for callee in record.uses.iter().filter_map(|symbol| by_symbol.get(symbol.as_str())) {
            callers[*callee].push(at);
        }
    }
    callers
}

/// `marked` with every record that reaches a marked one through calls marked too.
fn spread(callers: &[Vec<usize>], mut marked: Vec<bool>) -> Vec<bool> {
    let mut stack: Vec<usize> = (0..marked.len()).filter(|&at| marked[at]).collect();
    while let Some(at) = stack.pop() {
        for &caller in &callers[at] {
            if !std::mem::replace(&mut marked[caller], true) {
                stack.push(caller);
            }
        }
    }
    marked
}

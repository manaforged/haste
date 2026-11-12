use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use anyhow::{Result, bail};
use rustc_hash::FxHashMap as HashMap;

use crate::graph::Workspace;
use crate::names::ItemRef;
use crate::outline::Outline;
use crate::sema::Context;
use crate::sema::infer::Typed;
use crate::sema::reach::Instance;
use crate::sema::ty::Ty;

use super::bridge::Carry;
use super::edit::{Changed, Snapshot};
use super::ident;
use super::indirect::{Indirect, Mode};
use super::key::{Keys, Miss};
use super::record::{At, Facts, InstanceKind, Library, Parsed, Record};
use super::statics::{Shared, Statics};
use super::{conv, files};

const MAIN: &str = "main";
const START: &str = "std::rt::lang_start";
const PATH_SEPARATOR: &str = "::";

/// The file beside a driver library that holds the snapshot of the sources it was built from.
pub(super) const SNAPSHOT: &str = "snapshot";

/// What binding one reached function to rustc's code decided.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Decision {
    /// Bound to the code at this address.
    Bound(usize),
    /// No record names the function.
    Unmatched,
    /// More than one record names the function.
    Ambiguous,
    /// An edit since prep changed the function, code rustc inlined into it, or code it calls.
    Stale,
    /// It or code it calls touches state that Haste's code holds apart.
    State,
    /// rustc passes an argument or the return otherwise than Haste does, or its signature names other types.
    Convention,
    /// rustc lays out a type it crosses with otherwise than Haste does, or the type cannot cross.
    Layout,
    /// It names or crosses with a function pointer or trait object.
    Indirect,
    /// The open library does not export the record's symbol.
    Missing,
    /// rustc's code of it is a small leaf that rustc copies into its callers, so Haste's code copies its own.
    Leaf,
}

/// The driver's library of one build and what binding to it needs: its records by the key that names them, which records an edit or shared state rules out, and how keys name the build's definitions.
pub(super) struct Bindings {
    library: PathBuf,
    facts: Facts,
    records: Vec<Record>,
    index: HashMap<String, Vec<usize>>,
    callers: Vec<Vec<usize>>,
    stale: Vec<bool>,
    statics: Statics,
    settled: OnceLock<Settled>,
    keys: Keys,
    indirect: Indirect,
    current: bool,
    decided: Mutex<HashMap<Instance, Decided>>,
}

/// What binding needs Haste's sema to work out once: the statics Haste's code shares with the library, and the records that touch state Haste's code holds apart.
struct Settled {
    shared: Shared,
    state: Vec<bool>,
}

/// What binding one instance decided, the record that names it, and how its arguments reach rustc's code when any does not travel as Haste passes it.
#[derive(Clone)]
struct Decided {
    decision: Decision,
    record: Option<usize>,
    carries: Option<Vec<Carry>>,
}

impl Bindings {
    /// The driver library in `dir` with its records ruled out by the edits from its snapshot to `now`, its manifest taken from `ahead` when that read the same file; none when `dir` holds no library or an edit outside functions makes every record stale.
    pub(super) fn read(dir: &Path, now: &Snapshot, (outline, workspace): (&Outline, &Workspace), ahead: Option<Parsed>) -> Result<Option<Bindings>> {
        let Ok(dir) = dir.canonicalize() else { return Ok(None) };
        let (manifest, snapshot) = (dir.join(files::manifest(files::LIBRARY)), dir.join(SNAPSHOT));
        if !manifest.is_file() || !snapshot.is_file() {
            return Ok(None);
        }
        let then = Snapshot::read(&snapshot)?;
        let Some(changed) = now.changed(&then) else { return Ok(None) };
        let Library { library, record: records, object, pointer, statics, .. } = Library::load(&dir, ahead)?;
        let indirect = Indirect::new(object, pointer, library.unkeyed);
        let mut index: HashMap<String, Vec<usize>> = HashMap::default();
        for (at, record) in records.iter().enumerate() {
            if let Some(key) = &record.key {
                index.entry(key.clone()).or_default().push(at);
            }
        }
        let callers = callers(&records);
        let stale = spread(&callers, records.iter().map(|record| touched(record, &changed)).collect());
        let keys = Keys::new(outline, workspace, then);
        let current = changed.ranges.is_empty() && changed.crates.is_empty();
        let statics = Statics::new(statics);
        let opened = [files::DELTA, files::LIBRARY].into_iter().map(|name| dir.join(name)).find(|path| path.is_file()).unwrap_or_else(|| dir.join(files::LIBRARY));
        Ok(Some(Bindings { library: opened, facts: library, records, index, callers, stale, statics, settled: OnceLock::new(), keys, indirect, current, decided: Mutex::default() }))
    }

    /// The statics Haste's code shares and the records touching state it holds apart, worked out on first use: a record that reaches std state each std keeps apart or a static Haste does not share, or calls one that does; none when it is the program's `main`, whose code then runs every function; and the runtime's start always.
    fn settled(&self, cx: &mut Context<'_>) -> &Settled {
        if let Some(settled) = self.settled.get() {
            return settled;
        }
        let shared = self.statics.share(cx, (&self.keys, &self.indirect), &self.records);
        let marked = self.records.iter().map(|record| record.held || record.statics.iter().any(|symbol| !shared.holds(&self.statics, symbol))).collect();
        let mut state = spread(&self.callers, marked);
        for at in (0..self.records.len()).filter(|&at| entry(&self.records, &self.callers, at)) {
            state[at] = false;
        }
        for (at, _) in self.records.iter().enumerate().filter(|(_, record)| record.path.starts_with(START)) {
            state[at] = true;
        }
        self.settled.get_or_init(|| Settled { shared, state })
    }

    /// The address of the library's copy of static `item`, when Haste's code shares it.
    pub(super) fn static_address(&self, cx: &mut Context<'_>, item: ItemRef) -> Option<usize> {
        let settled = self.settled(cx);
        self.statics.address(cx, &self.keys, &settled.shared, item)
    }

    /// Whether the library was built from the local sources as they are now.
    pub(super) fn current(&self) -> bool {
        self.current
    }

    /// Loads the library into this process: the second library when there is one, which loads the first as it needs it and comes before it in every lookup.
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

    /// The records that name `instance` with body `typed`, which are copies of one instance that crates instantiated apart when there are several: they name one definition of one crate and cross the same layouts with one convention; or why no record binds.
    fn records(&self, cx: &mut Context<'_>, instance: Instance, typed: &Typed) -> Result<Vec<usize>, Decision> {
        let (key, once) = self.keys.instance(cx, instance, typed).map_err(|Miss::Unnamed| Decision::Unmatched)?;
        let fits = |at: &usize| {
            let record = &self.records[*at];
            match (once, &record.closure) {
                (Some(once), Some(closure)) => ident::called(once, record.kind, closure.kind),
                (None, None) => record.kind == InstanceKind::Item,
                _ => false,
            }
        };
        let found: Vec<usize> = self.index.get(&key).into_iter().flatten().copied().filter(fits).collect();
        let first = &self.records[*found.first().ok_or(Decision::Unmatched)?];
        let copy = |at: &usize| {
            let record = &self.records[*at];
            record.crate_id == first.crate_id && record.layouts == first.layouts && record.abi == first.abi
        };
        if found.iter().all(copy) { Ok(found) } else { Err(Decision::Ambiguous) }
    }

    /// What binding `instance` with body `typed` to rustc's code decides, worked out once per instance, and how its arguments reach that code when any does not travel as Haste passes it.
    pub(super) fn decide(&self, cx: &mut Context<'_>, instance: Instance, typed: &Typed) -> (Decision, Option<Vec<Carry>>) {
        if let Some(found) = self.decided.lock().ok().and_then(|decided| decided.get(&instance).cloned()) {
            return (found.decision, found.carries);
        }
        let found = self.work_out(cx, instance, typed);
        if let Ok(mut decided) = self.decided.lock() {
            decided.insert(instance, found.clone());
        }
        (found.decision, found.carries)
    }

    /// What binding decided for `instance`, when it was asked.
    pub(super) fn decision(&self, instance: Instance) -> Option<Decision> {
        self.decided.lock().ok()?.get(&instance).map(|found| found.decision)
    }

    /// What binding decides for `instance`: the first copy that binds, else why the first copy does not.
    fn work_out(&self, cx: &mut Context<'_>, instance: Instance, typed: &Typed) -> Decided {
        let copies = match self.records(cx, instance, typed) {
            Ok(copies) => copies,
            Err(decision) => return Decided { decision, record: None, carries: None },
        };
        let mut refused = None;
        for at in copies {
            match self.check(cx, typed, at) {
                Ok((decision, carries)) => return Decided { decision, record: Some(at), carries: carries.iter().any(|&carry| carry != Carry::Same).then_some(carries) },
                Err(decision) if refused.is_none() => refused = Some(Decided { decision, record: Some(at), carries: None }),
                Err(_) => {}
            }
        }
        refused.unwrap_or(Decided { decision: Decision::Unmatched, record: None, carries: None })
    }

    /// What binding a function with body `typed` to record `at`, which names it, decides, with how each argument reaches rustc's code.
    fn check(&self, cx: &mut Context<'_>, typed: &Typed, at: usize) -> Result<(Decision, Vec<Carry>), Decision> {
        let record = &self.records[at];
        if self.stale[at] {
            return Err(Decision::Stale);
        }
        if self.settled(cx).state[at] {
            return Err(Decision::State);
        }
        if record.leaf {
            return Err(Decision::Leaf);
        }
        let (params, ret) = super::signature(typed);
        let params = conv::counted(cx, &self.keys, (&params, ret), &record.abi, 0).ok_or(Decision::Convention)?;
        let types: Vec<Ty> = params.iter().copied().chain([ret.unwrap_or_else(|| cx.types.tuple(&[]))]).collect();
        let carries = conv::carries(cx, (&params, ret), &record.abi).ok_or(Decision::Convention)?;
        match conv::layouts(cx, &types) {
            Ok((found, _)) if record.layouts.as_ref() != Some(&found) => Err(Decision::Layout),
            Ok((_, indirect)) if !self.indirect.cross(cx, &self.keys, &indirect) => Err(Decision::Indirect),
            Ok(_) => super::exported(&record.symbol).map(|address| (Decision::Bound(address), carries)).ok_or(Decision::Missing),
            Err(_) => Err(Decision::Layout),
        }
    }

    /// What binding decided for each function this process asked about, and how many records of each kind those functions name.
    pub(super) fn decided(&self) -> Tally {
        let mut tally = Tally { records: self.records.len(), ..Tally::default() };
        let mut matched = vec![false; self.records.len()];
        for Decided { decision, record, carries } in self.decided.lock().map(|decided| decided.values().cloned().collect::<Vec<_>>()).unwrap_or_default() {
            tally.count(decision);
            let bound = matches!(decision, Decision::Bound(_));
            tally.adapted += usize::from(bound && carries.is_some());
            if let Some(at) = record {
                matched[at] = true;
                tally.release += usize::from(bound && self.records[at].release_copy);
            }
        }
        for (record, found) in self.records.iter().zip(&matched) {
            let kind = &mut tally.kinds[Category::of(record) as usize];
            kind.1 += 1;
            kind.0 += usize::from(*found);
        }
        tally.matched = matched.iter().filter(|&&found| found).count();
        tally
    }

    /// How code pointers of trait object or function pointer type `ty` travel between Haste's code and the library's.
    pub(super) fn mode(&self, cx: &mut Context<'_>, ty: Ty) -> Mode {
        self.indirect.mode(cx, &self.keys, ty)
    }

    /// How many records the library holds.
    pub(super) fn len(&self) -> usize {
        self.records.len()
    }
}

/// The kinds of records a tally counts apart: generic instances, closures, other functions, and shims.
#[derive(Clone, Copy)]
enum Category {
    Generic,
    Closure,
    Item,
    Shim,
}

/// The names a tally prints its record kinds by, in `Category` order.
const CATEGORIES: [&str; 4] = ["generic", "closures", "items", "shims"];

impl Category {
    fn of(record: &Record) -> Category {
        match (record.kind, &record.closure) {
            (InstanceKind::Item | InstanceKind::ClosureOnce, Some(_)) => Category::Closure,
            (InstanceKind::Item, None) if !record.args.is_empty() => Category::Generic,
            (InstanceKind::Item, None) => Category::Item,
            _ => Category::Shim,
        }
    }
}

/// What binding the reached functions of one build to the driver's library decided: per record kind how many records a reached function names of how many, and per decision how many functions it took.
#[derive(Clone, Copy, Debug, Default)]
pub struct Tally {
    pub records: usize,
    pub matched: usize,
    pub kinds: [(usize, usize); 4],
    pub bound: usize,
    pub adapted: usize,
    pub release: usize,
    pub stale: usize,
    pub state: usize,
    pub convention: usize,
    pub layout: usize,
    pub indirect: usize,
    pub ambiguous: usize,
    pub missing: usize,
    pub leaf: usize,
    pub unmatched: usize,
}

impl Tally {
    fn count(&mut self, decision: Decision) {
        let counter = match decision {
            Decision::Bound(_) => &mut self.bound,
            Decision::Stale => &mut self.stale,
            Decision::State => &mut self.state,
            Decision::Convention => &mut self.convention,
            Decision::Layout => &mut self.layout,
            Decision::Indirect => &mut self.indirect,
            Decision::Ambiguous => &mut self.ambiguous,
            Decision::Missing => &mut self.missing,
            Decision::Leaf => &mut self.leaf,
            Decision::Unmatched => &mut self.unmatched,
        };
        *counter += 1;
    }
}

impl std::fmt::Display for Tally {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kinds: Vec<String> = CATEGORIES.iter().zip(self.kinds).map(|(name, (found, all))| format!("{name} {found}/{all}")).collect();
        write!(out, "rustc {} records, {} matched ({}), {} bound ({} through adapters, {} release copies), refused {} stale {} state {} convention {} layout {} indirect {} ambiguous {} missing {} leaf, {} unmatched", self.records, self.matched, kinds.join(" "), self.bound, self.adapted, self.release, self.stale, self.state, self.convention, self.layout, self.indirect, self.ambiguous, self.missing, self.leaf, self.unmatched)
    }
}

/// Whether the edits `changed` make `record` stale: it, or code rustc inlined into it, lies in a changed function, or has no place in a crate with one.
fn touched(record: &Record, changed: &Changed) -> bool {
    std::iter::once(&record.at).chain(&record.inlined).any(|at| within(at, changed))
}

fn within(at: &At, changed: &Changed) -> bool {
    match (&at.file, at.lo) {
        (Some(file), Some(lo)) => changed.ranges.iter().any(|(changed, start, end)| changed == file && (*start..*end).contains(&lo)),
        _ => changed.crates.contains(&at.krate),
    }
}

/// Whether record `at` is a program's `main`, which only the runtime's start calls: when it runs rustc's code, every function the program runs does, so state that each std keeps apart never crosses; the start itself always runs Haste's code, which sets up the state of Haste's std.
fn entry(records: &[Record], callers: &[Vec<usize>], at: usize) -> bool {
    let record = &records[at];
    let main = record.closure.is_none() && record.args.is_empty() && record.path.rsplit(PATH_SEPARATOR).next() == Some(MAIN);
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

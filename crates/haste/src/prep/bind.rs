use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Result, bail};
use rustc_hash::FxHashMap as HashMap;

use crate::graph::Workspace;
use crate::outline::Outline;
use crate::sema::Context;
use crate::sema::infer::Typed;
use crate::sema::reach::Instance;

use super::edit::{Changed, Snapshot};
use super::ident;
use super::key::{Keys, Miss};
use super::record::{At, Facts, InstanceKind, Library, Record};
use super::{conv, driver};

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
}

/// The driver's library of one build and what binding to it needs: its records by the key that names them, which records an edit or shared state rules out, and how keys name the build's definitions.
pub(super) struct Bindings {
    library: PathBuf,
    facts: Facts,
    records: Vec<Record>,
    index: HashMap<String, Vec<usize>>,
    stale: Vec<bool>,
    state: Vec<bool>,
    keys: Keys,
    current: bool,
    decided: Mutex<HashMap<Instance, (Decision, Option<usize>)>>,
}

impl Bindings {
    /// The driver library in `dir` with its records ruled out by the edits from its snapshot to `now`; none when `dir` holds no library or an edit outside functions makes every record stale.
    pub(super) fn read(dir: &Path, now: &Snapshot, (outline, workspace): (&Outline, &Workspace)) -> Result<Option<Bindings>> {
        let Ok(dir) = dir.canonicalize() else { return Ok(None) };
        let (manifest, snapshot) = (dir.join(driver::MANIFEST), dir.join(SNAPSHOT));
        if !manifest.is_file() || !snapshot.is_file() {
            return Ok(None);
        }
        let then = Snapshot::read(&snapshot)?;
        let Some(changed) = now.changed(&then) else { return Ok(None) };
        let Library { library, record: records } = Library::read(&manifest)?;
        let mut index: HashMap<String, Vec<usize>> = HashMap::default();
        for (at, record) in records.iter().enumerate() {
            if let Some(key) = &record.key {
                index.entry(key.clone()).or_default().push(at);
            }
        }
        let callers = callers(&records);
        let stale = spread(&callers, records.iter().map(|record| touched(record, &changed)).collect());
        let mut state = spread(&callers, records.iter().map(|record| record.state).collect());
        for at in (0..records.len()).filter(|&at| entry(&records, &callers, at)) {
            state[at] = false;
        }
        let keys = Keys::new(outline, workspace, then);
        let current = changed.ranges.is_empty() && changed.crates.is_empty();
        Ok(Some(Bindings { library: dir.join(driver::LIBRARY), facts: library, records, index, stale, state, keys, current, decided: Mutex::default() }))
    }

    /// Whether the library was built from the local sources as they are now.
    pub(super) fn current(&self) -> bool {
        self.current
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

    /// The record that names `instance` with body `typed`: none when no record does, or the reason no single record does.
    fn record(&self, cx: &mut Context<'_>, instance: Instance, typed: &Typed) -> Result<usize, Decision> {
        let (key, once) = self.keys.instance(cx, instance, typed).map_err(refusal)?;
        let fits = |at: &usize| {
            let record = &self.records[*at];
            match (once, &record.closure) {
                (Some(once), Some(closure)) => ident::called(once, record.kind, closure.kind),
                (None, None) => record.kind == InstanceKind::Item,
                _ => false,
            }
        };
        let found: Vec<usize> = self.index.get(&key).into_iter().flatten().copied().filter(fits).collect();
        match found.as_slice() {
            [] => Err(Decision::Unmatched),
            [one] => Ok(*one),
            _ => Err(Decision::Ambiguous),
        }
    }

    /// What binding `instance` with body `typed` to rustc's code decides, worked out once per instance.
    pub(super) fn decide(&self, cx: &mut Context<'_>, instance: Instance, typed: &Typed) -> Decision {
        if let Some((found, _)) = self.decided.lock().ok().and_then(|decided| decided.get(&instance).copied()) {
            return found;
        }
        let (found, at) = self.work_out(cx, instance, typed);
        if let Ok(mut decided) = self.decided.lock() {
            decided.insert(instance, (found, at));
        }
        found
    }

    fn work_out(&self, cx: &mut Context<'_>, instance: Instance, typed: &Typed) -> (Decision, Option<usize>) {
        match self.record(cx, instance, typed) {
            Ok(at) => (self.check(cx, typed, at), Some(at)),
            Err(decision) => (decision, None),
        }
    }

    /// What binding a function with body `typed` to record `at`, which names it, decides.
    fn check(&self, cx: &mut Context<'_>, typed: &Typed, at: usize) -> Decision {
        let record = &self.records[at];
        if self.stale[at] {
            return Decision::Stale;
        }
        if self.state[at] {
            return Decision::State;
        }
        let (params, ret) = super::signature(typed);
        let mut types = params.clone();
        types.push(ret.unwrap_or_else(|| cx.types.tuple(&[])));
        if !self.same_signature(cx, &types, record) || !conv::agrees(cx, (&params, ret), &record.abi) {
            return Decision::Convention;
        }
        match conv::layouts(cx, &types) {
            Err(Miss::Indirect) => Decision::Indirect,
            Ok(found) if record.layouts.as_ref() == Some(&found) => super::exported(&record.symbol).map_or(Decision::Missing, Decision::Bound),
            _ => Decision::Layout,
        }
    }

    /// Whether the argument and return types `types` have the keys of the types `record` passes.
    fn same_signature(&self, cx: &mut Context<'_>, types: &[crate::sema::ty::Ty], record: &Record) -> bool {
        let theirs: Vec<Option<&str>> = record.abi.args.iter().chain([&record.abi.ret]).map(|arg| arg.key.as_deref()).collect();
        theirs.len() == types.len() && types.iter().zip(theirs).all(|(&ty, theirs)| theirs.is_some_and(|theirs| self.keys.ty(cx, ty).is_ok_and(|ours| ours == theirs)))
    }

    /// What binding decided for each function this process asked about, and how many records of each kind those functions name.
    pub(super) fn decided(&self) -> Tally {
        let mut tally = Tally { records: self.records.len(), ..Tally::default() };
        let mut matched = vec![false; self.records.len()];
        for (decision, at) in self.decided.lock().map(|decided| decided.values().copied().collect::<Vec<_>>()).unwrap_or_default() {
            tally.count(decision);
            if let Some(at) = at {
                matched[at] = true;
                tally.release += usize::from(matches!(decision, Decision::Bound(_)) && self.records[at].release_copy);
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
    pub release: usize,
    pub stale: usize,
    pub state: usize,
    pub convention: usize,
    pub layout: usize,
    pub indirect: usize,
    pub ambiguous: usize,
    pub missing: usize,
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
            Decision::Unmatched => &mut self.unmatched,
        };
        *counter += 1;
    }
}

impl std::fmt::Display for Tally {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kinds: Vec<String> = CATEGORIES.iter().zip(self.kinds).map(|(name, (found, all))| format!("{name} {found}/{all}")).collect();
        write!(out, "rustc {} records, {} matched ({}), {} bound ({} release copies), refused {} stale {} state {} convention {} layout {} indirect {} ambiguous {} missing, {} unmatched", self.records, self.matched, kinds.join(" "), self.bound, self.release, self.stale, self.state, self.convention, self.layout, self.indirect, self.ambiguous, self.missing, self.unmatched)
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

/// Whether record `at` is a program's `main`, which only the runtime's start calls: when it runs rustc's code, every function the program runs does, so state that each std keeps apart never crosses.
fn entry(records: &[Record], callers: &[Vec<usize>], at: usize) -> bool {
    let record = &records[at];
    let main = record.closure.is_none() && record.args.is_empty() && record.path.rsplit(PATH_SEPARATOR).next() == Some(MAIN);
    main && callers[at].iter().all(|&caller| records[caller].path.starts_with(START))
}

/// The reason no record binds when an instance has no key.
fn refusal(miss: Miss) -> Decision {
    match miss {
        Miss::Indirect => Decision::Indirect,
        Miss::Unnamed => Decision::Unmatched,
    }
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

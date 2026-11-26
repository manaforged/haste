//! Programs that compile each function on its first call: each runs in a forked child whose call table starts at stubs, and this process infers, lowers and encodes each function a stub asks for.

mod program;
pub use program::{Program, evaluator, run_on_demand};

#[cfg(unix)]
use std::io::{Read, Write};
#[cfg(unix)]
use std::os::fd::{AsRawFd, FromRawFd};
use std::sync::atomic::Ordering;

use anyhow::{Result, bail};
use rustc_hash::FxHashSet as HashSet;

use crate::lower::Slots;
use crate::macros::Compilation;
use crate::names::ItemRef;
use crate::names::path::Requests;
use crate::outline::Outline;
use crate::sema::reach::{Instance, Reached, calls, infer_declared, infer_quiet, main_roots};
use crate::sema::{Answer, Cache, Context, Edited};

use super::global_asm::Foreign;
#[cfg(unix)]
use super::wait;
use super::{CODE_BYTES, Image, SHARED_BYTES, STATE_BYTES, TABLE_SLOTS, stub};

/// What answers what a pass of inference asked for, and says what it added.
pub type Answerer<'f> = &'f dyn Fn(&mut Compilation, Requests) -> Result<Answer>;

/// What reads the watched files again after they change: the `(crate, file)` slots whose function bodies changed, or why the change needs a restart.
pub type Reread<'r> = &'r mut dyn FnMut(&mut Compilation) -> Result<Vec<(u32, u32)>, String>;

/// Why answering the children paused: a child asked for a function whose answer needs what inference asked for, or a watched file changed.
enum Pause {
    Asked(usize, usize),
    Edited,
}

/// Why a function has no code yet: inference asked for names or macro output not loaded, or it did not compile.
enum Stop {
    Asked,
    Failed(String, ItemRef),
}

/// The functions one answer inferred, taken back when inference asked for more.
type Journal = Vec<usize>;

/// A running child: its pid, the pipe ends that carry its asks and the answers, its output pipe, what it wrote there, and why the function it last asked for has no code.
#[cfg(unix)]
struct Child {
    pid: libc::pid_t,
    asks: Option<std::fs::File>,
    answers: std::fs::File,
    out: Option<std::fs::File>,
    bytes: Vec<u8>,
    failure: Option<(String, ItemRef)>,
}

#[cfg(not(unix))]
struct Child {
    bytes: Vec<u8>,
    failure: Option<(String, ItemRef)>,
}

/// How a child ended: its exit status, what it wrote to its output pipe, and why the function it asked for last has no code, with that function, when that ended it.
pub struct Ended {
    pub status: i32,
    pub bytes: Vec<u8>,
    pub failure: Option<(String, ItemRef)>,
}

/// What forks children on a demand and takes their ends.
pub trait Driver {
    /// Forks the children that can start now.
    fn start(&mut self, demand: &mut Jit) -> Result<()>;
    /// Takes the end of child `id`.
    fn ended(&mut self, id: usize, ended: Ended) -> Result<()>;
    /// Reads the watched files again after a change, and gives the `(crate, file)` slots whose function bodies changed, or why the change needs a restart.
    fn edited(&mut self, _: &mut Compilation) -> Result<Vec<(u32, u32)>, String> {
        Ok(Vec::new())
    }
}

/// The most functions that one ask compiles ahead of their first call.
const AHEAD_FUNCTIONS: usize = 4096;

/// The functions declared so far, the inference store whose types their instances name, the image that holds their stubs and code, the children that run it, whether an ask compiles the functions it reaches ahead, the functions that compiling ahead left to their first call, and the watched source files with whether they changed.
pub struct Jit {
    reached: Reached,
    store: Cache,
    image: Image,
    stubs: Vec<usize>,
    compiled: HashSet<usize>,
    children: Vec<Option<Child>>,
    pending: std::collections::VecDeque<(usize, usize)>,
    strict: bool,
    expanded: HashSet<usize>,
    ahead: bool,
    tried: HashSet<usize>,
    watcher: Option<super::watch::Watcher>,
    edited: bool,
}

/// For each program whose `main` is in `mains`, starting inference from `store`, its two roots, `lang_start` at the return type of `main` and `main`, or why it has none, once inference settles on them; with the store inference kept.
pub fn entries(compilation: &mut Compilation, (store, mains): (Cache, &[Option<ItemRef>])) -> Result<(Vec<Result<[Instance; 2], String>>, Cache)> {
    crate::macros::resolve_sema(compilation, (store, crate::macros::load), |mut cx, last| {
        let roots = mains.iter().map(|main| main.ok_or_else(|| "no `fn main`".to_owned()).and_then(|item| main_roots(&mut cx, item).map_err(|error| format!("{error:#}")))).collect();
        let settled = cx.resolved();
        let released = cx.release();
        Ok(if settled || last { Ok((roots, released.0)) } else { Err(released) })
    })
}

/// Answers the asks of the children `driver` forks on `demand` until none is left, answering what inference asks for with `ask`; `demand` keeps what inference did.
pub fn serve(compilation: &mut Compilation, ask: Answerer<'_>, demand: &mut Jit, driver: &mut dyn Driver) -> Result<()> {
    let mut store = std::mem::take(&mut demand.store);
    store.forget_lookups();
    let mut cx = Context::new(&compilation.outline, &compilation.names, store);
    let outcome = 'run: {
        if let Err(error) = driver.start(demand) {
            break 'run Err(error);
        }
        let mut pending = None;
        loop {
            let asked = match demand.pump(&mut cx, driver, pending.take()) {
                Ok(Some(Pause::Asked(id, index))) => (id, index),
                Ok(Some(Pause::Edited)) => {
                    let (store, _) = cx.release();
                    let store = demand.reload(compilation, driver, store);
                    cx = Context::new(&compilation.outline, &compilation.names, store);
                    continue;
                }
                done => break 'run done.map(drop),
            };
            let (kept, answer) = crate::macros::answer(compilation, ask, cx.release());
            cx = Context::new(&compilation.outline, &compilation.names, kept);
            match answer {
                Ok(answer) => demand.strict = answer != Answer::Nothing,
                Err(error) => break 'run Err(error),
            }
            pending = Some(asked);
        }
    };
    demand.strict = true;
    demand.kill_children();
    demand.store = cx.release().0;
    outcome
}

impl Jit {
    pub fn new(store: Cache) -> Result<Jit> {
        let mut jit = Self::create(store, (SHARED_BYTES, STATE_BYTES, CODE_BYTES, TABLE_SLOTS))?;
        jit.ahead = true;
        Ok(jit)
    }

    pub(crate) fn store(&self) -> &Cache {
        &self.store
    }

    fn create(store: Cache, (shared_bytes, state_bytes, code_bytes, table_slots): (usize, usize, usize, usize)) -> Result<Jit> {
        let mut reached = Reached::default();
        reached.set_incremental(true);
        let mut image = Image::create(&reached, super::ImageMode::Jit, (shared_bytes, state_bytes, table_slots))?;
        image.backend = super::ImageBackend::LazyJit { fast: false };
        image.stubs.reserve(table_slots * stub::STUB_BYTES)?;
        image.pass.memory.reserve(code_bytes)?;
        Ok(Jit {
            reached,
            store,
            image,
            stubs: Vec::new(),
            compiled: HashSet::default(),
            children: Vec::new(),
            pending: std::collections::VecDeque::new(),
            strict: true,
            expanded: HashSet::default(),
            ahead: false,
            tried: HashSet::default(),
            watcher: None,
            edited: false,
        })
    }

    fn forget(&mut self, outline: &Outline, edited: Edited<'_>) {
        let touched = self.store.invalidate(outline, self.reached.list.iter().map(|(instance, _)| instance), edited);
        if !touched.is_empty() {
            self.image.reset_caches();
        }
        for at in touched {
            self.reached.forget(at);
            self.compiled.remove(&at);
            self.image.slot(at).store(self.stubs[at], Ordering::Release);
        }
        self.image.link();
        self.expanded.clear();
        self.tried.clear();
    }

    /// Waits for the watched files to settle, then takes back the code of the functions whose bodies `driver` finds changed, so the child asks for each again on its next call and gets it compiled from the new text; keeps the old code when the edit needs a restart. Takes and gives back the inference store.
    fn reload(&mut self, compilation: &mut Compilation, driver: &mut dyn Driver, store: Cache) -> Cache {
        if let Some(watcher) = &mut self.watcher
            && let Err(error) = watcher.settle()
        {
            eprintln!("hot reload: stopped watching: {error:#}");
            self.watcher = None;
        }
        let started = std::time::Instant::now();
        self.store = store;
        match driver.edited(compilation) {
            Ok(stale) if stale.is_empty() => {}
            Ok(stale) => {
                let had = self.compiled.len();
                self.forget(&compilation.outline, Edited::Files(&stale));
                eprintln!("hot reload: {} functions in {:.1} ms", had - self.compiled.len(), started.elapsed().as_secs_f64() * 1e3);
            }
            Err(why) => eprintln!("hot reload: the edit needs a restart, the old code keeps running: {why}"),
        }
        std::mem::take(&mut self.store)
    }

    /// The index of `instance`, whose slot holds its stub when it is new.
    pub fn declare(&mut self, instance: Instance) -> usize {
        let (at, new) = self.reached.declare(instance);
        if new {
            let slot = self.image.slot(at).as_ptr() as usize;
            let code = self.image.stubs.write(&stub::code(at, slot)).unwrap_or_else(|error| panic!("{error}"));
            self.image.slot(at).store(code, Ordering::Release);
            self.stubs.push(code);
        }
        at
    }

    /// Forks a child that runs `body` on the image, with an output pipe when `piped`, and exits with the status `body` returns; returns the child's id.
    #[cfg(unix)]
    pub fn fork(&mut self, piped: bool, body: impl FnOnce(&Image, Option<std::fs::File>) -> i32) -> Result<usize> {
        let Some(stubs) = self.image.stubs.span() else { bail!("no stub memory") };
        let (ask, answer) = (super::pipe()?, super::pipe()?);
        let out = if piped { Some(super::pipe()?) } else { None };
        stub::connect(self.image.table, stubs, (ask[1], answer[0]));
        let image = &self.image;
        let pid = super::fork_child(|| {
            // SAFETY: the child closes the ends that only the compiler uses and owns the write end of its output pipe from here on.
            let out = unsafe {
                libc::close(ask[0]);
                libc::close(answer[1]);
                out.map(|[read, write]| {
                    libc::close(read);
                    std::fs::File::from_raw_fd(write)
                })
            };
            body(image, out)
        })?;
        // SAFETY: the parent owns ask[0], answer[1] and the read end of the output pipe from here on, and closes the ends that only the child uses.
        let child = unsafe {
            libc::close(ask[1]);
            libc::close(answer[0]);
            let out = out.map(|[read, write]| {
                libc::close(write);
                std::fs::File::from_raw_fd(read)
            });
            Child { pid, asks: Some(std::fs::File::from_raw_fd(ask[0])), answers: std::fs::File::from_raw_fd(answer[1]), out, bytes: Vec::new(), failure: None }
        };
        self.children.push(Some(child));
        Ok(self.children.len() - 1)
    }

    #[cfg(not(unix))]
    pub fn fork(&mut self, _piped: bool, _body: impl FnOnce(&Image, Option<std::fs::File>) -> i32) -> Result<usize> {
        bail!("lazy forked JIT execution requires Unix; use `haste build -o` on Windows")
    }

    /// Kills and reaps every child still running, as a run that stopped on an error leaves them waiting for answers.
    fn kill_children(&mut self) {
        self.pending.clear();
        #[cfg(unix)]
        for child in self.children.drain(..).flatten() {
            // SAFETY: pid is a child this process forked and has not reaped, so the signal reaches only that child.
            unsafe { libc::kill(child.pid, libc::SIGKILL) };
            wait(child.pid);
        }
        #[cfg(not(unix))]
        self.children.clear();
    }

    /// The children with something to read: `(id, true)` for an ask, `(id, false)` for output; notes when a watched file changed.
    #[cfg(unix)]
    fn ready(&mut self) -> Result<Vec<(usize, bool)>> {
        let mut polled = Vec::new();
        let mut fds = Vec::new();
        for (id, child) in self.children.iter().enumerate().filter_map(|(id, child)| Some((id, child.as_ref()?))) {
            for (file, asking) in [(child.asks.as_ref(), true), (child.out.as_ref(), false)] {
                if let Some(file) = file {
                    polled.push((id, asking));
                    fds.push(libc::pollfd { fd: file.as_raw_fd(), events: libc::POLLIN, revents: 0 });
                }
            }
        }
        if let Some(watcher) = &self.watcher {
            fds.push(libc::pollfd { fd: watcher.fd(), events: libc::POLLIN, revents: 0 });
        }
        // SAFETY: fds is a live array of fds.len() pollfd entries.
        if unsafe { libc::poll(fds.as_mut_ptr(), fds.len() as libc::nfds_t, -1) } < 0 {
            let error = std::io::Error::last_os_error();
            if error.kind() == std::io::ErrorKind::Interrupted {
                return Ok(Vec::new());
            }
            bail!("poll failed: {error}");
        }
        self.edited |= self.watcher.is_some() && fds.last().is_some_and(|fd| fd.revents != 0);
        Ok(polled.into_iter().zip(fds).filter(|(_, fd)| fd.revents != 0).map(|(ready, _)| ready).collect())
    }

    #[cfg(not(unix))]
    fn ready(&mut self) -> Result<Vec<(usize, bool)>> {
        Ok(Vec::new())
    }

    #[cfg(unix)]
    fn drain(&mut self, id: usize) {
        let Some(child) = self.children[id].as_mut() else {
            return;
        };
        let mut buffer = [0; 1 << 16];
        match child.out.as_mut().map(|out| out.read(&mut buffer)) {
            Some(Ok(read)) if read > 0 => child.bytes.extend_from_slice(&buffer[..read]),
            _ => child.out = None,
        }
    }

    #[cfg(not(unix))]
    fn drain(&mut self, _id: usize) {}

    #[cfg(unix)]
    fn read_ask(&mut self, id: usize) -> Option<usize> {
        let child = self.children[id].as_mut()?;
        let mut index = [0; 8];
        match child.asks.as_mut()?.read_exact(&mut index) {
            Ok(()) => Some(u64::from_le_bytes(index) as usize),
            Err(_) => {
                child.asks = None;
                None
            }
        }
    }

    #[cfg(not(unix))]
    fn read_ask(&mut self, _id: usize) -> Option<usize> {
        None
    }

    #[cfg(unix)]
    fn reply(&mut self, id: usize, code: usize) {
        let Some(child) = self.children[id].as_mut() else {
            return;
        };
        if child.answers.write_all(&(code as u64).to_le_bytes()).is_err() {
            child.asks = None;
        }
    }

    #[cfg(not(unix))]
    fn reply(&mut self, _id: usize, _code: usize) {}

    /// The children whose pipes closed, waited for, with how each ended.
    #[cfg(unix)]
    fn reap(&mut self) -> Vec<(usize, Ended)> {
        let mut ended = Vec::new();
        for (id, slot) in self.children.iter_mut().enumerate() {
            if slot.as_ref().is_some_and(|child| child.asks.is_none() && child.out.is_none()) {
                let child = slot.take().unwrap_or_else(|| unreachable!());
                ended.push((id, Ended { status: wait(child.pid), bytes: child.bytes, failure: child.failure }));
            }
        }
        ended
    }

    #[cfg(not(unix))]
    fn reap(&mut self) -> Vec<(usize, Ended)> {
        Vec::new()
    }

    /// Answers with `cx` the asks of the children `driver` forked, the ask `pending` first, until no child is left, an answer needs what inference asked for, which returns that child and the function it asked for, or a watched file changed.
    fn pump(&mut self, cx: &mut Context<'_>, driver: &mut dyn Driver, mut pending: Option<(usize, usize)>) -> Result<Option<Pause>> {
        while self.children.iter().any(Option::is_some) {
            let asked = match pending.take() {
                Some((id, index)) if self.respond(cx, id, index) => {
                    self.strict = true;
                    None
                }
                Some(asked) => Some(asked),
                None => self.round(cx)?,
            };
            if let Some((id, index)) = asked {
                return Ok(Some(Pause::Asked(id, index)));
            }
            for (id, ended) in self.reap() {
                driver.ended(id, ended)?;
            }
            driver.start(self)?;
            if std::mem::take(&mut self.edited) {
                return Ok(Some(Pause::Edited));
            }
        }
        Ok(None)
    }

    /// Reads what the children wrote and answers their asks once, draining every ready pipe and queuing sibling asks if an answer needs what inference asked for.
    fn round(&mut self, cx: &mut Context<'_>) -> Result<Option<(usize, usize)>> {
        if self.pending.is_empty() {
            for (id, asking) in self.ready()? {
                if !asking {
                    self.drain(id);
                } else if let Some(index) = self.read_ask(id) {
                    self.pending.push_back((id, index));
                }
            }
        }
        while let Some((id, index)) = self.pending.pop_front() {
            if !self.respond(cx, id, index) {
                return Ok(Some((id, index)));
            }
        }
        Ok(None)
    }

    /// Replies to child `id`'s ask for function `index` with its code, or with zero and why it has none; false when the answer needs what inference asked for.
    fn respond(&mut self, cx: &mut Context<'_>, id: usize, index: usize) -> bool {
        let reply = match self.answer(cx, index) {
            Ok(reply) => reply,
            Err(Stop::Failed(error, item)) => {
                if let Some(child) = self.children[id].as_mut() {
                    child.failure = Some((error, item));
                }
                0
            }
            Err(Stop::Asked) => return false,
        };
        self.reply(id, reply);
        true
    }

    fn answer(&mut self, cx: &mut Context<'_>, index: usize) -> Result<usize, Stop> {
        if index & (stub::TIER_UP_BIT as usize) != 0 {
            let real = index & !(stub::TIER_UP_BIT as usize);
            let item = self.reached.list[real].0.0;
            let code = self.image.tier_up(real).map_err(|err| Stop::Failed(format!("{}: {err:#}", cx.name(item)), item))?;
            super::perf::note(cx, &self.reached, (real, code), super::perf::Origin::Haste);
            return Ok(code);
        }
        let mut journal = Journal::new();
        let answer = self.code(cx, index, &mut journal);
        if matches!(answer, Err(Stop::Asked)) {
            journal.iter().for_each(|&at| self.reached.forget(at));
            self.expanded.clear();
            self.tried.clear();
            self.image.bodies.clear();
        } else if answer.is_ok() {
            self.compile_ahead(cx, index);
        }
        answer
    }

    /// Compiles ahead, on every core, the functions without code that function `index` reaches through calls, up to `AHEAD_FUNCTIONS` of them, but those a prepped library binds; one whose inference or compile asks for more, defers, or fails keeps its stub, so it compiles, or fails, on its first call as it would have.
    fn compile_ahead(&mut self, cx: &mut Context<'_>, index: usize) {
        if !self.ahead {
            return;
        }
        let members = self.frontier(cx, index);
        let (foreign, mut work): (Vec<usize>, Vec<usize>) = members.into_iter().filter(|&at| self.callees_typed(cx, at)).partition(|&at| cx.foreign(self.reached.list[at].0.0));
        for at in foreign {
            let reached = &self.reached;
            match self.image.with_pass(|image, pass| image.bind_foreign(cx, reached, at, Some(&mut pass.memory))) {
                Ok(Foreign::Asm(code) | Foreign::Host(_, code)) => {
                    self.publish(at, code);
                }
                Err(_) => {
                    self.tried.insert(at);
                }
            }
        }
        work.retain(|&at| crate::prep::bind(cx, &self.reached, at).is_none());
        if work.is_empty() {
            return;
        }
        let compiled = self.image.compile_batch(cx, &self.reached, &work);
        for &(at, code) in &compiled {
            super::perf::note(cx, &self.reached, (at, code), super::perf::Origin::Haste);
            self.publish(at, code);
        }
        self.tried.extend(work.into_iter().filter(|at| !self.compiled.contains(at)));
    }

    /// The inferred functions without code that function `index` reaches through calls, level by level until `AHEAD_FUNCTIONS` or no new one, each declared and inferred on every core; one whose inference asked for more or failed is left to its first call, with what it calls.
    fn frontier(&mut self, cx: &mut Context<'_>, index: usize) -> Vec<usize> {
        let mut seen = HashSet::default();
        seen.insert(index);
        let mut members = Vec::new();
        let mut level = self.fresh_callees(cx, index, &mut seen);
        while !level.is_empty() && members.len() < AHEAD_FUNCTIONS {
            level.truncate(AHEAD_FUNCTIONS - members.len());
            let inferred = infer_quiet(cx, &mut self.reached, &level);
            self.image.bind_foreign_statics(cx, &self.reached, &inferred);
            let (typed, left): (Vec<usize>, Vec<usize>) = level.into_iter().partition(|&at| self.reached.list[at].1.is_ok());
            self.tried.extend(left);
            level = typed;
            let mut next = Vec::new();
            for &at in &level {
                next.extend(self.fresh_callees(cx, at, &mut seen));
            }
            members.append(&mut level);
            level = next;
        }
        members
    }

    /// The functions that function `at` calls, declared, that have no code, were not left to their first call, and are not in `seen`, which takes them.
    fn fresh_callees(&mut self, cx: &Context<'_>, at: usize, seen: &mut HashSet<usize>) -> Vec<usize> {
        let callees: Vec<Instance> = calls(cx, &self.reached.list[at].1).collect();
        let mut fresh = Vec::new();
        for callee in callees {
            let callee = self.declare(callee);
            if !self.compiled.contains(&callee) && !self.tried.contains(&callee) && seen.insert(callee) {
                fresh.push(callee);
            }
        }
        fresh
    }

    /// Whether every function that function `at` calls is inferred, so lowering it reaches no function without a body.
    fn callees_typed(&self, cx: &Context<'_>, at: usize) -> bool {
        calls(cx, &self.reached.list[at].1).all(|callee| self.reached.index.get(&callee).is_some_and(|&callee| !self.reached.is_untyped(callee)))
    }

    /// Infers, lowers and encodes function `index` unless it has code, and returns its code.
    fn code(&mut self, cx: &mut Context<'_>, index: usize, journal: &mut Journal) -> Result<usize, Stop> {
        let item = self.reached.list[index].0.0;
        if cx.foreign(item) {
            return self.foreign_code(cx, index, journal);
        }
        if let Some(code) = self.prepped(cx, index, journal) {
            let origin = if crate::prep::bind(cx, &self.reached, index) == Some(code) { super::perf::Origin::Rustc } else { super::perf::Origin::Adapter };
            super::perf::note(cx, &self.reached, (index, code), origin);
            return Ok(self.publish(index, code));
        }
        let mut callees = self.infer(cx, index, journal)?;
        while !self.compiled.contains(&index) {
            match self.image.compile_one(cx, &self.reached, index) {
                Ok(code) => {
                    super::perf::note(cx, &self.reached, (index, code), super::perf::Origin::Haste);
                    self.publish(index, code);
                }
                Err(error) => {
                    let wanted = self.image.take_wanted();
                    if wanted.is_empty() {
                        return Err(self.stop(cx, self.failure(cx, index, &callees, &error), item));
                    }
                    callees.extend(self.reach(cx, wanted, journal)?);
                }
            }
        }
        Ok(self.image.slot(index).load(Ordering::Acquire))
    }

    /// Binds the foreign function `index` once the functions its `global_asm!` block calls are reached, publishes its code and returns it.
    fn foreign_code(&mut self, cx: &mut Context<'_>, index: usize, journal: &mut Journal) -> Result<usize, Stop> {
        let item = self.reached.list[index].0.0;
        let wanted: Vec<Instance> = calls(cx, &self.reached.list[index].1).collect();
        if !wanted.is_empty() {
            self.reach(cx, wanted, journal)?;
        }
        let reached = &self.reached;
        let bound = self.image.with_pass(|image, pass| image.bind_foreign(cx, reached, index, Some(&mut pass.memory)));
        match bound.map_err(|error| Stop::Failed(error, item))? {
            Foreign::Asm(code) | Foreign::Host(_, code) => Ok(self.publish(index, code)),
        }
    }

    /// Publishes `code` as function `index`'s code, in its slot and every table that names it, marks it compiled and returns it.
    fn publish(&mut self, index: usize, code: usize) -> usize {
        self.image.publish(index, code);
        self.compiled.insert(index);
        code
    }

    /// The address of the prepped code that runs function `index`, inferring it first, when the bound prepped library holds it: rustc's code, or an adapter to it written here.
    fn prepped(&mut self, cx: &mut Context<'_>, index: usize, journal: &mut Journal) -> Option<usize> {
        if infer_declared(cx, &mut self.reached, index) {
            journal.push(index);
        }
        match crate::prep::prepped(cx, &self.reached, index)? {
            crate::prep::Prepped::Direct(address) => Some(address),
            crate::prep::Prepped::Adapter(func) => {
                let code = haste_codegen::compile(func, &crate::host_target()).ok()?;
                self.image.with_pass(|_, pass| pass.memory.write_patched(&code, &[])).ok()
            }
        }
    }

    /// Infers function `index`, every function it calls, and every function that a callee calls copies inline, declaring those, and returns their indexes.
    fn infer(&mut self, cx: &mut Context<'_>, index: usize, journal: &mut Journal) -> Result<Vec<usize>, Stop> {
        if infer_declared(cx, &mut self.reached, index) {
            journal.push(index);
        }
        if let Err(error) = &self.reached.list[index].1 {
            return Err(self.stop(cx, error.clone(), self.reached.list[index].0.0));
        }
        let wanted: Vec<Instance> = calls(cx, &self.reached.list[index].1).collect();
        let mut callees = self.reach(cx, wanted, journal)?;
        if self.image.allow_inline() {
            let mut seen: rustc_hash::FxHashSet<usize> = callees.iter().copied().collect();
            let mut next = 0;
            while let Some(&at) = callees.get(next) {
                next += 1;
                if !self.expanded.insert(at) || !crate::lower::may_inline(cx, &self.reached, at) {
                    continue;
                }
                let inner: Vec<Instance> = calls(cx, &self.reached.list[at].1).collect();
                let reached = self.reach(cx, inner, journal)?;
                callees.extend(reached.into_iter().filter(|callee| seen.insert(*callee)));
            }
        }
        Ok(callees)
    }

    /// Configures whether this on-demand image compiles for one-shot host execution without cross-crate inlining.
    pub(crate) fn set_fast(&mut self, fast: bool) {
        if let super::ImageBackend::EagerJit { fast: f } | super::ImageBackend::LazyJit { fast: f } = &mut self.image.backend {
            *f = fast;
        }
    }

    /// Configures whether each stub ask speculatively compiles reachable callees ahead of their first call.
    pub(crate) fn set_ahead(&mut self, ahead: bool) {
        self.ahead = ahead;
    }

    /// Declares and infers the functions `wanted`, and returns their indexes.
    fn reach(&mut self, cx: &mut Context<'_>, wanted: Vec<Instance>, journal: &mut Journal) -> Result<Vec<usize>, Stop> {
        let mut callees = Vec::with_capacity(wanted.len());
        let mut untyped = Vec::new();
        for instance in wanted {
            let at = self.declare(instance);
            callees.push(at);
            if self.reached.is_untyped(at) {
                untyped.push(at);
            }
        }
        if untyped.len() >= 4 && crate::jobs() > 1 {
            let kept = infer_quiet(cx, &mut self.reached, &untyped);
            if !kept.is_empty() {
                self.image.bind_foreign_statics(cx, &self.reached, &kept);
                journal.extend(kept);
            }
        }
        for at in untyped {
            if infer_declared(cx, &mut self.reached, at) {
                journal.push(at);
                self.image.bind_foreign_statics(cx, &self.reached, &[at]);
            }
        }
        if self.strict && !cx.resolved() {
            return Err(Stop::Asked);
        }
        Ok(callees)
    }

    /// Why function `index` did not compile: `error`, and the errors of the functions it calls that did not type-check.
    fn failure(&self, cx: &Context<'_>, index: usize, callees: &[usize], error: &anyhow::Error) -> String {
        let broken = callees.iter().filter_map(|&at| self.reached.list[at].1.as_ref().err());
        std::iter::once(format!("{}: {error:#}", cx.name(self.reached.list[index].0.0))).chain(broken.cloned()).collect::<Vec<_>>().join("; ")
    }

    fn stop(&self, cx: &Context<'_>, error: String, item: ItemRef) -> Stop {
        if self.strict && !cx.resolved() { Stop::Asked } else { Stop::Failed(error, item) }
    }
}

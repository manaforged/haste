//! Programs that compile each function on its first call: each runs in a forked child whose call table starts at stubs, and this process infers, lowers and encodes each function a stub asks for.

use std::io::{Read, Write};
use std::os::fd::{AsRawFd, FromRawFd};
use std::sync::atomic::Ordering;

use anyhow::{Result, anyhow, bail};
use rustc_hash::FxHashSet as HashSet;

use crate::macros::Compilation;
use crate::names::ItemRef;
use crate::names::path::Requests;
use crate::sema::reach::{Instance, Reached, calls, infer_declared, main_roots};
use crate::sema::{Context, Cache};

use super::{CODE_BYTES, Image, TABLE_SLOTS, host_symbol, stub, wait};

/// What answers what a pass of inference asked for, and says whether anything new loaded.
pub type Answerer<'f> = &'f dyn Fn(&mut Compilation, Requests) -> Result<bool>;

/// Why a function has no code yet: inference asked for names or macro output not loaded, or it did not compile.
enum Stop {
    Asked,
    Failed(String, ItemRef),
}

/// The functions one answer inferred, taken back when inference asked for more.
type Journal = Vec<usize>;

/// A running child: its pid, the pipe ends that carry its asks and the answers, its output pipe, what it wrote there, and why the function it last asked for has no code.
struct Child {
    pid: libc::pid_t,
    asks: Option<std::fs::File>,
    answers: std::fs::File,
    out: Option<std::fs::File>,
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
}

/// The functions declared so far, the inference store whose types their instances name, the image that holds their stubs and code, and the children that run it.
pub struct Jit {
    reached: Reached,
    store: Cache,
    image: Image,
    compiled: HashSet<usize>,
    children: Vec<Option<Child>>,
    strict: bool,
    expanded: HashSet<usize>,
}

/// Runs the binary whose `main` is `main` with `args` as argv, starting inference from `store`, compiling each function when the program first calls it, and returns the program's exit status.
pub fn run_on_demand(compilation: &mut Compilation, (store, main): (Cache, ItemRef), args: &[String]) -> Result<i32> {
    let (mut entries, store) = entries(compilation, (store, &[Some(main)]))?;
    let roots = entries.pop().unwrap_or_else(|| Err("no `main`".to_owned())).map_err(|error| anyhow!(error))?;
    let mut demand = Jit::new(store)?;
    let [start, main] = roots.map(|root| demand.declare(root));
    let mut program = Program { entry: (start, main), args, status: None };
    serve(compilation, &crate::macros::load, &mut demand, &mut program)?;
    eprintln!("compiled {} of {} functions declared", demand.compiled.len(), demand.reached.list.len());
    let status = program.status.ok_or_else(|| anyhow!("the program did not start"));
    std::mem::forget(demand);
    status
}

/// The program `main` runs, and its exit status once it ends.
struct Program<'a> {
    entry: (usize, usize),
    args: &'a [String],
    status: Option<i32>,
}

impl Driver for Program<'_> {
    fn start(&mut self, demand: &mut Jit) -> Result<()> {
        if self.status.is_none() && demand.children.is_empty() {
            let (entry, args) = (self.entry, self.args);
            demand.fork(false, |image, _| image.enter_main(entry, args))?;
        }
        Ok(())
    }

    fn ended(&mut self, _: usize, ended: Ended) -> Result<()> {
        if let Some((failure, _)) = ended.failure {
            eprintln!("{failure}");
        }
        self.status = Some(ended.status);
        Ok(())
    }
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
    driver.start(demand)?;
    while demand.children.iter().any(Option::is_some) {
        for (id, asking) in demand.ready()? {
            if !asking {
                demand.drain(id);
                continue;
            }
            let Some(index) = demand.read_ask(id) else { continue };
            let reply = loop {
                match demand.answer(&mut cx, index) {
                    Ok(reply) => break reply,
                    Err(Stop::Failed(error, item)) => {
                        if let Some(child) = demand.children[id].as_mut() {
                            child.failure = Some((error, item));
                        }
                        break 0;
                    }
                    Err(Stop::Asked) => {
                        let (mut kept, asked) = cx.release();
                        demand.strict = ask(compilation, asked)?;
                        kept.forget_lookups();
                        cx = Context::new(&compilation.outline, &compilation.names, kept);
                    }
                }
            };
            demand.strict = true;
            demand.reply(id, reply);
        }
        for (id, ended) in demand.reap() {
            driver.ended(id, ended)?;
        }
        driver.start(demand)?;
    }
    demand.store = cx.release().0;
    Ok(())
}

impl Jit {
    /// A demand with no function declared that infers on from `store`, its stub and code memory mapped so that every child forked later shares them.
    pub fn new(store: Cache) -> Result<Jit> {
        let reached = Reached::default();
        let mut image = Image::new(&reached)?;
        image.stubs.reserve(TABLE_SLOTS * stub::STUB_BYTES)?;
        image.memory.reserve(CODE_BYTES)?;
        Ok(Jit { reached, store, image, compiled: HashSet::default(), children: Vec::new(), strict: true, expanded: HashSet::default() })
    }

    /// The index of `instance`, whose slot holds its stub when it is new.
    pub fn declare(&mut self, instance: Instance) -> usize {
        let (at, new) = self.reached.declare(instance);
        if new {
            let code = self.image.stubs.write(&stub::code(at)).unwrap_or_else(|error| panic!("{error}"));
            self.image.slot(at).store(code, Ordering::Release);
        }
        at
    }

    /// Forks a child that runs `body` on the image, with an output pipe when `piped`, and exits with the status `body` returns; returns the child's id.
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

    /// The children with something to read: `(id, true)` for an ask, `(id, false)` for output.
    fn ready(&self) -> Result<Vec<(usize, bool)>> {
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
        // SAFETY: fds is a live array of fds.len() pollfd entries.
        if unsafe { libc::poll(fds.as_mut_ptr(), fds.len() as libc::nfds_t, -1) } < 0 {
            let error = std::io::Error::last_os_error();
            if error.kind() == std::io::ErrorKind::Interrupted {
                return Ok(Vec::new());
            }
            bail!("poll failed: {error}");
        }
        Ok(polled.into_iter().zip(fds).filter(|(_, fd)| fd.revents != 0).map(|(ready, _)| ready).collect())
    }

    fn drain(&mut self, id: usize) {
        let Some(child) = self.children[id].as_mut() else { return };
        let mut buffer = [0; 1 << 16];
        match child.out.as_mut().map(|out| out.read(&mut buffer)) {
            Some(Ok(read)) if read > 0 => child.bytes.extend_from_slice(&buffer[..read]),
            _ => child.out = None,
        }
    }

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

    fn reply(&mut self, id: usize, code: usize) {
        let Some(child) = self.children[id].as_mut() else { return };
        if child.answers.write_all(&(code as u64).to_le_bytes()).is_err() {
            child.asks = None;
        }
    }

    /// The children whose pipes closed, waited for, with how each ended.
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

    /// The code of function `index`, or why it has none; what inference did is taken back when it asked for more.
    fn answer(&mut self, cx: &mut Context<'_>, index: usize) -> Result<usize, Stop> {
        let mut journal = Journal::new();
        let answer = self.code(cx, index, &mut journal);
        if matches!(answer, Err(Stop::Asked)) {
            journal.iter().for_each(|&at| self.reached.forget(at));
            self.expanded.clear();
        }
        answer
    }

    /// Infers, lowers and encodes function `index` unless it has code, and returns its code.
    fn code(&mut self, cx: &mut Context<'_>, index: usize, journal: &mut Journal) -> Result<usize, Stop> {
        let item = self.reached.list[index].0.0;
        if cx.foreign(item) {
            let code = host_symbol(cx, item).map_err(|error| Stop::Failed(error, item))?;
            self.image.slot(index).store(code, Ordering::Release);
            self.compiled.insert(index);
            return Ok(code);
        }
        let mut callees = self.infer(cx, index, journal)?;
        while !self.compiled.contains(&index) {
            match self.image.compile_one(cx, &self.reached, index) {
                Ok(code) => {
                    self.image.slot(index).store(code, Ordering::Release);
                    self.compiled.insert(index);
                    self.image.link();
                }
                Err(error) => {
                    let wanted = self.image.wanted.take();
                    if wanted.is_empty() {
                        return Err(Stop::Failed(self.failure(cx, index, &callees, &error), item));
                    }
                    callees.extend(self.reach(cx, wanted, journal)?);
                }
            }
        }
        Ok(self.image.slot(index).load(Ordering::Acquire))
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
        Ok(callees)
    }

    /// Declares and infers the functions `wanted`, and returns their indexes.
    fn reach(&mut self, cx: &mut Context<'_>, wanted: Vec<Instance>, journal: &mut Journal) -> Result<Vec<usize>, Stop> {
        let mut callees = Vec::new();
        for instance in wanted {
            let at = self.declare(instance);
            if infer_declared(cx, &mut self.reached, at) {
                journal.push(at);
                self.image.bind_foreign_statics(cx, &self.reached, &[at]);
            }
            callees.push(at);
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

mod child;
mod compile;
mod program;
pub use program::{Program, evaluator};

use std::sync::atomic::Ordering;

use anyhow::Result;
use rustc_hash::FxHashSet as HashSet;

use crate::macros::Compilation;
use crate::names::ItemRef;
use crate::names::path::Requests;
use crate::outline::Outline;
use crate::sema::reach::{Instance, Reached, main_roots};
use crate::sema::{Answer, Cache, Context, Edited};

use child::Child;
use compile::Stop;

use super::{CODE_BYTES, Image, SHARED_BYTES, STATE_BYTES, TABLE_SLOTS, stub};

/// What answers what a pass of inference asked for, and says what it added.
pub type Answerer<'f> = &'f dyn Fn(&mut Compilation, Requests) -> Result<Answer>;

/// What reads the watched files again after they change: the `(crate, file)` slots whose function bodies changed, or why the change needs a restart.
pub type Reread<'r> = &'r mut dyn FnMut(&mut Compilation) -> Result<Vec<(u32, u32)>, String>;

enum Pause {
    Asked(usize, usize),
    Edited,
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
pub fn entries(
    compilation: &mut Compilation,
    (store, mains): (Cache, &[Option<ItemRef>]),
) -> Result<(Vec<Result<[Instance; 2], String>>, Cache)> {
    crate::macros::resolve_sema(compilation, (store, crate::macros::load), |mut cx, last| {
        let roots = mains
            .iter()
            .map(|main| {
                main.ok_or_else(|| "no `fn main`".to_owned())
                    .and_then(|item| {
                        main_roots(&mut cx, item).map_err(|error| format!("{error:#}"))
                    })
            })
            .collect();
        let settled = cx.resolved();
        let released = cx.release();
        Ok(if settled || last {
            Ok((roots, released.0))
        } else {
            Err(released)
        })
    })
}

/// Answers the asks of the children `driver` forks on `demand` until none is left, answering what inference asks for with `ask`; `demand` keeps what inference did.
pub fn serve(
    compilation: &mut Compilation,
    ask: Answerer<'_>,
    demand: &mut Jit,
    driver: &mut dyn Driver,
) -> Result<()> {
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
    /// Creates an on-demand JIT initialized with `store`.
    pub fn new(store: Cache) -> Result<Jit> {
        let jit = Self::create(store, (SHARED_BYTES, STATE_BYTES, CODE_BYTES, TABLE_SLOTS))?;
        Ok(jit)
    }

    pub(crate) fn store(&self) -> &Cache {
        &self.store
    }

    pub(crate) fn image(&self) -> &Image {
        &self.image
    }

    fn create(
        store: Cache,
        (shared_bytes, state_bytes, code_bytes, table_slots): (usize, usize, usize, usize),
    ) -> Result<Jit> {
        let mut reached = Reached::default();
        reached.set_incremental(true);
        let mut image = Image::create(
            &reached,
            super::ImageMode::Jit,
            (shared_bytes, state_bytes, table_slots),
        )?;
        image.backend = super::ImageBackend::LazyJit {
            fast: false,
            optimize: false,
        };
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
        let touched = self.store.invalidate(
            outline,
            self.reached.list.iter().map(|(instance, _)| instance),
            edited,
        );
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

    fn reload(
        &mut self,
        compilation: &mut Compilation,
        driver: &mut dyn Driver,
        store: Cache,
    ) -> Cache {
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
                eprintln!(
                    "hot reload: {} functions in {:.1} ms",
                    had - self.compiled.len(),
                    started.elapsed().as_secs_f64() * 1e3
                );
            }
            Err(why) => {
                eprintln!("hot reload: the edit needs a restart, the old code keeps running: {why}")
            }
        }
        std::mem::take(&mut self.store)
    }

    /// The index of `instance`, whose slot holds its stub when it is new.
    pub fn declare(&mut self, instance: Instance) -> usize {
        let (at, new) = self.reached.declare(instance);
        if new {
            let slot = self.image.slot(at).as_ptr() as usize;
            let code = self
                .image
                .stubs
                .write(&stub::code(at, slot))
                .unwrap_or_else(|error| panic!("{error}"));
            self.image.slot(at).store(code, Ordering::Release);
            self.stubs.push(code);
        }
        at
    }

    fn pump(
        &mut self,
        cx: &mut Context<'_>,
        driver: &mut dyn Driver,
        mut pending: Option<(usize, usize)>,
    ) -> Result<Option<Pause>> {
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
        self.reply(id, index, reply);
        true
    }

    pub(crate) fn set_fast(&mut self, fast: bool) {
        if let super::ImageBackend::EagerJit { fast: f }
        | super::ImageBackend::LazyJit { fast: f, .. } = &mut self.image.backend
        {
            *f = fast;
        }
    }

    pub(crate) fn set_optimize(&mut self, optimize: bool) {
        if let super::ImageBackend::LazyJit { optimize: o, .. } = &mut self.image.backend {
            *o = optimize;
        }
    }

    pub(crate) fn set_ahead(&mut self, ahead: bool) {
        self.ahead = ahead;
    }
}

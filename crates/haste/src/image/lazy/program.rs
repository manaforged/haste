//! High-level runners on top of `Jit`: persistent `Program` execution, one-shot `run_on_demand`, and constant-item `Evaluate`.

#[cfg(unix)]
use std::io::Write;
#[cfg(unix)]
use std::sync::atomic::Ordering;

use anyhow::{Result, anyhow, bail};

use crate::macros::Compilation;
use crate::names::ItemRef;
use crate::outline::Outline;
use crate::sema::evaluate::Evaluate;
use crate::sema::reach::Instance;
use crate::sema::{Cache, Context, Edited};

use super::super::Image;
use super::{Driver, Ended, Jit, Reread, entries, serve};

/// A binary kept between runs: its `main`, and each function it called, compiled on that first call.
pub struct Program {
    jit: Jit,
    entry: (usize, usize),
    env: Vec<(String, String)>,
}

impl Program {
    /// The binary whose `main` is `main`, inferring on from `store`, with nothing compiled yet.
    pub fn new(compilation: &mut Compilation, store: Cache, main: ItemRef) -> Result<Program> {
        let env = compilation.outline.crates[main.krate as usize]
            .env
            .vars
            .clone();
        let (mut entries, store) = entries(compilation, (store, &[Some(main)]))?;
        let roots = entries
            .pop()
            .unwrap_or_else(|| Err("no `main`".to_owned()))
            .map_err(|error| anyhow!(error))?;
        let mut jit = Jit::new(store)?;
        let [start, main] = roots.map(|root| jit.declare(root));
        Ok(Program {
            jit,
            entry: (start, main),
            env,
        })
    }

    /// Runs `main` with `args` as argv in a forked child, compiling each function it calls that has no code yet, and returns its exit status.
    pub fn run(&mut self, compilation: &mut Compilation, args: &[String]) -> Result<i32> {
        self.run_with(compilation, args, None)
    }

    /// Runs `main` like `run` while it watches the files at `paths`, and on each change points the functions whose bodies `reread` finds changed back at their stubs, so the running child compiles each again on its next call.
    pub fn run_watched(
        &mut self,
        compilation: &mut Compilation,
        args: &[String],
        (paths, reread): (Vec<std::path::PathBuf>, Reread<'_>),
    ) -> Result<i32> {
        self.jit.watcher = Some(super::super::watch::Watcher::new(paths)?);
        let status = self.run_with(compilation, args, Some(reread));
        self.jit.watcher = None;
        status
    }

    /// Runs `main` with `args` as argv, asking `reread` what changed when a watched file changes.
    fn run_with(
        &mut self,
        compilation: &mut Compilation,
        args: &[String],
        reread: Option<Reread<'_>>,
    ) -> Result<i32> {
        let mut run = Run {
            entry: self.entry,
            args,
            env: &self.env,
            reread,
            status: None,
        };
        let served = serve(compilation, &crate::macros::load, &mut self.jit, &mut run);
        compilation.settle();
        served?;
        if haste_codegen::trace_enabled() {
            eprintln!(
                "compiled {} of {} functions declared",
                self.jit.compiled.len(),
                self.jit.reached.list.len()
            );
            if let Some(counts) = crate::prep::decided() {
                eprintln!("prep {counts}");
            }
        }
        run.status
            .ok_or_else(|| anyhow!("the program did not start"))
    }

    /// Takes back the code and inference of the functions that the bodies `edited` touch, so that the next run compiles them again.
    pub fn forget(&mut self, outline: &Outline, edited: Edited<'_>) {
        self.jit.forget(outline, edited);
    }

    /// Inference store and JIT image for persisting newly frozen summaries and stencils.
    pub(crate) fn store_and_image(&self) -> (&Cache, &Image) {
        (self.jit.store(), self.jit.image())
    }

    /// The inference store, without what the crates `edited` touch.
    pub fn into_store(mut self, outline: &Outline, edited: Edited<'_>) -> Cache {
        self.jit.store.forget(outline, edited);
        std::mem::take(&mut self.jit.store)
    }
}

/// One run of a program's `main`, what reads its watched files again when it is watched, and its exit status once it ends.
struct Run<'a, 'r> {
    entry: (usize, usize),
    args: &'a [String],
    env: &'a [(String, String)],
    reread: Option<Reread<'r>>,
    status: Option<i32>,
}

impl Driver for Run<'_, '_> {
    fn start(&mut self, demand: &mut Jit) -> Result<()> {
        if self.status.is_none() && demand.children.is_empty() {
            let (entry, args, env) = (self.entry, self.args, self.env);
            demand.fork(false, |image, _| {
                for (key, value) in env {
                    // SAFETY: single-threaded forked child process before invoking user `main`.
                    unsafe { std::env::set_var(key, value) };
                }
                image.enter_main(entry, args)
            })?;
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

    fn edited(&mut self, compilation: &mut Compilation) -> Result<Vec<(u32, u32)>, String> {
        self.reread
            .as_mut()
            .map_or(Ok(Vec::new()), |reread| reread(compilation))
    }
}

/// The CPU time, in seconds, that running one constant item may take before its child is stopped.
const EVALUATION_SECONDS: i64 = 10;

/// The one child that runs a constant item, and how it ended.
struct Once(Option<Ended>);

impl Driver for Once {
    fn start(&mut self, _: &mut Jit) -> Result<()> {
        Ok(())
    }

    fn ended(&mut self, _: usize, ended: Ended) -> Result<()> {
        self.0 = Some(ended);
        Ok(())
    }
}

const EVAL_SHARED_BYTES: usize = 64 << 20;
const EVAL_STATE_BYTES: usize = 64 << 20;
const EVAL_CODE_BYTES: usize = 64 << 20;
const EVAL_TABLE_SLOTS: usize = 1 << 16;

/// Makes the runner of constant items that the type checker installs: a demand of its own, whose functions compile on their first call.
pub fn evaluator() -> Result<Box<dyn Evaluate>> {
    Ok(Box::new(Jit::create(
        Cache::default(),
        (
            EVAL_SHARED_BYTES,
            EVAL_STATE_BYTES,
            EVAL_CODE_BYTES,
            EVAL_TABLE_SLOTS,
        ),
    )?))
}

impl Evaluate for Jit {
    fn evaluate(&mut self, cx: &mut Context<'_>, instance: &Instance, bits: u32) -> Result<u128> {
        crate::with_host_target(|| {
            let (entry, wide) = (self.declare(*instance), bits > 64);
            let mut once = Once(None);
            self.strict = false;
            let ran = self
                .fork(true, |image, out| image.evaluate_in_child(entry, wide, out))
                .and_then(|_| self.pump(cx, &mut once, None));
            self.strict = true;
            self.kill_children();
            if ran?.is_some() {
                bail!("running it needs names that are not loaded yet");
            }
            value_of(once.0)
        })
    }
}

/// The value that the child running a constant wrote, or why it wrote none.
fn value_of(ended: Option<Ended>) -> Result<u128> {
    match ended {
        Some(Ended {
            failure: Some((error, _)),
            ..
        }) => bail!("{error}"),
        Some(Ended {
            status: 0, bytes, ..
        }) => Ok(u128::from_le_bytes(<[u8; 16]>::try_from(bytes).map_err(
            |bytes| anyhow!("it gave no value: {} bytes, not 16", bytes.len()),
        )?)),
        #[cfg(unix)]
        Some(Ended { status, .. }) if status == 128 + libc::SIGVTALRM => {
            bail!("it ran for more than {EVALUATION_SECONDS} s of CPU time")
        }
        Some(Ended { status, .. }) => bail!("it stopped with status {status}"),
        None => bail!("it did not run"),
    }
}

impl Image {
    /// Runs the constant item at slot `entry` in this forked child under a CPU time limit and writes its value, 128 bits wide when `wide`, to `out`; returns the exit status.
    #[cfg(unix)]
    fn evaluate_in_child(&self, entry: usize, wide: bool, out: Option<std::fs::File>) -> i32 {
        let Some(mut out) = out else { return 1 };
        let limit = libc::itimerval {
            it_interval: libc::timeval {
                tv_sec: 0,
                tv_usec: 0,
            },
            it_value: libc::timeval {
                tv_sec: EVALUATION_SECONDS as _,
                tv_usec: 0,
            },
        };
        // SAFETY: setitimer arms only this child's virtual timer, from a live itimerval.
        unsafe { libc::setitimer(libc::ITIMER_VIRTUAL, &limit, std::ptr::null_mut()) };
        let address = self.slot(entry).load(Ordering::Acquire);
        // SAFETY: address is the stub or code of a constant item of an integer type, whose native signature takes nothing and returns the value in registers, as an I128 when `wide`.
        let value = unsafe {
            match wide {
                true => std::mem::transmute::<usize, extern "C" fn() -> u128>(address)(),
                false => u128::from(std::mem::transmute::<usize, extern "C" fn() -> u64>(
                    address,
                )()),
            }
        };
        i32::from(out.write_all(&value.to_le_bytes()).is_err())
    }

    #[cfg(not(unix))]
    fn evaluate_in_child(&self, _entry: usize, _wide: bool, _out: Option<std::fs::File>) -> i32 {
        1
    }
}

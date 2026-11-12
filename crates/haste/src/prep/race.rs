use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result};

use crate::graph::{Targets, Workspace};

use super::bind::SNAPSHOT;
use super::edit::Snapshot;
use super::files;
use super::record::Parsed;

/// How long a run waits by default, once it is ready to start the program, for a prep it raced to land.
pub const WAIT: Duration = Duration::from_millis(250);

/// How long a waiting run sleeps between two looks at the prep it waits for.
const POLL: Duration = Duration::from_millis(1);

/// The file in a build's prep directory that holds what the last background prep wrote to standard error.
const LOG: &str = "rustc.log";

const PREP_ARGS: [&str; 2] = ["--release", "prep"];

static BUDGET: OnceLock<Duration> = OnceLock::new();

/// Sets how long a run waits for a prep it raced, before any run reads it; zero turns racing off, so a run binds what the store holds and leaves the prep to the next run.
pub fn set_wait(budget: Duration) {
    BUDGET.get_or_init(|| budget);
}

static PREPARED: AtomicBool = AtomicBool::new(false);

/// Lets runs under the release profile call code rustc prepared; without it every function runs code this compiler made.
pub fn set_prepared() {
    PREPARED.store(true, Ordering::Relaxed);
}

/// Whether runs may call code rustc prepared.
pub fn prepared() -> bool {
    PREPARED.load(Ordering::Relaxed)
}

fn budget() -> Duration {
    BUDGET.get().copied().unwrap_or(WAIT)
}

/// What a run under the release profile starts before it loads the build's sources: the build's prep directory, the link that names its library, what that link named then, the prep it races when a source changed since the library was built, with the process when the run started it, whether the run waited for it, and the reading of the library's manifest.
pub struct Ahead {
    dir: PathBuf,
    out: PathBuf,
    before: Option<PathBuf>,
    race: Option<Option<Child>>,
    waited: bool,
    manifest: Option<JoinHandle<Result<Parsed>>>,
}

/// Starts reading the manifest of the library the store holds for the build of `workspace` before the run loads its sources, and, when a file that library was built from was written since and the budget is not zero, the prep of the build in the background, or finds the one that runs.
pub fn ahead(workspace: &Workspace, targets: &Targets) -> Option<Ahead> {
    if !super::SUPPORTED {
        return None;
    }
    start(workspace, targets).unwrap_or_else(|error| {
        eprintln!("haste: not preparing ahead: {error:#}");
        None
    })
}

fn start(workspace: &Workspace, targets: &Targets) -> Result<Option<Ahead>> {
    let dir = super::dir(super::key(&super::pins(workspace, targets), workspace, super::toolchain()?)?);
    let out = dir.join(super::RUSTC);
    let before = std::fs::read_link(&out).ok();
    let manifest = out.canonicalize().ok().map(|at| at.join(files::manifest(files::LIBRARY))).filter(|path| path.is_file()).map(|path| std::thread::spawn(move || Parsed::read(&path)));
    let stale = !Snapshot::read(&out.join(SNAPSHOT)).is_ok_and(|then| then.sources.unchanged());
    let race = if stale && !budget().is_zero() {
        std::fs::create_dir_all(&dir)?;
        Some(spawn(&dir, workspace)?)
    } else {
        None
    };
    Ok(Some(Ahead { dir, out, before, race, waited: false, manifest }))
}

/// Starts `haste --release prep` on the build's member package in the background, with its standard error in the build's prep directory `dir`, unless a prep of the build runs now; the prep takes over the lock this takes, on its standard input, so no other run starts one in between.
pub(super) fn spawn(dir: &Path, workspace: &Workspace) -> Result<Option<Child>> {
    let Some(lock) = super::lock(dir, false)? else { return Ok(None) };
    let log = std::fs::File::create(dir.join(LOG))?;
    let mut command = Command::new(std::env::current_exe()?);
    command.args(PREP_ARGS).arg(super::root(workspace)?).stdin(lock).stdout(Stdio::null()).stderr(log);
    std::os::unix::process::CommandExt::process_group(&mut command, 0);
    Ok(Some(command.spawn().context("starting a background prep")?))
}

impl Ahead {
    /// Waits, when the run races a prep of a library the store holds, until it lands, or no prep runs, or the budget runs out, and says so on standard error only when the budget ran out; the program then binds whatever library the store names, and a first prep finishes for the next run.
    pub fn wait(&mut self) {
        if self.race.is_none() || self.before.is_none() || std::mem::replace(&mut self.waited, true) {
            return;
        }
        let deadline = Instant::now() + budget();
        while !self.landed() {
            if Instant::now() >= deadline {
                eprintln!("haste: rustc's code is not ready after {} ms, so Haste's code runs what changed", budget().as_millis());
                return;
            }
            std::thread::sleep(POLL);
        }
    }

    /// Whether the store names a library other than the one it named when the run started, or no prep of the build runs.
    fn landed(&mut self) -> bool {
        if std::fs::read_link(&self.out).ok() != self.before {
            return true;
        }
        let running = self.race.iter_mut().flatten().any(|child| matches!(child.try_wait(), Ok(None)));
        !running && super::lock(&self.dir, false).is_ok_and(|held| held.is_some())
    }

    /// What the prep this run started wrote to standard error when it failed, waiting for it as `wait` does unless the run waited already.
    pub fn rejected(&mut self) -> Option<String> {
        self.wait();
        let status = self.race.as_mut()?.as_mut()?.try_wait().ok()??;
        if status.success() {
            return None;
        }
        std::fs::read_to_string(self.dir.join(LOG)).ok()
    }

    /// The manifest read ahead, once read; none when the store held no library or reading it failed.
    pub(super) fn manifest(&mut self) -> Option<Parsed> {
        self.manifest.take()?.join().ok()?.ok()
    }
}

impl Drop for Ahead {
    fn drop(&mut self) {
        if let Some(child) = self.race.take().flatten() {
            reap(child);
        }
    }
}

/// Collects the exit of the background prep `child` once it ends, on a thread of its own, so a long run leaves no zombie and never waits for it.
pub(super) fn reap(mut child: Child) {
    std::thread::spawn(move || {
        if let Err(error) = child.wait() {
            eprintln!("haste: the background prep: {error}");
        }
    });
}

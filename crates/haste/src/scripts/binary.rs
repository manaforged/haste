//! Build scripts kept as executables between runs, as Cargo keeps them: a script that runs again starts its executable while its crate and every crate it builds from hash the same, and one that ran before is compiled ahead of time once it runs again.

use std::hash::BuildHasher as _;
use std::os::unix::process::CommandExt as _;
use std::path::{Path, PathBuf};

use rustc_hash::FxHashMap as HashMap;

use crate::graph::{Workspace, store};
use crate::image::Script;
use crate::macros::Compilation;
use crate::outline::Outline;
use crate::substrate::CrateHashes;

const EXECUTABLE: &str = "script";
const KEY: &str = "script.key";

/// What the store holds for a script whose crate hashes to its key: its executable, the mark that it does not compile ahead of time, or nothing.
pub(super) enum Kept {
    Executable(PathBuf),
    Failed,
    Missing,
}

/// The key of each script unit's executable: the hash of its crate and of every crate that crate reaches, under this toolchain, target, profile and compiler.
pub(super) struct Binaries {
    keys: HashMap<usize, u64>,
}

impl Binaries {
    /// The keys of the scripts `scripts`, each a unit and its crate in `outline`; none while no script of them ran before, as a first run keeps nothing.
    pub(super) fn new(workspace: &Workspace, outline: &Outline, scripts: &[(usize, usize)]) -> Binaries {
        if !scripts.iter().any(|&(unit, _)| ran_before(workspace, unit)) {
            return Binaries { keys: HashMap::default() };
        }
        let target = crate::host_target();
        let hashes = CrateHashes::compute(outline, crate::substrate::toolchain_key(workspace, &target), crate::substrate::target_hash(&target));
        Binaries { keys: scripts.iter().map(|&(unit, krate)| (unit, closure_key(outline, &hashes, krate))).collect() }
    }

    /// What the store holds for the script of `unit` under its key; the caller holds the unit's lock.
    pub(super) fn kept(&self, workspace: &Workspace, unit: usize) -> Kept {
        let dir = store::dir(workspace, unit);
        let Some(key) = self.keys.get(&unit) else { return Kept::Missing };
        if std::fs::read_to_string(dir.join(KEY)).ok().as_deref() != Some(key_text(*key).as_str()) {
            return Kept::Missing;
        }
        let executable = dir.join(EXECUTABLE);
        if executable.is_file() { Kept::Executable(executable) } else { Kept::Failed }
    }

    /// Compiles the scripts `scripts`, each a unit and the crate of its `main`, ahead of time into the store in a detached process that outlives this one, so no run waits on the compile; the keeper lets go of this process's unit locks `locks`.
    pub(super) fn keep_detached(&self, workspace: &Workspace, compilation: &mut Compilation, scripts: &[(usize, usize)], locks: &mut HashMap<usize, store::Lock>) -> anyhow::Result<()> {
        if scripts.is_empty() {
            return Ok(());
        }
        if cfg!(target_os = "macos") {
            locks.clear();
            for &(unit, krate) in scripts {
                drop(self.keep(workspace, compilation, unit, krate));
            }
            return Ok(());
        }
        let child = crate::image::fork_child(|| {
            locks.clear();
            let keeper = crate::image::fork_child(|| {
                detach();
                let kept = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| scripts.iter().filter(|&&(unit, krate)| self.keep(workspace, compilation, unit, krate).is_err()).count()));
                i32::from(!matches!(kept, Ok(0)))
            });
            i32::from(keeper.is_err())
        })?;
        match crate::image::wait(child) {
            0 => Ok(()),
            status => anyhow::bail!("the keeper did not start: status {status}"),
        }
    }

    /// Compiles the script of `unit`, whose `main` is in crate `krate`, ahead of time into the store under its key, or marks that it does not compile, holding the unit's lock.
    fn keep(&self, workspace: &Workspace, compilation: &mut Compilation, unit: usize, krate: usize) -> std::io::Result<()> {
        let Some(&key) = self.keys.get(&unit) else { return Ok(()) };
        let _lock = store::lock(workspace, unit)?;
        let dir = store::dir(workspace, unit);
        let executable = dir.join(EXECUTABLE);
        store::remove(&dir.join(KEY))?;
        store::remove(&executable)?;
        let staged = store::fresh(&executable);
        let built = compilation.outline.main(krate).ok_or_else(|| anyhow::anyhow!("no `fn main`")).and_then(|main| crate::build::emit_main(workspace, compilation, main, &staged));
        match built {
            Ok(()) => std::fs::rename(&staged, &executable)?,
            Err(_) => store::remove(&staged)?,
        }
        store::write_new(&dir.join(KEY), key_text(key))
    }
}

/// Detaches this forked keeper from the session and from every pipe and socket it shares with its parent, so neither a terminal signal nor a reader waiting for end of file ties the two together.
fn detach() {
    let shared: Vec<i32> = std::fs::read_dir("/dev/fd").map(|dir| dir.filter_map(|entry| entry.ok()?.file_name().to_str()?.parse().ok()).collect()).unwrap_or_default();
    for fd in shared.into_iter().filter(|&fd| fd > 2) {
        // SAFETY: fstat writes only into the zeroed local `stat`; a descriptor that is not open fails without effect.
        let mode = unsafe {
            let mut stat = std::mem::zeroed::<libc::stat>();
            if libc::fstat(fd, &mut stat) != 0 {
                continue;
            }
            stat.st_mode & libc::S_IFMT
        };
        if mode == libc::S_IFIFO || mode == libc::S_IFSOCK {
            // SAFETY: fd is a pipe or socket this keeper inherited and never uses; closing it drops only this process's reference.
            unsafe { libc::close(fd) };
        }
    }
    // SAFETY: setsid, alarm, open and dup2 take only the static NUL-terminated path and descriptors of this process; 0, 1 and 2 then point at /dev/null.
    unsafe {
        libc::setsid();
        libc::alarm(60);
        let null = libc::open(c"/dev/null".as_ptr(), libc::O_RDWR);
        if null >= 0 {
            for fd in 0..3 {
                libc::dup2(null, fd);
            }
            if null > 2 {
                libc::close(null);
            }
        }
    }
}

/// Drops the executable of `unit`'s script after it failed, so later runs compile the script on demand and give their own verdict; the caller holds the unit's lock.
pub(super) fn distrust(workspace: &Workspace, unit: usize) {
    if let Err(error) = store::remove(&store::dir(workspace, unit).join(EXECUTABLE)) {
        eprintln!("cannot drop the executable of {}'s build script: {error}", super::package_name(workspace, unit));
    }
}

/// Whether the script of `unit` ran before, so its `OUT_DIR` is there and a change it watches is what runs it again.
pub(super) fn ran_before(workspace: &Workspace, unit: usize) -> bool {
    store::dir(workspace, unit).join(super::OUT).is_dir()
}

/// Runs `script` from `executable` in this forked child with `out` as its standard output, as the image runs a script's `main`; returns the exit status when it cannot start.
pub(super) fn exec(executable: &Path, script: &Script, out: std::fs::File) -> i32 {
    let Ok(log) = std::fs::File::create(&script.log) else { return 101 };
    let error = std::process::Command::new(executable).arg0("build-script-build").current_dir(&script.dir).envs(script.vars.iter().map(|(key, value)| (key, value))).stdout(out).stderr(log).exec();
    eprintln!("cannot start {}: {error}", executable.display());
    127
}

/// The hash of crate `krate` and of every crate it reaches in `outline`, since a fixed crate's hash leaves out its build dependencies.
fn closure_key(outline: &Outline, hashes: &CrateHashes, krate: usize) -> u64 {
    let mut seen = vec![false; outline.crates.len()];
    let mut stack = vec![krate];
    let mut found = Vec::new();
    while let Some(at) = stack.pop() {
        if std::mem::replace(&mut seen[at], true) {
            continue;
        }
        found.push(hashes.get(at));
        stack.extend(outline.deps[at].iter().map(|(_, dep)| *dep));
    }
    found.sort_unstable();
    rustc_hash::FxBuildHasher.hash_one((hashes.get(krate), found))
}

fn key_text(key: u64) -> String {
    format!("{key:016x}")
}

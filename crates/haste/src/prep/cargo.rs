use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context as _, Result, bail};

use super::shim::Shim;

/// The settings every shim builds with, which no user or workspace configuration can change: release optimization, no checks, abort on panic, no extra compiler flags.
pub(super) const SETTINGS: [&str; 6] = ["build.rustflags=[]", "profile.release.opt-level=3", "profile.release.debug-assertions=false", "profile.release.overflow-checks=false", "profile.release.panic=\"abort\"", "profile.release.lto=false"];

const TOOLCHAIN_DEPTH: usize = 5;
const CARGO_BIN: &str = "bin/cargo";
const RUSTC_BIN: &str = "bin/rustc";
const VERSION_FLAG: &str = "-vV";
const CONFIG_FLAG: &str = "--config";
const OFFLINE_FLAG: &str = "--offline";
const ROUNDS: usize = 4;
const CLEARED: [&str; 4] = ["RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "RUSTC_WRAPPER", "CARGO_BUILD_TARGET"];
const ERROR: &str = "error";
const LIB_SOURCE: &str = "src/lib.rs";
const CARGO_MANIFEST: &str = "Cargo.toml";
const SOURCE_DIR: &str = "src";
const CONFIG: &str = ".cargo/config.toml";
const LOCK: &str = "Cargo.lock";

/// The toolchain whose standard library sources Haste lowers: its cargo, its rustc, and what `rustc -vV` says of it.
pub(super) struct Toolchain {
    pub cargo: PathBuf,
    pub rustc: PathBuf,
    pub version: String,
    pub library: PathBuf,
}

impl Toolchain {
    /// The toolchain that holds the library sources `crate::graph::library` finds.
    pub(super) fn find() -> Result<Toolchain> {
        let library = crate::graph::library()?;
        let root = library.ancestors().nth(TOOLCHAIN_DEPTH).with_context(|| format!("no toolchain holds {}", library.display()))?;
        let (cargo, rustc) = (root.join(CARGO_BIN), root.join(RUSTC_BIN));
        let version = output(Command::new(&rustc).arg(VERSION_FLAG))?;
        Ok(Toolchain { cargo, rustc, version, library })
    }
}

/// The standard output of `command`, or its standard error when it fails.
pub(super) fn output(command: &mut Command) -> Result<String> {
    let ran = command.stdin(Stdio::null()).output().with_context(|| format!("running {command:?}"))?;
    if !ran.status.success() {
        bail!("{command:?} failed: {}", String::from_utf8_lossy(&ran.stderr));
    }
    Ok(String::from_utf8_lossy(&ran.stdout).into_owned())
}

/// Where a shim builds: its crate directory, the target directory every shim shares, and the workspace whose lockfile and configuration it follows.
pub(super) struct Dirs<'d> {
    pub shim: &'d Path,
    pub target: &'d Path,
    pub workspace: &'d Path,
}

/// Builds `shim` in release mode, offline, dropping each wrapper rustc rejects and building again, and returns the library it made.
pub(super) fn build(toolchain: &Toolchain, dirs: &Dirs<'_>, shim: &mut Shim) -> Result<PathBuf> {
    std::fs::create_dir_all(dirs.shim.join(SOURCE_DIR))?;
    std::fs::write(dirs.shim.join(CARGO_MANIFEST), shim.manifest())?;
    let lock = dirs.workspace.join(LOCK);
    if lock.is_file() {
        std::fs::copy(&lock, dirs.shim.join(LOCK))?;
    }
    for _ in 0..ROUNDS {
        let (source, owners) = shim.source();
        std::fs::write(dirs.shim.join(LIB_SOURCE), source)?;
        let ran = cargo(toolchain, dirs).output().with_context(|| format!("running {}", toolchain.cargo.display()))?;
        if ran.status.success() {
            return Ok(dirs.target.join("release").join(format!("{}{}{}", std::env::consts::DLL_PREFIX, shim.name, std::env::consts::DLL_SUFFIX)));
        }
        let errors = String::from_utf8_lossy(&ran.stderr).into_owned();
        let rejected = rejected(&errors, &owners);
        if rejected.is_empty() {
            bail!("building the prep shim at {} failed: {errors}", dirs.shim.display());
        }
        rejected.into_iter().for_each(|at| shim.dropped[at] = true);
    }
    bail!("building the prep shim at {} still failed after {ROUNDS} rounds", dirs.shim.display())
}

fn cargo(toolchain: &Toolchain, dirs: &Dirs<'_>) -> Command {
    let mut command = release(toolchain, &dirs.shim.join(CARGO_MANIFEST), dirs.target);
    command.arg(OFFLINE_FLAG);
    settled(&mut command);
    let config = dirs.workspace.join(CONFIG);
    if config.is_file() {
        command.arg(CONFIG_FLAG).arg(config);
    }
    command.current_dir(dirs.shim);
    command
}

/// A release build of the package at `manifest` with `toolchain` into `target`, with short messages, no input, and none of the variables that would change how cargo calls rustc.
pub(super) fn release(toolchain: &Toolchain, manifest: &Path, target: &Path) -> Command {
    let mut command = Command::new(&toolchain.cargo);
    command.args(["build", "--release", "--message-format=short", "--manifest-path"]).arg(manifest);
    for name in CLEARED {
        command.env_remove(name);
    }
    command.env("CARGO_TARGET_DIR", target).env("RUSTC", &toolchain.rustc).stdin(Stdio::null());
    command
}

/// Gives `command` the settings that every prepped build uses.
pub(super) fn settled(command: &mut Command) {
    for setting in SETTINGS {
        command.args([CONFIG_FLAG, setting]);
    }
}

/// The wrappers that the errors in `stderr` point into, by the lines of `src/lib.rs` that `owners` maps to them.
fn rejected(stderr: &str, owners: &[Option<usize>]) -> Vec<usize> {
    let mut found: Vec<usize> = stderr
        .lines()
        .filter_map(|line| {
            let rest = line.strip_prefix(LIB_SOURCE)?.strip_prefix(':')?;
            let (number, rest) = rest.split_once(':')?;
            let (_, rest) = rest.split_once(':')?;
            rest.trim_start().starts_with(ERROR).then_some(())?;
            *owners.get(number.parse::<usize>().ok()?.checked_sub(1)?)?
        })
        .collect();
    found.sort_unstable();
    found.dedup();
    found
}

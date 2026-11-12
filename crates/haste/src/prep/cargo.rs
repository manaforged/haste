use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context as _, Result, bail};

/// The settings every prep build uses, which no user or workspace configuration can change: release optimization, no checks, abort on panic, no extra compiler flags.
pub(super) const SETTINGS: [&str; 6] = ["build.rustflags=[]", "profile.release.opt-level=3", "profile.release.debug-assertions=false", "profile.release.overflow-checks=false", "profile.release.panic=\"abort\"", "profile.release.lto=false"];

const TOOLCHAIN_DEPTH: usize = 5;
const CARGO_BIN: &str = "bin/cargo";
const RUSTC_BIN: &str = "bin/rustc";
const VERSION_FLAG: &str = "-vV";
const CONFIG_FLAG: &str = "--config";
const CLEARED: [&str; 4] = ["RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "RUSTC_WRAPPER", "CARGO_BUILD_TARGET"];

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

//! Haste: the build graph, name resolution, type inference, lowering, and the JIT that runs the program.

#![cfg_attr(not(unix), allow(dead_code))]

#[cfg(not(any(
    all(target_arch = "x86_64", target_os = "linux"),
    all(target_arch = "aarch64", target_os = "linux"),
    all(target_arch = "aarch64", target_os = "macos"),
    all(target_arch = "x86_64", target_os = "macos"),
    all(target_arch = "x86_64", target_os = "windows"),
    all(target_arch = "aarch64", target_os = "windows"),
)))]
compile_error!("haste currently supports x86_64/aarch64 Linux, macOS, and Windows hosts");

mod aot;
mod body;
mod build;
mod cli;
mod dump;
mod graph;
mod host;
mod image;
mod lower;
mod macros;
mod names;
mod outline;
mod prep;
mod scripts;
mod sema;
mod serve;
mod substrate;
mod sync;

use std::sync::OnceLock;

/// How deep macro expansion, autoderef, trait selection and nested item frames go before failing, rustc's default `recursion_limit`.
pub(crate) const RECURSION_LIMIT: usize = 128;

pub use cli::main;

static JOBS: OnceLock<usize> = OnceLock::new();

/// How many processes or threads haste runs at once: the count `--jobs` set, else the cores this process may run on.
fn jobs() -> usize {
    *JOBS.get_or_init(|| std::thread::available_parallelism().map_or(1, std::num::NonZero::get))
}

/// Sets the job count once, before any work reads it.
fn set_jobs(count: usize) {
    _ = JOBS.set(count);
}

static PROFILE: OnceLock<outline::tree::Profile> = OnceLock::new();

/// The build profile: the one `--release` set, else the debug profile.
pub(crate) fn profile() -> outline::tree::Profile {
    *PROFILE.get_or_init(outline::tree::Profile::default)
}

static TARGET: OnceLock<&'static haste_codegen::Target> = OnceLock::new();
static HOST_SCOPE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// The target being compiled for: the host target inside a host build-script/proc-macro scope, else the `--target` override, else the host target.
pub(crate) fn compile_target() -> &'static haste_codegen::Target {
    if HOST_SCOPE.load(std::sync::atomic::Ordering::Acquire) > 0 {
        return haste_codegen::Target::host();
    }
    *TARGET.get_or_init(haste_codegen::Target::host)
}

/// Runs `f` with `compile_target()` pinned to the host target (for build scripts and proc macros).
pub(crate) fn with_host_target<R>(f: impl FnOnce() -> R) -> R {
    struct Guard;
    impl Drop for Guard {
        fn drop(&mut self) {
            HOST_SCOPE.fetch_sub(1, std::sync::atomic::Ordering::Release);
        }
    }
    HOST_SCOPE.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
    let _guard = Guard;
    f()
}

/// Sets the compilation target once, before any work reads it.
fn set_target(target: &'static haste_codegen::Target) {
    _ = TARGET.set(target);
}

/// The compilation target with the code generator's effort set by the build profile.
pub(crate) fn host_target() -> haste_codegen::Target {
    compile_target().effort(profile().into())
}

/// Sets the build profile once, before any work reads it.
fn set_profile(profile: outline::tree::Profile) {
    _ = PROFILE.set(profile);
}

/// Package, binary, and feature selection from the command line.
#[derive(Clone, Debug, Default, Hash, PartialEq, Eq)]
pub(crate) struct Selection {
    pub package: Option<String>,
    pub bin: Option<String>,
    pub example: Option<String>,
    pub features: Vec<String>,
    pub all_features: bool,
    pub no_default_features: bool,
    pub ignored_only: bool,
    pub include_ignored: bool,
}

static SELECTION: OnceLock<Selection> = OnceLock::new();

/// The package, binary, and feature selection for this run.
pub(crate) fn selection() -> &'static Selection {
    SELECTION.get_or_init(Selection::default)
}

/// Sets the package, binary, and feature selection once, before any work reads it.
fn set_selection(selection: Selection) {
    _ = SELECTION.set(selection);
}

static TEST_MODE: OnceLock<bool> = OnceLock::new();

/// Whether this invocation is running `haste test` (enabling `#[cfg(test)]` and member dev-dependencies).
pub(crate) fn test_mode() -> bool {
    *TEST_MODE.get_or_init(|| false)
}

/// Enables test mode once, before any workspace or outline loads.
fn set_test_mode() {
    _ = TEST_MODE.set(true);
}

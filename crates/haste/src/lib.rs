//! Haste: the build graph, name resolution, type inference, lowering, and the JIT that runs the program.

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
fn profile() -> outline::tree::Profile {
    *PROFILE.get_or_init(outline::tree::Profile::default)
}

/// Sets the build profile once, before any work reads it.
fn set_profile(profile: outline::tree::Profile) {
    _ = PROFILE.set(profile);
}

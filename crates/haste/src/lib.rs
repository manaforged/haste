//! Haste: the build graph, name resolution, type inference, lowering, and the JIT that runs the program.

mod body;
mod build;
mod cli;
mod graph;
mod host;
mod image;
mod lower;
mod macros;
mod names;
mod outline;
mod scripts;
mod sema;
mod serve;

use std::sync::OnceLock;

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

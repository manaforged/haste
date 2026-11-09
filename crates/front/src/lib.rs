//! Haste's front end: the build graph, the outline index, and analysis of reached code.

pub mod body;
pub mod graph;
pub mod host;
pub mod image;
pub mod lower;
pub mod macros;
pub mod names;
pub mod outline;
pub mod scripts;
pub mod sema;

/// The number of cores this process may run on, at least one.
pub fn cores() -> usize {
    std::thread::available_parallelism().map_or(1, std::num::NonZero::get)
}

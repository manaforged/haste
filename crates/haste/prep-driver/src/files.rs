//! The files a prep link writes into a library directory and the names and formats the driver writes and Haste reads.

use std::os::unix::fs::MetadataExt as _;
use std::path::Path;

/// The library a prep link makes from every unit.
pub const LIBRARY: &str = "library.so";
/// The second library a prep link makes from the units rebuilt since `LIBRARY` was linked, which resolves against it.
pub const DELTA: &str = "delta.so";
/// The file beside `LIBRARY` that names each unit it holds with the stamp of that unit's manifest.
pub const UNITS: &str = "units.txt";
/// The variable that names the prep directory to the driver when it runs as rustc's wrapper.
pub const PREP_DIR: &str = "PREP_DIR";
/// The separator of the fields of a line of a unit list or a snapshot.
pub const FIELD: char = '\t';

/// The manifest beside library `library`.
pub fn manifest(library: &str) -> String {
    format!("{library}.toml")
}

/// The stamp of the file at `path`, which every write or replacement of it changes: its inode, modification time and size; none when it cannot be read.
pub fn stamp(path: &Path) -> Option<String> {
    let meta = std::fs::metadata(path).ok()?;
    Some(format!("{}.{}.{}.{}", meta.ino(), meta.mtime(), meta.mtime_nsec(), meta.size()))
}

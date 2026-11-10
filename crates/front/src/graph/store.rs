//! The store that keeps build script outputs between runs, so a cold run reruns only the scripts whose inputs or watched files changed, as Cargo does.

use std::fs::File;
use std::hash::BuildHasher as _;
use std::io;
use std::os::fd::AsRawFd as _;
use std::os::unix::fs::MetadataExt as _;
use std::path::{Path, PathBuf};

use super::script::Output;
use super::{Side, Workspace};

const STAMP: &str = "stamp";
const OUTPUT: &str = "output";
const FAILED: &str = "failed";

/// The directory of the store: `$HASTE_HOME`, else `$XDG_CACHE_HOME/haste`, else `$HOME/.cache/haste`.
fn home() -> PathBuf {
    let var = |name: &str| std::env::var_os(name).filter(|value| !value.is_empty()).map(PathBuf::from);
    let home = var("HASTE_HOME").or_else(|| var("XDG_CACHE_HOME").map(|cache| cache.join("haste"))).or_else(|| var("HOME").map(|home| home.join(".cache/haste")));
    home.unwrap_or_else(|| std::env::temp_dir().join("haste"))
}

/// The directory of the build script outputs in the store.
pub fn root() -> PathBuf {
    home().join("scripts")
}

/// The key of a call of procedural macro `name` of the pinned crate `pin` on `texts`, by this program.
pub fn expansion_key(pin: u64, name: &str, texts: &[String]) -> u64 {
    rustc_hash::FxBuildHasher.hash_one((pin, name, texts, program()))
}

/// What the macro call with key `key` printed, when a run kept it.
pub fn kept_expansion(key: u64) -> Option<String> {
    std::fs::read_to_string(expansion(key)).ok()
}

/// Keeps what the macro call with key `key` printed.
pub fn keep_expansion(key: u64, text: &str) -> io::Result<()> {
    let path = expansion(key);
    std::fs::create_dir_all(path.parent().unwrap_or(&path))?;
    write_new(&path, text)
}

fn expansion(key: u64) -> PathBuf {
    home().join("macros").join(format!("{key:016x}"))
}

/// The size and modification time of this program, whose changes make what the store kept stale.
fn program() -> Option<(u64, i64, i64)> {
    std::env::current_exe().and_then(std::fs::metadata).map(|meta| (meta.len(), meta.mtime(), meta.mtime_nsec())).ok()
}

/// The directory that holds the build script output of `unit`: the script's `OUT_DIR`, its standard error, what it printed and the stamp that says when it is current.
pub fn dir(workspace: &Workspace, unit: usize) -> PathBuf {
    let entry = &workspace.graph.units[unit];
    let package = &workspace.packages[entry.package];
    let side = if entry.side == Side::Host { "host" } else { "target" };
    let features: Vec<&str> = entry.features.iter().map(String::as_str).collect();
    let key = rustc_hash::FxBuildHasher.hash_one((package.manifest.dir.as_path(), side, features.join(",")));
    root().join(format!("{}-{}-{key:016x}", package.locked.name, package.locked.version))
}

/// The kept result of `unit`'s script, its output or why it did not build or run, when it ran with `vars` under this program and no file or variable it watches changed since.
pub fn kept(workspace: &Workspace, unit: usize, vars: &[(String, String)]) -> Option<Result<Output, String>> {
    let dir = dir(workspace, unit);
    let stamp = std::fs::read_to_string(dir.join(STAMP)).ok()?;
    let mut lines = stamp.lines();
    let (inputs, watch) = (lines.next()?, lines.next()?);
    let watched = Watched::read(lines);
    (inputs == inputs_hash(vars) && watch == watched.hash()).then_some(())?;
    if let Ok(error) = std::fs::read_to_string(dir.join(FAILED)) {
        return Some(Err(error));
    }
    std::fs::read_to_string(dir.join(OUTPUT)).ok().map(|text| Ok(Output::parse(&text)))
}

/// Keeps what `unit`'s script printed when it ran with `vars`, with the files and variables it watches as they are now.
pub fn keep(workspace: &Workspace, unit: usize, vars: &[(String, String)], text: &str) -> io::Result<()> {
    let dir = dir(workspace, unit);
    remove(&dir.join(FAILED))?;
    write_new(&dir.join(OUTPUT), text)?;
    stamp(&dir, vars, &Watched::of(workspace, unit, &Output::parse(text)))
}

/// Keeps why `unit`'s script did not build or run with `vars`, so later runs report it again until this program, the inputs or the package's files change.
pub fn keep_failure(workspace: &Workspace, unit: usize, vars: &[(String, String)], error: &str) -> io::Result<()> {
    let dir = dir(workspace, unit);
    std::fs::create_dir_all(&dir)?;
    write_new(&dir.join(FAILED), error)?;
    stamp(&dir, vars, &Watched::of(workspace, unit, &Output::default()))
}

fn stamp(dir: &Path, vars: &[(String, String)], watched: &Watched) -> io::Result<()> {
    let mut stamp = format!("{}\n{}\n", inputs_hash(vars), watched.hash());
    stamp.extend(watched.files.iter().map(|path| format!("f {}\n", path.display())));
    stamp.extend(watched.vars.iter().map(|name| format!("e {name}\n")));
    write_new(&dir.join(STAMP), &stamp)
}

fn remove(path: &Path) -> io::Result<()> {
    match std::fs::remove_file(path) {
        Err(error) if error.kind() != io::ErrorKind::NotFound => Err(error),
        _ => Ok(()),
    }
}

/// Drops the stamp of `unit` before its script runs again, so an interrupted run leaves no output that looks current.
pub fn forget(workspace: &Workspace, unit: usize) -> io::Result<()> {
    remove(&dir(workspace, unit).join(STAMP))
}

/// An exclusive hold on the entry of one unit, so two runs never run the same script into one `OUT_DIR` at once; dropping it lets go.
pub struct Lock {
    _file: File,
}

/// Waits until this process holds the entry of `unit`.
pub fn lock(workspace: &Workspace, unit: usize) -> io::Result<Lock> {
    let dir = dir(workspace, unit);
    std::fs::create_dir_all(&dir)?;
    let file = File::options().create(true).truncate(false).write(true).open(dir.join("lock"))?;
    // SAFETY: the descriptor belongs to `file`, which stays open for the call; flock only reads it.
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(Lock { _file: file })
}

fn write_new(path: &Path, text: &str) -> io::Result<()> {
    let fresh = path.with_extension(format!("{}", std::process::id()));
    std::fs::write(&fresh, text)?;
    std::fs::rename(&fresh, path)
}

/// The hash of the variables a script ran with and of this program, whose compile of the script decides what it prints.
fn inputs_hash(vars: &[(String, String)]) -> String {
    let mut sorted: Vec<&(String, String)> = vars.iter().collect();
    sorted.sort();
    format!("{:016x}", rustc_hash::FxBuildHasher.hash_one((sorted, program())))
}

/// The files and environment variables whose change makes a script run again.
struct Watched {
    files: Vec<PathBuf>,
    vars: Vec<String>,
}

impl Watched {
    /// What Cargo watches for `unit`: the `rerun-if` lines the script printed, else every file of a package that is not an unpacked registry or git source; a local script's own source always.
    fn of(workspace: &Workspace, unit: usize, output: &Output) -> Watched {
        let package = &workspace.packages[workspace.graph.units[unit].package];
        let manifest = &package.manifest;
        let local = package.locked.source.is_none();
        let mut files: Vec<PathBuf> = output.watched.iter().map(|path| manifest.dir.join(path)).collect();
        if output.watched.is_empty() && output.watched_vars.is_empty() && local {
            files.push(manifest.dir.clone());
        }
        if local {
            files.extend(manifest.build.iter().cloned());
        }
        Watched { files, vars: output.watched_vars.clone() }
    }

    fn read<'t>(lines: impl Iterator<Item = &'t str>) -> Watched {
        let mut watched = Watched { files: Vec::new(), vars: Vec::new() };
        for line in lines {
            match line.split_at_checked(2) {
                Some(("f ", path)) => watched.files.push(PathBuf::from(path)),
                Some(("e ", name)) => watched.vars.push(name.to_owned()),
                _ => {}
            }
        }
        watched
    }

    /// The hash of the size and modification time of every watched file, directories walked, and of every watched variable's value.
    fn hash(&self) -> String {
        let mut stamps = Vec::new();
        for path in &self.files {
            walk(path, &mut stamps);
        }
        let vars: Vec<Option<std::ffi::OsString>> = self.vars.iter().map(std::env::var_os).collect();
        format!("{:016x}", rustc_hash::FxBuildHasher.hash_one((stamps, vars)))
    }
}

/// Adds the path, size and modification time of `path`, or of every file under it when it is a directory other than a build output or VCS directory.
fn walk(path: &Path, into: &mut Vec<(PathBuf, u64, i64, i64)>) {
    let Ok(meta) = std::fs::metadata(path) else {
        into.push((path.to_owned(), 0, 0, -1));
        return;
    };
    if !meta.is_dir() {
        into.push((path.to_owned(), meta.len(), meta.mtime(), meta.mtime_nsec()));
        return;
    }
    let Ok(entries) = std::fs::read_dir(path) else { return };
    let mut children: Vec<PathBuf> = entries.filter_map(|entry| entry.ok().map(|entry| entry.path())).collect();
    children.sort();
    for child in children.iter().filter(|child| !matches!(child.file_name().and_then(|name| name.to_str()), Some("target" | ".git"))) {
        walk(child, into);
    }
}

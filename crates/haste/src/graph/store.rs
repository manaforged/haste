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
const OK_MARK: &str = "+";
const FAILED_MARK: &str = "-";

/// The directory of the store: `$HASTE_HOME`, else `$XDG_CACHE_HOME/haste`, else `$HOME/.cache/haste`.
pub(crate) fn home() -> PathBuf {
    let var = |name: &str| std::env::var_os(name).filter(|value| !value.is_empty()).map(PathBuf::from);
    let home = var("HASTE_HOME").or_else(|| var("XDG_CACHE_HOME").map(|cache| cache.join("haste"))).or_else(|| var("HOME").map(|home| home.join(".cache/haste")));
    home.unwrap_or_else(|| std::env::temp_dir().join("haste"))
}

/// The directory of the build script outputs in the store.
pub fn root() -> PathBuf {
    home().join("scripts")
}

/// A call of a procedural macro of a pinned crate as the store keeps it: the file its inputs hash to, and the inputs, which a kept result must match exactly.
pub struct Expansion {
    path: PathBuf,
    inputs: String,
}

impl Expansion {
    /// The call of procedural macro `name` of the pinned crate `pin` on `texts`, by this program.
    pub fn new(pin: u64, name: &str, texts: &[String]) -> Expansion {
        let program = program();
        let key = rustc_hash::FxBuildHasher.hash_one((pin, name, texts, program));
        let inputs = format!("{pin:016x} {program:?} {name}\0{}", texts.join("\0"));
        Expansion { path: home().join("macros").join(format!("{key:016x}")), inputs }
    }

    /// What this call printed, or why it failed, when a run kept it for the same inputs.
    pub fn kept(&self) -> Option<Result<String, String>> {
        let text = std::fs::read_to_string(&self.path).ok()?;
        let (mark, rest) = text.split_at_checked(1)?;
        let (len, rest) = rest.split_once('\n')?;
        let (inputs, output) = rest.split_at_checked(len.parse().ok()?)?;
        if inputs != self.inputs {
            return None;
        }
        match mark {
            OK_MARK => Some(Ok(output.to_owned())),
            FAILED_MARK => Some(Err(output.to_owned())),
            _ => None,
        }
    }

    /// Keeps what this call printed, or why it failed, since this program fails the same call the same way.
    pub fn keep(&self, result: &Result<String, String>) -> io::Result<()> {
        std::fs::create_dir_all(self.path.parent().unwrap_or(&self.path))?;
        let (mark, text) = match result {
            Ok(output) => (OK_MARK, output),
            Err(error) => (FAILED_MARK, error),
        };
        write_new(&self.path, &format!("{mark}{}\n{}{text}", self.inputs.len(), self.inputs))
    }
}

/// The byte size and nanosecond modification timestamp of a file, or none when unreadable.
pub(crate) type FileStamp = Option<(u64, i128)>;

/// Extracts the byte size and nanosecond modification timestamp from file metadata `meta`.
pub(crate) fn meta_stamp(meta: &std::fs::Metadata) -> (u64, i128) {
    let nanos = i128::from(meta.mtime()) * 1_000_000_000 + i128::from(meta.mtime_nsec());
    (meta.len(), nanos)
}

/// The byte size and modification timestamp of `path`, or none when unreadable.
pub(crate) fn file_stamp(path: &Path) -> FileStamp {
    let meta = std::fs::metadata(path).ok()?;
    Some(meta_stamp(&meta))
}

/// The size and modification time of this program, whose changes make what the store kept stale.
pub(crate) fn program() -> FileStamp {
    static PROGRAM: std::sync::OnceLock<FileStamp> = std::sync::OnceLock::new();
    *PROGRAM.get_or_init(|| std::env::current_exe().ok().as_deref().and_then(file_stamp))
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

pub(crate) fn write_new(path: &Path, text: &str) -> io::Result<()> {
    let fresh = fresh(path);
    std::fs::write(&fresh, text)?;
    std::fs::rename(&fresh, path)
}

/// The path this process writes `path` at before it renames the file into place, so no reader sees it half written.
pub(crate) fn fresh(path: &Path) -> PathBuf {
    path.with_extension(format!("{}", std::process::id()))
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
    /// What Cargo watches for `unit`: the `rerun-if` lines the script printed, else every file of a package that is not an unpacked registry or git source; a local script's own sources always.
    fn of(workspace: &Workspace, unit: usize, output: &Output) -> Watched {
        let package = &workspace.packages[workspace.graph.units[unit].package];
        let manifest = &package.manifest;
        let local = package.locked.source.is_none();
        let mut files: Vec<PathBuf> = output.watched.iter().map(|path| manifest.dir.join(path)).collect();
        if output.watched.is_empty() && output.watched_vars.is_empty() && local {
            files.push(manifest.dir.clone());
        }
        if local && let Some(build) = &manifest.build {
            build_script_inputs(&manifest.dir, build, &mut files);
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

fn build_script_inputs(manifest_dir: &Path, build_path: &Path, into: &mut Vec<PathBuf>) {
    into.push(build_path.to_owned());
    let mod_dir = build_path.with_extension("");
    if mod_dir.is_dir() {
        into.push(mod_dir);
    }
    if let Some(parent) = build_path.parent()
        && parent != manifest_dir
    {
        into.push(parent.to_owned());
    }
    let Ok(entries) = std::fs::read_dir(manifest_dir) else {
        return;
    };
    let mut extras: Vec<PathBuf> = entries
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("rs") && path.is_file())
        .collect();
    extras.sort();
    into.extend(extras);
}

/// Stamps of all build-script sources, `cargo:rerun-if-changed` paths, and `cargo:rerun-if-env-changed` values for local packages in `workspace`.
pub(crate) fn watched_stamps(workspace: &Workspace) -> (Vec<(PathBuf, u64, i128)>, Vec<(String, Option<std::ffi::OsString>)>) {
    let mut files = Vec::new();
    let mut vars = Vec::new();
    for (unit, entry) in workspace.graph.units.iter().enumerate() {
        let package = &workspace.packages[entry.package];
        let Some(build) = &package.manifest.build else { continue };
        if !package.local() {
            continue;
        }
        build_script_inputs(&package.manifest.dir, build, &mut files);
        let watched = std::fs::read_to_string(dir(workspace, unit).join(STAMP))
            .ok()
            .map(|text| Watched::read(text.lines().skip(2)))
            .unwrap_or_else(|| Watched::of(workspace, unit, &Output::default()));
        files.extend(watched.files);
        vars.extend(watched.vars);
    }
    files.sort();
    files.dedup();
    vars.sort();
    vars.dedup();
    let mut stamps = Vec::new();
    for path in &files {
        walk(path, &mut stamps);
    }
    let env_vals = vars.into_iter().map(|name| (name.clone(), std::env::var_os(&name))).collect();
    (stamps, env_vals)
}

/// Adds the path, size and modification time of `path`, or of every file under it when it is a directory other than a build output or VCS directory.
pub(crate) fn walk(path: &Path, into: &mut Vec<(PathBuf, u64, i128)>) {
    let Ok(meta) = std::fs::metadata(path) else {
        into.push((path.to_owned(), 0, -1));
        return;
    };
    if !meta.is_dir() {
        let (len, nanos) = meta_stamp(&meta);
        into.push((path.to_owned(), len, nanos));
        return;
    }
    let Ok(entries) = std::fs::read_dir(path) else {
        return;
    };
    let mut children: Vec<PathBuf> = entries.filter_map(|entry| entry.ok().map(|entry| entry.path())).collect();
    children.sort();
    for child in children.iter().filter(|child| !matches!(child.file_name().and_then(|name| name.to_str()), Some("target" | ".git"))) {
        walk(child, into);
    }
}

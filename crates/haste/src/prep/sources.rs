use std::collections::BTreeMap;
use std::hash::BuildHasher as _;
use std::path::{Path, PathBuf};

use anyhow::Result;
use rustc_hash::FxBuildHasher;

use crate::graph::Workspace;
use crate::outline::Outline;
use crate::outline::source::Source;

/// The word that starts a line of a snapshot that names a source file.
pub(super) const FILE: &str = "file";

use super::files::FIELD;

const NONE: &str = "-";

/// The local source files of a build as a prep read them before rustc built them: by path, the hash of each one's bytes, none when it could not be read, and its stamp just before the read.
#[derive(PartialEq, Eq, Debug, Default, Clone)]
pub(super) struct Sources {
    files: BTreeMap<PathBuf, Read>,
}

/// One file as a prep read it: the hash of its bytes, and its stamp just before.
#[derive(PartialEq, Eq, Debug, Clone)]
struct Read {
    hash: Option<u64>,
    stamp: String,
}

impl Sources {
    /// The files at `paths` as they read now, each stamped first, so a write after the stamp shows as a newer stamp.
    pub(super) fn read(paths: impl IntoIterator<Item = PathBuf>) -> Sources {
        let read = |path: PathBuf| {
            let stamp = stamp(&path);
            let hash = std::fs::read(&path).ok().map(|bytes| FxBuildHasher.hash_one(&bytes));
            (path, Read { hash, stamp })
        };
        Sources { files: paths.into_iter().map(read).collect() }
    }

    /// The local source files of `outline` that rustc reads: each local crate's own files and the files their `include_str!` and `include_bytes!` calls read.
    pub(super) fn of(outline: &Outline, workspace: &Workspace) -> Vec<PathBuf> {
        let mut paths = Vec::new();
        for krate in (0..outline.crates.len()).filter(|&krate| outline.package(workspace, krate).local()) {
            paths.extend(outline.crates[krate].files.iter().filter(|file| !file.expanded).map(|file| file.source.path.clone()));
            paths.extend(outline.included(krate).into_iter().map(|(path, _)| path));
        }
        paths.sort();
        paths.dedup();
        paths
    }

    /// The paths of the files read.
    pub(super) fn paths(&self) -> Vec<PathBuf> {
        self.files.keys().cloned().collect()
    }

    /// Whether `now` read the same files with the same bytes as these.
    pub(super) fn same(&self, now: &Sources) -> bool {
        let equal = |(path, read): (&PathBuf, &Read)| read.hash.is_some() && now.files.get(path).is_some_and(|other| other.hash == read.hash);
        !self.files.is_empty() && self.files.len() == now.files.len() && self.files.iter().all(equal)
    }

    /// Whether every file still has the stamp it had when read, which no write since leaves.
    pub(super) fn unchanged(&self) -> bool {
        !self.files.is_empty() && self.files.iter().all(|(path, read)| stamp(path) == read.stamp)
    }

    /// Whether rustc read the sources `outline` holds: these name every local file of `outline` and every file it includes, each reads now with the bytes it had then, and each local file's text parses to the text `outline` holds.
    pub(super) fn settled(&self, outline: &Outline, workspace: &Workspace) -> Result<bool> {
        for krate in (0..outline.crates.len()).filter(|&krate| outline.package(workspace, krate).local()) {
            for file in outline.crates[krate].files.iter().filter(|file| !file.expanded) {
                let Some(text) = self.again(&file.source.path) else { return Ok(false) };
                if Source::parse(file.source.path.clone(), text)?.text != file.source.text {
                    return Ok(false);
                }
            }
            if !outline.included(krate).into_iter().all(|(path, _)| self.again(&path).is_some()) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// The text of the file at `path` now, when these read it and it holds the same bytes.
    fn again(&self, path: &Path) -> Option<String> {
        let then = self.files.get(path)?.hash?;
        let bytes = std::fs::read(path).ok()?;
        (FxBuildHasher.hash_one(&bytes) == then).then(|| String::from_utf8_lossy(&bytes).into_owned())
    }

    /// The snapshot lines that name these files.
    pub(super) fn lines(&self) -> String {
        let mut text = String::new();
        for (path, Read { hash, stamp }) in &self.files {
            let hash = hash.map_or_else(|| NONE.to_owned(), |hash| format!("{hash:016x}"));
            text.push_str(&format!("{FILE}{FIELD}{hash}{FIELD}{stamp}{FIELD}{}\n", path.display()));
        }
        text
    }

    /// Adds the file a snapshot line names, given the fields after its first.
    pub(super) fn line(&mut self, fields: &[&str]) -> Option<()> {
        let [hash, stamp, path] = fields else { return None };
        let hash = if *hash == NONE { None } else { Some(u64::from_str_radix(hash, 16).ok()?) };
        self.files.insert(PathBuf::from(path), Read { hash, stamp: (*stamp).to_owned() });
        Some(())
    }
}

/// The stamp of the file at `path` as text, which every write of it changes.
fn stamp(path: &Path) -> String {
    super::files::stamp(path).unwrap_or_else(|| NONE.to_owned())
}

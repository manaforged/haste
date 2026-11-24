use std::path::PathBuf;

use anyhow::Result;
use rustc_hash::FxHashMap as HashMap;

use crate::graph;
use crate::outline::{self, edit::Edit};

pub(super) use crate::graph::store::{FileStamp as Stamp, file_stamp as stamp};

const SOURCE_DIRS: [&str; 4] = ["src", "tests", "examples", "benches"];

/// A file that a kept build watches: its path, its last time and size, and for a file that a macro includes, the file that holds the call.
pub(super) struct Watch {
    pub(super) path: PathBuf,
    pub(super) stamp: Stamp,
    pub(super) by: Option<PathBuf>,
}

/// The source files of every crate whose package is local, which an edit can change, the files `include!` read into them, and the files their `include_str!` and `include_bytes!` calls read.
pub(super) fn stamps(workspace: &graph::Workspace, outline: &outline::Outline) -> Vec<Watch> {
    let local: Vec<usize> = (0..outline.roots.len())
        .filter(|&krate| outline.package(workspace, krate).local())
        .collect();
    let mut sources: Vec<PathBuf> = local
        .iter()
        .flat_map(|&krate| {
            outline.crates[krate]
                .files
                .iter()
                .map(|file| file.source.path.clone())
        })
        .collect();
    sources.sort();
    sources.dedup();
    let included = local
        .iter()
        .flat_map(|&krate| outline.included(krate))
        .map(|(path, by)| (path, Some(by)));
    sources
        .into_iter()
        .map(|path| (path, None))
        .chain(included)
        .map(|(path, by)| Watch {
            stamp: stamp(&path),
            path,
            by,
        })
        .collect()
}

/// The path of each file of `stamps`.
pub(super) fn paths(stamps: &[Watch]) -> Vec<PathBuf> {
    stamps.iter().map(|watch| watch.path.clone()).collect()
}

pub(super) fn pre_stamps(workspace: &graph::Workspace) -> HashMap<PathBuf, Stamp> {
    let mut out = HashMap::default();
    let mut stack = Vec::new();
    for pkg in workspace.packages.iter().filter(|pkg| pkg.local()) {
        for sub in SOURCE_DIRS {
            let dir = pkg.manifest.dir.join(sub);
            if dir.is_dir() {
                stack.push(dir);
            }
        }
        let roots = pkg
            .manifest
            .lib
            .iter()
            .chain(pkg.manifest.bins.iter().map(|(_, p)| p))
            .chain(pkg.manifest.examples.iter().map(|(_, p)| p))
            .chain(pkg.manifest.tests.iter().map(|(_, p)| p));
        for path in roots {
            if path.is_file() {
                out.insert(path.clone(), stamp(path));
            }
        }
    }
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.insert(path.clone(), stamp(&path));
            }
        }
    }
    out
}

/// What changed in the files of `stamps` since their last stamps: the `(crate, file)` slots whose function bodies changed, which `outline` now holds, and the crates to read again; fails when a file does not read.
pub(super) fn edits(
    workspace: &graph::Workspace,
    stamps: &mut [Watch],
    outline: &mut outline::Outline,
) -> Result<(Vec<(u32, u32)>, Vec<u32>)> {
    let (mut stale, mut crates) = (Vec::new(), Vec::new());
    for watch in stamps {
        let now = stamp(&watch.path);
        if now == watch.stamp {
            continue;
        }
        watch.stamp = now;
        let edit = match &watch.by {
            Some(by) => outline.reexpand(by)?,
            None => outline.reread(&watch.path)?,
        };
        match edit {
            Edit::Same => {}
            Edit::Bodies(slots)
                if slots
                    .iter()
                    .any(|&(krate, _)| expands(workspace, outline, krate)) =>
            {
                crates.extend(slots.iter().map(|&(krate, _)| krate))
            }
            Edit::Bodies(slots) => stale.extend(slots),
            Edit::Items(edited) => crates.extend(edited),
        }
    }
    Ok((stale, crates))
}

/// The `(crate, file)` slots whose function bodies changed in the files of `stamps`, which `outline` now holds, or why the edit needs a restart.
pub(super) fn body_edits(
    workspace: &graph::Workspace,
    stamps: &mut [Watch],
    outline: &mut outline::Outline,
) -> Result<Vec<(u32, u32)>, String> {
    match edits(workspace, stamps, outline) {
        Ok((stale, crates)) if crates.is_empty() => Ok(stale),
        Ok((_, mut crates)) => {
            crates.sort_unstable();
            crates.dedup();
            let names: Vec<&str> = crates
                .iter()
                .map(|&krate| outline.roots[krate as usize].name.as_str())
                .collect();
            Err(format!(
                "more than function bodies changed in {}",
                names.join(", ")
            ))
        }
        Err(error) => Err(format!("{error:#}")),
    }
}

/// Whether crate `krate` is a procedural macro crate, whose bodies decide the code of the crates that call its macros.
fn expands(workspace: &graph::Workspace, outline: &outline::Outline, krate: u32) -> bool {
    outline
        .package(workspace, krate as usize)
        .manifest
        .proc_macro
}

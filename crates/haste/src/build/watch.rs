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

fn collect_included(local: &[usize], outline: &outline::Outline) -> Vec<(PathBuf, PathBuf)> {
    let mut out = Vec::new();
    let mut uncached = Vec::new();
    for &krate in local {
        if let Some(cached) = &outline.crates[krate].cached_included {
            out.extend(cached.iter().cloned());
        } else {
            uncached.push(krate);
        }
    }
    if uncached.len() > 1 && crate::jobs() > 1 {
        for chunk in crate::sync::parallel_map(uncached, |krate| outline.included(krate)) {
            out.extend(chunk);
        }
    } else {
        for krate in uncached {
            out.extend(outline.included(krate));
        }
    }
    out
}

fn build_stamps(
    workspace: &graph::Workspace,
    outline: &outline::Outline,
    reuse: Option<&HashMap<PathBuf, Stamp>>,
    extra_sources: Vec<PathBuf>,
) -> Vec<Watch> {
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
        .chain(extra_sources)
        .collect();
    sources.sort();
    sources.dedup();
    let included = collect_included(&local, outline);
    let mut pending = Vec::with_capacity(sources.len() + included.len());
    let mut missing = Vec::new();
    for (path, by) in sources
        .into_iter()
        .map(|path| (path, None))
        .chain(included.into_iter().map(|(path, by)| (path, Some(by))))
    {
        if let Some(&cached) = reuse.and_then(|map| map.get(&path)) {
            pending.push(Some(Watch {
                path,
                stamp: cached,
                by,
            }));
        } else {
            let slot = pending.len();
            pending.push(None);
            missing.push((slot, path, by));
        }
    }
    if missing.len() >= 64 && crate::jobs() > 1 {
        let computed = crate::sync::parallel_map(missing, |(slot, path, by)| {
            let s = stamp(&path);
            (slot, Watch { path, stamp: s, by })
        });
        for (slot, watch) in computed {
            pending[slot] = Some(watch);
        }
    } else {
        for (slot, path, by) in missing {
            let s = stamp(&path);
            pending[slot] = Some(Watch { path, stamp: s, by });
        }
    }
    pending.into_iter().flatten().collect()
}

/// The source files of every crate whose package is local, which an edit can change, the files `include!` read into them, and the files their `include_str!` and `include_bytes!` calls read.
pub(super) fn stamps(workspace: &graph::Workspace, outline: &outline::Outline) -> Vec<Watch> {
    build_stamps(workspace, outline, None, local_source_paths(workspace))
}

/// Like [`stamps`], reusing paths and stamps already gathered by [`pre_stamps`].
pub(super) fn stamps_with_pre(
    workspace: &graph::Workspace,
    outline: &outline::Outline,
    pre: &HashMap<PathBuf, Stamp>,
) -> Vec<Watch> {
    build_stamps(workspace, outline, Some(pre), pre.keys().cloned().collect())
}

/// Like [`stamps`], reusing existing stamps from `seen` for unchanged paths across reloads.
pub(super) fn stamps_with_seen(
    workspace: &graph::Workspace,
    outline: &outline::Outline,
    seen: &HashMap<PathBuf, Stamp>,
) -> Vec<Watch> {
    build_stamps(
        workspace,
        outline,
        Some(seen),
        local_source_paths(workspace),
    )
}

/// The path of each file of `stamps`.
pub(super) fn paths(stamps: &[Watch]) -> Vec<PathBuf> {
    stamps.iter().map(|watch| watch.path.clone()).collect()
}

fn local_source_paths(workspace: &graph::Workspace) -> Vec<PathBuf> {
    let mut out = Vec::new();
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
            .chain(pkg.manifest.bins.iter().map(|t| &t.path))
            .chain(pkg.manifest.examples.iter().map(|t| &t.path))
            .chain(pkg.manifest.tests.iter().map(|t| &t.path))
            .chain(pkg.manifest.benches.iter().map(|t| &t.path));
        for path in roots {
            if path.is_file() {
                out.push(path.clone());
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
                out.push(path);
            }
        }
    }
    out
}

pub(super) fn pre_stamps(workspace: &graph::Workspace) -> HashMap<PathBuf, Stamp> {
    let paths = local_source_paths(workspace);
    if paths.len() < 64 || crate::jobs() <= 1 {
        return paths
            .into_iter()
            .map(|path| {
                let s = stamp(&path);
                (path, s)
            })
            .collect();
    }
    crate::sync::parallel_map(paths, |path| {
        let s = stamp(&path);
        (path, s)
    })
    .into_iter()
    .collect()
}

fn collect_edits(
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
        if now.is_none() {
            anyhow::bail!("watched file removed: {}", watch.path.display());
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

/// Returns body-only `(crate, file)` edits from `stamps`, failing on missing files or item/macro edits so callers trigger a full reload without growing `reached`.
pub(super) fn edits(
    workspace: &graph::Workspace,
    stamps: &mut [Watch],
    outline: &mut outline::Outline,
) -> Result<(Vec<(u32, u32)>, Vec<u32>)> {
    let (stale, crates) = collect_edits(workspace, stamps, outline)?;
    if !crates.is_empty() {
        anyhow::bail!("non-body edits require full build reload");
    }
    Ok((stale, crates))
}

/// The `(crate, file)` slots whose function bodies changed in the files of `stamps`, which `outline` now holds, or why the edit needs a restart.
pub(super) fn body_edits(
    workspace: &graph::Workspace,
    stamps: &mut [Watch],
    outline: &mut outline::Outline,
) -> Result<Vec<(u32, u32)>, String> {
    match collect_edits(workspace, stamps, outline) {
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

/// Whether crate `krate` is a procedural macro crate or a dependency of one, whose bodies decide the code of the crates that call those macros.
fn expands(workspace: &graph::Workspace, outline: &outline::Outline, krate: u32) -> bool {
    outline
        .dependents(krate as usize)
        .into_iter()
        .any(|dep| outline.package(workspace, dep).manifest.proc_macro)
}

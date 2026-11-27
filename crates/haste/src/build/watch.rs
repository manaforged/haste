use std::path::{Path, PathBuf};

use anyhow::Result;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

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

fn collect_sources_and_included(
    workspace: &graph::Workspace,
    outline: &outline::Outline,
    extra_sources: Vec<PathBuf>,
) -> (Vec<PathBuf>, Vec<(PathBuf, PathBuf)>) {
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
    (sources, included)
}

fn build_stamps(
    workspace: &graph::Workspace,
    outline: &outline::Outline,
    reuse: Option<&HashMap<PathBuf, Stamp>>,
    extra_sources: Vec<PathBuf>,
) -> Vec<Watch> {
    let (sources, included) = collect_sources_and_included(workspace, outline, extra_sources);
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

/// Local source and included file paths without stating files on disk.
pub(super) fn watch_paths(
    workspace: &graph::Workspace,
    outline: &outline::Outline,
) -> Vec<PathBuf> {
    let local: Vec<usize> = (0..outline.roots.len())
        .filter(|&krate| outline.package(workspace, krate).local())
        .collect();
    let mut sources = local_source_paths(workspace);
    for &krate in &local {
        for file in &outline.crates[krate].files {
            if !file.source.path.starts_with("/haste/") {
                sources.push(file.source.path.clone());
            }
        }
    }
    sources.sort();
    sources.dedup();
    let included = collect_included(&local, outline);
    sources.reserve(included.len());
    sources.extend(included.into_iter().map(|(path, _)| path));
    sources
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
            let Ok(ft) = entry.file_type() else {
                continue;
            };
            let name = entry.file_name();
            if ft.is_dir() || (ft.is_symlink() && dir.join(&name).is_dir()) {
                stack.push(dir.join(name));
            } else if std::path::Path::new(&name)
                .extension()
                .is_some_and(|e| e == "rs")
            {
                out.push(dir.join(name));
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

pub(super) fn refresh_edits(
    workspace: &graph::Workspace,
    stamps: &mut [Watch],
    outline: &mut outline::Outline,
) -> Result<(Vec<(u32, u32)>, Vec<u32>, Vec<PathBuf>)> {
    let changed: Vec<(usize, Stamp)> = if stamps.len() >= 64 && crate::jobs() > 1 {
        crate::sync::parallel_map((0..stamps.len()).collect(), |idx| {
            let now = stamp(&stamps[idx].path);
            (now != stamps[idx].stamp).then_some((idx, now))
        })
        .into_iter()
        .flatten()
        .collect()
    } else {
        stamps
            .iter()
            .enumerate()
            .filter_map(|(idx, watch)| {
                let now = stamp(&watch.path);
                (now != watch.stamp).then_some((idx, now))
            })
            .collect()
    };
    let (mut stale, mut crates, mut paths) = (Vec::new(), Vec::new(), Vec::new());
    for (idx, now) in changed {
        if now.is_none() {
            anyhow::bail!("watched file removed: {}", stamps[idx].path.display());
        }
        stamps[idx].stamp = now;
        let edit = match &stamps[idx].by {
            Some(by) => outline.reexpand(by)?,
            None => outline.reread(&stamps[idx].path)?,
        };
        match edit {
            Edit::Same => {}
            Edit::Bodies(slots) => {
                if slots
                    .iter()
                    .any(|&(krate, _)| expands(workspace, outline, krate))
                {
                    crates.extend(slots.iter().map(|&(krate, _)| krate));
                }
                stale.extend(slots);
                paths.push(stamps[idx].path.clone());
            }
            Edit::Items(edited) => {
                crates.extend(edited);
                paths.push(stamps[idx].path.clone());
                if let Some(by) = &stamps[idx].by {
                    paths.push(by.clone());
                }
            }
        }
    }
    Ok((stale, crates, paths))
}

pub(super) fn update_stamps(
    workspace: &graph::Workspace,
    stamps: &mut Vec<Watch>,
    outline: &outline::Outline,
    crates: &[u32],
) {
    if crates.len() * 4 >= outline.crates.len() {
        let seen: HashMap<PathBuf, Stamp> = stamps
            .iter()
            .map(|watch| (watch.path.clone(), watch.stamp))
            .collect();
        *stamps = stamps_with_seen(workspace, outline, &seen);
        return;
    }
    let reloaded_files: HashSet<&Path> = crates
        .iter()
        .flat_map(|&krate| {
            outline.crates[krate as usize]
                .files
                .iter()
                .map(|file| file.source.path.as_path())
        })
        .collect();
    stamps.retain(|watch| {
        !watch
            .by
            .as_ref()
            .is_some_and(|by| reloaded_files.contains(by.as_path()))
    });
    let existing: HashSet<&Path> = stamps
        .iter()
        .filter(|watch| watch.by.is_none())
        .map(|watch| watch.path.as_path())
        .collect();
    let mut new_sources = Vec::new();
    for &krate in crates {
        for file in &outline.crates[krate as usize].files {
            let path = &file.source.path;
            if !path.starts_with("/haste/") && !existing.contains(path.as_path()) {
                new_sources.push(path.clone());
            }
        }
    }
    drop(existing);
    drop(reloaded_files);
    new_sources.sort();
    new_sources.dedup();
    let crate_indices: Vec<usize> = crates.iter().map(|&k| k as usize).collect();
    let included = collect_included(&crate_indices, outline);
    stamps.reserve(new_sources.len() + included.len());
    for path in new_sources {
        let s = stamp(&path);
        stamps.push(Watch {
            path,
            stamp: s,
            by: None,
        });
    }
    for (path, by) in included {
        let s = stamp(&path);
        stamps.push(Watch {
            path,
            stamp: s,
            by: Some(by),
        });
    }
}

pub(super) fn sync_included_stamps(
    workspace: &graph::Workspace,
    stamps: &mut Vec<Watch>,
    outline: &outline::Outline,
) {
    let local: Vec<usize> = (0..outline.roots.len())
        .filter(|&krate| outline.package(workspace, krate).local())
        .collect();
    let mut existing: HashMap<(PathBuf, PathBuf), Stamp> = HashMap::default();
    stamps.retain(|watch| {
        if let Some(by) = &watch.by {
            existing.insert((watch.path.clone(), by.clone()), watch.stamp);
            false
        } else {
            true
        }
    });
    for (path, by) in collect_included(&local, outline) {
        let key = (path, by);
        let s = existing.get(&key).copied().unwrap_or_else(|| stamp(&key.0));
        stamps.push(Watch {
            path: key.0,
            stamp: s,
            by: Some(key.1),
        });
    }
}

pub(super) fn run_watched(
    workspace: &graph::Workspace,
    compilation: &mut crate::macros::Compilation,
    cache: crate::sema::Cache,
    main: crate::names::ItemRef,
    args: &[String],
) -> Result<i32> {
    let mut watched = stamps(workspace, &compilation.outline);
    let files = paths(&watched);
    let mut reread = |compilation: &mut crate::macros::Compilation| {
        let stale = body_edits(workspace, &mut watched, &mut compilation.outline)
            .map_err(crate::image::ReloadError::Restart)?;
        if stale.is_empty() {
            return Ok(stale);
        }
        let syntax_errors = compilation.outline.local_errors(workspace);
        if !syntax_errors.is_empty() {
            eprintln!("{}", syntax_errors.join("\n"));
            return Err(crate::image::ReloadError::Compile(
                "syntax error in edited file".to_owned(),
            ));
        }
        let (_, errors) = super::batch::check_member_items(
            workspace,
            compilation,
            crate::sema::Cache::default(),
            Some(main.krate as usize),
        )
        .map_err(|e| crate::image::ReloadError::Compile(format!("{e:#}")))?;
        if !errors.is_empty() {
            for err in &errors {
                eprintln!("{err}");
            }
            return Err(crate::image::ReloadError::Compile(
                "compile error in edited file".to_owned(),
            ));
        }
        Ok(stale)
    };
    let mut program = crate::image::Program::new(compilation, cache, main)?;
    let status = program.run_watched(compilation, args, (files, &mut reread));
    std::mem::forget(program);
    status
}

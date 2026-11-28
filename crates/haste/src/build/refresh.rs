use std::path::{Path, PathBuf};

use anyhow::Result;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::graph;
use crate::outline::{self, edit::Edit};

use super::watch::{Stamp, Watch, stamp, stamps_with_seen};

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
            Edit::Bodies(slots)
                if slots
                    .iter()
                    .any(|&(krate, _)| expands_proc_macro(workspace, outline, krate)) =>
            {
                crates.extend(slots.iter().map(|&(krate, _)| krate));
                paths.push(stamps[idx].path.clone());
            }
            Edit::Bodies(slots) => {
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
    let included = collect_reloaded_included(outline, crates);
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

fn collect_reloaded_included(
    outline: &outline::Outline,
    crates: &[u32],
) -> Vec<(PathBuf, PathBuf)> {
    let mut out = Vec::new();
    let mut uncached = Vec::new();
    for &krate in crates {
        if let Some(cached) = &outline.crates[krate as usize].cached_included {
            out.extend(cached.iter().cloned());
        } else {
            uncached.push(krate as usize);
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

fn expands_proc_macro(
    workspace: &graph::Workspace,
    outline: &outline::Outline,
    krate: u32,
) -> bool {
    outline
        .dependents(krate as usize)
        .into_iter()
        .any(|dep| outline.package(workspace, dep).manifest.proc_macro)
}

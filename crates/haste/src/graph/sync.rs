use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::Result;

use super::lock::{self, Locked};
use super::manifest::{self, Kind};
use super::{LOCK_FILE, MANIFEST_FILE, Package, manifest_name, patch_paths, run_cargo};

pub(super) fn sync_lockfile(top: &Path, offline: bool) -> Result<()> {
    let args: &[&str] = if offline {
        &["metadata", "--format-version", "1", "--offline"]
    } else {
        &["metadata", "--format-version", "1"]
    };
    run_cargo(top, args)
}

pub(super) fn missing_root_in_lock(root_names: &[String], locked: &[Locked]) -> bool {
    root_names.iter().any(|name| {
        !locked
            .iter()
            .any(|e| !e.is_registry() && e.git().is_none() && e.name == *name)
    })
}

fn dep_lock_mismatch(
    pkg: &Package,
    dep: &manifest::Dep,
    locked_deps: &[lock::LockedDep<'_>],
    all_locked: &[Locked],
) -> bool {
    let matching: Vec<_> = locked_deps
        .iter()
        .filter(|ld| ld.name == dep.package)
        .collect();
    if matching.is_empty() {
        return true;
    }
    let cands: Vec<&Locked> = all_locked
        .iter()
        .filter(|e| {
            e.name == dep.package
                && matching
                    .iter()
                    .any(|ld| ld.version.as_ref().is_none_or(|v| *v == e.version))
        })
        .collect();
    dep.req
        .as_ref()
        .is_some_and(|req| !cands.iter().any(|c| req.matches(&c.version)))
        || (dep.path.is_some()
            && !cands.is_empty()
            && cands.iter().all(|c| c.source.is_some())
            && pkg.manifest.name != dep.package)
}

fn pkg_lock_mismatch(pkg: &Package, all_locked: &[Locked], member_set: &BTreeSet<PathBuf>) -> bool {
    if pkg
        .manifest
        .version
        .as_ref()
        .is_some_and(|v| *v != pkg.locked.version)
    {
        return true;
    }
    let is_member = member_set.contains(&pkg.manifest.dir);
    let locked_deps: Vec<_> = pkg.locked.deps().collect();
    let bad = pkg.manifest.deps.iter().any(|dep| {
        (is_member || dep.kind != Kind::Dev)
            && dep.package != pkg.manifest.name
            && dep_lock_mismatch(pkg, dep, &locked_deps, all_locked)
    });
    bad || (is_member
        && locked_deps
            .iter()
            .any(|ld| !pkg.manifest.deps.iter().any(|dep| dep.package == ld.name)))
}

pub(super) fn lockfile_dirty(
    top: &Path,
    all_locked: &[Locked],
    packages: &[Package],
    missing: &[Option<anyhow::Error>],
    member_set: &BTreeSet<PathBuf>,
) -> Result<bool> {
    if packages.iter().zip(missing).any(|(pkg, err)| {
        err.is_none()
            && pkg.local()
            && pkg.locked.git().is_none()
            && pkg_lock_mismatch(pkg, all_locked, member_set)
    }) {
        return Ok(true);
    }
    let mtime = |p: PathBuf| std::fs::metadata(p).ok()?.modified().ok();
    if !mtime(top.join(MANIFEST_FILE))
        .zip(mtime(top.join(LOCK_FILE)))
        .is_some_and(|(t, l)| t > l)
    {
        return Ok(false);
    }
    Ok(patch_paths(top)?.into_iter().any(|dir| {
        manifest_name(&dir)
            .is_some_and(|name| all_locked.iter().any(|e| e.name == name && e.is_registry()))
    }))
}

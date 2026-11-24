use std::hash::{Hash as _, Hasher as _};
use std::path::{Path, PathBuf};

use haste_codegen::Target;
use rustc_hash::FxHasher;

use super::{hash_cache_header, hash_file_meta, hash_program};

type Stamp = (u64, i128);

fn sysroot_library_roots(workspace: &crate::graph::Workspace, units: &[usize]) -> Vec<PathBuf> {
    let mut roots: Vec<PathBuf> = units
        .iter()
        .map(|&idx| {
            let dir = &workspace.packages[workspace.graph.units[idx].package]
                .manifest
                .dir;
            dir.ancestors()
                .find(|a| a.file_name().is_some_and(|n| n == "library"))
                .unwrap_or(dir.as_path())
                .to_owned()
        })
        .collect();
    roots.sort();
    roots.dedup();
    roots
}

fn scan_sysroot_dir(dir: &Path) -> Vec<(PathBuf, Option<Stamp>)> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out: Vec<_> = entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let path = entry.path();
            if matches!(
                path.file_name().and_then(|n| n.to_str()),
                Some("target" | ".git")
            ) {
                return None;
            }
            let ft = entry.file_type().ok();
            if ft.is_some_and(|ft| ft.is_dir()) {
                return Some((path, None));
            }
            let is_rs = path.extension().and_then(|e| e.to_str()) == Some("rs");
            let meta = if ft.is_some_and(|ft| ft.is_file()) {
                is_rs.then(|| entry.metadata().ok()).flatten()?
            } else {
                std::fs::metadata(&path).ok()?
            };
            if meta.is_dir() {
                Some((path, None))
            } else if is_rs {
                Some((path, Some(crate::graph::store::meta_stamp(&meta))))
            } else {
                None
            }
        })
        .collect();
    out.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    out
}

fn walk_sysroot_dir(dir: &Path, out: &mut Vec<(PathBuf, Stamp)>) {
    for (path, stamp) in scan_sysroot_dir(dir) {
        match stamp {
            Some(s) => out.push((path, s)),
            None => walk_sysroot_dir(&path, out),
        }
    }
}

fn hash_sysroot_roots(roots: Vec<PathBuf>, hasher: &mut FxHasher) {
    let mut work = Vec::new();
    for root in roots {
        for (p, s) in scan_sysroot_dir(&root) {
            if s.is_some() {
                work.push((p, s));
            } else {
                work.extend(scan_sysroot_dir(&p));
            }
        }
    }
    let dirs: Vec<(usize, &Path)> = work
        .iter()
        .enumerate()
        .filter(|(_, (_, s))| s.is_none())
        .map(|(i, (p, _))| (i, p.as_path()))
        .collect();
    let mut chunks = vec![Vec::new(); work.len()];
    for (slot, items) in crate::sync::parallel_map(dirs, |(slot, dir)| {
        let mut items = Vec::new();
        walk_sysroot_dir(dir, &mut items);
        (slot, items)
    }) {
        chunks[slot] = items;
    }
    for (i, (path, stamp)) in work.into_iter().enumerate() {
        match stamp {
            Some(s) => (path, s).hash(hasher),
            None => chunks[i].iter().for_each(|e| e.hash(hasher)),
        }
    }
}

fn hash_fast_sysroot_roots(roots: Vec<PathBuf>, hasher: &mut FxHasher) {
    if let Some(rustc) = std::env::var_os("RUSTC") {
        let path = PathBuf::from(rustc);
        hash_file_meta(&path, hasher);
    }
    for root in roots {
        hash_file_meta(&root, hasher);
        if root.ends_with("lib/rustlib/src/rust/library")
            && let Some(toolchain) = root.ancestors().nth(5)
        {
            let rustc_name = if cfg!(windows) {
                "bin/rustc.exe"
            } else {
                "bin/rustc"
            };
            hash_file_meta(&toolchain.join(rustc_name), hasher);
        }
        hash_file_meta(&root.join("Cargo.lock"), hasher);
        hash_file_meta(&root.join("sysroot/Cargo.lock"), hasher);
        for (entry, stamp) in scan_sysroot_dir(&root) {
            (entry.as_path(), stamp).hash(hasher);
            if stamp.is_none() {
                hash_file_meta(&entry, hasher);
                hash_file_meta(&entry.join("Cargo.toml"), hasher);
                hash_file_meta(&entry.join("src"), hasher);
            }
        }
    }
}

fn sorted_sysroot_units(workspace: &crate::graph::Workspace) -> Vec<usize> {
    let mut sysroot_units: Vec<(&str, crate::graph::Side, usize)> = workspace
        .graph
        .units
        .iter()
        .enumerate()
        .filter(|(_, unit)| workspace.packages[unit.package].sysroot)
        .map(|(idx, unit)| {
            (
                workspace.packages[unit.package].locked.name.as_str(),
                unit.side,
                idx,
            )
        })
        .collect();
    sysroot_units.sort_unstable_by(|a, b| (a.0, a.1, a.2).cmp(&(b.0, b.1, b.2)));
    sysroot_units.into_iter().map(|(_, _, idx)| idx).collect()
}

fn toolchain_key_for(
    workspace: &crate::graph::Workspace,
    triple: &str,
    effort: haste_codegen::Effort,
) -> u64 {
    let unit_indices = sorted_sysroot_units(workspace);
    let mut hasher = FxHasher::default();
    triple.hash(&mut hasher);
    effort.hash(&mut hasher);
    crate::graph::store::program().hash(&mut hasher);
    hash_program(&mut hasher);
    super::super::MAGIC.hash(&mut hasher);
    super::super::VERSION.hash(&mut hasher);
    for &idx in &unit_indices {
        let unit = &workspace.graph.units[idx];
        let pkg = &workspace.packages[unit.package];
        let locked = &pkg.locked;
        locked.name.hash(&mut hasher);
        locked.version.to_string().hash(&mut hasher);
        locked.source.hash(&mut hasher);
        matches!(unit.side, crate::graph::Side::Host).hash(&mut hasher);
        if let Some(lib) = &pkg.manifest.lib {
            hash_file_meta(lib, &mut hasher);
        }
        for feat in &unit.features {
            feat.hash(&mut hasher);
        }
    }
    let roots = sysroot_library_roots(workspace, &unit_indices);
    if std::env::var_os("HASTE_LIBRARY").is_some() {
        hash_sysroot_roots(roots, &mut hasher);
    } else {
        hash_fast_sysroot_roots(roots, &mut hasher);
    }
    hasher.finish()
}

/// Computes a workspace-invariant toolchain key from sysroot packages, target, and compiler binary.
pub fn toolchain_key(workspace: &crate::graph::Workspace, target: &Target) -> u64 {
    toolchain_key_for(workspace, target.triple, target.effort)
}

/// Computes a cache key for the settled sysroot outline and resolved names for `targets`.
pub fn sysroot_key(workspace: &crate::graph::Workspace, targets: &crate::graph::Targets) -> u64 {
    let mut hasher = FxHasher::default();
    hash_cache_header(&mut hasher);
    super::super::sysroot::SYSROOT_VERSION.hash(&mut hasher);
    toolchain_key_for(workspace, &targets.target.triple, crate::profile().into()).hash(&mut hasher);
    (
        &targets.target.triple,
        &targets.target.names,
        &targets.target.pairs,
        &targets.target.internal,
    )
        .hash(&mut hasher);
    (
        &targets.host.triple,
        &targets.host.names,
        &targets.host.pairs,
        &targets.host.internal,
    )
        .hash(&mut hasher);
    crate::graph::script::tool("CARGO").hash(&mut hasher);
    for idx in sorted_sysroot_units(workspace) {
        let unit = &workspace.graph.units[idx];
        let pkg = &workspace.packages[unit.package];
        (
            &pkg.manifest.edition,
            &pkg.manifest.lib_name,
            &pkg.manifest.dir,
            pkg.manifest.build.is_some(),
        )
            .hash(&mut hasher);
        if let Some(script) = &unit.script {
            script.key_parts().hash(&mut hasher);
        }
        let mut deps: Vec<(&str, &str, bool)> = unit
            .deps
            .iter()
            .map(|(key, dep_idx, _)| {
                let dep_unit = &workspace.graph.units[*dep_idx];
                let dep_pkg = &workspace.packages[dep_unit.package];
                (
                    key.as_str(),
                    dep_pkg.locked.name.as_str(),
                    matches!(dep_unit.side, crate::graph::Side::Host),
                )
            })
            .collect();
        deps.sort_unstable();
        deps.hash(&mut hasher);
    }
    hasher.finish()
}

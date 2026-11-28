//! Persistent mmap-able cache for settled workspace graphs, sysroot manifests, and build-script outputs.

mod codec;

use std::collections::BTreeSet;
use std::hash::{Hash as _, Hasher as _};
use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use rustc_hash::FxHasher;

use crate::config::Config;
use crate::sema::summary::codec::{Cursor, read_str, write_str, write_u32, write_u64};
use crate::substrate::Backing;
use crate::substrate::noop::{hash_target_dir, target_dirs, target_root_files};

use codec::{
    read_env_hashes, read_file_stamps, read_opt_bool, read_opt_path, read_package, read_path,
    read_strs, read_unit, write_env_hashes, write_file_stamps, write_opt_bool, write_opt_path,
    write_package, write_path, write_strs, write_unit,
};

use super::manifest::ProfileOverride;
use super::script::{LinkKind, Output};
use super::store::{self, FileStamp};
use super::{Graph, LOCK_FILE, MANIFEST_FILE, Targets, Workspace};

const WS_MAGIC: [u8; 8] = *b"HSTWKS01";
const WS_VERSION: u32 = 7;
const HEADER_LEN: usize = 28;
const MAX_KEPT_SNAPSHOTS: usize = 128;

const STATIC_ENV_VARS: [&str; 24] = [
    "RUSTFLAGS",
    "CARGO_ENCODED_RUSTFLAGS",
    "CARGO_NET_OFFLINE",
    crate::LIBRARY_ENV,
    "RUSTC",
    "RUSTUP_HOME",
    "RUSTUP_TOOLCHAIN",
    "CARGO_HOME",
    "HOME",
    "USERPROFILE",
    "CARGO",
    "CC",
    "CXX",
    "AR",
    "LD",
    "CFLAGS",
    "CXXFLAGS",
    "LDFLAGS",
    "OPT_LEVEL",
    "DEBUG",
    "PROFILE",
    "PATH",
    "TARGET",
    "HOST",
];

fn enabled() -> bool {
    crate::substrate::enabled() && !crate::env_flag("HASTE_NO_GRAPH_CACHE")
}

fn cache_dir() -> PathBuf {
    store::home().join("graph")
}

fn cache_path(key: u64) -> PathBuf {
    cache_dir().join(format!("{key:016x}.ws"))
}

fn manifest_root_of(root: &Path) -> PathBuf {
    let canonical = std::fs::canonicalize(root).unwrap_or_else(|_| root.to_owned());
    canonical
        .ancestors()
        .find(|dir| dir.join(MANIFEST_FILE).is_file())
        .unwrap_or(&canonical)
        .to_owned()
}

fn snapshot_key(manifest_root: &Path, targets: &Targets, cfg: &Config) -> u64 {
    let mut hasher = FxHasher::default();
    WS_MAGIC.hash(&mut hasher);
    WS_VERSION.hash(&mut hasher);
    manifest_root.hash(&mut hasher);
    cfg.hash_cache_key(&mut hasher);
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
    store::program().hash(&mut hasher);
    for name in [
        "CARGO_HOME",
        "HOME",
        "USERPROFILE",
        "RUSTUP_HOME",
        "RUSTUP_TOOLCHAIN",
        "RUSTC",
        "CARGO_NET_OFFLINE",
        crate::LIBRARY_ENV,
    ] {
        crate::process_env_var(name).hash(&mut hasher);
    }
    hasher.finish()
}

fn hash_target_dirs(dirs: &[PathBuf], root_files: &[PathBuf]) -> u64 {
    let mut hasher = FxHasher::default();
    dirs.len().hash(&mut hasher);
    for dir in dirs {
        hash_target_dir(dir, &mut hasher);
    }
    root_files.len().hash(&mut hasher);
    for file in root_files {
        file.is_file().hash(&mut hasher);
    }
    hasher.finish()
}

pub(crate) fn rustup_settings_path() -> Option<PathBuf> {
    let home = crate::process_env_var("RUSTUP_HOME")
        .map(PathBuf::from)
        .or_else(|| {
            crate::process_env_var("HOME")
                .or_else(|| crate::process_env_var("USERPROFILE"))
                .map(|p| PathBuf::from(p).join(".rustup"))
        })?;
    Some(home.join("settings.toml"))
}

pub(crate) fn collect_static_lib_paths(pkg_dir: &Path, out: &Output, paths: &mut Vec<PathBuf>) {
    let static_names: Vec<&str> = out
        .link_libs()
        .filter_map(|(kind, name)| (kind == LinkKind::Static).then_some(name))
        .collect();
    if static_names.is_empty() {
        return;
    }
    for dir_spec in out.search_dirs() {
        let raw = Path::new(dir_spec);
        let dir = if raw.is_absolute() {
            raw.to_owned()
        } else {
            pkg_dir.join(raw)
        };
        for &name in &static_names {
            paths.push(dir.join(format!("lib{name}.a")));
            paths.push(dir.join(format!("{name}.lib")));
        }
    }
}

struct WatchedGraph {
    struct_files: Vec<(PathBuf, FileStamp)>,
    struct_envs: Vec<(String, Option<u64>)>,
    dirs: Vec<PathBuf>,
    root_files: Vec<PathBuf>,
    dirs_hash: u64,
    library: Option<PathBuf>,
    script_files: Vec<(PathBuf, FileStamp)>,
    script_envs: Vec<(String, Option<u64>)>,
}

fn hash_env_names(mut names: Vec<String>) -> Vec<(String, Option<u64>)> {
    names.sort();
    names.dedup();
    names
        .into_iter()
        .map(|name| {
            let hash = crate::process_env_var(&name)
                .as_deref()
                .map(|v| store::hash_env_val(&WS_MAGIC, &name, v));
            (name, hash)
        })
        .collect()
}

fn envs_match(envs: &[(String, Option<u64>)]) -> bool {
    envs.iter().all(|(name, expected)| {
        crate::process_env_var(name)
            .as_deref()
            .map(|v| store::hash_env_val(&WS_MAGIC, name, v))
            == *expected
    })
}

fn collect_watched(
    workspace: &Workspace,
    manifest_root: &Path,
    library: Option<&Path>,
) -> WatchedGraph {
    let mut build_files = BTreeSet::new();
    for pkg in &workspace.packages {
        if let Some(build) = &pkg.manifest.build {
            build_files.insert(build.clone());
        }
    }

    let mut struct_paths: Vec<PathBuf> = super::manifest_paths(workspace, manifest_root)
        .into_iter()
        .filter(|path| !build_files.contains(path))
        .collect();
    let root_files = target_root_files(workspace);
    if let Some(settings) = rustup_settings_path() {
        struct_paths.push(settings);
    }
    if let Some(lib) = library {
        let sysroot = lib.join("sysroot");
        struct_paths.push(sysroot.join(MANIFEST_FILE));
        struct_paths.push(sysroot.join(LOCK_FILE));
    }
    let top = super::workspace_root(manifest_root).unwrap_or_else(|_| manifest_root.to_owned());
    let mut script_paths: Vec<PathBuf> = build_files.into_iter().collect();
    for pkg in &workspace.packages {
        if pkg.sysroot {
            struct_paths.push(pkg.manifest.dir.join(MANIFEST_FILE));
        } else if pkg.local() {
            for anc in pkg.manifest.dir.ancestors() {
                let candidate = anc.join(MANIFEST_FILE);
                if candidate.is_file() {
                    struct_paths.push(candidate);
                }
                if anc == top {
                    break;
                }
            }
        }
    }
    struct_paths.sort();
    struct_paths.dedup();
    let struct_stamps = store::file_stamps(&struct_paths);
    let struct_files: Vec<(PathBuf, FileStamp)> =
        struct_paths.into_iter().zip(struct_stamps).collect();

    let (script_walked, script_vars) = store::watched_stamps(workspace);
    script_paths.extend(script_walked.into_iter().map(|(p, _, _)| p));

    let mut struct_env_names: Vec<String> = STATIC_ENV_VARS.iter().map(|&s| s.to_owned()).collect();
    struct_env_names.extend(workspace.config_env.iter().map(|(k, _, _)| k.clone()));

    let mut script_env_names: Vec<String> = script_vars.into_iter().map(|(k, _)| k).collect();
    let mut extra_walked = Vec::new();
    for (unit, entry) in workspace.graph.units.iter().enumerate() {
        let pkg = &workspace.packages[entry.package];
        if pkg.manifest.build.is_some() {
            let sdir = store::dir(workspace, unit);
            script_paths.push(sdir.join("stamp"));
            script_paths.push(sdir.join("output"));
            script_paths.push(sdir.join("failed"));
            script_paths.push(sdir.join("out"));
        }
        if let Some(out) = &entry.script {
            script_env_names.extend(out.watched_vars.iter().cloned());
            for rel in &out.watched {
                let watched_path = pkg.manifest.dir.join(rel);
                script_paths.push(watched_path.clone());
                store::walk(&watched_path, &mut extra_walked);
            }
            collect_static_lib_paths(&pkg.manifest.dir, out, &mut script_paths);
        }
    }
    script_paths.extend(extra_walked.into_iter().map(|(p, _, _)| p));
    script_paths.sort();
    script_paths.dedup();
    let script_stamps = store::file_stamps(&script_paths);
    let script_files: Vec<(PathBuf, FileStamp)> =
        script_paths.into_iter().zip(script_stamps).collect();

    let dirs = target_dirs(workspace);
    let dirs_hash = hash_target_dirs(&dirs, &root_files);
    WatchedGraph {
        struct_files,
        struct_envs: hash_env_names(struct_env_names),
        dirs,
        root_files,
        dirs_hash,
        library: library.map(Path::to_path_buf),
        script_files,
        script_envs: hash_env_names(script_env_names),
    }
}

enum SnapshotState {
    Miss,
    Hit(Workspace),
    ScriptsStale(Workspace, Option<PathBuf>),
}

fn encode_snapshot(key: u64, watched: &WatchedGraph, workspace: &Workspace) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(
        HEADER_LEN
            + (watched.struct_files.len() + watched.script_files.len()) * 80
            + (watched.struct_envs.len() + watched.script_envs.len()) * 32
            + workspace.packages.len() * 512,
    );
    out.extend_from_slice(&WS_MAGIC);
    write_u32(&mut out, WS_VERSION);
    write_u64(&mut out, key);
    write_u64(&mut out, 0);

    write_file_stamps(&mut out, &watched.struct_files)?;
    write_env_hashes(&mut out, &watched.struct_envs)?;
    write_u32(&mut out, u32::try_from(watched.dirs.len()).ok()?);
    for dir in &watched.dirs {
        write_path(&mut out, dir)?;
    }
    write_u32(&mut out, u32::try_from(watched.root_files.len()).ok()?);
    for file in &watched.root_files {
        write_path(&mut out, file)?;
    }
    write_u64(&mut out, watched.dirs_hash);

    write_opt_path(&mut out, watched.library.as_deref())?;
    write_file_stamps(&mut out, &watched.script_files)?;
    write_env_hashes(&mut out, &watched.script_envs)?;

    write_strs(&mut out, &workspace.rustflags)?;
    write_u32(&mut out, u32::try_from(workspace.config_env.len()).ok()?);
    for (k, v, force) in &workspace.config_env {
        write_str(&mut out, k)?;
        write_str(&mut out, v)?;
        out.push(u8::from(*force));
    }
    write_opt_bool(&mut out, workspace.profile_dev.debug_assertions);
    write_opt_bool(&mut out, workspace.profile_dev.overflow_checks);
    write_opt_bool(&mut out, workspace.profile_release.debug_assertions);
    write_opt_bool(&mut out, workspace.profile_release.overflow_checks);

    write_u32(&mut out, u32::try_from(workspace.packages.len()).ok()?);
    for pkg in &workspace.packages {
        write_package(&mut out, pkg)?;
    }
    write_u32(&mut out, u32::try_from(workspace.graph.units.len()).ok()?);
    for unit in &workspace.graph.units {
        write_unit(&mut out, unit)?;
    }
    write_u32(&mut out, u32::try_from(workspace.graph.members.len()).ok()?);
    for &m in &workspace.graph.members {
        write_u32(&mut out, u32::try_from(m).ok()?);
    }
    out.extend_from_slice(&WS_MAGIC);
    let total = u64::try_from(out.len()).ok()?;
    out[20..28].copy_from_slice(&total.to_le_bytes());
    Some(out)
}

fn decode_snapshot(path: &Path, expected_key: u64) -> Result<SnapshotState> {
    let backing = match Backing::open(path, HEADER_LEN + WS_MAGIC.len()) {
        Ok(b) => b,
        Err(_) => return Ok(SnapshotState::Miss),
    };
    let data = backing.as_slice();
    if data.len() < HEADER_LEN + WS_MAGIC.len()
        || data[..8] != WS_MAGIC
        || data[data.len() - 8..] != WS_MAGIC
    {
        bail!("corrupt workspace snapshot framing");
    }
    let mut cur = Cursor::new(&data[..data.len() - 8]);
    cur.take(8)?;
    if cur.u32()? != WS_VERSION || cur.u64()? != expected_key {
        return Ok(SnapshotState::Miss);
    }
    if cur.u64()? != data.len() as u64 {
        bail!("workspace snapshot length mismatch");
    }

    let struct_files = read_file_stamps(&mut cur)?;
    let struct_envs = read_env_hashes(&mut cur)?;
    let dir_count = cur.count()?;
    let mut dirs = Vec::with_capacity(dir_count.min(256));
    for _ in 0..dir_count {
        dirs.push(read_path(&mut cur)?);
    }
    let root_count = cur.count()?;
    let mut root_files = Vec::with_capacity(root_count.min(256));
    for _ in 0..root_count {
        root_files.push(read_path(&mut cur)?);
    }
    let dirs_hash = cur.u64()?;

    if !envs_match(&struct_envs)
        || hash_target_dirs(&dirs, &root_files) != dirs_hash
        || !store::stamps_match(&struct_files, true)
    {
        return Ok(SnapshotState::Miss);
    }

    let library = read_opt_path(&mut cur)?;
    let script_files = read_file_stamps(&mut cur)?;
    let script_envs = read_env_hashes(&mut cur)?;
    let scripts_ok = envs_match(&script_envs) && store::stamps_match(&script_files, true);

    let rustflags = read_strs(&mut cur)?;
    let cfg_env_len = cur.count()?;
    let mut config_env = Vec::with_capacity(cfg_env_len);
    for _ in 0..cfg_env_len {
        let k = read_str(&mut cur)?.to_owned();
        let v = read_str(&mut cur)?.to_owned();
        let force = cur.u8()? != 0;
        config_env.push((k, v, force));
    }
    let profile_dev = ProfileOverride {
        debug_assertions: read_opt_bool(&mut cur)?,
        overflow_checks: read_opt_bool(&mut cur)?,
    };
    let profile_release = ProfileOverride {
        debug_assertions: read_opt_bool(&mut cur)?,
        overflow_checks: read_opt_bool(&mut cur)?,
    };
    let pkg_len = cur.count()?;
    let mut packages = Vec::with_capacity(pkg_len);
    for _ in 0..pkg_len {
        packages.push(read_package(&mut cur)?);
    }
    let unit_len = cur.count()?;
    let mut units = Vec::with_capacity(unit_len);
    for _ in 0..unit_len {
        units.push(read_unit(&mut cur)?);
    }
    let member_len = cur.count()?;
    let mut members = Vec::with_capacity(member_len);
    for _ in 0..member_len {
        members.push(cur.u32()? as usize);
    }
    if cur.pos != data.len() - 8 {
        bail!("trailing bytes in workspace snapshot");
    }
    let workspace = Workspace {
        packages,
        graph: Graph { units, members },
        rustflags,
        config_env,
        profile_dev,
        profile_release,
    };
    if scripts_ok {
        Ok(SnapshotState::Hit(workspace))
    } else {
        Ok(SnapshotState::ScriptsStale(workspace, library))
    }
}

fn prune_stale(dir: &Path, keep: &Path, max_others: usize) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut files = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path == keep || path.extension().and_then(|e| e.to_str()) != Some("ws") {
            continue;
        }
        let stamp = entry
            .metadata()
            .ok()
            .and_then(|m| m.modified().ok())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        files.push((stamp, path));
    }
    if files.len() <= max_others {
        return;
    }
    files.sort_by(|a, b| b.0.cmp(&a.0));
    for (_, stale) in files.into_iter().skip(max_others) {
        drop(std::fs::remove_file(stale));
    }
}

fn save_snapshot(key: u64, workspace: &Workspace, manifest_root: &Path, library: Option<&Path>) {
    let watched = collect_watched(workspace, manifest_root, library);
    let Some(bytes) = encode_snapshot(key, &watched, workspace) else {
        return;
    };
    let path = cache_path(key);
    if store::write_new_bytes(&path, &bytes).is_ok() {
        prune_stale(&cache_dir(), &path, MAX_KEPT_SNAPSHOTS - 1);
    }
}

/// Loads a settled [`Workspace`] from `$HASTE_HOME/graph/` when valid, else builds and caches it.
pub(crate) fn load_or_build(root: &Path, targets: &Targets) -> Result<Workspace> {
    if !enabled() {
        let mut workspace = super::load(root, targets, &crate::selection())?;
        workspace.merge(super::load_sysroot(&super::library()?, targets)?);
        crate::scripts::settle(&mut workspace, targets)?;
        return Ok(workspace);
    }
    let manifest_root = manifest_root_of(root);
    let key = snapshot_key(&manifest_root, targets, &Config::current());
    let path = cache_path(key);
    if path.is_file() {
        match decode_snapshot(&path, key) {
            Ok(SnapshotState::Hit(workspace)) => return Ok(workspace),
            Ok(SnapshotState::ScriptsStale(mut workspace, library)) => {
                for unit in &mut workspace.graph.units {
                    unit.script = None;
                }
                crate::scripts::settle(&mut workspace, targets)?;
                let lib = match library {
                    Some(p) => Some(p),
                    None => super::library().ok(),
                };
                save_snapshot(key, &workspace, &manifest_root, lib.as_deref());
                return Ok(workspace);
            }
            Ok(SnapshotState::Miss) => {}
            Err(_) => drop(std::fs::remove_file(&path)),
        }
    }
    let mut workspace = super::load(root, targets, &crate::selection())?;
    let library = super::library()?;
    workspace.merge(super::load_sysroot(&library, targets)?);
    crate::scripts::settle(&mut workspace, targets)?;
    save_snapshot(key, &workspace, &manifest_root, Some(&library));
    Ok(workspace)
}

#[cfg(test)]
mod tests;

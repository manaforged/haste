//! Persistent mmap-able cache for settled workspace graphs, sysroot manifests, and build-script outputs.

mod codec;

use std::hash::{Hash as _, Hasher as _};
use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use rustc_hash::FxHasher;

use crate::config::Config;
use crate::sema::summary::codec::{Cursor, read_str, write_str, write_u32, write_u64};
use crate::substrate::Backing;
use crate::substrate::noop::{hash_target_dir, target_dirs, target_root_files};

use codec::{
    read_opt_bool, read_package, read_path, read_strs, read_unit, write_opt_bool, write_package,
    write_path, write_strs, write_unit,
};

use super::manifest::ProfileOverride;
use super::script::{LinkKind, Output};
use super::store::{self, FileStamp};
use super::{Graph, LOCK_FILE, MANIFEST_FILE, Targets, Workspace};

const WS_MAGIC: [u8; 8] = *b"HSTWKS01";
const WS_VERSION: u32 = 2;
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

fn rustup_settings_path() -> Option<PathBuf> {
    let home = crate::process_env_var("RUSTUP_HOME")
        .map(PathBuf::from)
        .or_else(|| {
            crate::process_env_var("HOME")
                .or_else(|| crate::process_env_var("USERPROFILE"))
                .map(|p| PathBuf::from(p).join(".rustup"))
        })?;
    Some(home.join("settings.toml"))
}

fn collect_static_lib_paths(pkg_dir: &Path, out: &Output, paths: &mut Vec<PathBuf>) {
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
    files: Vec<(PathBuf, FileStamp)>,
    envs: Vec<(String, Option<u64>)>,
    dirs: Vec<PathBuf>,
    root_files: Vec<PathBuf>,
    dirs_hash: u64,
}

fn collect_watched(
    workspace: &Workspace,
    manifest_root: &Path,
    library: Option<&Path>,
) -> WatchedGraph {
    let mut paths = super::manifest_paths(workspace, manifest_root);
    let root_files = target_root_files(workspace);
    if let Some(settings) = rustup_settings_path() {
        paths.push(settings);
    }
    if let Some(lib) = library {
        let sysroot = lib.join("sysroot");
        paths.push(sysroot.join(MANIFEST_FILE));
        paths.push(sysroot.join(LOCK_FILE));
    }
    let top = super::workspace_root(manifest_root).unwrap_or_else(|_| manifest_root.to_owned());
    for pkg in &workspace.packages {
        if pkg.sysroot {
            paths.push(pkg.manifest.dir.join(MANIFEST_FILE));
            if let Some(build) = &pkg.manifest.build {
                paths.push(build.clone());
            }
        } else if pkg.local() {
            paths.push(pkg.manifest.dir.join("build.rs"));
            for anc in pkg.manifest.dir.ancestors() {
                let candidate = anc.join(MANIFEST_FILE);
                if candidate.is_file() {
                    paths.push(candidate);
                }
                if anc == top {
                    break;
                }
            }
        }
    }
    let (script_walked, script_vars) = store::watched_stamps(workspace);
    paths.extend(script_walked.into_iter().map(|(p, _, _)| p));

    let mut env_names: Vec<String> = STATIC_ENV_VARS.iter().map(|&s| s.to_owned()).collect();
    env_names.extend(script_vars.into_iter().map(|(k, _)| k));
    env_names.extend(workspace.config_env.iter().map(|(k, _, _)| k.clone()));

    let mut extra_walked = Vec::new();
    for (unit, entry) in workspace.graph.units.iter().enumerate() {
        let pkg = &workspace.packages[entry.package];
        if pkg.manifest.build.is_some() {
            let sdir = store::dir(workspace, unit);
            paths.push(sdir.join("stamp"));
            paths.push(sdir.join("output"));
            paths.push(sdir.join("failed"));
            paths.push(sdir.join("out"));
        }
        if let Some(out) = &entry.script {
            env_names.extend(out.watched_vars.iter().cloned());
            for rel in &out.watched {
                let watched_path = pkg.manifest.dir.join(rel);
                paths.push(watched_path.clone());
                store::walk(&watched_path, &mut extra_walked);
            }
            collect_static_lib_paths(&pkg.manifest.dir, out, &mut paths);
        }
    }
    paths.extend(extra_walked.into_iter().map(|(p, _, _)| p));
    paths.sort();
    paths.dedup();
    let stamps = store::file_stamps(&paths);
    let watched_files: Vec<(PathBuf, FileStamp)> = paths.into_iter().zip(stamps).collect();

    env_names.sort();
    env_names.dedup();
    let watched_envs: Vec<(String, Option<u64>)> = env_names
        .into_iter()
        .map(|name| {
            let hash = crate::process_env_var(&name)
                .as_deref()
                .map(|v| store::hash_env_val(&WS_MAGIC, &name, v));
            (name, hash)
        })
        .collect();

    let dirs = target_dirs(workspace);
    let dirs_hash = hash_target_dirs(&dirs, &root_files);
    WatchedGraph {
        files: watched_files,
        envs: watched_envs,
        dirs,
        root_files,
        dirs_hash,
    }
}

fn encode_snapshot(key: u64, watched: &WatchedGraph, workspace: &Workspace) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(
        HEADER_LEN
            + watched.files.len() * 80
            + watched.envs.len() * 32
            + workspace.packages.len() * 512,
    );
    out.extend_from_slice(&WS_MAGIC);
    write_u32(&mut out, WS_VERSION);
    write_u64(&mut out, key);
    write_u64(&mut out, 0);

    write_u32(&mut out, u32::try_from(watched.files.len()).ok()?);
    for (path, stamp) in &watched.files {
        write_path(&mut out, path)?;
        match stamp {
            Some((len, nanos)) => {
                out.push(1);
                write_u64(&mut out, *len);
                out.extend_from_slice(&nanos.to_le_bytes());
            }
            None => out.push(0),
        }
    }
    write_u32(&mut out, u32::try_from(watched.envs.len()).ok()?);
    for (name, val) in &watched.envs {
        write_str(&mut out, name)?;
        match val {
            Some(hash) => {
                out.push(1);
                write_u64(&mut out, *hash);
            }
            None => out.push(0),
        }
    }
    write_u32(&mut out, u32::try_from(watched.dirs.len()).ok()?);
    for dir in &watched.dirs {
        write_path(&mut out, dir)?;
    }
    write_u32(&mut out, u32::try_from(watched.root_files.len()).ok()?);
    for file in &watched.root_files {
        write_path(&mut out, file)?;
    }
    write_u64(&mut out, watched.dirs_hash);

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

fn decode_snapshot(path: &Path, expected_key: u64) -> Result<Option<Workspace>> {
    let backing = match Backing::open(path, HEADER_LEN + WS_MAGIC.len()) {
        Ok(b) => b,
        Err(_) => return Ok(None),
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
        return Ok(None);
    }
    if cur.u64()? != data.len() as u64 {
        bail!("workspace snapshot length mismatch");
    }

    let file_count = cur.count()?;
    let mut watched_files = Vec::with_capacity(file_count.min(4096));
    for _ in 0..file_count {
        let p = read_path(&mut cur)?;
        let stamp = match cur.u8()? {
            0 => None,
            1 => Some((cur.u64()?, cur.u128()? as i128)),
            _ => bail!("invalid file stamp tag"),
        };
        watched_files.push((p, stamp));
    }
    let env_count = cur.count()?;
    let mut watched_envs = Vec::with_capacity(env_count.min(512));
    for _ in 0..env_count {
        let name = read_str(&mut cur)?.to_owned();
        let hash = match cur.u8()? {
            0 => None,
            1 => Some(cur.u64()?),
            _ => bail!("invalid env hash tag"),
        };
        watched_envs.push((name, hash));
    }
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

    let envs_ok = watched_envs.iter().all(|(name, expected)| {
        crate::process_env_var(name)
            .as_deref()
            .map(|v| store::hash_env_val(&WS_MAGIC, name, v))
            == *expected
    });
    if !envs_ok
        || hash_target_dirs(&dirs, &root_files) != dirs_hash
        || !store::stamps_match(&watched_files, true)
    {
        return Ok(None);
    }

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
    Ok(Some(Workspace {
        packages,
        graph: Graph { units, members },
        rustflags,
        config_env,
        profile_dev,
        profile_release,
    }))
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
            Ok(Some(workspace)) => return Ok(workspace),
            Ok(None) => {}
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

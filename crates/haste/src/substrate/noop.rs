//! Workspace no-op build fingerprint for skipping unchanged one-shot `haste build` runs.

use std::ffi::OsString;
use std::hash::{Hash as _, Hasher as _};
use std::path::{Path, PathBuf};

use super::wire::{read_path, read_u32, read_u64, read_utf8, write_len_bytes, write_path};

use rustc_hash::FxHasher;

const NOOP_MAGIC: [u8; 8] = *b"HSTNOP02";
const HEADER_LEN: usize = 36;
const TARGET_DIRS: [&str; 5] = ["src", "src/bin", "tests", "examples", "benches"];
const ROOT_FILES: [&str; 3] = ["src/lib.rs", "src/main.rs", "build.rs"];

/// Returns whether the CLI no-op fast path is active for this `build` invocation.
pub fn active(stats: Option<&Path>) -> bool {
    super::enabled()
        && stats.is_none()
        && !crate::image::verify_stencil()
        && !crate::env_flag("HASTE_NO_NOOP")
}

fn canonical_root(root: &Path) -> PathBuf {
    std::fs::canonicalize(root).unwrap_or_else(|_| root.to_owned())
}

fn abs_out(out: Option<&Path>) -> Option<PathBuf> {
    out.map(|p| std::path::absolute(p).unwrap_or_else(|_| p.to_owned()))
}

fn stamp_path_for(root: &Path, out: Option<&Path>, cfg: &crate::config::Config) -> PathBuf {
    let mut hasher = FxHasher::default();
    NOOP_MAGIC.hash(&mut hasher);
    root.hash(&mut hasher);
    out.hash(&mut hasher);
    cfg.hash_cache_key(&mut hasher);
    crate::graph::store::home()
        .join("noop")
        .join(format!("{:016x}.stamp", hasher.finish()))
}

fn stamp_path(root: &Path, out: Option<&Path>) -> PathBuf {
    stamp_path_for(root, out, &crate::config::Config::current())
}

fn target_dirs(workspace: &crate::graph::Workspace) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    for &member in &workspace.graph.members {
        if let Some(pkg) = workspace.packages.get(member) {
            for sub in TARGET_DIRS {
                dirs.push(pkg.manifest.dir.join(sub));
            }
        }
    }
    dirs.sort();
    dirs.dedup();
    dirs
}

fn target_root_files(workspace: &crate::graph::Workspace) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for pkg in workspace.packages.iter().filter(|pkg| pkg.local()) {
        for rel in ROOT_FILES {
            files.push(pkg.manifest.dir.join(rel));
        }
        files.extend(pkg.manifest.lib.iter().cloned());
        files.extend(pkg.manifest.build.iter().cloned());
        files.extend(pkg.manifest.bins.iter().map(|t| t.path.clone()));
        files.extend(pkg.manifest.examples.iter().map(|t| t.path.clone()));
        files.extend(pkg.manifest.tests.iter().map(|t| t.path.clone()));
        files.extend(pkg.manifest.benches.iter().map(|t| t.path.clone()));
    }
    files.sort();
    files.dedup();
    files
}

fn hash_target_dir(dir: &Path, hasher: &mut FxHasher) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        (dir, false).hash(hasher);
        return;
    };
    let mut found: Vec<OsString> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() && path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
            found.push(entry.file_name());
        } else if path.is_dir() && path.join("main.rs").is_file() {
            let mut name = entry.file_name();
            name.push("/main.rs");
            found.push(name);
        }
    }
    found.sort();
    (dir, true, found).hash(hasher);
}

/// Computes a fingerprint of auto-discovered target directories and root files for workspace reload checks.
pub(crate) fn target_discovery_stamp(workspace: &crate::graph::Workspace) -> u64 {
    let mut hasher = FxHasher::default();
    for dir in target_dirs(workspace) {
        hash_target_dir(&dir, &mut hasher);
    }
    for path in target_root_files(workspace) {
        (path.as_path(), path.is_file()).hash(&mut hasher);
    }
    hasher.finish()
}

fn compute_stamp(
    root: &Path,
    out: Option<&Path>,
    paths: &[PathBuf],
    dirs: &[PathBuf],
    env_names: &[String],
) -> Option<u64> {
    let target = crate::host_target();
    let cfg = crate::config::Config::current();
    let mut hasher = FxHasher::default();
    NOOP_MAGIC.hash(&mut hasher);
    super::target_hash(&target).hash(&mut hasher);
    cfg.hash_cache_key(&mut hasher);
    std::env::var_os("CARGO_HOME").hash(&mut hasher);
    crate::process_env_var("RUSTFLAGS").hash(&mut hasher);
    crate::process_env_var("CARGO_ENCODED_RUSTFLAGS").hash(&mut hasher);
    root.hash(&mut hasher);
    match out {
        Some(out_path) => {
            if !std::fs::metadata(out_path).is_ok_and(|m| m.is_file()) {
                return None;
            }
            true.hash(&mut hasher);
            out_path.hash(&mut hasher);
            crate::graph::store::file_stamp(out_path)?.hash(&mut hasher);
        }
        None => false.hash(&mut hasher),
    }
    paths.len().hash(&mut hasher);
    for path in paths {
        (path, crate::graph::store::file_stamp(path)).hash(&mut hasher);
    }
    dirs.len().hash(&mut hasher);
    for dir in dirs {
        hash_target_dir(dir, &mut hasher);
    }
    env_names.len().hash(&mut hasher);
    for name in env_names {
        (name, std::env::var_os(name)).hash(&mut hasher);
    }
    Some(hasher.finish())
}

fn read_paths(data: &[u8], pos: &mut usize, count: usize) -> Option<Vec<PathBuf>> {
    let mut paths = Vec::with_capacity(count.min(4096));
    for _ in 0..count {
        paths.push(read_path(data, pos)?);
    }
    Some(paths)
}

fn write_paths(out: &mut Vec<u8>, paths: &[PathBuf]) -> Option<()> {
    for path in paths {
        write_path(out, path)?;
    }
    Some(())
}

type DecodedStamp = (u64, usize, Vec<PathBuf>, Vec<PathBuf>, Vec<String>);

fn decode_stamp_file(data: &[u8]) -> Option<DecodedStamp> {
    if data.len() < HEADER_LEN || data[0..8] != NOOP_MAGIC {
        return None;
    }
    let stamp = read_u64(data, 8);
    let reached = usize::try_from(read_u64(data, 16)).ok()?;
    if reached == 0 {
        return None;
    }
    let path_count = read_u32(data, 24) as usize;
    let dir_count = read_u32(data, 28) as usize;
    let env_count = read_u32(data, 32) as usize;
    let mut pos = HEADER_LEN;
    let paths = read_paths(data, &mut pos, path_count)?;
    let dirs = read_paths(data, &mut pos, dir_count)?;
    let mut envs = Vec::with_capacity(env_count.min(1024));
    for _ in 0..env_count {
        envs.push(read_utf8(data, &mut pos)?);
    }
    (pos == data.len()).then_some((stamp, reached, paths, dirs, envs))
}

fn encode_stamp_file(
    stamp: u64,
    reached: usize,
    paths: &[PathBuf],
    dirs: &[PathBuf],
    envs: &[String],
) -> Option<Vec<u8>> {
    let path_count = u32::try_from(paths.len()).ok()?;
    let dir_count = u32::try_from(dirs.len()).ok()?;
    let env_count = u32::try_from(envs.len()).ok()?;
    let cap = HEADER_LEN + (paths.len() + dirs.len()) * 48 + envs.len() * 16;
    let mut out = Vec::with_capacity(cap);
    out.extend_from_slice(&NOOP_MAGIC);
    out.extend_from_slice(&stamp.to_le_bytes());
    out.extend_from_slice(&(reached as u64).to_le_bytes());
    out.extend_from_slice(&path_count.to_le_bytes());
    out.extend_from_slice(&dir_count.to_le_bytes());
    out.extend_from_slice(&env_count.to_le_bytes());
    write_paths(&mut out, paths)?;
    write_paths(&mut out, dirs)?;
    for env in envs {
        write_len_bytes(&mut out, env.as_bytes())?;
    }
    Some(out)
}

/// Returns the cached `reached` count when `root` and all watched inputs match the previous `0 failed` build.
pub fn try_hit(root: &Path, out: Option<&Path>) -> Option<usize> {
    let croot = canonical_root(root);
    let aout = abs_out(out);
    if let Some(out_path) = aout.as_deref()
        && !std::fs::metadata(out_path).is_ok_and(|m| m.is_file())
    {
        return None;
    }
    let data = std::fs::read(stamp_path(&croot, aout.as_deref())).ok()?;
    let (stored_stamp, reached, paths, dirs, envs) = decode_stamp_file(&data)?;
    let current = compute_stamp(&croot, aout.as_deref(), &paths, &dirs, &envs)?;
    (current == stored_stamp).then_some(reached)
}

/// Records the workspace no-op stamp after a `0 failed` build succeeds.
pub fn record(
    root: &Path,
    workspace: &crate::graph::Workspace,
    build_paths: &[PathBuf],
    env_reads: &[String],
    out: Option<&Path>,
    reached: usize,
) {
    if reached == 0 {
        return;
    }
    let croot = canonical_root(root);
    let aout = abs_out(out);
    let mut paths = crate::graph::manifest_paths(workspace, &croot);
    paths.extend(build_paths.iter().cloned());
    paths.extend(target_root_files(workspace));
    let (watched_files, watched_vars) = crate::graph::store::watched_stamps(workspace);
    paths.extend(watched_files.into_iter().map(|(path, _, _)| path));
    paths.sort();
    paths.dedup();
    let dirs = target_dirs(workspace);
    let mut envs: Vec<String> = watched_vars.into_iter().map(|(name, _)| name).collect();
    envs.extend(env_reads.iter().cloned());
    envs.sort();
    envs.dedup();
    let Some(stamp) = compute_stamp(&croot, aout.as_deref(), &paths, &dirs, &envs) else {
        return;
    };
    if let Some(encoded) = encode_stamp_file(stamp, reached, &paths, &dirs, &envs) {
        drop(crate::graph::store::write_new_bytes(
            &stamp_path(&croot, aout.as_deref()),
            &encoded,
        ));
    }
}

/// Removes any saved workspace no-op stamp after a failed build.
pub fn invalidate(root: &Path, out: Option<&Path>) {
    let croot = canonical_root(root);
    let aout = abs_out(out);
    drop(std::fs::remove_file(stamp_path(&croot, aout.as_deref())));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stamp_paths_separate_features_targets_and_env_flags() {
        let host = haste_codegen::Target::host();
        let cross = haste_codegen::TARGETS
            .iter()
            .find(|t| t.triple != host.triple)
            .expect("non-host target exists");
        let root = Path::new("/tmp/sample-ws");
        let base = crate::config::Config::for_target(host);
        let p_base = stamp_path_for(root, None, &base);
        let p_feat = stamp_path_for(
            root,
            None,
            &crate::config::Config {
                selection: crate::Selection {
                    features: vec!["extra".to_owned()],
                    ..crate::Selection::default()
                },
                ..base.clone()
            },
        );
        let p_target = stamp_path_for(root, None, &crate::config::Config::for_target(cross));
        let p_debug = stamp_path_for(
            root,
            None,
            &crate::config::Config {
                debug_info: !base.debug_info,
                ..base
            },
        );
        assert_ne!(p_base, p_feat);
        assert_ne!(p_base, p_target);
        assert_ne!(p_base, p_debug);
        assert_ne!(p_feat, p_target);
    }

    #[test]
    fn compute_stamp_distinguishes_rustflags_changes() {
        let stamp_with = |pairs: &[(&str, &str)]| {
            let map: rustc_hash::FxHashMap<String, String> = pairs
                .iter()
                .map(|&(k, v)| (k.to_owned(), v.to_owned()))
                .collect();
            crate::with_client_env(Some(std::sync::Arc::new(map)), || {
                compute_stamp(Path::new("/tmp/sample-ws"), None, &[], &[], &[])
            })
            .expect("compute_stamp returns Some when out is None")
        };
        let s_unset = stamp_with(&[]);
        let s_empty = stamp_with(&[("RUSTFLAGS", "")]);
        let s_flag = stamp_with(&[("RUSTFLAGS", "--cfg my_flag")]);
        let s_encoded = stamp_with(&[("CARGO_ENCODED_RUSTFLAGS", "--cfg\x1fmy_flag")]);
        assert_ne!(s_unset, s_empty);
        assert_ne!(s_unset, s_flag);
        assert_ne!(s_empty, s_flag);
        assert_ne!(s_unset, s_encoded);
        assert_ne!(s_flag, s_encoded);
    }
}

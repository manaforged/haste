//! Workspace no-op build fingerprint for skipping unchanged one-shot `haste build` runs.

use std::ffi::OsString;
use std::hash::{Hash as _, Hasher as _};
use std::os::unix::ffi::{OsStrExt as _, OsStringExt as _};
use std::path::{Path, PathBuf};

use rustc_hash::FxHasher;

const NOOP_MAGIC: [u8; 8] = *b"HSTNOP01";
const HEADER_LEN: usize = 32;

/// Returns whether the CLI no-op fast path is active for this `build` invocation.
pub fn active(stats: Option<&Path>) -> bool {
    super::enabled()
        && stats.is_none()
        && !crate::image::verify_stencil()
        && std::env::var_os("HASTE_NO_NOOP").is_none_or(|v| v != "1")
}

fn canonical_root(root: &Path) -> PathBuf {
    std::fs::canonicalize(root).unwrap_or_else(|_| root.to_owned())
}

fn abs_out(out: Option<&Path>) -> Option<PathBuf> {
    out.map(|p| std::path::absolute(p).unwrap_or_else(|_| p.to_owned()))
}

fn stamp_path(root: &Path, out: Option<&Path>) -> PathBuf {
    let mut hasher = FxHasher::default();
    NOOP_MAGIC.hash(&mut hasher);
    root.hash(&mut hasher);
    out.hash(&mut hasher);
    (crate::profile() == crate::outline::tree::Profile::Release).hash(&mut hasher);
    crate::graph::store::home().join("noop").join(format!("{:016x}.stamp", hasher.finish()))
}

fn compute_stamp(root: &Path, out: Option<&Path>, paths: &[PathBuf], env_names: &[String]) -> Option<u64> {
    let target = crate::host_target();
    let mut hasher = FxHasher::default();
    NOOP_MAGIC.hash(&mut hasher);
    super::target_hash(&target).hash(&mut hasher);
    (crate::profile() == crate::outline::tree::Profile::Release).hash(&mut hasher);
    super::enabled().hash(&mut hasher);
    crate::prep::prepared().hash(&mut hasher);
    std::env::var_os("HASTE_LIBRARY").hash(&mut hasher);
    std::env::var_os("CARGO_HOME").hash(&mut hasher);
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
    env_names.len().hash(&mut hasher);
    for name in env_names {
        (name, std::env::var_os(name)).hash(&mut hasher);
    }
    Some(hasher.finish())
}

fn read_bytes<'a>(data: &'a [u8], pos: &mut usize, len: usize) -> Option<&'a [u8]> {
    let end = pos.checked_add(len)?;
    let slice = data.get(*pos..end)?;
    *pos = end;
    Some(slice)
}

fn read_len_u32(data: &[u8], pos: &mut usize) -> Option<usize> {
    let slice = read_bytes(data, pos, 4)?;
    let mut buf = [0u8; 4];
    buf.copy_from_slice(slice);
    Some(u32::from_le_bytes(buf) as usize)
}

fn decode_stamp_file(data: &[u8]) -> Option<(u64, usize, Vec<PathBuf>, Vec<String>)> {
    if data.len() < HEADER_LEN || data[0..8] != NOOP_MAGIC {
        return None;
    }
    let stamp = super::read_u64(data, 8);
    let reached = usize::try_from(super::read_u64(data, 16)).ok()?;
    if reached == 0 {
        return None;
    }
    let path_count = super::read_u32(data, 24) as usize;
    let env_count = super::read_u32(data, 28) as usize;
    let mut pos = HEADER_LEN;
    let mut paths = Vec::with_capacity(path_count.min(4096));
    for _ in 0..path_count {
        let len = read_len_u32(data, &mut pos)?;
        let bytes = read_bytes(data, &mut pos, len)?;
        paths.push(PathBuf::from(OsString::from_vec(bytes.to_vec())));
    }
    let mut envs = Vec::with_capacity(env_count.min(1024));
    for _ in 0..env_count {
        let len = read_len_u32(data, &mut pos)?;
        let bytes = read_bytes(data, &mut pos, len)?;
        envs.push(String::from_utf8(bytes.to_vec()).ok()?);
    }
    (pos == data.len()).then_some((stamp, reached, paths, envs))
}

fn encode_stamp_file(stamp: u64, reached: usize, paths: &[PathBuf], envs: &[String]) -> Option<Vec<u8>> {
    let path_count = u32::try_from(paths.len()).ok()?;
    let env_count = u32::try_from(envs.len()).ok()?;
    let mut out = Vec::with_capacity(HEADER_LEN + paths.len() * 48 + envs.len() * 16);
    out.extend_from_slice(&NOOP_MAGIC);
    out.extend_from_slice(&stamp.to_le_bytes());
    out.extend_from_slice(&(reached as u64).to_le_bytes());
    out.extend_from_slice(&path_count.to_le_bytes());
    out.extend_from_slice(&env_count.to_le_bytes());
    for path in paths {
        let bytes = path.as_os_str().as_bytes();
        let len = u32::try_from(bytes.len()).ok()?;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(bytes);
    }
    for env in envs {
        let bytes = env.as_bytes();
        let len = u32::try_from(bytes.len()).ok()?;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(bytes);
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
    let (stored_stamp, reached, paths, envs) = decode_stamp_file(&data)?;
    let current = compute_stamp(&croot, aout.as_deref(), &paths, &envs)?;
    (current == stored_stamp).then_some(reached)
}

/// Records the workspace no-op stamp after a `0 failed` build succeeds.
pub fn record(root: &Path, workspace: &crate::graph::Workspace, build_paths: &[PathBuf], out: Option<&Path>, reached: usize) {
    if reached == 0 {
        return;
    }
    let croot = canonical_root(root);
    let aout = abs_out(out);
    let mut paths = crate::graph::manifest_paths(workspace, &croot);
    paths.extend(build_paths.iter().cloned());
    let (watched_files, watched_vars) = crate::graph::store::watched_stamps(workspace);
    paths.extend(watched_files.into_iter().map(|(path, _, _)| path));
    paths.sort();
    paths.dedup();
    let mut envs: Vec<String> = watched_vars.into_iter().map(|(name, _)| name).collect();
    envs.sort();
    envs.dedup();
    let Some(stamp) = compute_stamp(&croot, aout.as_deref(), &paths, &envs) else { return };
    if let Some(encoded) = encode_stamp_file(stamp, reached, &paths, &envs) {
        drop(crate::graph::store::write_new_bytes(&stamp_path(&croot, aout.as_deref()), &encoded));
    }
}

/// Removes any saved workspace no-op stamp after a failed build.
pub fn invalidate(root: &Path, out: Option<&Path>) {
    let croot = canonical_root(root);
    let aout = abs_out(out);
    drop(std::fs::remove_file(stamp_path(&croot, aout.as_deref())));
}

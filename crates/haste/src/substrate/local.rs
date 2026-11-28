//! Per-crate and per-file disk verdict caches for unchanged local crates in one-shot `haste check`.

use std::hash::{Hash as _, Hasher as _};
use std::path::{Path, PathBuf};

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet, FxHasher};

use crate::graph::Workspace;
use crate::graph::store::{FileStamp, file_stamp, write_new_bytes};
use crate::outline::Outline;
use crate::outline::edit::hash_outside_bodies;
use crate::outline::tree::File;

use super::enabled;
use super::keys::{CrateHashes, hash_crate_env};
use super::local_index::file_stamp_matches;
use super::sysroot::prune_cache_pair;
use super::wire::{
    read_bytes, read_i128, read_path, read_u32, read_u64, read_utf8, write_len_bytes, write_path,
};

const LOCAL_MAGIC: [u8; 8] = *b"HSTLCHK2";
const FILES_MAGIC: [u8; 8] = *b"HSTLFIL2";
const CHECK_HEADER_LEN: usize = 24;
const FILES_HEADER_LEN: usize = 28;
const MAX_KEPT_ENTRIES: usize = 512;

type WatchEnvs = Vec<(String, Option<u64>)>;
type WatchData = (Vec<(PathBuf, FileStamp)>, WatchEnvs);

/// Returns whether the per-crate local disk check cache is enabled.
pub fn active() -> bool {
    enabled() && !crate::image::verify_stencil() && !crate::env_flag("HASTE_NO_LOCAL_CACHE")
}

/// Dirty local crate scheduled for checking, with optional file-level scope.
pub struct DirtyCrate {
    pub krate: usize,
    pub key: u64,
    pub iface_key: Option<u64>,
    pub dirty_files: Option<Vec<usize>>,
    carried: WatchData,
}

/// Partition of checked local crates into cache hits and dirty crates.
pub struct Partition {
    pub hits: Vec<usize>,
    pub dirty: Vec<DirtyCrate>,
}

fn cache_dir() -> PathBuf {
    crate::graph::store::home().join("local")
}

fn entry_path(key: u64) -> PathBuf {
    cache_dir().join(format!("{key:016x}.check"))
}

fn files_path(iface_key: u64) -> PathBuf {
    cache_dir().join(format!("{iface_key:016x}.files"))
}

fn global_check_key(workspace: &Workspace) -> u64 {
    let mut hasher = FxHasher::default();
    LOCAL_MAGIC.hash(&mut hasher);
    crate::config::Config::current().hash_cache_key(&mut hasher);
    workspace.rustflags.hash(&mut hasher);
    hasher.finish()
}

fn crate_key(hashes: &CrateHashes, krate: usize, global_key: u64) -> u64 {
    let mut hasher = FxHasher::default();
    global_key.hash(&mut hasher);
    hashes.get(krate).hash(&mut hasher);
    hasher.finish()
}

fn rel_path<'a>(outline: &'a Outline, krate: usize, file: &'a File) -> &'a Path {
    let base_dir = outline
        .roots
        .get(krate)
        .and_then(|r| r.path.parent())
        .unwrap_or(Path::new(""));
    file.source
        .path
        .strip_prefix(base_dir)
        .unwrap_or(&file.source.path)
}

fn hash_env_val(name: &str, val: &str) -> u64 {
    crate::graph::store::hash_env_val(&LOCAL_MAGIC, name, val)
}

fn file_hashes(outline: &Outline, krate: usize, file: &File) -> (u64, u64) {
    let mut rh = FxHasher::default();
    rel_path(outline, krate, file).hash(&mut rh);
    (
        rh.finish(),
        crate::graph::store::hash_str(&file.source.text),
    )
}

fn is_local_proc_macro_dep(workspace: &Workspace, outline: &Outline, krate: usize) -> bool {
    outline.dependents(krate).into_iter().any(|dep| {
        outline.roots[dep].proc_macro || outline.package(workspace, dep).manifest.proc_macro
    })
}

fn can_file_cache(workspace: &Workspace, outline: &Outline, krate: usize) -> bool {
    !outline.roots[krate].proc_macro
        && !is_local_proc_macro_dep(workspace, outline, krate)
        && outline.crates[krate]
            .files
            .iter()
            .all(|f| f.expanded || f.source.marks.is_empty())
}

fn crate_iface_key(
    outline: &Outline,
    hashes: &CrateHashes,
    krate: usize,
    (toolchain_key, target_key): (u64, u64),
    global_key: u64,
) -> u64 {
    let mut hasher = FxHasher::default();
    FILES_MAGIC.hash(&mut hasher);
    global_key.hash(&mut hasher);
    hash_crate_env(outline, krate, toolchain_key, target_key, &mut hasher);
    let mut deps: Vec<(&str, Option<u64>)> = outline.deps[krate]
        .iter()
        .map(|(name, dep)| (name.as_str(), hashes.get(*dep)))
        .collect();
    deps.sort_unstable_by(|a, b| a.0.cmp(b.0));
    deps.hash(&mut hasher);
    let tree = &outline.crates[krate];
    let mut consumed: Vec<(u32, u32)> = tree.consumed.iter().copied().collect();
    consumed.sort_unstable();
    consumed.hash(&mut hasher);
    (&tree.errors, &tree.missing).hash(&mut hasher);
    let mut plain: Vec<(&Path, u64)> = Vec::new();
    let mut expanded: Vec<(&Path, u64, u64)> = Vec::new();
    for file in &tree.files {
        let rel = rel_path(outline, krate, file);
        if file.expanded {
            expanded.push((
                rel,
                file.expand_origin,
                crate::graph::store::hash_str(&file.source.text),
            ));
        } else {
            let mut fh = FxHasher::default();
            hash_outside_bodies(file, &mut fh);
            plain.push((rel, fh.finish()));
        }
    }
    plain.sort_unstable_by(|a, b| a.0.cmp(b.0));
    plain.hash(&mut hasher);
    expanded.sort_unstable();
    expanded.hash(&mut hasher);
    hasher.finish()
}

fn local_closure(outline: &Outline, start: usize) -> Vec<usize> {
    let mut seen = vec![false; outline.crates.len()];
    let mut stack = vec![start];
    let mut out = Vec::new();
    while let Some(k) = stack.pop() {
        if std::mem::replace(&mut seen[k], true) || outline.is_fixed(k as u32) {
            continue;
        }
        out.push(k);
        for &(_, dep) in &outline.deps[k] {
            stack.push(dep);
        }
    }
    out
}

fn collect_included_stamps(
    outline: &Outline,
    closure: &[usize],
    carried: &[(PathBuf, FileStamp)],
) -> Vec<(PathBuf, FileStamp)> {
    let mut paths: Vec<PathBuf> = closure
        .iter()
        .flat_map(|&k| outline.included(k).into_iter().map(|(path, _)| path))
        .chain(carried.iter().map(|(path, _)| path.clone()))
        .collect();
    paths.sort();
    paths.dedup();
    paths
        .into_iter()
        .map(|path| {
            let stamp = file_stamp(&path);
            (path, stamp)
        })
        .collect()
}

fn collect_env_reads(
    outline: &Outline,
    closure: &[usize],
    carried: &[(String, Option<u64>)],
) -> WatchEnvs {
    let mut merged: HashMap<String, Option<u64>> = HashMap::default();
    for (name, val_hash) in carried {
        merged.insert(name.clone(), *val_hash);
    }
    for &k in closure {
        let reads = crate::sync::lock(&outline.crates[k].env.env_reads);
        for (name, val) in reads.iter() {
            let hashed = val.as_deref().map(|v| hash_env_val(name, v));
            merged.insert(name.clone(), hashed);
        }
    }
    let mut out: WatchEnvs = merged.into_iter().collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

fn write_watch_data(
    out: &mut Vec<u8>,
    included: &[(PathBuf, FileStamp)],
    envs: &[(String, Option<u64>)],
) -> Option<()> {
    for (path, stamp) in included {
        write_path(out, path)?;
        match stamp {
            Some((len, nanos)) => {
                out.push(1);
                out.extend_from_slice(&len.to_le_bytes());
                out.extend_from_slice(&nanos.to_le_bytes());
            }
            None => out.push(0),
        }
    }
    for (name, val) in envs {
        write_len_bytes(out, name.as_bytes())?;
        match val {
            Some(hash) => {
                out.push(1);
                out.extend_from_slice(&hash.to_le_bytes());
            }
            None => out.push(0),
        }
    }
    Some(())
}

fn read_watch_data(
    data: &[u8],
    pos: &mut usize,
    inc_count: usize,
    env_count: usize,
) -> Option<WatchData> {
    let mut included = Vec::with_capacity(inc_count.min(4096));
    for _ in 0..inc_count {
        let path = read_path(data, pos)?;
        let present = *read_bytes(data, pos, 1)?.first()?;
        let stamp = match present {
            0 => None,
            1 => {
                let raw = read_bytes(data, pos, 24)?;
                Some((read_u64(raw, 0), read_i128(raw, 8)))
            }
            _ => return None,
        };
        included.push((path, stamp));
    }
    let mut envs = Vec::with_capacity(env_count.min(1024));
    for _ in 0..env_count {
        let name = read_utf8(data, pos)?;
        let present = *read_bytes(data, pos, 1)?.first()?;
        let val = match present {
            0 => None,
            1 => {
                let raw = read_bytes(data, pos, 8)?;
                Some(read_u64(raw, 0))
            }
            _ => return None,
        };
        envs.push((name, val));
    }
    Some((included, envs))
}

fn validate_watch(included: &[(PathBuf, FileStamp)], envs: &[(String, Option<u64>)]) -> bool {
    let mut buf = Vec::new();
    included
        .iter()
        .all(|(inc, expected)| file_stamp_matches(inc, *expected, &mut buf))
        && envs.iter().all(|(name, expected)| {
            crate::process_env_var(name)
                .as_deref()
                .map(|v| hash_env_val(name, v))
                == *expected
        })
}

fn encode_entry(
    key: u64,
    included: &[(PathBuf, FileStamp)],
    envs: &[(String, Option<u64>)],
) -> Option<Vec<u8>> {
    let inc_count = u32::try_from(included.len()).ok()?;
    let env_count = u32::try_from(envs.len()).ok()?;
    let mut out = Vec::with_capacity(CHECK_HEADER_LEN + included.len() * 72 + envs.len() * 32);
    out.extend_from_slice(&LOCAL_MAGIC);
    out.extend_from_slice(&key.to_le_bytes());
    out.extend_from_slice(&inc_count.to_le_bytes());
    out.extend_from_slice(&env_count.to_le_bytes());
    write_watch_data(&mut out, included, envs)?;
    Some(out)
}

fn decode_entry(data: &[u8], expected_key: u64) -> Option<WatchData> {
    if data.len() < CHECK_HEADER_LEN
        || data[0..8] != LOCAL_MAGIC
        || read_u64(data, 8) != expected_key
    {
        return None;
    }
    let inc_count = read_u32(data, 16) as usize;
    let env_count = read_u32(data, 20) as usize;
    let mut pos = CHECK_HEADER_LEN;
    let watch = read_watch_data(data, &mut pos, inc_count, env_count)?;
    (pos == data.len()).then_some(watch)
}

fn encode_files_entry(
    iface_key: u64,
    files: &[(u64, u64)],
    included: &[(PathBuf, FileStamp)],
    envs: &[(String, Option<u64>)],
) -> Option<Vec<u8>> {
    let file_count = u32::try_from(files.len()).ok()?;
    let inc_count = u32::try_from(included.len()).ok()?;
    let env_count = u32::try_from(envs.len()).ok()?;
    let mut out = Vec::with_capacity(
        FILES_HEADER_LEN + files.len() * 16 + included.len() * 72 + envs.len() * 32,
    );
    out.extend_from_slice(&FILES_MAGIC);
    out.extend_from_slice(&iface_key.to_le_bytes());
    out.extend_from_slice(&file_count.to_le_bytes());
    out.extend_from_slice(&inc_count.to_le_bytes());
    out.extend_from_slice(&env_count.to_le_bytes());
    for &(rel_hash, text_hash) in files {
        out.extend_from_slice(&rel_hash.to_le_bytes());
        out.extend_from_slice(&text_hash.to_le_bytes());
    }
    write_watch_data(&mut out, included, envs)?;
    Some(out)
}

fn decode_files_entry(data: &[u8], expected_key: u64) -> Option<(HashMap<u64, u64>, WatchData)> {
    if data.len() < FILES_HEADER_LEN
        || data[0..8] != FILES_MAGIC
        || read_u64(data, 8) != expected_key
    {
        return None;
    }
    let file_count = read_u32(data, 16) as usize;
    let inc_count = read_u32(data, 20) as usize;
    let env_count = read_u32(data, 24) as usize;
    let mut pos = FILES_HEADER_LEN;
    let mut files = HashMap::default();
    files.reserve(file_count.min(4096));
    for _ in 0..file_count {
        let raw = read_bytes(data, &mut pos, 16)?;
        files.insert(read_u64(raw, 0), read_u64(raw, 8));
    }
    let watch = read_watch_data(data, &mut pos, inc_count, env_count)?;
    (pos == data.len()).then_some((files, watch))
}

fn try_hit_at(path: &Path, key: u64) -> Option<WatchData> {
    let data = std::fs::read(path).ok()?;
    let (included, envs) = decode_entry(&data, key)?;
    validate_watch(&included, &envs).then_some((included, envs))
}

fn try_hit(key: u64) -> Option<WatchData> {
    try_hit_at(&entry_path(key), key)
}

fn try_hit_files_at(
    path: &Path,
    iface_key: u64,
    outline: &Outline,
    krate: usize,
) -> Option<(Vec<usize>, WatchData)> {
    let data = std::fs::read(path).ok()?;
    let (cached, (included, envs)) = decode_files_entry(&data, iface_key)?;
    if !validate_watch(&included, &envs) {
        return None;
    }
    let dirty: Vec<usize> = outline.crates[krate]
        .files
        .iter()
        .enumerate()
        .filter(|(_, f)| !f.expanded)
        .filter(|(_, f)| {
            let (rel_hash, text_hash) = file_hashes(outline, krate, f);
            cached.get(&rel_hash) != Some(&text_hash)
        })
        .map(|(idx, _)| idx)
        .collect();
    Some((dirty, (included, envs)))
}

fn has_partial_local(outline: &Outline, closure: &[usize]) -> bool {
    closure.iter().any(|&k| outline.crates[k].partial())
}

fn merge_watch_into(dst: &mut WatchData, src: &WatchData) {
    dst.0.extend(src.0.iter().cloned());
    dst.1.extend(src.1.iter().cloned());
}

/// Splits `crates` into cached-passing crates and dirty crates under `(toolchain_key, target_key)`.
pub fn partition(
    workspace: &Workspace,
    outline: &Outline,
    crates: &[usize],
    (toolchain_key, target_key): (u64, u64),
) -> Partition {
    let hashes = CrateHashes::compute(outline, toolchain_key, target_key);
    let global_key = global_check_key(workspace);
    let mut hits = Vec::new();
    let mut is_dirty = vec![false; outline.crates.len()];
    let mut watches_by_crate: HashMap<usize, WatchData> = HashMap::default();
    let mut pending_dirty = Vec::new();
    let probe = |&krate: &usize| {
        let key = crate_key(&hashes, krate, global_key);
        let closure = local_closure(outline, krate);
        if has_partial_local(outline, &closure) {
            return (krate, key, closure, None, None, None);
        }
        if let Some(watch) = try_hit(key) {
            return (krate, key, closure, None, None, Some(watch));
        }
        let iface_key = can_file_cache(workspace, outline, krate).then(|| {
            crate_iface_key(
                outline,
                &hashes,
                krate,
                (toolchain_key, target_key),
                global_key,
            )
        });
        if let Some(ik) = iface_key
            && let Some((dirty_files, watch)) =
                try_hit_files_at(&files_path(ik), ik, outline, krate)
        {
            return (
                krate,
                key,
                closure,
                Some(ik),
                Some(dirty_files),
                Some(watch),
            );
        }
        (krate, key, closure, iface_key, None, None)
    };
    let probed: Vec<_> = if crates.len() >= 4 && crate::jobs() > 1 {
        crate::sync::parallel_map(crates.to_vec(), |k| probe(&k))
    } else {
        crates.iter().map(probe).collect()
    };
    for (krate, key, closure, iface_key, dirty_files, watch) in probed {
        match (dirty_files, watch) {
            (None, Some(watch)) => {
                hits.push(krate);
                watches_by_crate.insert(krate, watch);
            }
            (Some(dirty_files), Some(watch)) => {
                watches_by_crate.insert(krate, watch);
                if dirty_files.is_empty() {
                    hits.push(krate);
                } else {
                    is_dirty[krate] = true;
                    pending_dirty.push((krate, key, iface_key, Some(dirty_files), closure));
                }
            }
            _ => {
                is_dirty[krate] = true;
                pending_dirty.push((krate, key, iface_key, None, closure));
            }
        }
    }
    let mut unhit_deps = HashSet::default();
    let mut dirty = Vec::with_capacity(pending_dirty.len());
    for (krate, key, iface_key, dirty_files, closure) in pending_dirty {
        let mut carried = (Vec::new(), Vec::new());
        for dep in closure {
            if let Some(watch) = watches_by_crate.get(&dep) {
                merge_watch_into(&mut carried, watch);
            } else if !is_dirty[dep] && !unhit_deps.contains(&dep) {
                if let Some(watch) = try_hit(crate_key(&hashes, dep, global_key)) {
                    merge_watch_into(&mut carried, &watch);
                    watches_by_crate.insert(dep, watch);
                } else {
                    unhit_deps.insert(dep);
                }
            }
        }
        dirty.push(DirtyCrate {
            krate,
            key,
            iface_key,
            dirty_files,
            carried,
        });
    }
    Partition { hits, dirty }
}

fn closure_has_failure(
    outline: &Outline,
    krate: usize,
    failed_crates: &HashSet<u32>,
    memo: &mut [Option<bool>],
) -> bool {
    if let Some(res) = memo[krate] {
        return res;
    }
    memo[krate] = Some(false);
    let failed = failed_crates.contains(&(krate as u32))
        || outline.deps[krate]
            .iter()
            .any(|&(_, dep)| closure_has_failure(outline, dep, failed_crates, memo));
    memo[krate] = Some(failed);
    failed
}

fn plain_file_hashes(outline: &Outline, krate: usize) -> Vec<(u64, u64)> {
    let mut out: Vec<(u64, u64)> = outline.crates[krate]
        .files
        .iter()
        .filter(|f| !f.expanded)
        .map(|f| file_hashes(outline, krate, f))
        .collect();
    out.sort_unstable_by_key(|&(rel, _)| rel);
    out
}

/// Saves passing `.check` and `.files` entries for every crate in `dirty` whose dependency closure has no errors in `failed_crates`.
pub fn record(outline: &Outline, dirty: &[DirtyCrate], failed_crates: &HashSet<u32>) {
    let mut memo = vec![None; outline.crates.len()];
    let mut last_check = None;
    let mut last_files = None;
    for entry in dirty {
        if closure_has_failure(outline, entry.krate, failed_crates, &mut memo) {
            continue;
        }
        let closure = local_closure(outline, entry.krate);
        if has_partial_local(outline, &closure) {
            continue;
        }
        let included = collect_included_stamps(outline, &closure, &entry.carried.0);
        let envs = collect_env_reads(outline, &closure, &entry.carried.1);
        if let Some(bytes) = encode_entry(entry.key, &included, &envs) {
            let path = entry_path(entry.key);
            if write_new_bytes(&path, &bytes).is_ok() {
                last_check = Some(path);
            }
        }
        if let Some(ik) = entry.iface_key {
            let files = plain_file_hashes(outline, entry.krate);
            if let Some(bytes) = encode_files_entry(ik, &files, &included, &envs) {
                let path = files_path(ik);
                if write_new_bytes(&path, &bytes).is_ok() {
                    last_files = Some(path);
                }
            }
        }
    }
    prune_cache_pair(
        &cache_dir(),
        last_check.as_deref(),
        "check",
        last_files.as_deref(),
        "files",
        MAX_KEPT_ENTRIES - 1,
    );
}

fn watch_matches_workspace(
    data: &[u8],
    mut pos: usize,
    inc_count: usize,
    env_count: usize,
    roots: &[PathBuf],
) -> bool {
    let Some((included, _)) = read_watch_data(data, &mut pos, inc_count, env_count) else {
        return false;
    };
    included
        .iter()
        .any(|(path, _)| super::noop::path_in_workspace(path, roots))
}

/// Returns whether an encoded `.check` entry watches any path inside `roots`.
pub(crate) fn check_matches_workspace(data: &[u8], roots: &[PathBuf]) -> bool {
    if data.len() < CHECK_HEADER_LEN || data[0..8] != LOCAL_MAGIC {
        return false;
    }
    let inc_count = read_u32(data, 16) as usize;
    let env_count = read_u32(data, 20) as usize;
    watch_matches_workspace(data, CHECK_HEADER_LEN, inc_count, env_count, roots)
}

/// Returns whether an encoded `.files` entry watches any path inside `roots`.
pub(crate) fn files_matches_workspace(data: &[u8], roots: &[PathBuf]) -> bool {
    if data.len() < FILES_HEADER_LEN || data[0..8] != FILES_MAGIC {
        return false;
    }
    let file_count = read_u32(data, 16) as usize;
    let inc_count = read_u32(data, 20) as usize;
    let env_count = read_u32(data, 24) as usize;
    let Some(pos) = FILES_HEADER_LEN.checked_add(file_count.saturating_mul(16)) else {
        return false;
    };
    watch_matches_workspace(data, pos, inc_count, env_count, roots)
}

#[cfg(test)]
pub(crate) fn encode_test_check(included: &Path) -> Vec<u8> {
    encode_entry(0, &[(included.to_path_buf(), None)], &[]).unwrap_or_default()
}

#[cfg(test)]
pub(crate) fn encode_test_files(included: &Path) -> Vec<u8> {
    encode_files_entry(0, &[], &[(included.to_path_buf(), None)], &[]).unwrap_or_default()
}

#[cfg(test)]
mod tests;

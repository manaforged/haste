//! Per-crate and per-file disk verdict caches for unchanged local crates in one-shot `haste check`.

use std::hash::{Hash as _, Hasher as _};
use std::path::{Path, PathBuf};

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet, FxHasher};

use crate::graph::Workspace;
use crate::graph::store::{FileStamp, file_stamp, write_new_bytes};
use crate::outline::Outline;
use crate::outline::edit::hash_outside_bodies;
use crate::outline::tree::File;

use super::keys::{CrateHashes, hash_crate_env};
use super::wire::{
    read_bytes, read_i128, read_path, read_u32, read_u64, read_utf8, write_len_bytes, write_path,
};
use super::{enabled, prune_stale};

const LOCAL_MAGIC: [u8; 8] = *b"HSTLCHK1";
const FILES_MAGIC: [u8; 8] = *b"HSTLFIL1";
const CHECK_HEADER_LEN: usize = 24;
const FILES_HEADER_LEN: usize = 28;
const MAX_KEPT_ENTRIES: usize = 512;

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

fn hash_str(s: &str) -> u64 {
    let mut hasher = FxHasher::default();
    s.hash(&mut hasher);
    hasher.finish()
}

fn file_hashes(outline: &Outline, krate: usize, file: &File) -> (u64, u64) {
    let mut rh = FxHasher::default();
    rel_path(outline, krate, file).hash(&mut rh);
    (rh.finish(), hash_str(&file.source.text))
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
            expanded.push((rel, file.expand_origin, hash_str(&file.source.text)));
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

fn collect_included_stamps(outline: &Outline, closure: &[usize]) -> Vec<(PathBuf, FileStamp)> {
    let mut paths: Vec<PathBuf> = closure
        .iter()
        .flat_map(|&k| outline.included(k).into_iter().map(|(path, _)| path))
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

fn collect_env_reads(outline: &Outline, closure: &[usize]) -> Vec<(String, Option<String>)> {
    let mut merged: HashMap<String, Option<String>> = HashMap::default();
    for &k in closure {
        let reads = crate::sync::lock(&outline.crates[k].env.env_reads);
        for (name, val) in reads.iter() {
            merged.entry(name.clone()).or_insert_with(|| val.clone());
        }
    }
    let mut out: Vec<(String, Option<String>)> = merged.into_iter().collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

fn write_watch_data(
    out: &mut Vec<u8>,
    included: &[(PathBuf, FileStamp)],
    envs: &[(String, Option<String>)],
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
            Some(v) => {
                out.push(1);
                write_len_bytes(out, v.as_bytes())?;
            }
            None => out.push(0),
        }
    }
    Some(())
}

type WatchData = (Vec<(PathBuf, FileStamp)>, Vec<(String, Option<String>)>);

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
            1 => Some(read_utf8(data, pos)?),
            _ => return None,
        };
        envs.push((name, val));
    }
    Some((included, envs))
}

fn validate_watch(included: &[(PathBuf, FileStamp)], envs: &[(String, Option<String>)]) -> bool {
    included
        .iter()
        .all(|(inc, expected)| file_stamp(inc) == *expected)
        && envs
            .iter()
            .all(|(name, expected)| crate::process_env_var(name) == *expected)
}

fn encode_entry(
    key: u64,
    included: &[(PathBuf, FileStamp)],
    envs: &[(String, Option<String>)],
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
    envs: &[(String, Option<String>)],
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

fn try_hit_at(path: &Path, key: u64) -> bool {
    let Ok(data) = std::fs::read(path) else {
        return false;
    };
    let Some((included, envs)) = decode_entry(&data, key) else {
        return false;
    };
    validate_watch(&included, &envs)
}

fn try_hit(key: u64) -> bool {
    try_hit_at(&entry_path(key), key)
}

fn try_hit_files_at(
    path: &Path,
    iface_key: u64,
    outline: &Outline,
    krate: usize,
) -> Option<Vec<usize>> {
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
    Some(dirty)
}

fn has_partial_local(outline: &Outline, closure: &[usize]) -> bool {
    closure.iter().any(|&k| outline.crates[k].partial())
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
    let mut dirty = Vec::new();
    for &krate in crates {
        let key = crate_key(&hashes, krate, global_key);
        let closure = local_closure(outline, krate);
        if has_partial_local(outline, &closure) {
            dirty.push(DirtyCrate {
                krate,
                key,
                iface_key: None,
                dirty_files: None,
            });
            continue;
        }
        if try_hit(key) {
            hits.push(krate);
            continue;
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
            && let Some(dirty_files) = try_hit_files_at(&files_path(ik), ik, outline, krate)
        {
            if dirty_files.is_empty() {
                hits.push(krate);
            } else {
                dirty.push(DirtyCrate {
                    krate,
                    key,
                    iface_key: Some(ik),
                    dirty_files: Some(dirty_files),
                });
            }
            continue;
        }
        dirty.push(DirtyCrate {
            krate,
            key,
            iface_key,
            dirty_files: None,
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
        let included = collect_included_stamps(outline, &closure);
        let envs = collect_env_reads(outline, &closure);
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
    let dir = cache_dir();
    if let Some(keep) = last_check {
        prune_stale(&dir, &keep, MAX_KEPT_ENTRIES - 1, |p: &Path| {
            p.extension().and_then(|e| e.to_str()) == Some("check")
        });
    }
    if let Some(keep) = last_files {
        prune_stale(&dir, &keep, MAX_KEPT_ENTRIES - 1, |p: &Path| {
            p.extension().and_then(|e| e.to_str()) == Some("files")
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_round_trip_and_key_mismatch() {
        let included = vec![
            (PathBuf::from("/tmp/a.txt"), Some((12, 345_678))),
            (PathBuf::from("/tmp/missing.txt"), None),
        ];
        let envs = vec![
            ("VAR_A".to_owned(), Some("val".to_owned())),
            ("VAR_B".to_owned(), None),
        ];
        let bytes = encode_entry(0xDEAD_BEEF, &included, &envs).expect("encode");
        assert!(decode_entry(&bytes, 0xCAFE_BABE).is_none());
        let (dec_inc, dec_env) = decode_entry(&bytes, 0xDEAD_BEEF).expect("decode");
        assert_eq!(dec_inc, included);
        assert_eq!(dec_env, envs);
    }

    #[test]
    fn files_entry_round_trip_and_key_mismatch() {
        let files = vec![(11, 22), (33, 44)];
        let included = vec![(PathBuf::from("/tmp/a.txt"), Some((1, 2)))];
        let envs = vec![("FOO".to_owned(), Some("bar".to_owned()))];
        let bytes = encode_files_entry(0xABCD_1234, &files, &included, &envs).expect("encode");
        assert!(decode_files_entry(&bytes, 0x9999_0000).is_none());
        let (dec_files, (dec_inc, dec_env)) =
            decode_files_entry(&bytes, 0xABCD_1234).expect("decode");
        assert_eq!(dec_files.get(&11), Some(&22));
        assert_eq!(dec_files.get(&33), Some(&44));
        assert_eq!(dec_inc, included);
        assert_eq!(dec_env, envs);
    }

    #[test]
    fn try_hit_detects_env_and_included_file_changes() {
        let tmp = std::env::temp_dir().join(format!("haste-local-test-{}", std::process::id()));
        drop(std::fs::remove_dir_all(&tmp));
        std::fs::create_dir_all(&tmp).expect("mkdir");
        let inc_file = tmp.join("data.txt");
        std::fs::write(&inc_file, "hello").expect("write");
        let stamp = file_stamp(&inc_file);
        let entry_id = 0x1111_2222_3333_4444_u64;
        let cache_file = tmp.join(format!("{entry_id:016x}.check"));
        let bytes = encode_entry(
            entry_id,
            &[(inc_file.clone(), stamp)],
            &[("HASTE_TEST_LOCAL_ENV".to_owned(), Some("v1".to_owned()))],
        )
        .expect("encode");
        write_new_bytes(&cache_file, &bytes).expect("write cache");
        let env_v1 = {
            let mut m = rustc_hash::FxHashMap::default();
            m.insert("HASTE_TEST_LOCAL_ENV".to_owned(), "v1".to_owned());
            std::sync::Arc::new(m)
        };
        let env_v2 = {
            let mut m = rustc_hash::FxHashMap::default();
            m.insert("HASTE_TEST_LOCAL_ENV".to_owned(), "v2".to_owned());
            std::sync::Arc::new(m)
        };
        assert!(crate::with_client_env(Some(env_v1.clone()), || {
            try_hit_at(&cache_file, entry_id)
        }));
        assert!(!crate::with_client_env(Some(env_v2), || {
            try_hit_at(&cache_file, entry_id)
        }));
        std::fs::write(&inc_file, "hello world!").expect("edit inc");
        assert!(!crate::with_client_env(Some(env_v1), || {
            try_hit_at(&cache_file, entry_id)
        }));
        drop(std::fs::remove_dir_all(&tmp));
    }
}

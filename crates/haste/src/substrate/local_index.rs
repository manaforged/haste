//! Persistent disk cache for settled, unchanged local library crate outlines, macros, names, and impl indices.

mod meta;

#[cfg(test)]
pub(crate) use meta::encode_test_meta;
pub(super) use meta::file_stamp_matches;
pub(crate) use meta::matches_workspace;

use std::path::PathBuf;
use std::sync::Arc;

use rustc_hash::FxHashMap as HashMap;

use crate::graph::store;
use crate::graph::{Targets, Workspace};
use crate::macros::Compilation;
#[cfg(test)]
use crate::names::{Crate as NamesCrate, Symbols};
use crate::outline::Outline;
use crate::sema::impls::Index as ImplIndex;

use meta::{
    CrateRefs, LOCAL_INDEX_MAGIC, LOCAL_INDEX_VERSION, LocalMeta, MetaCheck, PrepCache, check_meta,
    collect_watched_files, compute_content_key, hash_env_reads, local_lib_topo_order,
    target_fingerprint,
};

use super::sysroot::{
    DecodeSpec, HEADER_LEN, PreparedWire, SnapshotSpec, collect_env_reads, encode_local_snapshot,
    prepare_wire, prune_cache_pair,
};
#[cfg(test)]
use super::sysroot::{apply_decoded_crates, decode_prepared_batch, decode_snapshot};

const KEPT_LOCAL_INDEX: usize = 64;
pub(crate) const LOCAL_SCOPE_BIT: u32 = 1 << 29;

/// Cached local library symbols, resolved crate scopes, module impl indices, and hit tracking.
#[cfg(test)]
pub(crate) struct LoadedLocal {
    pub symbols: Option<Symbols>,
    pub names: Vec<(usize, NamesCrate)>,
    pub heads: HashMap<u32, Arc<ImplIndex>>,
    pub hits: Vec<bool>,
    pub keys: Vec<Option<(u64, u64)>>,
}

/// Validated, memory-mapped local library snapshot ready for batched decoding once `Symbols` is available.
pub(crate) struct PreparedLocal {
    pub krate: usize,
    pub structural_key: u64,
    pub content_key: u64,
    pub needs_sysroot: bool,
    pub needs_deps: bool,
    pub meta_path: PathBuf,
    pub bin_path: PathBuf,
    pub included: Vec<(PathBuf, PathBuf)>,
    pub wire: PreparedWire,
}

/// Returns whether the frozen local library crate snapshot cache is enabled.
pub(crate) fn enabled() -> bool {
    super::enabled() && !crate::env_flag("HASTE_NO_LOCAL_INDEX")
}

fn cache_dir() -> PathBuf {
    store::home().join("local-index")
}

fn bin_path(content_key: u64) -> PathBuf {
    cache_dir().join(format!("{content_key:016x}.bin"))
}

fn meta_path(structural_key: u64) -> PathBuf {
    cache_dir().join(format!("{structural_key:016x}.meta"))
}

fn local_scope_bit(slot: usize) -> u32 {
    LOCAL_SCOPE_BIT | (((slot as u32) & 0xff) << 21)
}

struct PreparedCandidate {
    krate: usize,
    sk: u64,
    refs: CrateRefs,
    expected_deps: Vec<(u64, u64)>,
}

enum PrepareOutcome {
    Miss,
    CorruptMeta(PathBuf),
    CorruptBin(PathBuf, PathBuf),
    Hit {
        krate: usize,
        sk: u64,
        content_key: u64,
        needs_sysroot: bool,
        needs_deps: bool,
        meta_path: PathBuf,
        bin_path: PathBuf,
        included: Vec<(PathBuf, PathBuf)>,
        wire: PreparedWire,
    },
}

fn prepare_one(
    candidate: PreparedCandidate,
    workspace: &Workspace,
    outline: &Outline,
    parallel_stat: bool,
) -> PrepareOutcome {
    let PreparedCandidate {
        krate,
        sk,
        refs,
        expected_deps,
    } = candidate;
    let mpath = meta_path(sk);
    let Ok(meta_bytes) = std::fs::read(&mpath) else {
        return PrepareOutcome::Miss;
    };
    let meta = match check_meta(&meta_bytes, sk, &expected_deps, parallel_stat) {
        MetaCheck::Valid(meta) => meta,
        MetaCheck::Corrupt => return PrepareOutcome::CorruptMeta(mpath),
        MetaCheck::Stale => return PrepareOutcome::Miss,
    };
    let bpath = bin_path(meta.content_key);
    let Ok(backing) = super::Backing::open(&bpath, HEADER_LEN).map(Arc::new) else {
        return PrepareOutcome::CorruptMeta(mpath);
    };
    let cached = [krate];
    let spec = DecodeSpec {
        magic: LOCAL_INDEX_MAGIC,
        version: LOCAL_INDEX_VERSION,
        key: meta.content_key,
        cached_crates: &cached,
        ref_crates: &refs.ref_crates,
        scope_bit: 0,
    };
    match prepare_wire(backing, &spec, workspace, outline) {
        Ok(Some(wire)) => PrepareOutcome::Hit {
            krate,
            sk,
            content_key: meta.content_key,
            needs_sysroot: !refs.sysroot_deps.is_empty(),
            needs_deps: !refs.pinned_deps.is_empty(),
            meta_path: mpath,
            bin_path: bpath,
            included: meta.included,
            wire,
        },
        Ok(None) | Err(_) => PrepareOutcome::CorruptBin(mpath, bpath),
    }
}

/// Validates file stamps and memory-maps settled local library snapshots without requiring `Symbols`.
pub(crate) fn prepare(
    workspace: &Workspace,
    targets: &Targets,
    outline: &Outline,
    sys_key: u64,
    parallel_stat: bool,
) -> Vec<PreparedLocal> {
    if !enabled() {
        return Vec::new();
    }
    let mut prep = PrepCache::new(workspace, outline);
    let topo = local_lib_topo_order(outline, &prep.ranks);
    if topo.is_empty() {
        return Vec::new();
    }
    let th = target_fingerprint(targets);
    let n = outline.crates.len();
    let mut depths = vec![0usize; n];
    let mut waves: Vec<Vec<(usize, CrateRefs)>> = Vec::new();
    for krate in topo {
        let Some(refs) = prep.crate_refs(outline, krate) else {
            continue;
        };
        let depth = refs
            .local_deps
            .iter()
            .map(|&d| depths[d] + 1)
            .max()
            .unwrap_or(0);
        depths[krate] = depth;
        if waves.len() <= depth {
            waves.resize_with(depth + 1, Vec::new);
        }
        waves[depth].push((krate, refs));
    }
    let mut keys: Vec<Option<(u64, u64)>> = vec![None; n];
    let mut prepared = Vec::new();
    let mut slot = 0usize;
    for wave in waves {
        let mut candidates = Vec::with_capacity(wave.len());
        for (krate, refs) in wave {
            let Some(expected_deps) = refs
                .local_deps
                .iter()
                .map(|&d| keys[d])
                .collect::<Option<Vec<_>>>()
            else {
                continue;
            };
            let Some(sk) =
                prep.structural_key(workspace, targets, outline, krate, &refs, sys_key, th)
            else {
                continue;
            };
            candidates.push(PreparedCandidate {
                krate,
                sk,
                refs,
                expected_deps,
            });
        }
        if candidates.is_empty() {
            continue;
        }
        let outcomes = if candidates.len() >= 2 && crate::jobs() > 1 {
            crate::sync::parallel_map(candidates, |c| prepare_one(c, workspace, outline, false))
        } else {
            candidates
                .into_iter()
                .map(|c| prepare_one(c, workspace, outline, parallel_stat))
                .collect()
        };
        for outcome in outcomes {
            match outcome {
                PrepareOutcome::Miss => {}
                PrepareOutcome::CorruptMeta(mpath) => {
                    drop(std::fs::remove_file(&mpath));
                }
                PrepareOutcome::CorruptBin(mpath, bpath) => {
                    drop(std::fs::remove_file(&mpath));
                    drop(std::fs::remove_file(&bpath));
                }
                PrepareOutcome::Hit {
                    krate,
                    sk,
                    content_key,
                    needs_sysroot,
                    needs_deps,
                    meta_path,
                    bin_path,
                    included,
                    mut wire,
                } => {
                    wire.scope_bit = local_scope_bit(slot);
                    keys[krate] = Some((sk, content_key));
                    prepared.push(PreparedLocal {
                        krate,
                        structural_key: sk,
                        content_key,
                        needs_sysroot,
                        needs_deps,
                        meta_path,
                        bin_path,
                        included,
                        wire,
                    });
                    slot += 1;
                }
            }
        }
    }
    prepared
}

/// Filters `prepared` in topological order against `sysroot_hit` and `deps_hit`, keeping only crates whose dependencies all hit.
pub(crate) fn filter_prepared(
    outline: &Outline,
    prepared: Vec<PreparedLocal>,
    sysroot_hit: bool,
    deps_hit: bool,
) -> Vec<PreparedLocal> {
    let mut ready = vec![false; outline.crates.len()];
    let mut out = Vec::with_capacity(prepared.len());
    for item in prepared {
        if (item.needs_sysroot && !sysroot_hit)
            || (item.needs_deps && !deps_hit)
            || outline.deps[item.krate]
                .iter()
                .any(|&(_, dep)| !outline.is_fixed(dep as u32) && !ready[dep])
        {
            continue;
        }
        ready[item.krate] = true;
        out.push(item);
    }
    out
}

/// Attempts to load cached settled local library crates into `outline`, interning symbols into `existing_symbols` if present.
#[cfg(test)]
pub(crate) fn try_load(
    workspace: &Workspace,
    targets: &Targets,
    outline: &mut Outline,
    existing_symbols: Option<&Symbols>,
    sys_key: u64,
    sysroot_hit: bool,
    deps_hit: bool,
) -> LoadedLocal {
    let n = outline.crates.len();
    let mut loaded = LoadedLocal {
        symbols: None,
        names: Vec::new(),
        heads: HashMap::default(),
        hits: vec![false; n],
        keys: vec![None; n],
    };
    let prepared = filter_prepared(
        outline,
        prepare(workspace, targets, outline, sys_key, true),
        sysroot_hit,
        deps_hit,
    );
    if prepared.is_empty() {
        return loaded;
    }
    let mut iter = prepared.into_iter();
    if existing_symbols.is_none()
        && let Some(first) = iter.next()
        && let Some(refs) = PrepCache::new(workspace, outline).crate_refs(outline, first.krate)
    {
        let cached = [first.krate];
        let spec = DecodeSpec {
            magic: LOCAL_INDEX_MAGIC,
            version: LOCAL_INDEX_VERSION,
            key: first.content_key,
            cached_crates: &cached,
            ref_crates: &refs.ref_crates,
            scope_bit: first.wire.scope_bit,
        };
        match decode_snapshot(&first.wire.backing, &spec, None, workspace, outline) {
            Ok(Some((crates, owned_syms, names, heads, env_reads))) => {
                loaded.symbols = owned_syms;
                apply_decoded_crates(outline, crates, env_reads);
                outline.crates[first.krate].cached_included = Some(first.included);
                loaded.names.extend(names);
                loaded.heads.extend(heads);
                loaded.hits[first.krate] = true;
                loaded.keys[first.krate] = Some((first.structural_key, first.content_key));
            }
            Ok(None) | Err(_) => {
                drop(std::fs::remove_file(&first.meta_path));
                drop(std::fs::remove_file(&first.bin_path));
                return loaded;
            }
        }
    }
    let rest: Vec<PreparedLocal> = iter.collect();
    if rest.is_empty() {
        return loaded;
    }
    let Some(syms) = existing_symbols.or(loaded.symbols.as_ref()) else {
        return loaded;
    };
    let mut meta_info = Vec::with_capacity(rest.len());
    let mut wires = Vec::with_capacity(rest.len());
    for p in rest {
        meta_info.push((
            p.krate,
            p.structural_key,
            p.content_key,
            p.meta_path,
            p.bin_path,
            p.included,
        ));
        wires.push(p.wire);
    }
    match decode_prepared_batch(wires, syms) {
        Ok(decoded) => {
            for ((krate, sk, ck, _, _, included), (crates, _, names, heads, env_reads)) in
                meta_info.into_iter().zip(decoded)
            {
                apply_decoded_crates(outline, crates, env_reads);
                outline.crates[krate].cached_included = Some(included);
                loaded.names.extend(names);
                loaded.heads.extend(heads);
                loaded.hits[krate] = true;
                loaded.keys[krate] = Some((sk, ck));
            }
        }
        Err(_) => {
            for (_, _, _, mpath, bpath, _) in meta_info {
                drop(std::fs::remove_file(&mpath));
                drop(std::fs::remove_file(&bpath));
            }
        }
    }
    loaded
}

/// Atomically persists settled local library crates from `compilation` to `$HASTE_HOME/local-index/` and returns their impl indices.
pub(crate) fn save(
    workspace: &Workspace,
    targets: &Targets,
    compilation: &Compilation,
    sys_key: u64,
    hits: &[bool],
    hit_keys: &[Option<(u64, u64)>],
) -> HashMap<u32, Arc<ImplIndex>> {
    let mut all_heads = HashMap::default();
    if !enabled() {
        return all_heads;
    }
    let outline = &compilation.outline;
    let th = target_fingerprint(targets);
    let mut prep = PrepCache::new(workspace, outline);
    let mut keys: Vec<Option<(u64, u64)>> = hit_keys.to_vec();
    keys.resize(outline.crates.len(), None);
    let (mut last_bin, mut last_meta) = (None, None);

    for krate in local_lib_topo_order(outline, &prep.ranks) {
        if hits.get(krate).copied().unwrap_or(false) {
            continue;
        }
        let tree = &outline.crates[krate];
        if tree.partial()
            || tree.unsettled()
            || !tree.errors.is_empty()
            || !compilation.names.crates.get(krate).is_some_and(|c| c.eager)
        {
            continue;
        }
        let Some(refs) = prep.crate_refs(outline, krate) else {
            continue;
        };
        let deps_settled = refs
            .sysroot_deps
            .iter()
            .chain(&refs.pinned_deps)
            .all(|&d| !outline.crates[d].partial() && !outline.crates[d].unsettled());
        let Some(dep_keys) = deps_settled
            .then(|| {
                refs.local_deps
                    .iter()
                    .map(|&d| keys[d])
                    .collect::<Option<Vec<_>>>()
            })
            .flatten()
        else {
            continue;
        };
        let Some(sk) = prep.structural_key(workspace, targets, outline, krate, &refs, sys_key, th)
        else {
            continue;
        };
        let (watched, included) = collect_watched_files(outline, krate);
        let env_reads_vec = collect_env_reads(outline, &[krate]);
        let env_hashes = hash_env_reads(env_reads_vec.first().map_or(&[][..], Vec::as_slice));
        let content_key = compute_content_key(sk, &watched, &included, &env_hashes, &dep_keys);
        let cached = [krate];
        let spec = SnapshotSpec {
            magic: LOCAL_INDEX_MAGIC,
            version: LOCAL_INDEX_VERSION,
            key: content_key,
            cached_crates: &cached,
            ref_crates: &refs.ref_crates,
            env_reads: &env_reads_vec,
        };
        let Some((bytes, heads)) = encode_local_snapshot(&spec, workspace, compilation) else {
            continue;
        };
        all_heads.extend(heads);
        let bpath = bin_path(content_key);
        if let Err(error) = super::backing::write_raw(&bpath, &bytes) {
            eprintln!("haste: cannot save local index cache: {error:#}");
            continue;
        }
        let meta = LocalMeta {
            content_key,
            dep_keys,
            watched,
            included,
            env_reads: env_hashes,
        };
        let Some(meta_bytes) = meta::encode_meta(sk, &meta) else {
            continue;
        };
        let mpath = meta_path(sk);
        if let Err(error) = super::backing::write_raw(&mpath, &meta_bytes) {
            eprintln!("haste: cannot save local index meta: {error:#}");
            continue;
        }
        keys[krate] = Some((sk, content_key));
        last_bin = Some(bpath);
        last_meta = Some(mpath);
    }

    prune_cache_pair(
        &cache_dir(),
        last_bin.as_deref(),
        "bin",
        last_meta.as_deref(),
        "meta",
        KEPT_LOCAL_INDEX,
    );
    all_heads
}

#[cfg(test)]
mod tests;

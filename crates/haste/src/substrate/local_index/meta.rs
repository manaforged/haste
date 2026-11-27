use std::hash::{Hash as _, Hasher as _};
use std::path::PathBuf;

use rustc_hash::FxHasher;

use crate::graph::store::{self, FileStamp};
use crate::graph::{Targets, Workspace};
use crate::outline::source::Origins;
use crate::outline::tree::Profile;
use crate::outline::{Outline, Role};
use crate::sema::summary::codec::{Cursor, read_str, write_str, write_u32, write_u64};

use super::super::backing::{bytes_to_path, path_to_bytes};
use super::super::sysroot::{SYSROOT_VERSION, crate_order, hash_identity};

pub(super) const LOCAL_INDEX_MAGIC: [u8; 8] = *b"HSTLOC01";
const LOCAL_META_MAGIC: [u8; 8] = *b"HSTLMT01";
pub(super) const LOCAL_INDEX_VERSION: u32 = 3;

pub(super) type EnvHashes = Vec<(String, Option<u64>)>;

pub(super) struct CrateRefs {
    pub local_deps: Vec<usize>,
    pub pinned_deps: Vec<usize>,
    pub sysroot_deps: Vec<usize>,
    pub ref_crates: Vec<usize>,
}

pub(super) struct LocalMeta {
    pub content_key: u64,
    pub dep_keys: Vec<(u64, u64)>,
    pub watched: Vec<(PathBuf, FileStamp)>,
    pub included: Vec<(PathBuf, PathBuf)>,
    pub env_reads: EnvHashes,
}

fn is_local_lib(outline: &Outline, krate: usize) -> bool {
    !outline.is_fixed(krate as u32)
        && outline
            .roots
            .get(krate)
            .is_some_and(|root| root.role == Role::Lib)
}

pub(super) fn crate_ranks(workspace: &Workspace, outline: &Outline) -> Vec<usize> {
    let ordered = crate_order(workspace, outline, |_| true);
    let mut ranks = vec![0usize; outline.crates.len()];
    for (rank, &krate) in ordered.iter().enumerate() {
        ranks[krate] = rank;
    }
    ranks
}

pub(super) fn crate_refs(outline: &Outline, ranks: &[usize], krate: usize) -> Option<CrateRefs> {
    let mut seen = vec![false; outline.crates.len()];
    let mut stack: Vec<usize> = outline.deps[krate].iter().map(|&(_, d)| d).collect();
    let mut closure = Vec::new();
    while let Some(dep) = stack.pop() {
        if dep >= outline.crates.len() || std::mem::replace(&mut seen[dep], true) {
            continue;
        }
        closure.push(dep);
        for &(_, next) in &outline.deps[dep] {
            stack.push(next);
        }
    }
    let (mut local_deps, mut pinned_deps, mut sysroot_deps) = (Vec::new(), Vec::new(), Vec::new());
    for &dep in &closure {
        if outline.is_sysroot(dep as u32) {
            sysroot_deps.push(dep);
        } else if outline.is_fixed(dep as u32) {
            pinned_deps.push(dep);
        } else {
            if !is_local_lib(outline, dep) {
                return None;
            }
            local_deps.push(dep);
        }
    }
    local_deps.sort_unstable_by_key(|&k| ranks[k]);
    pinned_deps.sort_unstable_by_key(|&k| ranks[k]);
    sysroot_deps.sort_unstable_by_key(|&k| ranks[k]);
    let mut ref_crates =
        Vec::with_capacity(1 + local_deps.len() + pinned_deps.len() + sysroot_deps.len());
    ref_crates.push(krate);
    ref_crates.extend_from_slice(&local_deps);
    ref_crates.extend_from_slice(&pinned_deps);
    ref_crates.extend_from_slice(&sysroot_deps);
    Some(CrateRefs {
        local_deps,
        pinned_deps,
        sysroot_deps,
        ref_crates,
    })
}

pub(super) fn local_lib_topo_order(outline: &Outline, ranks: &[usize]) -> Vec<usize> {
    let mut ordered: Vec<usize> = (0..outline.crates.len())
        .filter(|&k| is_local_lib(outline, k) && outline.crates[k].env.known)
        .collect();
    ordered.sort_unstable_by_key(|&k| ranks[k]);
    let mut state = vec![0u8; outline.crates.len()];
    let mut out = Vec::with_capacity(ordered.len());
    fn visit(outline: &Outline, k: usize, state: &mut [u8], out: &mut Vec<usize>) {
        if state[k] != 0 {
            return;
        }
        state[k] = 1;
        for &(_, dep) in &outline.deps[k] {
            if is_local_lib(outline, dep) && outline.crates[dep].env.known {
                visit(outline, dep, state, out);
            }
        }
        state[k] = 2;
        out.push(k);
    }
    for k in ordered {
        visit(outline, k, &mut state, &mut out);
    }
    out
}

pub(super) fn target_fingerprint(targets: &Targets) -> (u64, u64) {
    let host_th = super::super::target_hash(&crate::host_target());
    let mut hasher = FxHasher::default();
    (
        &targets.target.triple,
        &targets.target.names,
        &targets.target.pairs,
        &targets.target.internal,
        crate::with_cfg(|c| c.x64_baseline),
    )
        .hash(&mut hasher);
    (host_th, hasher.finish())
}

fn hash_local_env(workspace: &Workspace, outline: &Outline, krate: usize, hasher: &mut FxHasher) {
    let root = &outline.roots[krate];
    let pkg = &workspace.packages[workspace.graph.units[root.unit].package];
    let tree = &outline.crates[krate];
    (
        &pkg.manifest.dir,
        &root.edition,
        root.proc_macro,
        &root.path,
        tree.env.script.as_ref().map(|s| s.key_parts()),
    )
        .hash(hasher);
    (
        &tree.env.vars,
        tree.env.profile_override,
        tree.env.test,
        tree.env.known,
        tree.env.release,
    )
        .hash(hasher);
    (
        &tree.env.target.triple,
        &tree.env.target.names,
        &tree.env.target.pairs,
        &tree.env.target.internal,
    )
        .hash(hasher);
}

pub(super) fn structural_key(
    workspace: &Workspace,
    targets: &Targets,
    outline: &Outline,
    krate: usize,
    refs: &CrateRefs,
    sys_key: u64,
    (host_th, target_th): (u64, u64),
    pinned_stamps: &mut [Option<FileStamp>],
) -> Option<u64> {
    let mut hasher = FxHasher::default();
    (
        LOCAL_INDEX_MAGIC,
        LOCAL_INDEX_VERSION,
        SYSROOT_VERSION,
        super::super::MAGIC,
        super::super::VERSION,
        crate::profile() == Profile::Release,
        sys_key,
        host_th,
        target_th,
    )
        .hash(&mut hasher);
    (
        &targets.target.triple,
        &workspace.rustflags,
        &workspace.config_env,
        workspace.profile_dev,
        workspace.profile_release,
        refs.ref_crates.len(),
    )
        .hash(&mut hasher);
    let mut to_local = vec![None; outline.crates.len()];
    for (pos, &c) in refs.ref_crates.iter().enumerate() {
        to_local[c] = Some(pos as u32);
    }
    for &c in &refs.ref_crates {
        hash_identity(workspace, outline, c, &to_local, &mut hasher)?;
    }
    hash_local_env(workspace, outline, krate, &mut hasher);
    for &dep in &refs.local_deps {
        hash_local_env(workspace, outline, dep, &mut hasher);
    }
    for &dep in &refs.pinned_deps {
        let root = &outline.roots[dep];
        let pkg = &workspace.packages[workspace.graph.units[root.unit].package];
        let tree = &outline.crates[dep];
        let root_stamp = if let Some(slot) = pinned_stamps.get_mut(dep) {
            *slot.get_or_insert_with(|| store::file_stamp(&root.path))
        } else {
            store::file_stamp(&root.path)
        };
        (
            &pkg.locked.source,
            &pkg.locked.checksum,
            &pkg.manifest.dir,
            &root.edition,
            root.proc_macro,
            &root.path,
            root_stamp,
        )
            .hash(&mut hasher);
        (
            outline.pins.get(dep).copied().flatten(),
            tree.env.script.as_ref().map(|s| s.key_parts()),
            &tree.env.vars,
            tree.env.profile_override,
            tree.env.test,
            tree.env.known,
        )
            .hash(&mut hasher);
    }
    Some(hasher.finish())
}

pub(super) fn collect_watched_files(
    outline: &Outline,
    krate: usize,
) -> (Vec<(PathBuf, FileStamp)>, Vec<(PathBuf, PathBuf)>) {
    let tree = &outline.crates[krate];
    let mut included = outline.included(krate);
    included.sort();
    included.dedup();
    let mut watched: Vec<(PathBuf, FileStamp)> = tree
        .files
        .iter()
        .filter(|f| {
            matches!(f.source.origins, Origins::Read) && !f.source.path.starts_with("/haste/")
        })
        .map(|f| (f.source.path.clone(), store::file_stamp(&f.source.path)))
        .chain(
            included
                .iter()
                .map(|(p, _)| (p.clone(), store::file_stamp(p))),
        )
        .chain(tree.missing.iter().map(|m| (m.clone(), None)))
        .collect();
    watched.sort_by(|a, b| a.0.cmp(&b.0));
    watched.dedup_by(|a, b| a.0 == b.0);
    (watched, included)
}

pub(super) fn hash_env_reads(env_reads: &[(String, Option<String>)]) -> EnvHashes {
    env_reads
        .iter()
        .map(|(name, val)| {
            let hash = val
                .as_deref()
                .map(|v| store::hash_env_val(&LOCAL_META_MAGIC, name, v));
            (name.clone(), hash)
        })
        .collect()
}

pub(super) fn compute_content_key(
    structural: u64,
    watched: &[(PathBuf, FileStamp)],
    included: &[(PathBuf, PathBuf)],
    env_reads: &[(String, Option<u64>)],
    dep_keys: &[(u64, u64)],
) -> u64 {
    let mut hasher = FxHasher::default();
    (
        LOCAL_INDEX_MAGIC,
        LOCAL_INDEX_VERSION,
        structural,
        watched,
        included,
        env_reads,
        dep_keys,
    )
        .hash(&mut hasher);
    hasher.finish()
}

fn write_path_bytes(out: &mut Vec<u8>, path: &std::path::Path) -> Option<()> {
    let bytes = path_to_bytes(path)?;
    write_u32(out, u32::try_from(bytes.len()).ok()?);
    out.extend_from_slice(&bytes);
    Some(())
}

fn read_path_bytes(cur: &mut Cursor<'_>) -> Option<PathBuf> {
    let plen = cur.count().ok()?;
    bytes_to_path(cur.take(plen).ok()?)
}

pub(super) fn encode_meta(structural: u64, meta: &LocalMeta) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(
        64 + meta.dep_keys.len() * 16
            + meta.watched.len() * 80
            + meta.included.len() * 128
            + meta.env_reads.len() * 32,
    );
    out.extend_from_slice(&LOCAL_META_MAGIC);
    write_u32(&mut out, LOCAL_INDEX_VERSION);
    write_u64(&mut out, structural);
    write_u64(&mut out, meta.content_key);
    write_u32(&mut out, u32::try_from(meta.dep_keys.len()).ok()?);
    for &(sk, ck) in &meta.dep_keys {
        write_u64(&mut out, sk);
        write_u64(&mut out, ck);
    }
    write_u32(&mut out, u32::try_from(meta.watched.len()).ok()?);
    for (path, stamp) in &meta.watched {
        write_path_bytes(&mut out, path)?;
        match stamp {
            Some((len, nanos)) => {
                out.push(1);
                write_u64(&mut out, *len);
                out.extend_from_slice(&nanos.to_le_bytes());
            }
            None => out.push(0),
        }
    }
    write_u32(&mut out, u32::try_from(meta.env_reads.len()).ok()?);
    for (name, hash) in &meta.env_reads {
        write_str(&mut out, name)?;
        match hash {
            Some(h) => {
                out.push(1);
                write_u64(&mut out, *h);
            }
            None => out.push(0),
        }
    }
    write_u32(&mut out, u32::try_from(meta.included.len()).ok()?);
    for (path, by) in &meta.included {
        write_path_bytes(&mut out, path)?;
        write_path_bytes(&mut out, by)?;
    }
    out.extend_from_slice(&LOCAL_META_MAGIC);
    Some(out)
}

pub(super) fn decode_meta(data: &[u8], expected_structural: u64) -> Option<LocalMeta> {
    if data.len() < 36
        || data[..8] != LOCAL_META_MAGIC
        || data[data.len() - 8..] != LOCAL_META_MAGIC
    {
        return None;
    }
    let mut cur = Cursor::new(&data[..data.len() - 8]);
    cur.take(8).ok()?;
    if cur.u32().ok()? != LOCAL_INDEX_VERSION || cur.u64().ok()? != expected_structural {
        return None;
    }
    let content_key = cur.u64().ok()?;
    let dep_count = cur.count().ok()?;
    let mut dep_keys = Vec::with_capacity(dep_count.min(256));
    for _ in 0..dep_count {
        dep_keys.push((cur.u64().ok()?, cur.u64().ok()?));
    }
    let watch_count = cur.count().ok()?;
    let mut watched = Vec::with_capacity(watch_count.min(4096));
    for _ in 0..watch_count {
        let path = read_path_bytes(&mut cur)?;
        let stamp = match cur.u8().ok()? {
            0 => None,
            1 => {
                let len = cur.u64().ok()?;
                let raw = cur.take(16).ok()?;
                Some((len, i128::from_le_bytes(raw.try_into().ok()?)))
            }
            _ => return None,
        };
        watched.push((path, stamp));
    }
    let env_count = cur.count().ok()?;
    let mut env_reads = Vec::with_capacity(env_count.min(256));
    for _ in 0..env_count {
        let name = read_str(&mut cur).ok()?.to_owned();
        let hash = match cur.u8().ok()? {
            0 => None,
            1 => Some(cur.u64().ok()?),
            _ => return None,
        };
        env_reads.push((name, hash));
    }
    let included_count = cur.count().ok()?;
    let mut included = Vec::with_capacity(included_count.min(1024));
    for _ in 0..included_count {
        let path = read_path_bytes(&mut cur)?;
        let by = read_path_bytes(&mut cur)?;
        included.push((path, by));
    }
    (cur.pos == data.len() - 8).then_some(LocalMeta {
        content_key,
        dep_keys,
        watched,
        included,
        env_reads,
    })
}

pub(super) fn validate_watched(watched: &[(PathBuf, FileStamp)], parallel: bool) -> bool {
    let check = |slice: &[(PathBuf, FileStamp)]| {
        slice
            .iter()
            .all(|(path, expected)| store::file_stamp(path) == *expected)
    };
    if !parallel || watched.len() < 64 || crate::jobs() <= 1 {
        return check(watched);
    }
    let chunk = watched.len().div_ceil(crate::jobs().clamp(1, 8));
    crate::sync::scope(|scope| {
        let handles: Vec<_> = watched
            .chunks(chunk)
            .map(|s| scope.spawn(move || check(s)))
            .collect();
        let mut ok = true;
        for handle in handles {
            match handle.join() {
                Ok(res) => ok &= res,
                Err(payload) => std::panic::resume_unwind(payload),
            }
        }
        ok
    })
}

pub(super) fn validate_env_reads(env_reads: &[(String, Option<u64>)]) -> bool {
    env_reads.iter().all(|(name, expected)| {
        crate::process_env_var(name)
            .as_deref()
            .map(|v| store::hash_env_val(&LOCAL_META_MAGIC, name, v))
            == *expected
    })
}

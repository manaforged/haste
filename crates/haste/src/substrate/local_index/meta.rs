use std::hash::{Hash as _, Hasher as _};
use std::path::{Path, PathBuf};

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
pub(super) const LOCAL_INDEX_VERSION: u32 = 4;

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

fn crate_ranks(workspace: &Workspace, outline: &Outline) -> Vec<usize> {
    let ordered = crate_order(workspace, outline, |_| true);
    let mut ranks = vec![0usize; outline.crates.len()];
    for (rank, &krate) in ordered.iter().enumerate() {
        ranks[krate] = rank;
    }
    ranks
}

pub(super) struct PrepCache {
    pub ranks: Vec<usize>,
    seen: Vec<u32>,
    epoch: u32,
    to_local: Vec<Option<u32>>,
    pinned_stamps: Vec<Option<FileStamp>>,
}

impl PrepCache {
    pub(super) fn new(workspace: &Workspace, outline: &Outline) -> Self {
        let n = outline.crates.len();
        Self {
            ranks: crate_ranks(workspace, outline),
            seen: vec![0; n],
            epoch: 0,
            to_local: vec![None; n],
            pinned_stamps: vec![None; n],
        }
    }

    pub(super) fn crate_refs(&mut self, outline: &Outline, krate: usize) -> Option<CrateRefs> {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.seen.fill(0);
            self.epoch = 1;
        }
        let mark = self.epoch;
        let mut stack: Vec<usize> = outline.deps[krate].iter().map(|&(_, d)| d).collect();
        let (mut local_deps, mut pinned_deps, mut sysroot_deps) =
            (Vec::new(), Vec::new(), Vec::new());
        while let Some(dep) = stack.pop() {
            if dep >= outline.crates.len() || self.seen[dep] == mark {
                continue;
            }
            self.seen[dep] = mark;
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
            for &(_, next) in &outline.deps[dep] {
                stack.push(next);
            }
        }
        local_deps.sort_unstable_by_key(|&k| self.ranks[k]);
        pinned_deps.sort_unstable_by_key(|&k| self.ranks[k]);
        sysroot_deps.sort_unstable_by_key(|&k| self.ranks[k]);
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

    pub(super) fn structural_key(
        &mut self,
        workspace: &Workspace,
        targets: &Targets,
        outline: &Outline,
        krate: usize,
        refs: &CrateRefs,
        sys_key: u64,
        (host_th, target_th): (u64, u64),
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
        for (pos, &c) in refs.ref_crates.iter().enumerate() {
            self.to_local[c] = Some(pos as u32);
        }
        let res = (|| {
            for &c in &refs.ref_crates {
                hash_identity(workspace, outline, c, &self.to_local, &mut hasher)?;
            }
            Some(())
        })();
        for &c in &refs.ref_crates {
            self.to_local[c] = None;
        }
        res?;
        hash_local_env(workspace, outline, krate, &mut hasher);
        for &dep in &refs.local_deps {
            hash_local_env(workspace, outline, dep, &mut hasher);
        }
        for &dep in &refs.pinned_deps {
            let root = &outline.roots[dep];
            let pkg = &workspace.packages[workspace.graph.units[root.unit].package];
            let tree = &outline.crates[dep];
            let root_stamp = if let Some(slot) = self.pinned_stamps.get_mut(dep) {
                *slot.get_or_insert_with(|| {
                    std::fs::metadata(&root.path)
                        .ok()
                        .map(|m| store::meta_stamp(&m))
                })
            } else {
                std::fs::metadata(&root.path)
                    .ok()
                    .map(|m| store::meta_stamp(&m))
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

fn decode_meta_prefix<'a>(
    data: &'a [u8],
    expected_structural: Option<u64>,
) -> Option<(u64, Vec<(u64, u64)>, Vec<(PathBuf, FileStamp)>, Cursor<'a>)> {
    if data.len() < 36
        || data[..8] != LOCAL_META_MAGIC
        || data[data.len() - 8..] != LOCAL_META_MAGIC
    {
        return None;
    }
    let mut cur = Cursor::new(&data[..data.len() - 8]);
    cur.take(8).ok()?;
    let version = cur.u32().ok()?;
    if expected_structural.is_some() && version != LOCAL_INDEX_VERSION {
        return None;
    }
    let structural = cur.u64().ok()?;
    if expected_structural.is_some_and(|exp| structural != exp) {
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
    Some((content_key, dep_keys, watched, cur))
}

pub(super) enum MetaCheck {
    Corrupt,
    Stale,
    Valid(LocalMeta),
}

fn read_meta_envs(cur: &mut Cursor<'_>) -> Option<EnvHashes> {
    let env_count = cur.count().ok()?;
    let mut env_reads = Vec::with_capacity(env_count.min(256));
    for _ in 0..env_count {
        let name = read_str(cur).ok()?.to_owned();
        let hash = match cur.u8().ok()? {
            0 => None,
            1 => Some(cur.u64().ok()?),
            _ => return None,
        };
        env_reads.push((name, hash));
    }
    Some(env_reads)
}

fn read_meta_included(cur: &mut Cursor<'_>) -> Option<Vec<(PathBuf, PathBuf)>> {
    let included_count = cur.count().ok()?;
    let mut included = Vec::with_capacity(included_count.min(1024));
    for _ in 0..included_count {
        let path = read_path_bytes(cur)?;
        let by = read_path_bytes(cur)?;
        included.push((path, by));
    }
    Some(included)
}

#[cfg(test)]
pub(super) fn decode_meta(data: &[u8], expected_structural: u64) -> Option<LocalMeta> {
    let (content_key, dep_keys, watched, mut cur) =
        decode_meta_prefix(data, Some(expected_structural))?;
    let env_reads = read_meta_envs(&mut cur)?;
    let included = read_meta_included(&mut cur)?;
    (cur.pos == data.len() - 8).then_some(LocalMeta {
        content_key,
        dep_keys,
        watched,
        included,
        env_reads,
    })
}

pub(super) fn check_meta(
    data: &[u8],
    expected_structural: u64,
    expected_deps: &[(u64, u64)],
    parallel_stat: bool,
) -> MetaCheck {
    let Some((content_key, dep_keys, watched, mut cur)) =
        decode_meta_prefix(data, Some(expected_structural))
    else {
        return MetaCheck::Corrupt;
    };
    if dep_keys != expected_deps {
        if read_meta_envs(&mut cur).is_none()
            || read_meta_included(&mut cur).is_none()
            || cur.pos != data.len() - 8
        {
            return MetaCheck::Corrupt;
        }
        return MetaCheck::Stale;
    }
    let Some(env_reads) = read_meta_envs(&mut cur) else {
        return MetaCheck::Corrupt;
    };
    if !validate_env_reads(&env_reads) {
        if read_meta_included(&mut cur).is_none() || cur.pos != data.len() - 8 {
            return MetaCheck::Corrupt;
        }
        return MetaCheck::Stale;
    }
    if !validate_watched(&watched, parallel_stat) {
        if read_meta_included(&mut cur).is_none() || cur.pos != data.len() - 8 {
            return MetaCheck::Corrupt;
        }
        return MetaCheck::Stale;
    }
    let Some(included) = read_meta_included(&mut cur) else {
        return MetaCheck::Corrupt;
    };
    if cur.pos != data.len() - 8 {
        return MetaCheck::Corrupt;
    }
    if compute_content_key(
        expected_structural,
        &watched,
        &included,
        &env_reads,
        &dep_keys,
    ) != content_key
    {
        return MetaCheck::Stale;
    }
    MetaCheck::Valid(LocalMeta {
        content_key,
        dep_keys,
        watched,
        included,
        env_reads,
    })
}

pub(in crate::substrate) fn file_stamp_matches(
    path: &Path,
    expected: FileStamp,
    buf: &mut Vec<u8>,
) -> bool {
    let Ok(meta) = std::fs::metadata(path) else {
        return expected.is_none();
    };
    let Some((exp_len, exp_packed)) = expected else {
        return false;
    };
    let (len, nanos) = store::meta_stamp(&meta);
    if len != exp_len || (nanos as u64) != (exp_packed as u64) {
        return false;
    }
    buf.clear();
    let actual = match std::fs::File::open(path).and_then(|mut f| {
        if let Ok(cap) = usize::try_from(len) {
            buf.reserve(cap);
        }
        std::io::Read::read_to_end(&mut f, buf)
    }) {
        Ok(_) => {
            let hash = store::hash_bytes(buf);
            (len, ((hash as i128) << 64) | i128::from(nanos as u64))
        }
        Err(_) => (len, nanos),
    };
    actual == (exp_len, exp_packed)
}

pub(super) fn validate_watched(watched: &[(PathBuf, FileStamp)], parallel: bool) -> bool {
    let check = |slice: &[(PathBuf, FileStamp)]| {
        let mut buf = Vec::new();
        slice
            .iter()
            .all(|(path, expected)| file_stamp_matches(path, *expected, &mut buf))
    };
    if !parallel || watched.len() < 32 || crate::jobs() <= 1 {
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

/// Returns the `.bin` content key if an encoded `.meta` buffer watches any path inside `roots`.
pub(crate) fn matches_workspace(data: &[u8], roots: &[PathBuf]) -> Option<u64> {
    let (content_key, _, watched, _) = decode_meta_prefix(data, None)?;
    watched
        .iter()
        .any(|(path, _)| super::super::noop::path_in_workspace(path, roots))
        .then_some(content_key)
}

#[cfg(test)]
pub(crate) fn encode_test_meta(content_key: u64, watched: &std::path::Path) -> Vec<u8> {
    encode_meta(
        0,
        &LocalMeta {
            content_key,
            dep_keys: Vec::new(),
            watched: vec![(watched.to_path_buf(), None)],
            included: Vec::new(),
            env_reads: Vec::new(),
        },
    )
    .unwrap_or_default()
}

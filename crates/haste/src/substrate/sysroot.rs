//! Persistent disk cache for settled sysroot crate outlines, macros, and resolved name scopes.

mod impls;
mod names;
mod outline;
pub(crate) mod tokens;
mod wire;
pub(crate) use names::decode_scope_map;
pub(super) use wire::{DecodeSpec, SnapshotSpec, prepare_wire};
pub(crate) use wire::{PreparedWire, decode_prepared_batch};

use std::fmt::Write as _;
use std::hash::Hash as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rustc_hash::FxHashMap as HashMap;

use crate::graph::{Side, Targets, Workspace};
use crate::macros::Compilation;
use crate::names::{Crate as NamesCrate, Symbols};
use crate::outline::{Outline, Role};
use crate::sema::impls::Index as ImplIndex;
use crate::sema::summary::codec::Cursor;

pub(crate) const SYSROOT_VERSION: u32 = 16;
const SYSROOT_MAGIC: [u8; 8] = *b"HSTSYS01";
pub(super) const HEADER_LEN: usize = 36;
const KEPT_SYSROOTS: usize = 2;

pub(super) type EnvReads = Vec<(String, Option<String>)>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct CrateIdentity {
    pub unit_name: String,
    pub pkg_name: String,
    pub pkg_version: String,
    pub side_host: bool,
    pub role: u8,
    pub features: Vec<String>,
    pub deps: Vec<(String, u32)>,
}

const ROLES: [Role; 6] = [
    Role::Lib,
    Role::Bin,
    Role::Example,
    Role::Test,
    Role::Script,
    Role::Bench,
];

pub(super) fn role_tag(role: Role) -> u8 {
    ROLES.iter().position(|&r| r == role).unwrap_or(0) as u8
}

pub(super) fn crate_order(
    workspace: &Workspace,
    outline: &Outline,
    include: impl Fn(usize) -> bool,
) -> Vec<usize> {
    let mut crates: Vec<usize> = (0..outline.crates.len()).filter(|&k| include(k)).collect();
    crates.sort_unstable_by(|&a, &b| {
        let ra = &outline.roots[a];
        let rb = &outline.roots[b];
        let ua = &workspace.graph.units[ra.unit];
        let ub = &workspace.graph.units[rb.unit];
        let pa = &workspace.packages[ua.package].locked.name;
        let pb = &workspace.packages[ub.package].locked.name;
        (
            ra.name.as_str(),
            matches!(ua.side, Side::Host),
            pa.as_str(),
            role_tag(ra.role),
            a,
        )
            .cmp(&(
                rb.name.as_str(),
                matches!(ub.side, Side::Host),
                pb.as_str(),
                role_tag(rb.role),
                b,
            ))
    });
    crates
}

pub(super) fn sysroot_crate_order(workspace: &Workspace, outline: &Outline) -> Vec<usize> {
    crate_order(workspace, outline, |k| outline.is_sysroot(k as u32))
}

pub(super) fn identity_of(
    workspace: &Workspace,
    outline: &Outline,
    krate: usize,
    to_local: &[Option<u32>],
) -> Option<CrateIdentity> {
    let root = &outline.roots[krate];
    let unit = &workspace.graph.units[root.unit];
    let pkg = &workspace.packages[unit.package];
    let mut deps = Vec::with_capacity(outline.deps[krate].len());
    for (name, dep) in &outline.deps[krate] {
        deps.push((name.clone(), (*to_local.get(*dep)?)?));
    }
    Some(CrateIdentity {
        unit_name: root.name.clone(),
        pkg_name: pkg.locked.name.clone(),
        pkg_version: pkg.locked.version.to_string(),
        side_host: matches!(unit.side, Side::Host),
        role: role_tag(root.role),
        features: unit.features.iter().cloned().collect(),
        deps,
    })
}

struct StackStr<const N: usize> {
    buf: [u8; N],
    len: usize,
}

impl<const N: usize> StackStr<N> {
    fn new() -> Self {
        Self {
            buf: [0; N],
            len: 0,
        }
    }

    fn as_str(&self) -> Option<&str> {
        std::str::from_utf8(&self.buf[..self.len]).ok()
    }
}

impl<const N: usize> std::fmt::Write for StackStr<N> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        let bytes = s.as_bytes();
        let end = self.len.checked_add(bytes.len()).ok_or(std::fmt::Error)?;
        if end > N {
            return Err(std::fmt::Error);
        }
        self.buf[self.len..end].copy_from_slice(bytes);
        self.len = end;
        Ok(())
    }
}

fn with_version_str<R>(ver: &semver::Version, f: impl FnOnce(&str) -> R) -> R {
    let mut stack = StackStr::<64>::new();
    if write!(&mut stack, "{ver}").is_ok()
        && let Some(s) = stack.as_str()
    {
        return f(s);
    }
    f(&ver.to_string())
}

pub(super) fn hash_identity(
    workspace: &Workspace,
    outline: &Outline,
    krate: usize,
    to_local: &[Option<u32>],
    hasher: &mut impl std::hash::Hasher,
) -> Option<()> {
    let root = &outline.roots[krate];
    let unit = &workspace.graph.units[root.unit];
    let pkg = &workspace.packages[unit.package];
    root.name.hash(hasher);
    pkg.locked.name.hash(hasher);
    with_version_str(&pkg.locked.version, |v| v.hash(hasher));
    matches!(unit.side, Side::Host).hash(hasher);
    role_tag(root.role).hash(hasher);
    unit.features.len().hash(hasher);
    for feat in &unit.features {
        feat.hash(hasher);
    }
    outline.deps[krate].len().hash(hasher);
    for (name, dep) in &outline.deps[krate] {
        name.hash(hasher);
        (*to_local.get(*dep)?)?.hash(hasher);
    }
    Some(())
}

fn match_identity_wire(
    cur: &mut Cursor<'_>,
    workspace: &Workspace,
    outline: &Outline,
    krate: usize,
    to_local: &[Option<u32>],
) -> anyhow::Result<bool> {
    let root = &outline.roots[krate];
    let unit = &workspace.graph.units[root.unit];
    let pkg = &workspace.packages[unit.package];
    if tokens::read_str(cur)? != root.name
        || tokens::read_str(cur)? != pkg.locked.name
        || !with_version_str(&pkg.locked.version, |v| {
            tokens::read_str(cur).map(|wire| wire == v)
        })?
        || (cur.u8()? != 0) != matches!(unit.side, Side::Host)
        || cur.u8()? != role_tag(root.role)
    {
        return Ok(false);
    }
    let feat_len = cur.count()?;
    if feat_len != unit.features.len() {
        return Ok(false);
    }
    for feat in &unit.features {
        if tokens::read_str(cur)? != feat {
            return Ok(false);
        }
    }
    let dep_len = cur.count()?;
    if dep_len != outline.deps[krate].len() {
        return Ok(false);
    }
    for (name, dep) in &outline.deps[krate] {
        let Some(local_dep) = to_local.get(*dep).copied().flatten() else {
            return Ok(false);
        };
        if tokens::read_str(cur)? != name || cur.u32()? != local_dep {
            return Ok(false);
        }
    }
    Ok(true)
}

pub(super) fn match_crate_list_wire(
    cur: &mut Cursor<'_>,
    workspace: &Workspace,
    outline: &Outline,
    local: &[usize],
) -> anyhow::Result<bool> {
    let mut to_local = vec![None; outline.crates.len()];
    for (pos, &krate) in local.iter().enumerate() {
        to_local[krate] = Some(pos as u32);
    }
    for &krate in local {
        if !match_identity_wire(cur, workspace, outline, krate, &to_local)? {
            return Ok(false);
        }
    }
    Ok(true)
}

pub(super) fn collect_env_reads(outline: &Outline, crates: &[usize]) -> Vec<EnvReads> {
    crates
        .iter()
        .map(|&krate| {
            let guard = crate::sync::lock(&outline.crates[krate].env.env_reads);
            let mut reads: EnvReads = guard.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
            reads.sort();
            reads
        })
        .collect()
}

fn is_sysroot_cache(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return false;
    };
    name.starts_with("sysroot-") && name.ends_with(".bin")
}

pub(crate) fn apply_decoded_crates(
    outline: &mut Outline,
    crates: Vec<(usize, crate::outline::tree::Crate)>,
    env_reads: Vec<EnvReads>,
) {
    for (idx, (krate, mut tree)) in crates.into_iter().enumerate() {
        tree.env = std::mem::take(&mut outline.crates[krate].env);
        if let Some(reads) = env_reads.get(idx)
            && !reads.is_empty()
        {
            let mut guard = crate::sync::lock(&tree.env.env_reads);
            for (name, val) in reads {
                guard.entry(name.clone()).or_insert_with(|| val.clone());
            }
        }
        outline.crates[krate] = tree;
    }
}

/// Combined cached sysroot, pinned-dependency, and local-library snapshot load result.
pub(crate) struct CachedLoad {
    pub symbols: Option<Symbols>,
    pub names: Vec<(usize, NamesCrate)>,
    pub heads: HashMap<u32, Arc<ImplIndex>>,
    pub sysroot_hit: bool,
    pub deps_hit: bool,
    pub local_hits: Vec<bool>,
    pub local_keys: Vec<Option<(u64, u64)>>,
}

pub(super) fn prune_cache_pair(
    dir: &Path,
    keep_a: Option<&Path>,
    ext_a: &str,
    keep_b: Option<&Path>,
    ext_b: &str,
    keep_others: usize,
) {
    if keep_a.is_none() && keep_b.is_none() {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries_a = Vec::new();
    let mut entries_b = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
            continue;
        };
        if keep_a.is_some_and(|keep| ext == ext_a && path != keep) {
            entries_a.push((entry, path));
        } else if keep_b.is_some_and(|keep| ext == ext_b && path != keep) {
            entries_b.push((entry, path));
        }
    }
    for group in [entries_a, entries_b] {
        if group.len() <= keep_others {
            continue;
        }
        let mut dated: Vec<_> = group
            .into_iter()
            .filter_map(|(entry, path)| {
                let meta = entry.metadata().ok()?;
                meta.is_file().then(|| {
                    (
                        meta.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH),
                        path,
                    )
                })
            })
            .collect();
        if dated.len() <= keep_others {
            continue;
        }
        dated.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        dated
            .into_iter()
            .skip(keep_others)
            .for_each(|(_, p)| drop(std::fs::remove_file(p)));
    }
}

fn prepare_raw(
    workspace: &Workspace,
    outline: &Outline,
    key: u64,
) -> Option<(PathBuf, PreparedWire)> {
    if !super::enabled() {
        return None;
    }
    let path = super::sysroot_cache_path(key);
    let backing = Arc::new(super::Backing::open(&path, HEADER_LEN).ok()?);
    let sysroot_crates = sysroot_crate_order(workspace, outline);
    let spec = wire::DecodeSpec {
        magic: SYSROOT_MAGIC,
        version: SYSROOT_VERSION,
        key,
        cached_crates: &sysroot_crates,
        ref_crates: &sysroot_crates,
        scope_bit: 1 << 31,
    };
    match prepare_wire(backing, &spec, workspace, outline) {
        Ok(Some(wire)) => Some((path, wire)),
        Ok(None) | Err(_) => {
            drop(std::fs::remove_file(&path));
            None
        }
    }
}

/// Overlaps sysroot decoding with dependency/local snapshot preparation, then decodes prepared snapshots in one batch.
pub(crate) fn load_cached_crates(
    workspace: &Workspace,
    targets: &Targets,
    outline: &mut Outline,
    sys_key: Option<u64>,
    load_local: bool,
) -> CachedLoad {
    let mut local_hits = vec![false; outline.crates.len()];
    let mut local_keys = vec![None; outline.crates.len()];
    let Some(key) = sys_key else {
        return CachedLoad {
            symbols: None,
            names: Vec::new(),
            heads: HashMap::default(),
            sysroot_hit: false,
            deps_hit: false,
            local_hits,
            local_keys,
        };
    };
    let has_pinned = outline
        .crates
        .iter()
        .any(|c| c.is_fixed() && !c.is_sysroot());
    let has_local_lib = load_local
        && outline.roots.iter().enumerate().any(|(k, r)| {
            !outline.is_fixed(k as u32) && r.role == Role::Lib && outline.crates[k].env.known
        });
    if !has_pinned && !has_local_lib {
        return match load_raw(workspace, outline, key) {
            Some((crates, symbols, names, heads, env_reads)) => {
                apply_decoded_crates(outline, crates, env_reads);
                CachedLoad {
                    symbols,
                    names,
                    heads,
                    sysroot_hit: true,
                    deps_hit: false,
                    local_hits,
                    local_keys,
                }
            }
            None => CachedLoad {
                symbols: None,
                names: Vec::new(),
                heads: HashMap::default(),
                sysroot_hit: false,
                deps_hit: false,
                local_hits,
                local_keys,
            },
        };
    }
    let (sys_prep, deps_prep, local_prep) = crate::sync::scope(|scope| {
        let deps_h =
            has_pinned.then(|| scope.spawn(|| super::deps_index::prepare(workspace, outline, key)));
        let local_h = has_local_lib.then(|| {
            scope.spawn(|| super::local_index::prepare(workspace, targets, outline, key, true))
        });
        let sys_prep = prepare_raw(workspace, outline, key);
        let deps_prep = deps_h.and_then(|h| match h.join() {
            Ok(v) => v,
            Err(p) => std::panic::resume_unwind(p),
        });
        let local_prep = local_h.map(|h| match h.join() {
            Ok(v) => v,
            Err(p) => std::panic::resume_unwind(p),
        });
        (sys_prep, deps_prep, local_prep.unwrap_or_default())
    });
    let mut sysroot_hit = sys_prep.is_some();
    let mut deps_hit = false;
    let local_ready =
        super::local_index::filter_prepared(outline, local_prep, sysroot_hit, deps_prep.is_some());
    if !sysroot_hit {
        let mut symbols = None;
        let mut names = Vec::new();
        let mut heads = HashMap::default();
        if deps_prep.is_some()
            && let Some((deps_syms, deps_names, deps_heads)) =
                super::deps_index::try_load(workspace, outline, None, key)
        {
            symbols = deps_syms;
            names.extend(deps_names);
            heads.extend(deps_heads);
            deps_hit = true;
        }
        if !local_ready.is_empty() {
            let syms = symbols.get_or_insert_with(Symbols::default);
            let mut wires = Vec::with_capacity(local_ready.len());
            let local_meta: Vec<_> = local_ready
                .into_iter()
                .map(|p| {
                    wires.push(p.wire);
                    (
                        p.krate,
                        p.structural_key,
                        p.content_key,
                        p.meta_path,
                        p.bin_path,
                        p.included,
                    )
                })
                .collect();
            match decode_prepared_batch(wires, syms) {
                Ok(decoded) => {
                    for (
                        (krate, sk, ck, _, _, included),
                        (crates, _, l_names, l_heads, env_reads),
                    ) in local_meta.into_iter().zip(decoded)
                    {
                        apply_decoded_crates(outline, crates, env_reads);
                        outline.crates[krate].cached_included = Some(included);
                        names.extend(l_names);
                        heads.extend(l_heads);
                        local_hits[krate] = true;
                        local_keys[krate] = Some((sk, ck));
                    }
                }
                Err(_) => {
                    for (_, _, _, mpath, bpath, _) in local_meta {
                        drop(std::fs::remove_file(&mpath));
                        drop(std::fs::remove_file(&bpath));
                    }
                }
            }
        }
        return CachedLoad {
            symbols,
            names,
            heads,
            sysroot_hit: false,
            deps_hit,
            local_hits,
            local_keys,
        };
    }
    let total_wires =
        usize::from(sysroot_hit) + usize::from(deps_prep.is_some()) + local_ready.len();
    if total_wires == 0 {
        return CachedLoad {
            symbols: None,
            names: Vec::new(),
            heads: HashMap::default(),
            sysroot_hit: false,
            deps_hit: false,
            local_hits,
            local_keys,
        };
    }
    let syms = Symbols::default();
    let mut wires = Vec::with_capacity(total_wires);
    let sys_path = sys_prep.as_ref().map(|(p, _)| p.clone());
    if let Some((_, wire)) = sys_prep {
        wires.push(wire);
    }
    let deps_path = deps_prep.as_ref().map(|d| d.path.clone());
    if let Some(deps) = deps_prep {
        wires.push(deps.wire);
    }
    let local_meta: Vec<_> = local_ready
        .into_iter()
        .map(|p| {
            wires.push(p.wire);
            (
                p.krate,
                p.structural_key,
                p.content_key,
                p.meta_path,
                p.bin_path,
                p.included,
            )
        })
        .collect();
    let mut names = Vec::new();
    let mut heads = HashMap::default();
    match decode_prepared_batch(wires, &syms) {
        Ok(decoded) => {
            let mut iter = decoded.into_iter();
            if sysroot_hit && let Some((crates, _, s_names, s_heads, env_reads)) = iter.next() {
                apply_decoded_crates(outline, crates, env_reads);
                names.extend(s_names);
                heads.extend(s_heads);
            }
            if deps_path.is_some()
                && let Some((crates, _, d_names, d_heads, env_reads)) = iter.next()
            {
                apply_decoded_crates(outline, crates, env_reads);
                names.extend(d_names);
                heads.extend(d_heads);
                deps_hit = true;
            }
            for ((krate, sk, ck, _, _, included), (crates, _, l_names, l_heads, env_reads)) in
                local_meta.into_iter().zip(iter)
            {
                apply_decoded_crates(outline, crates, env_reads);
                outline.crates[krate].cached_included = Some(included);
                names.extend(l_names);
                heads.extend(l_heads);
                local_hits[krate] = true;
                local_keys[krate] = Some((sk, ck));
            }
            return CachedLoad {
                symbols: Some(syms),
                names,
                heads,
                sysroot_hit,
                deps_hit,
                local_hits,
                local_keys,
            };
        }
        Err(_) => {
            sysroot_hit = false;
            if let Some(path) = sys_path {
                drop(std::fs::remove_file(&path));
            }
            if let Some(path) = deps_path {
                drop(std::fs::remove_file(&path));
            }
            for (_, _, _, mpath, bpath, _) in local_meta {
                drop(std::fs::remove_file(&mpath));
                drop(std::fs::remove_file(&bpath));
            }
        }
    }
    CachedLoad {
        symbols: None,
        names,
        heads,
        sysroot_hit,
        deps_hit,
        local_hits,
        local_keys,
    }
}

/// Loads and decodes settled sysroot crates using only an immutable `&Outline`.
pub(crate) fn load_raw(
    workspace: &Workspace,
    outline: &Outline,
    key: u64,
) -> Option<wire::DecodedSnapshot> {
    if !super::enabled() {
        return None;
    }
    let path = super::sysroot_cache_path(key);
    let backing = Arc::new(super::Backing::open(&path, HEADER_LEN).ok()?);
    let sysroot_crates = sysroot_crate_order(workspace, outline);
    let spec = wire::DecodeSpec {
        magic: SYSROOT_MAGIC,
        version: SYSROOT_VERSION,
        key,
        cached_crates: &sysroot_crates,
        ref_crates: &sysroot_crates,
        scope_bit: 1 << 31,
    };
    match wire::decode_snapshot(&backing, &spec, None, workspace, outline) {
        Ok(Some(decoded)) => Some(decoded),
        Ok(None) | Err(_) => {
            drop(std::fs::remove_file(&path));
            None
        }
    }
}

/// Atomically persists the settled sysroot crates from `compilation` to the toolchain cache directory and returns their impl indices.
pub(crate) fn save(
    workspace: &Workspace,
    compilation: &Compilation,
    key: u64,
) -> Option<Arc<HashMap<u32, Arc<ImplIndex>>>> {
    if !super::enabled() {
        return None;
    }
    let sysroot_crates = sysroot_crate_order(workspace, &compilation.outline);
    if sysroot_crates.is_empty() {
        return None;
    }
    let path = super::sysroot_cache_path(key);
    let spec = wire::SnapshotSpec {
        magic: SYSROOT_MAGIC,
        version: SYSROOT_VERSION,
        key,
        cached_crates: &sysroot_crates,
        ref_crates: &sysroot_crates,
        env_reads: &[],
    };
    let (encoded, heads) = wire::encode_snapshot_chunks(&spec, workspace, compilation, false)?;
    let heads = Arc::new(heads);
    if let Err(error) = crate::graph::store::write_new_with(&path, |w| encoded.write_to(w)) {
        eprintln!("haste: cannot save sysroot cache: {error:#}");
        return Some(heads);
    }
    if let Some(dir) = path.parent() {
        super::prune_stale(dir, &path, KEPT_SYSROOTS, is_sysroot_cache);
    }
    Some(heads)
}

pub(super) fn encode_snapshot(
    spec: &wire::SnapshotSpec<'_>,
    workspace: &Workspace,
    compilation: &Compilation,
) -> Option<(Vec<u8>, HashMap<u32, Arc<ImplIndex>>)> {
    wire::encode_snapshot(spec, workspace, compilation)
}

pub(super) fn encode_local_snapshot(
    spec: &wire::SnapshotSpec<'_>,
    workspace: &Workspace,
    compilation: &Compilation,
) -> Option<(Vec<u8>, HashMap<u32, Arc<ImplIndex>>)> {
    wire::encode_snapshot_with(spec, workspace, compilation, true)
}

pub(super) fn decode_snapshot(
    backing: &Arc<super::Backing>,
    spec: &wire::DecodeSpec<'_>,
    existing_symbols: Option<&Symbols>,
    workspace: &Workspace,
    outline: &Outline,
) -> anyhow::Result<Option<wire::DecodedSnapshot>> {
    wire::decode_snapshot(backing, spec, existing_symbols, workspace, outline)
}

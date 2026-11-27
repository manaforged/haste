//! Persistent disk cache for settled pinned dependency crate outlines, macros, names, and impl indices.

use std::hash::{Hash as _, Hasher as _};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rustc_hash::{FxHashMap as HashMap, FxHasher};

use crate::graph::{Targets, Workspace, store};
use crate::macros::Compilation;
use crate::names::{Crate as NamesCrate, Symbols};
use crate::outline::Outline;
use crate::outline::tree::Profile;
use crate::sema::impls::Index as ImplIndex;

use super::sysroot::{
    DecodeSpec, EnvReads, HEADER_LEN, SYSROOT_VERSION, SnapshotSpec, apply_decoded_crates,
    crate_order, decode_snapshot, encode_snapshot, identity_of, sysroot_crate_order,
};

const DEPS_INDEX_MAGIC: [u8; 8] = *b"HSTDEP01";
const DEPS_INDEX_VERSION: u32 = 1;
const KEPT_DEPS_INDEX: usize = 16;
const DEPS_SCOPE_BIT: u32 = 1 << 30;

/// Cached pinned dependency symbols (when sysroot was uncached), resolved crate scopes, and pre-built module impl indices.
pub(crate) type CachedDeps = (
    Option<Symbols>,
    Vec<(usize, NamesCrate)>,
    HashMap<u32, Arc<ImplIndex>>,
);

/// Returns whether the pinned dependency snapshot cache is enabled.
pub(crate) fn enabled() -> bool {
    super::enabled() && !crate::env_flag("HASTE_NO_DEPS_INDEX")
}

fn is_pinned_dep(outline: &Outline, krate: usize) -> bool {
    outline.is_fixed(krate as u32) && !outline.is_sysroot(krate as u32)
}

/// Returns whether `outline` has at least one pinned dependency whose build script (if any) has settled.
pub(crate) fn can_cache(outline: &Outline) -> bool {
    if !enabled() {
        return false;
    }
    let mut any = false;
    for krate in 0..outline.crates.len() {
        if is_pinned_dep(outline, krate) {
            if !outline.crates[krate].env.known {
                return false;
            }
            any = true;
        }
    }
    any
}

fn pinned_crate_order(workspace: &Workspace, outline: &Outline) -> Vec<usize> {
    crate_order(workspace, outline, |k| is_pinned_dep(outline, k))
}

fn cache_path(key: u64) -> PathBuf {
    store::home()
        .join("deps-index")
        .join(format!("{key:016x}.bin"))
}

fn is_deps_index_cache(path: &Path) -> bool {
    path.extension().and_then(|e| e.to_str()) == Some("bin")
}

fn ref_crates(workspace: &Workspace, outline: &Outline) -> (Vec<usize>, Vec<usize>) {
    let pinned = pinned_crate_order(workspace, outline);
    let sysroot = sysroot_crate_order(workspace, outline);
    let mut refs = Vec::with_capacity(pinned.len() + sysroot.len());
    refs.extend_from_slice(&pinned);
    refs.extend_from_slice(&sysroot);
    (pinned, refs)
}

fn deps_index_key(
    workspace: &Workspace,
    targets: &Targets,
    outline: &Outline,
    pinned: &[usize],
    refs: &[usize],
) -> Option<u64> {
    let mut hasher = FxHasher::default();
    DEPS_INDEX_MAGIC.hash(&mut hasher);
    DEPS_INDEX_VERSION.hash(&mut hasher);
    SYSROOT_VERSION.hash(&mut hasher);
    super::MAGIC.hash(&mut hasher);
    super::VERSION.hash(&mut hasher);
    (crate::profile() == Profile::Release).hash(&mut hasher);
    super::sysroot_key(workspace, targets).hash(&mut hasher);
    super::target_hash(&crate::host_target()).hash(&mut hasher);
    workspace.rustflags.hash(&mut hasher);
    workspace.config_env.hash(&mut hasher);
    pinned.len().hash(&mut hasher);
    refs.len().hash(&mut hasher);

    let mut to_local = vec![None; outline.crates.len()];
    for (pos, &krate) in refs.iter().enumerate() {
        to_local[krate] = Some(pos as u32);
    }
    for &krate in refs {
        let ident = identity_of(workspace, outline, krate, &to_local)?;
        (
            ident.unit_name,
            ident.pkg_name,
            ident.pkg_version,
            ident.side_host,
            ident.role,
            ident.features,
            ident.deps,
        )
            .hash(&mut hasher);
    }
    for &krate in pinned {
        let root = &outline.roots[krate];
        let unit = &workspace.graph.units[root.unit];
        let pkg = &workspace.packages[unit.package];
        let tree = &outline.crates[krate];
        (
            &pkg.locked.source,
            &pkg.locked.checksum,
            &pkg.manifest.dir,
            &root.edition,
            root.proc_macro,
            &root.path,
            store::file_stamp(&root.path),
        )
            .hash(&mut hasher);
        (
            outline.pins.get(krate).copied().flatten(),
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

/// Attempts to load cached settled pinned dependency crates into `outline`, interning symbols into `existing_symbols` if present.
pub(crate) fn try_load(
    workspace: &Workspace,
    targets: &Targets,
    outline: &mut Outline,
    existing_symbols: Option<&Symbols>,
) -> Option<CachedDeps> {
    if !can_cache(outline) {
        return None;
    }
    let (pinned, refs) = ref_crates(workspace, outline);
    if pinned.is_empty() {
        return None;
    }
    let key = deps_index_key(workspace, targets, outline, &pinned, &refs)?;
    let path = cache_path(key);
    let backing = Arc::new(super::Backing::open(&path, HEADER_LEN).ok()?);
    let spec = DecodeSpec {
        magic: DEPS_INDEX_MAGIC,
        version: DEPS_INDEX_VERSION,
        key,
        cached_crates: &pinned,
        ref_crates: &refs,
        scope_bit: DEPS_SCOPE_BIT,
    };
    match decode_snapshot(&backing, &spec, existing_symbols, workspace, outline) {
        Ok(Some((crates, symbols, names, heads, env_reads))) => {
            apply_decoded_crates(outline, crates, env_reads);
            Some((symbols, names, heads))
        }
        Ok(None) | Err(_) => {
            drop(std::fs::remove_file(&path));
            None
        }
    }
}

fn collect_env_reads(outline: &Outline, pinned: &[usize]) -> Vec<EnvReads> {
    pinned
        .iter()
        .map(|&krate| {
            let guard = crate::sync::lock(&outline.crates[krate].env.env_reads);
            let mut reads: EnvReads = guard.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
            reads.sort();
            reads
        })
        .collect()
}

/// Atomically persists settled pinned dependency crates from `compilation` to `$HASTE_HOME/deps-index/` and returns their impl indices.
pub(crate) fn save(
    workspace: &Workspace,
    targets: &Targets,
    compilation: &Compilation,
) -> Option<HashMap<u32, Arc<ImplIndex>>> {
    if !can_cache(&compilation.outline) {
        return None;
    }
    let (pinned, refs) = ref_crates(workspace, &compilation.outline);
    if pinned.is_empty() {
        return None;
    }
    let all_settled = pinned.iter().all(|&krate| {
        let tree = &compilation.outline.crates[krate];
        !tree.partial() && !tree.unsettled()
    });
    if !all_settled {
        return None;
    }
    let key = deps_index_key(workspace, targets, &compilation.outline, &pinned, &refs)?;
    let path = cache_path(key);
    let env_reads = collect_env_reads(&compilation.outline, &pinned);
    let spec = SnapshotSpec {
        magic: DEPS_INDEX_MAGIC,
        version: DEPS_INDEX_VERSION,
        key,
        cached_crates: &pinned,
        ref_crates: &refs,
        env_reads: &env_reads,
    };
    let (bytes, heads) = encode_snapshot(&spec, workspace, compilation)?;
    if let Err(error) = super::backing::write_compressed(&path, &[&bytes]) {
        eprintln!("haste: cannot save deps index cache: {error:#}");
        return Some(heads);
    }
    if let Some(dir) = path.parent() {
        super::prune_stale(dir, &path, KEPT_DEPS_INDEX, is_deps_index_cache);
    }
    Some(heads)
}

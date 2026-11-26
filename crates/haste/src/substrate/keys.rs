//! Deterministic keys and cache paths for toolchain and dependency substrates.

mod items;
mod sysroot;

pub use items::{CrateHashes, file_key, layer_item_key, sysroot_item_key, sysroot_only_item_key};
pub use sysroot::{sysroot_key, toolchain_key};

use std::hash::{Hash as _, Hasher as _};
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use haste_codegen::Target;
use rustc_hash::FxHasher;

use crate::outline::tree::Profile;

/// Returns the toolchain-scoped substrate path under `$HASTE_HOME` / `$XDG_CACHE_HOME/haste` / `$HOME/.cache/haste`.
pub fn toolchain_path(toolchain_hash: u64) -> PathBuf {
    crate::graph::store::home()
        .join("toolchain")
        .join(format!("{toolchain_hash:016x}.substrate"))
}

/// Returns the sysroot outline and name-resolution cache path under `$HASTE_HOME` / `$XDG_CACHE_HOME/haste` / `$HOME/.cache/haste`.
pub fn sysroot_cache_path(sysroot_hash: u64) -> PathBuf {
    crate::graph::store::home()
        .join("toolchain")
        .join(format!("sysroot-{sysroot_hash:016x}.bin"))
}

/// Returns the dependency-scoped substrate path under `$HASTE_HOME` / `$XDG_CACHE_HOME/haste` / `$HOME/.cache/haste`.
pub fn deps_path(deps_hash: u64) -> PathBuf {
    crate::graph::store::home()
        .join("deps")
        .join(format!("{deps_hash:016x}.substrate"))
}

/// Returns the workspace-scoped substrate path under `$HASTE_HOME` / `$XDG_CACHE_HOME/haste` / `$HOME/.cache/haste`.
pub fn workspace_path(ws_hash: u64) -> PathBuf {
    crate::graph::store::home()
        .join("workspace")
        .join(format!("{ws_hash:016x}.substrate"))
}

/// Returns whether substrate caching is enabled (disabled when `HASTE_NO_SUBSTRATE=1`).
pub fn enabled() -> bool {
    std::env::var_os("HASTE_NO_SUBSTRATE").is_none_or(|v| v != "1")
}

/// Computes a 64-bit fingerprint for `target` and the running compiler binary.
pub fn target_hash(target: &Target) -> u64 {
    let mut hasher = FxHasher::default();
    target.triple.hash(&mut hasher);
    target.effort.hash(&mut hasher);
    hash_program(&mut hasher);
    hasher.finish()
}

fn compute_exe_content_hash(exe: &Path) -> Option<u64> {
    let mut file = std::fs::File::open(exe).ok()?;
    let mut hasher = FxHasher::default();
    let mut buf = vec![0_u8; 64 * 1024];
    let mut total = 0_u64;
    loop {
        let read = file.read(&mut buf).ok()?;
        if read == 0 {
            break;
        }
        hasher.write(&buf[..read]);
        total += read as u64;
    }
    total.hash(&mut hasher);
    Some(hasher.finish())
}

fn exe_meta_key(exe: &Path) -> Option<u64> {
    let meta = std::fs::metadata(exe).ok()?;
    let mut hasher = FxHasher::default();
    exe.hash(&mut hasher);
    crate::graph::store::meta_stamp(&meta).hash(&mut hasher);
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt as _;
        meta.ino().hash(&mut hasher);
    }
    Some(hasher.finish())
}

fn exe_content_hash() -> Option<u64> {
    static HASH: OnceLock<Option<u64>> = OnceLock::new();
    *HASH.get_or_init(|| {
        let exe = std::env::current_exe().ok()?;
        let meta_key = exe_meta_key(&exe)?;
        let cache_path = crate::graph::store::home().join("exe_hash.bin");
        if let Ok(bytes) = std::fs::read(&cache_path)
            && bytes.len() == 16
            && super::read_u64(&bytes, 0) == meta_key
        {
            return Some(super::read_u64(&bytes, 8));
        }
        let digest = compute_exe_content_hash(&exe)?;
        let mut out = Vec::with_capacity(16);
        out.extend_from_slice(&meta_key.to_le_bytes());
        out.extend_from_slice(&digest.to_le_bytes());
        drop(crate::graph::store::write_new_bytes(&cache_path, &out));
        Some(digest)
    })
}

fn hash_program(hasher: &mut FxHasher) {
    exe_content_hash().hash(hasher);
}

fn hash_file_meta(path: &Path, hasher: &mut FxHasher) {
    (path, crate::graph::store::file_stamp(path)).hash(hasher);
}

fn hash_cache_header(hasher: &mut FxHasher) {
    (
        super::MAGIC,
        super::VERSION,
        crate::profile() == Profile::Release,
    )
        .hash(hasher);
}

fn hash_deps_units(workspace: &crate::graph::Workspace, hasher: &mut FxHasher) -> bool {
    let mut units: Vec<usize> = workspace
        .graph
        .units
        .iter()
        .enumerate()
        .filter(|(_, unit)| {
            let pkg = &workspace.packages[unit.package];
            !pkg.local() && !pkg.sysroot
        })
        .map(|(idx, _)| idx)
        .collect();
    if units.is_empty() {
        return false;
    }
    units.sort_unstable_by(|&a, &b| {
        let (ua, ub) = (&workspace.graph.units[a], &workspace.graph.units[b]);
        let (pa, pb) = (
            &workspace.packages[ua.package],
            &workspace.packages[ub.package],
        );
        (
            &pa.locked.name,
            &pa.locked.version,
            &pa.locked.source,
            ua.side,
            &ua.features,
            &pa.manifest.dir,
        )
            .cmp(&(
                &pb.locked.name,
                &pb.locked.version,
                &pb.locked.source,
                ub.side,
                &ub.features,
                &pb.manifest.dir,
            ))
    });
    for idx in units {
        let unit = &workspace.graph.units[idx];
        let pkg = &workspace.packages[unit.package];
        let locked = &pkg.locked;
        (&locked.name, locked.version.to_string(), &locked.source).hash(hasher);
        (
            matches!(unit.side, crate::graph::Side::Host),
            &pkg.manifest.dir,
            &unit.features,
        )
            .hash(hasher);
        if let Some(script) = &unit.script {
            script.key_parts().hash(hasher);
        }
    }
    true
}

/// Computes a workspace-scoped key for non-local, non-sysroot fixed units, or `None` when none exist.
pub fn deps_key(
    workspace: &crate::graph::Workspace,
    target: &Target,
    toolchain_hash: u64,
) -> Option<u64> {
    let mut hasher = FxHasher::default();
    hash_cache_header(&mut hasher);
    (toolchain_hash, target_hash(target)).hash(&mut hasher);
    hash_deps_units(workspace, &mut hasher).then(|| hasher.finish())
}

fn fixed_files_paths(
    workspace: &crate::graph::Workspace,
    triple: &str,
) -> (PathBuf, Option<PathBuf>) {
    let home = crate::graph::store::home();
    let mut sys_hasher = FxHasher::default();
    hash_cache_header(&mut sys_hasher);
    (triple, crate::graph::store::program()).hash(&mut sys_hasher);
    hash_program(&mut sys_hasher);
    for unit in workspace
        .graph
        .units
        .iter()
        .filter(|u| workspace.packages[u.package].sysroot)
    {
        let pkg = &workspace.packages[unit.package];
        (&pkg.locked.name, &pkg.manifest.dir, &unit.features).hash(&mut sys_hasher);
    }
    let sys_key = sys_hasher.finish();
    let sys_path = home.join("toolchain").join(format!("{sys_key:016x}.files"));
    let mut dep_hasher = FxHasher::default();
    sys_key.hash(&mut dep_hasher);
    let dep_path = hash_deps_units(workspace, &mut dep_hasher).then(|| {
        home.join("deps")
            .join(format!("{:016x}.files", dep_hasher.finish()))
    });
    (sys_path, dep_path)
}

fn preload_paths(outline: &crate::outline::Outline, list_paths: impl Iterator<Item = PathBuf>) {
    let mut paths = Vec::new();
    for path in list_paths {
        if let Ok(text) = std::fs::read_to_string(path) {
            paths.extend(text.lines().filter(|l| !l.is_empty()).map(PathBuf::from));
        }
    }
    if !paths.is_empty() {
        outline.cache.read_all(&paths);
    }
}

/// Pre-reads and scans cached sysroot and fixed-dependency `.rs` files in parallel into `outline.cache`.
pub fn preload_fixed_files(
    workspace: &crate::graph::Workspace,
    triple: &str,
    outline: &crate::outline::Outline,
) {
    if !enabled() {
        return;
    }
    let (sys_path, dep_path) = fixed_files_paths(workspace, triple);
    preload_paths(outline, std::iter::once(sys_path).chain(dep_path));
}

/// Pre-reads and scans cached fixed-dependency `.rs` files in parallel into `outline.cache`, skipping sysroot.
pub fn preload_dep_files(
    workspace: &crate::graph::Workspace,
    triple: &str,
    outline: &crate::outline::Outline,
) {
    if !enabled() {
        return;
    }
    let (_, dep_path) = fixed_files_paths(workspace, triple);
    preload_paths(outline, dep_path.into_iter());
}

fn collect_fixed_file_list(outline: &crate::outline::Outline, want_sysroot: bool) -> String {
    let mut list: Vec<&str> = outline
        .crates
        .iter()
        .enumerate()
        .filter(|&(k, _)| {
            outline.is_fixed(k as u32) && (outline.is_sysroot(k as u32) == want_sysroot)
        })
        .flat_map(|(_, tree)| {
            tree.files
                .iter()
                .filter(|f| !f.expanded)
                .filter_map(|f| f.source.path.to_str())
        })
        .collect();
    list.sort_unstable();
    list.dedup();
    list.join("\n")
}

/// Saves non-expanded sysroot and fixed-dependency `.rs` file lists so later runs scan them in one parallel batch.
pub fn save_fixed_files(
    workspace: &crate::graph::Workspace,
    triple: &str,
    outline: &crate::outline::Outline,
) {
    if !enabled() {
        return;
    }
    let (sys_path, dep_path) = fixed_files_paths(workspace, triple);
    if !sys_path.exists() {
        let sys_list = collect_fixed_file_list(outline, true);
        if !sys_list.is_empty() {
            drop(crate::graph::store::write_new_bytes(
                &sys_path,
                sys_list.as_bytes(),
            ));
        }
    }
    if let Some(dep_file) = dep_path {
        let dep_list = collect_fixed_file_list(outline, false);
        if !dep_list.is_empty()
            && std::fs::read(&dep_file).ok().as_deref() != Some(dep_list.as_bytes())
        {
            drop(crate::graph::store::write_new_bytes(
                &dep_file,
                dep_list.as_bytes(),
            ));
        }
    }
}

/// Computes a stable workspace-root key for local workspace member substrates, or `None` when no local units exist.
pub fn workspace_key(
    workspace: &crate::graph::Workspace,
    target: &Target,
    toolchain_hash: u64,
) -> Option<u64> {
    let dirs: Vec<&Path> = workspace
        .graph
        .units
        .iter()
        .filter(|unit| workspace.packages[unit.package].local())
        .map(|unit| workspace.packages[unit.package].manifest.dir.as_path())
        .collect();
    if dirs.is_empty() {
        return None;
    }
    let root = dirs
        .iter()
        .find_map(|dir| crate::graph::workspace_root(dir).ok())
        .filter(|root| dirs.iter().all(|dir| dir.starts_with(root)));
    let mut dirs: Vec<&Path> = dirs
        .into_iter()
        .map(|dir| {
            root.as_deref()
                .and_then(|r| dir.strip_prefix(r).ok())
                .unwrap_or(dir)
        })
        .collect();
    dirs.sort_unstable();
    dirs.dedup();
    let mut hasher = FxHasher::default();
    hash_cache_header(&mut hasher);
    (toolchain_hash, target_hash(target), dirs).hash(&mut hasher);
    Some(hasher.finish())
}

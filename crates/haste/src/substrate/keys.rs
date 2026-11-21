//! Deterministic keys and cache paths for toolchain and dependency substrates.

use std::hash::{Hash as _, Hasher as _};
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use haste_codegen::Target;
use rustc_hash::FxHasher;

use crate::outline::tree::Profile;

type Stamp = (u64, i128);

/// Returns the toolchain-scoped substrate path under `$HASTE_HOME` / `$XDG_CACHE_HOME/haste` / `$HOME/.cache/haste`.
pub fn toolchain_path(toolchain_hash: u64) -> PathBuf {
    crate::graph::store::home().join("toolchain").join(format!("{toolchain_hash:016x}.substrate"))
}

/// Returns the dependency-scoped substrate path under `$HASTE_HOME` / `$XDG_CACHE_HOME/haste` / `$HOME/.cache/haste`.
pub fn deps_path(deps_hash: u64) -> PathBuf {
    crate::graph::store::home().join("deps").join(format!("{deps_hash:016x}.substrate"))
}

/// Returns the workspace-scoped substrate path under `$HASTE_HOME` / `$XDG_CACHE_HOME/haste` / `$HOME/.cache/haste`.
pub fn workspace_path(ws_hash: u64) -> PathBuf {
    crate::graph::store::home().join("workspace").join(format!("{ws_hash:016x}.substrate"))
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

/// Per-crate content hashes computed bottom-up over `Outline` dependencies for local-crate substrate keys.
pub struct CrateHashes {
    hashes: Vec<u64>,
}

impl CrateHashes {
    /// Computes bottom-up Merkle hashes for every crate in `outline` under `(toolchain_hash, target_hash)`.
    pub fn compute(outline: &crate::outline::Outline, toolchain_hash: u64, target_hash: u64) -> Self {
        let n = outline.crates.len();
        let mut hashes = vec![0u64; n];
        let mut state = vec![0u8; n];
        for idx in 0..n {
            hash_crate_rec(outline, idx, toolchain_hash, target_hash, &mut hashes, &mut state);
        }
        Self { hashes }
    }

    /// Returns the computed content hash for `krate`.
    pub fn get(&self, krate: usize) -> Option<u64> {
        self.hashes.get(krate).copied()
    }
}

fn hash_crate_rec(
    outline: &crate::outline::Outline,
    idx: usize,
    toolchain_hash: u64,
    target_hash: u64,
    hashes: &mut [u64],
    state: &mut [u8],
) -> u64 {
    if state[idx] == 2 {
        return hashes[idx];
    }
    if state[idx] == 1 {
        return 0;
    }
    state[idx] = 1;
    let mut hasher = FxHasher::default();
    toolchain_hash.hash(&mut hasher);
    target_hash.hash(&mut hasher);
    (crate::profile() == Profile::Release).hash(&mut hasher);
    if let (Some(krate), Some(root)) = (outline.crates.get(idx), outline.roots.get(idx)) {
        krate.env.fixed.hash(&mut hasher);
        krate.env.release.hash(&mut hasher);
        krate.env.overflow_checks().hash(&mut hasher);
        root.name.hash(&mut hasher);
        (root.role as u8).hash(&mut hasher);
        krate.env.target.triple.hash(&mut hasher);
        if krate.env.fixed {
            outline.pins.get(idx).copied().flatten().hash(&mut hasher);
        } else {
            krate.env.features.hash(&mut hasher);
            krate.env.vars.hash(&mut hasher);
            if let Some(script) = &krate.env.script {
                (&script.names, &script.pairs, &script.vars).hash(&mut hasher);
            }
            let base_dir = root.path.parent().unwrap_or(Path::new(""));
            let mut files: Vec<(&Path, u64)> = krate
                .files
                .iter()
                .filter(|f| !f.expanded)
                .map(|f| {
                    let rel = f.source.path.strip_prefix(base_dir).unwrap_or(&f.source.path);
                    let mut fh = FxHasher::default();
                    f.source.text.hash(&mut fh);
                    (rel, fh.finish())
                })
                .collect();
            files.sort_unstable_by(|a, b| a.0.cmp(b.0));
            files.hash(&mut hasher);
            let mut deps: Vec<(&str, u64)> = outline
                .deps
                .get(idx)
                .into_iter()
                .flatten()
                .map(|(name, dep)| (name.as_str(), hash_crate_rec(outline, *dep, toolchain_hash, target_hash, hashes, state)))
                .collect();
            deps.sort_unstable_by(|a, b| a.0.cmp(b.0));
            deps.hash(&mut hasher);
        }
    }
    let h = hasher.finish();
    hashes[idx] = h;
    state[idx] = 2;
    h
}

/// Computes a deterministic key for any fixed item (sysroot or non-local dependency), or `None` for local items.
pub fn sysroot_item_key(outline: &crate::outline::Outline, item: crate::names::ItemRef) -> Option<u64> {
    item_key_with_crates(outline, item, true, None)
}

/// Computes a workspace-invariant key for a sysroot-only item, or `None` for non-sysroot items.
pub fn sysroot_only_item_key(outline: &crate::outline::Outline, item: crate::names::ItemRef) -> Option<u64> {
    item_key_with_crates(outline, item, false, None)
}
/// Computes a deterministic key for any item (fixed or local when `crate_hashes` is provided).
pub fn item_key_with_crates(
    outline: &crate::outline::Outline,
    item: crate::names::ItemRef,
    include_deps: bool,
    crate_hashes: Option<&CrateHashes>,
) -> Option<u64> {
    let krate = outline.crates.get(item.krate as usize)?;
    if (!krate.env.fixed && crate_hashes.is_none()) || (!include_deps && !krate.env.release) {
        return None;
    }
    let file = krate.files.get(item.file as usize)?;
    let prefix = if krate.env.fixed { file_key(outline, item.krate, item.file)? } else { hash_file(outline, item.krate, file, crate_hashes)? };
    let (base_item, sub_idx) = match item.outer() {
        Some((outer, sub)) => (outer.item, Some(sub)),
        None => (item.item, None),
    };
    let entry = file.items.items.get(base_item as usize)?;
    let mut hasher = FxHasher::default();
    prefix.hash(&mut hasher);
    item.item.hash(&mut hasher);
    sub_idx.hash(&mut hasher);
    entry.start.hash(&mut hasher);
    entry.kind.hash(&mut hasher);
    if let Some(name_tok) = entry.name {
        file.source.text(name_tok).hash(&mut hasher);
    }
    Some(hasher.finish())
}

/// Computes the identity key of file `file` of fixed crate `krate`, once per file, or `None` for local or unkeyed files.
pub fn file_key(outline: &crate::outline::Outline, krate: u32, file: u32) -> Option<u64> {
    let tree = outline.crates.get(krate as usize).filter(|tree| tree.env.fixed)?;
    let at = tree.files.get(file as usize)?;
    at.key(krate, || hash_file(outline, krate, at, None))
}

fn hash_file(outline: &crate::outline::Outline, idx: u32, file: &crate::outline::tree::File, crate_hashes: Option<&CrateHashes>) -> Option<u64> {
    let krate = outline.crates.get(idx as usize)?;
    let root = outline.roots.get(idx as usize)?;
    if file.expanded && file.expand_origin == 0 {
        return None;
    }
    let base_dir = root.path.parent().unwrap_or(Path::new(""));
    let rel_path = file.source.path.strip_prefix(base_dir).unwrap_or(&file.source.path);
    let mut hasher = FxHasher::default();
    krate.env.release.hash(&mut hasher);
    krate.env.overflow_checks().hash(&mut hasher);
    (crate::profile() == Profile::Release).hash(&mut hasher);
    if !krate.env.fixed {
        crate_hashes?.get(idx as usize)?.hash(&mut hasher);
    } else if !krate.env.release {
        let pin = outline.pins.get(idx as usize).copied().flatten()?;
        pin.hash(&mut hasher);
        krate.env.features.hash(&mut hasher);
        if let Some(script) = &krate.env.script {
            (&script.names, &script.pairs, &script.vars).hash(&mut hasher);
        }
        krate.env.vars.hash(&mut hasher);
        base_dir.file_name().hash(&mut hasher);
        base_dir.parent().and_then(Path::file_name).hash(&mut hasher);
    }
    root.name.hash(&mut hasher);
    (root.role as u8).hash(&mut hasher);
    krate.env.target.triple.hash(&mut hasher);
    rel_path.hash(&mut hasher);
    file.expanded.hash(&mut hasher);
    if file.expanded {
        file.expand_origin.hash(&mut hasher);
        file.source.offset.hash(&mut hasher);
        file.source.text.hash(&mut hasher);
    }
    Some(hasher.finish())
}

/// Computes whether a fixed item belongs to the toolchain layer, and its key in that layer or the dependency layer, or `None` for local items.
pub fn layer_item_key(outline: &crate::outline::Outline, item: crate::names::ItemRef) -> Option<(bool, u64)> {
    if !outline.is_fixed(item.krate) {
        return None;
    }
    let is_sysroot = outline.is_sysroot(item.krate);
    let key = if is_sysroot { sysroot_only_item_key(outline, item)? } else { sysroot_item_key(outline, item)? };
    Some((is_sysroot, key))
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

/// Hashes the running compiler binary's contents into `hasher`.
fn hash_program(hasher: &mut FxHasher) {
    exe_content_hash().hash(hasher);
}

fn hash_file_meta(path: &Path, hasher: &mut FxHasher) {
    (path, crate::graph::store::file_stamp(path)).hash(hasher);
}

fn sysroot_library_roots(workspace: &crate::graph::Workspace, units: &[usize]) -> Vec<PathBuf> {
    let mut roots: Vec<PathBuf> = units
        .iter()
        .map(|&idx| {
            let dir = &workspace.packages[workspace.graph.units[idx].package].manifest.dir;
            dir.ancestors().find(|a| a.file_name().is_some_and(|n| n == "library")).unwrap_or(dir.as_path()).to_owned()
        })
        .collect();
    roots.sort();
    roots.dedup();
    roots
}

fn scan_sysroot_dir(dir: &Path) -> Vec<(PathBuf, Option<Stamp>)> {
    let Ok(entries) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut out: Vec<_> = entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let path = entry.path();
            if matches!(path.file_name().and_then(|n| n.to_str()), Some("target" | ".git")) {
                return None;
            }
            let ft = entry.file_type().ok();
            if ft.is_some_and(|ft| ft.is_dir()) {
                return Some((path, None));
            }
            let is_rs = path.extension().and_then(|e| e.to_str()) == Some("rs");
            let meta = if ft.is_some_and(|ft| ft.is_file()) {
                is_rs.then(|| entry.metadata().ok()).flatten()?
            } else {
                std::fs::metadata(&path).ok()?
            };
            if meta.is_dir() {
                Some((path, None))
            } else if is_rs {
                Some((path, Some(crate::graph::store::meta_stamp(&meta))))
            } else {
                None
            }
        })
        .collect();
    out.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    out
}

fn walk_sysroot_dir(dir: &Path, out: &mut Vec<(PathBuf, Stamp)>) {
    for (path, stamp) in scan_sysroot_dir(dir) {
        match stamp {
            Some(s) => out.push((path, s)),
            None => walk_sysroot_dir(&path, out),
        }
    }
}

fn hash_sysroot_roots(roots: Vec<PathBuf>, hasher: &mut FxHasher) {
    let mut work = Vec::new();
    for root in roots {
        for (p, s) in scan_sysroot_dir(&root) {
            if s.is_some() {
                work.push((p, s));
            } else {
                work.extend(scan_sysroot_dir(&p));
            }
        }
    }
    let dirs: Vec<(usize, &Path)> = work.iter().enumerate().filter(|(_, (_, s))| s.is_none()).map(|(i, (p, _))| (i, p.as_path())).collect();
    let mut chunks = vec![Vec::new(); work.len()];
    for (slot, items) in crate::sync::parallel_map(dirs, |(slot, dir)| {
        let mut items = Vec::new();
        walk_sysroot_dir(dir, &mut items);
        (slot, items)
    }) {
        chunks[slot] = items;
    }
    for (i, (path, stamp)) in work.into_iter().enumerate() {
        match stamp {
            Some(s) => (path, s).hash(hasher),
            None => chunks[i].iter().for_each(|e| e.hash(hasher)),
        }
    }
}

/// Computes a workspace-invariant toolchain key from sysroot packages, target, and compiler binary.
pub fn toolchain_key(workspace: &crate::graph::Workspace, target: &Target) -> u64 {
    let mut sysroot_units: Vec<(&str, crate::graph::Side, usize)> = workspace
        .graph
        .units
        .iter()
        .enumerate()
        .filter(|(_, unit)| workspace.packages[unit.package].sysroot)
        .map(|(idx, unit)| (workspace.packages[unit.package].locked.name.as_str(), unit.side, idx))
        .collect();
    sysroot_units.sort_unstable_by(|a, b| (a.0, a.1, a.2).cmp(&(b.0, b.1, b.2)));
    let mut hasher = FxHasher::default();
    target.triple.hash(&mut hasher);
    target.effort.hash(&mut hasher);
    crate::graph::store::program().hash(&mut hasher);
    hash_program(&mut hasher);
    super::MAGIC.hash(&mut hasher);
    super::VERSION.hash(&mut hasher);
    let mut unit_indices = Vec::with_capacity(sysroot_units.len());
    for (_, _, idx) in sysroot_units {
        unit_indices.push(idx);
        let unit = &workspace.graph.units[idx];
        let pkg = &workspace.packages[unit.package];
        let locked = &pkg.locked;
        locked.name.hash(&mut hasher);
        locked.version.to_string().hash(&mut hasher);
        locked.source.hash(&mut hasher);
        matches!(unit.side, crate::graph::Side::Host).hash(&mut hasher);
        if let Some(lib) = &pkg.manifest.lib {
            hash_file_meta(lib, &mut hasher);
        }
        for feat in &unit.features {
            feat.hash(&mut hasher);
        }
    }
    hash_sysroot_roots(sysroot_library_roots(workspace, &unit_indices), &mut hasher);
    hasher.finish()
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
        let (pa, pb) = (&workspace.packages[ua.package], &workspace.packages[ub.package]);
        (&pa.locked.name, &pa.locked.version, &pa.locked.source, ua.side, &ua.features, &pa.manifest.dir)
            .cmp(&(&pb.locked.name, &pb.locked.version, &pb.locked.source, ub.side, &ub.features, &pb.manifest.dir))
    });
    for idx in units {
        let unit = &workspace.graph.units[idx];
        let pkg = &workspace.packages[unit.package];
        let locked = &pkg.locked;
        (&locked.name, locked.version.to_string(), &locked.source).hash(hasher);
        (matches!(unit.side, crate::graph::Side::Host), &pkg.manifest.dir, &unit.features).hash(hasher);
        if let Some(script) = &unit.script {
            (&script.names, &script.pairs, &script.vars).hash(hasher);
        }
    }
    true
}

/// Computes a workspace-scoped key for non-local, non-sysroot fixed units, or `None` when none exist.
pub fn deps_key(workspace: &crate::graph::Workspace, target: &Target, toolchain_hash: u64) -> Option<u64> {
    let mut hasher = FxHasher::default();
    (super::MAGIC, super::VERSION, toolchain_hash, target_hash(target), crate::profile() == Profile::Release).hash(&mut hasher);
    hash_deps_units(workspace, &mut hasher).then(|| hasher.finish())
}

fn fixed_files_paths(workspace: &crate::graph::Workspace, triple: &str) -> (PathBuf, Option<PathBuf>) {
    let home = crate::graph::store::home();
    let mut sys_hasher = FxHasher::default();
    (super::MAGIC, super::VERSION, triple, crate::profile() == Profile::Release, crate::graph::store::program()).hash(&mut sys_hasher);
    for unit in workspace.graph.units.iter().filter(|u| workspace.packages[u.package].sysroot) {
        let pkg = &workspace.packages[unit.package];
        (&pkg.locked.name, &pkg.manifest.dir, &unit.features).hash(&mut sys_hasher);
    }
    let sys_key = sys_hasher.finish();
    let sys_path = home.join("toolchain").join(format!("{sys_key:016x}.files"));
    let mut dep_hasher = FxHasher::default();
    sys_key.hash(&mut dep_hasher);
    let dep_path = hash_deps_units(workspace, &mut dep_hasher).then(|| home.join("deps").join(format!("{:016x}.files", dep_hasher.finish())));
    (sys_path, dep_path)
}

/// Pre-reads and scans cached sysroot and fixed-dependency `.rs` files in parallel into `outline.cache`.
pub fn preload_fixed_files(workspace: &crate::graph::Workspace, triple: &str, outline: &crate::outline::Outline) {
    if !enabled() {
        return;
    }
    let (sys_path, dep_path) = fixed_files_paths(workspace, triple);
    let mut paths = Vec::new();
    for path in std::iter::once(sys_path).chain(dep_path) {
        if let Ok(text) = std::fs::read_to_string(path) {
            paths.extend(text.lines().filter(|l| !l.is_empty()).map(PathBuf::from));
        }
    }
    if !paths.is_empty() {
        outline.cache.read_all(&paths);
    }
}

fn collect_fixed_file_list(outline: &crate::outline::Outline, want_sysroot: bool) -> String {
    let mut list: Vec<&str> = outline
        .crates
        .iter()
        .enumerate()
        .filter(|&(k, _)| outline.is_fixed(k as u32) && (outline.is_sysroot(k as u32) == want_sysroot))
        .flat_map(|(_, tree)| tree.files.iter().filter(|f| !f.expanded).filter_map(|f| f.source.path.to_str()))
        .collect();
    list.sort_unstable();
    list.dedup();
    list.join("\n")
}

/// Saves non-expanded sysroot and fixed-dependency `.rs` file lists so later runs scan them in one parallel batch.
pub fn save_fixed_files(workspace: &crate::graph::Workspace, triple: &str, outline: &crate::outline::Outline) {
    if !enabled() {
        return;
    }
    let (sys_path, dep_path) = fixed_files_paths(workspace, triple);
    if !sys_path.exists() {
        let sys_list = collect_fixed_file_list(outline, true);
        if !sys_list.is_empty() {
            drop(crate::graph::store::write_new_bytes(&sys_path, sys_list.as_bytes()));
        }
    }
    if let Some(dep_file) = dep_path {
        let dep_list = collect_fixed_file_list(outline, false);
        if !dep_list.is_empty() && std::fs::read(&dep_file).ok().as_deref() != Some(dep_list.as_bytes()) {
            drop(crate::graph::store::write_new_bytes(&dep_file, dep_list.as_bytes()));
        }
    }
}

/// Computes a stable workspace-root key for local workspace member substrates, or `None` when no local units exist.
pub fn workspace_key(workspace: &crate::graph::Workspace, target: &Target, toolchain_hash: u64) -> Option<u64> {
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
        .map(|dir| root.as_deref().and_then(|r| dir.strip_prefix(r).ok()).unwrap_or(dir))
        .collect();
    dirs.sort_unstable();
    dirs.dedup();
    let mut hasher = FxHasher::default();
    super::MAGIC.hash(&mut hasher);
    super::VERSION.hash(&mut hasher);
    toolchain_hash.hash(&mut hasher);
    target_hash(target).hash(&mut hasher);
    (crate::profile() == Profile::Release).hash(&mut hasher);
    dirs.hash(&mut hasher);
    Some(hasher.finish())
}

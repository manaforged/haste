use std::hash::{Hash as _, Hasher as _};
use std::path::Path;

use rustc_hash::FxHasher;

use crate::outline::tree::Profile;

/// Per-crate content hashes computed bottom-up over `Outline` dependencies for local-crate substrate keys.
pub struct CrateHashes {
    hashes: Vec<u64>,
}

impl CrateHashes {
    /// Computes bottom-up Merkle hashes for every crate in `outline` under `(toolchain_hash, target_hash)`.
    pub fn compute(
        outline: &crate::outline::Outline,
        toolchain_hash: u64,
        target_hash: u64,
    ) -> Self {
        let n = outline.crates.len();
        let mut hashes = vec![0u64; n];
        let mut state = vec![0u8; n];
        for idx in 0..n {
            hash_crate_rec(
                outline,
                idx,
                toolchain_hash,
                target_hash,
                &mut hashes,
                &mut state,
            );
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
        krate.env.debug_assertions().hash(&mut hasher);
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
                    let rel = f
                        .source
                        .path
                        .strip_prefix(base_dir)
                        .unwrap_or(&f.source.path);
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
                .map(|(name, dep)| {
                    (
                        name.as_str(),
                        hash_crate_rec(outline, *dep, toolchain_hash, target_hash, hashes, state),
                    )
                })
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
pub fn sysroot_item_key(
    outline: &crate::outline::Outline,
    item: crate::names::ItemRef,
) -> Option<u64> {
    item_key_with_crates(outline, item, true, None)
}

/// Computes a workspace-invariant key for a sysroot-only item, or `None` for non-sysroot items.
pub fn sysroot_only_item_key(
    outline: &crate::outline::Outline,
    item: crate::names::ItemRef,
) -> Option<u64> {
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
    let prefix = if krate.env.fixed {
        file_key(outline, item.krate, item.file)?
    } else {
        hash_file(outline, item.krate, file, crate_hashes)?
    };
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
    let tree = outline
        .crates
        .get(krate as usize)
        .filter(|tree| tree.env.fixed)?;
    let at = tree.files.get(file as usize)?;
    at.key(krate, || hash_file(outline, krate, at, None))
}

fn hash_file(
    outline: &crate::outline::Outline,
    idx: u32,
    file: &crate::outline::tree::File,
    crate_hashes: Option<&CrateHashes>,
) -> Option<u64> {
    let krate = outline.crates.get(idx as usize)?;
    let root = outline.roots.get(idx as usize)?;
    if file.expanded && file.expand_origin == 0 {
        return None;
    }
    let base_dir = root.path.parent().unwrap_or(Path::new(""));
    let rel_path = file
        .source
        .path
        .strip_prefix(base_dir)
        .unwrap_or(&file.source.path);
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
            script.key_parts().hash(&mut hasher);
        }
        krate.env.vars.hash(&mut hasher);
        base_dir.file_name().hash(&mut hasher);
        base_dir
            .parent()
            .and_then(Path::file_name)
            .hash(&mut hasher);
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
pub fn layer_item_key(
    outline: &crate::outline::Outline,
    item: crate::names::ItemRef,
) -> Option<(bool, u64)> {
    if !outline.is_fixed(item.krate) {
        return None;
    }
    let is_sysroot = outline.is_sysroot(item.krate);
    let key = if is_sysroot {
        sysroot_only_item_key(outline, item)?
    } else {
        sysroot_item_key(outline, item)?
    };
    Some((is_sysroot, key))
}

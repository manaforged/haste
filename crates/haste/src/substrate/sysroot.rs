//! Persistent disk cache for settled sysroot crate outlines, macros, and resolved name scopes.

mod impls;
mod names;
mod outline;
pub(crate) mod tokens;
mod wire;

use std::path::Path;
use std::sync::Arc;

use rustc_hash::FxHashMap as HashMap;

use crate::graph::{Side, Targets, Workspace};
use crate::macros::Compilation;
use crate::names::{Crate as NamesCrate, Symbols};
use crate::outline::{Outline, Role};
use crate::sema::impls::Index as ImplIndex;

pub(crate) const SYSROOT_VERSION: u32 = 10;
const SYSROOT_MAGIC: [u8; 8] = *b"HSTSYS01";
const HEADER_LEN: usize = 32;
const KEPT_SYSROOTS: usize = 2;

#[derive(Clone, Debug, PartialEq, Eq)]
struct CrateIdentity {
    unit_name: String,
    pkg_name: String,
    pkg_version: String,
    side_host: bool,
    role: u8,
    features: Vec<String>,
    deps: Vec<(String, u32)>,
}

const ROLES: [Role; 6] = [
    Role::Lib,
    Role::Bin,
    Role::Example,
    Role::Test,
    Role::Script,
    Role::Bench,
];

fn role_tag(role: Role) -> u8 {
    ROLES.iter().position(|&r| r == role).unwrap_or(0) as u8
}

fn sysroot_crate_order(workspace: &Workspace, outline: &Outline) -> Vec<usize> {
    let mut crates: Vec<usize> = (0..outline.crates.len())
        .filter(|&k| outline.is_sysroot(k as u32))
        .collect();
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

fn identity_of(
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

fn match_crates(
    identities: &[CrateIdentity],
    workspace: &Workspace,
    outline: &Outline,
) -> Option<Vec<usize>> {
    let local = sysroot_crate_order(workspace, outline);
    if local.len() != identities.len() {
        return None;
    }
    let mut to_local = vec![None; outline.crates.len()];
    for (pos, &krate) in local.iter().enumerate() {
        to_local[krate] = Some(pos as u32);
    }
    for (pos, ident) in identities.iter().enumerate() {
        let current = identity_of(workspace, outline, local[pos], &to_local)?;
        if &current != ident {
            return None;
        }
    }
    Some(local)
}

fn is_sysroot_cache(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return false;
    };
    name.starts_with("sysroot-") && name.ends_with(".bin")
}

/// Cached sysroot name symbols, resolved crate scopes, and pre-built module impl indices.
pub(crate) type CachedSysroot = (
    Symbols,
    Vec<(usize, NamesCrate)>,
    HashMap<u32, Arc<ImplIndex>>,
);

/// Attempts to load cached settled sysroot crates into `outline` and return their `Symbols`, `Names`, and impl indices.
pub(crate) fn try_load(
    workspace: &Workspace,
    targets: &Targets,
    outline: &mut Outline,
) -> Option<CachedSysroot> {
    if !super::enabled() {
        return None;
    }
    let key = super::sysroot_key(workspace, targets);
    let path = super::sysroot_cache_path(key);
    let backing = Arc::new(super::Backing::open(&path, HEADER_LEN).ok()?);
    match wire::decode_sysroot(&backing, key, workspace, outline) {
        Ok(Some((crates, symbols, names, heads))) => {
            for (krate, mut tree) in crates {
                tree.env = std::mem::take(&mut outline.crates[krate].env);
                outline.crates[krate] = tree;
            }
            Some((symbols, names, heads))
        }
        Ok(None) | Err(_) => {
            drop(std::fs::remove_file(&path));
            None
        }
    }
}

/// Atomically persists the settled sysroot crates from `compilation` to the toolchain cache directory and returns their impl indices.
pub(crate) fn save(
    workspace: &Workspace,
    targets: &Targets,
    compilation: &Compilation,
) -> Option<Arc<HashMap<u32, Arc<ImplIndex>>>> {
    if !super::enabled() {
        return None;
    }
    let sysroot_crates = sysroot_crate_order(workspace, &compilation.outline);
    if sysroot_crates.is_empty() {
        return None;
    }
    let key = super::sysroot_key(workspace, targets);
    let path = super::sysroot_cache_path(key);
    let (bytes, heads) = wire::encode_sysroot(key, workspace, compilation, &sysroot_crates)?;
    if let Err(error) = super::backing::write_compressed(&path, &[&bytes]) {
        eprintln!("haste: cannot save sysroot cache: {error:#}");
        return Some(heads);
    }
    if let Some(dir) = path.parent() {
        super::prune_stale(dir, &path, KEPT_SYSROOTS, is_sysroot_cache);
    }
    Some(heads)
}

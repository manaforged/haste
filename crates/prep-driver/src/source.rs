use std::path::{Component, Path};

/// The directory whose parent starts a source path in a key, so one crate's file prints alike wherever its package lies.
const SOURCE_DIR: &str = "src";

/// What a key appends to the name of a crate from the toolchain's sysroot, so that no sysroot crate shares a key with a registry crate of its name.
const SYSROOT_MARK: &str = "!";

/// The components of `path` from the parent of its last `src` directory on, joined by slashes.
pub fn normalize(path: &Path) -> Option<String> {
    let parts: Vec<Component<'_>> = path.components().collect();
    let at = parts.iter().rposition(|part| part.as_os_str() == SOURCE_DIR)?;
    let kept = parts.get(at.checked_sub(1)?..)?.iter().map(|part| part.as_os_str().to_string_lossy());
    Some(kept.collect::<Vec<_>>().join("/"))
}

/// The name a key gives crate `name`, marked when the crate comes from the toolchain's sysroot.
pub fn crate_key(name: &str, sysroot: bool) -> String {
    if sysroot { format!("{name}{SYSROOT_MARK}") } else { name.to_string() }
}

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

/// What joins the key of a function to the name of a static its body declares.
const NESTED: &str = "::";

/// The key of a static named `name` that the body of the function keyed `parent` declares.
pub fn nested(parent: &str, name: &str) -> String {
    format!("{parent}{NESTED}{name}")
}

/// The key of the function and the name of the static its body declares that `key` names, when it names one.
pub fn parent_of(key: &str) -> Option<(&str, &str)> {
    key.rsplit_once(NESTED)
}

use std::hash::BuildHasher as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context as _, Result};
use rustc_hash::FxBuildHasher;

use super::cargo::{self, Toolchain};

macro_rules! sources {
    ($($path:literal),* $(,)?) => {
        [$(($path, include_str!(concat!("../../../prep-driver/", $path)))),*]
    };
}

/// The source of the prep driver, which Haste builds once per toolchain into its store.
const SOURCES: [(&str, &str); 18] = sources!(
    "Cargo.toml",
    "Cargo.lock",
    "build.rs",
    "src/main.rs",
    "src/abi.rs",
    "src/args.rs",
    "src/elf.rs",
    "src/error.rs",
    "src/export.rs",
    "src/key.rs",
    "src/layout.rs",
    "src/link.rs",
    "src/object.rs",
    "src/print.rs",
    "src/record.rs",
    "src/source.rs",
    "src/uses.rs",
    "src/wrap.rs",
);

const DRIVER: &str = "driver";
const BINARY: &str = "target/release/prep-driver";
const TARGET: &str = "target";
const UNITS: &str = "units";
const CARGO_MANIFEST: &str = "Cargo.toml";
const LINK: &str = "link";
const BOOTSTRAP: (&str, &str) = ("RUSTC_BOOTSTRAP", "1");
const WRAPPER: &str = "RUSTC_WRAPPER";
const PREP_DIR: &str = "PREP_DIR";

/// The library file a driver run leaves in its directory, and the manifest beside it.
pub(super) const LIBRARY: &str = "library.so";
/// The manifest the driver's link step writes beside `LIBRARY`.
pub(super) const MANIFEST: &str = "library.so.toml";

/// The hash of the driver's source, which every unit and library a driver build makes depends on.
pub(super) fn sources() -> u64 {
    FxBuildHasher.hash_one(SOURCES)
}

/// The prep driver binary for `toolchain`, built from the source Haste carries into the store on first use.
fn binary(toolchain: &Toolchain, store: &Path) -> Result<PathBuf> {
    let key = FxBuildHasher.hash_one((sources(), &toolchain.version));
    let dir = store.join(DRIVER).join(format!("{key:016x}"));
    let binary = dir.join(BINARY);
    if binary.is_file() {
        return Ok(binary);
    }
    for (path, text) in SOURCES {
        let file = dir.join(path);
        std::fs::create_dir_all(file.parent().context("a driver source with no directory")?)?;
        std::fs::write(&file, text)?;
    }
    let mut build = cargo::release(toolchain, &dir.join(CARGO_MANIFEST), &dir.join(TARGET));
    cargo::output(build.env(BOOTSTRAP.0, BOOTSTRAP.1)).context("building the prep driver")?;
    Ok(binary)
}

/// Builds the package at `package` in release with the driver as rustc's wrapper, links every function instance of its crate graph into one library, and leaves the library and its manifest in a new directory beside `out`, which it returns; the build and the driver's units stay beside `out`, so the next run rebuilds only what changed.
pub(super) fn run(toolchain: &Toolchain, store: &Path, package: &Path, out: &Path) -> Result<PathBuf> {
    let driver = binary(toolchain, store)?;
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |since| since.as_nanos());
    let fresh = out.with_extension(format!("{}.{stamp}", std::process::id()));
    let (units, target) = (out.with_extension(UNITS), out.with_extension(TARGET));
    std::fs::create_dir_all(&units)?;
    std::fs::create_dir_all(&fresh)?;
    let mut build = cargo::release(toolchain, &package.join(CARGO_MANIFEST), &target);
    cargo::settled(&mut build);
    build.env(WRAPPER, &driver).env(PREP_DIR, &units).current_dir(package);
    cargo::output(&mut build).with_context(|| format!("building {} with the prep driver", package.display()))?;
    cargo::output(Command::new(&driver).arg(LINK).arg(&units).arg(fresh.join(LIBRARY))).context("linking the prep library")?;
    Ok(fresh)
}

/// Points the link `out` at directory `fresh` in one rename, so a reader that resolves `out` sees one whole library and manifest, and removes every directory beside it but `fresh` and the one `out` named before; a process that mapped a removed library keeps it.
pub(super) fn publish(fresh: &Path, out: &Path) -> Result<()> {
    let before = std::fs::read_link(out).ok();
    if out.is_dir() && before.is_none() {
        std::fs::remove_dir_all(out)?;
    }
    let link = out.with_extension(format!("{LINK}{}", std::process::id()));
    if link.symlink_metadata().is_ok() {
        std::fs::remove_file(&link)?;
    }
    std::os::unix::fs::symlink(fresh.file_name().context("a prep directory with no name")?, &link)?;
    std::fs::rename(&link, out)?;
    let (Some(parent), Some(stem)) = (out.parent(), out.file_name()) else { return Ok(()) };
    let kept = [Some(fresh.to_path_buf()), before.map(|before| parent.join(before))];
    for entry in std::fs::read_dir(parent)? {
        let path = entry?.path();
        let sibling = path.file_name().is_some_and(|name| name.to_string_lossy().starts_with(&format!("{}.", stem.to_string_lossy())));
        let shared = [UNITS, TARGET].iter().any(|kept| path == out.with_extension(kept));
        if sibling && path.is_dir() && !shared && !kept.iter().flatten().any(|kept| *kept == path) {
            std::fs::remove_dir_all(&path)?;
        }
    }
    Ok(())
}

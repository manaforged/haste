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
const SOURCES: [(&str, &str); 14] = sources!(
    "Cargo.toml",
    "Cargo.lock",
    "build.rs",
    "src/main.rs",
    "src/abi.rs",
    "src/args.rs",
    "src/elf.rs",
    "src/error.rs",
    "src/export.rs",
    "src/layout.rs",
    "src/link.rs",
    "src/record.rs",
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

/// The prep driver binary for `toolchain`, built from the source Haste carries into the store on first use.
fn binary(toolchain: &Toolchain, store: &Path) -> Result<PathBuf> {
    let key = FxBuildHasher.hash_one((&SOURCES, &toolchain.version));
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

/// Builds the package at `package` in release with the driver as rustc's wrapper, links every function instance of its crate graph into one library, and leaves the library and its manifest in `out`, replacing what `out` held.
pub(super) fn run(toolchain: &Toolchain, store: &Path, package: &Path, out: &Path) -> Result<()> {
    let driver = binary(toolchain, store)?;
    let fresh = out.with_extension(format!("{}", std::process::id()));
    if fresh.exists() {
        std::fs::remove_dir_all(&fresh)?;
    }
    let units = fresh.join(UNITS);
    std::fs::create_dir_all(&units)?;
    let mut build = cargo::release(toolchain, &package.join(CARGO_MANIFEST), &fresh.join(TARGET));
    cargo::settled(&mut build);
    build.env(WRAPPER, &driver).env(PREP_DIR, &units).current_dir(package);
    cargo::output(&mut build).with_context(|| format!("building {} with the prep driver", package.display()))?;
    cargo::output(Command::new(&driver).arg(LINK).arg(&units).arg(fresh.join(LIBRARY))).context("linking the prep library")?;
    std::fs::remove_dir_all(&units)?;
    std::fs::remove_dir_all(fresh.join(TARGET))?;
    replace(&fresh, out)
}

/// Moves directory `fresh` to `out`, removing what `out` held; a process that mapped the old library keeps it.
fn replace(fresh: &Path, out: &Path) -> Result<()> {
    let old = out.with_extension(format!("old{}", std::process::id()));
    if out.exists() {
        std::fs::rename(out, &old)?;
    }
    std::fs::rename(fresh, out)?;
    if old.exists() {
        std::fs::remove_dir_all(&old)?;
    }
    Ok(())
}

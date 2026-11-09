use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

/// One `[[package]]` entry of `Cargo.lock`.
#[derive(Clone, Debug, Deserialize)]
pub struct Locked {
    pub name: String,
    pub version: semver::Version,
    pub source: Option<String>,
    #[serde(default)]
    pub dependencies: Vec<String>,
}

#[derive(Deserialize)]
struct LockFile {
    #[serde(default)]
    package: Vec<Locked>,
}

/// A dependency line of a locked package: `name`, `name version` or `name version (source)`.
pub struct LockedDep<'a> {
    pub name: &'a str,
    pub version: Option<semver::Version>,
}

impl Locked {
    /// Whether the package comes from crates.io.
    pub fn is_registry(&self) -> bool {
        self.source.as_deref().is_some_and(|source| source.starts_with("registry+") || source.starts_with("sparse+"))
    }

    /// The repository URL and commit of a package from git.
    pub fn git(&self) -> Option<(&str, &str)> {
        let rest = self.source.as_deref()?.strip_prefix("git+")?;
        let (url, commit) = rest.split_once('#')?;
        Some((url.split('?').next().unwrap_or(url), commit))
    }

    /// The parsed dependency lines.
    pub fn deps(&self) -> impl Iterator<Item = LockedDep<'_>> {
        self.dependencies.iter().map(|line| {
            let mut words = line.split_whitespace();
            let name = words.next().unwrap_or_default();
            let version = words.next().and_then(|text| semver::Version::parse(text).ok());
            LockedDep { name, version }
        })
    }
}

/// Reads `Cargo.lock` at `path`.
pub fn read(path: &Path) -> Result<Vec<Locked>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let lock: LockFile = toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
    Ok(lock.package)
}

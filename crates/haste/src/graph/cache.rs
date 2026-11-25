use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use super::lock::Locked;
use super::{MANIFEST_FILE, manifest, manifest_name};

pub(super) fn cargo_home() -> Result<PathBuf> {
    let home = std::env::var_os("CARGO_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cargo")));
    home.context("no CARGO_HOME or HOME")
}

fn read_subdirs(entries: std::fs::ReadDir) -> Vec<PathBuf> {
    entries
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_dir())
        .collect()
}

pub(super) fn subdirs(dir: &Path) -> Result<Vec<PathBuf>> {
    Ok(read_subdirs(
        std::fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))?,
    ))
}

fn registry_dirs() -> Result<Vec<PathBuf>> {
    let dir = cargo_home()?.join("registry/src");
    match std::fs::read_dir(&dir) {
        Ok(entries) => Ok(read_subdirs(entries)),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(err) => Err(err).with_context(|| format!("reading {}", dir.display())),
    }
}

fn cargo_configs(dir: &Path) -> Vec<(PathBuf, PathBuf)> {
    let find_cfg = |base: &Path, leaves: [&str; 2]| {
        leaves
            .into_iter()
            .map(|leaf| base.join(leaf))
            .find(|p| p.is_file())
    };
    let mut configs = Vec::new();
    for at in dir.ancestors() {
        if let Some(config) = find_cfg(at, [".cargo/config.toml", ".cargo/config"]) {
            configs.push((config, at.to_owned()));
        }
    }
    if let Ok(home) = cargo_home()
        && let Some(config) = find_cfg(&home, ["config.toml", "config"])
    {
        configs.push((config, home));
    }
    configs
}

fn source_directories_in(file: &Path, base: &Path) -> Result<Vec<PathBuf>> {
    let table: toml::Table = toml::from_str(&std::fs::read_to_string(file)?)
        .with_context(|| format!("parsing {}", file.display()))?;
    let sources = table
        .get("source")
        .and_then(toml::Value::as_table)
        .into_iter()
        .flat_map(|source| source.values())
        .filter_map(toml::Value::as_table);
    Ok(sources
        .filter_map(|entry| entry.get("directory")?.as_str())
        .map(|path| manifest::normalize(&base.join(path)))
        .collect())
}

pub(super) fn registries_with_vendor(top: &Path) -> Result<Vec<PathBuf>> {
    let mut dirs = Vec::new();
    for (config, base) in cargo_configs(top) {
        dirs.extend(source_directories_in(&config, &base)?);
    }
    let default_vendor = top.join("vendor");
    if !dirs.contains(&default_vendor) {
        dirs.push(default_vendor);
    }
    dirs.extend(registry_dirs()?);
    Ok(dirs)
}

fn manifest_version(dir: &Path) -> Option<semver::Version> {
    let table: toml::Table =
        toml::from_str(&std::fs::read_to_string(dir.join(MANIFEST_FILE)).ok()?).ok()?;
    let raw = table.get("package")?.get("version")?.as_str()?;
    semver::Version::parse(raw).ok()
}

fn find_registry_or_vendor(locked: &Locked, registries: &[PathBuf]) -> Option<PathBuf> {
    let versioned = format!("{}-{}", locked.name, locked.version);
    for dir in registries {
        let candidate = dir.join(&versioned);
        if candidate.is_dir() {
            return Some(candidate);
        }
        let plain = dir.join(&locked.name);
        if plain.is_dir() && manifest_version(&plain).as_ref() == Some(&locked.version) {
            return Some(plain);
        }
    }
    None
}

pub(super) fn source_dir(
    locked: &Locked,
    registries: &[PathBuf],
    paths: &BTreeMap<String, PathBuf>,
) -> Result<PathBuf> {
    if let Some((url, commit)) = locked.git() {
        return git_dir(&locked.name, url, commit)
            .or_else(|err| find_registry_or_vendor(locked, registries).ok_or(err));
    }
    if !locked.is_registry() {
        return paths
            .get(&locked.name)
            .cloned()
            .with_context(|| format!("no path for local package {}", locked.name));
    }
    let leaf = format!("{}-{}", locked.name, locked.version);
    find_registry_or_vendor(locked, registries)
        .with_context(|| format!("{leaf} is not in the registry source cache"))
}

fn git_dir(name: &str, url: &str, commit: &str) -> Result<PathBuf> {
    let repo = url
        .trim_end_matches('/')
        .rsplit('/')
        .next()
        .unwrap_or_default()
        .trim_end_matches(".git");
    let short = commit.get(..7).unwrap_or(commit);
    let checkouts = subdirs(&cargo_home()?.join("git/checkouts"))?;
    let mine = checkouts.into_iter().filter(|dir| {
        dir.file_name()
            .is_some_and(|leaf| leaf.to_string_lossy().starts_with(&format!("{repo}-")))
    });
    mine.map(|dir| dir.join(short))
        .filter(|dir| dir.is_dir())
        .find_map(|root| find_package(&root, name))
        .with_context(|| format!("{name} at {short} is not in the git checkouts of {url}"))
}

fn find_package(root: &Path, name: &str) -> Option<PathBuf> {
    let mut stack = vec![root.to_owned()];
    while let Some(dir) = stack.pop() {
        if manifest_name(&dir).as_deref() == Some(name) {
            return Some(dir);
        }
        let skip = |path: &PathBuf| {
            path.file_name()
                .is_some_and(|leaf| leaf == "target" || leaf.to_string_lossy().starts_with('.'))
        };
        stack.extend(
            subdirs(&dir)
                .unwrap_or_default()
                .into_iter()
                .filter(|path| !skip(path)),
        );
    }
    None
}

pub(super) fn patch_paths(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = patches_in(&dir.join(MANIFEST_FILE), dir)?;
    for (config, base) in cargo_configs(dir) {
        paths.extend(patches_in(&config, &base)?);
    }
    Ok(paths)
}

fn patches_in(file: &Path, base: &Path) -> Result<Vec<PathBuf>> {
    let table: toml::Table = toml::from_str(&std::fs::read_to_string(file)?)
        .with_context(|| format!("parsing {}", file.display()))?;
    let patches = table
        .get("patch")
        .and_then(toml::Value::as_table)
        .into_iter()
        .flat_map(|patch| patch.values());
    let entries = patches
        .filter_map(toml::Value::as_table)
        .flat_map(|source| source.values());
    Ok(entries
        .filter_map(|entry| entry.get("path")?.as_str())
        .map(|path| manifest::normalize(&base.join(path)))
        .collect())
}

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

pub mod cfg;
mod features;
pub mod lock;
pub mod manifest;

use lock::Locked;
use manifest::{Kind, Manifest};

/// Which machine a unit is built for: build scripts, their dependencies and proc macros run on the host.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Side {
    Target,
    Host,
}

/// The cfgs of the target and of the host.
pub struct Targets {
    pub target: cfg::Target,
    pub host: cfg::Target,
}

impl Targets {
    /// The cfgs that decide platform-specific dependencies on `side`.
    pub fn of(&self, side: Side) -> &cfg::Target {
        match side {
            Side::Target => &self.target,
            Side::Host => &self.host,
        }
    }
}

/// One locked package with its manifest and the locked package each dependency key names.
pub struct Package {
    pub locked: Locked,
    pub manifest: Manifest,
    pub resolved: BTreeMap<String, usize>,
    pub sysroot: bool,
}

/// A package built for one side with its resolved features and edges.
pub struct Unit {
    pub package: usize,
    pub side: Side,
    pub features: BTreeSet<String>,
    pub deps: Vec<(String, usize, Kind)>,
}

/// The resolved build graph of a workspace.
pub struct Graph {
    pub units: Vec<Unit>,
    pub members: Vec<usize>,
}

/// The packages of a workspace and its resolved graph.
pub struct Workspace {
    pub root: PathBuf,
    pub packages: Vec<Package>,
    pub graph: Graph,
}

fn registry_dirs() -> Result<Vec<PathBuf>> {
    let home = std::env::var_os("CARGO_HOME").map(PathBuf::from).or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cargo")));
    let src = home.context("no CARGO_HOME or HOME")?.join("registry/src");
    Ok(std::fs::read_dir(&src).with_context(|| format!("reading {}", src.display()))?.filter_map(|entry| entry.ok().map(|entry| entry.path())).collect())
}

fn source_dir(locked: &Locked, registries: &[PathBuf], paths: &BTreeMap<String, PathBuf>) -> Result<PathBuf> {
    if !locked.is_registry() {
        return paths.get(&locked.name).cloned().with_context(|| format!("no path for local package {}", locked.name));
    }
    let leaf = format!("{}-{}", locked.name, locked.version);
    registries.iter().map(|dir| dir.join(&leaf)).find(|dir| dir.is_dir()).with_context(|| format!("{leaf} is not in the registry source cache"))
}

fn patch_paths(dir: &Path) -> Result<Vec<PathBuf>> {
    let text = std::fs::read_to_string(dir.join("Cargo.toml"))?;
    let table: toml::Table = toml::from_str(&text)?;
    let patches = table.get("patch").and_then(toml::Value::as_table).into_iter().flat_map(|patch| patch.values());
    let entries = patches.filter_map(toml::Value::as_table).flat_map(|source| source.values());
    Ok(entries.filter_map(|entry| entry.get("path")?.as_str()).map(|path| dir.join(path)).collect())
}

fn workspace_root(root: &Path) -> Result<PathBuf> {
    let found = root.ancestors().find(|dir| dir.join("Cargo.lock").is_file());
    found.map(Path::to_owned).with_context(|| format!("no Cargo.lock at or above {}", root.display()))
}

fn local_paths(root: &Path, top: &Path, workspace: Option<&toml::Table>) -> Result<BTreeMap<String, PathBuf>> {
    let mut paths = BTreeMap::new();
    let mut stack = vec![root.to_owned()];
    stack.extend(patch_paths(top)?);
    let members = workspace.and_then(|ws| ws.get("members")).and_then(toml::Value::as_array).into_iter().flatten();
    stack.extend(members.filter_map(toml::Value::as_str).filter(|member| !member.contains('*')).map(|member| top.join(member)));
    while let Some(dir) = stack.pop() {
        let manifest = manifest::read(&dir, workspace)?;
        if paths.insert(manifest.name.clone(), dir.clone()).is_some() {
            continue;
        }
        stack.extend(manifest.deps.iter().filter_map(|dep| dep.path.clone()));
    }
    Ok(paths)
}

fn resolve_deps(packages: &mut [Package]) {
    let index: BTreeMap<(String, semver::Version), usize> =
        packages.iter().enumerate().map(|(at, package)| ((package.locked.name.clone(), package.locked.version.clone()), at)).collect();
    for at in 0..packages.len() {
        let locked: Vec<(String, Option<semver::Version>)> = packages[at].locked.deps().map(|dep| (dep.name.to_owned(), dep.version)).collect();
        let candidates = |name: &str| -> Vec<usize> {
            let named = index.iter().filter(|((locked_name, _), _)| locked_name == name);
            named.filter(|((_, version), _)| locked.iter().any(|(dep, want)| dep == name && want.as_ref().is_none_or(|want| want == version))).map(|(_, at)| *at).collect()
        };
        let mut resolved = BTreeMap::new();
        for dep in &packages[at].manifest.deps {
            let found = candidates(&dep.package);
            let pick = found.iter().copied().find(|at| dep.req.as_ref().is_none_or(|req| req.matches(&packages[*at].locked.version)));
            if let Some(pick) = pick.or_else(|| found.first().copied()) {
                resolved.insert(dep.key.clone(), pick);
            }
        }
        packages[at].resolved = resolved;
    }
}

/// Reads the package at `root` from its workspace's `Cargo.lock` and manifests and resolves its graph for `targets`.
pub fn load(root: &Path, targets: &Targets) -> Result<Workspace> {
    load_with(root, targets, &[])
}

/// Like [`load`], with `features` enabled on the root package besides its defaults.
pub fn load_with(root: &Path, targets: &Targets, features: &[&str]) -> Result<Workspace> {
    let top = workspace_root(root)?;
    let workspace = manifest::workspace_of(&top)?;
    let locked = lock::read(&top.join("Cargo.lock"))?;
    let mut registries = registry_dirs()?;
    registries.insert(0, top.join("vendor"));
    let paths = local_paths(root, &top, workspace.as_ref())?;
    let mut packages = Vec::with_capacity(locked.len());
    for entry in locked {
        let dir = source_dir(&entry, &registries, &paths)?;
        let manifest = manifest::read(&dir, if entry.is_registry() { None } else { workspace.as_ref() })?;
        packages.push(Package { locked: entry, manifest, resolved: BTreeMap::new(), sysroot: false });
    }
    resolve_deps(&mut packages);
    let root_name = manifest::read(root, workspace.as_ref())?.name;
    let members: Vec<usize> = packages.iter().enumerate().filter(|(_, package)| !package.locked.is_registry() && package.locked.name == root_name).map(|(at, _)| at).collect();
    if members.is_empty() {
        bail!("{root_name} is not in Cargo.lock");
    }
    let graph = features::resolve(&packages, targets, &members, features);
    Ok(Workspace { root: root.to_owned(), packages, graph })
}

/// The standard library sources of the toolchain: `HASTE_LIBRARY`, or rust-src of the first stable toolchain.
pub fn library() -> Result<PathBuf> {
    if let Some(dir) = std::env::var_os("HASTE_LIBRARY") {
        return Ok(PathBuf::from(dir));
    }
    let home = std::env::var_os("RUSTUP_HOME").map(PathBuf::from).or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".rustup")));
    let toolchains = home.context("no RUSTUP_HOME or HOME")?.join("toolchains");
    let mut found: Vec<PathBuf> = std::fs::read_dir(&toolchains)?.filter_map(|entry| entry.ok().map(|entry| entry.path())).collect();
    found.sort();
    let library = found.iter().filter(|dir| dir.file_name().is_some_and(|name| name.to_string_lossy().starts_with("stable-"))).map(|dir| dir.join("lib/rustlib/src/rust/library"));
    library.into_iter().find(|dir| dir.join("sysroot").is_dir()).context("no rust-src in a stable toolchain; set HASTE_LIBRARY")
}

/// Loads the standard library graph rooted at the `sysroot` package of `library`.
pub fn load_sysroot(library: &Path, targets: &Targets) -> Result<Workspace> {
    let mut sysroot = load_with(&library.join("sysroot"), targets, &["backtrace"])?;
    sysroot.packages.iter_mut().for_each(|package| package.sysroot = true);
    Ok(sysroot)
}

impl Workspace {
    /// Appends the packages and units of `other`, renumbered after this workspace's.
    pub fn merge(&mut self, other: Workspace) {
        let (packages, units) = (self.packages.len(), self.graph.units.len());
        for mut package in other.packages {
            package.resolved.values_mut().for_each(|at| *at += packages);
            self.packages.push(package);
        }
        for mut unit in other.graph.units {
            unit.package += packages;
            unit.deps.iter_mut().for_each(|dep| dep.1 += units);
            self.graph.units.push(unit);
        }
    }

    /// One line per unit, `side name version|features`, sorted, the shape that `cargo tree --format '{p}|{f}'` reports.
    pub fn unit_lines(&self) -> Vec<String> {
        let mut lines: Vec<String> = self
            .graph
            .units
            .iter()
            .map(|unit| {
                let package = &self.packages[unit.package].locked;
                let side = if unit.side == Side::Host { "host" } else { "target" };
                let features: Vec<&str> = unit.features.iter().map(String::as_str).collect();
                format!("{side} {} {}|{}", package.name, package.version, features.join(","))
            })
            .collect();
        lines.sort();
        lines
    }
}

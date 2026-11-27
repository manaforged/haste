use anyhow::{Context, Result, bail};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

mod cache;
pub mod cfg;
mod features;
pub mod lock;
pub mod manifest;
pub mod script;
pub(crate) mod snapshot;
pub mod store;
mod sync;
#[cfg(test)]
mod tests;

use cache::{SourceCatalog, patch_paths, source_dir};
use lock::Locked;
use manifest::{Kind, Manifest};
use sync::{lockfile_dirty, missing_root_in_lock, sync_lockfile};

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

impl Package {
    /// Whether its sources can change while haste runs: a path package outside the standard library.
    pub fn local(&self) -> bool {
        self.locked.source.is_none() && !self.sysroot
    }
}

/// A package built for one side with its resolved features and edges.
pub struct Unit {
    pub package: usize,
    pub side: Side,
    pub features: BTreeSet<String>,
    pub deps: Vec<(String, usize, Kind)>,
    pub script: Option<std::sync::Arc<script::Output>>,
}

/// The resolved build graph of a workspace.
pub struct Graph {
    pub units: Vec<Unit>,
    pub members: Vec<usize>,
}

/// The packages of a workspace and its resolved graph.
pub struct Workspace {
    pub packages: Vec<Package>,
    pub graph: Graph,
    pub(crate) rustflags: Vec<String>,
    pub(crate) config_env: Vec<(String, String, bool)>,
    pub(crate) profile_dev: manifest::ProfileOverride,
    pub(crate) profile_release: manifest::ProfileOverride,
}

pub(crate) const MANIFEST_FILE: &str = "Cargo.toml";
pub(crate) const LOCK_FILE: &str = "Cargo.lock";

pub(super) fn manifest_name(dir: &Path) -> Option<String> {
    let table: toml::Table =
        toml::from_str(&std::fs::read_to_string(dir.join(MANIFEST_FILE)).ok()?).ok()?;
    Some(table.get("package")?.get("name")?.as_str()?.to_owned())
}

/// The root directory and `[workspace]` table of the nearest manifest at or above `dir` that has one, which a local or git package inherits from.
fn owning_workspace(dir: &Path) -> Result<Option<(PathBuf, toml::Table)>> {
    for at in dir
        .ancestors()
        .filter(|at| at.join(MANIFEST_FILE).is_file())
    {
        if let Some(workspace) = manifest::workspace_of(at)? {
            return Ok(Some((at.to_owned(), workspace)));
        }
    }
    Ok(None)
}

fn member_dirs(top: &Path, workspace: Option<&toml::Table>, key: &str) -> Vec<PathBuf> {
    manifest::workspace_member_dirs(top, workspace, key)
}

fn run_cargo(manifest_dir: &Path, args: &[&str]) -> Result<()> {
    let cargo = crate::process_env_var("CARGO").map_or_else(
        || std::ffi::OsString::from("cargo"),
        std::ffi::OsString::from,
    );
    let mut cmd = std::process::Command::new(cargo);
    if let Some(cenv) = crate::client_env() {
        cmd.env_clear().envs(cenv.iter());
    }
    let out = cmd
        .current_dir(manifest_dir)
        .args(args)
        .args(["--quiet", "--manifest-path"])
        .arg(manifest_dir.join(MANIFEST_FILE))
        .output()
        .with_context(|| {
            format!(
                "failed to run `cargo {}` for {}",
                args.join(" "),
                manifest_dir.display()
            )
        })?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        bail!("`cargo {}` failed: {}", args.join(" "), stderr.trim());
    }
    Ok(())
}

pub(crate) fn workspace_root(root: &Path) -> Result<PathBuf> {
    if let Some(found) = root.ancestors().find(|dir| dir.join(LOCK_FILE).is_file()) {
        return Ok(found.to_owned());
    }
    let manifest_dir = root
        .ancestors()
        .find(|dir| dir.join(MANIFEST_FILE).is_file())
        .with_context(|| {
            format!(
                "no {LOCK_FILE} or {MANIFEST_FILE} at or above {}",
                root.display()
            )
        })?;
    let sel = crate::selection();
    if sel.locked {
        bail!("{LOCK_FILE} is missing and --locked was passed");
    }
    let args: &[&str] = if sel.offline {
        &["generate-lockfile", "--offline"]
    } else {
        &["generate-lockfile"]
    };
    run_cargo(manifest_dir, args)?;
    root.ancestors()
        .find(|dir| dir.join(LOCK_FILE).is_file())
        .map(Path::to_owned)
        .with_context(|| format!("no {LOCK_FILE} at or above {}", root.display()))
}

/// The lock file, workspace manifest, root manifest, and every local package's manifest and build script.
pub(crate) fn manifest_paths(workspace: &Workspace, root: &Path) -> Vec<PathBuf> {
    let top = workspace_root(root).unwrap_or_else(|_| root.to_owned());
    let head = [
        top.join(LOCK_FILE),
        top.join(MANIFEST_FILE),
        root.join(MANIFEST_FILE),
    ];
    let configs = cache::config_paths(&top)
        .into_iter()
        .chain(cache::config_paths(root));
    let local = workspace
        .packages
        .iter()
        .filter(|package| package.local())
        .flat_map(|package| {
            std::iter::once(package.manifest.dir.join(MANIFEST_FILE))
                .chain(package.manifest.build.clone())
        });
    let mut paths: Vec<PathBuf> = head.into_iter().chain(configs).chain(local).collect();
    paths.sort();
    paths.dedup();
    paths
}

fn local_paths(
    root: &Path,
    top: &Path,
    workspace: Option<&toml::Table>,
    wanted: &BTreeSet<String>,
) -> Result<BTreeMap<String, PathBuf>> {
    let mut paths = BTreeMap::new();
    let mut seen_ws = BTreeSet::from([top.to_owned()]);
    let mut stack = vec![root.to_owned(), top.to_owned()];
    stack.extend(patch_paths(top)?);
    stack.extend(member_dirs(top, workspace, "members"));
    while let Some(dir) = stack.pop() {
        let dir = manifest::normalize(&dir);
        let Some(name) = manifest_name(&dir) else {
            continue;
        };
        if !wanted.contains(&name) {
            continue;
        }
        let owner = owning_workspace(&dir)?;
        if let Some((ws_dir, ws_table)) = &owner
            && seen_ws.insert(ws_dir.clone())
        {
            stack.extend(patch_paths(ws_dir)?);
            stack.extend(member_dirs(ws_dir, Some(ws_table), "members"));
        }
        let manifest = manifest::read(&dir, owner.as_ref().map(|(_, ws)| ws))?;
        if paths.insert(manifest.name.clone(), dir).is_some() {
            continue;
        }
        stack.extend(manifest.deps.iter().filter_map(|dep| dep.path.clone()));
    }
    Ok(paths)
}

fn resolve_deps(packages: &mut [Package]) {
    let index: BTreeMap<(String, semver::Version), usize> = packages
        .iter()
        .enumerate()
        .map(|(at, package)| {
            (
                (package.locked.name.clone(), package.locked.version.clone()),
                at,
            )
        })
        .collect();
    for at in 0..packages.len() {
        let locked: Vec<(String, Option<semver::Version>)> = packages[at]
            .locked
            .deps()
            .map(|dep| (dep.name.to_owned(), dep.version))
            .collect();
        let candidates = |name: &str| -> Vec<usize> {
            let named = index
                .iter()
                .filter(|((locked_name, _), _)| locked_name == name);
            named
                .filter(|((_, version), _)| {
                    locked.iter().any(|(dep, want)| {
                        dep == name && want.as_ref().is_none_or(|want| want == version)
                    })
                })
                .map(|(_, at)| *at)
                .collect()
        };
        let mut resolved = BTreeMap::new();
        for dep in &packages[at].manifest.deps {
            let found = candidates(&dep.package);
            let pick = found.iter().copied().find(|at| {
                dep.req
                    .as_ref()
                    .is_none_or(|req| req.matches(&packages[*at].locked.version))
            });
            if let Some(pick) = pick.or_else(|| found.first().copied()) {
                resolved.insert(dep.key.clone(), pick);
            }
        }
        packages[at].resolved = resolved;
    }
}

/// Reads the package at `root` from its workspace's `Cargo.lock` and manifests and resolves its graph for `targets` using `selection`.
pub(crate) fn load(
    root: &Path,
    targets: &Targets,
    selection: &crate::Selection,
) -> Result<Workspace> {
    let features: Vec<&str> = selection.features.iter().map(String::as_str).collect();
    load_filtered(
        root,
        targets,
        &selection.packages,
        &features,
        (!selection.no_default_features, selection.all_features),
        (&[], true),
    )
}

fn load_package(
    entry: Locked,
    catalog: &SourceCatalog,
    paths: &BTreeMap<String, PathBuf>,
) -> Result<(Package, Option<anyhow::Error>)> {
    match source_dir(&entry, catalog, paths) {
        Ok(dir) => {
            let owner = if entry.is_registry() {
                None
            } else {
                owning_workspace(&dir)?
            };
            let manifest = manifest::read(&dir, owner.as_ref().map(|(_, ws)| ws))?;
            Ok((
                Package {
                    locked: entry,
                    manifest,
                    resolved: BTreeMap::new(),
                    sysroot: false,
                },
                None,
            ))
        }
        Err(err) => {
            let manifest = manifest::Manifest {
                name: entry.name.clone(),
                ..manifest::Manifest::default()
            };
            Ok((
                Package {
                    locked: entry,
                    manifest,
                    resolved: BTreeMap::new(),
                    sysroot: false,
                },
                Some(err),
            ))
        }
    }
}

fn load_packages(
    locked: Vec<Locked>,
    catalog: &SourceCatalog,
    paths: &BTreeMap<String, PathBuf>,
) -> Result<(Vec<Package>, Vec<Option<anyhow::Error>>)> {
    let results = crate::sync::parallel_map(locked, |entry| load_package(entry, catalog, paths));
    let mut packages = Vec::with_capacity(results.len());
    let mut missing = Vec::with_capacity(results.len());
    for res in results {
        let (pkg, err) = res?;
        packages.push(pkg);
        missing.push(err);
    }
    Ok((packages, missing))
}

fn root_package_names(
    root: &Path,
    top: &Path,
    workspace: Option<&toml::Table>,
) -> Result<Vec<String>> {
    if workspace.is_some()
        && (crate::selection().workspace || (manifest_name(root).is_none() && root == top))
    {
        let use_all = crate::selection().workspace;
        let mut dirs = if use_all {
            member_dirs(top, workspace, "members")
        } else {
            member_dirs(top, workspace, "default-members")
        };
        let include_root = use_all || dirs.is_empty();
        if dirs.is_empty() {
            dirs = member_dirs(top, workspace, "members");
        }
        let mut names = Vec::new();
        if include_root && let Some(root_name) = manifest_name(top) {
            names.push(root_name);
        }
        for dir in dirs {
            if let Some(name) = manifest_name(&dir)
                && !names.contains(&name)
            {
                names.push(name);
            }
        }
        return Ok(names);
    }
    let owner = owning_workspace(root)?;
    Ok(vec![
        manifest::read(root, owner.as_ref().map(|(_, ws)| ws))?.name,
    ])
}

fn mark_reachable_deps(
    locked: &[Locked],
    by_name: &BTreeMap<&str, Vec<usize>>,
    (idx, omit): (usize, &[&str]),
    keep: &mut [bool],
    stack: &mut Vec<usize>,
) {
    for dep in locked[idx].deps() {
        if omit.contains(&dep.name) {
            continue;
        }
        let Some(cands) = by_name.get(dep.name) else {
            continue;
        };
        for &cand in cands {
            if !keep[cand]
                && dep
                    .version
                    .as_ref()
                    .is_none_or(|v| *v == locked[cand].version)
            {
                keep[cand] = true;
                stack.push(cand);
            }
        }
    }
}

fn prune_unreachable_locked(
    locked: Vec<Locked>,
    root_names: &[String],
    omit_root_deps: &[&str],
) -> Vec<Locked> {
    let mut by_name: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (idx, entry) in locked.iter().enumerate() {
        by_name.entry(entry.name.as_str()).or_default().push(idx);
    }
    let mut keep = vec![false; locked.len()];
    let mut stack = Vec::new();
    for (idx, entry) in locked.iter().enumerate() {
        if !entry.is_registry() && root_names.contains(&entry.name) {
            keep[idx] = true;
            stack.push(idx);
        }
    }
    while let Some(idx) = stack.pop() {
        let omit = if !locked[idx].is_registry() && root_names.contains(&locked[idx].name) {
            omit_root_deps
        } else {
            &[]
        };
        mark_reachable_deps(&locked, &by_name, (idx, omit), &mut keep, &mut stack);
    }
    locked
        .into_iter()
        .enumerate()
        .filter_map(|(idx, entry)| keep[idx].then_some(entry))
        .collect()
}

fn load_graph_packages(
    (root, top, workspace): (&Path, &Path, Option<&toml::Table>),
    (raw_locked, root_names, omit_root_deps): (Vec<Locked>, &[String], &[&str]),
    catalog: &SourceCatalog,
) -> Result<(
    BTreeMap<String, PathBuf>,
    Vec<Package>,
    Vec<Option<anyhow::Error>>,
)> {
    let locked = prune_unreachable_locked(raw_locked, root_names, omit_root_deps);
    let wanted_local = locked
        .iter()
        .filter(|e| !e.is_registry())
        .map(|e| e.name.clone())
        .collect();
    let paths = local_paths(root, top, workspace, &wanted_local)?;
    let (packages, missing) = load_packages(locked, catalog, &paths)?;
    Ok((paths, packages, missing))
}

fn resync_lockfile(top: &Path, offline: bool) -> Result<Vec<Locked>> {
    if crate::selection().locked {
        bail!("{LOCK_FILE} needs to be updated but --locked was passed");
    }
    sync_lockfile(top, offline)?;
    lock::read(&top.join(LOCK_FILE))
}

fn load_synced_packages(
    (root, top, workspace): (&Path, &Path, Option<&toml::Table>),
    (root_names, omit_root_deps, allow_fetch): (&[String], &[&str], bool),
    catalog: &SourceCatalog,
) -> Result<(
    BTreeMap<String, PathBuf>,
    Vec<Package>,
    Vec<Option<anyhow::Error>>,
)> {
    let mut raw_locked = lock::read(&top.join(LOCK_FILE))?;
    if !allow_fetch {
        return load_graph_packages(
            (root, top, workspace),
            (raw_locked, root_names, omit_root_deps),
            catalog,
        );
    }
    if missing_root_in_lock(root_names, &raw_locked) {
        raw_locked = resync_lockfile(top, catalog.offline)?;
    }
    let (paths, packages, missing) = load_graph_packages(
        (root, top, workspace),
        (raw_locked.clone(), root_names, omit_root_deps),
        catalog,
    )?;
    let mut member_set: BTreeSet<PathBuf> = member_dirs(top, workspace, "members")
        .into_iter()
        .map(|d| manifest::normalize(&d))
        .collect();
    member_set.insert(manifest::normalize(root));
    if !lockfile_dirty(top, &raw_locked, &packages, &missing, &member_set)? {
        return Ok((paths, packages, missing));
    }
    raw_locked = resync_lockfile(top, catalog.offline)?;
    load_graph_packages(
        (root, top, workspace),
        (raw_locked, root_names, omit_root_deps),
        catalog,
    )
}

fn load_filtered(
    root: &Path,
    targets: &Targets,
    selected: &[String],
    features: &[&str],
    (default_features, all_features): (bool, bool),
    (omit_root_deps, allow_fetch): (&[&str], bool),
) -> Result<Workspace> {
    if !root.exists() {
        bail!("path does not exist: {}", root.display());
    }
    let canonical = std::fs::canonicalize(root).unwrap_or_else(|_| root.to_owned());
    let root = canonical
        .ancestors()
        .find(|dir| dir.join(MANIFEST_FILE).is_file())
        .unwrap_or(canonical.as_path());
    let top = workspace_root(root)?;
    let workspace = manifest::workspace_of(&top)?;
    let root_names = if selected.is_empty() {
        root_package_names(root, &top, workspace.as_ref())?
    } else {
        selected.to_vec()
    };
    let catalog = SourceCatalog::load(&top)?;
    let (paths, mut packages, mut missing) = load_synced_packages(
        (root, &top, workspace.as_ref()),
        (&root_names, omit_root_deps, allow_fetch),
        &catalog,
    )?;
    if !omit_root_deps.is_empty() {
        for pkg in packages
            .iter_mut()
            .filter(|p| !p.locked.is_registry() && root_names.contains(&p.locked.name))
        {
            pkg.manifest
                .deps
                .retain(|dep| !omit_root_deps.contains(&dep.package.as_str()));
        }
    }
    resolve_deps(&mut packages);
    let members: Vec<usize> = packages
        .iter()
        .enumerate()
        .filter(|(_, package)| {
            !package.locked.is_registry() && root_names.contains(&package.locked.name)
        })
        .map(|(at, _)| at)
        .collect();
    if !selected.is_empty() {
        for name in selected {
            if !members.iter().any(|&at| packages[at].locked.name == *name) {
                bail!("package `{name}` is not in {LOCK_FILE}");
            }
        }
    } else if members.is_empty() {
        bail!("{} is not in {LOCK_FILE}", root.display());
    }
    let resolve = |pkgs: &[Package]| {
        features::resolve(
            pkgs,
            targets,
            &members,
            features,
            (default_features, all_features, workspace.as_ref(), &top),
        )
    };
    let is_external = |pkg: &Package| pkg.locked.is_registry() || pkg.locked.git().is_some();
    let resolved = resolve(&packages);
    let needs_fetch = match &resolved {
        Ok(graph) => graph
            .units
            .iter()
            .any(|u| missing[u.package].is_some() && is_external(&packages[u.package])),
        Err(_) => missing
            .iter()
            .enumerate()
            .any(|(idx, err)| err.is_some() && is_external(&packages[idx])),
    };
    let graph = if allow_fetch && !catalog.offline && needs_fetch {
        eprintln!("haste: fetching missing dependencies with `cargo fetch`...");
        if let Err(fetch_err) = run_cargo(&top, &["fetch", "--locked"]) {
            let first = match &resolved {
                Ok(graph) => graph
                    .units
                    .iter()
                    .find_map(|u| missing[u.package].as_ref())
                    .map(|e| format!("{e:#}")),
                Err(err) => Some(format!("{err:#}")),
            }
            .unwrap_or_default();
            bail!("{first} (automatic `cargo fetch` failed: {fetch_err:#})");
        }
        let fresh_catalog = SourceCatalog::load(&top)?;
        let (retry_idxs, retry_entries): (Vec<usize>, Vec<Locked>) = missing
            .iter()
            .enumerate()
            .filter(|&(idx, err)| err.is_some() && is_external(&packages[idx]))
            .map(|(idx, _)| (idx, packages[idx].locked.clone()))
            .unzip();
        let (reloaded, retry_missing) = load_packages(retry_entries, &fresh_catalog, &paths)?;
        for ((idx, pkg), err) in retry_idxs.into_iter().zip(reloaded).zip(retry_missing) {
            (packages[idx], missing[idx]) = (pkg, err);
        }
        resolve_deps(&mut packages);
        resolve(&packages)?
    } else {
        resolved?
    };
    for unit in &graph.units {
        if let Some(err) = missing[unit.package].take() {
            return Err(err);
        }
    }
    let (rustflags, config_env) = if allow_fetch {
        (cache::resolved_rustflags(&catalog), catalog.env)
    } else {
        (Vec::new(), Vec::new())
    };
    let (profile_dev, profile_release) = manifest::profiles_of(&top);
    Ok(Workspace {
        packages,
        graph,
        rustflags,
        config_env,
        profile_dev,
        profile_release,
    })
}

/// The standard library sources of the toolchain: `HASTE_LIBRARY`, the active `rustc` sysroot's `rust-src`, or the newest rustup toolchain with `rust-src` (preferring `stable-*`).
pub fn library() -> Result<PathBuf> {
    if let Some(dir) = std::env::var_os(crate::LIBRARY_ENV) {
        let path = PathBuf::from(dir);
        if !path.join("sysroot").is_dir() {
            anyhow::bail!(
                "HASTE_LIBRARY `{}` does not contain a `sysroot` directory (expected rust-src library dir)",
                path.display()
            );
        }
        return Ok(path);
    }
    let sysroot_library = |root: &Path| {
        let dir = root.join("lib/rustlib/src/rust/library");
        dir.join("sysroot").is_dir().then_some(dir)
    };
    let home = std::env::var_os("RUSTUP_HOME")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .or_else(|| std::env::var_os("USERPROFILE"))
                .map(|home| PathBuf::from(home).join(".rustup"))
        });
    if std::env::var_os("RUSTC").is_none()
        && let Some(ref rustup_home) = home
    {
        let active = std::env::var("RUSTUP_TOOLCHAIN").ok().or_else(|| {
            let text = std::fs::read_to_string(rustup_home.join("settings.toml")).ok()?;
            text.lines().find_map(|line| {
                let rest = line
                    .trim()
                    .strip_prefix("default_toolchain")?
                    .trim()
                    .strip_prefix('=')?
                    .trim();
                Some(rest.trim_matches('"').to_owned())
            })
        });
        if let Some(tc) = active
            && let Some(dir) = sysroot_library(&rustup_home.join("toolchains").join(tc))
        {
            return Ok(dir);
        }
    }
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| std::ffi::OsString::from("rustc"));
    if let Ok(output) = std::process::Command::new(rustc)
        .args(["--print", "sysroot"])
        .output()
        && output.status.success()
        && let Ok(text) = std::str::from_utf8(&output.stdout)
        && let Some(dir) = sysroot_library(Path::new(text.trim()))
    {
        return Ok(dir);
    }
    if let Some(toolchains) = home.map(|home| home.join("toolchains"))
        && let Ok(entries) = std::fs::read_dir(&toolchains)
    {
        let mut found: Vec<PathBuf> = entries
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .collect();
        found.sort();
        found.reverse();
        let stable = found.iter().filter(|dir| {
            dir.file_name()
                .is_some_and(|name| name.to_string_lossy().starts_with("stable-"))
        });
        if let Some(dir) = stable
            .chain(found.iter())
            .find_map(|dir| sysroot_library(dir))
        {
            return Ok(dir);
        }
    }
    anyhow::bail!("no rust-src found; run `rustup component add rust-src` or set HASTE_LIBRARY")
}

/// Loads the standard library graph rooted at the `sysroot` package of `library`.
pub fn load_sysroot(library: &Path, targets: &Targets) -> Result<Workspace> {
    let sysroot_dir = library.join("sysroot");
    let raw_dylib = targets.target.has_pair("target_os", "windows")
        && std::fs::read_to_string(sysroot_dir.join("Cargo.toml"))
            .is_ok_and(|text| text.contains("windows_raw_dylib"));
    let features: &[&str] = if raw_dylib {
        &["windows_raw_dylib"]
    } else {
        &[]
    };
    let mut sysroot = load_filtered(
        &sysroot_dir,
        targets,
        &[],
        features,
        (true, false),
        (&["test"], false),
    )?;
    sysroot
        .packages
        .iter_mut()
        .for_each(|package| package.sysroot = true);
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
                let side = if unit.side == Side::Host {
                    "host"
                } else {
                    "target"
                };
                let features: Vec<&str> = unit.features.iter().map(String::as_str).collect();
                format!(
                    "{side} {} {}|{}",
                    package.name,
                    package.version,
                    features.join(",")
                )
            })
            .collect();
        lines.sort();
        lines
    }
}

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use toml::{Table, Value};

/// How a dependency is used.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    Normal,
    Build,
    Dev,
}

const KIND_TABLES: [(&str, Kind); 5] = [
    ("dependencies", Kind::Normal),
    ("build-dependencies", Kind::Build),
    ("build_dependencies", Kind::Build),
    ("dev-dependencies", Kind::Dev),
    ("dev_dependencies", Kind::Dev),
];

/// One dependency declaration.
#[derive(Clone, Debug)]
pub struct Dep {
    pub key: String,
    pub package: String,
    pub req: Option<semver::VersionReq>,
    pub path: Option<PathBuf>,
    pub optional: bool,
    pub default_features: bool,
    pub features: Vec<String>,
    pub kind: Kind,
    pub platform: Option<String>,
}

/// Package metadata strings exported as `CARGO_PKG_*` environment variables.
#[derive(Clone, Debug, Default)]
pub struct PackageMeta {
    pub authors: String,
    pub description: String,
    pub homepage: String,
    pub repository: String,
    pub license: String,
    pub license_file: String,
    pub readme: String,
    pub rust_version: String,
}

/// A binary, example, integration test, or benchmark target declared or auto-discovered in a manifest.
#[derive(Clone, Debug, Default)]
pub struct TargetSpec {
    pub name: String,
    pub path: PathBuf,
    pub required_features: Vec<String>,
}

impl TargetSpec {
    /// Whether every feature listed in `required-features` is enabled in `features`.
    pub fn enabled(&self, features: &std::collections::BTreeSet<String>) -> bool {
        self.required_features.iter().all(|f| features.contains(f))
    }
}

/// Profile overrides parsed from `[profile.dev]` or `[profile.release]`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ProfileOverride {
    pub debug_assertions: Option<bool>,
    pub overflow_checks: Option<bool>,
}

/// The parts of a `Cargo.toml` that the graph needs.
#[derive(Clone, Debug, Default)]
pub struct Manifest {
    pub dir: PathBuf,
    pub name: String,
    pub version: Option<semver::Version>,
    pub edition: String,
    pub resolver: Option<u8>,
    pub meta: PackageMeta,
    pub links: Option<String>,
    pub default_run: Option<String>,
    pub build: Option<PathBuf>,
    pub lib: Option<PathBuf>,
    pub lib_name: Option<String>,
    pub proc_macro: bool,
    pub bins: Vec<TargetSpec>,
    pub examples: Vec<TargetSpec>,
    pub tests: Vec<TargetSpec>,
    pub benches: Vec<TargetSpec>,
    pub features: BTreeMap<String, Vec<String>>,
    pub deps: Vec<Dep>,
}

/// `path` with `.` and `..` components removed lexically.
pub fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for part in path.components() {
        match part {
            std::path::Component::ParentDir => {
                out.pop();
            }
            std::path::Component::CurDir => {}
            other => out.push(other),
        }
    }
    out
}

fn text<'a>(table: &'a Table, key: &str) -> Option<&'a str> {
    table.get(key).and_then(Value::as_str)
}

fn flag(table: &Table, keys: &[&str]) -> Option<bool> {
    keys.iter()
        .find_map(|key| table.get(*key).and_then(Value::as_bool))
}

fn strings(value: Option<&Value>) -> Vec<String> {
    let items = value
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    items
        .iter()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect()
}

fn is_workspace_inherited(table: &Table) -> bool {
    table.get("workspace").and_then(Value::as_bool) == Some(true)
}

fn merged(spec: &Value, workspace: Option<&Table>, key: &str) -> Table {
    let mut table = match spec {
        Value::String(req) => {
            Table::from_iter([("version".to_owned(), Value::String(req.clone()))])
        }
        Value::Table(table) => table.clone(),
        _ => Table::new(),
    };
    if is_workspace_inherited(&table)
        && let Some(base) = workspace.and_then(|ws| ws.get(key))
    {
        let extra = strings(table.get("features"));
        let mut inherited = merged(base, None, key);
        let mut all = strings(inherited.get("features"));
        all.extend(extra);
        inherited.insert(
            "features".to_owned(),
            Value::Array(all.into_iter().map(Value::String).collect()),
        );
        if let Some(optional) = table.get("optional") {
            inherited.insert("optional".to_owned(), optional.clone());
        }
        if let Some(default_features) = flag(&table, &["default-features", "default_features"]) {
            inherited.remove("default-features");
            inherited.remove("default_features");
            inherited.insert(
                "default-features".to_owned(),
                Value::Boolean(default_features),
            );
        }
        table = inherited;
    }
    table
}

struct Scope<'a> {
    dir: &'a Path,
    workspace: Option<&'a Table>,
}

fn dep(
    scope: &Scope<'_>,
    key: &str,
    spec: &Value,
    kind: Kind,
    platform: Option<&str>,
) -> Result<Dep> {
    let table = merged(spec, scope.workspace, key);
    let req = text(&table, "version")
        .map(semver::VersionReq::parse)
        .transpose()
        .with_context(|| format!("version of {key}"))?;
    Ok(Dep {
        key: key.to_owned(),
        package: text(&table, "package").unwrap_or(key).to_owned(),
        req,
        path: text(&table, "path").map(|path| normalize(&scope.dir.join(path))),
        optional: flag(&table, &["optional"]).unwrap_or(false),
        default_features: flag(&table, &["default-features", "default_features"]).unwrap_or(true),
        features: strings(table.get("features")),
        kind,
        platform: platform.map(str::to_owned),
    })
}

fn deps_of(
    scope: &Scope<'_>,
    table: &Table,
    platform: Option<&str>,
    into: &mut Vec<Dep>,
) -> Result<()> {
    for (name, kind) in KIND_TABLES {
        let Some(entries) = table.get(name).and_then(Value::as_table) else {
            continue;
        };
        for (key, spec) in entries {
            into.push(dep(scope, key, spec, kind, platform)?);
        }
    }
    Ok(())
}

fn all_deps(scope: &Scope<'_>, root: &Table) -> Result<Vec<Dep>> {
    let mut deps = Vec::new();
    deps_of(scope, root, None, &mut deps)?;
    let targets = root
        .get("target")
        .and_then(Value::as_table)
        .into_iter()
        .flatten();
    for (platform, table) in targets {
        if let Some(table) = table.as_table() {
            deps_of(scope, table, Some(platform), &mut deps)?;
        }
    }
    Ok(deps)
}

fn features_of(root: &Table) -> BTreeMap<String, Vec<String>> {
    let table = root
        .get("features")
        .and_then(Value::as_table)
        .into_iter()
        .flatten();
    table
        .map(|(name, values)| (name.clone(), strings(Some(values))))
        .collect()
}

fn build_script(dir: &Path, package: &Table) -> Option<PathBuf> {
    match package.get("build") {
        Some(Value::Boolean(false)) => None,
        Some(Value::String(path)) => Some(dir.join(path)),
        _ => Some(dir.join("build.rs")).filter(|path| path.is_file()),
    }
}

fn lib_of(dir: &Path, root: &Table) -> (Option<PathBuf>, bool) {
    let lib = root.get("lib").and_then(Value::as_table);
    let proc_macro = lib
        .and_then(|lib| flag(lib, &["proc-macro", "proc_macro"]))
        .unwrap_or(false);
    let path = lib
        .and_then(|lib| text(lib, "path"))
        .map(|path| dir.join(path));
    let path = path.or_else(|| Some(dir.join("src/lib.rs")).filter(|path| path.is_file()));
    (path, proc_macro)
}

fn targets_of(
    dir: &Path,
    root: &Table,
    package: &Table,
    (table_key, auto_key, folder): (&str, &str, &str),
) -> Vec<TargetSpec> {
    let listed = root
        .get(table_key)
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_table);
    let mut found: Vec<TargetSpec> = listed
        .filter_map(|entry| {
            let target_name = text(entry, "name")?.to_owned();
            let path = text(entry, "path").map_or_else(
                || dir.join(format!("{folder}/{target_name}.rs")),
                |path| dir.join(path),
            );
            let required_features = strings(
                entry
                    .get("required-features")
                    .or_else(|| entry.get("required_features")),
            );
            Some(TargetSpec {
                name: target_name,
                path,
                required_features,
            })
        })
        .collect();
    if !flag(package, &[auto_key]).unwrap_or(true) {
        return found;
    }
    let Ok(entries) = std::fs::read_dir(dir.join(folder)) else {
        return found;
    };
    let mut auto: Vec<(String, PathBuf)> = entries
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter_map(|path| {
            if path.is_file() && path.extension().and_then(|e| e.to_str()) == Some("rs") {
                Some((path.file_stem()?.to_str()?.to_owned(), path))
            } else if path.is_dir() && path.join("main.rs").is_file() {
                Some((path.file_name()?.to_str()?.to_owned(), path.join("main.rs")))
            } else {
                None
            }
        })
        .collect();
    auto.sort();
    for (name, path) in auto {
        if !found.iter().any(|existing| existing.path == path) {
            found.push(TargetSpec {
                name,
                path,
                required_features: Vec::new(),
            });
        }
    }
    found
}

fn bins_of(dir: &Path, name: &str, root: &Table, package: &Table) -> Vec<TargetSpec> {
    let mut bins = targets_of(dir, root, package, ("bin", "autobins", "src/bin"));
    let auto = flag(package, &["autobins"]).unwrap_or(true);
    let main = dir.join("src/main.rs");
    if auto && main.is_file() && !bins.iter().any(|t| t.path == main) {
        bins.insert(
            0,
            TargetSpec {
                name: name.to_owned(),
                path: main,
                required_features: Vec::new(),
            },
        );
    }
    bins
}

/// Reads the manifest at `dir/Cargo.toml`, inheriting from `workspace` (the root's `[workspace]` table).
pub fn read(dir: &Path, workspace: Option<&Table>) -> Result<Manifest> {
    let path = dir.join(super::MANIFEST_FILE);
    let text_of =
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let root: Table =
        toml::from_str(&text_of).with_context(|| format!("parsing {}", path.display()))?;
    manifest_of(dir, &root, workspace)
}

/// Reads the package manifest enclosing `file`, inheriting any `[workspace]` fields from an ancestor manifest.
pub fn read_for_file(file: &Path) -> Option<Manifest> {
    let abs = std::path::absolute(file).ok()?;
    let norm = normalize(&abs);
    let start = if norm.is_dir() {
        norm.as_path()
    } else {
        norm.parent()?
    };
    let mut pkg_dir = None;
    let mut ws_table = None;
    for dir in start.ancestors() {
        let candidate = dir.join(super::MANIFEST_FILE);
        if !candidate.is_file() {
            continue;
        }
        if pkg_dir.is_none()
            && std::fs::read_to_string(&candidate)
                .ok()
                .and_then(|s| toml::from_str::<Table>(&s).ok())
                .is_some_and(|t| t.contains_key("package"))
        {
            pkg_dir = Some(dir.to_path_buf());
        }
        if ws_table.is_none()
            && let Ok(Some(ws)) = workspace_of(dir)
        {
            ws_table = Some(ws);
            if pkg_dir.is_some() {
                break;
            }
        }
    }
    read(&pkg_dir?, ws_table.as_ref()).ok()
}

/// Reads the `[workspace]` table of the manifest at `dir/Cargo.toml`, if it has one, with its dependency paths made absolute.
pub fn workspace_of(dir: &Path) -> Result<Option<Table>> {
    let path = dir.join(super::MANIFEST_FILE);
    let root: Table = toml::from_str(&std::fs::read_to_string(&path)?)
        .with_context(|| format!("parsing {}", path.display()))?;
    let Some(mut workspace) = root.get("workspace").and_then(Value::as_table).cloned() else {
        return Ok(None);
    };
    let deps = workspace
        .get_mut("dependencies")
        .and_then(Value::as_table_mut)
        .into_iter()
        .flat_map(|deps| deps.iter_mut());
    for entry in deps.filter_map(|(_, spec)| spec.as_table_mut()) {
        let Some(relative) = entry.get("path").and_then(Value::as_str) else {
            continue;
        };
        let absolute = normalize(&dir.join(relative))
            .to_string_lossy()
            .into_owned();
        entry.insert("path".to_owned(), Value::String(absolute));
    }
    Ok(Some(workspace))
}

/// Reads `[profile.dev]` and `[profile.release]` overrides from `dir/Cargo.toml`, if present.
pub fn profiles_of(dir: &Path) -> (ProfileOverride, ProfileOverride) {
    let parse = |table: &Table, name: &str| {
        let Some(prof) = table
            .get("profile")
            .and_then(Value::as_table)
            .and_then(|p| p.get(name))
            .and_then(Value::as_table)
        else {
            return ProfileOverride::default();
        };
        ProfileOverride {
            debug_assertions: flag(prof, &["debug-assertions", "debug_assertions"]),
            overflow_checks: flag(prof, &["overflow-checks", "overflow_checks"]),
        }
    };
    let Ok(text_of) = std::fs::read_to_string(dir.join(super::MANIFEST_FILE)) else {
        return Default::default();
    };
    let Ok(root) = toml::from_str::<Table>(&text_of) else {
        return Default::default();
    };
    (parse(&root, "dev"), parse(&root, "release"))
}

fn inherited_value<'a>(
    package: &'a Table,
    workspace: Option<&'a Table>,
    key: &str,
) -> Option<&'a Value> {
    let val = package.get(key)?;
    if val.as_table().is_some_and(is_workspace_inherited) {
        workspace
            .and_then(|ws| ws.get("package"))
            .and_then(Value::as_table)
            .and_then(|wp| wp.get(key))
    } else {
        Some(val)
    }
}

fn inherited_text_or(
    package: &Table,
    workspace: Option<&Table>,
    key: &str,
    default: &str,
) -> String {
    inherited_value(package, workspace, key)
        .and_then(Value::as_str)
        .unwrap_or(default)
        .to_owned()
}

fn inherited_text(package: &Table, workspace: Option<&Table>, key: &str) -> String {
    inherited_text_or(package, workspace, key, "")
}

fn edition_of(package: &Table, workspace: Option<&Table>) -> String {
    inherited_text_or(package, workspace, "edition", "2015")
}

fn resolver_of(table: &Table) -> Option<u8> {
    match text(table, "resolver")? {
        "1" => Some(1),
        "2" => Some(2),
        "3" => Some(3),
        _ => None,
    }
}

fn is_glob_pattern(pattern: &str) -> bool {
    pattern.bytes().any(|b| matches!(b, b'*' | b'?' | b'['))
}

fn match_char_class(p: &[char], pi: usize, ch: char) -> Option<(usize, bool)> {
    let mut i = pi + 1;
    let negated = matches!(p.get(i), Some('!' | '^'));
    if negated {
        i += 1;
    }
    let start = i;
    let mut matched = false;
    while i < p.len() {
        if p[i] == ']' && i > start {
            return Some((i + 1, if negated { !matched } else { matched }));
        }
        if i + 2 < p.len() && p[i + 1] == '-' && p[i + 2] != ']' {
            if ch >= p[i] && ch <= p[i + 2] {
                matched = true;
            }
            i += 3;
        } else {
            if ch == p[i] {
                matched = true;
            }
            i += 1;
        }
    }
    None
}

fn wildcard_match(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    let (mut pi, mut ti) = (0usize, 0usize);
    let (mut star, mut match_ti) = (None, 0usize);
    while ti < t.len() {
        if pi < p.len() && p[pi] == '*' {
            star = Some(pi);
            match_ti = ti;
            pi += 1;
            continue;
        }
        let step = if pi < p.len() && p[pi] == '?' {
            Some(pi + 1)
        } else if pi < p.len() && p[pi] == '[' {
            match match_char_class(&p, pi, t[ti]) {
                Some((next_pi, true)) => Some(next_pi),
                Some((_, false)) => None,
                None if t[ti] == '[' => Some(pi + 1),
                None => None,
            }
        } else if pi < p.len() && p[pi] == t[ti] {
            Some(pi + 1)
        } else {
            None
        };
        if let Some(next_pi) = step {
            pi = next_pi;
            ti += 1;
        } else if let Some(s) = star {
            pi = s + 1;
            match_ti += 1;
            ti = match_ti;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == '*' {
        pi += 1;
    }
    pi == p.len()
}

fn advance_pattern_segment(dirs: Vec<PathBuf>, seg: &str) -> Vec<PathBuf> {
    let mut next = Vec::new();
    if is_glob_pattern(seg) {
        for dir in dirs {
            for sub in super::cache::subdirs(&dir).unwrap_or_default() {
                if sub
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|name| wildcard_match(seg, name))
                {
                    next.push(sub);
                }
            }
        }
    } else {
        for dir in dirs {
            let candidate = dir.join(seg);
            if candidate.is_dir() {
                next.push(candidate);
            }
        }
    }
    next
}

fn expand_pattern(top: &Path, pattern: &str) -> Vec<PathBuf> {
    let trimmed = pattern.trim_end_matches('/');
    if !is_glob_pattern(trimmed) {
        let path = if trimmed.is_empty() || trimmed == "." {
            top.to_owned()
        } else {
            top.join(trimmed)
        };
        return vec![path];
    }
    let mut dirs = vec![top.to_owned()];
    for seg in trimmed.split('/') {
        if seg.is_empty() || seg == "." {
            continue;
        }
        dirs = advance_pattern_segment(dirs, seg);
    }
    dirs
}

/// Workspace member directories listed under `key`, honoring globs and `[workspace].exclude`.
pub(crate) fn workspace_member_dirs(
    top: &Path,
    workspace: Option<&Table>,
    key: &str,
) -> Vec<PathBuf> {
    let top_norm = normalize(top);
    let excluded: Vec<PathBuf> = workspace
        .and_then(|ws| ws.get("exclude"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .flat_map(|pat| expand_pattern(top, pat))
        .map(|dir| normalize(&dir))
        .collect();
    let is_excluded = |dir: &Path| {
        let norm = normalize(dir);
        excluded
            .iter()
            .any(|ex| norm == *ex || (*ex != top_norm && norm.starts_with(ex)))
    };
    let listed = workspace
        .and_then(|ws| ws.get(key))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str);
    let mut out = Vec::new();
    for member in listed {
        let is_glob = is_glob_pattern(member);
        for dir in expand_pattern(top, member) {
            if is_glob && !dir.join(super::MANIFEST_FILE).is_file() {
                continue;
            }
            if !is_excluded(&dir) && !out.contains(&dir) {
                out.push(dir);
            }
        }
    }
    out
}

/// Parent and intermediate directories of workspace `members` / `default-members` glob entries.
pub(crate) fn member_glob_dirs(top: &Path) -> Vec<PathBuf> {
    let Ok(Some(ws)) = workspace_of(top) else {
        return Vec::new();
    };
    let mut watched = Vec::new();
    for key in ["members", "default-members"] {
        let listed = ws
            .get(key)
            .and_then(toml::Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(toml::Value::as_str);
        for member in listed {
            let trimmed = member.trim_end_matches('/');
            if !is_glob_pattern(trimmed) {
                continue;
            }
            let mut dirs = vec![top.to_owned()];
            let mut seen_glob = false;
            for seg in trimmed.split('/') {
                if seg.is_empty() || seg == "." {
                    continue;
                }
                if is_glob_pattern(seg) {
                    seen_glob = true;
                }
                if seen_glob {
                    watched.extend(dirs.iter().filter(|dir| dir.is_dir()).cloned());
                }
                dirs = advance_pattern_segment(dirs, seg);
            }
        }
    }
    watched.sort();
    watched.dedup();
    watched
}

fn package_uses_resolver_v2(manifest: &Manifest) -> bool {
    match manifest.resolver {
        Some(1) => false,
        Some(2 | 3) => true,
        _ => manifest
            .edition
            .parse::<u32>()
            .is_ok_and(|year| year >= 2021),
    }
}

/// Whether feature resolution uses Cargo's v2/v3 host/target split for `workspace` and `members`.
pub(super) fn uses_resolver_v2(
    workspace: Option<&Table>,
    top: &Path,
    packages: &[super::Package],
    members: &[usize],
) -> bool {
    if let Some(ws) = workspace {
        if let Some(r) = resolver_of(ws) {
            return matches!(r, 2 | 3);
        }
        let top_norm = normalize(top);
        if let Some(root) = packages
            .iter()
            .find(|p| !p.locked.is_registry() && normalize(&p.manifest.dir) == top_norm)
        {
            return package_uses_resolver_v2(&root.manifest);
        }
        if let Ok(root_manifest) = read(&top_norm, Some(ws)) {
            return package_uses_resolver_v2(&root_manifest);
        }
        return false;
    }
    let Some(&member) = members.first() else {
        return true;
    };
    package_uses_resolver_v2(&packages[member].manifest)
}

fn package_meta(package: &Table, workspace: Option<&Table>) -> PackageMeta {
    let authors = strings(inherited_value(package, workspace, "authors")).join(":");
    PackageMeta {
        authors,
        description: inherited_text(package, workspace, "description"),
        homepage: inherited_text(package, workspace, "homepage"),
        repository: inherited_text(package, workspace, "repository"),
        license: inherited_text(package, workspace, "license"),
        license_file: inherited_text(package, workspace, "license-file"),
        readme: inherited_text(package, workspace, "readme"),
        rust_version: inherited_text(package, workspace, "rust-version"),
    }
}

fn manifest_of(dir: &Path, root: &Table, workspace: Option<&Table>) -> Result<Manifest> {
    let empty = Table::new();
    let package = root
        .get("package")
        .and_then(Value::as_table)
        .unwrap_or(&empty);
    let workspace_deps = workspace
        .and_then(|ws| ws.get("dependencies"))
        .and_then(Value::as_table);
    let name = text(package, "name")
        .context("manifest has no package name")?
        .to_owned();
    let version = inherited_value(package, workspace, "version")
        .and_then(Value::as_str)
        .map(semver::Version::parse)
        .transpose()
        .context("parsing package version")?;
    let (lib, proc_macro) = lib_of(dir, root);
    let scope = Scope {
        dir,
        workspace: workspace_deps,
    };
    Ok(Manifest {
        dir: dir.to_owned(),
        version,
        edition: edition_of(package, workspace),
        resolver: resolver_of(package),
        meta: package_meta(package, workspace),
        links: text(package, "links").map(str::to_owned),
        default_run: text(package, "default-run").map(str::to_owned),
        build: build_script(dir, package),
        lib,
        lib_name: root
            .get("lib")
            .and_then(Value::as_table)
            .and_then(|lib| text(lib, "name"))
            .map(str::to_owned),
        proc_macro,
        bins: bins_of(dir, &name, root, package),
        examples: targets_of(dir, root, package, ("example", "autoexamples", "examples")),
        tests: targets_of(dir, root, package, ("test", "autotests", "tests")),
        benches: targets_of(dir, root, package, ("bench", "autobenches", "benches")),
        features: features_of(root),
        deps: all_deps(&scope, root)?,
        name,
    })
}

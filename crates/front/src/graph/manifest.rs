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

/// The parts of a `Cargo.toml` that the graph needs.
#[derive(Clone, Debug)]
pub struct Manifest {
    pub dir: PathBuf,
    pub name: String,
    pub edition: String,
    pub links: Option<String>,
    pub build: Option<PathBuf>,
    pub lib: Option<PathBuf>,
    pub lib_name: Option<String>,
    pub proc_macro: bool,
    pub bins: Vec<(String, PathBuf)>,
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
    keys.iter().find_map(|key| table.get(*key).and_then(Value::as_bool))
}

fn strings(value: Option<&Value>) -> Vec<String> {
    let items = value.and_then(Value::as_array).map(Vec::as_slice).unwrap_or_default();
    items.iter().filter_map(Value::as_str).map(str::to_owned).collect()
}

fn merged(spec: &Value, workspace: Option<&Table>, key: &str) -> Table {
    let mut table = match spec {
        Value::String(req) => Table::from_iter([("version".to_owned(), Value::String(req.clone()))]),
        Value::Table(table) => table.clone(),
        _ => Table::new(),
    };
    if table.get("workspace").and_then(Value::as_bool) == Some(true)
        && let Some(base) = workspace.and_then(|ws| ws.get(key))
    {
        let extra = strings(table.get("features"));
        let mut inherited = merged(base, None, key);
        let mut all = strings(inherited.get("features"));
        all.extend(extra);
        inherited.insert("features".to_owned(), Value::Array(all.into_iter().map(Value::String).collect()));
        if let Some(optional) = table.get("optional") {
            inherited.insert("optional".to_owned(), optional.clone());
        }
        table = inherited;
    }
    table
}

struct Scope<'a> {
    dir: &'a Path,
    workspace: Option<&'a Table>,
}

fn dep(scope: &Scope<'_>, key: &str, spec: &Value, kind: Kind, platform: Option<&str>) -> Result<Dep> {
    let table = merged(spec, scope.workspace, key);
    let req = text(&table, "version").map(semver::VersionReq::parse).transpose().with_context(|| format!("version of {key}"))?;
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

fn deps_of(scope: &Scope<'_>, table: &Table, platform: Option<&str>, into: &mut Vec<Dep>) -> Result<()> {
    for (name, kind) in KIND_TABLES {
        let Some(entries) = table.get(name).and_then(Value::as_table) else { continue };
        for (key, spec) in entries {
            into.push(dep(scope, key, spec, kind, platform)?);
        }
    }
    Ok(())
}

fn all_deps(scope: &Scope<'_>, root: &Table) -> Result<Vec<Dep>> {
    let mut deps = Vec::new();
    deps_of(scope, root, None, &mut deps)?;
    let targets = root.get("target").and_then(Value::as_table).into_iter().flatten();
    for (platform, table) in targets {
        if let Some(table) = table.as_table() {
            deps_of(scope, table, Some(platform), &mut deps)?;
        }
    }
    Ok(deps)
}

fn features_of(root: &Table) -> BTreeMap<String, Vec<String>> {
    let table = root.get("features").and_then(Value::as_table).into_iter().flatten();
    table.map(|(name, values)| (name.clone(), strings(Some(values)))).collect()
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
    let proc_macro = lib.and_then(|lib| flag(lib, &["proc-macro", "proc_macro"])).unwrap_or(false);
    let path = lib.and_then(|lib| text(lib, "path")).map(|path| dir.join(path));
    let path = path.or_else(|| Some(dir.join("src/lib.rs")).filter(|path| path.is_file()));
    (path, proc_macro)
}

fn bins_of(dir: &Path, name: &str, root: &Table, package: &Table) -> Vec<(String, PathBuf)> {
    let listed = root.get("bin").and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_table);
    let mut bins: Vec<(String, PathBuf)> = listed
        .filter_map(|bin| {
            let bin_name = text(bin, "name")?.to_owned();
            let path = text(bin, "path").map_or_else(|| dir.join(format!("src/bin/{bin_name}.rs")), |path| dir.join(path));
            Some((bin_name, path))
        })
        .collect();
    let auto = flag(package, &["autobins"]).unwrap_or(true);
    let main = dir.join("src/main.rs");
    if auto && main.is_file() && !bins.iter().any(|(_, path)| *path == main) {
        bins.push((name.to_owned(), main));
    }
    bins
}

/// Reads the manifest at `dir/Cargo.toml`, inheriting from `workspace` (the root's `[workspace]` table).
pub fn read(dir: &Path, workspace: Option<&Table>) -> Result<Manifest> {
    let path = dir.join("Cargo.toml");
    let text_of = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let root: Table = toml::from_str(&text_of).with_context(|| format!("parsing {}", path.display()))?;
    manifest_of(dir, &root, workspace)
}

/// Reads the `[workspace]` table of the manifest at `dir/Cargo.toml`, if it has one.
pub fn workspace_of(dir: &Path) -> Result<Option<Table>> {
    let path = dir.join("Cargo.toml");
    let root: Table = toml::from_str(&std::fs::read_to_string(&path)?).with_context(|| format!("parsing {}", path.display()))?;
    Ok(root.get("workspace").and_then(Value::as_table).cloned())
}

fn edition_of(package: &Table, workspace: Option<&Table>) -> String {
    let inherited = || workspace.and_then(|ws| ws.get("package")).and_then(Value::as_table).and_then(|package| text(package, "edition"));
    text(package, "edition").or_else(inherited).unwrap_or("2015").to_owned()
}

fn manifest_of(dir: &Path, root: &Table, workspace: Option<&Table>) -> Result<Manifest> {
    let empty = Table::new();
    let package = root.get("package").and_then(Value::as_table).unwrap_or(&empty);
    let workspace_deps = workspace.and_then(|ws| ws.get("dependencies")).and_then(Value::as_table);
    let name = text(package, "name").context("manifest has no package name")?.to_owned();
    let (lib, proc_macro) = lib_of(dir, root);
    let scope = Scope { dir, workspace: workspace_deps };
    Ok(Manifest {
        dir: dir.to_owned(),
        edition: edition_of(package, workspace),
        links: text(package, "links").map(str::to_owned),
        build: build_script(dir, package),
        lib,
        lib_name: root.get("lib").and_then(Value::as_table).and_then(|lib| text(lib, "name")).map(str::to_owned),
        proc_macro,
        bins: bins_of(dir, &name, root, package),
        features: features_of(root),
        deps: all_deps(&scope, root)?,
        name,
    })
}

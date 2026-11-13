//! Build scripts as Cargo runs them: the directives a script prints and the environment it runs in.

use std::collections::BTreeSet;

use super::{Targets, Workspace};

/// What a build script printed that changes how its package builds.
#[derive(Clone, Debug, Default)]
pub struct Output {
    pub names: BTreeSet<String>,
    pub pairs: BTreeSet<(String, String)>,
    pub vars: Vec<(String, String)>,
    pub libs: Vec<String>,
    pub search: Vec<String>,
    pub metadata: Vec<(String, String)>,
    /// The `rerun-if-changed` paths, relative to the package.
    pub watched: Vec<String>,
    /// The `rerun-if-env-changed` variables.
    pub watched_vars: Vec<String>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Directive {
    Cfg,
    Env,
    LinkLib,
    LinkSearch,
    Metadata,
    Watch,
    WatchVar,
    Other,
}

const DIRECTIVES: [(&str, Directive); 19] = [
    ("rustc-cfg", Directive::Cfg),
    ("rustc-env", Directive::Env),
    ("rustc-link-lib", Directive::LinkLib),
    ("rustc-link-search", Directive::LinkSearch),
    ("rerun-if-changed", Directive::Watch),
    ("metadata", Directive::Metadata),
    ("rerun-if-env-changed", Directive::WatchVar),
    ("rustc-link-arg", Directive::Other),
    ("rustc-link-arg-cdylib", Directive::Other),
    ("rustc-cdylib-link-arg", Directive::Other),
    ("rustc-link-arg-bin", Directive::Other),
    ("rustc-link-arg-bins", Directive::Other),
    ("rustc-link-arg-tests", Directive::Other),
    ("rustc-link-arg-examples", Directive::Other),
    ("rustc-link-arg-benches", Directive::Other),
    ("rustc-flags", Directive::Other),
    ("rustc-check-cfg", Directive::Other),
    ("error", Directive::Other),
    ("warning", Directive::Other),
];

impl Output {
    /// Reads the `cargo::KEY=VALUE` lines of a script's standard output, and the older `cargo:KEY=VALUE` lines, where an unknown key is metadata.
    pub fn parse(text: &str) -> Output {
        let mut output = Output::default();
        for line in text.lines() {
            let (body, legacy) = match line.strip_prefix("cargo::") {
                Some(body) => (body, false),
                None => match line.strip_prefix("cargo:") {
                    Some(body) => (body, true),
                    None => continue,
                },
            };
            let Some((key, value)) = body.split_once('=') else {
                continue;
            };
            let directive = DIRECTIVES.iter().find(|(known, _)| *known == key).map(|(_, directive)| *directive);
            match directive {
                Some(directive) => output.apply(directive, value),
                None if legacy => output.metadata.push((key.to_owned(), value.to_owned())),
                None => {}
            }
        }
        output
    }

    fn apply(&mut self, directive: Directive, value: &str) {
        match directive {
            Directive::Cfg => match value.split_once('=') {
                Some((key, quoted)) => _ = self.pairs.insert((key.trim().to_owned(), quoted.trim().trim_matches('"').to_owned())),
                None => _ = self.names.insert(value.trim().to_owned()),
            },
            Directive::Env => self.vars.extend(value.split_once('=').map(|(key, value)| (key.to_owned(), value.to_owned()))),
            Directive::Metadata => self.metadata.extend(value.split_once('=').map(|(key, value)| (key.to_owned(), value.to_owned()))),
            Directive::LinkLib => self.libs.push(value.to_owned()),
            Directive::LinkSearch => self.search.push(value.to_owned()),
            Directive::Watch => self.watched.push(value.to_owned()),
            Directive::WatchVar => self.watched_vars.push(value.to_owned()),
            Directive::Other => {}
        }
    }
}

fn out_dir(workspace: &Workspace, unit: usize) -> String {
    super::store::dir(workspace, unit).join("out").display().to_string()
}

/// The variables `env!` reads in the crates of `unit`: Cargo's package variables and, when the package has a build script, its `OUT_DIR`.
pub fn crate_vars(workspace: &Workspace, unit: usize) -> Vec<(String, String)> {
    let mut vars = package_vars(workspace, unit);
    if workspace.packages[workspace.graph.units[unit].package].manifest.build.is_some() {
        vars.push(("OUT_DIR".to_owned(), out_dir(workspace, unit)));
    }
    vars
}

/// The variables Cargo sets for the package of `unit` when it builds the package's crates: its name, version and manifest paths.
pub fn package_vars(workspace: &Workspace, unit: usize) -> Vec<(String, String)> {
    let package = &workspace.packages[workspace.graph.units[unit].package];
    let version = &package.locked.version;
    let manifest = &package.manifest;
    let mut vars = vec![
        ("CARGO_PKG_NAME", package.locked.name.clone()),
        ("CARGO_PKG_VERSION", version.to_string()),
        ("CARGO_PKG_VERSION_MAJOR", version.major.to_string()),
        ("CARGO_PKG_VERSION_MINOR", version.minor.to_string()),
        ("CARGO_PKG_VERSION_PATCH", version.patch.to_string()),
        ("CARGO_PKG_VERSION_PRE", version.pre.to_string()),
        ("CARGO_MANIFEST_DIR", manifest.dir.display().to_string()),
        ("CARGO_MANIFEST_PATH", manifest.dir.join("Cargo.toml").display().to_string()),
    ];
    vars.extend(["AUTHORS", "DESCRIPTION", "HOMEPAGE", "REPOSITORY", "LICENSE", "LICENSE_FILE", "README", "RUST_VERSION"].map(|key| (key, String::new())));
    vars.into_iter().map(|(key, value)| (if key.starts_with("CARGO_") { key.to_owned() } else { format!("CARGO_PKG_{key}") }, value)).collect()
}

/// The variables Cargo sets when it runs the build script of `unit`, given the metadata its dependencies' scripts printed.
pub fn script_vars(workspace: &Workspace, targets: &Targets, unit: usize, metadata: &[(String, String)]) -> Vec<(String, String)> {
    let entry = &workspace.graph.units[unit];
    let manifest = &workspace.packages[entry.package].manifest;
    let target = targets.of(entry.side);
    let tool = |name: &str| std::env::var(name).unwrap_or_else(|_| name.to_lowercase());
    let mut vars = package_vars(workspace, unit);
    vars.extend(entry.features.iter().map(|feature| (format!("CARGO_FEATURE_{}", upper(feature)), "1".to_owned())));
    vars.extend(cfg_vars(target));
    vars.extend(manifest.links.iter().map(|links| ("CARGO_MANIFEST_LINKS".to_owned(), links.clone())));
    vars.extend(metadata.iter().cloned());
    let fixed = [
        ("OUT_DIR", out_dir(workspace, unit)),
        ("TARGET", target.triple.clone()),
        ("HOST", targets.host.triple.clone()),
        ("NUM_JOBS", crate::jobs().to_string()),
        ("OPT_LEVEL", "0".to_owned()),
        ("DEBUG", "true".to_owned()),
        ("PROFILE", "debug".to_owned()),
        ("CARGO", tool("CARGO")),
        ("RUSTC", tool("RUSTC")),
        ("RUSTDOC", tool("RUSTDOC")),
        ("CARGO_ENCODED_RUSTFLAGS", String::new()),
    ];
    vars.extend(fixed.map(|(key, value)| (key.to_owned(), value)));
    vars
}

/// `DEP_<LINKS>_<KEY>` for each metadata pair that the script of a dependency with `links` printed.
pub fn dep_vars(links: &str, output: &Output) -> Vec<(String, String)> {
    output.metadata.iter().map(|(key, value)| (format!("DEP_{}_{}", upper(links), upper(key)), value.clone())).collect()
}

fn upper(name: &str) -> String {
    name.to_uppercase().replace('-', "_")
}

fn cfg_vars(target: &super::cfg::Target) -> Vec<(String, String)> {
    let mut vars: Vec<(String, String)> = target.names.iter().chain(["debug_assertions".to_owned()].iter()).map(|name| (format!("CARGO_CFG_{}", upper(name)), String::new())).collect();
    let mut joined: std::collections::BTreeMap<&str, Vec<&str>> = std::collections::BTreeMap::new();
    for (key, value) in &target.pairs {
        joined.entry(key).or_default().push(value);
    }
    vars.extend(joined.into_iter().map(|(key, values)| (format!("CARGO_CFG_{}", upper(key)), values.join(","))));
    vars
}

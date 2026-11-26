use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use super::lock::Locked;
use super::{MANIFEST_FILE, manifest, manifest_name};

#[derive(Clone, Debug, Default)]
struct SourceEntry {
    registry: Option<String>,
    git: Option<String>,
    directory: Option<PathBuf>,
    replace_with: Option<String>,
}

/// Resolved Cargo `[source.*]` configuration, offline state, `[build].rustflags`, `[env]`, and local registry cache directories.
#[derive(Clone, Debug)]
pub(super) struct SourceCatalog {
    sources: BTreeMap<String, SourceEntry>,
    default_vendor: Option<PathBuf>,
    registry_dirs: Vec<PathBuf>,
    pub(super) offline: bool,
    pub(super) rustflags: Vec<String>,
    pub(super) env: Vec<(String, String, bool)>,
}

enum ResolvedSource<'a> {
    Directory(&'a Path),
    Registry {
        host: Option<String>,
        is_crates_io: bool,
    },
}

fn env_path(name: &str) -> Option<PathBuf> {
    if crate::client_env().is_some() {
        crate::process_env_var(name).map(PathBuf::from)
    } else {
        std::env::var_os(name).map(PathBuf::from)
    }
}

pub(super) fn cargo_home() -> Result<PathBuf> {
    let home = env_path("CARGO_HOME").or_else(|| env_path("HOME").map(|home| home.join(".cargo")));
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

pub(super) fn config_paths(dir: &Path) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = cargo_configs(dir)
        .into_iter()
        .map(|(path, _)| path)
        .collect();
    paths.push(dir.join(".cargo/config.toml"));
    paths.push(dir.join(".cargo/config"));
    paths
}

fn env_offline() -> bool {
    crate::process_env_var("CARGO_NET_OFFLINE")
        .is_some_and(|v| matches!(v.to_ascii_lowercase().as_str(), "true" | "1"))
}

fn parse_rustflags_value(raw: &toml::Value) -> Option<Vec<String>> {
    match raw {
        toml::Value::Array(items) => Some(
            items
                .iter()
                .filter_map(toml::Value::as_str)
                .map(str::to_owned)
                .collect(),
        ),
        toml::Value::String(text) => Some(text.split_whitespace().map(str::to_owned).collect()),
        _ => None,
    }
}

fn parse_env_entry(val: &toml::Value, base: &Path) -> Option<(String, bool)> {
    match val {
        toml::Value::String(s) => Some((s.clone(), false)),
        toml::Value::Table(t) => {
            let raw = t.get("value").and_then(toml::Value::as_str)?;
            let force = t
                .get("force")
                .and_then(toml::Value::as_bool)
                .unwrap_or(false);
            let relative = t
                .get("relative")
                .and_then(toml::Value::as_bool)
                .unwrap_or(false);
            let value = if relative {
                manifest::normalize(&base.join(raw))
                    .to_string_lossy()
                    .into_owned()
            } else {
                raw.to_owned()
            };
            Some((value, force))
        }
        _ => None,
    }
}

pub(super) fn resolved_rustflags(catalog: &SourceCatalog) -> Vec<String> {
    if let Some(raw) = crate::process_env_var("CARGO_ENCODED_RUSTFLAGS") {
        return if raw.is_empty() {
            Vec::new()
        } else {
            raw.split('\x1f').map(str::to_owned).collect()
        };
    }
    if let Some(raw) = crate::process_env_var("RUSTFLAGS") {
        return raw.split_whitespace().map(str::to_owned).collect();
    }
    catalog.rustflags.clone()
}

fn normalize_url(url: &str) -> &str {
    let stripped = url
        .strip_prefix("registry+")
        .or_else(|| url.strip_prefix("sparse+"))
        .or_else(|| url.strip_prefix("git+"))
        .unwrap_or(url);
    stripped.trim_end_matches('/')
}

fn is_crates_io_url(url: &str) -> bool {
    let norm = normalize_url(url);
    norm == "https://github.com/rust-lang/crates.io-index" || norm == "https://index.crates.io"
}

fn url_host(url: &str) -> Option<String> {
    let norm = normalize_url(url);
    let after_scheme = norm.split_once("://").map_or(norm, |(_, rest)| rest);
    let host_port = after_scheme.split('/').next()?;
    let host = host_port
        .rsplit_once('@')
        .map_or(host_port, |(_, h)| h)
        .split(':')
        .next()?;
    (!host.is_empty()).then(|| host.to_owned())
}

fn merge_sources(src_tbl: &toml::Table, base: &Path, sources: &mut BTreeMap<String, SourceEntry>) {
    for (name, val) in src_tbl {
        let Some(entry_tbl) = val.as_table() else {
            continue;
        };
        let entry = sources.entry(name.clone()).or_default();
        let text = |key: &str| {
            entry_tbl
                .get(key)
                .and_then(toml::Value::as_str)
                .map(str::to_owned)
        };
        entry.registry = entry.registry.take().or_else(|| text("registry"));
        entry.git = entry.git.take().or_else(|| text("git"));
        entry.directory = entry
            .directory
            .take()
            .or_else(|| text("directory").map(|p| manifest::normalize(&base.join(p))));
        entry.replace_with = entry.replace_with.take().or_else(|| text("replace-with"));
    }
}

impl SourceCatalog {
    pub(super) fn load(top: &Path) -> Result<Self> {
        let mut sources = BTreeMap::new();
        let (mut cfg_offline, mut rustflags, mut env_map) = (false, None, BTreeMap::new());
        for (config, base) in cargo_configs(top) {
            let table: toml::Table = toml::from_str(&std::fs::read_to_string(&config)?)
                .with_context(|| format!("parsing {}", config.display()))?;
            cfg_offline |= table
                .get("net")
                .and_then(|n| n.get("offline"))
                .and_then(toml::Value::as_bool)
                .unwrap_or(false);
            if rustflags.is_none()
                && let Some(raw) = table.get("build").and_then(|b| b.get("rustflags"))
            {
                rustflags = parse_rustflags_value(raw);
            }
            if let Some(env_tbl) = table.get("env").and_then(toml::Value::as_table) {
                for (key, val) in env_tbl {
                    if !env_map.contains_key(key)
                        && let Some(entry) = parse_env_entry(val, &base)
                    {
                        env_map.insert(key.clone(), entry);
                    }
                }
            }
            if let Some(src_tbl) = table.get("source").and_then(toml::Value::as_table) {
                merge_sources(src_tbl, &base, &mut sources);
            }
        }
        let vendor = top.join("vendor");
        Ok(Self {
            sources,
            default_vendor: vendor.is_dir().then_some(vendor),
            registry_dirs: registry_dirs()?,
            offline: env_offline() || cfg_offline || crate::selection().offline,
            rustflags: rustflags.unwrap_or_default(),
            env: env_map.into_iter().map(|(k, (v, f))| (k, v, f)).collect(),
        })
    }

    fn resolve_entry(&self, locked: &Locked) -> Option<&SourceEntry> {
        let raw_source = locked.source.as_deref()?;
        let norm_source = normalize_url(raw_source);
        let initial = if let Some((git_url, _)) = locked.git() {
            let norm_git = normalize_url(git_url);
            self.sources.values().find(|entry| {
                entry
                    .git
                    .as_deref()
                    .is_some_and(|g| normalize_url(g) == norm_git)
            })
        } else if is_crates_io_url(raw_source) {
            self.sources.get("crates-io").or_else(|| {
                self.sources.values().find(|entry| {
                    entry
                        .registry
                        .as_deref()
                        .is_some_and(|r| is_crates_io_url(r))
                })
            })
        } else {
            self.sources.values().find(|entry| {
                entry
                    .registry
                    .as_deref()
                    .is_some_and(|r| normalize_url(r) == norm_source)
            })
        };
        let mut current = initial?;
        for _ in 0..16 {
            let Some(next_name) = current.replace_with.as_deref() else {
                break;
            };
            let Some(next) = self.sources.get(next_name) else {
                break;
            };
            current = next;
        }
        Some(current)
    }

    fn resolve_source(&self, locked: &Locked) -> ResolvedSource<'_> {
        if let Some(entry) = self.resolve_entry(locked) {
            if let Some(dir) = entry.directory.as_deref() {
                return ResolvedSource::Directory(dir);
            }
            if let Some(reg) = entry.registry.as_deref() {
                return ResolvedSource::Registry {
                    host: url_host(reg),
                    is_crates_io: is_crates_io_url(reg),
                };
            }
        }
        let raw = locked.source.as_deref().unwrap_or_default();
        ResolvedSource::Registry {
            host: url_host(raw),
            is_crates_io: is_crates_io_url(raw),
        }
    }
}

fn manifest_version(dir: &Path) -> Option<semver::Version> {
    let table: toml::Table =
        toml::from_str(&std::fs::read_to_string(dir.join(MANIFEST_FILE)).ok()?).ok()?;
    let raw = table.get("package")?.get("version")?.as_str()?;
    semver::Version::parse(raw).ok()
}

struct JsonCursor<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> JsonCursor<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            bytes: text.as_bytes(),
            at: 0,
        }
    }

    fn skip_ws(&mut self) {
        while matches!(self.bytes.get(self.at), Some(b' ' | b'\t' | b'\r' | b'\n')) {
            self.at += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.at).copied()
    }

    fn eat(&mut self, byte: u8) -> Option<()> {
        (self.peek() == Some(byte)).then(|| self.at += 1)
    }

    fn eat_literal(&mut self, lit: &[u8]) -> Option<()> {
        self.bytes
            .get(self.at..self.at + lit.len())
            .filter(|slice| *slice == lit)
            .map(|_| self.at += lit.len())
    }

    fn read_escape(&mut self) -> Option<char> {
        let esc = *self.bytes.get(self.at)?;
        self.at += 1;
        match esc {
            b'"' => Some('"'),
            b'\\' => Some('\\'),
            b'/' => Some('/'),
            b'b' => Some('\x08'),
            b'f' => Some('\x0c'),
            b'n' => Some('\n'),
            b'r' => Some('\r'),
            b't' => Some('\t'),
            b'u' => {
                let hex = std::str::from_utf8(self.bytes.get(self.at..self.at + 4)?).ok()?;
                let code = u32::from_str_radix(hex, 16).ok()?;
                self.at += 4;
                Some(char::from_u32(code).unwrap_or(char::REPLACEMENT_CHARACTER))
            }
            _ => None,
        }
    }

    fn read_str(&mut self) -> Option<String> {
        self.eat(b'"')?;
        let start = self.at;
        while let Some(&b) = self.bytes.get(self.at) {
            if b == b'"' {
                let s = std::str::from_utf8(&self.bytes[start..self.at])
                    .ok()?
                    .to_owned();
                self.at += 1;
                return Some(s);
            }
            if b == b'\\' || b < 0x20 {
                break;
            }
            self.at += 1;
        }
        let mut out = std::str::from_utf8(&self.bytes[start..self.at])
            .ok()?
            .to_owned();
        loop {
            let rest_start = self.at;
            while let Some(&b) = self.bytes.get(self.at) {
                if b == b'"' || b == b'\\' {
                    break;
                }
                if b < 0x20 {
                    return None;
                }
                self.at += 1;
            }
            out.push_str(std::str::from_utf8(&self.bytes[rest_start..self.at]).ok()?);
            match self.bytes.get(self.at).copied()? {
                b'"' => {
                    self.at += 1;
                    return Some(out);
                }
                b'\\' => {
                    self.at += 1;
                    out.push(self.read_escape()?);
                }
                _ => return None,
            }
        }
    }

    fn skip_number(&mut self) -> Option<()> {
        if self.peek() == Some(b'-') {
            self.at += 1;
        }
        match self.bytes.get(self.at).copied()? {
            b'0' => self.at += 1,
            b'1'..=b'9' => {
                while self.bytes.get(self.at).is_some_and(u8::is_ascii_digit) {
                    self.at += 1;
                }
            }
            _ => return None,
        }
        if self.peek() == Some(b'.') {
            self.at += 1;
            if !self.bytes.get(self.at).is_some_and(u8::is_ascii_digit) {
                return None;
            }
            while self.bytes.get(self.at).is_some_and(u8::is_ascii_digit) {
                self.at += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.at += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.at += 1;
            }
            if !self.bytes.get(self.at).is_some_and(u8::is_ascii_digit) {
                return None;
            }
            while self.bytes.get(self.at).is_some_and(u8::is_ascii_digit) {
                self.at += 1;
            }
        }
        Some(())
    }

    fn skip_list(&mut self, open: u8, close: u8, object: bool) -> Option<()> {
        self.eat(open)?;
        self.skip_ws();
        if self.eat(close).is_some() {
            return Some(());
        }
        loop {
            if object {
                self.read_str()?;
                self.skip_ws();
                self.eat(b':')?;
                self.skip_ws();
            }
            self.skip_val()?;
            self.skip_ws();
            if self.eat(b',').is_some() {
                self.skip_ws();
                continue;
            }
            return self.eat(close);
        }
    }

    fn skip_val(&mut self) -> Option<()> {
        match self.peek()? {
            b'"' => self.read_str().map(|_| ()),
            b'{' => self.skip_list(b'{', b'}', true),
            b'[' => self.skip_list(b'[', b']', false),
            b't' => self.eat_literal(b"true"),
            b'f' => self.eat_literal(b"false"),
            b'n' => self.eat_literal(b"null"),
            b'-' | b'0'..=b'9' => self.skip_number(),
            _ => None,
        }
    }

    fn read_package(&mut self) -> Option<Option<String>> {
        self.skip_ws();
        self.eat(b'{')?;
        self.skip_ws();
        if self.eat(b'}').is_some() {
            self.skip_ws();
            return (self.at == self.bytes.len()).then_some(None);
        }
        let mut pkg = None;
        loop {
            let key = self.read_str()?;
            self.skip_ws();
            self.eat(b':')?;
            self.skip_ws();
            if key == "package" {
                pkg = if self.peek() == Some(b'"') {
                    Some(self.read_str()?)
                } else {
                    self.skip_val()?;
                    None
                };
            } else {
                self.skip_val()?;
            }
            self.skip_ws();
            if self.eat(b',').is_some() {
                self.skip_ws();
                continue;
            }
            self.eat(b'}')?;
            self.skip_ws();
            return (self.at == self.bytes.len()).then_some(pkg);
        }
    }
}

fn checksum_matches(dir: &Path, expected: Option<&str>) -> bool {
    let Some(expected) = expected else {
        return true;
    };
    match std::fs::read_to_string(dir.join(".cargo-checksum.json")) {
        Ok(text) => JsonCursor::new(&text).read_package().flatten().as_deref() == Some(expected),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => dir.join(".cargo-ok").is_file(),
        Err(_) => false,
    }
}

fn find_in_dir(locked: &Locked, dir: &Path) -> Option<PathBuf> {
    let versioned = format!("{}-{}", locked.name, locked.version);
    let candidate = dir.join(&versioned);
    if candidate.is_dir() && checksum_matches(&candidate, locked.checksum.as_deref()) {
        return Some(candidate);
    }
    let plain = dir.join(&locked.name);
    if plain.is_dir()
        && manifest_version(&plain).as_ref() == Some(&locked.version)
        && checksum_matches(&plain, locked.checksum.as_deref())
    {
        return Some(plain);
    }
    None
}

fn dir_matches_host(dir: &Path, host: Option<&str>, is_crates_io: bool) -> bool {
    let Some(leaf) = dir.file_name().and_then(|n| n.to_str()) else {
        return false;
    };
    if is_crates_io {
        return leaf.starts_with("index.crates.io-") || leaf.starts_with("github.com-");
    }
    match host {
        Some(h) => leaf.starts_with(&format!("{h}-")),
        None => true,
    }
}

fn find_registry_or_vendor(locked: &Locked, catalog: &SourceCatalog) -> Option<PathBuf> {
    match catalog.resolve_source(locked) {
        ResolvedSource::Directory(dir) => find_in_dir(locked, dir),
        ResolvedSource::Registry { host, is_crates_io } => {
            let matching: Vec<&PathBuf> = catalog
                .registry_dirs
                .iter()
                .filter(|d| dir_matches_host(d, host.as_deref(), is_crates_io))
                .collect();
            let search = if matching.is_empty() {
                catalog.registry_dirs.iter().collect()
            } else {
                matching
            };
            if let Some(found) = search.into_iter().find_map(|d| find_in_dir(locked, d)) {
                return Some(found);
            }
            if let Some(vendor) = catalog.default_vendor.as_deref()
                && let Some(found) = find_in_dir(locked, vendor)
            {
                return Some(found);
            }
            None
        }
    }
}

pub(super) fn source_dir(
    locked: &Locked,
    catalog: &SourceCatalog,
    paths: &BTreeMap<String, PathBuf>,
) -> Result<PathBuf> {
    if let Some((url, commit)) = locked.git() {
        if let ResolvedSource::Directory(dir) = catalog.resolve_source(locked) {
            return find_in_dir(locked, dir).with_context(|| {
                format!(
                    "{}-{} is not in vendored source {}",
                    locked.name,
                    locked.version,
                    dir.display()
                )
            });
        }
        return git_dir(&locked.name, url, commit)
            .or_else(|err| find_registry_or_vendor(locked, catalog).ok_or(err));
    }
    if !locked.is_registry() {
        return paths
            .get(&locked.name)
            .cloned()
            .with_context(|| format!("no path for local package {}", locked.name));
    }
    let leaf = format!("{}-{}", locked.name, locked.version);
    find_registry_or_vendor(locked, catalog)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_distinct_sources_with_same_package_and_version() {
        let base = std::env::temp_dir().join(format!("haste-src-id-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join(".cargo")).unwrap();
        std::fs::create_dir_all(base.join("vendor-pub/dup-1.0.0")).unwrap();
        std::fs::create_dir_all(base.join("vendor-corp/dup-1.0.0")).unwrap();
        std::fs::write(
            base.join(".cargo/config.toml"),
            r#"
[net]
offline = true

[source.crates-io]
replace-with = "vendor-pub"

[source.vendor-pub]
directory = "vendor-pub"

[source.corp]
registry = "sparse+https://corp.internal/index/"
replace-with = "vendor-corp"

[source.vendor-corp]
directory = "vendor-corp"
"#,
        )
        .unwrap();
        let catalog = SourceCatalog::load(&base).unwrap();
        assert!(catalog.offline);
        let pub_locked = Locked {
            name: "dup".into(),
            version: semver::Version::new(1, 0, 0),
            source: Some("registry+https://github.com/rust-lang/crates.io-index".into()),
            checksum: None,
            dependencies: Vec::new(),
        };
        let corp_locked = Locked {
            name: "dup".into(),
            version: semver::Version::new(1, 0, 0),
            source: Some("sparse+https://corp.internal/index/".into()),
            checksum: None,
            dependencies: Vec::new(),
        };
        let pub_dir = source_dir(&pub_locked, &catalog, &BTreeMap::new()).unwrap();
        let corp_dir = source_dir(&corp_locked, &catalog, &BTreeMap::new()).unwrap();
        assert!(pub_dir.ends_with("vendor-pub/dup-1.0.0"));
        assert!(corp_dir.ends_with("vendor-corp/dup-1.0.0"));
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn checksum_mismatch_rejects_wrong_registry_candidate() {
        let base = std::env::temp_dir().join(format!("haste-chk-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let good = base.join("dup-1.0.0");
        std::fs::create_dir_all(&good).unwrap();
        std::fs::write(
            good.join(".cargo-checksum.json"),
            r#"{"files":{},"package":"aaaa"}"#,
        )
        .unwrap();
        let locked_match = Locked {
            name: "dup".into(),
            version: semver::Version::new(1, 0, 0),
            source: Some("sparse+https://index.crates.io/".into()),
            checksum: Some("aaaa".into()),
            dependencies: Vec::new(),
        };
        let locked_wrong = Locked {
            checksum: Some("bbbb".into()),
            ..locked_match.clone()
        };
        assert!(find_in_dir(&locked_match, &base).is_some());
        assert!(find_in_dir(&locked_wrong, &base).is_none());
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn cargo_ok_fallback_when_checksum_json_missing_and_strict_when_present() {
        let base = std::env::temp_dir().join(format!("haste-chk-ok-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let dir = base.join("dup-1.0.0");
        std::fs::create_dir_all(&dir).expect("create dir");
        assert!(!checksum_matches(&dir, Some("aaaa")));
        std::fs::write(dir.join(".cargo-ok"), r#"{"v":1}"#).expect("write ok");
        assert!(checksum_matches(&dir, Some("aaaa")));
        std::fs::write(dir.join(".cargo-checksum.json"), "{bad").expect("write bad json");
        assert!(!checksum_matches(&dir, Some("aaaa")));
        std::fs::write(
            dir.join(".cargo-checksum.json"),
            r#"{"files":{},"package":"bbbb"}"#,
        )
        .expect("write wrong json");
        assert!(!checksum_matches(&dir, Some("aaaa")));
        std::fs::write(
            dir.join(".cargo-checksum.json"),
            r#"{"files":{},"package":"aaaa"}"#,
        )
        .expect("write good json");
        assert!(checksum_matches(&dir, Some("aaaa")));
        let _ = std::fs::remove_dir_all(&base);
    }
}

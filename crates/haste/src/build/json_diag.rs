use std::path::{Path, PathBuf};

use crate::graph::manifest::{self, Manifest};
use crate::outline::{Outline, Role};

use super::report::push_json_str;

struct JsonTarget {
    package_id: String,
    manifest_path: String,
    kind: &'static str,
    crate_type: &'static str,
    name: String,
    src_path: String,
    edition: String,
    doc: bool,
    doctest: bool,
    test: bool,
}

fn role_meta(role: Role, proc_macro: bool) -> (&'static str, &'static str, bool, bool, bool) {
    match role {
        Role::Lib if proc_macro => ("proc-macro", "proc-macro", true, true, true),
        Role::Lib => ("lib", "lib", true, true, true),
        Role::Bin => ("bin", "bin", true, false, true),
        Role::Test => ("test", "bin", false, false, true),
        Role::Example => ("example", "bin", false, false, false),
        Role::Bench => ("bench", "bin", false, false, false),
        Role::Script => ("custom-build", "bin", false, false, false),
    }
}

fn abs_norm(path: &Path) -> PathBuf {
    let abs = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
    manifest::normalize(&abs)
}

fn same_path(a: &Path, b: &Path) -> bool {
    if a == b {
        return true;
    }
    let na = abs_norm(a);
    let nb = abs_norm(b);
    if na == nb {
        return true;
    }
    matches!(
        (std::fs::canonicalize(&na), std::fs::canonicalize(&nb)),
        (Ok(ca), Ok(cb)) if ca == cb
    )
}

pub(super) fn cargo_package_id(dir: &Path, name: &str, version: &str) -> String {
    let abs = abs_norm(dir);
    let dir_str = abs.to_string_lossy();
    let same_name = abs.file_name().and_then(|n| n.to_str()) == Some(name);
    if same_name {
        format!("path+file://{dir_str}#{version}")
    } else {
        format!("path+file://{dir_str}#{name}@{version}")
    }
}

fn env_var<'a>(vars: &'a [(String, String)], key: &str) -> Option<&'a str> {
    vars.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
}

fn find_outline_crate(outline: &Outline, file: &Path) -> Option<usize> {
    let mut best: Option<(u8, usize)> = None;
    for (idx, root) in outline.roots.iter().enumerate() {
        let is_root = same_path(&root.path, file);
        let in_files = !is_root
            && outline
                .crates
                .get(idx)
                .is_some_and(|c| c.files.iter().any(|f| same_path(&f.source.path, file)));
        if !is_root && !in_files {
            continue;
        }
        // Prefer non-synthetic roots: a Role::Lib root beats its Role::Test clone sharing src/lib.rs.
        let is_lib_clone = root.role == Role::Test
            && outline
                .roots
                .iter()
                .any(|other| other.role == Role::Lib && same_path(&other.path, &root.path));
        let rank = match (is_root, is_lib_clone) {
            (true, false) => 3,
            (false, false) => 2,
            (true, true) => 1,
            (false, true) => 0,
        };
        if best.is_none_or(|(best_rank, _)| rank > best_rank) {
            best = Some((rank, idx));
        }
    }
    best.map(|(_, idx)| idx)
}

fn target_from_outline(outline: &Outline, file: Option<&str>) -> Option<JsonTarget> {
    let file_path = file.map(Path::new)?;
    let idx = find_outline_crate(outline, file_path)?;
    let root = &outline.roots[idx];
    let vars = &outline.crates[idx].env.vars;
    let dir = env_var(vars, "CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .or_else(|| root.path.parent().map(PathBuf::from))?;
    let pkg_name = env_var(vars, "CARGO_PKG_NAME").unwrap_or(&root.name);
    let version = env_var(vars, "CARGO_PKG_VERSION").unwrap_or("0.0.0");
    let manifest_path = env_var(vars, "CARGO_MANIFEST_PATH")
        .map(str::to_owned)
        .unwrap_or_else(|| {
            abs_norm(&dir.join("Cargo.toml"))
                .to_string_lossy()
                .into_owned()
        });
    let is_lib_clone = root.role == Role::Test
        && outline
            .roots
            .iter()
            .any(|other| other.role == Role::Lib && same_path(&other.path, &root.path));
    let role = if is_lib_clone { Role::Lib } else { root.role };
    let (kind, crate_type, doc, doctest, test) = role_meta(role, root.proc_macro);
    Some(JsonTarget {
        package_id: cargo_package_id(&dir, pkg_name, version),
        manifest_path,
        kind,
        crate_type,
        name: root.name.clone(),
        src_path: abs_norm(&root.path).to_string_lossy().into_owned(),
        edition: root.edition.clone(),
        doc,
        doctest,
        test,
    })
}

fn target_from_manifest(manifest: &Manifest, file: Option<&str>) -> JsonTarget {
    let version = manifest
        .version
        .as_ref()
        .map_or_else(|| "0.0.0".to_owned(), ToString::to_string);
    let abs_dir = abs_norm(&manifest.dir);
    let manifest_path = abs_dir.join("Cargo.toml").to_string_lossy().into_owned();
    let package_id = cargo_package_id(&abs_dir, &manifest.name, &version);
    let lib_name = manifest
        .lib_name
        .as_ref()
        .unwrap_or(&manifest.name)
        .replace('-', "_");
    let file_path = file.map(Path::new);
    let matched = file_path.and_then(|fp| {
        if let Some(t) = manifest.tests.iter().find(|t| same_path(&t.path, fp)) {
            return Some((Role::Test, t.name.clone(), t.path.clone()));
        }
        if let Some(t) = manifest.bins.iter().find(|t| same_path(&t.path, fp)) {
            return Some((Role::Bin, t.name.clone(), t.path.clone()));
        }
        if let Some(t) = manifest.examples.iter().find(|t| same_path(&t.path, fp)) {
            return Some((Role::Example, t.name.clone(), t.path.clone()));
        }
        if let Some(t) = manifest.benches.iter().find(|t| same_path(&t.path, fp)) {
            return Some((Role::Bench, t.name.clone(), t.path.clone()));
        }
        if let Some(path) = manifest.build.as_ref().filter(|p| same_path(p, fp)) {
            return Some((Role::Script, "build_script_build".to_owned(), path.clone()));
        }
        if let Some(path) = manifest.lib.as_ref().filter(|p| same_path(p, fp)) {
            return Some((Role::Lib, lib_name.clone(), path.clone()));
        }
        None
    });
    let (role, name, src_path) = matched.unwrap_or_else(|| {
        if let Some(path) = &manifest.lib {
            (Role::Lib, lib_name, path.clone())
        } else if let Some(t) = manifest.bins.first() {
            (Role::Bin, t.name.clone(), t.path.clone())
        } else {
            (
                Role::Lib,
                lib_name,
                file_path.map_or_else(|| abs_dir.join("src/lib.rs"), Path::to_path_buf),
            )
        }
    });
    let (kind, crate_type, doc, doctest, test) = role_meta(role, manifest.proc_macro);
    JsonTarget {
        package_id,
        manifest_path,
        kind,
        crate_type,
        name,
        src_path: abs_norm(&src_path).to_string_lossy().into_owned(),
        edition: manifest.edition.clone(),
        doc,
        doctest,
        test,
    }
}

fn fallback_target(file: Option<&str>) -> JsonTarget {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let abs_dir = abs_norm(&cwd);
    let name = abs_dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_owned();
    let src_path = file.map_or_else(
        || abs_dir.join("src/lib.rs").to_string_lossy().into_owned(),
        |f| abs_norm(Path::new(f)).to_string_lossy().into_owned(),
    );
    JsonTarget {
        package_id: cargo_package_id(&abs_dir, &name, "0.0.0"),
        manifest_path: abs_dir.join("Cargo.toml").to_string_lossy().into_owned(),
        kind: "lib",
        crate_type: "lib",
        name,
        src_path,
        edition: "2021".to_owned(),
        doc: true,
        doctest: true,
        test: true,
    }
}

pub(super) fn push_json_package_header(
    out: &mut String,
    outline: Option<&Outline>,
    file: Option<&str>,
) {
    use std::fmt::Write as _;
    let target = outline
        .and_then(|o| target_from_outline(o, file))
        .or_else(|| {
            file.and_then(|f| manifest::read_for_file(Path::new(f)))
                .map(|m| target_from_manifest(&m, file))
        })
        .unwrap_or_else(|| fallback_target(file));
    for (prefix, val) in [
        (",\"package_id\":", target.package_id.as_str()),
        (",\"manifest_path\":", target.manifest_path.as_str()),
        (",\"target\":{\"kind\":[", target.kind),
        ("],\"crate_types\":[", target.crate_type),
        ("],\"name\":", target.name.as_str()),
        (",\"src_path\":", target.src_path.as_str()),
        (",\"edition\":", target.edition.as_str()),
    ] {
        out.push_str(prefix);
        push_json_str(out, val);
    }
    let _ = write!(
        out,
        ",\"doc\":{},\"doctest\":{},\"test\":{}}}",
        target.doc, target.doctest, target.test
    );
}

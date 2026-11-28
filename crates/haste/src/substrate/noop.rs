//! Workspace no-op build fingerprint for skipping unchanged one-shot `haste build` runs.

use std::ffi::OsString;
use std::hash::{Hash as _, Hasher as _};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use super::Backing;
use super::wire::{
    read_bytes, read_i128, read_path, read_u32, read_u64, read_utf8, write_len_bytes, write_path,
};

use crate::graph::store::{FileStamp, file_stamp, hash_env_val, write_new_bytes};
use rustc_hash::FxHasher;

const NOOP_MAGIC: [u8; 8] = *b"HSTNOP03";
const NOOP_VERSION: u32 = 5;
const HEADER_LEN: usize = 52;
const TARGET_DIRS: [&str; 5] = ["src", "src/bin", "tests", "examples", "benches"];
const ROOT_FILES: [&str; 3] = ["src/lib.rs", "src/main.rs", "build.rs"];

/// Returns whether the CLI no-op fast path is active for this `build` invocation.
pub fn active(stats: Option<&Path>) -> bool {
    super::enabled()
        && stats.is_none()
        && !crate::image::verify_stencil()
        && !crate::env_flag("HASTE_NO_NOOP")
}

fn canonical_root(root: &Path) -> PathBuf {
    std::fs::canonicalize(root).unwrap_or_else(|_| root.to_owned())
}

fn abs_out(out: Option<&Path>) -> Option<PathBuf> {
    out.map(|p| std::path::absolute(p).unwrap_or_else(|_| p.to_owned()))
}

fn stamp_path_for(root: &Path, out: Option<&Path>, cfg: &crate::config::Config) -> PathBuf {
    let mut hasher = FxHasher::default();
    NOOP_MAGIC.hash(&mut hasher);
    root.hash(&mut hasher);
    out.hash(&mut hasher);
    cfg.hash_cache_key(&mut hasher);
    crate::graph::store::home()
        .join("noop")
        .join(format!("{:016x}.stamp", hasher.finish()))
}

fn stamp_path(root: &Path, out: Option<&Path>) -> PathBuf {
    stamp_path_for(root, out, &crate::config::Config::current())
}

pub(crate) fn target_dirs(workspace: &crate::graph::Workspace) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    let mut top = None;
    for &member in &workspace.graph.members {
        if let Some(pkg) = workspace.packages.get(member) {
            if top.is_none() {
                top = crate::graph::workspace_root(&pkg.manifest.dir).ok();
            }
            for sub in TARGET_DIRS {
                dirs.push(pkg.manifest.dir.join(sub));
            }
        }
    }
    if let Some(root) = top {
        dirs.extend(crate::graph::manifest::member_glob_dirs(&root));
    }
    dirs.sort();
    dirs.dedup();
    dirs
}

pub(crate) fn target_root_files(workspace: &crate::graph::Workspace) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for pkg in workspace.packages.iter().filter(|pkg| pkg.local()) {
        for rel in ROOT_FILES {
            files.push(pkg.manifest.dir.join(rel));
        }
        files.extend(pkg.manifest.lib.iter().cloned());
        files.extend(pkg.manifest.build.iter().cloned());
        files.extend(pkg.manifest.bins.iter().map(|t| t.path.clone()));
        files.extend(pkg.manifest.examples.iter().map(|t| t.path.clone()));
        files.extend(pkg.manifest.tests.iter().map(|t| t.path.clone()));
        files.extend(pkg.manifest.benches.iter().map(|t| t.path.clone()));
    }
    files.sort();
    files.dedup();
    files
}

pub(crate) fn hash_target_dir(dir: &Path, hasher: &mut FxHasher) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        (dir, false).hash(hasher);
        return;
    };
    let mut found: Vec<OsString> = Vec::new();
    for entry in entries.flatten() {
        let Ok(ft) = entry.file_type() else {
            continue;
        };
        let name = entry.file_name();
        let is_rs = Path::new(&name).extension().and_then(|ext| ext.to_str()) == Some("rs");
        if is_rs && (ft.is_file() || (ft.is_symlink() && dir.join(&name).is_file())) {
            found.push(name);
        } else if ft.is_dir() || (ft.is_symlink() && dir.join(&name).is_dir()) {
            let sub = dir.join(&name);
            let mut dir_tag = name.clone();
            dir_tag.push("/");
            found.push(dir_tag);
            if sub.join("main.rs").is_file() {
                let mut joined = name.clone();
                joined.push("/main.rs");
                found.push(joined);
            }
            if sub.join("Cargo.toml").is_file() {
                let mut joined = name;
                joined.push("/Cargo.toml");
                found.push(joined);
            }
        }
    }
    found.sort();
    (dir, true, found).hash(hasher);
}

/// Computes a fingerprint of auto-discovered target directories and root files for workspace reload checks.
pub(crate) fn target_discovery_stamp(workspace: &crate::graph::Workspace) -> u64 {
    let mut hasher = FxHasher::default();
    for dir in target_dirs(workspace) {
        hash_target_dir(&dir, &mut hasher);
    }
    for path in target_root_files(workspace) {
        (path.as_path(), path.is_file()).hash(&mut hasher);
    }
    hasher.finish()
}

fn hash_target_dirs(dirs: &[PathBuf]) -> u64 {
    let mut hasher = FxHasher::default();
    dirs.len().hash(&mut hasher);
    for dir in dirs {
        hash_target_dir(dir, &mut hasher);
    }
    hasher.finish()
}

fn compute_meta_hash(root: &Path, out: Option<&Path>) -> Option<u64> {
    let target = crate::host_target();
    let cfg = crate::config::Config::current();
    let mut hasher = FxHasher::default();
    NOOP_MAGIC.hash(&mut hasher);
    NOOP_VERSION.hash(&mut hasher);
    super::target_hash(&target).hash(&mut hasher);
    cfg.hash_cache_key(&mut hasher);
    for name in [
        "CARGO_HOME",
        "RUSTFLAGS",
        "CARGO_ENCODED_RUSTFLAGS",
        "RUSTC",
        "RUSTUP_TOOLCHAIN",
        "RUSTUP_HOME",
        "CARGO",
        "CC",
        "CFLAGS",
    ] {
        crate::process_env_var(name).hash(&mut hasher);
    }
    root.hash(&mut hasher);
    match out {
        Some(out_path) => {
            if !std::fs::metadata(out_path).is_ok_and(|m| m.is_file()) {
                return None;
            }
            true.hash(&mut hasher);
            out_path.hash(&mut hasher);
            file_stamp(out_path)?.hash(&mut hasher);
        }
        None => false.hash(&mut hasher),
    }
    Some(hasher.finish())
}

fn file_stamps_ref(paths: &[&Path]) -> Vec<FileStamp> {
    let workers = crate::jobs().clamp(1, 8);
    if paths.len() < 64 || workers <= 1 {
        return paths.iter().map(|p| file_stamp(p)).collect();
    }
    let chunk = paths.len().div_ceil(workers);
    crate::sync::scope(|scope| {
        let handles: Vec<_> = paths
            .chunks(chunk)
            .map(|slice| {
                scope.spawn(move || slice.iter().map(|p| file_stamp(p)).collect::<Vec<_>>())
            })
            .collect();
        let mut out = Vec::with_capacity(paths.len());
        for handle in handles {
            match handle.join() {
                Ok(part) => out.extend(part),
                Err(payload) => std::panic::resume_unwind(payload),
            }
        }
        out
    })
}

fn stamp_matches_raw(raw_path: &[u8], expected: FileStamp) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt as _;
        let path = Path::new(std::ffi::OsStr::from_bytes(raw_path));
        file_stamp(path) == expected
    }
    #[cfg(not(unix))]
    {
        std::str::from_utf8(raw_path)
            .ok()
            .is_some_and(|s| file_stamp(Path::new(s)) == expected)
    }
}

fn write_path_ref(out: &mut Vec<u8>, path: &Path) -> Option<()> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt as _;
        write_len_bytes(out, path.as_os_str().as_bytes())
    }
    #[cfg(not(unix))]
    {
        write_len_bytes(out, path.to_str()?.as_bytes())
    }
}

fn read_raw_entry<'a>(data: &'a [u8], pos: &mut usize) -> Option<(&'a [u8], FileStamp)> {
    let len = read_u32(read_bytes(data, pos, 4)?, 0) as usize;
    let raw_path = read_bytes(data, pos, len)?;
    let present = *read_bytes(data, pos, 1)?.first()?;
    let stamp = match present {
        0 => None,
        1 => {
            let raw = read_bytes(data, pos, 24)?;
            Some((read_u64(raw, 0), read_i128(raw, 8)))
        }
        _ => return None,
    };
    Some((raw_path, stamp))
}

fn encode_stamp_file(
    meta_hash: u64,
    reached: usize,
    dirs_hash: u64,
    priority_count: usize,
    entries: &[(&Path, FileStamp)],
    dirs: &[PathBuf],
    envs: &[(String, Option<u64>)],
) -> Option<Vec<u8>> {
    let p_count = u32::try_from(priority_count).ok()?;
    let total_count = u32::try_from(entries.len()).ok()?;
    let dir_count = u32::try_from(dirs.len()).ok()?;
    let env_count = u32::try_from(envs.len()).ok()?;
    let cap = HEADER_LEN + entries.len() * 72 + dirs.len() * 48 + envs.len() * 28;
    let mut out = Vec::with_capacity(cap);
    out.extend_from_slice(&NOOP_MAGIC);
    out.extend_from_slice(&NOOP_VERSION.to_le_bytes());
    out.extend_from_slice(&meta_hash.to_le_bytes());
    out.extend_from_slice(&(reached as u64).to_le_bytes());
    out.extend_from_slice(&dirs_hash.to_le_bytes());
    out.extend_from_slice(&p_count.to_le_bytes());
    out.extend_from_slice(&total_count.to_le_bytes());
    out.extend_from_slice(&dir_count.to_le_bytes());
    out.extend_from_slice(&env_count.to_le_bytes());
    for (name, val) in envs {
        write_len_bytes(&mut out, name.as_bytes())?;
        match val {
            Some(hash) => {
                out.push(1);
                out.extend_from_slice(&hash.to_le_bytes());
            }
            None => out.push(0),
        }
    }
    for &(path, stamp) in entries {
        write_path_ref(&mut out, path)?;
        match stamp {
            Some((len, nanos)) => {
                out.push(1);
                out.extend_from_slice(&len.to_le_bytes());
                out.extend_from_slice(&nanos.to_le_bytes());
            }
            None => out.push(0),
        }
    }
    for dir in dirs {
        write_path(&mut out, dir)?;
    }
    Some(out)
}

fn check_rest_entries(entries: &[(&[u8], FileStamp)]) -> bool {
    let workers = crate::jobs().clamp(1, 8);
    if entries.len() < 64 || workers <= 1 {
        return entries
            .iter()
            .all(|&(raw, stamp)| stamp_matches_raw(raw, stamp));
    }
    let failed = AtomicBool::new(false);
    let chunk = entries.len().div_ceil(workers);
    crate::sync::scope(|scope| {
        let handles: Vec<_> = entries
            .chunks(chunk)
            .map(|slice| {
                let stop = &failed;
                scope.spawn(move || {
                    for &(raw, stamp) in slice {
                        if stop.load(Ordering::Relaxed) {
                            return;
                        }
                        if !stamp_matches_raw(raw, stamp) {
                            stop.store(true, Ordering::Relaxed);
                            return;
                        }
                    }
                })
            })
            .collect();
        for handle in handles {
            if let Err(payload) = handle.join() {
                std::panic::resume_unwind(payload);
            }
        }
    });
    !failed.load(Ordering::Relaxed)
}

fn try_hit_at(path: &Path, croot: &Path, aout: Option<&Path>) -> Option<usize> {
    let meta_hash = compute_meta_hash(croot, aout)?;
    let backing = Backing::open(path, HEADER_LEN).ok()?;
    let data = backing.as_slice();
    if data.len() < HEADER_LEN
        || data[0..8] != NOOP_MAGIC
        || read_u32(data, 8) != NOOP_VERSION
        || read_u64(data, 12) != meta_hash
    {
        return None;
    }
    let reached = usize::try_from(read_u64(data, 20)).ok()?;
    if reached == 0 {
        return None;
    }
    let dirs_hash = read_u64(data, 28);
    let priority_count = read_u32(data, 36) as usize;
    let total_count = read_u32(data, 40) as usize;
    let dir_count = read_u32(data, 44) as usize;
    let env_count = read_u32(data, 48) as usize;
    if priority_count > total_count {
        return None;
    }
    let mut pos = 52;
    for _ in 0..env_count {
        let name = read_utf8(data, &mut pos)?;
        let present = *read_bytes(data, &mut pos, 1)?.first()?;
        let expected = match present {
            0 => None,
            1 => Some(read_u64(read_bytes(data, &mut pos, 8)?, 0)),
            _ => return None,
        };
        let actual = crate::process_env_var(&name)
            .as_deref()
            .map(|v| hash_env_val(&NOOP_MAGIC, &name, v));
        if actual != expected {
            return None;
        }
    }
    for _ in 0..priority_count {
        let (raw, stamp) = read_raw_entry(data, &mut pos)?;
        if !stamp_matches_raw(raw, stamp) {
            return None;
        }
    }
    let rest_count = total_count - priority_count;
    let mut rest = Vec::with_capacity(rest_count.min(4096));
    for _ in 0..rest_count {
        rest.push(read_raw_entry(data, &mut pos)?);
    }
    if !check_rest_entries(&rest) {
        return None;
    }
    let mut dirs = Vec::with_capacity(dir_count.min(256));
    for _ in 0..dir_count {
        dirs.push(read_path(data, &mut pos)?);
    }
    (pos == data.len() && hash_target_dirs(&dirs) == dirs_hash).then_some(reached)
}

/// Returns the cached `reached` count when `root` and all watched inputs match the previous `0 failed` build.
pub fn try_hit(root: &Path, out: Option<&Path>) -> Option<usize> {
    let croot = canonical_root(root);
    let aout = abs_out(out);
    try_hit_at(
        &stamp_path(&croot, aout.as_deref()),
        &croot,
        aout.as_deref(),
    )
}

/// Records the workspace no-op stamp after a `0 failed` build succeeds.
pub fn record(
    root: &Path,
    workspace: &crate::graph::Workspace,
    build_paths: &[PathBuf],
    env_reads: &[String],
    out: Option<&Path>,
    reached: usize,
) {
    if reached == 0 {
        return;
    }
    let croot = canonical_root(root);
    let aout = abs_out(out);
    let Some(meta_hash) = compute_meta_hash(&croot, aout.as_deref()) else {
        return;
    };
    let mut manifests = crate::graph::manifest_paths(workspace, &croot);
    if let Some(settings) = crate::graph::snapshot::rustup_settings_path() {
        manifests.push(settings);
    }
    let roots = target_root_files(workspace);
    let (watched_files, watched_vars) = crate::graph::store::watched_stamps(workspace);

    let mut static_lib_paths = Vec::new();
    for entry in &workspace.graph.units {
        if let Some(out) = &entry.script {
            let pkg_dir = &workspace.packages[entry.package].manifest.dir;
            crate::graph::snapshot::collect_static_lib_paths(pkg_dir, out, &mut static_lib_paths);
        }
    }

    let mut priority: Vec<&Path> = manifests
        .iter()
        .chain(&roots)
        .map(PathBuf::as_path)
        .collect();
    priority.sort_unstable();
    priority.dedup();
    let priority_count = priority.len();

    let mut rest: Vec<&Path> = build_paths
        .iter()
        .map(PathBuf::as_path)
        .chain(watched_files.iter().map(|(p, _, _)| p.as_path()))
        .chain(static_lib_paths.iter().map(PathBuf::as_path))
        .filter(|p| priority.binary_search(p).is_err())
        .collect();
    rest.sort_unstable();
    rest.dedup();

    let mut all_refs = Vec::with_capacity(priority.len() + rest.len());
    all_refs.extend(priority);
    all_refs.extend(rest);
    let stamps = file_stamps_ref(&all_refs);
    let entries: Vec<(&Path, FileStamp)> = all_refs.into_iter().zip(stamps).collect();

    let dirs = target_dirs(workspace);
    let dirs_hash = hash_target_dirs(&dirs);
    let mut env_names: Vec<String> = watched_vars.into_iter().map(|(name, _)| name).collect();
    env_names.extend(env_reads.iter().cloned());
    env_names.sort();
    env_names.dedup();
    let envs: Vec<(String, Option<u64>)> = env_names
        .into_iter()
        .map(|name| {
            let hash = crate::process_env_var(&name)
                .as_deref()
                .map(|v| hash_env_val(&NOOP_MAGIC, &name, v));
            (name, hash)
        })
        .collect();

    if let Some(encoded) = encode_stamp_file(
        meta_hash,
        reached,
        dirs_hash,
        priority_count,
        &entries,
        &dirs,
        &envs,
    ) {
        drop(write_new_bytes(
            &stamp_path(&croot, aout.as_deref()),
            &encoded,
        ));
    }
}

/// Removes any saved workspace no-op stamp after a failed build.
pub fn invalidate(root: &Path, out: Option<&Path>) {
    let croot = canonical_root(root);
    let aout = abs_out(out);
    drop(std::fs::remove_file(stamp_path(&croot, aout.as_deref())));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stamp_paths_separate_features_targets_and_env_flags() {
        let host = haste_codegen::Target::host();
        let cross = haste_codegen::TARGETS
            .iter()
            .find(|t| t.triple != host.triple)
            .expect("non-host target exists");
        let root = Path::new("/tmp/sample-ws");
        let base = crate::config::Config::for_target(host);
        let p_base = stamp_path_for(root, None, &base);
        let p_feat = stamp_path_for(
            root,
            None,
            &crate::config::Config {
                selection: crate::Selection {
                    features: vec!["extra".to_owned()],
                    ..crate::Selection::default()
                },
                ..base.clone()
            },
        );
        let p_target = stamp_path_for(root, None, &crate::config::Config::for_target(cross));
        let p_debug = stamp_path_for(
            root,
            None,
            &crate::config::Config {
                debug_info: !base.debug_info,
                ..base
            },
        );
        assert_ne!(p_base, p_feat);
        assert_ne!(p_base, p_target);
        assert_ne!(p_base, p_debug);
        assert_ne!(p_feat, p_target);
    }

    #[test]
    fn meta_hash_distinguishes_rustflags_changes() {
        let stamp_with = |pairs: &[(&str, &str)]| {
            let map: rustc_hash::FxHashMap<String, String> = pairs
                .iter()
                .map(|&(k, v)| (k.to_owned(), v.to_owned()))
                .collect();
            crate::with_client_env(Some(std::sync::Arc::new(map)), || {
                compute_meta_hash(Path::new("/tmp/sample-ws"), None)
            })
            .expect("compute_meta_hash returns Some when out is None")
        };
        let s_unset = stamp_with(&[]);
        let s_empty = stamp_with(&[("RUSTFLAGS", "")]);
        let s_flag = stamp_with(&[("RUSTFLAGS", "--cfg my_flag")]);
        let s_encoded = stamp_with(&[("CARGO_ENCODED_RUSTFLAGS", "--cfg\x1fmy_flag")]);
        assert_ne!(s_unset, s_empty);
        assert_ne!(s_unset, s_flag);
        assert_ne!(s_empty, s_flag);
        assert_ne!(s_unset, s_encoded);
        assert_ne!(s_flag, s_encoded);
    }

    #[test]
    fn noop_round_trips_hides_secrets_and_fails_fast_on_root_or_leaf_edits() {
        let tmp = std::env::temp_dir().join(format!("haste-noop-t-{}", std::process::id()));
        drop(std::fs::remove_dir_all(&tmp));
        std::fs::create_dir_all(&tmp).expect("mkdir");
        let root_rs = tmp.join("main.rs");
        let leaf_rs = tmp.join("leaf.rs");
        std::fs::write(&root_rs, "fn main() {}\n").expect("write root");
        std::fs::write(&leaf_rs, "pub fn x() {}\n").expect("write leaf");
        let stamp_file = tmp.join("test.stamp");
        let secret = "noop_secret_value_123";
        let env_map = std::sync::Arc::new(rustc_hash::FxHashMap::from_iter([(
            "MY_NOOP_SECRET".to_owned(),
            secret.to_owned(),
        )]));

        crate::with_client_env(Some(env_map.clone()), || {
            let meta_hash = compute_meta_hash(&tmp, None).expect("meta_hash");
            let entries = vec![
                (root_rs.as_path(), file_stamp(&root_rs)),
                (leaf_rs.as_path(), file_stamp(&leaf_rs)),
            ];
            let envs = vec![(
                "MY_NOOP_SECRET".to_owned(),
                Some(hash_env_val(&NOOP_MAGIC, "MY_NOOP_SECRET", secret)),
            )];
            let encoded = encode_stamp_file(
                meta_hash,
                42,
                hash_target_dirs(&[]),
                1,
                &entries,
                &[],
                &envs,
            )
            .expect("encode");
            assert!(
                !encoded
                    .windows(secret.len())
                    .any(|w| w == secret.as_bytes())
            );
            write_new_bytes(&stamp_file, &encoded).expect("write");
            assert_eq!(try_hit_at(&stamp_file, &tmp, None), Some(42));
        });

        crate::with_client_env(
            Some(std::sync::Arc::new(rustc_hash::FxHashMap::default())),
            || {
                assert_eq!(try_hit_at(&stamp_file, &tmp, None), None);
            },
        );
        crate::with_client_env(Some(env_map.clone()), || {
            std::fs::write(&leaf_rs, "pub fn x() { let _ = 1; }\n").expect("edit leaf");
            assert_eq!(try_hit_at(&stamp_file, &tmp, None), None);
            std::fs::write(&leaf_rs, "pub fn x() {}\n").expect("restore leaf");
            std::fs::write(&root_rs, "fn main() { let _ = 1; }\n").expect("edit root");
            assert_eq!(try_hit_at(&stamp_file, &tmp, None), None);
        });
        drop(std::fs::remove_dir_all(&tmp));
    }

    #[test]
    fn noop_tracks_toolchain_envs_rustup_settings_and_static_libs() {
        let base_meta = |vars: &[(&str, &str)]| {
            let map: rustc_hash::FxHashMap<String, String> = vars
                .iter()
                .map(|&(k, v)| (k.to_owned(), v.to_owned()))
                .collect();
            crate::with_client_env(Some(std::sync::Arc::new(map)), || {
                compute_meta_hash(Path::new("/tmp/sample-ws"), None)
            })
            .expect("meta_hash")
        };
        let baseline = base_meta(&[]);
        for var in [
            "RUSTC",
            "RUSTUP_TOOLCHAIN",
            "RUSTUP_HOME",
            "CARGO",
            "CC",
            "CFLAGS",
        ] {
            assert_ne!(
                baseline,
                base_meta(&[(var, "custom_val")]),
                "{var} must affect noop meta_hash"
            );
        }

        let _guard = crate::sync::lock(&crate::substrate::TEST_HOME_LOCK);
        let prev_home = std::env::var_os("HASTE_HOME");
        let seq = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let tmp = std::env::temp_dir().join(format!("haste-noop-s-{}-{seq}", std::process::id()));
        let home = tmp.join("haste_home");
        let rustup = tmp.join("rustup");
        let ws_dir = tmp.join("ws");
        std::fs::create_dir_all(&home).expect("mkdir home");
        std::fs::create_dir_all(&rustup).expect("mkdir rustup");
        std::fs::create_dir_all(ws_dir.join("src")).expect("mkdir src");
        std::fs::create_dir_all(ws_dir.join("native")).expect("mkdir native");
        std::fs::write(
            rustup.join("settings.toml"),
            "default_toolchain = \"stable\"\n",
        )
        .expect("write settings");
        std::fs::write(ws_dir.join("native/libdemo.a"), b"arch1").expect("write libdemo.a");
        std::fs::write(
            ws_dir.join("Cargo.toml"),
            "[package]\nname = \"demo\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        )
        .expect("write Cargo.toml");
        std::fs::write(
            ws_dir.join("Cargo.lock"),
            "version = 4\n[[package]]\nname = \"demo\"\nversion = \"0.1.0\"\n",
        )
        .expect("write Cargo.lock");
        std::fs::write(ws_dir.join("src/main.rs"), "fn main() {}\n").expect("write main.rs");

        // SAFETY: serialized by `TEST_HOME_LOCK` within unit tests.
        unsafe {
            std::env::set_var("HASTE_HOME", &home);
        }
        let host = crate::graph::cfg::Target::of(&crate::host_target());
        let targets = crate::graph::Targets {
            target: host.clone(),
            host,
        };
        let mut ws = crate::graph::load(&ws_dir, &targets, &Default::default()).expect("load ws");
        ws.graph.units[0].script = Some(std::sync::Arc::new(crate::graph::script::Output::parse(
            "cargo:rustc-link-search=native=native\ncargo:rustc-link-lib=static=demo\n",
        )));
        let env_map = std::sync::Arc::new(rustc_hash::FxHashMap::from_iter([(
            "RUSTUP_HOME".to_owned(),
            rustup.to_string_lossy().into_owned(),
        )]));
        crate::with_client_env(Some(env_map), || {
            record(&ws_dir, &ws, &[], &[], None, 1);
            assert_eq!(try_hit(&ws_dir, None), Some(1));

            std::fs::write(ws_dir.join("native/libdemo.a"), b"arch2_modified")
                .expect("edit libdemo.a");
            assert_eq!(try_hit(&ws_dir, None), None);

            record(&ws_dir, &ws, &[], &[], None, 1);
            assert_eq!(try_hit(&ws_dir, None), Some(1));

            std::fs::write(
                rustup.join("settings.toml"),
                "default_toolchain = \"nightly\"\n",
            )
            .expect("edit settings.toml");
            assert_eq!(try_hit(&ws_dir, None), None);
        });
        // SAFETY: serialized by `TEST_HOME_LOCK` within unit tests.
        unsafe {
            match prev_home {
                Some(v) => std::env::set_var("HASTE_HOME", v),
                None => std::env::remove_var("HASTE_HOME"),
            }
        }
        drop(std::fs::remove_dir_all(&tmp));
    }
}

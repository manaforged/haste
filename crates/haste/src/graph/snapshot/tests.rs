use super::*;
use crate::graph::cfg::Target;
use rustc_hash::FxHashMap as HashMap;
use std::sync::Arc;

fn temp_dir(tag: &str) -> PathBuf {
    let seq = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let dir = std::env::temp_dir().join(format!("haste-ws-{tag}-{}-{seq}", std::process::id()));
    drop(std::fs::remove_dir_all(&dir));
    std::fs::create_dir_all(&dir).expect("mkdir");
    dir.canonicalize().unwrap_or(dir)
}

fn with_graph_env<R>(home: &Path, no_graph: bool, f: impl FnOnce() -> R) -> R {
    let _guard = crate::sync::lock(&crate::substrate::TEST_HOME_LOCK);
    let prev_home = std::env::var_os("HASTE_HOME");
    let prev_no = std::env::var_os("HASTE_NO_GRAPH_CACHE");
    // SAFETY: serialized by `TEST_HOME_LOCK` within unit tests.
    unsafe {
        std::env::set_var("HASTE_HOME", home);
        if no_graph {
            std::env::set_var("HASTE_NO_GRAPH_CACHE", "1");
        } else {
            std::env::remove_var("HASTE_NO_GRAPH_CACHE");
        }
    }
    let out = f();
    // SAFETY: serialized by `TEST_HOME_LOCK` within unit tests.
    unsafe {
        match prev_home {
            Some(v) => std::env::set_var("HASTE_HOME", v),
            None => std::env::remove_var("HASTE_HOME"),
        }
        match prev_no {
            Some(v) => std::env::set_var("HASTE_NO_GRAPH_CACHE", v),
            None => std::env::remove_var("HASTE_NO_GRAPH_CACHE"),
        }
    }
    out
}

fn write_files(root: &Path, files: &[(&str, &str)]) {
    for &(rel, body) in files {
        let path = root.join(rel);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("mkdir");
        }
        std::fs::write(&path, body).expect("write");
    }
}

fn test_targets() -> Targets {
    let host = Target::of(&crate::host_target());
    Targets {
        target: host.clone(),
        host,
    }
}

fn sample_workspace(root: &Path, targets: &Targets) -> Workspace {
    let mut ws = super::super::load(root, targets, &Default::default()).expect("load ws");
    let mut out = Output::default();
    out.names.insert("my_cfg".to_owned());
    out.pairs.insert(("feat".to_owned(), "fast".to_owned()));
    out.vars.push(("GENERATED".to_owned(), "yes".to_owned()));
    out.libs.push("static=demo_native".to_owned());
    out.search.push("native=native_dir".to_owned());
    out.metadata.push(("key".to_owned(), "val".to_owned()));
    out.watched.push("watched_data.txt".to_owned());
    out.watched_vars.push("SECRET_SCRIPT_ENV".to_owned());
    ws.graph.units[0].script = Some(Arc::new(out));
    ws
}

#[test]
fn snapshot_round_trips_and_rejects_secret_leak_or_key_mismatch() {
    let root = temp_dir("rt-ws");
    let home = temp_dir("rt-home");
    write_files(
        &root,
        &[
            (
                "Cargo.toml",
                "[package]\nname = \"demo\"\nversion = \"0.2.3\"\nedition = \"2021\"\n",
            ),
            (
                "Cargo.lock",
                "version = 4\n[[package]]\nname = \"demo\"\nversion = \"0.2.3\"\n",
            ),
            ("src/lib.rs", "pub fn f() -> usize { 1 }\n"),
            ("watched_data.txt", "data_v1"),
            ("native_dir/libdemo_native.a", "archive_v1"),
        ],
    );
    let targets = test_targets();
    let ws = sample_workspace(&root, &targets);
    let secret = "super_secret_env_value_999";
    let mut env_map = HashMap::default();
    env_map.insert("SECRET_SCRIPT_ENV".to_owned(), secret.to_owned());
    let env_arc = Arc::new(env_map);

    with_graph_env(&home, false, || {
        let lib_dir = root.join("fake-lib");
        crate::with_client_env(Some(env_arc.clone()), || {
            let key = snapshot_key(&root, &targets, &Config::current());
            save_snapshot(key, &ws, &root, Some(&lib_dir));
            let path = cache_path(key);
            let raw = std::fs::read(&path).expect("read .ws");
            assert!(
                !raw.windows(secret.len()).any(|w| w == secret.as_bytes()),
                "plaintext env value leaked into .ws snapshot"
            );
            assert!(matches!(
                decode_snapshot(&path, key ^ 1).expect("decode"),
                SnapshotState::Miss
            ));
            let SnapshotState::Hit(loaded) = decode_snapshot(&path, key).expect("decode") else {
                panic!("expected snapshot hit");
            };
            assert_eq!(loaded.unit_lines(), ws.unit_lines());
            assert_eq!(loaded.packages.len(), ws.packages.len());
            assert_eq!(loaded.graph.units[0].script, ws.graph.units[0].script);
        });

        let other_env = Arc::new(HashMap::default());
        crate::with_client_env(Some(other_env), || {
            let key = snapshot_key(&root, &targets, &Config::current());
            assert!(matches!(
                decode_snapshot(&cache_path(key), key).expect("decode"),
                SnapshotState::ScriptsStale(_, Some(ref p)) if p == &lib_dir
            ));
        });
    });

    with_graph_env(&home, true, || {
        assert!(!enabled());
    });
    drop(std::fs::remove_dir_all(&root));
    drop(std::fs::remove_dir_all(&home));
}

#[test]
fn snapshot_invalidates_on_manifest_lockfile_target_and_watched_changes() {
    let root = temp_dir("inv-ws");
    let home = temp_dir("inv-home");
    write_files(
        &root,
        &[
            (
                "Cargo.toml",
                "[package]\nname = \"demo\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
            ),
            (
                "Cargo.lock",
                "version = 4\n[[package]]\nname = \"demo\"\nversion = \"0.1.0\"\n",
            ),
            ("src/lib.rs", "pub fn f() -> usize { 1 }\n"),
            ("watched_data.txt", "data_v1"),
            ("native_dir/libdemo_native.a", "archive_v1"),
        ],
    );
    let targets = test_targets();
    let ws = sample_workspace(&root, &targets);

    with_graph_env(&home, false, || {
        let key = snapshot_key(&root, &targets, &Config::current());
        let path = cache_path(key);
        let save_and_check = || {
            save_snapshot(key, &ws, &root, None);
            assert!(matches!(
                decode_snapshot(&path, key).expect("decode"),
                SnapshotState::Hit(_)
            ));
        };

        save_and_check();
        std::fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"demo\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
        )
        .expect("edit manifest");
        assert!(matches!(
            decode_snapshot(&path, key).expect("decode"),
            SnapshotState::Miss
        ));

        save_and_check();
        std::fs::write(
            root.join("Cargo.lock"),
            "version = 4\n[[package]]\nname = \"demo\"\nversion = \"0.1.1\"\n",
        )
        .expect("edit lock");
        assert!(matches!(
            decode_snapshot(&path, key).expect("decode"),
            SnapshotState::Miss
        ));

        save_and_check();
        std::fs::write(root.join("watched_data.txt"), "data_v2").expect("edit watched");
        assert!(matches!(
            decode_snapshot(&path, key).expect("decode"),
            SnapshotState::ScriptsStale(_, None)
        ));

        save_and_check();
        std::fs::write(root.join("native_dir/libdemo_native.a"), "archive_v2").expect("edit lib");
        assert!(matches!(
            decode_snapshot(&path, key).expect("decode"),
            SnapshotState::ScriptsStale(_, None)
        ));

        save_and_check();
        std::fs::write(root.join("src/lib.rs"), "pub fn f() -> usize { 42 }\n").expect("edit lib");
        assert!(matches!(
            decode_snapshot(&path, key).expect("decode"),
            SnapshotState::Hit(_)
        ));

        save_and_check();
        std::fs::create_dir_all(root.join("src/bin")).expect("mkdir bin");
        std::fs::write(root.join("src/bin/extra.rs"), "fn main() {}\n").expect("add bin");
        assert!(matches!(
            decode_snapshot(&path, key).expect("decode"),
            SnapshotState::Miss
        ));

        save_and_check();
        std::fs::remove_file(root.join("src/lib.rs")).expect("rm lib");
        assert!(matches!(
            decode_snapshot(&path, key).expect("decode"),
            SnapshotState::Miss
        ));
    });
    drop(std::fs::remove_dir_all(&root));
    drop(std::fs::remove_dir_all(&home));
}

#[test]
fn two_tier_preserves_structure_when_build_script_watches_source_file() {
    let root = temp_dir("tier-ws");
    let home = temp_dir("tier-home");
    write_files(
        &root,
        &[
            (
                "Cargo.toml",
                "[package]\nname = \"demo\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
            ),
            (
                "Cargo.lock",
                "version = 4\n[[package]]\nname = \"demo\"\nversion = \"0.1.0\"\n",
            ),
            (
                "build.rs",
                "fn main() { println!(\"cargo:rerun-if-changed=src/main.rs\"); }\n",
            ),
            ("src/main.rs", "fn main() {}\n"),
        ],
    );
    let targets = test_targets();
    let mut ws = super::super::load(&root, &targets, &Default::default()).expect("load ws");
    let mut out = Output::default();
    out.watched.push("src/main.rs".to_owned());
    ws.graph.units[0].script = Some(Arc::new(out));

    with_graph_env(&home, false, || {
        let key = snapshot_key(&root, &targets, &Config::current());
        let path = cache_path(key);
        let lib_dir = root.join("sysroot-lib");

        save_snapshot(key, &ws, &root, Some(&lib_dir));
        assert!(matches!(
            decode_snapshot(&path, key).expect("decode"),
            SnapshotState::Hit(_)
        ));

        std::fs::write(root.join("src/main.rs"), "fn main() { let _ = 1; }\n").expect("edit main");
        assert!(matches!(
            decode_snapshot(&path, key).expect("decode"),
            SnapshotState::ScriptsStale(_, Some(ref p)) if p == &lib_dir
        ));

        save_snapshot(key, &ws, &root, Some(&lib_dir));
        std::fs::write(root.join("build.rs"), "fn main() { let _ = 2; }\n").expect("edit build.rs");
        assert!(matches!(
            decode_snapshot(&path, key).expect("decode"),
            SnapshotState::ScriptsStale(_, Some(ref p)) if p == &lib_dir
        ));

        save_snapshot(key, &ws, &root, Some(&lib_dir));
        std::fs::remove_file(root.join("build.rs")).expect("rm build.rs");
        assert!(matches!(
            decode_snapshot(&path, key).expect("decode"),
            SnapshotState::Miss
        ));
    });
    drop(std::fs::remove_dir_all(&root));
    drop(std::fs::remove_dir_all(&home));
}

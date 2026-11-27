use std::path::{Path, PathBuf};

use super::*;
use crate::graph::cfg::Target;

fn temp_dir(tag: &str) -> PathBuf {
    let seq = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let dir =
        std::env::temp_dir().join(format!("haste-scripts-{tag}-{}-{seq}", std::process::id()));
    drop(std::fs::remove_dir_all(&dir));
    std::fs::create_dir_all(&dir).expect("mkdir");
    dir.canonicalize().unwrap_or(dir)
}

fn with_scripts_env<R>(home: &Path, f: impl FnOnce() -> R) -> R {
    let _guard = crate::sync::lock(&crate::substrate::TEST_HOME_LOCK);
    let prev_home = std::env::var_os("HASTE_HOME");
    // SAFETY: serialized by `TEST_HOME_LOCK` within unit tests.
    unsafe {
        std::env::set_var("HASTE_HOME", home);
    }
    let out = f();
    // SAFETY: serialized by `TEST_HOME_LOCK` within unit tests.
    unsafe {
        match prev_home {
            Some(v) => std::env::set_var("HASTE_HOME", v),
            None => std::env::remove_var("HASTE_HOME"),
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

fn install_mock_script(path: &Path, body: &str) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("mkdir script dir");
    }
    std::fs::write(path, body).expect("write mock script");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        let perms = std::fs::Permissions::from_mode(0o755);
        std::fs::set_permissions(path, perms).expect("chmod mock script");
    }
}

#[test]
fn fast_kept_reuses_executable_across_source_edits_and_invalidates_on_build_rs_edit() {
    let root = temp_dir("reuse");
    let home = temp_dir("reuse-home");
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
                "fn main() { println!(\"cargo:rustc-cfg=fast_hit\"); }\n",
            ),
            ("src/main.rs", "fn main() {}\n"),
        ],
    );
    let targets = test_targets();
    let ws = crate::graph::load(&root, &targets, &Default::default()).expect("load ws");

    with_scripts_env(&home, || {
        let dir = store::dir(&ws, 0);
        let build_path = ws.packages[ws.graph.units[0].package]
            .manifest
            .build
            .clone()
            .expect("build.rs path");
        let meta = binary::fast_meta_hash(&ws, &targets, 0);
        let exe_path = dir.join("script");

        assert!(binary::fast_kept(&ws, &targets, 0, &dir).is_none());

        install_mock_script(
            &exe_path,
            "#!/bin/sh\nprintf 'cargo:rustc-cfg=fast_hit\\n'\n",
        );
        binary::write_fast_manifest(&dir, meta, std::slice::from_ref(&build_path), &[])
            .expect("write script.fast");

        assert_eq!(
            binary::fast_kept(&ws, &targets, 0, &dir),
            Some(exe_path.clone())
        );

        std::fs::write(root.join("src/main.rs"), "fn main() { let _ = 42; }\n").expect("edit main");
        assert_eq!(
            binary::fast_kept(&ws, &targets, 0, &dir),
            Some(exe_path.clone()),
            "editing src/main.rs must preserve script.fast executable hit"
        );

        #[cfg(unix)]
        {
            let dag = ScriptDag::new(&ws);
            let mut locks = HashMap::default();
            locks.insert(0, store::lock(&ws, 0).expect("lock"));
            let mut ran = HashMap::default();
            let mut exes = HashMap::default();
            exes.insert(0, exe_path);
            let outcome =
                fast::run_executables(&ws, &targets, &dag, &[0], &exes, &mut locks, &mut ran)
                    .expect("run fast executables");
            assert_eq!(outcome.len(), 1);
            let out = outcome[0].1.as_ref().expect("script output");
            assert!(out.names.contains("fast_hit"));
            assert!(locks.is_empty());
        }

        std::fs::write(
            &build_path,
            "fn main() { println!(\"cargo:rustc-cfg=fast_edited\"); }\n",
        )
        .expect("edit build.rs");
        assert!(
            binary::fast_kept(&ws, &targets, 0, &dir).is_none(),
            "editing build.rs must invalidate script.fast"
        );
    });

    drop(std::fs::remove_dir_all(&root));
    drop(std::fs::remove_dir_all(&home));
}

#[test]
fn fast_manifest_isolates_profiles_rustflags_and_distrust_clears_fast_key() {
    let root = temp_dir("iso");
    let home = temp_dir("iso-home");
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
            ("build.rs", "fn main() {}\n"),
            ("src/main.rs", "fn main() {}\n"),
        ],
    );
    let targets = test_targets();
    let ws = crate::graph::load(&root, &targets, &Default::default()).expect("load ws");

    with_scripts_env(&home, || {
        let dir = store::dir(&ws, 0);
        let build_path = ws.packages[ws.graph.units[0].package]
            .manifest
            .build
            .clone()
            .expect("build.rs path");
        let meta = binary::fast_meta_hash(&ws, &targets, 0);
        install_mock_script(&dir.join("script"), "#!/bin/sh\nexit 0\n");
        binary::write_fast_manifest(&dir, meta, &[build_path], &[]).expect("write script.fast");
        assert!(binary::fast_kept(&ws, &targets, 0, &dir).is_some());

        let mut ws_flags =
            crate::graph::load(&root, &targets, &Default::default()).expect("load ws");
        ws_flags.rustflags.push("--cfg=extra".to_owned());
        assert!(binary::fast_kept(&ws_flags, &targets, 0, &dir).is_none());

        let mut rel_cfg = (*crate::config()).clone();
        rel_cfg.profile = crate::outline::tree::Profile::Release;
        crate::with_config(Arc::new(rel_cfg), || {
            assert!(binary::fast_kept(&ws, &targets, 0, &dir).is_none());
        });

        assert!(binary::fast_kept(&ws, &targets, 0, &dir).is_some());
        binary::distrust(&ws, 0);
        assert!(!dir.join("script").exists());
        assert!(!dir.join("script.fast").exists());
        assert!(binary::fast_kept(&ws, &targets, 0, &dir).is_none());
    });

    drop(std::fs::remove_dir_all(&root));
    drop(std::fs::remove_dir_all(&home));
}

#[cfg(unix)]
#[test]
fn runtime_script_exit_fails_fast_run_and_records_fatal_error() {
    let root = temp_dir("runtime-fail");
    let home = temp_dir("runtime-fail-home");
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
            ("build.rs", "fn main() { std::process::exit(1); }\n"),
            ("src/main.rs", "fn main() {}\n"),
        ],
    );
    let targets = test_targets();
    let mut ws = crate::graph::load(&root, &targets, &Default::default()).expect("load ws");

    with_scripts_env(&home, || {
        let dir = store::dir(&ws, 0);
        let exe_path = dir.join("script");
        let meta = binary::fast_meta_hash(&ws, &targets, 0);
        install_mock_script(&exe_path, "#!/bin/sh\nexit 1\n");
        binary::write_fast_manifest(&dir, meta, &[], &[]).expect("write script.fast");

        let dag = ScriptDag::new(&ws);
        let mut locks = HashMap::default();
        locks.insert(0, store::lock(&ws, 0).expect("lock"));
        let mut ran = HashMap::default();
        let mut exes = HashMap::default();
        exes.insert(0, exe_path.clone());
        let outcome = fast::run_executables(&ws, &targets, &dag, &[0], &exes, &mut locks, &mut ran)
            .expect("run fast executables");
        assert_eq!(outcome.len(), 1);
        let failure = outcome[0].1.as_ref().expect_err("exit(1) script must fail");
        assert!(failure.fatal, "runtime script failure must be fatal");
        assert!(!exe_path.exists());
        assert!(!dir.join("script.fast").exists());

        let mut state = SettleState {
            dag,
            ran,
            failed: HashSet::default(),
            fatal_errors: Vec::new(),
            kept_queue: VecDeque::new(),
        };
        finish_wave(&mut ws, &[0], outcome, &mut state);
        assert!(
            !state.fatal_errors.is_empty(),
            "finish_wave must record runtime script failure as fatal"
        );
    });

    drop(std::fs::remove_dir_all(&root));
    drop(std::fs::remove_dir_all(&home));
}

#[test]
fn script_manifests_invalidate_on_included_and_env_reads() {
    let root = temp_dir("watch");
    let home = temp_dir("watch-home");
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
                "fn main() { let _ = include_str!(\"data.txt\"); let _ = option_env!(\"BUILD_FLAG\"); }\n",
            ),
            ("data.txt", "one\n"),
            ("src/main.rs", "fn main() {}\n"),
        ],
    );
    let targets = test_targets();
    let ws = crate::graph::load(&root, &targets, &Default::default()).expect("load ws");

    with_scripts_env(&home, || {
        let dir = store::dir(&ws, 0);
        let exe_path = dir.join("script");
        let inc_path = root.join("data.txt");
        let meta = binary::fast_meta_hash(&ws, &targets, 0);
        install_mock_script(&exe_path, "#!/bin/sh\nexit 0\n");

        let fast_envs = vec![(
            "BUILD_FLAG".to_owned(),
            Some(binary::hash_fast_env("BUILD_FLAG", "alpha")),
        )];
        let key_envs = vec![(
            "BUILD_FLAG".to_owned(),
            Some(store::hash_env_val(
                &binary::KEY_MAGIC,
                "BUILD_FLAG",
                "alpha",
            )),
        )];
        let key_bytes = binary::encode_watch_manifest(
            binary::KEY_MAGIC,
            binary::KEY_VERSION,
            99,
            std::slice::from_ref(&inc_path),
            &key_envs,
        )
        .expect("encode key manifest");
        store::write_new_bytes(&dir.join("script.key"), &key_bytes).expect("write key manifest");
        binary::write_fast_manifest(&dir, meta, std::slice::from_ref(&inc_path), &fast_envs)
            .expect("write script.fast");

        let env_a = std::sync::Arc::new(HashMap::from_iter([(
            "BUILD_FLAG".to_owned(),
            "alpha".to_owned(),
        )]));
        crate::with_client_env(Some(env_a.clone()), || {
            assert_eq!(
                binary::fast_kept(&ws, &targets, 0, &dir),
                Some(exe_path.clone())
            );
            assert!(binary::validate_watch_manifest(
                &dir.join("script.key"),
                binary::KEY_MAGIC,
                binary::KEY_VERSION,
                99,
            ));
        });

        let env_b = std::sync::Arc::new(HashMap::from_iter([(
            "BUILD_FLAG".to_owned(),
            "beta".to_owned(),
        )]));
        crate::with_client_env(Some(env_b), || {
            assert!(binary::fast_kept(&ws, &targets, 0, &dir).is_none());
            assert!(!binary::validate_watch_manifest(
                &dir.join("script.key"),
                binary::KEY_MAGIC,
                binary::KEY_VERSION,
                99,
            ));
        });

        crate::with_client_env(Some(env_a), || {
            std::fs::write(&inc_path, "two_changed\n").expect("edit included file");
            assert!(binary::fast_kept(&ws, &targets, 0, &dir).is_none());
            assert!(!binary::validate_watch_manifest(
                &dir.join("script.key"),
                binary::KEY_MAGIC,
                binary::KEY_VERSION,
                99,
            ));
        });
    });

    drop(std::fs::remove_dir_all(&root));
    drop(std::fs::remove_dir_all(&home));
}

#[test]
fn host_unit_out_dir_stays_debug_under_release_profile() {
    let root = temp_dir("host-out");
    let home = temp_dir("host-out-home");
    write_files(
        &root,
        &[
            (
                "Cargo.toml",
                concat!(
                    "[package]\nname = \"app\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
                    "[build-dependencies]\nhelper = { path = \"helper\" }\n",
                ),
            ),
            (
                "helper/Cargo.toml",
                "[package]\nname = \"helper\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
            ),
            ("helper/build.rs", "fn main() {}\n"),
            ("helper/src/lib.rs", ""),
            ("build.rs", "fn main() {}\n"),
            ("src/main.rs", "fn main() {}\n"),
            (
                "Cargo.lock",
                concat!(
                    "version = 4\n",
                    "[[package]]\nname = \"app\"\nversion = \"0.1.0\"\ndependencies = [\"helper\"]\n",
                    "[[package]]\nname = \"helper\"\nversion = \"0.1.0\"\n",
                ),
            ),
        ],
    );
    let targets = test_targets();
    let ws = crate::graph::load(&root, &targets, &Default::default()).expect("load ws");
    let host_unit = ws
        .graph
        .units
        .iter()
        .position(|u| u.side == crate::graph::Side::Host)
        .expect("host unit");
    let target_unit = ws
        .graph
        .units
        .iter()
        .position(|u| u.side == crate::graph::Side::Target)
        .expect("target unit");

    with_scripts_env(&home, || {
        let debug_host = store::dir(&ws, host_unit);
        let debug_target = store::dir(&ws, target_unit);
        let mut rel_cfg = (*crate::config()).clone();
        rel_cfg.profile = crate::outline::tree::Profile::Release;
        crate::with_config(Arc::new(rel_cfg), || {
            assert_eq!(store::dir(&ws, host_unit), debug_host);
            assert_ne!(store::dir(&ws, target_unit), debug_target);
            binary::with_script_profile(|| {
                assert_eq!(crate::profile(), crate::outline::tree::Profile::Debug);
                assert_eq!(store::dir(&ws, host_unit), debug_host);
            });
        });
    });

    drop(std::fs::remove_dir_all(&root));
    drop(std::fs::remove_dir_all(&home));
}

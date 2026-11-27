use std::path::Path;

use super::meta::{decode_meta, encode_meta};
use super::*;
use crate::outline::Role;

fn temp_dir(tag: &str) -> PathBuf {
    let seq = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let dir = std::env::temp_dir().join(format!(
        "haste-local-idx-{tag}-{}-{seq}",
        std::process::id()
    ));
    drop(std::fs::remove_dir_all(&dir));
    std::fs::create_dir_all(&dir).expect("mkdir");
    dir.canonicalize().unwrap_or(dir)
}

fn with_local_env<R>(home: &Path, disable_local: bool, f: impl FnOnce() -> R) -> R {
    let _guard = crate::sync::lock(&crate::substrate::TEST_HOME_LOCK);
    let prev_home = std::env::var_os("HASTE_HOME");
    let prev_no_local = std::env::var_os("HASTE_NO_LOCAL_INDEX");
    // SAFETY: serialized by `TEST_HOME_LOCK` within unit tests.
    unsafe {
        std::env::set_var("HASTE_HOME", home);
        if disable_local {
            std::env::set_var("HASTE_NO_LOCAL_INDEX", "1");
        } else {
            std::env::remove_var("HASTE_NO_LOCAL_INDEX");
        }
    }
    let out = f();
    // SAFETY: serialized by `TEST_HOME_LOCK` within unit tests.
    unsafe {
        match prev_home {
            Some(v) => std::env::set_var("HASTE_HOME", v),
            None => std::env::remove_var("HASTE_HOME"),
        }
        match prev_no_local {
            Some(v) => std::env::set_var("HASTE_NO_LOCAL_INDEX", v),
            None => std::env::remove_var("HASTE_NO_LOCAL_INDEX"),
        }
    }
    out
}

fn test_targets() -> Targets {
    let host = crate::graph::cfg::Target::of(&crate::host_target());
    Targets {
        target: host.clone(),
        host,
    }
}

fn write_test_workspace(root: &Path) {
    std::fs::create_dir_all(root.join("crates/lib_a/src")).expect("mkdir lib_a");
    std::fs::create_dir_all(root.join("crates/app/src")).expect("mkdir app");
    std::fs::write(
        root.join("Cargo.toml"),
        "[workspace]\nmembers = [\"crates/lib_a\", \"crates/app\"]\nresolver = \"2\"\n",
    )
    .expect("write Cargo.toml");
    std::fs::write(
        root.join("Cargo.lock"),
        "version = 4\n\n[[package]]\nname = \"lib_a\"\nversion = \"0.1.0\"\n\n[[package]]\nname = \"app\"\nversion = \"0.1.0\"\ndependencies = [\"lib_a\"]\n",
    )
    .expect("write Cargo.lock");
    std::fs::write(
        root.join("crates/lib_a/Cargo.toml"),
        "[package]\nname = \"lib_a\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .expect("write lib_a Cargo.toml");
    std::fs::write(
        root.join("crates/lib_a/src/lib.rs"),
        "pub mod extra;\npub const MSG: &str = include_str!(\"msg.txt\");\n#[macro_export]\nmacro_rules! make_val {\n    () => { { let local_var = 41usize; local_var + 1 } };\n}\nmacro_rules! def_helper {\n    () => { pub fn helper() -> usize { let v = 41usize; v + 1 } };\n}\ndef_helper!();\npub fn base_val() -> usize { make_val!() + helper() + extra::ext() }\n",
    )
    .expect("write lib.rs");
    std::fs::write(
        root.join("crates/lib_a/src/extra.rs"),
        "pub fn ext() -> usize { 7 }\n",
    )
    .expect("write extra.rs");
    std::fs::write(root.join("crates/lib_a/src/msg.txt"), "hello").expect("write msg.txt");
    std::fs::write(
        root.join("crates/app/Cargo.toml"),
        "[package]\nname = \"app\"\nversion = \"0.1.0\"\nedition = \"2021\"\n[dependencies]\nlib_a = { path = \"../lib_a\" }\n",
    )
    .expect("write app Cargo.toml");
    std::fs::write(
        root.join("crates/app/src/main.rs"),
        "fn main() { let _ = lib_a::base_val(); }\n",
    )
    .expect("write main.rs");
}

fn build_comp(ws: &Workspace, targets: &Targets) -> Compilation {
    let mut outline = crate::outline::new(ws, targets, &[]);
    let selected = vec![true; outline.crates.len()];
    outline.load_selected(&selected).expect("load_selected");
    let mut names = crate::names::Names::new_with(ws, &outline, None);
    for k in 0..selected.len() {
        names.crates[k].set_eager();
    }
    names.resolve(&mut outline).expect("resolve");
    Compilation::for_test(outline, names)
}

#[test]
fn meta_round_trip_and_key_mismatch() {
    let meta = LocalMeta {
        content_key: 0x1234_5678_9abc_def0,
        dep_keys: vec![(11, 22)],
        watched: vec![
            (PathBuf::from("/tmp/a.rs"), Some((10, 20))),
            (PathBuf::from("/tmp/missing.rs"), None),
        ],
        included: vec![(PathBuf::from("/tmp/msg.txt"), PathBuf::from("/tmp/a.rs"))],
        env_reads: hash_env_reads(&[("K".to_owned(), Some("V".to_owned()))]),
    };
    let bytes = encode_meta(0xdead_beef, &meta).expect("encode meta");
    assert!(decode_meta(&bytes, 0xcafe_babe).is_none());
    let decoded = decode_meta(&bytes, 0xdead_beef).expect("decode meta");
    assert_eq!(decoded.content_key, meta.content_key);
    assert_eq!(decoded.dep_keys, meta.dep_keys);
    assert_eq!(decoded.watched, meta.watched);
    assert_eq!(decoded.included, meta.included);
    assert_eq!(decoded.env_reads, meta.env_reads);
}

#[test]
fn scope_bit_isolation_preserves_sysroot_and_deps_and_tags_local() {
    use crate::outline::source::Mark;
    let mut out = Vec::new();
    let marks = vec![
        (0, Mark::Scope { id: 1, parent: 0 }),
        (
            1,
            Mark::Scope {
                id: 2,
                parent: (1 << 31) | 7,
            },
        ),
        (
            2,
            Mark::Scope {
                id: 3,
                parent: (1 << 30) | 9,
            },
        ),
        (
            3,
            Mark::Scope {
                id: 4,
                parent: LOCAL_SCOPE_BIT | 11,
            },
        ),
    ];
    super::super::sysroot::tokens::write_marks(&mut out, &marks, &[Some(0)]).expect("write");
    let decoded =
        super::super::sysroot::tokens::read_marks(&out, &[0], LOCAL_SCOPE_BIT).expect("read");
    assert_eq!(
        decoded[0].1,
        Mark::Scope {
            id: LOCAL_SCOPE_BIT | 1,
            parent: 0,
        }
    );
    assert_eq!(
        decoded[1].1,
        Mark::Scope {
            id: LOCAL_SCOPE_BIT | 2,
            parent: (1 << 31) | 7,
        }
    );
    assert_eq!(
        decoded[2].1,
        Mark::Scope {
            id: LOCAL_SCOPE_BIT | 3,
            parent: (1 << 30) | 9,
        }
    );
    assert_eq!(
        decoded[3].1,
        Mark::Scope {
            id: LOCAL_SCOPE_BIT | 4,
            parent: LOCAL_SCOPE_BIT | 11,
        }
    );
    assert_eq!(LOCAL_SCOPE_BIT & ((1 << 31) | (1 << 30)), 0);
}

#[test]
fn file_stamp_invalidation_and_main_edit_cache_hit() {
    let root = temp_dir("ws");
    let home = temp_dir("home");
    write_test_workspace(&root);

    with_local_env(&home, false, || {
        let targets = test_targets();
        let ws = crate::graph::load(&root, &targets, &Default::default()).expect("load ws");
        let sys_key = super::super::sysroot_key(&ws, &targets);
        let comp1 = build_comp(&ws, &targets);
        let lib_krate = comp1
            .outline
            .roots
            .iter()
            .position(|r| r.name == "lib_a" && r.role == Role::Lib)
            .expect("lib_a");
        let empty_hits = vec![false; comp1.outline.crates.len()];
        let empty_keys = vec![None; comp1.outline.crates.len()];
        let heads = save(&ws, &targets, &comp1, sys_key, &empty_hits, &empty_keys);
        assert!(heads.contains_key(&(lib_krate as u32)));

        std::fs::write(
            root.join("crates/app/src/main.rs"),
            "fn main() { let x = lib_a::base_val(); let _ = x + 1; }\n",
        )
        .expect("edit main.rs");

        let mut outline2 = crate::outline::new(&ws, &targets, &[]);
        let loaded2 = try_load(&ws, &targets, &mut outline2, None, sys_key, true, true);
        assert!(loaded2.hits[lib_krate]);
        assert!(!outline2.crates[lib_krate].partial());
        let expected_included = vec![(
            root.join("crates/lib_a/src/msg.txt"),
            root.join("crates/lib_a/src/lib.rs"),
        )];
        assert_eq!(
            outline2.crates[lib_krate].cached_included.as_deref(),
            Some(expected_included.as_slice())
        );
        assert_eq!(outline2.included(lib_krate), expected_included);
        let has_local_scope = outline2.crates[lib_krate].files.iter().any(|f| {
            f.source.marks.iter().any(|&(_, m)| {
                matches!(
                    m,
                    crate::outline::source::Mark::Scope { id, .. }
                    if (id & LOCAL_SCOPE_BIT) != 0 && (id & ((1 << 31) | (1 << 30))) == 0
                )
            })
        });
        assert!(has_local_scope);

        std::fs::write(
            root.join("crates/lib_a/src/extra.rs"),
            "pub fn ext() -> usize { 99 }\n",
        )
        .expect("edit extra.rs");
        let mut outline3 = crate::outline::new(&ws, &targets, &[]);
        assert!(!try_load(&ws, &targets, &mut outline3, None, sys_key, true, true).hits[lib_krate]);

        std::fs::write(
            root.join("crates/lib_a/src/extra.rs"),
            "pub fn ext() -> usize { 7 }\n",
        )
        .expect("restore extra.rs");
        let comp4 = build_comp(&ws, &targets);
        save(&ws, &targets, &comp4, sys_key, &empty_hits, &empty_keys);

        std::fs::write(root.join("crates/lib_a/src/msg.txt"), "changed include")
            .expect("edit msg.txt");
        let mut outline5 = crate::outline::new(&ws, &targets, &[]);
        assert!(!try_load(&ws, &targets, &mut outline5, None, sys_key, true, true).hits[lib_krate]);
    });

    with_local_env(&home, true, || {
        assert!(!enabled());
    });

    drop(std::fs::remove_dir_all(&root));
    drop(std::fs::remove_dir_all(&home));
}

#[test]
fn env_reads_never_leak_secrets_on_disk_and_invalidate_on_change() {
    let root = temp_dir("env-secret-ws");
    let home = temp_dir("env-secret-home");
    write_test_workspace(&root);

    let key_name = "HASTE_TEST_SECRET_22B";
    let secret_v1 = "SECRET_TOKEN_DO_NOT_LEAK_9911";
    let secret_v2 = "SECRET_TOKEN_ROTATED_2222";

    with_local_env(&home, false, || {
        let prev = std::env::var_os(key_name);
        // SAFETY: serialized by `TEST_HOME_LOCK` via `with_local_env`.
        unsafe { std::env::set_var(key_name, secret_v1) };

        let targets = test_targets();
        let ws = crate::graph::load(&root, &targets, &Default::default()).expect("load ws");
        let sys_key = super::super::sysroot_key(&ws, &targets);
        let comp = build_comp(&ws, &targets);
        let lib_krate = comp
            .outline
            .roots
            .iter()
            .position(|r| r.name == "lib_a" && r.role == Role::Lib)
            .expect("lib_a");
        crate::sync::lock(&comp.outline.crates[lib_krate].env.env_reads)
            .insert(key_name.to_owned(), Some(secret_v1.to_owned()));

        let empty_hits = vec![false; comp.outline.crates.len()];
        let empty_keys = vec![None; comp.outline.crates.len()];
        save(&ws, &targets, &comp, sys_key, &empty_hits, &empty_keys);

        let mut stack = vec![home.clone()];
        let mut file_count = 0usize;
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).expect("read cache dir").flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else if path.is_file() {
                    file_count += 1;
                    let bytes = std::fs::read(&path).expect("read cached file");
                    assert!(
                        !bytes
                            .windows(secret_v1.len())
                            .any(|w| w == secret_v1.as_bytes()),
                        "plaintext env secret leaked in {}",
                        path.display()
                    );
                }
            }
        }
        assert!(file_count >= 2, "expected .meta and .bin cache files");

        let mut outline_hit = crate::outline::new(&ws, &targets, &[]);
        let loaded_hit = try_load(&ws, &targets, &mut outline_hit, None, sys_key, true, true);
        assert!(loaded_hit.hits[lib_krate]);
        let restored = crate::sync::lock(&outline_hit.crates[lib_krate].env.env_reads)
            .get(key_name)
            .cloned();
        assert_eq!(restored, Some(Some(secret_v1.to_owned())));

        // SAFETY: serialized by `TEST_HOME_LOCK` via `with_local_env`.
        unsafe { std::env::set_var(key_name, secret_v2) };
        let mut outline_miss = crate::outline::new(&ws, &targets, &[]);
        let loaded_miss = try_load(&ws, &targets, &mut outline_miss, None, sys_key, true, true);
        assert!(!loaded_miss.hits[lib_krate]);

        // SAFETY: serialized by `TEST_HOME_LOCK` via `with_local_env`.
        unsafe {
            match prev {
                Some(v) => std::env::set_var(key_name, v),
                None => std::env::remove_var(key_name),
            }
        }
    });

    drop(std::fs::remove_dir_all(&root));
    drop(std::fs::remove_dir_all(&home));
}

use super::*;
use crate::graph::Targets;
use crate::graph::cfg::Target;
use crate::outline::Role;
use std::sync::Arc;

fn temp_dir(tag: &str) -> PathBuf {
    let seq = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let dir = std::env::temp_dir().join(format!("haste-local-{tag}-{}-{seq}", std::process::id()));
    drop(std::fs::remove_dir_all(&dir));
    std::fs::create_dir_all(&dir).expect("mkdir");
    dir.canonicalize().unwrap_or(dir)
}

fn with_local_env<R>(home: &Path, f: impl FnOnce() -> R) -> R {
    let _guard = crate::sync::lock(&crate::substrate::TEST_HOME_LOCK);
    let prev_home = std::env::var_os("HASTE_HOME");
    // SAFETY: serialized by `TEST_HOME_LOCK` within unit tests.
    unsafe { std::env::set_var("HASTE_HOME", home) };
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

fn client_env(pairs: &[(&str, &str)]) -> Arc<HashMap<String, String>> {
    let mut map = HashMap::default();
    for &(k, v) in pairs {
        map.insert(k.to_owned(), v.to_owned());
    }
    Arc::new(map)
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

fn load_outline(root: &Path) -> (Workspace, Outline) {
    let host = Target::of(&crate::host_target());
    let targets = Targets {
        target: host.clone(),
        host,
    };
    let ws = crate::graph::load(root, &targets, &Default::default()).expect("load ws");
    let mut outline = crate::outline::new(&ws, &targets, &[]);
    let selected = vec![true; outline.crates.len()];
    outline.load_selected(&selected).expect("load_selected");
    (ws, outline)
}

fn find_crate(outline: &Outline, name: &str, role: Role) -> usize {
    outline
        .roots
        .iter()
        .position(|r| r.name == name && r.role == role)
        .expect("find crate")
}

fn verify_env_and_inc_guards(
    ws: &Workspace,
    outline: &Outline,
    crates: &[usize],
    env_set: &Arc<HashMap<String, String>>,
    inc_path: &Path,
    edited: &str,
) {
    let empty = Arc::new(HashMap::default());
    assert!(crate::with_client_env(Some(empty), || {
        partition(ws, outline, crates, (1, 2)).hits.is_empty()
    }));
    std::fs::write(inc_path, edited).expect("edit inc");
    assert!(crate::with_client_env(Some(env_set.clone()), || {
        partition(ws, outline, crates, (1, 2)).hits.is_empty()
    }));
}

#[test]
fn encode_decode_round_trip_and_hides_plaintext_env_values() {
    let secret = "super_secret_token_9876";
    let inc = vec![
        (PathBuf::from("/tmp/a.txt"), Some((12, 345_678))),
        (PathBuf::from("/tmp/missing.txt"), None),
    ];
    let envs = vec![
        ("VAR_A".to_owned(), Some(hash_env_val("VAR_A", secret))),
        ("VAR_B".to_owned(), None),
    ];
    let check_bytes = encode_entry(0xDEAD_BEEF, &inc, &envs).expect("encode");
    assert!(
        !check_bytes
            .windows(secret.len())
            .any(|w| w == secret.as_bytes())
    );
    assert!(decode_entry(&check_bytes, 0xCAFE_BABE).is_none());
    assert_eq!(
        decode_entry(&check_bytes, 0xDEAD_BEEF),
        Some((inc.clone(), envs.clone()))
    );

    let files = vec![(11, 22), (33, 44)];
    let f_bytes = encode_files_entry(0xABCD_1234, &files, &inc, &envs).expect("encode");
    assert!(
        !f_bytes
            .windows(secret.len())
            .any(|w| w == secret.as_bytes())
    );
    assert!(decode_files_entry(&f_bytes, 0x9999_0000).is_none());
    let (dec_files, watch) = decode_files_entry(&f_bytes, 0xABCD_1234).expect("decode");
    assert_eq!(
        (dec_files.get(&11), dec_files.get(&33)),
        (Some(&22), Some(&44))
    );
    assert_eq!(watch, (inc, envs));
}

#[test]
fn try_hit_detects_env_and_included_file_changes() {
    let tmp = temp_dir("env-inc");
    let inc_file = tmp.join("data.txt");
    std::fs::write(&inc_file, "hello").expect("write");
    let entry_id = 0x1111_2222_3333_4444_u64;
    let cache_file = tmp.join(format!("{entry_id:016x}.check"));
    let envs = vec![(
        "HASTE_TEST_LOCAL_ENV".to_owned(),
        Some(hash_env_val("HASTE_TEST_LOCAL_ENV", "v1")),
    )];
    let bytes = encode_entry(
        entry_id,
        &[(inc_file.clone(), file_stamp(&inc_file))],
        &envs,
    )
    .expect("encode");
    write_new_bytes(&cache_file, &bytes).expect("write");
    let env_v1 = client_env(&[("HASTE_TEST_LOCAL_ENV", "v1")]);
    let env_v2 = client_env(&[("HASTE_TEST_LOCAL_ENV", "v2")]);
    assert!(crate::with_client_env(Some(env_v1.clone()), || {
        try_hit_at(&cache_file, entry_id).is_some()
    }));
    assert!(crate::with_client_env(Some(env_v2), || {
        try_hit_at(&cache_file, entry_id).is_none()
    }));
    std::fs::write(&inc_file, "hello world!").expect("edit inc");
    assert!(crate::with_client_env(Some(env_v1), || {
        try_hit_at(&cache_file, entry_id).is_none()
    }));
    drop(std::fs::remove_dir_all(&tmp));
}

#[test]
fn partial_file_update_preserves_sibling_envs_and_included() {
    let root = temp_dir("partial-ws");
    let home = temp_dir("partial-home");
    write_files(
        &root,
        &[
            (
                "Cargo.toml",
                "[package]\nname = \"twofile\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
            ),
            (
                "Cargo.lock",
                "version = 4\n[[package]]\nname = \"twofile\"\nversion = \"0.1.0\"\n",
            ),
            ("src/inc_a.txt", "inc_v1"),
            ("src/lib.rs", "pub mod a;\npub mod b;\n"),
            ("src/a.rs", "pub fn fa() -> &'static str { \"a\" }\n"),
            ("src/b.rs", "pub fn fb() -> usize { 1 }\n"),
        ],
    );
    with_local_env(&home, || {
        let env_set = client_env(&[("PILOT_LABEL", "secret_pilot_val")]);
        let (ws1, mut outline1) = load_outline(&root);
        let k = find_crate(&outline1, "twofile", Role::Lib);
        crate::sync::lock(&outline1.crates[k].env.env_reads).insert(
            "PILOT_LABEL".to_owned(),
            Some("secret_pilot_val".to_owned()),
        );
        outline1.crates[k].cached_included =
            Some(vec![(root.join("src/inc_a.txt"), root.join("src/a.rs"))]);
        crate::with_client_env(Some(env_set.clone()), || {
            let p1 = partition(&ws1, &outline1, &[k], (1, 2));
            record(&outline1, &p1.dirty, &HashSet::default());
        });
        for entry in std::fs::read_dir(cache_dir()).expect("readdir").flatten() {
            let raw = std::fs::read(entry.path()).expect("read cache");
            assert!(!raw.windows(16).any(|w| w == b"secret_pilot_val"));
        }
        std::fs::write(root.join("src/b.rs"), "pub fn fb() -> usize { 2 }\n").expect("edit");
        let (ws2, outline2) = load_outline(&root);
        let b_idx = outline2.crates[k]
            .files
            .iter()
            .position(|f| f.source.path.ends_with("b.rs"))
            .expect("b.rs");
        crate::with_client_env(Some(env_set.clone()), || {
            let p2 = partition(&ws2, &outline2, &[k], (1, 2));
            assert!(p2.hits.is_empty());
            assert_eq!(p2.dirty[0].dirty_files.as_deref(), Some(&[b_idx][..]));
            record(&outline2, &p2.dirty, &HashSet::default());
            assert_eq!(partition(&ws2, &outline2, &[k], (1, 2)).hits, vec![k]);
        });
        verify_env_and_inc_guards(
            &ws2,
            &outline2,
            &[k],
            &env_set,
            &root.join("src/inc_a.txt"),
            "inc_v2",
        );
    });
    drop(std::fs::remove_dir_all(&root));
    drop(std::fs::remove_dir_all(&home));
}

#[test]
fn dependent_crate_record_inherits_hit_dependency_watch() {
    let root = temp_dir("dep-ws");
    let home = temp_dir("dep-home");
    write_files(
        &root,
        &[
            (
                "Cargo.toml",
                "[workspace]\nmembers = [\"crates/lib_a\", \"crates/app\"]\nresolver = \"2\"\n",
            ),
            (
                "Cargo.lock",
                "version = 4\n[[package]]\nname = \"lib_a\"\nversion = \"0.1.0\"\n[[package]]\nname = \"app\"\nversion = \"0.1.0\"\ndependencies = [\"lib_a\"]\n",
            ),
            (
                "crates/lib_a/Cargo.toml",
                "[package]\nname = \"lib_a\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
            ),
            ("crates/lib_a/src/inc.txt", "lib_v1"),
            (
                "crates/lib_a/src/lib.rs",
                "pub fn f() -> &'static str { \"v\" }\n",
            ),
            (
                "crates/app/Cargo.toml",
                "[package]\nname = \"app\"\nversion = \"0.1.0\"\nedition = \"2021\"\n[dependencies]\nlib_a = { path = \"../lib_a\" }\n",
            ),
            (
                "crates/app/src/main.rs",
                "fn main() { let _ = lib_a::f(); }\n",
            ),
        ],
    );
    with_local_env(&home, || {
        let env_set = client_env(&[("PILOT_LABEL", "dep_secret")]);
        let (ws1, mut outline1) = load_outline(&root);
        let lib_k = find_crate(&outline1, "lib_a", Role::Lib);
        let app_k = find_crate(&outline1, "app", Role::Bin);
        crate::sync::lock(&outline1.crates[lib_k].env.env_reads)
            .insert("PILOT_LABEL".to_owned(), Some("dep_secret".to_owned()));
        outline1.crates[lib_k].cached_included = Some(vec![(
            root.join("crates/lib_a/src/inc.txt"),
            root.join("crates/lib_a/src/lib.rs"),
        )]);
        crate::with_client_env(Some(env_set.clone()), || {
            let p1 = partition(&ws1, &outline1, &[app_k, lib_k], (1, 2));
            record(&outline1, &p1.dirty, &HashSet::default());
        });
        std::fs::write(
            root.join("crates/app/src/main.rs"),
            "fn main() { let x = lib_a::f().len(); let _ = x; }\n",
        )
        .expect("edit");
        let (ws2, outline2) = load_outline(&root);
        crate::with_client_env(Some(env_set.clone()), || {
            let p2 = partition(&ws2, &outline2, &[app_k, lib_k], (1, 2));
            assert_eq!(p2.hits, vec![lib_k]);
            assert_eq!(p2.dirty.len(), 1);
            assert_eq!(p2.dirty[0].krate, app_k);
            record(&outline2, &p2.dirty, &HashSet::default());
            assert_eq!(
                partition(&ws2, &outline2, &[app_k, lib_k], (1, 2))
                    .hits
                    .len(),
                2
            );
        });
        verify_env_and_inc_guards(
            &ws2,
            &outline2,
            &[app_k, lib_k],
            &env_set,
            &root.join("crates/lib_a/src/inc.txt"),
            "lib_v2",
        );
    });
    drop(std::fs::remove_dir_all(&root));
    drop(std::fs::remove_dir_all(&home));
}

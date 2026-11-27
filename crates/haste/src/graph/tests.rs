use std::path::{Path, PathBuf};

use super::*;

fn temp_dir(tag: &str) -> PathBuf {
    let seq = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let dir = std::env::temp_dir().join(format!("haste-graph-{tag}-{}-{seq}", std::process::id()));
    drop(std::fs::remove_dir_all(&dir));
    std::fs::create_dir_all(&dir).expect("mkdir");
    dir.canonicalize().unwrap_or(dir)
}

fn write_member(root: &Path, rel: &str, name: &str, edition: &str) {
    let dir = root.join(rel);
    std::fs::create_dir_all(dir.join("src")).expect("mkdir member");
    std::fs::write(
        dir.join("Cargo.toml"),
        format!("[package]\nname = \"{name}\"\nversion = \"0.1.0\"\nedition = \"{edition}\"\n"),
    )
    .expect("write Cargo.toml");
    std::fs::write(dir.join("src/lib.rs"), "").expect("write lib.rs");
}

#[test]
fn workspace_members_globs_and_exclude_work() {
    let top = temp_dir("ws-globs");
    std::fs::write(
        top.join("Cargo.toml"),
        concat!(
            "[workspace]\n",
            "members = [\"crates/*/\", \"nested/*/*\", \"groups/*/crates/*\", \"class/[ab]\", \"q/item-?\", \"prefixed/foo-*\", \"top-*\"]\n",
            "exclude = [\"crates/ignored\", \"nested/grp/skip-*\", \"top-excluded\"]\n",
        ),
    )
    .expect("write root Cargo.toml");

    write_member(&top, "top-keep", "top-keep", "2021");
    write_member(&top, "top-excluded", "top-excluded", "2021");
    write_member(&top, "crates/kept", "kept", "2021");
    write_member(&top, "crates/ignored", "ignored", "2021");
    write_member(&top, "nested/grp/ok", "nested-ok", "2021");
    write_member(&top, "nested/grp/skip-me", "nested-skip", "2021");
    write_member(&top, "groups/a/crates/first", "group-first", "2021");
    write_member(&top, "class/a", "class-a", "2021");
    write_member(&top, "class/b", "class-b", "2021");
    write_member(&top, "class/c", "class-c", "2021");
    write_member(&top, "q/item-1", "q-1", "2021");
    write_member(&top, "q/item-10", "q-10", "2021");
    write_member(&top, "prefixed/foo-one", "foo-one", "2021");
    write_member(&top, "prefixed/bar-two", "bar-two", "2021");

    let ws = manifest::workspace_of(&top).expect("read ws");
    let got: BTreeSet<PathBuf> = manifest::workspace_member_dirs(&top, ws.as_ref(), "members")
        .into_iter()
        .map(|p| manifest::normalize(&p))
        .collect();
    let expected: BTreeSet<PathBuf> = [
        top.join("top-keep"),
        top.join("crates/kept"),
        top.join("nested/grp/ok"),
        top.join("groups/a/crates/first"),
        top.join("class/a"),
        top.join("class/b"),
        top.join("q/item-1"),
        top.join("prefixed/foo-one"),
    ]
    .into_iter()
    .map(|p| manifest::normalize(&p))
    .collect();
    assert_eq!(got, expected);

    let glob_dirs: BTreeSet<PathBuf> = manifest::member_glob_dirs(&top)
        .into_iter()
        .map(|p| manifest::normalize(&p))
        .collect();
    for expected_parent in [
        top.clone(),
        top.join("crates"),
        top.join("nested"),
        top.join("nested/grp"),
        top.join("groups"),
        top.join("groups/a"),
        top.join("groups/a/crates"),
        top.join("class"),
        top.join("q"),
        top.join("prefixed"),
    ] {
        assert!(
            glob_dirs.contains(&manifest::normalize(&expected_parent)),
            "missing glob parent {}",
            expected_parent.display()
        );
    }

    drop(std::fs::remove_dir_all(&top));
}

#[test]
fn non_virtual_edition_2021_workspace_keeps_resolver_v2_when_root_pruned() {
    let top = temp_dir("resolver-v2");
    std::fs::write(
        top.join("Cargo.toml"),
        concat!(
            "[package]\n",
            "name = \"root-pkg\"\n",
            "version = \"0.1.0\"\n",
            "edition = \"2021\"\n",
            "[workspace]\n",
            "members = [\"crates/leaf\"]\n",
        ),
    )
    .expect("write root Cargo.toml");
    std::fs::create_dir_all(top.join("src")).expect("mkdir src");
    std::fs::write(top.join("src/lib.rs"), "").expect("write lib.rs");
    write_member(&top, "crates/leaf", "leaf", "2018");

    let leaf_manifest = manifest::read(&top.join("crates/leaf"), None).expect("read leaf");
    let leaf_pkg = Package {
        locked: Locked {
            name: "leaf".to_owned(),
            version: semver::Version::new(0, 1, 0),
            source: None,
            checksum: None,
            dependencies: Vec::new(),
        },
        manifest: leaf_manifest,
        resolved: BTreeMap::new(),
        sysroot: false,
    };
    let ws = manifest::workspace_of(&top).expect("read ws");
    assert!(
        manifest::uses_resolver_v2(ws.as_ref(), &top, &[leaf_pkg], &[0]),
        "pruned edition 2021 workspace root must preserve resolver v2"
    );

    drop(std::fs::remove_dir_all(&top));
}

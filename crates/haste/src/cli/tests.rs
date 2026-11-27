use super::*;
use crate::graph::manifest::Manifest;
use crate::graph::{Graph, Package, Side, Unit, Workspace, lock::Locked};
use std::collections::BTreeMap;

fn test_outline() -> outline::Outline {
    outline::Outline {
        roots: vec![
            outline::Root {
                unit: 0,
                name: "my-bin".to_owned(),
                path: std::path::PathBuf::from("src/bin/my-bin.rs"),
                role: outline::Role::Bin,
                edition: "2021".to_owned(),
                proc_macro: false,
            },
            outline::Root {
                unit: 0,
                name: "my-test".to_owned(),
                path: std::path::PathBuf::from("tests/my-test.rs"),
                role: outline::Role::Test,
                edition: "2021".to_owned(),
                proc_macro: false,
            },
        ],
        crates: Vec::new(),
        deps: Vec::new(),
        cache: outline::tree::Files::default(),
        procs: rustc_hash::FxHashMap::default(),
        pins: Vec::new(),
    }
}

#[test]
fn test_target_selected_matches_hyphenated_bin_and_test_names() {
    let outline = test_outline();
    let by_bin_hyphen = crate::Selection {
        bin: vec!["my-bin".to_owned()],
        ..crate::Selection::default()
    };
    let by_bin_under = crate::Selection {
        bin: vec!["my_bin".to_owned()],
        ..crate::Selection::default()
    };
    assert!(test_target_selected(&outline, 0, &by_bin_hyphen));
    assert!(test_target_selected(&outline, 0, &by_bin_under));
    assert!(!test_target_selected(&outline, 1, &by_bin_hyphen));

    let by_test_hyphen = crate::Selection {
        test: vec!["my-test".to_owned()],
        ..crate::Selection::default()
    };
    let by_test_under = crate::Selection {
        test: vec!["my_test".to_owned()],
        ..crate::Selection::default()
    };
    assert!(test_target_selected(&outline, 1, &by_test_hyphen));
    assert!(test_target_selected(&outline, 1, &by_test_under));
    assert!(!test_target_selected(&outline, 0, &by_test_hyphen));
}

#[test]
fn default_run_bin_selects_package_default_run_when_multiple_bins_exist() {
    let mut manifest = Manifest::default();
    manifest.bins = vec![
        graph::manifest::TargetSpec {
            name: "alpha".to_owned(),
            ..Default::default()
        },
        graph::manifest::TargetSpec {
            name: "beta".to_owned(),
            ..Default::default()
        },
    ];
    manifest.default_run = Some("beta".to_owned());
    let pkg = Package {
        locked: Locked {
            name: "demo".to_owned(),
            version: semver::Version::new(0, 1, 0),
            source: None,
            checksum: None,
            dependencies: Vec::new(),
        },
        manifest,
        resolved: BTreeMap::new(),
        sysroot: false,
    };
    let ws = Workspace {
        packages: vec![pkg],
        graph: Graph {
            units: vec![Unit {
                package: 0,
                side: Side::Target,
                features: Default::default(),
                deps: Vec::new(),
                script: None,
            }],
            members: vec![0],
        },
        rustflags: Vec::new(),
        config_env: Vec::new(),
        profile_dev: Default::default(),
        profile_release: Default::default(),
    };
    assert_eq!(default_run_bin(&ws).as_deref(), Some("beta"));

    let mut cfg = crate::config::Config::current();
    cfg.selection.bin = vec!["alpha".to_owned()];
    crate::with_config(std::sync::Arc::new(cfg), || {
        assert_eq!(default_run_bin(&ws), None);
    });
}

fn temp_test_workspace() -> (std::path::PathBuf, Workspace, Targets) {
    let seq = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let dir = std::env::temp_dir().join(format!("haste-cli-sel-{}-{seq}", std::process::id()));
    drop(std::fs::remove_dir_all(&dir));
    std::fs::create_dir_all(dir.join("src")).expect("mkdir src");
    std::fs::create_dir_all(dir.join("tests")).expect("mkdir tests");
    std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"demo\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .expect("write Cargo.toml");
    std::fs::write(
        dir.join("Cargo.lock"),
        "version = 4\n[[package]]\nname = \"demo\"\nversion = \"0.1.0\"\n",
    )
    .expect("write Cargo.lock");
    std::fs::write(dir.join("src/lib.rs"), "pub fn helper() -> u32 { 1 }\n").expect("lib.rs");
    std::fs::write(dir.join("src/main.rs"), "fn main() {}\n").expect("main.rs");
    std::fs::write(
        dir.join("tests/payments.rs"),
        "#[test]\nfn declined_payment() { let _: u32 = \"declined\"; }\n",
    )
    .expect("payments.rs");
    std::fs::write(dir.join("tests/receipts.rs"), "#[test]\nfn ok() {}\n").expect("receipts.rs");

    let host = cfg::Target::host();
    let targets = Targets {
        target: host.clone(),
        host,
    };
    let ws = graph::load(&dir, &targets, &crate::Selection::default()).expect("load ws");
    (dir, ws, targets)
}

#[test]
fn check_and_test_with_named_test_targets_preserve_all_targets_and_reject_unknown() {
    let (dir, ws, targets) = temp_test_workspace();

    let mut cfg = crate::config::Config::for_target(haste_codegen::Target::host());
    cfg.test_mode = true;
    cfg.selection.test = vec!["payments".to_owned()];
    crate::with_config(std::sync::Arc::new(cfg.clone()), || {
        validate_selected_targets(&ws).expect("payments target exists");
        let mut outline = crate::outline::new(&ws, &targets, &[]);
        outline.load_all().expect("load outline");
        let checked = crate::build::test_args::checked_member_crates(&ws, &outline, None);
        let checked_names: Vec<&str> = checked
            .iter()
            .map(|&k| outline.roots[k].name.as_str())
            .collect();
        assert!(checked_names.contains(&"payments"), "{checked_names:?}");
        assert!(!checked_names.contains(&"receipts"), "{checked_names:?}");
        assert!(
            !checked
                .iter()
                .any(|&k| outline.roots[k].role == outline::Role::Bin),
            "unrelated binary target should not be checked for --test payments: {checked_names:?}"
        );
        let fns = crate::build::test_args::check_fn_items(&ws, &outline, None);
        assert!(
            fns.iter().any(|item| {
                let file = &outline.crates[item.krate as usize].files[item.file as usize];
                outline.roots[item.krate as usize].name == "payments"
                    && file.items.items[item.item as usize]
                        .name
                        .is_some_and(|n| file.source.text(n) == "declined_payment")
            }),
            "expected declined_payment from tests/payments.rs to be checked"
        );
    });

    cfg.selection.test = vec!["payments".to_owned(), "receipts".to_owned()];
    crate::with_config(std::sync::Arc::new(cfg.clone()), || {
        validate_selected_targets(&ws).expect("both test targets exist");
        let mut outline = crate::outline::new(&ws, &targets, &[]);
        outline.load_all().expect("load outline");
        let sel = crate::selection();
        let selected: Vec<&str> = (0..outline.roots.len())
            .filter(|&k| outline.crates[k].env.test && test_target_selected(&outline, k, &sel))
            .map(|k| outline.roots[k].name.as_str())
            .collect();
        assert_eq!(selected, vec!["payments", "receipts"]);
    });

    cfg.selection.test = vec!["payments".to_owned(), "missing".to_owned()];
    crate::with_config(std::sync::Arc::new(cfg), || {
        let err = validate_selected_targets(&ws)
            .expect_err("missing test target must fail")
            .to_string();
        assert!(err.contains("no test target named 'missing'"), "{err}");
    });

    drop(std::fs::remove_dir_all(&dir));
}

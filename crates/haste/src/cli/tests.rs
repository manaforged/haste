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
        bin: Some("my-bin".to_owned()),
        ..crate::Selection::default()
    };
    let by_bin_under = crate::Selection {
        bin: Some("my_bin".to_owned()),
        ..crate::Selection::default()
    };
    assert!(test_target_selected(&outline, 0, &by_bin_hyphen));
    assert!(test_target_selected(&outline, 0, &by_bin_under));
    assert!(!test_target_selected(&outline, 1, &by_bin_hyphen));

    let by_test_hyphen = crate::Selection {
        test: Some("my-test".to_owned()),
        ..crate::Selection::default()
    };
    let by_test_under = crate::Selection {
        test: Some("my_test".to_owned()),
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
    cfg.selection.bin = Some("alpha".to_owned());
    crate::with_config(std::sync::Arc::new(cfg), || {
        assert_eq!(default_run_bin(&ws), None);
    });
}

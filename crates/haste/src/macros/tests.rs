use std::path::PathBuf;
use std::sync::Arc;

use rustc_hash::FxHashMap as HashMap;

use super::*;
use crate::names::collect::collect_module;
use crate::names::{Crate as NamesCrate, ModRef, Names, Symbols};
use crate::outline::scan::scan;
use crate::outline::tree::{Crate, Decl, Env, File, Files, Module, State};
use crate::outline::{Role, Root};

fn loaded_crate(path: &str, text: &str) -> Crate {
    let src = Source::parse(PathBuf::from(path), text.to_owned()).expect("parse source");
    let items = scan(&src).expect("scan source");
    Crate {
        files: vec![Arc::new(File::new(src, items, false))],
        modules: vec![Module {
            file: 0,
            item: None,
            parent: None,
            name: String::new(),
            dir: PathBuf::new(),
            parts: Vec::new(),
            vis: crate::outline::item::Vis::Pub,
            macro_use: false,
            state: State::Loaded,
            textual: None,
        }],
        env: Env::default(),
        ..Crate::default()
    }
}

fn eager_names(edition: &str, deps: Vec<(crate::names::Sym, usize)>) -> NamesCrate {
    let mut entry = NamesCrate::default();
    entry.edition = edition.to_owned();
    entry.edition_2015 = edition == "2015";
    entry.deps = deps;
    entry.eager = true;
    entry
}

#[test]
fn load_unsettles_cached_crate_when_pending_imports_asked() {
    let syms = Symbols::default();
    let macro_sym = syms.intern("hello");
    let dep_src = Source::parse(
        PathBuf::from("dep.rs"),
        "#[macro_export] macro_rules! hello { () => { 1 } }\n".to_owned(),
    )
    .expect("parse dep");
    let dep_items = scan(&dep_src).expect("scan dep");
    let dep_tree = Crate {
        files: vec![Arc::new(File::new(dep_src, dep_items, false))],
        modules: vec![Module {
            file: 0,
            item: None,
            parent: None,
            name: String::new(),
            dir: PathBuf::new(),
            parts: Vec::new(),
            vis: crate::outline::item::Vis::Pub,
            macro_use: false,
            state: State::Loaded,
            textual: None,
        }],
        env: Env::default(),
        ..Crate::default()
    };

    let reexport_src = Source::parse(PathBuf::from("reexport.rs"), "pub use dep::*;\n".to_owned())
        .expect("parse reexport");
    let reexport_items = scan(&reexport_src).expect("scan reexport");
    let reexport_tree = Crate {
        files: vec![Arc::new(File::new(reexport_src, reexport_items, false))],
        modules: vec![Module {
            file: 0,
            item: None,
            parent: None,
            name: String::new(),
            dir: PathBuf::new(),
            parts: Vec::new(),
            vis: crate::outline::item::Vis::Pub,
            macro_use: false,
            state: State::Loaded,
            textual: None,
        }],
        env: Env::default(),
        ..Crate::default()
    };
    let deps_list = vec![("dep".to_owned(), 0usize)];
    let outline = Outline {
        roots: vec![
            Root {
                unit: 0,
                name: "dep".to_owned(),
                path: PathBuf::from("dep.rs"),
                role: Role::Lib,
                edition: "2021".to_owned(),
                proc_macro: false,
            },
            Root {
                unit: 1,
                name: "reexport".to_owned(),
                path: PathBuf::from("reexport.rs"),
                role: Role::Lib,
                edition: "2021".to_owned(),
                proc_macro: false,
            },
        ],
        crates: vec![dep_tree, reexport_tree],
        deps: vec![Vec::new(), deps_list],
        cache: Files::default(),
        procs: HashMap::default(),
        pins: vec![None, None],
    };

    let mut dep_names = NamesCrate::default();
    dep_names.edition = "2021".to_owned();
    dep_names.scopes = vec![Default::default()];
    dep_names.collected = vec![Some(0)];
    dep_names.settled = true;
    let dep_ctx = crate::names::collect::Context {
        krate: 0,
        env: &outline.crates[0].env,
        deps: &[],
        symbols: &syms,
        children: &[Vec::new()],
    };
    let _ = collect_module(
        &outline.crates[0],
        0,
        &[(0, None)],
        &dep_ctx,
        &mut dep_names,
    );

    let mut reexport_names = NamesCrate::default();
    reexport_names.edition = "2021".to_owned();
    reexport_names.scopes = vec![Default::default()];
    reexport_names.deps = vec![(syms.intern("dep"), 0)];
    reexport_names.collected = vec![Some(0)];
    reexport_names.settled = true;
    let reexport_ctx = crate::names::collect::Context {
        krate: 1,
        env: &outline.crates[1].env,
        deps: &outline.deps[1],
        symbols: &syms,
        children: &[Vec::new()],
    };
    let pending = collect_module(
        &outline.crates[1],
        0,
        &[(0, None)],
        &reexport_ctx,
        &mut reexport_names,
    );
    reexport_names.add_pending(pending);

    let names = Names {
        crates: vec![dep_names, reexport_names],
        symbols: syms,
    };
    let mut compilation = Compilation::for_test(outline, names);
    let asks = Requests::default();
    asks.names.borrow_mut().push((
        ModRef {
            krate: 1,
            module: 0,
        },
        macro_sym,
    ));

    let answer = load(&mut compilation, asks).expect("macros::load");
    assert_eq!(answer, Answer::Names);
    assert!(!compilation.names.crates[1].settled);
    assert!(
        compilation.names.crates[1].scopes[0]
            .names
            .get(&macro_sym)
            .and_then(|ns| ns.macros)
            .is_some(),
        "reexported macro must resolve once cached crate is unsettled"
    );
}

#[test]
fn unresolved_import_does_not_reask_unsettled_crate() {
    let syms = Symbols::default();
    let macro_sym = syms.intern("crate_version");
    let root_tree = loaded_crate("build.rs", "use dep::crate_version;\nfn main() {}\n");
    let outline = Outline {
        roots: vec![Root {
            unit: 0,
            name: "app".to_owned(),
            path: PathBuf::from("build.rs"),
            role: Role::Script,
            edition: "2018".to_owned(),
            proc_macro: false,
        }],
        crates: vec![root_tree],
        deps: vec![Vec::new()],
        cache: Files::default(),
        procs: HashMap::default(),
        pins: vec![None],
    };

    let mut root_names = eager_names("2018", Vec::new());
    root_names.scopes = vec![Default::default()];
    root_names.collected = vec![Some(0)];
    let ctx = crate::names::collect::Context {
        krate: 0,
        env: &outline.crates[0].env,
        deps: &[],
        symbols: &syms,
        children: &[Vec::new()],
    };
    let pending = collect_module(&outline.crates[0], 0, &[(0, None)], &ctx, &mut root_names);
    root_names.add_pending(pending);

    let names = Names {
        crates: vec![root_names],
        symbols: syms,
    };
    let asks = Requests::default();
    let paths = crate::names::path::Paths {
        demands: Some(&asks),
        ..names.paths(&outline, 0)
    };
    let found = crate::names::resolve::macro_path(
        &paths,
        &crate::names::path::Path {
            segments: &[macro_sym],
            absolute: false,
            dollar: None,
        },
        ModRef {
            krate: 0,
            module: 0,
        },
    );
    assert!(found.is_none());
    assert_eq!(asks.count(), 0);
}

#[test]
fn include_part_reexport_macro_resolves() {
    let dir = std::env::temp_dir().join(format!("haste_include_macro_{}", std::process::id()));
    let src_dir = dir.join("src");
    std::fs::create_dir_all(&src_dir).expect("create test dir");
    let build_path = dir.join("build.rs");
    let cli_path = src_dir.join("cli.rs");
    std::fs::write(
        &build_path,
        "include!(\"src/cli.rs\");\nfn main() { let _ = build_command(); }\n",
    )
    .expect("write build.rs");
    std::fs::write(
        &cli_path,
        "use clap::crate_version;\nfn build_command() -> &'static str { crate_version!() }\n",
    )
    .expect("write src/cli.rs");

    let builder_tree = loaded_crate(
        "clap_builder.rs",
        "#[macro_export]\nmacro_rules! crate_version { () => { \"1.0\" }; }\n",
    );
    let clap_tree = loaded_crate("clap.rs", "pub use clap_builder::*;\n");
    let app_tree = Crate {
        modules: vec![Module {
            file: usize::MAX,
            item: None,
            parent: None,
            name: String::new(),
            dir: dir.clone(),
            parts: Vec::new(),
            vis: crate::outline::item::Vis::Pub,
            macro_use: false,
            state: State::Declared(Decl::File {
                candidates: vec![build_path.clone()],
                by_attr: true,
            }),
            textual: None,
        }],
        env: Env::default(),
        ..Crate::default()
    };

    let mut outline = Outline {
        roots: vec![
            Root {
                unit: 0,
                name: "clap_builder".to_owned(),
                path: PathBuf::from("clap_builder.rs"),
                role: Role::Lib,
                edition: "2021".to_owned(),
                proc_macro: false,
            },
            Root {
                unit: 1,
                name: "clap".to_owned(),
                path: PathBuf::from("clap.rs"),
                role: Role::Lib,
                edition: "2021".to_owned(),
                proc_macro: false,
            },
            Root {
                unit: 2,
                name: "build_script_build".to_owned(),
                path: build_path,
                role: Role::Script,
                edition: "2018".to_owned(),
                proc_macro: false,
            },
        ],
        crates: vec![builder_tree, clap_tree, app_tree],
        deps: vec![
            Vec::new(),
            vec![("clap_builder".to_owned(), 0)],
            vec![("clap".to_owned(), 1)],
        ],
        cache: Files::default(),
        procs: HashMap::default(),
        pins: vec![None, None, None],
    };
    outline.load_all().expect("load outline");

    let syms = Symbols::default();
    let macro_sym = syms.intern("crate_version");
    let mut names = Names {
        crates: vec![
            eager_names("2021", Vec::new()),
            eager_names("2021", vec![(syms.intern("clap_builder"), 0)]),
            eager_names("2018", vec![(syms.intern("clap"), 1)]),
        ],
        symbols: syms,
    };
    names.resolve(&mut outline).expect("resolve names");
    let asks = Requests::default();
    let paths = crate::names::path::Paths {
        demands: Some(&asks),
        ..names.paths(&outline, 2)
    };
    let found = crate::names::resolve::macro_path(
        &paths,
        &crate::names::path::Path {
            segments: &[macro_sym],
            absolute: false,
            dollar: None,
        },
        ModRef {
            krate: 2,
            module: 0,
        },
    );
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(
        found,
        Some(ItemRef {
            krate: 0,
            file: 0,
            item: 0,
        })
    );
    assert_eq!(asks.count(), 0);
}

#[test]
fn derive_input_with_cfg_pruned_member_keeps_doc_commented_siblings() {
    let code = concat!(
        "#[derive(Debug, Serialize, Deserialize)]\n",
        "pub struct SyntaxSet {\n",
        "    syntaxes: Vec<u8>,\n",
        "    /// Stores the syntax index for every path.\n",
        "    path_syntaxes: Vec<(String, usize)>,\n",
        "    #[serde(skip)]\n",
        "    cache: u8,\n",
        "    /// Metadata docs.\n",
        "    #[cfg(feature = \"metadata\")]\n",
        "    pub(crate) metadata: u8,\n",
        "}\n",
        "#[derive(Debug, thiserror::Error)]\n",
        "#[non_exhaustive]\n",
        "pub enum LoadingError {\n",
        "    /// error reading a file\n",
        "    #[error(\"error reading a file: {0}\")]\n",
        "    Io(#[from] u8),\n",
        "    /// a metadata file was invalid\n",
        "    #[cfg(feature = \"metadata\")]\n",
        "    #[error(\"Failed to parse JSON\")]\n",
        "    ParseMetadata(u8),\n",
        "    /// A path given to a method was invalid.\n",
        "    /// Possibly UTF-8.\n",
        "    #[error(\"Invalid path\")]\n",
        "    BadPath,\n",
        "}\n",
    );
    let src = Source::parse(PathBuf::from("syntect.rs"), code.to_owned()).expect("parse source");
    let items = scan(&src).expect("scan source");
    let env = Env::default();
    let deps = Vec::new();
    let render = |idx: usize| -> String {
        super::derive_input_in(&src, &items.items[idx], &env, Some((deps.as_slice(), 0)))
    };

    let syntax_out = render(0);
    assert!(!syntax_out.contains("metadata"), "{syntax_out}");
    assert!(!syntax_out.contains(", ,"), "{syntax_out}");
    assert!(
        syntax_out.contains("path_syntaxes : Vec < (String , usize) >")
            || syntax_out.contains("path_syntaxes: Vec<(String, usize)>"),
        "{syntax_out}"
    );
    assert!(syntax_out.contains("cache: u8"), "{syntax_out}");
    Source::parse(PathBuf::from("out_struct.rs"), syntax_out.clone()).expect("valid struct tokens");

    let err_out = render(1);
    assert!(!err_out.contains("ParseMetadata"), "{err_out}");
    assert!(err_out.contains("Io(#[from] u8)"), "{err_out}");
    assert!(err_out.contains("BadPath"), "{err_out}");
    assert!(err_out.contains("error reading a file"), "{err_out}");
    Source::parse(PathBuf::from("out_enum.rs"), err_out.clone()).expect("valid enum tokens");
}

fn declared_file_crate(path: PathBuf, env: Env) -> Crate {
    Crate {
        modules: vec![Module {
            file: usize::MAX,
            item: None,
            parent: None,
            name: String::new(),
            dir: path.parent().unwrap_or(path.as_path()).to_path_buf(),
            parts: Vec::new(),
            vis: crate::outline::item::Vis::Pub,
            macro_use: false,
            state: State::Declared(Decl::File {
                candidates: vec![path],
                by_attr: true,
            }),
            textual: None,
        }],
        env,
        ..Crate::default()
    }
}

#[test]
fn renamed_prelude_test_and_shadowed_user_test_attr_resolve_by_def() {
    let seq = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let dir = std::env::temp_dir().join(format!("haste_test_attr_{}-{seq}", std::process::id()));
    drop(std::fs::remove_dir_all(&dir));
    std::fs::create_dir_all(&dir).expect("mkdir");

    let write = |name: &str, body: &str| {
        let p = dir.join(name);
        std::fs::write(&p, body).expect("write");
        p
    };
    let core_p = write(
        "core.rs",
        concat!(
            "pub macro test($item:item) {}\n",
            "pub macro bench($item:item) {}\n",
            "pub mod prelude { pub mod v1 { pub use crate::{bench, test}; } }\n",
        ),
    );
    let std_p = write(
        "std.rs",
        "pub mod prelude { pub mod v1 { pub use core::prelude::v1::{bench, test}; } }\n",
    );
    let proc_p = write(
        "my_macros.rs",
        "#[proc_macro_attribute]\npub fn my_attr() {}\n",
    );
    let app_src = concat!(
        "use std::prelude::v1::test as case;\n",
        "use my_macros::my_attr as test;\n",
        "#[case]\nfn hidden_case() {}\n",
        "#[test]\nfn user_attr_fn() {}\n",
        "#[cfg_attr(all(), case)]\nfn cfg_hidden_case() {}\n",
    );
    let app_non_test_p = write("app_non_test.rs", app_src);
    let app_test_p = write("app_test.rs", app_src);

    let sys_env = Env {
        release: true,
        fixed: true,
        known: true,
        ..Env::default()
    };
    let non_test_env = Env {
        test: false,
        known: true,
        ..Env::default()
    };
    let test_env = Env {
        test: true,
        known: true,
        ..Env::default()
    };
    let mut outline = Outline {
        roots: vec![
            Root {
                unit: 0,
                name: "core".to_owned(),
                path: core_p.clone(),
                role: Role::Lib,
                edition: "2021".to_owned(),
                proc_macro: false,
            },
            Root {
                unit: 1,
                name: "std".to_owned(),
                path: std_p.clone(),
                role: Role::Lib,
                edition: "2021".to_owned(),
                proc_macro: false,
            },
            Root {
                unit: 2,
                name: "my_macros".to_owned(),
                path: proc_p.clone(),
                role: Role::Lib,
                edition: "2021".to_owned(),
                proc_macro: true,
            },
            Root {
                unit: 3,
                name: "app_non_test".to_owned(),
                path: app_non_test_p.clone(),
                role: Role::Bin,
                edition: "2021".to_owned(),
                proc_macro: false,
            },
            Root {
                unit: 4,
                name: "app_test".to_owned(),
                path: app_test_p.clone(),
                role: Role::Test,
                edition: "2021".to_owned(),
                proc_macro: false,
            },
        ],
        crates: vec![
            declared_file_crate(
                core_p,
                Env {
                    release: true,
                    fixed: true,
                    known: true,
                    ..Env::default()
                },
            ),
            declared_file_crate(std_p, sys_env),
            declared_file_crate(proc_p, Env::default()),
            declared_file_crate(app_non_test_p, non_test_env),
            declared_file_crate(app_test_p, test_env),
        ],
        deps: vec![
            Vec::new(),
            vec![("core".to_owned(), 0)],
            Vec::new(),
            vec![
                ("core".to_owned(), 0),
                ("std".to_owned(), 1),
                ("my_macros".to_owned(), 2),
            ],
            vec![
                ("core".to_owned(), 0),
                ("std".to_owned(), 1),
                ("my_macros".to_owned(), 2),
            ],
        ],
        cache: Files::default(),
        procs: HashMap::default(),
        pins: vec![None; 5],
    };
    outline.load_all().expect("load outline");

    let syms = Symbols::default();
    let deps_for = |k: usize| {
        outline.deps[k]
            .iter()
            .map(|(n, at)| (syms.intern(n), *at))
            .collect()
    };
    let names = Names {
        crates: vec![
            eager_names("2021", deps_for(0)),
            eager_names("2021", deps_for(1)),
            eager_names("2021", deps_for(2)),
            eager_names("2021", deps_for(3)),
            eager_names("2021", deps_for(4)),
        ],
        symbols: syms,
    };
    let mut comp = Compilation::for_test(outline, names);
    comp.names.resolve(&mut comp.outline).expect("resolve");
    let (bound, _) = comp.names.bind_procs(&mut comp.outline);

    assert_eq!(
        bound.len(),
        2,
        "only shadowed #[test] -> my_attr should bind"
    );
    assert!(bound.iter().all(|b| {
        b.run.krate == 2 && b.kind == ProcKind::Attribute && matches!(b.krate, 3 | 4)
    }));

    let item_named = |krate: usize, target: &str| -> (u32, u32) {
        let f = &comp.outline.crates[krate].files[0];
        let idx = f
            .items
            .items
            .iter()
            .position(|it| it.name.is_some_and(|n| f.source.text(n) == target))
            .expect("item exists");
        (0, idx as u32)
    };

    let non_test = &comp.outline.crates[3];
    assert!(non_test.consumed.contains(&item_named(3, "hidden_case")));
    assert!(
        non_test
            .consumed
            .contains(&item_named(3, "cfg_hidden_case"))
    );
    assert!(!non_test.consumed.contains(&item_named(3, "user_attr_fn")));
    assert!(non_test.resolved_tests.is_empty());

    let with_test = &comp.outline.crates[4];
    assert!(!with_test.consumed.contains(&item_named(4, "hidden_case")));
    assert!(
        with_test
            .resolved_tests
            .contains(&item_named(4, "hidden_case"))
    );
    assert!(
        with_test
            .resolved_tests
            .contains(&item_named(4, "cfg_hidden_case"))
    );
    assert!(
        !with_test
            .resolved_tests
            .contains(&item_named(4, "user_attr_fn"))
    );
    assert_eq!(crate::cli::outline_test_count(&comp.outline), 2);

    comp.names.reset(3);
    comp.names.resolve(&mut comp.outline).expect("re-resolve");
    let hidden_sym = comp.names.symbols.intern("hidden_case");
    let cfg_sym = comp.names.symbols.intern("cfg_hidden_case");
    let user_sym = comp.names.symbols.intern("user_attr_fn");
    let scopes = &comp.names.crates[3].scopes[0].names;
    assert!(!scopes.contains_key(&hidden_sym));
    assert!(!scopes.contains_key(&cfg_sym));
    assert!(scopes.contains_key(&user_sym));

    drop(std::fs::remove_dir_all(&dir));
}

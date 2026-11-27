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

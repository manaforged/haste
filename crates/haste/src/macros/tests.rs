use std::path::PathBuf;
use std::sync::Arc;

use rustc_hash::FxHashMap as HashMap;

use super::*;
use crate::names::collect::collect_module;
use crate::names::{Crate as NamesCrate, ModRef, Names, Symbols};
use crate::outline::scan::scan;
use crate::outline::tree::{Crate, Env, File, Files, Module, State};
use crate::outline::{Role, Root};

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

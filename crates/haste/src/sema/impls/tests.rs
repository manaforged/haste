use std::path::PathBuf;

use rustc_hash::FxHashMap as HashMap;

use crate::names::{Crate as NamesCrate, ItemRef, Names, Symbols};
use crate::outline::tree::{Crate, Decl, Files, Module, State};
use crate::outline::{Outline, Role, Root};
use crate::sema::{Answer, Cache, Context};

pub(crate) fn declared_crate(path: PathBuf) -> Crate {
    let dir = path.parent().map_or_else(PathBuf::new, PathBuf::from);
    Crate {
        modules: vec![Module {
            file: usize::MAX,
            item: None,
            parent: None,
            name: String::new(),
            dir,
            parts: Vec::new(),
            vis: crate::outline::item::Vis::Pub,
            macro_use: false,
            state: State::Declared(Decl::File {
                candidates: vec![path],
                by_attr: true,
            }),
            textual: None,
        }],
        ..Crate::default()
    }
}

#[test]
fn lazy_submodule_body_local_impl_survives_names_reload() {
    let dir = std::env::temp_dir().join(format!("haste_lazy_body_impl_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create test dir");
    let dep_path = dir.join("lib.rs");
    let col_path = dir.join("collector.rs");
    let app_path = dir.join("app.rs");
    std::fs::write(
        &dep_path,
        concat!(
            "pub mod collector;\n",
            "#[macro_export] macro_rules! report { () => { $crate::Report }; }\n",
            "pub struct Report;\n",
            "impl Report {\n",
            "    pub fn info<C: collector::Collector>(_r: Report, _c: C) -> u32 { 7 }\n",
            "}\n",
        ),
    )
    .expect("write dep.rs");
    std::fs::write(
        &col_path,
        "pub trait Collector { fn ping(&self) -> u32; }\n",
    )
    .expect("write collector.rs");
    std::fs::write(
        &app_path,
        concat!(
            "fn invoke_bugreport() -> u32 {\n",
            "    use dep::{report, collector::*};\n",
            "    struct ColorSchemeCollector;\n",
            "    impl Collector for ColorSchemeCollector { fn ping(&self) -> u32 { 7 } }\n",
            "    dep::Report::info(report!(), ColorSchemeCollector)\n",
            "}\n",
        ),
    )
    .expect("write app.rs");

    let app_deps = vec![("dep".to_owned(), 0usize)];
    let mut outline = Outline {
        roots: vec![
            Root {
                unit: 0,
                name: "dep".to_owned(),
                path: dep_path.clone(),
                role: Role::Lib,
                edition: "2021".to_owned(),
                proc_macro: false,
            },
            Root {
                unit: 1,
                name: "app".to_owned(),
                path: app_path.clone(),
                role: Role::Bin,
                edition: "2021".to_owned(),
                proc_macro: false,
            },
        ],
        crates: vec![declared_crate(dep_path), declared_crate(app_path)],
        deps: vec![Vec::new(), app_deps],
        cache: Files::default(),
        procs: HashMap::default(),
        pins: vec![None, None],
    };
    outline.load_all().expect("load outline");

    let syms = Symbols::default();
    let dep_sym = syms.intern("dep");
    let mut dep_names = NamesCrate::default();
    dep_names.edition = "2021".to_owned();
    let mut app_names = NamesCrate::default();
    app_names.edition = "2021".to_owned();
    app_names.deps = vec![(dep_sym, 0)];

    let mut names = Names {
        crates: vec![dep_names, app_names],
        symbols: syms,
    };
    names.resolve(&mut outline).expect("resolve names");

    let mut cx = Context::new(&outline, &names, Cache::default());
    let dummy_trait = ItemRef {
        krate: 0,
        file: 0,
        item: 0,
    };
    let _ = cx.trait_impls(dummy_trait, None, &[1]);
    let (mut store, asks) = cx.release();
    assert!(
        names.load(&mut outline, asks).expect("load lazy collector"),
        "indexing body-local impl before collector loads must ask for dep::collector"
    );
    store.lookups.reconcile(&outline, Some(Answer::Names));

    let mut cx2 = Context::new(&outline, &names, store);
    let collector_trait = outline.crates[0]
        .files
        .iter()
        .enumerate()
        .find_map(|(f, file)| {
            file.items.items.iter().enumerate().find_map(|(i, it)| {
                it.name
                    .is_some_and(|n| file.source.text(n) == "Collector")
                    .then_some(ItemRef {
                        krate: 0,
                        file: f as u32,
                        item: i as u32,
                    })
            })
        })
        .expect("Collector trait");
    let found = cx2.trait_impls(collector_trait, None, &[1]);
    assert!(
        found
            .iter()
            .any(|b| matches!(b, crate::sema::impls::ImplRef::Body(_))),
        "body-local impl for dep::collector::Collector must be indexed after Answer::Names: {found:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

use std::path::PathBuf;

use rustc_hash::FxHashMap as HashMap;

use crate::names::{Crate as NamesCrate, ItemRef, Names, Symbols};
use crate::outline::tree::Files;
use crate::outline::{Outline, Role, Root};
use crate::sema::impls::tests::declared_crate;
use crate::sema::infer::infer;
use crate::sema::substs::Substs;
use crate::sema::{Cache, Context};

struct TestCrate<'a> {
    name: &'a str,
    role: Role,
    sysroot: bool,
    deps: &'a [usize],
    src: &'a str,
}

fn load_fixture(tag: &str, specs: &[TestCrate<'_>]) -> (PathBuf, Outline, Names) {
    let dir = std::env::temp_dir().join(format!("haste_select_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create test dir");

    let mut roots = Vec::with_capacity(specs.len());
    let mut crates = Vec::with_capacity(specs.len());
    let mut deps = Vec::with_capacity(specs.len());
    let syms = Symbols::default();
    let mut name_crates = Vec::with_capacity(specs.len());

    for (unit, spec) in specs.iter().enumerate() {
        let path = dir.join(format!("{}.rs", spec.name));
        std::fs::write(&path, spec.src).expect("write test crate source");
        let unit_deps: Vec<(String, usize)> = spec
            .deps
            .iter()
            .map(|&dep| (specs[dep].name.to_owned(), dep))
            .collect();
        roots.push(Root {
            unit,
            name: spec.name.to_owned(),
            path: path.clone(),
            role: spec.role,
            edition: "2021".to_owned(),
            proc_macro: false,
        });
        let mut parsed = declared_crate(path);
        if spec.sysroot {
            parsed.env.release = true;
            parsed.env.fixed = true;
        }
        crates.push(parsed);
        deps.push(unit_deps);

        let mut nc = NamesCrate::default();
        nc.edition = "2021".to_owned();
        nc.eager = true;
        nc.deps = spec
            .deps
            .iter()
            .map(|&dep| (syms.intern(specs[dep].name), dep))
            .collect();
        name_crates.push(nc);
    }

    let pins = vec![None; specs.len()];
    let mut outline = Outline {
        roots,
        crates,
        deps,
        cache: Files::default(),
        procs: HashMap::default(),
        pins,
    };
    outline.load_all().expect("load outline");

    let mut names = Names {
        crates: name_crates,
        symbols: syms,
    };
    names.resolve(&mut outline).expect("resolve names");

    (dir, outline, names)
}

fn find_item(outline: &Outline, krate: usize, name: &str) -> ItemRef {
    outline.crates[krate].files[0]
        .items
        .items
        .iter()
        .enumerate()
        .find_map(|(i, it)| {
            it.name
                .is_some_and(|n| outline.crates[krate].files[0].source.text(n) == name)
                .then_some(ItemRef {
                    krate: krate as u32,
                    file: 0,
                    item: i as u32,
                })
        })
        .expect("find item by name")
}

#[test]
fn body_local_struct_impl_satisfies_trait() {
    let dep_src = concat!(
        "pub mod collector {\n",
        "    pub trait Collector {\n",
        "        fn ping(&self) -> u32;\n",
        "    }\n",
        "}\n",
        "pub struct Report;\n",
        "impl Report {\n",
        "    pub fn info<C: collector::Collector>(_r: Report, _c: C) -> u32 {\n",
        "        7\n",
        "    }\n",
        "}\n",
    );
    let app_src = concat!(
        "fn invoke_report() -> u32 {\n",
        "    use dep::{Report, collector::*};\n",
        "    struct LocalCollector;\n",
        "    impl Collector for LocalCollector {\n",
        "        fn ping(&self) -> u32 {\n",
        "            7\n",
        "        }\n",
        "    }\n",
        "    Report::info(Report, LocalCollector)\n",
        "}\n",
    );
    let (dir, outline, names) = load_fixture(
        "body_local",
        &[
            TestCrate {
                name: "dep",
                role: Role::Lib,
                sysroot: false,
                deps: &[],
                src: dep_src,
            },
            TestCrate {
                name: "app",
                role: Role::Bin,
                sysroot: false,
                deps: &[0],
                src: app_src,
            },
        ],
    );

    let mut cx = Context::new(&outline, &names, Cache::default());
    let collector_trait = find_item(&outline, 0, "Collector");
    let found = cx.trait_impls(collector_trait, None, &[1]);
    assert!(
        found
            .iter()
            .any(|b| matches!(b, crate::sema::impls::ImplRef::Body(_))),
        "body-local impl for Collector must be indexed: {found:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn try_from_across_external_crates_and_open_closure_params() {
    let core_src = concat!(
        "#![no_core]\n",
        "#[lang = \"fn_once\"]\n",
        "pub trait FnOnce<Args> {\n",
        "    type Output;\n",
        "    fn call_once(self, args: Args) -> Self::Output;\n",
        "}\n",
        "#[lang = \"deref\"]\n",
        "pub trait Deref {\n",
        "    type Target;\n",
        "    fn deref(&self) -> &Self::Target;\n",
        "}\n",
        "#[lang = \"drop\"]\n",
        "pub trait Drop {\n",
        "    fn drop(&mut self);\n",
        "}\n",
        "pub trait From<T> {\n",
        "    fn from(value: T) -> Self;\n",
        "}\n",
        "pub trait Into<T> {\n",
        "    fn into(self) -> T;\n",
        "}\n",
        "impl<T, U> Into<U> for T\n",
        "where\n",
        "    U: From<T>,\n",
        "{\n",
        "    fn into(self) -> U {\n",
        "        U::from(self)\n",
        "    }\n",
        "}\n",
        "pub trait TryFrom<T> {\n",
        "    type Error;\n",
        "    fn try_from(value: T) -> Outcome<Self, Self::Error>;\n",
        "}\n",
        "impl<T, U> TryFrom<U> for T\n",
        "where\n",
        "    U: Into<T>,\n",
        "{\n",
        "    type Error = ();\n",
        "    fn try_from(value: U) -> Outcome<Self, Self::Error> {\n",
        "        Outcome::Ok(U::into(value))\n",
        "    }\n",
        "}\n",
        "pub enum Outcome<T, E> {\n",
        "    Ok(T),\n",
        "    Err(E),\n",
        "}\n",
        "impl<T, E> Outcome<T, E> {\n",
        "    pub fn and_then<U, F: FnOnce(T) -> Outcome<U, E>>(self, op: F) -> Outcome<U, E> {\n",
        "        match self {\n",
        "            Outcome::Ok(t) => op(t),\n",
        "            Outcome::Err(e) => Outcome::Err(e),\n",
        "        }\n",
        "    }\n",
        "    pub fn map_err<O, F: FnOnce(E) -> O>(self, op: F) -> Outcome<T, O> {\n",
        "        match self {\n",
        "            Outcome::Ok(t) => Outcome::Ok(t),\n",
        "            Outcome::Err(e) => Outcome::Err(op(e)),\n",
        "        }\n",
        "    }\n",
        "}\n",
    );
    let std_src = concat!("pub mod time {\n", "    pub struct SystemTime;\n", "}\n",);
    let jiff_src = concat!(
        "use core::{From, Outcome, TryFrom};\n",
        "pub struct Timestamp;\n",
        "pub struct Error;\n",
        "impl TryFrom<Timestamp> for std::time::SystemTime {\n",
        "    type Error = Error;\n",
        "    fn try_from(_: Timestamp) -> Outcome<Self, Self::Error> {\n",
        "        Outcome::Ok(std::time::SystemTime)\n",
        "    }\n",
        "}\n",
        "pub struct FromOnly;\n",
        "impl From<FromOnly> for std::time::SystemTime {\n",
        "    fn from(_: FromOnly) -> Self {\n",
        "        std::time::SystemTime\n",
        "    }\n",
        "}\n",
        "pub struct Missing;\n",
    );
    let fd_src = concat!(
        "use core::{Outcome, TryFrom};\n",
        "use jiff::{Error, FromOnly, Missing, Timestamp};\n",
        "use std::time::SystemTime;\n",
        "fn parse_timestamp() -> Outcome<Timestamp, ()> {\n",
        "    Outcome::Ok(Timestamp)\n",
        "}\n",
        "fn parse_from_only() -> Outcome<FromOnly, ()> {\n",
        "    Outcome::Ok(FromOnly)\n",
        "}\n",
        "pub fn try_from_direct_jiff_in_and_then() -> Outcome<SystemTime, ()> {\n",
        "    parse_timestamp().and_then(|t| SystemTime::try_from(t).map_err(|_| ()))\n",
        "}\n",
        "pub fn try_from_explicit_param_in_closure() -> Outcome<SystemTime, ()> {\n",
        "    let convert = |t: Timestamp| SystemTime::try_from(t).map_err(|_| ());\n",
        "    convert(Timestamp)\n",
        "}\n",
        "pub fn try_from_via_from_blanket_in_and_then() -> Outcome<SystemTime, ()> {\n",
        "    parse_from_only().and_then(|t| SystemTime::try_from(t).map_err(|_| ()))\n",
        "}\n",
        "pub fn try_from_closure_deferred_direct() -> Outcome<SystemTime, Error> {\n",
        "    let convert = |t| SystemTime::try_from(t);\n",
        "    convert(Timestamp)\n",
        "}\n",
        "pub fn try_from_closure_deferred_from() -> Outcome<SystemTime, ()> {\n",
        "    let convert = |t| SystemTime::try_from(t);\n",
        "    convert(FromOnly)\n",
        "}\n",
        "pub fn missing_try_from_errors() -> Outcome<SystemTime, ()> {\n",
        "    SystemTime::try_from(Missing).map_err(|_| ())\n",
        "}\n",
    );
    let (dir, outline, names) = load_fixture(
        "external_try_from",
        &[
            TestCrate {
                name: "core",
                role: Role::Lib,
                sysroot: true,
                deps: &[],
                src: core_src,
            },
            TestCrate {
                name: "std",
                role: Role::Lib,
                sysroot: true,
                deps: &[0],
                src: std_src,
            },
            TestCrate {
                name: "jiff",
                role: Role::Lib,
                sysroot: false,
                deps: &[0, 1],
                src: jiff_src,
            },
            TestCrate {
                name: "fd",
                role: Role::Bin,
                sysroot: false,
                deps: &[0, 1, 2],
                src: fd_src,
            },
        ],
    );

    let mut cx = Context::new(&outline, &names, Cache::default());
    for name in [
        "try_from_direct_jiff_in_and_then",
        "try_from_explicit_param_in_closure",
        "try_from_via_from_blanket_in_and_then",
        "try_from_closure_deferred_direct",
        "try_from_closure_deferred_from",
    ] {
        let item = find_item(&outline, 3, name);
        let res = infer(&mut cx, item, Substs::EMPTY);
        assert!(res.is_ok(), "{name} should infer without error: {res:?}");
    }
    let bad = find_item(&outline, 3, "missing_try_from_errors");
    let bad_res = infer(&mut cx, bad, Substs::EMPTY);
    assert!(
        bad_res.is_err(),
        "missing_try_from_errors must fail trait selection"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

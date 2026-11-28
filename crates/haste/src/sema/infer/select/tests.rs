use std::path::PathBuf;

use rustc_hash::FxHashMap as HashMap;

use super::super::infer;
use super::super::poly::build_poly_summary;
use crate::names::{Crate as NamesCrate, ItemRef, Names, Symbols};
use crate::outline::tree::Files;
use crate::outline::{Outline, Role, Root};
use crate::sema::impls::tests::declared_crate;
use crate::sema::substs::Substs;
use crate::sema::summary::SummaryKind;
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

#[test]
fn open_var_assoc_binding_resolves_after_field_unify() {
    let core_src = concat!(
        "#![no_core]\n",
        "#[lang = \"sized\"]\n",
        "pub trait Sized {}\n",
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
        "impl<T> From<T> for T {\n",
        "    fn from(value: T) -> Self {\n",
        "        value\n",
        "    }\n",
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
        "pub enum Result<T, E> {\n",
        "    Ok(T),\n",
        "    Err(E),\n",
        "}\n",
    );
    let app_src = concat!(
        "use core::{Into, Result};\n",
        "pub trait Stream {\n",
        "    type Item;\n",
        "}\n",
        "pub trait HttpBody {\n",
        "    type Data;\n",
        "    type Error;\n",
        "}\n",
        "pub struct Event;\n",
        "pub struct Bytes;\n",
        "pub struct BoxError;\n",
        "pub struct SyncWrapper<T>(pub T);\n",
        "impl<T> SyncWrapper<T> {\n",
        "    pub fn new(v: T) -> Self {\n",
        "        Self(v)\n",
        "    }\n",
        "}\n",
        "pub struct Body;\n",
        "impl Body {\n",
        "    pub fn new<B>(_: B) -> Self\n",
        "    where\n",
        "        B: HttpBody,\n",
        "        B::Error: Into<BoxError>,\n",
        "    {\n",
        "        Body\n",
        "    }\n",
        "}\n",
        "pub struct SseBody<S> {\n",
        "    pub event_stream: SyncWrapper<S>,\n",
        "}\n",
        "impl<S, E> HttpBody for SseBody<S>\n",
        "where\n",
        "    S: Stream<Item = Result<Event, E>>,\n",
        "{\n",
        "    type Data = Bytes;\n",
        "    type Error = E;\n",
        "}\n",
        "pub struct Sse<S> {\n",
        "    pub stream: S,\n",
        "}\n",
        "pub fn sse_into_body<S, E>(sse: Sse<S>) -> Body\n",
        "where\n",
        "    S: Stream<Item = Result<Event, E>>,\n",
        "    E: Into<BoxError>,\n",
        "{\n",
        "    Body::new(SseBody {\n",
        "        event_stream: SyncWrapper::new(sse.stream),\n",
        "    })\n",
        "}\n",
        "pub fn sse_missing_into_errors<S, E>(sse: Sse<S>) -> Body\n",
        "where\n",
        "    S: Stream<Item = Result<Event, E>>,\n",
        "{\n",
        "    Body::new(SseBody {\n",
        "        event_stream: SyncWrapper::new(sse.stream),\n",
        "    })\n",
        "}\n",
    );
    let (dir, outline, names) = load_fixture(
        "open_var_assoc",
        &[
            TestCrate {
                name: "core",
                role: Role::Lib,
                sysroot: true,
                deps: &[],
                src: core_src,
            },
            TestCrate {
                name: "app",
                role: Role::Lib,
                sysroot: false,
                deps: &[0],
                src: app_src,
            },
        ],
    );
    let mut cx = Context::new(&outline, &names, Cache::default());
    let good = find_item(&outline, 1, "sse_into_body");
    let good_res = build_poly_summary(&mut cx, good, 2);
    assert!(
        !matches!(good_res, SummaryKind::Error(_)),
        "sse_into_body should infer cleanly: {good_res:?}"
    );
    let bad = find_item(&outline, 1, "sse_missing_into_errors");
    let bad_res = build_poly_summary(&mut cx, bad, 2);
    assert!(
        matches!(bad_res, SummaryKind::Error(_)),
        "sse_missing_into_errors must reject missing E: Into<BoxError>"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn pending_opaque_tuple_prefers_layer_over_unrelated_handler_trait() {
    let core_src = concat!(
        "#![no_core]\n",
        "#[lang = \"sized\"]\n",
        "pub trait Sized {}\n",
        "#[lang = \"deref\"]\n",
        "pub trait Deref {\n",
        "    type Target;\n",
        "    fn deref(&self) -> &Self::Target;\n",
        "}\n",
        "#[lang = \"drop\"]\n",
        "pub trait Drop {\n",
        "    fn drop(&mut self);\n",
        "}\n",
        "#[lang = \"clone\"]\n",
        "pub trait Clone {\n",
        "    fn clone(&self) -> Self;\n",
        "}\n",
        "impl<A: Clone, B: Clone> Clone for (A, B) {\n",
        "    fn clone(&self) -> Self {\n",
        "        (self.0.clone(), self.1.clone())\n",
        "    }\n",
        "}\n",
    );
    let app_src = concat!(
        "use core::Clone;\n",
        "pub trait Layer<S> {\n",
        "    type Service;\n",
        "    fn layer(&self, inner: S) -> Self::Service;\n",
        "}\n",
        "impl<A, B, S> Layer<S> for (A, B)\n",
        "where\n",
        "    A: Layer<B::Service>,\n",
        "    B: Layer<S>,\n",
        "{\n",
        "    type Service = A::Service;\n",
        "    fn layer(&self, inner: S) -> Self::Service {\n",
        "        self.0.layer(self.1.layer(inner))\n",
        "    }\n",
        "}\n",
        "pub trait Service {}\n",
        "pub trait IntoResponse {}\n",
        "pub trait IntoResponseParts {}\n",
        "impl<T1, R> IntoResponse for (T1, R)\n",
        "where\n",
        "    T1: IntoResponseParts,\n",
        "    R: IntoResponse,\n",
        "{\n",
        "}\n",
        "pub struct IntoResponseHandler;\n",
        "pub struct Layered<L, H, T, S>(L, H, T, S);\n",
        "pub trait Handler<T, S> {\n",
        "    fn layer<L>(self, layer: L) -> Layered<L, Self, T, S>\n",
        "    where\n",
        "        Self: Sized;\n",
        "}\n",
        "impl<T, S> Handler<IntoResponseHandler, S> for T\n",
        "where\n",
        "    T: IntoResponse + Clone,\n",
        "{\n",
        "    fn layer<L>(self, _layer: L) -> Layered<L, Self, IntoResponseHandler, S> {\n",
        "        loop {}\n",
        "    }\n",
        "}\n",
        "pub struct Passthrough;\n",
        "impl Clone for Passthrough {\n",
        "    fn clone(&self) -> Self {\n",
        "        Passthrough\n",
        "    }\n",
        "}\n",
        "impl<S> Layer<S> for Passthrough {\n",
        "    type Service = S;\n",
        "    fn layer(&self, inner: S) -> S {\n",
        "        inner\n",
        "    }\n",
        "}\n",
        "pub fn make_layer<S>() -> impl Layer<S, Service = S> + Clone {\n",
        "    Passthrough\n",
        "}\n",
        "pub struct Route;\n",
        "impl Route {\n",
        "    pub fn new<T: Service>(_: T) -> Self {\n",
        "        Route\n",
        "    }\n",
        "}\n",
        "pub fn nest_service<S: Service>(svc: S) -> Route {\n",
        "    let layer = (make_layer(), make_layer());\n",
        "    Route::new(layer.layer(svc))\n",
        "}\n",
        "pub struct DummySvc;\n",
        "impl Service for DummySvc {}\n",
        "fn require_into_response<T: IntoResponse>(_: T) {}\n",
        "pub fn opaque_tuple_not_into_response() {\n",
        "    let layer = (make_layer::<DummySvc>(), make_layer::<DummySvc>());\n",
        "    require_into_response(layer);\n",
        "}\n",
    );
    let (dir, outline, names) = load_fixture(
        "opaque_tuple_layer",
        &[
            TestCrate {
                name: "core",
                role: Role::Lib,
                sysroot: true,
                deps: &[],
                src: core_src,
            },
            TestCrate {
                name: "app",
                role: Role::Lib,
                sysroot: false,
                deps: &[0],
                src: app_src,
            },
        ],
    );
    let mut cx = Context::new(&outline, &names, Cache::default());
    let good = find_item(&outline, 1, "nest_service");
    let good_res = build_poly_summary(&mut cx, good, 1);
    assert!(
        !matches!(good_res, SummaryKind::Error(_)),
        "nest_service should pick Layer::layer over Handler::layer: {good_res:?}"
    );
    let bad = find_item(&outline, 1, "opaque_tuple_not_into_response");
    let bad_res = infer(&mut cx, bad, Substs::EMPTY);
    assert!(
        bad_res.is_err(),
        "pending opaque tuple must not satisfy unrelated IntoResponse trait"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

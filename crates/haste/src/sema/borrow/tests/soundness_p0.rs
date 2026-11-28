use std::path::PathBuf;

use rustc_hash::FxHashMap as HashMap;

use crate::names::{Crate as NamesCrate, ItemRef, Names, Symbols};
use crate::outline::tree::Files;
use crate::outline::{Outline, Role, Root};
use crate::sema::impls::tests::declared_crate;
use crate::sema::substs::Substs;
use crate::sema::{Cache, Context};

use super::{find_item_in, test_outline};

pub(super) const MIN_CORE_TRAITS: &str = concat!(
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
);

pub(super) fn load_borrow_fixture(
    tag: &str,
    edition: &str,
    core_src: &str,
    app_src: &str,
) -> (PathBuf, Outline, Names) {
    let dir = std::env::temp_dir().join(format!("haste_borrow_p0_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create test dir");

    let syms = Symbols::default();
    let core_path = dir.join("core.rs");
    let app_path = dir.join("app.rs");
    std::fs::write(&core_path, core_src).expect("write core source");
    std::fs::write(&app_path, app_src).expect("write app source");

    let mut core_parsed = declared_crate(core_path.clone());
    core_parsed.env.release = true;
    core_parsed.env.fixed = true;
    let app_parsed = declared_crate(app_path.clone());

    let mut core_nc = NamesCrate::default();
    core_nc.edition = edition.to_owned();
    core_nc.eager = true;

    let mut app_nc = NamesCrate::default();
    app_nc.edition = edition.to_owned();
    app_nc.eager = true;
    app_nc.deps.push((syms.intern("core"), 0));

    let mut outline = Outline {
        roots: vec![
            Root {
                unit: 0,
                name: "core".to_owned(),
                path: core_path,
                role: Role::Lib,
                edition: edition.to_owned(),
                proc_macro: false,
            },
            Root {
                unit: 1,
                name: "app".to_owned(),
                path: app_path,
                role: Role::Lib,
                edition: edition.to_owned(),
                proc_macro: false,
            },
        ],
        crates: vec![core_parsed, app_parsed],
        deps: vec![Vec::new(), vec![("core".to_owned(), 0)]],
        cache: Files::default(),
        procs: HashMap::default(),
        pins: vec![None, None],
    };
    outline.load_all().expect("load outline");

    let mut names = Names {
        crates: vec![core_nc, app_nc],
        symbols: syms,
    };
    names.resolve(&mut outline).expect("resolve names");
    (dir, outline, names)
}

pub(super) fn find_app_item(outline: &Outline, name: &str) -> ItemRef {
    outline.crates[1].files[0]
        .items
        .items
        .iter()
        .enumerate()
        .find_map(|(i, it)| {
            it.name
                .is_some_and(|n| outline.crates[1].files[0].source.text(n) == name)
                .then_some(ItemRef {
                    krate: 1,
                    file: 0,
                    item: i as u32,
                })
        })
        .expect("find item")
}

pub(super) fn run_borrow_on_item(
    cx: &mut Context<'_>,
    item: ItemRef,
    substs: Substs,
) -> anyhow::Result<()> {
    let typed = crate::sema::infer::infer(cx, item, substs)?;
    let sig = cx.signature(item).expect("sig");
    let body = cx.body(item).expect("body");
    let src = &cx.file(item).source;
    super::super::check(cx, item, (&body, &sig), src, &typed)
}

#[test]
fn escape_unbounded_type_param_alongside_named_lifetime_keeps_elided_lt() {
    use std::sync::Arc;

    use crate::sema::infer::{LazyTables, Local, Typed};

    let (outline, names) = test_outline(
        concat!(
            "pub fn bad<'a, T>(x: (&'a i32, T)) {}\n",
            "pub fn good<'a, T: 'a>(x: (&'a i32, T)) {}\n",
        ),
        "2021",
    );
    let mut cx = Context::new(&outline, &names, Cache::default());
    let unit = cx.types.unit();
    let typed = Typed {
        tables: LazyTables::Eager(Arc::default()),
        locals: vec![Local { ty: unit }],
        params: vec![0],
        ..Typed::default()
    };
    let src = &outline.crates[0].files[0].source;

    let bad_item = ItemRef {
        krate: 0,
        file: 0,
        item: 0,
    };
    let bad_sig = cx.signature(bad_item).expect("bad sig");
    let bad_info = super::super::calls::compute_param_lifetimes(
        &mut cx,
        bad_item,
        &bad_sig,
        src,
        &typed,
        (1, &[]),
    );
    assert!(bad_info.has_elided_lt[0]);
    assert_eq!(bad_info.named_lts[0], vec!["'a".to_owned()]);

    let good_item = ItemRef {
        krate: 0,
        file: 0,
        item: 1,
    };
    let good_sig = cx.signature(good_item).expect("good sig");
    let good_info = super::super::calls::compute_param_lifetimes(
        &mut cx,
        good_item,
        &good_sig,
        src,
        &typed,
        (1, &[]),
    );
    assert!(!good_info.has_elided_lt[0]);
    assert_eq!(good_info.named_lts[0], vec!["'a".to_owned()]);
}

#[test]
fn double_ref_to_temp_tracked_across_reborrow_assignment() {
    let core_src = MIN_CORE_TRAITS;
    let app_src = concat!(
        "pub struct Buf(u8);\n",
        "impl Buf {\n",
        "    pub fn new() -> Self { Buf(0) }\n",
        "}\n",
        "pub fn bad_double_ref_temp() {\n",
        "    let mut r: &Buf = &Buf(1);\n",
        "    {\n",
        "        let r2: &&Buf = &&Buf::new();\n",
        "        r = r2;\n",
        "    }\n",
        "    let _ = r;\n",
        "}\n",
        "pub fn ok_double_ref_static() {\n",
        "    static B: Buf = Buf(0);\n",
        "    let mut r: &Buf = &B;\n",
        "    {\n",
        "        let r2: &&Buf = &&B;\n",
        "        r = r2;\n",
        "    }\n",
        "    let _ = r;\n",
        "}\n",
    );
    let (dir, outline, names) = load_borrow_fixture("double_ref_temp", "2021", core_src, app_src);
    let mut cx = Context::new(&outline, &names, Cache::default());
    let bad_err = run_borrow_on_item(
        &mut cx,
        find_app_item(&outline, "bad_double_ref_temp"),
        Substs::EMPTY,
    )
    .expect_err("double-ref temp escape must fail");
    assert!(
        format!("{bad_err:#}").contains("temporary value dropped while borrowed"),
        "unexpected error: {bad_err:#}"
    );
    assert!(
        run_borrow_on_item(
            &mut cx,
            find_app_item(&outline, "ok_double_ref_static"),
            Substs::EMPTY
        )
        .is_ok()
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn edition_2024_typed_self_rpit_keeps_mut_borrow_inside_closure() {
    let core_src = format!(
        "{MIN_CORE_TRAITS}{}",
        concat!(
            "#[lang = \"copy\"]\n",
            "pub trait Copy {}\n",
            "impl Copy for u8 {}\n",
            "pub enum Poll<T> { Ready(T), Pending }\n",
            "pub struct Pin<P> { pub pointer: P }\n",
            "impl<P> Pin<P> {\n",
            "    pub fn new(pointer: P) -> Self { Pin { pointer } }\n",
            "}\n",
        )
    );
    let app_src = concat!(
        "use core::{Copy, Pin, Poll};\n",
        "pub struct Cell(u8);\n",
        "pub trait Lock {\n",
        "    fn lock(self: Pin<&mut Self>) -> impl Copy;\n",
        "    fn poll_val(self: Pin<&mut Self>) -> Poll<u8>;\n",
        "}\n",
        "impl Lock for Cell {\n",
        "    fn lock(self: Pin<&mut Self>) -> impl Copy { 0u8 }\n",
        "    fn poll_val(self: Pin<&mut Self>) -> Poll<u8> { Poll::Ready(0) }\n",
        "}\n",
        "pub fn bad_closure_rpit_alias() {\n",
        "    let mut c = Cell(0);\n",
        "    let f = || {\n",
        "        let g = Pin::new(&mut c).lock();\n",
        "        let _p = Pin::new(&mut c);\n",
        "        let _ = g;\n",
        "    };\n",
        "    let _ = f;\n",
        "}\n",
        "pub fn ok_closure_poll_no_rpit() {\n",
        "    let mut c = Cell(0);\n",
        "    let f = || {\n",
        "        let _r = Pin::new(&mut c).poll_val();\n",
        "        let _p = Pin::new(&mut c);\n",
        "    };\n",
        "    let _ = f;\n",
        "}\n",
    );
    let (dir, outline, names) = load_borrow_fixture("rpit_2024", "2024", &core_src, app_src);
    let mut cx = Context::new(&outline, &names, Cache::default());
    let bad_err = run_borrow_on_item(
        &mut cx,
        find_app_item(&outline, "bad_closure_rpit_alias"),
        Substs::EMPTY,
    )
    .expect_err("Edition 2024 RPIT lock must keep mutable borrow");
    assert!(
        format!("{bad_err:#}").contains("cannot borrow"),
        "unexpected error: {bad_err:#}"
    );
    assert!(
        run_borrow_on_item(
            &mut cx,
            find_app_item(&outline, "ok_closure_poll_no_rpit"),
            Substs::EMPTY
        )
        .is_ok()
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn multi_segment_assoc_string_not_treated_as_static() {
    use std::sync::Arc;

    use crate::sema::infer::{LazyTables, Local, Typed};

    let (outline, names) = test_outline(
        concat!(
            "pub trait Tr { type String; }\n",
            "pub fn bad<T: Tr>(s: T::String) {}\n",
            "pub fn bad_ref<'a, T: Tr<String = &'a str>>(s: &T::String) {}\n",
            "pub fn good_ref(s: &String) {}\n",
        ),
        "2021",
    );
    let mut cx = Context::new(&outline, &names, Cache::default());
    let unit = cx.types.unit();
    let typed = Typed {
        tables: LazyTables::Eager(Arc::default()),
        locals: vec![Local { ty: unit }],
        params: vec![0],
        ..Typed::default()
    };
    let src = &outline.crates[0].files[0].source;

    let bad_item = find_item_in(&outline, "bad");
    let bad_sig = cx.signature(bad_item).expect("bad sig");
    let bad_info = super::super::calls::compute_param_lifetimes(
        &mut cx,
        bad_item,
        &bad_sig,
        src,
        &typed,
        (1, &[]),
    );
    assert!(bad_info.has_elided_lt[0]);
    assert!(!bad_info.is_static[0]);

    let bad_ref_item = find_item_in(&outline, "bad_ref");
    let bad_ref_sig = cx.signature(bad_ref_item).expect("bad_ref sig");
    let bad_ref_info = super::super::calls::compute_param_lifetimes(
        &mut cx,
        bad_ref_item,
        &bad_ref_sig,
        src,
        &typed,
        (1, &[]),
    );
    assert!(!bad_ref_info.inner_is_static[0]);

    let good_item = find_item_in(&outline, "good_ref");
    let good_sig = cx.signature(good_item).expect("good_ref sig");
    let good_info = super::super::calls::compute_param_lifetimes(
        &mut cx,
        good_item,
        &good_sig,
        src,
        &typed,
        (1, &[]),
    );
    assert!(good_info.inner_is_static[0]);
}

#[test]
fn user_box_struct_with_cell_is_invariant_not_covariant() {
    let core_src = concat!(
        "#![no_core]\n",
        "#[lang = \"sized\"]\n",
        "pub trait Sized {}\n",
        "#[lang = \"unsafe_cell\"]\n",
        "pub struct UnsafeCell<T>(T);\n",
    );
    let app_src = concat!(
        "use core::UnsafeCell;\n",
        "pub struct Cell<T>(UnsafeCell<T>);\n",
        "pub struct Box<'a>(Cell<&'a i32>);\n",
        "pub struct Cov<'a>(&'a i32);\n",
        "pub fn user_box<'a>(_: Box<'a>) {}\n",
        "pub fn user_cov<'a>(_: Cov<'a>) {}\n",
    );
    let (dir, outline, names) = load_borrow_fixture("user_box_var", "2021", core_src, app_src);
    let mut cx = Context::new(&outline, &names, Cache::default());
    let src = &outline.crates[1].files[0].source;

    let box_fn = find_app_item(&outline, "user_box");
    let box_sig = cx.signature(box_fn).expect("box sig");
    let box_inv = super::super::checks::collect_ast_invariant_lifetimes(
        &mut cx,
        Some(box_fn),
        &box_sig.body,
        src,
        box_sig.params[0].1,
        None,
    );
    assert_eq!(box_inv.named, vec!["'a".to_owned()]);

    let cov_fn = find_app_item(&outline, "user_cov");
    let cov_sig = cx.signature(cov_fn).expect("cov sig");
    let cov_inv = super::super::checks::collect_ast_invariant_lifetimes(
        &mut cx,
        Some(cov_fn),
        &cov_sig.body,
        src,
        cov_sig.params[0].1,
        None,
    );
    assert!(cov_inv.is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn custom_iterator_trait_item_static_does_not_drop_for_loop_borrows() {
    use std::sync::Arc;

    use crate::body::Expr;
    use crate::names::Primitive;
    use crate::sema::infer::{BodyTables, LazyTables, Local, Res, SyntaxTables, Typed};
    use crate::sema::substs::PackedMap;
    use crate::sema::ty::{Kind, Mut};

    let core_src = format!(
        "{MIN_CORE_TRAITS}{}",
        concat!(
            "pub enum Option<T> { None, Some(T) }\n",
            "pub mod iter {\n",
            "    use super::Option;\n",
            "    pub trait Iterator {\n",
            "        type Item;\n",
            "        #[lang = \"next\"]\n",
            "        fn next(&mut self) -> Option<Self::Item>;\n",
            "    }\n",
            "    pub trait IntoIterator {\n",
            "        type Item;\n",
            "        type IntoIter: Iterator<Item = Self::Item>;\n",
            "        #[lang = \"into_iter\"]\n",
            "        fn into_iter(self) -> Self::IntoIter;\n",
            "    }\n",
            "}\n",
        )
    );
    let app_src = concat!(
        "pub trait Iterator { type Item; }\n",
        "pub fn bad_custom<'a, I: Iterator<Item = i32> + core::iter::Iterator<Item = &'a i32>>(it: I) -> &'static i32 {\n",
        "    for x in it {\n",
        "        return x;\n",
        "    }\n",
        "    &0\n",
        "}\n",
        "pub fn bad_multi_bound<'a, I: core::iter::IntoIterator<Item = i32, IntoIter = I> + core::iter::Iterator<Item = &'a i32>>(it: I) -> &'static i32 {\n",
        "    for x in it {\n",
        "        return x;\n",
        "    }\n",
        "    &0\n",
        "}\n",
        "pub fn ok_std_iter<'a, I: core::iter::Iterator<Item = i32>>(it: I) -> i32 {\n",
        "    for x in it {\n",
        "        return x;\n",
        "    }\n",
        "    0\n",
        "}\n",
    );
    let (dir, outline, names) = load_borrow_fixture("custom_iter", "2021", &core_src, app_src);
    let mut cx = Context::new(&outline, &names, Cache::default());
    let param0 = cx.types.intern(crate::sema::ty::Kind::Param(0));
    let i32_ty = cx.types.prim(Primitive::I32);
    let ref_i32 = cx.types.intern(Kind::Ref(Mut::Not, i32_ty));

    let check_for_fn = |cx: &mut Context<'_>, name: &str, item_ty| {
        let fn_item = find_app_item(&outline, name);
        let sig = cx.signature(fn_item).expect("sig");
        let body = cx.body(fn_item).expect("body");
        let src = &cx.file(fn_item).source;
        let mut exprs = vec![cx.types.unit(); body.exprs.len()];
        let mut binds = Vec::new();
        let mut res = Vec::new();
        for expr in &body.exprs {
            match *expr {
                Expr::For { pat, iter, .. } => {
                    binds.push((pat, 1u32));
                    res.push((iter, Res::Local(0)));
                    exprs[iter.0 as usize] = param0;
                }
                Expr::Return(Some(val)) => {
                    res.push((val, Res::Local(1)));
                    exprs[val.0 as usize] = item_ty;
                }
                _ => {}
            }
        }
        let tables = BodyTables {
            syntax: Arc::new(SyntaxTables {
                binds: PackedMap::pack(binds),
                ..SyntaxTables::default()
            }),
            exprs: exprs.into_boxed_slice(),
            res: PackedMap::pack(res),
            ..BodyTables::default()
        };
        let typed = Typed {
            tables: LazyTables::Eager(Arc::new(tables)),
            locals: vec![Local { ty: param0 }, Local { ty: item_ty }],
            params: vec![0],
            ret: Some(item_ty),
            ..Typed::default()
        };
        super::super::check(cx, fn_item, (&body, &sig), src, &typed)
    };

    assert!(check_for_fn(&mut cx, "bad_custom", ref_i32).is_err());
    assert!(check_for_fn(&mut cx, "bad_multi_bound", ref_i32).is_err());
    assert!(check_for_fn(&mut cx, "ok_std_iter", i32_ty).is_ok());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn match_ref_deref_coercion_peels_outer_container_temp_only() {
    let app_src = concat!(
        "pub struct Buf(u8);\n",
        "impl Buf {\n",
        "    pub fn new() -> Self { Buf(0) }\n",
        "}\n",
        "pub enum Opt<T> {\n",
        "    Some(T),\n",
        "    None,\n",
        "}\n",
        "pub struct Holder<'a>(&'a Buf);\n",
        "impl<'a> Holder<'a> {\n",
        "    pub fn strip(&self) -> Opt<&'a Buf> { Opt::Some(self.0) }\n",
        "    pub fn as_opt(&self) -> Opt<&'a Buf> { Opt::Some(self.0) }\n",
        "}\n",
        "pub fn ok_strip_ref<'a>(arg: &'a Buf) -> &'a Buf {\n",
        "    let h = Holder(arg);\n",
        "    match &h.strip() {\n",
        "        Opt::Some(s) => s,\n",
        "        Opt::None => arg,\n",
        "    }\n",
        "}\n",
        "pub fn ok_let_opt_match<'a>(arg: &'a Buf) -> &'a Buf {\n",
        "    let h = Holder(arg);\n",
        "    let opt = &h.as_opt();\n",
        "    match opt {\n",
        "        Opt::Some(s) => s,\n",
        "        Opt::None => arg,\n",
        "    }\n",
        "}\n",
        "pub struct Owned(Buf);\n",
        "impl core::Deref for Owned {\n",
        "    type Target = Buf;\n",
        "    fn deref(&self) -> &Buf { &self.0 }\n",
        "}\n",
        "pub enum StyleRef {\n",
        "    Reference(Owned),\n",
        "    Other,\n",
        "}\n",
        "pub struct Edges(StyleRef);\n",
        "impl Edges {\n",
        "    pub fn get(&self, _k: &&Buf) -> Opt<&StyleRef> { Opt::Some(&self.0) }\n",
        "}\n",
        "pub fn ok_nested_ref_deref_match<'a>(edges: &'a Edges, init: &'a Buf) -> &'a Buf {\n",
        "    let mut node = init;\n",
        "    match &edges.get(&node) {\n",
        "        Opt::Some(StyleRef::Reference(child)) => node = child,\n",
        "        Opt::Some(StyleRef::Other) | Opt::None => {}\n",
        "    }\n",
        "    node\n",
        "}\n",
        "pub fn bad_nested_inner_temp() {\n",
        "    static B: Buf = Buf(0);\n",
        "    let mut r: &Buf = &B;\n",
        "    {\n",
        "        let s = StyleRef::Reference(Owned(Buf::new()));\n",
        "        let mut tmp_r: &Buf = &B;\n",
        "        match &Opt::Some(&s) {\n",
        "            Opt::Some(StyleRef::Reference(child)) => tmp_r = child,\n",
        "            _ => {}\n",
        "        }\n",
        "        r = tmp_r;\n",
        "    }\n",
        "    let _ = r;\n",
        "}\n",
        "pub fn bad_inner_temp_in_match() {\n",
        "    static B: Buf = Buf(0);\n",
        "    let mut r: &Buf = &B;\n",
        "    {\n",
        "        match &Opt::Some(&Buf::new()) {\n",
        "            Opt::Some(s) => r = s,\n",
        "            Opt::None => {}\n",
        "        }\n",
        "    }\n",
        "    let _ = r;\n",
        "}\n",
        "pub fn bad_uncoerced_container_ref<'a>(arg: &'a Buf) {\n",
        "    let h = Holder(arg);\n",
        "    let init = Opt::None;\n",
        "    let mut out: &Opt<&Buf> = &init;\n",
        "    {\n",
        "        let r = &h.strip();\n",
        "        out = r;\n",
        "    }\n",
        "    let _ = out;\n",
        "}\n",
    );
    let (dir, outline, names) =
        load_borrow_fixture("match_ref_deref", "2021", MIN_CORE_TRAITS, app_src);
    let mut cx = Context::new(&outline, &names, Cache::default());
    run_borrow_on_item(
        &mut cx,
        find_app_item(&outline, "ok_strip_ref"),
        Substs::EMPTY,
    )
    .expect("outer container temp must peel on &&T -> &T");
    run_borrow_on_item(
        &mut cx,
        find_app_item(&outline, "ok_let_opt_match"),
        Substs::EMPTY,
    )
    .expect("bound &Container match must peel on &&T -> &T");
    run_borrow_on_item(
        &mut cx,
        find_app_item(&outline, "ok_nested_ref_deref_match"),
        Substs::EMPTY,
    )
    .expect("nested enum behind &Edges::get(&node) must peel outer temp on &&T -> &T");
    let nested_inner_err = run_borrow_on_item(
        &mut cx,
        find_app_item(&outline, "bad_nested_inner_temp"),
        Substs::EMPTY,
    )
    .expect_err("inner local behind nested ref in match must still fail");
    assert!(
        nested_inner_err
            .to_string()
            .contains("does not live long enough"),
        "unexpected error: {nested_inner_err}"
    );
    let inner_err = run_borrow_on_item(
        &mut cx,
        find_app_item(&outline, "bad_inner_temp_in_match"),
        Substs::EMPTY,
    )
    .expect_err("inner temp inside match must still fail");
    assert!(
        inner_err
            .to_string()
            .contains("temporary value dropped while borrowed"),
        "unexpected error: {inner_err}"
    );
    let outer_err = run_borrow_on_item(
        &mut cx,
        find_app_item(&outline, "bad_uncoerced_container_ref"),
        Substs::EMPTY,
    )
    .expect_err("uncoerced &Container temp must still fail");
    assert!(
        outer_err
            .to_string()
            .contains("temporary value dropped while borrowed"),
        "unexpected error: {outer_err}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn invariant_target_sharing_type_param_and_named_lifetime_passes() {
    let core_src = format!(
        "{MIN_CORE_TRAITS}\n\
         #[lang = \"drop_glue\"]\n\
         pub unsafe fn drop_in_place<T: ?Sized>(_: *mut T) {{}}\n\
         pub mod slice {{\n\
             pub struct Vec<T>(T);\n\
             impl<T> Vec<T> {{\n\
                 pub fn to_vec(&self) -> Vec<T> {{ loop {{}} }}\n\
                 pub fn push(&mut self, _v: T) {{}}\n\
             }}\n\
         }}\n",
    );
    let app_src = concat!(
        "use core::slice::Vec;\n",
        "pub trait Marker {}\n",
        "pub struct Opaque<'a>(&'a u8);\n",
        "pub struct LineSections<'a, S>(&'a u8, S);\n",
        "impl<'a, S> LineSections<'a, S> {\n",
        "    pub fn to_vec(&self) -> LineSections<'a, S> { loop {} }\n",
        "    pub fn as_opaque(&self) -> Opaque<'a> { Opaque(self.0) }\n",
        "}\n",
        "pub type AliasLines<'a, S> = Vec<(&'a u8, S)>;\n",
        "pub fn ok_push_shared<'a, S: Marker>(\n",
        "    wrapped: &mut Vec<LineSections<'a, S>>,\n",
        "    input_vec: LineSections<'a, S>,\n",
        ") {\n",
        "    wrapped.push(input_vec.to_vec());\n",
        "}\n",
        "pub fn ok_push_alias<'a, S: Marker>(\n",
        "    wrapped: &mut Vec<AliasLines<'a, S>>,\n",
        "    input_vec: AliasLines<'a, S>,\n",
        ") {\n",
        "    wrapped.push(input_vec.to_vec());\n",
        "}\n",
        "pub fn ok_identity<'a, S: Marker>(x: LineSections<'a, S>) -> LineSections<'a, S> {\n",
        "    x.to_vec()\n",
        "}\n",
        "pub fn bad_mismatch_lt<'a, 'b, S: Marker>(\n",
        "    wrapped: &mut Vec<LineSections<'a, S>>,\n",
        "    input_vec: LineSections<'b, S>,\n",
        ") {\n",
        "    wrapped.push(input_vec.to_vec());\n",
        "}\n",
        "pub fn bad_unshared_ty_param<'a, S: Marker>(\n",
        "    out: &mut Vec<Opaque<'a>>,\n",
        "    input_vec: LineSections<'a, S>,\n",
        ") {\n",
        "    out.push(input_vec.as_opaque());\n",
        "}\n",
    );
    let (dir, outline, names) =
        load_borrow_fixture("shared_ty_param_lt", "2021", &core_src, app_src);
    let mut cx = Context::new(&outline, &names, Cache::default());
    let param0 = cx.types.intern(crate::sema::ty::Kind::Param(0));
    let param_substs = cx.types.intern_substs(&[param0]);
    let u8_ty = cx.types.prim(crate::names::Primitive::U8);
    let mono_substs = cx.types.intern_substs(&[u8_ty]);
    run_borrow_on_item(
        &mut cx,
        find_app_item(&outline, "ok_push_shared"),
        param_substs,
    )
    .expect("shared unbounded type param + matching named lifetime must pass");
    run_borrow_on_item(
        &mut cx,
        find_app_item(&outline, "ok_push_shared"),
        mono_substs,
    )
    .expect("monomorphized shared unbounded type param + named lifetime must pass");
    run_borrow_on_item(
        &mut cx,
        find_app_item(&outline, "ok_push_alias"),
        param_substs,
    )
    .expect("type alias with shared type param + named lifetime must pass");
    run_borrow_on_item(
        &mut cx,
        find_app_item(&outline, "ok_push_alias"),
        mono_substs,
    )
    .expect("monomorphized type alias with shared type param + named lifetime must pass");
    run_borrow_on_item(
        &mut cx,
        find_app_item(&outline, "ok_identity"),
        param_substs,
    )
    .expect("returning LineSections<'a, S> with shared S must pass");
    run_borrow_on_item(&mut cx, find_app_item(&outline, "ok_identity"), mono_substs)
        .expect("monomorphized return of LineSections<'a, S> with shared S must pass");
    run_borrow_on_item(
        &mut cx,
        find_app_item(&outline, "bad_mismatch_lt"),
        param_substs,
    )
    .expect_err("mismatched named lifetime must still fail");
    run_borrow_on_item(
        &mut cx,
        find_app_item(&outline, "bad_unshared_ty_param"),
        param_substs,
    )
    .expect_err("unshared unbounded type param in target must still fail");
    run_borrow_on_item(
        &mut cx,
        find_app_item(&outline, "bad_unshared_ty_param"),
        mono_substs,
    )
    .expect_err("monomorphized unshared unbounded type param in target must still fail");
    let _ = std::fs::remove_dir_all(&dir);
}

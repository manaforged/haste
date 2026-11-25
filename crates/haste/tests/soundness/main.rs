#[path = "../common/mod.rs"]
mod common;
mod edge_cases;
mod lifetimes;
mod self_host;

use common::FixtureDir;
use haste::{Compiler, ErrorKind};

fn run_check(name: &str, src: &str) -> Result<(), haste::Error> {
    let fixture = FixtureDir::write_crate("sound", name, &format!("sound_{name}"), src);
    Compiler::builder()
        .jobs(1)
        .build()
        .expect("build compiler")
        .check(fixture.path())
}

pub(crate) fn check_fails(name: &str, src: &str) -> haste::Error {
    let err = run_check(name, src).expect_err("unsound program must be rejected");
    assert_eq!(err.kind(), ErrorKind::Compile);
    err
}

fn check_fails_borrow(name: &str, src: &str) {
    let fixture = FixtureDir::write_crate("sound", name, &format!("sound_{name}"), src);
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_haste"))
        .args(["build", "--jobs", "1"])
        .arg(fixture.path())
        .output()
        .expect("run haste build");
    assert!(!out.status.success(), "{name} should fail borrowck");
    let combined = format!(
        "{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        combined.contains("error[borrow]:"),
        "expected borrow diagnostic for {name}, got:\n{combined}"
    );
}

fn check_passes(name: &str, src: &str) {
    run_check(name, src).expect("valid program must compile");
}

#[test]
fn rejects_non_send_rc_across_thread_spawn() {
    check_fails(
        "rc_send",
        r#"
use std::rc::Rc;
fn main() {
    let rc = Rc::new(1);
    let _ = std::thread::spawn(move || drop(rc));
}
"#,
    );
    check_fails(
        "raw_ptr_send",
        r#"
fn need_send<T: Send>(_: T) {}
fn main() {
    let x = 1;
    let p: *const i32 = &x;
    need_send(p);
}
"#,
    );
    check_fails(
        "ref_cell_sync",
        r#"
use std::cell::RefCell;
fn need_sync<T: Sync>(_: T) {}
fn main() {
    let c = RefCell::new(1);
    need_sync(c);
}
"#,
    );
    check_fails(
        "poly_rc_send",
        r#"
use std::rc::Rc;
fn need_send<T: Send>(_: T) {}
fn bad<T>(x: Rc<T>) { need_send(x); }
fn main() { bad(Rc::new(1i32)); }
"#,
    );
}

#[test]
fn rejects_non_exhaustive_matches() {
    check_fails(
        "exh_enum",
        r#"
enum Color { Red, Green, Blue }
fn main() {
    let c = Color::Red;
    let _ = match c { Color::Red => 1, Color::Green => 2 };
}
"#,
    );
    check_fails(
        "exh_int_var",
        r#"
fn main() {
    let x = 1;
    let _ = match x { 0 => 1 };
}
"#,
    );
    check_fails(
        "exh_large_or_product",
        r#"
fn main() {
    let p: (u8, u8) = (0, 8);
    let _ = match p {
        (0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8, 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7) => 0,
        (9..=255, _) => 1,
    };
}
"#,
    );
    check_fails(
        "exh_guarded",
        r#"
enum Bit { Zero, One }
fn main() {
    let b = Bit::Zero;
    let _ = match b {
        Bit::Zero => 0,
        Bit::One if false => 1,
    };
}
"#,
    );
    check_fails(
        "exh_poly_bool",
        r#"
fn bad<T>(b: bool) -> i32 { match b { true => 1 } }
fn main() { let _ = bad::<i32>(false); }
"#,
    );
}

#[test]
fn rejects_private_field_and_item_access() {
    check_fails(
        "priv_field",
        r#"
mod inner {
    pub struct Boxed { secret: i32 }
    impl Boxed { pub fn new() -> Self { Self { secret: 7 } } }
}
fn main() {
    let b = inner::Boxed::new();
    println!("{}", b.secret);
}
"#,
    );
    check_fails(
        "priv_tuple_field",
        r#"
mod inner {
    pub struct Wrap(i32);
    impl Wrap { pub fn new() -> Self { Self(1) } }
}
fn main() {
    let w = inner::Wrap::new();
    println!("{}", w.0);
}
"#,
    );
    check_fails(
        "priv_item",
        r#"
mod inner {
    fn hidden() -> i32 { 42 }
}
fn main() {
    println!("{}", inner::hidden());
}
"#,
    );
    check_fails(
        "priv_block_use",
        r#"
mod inner {
    fn secret() -> i32 { 42 }
}
fn main() {
    use inner::secret;
    println!("{}", secret());
}
"#,
    );
    check_fails(
        "priv_struct_ctor",
        r#"
mod inner {
    struct PrivStruct { pub x: i32 }
}
fn main() {
    let _ = inner::PrivStruct { x: 1 };
}
"#,
    );
}

#[test]
fn accepts_valid_send_exhaustive_visibility_and_static_patterns() {
    check_passes(
        "valid_controls",
        r#"
use std::sync::{Arc, Mutex};

mod inner {
    pub struct Config {
        pub name: &'static str,
        pub(super) code: i32,
    }
    impl Config {
        pub fn name_ref(&self) -> &'static str {
            self.name
        }
    }
    pub fn ok_call() -> i32 {
        1
    }
}

enum Color {
    Red,
    Green,
    Blue,
}

fn ok_static(s: &'static str) -> &'static str {
    s
}

mod dyn_super {
    pub trait Base {
        fn ping(&self) -> i32;
    }
    pub trait Sub: Base {
        fn sub_ping(&self) -> i32;
    }
    pub struct Impl;
    impl Base for Impl {
        fn ping(&self) -> i32 {
            7
        }
    }
    impl Sub for Impl {
        fn sub_ping(&self) -> i32 {
            11
        }
    }
}

fn call_sub(s: &dyn dyn_super::Sub) -> i32 {
    s.ping() + s.sub_ping()
}

fn main() {
    let a = Arc::new(Mutex::new(1));
    let a2 = Arc::clone(&a);
    let h = std::thread::spawn(move || *a2.lock().unwrap());
    let _ = h.join();

    let cfg = inner::Config {
        name: "ok",
        code: inner::ok_call(),
    };
    let _ = ok_static(cfg.name_ref());

    let c = Color::Red;
    let _ = match c {
        Color::Red => 1,
        Color::Green => 2,
        Color::Blue => 3,
    };

    let b = true;
    let _ = match (b, Some(1u8)) {
        (true, Some(0..=127)) => 1,
        (true, Some(128..=255)) => 2,
        (true, None) => 3,
        (false, _) => 4,
    };

    let s: &[bool] = &[true, false];
    let _ = match s {
        [] => 0,
        [true, ..] => 1,
        [false, ..] => 2,
    };

    let arr = [true];
    let _ = match arr {
        [true | false] => 0,
    };

    let big = [0u8; 32];
    let _ = match big {
        [x, ..] => x,
    };

    let u: u128 = 5;
    let _ = match u {
        0 => 0,
        1..=u128::MAX => 1,
    };

    let _ = call_sub(&dyn_super::Impl);

    let sep = String::from(", ");
    let joined = [1, 2]
        .iter()
        .map(|n| format!("{n}"))
        .collect::<Vec<_>>()
        .join(&sep);
    let _ = joined.len();

    fn with_ref<R>(f: impl FnOnce(&i32) -> R) -> R {
        let local = 9;
        f(&local)
    }
    let _ = with_ref(|x| *x + 1);

    fn for_same_lt<'a>(s: &'a str, mut f: impl FnMut(&'a str)) {
        f(s);
    }
    let mut seen = "";
    for_same_lt("ok", |s| seen = s);
    let _ = seen.len();

    fn slice_static(xs: &[&'static str]) -> Vec<&'static str> {
        xs.to_vec()
    }
    let _ = slice_static(&["a", "b"]);

    struct StaticSelf<'a>(&'a str);
    impl<'a> StaticSelf<'a> {
        fn get(&'static self) -> &'static str {
            self.0
        }
    }
    static S_HOLD: StaticSelf<'static> = StaticSelf("ok");
    let _ = S_HOLD.get();

    fn stash_where<'a, 'b, F: FnMut(&'a i32)>(mut f: F, y: &'b i32)
    where
        'b: 'a,
    {
        f(y);
    }
    fn stash_trans<'a, 'b: 'a, 'c: 'b, F: FnMut(&'a i32)>(mut f: F, y: &'c i32) {
        f(y);
    }
    let val = 5;
    let mut got = 0;
    stash_where(|r| got += *r, &val);
    stash_trans(|r| got += *r, &val);
    let _ = got;

    fn vec_from_static(xs: &[&'static str]) -> Vec<&'static str> {
        Vec::from(xs)
    }
    let _ = vec_from_static(&["x", "y"]);

    fn first_ref<'a>(s: &[&'a str]) -> &'a str {
        s[0]
    }
    fn index_alias_static(xs: &[&'static str]) -> &'static str {
        let a = xs;
        a[0]
    }
    fn first_direct_static(xs: &[&'static str]) -> &'static str {
        first_ref(xs)
    }
    fn first_alias_static(xs: &[&'static str]) -> &'static str {
        let a = xs;
        first_ref(a)
    }
    fn vec_index_static(xs: &Vec<&'static str>) -> &'static str {
        xs[0]
    }
    let static_slice: &[&'static str] = &["a", "b"];
    let static_vec: Vec<&'static str> = vec!["a", "b"];
    let _ = index_alias_static(static_slice);
    let _ = first_direct_static(static_slice);
    let _ = first_alias_static(static_slice);
    let _ = vec_index_static(&static_vec);

    static STATIC_NUM: i32 = 7;
    let _ = with_ref(|x| Some(x).map(|_| &STATIC_NUM));
    let _ = with_ref(|x| {
        let _k = loop {
            break x;
        };
        &STATIC_NUM
    });
    let reassign_param = |mut x: &i32| {
        let y = x;
        x = &STATIC_NUM;
        *y + *x
    };
    let _ = reassign_param(&5);

    fn ok_mixed_tuple<'a>(s: &'a str, f: impl FnOnce((&'static str, &'a str))) {
        f(("lit", s));
    }
    fn ok_nested_tuple<'a>(s: &'a str, f: impl FnOnce(((&'static str, &'a str), &'a str))) {
        f((("lit", s), s));
    }
    fn ok_if_tuple<'a>(s: &'a str, f: impl FnOnce((&'static str, &'a str))) {
        f(if true { ("lit", s) } else { ("a", s) });
    }
    fn ok_iter_map_deref(xs: &[&'static str]) -> Vec<&'static str> {
        xs.iter().map(|s| *s).collect()
    }
    fn ok_first_map_deref(xs: &[&'static str]) -> Option<&'static str> {
        xs.first().map(|s| *s)
    }
    struct StaticNode {
        next: &'static StaticNode,
    }
    fn ok_static_node(x: &StaticNode) -> &'static StaticNode {
        x.next
    }
    static NODE: StaticNode = StaticNode { next: &NODE };
    let _ = ok_static_node(&NODE);
    let _ = ok_iter_map_deref(static_slice);
    let _ = ok_first_map_deref(static_slice);
    let local_s = String::from("ok");
    ok_mixed_tuple(&local_s, |(a, b)| {
        let _ = (a.len(), b.len());
    });
    ok_nested_tuple(&local_s, |((a, b), c)| {
        let _ = (a.len(), b.len(), c.len());
    });
    ok_if_tuple(&local_s, |(a, b)| {
        let _ = (a.len(), b.len());
    });
    fn ok_bound_mixed_tuple<'a>(s: &'a str, f: impl FnOnce((&'static str, &'a str))) {
        let t = ("lit", s);
        f(t);
    }
    fn ok_bound_nested_tuple<'a>(s: &'a str, f: impl FnOnce(((&'static str, &'a str), &'a str))) {
        let t = (("lit", s), s);
        f(t);
    }
    ok_bound_mixed_tuple(&local_s, |(a, b)| {
        let _ = (a.len(), b.len());
    });
    ok_bound_nested_tuple(&local_s, |((a, b), c)| {
        let _ = (a.len(), b.len(), c.len());
    });
    let get_static_elem = |v: &Vec<&'static u8>| -> &'static u8 { v[0] };
    let r_static;
    {
        let v = vec![&1u8];
        r_static = get_static_elem(&v);
    }
    let _ = *r_static;
    let out_static: Vec<&'static str>;
    {
        let v = vec![("a", 1u8)];
        out_static = v.iter().map(|p: &(&'static str, u8)| p.0).collect();
    }
    let _ = out_static.len();
    fn ok_byref_match_tup<'a>(x: (&'a i32,)) -> &'a i32 {
        match &x {
            &(y,) => y,
        }
    }
    fn ok_byref_match_nested<'a>(x: (&'a &'a i32,)) -> &'a i32 {
        match &x {
            (&y,) => y,
        }
    }
    fn ok_byref_match_ergo<'a>(x: (&'a i32,)) -> &'a i32 {
        match &x {
            (y,) => *y,
        }
    }
    let n_ok = 42;
    let r_ok = &n_ok;
    let _ = ok_byref_match_tup((&n_ok,));
    let _ = ok_byref_match_nested((&r_ok,));
    let _ = ok_byref_match_ergo((&n_ok,));

    fn ok_sync_async_closure<Fut>(f: impl FnOnce(&i32) -> Fut) -> Fut {
        let local = 9;
        f(&local)
    }
    let _fut = ok_sync_async_closure(|_x: &i32| async move { 42 });

    fn ok_for_loop_reassigned_tuple(mut f: impl FnMut((&&'static str, &'static str))) {
        static S: &str = "lit";
        for mut t in [(&S, "lit")] {
            f(t);
            let local: &'static str = "a";
            t = (&local, "lit");
            let _ = t;
        }
    }
    ok_for_loop_reassigned_tuple(|(a, b)| {
        let _ = (a.len(), b.len());
    });

    fn ok_bound_tuple_outer_hrtb<'a>(r: &'a str, f: impl FnOnce((&&'static str, &'a str))) {
        let s: &'static str = "lit";
        let t = (&s, r);
        f(t);
    }
    ok_bound_tuple_outer_hrtb(&local_s, |(a, b)| {
        let _ = (a.len(), b.len());
    });
}
"#,
    );
}

#[test]
fn rejects_core_borrow_checker_violations() {
    check_fails_borrow(
        "use_after_move",
        r#"
fn main() {
    let s = String::from("hi");
    let _t = s;
    let _ = s.len();
}
"#,
    );
    check_fails_borrow(
        "branch_move",
        r#"
fn main() {
    let s = String::from("hi");
    if s.len() > 0 {
        drop(s);
    }
    let _ = s.len();
}
"#,
    );
    check_fails_borrow(
        "mut_shared_alias",
        r#"
fn main() {
    let mut x = 1;
    let r = &x;
    let m = &mut x;
    *m += 1;
    let _ = *r;
}
"#,
    );
    check_fails_borrow(
        "mutate_while_borrowed",
        r#"
fn main() {
    let mut x = 1;
    let r = &x;
    x = 2;
    let _ = *r;
}
"#,
    );
    check_fails_borrow(
        "dangling_local_ref",
        r#"
fn bad() -> &'static i32 {
    let x = 7;
    &x
}
fn main() {
    let _ = bad();
}
"#,
    );
    check_passes(
        "nll_release_ok",
        r#"
fn main() {
    let mut x = 1;
    let r = &x;
    let y = *r;
    x = y + 1;
    let m = &mut x;
    *m += 1;
}
"#,
    );
}

mod common;

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

fn check_fails(name: &str, src: &str) -> haste::Error {
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
fn rejects_non_static_borrow_returned_as_static() {
    check_fails(
        "static_escape",
        r#"
fn leak(s: &str) -> &'static str { s }
fn main() {
    let local = String::from("hi");
    println!("{}", leak(&local));
}
"#,
    );
    check_fails(
        "static_escape_indirect",
        r#"
fn leak(s: &str) -> &'static str {
    let alias = s;
    alias
}
fn main() {
    let local = String::from("hi");
    println!("{}", leak(&local));
}
"#,
    );
    check_fails(
        "static_escape_double_ref",
        r#"
struct S { x: i32 }
fn leak(s: &S) -> &'static S {
    let rr = &s;
    *rr
}
fn main() {
    let s = S { x: 1 };
    println!("{}", leak(&s).x);
}
"#,
    );
    check_fails(
        "static_escape_self_param",
        r#"
struct W<'a> { s: &'a str }
impl<'a> W<'a> {
    fn leak(other: Self) -> &'static str { other.s }
}
fn main() {
    let local = String::from("hi");
    let w = W { s: &local };
    println!("{}", W::leak(w));
}
"#,
    );
    check_fails(
        "static_escape_autoderef_double_ref",
        r#"
fn leak(s: &str) -> &'static str {
    let rr = &s;
    rr
}
fn main() {
    let local = String::from("hi");
    println!("{}", leak(&local));
}
"#,
    );
    check_fails(
        "static_escape_local_struct_field",
        r#"
struct S { f: &'static str }
fn leak() -> &'static str {
    let local = String::from("dangling");
    let s = S { f: &local };
    s.f
}
fn main() {
    println!("{}", leak());
}
"#,
    );
    check_fails(
        "hrtb_outer_type_param_escape_identity",
        r#"
fn bad<R>(f: impl FnOnce(&i32) -> R) -> R {
    let local = 9;
    f(&local)
}
fn main() {
    let dangling: &i32 = bad(|x| x);
    println!("{dangling}");
}
"#,
    );
    check_fails(
        "hrtb_outer_type_param_escape_option",
        r#"
fn bad<R>(f: impl FnOnce(&i32) -> R) -> R {
    let local = 9;
    f(&local)
}
fn main() {
    let dangling: Option<&i32> = bad(|x| Some(x));
    println!("{dangling:?}");
}
"#,
    );
    check_fails(
        "copied_recv_alias_to_vec_escape",
        r#"
fn leak<'a>(xs: &[&'a str]) -> Vec<&'static str> {
    let alias = xs;
    alias.to_vec()
}
fn main() {
    let local = String::from("dangling");
    let s: &[&str] = &[&local];
    let v: Vec<&'static str> = leak(s);
    drop(local);
    println!("{}", v[0]);
}
"#,
    );
    check_fails(
        "copied_recv_elided_adt_clone_escape",
        r#"
#[derive(Clone)]
struct Wrap<'a>(&'a str);
fn leak(w: &Wrap) -> Wrap<'static> {
    w.clone()
}
fn main() {
    let local = String::from("dangling");
    let w = Wrap(&local);
    let leaked = leak(&w);
    drop(local);
    println!("{}", leaked.0);
}
"#,
    );
    check_fails(
        "caller_lt_mixed_static_tuple_escape",
        r#"
fn leak<'a>(s: &'a str, f: impl FnOnce((&'static str, &'a str))) {
    f((s, s));
}
fn main() {
    let local = String::from("dangling");
    leak(&local, |(s, _)| { let _: &'static str = s; });
}
"#,
    );
    check_fails(
        "caller_lt_unrelated_lifetime_escape",
        r#"
fn stash<'a, 'b, F: FnMut(&'a i32)>(mut f: F, y: &'b i32) {
    f(y);
}
fn main() {
    let long = 1;
    let mut out: &i32 = &long;
    {
        let short = 2;
        stash(|r| out = r, &short);
    }
    println!("{out}");
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

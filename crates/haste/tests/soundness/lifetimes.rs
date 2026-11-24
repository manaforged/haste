use crate::check_fails;

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
    check_fails(
        "hrtb_outer_type_param_escape_match_bind",
        r#"
fn bad<R>(f: impl FnOnce(&i32) -> R) -> R {
    let local = 9;
    f(&local)
}
fn main() {
    let dangling: &i32 = bad(|x| match x { y => y });
    println!("{dangling}");
}
"#,
    );
    check_fails(
        "hrtb_outer_type_param_escape_loop_break",
        r#"
fn bad<R>(f: impl FnOnce(&i32) -> R) -> R {
    let local = 9;
    f(&local)
}
fn main() {
    let dangling: &i32 = bad(|x| loop { break x; });
    println!("{dangling}");
}
"#,
    );
    check_fails(
        "hrtb_outer_type_param_escape_generic_fn_item",
        r#"
fn bad<R>(f: impl FnOnce(&i32) -> R) -> R {
    let local = 9;
    f(&local)
}
fn id<T>(x: T) -> T { x }
fn main() {
    let dangling: &i32 = bad(id);
    println!("{dangling}");
}
"#,
    );
    check_fails(
        "copied_arg_vec_from_alias_escape",
        r#"
fn leak<'a>(xs: &[&'a str]) -> Vec<&'static str> {
    let alias = xs;
    Vec::from(alias)
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
        "hrtb_outer_type_param_escape_labeled_loop_break",
        r#"
fn bad<R>(f: impl FnOnce(&i32) -> R) -> R {
    let local = 9;
    f(&local)
}
fn main() {
    let dangling: &i32 = bad(|x| 'a: loop { loop { break 'a x; } });
    println!("{dangling}");
}
"#,
    );
    check_fails(
        "hrtb_outer_type_param_escape_let_loop",
        r#"
fn bad<R>(f: impl FnOnce(&i32) -> R) -> R {
    let local = 9;
    f(&local)
}
fn main() {
    let dangling: &i32 = bad(|x| { let k = loop { break x; }; k });
    println!("{dangling}");
}
"#,
    );
    check_fails(
        "hrtb_outer_type_param_escape_option_map",
        r#"
fn bad<R>(f: impl FnOnce(&i32) -> R) -> R {
    let local = 9;
    f(&local)
}
fn main() {
    let dangling: Option<&i32> = bad(|x| Some(x).map(|y| y));
    println!("{dangling:?}");
}
"#,
    );
    check_fails(
        "hrtb_outer_type_param_escape_nested_move_closure",
        r#"
fn bad<R>(f: impl FnOnce(&i32) -> R) -> R {
    let local = 9;
    f(&local)
}
fn main() {
    let dangling: &i32 = bad(|x| { let f = move || x; f() });
    println!("{dangling}");
}
"#,
    );
    check_fails(
        "hrtb_outer_type_param_escape_reassigned_param",
        r#"
fn bad<R>(f: impl FnOnce(&i32) -> R) -> R {
    let local = 9;
    f(&local)
}
fn main() {
    static S: i32 = 1;
    let dangling: &i32 = bad(|mut x| { let y = x; x = &S; y });
    println!("{dangling}");
}
"#,
    );
    check_fails(
        "static_escape_slice_ref_of_static_elems",
        r#"
fn bad(xs: &[&'static str]) -> &'static [&'static str] {
    &xs[..]
}
fn main() {
    let arr = ["a", "b"];
    let _ = bad(&arr);
}
"#,
    );
    check_fails(
        "static_escape_elem_slot_ref",
        r#"
fn bad(xs: &[&'static str]) -> &'static &'static str {
    &xs[0]
}
fn main() {
    let arr = ["a"];
    let _ = bad(&arr);
}
"#,
    );
    check_fails(
        "static_escape_non_static_index_alias",
        r#"
fn bad<'a>(xs: &[&'a str]) -> &'static str {
    let a = xs;
    a[0]
}
fn main() {
    let local = String::from("dangling");
    let arr: [&str; 1] = [&local];
    let leaked = bad(&arr);
    drop(local);
    println!("{leaked}");
}
"#,
    );
    check_fails(
        "static_escape_non_static_first_call",
        r#"
fn first<'a>(s: &[&'a str]) -> &'a str {
    s[0]
}
fn bad<'a>(xs: &[&'a str]) -> &'static str {
    first(xs)
}
fn main() {
    let local = String::from("dangling");
    let arr: [&str; 1] = [&local];
    let leaked = bad(&arr);
    drop(local);
    println!("{leaked}");
}
"#,
    );
    check_fails(
        "static_escape_ref_ref_id_call",
        r#"
fn id_ref<'a>(x: &&'a str) -> &'a str {
    *x
}
fn bad<'a>(s: &'a str) -> &'static str {
    let r = &s;
    id_ref(r)
}
fn main() {
    let local = String::from("dangling");
    let leaked = bad(&local);
    drop(local);
    println!("{leaked}");
}
"#,
    );
    check_fails(
        "static_escape_box_deref",
        r#"
fn bad(s: &str) -> &'static str {
    let b = Box::new(s);
    *b
}
fn main() {
    let local = String::from("dangling");
    let leaked = bad(&local);
    drop(local);
    println!("{leaked}");
}
"#,
    );
    check_fails(
        "static_escape_ref_to_static_mut",
        r#"
fn bad<'a>(x: &'a &'static mut u8) -> &'static u8 {
    *x
}
fn main() {
    let r: &'static mut u8 = Box::leak(Box::new(1));
    let _ = bad(&r);
}
"#,
    );
    check_fails(
        "hrtb_block_tuple_projection_escape",
        r#"
fn bad<R>(f: impl FnOnce(&i32) -> R) -> R {
    let local = 9;
    f(&local)
}
fn main() {
    let d: &i32 = bad(|x| ({ let y = x; (y,) }).0);
    println!("{d}");
}
"#,
    );
    check_fails(
        "hrtb_reborrow_then_reassign_param_escape",
        r#"
fn bad<R>(f: impl FnOnce(&i32) -> R) -> R {
    let local = 9;
    f(&local)
}
static S: i32 = 1;
fn main() {
    let d: &i32 = bad(|mut x| {
        let y = &*x;
        x = &S;
        y
    });
    println!("{d}");
}
"#,
    );
    check_fails(
        "static_escape_to_vec_elem_refs",
        r#"
fn bad(xs: &[&'static str]) -> Vec<&'static &'static str> {
    [&xs[0]].to_vec()
}
fn main() {
    let arr = ["a"];
    let _ = bad(&arr);
}
"#,
    );
    check_fails(
        "static_escape_local_array_to_vec",
        r#"
fn bad() -> Vec<&'static i32> {
    let x = 1;
    [&x].to_vec()
}
fn main() {
    let _ = bad();
}
"#,
    );
    check_fails(
        "static_escape_cast_array_deref",
        r#"
fn bad(xs: &[&'static str; 1]) -> &'static [&'static str] {
    let r = &(xs as &[&'static str]);
    *r
}
fn main() {
    let arr = ["a"];
    let _ = bad(&arr);
}
"#,
    );
    check_fails(
        "static_escape_anon_impl_lifetime",
        r#"
struct Hold<'a>(&'a str);
impl Hold<'_> {
    fn bad(&self) -> &'static str {
        self.0
    }
}
fn main() {
    let local = String::from("dangling");
    let h = Hold(&local);
    let leaked = h.bad();
    drop(local);
    println!("{leaked}");
}
"#,
    );
    check_fails(
        "static_escape_async_fn_future",
        r#"
async fn helper(s: &str) -> &'static str {
    let _ = s;
    "ok"
}
fn need_static<T: 'static>(_: T) {}
fn main() {
    let local = String::from("dangling");
    let fut = helper(&local);
    need_static(fut);
}
"#,
    );
    check_fails(
        "static_escape_moved_vec_to_static",
        r#"
fn need_static<T: 'static>(_: T) {}
fn main() {
    let local = String::from("dangling");
    let v = vec![&local];
    need_static(v);
}
"#,
    );
    check_fails(
        "caller_lt_alias_unrelated_escape",
        r#"
fn bad_alias<'a, 'b>(s: &'a str, mut f: impl FnMut(&'b str)) {
    let r = s;
    f(r);
}
fn main() {
    let long = String::from("long");
    let mut out: &str = &long;
    {
        let short = String::from("short");
        bad_alias(&short, |r| out = r);
    }
    println!("{out}");
}
"#,
    );
    check_fails(
        "caller_lt_tuple_multi_lt_escape",
        r#"
fn bad_tuple_any<'a, 'b>(s: &'a str, f: impl FnOnce((&'a str, &'b str))) {
    let t = (s, s);
    f(t);
}
fn main() {
    let long = String::from("long");
    let mut out: &str = &long;
    {
        let short = String::from("short");
        bad_tuple_any(&short, |(_, r)| out = r);
    }
    println!("{out}");
}
"#,
    );
    check_fails(
        "hrtb_tuple_param_escape",
        r#"
fn bad_tup_hrtb<R>(f: impl FnOnce((&i32,)) -> R) -> R {
    let local = 9;
    f((&local,))
}
fn main() {
    let dangling: &i32 = bad_tup_hrtb(|(x,)| x);
    println!("{dangling}");
}
"#,
    );
}

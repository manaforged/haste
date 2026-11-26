use crate::check_fails;

#[test]
fn rejects_deref_and_byref_match_param_escapes() {
    check_fails(
        "deref_double_ref_scalar_param_escape",
        r#"
fn leak_scalar(x: i32) -> &'static i32 {
    let r = &x;
    let rr = &r;
    *rr
}
fn main() {
    let _ = leak_scalar(1);
}
"#,
    );
    check_fails(
        "deref_ref_to_string_slice_param_escape",
        r#"
fn leak_string(x: String) -> &'static str {
    let s = x.as_str();
    let r = &s;
    *r
}
fn main() {
    let _ = leak_string(String::from("hi"));
}
"#,
    );
    check_fails(
        "deref_double_ref_tuple_param_escape",
        r#"
fn leak_tup<'a>(x: (&'a i32,)) -> &'a (&'a i32,) {
    let r = &x;
    let rr = &r;
    *rr
}
fn main() {
    let n = 1;
    let _ = leak_tup((&n,));
}
"#,
    );
    check_fails(
        "deref_field_via_local_ref_escape",
        r#"
fn leak<'a>(x: (i32, &'a i32), cond: bool) -> &'a i32 {
    let a: &i32 = &x.0;
    let r: &&i32 = if cond { &a } else { &x.1 };
    *r
}
fn main() {
    let n = 1;
    let _ = leak((7, &n), true);
}
"#,
    );
    check_fails(
        "deref_field_via_reassigned_ref_param_escape",
        r#"
fn leak<'a>(x: (i32, &'a i32), mut p: &'a i32) -> &'a i32 {
    p = &x.0;
    let r = &p;
    *r
}
fn main() {
    let n = 1;
    let _ = leak((7, &n), &n);
}
"#,
    );
    check_fails(
        "deref_mutated_byval_param_escape",
        r#"
fn leak<'a>(mut x: (&'a i32,)) -> &'a i32 {
    let local = 9;
    x.0 = &local;
    match &x {
        (y,) => *y,
    }
}
fn main() {
    let n = 1;
    let _ = leak((&n,));
}
"#,
    );
    check_fails(
        "byref_match_block_expr_param_escape",
        r#"
fn leak<'a>(x: (&'a i32,)) -> &'a (&'a i32,) {
    match &{ (&x,) } {
        &(y,) => y,
    }
}
fn main() {
    let n = 1;
    let _ = leak((&n,));
}
"#,
    );
    check_fails(
        "byref_match_block_expr_closure_escape",
        r#"
fn main() {
    let n = 1;
    let c = |x: (&i32,)| match &{ (&x,) } {
        &(y,) => y,
    };
    let d = c((&n,));
    println!("{}", d.0);
}
"#,
    );
    check_fails(
        "byref_match_block_expr_scalar_escape",
        r#"
fn leak_scalar(x: i32) -> &'static i32 {
    match &{ (&x,) } {
        &(y,) => y,
    }
}
fn main() {
    let _ = leak_scalar(1);
}
"#,
    );
    check_fails(
        "closure_copies_inner_double_ref_escape",
        r#"
fn main() {
    let get = |p: &&u8| -> &u8 { *p };
    let r;
    {
        let x = 1u8;
        r = get(&&x);
    }
    println!("{r}");
}
"#,
    );
    check_fails(
        "generic_deref_copy_double_ref_escape",
        r#"
fn deref_copy<T: Copy>(r: &T) -> T {
    *r
}
fn main() {
    let r: &u8;
    {
        let x = 1u8;
        r = deref_copy(&&x);
    }
    println!("{r}");
}
"#,
    );
    check_fails(
        "bound_tuple_cell_alias_escape",
        r#"
use std::cell::Cell;
fn leak<'a>(s: &'a str, f: impl FnOnce((Cell<&'static str>, &'a str))) {
    let t = (Cell::new("lit"), s);
    let r = &t;
    r.0.set(s);
    f(t);
}
fn main() {
    let local = String::from("dangling");
    leak(&local, |(c, _)| {
        let _: &'static str = c.get();
    });
}
"#,
    );
    check_fails(
        "bound_tuple_ref_cell_inner_ref_escape",
        r#"
use std::cell::Cell;
fn leak<'a>(s: &'a str, f: impl FnOnce((&Cell<&'static str>, &'a str))) {
    let x = Cell::new("lit");
    let t = (&x, s);
    x.set(s);
    f(t);
}
fn main() {
    let local = String::from("dangling");
    leak(&local, |(a, _)| {
        let _: &'static str = a.get();
    });
}
"#,
    );
    check_fails(
        "sync_closure_pat_param_async_move_escape",
        r#"
fn bad<R>(f: impl FnOnce(&(i32,)) -> R) -> R {
    let local = (9,);
    f(&local)
}
fn main() {
    let _fut = bad(|&(ref x,)| async move { *x });
}
"#,
    );
    check_fails(
        "bound_tuple_loop_backedge_escape",
        r#"
fn leak<'a>(s: &'a str, mut f: impl FnMut((&'static str, &'a str))) {
    let mut t = ("lit", s);
    for _ in 0..2 {
        f(t);
        t = (s, s);
    }
}
fn main() {
    let local = String::from("dangling");
    leak(&local, |(a, _)| {
        let _: &'static str = a;
    });
}
"#,
    );
    check_fails(
        "bound_tuple_if_overwrite_escape",
        r#"
fn leak<'a>(s: &'a str, c: bool, f: impl FnOnce((&'static str, &'a str))) {
    let mut t = ("lit", s);
    if c {
        t = (s, s);
    }
    f(t);
}
fn main() {
    let local = String::from("dangling");
    leak(&local, true, |(a, _)| {
        let _: &'static str = a;
    });
}
"#,
    );
}

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

#[test]
fn deref_field_conflict_and_box_disjoint() {
    const PRELUDE: &str = "use std::ops::{Deref, DerefMut};\nstruct Target { x: i32 }\nstruct Boxed { flag: bool, inner: Target }\nimpl Deref for Boxed { type Target = Target; fn deref(&self) -> &Target { &self.inner } }\nimpl DerefMut for Boxed { fn deref_mut(&mut self) -> &mut Target { &mut self.inner } }\n";
    crate::check_fails_borrow(
        "deref_container_field_mutated_while_target_borrowed",
        &format!(
            "{PRELUDE}fn main() {{ let mut b = Boxed {{ flag: false, inner: Target {{ x: 1 }} }}; let r = &b.x; b.flag = true; let _ = *r; }}"
        ),
    );
    crate::check_fails_borrow(
        "deref_target_mutated_while_container_field_borrowed",
        &format!(
            "{PRELUDE}fn main() {{ let mut b = Boxed {{ flag: false, inner: Target {{ x: 1 }} }}; let r = &b.flag; b.x = 2; let _ = *r; }}"
        ),
    );

    crate::check_passes(
        "box_disjoint_field_mut_borrows",
        r#"
struct Pair {
    a: i32,
    b: i32,
}
fn main() {
    let mut p = Box::new(Pair { a: 1, b: 2 });
    let ra = &mut (*p).a;
    let rb = &mut (*p).b;
    *ra += *rb;
}
"#,
    );
}

#[test]
fn mut_temp_promotion_rules() {
    crate::check_fails_borrow(
        "rejects_returning_mut_ref_to_temp",
        "fn bad<'a>() -> &'a mut u32 { &mut 4 }\nfn main() { let _ = bad(); }",
    );
    crate::check_fails_borrow(
        "rejects_explicit_return_mut_ref_to_temp",
        "fn bad() -> &'static mut u32 { return &mut 4; }\nfn main() { let _ = bad(); }",
    );
    crate::check_passes(
        "allows_shared_temp_promotion_zst_and_static_mut",
        "static mut S_MUT: &'static mut u32 = &mut 4;\nfn ok() -> &'static u32 { &4 }\nfn ok_empty<'a>() -> &'a mut [u8; 0] { &mut [] }\nfn main() { let _ = ok(); let _ = ok_empty(); }",
    );
    crate::check_fails_borrow(
        "rejects_mut_ref_to_unit_zst",
        "fn bad<'a>() -> &'a mut () { &mut () }\nfn main() { let _ = bad(); }",
    );
    crate::check_fails_borrow(
        "rejects_mut_ref_to_zst_struct",
        "struct Z;\nfn bad<'a>() -> &'a mut Z { &mut Z }\nfn main() { let _ = bad(); }",
    );
    crate::check_fails_borrow(
        "rejects_mut_ref_through_overloaded_deref_temp",
        "use std::ops::{Deref, DerefMut};\nstruct Wrap(());\nimpl Deref for Wrap { type Target = (); fn deref(&self) -> &() { &self.0 } }\nimpl DerefMut for Wrap { fn deref_mut(&mut self) -> &mut () { &mut self.0 } }\nfn bad<'a>() -> &'a mut () { &mut *Wrap(()) }\nfn main() { let _ = bad(); }",
    );
    crate::check_fails_borrow(
        "rejects_const_nested_mut_temp",
        "const C: &'static &'static mut u32 = &&mut 4;\nfn main() { let _ = C; }",
    );
    crate::check_fails_borrow(
        "rejects_static_mut_loop_temp",
        "static mut S: &'static mut u32 = loop { break &mut 4; };\nfn main() { let _ = unsafe { &raw const S }; }",
    );
}

#[test]
fn loop_move_and_mut_borrow_rules() {
    crate::check_fails_borrow(
        "rejects_move_in_loop_without_reinit",
        r#"
fn main() {
    let mut s = String::from("a");
    let mut i = 0;
    while i < 2 {
        drop(s);
        i += 1;
        if i == 1 {
            continue;
        }
        s = String::from("b");
    }
}
"#,
    );

    crate::check_fails_borrow(
        "rejects_mut_ref_pushed_in_loop",
        r#"
fn main() {
    let mut x = 1;
    let mut v = Vec::new();
    let mut i = 0;
    while i < 2 {
        let r = &mut x;
        v.push(r);
        i += 1;
    }
    let _ = v;
}
"#,
    );

    crate::check_fails_borrow(
        "rejects_outer_mut_ref_moved_in_loop",
        r#"
fn main() {
    let mut x = 1;
    let mut v = Vec::new();
    let r = &mut x;
    let mut i = 0;
    while i < 2 {
        v.push(r);
        i += 1;
    }
    let _ = v;
}
"#,
    );

    crate::check_passes(
        "allows_reinit_and_non_escaping_mut_ref_in_loop",
        r#"
fn main() {
    let mut s = String::from("a");
    let mut v = 0;
    let mut i = 0;
    while i < 2 {
        drop(s);
        s = String::from("b");
        let r = &mut v;
        *r += 1;
        i += 1;
    }
    let _ = (s, v);
}
"#,
    );
}

#[test]
fn match_self_guarded_arms_and_catchall_reborrow() {
    crate::check_passes(
        "allows_match_self_guarded_arms_then_catchall_mut",
        r#"
enum Cmd {
    Build { path: String },
    Test { filter: Option<String> },
}
impl Cmd {
    fn update(&mut self, name: &str) {
        match self {
            Cmd::Build { path } if name == "build" => {
                let _ = path;
            }
            Cmd::Test { filter } if name == "test" => {
                let _ = filter;
            }
            s => {
                *s = Cmd::Build {
                    path: String::new(),
                };
            }
        }
    }
}
fn main() {
    let mut c = Cmd::Test { filter: None };
    c.update("other");
}
"#,
    );

    crate::check_fails_borrow(
        "rejects_match_catchall_move_while_scrutinee_borrowed",
        r#"
enum Cmd {
    Build { path: String },
    Test { filter: Option<String> },
}
fn main() {
    let mut c = Cmd::Test { filter: None };
    let c_ref = &mut c;
    let r = &*c_ref;
    match c_ref {
        s => {
            *s = Cmd::Build {
                path: String::new(),
            };
            let _ = r;
        }
    }
}
"#,
    );
}

#[test]
fn loop_liveness_opaque_and_jump_temp_cleanup() {
    crate::check_fails_borrow(
        "rejects_cross_iteration_opaque_borrow",
        r#"
fn main() {
    let mut v = Vec::new();
    v.push((1, 2));
    v.push((3, 4));
    let mut out: Option<&i32> = None;
    let mut i = 0;
    while i < 2 {
        let (ref a, ref mut b) = v[i];
        if i == 0 {
            out = Some(a);
        } else {
            *b += 1;
        }
        i += 1;
    }
    let _ = out;
}
"#,
    );

    crate::check_fails_borrow(
        "rejects_continue_before_rebind_while_borrowed",
        r#"
fn main() {
    let mut x = 1;
    let mut y = 2;
    let mut r = &mut x;
    let mut i = 0;
    while i < 2 {
        let _ = *r;
        i += 1;
        if i == 1 {
            x += 1;
            continue;
        }
        r = &mut y;
    }
}
"#,
    );

    crate::check_passes(
        "allows_break_killing_borrow_in_loop",
        r#"
fn main() {
    let mut v = Vec::new();
    v.push(1);
    let mut i = 0;
    loop {
        let r = &v[0];
        if i == 1 {
            v.push(2);
            break;
        }
        i += *r;
    }
}
"#,
    );

    crate::check_passes(
        "allows_jump_out_of_temp_holders_without_leaks",
        r#"
struct Guard<'a>(&'a mut i32);
impl Drop for Guard<'_> {
    fn drop(&mut self) {}
}
fn get(x: &mut i32) -> Option<Guard<'_>> {
    Some(Guard(x))
}
fn main() {
    let mut x = 0;
    while let Some(_g) = get(&mut x) {
        break;
    }
    x += 1;
    let _ = x;

    let mut v = Vec::new();
    let mut opt = Some(1);
    loop {
        v.push(match opt {
            Some(n) => n,
            None => break,
        });
        opt = None;
    }
    v.push(2);

    'outer: loop {
        for _x in &mut v {
            continue 'outer;
        }
        break;
    }
    v.push(3);
    let _ = v;
}
"#,
    );
}

#[test]
fn alias_and_callable_reborrow_rules() {
    crate::check_passes(
        "allows_type_alias_and_box_dyn_and_impl_fn_mut_reborrows",
        r#"
type MutRef<'a> = &'a mut i32;
struct Wrap<'a>(MutRef<'a>);
struct Named<'a> {
    f: MutRef<'a>,
}
fn take_alias(x: MutRef<'_>) {
    *x += 1;
}
fn call_impl(mut f: impl FnMut(&mut i32), r: &mut i32) {
    f(r);
    *r += 1;
}
struct Caller<F> {
    f: F,
}
impl<F: FnMut(&mut i32)> Caller<F> {
    fn run(&mut self, r: &mut i32) {
        (self.f)(r);
        *r += 1;
    }
}
fn main() {
    let mut x = 1;
    let r = &mut x;
    take_alias(r);
    *r += 1;
    let _w = Wrap { 0: r };
    *r += 1;
    let _n = Named { f: r };
    *r += 1;
    let mut b: Box<dyn FnMut(&mut i32)> = Box::new(|p: &mut i32| *p += 1);
    b(r);
    *r += 1;
    call_impl(|p: &mut i32| *p += 1, r);
    *r += 1;
    let mut c = Caller { f: b };
    c.run(r);
    *r += 1;
}
"#,
    );
}

#[test]
fn jump_liveness_const_clamp_and_move_loans() {
    crate::check_fails_borrow(
        "rejects_for_iter_mutation_when_body_returns",
        "fn f() -> i32 { let mut x = 1; let r = &x; for _ in { x += 1; [1] } { return 0; } *r }\nfn main() { let _ = f(); }",
    );
    crate::check_fails_borrow(
        "rejects_match_scrutinee_mutation_with_return_arm",
        "fn main() { let mut x = 1; let r = &x; match { x += 1; Some(0) } { Some(_) => { let _ = *r; } None => return, } }",
    );
    crate::check_fails_borrow(
        "rejects_const_struct_with_nested_mut_temp",
        "struct Sh<'a> { f: &'a &'a mut u32 }\nconst C1: Sh<'static> = Sh { f: &&mut 4 };\nfn main() { let _ = C1; }",
    );
    crate::check_fails_borrow(
        "rejects_const_fn_taking_mut_temp",
        "const fn id(x: &'static u32) -> &'static u32 { x }\nconst C2: &'static u32 = id(&mut 4);\nfn main() { let _ = C2; }",
    );
    crate::check_fails_borrow(
        "rejects_mutating_target_after_call_out_moved_src",
        "fn put<T>(dst: &mut T, src: T) { *dst = src; }\nfn main() { let mut x = 1; let mut y = 2; let mut d = &mut x; let s = &mut y; put(&mut d, s); y += 1; *d += 1; }",
    );
    crate::check_passes(
        "allows_break_from_index_lhs_without_temp_leak",
        "fn main() { let mut v = Vec::new(); v.push(0); let mut opt = Some(0usize); loop { v[match opt { Some(i) => i, None => break }] = 1; opt = None; } v.push(2); let _ = v; }",
    );
    crate::check_fails_borrow(
        "rejects_mutating_after_tuple_moved_mut_ref",
        "fn main() { let mut x = 1; let r = &mut x; let t = (r,); x += 1; *t.0 += 1; }",
    );
    crate::check_fails_borrow(
        "rejects_mutating_after_some_moved_mut_ref",
        "fn main() { let mut x = 1; let r = &mut x; let o = Some(r); x += 1; *o.unwrap() += 1; }",
    );
}

#[test]
fn tuple_reborrow_slice_promo_and_split_returns() {
    crate::check_passes(
        "allows_tuple_reborrow_coercion_in_calls_and_annotated_lets",
        r#"
fn f(t: (&mut i32, &mut i32)) { *t.0 += 1; *t.1 += 1; }
fn nested(t: ((&mut i32,), i32)) { *((t.0).0) += 1; }
fn main() {
    let (mut a, mut b) = (1, 2);
    let (r1, r2) = (&mut a, &mut b);
    f((r1, r2)); *r1 += 1; *r2 += 1;
    nested(((r1,), 0)); *r1 += 1;
    let _t: (&mut i32,) = (r1,); *r1 += 1;
}
"#,
    );
    crate::check_fails_borrow(
        "rejects_unannotated_let_tuple_moving_mut_ref",
        "fn main() { let mut x = 1; let r = &mut x; let _t = (r,); *r += 1; }",
    );
    crate::check_fails_borrow(
        "rejects_generic_call_tuple_moving_mut_ref",
        "fn g<T>(_: T) {} fn main() { let mut x = 1; let r = &mut x; g((r,)); *r += 1; }",
    );
    crate::check_fails_borrow(
        "rejects_live_annotated_let_tuple_aliasing_mut_ref",
        "fn main() { let mut x = 1; let r = &mut x; let _t: (&mut i32,) = (r,); *r += 1; *_t.0 += 1; }",
    );
    crate::check_passes(
        "allows_empty_slice_full_range_promo_and_ref_method_field_index",
        r#"
struct Block { params: Vec<i32> }
struct Func { blocks: Vec<Block> }
struct Emit(Func);
impl Emit { fn func(&self) -> &Func { &self.0 } }
fn pick(opt: Option<&[u32]>) -> &[u32] { opt.map_or(&[][..], |s| s) }
fn first_params(emit: &Emit) -> usize { emit.func().blocks[0].params.iter().count() }
fn main() {
    let e = Emit(Func { blocks: vec![Block { params: vec![1] }] });
    let _ = (pick(None), first_params(&e));
}
"#,
    );
    crate::check_fails_borrow(
        "rejects_mut_nonempty_slice_full_range_return",
        "fn bad<'a>() -> &'a mut [u8] { &mut [1][..] }\nfn main() { let _ = bad(); }",
    );
    crate::check_passes(
        "allows_tuple_return_lifetime_splitting_with_and_without_try",
        r#"
struct Frame<'a> { generics: Vec<(&'a str, u32)> }
struct Cx<'a>(&'a str);
impl<'a> Cx<'a> {
    fn parent_frame<'s>(&mut self, substs: &'s [u32]) -> (Vec<(&'a str, u32)>, &'s [u32]) {
        (vec![(self.0, 0)], substs)
    }
    fn parent_frame_res<'s>(&mut self, substs: &'s [u32]) -> Result<(Vec<(&'a str, u32)>, &'s [u32]), ()> {
        Ok((vec![(self.0, 0)], substs))
    }
    fn frame(&mut self, substs: &[u32]) -> Frame<'a> {
        let (generics, _rest) = self.parent_frame(substs);
        Frame { generics }
    }
    fn frame_res(&mut self, substs: &[u32]) -> Result<Frame<'a>, ()> {
        let (generics, _rest) = self.parent_frame_res(substs)?;
        Ok(Frame { generics })
    }
}
fn main() { let mut cx = Cx("ok"); let _ = (cx.frame(&[1]), cx.frame_res(&[1])); }
"#,
    );
    crate::check_fails_borrow(
        "rejects_returning_second_tuple_elem_borrowed_from_short_param",
        "fn split<'a, 's>(a: &'a str, b: &'s str) -> (&'a str, &'s str) { (a, b) }\nfn bad<'a>(a: &'a str, b: &str) -> &'a str { let (_x, y) = split(a, b); y }\nfn main() { let _ = bad(\"a\", \"b\"); }",
    );
}

#[test]
fn tuple_pattern_inner_ref_escape_rules() {
    crate::check_passes(
        "allows_filter_tuple_pat_reading_inner_ref_field",
        "use std::collections::HashSet;\nstruct Disk { device: String }\nstruct Storage { volumes: Vec<Disk> }\nimpl Storage { fn counted(&self) -> Vec<(usize, &Disk)> { let mut seen = HashSet::new(); self.volumes.iter().enumerate().filter(|(_, volume)| seen.insert(volume.device.as_str())).collect() } fn counted_pat(&self) -> Vec<(usize, &Disk)> { let mut seen = HashSet::new(); self.volumes.iter().enumerate().filter(|&(_, volume)| seen.insert(volume.device.as_str())).collect() } }\nfn main() { let st = Storage { volumes: vec![Disk { device: String::from(\"nvme0\") }] }; let _ = (st.counted().len(), st.counted_pat().len()); }",
    );
    crate::check_fails_borrow(
        "rejects_filter_tuple_pat_escaping_owned_string",
        "use std::collections::HashSet;\nfn main() { let xs = vec![String::from(\"a\")]; let mut seen = HashSet::new(); let _: Vec<_> = xs.into_iter().enumerate().filter(|(_, s)| seen.insert(s.as_str())).collect(); }",
    );
    crate::check_fails_borrow(
        "rejects_filter_tuple_pat_escaping_boxed_string",
        "use std::collections::HashSet;\nfn main() { let xs = vec![Box::new(String::from(\"a\"))]; let mut seen = HashSet::new(); let _: Vec<_> = xs.into_iter().enumerate().filter(|(_, b)| seen.insert(b.as_str())).collect(); }",
    );
    crate::check_fails_borrow(
        "rejects_filter_tuple_pat_boxed_ref_outliving_vec",
        "use std::collections::HashSet;\nfn main() { let mut seen = HashSet::new(); let xs = vec![Box::new(String::from(\"a\"))]; let _ = xs.iter().enumerate().filter(|(_, b)| seen.insert(b.as_str())).count(); drop(xs); let _ = seen.len(); }",
    );
    crate::check_fails_borrow(
        "rejects_filter_tuple_pat_escaping_mut_inner_ref",
        "use std::collections::HashSet;\nfn main() { let mut a = String::from(\"a\"); let xs: Vec<&mut String> = vec![&mut a]; let mut seen = HashSet::new(); let _: Vec<_> = xs.into_iter().enumerate().filter(|(_, s)| seen.insert(s.as_str())).collect(); }",
    );
    crate::check_fails_borrow(
        "rejects_nested_hrtb_tuple_pat_escape",
        "fn bad(f: impl FnOnce(&(usize, &u8))) { let local = 9u8; f(&(0, &local)); }\nfn main() { let mut out = &0u8; bad(|(_, r)| out = *r); println!(\"{out}\"); }",
    );
}

#[test]
fn downcast_replace_and_static_closure_cloned_handles() {
    crate::check_passes(
        "allows_downcast_mut_replace_and_restamp_monomorphized",
        "use std::any::Any;\n#[derive(Clone, Copy)] struct Hooks<'a>(&'a str);\nstruct Slot { value: Box<dyn Any> }\nstruct Rt { slot: Slot, key: u32 }\nfn live_slot(rt: &mut Rt, _k: u32) -> &mut Slot { &mut rt.slot }\nfn restamp(rt: &mut Rt, k: u32) { rt.key = k; }\nfn set_val<T: 'static>(rt: &mut Rt, key: u32, value: T) -> T { let current = live_slot(rt, key).value.downcast_mut::<T>().expect(\"type\"); let old = core::mem::replace(current, value); restamp(rt, key); old }\nfn main() { let mut rt1 = Rt { slot: Slot { value: Box::new(\"a\") }, key: 1 }; let _ = set_val(&mut rt1, 2, \"b\"); let mut rt2 = Rt { slot: Slot { value: Box::new(Hooks(\"a\")) }, key: 1 }; let _ = set_val(&mut rt2, 2, Hooks(\"b\")); }",
    );
    crate::check_passes(
        "allows_static_closures_capturing_cloned_and_copied_handles",
        "use std::cell::{Cell, RefCell};\nuse std::rc::Rc;\nuse std::sync::Arc;\n#[derive(Clone, Copy)] struct Handle<'a>(&'a str);\nstruct Item { handle: Handle<'static>, label: &'static str, cb: Arc<dyn Fn()> }\nstruct Bound { count: Rc<dyn Fn() -> usize> }\nstruct VList<'a> { _cx: &'a str, bound: Option<Bound> }\nstruct TableCol(&'static str);\nimpl TableCol { fn new(a: &'static str, _b: &'static str) -> Self { Self(a) } }\nstruct TableState<Row>(std::marker::PhantomData<Row>);\nimpl<Row> TableState<Row> { fn row_eq_fn(&self) -> Option<Rc<dyn Fn(&Row, &Row) -> bool>> { None } }\nmod row_sealed { pub trait Sealed<T>: 'static {} }\ntrait RowRender<T>: row_sealed::Sealed<T> + 'static { fn run(&self) -> usize; }\nstruct BoundStrip<T, R>(std::marker::PhantomData<T>, R);\nimpl<T, R> BoundStrip<T, R> { fn runner(&self) -> Rc<dyn Fn()> where R: Clone + RowRender<T> { let r = self.1.clone(); Rc::new(move || { let _ = r.run(); }) } }\n#[derive(Clone)] struct DummyRender;\nimpl<T: 'static> row_sealed::Sealed<T> for DummyRender {}\nimpl<T: 'static> RowRender<T> for DummyRender { fn run(&self) -> usize { 1 } }\nfn spawn(_: impl FnOnce() + 'static) {}\nfn fixed(ids: &[&'static str]) -> Vec<TableCol> { ids.iter().map(|&id| TableCol::new(id, id)).collect() }\nfn follow_bound(v: &VList<'_>) { let count = v.bound.as_ref().map(|bound| Rc::clone(&bound.count)); spawn(move || { let _n = count.as_ref().map_or(0, |c| c()); }); }\nfn build_row<Row: 'static>(state: &TableState<Row>) { let eq = state.row_eq_fn(); spawn(move || { let _eq = eq.clone(); }); }\nfn bound_strip<T, R>(spec: &BoundStrip<T, R>) where T: 'static, R: RowRender<T> + Clone { let runner = spec.runner(); spawn(move || runner()); }\nfn main() { let item = Item { handle: Handle(\"h\"), label: \"l\", cb: Arc::new(|| {}) }; let r = &item; let h1 = r.handle.clone(); let h2 = r.label; let h3 = Arc::clone(&r.cb); let state = RefCell::new(Item { handle: Handle(\"s\"), label: \"sl\", cb: Arc::new(|| {}) }); let h4 = state.borrow().handle.clone(); let items = vec![Handle(\"v\")]; for &h5 in &items { spawn(move || { let _v = h5; }); } let cell = Cell::new(Handle(\"c\")); let c_ref = &cell; let h6 = c_ref.get(); spawn(move || { let _ = (h1, h2, h4, h6); h3(); }); let _ = fixed(&[\"a\"]); let vl = VList { _cx: \"x\", bound: None }; follow_bound(&vl); let ts: TableState<u32> = TableState(std::marker::PhantomData); build_row(&ts); let bs: BoundStrip<u32, DummyRender> = BoundStrip(std::marker::PhantomData, DummyRender); bound_strip(&bs); }",
    );
    crate::check_fails_borrow(
        "rejects_mem_replace_on_non_static_mut_ref_param",
        "fn bad<'a>(rt: &'a mut &'a str, val: &'a str) -> &'static str { core::mem::replace(rt, val) }\nfn main() { let mut s = String::from(\"x\"); let mut r = s.as_str(); let _ = bad(&mut r, \"y\"); drop(s); }",
    );
    crate::check_fails_borrow(
        "rejects_downcast_mut_ref_escaping_without_replace",
        "fn bad(rt: &mut Box<dyn std::any::Any>) -> &'static mut &'static str { rt.downcast_mut::<&'static str>().expect(\"bad\") }\nfn main() { let mut b: Box<dyn std::any::Any> = Box::new(\"x\"); let _ = bad(&mut b); }",
    );
    crate::check_fails_borrow(
        "rejects_static_closure_capturing_cloned_borrowed_str",
        "fn spawn(_: impl FnOnce() + 'static) {}\nfn main() { let s = String::from(\"x\"); let r = s.as_str().clone(); spawn(move || println!(\"{r}\")); }",
    );
    crate::check_fails_borrow(
        "rejects_static_closure_capturing_copied_borrowed_handle",
        "#[derive(Clone, Copy)] struct Handle<'a>(&'a str);\nfn spawn(_: impl FnOnce() + 'static) {}\nfn main() { let s = String::from(\"x\"); let items = vec![Handle(s.as_str())]; for &h in &items { spawn(move || { let _v = h.0; }); } }",
    );
    crate::check_fails_borrow(
        "rejects_borrowed_slice_iter_map_to_static",
        "fn spawn(_: impl FnOnce() + 'static) {}\nfn bad(ids: &[&str]) { let cols: Vec<_> = ids.iter().map(|&id| id).collect(); spawn(move || { let _c = cols; }); }\nfn main() { let s = String::from(\"a\"); bad(&[s.as_str()]); }",
    );
    crate::check_fails_borrow(
        "rejects_static_closure_capturing_borrowed_option_map_clone",
        "use std::rc::Rc;\nstruct Bound<'a> { s: Rc<&'a str> }\nfn spawn(_: impl FnOnce() + 'static) {}\nfn bad(b: Option<&Bound<'_>>) { let c = b.map(|bound| Rc::clone(&bound.s)); spawn(move || { let _c = c; }); }\nfn main() { let s = String::from(\"a\"); let b = Bound { s: Rc::new(s.as_str()) }; bad(Some(&b)); }",
    );
    crate::check_fails_borrow(
        "rejects_static_closure_with_non_static_trait_bound",
        "trait NonStatic { fn as_str(&self) -> &str; }\nstruct Wrap<R>(R);\nfn spawn(_: impl FnOnce() + 'static) {}\nfn bad<R: NonStatic>(spec: &Wrap<R>) { let s = spec.0.as_str(); spawn(move || { let _s = s; }); }\nstruct S<'a>(&'a str);\nimpl NonStatic for S<'_> { fn as_str(&self) -> &str { self.0 } }\nfn main() { let local = String::from(\"a\"); let w = Wrap(S(local.as_str())); bad(&w); }",
    );
}

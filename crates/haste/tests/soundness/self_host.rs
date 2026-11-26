use super::{check_fails, check_passes};

#[test]
fn self_host_const_array_obligation_settles_after_exhaustiveness() {
    check_passes(
        "const_array_debug_match",
        r#"
use std::ptr::NonNull;
const TAIL: usize = core::mem::size_of::<NonNull<u8>>();
#[derive(Debug)]
struct Ident {
    head: u64,
    tail: [u8; TAIL],
}
fn main() {
    let id = Ident { head: 1, tail: [0; TAIL] };
    let _ = format!("{id:?}");
}
"#,
    );
}

#[test]
fn self_host_borrow_patterns_accept_valid_and_reject_unsound() {
    check_passes(
        "closure_param_static_field_read",
        r#"
struct Cfg { target: &'static str }
fn with_cfg<R>(f: impl FnOnce(&Cfg) -> R) -> R {
    let cfg = Cfg { target: "x86_64" };
    f(&cfg)
}
fn main() {
    let t: &'static str = with_cfg(|c| c.target);
    assert_eq!(t, "x86_64");
}
"#,
    );
    check_passes(
        "mutex_guard_take_by_value_handle",
        r#"
use std::sync::{Arc, Mutex};
struct Handle<'s, T> {
    slot: Arc<Mutex<Option<T>>>,
    _m: std::marker::PhantomData<&'s ()>,
}
impl<'s, T> Handle<'s, T> {
    fn join(self) -> Option<T> {
        let mut guard = self.slot.lock().unwrap();
        guard.take()
    }
}
fn main() {
    let h: Handle<'_, i32> = Handle {
        slot: Arc::new(Mutex::new(Some(7))),
        _m: std::marker::PhantomData,
    };
    assert_eq!(h.join(), Some(7));
}
"#,
    );
    check_fails(
        "mutex_guard_take_local_borrow_escape",
        r#"
use std::sync::Mutex;
fn bad(x: (i32, &i32)) -> Option<&i32> {
    let m = Mutex::new(Some(&x.0));
    let mut g = m.lock().unwrap();
    g.take()
}
fn main() {
    let _ = bad((1, &2));
}
"#,
    );
    check_passes(
        "generic_trait_mem_take_field",
        r#"
struct Stub<'a>(&'a [u32]);
struct Tails<'a> { stubs: Vec<Stub<'a>> }
trait Backend<'a> {
    fn tails(&mut self) -> &mut Tails<'a>;
    fn island(&mut self);
}
fn emit_stubs<'a, E: Backend<'a>>(emit: &mut E) {
    for _stub in std::mem::take(&mut emit.tails().stubs) {
        emit.island();
    }
}
struct S<'a> { t: Tails<'a>, _m: &'a () }
impl<'a> Backend<'a> for S<'a> {
    fn tails(&mut self) -> &mut Tails<'a> { &mut self.t }
    fn island(&mut self) {}
}
fn main() {
    let u = ();
    let mut s = S { t: Tails { stubs: vec![] }, _m: &u };
    emit_stubs(&mut s);
}
"#,
    );
    check_fails(
        "mem_take_local_mut_ref_conflict",
        r#"
fn bad<'a>(r: &'a mut i32) {
    let mut s = Some(&mut *r);
    let t = std::mem::take(&mut s);
    *r = 1;
    drop(t);
}
fn main() {
    let mut x = 0;
    bad(&mut x);
}
"#,
    );
    check_passes(
        "map_collect_origin_only_fork",
        r#"
struct Cx<'a> { s: &'a str, flag: bool }
impl<'a> Cx<'a> {
    fn fork(&self) -> Cx<'a> { Cx { s: self.s, flag: false } }
}
fn run(cx: &mut Cx<'_>) {
    let forks: Vec<Cx<'_>> = (0..2).map(|_| cx.fork()).collect();
    cx.flag = true;
    assert_eq!(forks.len(), 2);
}
fn main() {
    let mut cx = Cx { s: "ok", flag: false };
    run(&mut cx);
}
"#,
    );
    check_fails(
        "map_collect_real_borrow_blocks_mut",
        r#"
fn bad() {
    let mut v = vec![1, 2];
    let refs: Vec<&i32> = (0..1).map(|_| &v[0]).collect();
    v.push(3);
    drop(refs);
}
fn main() { bad(); }
"#,
    );
    check_fails(
        "map_collect_origin_only_cannot_escape_static",
        r#"
struct Cx<'a> { s: &'a str }
impl<'a> Cx<'a> {
    fn fork(&self) -> Cx<'a> { Cx { s: self.s } }
}
fn require_static<T: 'static>(_: T) {}
fn bad(cx: &mut Cx<'_>) {
    let forks: Vec<Cx<'_>> = (0..1).map(|_| cx.fork()).collect();
    require_static(forks);
}
fn main() {
    let s = String::from("hi");
    let mut cx = Cx { s: &s };
    bad(&mut cx);
}
"#,
    );
    check_passes(
        "outer_hrtb_inner_named_lt_closure",
        r#"
struct Cx<'a>(&'a str);
fn quietly<'a, T>(cx: &mut Cx<'a>, run: impl FnOnce(&mut Cx<'a>) -> T) -> T {
    run(cx)
}
fn main() {
    let mut cx = Cx("ok");
    let _ = quietly(&mut cx, |c| c.0);
}
"#,
    );
    check_fails(
        "outer_hrtb_wrong_inner_named_lt",
        r#"
struct Cx<'a>(&'a str);
fn bad<'a, 'b>(cx: &mut Cx<'a>, run: impl FnOnce(&mut Cx<'b>)) {
    run(cx);
}
fn main() {
    let s = String::from("hi");
    let mut cx = Cx(&s);
    bad(&mut cx, |_| {});
}
"#,
    );
    check_passes(
        "named_lt_closure_and_slice_find_map",
        r#"
fn with_slice<'a>(s: &'a [u32], mut visit: impl FnMut(&'a [u32])) {
    visit(s);
}
fn forward<'f>(s: &'f [u32], mut out: impl FnMut(&'f [u32])) {
    with_slice(s, |part| out(part));
}
fn find_it<'a>(items: &'a [u32], pick: &impl Fn(&'a u32) -> Option<&'a u32>) -> Option<&'a u32> {
    items.iter().find_map(|x| match x {
        v => pick(v),
    })
}
fn main() {
    let data = [1u32, 2, 3];
    forward(&data, |p| assert_eq!(p.len(), 3));
    assert_eq!(find_it(&data, &|x| Some(x)), Some(&1));
}
"#,
    );
    check_fails(
        "slice_find_map_wrong_caller_lt",
        r#"
fn bad<'a, 'b>(items: &'a [u32], pick: &impl Fn(&'b u32)) {
    let _: Option<()> = items.iter().find_map(|x| {
        pick(x);
        None
    });
}
fn main() {
    let data = [1u32];
    bad(&data, |_| {});
}
"#,
    );
    check_passes(
        "for_each_ref_pat_static_vec_temp",
        r#"
struct Plan<'a>(Vec<&'a str>);
impl<'a> Plan<'a> {
    fn add(&mut self, s: &'a str) { self.0.push(s); }
    fn run(&mut self) {
        vec!["a", "b"].iter().for_each(|&s| self.add(s));
    }
}
fn main() {
    let mut p = Plan(Vec::new());
    p.run();
    assert_eq!(p.0.len(), 2);
}
"#,
    );
    check_fails(
        "for_each_ref_pat_local_vec_escape",
        r#"
struct Plan<'a>(Vec<&'a str>);
impl<'a> Plan<'a> {
    fn add(&mut self, s: &'a str) { self.0.push(s); }
    fn bad(&mut self) {
        let local = String::from("nope");
        vec![local.as_str()].iter().for_each(|&s| self.add(s));
    }
}
fn main() {
    let mut p = Plan(Vec::new());
    p.bad();
}
"#,
    );
    check_passes(
        "multi_lt_adt_non_shared_field_ok",
        r#"
struct Ctx<'a, 'b> {
    shared: &'a str,
    local: &'b i32,
}
struct Store<'a>(Vec<&'a str>);
impl<'a> Store<'a> {
    fn record(&mut self, ctx: &Ctx<'a, '_>) {
        self.0.push(ctx.shared);
    }
    fn run(&mut self, s: &'a str) {
        let local = 42;
        let ctx = Ctx { shared: s, local: &local };
        self.record(&ctx);
    }
}
fn main() {
    let mut st = Store(Vec::new());
    st.run("ok");
}
"#,
    );
    check_fails(
        "multi_lt_adt_shared_field_rejects_local",
        r#"
struct Ctx<'a, 'b> {
    shared: &'a str,
    local: &'b i32,
}
struct Store<'a>(Vec<&'a str>);
impl<'a> Store<'a> {
    fn record(&mut self, ctx: &Ctx<'a, '_>) {
        self.0.push(ctx.shared);
    }
    fn bad(&mut self) {
        let local_s = String::from("dangling");
        let local_i = 42;
        let ctx = Ctx { shared: &local_s, local: &local_i };
        self.record(&ctx);
    }
}
fn main() {
    let mut st = Store(Vec::new());
    st.bad();
}
"#,
    );
    check_fails(
        "outer_hrtb_mut_invariance_rejects_shorter_inner",
        r#"
struct Cx<'a>(&'a str);
fn bad<'a, 'b: 'a>(cx: &mut Cx<'b>, run: impl FnOnce(&mut Cx<'a>)) {
    run(cx);
}
fn main() {
    let mut cx = Cx("static");
    {
        let s = String::from("short");
        bad(&mut cx, |c| c.0 = &s);
    }
    println!("{}", cx.0);
}
"#,
    );
    check_fails(
        "mut_param_map_collect_field_borrow_blocks_write",
        r#"
struct Cx { s: String }
fn bad(cx: &mut Cx) {
    let refs: Vec<&String> = (0..1).map(|_| &cx.s).collect();
    cx.s = String::new();
    drop(refs);
}
fn main() {
    let mut cx = Cx { s: String::from("a") };
    bad(&mut cx);
}
"#,
    );
    check_fails(
        "multi_lt_enum_variant_escape_rejected",
        r#"
enum Ctx<'a, 'b> {
    A { v: &'b str },
    B { v: &'a str },
}
struct Store<'a>(Vec<&'a str>);
impl<'a> Store<'a> {
    fn record(&mut self, ctx: &Ctx<'a, '_>) {
        if let Ctx::B { v } = ctx {
            self.0.push(*v);
        }
    }
    fn bad(&mut self) {
        let local = String::from("x");
        self.record(&Ctx::B { v: &local });
    }
}
fn main() {
    let mut st = Store(Vec::new());
    st.bad();
}
"#,
    );
    check_fails(
        "multi_lt_generic_field_escape_rejected",
        r#"
struct Ctx<'a, 'b, T> {
    shared: &'a str,
    t: T,
    local: &'b i32,
}
struct Store<'a>(Vec<&'a str>);
impl<'a> Store<'a> {
    fn record(&mut self, ctx: &Ctx<'a, '_, &'a str>) {
        self.0.push(ctx.t);
    }
    fn bad(&mut self, s: &'a str) {
        let local = String::from("x");
        let i = 1;
        self.record(&Ctx { shared: s, t: local.as_str(), local: &i });
    }
}
fn main() {
    let mut st = Store(Vec::new());
    st.bad("ok");
}
"#,
    );
    check_fails(
        "closure_lt_later_arg_escape_rejected",
        r#"
fn with<'a>(a: &'a u32, mut visit: impl FnMut(&'a u32), b: &'a u32) {
    visit(a);
    visit(b);
}
fn keep(_: &'static u32) {}
fn bad(a: &'static u32, b: &u32) {
    with(a, |p| keep(p), b);
}
fn main() {
    static S: u32 = 1;
    let local = 2u32;
    bad(&S, &local);
}
"#,
    );
    check_fails(
        "option_map_env_only_closure_static_rejected",
        r#"
fn require_static<T: 'static>(_: T) {}
fn main() {
    let s = String::from("x");
    let r = &s;
    let c = move || r.len();
    let _ = Some(c).map(|f| require_static(f));
}
"#,
    );
    check_fails(
        "fn_ptr_inner_non_static_outer_hrtb_rejected",
        r#"
fn bad<'a>(x: &fn() -> &'a i32, f: impl FnOnce(&fn() -> &'static i32)) {
    f(x);
}
fn main() {
    static N: i32 = 1;
    fn get() -> &'static i32 { &N }
    let fp: fn() -> &'static i32 = get;
    bad(&fp, |_| {});
}
"#,
    );
    check_fails(
        "tuple_sources_double_outer_hrtb_peel_rejected",
        r#"
fn bad<'a>(s: &'a str, f: impl FnOnce((&&'static str, i32))) {
    let t = (&s, 0);
    f(t);
}
fn main() {
    let local = String::from("dangling");
    bad(&local, |(r, _)| {
        let _: &'static str = *r;
    });
}
"#,
    );
    check_fails(
        "result_map_preserves_err_borrow_uaf_rejected",
        r#"
fn bad<'a>() -> Result<(), &'a str> {
    let local = String::from("dangling");
    let r: Result<i32, &str> = Err(&local);
    r.map(|_| ())
}
fn main() {
    let _ = bad();
}
"#,
    );
    check_fails(
        "closure_bail_ret_sets_keeps_real_mut_borrow",
        r#"
struct Cx { buf: String }
fn bad(cx: &mut Cx) {
    let refs: Vec<&mut String> = (0..1)
        .map(|_| {
            let p = &mut cx.buf;
            p
        })
        .collect();
    cx.buf = String::new();
    drop(refs);
}
fn main() {
    let mut cx = Cx { buf: String::from("a") };
    bad(&mut cx);
}
"#,
    );
    check_fails(
        "multi_lt_adt_reassigned_field_local_rejected",
        r#"
struct Ctx<'a, 'b> { shared: &'a str, local: &'b i32 }
struct Store<'a>(Vec<&'a str>);
impl<'a> Store<'a> {
    fn record(&mut self, ctx: &Ctx<'a, '_>) { self.0.push(ctx.shared); }
    fn bad(&mut self) {
        let local_s = String::from("dangling");
        let mut v = local_s.as_str();
        let local_i = 42;
        let ctx = Ctx { shared: v, local: &local_i };
        v = "ok";
        self.record(&ctx);
    }
}
fn main() {
    let mut st = Store(Vec::new());
    st.bad();
}
"#,
    );
    check_fails(
        "multi_lt_adt_outlives_bound_field_rejected",
        r#"
struct Ctx<'a, 'b> { shared: &'a str, sub: &'b str }
struct Store<'a>(Vec<&'a str>);
impl<'a> Store<'a> {
    fn record<'b: 'a>(&mut self, ctx: &Ctx<'a, 'b>) { self.0.push(ctx.sub); }
    fn bad(&mut self) {
        let local_s = String::from("dangling");
        self.record(&Ctx { shared: "ok", sub: &local_s });
    }
}
fn main() {
    let mut st = Store(Vec::new());
    st.bad();
}
"#,
    );
    check_fails(
        "multi_lt_adt_type_alias_swapped_rejected",
        r#"
struct Ctx<'first, 'second> { a: &'first str, b: &'second str }
type Swapped<'x, 'y> = Ctx<'y, 'x>;
struct Store<'a>(Vec<&'a str>);
impl<'a> Store<'a> {
    fn record(&mut self, ctx: &Swapped<'_, 'a>) { self.0.push(ctx.a); }
    fn bad(&mut self) {
        let local = String::from("dangling");
        self.record(&Ctx { a: &local, b: "ok" });
    }
}
fn main() {
    let mut st = Store(Vec::new());
    st.bad();
}
"#,
    );
    check_fails(
        "closure_lt_later_outlives_and_trait_bound_rejected",
        r#"
fn with<'a, 'b: 'a>(a: &'a u32, mut f: impl FnMut(&'a u32), b: &'b u32) {
    f(a);
    f(b);
}
fn bad_outlives<'c>(long: &'c u32, keep: &impl Fn(&'c u32)) {
    let local = 42u32;
    with(long, |p| keep(p), &local);
}
fn main() {
    let x = 1u32;
    bad_outlives(&x, &|_| {});
}
"#,
    );
    check_fails(
        "closure_lt_later_trait_bound_rejected",
        r#"
trait Get<'a> { fn get(&self) -> &'a u32; }
impl<'a> Get<'a> for &'a u32 { fn get(&self) -> &'a u32 { self } }
fn with_tr<'a, T: Get<'a>>(a: &'a u32, mut f: impl FnMut(&'a u32), b: T) {
    f(a);
    f(b.get());
}
fn bad_trait<'c>(long: &'c u32, keep: &impl Fn(&'c u32)) {
    let local = 42u32;
    with_tr(long, |p| keep(p), &local);
}
fn main() {
    let x = 1u32;
    bad_trait(&x, &|_| {});
}
"#,
    );
    check_fails(
        "let_tuple_infer_elem_moves_mut_ref",
        "fn main() { let mut x = 1; let r = &mut x; let (_a, _b): (_, i32) = (r, 0); *r += 1; }",
    );
    check_passes(
        "struct_field_tuple_reborrows_mut_refs",
        "struct Pair<'a> { f: (&'a mut i32, &'a mut i32) }\nfn main() { let (mut a, mut b) = (1, 2); let (r1, r2) = (&mut a, &mut b); let _p = Pair { f: (r1, r2) }; *r1 += 1; *r2 += 1; }",
    );
    check_fails(
        "overloaded_deref_temp_field_preserves_container_borrow",
        "use std::ops::Deref;\nstruct Inner { x: i32 }\nstruct Wrap<'a>(&'a mut Inner);\nimpl<'a> Deref for Wrap<'a> { type Target = Inner; fn deref(&self) -> &Inner { &*self.0 } }\nfn take_two(_: &i32, _: &mut Inner) {}\nfn main() { let mut inner = Inner { x: 1 }; take_two(&Wrap(&mut inner).x, &mut inner); }",
    );
    check_passes(
        "recursive_assoc_bounds_distinguish_trait_args",
        "struct S; struct Zero; struct Succ<T>(T);\ntrait Tr<A> { type Item; }\nimpl<A> Tr<A> for S { type Item = S; }\ntrait Step<N> {}\nimpl Step<Zero> for S {}\nimpl<N> Step<Succ<N>> for S where S: Tr<N, Item: Step<N>> {}\nfn check<T: Step<Succ<Succ<Zero>>>>() {}\nfn main() { check::<S>(); }",
    );
    check_fails(
        "swapped_result_alias_try_does_not_peel_wrong_tuple_elem",
        "mod m { pub type Result<E, T> = std::result::Result<T, E>; }\nfn split<'a, 'b>(a: &'a str, b: &'b str) -> m::Result<(&'a str, &'a str), (&'b str, &'a str)> { Ok((b, a)) }\nfn bad<'a, 'b>(a: &'a str, b: &'b str) -> std::result::Result<&'a str, (&'a str, &'a str)> { let (x, _y) = split(a, b)?; Ok(x) }\nfn main() { let _ = bad(\"a\", \"b\"); }",
    );
}

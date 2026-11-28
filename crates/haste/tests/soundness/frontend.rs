use super::{check_fails, check_fails_borrow, check_passes};

#[test]
fn rejects_unmet_concrete_and_generic_trait_bounds() {
    check_fails(
        "trait_not_impl_custom",
        r#"
trait Need {
    fn need(&self) -> i32;
}
struct NoImpl;
fn use_need<T: Need>(x: T) -> i32 {
    x.need()
}
fn main() {
    let _ = use_need(NoImpl);
}
"#,
    );
    check_fails(
        "trait_not_impl_clone",
        r#"
struct NoClone(i32);
fn dup<T: Clone>(x: &T) -> T {
    x.clone()
}
fn main() {
    let v = NoClone(1);
    let _ = dup(&v);
}
"#,
    );
    check_fails(
        "trait_not_impl_copy",
        r#"
struct NoCopy(String);
fn copy_it<T: Copy>(x: T) -> (T, T) {
    (x, x)
}
fn main() {
    let _ = copy_it(NoCopy(String::from("a")));
}
"#,
    );
    check_fails(
        "trait_not_impl_inherent_impl_bound",
        r#"
trait Marker {}
struct Wrap<T>(T);
impl<T: Marker> Wrap<T> {
    fn ping(&self) -> i32 {
        1
    }
}
struct Plain;
fn main() {
    let w = Wrap(Plain);
    let _ = w.ping();
}
"#,
    );
    check_fails(
        "generic_bound_missing_format",
        r#"
fn show<T>(t: T) -> String {
    format!("{}", t)
}
fn main() {
    let _ = show::<i32>;
}
"#,
    );
    check_fails(
        "generic_bound_missing_ref_call",
        r#"
fn need<U: std::fmt::Display>(_: &U) {}
fn show<T>(t: T) {
    need(&t);
}
fn main() {
    let _ = show::<i32>;
}
"#,
    );
    check_fails(
        "generic_method_missing_unused",
        r#"
fn call_missing<T>(x: T) {
    x.does_not_exist();
}
fn main() {}
"#,
    );
    check_fails(
        "generic_wrap_missing_bound",
        r#"
trait Need {}
struct Wrap<T>(T);
fn need<U: Need>(_: U) {}
fn bad<T>(x: Wrap<T>) {
    need(x);
}
fn main() {}
"#,
    );
    check_fails(
        "inherent_impl_where_predicate_bound",
        r#"
trait Marker {}
struct Wrap<T>(T);
impl<T> Wrap<T>
where
    Wrap<T>: Marker,
{
    fn ping(&self) -> i32 {
        1
    }
}
struct Plain;
fn main() {
    let w = Wrap(Plain);
    let _ = w.ping();
}
"#,
    );
    check_fails(
        "inherent_impl_assoc_const_bound",
        r#"
trait Marker {}
struct Wrap<T>(T);
impl<T: Marker> Wrap<T> {
    const C: i32 = 1;
}
struct Plain;
fn main() {
    let _ = Wrap::<Plain>::C;
}
"#,
    );
}

#[test]
fn rejects_const_eval_overflow_and_invalid_arithmetic() {
    check_fails(
        "const_u8_add_overflow",
        r#"
const X: u8 = 200 + 100;
fn main() {
    let _ = X;
}
"#,
    );
    check_fails(
        "const_i8_neg_min_overflow",
        r#"
const X: i8 = -i8::MIN;
fn main() {
    let _ = X;
}
"#,
    );
    check_fails(
        "const_shl_overflow",
        r#"
const X: u32 = 1 << 32;
fn main() {
    let _ = X;
}
"#,
    );
    check_fails(
        "const_div_by_zero",
        r#"
const X: i32 = 1 / 0;
fn main() {
    let _ = X;
}
"#,
    );
    check_fails(
        "const_signed_div_min_neg1",
        r#"
const X: i8 = i8::MIN / -1;
fn main() {
    let _ = X;
}
"#,
    );
    check_fails(
        "const_i128_neg_min_no_panic",
        r#"
const X: i128 = -i128::MIN;
fn main() {
    let _ = X;
}
"#,
    );
    check_fails(
        "const_pow_overflow",
        r#"
const X: u8 = 2u8.pow(8);
fn main() {
    let _ = X;
}
"#,
    );
    check_fails(
        "const_ilog2_zero",
        r#"
const X: u32 = 0u32.ilog2();
fn main() {
    let _ = X;
}
"#,
    );
    check_fails(
        "static_add_overflow",
        r#"
static S: u8 = 200 + 100;
fn main() {
    let _ = S;
}
"#,
    );
    check_fails(
        "const_tuple_overflow",
        r#"
const T: (u8,) = (255 + 1,);
fn main() {
    let _ = T;
}
"#,
    );
    check_fails(
        "const_array_elem_overflow",
        r#"
const A: [u8; 1] = [255 + 1];
fn main() {
    let _ = A;
}
"#,
    );
    check_fails(
        "const_array_len_overflow",
        r#"
fn main() {
    let _: [u8; (200u8 + 100u8) as usize] = [0; 44];
}
"#,
    );
}

#[test]
fn rejects_named_lifetime_return_subtyping_violations() {
    check_fails_borrow(
        "lifetime_longer_unrelated",
        r#"
fn bad<'a, 'b>(x: &'a str) -> &'b str {
    x
}
fn main() {
    let _ = bad("a");
}
"#,
    );
    check_fails_borrow(
        "lifetime_longer_elided_param",
        r#"
fn bad<'a>(x: &str) -> &'a str {
    x
}
fn main() {
    let _ = bad("a");
}
"#,
    );
    check_fails_borrow(
        "lifetime_longer_impl_method",
        r#"
struct Hold<'a>(&'a str);
impl<'a> Hold<'a> {
    fn bad<'b>(&self) -> &'b str {
        self.0
    }
}
fn main() {
    let h = Hold("a");
    let _ = h.bad();
}
"#,
    );
    check_fails_borrow(
        "lifetime_longer_nested_ref_plain_return",
        r#"
fn bad<'a, 'b>(x: &'a &'b str) -> &'b &'b str {
    x
}
fn main() {
    let s = "a";
    let _ = bad(&s);
}
"#,
    );
    check_fails_borrow(
        "hof_closure_early_return_local_escape",
        r#"
fn run_hof<T>(mut f: impl FnMut() -> Option<T>) -> Option<T> {
    f()
}
fn bad(flag: bool) -> Option<&'static usize> {
    let local = 42usize;
    run_hof(|| {
        if flag {
            return Some(&local);
        }
        None
    })
}
fn main() {
    let _ = bad(true);
}
"#,
    );
    check_fails_borrow(
        "hof_closure_tail_return_local_escape",
        r#"
fn run_hof<T>(mut f: impl FnMut() -> Option<T>) -> Option<T> {
    f()
}
fn bad(flag: bool) -> Option<&'static usize> {
    let local = 42usize;
    run_hof(|| {
        if flag {
            return None;
        }
        Some(&local)
    })
}
fn main() {
    let _ = bad(false);
}
"#,
    );
    check_fails_borrow(
        "lifetime_mut_ref_invariance_return",
        r#"
fn bad<'a, 'b: 'a>(x: &'a mut &'b str) -> &'a mut &'a str {
    x
}
fn main() {
    let mut s = "a";
    let _ = bad(&mut s);
}
"#,
    );
    check_fails_borrow(
        "lifetime_tuple_swap_return",
        r#"
fn bad<'a, 'b>(x: &'a str, y: &'b str) -> (&'a str, &'b str) {
    (y, x)
}
fn main() {
    let _ = bad("a", "b");
}
"#,
    );
    check_fails_borrow(
        "lifetime_option_generic_return",
        r#"
fn bad<'a, 'b, T>(x: &'a T) -> Option<&'b T> {
    Some(x)
}
fn main() {
    let _ = bad(&1);
}
"#,
    );
    check_fails_borrow(
        "lifetime_rebind_outer_ref_return",
        r#"
fn bad<'a, 'b>(x: &'a &'b str) -> &'b &'b str {
    let y = x;
    y
}
fn main() {
    let s = "a";
    let _ = bad(&s);
}
"#,
    );
    check_fails_borrow(
        "lifetime_owned_field_behind_short_ref",
        r#"
struct Hold<'a> {
    s: &'a str,
    buf: String,
}
fn bad<'short, 'long>(x: &'short Hold<'long>) -> &'long str {
    let _ = x.s;
    &x.buf
}
fn main() {
    let h = Hold { s: "a", buf: String::from("b") };
    let _ = bad(&h);
}
"#,
    );
    check_fails_borrow(
        "lifetime_inner_elided_pair_return",
        r#"
fn bad<'a, 'b>(x: &'b (&'a str, &'_ str)) -> (&'a str, &'a str) {
    *x
}
fn main() {
    let p = ("a", "b");
    let _ = bad(&p);
}
"#,
    );
}

#[test]
fn accepts_valid_bounds_const_and_named_lifetime_patterns() {
    check_passes(
        "wave3_valid_controls",
        r#"
use std::fmt::Display;
use std::ops::Deref;

const ADD_OK: u8 = 100 + 100;
const WRAPPING_OK: u8 = 200u8.wrapping_add(100);
const SHL_OK: u32 = 1 << 31;
const NEG_I8_MIN: i8 = -128;
const NEG_I128: i128 = -1;
const DIV_I128: i128 = 10 / 2;
const SHL_I8: i8 = 1 << 7;
const SHR_NEG_I8: i8 = -8 >> 1;

#[repr(i8)]
enum MinEnum {
    A = -128,
}

trait BlanketExt {
    fn ext(&self) -> i32;
}
impl<T> BlanketExt for T {
    fn ext(&self) -> i32 {
        1
    }
}
fn use_blanket<T>(x: T) -> i32 {
    x.ext()
}
fn use_ord_eq<T: Ord>(a: &T, b: &T) -> bool {
    a == b
}
fn dup_clone<T: Clone>(x: &T) -> T {
    x.clone()
}

struct Chain<'x, 'y>(&'x str, &'y str);
impl<'x: 'y, 'y> Chain<'x, 'y> {
    fn pass<'a: 'x>(&self, s: &'a str) -> &'y str {
        let _ = (self.0, self.1);
        s
    }
}

fn show_ok<T: Display>(t: T) -> String {
    format!("{}", t)
}

fn deref_str<T: Deref<Target = str>>(x: &T) -> &str {
    x
}

fn outlives_param<'a: 'b, 'b>(x: &'a str) -> &'b str {
    x
}

fn outlives_where<'a, 'b>(x: &'a str) -> &'b str
where
    'a: 'b,
{
    x
}

struct Ctx<'a>(&'a str);
fn outer_named_inner_elided<'c>(cx: &'c Ctx<'_>) -> &'c Ctx<'_> {
    cx
}

struct Two<'c, 'a>(&'c str, &'a str);
impl<'c, 'a> Two<'c, 'a> {
    fn second(&mut self) -> Option<&'a str> {
        Some(self.1)
    }
    fn into_second(self) -> &'a str {
        self.1
    }
}

fn take_second<'c, 'a>(t: Two<'c, 'a>) -> &'a str {
    t.1
}

struct MixedImpl<'c, 'a>(&'c i32, &'a str);
impl<'a> MixedImpl<'_, 'a> {
    fn text(&self) -> &'a str {
        let _ = self.0;
        self.1
    }
}

struct FileData {
    text: String,
}
struct Holder<'a> {
    data: &'a FileData,
}
impl<'a> Holder<'a> {
    fn inner(&self) -> Option<&'a FileData> {
        Some(self.data)
    }
    fn inner_text(&self) -> &'a String {
        &self.inner().expect("present").text
    }
}

#[derive(Clone, Copy)]
struct Tag(u32);
fn zip_copied_args<'a>(names: &'a [&'a str], tags: &[Tag]) -> Vec<(&'a str, Tag)> {
    names.iter().copied().zip(tags.iter().copied()).collect()
}

fn run_hof<T>(mut f: impl FnMut() -> Option<T>) -> Option<T> {
    f()
}

fn ok_hof_early_return(key: usize) -> Option<&'static str> {
    run_hof(|| {
        if key == 0 {
            return None;
        }
        Some("ok")
    })
}

fn main() {
    let _ = (
        ADD_OK,
        WRAPPING_OK,
        SHL_OK,
        NEG_I8_MIN,
        NEG_I128,
        DIV_I128,
        SHL_I8,
        SHR_NEG_I8,
        MinEnum::A as i8,
    );
    let _ = (use_blanket(1), use_ord_eq(&1, &1));
    let cl = || 1i32;
    let _ = dup_clone(&cl);
    let ch = Chain("a", "b");
    let _ = ch.pass("c");
    let _ = show_ok(42);
    let s = String::from("hello");
    let _ = deref_str(&s);
    let _ = outlives_param(&s);
    let _ = outlives_where(&s);
    let c = Ctx(&s);
    let _ = outer_named_inner_elided(&c);
    let mut t = Two("a", "b");
    let _ = t.second();
    let _ = t.into_second();
    let _ = take_second(Two("a", "b"));
    let one = 1i32;
    let mixed = MixedImpl(&one, "ok");
    let _ = mixed.text();
    let fd = FileData {
        text: String::from("ok"),
    };
    let h = Holder { data: &fd };
    let _ = h.inner_text();
    let _ = zip_copied_args(&["k"], &[Tag(1)]);
    let _ = ok_hof_early_return(1);
}
"#,
    );
}

#[test]
fn cmark_and_ruff_borrowck_patterns() {
    check_passes(
        "tuple_closure_param_mut_ref_reborrow",
        r#"
fn bump<'a>(opt: Option<(usize, &'a mut i32)>) -> Option<(usize, &'a mut i32)> {
    opt.and_then(|item| Some((item.0 + 1, item.1)))
}
fn main() {
    let mut x = 1;
    if let Some((_, r)) = bump(Some((0, &mut x))) {
        *r += 1;
    }
    assert_eq!(x, 2);
}
"#,
    );

    check_passes(
        "option_map_store_by_val_borrow_into_self",
        r#"
struct Attrs<'a>(&'a str);
struct Store<'a>(Vec<Attrs<'a>>);
impl<'a> Store<'a> {
    fn push_opt(&mut self, attrs: Option<Attrs<'a>>) -> Option<usize> {
        attrs.map(|a| {
            self.0.push(a);
            self.0.len()
        })
    }
}
fn main() {
    let mut s = Store(Vec::new());
    assert_eq!(s.push_opt(Some(Attrs("hi"))), Some(1));
}
"#,
    );

    check_passes(
        "option_map_unwrap_or_else_peels_outer_ref",
        r#"
use std::borrow::Cow;
struct Entry<'a> {
    dest: Cow<'a, str>,
    title: Option<Cow<'a, str>>,
}
struct Map<'a>(Entry<'a>);
impl<'a, 'b, 's> Map<'a>
where
    's: 'b,
{
    fn get(&'s self, _key: &'b str) -> Option<&'b Entry<'a>> {
        Some(&self.0)
    }
}
fn lookup<'a>(map: &Map<'a>, label: Cow<'a, str>) -> Option<(Cow<'a, str>, Cow<'a, str>)> {
    map.get(label.as_ref()).map(|e| {
        let title = e.title.as_ref().cloned().unwrap_or_else(|| "".into());
        let url = e.dest.clone();
        (url, title)
    })
}
fn main() {
    let m = Map(Entry {
        dest: Cow::Borrowed("u"),
        title: None,
    });
    assert!(lookup(&m, Cow::Borrowed("k")).is_some());
}
"#,
    );

    check_passes(
        "generic_fnmut_callback_named_output_lt",
        r#"
use std::borrow::Cow;
trait Cb<'a> {
    fn call(&mut self, input: Cow<'a, str>) -> Option<Cow<'a, str>>;
}
struct Wrap<F>(Option<F>);
impl<'a, F> Cb<'a> for Wrap<F>
where
    F: FnMut(Cow<'a, str>) -> Option<Cow<'a, str>>,
{
    fn call(&mut self, input: Cow<'a, str>) -> Option<Cow<'a, str>> {
        self.0.as_mut().and_then(|cb| cb(input))
    }
}
fn main() {
    let mut w = Wrap(Some(|s: Cow<'_, str>| Some(s)));
    assert_eq!(w.call(Cow::Borrowed("ok")).as_deref(), Some("ok"));
}
"#,
    );

    check_passes(
        "double_ended_iter_elided_impl_self_next_back",
        r#"
struct SliceIter<'a>(std::slice::Iter<'a, i32>);
impl<'a> Iterator for SliceIter<'a> {
    type Item = &'a i32;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}
impl DoubleEndedIterator for SliceIter<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back()
    }
}
fn main() {
    let data = [1, 2, 3];
    assert_eq!(SliceIter(data.iter()).last(), Some(&3));
}
"#,
    );

    check_passes(
        "match_mut_iter_next_unwraps_container_loans",
        r#"
enum Parts<'a> {
    Slice(std::slice::Iter<'a, &'a str>),
}
fn make_iter<'a>(xs: &'a [&'a str]) -> impl Iterator<Item = &'a str> {
    let mut parts = Parts::Slice(xs.iter());
    std::iter::from_fn(move || match &mut parts {
        Parts::Slice(it) => it.next().copied(),
    })
}
fn main() {
    let xs = ["a", "b"];
    let out: Vec<&str> = make_iter(&xs).collect();
    assert_eq!(out, vec!["a", "b"]);
}
"#,
    );

    check_fails_borrow(
        "option_map_store_local_borrow_into_self_rejected",
        r#"
struct Attrs<'a>(&'a str);
struct Store<'a>(Vec<Attrs<'a>>);
impl<'a> Store<'a> {
    fn bad(&mut self) {
        let local = String::from("dangling");
        let opt = Some(Attrs(local.as_str()));
        opt.map(|a| self.0.push(a));
    }
}
fn main() {
    let mut s = Store(Vec::new());
    s.bad();
}
"#,
    );

    check_fails_borrow(
        "generic_fnmut_callback_short_output_lt_rejected",
        r#"
fn bad<'a, 'b, F: FnMut() -> &'b str>(cb: &mut Option<F>) -> Option<&'a str> {
    cb.as_mut().and_then(|f| Some(f()))
}
fn main() {
    let local = String::from("short");
    let mut cb = Some(|| local.as_str());
    let _ = bad(&mut cb);
}
"#,
    );

    check_fails_borrow(
        "match_mut_iter_next_local_borrow_still_rejected",
        r#"
enum Parts<'a> {
    Slice(std::slice::Iter<'a, &'a str>),
}
fn bad<'a>() -> Option<&'a str> {
    let local = String::from("dangling");
    let arr = [local.as_str()];
    let mut parts = Parts::Slice(arr.iter());
    match &mut parts {
        Parts::Slice(it) => it.next().copied(),
    }
}
fn main() {
    let _ = bad();
}
"#,
    );
}

struct Flags(<Flags as Public>::Internal);

trait Public {
    type Internal;
}

const _: () = {
    #[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
    pub struct Hidden(u32);

    impl Public for Flags {
        type Internal = Hidden;
    }

    impl Flags {
        pub fn new(n: u32) -> Self {
            Flags(Hidden(n))
        }

        pub fn same(&self, other: &Self) -> bool {
            self.0.clone() == other.0
        }
    }
};

#[derive(PartialEq, Eq, Clone, Copy)]
struct Tag([u8; 4]);

impl Tag {
    const HEAD: Tag = Tag(*b"HEAD");
    const BODY: Tag = Tag(*b"BODY");
}

fn tag_value(tag: Tag) -> i64 {
    match tag {
        Tag::HEAD => 1,
        Tag::BODY => 2,
        _ => 3,
    }
}

struct Chain {
    value: i64,
    next: &'static Chain,
}

static LINK: Chain = Chain { value: 7, next: &LINK };

pub fn case_const_block_derive(n: i64) -> i64 {
    i64::from(Flags::new(n as u32).same(&Flags::new(n as u32))) * 2 + i64::from(Flags::new(1).same(&Flags::new(2)))
}

pub fn case_const_pattern(n: i64) -> i64 {
    let tag = if n > 0 { Tag::HEAD } else if n == 0 { Tag::BODY } else { Tag(*b"none") };
    let name = if n > 2 { "alpha" } else { "beta" };
    let text = match name {
        "alpha" => 10,
        "beta" => 20,
        _ => 30,
    };
    tag_value(tag) + text
}

pub fn case_tuple_param(n: i64) -> i64 {
    let pairs = [(n, 1_i64), (2, 3)];
    let total: i64 = pairs.iter().map(|(a, b)| a + b).sum();
    let whole = pairs.iter().copied().map(|pair: (i64, i64)| pair.0 * pair.1).fold(0, |sum, item| sum + item);
    let single: fn((i64,)) -> i64 = |wrapped| wrapped.0 + 1;
    let boxed: Box<dyn Fn((i64, i64)) -> i64> = Box::new(|pair| pair.0 - pair.1);
    total + whole + single((n,)) + boxed((n, 2))
}

pub fn case_self_static(n: i64) -> i64 {
    LINK.next.next.value + n
}

pub fn case_unicode(n: i64) -> i64 {
    let letters = ['a', '\u{e9}', '\u{4e2d}', '1', ' ', '\u{1F600}'];
    let picked = letters[(n.unsigned_abs() % 6) as usize];
    i64::from(picked.is_alphabetic()) + i64::from(picked.is_numeric()) * 2 + i64::from(picked.is_whitespace()) * 4 + i64::from(picked.is_uppercase()) * 8
}

pub fn case_fence(n: i64) -> i64 {
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
    n + 1
}

pub fn case_inline_const(n: i64) -> i64 {
    let limit = const {
        let mut total = 0;
        let mut i = 0;
        while i < 4 {
            total += i;
            i += 1;
        }
        total
    };
    n + limit
}

std::thread_local! {
    static LOCAL_COUNT: std::cell::Cell<i64> = const { std::cell::Cell::new(0) };
    static NAMES: std::cell::RefCell<Vec<String>> = std::cell::RefCell::new(Vec::new());
}

pub fn case_println(n: i64) -> i64 {
    println!("hello {n} {:?}", Some("x"));
    eprintln!("note {n}");
    n + 1
}

pub fn case_thread_local(n: i64) -> i64 {
    LOCAL_COUNT.with(|cell| cell.set(n));
    NAMES.with(|names| names.borrow_mut().push(n.to_string()));
    LOCAL_COUNT.with(std::cell::Cell::get) * 2 + NAMES.with(|names| names.borrow().last().map_or(0, |text| text.len() as i64))
}

pub fn case_files(n: i64) -> i64 {
    let manifest = std::fs::read_to_string("/etc/hosts").map_or(-1, |text| i64::from(!text.is_empty()));
    let home = std::env::var("HOME").map_or(0, |value| i64::from(!value.is_empty()));
    let elapsed = std::time::Instant::now().elapsed().as_secs() as i64;
    manifest + home + elapsed + n
}

pub fn case_sort_strings(n: i64) -> i64 {
    let mut names: Vec<String> = (0..20).map(|i| format!("item{}", (i * 7 + n) % 11)).collect();
    names.sort();
    names.dedup();
    let map: std::collections::BTreeMap<String, usize> = names.iter().cloned().enumerate().map(|(i, name)| (name, i)).collect();
    names.join(",").len() as i64 + map.len() as i64 + map.values().sum::<usize>() as i64
}

pub fn case_thread(n: i64) -> i64 {
    let handle = std::thread::spawn(move || n * 2);
    handle.join().unwrap_or(0) + 1
}

struct ZeroGuard<'a>(&'a std::cell::Cell<i64>);

struct Zero;

static ZERO_DROPS: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);

impl Drop for Zero {
    fn drop(&mut self) {
        ZERO_DROPS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
}

impl Drop for ZeroGuard<'_> {
    fn drop(&mut self) {
        self.0.set(self.0.get() + 1);
    }
}

pub fn case_zero_sized_drop(n: i64) -> i64 {
    let before = ZERO_DROPS.load(std::sync::atomic::Ordering::SeqCst);
    let kept = Zero;
    std::mem::forget(kept);
    let moved = Zero;
    let holder = (moved,);
    drop(holder);
    let cell = std::cell::Cell::new(n);
    std::mem::forget(ZeroGuard(&cell));
    {
        let _guard = ZeroGuard(&cell);
    }
    ZERO_DROPS.load(std::sync::atomic::Ordering::SeqCst) - before + cell.get() * 10
}

struct OnDrop<'a> {
    out: &'a mut i64,
    local: i64,
}

impl Drop for OnDrop<'_> {
    fn drop(&mut self) {
        *self.out = self.local;
    }
}

pub fn case_closure_owns_guard(n: i64) -> i64 {
    let mut out = 0;
    {
        let mut guard = OnDrop { out: &mut out, local: n };
        let mut call = move |x: i64| {
            let held = &mut guard;
            held.local += x;
        };
        call(1);
        call(2);
    }
    out
}

pub fn case_sort_dedup(n: i64) -> i64 {
    let mut v: Vec<i64> = vec![n, 3, 3, 1];
    v.sort();
    let sorted = v[0] * 100 + v[3];
    v.dedup();
    sorted * 100 + v.len() as i64
}

pub fn case_collect_strings(n: i64) -> i64 {
    let names: Vec<String> = (0..20).map(|i| format!("item{}", (i * 7 + n) % 11)).collect();
    names.len() as i64
}

pub fn case_btree_collect(n: i64) -> i64 {
    let names = vec![format!("a{n}"), String::from("b")];
    let map: std::collections::BTreeMap<String, usize> = names.iter().cloned().enumerate().map(|(i, name)| (name, i)).collect();
    map.len() as i64 * 100 + map.values().sum::<usize>() as i64
}

const COMPOUND_SHIFT: i32 = -3;
const COMPOUND_BIG: u64 = 1 << 40;

const fn compound_neg_i8() -> i8 {
    -3
}

const fn compound_neg_i32() -> i32 {
    -3
}

const fn compound_add_param<const N: i32>() -> i64 {
    ((5 + 5) + N) as i64
}

const COMPOUND_ADD: i64 = ((5 + 5) + COMPOUND_SHIFT) as i64;
const COMPOUND_CMP: i64 = if (1 + 1) < COMPOUND_BIG { 10 } else { 20 };
const COMPOUND_SHL: i64 = if (1 << 35) > 1_u64 { 30 } else { 40 };
const COMPOUND_SHR: i64 = (((-2) >> 1) as u16) as i64;
const COMPOUND_CALL: i64 = compound_neg_i8().count_ones() as i64;
const COMPOUND_CALL_ADD: i64 = ((5 + 5) + compound_neg_i32()) as i64;
const COMPOUND_PARAM: i64 = compound_add_param::<COMPOUND_SHIFT>();

#[repr(i64)]
enum CompoundConst {
    Total = COMPOUND_ADD + COMPOUND_CMP + COMPOUND_SHL + COMPOUND_SHR + COMPOUND_CALL + COMPOUND_CALL_ADD + COMPOUND_PARAM,
}

pub fn case_const_compound_types(n: i64) -> i64 {
    CompoundConst::Total as i64 + n
}

macro_rules! mq_kind {
    (3) => { 1 };
    ($t:ty) => { 2 };
    ($e:expr) => { 3 };
}

macro_rules! mq_pat_or_expr {
    ($e:expr) => { 10 };
    ($p:pat) => { 20 };
}

macro_rules! mq_vis_or_ident {
    ($id:ident) => { 100 };
    ($v:vis) => { 200 };
}

macro_rules! mq_lit_or_expr {
    ($l:literal) => { 2 };
    ($e:expr) => { 9 };
}

macro_rules! mq_path_or_tt {
    ($p:path) => { 3 };
    ($t:tt) => { 9 };
}

macro_rules! mq_pat_target {
    ($p:pat) => { 1 };
    ($t:tt) => { 9 };
}

macro_rules! mq_vis_ty {
    ($v:vis $t:ty) => { 4 };
}

macro_rules! mq_tt {
    ($t:tt) => { mq_kind!($t) };
}

macro_rules! mq_tts {
    ($($t:tt)*) => { mq_kind!($($t)*) };
}

macro_rules! mq_from_expr {
    ($e:expr) => { mq_kind!($e) + mq_tt!($e) * 10 + mq_tts!($e) * 100 };
}

macro_rules! mq_from_pat {
    ($p:pat) => { mq_pat_or_expr!($p) };
}

macro_rules! mq_from_vis {
    ($v:vis) => { mq_vis_or_ident!($v) };
}

macro_rules! mq_compat {
    ($pp:pat_param, $lit:expr, $ty:ty) => {
        mq_pat_target!($pp) + mq_lit_or_expr!($lit) + mq_path_or_tt!($ty) + mq_vis_ty!($ty)
    };
}

pub fn case_macro_opaque_forward(n: i64) -> i64 {
    let a = mq_from_expr!(i32);
    let b = mq_from_expr!(3);
    let c = mq_from_pat!(Some(1));
    let d = mq_from_vis!(pub);
    let e = mq_compat!(x, 42, Vec<i32>);
    a + b + c + d + e + n
}

static LAZY_TABLE: std::sync::LazyLock<[i64; 4]> = std::sync::LazyLock::new(|| std::array::from_fn(|i| i as i64 * 3));

pub fn case_lazy_table(n: i64) -> i64 {
    LAZY_TABLE[(n.unsigned_abs() % 4) as usize] + n
}

pub fn case_wide_float(n: i64) -> i64 {
    let x = n as f64 * 1.5e30;
    let y = n as f32 * 2.5;
    let down = (x as i128 / 1_000_000_000_000_000_000_000_000) as i64;
    let saturated = (x * 1e30) as u128 == u128::MAX;
    let back = (n as i128 * 3) as f64 + (n as u128 % 7) as f32 as f64;
    down + (y as i128) as i64 + saturated as i64 + back as i64
}

struct Plain(i64);

impl Iterator for Plain {
    type Item = i64;
    fn next(&mut self) -> Option<i64> {
        self.0 -= 1;
        (self.0 > 0).then_some(self.0)
    }
}

pub fn case_default_try_fold(n: i64) -> i64 {
    let limit = n.abs() % 9;
    let folded: Option<i64> = Plain(limit).try_fold(0_i64, |sum, item| sum.checked_add(item));
    let found = Plain(limit).any(|item| item == 2);
    let mapped = Plain(limit).map(|item| item * 2).any(|item| item == 6);
    folded.unwrap_or(-1) + found as i64 + mapped as i64 * 100
}

pub fn case_flat_map_any(n: i64) -> i64 {
    let rows = [vec![1_i64, 2], vec![3], vec![4, 5, 6]];
    let limit = n.abs();
    let hit = rows.iter().enumerate().flat_map(move |(index, row)| {
        row.iter().copied().filter(move |item| *item + index as i64 > limit)
    });
    let chained = [7_i64].iter().copied().chain(hit);
    let any = chained.clone().any(|item| item == limit + 1);
    let sum: i64 = chained.sum();
    sum + any as i64
}

trait Keyed {
    fn key(&self) -> i64;
}

struct Fixed(i64);

impl Keyed for Fixed {
    fn key(&self) -> i64 {
        self.0 * 2
    }
}

pub fn case_dyn_method_item(n: i64) -> i64 {
    let held = Fixed(n);
    let some: Option<&dyn Keyed> = Some(&held);
    let none: Option<&dyn Keyed> = None;
    let as_pointer: fn(&(dyn Keyed + 'static)) -> i64 = <dyn Keyed as Keyed>::key;
    some.map_or(1, Keyed::key) + none.map_or(5, <dyn Keyed>::key) + as_pointer(&held)
}

#[derive(Clone, Copy)]
enum Slot {
    Left,
    Right,
}

pub fn case_array_of_refs(n: i64) -> i64 {
    let (mut a, mut b) = (1.5_f32, 2.5_f32);
    for (slot, value) in [(Slot::Left, &mut a), (Slot::Right, &mut b)] {
        *value += match slot {
            Slot::Left => n as f32,
            Slot::Right => n as f32 * 2.0,
        };
    }
    (a + b * 10.0) as i64
}

pub fn case_call_once(n: i64) -> i64 {
    let once: Box<dyn FnOnce(i64, i64) -> i64> = Box::new(move |a, b| a * b + n);
    let single: Box<dyn FnOnce(i64) -> i64> = Box::new(move |a| a - n);
    once(2, 3) + single(4)
}

pub fn case_diverging_closure(n: i64) -> i64 {
    Some(n).unwrap_or_else(|| panic!("never")) + 1
}

pub const CASES: &[(&str, fn(i64) -> i64)] = &[
    ("case_lazy_table", case_lazy_table),
    ("case_wide_float", case_wide_float),
    ("case_default_try_fold", case_default_try_fold),
    ("case_flat_map_any", case_flat_map_any),
    ("case_dyn_method_item", case_dyn_method_item),
    ("case_array_of_refs", case_array_of_refs),
    ("case_call_once", case_call_once),
    ("case_diverging_closure", case_diverging_closure),
    ("case_collect_strings", case_collect_strings),
    ("case_sort_dedup", case_sort_dedup),
    ("case_closure_owns_guard", case_closure_owns_guard),
    ("case_btree_collect", case_btree_collect),
    ("case_zero_sized_drop", case_zero_sized_drop),
    ("case_println", case_println),
    ("case_thread_local", case_thread_local),
    ("case_files", case_files),
    ("case_sort_strings", case_sort_strings),
    ("case_thread", case_thread),
    ("case_unicode", case_unicode),
    ("case_fence", case_fence),
    ("case_inline_const", case_inline_const),
    ("case_const_block_derive", case_const_block_derive),
    ("case_const_pattern", case_const_pattern),
    ("case_tuple_param", case_tuple_param),
    ("case_self_static", case_self_static),
    ("case_const_compound_types", case_const_compound_types),
    ("case_macro_opaque_forward", case_macro_opaque_forward),
];

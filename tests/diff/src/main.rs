mod stencils;

#[derive(Clone, Copy)]
struct Pair {
    a: i64,
    b: i32,
}

#[derive(Clone, Copy)]
enum Shape {
    Empty,
    Dot(i64),
    Box { w: i64, h: i64 },
}

#[allow(dead_code)]
#[repr(u8)]
#[derive(Clone, Copy)]
enum Color {
    Red = 3,
    Green = 7,
    Blue = 11,
}

struct Wrap(i64);

fn area(shape: Shape) -> i64 {
    match shape {
        Shape::Empty => 0,
        Shape::Dot(v) => v,
        Shape::Box { w, h } => w * h,
    }
}

fn shape_for(x: i64) -> Shape {
    if x < 0 {
        Shape::Empty
    } else if x % 2 == 0 {
        Shape::Dot(x)
    } else {
        Shape::Box { w: x, h: x + 1 }
    }
}

fn apply<F: Fn(i64) -> i64>(f: F, x: i64) -> i64 {
    f(f(x))
}

fn twice_mut<F: FnMut() -> i64>(mut f: F) -> i64 {
    f();
    f()
}

fn once<F: FnOnce(i64) -> i64>(f: F, x: i64) -> i64 {
    f(x)
}

fn double(x: i64) -> i64 {
    x * 2
}

fn fib(x: i64) -> i64 {
    if x < 2 { x } else { fib(x - 1) + fib(x - 2) }
}

fn pick<T: Copy>(a: T, b: T, first: bool) -> T {
    if first { a } else { b }
}

trait Scale {
    fn scale(&self, x: i64) -> i64;
}

impl Scale for Pair {
    fn scale(&self, x: i64) -> i64 {
        self.a * x + self.b as i64
    }
}

impl Scale for Wrap {
    fn scale(&self, x: i64) -> i64 {
        self.0 - x
    }
}

fn scaled<T: Scale>(value: &T, x: i64) -> i64 {
    value.scale(x)
}

pub fn case_arith(x: i64) -> i64 {
    (x + 3) * 2 - x / 3 + x % 7
}

pub fn case_compare(x: i64) -> i64 {
    let mut n = 0;
    if x < 3 { n += 1 }
    if x <= 3 { n += 2 }
    if x > 3 { n += 4 }
    if x >= 3 { n += 8 }
    if x == 3 { n += 16 }
    if x != 3 { n += 32 }
    n
}

pub fn case_while(x: i64) -> i64 {
    let mut total = 0;
    let mut i = 0;
    while i < 5 {
        total = total + x + i;
        i = i + 1;
    }
    total
}

pub fn case_struct(x: i64) -> i64 {
    let mut pair = Pair { a: x, b: 2 };
    pair.a = pair.a + 1;
    pair.b = pair.b * 3;
    pair.a + pair.b as i64
}

pub fn case_tuple(x: i64) -> i64 {
    let t = (x, 2_i32, x * 2);
    let (a, b, c) = t;
    a + b as i64 + c
}

pub fn case_reference(x: i64) -> i64 {
    let mut v = x;
    let r = &mut v;
    *r = *r + 10;
    let s = &v;
    let t = &s;
    **t + 1
}

pub fn case_enum(x: i64) -> i64 {
    area(shape_for(x))
}

pub fn case_discriminant(x: i64) -> i64 {
    let color = if x < 0 { Color::Red } else if x == 0 { Color::Green } else { Color::Blue };
    color as i64
}

pub fn case_option(x: i64) -> i64 {
    let value = if x > 0 { Some(x) } else { None };
    match value {
        Some(v) => v * 2,
        None => -1,
    }
}

pub fn case_option_ref(x: i64) -> i64 {
    let held = x + 1;
    let value = if x > 0 { Some(&held) } else { None };
    match value {
        Some(v) => *v,
        None => -7,
    }
}

pub fn case_closure_ref(x: i64) -> i64 {
    let add = x + 1;
    apply(|v| v + add, 3)
}

pub fn case_closure_mut(x: i64) -> i64 {
    let mut seen = x;
    twice_mut(|| {
        seen = seen + 2;
        seen
    })
}

pub fn case_closure_once(x: i64) -> i64 {
    let held = Pair { a: x, b: 1 };
    once(move |v| held.a + v, 5)
}

pub fn case_fn_item(x: i64) -> i64 {
    apply(double, x)
}

pub fn case_recursion(x: i64) -> i64 {
    fib(x.rem_euclid(15))
}

pub fn case_generic(x: i64) -> i64 {
    let a = pick(x, 2, x > 0);
    let b = pick(1_i32, 2_i32, x > 0) as i64;
    let c = pick(Pair { a: x, b: 1 }, Pair { a: 9, b: 2 }, x < 0);
    a + b + c.a + c.b as i64
}

pub fn case_trait(x: i64) -> i64 {
    scaled(&Pair { a: 3, b: 4 }, x) + scaled(&Wrap(100), x)
}

pub fn case_array(x: i64) -> i64 {
    let values = [x, x + 1, x + 2, x + 3];
    let i = (x.rem_euclid(4)) as usize;
    values[i] + values[0]
}

pub fn case_range(x: i64) -> i64 {
    let mut total = 0;
    for i in 0..x.rem_euclid(10) {
        total += i;
    }
    total
}

pub fn case_cast(x: i64) -> i64 {
    let a = x as i8 as i64;
    let b = x as u8 as i64;
    let c = (x as i32 as u32) as i64;
    let d = (x as f64 * 1.5) as i64;
    let e = (x as f32 / 2.0) as i64;
    a + b + c + d + e
}

pub fn case_bits(x: i64) -> i64 {
    let u = x as u64;
    let a = (u >> 3) as i64;
    let b = x >> 2;
    let c = (u << 1) as i64;
    (a ^ b) | (c & 255)
}

pub fn case_wrapping(x: i64) -> i64 {
    x.wrapping_mul(1_000_000_007).wrapping_add(i64::MAX)
}

pub fn case_saturating(x: i64) -> i64 {
    (x as i8).saturating_add(100) as i64 + (x as u8).saturating_sub(200) as i64
}

pub fn case_rotate(x: i64) -> i64 {
    (x as u32).rotate_left(5) as i64 + (x as u64).count_ones() as i64 + (x as u32).leading_zeros() as i64
}

pub fn case_float(x: i64) -> i64 {
    let f = x as f64;
    let a = f.abs().sqrt().floor();
    let b = (f / 3.0).round();
    let c = f.max(2.0).min(50.0);
    (a + b + c) as i64
}

pub fn case_str(x: i64) -> i64 {
    let text = if x > 0 { "hello" } else { "hi" };
    text.len() as i64 + x
}

pub fn case_char(x: i64) -> i64 {
    let c = if x > 0 { 'a' } else { 'é' };
    c as i64 + c.len_utf8() as i64
}

pub fn case_i128(x: i64) -> i64 {
    let wide = (x as i128) * 1_000_000_000_000_i128;
    (wide / 1_000_000) as i64
}

fn sum_slice(items: &[i64]) -> i64 {
    let mut total = 0;
    let mut index = 0;
    while index < items.len() {
        total += items[index];
        index += 1;
    }
    total
}

fn case_slice(n: i64) -> i64 {
    let items = [n, n + 1, n + 2, n + 3];
    sum_slice(&items) + items.len() as i64
}

fn case_subslice(n: i64) -> i64 {
    let items = [n, n * 2, n * 3, n * 4];
    if let [first, middle @ .., last] = &items[..] {
        return *first + sum_slice(middle) - *last;
    }
    0
}

fn case_str_len(n: i64) -> i64 {
    let text = "hello world";
    let bytes = text.as_bytes();
    text.len() as i64 + bytes[(n.unsigned_abs() % 11) as usize] as i64
}

const WORDS: &[&str] = &["a", "bbb", "cc"];
const PAIRS: [(i64, &str); 2] = [(4, "four"), (9, "nine!")];
static TABLE: [i64; 3] = [10, 20, 30];
static COUNTER: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(5);

fn case_const_pointers(n: i64) -> i64 {
    let word = WORDS[(n.unsigned_abs() % 3) as usize];
    word.len() as i64 + PAIRS[1].0 + PAIRS[1].1.len() as i64
}

fn case_static(n: i64) -> i64 {
    TABLE[(n.unsigned_abs() % 3) as usize] + TABLE.len() as i64
}

fn case_atomic(n: i64) -> i64 {
    use std::sync::atomic::Ordering;
    let before = COUNTER.fetch_add(n, Ordering::SeqCst);
    COUNTER.store(before, Ordering::SeqCst);
    COUNTER.load(Ordering::SeqCst) + n
}

fn case_repeat(n: i64) -> i64 {
    let items = [n; 5];
    sum_slice(&items)
}

fn twice(v: i64) -> i64 {
    v * 2
}

fn apply_pointer(f: fn(i64) -> i64, v: i64) -> i64 {
    f(v)
}

fn case_fn_pointer(n: i64) -> i64 {
    let table: [fn(i64) -> i64; 2] = [twice, |v| v + 1];
    apply_pointer(table[0], n) + apply_pointer(table[1], n)
}

trait Speak {
    fn value(&self) -> i64;
    fn doubled(&self) -> i64 {
        self.value() * 2
    }
}

struct Dog(i64);
struct Cat(i64, i64);

impl Speak for Dog {
    fn value(&self) -> i64 {
        self.0 + 1
    }
}

impl Speak for Cat {
    fn value(&self) -> i64 {
        self.0 * self.1
    }
    fn doubled(&self) -> i64 {
        self.value() - 1
    }
}

fn case_dyn(n: i64) -> i64 {
    let dog = Dog(n);
    let cat = Cat(n, 3);
    let items: [&dyn Speak; 2] = [&dog, &cat];
    items[0].value() + items[1].value() + items[0].doubled() + items[1].doubled()
}

fn case_box(n: i64) -> i64 {
    let boxed = Box::new(n + 1);
    let other = Box::new((n, n + 2));
    *boxed + other.1
}

fn case_box_dyn(n: i64) -> i64 {
    let boxed: Box<dyn Speak> = if n > 0 { Box::new(Dog(n)) } else { Box::new(Cat(n, 2)) };
    boxed.value() + boxed.doubled()
}

struct Tracked<'a>(&'a std::cell::Cell<i64>, i64);

impl Drop for Tracked<'_> {
    fn drop(&mut self) {
        self.0.set(self.0.get() * 10 + self.1);
    }
}

fn consume(_value: Tracked<'_>) {}

fn case_drop_order(n: i64) -> i64 {
    let log = std::cell::Cell::new(0);
    {
        let _first = Tracked(&log, 1);
        let second = Tracked(&log, 2);
        let _third = Tracked(&log, 3);
        consume(second);
        if n > 0 {
            let _inner = Tracked(&log, 4);
        }
    }
    log.get()
}

fn case_drop_nested(n: i64) -> i64 {
    let log = std::cell::Cell::new(0);
    {
        let _pair = (Tracked(&log, 1), Tracked(&log, 2));
        let _array = [Tracked(&log, 3), Tracked(&log, 4)];
        let _some = if n > 0 { Some(Tracked(&log, 5)) } else { None };
        let _boxed = Box::new(Tracked(&log, 6));
    }
    log.get()
}

fn case_drop_match(n: i64) -> i64 {
    let log = std::cell::Cell::new(0);
    let value = if n > 0 { Some(Tracked(&log, 7)) } else { None };
    match value {
        Some(inner) => consume(inner),
        None => log.set(9),
    }
    log.get()
}

fn case_vec(n: i64) -> i64 {
    let mut items = Vec::new();
    let mut index = 0;
    while index < 40 {
        items.push(n + index);
        index += 1;
    }
    let popped = items.pop().unwrap_or(0);
    sum_slice(&items) + popped + items.len() as i64
}

fn case_vec_drop(n: i64) -> i64 {
    let log = std::cell::Cell::new(0);
    {
        let mut items = Vec::new();
        items.push(Tracked(&log, 1));
        items.push(Tracked(&log, 2));
        if n > 0 {
            items.push(Tracked(&log, 3));
        }
    }
    log.get()
}

fn case_string(n: i64) -> i64 {
    let mut text = String::from("ab");
    text.push('c');
    text.push_str("defg");
    if n > 1 {
        text.push_str(" and more than the first buffer held");
    }
    text.len() as i64 + text.as_bytes()[1] as i64
}

fn case_iterator(n: i64) -> i64 {
    let items = [n, n + 1, n + 2, n + 3, n + 4];
    items.iter().map(|v| v * 2).filter(|v| v % 3 != 0).sum::<i64>() + (0..4).map(|v| v + n).fold(0, |a, b| a + b)
}

fn case_rc(n: i64) -> i64 {
    let first = std::rc::Rc::new(n);
    let second = first.clone();
    let count = std::rc::Rc::strong_count(&second) as i64;
    drop(first);
    *second + count + std::rc::Rc::strong_count(&second) as i64
}

fn case_rc_dyn(n: i64) -> i64 {
    let shared: std::rc::Rc<dyn Speak> = std::rc::Rc::new(Cat(n, 5));
    let other = shared.clone();
    shared.value() + other.doubled()
}

fn case_box_closure(n: i64) -> i64 {
    let base = n + 1;
    let add: Box<dyn Fn(i64) -> i64> = Box::new(move |v| v + base);
    let by_ref: &dyn Fn(i64) -> i64 = &*add;
    let mut seen = 0;
    {
        let mut count: Box<dyn FnMut(i64)> = Box::new(|v| seen += v);
        count(by_ref(2));
        count(add(3));
    }
    let once: Box<dyn FnOnce() -> i64> = Box::new(move || base * 2);
    seen + once()
}

#[derive(Clone, PartialEq, Debug, Default)]
struct Record {
    id: i64,
    name: String,
    tags: Vec<i64>,
}

fn case_derive(n: i64) -> i64 {
    let first = Record { id: n, name: String::from("x"), tags: Vec::new() };
    let mut second = first.clone();
    let same = i64::from(first == second);
    second.tags.push(n);
    let other = i64::from(first == second);
    let text = format!("{second:?} {:?} {:?}", (n, "t"), Some(Marker::Pair(n, 2)));
    same * 10 + other + second.tags.len() as i64 + Record::default().id + text.len() as i64 + hash_of(&second) % 1000 + (Marker::Pair(n, 1) < Marker::Pair(n, 2)) as i64
}

#[derive(Debug, PartialEq, PartialOrd, Eq, Ord, Hash, Clone, Copy)]
enum Marker {
    Empty,
    Pair(i64, i64),
}

impl std::hash::Hash for Record {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        Marker::Pair(self.id, 1).hash(state);
        Marker::Empty.hash(state);
    }
}

fn hash_of(record: &Record) -> i64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    record.hash(&mut hasher);
    (hasher.finish() % 100_000) as i64
}

fn case_format(n: i64) -> i64 {
    let text = format!("{n}-{}", "x");
    text.len() as i64 + text.as_bytes()[0] as i64
}

fn case_hash_map(n: i64) -> i64 {
    let mut map = std::collections::HashMap::new();
    let mut index = 0;
    while index < 20 {
        map.insert(index, index * n);
        index += 1;
    }
    map.get(&7).copied().unwrap_or(0) + map.len() as i64
}






fn promoted_slice() -> &'static [i64] {
    &[3, 4, 5]
}

const BOXED_TEXT: &str = "constant text";

struct Limits { low: i64, high: i64 }
const LIMITS: Limits = Limits { low: 2, high: 9 };

fn limits() -> &'static Limits {
    &LIMITS
}

fn promoted_text() -> &'static &'static str {
    &BOXED_TEXT
}

fn case_promote(n: i64) -> i64 {
    let values = promoted_slice();
    let bound = limits();
    sum_slice(values) + values.len() as i64 + bound.low * bound.high + promoted_text().len() as i64 + n
}

#[derive(Default, Clone, Copy)]
enum Level {
    Low,
    #[default]
    High,
}

#[repr(u128)]
#[derive(Clone, Copy)]
enum Wide {
    Small = 1,
    Large = 1 << 100,
}

fn wide_of(n: i64) -> Wide {
    if n > 0 { Wide::Large } else { Wide::Small }
}

fn case_fn_ctor(n: i64) -> i64 {
    let wrap: fn(i64) -> Option<i64> = Some;
    let total: i64 = [n, 2, 3].iter().copied().map(Some).flatten().sum();
    total + wrap(n).unwrap_or(0)
}

fn case_enum_default(n: i64) -> i64 {
    let level = Level::default();
    match level {
        Level::Low => n,
        Level::High => n + 7,
    }
}

fn case_fat_compare(n: i64) -> i64 {
    let values = [n, 2, 3];
    let whole: *const [i64] = &values[..];
    let part: *const [i64] = &values[..1];
    i64::from(whole == whole) * 10 + i64::from(whole == part) * 5 + i64::from(part < whole) + i64::from(whole != part)
}

fn case_zip(n: i64) -> i64 {
    let left = [n, 2, 3, 4];
    let right = vec![5_i64, 6, 7];
    let mut total: i64 = left.iter().zip(right.iter()).fold(0, |sum, (a, b)| sum + a * b);
    for (a, b) in left.iter().zip(&right) {
        total += a - b;
    }
    let mut changed = [0_i64; 3];
    for (slot, value) in changed.iter_mut().zip(right.iter()) {
        *slot = value + 1;
    }
    total + changed[2]
}

fn case_math(n: i64) -> i64 {
    let x = n as f64 / 3.0 + 4.0;
    let y = x as f32;
    let values = [x.sin(), x.cos(), x.ln(), x.log2(), x.log10(), x.exp2() / 100.0, x.powi(3), x.round_ties_even()];
    let narrow = [y.sin(), y.cos(), y.ln(), y.powi(2)];
    let wide: f64 = values.iter().sum();
    let small: f32 = narrow.iter().sum();
    (wide * 1000.0) as i64 + (small * 100.0) as i64
}

fn case_wide_tag(n: i64) -> i64 {
    match wide_of(n) {
        Wide::Small => n + 1,
        Wide::Large => n + 2,
    }
}

fn case_nonzero(n: i64) -> i64 {
    use std::num::NonZero;
    let signed = NonZero::<i32>::new(n as i32);
    let small = NonZero::<i8>::new(n as i8);
    let unsigned = NonZero::<u32>::new(n as u32);
    let failed: Result<(), NonZero<i32>> = signed.map_or(Ok(()), Err);
    let sizes = size_of::<Option<NonZero<i32>>>() + size_of::<Option<NonZero<i128>>>() + size_of::<Result<(), NonZero<i32>>>();
    let got = signed.map_or(7, |v| v.get() as i64) + small.map_or(11, |v| v.get() as i64) + unsigned.map_or(13, |v| v.get() as i64);
    got + sizes as i64 + failed.map_or(100, |_| 200)
}

static LAZY_TABLE: std::sync::LazyLock<[i64; 4]> = std::sync::LazyLock::new(|| std::array::from_fn(|i| i as i64 * 3));

fn case_lazy_table(n: i64) -> i64 {
    LAZY_TABLE[(n.unsigned_abs() % 4) as usize] + n
}

fn case_wide_float(n: i64) -> i64 {
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

fn case_default_try_fold(n: i64) -> i64 {
    let limit = n.abs() % 9;
    let folded: Option<i64> = Plain(limit).try_fold(0_i64, |sum, item| sum.checked_add(item));
    let found = Plain(limit).any(|item| item == 2);
    let mapped = Plain(limit).map(|item| item * 2).any(|item| item == 6);
    folded.unwrap_or(-1) + found as i64 + mapped as i64 * 100
}

fn case_flat_map_any(n: i64) -> i64 {
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

fn case_dyn_method_item(n: i64) -> i64 {
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

fn case_array_of_refs(n: i64) -> i64 {
    let (mut a, mut b) = (1.5_f32, 2.5_f32);
    for (slot, value) in [(Slot::Left, &mut a), (Slot::Right, &mut b)] {
        *value += match slot {
            Slot::Left => n as f32,
            Slot::Right => n as f32 * 2.0,
        };
    }
    (a + b * 10.0) as i64
}

fn case_call_once(n: i64) -> i64 {
    let once: Box<dyn FnOnce(i64, i64) -> i64> = Box::new(move |a, b| a * b + n);
    let single: Box<dyn FnOnce(i64) -> i64> = Box::new(move |a| a - n);
    once(2, 3) + single(4)
}

fn case_diverging_closure(n: i64) -> i64 {
    Some(n).unwrap_or_else(|| panic!("never")) + 1
}

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

fn case_const_block_derive(n: i64) -> i64 {
    i64::from(Flags::new(n as u32).same(&Flags::new(n as u32))) * 2 + i64::from(Flags::new(1).same(&Flags::new(2)))
}

fn case_const_pattern(n: i64) -> i64 {
    let tag = if n > 0 { Tag::HEAD } else if n == 0 { Tag::BODY } else { Tag(*b"none") };
    let name = if n > 2 { "alpha" } else { "beta" };
    let text = match name {
        "alpha" => 10,
        "beta" => 20,
        _ => 30,
    };
    tag_value(tag) + text
}

fn case_tuple_param(n: i64) -> i64 {
    let pairs = [(n, 1_i64), (2, 3)];
    let total: i64 = pairs.iter().map(|(a, b)| a + b).sum();
    let whole = pairs.iter().copied().map(|pair: (i64, i64)| pair.0 * pair.1).fold(0, |sum, item| sum + item);
    let single: fn((i64,)) -> i64 = |wrapped| wrapped.0 + 1;
    let boxed: Box<dyn Fn((i64, i64)) -> i64> = Box::new(|pair| pair.0 - pair.1);
    total + whole + single((n,)) + boxed((n, 2))
}

fn case_self_static(n: i64) -> i64 {
    LINK.next.next.value + n
}

fn case_unicode(n: i64) -> i64 {
    let letters = ['a', '\u{e9}', '\u{4e2d}', '1', ' ', '\u{1F600}'];
    let picked = letters[(n.unsigned_abs() % 6) as usize];
    i64::from(picked.is_alphabetic()) + i64::from(picked.is_numeric()) * 2 + i64::from(picked.is_whitespace()) * 4 + i64::from(picked.is_uppercase()) * 8
}

fn case_fence(n: i64) -> i64 {
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
    n + 1
}

fn case_inline_const(n: i64) -> i64 {
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

fn case_println(n: i64) -> i64 {
    println!("hello {n} {:?}", Some("x"));
    eprintln!("note {n}");
    n + 1
}

fn case_thread_local(n: i64) -> i64 {
    LOCAL_COUNT.with(|cell| cell.set(n));
    NAMES.with(|names| names.borrow_mut().push(n.to_string()));
    LOCAL_COUNT.with(std::cell::Cell::get) * 2 + NAMES.with(|names| names.borrow().last().map_or(0, |text| text.len() as i64))
}

fn case_files(n: i64) -> i64 {
    let manifest = std::fs::read_to_string("/etc/hosts").map_or(-1, |text| i64::from(!text.is_empty()));
    let home = std::env::var("HOME").map_or(0, |value| i64::from(!value.is_empty()));
    let elapsed = std::time::Instant::now().elapsed().as_secs() as i64;
    manifest + home + elapsed + n
}

fn case_sort_strings(n: i64) -> i64 {
    let mut names: Vec<String> = (0..20).map(|i| format!("item{}", (i * 7 + n) % 11)).collect();
    names.sort();
    names.dedup();
    let map: std::collections::BTreeMap<String, usize> = names.iter().cloned().enumerate().map(|(i, name)| (name, i)).collect();
    names.join(",").len() as i64 + map.len() as i64 + map.values().sum::<usize>() as i64
}

fn case_thread(n: i64) -> i64 {
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

fn case_zero_sized_drop(n: i64) -> i64 {
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
fn case_closure_owns_guard(n: i64) -> i64 {
    let mut out = 0;
    {
        let mut guard = OnDrop { out: &mut out, local: n };
        let mut call = move |x: i64| { let held = &mut guard; held.local += x; };
        call(1);
        call(2);
    }
    out
}

fn case_sort_dedup(n: i64) -> i64 {
    let mut v: Vec<i64> = vec![n, 3, 3, 1];
    v.sort();
    let sorted = v[0] * 100 + v[3];
    v.dedup();
    sorted * 100 + v.len() as i64
}



fn case_collect_strings(n: i64) -> i64 {
    let names: Vec<String> = (0..20).map(|i| format!("item{}", (i * 7 + n) % 11)).collect();
    names.len() as i64
}
fn case_btree_collect(n: i64) -> i64 {
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

fn case_const_compound_types(n: i64) -> i64 {
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

fn case_macro_opaque_forward(n: i64) -> i64 {
    let a = mq_from_expr!(i32);
    let b = mq_from_expr!(3);
    let c = mq_from_pat!(Some(1));
    let d = mq_from_vis!(pub);
    let e = mq_compat!(x, 42, Vec<i32>);
    a + b + c + d + e + n
}

fn main() {
    let cases: &[(&str, fn(i64) -> i64)] = &[
        ("case_arith", case_arith),
        ("case_array", case_array),
        ("case_bits", case_bits),
        ("case_cast", case_cast),
        ("case_char", case_char),
        ("case_closure_mut", case_closure_mut),
        ("case_closure_once", case_closure_once),
        ("case_closure_ref", case_closure_ref),
        ("case_compare", case_compare),
        ("case_discriminant", case_discriminant),
        ("case_enum", case_enum),
        ("case_float", case_float),
        ("case_fn_item", case_fn_item),
        ("case_generic", case_generic),
        ("case_i128", case_i128),
        ("case_option", case_option),
        ("case_option_ref", case_option_ref),
        ("case_range", case_range),
        ("case_recursion", case_recursion),
        ("case_reference", case_reference),
        ("case_rotate", case_rotate),
        ("case_saturating", case_saturating),
        ("case_str", case_str),
        ("case_struct", case_struct),
        ("case_trait", case_trait),
        ("case_tuple", case_tuple),
        ("case_while", case_while),
        ("case_wrapping", case_wrapping),
        ("case_promote", case_promote),
        ("case_slice", case_slice),
        ("case_subslice", case_subslice),
        ("case_str_len", case_str_len),
        ("case_const_pointers", case_const_pointers),
        ("case_static", case_static),
        ("case_atomic", case_atomic),
        ("case_repeat", case_repeat),
        ("case_fn_pointer", case_fn_pointer),
        ("case_dyn", case_dyn),
        ("case_box", case_box),
        ("case_box_dyn", case_box_dyn),
        ("case_drop_order", case_drop_order),
        ("case_drop_nested", case_drop_nested),
        ("case_drop_match", case_drop_match),
        ("case_vec", case_vec),
        ("case_vec_drop", case_vec_drop),
        ("case_string", case_string),
        ("case_iterator", case_iterator),
        ("case_rc", case_rc),
        ("case_rc_dyn", case_rc_dyn),
        ("case_box_closure", case_box_closure),
        ("case_derive", case_derive),
        ("case_format", case_format),
        ("case_hash_map", case_hash_map),
        ("case_fn_ctor", case_fn_ctor),
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
        ("case_enum_default", case_enum_default),
        ("case_fat_compare", case_fat_compare),
        ("case_zip", case_zip),
        ("case_math", case_math),
        ("case_wide_tag", case_wide_tag),
        ("case_nonzero", case_nonzero),
        ("case_lazy_table", case_lazy_table),
        ("case_wide_float", case_wide_float),
        ("case_default_try_fold", case_default_try_fold),
        ("case_flat_map_any", case_flat_map_any),
        ("case_dyn_method_item", case_dyn_method_item),
        ("case_array_of_refs", case_array_of_refs),
        ("case_call_once", case_call_once),
        ("case_diverging_closure", case_diverging_closure),
        ("case_const_compound_types", case_const_compound_types),
        ("case_macro_opaque_forward", case_macro_opaque_forward),
    ];
    for (name, case) in cases.iter().chain(stencils::CASES) {
        for input in [-5_i64, 0, 1, 3, 20] {
            println!("{name} {input} {}", case(input));
        }
    }
}

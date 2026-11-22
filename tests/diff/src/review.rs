use std::cell::RefCell;
use std::hint::black_box;
use std::panic::Location;
use std::sync::Arc;

#[inline(never)]
fn switch_masked(x: u64, y: i64) -> i64 {
    match x & 63 {
        1 | 3 | 5 => y + 1,
        40 => y + 2,
        _ => y,
    }
}

#[inline(never)]
fn switch_one_sided(x: u64, y: i64) -> i64 {
    match x {
        56..=57 | 60 | 64..=66 => y + 1,
        61 | 69 => y + 2,
        _ => y,
    }
}

pub fn case_rev_bitmask_switch(seed: i64) -> i64 {
    let base = seed & 7;
    let mut acc = 0i64;
    for x in [0u64, 1, 3, 5, 40, 40 + 64, 55, 56, 57, 60, 61, 64, 65, 66, 69, 70, 128] {
        acc = acc.wrapping_mul(31).wrapping_add(switch_masked(black_box(x), base));
        acc = acc.wrapping_mul(31).wrapping_add(switch_one_sided(black_box(x), base));
    }
    acc
}

pub fn case_rev_u64_consts_and_ptr_cast(seed: i64) -> i64 {
    let max64 = black_box(u64::MAX);
    let r1 = max64 % 0xff00ff00ff00ff00;
    let q1 = max64 / 0xff00ff00ff00ff00;
    let r2 = (black_box(u32::MAX) % 0x8000_0000) as u64;
    let r3 = (black_box(u8::MAX) % 0x80) as u64;
    let x = black_box((seed.unsigned_abs() & 15) + 5);
    let xored = x ^ (81985529216486895 as u64);
    let ptr_addr = black_box(0x1234_5678_9abc as *const u8 as usize) as u64;
    (r1 ^ q1 ^ r2 ^ r3 ^ xored ^ ptr_addr) as i64
}

pub fn case_rev_str_cmp_and_unicode_upper(seed: i64) -> i64 {
    let a = black_box("abc");
    let b = black_box("abd");
    let lt = i64::from(a < b);
    let ord = match a.cmp(b) {
        std::cmp::Ordering::Less => 1,
        std::cmp::Ordering::Equal => 2,
        std::cmp::Ordering::Greater => 3,
    };
    let slice_lt = i64::from(black_box(&[1u8, 2, 3][..]) < black_box(&[1u8, 2, 4][..]));
    let up = black_box("привет héllo ß").to_uppercase();
    let expected = up == "ПРИВЕТ HÉLLO SS";
    let mut hash = (seed & 0xff) + lt * 10 + ord * 100 + slice_lt * 1000 + i64::from(expected) * 10_000;
    for byte in up.as_bytes() {
        hash = hash.wrapping_mul(131).wrapping_add(i64::from(*byte));
    }
    hash
}

macro_rules! spell_expr {
    ($c:expr) => {
        stringify!($c)
    };
}

#[track_caller]
#[inline(never)]
fn caller_line_here() -> (&'static str, u32) {
    let loc = Location::caller();
    (loc.file(), loc.line())
}

#[inline(never)]
fn caller_line_wrap() -> (&'static str, u32) {
    caller_line_here()
}

pub fn case_rev_stringify_and_track_caller(seed: i64) -> i64 {
    let s = spell_expr!(x > 3);
    let exact = i64::from(s == "x > 3");
    let (f1, l1) = caller_line_here();
    let (f2, l2) = caller_line_wrap();
    let direct = Location::caller();
    let ok_files = i64::from(f1.ends_with("review.rs") && f2.ends_with("review.rs") && direct.file().ends_with("review.rs"));
    let diff = i64::from(l1.saturating_sub(l2)) + i64::from(direct.line().saturating_sub(l1));
    exact * 1000 + ok_files * 100 + diff * 10 + (seed & 7)
}

fn reborrow_arc<'a>(w: Arc<&'a [u8]>) -> &'a [u8] {
    &w[..1]
}

struct ArcWrap<'a>(Arc<&'a [u8]>);

fn reborrow_arc_wrap<'a>(w: ArcWrap<'a>) -> &'a [u8] {
    &w.0[..1]
}

trait Val {
    fn get(&self) -> i64;
}

struct Num(i64);

impl Val for Num {
    fn get(&self) -> i64 {
        self.0
    }
}

pub fn case_rev_arc_reborrow_and_refcell_unsize(seed: i64) -> i64 {
    let buf = [((seed.unsigned_abs() % 50) + 1) as u8, 2, 3];
    let r1 = reborrow_arc(Arc::new(&buf[..]))[0] as i64;
    let r2 = reborrow_arc_wrap(ArcWrap(Arc::new(&buf[..])))[0] as i64;
    let arr: RefCell<Box<[i32; 3]>> = RefCell::new(Box::new([10, 20, seed as i32]));
    let slice: RefCell<Box<[i32]>> = arr;
    let slice_sum: i64 = slice.borrow().iter().map(|&v| i64::from(v)).sum();
    let slice_len = slice.borrow().len() as i64;
    let concrete: RefCell<Box<Num>> = RefCell::new(Box::new(Num(seed + 7)));
    let dyn_cell: RefCell<Box<dyn Val>> = concrete;
    let dyn_val = dyn_cell.borrow().get();
    r1 * 1000 + r2 * 100 + slice_len * 10 + slice_sum + dyn_val
}

pub const CASES: &[(&str, fn(i64) -> i64)] = &[
    ("case_rev_bitmask_switch", case_rev_bitmask_switch),
    ("case_rev_u64_consts_and_ptr_cast", case_rev_u64_consts_and_ptr_cast),
    ("case_rev_str_cmp_and_unicode_upper", case_rev_str_cmp_and_unicode_upper),
    ("case_rev_stringify_and_track_caller", case_rev_stringify_and_track_caller),
    ("case_rev_arc_reborrow_and_refcell_unsize", case_rev_arc_reborrow_and_refcell_unsize),
];

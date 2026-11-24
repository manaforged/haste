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
fn caller_line_here() -> (&'static str, u32, u32) {
    let loc = Location::caller();
    (loc.file(), loc.line(), loc.column())
}

#[inline(never)]
fn caller_line_wrap() -> (&'static str, u32, u32) {
    caller_line_here()
}

fn catch_panic_loc(f: impl FnOnce() + std::panic::UnwindSafe) -> (bool, u32, u32) {
    use std::sync::Mutex;
    static LAST: Mutex<(bool, u32, u32)> = Mutex::new((false, 0, 0));
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|info| {
        if let Some(loc) = info.location()
            && let Ok(mut slot) = LAST.lock()
        {
            *slot = (loc.file().ends_with("review.rs"), loc.line(), loc.column());
        }
    }));
    let caught = std::panic::catch_unwind(f).is_err();
    std::panic::set_hook(prev);
    let got = LAST.lock().map(|g| *g).unwrap_or((false, 0, 0));
    (caught && got.0, got.1, got.2)
}

trait TrackTr {
    fn loc(&self) -> u32;
}

struct TrackA;
struct TrackB;

impl TrackTr for TrackA {
    #[track_caller]
    #[inline(never)]
    fn loc(&self) -> u32 {
        Location::caller().line()
    }
}

impl TrackTr for TrackB {
    #[inline(never)]
    fn loc(&self) -> u32 {
        Location::caller().line()
    }
}

#[inline(never)]
fn call_track_tr<T: TrackTr>(t: &T) -> u32 {
    t.loc()
}

#[track_caller]
fn fnptr_here() -> u32 {
    Location::caller().line()
}

#[track_caller]
fn fnptr_outer() -> u32 {
    let f: fn() -> u32 = fnptr_here;
    f()
}

#[track_caller]
fn fnptr_chain() -> u32 {
    fnptr_here()
}

pub fn case_rev_stringify_and_track_caller(seed: i64) -> i64 {
    let s = spell_expr!(x > 3);
    let exact = i64::from(s == "x > 3");
    let (f1, l1, c1) = caller_line_here();
    let (f2, l2, c2) = caller_line_wrap();
    let direct = Location::caller();
    let qself = <std::panic::Location>::caller();
    let tb = call_track_tr(&TrackB);
    let ta = call_track_tr(&TrackA);
    let (u_ok, u_line, u_col) = catch_panic_loc(|| { let _ = black_box(None::<u32>).unwrap(); });
    let (e_ok, e_line, e_col) = catch_panic_loc(|| { let _ = black_box(Err::<u32, _>("e")).expect("bad"); });
    let (p_ok, p_line, p_col) = catch_panic_loc(|| { panic!("boom"); });
    let fptr: fn() -> u32 = fnptr_here;
    let gptr: fn() -> u32 = fnptr_chain;
    let fp1 = fptr();
    let fp2 = fnptr_outer();
    let fp3 = gptr();
    let ok_files = i64::from(f1.ends_with("review.rs") && f2.ends_with("review.rs") && direct.file().ends_with("review.rs") && u_ok && e_ok && p_ok);
    let mut h = exact * 1_000_000 + ok_files * 100_000 + (seed & 7);
    for v in [l1, c1, l2, c2, direct.line(), direct.column(), qself.line(), qself.column(), tb, ta, fp1, fp2, fp3, u_line, u_col, e_line, e_col, p_line, p_col] {
        h = h.wrapping_mul(131).wrapping_add(i64::from(v));
    }
    h
}

struct DerefWrap<'a>(&'a [u8]);

impl<'a> std::ops::Deref for DerefWrap<'a> {
    type Target = &'a [u8];
    fn deref(&self) -> &&'a [u8] {
        &self.0
    }
}

fn reborrow_user_deref<'a>(w: DerefWrap<'a>) -> &'a [u8] {
    &w[..1]
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
    let r3 = reborrow_user_deref(DerefWrap(&buf[..]))[0] as i64;
    let arr: RefCell<Box<[i32; 3]>> = RefCell::new(Box::new([10, 20, seed as i32]));
    let slice: RefCell<Box<[i32]>> = arr;
    let slice_sum: i64 = slice.borrow().iter().map(|&v| i64::from(v)).sum();
    let slice_len = slice.borrow().len() as i64;
    let concrete: RefCell<Box<Num>> = RefCell::new(Box::new(Num(seed + 7)));
    let dyn_cell: RefCell<Box<dyn Val>> = concrete;
    let dyn_val = dyn_cell.borrow().get();
    r1 * 1000 + r2 * 100 + r3 * 10000 + slice_len * 10 + slice_sum + dyn_val
}

struct TlsDropGuard<'a>(&'a mut i64);

impl Drop for TlsDropGuard<'_> {
    fn drop(&mut self) {
        *self.0 += 1;
    }
}

#[inline(never)]
fn tls_init_unwind_body(trigger: bool, drops: &mut i64) -> i64 {
    thread_local! {
        static PANIC_TLS: std::cell::Cell<i64> = {
            panic!("tls init boom");
        };
    }
    let guard = TlsDropGuard(drops);
    let mut out = 0;
    if trigger {
        out = PANIC_TLS.with(|c| c.get());
    }
    std::mem::forget(guard);
    out
}

pub fn case_rev_thread_local_init_panic_drop(seed: i64) -> i64 {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let (drops, caught) = std::thread::spawn(move || {
        let mut drops = 0_i64;
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            black_box(tls_init_unwind_body(black_box(true), &mut drops));
        }))
        .is_err();
        (drops, i64::from(caught))
    })
    .join()
    .unwrap_or((-1, -1));
    std::panic::set_hook(prev);
    let no_trigger_drops = {
        let mut d = 0_i64;
        black_box(tls_init_unwind_body(black_box(false), &mut d));
        d
    };
    drops * 100 + caught * 10 + no_trigger_drops * 1000 + (seed & 7)
}

#[repr(C)]
struct LatchState {
    flag: i64,
    a: i64,
    b: i64,
}

#[inline(never)]
fn cold_latch_step(a: &mut i64, b: &mut i64) -> i64 {
    *a = a.wrapping_add(3);
    *b = b.wrapping_add(5);
    *a ^ *b
}

#[inline(never)]
fn licm_cold_latch_inner(st: &mut LatchState, limit: i64) -> i64 {
    let mut acc = 0_i64;
    loop {
        if st.flag >= limit {
            return acc.wrapping_add(st.a).wrapping_add(st.b);
        }
        acc = acc.wrapping_add(cold_latch_step(&mut st.a, &mut st.b));
        st.flag += 1;
    }
}

pub fn case_rev_licm_cold_latch(seed: i64) -> i64 {
    let mut st = LatchState { flag: 0, a: seed, b: seed.wrapping_add(11) };
    let limit = (seed.unsigned_abs() % 4) as i64;
    licm_cold_latch_inner(black_box(&mut st), black_box(limit))
}

pub const CASES: &[(&str, fn(i64) -> i64)] = &[
    ("case_rev_bitmask_switch", case_rev_bitmask_switch),
    ("case_rev_u64_consts_and_ptr_cast", case_rev_u64_consts_and_ptr_cast),
    ("case_rev_str_cmp_and_unicode_upper", case_rev_str_cmp_and_unicode_upper),
    ("case_rev_stringify_and_track_caller", case_rev_stringify_and_track_caller),
    ("case_rev_arc_reborrow_and_refcell_unsize", case_rev_arc_reborrow_and_refcell_unsize),
    ("case_rev_thread_local_init_panic_drop", case_rev_thread_local_init_panic_drop),
    ("case_rev_licm_cold_latch", case_rev_licm_cold_latch),
];

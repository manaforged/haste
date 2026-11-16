use std::hint::black_box;

#[derive(Clone, Copy)]
struct BigQuad {
    a: i64,
    b: i64,
    c: i64,
    d: i64,
}

#[inline(never)]
fn sum_quad(q: BigQuad, extra: i64) -> i64 {
    q.a + q.b + q.c + q.d + extra
}

pub fn case_reg_byval_pair(n: i64) -> i64 {
    let mut q = BigQuad { a: n + 1, b: 2, c: 3, d: 4 };
    let first = sum_quad(q, {
        q.a = 100;
        q.d = 200;
        5
    });
    let arr = [
        q,
        {
            q.b = 77;
            q
        },
    ];
    first + arr[0].b * 10 + arr[1].b + q.a
}

#[inline(never)]
fn take_string(s: String, extra: i64) -> i64 {
    s.len() as i64 * 100 + s.as_bytes().first().copied().unwrap_or(0) as i64 + extra
}

pub fn case_reg_byval_string(n: i64) -> i64 {
    let mut s = format!("old_{n}");
    let got = take_string(s, {
        s = String::from("reassigned_longer_text");
        7
    });
    got + s.len() as i64
}

struct WipeGuard<'a>(&'a mut [i64; 4]);

impl Drop for WipeGuard<'_> {
    fn drop(&mut self) {
        *self.0 = [0, 0, 0, 0];
    }
}

pub fn case_reg_scope_drop_tail(n: i64) -> i64 {
    let mut backing = [n + 10, n + 20, n + 30, n + 40];
    let from_block = {
        let g = WipeGuard(&mut backing);
        *g.0
    };
    let mut backing2 = [n + 1, n + 2, n + 3, n + 4];
    let from_match = match WipeGuard(&mut backing2) {
        g if n >= -100 => *g.0,
        _ => [9, 9, 9, 9],
    };
    from_block[0] + from_block[3] + from_match[1] + backing[0] + backing2[1]
}

pub fn case_reg_u128_float(n: i64) -> i64 {
    let shift = 70 + (n.unsigned_abs() % 20);
    let big_u: u128 = (1_u128 << shift) | 13;
    let big_i: i128 = -((1_i128 << shift) + 9);
    let tie_even: u128 = (0x20_0000_0000_0000_u128 << 20) | (1_u128 << 19);
    let tie_up: u128 = tie_even | 1;
    let a = (big_u as f64 / (1_u128 << (shift - 10)) as f64) as i64;
    let b = (big_i as f64 / (1_u128 << (shift - 8)) as f64) as i64;
    let c = ((big_u as f32) > 1.0e20_f32) as i64;
    let d = ((u128::MAX as f32).is_infinite()) as i64;
    let e = ((tie_up as f64).to_bits() - (tie_even as f64).to_bits()) as i64;
    a + b + c * 10 + d * 100 + e * 1000
}

pub fn case_reg_f32_rounding(n: i64) -> i64 {
    let lit = 9007199531679745.0_f32;
    (lit.to_bits() as i64).wrapping_add(n)
}

pub fn case_reg_char_surrogate(n: i64) -> i64 {
    let mut count = 0_i64;
    let mut sum = 0_i64;
    for ch in '\u{D7FD}'..='\u{E002}' {
        count += 1;
        sum += ch as u32 as i64;
    }
    count * 1000 + (sum % 997) + n
}

#[repr(C)]
struct Cdiv {
    quot: i32,
    rem: i32,
}

unsafe extern "C" {
    fn div(numer: i32, denom: i32) -> Cdiv;
}

pub fn case_reg_c_abi_div(n: i64) -> i64 {
    let num = (n as i32) * 7 + 23;
    // SAFETY: `libc::div` is pure for non-zero denominator `5`.
    let out = unsafe { div(num, 5) };
    (out.quot as i64) * 100 + (out.rem as i64)
}

#[inline(never)]
fn narrow_thread_u8(flag: bool, k: u8) -> i64 {
    let v: u8 = if flag { 128 } else { k };
    let wrapped = v.wrapping_add(128);
    if wrapped == 0 { 222 } else { 111 + wrapped as i64 }
}

pub fn case_reg_narrow_thread(n: i64) -> i64 {
    let a = narrow_thread_u8(black_box(true), black_box(n as u8));
    let b = narrow_thread_u8(black_box(false), black_box((n.unsigned_abs() % 5) as u8));
    a + b
}

#[inline(never)]
fn switch_shared_const(tag: i64) -> i64 {
    let out = match tag {
        0 => 0,
        1 => 1,
        2 => 2,
        3 => 3,
        4 => 4,
        5 => 5,
        6 => 6,
        7 => 7,
        _ => 99,
    };
    let sparse = match tag {
        -5 => 10,
        1 => 1,
        3 => 3,
        7 => 7,
        20 => 20,
        _ => 50,
    };
    out * 100 + sparse
}

pub fn case_reg_switch_const(n: i64) -> i64 {
    switch_shared_const(black_box(n)) + switch_shared_const(black_box(5)) + switch_shared_const(black_box(7))
}

mod block_a {
    pub struct T(pub i64);
}

mod block_b {
    pub struct T(pub i64, pub i64);
}

pub fn case_reg_block_use(n: i64) -> i64 {
    let left = {
        use block_a::T;
        T(n + 10).0
    };
    let right = {
        use block_b::T;
        let v = T(n, 30);
        v.0 + v.1
    };
    left * 10 + right
}

pub fn case_reg_inner_generic_method(n: i64) -> i64 {
    struct Local(i64);
    impl Local {
        #[allow(dead_code, reason = "shadow inherent method must not resolve for generic &T")]
        fn tag(&self) -> i64 {
            self.0 + 100
        }
    }
    trait Fallback {
        fn tag(&self) -> i64;
    }
    impl<T> Fallback for &T {
        fn tag(&self) -> i64 {
            777
        }
    }
    fn probe<T>(r: &T) -> i64 {
        r.tag()
    }
    let local = Local(n);
    probe(&local) + n
}

struct ZstRhs;

impl std::ops::Mul<ZstRhs> for i32 {
    type Output = u64;
    fn mul(self, _: ZstRhs) -> u64 {
        (self as i64 + 30) as u64
    }
}

pub fn case_reg_overloaded_lhs_lit(n: i64) -> i64 {
    let prod: u64 = 2 * ZstRhs;
    prod as i64 + n
}

pub fn case_helper_not_runner(bytes: &[u8]) -> usize {
    bytes.len()
}

pub fn case_reg_wide_rot_bitrev(n: i64) -> i64 {
    let base = black_box(((n as u64 as u128) << 73) | 0x0123_4567_89ab_cdef_u128);
    let amt = black_box((n.unsigned_abs() as u32 % 40) + 13);
    let rl = base.rotate_left(amt);
    let rr = rl.rotate_right(amt);
    let rev = rr.reverse_bits().reverse_bits();
    let helper = case_helper_not_runner(b"abcd") as i64;
    ((rev ^ base) as i64) + ((rl >> 64) as i64 & 0xffff) + helper
}

macro_rules! branch_step_64 {
    ($acc:ident, $x:ident) => {
        branch_step_16!($acc, $x);
        branch_step_16!($acc, $x);
        branch_step_16!($acc, $x);
        branch_step_16!($acc, $x);
    };
}

macro_rules! branch_step_16 {
    ($acc:ident, $x:ident) => {
        $acc = $acc.wrapping_mul(131).wrapping_add($x ^ 0x55);
        $acc = $acc.rotate_left(7) ^ ($acc >> 3);
        $acc = $acc.wrapping_mul(17).wrapping_sub($x);
        $acc ^= $acc.wrapping_add(0x1234_5678);
    };
}

macro_rules! branch_step_1024 {
    ($acc:ident, $x:ident) => {
        branch_step_64!($acc, $x);
        branch_step_64!($acc, $x);
        branch_step_64!($acc, $x);
        branch_step_64!($acc, $x);
        branch_step_64!($acc, $x);
        branch_step_64!($acc, $x);
        branch_step_64!($acc, $x);
        branch_step_64!($acc, $x);
        branch_step_64!($acc, $x);
        branch_step_64!($acc, $x);
        branch_step_64!($acc, $x);
        branch_step_64!($acc, $x);
        branch_step_64!($acc, $x);
        branch_step_64!($acc, $x);
        branch_step_64!($acc, $x);
        branch_step_64!($acc, $x);
    };
}

#[inline(never)]
pub fn case_reg_large_branch_relax(n: i64) -> i64 {
    let x = black_box(n);
    let mut acc = x.wrapping_add(19);
    if (x & 1) == 0 {
        branch_step_1024!(acc, x);
        branch_step_1024!(acc, x);
    }
    if x < -1000 {
        return acc;
    }
    acc.wrapping_add(7)
}

#[cfg(target_arch = "x86_64")]
#[inline(never)]
fn simd_self_f32_lanes(base: f32) -> [f32; 4] {
    use core::arch::x86_64::{_mm_add_ps, _mm_div_ps, _mm_mul_ps, _mm_setr_ps, _mm_storeu_ps, _mm_sub_ps};
    let mut out_add = [0.0_f32; 4];
    let mut out_sub = [0.0_f32; 4];
    let mut out_mul = [0.0_f32; 4];
    let mut out_div = [0.0_f32; 4];
    // SAFETY: SSE baseline is always available on x86_64 and stack arrays are valid for 4 f32s.
    unsafe {
        let v = _mm_setr_ps(base, base + 1.0, base + 2.0, base + 3.0);
        _mm_storeu_ps(out_add.as_mut_ptr(), _mm_add_ps(v, v));
        _mm_storeu_ps(out_sub.as_mut_ptr(), _mm_sub_ps(v, v));
        _mm_storeu_ps(out_mul.as_mut_ptr(), _mm_mul_ps(v, v));
        _mm_storeu_ps(out_div.as_mut_ptr(), _mm_div_ps(v, v));
    }
    [
        out_add[0] + out_add[3],
        out_sub[1] + out_sub[2],
        out_mul[0] + out_mul[2],
        out_div[0] + out_div[3],
    ]
}

#[cfg(target_arch = "aarch64")]
#[inline(never)]
fn simd_self_f32_lanes(base: f32) -> [f32; 4] {
    use core::arch::aarch64::{vaddq_f32, vdivq_f32, vld1q_f32, vmulq_f32, vst1q_f32, vsubq_f32};
    let src = [base, base + 1.0, base + 2.0, base + 3.0];
    let mut out_add = [0.0_f32; 4];
    let mut out_sub = [0.0_f32; 4];
    let mut out_mul = [0.0_f32; 4];
    let mut out_div = [0.0_f32; 4];
    // SAFETY: NEON is baseline on aarch64 and stack arrays hold 4 contiguous f32 values.
    unsafe {
        let v = vld1q_f32(src.as_ptr());
        vst1q_f32(out_add.as_mut_ptr(), vaddq_f32(v, v));
        vst1q_f32(out_sub.as_mut_ptr(), vsubq_f32(v, v));
        vst1q_f32(out_mul.as_mut_ptr(), vmulq_f32(v, v));
        vst1q_f32(out_div.as_mut_ptr(), vdivq_f32(v, v));
    }
    [
        out_add[0] + out_add[3],
        out_sub[1] + out_sub[2],
        out_mul[0] + out_mul[2],
        out_div[0] + out_div[3],
    ]
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn simd_self_f32_lanes(base: f32) -> [f32; 4] {
    [
        (base * 2.0) + ((base + 3.0) * 2.0),
        0.0,
        (base * base) + ((base + 2.0) * (base + 2.0)),
        2.0,
    ]
}

pub fn case_reg_vec_self_ops(n: i64) -> i64 {
    let base = black_box(((n.unsigned_abs() % 7) + 2) as f32);
    let lanes = simd_self_f32_lanes(base);
    (lanes[0] as i64) * 1000 + (lanes[1] as i64) * 100 + (lanes[2] as i64) * 10 + (lanes[3] as i64)
}

#[inline(never)]
fn peeled_sextend_zero(flag: bool, y: i8) -> i64 {
    let t = ((y | 1) as i64).wrapping_add(1);
    let sel = if flag { t } else { 0 };
    let is_nz = (sel != 0) as i64;
    let is_eq = (sel == 0) as i64;
    (is_nz * 100 + is_eq * 10).wrapping_add(sel.wrapping_mul(3))
}

#[inline(never)]
fn peeled_uextend_zero(flag: bool, y: u8) -> i64 {
    let t = ((y | 1) as u64).wrapping_sub(1);
    let sel = if flag { t } else { 0 };
    let is_nz = (sel != 0) as i64;
    let is_eq = (sel == 0) as i64;
    (is_nz * 100 + is_eq * 10).wrapping_add(sel.wrapping_mul(3) as i64)
}

#[inline(never)]
fn peeled_select_zero(flag: bool, inner: bool, a: i64, b: i64) -> i64 {
    let nz = if inner { a | 1 } else { b | 2 };
    let shifted = nz.wrapping_sub(1);
    let sel = if flag { shifted } else { 0 };
    let is_nz = (sel != 0) as i64;
    let is_eq = (sel == 0) as i64;
    (is_nz * 100 + is_eq * 10).wrapping_add(sel.wrapping_mul(3))
}

#[inline(never)]
fn peeled_sextend_select_zero(flag: bool, inner: bool, a: i8, b: i8) -> i64 {
    let picked = if inner { a | 1 } else { b | 2 };
    let t = (picked as i64).wrapping_add(1);
    let sel = if flag { t } else { 0 };
    let is_nz = (sel != 0) as i64;
    let is_eq = (sel == 0) as i64;
    (is_nz * 100 + is_eq * 10).wrapping_add(sel.wrapping_mul(3))
}

pub fn case_reg_peeled_nonzero_offset(n: i64) -> i64 {
    let a = peeled_sextend_zero(black_box(true), black_box(-1_i8));
    let b = peeled_sextend_zero(black_box(true), black_box(n as i8));
    let c = peeled_uextend_zero(black_box(true), black_box(0_u8));
    let d = peeled_select_zero(black_box(true), black_box(true), black_box(0), black_box(n));
    let e = peeled_sextend_select_zero(black_box(true), black_box(true), black_box(-1_i8), black_box(n as i8));
    let f = peeled_sextend_select_zero(black_box(false), black_box(true), black_box(-1_i8), black_box(n as i8));
    a.wrapping_add(b).wrapping_add(c).wrapping_add(d).wrapping_add(e).wrapping_add(f)
}

struct CountDrop<'a>(&'a mut i64, i64);

impl Drop for CountDrop<'_> {
    fn drop(&mut self) {
        *self.0 += self.1;
    }
}

#[inline(never)]
fn pin_branch_probe(cond: bool, drops: &mut i64, n: i64) -> i64 {
    let text_len = if cond {
        core::pin::pin!(format!("pin_{n}")).len() as i64
    } else {
        17
    };
    let guard_val = if cond {
        let p = core::pin::pin!(CountDrop(drops, 100));
        p.1 + 3
    } else {
        29
    };
    text_len + guard_val
}

pub fn case_reg_pin_cond_drop(n: i64) -> i64 {
    let mut drops = 0_i64;
    let t = pin_branch_probe(black_box(true), &mut drops, n);
    let f = pin_branch_probe(black_box(false), &mut drops, n);
    t * 100 + f * 10 + drops
}

pub fn case_reg_closure_scalar_env(n: i64) -> i64 {
    let mut acc = black_box(n + 5);
    let zst = core::marker::PhantomData::<u64>;
    let step = {
        let mut cur = acc;
        let mut f = move |delta: i64| {
            let core::marker::PhantomData = zst;
            cur += delta;
            cur
        };
        f(3) + f(7)
    };
    let mut bump = |k: i64| {
        acc += k;
        acc
    };
    let r1 = bump(4);
    let r2 = bump(6);
    let pair_fn = |(x, y): (i64, i64)| x * 10 + y + step;
    r1 + r2 + pair_fn((acc, 2))
}

#[inline(never)]
fn invoke_once_pair<G: FnOnce(String, CountDrop<'_>) -> i64>(g: G, s: String, d: CountDrop<'_>) -> i64 {
    g(s, d)
}

pub fn case_reg_boxed_fnonce_tuple(n: i64) -> i64 {
    let mut drops = 0_i64;
    let out = {
        let g = Box::new(move |s: String, guard: CountDrop<'_>| (s.len() as i64) + guard.1 + n);
        invoke_once_pair(g, format!("val_{n}"), CountDrop(&mut drops, 100))
    };
    out * 1000 + drops
}

static TAILREC_PTRS: [core::sync::atomic::AtomicUsize; 3] = [
    core::sync::atomic::AtomicUsize::new(0),
    core::sync::atomic::AtomicUsize::new(0),
    core::sync::atomic::AtomicUsize::new(0),
];

#[inline(never)]
fn stash_frame_slot(depth: usize, slot: &i64) {
    if depth < 3 {
        TAILREC_PTRS[depth].store(core::ptr::from_ref(slot) as usize, core::sync::atomic::Ordering::SeqCst);
    }
}

#[inline(never)]
fn tailrec_escape_walk(depth: usize, seed: i64) -> i64 {
    let local = match depth {
        0 => seed + 3,
        1 => seed + 7,
        _ => seed + 10,
    };
    stash_frame_slot(depth, &local);
    if depth == 2 {
        let p0 = TAILREC_PTRS[0].load(core::sync::atomic::Ordering::SeqCst) as *const i64;
        let p1 = TAILREC_PTRS[1].load(core::sync::atomic::Ordering::SeqCst) as *const i64;
        let p2 = TAILREC_PTRS[2].load(core::sync::atomic::Ordering::SeqCst) as *const i64;
        // SAFETY: `p0`, `p1`, and `p2` point to live `local` stack slots in frames 0, 1, and 2 unless tail-call loopification illegally reused a single slot.
        unsafe { (*p0) * 100 + (*p1) * 10 + (*p2) }
    } else {
        tailrec_escape_walk(depth + 1, seed)
    }
}

pub fn case_reg_tailrec_escape_helper(n: i64) -> i64 {
    tailrec_escape_walk(0, black_box(n % 5))
}

#[inline(never)]
fn se_i8_ptr(p: *const i8) -> i64 {
    // SAFETY: caller passes a pointer at least 3 bytes into a >= 20-byte i8 array.
    unsafe { (*p as i64) * 3 + (*p.add(16) as i64) * 5 + (*p.offset(-3) as i64) * 7 }
}

#[inline(never)]
fn se_i16_ptr(p: *const i16) -> i64 {
    // SAFETY: caller passes a pointer at least 3 elements into a >= 12-element i16 array.
    unsafe { (*p as i64) * 3 + (*p.add(8) as i64) * 5 + (*p.offset(-3) as i64) * 7 }
}

#[inline(never)]
fn se_i32_ptr(p: *const i32) -> i64 {
    // SAFETY: caller passes a pointer at least 3 elements into a >= 8-element i32 array.
    unsafe { (*p as i64) * 3 + (*p.add(4) as i64) * 5 + (*p.offset(-3) as i64) * 7 }
}

#[inline(never)]
fn se_stack_large(seed: i64) -> i64 {
    let mut a8 = [0i8; 640];
    let mut a16 = [0i16; 320];
    let mut a32 = [0i32; 160];
    a8[0] = -128;
    a8[300] = -1;
    a8[600] = if (seed & 1) == 0 { -5 } else { 4 };
    a16[0] = -32768;
    a16[150] = -1;
    a16[300] = if (seed & 2) == 0 { -5 } else { 7 };
    a32[0] = i32::MIN + 7;
    a32[75] = -1;
    a32[150] = -128;
    black_box(&mut a8);
    black_box(&mut a16);
    black_box(&mut a32);
    (a8[0] as i64)
        + (a8[300] as i64) * 2
        + (a8[600] as i64) * 3
        + (a16[0] as i64) * 5
        + (a16[150] as i64) * 7
        + (a16[300] as i64) * 11
        + (a32[0] as i64)
        + (a32[75] as i64) * 13
        + (a32[150] as i64) * 17
}

#[inline(never)]
fn cmp_sload_i8(a: &[i8; 24], idx: usize, x: i64, y: i64) -> i64 {
    let b1 = if a[idx] < 0 { x + 11 } else { y + 23 };
    let b2 = if a[idx + 1] >= -5 { x + 31 } else { y + 43 };
    let mut acc = b1 + b2;
    if a[idx + 2] == -1i8 {
        acc += black_box(53);
    } else {
        acc += black_box(67);
    }
    if a[idx + 3] < 0 {
        acc += black_box(79);
    } else {
        acc += black_box(89);
    }
    acc
}

#[inline(never)]
fn cmp_sload_i16(a: &[i16; 16], idx: usize, x: i64, y: i64) -> i64 {
    let b1 = if a[idx] < 0 { x + 13 } else { y + 29 };
    let b2 = if a[idx + 1] >= -5 { x + 37 } else { y + 47 };
    let mut acc = b1 + b2;
    if a[idx + 2] == -1i16 {
        acc += black_box(59);
    } else {
        acc += black_box(71);
    }
    if a[idx + 3] >= -5 {
        acc += black_box(83);
    } else {
        acc += black_box(97);
    }
    acc
}

/// Exercises ARM64 signed narrow loads and signed icmp fusion.
pub fn case_reg_arm64_signed_load_icmp(seed: i64) -> i64 {
    let mut a8 = [
        -128i8, -5, -1, 12, -1, -128, -6, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, -1, -128, -5, 4,
        127,
    ];
    let mut a16 = [
        -32768i16, -6, -1, 17, -1, -32768, -5, 0, 1, 2, 3, -1, -32768, -5, -6, 32767,
    ];
    let mut a32 = [
        i32::MIN + 7,
        -1,
        -128,
        -32768,
        -1,
        0,
        1,
        i32::MIN + 7,
    ];
    if (seed & 1) != 0 {
        a8[3] = -1;
        a8[6] = -5;
        a16[3] = -5;
        a16[6] = -6;
    }
    let a8 = black_box(a8);
    let a16 = black_box(a16);
    let a32 = black_box(a32);
    // SAFETY: indices 3, 3, and 3 are within bounds with +16 bytes and -3 elements headroom.
    let se_sum = unsafe {
        se_i8_ptr(a8.as_ptr().add(3))
            + se_i16_ptr(a16.as_ptr().add(3))
            + se_i32_ptr(a32.as_ptr().add(3))
    };
    let idx8 = ((seed.unsigned_abs() as usize) % 2) * 3;
    let idx16 = ((seed.unsigned_abs() as usize) % 2) * 3;
    let cmp_sum = cmp_sload_i8(&a8, idx8, seed, seed + 10)
        + cmp_sload_i16(&a16, idx16, seed + 20, seed + 30);
    se_sum + se_stack_large(seed) + cmp_sum
}

pub fn case_reg_autoderef_borrow_var(seed: i64) -> i64 {
    use std::collections::HashSet;
    let base = (seed.unsigned_abs() as usize) % 7;
    let items = black_box([base, base + 1, base + 2, base + 3, base + 4]);
    let mut seen = HashSet::new();
    seen.insert(base + 1);
    seen.insert(base + 3);
    let kept: usize = items.iter().filter(|dep| !seen.contains(dep)).copied().sum();
    let hit: usize = items.iter().filter(|dep| seen.contains(dep)).copied().sum();
    (kept as i64) * 100 + (hit as i64)
}

struct DivErr(i64);

#[inline(never)]
fn run_diverging_try(x: i64) -> Result<i64, DivErr> {
    let v = (if x > 0 { Ok(x * 3) } else { return Err(DivErr(-x - 1)); })?;
    Ok(v + 5)
}

pub fn case_reg_try_diverging_err(seed: i64) -> i64 {
    let mut acc = 0_i64;
    for k in -4..4 {
        match run_diverging_try(black_box(seed + k)) {
            Ok(v) => acc += v,
            Err(DivErr(e)) => acc -= e,
        }
    }
    acc
}

pub const CASES: &[(&str, fn(i64) -> i64)] = &[
    ("case_reg_byval_pair", case_reg_byval_pair),
    ("case_reg_byval_string", case_reg_byval_string),
    ("case_reg_scope_drop_tail", case_reg_scope_drop_tail),
    ("case_reg_u128_float", case_reg_u128_float),
    ("case_reg_f32_rounding", case_reg_f32_rounding),
    ("case_reg_char_surrogate", case_reg_char_surrogate),
    ("case_reg_c_abi_div", case_reg_c_abi_div),
    ("case_reg_narrow_thread", case_reg_narrow_thread),
    ("case_reg_switch_const", case_reg_switch_const),
    ("case_reg_block_use", case_reg_block_use),
    ("case_reg_inner_generic_method", case_reg_inner_generic_method),
    ("case_reg_overloaded_lhs_lit", case_reg_overloaded_lhs_lit),
    ("case_reg_wide_rot_bitrev", case_reg_wide_rot_bitrev),
    ("case_reg_large_branch_relax", case_reg_large_branch_relax),
    ("case_reg_vec_self_ops", case_reg_vec_self_ops),
    ("case_reg_peeled_nonzero_offset", case_reg_peeled_nonzero_offset),
    ("case_reg_pin_cond_drop", case_reg_pin_cond_drop),
    ("case_reg_closure_scalar_env", case_reg_closure_scalar_env),
    ("case_reg_boxed_fnonce_tuple", case_reg_boxed_fnonce_tuple),
    ("case_reg_tailrec_escape_helper", case_reg_tailrec_escape_helper),
    ("case_reg_arm64_signed_load_icmp", case_reg_arm64_signed_load_icmp),
    ("case_reg_autoderef_borrow_var", case_reg_autoderef_borrow_var),
    ("case_reg_try_diverging_err", case_reg_try_diverging_err),
];

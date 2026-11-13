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
];

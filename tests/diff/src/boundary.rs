use std::hint::black_box;

const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

pub fn case_boundary_int_extremes(seed: i64) -> i64 {
    let vals: [i64; 10] = [
        i64::MIN,
        i64::MIN + 1,
        -(1 << 31),
        -1,
        0,
        1,
        1 << 31,
        1 << 62,
        i64::MAX - 1,
        i64::MAX,
    ];
    let mut acc = seed.wrapping_mul(0x517c_c1b7_2722_0a95u64 as i64);
    for &x in &vals {
        let v = black_box(x);
        let (a1, o1) = v.overflowing_add(seed);
        let (a2, o2) = v.overflowing_sub(seed);
        let (a3, o3) = v.overflowing_mul(3);
        let sat = v.saturating_add(seed) ^ v.saturating_sub(seed);
        let (q_neg, q_ov) = v.overflowing_div(-1);
        let chk_div = v.checked_div(-1).unwrap_or(0);
        let wrap_div = v.wrapping_div(-1);
        let rem = v.wrapping_rem(if seed == 0 { 7 } else { seed });
        let bits = (v.leading_zeros() as i64)
            | ((v.trailing_zeros() as i64) << 8)
            | ((v.count_ones() as i64) << 16);
        let rot = v.rotate_left((seed.unsigned_abs() & 63) as u32) ^ v.swap_bytes();
        let flags = i64::from(o1)
            | (i64::from(o2) << 1)
            | (i64::from(o3) << 2)
            | (i64::from(q_ov) << 3);
        acc = acc.wrapping_mul(31).wrapping_add(
            a1 ^ a2 ^ a3 ^ sat ^ q_neg ^ chk_div ^ wrap_div ^ rem ^ bits ^ rot ^ flags,
        );
    }
    let wide = [i128::MIN, -1, 0, 1, i128::MAX];
    for &w in &wide {
        let bw = black_box(w);
        let folded = bw.wrapping_add(i128::from(seed)).wrapping_mul(5);
        acc ^= (folded as i64) ^ ((folded >> 64) as i64);
    }
    let div_mask = (seed.unsigned_abs() | 1) & 0xFFFF;
    let uvals = [0u64, 1, 1 << 31, 1 << 62, 1 << 63, u64::MAX];
    for &u in &uvals {
        let bu = black_box(u);
        let d = bu / div_mask;
        let r = bu % div_mask;
        acc = acc.wrapping_add((d ^ r ^ bu.reverse_bits()) as i64);
    }
    acc
}

pub fn case_boundary_shifts_and_casts(seed: i64) -> i64 {
    let base = black_box((seed as u64).wrapping_mul(GOLDEN) ^ 0x8000_0001_0000_8001);
    let mut acc = 0u64;
    for sh in [0u32, 1, 7, 8, 15, 16, 31, 32, 63, 64, 65, 127] {
        let s = black_box(sh);
        let l = base.wrapping_shl(s);
        let r = base.wrapping_shr(s);
        let a = (base as i64).wrapping_shr(s) as u64;
        let i8_ext = (l as i8) as i64 as u64;
        let u8_ext = (r as u8) as u64;
        let i16_ext = (a as i16) as i64 as u64;
        let i32_ext = ((l ^ r) as i32) as i64 as u64;
        acc = acc
            .rotate_left(5)
            .wrapping_add(l ^ r ^ a ^ i8_ext ^ u8_ext ^ i16_ext ^ i32_ext);
    }
    acc as i64
}

pub fn case_boundary_float_edges(seed: i64) -> i64 {
    let bias = (seed as f64) * 0.25;
    let vals: [f64; 10] = [
        -0.0,
        0.0,
        1.0 + bias,
        -1.0 + bias,
        f64::from_bits(1),
        1e-308,
        f64::MIN,
        f64::MAX,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ];
    let mut acc = 0u64;
    for &x in &vals {
        let v = black_box(x);
        let abs = v.abs().to_bits();
        let neg = (-v).to_bits();
        let copy = v.copysign(seed as f64).to_bits();
        let fl = v.floor().to_bits();
        let cl = v.ceil().to_bits();
        let tr = v.trunc().to_bits();
        let sat_i = (v as i64) as u64;
        let sat_u = v as u64;
        let cmp = u64::from(v == 0.0)
            | (u64::from(v < 0.0) << 1)
            | (u64::from(v.is_infinite()) << 2)
            | (u64::from(v.is_sign_negative()) << 3);
        acc = acc
            .wrapping_mul(131)
            .wrapping_add(abs ^ neg ^ copy ^ fl ^ cl ^ tr ^ sat_i ^ sat_u ^ cmp);
    }
    let nan = black_box(f64::NAN);
    let nan2 = black_box(f64::NAN);
    let nan_flags = u64::from(nan.is_nan())
        | (u64::from(nan == nan2) << 1)
        | (u64::from(nan != nan2) << 2)
        | (u64::from(nan < bias) << 3)
        | (u64::from(nan >= bias) << 4);
    (acc ^ nan_flags) as i64
}

pub fn case_boundary_seeded_prng(seed: i64) -> i64 {
    let extra: u64 = std::env::var("HASTE_DIFF_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let mut state = (seed as u64).wrapping_add(GOLDEN).wrapping_add(extra);
    let mut buf = [0i64; 32];
    for slot in &mut buf {
        state = state.wrapping_add(GOLDEN);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        *slot = (z ^ (z >> 31)) as i64;
    }
    buf.sort_unstable();
    let mut acc = 0i64;
    for (idx, pair) in buf.windows(2).enumerate() {
        let ordered = i64::from(pair[0] <= pair[1]);
        let diff = pair[1].wrapping_sub(pair[0]);
        acc = acc
            .wrapping_mul(17)
            .wrapping_add((diff ^ (idx as i64)).wrapping_add(ordered));
    }
    acc
}

pub const CASES: &[(&str, fn(i64) -> i64)] = &[
    ("case_boundary_int_extremes", case_boundary_int_extremes),
    ("case_boundary_shifts_and_casts", case_boundary_shifts_and_casts),
    ("case_boundary_float_edges", case_boundary_float_edges),
    ("case_boundary_seeded_prng", case_boundary_seeded_prng),
];

#[cfg(target_arch = "x86_64")]
fn run_simd() {
    use std::arch::x86_64::*;

    fn lanes(v: __m128i) -> [u16; 8] {
        let mut out = [0u16; 8];
        // SAFETY: `out` is 16 bytes of valid writable stack memory.
        unsafe { _mm_storeu_si128(out.as_mut_ptr().cast(), v) };
        out
    }

    // SAFETY: SSE2 is baseline on `x86_64`.
    unsafe {
        let a = _mm_set_epi64x(0x0004_0003_0002_0001, 0x7fff_8000_ffff_1234);
        let b = _mm_set_epi64x(0x0100_0200_0300_0400, 0x0002_0003_0004_0005);
        println!("mulhi {:?}", lanes(_mm_mulhi_epu16(a, b)));
        println!("mullo {:?}", lanes(_mm_mullo_epi16(a, b)));
        println!("sub {:?}", lanes(_mm_sub_epi16(a, b)));
        println!("srli16 {:?}", lanes(_mm_srli_epi16::<3>(a)));
        println!("slli16 {:?}", lanes(_mm_slli_epi16::<2>(a)));
        println!("srli32 {:?}", lanes(_mm_srli_epi32::<5>(a)));
        println!("slli32 {:?}", lanes(_mm_slli_epi32::<7>(a)));
        println!("srli64 {:?}", lanes(_mm_srli_epi64::<9>(a)));
        println!("add64 {:?}", lanes(_mm_add_epi64(a, b)));
        println!("or {:?}", lanes(_mm_or_si128(a, b)));
        println!("mul32 {:?}", lanes(_mm_mul_epu32(a, b)));
        println!("shuf32 {:?}", lanes(_mm_shuffle_epi32::<0b00_01_10_11>(a)));
        println!("gt {:?}", lanes(_mm_cmpgt_epi8(a, b)));
        println!("mask {}", _mm_movemask_epi8(_mm_cmpgt_epi8(a, b)));
        println!("low {}", _mm_cvtsi128_si64(a));
        println!("zero {:?}", lanes(_mm_setzero_si128()));
    }
}

#[cfg(target_arch = "aarch64")]
fn run_simd() {
    use std::arch::aarch64::*;

    // SAFETY: NEON is baseline on `aarch64`.
    unsafe {
        let a = vld1q_u8([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16].as_ptr());
        let idx = vld1q_u8([15, 0, 7, 255, 3, 2, 1, 0, 8, 9, 10, 11, 99, 12, 13, 14].as_ptr());
        let mut tbl = [0_u8; 16];
        vst1q_u8(tbl.as_mut_ptr(), vqtbl1q_u8(a, idx));
        let s1 = vld1q_s32([i32::MIN, 0x4000_0000, -0x2000_0000, 123_456_789].as_ptr());
        let s2 = vld1q_s32([i32::MIN, 0x4000_0000, 0x4000_0000, 987_654_321].as_ptr());
        let mut qd = [0_i32; 4];
        vst1q_s32(qd.as_mut_ptr(), vqdmulhq_s32(s1, s2));
        println!("tbl {tbl:?}");
        println!("qdmulh {qd:?}");
    }
}

fn main() {
    run_simd();
}

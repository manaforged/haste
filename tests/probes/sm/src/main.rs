use std::arch::x86_64::*;

fn lanes(v: __m128i) -> [u16; 8] {
    let mut out = [0u16; 8];
    unsafe { _mm_storeu_si128(out.as_mut_ptr().cast(), v) };
    out
}

fn main() {
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

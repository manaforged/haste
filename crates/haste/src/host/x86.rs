//! The LLVM x86 intrinsics that `core::arch` declares as foreign functions, as host functions that take each vector by address and write the result through their first argument.

/// The bytes of a 128-bit vector.
type V128 = [u8; 16];
/// The bytes of a 256-bit vector.
type V256 = [u8; 32];

macro_rules! x86 {
    ($($variant:ident = $link:literal, $feature:literal, $name:ident($($arg:ident: $ty:ty),*) -> $out:ty = $body:expr;)*) => {
        /// An LLVM x86 intrinsic that lowering calls as a host function.
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
        pub enum X86Symbol {
            $($variant,)*
        }

        #[cfg(target_arch = "x86_64")]
        mod code {
            use std::arch::x86_64::*;

            use super::{V128, V256};
            use super::load::*;

            $(
                #[target_feature(enable = $feature)]
                pub(super) unsafe extern "C" fn $name(out: *mut $out, $($arg: $ty),*) {
                    // SAFETY: compiled code calls this only where it may call the intrinsic, so the CPU has the feature, with addresses of vectors in its frame and a result slot of the result's size.
                    unsafe { out.write_unaligned(std::mem::transmute::<_, $out>($body)) }
                }
            )*
        }

        impl X86Symbol {
            const ALL: &[Self] = &[$(Self::$variant),*];

            /// The intrinsic that the foreign link name `name` names, when the host has a function for it.
            pub fn from_link(name: &str) -> Option<Self> {
                Self::ALL.iter().copied().find(|sym| sym.link_name() == name)
            }

            /// The LLVM link name of this intrinsic.
            pub const fn link_name(self) -> &'static str {
                match self {
                    $(Self::$variant => $link,)*
                }
            }

            /// The size in bytes of the result this intrinsic writes.
            pub const fn out_bytes(self) -> u32 {
                match self {
                    $(Self::$variant => size_of::<$out>() as u32,)*
                }
            }

            /// The address of the host function of this intrinsic, which exists on x86-64 hosts only.
            pub fn host_addr(self) -> Option<usize> {
                #[cfg(target_arch = "x86_64")]
                return Some(match self {
                    $(Self::$variant => code::$name as unsafe extern "C" fn(*mut $out, $($ty),*) as usize,)*
                });
                #[cfg(not(target_arch = "x86_64"))]
                None
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
mod load {
    use std::arch::x86_64::*;

    use super::{V128, V256};

    /// The integer vector at `at`.
    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn int(at: *const V128) -> __m128i {
        // SAFETY: the caller passes the address of 16 readable bytes.
        unsafe { _mm_loadu_si128(at.cast()) }
    }

    /// The float vector at `at`.
    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn float(at: *const V128) -> __m128 {
        // SAFETY: the caller passes the address of 16 readable bytes.
        unsafe { _mm_loadu_ps(at.cast()) }
    }

    /// The 256-bit integer vector at `at`.
    #[target_feature(enable = "avx")]
    pub(super) unsafe fn wide(at: *const V256) -> __m256i {
        // SAFETY: the caller passes the address of 32 readable bytes.
        unsafe { _mm256_loadu_si256(at.cast()) }
    }

    /// The low 128 bits of `value` with the round constant `rcon` added to the two words that `aeskeygenassist` rotates.
    #[target_feature(enable = "aes")]
    pub(super) fn key_assist(value: __m128i, rcon: u8) -> __m128i {
        _mm_xor_si128(_mm_aeskeygenassist_si128::<0>(value), _mm_set_epi32(i32::from(rcon), 0, i32::from(rcon), 0))
    }

    /// The carry-less product of the halves of `a` and `b` that bits 0 and 4 of `imm` select.
    #[target_feature(enable = "pclmulqdq")]
    pub(super) fn clmul(a: __m128i, b: __m128i, imm: u8) -> __m128i {
        match imm & 0x11 {
            0x00 => _mm_clmulepi64_si128::<0x00>(a, b),
            0x01 => _mm_clmulepi64_si128::<0x01>(a, b),
            0x10 => _mm_clmulepi64_si128::<0x10>(a, b),
            _ => _mm_clmulepi64_si128::<0x11>(a, b),
        }
    }

    /// Four SHA-1 rounds with the round function and constant that the low two bits of `func` select.
    #[target_feature(enable = "sha")]
    pub(super) fn sha1_rounds(a: __m128i, b: __m128i, func: i8) -> __m128i {
        match func & 3 {
            0 => _mm_sha1rnds4_epu32::<0>(a, b),
            1 => _mm_sha1rnds4_epu32::<1>(a, b),
            2 => _mm_sha1rnds4_epu32::<2>(a, b),
            _ => _mm_sha1rnds4_epu32::<3>(a, b),
        }
    }

    /// Each lane of `a` compared with the same lane of `b` by the `cmpps` predicate `imm`, as all ones when it holds and zero when not.
    pub(super) fn compare(a: [f32; 4], b: [f32; 4], imm: i8) -> [u32; 4] {
        const LESS: u8 = 1;
        const EQUAL: u8 = 2;
        const GREATER: u8 = 4;
        const UNORDERED: u8 = 8;
        const HOLDS: [u8; 16] = [2, 1, 3, 8, 13, 14, 12, 7, 10, 9, 11, 0, 5, 6, 4, 15];
        std::array::from_fn(|lane| {
            let relation = match a[lane].partial_cmp(&b[lane]) {
                Some(std::cmp::Ordering::Less) => LESS,
                Some(std::cmp::Ordering::Equal) => EQUAL,
                Some(std::cmp::Ordering::Greater) => GREATER,
                None => UNORDERED,
            };
            if HOLDS[(imm & 15) as usize] & relation != 0 { u32::MAX } else { 0 }
        })
    }
}

x86! {
    Aesenc = "llvm.x86.aesni.aesenc", "aes", aesenc(a: *const V128, b: *const V128) -> V128 = _mm_aesenc_si128(int(a), int(b));
    Aesenclast = "llvm.x86.aesni.aesenclast", "aes", aesenclast(a: *const V128, b: *const V128) -> V128 = _mm_aesenclast_si128(int(a), int(b));
    Aesdec = "llvm.x86.aesni.aesdec", "aes", aesdec(a: *const V128, b: *const V128) -> V128 = _mm_aesdec_si128(int(a), int(b));
    Aesdeclast = "llvm.x86.aesni.aesdeclast", "aes", aesdeclast(a: *const V128, b: *const V128) -> V128 = _mm_aesdeclast_si128(int(a), int(b));
    Aesimc = "llvm.x86.aesni.aesimc", "aes", aesimc(a: *const V128) -> V128 = _mm_aesimc_si128(int(a));
    AesAssist = "llvm.x86.aesni.aeskeygenassist", "aes", aeskeygenassist(a: *const V128, rcon: u8) -> V128 = key_assist(int(a), rcon);
    Pclmulqdq = "llvm.x86.pclmulqdq", "pclmulqdq", pclmulqdq(a: *const V128, b: *const V128, imm: u8) -> V128 = clmul(int(a), int(b), imm);
    Sha1rnds4 = "llvm.x86.sha1rnds4", "sha", sha1rnds4(a: *const V128, b: *const V128, func: i8) -> V128 = sha1_rounds(int(a), int(b), func);
    Sha1nexte = "llvm.x86.sha1nexte", "sha", sha1nexte(a: *const V128, b: *const V128) -> V128 = _mm_sha1nexte_epu32(int(a), int(b));
    Sha1msg1 = "llvm.x86.sha1msg1", "sha", sha1msg1(a: *const V128, b: *const V128) -> V128 = _mm_sha1msg1_epu32(int(a), int(b));
    Sha1msg2 = "llvm.x86.sha1msg2", "sha", sha1msg2(a: *const V128, b: *const V128) -> V128 = _mm_sha1msg2_epu32(int(a), int(b));
    Sha256rnds2 = "llvm.x86.sha256rnds2", "sha", sha256rnds2(a: *const V128, b: *const V128, k: *const V128) -> V128 = _mm_sha256rnds2_epu32(int(a), int(b), int(k));
    Sha256msg1 = "llvm.x86.sha256msg1", "sha", sha256msg1(a: *const V128, b: *const V128) -> V128 = _mm_sha256msg1_epu32(int(a), int(b));
    Sha256msg2 = "llvm.x86.sha256msg2", "sha", sha256msg2(a: *const V128, b: *const V128) -> V128 = _mm_sha256msg2_epu32(int(a), int(b));
    Pmaddwd = "llvm.x86.sse2.pmadd.wd", "sse2", pmaddwd(a: *const V128, b: *const V128) -> V128 = _mm_madd_epi16(int(a), int(b));
    Psadbw = "llvm.x86.sse2.psad.bw", "sse2", psadbw(a: *const V128, b: *const V128) -> V128 = _mm_sad_epu8(int(a), int(b));
    Packssdw = "llvm.x86.sse2.packssdw.128", "sse2", packssdw(a: *const V128, b: *const V128) -> V128 = _mm_packs_epi32(int(a), int(b));
    Packuswb = "llvm.x86.sse2.packuswb.128", "sse2", packuswb(a: *const V128, b: *const V128) -> V128 = _mm_packus_epi16(int(a), int(b));
    Psrld = "llvm.x86.sse2.psrl.d", "sse2", psrld(a: *const V128, count: *const V128) -> V128 = _mm_srl_epi32(int(a), int(count));
    Cvtps2dq = "llvm.x86.sse2.cvtps2dq", "sse2", cvtps2dq(a: *const V128) -> V128 = _mm_cvtps_epi32(float(a));
    Cvttps2dq = "llvm.x86.sse2.cvttps2dq", "sse2", cvttps2dq(a: *const V128) -> V128 = _mm_cvttps_epi32(float(a));
    Minps = "llvm.x86.sse.min.ps", "sse2", minps(a: *const V128, b: *const V128) -> V128 = _mm_min_ps(float(a), float(b));
    Maxps = "llvm.x86.sse.max.ps", "sse2", maxps(a: *const V128, b: *const V128) -> V128 = _mm_max_ps(float(a), float(b));
    Rcpps = "llvm.x86.sse.rcp.ps", "sse2", rcpps(a: *const V128) -> V128 = _mm_rcp_ps(float(a));
    Cmpps = "llvm.x86.sse.cmp.ps", "sse2", cmpps(a: *const V128, b: *const V128, imm: i8) -> V128 = compare(std::mem::transmute(float(a)), std::mem::transmute(float(b)), imm);
    Lddqu = "llvm.x86.sse3.ldu.dq", "sse3", lddqu(at: *const i8) -> V128 = _mm_lddqu_si128(at.cast());
    Pmaddubsw128 = "llvm.x86.ssse3.pmadd.ub.sw.128", "ssse3", pmaddubsw128(a: *const V128, b: *const V128) -> V128 = _mm_maddubs_epi16(int(a), int(b));
    Packusdw = "llvm.x86.sse41.packusdw", "sse4.1", packusdw(a: *const V128, b: *const V128) -> V128 = _mm_packus_epi32(int(a), int(b));
    Vlddqu = "llvm.x86.avx.ldu.dq.256", "avx", vlddqu(at: *const i8) -> V256 = _mm256_lddqu_si256(at.cast());
    Pmaddwd256 = "llvm.x86.avx2.pmadd.wd", "avx2", pmaddwd256(a: *const V256, b: *const V256) -> V256 = _mm256_madd_epi16(wide(a), wide(b));
    Pmaddubsw256 = "llvm.x86.avx2.pmadd.ub.sw", "avx2", pmaddubsw256(a: *const V256, b: *const V256) -> V256 = _mm256_maddubs_epi16(wide(a), wide(b));
    Packssdw256 = "llvm.x86.avx2.packssdw", "avx2", packssdw256(a: *const V256, b: *const V256) -> V256 = _mm256_packs_epi32(wide(a), wide(b));
    Packuswb256 = "llvm.x86.avx2.packuswb", "avx2", packuswb256(a: *const V256, b: *const V256) -> V256 = _mm256_packus_epi16(wide(a), wide(b));
    Packusdw256 = "llvm.x86.avx2.packusdw", "avx2", packusdw256(a: *const V256, b: *const V256) -> V256 = _mm256_packus_epi32(wide(a), wide(b));
    Psadbw256 = "llvm.x86.avx2.psad.bw", "avx2", psadbw256(a: *const V256, b: *const V256) -> V256 = _mm256_sad_epu8(wide(a), wide(b));
    Permd = "llvm.x86.avx2.permd", "avx2", permd(a: *const V256, b: *const V256) -> V256 = _mm256_permutevar8x32_epi32(wide(a), wide(b));
    Addcarry64 = "llvm.x86.addcarry.64", "sse2", addcarry64(carry: u8, a: u64, b: u64) -> (u8, u64) = {
        let (sum, first) = a.overflowing_add(b);
        let (sum, second) = sum.overflowing_add(u64::from(carry != 0));
        (u8::from(first | second), sum)
    };
    Subborrow64 = "llvm.x86.subborrow.64", "sse2", subborrow64(borrow: u8, a: u64, b: u64) -> (u8, u64) = {
        let (difference, first) = a.overflowing_sub(b);
        let (difference, second) = difference.overflowing_sub(u64::from(borrow != 0));
        (u8::from(first | second), difference)
    };
}

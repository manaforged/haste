//! The LLVM x86 intrinsics that `core::arch` declares as foreign functions, as host functions that take each vector by address and write the result through their first argument.

/// The bytes of a 128-bit vector.
type V128 = [u8; 16];
/// The bytes of a 256-bit vector.
type V256 = [u8; 32];
/// The bytes of a 512-bit vector.
type V512 = [u8; 64];

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

            use super::{V128, V256, V512};
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

            /// The size in bytes of the integer this intrinsic returns, when it returns one rather than a vector.
            pub const fn scalar_bytes(self) -> Option<u32> {
                match self {
                    $(Self::$variant => <$out as Ret>::SCALAR,)*
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

/// How the result of a host intrinsic reaches compiled code: written to its result slot, or returned there as an integer that compiled code loads.
trait Ret {
    const SCALAR: Option<u32>;
}

impl Ret for V128 {
    const SCALAR: Option<u32> = None;
}

impl Ret for V256 {
    const SCALAR: Option<u32> = None;
}

impl Ret for V512 {
    const SCALAR: Option<u32> = None;
}

impl Ret for (u8, u64) {
    const SCALAR: Option<u32> = None;
}

impl Ret for i32 {
    const SCALAR: Option<u32> = Some(4);
}

impl Ret for u32 {
    const SCALAR: Option<u32> = Some(4);
}

impl Ret for u64 {
    const SCALAR: Option<u32> = Some(8);
}

#[cfg(target_arch = "x86_64")]
mod load {
    use std::arch::x86_64::*;

    use super::{V128, V256, V512};

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

    /// The double vector at `at`.
    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn double(at: *const V128) -> __m128d {
        // SAFETY: the caller passes the address of 16 readable bytes.
        unsafe { _mm_loadu_pd(at.cast()) }
    }

    /// The 256-bit float vector at `at`.
    #[target_feature(enable = "avx")]
    pub(super) unsafe fn wide_float(at: *const V256) -> __m256 {
        // SAFETY: the caller passes the address of 32 readable bytes.
        unsafe { _mm256_loadu_ps(at.cast()) }
    }

    /// The 256-bit double vector at `at`.
    #[target_feature(enable = "avx")]
    pub(super) unsafe fn wide_double(at: *const V256) -> __m256d {
        // SAFETY: the caller passes the address of 32 readable bytes.
        unsafe { _mm256_loadu_pd(at.cast()) }
    }

    /// The `N` lanes of type `T` at `at`.
    pub(super) unsafe fn lanes<T: Copy, const N: usize>(at: *const u8) -> [T; N] {
        // SAFETY: the caller passes the address of a vector of `N` lanes of `T`.
        unsafe { at.cast::<[T; N]>().read_unaligned() }
    }

    /// Each lane of `values` rounded as the `roundps` immediate `imm` selects, with the current rounding mode taken as round to nearest even.
    pub(super) fn round<const N: usize, T: Copy + Into<f64>>(values: [T; N], imm: i32, back: fn(f64) -> T) -> [T; N] {
        values.map(|value| {
            let value: f64 = value.into();
            back(match if imm & 4 != 0 { 0 } else { imm & 3 } {
                0 => value.round_ties_even(),
                1 => value.floor(),
                2 => value.ceil(),
                _ => value.trunc(),
            })
        })
    }

    /// Whether `a` and `b` stand in the relation that the `cmpps` predicate `imm` names.
    fn holds(a: f64, b: f64, imm: i8) -> bool {
        const LESS: u8 = 1;
        const EQUAL: u8 = 2;
        const GREATER: u8 = 4;
        const UNORDERED: u8 = 8;
        const HOLDS: [u8; 16] = [2, 1, 3, 8, 13, 14, 12, 7, 10, 9, 11, 0, 5, 6, 4, 15];
        let relation = match a.partial_cmp(&b) {
            Some(std::cmp::Ordering::Less) => LESS,
            Some(std::cmp::Ordering::Equal) => EQUAL,
            Some(std::cmp::Ordering::Greater) => GREATER,
            None => UNORDERED,
        };
        HOLDS[(imm & 15) as usize] & relation != 0
    }

    /// Each float lane of `a` compared with the same lane of `b` by the predicate `imm`, as all ones when it holds and zero when not.
    pub(super) fn compare<const N: usize>(a: [f32; N], b: [f32; N], imm: i8) -> [u32; N] {
        std::array::from_fn(|lane| if holds(a[lane].into(), b[lane].into(), imm) { u32::MAX } else { 0 })
    }

    /// Each double lane of `a` compared with the same lane of `b` by the predicate `imm`, as all ones when it holds and zero when not.
    pub(super) fn compare_double<const N: usize>(a: [f64; N], b: [f64; N], imm: i8) -> [u64; N] {
        std::array::from_fn(|lane| if holds(a[lane], b[lane], imm) { u64::MAX } else { 0 })
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

    /// The 512-bit integer vector at `at`.
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn zmm(at: *const V512) -> __m512i {
        // SAFETY: the caller passes the address of 64 readable bytes.
        unsafe { _mm512_loadu_si512(at.cast()) }
    }

    /// The carry-less products of the halves of each 128-bit lane of `a` and `b` that bits 0 and 4 of `imm` select.
    #[target_feature(enable = "vpclmulqdq,avx")]
    pub(super) fn clmul256(a: __m256i, b: __m256i, imm: u8) -> __m256i {
        match imm & 0x11 {
            0x00 => _mm256_clmulepi64_epi128::<0x00>(a, b),
            0x01 => _mm256_clmulepi64_epi128::<0x01>(a, b),
            0x10 => _mm256_clmulepi64_epi128::<0x10>(a, b),
            _ => _mm256_clmulepi64_epi128::<0x11>(a, b),
        }
    }

    /// The carry-less products of the halves of each 128-bit lane of `a` and `b` that bits 0 and 4 of `imm` select.
    #[target_feature(enable = "vpclmulqdq,avx512f")]
    pub(super) fn clmul512(a: __m512i, b: __m512i, imm: u8) -> __m512i {
        match imm & 0x11 {
            0x00 => _mm512_clmulepi64_epi128::<0x00>(a, b),
            0x01 => _mm512_clmulepi64_epi128::<0x01>(a, b),
            0x10 => _mm512_clmulepi64_epi128::<0x10>(a, b),
            _ => _mm512_clmulepi64_epi128::<0x11>(a, b),
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
}

x86! {
    Aesenc = "llvm.x86.aesni.aesenc", "aes", aesenc(a: *const V128, b: *const V128) -> V128 = _mm_aesenc_si128(int(a), int(b));
    Aesenclast = "llvm.x86.aesni.aesenclast", "aes", aesenclast(a: *const V128, b: *const V128) -> V128 = _mm_aesenclast_si128(int(a), int(b));
    Aesdec = "llvm.x86.aesni.aesdec", "aes", aesdec(a: *const V128, b: *const V128) -> V128 = _mm_aesdec_si128(int(a), int(b));
    Aesdeclast = "llvm.x86.aesni.aesdeclast", "aes", aesdeclast(a: *const V128, b: *const V128) -> V128 = _mm_aesdeclast_si128(int(a), int(b));
    Aesimc = "llvm.x86.aesni.aesimc", "aes", aesimc(a: *const V128) -> V128 = _mm_aesimc_si128(int(a));
    AesAssist = "llvm.x86.aesni.aeskeygenassist", "aes", aeskeygenassist(a: *const V128, rcon: u8) -> V128 = key_assist(int(a), rcon);
    Pclmulqdq = "llvm.x86.pclmulqdq", "pclmulqdq", pclmulqdq(a: *const V128, b: *const V128, imm: u8) -> V128 = clmul(int(a), int(b), imm);
    Pclmulqdq256 = "llvm.x86.pclmulqdq.256", "vpclmulqdq,avx", pclmulqdq256(a: *const V256, b: *const V256, imm: u8) -> V256 = clmul256(wide(a), wide(b), imm);
    Pclmulqdq512 = "llvm.x86.pclmulqdq.512", "vpclmulqdq,avx512f", pclmulqdq512(a: *const V512, b: *const V512, imm: u8) -> V512 = clmul512(zmm(a), zmm(b), imm);
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
    Cmpps = "llvm.x86.sse.cmp.ps", "sse2", cmpps(a: *const V128, b: *const V128, imm: i8) -> V128 = compare::<4>(lanes(a.cast()), lanes(b.cast()), imm);
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
    Rsqrtps = "llvm.x86.sse.rsqrt.ps", "sse", rsqrtps(a: *const V128) -> V128 = _mm_rsqrt_ps(float(a));
    Minss = "llvm.x86.sse.min.ss", "sse", minss(a: *const V128, b: *const V128) -> V128 = _mm_min_ss(float(a), float(b));
    Maxss = "llvm.x86.sse.max.ss", "sse", maxss(a: *const V128, b: *const V128) -> V128 = _mm_max_ss(float(a), float(b));
    Cmpss = "llvm.x86.sse.cmp.ss", "sse", cmpss(a: *const V128, b: *const V128, imm: i8) -> V128 = {
        let mut out: [u32; 4] = lanes(a.cast());
        out[0] = compare::<1>([lanes::<f32, 4>(a.cast())[0]], [lanes::<f32, 4>(b.cast())[0]], imm)[0];
        out
    };
    Psllw = "llvm.x86.sse2.psll.w", "sse2", psllw(a: *const V128, count: *const V128) -> V128 = _mm_sll_epi16(int(a), int(count));
    Pslld = "llvm.x86.sse2.psll.d", "sse2", pslld(a: *const V128, count: *const V128) -> V128 = _mm_sll_epi32(int(a), int(count));
    Psllq = "llvm.x86.sse2.psll.q", "sse2", psllq(a: *const V128, count: *const V128) -> V128 = _mm_sll_epi64(int(a), int(count));
    Psraw = "llvm.x86.sse2.psra.w", "sse2", psraw(a: *const V128, count: *const V128) -> V128 = _mm_sra_epi16(int(a), int(count));
    Psrad = "llvm.x86.sse2.psra.d", "sse2", psrad(a: *const V128, count: *const V128) -> V128 = _mm_sra_epi32(int(a), int(count));
    Psrlw = "llvm.x86.sse2.psrl.w", "sse2", psrlw(a: *const V128, count: *const V128) -> V128 = _mm_srl_epi16(int(a), int(count));
    Psrlq = "llvm.x86.sse2.psrl.q", "sse2", psrlq(a: *const V128, count: *const V128) -> V128 = _mm_srl_epi64(int(a), int(count));
    Packsswb = "llvm.x86.sse2.packsswb.128", "sse2", packsswb(a: *const V128, b: *const V128) -> V128 = _mm_packs_epi16(int(a), int(b));
    Minpd = "llvm.x86.sse2.min.pd", "sse2", minpd(a: *const V128, b: *const V128) -> V128 = _mm_min_pd(double(a), double(b));
    Maxpd = "llvm.x86.sse2.max.pd", "sse2", maxpd(a: *const V128, b: *const V128) -> V128 = _mm_max_pd(double(a), double(b));
    Minsd = "llvm.x86.sse2.min.sd", "sse2", minsd(a: *const V128, b: *const V128) -> V128 = _mm_min_sd(double(a), double(b));
    Maxsd = "llvm.x86.sse2.max.sd", "sse2", maxsd(a: *const V128, b: *const V128) -> V128 = _mm_max_sd(double(a), double(b));
    Cmppd = "llvm.x86.sse2.cmp.pd", "sse2", cmppd(a: *const V128, b: *const V128, imm: i8) -> V128 = compare_double::<2>(lanes(a.cast()), lanes(b.cast()), imm);
    Cvtpd2dq = "llvm.x86.sse2.cvtpd2dq", "sse2", cvtpd2dq(a: *const V128) -> V128 = _mm_cvtpd_epi32(double(a));
    Cvttpd2dq = "llvm.x86.sse2.cvttpd2dq", "sse2", cvttpd2dq(a: *const V128) -> V128 = _mm_cvttpd_epi32(double(a));
    Pmulhrsw128 = "llvm.x86.ssse3.pmul.hr.sw.128", "ssse3", pmulhrsw128(a: *const V128, b: *const V128) -> V128 = _mm_mulhrs_epi16(int(a), int(b));
    Psignb128 = "llvm.x86.ssse3.psign.b.128", "ssse3", psignb128(a: *const V128, b: *const V128) -> V128 = _mm_sign_epi8(int(a), int(b));
    Psignw128 = "llvm.x86.ssse3.psign.w.128", "ssse3", psignw128(a: *const V128, b: *const V128) -> V128 = _mm_sign_epi16(int(a), int(b));
    Psignd128 = "llvm.x86.ssse3.psign.d.128", "ssse3", psignd128(a: *const V128, b: *const V128) -> V128 = _mm_sign_epi32(int(a), int(b));
    Roundps = "llvm.x86.sse41.round.ps", "sse4.1", roundps(a: *const V128, imm: i32) -> V128 = round::<4, f32>(lanes(a.cast()), imm, |x| x as f32);
    Roundpd = "llvm.x86.sse41.round.pd", "sse4.1", roundpd(a: *const V128, imm: i32) -> V128 = round::<2, f64>(lanes(a.cast()), imm, |x| x);
    Ptestnzc = "llvm.x86.sse41.ptestnzc", "sse4.1", ptestnzc(a: *const V128, b: *const V128) -> i32 = _mm_testnzc_si128(int(a), int(b));
    Phminposuw = "llvm.x86.sse41.phminposuw", "sse4.1", phminposuw(a: *const V128) -> V128 = _mm_minpos_epu16(int(a));
    Crc32u8 = "llvm.x86.sse42.crc32.32.8", "sse4.2", crc32u8(crc: u32, value: u8) -> u32 = _mm_crc32_u8(crc, value);
    Crc32u16 = "llvm.x86.sse42.crc32.32.16", "sse4.2", crc32u16(crc: u32, value: u16) -> u32 = _mm_crc32_u16(crc, value);
    Crc32u32 = "llvm.x86.sse42.crc32.32.32", "sse4.2", crc32u32(crc: u32, value: u32) -> u32 = _mm_crc32_u32(crc, value);
    Crc32u64 = "llvm.x86.sse42.crc32.64.64", "sse4.2", crc32u64(crc: u64, value: u64) -> u64 = _mm_crc32_u64(crc, value);
    Minps256 = "llvm.x86.avx.min.ps.256", "avx", minps256(a: *const V256, b: *const V256) -> V256 = _mm256_min_ps(wide_float(a), wide_float(b));
    Maxps256 = "llvm.x86.avx.max.ps.256", "avx", maxps256(a: *const V256, b: *const V256) -> V256 = _mm256_max_ps(wide_float(a), wide_float(b));
    Minpd256 = "llvm.x86.avx.min.pd.256", "avx", minpd256(a: *const V256, b: *const V256) -> V256 = _mm256_min_pd(wide_double(a), wide_double(b));
    Maxpd256 = "llvm.x86.avx.max.pd.256", "avx", maxpd256(a: *const V256, b: *const V256) -> V256 = _mm256_max_pd(wide_double(a), wide_double(b));
    Roundps256 = "llvm.x86.avx.round.ps.256", "avx", roundps256(a: *const V256, imm: i32) -> V256 = round::<8, f32>(lanes(a.cast()), imm, |x| x as f32);
    Roundpd256 = "llvm.x86.avx.round.pd.256", "avx", roundpd256(a: *const V256, imm: i32) -> V256 = round::<4, f64>(lanes(a.cast()), imm, |x| x);
    Cmpps256 = "llvm.x86.avx.cmp.ps.256", "avx", cmpps256(a: *const V256, b: *const V256, imm: i8) -> V256 = compare::<8>(lanes(a.cast()), lanes(b.cast()), imm);
    Cmppd256 = "llvm.x86.avx.cmp.pd.256", "avx", cmppd256(a: *const V256, b: *const V256, imm: i8) -> V256 = compare_double::<4>(lanes(a.cast()), lanes(b.cast()), imm);
    Cvtps2dq256 = "llvm.x86.avx.cvt.ps2dq.256", "avx", cvtps2dq256(a: *const V256) -> V256 = _mm256_cvtps_epi32(wide_float(a));
    Cvttps2dq256 = "llvm.x86.avx.cvtt.ps2dq.256", "avx", cvttps2dq256(a: *const V256) -> V256 = _mm256_cvttps_epi32(wide_float(a));
    Cvtpd2dq256 = "llvm.x86.avx.cvt.pd2dq.256", "avx", cvtpd2dq256(a: *const V256) -> V128 = _mm256_cvtpd_epi32(wide_double(a));
    Cvttpd2dq256 = "llvm.x86.avx.cvtt.pd2dq.256", "avx", cvttpd2dq256(a: *const V256) -> V128 = _mm256_cvttpd_epi32(wide_double(a));
    Vpermilvarps = "llvm.x86.avx.vpermilvar.ps", "avx", vpermilvarps(a: *const V128, b: *const V128) -> V128 = _mm_permutevar_ps(float(a), int(b));
    Vpermilvarps256 = "llvm.x86.avx.vpermilvar.ps.256", "avx", vpermilvarps256(a: *const V256, b: *const V256) -> V256 = _mm256_permutevar_ps(wide_float(a), wide(b));
    Vpermilvarpd = "llvm.x86.avx.vpermilvar.pd", "avx", vpermilvarpd(a: *const V128, b: *const V128) -> V128 = _mm_permutevar_pd(double(a), int(b));
    Vpermilvarpd256 = "llvm.x86.avx.vpermilvar.pd.256", "avx", vpermilvarpd256(a: *const V256, b: *const V256) -> V256 = _mm256_permutevar_pd(wide_double(a), wide(b));
    Rcpps256 = "llvm.x86.avx.rcp.ps.256", "avx", rcpps256(a: *const V256) -> V256 = _mm256_rcp_ps(wide_float(a));
    Rsqrtps256 = "llvm.x86.avx.rsqrt.ps.256", "avx", rsqrtps256(a: *const V256) -> V256 = _mm256_rsqrt_ps(wide_float(a));
    Ptestnzc256 = "llvm.x86.avx.ptestnzc.256", "avx", ptestnzc256(a: *const V256, b: *const V256) -> i32 = _mm256_testnzc_si256(wide(a), wide(b));
    Vtestzps256 = "llvm.x86.avx.vtestz.ps.256", "avx", vtestzps256(a: *const V256, b: *const V256) -> i32 = _mm256_testz_ps(wide_float(a), wide_float(b));
    Vtestcps256 = "llvm.x86.avx.vtestc.ps.256", "avx", vtestcps256(a: *const V256, b: *const V256) -> i32 = _mm256_testc_ps(wide_float(a), wide_float(b));
    Vtestnzcps256 = "llvm.x86.avx.vtestnzc.ps.256", "avx", vtestnzcps256(a: *const V256, b: *const V256) -> i32 = _mm256_testnzc_ps(wide_float(a), wide_float(b));
    Vtestzpd256 = "llvm.x86.avx.vtestz.pd.256", "avx", vtestzpd256(a: *const V256, b: *const V256) -> i32 = _mm256_testz_pd(wide_double(a), wide_double(b));
    Vtestcpd256 = "llvm.x86.avx.vtestc.pd.256", "avx", vtestcpd256(a: *const V256, b: *const V256) -> i32 = _mm256_testc_pd(wide_double(a), wide_double(b));
    Vtestnzcpd256 = "llvm.x86.avx.vtestnzc.pd.256", "avx", vtestnzcpd256(a: *const V256, b: *const V256) -> i32 = _mm256_testnzc_pd(wide_double(a), wide_double(b));
    Vtestnzcps = "llvm.x86.avx.vtestnzc.ps", "avx", vtestnzcps(a: *const V128, b: *const V128) -> i32 = _mm_testnzc_ps(float(a), float(b));
    Vtestnzcpd = "llvm.x86.avx.vtestnzc.pd", "avx", vtestnzcpd(a: *const V128, b: *const V128) -> i32 = _mm_testnzc_pd(double(a), double(b));
    Pmulhrsw256 = "llvm.x86.avx2.pmul.hr.sw", "avx2", pmulhrsw256(a: *const V256, b: *const V256) -> V256 = _mm256_mulhrs_epi16(wide(a), wide(b));
    Packsswb256 = "llvm.x86.avx2.packsswb", "avx2", packsswb256(a: *const V256, b: *const V256) -> V256 = _mm256_packs_epi16(wide(a), wide(b));
    Psignb256 = "llvm.x86.avx2.psign.b", "avx2", psignb256(a: *const V256, b: *const V256) -> V256 = _mm256_sign_epi8(wide(a), wide(b));
    Psignw256 = "llvm.x86.avx2.psign.w", "avx2", psignw256(a: *const V256, b: *const V256) -> V256 = _mm256_sign_epi16(wide(a), wide(b));
    Psignd256 = "llvm.x86.avx2.psign.d", "avx2", psignd256(a: *const V256, b: *const V256) -> V256 = _mm256_sign_epi32(wide(a), wide(b));
    Psllw256 = "llvm.x86.avx2.psll.w", "avx2", psllw256(a: *const V256, count: *const V128) -> V256 = _mm256_sll_epi16(wide(a), int(count));
    Pslld256 = "llvm.x86.avx2.psll.d", "avx2", pslld256(a: *const V256, count: *const V128) -> V256 = _mm256_sll_epi32(wide(a), int(count));
    Psllq256 = "llvm.x86.avx2.psll.q", "avx2", psllq256(a: *const V256, count: *const V128) -> V256 = _mm256_sll_epi64(wide(a), int(count));
    Psraw256 = "llvm.x86.avx2.psra.w", "avx2", psraw256(a: *const V256, count: *const V128) -> V256 = _mm256_sra_epi16(wide(a), int(count));
    Psrad256 = "llvm.x86.avx2.psra.d", "avx2", psrad256(a: *const V256, count: *const V128) -> V256 = _mm256_sra_epi32(wide(a), int(count));
    Psrlw256 = "llvm.x86.avx2.psrl.w", "avx2", psrlw256(a: *const V256, count: *const V128) -> V256 = _mm256_srl_epi16(wide(a), int(count));
    Psrld256 = "llvm.x86.avx2.psrl.d", "avx2", psrld256(a: *const V256, count: *const V128) -> V256 = _mm256_srl_epi32(wide(a), int(count));
    Psrlq256 = "llvm.x86.avx2.psrl.q", "avx2", psrlq256(a: *const V256, count: *const V128) -> V256 = _mm256_srl_epi64(wide(a), int(count));
    Permps = "llvm.x86.avx2.permps", "avx2", permps(a: *const V256, b: *const V256) -> V256 = _mm256_permutevar8x32_ps(wide_float(a), wide(b));
    Bextr32 = "llvm.x86.bmi.bextr.32", "bmi1", bextr32(a: u32, control: u32) -> u32 = _bextr2_u32(a, control);
    Bextr64 = "llvm.x86.bmi.bextr.64", "bmi1", bextr64(a: u64, control: u64) -> u64 = _bextr2_u64(a, control);
    Bzhi32 = "llvm.x86.bmi.bzhi.32", "bmi2", bzhi32(a: u32, index: u32) -> u32 = _bzhi_u32(a, index);
    Bzhi64 = "llvm.x86.bmi.bzhi.64", "bmi2", bzhi64(a: u64, index: u64) -> u64 = _bzhi_u64(a, index as u32);
    Pdep32 = "llvm.x86.bmi.pdep.32", "bmi2", pdep32(a: u32, mask: u32) -> u32 = _pdep_u32(a, mask);
    Pdep64 = "llvm.x86.bmi.pdep.64", "bmi2", pdep64(a: u64, mask: u64) -> u64 = _pdep_u64(a, mask);
    Pext32 = "llvm.x86.bmi.pext.32", "bmi2", pext32(a: u32, mask: u32) -> u32 = _pext_u32(a, mask);
    Pext64 = "llvm.x86.bmi.pext.64", "bmi2", pext64(a: u64, mask: u64) -> u64 = _pext_u64(a, mask);
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

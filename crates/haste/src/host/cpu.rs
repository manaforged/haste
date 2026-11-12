//! The `cpuid` and `xgetbv` answers that compiled code sees: the host's own, with the families in [`UNENCODED`] reported absent, as a hypervisor hides extensions from a guest.

/// A family of x86 extensions that `cpuid` can report absent, with the families that need it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Family {
    /// AVX and every VEX- or EVEX-encoded vector extension, with the YMM state.
    Avx,
    /// Fused multiply-add on XMM and YMM registers.
    Fma,
    /// AVX2 and AVX-VNNI, with AVX-512.
    Avx2,
    /// The AVX-512 extensions, with the opmask and ZMM state.
    Avx512,
    /// VAES and VPCLMULQDQ on YMM and ZMM registers.
    VectorCrypto,
}

/// The families without native encodings in the code generator, which compiled code sees as absent so that run-time dispatch picks paths this backend runs well.
pub const UNENCODED: &[Family] = &[Family::Avx];

/// The `cpuid` and `xgetbv` bits that a set of families occupies.
#[derive(Clone, Copy)]
pub(crate) struct Bits {
    pub(crate) leaf1_ecx: u32,
    pub(crate) leaf7_ebx: u32,
    pub(crate) leaf7_ecx: u32,
    pub(crate) leaf7_edx: u32,
    pub(crate) leaf7_1_eax: u32,
    pub(crate) leaf7_1_edx: u32,
    pub(crate) extended_ecx: u32,
    pub(crate) xcr0: u64,
}

const NONE: Bits = Bits { leaf1_ecx: 0, leaf7_ebx: 0, leaf7_ecx: 0, leaf7_edx: 0, leaf7_1_eax: 0, leaf7_1_edx: 0, extended_ecx: 0, xcr0: 0 };

impl Bits {
    const fn with(self, other: Bits) -> Bits {
        Bits {
            leaf1_ecx: self.leaf1_ecx | other.leaf1_ecx,
            leaf7_ebx: self.leaf7_ebx | other.leaf7_ebx,
            leaf7_ecx: self.leaf7_ecx | other.leaf7_ecx,
            leaf7_edx: self.leaf7_edx | other.leaf7_edx,
            leaf7_1_eax: self.leaf7_1_eax | other.leaf7_1_eax,
            leaf7_1_edx: self.leaf7_1_edx | other.leaf7_1_edx,
            extended_ecx: self.extended_ecx | other.extended_ecx,
            xcr0: self.xcr0 | other.xcr0,
        }
    }
}

impl Family {
    /// The bits of this family and of every family that needs it.
    const fn bits(self) -> Bits {
        match self {
            Self::Avx => Bits {
                leaf1_ecx: 1 << 28 | 1 << 29,
                leaf7_1_eax: 0b111 | 1 << 4 | 1 << 23,
                leaf7_1_edx: 1 << 4 | 1 << 5 | 1 << 10 | 1 << 19,
                extended_ecx: 1 << 11 | 1 << 16,
                xcr0: 1 << 2,
                ..NONE
            }
            .with(Self::Fma.bits())
            .with(Self::Avx2.bits())
            .with(Self::VectorCrypto.bits()),
            Self::Fma => Bits { leaf1_ecx: 1 << 12, ..NONE },
            Self::Avx2 => Bits { leaf7_ebx: 1 << 5, leaf7_1_eax: 1 << 4, ..NONE }.with(Self::Avx512.bits()),
            Self::Avx512 => Bits {
                leaf7_ebx: 1 << 16 | 1 << 17 | 1 << 21 | 1 << 26 | 1 << 27 | 1 << 28 | 1 << 30 | 1 << 31,
                leaf7_ecx: 1 << 1 | 1 << 6 | 1 << 11 | 1 << 12 | 1 << 14,
                leaf7_edx: 1 << 2 | 1 << 3 | 1 << 8 | 1 << 23,
                leaf7_1_eax: 1 << 5,
                xcr0: 0b111 << 5,
                ..NONE
            },
            Self::VectorCrypto => Bits { leaf7_ecx: 1 << 9 | 1 << 10, ..NONE },
        }
    }
}

const fn hidden() -> Bits {
    let mut bits = NONE;
    let mut at = 0;
    while at < UNENCODED.len() {
        bits = bits.with(UNENCODED[at].bits());
        at += 1;
    }
    bits
}

pub(crate) const HIDDEN: Bits = hidden();

/// The registers `[eax, ebx, ecx, edx]` that `cpuid` leaf `leaf`, subleaf `sub_leaf` gives compiled code.
pub(super) fn registers(leaf: u32, sub_leaf: u32) -> [u32; 4] {
    let [eax, ebx, ecx, edx] = arch::registers(leaf, sub_leaf);
    match (leaf, sub_leaf) {
        (1, _) => [eax, ebx, ecx & !HIDDEN.leaf1_ecx, edx],
        (7, 0) => [eax, ebx & !HIDDEN.leaf7_ebx, ecx & !HIDDEN.leaf7_ecx, edx & !HIDDEN.leaf7_edx],
        (7, 1) => [eax & !HIDDEN.leaf7_1_eax, ebx, ecx, edx & !HIDDEN.leaf7_1_edx],
        (0x8000_0001, _) => [eax, ebx, ecx & !HIDDEN.extended_ecx, edx],
        _ => [eax, ebx, ecx, edx],
    }
}

/// The extended control register `register` that `xgetbv` gives compiled code, or zero when the OS has not enabled `xgetbv`.
pub(super) fn control(register: u32) -> u64 {
    let value = arch::control(register);
    if register == 0 { value & !HIDDEN.xcr0 } else { value }
}

#[cfg(target_arch = "x86_64")]
mod arch {
    use std::arch::x86_64::{__cpuid, __cpuid_count, _xgetbv};

    const OSXSAVE: u32 = 1 << 27;

    pub(super) fn registers(leaf: u32, sub_leaf: u32) -> [u32; 4] {
        let regs = __cpuid_count(leaf, sub_leaf);
        [regs.eax, regs.ebx, regs.ecx, regs.edx]
    }

    pub(super) fn control(register: u32) -> u64 {
        let enabled = __cpuid(1).ecx & OSXSAVE != 0;
        // SAFETY: the OS set CR4.OSXSAVE, which `cpuid` leaf 1 reports, so `xgetbv` does not fault.
        if enabled { unsafe { _xgetbv(register) } } else { 0 }
    }
}

#[cfg(not(target_arch = "x86_64"))]
mod arch {
    pub(super) fn registers(_leaf: u32, _sub_leaf: u32) -> [u32; 4] {
        [0; 4]
    }

    pub(super) fn control(_register: u32) -> u64 {
        0
    }
}

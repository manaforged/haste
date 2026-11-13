const RUNTIME_TAG: u32 = 0x8000_0000;

/// How a relocation patches an immediate in machine code.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum RelocKind {
    Abs32,
    Abs64,
    Arm64Movw64,
}

/// A runtime helper linked into generated code.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum RuntimeSymbol {
    Memcpy = 0,
    Memmove = 1,
    Memset = 2,
    Memcmp = 3,
    Malloc = 4,
    Free = 5,
    I128Sdiv = 6,
    I128Udiv = 7,
    I128Srem = 8,
    I128Urem = 9,
    F32ToI128 = 10,
    F64ToI128 = 11,
    F32ToU128 = 12,
    F64ToU128 = 13,
    I128ToF32 = 14,
    I128ToF64 = 15,
    U128ToF32 = 16,
    U128ToF64 = 17,
}

const ALL_RUNTIME: [RuntimeSymbol; 18] = [
    RuntimeSymbol::Memcpy,
    RuntimeSymbol::Memmove,
    RuntimeSymbol::Memset,
    RuntimeSymbol::Memcmp,
    RuntimeSymbol::Malloc,
    RuntimeSymbol::Free,
    RuntimeSymbol::I128Sdiv,
    RuntimeSymbol::I128Udiv,
    RuntimeSymbol::I128Srem,
    RuntimeSymbol::I128Urem,
    RuntimeSymbol::F32ToI128,
    RuntimeSymbol::F64ToI128,
    RuntimeSymbol::F32ToU128,
    RuntimeSymbol::F64ToU128,
    RuntimeSymbol::I128ToF32,
    RuntimeSymbol::I128ToF64,
    RuntimeSymbol::U128ToF32,
    RuntimeSymbol::U128ToF64,
];

const LINK_NAMES: [&str; 18] = [
    "memcpy",
    "memmove",
    "memset",
    "memcmp",
    "malloc",
    "free",
    "__divti3",
    "__udivti3",
    "__modti3",
    "__umodti3",
    "__fixsfti",
    "__fixdfti",
    "__fixunssfti",
    "__fixunsdfti",
    "__floattisf",
    "__floattidf",
    "__floatuntisf",
    "__floatuntidf",
];

impl RuntimeSymbol {
    /// Decodes a runtime helper from its compact tag.
    pub fn from_raw(raw: u8) -> Option<Self> {
        ALL_RUNTIME.get(usize::from(raw)).copied()
    }

    /// The external linker symbol name of this helper.
    pub fn link_name(self) -> &'static str {
        LINK_NAMES[self as usize]
    }

    /// The in-process code address of this helper.
    pub fn host_addr(self) -> usize {
        host_addrs()[self as usize]
    }
}

fn host_addrs() -> [usize; 18] {
    [
        libc::memcpy as *const () as usize,
        libc::memmove as *const () as usize,
        libc::memset as *const () as usize,
        libc::memcmp as *const () as usize,
        libc::malloc as *const () as usize,
        libc::free as *const () as usize,
        sdiv as extern "C" fn(i128, i128) -> i128 as usize,
        udiv as extern "C" fn(i128, i128) -> i128 as usize,
        srem as extern "C" fn(i128, i128) -> i128 as usize,
        urem as extern "C" fn(i128, i128) -> i128 as usize,
        f32_to_i128 as extern "C" fn(f32) -> i128 as usize,
        f64_to_i128 as extern "C" fn(f64) -> i128 as usize,
        f32_to_u128 as extern "C" fn(f32) -> u128 as usize,
        f64_to_u128 as extern "C" fn(f64) -> u128 as usize,
        i128_to_f32 as extern "C" fn(i128) -> f32 as usize,
        i128_to_f64 as extern "C" fn(i128) -> f64 as usize,
        u128_to_f32 as extern "C" fn(u128) -> f32 as usize,
        u128_to_f64 as extern "C" fn(u128) -> f64 as usize,
    ]
}

/// One relocation inside a compiled function stencil.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct StencilReloc {
    pub offset: u32,
    pub symbol: u32,
    pub kind: RelocKind,
}

impl StencilReloc {
    pub(crate) fn symbol(offset: u32, index: u32, kind: RelocKind) -> Self {
        Self { offset, symbol: index, kind }
    }

    pub(crate) fn runtime(offset: u32, sym: RuntimeSymbol, kind: RelocKind) -> Self {
        Self { offset, symbol: RUNTIME_TAG | u32::from(sym as u8), kind }
    }

    /// The runtime helper this relocation names, when it targets one.
    pub fn as_runtime(self) -> Option<RuntimeSymbol> {
        (self.symbol & RUNTIME_TAG != 0).then(|| RuntimeSymbol::from_raw((self.symbol & !RUNTIME_TAG) as u8)).flatten()
    }

    /// The caller relocation table index, when this relocation is not a runtime helper.
    pub fn symbol_index(self) -> Option<usize> {
        (self.symbol & RUNTIME_TAG == 0).then_some(self.symbol as usize)
    }
}

pub(crate) fn patch_bytes(relocs: &[StencilReloc], symbols: &[u64], out: &mut [u8]) {
    for reloc in relocs {
        let value = match reloc.as_runtime() {
            Some(rt) => rt.host_addr() as u64,
            None => symbols[reloc.symbol as usize],
        };
        let at = reloc.offset as usize;
        match reloc.kind {
            RelocKind::Abs32 => {
                assert!(value < (1 << 31), "Abs32 target {value:#x} exceeds low 2 GiB");
                out[at..at + 4].copy_from_slice(&(value as u32).to_le_bytes());
            }
            RelocKind::Abs64 => out[at..at + 8].copy_from_slice(&value.to_le_bytes()),
            RelocKind::Arm64Movw64 => patch_arm64_movw64(&mut out[at..at + 16], value),
        }
    }
}

fn patch_arm64_movw64(out: &mut [u8], value: u64) {
    for half in 0..4 {
        let off = half * 4;
        let word = u32::from_le_bytes([out[off], out[off + 1], out[off + 2], out[off + 3]]);
        let bits = ((value >> (16 * half)) as u32) & 0xFFFF;
        let patched = (word & !(0xFFFF << 5)) | (bits << 5);
        out[off..off + 4].copy_from_slice(&patched.to_le_bytes());
    }
}

/// Streams a function's IR structure into a 128-bit footprint digest.
#[derive(Clone, Copy, Debug)]
pub struct FootprintHasher {
    lo: u64,
    hi: u64,
}

impl Default for FootprintHasher {
    fn default() -> Self {
        Self::new()
    }
}

impl FootprintHasher {
    /// Starts a fresh 128-bit footprint digest.
    pub fn new() -> Self {
        Self { lo: 0x243F_6A88_85A3_08D3, hi: 0x1319_8A2E_0370_7344 }
    }

    /// Mixes one 64-bit word into both lanes.
    pub fn word(&mut self, x: u64) {
        let mut a = (self.lo ^ x).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        a = (a ^ (a >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        let mut b = (self.hi ^ x.rotate_left(31)).wrapping_mul(0x94D0_49BB_1331_11EB);
        b = (b ^ (b >> 27)).wrapping_mul(0xD6E8_FEB8_6659_FD93);
        self.lo = a ^ (b >> 32);
        self.hi = b ^ (a >> 29);
    }

    /// Mixes a 128-bit value into the digest.
    pub fn u128(&mut self, x: u128) {
        self.word(x as u64);
        self.word((x >> 64) as u64);
    }

    /// Returns the 128-bit digest.
    pub fn finish(self) -> u128 {
        u128::from(self.lo) | (u128::from(self.hi) << 64)
    }
}

/// Parameters for tiering up a hot dictionary stub in JIT mode.
#[derive(Clone, Copy, Debug)]
pub struct DictTierUp {
    pub counter_addr: usize,
    pub threshold: u32,
    pub stub_entry: usize,
    pub encoded_index: u32,
}

extern "C" fn sdiv(x: i128, y: i128) -> i128 {
    x.wrapping_div(y)
}

extern "C" fn udiv(x: i128, y: i128) -> i128 {
    ((x as u128) / (y as u128)) as i128
}

extern "C" fn srem(x: i128, y: i128) -> i128 {
    x.wrapping_rem(y)
}

extern "C" fn urem(x: i128, y: i128) -> i128 {
    ((x as u128) % (y as u128)) as i128
}

extern "C" fn f32_to_i128(x: f32) -> i128 {
    x as i128
}

extern "C" fn f64_to_i128(x: f64) -> i128 {
    x as i128
}

extern "C" fn f32_to_u128(x: f32) -> u128 {
    x as u128
}

extern "C" fn f64_to_u128(x: f64) -> u128 {
    x as u128
}

extern "C" fn i128_to_f32(x: i128) -> f32 {
    x as f32
}

extern "C" fn i128_to_f64(x: i128) -> f64 {
    x as f64
}

extern "C" fn u128_to_f32(x: u128) -> f32 {
    x as f32
}

extern "C" fn u128_to_f64(x: u128) -> f64 {
    x as f64
}

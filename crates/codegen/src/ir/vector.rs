use super::{IntCC, Type, Value};

/// One scalar lane type of a SIMD register vector.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[repr(u8)]
pub enum Lane {
    I8,
    I16,
    I32,
    I64,
    F32,
    F64,
}

impl Lane {
    /// Width of one lane in bytes.
    pub const fn bytes(self) -> u32 {
        match self {
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 | Self::F32 => 4,
            Self::I64 | Self::F64 => 8,
        }
    }

    /// Width of one lane in bits.
    pub const fn bits(self) -> u32 {
        self.bytes() * 8
    }

    /// Whether this lane holds a floating-point number.
    pub const fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F64)
    }

    /// Whether this lane holds an integer.
    pub const fn is_int(self) -> bool {
        !self.is_float()
    }

    /// The scalar IR type matching this lane.
    pub const fn to_type(self) -> Type {
        match self {
            Self::I8 => Type::I8,
            Self::I16 => Type::I16,
            Self::I32 => Type::I32,
            Self::I64 => Type::I64,
            Self::F32 => Type::F32,
            Self::F64 => Type::F64,
        }
    }

    /// Converts a scalar IR type other than `I128` into its lane kind.
    pub const fn from_type(ty: Type) -> Option<Self> {
        match ty {
            Type::I8 => Some(Self::I8),
            Type::I16 => Some(Self::I16),
            Type::I32 => Some(Self::I32),
            Type::I64 => Some(Self::I64),
            Type::F32 => Some(Self::F32),
            Type::F64 => Some(Self::F64),
            Type::I128 | Type::Vec { .. } => None,
        }
    }

    /// The integer mask lane type matching the width of this lane.
    pub const fn int_mask(self) -> Self {
        match self {
            Self::F32 => Self::I32,
            Self::F64 => Self::I64,
            _ => self,
        }
    }
}

impl Type {
    /// Whether `self` is a 32-byte power-of-two vector type of at most 16 lanes, which only a host with 256-bit registers holds in one register.
    pub const fn is_wide_vec(self) -> bool {
        match self {
            Type::Vec { lane, count } => {
                count.is_power_of_two()
                    && count as usize <= MASK_LANES
                    && lane.bytes() * (count as u32) == 32
            }
            _ => false,
        }
    }

    /// The 32-byte vector type that holds two 16-byte vectors of type `self`: twice its lanes, or four `I64` lanes in place of 32 byte lanes.
    pub const fn double_vec(self) -> Option<Type> {
        match self {
            Type::Vec { lane, count }
                if lane.bytes() * (count as u32) == 16 && count.is_power_of_two() =>
            {
                Some(if count as usize * 2 <= MASK_LANES {
                    Type::Vec {
                        lane,
                        count: count * 2,
                    }
                } else {
                    Type::Vec {
                        lane: Lane::I64,
                        count: 4,
                    }
                })
            }
            _ => None,
        }
    }

    /// Whether `target` holds a value of vector type `self` in one register.
    pub fn is_register_vec(self, target: &crate::Target) -> bool {
        self.is_valid_vec() || (self.is_wide_vec() && target.wide_vectors())
    }

    /// The 16-byte vector type with the lanes of the 32-byte vector type `self`.
    pub const fn half_vec(self) -> Option<Type> {
        match self {
            Type::Vec { lane, count } if self.is_wide_vec() => Some(Type::Vec {
                lane,
                count: count / 2,
            }),
            _ => None,
        }
    }
}

const MASK_LANES: usize = 16;

/// Lane selection mask for `Op::Shuffle(Value, Value, ShuffleMask)`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ShuffleMask {
    /// Static indices in `0..2*count` (`0..count` from `a`, `count..2*count` from `b`), trailing `count..16` = `0`.
    Const([u8; MASK_LANES]),
    /// Dynamic byte-index vector (`Vec { lane: I8, count: 8 | 16 }`).
    Dyn(Value),
}

/// Horizontal vector reduction operator for `Op::Reduce(ReduceOp, Value)`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum ReduceOp {
    Add,
    Smin,
    Smax,
    Umin,
    Umax,
    Fmin,
    Fmax,
    And,
    Or,
    Xor,
    Movemask,
}

/// A target-specific vector instruction without a portable IR equivalent.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum TargetOpId {
    Arm64Sha256H,
    Arm64Sha256H2,
    Arm64Sha256Su0,
    Arm64Sha256Su1,
    Arm64AesE,
    Arm64AesD,
    Arm64AesMc,
    Arm64AesImc,
    X64Sha1NextE,
    X64Sha1Msg1,
    X64Sha1Msg2,
    X64Sha256Rnds2,
    X64Sha256Msg1,
    X64Sha256Msg2,
    X64AesEnc,
    X64AesEncLast,
    X64AesDec,
    X64AesDecLast,
    X64AesImc,
    /// The low 16 bytes of a 32-byte vector.
    X64Low128,
    /// The high 16 bytes of a 32-byte vector.
    X64High128,
    /// The 32-byte vector whose low half is the first operand and whose high half is the second.
    X64Concat128,
    /// `vpshufb` on a 32-byte vector: each 16-byte half of the first operand permuted by the bytes of the same half of the second.
    X64Pshufb256,
    /// `vpcmpeqb` on two 32-byte vectors: all ones in each byte where the operands' bytes are equal.
    X64Pcmpeqb256,
    /// `vpcmpgtb` on two 32-byte vectors: all ones in each byte where the first operand's signed byte is greater.
    X64Pcmpgtb256,
    /// `vpmovmskb` of a 32-byte vector: an `I32` of the top bit of each byte.
    X64Pmovmskb256,
    /// `pclmulqdq` with immediate 0x00: the carry-less product of the low quadwords of both operands.
    X64ClmulLowLow,
    /// `pclmulqdq` with immediate 0x01: the carry-less product of the first operand's high quadword and the second's low quadword.
    X64ClmulHighLow,
    /// `pclmulqdq` with immediate 0x10: the carry-less product of the first operand's low quadword and the second's high quadword.
    X64ClmulLowHigh,
    /// `pclmulqdq` with immediate 0x11: the carry-less product of the high quadwords of both operands.
    X64ClmulHighHigh,
    /// `adc` of two `I64`s after an `I8` carry in: the `I64` sum.
    X64AddCarry,
    /// `adc` of two `I64`s after an `I8` carry in: the `I8` carry out.
    X64AddCarryOut,
    /// `sbb` of two `I64`s after an `I8` borrow in: the `I64` difference.
    X64SubBorrow,
    /// `sbb` of two `I64`s after an `I8` borrow in: the `I8` borrow out.
    X64SubBorrowOut,
    /// `div` of the 128-bit dividend whose high `I64` is the first operand and low `I64` the second by the third, which must exceed the high half: the `I64` quotient.
    X64DivWide,
    /// `div` of the 128-bit dividend whose high `I64` is the first operand and low `I64` the second by the third, which must exceed the high half: the `I64` remainder.
    X64RemWide,
    /// `vpsadbw` on two 32-byte vectors: each quadword the sum of the absolute differences of its eight unsigned bytes.
    X64Psadbw256,
    /// `vpmaddubsw` on two 32-byte vectors: each word the saturated sum of two products of an unsigned byte of the first operand and a signed byte of the second.
    X64Pmaddubsw256,
    /// `vpmaddwd` on two 32-byte vectors: each doubleword the sum of two products of signed words.
    X64Pmaddwd256,
    /// `psadbw` on two 16-byte vectors: each quadword the sum of the absolute differences of its eight unsigned bytes.
    X64Psadbw,
    /// NEON `sqdmulh`: signed saturating doubling multiply returning the high half of each lane.
    Arm64Sqdmulh,
}

impl IntCC {
    /// How packed compares, which test only equality and greater-than, compute this one: `(equal, swap, invert)`.
    pub const fn packed(self) -> (bool, bool, bool) {
        match self {
            IntCC::Equal => (true, false, false),
            IntCC::NotEqual => (true, false, true),
            IntCC::SignedGreaterThan | IntCC::UnsignedGreaterThan => (false, false, false),
            IntCC::SignedLessThan | IntCC::UnsignedLessThan => (false, true, false),
            IntCC::SignedLessThanOrEqual | IntCC::UnsignedLessThanOrEqual => (false, false, true),
            IntCC::SignedGreaterThanOrEqual | IntCC::UnsignedGreaterThanOrEqual => {
                (false, true, true)
            }
        }
    }
}

/// Inline 1-to-3 operand list for `Op::TargetOp` (unused slots canonicalized to `Value(0)`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TinyArgs {
    args: [Value; 3],
    len: u8,
}

impl TinyArgs {
    /// Constructs a 1-operand list.
    pub const fn one(a: Value) -> Self {
        Self {
            args: [a, Value(0), Value(0)],
            len: 1,
        }
    }
    /// Constructs a 2-operand list.
    pub const fn two(a: Value, b: Value) -> Self {
        Self {
            args: [a, b, Value(0)],
            len: 2,
        }
    }
    /// Constructs a 3-operand list.
    pub const fn three(a: Value, b: Value, c: Value) -> Self {
        Self {
            args: [a, b, c],
            len: 3,
        }
    }
    /// Constructs a list from 1 to 3 values.
    pub fn from_slice(values: &[Value]) -> Option<Self> {
        match values {
            &[a] => Some(Self::one(a)),
            &[a, b] => Some(Self::two(a, b)),
            &[a, b, c] => Some(Self::three(a, b, c)),
            _ => None,
        }
    }
    /// Returns the active operands.
    pub fn as_slice(&self) -> &[Value] {
        &self.args[..usize::from(self.len)]
    }
    /// Returns mutable access to the active operands.
    pub fn as_mut_slice(&mut self) -> &mut [Value] {
        &mut self.args[..usize::from(self.len)]
    }
}

/// Bitset of IR facets invalidated or changed by a pass.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(crate) struct Domain(pub(crate) u8);

impl Domain {
    pub(crate) const NONE: Self = Self(0);
    pub(crate) const VALUES: Self = Self(1);
    pub(crate) const CFG: Self = Self(2);
    pub(crate) const MEM: Self = Self(4);

    /// Whether no facet bit is set.
    pub(crate) const fn is_empty(self) -> bool {
        self.0 == 0
    }
    /// Whether any bit in `other` is set in `self`.
    pub(crate) const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) != 0
    }
    /// Union of two domain bitsets.
    pub(crate) const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl std::ops::BitOr for Domain {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for Domain {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl std::ops::BitAnd for Domain {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}

/// Thread-safe cached CFG tied to a function's `cfg_rev`.
pub(crate) struct CfgCache(
    pub(super) std::sync::Mutex<Option<(u32, std::sync::Arc<crate::cfg::Cfg>)>>,
);

impl Default for CfgCache {
    fn default() -> Self {
        Self(std::sync::Mutex::new(None))
    }
}

impl Clone for CfgCache {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl PartialEq for CfgCache {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl Eq for CfgCache {}

impl std::fmt::Debug for CfgCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CfgCache")
    }
}

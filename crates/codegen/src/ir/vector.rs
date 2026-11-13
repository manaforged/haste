use super::{Type, Value};

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
    pub const fn bits(self) -> u32 { self.bytes() * 8 }

    /// Whether this lane holds a floating-point number.
    pub const fn is_float(self) -> bool { matches!(self, Self::F32 | Self::F64) }

    /// Whether this lane holds an integer.
    pub const fn is_int(self) -> bool { !self.is_float() }

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

/// Lane selection mask for `Op::Shuffle(Value, Value, ShuffleMask)`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ShuffleMask {
    /// Static indices in `0..2*count` (`0..count` from `a`, `count..2*count` from `b`), trailing `count..16` = `0`.
    Const([u8; 16]),
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
}

/// Inline 1-to-3 operand list for `Op::TargetOp` (unused slots canonicalized to `Value(0)`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TinyArgs {
    args: [Value; 3],
    len: u8,
}

impl TinyArgs {
    /// Constructs a 1-operand list.
    pub const fn one(a: Value) -> Self { Self { args: [a, Value(0), Value(0)], len: 1 } }
    /// Constructs a 2-operand list.
    pub const fn two(a: Value, b: Value) -> Self { Self { args: [a, b, Value(0)], len: 2 } }
    /// Constructs a 3-operand list.
    pub const fn three(a: Value, b: Value, c: Value) -> Self { Self { args: [a, b, c], len: 3 } }
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
    pub fn as_slice(&self) -> &[Value] { &self.args[..usize::from(self.len)] }
    /// Returns mutable access to the active operands.
    pub fn as_mut_slice(&mut self) -> &mut [Value] { &mut self.args[..usize::from(self.len)] }
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
    pub(crate) const fn is_empty(self) -> bool { self.0 == 0 }
    /// Whether any bit in `other` is set in `self`.
    pub(crate) const fn contains(self, other: Self) -> bool { (self.0 & other.0) != 0 }
    /// Union of two domain bitsets.
    pub(crate) const fn union(self, other: Self) -> Self { Self(self.0 | other.0) }
}

impl std::ops::BitOr for Domain {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self { Self(self.0 | rhs.0) }
}

impl std::ops::BitOrAssign for Domain {
    fn bitor_assign(&mut self, rhs: Self) { self.0 |= rhs.0; }
}

impl std::ops::BitAnd for Domain {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self { Self(self.0 & rhs.0) }
}

/// Thread-safe cached CFG tied to a function's `cfg_rev`.
pub(crate) struct CfgCache(pub(super) std::sync::Mutex<Option<(u32, std::sync::Arc<crate::cfg::Cfg>)>>);

impl Default for CfgCache {
    fn default() -> Self { Self(std::sync::Mutex::new(None)) }
}

impl Clone for CfgCache {
    fn clone(&self) -> Self { Self::default() }
}

impl PartialEq for CfgCache {
    fn eq(&self, _other: &Self) -> bool { true }
}

impl Eq for CfgCache {}

impl std::fmt::Debug for CfgCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str("CfgCache") }
}

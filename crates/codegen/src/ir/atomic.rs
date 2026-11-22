/// The operation of an atomic read-modify-write.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum AtomicRmwOp {
    Add,
    Sub,
    And,
    Nand,
    Or,
    Xor,
    Xchg,
    Umin,
    Umax,
    Smin,
    Smax,
}

/// Memory ordering of an atomic access or fence, matching `core::intrinsics::AtomicOrdering`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum AtomicOrd {
    Relaxed = 0,
    Release = 1,
    Acquire = 2,
    AcqRel = 3,
    SeqCst = 4,
}

impl AtomicOrd {
    /// Decodes a `core::intrinsics::AtomicOrdering` discriminant.
    pub const fn from_u8(byte: u8) -> Option<Self> {
        match byte {
            0 => Some(Self::Relaxed),
            1 => Some(Self::Release),
            2 => Some(Self::Acquire),
            3 => Some(Self::AcqRel),
            4 => Some(Self::SeqCst),
            _ => None,
        }
    }

    /// Whether this ordering includes acquire semantics.
    pub const fn has_acquire(self) -> bool {
        matches!(self, Self::Acquire | Self::AcqRel | Self::SeqCst)
    }

    /// Whether this ordering includes release semantics.
    pub const fn has_release(self) -> bool {
        matches!(self, Self::Release | Self::AcqRel | Self::SeqCst)
    }

    /// Whether this ordering requires sequential consistency.
    pub const fn is_seq_cst(self) -> bool {
        matches!(self, Self::SeqCst)
    }

    /// Returns the weakest ordering at least as strong as both `self` and `other`.
    pub const fn join(self, other: Self) -> Self {
        if self.is_seq_cst() || other.is_seq_cst() {
            return Self::SeqCst;
        }
        match (self.has_acquire() || other.has_acquire(), self.has_release() || other.has_release()) {
            (false, false) => Self::Relaxed,
            (false, true) => Self::Release,
            (true, false) => Self::Acquire,
            (true, true) => Self::AcqRel,
        }
    }
}

use super::Ins;
use crate::ir::{ByteCC, Op, Type, Value, VectorOp};

impl Ins<'_> {
    /// The sixteen byte comparisons `cc` of vectors `x` and `y`.
    pub fn byte_cmp(mut self, cc: ByteCC, x: Value, y: Value) -> Value {
        self.value(Op::ByteCmp(cc, x, y), Type::V128)
    }

    /// The vector whose sixteen bytes are all the byte `x`.
    pub fn splat(mut self, x: Value) -> Value {
        self.value(Op::Splat(x), Type::V128)
    }

    /// The 32-bit integer whose low sixteen bits are the top bits of the bytes of vector `x`.
    pub fn movemask(mut self, x: Value) -> Value {
        self.value(Op::Movemask(x), Type::I32)
    }

    /// The x64 vector instruction `op` on vectors `x` and `y`.
    pub fn vector(mut self, op: VectorOp, x: Value, y: Value) -> Value {
        self.value(Op::Vector(op, x, y), Type::V128)
    }

    /// Two SHA-256 rounds on state vectors `x` and `y` with the message and constants in `k`, as `sha256rnds2` gives.
    pub fn sha256_rounds(mut self, x: Value, y: Value, k: Value) -> Value {
        self.value(Op::Sha256Rounds(x, y, k), Type::V128)
    }
}

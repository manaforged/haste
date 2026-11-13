use super::Ins;
use crate::ir::{ByteCC, Convert, Op, Sha256Op, Type, Value, VectorOp};

impl Ins<'_> {
    /// Zero-extending 64-bit float-register value `x` (`F64`) to a 128-bit vector (`V128`).
    pub fn widen_vec64(self, x: Value) -> Value {
        self.convert(Convert::WidenVec64, Type::V128, x)
    }

    /// Low 64 bits of a 128-bit vector `x` (`V128`) as `F64`.
    pub fn narrow_vec64(self, x: Value) -> Value {
        self.convert(Convert::NarrowVec64, Type::F64, x)
    }

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
    pub fn sha256_rounds(self, x: Value, y: Value, k: Value) -> Value {
        self.sha256(Sha256Op::X86Rnds2, x, y, k)
    }

    /// One ARM64 `sha256h` hash update round over `(x, y, z)`.
    pub fn arm64_sha256h(self, x: Value, y: Value, z: Value) -> Value {
        self.sha256(Sha256Op::Arm64H, x, y, z)
    }

    /// One ARM64 `sha256h2` hash update round over `(x, y, z)`.
    pub fn arm64_sha256h2(self, x: Value, y: Value, z: Value) -> Value {
        self.sha256(Sha256Op::Arm64H2, x, y, z)
    }

    /// One ARM64 `sha256su1` message schedule update over `(x, y, z)`.
    pub fn arm64_sha256su1(self, x: Value, y: Value, z: Value) -> Value {
        self.sha256(Sha256Op::Arm64Su1, x, y, z)
    }

    fn sha256(mut self, op: Sha256Op, x: Value, y: Value, z: Value) -> Value {
        self.value(Op::Sha256Rounds(op, x, y, z), Type::V128)
    }
}

use super::Ins;
use crate::ir::{Op, ReduceOp, ShuffleMask, TargetOpId, TinyArgs, Type, Value};

impl Ins<'_> {
    /// Vector whose lanes all equal `scalar`.
    pub fn splat(mut self, vec_ty: Type, scalar: Value) -> Value {
        let zero = self.value(Op::Const(0), vec_ty);
        let first = self.value(Op::Insert(zero, scalar, 0), vec_ty);
        self.value(Op::Shuffle(first, first, ShuffleMask::Const([0; 16])), vec_ty)
    }

    /// Static lane permute across `a` (`0..count`) and `b` (`count..2*count`).
    pub fn shuffle(mut self, a: Value, b: Value, mask: [u8; 16]) -> Value {
        let ty = self.func.ty(a);
        self.value(Op::Shuffle(a, b, ShuffleMask::Const(mask)), ty)
    }

    /// Dynamic byte permute across `a` and `b` indexed by byte-vector `index`.
    pub fn shuffle_dyn(mut self, a: Value, b: Value, index: Value) -> Value {
        let ty = self.func.ty(a);
        self.value(Op::Shuffle(a, b, ShuffleMask::Dyn(index)), ty)
    }

    /// Extracts scalar lane `lane` from `vec`.
    pub fn extract(mut self, vec: Value, lane: u8) -> Value {
        let ty = self.func.ty(vec).lane().expect("extract requires a vector").to_type();
        self.value(Op::Extract(vec, lane), ty)
    }

    /// Inserts `scalar` at lane `lane` of `vec`.
    pub fn insert(mut self, vec: Value, scalar: Value, lane: u8) -> Value {
        let ty = self.func.ty(vec);
        self.value(Op::Insert(vec, scalar, lane), ty)
    }

    /// Reduces `vec` horizontally according to `op`.
    pub fn reduce(mut self, op: ReduceOp, vec: Value) -> Value {
        let ty = match op {
            ReduceOp::Movemask => Type::I32,
            _ => self.func.ty(vec).lane().expect("reduce requires a vector").to_type(),
        };
        self.value(Op::Reduce(op, vec), ty)
    }

    /// Emits a target-specific vector operation with 1 to 3 operands.
    pub fn target_op(mut self, op: TargetOpId, ty: Type, args: &[Value]) -> Value {
        let tiny = TinyArgs::from_slice(args).expect("target_op requires 1 to 3 arguments");
        self.value(Op::TargetOp(op, tiny), ty)
    }
}

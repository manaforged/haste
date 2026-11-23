use super::{Convert, Function, Inst, Op, StackSlot, TargetOpId, Value};

impl Function {
    /// The stack slot and byte offset that `value` names when its defining instruction is `Op::StackAddr`.
    pub fn stack_addr_of(&self, value: Value) -> Option<(StackSlot, i32)> {
        self.insts.iter().rev().find(|data| data.result == Some(value)).and_then(|data| match data.op {
            Op::StackAddr(slot, offset) => Some((slot, offset)),
            _ => None,
        })
    }

    /// Whether `v` is the zero constant, seen through bitcasts and 128-bit halves joined into a wide vector, with `def` giving each value's defining instruction.
    pub(crate) fn is_zero_vec(&self, def: &dyn Fn(Value) -> Option<Inst>, mut v: Value) -> bool {
        for _ in 0..4 {
            let Some(inst) = def(v) else { return false };
            match self.insts[inst.index()].op {
                Op::Const(0) => return true,
                Op::Convert(Convert::Bitcast, inner) => v = inner,
                Op::TargetOp(TargetOpId::X64Concat128, args) => return args.as_slice().iter().all(|&half| self.is_zero_vec(def, half)),
                _ => return false,
            }
        }
        false
    }
}

use super::Ins;
use crate::ir::{
    AtomicOrd, AtomicRmwOp, Binary, Block, Convert, FloatCC, Inst, IntCC, Op, SigRef, StackSlot,
    Type, Unary, Value,
};
use crate::reloc::RuntimeSymbol;

fn cmp_result_ty(ty: Type) -> Type {
    if let Type::Vec { lane, count } = ty {
        Type::Vec {
            lane: lane.int_mask(),
            count,
        }
    } else {
        Type::I8
    }
}

impl Ins<'_> {
    /// An integer constant of type `ty`.
    pub fn iconst(mut self, ty: Type, imm: i64) -> Value {
        self.constant(ty, imm, true)
    }

    /// A relocated 31-bit call-table slot address at `index`.
    pub fn slot_addr(mut self, index: u32) -> Value {
        self.value(Op::SlotAddr(index), Type::I64)
    }

    /// A relocated 64-bit symbol word at `index`.
    pub fn symbol(mut self, index: u32) -> Value {
        self.value(Op::Symbol(index), Type::I64)
    }

    /// The address of runtime helper `symbol`.
    pub fn runtime(mut self, symbol: RuntimeSymbol) -> Value {
        self.value(Op::Runtime(symbol), Type::I64)
    }

    /// A 128-bit integer from two 64-bit halves.
    pub fn iconcat(mut self, low: Value, high: Value) -> Value {
        self.value(Op::Binary(Binary::Iconcat, low, high), Type::I128)
    }

    /// Wrapping addition.
    pub fn iadd(self, x: Value, y: Value) -> Value {
        self.binary(Binary::Iadd, x, y)
    }

    /// Wrapping addition of a constant.
    pub fn iadd_imm(self, x: Value, imm: i64) -> Value {
        self.binary_imm(Binary::Iadd, x, imm)
    }

    /// Wrapping subtraction.
    pub fn isub(self, x: Value, y: Value) -> Value {
        self.binary(Binary::Isub, x, y)
    }

    /// Wrapping multiplication.
    pub fn imul(self, x: Value, y: Value) -> Value {
        self.binary(Binary::Imul, x, y)
    }

    /// Wrapping multiplication by a constant.
    pub fn imul_imm(self, x: Value, imm: i64) -> Value {
        self.binary_imm(Binary::Imul, x, imm)
    }

    /// Signed division, rounding toward zero.
    pub fn sdiv(self, x: Value, y: Value) -> Value {
        self.binary(Binary::Sdiv, x, y)
    }

    /// Unsigned division.
    pub fn udiv(self, x: Value, y: Value) -> Value {
        self.binary(Binary::Udiv, x, y)
    }

    /// Signed remainder.
    pub fn srem(self, x: Value, y: Value) -> Value {
        self.binary(Binary::Srem, x, y)
    }

    /// Unsigned remainder.
    pub fn urem(self, x: Value, y: Value) -> Value {
        self.binary(Binary::Urem, x, y)
    }

    /// The high half of the unsigned double-width product.
    pub fn umulhi(self, x: Value, y: Value) -> Value {
        self.binary(Binary::Umulhi, x, y)
    }

    /// The high half of the signed double-width product.
    pub fn smulhi(self, x: Value, y: Value) -> Value {
        self.binary(Binary::Smulhi, x, y)
    }

    /// The signed minimum.
    pub fn smin(self, x: Value, y: Value) -> Value {
        self.binary(Binary::Smin, x, y)
    }

    /// The signed maximum.
    pub fn smax(self, x: Value, y: Value) -> Value {
        self.binary(Binary::Smax, x, y)
    }

    /// The unsigned minimum.
    pub fn umin(self, x: Value, y: Value) -> Value {
        self.binary(Binary::Umin, x, y)
    }

    /// The unsigned maximum.
    pub fn umax(self, x: Value, y: Value) -> Value {
        self.binary(Binary::Umax, x, y)
    }

    /// Bitwise and.
    pub fn band(self, x: Value, y: Value) -> Value {
        self.binary(Binary::Band, x, y)
    }

    /// Bitwise or.
    pub fn bor(self, x: Value, y: Value) -> Value {
        self.binary(Binary::Bor, x, y)
    }

    /// Bitwise exclusive or.
    pub fn bxor(self, x: Value, y: Value) -> Value {
        self.binary(Binary::Bxor, x, y)
    }

    /// Bitwise exclusive or with a constant.
    pub fn bxor_imm(self, x: Value, imm: i64) -> Value {
        self.binary_imm(Binary::Bxor, x, imm)
    }

    /// Shift left by `y` modulo the width.
    pub fn ishl(self, x: Value, y: Value) -> Value {
        self.binary(Binary::Ishl, x, y)
    }

    /// Logical shift right by `y` modulo the width.
    pub fn ushr(self, x: Value, y: Value) -> Value {
        self.binary(Binary::Ushr, x, y)
    }

    /// Arithmetic shift right by `y` modulo the width.
    pub fn sshr(self, x: Value, y: Value) -> Value {
        self.binary(Binary::Sshr, x, y)
    }

    /// Arithmetic shift right by a constant.
    pub fn sshr_imm(self, x: Value, imm: i64) -> Value {
        self.binary_imm(Binary::Sshr, x, imm)
    }

    /// Rotate left.
    pub fn rotl(self, x: Value, y: Value) -> Value {
        self.binary(Binary::Rotl, x, y)
    }

    /// Rotate right.
    pub fn rotr(self, x: Value, y: Value) -> Value {
        self.binary(Binary::Rotr, x, y)
    }

    /// Two's complement negation.
    pub fn ineg(self, x: Value) -> Value {
        self.unary(Unary::Ineg, x)
    }

    /// Bitwise not.
    pub fn bnot(self, x: Value) -> Value {
        self.unary(Unary::Bnot, x)
    }

    /// Byte order reversal.
    pub fn bswap(self, x: Value) -> Value {
        self.unary(Unary::Bswap, x)
    }

    /// Bit order reversal.
    pub fn bitrev(self, x: Value) -> Value {
        self.unary(Unary::Bitrev, x)
    }

    /// Leading zero bits.
    pub fn clz(self, x: Value) -> Value {
        self.unary(Unary::Clz, x)
    }

    /// Trailing zero bits.
    pub fn ctz(self, x: Value) -> Value {
        self.unary(Unary::Ctz, x)
    }

    /// Set bits.
    pub fn popcnt(self, x: Value) -> Value {
        self.unary(Unary::Popcnt, x)
    }

    /// Integer comparison producing 1 or 0 as `I8`, or a vector mask for vector operands.
    pub fn icmp(mut self, cc: IntCC, x: Value, y: Value) -> Value {
        let ty = cmp_result_ty(self.func.ty(x));
        self.value(Op::Icmp(cc, x, y), ty)
    }

    /// 1 as an `I8` when `x + y` fits the type of `x`, read as signed when `signed`, else 0.
    pub fn add_fits(mut self, signed: bool, x: Value, y: Value) -> Value {
        self.value(Op::Fits(Binary::Iadd, signed, x, y), Type::I8)
    }

    /// 1 as an `I8` when `x - y` fits the type of `x`, read as signed when `signed`, else 0.
    pub fn sub_fits(mut self, signed: bool, x: Value, y: Value) -> Value {
        self.value(Op::Fits(Binary::Isub, signed, x, y), Type::I8)
    }

    /// 1 as an `I8` when `x * y` fits the type of `x`, read as signed when `signed`, else 0.
    pub fn mul_fits(mut self, signed: bool, x: Value, y: Value) -> Value {
        self.value(Op::Fits(Binary::Imul, signed, x, y), Type::I8)
    }

    /// Integer comparison with a constant.
    pub fn icmp_imm(self, cc: IntCC, x: Value, imm: i64) -> Value {
        self.compare_imm(cc, x, imm, true)
    }

    /// Integer comparison with a constant that sign-extends to the width.
    pub fn icmp_imm_s(self, cc: IntCC, x: Value, imm: i64) -> Value {
        self.compare_imm(cc, x, imm, true)
    }

    /// Integer comparison with a constant that zero-extends to the width.
    pub fn icmp_imm_u(self, cc: IntCC, x: Value, imm: i64) -> Value {
        self.compare_imm(cc, x, imm, false)
    }

    /// Float comparison producing 1 or 0 as `I8`, or a vector mask for vector operands.
    pub fn fcmp(mut self, cc: FloatCC, x: Value, y: Value) -> Value {
        let ty = cmp_result_ty(self.func.ty(x));
        self.value(Op::Fcmp(cc, x, y), ty)
    }

    /// `x` when `condition` is nonzero, else `y`.
    pub fn select(mut self, condition: Value, x: Value, y: Value) -> Value {
        let ty = self.func.ty(x);
        self.value(Op::Select(condition, x, y), ty)
    }

    /// Zero extension to `ty`.
    pub fn uextend(self, ty: Type, x: Value) -> Value {
        self.convert(Convert::Uextend, ty, x)
    }

    /// Sign extension to `ty`.
    pub fn sextend(self, ty: Type, x: Value) -> Value {
        self.convert(Convert::Sextend, ty, x)
    }

    /// Truncation to `ty`.
    pub fn ireduce(self, ty: Type, x: Value) -> Value {
        self.convert(Convert::Ireduce, ty, x)
    }

    /// Float widening.
    pub fn fpromote(self, ty: Type, x: Value) -> Value {
        self.convert(Convert::Fpromote, ty, x)
    }

    /// Float narrowing with rounding.
    pub fn fdemote(self, ty: Type, x: Value) -> Value {
        self.convert(Convert::Fdemote, ty, x)
    }

    /// Signed integer to float.
    pub fn fcvt_from_sint(self, ty: Type, x: Value) -> Value {
        self.convert(Convert::FromSint, ty, x)
    }

    /// Unsigned integer to float.
    pub fn fcvt_from_uint(self, ty: Type, x: Value) -> Value {
        self.convert(Convert::FromUint, ty, x)
    }

    /// Float to signed integer, saturating, with NaN as zero.
    pub fn fcvt_to_sint_sat(self, ty: Type, x: Value) -> Value {
        self.convert(Convert::ToSintSat, ty, x)
    }

    /// Float to unsigned integer, saturating, with NaN as zero.
    pub fn fcvt_to_uint_sat(self, ty: Type, x: Value) -> Value {
        self.convert(Convert::ToUintSat, ty, x)
    }

    /// Reinterprets the bits of `x` as `ty`.
    pub fn bitcast(self, ty: Type, x: Value) -> Value {
        self.convert(Convert::Bitcast, ty, x)
    }

    /// Float addition.
    pub fn fadd(self, x: Value, y: Value) -> Value {
        self.binary(Binary::Fadd, x, y)
    }

    /// Float subtraction.
    pub fn fsub(self, x: Value, y: Value) -> Value {
        self.binary(Binary::Fsub, x, y)
    }

    /// Float multiplication.
    pub fn fmul(self, x: Value, y: Value) -> Value {
        self.binary(Binary::Fmul, x, y)
    }

    /// Float division.
    pub fn fdiv(self, x: Value, y: Value) -> Value {
        self.binary(Binary::Fdiv, x, y)
    }

    /// IEEE-754 `minNum` of `x` and `y`.
    pub fn fmin(self, x: Value, y: Value) -> Value {
        self.binary(Binary::Fmin, x, y)
    }

    /// IEEE-754 `maxNum` of `x` and `y`.
    pub fn fmax(self, x: Value, y: Value) -> Value {
        self.binary(Binary::Fmax, x, y)
    }

    /// `x` with the sign of `y`.
    pub fn fcopysign(self, x: Value, y: Value) -> Value {
        self.binary(Binary::Fcopysign, x, y)
    }

    /// Float negation.
    pub fn fneg(self, x: Value) -> Value {
        self.unary(Unary::Fneg, x)
    }

    /// Float absolute value.
    pub fn fabs(self, x: Value) -> Value {
        self.unary(Unary::Fabs, x)
    }

    /// Float square root.
    pub fn sqrt(self, x: Value) -> Value {
        self.unary(Unary::Sqrt, x)
    }

    /// Round toward negative infinity.
    pub fn floor(self, x: Value) -> Value {
        self.unary(Unary::Floor, x)
    }

    /// Round toward positive infinity.
    pub fn ceil(self, x: Value) -> Value {
        self.unary(Unary::Ceil, x)
    }

    /// Round toward zero.
    pub fn trunc(self, x: Value) -> Value {
        self.unary(Unary::Trunc, x)
    }

    /// Reads a `ty` at `addr + offset`.
    pub fn load(mut self, ty: Type, addr: Value, offset: i32) -> Value {
        self.value(Op::Load(addr, offset), ty)
    }

    /// Copies `bytes` bytes from `from` to `to`, which may overlap.
    pub fn copy(mut self, to: Value, from: Value, bytes: Value) -> Inst {
        self.effect(Op::Copy(to, from, bytes))
    }

    /// Copies a value of `bytes` bytes from `from` to `to`, two ranges that are the same or disjoint.
    pub fn move_value(mut self, to: Value, from: Value, bytes: u32) -> Inst {
        self.effect(Op::Move(to, from, bytes))
    }

    /// Writes `value` at `addr + offset`.
    pub fn store(mut self, value: Value, addr: Value, offset: i32) -> Inst {
        self.effect(Op::Store(value, addr, offset))
    }

    /// The address of `slot` plus `offset`.
    pub fn stack_addr(mut self, ty: Type, slot: StackSlot, offset: i32) -> Value {
        self.value(Op::StackAddr(slot, offset), ty)
    }

    /// Reads a `ty` from `slot` at `offset`.
    pub fn stack_load(mut self, ty: Type, slot: StackSlot, offset: i32) -> Value {
        self.value(Op::StackLoad(slot, offset), ty)
    }

    /// Writes `value` into `slot` at `offset`.
    pub fn stack_store(mut self, value: Value, slot: StackSlot, offset: i32) -> Inst {
        self.effect(Op::StackStore(value, slot, offset))
    }

    /// An atomic read of a `ty` at `addr` with memory ordering `ord`.
    pub fn atomic_load(mut self, ty: Type, addr: Value, ord: AtomicOrd) -> Value {
        self.value(Op::AtomicLoad(addr, ord), ty)
    }

    /// An atomic write of `value` at `addr` with memory ordering `ord`.
    pub fn atomic_store(mut self, value: Value, addr: Value, ord: AtomicOrd) -> Inst {
        self.effect(Op::AtomicStore(value, addr, ord))
    }

    /// A volatile read of a `ty` at `addr` plus `offset`.
    pub fn volatile_load(mut self, ty: Type, addr: Value, offset: i32) -> Value {
        self.value(Op::VolatileLoad(addr, offset), ty)
    }

    /// A volatile write of `value` at `addr` plus `offset`.
    pub fn volatile_store(mut self, value: Value, addr: Value, offset: i32) -> Inst {
        self.effect(Op::VolatileStore(value, addr, offset))
    }

    /// Applies `op` with `value` to the `ty` at `addr` with memory ordering `ord` and returns the old contents.
    pub fn atomic_rmw(
        mut self,
        ty: Type,
        op: AtomicRmwOp,
        ord: AtomicOrd,
        addr: Value,
        value: Value,
    ) -> Value {
        self.value(Op::AtomicRmw(op, ord, addr, value), ty)
    }

    /// Writes `replacement` at `addr` if it holds `expected` with memory ordering `ord`, and returns the old contents.
    pub fn atomic_cas(
        mut self,
        ord: AtomicOrd,
        addr: Value,
        expected: Value,
        replacement: Value,
    ) -> Value {
        let ty = self.func.ty(expected);
        self.value(Op::AtomicCas(ord, addr, expected, replacement), ty)
    }

    /// A memory barrier with ordering `ord`.
    pub fn fence(mut self, ord: AtomicOrd) -> Inst {
        self.effect(Op::Fence(ord))
    }

    /// This function's frame pointer.
    pub fn get_frame_pointer(mut self, ty: Type) -> Value {
        self.value(Op::FramePointer, ty)
    }

    /// Calls the code at `callee` with `args`.
    pub fn call_indirect(mut self, sig: SigRef, callee: Value, args: &[Value]) -> Inst {
        let returns: Vec<Type> = self.func.sigs[sig.index()]
            .returns
            .iter()
            .map(|param| param.value_type)
            .collect();
        let results = returns.into_iter().map(|ty| self.func.value(ty)).collect();
        self.effect(Op::Call {
            sig,
            callee,
            args: args.to_vec(),
            results,
        })
    }

    /// Calls `callee`, then goes to `normal`, or to `handler` with the exception when the callee unwinds.
    pub fn try_call_indirect(
        mut self,
        sig: SigRef,
        callee: Value,
        args: &[Value],
        normal: Block,
        handler: Block,
    ) -> Inst {
        self.effect(Op::TryCall {
            sig,
            callee,
            args: args.to_vec(),
            normal,
            handler,
        })
    }

    /// Goes to `block` with `args`.
    pub fn jump(mut self, block: Block, args: &[Value]) -> Inst {
        self.effect(Op::Jump(block, args.to_vec()))
    }

    /// Goes to `then` when `condition` is nonzero, else to `otherwise`.
    pub fn brif(
        mut self,
        condition: Value,
        then: Block,
        then_args: &[Value],
        otherwise: Block,
        else_args: &[Value],
    ) -> Inst {
        self.effect(Op::Brif(
            condition,
            then,
            then_args.to_vec(),
            otherwise,
            else_args.to_vec(),
        ))
    }

    /// Branches to `targets[index]` when `index` is in range, else to `default`.
    pub fn br_table(mut self, index: Value, default: Block, targets: &[Block]) -> Inst {
        self.effect(Op::BrTable(index, default, targets.to_vec()))
    }

    /// Returns `values`.
    pub fn return_(mut self, values: &[Value]) -> Inst {
        self.effect(Op::Return(values.to_vec()))
    }

    /// Stops with an illegal instruction.
    pub fn trap(mut self) -> Inst {
        self.effect(Op::Trap)
    }

    /// Stops with an illegal instruction when `condition` is zero.
    pub fn trapz(mut self, condition: Value) -> Inst {
        self.effect(Op::Trapz(condition))
    }

    /// Tells the optimizer that `condition` is nonzero here; emits no code.
    pub fn assume(mut self, condition: Value) -> Inst {
        self.effect(Op::Assume(condition))
    }
}

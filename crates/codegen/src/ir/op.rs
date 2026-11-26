use super::{
    AtomicOrd, AtomicRmwOp, Block, FloatCC, Inst, IntCC, ReduceOp, ShuffleMask, SigRef, StackSlot,
    TargetOpId, TinyArgs, Value,
};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub(crate) enum Unary {
    Ineg,
    Bnot,
    Bswap,
    Bitrev,
    Clz,
    Ctz,
    Popcnt,
    Fneg,
    Fabs,
    Sqrt,
    Floor,
    Ceil,
    Trunc,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub(crate) enum Binary {
    Iadd,
    Isub,
    Imul,
    Sdiv,
    Udiv,
    Srem,
    Urem,
    Umulhi,
    Smulhi,
    Smin,
    Smax,
    Umin,
    Umax,
    Band,
    Bor,
    Bxor,
    Ishl,
    Ushr,
    Sshr,
    Rotl,
    Rotr,
    Iconcat,
    Fadd,
    Fsub,
    Fmul,
    Fdiv,
    Fmin,
    Fmax,
    Fcopysign,
    SaddSat,
    UaddSat,
    SsubSat,
    UsubSat,
    AvgRound,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Convert {
    Uextend,
    Sextend,
    Ireduce,
    Fpromote,
    Fdemote,
    FromSint,
    FromUint,
    ToSintSat,
    ToUintSat,
    Bitcast,
    NarrowSat { signed: bool },
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) enum Op {
    Const(i64),
    SlotAddr(u32),
    Symbol(u32),
    Runtime(crate::reloc::RuntimeSymbol),
    Unary(Unary, Value),
    Binary(Binary, Value, Value),
    Fma(Value, Value, Value),
    Convert(Convert, Value),
    Icmp(IntCC, Value, Value),
    Fits(Binary, bool, Value, Value),
    Fcmp(FloatCC, Value, Value),
    Shuffle(Value, Value, ShuffleMask),
    Extract(Value, u8),
    Insert(Value, Value, u8),
    Reduce(ReduceOp, Value),
    TargetOp(TargetOpId, TinyArgs),
    Select(Value, Value, Value),
    Load(Value, i32),
    Store(Value, Value, i32),
    VolatileLoad(Value, i32),
    VolatileStore(Value, Value, i32),
    StackAddr(StackSlot, i32),
    StackLoad(StackSlot, i32),
    StackStore(Value, StackSlot, i32),
    AtomicLoad(Value, AtomicOrd),
    AtomicStore(Value, Value, AtomicOrd),
    AtomicRmw(AtomicRmwOp, AtomicOrd, Value, Value),
    AtomicCas(AtomicOrd, Value, Value, Value),
    Fence(AtomicOrd),
    FramePointer,
    Copy(Value, Value, Value),
    Move(Value, Value, u32),
    Call {
        sig: SigRef,
        callee: Value,
        args: Vec<Value>,
        results: Vec<Value>,
    },
    TryCall {
        sig: SigRef,
        callee: Value,
        args: Vec<Value>,
        normal: Block,
        handler: Block,
    },
    Jump(Block, Vec<Value>),
    Brif(Value, Block, Vec<Value>, Block, Vec<Value>),
    BrTable(Value, Block, Vec<Block>),
    Return(Vec<Value>),
    Trap,
    Trapz(Value),
    Assume(Value),
}

impl Op {
    pub(crate) fn is_terminator(&self) -> bool {
        matches!(
            self,
            Op::TryCall { .. }
                | Op::Jump(..)
                | Op::Brif(..)
                | Op::BrTable(..)
                | Op::Return(_)
                | Op::Trap
        )
    }

    pub(crate) fn is_speculable(&self) -> bool {
        match self {
            Op::Binary(bin, ..) => !matches!(
                bin,
                Binary::Sdiv | Binary::Udiv | Binary::Srem | Binary::Urem
            ),
            Op::Const(_)
            | Op::SlotAddr(_)
            | Op::Symbol(_)
            | Op::StackAddr(..)
            | Op::Unary(..)
            | Op::Convert(..)
            | Op::Icmp(..)
            | Op::Fits(..)
            | Op::Fcmp(..)
            | Op::Shuffle(..)
            | Op::Extract(..)
            | Op::Insert(..)
            | Op::Reduce(..)
            | Op::TargetOp(..)
            | Op::Select(..)
            | Op::Fma(..) => true,
            _ => false,
        }
    }

    pub(crate) fn is_removable(&self) -> bool {
        self.is_speculable()
            || matches!(
                self,
                Op::Binary(..)
                    | Op::Runtime(_)
                    | Op::Load(..)
                    | Op::StackLoad(..)
                    | Op::FramePointer
            )
    }

    #[inline]
    pub(crate) fn for_operands(&self, mut f: impl FnMut(Value)) {
        match self {
            Op::Const(_)
            | Op::SlotAddr(_)
            | Op::Symbol(_)
            | Op::Runtime(_)
            | Op::StackAddr(..)
            | Op::StackLoad(..)
            | Op::Fence(_)
            | Op::FramePointer
            | Op::Trap => {}
            Op::Unary(_, x)
            | Op::Convert(_, x)
            | Op::Extract(x, _)
            | Op::Reduce(_, x)
            | Op::Load(x, _)
            | Op::VolatileLoad(x, _)
            | Op::StackStore(x, ..)
            | Op::AtomicLoad(x, _)
            | Op::Trapz(x)
            | Op::Assume(x) => f(*x),
            Op::Binary(_, x, y)
            | Op::Icmp(_, x, y)
            | Op::Fits(_, _, x, y)
            | Op::Fcmp(_, x, y)
            | Op::Insert(x, y, _)
            | Op::Store(x, y, _)
            | Op::VolatileStore(x, y, _)
            | Op::AtomicStore(x, y, _)
            | Op::AtomicRmw(_, _, x, y)
            | Op::Move(x, y, _) => {
                f(*x);
                f(*y);
            }
            Op::Shuffle(a, b, mask) => {
                f(*a);
                f(*b);
                if let ShuffleMask::Dyn(idx) = mask {
                    f(*idx);
                }
            }
            Op::TargetOp(_, args) => args.as_slice().iter().for_each(|arg| f(*arg)),
            Op::Select(c, x, y)
            | Op::AtomicCas(_, c, x, y)
            | Op::Copy(c, x, y)
            | Op::Fma(c, x, y) => {
                f(*c);
                f(*x);
                f(*y);
            }
            Op::Call { callee, args, .. } | Op::TryCall { callee, args, .. } => {
                f(*callee);
                args.iter().for_each(|arg| f(*arg));
            }
            Op::Jump(_, args) | Op::Return(args) => args.iter().for_each(|arg| f(*arg)),
            Op::Brif(c, _, then, _, otherwise) => {
                f(*c);
                then.iter().chain(otherwise).for_each(|arg| f(*arg));
            }
            Op::BrTable(index, ..) => f(*index),
        }
    }

    pub(crate) fn map_operands(&mut self, mut f: impl FnMut(Value) -> Value) {
        match self {
            Op::Const(_)
            | Op::SlotAddr(_)
            | Op::Symbol(_)
            | Op::Runtime(_)
            | Op::StackAddr(..)
            | Op::StackLoad(..)
            | Op::Fence(_)
            | Op::FramePointer
            | Op::Trap => {}
            Op::Unary(_, x)
            | Op::Convert(_, x)
            | Op::Extract(x, _)
            | Op::Reduce(_, x)
            | Op::Load(x, _)
            | Op::VolatileLoad(x, _)
            | Op::StackStore(x, ..)
            | Op::AtomicLoad(x, _)
            | Op::Trapz(x)
            | Op::Assume(x) => {
                *x = f(*x);
            }
            Op::Binary(_, x, y)
            | Op::Icmp(_, x, y)
            | Op::Fits(_, _, x, y)
            | Op::Fcmp(_, x, y)
            | Op::Insert(x, y, _)
            | Op::Store(x, y, _)
            | Op::VolatileStore(x, y, _)
            | Op::AtomicStore(x, y, _)
            | Op::AtomicRmw(_, _, x, y)
            | Op::Move(x, y, _) => {
                *x = f(*x);
                *y = f(*y);
            }
            Op::Shuffle(a, b, mask) => {
                *a = f(*a);
                *b = f(*b);
                if let ShuffleMask::Dyn(idx) = mask {
                    *idx = f(*idx);
                }
            }
            Op::TargetOp(_, args) => args
                .as_mut_slice()
                .iter_mut()
                .for_each(|arg| *arg = f(*arg)),
            Op::Select(c, x, y)
            | Op::AtomicCas(_, c, x, y)
            | Op::Copy(c, x, y)
            | Op::Fma(c, x, y) => {
                *c = f(*c);
                *x = f(*x);
                *y = f(*y);
            }
            Op::Call { callee, args, .. } | Op::TryCall { callee, args, .. } => {
                *callee = f(*callee);
                args.iter_mut().for_each(|arg| *arg = f(*arg));
            }
            Op::Jump(_, args) | Op::Return(args) => {
                args.iter_mut().for_each(|arg| *arg = f(*arg));
            }
            Op::Brif(c, _, then, _, otherwise) => {
                *c = f(*c);
                then.iter_mut()
                    .chain(otherwise.iter_mut())
                    .for_each(|arg| *arg = f(*arg));
            }
            Op::BrTable(index, ..) => *index = f(*index),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct InstData {
    pub op: Op,
    pub result: Option<Value>,
    pub loc: u64,
}

#[derive(Clone, Default, Debug)]
pub(crate) struct BlockData {
    pub params: Vec<Value>,
    pub insts: Vec<Inst>,
}

#[derive(Clone, Debug)]
pub(crate) struct SlotData {
    pub size: u32,
    pub align_shift: u8,
    pub poll_local: bool,
}

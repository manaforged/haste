use std::fmt;

/// The machine type of one value.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Type {
    I8,
    I16,
    I32,
    I64,
    I128,
    F32,
    F64,
}

impl Type {
    /// Width in bytes.
    pub fn bytes(self) -> u32 {
        match self {
            Type::I8 => 1,
            Type::I16 => 2,
            Type::I32 | Type::F32 => 4,
            Type::I64 | Type::F64 => 8,
            Type::I128 => 16,
        }
    }

    /// Width in bits.
    pub fn bits(self) -> u32 {
        self.bytes() * 8
    }

    /// Whether the type holds a floating-point number.
    pub fn is_float(self) -> bool {
        matches!(self, Type::F32 | Type::F64)
    }

    /// Whether the type holds an integer that fits in one general-purpose register.
    pub fn is_small_int(self) -> bool {
        matches!(self, Type::I8 | Type::I16 | Type::I32 | Type::I64)
    }
}

macro_rules! ids {
    ($($(#[$doc:meta])* $name:ident),*) => {$(
        $(#[$doc])*
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
        pub struct $name(pub(crate) u32);

        impl $name {
            pub(crate) fn index(self) -> usize {
                self.0 as usize
            }
        }
    )*};
}

ids!(
    /// An SSA value.
    Value,
    /// A basic block.
    Block,
    /// A fixed area in the frame.
    StackSlot,
    /// A call signature imported into a function.
    SigRef,
    /// One instruction.
    Inst
);

/// How a function takes arguments and which registers it keeps.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum CallConv {
    /// Generated code calling generated code: the C argument registers, and the callee saves the pool registers it uses, or all of them when it has a handler.
    Internal,
    /// The platform C convention.
    Native,
}

/// What a parameter carries besides its value.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ArgumentPurpose {
    Normal,
    /// The pointer to caller memory for a return value.
    StructReturn,
    /// A pointer to a struct of this many bytes that travels by value on the stack.
    StructArgument(u32),
}

/// One parameter or return value of a signature.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct AbiParam {
    pub value_type: Type,
    pub purpose: ArgumentPurpose,
}

impl AbiParam {
    /// A plain value of `ty`.
    pub fn new(ty: Type) -> Self {
        Self { value_type: ty, purpose: ArgumentPurpose::Normal }
    }

    /// A value of `ty` with a special purpose.
    pub fn special(ty: Type, purpose: ArgumentPurpose) -> Self {
        Self { value_type: ty, purpose }
    }
}

/// The parameters, return values and convention of a function.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Signature {
    pub params: Vec<AbiParam>,
    pub returns: Vec<AbiParam>,
    pub call_conv: CallConv,
}

impl Signature {
    /// An empty signature in `call_conv`.
    pub fn new(call_conv: CallConv) -> Self {
        Self { params: Vec::new(), returns: Vec::new(), call_conv }
    }
}

/// An integer comparison.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum IntCC {
    Equal,
    NotEqual,
    SignedLessThan,
    SignedGreaterThanOrEqual,
    SignedGreaterThan,
    SignedLessThanOrEqual,
    UnsignedLessThan,
    UnsignedGreaterThanOrEqual,
    UnsignedGreaterThan,
    UnsignedLessThanOrEqual,
}

impl IntCC {
    /// Whether the comparison reads its operands as signed.
    pub fn is_signed(self) -> bool {
        matches!(
            self,
            IntCC::SignedLessThan | IntCC::SignedGreaterThanOrEqual | IntCC::SignedGreaterThan | IntCC::SignedLessThanOrEqual
        )
    }

    /// The comparison that holds exactly when this one does not.
    pub fn inverse(self) -> Self {
        match self {
            IntCC::Equal => IntCC::NotEqual,
            IntCC::NotEqual => IntCC::Equal,
            IntCC::SignedLessThan => IntCC::SignedGreaterThanOrEqual,
            IntCC::SignedGreaterThanOrEqual => IntCC::SignedLessThan,
            IntCC::SignedGreaterThan => IntCC::SignedLessThanOrEqual,
            IntCC::SignedLessThanOrEqual => IntCC::SignedGreaterThan,
            IntCC::UnsignedLessThan => IntCC::UnsignedGreaterThanOrEqual,
            IntCC::UnsignedGreaterThanOrEqual => IntCC::UnsignedLessThan,
            IntCC::UnsignedGreaterThan => IntCC::UnsignedLessThanOrEqual,
            IntCC::UnsignedLessThanOrEqual => IntCC::UnsignedGreaterThan,
        }
    }
}

/// A floating-point comparison; every one but `NotEqual` is false when an operand is NaN.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum FloatCC {
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
}

/// The operation of an atomic read-modify-write.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
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
    Fcopysign,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
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
}

#[derive(Clone, Debug)]
pub(crate) enum Op {
    Const(i64),
    Unary(Unary, Value),
    Binary(Binary, Value, Value),
    Convert(Convert, Value),
    Icmp(IntCC, Value, Value),
    /// 1 when `x op y` (add, sub or mul, signed when the flag is set) fits the type of `x`, else 0.
    Fits(Binary, bool, Value, Value),
    Fcmp(FloatCC, Value, Value),
    Select(Value, Value, Value),
    Load(Value, i32),
    Store(Value, Value, i32),
    StackAddr(StackSlot, i32),
    StackLoad(StackSlot, i32),
    StackStore(Value, StackSlot, i32),
    AtomicLoad(Value),
    AtomicStore(Value, Value),
    AtomicRmw(AtomicRmwOp, Value, Value),
    AtomicCas(Value, Value, Value),
    Fence,
    FramePointer,
    /// Copies the number of bytes in the third value from the address in the second to the address in the first; the two may overlap.
    Copy(Value, Value, Value),
    Call { sig: SigRef, callee: Value, args: Vec<Value>, results: Vec<Value> },
    TryCall { sig: SigRef, callee: Value, args: Vec<Value>, normal: Block, handler: Block },
    Jump(Block, Vec<Value>),
    Brif(Value, Block, Vec<Value>, Block, Vec<Value>),
    Return(Vec<Value>),
    Trap,
    Trapz(Value),
}

impl Op {
    pub(crate) fn is_terminator(&self) -> bool {
        matches!(self, Op::TryCall { .. } | Op::Jump(..) | Op::Brif(..) | Op::Return(_) | Op::Trap)
    }

    /// Every block this instruction can go to.
    pub(crate) fn targets(&self) -> impl Iterator<Item = Block> + use<> {
        let (first, second) = match *self {
            Op::Jump(block, _) => (Some(block), None),
            Op::Brif(_, then, _, otherwise, _) => (Some(then), Some(otherwise)),
            Op::TryCall { normal, handler, .. } => (Some(normal), Some(handler)),
            _ => (None, None),
        };
        first.into_iter().chain(second)
    }

    /// Replaces every block this instruction can go to with `f` of it.
    pub(crate) fn map_targets(&mut self, mut f: impl FnMut(Block) -> Block) {
        match self {
            Op::Jump(block, _) => *block = f(*block),
            Op::Brif(_, then, _, otherwise, _) => {
                *then = f(*then);
                *otherwise = f(*otherwise);
            }
            Op::TryCall { normal, handler, .. } => {
                *normal = f(*normal);
                *handler = f(*handler);
            }
            _ => {}
        }
    }

    /// Calls `f` with every value this instruction reads, once per read.
    pub(crate) fn for_operands(&self, mut f: impl FnMut(Value)) {
        match self {
            Op::Const(_) | Op::StackAddr(..) | Op::StackLoad(..) | Op::Fence | Op::FramePointer | Op::Trap => {}
            Op::Unary(_, x) | Op::Convert(_, x) | Op::Load(x, _) | Op::StackStore(x, ..) | Op::AtomicLoad(x) | Op::Trapz(x) => f(*x),
            Op::Binary(_, x, y) | Op::Icmp(_, x, y) | Op::Fits(_, _, x, y) | Op::Fcmp(_, x, y) | Op::Store(x, y, _) | Op::AtomicStore(x, y) | Op::AtomicRmw(_, x, y) => {
                f(*x);
                f(*y);
            }
            Op::Select(c, x, y) | Op::AtomicCas(c, x, y) | Op::Copy(c, x, y) => {
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
        }
    }

    /// Replaces every value this instruction reads with `f` of it.
    pub(crate) fn map_operands(&mut self, mut f: impl FnMut(Value) -> Value) {
        match self {
            Op::Const(_) | Op::StackAddr(..) | Op::StackLoad(..) | Op::Fence | Op::FramePointer | Op::Trap => {}
            Op::Unary(_, x) | Op::Convert(_, x) | Op::Load(x, _) | Op::StackStore(x, ..) | Op::AtomicLoad(x) | Op::Trapz(x) => {
                *x = f(*x);
            }
            Op::Binary(_, x, y)
            | Op::Icmp(_, x, y)
            | Op::Fits(_, _, x, y)
            | Op::Fcmp(_, x, y)
            | Op::Store(x, y, _)
            | Op::AtomicStore(x, y)
            | Op::AtomicRmw(_, x, y) => {
                *x = f(*x);
                *y = f(*y);
            }
            Op::Select(c, x, y) | Op::AtomicCas(c, x, y) | Op::Copy(c, x, y) => {
                *c = f(*c);
                *x = f(*x);
                *y = f(*y);
            }
            Op::Call { callee, args, .. } | Op::TryCall { callee, args, .. } => {
                *callee = f(*callee);
                args.iter_mut().for_each(|arg| *arg = f(*arg));
            }
            Op::Jump(_, args) | Op::Return(args) => args.iter_mut().for_each(|arg| *arg = f(*arg)),
            Op::Brif(c, _, then, _, otherwise) => {
                *c = f(*c);
                then.iter_mut().chain(otherwise.iter_mut()).for_each(|arg| *arg = f(*arg));
            }
        }
    }
}

#[derive(Clone)]
pub(crate) struct InstData {
    pub op: Op,
    pub result: Option<Value>,
}

#[derive(Clone, Default)]
pub(crate) struct BlockData {
    pub params: Vec<Value>,
    pub insts: Vec<Inst>,
}

#[derive(Clone)]
pub(crate) struct SlotData {
    pub size: u32,
    pub align_shift: u8,
}

/// One function body: blocks of instructions over typed values.
#[derive(Clone)]
pub struct Function {
    pub signature: Signature,
    pub(crate) types: Vec<Type>,
    pub(crate) blocks: Vec<BlockData>,
    pub(crate) insts: Vec<InstData>,
    pub(crate) slots: Vec<SlotData>,
    pub(crate) sigs: Vec<Signature>,
    pub(crate) entry: Option<Block>,
    /// The split that turns this function into a coroutine's resume function, when it is one.
    pub coroutine: Option<crate::coroutine::Coroutine>,
}

impl Function {
    /// An empty function with `signature`.
    pub fn new(signature: Signature) -> Self {
        Self { signature, types: Vec::new(), blocks: Vec::new(), insts: Vec::new(), slots: Vec::new(), sigs: Vec::new(), entry: None, coroutine: None }
    }

    /// The type of `value`.
    pub fn ty(&self, value: Value) -> Type {
        self.types[value.index()]
    }

    /// The number of instructions in the body.
    pub fn size(&self) -> usize {
        self.blocks.iter().map(|block| block.insts.len()).sum()
    }

    /// Whether `FunctionBuilder::inline` can copy this function into a caller: it is no coroutine, catches no unwind and reads no frame pointer.
    pub fn can_inline(&self) -> bool {
        self.coroutine.is_none() && self.entry.is_some() && !self.insts.iter().any(|data| matches!(data.op, Op::TryCall { .. } | Op::FramePointer))
    }
}

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "function {:?}", self.signature)?;
        for (index, block) in self.blocks.iter().enumerate() {
            let params: Vec<String> = block.params.iter().map(|v| format!("v{}: {:?}", v.0, self.ty(*v))).collect();
            writeln!(f, "block{index}({}):", params.join(", "))?;
            for inst in &block.insts {
                let data = &self.insts[inst.index()];
                match data.result {
                    Some(value) => writeln!(f, "    v{}: {:?} = {:?}", value.0, self.ty(value), data.op)?,
                    None => writeln!(f, "    {:?}", data.op)?,
                }
            }
        }
        Ok(())
    }
}

use std::fmt;

/// The machine type of one value.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum Type {
    I8,
    I16,
    I32,
    I64,
    I128,
    F32,
    F64,
    /// Sixteen bytes held in one vector register.
    V128,
}

impl Type {
    /// Width in bytes.
    pub fn bytes(self) -> u32 {
        match self {
            Type::I8 => 1,
            Type::I16 => 2,
            Type::I32 | Type::F32 => 4,
            Type::I64 | Type::F64 => 8,
            Type::I128 | Type::V128 => 16,
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

    /// Whether a value of the type lives in a float register when it has one.
    pub fn in_float_register(self) -> bool {
        self.is_float() || self == Type::V128
    }

    /// Whether the type holds an integer that fits in one general-purpose register.
    pub fn is_small_int(self) -> bool {
        matches!(self, Type::I8 | Type::I16 | Type::I32 | Type::I64)
    }

    pub(crate) const fn sign_bit(self) -> i64 {
        if matches!(self, Type::F64) { i64::MIN } else { 0x8000_0000 }
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
#[repr(u8)]
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
    /// The pointer to the static symbol dictionary of a shared generic body.
    Dict,
    /// An unnamed argument after `...` in a C variadic call.
    Variadic,
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

    /// A C variadic argument of `ty`.
    pub fn variadic(ty: Type) -> Self {
        Self { value_type: ty, purpose: ArgumentPurpose::Variadic }
    }
}

/// The parameters, return values and convention of a function.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Signature {
    pub params: Vec<AbiParam>,
    pub returns: Vec<AbiParam>,
    pub call_conv: CallConv,
    pub readonly_params: u64,
    pub readonly_bytes: [u32; 16],
}

impl Signature {
    /// An empty signature in `call_conv`.
    pub fn new(call_conv: CallConv) -> Self {
        Self { params: Vec::new(), returns: Vec::new(), call_conv, readonly_params: 0, readonly_bytes: [0; 16] }
    }

    /// Marks parameter `index` as pointing to immutable memory for the duration of the call.
    pub fn mark_readonly(&mut self, index: usize) {
        if index < 64 {
            self.readonly_params |= 1_u64 << index;
        }
    }

    /// Marks parameter `index` as pointing to `bytes` bytes of dereferenceable, immutable memory for the duration of the call.
    pub fn mark_readonly_bytes(&mut self, index: usize, bytes: u32) {
        self.mark_readonly(index);
        if index < 16 {
            self.readonly_bytes[index] = bytes;
        }
    }
}

/// An integer comparison.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
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

/// A comparison of the sixteen bytes of two vectors, giving a byte of all ones where it holds and of zeros where it does not.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum ByteCC {
    Equal,
    SignedGreaterThan,
}

/// An x64 instruction on two 128-bit vectors that computes in place into the first.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum VectorOp {
    ByteShuffle,
    AddBytes,
    AddWords,
    AddDwords,
    AddQwords,
    SubBytes,
    SubWords,
    SubDwords,
    SubQwords,
    Sha1NextE,
    Sha1Msg1,
    Sha1Msg2,
    Sha256Msg1,
    Sha256Msg2,
    AesEnc,
    AesEncLast,
    AesDec,
    AesDecLast,
}

/// A floating-point comparison; every one but `NotEqual` is false when an operand is NaN.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
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

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
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
    SlotAddr(u32),
    Symbol(u32),
    Runtime(crate::reloc::RuntimeSymbol),
    Unary(Unary, Value),
    Binary(Binary, Value, Value),
    Convert(Convert, Value),
    Icmp(IntCC, Value, Value),
    /// 1 when `x op y` (add, sub or mul, signed when the flag is set) fits the type of `x`, else 0.
    Fits(Binary, bool, Value, Value),
    Fcmp(FloatCC, Value, Value),
    /// The sixteen byte comparisons of two vectors.
    ByteCmp(ByteCC, Value, Value),
    /// The vector whose sixteen bytes are all the byte in the value.
    Splat(Value),
    /// The 32-bit integer whose bit `i` is the top bit of byte `i` of the vector, for `i` below 16, and whose other bits are zero.
    Movemask(Value),
    /// The x64 vector instruction on two vectors.
    Vector(VectorOp, Value, Value),
    /// Two SHA-256 rounds (`sha256rnds2`) on the state vectors in the first two values with the message and constants in the third.
    Sha256Rounds(Value, Value, Value),
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
    /// Copies a value of this many bytes from the address in the second value to the address in the first; the two ranges are the same or disjoint.
    Move(Value, Value, u32),
    Call { sig: SigRef, callee: Value, args: Vec<Value>, results: Vec<Value> },
    TryCall { sig: SigRef, callee: Value, args: Vec<Value>, normal: Block, handler: Block },
    Jump(Block, Vec<Value>),
    Brif(Value, Block, Vec<Value>, Block, Vec<Value>),
    BrTable(Value, Block, Vec<Block>),
    Return(Vec<Value>),
    Trap,
    Trapz(Value),
}

impl Op {
    pub(crate) fn is_terminator(&self) -> bool {
        matches!(self, Op::TryCall { .. } | Op::Jump(..) | Op::Brif(..) | Op::BrTable(..) | Op::Return(_) | Op::Trap)
    }

    /// Every block this instruction can go to.
    pub(crate) fn targets(&self) -> impl Iterator<Item = Block> + '_ {
        let (first, second, slice) = match self {
            Op::Jump(block, _) => (Some(*block), None, &[][..]),
            Op::Brif(_, then, _, otherwise, _) => (Some(*then), Some(*otherwise), &[][..]),
            Op::BrTable(_, default, targets) => (Some(*default), None, &targets[..]),
            Op::TryCall { normal, handler, .. } => (Some(*normal), Some(*handler), &[][..]),
            _ => (None, None, &[][..]),
        };
        first.into_iter().chain(second).chain(slice.iter().copied())
    }

    /// Replaces every block this instruction can go to with `f` of it.
    pub(crate) fn map_targets(&mut self, mut f: impl FnMut(Block) -> Block) {
        match self {
            Op::Jump(block, _) => *block = f(*block),
            Op::Brif(_, then, _, otherwise, _) => {
                *then = f(*then);
                *otherwise = f(*otherwise);
            }
            Op::BrTable(_, default, targets) => {
                *default = f(*default);
                targets.iter_mut().for_each(|target| *target = f(*target));
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
            Op::Const(_) | Op::SlotAddr(_) | Op::Symbol(_) | Op::Runtime(_) | Op::StackAddr(..) | Op::StackLoad(..) | Op::Fence | Op::FramePointer | Op::Trap => {}
            Op::Unary(_, x) | Op::Convert(_, x) | Op::Splat(x) | Op::Movemask(x) | Op::Load(x, _) | Op::StackStore(x, ..) | Op::AtomicLoad(x) | Op::Trapz(x) => f(*x),
            Op::Binary(_, x, y) | Op::Icmp(_, x, y) | Op::Fits(_, _, x, y) | Op::Fcmp(_, x, y) | Op::ByteCmp(_, x, y) | Op::Vector(_, x, y) | Op::Store(x, y, _) | Op::AtomicStore(x, y) | Op::AtomicRmw(_, x, y) | Op::Move(x, y, _) => {
                f(*x);
                f(*y);
            }
            Op::Select(c, x, y) | Op::Sha256Rounds(c, x, y) | Op::AtomicCas(c, x, y) | Op::Copy(c, x, y) => {
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

    /// Replaces every value this instruction reads with `f` of it.
    pub(crate) fn map_operands(&mut self, mut f: impl FnMut(Value) -> Value) {
        match self {
            Op::Const(_) | Op::SlotAddr(_) | Op::Symbol(_) | Op::Runtime(_) | Op::StackAddr(..) | Op::StackLoad(..) | Op::Fence | Op::FramePointer | Op::Trap => {}
            Op::Unary(_, x) | Op::Convert(_, x) | Op::Splat(x) | Op::Movemask(x) | Op::Load(x, _) | Op::StackStore(x, ..) | Op::AtomicLoad(x) | Op::Trapz(x) => {
                *x = f(*x);
            }
            Op::Binary(_, x, y)
            | Op::Icmp(_, x, y)
            | Op::Fits(_, _, x, y)
            | Op::Fcmp(_, x, y)
            | Op::ByteCmp(_, x, y)
            | Op::Vector(_, x, y)
            | Op::Store(x, y, _)
            | Op::AtomicStore(x, y)
            | Op::AtomicRmw(_, x, y)
            | Op::Move(x, y, _) => {
                *x = f(*x);
                *y = f(*y);
            }
            Op::Select(c, x, y) | Op::Sha256Rounds(c, x, y) | Op::AtomicCas(c, x, y) | Op::Copy(c, x, y) => {
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
            Op::BrTable(index, ..) => *index = f(*index),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct InstData {
    pub op: Op,
    pub result: Option<Value>,
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
}

/// One function body: blocks of instructions over typed values.
#[derive(Clone, Debug)]
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
    pub(crate) self_slots: Vec<u32>,
    pub(crate) vtable_slots: Vec<(u32, i32, u32, bool)>,
    pub(crate) rodata_bytes: Vec<(u32, Vec<u8>)>,
}

impl Function {
    /// An empty function with `signature`.
    pub fn new(signature: Signature) -> Self {
        Self { signature, types: Vec::new(), blocks: Vec::new(), insts: Vec::new(), slots: Vec::new(), sigs: Vec::new(), entry: None, coroutine: None, self_slots: Vec::new(), vtable_slots: Vec::new(), rodata_bytes: Vec::new() }
    }

    /// Associates symbol index `sym` with known read-only constant bytes for load folding.
    pub fn add_rodata_bytes(&mut self, sym: u32, bytes: Vec<u8>) {
        self.rodata_bytes.push((sym, bytes));
    }

    /// Marks relocation slot `sym` as a direct self-recursive target of this function.
    pub fn add_self_slot(&mut self, sym: u32) {
        if !self.self_slots.contains(&sym) {
            self.self_slots.push(sym);
        }
    }

    /// Records that an 8-byte load at `offset` from vtable symbol `vt_sym` resolves to call-table slot `slot_sym`.
    pub fn add_vtable_slot(&mut self, vt_sym: u32, offset: i32, slot_sym: u32, low: bool) {
        self.vtable_slots.push((vt_sym, offset, slot_sym, low));
    }

    /// Whether this function has any static vtable-slot entries recorded.
    pub fn has_vtable_slots(&self) -> bool {
        !self.vtable_slots.is_empty()
    }

    /// Whether this function is an unsplit coroutine body.
    pub fn is_coroutine(&self) -> bool {
        self.coroutine.is_some()
    }

    /// Returns `(block, inst_pos, callee_sym)` for every `Op::Call` whose callee is a symbol or slot-addr recorded in `vtable_slots`.
    pub fn devirt_call_sites(&self) -> Vec<(Block, usize, u32)> {
        if self.vtable_slots.is_empty() {
            return Vec::new();
        }
        let mut slot_syms = vec![None; self.types.len()];
        for data in &self.insts {
            if let (Op::Symbol(sym) | Op::SlotAddr(sym), Some(res)) = (&data.op, data.result) {
                slot_syms[res.index()] = Some(*sym);
            }
        }
        let mut syms = slot_syms.clone();
        for data in &self.insts {
            if let (Op::Load(addr, 0), Some(res)) = (&data.op, data.result)
                && let Some(sym) = slot_syms[addr.index()]
            {
                syms[res.index()] = Some(sym);
            }
        }
        let mut out = Vec::new();
        for (b_idx, block) in self.blocks.iter().enumerate() {
            for (pos, &inst) in block.insts.iter().enumerate() {
                if let Op::Call { callee, .. } = self.insts[inst.index()].op
                    && let Some(sym) = syms[callee.index()]
                    && self.vtable_slots.iter().any(|&(_, _, s, _)| s == sym)
                {
                    out.push((Block(b_idx as u32), pos, sym));
                }
            }
        }
        out
    }

    /// The type of `value`.
    pub fn ty(&self, value: Value) -> Type {
        self.types[value.index()]
    }

    /// A new value of type `ty`, for an instruction to define.
    pub(crate) fn value(&mut self, ty: Type) -> Value {
        self.types.push(ty);
        Value(self.types.len() as u32 - 1)
    }

    /// A new instruction outside every block that runs `op` and defines `result`.
    pub(crate) fn add(&mut self, op: Op, result: Option<Value>) -> Inst {
        self.insts.push(InstData { op, result });
        Inst(self.insts.len() as u32 - 1)
    }

    /// The operation of the last instruction of `block`.
    pub(crate) fn last(&self, block: Block) -> Option<&Op> {
        self.blocks[block.index()].insts.last().map(|inst| &self.insts[inst.index()].op)
    }

    /// The blocks the last instruction of `block` can go to.
    pub(crate) fn successors(&self, block: Block) -> impl Iterator<Item = Block> + '_ {
        self.last(block).map(Op::targets).into_iter().flatten()
    }

    /// The block that a jump to `block` ends in, past every block that only jumps on and carries no values.
    pub(crate) fn jump_target(&self, mut block: Block) -> Block {
        for _ in 0..self.blocks.len() {
            let data = &self.blocks[block.index()];
            let [only] = data.insts[..] else { return block };
            match &self.insts[only.index()].op {
                Op::Jump(next, args) if data.params.is_empty() && args.is_empty() => block = *next,
                _ => return block,
            }
        }
        block
    }

    /// Whether control that enters at `entry` can reach each block.
    pub(crate) fn reachable(&self, entry: Block) -> Vec<bool> {
        let mut seen = vec![false; self.blocks.len()];
        let mut stack = vec![entry];
        while let Some(block) = stack.pop() {
            if !std::mem::replace(&mut seen[block.index()], true) {
                stack.extend(self.successors(block));
            }
        }
        seen
    }

    /// The number of instructions in the body.
    pub fn size(&self) -> usize {
        self.blocks.iter().map(|block| block.insts.len()).sum()
    }

    /// Whether `FunctionBuilder::inline` can copy this function into a caller: it is no coroutine, catches no unwind and reads no frame pointer.
    pub fn can_inline(&self) -> bool {
        self.coroutine.is_none() && self.entry.is_some() && !self.insts.iter().any(|data| matches!(data.op, Op::TryCall { .. } | Op::FramePointer))
    }

    /// Whether any instruction in this body returns to its caller.
    pub fn has_return(&self) -> bool {
        self.insts.iter().any(|data| matches!(data.op, Op::Return(_)))
    }

    /// For each value, the block that defines it.
    pub(crate) fn def_blocks(&self) -> Vec<Option<Block>> {
        let mut defs = vec![None; self.types.len()];
        for (index, block) in self.blocks.iter().enumerate() {
            let here = Some(Block(index as u32));
            block.params.iter().for_each(|value| defs[value.index()] = here);
            for inst in &block.insts {
                crate::live::results(&self.insts[inst.index()]).for_each(|value| defs[value.index()] = here);
            }
        }
        defs
    }

    /// The number of instruction operand reads of each value.
    pub(crate) fn use_counts(&self) -> Vec<u32> {
        let mut uses = vec![0_u32; self.types.len()];
        for inst in self.blocks.iter().flat_map(|block| &block.insts) {
            self.insts[inst.index()].op.for_operands(|value| uses[value.index()] += 1);
        }
        uses
    }

    /// The number of control-flow edges into each block.
    pub(crate) fn pred_counts(&self) -> Vec<u32> {
        let mut preds = vec![0_u32; self.blocks.len()];
        for block in 0..self.blocks.len() {
            for target in self.successors(Block(block as u32)) {
                preds[target.index()] += 1;
            }
        }
        preds
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

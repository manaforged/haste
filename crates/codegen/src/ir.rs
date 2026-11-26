use std::fmt;

mod atomic;
mod compact;
mod consts;
mod edges;
mod flags;
mod op;
mod rodata;
mod vector;

pub use atomic::{AtomicOrd, AtomicRmwOp};
pub(crate) use compact::{compact, compact_slots};
pub(crate) use op::{Binary, BlockData, Convert, InstData, Op, SlotData, Unary};
pub(crate) use vector::{CfgCache, Domain};
pub use vector::{Lane, ReduceOp, ShuffleMask, TargetOpId, TinyArgs};

/// The machine type of one value.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Type {
    I8,
    I16,
    I32,
    I64,
    I128,
    F32,
    F64,
    Vec { lane: Lane, count: u8 },
}

impl Type {
    /// Width in bytes.
    pub const fn bytes(self) -> u32 {
        match self {
            Type::I8 => 1,
            Type::I16 => 2,
            Type::I32 | Type::F32 => 4,
            Type::I64 | Type::F64 => 8,
            Type::I128 => 16,
            Type::Vec { lane, count } => lane.bytes() * (count as u32),
        }
    }

    /// Width in bits.
    pub const fn bits(self) -> u32 {
        self.bytes() * 8
    }

    /// Whether the type holds a scalar floating-point number.
    pub const fn is_float(self) -> bool {
        matches!(self, Type::F32 | Type::F64)
    }

    /// Whether the type is a SIMD register vector.
    pub const fn is_vector(self) -> bool {
        matches!(self, Type::Vec { .. })
    }

    /// Whether a value of the type lives in a float register when it has one.
    pub const fn in_float_register(self) -> bool {
        self.is_float() || self.is_vector()
    }

    /// The integer type `bytes` wide: 1, 2, 4 or 8.
    pub const fn int_of_bytes(bytes: i64) -> Option<Type> {
        match bytes {
            1 => Some(Type::I8),
            2 => Some(Type::I16),
            4 => Some(Type::I32),
            8 => Some(Type::I64),
            _ => None,
        }
    }

    /// Whether the type holds an integer that fits in one general-purpose register.
    pub const fn is_small_int(self) -> bool {
        matches!(self, Type::I8 | Type::I16 | Type::I32 | Type::I64)
    }

    /// The lane kind when `self` is a vector type.
    pub const fn lane(self) -> Option<Lane> {
        match self {
            Type::Vec { lane, .. } => Some(lane),
            _ => None,
        }
    }

    /// The `(lane, count)` shape when `self` is a vector type.
    pub const fn vec_shape(self) -> Option<(Lane, u8)> {
        match self {
            Type::Vec { lane, count } => Some((lane, count)),
            _ => None,
        }
    }

    /// Whether `self` is a valid 8-byte or 16-byte power-of-two vector type.
    pub const fn is_valid_vec(self) -> bool {
        match self {
            Type::Vec { lane, count } => {
                count.is_power_of_two() && matches!(lane.bytes() * (count as u32), 8 | 16)
            }
            _ => false,
        }
    }

    pub(crate) fn unsigned_max(self) -> u64 {
        let bits = self.bits();
        if bits >= 64 {
            u64::MAX
        } else {
            (1_u64 << bits) - 1
        }
    }

    pub(crate) const fn sign_bit(self) -> i64 {
        if matches!(self, Type::F64) {
            i64::MIN
        } else {
            0x8000_0000
        }
    }
}

macro_rules! ids {
    ($($(#[$doc:meta])* $name:ident),*) => {$(
        $(#[$doc])*
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
        pub struct $name(pub(crate) u32);

        impl $name {
            pub(crate) fn index(self) -> usize { self.0 as usize }
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
    pub readonly: bool,
    pub noalias: bool,
    pub deref_bytes: u32,
}

impl AbiParam {
    /// A plain value of `ty`.
    pub fn new(ty: Type) -> Self {
        Self {
            value_type: ty,
            purpose: ArgumentPurpose::Normal,
            readonly: false,
            noalias: false,
            deref_bytes: 0,
        }
    }

    /// A value of `ty` with a special purpose.
    pub fn special(ty: Type, purpose: ArgumentPurpose) -> Self {
        Self {
            value_type: ty,
            purpose,
            readonly: false,
            noalias: false,
            deref_bytes: 0,
        }
    }

    /// A C variadic argument of `ty`.
    pub fn variadic(ty: Type) -> Self {
        Self {
            value_type: ty,
            purpose: ArgumentPurpose::Variadic,
            readonly: false,
            noalias: false,
            deref_bytes: 0,
        }
    }

    /// Marks this parameter as pointing to immutable memory for the duration of the call.
    pub const fn with_readonly(mut self) -> Self {
        self.readonly = true;
        self
    }

    /// Marks this parameter as unaliased with all other accessible memory objects at entry.
    pub const fn with_noalias(mut self) -> Self {
        self.noalias = true;
        self
    }

    /// Marks this parameter as pointing to `bytes` bytes of dereferenceable memory.
    pub const fn with_deref_size(mut self, bytes: u32) -> Self {
        self.deref_bytes = bytes;
        self
    }

    /// Marks this parameter as pointing to `bytes` bytes of dereferenceable, immutable memory.
    pub const fn with_deref_bytes(self, bytes: u32) -> Self {
        self.with_readonly().with_deref_size(bytes)
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
        Self {
            params: Vec::new(),
            returns: Vec::new(),
            call_conv,
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
            IntCC::SignedLessThan
                | IntCC::SignedGreaterThanOrEqual
                | IntCC::SignedGreaterThan
                | IntCC::SignedLessThanOrEqual
        )
    }
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
    pub(crate) value_flags: Vec<u8>,
    /// Whether this function must keep a standard frame pointer chain for unwind stack walking.
    pub keep_fp: bool,
    pub(crate) unwind_drops: Vec<crate::UnwindDrop>,
    pub(crate) no_unwind_syms: Vec<u32>,
    pub(crate) elide_drops: bool,
    pub(crate) frame_private: u32,
    pub(crate) cfg_rev: u32,
    pub(crate) cfg_cache: CfgCache,
}

impl Function {
    /// An empty function with `signature`.
    pub fn new(signature: Signature) -> Self {
        Self {
            signature,
            types: Vec::new(),
            blocks: Vec::new(),
            insts: Vec::new(),
            slots: Vec::new(),
            sigs: Vec::new(),
            entry: None,
            coroutine: None,
            self_slots: Vec::new(),
            vtable_slots: Vec::new(),
            rodata_bytes: Vec::new(),
            value_flags: Vec::new(),
            keep_fp: false,
            unwind_drops: Vec::new(),
            no_unwind_syms: Vec::new(),
            elide_drops: false,
            frame_private: 0,
            cfg_rev: 0,
            cfg_cache: CfgCache::default(),
        }
    }

    pub(crate) fn bump_cfg(&mut self) {
        self.cfg_rev = self.cfg_rev.wrapping_add(1);
    }

    pub(crate) fn cfg(&self) -> std::sync::Arc<crate::cfg::Cfg> {
        let mut guard = self.cfg_cache.0.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((rev, cfg)) = &mut *guard
            && (*rev == self.cfg_rev || cfg.matches(self))
        {
            *rev = self.cfg_rev;
            return std::sync::Arc::clone(cfg);
        }
        let cfg = std::sync::Arc::new(crate::cfg::Cfg::new(self));
        *guard = Some((self.cfg_rev, std::sync::Arc::clone(&cfg)));
        cfg
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

    pub(crate) fn block_has_call(&self, block: Block) -> bool {
        self.blocks[block.index()].insts.iter().any(|&i| {
            matches!(
                self.insts[i.index()].op,
                Op::Call { .. } | Op::TryCall { .. }
            )
        })
    }

    /// Whether this function contains any `Call` or `TryCall` instruction.
    pub fn has_call(&self) -> bool {
        (0..self.blocks.len()).any(|b| self.block_has_call(Block(b as u32)))
    }

    /// Marks relocation symbol `sym` as guaranteed not to unwind.
    pub fn mark_no_unwind_sym(&mut self, sym: u32) {
        self.no_unwind_syms.push(sym);
    }

    /// Requests post-optimization unwind-drop elision even at quick effort.
    pub fn enable_drop_elision(&mut self) {
        self.elide_drops = true;
    }

    /// Whether this function is an unsplit coroutine body.
    pub fn is_coroutine(&self) -> bool {
        self.coroutine.is_some()
    }

    /// Returns `(block, inst_pos, callee_sym)` for every `Op::Call` whose callee is a symbol, a slot address, or a load from either.
    pub fn symbol_call_sites(&self) -> Vec<(Block, usize, u32)> {
        let mut syms = vec![None; self.types.len()];
        for data in &self.insts {
            if let (Op::Symbol(sym) | Op::SlotAddr(sym), Some(res)) = (&data.op, data.result) {
                syms[res.index()] = Some(*sym);
            }
        }
        let named = syms.clone();
        for data in &self.insts {
            if let (Op::Load(addr, 0), Some(res)) = (&data.op, data.result)
                && let Some(sym) = named[addr.index()]
            {
                syms[res.index()] = Some(sym);
            }
        }
        let mut out = Vec::new();
        for (b_idx, block) in self.blocks.iter().enumerate() {
            for (pos, &inst) in block.insts.iter().enumerate() {
                if let Op::Call { callee, .. } = self.insts[inst.index()].op
                    && let Some(sym) = syms[callee.index()]
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

    pub(crate) fn value(&mut self, ty: Type) -> Value {
        self.types.push(ty);
        Value(self.types.len() as u32 - 1)
    }

    pub(crate) fn add(&mut self, op: Op, result: Option<Value>) -> Inst {
        self.insts.push(InstData { op, result, loc: 0 });
        Inst(self.insts.len() as u32 - 1)
    }

    pub(crate) fn last(&self, block: Block) -> Option<&Op> {
        self.blocks[block.index()]
            .insts
            .last()
            .map(|inst| &self.insts[inst.index()].op)
    }

    pub(crate) fn successors(&self, block: Block) -> impl Iterator<Item = Block> + '_ {
        self.last(block).map(Op::targets).into_iter().flatten()
    }

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

    pub(crate) fn home_loc(&self) -> u64 {
        self.blocks
            .iter()
            .flat_map(|block| &block.insts)
            .map(|inst| self.insts[inst.index()].loc)
            .find(|&loc| loc != 0)
            .unwrap_or(0)
    }

    pub(crate) fn entry_const(&mut self, imm: i64, ty: Type) -> Option<Value> {
        let entry = self.entry?;
        let val = self.value(ty);
        let inst = self.add(Op::Const(imm), Some(val));
        let block = &mut self.blocks[entry.index()].insts;
        block.insert(0, inst);
        Some(val)
    }

    /// The number of instructions in the body.
    pub fn size(&self) -> usize {
        self.blocks.iter().map(|block| block.insts.len()).sum()
    }

    /// The number of non-trivial instructions in the body, discounting constant and address-materialization ops.
    pub fn inline_size(&self) -> usize {
        self.blocks
            .iter()
            .flat_map(|b| &b.insts)
            .filter(|i| {
                !matches!(
                    self.insts[i.index()].op,
                    Op::Const(_) | Op::StackAddr(..) | Op::FramePointer | Op::SlotAddr(_)
                )
            })
            .count()
    }

    /// Whether `FunctionBuilder::inline` can copy this function into a caller: it is no coroutine and reads no frame pointer.
    pub fn can_inline(&self) -> bool {
        self.coroutine.is_none()
            && self.entry.is_some()
            && !self
                .blocks
                .iter()
                .flat_map(|b| &b.insts)
                .any(|i| matches!(self.insts[i.index()].op, Op::FramePointer))
    }

    /// Whether any instruction in this body returns to its caller.
    pub fn has_return(&self) -> bool {
        self.blocks
            .iter()
            .flat_map(|b| &b.insts)
            .any(|i| matches!(self.insts[i.index()].op, Op::Return(_)))
    }

    pub(crate) fn def_blocks(&self) -> Vec<Option<Block>> {
        let mut defs = vec![None; self.types.len()];
        for (index, block) in self.blocks.iter().enumerate() {
            let here = Some(Block(index as u32));
            block
                .params
                .iter()
                .for_each(|value| defs[value.index()] = here);
            for inst in &block.insts {
                crate::live::results(&self.insts[inst.index()])
                    .for_each(|value| defs[value.index()] = here);
            }
        }
        defs
    }

    pub(crate) fn use_counts(&self) -> Vec<u32> {
        let mut uses = vec![0_u32; self.types.len()];
        for inst in self.blocks.iter().flat_map(|block| &block.insts) {
            self.insts[inst.index()]
                .op
                .for_operands(|value| uses[value.index()] += 1);
        }
        uses
    }

    pub(crate) fn slot_offsets(
        &self,
        eval_const: impl Fn(Value) -> Option<i64>,
    ) -> (Vec<Option<(StackSlot, i64)>>, Vec<(Inst, StackSlot)>) {
        let mut addrs = vec![None; self.types.len()];
        if self.slots.is_empty() {
            return (addrs, Vec::new());
        }
        let mut addr_insts = Vec::new();
        let mut adds = Vec::new();
        for inst in self.blocks.iter().flat_map(|block| &block.insts) {
            let data = &self.insts[inst.index()];
            let Some(result) = data.result else { continue };
            match &data.op {
                Op::StackAddr(slot, offset) => {
                    addrs[result.index()] = Some((*slot, i64::from(*offset)));
                    addr_insts.push((*inst, *slot));
                }
                Op::Binary(Binary::Iadd, x, c) => adds.push((*inst, result, *x, *c)),
                _ => {}
            }
        }
        if addr_insts.is_empty() || adds.is_empty() {
            return (addrs, addr_insts);
        }
        let consts = self.const_values();
        loop {
            let before = adds.len();
            adds.retain(|&(inst, result, x, c)| {
                let imm_of = |v: Value| consts[v.index()].or_else(|| eval_const(v));
                let Some(((slot, base), imm)) = addrs[x.index()]
                    .zip(imm_of(c))
                    .or_else(|| addrs[c.index()].zip(imm_of(x)))
                else {
                    return true;
                };
                let Some(offset) = base.checked_add(imm) else {
                    return true;
                };
                addrs[result.index()] = Some((slot, offset));
                addr_insts.push((inst, slot));
                false
            });
            if adds.len() == before {
                return (addrs, addr_insts);
            }
        }
    }
}

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "function {:?}", self.signature)?;
        for (index, block) in self.blocks.iter().enumerate() {
            let params: Vec<String> = block
                .params
                .iter()
                .map(|v| format!("v{}: {:?}", v.0, self.ty(*v)))
                .collect();
            writeln!(f, "block{index}({}):", params.join(", "))?;
            for inst in &block.insts {
                let data = &self.insts[inst.index()];
                match data.result {
                    Some(value) if self.is_unpredictable(value) => {
                        writeln!(
                            f,
                            "    v{}: {:?} = {:?} unpredictable",
                            value.0,
                            self.ty(value),
                            data.op
                        )?;
                    }
                    Some(value) => {
                        writeln!(f, "    v{}: {:?} = {:?}", value.0, self.ty(value), data.op)?;
                    }
                    None => writeln!(f, "    {:?}", data.op)?,
                }
            }
        }
        Ok(())
    }
}

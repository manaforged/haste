use crate::ir::{
    Binary, Block, BlockData, Convert, Function, Inst, InstData, IntCC, Op, SigRef, Signature,
    SlotData, StackSlot, Type, Unary, Value,
};

mod inline;
mod ins;
mod query;
mod vector;

/// Appends blocks, values and instructions to a function.
#[derive(Debug)]
pub struct FunctionBuilder<'f> {
    pub func: &'f mut Function,
    current: Option<Block>,
    incoming: Vec<u32>,
    loc: u64,
}

impl<'f> FunctionBuilder<'f> {
    /// A builder that appends to `func`.
    pub fn new(func: &'f mut Function) -> Self {
        Self {
            incoming: vec![0; func.blocks.len()],
            func,
            current: None,
            loc: 0,
        }
    }

    /// Adds an empty block.
    pub fn create_block(&mut self) -> Block {
        self.func.blocks.push(BlockData::default());
        self.incoming.push(0);
        self.func.bump_cfg();
        Block(self.func.blocks.len() as u32 - 1)
    }

    /// Whether an instruction built so far branches to `block`.
    pub fn has_branch_to(&self, block: Block) -> bool {
        self.incoming
            .get(block.index())
            .is_some_and(|count| *count > 0)
    }

    /// Sends the next instructions to `block`.
    pub fn switch_to_block(&mut self, block: Block) {
        self.current = Some(block);
    }

    /// The parameters of `block`.
    pub fn block_params(&self, block: Block) -> &[Value] {
        &self.func.blocks[block.index()].params
    }

    /// Adds a parameter of type `ty` to `block`.
    pub fn append_block_param(&mut self, block: Block, ty: Type) -> Value {
        let value = self.func.value(ty);
        self.func.blocks[block.index()].params.push(value);
        value
    }

    /// Makes `block` the entry, with one parameter per signature parameter.
    pub fn append_block_params_for_function_params(&mut self, block: Block) {
        let types: Vec<Type> = self
            .func
            .signature
            .params
            .iter()
            .map(|param| param.value_type)
            .collect();
        for ty in types {
            self.append_block_param(block, ty);
        }
        self.func.entry = Some(block);
        self.func.bump_cfg();
    }

    /// Adds a frame area of `size` bytes aligned to `1 << align_shift`.
    pub fn create_stack_slot(&mut self, size: u32, align_shift: u8) -> StackSlot {
        self.func.slots.push(SlotData { size, align_shift });
        StackSlot(self.func.slots.len() as u32 - 1)
    }

    /// Makes `signature` available to calls.
    pub fn import_signature(&mut self, signature: Signature) -> SigRef {
        self.func.sigs.push(signature);
        SigRef(self.func.sigs.len() as u32 - 1)
    }

    /// The results of the call `inst`.
    pub fn inst_results(&self, inst: Inst) -> &[Value] {
        match &self.func.insts[inst.index()].op {
            Op::Call { results, .. } => results,
            _ => &[],
        }
    }

    /// The point that `rewind` returns the function to.
    pub fn mark(&self) -> Mark {
        let func = &self.func;
        Mark {
            types: func.types.len(),
            blocks: func.blocks.len(),
            insts: func.insts.len(),
            slots: func.slots.len(),
            sigs: func.sigs.len(),
            unwind_drops: func.unwind_drops.len(),
            current: self.current,
        }
    }

    /// Removes everything appended since `mark`, in every block.
    pub fn rewind(&mut self, mark: Mark) {
        let func = &mut *self.func;
        for data in &func.insts[mark.insts..] {
            for target in data.op.targets() {
                if let Some(count) = self.incoming.get_mut(target.index()) {
                    *count -= 1;
                }
            }
        }
        self.incoming.truncate(mark.blocks);
        func.types.truncate(mark.types);
        func.value_flags.truncate(mark.types);
        func.blocks.truncate(mark.blocks);
        func.insts.truncate(mark.insts);
        func.slots.truncate(mark.slots);
        func.sigs.truncate(mark.sigs);
        func.unwind_drops.truncate(mark.unwind_drops);
        for block in &mut func.blocks {
            block.insts.retain(|inst| inst.index() < mark.insts);
            block.params.retain(|value| value.index() < mark.types);
        }
        func.bump_cfg();
        self.current = mark.current;
    }

    fn append(&mut self, block: Block, data: InstData) {
        if data.op.is_terminator() {
            self.func.bump_cfg();
        }
        for target in data.op.targets() {
            self.incoming[target.index()] += 1;
        }
        self.func.insts.push(data);
        let inst = Inst(self.func.insts.len() as u32 - 1);
        self.func.blocks[block.index()].insts.push(inst);
    }

    /// Appends one instruction to the current block.
    pub fn ins(&mut self) -> Ins<'_> {
        let block = self.current.expect("a block is selected");
        Ins {
            func: self.func,
            block,
            incoming: &mut self.incoming,
            loc: self.loc,
        }
    }

    /// The source place that instructions built from now on carry.
    pub fn loc(&self) -> u64 {
        self.loc
    }

    /// Sets the source place that instructions built from now on carry: file key in the high 32 bits, byte offset in the low 32.
    pub fn set_loc(&mut self, loc: u64) {
        self.loc = loc;
    }
}

/// The sizes of a function's parts at one point of building.
#[derive(Clone, Debug)]
pub struct Mark {
    types: usize,
    blocks: usize,
    insts: usize,
    slots: usize,
    sigs: usize,
    unwind_drops: usize,
    current: Option<Block>,
}

/// Appends one instruction to a block.
#[derive(Debug)]
pub struct Ins<'b> {
    func: &'b mut Function,
    block: Block,
    incoming: &'b mut Vec<u32>,
    loc: u64,
}

impl Ins<'_> {
    fn push(&mut self, op: Op, ty: Option<Type>) -> (Inst, Option<Value>) {
        let result = ty.map(|ty| self.func.value(ty));
        self.func.insts.push(InstData {
            op,
            result,
            loc: self.loc,
        });
        let inst = Inst(self.func.insts.len() as u32 - 1);
        self.func.blocks[self.block.index()].insts.push(inst);
        (inst, result)
    }

    fn value(&mut self, op: Op, ty: Type) -> Value {
        self.push(op, Some(ty))
            .1
            .expect("a value instruction has a result")
    }

    fn effect(&mut self, op: Op) -> Inst {
        if op.is_terminator() {
            self.func.bump_cfg();
        }
        for target in op.targets() {
            if let Some(count) = self.incoming.get_mut(target.index()) {
                *count += 1;
            }
        }
        self.push(op, None).0
    }

    fn constant(&mut self, ty: Type, imm: i64, signed: bool) -> Value {
        if ty != Type::I128 {
            return self.value(Op::Const(imm), ty);
        }
        let low = self.value(Op::Const(imm), Type::I64);
        let high = self.value(Op::Const(if signed { imm >> 63 } else { 0 }), Type::I64);
        self.value(Op::Binary(Binary::Iconcat, low, high), Type::I128)
    }

    fn unary(mut self, op: Unary, x: Value) -> Value {
        let ty = self.func.ty(x);
        self.value(Op::Unary(op, x), ty)
    }

    fn binary(mut self, op: Binary, x: Value, y: Value) -> Value {
        let ty = self.func.ty(x);
        self.value(Op::Binary(op, x, y), ty)
    }

    fn binary_imm(mut self, op: Binary, x: Value, imm: i64) -> Value {
        let ty = self.func.ty(x);
        let y = self.constant(ty, imm, true);
        self.value(Op::Binary(op, x, y), ty)
    }

    fn convert(mut self, op: Convert, ty: Type, x: Value) -> Value {
        self.value(Op::Convert(op, x), ty)
    }

    fn compare_imm(mut self, cc: IntCC, x: Value, imm: i64, signed: bool) -> Value {
        let ty = self.func.ty(x);
        let y = self.constant(ty, imm, signed);
        self.value(Op::Icmp(cc, x, y), Type::I8)
    }
}

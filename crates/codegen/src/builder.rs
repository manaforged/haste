use crate::ir::{
    AtomicRmwOp, Binary, Block, BlockData, Convert, FloatCC, Function, Inst, InstData, IntCC, Op, SigRef, Signature, SlotData,
    StackSlot, Type, Unary, Value,
};
use crate::reloc::RuntimeSymbol;

mod vector;

/// Appends blocks, values and instructions to a function.
#[derive(Debug)]
pub struct FunctionBuilder<'f> {
    pub func: &'f mut Function,
    current: Option<Block>,
    incoming: Vec<u32>,
}

impl<'f> FunctionBuilder<'f> {
    /// A builder that appends to `func`.
    pub fn new(func: &'f mut Function) -> Self {
        let incoming = vec![0; func.blocks.len()];
        Self { func, current: None, incoming }
    }

    /// Adds an empty block.
    pub fn create_block(&mut self) -> Block {
        self.func.blocks.push(BlockData::default());
        self.incoming.push(0);
        Block(self.func.blocks.len() as u32 - 1)
    }

    /// Whether an instruction built so far branches to `block`.
    pub fn has_branch_to(&self, block: Block) -> bool {
        self.incoming.get(block.index()).is_some_and(|count| *count > 0)
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
        let types: Vec<Type> = self.func.signature.params.iter().map(|param| param.value_type).collect();
        for ty in types {
            self.append_block_param(block, ty);
        }
        self.func.entry = Some(block);
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
        Mark { types: func.types.len(), blocks: func.blocks.len(), insts: func.insts.len(), slots: func.slots.len(), sigs: func.sigs.len(), current: self.current }
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
        func.blocks.truncate(mark.blocks);
        func.insts.truncate(mark.insts);
        func.slots.truncate(mark.slots);
        func.sigs.truncate(mark.sigs);
        for block in &mut func.blocks {
            block.insts.retain(|inst| inst.index() < mark.insts);
            block.params.retain(|value| value.index() < mark.types);
        }
        self.current = mark.current;
    }

    /// Copies `callee` into the function as if the current block called it with `args`, then continues in a new block whose parameters are its results.
    pub fn inline(&mut self, callee: &Function, args: &[Value]) -> Vec<Value> {
        self.inline_remap(callee, args, 0, None)
    }

    /// Copies `callee` into the function while shifting its relocation indices by `sym_base`.
    pub fn inline_shifted(&mut self, callee: &Function, args: &[Value], sym_base: u32) -> Vec<Value> {
        self.inline_remap(callee, args, sym_base, None)
    }

    /// Copies `callee` into a dictionary-sharing function, loading its symbols from `dict` at `sym_base`.
    pub fn inline_dict(&mut self, callee: &Function, args: &[Value], dict: Value, sym_base: u32) -> Vec<Value> {
        self.inline_remap(callee, args, sym_base, Some(dict))
    }

    fn inline_remap(&mut self, callee: &Function, args: &[Value], sym_base: u32, dict: Option<Value>) -> Vec<Value> {
        let values: Vec<Value> = callee.types.iter().map(|ty| self.func.value(*ty)).collect();
        let blocks: Vec<Block> = callee.blocks.iter().map(|_| self.create_block()).collect();
        let after = self.create_block();
        let results: Vec<Value> = callee.signature.returns.iter().map(|ret| self.append_block_param(after, ret.value_type)).collect();
        let map = Remap { values, blocks, after, slots: self.func.slots.len() as u32, sigs: self.func.sigs.len() as u32, sym_base, dict };
        self.func.slots.extend(callee.slots.iter().cloned());
        self.func.sigs.extend(callee.sigs.iter().cloned());
        if dict.is_none() {
            for &(vt_sym, off, slot_sym, low) in &callee.vtable_slots {
                self.func.vtable_slots.push((vt_sym + sym_base, off, slot_sym + sym_base, low));
            }
        }
        for (sym, bytes) in &callee.rodata_bytes {
            self.func.rodata_bytes.push((sym + sym_base, bytes.clone()));
        }
        for (index, block) in callee.blocks.iter().enumerate() {
            let target = map.blocks[index];
            self.func.blocks[target.index()].params = block.params.iter().map(|param| map.values[param.index()]).collect();
            for inst in &block.insts {
                let data = map.inst(&callee.insts[inst.index()]);
                self.append(target, data);
            }
        }
        let entry = callee.entry.map_or(after, |entry| map.blocks[entry.index()]);
        self.ins().jump(entry, args);
        self.switch_to_block(after);
        results
    }

    fn append(&mut self, block: Block, data: InstData) {
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
        Ins { func: self.func, block, incoming: &mut self.incoming }
    }

    /// Whether `value` was defined as an immediate constant in the current function.
    pub fn is_const(&self, value: Value) -> bool {
        self.func.insts.iter().rev().take(32).any(|d| d.result == Some(value) && matches!(d.op, Op::Const(_)))
    }
}

impl Function {
    /// Replaces the `Op::Call` at `block[inst_pos]` with an inline copy of `callee`, shifting symbols by `sym_base` and splitting the tail into a continuation block.
    pub fn splice_call(&mut self, block: Block, inst_pos: usize, callee: &Function, sym_base: u32) {
        let old_inst = self.blocks[block.index()].insts[inst_pos];
        let Op::Call { args, results, .. } = std::mem::replace(&mut self.insts[old_inst.index()].op, Op::Trap) else {
            return;
        };
        self.insts[old_inst.index()].result = None;
        let values: Vec<Value> = callee.types.iter().map(|ty| self.value(*ty)).collect();
        let base_block = self.blocks.len() as u32;
        let blocks: Vec<Block> = (0..callee.blocks.len() as u32).map(|i| Block(base_block + i)).collect();
        self.blocks.resize_with(self.blocks.len() + callee.blocks.len(), BlockData::default);
        let after = Block(self.blocks.len() as u32);
        let tail = self.blocks[block.index()].insts.split_off(inst_pos + 1);
        self.blocks[block.index()].insts.pop();
        self.blocks.push(BlockData { params: results, insts: tail });
        let map = Remap { values, blocks, after, slots: self.slots.len() as u32, sigs: self.sigs.len() as u32, sym_base, dict: None };
        self.slots.extend(callee.slots.iter().cloned());
        self.sigs.extend(callee.sigs.iter().cloned());
        for &(vt_sym, off, slot_sym, low) in &callee.vtable_slots {
            self.vtable_slots.push((vt_sym + sym_base, off, slot_sym + sym_base, low));
        }
        for (sym, bytes) in &callee.rodata_bytes {
            self.rodata_bytes.push((sym + sym_base, bytes.clone()));
        }
        for (index, c_block) in callee.blocks.iter().enumerate() {
            let target = map.blocks[index];
            self.blocks[target.index()].params = c_block.params.iter().map(|param| map.values[param.index()]).collect();
            for inst in &c_block.insts {
                let data = map.inst(&callee.insts[inst.index()]);
                self.insts.push(data);
                let new_inst = Inst(self.insts.len() as u32 - 1);
                self.blocks[target.index()].insts.push(new_inst);
            }
        }
        let entry = callee.entry.map_or(after, |entry| map.blocks[entry.index()]);
        let jump = self.add(Op::Jump(entry, args), None);
        self.blocks[block.index()].insts.push(jump);
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
    current: Option<Block>,
}

/// Where an inlined callee's values, blocks, slots and signatures land in the caller.
struct Remap {
    values: Vec<Value>,
    blocks: Vec<Block>,
    after: Block,
    slots: u32,
    sigs: u32,
    sym_base: u32,
    dict: Option<Value>,
}

impl Remap {
    fn inst(&self, data: &InstData) -> InstData {
        let mut op = data.op.clone();
        op.map_operands(|value| self.values[value.index()]);
        op.map_targets(|block| self.blocks[block.index()]);
        match &mut op {
            Op::SlotAddr(sym) | Op::Symbol(sym) => match self.dict {
                Some(dict) => op = Op::Load(dict, ((*sym + self.sym_base) * 8) as i32),
                None => *sym += self.sym_base,
            },
            Op::StackAddr(slot, _) | Op::StackLoad(slot, _) | Op::StackStore(_, slot, _) => slot.0 += self.slots,
            Op::Call { sig, results, .. } => {
                sig.0 += self.sigs;
                results.iter_mut().for_each(|value| *value = self.values[value.index()]);
            }
            Op::TryCall { sig, .. } => sig.0 += self.sigs,
            Op::Return(values) => op = Op::Jump(self.after, std::mem::take(values)),
            _ => {}
        }
        InstData { op, result: data.result.map(|value| self.values[value.index()]) }
    }
}

/// Appends one instruction to a block.
#[derive(Debug)]
pub struct Ins<'b> {
    func: &'b mut Function,
    block: Block,
    incoming: &'b mut Vec<u32>,
}

impl Ins<'_> {
    fn push(&mut self, op: Op, ty: Option<Type>) -> (Inst, Option<Value>) {
        let result = ty.map(|ty| self.func.value(ty));
        self.func.insts.push(InstData { op, result });
        let inst = Inst(self.func.insts.len() as u32 - 1);
        self.func.blocks[self.block.index()].insts.push(inst);
        (inst, result)
    }

    fn value(&mut self, op: Op, ty: Type) -> Value {
        self.push(op, Some(ty)).1.expect("a value instruction has a result")
    }

    fn effect(&mut self, op: Op) -> Inst {
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

    /// Integer comparison producing 1 or 0 as an `I8`.
    pub fn icmp(mut self, cc: IntCC, x: Value, y: Value) -> Value {
        self.value(Op::Icmp(cc, x, y), Type::I8)
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

    /// Float comparison producing 1 or 0 as an `I8`.
    pub fn fcmp(mut self, cc: FloatCC, x: Value, y: Value) -> Value {
        self.value(Op::Fcmp(cc, x, y), Type::I8)
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

    /// A sequentially consistent read of a `ty` at `addr`.
    pub fn atomic_load(mut self, ty: Type, addr: Value) -> Value {
        self.value(Op::AtomicLoad(addr), ty)
    }

    /// A sequentially consistent write of `value` at `addr`.
    pub fn atomic_store(mut self, value: Value, addr: Value) -> Inst {
        self.effect(Op::AtomicStore(value, addr))
    }

    /// Applies `op` with `value` to the `ty` at `addr` and returns the old contents.
    pub fn atomic_rmw(mut self, ty: Type, op: AtomicRmwOp, addr: Value, value: Value) -> Value {
        self.value(Op::AtomicRmw(op, addr, value), ty)
    }

    /// Writes `replacement` at `addr` if it holds `expected`, and returns the old contents.
    pub fn atomic_cas(mut self, addr: Value, expected: Value, replacement: Value) -> Value {
        let ty = self.func.ty(expected);
        self.value(Op::AtomicCas(addr, expected, replacement), ty)
    }

    /// A full memory barrier.
    pub fn fence(mut self) -> Inst {
        self.effect(Op::Fence)
    }

    /// This function's frame pointer.
    pub fn get_frame_pointer(mut self, ty: Type) -> Value {
        self.value(Op::FramePointer, ty)
    }

    /// Calls the code at `callee` with `args`.
    pub fn call_indirect(mut self, sig: SigRef, callee: Value, args: &[Value]) -> Inst {
        let returns: Vec<Type> = self.func.sigs[sig.index()].returns.iter().map(|param| param.value_type).collect();
        let results = returns.into_iter().map(|ty| self.func.value(ty)).collect();
        self.effect(Op::Call { sig, callee, args: args.to_vec(), results })
    }

    /// Calls `callee`, then goes to `normal`, or to `handler` with the exception when the callee unwinds.
    pub fn try_call_indirect(mut self, sig: SigRef, callee: Value, args: &[Value], normal: Block, handler: Block) -> Inst {
        self.effect(Op::TryCall { sig, callee, args: args.to_vec(), normal, handler })
    }

    /// Goes to `block` with `args`.
    pub fn jump(mut self, block: Block, args: &[Value]) -> Inst {
        self.effect(Op::Jump(block, args.to_vec()))
    }

    /// Goes to `then` when `condition` is nonzero, else to `otherwise`.
    pub fn brif(mut self, condition: Value, then: Block, then_args: &[Value], otherwise: Block, else_args: &[Value]) -> Inst {
        self.effect(Op::Brif(condition, then, then_args.to_vec(), otherwise, else_args.to_vec()))
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
}

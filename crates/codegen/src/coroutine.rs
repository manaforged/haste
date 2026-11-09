//! The split of a coroutine's resume function into a state machine: its frame moves to the heap, values that cross blocks live in that frame, and a new entry jumps to the start or to the block after the suspension the state names.

use rustc_hash::FxHashMap as HashMap;

use crate::builder::FunctionBuilder;
use crate::ir::{AbiParam, Binary, Block, CallConv, Function, InstData, IntCC, Op, SlotData, StackSlot, Type, Value};

/// The state word of a coroutine that has not started.
pub const UNRESUMED: i64 = 0;
/// The state word of a coroutine that finished or was dropped.
pub const RETURNED: i64 = 1;
/// The state word of a coroutine suspended at its first suspension point; later points count up from it.
pub const FIRST_SUSPEND: i64 = 3;

/// How a resume function finds its coroutine and where it continues, which `compile` uses to split it.
#[derive(Clone, Debug)]
pub struct Coroutine {
    /// The function parameter that holds the address of the coroutine object.
    pub object: usize,
    /// The block that runs after each suspension point, in state order from `FIRST_SUSPEND`, each taking the function's parameters.
    pub resumes: Vec<Block>,
    /// The blocks whose return ends the coroutine, after which its frame is freed.
    pub done: Vec<Block>,
    /// The offsets of the state word and of the heap frame pointer in the coroutine object.
    pub header: (i32, i32),
    /// The code addresses of `malloc` and `free`.
    pub alloc: i64,
    pub free: i64,
}

struct Frame {
    size: u32,
    slots: Vec<u32>,
    spills: HashMap<Value, i32>,
}

impl Frame {
    fn place(&mut self, size: u32, align: u32) -> u32 {
        let at = self.size.next_multiple_of(align.max(1));
        self.size = at + size;
        at
    }
}

/// Splits `func` when it is a coroutine's resume function.
pub(crate) fn split(func: &mut Function) {
    let Some(spec) = func.coroutine.take() else { return };
    let old_entry = func.entry.expect("a coroutine has an entry");
    let mut frame = Frame { size: 0, slots: Vec::new(), spills: HashMap::default() };
    frame.slots = func.slots.iter().map(|slot| frame.place(slot.size, 1 << slot.align_shift)).collect();
    let defs = def_blocks(func);
    for value in crossing(func, &defs) {
        let bytes = func.types[value.index()].bytes();
        let at = frame.place(bytes, bytes);
        frame.spills.insert(value, at as i32);
    }
    func.slots = vec![SlotData { size: 8, align_shift: 3 }];
    let anchor = StackSlot(0);
    for block in 0..func.blocks.len() {
        rewrite(func, Block(block as u32), (&frame, anchor), spec.done.contains(&Block(block as u32)).then_some(spec.free));
    }
    entry(func, &spec, (old_entry, anchor), frame.size);
}

fn def_blocks(func: &Function) -> Vec<Option<Block>> {
    let mut defs = vec![None; func.types.len()];
    for (index, block) in func.blocks.iter().enumerate() {
        let here = Some(Block(index as u32));
        block.params.iter().for_each(|value| defs[value.index()] = here);
        for inst in &block.insts {
            let data = &func.insts[inst.index()];
            data.result.iter().chain(call_results(&data.op)).for_each(|value| defs[value.index()] = here);
        }
    }
    defs
}

fn call_results(op: &Op) -> &[Value] {
    match op {
        Op::Call { results, .. } => results,
        _ => &[],
    }
}

fn crossing(func: &Function, defs: &[Option<Block>]) -> Vec<Value> {
    let mut found = vec![false; func.types.len()];
    for (index, block) in func.blocks.iter().enumerate() {
        for inst in &block.insts {
            for value in func.insts[inst.index()].op.operands() {
                found[value.index()] |= defs[value.index()].is_some_and(|home| home.index() != index);
            }
        }
    }
    found.iter().enumerate().filter(|(_, crosses)| **crosses).map(|(at, _)| Value(at as u32)).collect()
}

fn push(func: &mut Function, insts: &mut Vec<crate::ir::Inst>, op: Op, ty: Option<Type>) -> Option<Value> {
    let result = ty.map(|ty| {
        func.types.push(ty);
        Value(func.types.len() as u32 - 1)
    });
    func.insts.push(InstData { op, result });
    insts.push(crate::ir::Inst(func.insts.len() as u32 - 1));
    result
}

fn rewrite(func: &mut Function, block: Block, (frame, anchor): (&Frame, StackSlot), free: Option<i64>) {
    let old = std::mem::take(&mut func.blocks[block.index()].insts);
    let mut insts = Vec::new();
    let fp = push(func, &mut insts, Op::StackLoad(anchor, 0), Some(Type::I64)).expect("a load has a result");
    for value in func.blocks[block.index()].params.clone() {
        if let Some(&at) = frame.spills.get(&value) {
            push(func, &mut insts, Op::Store(value, fp, at), None);
        }
    }
    for inst in old {
        let mut data = func.insts[inst.index()].clone();
        let mut loaded = HashMap::default();
        for value in data.op.operands() {
            if let (Some(&at), false) = (frame.spills.get(&value), loaded.contains_key(&value)) {
                let ty = func.types[value.index()];
                loaded.insert(value, push(func, &mut insts, Op::Load(fp, at), Some(ty)).expect("a load has a result"));
            }
        }
        data.op.map_operands(|value| loaded.get(&value).copied().unwrap_or(value));
        data.op = match data.op {
            Op::StackAddr(slot, offset) => {
                let at = push(func, &mut insts, Op::Const(i64::from(frame.slots[slot.index()]) + i64::from(offset)), Some(Type::I64)).expect("a constant has a result");
                Op::Binary(Binary::Iadd, fp, at)
            }
            Op::StackLoad(slot, offset) => Op::Load(fp, frame.slots[slot.index()] as i32 + offset),
            Op::StackStore(value, slot, offset) => Op::Store(value, fp, frame.slots[slot.index()] as i32 + offset),
            op => op,
        };
        if data.op.is_terminator() && let Some(free) = free {
            call(func, &mut insts, free, &[fp], false);
        }
        let spilled: Vec<(Value, i32)> = data.result.iter().chain(call_results(&data.op)).filter_map(|value| frame.spills.get(value).map(|&at| (*value, at))).collect();
        func.insts[inst.index()] = data;
        insts.push(inst);
        for (value, at) in spilled {
            push(func, &mut insts, Op::Store(value, fp, at), None);
        }
    }
    func.blocks[block.index()].insts = insts;
}

fn call(func: &mut Function, insts: &mut Vec<crate::ir::Inst>, code: i64, args: &[Value], returns: bool) -> Option<Value> {
    let mut signature = crate::ir::Signature::new(CallConv::Native);
    signature.params = args.iter().map(|_| AbiParam::new(Type::I64)).collect();
    if returns {
        signature.returns.push(AbiParam::new(Type::I64));
    }
    func.sigs.push(signature);
    let sig = crate::ir::SigRef(func.sigs.len() as u32 - 1);
    let callee = push(func, insts, Op::Const(code), Some(Type::I64)).expect("a constant has a result");
    let results: Vec<Value> = (0..usize::from(returns))
        .map(|_| {
            func.types.push(Type::I64);
            Value(func.types.len() as u32 - 1)
        })
        .collect();
    let result = results.first().copied();
    push(func, insts, Op::Call { sig, callee, args: args.to_vec(), results }, None);
    result
}

fn entry(func: &mut Function, spec: &Coroutine, (old_entry, anchor): (Block, StackSlot), size: u32) {
    let mut b = FunctionBuilder::new(func);
    let (start, init, dispatch) = (b.create_block(), b.create_block(), b.create_block());
    b.append_block_params_for_function_params(start);
    let params = b.block_params(start).to_vec();
    b.switch_to_block(start);
    let object = params[spec.object];
    let state = b.ins().load(Type::I64, object, spec.header.0);
    let fresh = b.ins().icmp_imm(IntCC::Equal, state, UNRESUMED);
    b.ins().brif(fresh, init, &[], dispatch, &[]);
    b.switch_to_block(init);
    let mut insts = std::mem::take(&mut b.func.blocks[init.index()].insts);
    let bytes = push(b.func, &mut insts, Op::Const(i64::from(size.max(8))), Some(Type::I64)).expect("a constant has a result");
    let fp = call(b.func, &mut insts, spec.alloc, &[bytes], true).expect("malloc returns");
    b.func.blocks[init.index()].insts = insts;
    b.ins().store(fp, object, spec.header.1);
    b.ins().stack_store(fp, anchor, 0);
    b.ins().jump(old_entry, &params);
    b.switch_to_block(dispatch);
    let fp = b.ins().load(Type::I64, object, spec.header.1);
    b.ins().stack_store(fp, anchor, 0);
    for (index, &resume) in spec.resumes.iter().enumerate() {
        let next = b.create_block();
        let here = b.ins().icmp_imm(IntCC::Equal, state, FIRST_SUSPEND + index as i64);
        b.ins().brif(here, resume, &params, next, &[]);
        b.switch_to_block(next);
    }
    b.ins().trap();
}

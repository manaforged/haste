//! The split of a coroutine's resume function into a state machine: its frame moves to the heap, values that cross blocks live in that frame, and a new entry jumps to the start or to the block after the suspension the state names.

use rustc_hash::FxHashMap as HashMap;

use crate::builder::FunctionBuilder;
use crate::live::results;
use crate::ir::{AbiParam, Binary, Block, CallConv, Function, InstData, IntCC, Op, SlotData, StackSlot, Type, Value};
use crate::reloc::RuntimeSymbol;

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
}

/// Where a stack slot of a resume function lives after the split: at an offset in the heap frame, or still on the native stack.
#[derive(Clone, Copy)]
enum Home {
    Frame(u32),
    Stack(StackSlot),
}

struct Frame {
    size: u32,
    slots: Vec<Home>,
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
    let kept = framed(func, &spec.resumes);
    let mut stack = vec![SlotData { size: 8, align_shift: 3 }];
    for (slot, framed) in func.slots.iter().zip(kept) {
        let home = match framed {
            true => Home::Frame(frame.place(slot.size, 1 << slot.align_shift)),
            false => {
                stack.push(slot.clone());
                Home::Stack(StackSlot(stack.len() as u32 - 1))
            }
        };
        frame.slots.push(home);
    }
    let defs = def_blocks(func);
    for value in crossing(func, &defs) {
        let bytes = func.types[value.index()].bytes();
        let at = frame.place(bytes, bytes);
        frame.spills.insert(value, at as i32);
    }
    func.slots = stack;
    let anchor = StackSlot(0);
    for block in 0..func.blocks.len() {
        rewrite(func, Block(block as u32), (&frame, anchor), spec.done.contains(&Block(block as u32)));
    }
    entry(func, &spec, (old_entry, anchor), frame.size);
}

/// Which stack slots must live in the heap frame: those whose address is taken, and those that some path from a resume block reads before writing them whole.
fn framed(func: &Function, resumes: &[Block]) -> Vec<bool> {
    let (taken, effects) = slot_effects(func);
    let live = live_in(func, &effects);
    (0..func.slots.len()).map(|at| taken[at] || resumes.iter().any(|resume| live[resume.index()][at])).collect()
}

/// Per slot, whether its address is taken; per block, the slots it reads before writing them whole and the slots it writes whole.
fn slot_effects(func: &Function) -> (Vec<bool>, Vec<(Vec<bool>, Vec<bool>)>) {
    let slots = func.slots.len();
    let mut taken = vec![false; slots];
    let mut effects = vec![(vec![false; slots], vec![false; slots]); func.blocks.len()];
    for (block, (reads, writes)) in func.blocks.iter().zip(effects.iter_mut()) {
        for inst in &block.insts {
            match func.insts[inst.index()].op {
                Op::StackAddr(slot, _) => taken[slot.index()] = true,
                Op::StackLoad(slot, _) => reads[slot.index()] |= !writes[slot.index()],
                Op::StackStore(value, slot, 0) => writes[slot.index()] |= func.types[value.index()].bytes() >= func.slots[slot.index()].size,
                _ => {}
            }
        }
    }
    (taken, effects)
}

/// The slots live on entry to each block, by backward dataflow to a fixed point over `effects`.
fn live_in(func: &Function, effects: &[(Vec<bool>, Vec<bool>)]) -> Vec<Vec<bool>> {
    let slots = func.slots.len();
    let mut live = vec![vec![false; slots]; func.blocks.len()];
    let mut changed = true;
    while changed {
        changed = false;
        for block in (0..func.blocks.len()).rev() {
            let mut out = vec![false; slots];
            for next in func.successors(Block(block as u32)) {
                out.iter_mut().zip(&live[next.index()]).for_each(|(o, &l)| *o |= l);
            }
            let (reads, writes) = &effects[block];
            let fresh: Vec<bool> = (0..slots).map(|at| reads[at] || (out[at] && !writes[at])).collect();
            changed |= fresh != live[block];
            live[block] = fresh;
        }
    }
    live
}

fn def_blocks(func: &Function) -> Vec<Option<Block>> {
    let mut defs = vec![None; func.types.len()];
    for (index, block) in func.blocks.iter().enumerate() {
        let here = Some(Block(index as u32));
        block.params.iter().for_each(|value| defs[value.index()] = here);
        for inst in &block.insts {
            let data = &func.insts[inst.index()];
            results(data).for_each(|value| defs[value.index()] = here);
        }
    }
    defs
}

fn crossing(func: &Function, defs: &[Option<Block>]) -> Vec<Value> {
    let mut found = vec![false; func.types.len()];
    for (index, block) in func.blocks.iter().enumerate() {
        for inst in &block.insts {
            func.insts[inst.index()].op.for_operands(|value| {
                found[value.index()] |= defs[value.index()].is_some_and(|home| home.index() != index);
            });
        }
    }
    found.iter().enumerate().filter(|(_, crosses)| **crosses).map(|(at, _)| Value(at as u32)).collect()
}

fn push(func: &mut Function, insts: &mut Vec<crate::ir::Inst>, op: Op, ty: Option<Type>) -> Option<Value> {
    let result = ty.map(|ty| func.value(ty));
    func.insts.push(InstData { op, result });
    insts.push(crate::ir::Inst(func.insts.len() as u32 - 1));
    result
}

fn rewrite(func: &mut Function, block: Block, (frame, anchor): (&Frame, StackSlot), free: bool) {
    let old = std::mem::take(&mut func.blocks[block.index()].insts);
    if old.is_empty() {
        return;
    }
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
        data.op.for_operands(|value| {
            if let (Some(&at), false) = (frame.spills.get(&value), loaded.contains_key(&value)) {
                let ty = func.types[value.index()];
                loaded.insert(value, push(func, &mut insts, Op::Load(fp, at), Some(ty)).expect("a load has a result"));
            }
        });
        data.op.map_operands(|value| loaded.get(&value).copied().unwrap_or(value));
        data.op = match data.op {
            Op::StackAddr(slot, offset) => match frame.slots[slot.index()] {
                Home::Frame(home) => {
                    let at = push(func, &mut insts, Op::Const(i64::from(home) + i64::from(offset)), Some(Type::I64)).expect("a constant has a result");
                    Op::Binary(Binary::Iadd, fp, at)
                }
                Home::Stack(native) => Op::StackAddr(native, offset),
            },
            Op::StackLoad(slot, offset) => match frame.slots[slot.index()] {
                Home::Frame(home) => Op::Load(fp, home as i32 + offset),
                Home::Stack(native) => Op::StackLoad(native, offset),
            },
            Op::StackStore(value, slot, offset) => match frame.slots[slot.index()] {
                Home::Frame(home) => Op::Store(value, fp, home as i32 + offset),
                Home::Stack(native) => Op::StackStore(value, native, offset),
            },
            op => op,
        };
        if data.op.is_terminator() && free {
            call(func, &mut insts, RuntimeSymbol::Free, &[fp], false);
        }
        let spilled: Vec<(Value, i32)> = results(&data).filter_map(|value| frame.spills.get(&value).map(|&at| (value, at))).collect();
        func.insts[inst.index()] = data;
        insts.push(inst);
        for (value, at) in spilled {
            push(func, &mut insts, Op::Store(value, fp, at), None);
        }
    }
    func.blocks[block.index()].insts = insts;
}

fn call(func: &mut Function, insts: &mut Vec<crate::ir::Inst>, symbol: RuntimeSymbol, args: &[Value], returns: bool) -> Option<Value> {
    let mut signature = crate::ir::Signature::new(CallConv::Native);
    signature.params = args.iter().map(|_| AbiParam::new(Type::I64)).collect();
    if returns {
        signature.returns.push(AbiParam::new(Type::I64));
    }
    func.sigs.push(signature);
    let sig = crate::ir::SigRef(func.sigs.len() as u32 - 1);
    let callee = push(func, insts, Op::Runtime(symbol), Some(Type::I64)).expect("a runtime symbol has a result");
    let results: Vec<Value> = (0..usize::from(returns)).map(|_| func.value(Type::I64)).collect();
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
    let fp = call(b.func, &mut insts, RuntimeSymbol::Malloc, &[bytes], true).expect("malloc returns");
    b.func.blocks[init.index()].insts = insts;
    b.ins().store(fp, object, spec.header.1);
    b.ins().stack_store(fp, anchor, 0);
    b.ins().jump(old_entry, &params);
    b.switch_to_block(dispatch);
    let fp = b.ins().load(Type::I64, object, spec.header.1);
    b.ins().stack_store(fp, anchor, 0);
    if spec.resumes.len() >= 2 {
        let idx = b.ins().iadd_imm(state, -FIRST_SUSPEND);
        let trap = b.create_block();
        let fwds: Vec<Block> = spec.resumes.iter().map(|_| b.create_block()).collect();
        b.ins().br_table(idx, trap, &fwds);
        for (&fwd, &resume) in fwds.iter().zip(&spec.resumes) {
            b.switch_to_block(fwd);
            b.ins().jump(resume, &params);
        }
        b.switch_to_block(trap);
        b.ins().trap();
        return;
    }
    for (index, &resume) in spec.resumes.iter().enumerate() {
        let next = b.create_block();
        let here = b.ins().icmp_imm(IntCC::Equal, state, FIRST_SUSPEND + index as i64);
        b.ins().brif(here, resume, &params, next, &[]);
        b.switch_to_block(next);
    }
    b.ins().trap();
}

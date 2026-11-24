//! The split of a coroutine's resume function into a state machine: its frame moves to the heap, values that cross suspensions live in that frame, and a new entry jumps to the start or to the block after the suspension the state names.

use rustc_hash::FxHashMap as HashMap;

use crate::builder::FunctionBuilder;
use crate::cfg::Cfg;
use crate::ir::{
    AbiParam, Binary, Block, CallConv, Function, Inst, InstData, IntCC, Op, SlotData, StackSlot,
    Type, Value,
};
use crate::live::{BitMatrix, results};
use crate::reloc::RuntimeSymbol;

mod fold_flags;
mod lazy;
mod slots;

/// The lifecycle state stored in word 0 of a coroutine object.
#[repr(i64)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoroutineState {
    Unresumed = 0,
    Returned = 1,
    Poisoned = 2,
    FirstSuspend = 3,
}

impl CoroutineState {
    /// Returns the ABI integer representation of this coroutine state.
    pub const fn code(self) -> i64 {
        self as i64
    }

    /// Returns the ABI integer state code for suspension point `index`.
    pub const fn suspended(index: usize) -> i64 {
        Self::FirstSuspend as i64 + index as i64
    }
}

/// How a resume function finds its coroutine and where it continues, which `compile` uses to split it.
#[derive(Clone, Debug)]
pub struct Coroutine {
    /// The function parameter that holds the address of the coroutine object.
    pub object: usize,
    /// The block that runs after each suspension point, in state order from `Coroutine::FIRST_SUSPEND`, each taking the function's parameters.
    pub resumes: Vec<Block>,
    /// The blocks whose return ends the coroutine, after which its frame is freed.
    pub done: Vec<Block>,
    /// The offsets of the state word and of the heap frame pointer in the coroutine object.
    pub header: (i32, i32),
}

impl Coroutine {
    /// The state word of a coroutine that has not started.
    pub const UNRESUMED: i64 = CoroutineState::Unresumed.code();
    /// The state word of a coroutine that finished or was dropped.
    pub const RETURNED: i64 = CoroutineState::Returned.code();
    /// The state word of a coroutine that panicked while running.
    pub const POISONED: i64 = CoroutineState::Poisoned.code();
    /// The state word of a coroutine suspended at its first suspension point; later points count up from it.
    pub const FIRST_SUSPEND: i64 = CoroutineState::FirstSuspend.code();

    /// Returns the state word of suspension point `index`.
    pub const fn suspend_state(index: usize) -> i64 {
        CoroutineState::suspended(index)
    }
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
    /// The heap size of the frame: a multiple of 16 bytes, so the frame cache picks a size class without rounding.
    const fn bytes(&self) -> u32 {
        if self.size < 16 {
            16
        } else {
            self.size.next_multiple_of(16)
        }
    }

    fn place(&mut self, size: u32, align: u32) -> u32 {
        let at = self.size.next_multiple_of(align.max(1));
        self.size = at + size;
        at
    }

    fn place_colored(
        &mut self,
        start: u32,
        slots: &[(usize, u32, u32)],
        conflicts: &BitMatrix,
    ) -> Vec<u32> {
        let mut order: Vec<usize> = (0..slots.len()).collect();
        order.sort_by_key(|&idx| std::cmp::Reverse((slots[idx].1, slots[idx].2)));
        let mut placed = vec![0_u32; slots.len()];
        let mut assigned = vec![false; slots.len()];
        let mut used: Vec<(u32, u32)> = Vec::new();
        for idx in order {
            let (slot, size, align) = slots[idx];
            let align = align.max(1);
            used.clear();
            for other in 0..slots.len() {
                if assigned[other] && conflicts.get(slot, slots[other].0) && slots[other].1 > 0 {
                    used.push((placed[other], placed[other] + slots[other].1));
                }
            }
            used.sort_unstable();
            let mut at = start.next_multiple_of(align);
            for &(start, end) in &used {
                if at + size <= start {
                    break;
                }
                if at < end {
                    at = end.next_multiple_of(align);
                }
            }
            placed[idx] = at;
            assigned[idx] = true;
            self.size = self.size.max(at + size);
        }
        placed
    }
}

fn build_frame(
    func: &mut Function,
    cfg: &Cfg,
    spec: &Coroutine,
    spills: Vec<Value>,
    is_spill: &[bool],
) -> (Frame, Vec<SlotData>) {
    let framed = slots::framed(func, cfg, (&spec.resumes, &spec.done), is_spill);
    let mut frame = Frame {
        size: 0,
        slots: vec![Home::Stack(StackSlot(0)); func.slots.len()],
        spills: HashMap::default(),
    };
    let mut stack = vec![SlotData {
        size: 8,
        align_shift: 3,
    }];
    let mut on_stack = vec![false; func.slots.len()];
    let (mut private, mut shared) = (Vec::new(), Vec::new());
    for (d, &raw_slot) in framed.raw_of.iter().enumerate() {
        if framed.kept[d] && !func.is_unwind_slot(StackSlot(raw_slot as u32)) {
            let data = &func.slots[raw_slot];
            let placed = if framed.escaped[d] {
                &mut shared
            } else {
                &mut private
            };
            placed.push((d, data.size, 1_u32 << data.align_shift));
        } else {
            stack.push(func.slots[raw_slot].clone());
            frame.slots[raw_slot] = Home::Stack(StackSlot(stack.len() as u32 - 1));
            on_stack[raw_slot] = true;
        }
    }
    for u in &func.unwind_drops {
        for slot in [u.local, u.flag] {
            if !on_stack[slot.index()] {
                stack.push(func.slots[slot.index()].clone());
                frame.slots[slot.index()] = Home::Stack(StackSlot(stack.len() as u32 - 1));
                on_stack[slot.index()] = true;
            }
        }
    }
    for (&(d, _, _), offset) in
        private
            .iter()
            .zip(frame.place_colored(0, &private, &framed.conflicts))
    {
        frame.slots[framed.raw_of[d]] = Home::Frame(offset);
    }
    for value in spills {
        let bytes = func.types[value.index()].bytes();
        let at = frame.place(bytes, bytes);
        frame.spills.insert(value, at as i32);
    }
    func.frame_private = frame.size;
    for (&(d, _, _), offset) in
        shared
            .iter()
            .zip(frame.place_colored(frame.size, &shared, &framed.conflicts))
    {
        frame.slots[framed.raw_of[d]] = Home::Frame(offset);
    }
    (frame, stack)
}

/// Splits `func` when it is a coroutine's resume function.
pub(crate) fn split(func: &mut Function) {
    let Some(spec) = func.coroutine.take() else {
        return;
    };
    let old_entry = func.entry.expect("a coroutine has an entry");
    if spec.resumes.is_empty() {
        ready_entry(func, &spec, old_entry);
        func.bump_cfg();
        return;
    }
    fold_flags::run(func, &spec, old_entry);
    let defs = func.def_blocks();
    let cfg = func.cfg();
    let spills = live_across_suspend(func, &cfg, &spec.resumes, &defs);
    let mut is_spill = vec![false; func.types.len()];
    spills.iter().for_each(|v| is_spill[v.index()] = true);
    let (frame, mut stack) = build_frame(func, &cfg, &spec, spills, &is_spill);
    let lazy = lazy::plan(func, &frame, &is_spill, &mut stack);
    func.slots = stack;
    for u in &mut func.unwind_drops {
        if let Home::Stack(slot) = frame.slots[u.local.index()] {
            u.local = slot;
        }
        if let Home::Stack(slot) = frame.slots[u.flag.index()] {
            u.flag = slot;
        }
    }
    let (anchor, orig_blocks) = (StackSlot(0), func.blocks.len());
    let (init, init_fp, dispatch, disp_fp) =
        entry(func, &spec, (old_entry, anchor), (frame.bytes(), lazy));
    func.bump_cfg();
    let split_cfg = func.cfg();
    let mut active_fp: Vec<Option<Value>> = vec![None; func.blocks.len()];
    active_fp[init.index()] = Some(init_fp);
    active_fp[dispatch.index()] = Some(disp_fp);
    for &block in &split_cfg.rpo {
        let idx = block.index();
        if active_fp[idx].is_none() {
            active_fp[idx] = split_cfg.idom[idx].and_then(|dom| active_fp[dom.index()]);
        }
        if idx < orig_blocks {
            let done = lazy.is_none() && spec.done.contains(&block);
            let chosen = rewrite(
                func,
                block,
                (&frame, anchor),
                (&split_cfg, &defs),
                (active_fp[idx], done),
            );
            active_fp[idx] = active_fp[idx].or(chosen);
        }
    }
    for idx in 0..orig_blocks {
        if !split_cfg.reachable[idx] {
            let block = Block(idx as u32);
            let done = lazy.is_none() && spec.done.contains(&block);
            rewrite(
                func,
                block,
                (&frame, anchor),
                (&split_cfg, &defs),
                (None, done),
            );
        }
    }
    if let Some(lazy) = lazy {
        lazy::settle(func, &spec, anchor, lazy, orig_blocks);
    }
    func.bump_cfg();
}

fn check_unresumed(b: &mut FunctionBuilder<'_>, spec: &Coroutine) -> (Vec<Value>, Value, Value) {
    let start = b.create_block();
    b.append_block_params_for_function_params(start);
    let params = b.block_params(start).to_vec();
    b.switch_to_block(start);
    let object = params[spec.object];
    let state = b.ins().load(Type::I64, object, spec.header.0);
    let fresh = b
        .ins()
        .icmp_imm(IntCC::Equal, state, CoroutineState::Unresumed.code());
    (params, state, fresh)
}

fn ready_entry(func: &mut Function, spec: &Coroutine, old_entry: Block) {
    let mut b = FunctionBuilder::new(func);
    let (params, _, fresh) = check_unresumed(&mut b, spec);
    let trap = b.create_block();
    b.ins().brif(fresh, old_entry, &params, trap, &[]);
    b.switch_to_block(trap);
    b.ins().trap();
}

fn live_across_suspend(
    func: &Function,
    cfg: &Cfg,
    resumes: &[Block],
    defs: &[Option<Block>],
) -> Vec<Value> {
    let mut can_resume = vec![false; func.blocks.len()];
    let mut stack: Vec<Block> = resumes.to_vec();
    while let Some(b) = stack.pop() {
        if !std::mem::replace(&mut can_resume[b.index()], true) {
            for next in func.successors(b) {
                stack.push(next);
            }
        }
    }
    let mut use_blocks: Vec<Vec<Block>> = vec![Vec::new(); func.types.len()];
    for (b_idx, block) in func.blocks.iter().enumerate() {
        if !can_resume[b_idx] {
            continue;
        }
        let blk = Block(b_idx as u32);
        for &inst in &block.insts {
            func.insts[inst.index()].op.for_operands(|value| {
                let idx = value.index();
                if defs[idx].is_some_and(|home| home != blk) && use_blocks[idx].last() != Some(&blk)
                {
                    use_blocks[idx].push(blk);
                }
            });
        }
    }
    let mut is_resume = vec![false; func.blocks.len()];
    for &resume in resumes {
        is_resume[resume.index()] = true;
    }
    let mut seen = vec![0_u32; func.blocks.len()];
    let mut queue = Vec::new();
    let mut spills = Vec::new();
    for (val_idx, uses) in use_blocks.iter().enumerate() {
        if uses.is_empty() {
            continue;
        }
        let Some(def_block) = defs[val_idx] else {
            continue;
        };
        let stamp = (val_idx + 1) as u32;
        queue.clear();
        for &use_blk in uses {
            if std::mem::replace(&mut seen[use_blk.index()], stamp) != stamp {
                queue.push(use_blk);
            }
        }
        let mut crossed = false;
        while let Some(blk) = queue.pop() {
            if is_resume[blk.index()] {
                crossed = true;
                break;
            }
            for &pred in cfg.preds_of(blk) {
                if pred == def_block {
                    continue;
                }
                if std::mem::replace(&mut seen[pred.index()], stamp) != stamp {
                    queue.push(pred);
                }
            }
        }
        if crossed {
            spills.push(Value(val_idx as u32));
        }
    }
    spills
}

fn push(func: &mut Function, insts: &mut Vec<Inst>, op: Op, ty: Option<Type>) -> Option<Value> {
    let result = ty.map(|ty| func.value(ty));
    func.insts.push(InstData { op, result, loc: 0 });
    insts.push(Inst(func.insts.len() as u32 - 1));
    result
}

fn block_needs_fp(func: &Function, block: Block, frame: &Frame, free: bool) -> bool {
    if free {
        return true;
    }
    for &value in &func.blocks[block.index()].params {
        if frame.spills.contains_key(&value) {
            return true;
        }
    }
    for &inst in &func.blocks[block.index()].insts {
        let data = &func.insts[inst.index()];
        match &data.op {
            Op::StackAddr(slot, _) | Op::StackLoad(slot, _) | Op::StackStore(_, slot, _)
                if matches!(frame.slots[slot.index()], Home::Frame(_)) =>
            {
                return true;
            }
            _ => {}
        }
        let mut uses_spill = false;
        data.op
            .for_operands(|value| uses_spill |= frame.spills.contains_key(&value));
        if uses_spill || results(data).any(|value| frame.spills.contains_key(&value)) {
            return true;
        }
    }
    false
}

fn rewrite(
    func: &mut Function,
    block: Block,
    (frame, anchor): (&Frame, StackSlot),
    (cfg, defs): (&Cfg, &[Option<Block>]),
    (inherited_fp, free): (Option<Value>, bool),
) -> Option<Value> {
    if func.blocks[block.index()].insts.is_empty() {
        return inherited_fp;
    }
    let needs_fp = block_needs_fp(func, block, frame, free);
    let old = std::mem::take(&mut func.blocks[block.index()].insts);
    let mut insts = Vec::new();
    let fp = if needs_fp {
        Some(match inherited_fp {
            Some(fp) => fp,
            None => {
                let fp = push(func, &mut insts, Op::StackLoad(anchor, 0), Some(Type::I64))
                    .expect("a load has a result");
                func.mark_frame_ptr(fp);
                fp
            }
        })
    } else {
        None
    };
    let mut loaded = HashMap::default();
    for value in func.blocks[block.index()].params.clone() {
        loaded.insert(value, value);
        if let (Some(&at), Some(fp)) = (frame.spills.get(&value), fp) {
            push(func, &mut insts, Op::Store(value, fp, at), None);
        }
    }
    for inst in old {
        let mut data = func.insts[inst.index()].clone();
        data.op.for_operands(|value| {
            if loaded.contains_key(&value) {
                return;
            }
            if let Some(&at) = frame.spills.get(&value) {
                let dom_live = cfg.reachable[block.index()]
                    && defs[value.index()]
                        .is_some_and(|def| def != block && cfg.dominates(def, block));
                if dom_live {
                    loaded.insert(value, value);
                } else {
                    let fp = fp.expect("spilled operand requires frame pointer");
                    let ty = func.types[value.index()];
                    loaded.insert(
                        value,
                        push(func, &mut insts, Op::Load(fp, at), Some(ty))
                            .expect("a load has a result"),
                    );
                }
            }
        });
        data.op
            .map_operands(|value| loaded.get(&value).copied().unwrap_or(value));
        data.op = match data.op {
            Op::StackAddr(slot, offset) => match frame.slots[slot.index()] {
                Home::Frame(home) => {
                    let fp = fp.expect("framed slot requires frame pointer");
                    let at = push(
                        func,
                        &mut insts,
                        Op::Const(i64::from(home) + i64::from(offset)),
                        Some(Type::I64),
                    )
                    .expect("a constant has a result");
                    Op::Binary(Binary::Iadd, fp, at)
                }
                Home::Stack(native) => Op::StackAddr(native, offset),
            },
            Op::StackLoad(slot, offset) => match frame.slots[slot.index()] {
                Home::Frame(home) => Op::Load(
                    fp.expect("framed load requires frame pointer"),
                    home as i32 + offset,
                ),
                Home::Stack(native) => Op::StackLoad(native, offset),
            },
            Op::StackStore(value, slot, offset) => match frame.slots[slot.index()] {
                Home::Frame(home) => Op::Store(
                    value,
                    fp.expect("framed store requires frame pointer"),
                    home as i32 + offset,
                ),
                Home::Stack(native) => Op::StackStore(value, native, offset),
            },
            op => op,
        };
        if data.op.is_terminator() && free {
            let bytes = push(
                func,
                &mut insts,
                Op::Const(i64::from(frame.bytes())),
                Some(Type::I64),
            )
            .expect("a constant has a result");
            call(
                func,
                &mut insts,
                RuntimeSymbol::Free,
                &[fp.expect("free requires frame pointer"), bytes],
                false,
            );
        }
        let spilled: Vec<(Value, i32)> = results(&data)
            .filter_map(|value| {
                loaded.insert(value, value);
                frame.spills.get(&value).map(|&at| (value, at))
            })
            .collect();
        func.insts[inst.index()] = data;
        insts.push(inst);
        if let Some(fp) = fp {
            for (value, at) in spilled {
                push(func, &mut insts, Op::Store(value, fp, at), None);
            }
        }
    }
    func.blocks[block.index()].insts = insts;
    fp.or(inherited_fp)
}

fn call(
    func: &mut Function,
    insts: &mut Vec<Inst>,
    symbol: RuntimeSymbol,
    args: &[Value],
    returns: bool,
) -> Option<Value> {
    let mut signature = crate::ir::Signature::new(CallConv::Native);
    signature.params = args.iter().map(|_| AbiParam::new(Type::I64)).collect();
    if returns {
        signature.returns.push(AbiParam::new(Type::I64));
    }
    func.sigs.push(signature);
    let sig = crate::ir::SigRef(func.sigs.len() as u32 - 1);
    let callee = push(func, insts, Op::Runtime(symbol), Some(Type::I64))
        .expect("a runtime symbol has a result");
    let results: Vec<Value> = (0..usize::from(returns))
        .map(|_| func.value(Type::I64))
        .collect();
    let result = results.first().copied();
    push(
        func,
        insts,
        Op::Call {
            sig,
            callee,
            args: args.to_vec(),
            results,
        },
        None,
    );
    result
}

fn entry(
    func: &mut Function,
    spec: &Coroutine,
    (old_entry, anchor): (Block, StackSlot),
    (size, lazy): (u32, Option<lazy::Lazy>),
) -> (Block, Value, Block, Value) {
    let mut b = FunctionBuilder::new(func);
    let (params, state, fresh) = check_unresumed(&mut b, spec);
    let object = params[spec.object];
    let (init, dispatch) = (b.create_block(), b.create_block());
    b.ins().brif(fresh, init, &[], dispatch, &[]);
    b.switch_to_block(init);
    let init_fp = if let Some(lazy) = lazy {
        let buf = b.ins().stack_addr(Type::I64, lazy.buf, 0);
        b.func.mark_frame_ptr(buf);
        b.ins().stack_store(object, anchor, 8);
        buf
    } else {
        let mut insts = std::mem::take(&mut b.func.blocks[init.index()].insts);
        let bytes = push(
            b.func,
            &mut insts,
            Op::Const(i64::from(size)),
            Some(Type::I64),
        )
        .expect("a constant has a result");
        let heap = call(b.func, &mut insts, RuntimeSymbol::Malloc, &[bytes], true)
            .expect("malloc returns");
        b.func.blocks[init.index()].insts = insts;
        b.ins().store(heap, object, spec.header.1);
        b.func.mark_unique_ptr(heap);
        b.func.mark_frame_ptr(heap);
        heap
    };
    b.ins().stack_store(init_fp, anchor, 0);
    b.ins().jump(old_entry, &params);
    b.switch_to_block(dispatch);
    let disp_fp = b.ins().load(Type::I64, object, spec.header.1);
    b.func.mark_unique_ptr(disp_fp);
    b.func.mark_frame_ptr(disp_fp);
    b.ins().stack_store(disp_fp, anchor, 0);
    if spec.resumes.len() >= 2 {
        let idx = b
            .ins()
            .iadd_imm(state, -CoroutineState::FirstSuspend.code());
        let trap = b.create_block();
        let fwds: Vec<Block> = spec.resumes.iter().map(|_| b.create_block()).collect();
        b.ins().br_table(idx, trap, &fwds);
        for (&fwd, &resume) in fwds.iter().zip(&spec.resumes) {
            b.switch_to_block(fwd);
            b.ins().jump(resume, &params);
        }
        b.switch_to_block(trap);
        b.ins().trap();
        return (init, init_fp, dispatch, disp_fp);
    }
    for (index, &resume) in spec.resumes.iter().enumerate() {
        let next = b.create_block();
        let here = b
            .ins()
            .icmp_imm(IntCC::Equal, state, CoroutineState::suspended(index));
        b.ins().brif(here, resume, &params, next, &[]);
        b.switch_to_block(next);
    }
    b.ins().trap();
    (init, init_fp, dispatch, disp_fp)
}

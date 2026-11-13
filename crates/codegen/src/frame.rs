use crate::alloc::{Location, Plan, Regs};
use crate::ir::{ArgumentPurpose, Block, Function, Inst, Op, Signature, Type, Value};

pub(crate) const WORD: u32 = 8;
const MAX_ALIGN_SHIFT: u8 = 4;
pub(crate) const PAGE: u32 = 4096;
pub(crate) const INCOMING: i32 = 16;
const EDGE_SLOT: u32 = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Ext {
    Zero,
    Sign,
}

impl Ext {
    pub(crate) const fn of(signed: bool) -> Self {
        if signed { Self::Sign } else { Self::Zero }
    }
}

pub(crate) const fn save_offset(index: usize) -> i32 {
    -((WORD as i32) * (index as i32 + 1))
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Loc<R> {
    Int(R),
    Pair(R, R),
    Float(u8),
    Stack(u32),
    Struct(u32, u32),
}

pub(crate) struct Abi<R> {
    pub params: Vec<Loc<R>>,
    pub returns: Vec<Loc<R>>,
    pub stack: u32,
}

pub(crate) fn round(value: u32, to: u32) -> u32 {
    value.div_ceil(to) * to
}

pub(crate) fn returns<R: Copy>(signature: &Signature, ints: &[R], floats: u8) -> Result<Vec<Loc<R>>, String> {
    let (mut int, mut float) = (0, 0);
    let place = |ty: Type| match ty {
        Type::F32 | Type::F64 | Type::V128 if float < floats => {
            float += 1;
            Ok(Loc::Float(float - 1))
        }
        Type::I128 if int == 0 => {
            int = 2;
            Ok(Loc::Pair(ints[0], ints[1]))
        }
        Type::I8 | Type::I16 | Type::I32 | Type::I64 if int < ints.len() => {
            int += 1;
            Ok(Loc::Int(ints[int - 1]))
        }
        _ => Err("too many return values".to_owned()),
    };
    signature.returns.iter().map(|param| param.value_type).map(place).collect()
}

#[derive(Clone, Copy)]
pub(crate) enum Step {
    Move(Value, Value),
    Save(Value, usize),
    Restore(usize, Value),
}

#[derive(Clone, Copy)]
enum Source {
    Value(Value),
    Scratch(usize),
}

pub(crate) fn edge_steps(args: &[Value], params: &[Value], same_home: impl Fn(Value, Value) -> bool) -> Vec<Step> {
    let mut pending: Vec<(Source, Value)> = args.iter().zip(params).filter(|(arg, param)| !same_home(**arg, **param)).map(|(arg, param)| (Source::Value(*arg), *param)).collect();
    let mut steps = Vec::with_capacity(pending.len());
    let mut saved = 0;
    while !pending.is_empty() {
        let read = |at: usize| pending.iter().enumerate().any(|(other, (source, _))| other != at && matches!(source, Source::Value(arg) if same_home(*arg, pending[at].1)));
        if let Some(at) = (0..pending.len()).find(|at| !read(*at)) {
            steps.push(match pending.remove(at) {
                (Source::Value(arg), param) => Step::Move(arg, param),
                (Source::Scratch(index), param) => Step::Restore(index, param),
            });
            continue;
        }
        let Some((at, arg)) = pending.iter().enumerate().find_map(|(at, (source, _))| if let Source::Value(arg) = source { Some((at, *arg)) } else { None }) else { break };
        steps.push(Step::Save(arg, saved));
        pending[at].0 = Source::Scratch(saved);
        saved += 1;
    }
    steps
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum ArgStep<T> {
    Move(T, T),
    Save(T),
    Restore(T),
    Load(Value, T),
}

#[derive(Clone, Copy)]
enum RegSrc<T> {
    Reg(T),
    Saved,
}

pub(crate) fn reg_steps<T: Copy + PartialEq>(pairs: impl IntoIterator<Item = (T, T)>) -> Vec<ArgStep<T>> {
    let mut reg_moves: Vec<(RegSrc<T>, T)> = pairs.into_iter().filter(|(src, dst)| src != dst).map(|(src, dst)| (RegSrc::Reg(src), dst)).collect();
    let mut steps = Vec::with_capacity(reg_moves.len() + 1);
    while !reg_moves.is_empty() {
        let blocked = |at: usize| reg_moves.iter().enumerate().any(|(other, &(s, _))| other != at && matches!(s, RegSrc::Reg(r) if r == reg_moves[at].1));
        if let Some(at) = (0..reg_moves.len()).find(|&at| !blocked(at)) {
            steps.push(match reg_moves.remove(at) {
                (RegSrc::Reg(src), dst) => ArgStep::Move(src, dst),
                (RegSrc::Saved, dst) => ArgStep::Restore(dst),
            });
            continue;
        }
        let Some((at, src)) = reg_moves.iter().enumerate().find_map(|(at, &(s, _))| if let RegSrc::Reg(r) = s { Some((at, r)) } else { None }) else { break };
        steps.push(ArgStep::Save(src));
        reg_moves[at].0 = RegSrc::Saved;
    }
    steps
}

pub(crate) fn arg_steps<T: Copy + PartialEq>(targets: &[(Value, T)], current: impl Fn(Value) -> Option<T>) -> Vec<ArgStep<T>> {
    let mut pairs = Vec::new();
    let mut loads = Vec::new();
    for &(val, dst) in targets {
        match current(val) {
            Some(src) => pairs.push((src, dst)),
            None => loads.push((val, dst)),
        }
    }
    let mut steps = reg_steps(pairs);
    steps.extend(loads.into_iter().map(|(val, dst)| ArgStep::Load(val, dst)));
    steps
}

pub(crate) struct Frame<R> {
    pub values: Vec<i32>,
    pub slots: Vec<i32>,
    pub scratch: i32,
    pub size: u32,
    pub saves: Vec<R>,
    pub float_saves: Vec<u8>,
}

impl<R> Frame<R> {
    pub(crate) fn edge_slot(&self, index: usize) -> i32 {
        self.scratch + (EDGE_SLOT as usize * index) as i32
    }
}

pub(crate) fn same_home<R: Copy + PartialEq>(locations: &[Location<R>], values: &[i32], a: Value, b: Value) -> bool {
    a == b
        || matches!((locations[a.index()], locations[b.index()]), (Location::Reg(x), Location::Reg(y)) if x == y)
        || matches!((locations[a.index()], locations[b.index()]), (Location::Float(x), Location::Float(y)) if x == y)
        || (matches!(locations[a.index()], Location::Slot) && matches!(locations[b.index()], Location::Slot) && values[a.index()] == values[b.index()])
}

fn live_ops(func: &Function) -> impl Iterator<Item = &Op> {
    func.blocks.iter().flat_map(|block| &block.insts).map(|inst| &func.insts[inst.index()].op)
}

pub(crate) fn saved<R: Copy>(func: &Function, used: &[R], all: &[R]) -> Vec<R> {
    let handles = live_ops(func).any(|op| matches!(op, Op::TryCall { .. }));
    if handles { all.to_vec() } else { used.to_vec() }
}

pub(crate) fn framed<R>(func: &Function, frame: &Frame<R>, stack_params: bool) -> bool {
    stack_params || frame.size > 0 || !frame.saves.is_empty() || !frame.float_saves.is_empty() || live_ops(func).any(|op| matches!(op, Op::Call { .. } | Op::TryCall { .. } | Op::FramePointer))
}

fn needs<R>(func: &Function, classify: &impl Fn(&Signature) -> Result<Abi<R>, String>) -> Result<u32, String> {
    let mut outgoing = 0;
    for op in live_ops(func) {
        if let Op::Call { sig, .. } | Op::TryCall { sig, .. } = op {
            outgoing = outgoing.max(classify(&func.sigs[sig.index()])?.stack);
        }
    }
    Ok(outgoing)
}

fn present(func: &Function) -> (Vec<bool>, Vec<bool>) {
    let (mut values, mut slots) = (vec![false; func.types.len()], vec![false; func.slots.len()]);
    for block in &func.blocks {
        block.params.iter().for_each(|param| values[param.index()] = true);
        for inst in &block.insts {
            let data = &func.insts[inst.index()];
            data.op.for_operands(|value| values[value.index()] = true);
            crate::live::results(data).for_each(|value| values[value.index()] = true);
            if let Op::StackAddr(slot, _) | Op::StackLoad(slot, _) | Op::StackStore(_, slot, _) = data.op {
                slots[slot.index()] = true;
            }
        }
    }
    (values, slots)
}

pub(crate) fn layout<R: Copy + PartialEq>(
    func: &Function,
    (saves, float_saves): (Vec<R>, Vec<u8>),
    (locations, shared, homed, elided): (&[Location<R>], &[Option<u32>], &[bool], &[bool]),
    classify: &impl Fn(&Signature) -> Result<Abi<R>, String>,
) -> Result<Frame<R>, String> {
    let outgoing = needs(func, classify)?;
    let (present, addressed) = present(func);
    let mut cursor = (saves.len() + float_saves.len()) as u32 * WORD;
    let mut values = Vec::with_capacity(func.types.len());
    let mut homes: Vec<i32> = Vec::new();
    for (value, ty) in func.types.iter().enumerate() {
        let static_loc = matches!(locations[value], Location::Imm(_) | Location::SlotAddr(_) | Location::Frame(..));
        let reg_unhomed = matches!(locations[value], Location::Reg(_) | Location::Float(_)) && homed.get(value) != Some(&true);
        if !present[value] || elided.get(value) == Some(&true) || static_loc || reg_unhomed {
            values.push(0);
            continue;
        }
        let Some(home) = shared.get(value).copied().flatten().map(|home| home as usize) else {
            cursor += round(ty.bytes(), WORD);
            values.push(-(cursor as i32));
            continue;
        };
        if homes.len() <= home {
            homes.resize(home + 1, 0);
        }
        if homes[home] == 0 {
            cursor += round(ty.bytes(), WORD);
            homes[home] = -(cursor as i32);
        }
        values.push(homes[home]);
    }
    let mut max_saved = 0_usize;
    let mut check_edge = |block: Block, args: &[Value]| {
        for step in edge_steps(args, &func.blocks[block.index()].params, |a, b| same_home(locations, &values, a, b)) {
            if let Step::Save(_, index) = step {
                max_saved = max_saved.max(index + 1);
            }
        }
    };
    for op in live_ops(func) {
        match op {
            Op::Jump(block, args) => check_edge(*block, args),
            Op::Brif(_, then, then_args, otherwise, else_args) => {
                check_edge(*then, then_args);
                check_edge(*otherwise, else_args);
            }
            _ => {}
        }
    }
    let scratch = max_saved as u32 * EDGE_SLOT;
    let mut slots = vec![0; func.slots.len()];
    let mut order: Vec<usize> = (0..func.slots.len()).filter(|&slot| addressed[slot]).collect();
    order.sort_by_key(|&slot| (func.slots[slot].size, slot));
    for slot in order {
        let data = &func.slots[slot];
        let align = 1 << data.align_shift.min(MAX_ALIGN_SHIFT);
        cursor = round(cursor + data.size, align);
        slots[slot] = -(cursor as i32);
    }
    cursor = round(cursor + scratch, 16);
    let scratch = -(cursor as i32);
    let size = round(cursor + outgoing, 16);
    Ok(Frame { values, slots, scratch, size, saves, float_saves })
}

pub(crate) struct Prepared<R> {
    pub order: Vec<Block>,
    pub copied: Vec<bool>,
    pub abi: Abi<R>,
    pub plan: Plan<R>,
    pub frame: Frame<R>,
    pub framed: bool,
}

pub(crate) fn prepare<R: Copy + PartialEq>(
    func: &Function,
    regs: &Regs<R>,
    weighed: bool,
    classify: impl Fn(&Signature) -> Result<Abi<R>, String>,
) -> Result<Prepared<R>, String> {
    let order = crate::live::order(func);
    let copied = crate::live::copied_tests(func, &order);
    let abi = classify(&func.signature)?;
    let plan = crate::alloc::plan(func, regs, (&order, &copied, weighed), &abi.params, &classify);
    let saves = saved(func, &plan.used, regs.saved);
    let float_saves = saved(func, &plan.used_floats, regs.saved_float);
    let frame = layout(func, (saves, float_saves), (&plan.locations, &plan.shared, &plan.homed, &plan.elided), &classify)?;
    let stack_params = abi.params.iter().any(|loc| matches!(loc, Loc::Stack(_) | Loc::Struct(..)));
    let framed = framed(func, &frame, stack_params);
    Ok(Prepared { order, copied, abi, plan, frame, framed })
}

#[derive(Clone, Copy)]
pub(crate) struct ShrinkWrap {
    pub fast_ret: Block,
    pub slow_entry: Block,
}

fn pure_value<R: Copy + PartialEq>(value: Value, plan: &Plan<R>, frame: &Frame<R>, clobbers: &[R]) -> bool {
    if plan.elided[value.index()] {
        return true;
    }
    match plan.location(value) {
        Location::Imm(_) => true,
        Location::Reg(r) => !frame.saves.contains(&r) && !clobbers.contains(&r),
        Location::Float(f) => !frame.float_saves.contains(&f),
        Location::Slot | Location::SlotAddr(_) | Location::Frame(..) => false,
    }
}

fn pure_op(op: &Op) -> bool {
    !matches!(
        op,
        Op::StackAddr(..)
            | Op::StackLoad(..)
            | Op::StackStore(..)
            | Op::FramePointer
            | Op::Store(..)
            | Op::Call { .. }
            | Op::TryCall { .. }
            | Op::Copy(..)
            | Op::Move(..)
            | Op::AtomicLoad(..)
            | Op::AtomicStore(..)
            | Op::AtomicRmw(..)
            | Op::AtomicCas(..)
            | Op::Fence
            | Op::Trap
            | Op::Trapz(..)
    )
}

fn pure_block<R: Copy + PartialEq>(func: &Function, block: Block, plan: &Plan<R>, frame: &Frame<R>, clobbers: &[R]) -> bool {
    let insts = &func.blocks[block.index()].insts;
    if insts.is_empty() {
        return false;
    }
    for &inst in &insts[..insts.len() - 1] {
        let data = &func.insts[inst.index()];
        if !pure_op(&data.op) || data.result.is_some_and(|r| !pure_value(r, plan, frame, clobbers)) {
            return false;
        }
        let mut ok = true;
        data.op.for_operands(|v| ok &= pure_value(v, plan, frame, clobbers));
        if !ok {
            return false;
        }
    }
    true
}

pub(crate) fn shrink_wrap<R: Copy + PartialEq>(func: &Function, framed: bool, plan: &Plan<R>, frame: &Frame<R>, params: &[Loc<R>], clobbers: &[R]) -> Option<ShrinkWrap> {
    if !framed || func.signature.params.iter().any(|p| p.purpose == ArgumentPurpose::StructReturn) {
        return None;
    }
    let entry = func.entry?;
    if params.iter().any(|loc| matches!(loc, Loc::Stack(_) | Loc::Struct(..) | Loc::Pair(..))) {
        return None;
    }
    if func.blocks[entry.index()].params.iter().any(|&p| !pure_value(p, plan, frame, clobbers)) {
        return None;
    }
    let mut preds = vec![0_u32; func.blocks.len()];
    for block in &func.blocks {
        if let Some(&term) = block.insts.last() {
            func.insts[term.index()].op.targets().for_each(|t| preds[t.index()] += 1);
        }
    }
    if preds[entry.index()] != 0 || !pure_block(func, entry, plan, frame, clobbers) {
        return None;
    }
    let term = *func.blocks[entry.index()].insts.last()?;
    let Op::Brif(cond, then_b, then_args, else_b, else_args) = &func.insts[term.index()].op else { return None };
    if *then_b == *else_b || preds[then_b.index()] != 1 || preds[else_b.index()] != 1 || !pure_value(*cond, plan, frame, clobbers) {
        return None;
    }
    let fast_ret_ok = |block: Block, edge_args: &[Value]| {
        func.jump_target(block) == block
            && edge_args.iter().all(|&a| pure_value(a, plan, frame, clobbers))
            && func.blocks[block.index()].params.iter().all(|&p| pure_value(p, plan, frame, clobbers))
            && !edge_steps(edge_args, &func.blocks[block.index()].params, |a, b| plan.same_home(frame, a, b))
                .iter()
                .any(|s| matches!(s, Step::Save(..) | Step::Restore(..)))
            && pure_block(func, block, plan, frame, clobbers)
            && func.blocks[block.index()].insts.last().is_some_and(|&t| match &func.insts[t.index()].op {
                Op::Return(vals) => vals.iter().all(|&v| pure_value(v, plan, frame, clobbers)),
                _ => false,
            })
    };
    if fast_ret_ok(*then_b, then_args) && func.jump_target(*else_b) == *else_b {
        Some(ShrinkWrap { fast_ret: *then_b, slow_entry: *else_b })
    } else if fast_ret_ok(*else_b, else_args) && func.jump_target(*then_b) == *then_b {
        Some(ShrinkWrap { fast_ret: *else_b, slow_entry: *then_b })
    } else {
        None
    }
}

pub(crate) trait Emitter<'a>: Sized {
    fn func(&self) -> &'a Function;
    fn prologue(&mut self);
    fn intake(&mut self, entry: Block) -> Result<(), String>;
    fn set_next(&mut self, next: Option<Block>);
    fn bind_block(&mut self, block: Block);
    fn clear_flags(&mut self);
    fn is_elided(&self, value: Value) -> bool;
    fn inst(&mut self, inst: Inst) -> Result<(), String>;
    fn trap(&mut self);
    fn finish(self) -> Result<crate::Code, String>;

    fn deferred_prologue(&self) -> bool {
        false
    }

    fn on_block_bound(&mut self, _block: Block) -> Result<(), String> {
        Ok(())
    }
}

pub(crate) fn emit_body<'a, E: Emitter<'a>>(emit: &mut E, block: Block) -> Result<(), String> {
    emit.clear_flags();
    let func = emit.func();
    let insts = &func.blocks[block.index()].insts;
    for &inst in insts {
        let data = &func.insts[inst.index()];
        if !matches!(data.op, Op::Call { .. }) && data.result.is_some_and(|res| emit.is_elided(res)) {
            continue;
        }
        emit.inst(inst)?;
    }
    if !insts.last().is_some_and(|inst| func.insts[inst.index()].op.is_terminator()) {
        emit.trap();
    }
    Ok(())
}

pub(crate) fn drive<'a, E: Emitter<'a>>(mut emit: E, order: &[Block]) -> Result<crate::Code, String> {
    let entry = emit.func().entry.ok_or("the function has no entry block")?;
    if !emit.deferred_prologue() {
        emit.prologue();
    }
    emit.intake(entry)?;
    for (index, &block) in order.iter().enumerate() {
        emit.set_next(order.get(index + 1).copied());
        emit.bind_block(block);
        emit.on_block_bound(block)?;
        emit_body(&mut emit, block)?;
    }
    emit.finish()
}

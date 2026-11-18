use crate::alloc::{Location, Plan, Regs};
use crate::ir::{ArgumentPurpose, Block, Function, Inst, Op, Signature, Type, Value};

pub(crate) const WORD: u32 = 8;
/// The most alignment, as a shift, that the frame gives a slot.
pub(crate) const MAX_ALIGN_SHIFT: u8 = 4;
pub(crate) const PAGE: u32 = 4096;
pub(crate) const INCOMING: i32 = 16;
const EDGE_SLOT: u32 = 16;

const fn edge_bytes(wide: bool) -> u32 {
    if wide { 2 * EDGE_SLOT } else { EDGE_SLOT }
}

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
        Type::F32 | Type::F64 | Type::Vec { .. } if float < floats => {
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

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum EdgeScratch {
    Int(usize),
    Float(usize),
    Stack(usize),
}

#[derive(Clone, Copy)]
pub(crate) enum Step {
    Move(Value, Value),
    Save(Value, EdgeScratch),
    Restore(EdgeScratch, Value),
}

#[derive(Clone, Copy)]
enum Source {
    Value(Value),
    Scratch(EdgeScratch),
}

fn edge_kind<R>(func: &Function, locations: &[Location<R>], arg: Value, param: Value) -> usize {
    if matches!(locations[arg.index()], Location::Reg(_)) && matches!(locations[param.index()], Location::Reg(_)) {
        return 0;
    }
    if matches!(locations[arg.index()], Location::Float(_)) && matches!(locations[param.index()], Location::Float(_)) {
        return 1;
    }
    let (a_ty, p_ty) = (func.ty(arg), func.ty(param));
    if a_ty.bytes() <= 8
        && p_ty.bytes() <= 8
        && !a_ty.in_float_register()
        && !p_ty.in_float_register()
        && !matches!(locations[arg.index()], Location::Float(_))
        && !matches!(locations[param.index()], Location::Float(_))
    {
        0
    } else if a_ty.in_float_register() && p_ty.in_float_register() {
        1
    } else {
        2
    }
}

fn alloc_scratch(busy: &mut Vec<bool>) -> usize {
    match busy.iter().position(|&used| !used) {
        Some(idx) => {
            busy[idx] = true;
            idx
        }
        None => {
            busy.push(true);
            busy.len() - 1
        }
    }
}

pub(crate) fn edge_steps<R: Copy + PartialEq>(
    func: &Function,
    locations: &[Location<R>],
    values: &[i32],
    args: &[Value],
    params: &[Value],
) -> Vec<Step> {
    let mut pending: Vec<(Source, Value)> = args
        .iter()
        .zip(params)
        .filter(|(arg, param)| !same_home(locations, values, **arg, **param))
        .map(|(arg, param)| (Source::Value(*arg), *param))
        .collect();
    let mut steps = Vec::with_capacity(pending.len());
    let mut busy = [Vec::new(), Vec::new(), Vec::new()];
    while !pending.is_empty() {
        let read = |at: usize| pending.iter().enumerate().any(|(other, (source, _))| other != at && matches!(source, Source::Value(arg) if same_home(locations, values, *arg, pending[at].1)));
        if let Some(at) = (0..pending.len()).find(|at| !read(*at)) {
            steps.push(match pending.remove(at) {
                (Source::Value(arg), param) => Step::Move(arg, param),
                (Source::Scratch(scratch), param) => {
                    match scratch {
                        EdgeScratch::Int(i) => busy[0][i] = false,
                        EdgeScratch::Float(i) => busy[1][i] = false,
                        EdgeScratch::Stack(i) => busy[2][i] = false,
                    }
                    Step::Restore(scratch, param)
                }
            });
            continue;
        }
        let Some((at, arg)) = pending.iter().enumerate().find_map(|(at, (source, _))| if let Source::Value(arg) = source { Some((at, *arg)) } else { None }) else { break };
        let kind = edge_kind(func, locations, arg, pending[at].1);
        let scratch = match kind {
            0 => match alloc_scratch(&mut busy[0]) {
                i if i < 2 => EdgeScratch::Int(i),
                i => {
                    busy[0][i] = false;
                    EdgeScratch::Stack(alloc_scratch(&mut busy[2]))
                }
            },
            1 => match alloc_scratch(&mut busy[1]) {
                i if i < 2 => EdgeScratch::Float(i),
                i => {
                    busy[1][i] = false;
                    EdgeScratch::Stack(alloc_scratch(&mut busy[2]))
                }
            },
            _ => EdgeScratch::Stack(alloc_scratch(&mut busy[2])),
        };
        steps.push(Step::Save(arg, scratch));
        pending[at].0 = Source::Scratch(scratch);
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
    pub wide: bool,
}

impl<R> Frame<R> {
    pub(crate) fn edge_slot(&self, index: usize) -> i32 {
        self.scratch + (edge_bytes(self.wide) as usize * index) as i32
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

fn needs_frame<R>(func: &Function, frame: &Frame<R>, stack_params: bool, bracketed: bool) -> bool {
    stack_params
        || frame.size > 0
        || !frame.saves.is_empty()
        || !frame.float_saves.is_empty()
        || (0..func.blocks.len() as u32).map(Block).any(|block| {
            let traps = bracketed && matches!(func.last(block), Some(Op::Trap));
            func.blocks[block.index()].insts.iter().any(|inst| match func.insts[inst.index()].op {
                Op::Call { .. } => !traps,
                Op::TryCall { .. } | Op::FramePointer => true,
                _ => false,
            })
        })
}

pub(crate) fn framed<R>(func: &Function, frame: &Frame<R>, stack_params: bool) -> bool {
    needs_frame(func, frame, stack_params, false)
}

/// Whether `func` needs a frame only for calls in blocks that end in a trap, so each of those calls can push its own frame record and the hot path runs frameless.
pub(crate) fn brackets_calls<R>(func: &Function, frame: &Frame<R>, abi: &Abi<R>) -> bool {
    let stack_params = stack_params(abi);
    needs_frame(func, frame, stack_params, false) && !needs_frame(func, frame, stack_params, true)
}

fn stack_params<R>(abi: &Abi<R>) -> bool {
    abi.params.iter().any(|loc| matches!(loc, Loc::Stack(_) | Loc::Struct(..)))
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

fn present(func: &Function, fused: &[bool]) -> (Vec<bool>, Vec<bool>) {
    let (mut values, mut slots) = (vec![false; func.types.len()], vec![false; func.slots.len()]);
    let mut flags = Vec::new();
    let cfg = func.cfg();
    for (at, block) in func.blocks.iter().enumerate() {
        let entered = func.entry == Some(Block(at as u32)) || cfg.pred_count(Block(at as u32)) != 0;
        if entered {
            block.params.iter().for_each(|param| values[param.index()] = true);
        }
        for inst in &block.insts {
            let data = &func.insts[inst.index()];
            data.op.for_operands(|value| values[value.index()] = true);
            crate::live::results(data).for_each(|value| values[value.index()] = true);
            if let Op::StackAddr(slot, _) | Op::StackLoad(slot, _) | Op::StackStore(_, slot, _) = data.op {
                slots[slot.index()] = true;
            }
            if let (Op::Icmp(..) | Op::Fits(..) | Op::Fcmp(..), Some(result)) = (&data.op, data.result)
                && fused.get(result.index()) == Some(&true)
            {
                flags.push(result);
            }
        }
    }
    flags.into_iter().for_each(|value| values[value.index()] = false);
    (values, slots)
}

pub(crate) fn terminator_edges<'a>(op: &'a Op, mut f: impl FnMut(Block, &'a [Value])) {
    match op {
        Op::Jump(target, args) => f(*target, args),
        Op::Brif(_, then, then_args, otherwise, else_args) => {
            f(*then, then_args);
            f(*otherwise, else_args);
        }
        _ => {}
    }
}

pub(crate) fn layout<R: Copy + PartialEq>(
    func: &Function,
    (saves, float_saves): (Vec<R>, Vec<u8>),
    (locations, shared, homed, elided, fused): (&[Location<R>], &[Option<u32>], &[bool], &[bool], &[bool]),
    classify: &impl Fn(&Signature) -> Result<Abi<R>, String>,
) -> Result<Frame<R>, String> {
    let outgoing = needs(func, classify)?;
    let (present, addressed) = present(func, fused);
    let mut cursor = (saves.len() + float_saves.len()) as u32 * WORD;
    let mut values = vec![0; func.types.len()];
    let mut homes: Vec<i32> = Vec::new();
    let mut value_order: Vec<usize> = (0..func.types.len()).collect();
    value_order.sort_by_key(|&v| (
        !matches!((homed.get(v), locations[v]), (Some(&true), Location::Reg(_))),
        !matches!((homed.get(v), locations[v], func.types[v].bytes()), (Some(&true), Location::Float(_), 8)),
        v,
    ));
    for value in value_order {
        let ty = func.types[value];
        let static_loc = locations[value].is_static();
        let reg_unhomed = matches!(locations[value], Location::Reg(_) | Location::Float(_)) && homed.get(value) != Some(&true);
        if !present[value] || elided.get(value) == Some(&true) || static_loc || reg_unhomed {
            continue;
        }
        let Some(home) = shared.get(value).copied().flatten().map(|home| home as usize) else {
            cursor += round(ty.bytes(), WORD);
            values[value] = -(cursor as i32);
            continue;
        };
        if homes.len() <= home {
            homes.resize(home + 1, 0);
        }
        if homes[home] == 0 {
            cursor += round(ty.bytes(), WORD);
            homes[home] = -(cursor as i32);
        }
        values[value] = homes[home];
    }
    let mut max_saved = 0_usize;
    let mut check_edge = |block: Block, args: &[Value]| {
        for step in edge_steps(func, locations, &values, args, &func.blocks[block.index()].params) {
            if let Step::Save(_, EdgeScratch::Stack(index)) = step {
                max_saved = max_saved.max(index + 1);
            }
        }
    };
    for op in live_ops(func) {
        terminator_edges(op, &mut check_edge);
    }
    let wide = func.types.iter().any(|ty| ty.is_wide_vec());
    let scratch = max_saved as u32 * edge_bytes(wide);
    let mut slots = vec![0; func.slots.len()];
    let mut order: Vec<usize> = (0..func.slots.len()).filter(|&slot| addressed[slot]).collect();
    order.sort_by_key(|&slot| (func.slots[slot].size, slot));
    for slot in order {
        let data = &func.slots[slot];
        let min_align = if data.size >= 8 { 8 } else if data.size >= 4 { 4 } else if data.size >= 2 { 2 } else { 1 };
        let align = (1 << data.align_shift.min(MAX_ALIGN_SHIFT)).max(min_align);
        cursor = round(cursor + data.size, align);
        slots[slot] = -(cursor as i32);
    }
    cursor = round(cursor + scratch, 16);
    let scratch = -(cursor as i32);
    let size = round(cursor + outgoing, 16);
    if crate::trace::enabled() {
        crate::trace::on_frame(func, (&values, &slots), size);
    }
    Ok(Frame { values, slots, scratch, size, saves, float_saves, wide })
}

pub(crate) struct Prepared<R> {
    pub order: Vec<Block>,
    pub copied: Vec<bool>,
    pub abi: Abi<R>,
    pub plan: Plan<R>,
    pub frame: Frame<R>,
    pub framed: bool,
}

fn build_frame<R: Copy + PartialEq>(
    func: &Function,
    (saves, float_saves): (Vec<R>, Vec<u8>),
    plan: &Plan<R>,
    abi: &Abi<R>,
    classify: &impl Fn(&Signature) -> Result<Abi<R>, String>,
) -> Result<(Frame<R>, bool), String> {
    let frame = layout(func, (saves, float_saves), (&plan.locations, &plan.shared, &plan.homed, &plan.elided, &plan.fused), classify)?;
    let framed = framed(func, &frame, stack_params(abi));
    Ok((frame, framed))
}

pub(crate) fn repack_after_elide<R: Copy + PartialEq>(
    func: &Function,
    plan: &Plan<R>,
    abi: &Abi<R>,
    frame: &mut Frame<R>,
    framed: &mut bool,
    classify: impl Fn(&Signature) -> Result<Abi<R>, String>,
) -> Result<(), String> {
    let saves = std::mem::take(&mut frame.saves);
    let float_saves = std::mem::take(&mut frame.float_saves);
    (*frame, *framed) = build_frame(func, (saves, float_saves), plan, abi, &classify)?;
    Ok(())
}

/// Prepares `func`; when a loop reads a value that spilled, splits it at that loop's header, keeps the splits whose parameter then gets a register, and returns the split copy with its preparation.
pub(crate) fn prepare_split<R: Copy + PartialEq>(
    func: &Function,
    regs: &Regs<R>,
    weighed: bool,
    classify: impl Fn(&Signature) -> Result<Abi<R>, String>,
) -> Result<(Option<Function>, Prepared<R>), String> {
    let first = prepare(func, regs, weighed, &classify, &[])?;
    if !weighed {
        return Ok((None, first));
    }
    let plan = &first.plan;
    let carried = |made: &[(Value, Value, Block)]| -> Vec<(Value, u64)> { made.iter().map(|&(param, origin, _)| (param, plan.density.get(origin.index()).copied().unwrap_or(0))).collect() };
    let mut split = func.clone();
    let made = crate::loopargs::run(&mut split, |_, v| v.index() < plan.locations.len() && matches!(plan.location(v), Location::Slot) && !plan.elided[v.index()] && !plan.fused[v.index()]);
    if made.is_empty() {
        return Ok((None, first));
    }
    let second = prepare(&split, regs, weighed, &classify, &carried(&made))?;
    let kept: Vec<(Block, Value)> = made.iter().filter(|&&(param, _, _)| matches!(second.plan.location(param), Location::Reg(_) | Location::Float(_))).map(|&(_, origin, head)| (head, origin)).collect();
    if crate::trace::enabled() {
        crate::trace::on_split(&made, &kept);
    }
    if kept.len() == made.len() {
        return Ok((Some(split), second));
    }
    if kept.is_empty() {
        return Ok((None, first));
    }
    let mut pruned = func.clone();
    let made = crate::loopargs::run(&mut pruned, |head, v| kept.contains(&(head, v)));
    let third = prepare(&pruned, regs, weighed, &classify, &carried(&made))?;
    Ok((Some(pruned), third))
}

pub(crate) fn prepare<R: Copy + PartialEq>(
    func: &Function,
    regs: &Regs<R>,
    weighed: bool,
    classify: impl Fn(&Signature) -> Result<Abi<R>, String>,
    split: &[(Value, u64)],
) -> Result<Prepared<R>, String> {
    let order = crate::live::order(func);
    let copied = crate::live::copied_tests(func, &order);
    let abi = classify(&func.signature)?;
    let plan = crate::alloc::plan(func, regs, (&order, &copied, weighed), (&abi.params, split), &classify);
    let saves = saved(func, &plan.used, regs.saved);
    let float_saves = saved(func, &plan.used_floats, regs.saved_float);
    let (frame, framed) = build_frame(func, (saves, float_saves), &plan, &abi, &classify)?;
    Ok(Prepared { order, copied, abi, plan, frame, framed })
}

#[derive(Clone, Copy)]
pub(crate) struct ShrinkWrap {
    pub fast_ret: Block,
    pub slow_entry: Block,
    pub use_slow_thunk: bool,
}

fn pure_value<R: Copy + PartialEq>(value: Value, plan: &Plan<R>, frame: &Frame<R>, clobbers: &[R]) -> bool {
    if plan.elided[value.index()] || plan.fused[value.index()] {
        return true;
    }
    match plan.location(value) {
        Location::Imm(_) => true,
        Location::Reg(r) => !frame.saves.contains(&r) && !clobbers.contains(&r),
        Location::Float(f) => !frame.float_saves.contains(&f),
        Location::Slot | Location::SlotAddr(_) | Location::Symbol(_) | Location::Frame(..) => false,
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
    let cfg = func.cfg();
    if cfg.pred_count(entry) != 0 || !pure_block(func, entry, plan, frame, clobbers) {
        return None;
    }
    let term = *func.blocks[entry.index()].insts.last()?;
    let Op::Brif(cond, then_b, then_args, else_b, else_args) = &func.insts[term.index()].op else { return None };
    if *then_b == *else_b || !pure_value(*cond, plan, frame, clobbers) {
        return None;
    }
    let fast_ret_ok = |block: Block, edge_args: &[Value]| {
        cfg.pred_count(block) == 1
            && func.jump_target(block) == block
            && edge_args.iter().all(|&a| pure_value(a, plan, frame, clobbers))
            && func.blocks[block.index()].params.iter().all(|&p| pure_value(p, plan, frame, clobbers))
            && !edge_steps(func, &plan.locations, &frame.values, edge_args, &func.blocks[block.index()].params)
                .iter()
                .any(|s| matches!(s, Step::Save(_, EdgeScratch::Stack(_)) | Step::Restore(EdgeScratch::Stack(_), _)))
            && pure_block(func, block, plan, frame, clobbers)
            && func.blocks[block.index()].insts.last().is_some_and(|&t| match &func.insts[t.index()].op {
                Op::Return(vals) => vals.iter().all(|&v| pure_value(v, plan, frame, clobbers)),
                _ => false,
            })
    };
    let slow_ok = |block: Block, edge_args: &[Value]| {
        edge_args.iter().all(|&a| pure_value(a, plan, frame, clobbers))
            .then(|| cfg.pred_count(block) != 1 || func.jump_target(block) != block)
    };
    if fast_ret_ok(*then_b, then_args)
        && let Some(use_slow_thunk) = slow_ok(*else_b, else_args)
    {
        Some(ShrinkWrap { fast_ret: *then_b, slow_entry: *else_b, use_slow_thunk })
    } else if fast_ret_ok(*else_b, else_args)
        && let Some(use_slow_thunk) = slow_ok(*then_b, then_args)
    {
        Some(ShrinkWrap { fast_ret: *else_b, slow_entry: *then_b, use_slow_thunk })
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
    fn pos(&self) -> u32;
    /// The code offsets where a new source place starts, with that place, in emission order.
    fn lines(&mut self) -> &mut Vec<(u32, u64)>;
    /// Finishes the code and moves `starts`, offsets taken while emitting, to where they land in it.
    fn finish(self, starts: &mut [u32]) -> Result<crate::Code, String>;

    fn bind_entry(&mut self, block: Block) {
        self.bind_block(block);
    }

    fn deferred_prologue(&self) -> bool {
        false
    }

    fn blocks_fallthrough_to(&self, _block: Block) -> bool {
        false
    }

    fn before_block_bind(&mut self, _block: Block) -> Result<(), String> {
        Ok(())
    }

    fn on_block_bound(&mut self, _block: Block) -> Result<(), String> {
        Ok(())
    }
}

pub(crate) fn emit_body<'a, E: Emitter<'a>>(emit: &mut E, block: Block) -> Result<(), String> {
    emit.clear_flags();
    let func = emit.func();
    let insts = &func.blocks[block.index()].insts;
    let (mut ahead, mut loc) = (0, 0);
    for (k, &inst) in insts.iter().enumerate() {
        let data = &func.insts[inst.index()];
        if !matches!(data.op, Op::Call { .. }) && data.result.is_some_and(|res| emit.is_elided(res)) {
            continue;
        }
        if ahead <= k {
            ahead = insts[k..].iter().position(|i| func.insts[i.index()].loc != 0).map_or(insts.len(), |at| k + at);
            loc = insts.get(ahead).map_or(loc, |i| func.insts[i.index()].loc);
        }
        mark_line(emit, loc);
        emit.inst(inst)?;
    }
    if !insts.last().is_some_and(|inst| func.insts[inst.index()].op.is_terminator()) {
        emit.trap();
    }
    Ok(())
}

/// Starts a line-table row for `loc` at the current offset unless the open row already has it; 0 marks code with no source.
fn mark_line<'a, E: Emitter<'a>>(emit: &mut E, loc: u64) {
    let at = emit.pos();
    let lines = emit.lines();
    match lines.last_mut() {
        Some(last) if last.1 == loc => {}
        Some(last) if last.0 == at => last.1 = loc,
        _ => lines.push((at, loc)),
    }
}

fn entry_has_preds(func: &Function, order: &[Block], entry: Block) -> bool {
    order.iter().any(|&b| func.successors(b).any(|s| func.jump_target(s) == entry))
}

pub(crate) fn drive<'a, E: Emitter<'a>>(mut emit: E, order: &[Block]) -> Result<crate::Code, String> {
    let func = emit.func();
    let entry = func.entry.ok_or("the function has no entry block")?;
    let mut starts = Vec::with_capacity(order.len() + 1);
    mark_line(&mut emit, 0);
    if !emit.deferred_prologue() {
        emit.prologue();
    }
    emit.intake(entry)?;
    let fallthrough_entry = !entry_has_preds(func, order, entry);
    for (index, &block) in order.iter().enumerate() {
        let next_blk = order.get(index + 1).copied().filter(|&nb| !emit.blocks_fallthrough_to(nb));
        emit.set_next(next_blk);
        emit.before_block_bind(block)?;
        starts.push(emit.pos());
        if index == 0 && block == entry && fallthrough_entry {
            emit.bind_entry(block);
        } else {
            emit.bind_block(block);
        }
        emit.on_block_bound(block)?;
        emit_body(&mut emit, block)?;
    }
    starts.push(emit.pos());
    let blocks = starts.len();
    let lines = std::mem::take(emit.lines());
    starts.extend(lines.iter().map(|&(at, _)| at));
    let mut code = emit.finish(&mut starts)?;
    code.lines = lines.iter().zip(&starts[blocks..]).map(|(&(_, loc), &at)| (at, loc)).collect();
    starts.truncate(blocks);
    code.loops = crate::emit::loop_spans(func, order, &starts);
    Ok(code)
}

mod emit;
mod shrink;
pub(crate) use emit::{emit_body, stack_zero_store};
pub(crate) use shrink::{ShrinkWrap, shrink_wrap};
use emit::mark_line;

use crate::alloc::{Location, Plan, Regs};
use crate::ir::{Block, Function, Inst, Op, Signature, Type, Value};

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

impl<R: Copy + PartialEq> Abi<R> {
    /// Whether placing `self.params` writes register `reg`.
    pub(crate) fn clobbers(&self, reg: R) -> bool {
        self.params.iter().any(|loc| matches!(*loc, Loc::Int(p) if p == reg) || matches!(*loc, Loc::Pair(a, b) if a == reg || b == reg))
    }
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
        || !func.unwind_drops.is_empty()
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
    for u in &func.unwind_drops {
        slots[u.local.index()] = true;
        slots[u.flag.index()] = true;
    }
    flags.into_iter().for_each(|value| values[value.index()] = false);
    (values, slots)
}

/// Resolves function-level `UnwindDrop` slots into frame-pointer-relative `CodeUnwindDrop` entries.
pub(crate) fn resolve_unwind_drops<R>(func: &Function, frame: &Frame<R>) -> Vec<crate::CodeUnwindDrop> {
    func.unwind_drops
        .iter()
        .map(|u| crate::CodeUnwindDrop {
            local_off: frame.slots[u.local.index()] + u.local_off,
            flag_off: frame.slots[u.flag.index()],
            drop_sym: u.drop_sym,
            indirect: u.indirect,
        })
        .collect()
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

#[derive(Clone, Copy)]
enum HomeTarget {
    Value(usize),
    Shared(usize),
}

fn place_slots(func: &Function, addressed: &[bool], descending: bool, cursor: &mut u32) -> Vec<i32> {
    let mut slots = vec![0; func.slots.len()];
    let mut order: Vec<usize> = (0..func.slots.len()).filter(|&slot| addressed[slot]).collect();
    if descending {
        order.sort_by_key(|&slot| (std::cmp::Reverse(func.slots[slot].size), slot));
    } else {
        order.sort_by_key(|&slot| (func.slots[slot].size, slot));
    }
    for slot in order {
        let data = &func.slots[slot];
        let min_align = if data.size >= 8 { 8 } else if data.size >= 4 { 4 } else if data.size >= 2 { 2 } else { 1 };
        let align = (1 << data.align_shift.min(MAX_ALIGN_SHIFT)).max(min_align);
        *cursor = round(*cursor + data.size, align);
        slots[slot] = -(*cursor as i32);
    }
    slots
}

fn collect_value_homes<R: Copy + PartialEq>(
    func: &Function,
    present: &[bool],
    (locations, shared, homed, elided): (&[Location<R>], &[Option<u32>], &[bool], &[bool]),
) -> (Vec<(HomeTarget, u32, u32)>, usize, Vec<(usize, usize)>) {
    let mut order: Vec<usize> = (0..func.types.len()).collect();
    order.sort_by_key(|&v| (
        !matches!((homed.get(v), locations[v]), (Some(&true), Location::Reg(_))),
        !matches!((homed.get(v), locations[v], func.types[v].bytes()), (Some(&true), Location::Float(_), 8)),
        v,
    ));
    let mut items: Vec<(HomeTarget, u32, u32)> = Vec::new();
    let mut home_pos: Vec<Option<usize>> = Vec::new();
    let mut shared_vals: Vec<(usize, usize)> = Vec::new();
    for v in order {
        let reg_unhomed = matches!(locations[v], Location::Reg(_) | Location::Float(_)) && homed.get(v) != Some(&true);
        if !present[v] || elided.get(v) == Some(&true) || locations[v].is_static() || reg_unhomed {
            continue;
        }
        let bytes = func.types[v].bytes();
        let align = if bytes >= 16 { 16 } else { WORD };
        let Some(home) = shared.get(v).copied().flatten().map(|h| h as usize) else {
            items.push((HomeTarget::Value(v), bytes, align));
            continue;
        };
        if home_pos.len() <= home {
            home_pos.resize(home + 1, None);
        }
        if let Some(at) = home_pos[home] {
            items[at] = (items[at].0, items[at].1.max(bytes), items[at].2.max(align));
        } else {
            home_pos[home] = Some(items.len());
            items.push((HomeTarget::Shared(home), bytes, align));
        }
        shared_vals.push((v, home));
    }
    (items, home_pos.len(), shared_vals)
}

fn place_values<R: Copy + PartialEq>(
    func: &Function,
    present: &[bool],
    (locations, shared, homed, elided): (&[Location<R>], &[Option<u32>], &[bool], &[bool]),
    sp_near: bool,
    cursor: &mut u32,
) -> Vec<i32> {
    let (mut items, home_len, shared_vals) = collect_value_homes(func, present, (locations, shared, homed, elided));
    if sp_near {
        items.reverse();
    }
    let mut values = vec![0; func.types.len()];
    let mut homes = vec![0; home_len];
    for (target, bytes, align) in items {
        *cursor = round(*cursor, align) + round(bytes, align);
        let disp = -(*cursor as i32);
        match target {
            HomeTarget::Value(v) => values[v] = disp,
            HomeTarget::Shared(h) => homes[h] = disp,
        }
    }
    for (v, h) in shared_vals {
        values[v] = homes[h];
    }
    values
}

pub(crate) fn layout<R: Copy + PartialEq>(
    func: &Function,
    (saves, float_saves): (Vec<R>, Vec<u8>),
    (locations, shared, homed, elided, fused): (&[Location<R>], &[Option<u32>], &[bool], &[bool], &[bool]),
    classify: &impl Fn(&Signature) -> Result<Abi<R>, String>,
) -> Result<Frame<R>, String> {
    let outgoing = needs(func, classify)?;
    let (present, addressed) = present(func, fused);
    let sp_near = !live_ops(func).any(|op| matches!(op, Op::TryCall { .. } | Op::FramePointer));
    let mut cursor = (saves.len() + float_saves.len()) as u32 * WORD;
    let (slots, values) = if sp_near {
        let s = place_slots(func, &addressed, true, &mut cursor);
        let v = place_values(func, &present, (locations, shared, homed, elided), true, &mut cursor);
        (s, v)
    } else {
        let v = place_values(func, &present, (locations, shared, homed, elided), false, &mut cursor);
        let s = place_slots(func, &addressed, false, &mut cursor);
        (s, v)
    };
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

/// Prepares `func`; when a loop reads a value that spilled, splits it at that loop's header, keeps the splits whose parameter then gets a register, and returns the split copy with its preparation when it leaves less weighted spill traffic than the unsplit one.
pub(crate) fn prepare_split<R: Copy + PartialEq>(
    func: &Function,
    regs: &Regs<R>,
    (weighed, preserve_tc): (bool, bool),
    classify: impl Fn(&Signature) -> Result<Abi<R>, String>,
) -> Result<(Option<Function>, Prepared<R>), String> {
    let first = prepare(func, regs, (weighed, preserve_tc), &classify, &[])?;
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
    let second = prepare(&split, regs, (weighed, preserve_tc), &classify, &carried(&made))?;
    let kept: Vec<(Block, Value)> = made.iter().filter(|&&(param, _, _)| matches!(second.plan.location(param), Location::Reg(_) | Location::Float(_))).map(|&(_, origin, head)| (head, origin)).collect();
    if crate::trace::enabled() {
        crate::trace::on_split(&made, &kept);
    }
    if kept.len() == made.len() {
        return Ok(if second.plan.spill_cost < first.plan.spill_cost { (Some(split), second) } else { (None, first) });
    }
    if kept.is_empty() {
        return Ok((None, first));
    }
    let mut pruned = func.clone();
    let made = crate::loopargs::run(&mut pruned, |head, v| kept.contains(&(head, v)));
    let third = prepare(&pruned, regs, (weighed, preserve_tc), &classify, &carried(&made))?;
    Ok(if third.plan.spill_cost < first.plan.spill_cost { (Some(pruned), third) } else { (None, first) })
}

pub(crate) fn prepare<R: Copy + PartialEq>(
    func: &Function,
    regs: &Regs<R>,
    (weighed, preserve_tc): (bool, bool),
    classify: impl Fn(&Signature) -> Result<Abi<R>, String>,
    split: &[(Value, u64)],
) -> Result<Prepared<R>, String> {
    let order = if weighed { crate::live::order(func) } else { crate::live::quick_order(func) };
    let copied = if weighed { crate::live::copied_tests(func, &order) } else { vec![false; func.insts.len()] };
    let abi = classify(&func.signature)?;
    let plan = crate::alloc::plan(func, regs, (&order, &copied, weighed, preserve_tc), (&abi.params, split), &classify);
    let saves = saved(func, &plan.used, regs.saved);
    let float_saves = saved(func, &plan.used_floats, regs.saved_float);
    let (frame, framed) = build_frame(func, (saves, float_saves), &plan, &abi, &classify)?;
    Ok(Prepared { order, copied, abi, plan, frame, framed })
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

    fn before_block_bind(&mut self, _block: Block) -> Result<(), String> {
        Ok(())
    }

    fn on_block_bound(&mut self, _block: Block) -> Result<(), String> {
        Ok(())
    }

    fn zero_stack_store(&self, _inst: Inst) -> Option<(i32, u32)> {
        None
    }

    fn emit_zero_frame(&mut self, _offset: i32, _bytes: u32) {}
}

fn entry_has_preds(func: &Function, order: &[Block], entry: Block) -> bool {
    order.iter().any(|&b| func.successors(b).any(|s| func.jump_target(s) == entry))
}

pub(crate) fn drive<'a, E: Emitter<'a>>(mut emit: E, order: &[Block]) -> Result<crate::Code, String> {
    let func = emit.func();
    let entry = func.entry.ok_or("the function has no entry block")?;
    let mut starts = Vec::with_capacity(order.len() + 1);
    mark_line(&mut emit, func.home_loc());
    if !emit.deferred_prologue() {
        emit.prologue();
    }
    emit.intake(entry)?;
    let fallthrough_entry = !entry_has_preds(func, order, entry);
    for (index, &block) in order.iter().enumerate() {
        emit.set_next(order.get(index + 1).copied());
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

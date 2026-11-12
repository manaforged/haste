//! Register planning shared by the encoders: static locations for constants and frame addresses, and registers for integer and float values over their live spans.

use crate::frame::{Abi, Loc};
use crate::live::{Live, live, results};
use crate::fold::{constant, holds};
use crate::ir::{ArgumentPurpose, Binary, Block, Convert, FloatCC, Function, Inst, Op, Signature, StackSlot, Type, Value};

/// The registers of one encoder that planning hands out.
pub(crate) struct Regs<R: 'static> {
    /// Registers that hold integer values across calls; the emitter never uses them as scratch, and a frame saves each one it uses.
    pub saved: &'static [R],
    /// Caller-saved registers the emitter never uses as scratch, which values that span no call live in without a save.
    pub leaf: &'static [R],
    /// Leaf registers that 128-bit operations use as scratch, so a function with one keeps no value in them.
    pub wide_scratch: &'static [R],
    /// Leaf registers that the operations `uses_scratch` picks use as scratch, so a function with one keeps no value in them.
    pub op_scratch: &'static [R],
    /// Whether an operation of a function uses `op_scratch`.
    pub uses_scratch: fn(&Function, &Op) -> bool,
    /// Leaf register that carries the dictionary parameter on entry, when it is in `leaf`.
    pub dict: Option<R>,
    /// Caller-saved float registers, by number, the emitter never uses as scratch, which float values that span no call live in.
    pub float: &'static [u8],
    /// The float compares whose result the next branch reads from the flags under one condition.
    pub float_branches: &'static [FloatCC],
    /// Whether a call can read its callee from a constant address below 2 GiB in place.
    pub absolute_calls: bool,
    /// Whether the encoder computes an add, subtract, multiply or bitwise operation in the register of its first operand, so the result may take that register where the operand dies.
    pub two_address: bool,
    /// Whether a value that spans calls may live in a caller-saved register that the emitter stores before and reloads after each call it spans.
    pub park: bool,
}

const FOLD_LIMIT: i64 = 1 << 24;

/// The `uses_scratch` of an encoder whose operations use no leaf register as scratch.
pub(crate) fn no_scratch(_: &Function, _: &Op) -> bool {
    false
}

/// Where a value lives.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Location<R> {
    Slot,
    Imm(i64),
    SlotAddr(u32),
    Reg(R),
    Float(u8),
    Frame(StackSlot, i32),
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum CallTarget {
    Abs(i32),
    Slot(u32),
}

/// What the emitter may do with each value, decided before emission.
pub(crate) struct Plan<R> {
    pub locations: Vec<Location<R>>,
    pub fused: Vec<bool>,
    pub used: Vec<R>,
    pub pairs: Vec<Option<Value>>,
    pub deferred: Vec<bool>,
    pub through: Vec<Option<CallTarget>>,
    pub shared: Vec<Option<u32>>,
    pub parked: Vec<(Inst, Vec<(Value, R, bool)>)>,
    pub homed: Vec<bool>,
}

/// Whether planning may give a value of type `ty` a register of `regs`.
fn registered<R>(regs: &Regs<R>, ty: Type) -> bool {
    ty.is_small_int() || (ty.in_float_register() && !regs.float.is_empty())
}

type Position = Option<(usize, usize)>;

struct Uses {
    uses: Vec<u32>,
    last_use: Vec<Position>,
    def: Vec<Position>,
}

impl Uses {
    fn visit(&mut self, data: &crate::ir::InstData, at: (usize, usize)) {
        data.op.for_operands(|value| {
            self.uses[value.index()] += 1;
            self.last_use[value.index()] = Some(at);
        });
        for result in results(data) {
            self.def[result.index()] = Some(at);
        }
    }
}

fn census(func: &Function) -> Uses {
    let count = func.types.len();
    let mut census = Uses { uses: vec![0; count], last_use: vec![None; count], def: vec![None; count] };
    for (b, block) in func.blocks.iter().enumerate() {
        for (i, inst) in block.insts.iter().enumerate() {
            census.visit(&func.insts[inst.index()], (b, i));
        }
    }
    census
}

fn consts<R>(func: &Function, absolute_calls: bool, locations: &mut [Location<R>]) {
    for data in &func.insts {
        let Some(result) = data.result else { continue };
        match data.op {
            Op::Const(imm) if func.ty(result).is_small_int() => locations[result.index()] = Location::Imm(imm),
            Op::SlotAddr(sym) if absolute_calls => locations[result.index()] = Location::SlotAddr(sym),
            _ => {}
        }
    }
}

fn fold<R>(slot: StackSlot, offset: i32, imm: i64) -> Option<Location<R>> {
    let sum = i64::from(offset).checked_add(imm)?;
    if !(-FOLD_LIMIT..=FOLD_LIMIT).contains(&sum) {
        return None;
    }
    Some(Location::Frame(slot, sum as i32))
}

fn static_location<R: Copy>(func: &Function, locations: &[Location<R>], op: &Op) -> Option<Location<R>> {
    match *op {
        Op::StackAddr(slot, offset) => Some(Location::Frame(slot, offset)),
        Op::Icmp(cc, x, y) => match (locations[x.index()], locations[y.index()]) {
            (Location::Imm(a), Location::Imm(b)) if func.ty(x).is_small_int() => Some(Location::Imm(i64::from(holds(cc, func.ty(x), a, b)))),
            _ => None,
        },
        Op::Binary(op, x, y) => match (locations[x.index()], locations[y.index()]) {
            (Location::Frame(slot, offset), Location::Imm(imm)) | (Location::Imm(imm), Location::Frame(slot, offset)) if op == Binary::Iadd => fold(slot, offset, imm),
            (Location::Imm(a), Location::Imm(b)) if func.ty(x).is_small_int() => constant(op, a, b).map(Location::Imm),
            _ => None,
        },
        Op::Convert(Convert::Uextend | Convert::Ireduce, x) => match locations[x.index()] {
            Location::Imm(imm) if func.ty(x).is_small_int() => Some(Location::Imm(crate::fold::extend(imm, func.ty(x), false))),
            _ => None,
        },
        _ => None,
    }
}

/// Gives frame addresses and compares of constants static locations, until no more change.
fn statics<R: Copy>(func: &Function, locations: &mut [Location<R>]) {
    let mut changed = true;
    while changed {
        changed = false;
        for data in &func.insts {
            let Some(result) = data.result else { continue };
            if !matches!(locations[result.index()], Location::Slot) || !func.ty(result).is_small_int() {
                continue;
            }
            if let Some(location) = static_location(func, locations, &data.op).filter(|location| func.ty(result) == Type::I64 || matches!(location, Location::Imm(_))) {
                locations[result.index()] = location;
                changed = true;
            }
        }
    }
}

fn fuses(func: &Function, census: &Uses, value: usize, float_branches: &[FloatCC]) -> bool {
    let (Some((block, at)), Some((_, last))) = (census.def[value], census.last_use[value]) else { return false };
    let insts = &func.blocks[block].insts;
    let op = &func.insts[insts[at].index()].op;
    let next = insts.get(at + 1).map(|inst| &func.insts[inst.index()].op);
    let flags = match op {
        Op::Icmp(..) | Op::Fits(..) => true,
        Op::Fcmp(cc, ..) => float_branches.contains(cc),
        _ => false,
    };
    census.uses[value] == 1
        && last == at + 1
        && flags
        && matches!(next, Some(Op::Brif(condition, ..) | Op::Trapz(condition)) if condition.index() == value)
}

/// For each overflow check that directly follows its own add, subtract or multiply, that operation's result, so one instruction computes both; and which results wait for their check.
fn pairs<R>(func: &Function, locations: &[Location<R>]) -> (Vec<Option<Value>>, Vec<bool>) {
    let (mut pairs, mut deferred) = (vec![None; func.types.len()], vec![false; func.types.len()]);
    let dynamic = |value: Value| matches!(locations[value.index()], Location::Slot | Location::Reg(_));
    for block in &func.blocks {
        for window in block.insts.windows(2) {
            let (first, second) = (&func.insts[window[0].index()], &func.insts[window[1].index()]);
            if let (Op::Binary(op, x, y), Op::Fits(check, _, a, b), Some(sum), Some(fits)) = (&first.op, &second.op, first.result, second.result)
                && op == check
                && (x, y) == (a, b)
                && func.ty(sum).is_small_int()
                && dynamic(sum)
                && dynamic(fits)
            {
                pairs[fits.index()] = Some(sum);
                deferred[sum.index()] = true;
            }
        }
    }
    (pairs, deferred)
}

/// For each callee that only one call reads, loaded from a constant address below 2 GiB, that address, so the call reads it in place.
fn through<R>(func: &Function, locations: &[Location<R>], census: &Uses) -> Vec<Option<CallTarget>> {
    let mut through = vec![None; func.types.len()];
    let mut loads = vec![None; func.types.len()];
    for data in &func.insts {
        if let (Op::Load(addr, 0), Some(result)) = (&data.op, data.result) {
            loads[result.index()] = match locations[addr.index()] {
                Location::Imm(at) => i32::try_from(at).ok().filter(|at| *at > 0).map(CallTarget::Abs),
                Location::SlotAddr(sym) => Some(CallTarget::Slot(sym)),
                _ => None,
            };
        }
    }
    for inst in func.blocks.iter().flat_map(|block| &block.insts) {
        if let Op::Call { callee, args, .. } | Op::TryCall { callee, args, .. } = &func.insts[inst.index()].op
            && census.uses[callee.index()] == 1
            && !args.contains(callee)
        {
            through[callee.index()] = loads[callee.index()];
        }
    }
    through
}

fn eligible(live: &Live, wanted: &[bool], value: usize) -> bool {
    let Some((start, end)) = live.span[value] else { return false };
    wanted[value] && !live.unwinds.iter().any(|at| start <= *at && *at < end)
}

/// One register class: the registers that keep values across calls, the caller-saved ones planning may use, the registers of this class each parameter arrives in, and the register a call takes each value in.
struct Class<'a, F> {
    saved: &'a [F],
    leaf: Vec<F>,
    arrivals: Vec<Vec<F>>,
    passed: Vec<Option<F>>,
}

/// For each value some call passes in an integer register, that register.
fn passed<R: Copy>(func: &Function, classify: &impl Fn(&Signature) -> Result<Abi<R>, String>) -> Vec<Option<R>> {
    let mut passed = vec![None; func.types.len()];
    for inst in func.blocks.iter().flat_map(|block| &block.insts) {
        if let Op::Call { sig, args, .. } | Op::TryCall { sig, args, .. } = &func.insts[inst.index()].op
            && let Ok(abi) = classify(&func.sigs[sig.index()])
        {
            for (arg, loc) in args.iter().zip(&abi.params) {
                if let Loc::Int(reg) = *loc {
                    passed[arg.index()] = Some(reg);
                }
            }
        }
    }
    passed
}

/// The integer class; a function with a 128-bit value keeps none in the registers 128-bit operations use as scratch.
fn int_class<'a, R: Copy + PartialEq>(func: &Function, regs: &'a Regs<R>, order: &[Block], params: &[Loc<R>], passed: Vec<Option<R>>) -> Class<'a, R> {
    let (mut wide, mut scratch) = (false, false);
    for data in order.iter().flat_map(|block| &func.blocks[block.index()].insts).map(|inst| &func.insts[inst.index()]) {
        wide |= data.result.is_some_and(|result| func.ty(result) == Type::I128);
        scratch |= (regs.uses_scratch)(func, &data.op);
    }
    let dict = func.signature.params.iter().any(|param| param.purpose == ArgumentPurpose::Dict);
    let kept = |reg: &R| !(wide && regs.wide_scratch.contains(reg)) && !(scratch && regs.op_scratch.contains(reg)) && !(dict && regs.dict == Some(*reg));
    let leaf = regs.leaf.iter().filter(|reg| kept(reg)).copied().collect();
    let arrivals = params
        .iter()
        .map(|loc| match *loc {
            Loc::Int(reg) => vec![reg],
            Loc::Pair(low, high) => vec![low, high],
            _ => Vec::new(),
        })
        .collect();
    Class { saved: regs.saved, leaf, arrivals, passed }
}

/// The float class, which has no register that keeps a value across a call.
fn float_class<R>(regs: &Regs<R>, params: &[Loc<R>]) -> Class<'static, u8> {
    let arrivals = params
        .iter()
        .map(|loc| match *loc {
            Loc::Float(reg) => vec![reg],
            _ => Vec::new(),
        })
        .collect();
    Class { saved: &[], leaf: regs.float.to_vec(), arrivals, passed: Vec::new() }
}

/// The registers values may live in, the entry parameters that keep the register they arrive in, the arrival registers no other value takes before the entry block, and the caller-saved register a call takes each value in.
struct Pool<R> {
    regs: Vec<R>,
    kept: Vec<Option<R>>,
    arrival: Vec<R>,
    passed: Vec<Option<R>>,
}

/// Whether the span `start..=end` holds a call after its start, which clobbers the caller-saved registers.
fn crosses(calls: &[u32], (start, end): (u32, u32)) -> bool {
    let next = calls.partition_point(|at| *at <= start);
    calls.get(next).is_some_and(|at| *at <= end)
}

/// The saved pool and the free caller-saved registers of `class`; values that span no call may take the caller-saved ones, and so may entry parameters that arrive in one.
fn pool<F: Copy + PartialEq>(func: &Function, class: &Class<F>, live: &Live, pooled: &[bool]) -> Pool<F> {
    let mut kept = vec![None; func.types.len()];
    let entry = func.entry.map_or(&[][..], |entry| &func.blocks[entry.index()].params[..]);
    for (value, arrival) in entry.iter().zip(&class.arrivals) {
        if let [reg] = arrival[..]
            && class.leaf.contains(&reg)
            && pooled[value.index()]
            && !live.trapped[value.index()]
            && live.span[value.index()].is_some_and(|span| !crosses(&live.calls, span))
        {
            kept[value.index()] = Some(reg);
        }
    }
    let arrival = class.arrivals.iter().flatten().copied().filter(|reg| class.leaf.contains(reg) && !kept.contains(&Some(*reg))).collect();
    let regs = class.saved.iter().chain(&class.leaf).copied().collect();
    let passed = class.passed.iter().map(|reg| reg.filter(|reg| class.leaf.contains(reg))).collect();
    Pool { regs, kept, arrival, passed }
}

/// Takes from `free` a caller-saved register when `saved` is false, else a saved one, preferring one the frame already saves.
fn take<R: Copy + PartialEq>(free: &mut Vec<R>, (saved, pool): (bool, &[R]), used: &[R]) -> Option<R> {
    let at = if saved {
        free.iter().rposition(|reg| used.contains(reg)).or_else(|| free.iter().rposition(|reg| pool.contains(reg)))
    } else {
        free.iter().rposition(|reg| !pool.contains(reg)).or_else(|| free.iter().rposition(|reg| pool.contains(reg)))
    };
    at.map(|at| free.swap_remove(at))
}

const ARRIVAL: usize = usize::MAX;

/// How much more one use inside a loop costs a spilled value than one outside it, per level of nesting.
const LOOP_WEIGHT: u64 = 8;

/// The deepest nesting that still raises a use's weight.
const MAX_DEPTH: u32 = 4;

/// For each block, the weight of one instruction in it: `LOOP_WEIGHT` for each loop around it.
fn block_weights(func: &Function) -> Vec<u64> {
    crate::licm::depths(func).into_iter().map(|depth| LOOP_WEIGHT.pow(depth.min(MAX_DEPTH))).collect()
}

/// For each value, the cost of keeping it in memory: the weight of its block for each definition and use.
fn weights(func: &Function, blocks: &[u64]) -> Vec<u64> {
    let mut weight = vec![0_u64; func.types.len()];
    for (block, data) in func.blocks.iter().enumerate() {
        let each = blocks[block];
        for inst in &data.insts {
            let data = &func.insts[inst.index()];
            data.op.for_operands(|value| weight[value.index()] += each);
            results(data).for_each(|value| weight[value.index()] += each);
        }
    }
    weight
}

/// What guides the choice of a register beyond the spans: the operand each result may compute in place over, and when planning weighs values, the spill cost of each, the values each block edge moves it to or from, and the weight of each call when values that span calls may park around them.
struct Hints<'a> {
    reuse: &'a [Option<Value>],
    weights: Option<&'a [u64]>,
    partners: Option<&'a [Vec<Value>]>,
    park: Option<&'a [u64]>,
}

/// What keeping a value in a caller-saved register over `start..=end` costs: a store and a reload around each call inside the span, and a store before a call that ends it.
fn park_cost(calls: &[u32], weights: &[u64], (start, end): (u32, u32)) -> u64 {
    let (first, last) = (calls.partition_point(|at| *at <= start), calls.partition_point(|at| *at <= end));
    (first..last).map(|at| weights[at] * if calls[at] < end { 2 } else { 1 }).sum()
}

/// Whether `value` may park in a caller-saved register around the calls its span holds, because none of them can unwind and that costs less than keeping it in memory.
fn parks(live: &Live, hints: &Hints, (start, end, value): (u32, u32, usize)) -> bool {
    let Some((calls, weights)) = hints.park.zip(hints.weights) else { return false };
    let unwinds = live.unwinds.partition_point(|at| *at <= start) < live.unwinds.partition_point(|at| *at <= end);
    !live.trapped[value] && !unwinds && park_cost(&live.calls, calls, (start, end)) < weights[value]
}

/// For each value a block edge passes, the parameters it becomes, and for each block parameter, the values its edges pass it.
fn partners(func: &Function) -> Vec<Vec<Value>> {
    let mut partners = vec![Vec::new(); func.types.len()];
    let mut pair = |params: &[Value], args: &[Value]| {
        for (param, arg) in params.iter().zip(args) {
            partners[param.index()].push(*arg);
            partners[arg.index()].push(*param);
        }
    };
    for inst in func.blocks.iter().flat_map(|block| &block.insts) {
        match &func.insts[inst.index()].op {
            Op::Jump(block, args) => pair(&func.blocks[block.index()].params, args),
            Op::Brif(_, then, then_args, otherwise, else_args) => {
                pair(&func.blocks[then.index()].params, then_args);
                pair(&func.blocks[otherwise.index()].params, else_args);
            }
            _ => {}
        }
    }
    partners
}

/// Takes from `free` the register of a value an edge moves `value` to or from, so the edge moves nothing there: a saved one when `value` spans a call, else one the frame already saves or a caller-saved one.
fn joined<F: Copy + PartialEq>(free: &mut Vec<F>, homes: &[Option<F>], partners: &[Value], (saved, saved_regs, used): (bool, &[F], &[F])) -> Option<F> {
    let suits = |reg: &F| if saved { saved_regs.contains(reg) } else { !saved_regs.contains(reg) || used.contains(reg) };
    let at = partners.iter().filter_map(|partner| homes[partner.index()]).find_map(|reg| free.iter().position(|free| *free == reg && suits(free)))?;
    Some(free.swap_remove(at))
}

/// The active entry whose register `value`, live until `end`, takes when none is free: without weights the one that ends last, when it outlives `value`; with weights the one with the least spill cost per instruction of its span, when that is less than `value`'s.
fn victim<F: Copy + PartialEq>(active: &[(u32, usize, F)], (end, value): (u32, usize), saved: Option<&[F]>, weights: Option<&[u64]>, spans: &[Option<(u32, u32)>]) -> Option<usize> {
    let candidates = active.iter().enumerate().filter(|(_, entry)| entry.1 != ARRIVAL && saved.is_none_or(|regs| regs.contains(&entry.2)));
    let Some(cost) = weights else {
        return candidates.max_by_key(|(_, entry)| entry.0).filter(|(_, entry)| entry.0 > end).map(|(at, _)| at);
    };
    let density = |value: usize| (cost[value], spans[value].map_or(1, |(start, end)| u64::from(end - start) + 1));
    let below = |(a, la): (u64, u64), (b, lb): (u64, u64)| u128::from(a) * u128::from(lb) < u128::from(b) * u128::from(la);
    let chosen = candidates.reduce(|best, next| if below(density(next.1.1), density(best.1.1)) { next } else { best });
    chosen.filter(|(_, entry)| below(density(entry.1), density(value))).map(|(at, _)| at)
}

/// Whether an encoder that computes in place reads the first operand of `op` once, into the result's register, before it reads the second.
fn in_place(op: Binary) -> bool {
    matches!(op, Binary::Iadd | Binary::Isub | Binary::Imul | Binary::Band | Binary::Bor | Binary::Bxor | Binary::Fadd | Binary::Fsub | Binary::Fmul | Binary::Fdiv)
}

/// For each result of an in-place operation whose first operand dies there, that operand; an overflow check right after its own operation reads the operands with it.
fn reusable(func: &Function, live: &Live) -> Vec<Option<Value>> {
    let mut reuse = vec![None; func.types.len()];
    for block in &func.blocks {
        for (at, inst) in block.insts.iter().enumerate() {
            let data = &func.insts[inst.index()];
            let (Op::Binary(op, x, y), Some(result)) = (&data.op, data.result) else { continue };
            let next = block.insts.get(at + 1).map(|next| &func.insts[next.index()].op);
            let checked = matches!(next, Some(Op::Fits(check, _, a, b)) if check == op && (a, b) == (x, y));
            let def = live.defs[result.index()];
            let dies = live.span[x.index()].is_some_and(|(_, end)| end == def + u32::from(checked));
            if in_place(*op) && dies && live.span[result.index()].is_some_and(|(start, _)| start == def) {
                reuse[result.index()] = Some(*x);
            }
        }
    }
    reuse
}

/// Takes the register of the operand `value` is computed over in place, when that operand dies there and the register suits `value`.
fn reused<R: Copy + PartialEq>(active: &mut Vec<(u32, usize, R)>, operand: Option<Value>, saved: bool, saved_regs: &[R]) -> Option<R> {
    let operand = operand?.index();
    let at = active.iter().position(|entry| entry.1 == operand && (!saved || saved_regs.contains(&entry.2)))?;
    Some(active.swap_remove(at).2)
}

/// Takes from `free` the register the call that ends the span `start..=end` reads `value` from, when that call cannot unwind and no earlier call clobbers it.
fn passed_in<R: Copy + PartialEq>(free: &mut Vec<R>, live: &Live, passed: &[Option<R>], (start, end, value): (u32, u32, usize)) -> Option<R> {
    let reg = passed.get(value).copied().flatten()?;
    let plain_call = live.calls.binary_search(&end).is_ok() && live.unwinds.binary_search(&end).is_err();
    if !plain_call || crosses(&live.calls, (start, end - 1)) {
        return None;
    }
    let at = free.iter().position(|free| *free == reg)?;
    Some(free.swap_remove(at))
}

/// Gives registers to values by linear scan over their spans, a saved one when the span holds a call, spilling the value that ends last when none is free; an entry parameter that keeps its arrival register takes it; a value takes the register of an operand that dies where it is computed in place, or of the call that reads it last, when it can.
fn assign<F: Copy + PartialEq>(live: &Live, pooled: &[bool], saved_regs: &[F], pool: Pool<F>, hints: &Hints) -> (Vec<Option<F>>, Vec<F>) {
    let mut homes = vec![None; pooled.len()];
    let mut values: Vec<(u32, u32, usize)> = (0..pooled.len()).filter(|value| pooled[*value]).filter_map(|value| live.span[value].map(|(start, end)| (start, end, value))).collect();
    values.sort_unstable();
    let kept: Vec<F> = pool.kept.iter().flatten().copied().collect();
    let mut free: Vec<F> = pool.regs.iter().filter(|reg| !pool.arrival.contains(reg) && !kept.contains(reg)).copied().collect();
    let mut active: Vec<(u32, usize, F)> = pool.arrival.iter().map(|reg| (0, ARRIVAL, *reg)).collect();
    let mut used = Vec::new();
    for (start, end, value) in values {
        active.retain(|&(until, _, reg)| {
            let ended = until < start;
            if ended {
                free.push(reg);
            }
            !ended
        });
        let saved = live.trapped[value] || crosses(&live.calls, (start, end));
        let partner = |free: &mut Vec<F>, homes: &[Option<F>]| hints.partners.and_then(|partners| joined(free, homes, &partners[value], (saved, saved_regs, &used)));
        let parked = saved && parks(live, hints, (start, end, value));
        let fresh = |free: &mut Vec<F>| {
            let any = passed_in(free, live, &pool.passed, (start, end, value)).or_else(|| take(free, (saved, saved_regs), &used));
            any.or_else(|| if parked { take(free, (false, saved_regs), &used) } else { None })
        };
        let reg = match pool.kept[value].or_else(|| reused(&mut active, hints.reuse[value], saved, saved_regs)).or_else(|| partner(&mut free, &homes)).or_else(|| fresh(&mut free)) {
            Some(reg) => reg,
            None => match victim(&active, (end, value), saved.then_some(saved_regs), hints.weights, &live.span) {
                Some(at) => {
                    let (_, other, reg) = active.swap_remove(at);
                    homes[other] = None;
                    reg
                }
                None => continue,
            },
        };
        homes[value] = Some(reg);
        active.push((end, value, reg));
        if saved_regs.contains(&reg) && !used.contains(&reg) {
            used.push(reg);
        }
    }
    (homes, used)
}

/// Registers of `class` for the `pooled` values, and the saved registers they use.
fn allocate<F: Copy + PartialEq>(func: &Function, live: &Live, class: &Class<F>, pooled: &[bool], hints: &Hints) -> (Vec<Option<F>>, Vec<F>) {
    assign(live, pooled, class.saved, pool(func, class, live, pooled), hints)
}

/// Writes each register in `homes` into the location of its value as `wrap` of it.
fn place<R, F>(locations: &mut [Location<R>], homes: Vec<Option<F>>, wrap: impl Fn(F) -> Location<R>) {
    for (location, home) in locations.iter_mut().zip(homes) {
        if let Some(reg) = home {
            *location = wrap(reg);
        }
    }
}

/// Whether the call `inst` takes `value` as an argument in `reg`, where it already lives.
fn passed_own<R: Copy + PartialEq>(func: &Function, inst: Inst, (value, reg): (Value, R), classify: &impl Fn(&Signature) -> Result<Abi<R>, String>) -> bool {
    let Op::Call { sig, args, .. } = &func.insts[inst.index()].op else { return false };
    let Ok(abi) = classify(&func.sigs[sig.index()]) else { return false };
    args.iter().zip(&abi.params).any(|(arg, loc)| *arg == value && matches!(*loc, Loc::Int(at) if at == reg))
}

/// For each call that a value in a caller-saved register spans, that value, its register and whether it lives on after the call, ordered by instruction; a value the call reads last from the register it takes it in stays.
fn parked<R: Copy + PartialEq>(func: &Function, live: &Live, (locations, saved): (&[Location<R>], &[R]), classify: &impl Fn(&Signature) -> Result<Abi<R>, String>) -> Vec<(Inst, Vec<(Value, R, bool)>)> {
    let mut at_call: Vec<Vec<(Value, R, bool)>> = vec![Vec::new(); live.calls.len()];
    for (index, location) in locations.iter().enumerate() {
        let (Location::Reg(reg), Some((start, end))) = (*location, live.span[index]) else { continue };
        if saved.contains(&reg) {
            continue;
        }
        let value = Value(index as u32);
        for call in live.calls.partition_point(|at| *at <= start)..live.calls.partition_point(|at| *at <= end) {
            let at = live.calls[call];
            if at < end || !passed_own(func, live.sites[call].0, (value, reg), classify) {
                at_call[call].push((value, reg, at < end));
            }
        }
    }
    let mut parked: Vec<(Inst, Vec<(Value, R, bool)>)> = live.sites.iter().map(|site| site.0).zip(at_call).filter(|(_, values)| !values.is_empty()).collect();
    parked.sort_unstable_by_key(|(inst, _)| inst.index());
    parked
}

/// Gives constants and frame addresses static locations, integers and floats pool registers over their live spans, and finds the compares that feed the next branch.
pub(crate) fn plan<R: Copy + PartialEq>(func: &Function, regs: &Regs<R>, (order, weighed): (&[Block], bool), params: &[Loc<R>], classify: impl Fn(&Signature) -> Result<Abi<R>, String>) -> Plan<R> {
    let count = func.types.len();
    let mut locations = vec![Location::Slot; count];
    consts(func, regs.absolute_calls, &mut locations);
    statics(func, &mut locations);
    let census = census(func);
    let wanted: Vec<bool> = (0..count).map(|value| matches!(locations[value], Location::Slot) && registered(regs, func.types[value])).collect();
    let tracked: Vec<bool> = (0..count).map(|value| wanted[value] || (weighed && matches!(locations[value], Location::Slot))).collect();
    let live = live(func, order, &tracked);
    let candidates: Vec<bool> = (0..count).map(|value| eligible(&live, &wanted, value)).collect();
    let fused: Vec<bool> = (0..count).map(|value| candidates[value] && fuses(func, &census, value, regs.float_branches)).collect();
    let through = if regs.absolute_calls { through(func, &locations, &census) } else { vec![None; count] };
    let ints: Vec<bool> = (0..count).map(|value| candidates[value] && func.types[value].is_small_int() && !fused[value] && through[value].is_none()).collect();
    let floats: Vec<bool> = (0..count).map(|value| candidates[value] && func.types[value].in_float_register() && !live.trapped[value] && live.span[value].is_some_and(|span| !crosses(&live.calls, span))).collect();
    let reuse = if regs.two_address { reusable(func, &live) } else { vec![None; count] };
    let blocks = weighed.then(|| block_weights(func));
    let weights = blocks.as_deref().map(|blocks| weights(func, blocks));
    let partners = weighed.then(|| partners(func));
    let park = blocks.as_deref().filter(|_| regs.park).map(|blocks| live.sites.iter().map(|(_, block)| blocks[block.index()]).collect::<Vec<u64>>());
    let hints = Hints { reuse: &reuse, weights: weights.as_deref(), partners: partners.as_deref(), park: park.as_deref() };
    let (homes, used) = allocate(func, &live, &int_class(func, regs, order, params, passed(func, &classify)), &ints, &hints);
    place(&mut locations, homes, Location::Reg);
    let parked = if park.is_some() { parked(func, &live, (&locations, regs.saved), &classify) } else { Vec::new() };
    let mut homed = vec![false; if parked.is_empty() { 0 } else { count }];
    parked.iter().flat_map(|(_, values)| values).for_each(|(value, _, _)| homed[value.index()] = true);
    let (homes, _) = allocate(func, &live, &float_class(regs, params), &floats, &hints);
    place(&mut locations, homes, Location::Float);
    let (pairs, deferred) = pairs(func, &locations);
    let shared = if weighed { crate::color::share(func, &live, &locations, &deferred) } else { Vec::new() };
    Plan { locations, fused, used, pairs, deferred, through, shared, parked, homed }
}

impl<R: Copy> Plan<R> {
    /// Where `value` lives.
    pub(crate) fn location(&self, value: Value) -> Location<R> {
        self.locations[value.index()]
    }

    /// Whether `value` needs no code where it is defined, because its location is a constant or a frame address.
    pub(crate) fn is_static(&self, value: Value) -> bool {
        matches!(self.location(value), Location::Imm(_) | Location::SlotAddr(_) | Location::Frame(..))
    }

    /// The constant `value` holds, when its location is one.
    pub(crate) fn const_of(&self, value: Value) -> Option<i64> {
        match self.location(value) {
            Location::Imm(imm) => Some(imm),
            _ => None,
        }
    }

    /// The register that holds `value`, when its location is one.
    pub(crate) fn reg_of(&self, value: Value) -> Option<R> {
        match self.location(value) {
            Location::Reg(reg) => Some(reg),
            _ => None,
        }
    }

    /// The values in caller-saved registers that the call `inst` spans, each with its register and whether it lives on after the call.
    pub(crate) fn parked_at(&self, inst: Inst) -> &[(Value, R, bool)] {
        self.parked.binary_search_by_key(&inst.index(), |(at, _)| at.index()).map_or(&[], |at| &self.parked[at].1)
    }

    /// The float register that holds `value`, when its location is one.
    pub(crate) fn float_of(&self, value: Value) -> Option<u8> {
        match self.location(value) {
            Location::Float(reg) => Some(reg),
            _ => None,
        }
    }
}

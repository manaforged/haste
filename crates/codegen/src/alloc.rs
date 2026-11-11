//! Register planning shared by the encoders: static locations for constants and frame addresses, and registers for integer and float values over their live spans.

use crate::frame::Loc;
use crate::live::{Live, live, results};
use crate::fold::{constant, holds};
use crate::ir::{Binary, Block, Convert, FloatCC, Function, Op, StackSlot, Type, Value};

/// The registers of one encoder that planning hands out.
pub(crate) struct Regs<R: 'static> {
    /// Registers that hold integer values across calls; the emitter never uses them as scratch, and a frame saves each one it uses.
    pub saved: &'static [R],
    /// Caller-saved registers the emitter never uses as scratch, which values that span no call live in without a save.
    pub leaf: &'static [R],
    /// Leaf registers that 128-bit operations use as scratch, so a function with one keeps no value in them.
    pub wide_scratch: &'static [R],
    /// Caller-saved float registers, by number, the emitter never uses as scratch, which float values that span no call live in.
    pub float: &'static [u8],
    /// The float compares whose result the next branch reads from the flags under one condition.
    pub float_branches: &'static [FloatCC],
    /// Whether a call can read its callee from a constant address below 2 GiB in place.
    pub absolute_calls: bool,
}

const FOLD_LIMIT: i64 = 1 << 24;

/// Where a value lives.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Location<R> {
    Slot,
    Imm(i64),
    Reg(R),
    Float(u8),
    Frame(StackSlot, i32),
}

/// What the emitter may do with each value, decided before emission.
pub(crate) struct Plan<R> {
    pub locations: Vec<Location<R>>,
    pub fused: Vec<bool>,
    pub used: Vec<R>,
    pub pairs: Vec<Option<Value>>,
    pub deferred: Vec<bool>,
    pub through: Vec<Option<i32>>,
}

pub(crate) fn small_int(ty: Type) -> bool {
    matches!(ty, Type::I8 | Type::I16 | Type::I32 | Type::I64)
}

/// Whether planning may give a value of type `ty` a register of `regs`.
fn registered<R>(regs: &Regs<R>, ty: Type) -> bool {
    small_int(ty) || (ty.is_float() && !regs.float.is_empty())
}

type Position = Option<(usize, usize)>;

struct Uses {
    uses: Vec<u32>,
    last_use: Vec<Position>,
    def: Vec<Position>,
}

impl Uses {
    fn visit(&mut self, data: &crate::ir::InstData, at: (usize, usize)) {
        for value in data.op.operands() {
            self.uses[value.index()] += 1;
            self.last_use[value.index()] = Some(at);
        }
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

fn consts<R>(func: &Function, locations: &mut [Location<R>]) {
    for data in &func.insts {
        if let (Some(result), Op::Const(imm)) = (data.result, &data.op)
            && small_int(func.ty(result))
        {
            locations[result.index()] = Location::Imm(*imm);
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
            (Location::Imm(a), Location::Imm(b)) if small_int(func.ty(x)) => Some(Location::Imm(i64::from(holds(cc, func.ty(x), a, b)))),
            _ => None,
        },
        Op::Binary(op, x, y) => match (locations[x.index()], locations[y.index()]) {
            (Location::Frame(slot, offset), Location::Imm(imm)) | (Location::Imm(imm), Location::Frame(slot, offset)) if op == Binary::Iadd => fold(slot, offset, imm),
            (Location::Imm(a), Location::Imm(b)) if small_int(func.ty(x)) => constant(op, a, b).map(Location::Imm),
            _ => None,
        },
        Op::Convert(Convert::Uextend | Convert::Ireduce, x) => match locations[x.index()] {
            Location::Imm(imm) if small_int(func.ty(x)) => Some(Location::Imm(crate::fold::extend(imm, func.ty(x), false))),
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
            if !matches!(locations[result.index()], Location::Slot) || !small_int(func.ty(result)) {
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
                && small_int(func.ty(sum))
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
fn through<R>(func: &Function, locations: &[Location<R>], census: &Uses) -> Vec<Option<i32>> {
    let mut through = vec![None; func.types.len()];
    let mut loads = vec![None; func.types.len()];
    for data in &func.insts {
        if let (Op::Load(addr, 0), Some(result)) = (&data.op, data.result)
            && let Location::Imm(at) = locations[addr.index()]
        {
            loads[result.index()] = i32::try_from(at).ok().filter(|at| *at > 0);
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

/// One register class: the registers that keep values across calls, the caller-saved ones planning may use, and the registers of this class each parameter arrives in.
struct Class<'a, F> {
    saved: &'a [F],
    leaf: Vec<F>,
    arrivals: Vec<Vec<F>>,
}

/// The integer class; a function with a 128-bit value keeps none in the registers 128-bit operations use as scratch.
fn int_class<'a, R: Copy + PartialEq>(func: &Function, regs: &'a Regs<R>, order: &[Block], params: &[Loc<R>]) -> Class<'a, R> {
    let wide = order.iter().flat_map(|block| &func.blocks[block.index()].insts).any(|inst| func.insts[inst.index()].result.is_some_and(|result| func.ty(result) == Type::I128));
    let leaf = regs.leaf.iter().filter(|reg| !(wide && regs.wide_scratch.contains(*reg))).copied().collect();
    let arrivals = params
        .iter()
        .map(|loc| match *loc {
            Loc::Int(reg) => vec![reg],
            Loc::Pair(low, high) => vec![low, high],
            _ => Vec::new(),
        })
        .collect();
    Class { saved: regs.saved, leaf, arrivals }
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
    Class { saved: &[], leaf: regs.float.to_vec(), arrivals }
}

/// The registers values may live in, the entry parameters that keep the register they arrive in, and the arrival registers no other value takes before the entry block.
struct Pool<R> {
    regs: Vec<R>,
    kept: Vec<Option<R>>,
    arrival: Vec<R>,
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
            && live.span[value.index()].is_some_and(|span| !crosses(&live.calls, span))
        {
            kept[value.index()] = Some(reg);
        }
    }
    let arrival = class.arrivals.iter().flatten().copied().filter(|reg| class.leaf.contains(reg) && !kept.contains(&Some(*reg))).collect();
    let regs = class.saved.iter().chain(&class.leaf).copied().collect();
    Pool { regs, kept, arrival }
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

/// Gives registers to values by linear scan over their spans, a saved one when the span holds a call, spilling the value that ends last when none is free; an entry parameter that keeps its arrival register takes it.
fn assign<F: Copy + PartialEq>(live: &Live, pooled: &[bool], saved_regs: &[F], pool: Pool<F>) -> (Vec<Option<F>>, Vec<F>) {
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
        let saved = crosses(&live.calls, (start, end));
        let reg = match pool.kept[value].or_else(|| take(&mut free, (saved, saved_regs), &used)) {
            Some(reg) => reg,
            None => match active.iter().enumerate().filter(|(_, entry)| entry.1 != ARRIVAL && (!saved || saved_regs.contains(&entry.2))).max_by_key(|(_, entry)| entry.0) {
                Some((at, &(until, other, reg))) if until > end => {
                    homes[other] = None;
                    active.swap_remove(at);
                    reg
                }
                _ => continue,
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
fn allocate<F: Copy + PartialEq>(func: &Function, live: &Live, class: &Class<F>, pooled: &[bool]) -> (Vec<Option<F>>, Vec<F>) {
    assign(live, pooled, class.saved, pool(func, class, live, pooled))
}

/// Writes each register in `homes` into the location of its value as `wrap` of it.
fn place<R, F>(locations: &mut [Location<R>], homes: Vec<Option<F>>, wrap: impl Fn(F) -> Location<R>) {
    for (location, home) in locations.iter_mut().zip(homes) {
        if let Some(reg) = home {
            *location = wrap(reg);
        }
    }
}

/// Gives constants and frame addresses static locations, integers and floats pool registers over their live spans, and finds the compares that feed the next branch.
pub(crate) fn plan<R: Copy + PartialEq>(func: &Function, regs: &Regs<R>, order: &[Block], params: &[Loc<R>]) -> Plan<R> {
    let count = func.types.len();
    let mut locations = vec![Location::Slot; count];
    consts(func, &mut locations);
    statics(func, &mut locations);
    let census = census(func);
    let wanted: Vec<bool> = (0..count).map(|value| matches!(locations[value], Location::Slot) && registered(regs, func.types[value])).collect();
    let live = live(func, order, &wanted);
    let candidates: Vec<bool> = (0..count).map(|value| eligible(&live, &wanted, value)).collect();
    let fused: Vec<bool> = (0..count).map(|value| candidates[value] && fuses(func, &census, value, regs.float_branches)).collect();
    let through = if regs.absolute_calls { through(func, &locations, &census) } else { vec![None; count] };
    let ints: Vec<bool> = (0..count).map(|value| candidates[value] && small_int(func.types[value]) && !fused[value] && through[value].is_none()).collect();
    let floats: Vec<bool> = (0..count).map(|value| candidates[value] && func.types[value].is_float() && live.span[value].is_some_and(|span| !crosses(&live.calls, span))).collect();
    let (homes, used) = allocate(func, &live, &int_class(func, regs, order, params), &ints);
    place(&mut locations, homes, Location::Reg);
    let (homes, _) = allocate(func, &live, &float_class(regs, params), &floats);
    place(&mut locations, homes, Location::Float);
    let (pairs, deferred) = pairs(func, &locations);
    Plan { locations, fused, used, pairs, deferred, through }
}

impl<R: Copy> Plan<R> {
    /// Where `value` lives.
    pub(crate) fn location(&self, value: Value) -> Location<R> {
        self.locations[value.index()]
    }

    /// Whether `value` needs no code where it is defined, because its location is a constant or a frame address.
    pub(crate) fn is_static(&self, value: Value) -> bool {
        matches!(self.location(value), Location::Imm(_) | Location::Frame(..))
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

    /// The float register that holds `value`, when its location is one.
    pub(crate) fn float_of(&self, value: Value) -> Option<u8> {
        match self.location(value) {
            Location::Float(reg) => Some(reg),
            _ => None,
        }
    }
}

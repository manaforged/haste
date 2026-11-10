use super::asm::{Mem, Reg, Rm, Size};
use super::frame::Loc;
use super::{Emit, Ext};
use super::live::{Live, live};
use crate::fold::{constant, holds};
use crate::ir::{Binary, Block, Convert, Function, InstData, Op, StackSlot, Type, Value};

/// Registers that hold integer values; the emitter never uses them as scratch, and a frame saves each one it uses.
const POOL: [Reg; 5] = super::frame::SAVED;

/// Caller-saved registers the emitter never uses as scratch, which values that span no call live in without a save.
const LEAF: [Reg; 4] = [Reg::Rdi, Reg::R9, Reg::R8, Reg::R10];

const FOLD_LIMIT: i64 = 1 << 24;

/// Where a value lives.
#[derive(Clone, Copy, Debug)]
pub(super) enum Location {
    Slot,
    Imm(i64),
    Reg(Reg),
    Frame(StackSlot, i32),
}

/// What the emitter may do with each value, decided before emission.
pub(super) struct Plan {
    pub locations: Vec<Location>,
    pub fused: Vec<bool>,
    pub used: Vec<Reg>,
    pub pairs: Vec<Option<Value>>,
    pub deferred: Vec<bool>,
    pub through: Vec<Option<i32>>,
}

fn small_int(ty: Type) -> bool {
    matches!(ty, Type::I8 | Type::I16 | Type::I32 | Type::I64)
}

type Position = Option<(usize, usize)>;

struct Uses {
    uses: Vec<u32>,
    last_use: Vec<Position>,
    def: Vec<Position>,
}

impl Uses {
    fn visit(&mut self, data: &InstData, at: (usize, usize)) {
        for value in data.op.operands() {
            self.uses[value.index()] += 1;
            self.last_use[value.index()] = Some(at);
        }
        for result in results(data) {
            self.def[result.index()] = Some(at);
        }
    }
}

/// The values an instruction defines: its result, or a call's results.
pub(super) fn results(data: &InstData) -> impl Iterator<Item = Value> + '_ {
    let calls: &[Value] = match &data.op {
        Op::Call { results, .. } => results,
        _ => &[],
    };
    data.result.into_iter().chain(calls.iter().copied())
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

fn consts(func: &Function, locations: &mut [Location]) {
    for data in &func.insts {
        if let (Some(result), Op::Const(imm)) = (data.result, &data.op)
            && small_int(func.ty(result))
        {
            locations[result.index()] = Location::Imm(*imm);
        }
    }
}

fn fold(slot: StackSlot, offset: i32, imm: i64) -> Option<Location> {
    let sum = i64::from(offset).checked_add(imm)?;
    if !(-FOLD_LIMIT..=FOLD_LIMIT).contains(&sum) {
        return None;
    }
    Some(Location::Frame(slot, sum as i32))
}

fn static_location(func: &Function, locations: &[Location], op: &Op) -> Option<Location> {
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
            Location::Imm(imm) if small_int(func.ty(x)) => Some(Location::Imm(extend(imm, func.ty(x), Ext::Zero))),
            _ => None,
        },
        _ => None,
    }
}

/// Gives frame addresses and compares of constants static locations, until no more change.
fn statics(func: &Function, locations: &mut [Location]) {
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

fn fuses(func: &Function, census: &Uses, value: usize) -> bool {
    let (Some((block, at)), Some((_, last))) = (census.def[value], census.last_use[value]) else { return false };
    let insts = &func.blocks[block].insts;
    let op = &func.insts[insts[at].index()].op;
    let next = insts.get(at + 1).map(|inst| &func.insts[inst.index()].op);
    census.uses[value] == 1
        && last == at + 1
        && matches!(op, Op::Icmp(..) | Op::Fits(..))
        && matches!(next, Some(Op::Brif(condition, ..) | Op::Trapz(condition)) if condition.index() == value)
}

/// For each overflow check that directly follows its own add, subtract or multiply, that operation's result, so one instruction computes both; and which results wait for their check.
fn pairs(func: &Function, locations: &[Location]) -> (Vec<Option<Value>>, Vec<bool>) {
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
fn through(func: &Function, locations: &[Location], census: &Uses) -> Vec<Option<i32>> {
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

fn eligible(func: &Function, live: &Live, locations: &[Location], value: usize) -> bool {
    let Some((start, end)) = live.span[value] else { return false };
    let unwinds = live.unwinds.iter().any(|at| start <= *at && *at < end);
    matches!(locations[value], Location::Slot) && small_int(func.types[value]) && !unwinds
}

/// The registers values may live in, the entry parameters that keep the register they arrive in, and the arrival registers no other value takes before the entry block.
struct Pool {
    regs: Vec<Reg>,
    kept: Vec<Option<Reg>>,
    arrival: Vec<Reg>,
}

/// Whether the span `start..=end` holds a call after its start, which clobbers the caller-saved registers.
fn crosses(calls: &[u32], (start, end): (u32, u32)) -> bool {
    let next = calls.partition_point(|at| *at <= start);
    calls.get(next).is_some_and(|at| *at <= end)
}

/// The saved pool and the free caller-saved registers; values that span no call may take the caller-saved ones, and so may entry parameters that arrive in one.
fn pool(func: &Function, (order, params): (&[Block], &[Loc]), live: &Live, pooled: &[bool]) -> Pool {
    let insts = || order.iter().flat_map(|block| &func.blocks[block.index()].insts).map(|inst| &func.insts[inst.index()]);
    let wide = insts().any(|data| data.result.is_some_and(|result| func.ty(result) == Type::I128));
    let usable = |reg: &Reg| LEAF.contains(reg) && !(wide && super::wide::WIDE_SCRATCH.contains(reg));
    let mut kept = vec![None; func.types.len()];
    let entry = func.entry.map_or(&[][..], |entry| &func.blocks[entry.index()].params[..]);
    for (value, loc) in entry.iter().zip(params) {
        if let Loc::Int(reg) = *loc
            && usable(&reg)
            && pooled[value.index()]
            && live.span[value.index()].is_some_and(|span| !crosses(&live.calls, span))
        {
            kept[value.index()] = Some(reg);
        }
    }
    let arrival = params
        .iter()
        .flat_map(|loc| match *loc {
            Loc::Int(reg) => vec![reg],
            Loc::Pair(low, high) => vec![low, high],
            _ => Vec::new(),
        })
        .filter(|reg| usable(reg) && !kept.contains(&Some(*reg)))
        .collect();
    let regs = POOL.iter().chain(LEAF.iter().filter(|reg| usable(reg))).copied().collect();
    Pool { regs, kept, arrival }
}

/// Takes from `free` a caller-saved register when `saved` is false, else a saved one, preferring one the frame already saves.
fn take(free: &mut Vec<Reg>, saved: bool, used: &[Reg]) -> Option<Reg> {
    let at = if saved {
        free.iter().rposition(|reg| used.contains(reg)).or_else(|| free.iter().rposition(|reg| POOL.contains(reg)))
    } else {
        free.iter().rposition(|reg| !POOL.contains(reg)).or_else(|| free.iter().rposition(|reg| POOL.contains(reg)))
    };
    at.map(|at| free.swap_remove(at))
}

const ARRIVAL: usize = usize::MAX;

/// Gives registers to values by linear scan over their spans, a saved one when the span holds a call, spilling the value that ends last when none is free; an entry parameter that keeps its arrival register takes it.
fn assign(live: &Live, pooled: &[bool], locations: &mut [Location], pool: Pool) -> Vec<Reg> {
    let mut values: Vec<(u32, u32, usize)> = (0..pooled.len()).filter(|value| pooled[*value]).filter_map(|value| live.span[value].map(|(start, end)| (start, end, value))).collect();
    values.sort_unstable();
    let kept: Vec<Reg> = pool.kept.iter().flatten().copied().collect();
    let mut free: Vec<Reg> = pool.regs.iter().filter(|reg| !pool.arrival.contains(reg) && !kept.contains(reg)).copied().collect();
    let mut active: Vec<(u32, usize, Reg)> = pool.arrival.iter().map(|reg| (0, ARRIVAL, *reg)).collect();
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
        let reg = match pool.kept[value].or_else(|| take(&mut free, saved, &used)) {
            Some(reg) => reg,
            None => match active.iter().enumerate().filter(|(_, entry)| entry.1 != ARRIVAL && (!saved || POOL.contains(&entry.2))).max_by_key(|(_, entry)| entry.0) {
                Some((at, &(until, other, reg))) if until > end => {
                    locations[other] = Location::Slot;
                    active.swap_remove(at);
                    reg
                }
                _ => continue,
            },
        };
        locations[value] = Location::Reg(reg);
        active.push((end, value, reg));
        if POOL.contains(&reg) && !used.contains(&reg) {
            used.push(reg);
        }
    }
    used
}

/// Gives constants and frame addresses static locations, integers pool registers over their live spans, and finds the compares that feed the next branch.
pub(super) fn plan(func: &Function, order: &[Block], params: &[Loc]) -> Plan {
    let count = func.types.len();
    let mut locations = vec![Location::Slot; count];
    consts(func, &mut locations);
    statics(func, &mut locations);
    let census = census(func);
    let wanted: Vec<bool> = (0..count).map(|value| matches!(locations[value], Location::Slot) && small_int(func.types[value])).collect();
    let live = live(func, order, &wanted);
    let candidates: Vec<bool> = (0..count).map(|value| eligible(func, &live, &locations, value)).collect();
    let fused: Vec<bool> = (0..count).map(|value| candidates[value] && fuses(func, &census, value)).collect();
    let through = through(func, &locations, &census);
    let pooled: Vec<bool> = (0..count).map(|value| candidates[value] && !fused[value] && through[value].is_none()).collect();
    let used = assign(&live, &pooled, &mut locations, pool(func, (order, params), &live, &pooled));
    let (pairs, deferred) = pairs(func, &locations);
    Plan { locations, fused, used, pairs, deferred, through }
}

fn extend(imm: i64, ty: Type, ext: Ext) -> i64 {
    crate::fold::extend(imm, ty, matches!(ext, Ext::Sign))
}

impl Emit<'_> {
    fn location(&self, value: Value) -> Location {
        self.plan.locations[value.index()]
    }

    /// Whether `value` needs no code where it is defined, because its location is a constant or a frame address.
    pub(super) fn is_static(&self, value: Value) -> bool {
        matches!(self.location(value), Location::Imm(_) | Location::Frame(..))
    }

    /// The constant `value` holds, when its location is one.
    pub(super) fn const_of(&self, value: Value) -> Option<i64> {
        match self.location(value) {
            Location::Imm(imm) => Some(imm),
            _ => None,
        }
    }

    pub(super) fn imm_of(&self, value: Value) -> Option<i32> {
        match self.location(value) {
            Location::Imm(imm) => i32::try_from(imm).ok(),
            _ => None,
        }
    }

    /// The constant `value` extended to 64 bits by `ext`, when it fits a sign-extended 32-bit immediate.
    pub(super) fn imm_extended(&self, value: Value, ext: Ext) -> Option<i32> {
        match self.location(value) {
            Location::Imm(imm) => i32::try_from(extend(imm, self.ty(value), ext)).ok(),
            _ => None,
        }
    }

    /// The register that holds `value`, when its location is one.
    pub(super) fn reg_of(&self, value: Value) -> Option<Reg> {
        match self.location(value) {
            Location::Reg(reg) => Some(reg),
            _ => None,
        }
    }

    /// The register to compute `value` in: its location register, or rax.
    pub(super) fn dst(&self, value: Value) -> Reg {
        self.reg_of(value).unwrap_or(Reg::Rax)
    }

    /// The memory at `addr + offset`: relative to rbp for a frame address, to the location register of `addr`, or through rcx.
    pub(super) fn address(&mut self, addr: Value, offset: i32) -> Mem {
        if let Some(reg) = self.reg_of(addr) {
            return Mem::new(reg, offset);
        }
        if let Location::Frame(slot, base) = self.location(addr)
            && let Some(disp) = self.frame.slots[slot.index()].checked_add(base).and_then(|disp| disp.checked_add(offset))
        {
            return Mem::new(Reg::Rbp, disp);
        }
        self.raw(Reg::Rcx, addr);
        Mem::new(Reg::Rcx, offset)
    }

    pub(super) fn get(&mut self, reg: Reg, value: Value, ext: Ext) {
        let ty = self.ty(value);
        let bytes = ty.bytes();
        let source: Rm = match self.location(value) {
            Location::Imm(imm) => return self.asm.mov_imm(reg, extend(imm, ty, ext)),
            Location::Frame(..) => return self.raw(reg, value),
            Location::Reg(from) => from.into(),
            Location::Slot => self.at(value).into(),
        };
        match ext {
            _ if bytes >= 8 => self.asm.load(Size::Q, reg, source),
            Ext::Zero => self.asm.load(Size::of(bytes), reg, source),
            Ext::Sign => self.asm.load_signed(Size::of(bytes), reg, source),
        }
    }

    pub(super) fn raw(&mut self, reg: Reg, value: Value) {
        match self.location(value) {
            Location::Imm(imm) => self.asm.mov_imm(reg, imm),
            Location::Frame(slot, offset) => {
                let at = self.slot(slot, offset);
                self.asm.lea(reg, at);
            }
            Location::Reg(from) if from == reg => {}
            Location::Reg(from) => self.asm.mov(reg, from),
            Location::Slot => {
                let at = self.at(value);
                self.asm.load(Size::Q, reg, at);
            }
        }
    }

    pub(super) fn put(&mut self, value: Value, reg: Reg) {
        if let Location::Reg(location) = self.location(value) {
            if location != reg {
                self.asm.mov(location, reg);
            }
            return;
        }
        let at = self.at(value);
        self.asm.store(Size::Q, at, reg);
    }

    /// A register or memory operand that holds `value`, using r11 for a frame address.
    pub(super) fn operand(&mut self, value: Value) -> Rm {
        match self.location(value) {
            Location::Reg(reg) => reg.into(),
            Location::Imm(_) => {
                let at = self.at(value);
                self.move_to(at, value);
                at.into()
            }
            Location::Frame(..) => {
                self.raw(Reg::R11, value);
                Reg::R11.into()
            }
            Location::Slot => self.at(value).into(),
        }
    }

    /// Writes the eight (or sixteen) bytes of `value` to `to`, using rax or r11 as scratch.
    pub(super) fn move_to(&mut self, to: Mem, value: Value) {
        match self.location(value) {
            Location::Imm(imm) => match i32::try_from(imm) {
                Ok(small) => self.asm.store_imm(to, small),
                Err(_) => {
                    self.asm.mov_imm(Reg::R11, imm);
                    self.asm.store(Size::Q, to, Reg::R11);
                }
            },
            Location::Frame(..) => {
                self.raw(Reg::R11, value);
                self.asm.store(Size::Q, to, Reg::R11);
            }
            Location::Reg(reg) => self.asm.store(Size::Q, to, reg),
            Location::Slot => self.copy_to(to, self.at(value), self.ty(value)),
        }
    }
}

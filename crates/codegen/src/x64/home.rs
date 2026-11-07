use super::asm::{Mem, Reg, Rm, Size};
use super::{Emit, Ext};
use crate::ir::{Binary, Function, InstData, Op, StackSlot, Type, Value};

/// Registers that hold block-local values; the emitter never uses them as scratch, and a frame saves each one it uses.
pub(super) const POOL: [Reg; 5] = [Reg::Rbx, Reg::R12, Reg::R13, Reg::R14, Reg::R15];

const FOLD_LIMIT: i64 = 1 << 24;

/// Where a value lives.
#[derive(Clone, Copy, Debug)]
pub(super) enum Home {
    Slot,
    Imm(i64),
    Reg(Reg),
    Frame(StackSlot, i32),
}

/// What the emitter may do with each value, decided before emission.
pub(super) struct Plan {
    pub homes: Vec<Home>,
    pub fused: Vec<bool>,
    pub used: Vec<Reg>,
}

fn small_int(ty: Type) -> bool {
    matches!(ty, Type::I8 | Type::I16 | Type::I32 | Type::I64)
}

type Position = Option<(usize, usize)>;

struct Census {
    uses: Vec<u32>,
    first_use: Vec<Position>,
    last_use: Vec<Position>,
    spread: Vec<bool>,
    def: Vec<Position>,
    try_calls: Vec<Vec<usize>>,
}

impl Census {
    fn record(&mut self, value: Value, at: (usize, usize)) {
        let index = value.index();
        self.uses[index] += 1;
        if self.last_use[index].is_some_and(|(block, _)| block != at.0) {
            self.spread[index] = true;
        }
        if self.first_use[index].is_none() {
            self.first_use[index] = Some(at);
        }
        self.last_use[index] = Some(at);
    }

    fn visit(&mut self, data: &InstData, at: (usize, usize)) {
        for value in data.op.operands() {
            self.record(value, at);
        }
        for result in results(data) {
            self.def[result.index()] = Some(at);
        }
    }
}

fn results(data: &InstData) -> impl Iterator<Item = Value> + '_ {
    let calls: &[Value] = match &data.op {
        Op::Call { results, .. } => results,
        _ => &[],
    };
    data.result.into_iter().chain(calls.iter().copied())
}

fn census(func: &Function) -> Census {
    let count = func.types.len();
    let mut census = Census {
        uses: vec![0; count],
        first_use: vec![None; count],
        last_use: vec![None; count],
        spread: vec![false; count],
        def: vec![None; count],
        try_calls: Vec::new(),
    };
    for (b, block) in func.blocks.iter().enumerate() {
        let mut try_calls = Vec::new();
        for (i, inst) in block.insts.iter().enumerate() {
            let data = &func.insts[inst.index()];
            census.visit(data, (b, i));
            if matches!(data.op, Op::TryCall { .. }) {
                try_calls.push(i);
            }
        }
        census.try_calls.push(try_calls);
    }
    census
}

fn consts(func: &Function, homes: &mut [Home]) {
    for data in &func.insts {
        if let (Some(result), Op::Const(imm)) = (data.result, &data.op)
            && small_int(func.ty(result))
        {
            homes[result.index()] = Home::Imm(*imm);
        }
    }
}

fn fold(slot: StackSlot, offset: i32, imm: i64) -> Option<Home> {
    let sum = i64::from(offset).checked_add(imm)?;
    if !(-FOLD_LIMIT..=FOLD_LIMIT).contains(&sum) {
        return None;
    }
    Some(Home::Frame(slot, sum as i32))
}

fn frame_home(homes: &[Home], op: &Op) -> Option<Home> {
    match *op {
        Op::StackAddr(slot, offset) => Some(Home::Frame(slot, offset)),
        Op::Binary(Binary::Iadd, x, y) => match (homes[x.index()], homes[y.index()]) {
            (Home::Frame(slot, offset), Home::Imm(imm)) | (Home::Imm(imm), Home::Frame(slot, offset)) => fold(slot, offset, imm),
            _ => None,
        },
        _ => None,
    }
}

fn frames(func: &Function, homes: &mut [Home]) {
    let mut changed = true;
    while changed {
        changed = false;
        for data in &func.insts {
            let Some(result) = data.result else { continue };
            if !matches!(homes[result.index()], Home::Slot) || func.ty(result) != Type::I64 {
                continue;
            }
            if let Some(home) = frame_home(homes, &data.op) {
                homes[result.index()] = home;
                changed = true;
            }
        }
    }
}

fn candidate(func: &Function, census: &Census, homes: &[Home], value: usize) -> bool {
    let (Some((block, at)), Some((first_block, first)), Some((_, last))) = (census.def[value], census.first_use[value], census.last_use[value])
    else {
        return false;
    };
    let unwinds = census.try_calls[block].iter().any(|call| *call > at && *call < last);
    matches!(homes[value], Home::Slot)
        && small_int(func.types[value])
        && !census.spread[value]
        && first_block == block
        && first > at
        && !unwinds
}

fn fuses(func: &Function, census: &Census, value: usize) -> bool {
    let (Some((block, at)), Some((_, last))) = (census.def[value], census.last_use[value]) else { return false };
    let insts = &func.blocks[block].insts;
    let op = &func.insts[insts[at].index()].op;
    let next = insts.get(at + 1).map(|inst| &func.insts[inst.index()].op);
    census.uses[value] == 1
        && last == at + 1
        && matches!(op, Op::Icmp(..))
        && matches!(next, Some(Op::Brif(condition, ..)) if condition.index() == value)
}

struct Pool {
    free: Vec<Reg>,
    live: Vec<(usize, Reg)>,
}

impl Pool {
    fn define(&mut self, value: Value, homes: &mut [Home], used: &mut [bool]) {
        let Some(reg) = self.free.pop() else { return };
        homes[value.index()] = Home::Reg(reg);
        self.live.push((value.index(), reg));
        if let Some(index) = POOL.iter().position(|pooled| *pooled == reg) {
            used[index] = true;
        }
    }

    fn release(&mut self, last_use: &[Position], at: (usize, usize)) {
        let free = &mut self.free;
        self.live.retain(|(value, reg)| {
            let dies = last_use[*value] == Some(at);
            if dies {
                free.push(*reg);
            }
            !dies
        });
    }
}

fn assign(func: &Function, census: &Census, pooled: &[bool], homes: &mut [Home]) -> Vec<Reg> {
    let mut used = [false; POOL.len()];
    for (b, block) in func.blocks.iter().enumerate() {
        let mut pool = Pool { free: POOL.to_vec(), live: Vec::new() };
        for (i, inst) in block.insts.iter().enumerate() {
            for result in results(&func.insts[inst.index()]) {
                if pooled[result.index()] {
                    pool.define(result, homes, &mut used);
                }
            }
            pool.release(&census.last_use, (b, i));
        }
    }
    POOL.iter().zip(used).filter(|(_, used)| *used).map(|(reg, _)| *reg).collect()
}

/// Gives constants and frame addresses static homes, block-local integers pool registers, and finds the compares that feed the next branch.
pub(super) fn plan(func: &Function) -> Plan {
    let count = func.types.len();
    let mut homes = vec![Home::Slot; count];
    consts(func, &mut homes);
    frames(func, &mut homes);
    let census = census(func);
    let candidates: Vec<bool> = (0..count).map(|value| candidate(func, &census, &homes, value)).collect();
    let fused: Vec<bool> = (0..count).map(|value| candidates[value] && fuses(func, &census, value)).collect();
    let pooled: Vec<bool> = (0..count).map(|value| candidates[value] && !fused[value]).collect();
    let used = assign(func, &census, &pooled, &mut homes);
    Plan { homes, fused, used }
}

fn extend(imm: i64, ty: Type, ext: Ext) -> i64 {
    let bits = ty.bits();
    if bits >= 64 {
        return imm;
    }
    let shift = 64 - bits;
    match ext {
        Ext::Zero => ((imm as u64) << shift >> shift) as i64,
        Ext::Sign => imm << shift >> shift,
    }
}

impl Emit<'_> {
    fn home(&self, value: Value) -> Home {
        self.plan.homes[value.index()]
    }

    /// Whether `value` needs no code where it is defined, because its home is a constant or a frame address.
    pub(super) fn is_static(&self, value: Value) -> bool {
        matches!(self.home(value), Home::Imm(_) | Home::Frame(..))
    }

    pub(super) fn imm_of(&self, value: Value) -> Option<i32> {
        match self.home(value) {
            Home::Imm(imm) => i32::try_from(imm).ok(),
            _ => None,
        }
    }

    /// The constant `value` extended to 64 bits by `ext`, when it fits a sign-extended 32-bit immediate.
    pub(super) fn imm_extended(&self, value: Value, ext: Ext) -> Option<i32> {
        match self.home(value) {
            Home::Imm(imm) => i32::try_from(extend(imm, self.ty(value), ext)).ok(),
            _ => None,
        }
    }

    /// The memory at `addr + offset`, relative to rbp when `addr` is a frame address and through rcx otherwise.
    pub(super) fn address(&mut self, addr: Value, offset: i32) -> Mem {
        if let Home::Frame(slot, base) = self.home(addr)
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
        let source: Rm = match self.home(value) {
            Home::Imm(imm) => return self.asm.mov_imm(reg, extend(imm, ty, ext)),
            Home::Frame(..) => return self.raw(reg, value),
            Home::Reg(from) => from.into(),
            Home::Slot => self.at(value).into(),
        };
        match ext {
            _ if bytes >= 8 => self.asm.load(Size::Q, reg, source),
            Ext::Zero => self.asm.load(Size::of(bytes), reg, source),
            Ext::Sign => self.asm.load_signed(Size::of(bytes), reg, source),
        }
    }

    pub(super) fn raw(&mut self, reg: Reg, value: Value) {
        match self.home(value) {
            Home::Imm(imm) => self.asm.mov_imm(reg, imm),
            Home::Frame(slot, offset) => {
                let at = self.slot(slot, offset);
                self.asm.lea(reg, at);
            }
            Home::Reg(from) => self.asm.mov(reg, from),
            Home::Slot => {
                let at = self.at(value);
                self.asm.load(Size::Q, reg, at);
            }
        }
    }

    pub(super) fn put(&mut self, value: Value, reg: Reg) {
        if let Home::Reg(home) = self.home(value) {
            self.asm.mov(home, reg);
            return;
        }
        let at = self.at(value);
        self.asm.store(Size::Q, at, reg);
    }

    /// A register or memory operand that holds `value`, using r11 for a frame address.
    pub(super) fn operand(&mut self, value: Value) -> Rm {
        match self.home(value) {
            Home::Reg(reg) => reg.into(),
            Home::Imm(_) => {
                let at = self.at(value);
                self.move_to(at, value);
                at.into()
            }
            Home::Frame(..) => {
                self.raw(Reg::R11, value);
                Reg::R11.into()
            }
            Home::Slot => self.at(value).into(),
        }
    }

    /// Writes the eight (or sixteen) bytes of `value` to `to`, using rax or r11 as scratch.
    pub(super) fn move_to(&mut self, to: Mem, value: Value) {
        match self.home(value) {
            Home::Imm(imm) => match i32::try_from(imm) {
                Ok(small) => self.asm.store_imm(to, small),
                Err(_) => {
                    self.asm.mov_imm(Reg::R11, imm);
                    self.asm.store(Size::Q, to, Reg::R11);
                }
            },
            Home::Frame(..) => {
                self.raw(Reg::R11, value);
                self.asm.store(Size::Q, to, Reg::R11);
            }
            Home::Reg(reg) => self.asm.store(Size::Q, to, reg),
            Home::Slot => self.copy_to(to, self.at(value), self.ty(value)),
        }
    }
}

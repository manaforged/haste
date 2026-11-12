use crate::fold::{extend, holds};
use crate::ir::{Binary, Block, Convert, Function, Inst, IntCC, Op, Type, Value};

/// The most facts a path keeps; past it the oldest goes.
const MAX_FACTS: usize = 16;

/// The most constants one fact records a value as differing from.
const MAX_NOT: usize = 32;

/// What is known about an integer value: its unsigned range at its type's width, and constants it differs from.
#[derive(Clone, PartialEq, Debug)]
pub(crate) struct Fact {
    lo: u64,
    hi: u64,
    not: Vec<u64>,
}

/// The largest unsigned value of type `ty`.
fn top(ty: Type) -> u64 {
    if ty.bits() >= 64 { u64::MAX } else { (1_u64 << ty.bits()) - 1 }
}

/// `imm` as an unsigned value at the width of `ty`.
fn unsigned(imm: i64, ty: Type) -> u64 {
    extend(imm, ty, false) as u64
}

impl Fact {
    /// Nothing known about a value of type `ty`.
    fn full(ty: Type) -> Self {
        Self { lo: 0, hi: top(ty), not: Vec::new() }
    }

    /// Exactly the constant `imm` of type `ty`.
    fn constant(ty: Type, imm: i64) -> Self {
        let value = unsigned(imm, ty);
        Self::range(value, value)
    }

    fn range(lo: u64, hi: u64) -> Self {
        Self { lo, hi, not: Vec::new() }
    }

    /// The single value the fact allows, when it allows one.
    pub(crate) fn exact(&self) -> Option<u64> {
        (self.lo == self.hi).then_some(self.lo)
    }

    /// What both facts say.
    fn meet(mut self, other: &Fact) -> Self {
        self.lo = self.lo.max(other.lo);
        self.hi = self.hi.min(other.hi);
        for value in &other.not {
            self.exclude(*value);
        }
        self
    }

    /// Records that the value differs from `value`, narrowing the range when it sits at an end.
    fn exclude(&mut self, value: u64) {
        if self.lo == value && self.lo < self.hi {
            self.lo += 1;
        } else if self.hi == value && self.lo < self.hi {
            self.hi -= 1;
        } else if (self.lo..=self.hi).contains(&value) && !self.not.contains(&value) && self.not.len() < MAX_NOT {
            self.not.push(value);
        }
    }

    /// Whether the value can equal `value`.
    fn admits(&self, value: u64) -> bool {
        (self.lo..=self.hi).contains(&value) && !self.not.contains(&value)
    }

    /// Whether the fact holds for both signed and unsigned readings alike: its range sits below the sign bit of `ty`.
    fn non_negative(&self, ty: Type) -> bool {
        self.hi <= top(ty) >> 1
    }
}

/// The unsigned comparison that agrees with `cc` on operands below the sign bit.
fn as_unsigned(cc: IntCC) -> IntCC {
    match cc {
        IntCC::SignedLessThan => IntCC::UnsignedLessThan,
        IntCC::SignedGreaterThanOrEqual => IntCC::UnsignedGreaterThanOrEqual,
        IntCC::SignedGreaterThan => IntCC::UnsignedGreaterThan,
        IntCC::SignedLessThanOrEqual => IntCC::UnsignedLessThanOrEqual,
        other => other,
    }
}

/// `x cc y` when the facts settle it for every pair of values they allow.
pub(crate) fn decide(cc: IntCC, ty: Type, x: &Fact, y: &Fact) -> Option<bool> {
    if let (Some(a), Some(b)) = (x.exact(), y.exact()) {
        return Some(holds(cc, ty, a as i64, b as i64));
    }
    let cc = match cc.is_signed() {
        true if x.non_negative(ty) && y.non_negative(ty) => as_unsigned(cc),
        true => return None,
        false => cc,
    };
    let disjoint = x.hi < y.lo || y.hi < x.lo || y.exact().is_some_and(|b| !x.admits(b)) || x.exact().is_some_and(|a| !y.admits(a));
    match cc {
        IntCC::Equal => disjoint.then_some(false),
        IntCC::NotEqual => disjoint.then_some(true),
        _ => ordered(cc, x, y),
    }
}

/// `x cc y` for an unsigned order when the ranges settle it.
fn ordered(cc: IntCC, x: &Fact, y: &Fact) -> Option<bool> {
    match cc {
        IntCC::UnsignedLessThan => order(x.hi < y.lo, x.lo >= y.hi),
        IntCC::UnsignedGreaterThanOrEqual => order(x.lo >= y.hi, x.hi < y.lo),
        IntCC::UnsignedGreaterThan => order(x.lo > y.hi, x.hi <= y.lo),
        IntCC::UnsignedLessThanOrEqual => order(x.hi <= y.lo, x.lo > y.hi),
        _ => None,
    }
}

fn order(always: bool, never: bool) -> Option<bool> {
    if always { Some(true) } else if never { Some(false) } else { None }
}

/// The fact on `x` that `x cc k` being `taken` gives, for an equality or unsigned `cc`.
fn compared(cc: IntCC, ty: Type, k: u64, taken: bool) -> Option<Fact> {
    let cc = if taken { cc } else { cc.inverse() };
    let max = top(ty);
    let mut fact = Fact::full(ty);
    match cc {
        IntCC::Equal => fact = Fact::range(k, k),
        IntCC::NotEqual => fact.exclude(k),
        IntCC::UnsignedLessThan if k > 0 => fact.hi = k - 1,
        IntCC::UnsignedLessThanOrEqual => fact.hi = k,
        IntCC::UnsignedGreaterThan if k < max => fact.lo = k + 1,
        IntCC::UnsignedGreaterThanOrEqual => fact.lo = k,
        _ => return None,
    }
    Some(fact)
}

/// The order with its operands swapped: `k cc x` is `x (swapped cc) k`.
fn swapped(cc: IntCC) -> IntCC {
    match cc {
        IntCC::SignedLessThan => IntCC::SignedGreaterThan,
        IntCC::SignedGreaterThan => IntCC::SignedLessThan,
        IntCC::SignedLessThanOrEqual => IntCC::SignedGreaterThanOrEqual,
        IntCC::SignedGreaterThanOrEqual => IntCC::SignedLessThanOrEqual,
        IntCC::UnsignedLessThan => IntCC::UnsignedGreaterThan,
        IntCC::UnsignedGreaterThan => IntCC::UnsignedLessThan,
        IntCC::UnsignedLessThanOrEqual => IntCC::UnsignedGreaterThanOrEqual,
        IntCC::UnsignedGreaterThanOrEqual => IntCC::UnsignedLessThanOrEqual,
        other => other,
    }
}

/// The range each value's definition proves, computed once in reverse postorder so that operands come before the values that read them.
pub(crate) struct Ranges<'a> {
    defs: &'a [Option<Inst>],
    table: Vec<(u64, u64)>,
}

impl<'a> Ranges<'a> {
    /// The ranges of `func`'s values over the blocks of `order`, with the constants `consts` an earlier fold found.
    pub(crate) fn new(func: &Function, (defs, consts): (&'a [Option<Inst>], &[Option<i64>]), order: &[Block]) -> Self {
        let table = func.types.iter().map(|ty| (0, top(*ty))).collect();
        let mut ranges = Self { defs, table };
        for inst in order.iter().flat_map(|block| &func.blocks[block.index()].insts) {
            let data = &func.insts[inst.index()];
            let Some(result) = data.result.filter(|result| func.ty(*result).is_small_int()) else { continue };
            let fact = match consts.get(result.index()).copied().flatten() {
                Some(imm) => Fact::constant(func.ty(result), imm),
                None => ranges.derive(func, &data.op, func.ty(result)),
            };
            ranges.table[result.index()] = (fact.lo, fact.hi);
        }
        ranges
    }

    fn op<'f>(&self, func: &'f Function, value: Value) -> Option<&'f Op> {
        self.defs.get(value.index()).copied().flatten().map(|inst| &func.insts[inst.index()].op)
    }

    /// The range of `value` its definition proves.
    pub(crate) fn of(&self, value: Value) -> Fact {
        let (lo, hi) = self.table[value.index()];
        Fact::range(lo, hi)
    }

    /// The constant `value` holds, read unsigned at its width, when its range is one value.
    fn constant(&self, value: Value) -> Option<u64> {
        self.of(value).exact()
    }

    /// The range of the result of `op`, of type `ty`, from the ranges of its operands.
    fn derive(&self, func: &Function, op: &Op, ty: Type) -> Fact {
        match *op {
            Op::Const(imm) => Fact::constant(ty, imm),
            Op::Icmp(..) | Op::Fcmp(..) | Op::Fits(..) => Fact::range(0, 1),
            Op::Convert(Convert::Uextend, x) => self.of(x),
            Op::Convert(Convert::Ireduce, x) => Some(self.of(x)).filter(|inner| inner.hi <= top(ty)).unwrap_or_else(|| Fact::full(ty)),
            Op::Binary(op, x, y) if func.ty(x).is_small_int() => self.binary(func, op, (x, y)),
            Op::Select(_, x, y) => {
                let (a, b) = (self.of(x), self.of(y));
                Fact::range(a.lo.min(b.lo), a.hi.max(b.hi))
            }
            _ => Fact::full(ty),
        }
    }

    /// The range of `x op y` that a constant operand proves.
    fn binary(&self, func: &Function, op: Binary, (x, y): (Value, Value)) -> Fact {
        let ty = func.ty(x);
        match op {
            Binary::Band => match self.constant(y).or_else(|| self.constant(x)) {
                Some(mask) => Fact::range(0, mask),
                None => Fact::full(ty),
            },
            Binary::Ushr => match self.constant(y) {
                Some(count) => {
                    let (inner, shift) = (self.of(x), (count & u64::from(ty.bits() - 1)) as u32);
                    Fact::range(inner.lo >> shift, inner.hi >> shift)
                }
                None => Fact::full(ty),
            },
            Binary::Urem => match self.constant(y) {
                Some(divisor) if divisor > 0 => Fact::range(0, divisor - 1),
                _ => Fact::full(ty),
            },
            Binary::Iadd => self.shifted(ty, (x, y), false),
            Binary::Isub => self.shifted(ty, (x, y), true),
            _ => Fact::full(ty),
        }
    }

    /// The range of `x + k` or `x - k` for a constant `k`, when no value of `x`'s range wraps at the width of `ty`.
    fn shifted(&self, ty: Type, (x, y): (Value, Value), subtract: bool) -> Fact {
        let Some(k) = self.constant(y) else { return Fact::full(ty) };
        let inner = self.of(x);
        let moved = match subtract {
            false => inner.lo.checked_add(k).zip(inner.hi.checked_add(k)).filter(|(_, hi)| *hi <= top(ty)),
            true => inner.lo.checked_sub(k).zip(inner.hi.checked_sub(k)),
        };
        moved.map_or_else(|| Fact::full(ty), |(lo, hi)| Fact::range(lo, hi))
    }
}

/// The conditions known to hold at one point of a function: facts on values that the branches taken to get there give.
#[derive(Clone, Default)]
pub(crate) struct Path {
    facts: Vec<(Value, Fact)>,
}

impl Path {
    /// The fact the path records on `value`.
    fn get(&self, value: Value) -> Option<&Fact> {
        self.facts.iter().rev().find(|(at, _)| *at == value).map(|(_, fact)| fact)
    }

    /// Everything known about `value` here: its definition's range and the path's facts.
    pub(crate) fn fact(&self, ranges: &Ranges, value: Value) -> Fact {
        let defined = ranges.of(value);
        match self.get(value) {
            Some(fact) => defined.meet(fact),
            None => defined,
        }
    }

    /// Records `fact` on `value`, forgetting the oldest fact past `MAX_FACTS`.
    fn add(&mut self, ranges: &Ranges, value: Value, fact: Fact) {
        let merged = self.fact(ranges, value).meet(&fact);
        if self.facts.len() >= MAX_FACTS {
            self.facts.remove(0);
        }
        self.facts.push((value, merged));
    }

    /// Records that the branch on `condition` went the `taken` way: the condition is nonzero or zero, and the integer compare that defines it held or failed.
    pub(crate) fn assume(&mut self, func: &Function, ranges: &Ranges, condition: Value, taken: bool) {
        let ty = func.ty(condition);
        if !ty.is_small_int() {
            return;
        }
        if let Some(truth) = compared(IntCC::NotEqual, ty, 0, taken) {
            self.add(ranges, condition, truth);
        }
        let Some(&Op::Icmp(cc, x, y)) = ranges.op(func, condition) else { return };
        let operand_ty = func.ty(x);
        if !operand_ty.is_small_int() {
            return;
        }
        let (left, right) = (self.fact(ranges, x), self.fact(ranges, y));
        let order = match cc.is_signed() {
            true if left.non_negative(operand_ty) && right.non_negative(operand_ty) => as_unsigned(cc),
            true => return,
            false => cc,
        };
        if let Some(fact) = right.exact().and_then(|k| compared(order, operand_ty, k, taken)) {
            self.add(ranges, x, fact);
        }
        if let Some(fact) = left.exact().and_then(|k| compared(swapped(order), operand_ty, k, taken)) {
            self.add(ranges, y, fact);
        }
    }

    /// Records what the edge from `from` to `into` says when `from` ends in a branch with two different targets.
    fn follow(&mut self, func: &Function, ranges: &Ranges, from: Block, into: Block) {
        if let Some(&Op::Brif(condition, then, _, otherwise, _)) = func.last(from)
            && then != otherwise
        {
            self.assume(func, ranges, condition, then == into);
        }
    }
}

/// The only block that branches to each block, for blocks with exactly one incoming edge.
pub(crate) fn single_preds(func: &Function) -> Vec<Option<Block>> {
    let mut single: Vec<Option<Option<Block>>> = vec![None; func.blocks.len()];
    for block in 0..func.blocks.len() as u32 {
        for target in func.successors(Block(block)) {
            let slot = &mut single[target.index()];
            *slot = match slot {
                None => Some(Some(Block(block))),
                Some(_) => Some(None),
            };
        }
    }
    if let Some(entry) = func.entry {
        single[entry.index()] = Some(None);
    }
    single.into_iter().map(Option::flatten).collect()
}

/// Settles the compares, traps and branches that the facts on entry to each block and its earlier instructions decide: a settled compare's result becomes a constant in `consts`, a trap that cannot fire goes, and a settled branch becomes a jump. A block's facts on entry are its only predecessor's facts on exit plus that edge's branch. Returns whether anything changed.
pub(crate) fn settle(func: &mut Function, defs: &[Option<Inst>], consts: &mut [Option<i64>]) -> bool {
    let Some(entry) = func.entry else { return false };
    let (single, order) = (single_preds(func), crate::dom::reverse_postorder(func, entry));
    let (mut decided, mut compares) = (Vec::new(), Vec::new());
    {
        let ranges = Ranges::new(func, (defs, consts), &order);
        let mut exits: Vec<Option<Path>> = vec![None; func.blocks.len()];
        for &block in &order {
            let mut path = match single[block.index()].and_then(|pred| exits[pred.index()].as_ref().map(|exit| (pred, exit.clone()))) {
                Some((pred, mut path)) => {
                    path.follow(func, &ranges, pred, block);
                    path
                }
                None => Path::default(),
            };
            for &inst in &func.blocks[block.index()].insts {
                settle_one(func, (&ranges, &mut path), inst, (&mut decided, &mut compares));
            }
            exits[block.index()] = Some(path);
        }
    }
    let settled = !compares.is_empty();
    for (result, known) in compares {
        consts[result.index()] = Some(i64::from(known));
    }
    apply(func, &decided) | settled
}

/// Settles `inst` under `path` and moves the path past it.
fn settle_one(func: &Function, (ranges, path): (&Ranges, &mut Path), inst: Inst, (decided, compares): (&mut Vec<(Inst, Option<bool>)>, &mut Vec<(Value, bool)>)) {
    let data = &func.insts[inst.index()];
    match (&data.op, data.result) {
        (&Op::Icmp(cc, x, y), Some(result)) if ranges.constant(result).is_none() && func.ty(x).is_small_int() => {
            if let Some(known) = decide(cc, func.ty(x), &path.fact(ranges, x), &path.fact(ranges, y)) {
                compares.push((result, known));
            }
        }
        (&Op::Trapz(condition), _) => {
            if !path.fact(ranges, condition).admits(0) {
                decided.push((inst, None));
            }
            path.assume(func, ranges, condition, true);
        }
        (&Op::Brif(condition, ..), _) if ranges.constant(condition).is_none() => {
            if let Some(taken) = truth(&path.fact(ranges, condition)) {
                decided.push((inst, Some(taken)));
            }
        }
        _ => {}
    }
}

/// Whether a value with `fact` is nonzero, when the fact settles it.
fn truth(fact: &Fact) -> Option<bool> {
    match fact.exact() {
        Some(known) => Some(known != 0),
        None => (!fact.admits(0)).then_some(true),
    }
}

/// Drops each trap in `decided` and turns each branch into a jump the way it goes.
fn apply(func: &mut Function, decided: &[(Inst, Option<bool>)]) -> bool {
    let mut dropped = vec![false; func.insts.len()];
    for &(inst, taken) in decided {
        let op = &mut func.insts[inst.index()].op;
        match (taken, std::mem::replace(op, Op::Trap)) {
            (Some(taken), Op::Brif(_, then, then_args, otherwise, else_args)) => {
                *op = if taken { Op::Jump(then, then_args) } else { Op::Jump(otherwise, else_args) };
            }
            (_, other) => {
                dropped[inst.index()] = taken.is_none();
                *op = other;
            }
        }
    }
    for block in &mut func.blocks {
        block.insts.retain(|inst| !dropped[inst.index()]);
    }
    !decided.is_empty()
}

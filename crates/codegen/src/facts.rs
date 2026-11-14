use crate::cfg::dominates;
use crate::fold::{extend, holds};
use crate::ir::{Binary, Block, Convert, Function, Inst, IntCC, Op, Type, Value};

/// The most facts a path keeps; past it the oldest goes.
const MAX_FACTS: usize = 16;

/// The most constants one fact records a value as differing from.
const MAX_NOT: usize = 32;

/// The most symbolic relations a path keeps; past it the oldest goes.
const MAX_RELS: usize = 32;

#[derive(Clone, Copy, PartialEq, Eq)]
enum RelKind {
    Eq,
    Ne,
    Ult,
    Ule,
    Slt,
    Sle,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Rel {
    kind: RelKind,
    lhs: Value,
    rhs: Value,
    ty: Type,
}

/// What is known about an integer value: its unsigned range at its type's width, and constants it differs from.
#[derive(Clone, PartialEq, Debug)]
pub(crate) struct Fact {
    lo: u64,
    hi: u64,
    not: Vec<u64>,
}

/// `imm` as an unsigned value at the width of `ty`.
fn unsigned(imm: i64, ty: Type) -> u64 {
    extend(imm, ty, false) as u64
}

impl Fact {
    /// Nothing known about a value of type `ty`.
    fn full(ty: Type) -> Self {
        Self { lo: 0, hi: ty.unsigned_max(), not: Vec::new() }
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
        self.hi <= ty.unsigned_max() >> 1
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

/// The fact on `x` that `x cc y` being `taken` gives, for an equality or unsigned `cc`.
fn compared_range(cc: IntCC, ty: Type, rhs: &Fact, taken: bool) -> Option<Fact> {
    let cc = if taken { cc } else { cc.inverse() };
    let max = ty.unsigned_max();
    let mut fact = Fact::full(ty);
    match cc {
        IntCC::Equal => fact = rhs.clone(),
        IntCC::NotEqual => fact.exclude(rhs.exact()?),
        IntCC::UnsignedLessThan if rhs.hi > 0 => fact.hi = rhs.hi - 1,
        IntCC::UnsignedLessThanOrEqual => fact.hi = rhs.hi,
        IntCC::UnsignedGreaterThan if rhs.lo < max => fact.lo = rhs.lo + 1,
        IntCC::UnsignedGreaterThanOrEqual => fact.lo = rhs.lo,
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

struct CfgInfo<'a> {
    consts: &'a [Option<i64>],
    incoming: &'a [Vec<(Block, &'a [Value])>],
    complete: &'a [bool],
    single: &'a [Option<Block>],
    idom: &'a [Option<Block>],
}

impl<'a> Ranges<'a> {
    /// The ranges of `func`'s values over the blocks of `order`, with the constants `consts` an earlier fold found.
    pub(crate) fn new(func: &Function, (defs, consts): (&'a [Option<Inst>], &[Option<i64>]), (order, single, idom): (&[Block], &[Option<Block>], &[Option<Block>])) -> Self {
        let table = func.types.iter().map(|ty| (0, ty.unsigned_max())).collect();
        let mut ranges = Self { defs, table };
        let mut incoming: Vec<Vec<(Block, &[Value])>> = vec![Vec::new(); func.blocks.len()];
        let mut complete = vec![true; func.blocks.len()];
        for pred_idx in 0..func.blocks.len() {
            let pred = Block(pred_idx as u32);
            match func.last(pred) {
                Some(Op::Jump(target, args)) => incoming[target.index()].push((pred, args.as_slice())),
                Some(Op::Brif(_, then, then_args, otherwise, else_args)) => {
                    incoming[then.index()].push((pred, then_args.as_slice()));
                    incoming[otherwise.index()].push((pred, else_args.as_slice()));
                }
                _ => {
                    for target in func.successors(pred) {
                        complete[target.index()] = false;
                    }
                }
            }
        }
        let info = CfgInfo { consts, incoming: &incoming, complete: &complete, single, idom };
        if ranges.propagate(func, &info, order) {
            ranges.propagate(func, &info, order);
        }
        ranges
    }

    fn propagate(&mut self, func: &Function, info: &CfgInfo, order: &[Block]) -> bool {
        let mut narrowed = false;
        for &block in order {
            narrowed |= self.refine_params(func, info, block);
            for inst in &func.blocks[block.index()].insts {
                let data = &func.insts[inst.index()];
                let Some(result) = data.result.filter(|result| func.ty(*result).is_small_int()) else { continue };
                let fact = match info.consts.get(result.index()).copied().flatten() {
                    Some(imm) => Fact::constant(func.ty(result), imm),
                    None => self.derive(func, &data.op, func.ty(result)),
                };
                self.table[result.index()] = (fact.lo, fact.hi);
            }
        }
        narrowed
    }

    fn refine_params(&mut self, func: &Function, info: &CfgInfo, block: Block) -> bool {
        if !info.complete[block.index()] || info.incoming[block.index()].is_empty() {
            return false;
        }
        let mut narrowed = false;
        for (k, &p) in func.blocks[block.index()].params.iter().enumerate() {
            if func.ty(p).is_small_int() {
                narrowed |= self.refine_param_lo(func, info, (block, p, k));
                narrowed |= self.refine_param_hi(func, info, (block, p, k));
            }
        }
        narrowed
    }

    fn refine_param_lo(&mut self, func: &Function, info: &CfgInfo, (block, p, k): (Block, Value, usize)) -> bool {
        let ty = func.ty(p);
        let (mut min_lo, mut has_outside, mut valid) = (u64::MAX, false, true);
        for &(pred, args) in &info.incoming[block.index()] {
            let Some(&arg) = args.get(k) else { valid = false; break };
            if let Some(step) = self.add_step(func, info.consts, arg, p, ty) {
                let Some(max_p) = ty.unsigned_max().checked_sub(step) else { valid = false; break };
                if !self.edge_bound(func, info, (block, pred), (p, true)).is_some_and(|ub| ub <= max_p) {
                    valid = false;
                    break;
                }
            } else {
                has_outside = true;
                min_lo = min_lo.min(self.of(arg).lo);
            }
        }
        if valid && has_outside && min_lo > self.table[p.index()].0 && min_lo <= ty.unsigned_max() {
            self.table[p.index()].0 = min_lo;
            return true;
        }
        false
    }

    fn refine_param_hi(&mut self, func: &Function, info: &CfgInfo, (block, p, k): (Block, Value, usize)) -> bool {
        let (mut max_hi, mut has_bound) = (0_u64, false);
        for &(pred, args) in &info.incoming[block.index()] {
            let Some(&arg) = args.get(k) else { return false };
            let Some(edge_hi) = self.edge_arm_hi(func, info, (block, pred), (p, arg, 0)) else { return false };
            if let Some(hi) = edge_hi {
                max_hi = max_hi.max(hi);
                has_bound = true;
            }
        }
        if has_bound && max_hi < self.table[p.index()].1 {
            self.table[p.index()].1 = max_hi;
            return true;
        }
        false
    }

    fn edge_arm_hi(&self, func: &Function, info: &CfgInfo, edge: (Block, Block), (p, arg, depth): (Value, Value, u8)) -> Option<Option<u64>> {
        if arg == p { return Some(None); }
        let ty = func.ty(p);
        if let Some(ub) = self.edge_bound(func, info, edge, (arg, true)) { return Some(Some(ub)); }
        if let Some(step) = self.add_step(func, info.consts, arg, p, ty)
            && let Some(ub) = self.edge_bound(func, info, edge, (p, true))
            && let Some(next_hi) = ub.checked_add(step).filter(|&hi| hi <= ty.unsigned_max())
        {
            return Some(Some(next_hi));
        }
        match self.op(func, arg) {
            Some(&Op::Binary(Binary::Ushr, base, _)) if base == p => return Some(None),
            Some(&Op::Binary(Binary::Band, a, b)) if a == p || b == p => return Some(None),
            Some(&Op::Binary(Binary::Isub, base, delta)) if base == p => {
                let sub_shr = matches!(self.op(func, delta), Some(&Op::Binary(Binary::Ushr, inner, _)) if inner == p);
                if sub_shr || self.const_val(func, info.consts, delta, ty).is_some_and(|k| self.edge_bound(func, info, edge, (p, false)).is_some_and(|lb| lb >= k)) {
                    return Some(None);
                }
            }
            Some(&Op::Select(_, a, b)) if depth < 2 => {
                return match (self.edge_arm_hi(func, info, edge, (p, a, depth + 1))?, self.edge_arm_hi(func, info, edge, (p, b, depth + 1))?) {
                    (None, None) => Some(None),
                    (Some(x), None) | (None, Some(x)) => Some(Some(x)),
                    (Some(x), Some(y)) => Some(Some(x.max(y))),
                };
            }
            _ => {}
        }
        let outside = self.const_val(func, info.consts, arg, ty).unwrap_or_else(|| self.of(arg).hi);
        (outside < ty.unsigned_max()).then_some(Some(outside))
    }

    fn const_val(&self, func: &Function, consts: &[Option<i64>], v: Value, ty: Type) -> Option<u64> {
        consts.get(v.index()).copied().flatten().map(|imm| unsigned(imm, ty)).or_else(|| match self.op(func, v) {
            Some(&Op::Const(imm)) => Some(unsigned(imm, ty)),
            _ => self.constant(v),
        })
    }

    fn add_step(&self, func: &Function, consts: &[Option<i64>], arg: Value, p: Value, ty: Type) -> Option<u64> {
        let &Op::Binary(Binary::Iadd, x, y) = self.op(func, arg)? else { return None };
        let other = if x == p { y } else if y == p { x } else { return None };
        self.const_val(func, consts, other, ty).filter(|&k| k > 0)
    }

    fn edge_bound(&self, func: &Function, info: &CfgInfo, (block, pred): (Block, Block), (val, upper): (Value, bool)) -> Option<u64> {
        let pick = |a: Option<u64>, b: Option<u64>| match (a, b) {
            (Some(x), Some(y)) => Some(if upper { x.min(y) } else { x.max(y) }),
            _ => a.or(b),
        };
        let mut best = None;
        if let Some(&Op::Brif(cond, then, _, otherwise, _)) = func.last(pred) && then != otherwise && (then == block || otherwise == block) {
            best = pick(best, self.proves_bound(func, cond, then == block, (val, upper), 0));
        }
        if let Some(&Op::Brif(cond, then, _, otherwise, _)) = func.last(block) && then != otherwise {
            if info.single[then.index()] == Some(block) && dominates(info.idom, then, pred) {
                best = pick(best, self.proves_bound(func, cond, true, (val, upper), 0));
            } else if info.single[otherwise.index()] == Some(block) && dominates(info.idom, otherwise, pred) {
                best = pick(best, self.proves_bound(func, cond, false, (val, upper), 0));
            }
        }
        best
    }

    fn proves_bound(&self, func: &Function, cond: Value, taken: bool, (val, upper): (Value, bool), depth: u8) -> Option<u64> {
        if depth > 4 { return None; }
        let ty = func.ty(val);
        let combine = |a: Option<u64>, b: Option<u64>| match (a, b) {
            (Some(x), Some(y)) => Some(if upper { x.min(y) } else { x.max(y) }),
            _ => a.or(b),
        };
        match self.op(func, cond) {
            Some(&Op::Binary(Binary::Band, a, b)) if taken => combine(self.proves_bound(func, a, true, (val, upper), depth + 1), self.proves_bound(func, b, true, (val, upper), depth + 1)),
            Some(&Op::Binary(Binary::Bor, a, b)) if !taken => combine(self.proves_bound(func, a, false, (val, upper), depth + 1), self.proves_bound(func, b, false, (val, upper), depth + 1)),
            Some(&Op::Icmp(cc, x, y)) if func.ty(x) == ty => {
                let raw = if taken { cc } else { cc.inverse() };
                let eff = match raw.is_signed() {
                    true if self.of(x).non_negative(ty) && self.of(y).non_negative(ty) => as_unsigned(raw),
                    true => return None,
                    false => raw,
                };
                let (kind, lhs, rhs) = rel_of(eff, x, y);
                match (upper, kind) {
                    (true, RelKind::Eq | RelKind::Ule) if lhs == val => Some(self.of(rhs).hi),
                    (true, RelKind::Ult) if lhs == val => self.of(rhs).hi.checked_sub(1),
                    (false, RelKind::Eq | RelKind::Ule) if rhs == val => Some(self.of(lhs).lo),
                    (false, RelKind::Ult) if rhs == val => (self.of(lhs).lo < ty.unsigned_max()).then(|| self.of(lhs).lo + 1),
                    (true, RelKind::Eq) if rhs == val => Some(self.of(lhs).hi),
                    (false, RelKind::Eq) if lhs == val => Some(self.of(rhs).lo),
                    _ => None,
                }
            }
            _ => None,
        }
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
            Op::StackAddr(..) if ty == Type::I64 => Fact::range(1, u64::MAX - 0x1000_0000),
            Op::Symbol(_) | Op::SlotAddr(_) => Fact::range(1, ty.unsigned_max()),
            Op::Convert(Convert::Uextend, x) => self.of(x),
            Op::Convert(Convert::Ireduce, x) => Some(self.of(x)).filter(|inner| inner.hi <= ty.unsigned_max()).unwrap_or_else(|| Fact::full(ty)),
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
            Binary::Band => Fact::range(0, self.of(x).hi.min(self.of(y).hi)),
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
            Binary::Isub => {
                if let Some(&Op::Binary(Binary::Ushr, base, sh)) = self.op(func, y)
                    && base == x
                {
                    let inner = self.of(x);
                    let s = self.constant(sh).map(|c| (c & u64::from(ty.bits() - 1)) as u32);
                    let (lo, hi) = s.map_or((0, inner.hi), |k| (inner.lo - (inner.lo >> k), inner.hi - (inner.hi >> k)));
                    return Fact::range(lo, hi);
                }
                self.shifted(ty, (x, y), true)
            }
            _ => Fact::full(ty),
        }
    }

    /// The range of `x + y` or `x - y`, when no values in their ranges wrap at the width of `ty`.
    fn shifted(&self, ty: Type, (x, y): (Value, Value), subtract: bool) -> Fact {
        let (a, b) = (self.of(x), self.of(y));
        let moved = match subtract {
            false => a.hi.checked_add(b.hi).filter(|&hi| hi <= ty.unsigned_max()).map(|hi| (a.lo + b.lo, hi)),
            true => (a.lo >= b.hi).then(|| (a.lo - b.hi, a.hi - b.lo)),
        };
        moved.map_or_else(|| Fact::full(ty), |(lo, hi)| Fact::range(lo, hi))
    }
}

/// The conditions known to hold at one point of a function: facts on values that the branches taken to get there give.
#[derive(Clone, Default)]
pub(crate) struct Path {
    facts: Vec<(Value, Fact)>,
    rels: Vec<Rel>,
}

fn rel_of(cc: IntCC, x: Value, y: Value) -> (RelKind, Value, Value) {
    match cc {
        IntCC::Equal => (RelKind::Eq, x, y),
        IntCC::NotEqual => (RelKind::Ne, x, y),
        IntCC::UnsignedLessThan => (RelKind::Ult, x, y),
        IntCC::UnsignedGreaterThan => (RelKind::Ult, y, x),
        IntCC::UnsignedLessThanOrEqual => (RelKind::Ule, x, y),
        IntCC::UnsignedGreaterThanOrEqual => (RelKind::Ule, y, x),
        IntCC::SignedLessThan => (RelKind::Slt, x, y),
        IntCC::SignedGreaterThan => (RelKind::Slt, y, x),
        IntCC::SignedLessThanOrEqual => (RelKind::Sle, x, y),
        IntCC::SignedGreaterThanOrEqual => (RelKind::Sle, y, x),
    }
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

    /// Records that the branch on `condition` went the `taken` way: the condition is nonzero or zero, and the integer compare or boolean op that defines it held or failed.
    pub(crate) fn assume(&mut self, func: &Function, ranges: &Ranges, condition: Value, taken: bool) {
        self.assume_depth(func, ranges, condition, taken, 0);
    }

    fn assume_depth(&mut self, func: &Function, ranges: &Ranges, condition: Value, taken: bool, depth: u8) {
        if depth > 4 {
            return;
        }
        let ty = func.ty(condition);
        if !ty.is_small_int() {
            return;
        }
        if let Some(truth) = compared_range(IntCC::NotEqual, ty, &Fact::range(0, 0), taken) {
            self.add(ranges, condition, truth);
        }
        match ranges.op(func, condition) {
            Some(&Op::Binary(Binary::Band, a, b)) if taken => {
                self.assume_depth(func, ranges, a, true, depth + 1);
                self.assume_depth(func, ranges, b, true, depth + 1);
            }
            Some(&Op::Binary(Binary::Bor, a, b)) if !taken => {
                self.assume_depth(func, ranges, a, false, depth + 1);
                self.assume_depth(func, ranges, b, false, depth + 1);
            }
            Some(&Op::Icmp(cc, x, y)) => {
                let operand_ty = func.ty(x);
                if !operand_ty.is_small_int() {
                    return;
                }
                let (left, right) = (self.fact(ranges, x), self.fact(ranges, y));
                let order = match cc.is_signed() {
                    true if left.non_negative(operand_ty) && right.non_negative(operand_ty) => as_unsigned(cc),
                    _ => cc,
                };
                if !order.is_signed() {
                    if let Some(fact) = compared_range(order, operand_ty, &right, taken) {
                        self.add(ranges, x, fact);
                    }
                    if let Some(fact) = compared_range(swapped(order), operand_ty, &left, taken) {
                        self.add(ranges, y, fact);
                    }
                }
                if x != y {
                    let eff = if taken { order } else { order.inverse() };
                    let (kind, lhs, rhs) = rel_of(eff, x, y);
                    let rel = Rel { kind, lhs, rhs, ty: operand_ty };
                    if !self.rels.contains(&rel) {
                        if self.rels.len() >= MAX_RELS {
                            self.rels.remove(0);
                        }
                        self.rels.push(rel);
                    }
                }
            }
            _ => {}
        }
    }

    fn reaches(&self, ty: Type, start: Value, target: Value, (strict_k, weak_k): (RelKind, RelKind), require_strict: bool) -> bool {
        if start == target {
            return !require_strict;
        }
        let mut seen: Vec<(Value, bool)> = vec![(start, false)];
        let mut idx = 0;
        while idx < seen.len() {
            let (cur, cur_strict) = seen[idx];
            idx += 1;
            for &r in &self.rels {
                if r.ty != ty {
                    continue;
                }
                let (next, step_strict) = if r.kind == RelKind::Eq && r.lhs == cur {
                    (r.rhs, false)
                } else if r.kind == RelKind::Eq && r.rhs == cur {
                    (r.lhs, false)
                } else if r.lhs == cur && r.kind == strict_k {
                    (r.rhs, true)
                } else if r.lhs == cur && r.kind == weak_k {
                    (r.rhs, false)
                } else {
                    continue;
                };
                let next_strict = cur_strict || step_strict;
                if next == target && (!require_strict || next_strict) {
                    return true;
                }
                if let Some(entry) = seen.iter_mut().find(|(v, _)| *v == next) {
                    if next_strict && !entry.1 {
                        entry.1 = true;
                        seen.push((next, true));
                    }
                } else {
                    seen.push((next, next_strict));
                }
            }
        }
        false
    }

    fn proves(&self, func: &Function, ranges: &Ranges, cc: IntCC, ty: Type, x: Value, y: Value) -> bool {
        if x == y {
            return holds(cc, ty, 0, 0);
        }
        let (kind, a, b) = rel_of(cc, x, y);
        match kind {
            RelKind::Eq => self.reaches(ty, a, b, (RelKind::Eq, RelKind::Eq), false),
            RelKind::Ult => {
                if self.reaches(ty, a, b, (RelKind::Ult, RelKind::Ule), true) {
                    return true;
                }
                if let Some(&Op::Binary(Binary::Iadd, p, q)) = ranges.op(func, b) {
                    for (base, c) in [(p, q), (q, p)] {
                        if let Some(k) = ranges.constant(c)
                            && k >= 1
                            && ranges.of(base).hi.checked_add(k).is_some_and(|hi| hi <= ty.unsigned_max())
                            && self.reaches(ty, a, base, (RelKind::Ult, RelKind::Ule), false)
                        {
                            return true;
                        }
                    }
                }
                if let Some(&Op::Binary(Binary::Isub, base, c)) = ranges.op(func, a) {
                    if let Some(k) = ranges.constant(c)
                        && self.fact(ranges, base).lo >= k
                        && self.reaches(ty, base, b, (RelKind::Ult, RelKind::Ule), k == 0)
                    {
                        return true;
                    }
                    if self.reaches(ty, c, base, (RelKind::Ult, RelKind::Ule), false)
                        && self.reaches(ty, base, b, (RelKind::Ult, RelKind::Ule), self.fact(ranges, c).lo == 0)
                    {
                        return true;
                    }
                    if let Some(&Op::Binary(Binary::Isub, outer, k_val)) = ranges.op(func, base)
                        && ranges.constant(k_val) == Some(1)
                        && self.reaches(ty, c, outer, (RelKind::Ult, RelKind::Ule), true)
                        && self.reaches(ty, outer, b, (RelKind::Ult, RelKind::Ule), false)
                    {
                        return true;
                    }
                }
                false
            }
            RelKind::Ule => {
                self.reaches(ty, a, b, (RelKind::Ult, RelKind::Ule), false)
                    || (matches!(ranges.op(func, a), Some(&Op::Binary(Binary::Isub, base, c)) if ranges.constant(c).is_some_and(|k| self.fact(ranges, base).lo >= k) && self.reaches(ty, base, b, (RelKind::Ult, RelKind::Ule), false)))
            }
            RelKind::Slt => self.reaches(ty, a, b, (RelKind::Slt, RelKind::Sle), true),
            RelKind::Sle => self.reaches(ty, a, b, (RelKind::Slt, RelKind::Sle), false),
            RelKind::Ne => {
                self.strictly_ordered(ty, a, b)
                    || [(a, b), (b, a)].into_iter().any(|(sub, zero)| {
                        self.fact(ranges, zero).exact() == Some(0)
                            && matches!(ranges.op(func, sub), Some(&Op::Binary(Binary::Isub, lhs, rhs)) if self.strictly_ordered(ty, lhs, rhs))
                    })
                    || self.rels.iter().any(|r| {
                        r.kind == RelKind::Ne
                            && r.ty == ty
                            && ((self.reaches(ty, a, r.lhs, (RelKind::Eq, RelKind::Eq), false) && self.reaches(ty, b, r.rhs, (RelKind::Eq, RelKind::Eq), false))
                                || (self.reaches(ty, a, r.rhs, (RelKind::Eq, RelKind::Eq), false) && self.reaches(ty, b, r.lhs, (RelKind::Eq, RelKind::Eq), false)))
                    })
            }
        }
    }

    fn strictly_ordered(&self, ty: Type, a: Value, b: Value) -> bool {
        self.reaches(ty, a, b, (RelKind::Ult, RelKind::Ule), true)
            || self.reaches(ty, b, a, (RelKind::Ult, RelKind::Ule), true)
            || self.reaches(ty, a, b, (RelKind::Slt, RelKind::Sle), true)
            || self.reaches(ty, b, a, (RelKind::Slt, RelKind::Sle), true)
    }

    fn decide_rel(&self, func: &Function, ranges: &Ranges, cc: IntCC, ty: Type, x: Value, y: Value) -> Option<bool> {
        if self.proves(func, ranges, cc, ty, x, y) {
            Some(true)
        } else if self.proves(func, ranges, cc.inverse(), ty, x, y) {
            Some(false)
        } else {
            None
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

/// Settles the compares, traps and branches that the facts on entry to each block and its earlier instructions decide: a settled compare's result becomes a constant in `consts`, a trap that cannot fire goes, and a settled branch becomes a jump. A block's facts on entry are its only predecessor's facts on exit plus that edge's branch. Returns whether anything changed.
pub(crate) fn settle(func: &mut Function, defs: &[Option<Inst>], consts: &mut [Option<i64>]) -> bool {
    let Some(entry) = func.entry else { return false };
    let cfg = func.cfg();
    let single: Vec<Option<Block>> = (0..func.blocks.len()).map(|b| cfg.single_pred(Block(b as u32))).collect();
    let (mut decided, mut compares) = (Vec::new(), Vec::new());
    {
        let ranges = Ranges::new(func, (defs, consts), (&cfg.rpo, &single, &cfg.idom));
        let mut exits: Vec<Option<Path>> = vec![None; func.blocks.len()];
        for &block in &cfg.rpo {
            let mut path = match single[block.index()].and_then(|pred| exits[pred.index()].as_ref().map(|exit| (pred, exit.clone()))) {
                Some((pred, mut path)) => {
                    path.follow(func, &ranges, pred, block);
                    path
                }
                None if block != entry => cfg.idom[block.index()].and_then(|d| exits[d.index()].clone()).unwrap_or_default(),
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
            if let Some(known) = decide(cc, func.ty(x), &path.fact(ranges, x), &path.fact(ranges, y)).or_else(|| path.decide_rel(func, ranges, cc, func.ty(x), x, y)) {
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

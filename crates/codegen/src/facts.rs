use crate::cfg::{Cfg, dominates};
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

/// The fact on `x` that `x cc k` being `taken` gives, for an equality or unsigned `cc`.
fn compared(cc: IntCC, ty: Type, k: u64, taken: bool) -> Option<Fact> {
    let cc = if taken { cc } else { cc.inverse() };
    let max = ty.unsigned_max();
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
        for &block in order {
            ranges.refine_params(func, consts, (&incoming, &complete, single, idom), block);
            for inst in &func.blocks[block.index()].insts {
                let data = &func.insts[inst.index()];
                let Some(result) = data.result.filter(|result| func.ty(*result).is_small_int()) else { continue };
                let fact = match consts.get(result.index()).copied().flatten() {
                    Some(imm) => Fact::constant(func.ty(result), imm),
                    None => ranges.derive(func, &data.op, func.ty(result)),
                };
                ranges.table[result.index()] = (fact.lo, fact.hi);
            }
        }
        ranges
    }

    fn refine_params(&mut self, func: &Function, consts: &[Option<i64>], (incoming, complete, single, idom): (&[Vec<(Block, &[Value])>], &[bool], &[Option<Block>], &[Option<Block>]), block: Block) {
        if !complete[block.index()] || incoming[block.index()].is_empty() {
            return;
        }
        for (k, &p) in func.blocks[block.index()].params.iter().enumerate() {
            let ty = func.ty(p);
            if !ty.is_small_int() {
                continue;
            }
            let (mut min_lo, mut has_outside, mut valid) = (u64::MAX, false, true);
            for &(pred, args) in &incoming[block.index()] {
                let Some(&arg) = args.get(k) else {
                    valid = false;
                    break;
                };
                if let Some(step) = self.add_step(func, consts, arg, p, ty) {
                    if !self.edge_proves_no_wrap(func, single, idom, (block, pred), (p, ty, step)) {
                        valid = false;
                        break;
                    }
                } else {
                    has_outside = true;
                    min_lo = min_lo.min(self.of(arg).lo);
                }
            }
            if valid && has_outside && min_lo > 0 && min_lo <= ty.unsigned_max() {
                self.table[p.index()].0 = min_lo;
            }
        }
    }

    fn add_step(&self, func: &Function, consts: &[Option<i64>], arg: Value, p: Value, ty: Type) -> Option<u64> {
        let &Op::Binary(Binary::Iadd, x, y) = self.op(func, arg)? else {
            return None;
        };
        let other = if x == p { y } else if y == p { x } else { return None };
        let k = consts.get(other.index()).copied().flatten().map(|imm| unsigned(imm, ty)).or_else(|| match self.op(func, other) {
            Some(&Op::Const(imm)) => Some(unsigned(imm, ty)),
            _ => self.constant(other),
        })?;
        (k > 0).then_some(k)
    }

    fn edge_proves_no_wrap(&self, func: &Function, single: &[Option<Block>], idom: &[Option<Block>], (block, pred): (Block, Block), (p, ty, step): (Value, Type, u64)) -> bool {
        if let Some(&Op::Brif(cond, then, _, otherwise, _)) = func.last(pred)
            && then != otherwise
            && (then == block || otherwise == block)
            && self.proves_no_wrap(func, cond, then == block, p, ty, step, 0)
        {
            return true;
        }
        if let Some(&Op::Brif(cond, then, _, otherwise, _)) = func.last(block)
            && then != otherwise
        {
            let via_then = single[then.index()] == Some(block) && dominates(idom, then, pred) && self.proves_no_wrap(func, cond, true, p, ty, step, 0);
            let via_else = single[otherwise.index()] == Some(block) && dominates(idom, otherwise, pred) && self.proves_no_wrap(func, cond, false, p, ty, step, 0);
            return via_then || via_else;
        }
        false
    }

    fn proves_no_wrap(&self, func: &Function, cond: Value, taken: bool, p: Value, ty: Type, step: u64, depth: u8) -> bool {
        if depth > 4 {
            return false;
        }
        let Some(max_p) = ty.unsigned_max().checked_sub(step) else {
            return false;
        };
        match self.op(func, cond) {
            Some(&Op::Binary(Binary::Band, a, b)) if taken => {
                self.proves_no_wrap(func, a, true, p, ty, step, depth + 1) || self.proves_no_wrap(func, b, true, p, ty, step, depth + 1)
            }
            Some(&Op::Binary(Binary::Bor, a, b)) if !taken => {
                self.proves_no_wrap(func, a, false, p, ty, step, depth + 1) || self.proves_no_wrap(func, b, false, p, ty, step, depth + 1)
            }
            Some(&Op::Icmp(cc, x, y)) => {
                let eff = if taken { cc } else { cc.inverse() };
                let (kind, lhs, rhs) = rel_of(eff, x, y);
                if lhs != p {
                    return false;
                }
                let rhs_hi = self.of(rhs).hi;
                match kind {
                    RelKind::Ult => rhs_hi > 0 && rhs_hi - 1 <= max_p,
                    RelKind::Ule => rhs_hi <= max_p,
                    _ => false,
                }
            }
            _ => false,
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
            false => inner.lo.checked_add(k).zip(inner.hi.checked_add(k)).filter(|(_, hi)| *hi <= ty.unsigned_max()),
            true => inner.lo.checked_sub(k).zip(inner.hi.checked_sub(k)),
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
        if let Some(truth) = compared(IntCC::NotEqual, ty, 0, taken) {
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
                    if let Some(fact) = right.exact().and_then(|k| compared(order, operand_ty, k, taken)) {
                        self.add(ranges, x, fact);
                    }
                    if let Some(fact) = left.exact().and_then(|k| compared(swapped(order), operand_ty, k, taken)) {
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
    let cfg = Cfg::new(func);
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

mod backprop;
mod live;
mod loops;
mod rel;

use crate::cfg::{Cfg, dominates};
use crate::fold::{extend, holds};
use crate::ir::{Binary, Block, Convert, Function, Inst, IntCC, Op, Type, Value};
use loops::{Env, ExtraPath};
use rel::{Path, RelKind, cond_truth, fits_truth, rel_of};

/// The most facts a path keeps; past it the oldest goes.
pub(super) const MAX_FACTS: usize = 64;

/// The most constants one fact records a value as differing from.
const MAX_NOT: usize = 32;

/// The most symbolic relations a path keeps; past it the oldest goes.
pub(super) const MAX_RELS: usize = 64;

/// What is known about an integer value: its unsigned range at its type's width, and constants it differs from.
#[derive(Clone, PartialEq, Debug)]
pub(crate) struct Fact {
    pub(super) lo: u64,
    pub(super) hi: u64,
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

    pub(super) fn range(lo: u64, hi: u64) -> Self {
        Self { lo, hi, not: Vec::new() }
    }

    /// The single value the fact allows, when it allows one.
    pub(crate) fn exact(&self) -> Option<u64> {
        (self.lo == self.hi).then_some(self.lo)
    }

    /// What both facts say.
    pub(super) fn meet(mut self, other: &Fact) -> Self {
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
    pub(super) fn admits(&self, value: u64) -> bool {
        (self.lo..=self.hi).contains(&value) && !self.not.contains(&value)
    }

    /// Whether the fact holds for both signed and unsigned readings alike: its range sits below the sign bit of `ty`.
    pub(super) fn non_negative(&self, ty: Type) -> bool {
        self.hi <= ty.unsigned_max() >> 1
    }
}

/// The unsigned comparison that agrees with `cc` on operands below the sign bit.
pub(super) fn as_unsigned(cc: IntCC) -> IntCC {
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
pub(super) fn compared_range(cc: IntCC, ty: Type, rhs: &Fact, taken: bool) -> Option<Fact> {
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
pub(super) fn swapped(cc: IntCC) -> IntCC {
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

fn collect_incoming(func: &Function) -> (Vec<Vec<(Block, &[Value])>>, Vec<bool>) {
    let mut incoming: Vec<Vec<(Block, &[Value])>> = vec![Vec::new(); func.blocks.len()];
    let mut complete = vec![true; func.blocks.len()];
    for pred_idx in 0..func.blocks.len() {
        let pred = Block(pred_idx as u32);
        let mut edge = |target: Block, args| {
            if !func.blocks[target.index()].params.is_empty() {
                incoming[target.index()].push((pred, args));
            }
        };
        match func.last(pred) {
            Some(Op::Jump(target, args)) => edge(*target, args.as_slice()),
            Some(Op::Brif(_, then, then_args, otherwise, else_args)) => {
                edge(*then, then_args.as_slice());
                edge(*otherwise, else_args.as_slice());
            }
            _ => {
                for target in func.successors(pred) {
                    complete[target.index()] = false;
                }
            }
        }
    }
    (incoming, complete)
}

impl<'a> Ranges<'a> {
    /// The ranges of `func`'s values over the blocks of `order`, with the constants `consts` an earlier fold found.
    pub(crate) fn new(func: &Function, parts: (&'a [Option<Inst>], &[Option<i64>]), blocks: (&[Block], &[Option<Block>], &[Option<Block>])) -> Self {
        let (incoming, complete) = collect_incoming(func);
        Self::over(func, parts, blocks, (&incoming, &complete))
    }

    fn over(func: &Function, (defs, consts): (&'a [Option<Inst>], &[Option<i64>]), (order, single, idom): (&[Block], &[Option<Block>], &[Option<Block>]), (incoming, complete): (&[Vec<(Block, &[Value])>], &[bool])) -> Self {
        let mut table: Vec<(u64, u64)> = func.types.iter().map(|ty| (0, ty.unsigned_max())).collect();
        if let Some(entry) = func.entry {
            for (&param, abi) in func.blocks[entry.index()].params.iter().zip(&func.signature.params) {
                if abi.deref_bytes > 0 && func.ty(param) == Type::I64 {
                    table[param.index()] = (1, u64::MAX - u64::from(abi.deref_bytes - 1));
                }
            }
        }
        let mut ranges = Self { defs, table };
        let info = CfgInfo { consts, incoming, complete, single, idom };
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

    pub(super) fn op<'f>(&self, func: &'f Function, value: Value) -> Option<&'f Op> {
        self.defs.get(value.index()).copied().flatten().map(|inst| &func.insts[inst.index()].op)
    }

    /// The range of `value` its definition proves.
    pub(crate) fn of(&self, value: Value) -> Fact {
        let (lo, hi) = self.table[value.index()];
        Fact::range(lo, hi)
    }

    /// The constant `value` holds, read unsigned at its width, when its range is one value.
    pub(super) fn constant(&self, value: Value) -> Option<u64> {
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
            Binary::Ishl => match self.constant(y) {
                Some(count) => {
                    let (inner, shift) = (self.of(x), (count & u64::from(ty.bits() - 1)) as u32);
                    if inner.hi <= (ty.unsigned_max() >> shift) {
                        Fact::range(inner.lo << shift, inner.hi << shift)
                    } else {
                        Fact::full(ty)
                    }
                }
                None => Fact::full(ty),
            },
            Binary::Udiv => match self.constant(y) {
                Some(divisor) if divisor > 0 => {
                    let inner = self.of(x);
                    Fact::range(inner.lo / divisor, inner.hi / divisor)
                }
                _ => Fact::full(ty),
            },
            Binary::Urem => match self.constant(y) {
                Some(divisor) if divisor > 0 => Fact::range(0, divisor - 1),
                _ => Fact::full(ty),
            },
            Binary::Imul => {
                let (a, b) = (self.of(x), self.of(y));
                a.hi.checked_mul(b.hi)
                    .filter(|&hi| hi <= ty.unsigned_max())
                    .map_or_else(|| Fact::full(ty), |hi| Fact::range(a.lo * b.lo, hi))
            }
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

#[derive(Clone, Copy)]
struct SelectRec {
    res: Value,
    cond: Value,
    t_val: Value,
    f_val: Value,
    block: Block,
}

#[derive(Clone, Default)]
struct DomState {
    aliases: Vec<(Value, Value)>,
}

impl DomState {
    fn resolve(&self, mut v: Value) -> Value {
        for _ in 0..4 {
            let Some(&(_, to)) = self.aliases.iter().rev().find(|&&(from, _)| from == v) else { break };
            v = to;
        }
        v
    }

    fn record_alias(&mut self, from: Value, to: Value) -> bool {
        let to = self.resolve(to);
        if to == from || self.resolve(from) == to {
            return false;
        }
        if self.aliases.len() >= MAX_RELS {
            self.aliases.remove(0);
        }
        self.aliases.push((from, to));
        true
    }

    fn propagate_selects(
        &mut self,
        func: &Function,
        (cfg, ranges, env, path, extra): (&Cfg, &Ranges, &Env, &mut Path, &ExtraPath),
        selects: &[SelectRec],
        block: Block,
        seen: &[bool],
    ) {
        for _ in 0..3 {
            let mut progress = false;
            for &rec in selects {
                if !seen[rec.res.index()] || self.resolve(rec.res) != rec.res {
                    continue;
                }
                let cond = self.resolve(rec.cond);
                let t_val = self.resolve(rec.t_val);
                let f_val = self.resolve(rec.f_val);
                if let Some(taken) = cond_truth(func, ranges, path, cond).or_else(|| extra.cond_truth(func, (cfg, ranges, env, path, block), cond, 0)) {
                    progress |= self.record_alias(rec.res, if taken { t_val } else { f_val });
                    continue;
                }
                let opt_fact = path.fact(ranges, rec.res);
                let t_exact = path.fact(ranges, t_val).exact();
                let f_exact = path.fact(ranges, f_val).exact();
                if f_exact.is_some_and(|k| !opt_fact.admits(k)) {
                    progress |= self.record_alias(rec.res, t_val);
                    path.assume(func, ranges, cond, true);
                    if f_exact == Some(0) {
                        path.assume(func, ranges, t_val, true);
                    }
                } else if t_exact.is_some_and(|k| !opt_fact.admits(k)) {
                    progress |= self.record_alias(rec.res, f_val);
                    path.assume(func, ranges, cond, false);
                    if t_exact == Some(0) {
                        path.assume(func, ranges, f_val, true);
                    }
                } else if let Some(k) = opt_fact.exact() {
                    if f_exact == Some(k) && !path.fact(ranges, t_val).admits(k) {
                        progress |= self.record_alias(rec.res, f_val);
                        path.assume(func, ranges, cond, false);
                    } else if t_exact == Some(k) && !path.fact(ranges, f_val).admits(k) {
                        progress |= self.record_alias(rec.res, t_val);
                        path.assume(func, ranges, cond, true);
                    }
                }
            }
            if !progress {
                break;
            }
        }
    }
}

fn collect_selects(func: &Function) -> Option<Vec<SelectRec>> {
    let (mut selects, mut open) = (Vec::new(), false);
    for (b_idx, block) in func.blocks.iter().enumerate() {
        for &inst in &block.insts {
            let data = &func.insts[inst.index()];
            open |= matches!(data.op, Op::Icmp(..) | Op::Fits(..) | Op::Trapz(_) | Op::Brif(..));
            if let (Op::Select(cond, t_val, f_val), Some(res)) = (&data.op, data.result)
                && func.ty(res).is_small_int()
            {
                selects.push(SelectRec { res, cond: *cond, t_val: *t_val, f_val: *f_val, block: Block(b_idx as u32) });
            }
        }
    }
    (open || !selects.is_empty()).then_some(selects)
}

/// Settles the compares, traps and branches that the facts on entry to each block and its earlier instructions decide: a settled compare's result becomes a constant in `consts`, a trap that cannot fire goes, and a settled branch becomes a jump. A block's facts on entry are its only predecessor's facts on exit plus that edge's branch. Returns whether anything changed.
pub(crate) fn settle(func: &mut Function, defs: &[Option<Inst>], consts: &mut [Option<i64>]) -> bool {
    if func.coroutine.is_some() {
        return false;
    }
    let Some(entry) = func.entry else { return false };
    let Some(selects) = collect_selects(func) else { return false };
    let cfg = func.cfg();
    let single: Vec<Option<Block>> = (0..func.blocks.len()).map(|b| cfg.single_pred(Block(b as u32))).collect();
    let (mut decided, mut compares, mut rewrites) = (Vec::new(), Vec::new(), Vec::new());
    {
        let (incoming, complete) = collect_incoming(func);
        let ranges = Ranges::over(func, (defs, consts), (&cfg.rpo, &single, &cfg.idom), (&incoming, &complete));
        let def_blocks = std::cell::OnceCell::new();
        let mut live = live::Live::default();
        let loops = cfg.loops();
        let mut env = Env::default();
        let mut exits: Vec<Option<Path>> = vec![None; func.blocks.len()];
        let mut extra_exits: Vec<Option<ExtraPath>> = vec![None; func.blocks.len()];
        let mut dom_exits: Vec<Option<DomState>> = vec![None; func.blocks.len()];
        let mut seen = vec![false; func.types.len()];
        for &block in &cfg.rpo {
            for &param in &func.blocks[block.index()].params {
                seen[param.index()] = true;
            }
            live.enter(&cfg, &selects, block);
            for each in &loops.list {
                if each.head == block {
                    env.collect_loop(func, (&cfg, &ranges, &incoming, &complete, &exits, &def_blocks), each);
                }
            }
            let follow_pred = single[block.index()].or_else(|| {
                (block != entry)
                    .then(|| {
                        let mut outside = cfg.preds_of(block).iter().copied().filter(|&p| cfg.reachable[p.index()] && !cfg.dominates(block, p));
                        let first = outside.next()?;
                        (outside.next().is_none() && cfg.idom[block.index()] == Some(first)).then_some(first)
                    })
                    .flatten()
            });
            let (mut path, mut extra, mut dom) = match follow_pred.and_then(|pred| {
                exits[pred.index()].as_ref().map(|exit| {
                    (
                        pred,
                        exit.clone(),
                        extra_exits[pred.index()].clone().unwrap_or_default(),
                        dom_exits[pred.index()].clone().unwrap_or_default(),
                    )
                })
            }) {
                Some((pred, mut path, mut extra, mut dom)) => {
                    path.follow(func, &ranges, pred, block);
                    extra.follow(func, (&cfg, &ranges, &env, &path), pred, block);
                    dom.propagate_selects(func, (&cfg, &ranges, &env, &mut path, &extra), &live.recs, block, &seen);
                    (path, extra, dom)
                }
                None if block != entry => (
                    cfg.idom[block.index()].and_then(|d| exits[d.index()].clone()).unwrap_or_default(),
                    cfg.idom[block.index()].and_then(|d| extra_exits[d.index()].clone()).unwrap_or_default(),
                    cfg.idom[block.index()].and_then(|d| dom_exits[d.index()].clone()).unwrap_or_default(),
                ),
                None => (Path::default(), ExtraPath::default(), DomState::default()),
            };
            for inv in &env.rel_invs {
                if inv.head == block {
                    let kind = if inv.strict { RelKind::Ult } else { RelKind::Ule };
                    path.push_rel_kind(kind, inv.param, inv.bound, inv.ty);
                }
            }
            for &inst in &func.blocks[block.index()].insts {
                settle_one(func, (&cfg, &ranges, &env, &mut path, &mut extra, &mut dom, &live.recs, block, &seen), inst, (&mut decided, &mut compares, &mut rewrites));
                let data = &func.insts[inst.index()];
                if let Some(res) = data.result {
                    seen[res.index()] = true;
                }
                if let Op::Call { results, .. } = &data.op {
                    for &res in results {
                        seen[res.index()] = true;
                    }
                }
            }
            exits[block.index()] = Some(path);
            extra_exits[block.index()] = Some(extra);
            dom_exits[block.index()] = Some(dom);
        }
    }
    let settled = !compares.is_empty();
    for (result, known) in compares {
        consts[result.index()] = Some(i64::from(known));
    }
    apply(func, &decided, &rewrites) | settled
}

/// Settles `inst` under `path` and moves the path past it.
fn settle_one(
    func: &Function,
    (cfg, ranges, env, path, extra, dom, selects, block, seen): (&Cfg, &Ranges, &Env, &mut Path, &mut ExtraPath, &mut DomState, &[SelectRec], Block, &[bool]),
    inst: Inst,
    (decided, compares, rewrites): (&mut Vec<(Inst, Option<bool>)>, &mut Vec<(Value, bool)>, &mut Vec<(Inst, Value, Value)>),
) {
    let data = &func.insts[inst.index()];
    if !dom.aliases.is_empty() {
        data.op.for_operands(|v| {
            let mapped = dom.resolve(v);
            if mapped != v {
                rewrites.push((inst, v, mapped));
            }
        });
    }
    match (&data.op, data.result) {
        (&Op::Icmp(cc, raw_x, raw_y), Some(result)) if ranges.constant(result).is_none() && func.ty(raw_x).is_small_int() => {
            let (x, y) = (dom.resolve(raw_x), dom.resolve(raw_y));
            let ty = func.ty(x);
            if let Some(known) = decide(cc, ty, &path.refined_fact(func, ranges, x, 0), &path.refined_fact(func, ranges, y, 0))
                .or_else(|| path.decide_rel(func, ranges, cc, ty, x, y))
                .or_else(|| extra.decide(func, (cfg, ranges, env, path, block), cc, x, y))
            {
                compares.push((result, known));
            }
        }
        (&Op::Fits(op, signed, raw_x, raw_y), Some(result)) if ranges.constant(result).is_none() && func.ty(raw_x).is_small_int() => {
            let (x, y) = (dom.resolve(raw_x), dom.resolve(raw_y));
            if let Some(known) = fits_truth(func, ranges, path, op, signed, x, y) {
                compares.push((result, known));
            }
        }
        (&Op::Select(raw_cond, raw_t, raw_f), Some(result)) if func.ty(result).is_small_int() => {
            let cond = dom.resolve(raw_cond);
            if let Some(taken) = cond_truth(func, ranges, path, cond)
                .or_else(|| extra.cond_truth(func, (cfg, ranges, env, path, block), cond, 0))
            {
                let chosen = dom.resolve(if taken { raw_t } else { raw_f });
                dom.record_alias(result, chosen);
            }
        }
        (&Op::Trapz(raw_cond), _) => {
            let condition = dom.resolve(raw_cond);
            if cond_truth(func, ranges, path, condition)
                .or_else(|| extra.cond_truth(func, (cfg, ranges, env, path, block), condition, 0))
                == Some(true)
            {
                decided.push((inst, None));
            }
            path.assume(func, ranges, condition, true);
            extra.assume(func, (cfg, ranges, env, path, block), condition, true, 0);
            dom.propagate_selects(func, (cfg, ranges, env, path, extra), selects, block, seen);
        }
        (&Op::Brif(raw_cond, ..), _) if ranges.constant(raw_cond).is_none() => {
            let condition = dom.resolve(raw_cond);
            if let Some(taken) = cond_truth(func, ranges, path, condition)
                .or_else(|| extra.cond_truth(func, (cfg, ranges, env, path, block), condition, 0))
            {
                decided.push((inst, Some(taken)));
            }
        }
        _ => {}
    }
}

/// Whether a value with `fact` is nonzero, when the fact settles it.
pub(super) fn truth(fact: &Fact) -> Option<bool> {
    match fact.exact() {
        Some(known) => Some(known != 0),
        None => (!fact.admits(0)).then_some(true),
    }
}

/// Drops each trap in `decided`, turns each branch into a jump the way it goes, and rewrites dominated operands.
fn apply(func: &mut Function, decided: &[(Inst, Option<bool>)], rewrites: &[(Inst, Value, Value)]) -> bool {
    if decided.is_empty() && rewrites.is_empty() { return false; }
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
    for &(inst, old, new) in rewrites {
        if !dropped[inst.index()] {
            func.insts[inst.index()].op.map_operands(|v| if v == old { new } else { v });
        }
    }
    for block in &mut func.blocks {
        block.insts.retain(|inst| !dropped[inst.index()]);
    }
    !decided.is_empty() || !rewrites.is_empty()
}

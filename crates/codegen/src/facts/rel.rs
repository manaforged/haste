use crate::fold::{extend, holds};
use crate::ir::{Binary, Block, Function, IntCC, Op, Type, Value};

use super::backprop::Uppers;
use super::{Fact, MAX_FACTS, MAX_RELS, Ranges, compared_range, decide, truth, unsigned_order};

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(super) enum RelKind {
    Eq = 0,
    Ne = 1,
    Ult = 2,
    Ule = 3,
    Slt = 4,
    Sle = 5,
}

#[inline]
pub(super) const fn kind_bit(kind: RelKind) -> u8 {
    1_u8 << (kind as u8)
}

#[inline]
fn val_bit(v: Value) -> u128 {
    1_u128 << (v.index() & 127)
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct Rel {
    pub(super) kind: RelKind,
    pub(super) lhs: Value,
    pub(super) rhs: Value,
    pub(super) ty: Type,
}

pub(super) const UNSIGNED: (RelKind, RelKind) = (RelKind::Ult, RelKind::Ule);
const SIGNED: (RelKind, RelKind) = (RelKind::Slt, RelKind::Sle);
const EQUAL: (RelKind, RelKind) = (RelKind::Eq, RelKind::Eq);
const WALK_SEEN: usize = 16;
const _: () = assert!(2 * MAX_RELS <= u128::BITS as usize);

pub(super) fn rel_step(r: Rel, cur: Value, ty: Type, (strict_k, weak_k): (RelKind, RelKind)) -> Option<(Value, bool)> {
    if r.ty != ty {
        return None;
    }
    if r.kind == RelKind::Eq && r.lhs == cur {
        Some((r.rhs, false))
    } else if r.kind == RelKind::Eq && r.rhs == cur {
        Some((r.lhs, false))
    } else if r.lhs == cur && r.kind == strict_k {
        Some((r.rhs, true))
    } else if r.lhs == cur && r.kind == weak_k {
        Some((r.rhs, false))
    } else {
        None
    }
}

pub(super) fn rel_of(cc: IntCC, x: Value, y: Value) -> (RelKind, Value, Value) {
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

/// The conditions known to hold at one point of a function: facts on values that the branches taken to get there give.
#[derive(Clone, Default)]
pub(super) struct Path {
    fact_vals: Vec<Value>,
    fact_data: Vec<Fact>,
    fact_mask: u128,
    rels: Vec<Rel>,
    rel_kinds: u8,
    rel_src_mask: u128,
    rel_dst_mask: u128,
    no_wrap_adds: Vec<(Value, Value)>,
    pub(super) uppers: Uppers,
}

/// The values one walk over a path's relations steps to from `start`, as a mask over relation ends.
#[derive(Clone, Copy)]
pub(super) struct Reached {
    start: Value,
    ends: u128,
}

impl Path {
    /// Whether this path has recorded no branch-specific facts, relations, or no-wrap sums.
    pub(super) fn is_empty(&self) -> bool {
        self.fact_vals.is_empty() && self.no_wrap_adds.is_empty()
    }

    /// Whether any recorded relation has a kind in `mask`.
    #[inline]
    pub(super) fn has_rel_kind(&self, mask: u8) -> bool {
        self.rel_kinds & mask != 0
    }

    /// The symbolic relations known here, oldest first.
    pub(super) fn rels(&self) -> &[Rel] {
        &self.rels
    }

    /// Whether any symbolic relation or no-wrap sum is recorded here.
    #[inline]
    pub(super) fn has_rels(&self) -> bool {
        !self.rels.is_empty() || !self.no_wrap_adds.is_empty()
    }

    #[inline]
    fn get(&self, value: Value) -> Option<&Fact> {
        if self.fact_mask & val_bit(value) == 0 {
            return None;
        }
        self.fact_vals
            .iter()
            .rposition(|&at| at == value)
            .map(|pos| &self.fact_data[pos])
    }

    /// Everything known about `value` here: its definition's range and the path's facts.
    #[inline]
    pub(super) fn fact(&self, ranges: &Ranges, value: Value) -> Fact {
        let defined = ranges.of(value);
        match self.get(value) {
            Some(fact) => defined.meet(fact),
            None => defined,
        }
    }

    /// Records `fact` on `value`, forgetting the oldest fact past `MAX_FACTS`.
    pub(super) fn add(&mut self, ranges: &Ranges, value: Value, fact: Fact) -> bool {
        let cur = self.fact(ranges, value);
        if cur.subsumes(&fact) {
            return false;
        }
        let merged = cur.meet(&fact);
        let bit = val_bit(value);
        if self.fact_mask & bit != 0
            && let Some(pos) = self.fact_vals.iter().position(|&at| at == value)
        {
            self.fact_vals.remove(pos);
            self.fact_data.remove(pos);
            self.fact_vals.push(value);
            self.fact_data.push(merged);
        } else if self.fact_vals.len() >= MAX_FACTS {
            self.fact_vals.remove(0);
            self.fact_data.remove(0);
            self.fact_vals.push(value);
            self.fact_data.push(merged);
            self.fact_mask = self.fact_vals.iter().fold(0_u128, |m, &v| m | val_bit(v));
        } else {
            self.fact_vals.push(value);
            self.fact_data.push(merged);
            self.fact_mask |= bit;
        }
        self.uppers.clear();
        true
    }

    fn push_rel(&mut self, rel: Rel) {
        if !self.rels.contains(&rel) {
            if self.rels.len() >= MAX_RELS {
                self.rels.remove(0);
                self.rels.push(rel);
                self.rebuild_rel_masks();
            } else {
                self.rels.push(rel);
                self.rel_kinds |= kind_bit(rel.kind);
                self.note_rel_masks(rel);
            }
            self.uppers.clear();
        }
    }

    fn note_rel_masks(&mut self, rel: Rel) {
        if rel.kind == RelKind::Eq {
            let both = val_bit(rel.lhs) | val_bit(rel.rhs);
            self.rel_src_mask |= both;
            self.rel_dst_mask |= both;
        } else if rel.kind != RelKind::Ne {
            self.rel_src_mask |= val_bit(rel.lhs);
            self.rel_dst_mask |= val_bit(rel.rhs);
        }
    }

    fn rebuild_rel_masks(&mut self) {
        self.rel_kinds = 0;
        self.rel_src_mask = 0;
        self.rel_dst_mask = 0;
        for i in 0..self.rels.len() {
            let r = self.rels[i];
            self.rel_kinds |= kind_bit(r.kind);
            self.note_rel_masks(r);
        }
    }

    /// Records a symbolic `kind(lhs, rhs)` relation on `ty`.
    pub(super) fn push_rel_kind(&mut self, kind: RelKind, lhs: Value, rhs: Value, ty: Type) {
        self.push_rel(Rel { kind, lhs, rhs, ty });
    }

    /// Returns whether `x + y` was previously proven not to wrap.
    pub(super) fn has_no_wrap_add(&self, x: Value, y: Value) -> bool {
        let pair = if x.index() <= y.index() { (x, y) } else { (y, x) };
        self.no_wrap_adds.contains(&pair)
    }


    /// Records that the branch on `condition` went the `taken` way: the condition is nonzero or zero, and the integer compare or boolean op that defines it held or failed.
    pub(super) fn assume(&mut self, func: &Function, ranges: &Ranges, condition: Value, taken: bool) {
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
            self.add_with_backprop(func, ranges, condition, truth, 0);
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
            Some(&Op::Binary(Binary::Bxor, a, b)) => {
                for (inner, mask) in [(a, b), (b, a)] {
                    if self.fact(ranges, mask).exact() == Some(1) && self.fact(ranges, inner).hi <= 1 {
                        self.assume_depth(func, ranges, inner, !taken, depth + 1);
                        break;
                    }
                }
            }
            Some(&Op::Fits(Binary::Iadd, false, x, y)) if taken => {
                let op_ty = func.ty(x);
                if op_ty.is_small_int() {
                    let pair = if x.index() <= y.index() { (x, y) } else { (y, x) };
                    if !self.no_wrap_adds.contains(&pair) {
                        if self.no_wrap_adds.len() >= MAX_RELS {
                            self.no_wrap_adds.remove(0);
                        }
                        self.no_wrap_adds.push(pair);
                        self.uppers.clear();
                    }
                    if let Some(k) = self.fact(ranges, y).exact() {
                        self.add_with_backprop(func, ranges, x, Fact::range(0, op_ty.unsigned_max() - k), 0);
                    }
                    if let Some(k) = self.fact(ranges, x).exact() {
                        self.add_with_backprop(func, ranges, y, Fact::range(0, op_ty.unsigned_max() - k), 0);
                    }
                }
            }
            Some(&Op::Fits(Binary::Isub, false, x, y)) if taken => {
                let op_ty = func.ty(x);
                if op_ty.is_small_int() {
                    if x != y {
                        self.push_rel(Rel { kind: RelKind::Ule, lhs: y, rhs: x, ty: op_ty });
                    }
                    let fy = self.fact(ranges, y);
                    self.add_with_backprop(func, ranges, x, Fact::range(fy.lo, op_ty.unsigned_max()), 0);
                    let fx = self.fact(ranges, x);
                    self.add_with_backprop(func, ranges, y, Fact::range(0, fx.hi), 0);
                }
            }
            _ => {}
        }
        if let Some((cc, x, y)) = ranges.op(func, condition).and_then(crate::algebra::branch_cmp) {
            let operand_ty = func.ty(x);
            if !operand_ty.is_small_int() {
                return;
            }
            let (left, right) = (self.fact(ranges, x), self.fact(ranges, y));
            let order = unsigned_order(cc, operand_ty, &left, &right).unwrap_or(cc);
            if !order.is_signed() {
                if let Some(fact) = compared_range(order, operand_ty, &right, taken) {
                    self.add_with_backprop(func, ranges, x, fact, 0);
                }
                if let Some(fact) = compared_range(order.swapped(), operand_ty, &left, taken) {
                    self.add_with_backprop(func, ranges, y, fact, 0);
                }
            }
            if x != y {
                let eff = if taken { order } else { order.inverse() };
                let (kind, lhs, rhs) = rel_of(eff, x, y);
                self.push_rel(Rel { kind, lhs, rhs, ty: operand_ty });
            }
        }
    }

    #[inline]
    pub(super) fn steps_from(&self, ty: Type, start: Value, kinds: (RelKind, RelKind)) -> bool {
        let needed = kind_bit(kinds.0) | kind_bit(kinds.1) | kind_bit(RelKind::Eq);
        if self.rel_kinds & needed == 0 || self.rel_src_mask & val_bit(start) == 0 {
            return false;
        }
        self.rels.iter().any(|&r| rel_step(r, start, ty, kinds).is_some())
    }

    #[inline]
    pub(super) fn steps_into(&self, ty: Type, target: Value, kinds: (RelKind, RelKind)) -> bool {
        let needed = kind_bit(kinds.0) | kind_bit(kinds.1) | kind_bit(RelKind::Eq);
        if self.rel_kinds & needed == 0 || self.rel_dst_mask & val_bit(target) == 0 {
            return false;
        }
        self.rels.iter().any(|&r| {
            r.ty == ty
                && ((r.rhs == target && (r.kind == RelKind::Eq || r.kind == kinds.0 || r.kind == kinds.1))
                    || (r.lhs == target && r.kind == RelKind::Eq))
        })
    }

    pub(super) fn reaches(&self, ty: Type, start: Value, target: Value, kinds: (RelKind, RelKind), require_strict: bool) -> bool {
        self.reach(ty, start, target, kinds, require_strict) == Some(true)
    }

    fn reach(&self, ty: Type, start: Value, target: Value, kinds: (RelKind, RelKind), require_strict: bool) -> Option<bool> {
        if start == target {
            return Some(!require_strict);
        }
        let needed = kind_bit(kinds.0) | kind_bit(kinds.1) | kind_bit(RelKind::Eq);
        if self.rel_kinds & needed == 0 || (require_strict && self.rel_kinds & kind_bit(kinds.0) == 0) {
            return None;
        }
        if !self.steps_from(ty, start, kinds) || !self.steps_into(ty, target, kinds) {
            return None;
        }
        let mut hit = None;
        let found = self.walk(ty, start, kinds, |next, next_strict| {
            if next == target {
                if !require_strict || next_strict {
                    return true;
                }
                hit = Some(false);
            }
            false
        });
        if found { Some(true) } else { hit }
    }

    /// Walks once from `start` and keeps every value it steps to.
    pub(super) fn reached(&self, ty: Type, start: Value, kinds: (RelKind, RelKind)) -> Reached {
        if !self.steps_from(ty, start, kinds) {
            return Reached { start, ends: 0 };
        }
        let mut ends = 0_u128;
        self.walk_rels(ty, start, kinds, |idx, next, _| {
            ends |= 1_u128 << (2 * idx + usize::from(next != self.rels[idx].rhs));
            false
        });
        Reached { start, ends }
    }

    /// Whether `target` is the start of `from` or a value its walk stepped to.
    pub(super) fn reaches_in(&self, from: Reached, target: Value) -> bool {
        if from.start == target {
            return true;
        }
        let mut ends = from.ends;
        while ends != 0 {
            let bit = ends.trailing_zeros() as usize;
            let r = self.rels[bit / 2];
            if (if bit % 2 == 0 { r.rhs } else { r.lhs }) == target {
                return true;
            }
            ends &= ends - 1;
        }
        false
    }

    pub(super) fn walk(&self, ty: Type, start: Value, kinds: (RelKind, RelKind), mut visit: impl FnMut(Value, bool) -> bool) -> bool {
        self.walk_rels(ty, start, kinds, |_, next, next_strict| visit(next, next_strict))
    }

    fn walk_rels(&self, ty: Type, start: Value, kinds: (RelKind, RelKind), mut visit: impl FnMut(usize, Value, bool) -> bool) -> bool {
        let (mut seen, mut len) = ([(start, false); WALK_SEEN], 1);
        let mut idx = 0;
        while idx < len {
            let (cur, cur_strict) = seen[idx];
            idx += 1;
            for (pos, &r) in self.rels.iter().enumerate() {
                let Some((next, step_strict)) = rel_step(r, cur, ty, kinds) else {
                    continue;
                };
                let next_strict = cur_strict || step_strict;
                if visit(pos, next, next_strict) {
                    return true;
                }
                if len < WALK_SEEN {
                    if let Some(entry) = seen[..len].iter_mut().find(|(v, _)| *v == next) {
                        if next_strict && !entry.1 {
                            entry.1 = true;
                            seen[len] = (next, true);
                            len += 1;
                        }
                    } else {
                        seen[len] = (next, next_strict);
                        len += 1;
                    }
                }
            }
        }
        false
    }

    fn proves_sub_rel(&self, func: &Function, ranges: &Ranges, ty: Type, a: Value, b: Value, strict: bool) -> bool {
        let Some(&Op::Binary(Binary::Isub, base, c)) = ranges.op(func, a) else { return false };
        if let Some(k) = ranges.constant(c)
            && self.fact(ranges, base).lo >= k
            && self.reaches(ty, base, b, UNSIGNED, strict && k == 0)
        {
            return true;
        }
        if !strict {
            return false;
        }
        if self.reaches(ty, c, base, UNSIGNED, false)
            && self.reaches(ty, base, b, UNSIGNED, self.fact(ranges, c).lo == 0)
        {
            return true;
        }
        if let Some(&Op::Binary(Binary::Isub, outer, k_val)) = ranges.op(func, base)
            && ranges.constant(k_val) == Some(1)
            && self.reaches(ty, c, outer, UNSIGNED, true)
            && self.reaches(ty, outer, b, UNSIGNED, false)
        {
            return true;
        }
        false
    }

    fn known_ne(&self, ty: Type, a: Value, b: Value) -> bool {
        if self.rel_kinds & kind_bit(RelKind::Ne) == 0 {
            return false;
        }
        let (mut from_a, mut from_b) = (None, None);
        self.rels.iter().filter(|r| r.kind == RelKind::Ne && r.ty == ty).any(|r| {
            let mut at_a = |target| a == target || self.reaches_in(*from_a.get_or_insert_with(|| self.reached(ty, a, EQUAL)), target);
            let mut at_b = |target| b == target || self.reaches_in(*from_b.get_or_insert_with(|| self.reached(ty, b, EQUAL)), target);
            (at_a(r.lhs) && at_b(r.rhs)) || (at_a(r.rhs) && at_b(r.lhs))
        })
    }

    fn has_strict_upper_expr(&self, func: &Function, ranges: &Ranges, ty: Type, val: Value, depth: u8) -> bool {
        if self.rel_kinds & kind_bit(RelKind::Ult) != 0
            && self.steps_from(ty, val, UNSIGNED)
            && self.rels.iter().any(|r| r.ty == ty && r.kind == RelKind::Ult && self.reaches(ty, val, r.rhs, UNSIGNED, true))
        {
            return true;
        }
        if depth >= 2 {
            return false;
        }
        match ranges.op(func, val) {
            Some(&Op::Binary(Binary::Band, x, y)) => {
                self.has_strict_upper_expr(func, ranges, ty, x, depth + 1)
                    || self.has_strict_upper_expr(func, ranges, ty, y, depth + 1)
            }
            Some(&Op::Binary(Binary::Ushr, x, _)) => self.has_strict_upper_expr(func, ranges, ty, x, depth + 1),
            Some(&Op::Select(_, t, f)) => {
                self.has_strict_upper_expr(func, ranges, ty, t, depth + 1)
                    && self.has_strict_upper_expr(func, ranges, ty, f, depth + 1)
            }
            Some(&Op::Binary(Binary::Isub, x, d)) => {
                self.refined_fact(func, ranges, d, 0).lo >= 1
                    && self.proves_unsigned(func, ranges, ty, d, x, false, 0)
            }
            Some(&Op::Binary(Binary::Urem, _, y)) => !self.refined_fact(func, ranges, y, 0).admits(0),
            _ => false,
        }
    }

    fn proves_unsigned(&self, func: &Function, ranges: &Ranges, ty: Type, a: Value, b: Value, strict: bool, depth: u8) -> bool {
        if a == b {
            return !strict;
        }
        let reach = self.reach(ty, a, b, UNSIGNED, strict);
        if reach == Some(true)
            || (reach.is_some() && self.known_ne(ty, a, b))
            || self.proves_add_rel(func, ranges, ty, a, b, strict)
            || self.proves_sub_rel(func, ranges, ty, a, b, strict)
        {
            return true;
        }
        if depth >= 2 {
            return false;
        }
        match ranges.op(func, a) {
            Some(&Op::Binary(Binary::Band, x, y)) => {
                self.proves_unsigned(func, ranges, ty, x, b, strict, depth + 1)
                    || self.proves_unsigned(func, ranges, ty, y, b, strict, depth + 1)
            }
            Some(&Op::Binary(Binary::Ushr, x, _)) => self.proves_unsigned(func, ranges, ty, x, b, strict, depth + 1),
            Some(&Op::Binary(Binary::Urem, x, y)) => {
                self.proves_unsigned(func, ranges, ty, x, b, strict, depth + 1)
                    || (!self.refined_fact(func, ranges, y, 0).admits(0)
                        && self.proves_unsigned(func, ranges, ty, y, b, false, depth + 1))
            }
            Some(&Op::Select(_, t, f)) => {
                self.proves_unsigned(func, ranges, ty, t, b, strict, depth + 1)
                    && self.proves_unsigned(func, ranges, ty, f, b, strict, depth + 1)
            }
            _ => false,
        }
    }

    fn proves(&self, func: &Function, ranges: &Ranges, cc: IntCC, ty: Type, x: Value, y: Value) -> bool {
        if x == y {
            return holds(cc, ty, 0, 0);
        }
        let (kind, a, b) = rel_of(cc, x, y);
        match kind {
            RelKind::Eq => self.reaches(ty, a, b, EQUAL, false),
            RelKind::Ult => self.proves_unsigned(func, ranges, ty, a, b, true, 0),
            RelKind::Ule => self.proves_unsigned(func, ranges, ty, a, b, false, 0),
            RelKind::Slt => {
                let reach = self.reach(ty, a, b, SIGNED, true);
                reach == Some(true) || (reach.is_some() && self.known_ne(ty, a, b))
            }
            RelKind::Sle => self.reaches(ty, a, b, SIGNED, false),
            RelKind::Ne => {
                self.strictly_ordered(ty, a, b)
                    || self.proves_add_rel(func, ranges, ty, a, b, true)
                    || self.proves_add_rel(func, ranges, ty, b, a, true)
                    || [(a, b), (b, a)].into_iter().any(|(sub, zero)| {
                        self.fact(ranges, zero).exact() == Some(0)
                            && matches!(ranges.op(func, sub), Some(&Op::Binary(Binary::Isub, lhs, rhs)) if self.strictly_ordered(ty, lhs, rhs))
                    })
                    || self.known_ne(ty, a, b)
            }
        }
    }

    fn strictly_ordered(&self, ty: Type, a: Value, b: Value) -> bool {
        let has_u = self.rel_kinds & kind_bit(RelKind::Ult) != 0;
        let has_s = self.rel_kinds & kind_bit(RelKind::Slt) != 0;
        (has_u && (self.reaches(ty, a, b, UNSIGNED, true) || self.reaches(ty, b, a, UNSIGNED, true)))
            || (has_s && (self.reaches(ty, a, b, SIGNED, true) || self.reaches(ty, b, a, SIGNED, true)))
    }

    pub(super) fn decide_rel(&self, func: &Function, ranges: &Ranges, cc: IntCC, ty: Type, x: Value, y: Value) -> Option<bool> {
        if self.rels.is_empty() && self.no_wrap_adds.is_empty() {
            return None;
        }
        if self.proves(func, ranges, cc, ty, x, y) {
            Some(true)
        } else if self.proves(func, ranges, cc.inverse(), ty, x, y) {
            Some(false)
        } else {
            None
        }
    }

    /// Records what the edge from `from` to `into` says when `from` ends in a branch with two different targets.
    pub(super) fn follow(&mut self, func: &Function, ranges: &Ranges, from: Block, into: Block) {
        if let Some(&Op::Brif(condition, then, _, otherwise, _)) = func.last(from)
            && then != otherwise
        {
            self.assume(func, ranges, condition, then == into);
        }
    }
}

fn signed_bounds(fact: &Fact, ty: Type) -> (i128, i128) {
    let sign_max = ty.unsigned_max() >> 1;
    if fact.hi <= sign_max {
        (i128::from(fact.lo), i128::from(fact.hi))
    } else if fact.lo > sign_max {
        (i128::from(extend(fact.lo as i64, ty, true)), i128::from(extend(fact.hi as i64, ty, true)))
    } else {
        (-(1_i128 << (ty.bits() - 1)), (1_i128 << (ty.bits() - 1)) - 1)
    }
}

/// Decides `Op::Fits(op, signed, x, y)` when path ranges or unsigned relations settle whether the result fits `ty`.
pub(super) fn fits_truth(func: &Function, ranges: &Ranges, path: &Path, op: Binary, signed: bool, x: Value, y: Value) -> Option<bool> {
    let ty = func.ty(x);
    if !ty.is_small_int() {
        return None;
    }
    let (fx, fy) = (path.refined_fact(func, ranges, x, 0), path.refined_fact(func, ranges, y, 0));
    if !signed {
        let max = ty.unsigned_max();
        let strict_upper = |v, f: &Fact| {
            if f.hi == max && path.has_strict_upper_expr(func, ranges, ty, v, 0) { max - 1 } else { f.hi }
        };
        let (x_hi, y_hi) = (strict_upper(x, &fx), strict_upper(y, &fy));
        let tight = |v, hi: u64| if path.rels.is_empty() { hi } else { hi.min(path.upper_bound(func, ranges, ty, v, 0)) };
        return match op {
            Binary::Iadd => {
                if path.has_no_wrap_add(x, y) || x_hi.checked_add(y_hi).is_some_and(|sum| sum <= max) {
                    Some(true)
                } else if fx.lo.checked_add(fy.lo).is_none_or(|sum| sum > max) {
                    Some(false)
                } else if !path.rels.is_empty() && tight(x, x_hi).checked_add(tight(y, y_hi)).is_some_and(|sum| sum <= max) {
                    Some(true)
                } else {
                    None
                }
            }
            Binary::Isub => {
                if fx.lo >= y_hi || path.proves(func, ranges, IntCC::UnsignedGreaterThanOrEqual, ty, x, y) {
                    Some(true)
                } else if x_hi < fy.lo || path.proves(func, ranges, IntCC::UnsignedLessThan, ty, x, y) {
                    Some(false)
                } else if !path.rels.is_empty() && fx.lo >= tight(y, y_hi) {
                    Some(true)
                } else if !path.rels.is_empty() && tight(x, x_hi) < fy.lo {
                    Some(false)
                } else {
                    None
                }
            }
            Binary::Imul => {
                if x_hi.checked_mul(y_hi).is_some_and(|prod| prod <= max)
                    || (!path.rels.is_empty() && tight(x, x_hi).checked_mul(tight(y, y_hi)).is_some_and(|prod| prod <= max))
                {
                    Some(true)
                } else if fx.lo.checked_mul(fy.lo).is_none_or(|prod| prod > max) {
                    Some(false)
                } else {
                    None
                }
            }
            _ => None,
        };
    }
    let (s_lo, s_hi) = (-(1_i128 << (ty.bits() - 1)), (1_i128 << (ty.bits() - 1)) - 1);
    let ((ax, bx), (ay, by)) = (signed_bounds(&fx, ty), signed_bounds(&fy, ty));
    let (min, max) = match op {
        Binary::Iadd => (ax + ay, bx + by),
        Binary::Isub => (ax - by, bx - ay),
        Binary::Imul => {
            let (p0, p1, p2, p3) = (ax * ay, ax * by, bx * ay, bx * by);
            (p0.min(p1).min(p2).min(p3), p0.max(p1).max(p2).max(p3))
        }
        _ => return None,
    };
    if min >= s_lo && max <= s_hi {
        Some(true)
    } else if max < s_lo || min > s_hi {
        Some(false)
    } else {
        None
    }
}

pub(super) fn cond_truth(func: &Function, ranges: &Ranges, path: &Path, condition: Value) -> Option<bool> {
    cond_truth_depth(func, ranges, path, condition, 0)
}

fn cond_truth_depth(func: &Function, ranges: &Ranges, path: &Path, condition: Value, depth: u8) -> Option<bool> {
    if let Some(taken) = truth(&path.refined_fact(func, ranges, condition, 0)) {
        return Some(taken);
    }
    if depth > 4 {
        return None;
    }
    let op = ranges.op(func, condition);
    let direct = match op {
        Some(&Op::Fits(op, signed, x, y)) => fits_truth(func, ranges, path, op, signed, x, y),
        Some(&Op::Binary(Binary::Bxor, a, b)) => [(a, b), (b, a)].into_iter().find_map(|(inner, mask)| {
            (path.fact(ranges, mask).exact() == Some(1) && path.fact(ranges, inner).hi <= 1)
                .then(|| cond_truth_depth(func, ranges, path, inner, depth + 1).map(|t| !t))
                .flatten()
        }),
        Some(&Op::Binary(Binary::Band, a, b)) if path.fact(ranges, a).hi <= 1 && path.fact(ranges, b).hi <= 1 => {
            match (
                cond_truth_depth(func, ranges, path, a, depth + 1),
                cond_truth_depth(func, ranges, path, b, depth + 1),
            ) {
                (Some(false), _) | (_, Some(false)) => Some(false),
                (Some(true), Some(true)) => Some(true),
                _ => None,
            }
        }
        Some(&Op::Binary(Binary::Bor, a, b)) => {
            match (
                cond_truth_depth(func, ranges, path, a, depth + 1),
                cond_truth_depth(func, ranges, path, b, depth + 1),
            ) {
                (Some(true), _) | (_, Some(true)) => Some(true),
                (Some(false), Some(false)) => Some(false),
                _ => None,
            }
        }
        _ => None,
    };
    if direct.is_some() {
        return direct;
    }
    let Some((cc, x, y)) = op.and_then(crate::algebra::branch_cmp) else {
        return None;
    };
    if !func.ty(x).is_small_int() {
        return None;
    }
    let ty = func.ty(x);
    decide(cc, ty, &path.refined_fact(func, ranges, x, 0), &path.refined_fact(func, ranges, y, 0))
        .or_else(|| path.decide_rel(func, ranges, cc, ty, x, y))
        .or_else(|| {
            matches!(cc, IntCC::Equal | IntCC::NotEqual).then(|| {
                [(x, y), (y, x)].into_iter().find_map(|(expr, imm)| {
                    let k = path.fact(ranges, imm).exact()?;
                    ((k == 0 || k == 1) && path.fact(ranges, expr).hi <= 1).then(|| {
                        cond_truth_depth(func, ranges, path, expr, depth + 1).map(|t| (t == (k == 1)) ^ (cc == IntCC::NotEqual))
                    }).flatten()
                })
            }).flatten()
        })
}

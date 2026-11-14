use crate::fold::{extend, holds};
use crate::ir::{Binary, Block, Function, IntCC, Op, Type, Value};

use super::{Fact, MAX_FACTS, MAX_RELS, Ranges, as_unsigned, compared_range, decide, swapped, truth};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum RelKind {
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
    facts: Vec<(Value, Fact)>,
    rels: Vec<Rel>,
    no_wrap_adds: Vec<(Value, Value)>,
}

impl Path {
    fn get(&self, value: Value) -> Option<&Fact> {
        self.facts.iter().rev().find(|(at, _)| *at == value).map(|(_, fact)| fact)
    }

    /// Everything known about `value` here: its definition's range and the path's facts.
    pub(super) fn fact(&self, ranges: &Ranges, value: Value) -> Fact {
        let defined = ranges.of(value);
        match self.get(value) {
            Some(fact) => defined.meet(fact),
            None => defined,
        }
    }

    /// Records `fact` on `value`, forgetting the oldest fact past `MAX_FACTS`.
    fn add(&mut self, ranges: &Ranges, value: Value, fact: Fact) {
        let merged = self.fact(ranges, value).meet(&fact);
        if let Some(pos) = self.facts.iter().position(|(at, _)| *at == value) {
            self.facts.remove(pos);
        } else if self.facts.len() >= MAX_FACTS {
            self.facts.remove(0);
        }
        self.facts.push((value, merged));
    }

    fn push_rel(&mut self, rel: Rel) {
        if !self.rels.contains(&rel) {
            if self.rels.len() >= MAX_RELS {
                self.rels.remove(0);
            }
            self.rels.push(rel);
        }
    }

    pub(super) fn refined_fact(&self, func: &Function, ranges: &Ranges, value: Value, depth: u8) -> Fact {
        let mut fact = self.fact(ranges, value);
        if depth >= 3 {
            return fact;
        }
        match ranges.op(func, value) {
            Some(&Op::Convert(crate::ir::Convert::Uextend, inner)) if func.ty(inner).is_small_int() => {
                fact = fact.meet(&self.refined_fact(func, ranges, inner, depth + 1));
            }
            Some(&Op::Binary(Binary::Band, a, b)) => {
                let hi = self.refined_fact(func, ranges, a, depth + 1).hi.min(self.refined_fact(func, ranges, b, depth + 1).hi);
                fact = fact.meet(&Fact::range(0, hi));
            }
            Some(&Op::Binary(Binary::Ushr, a, sh)) => {
                if let Some(c) = self.fact(ranges, sh).exact() {
                    let inner = self.refined_fact(func, ranges, a, depth + 1);
                    let k = (c & u64::from(func.ty(value).bits() - 1)) as u32;
                    fact = fact.meet(&Fact::range(inner.lo >> k, inner.hi >> k));
                }
            }
            Some(&Op::Binary(Binary::Isub, a, b)) => {
                let (fa, fb) = (self.refined_fact(func, ranges, a, depth + 1), self.refined_fact(func, ranges, b, depth + 1));
                if fa.lo >= fb.hi {
                    fact = fact.meet(&Fact::range(fa.lo - fb.hi, fa.hi - fb.lo));
                }
            }
            Some(&Op::Binary(Binary::Iadd, a, b)) => {
                let ty = func.ty(value);
                let (fa, fb) = (self.refined_fact(func, ranges, a, depth + 1), self.refined_fact(func, ranges, b, depth + 1));
                if let Some(hi) = fa.hi.checked_add(fb.hi).filter(|&h| h <= ty.unsigned_max()) {
                    fact = fact.meet(&Fact::range(fa.lo + fb.lo, hi));
                }
            }
            _ => {}
        }
        fact
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
                    }
                    if let Some(k) = self.fact(ranges, y).exact() {
                        self.add(ranges, x, Fact::range(0, op_ty.unsigned_max() - k));
                    }
                    if let Some(k) = self.fact(ranges, x).exact() {
                        self.add(ranges, y, Fact::range(0, op_ty.unsigned_max() - k));
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
                    self.add(ranges, x, Fact::range(fy.lo, op_ty.unsigned_max()));
                    let fx = self.fact(ranges, x);
                    self.add(ranges, y, Fact::range(0, fx.hi));
                }
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
                    self.push_rel(Rel { kind, lhs, rhs, ty: operand_ty });
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

    fn add_const(&self, func: &Function, ranges: &Ranges, val: Value) -> Option<(Value, u64)> {
        let &Op::Binary(Binary::Iadd, p, q) = ranges.op(func, val)? else { return None };
        [(p, q), (q, p)].into_iter().find_map(|(base, c)| self.fact(ranges, c).exact().map(|k| (base, k)))
    }

    fn is_no_wrap_pair(&self, ranges: &Ranges, ty: Type, base: Value, c: Value, k: u64) -> bool {
        let pair = if base.index() <= c.index() { (base, c) } else { (c, base) };
        self.no_wrap_adds.contains(&pair)
            || self.fact(ranges, base).hi.checked_add(k).is_some_and(|hi| hi <= ty.unsigned_max())
    }

    fn non_wrap_iadd(&self, func: &Function, ranges: &Ranges, val: Value, ty: Type) -> Option<(Value, u64)> {
        let &Op::Binary(Binary::Iadd, p, q) = ranges.op(func, val)? else { return None };
        for (base, c) in [(p, q), (q, p)] {
            if let Some(k) = self.fact(ranges, c).exact()
                && self.is_no_wrap_pair(ranges, ty, base, c, k)
            {
                return Some((base, k));
            }
        }
        None
    }

    fn proves_add_rel(&self, func: &Function, ranges: &Ranges, ty: Type, a: Value, b: Value, strict: bool) -> bool {
        let a_forms = [Some((a, 0_u64)), self.add_const(func, ranges, a)];
        let mut upper_terms: Vec<(Value, Value, u64)> = Vec::new();
        if let Some((b_base, k)) = self.non_wrap_iadd(func, ranges, b, ty) {
            upper_terms.push((b, b_base, k));
        }
        for &r in &self.rels {
            if r.ty != ty {
                continue;
            }
            let candidates = match r.kind {
                RelKind::Ult | RelKind::Ule => [Some(r.lhs), None],
                RelKind::Eq => [Some(r.lhs), Some(r.rhs)],
                _ => [None, None],
            };
            for s in candidates.into_iter().flatten() {
                upper_terms.push((s, s, 0));
                if let Some((s_base, k)) = self.non_wrap_iadd(func, ranges, s, ty) {
                    upper_terms.push((s, s_base, k));
                }
            }
        }
        for (a_base, c) in a_forms.into_iter().flatten() {
            for &(s, s_base, k) in &upper_terms {
                if !self.reaches(ty, a_base, s_base, (RelKind::Ult, RelKind::Ule), false) {
                    continue;
                }
                if strict {
                    if (c < k && self.reaches(ty, s, b, (RelKind::Ult, RelKind::Ule), false))
                        || (c <= k && self.reaches(ty, s, b, (RelKind::Ult, RelKind::Ule), true))
                    {
                        return true;
                    }
                } else if (c <= k && self.reaches(ty, s, b, (RelKind::Ult, RelKind::Ule), false))
                    || (k.checked_add(1) == Some(c) && self.reaches(ty, s, b, (RelKind::Ult, RelKind::Ule), true))
                {
                    return true;
                }
            }
        }
        false
    }

    fn proves_sub_rel(&self, func: &Function, ranges: &Ranges, ty: Type, a: Value, b: Value, strict: bool) -> bool {
        let Some(&Op::Binary(Binary::Isub, base, c)) = ranges.op(func, a) else { return false };
        if let Some(k) = ranges.constant(c)
            && self.fact(ranges, base).lo >= k
            && self.reaches(ty, base, b, (RelKind::Ult, RelKind::Ule), strict && k == 0)
        {
            return true;
        }
        if !strict {
            return false;
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
        false
    }

    fn known_ne(&self, ty: Type, a: Value, b: Value) -> bool {
        self.rels.iter().any(|r| {
            r.kind == RelKind::Ne
                && r.ty == ty
                && ((self.reaches(ty, a, r.lhs, (RelKind::Eq, RelKind::Eq), false) && self.reaches(ty, b, r.rhs, (RelKind::Eq, RelKind::Eq), false))
                    || (self.reaches(ty, a, r.rhs, (RelKind::Eq, RelKind::Eq), false) && self.reaches(ty, b, r.lhs, (RelKind::Eq, RelKind::Eq), false)))
        })
    }

    fn has_strict_upper_expr(&self, func: &Function, ranges: &Ranges, ty: Type, val: Value, depth: u8) -> bool {
        if self.rels.iter().any(|r| {
            r.ty == ty && r.kind == RelKind::Ult && self.reaches(ty, val, r.rhs, (RelKind::Ult, RelKind::Ule), true)
        }) {
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
        if self.reaches(ty, a, b, (RelKind::Ult, RelKind::Ule), strict)
            || (strict && self.reaches(ty, a, b, (RelKind::Ult, RelKind::Ule), false) && self.known_ne(ty, a, b))
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
            RelKind::Eq => self.reaches(ty, a, b, (RelKind::Eq, RelKind::Eq), false),
            RelKind::Ult => self.proves_unsigned(func, ranges, ty, a, b, true, 0),
            RelKind::Ule => self.proves_unsigned(func, ranges, ty, a, b, false, 0),
            RelKind::Slt => {
                self.reaches(ty, a, b, (RelKind::Slt, RelKind::Sle), true)
                    || (self.reaches(ty, a, b, (RelKind::Slt, RelKind::Sle), false) && self.known_ne(ty, a, b))
            }
            RelKind::Sle => self.reaches(ty, a, b, (RelKind::Slt, RelKind::Sle), false),
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
        self.reaches(ty, a, b, (RelKind::Ult, RelKind::Ule), true)
            || self.reaches(ty, b, a, (RelKind::Ult, RelKind::Ule), true)
            || self.reaches(ty, a, b, (RelKind::Slt, RelKind::Sle), true)
            || self.reaches(ty, b, a, (RelKind::Slt, RelKind::Sle), true)
    }

    pub(super) fn decide_rel(&self, func: &Function, ranges: &Ranges, cc: IntCC, ty: Type, x: Value, y: Value) -> Option<bool> {
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
        let x_hi = if fx.hi == max && path.has_strict_upper_expr(func, ranges, ty, x, 0) { max - 1 } else { fx.hi };
        let y_hi = if fy.hi == max && path.has_strict_upper_expr(func, ranges, ty, y, 0) { max - 1 } else { fy.hi };
        return match op {
            Binary::Iadd => {
                if x_hi.checked_add(y_hi).is_some_and(|sum| sum <= max) {
                    Some(true)
                } else if fx.lo.checked_add(fy.lo).is_none_or(|sum| sum > max) {
                    Some(false)
                } else {
                    None
                }
            }
            Binary::Isub => {
                if fx.lo >= y_hi || path.proves(func, ranges, IntCC::UnsignedGreaterThanOrEqual, ty, x, y) {
                    Some(true)
                } else if x_hi < fy.lo || path.proves(func, ranges, IntCC::UnsignedLessThan, ty, x, y) {
                    Some(false)
                } else {
                    None
                }
            }
            Binary::Imul => {
                if x_hi.checked_mul(y_hi).is_some_and(|prod| prod <= max) {
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
    if let Some(taken) = truth(&path.fact(ranges, condition)) {
        return Some(taken);
    }
    if depth > 4 {
        return None;
    }
    match ranges.op(func, condition) {
        Some(&Op::Fits(op, signed, x, y)) => fits_truth(func, ranges, path, op, signed, x, y),
        Some(&Op::Icmp(cc, x, y)) if func.ty(x).is_small_int() => {
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
    }
}

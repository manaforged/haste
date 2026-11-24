use crate::fold::{extend, holds};
use crate::ir::{Binary, Function, IntCC, Op, Type, Value};

use super::super::{Fact, Ranges, decide, truth};
use super::{Path, RelKind, UNSIGNED, kind_bit, rel_of};

const SIGNED: (RelKind, RelKind) = (RelKind::Slt, RelKind::Sle);
const EQUAL: (RelKind, RelKind) = (RelKind::Eq, RelKind::Eq);

impl Path {
    fn proves_sub_rel(
        &self,
        func: &Function,
        ranges: &Ranges,
        ty: Type,
        a: Value,
        b: Value,
        strict: bool,
    ) -> bool {
        let Some(&Op::Binary(Binary::Isub, base, c)) = ranges.op(func, a) else {
            return false;
        };
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
        self.rels
            .iter()
            .filter(|r| r.kind == RelKind::Ne && r.ty == ty)
            .any(|r| {
                let mut at_a = |target| {
                    a == target
                        || self.reaches_in(
                            *from_a.get_or_insert_with(|| self.reached(ty, a, EQUAL)),
                            target,
                        )
                };
                let mut at_b = |target| {
                    b == target
                        || self.reaches_in(
                            *from_b.get_or_insert_with(|| self.reached(ty, b, EQUAL)),
                            target,
                        )
                };
                (at_a(r.lhs) && at_b(r.rhs)) || (at_a(r.rhs) && at_b(r.lhs))
            })
    }

    fn has_strict_upper_expr(
        &self,
        func: &Function,
        ranges: &Ranges,
        ty: Type,
        val: Value,
        depth: u8,
    ) -> bool {
        if self.rel_kinds & kind_bit(RelKind::Ult) != 0
            && self.steps_from(ty, val, UNSIGNED)
            && self.rels.iter().any(|r| {
                r.ty == ty && r.kind == RelKind::Ult && self.reaches(ty, val, r.rhs, UNSIGNED, true)
            })
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
            Some(&Op::Binary(Binary::Ushr, x, _)) => {
                self.has_strict_upper_expr(func, ranges, ty, x, depth + 1)
            }
            Some(&Op::Select(_, t, f)) => {
                self.has_strict_upper_expr(func, ranges, ty, t, depth + 1)
                    && self.has_strict_upper_expr(func, ranges, ty, f, depth + 1)
            }
            Some(&Op::Binary(Binary::Isub, x, d)) => {
                self.refined_fact(func, ranges, d, 0).lo >= 1
                    && self.proves_unsigned(func, ranges, ty, d, x, false, 0)
            }
            Some(&Op::Binary(Binary::Urem, _, y)) => {
                !self.refined_fact(func, ranges, y, 0).admits(0)
            }
            _ => false,
        }
    }

    fn proves_unsigned(
        &self,
        func: &Function,
        ranges: &Ranges,
        ty: Type,
        a: Value,
        b: Value,
        strict: bool,
        depth: u8,
    ) -> bool {
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
            Some(&Op::Binary(Binary::Ushr, x, _)) => {
                self.proves_unsigned(func, ranges, ty, x, b, strict, depth + 1)
            }
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

    fn proves(
        &self,
        func: &Function,
        ranges: &Ranges,
        cc: IntCC,
        ty: Type,
        x: Value,
        y: Value,
    ) -> bool {
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
                            && matches!(
                                ranges.op(func, sub),
                                Some(&Op::Binary(Binary::Isub, lhs, rhs))
                                    if self.strictly_ordered(ty, lhs, rhs)
                            )
                    })
                    || self.known_ne(ty, a, b)
            }
        }
    }

    fn strictly_ordered(&self, ty: Type, a: Value, b: Value) -> bool {
        let has_u = self.rel_kinds & kind_bit(RelKind::Ult) != 0;
        let has_s = self.rel_kinds & kind_bit(RelKind::Slt) != 0;
        (has_u
            && (self.reaches(ty, a, b, UNSIGNED, true) || self.reaches(ty, b, a, UNSIGNED, true)))
            || (has_s
                && (self.reaches(ty, a, b, SIGNED, true) || self.reaches(ty, b, a, SIGNED, true)))
    }

    /// Decides `x cc y` from the symbolic relations recorded along this path.
    pub(in super::super) fn decide_rel(
        &self,
        func: &Function,
        ranges: &Ranges,
        cc: IntCC,
        ty: Type,
        x: Value,
        y: Value,
    ) -> Option<bool> {
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
}

fn signed_bounds(fact: &Fact, ty: Type) -> (i128, i128) {
    let sign_max = ty.unsigned_max() >> 1;
    if fact.hi <= sign_max {
        (i128::from(fact.lo), i128::from(fact.hi))
    } else if fact.lo > sign_max {
        (
            i128::from(extend(fact.lo as i64, ty, true)),
            i128::from(extend(fact.hi as i64, ty, true)),
        )
    } else {
        (
            -(1_i128 << (ty.bits() - 1)),
            (1_i128 << (ty.bits() - 1)) - 1,
        )
    }
}

/// Decides `Op::Fits(op, signed, x, y)` when path ranges or unsigned relations settle whether the result fits `ty`.
pub(in super::super) fn fits_truth(
    func: &Function,
    ranges: &Ranges,
    path: &Path,
    op: Binary,
    signed: bool,
    x: Value,
    y: Value,
) -> Option<bool> {
    let ty = func.ty(x);
    if !ty.is_small_int() {
        return None;
    }
    let (fx, fy) = (
        path.refined_fact(func, ranges, x, 0),
        path.refined_fact(func, ranges, y, 0),
    );
    if !signed {
        let max = ty.unsigned_max();
        let strict_upper = |v, f: &Fact| {
            if f.hi == max && path.has_strict_upper_expr(func, ranges, ty, v, 0) {
                max - 1
            } else {
                f.hi
            }
        };
        let (x_hi, y_hi) = (strict_upper(x, &fx), strict_upper(y, &fy));
        let tight = |v, hi: u64| {
            if path.rels.is_empty() {
                hi
            } else {
                hi.min(path.upper_bound(func, ranges, ty, v, 0))
            }
        };
        return match op {
            Binary::Iadd => {
                if path.has_no_wrap_add(x, y)
                    || x_hi.checked_add(y_hi).is_some_and(|sum| sum <= max)
                {
                    Some(true)
                } else if fx.lo.checked_add(fy.lo).is_none_or(|sum| sum > max) {
                    Some(false)
                } else if !path.rels.is_empty()
                    && tight(x, x_hi)
                        .checked_add(tight(y, y_hi))
                        .is_some_and(|sum| sum <= max)
                {
                    Some(true)
                } else {
                    None
                }
            }
            Binary::Isub => {
                if fx.lo >= y_hi
                    || path.proves(func, ranges, IntCC::UnsignedGreaterThanOrEqual, ty, x, y)
                {
                    Some(true)
                } else if x_hi < fy.lo
                    || path.proves(func, ranges, IntCC::UnsignedLessThan, ty, x, y)
                {
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
                    || (!path.rels.is_empty()
                        && tight(x, x_hi)
                            .checked_mul(tight(y, y_hi))
                            .is_some_and(|prod| prod <= max))
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
    let (s_lo, s_hi) = (
        -(1_i128 << (ty.bits() - 1)),
        (1_i128 << (ty.bits() - 1)) - 1,
    );
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

/// Decides a branch condition from ranges, fits checks, boolean ops, and path relations.
pub(in super::super) fn cond_truth(
    func: &Function,
    ranges: &Ranges,
    path: &Path,
    condition: Value,
) -> Option<bool> {
    cond_truth_depth(func, ranges, path, condition, 0)
}

fn cond_truth_depth(
    func: &Function,
    ranges: &Ranges,
    path: &Path,
    condition: Value,
    depth: u8,
) -> Option<bool> {
    if let Some(taken) = truth(&path.refined_fact(func, ranges, condition, 0)) {
        return Some(taken);
    }
    if depth > 4 {
        return None;
    }
    let op = ranges.op(func, condition);
    let direct = match op {
        Some(&Op::Fits(op, signed, x, y)) => fits_truth(func, ranges, path, op, signed, x, y),
        Some(&Op::Binary(Binary::Bxor, a, b)) => {
            [(a, b), (b, a)].into_iter().find_map(|(inner, mask)| {
                (path.fact(ranges, mask).exact() == Some(1) && path.fact(ranges, inner).hi <= 1)
                    .then(|| cond_truth_depth(func, ranges, path, inner, depth + 1).map(|t| !t))
                    .flatten()
            })
        }
        Some(&Op::Binary(Binary::Band, a, b))
            if path.fact(ranges, a).hi <= 1 && path.fact(ranges, b).hi <= 1 =>
        {
            match (
                cond_truth_depth(func, ranges, path, a, depth + 1),
                cond_truth_depth(func, ranges, path, b, depth + 1),
            ) {
                (Some(false), _) | (_, Some(false)) => Some(false),
                (Some(true), Some(true)) => Some(true),
                _ => None,
            }
        }
        Some(&Op::Binary(Binary::Bor, a, b)) => match (
            cond_truth_depth(func, ranges, path, a, depth + 1),
            cond_truth_depth(func, ranges, path, b, depth + 1),
        ) {
            (Some(true), _) | (_, Some(true)) => Some(true),
            (Some(false), Some(false)) => Some(false),
            _ => None,
        },
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
    decide(
        cc,
        ty,
        &path.refined_fact(func, ranges, x, 0),
        &path.refined_fact(func, ranges, y, 0),
    )
    .or_else(|| path.decide_rel(func, ranges, cc, ty, x, y))
    .or_else(|| {
        matches!(cc, IntCC::Equal | IntCC::NotEqual)
            .then(|| {
                [(x, y), (y, x)].into_iter().find_map(|(expr, imm)| {
                    let k = path.fact(ranges, imm).exact()?;
                    ((k == 0 || k == 1) && path.fact(ranges, expr).hi <= 1)
                        .then(|| {
                            cond_truth_depth(func, ranges, path, expr, depth + 1)
                                .map(|t| (t == (k == 1)) ^ (cc == IntCC::NotEqual))
                        })
                        .flatten()
                })
            })
            .flatten()
    })
}

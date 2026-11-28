mod prove;

use crate::ir::{Binary, Block, Function, IntCC, Op, Type, Value};

use super::backprop::Uppers;
use super::{Fact, MAX_FACTS, MAX_RELS, Ranges, compared_range, unsigned_order};

pub(super) use prove::{cond_truth, fits_truth};

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
const WALK_SEEN: usize = 16;
const REV_SEEN: usize = 32;
const _: () = assert!(2 * MAX_RELS <= u128::BITS as usize);
const _: () = assert!(MAX_RELS <= u64::BITS as usize);

#[inline]
fn rel_step_matching(r: Rel, cur: Value, strict_k: RelKind) -> Option<(Value, bool)> {
    if r.kind == RelKind::Eq {
        if r.lhs == cur {
            Some((r.rhs, false))
        } else if r.rhs == cur {
            Some((r.lhs, false))
        } else {
            None
        }
    } else if r.lhs == cur {
        Some((r.rhs, r.kind == strict_k))
    } else {
        None
    }
}

#[inline]
fn rel_rev_step_matching(r: Rel, cur: Value, strict_k: RelKind) -> Option<(Value, bool)> {
    if r.kind == RelKind::Eq {
        if r.rhs == cur {
            Some((r.lhs, false))
        } else if r.lhs == cur {
            Some((r.rhs, false))
        } else {
            None
        }
    } else if r.rhs == cur {
        Some((r.lhs, r.kind == strict_k))
    } else {
        None
    }
}

#[inline]
fn push_seen<const N: usize>(
    seen: &mut [(Value, bool); N],
    len: &mut usize,
    seen_mask: &mut u128,
    next: Value,
    next_strict: bool,
) -> bool {
    let bit = val_bit(next);
    if *seen_mask & bit != 0
        && let Some(entry) = seen[..*len].iter_mut().find(|(v, _)| *v == next)
    {
        if next_strict && !entry.1 {
            if *len >= N {
                return false;
            }
            entry.1 = true;
            seen[*len] = (next, true);
            *len += 1;
        }
        return true;
    }
    if *len >= N {
        return false;
    }
    seen[*len] = (next, next_strict);
    *len += 1;
    *seen_mask |= bit;
    true
}

pub(super) fn rel_step(
    r: Rel,
    cur: Value,
    ty: Type,
    (strict_k, weak_k): (RelKind, RelKind),
) -> Option<(Value, bool)> {
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

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum RevReach {
    No,
    Weak,
    Strict,
    Unknown,
}

pub(super) struct ReachedInto {
    target: Value,
    seen: [(Value, bool); REV_SEEN],
    len: usize,
    mask: u128,
    overflowed: bool,
}

impl ReachedInto {
    pub(super) fn is_empty(&self) -> bool {
        self.len <= 1 && !self.overflowed
    }

    pub(super) fn get(&self, v: Value) -> RevReach {
        if v == self.target {
            return RevReach::Weak;
        }
        if self.mask & val_bit(v) != 0 {
            let mut found_weak = false;
            for &(node, strict) in &self.seen[1..self.len] {
                if node == v {
                    if strict {
                        return RevReach::Strict;
                    }
                    found_weak = true;
                }
            }
            if found_weak {
                return if self.overflowed {
                    RevReach::Unknown
                } else {
                    RevReach::Weak
                };
            }
        }
        if self.overflowed {
            RevReach::Unknown
        } else {
            RevReach::No
        }
    }
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
        let pair = if x.index() <= y.index() {
            (x, y)
        } else {
            (y, x)
        };
        self.no_wrap_adds.contains(&pair)
    }

    /// Records that the branch on `condition` went the `taken` way: the condition is nonzero or zero, and the integer compare or boolean op that defines it held or failed.
    pub(super) fn assume(
        &mut self,
        func: &Function,
        ranges: &Ranges,
        condition: Value,
        taken: bool,
    ) {
        self.assume_depth(func, ranges, condition, taken, 0);
    }

    fn assume_depth(
        &mut self,
        func: &Function,
        ranges: &Ranges,
        condition: Value,
        taken: bool,
        depth: u8,
    ) {
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
                    if self.fact(ranges, mask).exact() == Some(1)
                        && self.fact(ranges, inner).hi <= 1
                    {
                        self.assume_depth(func, ranges, inner, !taken, depth + 1);
                        break;
                    }
                }
            }
            Some(&Op::Fits(Binary::Iadd, false, x, y)) if taken => {
                let op_ty = func.ty(x);
                if op_ty.is_small_int() {
                    let pair = if x.index() <= y.index() {
                        (x, y)
                    } else {
                        (y, x)
                    };
                    if !self.no_wrap_adds.contains(&pair) {
                        if self.no_wrap_adds.len() >= MAX_RELS {
                            self.no_wrap_adds.remove(0);
                        }
                        self.no_wrap_adds.push(pair);
                        self.uppers.clear();
                    }
                    if let Some(k) = self.fact(ranges, y).exact() {
                        self.add_with_backprop(
                            func,
                            ranges,
                            x,
                            Fact::range(0, op_ty.unsigned_max() - k),
                            0,
                        );
                    }
                    if let Some(k) = self.fact(ranges, x).exact() {
                        self.add_with_backprop(
                            func,
                            ranges,
                            y,
                            Fact::range(0, op_ty.unsigned_max() - k),
                            0,
                        );
                    }
                }
            }
            Some(&Op::Fits(Binary::Isub, false, x, y)) if taken => {
                let op_ty = func.ty(x);
                if op_ty.is_small_int() {
                    if x != y {
                        self.push_rel(Rel {
                            kind: RelKind::Ule,
                            lhs: y,
                            rhs: x,
                            ty: op_ty,
                        });
                    }
                    let fy = self.fact(ranges, y);
                    self.add_with_backprop(
                        func,
                        ranges,
                        x,
                        Fact::range(fy.lo, op_ty.unsigned_max()),
                        0,
                    );
                    let fx = self.fact(ranges, x);
                    self.add_with_backprop(func, ranges, y, Fact::range(0, fx.hi), 0);
                }
            }
            _ => {}
        }
        if let Some((cc, x, y)) = ranges
            .op(func, condition)
            .and_then(crate::algebra::branch_cmp)
        {
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
                self.push_rel(Rel {
                    kind,
                    lhs,
                    rhs,
                    ty: operand_ty,
                });
            }
        }
    }

    #[inline]
    pub(super) fn steps_from(&self, ty: Type, start: Value, kinds: (RelKind, RelKind)) -> bool {
        let needed = kind_bit(kinds.0) | kind_bit(kinds.1) | kind_bit(RelKind::Eq);
        if self.rel_kinds & needed == 0 || self.rel_src_mask & val_bit(start) == 0 {
            return false;
        }
        self.rels
            .iter()
            .any(|&r| rel_step(r, start, ty, kinds).is_some())
    }

    #[inline]
    pub(super) fn steps_into(&self, ty: Type, target: Value, kinds: (RelKind, RelKind)) -> bool {
        let needed = kind_bit(kinds.0) | kind_bit(kinds.1) | kind_bit(RelKind::Eq);
        if self.rel_kinds & needed == 0 || self.rel_dst_mask & val_bit(target) == 0 {
            return false;
        }
        self.rels.iter().any(|&r| {
            r.ty == ty
                && ((r.rhs == target
                    && (r.kind == RelKind::Eq || r.kind == kinds.0 || r.kind == kinds.1))
                    || (r.lhs == target && r.kind == RelKind::Eq))
        })
    }

    pub(super) fn reaches(
        &self,
        ty: Type,
        start: Value,
        target: Value,
        kinds: (RelKind, RelKind),
        require_strict: bool,
    ) -> bool {
        self.reach(ty, start, target, kinds, require_strict) == Some(true)
    }

    fn active_rel_mask(&self, ty: Type, kinds: (RelKind, RelKind)) -> u64 {
        let mut active = 0_u64;
        for (pos, &r) in self.rels.iter().enumerate() {
            if r.ty == ty && (r.kind == RelKind::Eq || r.kind == kinds.0 || r.kind == kinds.1) {
                active |= 1_u64 << pos;
            }
        }
        active
    }

    pub(super) fn has_strict_step(
        &self,
        ty: Type,
        start: Value,
        kinds: (RelKind, RelKind),
    ) -> bool {
        let needed = kind_bit(kinds.0) | kind_bit(kinds.1) | kind_bit(RelKind::Eq);
        if self.rel_kinds & needed == 0
            || self.rel_kinds & kind_bit(kinds.0) == 0
            || self.rel_src_mask & val_bit(start) == 0
        {
            return false;
        }
        self.walk_rels(ty, start, kinds, |_, _, strict| strict)
    }

    pub(super) fn reached_into(
        &self,
        ty: Type,
        target: Value,
        kinds: (RelKind, RelKind),
    ) -> ReachedInto {
        let mut out = ReachedInto {
            target,
            seen: [(target, false); REV_SEEN],
            len: 1,
            mask: val_bit(target),
            overflowed: false,
        };
        let needed = kind_bit(kinds.0) | kind_bit(kinds.1) | kind_bit(RelKind::Eq);
        if self.rel_kinds & needed == 0 || self.rel_dst_mask & val_bit(target) == 0 {
            return out;
        }
        let active = self.active_rel_mask(ty, kinds);
        if active == 0 {
            return out;
        }
        let mut idx = 0;
        while idx < out.len {
            let (cur, cur_strict) = out.seen[idx];
            idx += 1;
            if self.rel_dst_mask & val_bit(cur) == 0 {
                continue;
            }
            let mut m = active;
            while m != 0 {
                let pos = m.trailing_zeros() as usize;
                m &= m - 1;
                let Some((prev, step_strict)) = rel_rev_step_matching(self.rels[pos], cur, kinds.0)
                else {
                    continue;
                };
                if !push_seen(
                    &mut out.seen,
                    &mut out.len,
                    &mut out.mask,
                    prev,
                    cur_strict || step_strict,
                ) {
                    out.overflowed = true;
                }
            }
        }
        out
    }

    fn reach(
        &self,
        ty: Type,
        start: Value,
        target: Value,
        kinds: (RelKind, RelKind),
        require_strict: bool,
    ) -> Option<bool> {
        if start == target {
            return Some(!require_strict);
        }
        let needed = kind_bit(kinds.0) | kind_bit(kinds.1) | kind_bit(RelKind::Eq);
        if self.rel_kinds & needed == 0
            || (require_strict && self.rel_kinds & kind_bit(kinds.0) == 0)
            || self.rel_src_mask & val_bit(start) == 0
            || self.rel_dst_mask & val_bit(target) == 0
        {
            return None;
        }
        let (mut seen, mut len) = ([(start, false); WALK_SEEN], 1);
        let mut seen_mask = val_bit(start);
        let mut active = 0_u64;
        let mut has_into = false;
        let mut hit = None;
        for (pos, &r) in self.rels.iter().enumerate() {
            if r.ty != ty {
                continue;
            }
            let is_eq = r.kind == RelKind::Eq;
            if !is_eq && r.kind != kinds.0 && r.kind != kinds.1 {
                continue;
            }
            active |= 1_u64 << pos;
            if r.rhs == target || (is_eq && r.lhs == target) {
                has_into = true;
            }
            let Some((next, step_strict)) = rel_step_matching(r, start, kinds.0) else {
                continue;
            };
            if next == target {
                if !require_strict || step_strict {
                    return Some(true);
                }
                hit = Some(false);
            }
            push_seen(&mut seen, &mut len, &mut seen_mask, next, step_strict);
        }
        if !has_into || len == 1 {
            return hit;
        }
        let mut idx = 1;
        while idx < len {
            let (cur, cur_strict) = seen[idx];
            idx += 1;
            if self.rel_src_mask & val_bit(cur) == 0 {
                continue;
            }
            let mut m = active;
            while m != 0 {
                let pos = m.trailing_zeros() as usize;
                m &= m - 1;
                let Some((next, step_strict)) = rel_step_matching(self.rels[pos], cur, kinds.0)
                else {
                    continue;
                };
                let next_strict = cur_strict || step_strict;
                if next == target {
                    if !require_strict || next_strict {
                        return Some(true);
                    }
                    hit = Some(false);
                }
                push_seen(&mut seen, &mut len, &mut seen_mask, next, next_strict);
            }
        }
        hit
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

    pub(super) fn walk(
        &self,
        ty: Type,
        start: Value,
        kinds: (RelKind, RelKind),
        mut visit: impl FnMut(Value, bool) -> bool,
    ) -> bool {
        self.walk_rels(ty, start, kinds, |_, next, next_strict| {
            visit(next, next_strict)
        })
    }

    fn walk_rels(
        &self,
        ty: Type,
        start: Value,
        kinds: (RelKind, RelKind),
        mut visit: impl FnMut(usize, Value, bool) -> bool,
    ) -> bool {
        let active = self.active_rel_mask(ty, kinds);
        if active == 0 {
            return false;
        }
        let (mut seen, mut len) = ([(start, false); WALK_SEEN], 1);
        let mut seen_mask = val_bit(start);
        let mut idx = 0;
        while idx < len {
            let (cur, cur_strict) = seen[idx];
            idx += 1;
            if self.rel_src_mask & val_bit(cur) == 0 {
                continue;
            }
            let mut m = active;
            while m != 0 {
                let pos = m.trailing_zeros() as usize;
                m &= m - 1;
                let Some((next, step_strict)) = rel_step_matching(self.rels[pos], cur, kinds.0)
                else {
                    continue;
                };
                let next_strict = cur_strict || step_strict;
                if visit(pos, next, next_strict) {
                    return true;
                }
                push_seen(&mut seen, &mut len, &mut seen_mask, next, next_strict);
            }
        }
        false
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

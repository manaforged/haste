use std::cell::RefCell;

use crate::ir::{Binary, Convert, Function, Op, Type, Value};

use super::rel::{Path, Rel, RelKind, UNSIGNED, kind_bit};
use super::transfer::{transfer, unwrapped, valid};
use super::{Fact, Ranges};

const CHAIN_STEPS: usize = 4;
const OPERAND_MEMO: usize = 4;

type Chain = ([(Value, Value, u64, u64); CHAIN_STEPS], usize);
type Steps = ([(Value, u64); CHAIN_STEPS], usize);

/// The values a path's unsigned and equal relations on one type bound from below, each with its non-wrapping add-const chain once asked for; cleared whenever the path changes.
#[derive(Default)]
pub(super) struct Uppers(RefCell<Bounded>);

#[derive(Default)]
struct Bounded {
    ty: Option<Type>,
    any_add: bool,
    values: Vec<(Value, bool, Option<Steps>)>,
}

impl Clone for Uppers {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl Uppers {
    /// Forgets what was kept, after the path's facts, relations or no-wrap adds change.
    pub(super) fn clear(&mut self) {
        self.0.get_mut().ty = None;
    }
}

impl Bounded {
    fn of(&mut self, func: &Function, ranges: &Ranges, rels: &[Rel], ty: Type) -> (&mut [(Value, bool, Option<Steps>)], bool) {
        if self.ty != Some(ty) {
            self.values.clear();
            self.any_add = false;
            for r in rels.iter().filter(|r| r.ty == ty) {
                let ends = match r.kind {
                    RelKind::Ult | RelKind::Ule => [Some(r.lhs), None],
                    RelKind::Eq => [Some(r.lhs), Some(r.rhs)],
                    _ => [None, None],
                };
                for s in ends.into_iter().flatten() {
                    if !self.values.iter().any(|&(seen, _, _)| seen == s) {
                        let is_add = matches!(ranges.op(func, s), Some(Op::Binary(Binary::Iadd, ..)));
                        self.any_add |= is_add;
                        self.values.push((s, is_add, None));
                    }
                }
            }
            self.ty = Some(ty);
        }
        (&mut self.values, self.any_add)
    }
}

impl Path {
    pub(super) fn is_no_wrap_pair(&self, ranges: &Ranges, ty: Type, base: Value, c: Value, k: u64) -> bool {
        self.has_no_wrap_add(base, c)
            || self.fact(ranges, base).hi.checked_add(k).is_some_and(|hi| hi <= ty.unsigned_max())
    }

    pub(super) fn reaches_upper_at_most(&self, ranges: &Ranges, ty: Type, start: Value, limit: u64) -> bool {
        self.min_reachable_hi(ranges, ty, start) <= limit
    }

    pub(super) fn is_no_wrap_add(&self, ranges: &Ranges, ty: Type, base: Value, c_val: Value, k: u64) -> bool {
        self.is_no_wrap_pair(ranges, ty, base, c_val, k)
            || ty.unsigned_max().checked_sub(k).is_some_and(|limit| self.reaches_upper_at_most(ranges, ty, base, limit))
    }

    pub(super) fn add_with_backprop(&mut self, func: &Function, ranges: &Ranges, value: Value, fact: Fact, depth: u8) {
        self.add(ranges, value, fact);
        let merged = self.fact(ranges, value);
        if depth >= 3 || merged.lo > merged.hi {
            return;
        }
        let ty = func.ty(value);
        if !ty.is_small_int() {
            return;
        }
        let max = ty.unsigned_max();
        if merged.lo == 0 && merged.hi == max {
            return;
        }
        match ranges.op(func, value) {
            Some(&Op::Convert(Convert::Uextend, inner)) if func.ty(inner).is_small_int() => {
                let imax = func.ty(inner).unsigned_max();
                if merged.lo <= imax {
                    self.add_with_backprop(func, ranges, inner, Fact::range(merged.lo, merged.hi.min(imax)), depth + 1);
                }
            }
            Some(&Op::Binary(Binary::Isub, a, b)) => {
                let fb = self.fact(ranges, b);
                if fb.lo <= fb.hi && fb.hi.checked_add(merged.hi).is_some_and(|bh| bh <= max) {
                    self.add_with_backprop(func, ranges, a, Fact::range(fb.lo + merged.lo, fb.hi + merged.hi), depth + 1);
                }
                let fa = self.fact(ranges, a);
                if fa.lo <= fa.hi && merged.hi <= fa.lo {
                    self.add_with_backprop(func, ranges, b, Fact::range(fa.lo - merged.hi, fa.hi - merged.lo), depth + 1);
                }
            }
            Some(&Op::Binary(Binary::Iadd, a, b)) => {
                for (var, k_val) in [(a, b), (b, a)] {
                    let Some(k) = self.fact(ranges, k_val).exact() else { continue };
                    if merged.lo >= k {
                        self.add_with_backprop(func, ranges, var, Fact::range(merged.lo - k, merged.hi - k), depth + 1);
                    } else if merged.hi < k {
                        let c = (max - k) + 1;
                        if c.checked_add(merged.hi).is_some_and(|ch| ch <= max) {
                            self.add_with_backprop(func, ranges, var, Fact::range(c + merged.lo, c + merged.hi), depth + 1);
                        }
                    } else if self.is_no_wrap_add(ranges, ty, var, k_val, k) && merged.hi >= k {
                        self.add_with_backprop(func, ranges, var, Fact::range(merged.lo.saturating_sub(k), merged.hi - k), depth + 1);
                    }
                }
            }
            Some(&Op::Select(cond, a, b)) => self.backprop_select(func, ranges, &merged, (cond, a, b), depth),
            _ => {}
        }
    }

    fn backprop_select(&mut self, func: &Function, ranges: &Ranges, merged: &Fact, (cond, a, b): (Value, Value, Value), depth: u8) {
        let fc = self.fact(ranges, cond);
        let a_fit = self.fact(ranges, a).meet(merged);
        let b_fit = self.fact(ranges, b).meet(merged);
        let (can_a, can_b) = (fc.hi != 0 && valid(&a_fit), fc.admits(0) && valid(&b_fit));
        let cty = func.ty(cond);
        match (can_a, can_b) {
            (true, false) => {
                self.add_with_backprop(func, ranges, a, a_fit, depth + 1);
                if cty.is_small_int() {
                    self.add_with_backprop(func, ranges, cond, Fact::range(1, cty.unsigned_max()), depth + 1);
                }
            }
            (false, true) => {
                self.add_with_backprop(func, ranges, b, b_fit, depth + 1);
                if cty.is_small_int() {
                    self.add_with_backprop(func, ranges, cond, Fact::range(0, 0), depth + 1);
                }
            }
            _ => {}
        }
    }

    /// Everything known about `value` here, with its definition evaluated again on what the path knows of its operands.
    pub(super) fn refined_fact(&self, func: &Function, ranges: &Ranges, value: Value, depth: u8) -> Fact {
        if self.is_empty() {
            return ranges.of(value);
        }
        let fact = self.fact(ranges, value);
        if fact.exact().is_some() {
            return fact;
        }
        let ty = func.ty(value);
        let Some(op) = ranges.op(func, value).filter(|op| {
            depth < 3
                && ty.is_small_int()
                && matches!(op, Op::Binary(..) | Op::Unary(..) | Op::Convert(..) | Op::Select(..) | Op::Icmp(..) | Op::Fits(..))
        }) else {
            return fact;
        };
        let memo: RefCell<[Option<(Value, Fact)>; OPERAND_MEMO]> = RefCell::new(std::array::from_fn(|_| None));
        let operand = |inner: Value| {
            if let Some((_, known)) = memo.borrow().iter().flatten().find(|(at, _)| *at == inner) {
                return known.clone();
            }
            let known = Some(self.refined_fact(func, ranges, inner, depth + 1)).filter(|known| known.lo <= known.hi).unwrap_or_else(|| ranges.of(inner));
            if let Some(slot) = memo.borrow_mut().iter_mut().find(|slot| slot.is_none()) {
                *slot = Some((inner, known.clone()));
            }
            known
        };
        let fact = fact.meet(&transfer(func, ranges, op, ty, &operand));
        let sum = match *op {
            Op::Binary(Binary::Iadd, a, b) if self.has_no_wrap_add(a, b) => unwrapped(Binary::Iadd, ty, &operand(a), &operand(b)),
            _ => None,
        };
        match sum {
            Some(sum) => fact.meet(&sum),
            None => fact,
        }
    }

    fn iadd_chain(&self, func: &Function, ranges: &Ranges, val: Value, ty: Type) -> Chain {
        let max = ty.unsigned_max();
        let mut out = [(val, val, 0_u64, 0_u64); CHAIN_STEPS];
        let mut len = 0;
        let (mut cur, mut acc) = (val, 0_u64);
        while len < CHAIN_STEPS {
            let Some(&Op::Binary(Binary::Iadd, p, q)) = ranges.op(func, cur) else { break };
            let Some((base, c_val, k)) = [(p, q), (q, p)]
                .into_iter()
                .find_map(|(b, c)| self.fact(ranges, c).exact().map(|k| (b, c, k)))
            else {
                break;
            };
            let Some(next_acc) = acc.checked_add(k).filter(|&s| s <= max) else { break };
            out[len] = (base, c_val, k, next_acc);
            len += 1;
            cur = base;
            acc = next_acc;
        }
        (out, len)
    }

    fn add_const_chain(&self, func: &Function, ranges: &Ranges, val: Value, ty: Type) -> ([(Value, u64); CHAIN_STEPS + 1], usize) {
        let (steps, len) = self.iadd_chain(func, ranges, val, ty);
        let mut out = [(val, 0_u64); CHAIN_STEPS + 1];
        for (slot, &(base, _, _, acc)) in out[1..].iter_mut().zip(&steps[..len]) {
            *slot = (base, acc);
        }
        (out, len + 1)
    }

    fn non_wrap_iadd_chain(&self, func: &Function, ranges: &Ranges, val: Value, ty: Type) -> Steps {
        let max = ty.unsigned_max();
        let (steps, len) = self.iadd_chain(func, ranges, val, ty);
        let mut out = [(val, 0_u64); CHAIN_STEPS];
        let (mut kept, mut all_steps) = (0, true);
        for (step, &(base, c_val, k, next_acc)) in steps[..len].iter().enumerate() {
            let step_ok = self.is_no_wrap_add(ranges, ty, base, c_val, k);
            all_steps &= step_ok;
            let total_ok = if step == 0 {
                step_ok
            } else {
                all_steps || max.checked_sub(next_acc).is_some_and(|limit| self.reaches_upper_at_most(ranges, ty, base, limit))
            };
            if total_ok {
                out[kept] = (base, next_acc);
                kept += 1;
            }
        }
        (out, kept)
    }

    fn min_reachable_hi(&self, ranges: &Ranges, ty: Type, start: Value) -> u64 {
        let mut best = self.fact(ranges, start).hi;
        if self.rels().is_empty() {
            return best;
        }
        self.walk(ty, start, UNSIGNED, |next, next_strict| {
            let nhi = self.fact(ranges, next).hi;
            best = best.min(if next_strict { nhi.saturating_sub(1) } else { nhi });
            false
        });
        best
    }

    pub(super) fn upper_bound(&self, func: &Function, ranges: &Ranges, ty: Type, x: Value, depth: u8) -> u64 {
        let max = ty.unsigned_max();
        let mut hi = self.refined_fact(func, ranges, x, 0).hi;
        if self.rels().is_empty() {
            return hi;
        }
        hi = hi.min(self.min_reachable_hi(ranges, ty, x));
        if depth < 2
            && let Some(&Op::Binary(Binary::Iadd, p, q)) = ranges.op(func, x)
        {
            for (base, c_val) in [(p, q), (q, p)] {
                if let Some(k) = self.fact(ranges, c_val).exact() {
                    let base_hi = self.upper_bound(func, ranges, ty, base, depth + 1);
                    if let Some(sum) = base_hi.checked_add(k).filter(|&s| s <= max) {
                        hi = hi.min(sum);
                    } else if self.is_no_wrap_pair(ranges, ty, base, c_val, k) {
                        hi = hi.min(base_hi.min(max - k) + k);
                    }
                }
            }
        }
        let mut from_x = None;
        for &r in self.rels() {
            if r.ty != ty {
                continue;
            }
            let pairs = match r.kind {
                RelKind::Ult => [Some((r.lhs, r.rhs, 1_u64)), None],
                RelKind::Ule => [Some((r.lhs, r.rhs, 0_u64)), None],
                RelKind::Eq => [Some((r.lhs, r.rhs, 0_u64)), Some((r.rhs, r.lhs, 0_u64))],
                _ => [None, None],
            };
            for (lhs, rhs, extra) in pairs.into_iter().flatten() {
                if !matches!(ranges.op(func, lhs), Some(Op::Binary(Binary::Iadd, ..))) {
                    continue;
                }
                let mut rhs_hi = None;
                let (steps, len) = self.non_wrap_iadd_chain(func, ranges, lhs, ty);
                for &(s_base, k) in &steps[..len] {
                    if self.reaches_in(*from_x.get_or_insert_with(|| self.reached(ty, x, UNSIGNED)), s_base) {
                        let rhi = *rhs_hi.get_or_insert_with(|| self.min_reachable_hi(ranges, ty, rhs));
                        if let Some(bound) = k.checked_add(extra).and_then(|d| rhi.checked_sub(d)) {
                            hi = hi.min(bound);
                        }
                    }
                }
            }
        }
        hi
    }

    pub(super) fn proves_add_rel(&self, func: &Function, ranges: &Ranges, ty: Type, a: Value, b: Value, strict: bool) -> bool {
        let is_add = |v: Value| matches!(ranges.op(func, v), Some(Op::Binary(Binary::Iadd, ..)));
        let b_is_add = is_add(b);
        let has_u = self.has_rel_kind(kind_bit(RelKind::Ult) | kind_bit(RelKind::Ule) | kind_bit(RelKind::Eq));
        if !has_u && !b_is_add {
            return false;
        }
        let into_b = has_u && self.steps_into(ty, b, UNSIGNED);
        let a_is_add = is_add(a);
        if !b_is_add && !into_b && !a_is_add {
            return false;
        }
        let (forms, len) = if a_is_add { self.add_const_chain(func, ranges, a, ty) } else { ([(a, 0); CHAIN_STEPS + 1], 1) };
        let a_forms = &forms[..len];
        let multi = a_forms.len() > 1;
        if !b_is_add && !into_b && !multi {
            return false;
        }
        let (b_steps, b_len) = if b_is_add { self.non_wrap_iadd_chain(func, ranges, b, ty) } else { ([(b, 0); CHAIN_STEPS], 0) };
        let mut uppers = self.uppers.0.borrow_mut();
        let (values, any_add) = if has_u { uppers.of(func, ranges, self.rels(), ty) } else { (&mut [][..], false) };
        if !multi && b_len == 0 && !(into_b && any_add) {
            return false;
        }
        if into_b && any_add {
            for (s, s_is_add, steps) in values.iter_mut() {
                if *s_is_add && *s != b {
                    steps.get_or_insert_with(|| self.non_wrap_iadd_chain(func, ranges, *s, ty));
                }
            }
        }
        let mut has_any_term = b_len > 0;
        let mut any_pos = b_steps[..b_len].iter().any(|&(_, k)| k > 0);
        if !has_any_term || (!multi && !any_pos) {
            for &(s, s_is_add, ref steps) in values.iter().filter(|(s, _, _)| into_b || *s == b) {
                if multi {
                    has_any_term = true;
                }
                if s_is_add
                    && s != b
                    && let Some((st, slen)) = steps
                    && *slen > 0
                {
                    has_any_term = true;
                    any_pos |= st[..*slen].iter().any(|&(_, k)| k > 0);
                }
            }
            if !has_any_term || (!multi && !any_pos) {
                return false;
            }
        }
        for &(a_base, c) in a_forms {
            let mut from_a = None;
            let mut try_term = |s: Value, s_base: Value, k: u64| -> bool {
                let require_strict = if c <= k {
                    strict && c == k
                } else if !strict && k.checked_add(1) == Some(c) {
                    true
                } else {
                    return false;
                };
                (s == b || into_b)
                    && self.reaches_in(*from_a.get_or_insert_with(|| self.reached(ty, a_base, UNSIGNED)), s_base)
                    && self.reaches(ty, s, b, UNSIGNED, require_strict)
            };
            for &(b_base, k) in &b_steps[..b_len] {
                if try_term(b, b_base, k) {
                    return true;
                }
            }
            for &(s, s_is_add, ref steps) in values.iter().filter(|(s, _, _)| into_b || *s == b) {
                if multi && try_term(s, s, 0) {
                    return true;
                }
                if s_is_add
                    && s != b
                    && let Some((st, slen)) = steps
                {
                    for &(s_base, k) in &st[..*slen] {
                        if try_term(s, s_base, k) {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }
}

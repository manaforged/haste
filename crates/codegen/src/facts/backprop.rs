use crate::ir::{Binary, Convert, Function, Op, Type, Value};

use super::rel::{Path, RelKind, UNSIGNED};
use super::transfer::{transfer, unwrapped, valid};
use super::{Fact, Ranges};

const CHAIN_STEPS: usize = 4;

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
        let fact = self.fact(ranges, value);
        let ty = func.ty(value);
        let Some(op) = ranges.op(func, value).filter(|_| depth < 3 && ty.is_small_int()) else { return fact };
        let operand = |inner: Value| Some(self.refined_fact(func, ranges, inner, depth + 1)).filter(|known| known.lo <= known.hi).unwrap_or_else(|| ranges.of(inner));
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

    fn add_const_chain(&self, func: &Function, ranges: &Ranges, val: Value, ty: Type) -> ([(Value, u64); CHAIN_STEPS + 1], usize) {
        let max = ty.unsigned_max();
        let mut out = [(val, 0_u64); CHAIN_STEPS + 1];
        let mut len = 1;
        let (mut cur, mut acc) = (val, 0_u64);
        for _ in 0..CHAIN_STEPS {
            let Some(&Op::Binary(Binary::Iadd, p, q)) = ranges.op(func, cur) else { break };
            let Some((base, k)) = [(p, q), (q, p)].into_iter().find_map(|(b, c)| self.fact(ranges, c).exact().map(|k| (b, k))) else {
                break;
            };
            let Some(next_acc) = acc.checked_add(k).filter(|&s| s <= max) else { break };
            out[len] = (base, next_acc);
            len += 1;
            cur = base;
            acc = next_acc;
        }
        (out, len)
    }

    fn non_wrap_iadd_chain(&self, func: &Function, ranges: &Ranges, val: Value, ty: Type) -> Vec<(Value, u64)> {
        let max = ty.unsigned_max();
        let mut out = Vec::new();
        let (mut cur, mut acc, mut all_steps) = (val, 0_u64, true);
        for step in 0..CHAIN_STEPS {
            let Some(&Op::Binary(Binary::Iadd, p, q)) = ranges.op(func, cur) else { break };
            let Some((base, c_val, k)) = [(p, q), (q, p)]
                .into_iter()
                .find_map(|(b, c)| self.fact(ranges, c).exact().map(|k| (b, c, k)))
            else {
                break;
            };
            let Some(next_acc) = acc.checked_add(k).filter(|&s| s <= max) else { break };
            let step_ok = self.is_no_wrap_add(ranges, ty, base, c_val, k);
            all_steps &= step_ok;
            let total_ok = if step == 0 {
                step_ok
            } else {
                all_steps || max.checked_sub(next_acc).is_some_and(|limit| self.reaches_upper_at_most(ranges, ty, base, limit))
            };
            if total_ok {
                out.push((base, next_acc));
            }
            cur = base;
            acc = next_acc;
        }
        out
    }

    fn min_reachable_hi(&self, ranges: &Ranges, ty: Type, start: Value) -> u64 {
        let mut best = self.fact(ranges, start).hi;
        if self.rels.is_empty() {
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
        if self.rels.is_empty() {
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
        let mut linked = None;
        for &r in &self.rels {
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
                for (s_base, k) in self.non_wrap_iadd_chain(func, ranges, lhs, ty) {
                    if self.linked_reach(ty, *linked.get_or_insert_with(|| self.steps_from(ty, x, UNSIGNED)), x, s_base, UNSIGNED) {
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
        if self.rels.is_empty() && !is_add(b) {
            return false;
        }
        let (forms, len) = self.add_const_chain(func, ranges, a, ty);
        let a_forms = &forms[..len];
        let into_b = self.steps_into(ty, b, UNSIGNED);
        let mut upper_terms: Vec<(Value, Value, u64)> = Vec::new();
        if is_add(b) {
            for (b_base, k) in self.non_wrap_iadd_chain(func, ranges, b, ty) {
                upper_terms.push((b, b_base, k));
            }
        }
        let candidates = self.rels.iter().filter(|r| r.ty == ty).flat_map(|r| match r.kind {
            RelKind::Ult | RelKind::Ule => [Some(r.lhs), None],
            RelKind::Eq => [Some(r.lhs), Some(r.rhs)],
            _ => [None, None],
        });
        for (i, s) in candidates.clone().flatten().enumerate() {
            if (!into_b && s != b) || candidates.clone().flatten().take(i).any(|seen| seen == s) {
                continue;
            }
            if a_forms.len() > 1 {
                upper_terms.push((s, s, 0));
            }
            if s != b && is_add(s) {
                for (s_base, k) in self.non_wrap_iadd_chain(func, ranges, s, ty) {
                    upper_terms.push((s, s_base, k));
                }
            }
        }
        if upper_terms.is_empty() || (a_forms.len() == 1 && !upper_terms.iter().any(|&(_, _, k)| k > 0)) {
            return false;
        }
        for &(a_base, c) in a_forms {
            let linked = self.steps_from(ty, a_base, UNSIGNED);
            for &(s, s_base, k) in &upper_terms {
                let require_strict = if c <= k {
                    strict && c == k
                } else if !strict && k.checked_add(1) == Some(c) {
                    true
                } else {
                    continue;
                };
                if (s == b || into_b)
                    && self.linked_reach(ty, linked, a_base, s_base, UNSIGNED)
                    && self.reaches(ty, s, b, UNSIGNED, require_strict)
                {
                    return true;
                }
            }
        }
        false
    }
}

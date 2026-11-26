use crate::ir::{Function, Value};
use crate::live::{Live, overlaps};

use super::choose::reuse_pair;
use super::{Hints, Pool, SlotSet, crosses_call, slot_accepts};

/// What `join_back_args` reads besides the homes it rewrites.
pub(super) struct Join<'a, F> {
    pub(super) func: &'a Function,
    pub(super) live: &'a Live,
    pub(super) pool: &'a Pool<F>,
    pub(super) hints: &'a Hints<'a>,
    pub(super) back_partners: &'a [Vec<usize>],
    pub(super) reverse_reuse: &'a [Option<Value>],
}

/// The most values `chain` gathers into one group move.
const CHAIN: usize = 8;

impl<F: Copy + PartialEq> Join<'_, F> {
    fn keeps_copy(&self, homes: &[Option<F>], value: usize, except: &[usize], reg: F) -> bool {
        let at = |q: usize| !except.contains(&q) && homes.get(q).copied().flatten() == Some(reg);
        let reuse = reuse_pair(self.hints, self.reverse_reuse, value);
        self.back_partners[value].iter().any(|&q| at(q))
            || self
                .hints
                .partners
                .is_some_and(|m| m[value].iter().any(|q| at(q.index())))
            || (self.hints.two_address && reuse.into_iter().flatten().any(|q| at(q.index())))
            || self.pool.passed[value].is_some_and(|(r, _)| r == reg)
    }

    fn suits(&self, value: usize, from: usize, to: usize) -> bool {
        let saved = &self.pool.is_saved;
        !(self.func.types[value].bytes() > 8 && saved[to])
            && (saved[to] || !saved[from] || !crosses_call(self.live, value))
    }

    fn rehome(
        &self,
        (slots, homes): (&mut SlotSet, &mut [Option<F>]),
        value: usize,
        partner: usize,
    ) -> bool {
        let Some(to) = homes[partner].and_then(|reg| self.pool.slot_of(reg)) else {
            return false;
        };
        let chain = self.chain(homes, value);
        self.move_group((slots, homes), &chain, to, partner)
    }

    /// `value` and the values that reuse its register in place or whose register it reuses, transitively, while they share its register.
    fn chain(&self, homes: &[Option<F>], value: usize) -> Vec<usize> {
        let mut chain = vec![value];
        let mut at = 0;
        while self.hints.two_address && at < chain.len() && chain.len() < CHAIN {
            let next = chain[at];
            at += 1;
            for q in reuse_pair(self.hints, self.reverse_reuse, next)
                .into_iter()
                .flatten()
                .map(|q| q.index())
            {
                if !chain.contains(&q)
                    && homes.get(q).copied().flatten().is_some()
                    && homes[q] == homes[value]
                {
                    chain.push(q);
                }
            }
        }
        chain
    }

    /// Moves `value` into register slot `to` when it fits there and leaves no copy with a partner other than `partner` behind.
    fn move_to(
        &self,
        homes: (&mut SlotSet, &mut [Option<F>]),
        value: usize,
        to: usize,
        partner: usize,
    ) -> bool {
        self.move_group(homes, &[value], to, partner)
    }

    /// Moves every value of `group`, which share one register, into register slot `to` when each fits there and none leaves a copy behind with a value outside the group other than `partner`.
    fn move_group(
        &self,
        (slots, homes): (&mut SlotSet, &mut [Option<F>]),
        group: &[usize],
        to: usize,
        partner: usize,
    ) -> bool {
        let Some(from_reg) = homes[group[0]] else {
            return false;
        };
        let Some(from) = self.pool.slot_of(from_reg) else {
            return false;
        };
        let except: Vec<usize> = group.iter().copied().chain([partner]).collect();
        let fits = |value: usize| {
            self.live.bounds(value).is_some_and(|(start, _)| {
                slots.free(self.pool, to, self.live.ranges_of(value), start, None)
            }) && self.pool.kept[value].is_none()
                && !self.keeps_copy(homes, value, &except, from_reg)
                && self.suits(value, from, to)
        };
        if from == to || !group.iter().all(|&value| fits(value)) {
            return false;
        }
        for &value in group {
            let ranges = self.live.ranges_of(value);
            slots.remove(from, ranges, value);
            slots.insert(to, ranges, value);
            homes[value] = Some(self.pool.regs[to]);
        }
        true
    }

    /// Moves the one value that holds `partner`'s register during `value` to another free register, then `value` into that register; undoes the first move when the second fails.
    fn evict(
        &self,
        (slots, homes): (&mut SlotSet, &mut [Option<F>]),
        value: usize,
        partner: usize,
    ) -> bool {
        let Some(to) = homes[partner].and_then(|reg| self.pool.slot_of(reg)) else {
            return false;
        };
        let mut blockers = Vec::new();
        slots.collect_overlaps(to, self.live.ranges_of(value), &mut blockers);
        blockers.dedup();
        let [blocker] = blockers[..] else {
            return false;
        };
        if blocker == partner || blocker == value {
            return false;
        }
        let Some(old) = (0..self.pool.regs.len()).find(|&other| {
            other != to
                && homes[value] != Some(self.pool.regs[other])
                && self.move_to((&mut *slots, &mut *homes), blocker, other, usize::MAX)
        }) else {
            return false;
        };
        if self.move_to((&mut *slots, &mut *homes), value, to, partner) {
            return true;
        }
        let ranges = self.live.ranges_of(blocker);
        slots.remove(old, ranges, blocker);
        slots.insert(to, ranges, blocker);
        homes[blocker] = Some(self.pool.regs[to]);
        false
    }

    fn slot_holders(slots: &SlotSet, slot: usize) -> Vec<usize> {
        let mut out: Vec<usize> = slots.slots[slot].iter().map(|&(_, _, h)| h).collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    fn collect_swap(
        &self,
        holders_from: &[usize],
        holders_to: &[usize],
        (from, to): (usize, usize),
        seed: &[usize],
        anchor: usize,
    ) -> Option<(Vec<usize>, Vec<usize>)> {
        let (mut move_from, mut move_to) = (seed.to_vec(), Vec::new());
        let (mut fi, mut ti) = (0, 0);
        while fi < move_from.len() || ti < move_to.len() {
            while fi < move_from.len() {
                let ur = self.live.ranges_of(move_from[fi]);
                fi += 1;
                for &v in holders_to {
                    if !move_to.contains(&v) && overlaps(ur, self.live.ranges_of(v)) {
                        if v == anchor {
                            return None;
                        }
                        move_to.push(v);
                    }
                }
            }
            while ti < move_to.len() {
                let vr = self.live.ranges_of(move_to[ti]);
                ti += 1;
                for &u in holders_from {
                    if !move_from.contains(&u) && overlaps(vr, self.live.ranges_of(u)) {
                        move_from.push(u);
                    }
                }
            }
        }
        if move_from.is_empty() || move_to.is_empty() {
            return None;
        }
        let ok = |list: &[usize], src: usize, dst: usize| {
            list.iter().all(|&h| {
                self.pool.kept[h].is_none()
                    && self.live.bounds(h).is_some_and(|(start, _)| {
                        slot_accepts(self.pool, dst, self.live.ranges_of(h), start, None)
                    })
                    && self.suits(h, src, dst)
            })
        };
        (ok(&move_from, from, to) && ok(&move_to, to, from)).then_some((move_from, move_to))
    }

    fn swap_delta(
        &self,
        homes: &[Option<F>],
        (slot_a, slot_b): (usize, usize),
        (move_a, move_b): (&[usize], &[usize]),
    ) -> i128 {
        let (reg_a, reg_b) = (self.pool.regs[slot_a], self.pool.regs[slot_b]);
        let moved = |v: usize| move_a.contains(&v) || move_b.contains(&v);
        let pair_weight = |u: usize, v: usize| self.hints.weights.map_or(1, |w| w[u].min(w[v]));
        let mut delta = 0_i128;
        let mut account = |v: usize, q: usize, old_v: F, new_v: F| {
            if moved(q) {
                return;
            }
            let Some(reg_q) = homes.get(q).copied().flatten() else {
                return;
            };
            let w = i128::from(pair_weight(v, q));
            if reg_q == new_v {
                delta -= w;
            } else if reg_q == old_v {
                delta += w;
            }
        };
        for (list, old_v, new_v) in [(move_a, reg_a, reg_b), (move_b, reg_b, reg_a)] {
            for &v in list {
                if let Some(partners) = self.hints.partners {
                    for q in &partners[v] {
                        account(v, q.index(), old_v, new_v);
                    }
                } else {
                    for &q in &self.back_partners[v] {
                        account(v, q, old_v, new_v);
                    }
                }
                if self.hints.two_address {
                    for q in reuse_pair(self.hints, self.reverse_reuse, v)
                        .into_iter()
                        .flatten()
                    {
                        account(v, q.index(), old_v, new_v);
                    }
                }
            }
        }
        delta
    }

    fn swap_slots(
        &self,
        slots: &mut SlotSet,
        homes: &mut [Option<F>],
        (slot_a, slot_b): (usize, usize),
        (param, arg): (usize, usize),
    ) -> bool {
        if slot_a == slot_b {
            return false;
        }
        let holders_a = Self::slot_holders(slots, slot_a);
        let holders_b = Self::slot_holders(slots, slot_b);
        let mut best: Option<(i128, usize, usize, Vec<usize>, Vec<usize>)> = None;
        let mut consider = |from: usize,
                            to: usize,
                            holders_from: &[usize],
                            holders_to: &[usize],
                            seed: &[usize],
                            anchor: usize| {
            if let Some((move_from, move_to)) =
                self.collect_swap(holders_from, holders_to, (from, to), seed, anchor)
            {
                let delta = self.swap_delta(homes, (from, to), (&move_from, &move_to));
                if delta < 0 && best.as_ref().is_none_or(|b| delta < b.0) {
                    best = Some((delta, from, to, move_from, move_to));
                }
            }
        };
        for (from, to, h_from, h_to, val, partner) in [
            (slot_b, slot_a, &holders_b[..], &holders_a[..], arg, param),
            (slot_a, slot_b, &holders_a[..], &holders_b[..], param, arg),
        ] {
            let chain = self.chain(homes, val);
            if chain.len() > 1 {
                consider(from, to, h_from, h_to, &chain, partner);
            }
            consider(from, to, h_from, h_to, &[val], partner);
        }
        let Some((_, from, to, move_from, move_to)) = best else {
            return false;
        };
        for &h in &move_from {
            slots.remove(from, self.live.ranges_of(h), h);
        }
        for &h in &move_to {
            slots.remove(to, self.live.ranges_of(h), h);
        }
        for (list, dst) in [(&move_from, to), (&move_to, from)] {
            for &h in list {
                slots.insert(dst, self.live.ranges_of(h), h);
                homes[h] = Some(self.pool.regs[dst]);
            }
        }
        true
    }

    /// Gives each edge argument of `args` and its parameter one register when one of them can take the other's without losing another copy, in list order.
    pub(super) fn join_args(
        &self,
        args: &[(usize, usize)],
        slots: &mut SlotSet,
        homes: &mut [Option<F>],
    ) {
        let is_back = std::ptr::eq(args, self.hints.back_args);
        for &(param, arg) in args {
            if param == arg || homes[param].is_none() || homes[param] == homes[arg] {
                continue;
            }
            let aligned = self.rehome((&mut *slots, &mut *homes), arg, param)
                || self.rehome((&mut *slots, &mut *homes), param, arg)
                || self.evict((&mut *slots, &mut *homes), arg, param)
                || self.evict((&mut *slots, &mut *homes), param, arg);
            if !aligned
                && is_back
                && let Some(slot_a) = homes[param].and_then(|reg| self.pool.slot_of(reg))
                && let Some(slot_b) = homes[arg].and_then(|reg| self.pool.slot_of(reg))
            {
                let _ = self.swap_slots(slots, homes, (slot_a, slot_b), (param, arg));
            }
        }
    }

    /// Gives each zero-extension copy its source's register, which holds the same bits for as long as the copy lives.
    pub(super) fn share_copies(&self, slots: &mut SlotSet, homes: &mut [Option<F>]) {
        for &(copy, source) in self.hints.copies {
            let Some(reg) = homes[source] else { continue };
            if homes[copy] == Some(reg) {
                continue;
            }
            if let Some(from) = homes[copy].and_then(|old| self.pool.slot_of(old)) {
                slots.remove(from, self.live.ranges_of(copy), copy);
            }
            homes[copy] = Some(reg);
        }
    }
}

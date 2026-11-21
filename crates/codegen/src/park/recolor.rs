use crate::ir::{Function, Value};
use crate::live::Live;

use super::{Hints, Pool, SlotSet, crosses_call};

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
        let reuse = [self.hints.reuse[value], self.reverse_reuse.get(value).copied().flatten()];
        self.back_partners[value].iter().any(|&q| at(q))
            || self.hints.partners.is_some_and(|m| m[value].iter().any(|q| at(q.index())))
            || (self.hints.two_address && reuse.into_iter().flatten().any(|q| at(q.index())))
            || self.pool.passed[value].is_some_and(|(r, _)| r == reg)
    }

    fn suits(&self, value: usize, from: usize, to: usize) -> bool {
        let saved = &self.pool.is_saved;
        !(self.func.types[value].bytes() > 8 && saved[to]) && (saved[to] || !saved[from] || !crosses_call(self.live, value))
    }

    fn rehome(&self, (slots, homes): (&mut SlotSet, &mut [Option<F>]), value: usize, partner: usize) -> bool {
        let Some(to) = homes[partner].and_then(|reg| self.pool.slot_of(reg)) else { return false };
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
            for q in [self.hints.reuse[next], self.reverse_reuse.get(next).copied().flatten()].into_iter().flatten().map(|q| q.index()) {
                if !chain.contains(&q) && homes.get(q).copied().flatten().is_some() && homes[q] == homes[value] {
                    chain.push(q);
                }
            }
        }
        chain
    }

    /// Moves `value` into register slot `to` when it fits there and leaves no copy with a partner other than `partner` behind.
    fn move_to(&self, homes: (&mut SlotSet, &mut [Option<F>]), value: usize, to: usize, partner: usize) -> bool {
        self.move_group(homes, &[value], to, partner)
    }

    /// Moves every value of `group`, which share one register, into register slot `to` when each fits there and none leaves a copy behind with a value outside the group other than `partner`.
    fn move_group(&self, (slots, homes): (&mut SlotSet, &mut [Option<F>]), group: &[usize], to: usize, partner: usize) -> bool {
        let Some(from_reg) = homes[group[0]] else { return false };
        let Some(from) = self.pool.slot_of(from_reg) else { return false };
        let except: Vec<usize> = group.iter().copied().chain([partner]).collect();
        let fits = |value: usize| {
            self.live.bounds(value).is_some_and(|(start, _)| slots.free(self.pool, to, self.live.ranges_of(value), start, None))
                && self.pool.kept[value].is_none()
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
    fn evict(&self, (slots, homes): (&mut SlotSet, &mut [Option<F>]), value: usize, partner: usize) -> bool {
        let Some(to) = homes[partner].and_then(|reg| self.pool.slot_of(reg)) else { return false };
        let mut blockers = Vec::new();
        slots.collect_overlaps(to, self.live.ranges_of(value), &mut blockers);
        blockers.dedup();
        let [blocker] = blockers[..] else { return false };
        if blocker == partner || blocker == value {
            return false;
        }
        let Some(old) = (0..self.pool.regs.len()).find(|&other| other != to && homes[value] != Some(self.pool.regs[other]) && self.move_to((&mut *slots, &mut *homes), blocker, other, usize::MAX)) else { return false };
        if self.move_to((&mut *slots, &mut *homes), value, to, partner) {
            return true;
        }
        let ranges = self.live.ranges_of(blocker);
        slots.remove(old, ranges, blocker);
        slots.insert(to, ranges, blocker);
        homes[blocker] = Some(self.pool.regs[to]);
        false
    }

    /// Gives each edge argument of `args` and its parameter one register when one of them can take the other's without losing another copy, in list order.
    pub(super) fn join_args(&self, args: &[(usize, usize)], slots: &mut SlotSet, homes: &mut [Option<F>]) {
        for &(param, arg) in args {
            if param == arg || homes[param].is_none() || homes[param] == homes[arg] {
                continue;
            }
            let _ = self.rehome((&mut *slots, &mut *homes), arg, param)
                || self.rehome((&mut *slots, &mut *homes), param, arg)
                || self.evict((&mut *slots, &mut *homes), arg, param)
                || self.evict((&mut *slots, &mut *homes), param, arg);
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

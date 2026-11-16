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

impl<F: Copy + PartialEq> Join<'_, F> {
    fn keeps_copy(&self, homes: &[Option<F>], value: usize, partner: usize, reg: F) -> bool {
        let at = |q: usize| q != partner && homes.get(q).copied().flatten() == Some(reg);
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
        let (Some(from_reg), Some(to_reg)) = (homes[value], homes[partner]) else { return false };
        let (Some(from), Some(to), Some((start, _))) = (self.pool.slot_of(from_reg), self.pool.slot_of(to_reg), self.live.bounds(value)) else { return false };
        let ranges = self.live.ranges_of(value);
        if self.pool.kept[value].is_some() || self.keeps_copy(homes, value, partner, from_reg) || !self.suits(value, from, to) || !slots.free(self.pool, to, ranges, start, None) {
            return false;
        }
        slots.remove(from, ranges, value);
        slots.insert(to, ranges, value);
        homes[value] = Some(to_reg);
        true
    }

    /// Gives each back-edge argument and its parameter one register when one of them can take the other's without losing another copy, hottest edges first.
    pub(super) fn join_back_args(&self, slots: &mut SlotSet, homes: &mut [Option<F>]) {
        for &(param, arg) in self.hints.back_args {
            if param == arg || homes[param].is_none() || homes[param] == homes[arg] {
                continue;
            }
            if !self.rehome((&mut *slots, &mut *homes), arg, param) {
                self.rehome((&mut *slots, &mut *homes), param, arg);
            }
        }
    }
}

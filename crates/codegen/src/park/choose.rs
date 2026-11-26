use crate::ir::{Function, Value};
use crate::live::{Live, Range, overlaps};

use super::{
    BackIndex, Hints, Pool, SlotSet, crosses_call, crosses_ranges, parks, prefer_park, slot_free,
    try_place_weighted,
};

pub(super) fn reuse_pair(
    hints: &Hints,
    reverse_reuse: &[Option<Value>],
    value: usize,
) -> [Option<Value>; 2] {
    [
        hints.reuse[value],
        reverse_reuse.get(value).copied().flatten(),
    ]
}

fn order_pair<T>(two_address: bool, fwd: T, reuse: T) -> (T, T) {
    if two_address {
        (reuse, fwd)
    } else {
        (fwd, reuse)
    }
}

struct Unalloc {
    back: Vec<(usize, bool, u32)>,
    fwd: Vec<(usize, bool, u32)>,
    reuse: Vec<(usize, bool, u32)>,
}

struct Chooser<'a, F, U> {
    pool: &'a Pool<F>,
    holders: &'a [Vec<usize>],
    homes: &'a [Option<F>],
    live: &'a Live,
    hints: &'a Hints<'a>,
    back_partners: &'a [usize],
    reverse_reuse: &'a [Option<Value>],
    reserved: &'a [F],
    slot_used: U,
    value: usize,
    start: u32,
    end: u32,
    ranges: &'a [Range],
    kept_reg: Option<F>,
    wide_val: bool,
    needs_saved: bool,
    can_park: bool,
    prefer_park: bool,
    has_free_saved: bool,
}

impl<'a, F: Copy + PartialEq, U: Fn(usize) -> bool> Chooser<'a, F, U> {
    fn is_free(&self, s: usize) -> bool {
        slot_free(
            self.pool,
            s,
            &self.holders[s],
            self.live,
            self.ranges,
            self.start,
            self.kept_reg,
        )
    }

    fn suits_direct(&self, s: usize) -> bool {
        if self.needs_saved {
            self.pool.is_saved[s] && !self.wide_val
        } else {
            !self.wide_val || !self.pool.is_saved[s]
        }
    }

    fn suits_any(&self, s: usize) -> bool {
        self.suits_direct(s) || (self.can_park && !self.pool.is_saved[s])
    }

    fn can_use(&self, reg: F, check_reserved: bool) -> Option<usize> {
        if check_reserved && self.reserved.contains(&reg) {
            return None;
        }
        let s = self.pool.slot_of(reg)?;
        let ok = if self.needs_saved {
            (self.pool.is_saved[s] || (self.can_park && (!self.has_free_saved || self.prefer_park)))
                && !self.wide_val
        } else {
            !self.pool.is_saved[s] || (!self.wide_val && (self.slot_used)(s))
        };
        (ok && self.is_free(s)).then_some(s)
    }

    fn try_kept(&self) -> Option<usize> {
        let reg = self.kept_reg?;
        let s = self.pool.slot_of(reg)?;
        (self.suits_any(s) && self.is_free(s)).then_some(s)
    }

    fn try_partner(&self, p: usize, check_reserved: bool) -> Option<usize> {
        self.homes[p].and_then(|reg| self.can_use(reg, check_reserved))
    }

    fn try_fwd(&self) -> Option<usize> {
        self.hints.partners.and_then(|partners| {
            partners[self.value]
                .iter()
                .find_map(|p| self.try_partner(p.index(), true))
        })
    }

    fn unalloc_info(&self, p: usize) -> Option<(usize, bool, u32)> {
        if self.homes[p].is_some() {
            return None;
        }
        let pr = self.live.ranges_of(p);
        (!crosses_ranges(&self.live.unwinds, pr))
            .then(|| Some((p, crosses_call(self.live, p), pr.first()?.start)))
            .flatten()
    }

    fn collect_unalloc(&self) -> Unalloc {
        let partners = self.hints.partners.map_or(&[][..], |m| &m[self.value][..]);
        let back = self
            .back_partners
            .iter()
            .copied()
            .filter_map(|p| self.unalloc_info(p))
            .collect();
        let fwd = partners
            .iter()
            .map(|p| p.index())
            .filter(|p| !self.back_partners.contains(p))
            .filter_map(|p| self.unalloc_info(p))
            .collect();
        let reuse = reuse_pair(self.hints, self.reverse_reuse, self.value)
            .into_iter()
            .flatten()
            .map(|p| p.index())
            .filter(|&p| {
                p < self.homes.len()
                    && !self.back_partners.contains(&p)
                    && !partners.iter().any(|q| q.index() == p)
            })
            .filter_map(|p| self.unalloc_info(p))
            .collect();
        Unalloc { back, fwd, reuse }
    }

    fn partner_fits(&self, &(p, p_saved, p_start): &(usize, bool, u32), s: usize) -> bool {
        (!p_saved || self.wide_val || self.pool.is_saved[s])
            && slot_free(
                self.pool,
                s,
                &self.holders[s],
                self.live,
                self.live.ranges_of(p),
                p_start,
                self.pool.kept[p],
            )
    }

    fn all_fit(&self, list: &[(usize, bool, u32)], s: usize) -> bool {
        list.iter().all(|info| self.partner_fits(info, s))
    }

    fn try_reuse(&self, unalloc: &Unalloc) -> Option<usize> {
        for op in reuse_pair(self.hints, self.reverse_reuse, self.value)
            .into_iter()
            .flatten()
        {
            let Some(reg) = self.homes[op.index()] else {
                continue;
            };
            let Some(s) = self.pool.slot_of(reg) else {
                continue;
            };
            let suits = if self.has_free_saved {
                self.suits_direct(s)
            } else {
                self.suits_any(s)
            };
            if suits
                && (self.hints.two_address || self.all_fit(&unalloc.back, s))
                && !self.reserved.contains(&reg)
                && self.is_free(s)
            {
                return Some(s);
            }
        }
        None
    }

    fn try_passed(&self) -> Option<usize> {
        let (reg, at_call_end) = self.pool.passed.get(self.value).copied().flatten()?;
        let call_at = self.end + 1;
        let plain_call = self.live.calls.binary_search(&call_at).is_ok()
            && self.live.unwinds.binary_search(&call_at).is_err();
        if (!at_call_end || plain_call)
            && !crosses_call(self.live, self.value)
            && !self.reserved.contains(&reg)
            && let Some(s) = self.pool.slot_of(reg)
            && self.is_free(s)
        {
            return Some(s);
        }
        None
    }

    fn check_unalloc_reg(
        &self,
        unalloc: &Unalloc,
        info: (usize, bool, u32),
        pr: &[Range],
        reg: F,
        extra: Option<usize>,
    ) -> Option<usize> {
        let s = self.can_use(reg, true)?;
        if !self.partner_fits(&info, s) || !self.all_fit(&unalloc.back, s) {
            return None;
        }
        if let Some(q) = extra {
            let qr = self.live.ranges_of(q);
            if q == self.value || q == info.0 || overlaps(qr, self.ranges) || overlaps(qr, pr) {
                return None;
            }
        }
        Some(s)
    }

    fn try_unalloc_lookahead(&self, unalloc: &Unalloc) -> Option<usize> {
        let (first, second) = order_pair(self.hints.two_address, &unalloc.fwd, &unalloc.reuse);
        for &info in unalloc.back.iter().chain(first).chain(second) {
            let (p, p_saved, _) = info;
            let pr = self.live.ranges_of(p);
            if overlaps(pr, self.ranges) {
                continue;
            }
            let check = |reg: F, extra: Option<usize>| {
                self.check_unalloc_reg(unalloc, info, pr, reg, extra)
            };
            if let Some(s) = self.pool.kept[p].and_then(|r| check(r, None)) {
                return Some(s);
            }
            if !p_saved
                && let Some((r, _)) = self.pool.passed.get(p).copied().flatten()
                && let Some(s) = check(r, None)
            {
                return Some(s);
            }
            let p_reuse = reuse_pair(self.hints, self.reverse_reuse, p);
            for q in self
                .hints
                .partners
                .map_or(&[][..], |m| &m[p][..])
                .iter()
                .copied()
                .chain(p_reuse.into_iter().flatten())
                .map(|v| v.index())
                .filter(|&q| q != self.value && q < self.homes.len())
            {
                if let Some(s) = self.homes[q]
                    .and_then(|r| check(r, None))
                    .or_else(|| self.pool.kept[q].and_then(|r| check(r, Some(q))))
                {
                    return Some(s);
                }
            }
        }
        None
    }

    fn tier(&self, unalloc: &Unalloc, s: usize) -> u8 {
        let (sec, third) = order_pair(
            self.hints.two_address,
            !self.all_fit(&unalloc.fwd, s),
            !self.all_fit(&unalloc.reuse, s),
        );
        u8::from(!self.all_fit(&unalloc.back, s)) * 4 + u8::from(sec) * 2 + u8::from(third)
    }

    fn try_pick(
        &self,
        unalloc: &Unalloc,
        allow_reserved: bool,
        pred: impl Fn(usize) -> bool,
    ) -> Option<usize> {
        let free = |s: usize| !self.pool.is_saved[s] || (self.slot_used)(s);
        (0..self.pool.regs.len())
            .filter(|&s| {
                pred(s)
                    && (allow_reserved || !self.reserved.contains(&self.pool.regs[s]))
                    && self.is_free(s)
            })
            .min_by_key(|&s| {
                (
                    self.tier(unalloc, s),
                    !free(s),
                    self.holders[s].is_empty(),
                    s,
                )
            })
    }

    fn pick_saved(&self, unalloc: &Unalloc) -> Option<usize> {
        if self.prefer_park {
            return self
                .try_pick(unalloc, false, |s| {
                    self.pool.is_saved[s] && (self.slot_used)(s) && !self.wide_val
                })
                .or_else(|| self.try_pick(unalloc, false, |s| !self.pool.is_saved[s]))
                .or_else(|| {
                    self.try_pick(unalloc, false, |s| self.pool.is_saved[s] && !self.wide_val)
                })
                .or_else(|| self.try_pick(unalloc, true, |s| !self.pool.is_saved[s]))
                .or_else(|| {
                    self.try_pick(unalloc, true, |s| self.pool.is_saved[s] && !self.wide_val)
                });
        }
        self.try_pick(unalloc, false, |s| self.pool.is_saved[s] && !self.wide_val)
            .or_else(|| {
                self.can_park
                    .then(|| self.try_pick(unalloc, false, |s| !self.pool.is_saved[s]))
                    .flatten()
            })
            .or_else(|| self.try_pick(unalloc, true, |s| self.pool.is_saved[s] && !self.wide_val))
            .or_else(|| {
                self.can_park
                    .then(|| self.try_pick(unalloc, true, |s| !self.pool.is_saved[s]))
                    .flatten()
            })
    }

    fn pick_unsaved(&self, unalloc: &Unalloc) -> Option<usize> {
        let back_needs_saved =
            !self.wide_val && unalloc.back.iter().any(|&(_, p_saved, _)| p_saved);
        let fwd_needs_saved = !self.wide_val
            && unalloc
                .fwd
                .iter()
                .any(|&(p, p_saved, _)| p_saved && !parks(self.live, self.hints, p));
        let reuse_needs_saved = !self.wide_val
            && unalloc
                .reuse
                .iter()
                .any(|&(p, p_saved, _)| p_saved && !parks(self.live, self.hints, p));
        if back_needs_saved
            && let Some(s) = self.try_pick(unalloc, false, |s| {
                self.pool.is_saved[s] && self.all_fit(&unalloc.back, s)
            })
        {
            return Some(s);
        }
        if !unalloc.back.is_empty()
            && let Some(s) = self
                .try_pick(unalloc, false, |s| {
                    !self.pool.is_saved[s] && self.all_fit(&unalloc.back, s)
                })
                .or_else(|| {
                    (!self.wide_val)
                        .then(|| {
                            self.try_pick(unalloc, false, |s| {
                                self.pool.is_saved[s] && self.all_fit(&unalloc.back, s)
                            })
                        })
                        .flatten()
                })
        {
            return Some(s);
        }
        let (first_saved, second_saved) = order_pair(
            self.hints.two_address,
            (fwd_needs_saved, &unalloc.fwd[..]),
            (reuse_needs_saved, &unalloc.reuse[..]),
        );
        for (needed, list) in [first_saved, second_saved] {
            if needed
                && let Some(s) = self.try_pick(unalloc, false, |s| {
                    self.pool.is_saved[s] && self.all_fit(list, s)
                })
            {
                return Some(s);
            }
        }
        self.try_pick(unalloc, false, |s| !self.pool.is_saved[s])
            .or_else(|| {
                (!self.wide_val)
                    .then(|| {
                        self.try_pick(unalloc, false, |s| {
                            self.pool.is_saved[s] && (self.slot_used)(s)
                        })
                    })
                    .flatten()
            })
            .or_else(|| {
                (!self.wide_val)
                    .then(|| self.try_pick(unalloc, false, |s| self.pool.is_saved[s]))
                    .flatten()
            })
            .or_else(|| self.try_pick(unalloc, true, |s| !self.pool.is_saved[s]))
            .or_else(|| {
                (!self.wide_val)
                    .then(|| self.try_pick(unalloc, true, |s| self.pool.is_saved[s]))
                    .flatten()
            })
    }

    fn choose(&self) -> Option<usize> {
        if let Some(s) = self
            .try_kept()
            .or_else(|| {
                self.back_partners
                    .iter()
                    .find_map(|&p| self.try_partner(p, false))
            })
            .or_else(|| (!self.hints.two_address).then(|| self.try_fwd()).flatten())
        {
            return Some(s);
        }
        let unalloc = self.collect_unalloc();
        if let Some(s) = self
            .try_reuse(&unalloc)
            .or_else(|| self.hints.two_address.then(|| self.try_fwd()).flatten())
            .or_else(|| self.try_passed())
            .or_else(|| self.try_unalloc_lookahead(&unalloc))
        {
            return Some(s);
        }
        if self.needs_saved {
            self.pick_saved(&unalloc)
        } else {
            self.pick_unsaved(&unalloc)
        }
    }
}

pub(super) fn choose_free<F: Copy + PartialEq>(
    pool: &Pool<F>,
    holders: &[Vec<usize>],
    homes: &[Option<F>],
    live: &Live,
    hints: &Hints,
    back_partners: &[usize],
    reverse_reuse: &[Option<Value>],
    reserved: &[F],
    slot_used: impl Fn(usize) -> bool,
    (value, start, end, wide_val, needs_saved, can_park, has_free_saved): (
        usize,
        u32,
        u32,
        bool,
        bool,
        bool,
        bool,
    ),
) -> Option<usize> {
    let ctx = Chooser {
        pool,
        holders,
        homes,
        live,
        hints,
        back_partners,
        reverse_reuse,
        reserved,
        slot_used,
        value,
        start,
        end,
        ranges: live.ranges_of(value),
        kept_reg: pool.kept[value],
        wide_val,
        needs_saved,
        can_park,
        prefer_park: can_park && prefer_park(live, hints, value),
        has_free_saved,
    };
    ctx.choose()
}

/// Places the unhomed reuse partners of a just-placed value in its register now, when that is their own current choice.
pub(super) fn place_reuse_chain<'a, F: Copy + PartialEq>(
    ctx: (&Function, &'a Live, &Pool<F>),
    weighted: (&Hints, &[u64]),
    index: (&[Vec<usize>], &[Option<Value>], &BackIndex),
    (slots, homes): (&mut SlotSet, &mut [Option<F>]),
    (reserved, scratch_holders, query_ranges): (
        &mut Vec<F>,
        &mut Vec<Vec<usize>>,
        &mut Vec<(usize, &'a [Range])>,
    ),
    (pooled, value): (&[bool], usize),
) {
    let (_, _, pool) = ctx;
    let (hints, _) = weighted;
    let (_, reverse_reuse, _) = index;
    let Some(slot) = homes[value].and_then(|reg| pool.slot_of(reg)) else {
        return;
    };
    let mut queue = vec![value];
    while let Some(cur) = queue.pop() {
        for next in reuse_pair(hints, reverse_reuse, cur)
            .into_iter()
            .flatten()
            .map(|v| v.index())
        {
            if pooled.get(next) == Some(&true)
                && homes[next].is_none()
                && try_place_weighted(
                    ctx,
                    weighted,
                    index,
                    (&mut *slots, &mut *homes),
                    (&mut *reserved, &mut *scratch_holders, &mut *query_ranges),
                    (next, Some(slot)),
                )
                .0
            {
                queue.push(next);
            }
        }
    }
}

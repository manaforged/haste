use crate::ir::{Block, Function, Op};
use crate::live::{Live, Range, overlaps};

use super::{Hints, Pool, SlotSet, crosses_call, crosses_ranges};

pub(super) struct Victim {
    pub(super) slot: usize,
}

pub(super) struct Demote {
    pub(super) slot: usize,
    pub(super) moves: Vec<(usize, usize)>,
    pub(super) evicts: Vec<(usize, usize)>,
}

fn push_span(list: &mut Vec<Range>, start: u32, end: u32) {
    if let Some(last) = list.last_mut()
        && last.end + 1 >= start
    {
        last.end = last.end.max(end);
    } else {
        list.push(Range { start, end });
    }
}

fn writes_last(func: &Function, data: &crate::ir::InstData) -> bool {
    matches!(
        data.op,
        crate::ir::Op::Binary(crate::ir::Binary::Umulhi | crate::ir::Binary::Smulhi, ..)
    ) && data
        .result
        .is_some_and(|res| func.ty(res) == crate::ir::Type::I64)
}

/// Computes instruction-level clobber spans for `wide_scratch` and `op_scratch` using `live` numbering.
pub(crate) fn scratch_clobbers<R>(
    func: &Function,
    (wide_scratch, op_scratch, scratch_mask): (
        &'static [R],
        &'static [R],
        fn(&Function, &Op) -> u8,
    ),
    (order, copied, elided): (&[Block], &[bool], &[bool]),
) -> Vec<(&'static [R], Vec<Range>)> {
    if wide_scratch.is_empty() && op_scratch.is_empty() {
        return Vec::new();
    }
    let mut wide = Vec::new();
    let mut ops: Vec<Vec<Range>> = vec![Vec::new(); op_scratch.len()];
    let mut record = |data: &crate::ir::InstData, at_use: u32, at_def: u32| {
        let touches_i128 = crate::alloc::touches_i128(func, data);
        if !wide_scratch.is_empty() && touches_i128 {
            push_span(&mut wide, at_use, at_def);
        }
        let mask = if touches_i128 {
            u8::MAX
        } else {
            scratch_mask(func, &data.op)
        };
        if mask != 0 {
            let end = if writes_last(func, data) {
                at_use
            } else {
                at_def
            };
            for (i, ranges) in ops.iter_mut().enumerate() {
                if (mask >> i) & 1 != 0 {
                    push_span(ranges, at_use, end);
                }
            }
        }
    };
    let mut at = 0_u32;
    for &block in order {
        for &inst in &func.blocks[block.index()].insts {
            let data = &func.insts[inst.index()];
            if data
                .result
                .is_some_and(|res| elided.get(res.index()) == Some(&true))
            {
                continue;
            }
            if copied.get(inst.index()) == Some(&true)
                && let Op::Jump(head, _) = &data.op
            {
                at += 2;
                for &h_inst in &func.blocks[head.index()].insts {
                    let h_data = &func.insts[h_inst.index()];
                    if h_data
                        .result
                        .is_some_and(|res| elided.get(res.index()) == Some(&true))
                    {
                        continue;
                    }
                    at += 1;
                    record(h_data, at, at + 1);
                    at += 1;
                }
                continue;
            }
            at += 1;
            record(data, at, at + 1);
            at += 1;
        }
        at += 2;
    }
    let mut out = Vec::new();
    if !wide.is_empty() {
        out.push((wide_scratch, wide));
    }
    for (i, ranges) in ops.into_iter().enumerate() {
        if !ranges.is_empty() {
            out.push((&op_scratch[i..i + 1], ranges));
        }
    }
    out
}

pub(super) fn slot_accepts<F: Copy + PartialEq>(
    pool: &Pool<F>,
    s: usize,
    ranges: &[Range],
    start: u32,
    kept_reg: Option<F>,
) -> bool {
    !(start == 0 && pool.arrival_lock[s] && kept_reg != Some(pool.regs[s]))
        && !pool.slot_clobbered(s, ranges)
}

pub(super) fn park_cost(calls: &[u32], weights: &[u64], ranges: &[Range]) -> u64 {
    ranges
        .iter()
        .map(|r| {
            let first = calls.partition_point(|at| *at <= r.start);
            let last = calls.partition_point(|at| *at <= r.end);
            (first..last)
                .map(|i| weights[i] * if calls[i] < r.end { 2 } else { 1 })
                .sum::<u64>()
        })
        .sum()
}

pub(super) fn leaf_park_cost(live: &Live, hints: &Hints, value: usize) -> Option<u64> {
    let ranges = live.ranges_of(value);
    if ranges.is_empty() || crosses_ranges(&live.unwinds, ranges) {
        return None;
    }
    if !crosses_call(live, value) {
        return Some(0);
    }
    let (call_weights, val_weights) = hints.park.zip(hints.weights)?;
    let cost = park_cost(&live.calls, call_weights, ranges);
    (cost < val_weights[value]).then_some(cost)
}

pub(super) fn prefer_park(live: &Live, hints: &Hints, value: usize) -> bool {
    leaf_park_cost(live, hints, value).is_some_and(|cost| cost < 2 * crate::alloc::BASE_WEIGHT)
}

pub(super) fn holder_leaf_cost<F: Copy + PartialEq>(
    pool: &Pool<F>,
    live: &Live,
    hints: &Hints,
    cost: &[u64],
    slot: usize,
    h: usize,
) -> u64 {
    let h_park = if crosses_call(live, h) {
        leaf_park_cost(live, hints, h).unwrap_or(0)
    } else {
        0
    };
    cost[h]
        .saturating_add(u64::from(pool.kept[h] == Some(pool.regs[slot])))
        .saturating_sub(h_park)
}

fn holder_spill_cost<F: Copy + PartialEq>(
    pool: &Pool<F>,
    live: &Live,
    hints: &Hints,
    cost: &[u64],
    slot: usize,
    h: usize,
) -> u64 {
    if !pool.is_saved[slot] {
        holder_leaf_cost(pool, live, hints, cost, slot, h)
    } else {
        cost[h].saturating_add(u64::from(pool.kept[h] == Some(pool.regs[slot])))
    }
}

fn slot_tier<F>(pool: &Pool<F>, slots: &SlotSet, needs_saved: bool, s: usize) -> u8 {
    if needs_saved {
        u8::from(!pool.is_saved[s])
    } else if !pool.is_saved[s] {
        0
    } else if !slots.is_empty(s) {
        1
    } else {
        2
    }
}

fn moves_clear(live: &Live, moves: &[(usize, usize)], slot: usize, ranges: &[Range]) -> bool {
    moves
        .iter()
        .all(|&(mh, ml)| ml != slot || !overlaps(live.ranges_of(mh), ranges))
}

fn is_partner<F: Copy + PartialEq>(
    pool: &Pool<F>,
    homes: &[Option<F>],
    back_partners: &[usize],
    s: usize,
) -> bool {
    back_partners
        .iter()
        .any(|&p| homes[p] == Some(pool.regs[s]))
}

pub(super) fn find_demote_free<F: Copy + PartialEq>(
    pool: &Pool<F>,
    slots: &SlotSet,
    live: &Live,
    hints: &Hints,
    reserved: &[F],
    (ranges, start, value): (&[Range], u32, usize),
) -> Option<Demote> {
    let mut best: Option<(bool, Demote)> = None;
    let mut overlap_h = Vec::new();
    for s in (0..pool.regs.len())
        .filter(|&s| pool.is_saved[s] && slot_accepts(pool, s, ranges, start, pool.kept[value]))
    {
        slots.collect_overlaps(s, ranges, &mut overlap_h);
        if overlap_h.is_empty() {
            continue;
        }
        let is_res = reserved.contains(&pool.regs[s]);
        if best.as_ref().is_some_and(|&(br, _)| is_res >= br) {
            continue;
        }
        let mut moves = Vec::new();
        let mut ok = true;
        for &h in &overlap_h {
            if pool.kept[h].is_some() || leaf_park_cost(live, hints, h) != Some(0) {
                ok = false;
                break;
            }
            let h_ranges = live.ranges_of(h);
            let Some(h_start) = h_ranges.first().map(|r| r.start) else {
                ok = false;
                break;
            };
            let pick = |avoid_res: bool| {
                (0..pool.regs.len()).find(|&l| {
                    !pool.is_saved[l]
                        && (!avoid_res || !reserved.contains(&pool.regs[l]))
                        && slots.free(pool, l, h_ranges, h_start, pool.kept[h])
                        && moves_clear(live, &moves, l, h_ranges)
                })
            };
            let Some(leaf_slot) = pick(true).or_else(|| pick(false)) else {
                ok = false;
                break;
            };
            moves.push((h, leaf_slot));
        }
        if !ok || moves.is_empty() {
            continue;
        }
        best = Some((
            is_res,
            Demote {
                slot: s,
                moves,
                evicts: Vec::new(),
            },
        ));
        if !is_res {
            break;
        }
    }
    best.map(|(_, d)| d)
}

struct RelocateCtx<'a, 'h, F> {
    func: &'a Function,
    pool: &'a Pool<F>,
    slots: &'a SlotSet,
    live: &'a Live,
    hints: &'a Hints<'h>,
    reserved: &'a [F],
    cost: &'a [u64],
}

impl<F: Copy + PartialEq> RelocateCtx<'_, '_, F> {
    fn holder_leaf_park(&self, slot: usize, h: usize, h_ranges: &[Range]) -> Option<u64> {
        if !crosses_call(self.live, h) {
            Some(0)
        } else if self.pool.is_saved[slot] {
            leaf_park_cost(self.live, self.hints, h)
        } else if !crosses_ranges(&self.live.unwinds, h_ranges) {
            Some(0)
        } else {
            None
        }
    }

    fn find_free_alt(
        &self,
        slot: usize,
        (h, h_ranges, h_start): (usize, &[Range], u32),
        (h_leaf_park, h_spill): (Option<u64>, u64),
        (total_cost, cand_gain): (u64, u64),
        moves: &[(usize, usize)],
    ) -> Option<(u64, usize)> {
        let h_wide = self.func.types[h].bytes() > 8;
        (0..self.pool.regs.len())
            .filter(|&alt| {
                alt != slot
                    && (if self.pool.is_saved[alt] {
                        !h_wide
                    } else {
                        h_leaf_park.is_some()
                    })
                    && self
                        .slots
                        .free(self.pool, alt, h_ranges, h_start, self.pool.kept[h])
                    && moves_clear(self.live, moves, alt, h_ranges)
            })
            .filter_map(|alt| {
                let delta = if self.pool.is_saved[alt] {
                    0
                } else {
                    h_leaf_park?
                };
                ((delta == 0 || delta < h_spill) && total_cost.saturating_add(delta) < cand_gain)
                    .then_some((
                        delta,
                        self.pool.is_saved[alt] && self.slots.is_empty(alt),
                        self.reserved.contains(&self.pool.regs[alt]),
                        alt,
                    ))
            })
            .min()
            .map(|(delta, _, _, alt)| (delta, alt))
    }

    fn find_leaf_demote(
        &self,
        slot: usize,
        (h, h_ranges, h_start): (usize, &[Range], u32),
        (hp, h_spill): (u64, u64),
        (total_cost, cand_gain): (u64, u64),
        (moves, evicts): (&[(usize, usize)], &[(usize, usize)]),
        overlap_eh: &mut Vec<usize>,
    ) -> Option<(u64, usize, Vec<(usize, usize)>)> {
        let mut best_leaf: Option<(u64, bool, usize, Vec<(usize, usize)>)> = None;
        for l in (0..self.pool.regs.len()).filter(|&l| {
            l != slot
                && !self.pool.is_saved[l]
                && slot_accepts(self.pool, l, h_ranges, h_start, self.pool.kept[h])
                && moves_clear(self.live, moves, l, h_ranges)
        }) {
            self.slots.collect_overlaps(l, h_ranges, overlap_eh);
            let mut add_evict = 0_u64;
            let mut slot_evicts = Vec::with_capacity(overlap_eh.len());
            let mut over = false;
            for &eh in overlap_eh.iter() {
                if !evicts.iter().any(|&(_, prev)| prev == eh) {
                    add_evict = add_evict.saturating_add(holder_leaf_cost(
                        self.pool, self.live, self.hints, self.cost, l, eh,
                    ));
                    if add_evict.saturating_add(hp) >= h_spill
                        || total_cost.saturating_add(hp).saturating_add(add_evict) >= cand_gain
                    {
                        over = true;
                        break;
                    }
                }
                slot_evicts.push((l, eh));
            }
            if over {
                continue;
            }
            let l_res = self.reserved.contains(&self.pool.regs[l]);
            if best_leaf
                .as_ref()
                .is_none_or(|&(bc, br, _, _)| (add_evict, l_res) < (bc, br))
            {
                best_leaf = Some((add_evict, l_res, l, slot_evicts));
            }
        }
        best_leaf
            .map(|(add_evict, _, l, slot_evicts)| (hp.saturating_add(add_evict), l, slot_evicts))
    }

    fn plan_slot(
        &self,
        slot: usize,
        overlap_h: &[usize],
        (added_park, cand_gain): (u64, u64),
        best_penalty: Option<u64>,
        overlap_eh: &mut Vec<usize>,
    ) -> Option<(u64, Demote)> {
        let mut moves: Vec<(usize, usize)> = Vec::new();
        let mut evicts: Vec<(usize, usize)> = Vec::new();
        let mut total_cost = 0_u64;
        for &h in overlap_h {
            let h_spill = holder_spill_cost(self.pool, self.live, self.hints, self.cost, slot, h);
            let h_ranges = self.live.ranges_of(h);
            let h_start = h_ranges.first()?.start;
            let h_leaf_park = self.holder_leaf_park(slot, h, h_ranges);
            if let Some((delta, alt)) = self.find_free_alt(
                slot,
                (h, h_ranges, h_start),
                (h_leaf_park, h_spill),
                (total_cost, cand_gain),
                &moves,
            ) {
                total_cost = total_cost.saturating_add(delta);
                moves.push((h, alt));
                continue;
            }
            if let Some((move_cost, leaf_slot, slot_evicts)) = h_leaf_park.and_then(|hp| {
                self.find_leaf_demote(
                    slot,
                    (h, h_ranges, h_start),
                    (hp, h_spill),
                    (total_cost, cand_gain),
                    (&moves, &evicts),
                    overlap_eh,
                )
            }) {
                total_cost = total_cost.saturating_add(move_cost);
                moves.push((h, leaf_slot));
                for item in slot_evicts {
                    if !evicts.contains(&item) {
                        evicts.push(item);
                    }
                }
                continue;
            }
            total_cost = total_cost.saturating_add(h_spill);
            if total_cost >= cand_gain
                || best_penalty.is_some_and(|bp| total_cost.saturating_add(added_park) > bp)
            {
                return None;
            }
            if !evicts.iter().any(|&(_, prev)| prev == h) {
                evicts.push((slot, h));
            }
        }
        (total_cost < cand_gain).then_some((
            total_cost,
            Demote {
                slot,
                moves,
                evicts,
            },
        ))
    }
}

/// Finds the lowest-cost slot replacement by relocating overlapping holders to free/cheaper slots or evicting them.
pub(super) fn find_relocate<F: Copy + PartialEq>(
    func: &Function,
    pool: &Pool<F>,
    slots: &SlotSet,
    homes: &[Option<F>],
    live: &Live,
    hints: &Hints,
    back_partners: &[usize],
    reserved: &[F],
    cost: &[u64],
    (ranges, start, value): (&[Range], u32, usize),
    (needs_saved, can_park, wide_val): (bool, bool, bool),
) -> Option<Demote> {
    let ctx = RelocateCtx {
        func,
        pool,
        slots,
        live,
        hints,
        reserved,
        cost,
    };
    let suits = |s: usize| {
        if needs_saved {
            (pool.is_saved[s] && !wide_val) || (can_park && !pool.is_saved[s])
        } else {
            !wide_val || !pool.is_saved[s]
        }
    };
    let val_park = if needs_saved && can_park {
        leaf_park_cost(live, hints, value)
    } else {
        None
    };
    let mut best: Option<((u64, u8, bool, bool, usize, usize), Demote)> = None;
    let mut overlap_h = Vec::new();
    let mut overlap_eh = Vec::new();
    for s in (0..pool.regs.len())
        .filter(|&s| suits(s) && slot_accepts(pool, s, ranges, start, pool.kept[value]))
    {
        let added_park = if !pool.is_saved[s] && needs_saved {
            let Some(vp) = val_park else { continue };
            vp
        } else {
            0
        };
        let cand_gain = cost[value]
            .saturating_add(u64::from(pool.kept[value] == Some(pool.regs[s])))
            .saturating_sub(added_park);
        if cand_gain == 0 || best.as_ref().is_some_and(|b| added_park > (b.0).0) {
            continue;
        }
        slots.collect_overlaps(s, ranges, &mut overlap_h);
        if overlap_h.is_empty() {
            continue;
        }
        let best_penalty = best.as_ref().map(|b| (b.0).0);
        let Some((total_cost, demote)) = ctx.plan_slot(
            s,
            &overlap_h,
            (added_park, cand_gain),
            best_penalty,
            &mut overlap_eh,
        ) else {
            continue;
        };
        let penalty = total_cost.saturating_add(added_park);
        let key = (
            penalty,
            slot_tier(pool, slots, needs_saved, s),
            reserved.contains(&pool.regs[s]),
            !is_partner(pool, homes, back_partners, s),
            demote.evicts.len(),
            s,
        );
        if best.as_ref().is_none_or(|b| key < b.0) {
            best = Some((key, demote));
            if key.0 == 0 && key.1 == 0 && !key.2 && key.4 == 0 {
                break;
            }
        }
    }
    best.map(|(_, d)| d)
}

pub(super) fn apply_demote<F: Copy>(
    pool: &Pool<F>,
    live: &Live,
    mut holders: Option<&mut [Vec<usize>]>,
    slots: &mut SlotSet,
    homes: &mut [Option<F>],
    demote: &Demote,
) -> Vec<usize> {
    let mut dropped = Vec::new();
    for &(slot, eh) in &demote.evicts {
        if let Some(list) = holders.as_deref_mut() {
            list[slot].retain(|&x| x != eh);
        }
        slots.remove(slot, live.ranges_of(eh), eh);
        if homes[eh].take().is_some() {
            dropped.push(eh);
        }
    }
    for &(other, dst_slot) in &demote.moves {
        let other_ranges = live.ranges_of(other);
        if let Some(list) = holders.as_deref_mut() {
            list[demote.slot].retain(|&h| h != other);
            list[dst_slot].push(other);
        }
        slots.remove(demote.slot, other_ranges, other);
        slots.insert(dst_slot, other_ranges, other);
        homes[other] = Some(pool.regs[dst_slot]);
    }
    dropped
}

pub(super) fn evict_slot<F: Copy>(
    holders: Option<&mut Vec<usize>>,
    slots: &mut SlotSet,
    slot: usize,
    homes: &mut [Option<F>],
    live: &Live,
    ranges: &[Range],
) -> Vec<usize> {
    let mut dropped = Vec::new();
    slots.collect_overlaps(slot, ranges, &mut dropped);
    if let Some(list) = holders {
        list.retain(|other| !dropped.contains(other));
    }
    for &other in &dropped {
        slots.remove(slot, live.ranges_of(other), other);
        homes[other] = None;
    }
    dropped
}

pub(super) fn victim_slot_unweighted<F: Copy + PartialEq>(
    pool: &Pool<F>,
    slots: &SlotSet,
    homes: &[Option<F>],
    live: &Live,
    back_partners: &[usize],
    (start, end, value): (u32, u32, usize),
    suits: impl Fn(usize) -> bool,
) -> Option<Victim> {
    let ranges = live.ranges_of(value);
    let mut overlap_h = Vec::new();
    (0..pool.regs.len())
        .filter(|&s| suits(s) && slot_accepts(pool, s, ranges, start, pool.kept[value]))
        .filter_map(|s| {
            slots.collect_overlaps(s, ranges, &mut overlap_h);
            if overlap_h.len() != 1 {
                return None;
            }
            let single = overlap_h[0];
            if pool.kept[single].is_some() {
                return None;
            }
            live.last_end(single)
                .map(|u| (u, is_partner(pool, homes, back_partners, s), s))
        })
        .max_by_key(|&(u, partner, _)| (u, partner))
        .filter(|&(u, _, _)| u > end)
        .map(|(_, _, slot)| Victim { slot })
}

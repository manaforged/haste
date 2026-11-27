use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::ir::{Function, Op, Value};
use crate::live::{Live, Range, overlaps};

mod choose;
mod place;
mod recolor;
use choose::{choose_free, place_reuse_chain};
pub(crate) use place::scratch_clobbers;
use place::{
    apply_demote, evict_slot, find_demote_free, find_relocate, leaf_park_cost, prefer_park,
    slot_accepts, victim_slot_unweighted,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ParkReg<R> {
    Int(R),
    Float(u8),
}

pub(crate) struct Class<'a, F> {
    pub saved: &'a [F],
    pub leaf: Vec<F>,
    pub arrivals: Vec<Vec<F>>,
    pub passed: Vec<Option<(F, bool)>>,
    pub clobbers: Vec<(&'a [F], Vec<Range>)>,
}

pub(crate) struct Hints<'a> {
    pub reuse: &'a [Option<Value>],
    pub weights: Option<&'a [u64]>,
    pub partners: Option<&'a [Vec<Value>]>,
    pub back_args: &'a [(usize, usize)],
    pub edge_args: &'a [(usize, usize)],
    pub park: Option<&'a [u64]>,
    pub two_address: bool,
    pub copies: &'a [(usize, usize)],
}

pub(super) struct Pool<F> {
    pub(super) regs: Vec<F>,
    pub(super) is_saved: Vec<bool>,
    pub(super) kept: Vec<Option<F>>,
    pub(super) arrival_lock: Vec<bool>,
    pub(super) passed: Vec<Option<(F, bool)>>,
    clobbers: Vec<Vec<Range>>,
    slot_clobber: Vec<u8>,
}

impl<F: PartialEq> Pool<F> {
    /// The pool slot that holds `reg`, if any.
    pub(super) fn slot_of(&self, reg: F) -> Option<usize> {
        self.regs.iter().position(|r| *r == reg)
    }
}

impl<F> Pool<F> {
    pub(super) fn slot_clobbered(&self, s: usize, ranges: &[Range]) -> bool {
        let mut mask = self.slot_clobber[s];
        while mask != 0 {
            let idx = mask.trailing_zeros() as usize;
            if overlaps(&self.clobbers[idx], ranges) {
                return true;
            }
            mask &= mask - 1;
        }
        false
    }
}

pub(crate) fn crosses_span(calls: &[u32], (start, end): (u32, u32)) -> bool {
    let next = calls.partition_point(|at| *at <= start);
    calls.get(next).is_some_and(|at| *at <= end)
}

pub(crate) fn crosses_ranges(calls: &[u32], ranges: &[Range]) -> bool {
    ranges.iter().any(|r| crosses_span(calls, (r.start, r.end)))
}

pub(super) fn crosses_call(live: &Live, value: usize) -> bool {
    crosses_ranges(&live.calls, live.ranges_of(value))
}

pub(crate) fn parks(live: &Live, hints: &Hints, value: usize) -> bool {
    hints.park.is_some()
        && hints
            .weights
            .is_some_and(|w| leaf_park_cost(live, hints, value).is_some_and(|c| c < w[value]))
}

fn pool<F: Copy + PartialEq>(
    func: &Function,
    class: &Class<F>,
    live: &Live,
    pooled: &[bool],
    hints: &Hints,
) -> Pool<F> {
    let regs: Vec<F> = class.saved.iter().chain(&class.leaf).copied().collect();
    let is_saved: Vec<bool> = regs.iter().map(|r| class.saved.contains(r)).collect();
    let locked_arrivals: Vec<F> = class
        .arrivals
        .iter()
        .flatten()
        .copied()
        .filter(|reg| class.leaf.contains(reg))
        .collect();
    let arrival_lock: Vec<bool> = regs.iter().map(|r| locked_arrivals.contains(r)).collect();
    let clobbers: Vec<Vec<Range>> = class
        .clobbers
        .iter()
        .map(|(_, ranges)| ranges.clone())
        .collect();
    let slot_clobber: Vec<u8> = regs
        .iter()
        .enumerate()
        .map(|(s, r)| {
            let mut mask = 0_u8;
            if !is_saved[s] {
                for (i, (group, _)) in class.clobbers.iter().enumerate() {
                    if i < 8 && group.contains(r) {
                        mask |= 1 << i;
                    }
                }
            }
            mask
        })
        .collect();
    let mut out = Pool {
        regs,
        is_saved,
        kept: vec![None; func.types.len()],
        arrival_lock,
        passed: vec![None; func.types.len()],
        clobbers,
        slot_clobber,
    };
    let entry = func
        .entry
        .map_or(&[][..], |entry| &func.blocks[entry.index()].params[..]);
    for (value, arrival) in entry.iter().zip(&class.arrivals) {
        if let [reg] = arrival[..]
            && let Some(slot) = out.slot_of(reg)
            && !out.is_saved[slot]
            && pooled[value.index()]
        {
            let ranges = live.ranges_of(value.index());
            if !ranges.is_empty()
                && (!crosses_call(live, value.index()) || prefer_park(live, hints, value.index()))
                && !out.slot_clobbered(slot, ranges)
            {
                out.kept[value.index()] = Some(reg);
            }
        }
    }
    for (v, entry) in class.passed.iter().enumerate() {
        out.passed[v] = entry.filter(|(reg, _)| {
            out.slot_of(*reg).is_some_and(|slot| {
                !out.is_saved[slot] && !out.slot_clobbered(slot, live.ranges_of(v))
            })
        });
    }
    out
}

pub(super) fn slot_free<F: Copy + PartialEq>(
    pool: &Pool<F>,
    s: usize,
    holders: &[usize],
    live: &Live,
    ranges: &[Range],
    start: u32,
    kept_reg: Option<F>,
) -> bool {
    slot_accepts(pool, s, ranges, start, kept_reg)
        && holders
            .iter()
            .all(|&other| !overlaps(live.ranges_of(other), ranges))
}

pub(super) struct SlotSet {
    slots: Vec<Vec<(u32, u32, usize)>>,
}

impl SlotSet {
    fn new(count: usize) -> Self {
        Self {
            slots: vec![Vec::new(); count],
        }
    }

    pub(super) fn is_empty(&self, s: usize) -> bool {
        self.slots[s].is_empty()
    }

    fn first(&self, s: usize) -> Option<usize> {
        self.slots[s].first().map(|e| e.2)
    }

    pub(super) fn has_overlap(&self, s: usize, ranges: &[Range]) -> bool {
        let slot = &self.slots[s];
        ranges.iter().any(|r| {
            let idx = slot.partition_point(|&(_, end, _)| end < r.start);
            slot.get(idx).is_some_and(|&(start, _, _)| start <= r.end)
        })
    }

    fn first_overlap(&self, s: usize, ranges: &[Range]) -> Option<usize> {
        let slot = &self.slots[s];
        for r in ranges {
            let idx = slot.partition_point(|&(_, end, _)| end < r.start);
            if let Some(&(start, _, val)) = slot.get(idx)
                && start <= r.end
            {
                return Some(val);
            }
        }
        None
    }

    pub(super) fn collect_overlaps(&self, s: usize, ranges: &[Range], out: &mut Vec<usize>) {
        out.clear();
        let slot = &self.slots[s];
        for r in ranges {
            let mut idx = slot.partition_point(|&(_, end, _)| end < r.start);
            while let Some(&(start, _, val)) = slot.get(idx) {
                if start > r.end {
                    break;
                }
                if !out.contains(&val) {
                    out.push(val);
                }
                idx += 1;
            }
        }
    }

    pub(super) fn free<F: Copy + PartialEq>(
        &self,
        pool: &Pool<F>,
        s: usize,
        ranges: &[Range],
        start: u32,
        kept_reg: Option<F>,
    ) -> bool {
        slot_accepts(pool, s, ranges, start, kept_reg) && !self.has_overlap(s, ranges)
    }

    fn insert(&mut self, s: usize, ranges: &[Range], val: usize) {
        let slot = &mut self.slots[s];
        for r in ranges {
            let pos = slot.partition_point(|&(start, _, _)| start < r.start);
            slot.insert(pos, (r.start, r.end, val));
        }
    }

    fn remove(&mut self, s: usize, ranges: &[Range], val: usize) {
        let slot = &mut self.slots[s];
        for r in ranges {
            let pos = slot.partition_point(|&(start, _, _)| start < r.start);
            if slot.get(pos) == Some(&(r.start, r.end, val)) {
                slot.remove(pos);
            }
        }
    }
}

fn back_partner_map(
    count: usize,
    pooled: &[bool],
    back_args: &[(usize, usize)],
) -> Vec<Vec<usize>> {
    let mut map = vec![Vec::new(); count];
    for &(param, arg) in back_args {
        if pooled[param] && pooled[arg] {
            if !map[param].contains(&arg) {
                map[param].push(arg);
            }
            if !map[arg].contains(&param) {
                map[arg].push(param);
            }
        }
    }
    map
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Priority {
    density: u128,
    weight: u64,
    non_remat: bool,
    start: Reverse<u32>,
    tie: Reverse<usize>,
    value: usize,
}

fn value_priority(
    weights: &[u64],
    live: &Live,
    remat: &[bool],
    start: u32,
    value: usize,
) -> Priority {
    let weight = weights[value];
    let len = live.total_len(value).max(4);
    let density = (u128::from(weight) << 24) / u128::from(len);
    Priority {
        density,
        weight,
        non_remat: !remat[value],
        start: Reverse(start),
        tie: Reverse(value),
        value,
    }
}

const MAX_REQUEUE: u8 = 8;
const RES_CHUNK: usize = 32;

#[derive(Default)]
struct BackIndex {
    entries: Vec<(u32, u32, usize, usize, bool)>,
    chunk_max: Vec<u32>,
}

impl BackIndex {
    fn new(back_args: &[(usize, usize)], live: &Live) -> Self {
        if back_args.is_empty() {
            return Self::default();
        }
        let mut pairs = Vec::with_capacity(back_args.len() * 2);
        for &(param, arg) in back_args {
            if param != arg {
                pairs.push((arg, param, false));
                pairs.push((param, arg, true));
            }
        }
        pairs.sort_unstable();
        pairs.dedup();
        let mut entries: Vec<_> = pairs
            .into_iter()
            .filter_map(|(t, p, ip)| live.bounds(t).map(|(s, e)| (s, e, t, p, ip)))
            .collect();
        entries.sort_unstable_by_key(|e| e.0);
        let chunk_max = entries
            .chunks(RES_CHUNK)
            .map(|c| c.iter().map(|e| e.1).max().unwrap_or(0))
            .collect();
        Self { entries, chunk_max }
    }

    fn fill<F: Copy + PartialEq>(
        &self,
        homes: &[Option<F>],
        live: &Live,
        (value, start, end, ranges): (usize, u32, u32, &[Range]),
        check_order: bool,
        limit: usize,
        reserved: &mut Vec<F>,
    ) {
        reserved.clear();
        if self.entries.is_empty() {
            return;
        }
        let bound = self.entries.partition_point(|e| e.0 <= end);
        for (c, chunk) in self.entries[..bound].chunks(RES_CHUNK).enumerate() {
            if self.chunk_max[c] < start {
                continue;
            }
            for &(ts, te, target, partner, is_param) in chunk {
                if te >= start
                    && ts <= end
                    && target != value
                    && (!check_order || (!is_param && ts >= start))
                    && let Some(reg) = homes[partner]
                    && homes[target].is_none()
                    && !reserved.contains(&reg)
                    && overlaps(live.ranges_of(target), ranges)
                {
                    reserved.push(reg);
                    if reserved.len() == limit {
                        return;
                    }
                }
            }
        }
    }
}

fn fill_scratch_holders<'a, F>(
    slots: &SlotSet,
    homes: &[Option<F>],
    live: &'a Live,
    hints: &Hints,
    back_partners: &[usize],
    reverse_reuse: &[Option<Value>],
    value: usize,
    query_ranges: &mut Vec<(usize, &'a [Range])>,
    out: &mut [Vec<usize>],
) {
    query_ranges.clear();
    query_ranges.push((value, live.ranges_of(value)));
    let mut add_partner = |p: usize| {
        if p < homes.len() && homes[p].is_none() && !query_ranges.iter().any(|&(q, _)| q == p) {
            let r = live.ranges_of(p);
            if !r.is_empty() {
                query_ranges.push((p, r));
            }
        }
    };
    for &p in back_partners {
        add_partner(p);
    }
    if let Some(partners) = hints.partners {
        for p in &partners[value] {
            add_partner(p.index());
        }
    }
    if let Some(r) = hints.reuse[value] {
        add_partner(r.index());
    }
    if let Some(r) = reverse_reuse.get(value).copied().flatten() {
        add_partner(r.index());
    }
    for (s, list) in out.iter_mut().enumerate() {
        list.clear();
        for &(_, ranges) in query_ranges.iter() {
            if let Some(hit) = slots.first_overlap(s, ranges)
                && !list.contains(&hit)
            {
                list.push(hit);
            }
        }
        if list.is_empty()
            && let Some(first) = slots.first(s)
        {
            list.push(first);
        }
    }
}

fn try_place_weighted<'a, F: Copy + PartialEq>(
    (func, live, pool): (&Function, &'a Live, &Pool<F>),
    (hints, weights): (&Hints, &[u64]),
    (back_partners, reverse_reuse, back_index): (&[Vec<usize>], &[Option<Value>], &BackIndex),
    (slots, homes): (&mut SlotSet, &mut [Option<F>]),
    (reserved, scratch_holders, query_ranges): (
        &mut Vec<F>,
        &mut Vec<Vec<usize>>,
        &mut Vec<(usize, &'a [Range])>,
    ),
    (value, want): (usize, Option<usize>),
) -> (bool, Vec<usize>) {
    let Some((start, end)) = live.bounds(value) else {
        return (false, Vec::new());
    };
    let ranges = live.ranges_of(value);
    let wide_val = func.types[value].bytes() > 8;
    let needs_saved = crosses_call(live, value);
    let can_park = needs_saved && parks(live, hints, value);
    if needs_saved && wide_val && !can_park {
        return (false, Vec::new());
    }
    let has_free_saved = needs_saved
        && !wide_val
        && (0..pool.regs.len())
            .any(|s| pool.is_saved[s] && slots.free(pool, s, ranges, start, pool.kept[value]));
    back_index.fill(
        homes,
        live,
        (value, start, end, ranges),
        false,
        pool.regs.len(),
        reserved,
    );
    fill_scratch_holders(
        slots,
        homes,
        live,
        hints,
        &back_partners[value],
        reverse_reuse,
        value,
        query_ranges,
        scratch_holders,
    );
    let mut chosen = choose_free(
        pool,
        scratch_holders,
        homes,
        live,
        hints,
        &back_partners[value],
        reverse_reuse,
        reserved,
        |s| !slots.is_empty(s),
        (
            value,
            start,
            end,
            wide_val,
            needs_saved,
            can_park,
            has_free_saved,
        ),
    );
    if want.is_some_and(|s| chosen != Some(s)) {
        return (false, Vec::new());
    }
    let mut dropped = Vec::new();
    if chosen.is_none()
        && let Some(demote) = find_relocate(
            func,
            pool,
            slots,
            homes,
            live,
            hints,
            &back_partners[value],
            reserved,
            weights,
            (ranges, start, value),
            (needs_saved, can_park, wide_val),
        )
    {
        chosen = Some(demote.slot);
        dropped = apply_demote(pool, live, None, slots, homes, &demote);
    }
    let Some(slot) = chosen else {
        return (false, dropped);
    };
    homes[value] = Some(pool.regs[slot]);
    slots.insert(slot, ranges, value);
    (true, dropped)
}

fn allocate_weighted<F: Copy + PartialEq>(
    func: &Function,
    live: &Live,
    pool: &Pool<F>,
    pooled: &[bool],
    hints: &Hints,
    weights: &[u64],
) -> (Vec<Option<F>>, Vec<F>) {
    let mut homes = vec![None; pooled.len()];
    let mut slots = SlotSet::new(pool.regs.len());
    let mut remat = vec![false; pooled.len()];
    for data in &func.insts {
        if matches!(data.op, Op::Const(_) | Op::SlotAddr(_) | Op::Symbol(_))
            && let Some(res) = data.result
            && res.index() < remat.len()
        {
            remat[res.index()] = true;
        }
    }
    let mut reverse_reuse = vec![None; pooled.len()];
    for (dst, &src) in hints.reuse.iter().enumerate() {
        if let Some(src) = src
            && reverse_reuse[src.index()].is_none()
        {
            reverse_reuse[src.index()] = Some(Value(dst as u32));
        }
    }
    let mut heap = BinaryHeap::new();
    for (v, &is_pooled) in pooled.iter().enumerate() {
        if is_pooled && let Some((start, _)) = live.bounds(v) {
            heap.push(value_priority(weights, live, &remat, start, v));
        }
    }
    let back_partners = back_partner_map(pooled.len(), pooled, hints.back_args);
    let back_index = BackIndex::new(hints.back_args, live);
    let mut reserved: Vec<F> = Vec::new();
    let mut scratch_holders: Vec<Vec<usize>> = vec![Vec::new(); pool.regs.len()];
    let mut query_ranges: Vec<(usize, &[Range])> = Vec::new();
    let mut requeues = vec![0_u8; pooled.len()];
    let mut spilled = Vec::new();
    let mut in_spilled = vec![false; pooled.len()];
    while let Some(item) = heap.pop() {
        let value = item.value;
        if homes[value].is_some() {
            continue;
        }
        let (placed, dropped) = try_place_weighted(
            (func, live, pool),
            (hints, weights),
            (&back_partners, &reverse_reuse, &back_index),
            (&mut slots, &mut homes),
            (&mut reserved, &mut scratch_holders, &mut query_ranges),
            (value, None),
        );
        for eh in dropped {
            if requeues[eh] < MAX_REQUEUE
                && let Some((eh_start, _)) = live.bounds(eh)
            {
                requeues[eh] += 1;
                heap.push(value_priority(weights, live, &remat, eh_start, eh));
            } else if !std::mem::replace(&mut in_spilled[eh], true) {
                spilled.push(eh);
            }
        }
        if !placed && !std::mem::replace(&mut in_spilled[value], true) {
            spilled.push(value);
        }
        if placed && hints.two_address {
            place_reuse_chain(
                (func, live, pool),
                (hints, weights),
                (&back_partners, &reverse_reuse, &back_index),
                (&mut slots, &mut homes),
                (&mut reserved, &mut scratch_holders, &mut query_ranges),
                (pooled, value),
            );
        }
    }
    while !spilled.is_empty() {
        spilled.sort_unstable_by_key(|&v| (Reverse(weights[v]), v));
        let round = std::mem::take(&mut spilled);
        for value in round {
            in_spilled[value] = false;
            if homes[value].is_some() {
                continue;
            }
            let (_, dropped) = try_place_weighted(
                (func, live, pool),
                (hints, weights),
                (&back_partners, &reverse_reuse, &back_index),
                (&mut slots, &mut homes),
                (&mut reserved, &mut scratch_holders, &mut query_ranges),
                (value, None),
            );
            for eh in dropped {
                if requeues[eh] < MAX_REQUEUE + 2 && !std::mem::replace(&mut in_spilled[eh], true) {
                    requeues[eh] += 1;
                    spilled.push(eh);
                }
            }
        }
    }
    let join = recolor::Join {
        func,
        live,
        pool,
        hints,
        back_partners: &back_partners,
        reverse_reuse: &reverse_reuse,
    };
    join.join_args(hints.back_args, &mut slots, &mut homes);
    join.join_args(hints.edge_args, &mut slots, &mut homes);
    join.share_copies(&mut slots, &mut homes);
    let used = (0..pool.regs.len())
        .filter(|&s| pool.is_saved[s] && !slots.is_empty(s))
        .map(|s| pool.regs[s])
        .collect();
    (homes, used)
}

pub(crate) fn allocate<F: Copy + PartialEq>(
    func: &Function,
    live: &Live,
    class: &Class<F>,
    pooled: &[bool],
    hints: &Hints,
) -> (Vec<Option<F>>, Vec<F>) {
    if !pooled.iter().any(|&p| p) {
        return (vec![None; pooled.len()], Vec::new());
    }
    let pool = pool(func, class, live, pooled, hints);
    if let Some(weights) = hints.weights {
        return allocate_weighted(func, live, &pool, pooled, hints, weights);
    }
    let mut homes = vec![None; pooled.len()];
    let mut holders: Vec<Vec<usize>> = vec![Vec::new(); pool.regs.len()];
    let mut slots = SlotSet::new(pool.regs.len());
    for (v, reg) in pool.kept.iter().enumerate() {
        if let Some(reg) = *reg
            && let Some(slot) = pool.slot_of(reg)
        {
            homes[v] = Some(reg);
            holders[slot].push(v);
            slots.insert(slot, live.ranges_of(v), v);
        }
    }
    let mut values: Vec<(u32, u32, usize)> = (0..pooled.len())
        .filter(|&v| pooled[v] && pool.kept[v].is_none())
        .filter_map(|v| live.bounds(v).map(|(start, end)| (start, end, v)))
        .collect();
    values.sort_unstable();
    let back_partners = back_partner_map(pooled.len(), pooled, hints.back_args);
    let back_index = BackIndex::new(hints.back_args, live);
    let mut used: Vec<F> = Vec::new();
    let mut reserved: Vec<F> = Vec::new();
    for (start, end, value) in values {
        for list in &mut holders {
            list.retain(|&other| live.last_end(other).is_some_and(|u| u >= start));
        }
        let ranges = live.ranges_of(value);
        let wide_val = func.types[value].bytes() > 8;
        let needs_saved = crosses_call(live, value);
        let can_park = needs_saved && parks(live, hints, value);
        if needs_saved && wide_val && !can_park {
            continue;
        }
        let suits_direct = |s: usize| {
            if needs_saved {
                pool.is_saved[s] && !wide_val
            } else {
                !wide_val || !pool.is_saved[s]
            }
        };
        let suits_any = |s: usize| suits_direct(s) || (can_park && !pool.is_saved[s]);
        let has_free_saved = needs_saved
            && !wide_val
            && (0..pool.regs.len()).any(|s| {
                pool.is_saved[s] && slot_free(&pool, s, &holders[s], live, ranges, start, None)
            });
        back_index.fill(
            &homes,
            live,
            (value, start, end, ranges),
            true,
            pool.regs.len(),
            &mut reserved,
        );
        let key = (
            value,
            start,
            end,
            wide_val,
            needs_saved,
            can_park,
            has_free_saved,
        );
        let mut chosen = choose_free(
            &pool,
            &holders,
            &homes,
            live,
            hints,
            &back_partners[value],
            &[],
            &reserved,
            |s| used.contains(&pool.regs[s]),
            key,
        );
        if chosen.is_none() {
            if let Some(demote) = (!wide_val)
                .then(|| {
                    find_demote_free(
                        &pool,
                        &slots,
                        live,
                        hints,
                        &reserved,
                        (ranges, start, value),
                    )
                })
                .flatten()
            {
                apply_demote(
                    &pool,
                    live,
                    Some(&mut holders),
                    &mut slots,
                    &mut homes,
                    &demote,
                );
                chosen = Some(demote.slot);
            } else if let Some(v) = victim_slot_unweighted(
                &pool,
                &slots,
                &homes,
                live,
                &back_partners[value],
                (start, end, value),
                suits_any,
            ) {
                evict_slot(
                    Some(&mut holders[v.slot]),
                    &mut slots,
                    v.slot,
                    &mut homes,
                    live,
                    ranges,
                );
                chosen = Some(v.slot);
            }
        }
        let Some(slot) = chosen else { continue };
        let reg = pool.regs[slot];
        homes[value] = Some(reg);
        holders[slot].push(value);
        slots.insert(slot, ranges, value);
        if pool.is_saved[slot] && !used.contains(&reg) {
            used.push(reg);
        }
    }
    (homes, used)
}

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::ir::{Function, Op, Value};
use crate::live::{Live, Range, overlaps};

mod choose;
use choose::choose_free;

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
}

pub(crate) struct Hints<'a> {
    pub reuse: &'a [Option<Value>],
    pub weights: Option<&'a [u64]>,
    pub partners: Option<&'a [Vec<Value>]>,
    pub back_args: &'a [(usize, usize)],
    pub park: Option<&'a [u64]>,
    pub two_address: bool,
}

struct Pool<F> {
    regs: Vec<F>,
    is_saved: Vec<bool>,
    kept: Vec<Option<F>>,
    arrival_lock: Vec<bool>,
    passed: Vec<Option<(F, bool)>>,
}

pub(crate) fn crosses_span(calls: &[u32], (start, end): (u32, u32)) -> bool {
    let next = calls.partition_point(|at| *at <= start);
    calls.get(next).is_some_and(|at| *at <= end)
}

pub(crate) fn crosses_ranges(calls: &[u32], ranges: &[Range]) -> bool {
    ranges.iter().any(|r| crosses_span(calls, (r.start, r.end)))
}

fn park_cost(calls: &[u32], weights: &[u64], ranges: &[Range]) -> u64 {
    ranges
        .iter()
        .map(|r| {
            let first = calls.partition_point(|at| *at <= r.start);
            let last = calls.partition_point(|at| *at <= r.end);
            (first..last).map(|i| weights[i] * if calls[i] < r.end { 2 } else { 1 }).sum::<u64>()
        })
        .sum()
}

pub(crate) fn parks(live: &Live, hints: &Hints, value: usize) -> bool {
    hints.park.is_some() && hints.weights.is_some_and(|w| leaf_park_cost(live, hints, value).is_some_and(|c| c < w[value]))
}

fn pool<F: Copy + PartialEq>(func: &Function, class: &Class<F>, live: &Live, pooled: &[bool]) -> Pool<F> {
    let mut kept = vec![None; func.types.len()];
    let entry = func.entry.map_or(&[][..], |entry| &func.blocks[entry.index()].params[..]);
    for (value, arrival) in entry.iter().zip(&class.arrivals) {
        if let [reg] = arrival[..]
            && class.leaf.contains(&reg)
            && pooled[value.index()]
            && !live.trapped[value.index()]
        {
            let ranges = live.ranges_of(value.index());
            if !ranges.is_empty() && !crosses_ranges(&live.calls, ranges) {
                kept[value.index()] = Some(reg);
            }
        }
    }
    let regs: Vec<F> = class.saved.iter().chain(&class.leaf).copied().collect();
    let is_saved: Vec<bool> = regs.iter().map(|r| class.saved.contains(r)).collect();
    let locked_arrivals: Vec<F> = class.arrivals.iter().flatten().copied().filter(|reg| class.leaf.contains(reg)).collect();
    let arrival_lock: Vec<bool> = regs.iter().map(|r| locked_arrivals.contains(r)).collect();
    let passed = class.passed.iter().map(|entry| entry.filter(|(reg, _)| class.leaf.contains(reg))).collect();
    Pool { regs, is_saved, kept, arrival_lock, passed }
}

fn slot_free<F: Copy + PartialEq>(holders: &[usize], live: &Live, ranges: &[Range], locked_entry: bool, start: u32, slot_reg: F, kept_reg: Option<F>) -> bool {
    !(start == 0 && locked_entry && kept_reg != Some(slot_reg)) && holders.iter().all(|&other| !overlaps(live.ranges_of(other), ranges))
}

fn back_partner_map(count: usize, pooled: &[bool], back_args: &[(usize, usize)]) -> Vec<Vec<usize>> {
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

fn leaf_park_cost(live: &Live, hints: &Hints, value: usize) -> Option<u64> {
    let ranges = live.ranges_of(value);
    if ranges.is_empty() || live.trapped[value] || crosses_ranges(&live.unwinds, ranges) {
        return None;
    }
    if !crosses_ranges(&live.calls, ranges) {
        return Some(0);
    }
    let (call_weights, val_weights) = hints.park.zip(hints.weights)?;
    let cost = park_cost(&live.calls, call_weights, ranges);
    (cost < val_weights[value]).then_some(cost)
}

#[derive(Clone, Copy)]
struct Victim {
    slot: usize,
    penalty: u64,
}

struct Demote {
    slot: usize,
    cost: u64,
    moves: Vec<(usize, usize)>,
    evicts: Vec<(usize, usize)>,
}

fn holder_leaf_cost<F: Copy + PartialEq>(pool: &Pool<F>, live: &Live, hints: &Hints, cost: &[u64], slot: usize, h: usize) -> u64 {
    let h_park = if crosses_ranges(&live.calls, live.ranges_of(h)) {
        leaf_park_cost(live, hints, h).unwrap_or(0)
    } else {
        0
    };
    cost[h]
        .saturating_add(u64::from(pool.kept[h] == Some(pool.regs[slot])))
        .saturating_sub(h_park)
}

fn find_demote<F: Copy + PartialEq>(
    pool: &Pool<F>,
    holders: &[Vec<usize>],
    assigned: &[Vec<usize>],
    live: &Live,
    hints: &Hints,
    reserved: &[F],
    (ranges, start, value): (&[Range], u32, usize),
) -> Option<Demote> {
    let val_cost = hints.weights.map(|w| w[value]);
    let mut best: Option<(u64, bool, Demote)> = None;
    for s in (0..pool.regs.len()).filter(|&s| pool.is_saved[s] && !(start == 0 && pool.arrival_lock[s] && pool.kept[value] != Some(pool.regs[s]))) {
        let mut moves: Vec<(usize, usize)> = Vec::new();
        let mut evicts: Vec<(usize, usize)> = Vec::new();
        let mut total_cost = 0_u64;
        let mut ok = true;
        for &h in &holders[s] {
            if !overlaps(live.ranges_of(h), ranges) {
                continue;
            }
            if hints.weights.is_none() && pool.kept[h].is_some() {
                ok = false;
                break;
            }
            let Some(h_park) = leaf_park_cost(live, hints, h) else {
                ok = false;
                break;
            };
            let h_ranges = live.ranges_of(h);
            let Some(h_start) = h_ranges.first().map(|r| r.start) else {
                ok = false;
                break;
            };
            let pick_free = |avoid_reserved: bool| {
                (0..pool.regs.len()).find(|&l| {
                    !pool.is_saved[l]
                        && (!avoid_reserved || !reserved.contains(&pool.regs[l]))
                        && slot_free(&assigned[l], live, h_ranges, pool.arrival_lock[l], h_start, pool.regs[l], pool.kept[h])
                        && moves.iter().all(|&(mh, ml)| ml != l || !overlaps(live.ranges_of(mh), h_ranges))
                })
            };
            if let Some(leaf_slot) = pick_free(true).or_else(|| pick_free(false)) {
                total_cost = total_cost.saturating_add(h_park);
                moves.push((h, leaf_slot));
                continue;
            }
            let Some(cost) = hints.weights else {
                ok = false;
                break;
            };
            let mut best_leaf: Option<(u64, bool, usize, Vec<(usize, usize)>)> = None;
            for l in (0..pool.regs.len()).filter(|&l| !pool.is_saved[l]) {
                if !slot_free(&[], live, h_ranges, pool.arrival_lock[l], h_start, pool.regs[l], pool.kept[h]) {
                    continue;
                }
                if moves.iter().any(|&(mh, ml)| ml == l && overlaps(live.ranges_of(mh), h_ranges)) {
                    continue;
                }
                let mut add_evict = 0_u64;
                let mut slot_evicts = Vec::new();
                for &eh in &assigned[l] {
                    if overlaps(live.ranges_of(eh), h_ranges) {
                        if !evicts.iter().any(|&(_, prev)| prev == eh) {
                            add_evict = add_evict.saturating_add(holder_leaf_cost(pool, live, hints, cost, l, eh));
                        }
                        slot_evicts.push((l, eh));
                    }
                }
                let is_res = reserved.contains(&pool.regs[l]);
                if best_leaf.as_ref().is_none_or(|&(bc, br, _, _)| (add_evict, is_res) < (bc, br)) {
                    best_leaf = Some((add_evict, is_res, l, slot_evicts));
                }
            }
            let Some((add_evict, _, leaf_slot, slot_evicts)) = best_leaf else {
                ok = false;
                break;
            };
            total_cost = total_cost.saturating_add(h_park).saturating_add(add_evict);
            moves.push((h, leaf_slot));
            for item in slot_evicts {
                if !evicts.contains(&item) {
                    evicts.push(item);
                }
            }
        }
        if !ok || moves.is_empty() {
            continue;
        }
        if total_cost > 0 && !val_cost.is_some_and(|vc| total_cost < vc) {
            continue;
        }
        let is_res = reserved.contains(&pool.regs[s]);
        if best.as_ref().is_none_or(|(bc, br, _)| (total_cost, is_res) < (*bc, *br)) {
            best = Some((total_cost, is_res, Demote { slot: s, cost: total_cost, moves, evicts }));
            if total_cost == 0 && !is_res {
                break;
            }
        }
    }
    best.map(|(_, _, demote)| demote)
}

fn apply_demote<F: Copy>(
    pool: &Pool<F>,
    holders: &mut [Vec<usize>],
    assigned: &mut [Vec<usize>],
    homes: &mut [Option<F>],
    demote: &Demote,
) -> Vec<usize> {
    let mut dropped = Vec::new();
    for &(leaf_slot, eh) in &demote.evicts {
        holders[leaf_slot].retain(|&x| x != eh);
        assigned[leaf_slot].retain(|&x| x != eh);
        if homes[eh].take().is_some() {
            dropped.push(eh);
        }
    }
    for &(other, leaf_slot) in &demote.moves {
        holders[demote.slot].retain(|&h| h != other);
        holders[leaf_slot].push(other);
        assigned[demote.slot].retain(|&h| h != other);
        assigned[leaf_slot].push(other);
        homes[other] = Some(pool.regs[leaf_slot]);
    }
    dropped
}

fn evict_slot<F: Copy>(
    holders: &mut Vec<usize>,
    assigned: &mut Vec<usize>,
    homes: &mut [Option<F>],
    live: &Live,
    ranges: &[Range],
) -> Vec<usize> {
    let mut dropped = Vec::new();
    holders.retain(|&other| {
        if overlaps(live.ranges_of(other), ranges) {
            assigned.retain(|&h| h != other);
            if homes[other].take().is_some() {
                dropped.push(other);
            }
            false
        } else {
            true
        }
    });
    dropped
}

fn victim_slot<F: Copy + PartialEq>(
    pool: &Pool<F>,
    holders: &[Vec<usize>],
    homes: &[Option<F>],
    live: &Live,
    hints: &Hints,
    back_partners: &[usize],
    remat: &[bool],
    (start, end, value): (u32, u32, usize),
    (needs_saved, can_park): (bool, bool),
    reserved: &[F],
    slot_used: impl Fn(usize) -> bool,
    suits: impl Fn(usize) -> bool,
) -> Option<Victim> {
    let ranges = live.ranges_of(value);
    let is_partner = |s: usize| back_partners.iter().any(|&p| homes[p] == Some(pool.regs[s]));
    let candidates = (0..pool.regs.len()).filter(|&s| suits(s) && !(start == 0 && pool.arrival_lock[s] && pool.kept[value] != Some(pool.regs[s])));
    let Some(cost) = hints.weights else {
        return candidates
            .filter_map(|s| {
                let mut count = 0_usize;
                let mut single = 0_usize;
                for &h in &holders[s] {
                    if overlaps(live.ranges_of(h), ranges) {
                        if pool.kept[h].is_some() {
                            return None;
                        }
                        single = h;
                        count += 1;
                    }
                }
                (count == 1).then(|| live.last_end(single).map(|u| (u, is_partner(s), s))).flatten()
            })
            .max_by_key(|&(u, partner, _)| (u, partner))
            .filter(|&(u, _, _)| u > end)
            .map(|(_, _, slot)| Victim { slot, penalty: 0 });
    };
    let val_park = if needs_saved && can_park { leaf_park_cost(live, hints, value).unwrap_or(0) } else { 0 };
    let mut best: Option<(u64, u8, bool, bool, bool, Reverse<u64>, usize)> = None;
    for s in candidates {
        let val_parks_here = !pool.is_saved[s] && needs_saved;
        let added_park = if val_parks_here { val_park } else { 0 };
        let cand_gain = cost[value]
            .saturating_add(u64::from(pool.kept[value] == Some(pool.regs[s])))
            .saturating_sub(added_park);
        if cand_gain == 0 {
            continue;
        }
        let mut evict_cost = 0_u64;
        let mut max_len = 0_u64;
        let mut count = 0_usize;
        let mut has_non_remat = false;
        for &h in &holders[s] {
            if overlaps(live.ranges_of(h), ranges) {
                let h_gain = if !pool.is_saved[s] {
                    holder_leaf_cost(pool, live, hints, cost, s, h)
                } else {
                    cost[h].saturating_add(u64::from(pool.kept[h] == Some(pool.regs[s])))
                };
                evict_cost = evict_cost.saturating_add(h_gain);
                has_non_remat |= !remat[h];
                max_len = max_len.max(live.total_len(h));
                count += 1;
            }
        }
        if count == 0 || evict_cost >= cand_gain {
            continue;
        }
        let penalty = evict_cost.saturating_add(added_park);
        let tier: u8 = if needs_saved {
            u8::from(!pool.is_saved[s])
        } else if !pool.is_saved[s] {
            0
        } else if slot_used(s) {
            1
        } else {
            2
        };
        let key = (penalty, tier, reserved.contains(&pool.regs[s]), !is_partner(s), has_non_remat, Reverse(max_len), s);
        if best.is_none_or(|b| key < b) {
            best = Some(key);
        }
    }
    best.map(|(penalty, _, _, _, _, _, slot)| Victim { slot, penalty })
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

fn value_priority(weights: &[u64], live: &Live, remat: &[bool], start: u32, value: usize) -> Priority {
    let weight = weights[value];
    let len = live.total_len(value).max(4);
    let density = (u128::from(weight) << 24) / u128::from(len);
    Priority { density, weight, non_remat: !remat[value], start: Reverse(start), tie: Reverse(value), value }
}

const MAX_REQUEUE: u8 = 4;

fn fill_reserved<F: Copy + PartialEq>(back_args: &[(usize, usize)], homes: &[Option<F>], live: &Live, (value, start, ranges): (usize, u32, &[Range]), check_order: bool, reserved: &mut Vec<F>) {
    reserved.clear();
    for &(param, arg) in back_args {
        if arg != value
            && let Some(reg) = homes[param]
            && homes[arg].is_none()
            && (!check_order || live.bounds(arg).is_some_and(|(arg_start, _)| arg_start >= start))
            && overlaps(live.ranges_of(arg), ranges)
            && !reserved.contains(&reg)
        {
            reserved.push(reg);
        }
        if !check_order
            && param != value
            && let Some(reg) = homes[arg]
            && homes[param].is_none()
            && overlaps(live.ranges_of(param), ranges)
            && !reserved.contains(&reg)
        {
            reserved.push(reg);
        }
    }
}

fn allocate_weighted<F: Copy + PartialEq>(func: &Function, live: &Live, pool: &Pool<F>, pooled: &[bool], hints: &Hints, weights: &[u64]) -> (Vec<Option<F>>, Vec<F>) {
    let mut homes = vec![None; pooled.len()];
    let mut assigned: Vec<Vec<usize>> = vec![Vec::new(); pool.regs.len()];
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
    let mut reserved: Vec<F> = Vec::new();
    let mut requeues = vec![0_u8; pooled.len()];
    while let Some(item) = heap.pop() {
        let value = item.value;
        if homes[value].is_some() {
            continue;
        }
        let Some((start, end)) = live.bounds(value) else { continue };
        let ranges = live.ranges_of(value);
        let wide_val = func.types[value].bytes() > 8;
        let needs_saved = live.trapped[value] || crosses_ranges(&live.calls, ranges);
        let can_park = needs_saved && parks(live, hints, value);
        if needs_saved && wide_val && !can_park {
            continue;
        }
        let has_free_saved = needs_saved && !wide_val && (0..pool.regs.len()).any(|s| pool.is_saved[s] && slot_free(&assigned[s], live, ranges, pool.arrival_lock[s], start, pool.regs[s], pool.kept[value]));
        fill_reserved(hints.back_args, &homes, live, (value, start, ranges), false, &mut reserved);
        let mut chosen = choose_free(
            pool,
            &assigned,
            &homes,
            live,
            hints,
            &back_partners[value],
            &reverse_reuse,
            &reserved,
            |s| !assigned[s].is_empty(),
            (value, start, end, wide_val, needs_saved, can_park, has_free_saved),
        );
        if chosen.is_none() {
            let demote = (!wide_val)
                .then(|| find_demote(pool, &assigned, &assigned, live, hints, &reserved, (ranges, start, value)))
                .flatten();
            let suits_any = |s: usize| if needs_saved { (pool.is_saved[s] && !wide_val) || (can_park && !pool.is_saved[s]) } else { !wide_val || !pool.is_saved[s] };
            let victim = if demote.as_ref().is_some_and(|d| d.cost == 0) {
                None
            } else {
                victim_slot(pool, &assigned, &homes, live, hints, &back_partners[value], &remat, (start, end, value), (needs_saved, can_park), &reserved, |s| !assigned[s].is_empty(), suits_any)
            };
            let dropped = if let Some(demote) = demote
                && victim.is_none_or(|v| demote.cost <= v.penalty)
            {
                chosen = Some(demote.slot);
                let mut unused_holders = vec![Vec::new(); pool.regs.len()];
                apply_demote(pool, &mut unused_holders, &mut assigned, &mut homes, &demote)
            } else if let Some(v) = victim {
                chosen = Some(v.slot);
                let mut slot_assigned = std::mem::take(&mut assigned[v.slot]);
                let mut shadow = slot_assigned.clone();
                let out = evict_slot(&mut shadow, &mut slot_assigned, &mut homes, live, ranges);
                assigned[v.slot] = slot_assigned;
                out
            } else {
                Vec::new()
            };
            for eh in dropped {
                if requeues[eh] < MAX_REQUEUE
                    && let Some((eh_start, _)) = live.bounds(eh)
                {
                    requeues[eh] += 1;
                    heap.push(value_priority(weights, live, &remat, eh_start, eh));
                }
            }
        }
        let Some(slot) = chosen else { continue };
        homes[value] = Some(pool.regs[slot]);
        assigned[slot].push(value);
    }
    let used = (0..pool.regs.len()).filter(|&s| pool.is_saved[s] && !assigned[s].is_empty()).map(|s| pool.regs[s]).collect();
    (homes, used)
}

pub(crate) fn allocate<F: Copy + PartialEq>(func: &Function, live: &Live, class: &Class<F>, pooled: &[bool], hints: &Hints) -> (Vec<Option<F>>, Vec<F>) {
    let pool = pool(func, class, live, pooled);
    if let Some(weights) = hints.weights {
        return allocate_weighted(func, live, &pool, pooled, hints, weights);
    }
    let mut homes = vec![None; pooled.len()];
    let mut holders: Vec<Vec<usize>> = vec![Vec::new(); pool.regs.len()];
    let mut assigned: Vec<Vec<usize>> = vec![Vec::new(); pool.regs.len()];
    for (v, reg) in pool.kept.iter().enumerate() {
        if let Some(reg) = *reg
            && let Some(slot) = pool.regs.iter().position(|r| *r == reg)
        {
            homes[v] = Some(reg);
            holders[slot].push(v);
            assigned[slot].push(v);
        }
    }
    let mut values: Vec<(u32, u32, usize)> = (0..pooled.len())
        .filter(|&v| pooled[v] && pool.kept[v].is_none())
        .filter_map(|v| live.bounds(v).map(|(start, end)| (start, end, v)))
        .collect();
    values.sort_unstable();
    let back_partners = back_partner_map(pooled.len(), pooled, hints.back_args);
    let mut used: Vec<F> = Vec::new();
    let mut reserved: Vec<F> = Vec::new();
    let remat = vec![false; pooled.len()];
    for (start, end, value) in values {
        for list in &mut holders {
            list.retain(|&other| live.last_end(other).is_some_and(|u| u >= start));
        }
        let ranges = live.ranges_of(value);
        let wide_val = func.types[value].bytes() > 8;
        let needs_saved = live.trapped[value] || crosses_ranges(&live.calls, ranges);
        let can_park = needs_saved && parks(live, hints, value);
        if needs_saved && wide_val && !can_park {
            continue;
        }
        let suits_direct = |s: usize| if needs_saved { pool.is_saved[s] && !wide_val } else { !wide_val || !pool.is_saved[s] };
        let suits_any = |s: usize| suits_direct(s) || (can_park && !pool.is_saved[s]);
        let has_free_saved = needs_saved && !wide_val && (0..pool.regs.len()).any(|s| pool.is_saved[s] && slot_free(&holders[s], live, ranges, pool.arrival_lock[s], start, pool.regs[s], None));
        fill_reserved(hints.back_args, &homes, live, (value, start, ranges), true, &mut reserved);
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
            (value, start, end, wide_val, needs_saved, can_park, has_free_saved),
        );
        if chosen.is_none() {
            let demote = (!wide_val)
                .then(|| find_demote(&pool, &holders, &assigned, live, hints, &reserved, (ranges, start, value)))
                .flatten();
            let victim = if demote.as_ref().is_some_and(|d| d.cost == 0) {
                None
            } else {
                victim_slot(&pool, &holders, &homes, live, hints, &back_partners[value], &remat, (start, end, value), (needs_saved, can_park), &reserved, |s| used.contains(&pool.regs[s]), suits_any)
            };
            if let Some(demote) = demote
                && victim.is_none_or(|v| demote.cost <= v.penalty)
            {
                apply_demote(&pool, &mut holders, &mut assigned, &mut homes, &demote);
                chosen = Some(demote.slot);
            } else if let Some(v) = victim {
                evict_slot(&mut holders[v.slot], &mut assigned[v.slot], &mut homes, live, ranges);
                chosen = Some(v.slot);
            }
        }
        let Some(slot) = chosen else { continue };
        let reg = pool.regs[slot];
        homes[value] = Some(reg);
        holders[slot].push(value);
        assigned[slot].push(value);
        if pool.is_saved[slot] && !used.contains(&reg) {
            used.push(reg);
        }
    }
    (homes, used)
}

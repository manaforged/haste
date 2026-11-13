use crate::alloc::Location;
use crate::frame::{Abi, Loc};
use crate::ir::{Function, Inst, Op, Signature, Value};
use crate::live::{Live, Range, overlaps};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ParkReg<R> {
    Int(R),
    Float(u8),
}

pub(crate) struct Class<'a, F> {
    pub saved: &'a [F],
    pub leaf: Vec<F>,
    pub arrivals: Vec<Vec<F>>,
    pub passed: Vec<Option<F>>,
}

pub(crate) struct Hints<'a> {
    pub reuse: &'a [Option<Value>],
    pub weights: Option<&'a [u64]>,
    pub partners: Option<&'a [Vec<Value>]>,
    pub park: Option<&'a [u64]>,
}

struct Pool<F> {
    regs: Vec<F>,
    is_saved: Vec<bool>,
    kept: Vec<Option<F>>,
    arrival_lock: Vec<bool>,
    passed: Vec<Option<F>>,
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
    let locked_arrivals: Vec<F> = class.arrivals.iter().flatten().copied().filter(|reg| class.leaf.contains(reg) && !kept.contains(&Some(*reg))).collect();
    let arrival_lock: Vec<bool> = regs.iter().map(|r| locked_arrivals.contains(r)).collect();
    let passed = class.passed.iter().map(|reg| reg.filter(|reg| class.leaf.contains(reg))).collect();
    Pool { regs, is_saved, kept, arrival_lock, passed }
}

fn slot_free(holders: &[usize], live: &Live, ranges: &[Range], locked_entry: bool, start: u32) -> bool {
    !(start == 0 && locked_entry) && holders.iter().all(|&other| !overlaps(live.ranges_of(other), ranges))
}

fn pick_slot<F: Copy + PartialEq>(
    pool: &Pool<F>,
    holders: &[Vec<usize>],
    live: &Live,
    ranges: &[Range],
    start: u32,
    used: &[F],
    reserved: &[F],
    allow_reserved: bool,
    pred: impl Fn(usize) -> bool,
) -> Option<usize> {
    let valid = |slot: usize| pred(slot) && (allow_reserved || !reserved.contains(&pool.regs[slot])) && slot_free(&holders[slot], live, ranges, pool.arrival_lock[slot], start);
    (0..pool.regs.len())
        .find(|&s| valid(s) && holders[s].is_empty() && (!pool.is_saved[s] || used.contains(&pool.regs[s])))
        .or_else(|| (0..pool.regs.len()).find(|&s| valid(s) && (!pool.is_saved[s] || used.contains(&pool.regs[s]))))
        .or_else(|| (0..pool.regs.len()).find(|&s| valid(s) && holders[s].is_empty()))
        .or_else(|| (0..pool.regs.len()).find(|&s| valid(s)))
}

fn loop_back_args(func: &Function) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for (b_idx, block) in func.blocks.iter().enumerate() {
        let mut edge = |target: crate::ir::Block, args: &[crate::ir::Value]| {
            if target.index() <= b_idx {
                for (param, arg) in func.blocks[target.index()].params.iter().zip(args) {
                    if param != arg {
                        pairs.push((param.index(), arg.index()));
                    }
                }
            }
        };
        for &inst in &block.insts {
            crate::frame::terminator_edges(&func.insts[inst.index()].op, &mut edge);
        }
    }
    pairs
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

fn find_demote<F: Copy + PartialEq>(
    pool: &Pool<F>,
    holders: &[Vec<usize>],
    assigned: &[Vec<usize>],
    live: &Live,
    hints: &Hints,
    reserved: &[F],
    (ranges, start, value): (&[Range], u32, usize),
) -> Option<(usize, u64, Vec<(usize, usize)>)> {
    let val_cost = hints.weights.map(|w| w[value]);
    let mut best: Option<(u64, bool, usize, Vec<(usize, usize)>)> = None;
    for s in (0..pool.regs.len()).filter(|&s| pool.is_saved[s] && !(start == 0 && pool.arrival_lock[s])) {
        let mut moves: Vec<(usize, usize)> = Vec::new();
        let mut total_park = 0_u64;
        let mut ok = true;
        for &h in &holders[s] {
            if !overlaps(live.ranges_of(h), ranges) {
                continue;
            }
            if pool.kept[h].is_some() {
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
            let pick_leaf = |avoid_reserved: bool| {
                (0..pool.regs.len()).find(|&l| {
                    !pool.is_saved[l]
                        && (!avoid_reserved || !reserved.contains(&pool.regs[l]))
                        && slot_free(&assigned[l], live, h_ranges, pool.arrival_lock[l], h_start)
                        && moves.iter().all(|&(mh, ml)| ml != l || !overlaps(live.ranges_of(mh), h_ranges))
                })
            };
            let Some(leaf_slot) = pick_leaf(true).or_else(|| pick_leaf(false)) else {
                ok = false;
                break;
            };
            total_park = total_park.saturating_add(h_park);
            moves.push((h, leaf_slot));
        }
        if !ok || moves.is_empty() {
            continue;
        }
        if total_park > 0 && !val_cost.is_some_and(|vc| total_park < vc) {
            continue;
        }
        let is_res = reserved.contains(&pool.regs[s]);
        if best.as_ref().is_none_or(|&(bp, br, _, _)| (total_park, is_res) < (bp, br)) {
            best = Some((total_park, is_res, s, moves));
            if total_park == 0 && !is_res {
                break;
            }
        }
    }
    best.map(|(cost, _, s, moves)| (s, cost, moves))
}

fn apply_demote<F: Copy>(
    pool: &Pool<F>,
    holders: &mut [Vec<usize>],
    assigned: &mut [Vec<usize>],
    homes: &mut [Option<F>],
    slot: usize,
    moves: &[(usize, usize)],
) {
    for &(other, leaf_slot) in moves {
        holders[slot].retain(|&h| h != other);
        holders[leaf_slot].push(other);
        assigned[slot].retain(|&h| h != other);
        assigned[leaf_slot].push(other);
        homes[other] = Some(pool.regs[leaf_slot]);
    }
}

fn evict_slot<F: Copy>(
    holders: &mut Vec<usize>,
    assigned: &mut Vec<usize>,
    homes: &mut [Option<F>],
    live: &Live,
    ranges: &[Range],
) {
    holders.retain(|&other| {
        if overlaps(live.ranges_of(other), ranges) {
            assigned.retain(|&h| h != other);
            homes[other] = None;
            false
        } else {
            true
        }
    });
}

fn victim_slot<F: Copy + PartialEq>(
    pool: &Pool<F>,
    holders: &[Vec<usize>],
    live: &Live,
    hints: &Hints,
    (start, end, value): (u32, u32, usize),
    (needs_saved, can_park): (bool, bool),
    (used, reserved): (&[F], &[F]),
    suits: impl Fn(usize) -> bool,
) -> Option<Victim> {
    let ranges = live.ranges_of(value);
    let candidates = (0..pool.regs.len()).filter(|&s| suits(s) && !(start == 0 && pool.arrival_lock[s]));
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
                (count == 1).then(|| live.last_end(single).map(|u| (u, s))).flatten()
            })
            .max_by_key(|&(u, _)| u)
            .filter(|&(u, _)| u > end)
            .map(|(_, slot)| Victim { slot, penalty: 0 });
    };
    let val_park = if needs_saved && can_park { leaf_park_cost(live, hints, value).unwrap_or(0) } else { 0 };
    let mut best: Option<(u64, u8, bool, std::cmp::Reverse<u64>, usize)> = None;
    for s in candidates {
        let val_parks_here = !pool.is_saved[s] && needs_saved;
        let added_park = if val_parks_here { val_park } else { 0 };
        let cand_gain = cost[value].saturating_sub(added_park);
        if cand_gain == 0 {
            continue;
        }
        let mut evict_cost = 0_u64;
        let mut max_len = 0_u64;
        let mut count = 0_usize;
        let mut single = 0_usize;
        let mut ok = true;
        for &h in &holders[s] {
            if overlaps(live.ranges_of(h), ranges) {
                if pool.kept[h].is_some() {
                    ok = false;
                    break;
                }
                let h_park = if !pool.is_saved[s] && crosses_ranges(&live.calls, live.ranges_of(h)) {
                    leaf_park_cost(live, hints, h).unwrap_or(0)
                } else {
                    0
                };
                evict_cost = evict_cost.saturating_add(cost[h].saturating_sub(h_park));
                max_len = max_len.max(live.total_len(h));
                single = h;
                count += 1;
            }
        }
        if !ok || count == 0 {
            continue;
        }
        let cheaper = if count == 1 {
            if val_parks_here {
                evict_cost < cand_gain
            } else {
                let (len_single, len_val) = (live.total_len(single), live.total_len(value));
                evict_cost < cand_gain
                    || (evict_cost == cand_gain && len_single > len_val)
                    || (len_single >= len_val.max(2).saturating_mul(4) && u128::from(evict_cost) * u128::from(len_val.max(2)) < u128::from(cand_gain) * u128::from(len_single))
            }
        } else {
            evict_cost.saturating_mul(2) < cand_gain
        };
        if !cheaper {
            continue;
        }
        let penalty = evict_cost.saturating_add(added_park);
        let tier: u8 = if needs_saved {
            u8::from(!pool.is_saved[s])
        } else if !pool.is_saved[s] {
            0
        } else if used.contains(&pool.regs[s]) {
            1
        } else {
            2
        };
        let key = (penalty, tier, reserved.contains(&pool.regs[s]), std::cmp::Reverse(max_len), s);
        if best.is_none_or(|b| key < b) {
            best = Some(key);
        }
    }
    best.map(|(penalty, _, _, _, slot)| Victim { slot, penalty })
}

pub(crate) fn allocate<F: Copy + PartialEq>(func: &Function, live: &Live, class: &Class<F>, pooled: &[bool], hints: &Hints) -> (Vec<Option<F>>, Vec<F>) {
    let pool = pool(func, class, live, pooled);
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
    let back_args = loop_back_args(func);
    let mut used: Vec<F> = Vec::new();
    let mut reserved: Vec<F> = Vec::new();
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
        let has_free_saved = needs_saved && !wide_val && (0..pool.regs.len()).any(|s| pool.is_saved[s] && slot_free(&holders[s], live, ranges, pool.arrival_lock[s], start));
        reserved.clear();
        for &(param, arg) in &back_args {
            if arg != value
                && let Some(reg) = homes[param]
                && homes[arg].is_none()
                && live.bounds(arg).is_some_and(|(arg_start, _)| arg_start >= start)
                && !reserved.contains(&reg)
            {
                reserved.push(reg);
            }
        }
        let mut chosen = hints.reuse[value].and_then(|op| {
            let op_idx = op.index();
            let reg = homes[op_idx]?;
            let s = pool.regs.iter().position(|r| *r == reg)?;
            let suits_reuse = if has_free_saved { suits_direct(s) } else { suits_any(s) };
            if !suits_reuse || (start == 0 && pool.arrival_lock[s]) || reserved.contains(&reg) {
                return None;
            }
            if !holders[s].iter().all(|&h| h == op_idx || !overlaps(live.ranges_of(h), ranges)) {
                return None;
            }
            holders[s].retain(|&h| h != op_idx);
            Some(s)
        });
        if chosen.is_none()
            && let Some(partners) = hints.partners
        {
            chosen = partners[value].iter().filter_map(|p| homes[p.index()]).find_map(|reg| {
                let s = pool.regs.iter().position(|r| *r == reg)?;
                let ok = if needs_saved { (pool.is_saved[s] || (can_park && !has_free_saved)) && !wide_val } else { !pool.is_saved[s] || (!wide_val && used.contains(&reg)) };
                (ok && slot_free(&holders[s], live, ranges, pool.arrival_lock[s], start)).then_some(s)
            });
        }
        if chosen.is_none()
            && let Some(reg) = pool.passed.get(value).copied().flatten()
        {
            let call_at = end + 1;
            let plain_call = live.calls.binary_search(&call_at).is_ok() && live.unwinds.binary_search(&call_at).is_err();
            if plain_call && !crosses_ranges(&live.calls, ranges)
                && let Some(s) = pool.regs.iter().position(|r| *r == reg)
                && slot_free(&holders[s], live, ranges, pool.arrival_lock[s], start)
            {
                chosen = Some(s);
            }
        }
        if chosen.is_none() {
            chosen = if needs_saved {
                pick_slot(&pool, &holders, live, ranges, start, &used, &reserved, false, |s| pool.is_saved[s] && !wide_val)
                    .or_else(|| can_park.then(|| pick_slot(&pool, &holders, live, ranges, start, &used, &reserved, false, |s| !pool.is_saved[s])).flatten())
                    .or_else(|| pick_slot(&pool, &holders, live, ranges, start, &used, &reserved, true, |s| pool.is_saved[s] && !wide_val))
                    .or_else(|| can_park.then(|| pick_slot(&pool, &holders, live, ranges, start, &used, &reserved, true, |s| !pool.is_saved[s])).flatten())
            } else {
                pick_slot(&pool, &holders, live, ranges, start, &used, &reserved, false, |s| !pool.is_saved[s])
                    .or_else(|| (!wide_val).then(|| pick_slot(&pool, &holders, live, ranges, start, &used, &reserved, false, |s| pool.is_saved[s] && used.contains(&pool.regs[s]))).flatten())
                    .or_else(|| (!wide_val).then(|| pick_slot(&pool, &holders, live, ranges, start, &used, &reserved, false, |s| pool.is_saved[s])).flatten())
                    .or_else(|| pick_slot(&pool, &holders, live, ranges, start, &used, &reserved, true, |s| !pool.is_saved[s]))
                    .or_else(|| (!wide_val).then(|| pick_slot(&pool, &holders, live, ranges, start, &used, &reserved, true, |s| pool.is_saved[s])).flatten())
            };
        }
        if chosen.is_none() {
            let demote = (!wide_val)
                .then(|| find_demote(&pool, &holders, &assigned, live, hints, &reserved, (ranges, start, value)))
                .flatten();
            let victim = if demote.as_ref().is_some_and(|(_, cost, _)| *cost == 0) {
                None
            } else {
                victim_slot(&pool, &holders, live, hints, (start, end, value), (needs_saved, can_park), (&used, &reserved), suits_any)
            };
            if let Some((s, d_cost, moves)) = demote
                && victim.is_none_or(|v| d_cost <= v.penalty)
            {
                apply_demote(&pool, &mut holders, &mut assigned, &mut homes, s, &moves);
                chosen = Some(s);
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

fn passed_own<R: Copy + PartialEq>(func: &Function, inst: Inst, value: Value, reg: ParkReg<R>, classify: &impl Fn(&Signature) -> Result<Abi<R>, String>) -> bool {
    let Op::Call { sig, args, .. } = &func.insts[inst.index()].op else { return false };
    let Ok(abi) = classify(&func.sigs[sig.index()]) else { return false };
    args.iter().zip(&abi.params).any(|(arg, loc)| {
        *arg == value
            && match (reg, *loc) {
                (ParkReg::Int(r), Loc::Int(at)) => r == at,
                (ParkReg::Float(v), Loc::Float(at)) => v == at,
                _ => false,
            }
    })
}

pub(crate) fn parked<R: Copy + PartialEq>(
    func: &Function,
    live: &Live,
    locations: &[Location<R>],
    (saved_int, saved_float): (&[R], &[u8]),
    classify: &impl Fn(&Signature) -> Result<Abi<R>, String>,
) -> Vec<(Inst, Vec<(Value, ParkReg<R>, bool)>)> {
    let mut at_call: Vec<Vec<(Value, ParkReg<R>, bool)>> = vec![Vec::new(); live.calls.len()];
    for (index, location) in locations.iter().enumerate() {
        let park_reg = match *location {
            Location::Reg(r) if !saved_int.contains(&r) => ParkReg::Int(r),
            Location::Float(v) if !saved_float.contains(&v) => ParkReg::Float(v),
            _ => continue,
        };
        let ranges = live.ranges_of(index);
        if ranges.is_empty() {
            continue;
        }
        let value = Value(index as u32);
        for r in ranges {
            let first = live.calls.partition_point(|at| *at <= r.start);
            let last = live.calls.partition_point(|at| *at <= r.end);
            for call in first..last {
                let at = live.calls[call];
                if at < r.end || !passed_own(func, live.sites[call].0, value, park_reg, classify) {
                    at_call[call].push((value, park_reg, at < r.end));
                }
            }
        }
    }
    let mut out: Vec<(Inst, Vec<(Value, ParkReg<R>, bool)>)> = live.sites.iter().map(|site| site.0).zip(at_call).filter(|(_, values)| !values.is_empty()).collect();
    out.sort_unstable_by_key(|(inst, _)| inst.index());
    out
}

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
    let Some((call_weights, val_weights)) = hints.park.zip(hints.weights) else { return false };
    let ranges = live.ranges_of(value);
    if ranges.is_empty() || live.trapped[value] || crosses_ranges(&live.unwinds, ranges) {
        return false;
    }
    park_cost(&live.calls, call_weights, ranges) < val_weights[value]
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
            match &func.insts[inst.index()].op {
                crate::ir::Op::Jump(target, args) => edge(*target, args),
                crate::ir::Op::Brif(_, then, then_args, otherwise, else_args) => {
                    edge(*then, then_args);
                    edge(*otherwise, else_args);
                }
                _ => {}
            }
        }
    }
    pairs
}

fn try_demote<F: Copy + PartialEq>(
    pool: &Pool<F>,
    holders: &mut [Vec<usize>],
    assigned: &mut [Vec<usize>],
    homes: &mut [Option<F>],
    live: &Live,
    hints: &Hints,
    ranges: &[Range],
    start: u32,
) -> Option<usize> {
    for s in (0..pool.regs.len()).filter(|&s| pool.is_saved[s] && !(start == 0 && pool.arrival_lock[s])) {
        let mut iter = holders[s].iter().copied().filter(|&h| overlaps(live.ranges_of(h), ranges));
        let Some(other) = iter.next() else { continue };
        if iter.next().is_some() || pool.kept[other].is_some() || !parks(live, hints, other) {
            continue;
        }
        let other_ranges = live.ranges_of(other);
        let Some(other_start) = other_ranges.first().map(|r| r.start) else { continue };
        let Some(leaf_slot) = (0..pool.regs.len()).find(|&l| !pool.is_saved[l] && slot_free(&assigned[l], live, other_ranges, pool.arrival_lock[l], other_start)) else {
            continue;
        };
        holders[s].retain(|&h| h != other);
        holders[leaf_slot].push(other);
        assigned[s].retain(|&h| h != other);
        assigned[leaf_slot].push(other);
        homes[other] = Some(pool.regs[leaf_slot]);
        return Some(s);
    }
    None
}

fn victim_slot<F: Copy + PartialEq>(
    pool: &Pool<F>,
    holders: &[Vec<usize>],
    live: &Live,
    (start, end, value): (u32, u32, usize),
    weights: Option<&[u64]>,
    suits: impl Fn(usize) -> bool,
) -> Option<usize> {
    let candidates = (0..pool.regs.len()).filter(|&s| suits(s) && !(start == 0 && pool.arrival_lock[s]) && holders[s].len() == 1);
    let Some(cost) = weights else {
        return candidates
            .filter_map(|s| live.last_end(holders[s][0]).map(|u| (u, s)))
            .max_by_key(|&(u, _)| u)
            .filter(|&(u, _)| u > end)
            .map(|(_, s)| s);
    };
    let below = |v: usize, u: usize| {
        if cost[v] <= 2 && cost[u] <= 2 {
            u128::from(cost[v]) * u128::from(live.total_len(u)) < u128::from(cost[u]) * u128::from(live.total_len(v))
        } else {
            cost[v] < cost[u] || (cost[v] == cost[u] && live.total_len(v) > live.total_len(u))
        }
    };
    let best = candidates.reduce(|a, b| if below(holders[b][0], holders[a][0]) { b } else { a })?;
    below(holders[best][0], value).then_some(best)
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
        if chosen.is_none() && needs_saved && !wide_val {
            chosen = try_demote(&pool, &mut holders, &mut assigned, &mut homes, live, hints, ranges, start);
        }
        if chosen.is_none()
            && let Some(s) = victim_slot(&pool, &holders, live, (start, end, value), hints.weights, suits_any)
            && let Some(other) = holders[s].pop()
        {
            assigned[s].retain(|&h| h != other);
            homes[other] = None;
            chosen = Some(s);
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

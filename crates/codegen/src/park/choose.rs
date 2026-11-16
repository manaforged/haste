use crate::ir::{Function, Value};
use crate::live::{Live, Range, overlaps};

use super::{BackIndex, Hints, Pool, SlotSet, crosses_call, crosses_ranges, parks, slot_free, try_place_weighted};

fn reuse_pair(hints: &Hints, reverse_reuse: &[Option<Value>], value: usize) -> [Option<Value>; 2] {
    [hints.reuse[value], reverse_reuse.get(value).copied().flatten()]
}

fn pick_slot<F: Copy + PartialEq>(
    pool: &Pool<F>,
    holders: &[Vec<usize>],
    live: &Live,
    ranges: &[Range],
    start: u32,
    kept_reg: Option<F>,
    reserved: &[F],
    allow_reserved: bool,
    slot_used: impl Fn(usize) -> bool,
    tier: impl Fn(usize) -> u8,
    pred: impl Fn(usize) -> bool,
) -> Option<usize> {
    let free = |s: usize| !pool.is_saved[s] || slot_used(s);
    (0..pool.regs.len())
        .filter(|&s| {
            pred(s)
                && (allow_reserved || !reserved.contains(&pool.regs[s]))
                && slot_free(pool, s, &holders[s], live, ranges, start, kept_reg)
        })
        .min_by_key(|&s| (tier(s), !free(s), holders[s].is_empty(), s))
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
    (value, start, end, wide_val, needs_saved, can_park, has_free_saved): (usize, u32, u32, bool, bool, bool, bool),
) -> Option<usize> {
    let ranges = live.ranges_of(value);
    let kept_reg = pool.kept[value];
    let suits_direct = |s: usize| if needs_saved { pool.is_saved[s] && !wide_val } else { !wide_val || !pool.is_saved[s] };
    let suits_any = |s: usize| suits_direct(s) || (can_park && !pool.is_saved[s]);
    if let Some(reg) = kept_reg
        && let Some(s) = pool.regs.iter().position(|r| *r == reg)
        && suits_any(s)
        && slot_free(pool, s, &holders[s], live, ranges, start, kept_reg)
    {
        return Some(s);
    }
    let can_use = |reg: F, check_reserved: bool| -> Option<usize> {
        if check_reserved && reserved.contains(&reg) {
            return None;
        }
        let s = pool.regs.iter().position(|r| *r == reg)?;
        let ok = if needs_saved { (pool.is_saved[s] || (can_park && !has_free_saved)) && !wide_val } else { !pool.is_saved[s] || (!wide_val && slot_used(s)) };
        (ok && slot_free(pool, s, &holders[s], live, ranges, start, kept_reg)).then_some(s)
    };
    let try_partner = |p: usize, check_reserved: bool| homes[p].and_then(|reg| can_use(reg, check_reserved));
    if let Some(s) = back_partners.iter().find_map(|&p| try_partner(p, false)) {
        return Some(s);
    }
    if !hints.two_address
        && let Some(partners) = hints.partners
        && let Some(s) = partners[value].iter().find_map(|p| try_partner(p.index(), true))
    {
        return Some(s);
    }
    let unalloc_info = |p: usize| -> Option<(usize, bool, u32)> {
        if homes[p].is_some() {
            return None;
        }
        let pr = live.ranges_of(p);
        (!crosses_ranges(&live.unwinds, pr)).then(|| Some((p, crosses_call(live, p), pr.first()?.start))).flatten()
    };
    let partners = hints.partners.map_or(&[][..], |m| &m[value][..]);
    let unalloc_back: Vec<(usize, bool, u32)> = back_partners.iter().copied().filter_map(unalloc_info).collect();
    let unalloc_fwd: Vec<(usize, bool, u32)> = partners
        .iter()
        .map(|p| p.index())
        .filter(|p| !back_partners.contains(p))
        .filter_map(unalloc_info)
        .collect();
    let unalloc_reuse: Vec<(usize, bool, u32)> = reuse_pair(hints, reverse_reuse, value)
        .into_iter()
        .flatten()
        .map(|p| p.index())
        .filter(|&p| p < homes.len() && !back_partners.contains(&p) && !partners.iter().any(|q| q.index() == p))
        .filter_map(unalloc_info)
        .collect();
    let p_fits = |&(p, p_saved, p_start): &(usize, bool, u32), s: usize| {
        (!p_saved || wide_val || pool.is_saved[s]) && slot_free(pool, s, &holders[s], live, live.ranges_of(p), p_start, pool.kept[p])
    };
    let all_fit = |list: &[(usize, bool, u32)], s: usize| list.iter().all(|info| p_fits(info, s));
    let compat_back = |s: usize| all_fit(&unalloc_back, s);
    let compat_fwd = |s: usize| all_fit(&unalloc_fwd, s);
    let compat_reuse = |s: usize| all_fit(&unalloc_reuse, s);
    for op in reuse_pair(hints, reverse_reuse, value).into_iter().flatten() {
        let Some(reg) = homes[op.index()] else { continue };
        let Some(s) = pool.regs.iter().position(|r| *r == reg) else { continue };
        let suits_reuse = if has_free_saved { suits_direct(s) } else { suits_any(s) };
        if suits_reuse && (hints.two_address || compat_back(s)) && !reserved.contains(&reg) && slot_free(pool, s, &holders[s], live, ranges, start, kept_reg) {
            return Some(s);
        }
    }
    if hints.two_address
        && let Some(partners) = hints.partners
        && let Some(s) = partners[value].iter().find_map(|p| try_partner(p.index(), true))
    {
        return Some(s);
    }
    if let Some((reg, at_call_end)) = pool.passed.get(value).copied().flatten() {
        let call_at = end + 1;
        let plain_call = live.calls.binary_search(&call_at).is_ok() && live.unwinds.binary_search(&call_at).is_err();
        if (!at_call_end || plain_call)
            && !crosses_call(live, value)
            && !reserved.contains(&reg)
            && let Some(s) = pool.regs.iter().position(|r| *r == reg)
            && slot_free(pool, s, &holders[s], live, ranges, start, kept_reg)
        {
            return Some(s);
        }
    }
    let (first_unalloc, second_unalloc) = if hints.two_address { (&unalloc_reuse, &unalloc_fwd) } else { (&unalloc_fwd, &unalloc_reuse) };
    for &(p, p_saved, p_start) in unalloc_back.iter().chain(first_unalloc).chain(second_unalloc) {
        let pr = live.ranges_of(p);
        if overlaps(pr, ranges) {
            continue;
        }
        let check = |reg: F, extra: Option<usize>| {
            let s = can_use(reg, true)?;
            if !p_fits(&(p, p_saved, p_start), s) || !compat_back(s) {
                return None;
            }
            if let Some(q) = extra {
                let qr = live.ranges_of(q);
                if q == value || q == p || overlaps(qr, ranges) || overlaps(qr, pr) {
                    return None;
                }
            }
            Some(s)
        };
        if let Some(s) = pool.kept[p].and_then(|r| check(r, None)) {
            return Some(s);
        }
        if !p_saved
            && let Some((r, _)) = pool.passed.get(p).copied().flatten()
            && let Some(s) = check(r, None)
        {
            return Some(s);
        }
        let p_reuse = reuse_pair(hints, reverse_reuse, p);
        for q in hints
            .partners
            .map_or(&[][..], |m| &m[p][..])
            .iter()
            .copied()
            .chain(p_reuse.into_iter().flatten())
            .map(|v| v.index())
            .filter(|&q| q != value && q < homes.len())
        {
            if let Some(s) = homes[q].and_then(|r| check(r, None)).or_else(|| pool.kept[q].and_then(|r| check(r, Some(q)))) {
                return Some(s);
            }
        }
    }
    let tier = |s: usize| {
        let (sec, third) = if hints.two_address {
            (!compat_reuse(s), !compat_fwd(s))
        } else {
            (!compat_fwd(s), !compat_reuse(s))
        };
        u8::from(!compat_back(s)) * 4 + u8::from(sec) * 2 + u8::from(third)
    };
    let try_pick = |allow_res: bool, pred: &dyn Fn(usize) -> bool| pick_slot(pool, holders, live, ranges, start, kept_reg, reserved, allow_res, &slot_used, &tier, pred);
    if needs_saved {
        try_pick(false, &|s| pool.is_saved[s] && !wide_val)
            .or_else(|| can_park.then(|| try_pick(false, &|s| !pool.is_saved[s])).flatten())
            .or_else(|| try_pick(true, &|s| pool.is_saved[s] && !wide_val))
            .or_else(|| can_park.then(|| try_pick(true, &|s| !pool.is_saved[s])).flatten())
    } else {
        let back_needs_saved = !wide_val && unalloc_back.iter().any(|&(_, p_saved, _)| p_saved);
        let fwd_needs_saved = !wide_val && unalloc_fwd.iter().any(|&(p, p_saved, _)| p_saved && !parks(live, hints, p));
        let reuse_needs_saved = !wide_val && unalloc_reuse.iter().any(|&(p, p_saved, _)| p_saved && !parks(live, hints, p));
        if back_needs_saved
            && let Some(s) = try_pick(false, &|s| pool.is_saved[s] && compat_back(s))
        {
            return Some(s);
        }
        if !unalloc_back.is_empty()
            && let Some(s) = try_pick(false, &|s| !pool.is_saved[s] && compat_back(s))
                .or_else(|| (!wide_val).then(|| try_pick(false, &|s| pool.is_saved[s] && compat_back(s))).flatten())
        {
            return Some(s);
        }
        let (first_saved, second_saved) = if hints.two_address {
            ((reuse_needs_saved, &compat_reuse as &dyn Fn(usize) -> bool), (fwd_needs_saved, &compat_fwd as &dyn Fn(usize) -> bool))
        } else {
            ((fwd_needs_saved, &compat_fwd as &dyn Fn(usize) -> bool), (reuse_needs_saved, &compat_reuse as &dyn Fn(usize) -> bool))
        };
        for (needed, compat) in [first_saved, second_saved] {
            if needed
                && let Some(s) = try_pick(false, &|s| pool.is_saved[s] && compat(s))
            {
                return Some(s);
            }
        }
        try_pick(false, &|s| !pool.is_saved[s])
            .or_else(|| (!wide_val).then(|| try_pick(false, &|s| pool.is_saved[s] && slot_used(s))).flatten())
            .or_else(|| (!wide_val).then(|| try_pick(false, &|s| pool.is_saved[s])).flatten())
            .or_else(|| try_pick(true, &|s| !pool.is_saved[s]))
            .or_else(|| (!wide_val).then(|| try_pick(true, &|s| pool.is_saved[s])).flatten())
    }
}

/// Places the unhomed reuse partners of a just-placed value in its register now, when that is their own current choice.
pub(super) fn place_reuse_chain<'a, F: Copy + PartialEq>(
    ctx: (&Function, &'a Live, &Pool<F>),
    weighted: (&Hints, &[u64]),
    index: (&[Vec<usize>], &[Option<Value>], &BackIndex),
    (slots, homes): (&mut SlotSet, &mut [Option<F>]),
    (reserved, scratch_holders, query_ranges): (&mut Vec<F>, &mut Vec<Vec<usize>>, &mut Vec<(usize, &'a [Range])>),
    (pooled, value): (&[bool], usize),
) {
    let (_, _, pool) = ctx;
    let (hints, _) = weighted;
    let (_, reverse_reuse, _) = index;
    let Some(slot) = homes[value].and_then(|reg| pool.regs.iter().position(|r| *r == reg)) else { return };
    let mut queue = vec![value];
    while let Some(cur) = queue.pop() {
        for next in reuse_pair(hints, reverse_reuse, cur).into_iter().flatten().map(|v| v.index()) {
            if pooled.get(next) == Some(&true)
                && homes[next].is_none()
                && try_place_weighted(ctx, weighted, index, (&mut *slots, &mut *homes), (&mut *reserved, &mut *scratch_holders, &mut *query_ranges), (next, Some(slot))).0
            {
                queue.push(next);
            }
        }
    }
}

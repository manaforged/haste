use rustc_hash::FxHashSet as HashSet;

use crate::names::ItemRef;
use crate::sema::impls::{Holder, ImplRef, Index as ImplIndex};
use crate::sema::infer::Form;
use crate::sema::ty::Ty;

use super::super::SummaryKind;
use super::{
    WARM_DEF_SHARDS, WARM_HDR_SHARDS, WARM_NON_SHARD_SECTIONS, WARM_SUM_SHARDS, WarmFlights,
    debug_warm,
};

fn collect_impl_owners(r: ImplRef, owners: &mut HashSet<ItemRef>) {
    if let ImplRef::Body(item) = r
        && let Some((outer, _)) = item.outer()
    {
        owners.insert(outer.body_owner());
    }
}

fn collect_index_owners(holder: Holder, idx: &ImplIndex, owners: &mut HashSet<ItemRef>) {
    if let Holder::Body(root) = holder {
        owners.insert(root.body_owner());
    }
    for &r in idx.inherent.values().flatten() {
        collect_impl_owners(r, owners);
    }
    for b in idx.traits.values() {
        for &r in b
            .all
            .iter()
            .chain(&b.blanket)
            .chain(b.by_head.values().flatten())
        {
            collect_impl_owners(r, owners);
        }
    }
}

fn impl_ref_ready(cx: &crate::sema::Context<'_>, r: ImplRef) -> bool {
    match r {
        ImplRef::Module { .. } => true,
        ImplRef::Body(item) => match item.outer() {
            Some((outer, at)) => cx
                .expansion(outer)
                .is_some_and(|exp| (at as usize) < exp.file.items.items.len()),
            None => {
                (item.item as usize)
                    < cx.outline.crates[item.krate as usize].files[item.file as usize]
                        .items
                        .items
                        .len()
            }
        },
    }
}

fn impl_index_ready(cx: &crate::sema::Context<'_>, holder: Holder, idx: &ImplIndex) -> bool {
    if let Holder::Body(root) = holder
        && !impl_ref_ready(cx, ImplRef::Body(root))
    {
        return false;
    }
    idx.inherent
        .values()
        .flatten()
        .all(|&r| impl_ref_ready(cx, r))
        && idx.traits.values().all(|b| {
            b.all
                .iter()
                .chain(&b.blanket)
                .chain(b.by_head.values().flatten())
                .all(|&r| impl_ref_ready(cx, r))
        })
}

fn expand_body_owners(cx: &mut crate::sema::Context<'_>, todo: Vec<ItemRef>) {
    let workers = crate::jobs().min(todo.len());
    if workers <= 1 || todo.len() < 2 {
        for owner in todo {
            let _ = cx.parse_body(owner);
        }
        return;
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let asks_list = crate::sync::scope(|s| {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                let mut worker = cx.fork();
                s.spawn(|| {
                    loop {
                        let idx = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if idx >= todo.len() {
                            break;
                        }
                        let _ = worker.parse_body(todo[idx]);
                    }
                    worker.asks
                })
            })
            .collect();
        let mut out = Vec::with_capacity(workers);
        for h in handles {
            match h.join() {
                Ok(asks) => out.push(asks),
                Err(payload) => std::panic::resume_unwind(payload),
            }
        }
        out
    });
    for asks in asks_list {
        cx.asks.repeat(&asks);
    }
}

fn populate_flight_chunks(
    cx: &mut crate::sema::Context<'_>,
    reveals_list: Vec<(crate::sema::reach::Instance, Ty)>,
    checked_list: Vec<(ImplRef, bool)>,
    summaries_list: Vec<(ItemRef, SummaryKind)>,
) {
    let reveals = cx.store().lookups.reveals.clone();
    let checked = cx.store().lookups.checked_impls.clone();
    let summaries = cx.store().lookups.summaries.clone();
    let total = reveals_list.len() + checked_list.len() + summaries_list.len();
    let workers = crate::jobs().min(8);
    if workers <= 1 || total < 64 {
        for (inst, ty) in reveals_list {
            let _ = reveals.run_cached(cx, inst, |_| Ok(ty), |_| Ok(Ty::NONE));
        }
        for (block, val) in checked_list {
            let _ = checked.run_cached(cx, block, |_| Ok(val), |_| Ok(false));
        }
        for (item, kind) in summaries_list {
            let _ = summaries.run_cached(
                cx,
                item,
                |_| Ok(kind.clone()),
                |_| Ok(SummaryKind::MonomorphicFallback),
            );
        }
        return;
    }
    let r_next = std::sync::atomic::AtomicUsize::new(0);
    let c_next = std::sync::atomic::AtomicUsize::new(0);
    let s_next = std::sync::atomic::AtomicUsize::new(0);
    let (reveals_ref, checked_ref, summaries_ref) = (&reveals_list, &checked_list, &summaries_list);
    let (r_at, c_at, s_at) = (&r_next, &c_next, &s_next);
    crate::sync::scope(|s| {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                let mut worker = cx.fork();
                let (reveals, checked, summaries) =
                    (reveals.clone(), checked.clone(), summaries.clone());
                s.spawn(move || {
                    loop {
                        let i = r_at.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if i >= reveals_ref.len() {
                            break;
                        }
                        let (inst, ty) = reveals_ref[i];
                        let _ = reveals.run_cached(&mut worker, inst, |_| Ok(ty), |_| Ok(Ty::NONE));
                    }
                    loop {
                        let i = c_at.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if i >= checked_ref.len() {
                            break;
                        }
                        let (block, val) = checked_ref[i];
                        let _ = checked.run_cached(&mut worker, block, |_| Ok(val), |_| Ok(false));
                    }
                    loop {
                        let i = s_at.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if i >= summaries_ref.len() {
                            break;
                        }
                        let (item, kind) = &summaries_ref[i];
                        let _ = summaries.run_cached(
                            &mut worker,
                            *item,
                            |_| Ok(kind.clone()),
                            |_| Ok(SummaryKind::MonomorphicFallback),
                        );
                    }
                })
            })
            .collect();
        for h in handles {
            if let Err(payload) = h.join() {
                std::panic::resume_unwind(payload);
            }
        }
    });
}

/// Inserts preloaded `Flight` entries (`reveals`, `checked_impls`, `summaries`, and body-item caches) onto `cx`.
pub(in crate::sema) fn warm_populate_flights(
    cx: &mut crate::sema::Context<'_>,
    flights: WarmFlights,
) {
    let t0 = debug_warm().then(std::time::Instant::now);
    let mut owners = HashSet::default();
    for (holder, idx) in &flights.body_heads {
        collect_index_owners(*holder, idx, &mut owners);
    }
    for &(r, _) in &flights.body_headers {
        collect_impl_owners(r, &mut owners);
    }
    for &((form, _), _) in &flights.body_forms {
        if let Form::Impl(r) = form {
            collect_impl_owners(r, &mut owners);
        }
    }
    for &(r, _) in &flights.body_impl_traits {
        collect_impl_owners(r, &mut owners);
    }
    for &(_, val) in &flights.body_selections {
        if let Some((r, _)) = val {
            collect_impl_owners(r, &mut owners);
        }
    }
    for &(r, _) in &flights.checked_impls {
        collect_impl_owners(r, &mut owners);
    }
    let todo: Vec<ItemRef> = owners
        .into_iter()
        .filter(|&owner| cx.expansion(owner).is_none())
        .collect();
    if !todo.is_empty() {
        expand_body_owners(cx, todo);
    }
    let t_owners = t0.map(|s| s.elapsed().as_micros() as u64);

    let lk = &cx.store().lookups;
    for (holder, idx) in flights.body_heads {
        if impl_index_ready(cx, holder, &idx) {
            lk.heads.insert(holder, idx);
        }
    }
    for (r, hdr) in flights.body_headers {
        if impl_ref_ready(cx, r) {
            cx.store().headers.insert(r, hdr);
        }
    }
    for (key, fields) in flights.body_forms {
        if !matches!(key.0, Form::Impl(r) if !impl_ref_ready(cx, r)) {
            lk.forms.insert(key, fields);
        }
    }
    for (r, tr) in flights.body_impl_traits {
        if impl_ref_ready(cx, r) {
            lk.impl_traits.insert(r, tr);
        }
    }
    for (key, val) in flights.body_selections {
        if val.is_none_or(|(r, _)| impl_ref_ready(cx, r)) {
            lk.selections.insert(key, val);
        }
    }
    let ready_checked: Vec<(ImplRef, bool)> = flights
        .checked_impls
        .into_iter()
        .filter(|&(block, _)| impl_ref_ready(cx, block))
        .collect();
    let (r_n, c_n, s_n) = (
        flights.reveals.len(),
        ready_checked.len(),
        flights.summaries.len(),
    );
    populate_flight_chunks(cx, flights.reveals, ready_checked, flights.summaries);
    if let (Some(start), Some(owners_us)) = (t0, t_owners) {
        let flights_us = start.elapsed().as_micros() as u64 - owners_us;
        let tm = &flights.timings;
        let defs_end = WARM_NON_SHARD_SECTIONS + WARM_DEF_SHARDS;
        let hdrs_end = defs_end + WARM_HDR_SHARDS;
        let sums_end = hdrs_end + WARM_SUM_SHARDS;
        let sum_range = |from: usize, to: usize| -> u64 {
            tm.section_bytes[from..to]
                .iter()
                .map(|&b| u64::from(b))
                .sum()
        };
        let tdefs_bytes = sum_range(WARM_NON_SHARD_SECTIONS, defs_end);
        let hdrs_bytes = sum_range(defs_end, hdrs_end);
        let sums_bytes = sum_range(hdrs_end, sums_end);
        let decode_us = tm.prelude_us + tm.sections_us;
        eprintln!(
            "[warm] decode_ms={:.2} (prelude={:.2} sections={:.2}) owners_ms={:.2} flights_ms={:.2} kb={:.1} items={} tys={} syms={} flights=({},{},{}) sec_bytes=[defs={},tdefs={},heads={},sigs={},adts={},imeta={},hdrs={},lay={},sel={},misc={},sums={}]",
            decode_us as f64 / 1000.0,
            tm.prelude_us as f64 / 1000.0,
            tm.sections_us as f64 / 1000.0,
            owners_us as f64 / 1000.0,
            flights_us as f64 / 1000.0,
            tm.payload_bytes as f64 / 1024.0,
            tm.item_count,
            tm.ty_count,
            tm.sym_count,
            r_n,
            c_n,
            s_n,
            tm.section_bytes[0],
            tdefs_bytes,
            tm.section_bytes[1],
            tm.section_bytes[2],
            tm.section_bytes[3],
            tm.section_bytes[4],
            hdrs_bytes,
            tm.section_bytes[5],
            tm.section_bytes[6],
            tm.section_bytes[7],
            sums_bytes,
        );
    }
}

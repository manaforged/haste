use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use rustc_hash::FxHashSet;

use crate::names::path::Requests;
use crate::sema::Context;
use crate::sema::infer::Typed;
use crate::sema::substs::Substs;
use crate::sema::ty::{Kind, Ty, Types};

use super::filter::collect_filtered_callees;
use super::wave;
use super::{
    Instance, MIN_PARALLEL_WAVE, PANIC_BOUNDS_CHECK, Reached, ReachedEntry, infer_entry,
    unwind_enabled,
};

pub(in crate::sema) struct WaveItem {
    pub(in crate::sema) pos: usize,
    pub(in crate::sema) typed: Result<Arc<Typed>, String>,
    pub(in crate::sema) unlaid: Vec<String>,
    pub(in crate::sema) deferred: bool,
    pub(in crate::sema) asked: bool,
}

pub(in crate::sema) type WorkerOut = (Requests, Vec<(Instance, u32)>, Vec<WaveItem>);

fn enqueue_bounds_check(cx: &mut Context<'_>, reached: &mut Reached) -> Vec<usize> {
    if !unwind_enabled(cx, reached) {
        return Vec::new();
    }
    let Some(bounds) = cx.lang_item(PANIC_BOUNDS_CHECK) else {
        return Vec::new();
    };
    enqueue(reached, vec![(bounds, Substs::EMPTY)])
}

/// Advances `reached` monotonically from `roots` and any previously blocked instances, parking instances that emit new asks unless `force_commit` is set.
pub fn reach_wave(
    cx: &mut Context<'_>,
    reached: &mut Reached,
    roots: &[Instance],
    force_commit: bool,
) {
    let mut initial = reached.take_blocked();
    initial.extend(enqueue(reached, roots.to_vec()));
    initial.extend(enqueue_bounds_check(cx, reached));
    reached.pin_roots(roots);
    let mut settled = drain_wave(cx, reached, initial, force_commit);
    let extra = enqueue_bounds_check(cx, reached);
    if !extra.is_empty() {
        settled.extend(drain_wave(cx, reached, extra, force_commit));
    }
    super::crate_check::check_local_crate_items(cx, reached, roots, force_commit);
    reached.link(&settled);
}

/// Infers again the reached functions `stale` that an edit did not replace, whose source changed, and every function they now call that was not reached yet, and returns the indexes of both.
pub fn reach_again(cx: &mut Context<'_>, reached: &mut Reached, stale: &[usize]) -> Vec<usize> {
    let stale: Vec<usize> = stale
        .iter()
        .copied()
        .filter(|index| reached.index.get(&reached.list[*index].0) == Some(index))
        .collect();
    let waves = stale_waves(&cx.types, reached, &stale);
    let mut queue = Vec::new();
    for wave in &waves {
        let (_, next) = run_wave(cx, reached, wave, true, true);
        queue.extend(next);
    }
    let mut found = stale;
    found.extend(drain(cx, reached, queue));
    let extra = enqueue_bounds_check(cx, reached);
    if !extra.is_empty() {
        found.extend(drain_wave(cx, reached, extra, true));
    }
    reached.link(&found);
    found
}

fn stale_waves(types: &Types, reached: &Reached, stale: &[usize]) -> Vec<Vec<usize>> {
    let mut left: FxHashSet<usize> = stale.iter().copied().collect();
    let mut waves = Vec::new();
    while !left.is_empty() {
        let ready: Vec<usize> = stale
            .iter()
            .copied()
            .filter(|at| left.contains(at) && !waits_on_stale(types, reached, *at, &left))
            .collect();
        let wave = if ready.is_empty() {
            stale
                .iter()
                .copied()
                .filter(|at| left.contains(at))
                .collect()
        } else {
            ready
        };
        for &at in &wave {
            left.remove(&at);
        }
        waves.push(wave);
    }
    waves
}

fn waits_on_stale(types: &Types, reached: &Reached, at: usize, left: &FxHashSet<usize>) -> bool {
    let (item, substs) = reached.list[at].0;
    if let Some(root) = item.declaring_body()
        && left
            .iter()
            .any(|&other| other != at && reached.list[other].0.0 == root)
    {
        return true;
    }
    let mut owners = Vec::new();
    for &ty in types.substs(substs) {
        closure_owners(types, ty, &mut owners);
    }
    owners.iter().any(|owner| {
        reached
            .index
            .get(owner)
            .is_some_and(|dep| *dep != at && left.contains(dep))
    })
}

fn closure_owners(types: &Types, ty: Ty, out: &mut Vec<Instance>) {
    match types.kind(ty) {
        Kind::Closure(parent, args, _) => {
            for &arg in types.substs(*args) {
                closure_owners(types, arg, out);
            }
            out.push((*parent, *args));
        }
        kind => {
            for child in kind.children(types) {
                closure_owners(types, child, out);
            }
        }
    }
}

/// Drops `dead` and re-reaches from `roots`, optionally limiting crate-level ADT/const/impl checks to `only_crates`.
pub fn reach_anew_in(
    cx: &mut Context<'_>,
    reached: &mut Reached,
    dead: &[usize],
    roots: &[Instance],
    only_crates: Option<&[u32]>,
) -> Vec<usize> {
    reached.filter.forget_crates(only_crates);
    reached.replace_dead(dead);
    let mut found = drain(cx, reached, roots.to_vec());
    let extra = enqueue_bounds_check(cx, reached);
    if !extra.is_empty() {
        found.extend(drain_wave(cx, reached, extra, true));
    }
    super::crate_check::check_local_crate_items_in(cx, reached, roots, true, only_crates);
    reached.trim_replaced();
    reached.pin_roots(roots);
    reached.link(&found);
    found
}

fn drain(cx: &mut Context<'_>, reached: &mut Reached, queue: Vec<Instance>) -> Vec<usize> {
    let wave = enqueue(reached, queue);
    drain_wave(cx, reached, wave, true)
}

fn drain_wave(
    cx: &mut Context<'_>,
    reached: &mut Reached,
    mut wave: Vec<usize>,
    force_commit: bool,
) -> Vec<usize> {
    let mut added = Vec::new();
    while !wave.is_empty() {
        if crate::jobs() > 1 && wave.len() >= MIN_PARALLEL_WAVE {
            added.extend(wave::parallel_drain(cx, reached, wave, force_commit));
            break;
        }
        let (settled, next) = serial_wave(cx, reached, &wave, false, force_commit);
        added.extend_from_slice(&settled);
        wave = enqueue(reached, next);
    }
    added
}

fn enqueue(reached: &mut Reached, candidates: Vec<Instance>) -> Vec<usize> {
    let mut wave = Vec::new();
    for instance in candidates {
        if let std::collections::hash_map::Entry::Vacant(slot) = reached.index.entry(instance) {
            let at = if let Some(free) = reached.free.pop() {
                reached.list[free] = (instance, Err(String::new()));
                reached.states[free] = ReachedEntry::Enqueued;
                free
            } else {
                let end = reached.list.len();
                reached.list.push((instance, Err(String::new())));
                reached.states.push(ReachedEntry::Enqueued);
                end
            };
            slot.insert(at);
            wave.push(at);
        }
    }
    wave
}

fn run_wave(
    cx: &mut Context<'_>,
    reached: &mut Reached,
    wave: &[usize],
    check_undeclared: bool,
    force_commit: bool,
) -> (Vec<usize>, Vec<Instance>) {
    let workers = wave::wave_workers(wave.len());
    if workers <= 1 {
        return serial_wave(cx, reached, wave, check_undeclared, force_commit);
    }
    parallel_wave(cx, reached, wave, workers, (check_undeclared, force_commit))
}

fn serial_wave(
    cx: &mut Context<'_>,
    reached: &mut Reached,
    wave: &[usize],
    check_undeclared: bool,
    force_commit: bool,
) -> (Vec<usize>, Vec<Instance>) {
    let mut settled = Vec::with_capacity(wave.len());
    let mut next = Vec::new();
    for &at in wave {
        let instance = reached.list[at].0;
        let asks0 = cx.asks.count();
        let mut unlaid = Vec::new();
        let typed = infer_entry(cx, &instance, &mut unlaid, check_undeclared);
        if !force_commit && cx.asks.count() > asks0 {
            reached.set_entry(at, ReachedEntry::Blocked, typed);
            continue;
        }
        reached.unlaid.extend(unlaid);
        collect_filtered_callees(cx, &mut reached.filter, instance, &typed, &mut next);
        reached.set_entry(at, ReachedEntry::Settled, typed);
        settled.push(at);
    }
    (settled, next)
}

fn parallel_wave(
    cx: &mut Context<'_>,
    reached: &mut Reached,
    wave: &[usize],
    workers: usize,
    (check_undeclared, force_commit): (bool, bool),
) -> (Vec<usize>, Vec<Instance>) {
    let forks: Vec<Context<'_>> = (0..workers - 1).map(|_| cx.fork()).collect();
    let next = AtomicUsize::new(0);
    cx.in_parallel = true;
    let (mut items, joined) = run_scoped(cx, forks, reached, (wave, &next, check_undeclared));
    cx.in_parallel = false;
    let mut deferred = cx.take_deferred_evals();
    for (asks, worker_deferred, worker_items) in joined.into_iter().flatten() {
        cx.asks.repeat(&asks);
        deferred.extend(worker_deferred);
        items.extend(worker_items);
    }
    items.sort_by_key(|item| item.pos);
    if !deferred.is_empty() || items.iter().any(|item| item.deferred) {
        run_deferred(
            cx,
            (reached, wave, workers, check_undeclared),
            deferred,
            &mut items,
        );
    }
    commit_wave(cx, reached, wave, items, force_commit)
}

fn run_deferred(
    cx: &mut Context<'_>,
    (reached, wave, workers, check_undeclared): (&Reached, &[usize], usize, bool),
    deferred: Vec<(Instance, u32)>,
    items: &mut [WaveItem],
) {
    wave::eval_deferred_list(cx, deferred);
    for _ in 0..4 {
        let pending: Vec<usize> = items
            .iter()
            .enumerate()
            .filter(|(_, item)| item.deferred)
            .map(|(i, _)| i)
            .collect();
        if pending.len() < MIN_PARALLEL_WAVE || workers <= 1 {
            break;
        }
        let sub_wave: Vec<usize> = pending.iter().map(|&i| wave[items[i].pos]).collect();
        let forks: Vec<Context<'_>> = (0..workers - 1).map(|_| cx.fork()).collect();
        let next = AtomicUsize::new(0);
        cx.in_parallel = true;
        let (mut sub_items, joined) =
            run_scoped(cx, forks, reached, (&sub_wave, &next, check_undeclared));
        cx.in_parallel = false;
        let mut next_deferred = cx.take_deferred_evals();
        for (asks, worker_deferred, worker_items) in joined.into_iter().flatten() {
            cx.asks.repeat(&asks);
            next_deferred.extend(worker_deferred);
            sub_items.extend(worker_items);
        }
        for sub_item in sub_items {
            let slot = &mut items[pending[sub_item.pos]];
            *slot = WaveItem {
                pos: slot.pos,
                asked: slot.asked | sub_item.asked,
                ..sub_item
            };
        }
        if next_deferred.is_empty() {
            break;
        }
        wave::eval_deferred_list(cx, next_deferred);
    }
    for item in items.iter_mut().filter(|item| item.deferred) {
        let asks0 = cx.asks.count();
        let mut unlaid = Vec::new();
        let typed = infer_entry(
            cx,
            &reached.list[wave[item.pos]].0,
            &mut unlaid,
            check_undeclared,
        );
        item.asked |= cx.asks.count() > asks0;
        item.typed = typed;
        item.unlaid = unlaid;
        item.deferred = false;
    }
}

pub(in crate::sema) fn run_scoped<'a>(
    cx: &mut Context<'a>,
    forks: Vec<Context<'a>>,
    reached: &Reached,
    (wave, next, check_undeclared): (&[usize], &AtomicUsize, bool),
) -> (Vec<WaveItem>, Vec<Option<WorkerOut>>) {
    crate::sync::scope(|scope| {
        let handles: Vec<_> = forks
            .into_iter()
            .map(|mut fork| {
                scope.spawn(move || {
                    let items = worker_loop(&mut fork, reached, wave, next, check_undeclared);
                    let deferred = fork.take_deferred_evals();
                    let (_, asks) = fork.release();
                    (asks, deferred, items)
                })
            })
            .collect();
        let own = worker_loop(cx, reached, wave, next, check_undeclared);
        let joined = handles
            .into_iter()
            .map(|handle| handle.join().map_or(None, Some))
            .collect();
        (own, joined)
    })
}

fn worker_loop(
    cx: &mut Context<'_>,
    reached: &Reached,
    wave: &[usize],
    next: &AtomicUsize,
    check_undeclared: bool,
) -> Vec<WaveItem> {
    let batch = wave::wave_batch(wave.len());
    let mut items = Vec::new();
    loop {
        let start = next.fetch_add(batch, Ordering::Relaxed);
        if start >= wave.len() {
            return items;
        }
        let end = (start + batch).min(wave.len());
        for (offset, &at) in wave[start..end].iter().enumerate() {
            let asks0 = cx.asks.count();
            let mut unlaid = Vec::new();
            let before = cx.deferred_evals.len();
            let typed = infer_entry(cx, &reached.list[at].0, &mut unlaid, check_undeclared);
            let deferred = cx.deferred_evals.len() > before;
            let asked = cx.asks.count() > asks0;
            items.push(WaveItem {
                pos: start + offset,
                typed,
                unlaid,
                deferred,
                asked,
            });
        }
    }
}

fn commit_wave(
    cx: &mut Context<'_>,
    reached: &mut Reached,
    wave: &[usize],
    items: Vec<WaveItem>,
    force_commit: bool,
) -> (Vec<usize>, Vec<Instance>) {
    let mut slots: Vec<Option<WaveItem>> = (0..wave.len()).map(|_| None).collect();
    for item in items {
        let pos = item.pos;
        slots[pos] = Some(item);
    }
    let mut settled = Vec::with_capacity(wave.len());
    let mut next = Vec::new();
    for (pos, slot) in slots.into_iter().enumerate() {
        let at = wave[pos];
        match slot {
            Some(item) if !force_commit && item.asked => {
                reached.set_entry(at, ReachedEntry::Blocked, item.typed);
            }
            Some(item) => {
                let caller = reached.list[at].0;
                collect_filtered_callees(cx, &mut reached.filter, caller, &item.typed, &mut next);
                reached.set_entry(at, ReachedEntry::Settled, item.typed);
                reached.unlaid.extend(item.unlaid);
                settled.push(at);
            }
            None => {
                let name = cx.name(reached.list[at].0.0);
                reached.set_entry(
                    at,
                    ReachedEntry::Settled,
                    Err(format!("{name}: sema worker panicked")),
                );
                settled.push(at);
            }
        }
    }
    (settled, next)
}

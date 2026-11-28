//! How a wave of inference splits across cores, and the inference of a list of declared functions ahead of their first call.

use std::collections::VecDeque;
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Condvar, Mutex};

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::sema::Context;
use crate::sema::infer::{LazyTables, Typed};

use super::filter::{
    ReachFilter, collect_filtered_callees, commit_filter_caller, pre_collect_callees,
};
use super::{
    Instance, MAX_WORKER_BATCH, MIN_PARALLEL_WAVE, Reached, ReachedEntry, infer_entry, run_scoped,
};

/// How many functions of a wave of `len` one worker takes at a time.
pub(super) fn wave_batch(len: usize) -> usize {
    (len / crate::jobs().max(1)).clamp(1, MAX_WORKER_BATCH)
}

/// How many workers infer a wave of `len` functions: one when the wave is too small to split.
pub(super) fn wave_workers(len: usize) -> usize {
    if len < MIN_PARALLEL_WAVE {
        return 1;
    }
    crate::jobs().max(1).min(len.div_ceil(wave_batch(len)))
}

/// Infers on every core the functions of `indices` that are declared without a body, keeps each body whose inference asked for nothing and deferred no constant evaluation, and returns the indexes it kept; the others stay declared, so `infer_declared` infers them on their first call.
pub fn infer_quiet(cx: &Context<'_>, reached: &mut Reached, indices: &[usize]) -> Vec<usize> {
    let wave: Vec<usize> = indices
        .iter()
        .copied()
        .filter(|&at| reached.is_untyped(at))
        .collect();
    if wave.is_empty() {
        return Vec::new();
    }
    let mut own = cx.fork();
    let forks = (1..wave_workers(wave.len())).map(|_| cx.fork()).collect();
    let next = AtomicUsize::new(0);
    let (mut items, joined) = run_scoped(&mut own, forks, reached, (&wave, &next, false));
    items.extend(
        joined
            .into_iter()
            .flatten()
            .flat_map(|(_, _, worker_items)| worker_items),
    );
    items.sort_by_key(|item| item.pos);
    let mut kept = Vec::new();
    for item in items
        .into_iter()
        .filter(|item| !item.asked && !item.deferred)
    {
        let at = wave[item.pos];
        reached.unlaid.extend(item.unlaid);
        reached.set_entry(at, ReachedEntry::Settled, item.typed);
        kept.push(at);
    }
    kept
}

type WorkerResult = (
    usize,
    Instance,
    Result<Arc<Typed>, String>,
    Vec<String>,
    bool,
    bool,
);

struct WaveQueue {
    seen: HashMap<Instance, usize>,
    filter: ReachFilter,
    instances: Vec<Instance>,
    free: Vec<usize>,
    recycled: Vec<usize>,
    queue: VecDeque<(usize, Instance)>,
    settled: Vec<(usize, Result<Arc<Typed>, String>, Vec<String>, bool)>,
    deferred_items: Vec<(usize, Instance)>,
    active: usize,
    waiting: usize,
}

impl WaveQueue {
    fn push_callee(&mut self, callee: Instance) {
        if let std::collections::hash_map::Entry::Vacant(slot) = self.seen.entry(callee) {
            let at = if let Some(free) = self.free.pop() {
                self.instances[free] = callee;
                self.recycled.push(free);
                free
            } else {
                let end = self.instances.len();
                self.instances.push(callee);
                end
            };
            slot.insert(at);
            self.queue.push_back((at, callee));
        }
    }
}

struct ActiveGuard<'a>(&'a Mutex<WaveQueue>, &'a Condvar, usize);

impl Drop for ActiveGuard<'_> {
    fn drop(&mut self) {
        if self.2 > 0 {
            let mut q = crate::sync::lock(self.0);
            q.active = q.active.saturating_sub(self.2);
            if q.waiting > 0 {
                self.1.notify_all();
            }
        }
    }
}

pub(super) fn eval_deferred_list(cx: &mut Context<'_>, deferred: Vec<(Instance, u32)>) {
    let mut seen = HashSet::default();
    for (inst, bits) in deferred {
        if seen.insert((inst, bits)) {
            crate::sema::evaluate::prime(cx, &inst, bits);
        }
    }
}

pub(super) fn parallel_drain(
    cx: &mut Context<'_>,
    reached: &mut Reached,
    wave: Vec<usize>,
    force_commit: bool,
) -> Vec<usize> {
    let workers = crate::jobs();
    let seen = std::mem::take(&mut reached.index);
    let filter = std::mem::take(&mut reached.filter);
    let free = std::mem::take(&mut reached.free);
    let instances: Vec<Instance> = reached.list.iter().map(|&(inst, _)| inst).collect();
    let queue: VecDeque<(usize, Instance)> = wave.iter().map(|&at| (at, instances[at])).collect();
    let shared = (
        Mutex::new(WaveQueue {
            seen,
            filter,
            instances,
            free,
            recycled: Vec::new(),
            queue,
            settled: Vec::new(),
            deferred_items: Vec::new(),
            active: 0,
            waiting: 0,
        }),
        Condvar::new(),
    );
    loop {
        let seed = {
            let q = crate::sync::lock(&shared.0);
            (q.filter.seed_clone(), q.filter.log_lens())
        };
        let forks: Vec<Context<'_>> = (0..workers - 1).map(|_| cx.fork()).collect();
        let shared_ref = &shared;
        let seed_ref = &seed;
        cx.in_parallel = true;
        let joined = crate::sync::scope(|scope| {
            let handles: Vec<_> = forks
                .into_iter()
                .map(|mut fork| {
                    scope.spawn(move || {
                        wavefront_worker(&mut fork, shared_ref, seed_ref, workers, force_commit);
                        let deferred = fork.take_deferred_evals();
                        let (_, asks) = fork.release();
                        (asks, deferred)
                    })
                })
                .collect();
            wavefront_worker(cx, shared_ref, seed_ref, workers, force_commit);
            handles
                .into_iter()
                .map(|handle| handle.join().map_or(None, Some))
                .collect::<Vec<_>>()
        });
        cx.in_parallel = false;
        let mut deferred = cx.take_deferred_evals();
        for (asks, worker_deferred) in joined.into_iter().flatten() {
            cx.asks.repeat(&asks);
            deferred.extend(worker_deferred);
        }
        if !deferred.is_empty() {
            eval_deferred_list(cx, deferred);
        }
        let deferred_items = {
            let mut q = crate::sync::lock(&shared.0);
            std::mem::take(&mut q.deferred_items)
        };
        if deferred_items.is_empty() {
            break;
        }
        if deferred_items.len() < MIN_PARALLEL_WAVE {
            let mut q = crate::sync::lock(&shared.0);
            q.queue.extend(deferred_items);
            if drain_serial_tail(cx, &mut q, force_commit) {
                break;
            }
        } else {
            let mut q = crate::sync::lock(&shared.0);
            q.queue.extend(deferred_items);
        }
    }
    let q = crate::sync::into_inner(shared.0);
    reached.index = q.seen;
    reached.filter = q.filter;
    reached.free = q.free;
    for at in q.recycled {
        reached.list[at] = (q.instances[at], Err(String::new()));
        reached.states[at] = ReachedEntry::Enqueued;
    }
    let extra = q.instances.len().saturating_sub(reached.list.len());
    reached.list.reserve(extra);
    reached.states.reserve(extra);
    for inst in q.instances.into_iter().skip(reached.list.len()) {
        reached.list.push((inst, Err(String::new())));
        reached.states.push(ReachedEntry::Enqueued);
    }
    let mut added = Vec::with_capacity(q.settled.len());
    for (at, typed, unlaid, blocked) in q.settled {
        if blocked {
            reached.set_entry(at, ReachedEntry::Blocked, typed);
        } else {
            reached.set_entry(at, ReachedEntry::Settled, typed);
            reached.unlaid.extend(unlaid);
            added.push(at);
        }
    }
    added
}

fn drain_serial_tail(cx: &mut Context<'_>, q: &mut WaveQueue, force_commit: bool) -> bool {
    let mut callees = Vec::new();
    while !q.queue.is_empty() && q.queue.len() < MIN_PARALLEL_WAVE {
        let (at, instance) = q.queue.pop_front().expect("non-empty queue");
        let asks0 = cx.asks.count();
        let mut unlaid = Vec::new();
        let typed = infer_entry(cx, &instance, &mut unlaid, false);
        let blocked = !force_commit && cx.asks.count() > asks0;
        callees.clear();
        if !blocked {
            collect_filtered_callees(cx, &mut q.filter, instance, &typed, &mut callees);
        }
        q.settled.push((at, typed, unlaid, blocked));
        for callee in callees.drain(..) {
            q.push_callee(callee);
        }
    }
    q.queue.is_empty()
}

fn wavefront_worker(
    cx: &mut Context<'_>,
    (mutex, cvar): &(Mutex<WaveQueue>, Condvar),
    (seed_filter, seed_lens): &(ReachFilter, (usize, usize)),
    workers: usize,
    force_commit: bool,
) {
    let mut batch = Vec::with_capacity(MAX_WORKER_BATCH);
    let mut results = Vec::with_capacity(MAX_WORKER_BATCH);
    let mut item_callees = Vec::new();
    let mut pending_callees = Vec::new();
    let mut local_seen = HashSet::default();
    let mut batch_sealed = HashSet::default();
    let mut local_filter = seed_filter.seed_clone();
    let mut synced_len = *seed_lens;
    let sync_filter = cx.store.lookups.skip_warm_sysroot_bodies;
    let reach_memo = cx.store.lookups.reach_memo.clone();
    let mut memo_scratch = super::memo::MemoLookupScratch::default();
    let mut active = ActiveGuard(mutex, cvar, 0);
    loop {
        let delta = {
            let mut q = crate::sync::lock(mutex);
            if active.2 > 0 {
                q.active = q.active.saturating_sub(active.2);
                active.2 = 0;
                commit_worker_batch(
                    cx,
                    &mut q,
                    &mut local_filter,
                    (&mut results, &mut pending_callees),
                    force_commit,
                );
            }
            while q.queue.is_empty() {
                if q.active == 0 {
                    if q.waiting > 0 {
                        cvar.notify_all();
                    }
                    return;
                }
                q.waiting += 1;
                q = cvar
                    .wait(q)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                q.waiting -= 1;
            }
            let delta = sync_filter.then(|| q.filter.take_delta(&mut synced_len));
            batch.clear();
            let take = (q.queue.len() / (workers * 2).max(1)).clamp(1, MAX_WORKER_BATCH);
            for _ in 0..take {
                let Some((at, instance)) = q.queue.pop_front() else {
                    break;
                };
                let include_fixed = !q.filter.seen_fixed_poly.contains(&instance.0);
                batch.push((at, instance, include_fixed));
            }
            q.active += batch.len();
            active.2 = batch.len();
            let remaining = q.queue.len();
            if remaining > 0 && q.waiting > 0 {
                if remaining >= q.waiting {
                    cvar.notify_all();
                } else {
                    for _ in 0..remaining {
                        cvar.notify_one();
                    }
                }
            }
            delta
        };
        if let Some((sealed, seen)) = delta {
            local_filter.apply_delta(&sealed, &seen);
        }
        results.clear();
        pending_callees.clear();
        batch_sealed.clear();
        for &(at, instance, include_fixed) in &batch {
            let asks0 = cx.asks.count();
            let mut unlaid = Vec::new();
            let before = cx.deferred_evals.len();
            let typed = if let Some(hit) = reach_memo
                .as_deref()
                .and_then(|m| m.lookup(cx, instance, &mut memo_scratch))
            {
                Ok(hit)
            } else {
                infer_entry(cx, &instance, &mut unlaid, false)
            };
            let deferred = cx.deferred_evals.len() > before;
            let asked = cx.asks.count() > asks0;
            if !deferred && (force_commit || !asked) {
                let first_fixed =
                    include_fixed && !local_filter.seen_fixed_poly.contains(&instance.0);
                item_callees.clear();
                pre_collect_callees(
                    cx,
                    &mut local_filter,
                    (instance, &typed, first_fixed),
                    &mut item_callees,
                );
                for (callee, sealed) in item_callees.drain(..) {
                    if sealed {
                        if !local_filter.seen_sealed_sysroot.contains(&callee.0)
                            && batch_sealed.insert(callee.0)
                        {
                            pending_callees.push((callee, true));
                        }
                    } else if local_seen.insert(callee) {
                        pending_callees.push((callee, false));
                    }
                }
                if let Ok(t) = &typed
                    && matches!(t.tables, LazyTables::Poly(_) | LazyTables::CheckOnly(_))
                {
                    local_filter.seen_fixed_poly.insert(instance.0);
                }
            }
            results.push((at, instance, typed, unlaid, asked, deferred));
        }
    }
}

fn commit_worker_batch(
    cx: &Context<'_>,
    q: &mut WaveQueue,
    local_filter: &mut ReachFilter,
    (results, pending_callees): (&mut Vec<WorkerResult>, &mut Vec<(Instance, bool)>),
    force_commit: bool,
) {
    local_filter.drain_to(&mut q.filter);
    for (at, instance, typed, unlaid, asked, deferred) in results.drain(..) {
        if deferred {
            q.deferred_items.push((at, instance));
        } else if !force_commit && asked {
            q.settled.push((at, typed, unlaid, true));
        } else {
            commit_filter_caller(cx, &mut q.filter, instance, &typed);
            q.settled.push((at, typed, unlaid, false));
        }
    }
    for (callee, sealed) in pending_callees.drain(..) {
        if sealed && !q.filter.mark_seen_sealed(callee.0) {
            continue;
        }
        q.push_callee(callee);
    }
}

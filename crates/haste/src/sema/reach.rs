use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::names::ItemRef;
use crate::names::path::Requests;

use super::Context;
use super::infer::{Res, Typed, mark_parent_laid, record_parent_drops, register};
use super::substs::Substs;
use super::summary::thaw;
pub(in crate::sema) mod filter;
mod order;
mod unwind;
mod wave;

pub(crate) use unwind::host_symbol_may_unwind;
pub use wave::infer_quiet;

use filter::{ReachFilter, collect_filtered_callees};
use super::ty::{Kind, Ty};

pub(super) const MIN_PARALLEL_WAVE: usize = 4;
pub(super) const MAX_WORKER_BATCH: usize = 128;

/// One function at concrete type arguments.
pub type Instance = (ItemRef, Substs);

const _: () = assert!(std::mem::size_of::<Instance>() == 16);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ReachedEntry {
    Declared,
    Enqueued,
    Blocked,
    Settled,
    Replaced,
}

const CATCH_UNWIND_FOUND: usize = 1usize << (usize::BITS - 1);
const CRATE_STD: u8 = 1;
const CRATE_SKIP_BODY: u8 = 2;

#[derive(Default)]
struct CatchUnwindCache {
    checked_items: HashSet<ItemRef>,
    checked_callees: HashSet<ItemRef>,
}

/// Every function reached from some roots, each with its inferred body or the reason it has none.
#[derive(Default)]
pub struct Reached {
    pub list: Vec<(Instance, Result<Arc<Typed>, String>)>,
    pub index: HashMap<Instance, usize>,
    /// The bodies whose types have no layout yet, which lowering would fail on.
    pub unlaid: Vec<String>,
    calls: Vec<Vec<usize>>,
    indirect: Vec<Vec<usize>>,
    call_sites: super::infer::CallSites,
    pinned: Vec<bool>,
    pub(in crate::sema::reach) states: Vec<ReachedEntry>,
    untyped_count: usize,
    pub(in crate::sema::reach) filter: ReachFilter,
    catch_unwind_state: AtomicUsize,
    catch_unwind_cache: Mutex<CatchUnwindCache>,
    may_unwind_cache: RwLock<Option<(usize, Vec<bool>)>>,
    incremental: bool,
}

fn is_thread_spawn_name(name: &str, src: &crate::outline::source::Source) -> bool {
    matches!(name, "spawn" | "spawn_unchecked" | "spawn_scoped" | "scope")
        && src.path.parent().and_then(|p| p.file_name()).and_then(|n| n.to_str()) == Some("thread")
}

fn crate_unwind_flags(cx: &Context) -> Vec<u8> {
    cx.outline
        .roots
        .iter()
        .map(|root| match root.name.as_str() {
            "std" => CRATE_STD | CRATE_SKIP_BODY,
            "core" | "alloc" | "panic_unwind" | "panic_abort" => CRATE_SKIP_BODY,
            _ => 0,
        })
        .collect()
}

impl Reached {
    /// Whether any reached function calls the `catch_unwind` intrinsic or spawns a thread via `std::thread`.
    pub fn has_catch_unwind(&self, cx: &Context) -> bool {
        let state = self.catch_unwind_state.load(Ordering::Acquire);
        if state & CATCH_UNWIND_FOUND != 0 {
            return true;
        }
        let len = self.list.len();
        if state >= len {
            return false;
        }
        let mut cache = crate::sync::lock(&self.catch_unwind_cache);
        let start = self.catch_unwind_state.load(Ordering::Relaxed);
        if start & CATCH_UNWIND_FOUND != 0 {
            return true;
        }
        if start >= len {
            return false;
        }
        let crate_flags = crate_unwind_flags(cx);
        for (idx, ((item, _), res)) in self.list[start..len].iter().enumerate() {
            let flags = crate_flags.get(item.krate as usize).copied().unwrap_or(0);
            if flags & CRATE_SKIP_BODY != 0 {
                if flags & CRATE_STD != 0 && cache.checked_items.insert(*item) {
                    let (src, entry) = cx.item(*item);
                    if entry.name.is_some_and(|at| is_thread_spawn_name(src.text(at), src)) {
                        self.catch_unwind_state.store(CATCH_UNWIND_FOUND | (start + idx + 1), Ordering::Release);
                        return true;
                    }
                }
                continue;
            }
            let Ok(typed) = res else { continue };
            for (callee, _) in typed.calls() {
                if !cache.checked_callees.insert(callee) {
                    continue;
                }
                let (src, entry) = cx.item(callee);
                let is_std = crate_flags.get(callee.krate as usize).copied().unwrap_or(0) & CRATE_STD != 0;
                if entry.name.is_some_and(|at| {
                    let name = src.text(at);
                    name == "catch_unwind" || (is_std && is_thread_spawn_name(name, src))
                }) {
                    self.catch_unwind_state.store(CATCH_UNWIND_FOUND | (start + idx + 1), Ordering::Release);
                    return true;
                }
            }
        }
        self.catch_unwind_state.store(len, Ordering::Release);
        false
    }

    /// Marks this reach graph as incrementally grown (lazy JIT), skipping whole-graph unwind fixed-point scans.
    pub fn set_incremental(&mut self, incremental: bool) {
        self.incremental = incremental;
    }

    /// Whether this reach graph grows incrementally during lazy JIT execution.
    pub fn is_incremental(&self) -> bool {
        self.incremental
    }

    /// Whether reached function `index` may unwind directly or transitively through any reachable call.
    pub fn may_unwind(&self, cx: &mut Context<'_>, index: usize) -> bool {
        if self.incremental {
            return true;
        }
        let len = self.list.len();
        if index >= len {
            return true;
        }
        if let Some(bit) = crate::sync::read_lock(&self.may_unwind_cache).as_ref().filter(|(n, _)| *n == len).and_then(|(_, bits)| bits.get(index).copied()) {
            return bit;
        }
        let mut cache = crate::sync::write_lock(&self.may_unwind_cache);
        if cache.as_ref().is_none_or(|(n, _)| *n != len) {
            *cache = Some((len, unwind::compute_may_unwind(cx, self)));
        }
        cache.as_ref().and_then(|(_, bits)| bits.get(index).copied()).unwrap_or(true)
    }
    /// The typed body of function `index`, or an error naming `name` and why its body did not type-check.
    pub fn typed(&self, index: usize, name: &str) -> anyhow::Result<&Arc<Typed>> {
        self.list[index].1.as_ref().map_err(|why| anyhow::anyhow!("{name} did not type-check: {why}"))
    }

    /// `roots` and every function they reach through the bodies that type-checked, each once and after every function it reaches.
    pub fn closure(&self, roots: &[usize]) -> Vec<usize> {
        let mut seen = vec![false; self.list.len()];
        let (mut stack, mut found) = (roots.iter().rev().map(|&root| (root, false)).collect::<Vec<_>>(), Vec::new());
        while let Some((at, done)) = stack.pop() {
            if done {
                found.push(at);
                continue;
            }
            if std::mem::replace(&mut seen[at], true) {
                continue;
            }
            stack.push((at, true));
            stack.extend(self.calls[at].iter().map(|&callee| (callee, false)));
        }
        found
    }

    fn push_entry(&mut self, instance: Instance, state: ReachedEntry) -> usize {
        let at = self.list.len();
        self.index.insert(instance, at);
        self.list.push((instance, Err(String::new())));
        self.states.push(state);
        if state == ReachedEntry::Declared {
            self.untyped_count += 1;
        }
        at
    }

    pub(super) fn set_entry(&mut self, index: usize, next: ReachedEntry, result: Result<Arc<Typed>, String>) {
        if self.states[index] == ReachedEntry::Declared {
            self.untyped_count -= 1;
        }
        if next == ReachedEntry::Declared {
            self.untyped_count += 1;
        }
        self.states[index] = next;
        self.list[index].1 = result;
        let checked = *self.catch_unwind_state.get_mut() & !CATCH_UNWIND_FOUND;
        if index < checked {
            *self.catch_unwind_state.get_mut() = index;
            let cache = self.catch_unwind_cache.get_mut().unwrap_or_else(std::sync::PoisonError::into_inner);
            cache.checked_items.clear();
            cache.checked_callees.clear();
        }
        *self.may_unwind_cache.get_mut().unwrap_or_else(std::sync::PoisonError::into_inner) = None;
    }

    fn take_blocked(&mut self) -> Vec<usize> {
        self.states
            .iter_mut()
            .enumerate()
            .filter_map(|(at, state)| {
                if *state == ReachedEntry::Blocked {
                    *state = ReachedEntry::Enqueued;
                    Some(at)
                } else {
                    None
                }
            })
            .collect()
    }

    /// The index of `instance`, and whether it is new: a new one is listed without its body until `infer_declared` infers it.
    pub fn declare(&mut self, instance: Instance) -> (usize, bool) {
        if let Some(&at) = self.index.get(&instance) {
            return (at, false);
        }
        let at = self.push_entry(instance, ReachedEntry::Declared);
        (at, true)
    }

    /// Takes back the body of declared function `index`, so the next `infer_declared` infers it again.
    pub fn forget(&mut self, index: usize) {
        self.set_entry(index, ReachedEntry::Declared, Err(String::new()));
    }

    /// Whether declared function `index` has not been inferred yet.
    pub fn is_untyped(&self, index: usize) -> bool {
        self.untyped_count > 0 && self.states.get(index) == Some(&ReachedEntry::Declared)
    }

    /// The prelinked callee indices of reached function `index`, when `link` has populated them.
    pub fn direct_calls(&self, index: usize) -> Option<&[usize]> {
        self.calls.get(index).map(Vec::as_slice)
    }

    /// Whether reached function `index` has exactly one static direct call site and no indirect or root entry.
    pub(crate) fn is_sole_called(&self, index: usize) -> bool {
        self.call_sites.get(index) == Some(1)
    }

    fn pin_roots(&mut self, roots: &[Instance]) {
        self.pinned.resize(self.list.len(), false);
        for inst in roots {
            if let Some(&id) = self.index.get(inst) { self.pinned[id] = true; }
        }
    }

    fn link(&mut self, indices: &[usize]) {
        self.calls.resize(self.list.len(), Vec::new());
        self.indirect.resize(self.list.len(), Vec::new());
        self.call_sites.retract(indices, &self.calls, &self.indirect);
        let workers = crate::jobs().min(indices.len() / 256);
        let edges = |at: usize| self.list[at].1.as_ref().map(|t| t.collect_link_edges(&self.index)).unwrap_or_default();
        if workers <= 1 {
            for &at in indices { (self.calls[at], self.indirect[at]) = edges(at); }
        } else {
            let next = AtomicUsize::new(0);
            let worker = || {
                let mut local = Vec::new();
                while let Some(slice) = indices.get(next.fetch_add(256, Ordering::Relaxed)..) {
                    for &at in &slice[..slice.len().min(256)] { local.push((at, edges(at))); }
                }
                local
            };
            let (own, joined) = crate::sync::scope(|scope| {
                let handles: Vec<_> = (1..workers).map(|_| scope.spawn(worker)).collect();
                (worker(), handles.into_iter().flat_map(|h| h.join().unwrap_or_default()).collect::<Vec<_>>())
            });
            for (at, (c, i)) in own.into_iter().chain(joined) { (self.calls[at], self.indirect[at]) = (c, i); }
        }
        super::infer::recount_call_sites(self.list.len(), |at| self.states[at] == ReachedEntry::Settled, &self.calls, &self.indirect, &self.pinned, &mut self.call_sites);
        *self.may_unwind_cache.get_mut().unwrap_or_else(std::sync::PoisonError::into_inner) = None;
    }
}

pub(super) fn push_tail_calls(typed: &Typed, push: &mut impl FnMut(Instance)) {
    for r in typed.drops.iter().chain(super::infer::shape_calls(&typed.shape)) {
        if let Res::Fn(item, substs) = *r { push((item, substs)); }
    }
}

pub(crate) const PANIC_BOUNDS_CHECK: &str = "panic_bounds_check";

pub(crate) fn unwind_enabled(cx: &Context<'_>, reached: &Reached) -> bool {
    crate::profile() != crate::outline::tree::Profile::Release || reached.has_catch_unwind(cx)
}

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
pub fn reach_wave(cx: &mut Context<'_>, reached: &mut Reached, roots: &[Instance], force_commit: bool) {
    let mut initial = reached.take_blocked();
    initial.extend(enqueue(reached, roots.to_vec()));
    initial.extend(enqueue_bounds_check(cx, reached));
    reached.pin_roots(roots);
    let mut settled = drain_wave(cx, reached, initial, force_commit);
    let extra = enqueue_bounds_check(cx, reached);
    if !extra.is_empty() {
        settled.extend(drain_wave(cx, reached, extra, force_commit));
    }
    reached.link(&settled);
}

/// Infers again the reached functions `stale` that an edit did not replace, whose source changed, and every function they now call that was not reached yet, and returns the indexes of both.
pub fn reach_again(cx: &mut Context<'_>, reached: &mut Reached, stale: &[usize]) -> Vec<usize> {
    let stale: Vec<usize> = stale.iter().copied().filter(|index| reached.index.get(&reached.list[*index].0) == Some(index)).collect();
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

fn stale_waves(types: &super::ty::Types, reached: &Reached, stale: &[usize]) -> Vec<Vec<usize>> {
    let mut left: rustc_hash::FxHashSet<usize> = stale.iter().copied().collect();
    let mut waves = Vec::new();
    while !left.is_empty() {
        let ready: Vec<usize> = stale.iter().copied().filter(|at| left.contains(at) && !waits_on_stale(types, reached, *at, &left)).collect();
        let wave = if ready.is_empty() { stale.iter().copied().filter(|at| left.contains(at)).collect() } else { ready };
        for &at in &wave {
            left.remove(&at);
        }
        waves.push(wave);
    }
    waves
}

fn waits_on_stale(types: &super::ty::Types, reached: &Reached, at: usize, left: &rustc_hash::FxHashSet<usize>) -> bool {
    let (item, substs) = reached.list[at].0;
    if let Some(root) = item.declaring_body()
        && left.iter().any(|&other| other != at && reached.list[other].0.0 == root)
    {
        return true;
    }
    let mut owners = Vec::new();
    for &ty in types.substs(substs) {
        closure_owners(types, ty, &mut owners);
    }
    owners.iter().any(|owner| reached.index.get(owner).is_some_and(|dep| *dep != at && left.contains(dep)))
}

fn closure_owners(types: &super::ty::Types, ty: Ty, out: &mut Vec<Instance>) {
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

/// Drops the reached functions `dead`, whose items an edit replaced, then infers `roots` and every function they call that is not reached, and returns the indexes of those.
pub fn reach_anew(cx: &mut Context<'_>, reached: &mut Reached, dead: &[usize], roots: &[Instance]) -> Vec<usize> {
    for &index in dead {
        reached.index.remove(&reached.list[index].0);
        reached.set_entry(index, ReachedEntry::Replaced, Err("replaced by an edit".to_owned()));
    }
    reached.link(dead);
    let mut found = drain(cx, reached, roots.to_vec());
    let extra = enqueue_bounds_check(cx, reached);
    if !extra.is_empty() {
        found.extend(drain_wave(cx, reached, extra, true));
    }
    reached.pin_roots(roots);
    reached.link(&found);
    found
}

/// Infers declared function `index` when it has no body yet, and returns whether it did.
pub fn infer_declared(cx: &mut Context<'_>, reached: &mut Reached, index: usize) -> bool {
    if reached.states.get(index) != Some(&ReachedEntry::Declared) {
        return false;
    }
    let instance = reached.list[index].0;
    let typed = typed_once(cx, &instance, &mut reached.unlaid);
    reached.set_entry(index, ReachedEntry::Settled, typed);
    true
}

/// The functions that the inferred body `typed` calls, other than intrinsics.
pub fn calls<'t>(cx: &'t Context<'_>, typed: &'t Result<Arc<Typed>, String>) -> impl Iterator<Item = Instance> + 't {
    typed.iter().flat_map(|typed| typed.calls()).filter(|(item, _)| cx.intrinsic(*item).is_none())
}

pub(super) struct WaveItem {
    pub(super) pos: usize,
    pub(super) typed: Result<Arc<Typed>, String>,
    pub(super) unlaid: Vec<String>,
    pub(super) deferred: bool,
    pub(super) asked: bool,
}

pub(super) type WorkerOut = (Requests, Vec<(Instance, u32)>, Vec<WaveItem>);

fn drain(cx: &mut Context<'_>, reached: &mut Reached, queue: Vec<Instance>) -> Vec<usize> {
    let wave = enqueue(reached, queue);
    drain_wave(cx, reached, wave, true)
}

fn drain_wave(cx: &mut Context<'_>, reached: &mut Reached, mut wave: Vec<usize>, force_commit: bool) -> Vec<usize> {
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
            let at = reached.list.len();
            slot.insert(at);
            reached.list.push((instance, Err(String::new())));
            reached.states.push(ReachedEntry::Enqueued);
            wave.push(at);
        }
    }
    wave
}

fn run_wave(cx: &mut Context<'_>, reached: &mut Reached, wave: &[usize], check_undeclared: bool, force_commit: bool) -> (Vec<usize>, Vec<Instance>) {
    let workers = wave::wave_workers(wave.len());
    if workers <= 1 {
        return serial_wave(cx, reached, wave, check_undeclared, force_commit);
    }
    parallel_wave(cx, reached, wave, workers, check_undeclared, force_commit)
}

fn serial_wave(cx: &mut Context<'_>, reached: &mut Reached, wave: &[usize], check_undeclared: bool, force_commit: bool) -> (Vec<usize>, Vec<Instance>) {
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

fn parallel_wave(cx: &mut Context<'_>, reached: &mut Reached, wave: &[usize], workers: usize, check_undeclared: bool, force_commit: bool) -> (Vec<usize>, Vec<Instance>) {
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
        run_deferred(cx, (reached, wave, workers, check_undeclared), deferred, &mut items);
    }
    commit_wave(cx, reached, wave, items, force_commit)
}

fn run_deferred(cx: &mut Context<'_>, (reached, wave, workers, check_undeclared): (&Reached, &[usize], usize, bool), deferred: Vec<(Instance, u32)>, items: &mut [WaveItem]) {
    wave::eval_deferred_list(cx, deferred);
    for _ in 0..4 {
        let pending: Vec<usize> = items.iter().enumerate().filter(|(_, item)| item.deferred).map(|(i, _)| i).collect();
        if pending.len() < MIN_PARALLEL_WAVE || workers <= 1 {
            break;
        }
        let sub_wave: Vec<usize> = pending.iter().map(|&i| wave[items[i].pos]).collect();
        let forks: Vec<Context<'_>> = (0..workers - 1).map(|_| cx.fork()).collect();
        let next = AtomicUsize::new(0);
        cx.in_parallel = true;
        let (mut sub_items, joined) = run_scoped(cx, forks, reached, (&sub_wave, &next, check_undeclared));
        cx.in_parallel = false;
        let mut next_deferred = cx.take_deferred_evals();
        for (asks, worker_deferred, worker_items) in joined.into_iter().flatten() {
            cx.asks.repeat(&asks);
            next_deferred.extend(worker_deferred);
            sub_items.extend(worker_items);
        }
        for sub_item in sub_items {
            let slot = &mut items[pending[sub_item.pos]];
            *slot = WaveItem { pos: slot.pos, asked: slot.asked | sub_item.asked, ..sub_item };
        }
        if next_deferred.is_empty() {
            break;
        }
        wave::eval_deferred_list(cx, next_deferred);
    }
    for item in items.iter_mut().filter(|item| item.deferred) {
        let asks0 = cx.asks.count();
        let mut unlaid = Vec::new();
        let typed = infer_entry(cx, &reached.list[wave[item.pos]].0, &mut unlaid, check_undeclared);
        item.asked |= cx.asks.count() > asks0;
        item.typed = typed;
        item.unlaid = unlaid;
        item.deferred = false;
    }
}

pub(super) fn run_scoped<'a>(cx: &mut Context<'a>, forks: Vec<Context<'a>>, reached: &Reached, (wave, next, check_undeclared): (&[usize], &AtomicUsize, bool)) -> (Vec<WaveItem>, Vec<Option<WorkerOut>>) {
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
        let joined = handles.into_iter().map(|handle| handle.join().map_or(None, Some)).collect();
        (own, joined)
    })
}

fn worker_loop(cx: &mut Context<'_>, reached: &Reached, wave: &[usize], next: &AtomicUsize, check_undeclared: bool) -> Vec<WaveItem> {
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
            items.push(WaveItem { pos: start + offset, typed, unlaid, deferred, asked });
        }
    }
}

fn commit_wave(cx: &mut Context<'_>, reached: &mut Reached, wave: &[usize], items: Vec<WaveItem>, force_commit: bool) -> (Vec<usize>, Vec<Instance>) {
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
                reached.set_entry(at, ReachedEntry::Settled, Err(format!("{name}: sema worker panicked")));
                settled.push(at);
            }
        }
    }
    (settled, next)
}

pub(super) fn infer_entry(cx: &mut Context<'_>, instance: &Instance, unlaid: &mut Vec<String>, check_undeclared: bool) -> Result<Arc<Typed>, String> {
    if check_undeclared && cx.undeclared(instance) {
        return Err("no longer declared".to_owned());
    }
    typed_once(cx, instance, unlaid)
}

std::thread_local! {
    static INFERRING: std::cell::RefCell<Vec<Instance>> = const { std::cell::RefCell::new(Vec::new()) };
}

struct InferGuard(bool);

impl InferGuard {
    fn enter(instance: Instance) -> Self {
        INFERRING.with(|active| active.borrow_mut().push(instance));
        Self(true)
    }
}

impl Drop for InferGuard {
    fn drop(&mut self) {
        if self.0 {
            INFERRING.with(|active| {
                active.borrow_mut().pop();
            });
        }
    }
}

pub(super) fn is_inferring(instance: Instance) -> bool {
    INFERRING.with(|active| active.borrow().contains(&instance))
}

/// The inferred body of `instance`: the one an earlier pass kept, or a fresh inference that is kept when it type-checked, laid out, and asked for nothing.
pub(crate) fn typed_once(cx: &mut Context<'_>, instance: &Instance, unlaid: &mut Vec<String>) -> Result<Arc<Typed>, String> {
    let is_alloc_shim = cx.global_alloc_shim(instance.0).is_some();
    if !is_alloc_shim
        && let Some(typed) = cx.store.finals.get(instance).filter(|t| cx.store.lookups.skip_warm_sysroot_bodies || !matches!(&t.tables, super::infer::LazyTables::CheckOnly(_)))
    {
        register(cx, *instance, &typed);
        record_parent_drops(cx, *instance, &typed);
        mark_parent_laid(cx, *instance, &typed);
        return Ok(typed);
    }
    if !is_alloc_shim
        && let Some(result) = thaw::thawed(cx, instance, unlaid)
    {
        return result;
    }
    let typed = typed_fresh(cx, instance, unlaid)?;
    if !is_alloc_shim {
        thaw::note_fresh(cx, *instance, &typed);
    }
    Ok(typed)
}

fn typed_fresh(cx: &mut Context<'_>, instance: &Instance, unlaid: &mut Vec<String>) -> Result<Arc<Typed>, String> {
    let _guard = InferGuard::enter(*instance);
    let before = cx.quiet_mark();
    match super::infer::compiler_made(cx, instance.0, instance.1) {
        Ok(Some(typed)) => {
            let typed = Arc::new(typed);
            thaw::keep_laid(cx, *instance, &typed, unlaid, before);
            return Ok(typed);
        }
        Err(error) => return Err(format!("{}: {error:#}", cx.name(instance.0))),
        Ok(None) => {}
    }
    if let Some(result) = super::summary::typed_once_poly(cx, instance, unlaid) {
        return result;
    }
    let typed = Arc::new(super::infer::infer_after_compiler_made(cx, instance.0, instance.1, None).map_err(|error| format!("{}: {error:#}", cx.name(instance.0)))?);
    thaw::keep_laid(cx, *instance, &typed, unlaid, before);
    Ok(typed)
}

fn lay_out_ty(cx: &mut Context<'_>, ty: Ty) -> anyhow::Result<()> {
    if let Kind::Adt(item, _) = *cx.types.kind(ty)
        && cx.item(item).1.kind == crate::outline::item::ItemKind::Enum
    {
        cx.discriminants(item)?;
    }
    if matches!(cx.types.kind(ty), Kind::Adt(..) | Kind::Tuple(_) | Kind::Array(..) | Kind::Closure(..)) {
        cx.layout(ty)?;
    }
    Ok(())
}

/// Computes the layout of each ADT, tuple and array type that lowering `typed` reads, so the names those layouts need are asked for while inference can still retry.
pub(super) fn lay_out(cx: &mut Context<'_>, typed: &Typed) -> anyhow::Result<()> {
    if let super::infer::Shape::Closure(body) = &typed.shape
        && body.parent_laid
    {
        for ty in typed.locals[body.parent_locals as usize..].iter().map(|local| local.ty).chain(typed.ret) {
            if ty != Ty::NONE {
                lay_out_ty(cx, ty)?;
            }
        }
        return Ok(());
    }
    let scoped = typed.scoped(&cx.types);
    let tables = scoped.ensure_tables(&cx.types);
    let mut intrinsic_args = Vec::new();
    for (item, substs) in typed.calls() {
        if cx.intrinsic(item).is_some() {
            intrinsic_args.extend_from_slice(cx.types.substs(substs));
        }
    }
    let locals = typed.locals.iter().map(|local| local.ty);
    let pats = tables.pats.values().copied();
    let mut seen = rustc_hash::FxHashSet::default();
    for ty in tables.exprs.iter().copied().chain(locals).chain(pats).chain(intrinsic_args) {
        if ty == Ty::NONE || !seen.insert(ty) {
            continue;
        }
        lay_out_ty(cx, ty)?;
    }
    Ok(())
}

/// The two roots a program run calls: std's `start` lang item at the return type of `main`, and `main`.
pub fn main_roots(cx: &mut Context<'_>, main: ItemRef) -> anyhow::Result<[Instance; 2]> {
    let start = cx.lang_item("start").ok_or_else(|| anyhow::anyhow!("no `start` lang item"))?;
    let ret = super::infer::return_type(cx, main)?;
    Ok([(start, cx.types.intern_substs(&[ret])), (main, Substs::EMPTY)])
}

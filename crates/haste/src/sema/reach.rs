use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use rustc_hash::FxHashMap as HashMap;

use crate::names::ItemRef;
use crate::names::path::Requests;

use super::Context;
use super::infer::{Typed, infer, register};
use super::substs::Substs;
use super::ty::{Kind, Ty};

const MIN_PARALLEL_WAVE: usize = 4;
const MAX_WORKER_BATCH: usize = 32;

/// One function at concrete type arguments.
pub type Instance = (ItemRef, Substs);

const _: () = assert!(std::mem::size_of::<Instance>() == 16);

/// Every function reached from some roots, each with its inferred body or the reason it has none.
#[derive(Default)]
pub struct Reached {
    pub list: Vec<(Instance, Result<Arc<Typed>, String>)>,
    pub index: HashMap<Instance, usize>,
    /// The bodies whose types have no layout yet, which lowering would fail on.
    pub unlaid: Vec<String>,
    calls: Vec<Vec<usize>>,
    untyped: rustc_hash::FxHashSet<usize>,
    blocked: Vec<usize>,
}

impl Reached {
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

    /// The index of `instance`, and whether it is new: a new one is listed without its body until `infer_declared` infers it.
    pub fn declare(&mut self, instance: Instance) -> (usize, bool) {
        if let Some(&at) = self.index.get(&instance) {
            return (at, false);
        }
        let at = self.list.len();
        self.index.insert(instance, at);
        self.list.push((instance, Err(String::new())));
        self.untyped.insert(at);
        (at, true)
    }

    /// Takes back the body of declared function `index`, so the next `infer_declared` infers it again.
    pub fn forget(&mut self, index: usize) {
        self.list[index].1 = Err(String::new());
        self.untyped.insert(index);
    }

    /// Whether declared function `index` has not been inferred yet.
    pub fn is_untyped(&self, index: usize) -> bool {
        !self.untyped.is_empty() && self.untyped.contains(&index)
    }

    /// The prelinked callee indices of reached function `index`, when `link` has populated them.
    pub fn direct_calls(&self, index: usize) -> Option<&[usize]> {
        self.calls.get(index).map(Vec::as_slice)
    }

    fn link(&mut self, indices: &[usize]) {
        self.calls.resize(self.list.len(), Vec::new());
        for &at in indices {
            let callees = self.list[at].1.as_ref().map(|typed| self.callees(typed).collect()).unwrap_or_default();
            self.calls[at] = callees;
        }
    }

    fn callees<'r>(&'r self, typed: &'r Typed) -> impl Iterator<Item = usize> + 'r {
        typed.calls().filter_map(|callee| self.index.get(&callee).copied())
    }
}

/// Advances `reached` monotonically from `roots` and any previously blocked instances, parking instances that emit new asks unless `force_commit` is set.
pub fn reach_wave(cx: &mut Context<'_>, reached: &mut Reached, roots: &[Instance], force_commit: bool) {
    let mut initial = std::mem::take(&mut reached.blocked);
    initial.extend(enqueue(reached, roots.to_vec()));
    let settled = drain_wave(cx, reached, initial, force_commit);
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
        reached.list[index].1 = Err("replaced by an edit".to_owned());
    }
    reached.link(dead);
    let found = drain(cx, reached, roots.to_vec());
    reached.link(&found);
    found
}

/// Infers declared function `index` when it has no body yet, and returns whether it did.
pub fn infer_declared(cx: &mut Context<'_>, reached: &mut Reached, index: usize) -> bool {
    if !reached.untyped.remove(&index) {
        return false;
    }
    let instance = reached.list[index].0;
    reached.list[index].1 = typed_once(cx, &instance, &mut reached.unlaid);
    true
}

/// The functions that the inferred body `typed` calls, other than intrinsics.
pub fn calls<'t>(cx: &'t Context<'_>, typed: &'t Result<Arc<Typed>, String>) -> impl Iterator<Item = Instance> + 't {
    typed.iter().flat_map(|typed| typed.calls()).filter(|(item, _)| cx.intrinsic(*item).is_none())
}

struct WaveItem {
    pos: usize,
    typed: Result<Arc<Typed>, String>,
    unlaid: Vec<String>,
    callees: Vec<Instance>,
    deferred: bool,
    asked: bool,
}

type WorkerOut = (Requests, Vec<(Instance, u32)>, Vec<WaveItem>);

fn drain(cx: &mut Context<'_>, reached: &mut Reached, queue: Vec<Instance>) -> Vec<usize> {
    let wave = enqueue(reached, queue);
    drain_wave(cx, reached, wave, true)
}

fn drain_wave(cx: &mut Context<'_>, reached: &mut Reached, mut wave: Vec<usize>, force_commit: bool) -> Vec<usize> {
    let mut added = Vec::new();
    while !wave.is_empty() {
        if crate::jobs() > 1 && wave.len() >= MIN_PARALLEL_WAVE {
            added.extend(parallel_drain(cx, reached, wave, force_commit));
            break;
        }
        let (settled, next) = serial_wave(cx, reached, &wave, false, force_commit);
        added.extend_from_slice(&settled);
        wave = enqueue(reached, next);
    }
    added
}

struct WaveQueue {
    seen: HashMap<Instance, usize>,
    instances: Vec<Instance>,
    queue: VecDeque<(usize, Instance)>,
    settled: Vec<(usize, Result<Arc<Typed>, String>, Vec<String>, bool)>,
    deferred_items: Vec<(usize, Instance)>,
    active: usize,
}

struct ActiveGuard<'a>(&'a Mutex<WaveQueue>, &'a Condvar, usize);

impl Drop for ActiveGuard<'_> {
    fn drop(&mut self) {
        if self.2 > 0 {
            let mut q = crate::sync::lock(self.0);
            q.active = q.active.saturating_sub(self.2);
            self.1.notify_all();
        }
    }
}

fn parallel_drain(cx: &mut Context<'_>, reached: &mut Reached, wave: Vec<usize>, force_commit: bool) -> Vec<usize> {
    let workers = crate::jobs();
    let seen = std::mem::take(&mut reached.index);
    let instances: Vec<Instance> = reached.list.iter().map(|&(inst, _)| inst).collect();
    let queue: VecDeque<(usize, Instance)> = wave.iter().map(|&at| (at, instances[at])).collect();
    let shared = (Mutex::new(WaveQueue { seen, instances, queue, settled: Vec::new(), deferred_items: Vec::new(), active: 0 }), Condvar::new());
    loop {
        let forks: Vec<Context<'_>> = (0..workers - 1).map(|_| cx.fork()).collect();
        let shared_ref = &shared;
        cx.in_parallel = true;
        let joined = std::thread::scope(|scope| {
            let handles: Vec<_> = forks
                .into_iter()
                .map(|mut fork| {
                    scope.spawn(move || {
                        wavefront_worker(&mut fork, shared_ref, workers, force_commit);
                        let deferred = fork.take_deferred_evals();
                        let (_, asks) = fork.release();
                        (asks, deferred)
                    })
                })
                .collect();
            wavefront_worker(cx, shared_ref, workers, force_commit);
            handles.into_iter().map(|handle| handle.join().map_or(None, Some)).collect::<Vec<_>>()
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
            let mut extra = Vec::new();
            for (at, instance) in deferred_items {
                let asks0 = cx.asks.count();
                let mut unlaid = Vec::new();
                let typed = infer_entry(cx, &instance, &mut unlaid, false);
                let asked = cx.asks.count() > asks0;
                let callees: Vec<Instance> = if !force_commit && asked { Vec::new() } else { calls(cx, &typed).collect() };
                extra.push((at, typed, unlaid, !force_commit && asked, callees));
            }
            let mut q = crate::sync::lock(&shared.0);
            for (at, typed, unlaid, blocked, callees) in extra {
                q.settled.push((at, typed, unlaid, blocked));
                for callee in callees {
                    if !q.seen.contains_key(&callee) {
                        let next_at = q.instances.len();
                        q.seen.insert(callee, next_at);
                        q.instances.push(callee);
                        q.queue.push_back((next_at, callee));
                    }
                }
            }
            if q.queue.is_empty() {
                break;
            }
        } else {
            let mut q = crate::sync::lock(&shared.0);
            q.queue.extend(deferred_items);
        }
    }
    let q = shared.0.into_inner().unwrap_or_else(std::sync::PoisonError::into_inner);
    reached.index = q.seen;
    for inst in q.instances.into_iter().skip(reached.list.len()) {
        reached.list.push((inst, Err(String::new())));
    }
    let mut added = Vec::with_capacity(q.settled.len());
    for (at, typed, unlaid, blocked) in q.settled {
        reached.list[at].1 = typed;
        if blocked {
            reached.blocked.push(at);
        } else {
            reached.unlaid.extend(unlaid);
            added.push(at);
        }
    }
    added
}

fn wavefront_worker(cx: &mut Context<'_>, (mutex, cvar): &(Mutex<WaveQueue>, Condvar), workers: usize, force_commit: bool) {
    let mut batch = Vec::with_capacity(MAX_WORKER_BATCH);
    let mut results = Vec::with_capacity(MAX_WORKER_BATCH);
    loop {
        batch.clear();
        {
            let mut q = crate::sync::lock(mutex);
            while q.queue.is_empty() {
                if q.active == 0 {
                    cvar.notify_all();
                    return;
                }
                q = cvar.wait(q).unwrap_or_else(std::sync::PoisonError::into_inner);
            }
            let take = (q.queue.len() / workers.max(1)).clamp(1, MAX_WORKER_BATCH);
            for _ in 0..take {
                let Some(item) = q.queue.pop_front() else { break };
                batch.push(item);
            }
            q.active += batch.len();
        }
        let mut active = ActiveGuard(mutex, cvar, batch.len());
        results.clear();
        for &(at, instance) in &batch {
            let asks0 = cx.asks.count();
            let mut unlaid = Vec::new();
            let before = cx.deferred_evals.len();
            let typed = infer_entry(cx, &instance, &mut unlaid, false);
            let deferred = cx.deferred_evals.len() > before;
            let asked = cx.asks.count() > asks0;
            let callees: Vec<Instance> = if !deferred && (force_commit || !asked) { calls(cx, &typed).collect() } else { Vec::new() };
            results.push((at, instance, typed, unlaid, asked, deferred, callees));
        }
        {
            let mut q = crate::sync::lock(mutex);
            q.active = q.active.saturating_sub(active.2);
            active.2 = 0;
            let mut pushed = false;
            for (at, instance, typed, unlaid, asked, deferred, callees) in results.drain(..) {
                if deferred {
                    q.deferred_items.push((at, instance));
                } else if !force_commit && asked {
                    q.settled.push((at, typed, unlaid, true));
                } else {
                    q.settled.push((at, typed, unlaid, false));
                    for callee in callees {
                        if !q.seen.contains_key(&callee) {
                            let next_at = q.instances.len();
                            q.seen.insert(callee, next_at);
                            q.instances.push(callee);
                            q.queue.push_back((next_at, callee));
                            pushed = true;
                        }
                    }
                }
            }
            if pushed || (q.queue.is_empty() && q.active == 0) {
                cvar.notify_all();
            }
        }
    }
}

fn enqueue(reached: &mut Reached, candidates: Vec<Instance>) -> Vec<usize> {
    let mut wave = Vec::new();
    for instance in candidates {
        if reached.index.contains_key(&instance) {
            continue;
        }
        let at = reached.list.len();
        reached.index.insert(instance, at);
        reached.list.push((instance, Err(String::new())));
        wave.push(at);
    }
    wave
}

fn run_wave(cx: &mut Context<'_>, reached: &mut Reached, wave: &[usize], check_undeclared: bool, force_commit: bool) -> (Vec<usize>, Vec<Instance>) {
    let jobs = crate::jobs().max(1);
    let batch = (wave.len() / jobs).clamp(1, MAX_WORKER_BATCH);
    let workers = jobs.min(wave.len().div_ceil(batch));
    if workers <= 1 || wave.len() < MIN_PARALLEL_WAVE {
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
            reached.blocked.push(at);
            reached.list[at].1 = typed;
            continue;
        }
        reached.unlaid.extend(unlaid);
        next.extend(calls(cx, &typed));
        reached.list[at].1 = typed;
        settled.push(at);
    }
    (settled, next)
}

fn parallel_wave(cx: &mut Context<'_>, reached: &mut Reached, wave: &[usize], workers: usize, check_undeclared: bool, force_commit: bool) -> (Vec<usize>, Vec<Instance>) {
    let forks: Vec<Context<'_>> = (0..workers - 1).map(|_| cx.fork()).collect();
    let next = AtomicUsize::new(0);
    cx.in_parallel = true;
    let (mut items, joined) = run_scoped(cx, forks, reached, (wave, &next, check_undeclared, force_commit));
    cx.in_parallel = false;
    let mut deferred = cx.take_deferred_evals();
    for (asks, worker_deferred, worker_items) in joined.into_iter().flatten() {
        cx.asks.repeat(&asks);
        deferred.extend(worker_deferred);
        items.extend(worker_items);
    }
    items.sort_by_key(|item| item.pos);
    if !deferred.is_empty() || items.iter().any(|item| item.deferred) {
        run_deferred(cx, (reached, wave, workers, check_undeclared, force_commit), deferred, &mut items);
    }
    commit_wave(cx, reached, wave, items, force_commit)
}

fn eval_deferred_list(cx: &mut Context<'_>, deferred: Vec<(Instance, u32)>) {
    let mut seen = rustc_hash::FxHashSet::default();
    for (inst, bits) in deferred {
        if seen.insert((inst, bits)) {
            super::evaluate::prime(cx, &inst, bits);
        }
    }
}

fn run_deferred(cx: &mut Context<'_>, (reached, wave, workers, check_undeclared, force_commit): (&Reached, &[usize], usize, bool, bool), deferred: Vec<(Instance, u32)>, items: &mut [WaveItem]) {
    eval_deferred_list(cx, deferred);
    for _ in 0..4 {
        let pending: Vec<usize> = items.iter().enumerate().filter(|(_, item)| item.deferred).map(|(i, _)| i).collect();
        if pending.len() < MIN_PARALLEL_WAVE || workers <= 1 {
            break;
        }
        let sub_wave: Vec<usize> = pending.iter().map(|&i| wave[items[i].pos]).collect();
        let forks: Vec<Context<'_>> = (0..workers - 1).map(|_| cx.fork()).collect();
        let next = AtomicUsize::new(0);
        cx.in_parallel = true;
        let (mut sub_items, joined) = run_scoped(cx, forks, reached, (&sub_wave, &next, check_undeclared, force_commit));
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
        eval_deferred_list(cx, next_deferred);
    }
    for item in items.iter_mut().filter(|item| item.deferred) {
        let asks0 = cx.asks.count();
        let mut unlaid = Vec::new();
        let typed = infer_entry(cx, &reached.list[wave[item.pos]].0, &mut unlaid, check_undeclared);
        item.asked |= cx.asks.count() > asks0;
        item.callees = if !force_commit && item.asked { Vec::new() } else { calls(cx, &typed).collect() };
        item.typed = typed;
        item.unlaid = unlaid;
        item.deferred = false;
    }
}

fn run_scoped<'a>(cx: &mut Context<'a>, forks: Vec<Context<'a>>, reached: &Reached, (wave, next, check_undeclared, force_commit): (&[usize], &AtomicUsize, bool, bool)) -> (Vec<WaveItem>, Vec<Option<WorkerOut>>) {
    std::thread::scope(|scope| {
        let handles: Vec<_> = forks
            .into_iter()
            .map(|mut fork| {
                scope.spawn(move || {
                    let items = worker_loop(&mut fork, reached, wave, next, check_undeclared, force_commit);
                    let deferred = fork.take_deferred_evals();
                    let (_, asks) = fork.release();
                    (asks, deferred, items)
                })
            })
            .collect();
        let own = worker_loop(cx, reached, wave, next, check_undeclared, force_commit);
        let joined = handles.into_iter().map(|handle| handle.join().map_or(None, Some)).collect();
        (own, joined)
    })
}

fn worker_loop(cx: &mut Context<'_>, reached: &Reached, wave: &[usize], next: &AtomicUsize, check_undeclared: bool, force_commit: bool) -> Vec<WaveItem> {
    let batch = (wave.len() / crate::jobs().max(1)).clamp(1, MAX_WORKER_BATCH);
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
            let callees = if !force_commit && asked { Vec::new() } else { calls(cx, &typed).collect() };
            items.push(WaveItem { pos: start + offset, typed, unlaid, callees, deferred, asked });
        }
    }
}

fn commit_wave(cx: &Context<'_>, reached: &mut Reached, wave: &[usize], items: Vec<WaveItem>, force_commit: bool) -> (Vec<usize>, Vec<Instance>) {
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
                reached.blocked.push(at);
                reached.list[at].1 = item.typed;
            }
            Some(item) => {
                reached.list[at].1 = item.typed;
                reached.unlaid.extend(item.unlaid);
                next.extend(item.callees);
                settled.push(at);
            }
            None => {
                reached.list[at].1 = Err(format!("{}: sema worker panicked", cx.name(reached.list[at].0.0)));
                settled.push(at);
            }
        }
    }
    (settled, next)
}

fn infer_entry(cx: &mut Context<'_>, instance: &Instance, unlaid: &mut Vec<String>, check_undeclared: bool) -> Result<Arc<Typed>, String> {
    if check_undeclared && cx.undeclared(instance) {
        return Err("no longer declared".to_owned());
    }
    typed_once(cx, instance, unlaid)
}

/// The inferred body of `instance`: the one an earlier pass kept, or a fresh inference that is kept when it type-checked, laid out, and asked for nothing.
pub(super) fn typed_once(cx: &mut Context<'_>, instance: &Instance, unlaid: &mut Vec<String>) -> Result<Arc<Typed>, String> {
    if let Some(typed) = cx.store.finals.get(instance) {
        register(cx, *instance, &typed);
        return Ok(typed);
    }
    if let Some(result) = super::summary::typed_once_poly(cx, instance, unlaid) {
        return result;
    }
    let before = cx.quiet_mark();
    let typed = Arc::new(infer(cx, instance.0, instance.1).map_err(|error| format!("{}: {error:#}", cx.name(instance.0)))?);
    let laid = lay_out(cx, &typed).map_err(|error| unlaid.push(format!("{}: {error:#}", cx.name(instance.0))));
    if laid.is_ok() && cx.quiet_mark() == before {
        cx.store.finals.insert(*instance, Arc::clone(&typed));
    }
    Ok(typed)
}

/// Computes the layout of each ADT, tuple and array type that lowering `typed` reads, so the names those layouts need are asked for while inference can still retry.
pub(super) fn lay_out(cx: &mut Context<'_>, typed: &Typed) -> anyhow::Result<()> {
    let scoped = typed.scoped(&cx.types);
    let tables = scoped.ensure_tables(&cx.types);
    let intrinsic_args: Vec<Ty> = typed.calls().filter(|(item, _)| cx.intrinsic(*item).is_some()).flat_map(|(_, substs)| cx.types.substs(substs).iter().copied()).collect();
    let locals = typed.locals.iter().map(|local| local.ty);
    let pats = tables.pats.values().copied();
    for ty in tables.exprs.iter().copied().chain(locals).chain(pats).chain(intrinsic_args) {
        if ty == Ty::NONE {
            continue;
        }
        if let Kind::Adt(item, _) = *cx.types.kind(ty)
            && cx.item(item).1.kind == crate::outline::item::ItemKind::Enum
        {
            cx.discriminants(item)?;
        }
        if matches!(cx.types.kind(ty), Kind::Adt(..) | Kind::Tuple(_) | Kind::Array(..)) {
            cx.layout(ty)?;
        }
    }
    Ok(())
}

/// The two roots a program run calls: std's `start` lang item at the return type of `main`, and `main`.
pub fn main_roots(cx: &mut Context<'_>, main: ItemRef) -> anyhow::Result<[Instance; 2]> {
    let start = cx.lang_item("start").ok_or_else(|| anyhow::anyhow!("no `start` lang item"))?;
    let ret = super::infer::return_type(cx, main)?;
    Ok([(start, cx.types.intern_substs(&[ret])), (main, Substs::EMPTY)])
}

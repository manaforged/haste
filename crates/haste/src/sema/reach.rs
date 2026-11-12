use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use rustc_hash::FxHashMap as HashMap;

use crate::names::ItemRef;
use crate::names::path::Requests;

use super::Context;
use super::infer::{Typed, infer, register};
use super::substs::Substs;
use super::ty::{Kind, Ty};

const MIN_PARALLEL_WAVE: usize = 16;
const WAVE_BATCH: usize = 4;

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

/// Infers `roots` and every function their bodies call, at the type arguments the calls give.
pub fn reach(cx: &mut Context<'_>, roots: &[Instance]) -> Reached {
    let mut reached = Reached::default();
    let found = drain(cx, &mut reached, roots.to_vec());
    reached.link(&found);
    reached
}

/// Infers again the reached functions `stale` that an edit did not replace, whose source changed, and every function they now call that was not reached yet, and returns the indexes of both.
pub fn reach_again(cx: &mut Context<'_>, reached: &mut Reached, stale: &[usize]) -> Vec<usize> {
    let stale: Vec<usize> = stale.iter().copied().filter(|index| reached.index.get(&reached.list[*index].0) == Some(index)).collect();
    let waves = stale_waves(&cx.types, reached, &stale);
    let mut queue = Vec::new();
    for wave in &waves {
        queue.extend(run_wave(cx, reached, wave, true));
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
}

type WorkerOut = (Requests, Vec<(Instance, u32)>, Vec<WaveItem>);

fn drain(cx: &mut Context<'_>, reached: &mut Reached, queue: Vec<Instance>) -> Vec<usize> {
    let mut added = Vec::new();
    let mut wave = enqueue(reached, queue);
    while !wave.is_empty() {
        let next = run_wave(cx, reached, &wave, false);
        added.extend_from_slice(&wave);
        wave = enqueue(reached, next);
    }
    added
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

fn run_wave(cx: &mut Context<'_>, reached: &mut Reached, wave: &[usize], check_undeclared: bool) -> Vec<Instance> {
    let workers = crate::jobs().min(wave.len().div_ceil(WAVE_BATCH));
    if workers <= 1 || wave.len() < MIN_PARALLEL_WAVE {
        return serial_wave(cx, reached, wave, check_undeclared);
    }
    parallel_wave(cx, reached, wave, workers, check_undeclared)
}

fn serial_wave(cx: &mut Context<'_>, reached: &mut Reached, wave: &[usize], check_undeclared: bool) -> Vec<Instance> {
    let mut next = Vec::new();
    for &at in wave {
        let instance = reached.list[at].0;
        let typed = infer_entry(cx, &instance, &mut reached.unlaid, check_undeclared);
        next.extend(calls(cx, &typed));
        reached.list[at].1 = typed;
    }
    next
}

fn parallel_wave(cx: &mut Context<'_>, reached: &mut Reached, wave: &[usize], workers: usize, check_undeclared: bool) -> Vec<Instance> {
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
        run_deferred(cx, (reached, wave, check_undeclared), deferred, &mut items);
    }
    commit_wave(cx, reached, wave, items)
}

fn run_deferred(cx: &mut Context<'_>, (reached, wave, check_undeclared): (&Reached, &[usize], bool), deferred: Vec<(Instance, u32)>, items: &mut [WaveItem]) {
    let mut seen = rustc_hash::FxHashSet::default();
    for (inst, bits) in deferred {
        if seen.insert((inst, bits)) {
            drop(super::evaluate::run(cx, &inst, bits));
        }
    }
    for item in items.iter_mut().filter(|item| item.deferred) {
        let mut unlaid = Vec::new();
        let typed = infer_entry(cx, &reached.list[wave[item.pos]].0, &mut unlaid, check_undeclared);
        item.callees = calls(cx, &typed).collect();
        item.typed = typed;
        item.unlaid = unlaid;
        item.deferred = false;
    }
}

fn run_scoped<'a>(cx: &mut Context<'a>, forks: Vec<Context<'a>>, reached: &Reached, (wave, next, check_undeclared): (&[usize], &AtomicUsize, bool)) -> (Vec<WaveItem>, Vec<Option<WorkerOut>>) {
    std::thread::scope(|scope| {
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
    let mut items = Vec::new();
    loop {
        let start = next.fetch_add(WAVE_BATCH, Ordering::Relaxed);
        if start >= wave.len() {
            return items;
        }
        let end = (start + WAVE_BATCH).min(wave.len());
        for (offset, &at) in wave[start..end].iter().enumerate() {
            let mut unlaid = Vec::new();
            let before = cx.deferred_evals.len();
            let typed = infer_entry(cx, &reached.list[at].0, &mut unlaid, check_undeclared);
            let deferred = cx.deferred_evals.len() > before;
            let callees = calls(cx, &typed).collect();
            items.push(WaveItem { pos: start + offset, typed, unlaid, callees, deferred });
        }
    }
}

fn commit_wave(cx: &Context<'_>, reached: &mut Reached, wave: &[usize], items: Vec<WaveItem>) -> Vec<Instance> {
    let mut slots: Vec<Option<WaveItem>> = (0..wave.len()).map(|_| None).collect();
    for item in items {
        let pos = item.pos;
        slots[pos] = Some(item);
    }
    let mut next = Vec::new();
    for (pos, slot) in slots.into_iter().enumerate() {
        let at = wave[pos];
        match slot {
            Some(item) => {
                reached.list[at].1 = item.typed;
                reached.unlaid.extend(item.unlaid);
                next.extend(item.callees);
            }
            None => reached.list[at].1 = Err(format!("{}: sema worker panicked", cx.name(reached.list[at].0.0))),
        }
    }
    next
}

fn infer_entry(cx: &mut Context<'_>, instance: &Instance, unlaid: &mut Vec<String>, check_undeclared: bool) -> Result<Arc<Typed>, String> {
    if check_undeclared && cx.undeclared(instance) {
        return Err("no longer declared".to_owned());
    }
    typed_once(cx, instance, unlaid)
}

/// The inferred body of `instance`: the one an earlier pass kept, or a fresh inference that is kept when it type-checked, laid out, and asked for nothing.
fn typed_once(cx: &mut Context<'_>, instance: &Instance, unlaid: &mut Vec<String>) -> Result<Arc<Typed>, String> {
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
    let tables = typed.ensure_tables(&cx.types);
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

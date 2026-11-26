use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::names::ItemRef;

use super::Context;
use super::infer::{Res, Typed};
use super::substs::Substs;

mod collect;
mod crate_check;
pub(in crate::sema) mod filter;
mod items;
mod order;
mod unwind;
mod wave;

pub(super) use collect::run_scoped;
pub use collect::{reach_again, reach_anew, reach_wave};
pub(crate) use items::lay_out;
pub(crate) use items::typed_once;
pub(super) use items::{infer_entry, is_inferring};
pub(crate) use unwind::{foreign_dynamic_may_unwind, host_symbol_may_unwind};
pub use wave::infer_quiet;

use filter::ReachFilter;

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
    pub(in crate::sema::reach) crate_errors: Vec<usize>,
    pub(in crate::sema::reach) states: Vec<ReachedEntry>,
    untyped_count: usize,
    pub(in crate::sema::reach) filter: ReachFilter,
    catch_unwind_state: AtomicUsize,
    catch_unwind_cache: Mutex<CatchUnwindCache>,
    may_unwind_cache: RwLock<Option<(usize, Vec<bool>)>>,
    incremental: bool,
    force_unwind: bool,
}

fn is_thread_spawn_name(name: &str, src: &crate::outline::source::Source) -> bool {
    matches!(name, "spawn" | "spawn_unchecked" | "spawn_scoped" | "scope")
        && src
            .path
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            == Some("thread")
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
                    if entry
                        .name
                        .is_some_and(|at| is_thread_spawn_name(src.text(at), src))
                    {
                        self.catch_unwind_state
                            .store(CATCH_UNWIND_FOUND | (start + idx + 1), Ordering::Release);
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
                let is_std =
                    crate_flags.get(callee.krate as usize).copied().unwrap_or(0) & CRATE_STD != 0;
                if entry.name.is_some_and(|at| {
                    let name = src.text(at);
                    name == "catch_unwind" || (is_std && is_thread_spawn_name(name, src))
                }) {
                    self.catch_unwind_state
                        .store(CATCH_UNWIND_FOUND | (start + idx + 1), Ordering::Release);
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

    /// Forces unwinding in release builds (used when compiling test suites).
    pub fn set_force_unwind(&mut self, force: bool) {
        self.force_unwind = force;
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
        if let Some(bit) = crate::sync::read_lock(&self.may_unwind_cache)
            .as_ref()
            .filter(|(n, _)| *n == len)
            .and_then(|(_, bits)| bits.get(index).copied())
        {
            return bit;
        }
        let mut cache = crate::sync::write_lock(&self.may_unwind_cache);
        if cache.as_ref().is_none_or(|(n, _)| *n != len) {
            *cache = Some((len, unwind::compute_may_unwind(cx, self)));
        }
        cache
            .as_ref()
            .and_then(|(_, bits)| bits.get(index).copied())
            .unwrap_or(true)
    }

    /// The typed body of function `index`, or an error naming `name` and why its body did not type-check.
    pub fn typed(&self, index: usize, name: &str) -> anyhow::Result<&Arc<Typed>> {
        self.list[index]
            .1
            .as_ref()
            .map_err(|why| anyhow::anyhow!("{name} did not type-check: {why}"))
    }

    /// `roots` and every function they reach through the bodies that type-checked, each once and after every function it reaches.
    pub fn closure(&self, roots: &[usize]) -> Vec<usize> {
        let mut seen = vec![false; self.list.len()];
        let (mut stack, mut found) = (
            roots
                .iter()
                .chain(self.crate_errors.iter())
                .rev()
                .map(|&root| (root, false))
                .collect::<Vec<_>>(),
            Vec::new(),
        );
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

    pub(super) fn set_entry(
        &mut self,
        index: usize,
        next: ReachedEntry,
        result: Result<Arc<Typed>, String>,
    ) {
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
            let cache = self
                .catch_unwind_cache
                .get_mut()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            cache.checked_items.clear();
            cache.checked_callees.clear();
        }
        *self
            .may_unwind_cache
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
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

    pub(in crate::sema::reach) fn clear_crate_errors(&mut self) {
        self.crate_errors.clear();
    }

    pub(in crate::sema::reach) fn note_crate_error(&mut self, at: usize) {
        if !self.crate_errors.contains(&at) {
            self.crate_errors.push(at);
        }
    }

    pub(in crate::sema::reach) fn record_crate_error(&mut self, instance: Instance, error: String) {
        let at = match self.index.get(&instance).copied() {
            Some(at) => {
                self.set_entry(at, ReachedEntry::Settled, Err(error));
                at
            }
            None => {
                let at = self.push_entry(instance, ReachedEntry::Settled);
                self.list[at].1 = Err(error);
                at
            }
        };
        self.note_crate_error(at);
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
            if let Some(&id) = self.index.get(inst) {
                self.pinned[id] = true;
            }
        }
    }

    fn link(&mut self, indices: &[usize]) {
        self.calls.resize(self.list.len(), Vec::new());
        self.indirect.resize(self.list.len(), Vec::new());
        self.call_sites
            .retract(indices, &self.calls, &self.indirect);
        let workers = crate::jobs().min(indices.len() / 256);
        let edges = |at: usize| {
            self.list[at]
                .1
                .as_ref()
                .map(|t| t.collect_link_edges(&self.index))
                .unwrap_or_default()
        };
        if workers <= 1 {
            for &at in indices {
                (self.calls[at], self.indirect[at]) = edges(at);
            }
        } else {
            let next = AtomicUsize::new(0);
            let worker = || {
                let mut local = Vec::new();
                while let Some(slice) = indices.get(next.fetch_add(256, Ordering::Relaxed)..) {
                    for &at in &slice[..slice.len().min(256)] {
                        local.push((at, edges(at)));
                    }
                }
                local
            };
            let (own, joined) = crate::sync::scope(|scope| {
                let handles: Vec<_> = (1..workers).map(|_| scope.spawn(worker)).collect();
                (
                    worker(),
                    handles
                        .into_iter()
                        .flat_map(|h| h.join().unwrap_or_default())
                        .collect::<Vec<_>>(),
                )
            });
            for (at, (c, i)) in own.into_iter().chain(joined) {
                (self.calls[at], self.indirect[at]) = (c, i);
            }
        }
        super::infer::recount_call_sites(
            self.list.len(),
            |at| self.states[at] == ReachedEntry::Settled,
            &self.calls,
            &self.indirect,
            &self.pinned,
            &mut self.call_sites,
        );
        *self
            .may_unwind_cache
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
    }
}

pub(super) fn push_tail_calls(typed: &Typed, push: &mut impl FnMut(Instance)) {
    for r in typed
        .drops
        .iter()
        .chain(super::infer::shape_calls(&typed.shape))
    {
        if let Res::Fn(item, substs) = *r {
            push((item, substs));
        }
    }
}

pub(crate) const PANIC_BOUNDS_CHECK: &str = "panic_bounds_check";

pub(crate) fn unwind_enabled(cx: &Context<'_>, reached: &Reached) -> bool {
    crate::profile() != crate::outline::tree::Profile::Release
        || reached.force_unwind
        || reached.is_incremental()
        || reached.has_catch_unwind(cx)
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
pub fn calls<'t>(
    cx: &'t Context<'_>,
    typed: &'t Result<Arc<Typed>, String>,
) -> impl Iterator<Item = Instance> + 't {
    typed
        .iter()
        .flat_map(|typed| typed.calls())
        .filter(|(item, _)| cx.intrinsic(*item).is_none())
}

/// The two roots a program run calls: std's `start` lang item at the return type of `main`, and `main`.
pub fn main_roots(cx: &mut Context<'_>, main: ItemRef) -> anyhow::Result<[Instance; 2]> {
    let start = cx
        .lang_item("start")
        .ok_or_else(|| anyhow::anyhow!("no `start` lang item"))?;
    let ret = super::infer::return_type(cx, main)?;
    Ok([
        (start, cx.types.intern_substs(&[ret])),
        (main, Substs::EMPTY),
    ])
}

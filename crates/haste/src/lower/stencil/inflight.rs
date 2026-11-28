use std::sync::{Arc, Condvar, Mutex};
use std::thread::ThreadId;

use rustc_hash::FxHashMap;

use super::{InlineTemplate, TemplateKey};

const SHARDS: usize = 32;

pub(super) struct InFlight {
    shards: [Mutex<FxHashMap<usize, u32>>; SHARDS],
}

impl Default for InFlight {
    fn default() -> Self {
        Self {
            shards: std::array::from_fn(|_| Mutex::new(FxHashMap::default())),
        }
    }
}

pub(super) struct Guard<'a> {
    shard: &'a Mutex<FxHashMap<usize, u32>>,
    index: usize,
}

impl Drop for Guard<'_> {
    fn drop(&mut self) {
        let mut map = self.shard.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(count) = map.get_mut(&self.index) {
            if *count <= 1 {
                map.remove(&self.index);
            } else {
                *count -= 1;
            }
        }
    }
}

impl InFlight {
    pub(super) fn enter(&self, index: usize, abort_if_busy: bool) -> Option<Guard<'_>> {
        let shard = &self.shards[index & (SHARDS - 1)];
        let mut map = shard.lock().unwrap_or_else(|e| e.into_inner());
        if abort_if_busy && map.get(&index).copied().unwrap_or(0) > 0 {
            return None;
        }
        *map.entry(index).or_insert(0) += 1;
        Some(Guard { shard, index })
    }
}

enum SlotOutcome {
    Running,
    Cached(Option<Arc<InlineTemplate>>),
    Uncached,
}

struct TemplateSlot {
    owner: ThreadId,
    done: Mutex<SlotOutcome>,
    cv: Condvar,
}

impl TemplateSlot {
    fn new(owner: ThreadId) -> Self {
        Self {
            owner,
            done: Mutex::new(SlotOutcome::Running),
            cv: Condvar::new(),
        }
    }
}

#[derive(Default)]
struct FlightState {
    active: FxHashMap<TemplateKey, Arc<TemplateSlot>>,
    waits: FxHashMap<ThreadId, ThreadId>,
}

#[derive(Default)]
pub(super) struct TemplateFlight {
    state: Mutex<FlightState>,
}

pub(super) enum TemplateClaim<'a> {
    Build(TemplateGuard<'a>),
    Ready(Option<Arc<InlineTemplate>>),
    Cycle,
    Abort,
}

pub(super) struct TemplateGuard<'a> {
    flight: &'a TemplateFlight,
    key: TemplateKey,
    slot: Arc<TemplateSlot>,
    settled: bool,
}

impl TemplateGuard<'_> {
    pub(super) fn finish(mut self, outcome: Option<Option<Arc<InlineTemplate>>>) {
        self.settled = true;
        self.flight.complete(self.key, &self.slot, outcome);
    }
}

impl Drop for TemplateGuard<'_> {
    fn drop(&mut self) {
        if !self.settled {
            self.flight.complete(self.key, &self.slot, None);
        }
    }
}

struct WaitGuard<'a> {
    flight: &'a TemplateFlight,
    me: ThreadId,
}

impl Drop for WaitGuard<'_> {
    fn drop(&mut self) {
        self.flight
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .waits
            .remove(&self.me);
    }
}

fn would_cycle(waits: &FxHashMap<ThreadId, ThreadId>, owner: ThreadId, me: ThreadId) -> bool {
    let mut cursor = owner;
    while let Some(&next) = waits.get(&cursor) {
        if next == me {
            return true;
        }
        cursor = next;
    }
    false
}

impl TemplateFlight {
    pub(super) fn claim(
        &self,
        key: TemplateKey,
        abort_if_busy: bool,
        check_cached: impl Fn(&TemplateKey) -> Option<Option<Arc<InlineTemplate>>>,
    ) -> TemplateClaim<'_> {
        let me = std::thread::current().id();
        loop {
            let wait_slot = {
                let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
                match state.active.get(&key).cloned() {
                    None => {
                        if let Some(cached) = check_cached(&key) {
                            return TemplateClaim::Ready(cached);
                        }
                        let slot = Arc::new(TemplateSlot::new(me));
                        state.active.insert(key, Arc::clone(&slot));
                        return TemplateClaim::Build(TemplateGuard {
                            flight: self,
                            key,
                            slot,
                            settled: false,
                        });
                    }
                    Some(_) if abort_if_busy => return TemplateClaim::Abort,
                    Some(slot) => {
                        if slot.owner == me || would_cycle(&state.waits, slot.owner, me) {
                            return TemplateClaim::Cycle;
                        }
                        state.waits.insert(me, slot.owner);
                        slot
                    }
                }
            };
            let _wait = WaitGuard { flight: self, me };
            let mut outcome = wait_slot.done.lock().unwrap_or_else(|e| e.into_inner());
            while matches!(*outcome, SlotOutcome::Running) {
                outcome = wait_slot
                    .cv
                    .wait(outcome)
                    .unwrap_or_else(|e| e.into_inner());
            }
            if let SlotOutcome::Cached(ref cached) = *outcome {
                return TemplateClaim::Ready(cached.clone());
            }
        }
    }

    fn complete(
        &self,
        key: TemplateKey,
        slot: &Arc<TemplateSlot>,
        outcome: Option<Option<Arc<InlineTemplate>>>,
    ) {
        {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            if state
                .active
                .get(&key)
                .is_some_and(|cur| Arc::ptr_eq(cur, slot))
            {
                state.active.remove(&key);
            }
        }
        let mut done = slot.done.lock().unwrap_or_else(|e| e.into_inner());
        *done = match outcome {
            Some(cached) => SlotOutcome::Cached(cached),
            None => SlotOutcome::Uncached,
        };
        slot.cv.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};

    use super::*;
    use crate::lower::stencil::FootprintId;
    use crate::names::ItemRef;

    fn key(id: u32) -> TemplateKey {
        TemplateKey {
            item: ItemRef {
                krate: 0,
                file: 0,
                item: id,
            },
            footprint: FootprintId(u128::from(id)),
            limit: 64,
            low: false,
            compact: false,
            sole: false,
        }
    }

    #[test]
    fn template_flight_dedups_concurrent_builders() {
        let flight = Arc::new(TemplateFlight::default());
        let barrier = Arc::new(Barrier::new(8));
        let built = Arc::new(AtomicUsize::new(0));
        let k = key(1);
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..8)
                .map(|_| {
                    let flight = Arc::clone(&flight);
                    let barrier = Arc::clone(&barrier);
                    let built = Arc::clone(&built);
                    scope.spawn(move || {
                        barrier.wait();
                        match flight.claim(k, false, |_| None) {
                            TemplateClaim::Build(guard) => {
                                built.fetch_add(1, Ordering::SeqCst);
                                std::thread::sleep(std::time::Duration::from_millis(10));
                                guard.finish(Some(None));
                            }
                            TemplateClaim::Ready(None) => {}
                            _ => panic!("unexpected claim outcome"),
                        }
                    })
                })
                .collect();
            for h in handles {
                h.join().unwrap();
            }
        });
        assert_eq!(built.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn template_flight_breaks_mutual_cycle_without_deadlock() {
        let flight = Arc::new(TemplateFlight::default());
        let barrier = Arc::new(Barrier::new(2));
        let cycles = Arc::new(AtomicUsize::new(0));
        let (k1, k2) = (key(10), key(20));
        std::thread::scope(|scope| {
            let f1 = Arc::clone(&flight);
            let b1 = Arc::clone(&barrier);
            let c1 = Arc::clone(&cycles);
            let h1 = scope.spawn(move || {
                let TemplateClaim::Build(g1) = f1.claim(k1, false, |_| None) else {
                    panic!("expected build for k1");
                };
                b1.wait();
                if matches!(f1.claim(k2, false, |_| None), TemplateClaim::Cycle) {
                    c1.fetch_add(1, Ordering::SeqCst);
                }
                g1.finish(Some(None));
            });
            let f2 = Arc::clone(&flight);
            let b2 = Arc::clone(&barrier);
            let c2 = Arc::clone(&cycles);
            let h2 = scope.spawn(move || {
                let TemplateClaim::Build(g2) = f2.claim(k2, false, |_| None) else {
                    panic!("expected build for k2");
                };
                b2.wait();
                if matches!(f2.claim(k1, false, |_| None), TemplateClaim::Cycle) {
                    c2.fetch_add(1, Ordering::SeqCst);
                }
                g2.finish(Some(None));
            });
            h1.join().unwrap();
            h2.join().unwrap();
        });
        assert!(cycles.load(Ordering::SeqCst) >= 1);
    }

    #[test]
    fn template_flight_drop_wakes_waiter_for_takeover() {
        let flight = Arc::new(TemplateFlight::default());
        let barrier = Arc::new(Barrier::new(2));
        let k = key(30);
        std::thread::scope(|scope| {
            let f1 = Arc::clone(&flight);
            let b1 = Arc::clone(&barrier);
            let h1 = scope.spawn(move || {
                let TemplateClaim::Build(g1) = f1.claim(k, false, |_| None) else {
                    panic!("expected build");
                };
                b1.wait();
                std::thread::sleep(std::time::Duration::from_millis(10));
                drop(g1);
            });
            let f2 = Arc::clone(&flight);
            let b2 = Arc::clone(&barrier);
            let h2 = scope.spawn(move || {
                b2.wait();
                match f2.claim(k, false, |_| None) {
                    TemplateClaim::Build(g2) => g2.finish(Some(None)),
                    _ => panic!("expected takeover build after drop"),
                }
            });
            h1.join().unwrap();
            h2.join().unwrap();
        });
    }
}

use std::borrow::Borrow;
use std::hash::Hash;
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread::ThreadId;

use anyhow::Result;
use rustc_hash::FxHashMap;

use crate::sync::{lock, read_lock, shard_index, write_lock};

const SHARDS: usize = 32;

/// A sharded concurrent hash map shared across worker contexts by `Arc`.
pub struct Table<K, V> {
    shards: Arc<[RwLock<FxHashMap<K, V>>; SHARDS]>,
}

impl<K, V> Clone for Table<K, V> {
    fn clone(&self) -> Self {
        Self { shards: Arc::clone(&self.shards) }
    }
}

impl<K, V> Default for Table<K, V> {
    fn default() -> Self {
        Self { shards: Arc::new(std::array::from_fn(|_| RwLock::new(FxHashMap::default()))) }
    }
}

impl<K, V> Table<K, V> {
    /// A clone of the value stored under `key`.
    pub fn get<Q>(&self, key: &Q) -> Option<V>
    where
        K: Borrow<Q> + Eq + Hash,
        Q: Eq + Hash + ?Sized,
        V: Clone,
    {
        read_lock(&self.shards[shard_index::<_, SHARDS>(key)]).get(key).cloned()
    }

    /// Whether `key` holds a value in the table.
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q> + Eq + Hash,
        Q: Eq + Hash + ?Sized,
    {
        read_lock(&self.shards[shard_index::<_, SHARDS>(key)]).contains_key(key)
    }

    /// Stores `value` under `key`, replacing any earlier entry.
    pub fn insert(&self, key: K, value: V)
    where
        K: Eq + Hash,
    {
        write_lock(&self.shards[shard_index::<_, SHARDS>(&key)]).insert(key, value);
    }

    /// A clone of the value stored under `key`, inserting `make()` when absent.
    pub fn get_or_insert_with(&self, key: K, make: impl FnOnce() -> V) -> V
    where
        K: Eq + Hash,
        V: Clone,
    {
        let shard = shard_index::<_, SHARDS>(&key);
        if let Some(found) = read_lock(&self.shards[shard]).get(&key).cloned() {
            return found;
        }
        let made = make();
        write_lock(&self.shards[shard]).entry(key).or_insert(made).clone()
    }

    /// Removes and returns the value stored under `key`.
    pub fn remove<Q>(&self, key: &Q) -> Option<V>
    where
        K: Borrow<Q> + Eq + Hash,
        Q: Eq + Hash + ?Sized,
    {
        write_lock(&self.shards[shard_index::<_, SHARDS>(key)]).remove(key)
    }

    /// Empties every shard of the table.
    pub fn clear(&self) {
        for shard in self.shards.iter() {
            write_lock(shard).clear();
        }
    }

    /// Keeps only the entries for which `keep` holds.
    pub fn retain(&self, mut keep: impl FnMut(&K, &mut V) -> bool) {
        for shard in self.shards.iter() {
            write_lock(shard).retain(&mut keep);
        }
    }

    /// A new table holding clones of the entries for which `keep` holds.
    pub fn retained(&self, mut keep: impl FnMut(&K, &V) -> bool) -> Self
    where
        K: Clone + Eq + Hash,
        V: Clone,
    {
        let out = Self::default();
        for (index, shard) in self.shards.iter().enumerate() {
            let guard = read_lock(shard);
            let mut target = write_lock(&out.shards[index]);
            for (key, value) in guard.iter() {
                if keep(key, value) {
                    target.insert(key.clone(), value.clone());
                }
            }
        }
        out
    }
}

enum SlotState<V> {
    Running(ThreadId),
    Ready(V),
    Failed,
}

struct Slot<V> {
    state: Mutex<SlotState<V>>,
    ready: Condvar,
}

enum Step<V> {
    Ready(V),
    Compute(Arc<Slot<V>>),
    Cycle,
    Retry,
}

struct FlightInner<K, V> {
    slots: Table<K, Arc<Slot<V>>>,
    waits: Mutex<FxHashMap<ThreadId, ThreadId>>,
}

pub(super) struct Flight<K, V> {
    inner: Arc<FlightInner<K, V>>,
}

impl<K, V> Clone for Flight<K, V> {
    fn clone(&self) -> Self {
        Self { inner: Arc::clone(&self.inner) }
    }
}

impl<K, V> Default for Flight<K, V> {
    fn default() -> Self {
        Self { inner: Arc::new(FlightInner { slots: Table::default(), waits: Mutex::new(FxHashMap::default()) }) }
    }
}

struct FlightGuard<'f, K: Eq + Hash, V> {
    flight: &'f Flight<K, V>,
    key: K,
    slot: Arc<Slot<V>>,
    done: bool,
}

impl<K: Eq + Hash, V> Drop for FlightGuard<'_, K, V> {
    fn drop(&mut self) {
        if !self.done {
            self.flight.fail(&self.key, &self.slot);
        }
    }
}

struct WaitGuard<'f, K, V> {
    flight: &'f Flight<K, V>,
    me: ThreadId,
}

impl<K, V> Drop for WaitGuard<'_, K, V> {
    fn drop(&mut self) {
        self.flight.end_wait(self.me);
    }
}

impl<K, V> Flight<K, V> {
    pub(super) fn retain(&self, mut keep: impl FnMut(&K) -> bool) {
        self.inner.slots.retain(|key, _| keep(key));
    }

    fn begin_wait(&self, me: ThreadId, owner: ThreadId) -> bool {
        if me == owner {
            return false;
        }
        let mut waits = lock(&self.inner.waits);
        let mut cursor = owner;
        while let Some(&next) = waits.get(&cursor) {
            if next == me {
                return false;
            }
            cursor = next;
        }
        waits.insert(me, owner);
        true
    }

    fn end_wait(&self, me: ThreadId) {
        lock(&self.inner.waits).remove(&me);
    }

    fn fail(&self, key: &K, slot: &Slot<V>)
    where
        K: Eq + Hash,
    {
        self.inner.slots.remove(key);
        *lock(&slot.state) = SlotState::Failed;
        slot.ready.notify_all();
    }
}

impl<K: Clone + Eq + Hash, V: Clone> Flight<K, V> {
    pub(super) fn run(&self, cx: &mut super::Context<'_>, key: K, compute: impl FnOnce(&mut super::Context<'_>) -> Result<V>, on_cycle: impl FnOnce(&mut super::Context<'_>) -> Result<V>) -> Result<V> {
        let me = std::thread::current().id();
        loop {
            match self.acquire(&key, me) {
                Step::Ready(value) => return Ok(value),
                Step::Cycle => return on_cycle(cx),
                Step::Retry => continue,
                Step::Compute(slot) => {
                    let mut guard = FlightGuard { flight: self, key: key.clone(), slot: Arc::clone(&slot), done: false };
                    let before = cx.quiet_mark();
                    let result = compute(cx);
                    guard.done = true;
                    if result.is_ok() && cx.quiet_mark() != before {
                        self.fail(&key, &slot);
                    } else {
                        self.finish(&key, &slot, &result);
                    }
                    return result;
                }
            }
        }
    }

    fn acquire(&self, key: &K, me: ThreadId) -> Step<V> {
        if let Some(slot) = self.inner.slots.get(key) {
            return self.wait_or_cycle(&slot, me);
        }
        let fresh = Arc::new(Slot { state: Mutex::new(SlotState::Running(me)), ready: Condvar::new() });
        let slot = self.inner.slots.get_or_insert_with(key.clone(), || Arc::clone(&fresh));
        if Arc::ptr_eq(&slot, &fresh) {
            return Step::Compute(fresh);
        }
        self.wait_or_cycle(&slot, me)
    }

    fn wait_or_cycle(&self, slot: &Slot<V>, me: ThreadId) -> Step<V> {
        let mut guard = lock(&slot.state);
        let owner = match &*guard {
            SlotState::Ready(value) => return Step::Ready(value.clone()),
            SlotState::Failed => return Step::Retry,
            SlotState::Running(owner) => *owner,
        };
        if !self.begin_wait(me, owner) {
            return Step::Cycle;
        }
        let wait = WaitGuard { flight: self, me };
        while matches!(*guard, SlotState::Running(_)) {
            guard = slot.ready.wait(guard).unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        let step = match &*guard {
            SlotState::Ready(value) => Step::Ready(value.clone()),
            _ => Step::Retry,
        };
        drop(guard);
        drop(wait);
        step
    }

    fn finish(&self, key: &K, slot: &Slot<V>, result: &Result<V>) {
        match result {
            Ok(value) => {
                *lock(&slot.state) = SlotState::Ready(value.clone());
                slot.ready.notify_all();
            }
            Err(_) => self.fail(key, slot),
        }
    }
}

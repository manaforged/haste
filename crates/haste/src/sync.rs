use std::borrow::Borrow;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU8, AtomicU32, AtomicU64, Ordering};
use std::sync::{
    Mutex, MutexGuard, OnceLock, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard,
};

use rustc_hash::FxHasher;

mod scope;

pub(crate) use scope::scope;

pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

pub(crate) fn panic_text(payload: &(dyn std::any::Any + Send)) -> &str {
    payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("a panic with no message")
}

pub(crate) fn into_inner<T>(mutex: Mutex<T>) -> T {
    mutex.into_inner().unwrap_or_else(PoisonError::into_inner)
}

pub(crate) fn read_lock<T>(rw: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    rw.read().unwrap_or_else(PoisonError::into_inner)
}

pub(crate) fn write_lock<T>(rw: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    rw.write().unwrap_or_else(PoisonError::into_inner)
}

#[repr(align(128))]
pub(crate) struct CachePadded<T>(pub(crate) T);

impl<T: Default> Default for CachePadded<T> {
    fn default() -> Self {
        Self(T::default())
    }
}

#[inline]
fn hash_key<Q: Hash + ?Sized>(key: &Q) -> u64 {
    let mut hasher = FxHasher::default();
    key.hash(&mut hasher);
    hasher.finish()
}

#[inline]
fn shard_of<const SHARDS: usize>(hash: u64) -> usize {
    const { assert!(SHARDS.is_power_of_two()) };
    (((hash >> 58) ^ (hash >> 29) ^ hash) as usize) & (SHARDS - 1)
}

pub(crate) fn shard_index<Q: Hash + ?Sized, const SHARDS: usize>(key: &Q) -> usize {
    shard_of::<SHARDS>(hash_key(key))
}

/// Evaluates `work` over `items` on worker threads, returning outputs in input order.
pub(crate) fn parallel_map<T: Send, R: Send>(
    items: Vec<T>,
    work: impl Fn(T) -> R + Sync,
) -> Vec<R> {
    let total = items.len();
    let workers = crate::jobs().min(total);
    if total < 2 || workers <= 1 {
        return items.into_iter().map(work).collect();
    }
    let queue = Mutex::new(items.into_iter().enumerate());
    let worker = || {
        let mut local = Vec::new();
        loop {
            let next = lock(&queue).next();
            let Some((idx, item)) = next else { break };
            local.push((idx, work(item)));
        }
        local
    };
    let chunks = scope(|scope| {
        let handles: Vec<_> = (0..workers - 1).map(|_| scope.spawn(&worker)).collect();
        let mut all = worker();
        for handle in handles {
            match handle.join() {
                Ok(chunk) => all.extend(chunk),
                Err(payload) => std::panic::resume_unwind(payload),
            }
        }
        all
    });
    let mut slots: Vec<Option<R>> = (0..total).map(|_| None).collect();
    for (idx, res) in chunks {
        slots[idx] = Some(res);
    }
    slots.into_iter().flatten().collect()
}

pub(crate) struct Slab<T, const SLAB: usize, const SLABS: usize> {
    slabs: Box<[OnceLock<Box<[OnceLock<T>]>>]>,
}

impl<T, const SLAB: usize, const SLABS: usize> Default for Slab<T, SLAB, SLABS> {
    fn default() -> Self {
        Self {
            slabs: (0..SLABS)
                .map(|_| OnceLock::new())
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }
}

impl<T, const SLAB: usize, const SLABS: usize> Slab<T, SLAB, SLABS> {
    pub(crate) fn set(&self, index: usize, value: T) {
        assert!(index < SLAB * SLABS, "slab capacity exceeded");
        let slab = self.slabs[index / SLAB].get_or_init(|| {
            (0..SLAB)
                .map(|_| OnceLock::new())
                .collect::<Vec<_>>()
                .into_boxed_slice()
        });
        slab[index % SLAB].get_or_init(|| value);
    }

    #[inline]
    pub(crate) fn get(&self, index: usize) -> Option<&T> {
        self.slabs
            .get(index / SLAB)?
            .get()?
            .get(index % SLAB)?
            .get()
    }
}

pub(crate) trait AtomicWord: Sized {
    type Val: Copy + Eq;
    const ZERO: Self::Val;
    fn new_zero() -> Self;
    fn load_acquire(&self) -> Self::Val;
    fn store_release(&self, value: Self::Val);
}

impl AtomicWord for AtomicU8 {
    type Val = u8;
    const ZERO: u8 = 0;
    fn new_zero() -> Self {
        Self::new(0)
    }
    #[inline]
    fn load_acquire(&self) -> u8 {
        self.load(Ordering::Acquire)
    }
    #[inline]
    fn store_release(&self, value: u8) {
        self.store(value, Ordering::Release)
    }
}

impl AtomicWord for AtomicU32 {
    type Val = u32;
    const ZERO: u32 = 0;
    fn new_zero() -> Self {
        Self::new(0)
    }
    #[inline]
    fn load_acquire(&self) -> u32 {
        self.load(Ordering::Acquire)
    }
    #[inline]
    fn store_release(&self, value: u32) {
        self.store(value, Ordering::Release)
    }
}

pub(crate) struct AtomicSlab<A, const SLAB: usize, const SLABS: usize> {
    slabs: Box<[OnceLock<Box<[A]>>]>,
}

impl<A, const SLAB: usize, const SLABS: usize> Default for AtomicSlab<A, SLAB, SLABS> {
    fn default() -> Self {
        Self {
            slabs: (0..SLABS)
                .map(|_| OnceLock::new())
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }
}

impl<A: AtomicWord, const SLAB: usize, const SLABS: usize> AtomicSlab<A, SLAB, SLABS> {
    #[inline]
    pub(crate) fn get(&self, index: usize) -> Option<A::Val> {
        let val = self.slabs.get(index / SLAB)?.get()?[index % SLAB].load_acquire();
        (val != A::ZERO).then_some(val)
    }

    pub(crate) fn set(&self, index: usize, value: A::Val) {
        debug_assert!(value != A::ZERO);
        if index >= SLAB * SLABS {
            return;
        }
        let slab = self.slabs[index / SLAB].get_or_init(|| {
            (0..SLAB)
                .map(|_| A::new_zero())
                .collect::<Vec<_>>()
                .into_boxed_slice()
        });
        slab[index % SLAB].store_release(value);
    }
}

pub(crate) type AtomicU8Slab<const SLAB: usize, const SLABS: usize> =
    AtomicSlab<AtomicU8, SLAB, SLABS>;
pub(crate) type AtomicU32Slab<const SLAB: usize, const SLABS: usize> =
    AtomicSlab<AtomicU32, SLAB, SLABS>;

#[derive(Default)]
struct IndexTable {
    entries: Box<[(u32, u32)]>,
    len: usize,
}

impl IndexTable {
    #[inline]
    fn tag(hash: u64) -> u32 {
        ((hash >> 32) as u32) | 1
    }

    #[inline]
    fn find(&self, hash: u64, mut eq: impl FnMut(u32) -> bool) -> Option<u32> {
        if self.entries.is_empty() {
            return None;
        }
        let mask = self.entries.len() - 1;
        let tag = Self::tag(hash);
        let mut pos = ((hash >> 7) as usize) & mask;
        loop {
            let (entry_tag, id) = self.entries[pos];
            if entry_tag == 0 {
                return None;
            }
            if entry_tag == tag && eq(id) {
                return Some(id);
            }
            pos = (pos + 1) & mask;
        }
    }

    fn insert(&mut self, hash: u64, id: u32, mut hash_of: impl FnMut(u32) -> u64) {
        if self.len * 4 >= self.entries.len() * 3 {
            let next_cap = if self.entries.is_empty() {
                16
            } else {
                self.entries.len() * 2
            };
            let mut next = vec![(0u32, 0u32); next_cap].into_boxed_slice();
            let mask = next_cap - 1;
            for &(entry_tag, entry_id) in &self.entries {
                if entry_tag == 0 {
                    continue;
                }
                let mut pos = ((hash_of(entry_id) >> 7) as usize) & mask;
                while next[pos].0 != 0 {
                    pos = (pos + 1) & mask;
                }
                next[pos] = (entry_tag, entry_id);
            }
            self.entries = next;
        }
        let mask = self.entries.len() - 1;
        let mut pos = ((hash >> 7) as usize) & mask;
        while self.entries[pos].0 != 0 {
            pos = (pos + 1) & mask;
        }
        self.entries[pos] = (Self::tag(hash), id);
        self.len += 1;
    }
}

const INTERNER_CACHE: usize = 65_536;

pub(crate) struct Interner<K, const SHARDS: usize, const SLAB: usize, const SLABS: usize> {
    cache: Box<[AtomicU64]>,
    shards: Box<[CachePadded<RwLock<IndexTable>>]>,
    slabs: Slab<K, SLAB, SLABS>,
    next: AtomicU32,
}

impl<K, const SHARDS: usize, const SLAB: usize, const SLABS: usize> Default
    for Interner<K, SHARDS, SLAB, SLABS>
{
    fn default() -> Self {
        Self {
            cache: (0..INTERNER_CACHE).map(|_| AtomicU64::new(0)).collect(),
            shards: (0..SHARDS).map(|_| CachePadded::default()).collect(),
            slabs: Slab::default(),
            next: AtomicU32::new(0),
        }
    }
}

impl<K: Eq + Hash, const SHARDS: usize, const SLAB: usize, const SLABS: usize>
    Interner<K, SHARDS, SLAB, SLABS>
{
    /// The id of `key`, calling `make` with the new id under the shard's write lock, before the id is visible, when `key` is new.
    pub(crate) fn intern<Q>(&self, key: &Q, make: impl FnOnce(&Q, u32) -> K) -> u32
    where
        K: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        let hash = hash_key(key);
        self.intern_with(hash, key, |stored, key| stored.borrow() == *key, make)
    }

    /// The id of owned `key`, moving it into the slab without cloning when new and invoking `on_new` before publishing it.
    pub(crate) fn intern_owned(&self, key: K, on_new: impl FnOnce(&K, u32)) -> u32 {
        let hash = hash_key(&key);
        self.intern_with(
            hash,
            key,
            |stored, key| stored == key,
            |key, id| {
                on_new(&key, id);
                key
            },
        )
    }

    fn intern_with<T>(
        &self,
        hash: u64,
        key: T,
        eq: impl Fn(&K, &T) -> bool,
        make: impl FnOnce(T, u32) -> K,
    ) -> u32 {
        let slot = &self.cache[(hash as usize) & (INTERNER_CACHE - 1)];
        let hi = hash >> 32;
        let packed = slot.load(Ordering::Acquire);
        if packed != 0 && (packed >> 32) == hi {
            let id = (packed as u32) - 1;
            if eq(self.get(id), &key) {
                return id;
            }
        }
        let shard = &self.shards[shard_of::<SHARDS>(hash)].0;
        if let Some(id) = read_lock(shard).find(hash, |id| eq(self.get(id), &key)) {
            slot.store((hi << 32) | u64::from(id + 1), Ordering::Release);
            return id;
        }
        let mut write = write_lock(shard);
        if let Some(id) = write.find(hash, |id| eq(self.get(id), &key)) {
            slot.store((hi << 32) | u64::from(id + 1), Ordering::Release);
            return id;
        }
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        let owned = make(key, id);
        self.slabs.set(id as usize, owned);
        write.insert(hash, id, |old| hash_key(self.get(old)));
        slot.store((hi << 32) | u64::from(id + 1), Ordering::Release);
        id
    }

    #[inline]
    pub(crate) fn get(&self, id: u32) -> &K {
        self.slabs.get(id as usize).expect("interned entry exists")
    }

    pub(crate) fn len(&self) -> u32 {
        self.next.load(Ordering::Relaxed)
    }
}

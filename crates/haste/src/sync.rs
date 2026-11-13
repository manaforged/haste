use std::borrow::Borrow;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

use rustc_hash::{FxHashMap, FxHasher};

pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
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

pub(crate) fn shard_index<Q: Hash + ?Sized, const SHARDS: usize>(key: &Q) -> usize {
    let mut hasher = FxHasher::default();
    key.hash(&mut hasher);
    let hash = hasher.finish();
    (((hash >> 58) ^ (hash >> 29) ^ hash) as usize) & (SHARDS - 1)
}

pub(crate) struct Slab<T, const SLAB: usize, const SLABS: usize> {
    slabs: Box<[OnceLock<Box<[OnceLock<T>]>>]>,
}

impl<T, const SLAB: usize, const SLABS: usize> Default for Slab<T, SLAB, SLABS> {
    fn default() -> Self {
        Self { slabs: (0..SLABS).map(|_| OnceLock::new()).collect::<Vec<_>>().into_boxed_slice() }
    }
}

impl<T, const SLAB: usize, const SLABS: usize> Slab<T, SLAB, SLABS> {
    pub(crate) fn set(&self, index: usize, value: T) {
        assert!(index < SLAB * SLABS, "slab capacity exceeded");
        let slab = self.slabs[index / SLAB].get_or_init(|| (0..SLAB).map(|_| OnceLock::new()).collect::<Vec<_>>().into_boxed_slice());
        slab[index % SLAB].get_or_init(|| value);
    }

    pub(crate) fn get(&self, index: usize) -> Option<&T> {
        self.slabs.get(index / SLAB)?.get()?.get(index % SLAB)?.get()
    }
}

pub(crate) struct Interner<K, const SHARDS: usize, const SLAB: usize, const SLABS: usize> {
    shards: [RwLock<FxHashMap<K, u32>>; SHARDS],
    slabs: Slab<K, SLAB, SLABS>,
    next: AtomicU32,
}

impl<K, const SHARDS: usize, const SLAB: usize, const SLABS: usize> Default for Interner<K, SHARDS, SLAB, SLABS> {
    fn default() -> Self {
        Self { shards: std::array::from_fn(|_| RwLock::new(FxHashMap::default())), slabs: Slab::default(), next: AtomicU32::new(0) }
    }
}

impl<K: Eq + Hash + Clone, const SHARDS: usize, const SLAB: usize, const SLABS: usize> Interner<K, SHARDS, SLAB, SLABS> {
    pub(crate) fn intern<Q>(&self, key: &Q, make: impl FnOnce(&Q) -> K) -> u32
    where
        K: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        let shard = &self.shards[shard_index::<_, SHARDS>(key)];
        if let Some(&id) = read_lock(shard).get(key) {
            return id;
        }
        let mut write = write_lock(shard);
        if let Some(&id) = write.get(key) {
            return id;
        }
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        let owned = make(key);
        self.slabs.set(id as usize, owned.clone());
        write.insert(owned, id);
        id
    }

    pub(crate) fn get(&self, id: u32) -> &K {
        self.slabs.get(id as usize).expect("interned entry exists")
    }

    pub(crate) fn len(&self) -> u32 {
        self.next.load(Ordering::Relaxed)
    }
}

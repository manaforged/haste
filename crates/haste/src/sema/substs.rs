use std::sync::{Arc, RwLock};

use rustc_hash::FxHashMap;

use crate::names::ItemRef;
use crate::sync::{read_lock, shard_index, write_lock};

use super::table::SHARDS;
use super::ty::Ty;

fn pack_handle(shard: usize, index: usize) -> u32 {
    (((index as u32) << 5) | (shard as u32)) + 1
}

fn unpack_handle(raw: u32) -> Option<(usize, usize)> {
    let raw = raw.checked_sub(1)?;
    Some(((raw as usize) & (SHARDS - 1), (raw >> 5) as usize))
}

/// An interned slice of type arguments.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Substs(pub u32);

impl Substs {
    /// The empty slice of type arguments.
    pub const EMPTY: Substs = Substs(0);

    /// Whether this slice holds no type arguments.
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }
}

#[derive(Default)]
struct Shard {
    map: FxHashMap<&'static [Ty], Substs>,
    entries: Vec<*const [Ty]>,
}

// SAFETY: `Ty` is `Copy + Send + Sync`, and every pointer in `entries` owns an immutable heap slice freed only in `Shard::drop`.
unsafe impl Send for Shard {}
// SAFETY: `Shard` shares slice pointers only behind `&[Ty]` after `RwLock` synchronization.
unsafe impl Sync for Shard {}

impl Drop for Shard {
    fn drop(&mut self) {
        self.map.clear();
        for ptr in self.entries.drain(..) {
            // SAFETY: Each pointer in `entries` was created once by `Box::into_raw` in `SubstsPool::intern` and is dropped once here.
            unsafe { drop(Box::from_raw(ptr as *mut [Ty])) };
        }
    }
}

/// A sharded thread-safe interner of `[Ty]` slices.
#[derive(Clone)]
pub struct SubstsPool {
    shards: Arc<[RwLock<Shard>; SHARDS]>,
}

impl Default for SubstsPool {
    fn default() -> Self {
        Self { shards: Arc::new(std::array::from_fn(|_| RwLock::new(Shard::default()))) }
    }
}

impl SubstsPool {
    /// Interns `slice` and returns its handle, returning `Substs::EMPTY` without locking when empty.
    pub fn intern(&self, slice: &[Ty]) -> Substs {
        if slice.is_empty() {
            return Substs::EMPTY;
        }
        let shard_idx = shard_index::<_, SHARDS>(slice);
        let lock = &self.shards[shard_idx];
        let guard = read_lock(lock);
        if let Some(&id) = guard.map.get(slice) {
            return id;
        }
        drop(guard);
        let mut guard = write_lock(lock);
        if let Some(&id) = guard.map.get(slice) {
            return id;
        }
        let id = Substs(pack_handle(shard_idx, guard.entries.len()));
        let boxed: Box<[Ty]> = slice.into();
        let raw: *const [Ty] = Box::into_raw(boxed);
        // SAFETY: `raw` points to a heap allocation that stays immutable and alive until `Shard::drop`.
        let key: &'static [Ty] = unsafe { &*raw };
        guard.entries.push(raw);
        guard.map.insert(key, id);
        id
    }

    /// The slice of type arguments that `id` names.
    pub fn get(&self, id: Substs) -> &[Ty] {
        let Some((shard_idx, index)) = unpack_handle(id.0) else { return &[] };
        let guard = read_lock(&self.shards[shard_idx]);
        let Some(&ptr) = guard.entries.get(index) else { return &[] };
        // SAFETY: `ptr` belongs to `self.shards`, whose `Arc` outlives `&self`, and its `[Ty]` elements never mutate.
        unsafe { &*ptr }
    }

    /// Maps `f` across `id`, reusing `id` without allocating when every element is unchanged.
    pub fn map(&self, id: Substs, f: &mut impl FnMut(Ty) -> Ty) -> Substs {
        if id.is_empty() {
            return Substs::EMPTY;
        }
        let slice = self.get(id);
        for (idx, &ty) in slice.iter().enumerate() {
            let mapped = f(ty);
            if mapped != ty {
                let mut out = Vec::with_capacity(slice.len());
                out.extend_from_slice(&slice[..idx]);
                out.push(mapped);
                out.extend(slice[idx + 1..].iter().map(|&rest| f(rest)));
                return self.intern(&out);
            }
        }
        id
    }
}

/// An interned unnormalized associated type projection.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProjId(pub u32);

/// The trait, `Self` type, trait arguments, associated type item and GAT arguments of a projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ProjDef {
    pub of: ItemRef,
    pub self_ty: Ty,
    pub args: Substs,
    pub assoc: ItemRef,
    pub own: Substs,
}

#[derive(Default)]
struct ProjShard {
    map: FxHashMap<ProjDef, ProjId>,
    entries: Vec<ProjDef>,
}

/// A sharded thread-safe interner of associated type projections.
#[derive(Clone)]
pub struct ProjPool {
    shards: Arc<[RwLock<ProjShard>; SHARDS]>,
}

impl Default for ProjPool {
    fn default() -> Self {
        Self { shards: Arc::new(std::array::from_fn(|_| RwLock::new(ProjShard::default()))) }
    }
}

impl ProjPool {
    /// Interns `def` and returns its handle.
    pub fn intern(&self, def: ProjDef) -> ProjId {
        let shard_idx = shard_index::<_, SHARDS>(&def);
        let lock = &self.shards[shard_idx];
        let guard = read_lock(lock);
        if let Some(&id) = guard.map.get(&def) {
            return id;
        }
        drop(guard);
        let mut guard = write_lock(lock);
        if let Some(&id) = guard.map.get(&def) {
            return id;
        }
        let id = ProjId(pack_handle(shard_idx, guard.entries.len()));
        guard.entries.push(def);
        guard.map.insert(def, id);
        id
    }

    /// The projection definition that `id` names.
    pub fn get(&self, id: ProjId) -> ProjDef {
        let (shard_idx, index) = unpack_handle(id.0).unwrap_or((0, 0));
        let guard = read_lock(&self.shards[shard_idx]);
        guard.entries[index]
    }
}

/// Cloned handles to the slice and projection interners of `Types`.
#[derive(Clone, Default)]
pub struct Pools {
    pub substs: SubstsPool,
    pub projs: ProjPool,
}

/// A sorted immutable key-value table backed by a single boxed slice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedMap<K, V>(Box<[(K, V)]>);

impl<K, V> Default for PackedMap<K, V> {
    fn default() -> Self {
        Self(Box::default())
    }
}

impl<K: Ord, V> PackedMap<K, V> {
    /// Packs `(K, V)` pairs from `iter` into a key-sorted slice, keeping the last value on duplicate keys.
    pub fn pack(iter: impl IntoIterator<Item = (K, V)>) -> Self {
        let mut vec: Vec<(K, V)> = iter.into_iter().collect();
        if vec.is_empty() {
            return Self::default();
        }
        vec.sort_by(|a, b| a.0.cmp(&b.0));
        vec.dedup_by(|a, b| {
            if a.0 == b.0 {
                std::mem::swap(&mut a.1, &mut b.1);
                true
            } else {
                false
            }
        });
        Self(vec.into_boxed_slice())
    }

    /// Wraps an already strictly sorted vector of pairs without re-sorting.
    pub fn from_sorted(vec: Vec<(K, V)>) -> Self {
        if vec.is_empty() { Self::default() } else { Self(vec.into_boxed_slice()) }
    }

    /// The value stored under `key`, found by binary search.
    pub fn get(&self, key: &K) -> Option<&V> {
        let idx = self.0.binary_search_by(|probe| probe.0.cmp(key)).ok()?;
        Some(&self.0[idx].1)
    }

    /// Whether `key` has an entry in this table.
    pub fn contains_key(&self, key: &K) -> bool {
        self.0.binary_search_by(|probe| probe.0.cmp(key)).is_ok()
    }
}

impl<K: Copy, V> PackedMap<K, V> {
    /// Maps values in sorted order without re-sorting keys.
    pub fn try_map_values<U, E>(&self, mut f: impl FnMut(&V) -> Result<U, E>) -> Result<PackedMap<K, U>, E> {
        if self.0.is_empty() {
            return Ok(PackedMap::default());
        }
        let mut out = Vec::with_capacity(self.0.len());
        for (key, val) in &self.0 {
            out.push((*key, f(val)?));
        }
        Ok(PackedMap(out.into_boxed_slice()))
    }
}

impl<K, V> PackedMap<K, V> {
    /// Whether this table holds no entries.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// How many entries this table holds.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Iterates over `(&K, &V)` pairs in key order.
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.0.iter().map(|(k, v)| (k, v))
    }

    /// Iterates over `&V` values in key order.
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.0.iter().map(|(_, v)| v)
    }

    /// Iterates mutably over `&mut V` values in key order.
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut V> {
        self.0.iter_mut().map(|(_, v)| v)
    }

    /// Borrows the underlying sorted `(K, V)` slice.
    pub fn as_slice(&self) -> &[(K, V)] {
        &self.0
    }
}

impl<'a, K, V> IntoIterator for &'a PackedMap<K, V> {
    type Item = (&'a K, &'a V);
    type IntoIter = std::iter::Map<std::slice::Iter<'a, (K, V)>, fn(&'a (K, V)) -> (&'a K, &'a V)>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter().map(|(k, v)| (k, v))
    }
}

/// A sorted immutable set backed by a single boxed slice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedSet<K>(Box<[K]>);

impl<K> Default for PackedSet<K> {
    fn default() -> Self {
        Self(Box::default())
    }
}

impl<K: Ord> PackedSet<K> {
    /// Packs elements from `iter` into a sorted deduplicated slice.
    pub fn pack(iter: impl IntoIterator<Item = K>) -> Self {
        let mut vec: Vec<K> = iter.into_iter().collect();
        if vec.is_empty() {
            return Self::default();
        }
        vec.sort();
        vec.dedup();
        Self(vec.into_boxed_slice())
    }

    /// Whether `key` belongs to this set.
    pub fn contains(&self, key: &K) -> bool {
        self.0.binary_search(key).is_ok()
    }
}

impl<K> PackedSet<K> {
    /// Borrows the underlying sorted element slice.
    pub fn as_slice(&self) -> &[K] {
        &self.0
    }
}

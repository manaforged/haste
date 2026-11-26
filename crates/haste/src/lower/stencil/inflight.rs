use std::sync::Mutex;

use rustc_hash::FxHashMap;

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

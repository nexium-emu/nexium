//! Bounded LRU cache of decompressed RomFS compression blocks.
//!
//! One process-wide cache serves every compressed storage, so the memory bound
//! holds however many archives are mounted. Its size comes from
//! `NEXIUM_ROMFS_CACHE_MIB` (default 64 MiB, 0 disables caching).

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

pub const DEFAULT_CAPACITY_MIB: usize = 64;

/// (storage id, compression entry index)
pub(crate) type BlockKey = (u64, usize);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BlockCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub entries: usize,
    pub bytes: usize,
    pub capacity: usize,
}

pub struct BlockCache {
    capacity: usize,
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    blocks: HashMap<BlockKey, (Arc<[u8]>, u64)>,
    recency: BTreeMap<u64, BlockKey>,
    tick: u64,
    bytes: usize,
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl BlockCache {
    pub fn new(capacity: usize) -> Self {
        Self { capacity, state: Mutex::new(State::default()) }
    }

    /// The shared cache every compressed RomFS uses unless a test swaps it.
    pub fn global() -> Arc<BlockCache> {
        static GLOBAL: OnceLock<Arc<BlockCache>> = OnceLock::new();
        GLOBAL
            .get_or_init(|| {
                let mib = std::env::var("NEXIUM_ROMFS_CACHE_MIB")
                    .ok()
                    .and_then(|value| value.trim().parse::<usize>().ok())
                    .unwrap_or(DEFAULT_CAPACITY_MIB);
                Arc::new(Self::new(mib.saturating_mul(1 << 20)))
            })
            .clone()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn stats(&self) -> BlockCacheStats {
        let state = self.lock();
        BlockCacheStats {
            hits: state.hits,
            misses: state.misses,
            evictions: state.evictions,
            entries: state.blocks.len(),
            bytes: state.bytes,
            capacity: self.capacity,
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub(crate) fn get(&self, key: BlockKey) -> Option<Arc<[u8]>> {
        let mut state = self.lock();
        let state = &mut *state;
        state.tick += 1;
        let tick = state.tick;
        let Some((block, last)) = state.blocks.get_mut(&key) else {
            state.misses += 1;
            return None;
        };
        state.recency.remove(last);
        *last = tick;
        state.recency.insert(tick, key);
        state.hits += 1;
        Some(Arc::clone(block))
    }

    /// Caches `block`, evicting the least recently used blocks to stay
    /// within capacity. A block larger than the whole cache is not kept.
    pub(crate) fn insert(&self, key: BlockKey, block: Arc<[u8]>) {
        if block.len() > self.capacity {
            return;
        }
        let mut state = self.lock();
        let state = &mut *state;
        state.tick += 1;
        let tick = state.tick;
        if let Some((old, last)) = state.blocks.insert(key, (Arc::clone(&block), tick)) {
            state.recency.remove(&last);
            state.bytes -= old.len();
        }
        state.recency.insert(tick, key);
        state.bytes += block.len();
        while state.bytes > self.capacity {
            let Some((_, victim)) = state.recency.pop_first() else { break };
            if let Some((evicted, _)) = state.blocks.remove(&victim) {
                state.bytes -= evicted.len();
                state.evictions += 1;
            }
        }
    }
}

/// A process-unique id for each compressed storage, so cache keys of two
/// archives never collide.
pub(crate) fn next_storage_id() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(byte: u8, len: usize) -> Arc<[u8]> {
        vec![byte; len].into()
    }

    #[test]
    fn evicts_least_recently_used_blocks_to_stay_within_capacity() {
        let cache = BlockCache::new(300);
        cache.insert((1, 0), block(0, 100));
        cache.insert((1, 1), block(1, 100));
        cache.insert((1, 2), block(2, 100));
        assert!(cache.get((1, 0)).is_some());
        cache.insert((1, 3), block(3, 100));
        assert!(cache.get((1, 1)).is_none(), "the least recently used block goes first");
        assert_eq!(&*cache.get((1, 0)).unwrap(), &[0; 100][..]);
        assert!(cache.get((1, 2)).is_some());
        assert!(cache.get((1, 3)).is_some());
        let stats = cache.stats();
        assert_eq!((stats.entries, stats.bytes, stats.evictions), (3, 300, 1));
        cache.insert((2, 0), block(9, 301));
        assert!(cache.get((2, 0)).is_none(), "an oversized block is never cached");
        assert_eq!(cache.stats().bytes, 300);
    }

    #[test]
    fn reinserting_a_key_replaces_it_without_double_counting() {
        let cache = BlockCache::new(1000);
        cache.insert((7, 7), block(1, 400));
        cache.insert((7, 7), block(2, 300));
        assert_eq!(cache.stats().bytes, 300);
        assert_eq!(cache.get((7, 7)).unwrap()[0], 2);
        let disabled = BlockCache::new(0);
        disabled.insert((1, 1), block(1, 1));
        assert!(disabled.get((1, 1)).is_none());
    }
}

use std::sync::{Arc, Mutex, OnceLock, Weak};

use nexium_gpu::compute::ComputeRawStorageKey;
use nexium_gpu::draw::GraphicsStorageResidentRef;

const MAX_OWNED_RANGES: usize = 32;
const MAX_RANGE_BYTES: u64 = 4 * 1024 * 1024;
const MAX_VERIFIED_SNAPSHOTS: usize = 8;

struct OwnedRange {
    key: ComputeRawStorageKey,
    generation: u64,
    seed: Arc<Vec<u8>>,
    verified: Vec<Weak<Vec<u8>>>,
    last_use: u64,
}

impl OwnedRange {
    fn is_verified(&self, snapshot: &Arc<Vec<u8>>) -> bool {
        self.verified
            .iter()
            .any(|weak| std::ptr::eq(weak.as_ptr(), Arc::as_ptr(snapshot)))
    }

    fn remember(&mut self, snapshot: &Arc<Vec<u8>>) {
        if self.is_verified(snapshot) {
            return;
        }
        self.verified.retain(|weak| weak.strong_count() != 0);
        if self.verified.len() >= MAX_VERIFIED_SNAPSHOTS {
            self.verified.remove(0);
        }
        self.verified.push(Arc::downgrade(snapshot));
    }

    fn matches_snapshot(&self, offset: u64, snapshot_offset: usize, snapshot: &[u8]) -> bool {
        let Some(start) = usize::try_from(offset)
            .ok()
            .and_then(|offset| offset.checked_add(snapshot_offset))
        else {
            return false;
        };
        start
            .checked_add(snapshot.len())
            .and_then(|end| self.seed.get(start..end))
            .is_some_and(|seed| seed == snapshot)
    }
}

#[derive(Default)]
struct Registry {
    owned: Vec<OwnedRange>,
    next_generation: u64,
    uses: u64,
}

impl Registry {
    fn next_generation(&mut self) -> u64 {
        self.next_generation = self.next_generation.wrapping_add(1).max(1);
        self.next_generation
    }

    fn overlapping(&self, key: ComputeRawStorageKey) -> Vec<usize> {
        self.owned
            .iter()
            .enumerate()
            .filter(|(_, owned)| keys_overlap(owned.key, key))
            .map(|(index, _)| index)
            .collect()
    }

    fn remove_indices(&mut self, mut indices: Vec<usize>) -> Vec<OwnedRange> {
        indices.sort_unstable_by(|a, b| b.cmp(a));
        indices
            .into_iter()
            .map(|index| self.owned.swap_remove(index))
            .collect()
    }
}

fn registry() -> &'static Mutex<Registry> {
    static REGISTRY: OnceLock<Mutex<Registry>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(Registry::default()))
}

pub(crate) fn residency_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("NEXIUM_GRAPHICS_STORAGE_RESIDENCY").ok().as_deref(),
            Some("0" | "false" | "off" | "no")
        )
    })
}

fn trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("NEXIUM_GRAPHICS_STORAGE_TRACE").is_some())
}

fn ranges_overlap(a: u64, a_len: u64, b: u64, b_len: u64) -> bool {
    a_len != 0 && b_len != 0 && a < b.saturating_add(b_len) && b < a.saturating_add(a_len)
}

fn keys_overlap(a: ComputeRawStorageKey, b: ComputeRawStorageKey) -> bool {
    ranges_overlap(a.gpu_va, a.size, b.gpu_va, b.size)
        || ranges_overlap(a.cpu_addr, a.size, b.cpu_addr, b.size)
}

fn contained_offset(outer: ComputeRawStorageKey, inner: ComputeRawStorageKey) -> Option<u64> {
    let offset = inner.gpu_va.checked_sub(outer.gpu_va)?;
    let end = offset.checked_add(inner.size)?;
    (end <= outer.size
        && offset % 16 == 0
        && outer.mapping_epoch == inner.mapping_epoch
        && outer.nvmap_id == inner.nvmap_id
        && outer.cpu_addr.checked_add(offset) == Some(inner.cpu_addr))
    .then_some(offset)
}

fn note_dropped(reason: &str, key: ComputeRawStorageKey, dropped: &[OwnedRange]) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static DROPS: AtomicU64 = AtomicU64::new(0);
    let drops = DROPS.fetch_add(dropped.len() as u64, Ordering::Relaxed) + dropped.len() as u64;
    if trace_enabled() || drops <= 8 {
        for owned in dropped {
            log::warn!(
                "[graphics-storage] {reason}: dropped gpu={:#x}+{:#x} for gpu={:#x}+{:#x} (drops={drops})",
                owned.key.gpu_va,
                owned.key.size,
                key.gpu_va,
                key.size,
            );
        }
    }
}

pub(crate) fn bind(
    key: ComputeRawStorageKey,
    writes: bool,
    snapshot: &Arc<Vec<u8>>,
    snapshot_offset: usize,
    read_range: &mut dyn FnMut(ComputeRawStorageKey) -> Option<Arc<Vec<u8>>>,
) -> Option<GraphicsStorageResidentRef> {
    if key.size == 0 || key.size > MAX_RANGE_BYTES {
        return None;
    }
    let mut registry = registry().lock().unwrap_or_else(|error| error.into_inner());
    registry.uses = registry.uses.wrapping_add(1);
    let uses = registry.uses;
    let overlapping = registry.overlapping(key);
    if let [index] = overlapping[..] {
        if let Some(offset) = contained_offset(registry.owned[index].key, key) {
            let stale = {
                let owned = &registry.owned[index];
                !owned.is_verified(snapshot)
                    && !owned.matches_snapshot(offset, snapshot_offset, snapshot)
            };
            if stale {
                let owned_key = registry.owned[index].key;
                let Some(seed) = read_range(owned_key) else {
                    let dropped = registry.remove_indices(vec![index]);
                    note_dropped("reseed read failed", key, &dropped);
                    return None;
                };
                let generation = registry.next_generation();
                let owned = &mut registry.owned[index];
                owned.generation = generation;
                owned.seed = seed;
                owned.verified.clear();
                if trace_enabled() {
                    log::warn!(
                        "[graphics-storage] reseed gpu={:#x}+{:#x} generation={} after guest write",
                        owned_key.gpu_va,
                        owned_key.size,
                        generation,
                    );
                }
            }
            let owned = &mut registry.owned[index];
            if owned.matches_snapshot(offset, snapshot_offset, snapshot) {
                owned.remember(snapshot);
            }
            owned.last_use = uses;
            return Some(GraphicsStorageResidentRef {
                key: owned.key,
                generation: owned.generation,
                offset,
                writes,
                seed: Some(Arc::clone(&owned.seed)),
            });
        }
    }
    if !overlapping.is_empty() {
        let dropped = registry.remove_indices(overlapping);
        note_dropped("layout conflict", key, &dropped);
    }
    if !writes {
        return None;
    }
    let seed = read_range(key)?;
    if seed.len() as u64 != key.size {
        return None;
    }
    if registry.owned.len() >= MAX_OWNED_RANGES {
        if let Some(oldest) = registry
            .owned
            .iter()
            .enumerate()
            .min_by_key(|(_, owned)| owned.last_use)
            .map(|(index, _)| index)
        {
            let dropped = registry.remove_indices(vec![oldest]);
            note_dropped("capacity", key, &dropped);
        }
    }
    let generation = registry.next_generation();
    let mut owned = OwnedRange {
        key,
        generation,
        seed: Arc::clone(&seed),
        verified: Vec::new(),
        last_use: uses,
    };
    if owned.matches_snapshot(0, snapshot_offset, snapshot) {
        owned.remember(snapshot);
    }
    registry.owned.push(owned);
    if trace_enabled() {
        log::warn!(
            "[graphics-storage] own gpu={:#x}+{:#x} cpu={:#x} generation={}",
            key.gpu_va,
            key.size,
            key.cpu_addr,
            generation,
        );
    }
    Some(GraphicsStorageResidentRef {
        key,
        generation,
        offset: 0,
        writes,
        seed: Some(seed),
    })
}

pub(crate) fn note_seed_read(
    key: ComputeRawStorageKey,
    pending_before: bool,
    pending_after: bool,
    read: bool,
    seed: &[u8],
) {
    if trace_enabled() {
        let nonzero_words = seed.chunks_exact(4).filter(|word| *word != [0u8; 4]).count();
        log::warn!(
            "[graphics-storage] seed read gpu={:#x}+{:#x} pending_before={pending_before} pending_after={pending_after} read={read} nonzero_words={nonzero_words}/{}",
            key.gpu_va,
            key.size,
            seed.len() / 4,
        );
    }
}

pub(crate) fn note_writeback(key: ComputeRawStorageKey, bytes: &[u8], published: bool) {
    if trace_enabled() {
        let nonzero_words = bytes.chunks_exact(4).filter(|word| *word != [0u8; 4]).count();
        log::warn!(
            "[graphics-storage] writeback gpu={:#x}+{:#x} published={published} nonzero_words={nonzero_words}/{}",
            key.gpu_va,
            key.size,
            bytes.len() / 4,
        );
    }
}

pub(crate) fn note_unresident_writer(
    gpu_va: u64,
    size: usize,
    binding: usize,
    indirect: bool,
    unknown_writes: bool,
    has_snapshot: bool,
) {
    if trace_enabled() {
        log::warn!(
            "[graphics-storage] writer not resident gpu={gpu_va:#x}+{size:#x} binding={binding} indirect={indirect} unknown_writes={unknown_writes} snapshot={has_snapshot}"
        );
    }
}

pub(crate) fn take_for_compute(key: ComputeRawStorageKey) -> Option<(ComputeRawStorageKey, u64)> {
    let mut registry = registry().lock().unwrap_or_else(|error| error.into_inner());
    if registry.owned.is_empty() {
        return None;
    }
    let overlapping = registry.overlapping(key);
    if overlapping.is_empty() {
        return None;
    }
    let dropped = registry.remove_indices(overlapping);
    if let [owned] = &dropped[..] {
        if owned.key == key {
            if trace_enabled() {
                log::warn!(
                    "[graphics-storage] compute adopts gpu={:#x}+{:#x} generation={}",
                    key.gpu_va,
                    key.size,
                    owned.generation,
                );
            }
            return Some((owned.key, owned.generation));
        }
    }
    note_dropped("compute overlap", key, &dropped);
    None
}

pub(crate) fn release_overlapping(gpu_va: u64, cpu_addr: u64, size: u64, reason: &str) {
    let key = ComputeRawStorageKey {
        mapping_epoch: 0,
        nvmap_id: 0,
        gpu_va,
        cpu_addr,
        size,
    };
    let mut registry = registry().lock().unwrap_or_else(|error| error.into_inner());
    if registry.owned.is_empty() {
        return;
    }
    let overlapping = registry.overlapping(key);
    if overlapping.is_empty() {
        return;
    }
    let dropped = registry.remove_indices(overlapping);
    note_dropped(reason, key, &dropped);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: Mutex<()> = Mutex::new(());
        let guard = LOCK.lock().unwrap_or_else(|error| error.into_inner());
        *registry().lock().unwrap_or_else(|error| error.into_inner()) = Registry::default();
        guard
    }

    fn key(gpu_va: u64, cpu_addr: u64, size: u64) -> ComputeRawStorageKey {
        ComputeRawStorageKey {
            mapping_epoch: 7,
            nvmap_id: 3,
            gpu_va,
            cpu_addr,
            size,
        }
    }

    fn bytes(len: usize, value: u8) -> Arc<Vec<u8>> {
        Arc::new(vec![value; len])
    }

    #[test]
    fn writer_owns_range_and_contained_reader_binds_at_offset() {
        let _guard = test_lock();
        let owner = key(0x5000_0000, 0x1_0000, 0x100);
        let guest = bytes(0x100, 0);
        let writer = bind(owner, true, &guest, 0, &mut |range| {
            assert_eq!(range, owner);
            Some(bytes(0x100, 0))
        })
        .expect("writer takes ownership");
        assert_eq!((writer.key, writer.offset, writer.writes), (owner, 0, true));
        assert_eq!(writer.seed.as_deref().map(Vec::len), Some(0x100));

        let reader_key = key(0x5000_0040, 0x1_0040, 0x80);
        let reader_snapshot = bytes(0x80, 0);
        let reader = bind(reader_key, false, &reader_snapshot, 0, &mut |_| {
            panic!("an unchanged reader must not reseed")
        })
        .expect("contained reader uses the owner");
        assert_eq!((reader.key, reader.offset, reader.writes), (owner, 0x40, false));
        assert_eq!(reader.generation, writer.generation);
    }

    #[test]
    fn reader_without_owner_keeps_its_snapshot() {
        let _guard = test_lock();
        let snapshot = bytes(0x40, 1);
        assert!(bind(key(0x6000_0000, 0x2_0000, 0x40), false, &snapshot, 0, &mut |_| {
            panic!("readers never seed ownership")
        })
        .is_none());
    }

    #[test]
    fn changed_guest_bytes_reseed_with_a_new_generation() {
        let _guard = test_lock();
        let owner = key(0x5000_0000, 0x1_0000, 0x40);
        let first = bind(owner, true, &bytes(0x40, 0), 0, &mut |_| Some(bytes(0x40, 0))).unwrap();
        let unchanged = bind(owner, true, &bytes(0x40, 0), 0, &mut |_| {
            panic!("matching guest bytes keep GPU ownership")
        })
        .unwrap();
        assert_eq!(unchanged.generation, first.generation);

        let reseeded = bind(owner, true, &bytes(0x40, 9), 0, &mut |_| Some(bytes(0x40, 9))).unwrap();
        assert_ne!(reseeded.generation, first.generation);
        assert_eq!(reseeded.seed.as_deref(), Some(&vec![9u8; 0x40]));
    }

    #[test]
    fn verified_snapshot_skips_the_byte_comparison() {
        let _guard = test_lock();
        let owner = key(0x5000_0000, 0x1_0000, 0x40);
        let snapshot = bytes(0x40, 4);
        let first = bind(owner, true, &snapshot, 0, &mut |_| Some(bytes(0x40, 4))).unwrap();
        let again = bind(owner, false, &snapshot, 0, &mut |_| panic!("verified snapshot reseeded")).unwrap();
        assert_eq!(again.generation, first.generation);
    }

    #[test]
    fn compute_adopts_an_exact_owner_and_drops_overlaps() {
        let _guard = test_lock();
        let owner = key(0x5000_0000, 0x1_0000, 0x40);
        let owned = bind(owner, true, &bytes(0x40, 0), 0, &mut |_| Some(bytes(0x40, 0))).unwrap();
        assert_eq!(take_for_compute(owner), Some((owner, owned.generation)));
        assert_eq!(take_for_compute(owner), None);
        assert!(bind(owner, false, &bytes(0x40, 0), 0, &mut |_| panic!("no owner")).is_none());

        bind(owner, true, &bytes(0x40, 0), 0, &mut |_| Some(bytes(0x40, 0))).unwrap();
        assert_eq!(take_for_compute(key(0x5000_0020, 0x1_0020, 0x40)), None);
        assert!(bind(owner, false, &bytes(0x40, 0), 0, &mut |_| panic!("no owner")).is_none());
    }

    #[test]
    fn misaligned_reader_drops_the_owner_instead_of_binding_stale_offsets() {
        let _guard = test_lock();
        let owner = key(0x5000_0000, 0x1_0000, 0x100);
        bind(owner, true, &bytes(0x100, 0), 0, &mut |_| Some(bytes(0x100, 0))).unwrap();
        let misaligned = key(0x5000_0008, 0x1_0008, 0x20);
        assert!(bind(misaligned, false, &bytes(0x20, 0), 0, &mut |_| panic!("no seed")).is_none());
        assert!(bind(owner, false, &bytes(0x100, 0), 0, &mut |_| panic!("owner was dropped")).is_none());
    }

    #[test]
    fn released_ranges_no_longer_bind() {
        let _guard = test_lock();
        let owner = key(0x5000_0000, 0x1_0000, 0x40);
        bind(owner, true, &bytes(0x40, 0), 0, &mut |_| Some(bytes(0x40, 0))).unwrap();
        release_overlapping(0x5000_0010, 0x1_0010, 4, "test");
        assert!(bind(owner, false, &bytes(0x40, 0), 0, &mut |_| panic!("released")).is_none());
    }
}

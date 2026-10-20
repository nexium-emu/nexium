use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use nexium_gpu::compute::ComputeRawStorageKey;
use nexium_gpu::draw::{ComputeOverlay, GraphicsStorageResidentRef};

const MAX_OWNED_RANGES: usize = 32;
const MAX_RANGE_BYTES: u64 = 4 * 1024 * 1024;
const MAX_VERIFIED_SNAPSHOTS: usize = 8;

static PENDING_COMPUTE_SEEDS: AtomicUsize = AtomicUsize::new(0);
const MAX_LANDED_SEEDS: usize = 64;
const MAX_LANDED_SEED_BYTES: u64 = 16 * 1024 * 1024;
const STORAGE_OFFSET_ALIGNMENT: u64 = 256;

struct LandedSeed {
    cpu_addr: u64,
    serial: u64,
    resource_index: u32,
    bytes: Arc<Vec<u8>>,
}

fn landed_seeds() -> &'static Mutex<Vec<LandedSeed>> {
    static LANDED: OnceLock<Mutex<Vec<LandedSeed>>> = OnceLock::new();
    LANDED.get_or_init(|| Mutex::new(Vec::new()))
}

struct OwnedRange {
    key: ComputeRawStorageKey,
    generation: u64,
    seed: Arc<Vec<u8>>,
    seed_pending: bool,
    compute_seed: Option<ComputeOverlay>,
    verified: Vec<Weak<Vec<u8>>>,
    last_use: u64,
}

impl OwnedRange {
    fn is_verified(&self, snapshot: &Arc<Vec<u8>>) -> bool {
        self.verified
            .iter()
            .any(|weak| std::ptr::eq(weak.as_ptr(), Arc::as_ptr(snapshot)))
    }

    fn resident_ref(&self, offset: u64, writes: bool) -> GraphicsStorageResidentRef {
        GraphicsStorageResidentRef {
            key: self.key,
            generation: self.generation,
            offset,
            writes,
            seed: (!self.seed_pending).then(|| Arc::clone(&self.seed)),
            compute_seed: self.compute_seed,
        }
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
        let removed = indices
            .into_iter()
            .map(|index| self.owned.swap_remove(index))
            .collect();
        self.sync_pending_seeds();
        removed
    }

    fn apply_landed_seeds(&mut self) {
        let landed = landed_seeds().lock().unwrap_or_else(|error| error.into_inner());
        if landed.is_empty() || !self.owned.iter().any(|owned| owned.seed_pending) {
            return;
        }
        let mut filled = false;
        for owned in self.owned.iter_mut().filter(|owned| owned.seed_pending) {
            let Some(source) = owned.compute_seed else {
                continue;
            };
            let seed = landed.iter().rev().find_map(|landed| {
                if landed.serial != source.serial || landed.resource_index != source.resource_index {
                    return None;
                }
                let start = usize::try_from(owned.key.cpu_addr.checked_sub(landed.cpu_addr)?).ok()?;
                let len = usize::try_from(owned.key.size).ok()?;
                landed.bytes.get(start..start.checked_add(len)?)
            });
            if let Some(seed) = seed {
                owned.seed = Arc::new(seed.to_vec());
                owned.seed_pending = false;
                owned.verified.clear();
                filled = true;
            }
        }
        drop(landed);
        if filled {
            self.sync_pending_seeds();
        }
    }

    fn sync_pending_seeds(&self) {
        let pending = self.owned.iter().filter(|owned| owned.seed_pending).count();
        PENDING_COMPUTE_SEEDS.store(pending, Ordering::Release);
    }

    fn evict_for_capacity(&mut self, key: ComputeRawStorageKey) {
        if self.owned.len() < MAX_OWNED_RANGES {
            return;
        }
        if let Some(oldest) = self
            .owned
            .iter()
            .enumerate()
            .min_by_key(|(_, owned)| owned.last_use)
            .map(|(index, _)| index)
        {
            let dropped = self.remove_indices(vec![oldest]);
            note_dropped("capacity", key, &dropped);
        }
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
    registry.apply_landed_seeds();
    registry.uses = registry.uses.wrapping_add(1);
    let uses = registry.uses;
    let overlapping = registry.overlapping(key);
    if let [index] = overlapping[..] {
        if let Some(offset) = contained_offset(registry.owned[index].key, key) {
            let stale = {
                let owned = &registry.owned[index];
                !owned.seed_pending
                    && !owned.is_verified(snapshot)
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
            if !owned.seed_pending && owned.matches_snapshot(offset, snapshot_offset, snapshot) {
                owned.remember(snapshot);
            }
            owned.last_use = uses;
            return Some(owned.resident_ref(offset, writes));
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
    registry.evict_for_capacity(key);
    let generation = registry.next_generation();
    let mut owned = OwnedRange {
        key,
        generation,
        seed: Arc::clone(&seed),
        seed_pending: false,
        compute_seed: None,
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
        compute_seed: None,
    })
}

pub(crate) fn has_pending_compute_seeds() -> bool {
    PENDING_COMPUTE_SEEDS.load(Ordering::Acquire) != 0
}

pub(crate) fn bind_compute_output(
    key: ComputeRawStorageKey,
    writes: bool,
    source: Option<ComputeOverlay>,
) -> Option<GraphicsStorageResidentRef> {
    if key.size == 0 || key.size > MAX_RANGE_BYTES {
        return None;
    }
    let mut registry = registry().lock().unwrap_or_else(|error| error.into_inner());
    registry.apply_landed_seeds();
    registry.uses = registry.uses.wrapping_add(1);
    let uses = registry.uses;
    let overlapping = registry.overlapping(key);
    if let [index] = overlapping[..] {
        if let Some(offset) = contained_offset(registry.owned[index].key, key)
            .filter(|offset| offset % STORAGE_OFFSET_ALIGNMENT == 0)
        {
            let owned = &mut registry.owned[index];
            let same_source = source.is_some_and(|source| {
                owned.compute_seed.is_some_and(|seed| {
                    seed.serial == source.serial && seed.resource_index == source.resource_index
                })
            });
            if owned.seed_pending || same_source {
                owned.last_use = uses;
                return Some(owned.resident_ref(offset, writes));
            }
        }
    }
    if overlapping.iter().any(|&index| registry.owned[index].seed_pending) {
        return None;
    }
    let source = source.filter(|source| writes && source.dst_offset == 0 && source.len == key.size)?;
    if !overlapping.is_empty() {
        let dropped = registry.remove_indices(overlapping);
        note_dropped("compute adoption", key, &dropped);
    }
    registry.evict_for_capacity(key);
    let generation = registry.next_generation();
    let owned = OwnedRange {
        key,
        generation,
        seed: Arc::new(Vec::new()),
        seed_pending: true,
        compute_seed: Some(source),
        verified: Vec::new(),
        last_use: uses,
    };
    let resident = owned.resident_ref(0, writes);
    registry.owned.push(owned);
    registry.sync_pending_seeds();
    registry.apply_landed_seeds();
    if trace_enabled() {
        log::warn!(
            "[graphics-storage] adopt compute gpu={:#x}+{:#x} cpu={:#x} generation={} serial={} resource={}",
            key.gpu_va,
            key.size,
            key.cpu_addr,
            generation,
            source.serial,
            source.resource_index,
        );
    }
    Some(resident)
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

pub(crate) fn note_writeback(
    key: ComputeRawStorageKey,
    serial: u64,
    resource_index: u32,
    bytes: &[u8],
    published: bool,
) {
    if published && serial != 0 && bytes.len() as u64 <= MAX_LANDED_SEED_BYTES {
        {
            let mut landed = landed_seeds().lock().unwrap_or_else(|error| error.into_inner());
            landed.push(LandedSeed {
                cpu_addr: key.cpu_addr,
                serial,
                resource_index,
                bytes: Arc::new(bytes.to_vec()),
            });
            while landed.len() > MAX_LANDED_SEEDS
                || landed.iter().map(|seed| seed.bytes.len() as u64).sum::<u64>() > MAX_LANDED_SEED_BYTES
            {
                landed.remove(0);
            }
        }
        if let Ok(mut registry) = registry().try_lock() {
            registry.apply_landed_seeds();
        }
    }
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

pub(crate) fn reap_stale_compute_seeds(tracked: impl Fn(u64) -> bool) {
    let queued: Vec<u64> = landed_seeds()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .iter()
        .map(|landed| landed.serial)
        .collect();
    if PENDING_COMPUTE_SEEDS.load(Ordering::Acquire) == 0 {
        if !queued.is_empty() {
            let dead: Vec<u64> = queued.into_iter().filter(|serial| !tracked(*serial)).collect();
            landed_seeds()
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .retain(|landed| !dead.contains(&landed.serial));
        }
        return;
    }
    let mut registry = registry().lock().unwrap_or_else(|error| error.into_inner());
    registry.apply_landed_seeds();
    let stale: Vec<usize> = registry
        .owned
        .iter()
        .enumerate()
        .filter(|(_, owned)| {
            owned.seed_pending
                && owned.compute_seed.is_none_or(|source| !tracked(source.serial))
        })
        .map(|(index, _)| index)
        .collect();
    if let Some(&first) = stale.first() {
        let key = registry.owned[first].key;
        let dropped = registry.remove_indices(stale);
        note_dropped("compute seed never landed", key, &dropped);
    }
    drop(registry);
    let dead: Vec<u64> = queued.into_iter().filter(|serial| !tracked(*serial)).collect();
    landed_seeds()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .retain(|landed| !dead.contains(&landed.serial));
}

pub(crate) fn take_for_compute(key: ComputeRawStorageKey) -> Option<(ComputeRawStorageKey, u64)> {
    let mut registry = registry().lock().unwrap_or_else(|error| error.into_inner());
    registry.apply_landed_seeds();
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
        PENDING_COMPUTE_SEEDS.store(0, Ordering::Release);
        landed_seeds().lock().unwrap_or_else(|error| error.into_inner()).clear();
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

    fn overlay(serial: u64, len: u64) -> ComputeOverlay {
        ComputeOverlay {
            dst_offset: 0,
            len,
            serial,
            resource_index: 1,
            src_offset: 0,
        }
    }

    #[test]
    fn compute_output_is_adopted_by_a_writer_and_shared_until_it_lands() {
        let _guard = test_lock();
        let owner = key(0x5000_0000, 0x1_0000, 0x40);
        assert!(bind_compute_output(owner, false, Some(overlay(5, 0x40))).is_none());
        let adopted = bind_compute_output(owner, true, Some(overlay(5, 0x40))).unwrap();
        assert_eq!((adopted.key, adopted.offset, adopted.writes), (owner, 0, true));
        assert!(adopted.seed.is_none());
        assert_eq!(adopted.compute_seed, Some(overlay(5, 0x40)));
        assert!(has_pending_compute_seeds());

        assert!(bind_compute_output(key(0x5000_0010, 0x1_0010, 0x20), false, None).is_none());
        let stale_guest = bind(owner, true, &bytes(0x40, 3), 0, &mut |_| {
            panic!("a pending compute seed must not reseed from guest bytes")
        })
        .unwrap();
        assert_eq!(stale_guest.generation, adopted.generation);

        note_writeback(key(0x5000_0000, 0x1_0000, 0x40), 4, 1, &[9u8; 0x40], true);
        assert!(has_pending_compute_seeds());
        note_writeback(key(0x5000_0000, 0x1_0000, 0x40), 5, 1, &[7u8; 0x40], true);
        assert!(!has_pending_compute_seeds());
        let landed = bind(owner, false, &bytes(0x40, 7), 0, &mut |_| panic!("landed bytes match")).unwrap();
        assert_eq!(landed.generation, adopted.generation);
        assert_eq!(landed.seed.as_deref(), Some(&vec![7u8; 0x40]));
        let again = bind_compute_output(owner, true, Some(overlay(5, 0x40))).unwrap();
        assert_eq!(again.generation, adopted.generation);
        assert!(bind_compute_output(owner, true, None).is_none());

        let newer = bind_compute_output(owner, true, Some(overlay(6, 0x40))).unwrap();
        assert_ne!(newer.generation, adopted.generation);
        assert_eq!(
            bind_compute_output(owner, true, Some(overlay(7, 0x20))).map(|resident| resident.generation),
            Some(newer.generation)
        );
        assert!(bind_compute_output(key(0x6000_0000, 0x2_0000, 0x40), true, Some(overlay(8, 0x20))).is_none());
        assert_eq!(take_for_compute(owner), Some((owner, newer.generation)));
        assert!(!has_pending_compute_seeds());
    }

    #[test]
    fn early_landing_fills_a_later_adoption_and_failed_landings_are_reaped() {
        let _guard = test_lock();
        let owner = key(0x5000_0000, 0x1_0000, 0x40);
        bind_compute_output(key(0x7000_0000, 0x3_0000, 0x40), true, Some(overlay(11, 0x40))).unwrap();
        note_writeback(key(0x5000_0000, 0x1_0000, 0x80), 9, 1, &[3u8; 0x80], true);
        let adopted = bind_compute_output(owner, true, Some(overlay(9, 0x40))).unwrap();
        let landed = bind(owner, false, &bytes(0x40, 3), 0, &mut |_| panic!("landed bytes match")).unwrap();
        assert_eq!(landed.generation, adopted.generation);
        assert!(landed.seed.is_some());

        reap_stale_compute_seeds(|serial| serial == 11);
        assert!(has_pending_compute_seeds());
        reap_stale_compute_seeds(|_| false);
        assert!(!has_pending_compute_seeds());
        assert!(bind_compute_output(key(0x7000_0000, 0x3_0000, 0x40), false, None).is_none());
    }

    #[test]
    fn misaligned_contained_reader_of_a_pending_owner_takes_the_guest_path() {
        let _guard = test_lock();
        let owner = key(0x5000_0000, 0x1_0000, 0x400);
        bind_compute_output(owner, true, Some(overlay(5, 0x400))).unwrap();
        assert!(bind_compute_output(key(0x5000_0010, 0x1_0010, 0x20), false, None).is_none());
        assert!(bind_compute_output(key(0x5000_0100, 0x1_0100, 0x20), false, None).is_some());
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

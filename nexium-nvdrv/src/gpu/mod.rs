pub(crate) mod clock;
pub(crate) mod completion;
pub mod engines;
pub mod flat_allocator;
mod formats;
pub(crate) mod prep;
pub mod pusher;
pub mod stackdump;
pub mod vk_dispatch;
pub mod watchdog;

pub use engines::{
    Fermi2D, KeplerCompute, KeplerMemory, Maxwell3D, Maxwell3DRegisters, MaxwellDma,
};
pub use pusher::{CommandListHeader, Pusher};

use parking_lot::{Mutex, RwLock};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;

pub type GuestMemoryWriter = Arc<dyn Fn(u64, &[u8]) -> bool + Send + Sync + 'static>;

#[derive(Clone)]
pub(crate) struct GuestMemoryAccess {
    mappings: Arc<RwLock<GpuMappings>>,
    writer: Arc<Mutex<Option<GuestMemoryWriter>>>,
}

fn nvprof_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("NEXIUM_NVDRV_PROFILE").is_ok())
}

fn elapsed_ms(start: std::time::Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

#[derive(Clone)]
pub struct GpuMapping {
    pub gpu_va: u64,
    pub size: u64,
    pub cpu_addr: u64,
    pub nvmap_id: u32,
    epoch: u64,
    record_id: u64,
    sparse: bool,
    owner_record_id: Option<u64>,
    owned_va_range: Option<(u64, u64)>,
    as_gpu_fd: Option<u32>,
    as_gpu_allocation_base: Option<u64>,
    as_gpu_root: bool,
    as_gpu_unmap_barrier: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RemovedGpuMapping {
    pub gpu_va: u64,
    pub size: u64,
    pub cpu_addr: u64,
    pub nvmap_id: u32,
    pub epoch: u64,
    pub owned_va_range: Option<(u64, u64)>,
    pub changed_gpu_ranges: Vec<(u64, u64)>,
    pub epoch_transitions: Vec<GpuMappingEpochTransition>,
    pub unmapped_gpu_ranges: Vec<(u64, u64)>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RemovedGpuMappingSet {
    pub update: GpuMappingUpdate,
    pub owned_va_ranges: Vec<(u64, u64)>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuMappingChange {
    Fresh,
    Extended,
    Idempotent,
    Replaced,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuMappingEpochTransition {
    pub gpu_va: u64,
    pub size: u64,
    pub old_epoch: u64,
    pub new_epoch: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuMappingUpdate {
    pub change: GpuMappingChange,
    pub changed_gpu_ranges: Vec<(u64, u64)>,
    pub epoch_transitions: Vec<GpuMappingEpochTransition>,
}

impl GpuMappingChange {
    pub fn invalidates_render_targets(self) -> bool {
        self == Self::Replaced
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PhysicalMappingIdentity {
    Cpu { cpu_addr: u64, nvmap_id: u32 },
    Sparse,
}

const MAPPING_LOOKUP_CACHE_SIZE: usize = 64;
const MAPPING_LOOKUP_CACHE_SHIFT: u32 = 16;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct MappingLookupCacheEntry {
    gpu_lo: u64,
    gpu_hi: u64,
    mapping_index: usize,
}

struct ThreadMappingLookupCache {
    instance_id: u64,
    generation: u64,
    entries: [Option<MappingLookupCacheEntry>; MAPPING_LOOKUP_CACHE_SIZE],
    misses: u64,
}

impl ThreadMappingLookupCache {
    const fn empty() -> Self {
        Self {
            instance_id: 0,
            generation: 0,
            entries: [None; MAPPING_LOOKUP_CACHE_SIZE],
            misses: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct EffectiveMappingSegment {
    gpu_lo: u64,
    gpu_hi: u64,
    mapping_index: usize,
}

struct MappingSegmentIndex {
    generation: u64,
    by_gpu: Vec<EffectiveMappingSegment>,
    by_cpu: Vec<(u64, u64, usize)>,
    cpu_hi_prefix_max: Vec<u64>,
}

const MAPPING_INDEX_REBUILD_AFTER_STALE_LOOKUPS: u32 = 32;

#[derive(Default)]
struct MappingIndexState {
    index: Option<Arc<MappingSegmentIndex>>,
    stale_lookups: u32,
}

impl MappingSegmentIndex {
    fn build(mappings: &[GpuMapping], generation: u64) -> Self {
        let mut painted: BTreeMap<u64, EffectiveMappingSegment> = BTreeMap::new();
        for (mapping_index, mapping) in mappings.iter().enumerate() {
            let lo = mapping.gpu_va;
            let hi = mapping.gpu_va.saturating_add(mapping.size);
            let mut overlapped = Vec::new();
            for (&key, segment) in painted.range(..hi).rev() {
                if segment.gpu_hi <= lo {
                    break;
                }
                overlapped.push(key);
            }
            if lo >= hi {
                if let Some(key) = overlapped.first().copied() {
                    let segment = painted.remove(&key).unwrap();
                    if segment.gpu_lo < lo && segment.gpu_hi > lo {
                        painted.insert(
                            segment.gpu_lo,
                            EffectiveMappingSegment {
                                gpu_lo: segment.gpu_lo,
                                gpu_hi: lo,
                                mapping_index: segment.mapping_index,
                            },
                        );
                        painted.insert(
                            lo,
                            EffectiveMappingSegment {
                                gpu_lo: lo,
                                gpu_hi: segment.gpu_hi,
                                mapping_index: segment.mapping_index,
                            },
                        );
                    } else {
                        painted.insert(segment.gpu_lo, segment);
                    }
                }
                continue;
            }
            for key in overlapped {
                let segment = painted.remove(&key).unwrap();
                if segment.gpu_lo < lo {
                    painted.insert(
                        segment.gpu_lo,
                        EffectiveMappingSegment {
                            gpu_lo: segment.gpu_lo,
                            gpu_hi: lo,
                            mapping_index: segment.mapping_index,
                        },
                    );
                }
                if segment.gpu_hi > hi {
                    painted.insert(
                        hi,
                        EffectiveMappingSegment {
                            gpu_lo: hi,
                            gpu_hi: segment.gpu_hi,
                            mapping_index: segment.mapping_index,
                        },
                    );
                }
            }
            painted.insert(
                lo,
                EffectiveMappingSegment {
                    gpu_lo: lo,
                    gpu_hi: hi,
                    mapping_index,
                },
            );
        }
        let by_gpu: Vec<EffectiveMappingSegment> = painted.into_values().collect();
        let mut by_cpu: Vec<(u64, u64, usize)> = by_gpu
            .iter()
            .enumerate()
            .filter_map(|(segment_index, segment)| {
                let mapping = &mappings[segment.mapping_index];
                if mapping.sparse {
                    return None;
                }
                let cpu_lo = mapping
                    .cpu_addr
                    .checked_add(segment.gpu_lo - mapping.gpu_va)?;
                let cpu_hi = cpu_lo.saturating_add(segment.gpu_hi - segment.gpu_lo);
                Some((cpu_lo, cpu_hi, segment_index))
            })
            .collect();
        by_cpu.sort_unstable();
        let mut cpu_hi_prefix_max = Vec::with_capacity(by_cpu.len());
        let mut running = 0u64;
        for (_, cpu_hi, _) in &by_cpu {
            running = running.max(*cpu_hi);
            cpu_hi_prefix_max.push(running);
        }
        Self {
            generation,
            by_gpu,
            by_cpu,
            cpu_hi_prefix_max,
        }
    }

    fn lookup(&self, gpu_va: u64) -> Option<MappingLookupCacheEntry> {
        let index = self
            .by_gpu
            .partition_point(|segment| segment.gpu_lo <= gpu_va)
            .checked_sub(1)?;
        let segment = self.by_gpu[index];
        (gpu_va < segment.gpu_hi).then_some(MappingLookupCacheEntry {
            gpu_lo: segment.gpu_lo,
            gpu_hi: segment.gpu_hi,
            mapping_index: segment.mapping_index,
        })
    }

    fn gpu_regions_for_cpu_range(&self, cpu_addr: u64, cpu_end: u64) -> Vec<(u64, u64)> {
        let mut regions = Vec::new();
        let mut index = self
            .by_cpu
            .partition_point(|(cpu_lo, _, _)| *cpu_lo < cpu_end);
        while index > 0 {
            index -= 1;
            if self.cpu_hi_prefix_max[index] <= cpu_addr {
                break;
            }
            let (cpu_lo, cpu_hi, segment_index) = self.by_cpu[index];
            if cpu_hi <= cpu_addr {
                continue;
            }
            let segment = self.by_gpu[segment_index];
            let overlap_start = cpu_addr.max(cpu_lo);
            let overlap_end = cpu_end.min(cpu_hi);
            let start = segment.gpu_lo + (overlap_start - cpu_lo);
            let end = start.saturating_add(overlap_end - overlap_start);
            if start < end {
                regions.push((start, end - start));
            }
        }
        coalesce_gpu_regions(regions)
    }
}

fn coalesce_gpu_regions(mut regions: Vec<(u64, u64)>) -> Vec<(u64, u64)> {
    regions.sort_unstable();
    let mut coalesced = Vec::<(u64, u64)>::with_capacity(regions.len());
    for (start, size) in regions {
        if let Some((previous_start, previous_size)) = coalesced.last_mut() {
            let previous_end = previous_start.saturating_add(*previous_size);
            if start <= previous_end {
                *previous_size = previous_end.max(start.saturating_add(size)) - *previous_start;
                continue;
            }
        }
        coalesced.push((start, size));
    }
    coalesced
}

thread_local! {
    static MAPPING_LOOKUP_CACHE: std::cell::RefCell<ThreadMappingLookupCache> =
        const { std::cell::RefCell::new(ThreadMappingLookupCache::empty()) };
}

pub struct GpuMappings {
    mappings: Vec<GpuMapping>,
    instance_id: u64,
    next_mapping_epoch: u64,
    next_mapping_record_id: u64,
    generation: u64,
    segment_index: Mutex<MappingIndexState>,
}

impl GpuMappings {
    pub fn new() -> Self {
        static NEXT_INSTANCE_ID: std::sync::atomic::AtomicU64 =
            std::sync::atomic::AtomicU64::new(1);
        Self {
            mappings: Vec::new(),
            instance_id: NEXT_INSTANCE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            next_mapping_epoch: 1,
            next_mapping_record_id: 1,
            generation: 1,
            segment_index: Mutex::new(MappingIndexState::default()),
        }
    }

    fn segment_index(&self) -> Option<Arc<MappingSegmentIndex>> {
        let mut state = self.segment_index.lock();
        if let Some(index) = &state.index {
            if index.generation == self.generation {
                return Some(Arc::clone(index));
            }
            state.stale_lookups += 1;
            if state.stale_lookups < MAPPING_INDEX_REBUILD_AFTER_STALE_LOOKUPS {
                return None;
            }
        }
        let index = Arc::new(MappingSegmentIndex::build(&self.mappings, self.generation));
        state.index = Some(Arc::clone(&index));
        state.stale_lookups = 0;
        Some(index)
    }

    #[inline]
    fn contains(mapping: &GpuMapping, gpu_va: u64) -> bool {
        gpu_va >= mapping.gpu_va && gpu_va < mapping.gpu_va.saturating_add(mapping.size)
    }

    fn mapping_lookup_slow(&self, gpu_va: u64) -> Option<MappingLookupCacheEntry> {
        match self.segment_index() {
            Some(index) => index.lookup(gpu_va),
            None => self.mapping_lookup_linear(gpu_va),
        }
    }

    fn mapping_lookup_linear(&self, gpu_va: u64) -> Option<MappingLookupCacheEntry> {
        let mapping_index = self
            .mappings
            .iter()
            .rposition(|mapping| Self::contains(mapping, gpu_va))?;
        let mapping = &self.mappings[mapping_index];
        let mut gpu_lo = mapping.gpu_va;
        let mut gpu_hi = mapping.gpu_va.saturating_add(mapping.size);

        for newer in &self.mappings[mapping_index + 1..] {
            let newer_lo = newer.gpu_va;
            let newer_hi = newer.gpu_va.saturating_add(newer.size);
            if newer_hi <= gpu_va {
                gpu_lo = gpu_lo.max(newer_hi);
            } else if newer_lo > gpu_va {
                gpu_hi = gpu_hi.min(newer_lo);
            } else {
                debug_assert!(
                    !Self::contains(newer, gpu_va),
                    "reverse lookup skipped a newer mapping"
                );
            }
        }

        debug_assert!(gpu_va >= gpu_lo && gpu_va < gpu_hi);
        Some(MappingLookupCacheEntry {
            gpu_lo,
            gpu_hi,
            mapping_index,
        })
    }

    #[inline]
    fn mapping_lookup_for(&self, gpu_va: u64) -> Option<MappingLookupCacheEntry> {
        MAPPING_LOOKUP_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            if cache.instance_id != self.instance_id || cache.generation != self.generation {
                cache.instance_id = self.instance_id;
                cache.generation = self.generation;
                cache.entries = [None; MAPPING_LOOKUP_CACHE_SIZE];
            }
            let slot = (gpu_va >> MAPPING_LOOKUP_CACHE_SHIFT) as usize % MAPPING_LOOKUP_CACHE_SIZE;
            if let Some(cached) = cache.entries[slot] {
                if gpu_va >= cached.gpu_lo && gpu_va < cached.gpu_hi {
                    debug_assert!(
                        self.mappings
                            .get(cached.mapping_index)
                            .is_some_and(|mapping| Self::contains(mapping, gpu_va)),
                        "stale GMMU lookup cache entry"
                    );
                    return Some(cached);
                }
            }
            let cached = self.mapping_lookup_slow(gpu_va)?;
            cache.entries[slot] = Some(cached);
            cache.misses = cache.misses.wrapping_add(1);
            Some(cached)
        })
    }

    #[inline]
    fn mapping_index_for(&self, gpu_va: u64) -> Option<usize> {
        Some(self.mapping_lookup_for(gpu_va)?.mapping_index)
    }

    fn effective_physical_identity(&self, gpu_va: u64) -> Option<(PhysicalMappingIdentity, u64)> {
        let mapping = &self.mappings[self.mapping_index_for(gpu_va)?];
        let physical = if mapping.sparse {
            PhysicalMappingIdentity::Sparse
        } else {
            PhysicalMappingIdentity::Cpu {
                cpu_addr: mapping
                    .cpu_addr
                    .checked_add(gpu_va.saturating_sub(mapping.gpu_va))?,
                nvmap_id: mapping.nvmap_id,
            }
        };
        Some((physical, mapping.epoch))
    }

    fn effective_range_snapshot(
        &self,
        gpu_va: u64,
        size: u64,
    ) -> Vec<(u64, u64, Option<(PhysicalMappingIdentity, u64)>)> {
        let Some(gpu_end) = gpu_va.checked_add(size) else {
            return Vec::new();
        };
        let mut boundaries = vec![gpu_va, gpu_end];
        for mapping in &self.mappings {
            let mapping_end = mapping.gpu_va.saturating_add(mapping.size);
            let overlap_start = gpu_va.max(mapping.gpu_va);
            let overlap_end = gpu_end.min(mapping_end);
            if overlap_start < overlap_end {
                boundaries.push(overlap_start);
                boundaries.push(overlap_end);
            }
        }
        boundaries.sort_unstable();
        boundaries.dedup();
        boundaries
            .windows(2)
            .filter_map(|window| {
                (window[0] < window[1]).then(|| {
                    (
                        window[0],
                        window[1],
                        self.effective_physical_identity(window[0]),
                    )
                })
            })
            .collect()
    }

    fn classify_sparse_add(&self, gpu_va: u64, size: u64) -> GpuMappingChange {
        if size == 0 {
            return GpuMappingChange::Fresh;
        }
        let Some(gpu_end) = gpu_va.checked_add(size) else {
            return GpuMappingChange::Replaced;
        };
        let mut boundaries = vec![gpu_va, gpu_end];
        for mapping in &self.mappings {
            let mapping_end = mapping.gpu_va.saturating_add(mapping.size);
            let overlap_start = gpu_va.max(mapping.gpu_va);
            let overlap_end = gpu_end.min(mapping_end);
            if overlap_start < overlap_end {
                boundaries.push(overlap_start);
                boundaries.push(overlap_end);
            }
        }
        boundaries.sort_unstable();
        boundaries.dedup();
        let mut has_gap = false;
        let mut has_sparse = false;
        for window in boundaries.windows(2) {
            if window[0] >= window[1] {
                continue;
            }
            match self.mapping_index_for(window[0]) {
                Some(index) if self.mappings[index].sparse => has_sparse = true,
                Some(_) => return GpuMappingChange::Replaced,
                None => has_gap = true,
            }
        }
        if has_sparse && !has_gap {
            GpuMappingChange::Idempotent
        } else {
            GpuMappingChange::Fresh
        }
    }

    fn push_coalesced_gpu_range(ranges: &mut Vec<(u64, u64)>, gpu_va: u64, size: u64) {
        if size == 0 {
            return;
        }
        if let Some((last_gpu_va, last_size)) = ranges.last_mut() {
            if last_gpu_va.checked_add(*last_size) == Some(gpu_va) {
                *last_size += size;
                return;
            }
        }
        ranges.push((gpu_va, size));
    }

    fn push_epoch_transition(
        transitions: &mut Vec<GpuMappingEpochTransition>,
        gpu_va: u64,
        size: u64,
        old_epoch: u64,
        new_epoch: u64,
    ) {
        if size == 0 || old_epoch == new_epoch {
            return;
        }
        if let Some(last) = transitions.last_mut() {
            if last.gpu_va.checked_add(last.size) == Some(gpu_va)
                && last.old_epoch == old_epoch
                && last.new_epoch == new_epoch
            {
                last.size += size;
                return;
            }
        }
        transitions.push(GpuMappingEpochTransition {
            gpu_va,
            size,
            old_epoch,
            new_epoch,
        });
    }

    fn classify_add(
        &self,
        gpu_va: u64,
        size: u64,
        cpu_addr: u64,
        nvmap_id: u32,
    ) -> (GpuMappingChange, Option<u64>) {
        if size == 0 {
            return (GpuMappingChange::Fresh, None);
        }
        let Some(gpu_end) = gpu_va.checked_add(size) else {
            return (GpuMappingChange::Replaced, None);
        };
        let mut boundaries = vec![gpu_va, gpu_end];
        for mapping in &self.mappings {
            let mapping_end = mapping.gpu_va.saturating_add(mapping.size);
            let overlap_start = gpu_va.max(mapping.gpu_va);
            let overlap_end = gpu_end.min(mapping_end);
            if overlap_start < overlap_end {
                boundaries.push(overlap_start);
                boundaries.push(overlap_end);
            }
        }
        boundaries.sort_unstable();
        boundaries.dedup();
        let mut overlaps = false;
        let mut has_gaps = false;
        let mut overlap_epoch = None;
        let mut multiple_overlap_epochs = false;
        for window in boundaries.windows(2) {
            let segment_start = window[0];
            let segment_end = window[1];
            if segment_start >= segment_end {
                continue;
            }
            let requested_cpu = cpu_addr.checked_add(segment_start.saturating_sub(gpu_va));
            let Some(mapping_index) = self.mapping_index_for(segment_start) else {
                if requested_cpu.is_none() {
                    return (GpuMappingChange::Replaced, None);
                }
                has_gaps = true;
                continue;
            };
            overlaps = true;
            let mapping = &self.mappings[mapping_index];
            let existing_cpu = if mapping.sparse {
                None
            } else {
                mapping
                    .cpu_addr
                    .checked_add(segment_start.saturating_sub(mapping.gpu_va))
            };
            if mapping.nvmap_id != nvmap_id
                || existing_cpu.is_none()
                || existing_cpu != requested_cpu
            {
                return (GpuMappingChange::Replaced, None);
            }
            match overlap_epoch {
                Some(epoch) if epoch != mapping.epoch => multiple_overlap_epochs = true,
                None => overlap_epoch = Some(mapping.epoch),
                _ => {}
            }
        }
        if !overlaps {
            (GpuMappingChange::Fresh, None)
        } else if multiple_overlap_epochs {
            (GpuMappingChange::Replaced, None)
        } else if !has_gaps {
            (GpuMappingChange::Idempotent, overlap_epoch)
        } else {
            (GpuMappingChange::Extended, overlap_epoch)
        }
    }

    fn insert_mapping(
        &mut self,
        gpu_va: u64,
        size: u64,
        cpu_addr: u64,
        nvmap_id: u32,
        sparse: bool,
        owner_record_id: Option<u64>,
        owned_va_range: Option<(u64, u64)>,
        change: GpuMappingChange,
        preserved_epoch: Option<u64>,
        retain_idempotent: bool,
        as_gpu_fd: Option<u32>,
        as_gpu_allocation_base: Option<u64>,
        as_gpu_root: bool,
    ) -> GpuMappingUpdate {
        let before = self.effective_range_snapshot(gpu_va, size);
        nexium_gpu::tex_invalidate::bump_region(gpu_va, size);
        if change == GpuMappingChange::Idempotent && !retain_idempotent {
            return GpuMappingUpdate {
                change,
                changed_gpu_ranges: Vec::new(),
                epoch_transitions: Vec::new(),
            };
        }
        let epoch = if matches!(
            change,
            GpuMappingChange::Extended | GpuMappingChange::Idempotent
        ) {
            preserved_epoch.unwrap_or_else(|| {
                let epoch = self.next_mapping_epoch;
                self.next_mapping_epoch = self.next_mapping_epoch.wrapping_add(1).max(1);
                epoch
            })
        } else {
            let epoch = self.next_mapping_epoch;
            self.next_mapping_epoch = self.next_mapping_epoch.wrapping_add(1).max(1);
            epoch
        };
        let record_id = self.next_mapping_record_id;
        self.next_mapping_record_id = self.next_mapping_record_id.wrapping_add(1).max(1);
        self.generation = self.generation.wrapping_add(1).max(1);
        self.mappings.push(GpuMapping {
            gpu_va,
            size,
            cpu_addr,
            nvmap_id,
            epoch,
            record_id,
            sparse,
            owner_record_id,
            owned_va_range,
            as_gpu_fd,
            as_gpu_allocation_base,
            as_gpu_root,
            as_gpu_unmap_barrier: false,
        });
        let mut changed_gpu_ranges = Vec::new();
        let mut epoch_transitions = Vec::new();
        for (range_start, range_end, previous) in before {
            let current = self.effective_physical_identity(range_start);
            let range_size = range_end - range_start;
            match (previous, current) {
                (Some((old_physical, old_epoch)), Some((new_physical, new_epoch)))
                    if old_physical == new_physical =>
                {
                    Self::push_epoch_transition(
                        &mut epoch_transitions,
                        range_start,
                        range_size,
                        old_epoch,
                        new_epoch,
                    );
                }
                (old, new) if old != new => {
                    Self::push_coalesced_gpu_range(
                        &mut changed_gpu_ranges,
                        range_start,
                        range_size,
                    );
                }
                _ => {}
            }
        }
        for &(range_gpu, range_size) in &changed_gpu_ranges {
            nexium_gpu::pitch_oracle::clear_pitch_range(range_gpu, range_size);
        }
        GpuMappingUpdate {
            change,
            changed_gpu_ranges,
            epoch_transitions,
        }
    }

    pub fn add_with_metadata(
        &mut self,
        gpu_va: u64,
        size: u64,
        cpu_addr: u64,
        nvmap_id: u32,
    ) -> GpuMappingUpdate {
        log::debug!(
            "GpuMap: gpu_va={:#x} size={:#x} cpu_addr={:#x} nvmap_id={}",
            gpu_va,
            size,
            cpu_addr,
            nvmap_id
        );
        if size == 0 || gpu_va.checked_add(size).is_none() || cpu_addr.checked_add(size).is_none() {
            return GpuMappingUpdate {
                change: GpuMappingChange::Replaced,
                changed_gpu_ranges: Vec::new(),
                epoch_transitions: Vec::new(),
            };
        }
        let (change, preserved_epoch) = self.classify_add(gpu_va, size, cpu_addr, nvmap_id);
        self.insert_mapping(
            gpu_va,
            size,
            cpu_addr,
            nvmap_id,
            false,
            None,
            None,
            change,
            preserved_epoch,
            false,
            None,
            None,
            false,
        )
    }

    #[cfg(test)]
    pub(crate) fn add_tracked_with_metadata(
        &mut self,
        gpu_va: u64,
        size: u64,
        cpu_addr: u64,
        nvmap_id: u32,
    ) -> GpuMappingUpdate {
        if size == 0 || gpu_va.checked_add(size).is_none() || cpu_addr.checked_add(size).is_none() {
            return GpuMappingUpdate {
                change: GpuMappingChange::Replaced,
                changed_gpu_ranges: Vec::new(),
                epoch_transitions: Vec::new(),
            };
        }
        let (change, preserved_epoch) = self.classify_add(gpu_va, size, cpu_addr, nvmap_id);
        self.insert_mapping(
            gpu_va,
            size,
            cpu_addr,
            nvmap_id,
            false,
            None,
            None,
            change,
            preserved_epoch,
            true,
            None,
            None,
            false,
        )
    }

    pub(crate) fn add_as_gpu_mapping(
        &mut self,
        fd: u32,
        gpu_va: u64,
        size: u64,
        cpu_addr: u64,
        nvmap_id: u32,
        owned_va_range: Option<(u64, u64)>,
        allocation_base: Option<u64>,
        root: bool,
    ) -> GpuMappingUpdate {
        if size == 0 || gpu_va.checked_add(size).is_none() || cpu_addr.checked_add(size).is_none() {
            return GpuMappingUpdate {
                change: GpuMappingChange::Replaced,
                changed_gpu_ranges: Vec::new(),
                epoch_transitions: Vec::new(),
            };
        }
        let (change, preserved_epoch) = self.classify_add(gpu_va, size, cpu_addr, nvmap_id);
        self.insert_mapping(
            gpu_va,
            size,
            cpu_addr,
            nvmap_id,
            false,
            None,
            owned_va_range,
            change,
            preserved_epoch,
            true,
            Some(fd),
            allocation_base,
            root,
        )
    }

    pub fn add(
        &mut self,
        gpu_va: u64,
        size: u64,
        cpu_addr: u64,
        nvmap_id: u32,
    ) -> GpuMappingChange {
        self.add_with_metadata(gpu_va, size, cpu_addr, nvmap_id)
            .change
    }

    pub(crate) fn add_with_va_ownership(
        &mut self,
        gpu_va: u64,
        size: u64,
        cpu_addr: u64,
        nvmap_id: u32,
        owned_va_range: (u64, u64),
    ) -> GpuMappingUpdate {
        if size == 0 || gpu_va.checked_add(size).is_none() || cpu_addr.checked_add(size).is_none() {
            return GpuMappingUpdate {
                change: GpuMappingChange::Replaced,
                changed_gpu_ranges: Vec::new(),
                epoch_transitions: Vec::new(),
            };
        }
        let (change, preserved_epoch) = self.classify_add(gpu_va, size, cpu_addr, nvmap_id);
        self.insert_mapping(
            gpu_va,
            size,
            cpu_addr,
            nvmap_id,
            false,
            None,
            Some(owned_va_range),
            change,
            preserved_epoch,
            true,
            None,
            None,
            false,
        )
    }

    pub(crate) fn add_owned(
        &mut self,
        fd: u32,
        gpu_va: u64,
        size: u64,
        cpu_addr: u64,
        nvmap_id: u32,
        owner_record_id: u64,
    ) -> GpuMappingUpdate {
        if size == 0 || gpu_va.checked_add(size).is_none() || cpu_addr.checked_add(size).is_none() {
            return GpuMappingUpdate {
                change: GpuMappingChange::Replaced,
                changed_gpu_ranges: Vec::new(),
                epoch_transitions: Vec::new(),
            };
        }
        let (change, preserved_epoch) = self.classify_add(gpu_va, size, cpu_addr, nvmap_id);
        self.insert_mapping(
            gpu_va,
            size,
            cpu_addr,
            nvmap_id,
            false,
            Some(owner_record_id),
            None,
            change,
            preserved_epoch,
            true,
            Some(fd),
            None,
            false,
        )
    }

    #[cfg(test)]
    pub(crate) fn add_sparse_with_metadata(&mut self, gpu_va: u64, size: u64) -> GpuMappingUpdate {
        if size == 0 || gpu_va.checked_add(size).is_none() {
            return GpuMappingUpdate {
                change: GpuMappingChange::Replaced,
                changed_gpu_ranges: Vec::new(),
                epoch_transitions: Vec::new(),
            };
        }
        let change = self.classify_sparse_add(gpu_va, size);
        self.insert_mapping(
            gpu_va, size, 0, 0, true, None, None, change, None, false, None, None, false,
        )
    }

    #[cfg(test)]
    pub(crate) fn add_sparse(&mut self, gpu_va: u64, size: u64) -> GpuMappingChange {
        self.add_sparse_with_metadata(gpu_va, size).change
    }

    pub(crate) fn add_sparse_as_gpu_with_metadata(
        &mut self,
        fd: u32,
        gpu_va: u64,
        size: u64,
    ) -> GpuMappingUpdate {
        if size == 0 || gpu_va.checked_add(size).is_none() {
            return GpuMappingUpdate {
                change: GpuMappingChange::Replaced,
                changed_gpu_ranges: Vec::new(),
                epoch_transitions: Vec::new(),
            };
        }
        let change = self.classify_sparse_add(gpu_va, size);
        self.insert_mapping(
            gpu_va,
            size,
            0,
            0,
            true,
            None,
            None,
            change,
            None,
            true,
            Some(fd),
            None,
            false,
        )
    }

    pub fn cpu_address_for_any32(&self, gpu_va: u64) -> Option<(u64, u64, u64)> {
        fn segment_for(mapping: &GpuMapping, address: u64) -> Option<(u64, u64)> {
            const DOMAIN: u64 = 1u64 << 32;
            let base = mapping.gpu_va & 0xFFFF_FFFF;
            let covered = mapping.size.min(DOMAIN);
            let first_len = covered.min(DOMAIN - base);
            if address >= base && address - base < first_len {
                return Some((address - base, base + first_len));
            }
            let wrapped_len = covered - first_len;
            (address < wrapped_len).then_some((first_len + address, wrapped_len))
        }

        let lo = gpu_va & 0xFFFF_FFFF;
        let mapping_index = self
            .mappings
            .iter()
            .rposition(|mapping| segment_for(mapping, lo).is_some())?;
        let mapping = &self.mappings[mapping_index];
        if mapping.sparse {
            return None;
        }
        let (offset, segment_end) = segment_for(mapping, lo)?;
        let mut remaining = segment_end - lo;
        for newer in &self.mappings[mapping_index + 1..] {
            let newer_lo = newer.gpu_va & 0xFFFF_FFFF;
            if newer_lo > lo {
                remaining = remaining.min(newer_lo - lo);
            }
        }
        Some((
            mapping.gpu_va,
            mapping.cpu_addr.checked_add(offset)?,
            remaining,
        ))
    }

    pub fn bracket(&self, gpu_va: u64) -> String {
        let mut below: Option<&GpuMapping> = None;
        let mut above: Option<&GpuMapping> = None;
        for m in &self.mappings {
            if m.gpu_va <= gpu_va {
                if below.map_or(true, |b| m.gpu_va > b.gpu_va) {
                    below = Some(m);
                }
            } else if above.map_or(true, |a| m.gpu_va < a.gpu_va) {
                above = Some(m);
            }
        }
        let f = |o: Option<&GpuMapping>| match o {
            Some(m) => format!(
                "{:#x}..{:#x}(nv{} cpu{:#x})",
                m.gpu_va,
                m.gpu_va + m.size,
                m.nvmap_id,
                m.cpu_addr
            ),
            None => "none".to_string(),
        };
        format!(
            "below={} above={} total={}",
            f(below),
            f(above),
            self.mappings.len()
        )
    }

    pub fn remove_with_metadata(&mut self, gpu_va: u64) -> Option<RemovedGpuMapping> {
        let pos = self.mappings.iter().rposition(|mapping| {
            mapping.gpu_va == gpu_va && !mapping.sparse && mapping.owner_record_id.is_none()
        })?;
        let removed = self.mappings[pos].clone();
        let removed_end = removed.gpu_va.saturating_add(removed.size);
        let mut remove_mask = vec![false; self.mappings.len()];
        let mut removed_ranges = Vec::new();
        for (index, mapping) in self.mappings.iter().enumerate() {
            if index == pos {
                remove_mask[index] = true;
                removed_ranges.push((mapping.gpu_va, mapping.gpu_va.saturating_add(mapping.size)));
            }
        }
        let mut boundaries = Vec::new();
        for &(range_start, range_end) in &removed_ranges {
            boundaries.push(range_start);
            boundaries.push(range_end);
            for mapping in &self.mappings {
                let mapping_end = mapping.gpu_va.saturating_add(mapping.size);
                let overlap_start = range_start.max(mapping.gpu_va);
                let overlap_end = range_end.min(mapping_end);
                if overlap_start < overlap_end {
                    boundaries.push(overlap_start);
                    boundaries.push(overlap_end);
                }
            }
        }
        boundaries.sort_unstable();
        boundaries.dedup();
        let before = boundaries
            .windows(2)
            .filter_map(|window| {
                (window[0] < window[1]
                    && removed_ranges
                        .iter()
                        .any(|&(start, end)| window[0] >= start && window[0] < end))
                .then(|| {
                    (
                        window[0],
                        window[1],
                        self.effective_physical_identity(window[0]),
                    )
                })
            })
            .collect::<Vec<_>>();

        let mut index = 0;
        self.mappings.retain(|_| {
            let keep = !remove_mask[index];
            index += 1;
            keep
        });
        self.generation = self.generation.wrapping_add(1).max(1);
        let mut changed_gpu_ranges = Vec::new();
        let mut epoch_transitions = Vec::new();
        let mut unmapped_gpu_ranges = Vec::new();
        for (range_start, range_end, previous) in before {
            let current = self.effective_physical_identity(range_start);
            let range_size = range_end - range_start;
            match (previous, current) {
                (Some((old_physical, old_epoch)), Some((new_physical, new_epoch)))
                    if old_physical == new_physical =>
                {
                    Self::push_epoch_transition(
                        &mut epoch_transitions,
                        range_start,
                        range_size,
                        old_epoch,
                        new_epoch,
                    );
                }
                (old, new) if old != new => {
                    Self::push_coalesced_gpu_range(
                        &mut changed_gpu_ranges,
                        range_start,
                        range_size,
                    );
                    if new.is_none() && range_start >= removed.gpu_va && range_end <= removed_end {
                        Self::push_coalesced_gpu_range(
                            &mut unmapped_gpu_ranges,
                            range_start,
                            range_size,
                        );
                    }
                }
                _ => {}
            }
        }
        for &(range_gpu, range_size) in &changed_gpu_ranges {
            nexium_gpu::pitch_oracle::clear_pitch_range(range_gpu, range_size);
            nexium_gpu::tex_invalidate::bump_region(range_gpu, range_size);
        }
        for transition in &epoch_transitions {
            nexium_gpu::tex_invalidate::bump_region(transition.gpu_va, transition.size);
        }
        Some(RemovedGpuMapping {
            gpu_va: removed.gpu_va,
            size: removed.size,
            cpu_addr: removed.cpu_addr,
            nvmap_id: removed.nvmap_id,
            epoch: removed.epoch,
            owned_va_range: removed.owned_va_range,
            changed_gpu_ranges,
            epoch_transitions,
            unmapped_gpu_ranges,
        })
    }

    pub(crate) fn remove_all_contained_with_metadata(
        &mut self,
        fd: u32,
        gpu_va: u64,
        size: u64,
    ) -> Result<RemovedGpuMappingSet, ()> {
        let gpu_end = gpu_va.checked_add(size).filter(|_| size != 0).ok_or(())?;
        let remove_mask = self
            .mappings
            .iter()
            .map(|mapping| {
                mapping.as_gpu_fd == Some(fd)
                    && mapping
                        .gpu_va
                        .checked_add(mapping.size)
                        .is_some_and(|end| mapping.gpu_va >= gpu_va && end <= gpu_end)
            })
            .collect::<Vec<_>>();
        if self
            .mappings
            .iter()
            .zip(&remove_mask)
            .any(|(mapping, selected)| {
                !selected
                    && mapping.as_gpu_fd == Some(fd)
                    && mapping.gpu_va < gpu_end
                    && mapping.gpu_va.saturating_add(mapping.size) > gpu_va
                    && mapping.owned_va_range.is_some()
            })
        {
            return Err(());
        }
        let mut owned_va_ranges = self
            .mappings
            .iter()
            .zip(&remove_mask)
            .filter_map(|(mapping, selected)| selected.then_some(mapping.owned_va_range).flatten())
            .collect::<Vec<_>>();
        owned_va_ranges.sort_unstable();
        owned_va_ranges.dedup();
        let has_crossing = self
            .mappings
            .iter()
            .zip(&remove_mask)
            .any(|(mapping, selected)| {
                !selected
                    && mapping.as_gpu_fd == Some(fd)
                    && mapping.gpu_va < gpu_end
                    && mapping.gpu_va.saturating_add(mapping.size) > gpu_va
            });
        if !remove_mask.iter().any(|selected| *selected) && !has_crossing {
            return Ok(RemovedGpuMappingSet {
                update: GpuMappingUpdate {
                    change: GpuMappingChange::Replaced,
                    changed_gpu_ranges: Vec::new(),
                    epoch_transitions: Vec::new(),
                },
                owned_va_ranges,
            });
        }
        let mut boundaries = vec![gpu_va, gpu_end];
        for mapping in &self.mappings {
            let mapping_end = mapping.gpu_va.saturating_add(mapping.size);
            let overlap_start = gpu_va.max(mapping.gpu_va);
            let overlap_end = gpu_end.min(mapping_end);
            if overlap_start < overlap_end {
                boundaries.push(overlap_start);
                boundaries.push(overlap_end);
            }
        }
        boundaries.sort_unstable();
        boundaries.dedup();
        let before = boundaries
            .windows(2)
            .filter_map(|window| {
                (window[0] < window[1]).then(|| {
                    (
                        window[0],
                        window[1],
                        self.effective_physical_identity(window[0]),
                    )
                })
            })
            .collect::<Vec<_>>();
        let mappings = std::mem::take(&mut self.mappings);
        self.mappings = Vec::with_capacity(mappings.len() + 2);
        for (index, mapping) in mappings.into_iter().enumerate() {
            if remove_mask[index] {
                continue;
            }
            let mapping_end = mapping.gpu_va.saturating_add(mapping.size);
            if mapping.as_gpu_fd != Some(fd) || mapping.gpu_va >= gpu_end || mapping_end <= gpu_va {
                self.mappings.push(mapping);
                continue;
            }
            let left_size = gpu_va.saturating_sub(mapping.gpu_va);
            let right_size = mapping_end.saturating_sub(gpu_end);
            let right_cpu_offset = gpu_end.saturating_sub(mapping.gpu_va);
            if left_size != 0 {
                let mut left = mapping.clone();
                left.size = left_size;
                self.mappings.push(left);
            }
            if right_size != 0 {
                let mut right = mapping;
                right.gpu_va = gpu_end;
                right.size = right_size;
                if !right.sparse {
                    right.cpu_addr = right
                        .cpu_addr
                        .checked_add(right_cpu_offset)
                        .unwrap_or(right.cpu_addr);
                }
                if left_size != 0 {
                    right.record_id = self.next_mapping_record_id;
                    self.next_mapping_record_id =
                        self.next_mapping_record_id.wrapping_add(1).max(1);
                }
                self.mappings.push(right);
            }
        }
        self.generation = self.generation.wrapping_add(1).max(1);
        let mut changed_gpu_ranges = Vec::new();
        let mut epoch_transitions = Vec::new();
        for (range_start, range_end, previous) in before {
            let current = self.effective_physical_identity(range_start);
            let range_size = range_end - range_start;
            match (previous, current) {
                (Some((old_physical, old_epoch)), Some((new_physical, new_epoch)))
                    if old_physical == new_physical =>
                {
                    Self::push_epoch_transition(
                        &mut epoch_transitions,
                        range_start,
                        range_size,
                        old_epoch,
                        new_epoch,
                    );
                }
                (old, new) if old != new => {
                    Self::push_coalesced_gpu_range(
                        &mut changed_gpu_ranges,
                        range_start,
                        range_size,
                    );
                }
                _ => {}
            }
        }
        for &(range_gpu, range_size) in &changed_gpu_ranges {
            nexium_gpu::pitch_oracle::clear_pitch_range(range_gpu, range_size);
            nexium_gpu::tex_invalidate::bump_region(range_gpu, range_size);
        }
        for transition in &epoch_transitions {
            nexium_gpu::tex_invalidate::bump_region(transition.gpu_va, transition.size);
        }
        Ok(RemovedGpuMappingSet {
            update: GpuMappingUpdate {
                change: GpuMappingChange::Replaced,
                changed_gpu_ranges,
                epoch_transitions,
            },
            owned_va_ranges,
        })
    }

    pub(crate) fn remove_all_for_allocation_with_metadata(
        &mut self,
        fd: u32,
        allocation_base: u64,
    ) -> RemovedGpuMappingSet {
        let remove_mask = self
            .mappings
            .iter()
            .map(|mapping| {
                mapping.as_gpu_fd == Some(fd)
                    && mapping.as_gpu_allocation_base == Some(allocation_base)
            })
            .collect::<Vec<_>>();
        let mut owned_va_ranges = self
            .mappings
            .iter()
            .zip(&remove_mask)
            .filter_map(|(mapping, selected)| selected.then_some(mapping.owned_va_range).flatten())
            .collect::<Vec<_>>();
        owned_va_ranges.sort_unstable();
        owned_va_ranges.dedup();
        let mut affected_ranges = self
            .mappings
            .iter()
            .zip(&remove_mask)
            .filter_map(|(mapping, selected)| {
                selected.then(|| (mapping.gpu_va, mapping.gpu_va.saturating_add(mapping.size)))
            })
            .collect::<Vec<_>>();
        affected_ranges.sort_unstable();
        let mut merged_ranges: Vec<(u64, u64)> = Vec::new();
        for (start, end) in affected_ranges {
            if let Some(last) = merged_ranges.last_mut() {
                if start <= last.1 {
                    last.1 = last.1.max(end);
                    continue;
                }
            }
            merged_ranges.push((start, end));
        }
        if merged_ranges.is_empty() {
            return RemovedGpuMappingSet {
                update: GpuMappingUpdate {
                    change: GpuMappingChange::Replaced,
                    changed_gpu_ranges: Vec::new(),
                    epoch_transitions: Vec::new(),
                },
                owned_va_ranges,
            };
        }
        let mut before = Vec::new();
        for &(range_start, range_end) in &merged_ranges {
            let mut boundaries = vec![range_start, range_end];
            for mapping in &self.mappings {
                let mapping_end = mapping.gpu_va.saturating_add(mapping.size);
                let overlap_start = range_start.max(mapping.gpu_va);
                let overlap_end = range_end.min(mapping_end);
                if overlap_start < overlap_end {
                    boundaries.push(overlap_start);
                    boundaries.push(overlap_end);
                }
            }
            boundaries.sort_unstable();
            boundaries.dedup();
            before.extend(boundaries.windows(2).filter_map(|window| {
                (window[0] < window[1]).then(|| {
                    (
                        window[0],
                        window[1],
                        self.effective_physical_identity(window[0]),
                    )
                })
            }));
        }
        let mut index = 0;
        self.mappings.retain(|_| {
            let keep = !remove_mask[index];
            index += 1;
            keep
        });
        for &(range_start, range_end) in &merged_ranges {
            let epoch = self.next_mapping_epoch;
            self.next_mapping_epoch = self.next_mapping_epoch.wrapping_add(1).max(1);
            let record_id = self.next_mapping_record_id;
            self.next_mapping_record_id = self.next_mapping_record_id.wrapping_add(1).max(1);
            self.mappings.push(GpuMapping {
                gpu_va: range_start,
                size: range_end - range_start,
                cpu_addr: 0,
                nvmap_id: 0,
                epoch,
                record_id,
                sparse: true,
                owner_record_id: None,
                owned_va_range: None,
                as_gpu_fd: Some(fd),
                as_gpu_allocation_base: None,
                as_gpu_root: false,
                as_gpu_unmap_barrier: true,
            });
        }
        self.generation = self.generation.wrapping_add(1).max(1);
        let mut changed_gpu_ranges = Vec::new();
        let mut epoch_transitions = Vec::new();
        for (range_start, range_end, previous) in before {
            let current = self.effective_physical_identity(range_start);
            let range_size = range_end - range_start;
            match (previous, current) {
                (Some((old_physical, old_epoch)), Some((new_physical, new_epoch)))
                    if old_physical == new_physical =>
                {
                    Self::push_epoch_transition(
                        &mut epoch_transitions,
                        range_start,
                        range_size,
                        old_epoch,
                        new_epoch,
                    );
                }
                (old, new) if old != new => {
                    Self::push_coalesced_gpu_range(
                        &mut changed_gpu_ranges,
                        range_start,
                        range_size,
                    );
                }
                _ => {}
            }
        }
        for &(range_gpu, range_size) in &changed_gpu_ranges {
            nexium_gpu::pitch_oracle::clear_pitch_range(range_gpu, range_size);
            nexium_gpu::tex_invalidate::bump_region(range_gpu, range_size);
        }
        for transition in &epoch_transitions {
            nexium_gpu::tex_invalidate::bump_region(transition.gpu_va, transition.size);
        }
        RemovedGpuMappingSet {
            update: GpuMappingUpdate {
                change: GpuMappingChange::Replaced,
                changed_gpu_ranges,
                epoch_transitions,
            },
            owned_va_ranges,
        }
    }

    pub fn remove(&mut self, gpu_va: u64) -> Option<u64> {
        self.remove_with_metadata(gpu_va)
            .map(|removed| removed.size)
    }

    pub(crate) fn unmap_as_gpu_with_metadata(
        &mut self,
        fd: u32,
        gpu_va: u64,
    ) -> Option<RemovedGpuMapping> {
        let barrier = self.mappings.iter().rposition(|mapping| {
            mapping.as_gpu_fd == Some(fd)
                && mapping.as_gpu_unmap_barrier
                && Self::contains(mapping, gpu_va)
        });
        let pos = self
            .mappings
            .iter()
            .enumerate()
            .rposition(|(index, mapping)| {
                barrier.is_none_or(|barrier| index > barrier)
                    && mapping.gpu_va == gpu_va
                    && !mapping.sparse
                    && mapping.owner_record_id.is_none()
                    && mapping.as_gpu_fd == Some(fd)
                    && mapping.as_gpu_root
            })?;
        let removed = self.mappings[pos].clone();
        let before = self.effective_range_snapshot(removed.gpu_va, removed.size);
        self.mappings.remove(pos);
        let epoch = self.next_mapping_epoch;
        self.next_mapping_epoch = self.next_mapping_epoch.wrapping_add(1).max(1);
        let record_id = self.next_mapping_record_id;
        self.next_mapping_record_id = self.next_mapping_record_id.wrapping_add(1).max(1);
        self.mappings.push(GpuMapping {
            gpu_va: removed.gpu_va,
            size: removed.size,
            cpu_addr: 0,
            nvmap_id: 0,
            epoch,
            record_id,
            sparse: true,
            owner_record_id: None,
            owned_va_range: None,
            as_gpu_fd: Some(fd),
            as_gpu_allocation_base: removed.as_gpu_allocation_base,
            as_gpu_root: false,
            as_gpu_unmap_barrier: true,
        });
        self.generation = self.generation.wrapping_add(1).max(1);
        let mut changed_gpu_ranges = Vec::new();
        let mut epoch_transitions = Vec::new();
        let mut unmapped_gpu_ranges = Vec::new();
        for (range_start, range_end, previous) in before {
            let current = self.effective_physical_identity(range_start);
            let range_size = range_end - range_start;
            match (previous, current) {
                (Some((old_physical, old_epoch)), Some((new_physical, new_epoch)))
                    if old_physical == new_physical =>
                {
                    Self::push_epoch_transition(
                        &mut epoch_transitions,
                        range_start,
                        range_size,
                        old_epoch,
                        new_epoch,
                    );
                }
                (old, new) if old != new => {
                    Self::push_coalesced_gpu_range(
                        &mut changed_gpu_ranges,
                        range_start,
                        range_size,
                    );
                    Self::push_coalesced_gpu_range(
                        &mut unmapped_gpu_ranges,
                        range_start,
                        range_size,
                    );
                }
                _ => {}
            }
        }
        for &(range_gpu, range_size) in &changed_gpu_ranges {
            nexium_gpu::pitch_oracle::clear_pitch_range(range_gpu, range_size);
            nexium_gpu::tex_invalidate::bump_region(range_gpu, range_size);
        }
        for transition in &epoch_transitions {
            nexium_gpu::tex_invalidate::bump_region(transition.gpu_va, transition.size);
        }
        Some(RemovedGpuMapping {
            gpu_va: removed.gpu_va,
            size: removed.size,
            cpu_addr: removed.cpu_addr,
            nvmap_id: removed.nvmap_id,
            epoch: removed.epoch,
            owned_va_range: removed.owned_va_range,
            changed_gpu_ranges,
            epoch_transitions,
            unmapped_gpu_ranges,
        })
    }

    #[inline]
    pub fn cpu_address_for(&self, gpu_va: u64) -> Option<u64> {
        let mapping = &self.mappings[self.mapping_index_for(gpu_va)?];
        (!mapping.sparse)
            .then(|| mapping.cpu_addr.checked_add(gpu_va - mapping.gpu_va))
            .flatten()
    }

    #[inline]
    pub fn mapping_at(&self, gpu_va: u64) -> Option<(u64, u64, u64)> {
        let cached = self.mapping_lookup_for(gpu_va)?;
        let mapping = &self.mappings[cached.mapping_index];
        if mapping.sparse {
            return None;
        }
        let cpu_addr = mapping
            .cpu_addr
            .checked_add(cached.gpu_lo.saturating_sub(mapping.gpu_va))?;
        Some((cached.gpu_lo, cached.gpu_hi - cached.gpu_lo, cpu_addr))
    }

    pub(crate) fn texture_memory_range(&self, gpu_va: u64) -> Option<(Option<u64>, u64)> {
        let cached = self.mapping_lookup_for(gpu_va)?;
        let mapping = &self.mappings[cached.mapping_index];
        let cpu = if mapping.sparse {
            None
        } else {
            Some(mapping.cpu_addr.checked_add(gpu_va - mapping.gpu_va)?)
        };
        Some((cpu, cached.gpu_hi - gpu_va))
    }

    #[inline]
    pub fn cpu_range_for(&self, gpu_va: u64) -> Option<(u64, u64)> {
        let cached = self.mapping_lookup_for(gpu_va)?;
        let mapping = &self.mappings[cached.mapping_index];
        if mapping.sparse {
            return None;
        }
        let offset = gpu_va - mapping.gpu_va;
        Some((
            mapping.cpu_addr.checked_add(offset)?,
            cached.gpu_hi - gpu_va,
        ))
    }

    pub fn iter(&self) -> impl Iterator<Item = &GpuMapping> {
        self.mappings.iter()
    }

    pub fn mapping_starting_at(&self, gpu_va: u64) -> Option<&GpuMapping> {
        self.mappings
            .iter()
            .rev()
            .find(|mapping| mapping.gpu_va == gpu_va)
    }

    pub(crate) fn remap_source_starting_at(
        &self,
        fd: u32,
        gpu_va: u64,
    ) -> Option<(u64, u64, u32, u64)> {
        let barrier = self.mappings.iter().rposition(|mapping| {
            mapping.as_gpu_fd == Some(fd)
                && mapping.as_gpu_unmap_barrier
                && Self::contains(mapping, gpu_va)
        });
        self.mappings
            .iter()
            .enumerate()
            .rfind(|(index, mapping)| {
                barrier.is_none_or(|barrier| *index > barrier)
                    && mapping.gpu_va == gpu_va
                    && !mapping.sparse
                    && mapping.owner_record_id.is_none()
                    && mapping.as_gpu_fd == Some(fd)
                    && mapping.as_gpu_root
            })
            .map(|(_, mapping)| {
                (
                    mapping.cpu_addr,
                    mapping.size,
                    mapping.nvmap_id,
                    mapping.record_id,
                )
            })
    }

    pub fn gpu_regions_for_cpu_range(&self, cpu_addr: u64, size: u64) -> Vec<(u64, u64)> {
        if size == 0 {
            return Vec::new();
        }
        match self.segment_index() {
            Some(index) => index.gpu_regions_for_cpu_range(cpu_addr, cpu_addr.saturating_add(size)),
            None => self.gpu_regions_for_cpu_range_linear(cpu_addr, size),
        }
    }

    fn gpu_regions_for_cpu_range_linear(&self, cpu_addr: u64, size: u64) -> Vec<(u64, u64)> {
        if size == 0 {
            return Vec::new();
        }
        let cpu_end = cpu_addr.saturating_add(size);
        let mut regions = Vec::new();
        for (index, mapping) in self.mappings.iter().enumerate() {
            if mapping.sparse {
                continue;
            }
            let mapping_end = mapping.cpu_addr.saturating_add(mapping.size);
            let overlap_start = cpu_addr.max(mapping.cpu_addr);
            let overlap_end = cpu_end.min(mapping_end);
            if overlap_start >= overlap_end {
                continue;
            }
            let Some(fragment_start) = mapping.gpu_va.checked_add(overlap_start - mapping.cpu_addr)
            else {
                continue;
            };
            let fragment_end = fragment_start.saturating_add(overlap_end - overlap_start);
            let mut fragments = vec![(fragment_start, fragment_end)];
            for newer in &self.mappings[index + 1..] {
                let newer_start = newer.gpu_va;
                let newer_end = newer.gpu_va.saturating_add(newer.size);
                let mut surviving = Vec::with_capacity(fragments.len() + 1);
                for (start, end) in fragments {
                    if end <= newer_start || start >= newer_end {
                        surviving.push((start, end));
                        continue;
                    }
                    if start < newer_start {
                        surviving.push((start, newer_start));
                    }
                    if end > newer_end {
                        surviving.push((newer_end, end));
                    }
                }
                fragments = surviving;
                if fragments.is_empty() {
                    break;
                }
            }
            regions.extend(
                fragments
                    .into_iter()
                    .filter_map(|(start, end)| (start < end).then_some((start, end - start))),
            );
        }
        coalesce_gpu_regions(regions)
    }

    pub fn describe_around(&self, gpu_va: u64) -> String {
        let lo = gpu_va.saturating_sub(0x40000);
        let hi = gpu_va.saturating_add(0x60000);
        let mut parts: Vec<String> = Vec::new();
        for m in &self.mappings {
            if m.gpu_va < hi && m.gpu_va + m.size > lo {
                let contains = gpu_va >= m.gpu_va && gpu_va < m.gpu_va + m.size;
                parts.push(format!(
                    "[{}gpu={:#x} size={:#x} cpu={:#x} nvmap={}]",
                    if contains { "*" } else { "" },
                    m.gpu_va,
                    m.size,
                    m.cpu_addr,
                    m.nvmap_id
                ));
            }
        }
        format!(
            "{} mappings near {:#x}: {}",
            parts.len(),
            gpu_va,
            parts.join(" ")
        )
    }

    #[inline]
    pub fn nvmap_id_for(&self, gpu_va: u64) -> Option<u32> {
        let mapping = &self.mappings[self.mapping_index_for(gpu_va)?];
        (!mapping.sparse).then_some(mapping.nvmap_id)
    }

    #[inline]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub(crate) fn instance_id(&self) -> u64 {
        self.instance_id
    }

    #[inline]
    pub fn mapping_epoch_for(&self, gpu_va: u64) -> Option<u64> {
        let mapping = &self.mappings[self.mapping_index_for(gpu_va)?];
        (!mapping.sparse).then_some(mapping.epoch)
    }
}

impl Default for GpuMappings {
    fn default() -> Self {
        Self::new()
    }
}

impl GuestMemoryAccess {
    fn new(mappings: Arc<RwLock<GpuMappings>>) -> Self {
        Self {
            mappings,
            writer: Arc::new(Mutex::new(None)),
        }
    }

    fn set_writer(&self, writer: Option<GuestMemoryWriter>) {
        *self.writer.lock() = writer;
    }

    pub(crate) fn is_available(&self) -> bool {
        self.writer.lock().is_some()
    }

    pub(crate) fn writer(&self) -> Option<GuestMemoryWriter> {
        self.writer.lock().clone()
    }

    pub(crate) fn read_mappings(&self) -> impl std::ops::Deref<Target = GpuMappings> + '_ {
        self.mappings.read()
    }

    pub(crate) fn write_gpu(&self, gpu_va: u64, bytes: &[u8]) -> Option<(u64, bool)> {
        let mappings = self.mappings.read();
        self.write_gpu_with_mappings(&mappings, gpu_va, bytes)
    }

    pub(crate) fn write_gpu_with_mappings(
        &self,
        mappings: &GpuMappings,
        gpu_va: u64,
        bytes: &[u8],
    ) -> Option<(u64, bool)> {
        let cpu_addr = mappings.cpu_address_for(gpu_va)?;
        let writer = self.writer.lock().clone()?;
        let written = writer(cpu_addr, bytes);
        if written && !bytes.is_empty() {
            nexium_gpu::tex_invalidate::bump_region(gpu_va, bytes.len() as u64);
        }
        Some((cpu_addr, written))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PendingSyncpointEvent {
    Increment {
        syncpt_id: u32,
        count: u32,
    },
    Completion {
        fd: u32,
        syncpt_id: u32,
        threshold: u32,
    },
}

struct GatedSyncpointCompletionState {
    callback: Option<Box<dyn FnOnce() + Send>>,
    fired: bool,
    released: bool,
}

pub(crate) struct GatedSyncpointCompletion {
    state: Mutex<GatedSyncpointCompletionState>,
}

impl GatedSyncpointCompletion {
    fn fire(&self) {
        let callback = {
            let mut state = self.state.lock();
            if state.released {
                state.callback.take()
            } else {
                state.fired = true;
                None
            }
        };
        if let Some(callback) = callback {
            callback();
        }
    }

    pub(crate) fn release(&self) {
        let callback = {
            let mut state = self.state.lock();
            state.released = true;
            if state.fired {
                state.callback.take()
            } else {
                None
            }
        };
        if let Some(callback) = callback {
            callback();
        }
    }
}

pub(crate) fn gate_syncpoint_completion(
    callback: Option<Box<dyn FnOnce() + Send>>,
) -> (
    Option<Box<dyn FnOnce() + Send>>,
    Option<Arc<GatedSyncpointCompletion>>,
) {
    let Some(callback) = callback else {
        return (None, None);
    };
    let gate = Arc::new(GatedSyncpointCompletion {
        state: Mutex::new(GatedSyncpointCompletionState {
            callback: Some(callback),
            fired: false,
            released: false,
        }),
    });
    let fired_gate = Arc::clone(&gate);
    let wrapped = Box::new(move || fired_gate.fire()) as Box<dyn FnOnce() + Send>;
    (Some(wrapped), Some(gate))
}

fn merge_engine_syncpt_incrs(incrs: &mut Vec<(u32, u32)>, engine_incrs: Vec<u32>) {
    for id in engine_incrs {
        match incrs.iter_mut().find(|(pending_id, _)| *pending_id == id) {
            Some((_, count)) => *count = count.wrapping_add(1),
            None => incrs.push((id, 1)),
        }
    }
}

pub struct GpuContext {
    pub mappings: Arc<RwLock<GpuMappings>>,
    pub maxwell3d: Arc<Mutex<Maxwell3D>>,
    pub maxwell_dma: Arc<Mutex<MaxwellDma>>,
    pub fermi_2d: Arc<Mutex<Fermi2D>>,
    pub kepler_compute: Arc<Mutex<KeplerCompute>>,
    pub kepler_memory: Arc<Mutex<KeplerMemory>>,
    pub pusher: Arc<Mutex<Pusher>>,
    pub small_alloc: Arc<Mutex<flat_allocator::FlatAllocator>>,
    pub big_alloc: Arc<Mutex<flat_allocator::FlatAllocator>>,
    pub channels: Arc<Mutex<HashMap<u32, ChannelState>>>,
    pub stats: Arc<super::PipelineStats>,
    pending_syncpoint_events: Mutex<VecDeque<PendingSyncpointEvent>>,
    guest_memory: GuestMemoryAccess,
    decoder_stub_engines: Mutex<StubEngines>,
}

struct StubEngines {
    maxwell_dma: MaxwellDma,
    fermi_2d: Fermi2D,
    kepler_compute: KeplerCompute,
    kepler_memory: KeplerMemory,
}

pub(crate) fn experimental_gpu_scheduling_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("NEXIUM_EXPERIMENTAL_GPU_SCHEDULING")
                .ok()
                .as_deref(),
            Some("1") | Some("true") | Some("on") | Some("yes")
        )
    })
}

fn eager_small_rt_writeback_value_enabled(value: Option<&std::ffi::OsStr>) -> bool {
    value.is_some_and(|value| {
        matches!(
            value.to_string_lossy().trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "on" | "yes"
        )
    })
}

pub(crate) fn eager_small_rt_writeback_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        let enabled = eager_small_rt_writeback_value_enabled(
            std::env::var_os("NEXIUM_EAGER_SMALL_RT_WRITEBACK").as_deref(),
        );
        if enabled {
            log::warn!(
                "nexium-nvdrv: eager small render-target guest writeback enabled; GPU submissions will serialize"
            );
        } else {
            log::info!(
                "nexium-nvdrv: small render-target guest writeback is dependency-driven"
            );
        }
        enabled
    })
}

pub(crate) fn gpu_pipeline_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        let requested = matches!(
            std::env::var("NEXIUM_GPU_PIPELINE").ok().as_deref(),
            Some("1") | Some("true") | Some("on") | Some("yes")
        );
        let on = requested && experimental_gpu_scheduling_enabled();
        if on {
            log::info!("nexium-nvdrv: GPU decode|prep pipeline ENABLED");
        } else if requested {
            log::warn!(
                "nexium-nvdrv: GPU decode|prep pipeline quarantined; developer opt-in requires NEXIUM_EXPERIMENTAL_GPU_SCHEDULING=1"
            );
        }
        on
    })
}

const BIG_VA_BASE: u64 = 0x4_0000_0000;

pub(crate) struct PusherGuard<'a> {
    guard: parking_lot::MutexGuard<'a, Pusher>,
}

impl std::ops::Deref for PusherGuard<'_> {
    type Target = Pusher;
    fn deref(&self) -> &Pusher {
        &self.guard
    }
}

impl std::ops::DerefMut for PusherGuard<'_> {
    fn deref_mut(&mut self) -> &mut Pusher {
        &mut self.guard
    }
}

impl Drop for PusherGuard<'_> {
    fn drop(&mut self) {
        watchdog::pusher_lock_released();
    }
}

#[derive(Default)]
pub struct ChannelState {
    pub bound_engine: u32,
    pub bound_obj_class: u32,
    pub syncpt_id: u32,
    pub syncpt_min: u32,
    pub syncpt_max: u32,
}

impl GpuContext {
    pub fn new() -> Self {
        Self::with_stats(Arc::new(super::PipelineStats::default()))
    }

    pub fn with_stats(stats: Arc<super::PipelineStats>) -> Self {
        let mappings = Arc::new(RwLock::new(GpuMappings::new()));
        let guest_memory = GuestMemoryAccess::new(Arc::clone(&mappings));
        let mut pusher = Pusher::new();
        pusher.set_guest_memory_access(Some(guest_memory.clone()));
        Self {
            mappings,
            maxwell3d: Arc::new(Mutex::new(Maxwell3D::new())),
            maxwell_dma: Arc::new(Mutex::new(MaxwellDma::new())),
            fermi_2d: Arc::new(Mutex::new(Fermi2D::new())),
            kepler_compute: Arc::new(Mutex::new(KeplerCompute::new())),
            kepler_memory: Arc::new(Mutex::new(KeplerMemory::new())),
            pusher: Arc::new(Mutex::new(pusher)),
            small_alloc: Arc::new(Mutex::new(flat_allocator::FlatAllocator::new(
                0x0400_0000,
                BIG_VA_BASE,
            ))),
            big_alloc: Arc::new(Mutex::new(flat_allocator::FlatAllocator::new(
                BIG_VA_BASE,
                1u64 << 37,
            ))),
            channels: Arc::new(Mutex::new(HashMap::new())),
            stats,
            pending_syncpoint_events: Mutex::new(VecDeque::new()),
            guest_memory,
            decoder_stub_engines: Mutex::new(StubEngines {
                maxwell_dma: MaxwellDma::new(),
                fermi_2d: Fermi2D::new(),
                kepler_compute: KeplerCompute::new(),
                kepler_memory: KeplerMemory::new(),
            }),
        }
    }

    pub(crate) fn record_syncpoint_completion(&self, fd: u32, syncpt_id: u32, threshold: u32) {
        if crate::kick_timeline_enabled() {
            log::warn!(
                "[ktl] us={} complete fd={} syncpt={} threshold={}",
                crate::timeline_us(),
                fd,
                syncpt_id,
                threshold
            );
        }
        self.pending_syncpoint_events
            .lock()
            .push_back(PendingSyncpointEvent::Completion {
                fd,
                syncpt_id,
                threshold,
            });
        nexium_common::host_wake::signal();
    }

    pub(crate) fn syncpoint_events(
        &self,
    ) -> parking_lot::MutexGuard<'_, VecDeque<PendingSyncpointEvent>> {
        self.pending_syncpoint_events.lock()
    }

    pub(crate) fn install_prep_thread(
        &self,
        resources: prep::PrepThreadResources,
        behavior: prep::PrepThreadBehavior,
    ) -> bool {
        let mut pusher = self.lock_pusher(concat!("gpu/mod.rs:", line!()));
        let previous = std::mem::replace(
            &mut pusher.prep,
            prep::PrepLane::Inline(prep::PrepState::new()),
        );
        let prep::PrepLane::Inline(state) = previous else {
            pusher.prep = previous;
            return false;
        };
        let handle = prep::spawn_prep_thread(state, resources, behavior);
        pusher.prep = prep::PrepLane::Threaded(handle);
        true
    }

    pub(crate) fn prep_present(
        &self,
        job: crate::render_thread::RenderJob,
        flush_small_rts: bool,
        on_prepared: Option<crate::PresentPrepared>,
    ) -> Result<(), (crate::render_thread::RenderJob, Option<crate::PresentPrepared>)> {
        let mut pusher = self.lock_pusher(concat!("gpu/mod.rs:", line!()));
        match &mut pusher.prep {
            prep::PrepLane::Threaded(handle) => {
                match handle.send_recover(prep::PrepEvent::Present {
                    job,
                    flush_small_rts,
                    on_prepared,
                }) {
                    Ok(()) => Ok(()),
                    Err(prep::PrepEvent::Present {
                        job, on_prepared, ..
                    }) => Err((job, on_prepared)),
                    Err(_) => unreachable!("prep present returned a different event"),
                }
            }
            prep::PrepLane::Inline(_) => Err((job, on_prepared)),
        }
    }

    pub(crate) fn prep_drain_barrier(
        &self,
        done: crossbeam::channel::Sender<bool>,
        flush_small_rts: bool,
    ) -> prep::PrepBarrierDispatch {
        let mut pusher = self.lock_pusher(concat!("gpu/mod.rs:", line!()));
        match &mut pusher.prep {
            prep::PrepLane::Threaded(handle) => {
                match handle.send_recover(prep::PrepEvent::DrainBarrier {
                    done,
                    flush_small_rts,
                }) {
                    Ok(()) => prep::PrepBarrierDispatch::Queued,
                    Err(_) => prep::PrepBarrierDispatch::Disconnected,
                }
            }
            prep::PrepLane::Inline(_) => prep::PrepBarrierDispatch::Inline,
        }
    }

    fn drain_prep_thread_matching(
        &self,
        flush_small_rts: bool,
        behavior: Option<prep::PrepThreadBehavior>,
    ) -> bool {
        let (done_tx, done_rx) = crossbeam::channel::bounded(1);
        let failed = {
            let mut pusher = self.lock_pusher(concat!("gpu/mod.rs:", line!()));
            let prep::PrepLane::Threaded(handle) = &mut pusher.prep else {
                return true;
            };
            if behavior.is_some_and(|behavior| handle.behavior() != behavior) {
                return true;
            }
            let failed = handle.failure_latch();
            if handle
                .send_recover(prep::PrepEvent::DrainBarrier {
                    done: done_tx,
                    flush_small_rts,
                })
                .is_err()
            {
                return false;
            }
            failed
        };
        let completed = match done_rx.recv_timeout(std::time::Duration::from_secs(3)) {
            Ok(completed) => completed,
            Err(error) => {
                log::error!("[gpu-prep] drain barrier failed: {error}");
                false
            }
        };
        if !completed {
            failed.store(true, std::sync::atomic::Ordering::Release);
        }
        completed
    }

    pub(crate) fn drain_prep_after_kick(&self) -> bool {
        self.drain_prep_thread_matching(false, Some(prep::PrepThreadBehavior::DrainEachKick))
    }

    pub(crate) fn drain_prep_thread(&self, flush_small_rts: bool) -> bool {
        self.drain_prep_thread_matching(flush_small_rts, None)
    }

    pub(crate) fn shutdown_prep_thread(&self, flush_small_rts: bool) -> bool {
        let mut pusher = self.lock_pusher(concat!("gpu/mod.rs:", line!()));
        let previous = std::mem::replace(
            &mut pusher.prep,
            prep::PrepLane::Inline(prep::PrepState::new()),
        );
        match previous {
            prep::PrepLane::Inline(state) => {
                pusher.prep = prep::PrepLane::Inline(state);
                true
            }
            prep::PrepLane::Threaded(handle) => match handle.shutdown(flush_small_rts) {
                prep::PrepThreadShutdown::Joined { state, drained } => {
                    pusher.prep = prep::PrepLane::Inline(state);
                    if !drained {
                        log::error!("[gpu-prep] shutdown barrier failed");
                    }
                    drained
                }
                prep::PrepThreadShutdown::Panicked { drained } => {
                    log::error!(
                        "[gpu-prep] worker panicked during shutdown drained={}",
                        drained
                    );
                    false
                }
            },
        }
    }

    pub fn set_guest_memory_writer(&self, writer: GuestMemoryWriter) {
        self.guest_memory.set_writer(Some(writer));
    }

    pub fn alloc_gpu_va(&self, size: u64) -> u64 {
        self.alloc_va(size, false)
    }

    pub fn alloc_gpu_va_aligned(&self, size: u64, align: u64) -> u64 {
        self.alloc_va_with_page_size(size, align.max(0x1000))
    }

    pub fn alloc_va(&self, size: u64, big: bool) -> u64 {
        self.alloc_va_with_page_size(size, if big { 0x10000 } else { 0x1000 })
    }

    pub fn alloc_va_with_page_size(&self, size: u64, page: u64) -> u64 {
        if page < 0x1000 || !page.is_power_of_two() {
            return 0;
        }
        let alloc = if page > 0x1000 {
            &self.big_alloc
        } else {
            &self.small_alloc
        };
        let Some(padded) = size.checked_add(page - 1).map(|value| value & !(page - 1)) else {
            return 0;
        };
        alloc.lock().allocate_aligned(padded, page)
    }

    pub fn alloc_va_fixed(&self, gpu_va: u64, size: u64) -> bool {
        if size == 0 {
            return false;
        }
        let (alloc, page) = if gpu_va >= BIG_VA_BASE {
            (&self.big_alloc, 0x10000u64)
        } else {
            (&self.small_alloc, 0x1000u64)
        };
        let base = gpu_va & !(page - 1);
        let Some(padded) = (gpu_va - base)
            .checked_add(size)
            .and_then(|value| value.checked_add(page - 1))
            .map(|value| value & !(page - 1))
        else {
            return false;
        };
        alloc.lock().allocate_fixed(base, padded)
    }

    pub fn alloc_va_fixed_exclusive(&self, gpu_va: u64, size: u64) -> bool {
        if size == 0 {
            return false;
        }
        let (alloc, page) = if gpu_va >= BIG_VA_BASE {
            (&self.big_alloc, 0x10000u64)
        } else {
            (&self.small_alloc, 0x1000u64)
        };
        let base = gpu_va & !(page - 1);
        let Some(padded) = (gpu_va - base)
            .checked_add(size)
            .and_then(|value| value.checked_add(page - 1))
            .map(|value| value & !(page - 1))
        else {
            return false;
        };
        alloc.lock().allocate_fixed_exclusive(base, padded)
    }

    pub fn free_va(&self, gpu_va: u64, size: u64) -> bool {
        if size == 0 {
            return false;
        }
        let (alloc, page) = if gpu_va >= BIG_VA_BASE {
            (&self.big_alloc, 0x10000u64)
        } else {
            (&self.small_alloc, 0x1000u64)
        };
        let base = gpu_va & !(page - 1);
        let Some(padded) = (gpu_va - base)
            .checked_add(size)
            .and_then(|value| value.checked_add(page - 1))
            .map(|value| value & !(page - 1))
        else {
            return false;
        };
        alloc.lock().free(base, padded)
    }

    pub(crate) fn lock_pusher(&self, site: &'static str) -> PusherGuard<'_> {
        let started = std::time::Instant::now();
        let mut reported = 0u64;
        loop {
            if let Some(guard) = self.pusher.try_lock_for(std::time::Duration::from_secs(1)) {
                watchdog::pusher_lock_acquired(site);
                return PusherGuard { guard };
            }
            let waited = started.elapsed();
            if waited.as_secs() / 2 > reported {
                reported = waited.as_secs() / 2;
                watchdog::pusher_lock_wait_report(site, waited);
            }
        }
    }

    pub fn submit_gpfifo(
        &self,
        address: u64,
        num_entries: u32,
        mem_read: impl Fn(u64, &mut [u8]) -> bool,
        mem_write: impl Fn(u64, &[u8]) -> bool,
        mem_copy: impl Fn(u64, u64, usize) -> bool,
        on_complete: Option<Box<dyn FnOnce() + Send>>,
    ) -> (u32, u32) {
        self.submit_gpfifo_with_boundary(
            address,
            num_entries,
            mem_read,
            mem_write,
            mem_copy,
            true,
            eager_small_rt_writeback_enabled(),
            on_complete,
        )
    }

    pub fn snapshot_gpfifo_entries(
        &self,
        address: u64,
        num_entries: u32,
        mem_read: impl Fn(u64, &mut [u8]) -> bool,
    ) -> Option<Vec<CommandListHeader>> {
        let mappings = self.mappings.read();
        mappings.cpu_address_for(address)?;
        let bytes_needed = (num_entries as usize).checked_mul(8)?;
        let mut bytes = vec![0u8; bytes_needed];
        pusher::read_gpu_scattered(&mappings, address, &mut bytes, &mem_read);
        Some(pusher::decode_command_list_headers(&bytes))
    }

    pub fn submit_gpfifo_soft(
        &self,
        address: u64,
        num_entries: u32,
        mem_read: impl Fn(u64, &mut [u8]) -> bool,
        mem_write: impl Fn(u64, &[u8]) -> bool,
        mem_copy: impl Fn(u64, u64, usize) -> bool,
        on_complete: Option<Box<dyn FnOnce() + Send>>,
    ) -> (u32, u32) {
        self.submit_gpfifo_with_boundary(
            address,
            num_entries,
            mem_read,
            mem_write,
            mem_copy,
            false,
            true,
            on_complete,
        )
    }

    pub fn submit_gpfifo_soft_deferred(
        &self,
        address: u64,
        num_entries: u32,
        mem_read: impl Fn(u64, &mut [u8]) -> bool,
        mem_write: impl Fn(u64, &[u8]) -> bool,
        mem_copy: impl Fn(u64, u64, usize) -> bool,
        on_complete: Option<Box<dyn FnOnce() + Send>>,
    ) -> (u32, u32) {
        self.submit_gpfifo_with_boundary(
            address,
            num_entries,
            mem_read,
            mem_write,
            mem_copy,
            false,
            false,
            on_complete,
        )
    }

    fn submit_gpfifo_with_boundary(
        &self,
        address: u64,
        num_entries: u32,
        mem_read: impl Fn(u64, &mut [u8]) -> bool,
        mem_write: impl Fn(u64, &[u8]) -> bool,
        mem_copy: impl Fn(u64, u64, usize) -> bool,
        hard_after: bool,
        writeback_small_rts: bool,
        on_complete: Option<Box<dyn FnOnce() + Send>>,
    ) -> (u32, u32) {
        let kp_total = pusher::kickprof::kick_start();
        let kp_locks = pusher::kickprof::start();
        let mut pusher = self.lock_pusher(concat!("gpu/mod.rs:", line!()));
        let (on_complete, completion_gate) = gate_syncpoint_completion(on_complete);
        let threaded = pusher.prep.is_threaded();
        let mut maxwell = self.maxwell3d.lock();
        let mut stub = threaded.then(|| self.decoder_stub_engines.lock());
        let mut dma_guard = (!threaded).then(|| self.maxwell_dma.lock());
        let mut fermi_guard = (!threaded).then(|| self.fermi_2d.lock());
        let mut kc_guard = (!threaded).then(|| self.kepler_compute.lock());
        let mut km_guard = (!threaded).then(|| self.kepler_memory.lock());
        let (maxwell_dma, fermi_2d, kepler_compute, kepler_memory) = match stub.as_mut() {
            Some(stub) => {
                let stub = &mut **stub;
                (
                    &mut stub.maxwell_dma,
                    &mut stub.fermi_2d,
                    &mut stub.kepler_compute,
                    &mut stub.kepler_memory,
                )
            }
            None => (
                &mut **dma_guard.as_mut().unwrap(),
                &mut **fermi_guard.as_mut().unwrap(),
                &mut **kc_guard.as_mut().unwrap(),
                &mut **km_guard.as_mut().unwrap(),
            ),
        };
        let mappings = self.mappings.read();
        pusher::kickprof::add(pusher::kickprof::LOCKS, kp_locks);

        if hard_after {
            pusher.process_gpfifo(
                address,
                num_entries,
                &mappings,
                &mut *maxwell,
                maxwell_dma,
                fermi_2d,
                kepler_compute,
                kepler_memory,
                &*self.stats,
                &mem_read,
                &mem_write,
                &mem_copy,
                writeback_small_rts,
                on_complete,
            );
        } else if writeback_small_rts {
            pusher.process_gpfifo_soft(
                address,
                num_entries,
                &mappings,
                &mut *maxwell,
                maxwell_dma,
                fermi_2d,
                kepler_compute,
                kepler_memory,
                &*self.stats,
                &mem_read,
                &mem_write,
                &mem_copy,
                on_complete,
            );
        } else {
            pusher.process_gpfifo_soft_deferred(
                address,
                num_entries,
                &mappings,
                &mut *maxwell,
                maxwell_dma,
                fermi_2d,
                kepler_compute,
                kepler_memory,
                &*self.stats,
                &mem_read,
                &mem_write,
                &mem_copy,
                on_complete,
            );
        }
        let mut embedded_incrs = std::mem::take(&mut pusher.pending_syncpt_incrs);
        merge_engine_syncpt_incrs(&mut embedded_incrs, maxwell.take_pending_syncpt_incrs());
        self.record_embedded_syncpt_incrs(embedded_incrs);
        if let Some(gate) = completion_gate {
            gate.release();
        }
        pusher.syncpt_value = pusher.syncpt_value.wrapping_add(2);
        pusher::kickprof::kick_done(kp_total);

        let syncpt_id = 0u32;
        let syncpt_value = pusher.syncpt_value;
        (syncpt_id, syncpt_value)
    }

    pub(crate) fn record_embedded_syncpt_incrs(&self, incrs: Vec<(u32, u32)>) {
        if incrs.is_empty() {
            return;
        }
        let mut events = self.pending_syncpoint_events.lock();
        let mut recorded = false;
        for (id, count) in incrs {
            if id == 0 || count == 0 {
                if id == 0 && count != 0 {
                    log::warn!("[syncpt-orphan] rejected increment id=0 count={}", count);
                }
                continue;
            }
            if crate::kick_timeline_enabled() {
                log::warn!(
                    "[ktl] us={} incr syncpt={} count={}",
                    crate::timeline_us(),
                    id,
                    count
                );
            }
            events.push_back(PendingSyncpointEvent::Increment {
                syncpt_id: id,
                count,
            });
            recorded = true;
        }
        drop(events);
        if recorded {
            nexium_common::host_wake::signal();
        }
    }

    pub fn process_inline_gpfifo(
        &self,
        entries: &[CommandListHeader],
        mem_read: impl Fn(u64, &mut [u8]) -> bool,
        mem_write: impl Fn(u64, &[u8]) -> bool,
        mem_copy: impl Fn(u64, u64, usize) -> bool,
        on_complete: Option<Box<dyn FnOnce() + Send>>,
    ) -> (u32, u32) {
        self.process_inline_gpfifo_with_options(
            entries,
            mem_read,
            mem_write,
            mem_copy,
            true,
            eager_small_rt_writeback_enabled(),
            on_complete,
        )
    }

    pub fn process_inline_gpfifo_soft(
        &self,
        entries: &[CommandListHeader],
        mem_read: impl Fn(u64, &mut [u8]) -> bool,
        mem_write: impl Fn(u64, &[u8]) -> bool,
        mem_copy: impl Fn(u64, u64, usize) -> bool,
        on_complete: Option<Box<dyn FnOnce() + Send>>,
    ) -> (u32, u32) {
        self.process_inline_gpfifo_with_options(
            entries,
            mem_read,
            mem_write,
            mem_copy,
            false,
            true,
            on_complete,
        )
    }

    pub fn process_inline_gpfifo_soft_deferred(
        &self,
        entries: &[CommandListHeader],
        mem_read: impl Fn(u64, &mut [u8]) -> bool,
        mem_write: impl Fn(u64, &[u8]) -> bool,
        mem_copy: impl Fn(u64, u64, usize) -> bool,
        on_complete: Option<Box<dyn FnOnce() + Send>>,
    ) -> (u32, u32) {
        self.process_inline_gpfifo_with_options(
            entries,
            mem_read,
            mem_write,
            mem_copy,
            false,
            false,
            on_complete,
        )
    }

    fn process_inline_gpfifo_with_options(
        &self,
        entries: &[CommandListHeader],
        mem_read: impl Fn(u64, &mut [u8]) -> bool,
        mem_write: impl Fn(u64, &[u8]) -> bool,
        mem_copy: impl Fn(u64, u64, usize) -> bool,
        hard_after: bool,
        writeback_small_rts: bool,
        on_complete: Option<Box<dyn FnOnce() + Send>>,
    ) -> (u32, u32) {
        let profile = nvprof_enabled();
        let kp_total = pusher::kickprof::kick_start();
        let kp_locks = pusher::kickprof::start();
        let t0 = std::time::Instant::now();
        let mut pusher = self.lock_pusher(concat!("gpu/mod.rs:", line!()));
        let (on_complete, completion_gate) = gate_syncpoint_completion(on_complete);
        let threaded = pusher.prep.is_threaded();
        let mut maxwell = self.maxwell3d.lock();
        let mut stub = threaded.then(|| self.decoder_stub_engines.lock());
        let mut dma_guard = (!threaded).then(|| self.maxwell_dma.lock());
        let mut fermi_guard = (!threaded).then(|| self.fermi_2d.lock());
        let mut kc_guard = (!threaded).then(|| self.kepler_compute.lock());
        let mut km_guard = (!threaded).then(|| self.kepler_memory.lock());
        let (maxwell_dma, fermi_2d, kepler_compute, kepler_memory) = match stub.as_mut() {
            Some(stub) => {
                let stub = &mut **stub;
                (
                    &mut stub.maxwell_dma,
                    &mut stub.fermi_2d,
                    &mut stub.kepler_compute,
                    &mut stub.kepler_memory,
                )
            }
            None => (
                &mut **dma_guard.as_mut().unwrap(),
                &mut **fermi_guard.as_mut().unwrap(),
                &mut **kc_guard.as_mut().unwrap(),
                &mut **km_guard.as_mut().unwrap(),
            ),
        };
        let mappings = self.mappings.read();
        pusher::kickprof::add(pusher::kickprof::LOCKS, kp_locks);
        let locks_ms = if profile { elapsed_ms(t0) } else { 0.0 };

        pusher.prep_kick_begin();

        let t_entries = std::time::Instant::now();
        pusher::record_gpfifo_entries(entries, &mappings);
        if pusher::direct_forensics() && entries.iter().any(|e| e.entry_count() > 4096) {
            use std::sync::atomic::{AtomicU32, Ordering};
            static N: AtomicU32 = AtomicU32::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 8 {
                let raw: Vec<String> = entries
                    .iter()
                    .map(|e| {
                        format!(
                            "{:08x}:{:08x}(va={:#x},n={})",
                            e.address_lo,
                            e.address_hi_and_count,
                            e.address(),
                            e.entry_count()
                        )
                    })
                    .collect();
                log::warn!(
                    "[el-dump] inline num_entries={} {}",
                    entries.len(),
                    raw.join(" ")
                );
                for entry in entries.iter().filter(|e| e.entry_count() > 4096).take(2) {
                    let base = entry.address() & !0xF_FFFF;
                    let cf = entry.entry_count() as u64;
                    for (tag, probe) in [
                        ("recs", base + cf.saturating_sub(0x30)),
                        ("segva", entry.address().saturating_sub(0x20)),
                    ] {
                        let mut buf = vec![0u8; 0x120];
                        pusher::read_gpu_scattered(&mappings, probe, &mut buf, &mem_read);
                        let words: Vec<String> = buf
                            .chunks_exact(4)
                            .map(|c| {
                                format!("{:08x}", u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            })
                            .collect();
                        log::warn!(
                            "[ctrl-dump] {} entry_va={:#x} cf={:#x} probe={:#x} map={:?} {}",
                            tag,
                            entry.address(),
                            cf,
                            probe,
                            mappings
                                .mapping_at(probe)
                                .map(|(g, s, c)| { format!("gpu={:#x}+{:#x} cpu={:#x}", g, s, c) }),
                            words.join(" ")
                        );
                    }
                    let base = entry.address() & !0xF_FFFF;
                    let mut cursor = base;
                    let mut spans: Vec<String> = Vec::new();
                    while cursor < base + 0x10_0000 && spans.len() < 16 {
                        match mappings.mapping_at(cursor) {
                            Some((g, s, c)) => {
                                spans.push(format!("gpu={:#x}+{:#x} cpu={:#x}", g, s, c));
                                cursor = g + s;
                            }
                            None => {
                                spans.push(format!("HOLE@{:#x}", cursor));
                                cursor += 0x1000;
                            }
                        }
                    }
                    log::warn!("[ctrl-dump] arena {:#x} spans: {}", base, spans.join(" | "));
                }
            }
        }
        let trace_entries = std::env::var_os("NEXIUM_ENTRY_TRACE").is_some();
        for entry in entries {
            if trace_entries {
                let va = entry.address();
                let count = entry.entry_count();
                let mut head = [0u8; 16];
                pusher::read_gpu_scattered(&mappings, va, &mut head, &mem_read);
                let words: Vec<String> = head
                    .chunks_exact(4)
                    .map(|c| format!("{:08x}", u32::from_le_bytes([c[0], c[1], c[2], c[3]])))
                    .collect();
                log::info!(
                    "[entry] va={:#x} count={} cpu={:?} head=[{}] methods_before={}",
                    va,
                    count,
                    mappings.cpu_address_for(va),
                    words.join(","),
                    self.stats
                        .methods_dispatched
                        .load(std::sync::atomic::Ordering::Relaxed)
                );
            }
            pusher.process_entry(
                entry,
                &mappings,
                &mut *maxwell,
                maxwell_dma,
                fermi_2d,
                kepler_compute,
                kepler_memory,
                &*self.stats,
                &mem_read,
                &mem_write,
                &mem_copy,
            );
        }
        let entries_ms = if profile { elapsed_ms(t_entries) } else { 0.0 };
        let t_flush = std::time::Instant::now();
        pusher.prep_kick_end(
            hard_after,
            writeback_small_rts,
            on_complete,
            &mappings,
            &mem_read,
            &mem_write,
        );
        vk_dispatch::guest_probe(&mappings, &mem_read);
        let flush_ms = if profile { elapsed_ms(t_flush) } else { 0.0 };
        let mut embedded_incrs = std::mem::take(&mut pusher.pending_syncpt_incrs);
        merge_engine_syncpt_incrs(&mut embedded_incrs, maxwell.take_pending_syncpt_incrs());
        self.record_embedded_syncpt_incrs(embedded_incrs);
        if let Some(gate) = completion_gate {
            gate.release();
        }
        pusher.syncpt_value = pusher.syncpt_value.wrapping_add(2);
        pusher::kickprof::kick_done(kp_total);
        if profile {
            log::warn!(
                "[nvprof] inline entries={} locks_ms={:.3} entries_ms={:.3} flush_ms={:.3} total_ms={:.3}",
                entries.len(),
                locks_ms,
                entries_ms,
                flush_ms,
                elapsed_ms(t0)
            );
        }
        (0, pusher.syncpt_value)
    }

    pub fn flush_small_rt_writebacks(&self, mem_write: impl Fn(u64, &[u8]) -> bool) -> bool {
        let mut pusher = self.lock_pusher(concat!("gpu/mod.rs:", line!()));
        let mappings = self.mappings.read();
        let Some(state) = pusher.prep.inline_state() else {
            return false;
        };
        let Some(renderer) = state.renderer.clone() else {
            return false;
        };
        state.writeback_small_rts(&renderer, &mappings, &mem_write)
    }

    pub fn flush_cpu_readable_rt_writebacks(&self, mem_write: impl Fn(u64, &[u8]) -> bool) -> bool {
        let mut pusher = self.lock_pusher(concat!("gpu/mod.rs:", line!()));
        let mappings = self.mappings.read();
        let Some(state) = pusher.prep.inline_state() else {
            return false;
        };
        let Some(renderer) = state.renderer.clone() else {
            return false;
        };
        state.writeback_cpu_readable_rts(&renderer, &mappings, &mem_write)
    }

    pub(crate) fn flush_prepared_draw_packets(&self) -> bool {
        match self
            .lock_pusher(concat!("gpu/mod.rs:", line!()))
            .prep
            .inline_state()
        {
            Some(state) => state.flush_prepared_draw_packets(),
            None => false,
        }
    }

    pub(crate) fn has_prepared_draw_packets(&self) -> bool {
        match self
            .lock_pusher(concat!("gpu/mod.rs:", line!()))
            .prep
            .inline_state()
        {
            Some(state) => state.has_prepared_draw_packets(),
            None => false,
        }
    }

    pub fn read_rt(
        &self,
        mem_read: impl Fn(u64, &mut [u8]) -> bool,
    ) -> Option<(u32, u32, Vec<u8>)> {
        let mappings = self.mappings.read();
        let maxwell = self.maxwell3d.lock();
        let rt = &maxwell.regs.rt[0];
        if rt.width == 0 || rt.height == 0 {
            return None;
        }
        let gpu_va = ((rt.address_hi as u64) << 32) | rt.address_lo as u64;
        let cpu = mappings.cpu_address_for(gpu_va)?;
        let size = (rt.width * rt.height * 4) as usize;
        let mut buf = vec![0u8; size];
        if mem_read(cpu, &mut buf) {
            Some((rt.width, rt.height, buf))
        } else {
            None
        }
    }
}

impl Default for GpuContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        eager_small_rt_writeback_value_enabled, prep, GpuContext, GpuMapping, GpuMappingChange,
        GpuMappings, GuestMemoryAccess, PendingSyncpointEvent,
        MAPPING_INDEX_REBUILD_AFTER_STALE_LOOKUPS,
    };
    use parking_lot::RwLock;
    use std::sync::{Arc, Mutex as StdMutex};

    #[test]
    fn eager_small_rt_writeback_is_an_explicit_compatibility_opt_in() {
        use std::ffi::OsStr;

        assert!(!eager_small_rt_writeback_value_enabled(None));
        for disabled in ["", "0", "false", "OFF", " no ", "unexpected"] {
            assert!(!eager_small_rt_writeback_value_enabled(Some(OsStr::new(
                disabled
            ))));
        }
        for enabled in ["1", "true", "TRUE", "on", " yes "] {
            assert!(eager_small_rt_writeback_value_enabled(Some(OsStr::new(
                enabled
            ))));
        }
    }

    #[test]
    fn syncpoint_events_are_owned_by_the_gpu_context() {
        let first = GpuContext::new();
        let second = GpuContext::new();

        first.record_embedded_syncpt_incrs(vec![(7, 2), (0, 1)]);

        assert_eq!(
            first.syncpoint_events().drain(..).collect::<Vec<_>>(),
            vec![PendingSyncpointEvent::Increment {
                syncpt_id: 7,
                count: 2
            }]
        );
        assert!(first.syncpoint_events().is_empty());
        assert!(second.syncpoint_events().is_empty());
    }

    #[test]
    fn guest_memory_access_resolves_latest_mapping_when_written() {
        let mappings = Arc::new(RwLock::new(GpuMappings::new()));
        mappings.write().add(0x1000, 0x1000, 0x1_0000, 1);
        let access = GuestMemoryAccess::new(Arc::clone(&mappings));
        let observed = Arc::new(StdMutex::new(Vec::new()));
        let writer_observed = Arc::clone(&observed);
        access.set_writer(Some(Arc::new(move |cpu_addr, bytes| {
            writer_observed
                .lock()
                .unwrap()
                .push((cpu_addr, bytes.to_vec()));
            true
        })));

        assert_eq!(mappings.write().remove(0x1000), Some(0x1000));
        mappings.write().add(0x1000, 0x1000, 0x2_0000, 2);

        assert_eq!(
            access.write_gpu(0x1080, &[1, 2, 3, 4]),
            Some((0x2_0080, true))
        );
        assert_eq!(
            *observed.lock().unwrap(),
            vec![(0x2_0080, vec![1, 2, 3, 4])]
        );
    }

    #[test]
    fn guest_memory_access_keeps_mapping_locked_through_write() {
        let mappings = Arc::new(RwLock::new(GpuMappings::new()));
        mappings.write().add(0x1000, 0x1000, 0x1_0000, 1);
        let access = GuestMemoryAccess::new(Arc::clone(&mappings));
        let writer_mappings = Arc::clone(&mappings);
        access.set_writer(Some(Arc::new(move |_, _| {
            assert!(writer_mappings.try_write().is_none());
            true
        })));

        assert_eq!(access.write_gpu(0x1000, &[7]), Some((0x1_0000, true)));
    }

    #[test]
    fn guest_memory_access_versions_only_successful_nonempty_writes() {
        let mappings = Arc::new(RwLock::new(GpuMappings::new()));
        mappings.write().add(0x1000, 0x1000, 0x1_0000, 1);
        let access = GuestMemoryAccess::new(Arc::clone(&mappings));
        access.set_writer(Some(Arc::new(|_, bytes| bytes.first() == Some(&1))));

        let before_success = nexium_gpu::tex_invalidate::region_gen(0x1080);
        assert_eq!(
            access.write_gpu(0x1080, &[1, 2, 3, 4]),
            Some((0x1_0080, true))
        );
        assert_ne!(
            nexium_gpu::tex_invalidate::region_gen(0x1080),
            before_success
        );
        let before_failed = nexium_gpu::tex_invalidate::region_gen(0x1090);
        assert_eq!(
            access.write_gpu(0x1090, &[0, 2, 3, 4]),
            Some((0x1_0090, false))
        );
        assert_eq!(
            nexium_gpu::tex_invalidate::region_gen(0x1090),
            before_failed
        );
        let before_empty = nexium_gpu::tex_invalidate::region_gen(0x10a0);
        assert_eq!(access.write_gpu(0x10a0, &[]), Some((0x1_00a0, false)));
        assert_eq!(nexium_gpu::tex_invalidate::region_gen(0x10a0), before_empty);
    }

    #[test]
    fn present_and_queue_barrier_use_the_hard_prepared_packet_drain() {
        let gpu = GpuContext::new();

        assert!(gpu.flush_prepared_draw_packets());
        assert!(gpu.flush_prepared_draw_packets());

        assert_eq!(
            gpu.pusher
                .lock()
                .inline_prep()
                .prepared_packet_drain_counts(),
            (2, 0)
        );
    }

    #[test]
    fn prep_thread_barrier_shutdown_restores_inline_lane() {
        let gpu = GpuContext::new();
        let read: crate::AsyncMemoryRead = Arc::new(|_, bytes| {
            bytes.fill(0);
            true
        });
        let write: crate::AsyncMemoryWrite = Arc::new(|_, _| true);
        let copy: crate::AsyncMemoryCopy = Arc::new(|_, _, _| true);
        assert!(gpu.install_prep_thread(
            prep::PrepThreadResources {
                maxwell_dma: Arc::clone(&gpu.maxwell_dma),
                fermi_2d: Arc::clone(&gpu.fermi_2d),
                kepler_compute: Arc::clone(&gpu.kepler_compute),
                kepler_memory: Arc::clone(&gpu.kepler_memory),
                mappings: Arc::clone(&gpu.mappings),
                stats: Arc::clone(&gpu.stats),
                mem_read: read,
                mem_write: write,
                mem_copy: copy,
            },
            prep::PrepThreadBehavior::Pipeline,
        ));
        assert!(gpu.pusher.lock().prep.is_threaded());
        let (done_tx, done_rx) = crossbeam::channel::bounded(1);
        assert_eq!(
            gpu.prep_drain_barrier(done_tx, false),
            prep::PrepBarrierDispatch::Queued
        );
        assert!(done_rx.recv().unwrap());

        assert!(gpu.shutdown_prep_thread(false));
        assert!(!gpu.pusher.lock().prep.is_threaded());
        assert!(gpu.shutdown_prep_thread(false));
    }

    #[test]
    fn drain_each_kick_waits_for_kick_completion() {
        let gpu = GpuContext::new();
        let read: crate::AsyncMemoryRead = Arc::new(|_, bytes| {
            bytes.fill(0);
            true
        });
        let write: crate::AsyncMemoryWrite = Arc::new(|_, _| true);
        let copy: crate::AsyncMemoryCopy = Arc::new(|_, _, _| true);
        let completed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let completed_callback = Arc::clone(&completed);
        assert!(gpu.install_prep_thread(
            prep::PrepThreadResources {
                maxwell_dma: Arc::clone(&gpu.maxwell_dma),
                fermi_2d: Arc::clone(&gpu.fermi_2d),
                kepler_compute: Arc::clone(&gpu.kepler_compute),
                kepler_memory: Arc::clone(&gpu.kepler_memory),
                mappings: Arc::clone(&gpu.mappings),
                stats: Arc::clone(&gpu.stats),
                mem_read: Arc::clone(&read),
                mem_write: Arc::clone(&write),
                mem_copy: Arc::clone(&copy),
            },
            prep::PrepThreadBehavior::DrainEachKick,
        ));

        gpu.process_inline_gpfifo(
            &[],
            move |address, bytes| read(address, bytes),
            move |address, bytes| write(address, bytes),
            move |source, destination, size| copy(source, destination, size),
            Some(Box::new(move || {
                completed_callback.store(true, std::sync::atomic::Ordering::Release)
            })),
        );

        assert!(gpu.drain_prep_after_kick());
        assert!(completed.load(std::sync::atomic::Ordering::Acquire));
        assert!(gpu.shutdown_prep_thread(false));
    }

    #[test]
    fn inline_prep_barrier_is_explicit() {
        let gpu = GpuContext::new();
        let (done_tx, done_rx) = crossbeam::channel::bounded(1);

        assert_eq!(
            gpu.prep_drain_barrier(done_tx, false),
            prep::PrepBarrierDispatch::Inline
        );
        assert!(done_rx.recv().is_err());
    }

    #[test]
    fn mapping_segment_index_matches_linear_scans() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for round in 0..48 {
            let mut mappings = GpuMappings::new();
            let count = 1 + (next() % 40) as usize;
            for record in 0..count {
                let gpu_va = (next() % 0x40) * 0x1000;
                let size = match next() % 8 {
                    0 => 0,
                    _ => (1 + next() % 6) * 0x1000 + (next() % 3) * 0x200,
                };
                let cpu_addr = 0x10_0000 + (next() % 0x20) * 0x1000;
                let sparse = round % 3 == 0 && next() % 5 == 0;
                mappings.mappings.push(GpuMapping {
                    gpu_va,
                    size,
                    cpu_addr,
                    nvmap_id: record as u32 + 1,
                    epoch: record as u64 + 1,
                    record_id: record as u64 + 1,
                    sparse,
                    owner_record_id: None,
                    owned_va_range: None,
                    as_gpu_fd: None,
                    as_gpu_allocation_base: None,
                    as_gpu_root: false,
                    as_gpu_unmap_barrier: false,
                });
                mappings.generation = mappings.generation.wrapping_add(1).max(1);
            }
            let index = mappings.segment_index().expect("index builds on first use");
            for probe in 0..600u64 {
                let gpu_va = probe * 0x100 + (next() % 0x100);
                assert_eq!(
                    index.lookup(gpu_va),
                    mappings.mapping_lookup_linear(gpu_va),
                    "round {round} gpu_va {gpu_va:#x}"
                );
            }
            for _ in 0..200 {
                let cpu_addr = 0x10_0000 + next() % 0x2_8000;
                let size = next() % 0x9000;
                assert_eq!(
                    mappings.gpu_regions_for_cpu_range(cpu_addr, size),
                    mappings.gpu_regions_for_cpu_range_linear(cpu_addr, size),
                    "round {round} cpu {cpu_addr:#x}+{size:#x}"
                );
            }
        }
    }

    #[test]
    fn mapping_segment_index_defers_rebuilds_while_mappings_churn() {
        let mut mappings = GpuMappings::new();
        mappings.add(0x1000, 0x1000, 0x1_0000, 1);
        let mut deferred = 0usize;
        let first = loop {
            match mappings.segment_index() {
                Some(index) if index.generation == mappings.generation => break index,
                _ => deferred += 1,
            }
        };
        assert!(deferred < MAPPING_INDEX_REBUILD_AFTER_STALE_LOOKUPS as usize);
        assert_eq!(first.lookup(0x1010), mappings.mapping_lookup_linear(0x1010));
        mappings.add(0x3000, 0x1000, 0x2_0000, 2);
        let mut deferred = 0usize;
        let rebuilt = loop {
            match mappings.segment_index() {
                Some(index) => break index,
                None => deferred += 1,
            }
        };
        assert!(deferred < MAPPING_INDEX_REBUILD_AFTER_STALE_LOOKUPS as usize);
        assert_eq!(rebuilt.generation, mappings.generation);
        assert_eq!(
            rebuilt.lookup(0x3010),
            mappings.mapping_lookup_linear(0x3010)
        );
        assert_eq!(mappings.cpu_address_for(0x3010), Some(0x2_0010));
        assert_eq!(mappings.cpu_address_for(0x1010), Some(0x1_0010));
    }

    #[test]
    fn cpu_range_aliases_include_partial_overlaps() {
        let mut mappings = GpuMappings::new();
        mappings.add(0x1000, 0x1000, 0x1_0000, 7);
        mappings.add(0x3000, 0x800, 0x1_0800, 7);

        assert_eq!(
            mappings.gpu_regions_for_cpu_range(0x1_0700, 0x300),
            vec![(0x1700, 0x300), (0x3000, 0x200)]
        );
        assert!(mappings.gpu_regions_for_cpu_range(0x1_0700, 0).is_empty());
    }

    #[test]
    fn cpu_range_aliases_exclude_shadowed_mapping_records() {
        let mut mappings = GpuMappings::new();
        mappings.add(0x1000, 0x1000, 0xa000, 1);
        mappings.add(0x3000, 0x1000, 0xa000, 2);
        mappings.add(0x1000, 0x1000, 0xb000, 3);

        assert_eq!(
            mappings.gpu_regions_for_cpu_range(0xa100, 0x100),
            vec![(0x3100, 0x100)]
        );
        assert_eq!(
            mappings.gpu_regions_for_cpu_range(0xb100, 0x100),
            vec![(0x1100, 0x100)]
        );
    }

    #[test]
    fn cpu_range_aliases_split_around_partial_newer_overlays() {
        let mut mappings = GpuMappings::new();
        mappings.add(0x1000, 0x2000, 0xa000, 1);
        mappings.add(0x1800, 0x800, 0xd000, 2);

        assert_eq!(
            mappings.gpu_regions_for_cpu_range(0xa000, 0x2000),
            vec![(0x1000, 0x800), (0x2000, 0x1000)]
        );
        assert_eq!(
            mappings.gpu_regions_for_cpu_range(0xd000, 0x800),
            vec![(0x1800, 0x800)]
        );
    }

    #[test]
    fn mapping_lookup_cache_respects_newest_overlapping_mapping() {
        let mut mappings = GpuMappings::new();
        mappings.add(0x1000, 0x2000, 0x1_0000, 1);
        mappings.add(0x1400, 0x800, 0x2_0000, 2);
        mappings.add(0x1800, 0x200, 0x3_0000, 3);

        assert_eq!(mappings.cpu_address_for(0x1200), Some(0x1_0200));
        let misses_after_miss = super::MAPPING_LOOKUP_CACHE.with(|cache| cache.borrow().misses);

        assert_eq!(mappings.nvmap_id_for(0x1300), Some(1));
        assert_eq!(
            super::MAPPING_LOOKUP_CACHE.with(|cache| cache.borrow().misses),
            misses_after_miss
        );

        assert_eq!(mappings.cpu_address_for(0x1500), Some(0x2_0100));
        assert_eq!(mappings.nvmap_id_for(0x1900), Some(3));
        assert_eq!(mappings.mapping_at(0x1200), Some((0x1000, 0x400, 0x1_0000)));
        assert_eq!(mappings.mapping_at(0x1b00), Some((0x1a00, 0x200, 0x2_0600)));
        assert_eq!(mappings.cpu_range_for(0x1200), Some((0x1_0200, 0x200)));
        assert_eq!(mappings.cpu_range_for(0x1500), Some((0x2_0100, 0x300)));
        assert_eq!(mappings.cpu_range_for(0x1900), Some((0x3_0100, 0x100)));
        assert_eq!(mappings.cpu_range_for(0x1b00), Some((0x2_0700, 0x100)));
        assert_eq!(mappings.cpu_range_for(0x1d00), Some((0x1_0d00, 0x1300)));
    }

    #[test]
    fn mapping_add_classifies_fresh_idempotent_and_replaced_ranges() {
        let mut mappings = GpuMappings::new();
        let base = 0x7f11_3000;
        let cpu = 0x4e22_0000;

        let fresh = mappings.add(base, 0x2000, cpu, 17);
        assert_eq!(fresh, GpuMappingChange::Fresh);
        assert!(!fresh.invalidates_render_targets());
        let original_epoch = mappings.mapping_epoch_for(base + 0x800).unwrap();
        let original_generation = mappings.generation();

        let texture_generation = nexium_gpu::tex_invalidate::region_gen(base);
        let idempotent = mappings.add(base, 0x2000, cpu, 17);
        assert_eq!(idempotent, GpuMappingChange::Idempotent);
        assert!(!idempotent.invalidates_render_targets());
        assert_eq!(
            mappings.mapping_epoch_for(base + 0x800),
            Some(original_epoch)
        );
        assert_eq!(mappings.generation(), original_generation);
        assert_ne!(
            nexium_gpu::tex_invalidate::region_gen(base),
            texture_generation
        );

        assert_eq!(
            mappings.add(base + 0x400, 0x800, cpu + 0x400, 17),
            GpuMappingChange::Idempotent
        );
        assert_eq!(
            mappings.mapping_epoch_for(base + 0x800),
            Some(original_epoch)
        );
        assert_eq!(mappings.iter().count(), 1);

        let replaced_identity = mappings.add(base + 0x400, 0x200, cpu + 0x400, 18);
        assert_eq!(replaced_identity, GpuMappingChange::Replaced);
        assert!(replaced_identity.invalidates_render_targets());

        let replaced = mappings.add(base, 0x2000, cpu + 0x8000, 18);
        assert_eq!(replaced, GpuMappingChange::Replaced);
        assert!(replaced.invalidates_render_targets());
        assert_ne!(
            mappings.mapping_epoch_for(base + 0x800),
            Some(original_epoch)
        );
        assert_eq!(mappings.cpu_address_for(base + 0x800), Some(cpu + 0x8800));

        let disjoint = mappings.add(base + 0x4000, 0x1000, cpu + 0x10_000, 19);
        assert_eq!(disjoint, GpuMappingChange::Fresh);
        assert!(!disjoint.invalidates_render_targets());
    }

    #[test]
    fn mapping_add_classification_uses_the_newest_effective_overlay() {
        let mut mappings = GpuMappings::new();
        let base = 0x7f22_0000;
        let cpu = 0x4f33_0000;
        assert_eq!(mappings.add(base, 0x2000, cpu, 21), GpuMappingChange::Fresh);
        assert_eq!(
            mappings.add(base + 0x800, 0x400, cpu + 0x8000, 22),
            GpuMappingChange::Replaced
        );

        assert_eq!(
            mappings.add(base, 0x2000, cpu, 21),
            GpuMappingChange::Replaced
        );
        let epoch = mappings.mapping_epoch_for(base + 0x900).unwrap();
        assert_eq!(
            mappings.add(base, 0x2000, cpu, 21),
            GpuMappingChange::Idempotent
        );
        assert_eq!(mappings.mapping_epoch_for(base + 0x900), Some(epoch));
    }

    #[test]
    fn any32_lookup_uses_the_newest_effective_overlay() {
        let mut mappings = GpuMappings::new();
        mappings.add(0x1_0000_1000, 0x1000, 0xa000, 1);
        mappings.add(0x2_0000_1000, 0x1000, 0xb000, 2);

        assert_eq!(
            mappings.cpu_address_for_any32(0x1800),
            Some((0x2_0000_1000, 0xb800, 0x800))
        );
    }

    #[test]
    fn any32_lookup_clamps_remaining_length_at_newer_overlays() {
        let mut mappings = GpuMappings::new();
        mappings.add(0x1_0000_1000, 0x3000, 0xa000, 1);
        mappings.add(0x2_0000_2000, 0x800, 0xd000, 2);

        assert_eq!(
            mappings.cpu_address_for_any32(0x1800),
            Some((0x1_0000_1000, 0xa800, 0x800))
        );
        assert_eq!(
            mappings.cpu_address_for_any32(0x2100),
            Some((0x2_0000_2000, 0xd100, 0x700))
        );

        mappings.add_sparse(0x3_0000_3000, 0x400);
        assert_eq!(
            mappings.cpu_address_for_any32(0x2900),
            Some((0x1_0000_1000, 0xb900, 0x700))
        );
        assert_eq!(mappings.cpu_address_for_any32(0x3100), None);
    }

    #[test]
    fn sparse_overlays_shadow_cpu_lookups_and_reverse_aliases_until_refilled() {
        let mut mappings = GpuMappings::new();
        let base = 0x6d20_0000;
        let cpu = 0x4d20_0000;
        mappings.add(base, 0x3000, cpu, 11);

        assert_eq!(
            mappings.add_sparse(base + 0x1000, 0x1000),
            GpuMappingChange::Replaced
        );
        assert_eq!(mappings.cpu_address_for(base + 0x800), Some(cpu + 0x800));
        assert_eq!(mappings.cpu_address_for(base + 0x1800), None);
        assert_eq!(mappings.mapping_at(base + 0x1800), None);
        assert_eq!(mappings.cpu_range_for(base + 0x1800), None);
        assert_eq!(mappings.nvmap_id_for(base + 0x1800), None);
        assert_eq!(mappings.mapping_epoch_for(base + 0x1800), None);
        assert_eq!(
            mappings.gpu_regions_for_cpu_range(cpu, 0x3000),
            vec![(base, 0x1000), (base + 0x2000, 0x1000)]
        );

        assert_eq!(
            mappings.add(base + 0x1000, 0x1000, 0x5d20_0000, 12),
            GpuMappingChange::Replaced
        );
        assert_eq!(mappings.cpu_address_for(base + 0x1800), Some(0x5d20_0800));
    }

    #[test]
    fn replacement_mapping_clears_stale_pitch_identity() {
        let mut mappings = GpuMappings::new();
        let base = 0x6e71_0000;
        mappings.add(base, 0x1000, 0x3e71_0000, 1);
        nexium_gpu::pitch_oracle::record_pitch_dst(base, 0x1000);
        assert!(nexium_gpu::pitch_oracle::is_pitch_dst(base + 0x800));

        assert_eq!(
            mappings.add(base, 0x1000, 0x4e71_0000, 2),
            GpuMappingChange::Replaced
        );
        assert!(!nexium_gpu::pitch_oracle::is_pitch_dst(base + 0x800));
    }

    #[test]
    fn mapping_add_extension_removes_as_one_overlay_and_preserves_overlap_epoch() {
        let mut mappings = GpuMappings::new();
        let base = 0x7f44_0000;
        let cpu = 0x4f55_0000;
        assert_eq!(
            mappings.add(base + 0x1000, 0x1000, cpu + 0x1000, 31),
            GpuMappingChange::Fresh
        );
        let middle_epoch = mappings.mapping_epoch_for(base + 0x1800).unwrap();

        let extended = mappings.add(base, 0x3000, cpu, 31);
        assert_eq!(extended, GpuMappingChange::Extended);
        assert!(!extended.invalidates_render_targets());
        assert_eq!(mappings.mapping_epoch_for(base + 0x800), Some(middle_epoch));
        assert_eq!(
            mappings.mapping_epoch_for(base + 0x1800),
            Some(middle_epoch)
        );
        assert_eq!(
            mappings.mapping_epoch_for(base + 0x2800),
            Some(middle_epoch)
        );
        assert_eq!(mappings.cpu_address_for(base + 0x800), Some(cpu + 0x800));
        assert_eq!(mappings.cpu_address_for(base + 0x2800), Some(cpu + 0x2800));
        assert_eq!(mappings.iter().count(), 2);

        let generation = mappings.generation();
        assert_eq!(
            mappings.add(base, 0x3000, cpu, 31),
            GpuMappingChange::Idempotent
        );
        assert_eq!(mappings.generation(), generation);
        assert_eq!(
            mappings.mapping_epoch_for(base + 0x1800),
            Some(middle_epoch)
        );
        assert_eq!(
            mappings.mapping_epoch_for(base + 0x2800),
            Some(middle_epoch)
        );

        let removed = mappings.remove_with_metadata(base).unwrap();
        assert_eq!((removed.gpu_va, removed.size), (base, 0x3000));
        assert_eq!(removed.cpu_addr, cpu);
        assert_eq!(removed.epoch, middle_epoch);
        assert_eq!(
            removed.changed_gpu_ranges,
            vec![(base, 0x1000), (base + 0x2000, 0x1000)]
        );
        assert_eq!(
            removed.unmapped_gpu_ranges,
            vec![(base, 0x1000), (base + 0x2000, 0x1000)]
        );
        assert_eq!(mappings.cpu_address_for(base + 0x800), None);
        assert_eq!(mappings.cpu_address_for(base + 0x1800), Some(cpu + 0x1800));
        assert_eq!(
            mappings.mapping_epoch_for(base + 0x1800),
            Some(middle_epoch)
        );
        assert_eq!(mappings.cpu_address_for(base + 0x2800), None);
        assert_eq!(mappings.iter().count(), 1);
    }

    #[test]
    fn mapping_add_matching_multi_epoch_range_falls_back_to_replacement() {
        let mut mappings = GpuMappings::new();
        let base = 0x7f66_0000;
        let cpu = 0x4f77_0000;
        assert_eq!(mappings.add(base, 0x1000, cpu, 37), GpuMappingChange::Fresh);
        let first_epoch = mappings.mapping_epoch_for(base + 0x800).unwrap();
        assert_eq!(
            mappings.add(base + 0x1000, 0x1000, cpu + 0x1000, 37),
            GpuMappingChange::Fresh
        );
        let second_epoch = mappings.mapping_epoch_for(base + 0x1800).unwrap();
        assert_ne!(first_epoch, second_epoch);

        let update = mappings.add_with_metadata(base, 0x3000, cpu, 37);
        assert_eq!(update.change, GpuMappingChange::Replaced);
        assert_eq!(update.changed_gpu_ranges, vec![(base + 0x2000, 0x1000)]);
        let replacement_epoch = mappings.mapping_epoch_for(base + 0x800).unwrap();
        assert_ne!(replacement_epoch, first_epoch);
        assert_ne!(replacement_epoch, second_epoch);
        assert_eq!(
            mappings.mapping_epoch_for(base + 0x1800),
            Some(replacement_epoch)
        );
        assert_eq!(
            mappings.mapping_epoch_for(base + 0x2800),
            Some(replacement_epoch)
        );
        assert_eq!(
            update.epoch_transitions,
            vec![
                super::GpuMappingEpochTransition {
                    gpu_va: base,
                    size: 0x1000,
                    old_epoch: first_epoch,
                    new_epoch: replacement_epoch,
                },
                super::GpuMappingEpochTransition {
                    gpu_va: base + 0x1000,
                    size: 0x1000,
                    old_epoch: second_epoch,
                    new_epoch: replacement_epoch,
                },
            ]
        );

        let removed = mappings.remove_with_metadata(base).unwrap();
        assert_eq!(removed.changed_gpu_ranges, vec![(base + 0x2000, 0x1000)]);
        assert_eq!(
            removed.epoch_transitions,
            vec![
                super::GpuMappingEpochTransition {
                    gpu_va: base,
                    size: 0x1000,
                    old_epoch: replacement_epoch,
                    new_epoch: first_epoch,
                },
                super::GpuMappingEpochTransition {
                    gpu_va: base + 0x1000,
                    size: 0x1000,
                    old_epoch: replacement_epoch,
                    new_epoch: second_epoch,
                },
            ]
        );
        assert_eq!(removed.unmapped_gpu_ranges, vec![(base + 0x2000, 0x1000)]);
        assert_eq!(mappings.mapping_epoch_for(base + 0x800), Some(first_epoch));
        assert_eq!(
            mappings.mapping_epoch_for(base + 0x1800),
            Some(second_epoch)
        );
        assert_eq!(mappings.cpu_address_for(base + 0x2800), None);
    }

    #[test]
    fn matching_multi_epoch_range_reports_only_epoch_transitions() {
        let mut mappings = GpuMappings::new();
        let base = 0x7f77_0000;
        let cpu = 0x4f88_0000;
        mappings.add(base, 0x1000, cpu, 38);
        mappings.add(base + 0x1000, 0x1000, cpu + 0x1000, 38);
        let epochs = [
            mappings.mapping_epoch_for(base + 0x800).unwrap(),
            mappings.mapping_epoch_for(base + 0x1800).unwrap(),
        ];
        let update = mappings.add_with_metadata(base, 0x2000, cpu, 38);
        let new_epoch = mappings.mapping_epoch_for(base + 0x800).unwrap();

        assert_eq!(update.change, GpuMappingChange::Replaced);
        assert!(update.changed_gpu_ranges.is_empty());
        assert_eq!(
            update.epoch_transitions,
            vec![
                super::GpuMappingEpochTransition {
                    gpu_va: base,
                    size: 0x1000,
                    old_epoch: epochs[0],
                    new_epoch,
                },
                super::GpuMappingEpochTransition {
                    gpu_va: base + 0x1000,
                    size: 0x1000,
                    old_epoch: epochs[1],
                    new_epoch,
                },
            ]
        );
        assert_eq!(mappings.mapping_epoch_for(base + 0x1800), Some(new_epoch));
    }

    #[test]
    fn mapping_add_range_overflow_fails_closed() {
        let mut mappings = GpuMappings::new();
        let change = mappings.add(u64::MAX - 0x100, 0x200, 0x1000, 41);
        assert_eq!(change, GpuMappingChange::Replaced);
        assert!(change.invalidates_render_targets());
        assert_eq!(mappings.iter().count(), 0);
        assert_eq!(
            mappings.add(0x1000, 0, 0x2000, 42),
            GpuMappingChange::Replaced
        );
        assert_eq!(mappings.iter().count(), 0);
    }

    #[test]
    fn tracked_idempotent_subrange_retains_an_unmap_root() {
        let mut mappings = GpuMappings::new();
        let base = 0x7f88_0000;
        let cpu = 0x4f99_0000;
        mappings.add_tracked_with_metadata(base, 0x3000, cpu, 51);
        let epoch = mappings.mapping_epoch_for(base + 0x1800).unwrap();

        let update = mappings.add_tracked_with_metadata(base + 0x1000, 0x1000, cpu + 0x1000, 51);

        assert_eq!(update.change, GpuMappingChange::Idempotent);
        assert!(update.changed_gpu_ranges.is_empty());
        assert!(update.epoch_transitions.is_empty());
        assert_eq!(mappings.iter().count(), 2);
        let removed = mappings.remove_with_metadata(base + 0x1000).unwrap();
        assert_eq!((removed.gpu_va, removed.size), (base + 0x1000, 0x1000));
        assert!(removed.changed_gpu_ranges.is_empty());
        assert!(removed.epoch_transitions.is_empty());
        assert_eq!(mappings.mapping_epoch_for(base + 0x1800), Some(epoch));
    }

    #[test]
    fn unmap_does_not_select_sparse_or_owned_alias_records() {
        let mut mappings = GpuMappings::new();
        let base = 0x7fa0_0000;
        let alias = base + 0x2000;
        mappings.add_sparse_with_metadata(base, 0x1000);
        assert!(mappings.remove_with_metadata(base).is_none());

        mappings.add_tracked_with_metadata(base + 0x1000, 0x1000, 0x5fa0_0000, 52);
        let owner = mappings
            .mapping_starting_at(base + 0x1000)
            .unwrap()
            .record_id;
        mappings.add_owned(1, alias, 0x1000, 0x5fa0_0000, 52, owner);
        assert!(mappings.remove_with_metadata(alias).is_none());
        assert_eq!(mappings.cpu_address_for(alias + 0x100), Some(0x5fa0_0100));
    }

    #[test]
    fn sparse_remap_holes_do_not_supersede_roots_but_unmap_barriers_do() {
        let mut mappings = GpuMappings::new();
        let fd = 1;
        let base = 0x7fa8_0000;
        mappings.add_sparse_as_gpu_with_metadata(fd, base, 0x1000);
        mappings.add_as_gpu_mapping(fd, base, 0x1000, 0x5fa8_0000, 58, None, None, true);
        assert!(mappings.remap_source_starting_at(fd, base).is_some());

        mappings.add_sparse_as_gpu_with_metadata(fd, base, 0x1000);
        assert!(mappings.unmap_as_gpu_with_metadata(fd, base).is_some());
        assert!(mappings.remap_source_starting_at(fd, base).is_none());
        assert!(mappings.unmap_as_gpu_with_metadata(fd, base).is_none());
        assert_eq!(mappings.cpu_address_for(base + 0x800), None);
    }

    #[test]
    fn allocation_scoped_teardown_masks_members_and_preserves_other_aliases() {
        let mut mappings = GpuMappings::new();
        let fd = 1;
        let allocation = 0x7fac_0000;
        mappings.add_as_gpu_mapping(fd, allocation, 0x2000, 0x5fac_0000, 59, None, None, false);
        mappings.add_as_gpu_mapping(
            fd,
            allocation + 0x1000,
            0x1000,
            0x6fac_1000,
            60,
            None,
            Some(allocation),
            true,
        );

        let removed = mappings.remove_all_for_allocation_with_metadata(fd, allocation);

        assert_eq!(
            removed.update.changed_gpu_ranges,
            vec![(allocation + 0x1000, 0x1000)]
        );
        assert_eq!(
            mappings.cpu_address_for(allocation + 0x800),
            Some(0x5fac_0800)
        );
        assert_eq!(mappings.cpu_address_for(allocation + 0x1800), None);
    }

    #[test]
    fn contained_mapping_teardown_compares_only_initial_and_final_state() {
        let mut mappings = GpuMappings::new();
        let base = 0x7fb0_0000;
        mappings.add_sparse_as_gpu_with_metadata(1, base, 0x3000);
        mappings.add_as_gpu_mapping(1, base + 0x1000, 0x1000, 0x5fb0_0000, 53, None, None, false);
        mappings.add_as_gpu_mapping(1, base + 0x1000, 0x1000, 0x6fb0_0000, 54, None, None, false);

        let removed = mappings
            .remove_all_contained_with_metadata(1, base, 0x3000)
            .unwrap();

        assert_eq!(removed.update.changed_gpu_ranges, vec![(base, 0x3000)]);
        assert!(removed.update.epoch_transitions.is_empty());
        assert!(removed.owned_va_ranges.is_empty());
        assert_eq!(mappings.iter().count(), 0);
    }

    #[test]
    fn contained_mapping_teardown_splits_crossing_aliases() {
        let mut mappings = GpuMappings::new();
        let base = 0x7fc0_0000;
        mappings.add_as_gpu_mapping(1, base - 0x1000, 0x3000, 0x5fc0_0000, 55, None, None, false);

        let removed = mappings
            .remove_all_contained_with_metadata(1, base, 0x1000)
            .unwrap();

        assert_eq!(removed.update.changed_gpu_ranges, vec![(base, 0x1000)]);
        assert_eq!(mappings.cpu_address_for(base - 0x800), Some(0x5fc0_0800));
        assert_eq!(mappings.cpu_address_for(base + 0x800), None);
        assert_eq!(mappings.cpu_address_for(base + 0x1800), Some(0x5fc0_2800));
        assert_eq!(mappings.iter().count(), 2);
    }

    #[test]
    fn any32_lookup_splits_mappings_that_wrap_the_low_address_domain() {
        let mut mappings = GpuMappings::new();
        let base = 0xffff_f000;
        let cpu = 0x6fd0_0000;
        mappings.add(base, 0x3000, cpu, 56);

        assert_eq!(
            mappings.cpu_address_for_any32(0xffff_f800),
            Some((base, cpu + 0x800, 0x800))
        );
        assert_eq!(
            mappings.cpu_address_for_any32(0x800),
            Some((base, cpu + 0x1800, 0x1800))
        );

        mappings.add(0x1000, 0x800, 0x7fd0_0000, 57);
        assert_eq!(
            mappings.cpu_address_for_any32(0x800),
            Some((base, cpu + 0x1800, 0x800))
        );
    }

    #[test]
    fn fixed_va_reservations_reject_cross_domain_and_overflow_ranges() {
        let gpu = GpuContext::new();

        assert!(!gpu.alloc_va_fixed(0x0400_0800, 0));
        assert!(!gpu.free_va(0x0400_0800, 0));
        assert!(!gpu.alloc_va_fixed(super::BIG_VA_BASE - 0x1000, 0x2000));
        assert!(!gpu.alloc_va_fixed(u64::MAX - 0x800, 0x1000));
        assert!(!gpu.free_va(super::BIG_VA_BASE - 0x1000, 0x2000));
        assert_eq!(gpu.alloc_va(0x1000, false), 0x0400_0000);
    }

    #[test]
    fn mapping_mutations_invalidate_cached_intervals() {
        let mut mappings = GpuMappings::new();
        mappings.add(0x1000, 0x1000, 0x1_0000, 1);

        assert_eq!(mappings.cpu_address_for(0x1500), Some(0x1_0500));
        let generation_before = mappings.generation;

        mappings.add(0x1400, 0x200, 0x2_0000, 2);
        assert_ne!(mappings.generation, generation_before);
        assert_eq!(mappings.cpu_address_for(0x1500), Some(0x2_0100));

        let generation_before = mappings.generation;
        assert_eq!(mappings.remove(0x1400), Some(0x200));
        assert_ne!(mappings.generation, generation_before);
        assert_eq!(mappings.cpu_address_for(0x1500), Some(0x1_0500));
        assert_eq!(mappings.nvmap_id_for(0x1500), Some(1));
    }

    #[test]
    fn mapping_mutations_bump_texture_generations_across_the_full_range() {
        let mut mappings = GpuMappings::new();
        let base = 0x6f21_8000;
        let size = 0x20000;
        let generations_before = [
            nexium_gpu::tex_invalidate::region_gen(base),
            nexium_gpu::tex_invalidate::region_gen(base + 0x10000),
            nexium_gpu::tex_invalidate::region_gen(base + 0x20000),
            nexium_gpu::tex_invalidate::region_gen(base + 0x30000),
        ];

        mappings.add(base, size, 0x4f00_0000, 7);

        for (index, before) in generations_before[..3].iter().enumerate() {
            assert_ne!(
                nexium_gpu::tex_invalidate::region_gen(base + index as u64 * 0x10000),
                *before
            );
        }
        assert_eq!(
            nexium_gpu::tex_invalidate::region_gen(base + 0x30000),
            generations_before[3]
        );

        let generations_before_remove = [
            nexium_gpu::tex_invalidate::region_gen(base),
            nexium_gpu::tex_invalidate::region_gen(base + 0x10000),
            nexium_gpu::tex_invalidate::region_gen(base + 0x20000),
            nexium_gpu::tex_invalidate::region_gen(base + 0x30000),
        ];
        assert_eq!(mappings.remove(base), Some(size));

        for (index, before) in generations_before_remove[..3].iter().enumerate() {
            assert_ne!(
                nexium_gpu::tex_invalidate::region_gen(base + index as u64 * 0x10000),
                *before
            );
        }
        assert_eq!(
            nexium_gpu::tex_invalidate::region_gen(base + 0x30000),
            generations_before_remove[3]
        );
    }
}

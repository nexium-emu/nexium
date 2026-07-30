pub mod engines;
pub mod flat_allocator;
mod formats;
pub mod pusher;
pub mod vk_dispatch;

pub use engines::{
    Fermi2D, KeplerCompute, KeplerMemory, Maxwell3D, Maxwell3DRegisters, MaxwellDma,
};
pub use pusher::{CommandListHeader, Pusher};

use parking_lot::Mutex;
use std::cell::Cell;
use std::collections::HashMap;
use std::sync::Arc;

fn nvprof_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("NEXIUM_NVDRV_PROFILE").is_ok())
}

fn elapsed_ms(start: std::time::Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

pub struct GpuMapping {
    pub gpu_va: u64,
    pub size: u64,
    pub cpu_addr: u64,
    pub nvmap_id: u32,
}

const MAPPING_LOOKUP_CACHE_SIZE: usize = 8;

#[derive(Clone, Copy, Debug, Default)]
struct MappingLookupCacheEntry {
    gpu_lo: u64,
    gpu_hi: u64,
    mapping_index: usize,
}

pub struct GpuMappings {
    mappings: Vec<GpuMapping>,
    lookup_cache: [Cell<Option<MappingLookupCacheEntry>>; MAPPING_LOOKUP_CACHE_SIZE],
    lookup_cache_cursor: Cell<usize>,
}

impl GpuMappings {
    pub fn new() -> Self {
        Self {
            mappings: Vec::new(),
            lookup_cache: std::array::from_fn(|_| Cell::new(None)),
            lookup_cache_cursor: Cell::new(0),
        }
    }

    #[inline]
    fn clear_lookup_cache(&self) {
        for entry in &self.lookup_cache {
            entry.set(None);
        }
        self.lookup_cache_cursor.set(0);
    }

    #[inline]
    fn contains(mapping: &GpuMapping, gpu_va: u64) -> bool {
        gpu_va >= mapping.gpu_va && gpu_va < mapping.gpu_va.saturating_add(mapping.size)
    }

    #[inline]
    fn mapping_index_for(&self, gpu_va: u64) -> Option<usize> {
        for cached in &self.lookup_cache {
            let Some(cached) = cached.get() else {
                continue;
            };
            if gpu_va >= cached.gpu_lo && gpu_va < cached.gpu_hi {
                debug_assert!(
                    self.mappings
                        .get(cached.mapping_index)
                        .is_some_and(|mapping| Self::contains(mapping, gpu_va)),
                    "stale GMMU lookup cache entry"
                );
                return Some(cached.mapping_index);
            }
        }

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
        let slot = self.lookup_cache_cursor.get() % MAPPING_LOOKUP_CACHE_SIZE;
        self.lookup_cache[slot].set(Some(MappingLookupCacheEntry {
            gpu_lo,
            gpu_hi,
            mapping_index,
        }));
        self.lookup_cache_cursor
            .set((slot + 1) % MAPPING_LOOKUP_CACHE_SIZE);
        Some(mapping_index)
    }

    pub fn add(&mut self, gpu_va: u64, size: u64, cpu_addr: u64, nvmap_id: u32) {
        log::debug!(
            "GpuMap: gpu_va={:#x} size={:#x} cpu_addr={:#x} nvmap_id={}",
            gpu_va,
            size,
            cpu_addr,
            nvmap_id
        );
        self.mappings.push(GpuMapping {
            gpu_va,
            size,
            cpu_addr,
            nvmap_id,
        });
        self.clear_lookup_cache();
    }

    pub fn cpu_address_for_any32(&self, gpu_va: u64) -> Option<(u64, u64, u64)> {
        let lo = gpu_va & 0xFFFF_FFFF;
        let mut best: Option<&GpuMapping> = None;
        for m in &self.mappings {
            let mlo = m.gpu_va & 0xFFFF_FFFF;
            if lo >= mlo && lo < mlo + m.size {
                let better = match best {
                    Some(b) => (m.gpu_va & 0xFFFF_FFFF) > (b.gpu_va & 0xFFFF_FFFF),
                    None => true,
                };
                if better {
                    best = Some(m);
                }
            }
        }
        best.map(|m| {
            let off = lo - (m.gpu_va & 0xFFFF_FFFF);
            (m.gpu_va, m.cpu_addr + off, m.size - off)
        })
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

    pub fn remove(&mut self, gpu_va: u64) -> Option<u64> {
        if let Some(pos) = self.mappings.iter().rposition(|m| m.gpu_va == gpu_va) {
            let size = self.mappings.remove(pos).size;
            self.clear_lookup_cache();
            Some(size)
        } else {
            None
        }
    }

    #[inline]
    pub fn cpu_address_for(&self, gpu_va: u64) -> Option<u64> {
        let mapping = &self.mappings[self.mapping_index_for(gpu_va)?];
        Some(mapping.cpu_addr + (gpu_va - mapping.gpu_va))
    }

    #[inline]
    pub fn mapping_at(&self, gpu_va: u64) -> Option<(u64, u64, u64)> {
        let mapping = &self.mappings[self.mapping_index_for(gpu_va)?];
        Some((mapping.gpu_va, mapping.size, mapping.cpu_addr))
    }

    #[inline]
    pub fn cpu_range_for(&self, gpu_va: u64) -> Option<(u64, u64)> {
        let mapping_index = self.mapping_index_for(gpu_va)?;
        let mapping = &self.mappings[mapping_index];
        let offset = gpu_va - mapping.gpu_va;
        let mapping_end = mapping.gpu_va.saturating_add(mapping.size);
        let contiguous_end = self.mappings[mapping_index + 1..]
            .iter()
            .filter_map(|newer| {
                (newer.gpu_va > gpu_va && newer.gpu_va < mapping_end).then_some(newer.gpu_va)
            })
            .min()
            .unwrap_or(mapping_end);
        Some((mapping.cpu_addr + offset, contiguous_end - gpu_va))
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

    pub fn gpu_regions_for_cpu_range(&self, cpu_addr: u64, size: u64) -> Vec<(u64, u64)> {
        if size == 0 {
            return Vec::new();
        }
        let cpu_end = cpu_addr.saturating_add(size);
        let mut regions = self
            .mappings
            .iter()
            .filter_map(|mapping| {
                let mapping_end = mapping.cpu_addr.saturating_add(mapping.size);
                let overlap_start = cpu_addr.max(mapping.cpu_addr);
                let overlap_end = cpu_end.min(mapping_end);
                (overlap_start < overlap_end).then(|| {
                    (
                        mapping.gpu_va + (overlap_start - mapping.cpu_addr),
                        overlap_end - overlap_start,
                    )
                })
            })
            .collect::<Vec<_>>();
        regions.sort_unstable();
        regions.dedup();
        regions
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
        Some(self.mappings[self.mapping_index_for(gpu_va)?].nvmap_id)
    }
}

impl Default for GpuMappings {
    fn default() -> Self {
        Self::new()
    }
}

pub struct GpuContext {
    pub mappings: Arc<Mutex<GpuMappings>>,
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
}

const BIG_VA_BASE: u64 = 0x4_0000_0000;

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
        Self {
            mappings: Arc::new(Mutex::new(GpuMappings::new())),
            maxwell3d: Arc::new(Mutex::new(Maxwell3D::new())),
            maxwell_dma: Arc::new(Mutex::new(MaxwellDma::new())),
            fermi_2d: Arc::new(Mutex::new(Fermi2D::new())),
            kepler_compute: Arc::new(Mutex::new(KeplerCompute::new())),
            kepler_memory: Arc::new(Mutex::new(KeplerMemory::new())),
            pusher: Arc::new(Mutex::new(Pusher::new())),
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
        }
    }

    pub fn alloc_gpu_va(&self, size: u64) -> u64 {
        self.alloc_va(size, false)
    }

    pub fn alloc_gpu_va_aligned(&self, size: u64, align: u64) -> u64 {
        self.alloc_va(size, align >= 0x10000)
    }

    pub fn alloc_va(&self, size: u64, big: bool) -> u64 {
        let (alloc, page) = if big {
            (&self.big_alloc, 0x10000u64)
        } else {
            (&self.small_alloc, 0x1000u64)
        };
        let padded = (size + (page - 1)) & !(page - 1);
        alloc.lock().allocate(padded)
    }

    pub fn alloc_va_fixed(&self, gpu_va: u64, size: u64) {
        let (alloc, page) = if gpu_va >= BIG_VA_BASE {
            (&self.big_alloc, 0x10000u64)
        } else {
            (&self.small_alloc, 0x1000u64)
        };
        let base = gpu_va & !(page - 1);
        let padded = ((gpu_va - base) + size + (page - 1)) & !(page - 1);
        alloc.lock().allocate_fixed(base, padded);
    }

    pub fn free_va(&self, gpu_va: u64, size: u64) {
        let (alloc, page) = if gpu_va >= BIG_VA_BASE {
            (&self.big_alloc, 0x10000u64)
        } else {
            (&self.small_alloc, 0x1000u64)
        };
        let base = gpu_va & !(page - 1);
        let padded = ((gpu_va - base) + size + (page - 1)) & !(page - 1);
        alloc.lock().free(base, padded);
    }

    pub fn submit_gpfifo(
        &self,
        address: u64,
        num_entries: u32,
        mem_read: impl Fn(u64, &mut [u8]) -> bool,
        mem_write: impl Fn(u64, &[u8]) -> bool,
        mem_copy: impl Fn(u64, u64, usize) -> bool,
    ) -> (u32, u32) {
        let kp_total = pusher::kickprof::start();
        let kp_locks = pusher::kickprof::start();
        let mut pusher = self.pusher.lock();
        let mut maxwell = self.maxwell3d.lock();
        let mut maxwell_dma = self.maxwell_dma.lock();
        let mut fermi_2d = self.fermi_2d.lock();
        let mut kepler_compute = self.kepler_compute.lock();
        let mut kepler_memory = self.kepler_memory.lock();
        let mappings = self.mappings.lock();
        pusher::kickprof::add(pusher::kickprof::LOCKS, kp_locks);

        pusher.process_gpfifo(
            address,
            num_entries,
            &mappings,
            &mut *maxwell,
            &mut *maxwell_dma,
            &mut *fermi_2d,
            &mut *kepler_compute,
            &mut *kepler_memory,
            &*self.stats,
            &mem_read,
            &mem_write,
            &mem_copy,
        );
        pusher.syncpt_value = pusher.syncpt_value.wrapping_add(2);
        pusher::kickprof::kick_done(kp_total);

        let syncpt_id = 0u32;
        let syncpt_value = pusher.syncpt_value;
        (syncpt_id, syncpt_value)
    }

    pub fn process_inline_gpfifo(
        &self,
        entries: &[CommandListHeader],
        mem_read: impl Fn(u64, &mut [u8]) -> bool,
        mem_write: impl Fn(u64, &[u8]) -> bool,
        mem_copy: impl Fn(u64, u64, usize) -> bool,
    ) -> (u32, u32) {
        let profile = nvprof_enabled();
        let kp_total = pusher::kickprof::start();
        let kp_locks = pusher::kickprof::start();
        let t0 = std::time::Instant::now();
        let mut pusher = self.pusher.lock();
        let mut maxwell = self.maxwell3d.lock();
        let mut maxwell_dma = self.maxwell_dma.lock();
        let mut fermi_2d = self.fermi_2d.lock();
        let mut kepler_compute = self.kepler_compute.lock();
        let mut kepler_memory = self.kepler_memory.lock();
        let mappings = self.mappings.lock();
        pusher::kickprof::add(pusher::kickprof::LOCKS, kp_locks);
        let locks_ms = if profile { elapsed_ms(t0) } else { 0.0 };

        pusher.begin_ssbo_snapshot_epoch();

        let t_entries = std::time::Instant::now();
        let addrs: Vec<u64> = entries.iter().map(|e| e.address()).collect();
        for (i, entry) in entries.iter().enumerate() {
            pusher.entry_word_limit = if entry.entry_count() > 4096 {
                crate::gpu::pusher::nearest_forward_gap(&addrs, i)
            } else {
                0
            };
            pusher.process_entry(
                entry,
                &mappings,
                &mut *maxwell,
                &mut *maxwell_dma,
                &mut *fermi_2d,
                &mut *kepler_compute,
                &mut *kepler_memory,
                &*self.stats,
                &mem_read,
                &mem_write,
                &mem_copy,
            );
        }
        pusher.entry_word_limit = 0;
        let entries_ms = if profile { elapsed_ms(t_entries) } else { 0.0 };
        let t_flush = std::time::Instant::now();
        pusher.resolve_pending_compute(&mappings, &mem_write);
        pusher.flush_vk(&mappings, &mem_read, &mem_write);
        if let Some(r) = pusher.renderer.clone() {
            let kp_wb = pusher::kickprof::start();
            vk_dispatch::writeback_small_rts(&r, &mappings, &mem_write);
            pusher::kickprof::add(pusher::kickprof::SMALLRT, kp_wb);
        }
        vk_dispatch::guest_probe(&mappings, &mem_read);
        pusher.end_ssbo_snapshot_epoch();
        let flush_ms = if profile { elapsed_ms(t_flush) } else { 0.0 };
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

    pub fn read_rt(
        &self,
        mem_read: impl Fn(u64, &mut [u8]) -> bool,
    ) -> Option<(u32, u32, Vec<u8>)> {
        let mappings = self.mappings.lock();
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
    use super::GpuMappings;

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
    fn mapping_lookup_cache_respects_newest_overlapping_mapping() {
        let mut mappings = GpuMappings::new();
        mappings.add(0x1000, 0x2000, 0x1_0000, 1);
        mappings.add(0x1400, 0x800, 0x2_0000, 2);
        mappings.add(0x1800, 0x200, 0x3_0000, 3);

        assert_eq!(mappings.cpu_address_for(0x1200), Some(0x1_0200));
        let cursor_after_miss = mappings.lookup_cache_cursor.get();

        assert_eq!(mappings.nvmap_id_for(0x1300), Some(1));
        assert_eq!(mappings.lookup_cache_cursor.get(), cursor_after_miss);

        assert_eq!(mappings.cpu_address_for(0x1500), Some(0x2_0100));
        assert_eq!(mappings.nvmap_id_for(0x1900), Some(3));
        assert_eq!(mappings.mapping_at(0x1b00), Some((0x1400, 0x800, 0x2_0000)));
        assert_eq!(mappings.cpu_range_for(0x1200), Some((0x1_0200, 0x200)));
        assert_eq!(mappings.cpu_range_for(0x1500), Some((0x2_0100, 0x300)));
        assert_eq!(mappings.cpu_range_for(0x1900), Some((0x3_0100, 0x100)));
        assert_eq!(mappings.cpu_range_for(0x1b00), Some((0x2_0700, 0x100)));
        assert_eq!(mappings.cpu_range_for(0x1d00), Some((0x1_0d00, 0x1300)));
    }

    #[test]
    fn mapping_mutations_invalidate_cached_intervals() {
        let mut mappings = GpuMappings::new();
        mappings.add(0x1000, 0x1000, 0x1_0000, 1);

        assert_eq!(mappings.cpu_address_for(0x1500), Some(0x1_0500));
        assert!(mappings
            .lookup_cache
            .iter()
            .any(|entry| entry.get().is_some()));

        mappings.add(0x1400, 0x200, 0x2_0000, 2);
        assert!(mappings
            .lookup_cache
            .iter()
            .all(|entry| entry.get().is_none()));
        assert_eq!(mappings.cpu_address_for(0x1500), Some(0x2_0100));

        assert_eq!(mappings.remove(0x1400), Some(0x200));
        assert!(mappings
            .lookup_cache
            .iter()
            .all(|entry| entry.get().is_none()));
        assert_eq!(mappings.cpu_address_for(0x1500), Some(0x1_0500));
        assert_eq!(mappings.nvmap_id_for(0x1500), Some(1));
    }
}

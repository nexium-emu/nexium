pub(crate) mod completion;
pub mod engines;
pub mod flat_allocator;
mod formats;
pub(crate) mod prep;
pub mod pusher;
pub mod vk_dispatch;

pub use engines::{
    Fermi2D, KeplerCompute, KeplerMemory, Maxwell3D, Maxwell3DRegisters, MaxwellDma,
};
pub use pusher::{CommandListHeader, Pusher};

use parking_lot::{Mutex, RwLock};
use std::collections::HashMap;
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

pub struct GpuMapping {
    pub gpu_va: u64,
    pub size: u64,
    pub cpu_addr: u64,
    pub nvmap_id: u32,
    epoch: u64,
}

const MAPPING_LOOKUP_CACHE_SIZE: usize = 64;

#[derive(Clone, Copy, Debug, Default)]
struct MappingLookupCacheEntry {
    gpu_lo: u64,
    gpu_hi: u64,
    mapping_index: usize,
}

struct ThreadMappingLookupCache {
    instance_id: u64,
    generation: u64,
    entries: [Option<MappingLookupCacheEntry>; MAPPING_LOOKUP_CACHE_SIZE],
    cursor: usize,
}

impl ThreadMappingLookupCache {
    const fn empty() -> Self {
        Self {
            instance_id: 0,
            generation: 0,
            entries: [None; MAPPING_LOOKUP_CACHE_SIZE],
            cursor: 0,
        }
    }
}

thread_local! {
    static MAPPING_LOOKUP_CACHE: std::cell::RefCell<ThreadMappingLookupCache> =
        const { std::cell::RefCell::new(ThreadMappingLookupCache::empty()) };
}

pub struct GpuMappings {
    mappings: Vec<GpuMapping>,
    instance_id: u64,
    next_mapping_epoch: u64,
    generation: u64,
}

impl GpuMappings {
    pub fn new() -> Self {
        static NEXT_INSTANCE_ID: std::sync::atomic::AtomicU64 =
            std::sync::atomic::AtomicU64::new(1);
        Self {
            mappings: Vec::new(),
            instance_id: NEXT_INSTANCE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            next_mapping_epoch: 1,
            generation: 1,
        }
    }

    #[inline]
    fn contains(mapping: &GpuMapping, gpu_va: u64) -> bool {
        gpu_va >= mapping.gpu_va && gpu_va < mapping.gpu_va.saturating_add(mapping.size)
    }

    fn mapping_lookup_slow(&self, gpu_va: u64) -> Option<MappingLookupCacheEntry> {
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
                cache.cursor = 0;
            }
            for cached in cache.entries.iter().flatten() {
                if gpu_va >= cached.gpu_lo && gpu_va < cached.gpu_hi {
                    debug_assert!(
                        self.mappings
                            .get(cached.mapping_index)
                            .is_some_and(|mapping| Self::contains(mapping, gpu_va)),
                        "stale GMMU lookup cache entry"
                    );
                    return Some(*cached);
                }
            }
            let cached = self.mapping_lookup_slow(gpu_va)?;
            let slot = cache.cursor % MAPPING_LOOKUP_CACHE_SIZE;
            cache.entries[slot] = Some(cached);
            cache.cursor = (slot + 1) % MAPPING_LOOKUP_CACHE_SIZE;
            Some(cached)
        })
    }

    #[inline]
    fn mapping_index_for(&self, gpu_va: u64) -> Option<usize> {
        Some(self.mapping_lookup_for(gpu_va)?.mapping_index)
    }

    pub fn add(&mut self, gpu_va: u64, size: u64, cpu_addr: u64, nvmap_id: u32) {
        log::debug!(
            "GpuMap: gpu_va={:#x} size={:#x} cpu_addr={:#x} nvmap_id={}",
            gpu_va,
            size,
            cpu_addr,
            nvmap_id
        );
        let epoch = self.next_mapping_epoch;
        self.next_mapping_epoch = self.next_mapping_epoch.wrapping_add(1).max(1);
        self.generation = self.generation.wrapping_add(1).max(1);
        self.mappings.push(GpuMapping {
            gpu_va,
            size,
            cpu_addr,
            nvmap_id,
            epoch,
        });
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
            self.generation = self.generation.wrapping_add(1).max(1);
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
        let cached = self.mapping_lookup_for(gpu_va)?;
        let mapping = &self.mappings[cached.mapping_index];
        let offset = gpu_va - mapping.gpu_va;
        Some((mapping.cpu_addr + offset, cached.gpu_hi - gpu_va))
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

    #[inline]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    #[inline]
    pub fn mapping_epoch_for(&self, gpu_va: u64) -> Option<u64> {
        Some(self.mappings[self.mapping_index_for(gpu_va)?].epoch)
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
        Some((cpu_addr, writer(cpu_addr, bytes)))
    }
}

pub(crate) static PENDING_ENGINE_SYNCPT_INCRS: parking_lot::Mutex<Vec<u32>> =
    parking_lot::Mutex::new(Vec::new());

pub(crate) fn record_engine_syncpt_increment(id: u32) {
    PENDING_ENGINE_SYNCPT_INCRS.lock().push(id);
    nexium_common::host_wake::signal();
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
            guest_memory,
            decoder_stub_engines: Mutex::new(StubEngines {
                maxwell_dma: MaxwellDma::new(),
                fermi_2d: Fermi2D::new(),
                kepler_compute: KeplerCompute::new(),
                kepler_memory: KeplerMemory::new(),
            }),
        }
    }

    pub(crate) fn install_prep_thread(&self, resources: prep::PrepThreadResources) {
        let mut pusher = self.pusher.lock();
        let previous = std::mem::replace(
            &mut pusher.prep,
            prep::PrepLane::Inline(prep::PrepState::new()),
        );
        let prep::PrepLane::Inline(state) = previous else {
            pusher.prep = previous;
            return;
        };
        let handle = prep::spawn_prep_thread(state, resources);
        pusher.prep = prep::PrepLane::Threaded(handle);
    }

    pub(crate) fn prep_present(
        &self,
        job: crate::render_thread::RenderJob,
        flush_small_rts: bool,
    ) -> Result<(), crate::render_thread::RenderJob> {
        let mut pusher = self.pusher.lock();
        match &mut pusher.prep {
            prep::PrepLane::Threaded(handle) => {
                match handle.send_recover(prep::PrepEvent::Present {
                    job,
                    flush_small_rts,
                }) {
                    Ok(()) => Ok(()),
                    Err(prep::PrepEvent::Present { job, .. }) => Err(job),
                    Err(_) => unreachable!("prep present returned a different event"),
                }
            }
            prep::PrepLane::Inline(_) => Err(job),
        }
    }

    pub(crate) fn prep_drain_barrier(
        &self,
        done: crossbeam::channel::Sender<()>,
        flush_small_rts: bool,
    ) -> bool {
        let mut pusher = self.pusher.lock();
        match &mut pusher.prep {
            prep::PrepLane::Threaded(handle) => {
                handle.send(prep::PrepEvent::DrainBarrier {
                    done,
                    flush_small_rts,
                });
                true
            }
            prep::PrepLane::Inline(_) => false,
        }
    }

    pub fn set_guest_memory_writer(&self, writer: GuestMemoryWriter) {
        self.guest_memory.set_writer(Some(writer));
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
        let mut pusher = self.pusher.lock();
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
        let embedded_incrs = std::mem::take(&mut pusher.pending_syncpt_incrs);
        self.apply_embedded_syncpt_incrs(embedded_incrs);
        pusher.syncpt_value = pusher.syncpt_value.wrapping_add(2);
        pusher::kickprof::kick_done(kp_total);

        let syncpt_id = 0u32;
        let syncpt_value = pusher.syncpt_value;
        (syncpt_id, syncpt_value)
    }

    fn apply_embedded_syncpt_incrs(&self, incrs: Vec<(u32, u32)>) {
        if incrs.is_empty() {
            return;
        }
        let mut channels = self.channels.lock();
        for (id, count) in incrs {
            if let Some(channel) = channels.values_mut().find(|c| c.syncpt_id == id) {
                channel.syncpt_min = channel.syncpt_min.wrapping_add(count);
                if crate::syncpoint_reached(channel.syncpt_max, channel.syncpt_min) {
                    channel.syncpt_max = channel.syncpt_min;
                }
            }
        }
        drop(channels);
        nexium_common::host_wake::signal();
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
        let mut pusher = self.pusher.lock();
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
        let entries = pusher::repair_endform_entries(entries);
        let entries = entries.as_slice();
        let _addrs: Vec<u64> = entries.iter().map(|e| e.address()).collect();
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
        for entry in entries.iter() {
            pusher.entry_word_limit = 0;
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
        pusher.entry_word_limit = 0;
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
        let embedded_incrs = std::mem::take(&mut pusher.pending_syncpt_incrs);
        self.apply_embedded_syncpt_incrs(embedded_incrs);
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
        let mut pusher = self.pusher.lock();
        let mappings = self.mappings.read();
        let Some(state) = pusher.prep.inline_state() else {
            return false;
        };
        let Some(renderer) = state.renderer.clone() else {
            return false;
        };
        state.writeback_small_rts(&renderer, &mappings, &mem_write)
    }

    pub(crate) fn flush_prepared_draw_packets(&self) -> bool {
        match self.pusher.lock().prep.inline_state() {
            Some(state) => state.flush_prepared_draw_packets(),
            None => false,
        }
    }

    pub(crate) fn has_prepared_draw_packets(&self) -> bool {
        match self.pusher.lock().prep.inline_state() {
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
        eager_small_rt_writeback_value_enabled, GpuContext, GpuMappings, GuestMemoryAccess,
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
        let cursor_after_miss = super::MAPPING_LOOKUP_CACHE.with(|cache| cache.borrow().cursor);

        assert_eq!(mappings.nvmap_id_for(0x1300), Some(1));
        assert_eq!(
            super::MAPPING_LOOKUP_CACHE.with(|cache| cache.borrow().cursor),
            cursor_after_miss
        );

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
}

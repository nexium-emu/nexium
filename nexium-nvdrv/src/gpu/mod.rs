pub mod engines;
pub mod flat_allocator;
pub mod pusher;
pub mod vk_dispatch;

pub use engines::{
    Fermi2D, KeplerCompute, KeplerMemory, Maxwell3D, Maxwell3DRegisters, MaxwellDma,
};
pub use pusher::{CommandListHeader, Pusher};

use parking_lot::Mutex;
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

pub struct GpuMappings {
    mappings: Vec<GpuMapping>,
}

impl GpuMappings {
    pub fn new() -> Self {
        Self {
            mappings: Vec::new(),
        }
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
            Some(self.mappings.remove(pos).size)
        } else {
            None
        }
    }

    pub fn cpu_address_for(&self, gpu_va: u64) -> Option<u64> {
        for m in self.mappings.iter().rev() {
            if gpu_va >= m.gpu_va && gpu_va < m.gpu_va + m.size {
                let offset = gpu_va - m.gpu_va;
                return Some(m.cpu_addr + offset);
            }
        }
        None
    }

    pub fn cpu_range_for(&self, gpu_va: u64) -> Option<(u64, u64)> {
        for m in self.mappings.iter().rev() {
            if gpu_va >= m.gpu_va && gpu_va < m.gpu_va + m.size {
                let offset = gpu_va - m.gpu_va;
                return Some((m.cpu_addr + offset, m.size - offset));
            }
        }
        None
    }

    pub fn iter(&self) -> impl Iterator<Item = &GpuMapping> {
        self.mappings.iter()
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

    pub fn nvmap_id_for(&self, gpu_va: u64) -> Option<u32> {
        for m in self.mappings.iter().rev() {
            if gpu_va >= m.gpu_va && gpu_va < m.gpu_va + m.size {
                return Some(m.nvmap_id);
            }
        }
        None
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
    pub syncpt_value: u32,
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
    ) -> (u32, u32) {
        let mut pusher = self.pusher.lock();
        let mut maxwell = self.maxwell3d.lock();
        let mut maxwell_dma = self.maxwell_dma.lock();
        let mut fermi_2d = self.fermi_2d.lock();
        let mut kepler_compute = self.kepler_compute.lock();
        let mut kepler_memory = self.kepler_memory.lock();
        let mappings = self.mappings.lock();

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
        );
        pusher.syncpt_value = pusher.syncpt_value.wrapping_add(2);

        let syncpt_id = 0u32;
        let syncpt_value = pusher.syncpt_value;
        (syncpt_id, syncpt_value)
    }

    pub fn process_inline_gpfifo(
        &self,
        entries: &[CommandListHeader],
        mem_read: impl Fn(u64, &mut [u8]) -> bool,
        mem_write: impl Fn(u64, &[u8]) -> bool,
    ) -> (u32, u32) {
        let profile = nvprof_enabled();
        let t0 = std::time::Instant::now();
        let mut pusher = self.pusher.lock();
        let mut maxwell = self.maxwell3d.lock();
        let mut maxwell_dma = self.maxwell_dma.lock();
        let mut fermi_2d = self.fermi_2d.lock();
        let mut kepler_compute = self.kepler_compute.lock();
        let mut kepler_memory = self.kepler_memory.lock();
        let mappings = self.mappings.lock();
        let locks_ms = if profile { elapsed_ms(t0) } else { 0.0 };

        let t_entries = std::time::Instant::now();
        for entry in entries {
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
            );
        }
        let entries_ms = if profile { elapsed_ms(t_entries) } else { 0.0 };
        let t_flush = std::time::Instant::now();
        pusher.flush_vk(&mappings, &mem_read);
        if let Some(r) = pusher.renderer.clone() {
            vk_dispatch::writeback_small_rts(&r, &mappings, &mem_write);
        }
        let flush_ms = if profile { elapsed_ms(t_flush) } else { 0.0 };
        pusher.syncpt_value = pusher.syncpt_value.wrapping_add(2);
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

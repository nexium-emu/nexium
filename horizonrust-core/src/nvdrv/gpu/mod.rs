pub mod pusher;
pub mod engines;

pub use pusher::{CommandListHeader, Pusher};
pub use engines::{Maxwell3D, Maxwell3DRegisters};

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

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
        Self { mappings: Vec::new() }
    }

    pub fn add(&mut self, gpu_va: u64, size: u64, cpu_addr: u64, nvmap_id: u32) {
        log::debug!("GpuMap: gpu_va={:#x} size={:#x} cpu_addr={:#x} nvmap_id={}",
            gpu_va, size, cpu_addr, nvmap_id);
        self.mappings.push(GpuMapping { gpu_va, size, cpu_addr, nvmap_id });
    }

    pub fn cpu_address_for(&self, gpu_va: u64) -> Option<u64> {
        for m in &self.mappings {
            if gpu_va >= m.gpu_va && gpu_va < m.gpu_va + m.size {
                let offset = gpu_va - m.gpu_va;
                return Some(m.cpu_addr + offset);
            }
        }
        None
    }

    pub fn iter(&self) -> impl Iterator<Item = &GpuMapping> {
        self.mappings.iter()
    }

    pub fn nvmap_id_for(&self, gpu_va: u64) -> Option<u32> {
        for m in &self.mappings {
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
    pub pusher: Arc<Mutex<Pusher>>,
    pub next_gpu_va: Arc<Mutex<u64>>,
    pub channels: Arc<Mutex<HashMap<u32, ChannelState>>>,
}

#[derive(Default)]
pub struct ChannelState {
    pub bound_engine: u32,
    pub bound_obj_class: u32,
    pub syncpt_id: u32,
    pub syncpt_value: u32,
}

impl GpuContext {
    pub fn new() -> Self {
        Self {
            mappings: Arc::new(Mutex::new(GpuMappings::new())),
            maxwell3d: Arc::new(Mutex::new(Maxwell3D::new())),
            pusher: Arc::new(Mutex::new(Pusher::new())),
            next_gpu_va: Arc::new(Mutex::new(0x1_0000_0000u64)),
            channels: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn alloc_gpu_va(&self, size: u64) -> u64 {
        let mut next = self.next_gpu_va.lock();
        let aligned_size = (size + 0xFFF) & !0xFFF;
        let va = *next;
        *next += aligned_size;
        va
    }

    pub fn submit_gpfifo(&self, address: u64, num_entries: u32, mem_read: impl Fn(u64, &mut [u8]) -> bool) -> (u32, u32) {
        let mut pusher = self.pusher.lock();
        let mut maxwell = self.maxwell3d.lock();
        let mappings = self.mappings.lock();

        pusher.process_gpfifo(address, num_entries, &mappings, &mut *maxwell, &mem_read);
        pusher.syncpt_value = pusher.syncpt_value.wrapping_add(1);

        let syncpt_id = 0u32;
        let syncpt_value = pusher.syncpt_value;
        (syncpt_id, syncpt_value)
    }

    pub fn process_inline_gpfifo(&self, entries: &[CommandListHeader], mem_read: impl Fn(u64, &mut [u8]) -> bool) -> (u32, u32) {
        let mut pusher = self.pusher.lock();
        let mut maxwell = self.maxwell3d.lock();
        let mappings = self.mappings.lock();

        for entry in entries {
            pusher.process_entry(entry, &mappings, &mut *maxwell, &mem_read);
        }
        pusher.syncpt_value = pusher.syncpt_value.wrapping_add(1);
        (0, pusher.syncpt_value)
    }

    pub fn read_rt(&self, mem_read: impl Fn(u64, &mut [u8]) -> bool) -> Option<(u32, u32, Vec<u8>)> {
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

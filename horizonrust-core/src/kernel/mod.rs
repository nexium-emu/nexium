pub mod svc;
pub mod svc_defs;
pub mod threads;
pub mod handles;
pub mod hid;
pub mod session;
pub mod cpu_context;

use crate::memory::AddressSpace;
use crate::nvdrv::Nvdrv;
use crate::services::Services;
use crate::services::FrameOut;
use crate::cpu::Cpu;
use std::sync::Arc;
use std::collections::HashMap;
use parking_lot::Mutex;

pub struct Kernel {
    pub address_space: Arc<AddressSpace>,
    pub handles: handles::HandleTable,
    pub threads: threads::Threads,
    pub services: Services,
    pub nvdrv: Nvdrv,
    pub hid: Arc<Mutex<hid::HidShared>>,
    pub sessions: HashMap<u32, session::Session>,
    pub event_signals: HashMap<u32, bool>,
    pub tls_buffer: [u8; 0x100],
    pub cpu: Option<Cpu>,
    pub pending_frames: Vec<FrameOut>,

    pub code_base: u64,
    pub code_size: u64,
    pub heap_base: u64,
    pub heap_size: u64,
    pub stack_base: u64,
    pub stack_size: u64,

    pub cycle_count: u64,
    pub next_vsync_cycle: u64,
    pub display_ready: bool,
    pub process_exited: bool,
}

impl Kernel {
    pub fn new(
        address_space: Arc<AddressSpace>,
        code_base: u64,
        code_size: u64,
        heap_base: u64,
        heap_size: u64,
        stack_base: u64,
        stack_size: u64,
    ) -> Self {
        Self {
            address_space,
            handles: handles::HandleTable::new(),
            threads: threads::Threads::new(),
            services: Services::new(),
            nvdrv: Nvdrv::new(),
            hid: Arc::new(Mutex::new(hid::HidShared::new())),
            sessions: HashMap::new(),
            event_signals: HashMap::new(),
            tls_buffer: [0u8; 0x100],
            cpu: None,
            pending_frames: Vec::new(),
            code_base,
            code_size,
            heap_base,
            heap_size,
            stack_base,
            stack_size,
            cycle_count: 0,
            next_vsync_cycle: 16_666_667,
            display_ready: false,
            process_exited: false,
        }
    }

    pub fn init_cpu(&mut self) -> Result<(), String> {
        let mut cpu = Cpu::new_dynarmic()?;
        for region in self.address_space.host_regions() {
            unsafe {
                cpu.map_host(region.base, region.size, region.perm, region.host_ptr)
                    .map_err(|e| format!("CPU map_host failed for {:#x}: {}", region.base, e))?;
            }
        }
        log::info!("Kernel CPU initialized with {} mapped regions", self.address_space.host_regions().len());
        self.cpu = Some(cpu);
        Ok(())
    }

    pub fn drain_frames(&mut self) -> Vec<FrameOut> {
        std::mem::take(&mut self.pending_frames)
    }

    pub fn dispatch_svc(&mut self, imm: u16) -> u32 {
        svc::dispatch(self, imm)
    }
}

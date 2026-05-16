pub mod svc;
pub mod svc_defs;
pub mod threads;
pub mod handles;
pub mod hid;
pub mod session;

use crate::memory::AddressSpace;
use crate::nvdrv::Nvdrv;
use crate::services::Services;
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

    pub code_base: u64,
    pub code_size: u64,
    pub heap_base: u64,
    pub heap_size: u64,
    pub stack_base: u64,
    pub stack_size: u64,
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
            code_base,
            code_size,
            heap_base,
            heap_size,
            stack_base,
            stack_size,
        }
    }

    pub fn dispatch_svc(&mut self, imm: u16) -> u32 {
        svc::dispatch(self, imm)
    }
}

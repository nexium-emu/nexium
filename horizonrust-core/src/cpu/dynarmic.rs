use std::sync::Arc;
use parking_lot::Mutex;

pub struct DynarmicCpu {
    x_regs: [u64; 31],
    pc: u64,
    sp: u64,
    tpidrro_el0: u64,
    memory: Arc<Mutex<Vec<u8>>>,
    pending_svc: Option<u16>,
}

impl DynarmicCpu {
    pub fn new(memory_ptr: *mut u8, memory_size: usize) -> Result<Self, String> {
        log::debug!("DynarmicCpu initialized with {} bytes of memory", memory_size);

        let memory = unsafe {
            let slice = std::slice::from_raw_parts_mut(memory_ptr, memory_size);
            Arc::new(Mutex::new(slice.to_vec()))
        };

        Ok(Self {
            x_regs: [0; 31],
            pc: 0,
            sp: 0,
            tpidrro_el0: 0,
            memory,
            pending_svc: None,
        })
    }

    pub fn inject_svc(&mut self, imm: u16) {
        self.pending_svc = Some(imm);
    }

    pub fn set_register(&mut self, reg: u32, val: u64) {
        if reg < 31 {
            self.x_regs[reg as usize] = val;
        } else if reg == 31 {
            self.sp = val;
        }
    }

    pub fn get_register(&self, reg: u32) -> u64 {
        if reg < 31 {
            self.x_regs[reg as usize]
        } else if reg == 31 {
            self.sp
        } else {
            0
        }
    }

    pub fn set_pc(&mut self, pc: u64) {
        self.pc = pc;
    }

    pub fn get_pc(&self) -> u64 {
        self.pc
    }

    pub fn set_sp(&mut self, sp: u64) {
        self.sp = sp;
    }

    pub fn get_sp(&self) -> u64 {
        self.sp
    }

    pub fn set_tpidrro_el0(&mut self, val: u64) {
        self.tpidrro_el0 = val;
    }

    pub fn get_tpidrro_el0(&self) -> u64 {
        self.tpidrro_el0
    }

    pub fn run(&mut self, cycle_count: u64) -> CpuEvent {
        if let Some(svc) = self.pending_svc.take() {
            return CpuEvent::Svc(svc);
        }

        let initial_pc = self.pc;
        log::debug!("CPU running {} cycles from PC {:#x}", cycle_count, initial_pc);

        self.pc = initial_pc.wrapping_add(4);

        if self.pc == initial_pc {
            log::debug!("CPU stalled at PC {:#x}", self.pc);
            CpuEvent::Stalled
        } else {
            CpuEvent::Running
        }
    }

    pub fn step(&mut self) -> CpuEvent {
        self.run(1)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum CpuEvent {
    Running,
    Stalled,
    Interrupted,
    Svc(u16),
    Exception(u32),
}

impl Default for DynarmicCpu {
    fn default() -> Self {
        Self {
            x_regs: [0; 31],
            pc: 0,
            sp: 0,
            tpidrro_el0: 0,
            memory: Arc::new(Mutex::new(vec![0u8; 0x10000])),
            pending_svc: None,
        }
    }
}

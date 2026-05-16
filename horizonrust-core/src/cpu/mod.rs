pub mod dynarmic;

use dynarmic::DynarmicCpu;

pub enum Cpu {
    Dynarmic(DynarmicCpu),
}

impl Cpu {
    pub fn new_dynarmic(memory_ptr: *mut u8, memory_size: usize) -> Result<Self, String> {
        DynarmicCpu::new(memory_ptr, memory_size).map(Cpu::Dynarmic)
    }

    pub fn from_env(memory_ptr: *mut u8, memory_size: usize) -> Result<Self, String> {
        let backend = std::env::var("HORIZONRUST_CPU").unwrap_or_else(|_| "dynarmic".to_string());

        match backend.as_str() {
            "dynarmic" => Self::new_dynarmic(memory_ptr, memory_size),
            _ => {
                log::warn!("Unknown CPU backend: {}, defaulting to dynarmic", backend);
                Self::new_dynarmic(memory_ptr, memory_size)
            }
        }
    }

    pub fn set_register(&mut self, reg: u32, val: u64) {
        match self {
            Cpu::Dynarmic(cpu) => cpu.set_register(reg, val),
        }
    }

    pub fn get_register(&self, reg: u32) -> u64 {
        match self {
            Cpu::Dynarmic(cpu) => cpu.get_register(reg),
        }
    }

    pub fn set_pc(&mut self, pc: u64) {
        match self {
            Cpu::Dynarmic(cpu) => cpu.set_pc(pc),
        }
    }

    pub fn get_pc(&self) -> u64 {
        match self {
            Cpu::Dynarmic(cpu) => cpu.get_pc(),
        }
    }

    pub fn set_sp(&mut self, sp: u64) {
        match self {
            Cpu::Dynarmic(cpu) => cpu.set_sp(sp),
        }
    }

    pub fn get_sp(&self) -> u64 {
        match self {
            Cpu::Dynarmic(cpu) => cpu.get_sp(),
        }
    }

    pub fn set_tpidrro_el0(&mut self, val: u64) {
        match self {
            Cpu::Dynarmic(cpu) => cpu.set_tpidrro_el0(val),
        }
    }

    pub fn get_tpidrro_el0(&self) -> u64 {
        match self {
            Cpu::Dynarmic(cpu) => cpu.get_tpidrro_el0(),
        }
    }

    pub fn run(&mut self, cycle_count: u64) -> CpuEvent {
        match self {
            Cpu::Dynarmic(cpu) => cpu.run(cycle_count),
        }
    }

    pub fn step(&mut self) -> CpuEvent {
        match self {
            Cpu::Dynarmic(cpu) => cpu.step(),
        }
    }

    pub fn inject_svc(&mut self, imm: u16) {
        match self {
            Cpu::Dynarmic(cpu) => cpu.inject_svc(imm),
        }
    }
}

pub use dynarmic::CpuEvent;

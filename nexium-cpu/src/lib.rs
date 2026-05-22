pub mod dynarmic;

pub use dynarmic::{HaltHandle, FaultSnapshot};
use dynarmic::DynarmicCpu;
use nexium_memory::Perm;

pub enum Cpu {
    Dynarmic(DynarmicCpu),
}

impl Cpu {
    pub fn new_dynarmic() -> Result<Self, String> {
        DynarmicCpu::new().map(Cpu::Dynarmic)
    }

    pub fn halt_handle(&self) -> HaltHandle {
        match self {
            Cpu::Dynarmic(cpu) => cpu.halt_handle(),
        }
    }

    pub unsafe fn map_host(&mut self, va: u64, len: u64, perm: Perm, ptr: *mut u8) -> Result<(), String> {
        match self {
            Cpu::Dynarmic(cpu) => cpu.map_host(va, len, perm, ptr),
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

    pub fn write_bytes(&self, va: u64, bytes: &[u8]) -> Result<(), String> {
        match self {
            Cpu::Dynarmic(cpu) => cpu.write_bytes(va, bytes),
        }
    }

    pub fn read_bytes(&self, va: u64, buf: &mut [u8]) -> Result<(), String> {
        match self {
            Cpu::Dynarmic(cpu) => cpu.read_bytes(va, buf),
        }
    }

    pub fn take_fault(&self) -> Option<FaultSnapshot> {
        match self {
            Cpu::Dynarmic(cpu) => cpu.take_fault(),
        }
    }

    pub fn set_continue_on_null(&self, enable: bool) {
        match self {
            Cpu::Dynarmic(cpu) => cpu.set_continue_on_null(enable),
        }
    }

    pub fn null_skip_count(&self) -> u32 {
        match self {
            Cpu::Dynarmic(cpu) => cpu.null_skip_count(),
        }
    }
}

pub use dynarmic::CpuEvent;

#[cfg(feature = "backend-dynarmic")]
pub mod dynarmic;
#[cfg(feature = "backend-rustarmic")]
pub mod rustarmic;

#[cfg(feature = "backend-dynarmic")]
use dynarmic::DynarmicCpu;
#[cfg(feature = "backend-rustarmic")]
use rustarmic::RustarmicCpu;

use nexium_memory::Perm;

#[derive(Clone)]
pub struct HaltHandle {
    pub(crate) inner: std::sync::Arc<dyn Fn() + Send + Sync>,
    pub(crate) peek:  std::sync::Arc<dyn Fn() -> (u64, u64, u64) + Send + Sync>,
}

impl HaltHandle {
    pub fn halt(&self) { (self.inner)(); }
    pub fn peek_pc_lr_sp(&self) -> (u64, u64, u64) { (self.peek)() }
}

#[derive(Clone, Debug, Default)]
pub struct FaultSnapshot {
    pub pc: u64,
    pub lr: u64,
    pub sp: u64,
    pub addr: u64,
    pub size: u32,
    pub is_write: bool,
    pub value: u64,
    pub regs: [u64; 31],
}

#[derive(Debug, Clone, Copy)]
pub enum CpuEvent {
    Running,
    Stalled,
    Interrupted,
    Svc(u16),
    Exception(u32),
}

pub enum Cpu {
    #[cfg(feature = "backend-dynarmic")]
    Dynarmic(DynarmicCpu),
    #[cfg(feature = "backend-rustarmic")]
    Rustarmic(RustarmicCpu),
}

macro_rules! dispatch {
    ($self:ident, $cpu:ident => $call:expr) => {
        match $self {
            #[cfg(feature = "backend-dynarmic")]
            Cpu::Dynarmic($cpu) => $call,
            #[cfg(feature = "backend-rustarmic")]
            Cpu::Rustarmic($cpu) => $call,
        }
    };
}

impl Cpu {
    #[cfg(feature = "backend-dynarmic")]
    pub fn new_dynarmic() -> Result<Self, String> {
        DynarmicCpu::new().map(Cpu::Dynarmic)
    }

    #[cfg(feature = "backend-rustarmic")]
    pub fn new_rustarmic() -> Result<Self, String> {
        RustarmicCpu::new().map(Cpu::Rustarmic)
    }

    pub fn halt_handle(&self) -> HaltHandle {
        dispatch!(self, cpu => cpu.halt_handle())
    }

    pub unsafe fn map_host(&mut self, va: u64, len: u64, perm: Perm, ptr: *mut u8) -> Result<(), String> {
        dispatch!(self, cpu => unsafe { cpu.map_host(va, len, perm, ptr) })
    }

    pub fn set_register(&mut self, reg: u32, val: u64) {
        dispatch!(self, cpu => cpu.set_register(reg, val))
    }
    pub fn get_register(&self, reg: u32) -> u64 {
        dispatch!(self, cpu => cpu.get_register(reg))
    }

    pub fn set_pc(&mut self, pc: u64) { dispatch!(self, cpu => cpu.set_pc(pc)) }
    pub fn get_pc(&self) -> u64       { dispatch!(self, cpu => cpu.get_pc()) }
    pub fn set_sp(&mut self, sp: u64) { dispatch!(self, cpu => cpu.set_sp(sp)) }
    pub fn get_sp(&self) -> u64       { dispatch!(self, cpu => cpu.get_sp()) }

    pub fn set_tpidrro_el0(&mut self, val: u64) { dispatch!(self, cpu => cpu.set_tpidrro_el0(val)) }
    pub fn get_tpidrro_el0(&self) -> u64        { dispatch!(self, cpu => cpu.get_tpidrro_el0()) }

    pub fn run(&mut self, cycle_count: u64) -> CpuEvent { dispatch!(self, cpu => cpu.run(cycle_count)) }
    pub fn step(&mut self) -> CpuEvent                  { dispatch!(self, cpu => cpu.step()) }
    pub fn inject_svc(&mut self, imm: u16)              { dispatch!(self, cpu => cpu.inject_svc(imm)) }

    pub fn write_bytes(&self, va: u64, bytes: &[u8]) -> Result<(), String> {
        dispatch!(self, cpu => cpu.write_bytes(va, bytes))
    }
    pub fn read_bytes(&self, va: u64, buf: &mut [u8]) -> Result<(), String> {
        dispatch!(self, cpu => cpu.read_bytes(va, buf))
    }

    pub fn take_fault(&self) -> Option<FaultSnapshot> { dispatch!(self, cpu => cpu.take_fault()) }
    pub fn set_continue_on_null(&self, enable: bool)  { dispatch!(self, cpu => cpu.set_continue_on_null(enable)) }
    pub fn null_skip_count(&self) -> u32              { dispatch!(self, cpu => cpu.null_skip_count()) }
}

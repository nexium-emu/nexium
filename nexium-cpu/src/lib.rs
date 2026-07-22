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
    pub(crate) peek: std::sync::Arc<dyn Fn() -> (u64, u64, u64) + Send + Sync>,
    pub(crate) peek_dump: std::sync::Arc<dyn Fn() -> String + Send + Sync>,
}

impl HaltHandle {
    pub fn halt(&self) {
        (self.inner)();
    }
    pub fn peek_pc_lr_sp(&self) -> (u64, u64, u64) {
        (self.peek)()
    }
    pub fn peek_dump(&self) -> String {
        (self.peek_dump)()
    }
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

#[derive(Clone)]
pub enum CpuThreadContext {
    #[cfg(feature = "backend-dynarmic")]
    Dynarmic(dynarmic_sys::DynarmicContext),
    #[cfg(feature = "backend-rustarmic")]
    Rustarmic(rustarmic::RustarmicThreadContext),
}

impl std::fmt::Debug for CpuThreadContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            #[cfg(feature = "backend-dynarmic")]
            Self::Dynarmic(_) => f.write_str("Dynarmic"),
            #[cfg(feature = "backend-rustarmic")]
            Self::Rustarmic(_) => f.write_str("Rustarmic"),
        }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CpuBackendKind {
    Dynarmic,
    Rustarmic,
}

impl Default for CpuBackendKind {
    fn default() -> Self {
        #[cfg(feature = "backend-dynarmic")]
        {
            return CpuBackendKind::Dynarmic;
        }
        #[cfg(all(not(feature = "backend-dynarmic"), feature = "backend-rustarmic"))]
        {
            return CpuBackendKind::Rustarmic;
        }
        #[cfg(not(any(feature = "backend-dynarmic", feature = "backend-rustarmic")))]
        {
            CpuBackendKind::Dynarmic
        }
    }
}

impl CpuBackendKind {
    pub fn label(&self) -> &'static str {
        match self {
            CpuBackendKind::Dynarmic => "Dynarmic (C++)",
            CpuBackendKind::Rustarmic => "Rustarmic (Rust JIT)",
        }
    }

    pub fn available() -> Vec<CpuBackendKind> {
        let mut v = Vec::new();
        #[cfg(feature = "backend-dynarmic")]
        v.push(CpuBackendKind::Dynarmic);
        #[cfg(feature = "backend-rustarmic")]
        v.push(CpuBackendKind::Rustarmic);
        v
    }

    pub fn is_compiled_in(&self) -> bool {
        match self {
            CpuBackendKind::Dynarmic => cfg!(feature = "backend-dynarmic"),
            CpuBackendKind::Rustarmic => cfg!(feature = "backend-rustarmic"),
        }
    }
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

    pub fn new(backend: CpuBackendKind) -> Result<Self, String> {
        match backend {
            CpuBackendKind::Dynarmic => {
                #[cfg(feature = "backend-dynarmic")]
                {
                    return Self::new_dynarmic();
                }
                #[cfg(not(feature = "backend-dynarmic"))]
                {
                    return Err("Dynarmic backend not compiled in (rebuild with --features backend-dynarmic)".into());
                }
            }
            CpuBackendKind::Rustarmic => {
                #[cfg(feature = "backend-rustarmic")]
                {
                    return Self::new_rustarmic();
                }
                #[cfg(not(feature = "backend-rustarmic"))]
                {
                    return Err("Rustarmic backend not compiled in (rebuild with --features backend-rustarmic)".into());
                }
            }
        }
    }

    pub fn halt_handle(&self) -> HaltHandle {
        dispatch!(self, cpu => cpu.halt_handle())
    }

    pub unsafe fn map_host(
        &mut self,
        va: u64,
        len: u64,
        perm: Perm,
        ptr: *mut u8,
    ) -> Result<(), String> {
        dispatch!(self, cpu => unsafe { cpu.map_host(va, len, perm, ptr) })
    }

    pub unsafe fn unmap_host(&mut self, va: u64, len: u64) -> Result<(), String> {
        dispatch!(self, cpu => unsafe { cpu.unmap_host(va, len) })
    }

    pub fn set_register(&mut self, reg: u32, val: u64) {
        dispatch!(self, cpu => cpu.set_register(reg, val))
    }
    pub fn get_register(&self, reg: u32) -> u64 {
        dispatch!(self, cpu => cpu.get_register(reg))
    }

    pub fn set_pc(&mut self, pc: u64) {
        dispatch!(self, cpu => cpu.set_pc(pc))
    }
    pub fn get_pc(&self) -> u64 {
        dispatch!(self, cpu => cpu.get_pc())
    }
    pub fn set_sp(&mut self, sp: u64) {
        dispatch!(self, cpu => cpu.set_sp(sp))
    }
    pub fn get_sp(&self) -> u64 {
        dispatch!(self, cpu => cpu.get_sp())
    }

    pub fn set_tpidrro_el0(&mut self, val: u64) {
        dispatch!(self, cpu => cpu.set_tpidrro_el0(val))
    }
    pub fn get_tpidrro_el0(&self) -> u64 {
        dispatch!(self, cpu => cpu.get_tpidrro_el0())
    }

    pub fn save_thread_context(
        &self,
        context: &mut Option<CpuThreadContext>,
    ) -> Result<(), String> {
        match self {
            #[cfg(feature = "backend-dynarmic")]
            Cpu::Dynarmic(cpu) => {
                if !matches!(context.as_ref(), Some(CpuThreadContext::Dynarmic(_))) {
                    *context = Some(CpuThreadContext::Dynarmic(cpu.alloc_thread_context()));
                }
                let Some(CpuThreadContext::Dynarmic(context)) = context.as_mut() else {
                    unreachable!()
                };
                cpu.save_thread_context(context)
            }
            #[cfg(feature = "backend-rustarmic")]
            Cpu::Rustarmic(cpu) => {
                *context = Some(CpuThreadContext::Rustarmic(cpu.save_thread_context()));
                Ok(())
            }
        }
    }

    pub fn restore_thread_context(&mut self, context: &CpuThreadContext) -> Result<(), String> {
        match (self, context) {
            #[cfg(feature = "backend-dynarmic")]
            (Cpu::Dynarmic(cpu), CpuThreadContext::Dynarmic(context)) => {
                cpu.restore_thread_context(context)
            }
            #[cfg(feature = "backend-rustarmic")]
            (Cpu::Rustarmic(cpu), CpuThreadContext::Rustarmic(context)) => {
                cpu.restore_thread_context(context);
                Ok(())
            }
            #[allow(unreachable_patterns)]
            _ => Err("CPU thread context backend mismatch".to_string()),
        }
    }

    pub fn reset_thread_context(&mut self) -> Result<(), String> {
        match self {
            #[cfg(feature = "backend-dynarmic")]
            Cpu::Dynarmic(cpu) => cpu.reset_thread_context(),
            #[cfg(feature = "backend-rustarmic")]
            Cpu::Rustarmic(cpu) => {
                cpu.reset_thread_context();
                Ok(())
            }
        }
    }

    pub fn run(&mut self, cycle_count: u64) -> CpuEvent {
        dispatch!(self, cpu => cpu.run(cycle_count))
    }
    pub fn step(&mut self) -> CpuEvent {
        dispatch!(self, cpu => cpu.step())
    }
    pub fn inject_svc(&mut self, imm: u16) {
        dispatch!(self, cpu => cpu.inject_svc(imm))
    }

    pub fn write_bytes(&self, va: u64, bytes: &[u8]) -> Result<(), String> {
        dispatch!(self, cpu => cpu.write_bytes(va, bytes))
    }
    pub fn read_bytes(&self, va: u64, buf: &mut [u8]) -> Result<(), String> {
        dispatch!(self, cpu => cpu.read_bytes(va, buf))
    }

    pub fn take_fault(&self) -> Option<FaultSnapshot> {
        dispatch!(self, cpu => cpu.take_fault())
    }
    pub fn set_continue_on_null(&self, enable: bool) {
        dispatch!(self, cpu => cpu.set_continue_on_null(enable))
    }
    pub fn null_skip_count(&self) -> u32 {
        dispatch!(self, cpu => cpu.null_skip_count())
    }

    pub fn invalidate_range(&mut self, va: u64, len: u64) {
        dispatch!(self, cpu => cpu.invalidate_range(va, len))
    }
}

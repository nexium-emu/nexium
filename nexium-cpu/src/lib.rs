#[cfg(feature = "backend-dynarmic")]
pub mod dynarmic;
#[cfg(feature = "backend-rustarmic")]
pub mod rustarmic;
#[cfg(nce_runtime)]
pub mod nce;
pub mod nce_layout;
pub mod nce_patch;
pub mod system;

pub use system::{CpuCore, CpuSystem, CpuSystemConfig};

#[cfg(feature = "backend-dynarmic")]
use dynarmic::DynarmicCpu;
#[cfg(feature = "backend-rustarmic")]
use rustarmic::RustarmicCpu;
#[cfg(nce_runtime)]
use nce::NceCpu;

pub fn nce_runtime_available() -> bool {
    #[cfg(nce_runtime)]
    {
        return nce::supported();
    }
    #[cfg(not(nce_runtime))]
    {
        false
    }
}

pub fn nce_register_post_handlers(entries: &[(u64, u64)]) {
    #[cfg(nce_runtime)]
    {
        nce::register_post_handlers(entries);
    }
    #[cfg(not(nce_runtime))]
    {
        let _ = entries;
    }
}

pub fn nce_host_counter_hz() -> u64 {
    #[cfg(target_arch = "aarch64")]
    {
        let value: u64;
        unsafe {
            std::arch::asm!("mrs {0}, cntfrq_el0", out(reg) value, options(nomem, nostack, preserves_flags));
        }
        return value;
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        nce_layout::GUEST_CNTFRQ_HZ
    }
}

use nexium_memory::Perm;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CpuError {
    Backend(String),
    Stalled,
}

impl std::fmt::Display for CpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Backend(message) => write!(f, "CPU backend error: {message}"),
            Self::Stalled => f.write_str("CPU backend stalled"),
        }
    }
}

impl std::error::Error for CpuError {}

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
    #[cfg(nce_runtime)]
    Nce(nce::NceThreadContext),
}

impl std::fmt::Debug for CpuThreadContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            #[cfg(feature = "backend-dynarmic")]
            Self::Dynarmic(_) => f.write_str("Dynarmic"),
            #[cfg(feature = "backend-rustarmic")]
            Self::Rustarmic(_) => f.write_str("Rustarmic"),
            #[cfg(nce_runtime)]
            Self::Nce(_) => f.write_str("Nce"),
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

#[derive(Debug, Clone, Copy)]
pub struct CpuRunResult {
    pub event: CpuEvent,
    pub retired: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CpuBackendKind {
    Dynarmic,
    Rustarmic,
    Nce,
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
            CpuBackendKind::Nce => "NCE (native execution)",
        }
    }

    pub fn available() -> Vec<CpuBackendKind> {
        let mut v = Vec::new();
        #[cfg(feature = "backend-dynarmic")]
        v.push(CpuBackendKind::Dynarmic);
        #[cfg(feature = "backend-rustarmic")]
        v.push(CpuBackendKind::Rustarmic);
        #[cfg(nce_runtime)]
        v.push(CpuBackendKind::Nce);
        v
    }

    pub fn is_compiled_in(&self) -> bool {
        match self {
            CpuBackendKind::Dynarmic => cfg!(feature = "backend-dynarmic"),
            CpuBackendKind::Rustarmic => cfg!(feature = "backend-rustarmic"),
            CpuBackendKind::Nce => cfg!(nce_runtime),
        }
    }
}

pub enum Cpu {
    #[cfg(feature = "backend-dynarmic")]
    Dynarmic(DynarmicCpu),
    #[cfg(feature = "backend-rustarmic")]
    Rustarmic(RustarmicCpu),
    #[cfg(nce_runtime)]
    Nce(NceCpu),
}

macro_rules! dispatch {
    ($self:ident, $cpu:ident => $call:expr) => {
        match $self {
            #[cfg(feature = "backend-dynarmic")]
            Cpu::Dynarmic($cpu) => $call,
            #[cfg(feature = "backend-rustarmic")]
            Cpu::Rustarmic($cpu) => $call,
            #[cfg(nce_runtime)]
            Cpu::Nce($cpu) => $call,
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

    #[cfg(feature = "backend-rustarmic")]
    pub fn new_rustarmic_with_config(config: &::rustarmic::EngineConfig) -> Result<Self, String> {
        RustarmicCpu::new_with_engine_config(config).map(Cpu::Rustarmic)
    }

    #[cfg(nce_runtime)]
    pub fn new_nce() -> Result<Self, String> {
        NceCpu::new().map(Cpu::Nce)
    }

    pub fn new(backend: CpuBackendKind) -> Result<Self, String> {
        match backend {
            CpuBackendKind::Nce => {
                #[cfg(nce_runtime)]
                {
                    return Self::new_nce();
                }
                #[cfg(not(nce_runtime))]
                {
                    return Err("NCE backend is only available on aarch64 Android/Linux builds with --features backend-nce".into());
                }
            }
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

    pub fn set_core_id(&mut self, core_id: u64) {
        match self {
            #[cfg(feature = "backend-dynarmic")]
            Cpu::Dynarmic(_) => {
                let _ = core_id;
            }
            #[cfg(feature = "backend-rustarmic")]
            Cpu::Rustarmic(cpu) => cpu.set_core_id(core_id),
            #[cfg(nce_runtime)]
            Cpu::Nce(cpu) => cpu.set_core_id(core_id),
        }
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
            #[cfg(nce_runtime)]
            Cpu::Nce(cpu) => {
                *context = Some(CpuThreadContext::Nce(cpu.save_thread_context()));
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
            #[cfg(nce_runtime)]
            (Cpu::Nce(cpu), CpuThreadContext::Nce(context)) => {
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
            #[cfg(nce_runtime)]
            Cpu::Nce(cpu) => {
                cpu.reset_thread_context();
                Ok(())
            }
        }
    }

    pub fn run_with_count(&mut self, cycle_count: u64) -> CpuRunResult {
        match self {
            #[cfg(feature = "backend-dynarmic")]
            Cpu::Dynarmic(cpu) => {
                let (event, retired) = cpu.run_with_count(cycle_count);
                CpuRunResult { event, retired }
            }
            #[cfg(feature = "backend-rustarmic")]
            Cpu::Rustarmic(cpu) => {
                let (event, retired) = cpu.run_with_count(cycle_count);
                CpuRunResult { event, retired }
            }
            #[cfg(nce_runtime)]
            Cpu::Nce(cpu) => {
                let (event, retired) = cpu.run_with_count(cycle_count);
                CpuRunResult { event, retired }
            }
        }
    }

    pub fn run(&mut self, cycle_count: u64) -> Result<CpuRunResult, CpuError> {
        let result = self.run_with_count(cycle_count);
        if matches!(result.event, CpuEvent::Stalled) {
            Err(CpuError::Stalled)
        } else {
            Ok(result)
        }
    }

    pub fn run_event(&mut self, cycle_count: u64) -> CpuEvent {
        self.run(cycle_count)
            .map(|result| result.event)
            .unwrap_or(CpuEvent::Stalled)
    }

    pub fn step(&mut self) -> CpuEvent {
        self.run_event(1)
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

    pub fn nce_stats(&self) -> Option<String> {
        match self {
            #[cfg(nce_runtime)]
            Cpu::Nce(cpu) => Some(cpu.stats()),
            #[allow(unreachable_patterns)]
            _ => None,
        }
    }
}

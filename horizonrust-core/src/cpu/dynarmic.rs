use std::sync::{Arc, Mutex};
use crate::memory::Perm;

pub(crate) struct SharedDynarmic {
    pub(crate) emu: dynarmic_sys::Dynarmic<'static, ()>,
}
unsafe impl Send for SharedDynarmic {}
unsafe impl Sync for SharedDynarmic {}

pub struct DynarmicCpu {
    emu: Arc<SharedDynarmic>,
    last_event: Arc<Mutex<Option<CpuEvent>>>,
}

unsafe impl Send for DynarmicCpu {}
unsafe impl Sync for DynarmicCpu {}

impl DynarmicCpu {
    pub fn new() -> Result<Self, String> {
        let emu: dynarmic_sys::Dynarmic<'static, ()> =
            dynarmic_sys::Dynarmic::new();

        let last_event = Arc::new(Mutex::new(None::<CpuEvent>));

        let event_for_svc = last_event.clone();
        emu.set_svc_callback(move |dyn_, swi, _until, pc| {
            *event_for_svc.lock().unwrap() = Some(CpuEvent::Svc(swi as u16));
            let _ = dyn_.emu_stop();
            let _ = pc;
        });

        let event_for_unmapped = last_event.clone();
        emu.set_unmapped_mem_callback(move |dyn_, addr, size, _value| {
            log::warn!("dynarmic: unmapped memory {:#x} size={}", addr, size);
            *event_for_unmapped.lock().unwrap() = Some(CpuEvent::Exception(0x0E));
            let _ = dyn_.emu_stop();
            true
        });

        Ok(Self {
            emu: Arc::new(SharedDynarmic { emu }),
            last_event,
        })
    }

    pub unsafe fn map_host(&mut self, va: u64, len: u64, perm: Perm, ptr: *mut u8) -> Result<(), String> {
        self.emu.emu.mem_map_ptr(va, len as usize, perm_to_dyn(perm), ptr.cast())
            .map_err(|e| format!("map_host failed: {:?}", e))
    }

    pub fn write_bytes(&self, va: u64, bytes: &[u8]) -> Result<(), String> {
        self.emu.emu.mem_write(va, bytes)
            .map_err(|e| format!("write_bytes failed: {:?}", e))
    }

    pub fn read_bytes(&self, va: u64, buf: &mut [u8]) -> Result<(), String> {
        self.emu.emu.mem_read(va, buf)
            .map_err(|e| format!("read_bytes failed: {:?}", e))
    }

    pub fn set_register(&mut self, reg: u32, val: u64) {
        if reg < 31 {
            let _ = self.emu.emu.reg_write_raw(reg as usize, val);
        } else if reg == 31 {
            let _ = self.emu.emu.reg_write_sp(val);
        }
    }

    pub fn get_register(&self, reg: u32) -> u64 {
        if reg < 31 {
            self.emu.emu.reg_read(reg as usize).unwrap_or(0)
        } else if reg == 31 {
            self.emu.emu.reg_read_sp().unwrap_or(0)
        } else {
            0
        }
    }

    pub fn set_pc(&mut self, pc: u64) {
        let _ = self.emu.emu.reg_write_pc(pc);
    }

    pub fn get_pc(&self) -> u64 {
        self.emu.emu.reg_read_pc().unwrap_or(0)
    }

    pub fn set_sp(&mut self, sp: u64) {
        let _ = self.emu.emu.reg_write_sp(sp);
    }

    pub fn get_sp(&self) -> u64 {
        self.emu.emu.reg_read_sp().unwrap_or(0)
    }

    pub fn set_tpidrro_el0(&mut self, val: u64) {
        let _ = self.emu.emu.reg_write_tpidrr0_el0(val);
    }

    pub fn get_tpidrro_el0(&self) -> u64 {
        self.emu.emu.reg_read_tpidrr0_el0().unwrap_or(0)
    }

    pub fn run(&mut self, _max_insn: u64) -> CpuEvent {
        *self.last_event.lock().unwrap() = None;
        let pc = self.get_pc();
        let _ = self.emu.emu.emu_start(pc, u64::MAX - 16);
        let event = self.last_event.lock().unwrap().take();
        match event {
            Some(CpuEvent::Svc(imm)) => CpuEvent::Svc(imm),
            Some(other) => other,
            None => CpuEvent::Running,
        }
    }

    pub fn step(&mut self) -> CpuEvent {
        self.run(1)
    }

    pub fn inject_svc(&mut self, imm: u16) {
        *self.last_event.lock().unwrap() = Some(CpuEvent::Svc(imm));
    }
}

fn perm_to_dyn(p: Perm) -> u32 {
    let mut out = 0u32;
    if p.contains(Perm::R) { out |= 1; }
    if p.contains(Perm::W) { out |= 2; }
    if p.contains(Perm::X) { out |= 4; }
    out
}

#[derive(Debug, Clone, Copy)]
pub enum CpuEvent {
    Running,
    Stalled,
    Interrupted,
    Svc(u16),
    Exception(u32),
}

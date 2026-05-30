use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use nexium_memory::Perm;

use crate::{CpuEvent, FaultSnapshot, HaltHandle};

pub(crate) struct SharedDynarmic {
    pub(crate) emu: dynarmic_sys::Dynarmic<'static, ()>,
}
unsafe impl Send for SharedDynarmic {}
unsafe impl Sync for SharedDynarmic {}

const NULL_SKIP_MAX: u32 = 64;

pub struct DynarmicCpu {
    emu: Arc<SharedDynarmic>,
    last_event: Arc<Mutex<Option<CpuEvent>>>,
    last_fault: Arc<Mutex<Option<FaultSnapshot>>>,
    continue_on_null: Arc<AtomicBool>,
    null_skip_count: Arc<AtomicU32>,
}

unsafe impl Send for DynarmicCpu {}
unsafe impl Sync for DynarmicCpu {}

impl DynarmicCpu {
    pub fn new() -> Result<Self, String> {
        let emu: dynarmic_sys::Dynarmic<'static, ()> =
            dynarmic_sys::Dynarmic::new();

        let last_event = Arc::new(Mutex::new(None::<CpuEvent>));
        let last_fault: Arc<Mutex<Option<FaultSnapshot>>> = Arc::new(Mutex::new(None));
        let continue_on_null = Arc::new(AtomicBool::new(false));
        let null_skip_count = Arc::new(AtomicU32::new(0));

        let event_for_svc = last_event.clone();
        emu.set_svc_callback(move |dyn_, swi, _until, pc| {
            log::trace!("dynarmic SVC callback triggered: swi={:#04x}, pc={:#x}", swi, pc);
            *event_for_svc.lock().unwrap() = Some(CpuEvent::Svc(swi as u16));
            let _ = dyn_.emu_stop();
        });
        log::info!("dynarmic: SVC callback registered");

        let event_for_unmapped = last_event.clone();
        let fault_for_unmapped = last_fault.clone();
        let continue_flag = continue_on_null.clone();
        let skip_counter = null_skip_count.clone();
        emu.set_unmapped_mem_callback(move |dyn_, addr, size, value| {
            let pc = dyn_.reg_read_pc().unwrap_or(0);
            let lr = dyn_.reg_read_lr().unwrap_or(0);
            let sp = dyn_.reg_read_sp().unwrap_or(0);
            let mut regs = [0u64; 31];
            for i in 0..31 {
                regs[i] = dyn_.reg_read(i).unwrap_or(0);
            }
            let is_null_zone = addr < 0x1000;
            if is_null_zone {
                log::error!(
                    "[null-deref] addr={:#x} size={} val={:#x} pc={:#x} lr={:#x} sp={:#x} x0={:#x} x1={:#x} x2={:#x} x3={:#x} x4={:#x} x5={:#x}",
                    addr, size, value, pc, lr, sp, regs[0], regs[1], regs[2], regs[3], regs[4], regs[5]
                );
            } else {
                log::warn!("dynarmic: unmapped memory {:#x} size={} pc={:#x}", addr, size, pc);
            }
            let snap = FaultSnapshot {
                pc, lr, sp, addr,
                size: size as u32,
                is_write: false,
                value,
                regs,
            };
            *fault_for_unmapped.lock().unwrap() = Some(snap);

            if is_null_zone && continue_flag.load(Ordering::Relaxed) {
                let n = skip_counter.fetch_add(1, Ordering::Relaxed) + 1;
                if n > NULL_SKIP_MAX {
                    log::error!("[null-deref] skip cap ({}) exceeded — emitting Exception", NULL_SKIP_MAX);
                    *event_for_unmapped.lock().unwrap() = Some(CpuEvent::Exception(0x0E));
                    let _ = dyn_.emu_stop();
                    return true;
                }
                return true;
            }

            *event_for_unmapped.lock().unwrap() = Some(CpuEvent::Exception(0x0E));
            let _ = dyn_.emu_stop();
            true
        });

        Ok(Self {
            emu: Arc::new(SharedDynarmic { emu }),
            last_event,
            last_fault,
            continue_on_null,
            null_skip_count,
        })
    }

    pub fn take_fault(&self) -> Option<FaultSnapshot> {
        self.last_fault.lock().unwrap().take()
    }

    pub fn set_continue_on_null(&self, enable: bool) {
        self.continue_on_null.store(enable, Ordering::Relaxed);
    }

    pub fn null_skip_count(&self) -> u32 {
        self.null_skip_count.load(Ordering::Relaxed)
    }

    pub fn halt_handle(&self) -> HaltHandle {
        let emu = Arc::clone(&self.emu);
        let emu_peek = Arc::clone(&self.emu);
        HaltHandle {
            inner: Arc::new(move || {
                let _ = emu.emu.emu_stop();
            }),
            peek: Arc::new(move || {
                let pc = emu_peek.emu.reg_read_pc().unwrap_or(0);
                let lr = emu_peek.emu.reg_read_lr().unwrap_or(0);
                let sp = emu_peek.emu.reg_read_sp().unwrap_or(0);
                (pc, lr, sp)
            }),
        }
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
        log::trace!("dynarmic run: PC={:#x}", pc);
        let _ = self.emu.emu.emu_start(pc, u64::MAX - 16);
        let event = self.last_event.lock().unwrap().take();
        match event {
            Some(CpuEvent::Svc(imm)) => {
                log::debug!("dynarmic SVC {:#04x} hit at PC={:#x}", imm, pc);
                CpuEvent::Svc(imm)
            }
            Some(other) => {
                log::warn!("dynarmic event: {:?}", other);
                other
            }
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

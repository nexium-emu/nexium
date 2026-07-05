use nexium_memory::Perm;
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::Arc;

use crate::{CpuEvent, FaultSnapshot, HaltHandle};

pub(crate) struct SharedDynarmic {
    pub(crate) emu: dynarmic_sys::Dynarmic<'static, ()>,
}
unsafe impl Send for SharedDynarmic {}
unsafe impl Sync for SharedDynarmic {}

const NULL_SKIP_MAX: u32 = 64;

pub struct DynarmicCpu {
    emu: Arc<SharedDynarmic>,
    last_event: Rc<Cell<Option<CpuEvent>>>,
    last_fault: Rc<RefCell<Option<FaultSnapshot>>>,
    continue_on_null: Rc<Cell<bool>>,
    null_skip_count: Rc<Cell<u32>>,
}

unsafe impl Send for DynarmicCpu {}
unsafe impl Sync for DynarmicCpu {}

impl DynarmicCpu {
    pub fn new() -> Result<Self, String> {
        let emu: dynarmic_sys::Dynarmic<'static, ()> = match nexium_memory::fastmem::base() {
            Some(base) => {
                log::info!("dynarmic: fastmem enabled, arena base={:p}", base);
                dynarmic_sys::Dynarmic::new_fastmem(base.cast())
            }
            None => dynarmic_sys::Dynarmic::new(),
        };

        let last_event = Rc::new(Cell::new(None::<CpuEvent>));
        let last_fault: Rc<RefCell<Option<FaultSnapshot>>> = Rc::new(RefCell::new(None));
        let continue_on_null = Rc::new(Cell::new(false));
        let null_skip_count = Rc::new(Cell::new(0u32));

        let event_for_svc = last_event.clone();
        emu.set_svc_callback(move |dyn_, swi, _until, pc| {
            log::trace!(
                "dynarmic SVC callback triggered: swi={:#04x}, pc={:#x}",
                swi,
                pc
            );
            event_for_svc.set(Some(CpuEvent::Svc(swi as u16)));
            let _ = dyn_.emu_stop();
        });
        log::info!("dynarmic: SVC callback registered");

        let event_for_unmapped = last_event.clone();
        let fault_for_unmapped = last_fault.clone();
        let continue_flag = continue_on_null.clone();
        let skip_counter = null_skip_count.clone();
        emu.set_unmapped_mem_callback(move |dyn_, addr, size, value| {
            if let Some((lo, hi)) = nexium_memory::fastmem::watch_range() {
                if addr >= lo && addr < hi {
                    use std::sync::atomic::{AtomicU64, Ordering};
                    static HITS: AtomicU64 = AtomicU64::new(0);
                    let n = HITS.fetch_add(1, Ordering::SeqCst);
                    let pc = dyn_.reg_read_pc().unwrap_or(0);
                    let lr = dyn_.reg_read_lr().unwrap_or(0);
                    if n < 64 {
                        log::warn!(
                            "[watch-write] #{} addr={:#x} size={} val={:#x} pc={:#x} lr={:#x}",
                            n, addr, size, value, pc, lr
                        );
                    }
                    if n >= 64 || value != 0 {
                        if value != 0 {
                            log::warn!(
                                "[watch-write] NONZERO writer pc={:#x} lr={:#x} — disarming",
                                pc, lr
                            );
                        } else {
                            log::warn!("[watch-write] cap reached, disarming");
                        }
                        nexium_memory::fastmem::watch_disarm();
                        unsafe {
                            let base = nexium_memory::fastmem::base().unwrap();
                            let bytes = value.to_le_bytes();
                            std::ptr::copy_nonoverlapping(
                                bytes.as_ptr(),
                                base.add(addr as usize),
                                (size as usize).min(8),
                            );
                        }
                        return true;
                    }
                    if nexium_memory::fastmem::watch_write_through(addr, size as usize, value) {
                        return true;
                    }
                }
            }
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
            *fault_for_unmapped.borrow_mut() = Some(snap);

            if is_null_zone && continue_flag.get() {
                let n = skip_counter.get() + 1;
                skip_counter.set(n);
                if n > NULL_SKIP_MAX {
                    log::error!("[null-deref] skip cap ({}) exceeded — emitting Exception", NULL_SKIP_MAX);
                    event_for_unmapped.set(Some(CpuEvent::Exception(0x0E)));
                    let _ = dyn_.emu_stop();
                    return true;
                }
                return true;
            }

            event_for_unmapped.set(Some(CpuEvent::Exception(0x0E)));
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
        self.last_fault.borrow_mut().take()
    }

    pub fn set_continue_on_null(&self, enable: bool) {
        self.continue_on_null.set(enable);
    }

    pub fn null_skip_count(&self) -> u32 {
        self.null_skip_count.get()
    }

    pub fn halt_handle(&self) -> HaltHandle {
        let emu = Arc::clone(&self.emu);
        let emu_peek = Arc::clone(&self.emu);
        let emu_dump = Arc::clone(&self.emu);
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
            peek_dump: Arc::new(move || {
                let e = &emu_dump.emu;
                let pc = e.reg_read_pc().unwrap_or(0);
                let mut code = [0u8; 64];
                let _ = e.mem_read(pc, &mut code);
                let mut s = format!("pc={:#x}\n  code={:02x?}\n  regs:", pc, &code[..]);
                for i in 0..31 {
                    let r = e.reg_read(i).unwrap_or(0);
                    let mut b = [0u8; 8];
                    let v = if e.mem_read(r, &mut b).is_ok() {
                        u64::from_le_bytes(b)
                    } else {
                        0
                    };
                    s.push_str(&format!(" x{}={:#x}([x{}]={:#x})", i, r, i, v));
                }
                s
            }),
        }
    }

    pub unsafe fn map_host(
        &mut self,
        va: u64,
        len: u64,
        perm: Perm,
        ptr: *mut u8,
    ) -> Result<(), String> {
        self.emu
            .emu
            .mem_map_ptr(va, len as usize, perm_to_dyn(perm), ptr.cast())
            .map_err(|e| format!("map_host failed: {:?}", e))
    }

    pub fn write_bytes(&self, va: u64, bytes: &[u8]) -> Result<(), String> {
        self.emu
            .emu
            .mem_write(va, bytes)
            .map_err(|e| format!("write_bytes failed: {:?}", e))
    }

    pub fn read_bytes(&self, va: u64, buf: &mut [u8]) -> Result<(), String> {
        self.emu
            .emu
            .mem_read(va, buf)
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
        self.last_event.set(None);
        let pc = self.get_pc();
        log::trace!("dynarmic run: PC={:#x}", pc);
        let until = if _max_insn > 0 {
            pc.saturating_add(_max_insn.saturating_mul(4))
        } else {
            u64::MAX - 16
        };
        let _ = self.emu.emu.emu_start(pc, until);
        let event = self.last_event.take();
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
        self.last_event.set(Some(CpuEvent::Svc(imm)));
    }

    pub fn invalidate_range(&mut self, _va: u64, _len: u64) {}
}

fn perm_to_dyn(p: Perm) -> u32 {
    let mut out = 0u32;
    if p.contains(Perm::R) {
        out |= 1;
    }
    if p.contains(Perm::W) {
        out |= 2;
    }
    if p.contains(Perm::X) {
        out |= 4;
    }
    out
}

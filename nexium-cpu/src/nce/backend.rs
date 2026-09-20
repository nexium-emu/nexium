use super::context::{CoreState, GuestContext, NativeExecutionParameters, NceThreadContext};
use super::{asm, signal};
use crate::nce_layout::*;
use crate::{CpuEvent, FaultSnapshot, HaltHandle};
use nexium_memory::Perm;
use std::cell::{Cell, UnsafeCell};
use std::sync::atomic::Ordering;
use std::sync::Arc;

const NULL_SKIP_MAX: u32 = 64;
const MIN_SLICE_NS: u64 = 20_000;
const MAX_SLICE_NS: u64 = 50_000_000;

pub struct CoreBlock {
    ctx: UnsafeCell<GuestContext>,
    nep: UnsafeCell<NativeExecutionParameters>,
    state: Box<CoreState>,
}

unsafe impl Send for CoreBlock {}
unsafe impl Sync for CoreBlock {}

impl CoreBlock {
    fn ctx_ptr(&self) -> *mut GuestContext {
        self.ctx.get()
    }

    fn nep_ptr(&self) -> *mut NativeExecutionParameters {
        self.nep.get()
    }
}

pub struct NceCpu {
    block: Arc<CoreBlock>,
    pending: Cell<Option<CpuEvent>>,
    timer: Cell<Option<libc::timer_t>>,
    timer_tid: Cell<i32>,
    counter_hz: u64,
    core_id: u64,
}

unsafe impl Send for NceCpu {}
unsafe impl Sync for NceCpu {}

fn read_counter_hz() -> u64 {
    let value: u64;
    unsafe {
        std::arch::asm!("mrs {0}, cntfrq_el0", out(reg) value, options(nomem, nostack, preserves_flags));
    }
    value
}

pub unsafe fn flush_icache(start: usize, len: usize) {
    if len == 0 {
        return;
    }
    let ctr: u64;
    std::arch::asm!("mrs {0}, ctr_el0", out(reg) ctr, options(nomem, nostack, preserves_flags));
    let dline = 4usize << ((ctr >> 16) & 0xF);
    let iline = 4usize << (ctr & 0xF);
    let end = start + len;
    let mut addr = start & !(dline - 1);
    while addr < end {
        std::arch::asm!("dc cvau, {0}", in(reg) addr, options(nostack, preserves_flags));
        addr += dline;
    }
    std::arch::asm!("dsb ish", options(nostack, preserves_flags));
    let mut addr = start & !(iline - 1);
    while addr < end {
        std::arch::asm!("ic ivau, {0}", in(reg) addr, options(nostack, preserves_flags));
        addr += iline;
    }
    std::arch::asm!("dsb ish", "isb", options(nostack, preserves_flags));
}

fn prot_for(perm: Perm) -> libc::c_int {
    let mut prot = libc::PROT_NONE;
    if perm.intersects(Perm::R | Perm::W | Perm::X) {
        prot |= libc::PROT_READ | libc::PROT_WRITE;
    }
    if perm.contains(Perm::X) {
        prot |= libc::PROT_EXEC;
    }
    prot
}

fn lock_nep(nep: *mut NativeExecutionParameters) {
    let lock = unsafe { &(*nep).lock };
    let mut spins = 0u64;
    loop {
        if lock
            .compare_exchange_weak(LOCK_UNLOCKED, LOCK_LOCKED, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
        {
            return;
        }
        spins += 1;
        if spins == 50_000_000 {
            log::error!(
                "nce: thread parameters lock stuck (value={} is_running={}); the guest left a trampoline critical section",
                lock.load(Ordering::Relaxed),
                unsafe { (*nep).is_running.load(Ordering::Relaxed) }
            );
        }
        std::hint::spin_loop();
    }
}

#[repr(C)]
struct SigEvent {
    value: u64,
    signo: i32,
    notify: i32,
    tid: i32,
    pad: [u8; 44],
}

impl NceCpu {
    pub fn new() -> Result<Self, String> {
        if !super::supported() {
            return Err("NCE requires the direct-mapped fastmem arena (aarch64 host, 4 KiB pages)".to_string());
        }
        signal::install_handlers()?;
        let state = Box::new(CoreState::new(NULL_SKIP_MAX));
        let state_ptr = &*state as *const CoreState as *mut CoreState;
        let block = Arc::new(CoreBlock {
            ctx: UnsafeCell::new(GuestContext::new(state_ptr)),
            nep: UnsafeCell::new(NativeExecutionParameters::new()),
            state,
        });
        let counter_hz = read_counter_hz();
        log::info!(
            "nce: native execution backend ready (host counter {} Hz, guest {} Hz)",
            counter_hz,
            GUEST_CNTFRQ_HZ
        );
        Ok(Self {
            block,
            pending: Cell::new(None),
            timer: Cell::new(None),
            timer_tid: Cell::new(-1),
            counter_hz,
            core_id: 0,
        })
    }

    pub fn counter_hz(&self) -> u64 {
        self.counter_hz
    }

    pub fn stats(&self) -> String {
        let state = &self.block.state;
        format!(
            "nce core{} entries tramp={} signal={} exits svc={} break={} fault={} idle={} null_skips={}",
            self.core_id,
            state.entries_trampoline.load(Ordering::Relaxed),
            state.entries_signal.load(Ordering::Relaxed),
            state.exits_svc.load(Ordering::Relaxed),
            state.exits_break.load(Ordering::Relaxed),
            state.exits_fault.load(Ordering::Relaxed),
            state.exits_idle.load(Ordering::Relaxed),
            state.null_skips.load(Ordering::Relaxed),
        )
    }

    pub fn set_core_id(&mut self, core_id: u64) {
        self.core_id = core_id;
    }

    fn ctx(&self) -> &GuestContext {
        unsafe { &*self.block.ctx_ptr() }
    }

    fn ctx_mut(&mut self) -> &mut GuestContext {
        unsafe { &mut *self.block.ctx_ptr() }
    }

    fn prepare_thread(&self) -> i32 {
        let tid = unsafe { libc::gettid() };
        self.block.state.tid.store(tid, Ordering::Release);
        signal::ensure_alt_stack();
        if self.timer_tid.get() != tid {
            if let Some(timer) = self.timer.take() {
                unsafe {
                    libc::timer_delete(timer);
                }
            }
            let event = SigEvent {
                value: 0,
                signo: signal::SIGNAL_BREAK,
                notify: libc::SIGEV_THREAD_ID,
                tid,
                pad: [0; 44],
            };
            let mut timer: libc::timer_t = std::ptr::null_mut();
            let rc = unsafe {
                libc::timer_create(
                    libc::CLOCK_MONOTONIC,
                    &event as *const SigEvent as *mut libc::sigevent,
                    &mut timer,
                )
            };
            if rc == 0 {
                self.timer.set(Some(timer));
            } else {
                log::warn!(
                    "nce: timer_create failed ({}); slices will only end on SVCs and halts",
                    std::io::Error::last_os_error()
                );
            }
            self.timer_tid.set(tid);
        }
        tid
    }

    fn arm_timer(&self, ns: u64) {
        let Some(timer) = self.timer.get() else {
            return;
        };
        let period = libc::timespec {
            tv_sec: (ns / 1_000_000_000) as libc::time_t,
            tv_nsec: (ns % 1_000_000_000) as libc::c_long,
        };
        let spec = libc::itimerspec {
            it_interval: period,
            it_value: period,
        };
        unsafe {
            libc::timer_settime(timer, 0, &spec, std::ptr::null_mut());
        }
    }

    fn disarm_timer(&self) {
        let Some(timer) = self.timer.get() else {
            return;
        };
        let spec: libc::itimerspec = unsafe { std::mem::zeroed() };
        unsafe {
            libc::timer_settime(timer, 0, &spec, std::ptr::null_mut());
        }
    }

    pub fn run_with_count(&mut self, max_insn: u64) -> (CpuEvent, u64) {
        if let Some(event) = self.pending.take() {
            return (event, 0);
        }
        let slice_ns = if max_insn == 0 {
            MAX_SLICE_NS
        } else {
            max_insn.clamp(MIN_SLICE_NS, MAX_SLICE_NS)
        };
        let tid = self.prepare_thread();
        let ctx = self.block.ctx_ptr();
        let nep = self.block.nep_ptr();
        unsafe {
            if (*ctx).esr.swap(0, Ordering::AcqRel) & HALT_BREAK_LOOP != 0 {
                self.block.state.exits_idle.fetch_add(1, Ordering::Relaxed);
                return (CpuEvent::Running, 0);
            }
            lock_nep(nep);
            (*nep).native_context = ctx;
            (*nep).tpidr_el0 = (*ctx).tpidr_el0;
            (*nep).tpidrro_el0 = (*ctx).tpidrro_el0;
            (*nep).is_running.store(1, Ordering::Release);
            signal::CURRENT_NEP.with(|slot| slot.set(nep));
            self.arm_timer(slice_ns);
            let started = std::time::Instant::now();
            let pc = (*ctx).pc;
            let halt = match super::lookup_post_handler(pc) {
                Some(trampoline) => {
                    self.block.state.entries_trampoline.fetch_add(1, Ordering::Relaxed);
                    asm::nexium_nce_enter_trampoline(nep as *mut u8, ctx as *mut u8, trampoline)
                }
                None => {
                    self.block.state.entries_signal.fetch_add(1, Ordering::Relaxed);
                    asm::nexium_nce_enter_signal(tid, nep as *mut u8)
                }
            };
            self.disarm_timer();
            let counter = if halt & HALT_SUPERVISOR_CALL != 0 {
                &self.block.state.exits_svc
            } else if halt & (HALT_PREFETCH_ABORT | HALT_DATA_ABORT | HALT_ALIGNMENT | HALT_BREAKPOINT) != 0 {
                &self.block.state.exits_fault
            } else {
                &self.block.state.exits_break
            };
            counter.fetch_add(1, Ordering::Relaxed);
            (*nep).is_running.store(0, Ordering::Release);
            (*ctx).tpidr_el0 = (*nep).tpidr_el0;
            (*nep).native_context = std::ptr::null_mut();
            (*nep).lock.store(LOCK_UNLOCKED, Ordering::Release);
            let elapsed = started.elapsed().as_nanos() as u64;
            let event = if halt & HALT_SUPERVISOR_CALL != 0 {
                CpuEvent::Svc((*ctx).svc as u16)
            } else if halt & HALT_PREFETCH_ABORT != 0 {
                CpuEvent::Exception(0x21)
            } else if halt & HALT_DATA_ABORT != 0 {
                CpuEvent::Exception(0x25)
            } else if halt & HALT_ALIGNMENT != 0 {
                CpuEvent::Exception(0x22)
            } else if halt & HALT_BREAKPOINT != 0 {
                CpuEvent::Exception(0x3C)
            } else {
                CpuEvent::Running
            };
            (event, elapsed.clamp(1, slice_ns))
        }
    }

    pub fn step(&mut self) -> CpuEvent {
        self.run_with_count(1).0
    }

    pub fn inject_svc(&mut self, imm: u16) {
        self.pending.set(Some(CpuEvent::Svc(imm)));
    }

    pub fn halt_handle(&self) -> HaltHandle {
        let block = Arc::clone(&self.block);
        let block_peek = Arc::clone(&self.block);
        let block_dump = Arc::clone(&self.block);
        HaltHandle {
            inner: Arc::new(move || unsafe {
                let ctx = block.ctx_ptr();
                let nep = block.nep_ptr();
                (*ctx).esr.fetch_or(HALT_BREAK_LOOP, Ordering::AcqRel);
                lock_nep(nep);
                if (*nep).is_running.load(Ordering::Acquire) != 0 {
                    let tid = block.state.tid.load(Ordering::Acquire);
                    libc::tgkill(libc::getpid(), tid, signal::SIGNAL_BREAK);
                } else {
                    (*nep).lock.store(LOCK_UNLOCKED, Ordering::Release);
                }
            }),
            peek: Arc::new(move || unsafe {
                let ctx = block_peek.ctx_ptr();
                let pc = std::ptr::read_volatile(std::ptr::addr_of!((*ctx).pc));
                let lr = std::ptr::read_volatile(std::ptr::addr_of!((*ctx).x[30]));
                let sp = std::ptr::read_volatile(std::ptr::addr_of!((*ctx).sp));
                (pc, lr, sp)
            }),
            peek_dump: Arc::new(move || unsafe {
                let ctx = block_dump.ctx_ptr();
                let pc = std::ptr::read_volatile(std::ptr::addr_of!((*ctx).pc));
                let mut s = format!("pc={:#x}\n  regs:", pc);
                for i in 0..31 {
                    let r = std::ptr::read_volatile(std::ptr::addr_of!((*ctx).x[i]));
                    s.push_str(&format!(" x{}={:#x}", i, r));
                }
                s
            }),
        }
    }

    pub unsafe fn map_host(&mut self, va: u64, len: u64, perm: Perm, ptr: *mut u8) -> Result<(), String> {
        if ptr as u64 != va {
            return Err(format!(
                "nce: region {:#x} is not identity mapped (host {:p}); direct fastmem is required",
                va, ptr
            ));
        }
        if libc::mprotect(va as *mut libc::c_void, len as usize, prot_for(perm)) != 0 {
            return Err(format!(
                "nce: mprotect({:#x}, {:#x}, {}) failed: {}",
                va,
                len,
                perm,
                std::io::Error::last_os_error()
            ));
        }
        if perm.contains(Perm::X) {
            flush_icache(va as usize, len as usize);
        }
        Ok(())
    }

    pub unsafe fn unmap_host(&mut self, va: u64, len: u64) -> Result<(), String> {
        if !super::direct_window_contains(va, len) {
            return Ok(());
        }
        if libc::mprotect(va as *mut libc::c_void, len as usize, libc::PROT_NONE) != 0 {
            return Err(format!(
                "nce: mprotect(none) failed for {:#x}: {}",
                va,
                std::io::Error::last_os_error()
            ));
        }
        Ok(())
    }

    pub fn set_register(&mut self, reg: u32, val: u64) {
        let ctx = self.ctx_mut();
        match reg {
            0..=30 => ctx.x[reg as usize] = val,
            31 => ctx.sp = val,
            _ => {}
        }
    }

    pub fn get_register(&self, reg: u32) -> u64 {
        let ctx = self.ctx();
        match reg {
            0..=30 => ctx.x[reg as usize],
            31 => ctx.sp,
            _ => 0,
        }
    }

    pub fn set_pc(&mut self, pc: u64) {
        self.ctx_mut().pc = pc;
    }

    pub fn get_pc(&self) -> u64 {
        self.ctx().pc
    }

    pub fn set_sp(&mut self, sp: u64) {
        self.ctx_mut().sp = sp;
    }

    pub fn get_sp(&self) -> u64 {
        self.ctx().sp
    }

    pub fn set_tpidrro_el0(&mut self, val: u64) {
        self.ctx_mut().tpidrro_el0 = val;
    }

    pub fn get_tpidrro_el0(&self) -> u64 {
        self.ctx().tpidrro_el0
    }

    pub fn save_thread_context(&self) -> NceThreadContext {
        NceThreadContext::capture(self.ctx())
    }

    pub fn restore_thread_context(&mut self, context: &NceThreadContext) {
        context.apply(self.ctx_mut());
    }

    pub fn reset_thread_context(&mut self) {
        self.ctx_mut().reset_thread_state();
    }

    pub fn write_bytes(&self, va: u64, bytes: &[u8]) -> Result<(), String> {
        if !super::direct_window_contains(va, bytes.len() as u64) {
            return Err(format!("nce: write outside guest window at {:#x}", va));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), va as *mut u8, bytes.len());
        }
        Ok(())
    }

    pub fn read_bytes(&self, va: u64, buf: &mut [u8]) -> Result<(), String> {
        if !super::direct_window_contains(va, buf.len() as u64) {
            return Err(format!("nce: read outside guest window at {:#x}", va));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(va as *const u8, buf.as_mut_ptr(), buf.len());
        }
        Ok(())
    }

    pub fn take_fault(&self) -> Option<FaultSnapshot> {
        let fault = &self.block.state.fault;
        if fault.valid.swap(0, Ordering::AcqRel) == 0 {
            return None;
        }
        let mut regs = [0u64; 31];
        for (dst, src) in regs.iter_mut().zip(fault.regs.iter()) {
            *dst = src.load(Ordering::Relaxed);
        }
        Some(FaultSnapshot {
            pc: fault.pc.load(Ordering::Relaxed),
            lr: fault.lr.load(Ordering::Relaxed),
            sp: fault.sp.load(Ordering::Relaxed),
            addr: fault.addr.load(Ordering::Relaxed),
            size: 0,
            is_write: fault.is_write.load(Ordering::Relaxed) != 0,
            value: 0,
            regs,
        })
    }

    pub fn set_continue_on_null(&self, enable: bool) {
        self.block
            .state
            .continue_on_null
            .store(enable as u32, Ordering::Relaxed);
    }

    pub fn null_skip_count(&self) -> u32 {
        self.block.state.null_skips.load(Ordering::Relaxed)
    }

    pub fn invalidate_range(&mut self, va: u64, len: u64) {
        if super::direct_window_contains(va, len) {
            unsafe {
                flush_icache(va as usize, len as usize);
            }
        }
    }
}

impl Drop for NceCpu {
    fn drop(&mut self) {
        if let Some(timer) = self.timer.take() {
            unsafe {
                libc::timer_delete(timer);
            }
        }
    }
}

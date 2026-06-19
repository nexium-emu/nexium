use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use nexium_memory::Perm;
use rustarmic::{Jit, JitConfig, CpuContext, ExitReason, Memory};

use crate::{CpuEvent, FaultSnapshot, HaltHandle};

const NULL_SKIP_MAX: u32 = 64;

#[derive(Clone, Copy, Debug)]
struct Region {
    va: u64,
    end: u64,
    host_ptr: *mut u8,
    perm: Perm,
}
unsafe impl Send for Region {}
unsafe impl Sync for Region {}

#[repr(C)]
struct State {
    ctx: CpuContext,
    regions: parking_lot::RwLock<Vec<Region>>,
    halt: AtomicBool,
    last_event: Mutex<Option<CpuEvent>>,
    last_fault: Mutex<Option<FaultSnapshot>>,
    continue_on_null: AtomicBool,
    null_skip_count: AtomicU32,
    peek_pc: AtomicU64,
    peek_lr: AtomicU64,
    peek_sp: AtomicU64,
    pending_invalidations: Mutex<Vec<(u64, u64)>>,
}

pub struct RustarmicCpu {
    state: Box<State>,
    jit: Jit,
}

unsafe impl Send for RustarmicCpu {}
unsafe impl Sync for RustarmicCpu {}

impl RustarmicCpu {
    pub fn new() -> Result<Self, String> {
        let mut state = Box::new(State {
            ctx: CpuContext::default(),
            regions: parking_lot::RwLock::new(Vec::new()),
            halt: AtomicBool::new(false),
            last_event: Mutex::new(None),
            last_fault: Mutex::new(None),
            continue_on_null: AtomicBool::new(false),
            null_skip_count: AtomicU32::new(0),
            peek_pc: AtomicU64::new(0),
            peek_lr: AtomicU64::new(0),
            peek_sp: AtomicU64::new(0),
            pending_invalidations: Mutex::new(Vec::new()),
        });
        state.ctx.mem_read  = mem_read_hook;
        state.ctx.mem_write = mem_write_hook;
        let jit = Jit::new(JitConfig::default()).map_err(|e| format!("rustarmic Jit init: {:?}", e))?;
        Ok(Self { state, jit })
    }

    pub fn take_fault(&self) -> Option<FaultSnapshot> {
        self.state.last_fault.lock().unwrap().take()
    }

    pub fn set_continue_on_null(&self, enable: bool) {
        self.state.continue_on_null.store(enable, Ordering::Relaxed);
    }

    pub fn null_skip_count(&self) -> u32 {
        self.state.null_skip_count.load(Ordering::Relaxed)
    }

    pub fn halt_handle(&self) -> HaltHandle {
        let halt_addr = &self.state.halt as *const AtomicBool as usize;
        let ctx_addr  = &self.state.ctx  as *const CpuContext as usize;
        HaltHandle {
            inner: Arc::new(move || {
                unsafe { (*(halt_addr as *const AtomicBool)).store(true, Ordering::Relaxed); }
            }),
            // Block-boundary peek: rustarmic's dispatcher writes ctx.pc after
            // every block exit, so reading directly here gives the last-block
            // PC rather than only the last-run() PC (which is what the prior
            // `peek_pc` cache served). The watchdog in nexium-gui samples this
            // mid-execution; stale snapshots produced false hang verdicts.
            peek: Arc::new(move || unsafe {
                let ctx = ctx_addr as *const CpuContext;
                let p = std::ptr::read_volatile(&(*ctx).pc);
                let l = std::ptr::read_volatile(&(*ctx).x[30]);
                let s = std::ptr::read_volatile(&(*ctx).sp);
                (p, l, s)
            }),
            peek_dump: Arc::new(|| String::from("(peek_dump unsupported on rustarmic)")),
        }
    }

    pub unsafe fn map_host(&mut self, va: u64, len: u64, perm: Perm, ptr: *mut u8) -> Result<(), String> {
        let end = va.checked_add(len).ok_or_else(|| format!("map_host overflow va={:#x} len={:#x}", va, len))?;
        log::debug!("rustarmic map_host va={:#x}..{:#x} perm={} ptr={:p}", va, end, perm, ptr);
        if perm.contains(Perm::X) {
            self.jit.invalidate_range(va, len);
        }
        self.state.regions.write().push(Region { va, end, host_ptr: ptr, perm });
        Ok(())
    }

    pub fn write_bytes(&self, va: u64, bytes: &[u8]) -> Result<(), String> {
        let regions = self.state.regions.read();
        for r in regions.iter() {
            if va >= r.va && va.saturating_add(bytes.len() as u64) <= r.end {
                if !r.perm.contains(Perm::W) {
                    return Err(format!("write_bytes: read-only region va={:#x} perm={}", va, r.perm));
                }
                unsafe {
                    let dst = r.host_ptr.add((va - r.va) as usize);
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
                }
                if r.perm.contains(Perm::X) {
                    self.state.pending_invalidations
                        .lock().unwrap()
                        .push((va, bytes.len() as u64));
                }
                return Ok(());
            }
        }
        Err(format!("write_bytes: unmapped va={:#x}", va))
    }

    pub fn invalidate_range(&mut self, va: u64, len: u64) {
        let queued: Vec<(u64, u64)> =
            std::mem::take(&mut *self.state.pending_invalidations.lock().unwrap());
        for (qva, qlen) in queued {
            self.jit.invalidate_range(qva, qlen);
        }
        self.jit.invalidate_range(va, len);
    }

    pub fn read_bytes(&self, va: u64, buf: &mut [u8]) -> Result<(), String> {
        let regions = self.state.regions.read();
        for r in regions.iter() {
            if va >= r.va && va.saturating_add(buf.len() as u64) <= r.end {
                unsafe {
                    let src = r.host_ptr.add((va - r.va) as usize);
                    std::ptr::copy_nonoverlapping(src, buf.as_mut_ptr(), buf.len());
                }
                return Ok(());
            }
        }
        Err(format!("read_bytes: unmapped va={:#x}", va))
    }

    pub fn set_register(&mut self, reg: u32, val: u64) {
        if reg < 31 { self.state.ctx.x[reg as usize] = val; }
        else if reg == 31 { self.state.ctx.sp = val; }
    }

    pub fn get_register(&self, reg: u32) -> u64 {
        if reg < 31 { self.state.ctx.x[reg as usize] }
        else if reg == 31 { self.state.ctx.sp }
        else { 0 }
    }

    pub fn set_pc(&mut self, pc: u64) { self.state.ctx.pc = pc; }
    pub fn get_pc(&self) -> u64       { self.state.ctx.pc }
    pub fn set_sp(&mut self, sp: u64) { self.state.ctx.sp = sp; }
    pub fn get_sp(&self) -> u64       { self.state.ctx.sp }

    pub fn set_tpidrro_el0(&mut self, val: u64) { self.state.ctx.tpidrro_el0 = val; }
    pub fn get_tpidrro_el0(&self) -> u64        { self.state.ctx.tpidrro_el0 }

    pub fn run(&mut self, _max_insn: u64) -> CpuEvent {
        let queued: Vec<(u64, u64)> =
            std::mem::take(&mut *self.state.pending_invalidations.lock().unwrap());
        if !queued.is_empty() {
            log::debug!("rustarmic: applying {} queued SMC invalidations", queued.len());
            for (va, len) in queued {
                self.jit.invalidate_range(va, len);
            }
        }

        // Consume any event posted between runs (inject_svc, NULL_SKIP_MAX
        // overflow). Clearing unconditionally — as the prior code did —
        // dropped injected SVCs and the null-deref escalation on the floor.
        if let Some(ev) = self.state.last_event.lock().unwrap().take() {
            return ev;
        }
        self.state.halt.store(false, Ordering::Relaxed);

        let regions: Vec<Region> = self.state.regions.read().iter().copied().collect();
        let mut mem = RegionMemory { regions };

        let exit = self.jit.run(&mut self.state.ctx, &mut mem);

        self.state.peek_pc.store(self.state.ctx.pc,    Ordering::Relaxed);
        self.state.peek_lr.store(self.state.ctx.x[30], Ordering::Relaxed);
        self.state.peek_sp.store(self.state.ctx.sp,    Ordering::Relaxed);

        match exit {
            Ok(ExitReason::Svc(imm))      => CpuEvent::Svc(imm as u16),
            Ok(ExitReason::Brk(imm))      => CpuEvent::Exception(0x100 | imm),
            Ok(ExitReason::Hvc(imm))      => CpuEvent::Exception(0x200 | imm),
            Ok(ExitReason::MemoryFault(_)) => CpuEvent::Exception(0x0E),
            Ok(ExitReason::Stopped)        => CpuEvent::Interrupted,
            Err(e) => {
                match &e {
                    rustarmic::Error::Unsupported { pc, opcode } => {
                        let mut b = [0u8; 4];
                        let word = if self.read_bytes(*pc, &mut b).is_ok() { u32::from_le_bytes(b) } else { *opcode };
                        log::warn!("rustarmic Jit::run unsupported: pc={:#x} opcode={:#010x}", pc, word);
                    }
                    rustarmic::Error::Decode { pc, opcode } => {
                        let mut b = [0u8; 4];
                        let word = if self.read_bytes(*pc, &mut b).is_ok() { u32::from_le_bytes(b) } else { *opcode };
                        log::warn!("rustarmic Jit::run decode-fail: pc={:#x} opcode={:#010x}", pc, word);
                    }
                    other => log::warn!("rustarmic Jit::run error: {:?}", other),
                }
                CpuEvent::Stalled
            }
        }
    }

    pub fn step(&mut self) -> CpuEvent { self.run(1) }

    pub fn inject_svc(&mut self, imm: u16) {
        *self.state.last_event.lock().unwrap() = Some(CpuEvent::Svc(imm));
    }
}

struct RegionMemory {
    regions: Vec<Region>,
}

impl Memory for RegionMemory {
    fn fetch_inst(&mut self, addr: u64) -> Option<u32> {
        for r in &self.regions {
            if addr >= r.va && addr.checked_add(4)? <= r.end && r.perm.contains(Perm::X) {
                unsafe {
                    let host = r.host_ptr.add((addr - r.va) as usize);
                    let mut buf = [0u8; 4];
                    std::ptr::copy_nonoverlapping(host, buf.as_mut_ptr(), 4);
                    return Some(u32::from_le_bytes(buf));
                }
            }
        }
        log::warn!("rustarmic fetch_inst miss at {:#x} ({} regions)", addr, self.regions.len());
        for r in &self.regions {
            log::warn!("  region {:#x}..{:#x} perm={} x={}", r.va, r.end, r.perm, r.perm.contains(Perm::X));
        }
        None
    }
}

fn find_region(regions: &[Region], va: u64, len: u64) -> Option<&Region> {
    regions.iter().find(|r| va >= r.va && va.checked_add(len).map_or(false, |e| e <= r.end))
}

unsafe extern "C" fn mem_read_hook(ctx_ptr: *mut CpuContext, addr: u64, size: u8) {
    let state = unsafe { &*(ctx_ptr as *mut State) };
    let regions = state.regions.read();
    if let Some(r) = find_region(&regions, addr, size as u64) {
        unsafe {
            let host = r.host_ptr.add((addr - r.va) as usize);
            let mut buf = [0u8; 16];
            std::ptr::copy_nonoverlapping(host, buf.as_mut_ptr(), size as usize);
            let lo = u64::from_le_bytes(buf[..8].try_into().unwrap());
            let hi = u64::from_le_bytes(buf[8..].try_into().unwrap());
            (*ctx_ptr).io_value = [lo, hi];
        }
        return;
    }
    handle_unmapped(state, ctx_ptr, addr, size, false, 0);
    unsafe { (*ctx_ptr).io_value = [0, 0]; }
}

unsafe extern "C" fn mem_write_hook(ctx_ptr: *mut CpuContext, addr: u64, size: u8) {
    let state = unsafe { &*(ctx_ptr as *mut State) };
    let io = unsafe { (*ctx_ptr).io_value };
    let mut buf = [0u8; 16];
    buf[..8].copy_from_slice(&io[0].to_le_bytes());
    buf[8..].copy_from_slice(&io[1].to_le_bytes());
    let value = io[0];
    let regions = state.regions.read();
    if let Some(r) = find_region(&regions, addr, size as u64) {
        if r.perm.contains(Perm::W) {
            unsafe {
                let host = r.host_ptr.add((addr - r.va) as usize);
                std::ptr::copy_nonoverlapping(buf.as_ptr(), host, size as usize);
            }
            if r.perm.contains(Perm::X) {
                state.pending_invalidations
                    .lock().unwrap()
                    .push((addr, size as u64));
                unsafe { (*ctx_ptr).should_halt = 1; }
            }
            return;
        }
    }
    handle_unmapped(state, ctx_ptr, addr, size, true, value);
}

fn handle_unmapped(state: &State, ctx_ptr: *mut CpuContext, addr: u64, size: u8, is_write: bool, value: u64) {
    let is_null_zone = addr < 0x1000;
    let mut regs = [0u64; 31];
    let (live_pc, live_lr, live_sp);
    unsafe {
        let ctx = &*ctx_ptr;
        for i in 0..31 { regs[i] = ctx.x[i]; }
        live_pc = ctx.pc;
        live_lr = ctx.x[30];
        live_sp = ctx.sp;
    }
    let snap = FaultSnapshot {
        pc: live_pc,
        lr: live_lr,
        sp: live_sp,
        addr, size: size as u32, is_write, value,
        regs,
    };
    *state.last_fault.lock().unwrap() = Some(snap);
    if is_null_zone {
        log::error!("[null-deref] addr={:#x} size={} write={} val={:#x}", addr, size, is_write, value);
        if state.continue_on_null.load(Ordering::Relaxed) {
            let n = state.null_skip_count.fetch_add(1, Ordering::Relaxed) + 1;
            if n > NULL_SKIP_MAX {
                *state.last_event.lock().unwrap() = Some(CpuEvent::Exception(0x0E));
            }
        }
    } else {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNT: AtomicU64 = AtomicU64::new(0);
        const RUNAWAY_THRESHOLD: u64 = 1_000_000;

        let n = COUNT.fetch_add(1, Ordering::Relaxed);
        if n < 16 {
            log::warn!("rustarmic: unmapped {:#x} size={} write={}", addr, size, is_write);
        } else if n & 0xFFFF == 0 {
            log::warn!("rustarmic: unmapped {:#x} size={} write={} (total {} so far — likely runaway loop)",
                addr, size, is_write, n + 1);
        }
        if n == RUNAWAY_THRESHOLD {
            let (pc, sp, regs) = unsafe {
                let ctx = &*ctx_ptr;
                let mut r = [0u64; 31];
                for i in 0..31 { r[i] = ctx.x[i]; }
                (ctx.pc, ctx.sp, r)
            };
            log::error!(
                "rustarmic: runaway-loop threshold ({} unmapped accesses) reached at addr={:#x}",
                RUNAWAY_THRESHOLD, addr
            );
            log::error!("  PC = {:#018x}  SP = {:#018x}  LR(x30) = {:#018x}", pc, sp, regs[30]);
            log::error!("  x0..x7   {:#018x} {:#018x} {:#018x} {:#018x} {:#018x} {:#018x} {:#018x} {:#018x}",
                regs[0], regs[1], regs[2], regs[3], regs[4], regs[5], regs[6], regs[7]);
            log::error!("  x8..x15  {:#018x} {:#018x} {:#018x} {:#018x} {:#018x} {:#018x} {:#018x} {:#018x}",
                regs[8], regs[9], regs[10], regs[11], regs[12], regs[13], regs[14], regs[15]);
            log::error!("  x16..x23 {:#018x} {:#018x} {:#018x} {:#018x} {:#018x} {:#018x} {:#018x} {:#018x}",
                regs[16], regs[17], regs[18], regs[19], regs[20], regs[21], regs[22], regs[23]);
            log::error!("  x24..x30 {:#018x} {:#018x} {:#018x} {:#018x} {:#018x} {:#018x} {:#018x}",
                regs[24], regs[25], regs[26], regs[27], regs[28], regs[29], regs[30]);
            log::error!("  → halting JIT cooperatively");

            unsafe { (*ctx_ptr).should_halt = 1; }
            *state.last_event.lock().unwrap() = Some(CpuEvent::Exception(0x0E));
        }
    }
}

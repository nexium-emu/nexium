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
        let halt_addr = &self.state.halt    as *const AtomicBool as usize;
        let pc_addr   = &self.state.peek_pc as *const AtomicU64  as usize;
        let lr_addr   = &self.state.peek_lr as *const AtomicU64  as usize;
        let sp_addr   = &self.state.peek_sp as *const AtomicU64  as usize;
        HaltHandle {
            inner: Arc::new(move || {
                unsafe { (*(halt_addr as *const AtomicBool)).store(true, Ordering::Relaxed); }
            }),
            peek: Arc::new(move || unsafe {
                let p = (*(pc_addr as *const AtomicU64)).load(Ordering::Relaxed);
                let l = (*(lr_addr as *const AtomicU64)).load(Ordering::Relaxed);
                let s = (*(sp_addr as *const AtomicU64)).load(Ordering::Relaxed);
                (p, l, s)
            }),
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
                unsafe {
                    let dst = r.host_ptr.add((va - r.va) as usize);
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
                }
                return Ok(());
            }
        }
        Err(format!("write_bytes: unmapped va={:#x}", va))
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
        *self.state.last_event.lock().unwrap() = None;
        self.state.halt.store(false, Ordering::Relaxed);

        let regions: Vec<Region> = self.state.regions.read().iter().copied().collect();
        eprintln!("[rustarmic] run pc={:#x} regions={}", self.state.ctx.pc, regions.len());
        for r in &regions {
            eprintln!("  region {:#x}..{:#x} perm={} x={}", r.va, r.end, r.perm, r.perm.contains(Perm::X));
        }
        let mut mem = RegionMemory { regions };

        let exit = self.jit.run(&mut self.state.ctx, &mut mem);
        eprintln!("[rustarmic] exit = {:?}", exit);

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
                log::warn!("rustarmic Jit::run error: {:?}", e);
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
            return;
        }
    }
    handle_unmapped(state, ctx_ptr, addr, size, true, value);
}

fn handle_unmapped(state: &State, _ctx_ptr: *mut CpuContext, addr: u64, size: u8, is_write: bool, value: u64) {
    let is_null_zone = addr < 0x1000;
    let snap = FaultSnapshot {
        pc: state.peek_pc.load(Ordering::Relaxed),
        lr: state.peek_lr.load(Ordering::Relaxed),
        sp: state.peek_sp.load(Ordering::Relaxed),
        addr, size: size as u32, is_write, value,
        regs: [0; 31],
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
        log::warn!("rustarmic: unmapped {:#x} size={} write={}", addr, size, is_write);
    }
}

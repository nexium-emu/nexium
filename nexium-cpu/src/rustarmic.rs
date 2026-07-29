use nexium_memory::Perm;
use rustarmic::{CpuContext, Jit, JitConfig, Memory, StopReason};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

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

#[derive(Clone, Debug)]
pub struct RustarmicThreadContext {
    v: [[u64; 2]; 32],
    tpidr_el0: u64,
    fpcr: u32,
    fpsr: u32,
    nzcv: u8,
}

#[repr(C)]
struct State {
    ctx: CpuContext,
    regions: parking_lot::RwLock<Vec<Region>>,
    control: Arc<Control>,
    last_event: Mutex<Option<CpuEvent>>,
    last_fault: Mutex<Option<FaultSnapshot>>,
    continue_on_null: AtomicBool,
    null_skip_count: AtomicU32,
    pending_invalidations: Mutex<Vec<(u64, u64)>>,
}

#[repr(C)]
struct Control {
    halt: AtomicBool,
    snapshot_seq: AtomicU64,
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
        Self::new_with_jit_config(JitConfig::default())
    }

    pub fn new_with_engine_config(config: &rustarmic::EngineConfig) -> Result<Self, String> {
        let mut jit = JitConfig::default();
        jit.code_cache_bytes = config.code_cache_bytes;
        jit.use_fastmem = matches!(config.memory_mode, rustarmic::MemoryMode::Fastmem);
        jit.translate.max_insts = config.max_block_insts;
        jit.host_features = Some(config.host_features);
        Self::new_with_jit_config(jit)
    }

    fn new_with_jit_config(jit_config: JitConfig) -> Result<Self, String> {
        let mut state = Box::new(State {
            ctx: CpuContext::default(),
            regions: parking_lot::RwLock::new(Vec::new()),
            control: Arc::new(Control {
                halt: AtomicBool::new(false),
                snapshot_seq: AtomicU64::new(0),
                peek_pc: AtomicU64::new(0),
                peek_lr: AtomicU64::new(0),
                peek_sp: AtomicU64::new(0),
            }),
            last_event: Mutex::new(None),
            last_fault: Mutex::new(None),
            continue_on_null: AtomicBool::new(false),
            null_skip_count: AtomicU32::new(0),
            pending_invalidations: Mutex::new(Vec::new()),
        });
        state.ctx.stop_token = Arc::as_ptr(&state.control).cast();
        state.ctx.mem_read = mem_read_hook;
        state.ctx.mem_write = mem_write_hook;
        let jit = Jit::new(jit_config).map_err(|e| format!("rustarmic Jit init: {:?}", e))?;
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
        let control = Arc::clone(&self.state.control);
        let peek_control = Arc::clone(&self.state.control);
        HaltHandle {
            inner: Arc::new(move || {
                control.halt.store(true, Ordering::Release);
            }),
            peek: Arc::new(move || loop {
                let before = peek_control.snapshot_seq.load(Ordering::Acquire);
                if before & 1 != 0 {
                    std::hint::spin_loop();
                    continue;
                }
                let p = peek_control.peek_pc.load(Ordering::Relaxed);
                let l = peek_control.peek_lr.load(Ordering::Relaxed);
                let s = peek_control.peek_sp.load(Ordering::Relaxed);
                let after = peek_control.snapshot_seq.load(Ordering::Acquire);
                if before == after {
                    break (p, l, s);
                }
            }),
            peek_dump: Arc::new(|| String::from("(peek_dump unsupported on rustarmic)")),
        }
    }

    pub unsafe fn map_host(
        &mut self,
        va: u64,
        len: u64,
        perm: Perm,
        ptr: *mut u8,
    ) -> Result<(), String> {
        let end = va
            .checked_add(len)
            .ok_or_else(|| format!("map_host overflow va={:#x} len={:#x}", va, len))?;
        log::debug!(
            "rustarmic map_host va={:#x}..{:#x} perm={} ptr={:p}",
            va,
            end,
            perm,
            ptr
        );
        if perm.contains(Perm::X) {
            self.jit.invalidate_range(va, len);
        }
        self.state.regions.write().push(Region {
            va,
            end,
            host_ptr: ptr,
            perm,
        });
        Ok(())
    }

    pub unsafe fn unmap_host(&mut self, va: u64, len: u64) -> Result<(), String> {
        let end = va
            .checked_add(len)
            .ok_or_else(|| format!("unmap_host overflow va={:#x} len={:#x}", va, len))?;
        self.state
            .regions
            .write()
            .retain(|r| r.end <= va || r.va >= end);
        Ok(())
    }

    pub fn write_bytes(&self, va: u64, bytes: &[u8]) -> Result<(), String> {
        let regions = self.state.regions.read();
        copy_to_regions(&regions, va, bytes, true)?;
        if !bytes.is_empty()
            && regions.iter().any(|r| {
                r.perm.contains(Perm::X)
                    && va < r.end
                    && va
                        .checked_add(bytes.len() as u64)
                        .is_some_and(|end| end > r.va)
            })
        {
            self.state
                .pending_invalidations
                .lock()
                .unwrap()
                .push((va, bytes.len() as u64));
        }
        Ok(())
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
        copy_from_regions(&regions, va, buf, false)
    }

    pub fn set_register(&mut self, reg: u32, val: u64) {
        if reg < 31 {
            self.state.ctx.x[reg as usize] = val;
        } else if reg == 31 {
            self.state.ctx.sp = val;
        }
    }

    pub fn get_register(&self, reg: u32) -> u64 {
        if reg < 31 {
            self.state.ctx.x[reg as usize]
        } else if reg == 31 {
            self.state.ctx.sp
        } else {
            0
        }
    }

    pub fn set_pc(&mut self, pc: u64) {
        self.state.ctx.pc = pc;
    }

    pub fn set_core_id(&mut self, core_id: u64) {
        self.state.ctx.core_id = core_id;
    }
    pub fn get_pc(&self) -> u64 {
        self.state.ctx.pc
    }
    pub fn set_sp(&mut self, sp: u64) {
        self.state.ctx.sp = sp;
    }
    pub fn get_sp(&self) -> u64 {
        self.state.ctx.sp
    }

    pub fn set_tpidrro_el0(&mut self, val: u64) {
        self.state.ctx.tpidrro_el0 = val;
    }
    pub fn get_tpidrro_el0(&self) -> u64 {
        self.state.ctx.tpidrro_el0
    }

    pub fn save_thread_context(&self) -> RustarmicThreadContext {
        RustarmicThreadContext {
            v: self.state.ctx.v,
            tpidr_el0: self.state.ctx.tpidr_el0,
            fpcr: self.state.ctx.fpcr,
            fpsr: self.state.ctx.fpsr,
            nzcv: self.state.ctx.nzcv,
        }
    }

    pub fn restore_thread_context(&mut self, context: &RustarmicThreadContext) {
        self.state.ctx.v = context.v;
        self.state.ctx.tpidr_el0 = context.tpidr_el0;
        self.state.ctx.fpcr = context.fpcr;
        self.state.ctx.fpsr = context.fpsr;
        self.state.ctx.nzcv = context.nzcv;
        self.state.ctx.exclusive_addr = 0;
        self.state.ctx.exclusive_size = 0;
    }

    pub fn reset_thread_context(&mut self) {
        self.state.ctx.v = [[0; 2]; 32];
        self.state.ctx.tpidr_el0 = 0;
        self.state.ctx.fpcr = 0;
        self.state.ctx.fpsr = 0;
        self.state.ctx.nzcv = 0;
        self.state.ctx.exclusive_addr = 0;
        self.state.ctx.exclusive_size = 0;
    }

    pub fn run_with_count(&mut self, max_insn: u64) -> (CpuEvent, u64) {
        let queued: Vec<(u64, u64)> =
            std::mem::take(&mut *self.state.pending_invalidations.lock().unwrap());
        if !queued.is_empty() {
            log::debug!(
                "rustarmic: applying {} queued SMC invalidations",
                queued.len()
            );
            for (va, len) in queued {
                self.jit.invalidate_range(va, len);
            }
        }

        if let Some(ev) = self.state.last_event.lock().unwrap().take() {
            return (ev, 0);
        }
        if self.state.control.halt.load(Ordering::Acquire) {
            self.state.ctx.should_halt = 1;
        }

        let regions: Vec<Region> = self.state.regions.read().iter().copied().collect();
        let mut mem = RegionMemory { regions };

        self.state
            .control
            .snapshot_seq
            .fetch_add(1, Ordering::Release);
        let exit = self
            .jit
            .run_bounded(&mut self.state.ctx, &mut mem, max_insn);

        self.state
            .control
            .peek_pc
            .store(self.state.ctx.pc, Ordering::Relaxed);
        self.state
            .control
            .peek_lr
            .store(self.state.ctx.x[30], Ordering::Relaxed);
        self.state
            .control
            .peek_sp
            .store(self.state.ctx.sp, Ordering::Relaxed);
        self.state
            .control
            .snapshot_seq
            .fetch_add(1, Ordering::Release);

        match exit {
            Ok(outcome) => {
                self.state.control.halt.store(false, Ordering::Release);
                let event = match outcome.stop {
                    StopReason::Unsupported(info) => {
                        log::warn!(
                            "rustarmic unsupported stop: pc={:#x} opcode={:#010x} class={} retired={}",
                            info.pc,
                            info.opcode,
                            info.decoded_class,
                            outcome.retired,
                        );
                        CpuEvent::Stalled
                    }
                    StopReason::MemoryFault(fault) => {
                        log::warn!(
                            "rustarmic memory fault: pc={:#x} address={:#x} size={} access={:?} cause={:?} retired={}",
                            fault.pc,
                            fault.address,
                            fault.size,
                            fault.access,
                            fault.cause,
                            outcome.retired,
                        );
                        CpuEvent::Exception(0x0E)
                    }
                    StopReason::BudgetExhausted => CpuEvent::Running,
                    StopReason::Halted => CpuEvent::Interrupted,
                    StopReason::Svc(imm) => CpuEvent::Svc(imm as u16),
                    StopReason::Brk(imm) => CpuEvent::Exception(0x100 | imm),
                    StopReason::Hvc(imm) => CpuEvent::Exception(0x200 | imm),
                    StopReason::Yield | StopReason::Wait => CpuEvent::Interrupted,
                };
                return (event, outcome.retired);
            }
            Err(e) => {
                match &e {
                    rustarmic::Error::Unsupported { pc, opcode } => {
                        let mut b = [0u8; 4];
                        let word = if self.read_bytes(*pc, &mut b).is_ok() {
                            u32::from_le_bytes(b)
                        } else {
                            *opcode
                        };
                        log::warn!(
                            "rustarmic Jit::run unsupported: pc={:#x} opcode={:#010x}",
                            pc,
                            word
                        );
                    }
                    rustarmic::Error::Decode { pc, opcode } => {
                        let mut b = [0u8; 4];
                        let word = if self.read_bytes(*pc, &mut b).is_ok() {
                            u32::from_le_bytes(b)
                        } else {
                            *opcode
                        };
                        log::warn!(
                            "rustarmic Jit::run decode-fail: pc={:#x} opcode={:#010x}",
                            pc,
                            word
                        );
                    }
                    other => log::warn!("rustarmic Jit::run error: {:?}", other),
                }
                (CpuEvent::Stalled, 0)
            }
        }
    }

    pub fn run(&mut self, max_insn: u64) -> CpuEvent {
        self.run_with_count(max_insn).0
    }

    pub fn step(&mut self) -> CpuEvent {
        self.run(1)
    }

    pub fn inject_svc(&mut self, imm: u16) {
        *self.state.last_event.lock().unwrap() = Some(CpuEvent::Svc(imm));
    }
}

struct RegionMemory {
    regions: Vec<Region>,
}

impl Memory for RegionMemory {
    fn fetch_inst(&mut self, addr: u64) -> Option<u32> {
        let mut buf = [0u8; 4];
        if copy_from_regions(&self.regions, addr, &mut buf, true).is_ok() {
            return Some(u32::from_le_bytes(buf));
        }
        log::warn!(
            "rustarmic fetch_inst miss at {:#x} ({} regions)",
            addr,
            self.regions.len()
        );
        for r in &self.regions {
            log::warn!(
                "  region {:#x}..{:#x} perm={} x={}",
                r.va,
                r.end,
                r.perm,
                r.perm.contains(Perm::X)
            );
        }
        None
    }
}

fn region_at(regions: &[Region], va: u64) -> Option<&Region> {
    regions.iter().find(|r| va >= r.va && va < r.end)
}

fn copy_from_regions(
    regions: &[Region],
    va: u64,
    out: &mut [u8],
    execute: bool,
) -> Result<(), String> {
    va.checked_add(out.len() as u64)
        .ok_or_else(|| format!("read: address overflow va={va:#x} len={}", out.len()))?;
    let mut cursor = va;
    let mut copied = 0usize;
    while copied < out.len() {
        let region =
            region_at(regions, cursor).ok_or_else(|| format!("read: unmapped va={cursor:#x}"))?;
        let perm = if execute { Perm::X } else { Perm::R };
        if !region.perm.contains(perm) {
            return Err(format!("read: permission fault va={cursor:#x}"));
        }
        let available = usize::try_from(region.end - cursor).unwrap_or(usize::MAX);
        let count = available.min(out.len() - copied);
        unsafe {
            std::ptr::copy_nonoverlapping(
                region.host_ptr.add((cursor - region.va) as usize),
                out.as_mut_ptr().add(copied),
                count,
            );
        }
        copied += count;
        cursor = cursor.saturating_add(count as u64);
    }
    Ok(())
}

fn copy_to_regions(regions: &[Region], va: u64, input: &[u8], write: bool) -> Result<(), String> {
    va.checked_add(input.len() as u64)
        .ok_or_else(|| format!("write: address overflow va={va:#x} len={}", input.len()))?;
    let mut cursor = va;
    let mut copied = 0usize;
    while copied < input.len() {
        let region =
            region_at(regions, cursor).ok_or_else(|| format!("write: unmapped va={cursor:#x}"))?;
        if write && !region.perm.contains(Perm::W) {
            return Err(format!("write: permission fault va={cursor:#x}"));
        }
        let available = usize::try_from(region.end - cursor).unwrap_or(usize::MAX);
        let count = available.min(input.len() - copied);
        unsafe {
            std::ptr::copy_nonoverlapping(
                input.as_ptr().add(copied),
                region.host_ptr.add((cursor - region.va) as usize),
                count,
            );
        }
        copied += count;
        cursor = cursor.saturating_add(count as u64);
    }
    Ok(())
}

unsafe extern "C" fn mem_read_hook(ctx_ptr: *mut CpuContext, addr: u64, size: u8) {
    let state = unsafe { &*(ctx_ptr as *mut State) };
    let regions = state.regions.read();
    let mut buf = [0u8; 16];
    if size <= 16 && copy_from_regions(&regions, addr, &mut buf[..size as usize], false).is_ok() {
        unsafe {
            let lo = u64::from_le_bytes(buf[..8].try_into().unwrap());
            let hi = u64::from_le_bytes(buf[8..].try_into().unwrap());
            (*ctx_ptr).io_value = [lo, hi];
        }
        return;
    }
    let cause = if region_at(&regions, addr).is_some() {
        1
    } else {
        0
    };
    handle_unmapped(state, ctx_ptr, addr, size, false, 0, cause);
    unsafe {
        (*ctx_ptr).io_value = [0, 0];
    }
}

unsafe extern "C" fn mem_write_hook(ctx_ptr: *mut CpuContext, addr: u64, size: u8) {
    let state = unsafe { &*(ctx_ptr as *mut State) };
    let io = unsafe { (*ctx_ptr).io_value };
    let mut buf = [0u8; 16];
    buf[..8].copy_from_slice(&io[0].to_le_bytes());
    buf[8..].copy_from_slice(&io[1].to_le_bytes());
    let value = io[0];
    if rustarmic_watch_overlaps(addr, size as u64) {
        use std::sync::atomic::{AtomicU64, Ordering};
        static HITS: AtomicU64 = AtomicU64::new(0);
        let n = HITS.fetch_add(1, Ordering::SeqCst);
        if n < 128 {
            let ctx = unsafe { &*ctx_ptr };
            log::warn!(
                "[watch-write] #{} addr={:#x} size={} val={:#x}/{:#x} pc={:#x} lr={:#x}",
                n,
                addr,
                size,
                io[0],
                io[1],
                ctx.pc,
                ctx.x[30]
            );
        }
    }
    let regions = state.regions.read();
    if size <= 16 && copy_to_regions(&regions, addr, &buf[..size as usize], true).is_ok() {
        if regions.iter().any(|r| {
            r.perm.contains(Perm::X)
                && addr < r.end
                && addr.checked_add(size as u64).is_some_and(|end| end > r.va)
        }) {
            state
                .pending_invalidations
                .lock()
                .unwrap()
                .push((addr, size as u64));
            unsafe {
                (*ctx_ptr).should_halt = 1;
            }
        }
        return;
    }
    let cause = if region_at(&regions, addr).is_some() {
        1
    } else {
        0
    };
    handle_unmapped(state, ctx_ptr, addr, size, true, value, cause);
}

fn rustarmic_watch_overlaps(addr: u64, size: u64) -> bool {
    let Some((lo, hi)) = rustarmic_watch_range() else {
        return false;
    };
    addr < hi && addr.saturating_add(size) > lo
}

fn rustarmic_watch_range() -> Option<(u64, u64)> {
    use std::sync::OnceLock;
    static RANGE: OnceLock<Option<(u64, u64)>> = OnceLock::new();
    *RANGE.get_or_init(|| {
        let spec = std::env::var("NEXIUM_WATCH_WRITE_CPU").ok()?;
        let (va, len) = spec.trim().split_once(':')?;
        let va = parse_watch_u64(va.trim())?;
        let len = parse_watch_u64(len.trim()).unwrap_or(0x80);
        (va != 0 && len != 0).then_some((va, va.saturating_add(len)))
    })
}

fn parse_watch_u64(s: &str) -> Option<u64> {
    let s = s.trim();
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16).ok()
    } else {
        s.parse::<u64>()
            .ok()
            .or_else(|| u64::from_str_radix(s, 16).ok())
    }
}

fn handle_unmapped(
    state: &State,
    ctx_ptr: *mut CpuContext,
    addr: u64,
    size: u8,
    is_write: bool,
    value: u64,
    cause: u8,
) {
    let is_null_zone = addr < 0x1000;
    let mut regs = [0u64; 31];
    let (live_pc, live_lr, live_sp);
    unsafe {
        let ctx = &*ctx_ptr;
        for i in 0..31 {
            regs[i] = ctx.x[i];
        }
        live_pc = ctx.pc;
        live_lr = ctx.x[30];
        live_sp = ctx.sp;
    }
    let snap = FaultSnapshot {
        pc: live_pc,
        lr: live_lr,
        sp: live_sp,
        addr,
        size: size as u32,
        is_write,
        value,
        regs,
    };
    *state.last_fault.lock().unwrap() = Some(snap);
    unsafe {
        let ctx = &mut *ctx_ptr;
        ctx.mem_fault = 1;
        ctx.mem_fault_access = u8::from(is_write);
        ctx.mem_fault_size = size;
        ctx.mem_fault_cause = cause;
        ctx.mem_fault_addr = addr;
        ctx.mem_fault_pc = ctx.pc;
    }
    let continue_null = state.continue_on_null.load(Ordering::Relaxed);
    if !is_null_zone || !continue_null {
        unsafe {
            (*ctx_ptr).should_halt = 1;
        }
        *state.last_event.lock().unwrap() = Some(CpuEvent::Exception(0x0E));
    }
    if is_null_zone {
        log::error!(
            "[null-deref] addr={:#x} size={} write={} val={:#x}",
            addr,
            size,
            is_write,
            value
        );
        if continue_null {
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
            log::warn!(
                "rustarmic: unmapped {:#x} size={} write={}",
                addr,
                size,
                is_write
            );
        } else if n & 0xFFFF == 0 {
            log::warn!("rustarmic: unmapped {:#x} size={} write={} (total {} so far — likely runaway loop)",
                addr, size, is_write, n + 1);
        }
        if n == RUNAWAY_THRESHOLD {
            let (pc, sp, regs) = unsafe {
                let ctx = &*ctx_ptr;
                let mut r = [0u64; 31];
                for i in 0..31 {
                    r[i] = ctx.x[i];
                }
                (ctx.pc, ctx.sp, r)
            };
            log::error!(
                "rustarmic: runaway-loop threshold ({} unmapped accesses) reached at addr={:#x}",
                RUNAWAY_THRESHOLD,
                addr
            );
            log::error!(
                "  PC = {:#018x}  SP = {:#018x}  LR(x30) = {:#018x}",
                pc,
                sp,
                regs[30]
            );
            log::error!("  x0..x7   {:#018x} {:#018x} {:#018x} {:#018x} {:#018x} {:#018x} {:#018x} {:#018x}",
                regs[0], regs[1], regs[2], regs[3], regs[4], regs[5], regs[6], regs[7]);
            log::error!("  x8..x15  {:#018x} {:#018x} {:#018x} {:#018x} {:#018x} {:#018x} {:#018x} {:#018x}",
                regs[8], regs[9], regs[10], regs[11], regs[12], regs[13], regs[14], regs[15]);
            log::error!("  x16..x23 {:#018x} {:#018x} {:#018x} {:#018x} {:#018x} {:#018x} {:#018x} {:#018x}",
                regs[16], regs[17], regs[18], regs[19], regs[20], regs[21], regs[22], regs[23]);
            log::error!(
                "  x24..x30 {:#018x} {:#018x} {:#018x} {:#018x} {:#018x} {:#018x} {:#018x}",
                regs[24],
                regs[25],
                regs[26],
                regs[27],
                regs[28],
                regs[29],
                regs[30]
            );
            log::error!("  → halting JIT cooperatively");

            unsafe {
                (*ctx_ptr).should_halt = 1;
            }
            *state.last_event.lock().unwrap() = Some(CpuEvent::Exception(0x0E));
        }
    }
}

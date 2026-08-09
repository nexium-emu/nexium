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
    watch_applied: Rc<Cell<Option<(u64, u64)>>>,
    watch_protected: Rc<Cell<bool>>,
    cache_profile_enabled: bool,
    cache_profile_last: std::time::Instant,
}

unsafe impl Send for DynarmicCpu {}
unsafe impl Sync for DynarmicCpu {}

impl DynarmicCpu {
    pub fn new() -> Result<Self, String> {
        let force_no_fastmem = env_flag("NEXIUM_DYNARMIC_NO_FASTMEM")
            || ((std::env::var("NEXIUM_WATCH_WRITE_CPU").is_ok()
                || std::env::var("NEXIUM_WATCH_WRITE_GPU").is_ok())
                && !env_flag("NEXIUM_WATCH_PAGE_PROTECT")
                && !env_flag("NEXIUM_WATCH_INLINE"));
        let emu: dynarmic_sys::Dynarmic<'static, ()> =
            match (nexium_memory::fastmem::base(), force_no_fastmem) {
                (Some(base), false) => {
                    log::info!("dynarmic: fastmem enabled, arena base={:p}", base);
                    dynarmic_sys::Dynarmic::new_fastmem(base.cast())
                }
                (Some(base), true) => {
                    log::info!(
                        "dynarmic: fastmem arena reserved at {:p}, CPU fastmem disabled for watch",
                        base
                    );
                    dynarmic_sys::Dynarmic::new()
                }
                (None, _) => dynarmic_sys::Dynarmic::new(),
            };
        log::info!(
            "dynarmic: guarded RSB/FastDispatch {} (DYNARMIC_FAST_PATHS=1 enables)",
            if dynarmic_sys::guarded_fast_paths_enabled() {
                "enabled"
            } else {
                "disabled"
            }
        );
        log::info!(
            "dynarmic: code-page fetch cache {} (DYNARMIC_CODE_PAGE_CACHE=0 disables)",
            if dynarmic_sys::code_page_cache_enabled() {
                "enabled"
            } else {
                "disabled"
            }
        );
        log::info!(
            "dynarmic: fastmem page-table mode {} (NEXIUM_DYNARMIC_SHARED_PAGE_TABLE=1 enables experimental sharing)",
            if dynarmic_sys::shared_fastmem_page_table_enabled() {
                "process-shared"
            } else {
                "private per JIT"
            }
        );
        if dynarmic_sys::unsafe_fastmem_enabled() {
            log::warn!(
                "dynarmic: UNSAFE 64-bit fastmem addressing enabled; invalid guest addresses may escape the reserved arena"
            );
        } else {
            log::info!("dynarmic: bounded fastmem address width enabled (safe default)");
        }

        let last_event = Rc::new(Cell::new(None::<CpuEvent>));
        let last_fault: Rc<RefCell<Option<FaultSnapshot>>> = Rc::new(RefCell::new(None));
        let continue_on_null = Rc::new(Cell::new(false));
        let null_skip_count = Rc::new(Cell::new(0u32));
        let watch_applied = Rc::new(Cell::new(None::<(u64, u64)>));
        let watch_protected = Rc::new(Cell::new(false));

        if let Some((cpu_va, len)) = initial_cpu_watch() {
            if nexium_memory::fastmem::watch_mark(cpu_va, len) {
                log::warn!("[watch-write] ARMED cpu_va={:#x} len={:#x}", cpu_va, len);
            } else {
                log::warn!(
                    "[watch-write] arm FAILED cpu_va={:#x} len={:#x}",
                    cpu_va,
                    len
                );
            }
        }

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
                    if !watch_target_overlaps(addr, size as u64)
                        || !watch_value_matches(size, value)
                    {
                        if nexium_memory::fastmem::watch_write_through(addr, size as usize, value) {
                            return true;
                        }
                    }
                    use std::sync::atomic::{AtomicU64, Ordering};
                    static HITS: AtomicU64 = AtomicU64::new(0);
                    let n = HITS.fetch_add(1, Ordering::SeqCst);
                    let pc = dyn_.reg_read_pc().unwrap_or(0);
                    let lr = dyn_.reg_read_lr().unwrap_or(0);
                    let limit = watch_write_limit();
                    if limit == 0 || n < limit {
                        if let Some(regs) = watch_write_reg_summary(dyn_) {
                            log::warn!(
                                "[watch-write] #{} addr={:#x} size={} val={:#x} pc={:#x} lr={:#x} {}",
                                n, addr, size, value, pc, lr, regs
                            );
                        } else {
                            log::warn!(
                                "[watch-write] #{} addr={:#x} size={} val={:#x} pc={:#x} lr={:#x}",
                                n, addr, size, value, pc, lr
                            );
                        }
                    }
                    let stop = limit != 0 && n + 1 >= limit;
                    if stop || (value != 0 && !watch_keep_after_hit()) {
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
            watch_applied,
            watch_protected,
            cache_profile_enabled: dynarmic_cache_profile_enabled(),
            cache_profile_last: std::time::Instant::now(),
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
        let result = self
            .emu
            .emu
            .mem_map_ptr(va, len as usize, perm_to_dyn(perm), ptr.cast())
            .map_err(|e| format!("map_host failed: {:?}", e));
        if result.is_ok() {
            self.emu.emu.invalidate_cache_range(va, len);
            if let Some((lo, hi)) = nexium_memory::fastmem::watch_range() {
                if va < hi && va.saturating_add(len) > lo {
                    self.watch_protected.set(false);
                }
            }
            self.apply_watch_range();
        }
        result
    }

    pub unsafe fn unmap_host(&mut self, va: u64, len: u64) -> Result<(), String> {
        let result = self
            .emu
            .emu
            .mem_unmap(va, len as usize)
            .map_err(|e| format!("unmap_host failed: {:?}", e));
        if result.is_ok() {
            self.emu.emu.invalidate_cache_range(va, len);
        }
        result
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

    pub fn alloc_thread_context(&self) -> dynarmic_sys::DynarmicContext {
        self.emu.emu.context_alloc()
    }

    pub fn save_thread_context(
        &self,
        context: &mut dynarmic_sys::DynarmicContext,
    ) -> Result<(), String> {
        self.emu
            .emu
            .context_save(context)
            .map_err(|e| format!("save context failed: {:?}", e))
    }

    pub fn restore_thread_context(
        &self,
        context: &dynarmic_sys::DynarmicContext,
    ) -> Result<(), String> {
        self.emu
            .emu
            .context_restore(context)
            .map_err(|e| format!("restore context failed: {:?}", e))
    }

    pub fn reset_thread_context(&self) -> Result<(), String> {
        let context = self.alloc_thread_context();
        self.restore_thread_context(&context)
    }

    pub fn run(&mut self, _max_insn: u64) -> CpuEvent {
        self.last_event.set(None);
        if nexium_memory::fastmem::watch_range().is_none() {
            if let Some((cpu_va, len)) = delayed_cpu_watch() {
                if nexium_memory::fastmem::watch_mark(cpu_va, len) {
                    log::warn!("[watch-write] ARMED cpu_va={:#x} len={:#x}", cpu_va, len);
                    self.watch_protected.set(false);
                }
            }
        }
        self.apply_watch_range();
        let pc = self.get_pc();
        log::trace!("dynarmic run: PC={:#x}", pc);
        let pc_until = pc_until_target().filter(|(target, label)| {
            pc_until_log_config(*target, label);
            pc_until_hits().load(std::sync::atomic::Ordering::Relaxed) < pc_until_max_hits()
        });
        let until = if let Some((target, _)) = pc_until.as_ref() {
            let target = *target;
            if pc == target {
                if _max_insn > 0 {
                    pc.saturating_add(_max_insn.saturating_mul(4))
                } else {
                    u64::MAX - 16
                }
            } else {
                target
            }
        } else if _max_insn > 0 {
            pc.saturating_add(_max_insn.saturating_mul(4))
        } else {
            u64::MAX - 16
        };
        let _ = if _max_insn == 0 {
            self.emu.emu.emu_start(pc, until)
        } else {
            self.emu.emu.emu_start_bounded(pc, until, _max_insn)
        };
        self.maybe_profile_cache();
        if let Some((target, label)) = pc_until {
            let after = self.get_pc();
            if after == target && pc != target {
                let hit = pc_until_hits().fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let x0 = self.get_register(0);
                let x1 = self.get_register(1);
                let x2 = self.get_register(2);
                let x3 = self.get_register(3);
                let x19 = self.get_register(19);
                let x20 = self.get_register(20);
                let x21 = self.get_register(21);
                let x22 = self.get_register(22);
                log::warn!(
                    "[pc-until] hit={} label={} pc={:#x} lr={:#x} sp={:#x} x0={:#x} x1={:#x} x2={:#x} x3={:#x} x19={:#x} x20={:#x} x21={:#x} x22={:#x}",
                    hit,
                    label,
                    after,
                    self.get_register(30),
                    self.get_sp(),
                    x0,
                    x1,
                    x2,
                    x3,
                    x19,
                    x20,
                    x21,
                    x22
                );
                pc_until_video_probe(&self.emu.emu, &label, x0);
                if pc_until_strings_enabled() {
                    let ptrs = pc_until_string_probe(
                        &self.emu.emu,
                        &[
                            ("x0", x0),
                            ("x1", x1),
                            ("x2", x2),
                            ("x3", x3),
                            ("x19", x19),
                            ("x20", x20),
                            ("x21", x21),
                            ("x22", x22),
                        ],
                    );
                    if !ptrs.is_empty() {
                        log::warn!("[pc-until-str] hit={} {}", hit, ptrs.join(" "));
                    }
                }
                if hit + 1 == pc_until_max_hits() {
                    log::warn!("[pc-until] cap reached; disabling until-probe");
                }
            }
        }
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

    pub fn run_with_count(&mut self, max_insn: u64) -> (CpuEvent, u64) {
        let event = self.run(max_insn);
        let retired = if max_insn == 0 || !dynarmic_exact_retired_enabled() {
            max_insn
        } else {
            let remaining = self.emu.emu.emu_ticks_remaining().min(max_insn);
            max_insn.saturating_sub(remaining)
        };
        (event, retired)
    }

    pub fn step(&mut self) -> CpuEvent {
        self.run(1)
    }

    pub fn inject_svc(&mut self, imm: u16) {
        self.last_event.set(Some(CpuEvent::Svc(imm)));
    }

    pub fn invalidate_range(&mut self, va: u64, len: u64) {
        self.emu.emu.invalidate_cache_range(va, len);
    }

    fn maybe_profile_cache(&mut self) {
        if !self.cache_profile_enabled {
            return;
        }

        let now = std::time::Instant::now();
        if !cache_profile_due(self.cache_profile_last, now) {
            return;
        }
        self.cache_profile_last = now;

        let thread = std::thread::current();
        log::warn!(
            "[dynarmic-cache] thread={} used_bytes={} capacity_bytes={} evacuations={}",
            thread.name().unwrap_or("<unnamed>"),
            self.emu.emu.get_cache_size(),
            self.emu.emu.get_cache_capacity(),
            self.emu.emu.get_cache_evacuation_count()
        );
    }

    fn apply_watch_range(&self) {
        let page_protect = env_flag("NEXIUM_WATCH_PAGE_PROTECT");
        let desired = if page_protect {
            nexium_memory::fastmem::watch_range()
        } else {
            nexium_memory::fastmem::watch_exact_range()
        };

        if self.watch_applied.get() == desired && (!page_protect || self.watch_protected.get()) {
            return;
        }

        if let Some((lo, hi)) = desired {
            if page_protect {
                if nexium_memory::fastmem::watch_reprotect() {
                    log::warn!(
                        "[watch-write] fastmem read-only armed va={:#x} len={:#x}",
                        lo,
                        hi - lo
                    );
                    self.watch_protected.set(true);
                } else {
                    log::warn!(
                        "[watch-write] fastmem read-only arm failed va={:#x} len={:#x}",
                        lo,
                        hi - lo
                    );
                    self.watch_protected.set(false);
                }
            } else {
                log::warn!(
                    "[watch-write] dynarmic watch callbacks unavailable; set NEXIUM_WATCH_PAGE_PROTECT=1 for va={:#x} len={:#x}",
                    lo,
                    hi - lo
                );
            }
        } else {
            self.watch_protected.set(false);
        }

        self.watch_applied.set(desired);
    }
}

fn pc_until_target() -> Option<(u64, String)> {
    let spec = std::env::var("NEXIUM_PC_UNTIL").ok()?;
    let item = spec.split(',').next()?.trim();
    if item.is_empty() {
        return None;
    }
    let (label, addr) = item
        .rsplit_once(':')
        .map(|(label, addr)| (label.trim().to_string(), addr.trim()))
        .unwrap_or_else(|| (item.to_string(), item));
    parse_pc_until_u64(addr).map(|addr| (addr, label))
}

fn pc_until_hits() -> &'static std::sync::atomic::AtomicU64 {
    static HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    &HITS
}

fn pc_until_max_hits() -> u64 {
    std::env::var("NEXIUM_PC_UNTIL_MAX")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(32)
}

fn pc_until_strings_enabled() -> bool {
    env_flag("NEXIUM_PC_UNTIL_STRINGS")
}

fn pc_until_video_probe(emu: &dynarmic_sys::Dynarmic<'static, ()>, label: &str, sink: u64) {
    if !env_flag("NEXIUM_PC_UNTIL_VIDEO") || !label.starts_with("video-") {
        return;
    }
    static HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let max_hits = std::env::var("NEXIUM_PC_UNTIL_VIDEO_MAX")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(4);
    let hit = HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if hit >= max_hits {
        return;
    }
    let mut bytes = [0u8; 0x40];
    if emu.mem_read(sink.saturating_add(0x40), &mut bytes).is_err() {
        log::warn!(
            "[pc-until-video] hit={} label={} sink={:#x} bytes40_80=<unreadable>",
            hit,
            label,
            sink
        );
        return;
    }
    let u32_at = |offset: usize| {
        u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ])
    };
    log::warn!(
        "[pc-until-video] hit={} label={} sink={:#x} output_format={} field54={} decoder_mode={} width={} height={} native={} bytes40_80={:02x?}",
        hit,
        label,
        sink,
        u32_at(0x08),
        u32_at(0x14),
        u32_at(0x18),
        u32_at(0x1c),
        u32_at(0x20),
        bytes[0x2f],
        bytes
    );
}

fn pc_until_string_probe(
    emu: &dynarmic_sys::Dynarmic<'static, ()>,
    regs: &[(&str, u64)],
) -> Vec<String> {
    regs.iter()
        .filter_map(|(name, ptr)| {
            pc_until_read_string(emu, *ptr).map(|s| format!("{}={:#x}:{}", name, ptr, s))
        })
        .collect()
}

fn pc_until_read_string(emu: &dynarmic_sys::Dynarmic<'static, ()>, ptr: u64) -> Option<String> {
    if ptr < 0x0800_0000 || ptr > 0x1_0000_0000_0000 {
        return None;
    }
    let mut buf = [0u8; 96];
    emu.mem_read(ptr, &mut buf).ok()?;
    let len = buf
        .iter()
        .position(|b| *b == 0)
        .unwrap_or(buf.len())
        .min(80);
    if len < 3 {
        return None;
    }
    let bytes = &buf[..len];
    let printable = bytes
        .iter()
        .filter(|b| matches!(**b, 0x20..=0x7e | b'\t'))
        .count();
    if printable * 4 < bytes.len() * 3 {
        return None;
    }
    let text = String::from_utf8_lossy(bytes).replace(['\r', '\n', '\t'], " ");
    if text.chars().any(|c| c.is_ascii_alphabetic()) {
        Some(text)
    } else {
        None
    }
}

fn pc_until_log_config(target: u64, label: &str) {
    static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    log::warn!(
        "[pc-until] configured label={} target={:#x} max_hits={}",
        label,
        target,
        pc_until_max_hits()
    );
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

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
        .unwrap_or(false)
}

fn dynarmic_exact_retired_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("NEXIUM_DYNARMIC_EXACT_RETIRED"))
}

fn dynarmic_cache_profile_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("NEXIUM_DYNARMIC_CACHE_PROFILE"))
}

fn cache_profile_due(last: std::time::Instant, now: std::time::Instant) -> bool {
    now.duration_since(last) >= std::time::Duration::from_secs(1)
}

fn watch_write_limit() -> u64 {
    std::env::var("NEXIUM_WATCH_WRITE_LIMIT")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(128)
}

fn watch_keep_after_hit() -> bool {
    env_flag("NEXIUM_WATCH_WRITE_KEEP")
}

fn watch_write_reg_summary<'a>(emu: &dynarmic_sys::Dynarmic<'a, ()>) -> Option<String> {
    if !env_flag("NEXIUM_WATCH_WRITE_REGS") {
        return None;
    }
    let mut parts = Vec::new();
    for reg in [
        0usize, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 19, 20, 21, 22, 23, 24, 25, 26,
        27, 28, 29, 30,
    ] {
        parts.push(format!("x{}={:#x}", reg, emu.reg_read(reg).unwrap_or(0)));
    }
    parts.push(format!("sp={:#x}", emu.reg_read_sp().unwrap_or(0)));
    Some(parts.join(" "))
}

fn initial_cpu_watch() -> Option<(u64, u64)> {
    if watch_arm_delay_ms().is_some() {
        return None;
    }
    parse_cpu_watch()
}

fn delayed_cpu_watch() -> Option<(u64, u64)> {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::OnceLock;
    use std::time::Instant;

    static START: OnceLock<Instant> = OnceLock::new();
    static ARMED: AtomicBool = AtomicBool::new(false);

    let delay = watch_arm_delay_ms()?;
    let start = START.get_or_init(Instant::now);
    if start.elapsed().as_millis() < delay as u128 {
        return None;
    }
    if ARMED.swap(true, Ordering::SeqCst) {
        return None;
    }
    parse_cpu_watch()
}

fn watch_arm_delay_ms() -> Option<u64> {
    std::env::var("NEXIUM_WATCH_ARM_DELAY_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
}

fn parse_pc_until_u64(s: &str) -> Option<u64> {
    let s = s.trim();
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16).ok()
    } else {
        s.parse::<u64>()
            .ok()
            .or_else(|| u64::from_str_radix(s, 16).ok())
    }
}

fn parse_cpu_watch() -> Option<(u64, u64)> {
    let spec = std::env::var("NEXIUM_WATCH_WRITE_CPU").ok()?;
    let (va, len) = spec.trim().split_once(':')?;
    let va = u64::from_str_radix(va.trim().trim_start_matches("0x"), 16).ok()?;
    let len = u64::from_str_radix(len.trim().trim_start_matches("0x"), 16)
        .ok()
        .unwrap_or(0x60);
    (va != 0 && len != 0).then_some((va, len))
}

fn watch_target_overlaps(addr: u64, size: u64) -> bool {
    let Some((lo, hi)) = nexium_memory::fastmem::watch_exact_range() else {
        return false;
    };
    addr < hi && addr.saturating_add(size) > lo
}

fn watch_value_matches(size: usize, value: u64) -> bool {
    if let Some(filter) = std::env::var("NEXIUM_WATCH_WRITE_VALUE")
        .ok()
        .and_then(|v| parse_pc_until_u64(v.trim()))
    {
        return watch_value_eq(size, value, filter);
    }
    if let Some(filter) = std::env::var("NEXIUM_WATCH_WRITE_VALUE_NE")
        .ok()
        .and_then(|v| parse_pc_until_u64(v.trim()))
    {
        return watch_value_ne(size, value, filter);
    }
    let min = std::env::var("NEXIUM_WATCH_WRITE_VALUE_MIN")
        .ok()
        .and_then(|v| parse_pc_until_u64(v.trim()));
    let max = std::env::var("NEXIUM_WATCH_WRITE_VALUE_MAX")
        .ok()
        .and_then(|v| parse_pc_until_u64(v.trim()));
    if min.is_some() || max.is_some() {
        return value >= min.unwrap_or(0) && value <= max.unwrap_or(u64::MAX);
    }
    true
}

fn watch_value_eq(size: usize, value: u64, filter: u64) -> bool {
    let mask = match size {
        1 => 0xff,
        2 => 0xffff,
        4 => 0xffff_ffff,
        _ => u64::MAX,
    };
    if size < 4 && filter > mask {
        return false;
    }
    if size == 8 && filter <= u32::MAX as u64 {
        let f = filter & 0xffff_ffff;
        return (value & 0xffff_ffff) == f || ((value >> 32) & 0xffff_ffff) == f;
    }
    (value & mask) == (filter & mask)
}

fn watch_value_ne(size: usize, value: u64, filter: u64) -> bool {
    let mask = match size {
        1 => 0xff,
        2 => 0xffff,
        4 => 0xffff_ffff,
        _ => u64::MAX,
    };
    if size < 4 && filter > mask {
        return true;
    }
    if size == 8 && filter <= u32::MAX as u64 {
        let f = filter & 0xffff_ffff;
        return (value & 0xffff_ffff) != f || ((value >> 32) & 0xffff_ffff) != f;
    }
    (value & mask) != (filter & mask)
}

#[cfg(test)]
mod tests {
    use super::cache_profile_due;
    use std::time::{Duration, Instant};

    #[test]
    fn cache_profile_rate_limit_is_one_second() {
        let last = Instant::now();
        assert!(!cache_profile_due(last, last));
        assert!(!cache_profile_due(last, last + Duration::from_millis(999)));
        assert!(cache_profile_due(last, last + Duration::from_secs(1)));
    }
}

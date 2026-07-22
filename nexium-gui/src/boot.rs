use nexium_core::boot::{BootConfig, BootContext};
use nexium_core::services::FrameOut;
use parking_lot::Mutex;
use std::path::Path;
use std::sync::mpsc::{self, Receiver};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};
use std::thread;

pub static EMU_ALIVE: AtomicBool = AtomicBool::new(false);

pub fn emu_alive() -> bool {
    EMU_ALIVE.load(Ordering::Acquire)
}

struct AliveGuard;
impl Drop for AliveGuard {
    fn drop(&mut self) {
        EMU_ALIVE.store(false, Ordering::Release);
    }
}

fn now_millis() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn sync_host_region_changes(
    cpu: &mut nexium_core::cpu::Cpu,
    address_space: &nexium_memory::AddressSpace,
    last_generation: &mut u64,
    core_id: usize,
) -> Result<usize, String> {
    let updates = address_space.host_region_changes_since(*last_generation);
    let count = updates.changes.len();
    for change in updates.changes {
        match change {
            nexium_memory::HostRegionChange::Upsert(region) => unsafe {
                let _ = cpu.unmap_host(region.base, region.size);
                cpu.map_host(region.base, region.size, region.perm, region.host_ptr)
                    .map_err(|error| {
                        format!(
                            "core{} host-map sync failed for {:#x} len={:#x}: {}",
                            core_id, region.base, region.size, error
                        )
                    })?;
            },
            nexium_memory::HostRegionChange::Remove { base, size } => unsafe {
                let _ = cpu.unmap_host(base, size);
            },
        }
    }
    if count != 0 {
        log::trace!(
            "[core{}] host-map sync generation {} -> {} changes={}",
            core_id,
            *last_generation,
            updates.generation,
            count
        );
    }
    *last_generation = updates.generation;
    Ok(count)
}

struct CpuPollWatch {
    va: u64,
    len: usize,
    last: Option<Vec<u8>>,
    hits: u32,
    max_hits: u32,
}

impl CpuPollWatch {
    fn from_env() -> Option<Self> {
        let spec = std::env::var("NEXIUM_CPU_POLL_WATCH").ok()?;
        let (va, len) = spec.trim().split_once(':')?;
        let va = parse_watch_u64(va)?;
        let len = parse_watch_u64(len).unwrap_or(0x80) as usize;
        if va == 0 || len == 0 {
            return None;
        }
        let max_hits = std::env::var("NEXIUM_CPU_POLL_WATCH_MAX")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(8);
        log::warn!(
            "[cpu-poll-watch] configured va={:#x} len={:#x} max_hits={}",
            va,
            len,
            max_hits
        );
        Some(Self {
            va,
            len,
            last: None,
            hits: 0,
            max_hits,
        })
    }

    fn check(
        &mut self,
        kernel: &nexium_core::kernel::Kernel,
        cpu: &nexium_core::cpu::Cpu,
        event: nexium_core::cpu::CpuEvent,
        cycles: u64,
        svcs: u32,
    ) -> bool {
        let mut cur = vec![0u8; self.len];
        if kernel.address_space.read(self.va, &mut cur).is_err() {
            return true;
        }
        let Some(last) = self.last.as_ref() else {
            self.last = Some(cur);
            log::warn!("[cpu-poll-watch] baseline va={:#x}", self.va);
            return true;
        };
        if last == &cur {
            return true;
        }
        let changed = changed_words(last, &cur);
        let floats = float_preview_local(&cur);
        let hex = hex_preview_local(&cur);
        let handle = kernel.threads.current_handle();
        let pc = cpu.get_pc();
        let lr = cpu.get_register(30);
        let sp = cpu.get_sp();
        log::warn!(
            "[cpu-poll-watch] hit={} va={:#x} changed={} handle={:?} event={:?} cycles={} svcs={} pc={:#x} lr={:#x} sp={:#x} x0={:#x} x1={:#x} x2={:#x} x3={:#x} x19={:#x} x20={:#x} x21={:#x} x22={:#x} x29={:#x}",
            self.hits,
            self.va,
            changed,
            handle,
            event,
            cycles,
            svcs,
            pc,
            lr,
            sp,
            cpu.get_register(0),
            cpu.get_register(1),
            cpu.get_register(2),
            cpu.get_register(3),
            cpu.get_register(19),
            cpu.get_register(20),
            cpu.get_register(21),
            cpu.get_register(22),
            cpu.get_register(29)
        );
        log::warn!("[cpu-poll-watch] floats=[{}]", floats);
        log::warn!("[cpu-poll-watch] bytes={}", hex);
        self.last = Some(cur);
        self.hits += 1;
        if self.max_hits != 0 && self.hits >= self.max_hits {
            log::warn!("[cpu-poll-watch] disarmed");
            return false;
        }
        true
    }
}

struct PcTraceRange {
    label: String,
    start: u64,
    end: u64,
    hits: u32,
}

struct PcTrace {
    ranges: Vec<PcTraceRange>,
    max_hits: u32,
}

impl PcTrace {
    fn from_env() -> Option<Self> {
        let spec = std::env::var("NEXIUM_PC_TRACE").ok()?;
        let mut ranges = Vec::new();
        for item in spec.split(',') {
            let item = item.trim();
            if item.is_empty() {
                continue;
            }
            let (label, range) = item
                .rsplit_once(':')
                .map(|(label, range)| (label.trim().to_string(), range.trim()))
                .unwrap_or_else(|| (item.to_string(), item));
            let Some((start, end)) = parse_pc_trace_range(range) else {
                log::warn!("[pc-trace] ignored invalid range {}", item);
                continue;
            };
            if start >= end {
                log::warn!("[pc-trace] ignored empty range {}", item);
                continue;
            }
            ranges.push(PcTraceRange {
                label,
                start,
                end,
                hits: 0,
            });
        }
        if ranges.is_empty() {
            return None;
        }
        let max_hits = std::env::var("NEXIUM_PC_TRACE_MAX")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(8);
        for r in &ranges {
            log::warn!(
                "[pc-trace] configured {} {:#x}-{:#x} max_hits={}",
                r.label,
                r.start,
                r.end,
                max_hits
            );
        }
        Some(Self { ranges, max_hits })
    }

    fn check(
        &mut self,
        core: usize,
        kernel: &nexium_core::kernel::Kernel,
        cpu: &nexium_core::cpu::Cpu,
        event: nexium_core::cpu::CpuEvent,
        cycles: u64,
        svcs: u32,
    ) {
        let pc = cpu.get_pc();
        for r in &mut self.ranges {
            if pc < r.start || pc >= r.end {
                continue;
            }
            if self.max_hits != 0 && r.hits >= self.max_hits {
                continue;
            }
            let hit = r.hits;
            r.hits = r.hits.saturating_add(1);
            log::warn!(
                "[pc-trace] core={} hit={} label={} pc={:#x} off={:#x} lr={:#x} sp={:#x} handle={:?} event={:?} cycles={} svcs={} x0={:#x} x1={:#x} x2={:#x} x3={:#x} x19={:#x} x20={:#x} x21={:#x} x22={:#x} x29={:#x}",
                core,
                hit,
                r.label,
                pc,
                pc.saturating_sub(r.start),
                cpu.get_register(30),
                cpu.get_sp(),
                kernel.threads.current_handle(),
                event,
                cycles,
                svcs,
                cpu.get_register(0),
                cpu.get_register(1),
                cpu.get_register(2),
                cpu.get_register(3),
                cpu.get_register(19),
                cpu.get_register(20),
                cpu.get_register(21),
                cpu.get_register(22),
                cpu.get_register(29)
            );
        }
    }
}

fn parse_pc_trace_range(s: &str) -> Option<(u64, u64)> {
    if let Some((start, len)) = s.split_once('+') {
        let start = parse_watch_u64(start)?;
        let len = parse_watch_u64(len)?;
        return Some((start, start.saturating_add(len)));
    }
    if let Some((start, end)) = s.split_once('-') {
        return Some((parse_watch_u64(start)?, parse_watch_u64(end)?));
    }
    let start = parse_watch_u64(s)?;
    Some((start, start.saturating_add(4)))
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

fn changed_words(old: &[u8], new: &[u8]) -> String {
    let mut out = Vec::new();
    for (i, (a, b)) in old.chunks(4).zip(new.chunks(4)).enumerate() {
        if a != b {
            out.push(format!("{:#x}", i * 4));
        }
        if out.len() >= 24 {
            break;
        }
    }
    out.join(",")
}

fn hex_preview_local(buf: &[u8]) -> String {
    buf.iter()
        .take(96)
        .map(|b| format!("{:02x}", b))
        .collect::<Vec<_>>()
        .join(" ")
}

fn float_preview_local(buf: &[u8]) -> String {
    buf.chunks_exact(4)
        .take(24)
        .map(|c| format!("{:.3}", f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
        .collect::<Vec<_>>()
        .join(",")
}

fn decode_a64_brief(insn: u32) -> String {
    let top8 = (insn >> 24) & 0xFF;
    let size = (insn >> 30) & 0b11;
    let rt = insn & 0x1F;
    let rn = (insn >> 5) & 0x1F;
    let opc = (insn >> 22) & 0b11;

    if (insn & 0x3B00_0000) == 0x3900_0000 {
        let imm12 = ((insn >> 10) & 0xFFF) as u64;
        let scale = match size {
            0 => 1u64,
            1 => 2,
            2 => 4,
            3 => 8,
            _ => 1,
        };
        let off = imm12 * scale;
        let (mnemonic, width) = match (size, opc) {
            (0, 0) => ("STRB", "W"),
            (0, 1) => ("LDRB", "W"),
            (1, 0) => ("STRH", "W"),
            (1, 1) => ("LDRH", "W"),
            (2, 0) => ("STR", "W"),
            (2, 1) => ("LDR", "W"),
            (3, 0) => ("STR", "X"),
            (3, 1) => ("LDR", "X"),
            _ => ("?LSI", "?"),
        };
        return format!("{} {}{}, [X{}, #{}]", mnemonic, width, rt, rn, off);
    }

    if (insn & 0x3B20_0C00) == 0x3800_0000 {
        let imm9 = ((insn >> 12) & 0x1FF) as i32;
        let imm9 = if imm9 & 0x100 != 0 {
            imm9 | !0x1FF
        } else {
            imm9
        };
        let (mnemonic, width) = match (size, opc) {
            (0, 0) => ("STURB", "W"),
            (0, 1) => ("LDURB", "W"),
            (1, 0) => ("STURH", "W"),
            (1, 1) => ("LDURH", "W"),
            (2, 0) => ("STUR", "W"),
            (2, 1) => ("LDUR", "W"),
            (3, 0) => ("STUR", "X"),
            (3, 1) => ("LDUR", "X"),
            _ => ("?LSU", "?"),
        };
        return format!("{} {}{}, [X{}, #{}]", mnemonic, width, rt, rn, imm9);
    }

    if top8 == 0xA9 || top8 == 0x29 || top8 == 0x69 {
        let rt2 = (insn >> 10) & 0x1F;
        let imm7 = ((insn >> 15) & 0x7F) as i32;
        let imm7 = if imm7 & 0x40 != 0 { imm7 | !0x7F } else { imm7 };
        let is_64 = (insn >> 31) & 1 == 1;
        let scale = if is_64 { 8 } else { 4 };
        let off = imm7 * scale;
        let width = if is_64 { "X" } else { "W" };
        let mnemonic = if (insn >> 22) & 1 == 1 { "LDP" } else { "STP" };
        return format!(
            "{} {}{}, {}{}, [X{}, #{}]",
            mnemonic, width, rt, width, rt2, rn, off
        );
    }

    if (insn & 0xFE1F_FC00) == 0xD61F_0000 {
        let kind = match (insn >> 21) & 0x3 {
            0 => "BR",
            1 => "BLR",
            2 => "RET",
            _ => "?BR",
        };
        return format!("{} X{}", kind, rn);
    }
    format!("?? raw={:08x}", insn)
}

pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

impl From<FrameOut> for Frame {
    fn from(f: FrameOut) -> Self {
        Self {
            width: f.width,
            height: f.height,
            pixels: f.pixels,
        }
    }
}

#[derive(Clone, Default)]
pub struct CpuSnapshot {
    pub pc: u64,
    pub sp: u64,
    pub tpidrro_el0: u64,
    pub x: [u64; 31],
    pub instruction_bytes: Vec<u8>,
    pub mem_address: u64,
    pub mem_data: Vec<u8>,
    pub mem_request_address: u64,
}

#[derive(Clone, Default)]
pub struct EmuStats {
    pub svc_count: u64,
    pub cycle_count: u64,
}

pub struct EmulationHandle {
    pub stop_flag: Arc<AtomicBool>,
    pub pause_flag: Arc<AtomicBool>,
    pub frame_rx: Receiver<Frame>,
    pub thread_handle: Option<thread::JoinHandle<Result<(), String>>>,
    pub cpu_snapshot: Arc<Mutex<CpuSnapshot>>,
    pub mem_request: Arc<Mutex<u64>>,
    pub stats: Arc<Mutex<EmuStats>>,
}

impl EmulationHandle {
    pub fn new(
        nro_path: &str,
        cpu_backend: nexium_cpu::CpuBackendKind,
        repaint_ctx: Option<eframe::egui::Context>,
    ) -> Result<Self, String> {
        let nro_path = nro_path.to_string();

        let wait_start = std::time::Instant::now();
        while EMU_ALIVE.load(Ordering::Acquire) {
            if wait_start.elapsed().as_secs_f32() > 6.0 {
                return Err("previous game is still shutting down".into());
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        EMU_ALIVE.store(true, Ordering::Release);

        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop_flag_clone = Arc::clone(&stop_flag);
        let pause_flag = Arc::new(AtomicBool::new(false));
        let pause_flag_clone = Arc::clone(&pause_flag);
        let (frame_tx, frame_rx) = mpsc::sync_channel::<Frame>(8);
        let cpu_snapshot = Arc::new(Mutex::new(CpuSnapshot::default()));
        let cpu_snapshot_clone = Arc::clone(&cpu_snapshot);
        let mem_request = Arc::new(Mutex::new(0u64));
        let mem_request_clone = Arc::clone(&mem_request);
        let stats = Arc::new(Mutex::new(EmuStats::default()));
        let stats_clone = Arc::clone(&stats);

        let thread_handle = thread::spawn(move || {
            let _alive_guard = AliveGuard;
            let initial_loader_path = nro_path.clone();
            let initial_loader_filename = Path::new(&initial_loader_path)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("hbmenu.nro")
                .to_string();
            let initial_loader_app = nexium_common::paths::app_name_from_nro(&initial_loader_path);
            let initial_loader_argv = format!(
                "sdmc:/switch/{}/{}",
                initial_loader_app, initial_loader_filename
            );
            let mut cur_nro_path = nro_path;

            let mut chained_argv: Option<String> = None;
            'launcher: loop {
                log::info!("Booting NRO: {}", cur_nro_path);

                if let Some(stem) = Path::new(&cur_nro_path)
                    .file_stem()
                    .and_then(|s| s.to_str())
                {
                    nexium_common::title::set_title_key(stem);
                }

                if !Path::new(&cur_nro_path).exists() {
                    return Err(format!("NRO file not found: {}", cur_nro_path));
                }

                let mut config = BootConfig::new(&cur_nro_path);
                config.loader_path = Some(initial_loader_argv.clone());
                config.argv_override = chained_argv.take();
                config.cpu_backend = cpu_backend;
                let mut boot_ctx = BootContext::new(config)?;

                let mut cpu = boot_ctx
                    .cpu
                    .take()
                    .expect("BootContext CPU not initialized");
                cpu.set_continue_on_null(true);
                log::info!("dynarmic: continue_on_null=ON (will absorb null-zone reads as 0 to keep going)");
                let halt = Some(cpu.halt_handle());
                let _cpu_guard = nexium_kernel::kernel::cpu_local::set_current_cpu(&mut cpu, 0);
                use nexium_kernel::kernel::cpu_local::{cpu_mut, cpu_ref};
                let mut last_map_gen0 = boot_ctx.address_space.generation();

                let aux_core_stop = Arc::new(AtomicBool::new(false));
                let mut aux_core_handles = Vec::new();
                let active_cores = if std::env::var("NEXIUM_SINGLECORE").is_ok() {
                    1usize
                } else {
                    std::env::var("NEXIUM_CPU_CORES")
                        .ok()
                        .and_then(|v| v.parse::<usize>().ok())
                        .unwrap_or(nexium_core::kernel::threads::NUM_CORES)
                        .clamp(1, nexium_core::kernel::threads::NUM_CORES)
                };
                if active_cores > 1 {
                    for core_id in 1..active_cores {
                        let kernel_aux = Arc::clone(&boot_ctx.kernel);
                        let stop_aux = Arc::clone(&aux_core_stop);
                        let pause_aux = Arc::clone(&pause_flag_clone);
                        let backend_aux = cpu_backend;
                        let addr_aux = Arc::clone(&boot_ctx.address_space);
                        if let Ok(handle) = thread::Builder::new()
                            .name(format!("nexium-core{}", core_id))
                            .spawn(move || {
                                let mut last_map_gen = addr_aux.generation();
                                let mut cpu_aux = match kernel_aux.lock().init_cpu(backend_aux) {
                                    Ok(c) => c,
                                    Err(e) => {
                                        log::error!("[core{}] init_cpu failed: {}", core_id, e);
                                        return;
                                    }
                                };
                                cpu_aux.set_continue_on_null(true);
                                let _g_aux = nexium_kernel::kernel::cpu_local::set_current_cpu(
                                    &mut cpu_aux,
                                    core_id,
                                );
                                let mut pc_trace = PcTrace::from_env();
                                let mut aux_cycles = 0u64;
                                let mut aux_svcs = 0u32;
                                let spin_yield_n: u32 = std::env::var("NEXIUM_SPIN_YIELD")
                                    .ok()
                                    .and_then(|v| v.parse::<u32>().ok())
                                    .unwrap_or(4);
                                let mut slice_iters: u32 = 0;
                                log::info!(
                                    "[core{}] started spin_yield_n={}",
                                    core_id,
                                    spin_yield_n
                                );
                                while !stop_aux.load(Ordering::Relaxed) {
                                    if pause_aux.load(Ordering::Relaxed) {
                                        std::thread::sleep(std::time::Duration::from_micros(500));
                                        continue;
                                    }
                                    let has = {
                                        let mut k = kernel_aux.lock();
                                        k.threads.wake_due_sleepers(std::time::Instant::now());
                                        k.ensure_thread_loaded().is_some()
                                    };
                                    if !has {
                                        std::thread::sleep(std::time::Duration::from_micros(200));
                                        continue;
                                    }
                                    let gen = addr_aux.generation();
                                    if gen != last_map_gen {
                                        if let Err(error) = sync_host_region_changes(
                                            cpu_mut().unwrap(),
                                            &addr_aux,
                                            &mut last_map_gen,
                                            core_id,
                                        ) {
                                            log::error!("{}", error);
                                        }
                                    }
                                    let event = cpu_mut().unwrap().run(200_000);
                                    aux_cycles = aux_cycles.saturating_add(200_000);
                                    if pc_trace.is_some()
                                        || matches!(event, nexium_core::cpu::CpuEvent::Svc(_))
                                    {
                                        let mut k = kernel_aux.lock();
                                        k.threads.save_current_ctx(cpu_ref().unwrap());
                                        if let Some(trace) = pc_trace.as_mut() {
                                            trace.check(
                                                core_id,
                                                &k,
                                                cpu_ref().unwrap(),
                                                event,
                                                aux_cycles,
                                                aux_svcs,
                                            );
                                        }
                                        if let nexium_core::cpu::CpuEvent::Svc(imm) = event {
                                            aux_svcs = aux_svcs.saturating_add(1);
                                            let result = k.dispatch_svc(imm);
                                            k.tick_audio_renderers();
                                            if imm != 0x7f {
                                                cpu_mut().unwrap().set_register(0, result as u64);
                                            }
                                            let pace_until = k.present_pace_until.take();
                                            let should_yield =
                                                k.yield_after_svc || pace_until.is_some();
                                            k.yield_after_svc = false;
                                            if should_yield {
                                                let state = match pace_until {
                                                    Some(wake_at)
                                                        if wake_at > std::time::Instant::now() =>
                                                    {
                                                        nexium_core::kernel::threads::ThreadState::Sleeping {
                                                            wake_at,
                                                        }
                                                    }
                                                    _ => nexium_core::kernel::threads::ThreadState::Ready,
                                                };
                                                k.threads
                                                    .yield_with_state(cpu_ref().unwrap(), state);
                                            }
                                        }
                                        drop(k);
                                    }
                                    if spin_yield_n != 0 {
                                        slice_iters = slice_iters.saturating_add(1);
                                        if slice_iters >= spin_yield_n {
                                            slice_iters = 0;
                                            let mut k = kernel_aux.lock();
                                            if k.threads.has_ready_for_core(core_id as i32) {
                                                k.threads.yield_with_state(
                                                    cpu_ref().unwrap(),
                                                    nexium_core::kernel::threads::ThreadState::Ready,
                                                );
                                            }
                                        }
                                    }
                                }
                                log::info!("[core{}] stopped", core_id);
                            })
                        {
                            aux_core_handles.push(handle);
                        }
                    }
                }

                let last_svc_ms = Arc::new(AtomicU64::new(0));
                let watchdog_stop = Arc::new(AtomicBool::new(false));
                let watchdog_halts = Arc::new(AtomicU64::new(0));
                if let Some(halt) = halt {
                    let last_svc_ms_wd = Arc::clone(&last_svc_ms);
                    let watchdog_stop_wd = Arc::clone(&watchdog_stop);
                    let watchdog_halts_wd = Arc::clone(&watchdog_halts);
                    let _ = thread::Builder::new()
                    .name("nexium-cpu-watchdog".into())
                    .spawn(move || {
                        let mut peek_counter: u64 = 0;
                        let mut prev_pc: u64 = 0;
                        let mut same_pc_streak: u32 = 0;
                        while !watchdog_stop_wd.load(Ordering::Relaxed) {
                            thread::sleep(std::time::Duration::from_millis(50));
                            let last = last_svc_ms_wd.load(Ordering::Relaxed);
                            if last == 0 {
                                continue;
                            }
                            if now_millis().saturating_sub(last) > 250 {
                                let (pc, lr, sp) = halt.peek_pc_lr_sp();
                                if pc == prev_pc {
                                    same_pc_streak = same_pc_streak.saturating_add(1);
                                } else {
                                    same_pc_streak = 0;
                                    prev_pc = pc;
                                }
                                if same_pc_streak < 3 {
                                    continue;
                                }
                                peek_counter += 1;
                                if peek_counter % 25 == 1 {
                                    log::warn!(
                                        "[watchdog-peek #{}] pc={:#x} lr={:#x} sp={:#x} same_pc_streak={}",
                                        peek_counter, pc, lr, sp, same_pc_streak
                                    );
                                }
                                if peek_counter == 1 {
                                    log::warn!("[spin-dump] {}", halt.peek_dump());
                                }
                                halt.halt();
                                watchdog_halts_wd.fetch_add(1, Ordering::Relaxed);
                                last_svc_ms_wd.store(now_millis(), Ordering::Relaxed);
                                same_pc_streak = 0;
                            }
                        }
                    });
                }
                struct WatchdogGuard(Arc<AtomicBool>);
                impl Drop for WatchdogGuard {
                    fn drop(&mut self) {
                        self.0.store(true, Ordering::Relaxed);
                    }
                }
                let _wd_guard = WatchdogGuard(Arc::clone(&watchdog_stop));

                let vsync_stop = Arc::new(AtomicBool::new(false));
                let vsync_handle = {
                    let kernel_v = Arc::clone(&boot_ctx.kernel);
                    let stop_v = Arc::clone(&vsync_stop);
                    thread::Builder::new()
                        .name("nexium-vsync".into())
                        .spawn(move || {
                            const PERIOD: std::time::Duration =
                                std::time::Duration::from_nanos(16_666_667);
                            let mut next = std::time::Instant::now() + PERIOD;
                            while !stop_v.load(Ordering::Relaxed) {
                                let now = std::time::Instant::now();
                                if now < next {
                                    let rem = next - now;
                                    if rem > std::time::Duration::from_millis(2) {
                                        thread::sleep(rem - std::time::Duration::from_millis(1));
                                    } else {
                                        while std::time::Instant::now() < next {
                                            std::hint::spin_loop();
                                        }
                                    }
                                    continue;
                                }
                                kernel_v.lock().signal_vsync();
                                next += PERIOD;
                                let after = std::time::Instant::now();
                                if after > next + PERIOD {
                                    next = after + PERIOD;
                                }
                            }
                        })
                        .ok()
                };
                struct VsyncGuard(Arc<AtomicBool>, Option<thread::JoinHandle<()>>);
                impl Drop for VsyncGuard {
                    fn drop(&mut self) {
                        self.0.store(true, Ordering::Relaxed);
                        if let Some(h) = self.1.take() {
                            let _ = h.join();
                        }
                    }
                }
                let _vsync_guard = VsyncGuard(Arc::clone(&vsync_stop), vsync_handle);

                log::info!("Starting emulation loop [BUILD: heartbeat-v2-gpu-diag]");
                let max_cycles = u64::MAX;
                let mut cycle_count = 0u64;
                let mut svc_count = 0u32;
                let mut stuck_log_counter: u64 = 0;
                let mut pc_check_count = 0u32;
                let mut stuck_pc: Option<u64> = None;
                let mut stuck_count = 0u32;
                let mut last_svc_cycle = 0u64;
                let no_svc_exit = std::env::var("NEXIUM_NO_SVC_EXIT").is_ok();
                let mut last_no_svc_warn = 0u64;
                let mut no_svc_in_spin = 0u32;
                let mut last_heartbeat = std::time::Instant::now();
                let mut last_heartbeat_svc = 0u32;
                let mut last_heartbeat_cycles = 0u64;
                let mut last_pipeline_stats = boot_ctx.kernel.lock().nvdrv.stats.snapshot();
                let mut last_render_progress = std::time::Instant::now();
                let mut seen_render_progress = last_pipeline_stats.gpfifo_submits > 0
                    || last_pipeline_stats.gpfifo_entries > 0
                    || last_pipeline_stats.methods_dispatched > 0
                    || last_pipeline_stats.maxwell3d_draws > 0
                    || last_pipeline_stats.frames_submitted > 0;
                let mut last_sync_snapshot = std::time::Instant::now()
                    .checked_sub(std::time::Duration::from_secs(10))
                    .unwrap_or_else(std::time::Instant::now);
                let mut cpu_poll_watch = CpuPollWatch::from_env();
                let mut pc_trace = PcTrace::from_env();

                let mut loop_iter: u64 = 0;
                let mut last_loop_log = std::time::Instant::now();
                let mut last_halts: u64 = 0;
                let mut preempt_count: u64 = 0;
                loop {
                    loop_iter += 1;
                    let mut guard = boot_ctx.kernel.lock();
                    if last_loop_log.elapsed() >= std::time::Duration::from_secs(1) {
                        let cur = guard.threads.current_handle();
                        let pc_now = cpu_ref().map(|c| c.get_pc()).unwrap_or(0);
                        let halts = watchdog_halts.load(Ordering::Relaxed);
                        log::warn!(
                            "[loop-tick] iter={} svc={} cyc={} cur={:?} pc={:#x} halts={}",
                            loop_iter,
                            svc_count,
                            cycle_count,
                            cur,
                            pc_now,
                            halts
                        );
                        last_loop_log = std::time::Instant::now();
                    }

                    if stop_flag_clone.load(Ordering::Relaxed) {
                        log::info!("Stopping emulation");
                        break;
                    }

                    if pause_flag_clone.load(Ordering::Relaxed) {
                        drop(guard);
                        std::thread::sleep(std::time::Duration::from_millis(12));
                        continue;
                    }

                    if let Some(va) = nexium_memory::fastmem::take_guest_probe_event() {
                        log::warn!("[guest-probe-snapshot] va={:#x}", va);
                        guard.log_thread_snapshot(&format!("guest-probe-{:#x}", va));
                    }

                    {
                        let elapsed = last_heartbeat.elapsed();
                        if elapsed >= std::time::Duration::from_secs(1) {
                            let secs = elapsed.as_secs_f64();
                            let svc_rate = (svc_count - last_heartbeat_svc) as f64 / secs;
                            let cycle_rate = (cycle_count - last_heartbeat_cycles) as f64 / secs;
                            let cur = guard.threads.current_handle();
                            let nthreads = guard.threads.threads.len();
                            let nready = guard.threads.ready.len();
                            let halts = watchdog_halts.load(Ordering::Relaxed);
                            let null_skips = cpu_ref().map(|c| c.null_skip_count()).unwrap_or(0);
                            let (cur_pc, lr, sp, x0, x1, x19, x20, x21, x22) =
                                if let Some(c) = cpu_ref() {
                                    (
                                        c.get_pc(),
                                        c.get_register(30),
                                        c.get_sp(),
                                        c.get_register(0),
                                        c.get_register(1),
                                        c.get_register(19),
                                        c.get_register(20),
                                        c.get_register(21),
                                        c.get_register(22),
                                    )
                                } else {
                                    (0, 0, 0, 0, 0, 0, 0, 0, 0)
                                };
                            if null_skips > 0 {
                                log::info!("[heartbeat] null_skips_total={}", null_skips);
                            }
                            let mut x20_bytes = [0u8; 32];
                            let x20_read = guard.address_space.read(x20, &mut x20_bytes).is_ok();
                            log::info!(
                            "[heartbeat] svc={} cyc={} svc/s={:.0} cyc/s={:.0} halts={} pc={:#x} lr={:#x} sp={:#x} x0={:#x} x1={:#x} x19={:#x} x20={:#x} x21={:#x} x22={:#x}",
                            svc_count, cycle_count, svc_rate, cycle_rate, halts, cur_pc, lr, sp, x0, x1, x19, x20, x21, x22
                        );
                            if x20_read && x20 != 0 {
                                log::info!(
                                "[heartbeat] *x20[0..32] = {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} | {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} | {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} | {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x}",
                                x20_bytes[0], x20_bytes[1], x20_bytes[2], x20_bytes[3], x20_bytes[4], x20_bytes[5], x20_bytes[6], x20_bytes[7],
                                x20_bytes[8], x20_bytes[9], x20_bytes[10], x20_bytes[11], x20_bytes[12], x20_bytes[13], x20_bytes[14], x20_bytes[15],
                                x20_bytes[16], x20_bytes[17], x20_bytes[18], x20_bytes[19], x20_bytes[20], x20_bytes[21], x20_bytes[22], x20_bytes[23],
                                x20_bytes[24], x20_bytes[25], x20_bytes[26], x20_bytes[27], x20_bytes[28], x20_bytes[29], x20_bytes[30], x20_bytes[31],
                            );
                            }
                            if std::env::var_os("NEXIUM_HEARTBEAT_THREAD_SNAPSHOT").is_some()
                                && last_sync_snapshot.elapsed() >= std::time::Duration::from_secs(5)
                            {
                                guard.log_thread_snapshot("heartbeat");
                                last_sync_snapshot = std::time::Instant::now();
                            }
                            let _ = (cur, nthreads, nready);

                            let cur_stats = guard.nvdrv.stats.snapshot();
                            let d = |cur: u64, prev: u64| -> u64 { cur.saturating_sub(prev) };
                            let gpu_progress =
                                d(cur_stats.gpfifo_submits, last_pipeline_stats.gpfifo_submits) > 0
                                    || d(
                                        cur_stats.gpfifo_entries,
                                        last_pipeline_stats.gpfifo_entries,
                                    ) > 0
                                    || d(
                                        cur_stats.methods_dispatched,
                                        last_pipeline_stats.methods_dispatched,
                                    ) > 0
                                    || d(
                                        cur_stats.maxwell3d_draws,
                                        last_pipeline_stats.maxwell3d_draws,
                                    ) > 0
                                    || d(
                                        cur_stats.frames_submitted,
                                        last_pipeline_stats.frames_submitted,
                                    ) > 0;
                            log::info!(
                            "[gpu] gpfifo_submits={} (+{}/s) entries={} (+{}/s) methods={} (+{}/s) | mw3d draws={} (+{}) clears={} (+{}) | fermi2d blits={} (+{}) | mwdma blits={} (+{}) | fences={} (+{})",
                            cur_stats.gpfifo_submits, ((d(cur_stats.gpfifo_submits, last_pipeline_stats.gpfifo_submits) as f64) / secs) as u64,
                            cur_stats.gpfifo_entries, ((d(cur_stats.gpfifo_entries, last_pipeline_stats.gpfifo_entries) as f64) / secs) as u64,
                            cur_stats.methods_dispatched, ((d(cur_stats.methods_dispatched, last_pipeline_stats.methods_dispatched) as f64) / secs) as u64,
                            cur_stats.maxwell3d_draws, d(cur_stats.maxwell3d_draws, last_pipeline_stats.maxwell3d_draws),
                            cur_stats.maxwell3d_clears, d(cur_stats.maxwell3d_clears, last_pipeline_stats.maxwell3d_clears),
                            cur_stats.fermi_2d_blits, d(cur_stats.fermi_2d_blits, last_pipeline_stats.fermi_2d_blits),
                            cur_stats.maxwell_dma_blits, d(cur_stats.maxwell_dma_blits, last_pipeline_stats.maxwell_dma_blits),
                            cur_stats.fence_releases, d(cur_stats.fence_releases, last_pipeline_stats.fence_releases),
                        );
                            let frame_q_depth = guard.nvdrv.frame_queue_depth();
                            log::info!(
                            "[fb]  rb={} deq={} q={} submit={} drain={} vsync={} | nvmaps create={} alloc={} | frame_q_depth={}",
                            cur_stats.request_buffer_calls, cur_stats.dequeue_buffer_calls, cur_stats.queue_buffer_calls,
                            cur_stats.frames_submitted, cur_stats.frames_drained, cur_stats.vsync_signals,
                            cur_stats.nvmap_creates, cur_stats.nvmap_allocs, frame_q_depth,
                        );

                            let bq_info = guard.nvdrv.with_bufferqueue(256, |bq| {
                                (
                                    bq.slots.len(),
                                    bq.free.len(),
                                    bq.dequeued.len(),
                                    bq.queued.len(),
                                    bq.last_queued,
                                    bq.connected_api,
                                )
                            });
                            log::info!(
                            "[bq]  binder=256 slots={} free={} dequeued={} queued={} last_queued={:?} connected_api={}",
                            bq_info.0, bq_info.1, bq_info.2, bq_info.3, bq_info.4, bq_info.5,
                        );

                            let top_methods = {
                                let mut mw = guard.nvdrv.gpu.maxwell3d.lock();
                                mw.take_top_methods(10)
                            };
                            if !top_methods.is_empty() {
                                let pretty: Vec<String> = top_methods
                                    .iter()
                                    .map(|(m, n)| format!("{:#x}={}", m, n))
                                    .collect();
                                log::info!("[mw3d-methods] {}", pretty.join(" "));
                            }
                            if gpu_progress {
                                last_render_progress = std::time::Instant::now();
                                seen_render_progress = true;
                            } else if seen_render_progress
                                && svc_count > last_heartbeat_svc
                                && last_render_progress.elapsed()
                                    >= std::time::Duration::from_secs(2)
                                && last_sync_snapshot.elapsed() >= std::time::Duration::from_secs(2)
                            {
                                log::warn!(
                                    "[render-idle-live] no render progress for {:.2}s while svc_count advanced by {}",
                                    last_render_progress.elapsed().as_secs_f64(),
                                    svc_count.saturating_sub(last_heartbeat_svc)
                                );
                                guard.log_thread_snapshot("render-idle-live");
                                last_sync_snapshot = std::time::Instant::now();
                            }
                            if halts.saturating_sub(last_halts) >= 3 {
                                log::warn!(
                                    "[spin-detected] watchdog halts +{} since last heartbeat (no-SVC spinner) pc={:#x} x0={:#x} x1={:#x}",
                                    halts.saturating_sub(last_halts), cur_pc, x0, x1
                                );
                                let mut nm = [0u8; 64];
                                if x1 != 0 && guard.address_space.read(x1, &mut nm).is_ok() {
                                    let end = nm.iter().position(|&b| b == 0).unwrap_or(nm.len());
                                    log::warn!(
                                        "[spin-detected] *x1 ascii=\"{}\"",
                                        String::from_utf8_lossy(&nm[..end])
                                    );
                                }
                                if x0 != 0 && guard.address_space.read(x0, &mut nm).is_ok() {
                                    let end = nm.iter().position(|&b| b == 0).unwrap_or(nm.len());
                                    log::warn!(
                                        "[spin-detected] *x0 ascii=\"{}\"",
                                        String::from_utf8_lossy(&nm[..end])
                                    );
                                }
                                let code_start = cur_pc.saturating_sub(0x40);
                                let mut code = [0u8; 0x100];
                                if guard.address_space.read(code_start, &mut code).is_ok() {
                                    let words: Vec<String> = code
                                        .chunks_exact(4)
                                        .map(|c| {
                                            format!(
                                                "{:08x}",
                                                u32::from_le_bytes([c[0], c[1], c[2], c[3]])
                                            )
                                        })
                                        .collect();
                                    log::warn!(
                                        "[spin-detected] code@{:#x} pc={:#x} words=[{}]",
                                        code_start,
                                        cur_pc,
                                        words.join(",")
                                    );
                                }
                                guard.log_thread_snapshot("spin-detected");
                            }
                            last_halts = halts;
                            last_pipeline_stats = cur_stats;

                            nexium_core::kernel::profile::dump_heartbeat_with_kernel(&guard);

                            last_heartbeat = std::time::Instant::now();
                            last_heartbeat_svc = svc_count;
                            last_heartbeat_cycles = cycle_count;

                            let mut st = stats_clone.lock();
                            st.svc_count = svc_count as u64;
                            st.cycle_count = cycle_count;
                        }
                    }

                    if guard.ensure_thread_loaded().is_none() {
                        guard.tick_audio_renderers();
                        guard.threads.wake_due_sleepers(std::time::Instant::now());
                        if guard.ensure_thread_loaded().is_some() {
                            continue;
                        }
                        let wake_opt = guard.threads.earliest_wake();
                        drop(guard);
                        match wake_opt {
                            Some(wake) => {
                                let now = std::time::Instant::now();
                                if wake > now {
                                    let remaining = wake - now;
                                    if remaining > std::time::Duration::from_micros(1500) {
                                        let coarse = (remaining
                                            - std::time::Duration::from_millis(1))
                                        .min(std::time::Duration::from_millis(2));
                                        std::thread::sleep(coarse);
                                    } else {
                                        while std::time::Instant::now() < wake {
                                            std::hint::spin_loop();
                                        }
                                    }
                                }
                            }
                            None => std::thread::sleep(std::time::Duration::from_millis(2)),
                        }
                        continue;
                    }

                    guard.threads.wake_due_sleepers(std::time::Instant::now());

                    if let Some(cpu) = cpu_mut() {
                        let pc_before = cpu.get_pc();
                        let cpu_slice: u64 = std::env::var("NEXIUM_CPU_SLICE")
                            .ok()
                            .and_then(|v| v.parse::<u64>().ok())
                            .unwrap_or(1_000_000);
                        drop(guard);
                        let gen = boot_ctx.address_space.generation();
                        if gen != last_map_gen0 {
                            if let Err(error) = sync_host_region_changes(
                                cpu,
                                &boot_ctx.address_space,
                                &mut last_map_gen0,
                                0,
                            ) {
                                log::error!("{}", error);
                            }
                        }
                        let event = cpu.run(cpu_slice);
                        let pc_after = cpu.get_pc();
                        let mut guard = boot_ctx.kernel.lock();
                        guard.threads.save_current_ctx(cpu);
                        cycle_count += cpu_slice;
                        guard.cycle_count += cpu_slice;
                        if let Some(watch) = cpu_poll_watch.as_mut() {
                            if !watch.check(&guard, cpu, event, cycle_count, svc_count) {
                                cpu_poll_watch = None;
                            }
                        }
                        if let Some(trace) = pc_trace.as_mut() {
                            trace.check(0, &guard, cpu, event, cycle_count, svc_count);
                        }

                        if pc_after < 0x10000 {
                            let cur = guard.threads.current_handle();
                            let lr = cpu.get_register(30);
                            let sp = cpu.get_sp();
                            let mut probe_target = None;
                            let mut regs = [0u64; 32];
                            for i in 0..31 {
                                regs[i] = cpu.get_register(i as u32);
                            }
                            log::error!("[null-pc] PC entered null page ({:#x}) — likely null function pointer / corrupted vtable", pc_after);
                            log::error!("[null-pc] handle={:?} lr={:#x} sp={:#x}", cur, lr, sp);
                            for chunk in 0..4u32 {
                                let b = (chunk * 8) as usize;
                                log::error!(
                                "[null-pc] x{:>2}={:#018x} x{:>2}={:#018x} x{:>2}={:#018x} x{:>2}={:#018x} x{:>2}={:#018x} x{:>2}={:#018x} x{:>2}={:#018x} x{:>2}={:#018x}",
                                b, regs[b], b+1, regs[b+1], b+2, regs[b+2], b+3, regs[b+3],
                                b+4, regs[b+4], b+5, regs[b+5], b+6, regs[b+6], b+7, regs[b+7],
                            );
                            }
                            if lr >= 0x20 {
                                let mut instrs = [0u8; 48];
                                if guard
                                    .address_space
                                    .read(lr.wrapping_sub(0x20), &mut instrs)
                                    .is_ok()
                                {
                                    for off in 0..12usize {
                                        let bytes = &instrs[off * 4..off * 4 + 4];
                                        let insn = u32::from_le_bytes([
                                            bytes[0], bytes[1], bytes[2], bytes[3],
                                        ]);
                                        let addr = lr.wrapping_sub(0x20) + (off as u64) * 4;
                                        let mark = if addr + 4 == lr {
                                            " <- BL/BLR site (target was null)"
                                        } else if addr == lr {
                                            " <- LR (return target)"
                                        } else {
                                            ""
                                        };
                                        log::error!(
                                            "[null-pc] {:#x}: {:08x} ({}){}",
                                            addr,
                                            insn,
                                            decode_a64_brief(insn),
                                            mark
                                        );
                                    }
                                }
                            }
                            {
                                let mut fp = regs[29];
                                let mut frames: Vec<String> = Vec::new();
                                for _ in 0..16 {
                                    if fp < 0x10000 {
                                        break;
                                    }
                                    let mut b = [0u8; 16];
                                    if guard.address_space.read(fp, &mut b).is_err() {
                                        break;
                                    }
                                    let next = u64::from_le_bytes(b[0..8].try_into().unwrap());
                                    let ret = u64::from_le_bytes(b[8..16].try_into().unwrap());
                                    if ret == 0 {
                                        break;
                                    }
                                    frames.push(format!("+{:#x}", ret.wrapping_sub(0x8000000)));
                                    if next <= fp {
                                        break;
                                    }
                                    fp = next;
                                }
                                log::error!("[null-pc] callstack(+base): {}", frames.join(" <- "));
                            }
                            {
                                let x19 = regs[19];
                                let mut slot = [0u8; 8];
                                let readable = x19 >= 0x1000
                                    && guard.address_space.read(x19, &mut slot).is_ok();
                                let slotval = if readable {
                                    u64::from_le_bytes(slot)
                                } else {
                                    0
                                };
                                log::error!(
                                    "[null-pc] x19(slot)={:#x} readable={} *x19={:#x} cores.current={:?}",
                                    x19, readable, slotval, guard.threads.current
                                );
                                let mut frame = [0u8; 48];
                                if guard.address_space.read(sp, &mut frame).is_ok() {
                                    let caller_x21 =
                                        u64::from_le_bytes(frame[8..16].try_into().unwrap());
                                    log::error!(
                                        "[null-pc] frame@sp={:02x?} caller_x21={:#x}",
                                        &frame,
                                        caller_x21
                                    );
                                    if caller_x21 >= 0x20 {
                                        probe_target = Some(caller_x21.saturating_add(8));
                                        let mut record = [0u8; 96];
                                        if guard
                                            .address_space
                                            .read(caller_x21 - 0x20, &mut record)
                                            .is_ok()
                                        {
                                            log::error!(
                                                "[null-pc] control@x21-0x20={:02x?}",
                                                &record
                                            );
                                        }
                                    }
                                }
                                let gpu_cpu = guard.nvdrv.gpu.mappings.lock().cpu_address_for(x19);
                                if let Some(gpu_cpu) = gpu_cpu {
                                    let mut backing = [0u8; 96];
                                    let base = gpu_cpu.saturating_sub(0x20);
                                    let readable =
                                        guard.address_space.read(base, &mut backing).is_ok();
                                    log::error!(
                                        "[null-pc] x19_gpu_cpu={:#x} readable={} backing[-0x20..+0x40]={:02x?}",
                                        gpu_cpu,
                                        readable,
                                        &backing
                                    );
                                }
                            }
                            let x20 = regs[20];
                            if x20 >= 0x10000 {
                                let mut peek = [0u8; 64];
                                if guard.address_space.read(x20, &mut peek).is_ok() {
                                    log::error!("[null-pc] *x20[0..64] = {:02x?}", &peek);
                                }
                            }
                            let x22 = regs[22];
                            if x22 >= 0x10000 {
                                let mut peek = [0u8; 64];
                                if guard.address_space.read(x22, &mut peek).is_ok() {
                                    log::error!("[null-pc] *x22[0..64] = {:02x?}", &peek);
                                }
                            }
                            if std::env::var("NEXIUM_NULL_PROBE_CONTINUE").is_ok() {
                                if let Some(target) = probe_target {
                                    if nexium_memory::fastmem::watch_mark(target, 8) {
                                        log::warn!(
                                            "[null-probe] armed payload va={:#x}; skipping null call at lr={:#x}",
                                            target,
                                            lr
                                        );
                                    }
                                }
                                cpu.set_pc(lr);
                                guard.threads.save_current_ctx(cpu);
                                continue;
                            }
                            log::error!("[null-pc] halting emulation for diagnosis");
                            break;
                        }

                        let no_svc_progress = matches!(event, nexium_core::cpu::CpuEvent::Running);
                        if matches!(event, nexium_core::cpu::CpuEvent::Svc(_)) {
                            no_svc_in_spin = 0;
                        }
                        const SPIN_PREEMPT_THRESHOLD: u32 = 2;
                        if no_svc_progress {
                            no_svc_in_spin = no_svc_in_spin.saturating_add(1);
                        }
                        if no_svc_progress && no_svc_in_spin >= SPIN_PREEMPT_THRESHOLD {
                            let hid = nexium_core::hid_state::get_hid_state();
                            let mut h = hid.lock();
                            if h.shmem_va.is_some() {
                                let cur = h.input.clone();
                                h.tick(cur);
                            }
                            drop(h);
                            let n_ready = guard.threads.ready.len();
                            let n_threads = guard.threads.threads.len();
                            if n_ready > 0 {
                                let from = guard.threads.current_handle();
                                guard.threads.yield_with_state(
                                    cpu,
                                    nexium_core::kernel::threads::ThreadState::Ready,
                                );
                                no_svc_in_spin = 0;
                                preempt_count += 1;
                                if preempt_count % 512 == 1 {
                                    log::info!("[preempt] #{} slice-expired pc={:#x}, yielded handle={:?}, ready_q={} total={}", preempt_count, pc_after, from, n_ready, n_threads);
                                }
                            } else {
                                stuck_log_counter += 1;
                                if stuck_log_counter % 50 == 1 {
                                    let cur = guard.threads.current_handle();
                                    let lr = cpu.get_register(30);
                                    let x0 = cpu.get_register(0);
                                    let x1 = cpu.get_register(1);
                                    let x8 = cpu.get_register(8);
                                    let x16 = cpu.get_register(16);
                                    let x19 = cpu.get_register(19);
                                    let x20 = cpu.get_register(20);
                                    let mut instr = [0u8; 16];
                                    let _ = guard.address_space.read(pc_after, &mut instr);
                                    let i0 = u32::from_le_bytes([
                                        instr[0], instr[1], instr[2], instr[3],
                                    ]);
                                    let i1 = u32::from_le_bytes([
                                        instr[4], instr[5], instr[6], instr[7],
                                    ]);
                                    let i2 = u32::from_le_bytes([
                                        instr[8], instr[9], instr[10], instr[11],
                                    ]);
                                    let i3 = u32::from_le_bytes([
                                        instr[12], instr[13], instr[14], instr[15],
                                    ]);
                                    let n_threads_total = guard.threads.threads.len();
                                    let states: Vec<String> = guard
                                        .threads
                                        .threads
                                        .iter()
                                        .map(|(h, t)| {
                                            format!(
                                                "{:#x}={:?}",
                                                h,
                                                std::mem::discriminant(&t.state)
                                            )
                                        })
                                        .collect();
                                    log::warn!(
                                    "[stuck-cpu #{}] handle={:?} pc={:#x} lr={:#x} x0={:#x} x1={:#x} x8={:#x} x16={:#x} x19={:#x} x20={:#x} insn=[{:#010x} {:#010x} {:#010x} {:#010x}] threads={} states=[{}]",
                                    stuck_log_counter, cur, pc_after, lr, x0, x1, x8, x16, x19, x20, i0, i1, i2, i3, n_threads_total, states.join(",")
                                );
                                }
                            }
                        }

                        if cycle_count % 1_000_000 == 0 {
                            let mut st = stats_clone.lock();
                            st.svc_count = svc_count as u64;
                            st.cycle_count = cycle_count;
                            drop(st);

                            let mut snap = cpu_snapshot_clone.lock();
                            snap.pc = cpu.get_pc();
                            snap.sp = cpu.get_register(31);
                            snap.tpidrro_el0 = cpu.get_tpidrro_el0();
                            for i in 0..31 {
                                snap.x[i] = cpu.get_register(i as u32);
                            }
                            let mut instr_buf = vec![0u8; 64];
                            if guard.address_space.read(snap.pc, &mut instr_buf).is_ok() {
                                snap.instruction_bytes = instr_buf;
                            }
                            let mem_req = *mem_request_clone.lock();
                            if mem_req != 0 {
                                snap.mem_request_address = mem_req;
                                let mut mem_buf = vec![0u8; 256];
                                if guard.address_space.read(mem_req, &mut mem_buf).is_ok() {
                                    snap.mem_address = mem_req;
                                    snap.mem_data = mem_buf;
                                }
                            }
                        }

                        if guard.cycle_count >= guard.next_vsync_cycle && !guard.display_ready {
                            guard.display_ready = true;
                            log::info!("Simulating display ready");
                        }

                        if pc_check_count < 5 {
                            log::info!(
                                "CPU exec: PC {:#x} → {:#x} (event: {:?})",
                                pc_before,
                                pc_after,
                                event
                            );
                            pc_check_count += 1;
                        }

                        if pc_before == pc_after
                            && matches!(event, nexium_core::cpu::CpuEvent::Running)
                        {
                            if stuck_pc == Some(pc_before) {
                                stuck_count += 1;
                                if stuck_count == 100 {
                                    log::error!(
                                        "STUCK: CPU looping at PC {:#x} for 10M+ cycles, no SVCs",
                                        pc_before
                                    );
                                }
                            } else {
                                stuck_pc = Some(pc_before);
                                stuck_count = 1;
                            }
                        } else {
                            stuck_pc = None;
                            stuck_count = 0;
                        }

                        match event {
                            nexium_core::cpu::CpuEvent::Running => {
                                if cycle_count % 10_000_000 == 0 {
                                    log::trace!("CPU running... {} cycles executed", cycle_count);
                                }
                            }
                            nexium_core::cpu::CpuEvent::Svc(imm) => {
                                svc_count += 1;
                                last_svc_cycle = cycle_count;
                                last_svc_ms.store(now_millis(), Ordering::Relaxed);
                                log::trace!("SVC {:#04x} (count: {})", imm, svc_count);
                            }
                            nexium_core::cpu::CpuEvent::Stalled => {
                                log::info!("CPU stalled at {:#x}", cpu.get_pc());
                                break;
                            }
                            nexium_core::cpu::CpuEvent::Interrupted => {
                                log::info!("CPU interrupted");
                                break;
                            }
                            nexium_core::cpu::CpuEvent::Exception(code) => {
                                let pc = cpu.get_pc();
                                let lr = cpu.get_register(30);
                                let sp = cpu.get_sp();
                                let mut instr = [0u8; 4];
                                let instr_ok = guard.address_space.read(pc, &mut instr).is_ok();
                                let instr_word = u32::from_le_bytes(instr);
                                log::error!("=== CPU EXCEPTION {:#x} ===", code);
                                log::error!("  PC={:#x}  LR={:#x}  SP={:#x}", pc, lr, sp);
                                if instr_ok {
                                    log::error!(
                                        "  instr@PC = {:08x}  ({})",
                                        instr_word,
                                        decode_a64_brief(instr_word)
                                    );
                                } else {
                                    log::error!("  instr@PC = <unreadable>");
                                }
                                for chunk in 0..4u32 {
                                    let b = chunk * 8;
                                    log::error!(
                                    "  X{:>2}={:#018x} X{:>2}={:#018x} X{:>2}={:#018x} X{:>2}={:#018x} X{:>2}={:#018x} X{:>2}={:#018x} X{:>2}={:#018x} X{:>2}={:#018x}",
                                    b, cpu.get_register(b),
                                    b+1, cpu.get_register(b+1),
                                    b+2, cpu.get_register(b+2),
                                    b+3, cpu.get_register(b+3),
                                    b+4, cpu.get_register(b+4),
                                    b+5, cpu.get_register(b+5),
                                    b+6, cpu.get_register(b+6),
                                    b+7, cpu.get_register(b+7),
                                );
                                }
                                if let Some(f) = cpu.take_fault() {
                                    log::error!(
                                    "  fault addr={:#x} size={} write={} value={:#x} (fault.pc={:#x})",
                                    f.addr, f.size, f.is_write, f.value, f.pc
                                );
                                }
                                let mut stk = [0u8; 64];
                                if guard.address_space.read(sp, &mut stk).is_ok() {
                                    log::error!("  stack@SP[0..64] = {:02x?}", &stk[..]);
                                }
                                let mut prev = [0u8; 16];
                                if pc >= 16 && guard.address_space.read(pc - 16, &mut prev).is_ok()
                                {
                                    log::error!("  code@PC-16..PC = {:02x?}", &prev[..]);
                                }
                                break;
                            }
                        }

                        if cycle_count > max_cycles {
                            log::warn!("Max cycles exceeded");
                            break;
                        }

                        if svc_count > 0 && (cycle_count - last_svc_cycle) > 500_000_000 {
                            if no_svc_exit {
                                log::warn!("Program stuck without SVCs for 500M+ cycles at PC {:#x}. Likely waiting for events/interrupts that aren't implemented. Exiting.", pc_before);
                                break;
                            }
                            if cycle_count - last_no_svc_warn > 2_000_000_000 {
                                last_no_svc_warn = cycle_count;
                                log::warn!(
                                    "[no-svc] {}M cycles without SVCs at PC {:#x} (continuing)",
                                    (cycle_count - last_svc_cycle) / 1_000_000,
                                    pc_before
                                );
                            }
                        }

                        let event_copy = event;
                        let _ = cpu;

                        if let nexium_core::cpu::CpuEvent::Svc(imm) = event_copy {
                            let result = guard.dispatch_svc(imm);
                            if imm != 0x7f {
                                if let Some(cpu) = cpu_mut() {
                                    cpu.set_register(0, result as u64);
                                }
                            }
                            let pace_present = guard.present_pace_until.take();
                            let timeslice = guard
                                .threads
                                .timeslice_expired(std::time::Duration::from_millis(4));
                            let should_yield =
                                guard.yield_after_svc || timeslice || pace_present.is_some();
                            if should_yield {
                                let reason = if pace_present.is_some() {
                                    "present-pace"
                                } else if guard.yield_after_svc {
                                    "flag"
                                } else {
                                    "timeslice"
                                };
                                let from = guard.threads.current_handle();
                                let ready_len = guard.threads.ready.len();
                                guard.yield_after_svc = false;
                                if let Some(cpu) = cpu_ref() {
                                    let state = match pace_present {
                                        Some(wake_at) if wake_at > std::time::Instant::now() => {
                                            nexium_core::kernel::threads::ThreadState::Sleeping {
                                                wake_at,
                                            }
                                        }
                                        _ => nexium_core::kernel::threads::ThreadState::Ready,
                                    };
                                    guard.threads.yield_with_state(cpu, state);
                                }
                                if reason != "present-pace" {
                                    log::trace!(
                                        "[yield] reason={} from={:?} ready_before={}",
                                        reason,
                                        from,
                                        ready_len
                                    );
                                }
                            }
                        }

                        if svc_count % 16 == 0 {
                            let state = nexium_core::hid_state::get_hid_state();
                            let mut hid = state.lock();
                            if hid.shmem_va.is_some() {
                                let cur = hid.input.clone();
                                hid.tick(cur);
                            }
                        }

                        if let Some(f) = guard.nvdrv.drain_latest_frame() {
                            if frame_tx
                                .try_send(Frame {
                                    width: f.width,
                                    height: f.height,
                                    pixels: f.pixels,
                                })
                                .is_ok()
                            {
                                if let Some(ctx) = &repaint_ctx {
                                    ctx.request_repaint();
                                }
                            }
                        }

                        guard.tick_audio_renderers();

                        if svc_count % 256 == 0 {
                            guard.threads.drop_exited();
                        }

                        if guard.process_exited {
                            log::info!("Process exited");
                            break;
                        }
                    } else {
                        return Err("CPU not initialized".to_string());
                    }
                }

                log::info!(
                    "Emulation complete: {} cycles, {} SVCs",
                    cycle_count,
                    svc_count
                );

                aux_core_stop.store(true, Ordering::Relaxed);
                for h in aux_core_handles {
                    let _ = h.join();
                }

                if stop_flag_clone.load(Ordering::Relaxed) {
                    return Ok(());
                }

                if let Some(next_path) = boot_ctx.chained_load_path() {
                    chained_argv = boot_ctx.chained_load_argv();
                    log::info!(
                        "Chain-launch: {} -> {} argv={:?}",
                        cur_nro_path,
                        next_path,
                        chained_argv
                    );
                    drop(_wd_guard);
                    drop(boot_ctx);
                    cur_nro_path = next_path;
                    continue 'launcher;
                }

                let _ = initial_loader_path;
                return Ok(());
            }
        });

        Ok(Self {
            stop_flag,
            pause_flag,
            frame_rx,
            thread_handle: Some(thread_handle),
            cpu_snapshot,
            mem_request,
            stats,
        })
    }

    pub fn request_memory_read(&self, address: u64) {
        *self.mem_request.lock() = address;
    }

    pub fn snapshot(&self) -> CpuSnapshot {
        self.cpu_snapshot.lock().clone()
    }

    pub fn stop(&mut self) {
        self.stop_flag.store(true, Ordering::Relaxed);
        if let Some(handle) = self.thread_handle.take() {
            std::thread::spawn(move || {
                let _ = handle.join();
            });
        }
    }

    pub fn stop_blocking(&mut self) {
        self.stop_flag.store(true, Ordering::Relaxed);
        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }
    }

    pub fn is_running(&self) -> bool {
        !self.stop_flag.load(Ordering::Relaxed)
    }

    pub fn is_paused(&self) -> bool {
        self.pause_flag.load(Ordering::Relaxed)
    }

    pub fn pause(&self) {
        self.pause_flag.store(true, Ordering::Relaxed);
    }

    pub fn resume(&self) {
        self.pause_flag.store(false, Ordering::Relaxed);
    }

    pub fn toggle_pause(&self) -> bool {
        let now = !self.is_paused();
        self.pause_flag.store(now, Ordering::Relaxed);
        now
    }
}

impl Drop for EmulationHandle {
    fn drop(&mut self) {
        self.stop();
    }
}

use nexium_core::boot::{BootConfig, BootContext};
use nexium_core::services::FrameOut;
use parking_lot::Mutex;
use std::path::Path;
use std::sync::{Arc, atomic::{AtomicBool, AtomicU64, Ordering}};
use std::sync::mpsc::{self, Receiver};
use std::thread;

fn now_millis() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

fn decode_a64_brief(insn: u32) -> String {

    let top8 = (insn >> 24) & 0xFF;
    let size = (insn >> 30) & 0b11;
    let rt = insn & 0x1F;
    let rn = (insn >> 5) & 0x1F;
    let opc = (insn >> 22) & 0b11;

    if (insn & 0x3B00_0000) == 0x3900_0000 {
        let imm12 = ((insn >> 10) & 0xFFF) as u64;
        let scale = match size { 0 => 1u64, 1 => 2, 2 => 4, 3 => 8, _ => 1 };
        let off = imm12 * scale;
        let (mnemonic, width) = match (size, opc) {
            (0, 0) => ("STRB", "W"),
            (0, 1) => ("LDRB", "W"),
            (1, 0) => ("STRH", "W"),
            (1, 1) => ("LDRH", "W"),
            (2, 0) => ("STR",  "W"),
            (2, 1) => ("LDR",  "W"),
            (3, 0) => ("STR",  "X"),
            (3, 1) => ("LDR",  "X"),
            _ => ("?LSI", "?"),
        };
        return format!("{} {}{}, [X{}, #{}]", mnemonic, width, rt, rn, off);
    }

    if (insn & 0x3B20_0C00) == 0x3800_0000 {
        let imm9 = ((insn >> 12) & 0x1FF) as i32;
        let imm9 = if imm9 & 0x100 != 0 { imm9 | !0x1FF } else { imm9 };
        let (mnemonic, width) = match (size, opc) {
            (0, 0) => ("STURB", "W"),
            (0, 1) => ("LDURB", "W"),
            (1, 0) => ("STURH", "W"),
            (1, 1) => ("LDURH", "W"),
            (2, 0) => ("STUR",  "W"),
            (2, 1) => ("LDUR",  "W"),
            (3, 0) => ("STUR",  "X"),
            (3, 1) => ("LDUR",  "X"),
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
        return format!("{} {}{}, {}{}, [X{}, #{}]", mnemonic, width, rt, width, rt2, rn, off);
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
        Self { width: f.width, height: f.height, pixels: f.pixels }
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
    pub frame_rx: Receiver<Frame>,
    pub thread_handle: Option<thread::JoinHandle<Result<(), String>>>,
    pub cpu_snapshot: Arc<Mutex<CpuSnapshot>>,
    pub mem_request: Arc<Mutex<u64>>,
    pub stats: Arc<Mutex<EmuStats>>,
}

impl EmulationHandle {
    pub fn new(nro_path: &str, cpu_backend: nexium_cpu::CpuBackendKind) -> Result<Self, String> {
        let nro_path = nro_path.to_string();
        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop_flag_clone = Arc::clone(&stop_flag);
        let (frame_tx, frame_rx) = mpsc::sync_channel::<Frame>(2);
        let cpu_snapshot = Arc::new(Mutex::new(CpuSnapshot::default()));
        let cpu_snapshot_clone = Arc::clone(&cpu_snapshot);
        let mem_request = Arc::new(Mutex::new(0u64));
        let mem_request_clone = Arc::clone(&mem_request);
        let stats = Arc::new(Mutex::new(EmuStats::default()));
        let stats_clone = Arc::clone(&stats);

        let thread_handle = thread::spawn(move || {
            let initial_loader_path = nro_path.clone();
            let initial_loader_filename = Path::new(&initial_loader_path)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("hbmenu.nro")
                .to_string();
            let initial_loader_argv = format!("sdmc:/{}", initial_loader_filename);
            let mut cur_nro_path = nro_path;

            'launcher: loop {
            log::info!("Booting NRO: {}", cur_nro_path);

            if !Path::new(&cur_nro_path).exists() {
                return Err(format!("NRO file not found: {}", cur_nro_path));
            }

            let mut config = BootConfig::new(&cur_nro_path);
            config.loader_path = Some(initial_loader_argv.clone());
            config.cpu_backend = cpu_backend;
            let mut boot_ctx = BootContext::new(config)?;

            let mut cpu = boot_ctx.cpu.take().expect("BootContext CPU not initialized");
            cpu.set_continue_on_null(true);
            log::info!("dynarmic: continue_on_null=ON (will absorb null-zone reads as 0 to keep going)");
            let halt = Some(cpu.halt_handle());
            let _cpu_guard = nexium_kernel::kernel::cpu_local::set_current_cpu(&mut cpu, 0);
            use nexium_kernel::kernel::cpu_local::{cpu_mut, cpu_ref};
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

            log::info!("Starting emulation loop [BUILD: heartbeat-v2-gpu-diag]");
            let max_cycles = u64::MAX;
            let mut cycle_count = 0u64;
            let mut svc_count = 0u32;
            let mut stuck_log_counter: u64 = 0;
            let mut pc_check_count = 0u32;
            let mut stuck_pc: Option<u64> = None;
            let mut stuck_count = 0u32;
            let mut last_svc_cycle = 0u64;
            let mut no_svc_in_spin = 0u32;
            let mut last_heartbeat = std::time::Instant::now();
            let mut last_heartbeat_svc = 0u32;
            let mut last_heartbeat_cycles = 0u64;
            let mut last_pipeline_stats = boot_ctx.kernel.nvdrv.stats.snapshot();

            let mut loop_iter: u64 = 0;
            let mut last_loop_log = std::time::Instant::now();
            loop {
                loop_iter += 1;
                if last_loop_log.elapsed() >= std::time::Duration::from_secs(1) {
                    let cur = boot_ctx.kernel.threads.current_handle();
                    let pc_now = cpu_ref().map(|c| c.get_pc()).unwrap_or(0);
                    let halts = watchdog_halts.load(Ordering::Relaxed);
                    log::warn!(
                        "[loop-tick] iter={} svc={} cyc={} cur={:?} pc={:#x} halts={}",
                        loop_iter, svc_count, cycle_count, cur, pc_now, halts
                    );
                    last_loop_log = std::time::Instant::now();
                }

                if stop_flag_clone.load(Ordering::Relaxed) {
                    log::info!("Stopping emulation");
                    break;
                }

                {
                    let elapsed = last_heartbeat.elapsed();
                    if elapsed >= std::time::Duration::from_secs(1) {
                        let secs = elapsed.as_secs_f64();
                        let svc_rate = (svc_count - last_heartbeat_svc) as f64 / secs;
                        let cycle_rate = (cycle_count - last_heartbeat_cycles) as f64 / secs;
                        let cur = boot_ctx.kernel.threads.current_handle();
                        let nthreads = boot_ctx.kernel.threads.threads.len();
                        let nready = boot_ctx.kernel.threads.ready.len();
                        let halts = watchdog_halts.load(Ordering::Relaxed);
                        let null_skips = cpu_ref().map(|c| c.null_skip_count()).unwrap_or(0);
                        let (cur_pc, lr, sp, x0, x1, x19, x20, x21, x22) = if let Some(c) = cpu_ref() {
                            (c.get_pc(), c.get_register(30), c.get_sp(), c.get_register(0), c.get_register(1), c.get_register(19), c.get_register(20), c.get_register(21), c.get_register(22))
                        } else {
                            (0, 0, 0, 0, 0, 0, 0, 0, 0)
                        };
                        if null_skips > 0 {
                            log::info!("[heartbeat] null_skips_total={}", null_skips);
                        }
                        let mut x20_bytes = [0u8; 32];
                        let x20_read = boot_ctx.kernel.address_space.read(x20, &mut x20_bytes).is_ok();
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
                        let _ = (cur, nthreads, nready);

                        let cur_stats = boot_ctx.kernel.nvdrv.stats.snapshot();
                        let d = |cur: u64, prev: u64| -> u64 { cur.saturating_sub(prev) };
                        log::info!(
                            "[gpu] gpfifo_submits={} (+{}/s) entries={} (+{}/s) methods={} (+{}/s) | mw3d draws={} (+{}) clears={} (+{}) | fermi2d blits={} (+{}) | mwdma blits={} (+{})",
                            cur_stats.gpfifo_submits, ((d(cur_stats.gpfifo_submits, last_pipeline_stats.gpfifo_submits) as f64) / secs) as u64,
                            cur_stats.gpfifo_entries, ((d(cur_stats.gpfifo_entries, last_pipeline_stats.gpfifo_entries) as f64) / secs) as u64,
                            cur_stats.methods_dispatched, ((d(cur_stats.methods_dispatched, last_pipeline_stats.methods_dispatched) as f64) / secs) as u64,
                            cur_stats.maxwell3d_draws, d(cur_stats.maxwell3d_draws, last_pipeline_stats.maxwell3d_draws),
                            cur_stats.maxwell3d_clears, d(cur_stats.maxwell3d_clears, last_pipeline_stats.maxwell3d_clears),
                            cur_stats.fermi_2d_blits, d(cur_stats.fermi_2d_blits, last_pipeline_stats.fermi_2d_blits),
                            cur_stats.maxwell_dma_blits, d(cur_stats.maxwell_dma_blits, last_pipeline_stats.maxwell_dma_blits),
                        );
                        let frame_q_depth = boot_ctx.kernel.nvdrv.frame_queue_depth();
                        log::info!(
                            "[fb]  rb={} deq={} q={} submit={} drain={} vsync={} | nvmaps create={} alloc={} | frame_q_depth={}",
                            cur_stats.request_buffer_calls, cur_stats.dequeue_buffer_calls, cur_stats.queue_buffer_calls,
                            cur_stats.frames_submitted, cur_stats.frames_drained, cur_stats.vsync_signals,
                            cur_stats.nvmap_creates, cur_stats.nvmap_allocs, frame_q_depth,
                        );

                        let bq_info = boot_ctx.kernel.nvdrv.with_bufferqueue(256, |bq| {
                            (bq.slots.len(), bq.free.len(), bq.dequeued.len(), bq.queued.len(),
                             bq.last_queued, bq.connected_api)
                        });
                        log::info!(
                            "[bq]  binder=256 slots={} free={} dequeued={} queued={} last_queued={:?} connected_api={}",
                            bq_info.0, bq_info.1, bq_info.2, bq_info.3, bq_info.4, bq_info.5,
                        );

                        let top_methods = {
                            let mut mw = boot_ctx.kernel.nvdrv.gpu.maxwell3d.lock();
                            mw.take_top_methods(10)
                        };
                        if !top_methods.is_empty() {
                            let pretty: Vec<String> = top_methods.iter()
                                .map(|(m, n)| format!("{:#x}={}", m, n))
                                .collect();
                            log::info!("[mw3d-methods] {}", pretty.join(" "));
                        }
                        last_pipeline_stats = cur_stats;

                        nexium_core::kernel::profile::dump_heartbeat_with_kernel(&boot_ctx.kernel);

                        last_heartbeat = std::time::Instant::now();
                        last_heartbeat_svc = svc_count;
                        last_heartbeat_cycles = cycle_count;

                        let mut st = stats_clone.lock();
                        st.svc_count = svc_count as u64;
                        st.cycle_count = cycle_count;
                    }
                }

                if boot_ctx.kernel.ensure_thread_loaded().is_none() {
                    boot_ctx.kernel.tick_audio_renderers();
                    boot_ctx.kernel.threads.wake_due_sleepers(std::time::Instant::now());
                    if boot_ctx.kernel.ensure_thread_loaded().is_some() {
                        continue;
                    }
                    if let Some(wake) = boot_ctx.kernel.threads.earliest_wake() {
                        let now = std::time::Instant::now();
                        if wake > now {
                            let dur = (wake - now).min(std::time::Duration::from_millis(2));
                            std::thread::sleep(dur);
                        }
                        boot_ctx.kernel.threads.wake_due_sleepers(std::time::Instant::now());
                        continue;
                    } else {
                        std::thread::sleep(std::time::Duration::from_millis(2));
                        continue;
                    }
                }

                boot_ctx.kernel.threads.wake_due_sleepers(std::time::Instant::now());

                if let Some(cpu) = cpu_mut() {
                    let pc_before = cpu.get_pc();
                    let cpu_slice: u64 = std::env::var("NEXIUM_CPU_SLICE")
                        .ok()
                        .and_then(|v| v.parse::<u64>().ok())
                        .unwrap_or(1_000_000);
                    let event = cpu.run(cpu_slice);
                    let pc_after = cpu.get_pc();
                    cycle_count += cpu_slice;
                    boot_ctx.kernel.cycle_count += cpu_slice;

                    if pc_after < 0x10000 {
                        let cur = boot_ctx.kernel.threads.current_handle();
                        let lr = cpu.get_register(30);
                        let sp = cpu.get_sp();
                        let mut regs = [0u64; 31];
                        for i in 0..31 { regs[i] = cpu.get_register(i as u32); }
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
                            if boot_ctx.kernel.address_space.read(lr.wrapping_sub(0x20), &mut instrs).is_ok() {
                                for off in 0..12usize {
                                    let bytes = &instrs[off*4..off*4+4];
                                    let insn = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                                    let addr = lr.wrapping_sub(0x20) + (off as u64) * 4;
                                    let mark = if addr + 4 == lr { " <- BL/BLR site (target was null)" }
                                              else if addr == lr { " <- LR (return target)" }
                                              else { "" };
                                    log::error!("[null-pc] {:#x}: {:08x} ({}){}", addr, insn, decode_a64_brief(insn), mark);
                                }
                            }
                        }
                        let x20 = regs[20];
                        if x20 >= 0x10000 {
                            let mut peek = [0u8; 64];
                            if boot_ctx.kernel.address_space.read(x20, &mut peek).is_ok() {
                                log::error!("[null-pc] *x20[0..64] = {:02x?}", &peek);
                            }
                        }
                        let x22 = regs[22];
                        if x22 >= 0x10000 {
                            let mut peek = [0u8; 64];
                            if boot_ctx.kernel.address_space.read(x22, &mut peek).is_ok() {
                                log::error!("[null-pc] *x22[0..64] = {:02x?}", &peek);
                            }
                        }
                        log::error!("[null-pc] halting emulation for diagnosis");
                        break;
                    }

                    let in_libnx = pc_after >= 0x8000_0000_00 && pc_after < 0x8000_a0_0000;
                    let no_svc_progress = matches!(event, nexium_core::cpu::CpuEvent::Running) && in_libnx;
                    if matches!(event, nexium_core::cpu::CpuEvent::Svc(_)) {
                        no_svc_in_spin = 0;
                    }
                    const SPIN_PREEMPT_THRESHOLD: u32 = 20;
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
                        let n_ready = boot_ctx.kernel.threads.ready.len();
                        let n_threads = boot_ctx.kernel.threads.threads.len();
                        if n_ready > 0 {
                            let from = boot_ctx.kernel.threads.current_handle();
                            boot_ctx.kernel.threads.yield_with_state(
                                cpu,
                                nexium_core::kernel::threads::ThreadState::Ready,
                            );
                            no_svc_in_spin = 0;
                            log::info!("[preempt] halted in libnx pc={:#x}, yielded handle={:?}, ready_q={} total={}", pc_after, from, n_ready, n_threads);
                        } else {
                            stuck_log_counter += 1;
                            if stuck_log_counter % 50 == 1 {
                                let cur = boot_ctx.kernel.threads.current_handle();
                                let lr = cpu.get_register(30);
                                let x0 = cpu.get_register(0);
                                let x1 = cpu.get_register(1);
                                let x8 = cpu.get_register(8);
                                let x16 = cpu.get_register(16);
                                let x19 = cpu.get_register(19);
                                let x20 = cpu.get_register(20);
                                let mut instr = [0u8; 16];
                                let _ = boot_ctx.kernel.address_space.read(pc_after, &mut instr);
                                let i0 = u32::from_le_bytes([instr[0], instr[1], instr[2], instr[3]]);
                                let i1 = u32::from_le_bytes([instr[4], instr[5], instr[6], instr[7]]);
                                let i2 = u32::from_le_bytes([instr[8], instr[9], instr[10], instr[11]]);
                                let i3 = u32::from_le_bytes([instr[12], instr[13], instr[14], instr[15]]);
                                let n_threads_total = boot_ctx.kernel.threads.threads.len();
                                let states: Vec<String> = boot_ctx.kernel.threads.threads.iter()
                                    .map(|(h, t)| format!("{:#x}={:?}", h, std::mem::discriminant(&t.state)))
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
                        if boot_ctx.kernel.address_space.read(snap.pc, &mut instr_buf).is_ok() {
                            snap.instruction_bytes = instr_buf;
                        }
                        let mem_req = *mem_request_clone.lock();
                        if mem_req != 0 {
                            snap.mem_request_address = mem_req;
                            let mut mem_buf = vec![0u8; 256];
                            if boot_ctx.kernel.address_space.read(mem_req, &mut mem_buf).is_ok() {
                                snap.mem_address = mem_req;
                                snap.mem_data = mem_buf;
                            }
                        }
                    }

                    if boot_ctx.kernel.cycle_count >= boot_ctx.kernel.next_vsync_cycle && !boot_ctx.kernel.display_ready {
                        boot_ctx.kernel.display_ready = true;
                        log::info!("Simulating display ready");
                    }

                    if pc_check_count < 5 {
                        log::info!("CPU exec: PC {:#x} → {:#x} (event: {:?})", pc_before, pc_after, event);
                        pc_check_count += 1;
                    }

                    if pc_before == pc_after && matches!(event, nexium_core::cpu::CpuEvent::Running) {
                        if stuck_pc == Some(pc_before) {
                            stuck_count += 1;
                            if stuck_count == 100 {
                                log::error!("STUCK: CPU looping at PC {:#x} for 10M+ cycles, no SVCs", pc_before);
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
                                log::info!("CPU running... {} cycles executed", cycle_count);
                            }
                        }
                        nexium_core::cpu::CpuEvent::Svc(imm) => {
                            svc_count += 1;
                            last_svc_cycle = cycle_count;
                            last_svc_ms.store(now_millis(), Ordering::Relaxed);
                            log::debug!("SVC {:#04x} (count: {})", imm, svc_count);
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
                            let instr_ok = boot_ctx.kernel.address_space.read(pc, &mut instr).is_ok();
                            let instr_word = u32::from_le_bytes(instr);
                            log::error!("=== CPU EXCEPTION {:#x} ===", code);
                            log::error!("  PC={:#x}  LR={:#x}  SP={:#x}", pc, lr, sp);
                            if instr_ok {
                                log::error!("  instr@PC = {:08x}  ({})", instr_word, decode_a64_brief(instr_word));
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
                            if boot_ctx.kernel.address_space.read(sp, &mut stk).is_ok() {
                                log::error!("  stack@SP[0..64] = {:02x?}", &stk[..]);
                            }
                            let mut prev = [0u8; 16];
                            if pc >= 16 && boot_ctx.kernel.address_space.read(pc - 16, &mut prev).is_ok() {
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
                        log::warn!("Program stuck without SVCs for 500M+ cycles at PC {:#x}. Likely waiting for events/interrupts that aren't implemented. Exiting.", pc_before);
                        break;
                    }

                    let event_copy = event;
                    let _ = cpu;

                    if let nexium_core::cpu::CpuEvent::Svc(imm) = event_copy {
                        let result = boot_ctx.kernel.dispatch_svc(imm);
                        if let Some(cpu) = cpu_mut() {
                            cpu.set_register(0, result as u64);
                        }
                        let pace_present = boot_ctx.kernel.present_pace_until.take();
                        let timeslice = boot_ctx.kernel.threads.timeslice_expired(std::time::Duration::from_millis(4));
                        let should_yield =
                            boot_ctx.kernel.yield_after_svc || timeslice || pace_present.is_some();
                        if should_yield {
                            let reason = if pace_present.is_some() {
                                "present-pace"
                            } else if boot_ctx.kernel.yield_after_svc {
                                "flag"
                            } else {
                                "timeslice"
                            };
                            let from = boot_ctx.kernel.threads.current_handle();
                            let ready_len = boot_ctx.kernel.threads.ready.len();
                            boot_ctx.kernel.yield_after_svc = false;
                            if let Some(cpu) = cpu_ref() {
                                let state = match pace_present {
                                    Some(wake_at) if wake_at > std::time::Instant::now() => {
                                        nexium_core::kernel::threads::ThreadState::Sleeping { wake_at }
                                    }
                                    _ => nexium_core::kernel::threads::ThreadState::Ready,
                                };
                                boot_ctx.kernel.threads.yield_with_state(cpu, state);
                            }
                            if reason != "present-pace" {
                                log::info!("[yield] reason={} from={:?} ready_before={}", reason, from, ready_len);
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

                    for f in boot_ctx.kernel.drain_frames() {
                        let _ = frame_tx.try_send(f.into());
                    }

                    boot_ctx.kernel.tick_audio_renderers();

                    if svc_count % 256 == 0 {
                        boot_ctx.kernel.threads.drop_exited();
                    }

                    if boot_ctx.kernel.process_exited {
                        log::info!("Process exited");
                        break;
                    }
                } else {
                    return Err("CPU not initialized".to_string());
                }
            }

            log::info!("Emulation complete: {} cycles, {} SVCs", cycle_count, svc_count);

            if stop_flag_clone.load(Ordering::Relaxed) {
                return Ok(());
            }

            if let Some(next_path) = boot_ctx.chained_load_path() {
                log::info!("Chain-launch: {} -> {}", cur_nro_path, next_path);
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
            let _ = handle.join();
        }
    }

    pub fn is_running(&self) -> bool {
        !self.stop_flag.load(Ordering::Relaxed)
    }
}

impl Drop for EmulationHandle {
    fn drop(&mut self) {
        self.stop();
    }
}

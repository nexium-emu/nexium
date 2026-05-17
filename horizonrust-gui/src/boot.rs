use horizonrust_core::boot::{BootConfig, BootContext};
use horizonrust_core::services::FrameOut;
use parking_lot::Mutex;
use std::path::Path;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use std::sync::mpsc::{self, Receiver};
use std::thread;

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
    pub fn new(nro_path: &str) -> Result<Self, String> {
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
            log::info!("Booting NRO: {}", nro_path);

            if !Path::new(&nro_path).exists() {
                return Err(format!("NRO file not found: {}", nro_path));
            }

            let config = BootConfig::new(&nro_path);
            let mut boot_ctx = BootContext::new(config)?;

            log::info!("Starting emulation loop");
            let max_cycles = u64::MAX;
            let mut cycle_count = 0u64;
            let mut svc_count = 0u32;
            let mut pc_check_count = 0u32;
            let mut stuck_pc: Option<u64> = None;
            let mut stuck_count = 0u32;
            let mut last_svc_cycle = 0u64;

            loop {
                if stop_flag_clone.load(Ordering::Relaxed) {
                    log::info!("Stopping emulation");
                    break;
                }

                if let Some(cpu) = &mut boot_ctx.kernel.cpu {
                    let pc_before = cpu.get_pc();
                    let event = cpu.run(100_000);
                    let pc_after = cpu.get_pc();
                    cycle_count += 100_000;
                    boot_ctx.kernel.cycle_count += 100_000;

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

                    if pc_before == pc_after && matches!(event, horizonrust_core::cpu::CpuEvent::Running) {
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
                        horizonrust_core::cpu::CpuEvent::Running => {
                            if cycle_count % 10_000_000 == 0 {
                                log::info!("CPU running... {} cycles executed", cycle_count);
                            }
                        }
                        horizonrust_core::cpu::CpuEvent::Svc(imm) => {
                            svc_count += 1;
                            last_svc_cycle = cycle_count;
                            log::debug!("SVC {:#04x} (count: {})", imm, svc_count);
                        }
                        horizonrust_core::cpu::CpuEvent::Stalled => {
                            log::info!("CPU stalled at {:#x}", cpu.get_pc());
                            break;
                        }
                        horizonrust_core::cpu::CpuEvent::Interrupted => {
                            log::info!("CPU interrupted");
                            break;
                        }
                        horizonrust_core::cpu::CpuEvent::Exception(code) => {
                            log::error!("CPU exception {:#x}", code);
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

                    if let horizonrust_core::cpu::CpuEvent::Svc(imm) = event_copy {
                        let result = boot_ctx.kernel.dispatch_svc(imm);
                        if let Some(cpu) = &mut boot_ctx.kernel.cpu {
                            cpu.set_register(0, result as u64);
                        }
                    }

                    if svc_count % 16 == 0 {
                        let state = horizonrust_core::hid_state::get_hid_state();
                        let mut hid = state.lock();
                        if let Some(va) = hid.shmem_va {
                            let cur = hid.input.clone();
                            let log_input = cur.buttons != 0;
                            hid.update_input(cur);
                            let sampling = hid.sampling_number;
                            let buf = hid.build_initial_shmem();
                            let _ = boot_ctx.kernel.address_space.write(va, &buf);
                            if log_input {
                                let entry0_lifo_va = va + 0x9A00 + 0x28;
                                let entry8_handheld_lifo_va = va + 0x9A00 + 8 * 0x5000 + 0x378;
                                let mut header = [0u8; 32];
                                let _ = boot_ctx.kernel.address_space.read(entry0_lifo_va, &mut header);
                                let count0 = u64::from_le_bytes(header[24..32].try_into().unwrap_or([0;8]));
                                let buttons0_va = entry0_lifo_va + 0x20 + 8 + 8;
                                let mut cpu_read = [0u8; 8];
                                if let Some(cpu) = &boot_ctx.kernel.cpu {
                                    let _ = cpu.read_bytes(buttons0_va, &mut cpu_read);
                                }
                                let cpu_buttons0 = u64::from_le_bytes(cpu_read);
                                let buttons8_va = entry8_handheld_lifo_va + 0x20 + 8 + 8;
                                if let Some(cpu) = &boot_ctx.kernel.cpu {
                                    let _ = cpu.read_bytes(buttons8_va, &mut cpu_read);
                                }
                                let cpu_buttons8 = u64::from_le_bytes(cpu_read);
                                let style0_va = va + 0x9A00;
                                let mut style_buf = [0u8; 4];
                                if let Some(cpu) = &boot_ctx.kernel.cpu {
                                    let _ = cpu.read_bytes(style0_va, &mut style_buf);
                                }
                                let style0 = u32::from_le_bytes(style_buf);
                                log::info!("HID push: btn={:#x} sampling={} | e0 style={:#x} count={} cpu_buttons={:#x} | e8 cpu_buttons={:#x}",
                                    cur.buttons, sampling, style0, count0, cpu_buttons0, cpu_buttons8);
                            }
                        }
                    }

                    for f in boot_ctx.kernel.drain_frames() {
                        let _ = frame_tx.try_send(f.into());
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
            Ok(())
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

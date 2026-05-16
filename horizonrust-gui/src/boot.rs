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

pub struct EmulationHandle {
    pub stop_flag: Arc<AtomicBool>,
    pub frame_rx: Receiver<Frame>,
    pub thread_handle: Option<thread::JoinHandle<Result<(), String>>>,
    pub cpu_snapshot: Arc<Mutex<CpuSnapshot>>,
    pub mem_request: Arc<Mutex<u64>>,
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

        let thread_handle = thread::spawn(move || {
            log::info!("Booting NRO: {}", nro_path);

            if !Path::new(&nro_path).exists() {
                return Err(format!("NRO file not found: {}", nro_path));
            }

            let config = BootConfig::new(&nro_path);
            let mut boot_ctx = BootContext::new(config)?;

            log::info!("Starting emulation loop");
            let max_cycles = 1_000_000_000u64;
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

                    if boot_ctx.kernel.cycle_count >= boot_ctx.kernel.next_vsync_cycle {
                        boot_ctx.kernel.next_vsync_cycle += 16_666_667;
                        if !boot_ctx.kernel.display_ready {
                            boot_ctx.kernel.display_ready = true;
                            log::info!("Simulating display ready");
                        }
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
                            log::info!("SVC {:#04x} (count: {})", imm, svc_count);

                            let _result = boot_ctx.kernel.dispatch_svc(imm);

                            for f in boot_ctx.kernel.drain_frames() {
                                let _ = frame_tx.try_send(f.into());
                            }

                            if boot_ctx.kernel.process_exited {
                                log::info!("Process exited via svcBreak");
                                break;
                            }
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
